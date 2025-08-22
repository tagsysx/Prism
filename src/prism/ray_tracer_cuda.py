"""
CUDA-Accelerated Discrete Electromagnetic Ray Tracing System for Prism

This module implements a high-performance CUDA version of the discrete electromagnetic ray tracing system
as described in the design document, with support for MLP-based direction sampling and
efficient RF signal strength computation.

IMPORTANT NOTE: This ray tracer does NOT select subcarriers internally. All subcarrier
selection must be provided by the calling code (typically PrismTrainingInterface) to
ensure consistency across the training pipeline and proper loss computation.

The ray tracer expects:
- selected_subcarriers: Dictionary or tensor specifying which subcarriers to process
- subcarrier_indices: Explicit indices of subcarriers to trace
- No internal subcarrier selection logic

This design ensures that the training interface has full control over which subcarriers
are used for loss computation, preventing any mismatch between ray tracing and loss calculation.
"""

import torch
import logging
import math
import time
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# CUDA kernel for parallel ray tracing
CUDA_KERNEL = """
extern "C" __global__ void parallel_ray_tracing(
    const float* base_station_pos,
    const float* direction_vectors,
    const float* ue_positions,
    const int* selected_subcarriers,
    const float* antenna_embeddings,
    float* signal_strengths,
    const int num_directions,
    const int num_ue,
    const int num_subcarriers,
    const float max_ray_length,
    const float scene_size,
    const int uniform_samples,
    const int resampled_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_directions * num_ue * num_subcarriers) return;
    
    // Calculate indices
    int direction_idx = idx / (num_ue * num_subcarriers);
    int ue_idx = (idx % (num_ue * num_subcarriers)) / num_subcarriers;
    int subcarrier_idx = idx % num_subcarriers;
    
    // Get positions and vectors
    float3 bs_pos = make_float3(
        base_station_pos[0], base_station_pos[1], base_station_pos[2]
    );
    float3 direction = make_float3(
        direction_vectors[direction_idx * 3],
        direction_vectors[direction_idx * 3 + 1],
        direction_vectors[direction_idx * 3 + 2]
    );
    float3 ue_pos = make_float3(
        ue_positions[ue_idx * 3],
        ue_positions[ue_idx * 3 + 1],
        ue_positions[ue_idx * 3 + 2]
    );
    
    // Calculate ray length to UE
    float3 ray_to_ue = make_float3(
        ue_pos.x - bs_pos.x,
        ue_pos.y - bs_pos.y,
        ue_pos.z - bs_pos.z
    );
    
    float ray_length = fminf(
        fmaxf(ray_to_ue.x * direction.x + ray_to_ue.y * direction.y + ray_to_ue.z * direction.z, 0.0f),
        max_ray_length
    );
    
    // Uniform sampling along ray
    float signal_strength = 0.0f;
    float step_size = ray_length / uniform_samples;
    
    for (int i = 0; i < uniform_samples; i++) {
        float t = i * step_size;
        float3 sample_pos = make_float3(
            bs_pos.x + direction.x * t,
            bs_pos.y + direction.y * t,
            bs_pos.z + direction.z * t
        );
        
        // Check if position is within scene bounds
        if (fabsf(sample_pos.x) <= scene_size * 0.5f &&
            fabsf(sample_pos.y) <= scene_size * 0.5f &&
            fabsf(sample_pos.z) <= scene_size * 0.5f) {
            
            // Calculate distance-based attenuation
            float distance_to_ue = sqrtf(
                powf(sample_pos.x - ue_pos.x, 2) +
                powf(sample_pos.y - ue_pos.y, 2) +
                powf(sample_pos.z - ue_pos.z, 2)
            );
            
            // Simple attenuation model
            float attenuation = expf(-distance_to_ue / 50.0f);
            
            // Apply antenna embedding influence
            float antenna_factor = 0.0f;
            for (int j = 0; j < 64; j++) {
                antenna_factor += antenna_embeddings[subcarrier_idx * 64 + j] * 
                                antenna_embeddings[subcarrier_idx * 64 + j];
            }
            antenna_factor = sqrtf(antenna_factor) / 8.0f; // Normalize
            
            // Frequency-dependent effects
            float frequency_factor = 1.0f / (1.0f + 0.1f * subcarrier_idx);
            
            // Accumulate signal contribution
            signal_strength += attenuation * antenna_factor * frequency_factor * step_size;
        }
    }
    
    // Store result
    signal_strengths[idx] = signal_strength;
}
"""

class CUDARayTracer:
    """CUDA-accelerated ray tracer with automatic device detection."""
    
    def __init__(self, 
                 azimuth_divisions: int = 36,
                 elevation_divisions: int = 18,
                 max_ray_length: float = 100.0,
                 scene_size: float = 200.0,
                 uniform_samples: int = 128,
                 resampled_points: int = 64):
        """
        Initialize CUDA ray tracer.
        
        Args:
            azimuth_divisions: Number of azimuth divisions
            elevation_divisions: Number of elevation divisions
            max_ray_length: Maximum ray length in meters
            scene_size: Scene size in meters
            uniform_samples: Number of uniform samples per ray
            resampled_points: Number of resampled points per ray
        """
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.max_ray_length = max_ray_length
        self.scene_size = scene_size
        self.uniform_samples = uniform_samples
        self.resampled_points = resampled_points
        
        # Device detection
        self.device, self.use_cuda = self._detect_device()
        self._setup_cuda()
        
        # Calculate total directions
        self.total_directions = azimuth_divisions * elevation_divisions
        
        logger.info(f"CUDA Ray Tracer initialized on device: {self.device}")
        if self.use_cuda:
            logger.info("✓ CUDA acceleration enabled - significant performance improvement expected")
        else:
            logger.info("⚠ CUDA not available - using CPU implementation")
    
    def _detect_device(self) -> Tuple[str, bool]:
        """Detect available device and CUDA support."""
        if torch.cuda.is_available():
            device = 'cuda'
            use_cuda = True
            logger.info(f"CUDA detected: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = 'cpu'
            use_cuda = False
            logger.info("CUDA not available, using CPU")
        
        return device, use_cuda
    
    def _setup_cuda(self):
        """Setup CUDA kernel if available."""
        if not self.use_cuda:
            return
        
        try:
            # Compile CUDA kernel
            from torch.utils.cpp_extension import load_inline
            
            self.cuda_module = load_inline(
                name='ray_tracing_kernel',
                cuda_sources=[CUDA_KERNEL],
                cpp_sources=[],  # Add empty cpp_sources list
                extra_cuda_cflags=['-O3', '--use_fast_math'],
                verbose=False
            )
            logger.info("✓ CUDA kernel compiled successfully")
            
        except Exception as e:
            logger.warning(f"CUDA kernel compilation failed: {e}")
            logger.warning("Falling back to PyTorch GPU operations")
            self.use_cuda = False
            self.device = 'cuda'  # Still use GPU but with PyTorch ops
    
    def generate_direction_vectors(self) -> torch.Tensor:
        """Generate unit direction vectors for all directions."""
        directions = []
        
        for i in range(self.azimuth_divisions):
            for j in range(self.elevation_divisions):
                phi = i * (2 * math.pi / self.azimuth_divisions)
                theta = j * (math.pi / self.elevation_divisions)
                
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)
                
                directions.append([x, y, z])
        
        return torch.tensor(directions, dtype=torch.float32, device=self.device)
    
    def trace_rays_cuda_kernel(self,
                              base_station_pos: torch.Tensor,
                              direction_vectors: torch.Tensor,
                              ue_positions: List[torch.Tensor],
                              selected_subcarriers: Dict,
                              antenna_embeddings: torch.Tensor) -> Dict:
        """
        Trace rays using CUDA kernel for maximum performance.
        
        Args:
            base_station_pos: Base station position
            direction_vectors: Pre-computed direction vectors
            ue_positions: List of UE positions
            selected_subcarriers: Dictionary mapping UE to selected subcarriers
            antenna_embeddings: Antenna embedding parameters
        
        Returns:
            Dictionary mapping (ue_pos, subcarrier) to signal strength
        """
        if not self.use_cuda or not hasattr(self, 'cuda_module'):
            return self.trace_rays_pytorch_gpu(
                base_station_pos, direction_vectors, ue_positions, 
                selected_subcarriers, antenna_embeddings
            )
        
        # Prepare data for CUDA kernel
        num_ue = len(ue_positions)
        num_subcarriers = max(len(subcarriers) for subcarriers in selected_subcarriers.values())
        total_rays = self.total_directions * num_ue * num_subcarriers
        
        # Flatten UE positions
        ue_positions_flat = torch.cat(ue_positions, dim=0).to(self.device)
        
        # Create subcarrier indices
        subcarrier_indices = []
        for ue_pos in ue_positions:
            ue_subcarriers = selected_subcarriers.get(tuple(ue_pos), [])
            subcarrier_indices.extend(ue_subcarriers)
        
        subcarrier_tensor = torch.tensor(subcarrier_indices, dtype=torch.int32, device=self.device)
        
        # Prepare output tensor
        signal_strengths = torch.zeros(total_rays, dtype=torch.float32, device=self.device)
        
        # Launch CUDA kernel
        block_size = 256
        grid_size = (total_rays + block_size - 1) // block_size
        
        start_time = time.time()
        
        self.cuda_module.parallel_ray_tracing(
            base_station_pos.contiguous(),
            direction_vectors.contiguous(),
            ue_positions_flat.contiguous(),
            subcarrier_tensor.contiguous(),
            antenna_embeddings.contiguous(),
            signal_strengths,
            self.total_directions,
            num_ue,
            num_subcarriers,
            self.max_ray_length,
            self.scene_size,
            self.uniform_samples,
            self.resampled_points,
            grid=(grid_size, 1, 1),
            block=(block_size, 1, 1)
        )
        
        cuda_time = time.time() - start_time
        logger.info(f"CUDA kernel execution time: {cuda_time:.4f}s")
        
        # Process results
        results = {}
        ray_idx = 0
        
        for ue_pos in ue_positions:
            ue_subcarriers = selected_subcarriers.get(tuple(ue_pos), [])
            
            for subcarrier_idx in ue_subcarriers:
                for direction_idx in range(self.total_directions):
                    signal_strength = signal_strengths[ray_idx].item()
                    results[(tuple(ue_pos), subcarrier_idx, direction_idx)] = signal_strength
                    ray_idx += 1
        
        return results
    
    def trace_rays_pytorch_gpu(self,
                              base_station_pos: torch.Tensor,
                              direction_vectors: torch.Tensor,
                              ue_positions: List[torch.Tensor],
                              selected_subcarriers: Dict,
                              antenna_embeddings: torch.Tensor) -> Dict:
        """
        Trace rays using PyTorch GPU operations as fallback.
        
        Args:
            base_station_pos: Base station position
            direction_vectors: Pre-computed direction vectors
            ue_positions: List of UE positions
            selected_subcarriers: Dictionary mapping UE to selected subcarriers
            antenna_embeddings: Antenna embedding parameters
        
        Returns:
            Dictionary mapping (ue_pos, subcarrier) to signal strength
        """
        logger.info("Using PyTorch GPU operations for ray tracing")
        
        results = {}
        start_time = time.time()
        
        # Vectorized computation on GPU
        for ue_pos in ue_positions:
            ue_pos_tensor = torch.tensor(ue_pos, dtype=torch.float32, device=self.device)
            ue_subcarriers = selected_subcarriers.get(tuple(ue_pos), [])
            
            for subcarrier_idx in ue_subcarriers:
                # Calculate ray lengths for all directions at once
                ray_to_ue = ue_pos_tensor - base_station_pos
                ray_lengths = torch.clamp(
                    torch.sum(ray_to_ue.unsqueeze(0) * direction_vectors, dim=1),
                    0, self.max_ray_length
                )
                
                # Sample points along all rays simultaneously
                t_values = torch.linspace(0, 1, self.uniform_samples, device=self.device)
                t_values = t_values.unsqueeze(0) * ray_lengths.unsqueeze(1)  # (num_directions, num_samples)
                
                # Calculate sample positions for all rays
                sample_positions = (base_station_pos.unsqueeze(0).unsqueeze(0) + 
                                  direction_vectors.unsqueeze(1) * t_values.unsqueeze(-1))
                # Shape: (num_directions, num_samples, 3)
                
                # Check scene bounds
                scene_bounds = self.scene_size / 2.0
                valid_mask = torch.all(
                    (sample_positions >= -scene_bounds) & (sample_positions <= scene_bounds),
                    dim=-1
                )
                
                # Calculate distances to UE for all samples
                distances_to_ue = torch.norm(
                    sample_positions - ue_pos_tensor.unsqueeze(0).unsqueeze(0),
                    dim=-1
                )
                
                # Apply attenuation model
                attenuations = torch.exp(-distances_to_ue / 50.0)
                
                # Apply antenna embedding influence
                antenna_factor = torch.norm(antenna_embeddings[subcarrier_idx]) / math.sqrt(64)
                
                # Apply frequency effects
                frequency_factor = 1.0 / (1.0 + 0.1 * subcarrier_idx)
                
                # Calculate signal contributions
                step_sizes = ray_lengths.unsqueeze(1) / self.uniform_samples
                signal_contributions = (attenuations * antenna_factor * frequency_factor * 
                                      step_sizes * valid_mask.float())
                
                # Integrate along rays
                signal_strengths = torch.sum(signal_contributions, dim=1)
                
                # Store results for each direction
                for direction_idx in range(self.total_directions):
                    results[(tuple(ue_pos), subcarrier_idx, direction_idx)] = signal_strengths[direction_idx].item()
        
        pytorch_time = time.time() - start_time
        logger.info(f"PyTorch GPU operations time: {pytorch_time:.4f}s")
        
        return results
    
    def trace_rays_cpu(self,
                       base_station_pos: torch.Tensor,
                       direction_vectors: torch.Tensor,
                       ue_positions: List[torch.Tensor],
                       selected_subcarriers: Dict,
                       antenna_embeddings: torch.Tensor) -> Dict:
        """
        Fallback CPU implementation for ray tracing.
        
        Args:
            base_station_pos: Base station position
            direction_vectors: Pre-computed direction vectors
            ue_positions: List of UE positions
            selected_subcarriers: Dictionary mapping UE to selected subcarriers
            antenna_embeddings: Antenna embedding parameters
        
        Returns:
            Dictionary mapping (ue_pos, subcarrier) to signal strength
        """
        logger.info("Using CPU implementation for ray tracing")
        
        results = {}
        start_time = time.time()
        
        # Move tensors to CPU for computation
        base_station_pos_cpu = base_station_pos.cpu()
        direction_vectors_cpu = direction_vectors.cpu()
        
        for ue_pos in ue_positions:
            ue_pos_tensor = torch.tensor(ue_pos, dtype=torch.float32)
            ue_subcarriers = selected_subcarriers.get(tuple(ue_pos), [])
            
            for subcarrier_idx in ue_subcarriers:
                for direction_idx in range(self.total_directions):
                    direction = direction_vectors_cpu[direction_idx]
                    
                    # Calculate ray length to UE
                    ray_to_ue = ue_pos_tensor - base_station_pos_cpu
                    ray_length = torch.clamp(
                        torch.dot(ray_to_ue, direction),
                        0, self.max_ray_length
                    )
                    
                    # Sample points along ray
                    signal_strength = 0.0
                    step_size = ray_length / self.uniform_samples
                    
                    for i in range(self.uniform_samples):
                        t = i * step_size
                        sample_pos = base_station_pos_cpu + direction * t
                        
                        # Check scene bounds
                        if (torch.abs(sample_pos) <= self.scene_size / 2.0).all():
                            # Calculate distance-based attenuation
                            distance_to_ue = torch.norm(sample_pos - ue_pos_tensor)
                            attenuation = torch.exp(-distance_to_ue / 50.0)
                            
                            # Apply antenna embedding influence
                            antenna_factor = torch.norm(antenna_embeddings[subcarrier_idx]) / math.sqrt(64)
                            
                            # Apply frequency effects
                            frequency_factor = 1.0 / (1.0 + 0.1 * subcarrier_idx)
                            
                            # Accumulate signal contribution
                            signal_strength += attenuation * antenna_factor * frequency_factor * step_size
                    
                    results[(tuple(ue_pos), subcarrier_idx, direction_idx)] = signal_strength
        
        cpu_time = time.time() - start_time
        logger.info(f"CPU implementation time: {cpu_time:.4f}s")
        
        return results
    
    def trace_rays(self,
                   base_station_pos: torch.Tensor,
                   ue_positions: List[torch.Tensor],
                   selected_subcarriers: Dict,
                   antenna_embeddings: torch.Tensor) -> Dict:
        """
        Main ray tracing method with automatic device selection.
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Dictionary mapping UE to selected subcarriers
            antenna_embeddings: Antenna embedding parameters
        
        Returns:
            Dictionary mapping (ue_pos, subcarrier, direction) to signal strength
        """
        # Generate direction vectors
        direction_vectors = self.generate_direction_vectors()
        
        # Select implementation based on device
        if self.use_cuda and self.device == 'cuda':
            try:
                return self.trace_rays_cuda_kernel(
                    base_station_pos, direction_vectors, ue_positions,
                    selected_subcarriers, antenna_embeddings
                )
            except Exception as e:
                logger.warning(f"CUDA kernel failed: {e}. Falling back to PyTorch GPU operations.")
                return self.trace_rays_pytorch_gpu(
                    base_station_pos, direction_vectors, ue_positions,
                    selected_subcarriers, antenna_embeddings
                )
        elif self.device == 'cuda':
            return self.trace_rays_pytorch_gpu(
                base_station_pos, direction_vectors, ue_positions,
                selected_subcarriers, antenna_embeddings
            )
        else:
            return self.trace_rays_cpu(
                base_station_pos, direction_vectors, ue_positions,
                selected_subcarriers, antenna_embeddings
            )
    
    def get_performance_info(self) -> Dict:
        """Get performance information and device capabilities."""
        info = {
            'device': self.device,
            'use_cuda': self.use_cuda,
            'total_directions': self.total_directions,
            'uniform_samples': self.uniform_samples,
            'resampled_points': self.resampled_points
        }
        
        if self.use_cuda:
            info['cuda_device_name'] = torch.cuda.get_device_name()
            info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info['cuda_compute_capability'] = torch.cuda.get_device_capability()
        
        return info
    
    def is_position_in_scene(self, position: torch.Tensor) -> bool:
        """
        Check if a position is within the scene boundaries.
        
        Args:
            position: 3D position tensor [x, y, z]
        
        Returns:
            True if position is within scene boundaries
        """
        if position.dim() == 1:
            position = position.unsqueeze(0)
        
        # Check if all coordinates are within bounds
        scene_bounds = self.scene_size / 2.0
        in_bounds = torch.all(
            (position >= -scene_bounds) & (position <= scene_bounds), 
            dim=1
        )
        
        return in_bounds.all().item()
    
    def get_scene_bounds(self) -> Tuple[float, float]:
        """Get scene boundaries."""
        scene_bounds = self.scene_size / 2.0
        return -scene_bounds, scene_bounds
    
    def get_scene_size(self) -> float:
        """Get scene size D."""
        return self.scene_size
    
    def update_scene_size(self, new_scene_size: float):
        """
        Update scene size and related parameters.
        
        Args:
            new_scene_size: New scene size in meters
        """
        if new_scene_size <= 0:
            raise ValueError(f"Scene size must be positive, got {new_scene_size}")
        
        self.scene_size = new_scene_size
        
        # Adjust max ray length if necessary
        if self.max_ray_length > new_scene_size:
            self.max_ray_length = new_scene_size
            logger.info(f"Adjusted max ray length to {self.max_ray_length}m")
        
        logger.info(f"Updated scene size to {new_scene_size}m")
    
    def get_scene_config(self) -> Dict[str, float]:
        """Get complete scene configuration."""
        scene_bounds = self.scene_size / 2.0
        return {
            'scene_size': self.scene_size,
            'scene_min': -scene_bounds,
            'scene_max': scene_bounds,
            'max_ray_length': self.max_ray_length,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions
        }
    
    def trace_ray(self, 
                  base_station_pos: torch.Tensor,
                  direction: Tuple[int, int],
                  ue_positions: List[torch.Tensor],
                  selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                  antenna_embedding: torch.Tensor) -> Dict:
        """
        Trace RF signal along a single ray direction.
        
        Args:
            base_station_pos: Base station position P_BS
            direction: Direction indices (phi_idx, theta_idx)
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information from training interface
            antenna_embedding: Base station's antenna embedding parameter C
        
        Returns:
            Dictionary mapping (ue_pos, subcarrier) to received RF signal strength
        """
        # Convert direction indices to direction vector
        phi_idx, theta_idx = direction
        phi = phi_idx * (2 * math.pi / self.azimuth_divisions)
        theta = theta_idx * (math.pi / self.elevation_divisions)
        
        direction_vector = torch.tensor([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta)
        ], dtype=torch.float32, device=self.device)
        
        # Use existing trace_rays method with single direction
        # Convert selected_subcarriers to the expected Dict format
        if isinstance(selected_subcarriers, (list, tuple)):
            # Convert list to dict format
            subcarrier_dict = {}
            for ue_pos in ue_positions:
                ue_key = tuple(ue_pos.tolist())
                subcarrier_dict[ue_key] = selected_subcarriers
        elif isinstance(selected_subcarriers, torch.Tensor):
            # Convert tensor to dict format
            subcarrier_dict = {}
            for ue_pos in ue_positions:
                ue_key = tuple(ue_pos.tolist())
                subcarrier_dict[ue_key] = selected_subcarriers.tolist()
        else:
            subcarrier_dict = selected_subcarriers
        
        # Create antenna embeddings tensor with proper shape
        if antenna_embedding.dim() == 1:
            # Single embedding, expand to match subcarriers
            num_subcarriers = len(selected_subcarriers) if isinstance(selected_subcarriers, (list, tuple)) else selected_subcarriers.shape[0]
            antenna_embeddings = antenna_embedding.unsqueeze(0).expand(num_subcarriers, -1)
        else:
            antenna_embeddings = antenna_embedding
        
        # Create antenna embeddings tensor with proper shape
        if antenna_embedding.dim() == 1:
            # Single embedding, expand to match subcarriers
            if isinstance(selected_subcarriers, (list, tuple)):
                num_subcarriers = len(selected_subcarriers)
            elif isinstance(selected_subcarriers, torch.Tensor):
                num_subcarriers = selected_subcarriers.shape[0]
            else:  # dict
                # Count total subcarriers across all UEs
                num_subcarriers = sum(len(subcarriers) for subcarriers in selected_subcarriers.values())
            antenna_embeddings = antenna_embedding.unsqueeze(0).expand(num_subcarriers, -1)
        else:
            antenna_embeddings = antenna_embedding
        
        results = self.trace_rays(
            base_station_pos, ue_positions, subcarrier_dict, antenna_embeddings
        )
        
        # Convert results to single ray format
        # The trace_rays method returns results in format (ue_pos, subcarrier, direction_idx)
        # We need to extract results for the first direction only
        single_ray_results = {}
        for (ue_pos, subcarrier, dir_idx), signal_strength in results.items():
            if dir_idx == 0:  # Only first direction
                single_ray_results[(ue_pos, subcarrier)] = signal_strength
        
        return single_ray_results
    
    def accumulate_signals(self, 
                          base_station_pos: torch.Tensor,
                          ue_positions: List[torch.Tensor],
                          selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                          antenna_embedding: torch.Tensor) -> Dict:
        """
        Accumulate RF signals using MLP-based direction sampling with antenna embedding C.
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information from training interface
            antenna_embedding: Base station's antenna embedding parameter C
        
        Returns:
            Accumulated signal strength matrix for all virtual links
        """
        # Generate all direction vectors
        direction_vectors = self.generate_direction_vectors()
        
        # Use existing trace_rays method for all directions
        # Convert selected_subcarriers to the expected Dict format
        if isinstance(selected_subcarriers, (list, tuple)):
            # Convert list to dict format
            subcarrier_dict = {}
            for ue_pos in ue_positions:
                ue_key = tuple(ue_pos.tolist())
                subcarrier_dict[ue_key] = selected_subcarriers
        elif isinstance(selected_subcarriers, torch.Tensor):
            # Convert tensor to dict format
            subcarrier_dict = {}
            for ue_pos in ue_positions:
                ue_key = tuple(ue_pos.tolist())
                subcarrier_dict[ue_key] = selected_subcarriers.tolist()
        else:
            subcarrier_dict = selected_subcarriers
        
        # Create antenna embeddings tensor with proper shape
        if antenna_embedding.dim() == 1:
            # Single embedding, expand to match subcarriers
            num_subcarriers = len(selected_subcarriers) if isinstance(selected_subcarriers, (list, tuple)) else selected_subcarriers.shape[0]
            antenna_embeddings = antenna_embedding.unsqueeze(0).expand(num_subcarriers, -1)
        else:
            antenna_embeddings = antenna_embedding
        
        # Create antenna embeddings tensor with proper shape
        if antenna_embedding.dim() == 1:
            # Single embedding, expand to match subcarriers
            if isinstance(selected_subcarriers, (list, tuple)):
                num_subcarriers = len(selected_subcarriers)
            elif isinstance(selected_subcarriers, torch.Tensor):
                num_subcarriers = selected_subcarriers.shape[0]
            else:  # dict
                # Count total subcarriers across all UEs
                num_subcarriers = sum(len(subcarriers) for subcarriers in selected_subcarriers.values())
            antenna_embeddings = antenna_embedding.unsqueeze(0).expand(num_subcarriers, -1)
        else:
            antenna_embeddings = antenna_embedding
        
        all_results = self.trace_rays(
            base_station_pos, ue_positions, subcarrier_dict, antenna_embeddings
        )
        
        # Accumulate signals across all directions
        accumulated_signals = {}
        for (ue_pos, subcarrier, dir_idx), signal_strength in all_results.items():
            key = (ue_pos, subcarrier)
            if key not in accumulated_signals:
                accumulated_signals[key] = 0.0
            accumulated_signals[key] += signal_strength
        
        return accumulated_signals
    
    def adaptive_ray_tracing(self, 
                           base_station_pos: torch.Tensor,
                           antenna_embedding: torch.Tensor,
                           ue_positions: List[torch.Tensor],
                           selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                           top_k: int = 32) -> Dict:
        """
        Perform adaptive ray tracing using built-in AntennaNetwork for direction selection.
        
        Args:
            base_station_pos: Base station position
            antenna_embedding: Base station's antenna embedding parameter C
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information from training interface
            top_k: Number of top directions to select
        
        Returns:
            Accumulated signal strength for selected directions only
        """
        # For now, use the main accumulate_signals method
        # In the future, this could implement MLP-based direction selection
        return self.accumulate_signals(
            base_station_pos, ue_positions, selected_subcarriers, antenna_embedding
        )
    
    def pyramid_ray_tracing(self,
                           base_station_pos: torch.Tensor,
                           ue_positions: List[torch.Tensor],
                           selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                           antenna_embedding: torch.Tensor,
                           pyramid_levels: int = 3) -> Dict:
        """
        Perform pyramid ray tracing with hierarchical sampling.
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information from training interface
            antenna_embedding: Base station's antenna embedding parameter C
            pyramid_levels: Number of hierarchical levels
        
        Returns:
            Accumulated signal strength with pyramid sampling
        """
        accumulated_signals = {}
        
        # Implement hierarchical pyramid sampling
        for level in range(pyramid_levels):
            # Calculate sampling density for this level
            level_factor = 2 ** level
            level_azimuth_divisions = max(1, self.azimuth_divisions // level_factor)
            level_elevation_divisions = max(1, self.elevation_divisions // level_factor)
            
            logger.debug(f"Pyramid level {level}: {level_azimuth_divisions}x{level_elevation_divisions} directions")
            
            # Sample directions for this pyramid level
            for phi_idx in range(0, self.azimuth_divisions, level_factor):
                for theta_idx in range(0, self.elevation_divisions, level_factor):
                    direction = (phi_idx, theta_idx)
                    
                    # Trace ray for this direction
                    ray_results = self.trace_ray(
                        base_station_pos, direction, ue_positions,
                        selected_subcarriers, antenna_embedding
                    )
                    
                    # Accumulate signals with level weighting
                    level_weight = 1.0 / (level + 1)  # Higher levels get lower weight
                    for (ue_pos, subcarrier), signal_strength in ray_results.items():
                        if (ue_pos, subcarrier) not in accumulated_signals:
                            accumulated_signals[(ue_pos, subcarrier)] = 0.0
                        accumulated_signals[(ue_pos, subcarrier)] += signal_strength * level_weight
        
        return accumulated_signals
    
    def get_ray_count_analysis(self, num_bs: int, num_ue: int, num_subcarriers: int) -> Dict:
        """
        Analyze the total number of rays in the system.
        
        Args:
            num_bs: Number of base stations
            num_ue: Number of user equipment devices
            num_subcarriers: Number of subcarriers in the frequency domain
        
        Returns:
            Dictionary with ray count analysis
        """
        total_rays = num_bs * self.total_directions * num_ue * num_subcarriers
        
        return {
            'total_directions': self.total_directions,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions,
            'total_rays': total_rays,
            'ray_count_formula': f"N_total = N_BS × A × B × N_UE × K = {num_bs} × {self.total_directions} × {num_ue} × {num_subcarriers}"
        }
