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

class Ray:
    """Represents a single ray for ray tracing."""
    
    def __init__(self, origin: torch.Tensor, direction: torch.Tensor, max_length: float = 100.0, device: str = 'cpu'):
        """
        Initialize a ray.
        
        Args:
            origin: Ray origin point [3]
            direction: Ray direction vector [3]
            max_length: Maximum ray length
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.device = device
        self.origin = origin.clone().detach().to(dtype=torch.float32, device=device)
        self.direction = self._normalize(direction.clone().detach().to(dtype=torch.float32, device=device))
        self.max_length = max_length
    
    def _normalize(self, vector: torch.Tensor) -> torch.Tensor:
        """Normalize direction vector."""
        norm = torch.norm(vector)
        if norm < 1e-10:
            return vector
        return vector / norm

class BaseStation:
    """Represents a base station with configurable location and antennas."""
    
    def __init__(self, position: torch.Tensor = None, num_antennas: int = 1, device: str = 'cpu'):
        """
        Initialize base station.
        
        Args:
            position: Base station position [3], defaults to origin (0, 0, 0)
            num_antennas: Number of antennas at this base station
            device: Device to run computations on
        """
        self.device = device
        self.position = torch.tensor([0.0, 0.0, 0.0], device=device) if position is None else torch.tensor(position, device=device)
        self.num_antennas = num_antennas
        self.antenna_embeddings = torch.randn(num_antennas, 128, device=device)  # 128D antenna embedding
    
    def get_antenna_embedding(self, antenna_idx: int = 0) -> torch.Tensor:
        """Get antenna embedding parameter C for the specified antenna."""
        return self.antenna_embeddings[antenna_idx]

class UserEquipment:
    """Represents user equipment at a specific location."""
    
    def __init__(self, position: torch.Tensor, device: str = 'cpu'):
        """
        Initialize user equipment.
        
        Args:
            position: UE position [3]
            device: Device to run computations on
        """
        self.position = position.clone().detach().to(dtype=torch.float32, device=device)

# CUDA kernel for parallel ray tracing with enhanced features
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
    const int resampled_points,
    const float signal_threshold
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
    
    // Early termination if ray length is too short
    if (ray_length < 1e-6f) {
        signal_strengths[idx] = 0.0f;
        return;
    }
    
    // Uniform sampling along ray with importance sampling
    float signal_strength = 0.0f;
    float step_size = ray_length / uniform_samples;
    float cumulative_attenuation = 1.0f;
    
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
            
            // Enhanced attenuation model with distance and frequency
            float distance_attenuation = expf(-distance_to_ue / 50.0f);
            float frequency_attenuation = 1.0f / (1.0f + 0.1f * subcarrier_idx);
            float attenuation = distance_attenuation * frequency_attenuation;
            
            // Apply antenna embedding influence (128D embedding)
            float antenna_factor = 0.0f;
            for (int j = 0; j < 128; j++) {
                antenna_factor += antenna_embeddings[subcarrier_idx * 128 + j] * 
                                antenna_embeddings[subcarrier_idx * 128 + j];
            }
            antenna_factor = sqrtf(antenna_factor) / 11.3137f; // Normalize to [0, 1]
            
            // Apply cumulative attenuation and accumulate signal
            float local_contribution = attenuation * antenna_factor * step_size;
            signal_strength += cumulative_attenuation * local_contribution;
            
            // Update cumulative attenuation for next sample
            cumulative_attenuation *= expf(-attenuation * step_size);
            
            // Early termination if signal strength falls below threshold
            if (cumulative_attenuation < signal_threshold) {
                break;
            }
        }
    }
    
    // Store result
    signal_strengths[idx] = signal_strength;
}
"""

class CUDARayTracer:
    """CUDA-accelerated discrete ray tracer implementing the design document specifications."""
    
    def __init__(self, 
                 azimuth_divisions: int = 36,
                 elevation_divisions: int = 18,
                 max_ray_length: float = 100.0,
                 scene_size: float = 200.0,
                 device: str = 'cpu',
                 prism_network=None,
                 signal_threshold: float = 1e-6,
                 enable_early_termination: bool = True,
                 uniform_samples: int = 128,
                 resampled_points: int = 64):
        """
        Initialize CUDA discrete ray tracer.
        
        Args:
            azimuth_divisions: Number of azimuth divisions A (0° to 360°)
            elevation_divisions: Number of elevation divisions B (-90° to +90°)
            max_ray_length: Maximum ray length in meters
            scene_size: Scene size D in meters (cubic environment: [-D/2, D/2]³)
            device: Device to run computations on
            prism_network: PrismNetwork instance for getting attenuation and radiance properties
            signal_threshold: Minimum signal strength threshold for early termination
            enable_early_termination: Enable early termination optimization
            uniform_samples: Number of uniform samples per ray
            resampled_points: Number of resampled points per ray
        """
        self.device = device
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.max_ray_length = max_ray_length
        self.scene_size = scene_size
        self.prism_network = prism_network
        self.signal_threshold = signal_threshold
        self.enable_early_termination = enable_early_termination
        self.uniform_samples = uniform_samples
        self.resampled_points = resampled_points
        
        # Calculate angular resolutions
        self.azimuth_resolution = 2 * math.pi / azimuth_divisions
        self.elevation_resolution = math.pi / elevation_divisions
        
        # Total number of directions
        self.total_directions = azimuth_divisions * elevation_divisions
        
        # Scene boundaries
        self.scene_min = -scene_size / 2.0
        self.scene_max = scene_size / 2.0
        
        # Device detection and CUDA setup
        self.device, self.use_cuda = self._detect_device()
        self._setup_cuda()
        
        # Validate scene configuration
        self._validate_scene_config()
        
        logger.info(f"CUDA Ray Tracer initialized with {azimuth_divisions}x{elevation_divisions} = {self.total_directions} directions")
        logger.info(f"Scene size: {scene_size}m, boundaries: [{self.scene_min:.1f}, {self.scene_max:.1f}]³")
        if self.use_cuda:
            logger.info("✓ CUDA acceleration enabled - significant performance improvement expected")
        else:
            logger.info("⚠ CUDA not available - using CPU implementation")
    
    def _validate_scene_config(self):
        """Validate scene configuration parameters."""
        if self.scene_size <= 0:
            raise ValueError(f"Scene size must be positive, got {self.scene_size}")
        
        if self.max_ray_length > self.scene_size:
            logger.warning(f"Max ray length ({self.max_ray_length}m) exceeds scene size ({self.scene_size}m)")
            # Adjust max ray length to scene size
            self.max_ray_length = min(self.max_ray_length, self.scene_size)
            logger.info(f"Adjusted max ray length to {self.max_ray_length}m")
        
        if self.azimuth_divisions <= 0 or self.elevation_divisions <= 0:
            raise ValueError("Azimuth and elevation divisions must be positive")
    
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
        """Generate unit direction vectors for all A×B directions."""
        directions = []
        
        for i in range(self.azimuth_divisions):
            for j in range(self.elevation_divisions):
                phi = i * self.azimuth_resolution  # Azimuth angle
                theta = j * self.elevation_resolution  # Elevation angle
                
                # Convert to Cartesian coordinates
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
            self.signal_threshold,
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
                           - Dict: Mapping UE to subcarrier indices
                           - torch.Tensor: Tensor of subcarrier indices
                           - List[int]: List of subcarrier indices
                           Note: This MUST be provided by the calling code
            antenna_embedding: Base station's antenna embedding parameter C
        
        Returns:
            Dictionary mapping (ue_pos, subcarrier) to received RF signal strength
        """
        # Validate and normalize selected_subcarriers input
        subcarrier_indices = self._normalize_subcarrier_input(selected_subcarriers, ue_positions)
        
        phi_idx, theta_idx = direction
        
        # Convert indices to angles
        phi = phi_idx * self.azimuth_resolution
        theta = theta_idx * self.elevation_resolution
        
        # Create direction vector
        direction_vector = torch.tensor([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta)
        ], dtype=torch.float32, device=self.device)
        
        # Create ray
        ray = Ray(base_station_pos, direction_vector, self.max_ray_length, self.device)
        
        results = {}
        
        for ue_pos in ue_positions:
            ue_pos_tensor = ue_pos.clone().detach().to(dtype=torch.float32, device=self.device)
            
            # Use the normalized subcarrier indices
            for subcarrier_idx in subcarrier_indices:
                # Apply discrete radiance field model for ray tracing
                signal_strength = self._discrete_radiance_ray_tracing(
                    ray, ue_pos_tensor, subcarrier_idx, antenna_embedding
                )
                results[(tuple(ue_pos), subcarrier_idx)] = signal_strength
        
        return results
    
    def _normalize_subcarrier_input(self, 
                                  selected_subcarriers: Union[Dict, torch.Tensor, List[int]], 
                                  ue_positions: List[torch.Tensor]) -> List[int]:
        """
        Normalize subcarrier input to a list of indices.
        
        Args:
            selected_subcarriers: Various formats of subcarrier selection
            ue_positions: List of UE positions for validation
            
        Returns:
            Normalized list of subcarrier indices
            
        Raises:
            ValueError: If subcarrier input is invalid or empty
        """
        if selected_subcarriers is None:
            raise ValueError("selected_subcarriers cannot be None. Must be provided by calling code.")
        
        if isinstance(selected_subcarriers, dict):
            # Dictionary format: extract unique subcarrier indices
            all_indices = set()
            
            for ue_pos in ue_positions:
                # Convert tensor to tuple for comparison
                ue_key = tuple(ue_pos.tolist())
                
                if ue_key in selected_subcarriers:
                    indices = selected_subcarriers[ue_key]
                    
                    if isinstance(indices, (list, tuple)):
                        all_indices.update(indices)
                    elif isinstance(indices, torch.Tensor):
                        all_indices.update(indices.tolist())
                    elif isinstance(indices, (int, float)):
                        all_indices.add(int(indices))
                    else:
                        logger.debug(f"  Unknown type: {type(indices)}, trying to convert")
                        # Try to convert to int if possible
                        try:
                            all_indices.add(int(indices))
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert indices {indices} (type: {type(indices)}) to int")
                            continue
                else:
                    pass
            
            if not all_indices:
                raise ValueError("No valid subcarrier indices found in selected_subcarriers dictionary")
            
            return sorted(list(all_indices))
            
        elif isinstance(selected_subcarriers, torch.Tensor):
            # Tensor format: convert to list
            if selected_subcarriers.numel() == 0:
                raise ValueError("selected_subcarriers tensor is empty")
            return selected_subcarriers.flatten().tolist()
            
        elif isinstance(selected_subcarriers, (list, tuple)):
            # List/tuple format: validate and return
            if not selected_subcarriers:
                raise ValueError("selected_subcarriers list is empty")
            return [int(idx) for idx in selected_subcarriers]
            
        else:
            raise ValueError(f"Unsupported selected_subcarriers type: {type(selected_subcarriers)}")
    
    def _discrete_radiance_ray_tracing(self, 
                                     ray: Ray,
                                     ue_pos: torch.Tensor,
                                     subcarrier_idx: int,
                                     antenna_embedding: torch.Tensor) -> float:
        """
        Apply discrete radiance field model for ray tracing using importance-based sampling.
        
        This method implements the two-stage importance-based sampling:
        1. Uniform sampling with weight computation
        2. Importance-based resampling based on computed weights
        
        Args:
            ray: Ray object
            ue_pos: UE position
            subcarrier_idx: Subcarrier index
            antenna_embedding: Antenna embedding parameter
        
        Returns:
            Computed signal strength using discrete radiance field model
        """
        if self.prism_network is None:
            # Fallback to simple distance-based model if no network is provided
            return self._simple_distance_model(ray, ue_pos, subcarrier_idx, antenna_embedding)
        
        # Stage 1: Uniform sampling with weight computation
        num_uniform_samples = 128  # Higher initial sampling for better weight estimation
        uniform_positions = self._sample_ray_points(ray, ue_pos, num_uniform_samples)
        
        if len(uniform_positions) == 0:
            return 0.0
        
        # Get viewing directions for uniform samples
        uniform_view_directions = ue_pos.unsqueeze(0).expand(num_uniform_samples, -1) - uniform_positions
        uniform_view_directions = uniform_view_directions / (torch.norm(uniform_view_directions, dim=1, keepdim=True) + 1e-8)
        
        # Create antenna indices
        antenna_indices = torch.zeros(1, dtype=torch.long, device=self.device)
        
        try:
            # Get network properties for uniform samples
            with torch.no_grad():
                uniform_network_outputs = self.prism_network(
                    sampled_positions=uniform_positions.unsqueeze(0),
                    ue_positions=ue_pos.unsqueeze(0),
                    view_directions=uniform_view_directions.mean(dim=0, keepdim=True),
                    antenna_indices=antenna_indices,
                    return_intermediates=False
                )
            
            # Extract attenuation factors for weight computation
            uniform_attenuation = uniform_network_outputs['attenuation_factors'][0, :, 0, subcarrier_idx]  # (num_uniform_samples,)
            
            # Stage 2: Importance-based resampling
            importance_weights = self._compute_importance_weights(uniform_attenuation)
            resampled_positions = self._importance_based_resampling(
                uniform_positions, importance_weights, num_samples=64
            )
            
            # Get network properties for resampled points
            resampled_view_directions = ue_pos.unsqueeze(0).expand(len(resampled_positions), -1) - resampled_positions
            resampled_view_directions = resampled_view_directions / (torch.norm(resampled_view_directions, dim=1, keepdim=True) + 1e-8)
            
            with torch.no_grad():
                resampled_network_outputs = self.prism_network(
                    sampled_positions=resampled_positions.unsqueeze(0),
                    ue_positions=ue_pos.unsqueeze(0),
                    view_directions=resampled_view_directions.mean(dim=0, keepdim=True),
                    antenna_indices=antenna_indices,
                    return_intermediates=False
                )
            
            # Extract final attenuation and radiation factors
            final_attenuation_factors = resampled_network_outputs['attenuation_factors']
            final_radiation_factors = resampled_network_outputs['radiation_factors']
            
            # Apply discrete radiance field integration with importance sampling
            signal_strength = self._integrate_along_ray_with_importance(
                resampled_positions, final_attenuation_factors, final_radiation_factors, 
                subcarrier_idx, importance_weights
            )
            
            return signal_strength
            
        except Exception as e:
            logger.warning(f"Neural network computation failed: {e}. Using fallback model.")
            return self._simple_distance_model(ray, ue_pos, subcarrier_idx, antenna_embedding)
    
    def _sample_ray_points(self, ray: Ray, ue_pos: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Sample points along the ray for discrete radiance field computation.
        
        Args:
            ray: Ray object
            ue_pos: UE position
            num_samples: Number of sample points
        
        Returns:
            Sampled positions along the ray
        """
        # Calculate ray length to UE
        ray_to_ue = ue_pos - ray.origin
        ray_length = torch.dot(ray_to_ue, ray.direction)
        ray_length = torch.clamp(ray_length, 0, self.max_ray_length)
        
        # Sample points along the ray
        t_values = torch.linspace(0, ray_length, num_samples, device=self.device)
        sampled_positions = ray.origin.unsqueeze(0) + t_values.unsqueeze(1) * ray.direction.unsqueeze(0)
        
        # Filter out points outside scene boundaries
        valid_mask = self.is_position_in_scene(sampled_positions)
        if not valid_mask:
            # If no valid positions, return empty tensor
            return torch.empty(0, 3, device=self.device)
        
        # Return only valid positions
        valid_positions = sampled_positions[valid_mask]
        
        # Ensure we have at least some samples
        if len(valid_positions) < num_samples // 2:
            logger.warning(f"Only {len(valid_positions)} valid positions out of {num_samples} requested")
        
        return valid_positions
    
    def _compute_importance_weights(self, attenuation_factors: torch.Tensor) -> torch.Tensor:
        """
        Compute importance weights based on attenuation factors.
        
        Higher attenuation regions get higher weights for importance sampling.
        
        Args:
            attenuation_factors: Attenuation factors from uniform sampling (num_samples,)
        
        Returns:
            Importance weights for resampling (num_samples,)
        """
        # Convert complex attenuation to magnitude
        attenuation_magnitude = torch.abs(attenuation_factors)
        
        # Normalize to [0, 1] range
        if torch.max(attenuation_magnitude) > 0:
            normalized_attenuation = attenuation_magnitude / torch.max(attenuation_magnitude)
        else:
            normalized_attenuation = torch.ones_like(attenuation_magnitude)
        
        # Apply non-linear transformation to emphasize high-attenuation regions
        # Use power function to increase contrast
        importance_weights = torch.pow(normalized_attenuation, 2.0)
        
        # Add small epsilon to avoid zero weights
        importance_weights = importance_weights + 1e-6
        
        # Normalize weights to sum to 1
        importance_weights = importance_weights / torch.sum(importance_weights)
        
        return importance_weights
    
    def _importance_based_resampling(self, 
                                   uniform_positions: torch.Tensor,
                                   importance_weights: torch.Tensor,
                                   num_samples: int) -> torch.Tensor:
        """
        Perform importance-based resampling based on computed weights.
        
        Args:
            uniform_positions: Uniformly sampled positions (num_uniform_samples, 3)
            importance_weights: Importance weights for each position (num_uniform_samples,)
            num_samples: Number of samples to select
        
        Returns:
            Resampled positions based on importance (num_samples, 3)
        """
        num_uniform_samples = uniform_positions.shape[0]
        
        if num_samples >= num_uniform_samples:
            # If we want more samples than available, return all with repetition
            return uniform_positions
        
        # Use importance sampling to select positions
        # Higher weight positions have higher probability of being selected
        selected_indices = torch.multinomial(importance_weights, num_samples, replacement=True)
        
        # Get resampled positions
        resampled_positions = uniform_positions[selected_indices]
        
        return resampled_positions
    
    def _integrate_along_ray_with_importance(self,
                                           sampled_positions: torch.Tensor,
                                           attenuation_factors: torch.Tensor,
                                           radiation_factors: torch.Tensor,
                                           subcarrier_idx: int,
                                           importance_weights: torch.Tensor) -> float:
        """
        Integrate signal strength along the ray using importance sampling.
        
        Args:
            sampled_positions: Sampled positions along ray (num_samples, 3)
            attenuation_factors: Attenuation factors from network (1, num_samples, N_UE, K)
            radiation_factors: Radiation factors from network (1, N_UE, K)
            subcarrier_idx: Subcarrier index
            importance_weights: Importance weights for importance sampling
        
        Returns:
            Integrated signal strength with importance sampling correction
        """
        num_samples = sampled_positions.shape[0]
        
        # Extract attenuation and radiation for the specific subcarrier
        if subcarrier_idx >= attenuation_factors.shape[-1]:
            subcarrier_idx = 0  # Fallback to first subcarrier
        
        attenuation = attenuation_factors[0, :, 0, subcarrier_idx]  # (num_samples,)
        radiation = radiation_factors[0, 0, subcarrier_idx]  # scalar
        
        # Calculate step size
        if num_samples > 1:
            distances = torch.norm(sampled_positions[1:] - sampled_positions[:-1], dim=1)
            step_size = distances.mean()
        else:
            step_size = 1.0
        
        # Apply discrete radiance field integration with importance sampling
        # S(P_RX, ω) ≈ Σ exp(-Σ ρ(P_v^j) Δt) ρ(P_v^k) S(P_v^k, -ω) Δt / p(P_v^k)
        cumulative_attenuation = torch.zeros(num_samples, device=self.device)
        signal_contributions = torch.zeros(num_samples, device=self.device)
        
        for k in range(num_samples):
            # Calculate cumulative attenuation up to point k
            if k > 0:
                cumulative_attenuation[k] = cumulative_attenuation[k-1] + torch.abs(attenuation[k-1]) * step_size
            
            # Calculate signal contribution from point k with importance sampling correction
            attenuation_factor = torch.exp(-cumulative_attenuation[k])
            local_contribution = torch.abs(attenuation[k]) * torch.abs(radiation) * step_size
            
            # Apply importance sampling correction factor
            # The correction factor accounts for the probability of selecting this sample
            if k < len(importance_weights):
                importance_correction = 1.0 / (importance_weights[k] + 1e-8)
            else:
                importance_correction = 1.0
            
            signal_contributions[k] = attenuation_factor * local_contribution * importance_correction
            
            # Early termination: stop if signal strength falls below threshold
            if self.enable_early_termination and attenuation_factor < self.signal_threshold:
                logger.debug(f"Early termination at sample {k}/{num_samples}, signal strength: {attenuation_factor:.2e}")
                # Zero out remaining contributions
                signal_contributions[k+1:] = 0.0
                break
        
        # Sum all contributions
        total_signal = torch.sum(signal_contributions)
        
        return total_signal.item()
    
    def _simple_distance_model(self, 
                              ray: Ray,
                              ue_pos: torch.Tensor,
                              subcarrier_idx: int,
                              antenna_embedding: torch.Tensor) -> float:
        """
        Simple distance-based model as fallback when neural network is not available.
        
        Args:
            ray: Ray object
            ue_pos: UE position
            subcarrier_idx: Subcarrier index
            antenna_embedding: Antenna embedding parameter
        
        Returns:
            Computed signal strength using simple model
        """
        # Calculate distance from base station to UE
        distance = torch.norm(ue_pos - ray.origin)
        
        # Apply distance-based attenuation (exponential decay model)
        base_attenuation = torch.exp(-distance / 50.0)  # 50m characteristic distance
        
        # Apply antenna embedding influence
        antenna_factor = torch.norm(antenna_embedding) / math.sqrt(128)  # Normalize to [0, 1]
        
        # Apply frequency-dependent effects (subcarrier index)
        frequency_factor = 1.0 / (1.0 + 0.1 * subcarrier_idx)  # Simple frequency dependency
        
        # Combine factors
        signal_strength = base_attenuation * antenna_factor * frequency_factor
        
        return signal_strength.item()
    
    def accumulate_signals(self, 
                          base_station_pos: torch.Tensor,
                          ue_positions: List[torch.Tensor],
                          selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                          antenna_embedding: torch.Tensor) -> Dict:
        """
        Accumulate RF signals using MLP-based direction sampling with antenna embedding C.
        
        This method implements the design document's MLP-based direction sampling:
        1. Use AntennaNetwork to compute directional importance based on antenna embedding C
        2. Select top-K directions based on importance
        3. Only trace rays for selected directions
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information from training interface
                           - Dict: Mapping UE to subcarrier indices
                           - torch.Tensor: Tensor of subcarrier indices
                           - List[int]: List of subcarrier indices
                           Note: This MUST be provided by the calling code
            antenna_embedding: Base station's antenna embedding parameter C
        
        Returns:
            Accumulated signal strength matrix for all virtual links
        """
        accumulated_signals = {}
        
        # Debug logging
        logger.debug(f"accumulate_signals called with selected_subcarriers type: {type(selected_subcarriers)}")
        logger.debug(f"ue_positions: {len(ue_positions)} positions")
        
        # Additional debugging for dictionary format
        if isinstance(selected_subcarriers, dict):
            logger.debug(f"Dictionary keys count: {len(selected_subcarriers.keys())}")
        
        subcarrier_indices = self._normalize_subcarrier_input(selected_subcarriers, ue_positions)
        
        if self.prism_network is None:
            # Fallback: iterate through all directions if no network is available
            return self._accumulate_signals_fallback(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding
            )
        
        try:
            # Use AntennaNetwork to get directional importance based on antenna embedding C
            with torch.no_grad():
                # Get directional importance matrix from AntennaNetwork
                directional_importance = self.prism_network.antenna_network(antenna_embedding.unsqueeze(0))
                
                # Get top-K directions for efficient sampling
                top_k_directions, top_k_importance = self.prism_network.antenna_network.get_top_k_directions(
                    directional_importance, k=min(32, self.azimuth_divisions * self.elevation_divisions // 4)
                )
                
                # Extract direction indices for the first batch element
                selected_directions = top_k_directions[0]  # Shape: (k, 2)
                
            # Only trace rays for MLP-selected directions
            for i in range(selected_directions.shape[0]):
                phi_idx = selected_directions[i, 0].item()
                theta_idx = selected_directions[i, 1].item()
                direction = (phi_idx, theta_idx)
                
                # Trace ray for this selected direction with antenna embedding
                ray_results = self.trace_ray(
                    base_station_pos, direction, ue_positions, selected_subcarriers, antenna_embedding
                )
                
                # Accumulate signals for each virtual link
                for (ue_pos, subcarrier), signal_strength in ray_results.items():
                    if (ue_pos, subcarrier) not in accumulated_signals:
                        accumulated_signals[(ue_pos, subcarrier)] = 0.0
                    accumulated_signals[(ue_pos, subcarrier)] += signal_strength
            
            return accumulated_signals
            
        except Exception as e:
            logger.warning(f"MLP-based direction sampling failed: {e}. Using fallback method.")
            return self._accumulate_signals_fallback(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding
            )
    
    def _accumulate_signals_fallback(self, 
                                   base_station_pos: torch.Tensor,
                                   ue_positions: List[torch.Tensor],
                                   selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                                   antenna_embedding: torch.Tensor) -> Dict:
        """
        Fallback method: accumulate signals from all directions (traditional approach).
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information from training interface
                           - Dict: Mapping UE to subcarrier indices
                           - torch.Tensor: Tensor of subcarrier indices
                           - List[int]: List of subcarrier indices
                           Note: This MUST be provided by the calling code
            antenna_embedding: Base station's antenna embedding parameter C
        
        Returns:
            Accumulated signal strength matrix for all virtual links
        """
        accumulated_signals = {}
        
        # Debug logging
        logger.debug(f"_accumulate_signals_fallback called with selected_subcarriers type: {type(selected_subcarriers)}")
        
        # Iterate through all A × B directions
        for phi in range(self.azimuth_divisions):
            for theta in range(self.elevation_divisions):
                direction = (phi, theta)
                
                # Trace ray for this direction with antenna embedding
                ray_results = self.trace_ray(
                    base_station_pos, direction, ue_positions, selected_subcarriers, antenna_embedding
                )
                
                # Debug: print ray results for first few directions
                if phi == 0 and theta == 0:
                    logger.debug(f"First direction results: {ray_results}")
                    logger.debug(f"First direction keys: {list(ray_results.keys())}")
                
                # Accumulate signals for each virtual link
                for (ue_pos, subcarrier), signal_strength in ray_results.items():
                    # Ensure consistent key format
                    if isinstance(ue_pos, torch.Tensor):
                        ue_key = tuple(ue_pos.tolist())
                    else:
                        ue_key = ue_pos
                    
                    key = (ue_key, subcarrier)
                    
                    if key not in accumulated_signals:
                        accumulated_signals[key] = 0.0
                    accumulated_signals[key] += signal_strength
        
        logger.debug(f"Final accumulated signals: {len(accumulated_signals)} results")
        logger.debug(f"Final keys: {list(accumulated_signals.keys())}")
        
        return accumulated_signals
    
    def adaptive_ray_tracing(self, 
                           base_station_pos: torch.Tensor,
                           antenna_embedding: torch.Tensor,
                           ue_positions: List[torch.Tensor],
                           selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                           top_k: int = 32) -> Dict:
        """
        Perform adaptive ray tracing using built-in AntennaNetwork for direction selection.
        
        This method uses the integrated AntennaNetwork to select important directions
        based on antenna embedding C, providing better integration with the neural network.
        
        Args:
            base_station_pos: Base station position
            antenna_embedding: Base station's antenna embedding parameter C
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information from training interface
                           - Dict: Mapping UE to subcarrier indices
                           - torch.Tensor: Tensor of subcarrier indices
                           - List[int]: List of subcarrier indices
                           Note: This MUST be provided by the calling code
            top_k: Number of top directions to select
        
        Returns:
            Accumulated signal strength for selected directions only
        """
        # Use the main accumulate_signals method which already implements MLP-based sampling
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
        
        This method implements the pyramid ray tracing technique from the design document:
        1. Spatial subdivision into pyramidal regions
        2. Hierarchical sampling strategy
        3. Monte Carlo integration within truncated cone regions
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information from training interface
                           - Dict: Mapping UE to subcarrier indices
                           - torch.Tensor: Tensor of subcarrier indices
                           - List[int]: List of subcarrier indices
                           Note: This MUST be provided by the calling code
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
                    
                    # Apply Monte Carlo integration within the pyramidal region
                    ray_results = self._monte_carlo_pyramid_integration(
                        base_station_pos, direction, ue_positions, 
                        selected_subcarriers, antenna_embedding, level_factor
                    )
                    
                    # Accumulate signals with level weighting
                    level_weight = 1.0 / (level + 1)  # Higher levels get lower weight
                    for (ue_pos, subcarrier), signal_strength in ray_results.items():
                        if (ue_pos, subcarrier) not in accumulated_signals:
                            accumulated_signals[(ue_pos, subcarrier)] = 0.0
                        accumulated_signals[(ue_pos, subcarrier)] += signal_strength * level_weight
        
        return accumulated_signals
    
    def _monte_carlo_pyramid_integration(self,
                                       base_station_pos: torch.Tensor,
                                       center_direction: Tuple[int, int],
                                       ue_positions: List[torch.Tensor],
                                       selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                                       antenna_embedding: torch.Tensor,
                                       pyramid_size: int,
                                       num_samples: int = 4) -> Dict:
        """
        Perform Monte Carlo integration within a pyramidal region.
        
        Args:
            base_station_pos: Base station position
            center_direction: Center direction of the pyramid
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information from training interface
                           - Dict: Mapping UE to subcarrier indices
                           - torch.Tensor: Tensor of subcarrier indices
                           - List[int]: List of subcarrier indices
                           Note: This MUST be provided by the calling code
            antenna_embedding: Antenna embedding parameter
            pyramid_size: Size of the pyramidal region
            num_samples: Number of Monte Carlo samples
        
        Returns:
            Integrated signal strength for the pyramidal region
        """
        phi_center, theta_center = center_direction
        results = {}
        
        # Generate random samples within the pyramidal region
        for _ in range(num_samples):
            # Random offset within the pyramid
            phi_offset = torch.randint(-pyramid_size//2, pyramid_size//2 + 1, (1,)).item()
            theta_offset = torch.randint(-pyramid_size//2, pyramid_size//2 + 1, (1,)).item()
            
            # Clamp to valid ranges
            phi_sample = max(0, min(self.azimuth_divisions - 1, phi_center + phi_offset))
            theta_sample = max(0, min(self.elevation_divisions - 1, theta_center + theta_offset))
            
            sample_direction = (phi_sample, theta_sample)
            
            # Trace ray for this sample direction
            sample_results = self.trace_ray(
                base_station_pos, sample_direction, ue_positions,
                selected_subcarriers, antenna_embedding
            )
            
            # Accumulate Monte Carlo samples
            for (ue_pos, subcarrier), signal_strength in sample_results.items():
                if (ue_pos, subcarrier) not in results:
                    results[(ue_pos, subcarrier)] = 0.0
                results[(ue_pos, subcarrier)] += signal_strength / num_samples
        
        return results
    
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
    
    # NOTE: This ray tracer does NOT select subcarriers internally.
    # All subcarrier selection must be provided by the calling code (typically PrismTrainingInterface)
    # to ensure consistency across the training pipeline and proper loss computation.
    #
    # The ray tracer expects:
    # - selected_subcarriers: Dictionary, tensor, or list specifying which subcarriers to process
    # - No internal subcarrier selection logic
    # - Full control by the training interface over which subcarriers are used
    #
    # This design ensures that the training interface has full control over which subcarriers
    # are used for loss computation, preventing any mismatch between ray tracing and loss calculation.
