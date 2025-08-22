"""
CUDA-Accelerated Discrete Electromagnetic Ray Tracing System for Prism

This module implements a high-performance CUDA version of the ray tracing system
with automatic device detection and fallback to CPU implementation.
"""

import torch
import logging
import math
import time
from typing import Dict, List, Tuple, Optional
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
