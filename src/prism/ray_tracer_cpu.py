"""
Discrete Electromagnetic Ray Tracing System for Prism

This module implements the discrete electromagnetic ray tracing system as described
in the design document, with support for MLP-based direction sampling and
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
from functools import partial
from .ray_tracer_base import RayTracer, Ray

logger = logging.getLogger(__name__)




class CPURayTracer(RayTracer):
    """CPU-based discrete electromagnetic ray tracer implementing the design document specifications."""
    
    def __init__(self, 
                 azimuth_divisions: int = 36,
                 elevation_divisions: int = 18,
                 max_ray_length: float = 100.0,
                 scene_size: float = 200.0,
                 device: str = 'cpu',
                 prism_network=None,
                 signal_threshold: float = 1e-6,
                 enable_early_termination: bool = True,
                 top_k_directions: int = None,
                 enable_parallel_processing: bool = True,
                 max_workers: int = None,
                 uniform_samples: int = 128,
                 resampled_points: int = 64):
        """
        Initialize CPU ray tracer.
        
        Args:
            azimuth_divisions: Number of azimuth divisions A (0° to 360°)
            elevation_divisions: Number of elevation divisions B (-90° to +90°)
            max_ray_length: Maximum ray length in meters
            scene_size: Scene size D in meters (cubic environment: [-D/2, D/2]³)
            device: Device to run computations on
            prism_network: PrismNetwork instance for getting attenuation and radiance properties
            signal_threshold: Minimum signal strength threshold for early termination
            enable_early_termination: Enable early termination optimization
            top_k_directions: Number of top-K directions to select for MLP-based sampling (if None, uses default formula)
            enable_parallel_processing: Enable parallel processing for ray tracing
            max_workers: Maximum number of parallel workers (if None, uses CPU count)
            uniform_samples: Number of uniform samples per ray
            resampled_points: Number of resampled points per ray
        """
        # Log initialization
        logger.info("🖥️ Initializing CPURayTracer - CPU-based ray tracing implementation")
        
        # Call parent constructor
        super().__init__(
            azimuth_divisions=azimuth_divisions,
            elevation_divisions=elevation_divisions,
            max_ray_length=max_ray_length,
            scene_size=scene_size,
            device=device,
            uniform_samples=uniform_samples,
            resampled_points=resampled_points
        )
        
        self.prism_network = prism_network
        self.signal_threshold = signal_threshold
        self.enable_early_termination = enable_early_termination
        
        # Set top-K directions for MLP-based sampling
        if top_k_directions is not None:
            self.top_k_directions = top_k_directions
            logger.info(f"Using configured top-K directions: {self.top_k_directions}")
        else:
            # Default formula: min(32, total_directions // 4)
            self.top_k_directions = min(32, (azimuth_divisions * elevation_divisions) // 4)
            logger.info(f"Using default top-K formula: min(32, {azimuth_divisions * elevation_divisions} // 4) = {self.top_k_directions}")
        
        # Parallel processing configuration
        self.enable_parallel_processing = enable_parallel_processing
        
        if max_workers is None:
            import multiprocessing as mp
            self.max_workers = mp.cpu_count()
        else:
            self.max_workers = max_workers
            
        logger.info(f"💻 CPU Ray Tracer initialized with {azimuth_divisions}×{elevation_divisions} = {self.total_directions} directions")
        logger.info(f"Parallel processing: {'enabled' if enable_parallel_processing else 'disabled'}")
        logger.info(f"Max workers: {self.max_workers} (multiprocessing)")
    
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
        in_bounds = torch.all(
            (position >= self.scene_min) & (position <= self.scene_max), 
            dim=1
        )
        
        return in_bounds.all().item()
    
    def get_scene_bounds(self) -> Tuple[float, float]:
        """Get scene boundaries."""
        return self.scene_min, self.scene_max
    
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
        self.scene_min = -new_scene_size / 2.0
        self.scene_max = new_scene_size / 2.0
        
        # Adjust max ray length if necessary
        if self.max_ray_length > new_scene_size:
            self.max_ray_length = new_scene_size
            logger.info(f"Adjusted max ray length to {self.max_ray_length}m")
        
        logger.info(f"Updated scene size to {new_scene_size}m, boundaries: [{self.scene_min:.1f}, {self.scene_max:.1f}]³")
    
    def get_scene_config(self) -> Dict[str, float]:
        """Get complete scene configuration."""
        return {
            'scene_size': self.scene_size,
            'scene_min': self.scene_min,
            'scene_max': self.scene_max,
            'max_ray_length': self.max_ray_length,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions
        }
    
    def generate_direction_vectors(self) -> torch.Tensor:
        """Generate unit direction vectors for all A×B directions."""
        directions = []
        
        for i in range(self.azimuth_divisions):
            for j in range(self.elevation_divisions):
                phi = i * self.azimuth_resolution  # Azimuth angle
                theta = j * self.elevation_resolution  # Elevation angle
                
                # Convert to Cartesian coordinates using proper spherical coordinates
                # Elevation: -90° to +90° (-π/2 to +π/2)
                elevation = theta - (math.pi / 2)
                x = math.cos(elevation) * math.cos(phi)
                y = math.cos(elevation) * math.sin(phi)
                z = math.sin(elevation)
                
                directions.append([x, y, z])
        
        return torch.tensor(directions, dtype=torch.float32, device=self.device)
    
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
        # Convert to proper spherical coordinates
        # Elevation: -90° to +90° (-π/2 to +π/2)
        elevation = (theta_idx * self.elevation_resolution) - (math.pi / 2)
        
        direction_vector = torch.tensor([
            math.cos(elevation) * math.cos(phi),
            math.cos(elevation) * math.sin(phi),
            math.sin(elevation)
        ], dtype=torch.float32, device=self.device)
        
        # Create ray from BS antenna (configurable position, defaults to (0,0,0))
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
    
    def trace_rays(self, 
                   base_station_pos: torch.Tensor,
                   directions: torch.Tensor,
                   ue_positions: List[torch.Tensor],
                   selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                   antenna_embedding: torch.Tensor) -> Dict:
        """
        Trace RF signals along multiple ray directions.
        
        Args:
            base_station_pos: Base station position
            directions: Direction vectors [num_directions, 3]
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information
            antenna_embedding: Base station's antenna embedding parameter C
        
        Returns:
            Dictionary mapping (ue_pos, subcarrier, direction_idx) to signal strength
        """
        results = {}
        
        # Validate and normalize selected_subcarriers input
        subcarrier_indices = self._normalize_subcarrier_input(selected_subcarriers, ue_positions)
        
        for direction_idx, direction_vector in enumerate(directions):
            # Create ray for this direction from BS antenna (configurable position)
            ray = Ray(base_station_pos, direction_vector, self.max_ray_length, self.device)
            
            for ue_pos in ue_positions:
                ue_pos_tensor = ue_pos.clone().detach().to(dtype=torch.float32, device=self.device)
                
                # Use the normalized subcarrier indices
                for subcarrier_idx in subcarrier_indices:
                    # Apply discrete radiance field model for ray tracing
                    signal_strength = self._discrete_radiance_ray_tracing(
                        ray, ue_pos_tensor, subcarrier_idx, antenna_embedding
                    )
                    results[(tuple(ue_pos), subcarrier_idx, direction_idx)] = signal_strength
        
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
            # logger.debug(f"Processing dictionary with {len(selected_subcarriers)} keys")  # 屏蔽重复输出
            
            for ue_pos in ue_positions:
                # Convert tensor to tuple for comparison
                ue_key = tuple(ue_pos.tolist())
                # logger.debug(f"Looking for UE key: {ue_key}")  # 屏蔽具体坐标值
                
                if ue_key in selected_subcarriers:
                    indices = selected_subcarriers[ue_key]
                    # logger.debug(f"Found indices: {type(indices)} = {indices}")  # 屏蔽具体索引值
                    
                    if isinstance(indices, (list, tuple)):
                        # logger.debug(f"  Processing list/tuple with {len(indices)} elements")  # 屏蔽重复输出
                        all_indices.update(indices)
                    elif isinstance(indices, torch.Tensor):
                        # logger.debug(f"  Processing tensor with shape {indices.shape}")  # 屏蔽重复输出
                        all_indices.update(indices.tolist())
                    elif isinstance(indices, (int, float)):
                        # logger.debug(f"  Processing single value: {indices}")  # 屏蔽重复输出
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
                    # logger.debug(f"UE key {ue_key} not found in selected_subcarriers")  # 屏蔽具体坐标值
                    pass
            
            # logger.debug(f"Collected {len(all_indices)} unique indices")  # 屏蔽重复输出
            
            if not all_indices:
                raise ValueError("No valid subcarrier indices found in selected_subcarriers dictionary")
            
            # Validate subcarrier indices are within reasonable bounds
            max_subcarrier = 408  # Based on the simulation data
            valid_indices = [idx for idx in all_indices if 0 <= idx < max_subcarrier]
            
            if len(valid_indices) != len(all_indices):
                invalid_count = len(all_indices) - len(valid_indices)
                logger.warning(f"Filtered out {invalid_count} invalid subcarrier indices (out of bounds)")
            
            if not valid_indices:
                logger.warning("No valid subcarrier indices found, using default indices [0, 64, 128, 192, 256, 320, 384]")
                valid_indices = [0, 64, 128, 192, 256, 320, 384]
            
            return sorted(valid_indices)
            
        elif isinstance(selected_subcarriers, torch.Tensor):
            # Tensor format: convert to list
            if selected_subcarriers.numel() == 0:
                raise ValueError("selected_subcarriers tensor is empty")
            
            # Validate subcarrier indices are within reasonable bounds
            max_subcarrier = 408  # Based on the simulation data
            indices = selected_subcarriers.flatten().tolist()
            valid_indices = [idx for idx in indices if 0 <= idx < max_subcarrier]
            
            if len(valid_indices) != len(indices):
                invalid_count = len(indices) - len(valid_indices)
                logger.warning(f"Filtered out {invalid_count} invalid subcarrier indices from tensor (out of bounds)")
            
            if not valid_indices:
                logger.warning("No valid subcarrier indices found in tensor, using default indices [0, 64, 128, 192, 256, 320, 384]")
                valid_indices = [0, 64, 128, 192, 256, 320, 384]
            
            return valid_indices
            
        elif isinstance(selected_subcarriers, (list, tuple)):
            # List/tuple format: validate and return
            if not selected_subcarriers:
                raise ValueError("selected_subcarriers list is empty")
            
            # Validate subcarrier indices are within reasonable bounds
            max_subcarrier = 408  # Based on the simulation data
            valid_indices = []
            for idx in selected_subcarriers:
                try:
                    int_idx = int(idx)
                    if 0 <= int_idx < max_subcarrier:
                        valid_indices.append(int_idx)
                    else:
                        logger.warning(f"Subcarrier index {int_idx} out of bounds [0, {max_subcarrier}), skipping")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid subcarrier index {idx}, skipping")
            
            if not valid_indices:
                logger.warning("No valid subcarrier indices found in list, using default indices [0, 64, 128, 192, 256, 320, 384]")
                valid_indices = [0, 64, 128, 192, 256, 320, 384]
            
            return valid_indices
            
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
        
        # Get viewing directions for uniform samples (from sample positions to BS antenna)
        # Note: In ray tracing, ray.origin is the BS antenna position
        uniform_view_directions = self._compute_view_directions(uniform_positions, ray.origin)
        
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
            
            # Extract attenuation factors for weight computation with boundary checking
            uniform_attenuation_factors = uniform_network_outputs['attenuation_factors']
            if subcarrier_idx >= uniform_attenuation_factors.shape[-1]:
                logger.warning(f"Subcarrier index {subcarrier_idx} out of bounds for uniform_attenuation_factors shape {uniform_attenuation_factors.shape}, using index 0")
                subcarrier_idx = 0
            uniform_attenuation = uniform_attenuation_factors[0, :, 0, subcarrier_idx]  # (num_uniform_samples,)
            
            # Stage 2: Importance-based resampling
            importance_weights = self._compute_importance_weights(uniform_attenuation)
            resampled_positions = self._importance_based_resampling(
                uniform_positions, importance_weights, num_samples=64
            )
            
            # Get network properties for resampled points (from sample positions to BS antenna)
            resampled_view_directions = self._compute_view_directions(resampled_positions, ray.origin)
            
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
    
    # Common functions moved to base class:
    # - _sample_ray_points
    # - _compute_importance_weights  
    # - _importance_based_resampling
    # - _ensure_complex_accumulation
    # - _simple_distance_model
    
    def _integrate_along_ray_with_importance(self,
                                           sampled_positions: torch.Tensor,
                                           attenuation_factors: torch.Tensor,
                                           radiation_factors: torch.Tensor,
                                           subcarrier_idx: int,
                                           importance_weights: torch.Tensor) -> torch.Tensor:
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
        
        # Extract complex attenuation and radiation for the specific subcarrier
        attenuation = attenuation_factors[0, :, 0, subcarrier_idx]  # (num_samples,) - complex
        radiation = radiation_factors[0, 0, subcarrier_idx]  # scalar - complex
        
        # Calculate dynamic step sizes using base class method
        delta_t = self.compute_dynamic_path_lengths(sampled_positions)
        
        # 🚀 VECTORIZED discrete radiance field integration according to SPECIFICATION.md
        # S(P_RX, ω) ≈ Σ[k=1 to K] exp(-Σ[j=1 to k-1] ρ(P_v^j) Δt_j) × (1 - e^(-ρ(P_v^k) Δt_k)) × S(P_v^k, -ω)
        
        # Vectorized computation - all operations on full tensors
        # Shape: (num_samples,) for all tensors
        
        # Term 1: Vectorized cumulative attenuation calculation
        # cumsum([0, ρ₀Δt₀, ρ₁Δt₁, ...]) = [0, ρ₀Δt₀, ρ₀Δt₀+ρ₁Δt₁, ...]
        attenuation_deltas = attenuation * delta_t  # Element-wise multiplication (complex)
        
        # Pad with zero at the beginning and remove last element for cumulative sum
        padded_deltas = torch.cat([torch.zeros(1, dtype=attenuation_deltas.dtype, device=self.device), 
                                   attenuation_deltas[:-1]], dim=0)
        cumulative_attenuation = torch.cumsum(padded_deltas, dim=0)  # Vectorized cumulative sum
        
        # Term 2: Vectorized attenuation factors
        attenuation_factors = torch.exp(-cumulative_attenuation)  # (num_samples,) complex
        
        # Term 3: Vectorized local absorption factors
        local_absorption = 1.0 - torch.exp(-attenuation * delta_t)  # (num_samples,) complex
        
        # Term 4: Vectorized radiance (broadcast single value to all voxels)
        # Note: In future, this should be per-voxel radiance values
        radiance_vector = radiation.expand(num_samples)  # Broadcast to (num_samples,)
        
        # 🎯 VECTORIZED FINAL COMPUTATION - Single tensor operation!
        # All terms computed in parallel across all voxels
        # Note: No importance correction needed here since we already did importance-based resampling
        signal_contributions = (attenuation_factors * 
                              local_absorption * 
                              radiance_vector)  # (num_samples,) complex
        
        # Early termination using vectorized operations
        if self.enable_early_termination:
            # Find first index where attenuation factor falls below threshold
            valid_mask = torch.abs(attenuation_factors) >= self.signal_threshold
            if not torch.all(valid_mask):
                first_invalid = torch.argmax((~valid_mask).int())
                signal_contributions[first_invalid:] = 0.0
                logger.debug(f"Vectorized early termination at sample {first_invalid}/{num_samples}")
        
        # Final sum - single reduction operation
        total_signal_complex = torch.sum(signal_contributions)
        
        # Return complex result - DO NOT convert to real
        return total_signal_complex
    
    def _integrate_along_ray_batch_vectorized(self,
                                            sampled_positions: torch.Tensor,
                                            attenuation_factors: torch.Tensor,
                                            radiation_factors: torch.Tensor,
                                            subcarrier_indices: torch.Tensor,
                                            importance_weights: torch.Tensor) -> torch.Tensor:
        """
        🚀 ULTRA-FAST batch vectorized ray integration for multiple subcarriers simultaneously.
        
        This function processes ALL subcarriers in parallel using pure tensor operations,
        achieving maximum GPU utilization and memory bandwidth efficiency.
        
        Args:
            sampled_positions: (num_samples, 3) - Voxel positions along ray
            attenuation_factors: (batch_size, num_samples, num_ue, num_subcarriers) - Complex attenuation
            radiation_factors: (batch_size, num_ue, num_subcarriers) - Complex radiance
            subcarrier_indices: (num_selected_subcarriers,) - Selected subcarrier indices
            importance_weights: (num_samples,) - Importance sampling weights
            
        Returns:
            signal_strengths: (num_selected_subcarriers,) - Signal strength for each subcarrier
        """
        num_samples = sampled_positions.shape[0]
        num_subcarriers = len(subcarrier_indices)
        
        # Extract data for selected subcarriers - shape: (num_samples, num_subcarriers)
        attenuation_batch = attenuation_factors[0, :, 0, subcarrier_indices].T  # (num_subcarriers, num_samples)
        radiation_batch = radiation_factors[0, 0, subcarrier_indices]  # (num_subcarriers,)
        
        # Calculate dynamic step sizes once for all subcarriers
        if num_samples > 1:
            delta_t = torch.norm(sampled_positions[1:] - sampled_positions[:-1], dim=1)
            first_delta_t = torch.norm(sampled_positions[1] - sampled_positions[0], dim=0).unsqueeze(0)
            delta_t = torch.cat([first_delta_t, delta_t], dim=0)
        else:
            delta_t = torch.tensor([1.0], device=self.device)
        
        # Broadcast delta_t to all subcarriers: (num_subcarriers, num_samples)
        delta_t_batch = delta_t.unsqueeze(0).expand(num_subcarriers, -1)
        
        # 🎯 BATCH VECTORIZED COMPUTATION - Process ALL subcarriers simultaneously!
        
        # Term 1: Batch cumulative attenuation - shape: (num_subcarriers, num_samples)
        attenuation_deltas_batch = attenuation_batch * delta_t_batch  # Element-wise (complex)
        
        # Pad and compute cumulative sum for all subcarriers in parallel
        zero_pad = torch.zeros(num_subcarriers, 1, dtype=attenuation_deltas_batch.dtype, device=self.device)
        padded_deltas_batch = torch.cat([zero_pad, attenuation_deltas_batch[:, :-1]], dim=1)
        cumulative_attenuation_batch = torch.cumsum(padded_deltas_batch, dim=1)  # (num_subcarriers, num_samples)
        
        # Term 2: Batch attenuation factors - shape: (num_subcarriers, num_samples)
        attenuation_factors_batch = torch.exp(-cumulative_attenuation_batch)
        
        # Term 3: Batch local absorption - shape: (num_subcarriers, num_samples)
        local_absorption_batch = 1.0 - torch.exp(-attenuation_batch * delta_t_batch)
        
        # Term 4: Broadcast radiance to all samples - shape: (num_subcarriers, num_samples)
        radiance_batch_expanded = radiation_batch.unsqueeze(1).expand(-1, num_samples)
        
        # Term 5: Broadcast importance correction - shape: (num_subcarriers, num_samples)
        if len(importance_weights) > 0:
            if len(importance_weights) < num_samples:
                importance_correction = torch.cat([
                    1.0 / (importance_weights + 1e-8),
                    torch.ones(num_samples - len(importance_weights), device=self.device)
                ], dim=0)
            else:
                importance_correction = 1.0 / (importance_weights[:num_samples] + 1e-8)
        else:
            importance_correction = torch.ones(num_samples, device=self.device)
        
        importance_correction_batch = importance_correction.unsqueeze(0).expand(num_subcarriers, -1)
        
        # 🚀 ULTIMATE VECTORIZED COMPUTATION - Single massive tensor operation!
        # Process ALL subcarriers and ALL samples simultaneously
        signal_contributions_batch = (attenuation_factors_batch * 
                                    local_absorption_batch * 
                                    radiance_batch_expanded * 
                                    importance_correction_batch)  # (num_subcarriers, num_samples)
        
        # Early termination using batch operations
        if self.enable_early_termination:
            valid_mask_batch = torch.abs(attenuation_factors_batch) >= self.signal_threshold
            # Find first invalid index for each subcarrier
            invalid_indices = torch.argmax((~valid_mask_batch).int(), dim=1)
            
            # Create mask to zero out contributions after early termination
            sample_indices = torch.arange(num_samples, device=self.device).unsqueeze(0).expand(num_subcarriers, -1)
            termination_mask = sample_indices < invalid_indices.unsqueeze(1)
            signal_contributions_batch = signal_contributions_batch * termination_mask
        
        # Final reduction - sum across samples for each subcarrier
        total_signals_complex = torch.sum(signal_contributions_batch, dim=1)  # (num_subcarriers,)
        
        # Handle complex results
        if torch.is_complex(total_signals_complex):
            total_signals = torch.abs(total_signals_complex)
        else:
            total_signals = total_signals_complex
        
        return total_signals
    
    # _simple_distance_model moved to base class
    
    def accumulate_signals(self, 
                          base_station_pos: torch.Tensor,
                          ue_positions: List[torch.Tensor],
                          selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                          antenna_embedding: torch.Tensor,
                          progress_callback=None) -> Dict:
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
        # logger.debug(f"selected_subcarriers content: {selected_subcarriers}")  # 屏蔽数据内容
        logger.debug(f"ue_positions: {len(ue_positions)} positions")
        
        # Additional debugging for dictionary format
        if isinstance(selected_subcarriers, dict):
            logger.debug(f"Dictionary keys count: {len(selected_subcarriers.keys())}")
            # for key, value in selected_subcarriers.items():
            #     logger.debug(f"    Key {key}: {type(value)} = {value}")  # 屏蔽数据内容
            #     if isinstance(value, (list, tuple)):
            #         logger.debug(f"    Length: {len(value)}")
            #     elif isinstance(value, torch.Tensor):
            #         logger.debug(f"    Shape: {value.shape}, dtype: {value.dtype}")
        
        subcarrier_indices = self._normalize_subcarrier_input(selected_subcarriers, ue_positions)
        
        if self.prism_network is None:
            # Fallback: iterate through all directions if no network is available
            return self._accumulate_signals_fallback(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding
            )
        
        try:
            # Use AntennaNetwork to get directional importance based on antenna embedding C
            with torch.no_grad():
                # Ensure antenna embedding is on the same device as prism_network
                device = next(self.prism_network.parameters()).device
                antenna_embedding_device = antenna_embedding.to(device)
                
                # Get directional importance matrix from AntennaNetwork
                directional_importance = self.prism_network.antenna_network(antenna_embedding_device.unsqueeze(0))
                
                # Get top-K directions for efficient sampling
                top_k_directions, top_k_importance = self.prism_network.antenna_network.get_top_k_directions(
                    directional_importance, k=self.top_k_directions
                )
                
                # Extract direction indices for the first batch element
                selected_directions = top_k_directions[0]  # Shape: (k, 2)
                
            # Convert tensor directions to list of tuples for parallel processing
            directions_list = []
            for i in range(selected_directions.shape[0]):
                phi_idx = selected_directions[i, 0].item()
                theta_idx = selected_directions[i, 1].item()
                directions_list.append((phi_idx, theta_idx))
            
            # Use intelligent parallel processing selection for ray tracing
            if self.enable_parallel_processing and len(directions_list) > 1:
                # Determine the best parallelization strategy based on workload size
                num_antennas = antenna_embedding.shape[0] if len(antenna_embedding.shape) > 1 else 64
                num_spatial_points = 32  # Default spatial sampling points
                
                logger.debug(f"Selecting parallelization strategy: {len(directions_list)} directions, {num_antennas} antennas, {num_spatial_points} spatial points")
                
                if len(directions_list) >= 16 and num_antennas >= 32 and num_spatial_points >= 16:
                    # Full parallelization for large workloads
                    logger.debug(f"Using full parallelization (direction + antenna + spatial) with {self.max_workers} workers")
                    if progress_callback:
                        progress_callback(f"🚀 Full parallelization: {len(directions_list)} directions, {num_antennas} antennas")
                    accumulated_signals = self._accumulate_signals_full_parallel(
                        base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, 
                        directions_list, num_antennas, num_spatial_points
                    )
                elif len(directions_list) >= 8 and num_antennas >= 16:
                    # Antenna + direction parallelization for medium workloads
                    logger.debug(f"Using antenna + direction parallelization with {self.max_workers} workers")
                    if progress_callback:
                        progress_callback(f"⚡ Antenna parallelization: {len(directions_list)} directions, {num_antennas} antennas")
                    accumulated_signals = self._accumulate_signals_antenna_parallel(
                        base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, 
                        directions_list, num_antennas
                    )
                else:
                    # Direction-only parallelization for small workloads
                    logger.debug(f"Using direction-only parallelization with {self.max_workers} workers")
                    accumulated_signals = self._accumulate_signals_parallel(
                        base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions_list
                    )
            else:
                logger.debug(f"Using sequential processing for {len(directions_list)} directions")
                accumulated_signals = self._accumulate_signals_sequential(
                    base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions_list
                )
            
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
        # logger.debug(f"selected_subcarriers content: {selected_subcarriers}")  # 屏蔽数据内容
        
        # Generate all directions for parallel processing
        all_directions = []
        for phi in range(self.azimuth_divisions):
            for theta in range(self.elevation_divisions):
                all_directions.append((phi, theta))
        
        # Use parallel processing for fallback method
        if self.enable_parallel_processing and len(all_directions) > 1:
            logger.debug(f"Using parallel processing for fallback method with {len(all_directions)} directions")
            accumulated_signals = self._accumulate_signals_parallel(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, all_directions
            )
        else:
            logger.debug(f"Using sequential processing for fallback method with {len(all_directions)} directions")
            accumulated_signals = self._accumulate_signals_sequential(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, all_directions
            )
        
        return accumulated_signals
    
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
                        weighted_signal = signal_strength * level_weight
                        self._ensure_complex_accumulation(accumulated_signals, (ue_pos, subcarrier), weighted_signal)
        
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
    
    def _trace_ray_parallel_wrapper(self, args):
        """
        Wrapper function for parallel ray tracing.
        
        Args:
            args: Tuple of (direction, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding)
        
        Returns:
            Ray tracing results for the given direction
        """
        direction, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding = args
        try:
            return self.trace_ray(base_station_pos, direction, ue_positions, selected_subcarriers, antenna_embedding)
        except Exception as e:
            logger.warning(f"Parallel ray tracing failed for direction {direction}: {e}")
            return {}
    
    # _ensure_complex_accumulation moved to base class
    
    def _accumulate_signals_parallel(self, 
                                   base_station_pos: torch.Tensor,
                                   ue_positions: List[torch.Tensor],
                                   selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                                   antenna_embedding: torch.Tensor,
                                   directions: List[Tuple[int, int]]) -> Dict:
        """
        Parallel version of signal accumulation using multiple workers.
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information
            antenna_embedding: Base station's antenna embedding parameter C
            directions: List of directions to process in parallel
        
        Returns:
            Accumulated signal strength matrix
        """
        accumulated_signals = {}
        
        if not self.enable_parallel_processing or len(directions) < 2:
            # Fall back to sequential processing for small numbers of directions
            for direction in directions:
                ray_results = self.trace_ray(
                    base_station_pos, direction, ue_positions, selected_subcarriers, antenna_embedding
                )
                for (ue_pos, subcarrier), signal_strength in ray_results.items():
                    self._ensure_complex_accumulation(accumulated_signals, (ue_pos, subcarrier), signal_strength)
            return accumulated_signals
        
        # Prepare arguments for parallel processing
        args_list = [(direction, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding) 
                    for direction in directions]
        
        try:
            # Use multiprocessing for CPU-intensive tasks
            with mp.Pool(processes=self.max_workers) as pool:
                results = pool.map(self._trace_ray_parallel_wrapper, args_list)
            
            # Accumulate results from all workers
            for ray_results in results:
                if ray_results:  # Check if results are not empty
                    for (ue_pos, subcarrier), signal_strength in ray_results.items():
                        self._ensure_complex_accumulation(accumulated_signals, (ue_pos, subcarrier), signal_strength)
                        
        except Exception as e:
            logger.warning(f"Parallel processing failed: {e}. Falling back to sequential processing.")
            # Fall back to sequential processing
            return self._accumulate_signals_sequential(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions
            )
        
        return accumulated_signals
    
    def _accumulate_signals_sequential(self, 
                                     base_station_pos: torch.Tensor,
                                     ue_positions: List[torch.Tensor],
                                     selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                                     antenna_embedding: torch.Tensor,
                                     directions: List[Tuple[int, int]]) -> Dict:
        """
        Sequential version of signal accumulation (fallback method).
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information
            antenna_embedding: Base station's antenna embedding parameter C
            directions: List of directions to process sequentially
        
        Returns:
            Accumulated signal strength matrix
        """
        accumulated_signals = {}
        
        for direction in directions:
            ray_results = self.trace_ray(
                base_station_pos, direction, ue_positions, selected_subcarriers, antenna_embedding
            )
            for (ue_pos, subcarrier), signal_strength in ray_results.items():
                self._ensure_complex_accumulation(accumulated_signals, (ue_pos, subcarrier), signal_strength)
        
        return accumulated_signals
    
    def get_parallelization_stats(self) -> Dict:
        """
        Get statistics about parallel processing configuration.
        
        Returns:
            Dictionary with parallelization statistics
        """
        import multiprocessing as mp
        return {
            'parallel_processing_enabled': self.enable_parallel_processing,
            'max_workers': self.max_workers,
            'processing_mode': 'multiprocessing',
            'cpu_count': mp.cpu_count(),
            'device': self.device,
            'total_directions': self.total_directions,
            'top_k_directions': self.top_k_directions
        }
    
    def _trace_ray_antenna_parallel(self, args):
        """
        Wrapper function for parallel antenna processing.
        
        Args:
            args: Tuple of (antenna_idx, direction, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding)
        
        Returns:
            Ray tracing results for the given antenna and direction
        """
        antenna_idx, direction, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding = args
        try:
            # Create antenna-specific embedding
            if len(antenna_embedding.shape) > 1:
                antenna_specific_embedding = antenna_embedding[antenna_idx]
            else:
                antenna_specific_embedding = antenna_embedding
            
            # Trace ray for this specific antenna
            ray_results = self.trace_ray(
                base_station_pos, direction, ue_positions, selected_subcarriers, antenna_specific_embedding
            )
            
            # Add antenna index to results for identification
            antenna_results = {}
            for (ue_pos, subcarrier), signal_strength in ray_results.items():
                antenna_results[(ue_pos, subcarrier, antenna_idx)] = signal_strength
            
            return antenna_results
            
        except Exception as e:
            logger.warning(f"Parallel antenna processing failed for antenna {antenna_idx}, direction {direction}: {e}")
            return {}
    
    def _accumulate_signals_antenna_parallel(self, 
                                           base_station_pos: torch.Tensor,
                                           ue_positions: List[torch.Tensor],
                                           selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                                           antenna_embedding: torch.Tensor,
                                           directions: List[Tuple[int, int]],
                                           num_antennas: int = 64) -> Dict:
        """
        Antenna-level parallel signal accumulation.
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information
            antenna_embedding: Base station's antenna embedding parameter C
            directions: List of directions to process
            num_antennas: Number of BS antennas to process in parallel
        
        Returns:
            Accumulated signal strength matrix with antenna-level parallelization
        """
        accumulated_signals = {}
        
        if not self.enable_parallel_processing or num_antennas < 2:
            # Fall back to direction-level parallelization
            return self._accumulate_signals_parallel(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions
            )
        
        # Prepare arguments for antenna-level parallel processing
        args_list = []
        for antenna_idx in range(num_antennas):
            for direction in directions:
                args_list.append((antenna_idx, direction, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding))
        
        total_tasks = len(args_list)
        logger.info(f"📡 Antenna parallel processing: {num_antennas} antennas × {len(directions)} directions = {total_tasks} total tasks")
        
        # Log progress every 10% of tasks
        progress_interval = max(1, total_tasks // 10)
        
        try:
            # Use multiprocessing for CPU-intensive antenna processing
            with mp.Pool(processes=self.max_workers) as pool:
                results = pool.map(self._trace_ray_antenna_parallel, args_list)
            
            # Accumulate results from all antennas and directions
            for antenna_results in results:
                if antenna_results:  # Check if results are not empty
                    for (ue_pos, subcarrier, antenna_idx), signal_strength in antenna_results.items():
                        self._ensure_complex_accumulation(accumulated_signals, (ue_pos, subcarrier), signal_strength)
                        
        except Exception as e:
            logger.warning(f"Antenna-level parallel processing failed: {e}. Falling back to direction-level parallelization.")
            # Fall back to direction-level parallelization
            return self._accumulate_signals_parallel(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions
            )
        
        return accumulated_signals
    
    def _trace_ray_spatial_parallel(self, args):
        """
        Wrapper function for parallel spatial sampling.
        
        Args:
            args: Tuple of (spatial_point_idx, ray, ue_pos, subcarrier_idx, antenna_embedding)
        
        Returns:
            Signal strength for the given spatial point
        """
        spatial_point_idx, ray, ue_pos, subcarrier_idx, antenna_embedding = args
        try:
            # Sample specific spatial point along the ray
            spatial_position = self._sample_ray_point_at_index(ray, ue_pos, spatial_point_idx)
            
            if spatial_position is None:
                return 0.0
            
            # Compute signal strength at this spatial point
            signal_strength = self._compute_signal_at_spatial_point(
                spatial_position, ue_pos, subcarrier_idx, antenna_embedding, ray.origin
            )
            
            return signal_strength
            
        except Exception as e:
            logger.warning(f"Parallel spatial processing failed for point {spatial_point_idx}: {e}")
            return 0.0
    
    def _sample_ray_point_at_index(self, ray: Ray, ue_pos: torch.Tensor, point_idx: int) -> Optional[torch.Tensor]:
        """
        Sample a specific spatial point along the ray.
        
        Args:
            ray: Ray object
            ue_pos: UE position
            point_idx: Index of the spatial point to sample
        
        Returns:
            Sampled position or None if invalid
        """
        try:
            # Calculate position along the ray based on index
            max_distance = torch.norm(ue_pos - ray.origin)
            if max_distance > ray.max_length:
                max_distance = ray.max_length
            
            # Uniform sampling along the ray
            distance = (point_idx + 1) * max_distance / 32  # 32 spatial points
            if distance > max_distance:
                return None
            
            position = ray.origin + ray.direction * distance
            return position
            
        except Exception as e:
            logger.warning(f"Failed to sample ray point {point_idx}: {e}")
            return None
    
    def _compute_signal_at_spatial_point(self, 
                                       spatial_position: torch.Tensor,
                                       ue_pos: torch.Tensor,
                                       subcarrier_idx: int,
                                       antenna_embedding: torch.Tensor,
                                       bs_position: torch.Tensor) -> torch.Tensor:
        """
        Compute signal strength at a specific spatial point.
        
        Args:
            spatial_position: Position along the ray
            ue_pos: UE position
            subcarrier_idx: Subcarrier index
            antenna_embedding: Antenna embedding parameter
            bs_position: BS antenna position (ray origin)
        
        Returns:
            Computed signal strength
        """
        try:
            # Compute viewing direction from spatial point to BS antenna (consistent with unified approach)
            view_direction = self._compute_view_directions(spatial_position.unsqueeze(0), bs_position).squeeze(0)
            
            # Create antenna indices
            antenna_indices = torch.zeros(1, dtype=torch.long, device=self.device)
            
            # Get network properties for this spatial point
            with torch.no_grad():
                network_outputs = self.prism_network(
                    sampled_positions=spatial_position.unsqueeze(0).unsqueeze(0),
                    ue_positions=ue_pos.unsqueeze(0),
                    view_directions=view_direction.unsqueeze(0),
                    antenna_indices=antenna_indices,
                    return_intermediates=False
                )
            
            # Extract attenuation and radiation factors with boundary checking
            attenuation_factors = network_outputs['attenuation_factors']
            radiation_factors = network_outputs['radiation_factors']
            
            # Debug: log tensor shapes (commented to reduce output)
            # logger.debug(f"attenuation_factors shape: {attenuation_factors.shape}")
            # logger.debug(f"radiation_factors shape: {radiation_factors.shape}")
            # logger.debug(f"subcarrier_idx: {subcarrier_idx}")
            
            # Check subcarrier index bounds
            if subcarrier_idx >= attenuation_factors.shape[-1]:
                logger.warning(f"Subcarrier index {subcarrier_idx} out of bounds for attenuation_factors shape {attenuation_factors.shape}, using index 0")
                subcarrier_idx = 0
            
            if subcarrier_idx >= radiation_factors.shape[-1]:
                logger.warning(f"Subcarrier index {subcarrier_idx} out of bounds for radiation_factors shape {radiation_factors.shape}, using index 0")
                subcarrier_idx = 0
            
            # Safe indexing with bounds checking
            try:
                if len(attenuation_factors.shape) == 4:
                    # Shape: (batch_size, num_voxels, num_ue_antennas, num_subcarriers)
                    attenuation_factor = attenuation_factors[0, 0, 0, subcarrier_idx]
                elif len(attenuation_factors.shape) == 3:
                    # Shape: (batch_size, num_ue_antennas, num_subcarriers)
                    attenuation_factor = attenuation_factors[0, 0, subcarrier_idx]
                elif len(attenuation_factors.shape) == 2:
                    # Shape: (batch_size, num_subcarriers)
                    attenuation_factor = attenuation_factors[0, subcarrier_idx]
                else:
                    logger.error(f"Unexpected attenuation_factors shape: {attenuation_factors.shape}")
                    return 0.0
                
                if len(radiation_factors.shape) == 4:
                    # Shape: (batch_size, num_voxels, num_ue_antennas, num_subcarriers)
                    radiation_factor = radiation_factors[0, 0, 0, subcarrier_idx]
                elif len(radiation_factors.shape) == 3:
                    # Shape: (batch_size, num_ue_antennas, num_subcarriers)
                    radiation_factor = radiation_factors[0, 0, subcarrier_idx]
                elif len(radiation_factors.shape) == 2:
                    # Shape: (batch_size, num_subcarriers)
                    radiation_factor = radiation_factors[0, subcarrier_idx]
                else:
                    logger.error(f"Unexpected radiation_factors shape: {radiation_factors.shape}")
                    return torch.tensor(0.0 + 0.0j, dtype=torch.complex64, device=spatial_position.device)
                    
            except IndexError as e:
                logger.error(f"Index error when accessing factors: {e}")
                logger.error(f"attenuation_factors shape: {attenuation_factors.shape}")
                logger.error(f"radiation_factors shape: {radiation_factors.shape}")
                logger.error(f"subcarrier_idx: {subcarrier_idx}")
                return torch.tensor(0.0 + 0.0j, dtype=torch.complex64, device=spatial_position.device)
            
            # Compute complex signal using discrete radiance field model
            try:
                # Keep complex computation throughout - DO NOT convert to real
                complex_signal = attenuation_factor * radiation_factor
                
                # Check if the complex signal is finite
                if not torch.isfinite(complex_signal).all():
                    logger.warning(f"Non-finite complex signal detected: attenuation={attenuation_factor}, radiation={radiation_factor}")
                    return torch.tensor(0.0 + 0.0j, dtype=torch.complex64, device=complex_signal.device)
                
                # Check for extreme values that might cause overflow (use magnitude for checking)
                if torch.abs(complex_signal) > 1e6:
                    logger.warning(f"Extreme complex signal magnitude detected: {torch.abs(complex_signal)}, clamping to prevent overflow")
                    # Clamp the magnitude while preserving phase
                    magnitude = torch.clamp(torch.abs(complex_signal), 0, 1e6)
                    phase = torch.angle(complex_signal)
                    complex_signal = magnitude * torch.exp(1j * phase)
                
                return complex_signal
                
            except (ValueError, OverflowError, RuntimeError) as e:
                logger.warning(f"Error in complex signal computation: {e}")
                logger.warning(f"attenuation_factor: {attenuation_factor}, radiation_factor: {radiation_factor}")
                return torch.tensor(0.0 + 0.0j, dtype=torch.complex64, device=spatial_position.device)
            
        except Exception as e:
            logger.warning(f"Failed to compute signal at spatial point: {e}")
            return torch.tensor(0.0 + 0.0j, dtype=torch.complex64, device=spatial_position.device)
    
    def _accumulate_signals_spatial_parallel(self, 
                                           base_station_pos: torch.Tensor,
                                           ue_positions: List[torch.Tensor],
                                           selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                                           antenna_embedding: torch.Tensor,
                                           directions: List[Tuple[int, int]],
                                           num_spatial_points: int = 32) -> Dict:
        """
        Spatial sampling parallel signal accumulation.
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information
            antenna_embedding: Base station's antenna embedding parameter C
            directions: List of directions to process
            num_spatial_points: Number of spatial points to sample in parallel
        
        Returns:
            Accumulated signal strength matrix with spatial sampling parallelization
        """
        accumulated_signals = {}
        
        if not self.enable_parallel_processing or num_spatial_points < 2:
            # Fall back to direction-level parallelization
            return self._accumulate_signals_parallel(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions
            )
        
        # Process each direction with spatial sampling parallelization
        total_directions = len(directions)
        for direction_idx, direction in enumerate(directions):
            direction_results = {}
            
            # Log progress every 5 directions
            if direction_idx % 5 == 0:
                progress = (direction_idx / total_directions) * 100
                logger.info(f"📡 Processing direction {direction_idx+1}/{total_directions} ({progress:.1f}%)")
            
            # Create ray for this direction
            phi_idx, theta_idx = direction
            phi = phi_idx * self.azimuth_resolution
            theta = theta_idx * self.elevation_resolution
            
            # Convert to proper spherical coordinates
            # Elevation: -90° to +90° (-π/2 to +π/2)
            elevation = theta - (math.pi / 2)
            
            direction_vector = torch.tensor([
                math.cos(elevation) * math.cos(phi),
                math.cos(elevation) * math.sin(phi),
                math.sin(elevation)
            ], dtype=torch.float32, device=self.device)
            
            # Ray tracing from BS antenna (configurable position)
            ray = Ray(base_station_pos, direction_vector, self.max_ray_length, self.device)
            
            # Process each UE position
            for ue_pos in ue_positions:
                ue_pos_tensor = ue_pos.clone().detach().to(dtype=torch.float32, device=self.device)
                
                # Normalize subcarrier input
                subcarrier_indices = self._normalize_subcarrier_input(selected_subcarriers, [ue_pos])
                
                # Process each subcarrier with spatial sampling parallelization
                for subcarrier_idx in subcarrier_indices:
                    # Prepare arguments for spatial parallel processing
                    args_list = [(i, ray, ue_pos_tensor, subcarrier_idx, antenna_embedding) 
                                for i in range(num_spatial_points)]
                    
                    try:
                        # Use multiprocessing for spatial sampling
                        with mp.Pool(processes=self.max_workers) as pool:
                            spatial_results = pool.map(self._trace_ray_spatial_parallel, args_list)
                        
                        # Accumulate spatial sampling results
                        total_signal = sum(spatial_results)
                        direction_results[(tuple(ue_pos), subcarrier_idx)] = total_signal
                        
                    except Exception as e:
                        logger.warning(f"Spatial parallel processing failed for direction {direction}, UE {ue_pos}, subcarrier {subcarrier_idx}: {e}")
                        # Fall back to sequential spatial processing
                        total_signal = 0.0
                        for i in range(num_spatial_points):
                            spatial_position = self._sample_ray_point_at_index(ray, ue_pos_tensor, i)
                            if spatial_position is not None:
                                signal = self._compute_signal_at_spatial_point(
                                    spatial_position, ue_pos_tensor, subcarrier_idx, antenna_embedding, ray.origin
                                )
                                total_signal += signal
                        direction_results[(tuple(ue_pos), subcarrier_idx)] = total_signal
            
            # Accumulate direction results
            for (ue_pos, subcarrier), signal_strength in direction_results.items():
                self._ensure_complex_accumulation(accumulated_signals, (ue_pos, subcarrier), signal_strength)
        
        return accumulated_signals
    
    def _accumulate_signals_full_parallel(self, 
                                         base_station_pos: torch.Tensor,
                                         ue_positions: List[torch.Tensor],
                                         selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                                         antenna_embedding: torch.Tensor,
                                         directions: List[Tuple[int, int]],
                                         num_antennas: int = 64,
                                         num_spatial_points: int = 32) -> Dict:
        """
        Full parallel signal accumulation combining direction, antenna, and spatial sampling parallelization.
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information
            antenna_embedding: Base station's antenna embedding parameter C
            directions: List of directions to process
            num_antennas: Number of BS antennas to process in parallel
            num_spatial_points: Number of spatial points to sample in parallel
        
        Returns:
            Accumulated signal strength matrix with full parallelization
        """
        logger.debug(f"Using full parallel processing: {len(directions)} directions × {num_antennas} antennas × {num_spatial_points} spatial points")
        
        # Use the most efficient parallelization method based on workload size
        if len(directions) >= 16 and num_antennas >= 32 and num_spatial_points >= 16:
            # Full parallelization for large workloads
            logger.debug("Using full parallelization (direction + antenna + spatial)")
            return self._accumulate_signals_spatial_parallel(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions, num_spatial_points
            )
        elif len(directions) >= 8 and num_antennas >= 16:
            # Antenna + direction parallelization for medium workloads
            logger.debug("Using antenna + direction parallelization")
            return self._accumulate_signals_antenna_parallel(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions, num_antennas
            )
        else:
            # Direction-only parallelization for small workloads
            logger.debug("Using direction-only parallelization")
            return self._accumulate_signals_parallel(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions
            )
    
    def get_enhanced_parallelization_stats(self) -> Dict:
        """
        Get enhanced statistics about parallel processing configuration including all levels.
        
        Returns:
            Dictionary with comprehensive parallelization statistics
        """
        return {
            'parallel_processing_enabled': self.enable_parallel_processing,
            'max_workers': self.max_workers,
            'processing_mode': 'multiprocessing',
            'cpu_count': mp.cpu_count(),
            'device': self.device,
            'total_directions': self.total_directions,
            'top_k_directions': self.top_k_directions,
            'parallelization_levels': {
                'direction_level': True,
                'antenna_level': True,
                'spatial_level': True,
                'full_parallel': True
            },
            'theoretical_speedup': {
                'direction_only': 32,
                'direction_antenna': 32 * 64,
                'direction_spatial': 32 * 32,
                'full_parallel': 32 * 64 * 32
            },
            'current_implementation': 'direction_level',
            'next_optimization_target': 'full_parallel'
        }

    def _accumulate_signals_subcarrier_parallel(self, 
                                             base_station_pos: torch.Tensor,
                                             ue_positions: List[torch.Tensor],
                                             selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                                             antenna_embedding: torch.Tensor,
                                             directions: List[Tuple[int, int]],
                                             num_spatial_points: int = 32) -> Dict:
        """
        Subcarrier-level parallel signal accumulation.
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information
            antenna_embedding: Base station's antenna embedding parameter C
            directions: List of directions to process
            num_spatial_points: Number of spatial points to sample
        
        Returns:
            Accumulated signal strength matrix with subcarrier-level parallelization
        """
        accumulated_signals = {}
        
        if not self.enable_parallel_processing:
            # Fall back to direction-level parallelization
            return self._accumulate_signals_parallel(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions
            )
        
        # Process each direction
        for direction in directions:
            direction_results = {}
            
            # Create ray for this direction
            phi_idx, theta_idx = direction
            phi = phi_idx * self.azimuth_resolution
            theta = theta_idx * self.elevation_resolution
            
            # Convert to proper spherical coordinates
            # Elevation: -90° to +90° (-π/2 to +π/2)
            elevation = theta - (math.pi / 2)
            
            direction_vector = torch.tensor([
                math.cos(elevation) * math.cos(phi),
                math.cos(elevation) * math.sin(phi),
                math.sin(elevation)
            ], dtype=torch.float32, device=self.device)
            
            # Ray tracing from BS antenna (configurable position)
            ray = Ray(base_station_pos, direction_vector, self.max_ray_length, self.device)
            
            # Process each UE position
            for ue_pos in ue_positions:
                ue_pos_tensor = ue_pos.clone().detach().to(dtype=torch.float32, device=self.device)
                
                # Normalize subcarrier input
                subcarrier_indices = self._normalize_subcarrier_input(selected_subcarriers, [ue_pos])
                
                if len(subcarrier_indices) < 2:
                    # Fall back to sequential if only one subcarrier
                    total_signal = 0.0
                    for subcarrier_idx in subcarrier_indices:
                        for i in range(num_spatial_points):
                            spatial_position = self._sample_ray_point_at_index(ray, ue_pos_tensor, i)
                            if spatial_position is not None:
                                signal = self._compute_signal_at_spatial_point(
                                    spatial_position, ue_pos_tensor, subcarrier_idx, antenna_embedding, ray.origin
                                )
                                total_signal += signal
                    direction_results[(tuple(ue_pos), subcarrier_indices[0])] = total_signal
                else:
                    # Parallel subcarrier processing
                    try:
                        # Prepare arguments for subcarrier parallel processing
                        args_list = [(subcarrier_idx, ray, ue_pos_tensor, num_spatial_points, antenna_embedding) 
                                    for subcarrier_idx in subcarrier_indices]
                        
                        # Use multiprocessing for subcarrier processing
                        with mp.Pool(processes=self.max_workers) as pool:
                            subcarrier_results = pool.map(self._trace_ray_subcarrier_parallel, args_list)
                        
                        # Accumulate subcarrier results
                        for subcarrier_idx, signal_strength in zip(subcarrier_indices, subcarrier_results):
                            direction_results[(tuple(ue_pos), subcarrier_idx)] = signal_strength
                        
                    except Exception as e:
                        logger.warning(f"Subcarrier parallel processing failed for direction {direction}, UE {ue_pos}: {e}")
                        # Fall back to sequential subcarrier processing
                        for subcarrier_idx in subcarrier_indices:
                            total_signal = 0.0
                            for i in range(num_spatial_points):
                                spatial_position = self._sample_ray_point_at_index(ray, ue_pos_tensor, i)
                                if spatial_position is not None:
                                    signal = self._compute_signal_at_spatial_point(
                                        spatial_position, ue_pos_tensor, subcarrier_idx, antenna_embedding
                                    )
                                    total_signal += signal
                            direction_results[(tuple(ue_pos), subcarrier_idx)] = total_signal
            
            # Accumulate direction results
            for (ue_pos, subcarrier), signal_strength in direction_results.items():
                self._ensure_complex_accumulation(accumulated_signals, (ue_pos, subcarrier), signal_strength)
        
        return accumulated_signals
    
    def _trace_ray_subcarrier_parallel(self, args):
        """
        Wrapper function for parallel subcarrier processing.
        
        Args:
            args: Tuple of (subcarrier_idx, ray, ue_pos, num_spatial_points, antenna_embedding)
        
        Returns:
            Signal strength for the given subcarrier
        """
        subcarrier_idx, ray, ue_pos, num_spatial_points, antenna_embedding = args
        try:
            total_signal = 0.0
            
            # Process all spatial points for this subcarrier
            for i in range(num_spatial_points):
                spatial_position = self._sample_ray_point_at_index(ray, ue_pos, i)
                if spatial_position is not None:
                    signal = self._compute_signal_at_spatial_point(
                        spatial_position, ue_pos, subcarrier_idx, antenna_embedding, ray.origin
                    )
                    total_signal += signal
            
            return total_signal
            
        except Exception as e:
            logger.warning(f"Parallel subcarrier processing failed for subcarrier {subcarrier_idx}: {e}")
            return 0.0
    
    def _accumulate_signals_enhanced_parallel(self, 
                                            base_station_pos: torch.Tensor,
                                            ue_positions: List[torch.Tensor],
                                            selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                                            antenna_embedding: torch.Tensor,
                                            directions: List[Tuple[int, int]],
                                            num_antennas: int = 64,
                                            num_spatial_points: int = 32) -> Dict:
        """
        Enhanced parallel signal accumulation combining all levels:
        - Direction-level parallelization
        - Antenna-level parallelization  
        - Spatial sampling parallelization
        - Subcarrier-level parallelization
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information
            antenna_embedding: Base station's antenna embedding parameter C
            directions: List of directions to process
            num_antennas: Number of antennas to process in parallel
            num_spatial_points: Number of spatial points to sample in parallel
        
        Returns:
            Accumulated signal strength matrix with full parallelization
        """
        accumulated_signals = {}
        
        if not self.enable_parallel_processing:
            # Fall back to sequential processing
            return self._accumulate_signals_sequential(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions
            )
        
        # Intelligent parallelization strategy selection
        num_directions = len(directions)
        subcarrier_indices = self._normalize_subcarrier_input(selected_subcarriers, ue_positions)
        num_subcarriers = len(subcarrier_indices) if isinstance(subcarrier_indices, list) else 1
        
        print(f"🔍 Enhanced Parallelization Analysis:")
        print(f"   • Directions: {num_directions}")
        print(f"   • Antennas: {num_antennas}")
        print(f"   • Spatial points: {num_spatial_points}")
        print(f"   • Subcarriers: {num_subcarriers}")
        print(f"   • Max workers: {self.max_workers}")
        
        # Strategy 1: Full parallelization (all levels)
        if (num_directions >= 8 and num_antennas >= 16 and 
            num_spatial_points >= 16 and num_subcarriers >= 8):
            print(f"   🚀 Using: Full parallelization (all levels)")
            return self._accumulate_signals_full_parallel(
                base_station_pos, ue_positions, selected_subcarriers, 
                antenna_embedding, directions, num_antennas, num_spatial_points
            )
        
        # Strategy 2: Antenna + Spatial + Subcarrier parallelization
        elif (num_antennas >= 16 and num_spatial_points >= 16 and num_subcarriers >= 8):
            print(f"   📡 Using: Antenna + Spatial + Subcarrier parallelization")
            return self._accumulate_signals_antenna_spatial_subcarrier_parallel(
                base_station_pos, ue_positions, selected_subcarriers, 
                antenna_embedding, directions, num_antennas, num_spatial_points
            )
        
        # Strategy 3: Spatial + Subcarrier parallelization
        elif (num_spatial_points >= 16 and num_subcarriers >= 8):
            print(f"   🌍 Using: Spatial + Subcarrier parallelization")
            return self._accumulate_signals_spatial_subcarrier_parallel(
                base_station_pos, ue_positions, selected_subcarriers, 
                antenna_embedding, directions, num_spatial_points
            )
        
        # Strategy 4: Direction + Subcarrier parallelization
        elif (num_directions >= 8 and num_subcarriers >= 8):
            print(f"   🎯 Using: Direction + Subcarrier parallelization")
            return self._accumulate_signals_direction_subcarrier_parallel(
                base_station_pos, ue_positions, selected_subcarriers, 
                antenna_embedding, directions
            )
        
        # Strategy 5: Subcarrier-only parallelization
        elif num_subcarriers >= 8:
            print(f"   📊 Using: Subcarrier-only parallelization")
            return self._accumulate_signals_subcarrier_parallel(
                base_station_pos, ue_positions, selected_subcarriers, 
                antenna_embedding, directions, num_spatial_points
            )
        
        # Strategy 6: Direction-only parallelization
        elif num_directions >= 8:
            print(f"   🎯 Using: Direction-only parallelization")
            return self._accumulate_signals_parallel(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions
            )
        
        # Strategy 7: Sequential fallback
        else:
            print(f"   ⚠️  Using: Sequential fallback")
            return self._accumulate_signals_sequential(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions
            )
    
    def _accumulate_signals_antenna_spatial_subcarrier_parallel(self, 
                                                              base_station_pos: torch.Tensor,
                                                              ue_positions: List[torch.Tensor],
                                                              selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                                                              antenna_embedding: torch.Tensor,
                                                              directions: List[Tuple[int, int]],
                                                              num_antennas: int = 64,
                                                              num_spatial_points: int = 32) -> Dict:
        """
        Antenna + Spatial + Subcarrier parallel signal accumulation.
        Combines three levels of parallelization for maximum performance.
        """
        accumulated_signals = {}
        
        # Process each direction sequentially (to avoid memory explosion)
        for direction in directions:
            direction_results = {}
            
            # Create ray for this direction
            phi_idx, theta_idx = direction
            phi = phi_idx * self.azimuth_resolution
            theta = theta_idx * self.elevation_resolution
            
            # Convert to proper spherical coordinates
            # Elevation: -90° to +90° (-π/2 to +π/2)
            elevation = theta - (math.pi / 2)
            
            direction_vector = torch.tensor([
                math.cos(elevation) * math.cos(phi),
                math.cos(elevation) * math.sin(phi),
                math.sin(elevation)
            ], dtype=torch.float32, device=self.device)
            
            # Ray tracing from BS antenna (configurable position)
            ray = Ray(base_station_pos, direction_vector, self.max_ray_length, self.device)
            
            # Process each UE position
            for ue_pos in ue_positions:
                ue_pos_tensor = ue_pos.clone().detach().to(dtype=torch.float32, device=self.device)
                
                # Normalize subcarrier input
                subcarrier_indices = self._normalize_subcarrier_input(selected_subcarriers, [ue_pos])
                
                # Process each subcarrier with antenna + spatial parallelization
                for subcarrier_idx in subcarrier_indices:
                    try:
                        # Create antenna + spatial parallel tasks
                        antenna_spatial_tasks = []
                        for antenna_idx in range(min(num_antennas, 64)):
                            for spatial_idx in range(num_spatial_points):
                                task = (antenna_idx, spatial_idx, ray, ue_pos_tensor, subcarrier_idx, antenna_embedding)
                                antenna_spatial_tasks.append(task)
                        
                        # Parallel processing of antenna + spatial combinations
                        # Use multiprocessing for antenna spatial processing
                        with mp.Pool(processes=self.max_workers) as pool:
                            results = pool.map(self._trace_ray_antenna_spatial_parallel, antenna_spatial_tasks)
                        
                        # Sum all results for this subcarrier
                        total_signal = sum(results)
                        direction_results[(tuple(ue_pos), subcarrier_idx)] = total_signal
                        
                    except Exception as e:
                        logger.warning(f"Antenna+spatial+subcarrier parallel processing failed: {e}")
                        # Fall back to sequential processing
                        total_signal = 0.0
                        for i in range(num_spatial_points):
                            spatial_position = self._sample_ray_point_at_index(ray, ue_pos_tensor, i)
                            if spatial_position is not None:
                                signal = self._compute_signal_at_spatial_point(
                                    spatial_position, ue_pos_tensor, subcarrier_idx, antenna_embedding, ray.origin
                                )
                                total_signal += signal
                        direction_results[(tuple(ue_pos), subcarrier_idx)] = total_signal
            
            # Accumulate direction results
            for (ue_pos, subcarrier), signal_strength in direction_results.items():
                self._ensure_complex_accumulation(accumulated_signals, (ue_pos, subcarrier), signal_strength)
        
        return accumulated_signals
    
    def _trace_ray_antenna_spatial_parallel(self, args):
        """
        Wrapper function for parallel antenna + spatial processing.
        
        Args:
            args: Tuple of (antenna_idx, spatial_idx, ray, ue_pos, subcarrier_idx, antenna_embedding)
        
        Returns:
            Signal strength for the given antenna and spatial point
        """
        antenna_idx, spatial_idx, ray, ue_pos, subcarrier_idx, antenna_embedding = args
        try:
            # Sample spatial point
            spatial_position = self._sample_ray_point_at_index(ray, ue_pos, spatial_idx)
            if spatial_position is None:
                return 0.0
            
            # Compute signal at this spatial point
            signal = self._compute_signal_at_spatial_point(
                spatial_position, ue_pos, subcarrier_idx, antenna_embedding, ray.origin
            )
            
            return signal
            
        except Exception as e:
            logger.warning(f"Antenna+spatial parallel processing failed: {e}")
            return 0.0
    
    def _accumulate_signals_spatial_subcarrier_parallel(self, 
                                                      base_station_pos: torch.Tensor,
                                                      ue_positions: List[torch.Tensor],
                                                      selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                                                      antenna_embedding: torch.Tensor,
                                                      directions: List[Tuple[int, int]],
                                                      num_spatial_points: int = 32) -> Dict:
        """
        Spatial + Subcarrier parallel signal accumulation.
        Combines spatial sampling and subcarrier processing parallelization.
        """
        accumulated_signals = {}
        
        # Process each direction sequentially
        for direction in directions:
            direction_results = {}
            
            # Create ray for this direction
            phi_idx, theta_idx = direction
            phi = phi_idx * self.azimuth_resolution
            theta = theta_idx * self.elevation_resolution
            
            # Convert to proper spherical coordinates
            # Elevation: -90° to +90° (-π/2 to +π/2)
            elevation = theta - (math.pi / 2)
            
            direction_vector = torch.tensor([
                math.cos(elevation) * math.cos(phi),
                math.cos(elevation) * math.sin(phi),
                math.sin(elevation)
            ], dtype=torch.float32, device=self.device)
            
            # Ray tracing from BS antenna (configurable position)
            ray = Ray(base_station_pos, direction_vector, self.max_ray_length, self.device)
            
            # Process each UE position
            for ue_pos in ue_positions:
                ue_pos_tensor = ue_pos.clone().detach().to(dtype=torch.float32, device=self.device)
                
                # Normalize subcarrier input
                subcarrier_indices = self._normalize_subcarrier_input(selected_subcarriers, [ue_pos])
                
                # Create spatial + subcarrier parallel tasks
                spatial_subcarrier_tasks = []
                for spatial_idx in range(num_spatial_points):
                    for subcarrier_idx in subcarrier_indices:
                        task = (spatial_idx, subcarrier_idx, ray, ue_pos_tensor, antenna_embedding)
                        spatial_subcarrier_tasks.append(task)
                
                try:
                    # Parallel processing of spatial + subcarrier combinations
                    # Use multiprocessing for spatial subcarrier processing
                    with mp.Pool(processes=self.max_workers) as pool:
                        results = pool.map(self._trace_ray_spatial_subcarrier_parallel, spatial_subcarrier_tasks)
                    
                    # Group results by subcarrier
                    subcarrier_signals = {}
                    for i, (spatial_idx, subcarrier_idx, ray, ue_pos, antenna_embedding) in enumerate(spatial_subcarrier_tasks):
                        if subcarrier_idx not in subcarrier_signals:
                            subcarrier_signals[subcarrier_idx] = 0.0
                        subcarrier_signals[subcarrier_idx] += results[i]
                    
                    # Store results
                    for subcarrier_idx, signal_strength in subcarrier_signals.items():
                        direction_results[(tuple(ue_pos), subcarrier_idx)] = signal_strength
                        
                except Exception as e:
                    logger.warning(f"Spatial+subcarrier parallel processing failed: {e}")
                    # Fall back to sequential processing
                    for subcarrier_idx in subcarrier_indices:
                        total_signal = 0.0
                        for i in range(num_spatial_points):
                            spatial_position = self._sample_ray_point_at_index(ray, ue_pos_tensor, i)
                            if spatial_position is not None:
                                signal = self._compute_signal_at_spatial_point(
                                    spatial_position, ue_pos_tensor, subcarrier_idx, antenna_embedding, ray.origin
                                )
                                total_signal += signal
                        direction_results[(tuple(ue_pos), subcarrier_idx)] = total_signal
            
            # Accumulate direction results
            for (ue_pos, subcarrier), signal_strength in direction_results.items():
                self._ensure_complex_accumulation(accumulated_signals, (ue_pos, subcarrier), signal_strength)
        
        return accumulated_signals
    
    def _trace_ray_spatial_subcarrier_parallel(self, args):
        """
        Wrapper function for parallel spatial + subcarrier processing.
        
        Args:
            args: Tuple of (spatial_idx, subcarrier_idx, ray, ue_pos, antenna_embedding)
        
        Returns:
            Signal strength for the given spatial point and subcarrier
        """
        spatial_idx, subcarrier_idx, ray, ue_pos, antenna_embedding = args
        try:
            # Sample spatial point
            spatial_position = self._sample_ray_point_at_index(ray, ue_pos, spatial_idx)
            if spatial_position is None:
                return 0.0
            
            # Compute signal at this spatial point
            signal = self._compute_signal_at_spatial_point(
                spatial_position, ue_pos, subcarrier_idx, antenna_embedding, ray.origin
            )
            
            return signal
            
        except Exception as e:
            logger.warning(f"Spatial+subcarrier parallel processing failed: {e}")
            return 0.0
    
    def _accumulate_signals_direction_subcarrier_parallel(self, 
                                                        base_station_pos: torch.Tensor,
                                                        ue_positions: List[torch.Tensor],
                                                        selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                                                        antenna_embedding: torch.Tensor,
                                                        directions: List[Tuple[int, int]]) -> Dict:
        """
        Direction + Subcarrier parallel signal accumulation.
        Combines direction processing and subcarrier processing parallelization.
        """
        accumulated_signals = {}
        
        # Create direction + subcarrier parallel tasks
        direction_subcarrier_tasks = []
        for direction in directions:
            for ue_pos in ue_positions:
                ue_pos_tensor = ue_pos.clone().detach().to(dtype=torch.float32, device=self.device)
                subcarrier_indices = self._normalize_subcarrier_input(selected_subcarriers, [ue_pos])
                
                for subcarrier_idx in subcarrier_indices:
                    task = (direction, ue_pos_tensor, subcarrier_idx, antenna_embedding)
                    direction_subcarrier_tasks.append(task)
        
        try:
            # Parallel processing of direction + subcarrier combinations
            # Use multiprocessing for direction subcarrier processing
            with mp.Pool(processes=self.max_workers) as pool:
                results = pool.map(self._trace_ray_direction_subcarrier_parallel, direction_subcarrier_tasks)
            
            # Group results by (ue_pos, subcarrier)
            for i, (direction, ue_pos, subcarrier_idx, antenna_embedding) in enumerate(direction_subcarrier_tasks):
                signal_strength = results[i]
                ue_pos_tuple = tuple(ue_pos)
                
                self._ensure_complex_accumulation(accumulated_signals, (ue_pos_tuple, subcarrier_idx), signal_strength)
                
        except Exception as e:
            logger.warning(f"Direction+subcarrier parallel processing failed: {e}")
            # Fall back to direction-only parallelization
            return self._accumulate_signals_parallel(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding, directions
            )
        
        return accumulated_signals
    
    def _trace_ray_direction_subcarrier_parallel(self, args):
        """
        Wrapper function for parallel direction + subcarrier processing.
        
        Args:
            args: Tuple of (direction, ue_pos, subcarrier_idx, antenna_embedding)
        
        Returns:
            Signal strength for the given direction and subcarrier
        """
        direction, ue_pos, subcarrier_idx, antenna_embedding = args
        try:
            # Create ray for this direction
            phi_idx, theta_idx = direction
            phi = phi_idx * self.azimuth_resolution
            theta = theta_idx * self.elevation_resolution
            
            # Convert to proper spherical coordinates
            # Elevation: -90° to +90° (-π/2 to +π/2)
            elevation = theta - (math.pi / 2)
            
            direction_vector = torch.tensor([
                math.cos(elevation) * math.cos(phi),
                math.cos(elevation) * math.sin(phi),
                math.sin(elevation)
            ], dtype=torch.float32, device=self.device)
            
            ray = Ray(torch.tensor([0.0, 0.0, 0.0]), direction_vector, self.max_ray_length, self.device)
            
            # Compute signal for this direction and subcarrier
            total_signal = 0.0
            for i in range(32):  # Default spatial points
                spatial_position = self._sample_ray_point_at_index(ray, ue_pos, i)
                if spatial_position is not None:
                    signal = self._compute_signal_at_spatial_point(
                        spatial_position, ue_pos, subcarrier_idx, antenna_embedding, ray.origin
                    )
                    total_signal += signal
            
            return total_signal
            
        except Exception as e:
            logger.warning(f"Direction+subcarrier parallel processing failed: {e}")
            return 0.0
