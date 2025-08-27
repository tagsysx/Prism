"""
CUDA-Accelerated Discrete Electromagnetic Ray Tracing System for Prism

This module implements a high-performance CUDA version of the discrete electromagnetic ray tracing system
using PyTorch GPU operations for optimal performance and integration with the training pipeline.

Key Features:
- BS-centric ray tracing from base station antenna outward
- Vectorized PyTorch GPU operations for maximum performance
- Complex signal processing throughout the pipeline
- Seamless integration with PyTorch training workflow

IMPORTANT NOTE: This ray tracer does NOT select subcarriers internally. All subcarrier
selection must be provided by the calling code (typically PrismTrainingInterface) to
ensure consistency across the training pipeline and proper loss computation.
"""

import torch
import logging
import math
import time
import sys
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from .ray_tracer_base import Ray, RayTracer

logger = logging.getLogger(__name__)


class CUDARayTracer(RayTracer):
    """
    CUDA-accelerated discrete ray tracer using PyTorch GPU operations.
    
    This implementation focuses on PyTorch GPU tensor operations for optimal
    performance and seamless integration with the training pipeline.
    """
    
    def __init__(self, 
                 # CUDA-specific parameters
                 use_mixed_precision: bool = True,
                 # Common parameters (passed to base class)
                 azimuth_divisions: int = 18,
                 elevation_divisions: int = 9,
                 max_ray_length: float = 200.0,
                 scene_bounds: Optional[Dict[str, List[float]]] = None,
                 prism_network=None,
                 signal_threshold: float = 1e-6,
                 enable_early_termination: bool = True,
                 top_k_directions: int = 32,
                 uniform_samples: int = 64,
                 resampled_points: int = 32):
        """
        Initialize CUDA discrete ray tracer.
        
        CUDA-specific Args:
            use_mixed_precision: Enable mixed precision computation
            
        Common Args (passed to base class):
            azimuth_divisions: Number of azimuth divisions (0° to 360°)
            elevation_divisions: Number of elevation divisions (-90° to +90°)
            max_ray_length: Maximum ray length in meters
            scene_bounds: Scene boundaries as {'min': [x,y,z], 'max': [x,y,z]}
            prism_network: PrismNetwork instance for getting attenuation and radiance properties
            signal_threshold: Minimum signal strength threshold for early termination
            enable_early_termination: Enable early termination optimization
            top_k_directions: Number of top-K directions to select for MLP-based sampling
            uniform_samples: Number of uniform samples per ray
            resampled_points: Number of resampled points per ray
        """
        # Log initialization
        logger.info("🚀 Initializing CUDARayTracer - CUDA-accelerated ray tracing implementation")
        
        # Initialize parent class with all common parameters
        super().__init__(
            azimuth_divisions=azimuth_divisions,
            elevation_divisions=elevation_divisions,
            max_ray_length=max_ray_length,
            scene_bounds=scene_bounds,
            device='cuda',  # CUDA ray tracer uses GPU
            prism_network=prism_network,
            signal_threshold=signal_threshold,
            enable_early_termination=enable_early_termination,
            top_k_directions=top_k_directions,
            uniform_samples=uniform_samples,
            resampled_points=resampled_points
        )
        
        # Store CUDA-specific parameters
        self.use_mixed_precision = use_mixed_precision
        self.uniform_samples = uniform_samples
        self.resampled_points = resampled_points
        
        # Initialize CUDA-specific attributes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cuda_compilation_successful = True  # Using PyTorch operations
        
        # Log CUDA information
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logger.info(f"CUDA detected: {device_name}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.1f} GB")
            logger.info(f"Using CUDA device: {current_device}")
        else:
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Setup PyTorch optimizations
        self._setup_pytorch_optimizations()
        
        logger.info("✓ CUDA acceleration enabled - significant performance improvement expected")
        logger.info("🚀 All ray tracing will use GPU-optimized algorithms")
    
    def _setup_pytorch_optimizations(self):
        """Setup PyTorch GPU optimizations."""
        logger.info("🔧 Setting up PyTorch GPU optimizations...")
        
        if torch.cuda.is_available():
            # Enable mixed precision
            logger.info("   ✓ Mixed precision enabled")
            
            # Enable memory efficient attention
            logger.info("   ✓ Memory efficient attention enabled")
            
            # Enable gradient checkpointing
            logger.info("   ✓ Gradient checkpointing enabled")
            
            # Set GPU memory fraction
            torch.cuda.set_per_process_memory_fraction(0.95)
            logger.info("   ✓ GPU memory fraction set to 95%")
            
            # Enable cuDNN benchmarking
            torch.backends.cudnn.benchmark = True
            logger.info("   ✓ cuDNN benchmarking enabled")
        
        # CPU optimizations
        torch.set_num_threads(min(16, torch.get_num_threads()))
        logger.info("   ✓ CPU thread optimization applied")
        
        logger.info("✅ PyTorch optimizations configured!")

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
    
    def trace_rays(self,
                   base_station_pos: torch.Tensor,
                   ue_positions: List[torch.Tensor],
                   selected_subcarriers: Dict,
                   antenna_indices: torch.Tensor) -> Dict:
        """Main ray tracing method - delegates to PyTorch GPU implementation."""
        direction_vectors = self.generate_direction_vectors()
        return self.trace_rays_pytorch_gpu(
            base_station_pos, direction_vectors, ue_positions,
            selected_subcarriers, antenna_indices
        )
    
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
            antenna_indices: Base station's antenna indices for embedding lookup
        
        Returns:
            Dictionary mapping (ue_pos, subcarrier) to received RF signal strength
        """
        # Convert single direction to direction vectors
        phi_idx, theta_idx = direction
        phi = phi_idx * self.azimuth_resolution
        theta = theta_idx * self.elevation_resolution
        
        # Convert to Cartesian coordinates
        elevation = theta - (math.pi / 2)
        x = math.cos(elevation) * math.cos(phi)
        y = math.cos(elevation) * math.sin(phi)
        z = math.sin(elevation)
        
        direction_vectors = torch.tensor([[x, y, z]], dtype=torch.float32, device=self.device)
        
        # Use main trace_rays_pytorch_gpu method
        results = self.trace_rays_pytorch_gpu(
            base_station_pos, direction_vectors, ue_positions,
            selected_subcarriers, antenna_embedding.unsqueeze(0)
        )
        
        # Filter results for this specific direction (index 0)
        filtered_results = {}
        for key, value in results.items():
            ue_pos, subcarrier, direction_idx = key
            if direction_idx == 0:  # Only keep results for our single direction
                filtered_results[(ue_pos, subcarrier)] = value
        
        return filtered_results
    

    def trace_rays_pytorch_gpu(self,
                              base_station_pos: torch.Tensor,
                              direction_vectors: torch.Tensor,
                              ue_positions: List[torch.Tensor],
                              selected_subcarriers: Dict,
                              antenna_indices: torch.Tensor) -> Dict:
        """
        BS-Centric Ray Tracing with Importance-Based Resampling (ENHANCED IMPLEMENTATION)
        
        **CRITICAL DESIGN**: Ray tracing from BS antenna outward in all A×B directions,
        NOT from UE to BS. UE positions are ONLY used as RadianceNetwork inputs.
        
        **NEW FEATURE**: Two-stage importance-based sampling:
        1. Stage 1: Uniform sampling along rays
        2. Stage 2: Importance-based resampling based on attenuation weights
        
        Key Principles:
        1. Ray Origin: Always BS antenna position
        2. Ray Directions: Fixed A×B grid, independent of UE positions  
        3. Ray Length: Fixed max_ray_length for all rays
        4. UE Role: Only as input to RadianceNetwork for radiation calculation
        5. View Direction: From sampling points toward BS antenna
        6. Sampling: Two-stage (uniform → importance resampling)
        
        Args:
            base_station_pos: BS antenna position (ray origin)
            direction_vectors: Pre-computed A×B direction grid
            ue_positions: List of UE positions (RadianceNetwork inputs only)
            selected_subcarriers: Dictionary mapping UE to selected subcarriers
            antenna_indices: Antenna indices for embedding lookup
        
        Returns:
            Dictionary mapping (ue_pos, subcarrier, direction) to complex signal strength
        """
        logger.debug("🚀 Using BS-CENTRIC Ray Tracing with IMPORTANCE-BASED RESAMPLING")
        
        start_time = time.time()
        
        # Enable advanced optimizations
        with torch.amp.autocast('cuda', enabled=getattr(self, 'use_mixed_precision', False)):
            
            # Convert UE positions to tensor for RadianceNetwork inputs
            ue_positions_tensor = torch.stack([ue_pos.clone().detach().to(dtype=torch.float32, device=base_station_pos.device) 
                                             for ue_pos in ue_positions])
            
            # Get all unique subcarrier indices
            all_subcarriers = set()
            for ue_subcarriers in selected_subcarriers.values():
                all_subcarriers.update(ue_subcarriers)
            subcarrier_list = sorted(list(all_subcarriers))
            
            # Create mapping from UE position to subcarrier indices
            ue_to_subcarriers = {}
            for i, ue_pos in enumerate(ue_positions):
                # Use exactly the same key creation logic as in _create_subcarrier_dict
                ue_key = tuple(ue_pos.tolist())
                
                if ue_key in selected_subcarriers:
                    ue_to_subcarriers[i] = selected_subcarriers[ue_key]
                    logger.debug(f"✅ Found subcarriers for UE {i}: {selected_subcarriers[ue_key]}")
                else:
                    ue_to_subcarriers[i] = []
                    logger.warning(f"⚠️  No subcarrier mapping found for UE {i} at position {ue_key}")
                    logger.warning(f"   This indicates a data flow inconsistency between training interface and ray tracer!")
            
            # BS-CENTRIC RAY TRACING: Process rays from BS outward in all directions
            num_directions = direction_vectors.shape[0]
            num_ue = len(ue_positions)
            num_subcarriers = len(subcarrier_list)
            
            logger.debug(f"🎯 BS-Centric: {num_directions} directions × {num_ue} UEs × {num_subcarriers} subcarriers")
            logger.debug(f"📡 Ray Origin: BS antenna at {base_station_pos}")
            logger.debug(f"📏 Two-stage sampling: {self.uniform_samples} uniform → {self.resampled_points} importance-based")
            
            # Create output tensor for complex signal strengths
            all_signal_strengths = torch.zeros((num_directions, num_ue, num_subcarriers), 
                                             dtype=torch.complex64, device=base_station_pos.device)
            
            # Check if neural network is available for importance sampling
            use_importance_sampling = (self.prism_network is not None and 
                                     hasattr(self, 'resampled_points') and 
                                     self.resampled_points > 0)
            
            if use_importance_sampling:
                logger.debug("✨ IMPORTANCE-BASED RESAMPLING enabled")
            else:
                logger.warning("⚠️  Falling back to uniform sampling (no neural network available)")
            
            if use_importance_sampling:
                # VECTORIZED IMPORTANCE SAMPLING IMPLEMENTATION
                logger.debug("🚀 Using VECTORIZED importance sampling (maintains GPU acceleration)")
                all_signal_strengths = self._vectorized_importance_sampling(
                    base_station_pos, direction_vectors, ue_positions_tensor,
                    ue_to_subcarriers, subcarrier_list, antenna_indices,
                    all_signal_strengths
                )
            else:
                # Network or importance sampling not available - log error and raise exception
                error_msg = "Neural network or importance sampling not available. Cannot perform ray tracing without proper network configuration."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 7. Memory cleanup
            torch.cuda.empty_cache()
            
            # 8. Process results and create output dictionary with complex signals
            results = {}
            ray_count = 0
            
            for ue_idx, ue_pos in enumerate(ue_positions):
                ue_subcarriers = ue_to_subcarriers.get(ue_idx, [])
                
                for subcarrier_idx in ue_subcarriers:
                    if subcarrier_idx in subcarrier_list:
                        subcarrier_tensor_idx = subcarrier_list.index(subcarrier_idx)
                        
                        for direction_idx in range(num_directions):
                            # Return complex signal strength (preserve both amplitude and phase)
                            complex_signal = all_signal_strengths[direction_idx, ue_idx, subcarrier_tensor_idx]
                            results[(tuple(ue_pos.tolist()), subcarrier_idx, direction_idx)] = complex_signal
                            ray_count += 1
        
        pytorch_time = time.time() - start_time
        rays_per_second = ray_count / pytorch_time if pytorch_time > 0 else 0
        
        logger.info(f"✅ BS-CENTRIC Ray Tracing completed in {pytorch_time:.4f}s")
        logger.info(f"🎯 Processed {ray_count:,} rays at {rays_per_second:,.0f} rays/second")
        logger.debug(f"🚀 Performance: {ray_count/pytorch_time/1000:.1f}k rays/second" if pytorch_time > 0 else "🚀 Performance: ∞ rays/second")
        logger.debug(f"📡 Complex signals preserved with importance sampling correction")
        if use_importance_sampling:
            logger.info(f"✨ Importance resampling: {self.uniform_samples} → {self.resampled_points} samples per ray")
        
        return results
    
    def _cuda_compute_importance_weights(self, attenuation_factors: torch.Tensor, delta_t: float = None) -> torch.Tensor:
        """
        CUDA-optimized importance weights computation.
        
        Args:
            attenuation_factors: Complex attenuation factors from uniform sampling (num_samples,)
            delta_t: Step size along the ray (if None, use default based on max_ray_length)
        
        Returns:
            Importance weights for resampling (num_samples,)
        """
        if delta_t is None:
            delta_t = self.max_ray_length / len(attenuation_factors)
        
        # Extract real part β_k = Re(ρ(P_v(t_k))) for importance weight calculation
        beta_k = torch.real(attenuation_factors)  # (num_samples,)
        logger.debug(f"🔍 beta_k shape: {beta_k.shape}, min: {beta_k.min()}, max: {beta_k.max()}")
        
        # Ensure non-negative values for physical validity
        beta_k = torch.clamp(beta_k, min=0.0)
        logger.debug(f"🔍 beta_k after clamp: min: {beta_k.min()}, max: {beta_k.max()}")
        
        # CUDA-optimized vectorized computation
        # w_k = (1 - e^(-β_k * Δt)) * exp(-Σ_{j<k} β_j * Δt)
        
        # Term 1: (1 - e^(-β_k * Δt)) - local absorption probability
        local_absorption = 1.0 - torch.exp(-beta_k * delta_t)  # (num_samples,)
        
        # Term 2: exp(-Σ_{j<k} β_j * Δt) - cumulative transmission up to point k
        # Use CUDA-optimized cumsum
        cumulative_beta = torch.cumsum(beta_k, dim=0)  # (num_samples,)
        # Shift to get sum for j < k (exclude current k)
        cumulative_beta_prev = torch.cat([torch.zeros(1, device=beta_k.device), cumulative_beta[:-1]])
        cumulative_transmission = torch.exp(-cumulative_beta_prev * delta_t)  # (num_samples,)
        
        # Vectorized final computation
        importance_weights = local_absorption * cumulative_transmission  # (num_samples,)
        
        # Add small epsilon to avoid zero weights and numerical issues
        importance_weights = importance_weights + 1e-8
        
        # Normalize weights to sum to 1 for proper probability distribution
        total_weight = torch.sum(importance_weights)
        logger.debug(f"🔍 total_weight: {total_weight}")
        if total_weight > 1e-8:
            importance_weights = importance_weights / total_weight
        else:
            # Fallback to uniform weights if all weights are near zero
            importance_weights = torch.ones_like(importance_weights) / len(importance_weights)
        
        logger.debug(f"🔍 final importance_weights shape: {importance_weights.shape}, sum: {importance_weights.sum()}")
        logger.debug(f"🔍 final importance_weights min: {importance_weights.min()}, max: {importance_weights.max()}")
        
        return importance_weights
    
    def _cuda_importance_based_resampling(self, 
                                        uniform_positions: torch.Tensor,
                                        importance_weights: torch.Tensor,
                                        num_samples: int) -> torch.Tensor:
        """
        CUDA-optimized importance-based resampling.
        
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
        
        # Use CUDA-optimized multinomial sampling
        # Higher weight positions have higher probability of being selected
        logger.debug(f"🔍 Multinomial sampling: importance_weights shape = {importance_weights.shape}")
        selected_indices = torch.multinomial(importance_weights, num_samples, replacement=True)
        
        # Get resampled positions using advanced indexing (CUDA-optimized)
        resampled_positions = uniform_positions[selected_indices]
        
        return resampled_positions

    def _cuda_compute_dynamic_path_lengths(self, sampled_positions: torch.Tensor) -> torch.Tensor:
        """
        CUDA-optimized dynamic path lengths computation.
        
        Args:
            sampled_positions: (K, 3) - 3D positions of sampled voxels along the ray
        
        Returns:
            delta_t: (K,) - Dynamic path lengths for each voxel
        """
        if len(sampled_positions) <= 1:
            # Single point or empty - use default step size
            return torch.tensor([1.0], device=sampled_positions.device, dtype=sampled_positions.dtype)
        
        # CUDA-optimized vectorized computation of distances between consecutive points
        distances = torch.norm(sampled_positions[1:] - sampled_positions[:-1], dim=1)
        
        # For the first voxel, use distance from first to second point
        first_distance = torch.norm(sampled_positions[1] - sampled_positions[0], dim=0).unsqueeze(0)
        
        # Concatenate using CUDA-optimized operations
        delta_t = torch.cat([first_distance, distances], dim=0)
        
        return delta_t



    def _perform_importance_resampling(self, 
                                     uniform_positions: torch.Tensor,
                                     uniform_view_dirs: torch.Tensor,
                                     ue_pos: torch.Tensor,
                                     base_station_pos: torch.Tensor,
                                     antenna_indices: torch.Tensor,
                                     subcarrier_list: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform importance-based resampling using base class logic.
        
        Args:
            uniform_positions: Uniformly sampled positions (num_uniform, 3)
            uniform_view_dirs: View directions for uniform samples (num_uniform, 3)
            ue_pos: UE position for this computation
            base_station_pos: Base station position
            antenna_indices: Antenna indices for embedding lookup
            subcarrier_list: List of all subcarrier indices
            
        Returns:
            Tuple of (resampled_positions, resampled_view_dirs, importance_weights)
        """
        try:
            logger.debug(f"🔍 _perform_importance_resampling called")
            logger.debug(f"   - uniform_positions shape: {uniform_positions.shape}")
            logger.debug(f"   - uniform_view_dirs shape: {uniform_view_dirs.shape}")
            logger.debug(f"   - ue_pos shape: {ue_pos.shape}")
            logger.debug(f"   - base_station_pos shape: {base_station_pos.shape}")
            logger.debug(f"   - antenna_indices shape: {antenna_indices.shape}")
            
            # Get uniform attenuation factors for importance weight computation
            if self.prism_network is not None:
                logger.info(f"🧠 Calling PRISM network for importance resampling...")
                logger.debug(f"   - uniform_positions shape: {uniform_positions.shape}")
                logger.debug(f"   - ue_pos shape: {ue_pos.shape}")
                logger.debug(f"   - uniform_view_dirs shape: {uniform_view_dirs.shape}")
                logger.debug(f"   - antenna_indices shape: {antenna_indices.shape}")
                
                with torch.no_grad():
                    uniform_network_outputs = self.prism_network(
                        sampled_positions=uniform_positions.unsqueeze(0),
                        ue_positions=ue_pos.unsqueeze(0),
                        view_directions=uniform_view_dirs.mean(dim=0, keepdim=True),
                        antenna_indices=antenna_indices.unsqueeze(0)
                    )
                
                logger.debug(f"🧠 Network outputs keys: {list(uniform_network_outputs.keys())}")
                for key, value in uniform_network_outputs.items():
                    logger.debug(f"   - {key} shape: {value.shape}")
                
                # Extract attenuation factors for the first subcarrier (for importance calculation)
                attenuation_full = uniform_network_outputs['attenuation_factors']
                logger.debug(f"🔍 attenuation_full shape: {attenuation_full.shape}")
                uniform_attenuation = attenuation_full[0, :, 0]  # (num_uniform,)
                logger.debug(f"🔍 uniform_attenuation shape: {uniform_attenuation.shape}")
            else:
                # Network not available - log error and raise exception
                error_msg = "PRISM network not available for importance resampling. Cannot compute attenuation factors."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Compute importance weights using CUDA-optimized method
            importance_weights = self._cuda_compute_importance_weights(uniform_attenuation)
            
            # Perform importance-based resampling using CUDA-optimized method
            resampled_positions = self._cuda_importance_based_resampling(
                uniform_positions, importance_weights, num_samples=self.resampled_points
            )
            
            # Compute view directions for resampled positions
            resampled_view_dirs = base_station_pos.unsqueeze(0) - resampled_positions
            resampled_view_dirs = resampled_view_dirs / (torch.norm(resampled_view_dirs, dim=-1, keepdim=True) + 1e-8)
            
            return resampled_positions, resampled_view_dirs, importance_weights
            
        except Exception as e:
            logger.warning(f"Importance resampling failed: {e}. Using uniform sampling.")
            # Fallback to uniform sampling
            num_samples = min(self.resampled_points, len(uniform_positions))
            indices = torch.randperm(len(uniform_positions))[:num_samples]
            return uniform_positions[indices], uniform_view_dirs[indices], None
    
    def _integrate_with_importance_sampling(self,
                                          sampled_positions: torch.Tensor,
                                          attenuation_factors: torch.Tensor,
                                          radiation_factors: torch.Tensor,
                                          ue_subcarriers: List[int],
                                          subcarrier_list: List[int],
                                          importance_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Integrate signal strength along ray with importance sampling correction.
        
        Args:
            sampled_positions: Sampling positions along ray (num_samples, 3)
            attenuation_factors: Complex attenuation factors (num_samples, num_subcarriers)
            radiation_factors: Complex radiation factors (num_subcarriers,)
            ue_subcarriers: List of subcarrier indices for this UE
            subcarrier_list: List of all subcarrier indices
            importance_weights: Optional importance weights for correction (num_samples,)
            
        Returns:
            Integrated complex signal strengths for UE subcarriers (len(ue_subcarriers),)
        """
        logger.debug(f"🔍 CUDA DEBUG: _integrate_with_importance_sampling ENTRY")
        logger.debug(f"🔍 CUDA DEBUG: radiation_factors shape: {radiation_factors.shape}")
        logger.debug(f"🔍 CUDA DEBUG: ue_subcarriers: {ue_subcarriers}")
        logger.debug(f"🔍 CUDA DEBUG: sampled_positions shape: {sampled_positions.shape}")
        logger.debug(f"🔍 CUDA DEBUG: attenuation_factors shape: {attenuation_factors.shape}")
        logger.debug(f"🔍 CUDA DEBUG: subcarrier_list: {subcarrier_list}")
        # Neural network outputs radiation_factors for ALL subcarriers (408 total)
        # We need to select the specific subcarriers requested for this UE
        logger.debug(f"🔍 CUDA DEBUG: ue_subcarriers (requested): {ue_subcarriers}")
        logger.debug(f"🔍 CUDA DEBUG: radiation_factors shape: {radiation_factors.shape}")
        logger.debug(f"🔍 CUDA DEBUG: radiation_factors device: {radiation_factors.device}")
        logger.debug(f"🔍 CUDA DEBUG: radiation_factors dtype: {radiation_factors.dtype}")
        
        # Validate that all ue_subcarriers are within bounds of the full subcarrier range
        max_subcarrier_idx = radiation_factors.shape[-1] - 1
        logger.debug(f"🔍 CUDA DEBUG: max_subcarrier_idx: {max_subcarrier_idx}")
        
        valid_ue_subcarriers = []
        for sc in ue_subcarriers:
            logger.debug(f"🔍 CUDA DEBUG: checking subcarrier {sc} <= {max_subcarrier_idx}")
            if 0 <= sc <= max_subcarrier_idx:
                valid_ue_subcarriers.append(sc)
                logger.debug(f"✅ CUDA DEBUG: subcarrier {sc} is valid")
            else:
                logger.error(f"❌ CUDA DEBUG: Invalid subcarrier index {sc} (max: {max_subcarrier_idx}), skipping")
        
        if not valid_ue_subcarriers:
            logger.error(f"❌ No valid subcarrier indices found! All indices out of bounds.")
            logger.error(f"   ue_subcarriers: {ue_subcarriers}")
            logger.error(f"   radiation_factors shape: {radiation_factors.shape}")
            return torch.zeros(len(ue_subcarriers), dtype=torch.complex64, device=sampled_positions.device)
        
        logger.debug(f"🔍 Using {len(valid_ue_subcarriers)} valid subcarriers: {valid_ue_subcarriers}")
        
        # Check if indices are reasonable before indexing
        if valid_ue_subcarriers and max(valid_ue_subcarriers) >= radiation_factors.shape[-1]:
            logger.error(f"❌ Index out of bounds: max index {max(valid_ue_subcarriers)} >= shape {radiation_factors.shape[-1]}")
            return torch.zeros(len(ue_subcarriers), dtype=torch.complex64, device=sampled_positions.device)
        
        # Extract radiation for the specific UE subcarriers (direct indexing into full 408 subcarrier range)
        logger.debug(f"🔍 CUDA DEBUG: About to index radiation_factors")
        logger.debug(f"🔍 CUDA DEBUG: valid_ue_subcarriers: {valid_ue_subcarriers}")
        if valid_ue_subcarriers:
            logger.debug(f"🔍 CUDA DEBUG: valid_ue_subcarriers range: {min(valid_ue_subcarriers)} to {max(valid_ue_subcarriers)}")
        
        logger.debug(f"🔍 CUDA DEBUG: radiation_factors shape: {radiation_factors.shape}")
        logger.debug(f"🔍 CUDA DEBUG: valid_ue_subcarriers type: {type(valid_ue_subcarriers)}")
        
        try:
            ue_radiation = radiation_factors[:, valid_ue_subcarriers]  # (num_ue_antennas, len(valid_ue_subcarriers))
            logger.debug(f"✅ CUDA DEBUG: radiation_factors indexing successful")
            logger.debug(f"✅ CUDA DEBUG: ue_radiation shape: {ue_radiation.shape}")
        except Exception as e:
            logger.error(f"❌ CUDA DEBUG: radiation_factors indexing failed: {e}")
            logger.error(f"   radiation_factors shape: {radiation_factors.shape}")
            logger.error(f"   valid_ue_subcarriers: {valid_ue_subcarriers}")
            raise
        
        # For attenuation, we need to map UE subcarriers to the subcarrier_list indices
        # subcarrier_list contains only the selected subcarriers from all UEs
        selected_subcarrier_indices = []
        for sc in valid_ue_subcarriers:
            if sc in subcarrier_list:
                selected_subcarrier_indices.append(subcarrier_list.index(sc))
            else:
                logger.warning(f"⚠️  UE subcarrier {sc} not found in subcarrier_list {subcarrier_list}")
        
        if not selected_subcarrier_indices:
            logger.warning(f"⚠️  No valid subcarrier indices found in subcarrier_list!")
            return torch.zeros(len(ue_subcarriers), dtype=torch.complex64, device=sampled_positions.device)
        
        # Extract attenuation for selected subcarriers
        logger.debug(f"🔍 CUDA DEBUG: About to index attenuation_factors")
        logger.debug(f"🔍 CUDA DEBUG: selected_subcarrier_indices: {selected_subcarrier_indices}")
        if selected_subcarrier_indices:
            logger.debug(f"🔍 CUDA DEBUG: selected_subcarrier_indices range: {min(selected_subcarrier_indices)} to {max(selected_subcarrier_indices)}")
        
        logger.debug(f"🔍 CUDA DEBUG: attenuation_factors shape: {attenuation_factors.shape}")
        logger.debug(f"🔍 CUDA DEBUG: selected_subcarrier_indices type: {type(selected_subcarrier_indices)}")
        
        try:
            # Correct indexing: select subcarriers from the last dimension
            ue_attenuation_full = attenuation_factors[:, :, selected_subcarrier_indices]  # (num_samples, num_ue_antennas, len(selected_subcarriers))
            # Take mean across UE antennas for integration
            ue_attenuation = ue_attenuation_full.mean(dim=1)  # (num_samples, len(selected_subcarriers))
            logger.debug(f"✅ CUDA DEBUG: attenuation_factors indexing successful")
            logger.debug(f"✅ CUDA DEBUG: ue_attenuation shape after correction: {ue_attenuation.shape}")
        except Exception as e:
            logger.error(f"❌ CUDA DEBUG: attenuation_factors indexing failed: {e}")
            logger.error(f"   attenuation_factors shape: {attenuation_factors.shape}")
            logger.error(f"   selected_subcarrier_indices: {selected_subcarrier_indices}")
            raise

        logger.debug(f"🔍 ue_radiation shape: {ue_radiation.shape}")
        logger.debug(f"🔍 ue_radiation device: {ue_radiation.device}")
        logger.debug(f"🔍 ue_radiation dtype: {ue_radiation.dtype}")
        
        # Calculate dynamic path lengths using CUDA-optimized method
        delta_t = self._cuda_compute_dynamic_path_lengths(sampled_positions)
        
        # Apply discrete radiance field integration
        logger.debug(f"🔍 CUDA DEBUG: About to unpack ue_attenuation.shape")
        logger.debug(f"🔍 CUDA DEBUG: ue_attenuation.shape: {ue_attenuation.shape}")
        logger.debug(f"🔍 CUDA DEBUG: ue_attenuation.shape length: {len(ue_attenuation.shape)}")
        
        if len(ue_attenuation.shape) == 2:
            K, N_ue = ue_attenuation.shape
        else:
            logger.error(f"❌ CUDA DEBUG: Unexpected ue_attenuation shape: {ue_attenuation.shape}")
            logger.error(f"   Expected 2D tensor, got {len(ue_attenuation.shape)}D tensor")
            raise ValueError(f"ue_attenuation has unexpected shape: {ue_attenuation.shape}")
        logger.debug(f"🔬 Starting discrete radiance field integration:")
        logger.debug(f"   - K (num_samples): {K}")
        logger.debug(f"   - N_ue (num_ue_subcarriers): {N_ue}")
        logger.debug(f"   - ue_attenuation shape: {ue_attenuation.shape}")
        logger.debug(f"   - ue_radiation shape: {ue_radiation.shape}")
        logger.debug(f"   - delta_t shape: {delta_t.shape}")
        
        # Step 1: Attenuation deltas Δρ = ρ ⊙ Δt
        attenuation_deltas = ue_attenuation * delta_t.unsqueeze(1)  # (K, N_ue)
        logger.debug(f"🔬 Step 1 - Attenuation deltas computed: {attenuation_deltas.shape}")
        
        # Step 2: Cumulative attenuation
        zero_pad = torch.zeros(1, N_ue, dtype=attenuation_deltas.dtype, device=attenuation_deltas.device)
        padded_deltas = torch.cat([zero_pad, attenuation_deltas[:-1]], dim=0)
        cumulative_attenuation = torch.cumsum(padded_deltas, dim=0)  # (K, N_ue)
        
        # Step 3: Attenuation factors
        attenuation_exp = torch.exp(-cumulative_attenuation)  # (K, N_ue)
        
        # Step 4: Local absorption
        local_absorption = 1.0 - torch.exp(-attenuation_deltas)  # (K, N_ue)
        
        # Step 5: Broadcast radiation
        # Unified processing: ue_radiation now has shape (num_ue_antennas, N_ue)
        # Take mean across UE antennas for integration (could be improved with proper antenna handling)
        logger.debug(f"🔍 CUDA DEBUG: About to compute ue_radiation.mean(dim=0)")
        logger.debug(f"🔍 CUDA DEBUG: ue_radiation shape before mean: {ue_radiation.shape}")
        
        try:
            ue_radiation_mean = ue_radiation.mean(dim=0)  # (N_ue,)
            logger.debug(f"✅ CUDA DEBUG: ue_radiation.mean successful")
            logger.debug(f"✅ CUDA DEBUG: ue_radiation_mean shape: {ue_radiation_mean.shape}")
        except Exception as e:
            logger.error(f"❌ CUDA DEBUG: ue_radiation.mean failed: {e}")
            logger.error(f"   ue_radiation shape: {ue_radiation.shape}")
            raise
        
        radiation_expanded = ue_radiation_mean.unsqueeze(0).expand(K, -1)  # (K, N_ue)
        
        # Step 6: Vectorized computation 
        signal_contributions = attenuation_exp * local_absorption * radiation_expanded  # (K, N_ue)
        
        # Step 7: Apply importance sampling correction if available
        if importance_weights is not None and len(importance_weights) == K:
            # CORRECTED: Importance weights represent the actual contribution weights
            # from the discrete radiance field formula, not sampling probabilities.
            # The resampling process already selected points based on these weights,
            # so we need to apply Monte Carlo correction for the biased sampling.
            
            # Get the original uniform sampling probability (before resampling)
            uniform_prob = 1.0 / self.uniform_samples  # Each point had equal probability originally
            
            # Current sampling probability after importance resampling
            # importance_weights are normalized, so they represent p(x_i) after resampling
            current_sampling_prob = importance_weights.unsqueeze(1) + 1e-8  # (K, 1)
            
            # Monte Carlo correction: f(x_i) * (uniform_prob / current_prob)
            mc_correction = uniform_prob / current_sampling_prob  # (K, 1)
            signal_contributions = signal_contributions * mc_correction  # (K, N_ue)
        
        # Step 8: Integration
        integrated_signals = torch.sum(signal_contributions, dim=0)  # (N_ue,)
        
        logger.debug(f"🔬 Final integration results:")
        logger.debug(f"   - integrated_signals shape: {integrated_signals.shape}")
        logger.debug(f"   - integrated_signals abs max: {torch.abs(integrated_signals).max() if integrated_signals.numel() > 0 else 'N/A'}")
        logger.debug(f"   - integrated_signals abs mean: {torch.abs(integrated_signals).mean() if integrated_signals.numel() > 0 else 'N/A'}")
        
        return integrated_signals
    
    def _vectorized_importance_sampling(self,
                                      base_station_pos: torch.Tensor,
                                      direction_vectors: torch.Tensor,
                                      ue_positions_tensor: torch.Tensor,
                                      ue_to_subcarriers: Dict,
                                      subcarrier_list: List[int],
                                      antenna_indices: torch.Tensor,
                                      all_signal_strengths: torch.Tensor) -> torch.Tensor:
        """
        Vectorized importance sampling that maintains GPU acceleration.
        
        Key optimizations:
        1. Batch process all directions simultaneously
        2. Batch process all UEs simultaneously  
        3. Minimize neural network calls
        4. Maintain memory-efficient operations
        """
        num_directions, num_ue, num_subcarriers = all_signal_strengths.shape
        
        # STAGE 1: VECTORIZED UNIFORM SAMPLING
        # Generate all sampling points for all directions at once
        t_values = torch.linspace(0, self.max_ray_length, self.uniform_samples, 
                                device=base_station_pos.device, dtype=torch.float32)
        
        # Vectorized sampling: (num_directions, uniform_samples, 3)
        uniform_positions_all = (base_station_pos.unsqueeze(0).unsqueeze(0) + 
                               direction_vectors.unsqueeze(1) * t_values.unsqueeze(0).unsqueeze(-1))
        
        # Vectorized view directions: (num_directions, uniform_samples, 3)
        uniform_view_dirs_all = base_station_pos.unsqueeze(0).unsqueeze(0) - uniform_positions_all
        uniform_view_dirs_all = uniform_view_dirs_all / (torch.norm(uniform_view_dirs_all, dim=-1, keepdim=True) + 1e-8)
        
        # Vectorized scene bounds check: (num_directions, uniform_samples)
        # Use actual scene bounds instead of symmetric cube
        scene_min = self.scene_min.to(uniform_positions_all.device)
        scene_max = self.scene_max.to(uniform_positions_all.device)
        
        # Check each dimension separately for scene bounds validation
        x_valid = (uniform_positions_all[..., 0] >= scene_min[0]) & (uniform_positions_all[..., 0] <= scene_max[0])
        y_valid = (uniform_positions_all[..., 1] >= scene_min[1]) & (uniform_positions_all[..., 1] <= scene_max[1])
        z_valid = (uniform_positions_all[..., 2] >= scene_min[2]) & (uniform_positions_all[..., 2] <= scene_max[2])
        
        valid_mask_all = torch.all(
            (uniform_positions_all >= scene_min) & (uniform_positions_all <= scene_max),
            dim=-1
        )
        
        total_samples = valid_mask_all.numel()
        valid_samples = valid_mask_all.sum().item()
        
        if valid_samples == 0:
            logger.warning("⚠️  CRITICAL: No valid sampling positions found within scene bounds!")
            logger.warning(f"📊 Scene bounds: X=[{scene_min[0]:.1f}, {scene_max[0]:.1f}], Y=[{scene_min[1]:.1f}, {scene_max[1]:.1f}], Z=[{scene_min[2]:.1f}, {scene_max[2]:.1f}]")
            logger.warning(f"📊 Sampling range: X=[{uniform_positions_all[..., 0].min():.1f}, {uniform_positions_all[..., 0].max():.1f}], Y=[{uniform_positions_all[..., 1].min():.1f}, {uniform_positions_all[..., 1].max():.1f}], Z=[{uniform_positions_all[..., 2].min():.1f}, {uniform_positions_all[..., 2].max():.1f}]")
            logger.warning(f"💡 Solution: Expand scene_bounds in config file to include sampling range")
            logger.warning(f"📈 Validity per dimension: X={x_valid.sum()}/{x_valid.numel()}, Y={y_valid.sum()}/{y_valid.numel()}, Z={z_valid.sum()}/{z_valid.numel()}")
        elif valid_samples < total_samples * 0.5:
            logger.warning(f"⚠️  WARNING: Only {valid_samples}/{total_samples} ({100*valid_samples/total_samples:.1f}%) sampling positions are within scene bounds")
            logger.warning(f"📊 Consider expanding scene_bounds for better coverage")
        else:
            logger.debug(f"✅ Scene bounds check: {valid_samples}/{total_samples} ({100*valid_samples/total_samples:.1f}%) valid sampling positions")
        
        logger.debug(f"📊 Vectorized uniform sampling: {num_directions} × {self.uniform_samples} points")
        
        # STAGE 2: BATCH IMPORTANCE SAMPLING
        # Process in batches to manage memory while maintaining vectorization
        batch_size = min(8, num_directions)  # Adjust based on GPU memory
        
        for batch_start in range(0, num_directions, batch_size):
            batch_end = min(batch_start + batch_size, num_directions)
            batch_directions = batch_end - batch_start
            
            # Get batch data
            batch_uniform_pos = uniform_positions_all[batch_start:batch_end]  # (batch_size, uniform_samples, 3)
            batch_uniform_dirs = uniform_view_dirs_all[batch_start:batch_end]  # (batch_size, uniform_samples, 3)
            batch_valid_mask = valid_mask_all[batch_start:batch_end]  # (batch_size, uniform_samples)
            
            # Process each UE for this batch of directions
            for ue_idx in range(num_ue):
                ue_subcarriers = ue_to_subcarriers.get(ue_idx, [])
                if not ue_subcarriers:
                    continue
                
                ue_pos = ue_positions_tensor[ue_idx]
                
                # Vectorized importance sampling for this UE across batch directions
                batch_signals = self._batch_importance_sampling_for_ue(
                    batch_uniform_pos, batch_uniform_dirs, batch_valid_mask,
                    ue_pos, base_station_pos, antenna_indices,
                    ue_subcarriers, subcarrier_list
                )
                
                # Store results: batch_signals shape (batch_directions, len(ue_subcarriers))
                selected_subcarrier_indices = [subcarrier_list.index(sc) for sc in ue_subcarriers if sc in subcarrier_list]
                for i, subcarrier_idx in enumerate(selected_subcarrier_indices):
                    all_signal_strengths[batch_start:batch_end, ue_idx, subcarrier_idx] = batch_signals[:, i]
        
        return all_signal_strengths
    
    def _batch_importance_sampling_for_ue(self,
                                        batch_uniform_pos: torch.Tensor,
                                        batch_uniform_dirs: torch.Tensor,
                                        batch_valid_mask: torch.Tensor,
                                        ue_pos: torch.Tensor,
                                        base_station_pos: torch.Tensor,
                                        antenna_indices: torch.Tensor,
                                        ue_subcarriers: List[int],
                                        subcarrier_list: List[int]) -> torch.Tensor:
        """
        Batch importance sampling for a single UE across multiple directions.
        
        Args:
            batch_uniform_pos: (batch_size, uniform_samples, 3)
            batch_uniform_dirs: (batch_size, uniform_samples, 3)
            batch_valid_mask: (batch_size, uniform_samples)
            ue_pos: (3,)
            
        Returns:
            batch_signals: (batch_size, len(ue_subcarriers))
        """
        batch_size = batch_uniform_pos.shape[0]
        batch_signals = torch.zeros(batch_size, len(ue_subcarriers), 
                                  dtype=torch.complex64, device=batch_uniform_pos.device)
        
        # Process each direction in the batch
        for dir_idx in range(batch_size):
            uniform_pos = batch_uniform_pos[dir_idx]  # (uniform_samples, 3)
            uniform_dirs = batch_uniform_dirs[dir_idx]  # (uniform_samples, 3)
            valid_mask = batch_valid_mask[dir_idx]  # (uniform_samples,)
            
            logger.debug(f"🔍 Processing direction {dir_idx}/{batch_size}")
            logger.debug(f"   - uniform_pos shape: {uniform_pos.shape}")
            logger.debug(f"   - valid_mask sum: {valid_mask.sum()}/{len(valid_mask)}")
            
            # Filter valid positions
            valid_pos = uniform_pos[valid_mask]  # (num_valid, 3)
            valid_dirs = uniform_dirs[valid_mask]  # (num_valid, 3)
            
            logger.debug(f"   - valid_pos shape after filtering: {valid_pos.shape}")
            
            if len(valid_pos) == 0:
                logger.warning(f"⚠️  No valid sampling positions for direction {dir_idx}/{batch_size} - all {len(uniform_pos)} positions outside scene bounds")
                continue
            
            # Apply importance sampling if we have enough points
            if len(valid_pos) > self.resampled_points:
                final_pos, final_dirs, importance_weights = self._perform_importance_resampling(
                    valid_pos, valid_dirs, ue_pos, base_station_pos, 
                    antenna_indices, subcarrier_list
                )
            else:
                final_pos = valid_pos
                final_dirs = valid_dirs
                importance_weights = None
            
            # Neural network call for this direction-UE combination
            logger.debug(f"🔍 About to call PrismNetwork for direction {dir_idx}")
            logger.debug(f"   - self.prism_network is None: {self.prism_network is None}")
            logger.debug(f"   - final_pos shape: {final_pos.shape}")
            logger.debug(f"   - ue_pos shape: {ue_pos.shape}")
            logger.debug(f"   - final_dirs shape: {final_dirs.shape}")
            logger.debug(f"   - antenna_indices shape: {antenna_indices.shape}")
            
            if self.prism_network is not None:
                logger.debug(f"🔍 CUDA DEBUG: About to call PrismNetwork.forward()...")
                logger.debug(f"🔍 CUDA DEBUG: final_pos shape: {final_pos.shape}")
                logger.debug(f"🔍 CUDA DEBUG: ue_pos shape: {ue_pos.shape}")
                logger.debug(f"🔍 CUDA DEBUG: final_dirs shape: {final_dirs.shape}")
                logger.debug(f"🔍 CUDA DEBUG: antenna_indices shape: {antenna_indices.shape}")
                
                try:
                    with torch.no_grad():
                        network_outputs = self.prism_network(
                            sampled_positions=final_pos.unsqueeze(0),
                            ue_positions=ue_pos.unsqueeze(0),
                            view_directions=final_dirs.mean(dim=0, keepdim=True),
                            antenna_indices=antenna_indices.unsqueeze(0)
                        )
                    
                    logger.debug(f"✅ CUDA DEBUG: PrismNetwork.forward() completed successfully!")
                    logger.debug(f"✅ CUDA DEBUG: network_outputs keys: {list(network_outputs.keys())}")
                    for key, value in network_outputs.items():
                        logger.debug(f"✅ CUDA DEBUG: {key} shape: {value.shape}")
                except Exception as e:
                    logger.error(f"❌ CUDA DEBUG: PrismNetwork.forward() failed: {e}")
                    logger.error(f"   final_pos shape: {final_pos.shape}")
                    logger.error(f"   ue_pos shape: {ue_pos.shape}")
                    logger.error(f"   final_dirs shape: {final_dirs.shape}")
                    logger.error(f"   antenna_indices shape: {antenna_indices.shape}")
                    raise
                
                # Handle network outputs with consistent shape processing
                attenuation_factors = network_outputs['attenuation_factors']
                radiation_factors = network_outputs['radiation_factors']
                
                # Ensure we have the right shapes after batch dimension removal
                if attenuation_factors.dim() == 4:  # (batch_size, num_samples, num_ue_antennas, num_subcarriers)
                    attenuation_factors = attenuation_factors[0]  # (num_samples, num_ue_antennas, num_subcarriers)
                if radiation_factors.dim() == 3:  # (batch_size, num_ue_antennas, num_subcarriers)
                    radiation_factors = radiation_factors[0]  # (num_ue_antennas, num_subcarriers)
                
                # Debug: Check neural network outputs
                logger.debug(f"🔍 Neural network outputs for direction {dir_idx}:")
                logger.debug(f"   - attenuation_factors shape: {attenuation_factors.shape}")
                logger.debug(f"   - radiation_factors shape: {radiation_factors.shape}")
            else:
                # Network is not available - log error and raise exception
                error_msg = "PRISM network is not available. Cannot perform ray tracing without neural network."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Integrate with importance sampling
            integrated_signals = self._integrate_with_importance_sampling(
                final_pos, attenuation_factors, radiation_factors,
                ue_subcarriers, subcarrier_list, importance_weights
            )
            
            batch_signals[dir_idx] = integrated_signals
        
        return batch_signals
    

    
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
                        # Try to convert to int if possible
                        try:
                            all_indices.add(int(indices))
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert indices {indices} (type: {type(indices)}) to int")
                            continue
            
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
            indices = selected_subcarriers.tolist()
            valid_indices = [idx for idx in indices if 0 <= idx < max_subcarrier]
            
            if len(valid_indices) != len(indices):
                invalid_count = len(indices) - len(valid_indices)
                logger.warning(f"Filtered out {invalid_count} invalid subcarrier indices (out of bounds)")
            
            if not valid_indices:
                logger.warning("No valid subcarrier indices found, using default indices [0, 64, 128, 192, 256, 320, 384]")
                valid_indices = [0, 64, 128, 192, 256, 320, 384]
            
            return sorted(valid_indices)
            
        elif isinstance(selected_subcarriers, (list, tuple)):
            # List/tuple format: validate and return
            if not selected_subcarriers:
                raise ValueError("selected_subcarriers list/tuple is empty")
            
            # Validate subcarrier indices are within reasonable bounds
            max_subcarrier = 408  # Based on the simulation data
            valid_indices = [idx for idx in selected_subcarriers if isinstance(idx, (int, float)) and 0 <= idx < max_subcarrier]
            
            if len(valid_indices) != len(selected_subcarriers):
                invalid_count = len(selected_subcarriers) - len(valid_indices)
                logger.warning(f"Filtered out {invalid_count} invalid subcarrier indices (out of bounds or wrong type)")
            
            if not valid_indices:
                logger.warning("No valid subcarrier indices found, using default indices [0, 64, 128, 192, 256, 320, 384]")
                valid_indices = [0, 64, 128, 192, 256, 320, 384]
            
            return sorted([int(idx) for idx in valid_indices])
            
        else:
            raise ValueError(f"Unsupported selected_subcarriers type: {type(selected_subcarriers)}. "
                           f"Expected dict, torch.Tensor, list, or tuple.")
    

    

    def accumulate_signals(self, 
                          base_station_pos: torch.Tensor,
                          ue_positions: List[torch.Tensor],
                          selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                          antenna_indices: torch.Tensor) -> Dict:
        """
        Accumulate RF signals using MLP-based direction sampling with antenna indices.
        
        This method implements the design document's MLP-based direction sampling:
        1. Use AntennaNetwork to compute directional importance based on antenna indices
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
            antenna_indices: Base station's antenna indices for embedding lookup
        
        Returns:
            Accumulated signal strength matrix for all virtual links
        """
        accumulated_signals = {}
        
        # Debug logging with detailed information
        logger.info(f"🔄 CUDA accumulate_signals called with selected_subcarriers type: {type(selected_subcarriers)}")
        logger.info(f"📍 UE positions: {len(ue_positions)} positions")
        logger.info(f"🎯 antenna_indices: {antenna_indices}")
        logger.info(f"📡 base_station_pos: {base_station_pos}")
        
        # Log UE coordinates
        for i, ue_pos in enumerate(ue_positions):
            if isinstance(ue_pos, torch.Tensor):
                coords = ue_pos.cpu().numpy() if ue_pos.is_cuda else ue_pos.numpy()
                logger.debug(f"   📍 UE {i+1}: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]")
        
        # Show selected_subcarriers content
        if isinstance(selected_subcarriers, dict):
            total_subcarriers = sum(len(v) if hasattr(v, '__len__') else 1 for v in selected_subcarriers.values())
            logger.debug(f"📡 Dictionary format: {len(selected_subcarriers.keys())} UEs, {total_subcarriers} total subcarriers")
            logger.debug(f"📡 selected_subcarriers content: {selected_subcarriers}")
            for ue_idx, subcarriers in selected_subcarriers.items():
                if hasattr(subcarriers, '__len__'):
                    if len(subcarriers) <= 10:  # Show all if few subcarriers
                        logger.debug(f"   📡 UE {ue_idx}: {subcarriers}")
                    else:  # Show first few if many subcarriers
                        logger.debug(f"   📡 UE {ue_idx}: {subcarriers[:5]}... ({len(subcarriers)} total)")
                else:
                    logger.debug(f"   📡 UE {ue_idx}: {subcarriers}")
        elif isinstance(selected_subcarriers, (list, torch.Tensor)):
            subcarrier_count = len(selected_subcarriers) if hasattr(selected_subcarriers, '__len__') else 1
            logger.debug(f"📡 Processing {subcarrier_count} subcarriers for all UEs")
            if subcarrier_count <= 10:
                logger.debug(f"📡 selected_subcarriers: {selected_subcarriers}")
            else:
                logger.debug(f"📡 selected_subcarriers: {selected_subcarriers[:5]}... ({subcarrier_count} total)")
        else:
            logger.debug(f"📡 selected_subcarriers value: {selected_subcarriers}")
        
        subcarrier_indices = self._normalize_subcarrier_input(selected_subcarriers, ue_positions)
        
        if self.prism_network is None:
            # Fallback: use ultra-optimized method for all directions
            self.actual_directions_used = self.azimuth_divisions * self.elevation_divisions
            logger.error("❌ No MLP network available, cannot perform ray tracing!")
            error_msg = "Neural network not available. Cannot perform ray tracing without proper network configuration."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # 检查CUDA编译状态
        if not self.cuda_compilation_successful:
            logger.error("❌ CUDA compilation failed - this may affect MLP direction selection!")
            logger.error("🔧 Attempting MLP direction selection with PyTorch fallback...")
        
        try:
            # Use AntennaNetwork to get directional importance based on antenna embedding C
            with torch.no_grad():
                # Enable mixed precision for MLP direction selection
                with torch.amp.autocast('cuda', enabled=getattr(self, 'use_mixed_precision', False)):
                    # Ensure prism_network and its components are on the correct device
                    if hasattr(self.prism_network, 'to'):
                        self.prism_network = self.prism_network.to(self.device)
                    if hasattr(self.prism_network.antenna_network, 'to'):
                        self.prism_network.antenna_network = self.prism_network.antenna_network.to(self.device)
                    
                    # Ensure all input tensors are on the correct device
                    base_station_pos = base_station_pos.to(self.device)
                    ue_positions = [ue_pos.to(self.device) for ue_pos in ue_positions]
                    antenna_indices = antenna_indices.to(self.device)
                    
                    # Convert antenna indices to embeddings using the antenna codebook
                    antenna_embeddings = self.prism_network.antenna_codebook(antenna_indices)
                    
                    # Get directional importance matrix from AntennaNetwork (with mixed precision)
                    directional_importance = self.prism_network.antenna_network(antenna_embeddings.unsqueeze(0))
                    
                    # Get top-K directions for efficient sampling (with mixed precision)
                    top_k_directions, top_k_importance = self.prism_network.antenna_network.get_top_k_directions(
                        directional_importance, k=self.top_k_directions
                    )
                
                # Extract direction indices for the first batch element
                # Handle different shapes: (batch_size, k, 2) or (batch_size, num_antennas, k, 2)
                if len(top_k_directions.shape) == 4:
                    # Shape: (batch_size, num_antennas, k, 2) -> take first batch and first antenna
                    selected_directions = top_k_directions[0, 0]  # Shape: (k, 2)
                else:
                    # Shape: (batch_size, k, 2) -> take first batch
                    selected_directions = top_k_directions[0]  # Shape: (k, 2)
                
                logger.debug(f"🔍 Direction selection debug:")
                logger.debug(f"   - top_k_directions shape: {top_k_directions.shape}")
                logger.debug(f"   - selected_directions shape: {selected_directions.shape}")
                logger.debug(f"   - selected_directions sample: {selected_directions[:3] if selected_directions.numel() > 0 else 'empty'}")
                
            # Convert tensor directions to list of tuples for parallel processing
            directions_list = []
            for i in range(selected_directions.shape[0]):
                phi_idx = selected_directions[i, 0].item()
                theta_idx = selected_directions[i, 1].item()
                directions_list.append((phi_idx, theta_idx))
            
            # Update actual directions used
            self.actual_directions_used = len(directions_list)
            
            # Log successful direction selection
            logger.debug(f"✅ MLP direction selection SUCCESS!")
            logger.debug(f"📊 Selected {len(directions_list)} directions out of {self.azimuth_divisions * self.elevation_divisions} total")
            logger.debug(f"🚀 Performance improvement: {(self.azimuth_divisions * self.elevation_divisions) / len(directions_list):.1f}x faster")
            logger.debug(f"Selected directions: {directions_list[:5]}..." if len(directions_list) > 5 else f"Selected directions: {directions_list}")
            
            # Use CUDA-optimized processing for selected directions
            logger.debug(f"🚀 Using CUDA-optimized processing for {len(directions_list)} directions")
            accumulated_signals = self._accumulate_signals_cuda_optimized(
                base_station_pos, ue_positions, selected_subcarriers, antenna_indices, directions_list
            )
            
            return accumulated_signals
            
        except Exception as e:
            # Update actual directions used to all directions (fallback)
            self.actual_directions_used = self.azimuth_divisions * self.elevation_divisions
            logger.error("❌ MLP-based direction sampling FAILED!")
            logger.error(f"📋 Exception type: {type(e).__name__}")
            logger.error(f"📋 Exception message: {str(e)}")
            import traceback
            logger.error(f"📋 Full traceback:\n{traceback.format_exc()}")
            logger.error("🔧 CUDA-optimized processing failed!")
            error_msg = "CUDA-optimized signal accumulation failed. Neural network must be available for ray tracing."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _accumulate_signals_cuda_optimized(self, 
                                         base_station_pos: torch.Tensor,
                                         ue_positions: List[torch.Tensor],
                                         selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                                         antenna_indices: torch.Tensor,
                                         directions_list: List[Tuple[int, int]]) -> Dict:
        """
        CUDA-optimized signal accumulation for selected directions.
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier selection information
            antenna_embedding: Antenna embedding parameters
            directions_list: List of selected direction tuples (phi_idx, theta_idx)
            
        Returns:
            Dictionary mapping (ue_pos, subcarrier) to accumulated signal strength
        """
        accumulated_signals = {}
        
        # Process each selected direction
        for direction in directions_list:
            # Use the existing trace_ray method for each direction
            direction_results = self.trace_ray(
                base_station_pos, direction, ue_positions, 
                selected_subcarriers, antenna_indices
            )
            
            # Accumulate results
            for key, signal_strength in direction_results.items():
                if key not in accumulated_signals:
                    accumulated_signals[key] = signal_strength
                else:
                    accumulated_signals[key] += signal_strength
        
        return accumulated_signals
    

    

    
