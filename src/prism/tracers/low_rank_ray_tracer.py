"""
Low-Rank Ray Tracer with LRU Caching

This module provides LowRankRayTracer, an optimized ray tracer that implements
low-rank factorization and LRU caching for efficient CSI prediction.
"""

from typing import Dict, List, Tuple, Union, Optional
import torch
import torch.nn as nn
import logging
from collections import OrderedDict
import hashlib

logger = logging.getLogger(__name__)


class LowRankRayTracer(nn.Module):
    """
    Low-Rank Ray Tracer with LRU caching for efficient CSI prediction.
    
    This class implements advanced optimization strategies:
    1. Low-rank factorization of CSI tensors
    2. LRU caching for spatial computations
    3. Memory-efficient ray tracing
    """
    
    def __init__(self, 
                 prism_network,
                 enable_caching: bool = True,
                 enable_profiling: bool = False,
                 max_cache_size: int = 50):
        """
        Initialize the low-rank ray tracer.
        
        Args:
            prism_network: PrismNetwork instance for feature extraction
            enable_caching: Whether to enable LRU caching
            enable_profiling: Whether to enable performance profiling
            max_cache_size: Maximum number of cached entries
        """
        super().__init__()
        
        if prism_network is None:
            raise ValueError("PrismNetwork is required")
        
        self.prism_network = prism_network
        self.enable_caching = enable_caching
        self.enable_profiling = enable_profiling
        self.max_cache_size = max_cache_size
        
        # Get configuration from PrismNetwork
        self.azimuth_divisions = prism_network.azimuth_divisions
        self.elevation_divisions = prism_network.elevation_divisions
        self.max_ray_length = prism_network.max_ray_length
        self.num_sampling_points = prism_network.num_sampling_points
        self.num_subcarriers = prism_network.num_subcarriers
        self.num_bs_antennas = prism_network.num_bs_antennas
        
        # Calculate derived parameters
        self.azimuth_resolution = 2 * torch.pi / self.azimuth_divisions
        self.elevation_resolution = (torch.pi / 2) / self.elevation_divisions
        self.total_directions = self.azimuth_divisions * self.elevation_divisions
        
        # Initialize LRU cache for spatial computations
        if self.enable_caching:
            self._spatial_cache = OrderedDict()
            logger.info(f"🗄️ LRU cache initialized with max_size={max_cache_size}")
        
        # Profiling counters
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"🚀 LowRankRayTracer initialized")
        logger.info(f"   - PrismNetwork: {type(prism_network).__name__}")
        logger.info(f"   - Caching: {'enabled' if enable_caching else 'disabled'}")
        logger.info(f"   - Profiling: {'enabled' if enable_profiling else 'disabled'}")
        logger.info(f"   - Subcarriers: {self.num_subcarriers}")
        logger.info(f"   - BS Antennas: {self.num_bs_antennas}")
        logger.info(f"   - Directions: {self.azimuth_divisions}×{self.elevation_divisions} = {self.total_directions}")
    
    def _generate_cache_key(self, bs_pos: torch.Tensor, ue_pos: torch.Tensor, ant_idx: int) -> str:
        """Generate a cache key for spatial computations."""
        # Create a hash of the input parameters
        key_data = f"{bs_pos.cpu().numpy().tobytes()}_{ue_pos.cpu().numpy().tobytes()}_{ant_idx}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _compute_or_cache_spatial_tensors(self, bs_pos: torch.Tensor, ue_pos: torch.Tensor, ant_idx: int):
        """Compute spatial tensors with LRU caching."""
        if not self.enable_caching:
            return self._compute_spatial_tensors(bs_pos, ue_pos, ant_idx)
        
        cache_key = self._generate_cache_key(bs_pos, ue_pos, ant_idx)
        
        # Cache hit: mark as recently used
        if cache_key in self._spatial_cache:
            value = self._spatial_cache.pop(cache_key)
            self._spatial_cache[cache_key] = value  # Move to end
            self._cache_hits += 1
            return value
        
        # Cache miss: compute and store
        spatial_tensors = self._compute_spatial_tensors(bs_pos, ue_pos, ant_idx)
        self._spatial_cache[cache_key] = spatial_tensors
        self._cache_misses += 1
        
        # LRU eviction: remove oldest entry
        if len(self._spatial_cache) > self.max_cache_size:
            self._spatial_cache.popitem(last=False)
        
        return spatial_tensors
    
    def _compute_spatial_tensors(self, bs_pos: torch.Tensor, ue_pos: torch.Tensor, ant_idx: int):
        """Compute spatial tensors using PrismNetwork."""
        # Use PrismNetwork to compute spatial features
        result = self.prism_network(
            bs_position=bs_pos,
            ue_position=ue_pos,
            antenna_index=ant_idx,
            return_intermediates=True
        )
        
        return {
            'attenuation_vectors': result['attenuation_vectors'],
            'radiation_vectors': result['radiation_vectors'],
            'frequency_basis_vectors': result['frequency_basis_vectors'],
            'sampled_positions': result['sampled_positions'],
            'directions': result['directions']
        }
    
    def trace_rays(self, 
                   attenuation_vectors: torch.Tensor,
                   radiation_vectors: torch.Tensor,
                   frequency_basis_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Trace rays using low-rank factorization and caching.
        
        Args:
            attenuation_vectors: Attenuation vectors tensor (num_directions, num_points, R)
            radiation_vectors: Radiation vectors tensor (num_directions, num_points, R)
            frequency_basis_vectors: Frequency basis vectors tensor (num_subcarriers, R)
            
        Returns:
            Dictionary containing CSI predictions and metadata
        """
        # Use provided tensors
        u_rho = attenuation_vectors
        u_s = radiation_vectors
        frequency_basis = frequency_basis_vectors
        
        # Move tensors to device
        device = u_rho.device
        u_rho = u_rho.to(device)
        u_s = u_s.to(device)
        frequency_basis = frequency_basis.to(device)
        
        # Low-rank CSI computation
        with torch.amp.autocast('cuda', enabled=True):
            num_directions = u_rho.shape[0]
            num_subcarriers = frequency_basis.shape[0]
            
            # Initialize CSI accumulator
            csi_accumulator = torch.zeros(num_subcarriers, dtype=torch.complex64, device=device)
            
            # Process each ray direction
            for ray_idx in range(num_directions):
                # Extract vectors for current ray: [num_sampling_points, R]
                ray_attenuation = u_rho[ray_idx]  # [K, R]
                ray_radiation = u_s[ray_idx]      # [K, R]
                
                # Low-rank factorization: S_f = ⟨U^(1), V^(1)(f)⟩ × ⟨U^(2), V^(2)(f)⟩
                # For each subcarrier, compute the inner product
                for subcarrier_idx in range(num_subcarriers):
                    # Get frequency basis vector for this subcarrier
                    freq_basis = frequency_basis[subcarrier_idx]  # [R]
                    
                    # Compute attenuation contribution
                    atten_contrib = torch.sum(ray_attenuation * freq_basis.unsqueeze(0), dim=1)  # [K]
                    atten_factor = torch.sum(atten_contrib)  # scalar
                    
                    # Compute radiation contribution
                    rad_contrib = torch.sum(ray_radiation * freq_basis.unsqueeze(0), dim=1)  # [K]
                    rad_factor = torch.sum(rad_contrib)  # scalar
                    
                    # Combine contributions
                    ray_contribution = atten_factor * rad_factor
                    csi_accumulator[subcarrier_idx] += ray_contribution
            
            # Normalize by number of directions
            csi_accumulator = csi_accumulator / num_directions
        
        # Return results
        result = {
            'csi_predictions': csi_accumulator,
            'num_directions': num_directions,
            'num_subcarriers': num_subcarriers,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses
        }
        
        if self.enable_profiling:
            cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
            result['cache_hit_rate'] = cache_hit_rate
            logger.debug(f"📊 Cache performance: hits={self._cache_hits}, misses={self._cache_misses}, hit_rate={cache_hit_rate:.3f}")
        
        return result
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_size': len(self._spatial_cache) if self.enable_caching else 0,
            'max_cache_size': self.max_cache_size
        }
    
    def clear_cache(self):
        """Clear the LRU cache."""
        if self.enable_caching:
            self._spatial_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.info("🗑️ Cache cleared")
    
    def get_learnable_parameters(self):
        """Get learnable parameters from the PrismNetwork."""
        return {
            'prism_network': list(self.prism_network.parameters())
        }
