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
    
    def __init__(self, prism_network, config=None):
        """
        Initialize the low-rank ray tracer.
        
        Args:
            prism_network: PrismNetwork instance for feature extraction
            config: Configuration dictionary (optional)
        """
        super().__init__()
        
        if prism_network is None:
            raise ValueError("PrismNetwork is required")
        
        self.prism_network = prism_network
        
        # Get configuration from PrismNetwork
        self.max_ray_length = prism_network.max_ray_length
        
        # Extract tracing configuration for chunk sizes
        tracing_config = config.get('tracing', {}) if config else {}
        
        # Set all chunk sizes from configuration with fallback defaults
        self.ray_chunk_size = tracing_config.get('ray_chunk_size', 128)
        self.freq_chunk_size = tracing_config.get('freq_chunk_size', 16)
        self.ray_sub_chunk_size = tracing_config.get('ray_sub_chunk_size', 8)
        self.point_chunk_size = tracing_config.get('point_chunk_size', 8)
        
        # Memory optimization options
        self.enable_second_order = tracing_config.get('enable_second_order', True)  # Can disable for extreme memory saving
        self.ultra_memory_mode = tracing_config.get('ultra_memory_mode', False)    # Ultra aggressive memory optimization
        
        # Note: Caching and profiling features removed - simplified implementation
        
        # Note: amplitude_scaling removed - CSI enhancement network handles amplitude control
        
        logger.info(f"🚀 LowRankRayTracer initialized")
        logger.info(f"   - PrismNetwork: {type(prism_network).__name__}")
        logger.info(f"   - Ray chunk size: {self.ray_chunk_size}")
        logger.info(f"   - Frequency chunk size: {self.freq_chunk_size}")
        logger.info(f"   - Ray sub-chunk size: {self.ray_sub_chunk_size}")
        logger.info(f"   - Point chunk size: {self.point_chunk_size}")
        logger.info(f"   - Second-order terms: {'enabled' if self.enable_second_order else 'disabled'}")
        logger.info(f"   - Ultra memory mode: {'enabled' if self.ultra_memory_mode else 'disabled'}")
        logger.info(f"   - Amplitude control: handled by CSI enhancement network")
    
    
    def trace_rays(self, 
                   attenuation_vectors: torch.Tensor,
                   radiation_vectors: torch.Tensor,
                   frequency_basis_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Optimized ray tracing using vectorized operations.
        
        This method performs ray tracing for all virtual subcarriers, which includes
        num_ue_antennas × num_subcarriers total subcarriers. The virtual subcarriers
        are organized such that each UE antenna gets its own set of subcarriers.
        
        Args:
            attenuation_vectors: Attenuation vectors tensor (num_directions, num_points, R)
            radiation_vectors: Radiation vectors tensor (num_directions, num_points, R)
            frequency_basis_vectors: Frequency basis vectors tensor (num_virtual_subcarriers, R)
                                   where num_virtual_subcarriers = num_ue_antennas × num_subcarriers
            
        Returns:
            Dictionary containing:
                - csi: Complex CSI predictions tensor [num_virtual_subcarriers]
                       Contains predictions for all virtual subcarriers (num_ue_antennas × num_subcarriers)
        """
        device = attenuation_vectors.device
        num_directions, num_points, rank = attenuation_vectors.shape
        num_virtual_subcarriers = frequency_basis_vectors.shape[0]  # num_ue_antennas × num_subcarriers
        
        logger.debug(f"🔧 LowRankRayTracer inputs: num_virtual_subcarriers={num_virtual_subcarriers}, frequency_basis_vectors.shape={frequency_basis_vectors.shape}")
        
        # Vectorized computation with ray chunking for memory efficiency
        with torch.amp.autocast('cuda', enabled=True):
            # Calculate Δt (step size along rays) - same as naive_ray_tracer
            delta_t = self.max_ray_length / num_points
            
            # Precompute cumulative attenuation for all rays: [num_directions, num_points, rank]
            # Include Δt in the cumulative sum: Û^ρ(P_k) = Σ_{j=1}^{k-1} U^ρ(P_j) * Δt
            u_hat_rho = torch.cumsum(attenuation_vectors.conj() * delta_t, dim=1)  # Cumulative sum with Δt
            
            # Initialize CSI accumulator
            csi_accumulator = torch.zeros(num_virtual_subcarriers, dtype=torch.complex64, device=device)
            
            # Process rays in chunks to manage memory usage
            ray_chunk_size = min(self.ray_chunk_size, num_directions)  # Use configurable chunk size
            
            for ray_start in range(0, num_directions, ray_chunk_size):
                ray_end = min(ray_start + ray_chunk_size, num_directions)
                ray_slice = slice(ray_start, ray_end)
                
                # Extract chunk of rays
                ray_attenuation_chunk = attenuation_vectors[ray_slice]  # [ray_chunk_size, num_points, rank]
                ray_radiation_chunk = radiation_vectors[ray_slice]      # [ray_chunk_size, num_points, rank]
                ray_u_hat_rho_chunk = u_hat_rho[ray_slice]              # [ray_chunk_size, num_points, rank]
                
                # 🚀 MEMORY OPTIMIZATION: Avoid creating large intermediate tensors
                # Instead of storing us_outer [ray_chunk_size, num_points, rank, rank], compute incrementally
                
                # Initialize first_order_chunk for incremental accumulation
                first_order_chunk = torch.zeros(ray_chunk_size, num_virtual_subcarriers, dtype=torch.complex64, device=device)
                
                # Process frequency basis vectors in chunks to avoid large v_outer tensor
                freq_chunk_size_first = min(self.freq_chunk_size, num_virtual_subcarriers)
                for freq_start in range(0, num_virtual_subcarriers, freq_chunk_size_first):
                    freq_end = min(freq_start + freq_chunk_size_first, num_virtual_subcarriers)
                    freq_chunk = frequency_basis_vectors[freq_start:freq_end]  # [chunk_size, rank]
                    
                    # Compute v_outer only for this frequency chunk: [chunk_size, rank, rank]
                    v_outer_chunk = torch.einsum('fi,fj->fij', freq_chunk, freq_chunk)
                    
                    # 🔥 AVOID us_outer: Compute first-order term directly without storing full us_outer
                    # Original: us_outer = einsum('rki,rkj->rkij', ray_radiation_chunk.conj(), ray_attenuation_chunk.conj())
                    # Then: first_order = einsum('rkij,fij->rf', us_outer, v_outer_chunk) * delta_t
                    # Combined: first_order = einsum('rki,rkj,fij->rf', ray_radiation_chunk.conj(), ray_attenuation_chunk.conj(), v_outer_chunk) * delta_t
                    
                    # Process points in chunks to further reduce memory
                    point_chunk_size_first = min(self.point_chunk_size, num_points)
                    for point_start in range(0, num_points, point_chunk_size_first):
                        point_end = min(point_start + point_chunk_size_first, num_points)
                        point_slice = slice(point_start, point_end)
                        
                        # Extract point chunk
                        ray_radiation_point = ray_radiation_chunk[:, point_slice].conj()  # [ray_chunk_size, point_chunk_size, rank]
                        ray_attenuation_point = ray_attenuation_chunk[:, point_slice].conj()  # [ray_chunk_size, point_chunk_size, rank]
                        
                        # Compute first-order contribution for this point chunk without storing us_outer
                        point_contribution = torch.einsum('rki,rkj,fij->rf', 
                                                         ray_radiation_point, 
                                                         ray_attenuation_point, 
                                                         v_outer_chunk) * delta_t
                        
                        first_order_chunk[:, freq_start:freq_end] += point_contribution
                        
                        # Clear point chunk tensors
                        del ray_radiation_point, ray_attenuation_point, point_contribution
                    
                    # Clear frequency chunk tensors
                    del v_outer_chunk
                
                # Second-order term: sum over points of (U^S ⊗ U^ρ ⊗ Û^ρ) · (V ⊗ V ⊗ V) * Δt
                # Only for k > 0 (skip first point)
                # Initialize second_order_chunk with actual ray count
                actual_ray_count = ray_end - ray_start
                second_order_chunk = torch.zeros(actual_ray_count, num_virtual_subcarriers, dtype=torch.complex64, device=device)
                
                # 🚀 MEMORY OPTIMIZATION: Make second-order term optional for extreme memory saving
                if self.enable_second_order and num_points > 1:
                    # Memory-optimized computation: avoid creating large 5D tensor us_outer_hat
                    # Instead of: us_outer_hat = torch.einsum('rki,rkj,rkl->rkijl', ...)
                    # We compute the einsum directly without storing the intermediate tensor
                    
                    # Process frequency basis vectors in chunks to avoid OOM
                    freq_chunk_size = min(self.freq_chunk_size, num_virtual_subcarriers)  # Use configurable frequency chunk size
                    for freq_start in range(0, num_virtual_subcarriers, freq_chunk_size):
                        freq_end = min(freq_start + freq_chunk_size, num_virtual_subcarriers)
                        freq_chunk = frequency_basis_vectors[freq_start:freq_end]  # [chunk_size, rank]
                        
                        # Compute v_outer_outer for this chunk: [chunk_size, rank, rank, rank]
                        v_outer_outer_chunk = torch.einsum('fi,fj,fk->fijk', 
                                                          freq_chunk, 
                                                          freq_chunk, 
                                                          freq_chunk)
                        
                        # Compute us_outer_hat contribution for this frequency chunk
                        # Instead of creating the full 5D tensor, compute the contribution directly
                        # This avoids storing the large us_outer_hat tensor
                        chunk_contribution = torch.zeros(ray_chunk_size, freq_end - freq_start, dtype=torch.complex64, device=device)
                        
                        # Process rays in sub-chunks for memory optimization
                        ray_sub_chunk_size = min(self.ray_sub_chunk_size, ray_chunk_size)  # Use configurable ray sub-chunk size
                        
                        # Process points in chunks for memory optimization
                        point_chunk_size = min(self.point_chunk_size, num_points - 1)  # Use configurable point chunk size
                        for ray_sub_start in range(0, ray_chunk_size, ray_sub_chunk_size):  # Renamed to avoid conflict
                            ray_sub_end = min(ray_sub_start + ray_sub_chunk_size, ray_chunk_size)
                            ray_sub_slice = slice(ray_sub_start, ray_sub_end)
                            
                            # Process points in chunks to further reduce memory
                            ray_contribution = torch.zeros(ray_sub_end - ray_sub_start, freq_end - freq_start, dtype=torch.complex64, device=device)
                            
                            for point_start in range(1, num_points, point_chunk_size):  # Start from 1 (skip point 0)
                                point_end = min(point_start + point_chunk_size, num_points)
                                point_slice = slice(point_start, point_end)
                                
                                # Extract smaller chunk of rays and points
                                ray_radiation_sub = ray_radiation_chunk[ray_sub_slice, point_slice].conj()  # [sub_chunk_size, point_chunk_size, rank]
                                ray_attenuation_sub = ray_attenuation_chunk[ray_sub_slice, point_slice].conj()  # [sub_chunk_size, point_chunk_size, rank]
                                ray_u_hat_rho_sub = ray_u_hat_rho_chunk[ray_sub_slice, point_slice].conj()  # [sub_chunk_size, point_chunk_size, rank]
                                
                                # Compute us_outer_hat for this ray-point sub-chunk: [sub_chunk_size, point_chunk_size, rank, rank, rank]
                                us_outer_hat_sub = torch.einsum('rki,rkj,rkl->rkijl', 
                                                               ray_radiation_sub, 
                                                               ray_attenuation_sub, 
                                                               ray_u_hat_rho_sub)
                                
                                # Compute contribution: sum over points (k dimension) to match v_outer_outer_chunk
                                # us_outer_hat_sub: [sub_chunk_size, point_chunk_size, rank, rank, rank]
                                # v_outer_outer_chunk: [chunk_size, rank, rank, rank]
                                # Sum over point dimension k, then contract remaining rank dimensions
                                sub_chunk_contribution = torch.einsum('rijk,fijk->rf', us_outer_hat_sub.sum(dim=1), v_outer_outer_chunk) * delta_t
                                ray_contribution += sub_chunk_contribution
                                
                                # Clear intermediate variables
                                del us_outer_hat_sub, sub_chunk_contribution, ray_radiation_sub, ray_attenuation_sub, ray_u_hat_rho_sub
                            
                            # Store the accumulated contribution for this ray (use local indices)
                            chunk_contribution[ray_sub_start:ray_sub_end, :] = ray_contribution
                            del ray_contribution
                        
                        # Store the contribution for this frequency chunk
                        second_order_chunk[:, freq_start:freq_end] = chunk_contribution
                        
                        # Clear intermediate variables
                        del v_outer_outer_chunk, chunk_contribution
                
                # Combine terms and accumulate
                # According to the mathematical derivation: S_f ≈ ⟨U^(1), V^(1)⟩ + ⟨U^(2), V^(2)⟩
                ray_contribution_chunk = first_order_chunk + second_order_chunk  # [ray_chunk_size, num_virtual_subcarriers]
                csi_accumulator += ray_contribution_chunk.sum(dim=0)  # Sum over rays in chunk
                
                # Clear intermediate tensors to free memory
                del first_order_chunk, second_order_chunk, ray_contribution_chunk
                torch.cuda.empty_cache()
        
        # Normalize by the number of directions
        # CSI enhancement network will handle amplitude control
        csi_accumulator = csi_accumulator / num_directions
        
        logger.debug(f"🔧 LowRankRayTracer output: csi_accumulator.shape={csi_accumulator.shape}")
        
        return {
            'csi': csi_accumulator
        }
    
    def get_learnable_parameters(self):
        """Get learnable parameters from the PrismNetwork and ray tracer."""
        return {
            'prism_network': list(self.prism_network.parameters())
        }
