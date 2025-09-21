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
    
    def __init__(self, prism_network):
        """
        Initialize the low-rank ray tracer.
        
        Args:
            prism_network: PrismNetwork instance for feature extraction
        """
        super().__init__()
        
        if prism_network is None:
            raise ValueError("PrismNetwork is required")
        
        self.prism_network = prism_network
        
        # Get configuration from PrismNetwork
        self.max_ray_length = prism_network.max_ray_length
        
        # Note: Caching and profiling features removed - simplified implementation
        
        # Note: amplitude_scaling removed - CSI enhancement network handles amplitude control
        
        logger.info(f"🚀 LowRankRayTracer initialized")
        logger.info(f"   - PrismNetwork: {type(prism_network).__name__}")
        logger.info(f"   - Amplitude control: handled by CSI enhancement network")
    
    
    def trace_rays(self, 
                   attenuation_vectors: torch.Tensor,
                   radiation_vectors: torch.Tensor,
                   frequency_basis_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Optimized ray tracing using vectorized operations.
        
        Args:
            attenuation_vectors: Attenuation vectors tensor (num_directions, num_points, R)
            radiation_vectors: Radiation vectors tensor (num_directions, num_points, R)
            frequency_basis_vectors: Frequency basis vectors tensor (num_subcarriers, R)
            
        Returns:
            Dictionary containing CSI predictions
        """
        device = attenuation_vectors.device
        num_directions, num_points, rank = attenuation_vectors.shape
        num_subcarriers = frequency_basis_vectors.shape[0]
        
        # Vectorized computation with ray chunking for memory efficiency
        with torch.amp.autocast('cuda', enabled=True):
            # Calculate Δt (step size along rays) - same as naive_ray_tracer
            delta_t = self.max_ray_length / num_points
            
            # Precompute cumulative attenuation for all rays: [num_directions, num_points, rank]
            # Include Δt in the cumulative sum: Û^ρ(P_k) = Σ_{j=1}^{k-1} U^ρ(P_j) * Δt
            u_hat_rho = torch.cumsum(attenuation_vectors.conj() * delta_t, dim=1)  # Cumulative sum with Δt
            
            # Initialize CSI accumulator
            csi_accumulator = torch.zeros(num_subcarriers, dtype=torch.complex64, device=device)
            
            # Process rays in chunks to manage memory usage
            ray_chunk_size = min(108, num_directions)  # Process all rays at once, but limit to 108
            
            for ray_start in range(0, num_directions, ray_chunk_size):
                ray_end = min(ray_start + ray_chunk_size, num_directions)
                ray_slice = slice(ray_start, ray_end)
                
                # Extract chunk of rays
                ray_attenuation_chunk = attenuation_vectors[ray_slice]  # [ray_chunk_size, num_points, rank]
                ray_radiation_chunk = radiation_vectors[ray_slice]      # [ray_chunk_size, num_points, rank]
                ray_u_hat_rho_chunk = u_hat_rho[ray_slice]              # [ray_chunk_size, num_points, rank]
                
                # Vectorized computation for all subcarriers and current ray chunk
                # First-order term: sum over points of (U^S ⊗ U^ρ) · (V ⊗ V) * Δt
                # Shape: [ray_chunk_size, num_points, rank, rank] -> [ray_chunk_size, num_subcarriers]
                us_outer = torch.einsum('rki,rkj->rkij', ray_radiation_chunk.conj(), ray_attenuation_chunk.conj())  # [ray_chunk_size, num_points, rank, rank]
                v_outer = torch.einsum('fi,fj->fij', frequency_basis_vectors, frequency_basis_vectors)  # [num_subcarriers, rank, rank]
                first_order_chunk = torch.einsum('rkij,fij->rf', us_outer, v_outer) * delta_t  # [ray_chunk_size, num_subcarriers]
                
                # Second-order term: sum over points of (U^S ⊗ U^ρ ⊗ Û^ρ) · (V ⊗ V ⊗ V) * Δt
                # Only for k > 0 (skip first point)
                if num_points > 1:
                    us_outer_hat = torch.einsum('rki,rkj,rkl->rkijl', 
                                               ray_radiation_chunk[:, 1:].conj(), 
                                               ray_attenuation_chunk[:, 1:].conj(), 
                                               ray_u_hat_rho_chunk[:, 1:].conj())  # [ray_chunk_size, num_points-1, rank, rank, rank]
                    v_outer_outer = torch.einsum('fi,fj,fk->fijk', 
                                                 frequency_basis_vectors, 
                                                 frequency_basis_vectors, 
                                                 frequency_basis_vectors)  # [num_subcarriers, rank, rank, rank]
                    second_order_chunk = torch.einsum('rkijl,fijl->rf', us_outer_hat, v_outer_outer) * delta_t  # [ray_chunk_size, num_subcarriers]
                else:
                    second_order_chunk = torch.zeros(ray_chunk_size, num_subcarriers, dtype=torch.complex64, device=device)
                
                # Combine terms and accumulate
                # According to the mathematical derivation: S_f ≈ ⟨U^(1), V^(1)⟩ + ⟨U^(2), V^(2)⟩
                ray_contribution_chunk = first_order_chunk + second_order_chunk  # [ray_chunk_size, num_subcarriers]
                csi_accumulator += ray_contribution_chunk.sum(dim=0)  # Sum over rays in chunk
                
                # Clear intermediate tensors to free memory
                del us_outer, v_outer, first_order_chunk
                if num_points > 1:
                    del us_outer_hat, v_outer_outer, second_order_chunk
                del ray_contribution_chunk
                torch.cuda.empty_cache()
        
        # Normalize by the number of directions
        # CSI enhancement network will handle amplitude control
        csi_accumulator = csi_accumulator / num_directions
        
        return {
            'csi': csi_accumulator
        }
    
    def get_learnable_parameters(self):
        """Get learnable parameters from the PrismNetwork and ray tracer."""
        return {
            'prism_network': list(self.prism_network.parameters())
        }
