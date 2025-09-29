"""
Neural Network Ray Tracer: Direct CSI prediction using PrismNetwork as TraceNetwork

This module provides NNRayTracer, a neural network-based ray tracer that
bypasses traditional ray generation and tracing calculations by directly
using a neural network (PrismNetwork) for CSI prediction.
"""

from typing import Dict, List, Tuple, Union, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class NNRayTracer:
    """
    Neural Network-based Ray Tracer using TraceNetwork.
    
    This class provides a simplified interface where ray tracing is performed
    directly by a neural network (TraceNetwork), eliminating the need for
    explicit ray generation and tracing calculations.
    """
    
    def __init__(self, prism_network, trace_network):
        """
        Initialize the neural network ray tracer.
        
        Args:
            prism_network: PrismNetwork instance for feature extraction
            trace_network: TraceNetwork instance for CSI prediction
        """
        # Validate inputs
        if prism_network is None:
            raise ValueError("PrismNetwork is required for feature extraction")
        if trace_network is None:
            raise ValueError("TraceNetwork is required for CSI prediction")
        
        self.prism_network = prism_network
        self.trace_network = trace_network
        
        # Get configuration from PrismNetwork (for feature extraction)
        self.azimuth_divisions = prism_network.azimuth_divisions
        self.elevation_divisions = prism_network.elevation_divisions
        self.max_ray_length = prism_network.max_ray_length
        self.num_sampling_points = prism_network.num_sampling_points
        self.num_subcarriers = prism_network.num_subcarriers
        self.num_bs_antennas = prism_network.num_bs_antennas
        
        # Calculate derived parameters
        self.azimuth_resolution = 2 * torch.pi / self.azimuth_divisions  # 0° to 360° (2π range)
        self.elevation_resolution = (torch.pi / 2) / self.elevation_divisions
        self.total_directions = self.azimuth_divisions * self.elevation_divisions
        
        logger.info(f"🧠 NNRayTracer initialized")
        logger.info(f"   - PrismNetwork: {type(prism_network).__name__} (feature extraction)")
        logger.info(f"   - TraceNetwork: {type(trace_network).__name__} (CSI prediction)")
        logger.info(f"   - Subcarriers: {self.num_subcarriers}")
        logger.info(f"   - BS Antennas: {self.num_bs_antennas}")
        logger.info(f"   - Directions: {self.azimuth_divisions}×{self.elevation_divisions} = {self.total_directions}")
        logger.info(f"   - Max ray length: {self.max_ray_length}m")
        logger.info(f"   - Sampling points per ray: {self.num_sampling_points}")
    
    def trace_rays(self, 
                   attenuation_vectors: torch.Tensor,
                   radiation_vectors: torch.Tensor,
                   frequency_basis_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Trace RF signals using neural network (TraceNetwork) with pre-computed spatial factors.
        
        This method takes pre-computed spatial factors from PrismNetwork and uses TraceNetwork to predict CSI values.
        
        Args:
            attenuation_vectors: Attenuation vectors (num_directions, num_points, R)
            radiation_vectors: Radiation vectors (num_directions, num_points, R)
            frequency_basis_vectors: Frequency basis vectors (num_subcarriers, R)
        
        Returns:
            Dictionary containing CSI predictions and metadata
        """
        # Move tensors to device
        device = attenuation_vectors.device
        u_rho = attenuation_vectors.to(device)
        u_s = radiation_vectors.to(device)
        frequency_basis = frequency_basis_vectors.to(device)
        
        # Process each ray direction separately according to the original formula
        # Note: Mixed precision disabled for complex tensor operations
        with torch.amp.autocast('cuda', enabled=False):
            num_directions = u_rho.shape[0]  # A*B
            num_subcarriers = frequency_basis.shape[0]
            
            # Initialize CSI accumulator
            csi_accumulator = torch.zeros(num_subcarriers, dtype=torch.complex64, device=device)
            
            # Process each direction (ray) separately with memory optimization
            for ray_idx in range(num_directions):
                # Extract spatial factors for this ray
                u_rho_ray = u_rho[ray_idx]  # (num_points, R)
                u_s_ray = u_s[ray_idx]      # (num_points, R)
                
                # Compute Kronecker product for this ray: U^ρ ⊗ U^S
                # Reshape to (num_points, R, 1) and (num_points, 1, R) for broadcasting
                u_rho_expanded = u_rho_ray.unsqueeze(-1)  # (num_points, R, 1)
                u_s_expanded = u_s_ray.unsqueeze(-2)      # (num_points, 1, R)
                
                # Kronecker product: (num_points, R, R)
                spatial_tensor = u_rho_expanded * u_s_expanded
                
                # Sum over sampling points: (R, R)
                spatial_sum = spatial_tensor.sum(dim=0)
                
                # Compute frequency-domain signal for this ray
                # V^T * (U^ρ ⊗ U^S) * V
                # frequency_basis: (num_subcarriers, R)
                # spatial_sum: (R, R)
                
                # First: V^T * spatial_sum -> (num_subcarriers, R)
                intermediate = torch.matmul(frequency_basis, spatial_sum)
                
                # Second: intermediate * V^T -> (num_subcarriers, num_subcarriers)
                # But we only need diagonal elements for CSI
                ray_csi = torch.sum(intermediate * frequency_basis.conj(), dim=1)  # (num_subcarriers,)
                
                # Accumulate CSI from this ray
                csi_accumulator += ray_csi
                
                # Immediate memory cleanup for this ray
                del u_rho_ray, u_s_ray, u_rho_expanded, u_s_expanded
                del spatial_tensor, spatial_sum, intermediate, ray_csi
                
                # Clear cache periodically
                if ray_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Convert to real-valued CSI predictions
            csi_vector = csi_accumulator.real  # [num_subcarriers]
        
        # Format output to match training_interface expectations
        # Expected format: [1, 1, num_subcarriers, 1]
        csi_formatted = csi_vector.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        
        logger.debug(f"🧠 Neural traced CSI shape {csi_formatted.shape} for {num_directions} directions")
        
        # Return in the same format as other ray tracers
        return {
            'csi': csi_formatted
        }


# Export class
__all__ = ['NNRayTracer']
