"""
RayTracer: CUDA-accelerated ray tracing with PrismNetwork integration

Implements ray tracing using PrismNetwork's internal ray processing capabilities.
"""

from typing import Dict, List, Tuple, Union, Optional
import torch
import logging
import math

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
        self.origin = origin.clone().to(dtype=torch.float32, device=device)
        self.direction = self._normalize(direction.clone().to(dtype=torch.float32, device=device))
        self.max_length = max_length
    
    def _normalize(self, vector: torch.Tensor) -> torch.Tensor:
        """Normalize direction vector."""
        norm = torch.norm(vector)
        if norm < 1e-10:
            return vector
        return vector / norm

class RayTracer:
    """
    CUDA-accelerated ray tracer using PrismNetwork for electromagnetic simulation.
    
    This class provides ray tracing functionality by leveraging PrismNetwork's
    internal ray processing capabilities for efficient electromagnetic wave propagation modeling.
    """
    
    def __init__(self, 
                 prism_network):
        """
        Initialize the ray tracer with PrismNetwork integration.
        
        Args:
            prism_network: PrismNetwork instance for electromagnetic simulation
        """
        # Validate PrismNetwork
        if prism_network is None:
            raise ValueError("PrismNetwork is required for ray tracing")
        
        self.prism_network = prism_network
        
        # Get ray tracing parameters from PrismNetwork
        self.azimuth_divisions = prism_network.azimuth_divisions
        self.elevation_divisions = prism_network.elevation_divisions
        self.max_ray_length = prism_network.max_ray_length
        self.num_sampling_points = prism_network.num_sampling_points
        
        # Calculate derived parameters
        self.azimuth_resolution = 2 * 3.14159 / self.azimuth_divisions  # 0° to 360°
        self.elevation_resolution = (3.14159 / 2) / self.elevation_divisions   # 0° to 90° (π/2 range)
        self.total_directions = self.azimuth_divisions * self.elevation_divisions
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"🚀 RayTracer initialized with PrismNetwork")
        logger.info(f"   - Directions: {self.azimuth_divisions}×{self.elevation_divisions} = {self.total_directions}")
        logger.info(f"   - Max ray length: {self.max_ray_length}m")
        logger.info(f"   - Sampling points per ray: {self.num_sampling_points}")
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.azimuth_divisions <= 0:
            raise ValueError(f"Azimuth divisions must be positive, got {self.azimuth_divisions}")
        
        if self.elevation_divisions <= 0:
            raise ValueError(f"Elevation divisions must be positive, got {self.elevation_divisions}")
        
        if self.max_ray_length <= 0:
            raise ValueError(f"Max ray length must be positive, got {self.max_ray_length}")
        
        if self.num_sampling_points <= 0:
            raise ValueError(f"Number of sampling points must be positive, got {self.num_sampling_points}")
    
    def accumulate_signals(self, 
                          base_station_pos: torch.Tensor,
                          ue_positions: List[torch.Tensor],
                          antenna_index: int,
                          selected_subcarriers: Optional[List[int]] = None) -> Dict:
        """
        Accumulate signals using PrismNetwork's internal ray tracing.
        
        Args:
            base_station_pos: Base station position (3,)
            ue_positions: List of UE positions, each (3,)
            antenna_index: BS antenna index
            selected_subcarriers: Optional list of subcarrier indices
        
        Returns:
            Dictionary containing accumulated CSI and ray tracing results
        """
        # Move tensors to device
        bs_pos = base_station_pos.to(self.device)
        
        # Process each UE position
        accumulated_results = {
            'csi_matrix': [],
            'attenuation_vectors': [],
            'radiation_vectors': [],
            'frequency_basis_vectors': None,
            'sampled_positions': [],
            'directions': None
        }
        
        for ue_idx, ue_position in enumerate(ue_positions):
            ue_pos = ue_position.to(self.device)
            
            # Use PrismNetwork for ray tracing
            with torch.amp.autocast('cuda', enabled=True):
                outputs = self.prism_network(
                    bs_position=bs_pos,
                    ue_position=ue_pos,
                    antenna_index=antenna_index,
                    selected_subcarriers=selected_subcarriers,
                    return_intermediates=False
                )
            
            # Store results
            accumulated_results['attenuation_vectors'].append(outputs['attenuation_vectors'])
            accumulated_results['radiation_vectors'].append(outputs['radiation_vectors'])
            accumulated_results['sampled_positions'].append(outputs['sampled_positions'])
            
            # Store shared results (same for all UEs)
            if accumulated_results['frequency_basis_vectors'] is None:
                accumulated_results['frequency_basis_vectors'] = outputs['frequency_basis_vectors']
                accumulated_results['directions'] = outputs['directions']
            
            # Compute CSI using low-rank factorization
            # CSI = sum over rays and voxels of (attenuation * radiation * frequency_basis)
            attn = outputs['attenuation_vectors']  # (360, 64, output_dim)
            rad = outputs['radiation_vectors']     # (360, 64, output_dim)
            freq = outputs['frequency_basis_vectors']  # (num_subcarriers, output_dim)
            
            # Element-wise multiplication and summation over spatial dimensions
            spatial_response = (attn * rad).sum(dim=(0, 1))  # (output_dim,)
            
            # Compute CSI for each subcarrier
            csi_vector = torch.matmul(freq, spatial_response.unsqueeze(-1)).squeeze(-1)  # (num_subcarriers,)
            accumulated_results['csi_matrix'].append(csi_vector)
        
        # Stack results
        accumulated_results['csi_matrix'] = torch.stack(accumulated_results['csi_matrix'], dim=0)  # (num_ues, num_subcarriers)
        
        logger.debug(f"🔍 Accumulated signals for {len(ue_positions)} UEs")
        return accumulated_results

        
    def trace_rays(self, 
                   base_station_pos: torch.Tensor,
                   ue_positions: List[torch.Tensor],
                   antenna_index: int,
                   selected_subcarriers: Optional[List[int]] = None) -> Dict:
        """
        Trace RF signals using PrismNetwork's internal ray generation.
        
        Note: This method uses PrismNetwork's internal uniform ray generation
        rather than external direction specification for consistency.
        
        Args:
            base_station_pos: Base station position (3,)
            ue_positions: List of UE positions, each (3,)
            antenna_index: BS antenna index
            selected_subcarriers: Optional list of subcarrier indices
        
        Returns:
            Dictionary containing ray tracing results for all UEs
        """
        # This method is essentially the same as accumulate_signals
        # since PrismNetwork handles ray generation internally
        return self.accumulate_signals(
            base_station_pos=base_station_pos,
            ue_positions=ue_positions,
            antenna_index=antenna_index,
            selected_subcarriers=selected_subcarriers
        )
    
  

