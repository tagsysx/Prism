"""
NaiveRayTracer: Traditional ray tracing with explicit mathematical calculations

This module implements the traditional (naive) approach to electromagnetic ray tracing,
using explicit mathematical formulas for ray generation, propagation, and signal accumulation.

For neural network-based ray tracing, see nn_ray_tracer.py.
"""

from typing import Dict, List, Tuple, Union, Optional
import torch
import logging
import math

logger = logging.getLogger(__name__)


class NaiveRayTracer:
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
        
        # Calculate derived parameters using precise π values
        self.azimuth_resolution = 2 * torch.pi / self.azimuth_divisions  # 0° to 360°
        self.elevation_resolution = (torch.pi / 2) / self.elevation_divisions   # 0° to 90° (π/2 range)
        self.total_directions = self.azimuth_divisions * self.elevation_divisions
        
        logger.info(f"🚀 NaiveRayTracer initialized with PrismNetwork")
        logger.info(f"   - Directions: {self.azimuth_divisions}×{self.elevation_divisions} = {self.total_directions}")
        logger.info(f"   - Max ray length: {self.max_ray_length}m")
        logger.info(f"   - Sampling points per ray: {self.num_sampling_points}")

    def trace_rays(self, 
                   base_station_pos: torch.Tensor,
                   ue_positions: List[torch.Tensor],
                   antenna_index: int,
                   selected_subcarriers: Optional[List[int]] = None) -> torch.Tensor:
        """
        Trace RF signals using PrismNetwork's internal ray generation.
        
        Implements the mathematical formula:
        S_f(P_RX, ω) = Σ_{k=1}^K H_f(P_k) * ρ_f(P_k) * S_f(P_k, ω) * Δt
        
        Where H_f(P_k) = 1 - Σ_{j=1}^{k-1} ρ_f(P_j) * Δt (channel coefficient)
        
        Then accumulates over all directions:
        S_accumulated = Σ_{i=1}^A Σ_{j=1}^B S_ray(φ_i, θ_j)
        
        Note: This method uses PrismNetwork's internal uniform ray generation
        rather than external direction specification for consistency.
        
        Args:
            base_station_pos: Base station position (3,)
            ue_positions: List of UE positions, each (3,)
            antenna_index: BS antenna index
            selected_subcarriers: Optional list of subcarrier indices
        
        Returns:
            csi_matrix: (num_ues, num_subcarriers) - CSI for each UE and each selected subcarrier
        """
        # Move tensors to device
        device = self.prism_network.device if hasattr(self.prism_network, 'device') else 'cuda'
        bs_pos = base_station_pos.to(device)
        
        # Process each UE position
        csi_results = []
        
        for ue_idx, ue_position in enumerate(ue_positions):
            ue_pos = ue_position.to(device)
            
            # Use PrismNetwork for ray tracing
            with torch.amp.autocast('cuda', enabled=True):
                outputs = self.prism_network(
                    bs_position=bs_pos,
                    ue_position=ue_pos,
                    antenna_index=antenna_index,
                    selected_subcarriers=selected_subcarriers,
                    return_intermediates=False
                )
            
            # Compute CSI using proper ray tracing formula
            csi_vector = self._compute_ray_traced_signal(
                attenuation_vectors=outputs['attenuation_vectors'],  # (num_directions, num_points, output_dim)
                radiation_vectors=outputs['radiation_vectors'],      # (num_directions, num_points, output_dim)
                frequency_basis_vectors=outputs['frequency_basis_vectors'],  # (num_subcarriers, output_dim)
                sampled_positions=outputs['sampled_positions']       # (num_directions, num_points, 3)
            )
            
            csi_results.append(csi_vector)
        
        # Stack results: (num_ues, num_subcarriers)
        csi_matrix = torch.stack(csi_results, dim=0)
        
        logger.debug(f"🔍 Computed CSI matrix shape {csi_matrix.shape} for {len(ue_positions)} UEs, antenna {antenna_index}")
        return csi_matrix
    
    def _compute_ray_traced_signal(self,
                                 attenuation_vectors: torch.Tensor,
                                 radiation_vectors: torch.Tensor, 
                                 frequency_basis_vectors: torch.Tensor,
                                 sampled_positions: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized ray-traced signal computation, avoiding frequency loops.
        
        Key insight: rho_k and S_k are R-dimensional low-rank decomposed values.
        Must compute Hermitian inner product with frequency_basis_vectors first to get 
        frequency-specific values before ray tracing.
        
        Implements:
        For all frequencies f simultaneously:
            ρ_f(P_k) = <ρ_R(P_k), V(f)>  (Hermitian inner product)
            S_f(P_k) = <S_R(P_k), V(f)>  (Hermitian inner product)
            S_f(P_RX, ω) = Σ_{k=1}^K H_f(P_k) * ρ_f(P_k) * S_f(P_k, ω) * Δt
        
        Where H_f(P_k) = exp(-Σ_{j=1}^{k-1} ρ_f(P_j) * Δt) (exponential attenuation model)
        
        Key performance improvements:
        - Fully vectorized across all frequencies, directions, and points
        - Single cumsum operation for all frequencies simultaneously
        - Eliminates frequency loop for maximum GPU efficiency
        - Uses optimized tensor operations throughout
        
        Args:
            attenuation_vectors: (num_directions, num_points, R) - ρ_R(P_k) low-rank values
            radiation_vectors: (num_directions, num_points, R) - S_R(P_k, ω) low-rank values
            frequency_basis_vectors: (num_subcarriers, R) - frequency basis V(f)
            sampled_positions: (num_directions, num_points, 3) - sampling positions
            
        Returns:
            csi_vector: (num_subcarriers,) - accumulated CSI for all subcarriers
        """
        num_directions, num_points, R = attenuation_vectors.shape
        num_subcarriers = frequency_basis_vectors.shape[0]
        
        # Calculate Δt (step size along rays)
        delta_t = self.max_ray_length / num_points
        
        # Ensure complex type for RF signals before einsum operations
        attenuation_vectors_complex = attenuation_vectors.to(torch.complex64)
        radiation_vectors_complex = radiation_vectors.to(torch.complex64)
        
        # Check dimension compatibility - DO NOT provide fallback, FAIL IMMEDIATELY
        attenuation_R = attenuation_vectors_complex.shape[-1]
        frequency_R = frequency_basis_vectors.shape[-1]
        if attenuation_R != frequency_R:
            raise RuntimeError(
                f"❌ DIMENSION MISMATCH in ray tracing computation:\n"
                f"   attenuation_vectors R dimension: {attenuation_R}\n"
                f"   frequency_basis_vectors R dimension: {frequency_R}\n"
                f"   attenuation_vectors shape: {attenuation_vectors_complex.shape}\n"
                f"   frequency_basis_vectors shape: {frequency_basis_vectors.shape}\n"
                f"   These MUST be equal for einsum operation to work.\n"
                f"   Check AttenuationNetwork.output_dim vs FrequencyNetwork.output_dim configuration."
            )
        
        # Memory optimization: Process subcarriers in chunks to reduce peak memory
        chunk_size = 16  # Adjust based on GPU memory; smaller = less memory, more overhead
        csi_results = torch.zeros(num_subcarriers, dtype=torch.complex64, device=attenuation_vectors.device)
        
        for f_start in range(0, num_subcarriers, chunk_size):
            f_end = min(f_start + chunk_size, num_subcarriers)
            f_slice = slice(f_start, f_end)
            
            freq_basis_chunk = frequency_basis_vectors[f_slice]  # (chunk_size, R)
            
            # Vectorized computation for current chunk
            rho_f_chunk = torch.einsum('ijr,fr->fij', attenuation_vectors_complex, freq_basis_chunk.conj())  # (chunk_size, num_directions, num_points)
            S_f_chunk = torch.einsum('ijr,fr->fij', radiation_vectors_complex, freq_basis_chunk.conj())      # (chunk_size, num_directions, num_points)
            
            attenuation_contributions = rho_f_chunk * delta_t  # (chunk_size, num_directions, num_points)
            
            cumulative_attenuation = torch.zeros_like(attenuation_contributions)  # (chunk_size, num_directions, num_points)
            
            if num_points > 1:
                # Cannot use out= parameter with autograd - must use direct assignment
                cumulative_attenuation[:, :, 1:] = torch.cumsum(attenuation_contributions[:, :, :-1], dim=2)
            
            # Compute channel coefficients H_f(P_k) = exp(-cumulative_attenuation) for current chunk
            max_attenuation = 50.0
            cumulative_attenuation_clamped = torch.clamp(cumulative_attenuation.real, min=-max_attenuation, max=max_attenuation)
            
            eps = torch.tensor(1e-14, device=attenuation_vectors.device, dtype=torch.complex64)
            H_f = torch.exp(-cumulative_attenuation_clamped.to(torch.complex64) + eps)  # (chunk_size, num_directions, num_points)
            
            # Vectorized computation of contributions: H_f * ρ_f * S_f * Δt for current chunk
            contributions = H_f * rho_f_chunk * S_f_chunk * delta_t  # (chunk_size, num_directions, num_points)
            
            # Apply numerical stability protection
            max_contribution = 1e15
            contributions_clamped = torch.clamp(contributions.abs(), max=max_contribution) * torch.exp(1j * contributions.angle())
            
            # Sum contributions along points and directions dimensions for each frequency in chunk
            csi_chunk = contributions_clamped.sum(dim=(1, 2))  # (chunk_size,)
            
            csi_results[f_slice] = csi_chunk
        
        # Final NaN/Inf check and replacement with zeros if needed
        csi_results = torch.where(
            torch.isnan(csi_results) | torch.isinf(csi_results),
            torch.zeros_like(csi_results),
            csi_results
        )
        
        # Format output to match training_interface expectations
        # Expected format: [1, 1, num_subcarriers, 1] for single UE case
        if len(ue_positions) == 1:
            csi_formatted = csi_results.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [num_subcarriers] -> [1, 1, num_subcarriers, 1]
        else:
            # Multi-UE case (though currently trace_rays processes one UE at a time)
            csi_formatted = csi_results.unsqueeze(0).unsqueeze(-1)  # [num_subcarriers] -> [1, num_subcarriers, 1]
        
        # Return in the same format as other ray tracers
        return {
            'csi': csi_formatted,
            'selected_subcarriers': selected_subcarriers
        }


# Export classes
__all__ = ['NaiveRayTracer']


