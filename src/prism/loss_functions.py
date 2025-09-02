"""
Loss Functions for Prism: Neural Network-Based Electromagnetic Ray Tracing

This module provides specialized loss functions for CSI (Channel State Information) 
and spatial spectrum estimation tasks, all supporting automatic differentiation.

Classes:
- PrismLossFunction: Main loss function class with CSI and PDP losses
- CSILoss: Specialized CSI loss functions
- PDPLoss: Power Delay Profile loss functions
- SpatialSpectrumLoss: Spatial spectrum loss functions (reserved for future use)

All loss functions are designed to work with PyTorch tensors and support backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List
import numpy as np
import logging

# Get logger for this module
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import os
from pathlib import Path


# Default configuration for loss functions
DEFAULT_LOSS_CONFIG = {
    'csi_weight': 0.7,
    'pdp_weight': 0.3,
    'regularization_weight': 0.01,
    'csi_loss': {
        'type': 'hybrid',  # 'mse', 'mae', 'complex_mse', 'magnitude_phase', 'hybrid' (correlation disabled)
        'phase_weight': 1.0,
        'magnitude_weight': 1.0,
        'cmse_weight': 1.0
        # correlation_weight removed - correlation loss disabled
    },
    'pdp_loss': {
        'type': 'hybrid',  # 'mse', 'delay', 'hybrid' (correlation disabled)
        'fft_size': 1024,
        'normalize_pdp': True
        # mse_weight and delay_weight now hardcoded in hybrid loss (0.7, 0.3)
        # correlation_weight removed - correlation loss disabled
    }
}


class CSILoss(nn.Module):
    """
    CSI (Channel State Information) Loss Functions
    
    Provides various loss functions for comparing complex-valued CSI matrices,
    including magnitude, phase, correlation, and hybrid losses.
    """
    
    def __init__(self, loss_type: str = 'mse', phase_weight: float = 1.0, 
                 magnitude_weight: float = 1.0, cmse_weight: float = 1.0):
        """
        Initialize CSI loss function
        
        Args:
            loss_type: Type of loss ('mse', 'mae', 'complex_mse', 'magnitude_phase', 'hybrid')
                      Note: 'correlation' type is disabled due to incorrect implementation
            phase_weight: Weight for phase component in combined losses
            magnitude_weight: Weight for magnitude component in combined losses
            cmse_weight: Weight for CMSE component in hybrid loss
        """
        super(CSILoss, self).__init__()
        self.loss_type = loss_type
        self.phase_weight = phase_weight
        self.magnitude_weight = magnitude_weight
        self.cmse_weight = cmse_weight
        
    def forward(self, predicted_csi: torch.Tensor, target_csi: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute CSI loss between predicted and target CSI
        
        Args:
            predicted_csi: Predicted CSI tensor (complex)
                          Shape: (batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas)
            target_csi: Target CSI tensor (complex)
                       Shape: (batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas)
            mask: Optional mask for selective loss computation
                  Shape: same as CSI tensors
        
        Returns:
            loss: Computed loss value (scalar tensor)
        """
        if predicted_csi.shape != target_csi.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_csi.shape} vs target {target_csi.shape}")
        
        # Apply mask if provided
        if mask is not None:
            predicted_csi = predicted_csi * mask
            target_csi = target_csi * mask
        
        if self.loss_type == 'mse':
            # Standard MSE loss for complex numbers - compute manually
            diff = predicted_csi - target_csi
            loss = torch.mean(torch.abs(diff)**2)
            
        elif self.loss_type == 'mae':
            # Mean Absolute Error for complex numbers
            diff = predicted_csi - target_csi
            loss = torch.mean(torch.abs(diff))
            
        elif self.loss_type == 'complex_mse':
            # Separate real and imaginary parts
            real_loss = F.mse_loss(predicted_csi.real, target_csi.real)
            imag_loss = F.mse_loss(predicted_csi.imag, target_csi.imag)
            loss = real_loss + imag_loss
            
        elif self.loss_type == 'magnitude_phase':
            # Separate magnitude and phase losses
            pred_mag = torch.abs(predicted_csi)
            target_mag = torch.abs(target_csi)
            magnitude_loss = F.mse_loss(pred_mag, target_mag)
            
            # Phase loss (handle zero magnitudes)
            pred_phase = torch.angle(predicted_csi + 1e-8)
            target_phase = torch.angle(target_csi + 1e-8)
            
            # Circular phase difference
            phase_diff = torch.remainder(pred_phase - target_phase + np.pi, 2*np.pi) - np.pi
            phase_loss = torch.mean(phase_diff**2)
            
            loss = self.magnitude_weight * magnitude_loss + self.phase_weight * phase_loss
            

            
        elif self.loss_type == 'hybrid':
            # Modified Hybrid CSI Loss: CMSE + Magnitude + Phase (removed correlation)
            # 1. Complex MSE Loss
            diff = predicted_csi - target_csi
            cmse_loss = torch.mean(torch.abs(diff)**2)
            
            # 2. Magnitude Loss
            pred_mag = torch.abs(predicted_csi)
            target_mag = torch.abs(target_csi)
            magnitude_loss = F.mse_loss(pred_mag, target_mag)
            
            # 3. Phase Loss (handle zero magnitudes)
            pred_phase = torch.angle(predicted_csi + 1e-8)
            target_phase = torch.angle(target_csi + 1e-8)
            
            # Circular phase difference
            phase_diff = torch.remainder(pred_phase - target_phase + np.pi, 2*np.pi) - np.pi
            phase_loss = torch.mean(phase_diff**2)
            
            # Combine three loss components (removed correlation)
            loss = (self.cmse_weight * cmse_loss + 
                   self.magnitude_weight * magnitude_loss + 
                   self.phase_weight * phase_loss)
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    # _complex_correlation method removed - was incorrectly implemented
    # (mixed all subcarriers and antennas together)


class PDPLoss(nn.Module):
    """
    Power Delay Profile (PDP) Loss Functions
    
    Provides time-domain validation by comparing PDPs derived from CSI data.
    Supports MSE, correlation, delay, and hybrid PDP losses.
    """
    
    def __init__(self, loss_type: str = 'hybrid', fft_size: int = 1024,
                 normalize_pdp: bool = True):
        """
        Initialize PDP loss function
        
        Args:
            loss_type: Type of PDP loss ('mse', 'delay', 'hybrid')
                      Note: 'correlation' type is disabled due to incorrect implementation
            fft_size: FFT size for PDP computation
            normalize_pdp: Whether to normalize PDPs before comparison
        """
        super(PDPLoss, self).__init__()
        self.loss_type = loss_type
        self.fft_size = int(fft_size)  # Ensure fft_size is always an integer
        self.normalize_pdp = normalize_pdp
        
    def forward(self, predicted_csi: torch.Tensor, target_csi: torch.Tensor) -> torch.Tensor:
        """
        Compute PDP loss between predicted and target CSI
        
        Args:
            predicted_csi: Predicted CSI tensor (complex)
                          Shape: (N,) - selected subcarriers
            target_csi: Target CSI tensor (complex)
                       Shape: (N,) - selected subcarriers
        
        Returns:
            loss: Computed PDP loss value (scalar tensor)
        """
        # Ensure complex tensors
        if not predicted_csi.is_complex():
            predicted_csi = predicted_csi.to(torch.complex64)
        if not target_csi.is_complex():
            target_csi = target_csi.to(torch.complex64)
            
        # Compute PDPs
        pdp_pred = self._compute_pdp(predicted_csi)
        pdp_target = self._compute_pdp(target_csi)
        
        # Normalize PDPs if required
        if self.normalize_pdp:
            pdp_pred = self._normalize_pdp(pdp_pred)
            pdp_target = self._normalize_pdp(pdp_target)
        
        if self.loss_type == 'mse':
            # PDP MSE Loss
            loss = F.mse_loss(pdp_pred, pdp_target)
            

            
        elif self.loss_type == 'delay':
            # Dominant Path Delay Loss
            loss = self._compute_delay_loss(pdp_pred, pdp_target)
            
        elif self.loss_type == 'hybrid':
            # Modified Hybrid PDP Loss: MSE + Delay (removed correlation)
            mse_loss = F.mse_loss(pdp_pred, pdp_target)
            delay_loss = self._compute_delay_loss(pdp_pred, pdp_target)
            
            # Rebalance weights since we removed correlation component
            # Original: mse_weight=0.5, correlation_weight=0.3, delay_weight=0.2
            # New: mse_weight=0.7, delay_weight=0.3 (maintain relative importance)
            loss = (0.7 * mse_loss + 0.3 * delay_loss)
            
        else:
            raise ValueError(f"Unknown PDP loss type: {self.loss_type}")
        
        return loss
    
    def _compute_pdp(self, csi_data: torch.Tensor) -> torch.Tensor:
        """
        Compute Power Delay Profile from CSI data using IFFT
        
        Args:
            csi_data: CSI data tensor (complex)
                     Shape: (..., N) - any shape with last dimension as subcarriers
        
        Returns:
            pdp: Power delay profile tensor
                Shape: (..., fft_size)
        """
        device = csi_data.device
        original_shape = csi_data.shape
        
        # Flatten all dimensions except the last one (subcarriers)
        if len(original_shape) > 1:
            # Reshape to (batch_size, N) where batch_size is product of all dims except last
            batch_size = torch.prod(torch.tensor(original_shape[:-1])).item()
            N = original_shape[-1]
            csi_flat = csi_data.reshape(batch_size, N)
        else:
            batch_size = 1
            N = original_shape[0]
            csi_flat = csi_data.unsqueeze(0)
        
        # Validate dimensions
        if batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {batch_size}, original_shape: {original_shape}")
        if N <= 0:
            raise ValueError(f"Invalid number of subcarriers: {N}, original_shape: {original_shape}")
        if self.fft_size <= 0:
            raise ValueError(f"Invalid fft_size: {self.fft_size}")
        
        # Zero-pad to fft_size for each batch
        if N >= self.fft_size:
            # If we have more data than FFT size, truncate
            padded_csi = csi_flat[:, :self.fft_size]
        else:
            # Zero-pad to fft_size
            padded_csi = torch.zeros(batch_size, self.fft_size, dtype=csi_data.dtype, device=device)
            padded_csi[:, :N] = csi_flat
        
        # Compute IFFT along the last dimension
        time_domain = torch.fft.ifft(padded_csi, dim=-1)
        
        # Compute power delay profile
        pdp = torch.abs(time_domain) ** 2
        
        # Reshape back to original shape (except last dim is now fft_size)
        if len(original_shape) > 1:
            new_shape = original_shape[:-1] + (self.fft_size,)
            pdp = pdp.reshape(new_shape)
        else:
            pdp = pdp.squeeze(0)
        
        return pdp
    
    def _normalize_pdp(self, pdp: torch.Tensor) -> torch.Tensor:
        """
        Normalize PDP (peak normalization)
        """
        max_val = torch.max(pdp)
        if max_val < 1e-8:
            return pdp
        return pdp / max_val
    
    # _compute_pdp_correlation_loss method removed - was incorrectly implemented
    # (mixed all delay bins together)
    
    def _compute_delay_loss(self, pdp_pred: torch.Tensor, 
                           pdp_target: torch.Tensor) -> torch.Tensor:
        """
        Compute dominant path delay loss using soft argmax for differentiability
        """
        # Handle multi-dimensional PDPs by working on the last dimension
        original_shape = pdp_pred.shape
        last_dim_size = original_shape[-1]
        
        # Create indices for the last dimension
        indices = torch.arange(last_dim_size, dtype=torch.float32, device=pdp_pred.device)
        
        # Flatten all dimensions except the last one
        if len(original_shape) > 1:
            batch_size = torch.prod(torch.tensor(original_shape[:-1])).item()
            pdp_pred_flat = pdp_pred.reshape(batch_size, last_dim_size)
            pdp_target_flat = pdp_target.reshape(batch_size, last_dim_size)
            
            # Expand indices to match batch size
            indices = indices.unsqueeze(0).expand(batch_size, -1)
        else:
            pdp_pred_flat = pdp_pred.unsqueeze(0)
            pdp_target_flat = pdp_target.unsqueeze(0)
            indices = indices.unsqueeze(0)
        
        # Soft argmax for predicted PDP (along last dimension)
        pred_weights = torch.softmax(pdp_pred_flat * 10, dim=-1)  # Temperature scaling
        pred_soft_idx = torch.sum(indices * pred_weights, dim=-1)
        
        # Soft argmax for target PDP (along last dimension)
        target_weights = torch.softmax(pdp_target_flat * 10, dim=-1)
        target_soft_idx = torch.sum(indices * target_weights, dim=-1)
        
        # Compute delay difference (normalized by FFT size)
        delay_diff = torch.abs(pred_soft_idx - target_soft_idx) / self.fft_size
        
        # Return mean across all batch dimensions
        return torch.mean(delay_diff)





class SpatialSpectrumLoss(nn.Module):
    """
    Spatial Spectrum Loss Functions
    
    Computes MSE loss between spatial spectrum matrices derived from CSI data.
    Extracts configuration from base_station.antenna_array and training.loss.spatial_spectrum_loss sections.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize spatial spectrum loss function from configuration
        
        Args:
            config: Full configuration dictionary containing base_station and training sections
        """
        super(SpatialSpectrumLoss, self).__init__()
        
        # Extract base station configuration
        if 'base_station' not in config:
            raise ValueError("Configuration must contain 'base_station' section")
        bs_config = config['base_station']
        
        # Extract spatial spectrum loss configuration
        if 'training' not in config or 'loss' not in config['training'] or 'spatial_spectrum_loss' not in config['training']['loss']:
            raise ValueError("Configuration must contain 'training.loss.spatial_spectrum_loss' section")
        ssl_config = config['training']['loss']['spatial_spectrum_loss']
        
        # Algorithm and fusion method - no defaults, must be specified
        if 'algorithm' not in ssl_config:
            raise ValueError("Configuration must contain 'training.loss.spatial_spectrum_loss.algorithm'")
        if 'fusion_method' not in ssl_config:
            raise ValueError("Configuration must contain 'training.loss.spatial_spectrum_loss.fusion_method'")
        
        self.algorithm = ssl_config['algorithm']
        self.fusion_method = ssl_config['fusion_method']
        
        # Antenna array configuration
        if 'antenna_array' not in bs_config:
            raise ValueError("Base station configuration must contain 'antenna_array' section")
        antenna_config = bs_config['antenna_array']
        
        # Parse antenna array configuration (e.g., '8x8' -> M=8, N=8)
        if 'configuration' not in antenna_config:
            raise ValueError("Antenna array configuration must contain 'configuration' key")
        array_config = antenna_config['configuration']
        self.M, self.N = map(int, array_config.split('x'))
        self.num_antennas = self.M * self.N
        
        # OFDM system parameters
        if 'ofdm' not in bs_config:
            raise ValueError("Base station configuration must contain 'ofdm' section")
        ofdm_config = bs_config['ofdm']
        
        # Handle string values from YAML (scientific notation) - no defaults
        if 'center_frequency' not in ofdm_config:
            raise ValueError("Configuration must contain 'base_station.ofdm.center_frequency'")
        if 'bandwidth' not in ofdm_config:
            raise ValueError("Configuration must contain 'base_station.ofdm.bandwidth'")
        if 'num_subcarriers' not in ofdm_config:
            raise ValueError("Configuration must contain 'base_station.ofdm.num_subcarriers'")
        
        center_freq = ofdm_config['center_frequency']
        bandwidth = ofdm_config['bandwidth']
        
        self.center_frequency = float(center_freq) if isinstance(center_freq, str) else center_freq
        self.bandwidth = float(bandwidth) if isinstance(bandwidth, str) else bandwidth
        self.num_subcarriers = ofdm_config['num_subcarriers']
        
        # Calculate wavelength and antenna spacing - no defaults
        self.wavelength = 3e8 / self.center_frequency
        
        if 'element_spacing' not in antenna_config:
            raise ValueError("Configuration must contain 'base_station.antenna_array.element_spacing'")
        
        element_spacing_type = antenna_config['element_spacing']
        
        if element_spacing_type == 'half_wavelength':
            self.dx = self.dy = 0.5 * self.wavelength
        elif element_spacing_type == 'custom':
            if 'custom_spacing' not in antenna_config:
                raise ValueError("Configuration must contain 'base_station.antenna_array.custom_spacing' when element_spacing is 'custom'")
            self.dx = self.dy = antenna_config['custom_spacing']
        else:
            raise ValueError(f"Unsupported element spacing type: {element_spacing_type}")
        
        # Parse angle ranges from [min, step, max] format - no defaults
        if 'theta_range' not in ssl_config:
            raise ValueError("Configuration must contain 'training.loss.spatial_spectrum_loss.theta_range'")
        if 'phi_range' not in ssl_config:
            raise ValueError("Configuration must contain 'training.loss.spatial_spectrum_loss.phi_range'")
        
        theta_range = ssl_config['theta_range']
        phi_range = ssl_config['phi_range']
        
        # Convert degrees to radians and generate grids
        theta_min, theta_step, theta_max = theta_range
        phi_min, phi_step, phi_max = phi_range
        
        # Convert to radians
        theta_min_rad = np.deg2rad(theta_min)
        theta_max_rad = np.deg2rad(theta_max)
        theta_step_rad = np.deg2rad(theta_step)
        
        phi_min_rad = np.deg2rad(phi_min)
        phi_max_rad = np.deg2rad(phi_max)
        phi_step_rad = np.deg2rad(phi_step)
        
        # Generate angle grids
        self.theta_grid = torch.arange(theta_min_rad, theta_max_rad + theta_step_rad/2, theta_step_rad)
        self.phi_grid = torch.arange(phi_min_rad, phi_max_rad + phi_step_rad/2, phi_step_rad)
        
        # Precompute angle combinations for vectorized processing
        theta_points = len(self.theta_grid)
        phi_points = len(self.phi_grid)
        theta_mesh, phi_mesh = torch.meshgrid(self.theta_grid, self.phi_grid, indexing='ij')
        self.theta_flat = theta_mesh.flatten()  # (num_angles,)
        self.phi_flat = phi_mesh.flatten()      # (num_angles,)
        self.num_angles = len(self.theta_flat)
        
        # Pre-compute antenna positions for steering vector calculation
        self.antenna_positions = self._compute_antenna_positions()
        
        logger.info(f"🔧 Spatial spectrum grid: {theta_points} × {phi_points} = {self.num_angles} angles")
        logger.info(f"🚀 Optimization: Precomputed angle combinations for vectorized processing")
    
    def _compute_antenna_positions(self) -> torch.Tensor:
        """
        Compute 3D positions of all antennas in the array
        
        Returns:
            positions: Tensor of antenna positions (num_antennas, 3) in meters
        """
        positions = []
        for m in range(self.M):
            for n in range(self.N):
                # Antenna position relative to array center
                x = (m - (self.M - 1) / 2) * self.dx
                y = (n - (self.N - 1) / 2) * self.dy
                z = 0.0  # Assume planar array
                positions.append([x, y, z])
        
        return torch.tensor(positions, dtype=torch.float32)
    
    def _compute_subcarrier_frequencies(self) -> torch.Tensor:
        """
        Compute frequencies for all subcarriers
        
        Returns:
            frequencies: Tensor of subcarrier frequencies (num_subcarriers,)
        """
        # Generate subcarrier indices centered around 0
        indices = torch.arange(self.num_subcarriers) - self.num_subcarriers // 2
        
        # Calculate frequency offsets
        subcarrier_spacing = self.bandwidth / self.num_subcarriers
        freq_offsets = indices * subcarrier_spacing
        
        # Calculate actual frequencies
        frequencies = self.center_frequency + freq_offsets
        
        return frequencies
    
    def _generate_steering_vectors(self, theta_batch: torch.Tensor, phi_batch: torch.Tensor, 
                                 frequency: float, device: torch.device) -> torch.Tensor:
        """
        Generate steering vectors for angles (supports both single and batch processing)
        
        Args:
            theta_batch: Elevation angles tensor (scalar or num_angles,) in radians
            phi_batch: Azimuth angles tensor (scalar or num_angles,) in radians
            frequency: Signal frequency in Hz
            device: Device for tensor computation
            
        Returns:
            steering_vectors: Complex steering vectors (num_antennas, 1) or (num_antennas, num_angles)
        """
        wavelength = 3e8 / frequency
        positions = self.antenna_positions.to(device)  # (num_antennas, 3)
        
        # Handle both scalar and batch inputs
        if theta_batch.dim() == 0:  # Scalar input
            theta_batch = theta_batch.unsqueeze(0)
            phi_batch = phi_batch.unsqueeze(0)
            single_output = True
        else:
            single_output = False
        
        # Compute direction vectors for all angles at once
        # directions: (num_angles, 3)
        directions = torch.stack([
            torch.sin(theta_batch) * torch.cos(phi_batch),  # x components
            torch.sin(theta_batch) * torch.sin(phi_batch),  # y components  
            torch.cos(theta_batch)                          # z components
        ], dim=1)  # (num_angles, 3)
        
        # Calculate phase shifts for all antennas and all angles
        k = 2 * np.pi / wavelength
        phase_shifts = k * torch.matmul(positions, directions.T)  # (num_antennas, num_angles)
        
        # Generate complex steering vectors for all angles
        steering_vectors = torch.exp(-1j * phase_shifts)  # (num_antennas, num_angles)
        
        # Return appropriate shape
        if single_output:
            return steering_vectors.unsqueeze(-1)  # (num_antennas, 1)
        else:
            return steering_vectors  # (num_antennas, num_angles)
    
    def _compute_spatial_spectrum(self, csi: torch.Tensor, subcarrier_idx: int) -> torch.Tensor:
        """
        Compute spatial spectrum from CSI for a single subcarrier (optimized vectorized version)
        
        Args:
            csi: CSI tensor for single subcarrier (num_antennas, 1) - single snapshot
            subcarrier_idx: Subcarrier index for frequency calculation
            
        Returns:
            spectrum: Spatial spectrum (theta_points, phi_points)
        """
        device = csi.device
        
        # Get frequency for this subcarrier
        subcarrier_frequencies = self._compute_subcarrier_frequencies()
        frequency = subcarrier_frequencies[subcarrier_idx].item()
        
        # Normalize CSI magnitude to reduce numerical issues
        # Keep phase information but normalize magnitude
        csi_magnitude = torch.abs(csi) + 1e-8  # Add small epsilon to avoid division by zero
        csi_normalized = csi / csi_magnitude  # Magnitude = 1, phase preserved
        
        # For single snapshot, covariance matrix is R = csi * csi^H
        R_xx = torch.matmul(csi_normalized, torch.conj(csi_normalized).transpose(-2, -1))  # (num_antennas, num_antennas)
        
        # Use precomputed angle combinations
        theta_flat = self.theta_flat.to(device)
        phi_flat = self.phi_flat.to(device)
        
        # Generate all steering vectors at once using unified method
        all_steering_vectors = self._generate_steering_vectors(theta_flat, phi_flat, frequency, device)
        # Shape: (num_antennas, num_angles)
        
        if self.algorithm == 'bartlett':
            # Vectorized Bartlett beamformer: a^H * R * a for all angles
            R_A = torch.matmul(R_xx, all_steering_vectors)  # (num_antennas, num_angles)
            powers = torch.real(torch.sum(torch.conj(all_steering_vectors) * R_A, dim=0))  # (num_angles,)
            
            # Reshape back to (theta_points, phi_points)
            theta_points = len(self.theta_grid)
            phi_points = len(self.phi_grid)
            spectrum = powers.reshape(theta_points, phi_points)
        else:
            raise NotImplementedError(f"Algorithm '{self.algorithm}' not implemented")
        
        return spectrum
    
    def _compute_spectrum_all_subcarriers(self, csi_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Compute spatial spectrum for all subcarriers at once (optimized)
        
        Args:
            csi_batch: CSI tensor for one batch sample (num_subcarriers, 1, num_bs_antennas)
            device: Device for computation
            
        Returns:
            fused_spectrum: Fused spatial spectrum (theta_points, phi_points)
        """
        num_subcarriers = csi_batch.shape[0]
        theta_points = len(self.theta_grid)
        phi_points = len(self.phi_grid)
        
        # Use precomputed angle combinations
        theta_flat = self.theta_flat.to(device)
        phi_flat = self.phi_flat.to(device)
        
        # Get all subcarrier frequencies
        subcarrier_frequencies = self._compute_subcarrier_frequencies()
        
        # Initialize spectrum accumulator
        if self.fusion_method == 'average':
            spectrum_accumulator = torch.zeros((theta_points, phi_points), device=device, dtype=torch.float32)
        elif self.fusion_method == 'max':
            spectrum_accumulator = torch.full((theta_points, phi_points), -float('inf'), device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        # Process all subcarriers
        for k in range(num_subcarriers):
            # Get CSI for this subcarrier: (num_bs_antennas, 1)
            csi_k = csi_batch[k, 0, :].unsqueeze(1)  # (num_bs_antennas, 1)
            frequency = subcarrier_frequencies[k].item()
            
            # Normalize CSI magnitude to use only phase information
            # Keep phase information but normalize magnitude to 1
            csi_k_magnitude = torch.abs(csi_k) + 1e-8  # Add small epsilon to avoid division by zero
            csi_k_normalized = csi_k / csi_k_magnitude  # Magnitude = 1, phase preserved
            
            # Covariance matrix using normalized CSI (phase-only)
            R_xx = torch.matmul(csi_k_normalized, torch.conj(csi_k_normalized).transpose(-2, -1))  # (num_bs_antennas, num_bs_antennas)
            
            # Generate all steering vectors for this frequency using unified method
            all_steering_vectors = self._generate_steering_vectors(theta_flat, phi_flat, frequency, device)
            # Shape: (num_bs_antennas, num_angles)
            
            if self.algorithm == 'bartlett':
                # Vectorized Bartlett beamformer for all angles
                R_A = torch.matmul(R_xx, all_steering_vectors)  # (num_bs_antennas, num_angles)
                powers = torch.real(torch.sum(torch.conj(all_steering_vectors) * R_A, dim=0))  # (num_angles,)
                spectrum_k = powers.reshape(theta_points, phi_points)
            else:
                raise NotImplementedError(f"Algorithm '{self.algorithm}' not implemented")
            
            # Accumulate spectrum
            if self.fusion_method == 'average':
                spectrum_accumulator += spectrum_k
            elif self.fusion_method == 'max':
                spectrum_accumulator = torch.maximum(spectrum_accumulator, spectrum_k)
        
        # Finalize fusion
        if self.fusion_method == 'average':
            fused_spectrum = spectrum_accumulator / num_subcarriers
        elif self.fusion_method == 'max':
            fused_spectrum = spectrum_accumulator
        
        return fused_spectrum
    
    def _csi_to_spatial_spectrum(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Convert CSI tensor to spatial spectrum
        
        Args:
            csi: CSI tensor with shape (batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas)
            
        Returns:
            spectrum: Spatial spectrum tensor (batch_size, theta_points, phi_points)
        """
        batch_size = csi.shape[0]
        device = csi.device
        
        # Validate CSI tensor dimensions
        
        # Validate input shape
        if csi.dim() != 4:
            raise ValueError(f"Expected 4D CSI tensor (batch, subcarriers, ue_antennas, bs_antennas), got {csi.shape}")
        
        num_subcarriers, num_ue_antennas, num_bs_antennas = csi.shape[1], csi.shape[2], csi.shape[3]
        
        # Extract single UE antenna (should always be 1 after data loading)
        if num_ue_antennas != 1:
            logger.warning(f"Expected single UE antenna, got {num_ue_antennas}. Using first antenna.")
            csi = csi[:, :, :1, :]  # Take only the first UE antenna
            num_ue_antennas = 1
        
        # Validate BS antenna count matches array configuration
        if num_bs_antennas != self.num_antennas:
            raise ValueError(f"BS antennas ({num_bs_antennas}) doesn't match array config ({self.M}x{self.N}={self.num_antennas})")
        
        # Initialize output spectrum
        theta_points = len(self.theta_grid)
        phi_points = len(self.phi_grid)
        batch_spectrums = torch.zeros((batch_size, theta_points, phi_points), 
                                    device=device, dtype=torch.float32)
        
        # Process each batch sample
        for b in range(batch_size):
            # Get CSI for this batch: (num_subcarriers, 1, num_bs_antennas)
            csi_batch = csi[b]  # (num_subcarriers, 1, num_bs_antennas)
            
            # Vectorized computation for all subcarriers at once
            fused_spectrum = self._compute_spectrum_all_subcarriers(csi_batch, device)
            batch_spectrums[b] = fused_spectrum
        
        return batch_spectrums
        
    def forward(self, predicted_csi: torch.Tensor, target_csi: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute spatial spectrum loss between predicted and target CSI
        
        Args:
            predicted_csi: Predicted CSI tensor 
                          Shape: (batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas)
            target_csi: Target CSI tensor
                       Shape: same as predicted_csi
            mask: Optional mask for selective loss computation (not currently used)
        
        Returns:
            loss: Computed MSE loss between spatial spectrums (scalar tensor)
        """
        # Validate input shapes
        
        if predicted_csi.shape != target_csi.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_csi.shape} vs target {target_csi.shape}")
        
        # Convert CSI to spatial spectrum
        predicted_spectrum = self._csi_to_spatial_spectrum(predicted_csi)
        target_spectrum = self._csi_to_spatial_spectrum(target_csi)
        
        # Apply mask if provided
        if mask is not None:
            if mask.shape != predicted_spectrum.shape:
                raise ValueError(f"Mask shape {mask.shape} doesn't match spectrum shape {predicted_spectrum.shape}")
            predicted_spectrum = predicted_spectrum * mask
            target_spectrum = target_spectrum * mask
        
        # Normalize spectrums to reduce magnitude differences
        # Use L2 normalization to make spectrums comparable
        pred_norm = torch.norm(predicted_spectrum.flatten(), p=2)
        target_norm = torch.norm(target_spectrum.flatten(), p=2)
        
        if pred_norm > 1e-8 and target_norm > 1e-8:
            predicted_spectrum_normalized = predicted_spectrum / pred_norm
            target_spectrum_normalized = target_spectrum / target_norm
        else:
            # Fallback if normalization fails
            predicted_spectrum_normalized = predicted_spectrum
            target_spectrum_normalized = target_spectrum
        
        # Compute MSE loss between normalized spatial spectrums
        loss = F.mse_loss(predicted_spectrum_normalized, target_spectrum_normalized)
        
        return loss
    
    def compute_and_visualize_loss(self, predicted_csi: torch.Tensor, target_csi: torch.Tensor,
                                  save_path: str, sample_idx: int = 0, 
                                  mask: Optional[torch.Tensor] = None) -> Tuple[float, str]:
        """
        Compute spatial spectrum loss and save visualization for testing
        
        Args:
            predicted_csi: Predicted CSI tensor 
                          Shape: (batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas)
            target_csi: Target CSI tensor (same shape as predicted_csi)
            save_path: Directory path to save the visualization plot
            sample_idx: Which sample in the batch to visualize (default: 0)
            mask: Optional mask for selective loss computation
        
        Returns:
            loss_value: Computed loss value (float)
            plot_path: Path to the saved plot file
        """
        if predicted_csi.shape != target_csi.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_csi.shape} vs target {target_csi.shape}")
        
        # Compute loss
        loss = self.forward(predicted_csi, target_csi, mask)
        loss_value = loss.item()
        
        # Convert CSI to spatial spectrum for visualization
        predicted_spectrum = self._csi_to_spatial_spectrum(predicted_csi)
        target_spectrum = self._csi_to_spatial_spectrum(target_csi)
        
        # Select sample to visualize
        if sample_idx >= predicted_spectrum.shape[0]:
            sample_idx = 0
        
        pred_spec = predicted_spectrum[sample_idx].detach().cpu().numpy()
        target_spec = target_spectrum[sample_idx].detach().cpu().numpy()
        
        # Create visualization
        plot_path = self._create_spectrum_comparison_plot(
            pred_spec, target_spec, save_path, loss_value, sample_idx
        )
        
        return loss_value, plot_path
    
    def _create_spectrum_comparison_plot(self, predicted_spectrum: np.ndarray, 
                                       target_spectrum: np.ndarray,
                                       save_path: str, loss_value: float,
                                       sample_idx: int) -> str:
        """
        Create and save spatial spectrum comparison plot
        
        Args:
            predicted_spectrum: Predicted spatial spectrum (theta_points, phi_points)
            target_spectrum: Target spatial spectrum (theta_points, phi_points)
            save_path: Directory to save the plot
            loss_value: Loss value to display in title
            sample_idx: Sample index for filename
            
        Returns:
            plot_path: Path to saved plot file
        """
        # Create save directory if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Convert angle grids to degrees for display
        theta_deg = np.rad2deg(self.theta_grid.cpu().numpy())
        phi_deg = np.rad2deg(self.phi_grid.cpu().numpy())
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot predicted spectrum
        im1 = ax1.imshow(predicted_spectrum, aspect='auto', origin='lower', 
                        extent=[phi_deg[0], phi_deg[-1], theta_deg[0], theta_deg[-1]],
                        cmap='viridis')
        ax1.set_title(f'Predicted Spatial Spectrum\nSample {sample_idx}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Azimuth (degrees)', fontsize=10)
        ax1.set_ylabel('Elevation (degrees)', fontsize=10)
        plt.colorbar(im1, ax=ax1, label='Power')
        
        # Plot target spectrum
        im2 = ax2.imshow(target_spectrum, aspect='auto', origin='lower',
                        extent=[phi_deg[0], phi_deg[-1], theta_deg[0], theta_deg[-1]],
                        cmap='viridis')
        ax2.set_title(f'Target Spatial Spectrum\nSample {sample_idx}', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Azimuth (degrees)', fontsize=10)
        ax2.set_ylabel('Elevation (degrees)', fontsize=10)
        plt.colorbar(im2, ax=ax2, label='Power')
        
        # Add overall title with loss information
        fig.suptitle(f'Spatial Spectrum Comparison - Loss: {loss_value:.6f}\n'
                    f'Algorithm: {self.algorithm.upper()}, Fusion: {self.fusion_method}', 
                    fontsize=14, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        filename = f'spatial_spectrum_comparison_sample_{sample_idx}.png'
        plot_path = os.path.join(save_path, filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path



class PrismLossFunction(nn.Module):
    """
    Main Loss Function Class for Prism Framework
    
    Combines CSI and PDP losses with configurable weights
    and provides a unified interface for training.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Prism loss function
        
        Args:
            config: Configuration dictionary containing loss parameters
        """
        super(PrismLossFunction, self).__init__()
        
        # Extract configuration with reasonable defaults for loss weights
        # These are algorithm parameters, not critical system config
        self.csi_weight = config.get('csi_weight', 0.7)
        self.pdp_weight = config.get('pdp_weight', 0.3)
        self.spatial_spectrum_weight = config.get('spatial_spectrum_weight', 0.0)
        self.regularization_weight = config.get('regularization_weight', 0.01)
        
        # Initialize component losses with reasonable defaults
        csi_config = config.get('csi_loss', {})
        self.csi_enabled = csi_config.get('enabled', True)  # Default enabled for backward compatibility
        self.csi_loss = CSILoss(
            loss_type=csi_config.get('type', 'hybrid'),
            phase_weight=csi_config.get('phase_weight', 1.0),
            magnitude_weight=csi_config.get('magnitude_weight', 1.0),
            cmse_weight=csi_config.get('cmse_weight', 1.0)
            # correlation_weight parameter removed
        )
        
        # Initialize PDP loss
        pdp_config = config.get('pdp_loss', {})
        self.pdp_enabled = pdp_config.get('enabled', True)  # Default enabled for backward compatibility
        self.pdp_loss = PDPLoss(
            loss_type=pdp_config.get('type', 'hybrid'),
            fft_size=pdp_config.get('fft_size', 1024),
            normalize_pdp=pdp_config.get('normalize_pdp', True)
            # mse_weight, correlation_weight, delay_weight parameters removed
        )
        
        # Initialize regularization loss enabled flag
        reg_config = config.get('regularization_loss', {})
        self.regularization_enabled = reg_config.get('enabled', True)  # Default enabled for backward compatibility
        
        # Initialize Spatial Spectrum loss (only if enabled and weight > 0)
        ssl_config = config.get('spatial_spectrum_loss', {})
        self.spatial_spectrum_enabled = ssl_config.get('enabled', False)
        self.spatial_spectrum_loss = None
        if self.spatial_spectrum_weight > 0 and self.spatial_spectrum_enabled:
            # Pass the full config to SpatialSpectrumLoss (it needs base_station and training sections)
            full_config = {'base_station': config.get('base_station', {}), 
                          'training': {'loss': {'spatial_spectrum_loss': ssl_config}}}
            self.spatial_spectrum_loss = SpatialSpectrumLoss(full_config)
        
        # Loss components tracking
        self.loss_components = {}
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                masks: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss
        
        Args:
            predictions: Dictionary containing predicted values
                        - 'csi': Predicted CSI tensor
            targets: Dictionary containing target values
                    - 'csi': Target CSI tensor
            masks: Optional masks for selective loss computation
        
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary of individual loss components
        """
        # Initialize total_loss properly to maintain gradients
        # Use a tensor derived from predictions to ensure gradient flow, but ensure it's real
        total_loss = torch.real(predictions['csi'].sum()) * 0.0  # This maintains gradient connection to predictions
        loss_components = {}
        
        if masks is None:
            masks = {}
        
        # CSI loss (hybrid: CMSE + Correlation) - use traced CSI for subcarrier-level loss
        if ('traced_csi' in predictions and 'traced_csi' in targets and self.csi_enabled):
            # Convert traced CSI back to tensor format for CSI loss
            traced_pred = predictions['traced_csi']
            traced_target = targets['traced_csi']
            
            # CSI loss expects 1D tensors of traced subcarriers
            # Handle complex tensors by manually computing MSE to avoid CUDA issues
            if traced_pred.is_complex():
                # Compute MSE manually for complex tensors
                diff = traced_pred - traced_target
                csi_loss_val = torch.mean(torch.abs(diff) ** 2)
            else:
                csi_loss_val = F.mse_loss(traced_pred, traced_target)
            total_loss = total_loss + self.csi_weight * csi_loss_val
            loss_components['csi_loss'] = csi_loss_val.item()
        
        # PDP loss (hybrid: MSE + Correlation + Delay) - use full CSI for frequency domain analysis
        if ('csi' in predictions and 'csi' in targets and 
            self.pdp_enabled and self.pdp_weight > 0):
            pdp_loss_val = self.pdp_loss(
                predictions['csi'], 
                targets['csi']
            )
            total_loss = total_loss + self.pdp_weight * pdp_loss_val
            loss_components['pdp_loss'] = pdp_loss_val.item()
        
        # Spatial Spectrum loss
        if (self.spatial_spectrum_loss is not None and 
            self.spatial_spectrum_enabled and
            'csi' in predictions and 'csi' in targets and 
            self.spatial_spectrum_weight > 0):
            # Use full 4D CSI tensors for spatial spectrum loss
            spatial_loss_val = self.spatial_spectrum_loss(
                predictions['csi'], 
                targets['csi'],
                masks.get('spatial_spectrum') if masks else None
            )
            total_loss = total_loss + self.spatial_spectrum_weight * spatial_loss_val
            loss_components['spatial_spectrum_loss'] = spatial_loss_val.item()
        
        # Regularization losses
        if ('regularization' in predictions and 
            self.regularization_enabled and self.regularization_weight > 0):
            reg_loss = predictions['regularization']
            total_loss = total_loss + self.regularization_weight * reg_loss
            loss_components['regularization_loss'] = reg_loss.item()
        
        loss_components['total_loss'] = total_loss.item()
        self.loss_components = loss_components
        
        return total_loss, loss_components
    
    def get_loss_components(self) -> Dict[str, float]:
        """
        Get the most recent loss components
        
        Returns:
            loss_components: Dictionary of loss component values
        """
        return self.loss_components.copy()
    
    def compute_and_visualize_spatial_spectrum_loss(self, predicted_csi: torch.Tensor, 
                                                   target_csi: torch.Tensor,
                                                   save_path: str, sample_idx: int = 0) -> Optional[Tuple[float, str]]:
        """
        Compute spatial spectrum loss and create visualization (for testing)
        
        Args:
            predicted_csi: Predicted CSI tensor
            target_csi: Target CSI tensor  
            save_path: Directory to save visualization
            sample_idx: Sample index to visualize
            
        Returns:
            (loss_value, plot_path) if spatial spectrum loss is enabled, None otherwise
        """
        if self.spatial_spectrum_loss is None:
            return None
            
        return self.spatial_spectrum_loss.compute_and_visualize_loss(
            predicted_csi, target_csi, save_path, sample_idx
        )






