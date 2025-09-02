"""
Spatial Spectrum Loss Functions

Computes MSE loss between spatial spectrum matrices derived from CSI data.
Extracts configuration from base_station.antenna_array and training.loss.spatial_spectrum_loss sections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Get logger for this module
logger = logging.getLogger(__name__)


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
    
    def _compute_spectrum_all_subcarriers(self, csi_batch: torch.Tensor, device: torch.device, 
                                        subcarrier_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute spatial spectrum for selected subcarriers only (optimized)
        
        Args:
            csi_batch: CSI tensor for one batch sample (num_subcarriers, 1, num_bs_antennas)
            device: Device for computation
            subcarrier_mask: Boolean mask indicating which subcarriers to use (num_subcarriers,)
            
        Returns:
            fused_spectrum: Fused spatial spectrum (theta_points, phi_points)
        """
        num_subcarriers = csi_batch.shape[0]
        theta_points = len(self.theta_grid)
        phi_points = len(self.phi_grid)
        
        # Determine which subcarriers to use
        if subcarrier_mask is not None:
            # Use provided mask to select subcarriers
            valid_indices = torch.where(subcarrier_mask)[0].tolist()
        else:
            # Use all subcarriers in the input (assumes they are already selected)
            valid_indices = list(range(num_subcarriers))
        
        if len(valid_indices) == 0:
            # No valid subcarriers, return zero spectrum
            return torch.zeros((theta_points, phi_points), device=device, dtype=torch.float32)
        
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
        
        # Process only selected subcarriers
        valid_count = 0
        for i, k in enumerate(valid_indices):
            # Get CSI for this subcarrier: (num_bs_antennas, 1)
            csi_k = csi_batch[k, 0, :].unsqueeze(1)  # (num_bs_antennas, 1)
            
            # For frequency calculation:
            # If we have a mask, k is the original subcarrier index
            # If no mask, k is just the index in the selected data, use center frequency
            if subcarrier_mask is not None:
                frequency = subcarrier_frequencies[k].item()
            else:
                # Use center frequency for simplicity when processing already-selected data
                frequency = self.center_frequency
            
            # Skip zero-valued subcarriers
            csi_magnitude = torch.abs(csi_k)
            if torch.max(csi_magnitude) < 1e-10:
                continue
            
            valid_count += 1
            
            # Normalize CSI magnitude to use only phase information
            csi_k_normalized = csi_k / (csi_magnitude + 1e-8)  # Magnitude = 1, phase preserved
            
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
        
        # Finalize fusion (use actual number of valid subcarriers)
        if self.fusion_method == 'average' and valid_count > 0:
            fused_spectrum = spectrum_accumulator / valid_count
        elif self.fusion_method == 'max':
            fused_spectrum = spectrum_accumulator
        else:
            # No valid subcarriers processed
            fused_spectrum = torch.zeros((theta_points, phi_points), device=device, dtype=torch.float32)
        
        return fused_spectrum
    
    def _csi_to_spatial_spectrum(self, csi: torch.Tensor, subcarrier_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convert CSI tensor to spatial spectrum
        
        Args:
            csi: CSI tensor with shape (batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas)
                 If subcarrier_mask is None, assumes all subcarriers in csi are already selected
            subcarrier_mask: Optional mask indicating which subcarriers to use 
                           Shape: (batch_size, num_ue_antennas, num_subcarriers) or (batch_size, num_subcarriers)
                           If None, uses all subcarriers in the input csi tensor
            
        Returns:
            spectrum: Spatial spectrum tensor (batch_size, theta_points, phi_points)
        """
        batch_size = csi.shape[0]
        device = csi.device
        
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
            
            # Get subcarrier mask for this batch
            batch_mask = None
            if subcarrier_mask is not None:
                if subcarrier_mask.dim() == 3:
                    # Shape: (batch_size, num_ue_antennas, num_subcarriers)
                    batch_mask = subcarrier_mask[b, 0, :]  # Use first UE antenna
                elif subcarrier_mask.dim() == 2:
                    # Shape: (batch_size, num_subcarriers)
                    batch_mask = subcarrier_mask[b, :]
                else:
                    logger.warning(f"Unexpected subcarrier_mask shape: {subcarrier_mask.shape}")
            
            # Vectorized computation for selected subcarriers only
            fused_spectrum = self._compute_spectrum_all_subcarriers(csi_batch, device, batch_mask)
            batch_spectrums[b] = fused_spectrum
        
        return batch_spectrums
        
    def forward(self, predicted_csi: torch.Tensor, target_csi: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial spectrum loss between predicted and target CSI
        
        Args:
            predicted_csi: Predicted CSI tensor from selected subcarriers
                          Shape: (batch_size, num_selected_subcarriers, num_ue_antennas, num_bs_antennas)
            target_csi: Target CSI tensor from selected subcarriers
                       Shape: same as predicted_csi
        
        Returns:
            loss: Computed MSE loss between spatial spectrums (scalar tensor)
        """
        # Validate input shapes
        if predicted_csi.shape != target_csi.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_csi.shape} vs target {target_csi.shape}")
        
        # Ensure complex tensors
        if not predicted_csi.is_complex():
            predicted_csi = predicted_csi.to(torch.complex64)
        if not target_csi.is_complex():
            target_csi = target_csi.to(torch.complex64)
        
        # Convert CSI to spatial spectrum (no mask needed since data is already selected)
        predicted_spectrum = self._csi_to_spatial_spectrum(predicted_csi)
        target_spectrum = self._csi_to_spatial_spectrum(target_csi)
        
        # Normalize spectrums using maximum value normalization
        # This preserves relative magnitude information better than L2 normalization
        pred_max = torch.max(predicted_spectrum)
        target_max = torch.max(target_spectrum)
        
        if pred_max > 1e-8 and target_max > 1e-8:
            predicted_spectrum_normalized = predicted_spectrum / pred_max
            target_spectrum_normalized = target_spectrum / target_max
        else:
            # Fallback if normalization fails
            predicted_spectrum_normalized = predicted_spectrum
            target_spectrum_normalized = target_spectrum
        
        # Compute MSE loss between max-normalized spatial spectrums
        loss = F.mse_loss(predicted_spectrum_normalized, target_spectrum_normalized)
        
        return loss
    
    def compute_and_visualize_loss(self, predicted_csi: torch.Tensor, target_csi: torch.Tensor,
                                  save_path: str, sample_idx: int = 0) -> Tuple[float, str]:
        """
        Compute spatial spectrum loss and save visualization for testing
        
        Args:
            predicted_csi: Predicted CSI tensor from selected subcarriers
                          Shape: (batch_size, num_selected_subcarriers, num_ue_antennas, num_bs_antennas)
            target_csi: Target CSI tensor from selected subcarriers (same shape as predicted_csi)
            save_path: Directory path to save the visualization plot
            sample_idx: Which sample in the batch to visualize (default: 0)
        
        Returns:
            loss_value: Computed loss value (float)
            plot_path: Path to the saved plot file
        """
        if predicted_csi.shape != target_csi.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_csi.shape} vs target {target_csi.shape}")
        
        # Compute loss (no mask needed since data is already selected)
        loss = self.forward(predicted_csi, target_csi)
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
