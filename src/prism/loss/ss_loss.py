"""
Spatial Spectrum Loss Functions

Simplified implementation: CSI -> Spatial Spectrum -> SSIM Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SSLoss(nn.Module):
    """
    Spatial Spectrum Loss using SSIM
    
    Core workflow:
    1. CSI -> Spatial Spectrum (using Bartlett beamforming)
    2. SSIM loss between predicted and target spectrums
    
    Configuration options:
    - orientation: 'bs' (default), 'ue', or 'orientation' - choose which antenna array to use for spectrum calculation
      - 'bs': Use BS antenna array configuration (from base_station.antenna_array.configuration)
      - 'ue': Use UE antenna array configuration (from user_equipment.num_ue_antennas)
      - 'orientation': Use orientation-based antenna array configuration (from base_station.antenna_array.configuration)
    """
    
    def __init__(self, config: Dict):
        """Initialize spatial spectrum loss from configuration"""
        super(SSLoss, self).__init__()
        
        # Extract configurations
        bs_config = config['base_station']
        ue_config = config['user_equipment']
        ssl_config = config['training']['loss']['spatial_spectrum_loss']
        
        # Antenna array configuration - choose between BS, UE, or orientation array
        self.orientation = ssl_config.get('orientation', 'bs')  # 'bs', 'ue', or 'orientation'
        
        if self.orientation == 'bs':
            # Use BS antenna array configuration
            array_config = bs_config['antenna_array']['configuration']  # e.g., "8x8"
            self.M, self.N = map(int, array_config.split('x'))
            self.num_antennas = self.M * self.N
            logger.info(f"🔧 SSLoss: Using BS antenna array ({self.M}x{self.N}={self.num_antennas} antennas)")
        else:
            # Use UE antenna array configuration
            self.num_antennas = ue_config['num_ue_antennas']
            # Get UE antenna array configuration
            if 'antenna_array' in ue_config and 'configuration' in ue_config['antenna_array']:
                array_config = ue_config['antenna_array']['configuration']  # e.g., "2x4"
                self.M, self.N = map(int, array_config.split('x'))
                logger.info(f"🔧 SSLoss: Using UE antenna array ({self.M}x{self.N}={self.num_antennas} antennas)")
            else:
                # Fallback to linear array (1xN)
                self.M = 1
                self.N = self.num_antennas
                logger.info(f"🔧 SSLoss: Using UE antenna array (1x{self.N}={self.num_antennas} antennas, linear array)")
        
        # OFDM parameters
        ofdm_config = bs_config['ofdm']
        self.center_frequency = float(ofdm_config['center_frequency'])
        self.bandwidth = float(ofdm_config['bandwidth'])
        self.num_subcarriers = ofdm_config['num_subcarriers']
        
        # Physical parameters
        self.wavelength = 3e8 / self.center_frequency
        self.dx = self.dy = 0.5 * self.wavelength  # Half-wavelength spacing
        
        # Angle ranges
        theta_range = ssl_config['theta_range']  # [min, step, max]
        phi_range = ssl_config['phi_range']
        
        # SSIM parameters
        self.ssim_window_size = ssl_config.get('ssim_window_size', 11)
        self.ssim_k1 = ssl_config.get('ssim_k1', 0.01)
        self.ssim_k2 = ssl_config.get('ssim_k2', 0.03)
        
        # Spectrum computation method
        self.use_covariance_bartlett = ssl_config.get('use_covariance_bartlett', True)
        
        # Frequency-dependent steering
        self.use_frequency_dependent_steering = ssl_config.get('use_frequency_dependent_steering', True)
        
        # Subcarrier sampling for performance optimization
        self.num_sampled_subcarriers = ssl_config.get('num_sampled_subcarriers', 20)
        
        # 预计算固定的子载波采样索引，确保GT和预测保持一致
        if self.num_sampled_subcarriers < self.num_subcarriers:
            # 均匀采样子载波索引
            sampled_indices = torch.linspace(0, self.num_subcarriers - 1, self.num_sampled_subcarriers, dtype=torch.long)
            self.register_buffer('sampled_subcarrier_indices', sampled_indices)
            logger.info(f"子载波采样: 从{self.num_subcarriers}个中均匀采样{self.num_sampled_subcarriers}个")
        else:
            # 如果采样数量大于等于总数，使用所有子载波
            self.register_buffer('sampled_subcarrier_indices', torch.arange(self.num_subcarriers))
            logger.info(f"使用所有{self.num_subcarriers}个子载波")
        
        # Create angle grids and register as buffers first
        theta_grid = self._create_angle_grid(*theta_range)
        phi_grid = self._create_angle_grid(*phi_range)
        self.register_buffer('theta_grid', theta_grid)
        self.register_buffer('phi_grid', phi_grid)
        
        # Compute and register antenna positions first
        antenna_positions = self._compute_antenna_positions()
        self.register_buffer('antenna_positions', antenna_positions)
        
        # Precompute steering vectors for center frequency (after antenna_positions is registered)
        steering_vectors = self._precompute_steering_vectors()
        self.register_buffer('steering_vectors', steering_vectors)
        
        # Precompute subcarrier frequencies if using frequency-dependent steering
        if self.use_frequency_dependent_steering:
            subcarrier_frequencies = self._compute_subcarrier_frequencies()
            self.register_buffer('subcarrier_frequencies', subcarrier_frequencies)
        
        logger.info(f"🔧 SSLoss: {len(self.theta_grid)}×{len(self.phi_grid)} angles "
                   f"(θ: {theta_range[0]}°-{theta_range[2]}° step {theta_range[1]}°, "
                   f"φ: {phi_range[0]}°-{phi_range[2]}° step {phi_range[1]}°), "
                   f"{self.M}×{self.N} antenna array, "
                   f"method: {'covariance-Bartlett' if self.use_covariance_bartlett else 'classical-Bartlett'}, "
                   f"frequency-dependent: {self.use_frequency_dependent_steering}")
    
    def _create_angle_grid(self, min_deg: float, step_deg: float, max_deg: float) -> torch.Tensor:
        """Create angle grid in radians"""
        # Use arange to avoid endpoint duplication for azimuth
        angles_deg = torch.arange(min_deg, max_deg, step_deg)
        if len(angles_deg) == 0 or angles_deg[-1] + step_deg <= max_deg + 1e-6:
            # Include max_deg if it's exactly reachable
            angles_deg = torch.cat([angles_deg, torch.tensor([max_deg])])
        return torch.deg2rad(angles_deg)
    
    def _compute_antenna_positions(self) -> torch.Tensor:
        """Compute 3D positions of all antennas in the array"""
        # Create antenna grid centered at origin
        m_range = torch.arange(self.M, dtype=torch.float32) - (self.M - 1) / 2
        n_range = torch.arange(self.N, dtype=torch.float32) - (self.N - 1) / 2
        
        m_grid, n_grid = torch.meshgrid(m_range, n_range, indexing='ij')
        
        # Calculate positions
        x = m_grid.flatten() * self.dx
        y = n_grid.flatten() * self.dy
        z = torch.zeros_like(x)  # Planar array
        
        return torch.stack([x, y, z], dim=1)  # [num_antennas, 3]
    
    def _compute_subcarrier_frequencies(self) -> torch.Tensor:
        """Compute frequencies for all subcarriers"""
        # Frequency spacing
        delta_f = self.bandwidth / self.num_subcarriers
        
        # Subcarrier indices (centered around 0)
        subcarrier_indices = torch.arange(self.num_subcarriers, dtype=torch.float32) - (self.num_subcarriers - 1) / 2
        
        # Frequencies for each subcarrier
        frequencies = self.center_frequency + subcarrier_indices * delta_f
        
        return frequencies
    
    def _precompute_steering_vectors(self) -> torch.Tensor:
        """Precompute steering vectors for all angle combinations"""
        # Create angle meshgrid
        theta_mesh, phi_mesh = torch.meshgrid(self.theta_grid, self.phi_grid, indexing='ij')
        theta_flat = theta_mesh.flatten()
        phi_flat = phi_mesh.flatten()
        
        # Compute direction vectors for all angles
        directions = torch.stack([
            torch.sin(theta_flat) * torch.cos(phi_flat),  # x
            torch.sin(theta_flat) * torch.sin(phi_flat),  # y
            torch.cos(theta_flat)                         # z
        ], dim=1)  # [num_angles, 3]
        
        # Compute steering vectors: exp(+j * k * r · d)
        # Note: Using positive phase for standard array response definition
        k = 2 * np.pi / self.wavelength
        phase_shifts = k * torch.matmul(self.antenna_positions, directions.T)  # [num_antennas, num_angles]
        steering_vectors = torch.exp(1j * phase_shifts)
        
        # Reshape to [theta_points, phi_points, num_antennas]
        return steering_vectors.T.reshape(len(self.theta_grid), len(self.phi_grid), self.num_antennas)
    
    def _csi_to_spatial_spectrum(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Convert CSI to spatial spectrum using Bartlett beamforming
        
        Two methods available:
        1. Covariance-based Bartlett (more robust): R = E[x*x^H], spectrum = a^H * R * a
        2. Classical averaged power: spectrum = mean_s |a^H * x_s|^2
        
        Args:
            csi: CSI tensor [batch_size, bs_antennas, subcarriers] or [batch_size, bs_antennas, ue_antennas, subcarriers]
                - Data is always in BS antenna format, but orientation config determines which array to use
                - If 4D format, it will be converted to 3D for spatial spectrum calculation
            
        Returns:
            spectrum: Spatial spectrum [batch_size, theta_points, phi_points]
        """
        device = csi.device
        
        # Handle CSI format conversion if needed
        if csi.dim() == 4:
            # 4D format: [batch_size, bs_antennas, ue_antennas, subcarriers]
            # Convert to 3D format for spatial spectrum calculation
            batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = csi.shape
            
            logger.debug(f"🔄 Converting 4D CSI to 3D for spatial spectrum: {csi.shape}")
            
            # For spatial spectrum calculation, we need to decide which antenna array to use
            if self.orientation == 'ue':
                # Use UE antenna array: reshape to [batch_size, ue_antennas, bs_antennas * subcarriers]
                csi_data = csi.permute(0, 2, 1, 3)  # [batch, ue_antennas, bs_antennas, subcarriers]
                csi_data = csi_data.reshape(batch_size, num_ue_antennas, -1)  # [batch, ue_antennas, bs_antennas * subcarriers]
                expected_antennas = num_ue_antennas
            else:
                # Use BS antenna array: reshape to [batch_size, bs_antennas, ue_antennas * subcarriers]
                csi_data = csi.reshape(batch_size, num_bs_antennas, -1)  # [batch, bs_antennas, ue_antennas * subcarriers]
                expected_antennas = num_bs_antennas
            
            logger.debug(f"   Converted to 3D format: {csi_data.shape}")
            
        elif csi.dim() == 3:
            # 3D format: [batch_size, antennas, subcarriers]
            batch_size, num_antennas, num_subcarriers = csi.shape
            csi_data = csi
            expected_antennas = num_antennas
            logger.debug(f"📊 Using 4D CSI format: {csi_data.shape}")
            
            # Special handling for 3D format when UE antennas were converted to subcarriers
            if self.orientation == 'ue' and expected_antennas != self.num_antennas:
                # This is likely a case where UE antennas were converted to subcarriers
                # We need to reconstruct the UE antenna dimension
                logger.debug(f"🔄 Detected UE antenna conversion case: {expected_antennas} antennas, expected {self.num_antennas}")
                
                # Try to reconstruct UE antenna structure
                # Assume the subcarriers are organized as: [ue_antennas * original_subcarriers]
                if num_subcarriers % self.num_antennas == 0:
                    # Reconstruct UE antenna structure
                    original_subcarriers_per_ue = num_subcarriers // self.num_antennas
                    csi_data = csi_data.reshape(batch_size, expected_antennas, self.num_antennas, original_subcarriers_per_ue)
                    csi_data = csi_data.permute(0, 2, 1, 3)  # [batch, ue_antennas, bs_antennas, subcarriers]
                    csi_data = csi_data.reshape(batch_size, self.num_antennas, -1)  # [batch, ue_antennas, bs_antennas * subcarriers]
                    expected_antennas = self.num_antennas
                    logger.debug(f"   Reconstructed UE antenna structure: {csi_data.shape}")
                else:
                    logger.warning(f"⚠️ Cannot reconstruct UE antenna structure: {num_subcarriers} subcarriers not divisible by {self.num_antennas} UE antennas")
            
        else:
            raise ValueError(f"Unsupported CSI tensor dimensions: {csi.shape}. Expected 3D or 4D.")
        
        # Validate antenna count based on array type
        if expected_antennas != self.num_antennas:
            if self.orientation == 'bs':
                array_name = "BS"
            elif self.orientation == 'orientation':
                array_name = "orientation-based"
            else:
                array_name = "UE"
            logger.warning(f"⚠️ CSI antenna count {expected_antennas} does not match expected {self.num_antennas} for {array_name} array.")
            logger.warning(f"   This may indicate that UE antennas were converted to subcarriers during data loading.")
            logger.warning(f"   Continuing with available antenna count: {expected_antennas}")
        
        # Get angle grid dimensions
        theta_points, phi_points = len(self.theta_grid), len(self.phi_grid)
        
        if self.use_covariance_bartlett:
            return self._covariance_bartlett_spectrum(csi_data, theta_points, phi_points)
        else:
            return self._classical_bartlett_spectrum(csi_data, theta_points, phi_points)
    
    def _covariance_bartlett_spectrum(self, csi_data: torch.Tensor, theta_points: int, phi_points: int) -> torch.Tensor:
        """
        Compute spatial spectrum using covariance-based Bartlett method
        
        Args:
            csi_data: [batch_size, bs_antennas, subcarriers]
            
        Returns:
            spectrum: [batch_size, theta_points, phi_points]
        """
        batch_size, num_antennas, num_subcarriers = csi_data.shape
        device = csi_data.device
        
        batch_spectrums = []
        
        for b in range(batch_size):
            # Compute sample covariance matrix: R = (1/S) * sum_s x_s * x_s^H
            csi_batch = csi_data[b]  # [num_antennas, num_subcarriers]
            
            # Check for zero CSI
            if torch.abs(csi_batch).max() < 1e-8:
                logger.info(f"Ground truth CSI is zero for batch index {b} (likely no signal coverage). Using zero spectrum.")
                spectrum = torch.zeros(theta_points, phi_points, device=device)
                batch_spectrums.append(spectrum)
                continue
            
            if self.use_frequency_dependent_steering:
                # 优化的频率相关协方差方法：批量计算提高性能
                spectrum = torch.zeros(theta_points, phi_points, device=device)
                
                # 预计算所有角度的方向向量 [num_angles, 3]
                theta_mesh, phi_mesh = torch.meshgrid(self.theta_grid, self.phi_grid, indexing='ij')
                theta_flat = theta_mesh.flatten().to(device)
                phi_flat = phi_mesh.flatten().to(device)
                
                directions = torch.stack([
                    torch.sin(theta_flat) * torch.cos(phi_flat),
                    torch.sin(theta_flat) * torch.sin(phi_flat),
                    torch.cos(theta_flat)
                ], dim=1)  # [num_angles, 3]
                
                # 使用采样的子载波批量计算所有角度的功率
                subcarrier_spectrums = []
                antenna_positions = self.antenna_positions.to(device)
                sampled_indices = self.sampled_subcarrier_indices.to(device)
                
                for s_idx in sampled_indices:
                    s = s_idx.item()  # 转换为标量索引
                    csi_subcarrier = csi_batch[:, s]  # [num_antennas]
                    
                    # 跳过零子载波
                    if torch.abs(csi_subcarrier).max() < 1e-8:
                        continue
                    
                    # 计算该子载波的波数
                    freq_s = self.subcarrier_frequencies[s].to(device)
                    k_s = 2 * np.pi * freq_s / 3e8
                    
                    # 批量计算所有角度的相位偏移 [num_antennas, num_angles]
                    phase_shifts = k_s * torch.matmul(antenna_positions, directions.T)
                    
                    # 批量计算所有角度的导向矢量 [num_antennas, num_angles]
                    steering_vectors = torch.exp(1j * phase_shifts)
                    
                    # 批量计算所有角度的Bartlett功率 [num_angles]
                    beamformer_output = torch.matmul(torch.conj(steering_vectors).T, csi_subcarrier)  # [num_angles]
                    powers = torch.abs(beamformer_output) ** 2 / self.num_antennas
                    
                    # 重塑为频谱形状 [theta_points, phi_points]
                    spectrum_s = powers.reshape(theta_points, phi_points)
                    subcarrier_spectrums.append(spectrum_s)
                
                # 对所有子载波的频谱求平均
                if len(subcarrier_spectrums) > 0:
                    spectrum = torch.stack(subcarrier_spectrums).mean(dim=0)
                else:
                    logger.info(f"Ground truth CSI is zero for batch index {b} (likely no signal coverage). Using zero spectrum.")
                    spectrum = torch.zeros(theta_points, phi_points, device=device)
            else:
                # 传统协方差方法：假设窄带信号
                # Covariance matrix: R = E[x * x^H]
                R = torch.matmul(csi_batch, torch.conj(csi_batch).T) / num_subcarriers  # [num_antennas, num_antennas]
                
                # Compute spectrum for all angle pairs
                spectrum = torch.zeros(theta_points, phi_points, device=device)
                
                for t_idx in range(theta_points):
                    for p_idx in range(phi_points):
                        # Get steering vector for this angle (center frequency)
                        steering_vec = self.steering_vectors[t_idx, p_idx, :].to(device)  # [num_antennas]
                        
                        # Bartlett spectrum: a^H * R * a
                        spectrum_value = torch.real(torch.matmul(torch.conj(steering_vec), torch.matmul(R, steering_vec)))
                        spectrum[t_idx, p_idx] = spectrum_value / (self.num_antennas ** 2)  # Normalize
            
            batch_spectrums.append(spectrum)
        
        return torch.stack(batch_spectrums)
    
    def _classical_bartlett_spectrum(self, csi_data: torch.Tensor, theta_points: int, phi_points: int) -> torch.Tensor:
        """
        Compute spatial spectrum using classical Bartlett method with frequency-dependent steering
        
        Args:
            csi_data: [batch_size, bs_antennas, subcarriers]
            
        Returns:
            spectrum: [batch_size, theta_points, phi_points]
        """
        batch_size, num_antennas, num_subcarriers = csi_data.shape
        device = csi_data.device
        
        batch_spectrums = []
        
        for b in range(batch_size):
            subcarrier_spectrums = []
            sampled_indices = self.sampled_subcarrier_indices.to(device)
            
            for s_idx in sampled_indices:
                s = s_idx.item()  # 转换为标量索引
                csi_subcarrier = csi_data[b, :, s]  # [num_antennas]
                
                # Skip if all zeros
                if torch.abs(csi_subcarrier).max() < 1e-8:
                    continue
                
                # Get steering vectors for this subcarrier
                if self.use_frequency_dependent_steering:
                    # Compute frequency-dependent steering vectors
                    freq_s = self.subcarrier_frequencies[s].to(device)
                    k_s = 2 * np.pi * freq_s / 3e8  # c = 3e8 m/s
                    steering_vectors_s = self._compute_steering_vectors_for_frequency(k_s, device)
                else:
                    steering_vectors_s = self.steering_vectors.to(device)
                
                # Bartlett beamforming: |a^H * x|^2
                beamformer_output = torch.matmul(torch.conj(steering_vectors_s), csi_subcarrier)  # [theta_points, phi_points]
                spectrum_subcarrier = torch.abs(beamformer_output) ** 2  # [theta_points, phi_points]
                spectrum_subcarrier = spectrum_subcarrier / self.num_antennas  # Normalize
                
                subcarrier_spectrums.append(spectrum_subcarrier)
            
            # Average over all valid subcarriers
            if len(subcarrier_spectrums) > 0:
                avg_spectrum = torch.stack(subcarrier_spectrums).mean(dim=0)
            else:
                logger.info(f"Ground truth CSI is zero for batch index {b} (likely no signal coverage). Using zero spectrum.")
                avg_spectrum = torch.zeros(theta_points, phi_points, device=device)
            
            batch_spectrums.append(avg_spectrum)
        
        return torch.stack(batch_spectrums)
    
    def _compute_steering_vectors_for_frequency(self, k: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Compute steering vectors for a specific wavenumber k
        
        Args:
            k: Wavenumber (2π * frequency / c)
            device: Target device
            
        Returns:
            steering_vectors: [theta_points, phi_points, num_antennas]
        """
        # Create angle meshgrid
        theta_mesh, phi_mesh = torch.meshgrid(self.theta_grid, self.phi_grid, indexing='ij')
        theta_flat = theta_mesh.flatten()
        phi_flat = phi_mesh.flatten()
        
        # Compute direction vectors for all angles
        directions = torch.stack([
            torch.sin(theta_flat) * torch.cos(phi_flat),  # x
            torch.sin(theta_flat) * torch.sin(phi_flat),  # y
            torch.cos(theta_flat)                         # z
        ], dim=1).to(device)  # [num_angles, 3]
        
        # Get antenna positions on device
        antenna_positions = self.antenna_positions.to(device)
        
        # Compute phase shifts
        phase_shifts = k * torch.matmul(antenna_positions, directions.T)  # [num_antennas, num_angles]
        steering_vectors = torch.exp(1j * phase_shifts)
        
        # Reshape to [theta_points, phi_points, num_antennas]
        return steering_vectors.T.reshape(len(self.theta_grid), len(self.phi_grid), self.num_antennas)
    
    def _compute_ssim_loss(self, pred_spectrum: torch.Tensor, target_spectrum: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss between spatial spectrums"""
        # Add channel dimension for SSIM: [batch, 1, height, width]
        pred_4d = pred_spectrum.unsqueeze(1)
        target_4d = target_spectrum.unsqueeze(1)
        
        # Compute SSIM
        ssim_value = self._ssim(pred_4d, target_4d)
        
        # Convert to loss (1 - SSIM)
        return 1.0 - ssim_value
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between two images"""
        device = img1.device
        channel = img1.size(1)
        
        # Create Gaussian window
        window_size = self.ssim_window_size
        sigma = window_size / 6.0
        
        # 1D Gaussian kernel
        coords = torch.arange(window_size, dtype=torch.float32, device=device)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # 2D Gaussian kernel
        window = g[:, None] * g[None, :]
        window = window.expand(channel, 1, window_size, window_size)
        
        # SSIM constants
        C1 = (self.ssim_k1) ** 2
        C2 = (self.ssim_k2) ** 2
        
        # Compute local means
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        # Compute SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator
        
        return ssim_map.mean()
    
    def forward(self, predicted_csi: torch.Tensor, target_csi: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial spectrum SSIM loss
        
        Args:
            predicted_csi: Predicted CSI [batch_size, bs_antennas, subcarriers]
            target_csi: Target CSI [batch_size, bs_antennas, subcarriers]
            
        Returns:
            loss: SSIM loss scalar
        """
        # Convert CSI to spatial spectrum
        pred_spectrum = self._csi_to_spatial_spectrum(predicted_csi)
        target_spectrum = self._csi_to_spatial_spectrum(target_csi)
        
        # Normalize spectrums to [0, 1] using consistent normalization
        # Use the maximum value across both predicted and target spectrums for each sample
        max_values = torch.maximum(
            pred_spectrum.amax(dim=(1, 2), keepdim=True),
            target_spectrum.amax(dim=(1, 2), keepdim=True)
        )
        pred_norm = pred_spectrum / (max_values + 1e-8)
        target_norm = target_spectrum / (max_values + 1e-8)
        
        # Compute SSIM loss
        return self._compute_ssim_loss(pred_norm, target_norm)
    
    def compute_spatial_spectrum(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Public interface to compute spatial spectrum from CSI
        
        Args:
            csi: CSI tensor [batch_size, bs_antennas, subcarriers]
            
        Returns:
            spectrum: Spatial spectrum [batch_size, theta_points, phi_points]
        """
        return self._csi_to_spatial_spectrum(csi)
    
    def validate_with_plane_wave(self, theta_deg: float, phi_deg: float, 
                                 amplitude: float = 1.0, noise_level: float = 0.01) -> Dict[str, float]:
        """
        Validate spatial spectrum implementation with a synthetic plane wave
        
        Args:
            theta_deg: Elevation angle in degrees [0, 90]
            phi_deg: Azimuth angle in degrees [0, 360]
            amplitude: Plane wave amplitude
            noise_level: Additive noise standard deviation
            
        Returns:
            validation_metrics: Dictionary with peak location and beam quality metrics
        """
        device = self.steering_vectors.device
        
        # Convert angles to radians
        theta_rad = torch.deg2rad(torch.tensor(theta_deg, device=device))
        phi_rad = torch.deg2rad(torch.tensor(phi_deg, device=device))
        
        # Generate direction vector for the plane wave
        direction = torch.tensor([
            torch.sin(theta_rad) * torch.cos(phi_rad),
            torch.sin(theta_rad) * torch.sin(phi_rad),
            torch.cos(theta_rad)
        ], device=device)
        
        # Compute phase shifts for all antennas
        k = 2 * np.pi / self.wavelength
        phase_shifts = k * torch.matmul(self.antenna_positions, direction)
        
        # Generate synthetic CSI as a plane wave (matching steering vector convention)
        synthetic_csi = amplitude * torch.exp(1j * phase_shifts)  # [num_antennas]
        
        # Add noise
        if noise_level > 0:
            noise_real = torch.randn_like(synthetic_csi.real) * noise_level
            noise_imag = torch.randn_like(synthetic_csi.imag) * noise_level
            synthetic_csi += torch.complex(noise_real, noise_imag)
        
        # Create batch format: [1, bs_antennas, num_subcarriers]
        # 复制CSI到足够的子载波数量以支持采样
        csi_subcarriers = synthetic_csi.unsqueeze(1).repeat(1, self.num_subcarriers)  # [num_antennas, num_subcarriers]
        csi_batch = csi_subcarriers.unsqueeze(0)  # [1, num_antennas, num_subcarriers]
        
        # Compute spatial spectrum
        spectrum = self._csi_to_spatial_spectrum(csi_batch)[0]  # [theta_points, phi_points]
        
        # Find peak location
        peak_idx = torch.argmax(spectrum.flatten())
        peak_theta_idx = peak_idx // spectrum.shape[1]
        peak_phi_idx = peak_idx % spectrum.shape[1]
        
        # Convert back to degrees
        peak_theta_deg = torch.rad2deg(self.theta_grid[peak_theta_idx]).item()
        peak_phi_deg = torch.rad2deg(self.phi_grid[peak_phi_idx]).item()
        
        # Compute validation metrics
        theta_error = abs(peak_theta_deg - theta_deg)
        phi_error = abs(peak_phi_deg - phi_deg)
        if phi_error > 180:
            phi_error = 360 - phi_error  # Handle wrap-around
        
        # Beam quality metrics
        peak_value = spectrum.max().item()
        mean_value = spectrum.mean().item()
        peak_to_average_ratio = peak_value / (mean_value + 1e-8)
        
        validation_metrics = {
            'input_theta_deg': theta_deg,
            'input_phi_deg': phi_deg,
            'peak_theta_deg': peak_theta_deg,
            'peak_phi_deg': peak_phi_deg,
            'theta_error_deg': theta_error,
            'phi_error_deg': phi_error,
            'peak_value': peak_value,
            'peak_to_average_ratio': peak_to_average_ratio,
            'validation_passed': theta_error < 2.0 and phi_error < 4.0  # Allow small tolerance
        }
        
        return validation_metrics
    
    def validate_antenna_mapping(self, csi_sample: torch.Tensor, expected_theta_deg: float, 
                                expected_phi_deg: float, tolerance_deg: float = 5.0) -> Dict[str, float]:
        """
        Validate antenna index mapping by checking if a known signal direction is correctly detected
        
        Args:
            csi_sample: CSI sample [bs_antennas, subcarriers] for a known signal direction
            expected_theta_deg: Expected elevation angle in degrees
            expected_phi_deg: Expected azimuth angle in degrees  
            tolerance_deg: Tolerance for angle error in degrees
            
        Returns:
            validation_result: Dictionary with detected angles and validation status
            
        Raises:
            ValueError: If antenna mapping validation fails
        """
        # Add batch dimension if needed
        if csi_sample.dim() == 2:
            csi_batch = csi_sample.unsqueeze(0)  # [1, bs_antennas, subcarriers]
        else:
            csi_batch = csi_sample
            
        # Validate antenna count based on array type
        if csi_batch.shape[1] != self.num_antennas:
            if self.orientation == 'bs':
                array_name = "BS"
            elif self.orientation == 'orientation':
                array_name = "orientation-based"
            else:
                array_name = "UE"
            raise ValueError(f"CSI antenna count {csi_batch.shape[1]} does not match expected {self.num_antennas} for {array_name} array")
        
        # Compute spatial spectrum
        spectrum = self._csi_to_spatial_spectrum(csi_batch)[0]  # [theta_points, phi_points]
        
        # Find peak location
        peak_idx = torch.argmax(spectrum.flatten())
        peak_theta_idx = peak_idx // spectrum.shape[1]
        peak_phi_idx = peak_idx % spectrum.shape[1]
        
        # Convert back to degrees
        detected_theta_deg = torch.rad2deg(self.theta_grid[peak_theta_idx]).item()
        detected_phi_deg = torch.rad2deg(self.phi_grid[peak_phi_idx]).item()
        
        # Compute errors
        theta_error = abs(detected_theta_deg - expected_theta_deg)
        phi_error = abs(detected_phi_deg - expected_phi_deg)
        if phi_error > 180:
            phi_error = 360 - phi_error  # Handle wrap-around
        
        # Validation result
        validation_passed = theta_error <= tolerance_deg and phi_error <= tolerance_deg
        
        result = {
            'expected_theta_deg': expected_theta_deg,
            'expected_phi_deg': expected_phi_deg,
            'detected_theta_deg': detected_theta_deg,
            'detected_phi_deg': detected_phi_deg,
            'theta_error_deg': theta_error,
            'phi_error_deg': phi_error,
            'peak_value': spectrum.max().item(),
            'validation_passed': validation_passed
        }
        
        if not validation_passed:
            error_msg = (f"Antenna mapping validation failed! "
                        f"Expected: (θ={expected_theta_deg}°, φ={expected_phi_deg}°), "
                        f"Detected: (θ={detected_theta_deg:.1f}°, φ={detected_phi_deg:.1f}°), "
                        f"Errors: (θ={theta_error:.1f}°, φ={phi_error:.1f}°). "
                        f"This indicates incorrect antenna index → (m,n) position mapping. "
                        f"Check your CSI data antenna ordering.")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"✅ Antenna mapping validation passed: "
                   f"θ_error={theta_error:.1f}°, φ_error={phi_error:.1f}°")
        
        return result
    
    def debug_antenna_mapping(self, csi_sample: torch.Tensor, save_debug_plot: bool = True) -> Dict[str, any]:
        """
        Debug antenna mapping using phase pattern analysis
        
        Args:
            csi_sample: CSI sample [bs_antennas, subcarriers] 
            save_debug_plot: Whether to save debug visualization to .temp/
            
        Returns:
            debug_result: Dictionary with analysis results and suggestions
        """
        try:
            # Import debug tool (only when needed)
            import sys
            import os
            debug_tool_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                          '..', '..', '.temp')
            sys.path.insert(0, debug_tool_path)
            from antenna_mapping_debug import visualize_csi_phase_pattern, suggest_antenna_permutation
            
            # Prepare save path
            save_path = None
            if save_debug_plot:
                import os
                temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                       '..', '..', '.temp')
                save_path = os.path.join(temp_dir, 'antenna_mapping_debug.png')
                
            # Run analysis
            analysis_result = visualize_csi_phase_pattern(
                csi_sample, (self.M, self.N), save_path=save_path
            )
            
            # Get suggestions
            suggestions = suggest_antenna_permutation(csi_sample, (self.M, self.N))
            
            debug_result = {
                'phase_linearity_analysis': analysis_result,
                'mapping_suggestions': suggestions,
                'array_shape': (self.M, self.N),
                'debug_plot_path': save_path if save_debug_plot else None
            }
            
            logger.info(f"Antenna mapping debug completed. Quality: {analysis_result['mapping_quality']}")
            if analysis_result['mapping_quality'] == 'poor':
                logger.warning("⚠️ Poor antenna mapping detected! Check debug plot and suggestions.")
                logger.warning(suggestions)
            
            return debug_result
            
        except ImportError as e:
            logger.error(f"Could not import debug tool: {e}")
            raise ValueError(f"Debug tool not available: {e}")
        except Exception as e:
            logger.error(f"Antenna mapping debug failed: {e}")
            raise ValueError(f"Debug analysis failed: {e}")
