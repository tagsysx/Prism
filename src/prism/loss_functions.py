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
from typing import Dict, Optional, Tuple, Union
import numpy as np


# Default configuration for loss functions
DEFAULT_LOSS_CONFIG = {
    'csi_weight': 0.7,
    'pdp_weight': 0.3,
    'regularization_weight': 0.01,
    'csi_loss': {
        'type': 'hybrid',  # 'mse', 'mae', 'complex_mse', 'magnitude_phase', 'correlation', 'hybrid'
        'phase_weight': 1.0,
        'magnitude_weight': 1.0,
        'cmse_weight': 1.0,
        'correlation_weight': 1.0
    },
    'pdp_loss': {
        'type': 'hybrid',  # 'mse', 'correlation', 'delay', 'hybrid'
        'fft_size': 1024,
        'normalize_pdp': True,
        'mse_weight': 0.5,
        'correlation_weight': 0.3,
        'delay_weight': 0.2
    }
}


class CSILoss(nn.Module):
    """
    CSI (Channel State Information) Loss Functions
    
    Provides various loss functions for comparing complex-valued CSI matrices,
    including magnitude, phase, correlation, and hybrid losses.
    """
    
    def __init__(self, loss_type: str = 'mse', phase_weight: float = 1.0, 
                 magnitude_weight: float = 1.0, cmse_weight: float = 1.0,
                 correlation_weight: float = 1.0):
        """
        Initialize CSI loss function
        
        Args:
            loss_type: Type of loss ('mse', 'mae', 'complex_mse', 'magnitude_phase', 
                      'correlation', 'hybrid')
            phase_weight: Weight for phase component in combined losses
            magnitude_weight: Weight for magnitude component in combined losses
            cmse_weight: Weight for CMSE component in hybrid loss
            correlation_weight: Weight for correlation component in hybrid loss
        """
        super(CSILoss, self).__init__()
        self.loss_type = loss_type
        self.phase_weight = phase_weight
        self.magnitude_weight = magnitude_weight
        self.cmse_weight = cmse_weight
        self.correlation_weight = correlation_weight
        
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
            
        elif self.loss_type == 'correlation':
            # Complex correlation loss (1 - |correlation|)
            correlation = self._complex_correlation(predicted_csi, target_csi)
            loss = 1.0 - torch.abs(correlation)
            
        elif self.loss_type == 'hybrid':
            # Hybrid CSI Loss: CMSE + Correlation
            # 1. Complex MSE Loss
            diff = predicted_csi - target_csi
            cmse_loss = torch.mean(torch.abs(diff)**2)
            
            # 2. Correlation Loss
            correlation = self._complex_correlation(predicted_csi, target_csi)
            corr_loss = 1.0 - torch.abs(correlation)
            
            # Combine losses
            loss = self.cmse_weight * cmse_loss + self.correlation_weight * corr_loss
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def _complex_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute complex correlation coefficient between two complex tensors
        
        Args:
            x: First complex tensor
            y: Second complex tensor
            
        Returns:
            correlation: Complex correlation coefficient
        """
        # Flatten tensors for correlation computation
        x_flat = x.view(-1)
        y_flat = y.view(-1)
        
        # Compute means
        x_mean = torch.mean(x_flat)
        y_mean = torch.mean(y_flat)
        
        # Center the data
        x_centered = x_flat - x_mean
        y_centered = y_flat - y_mean
        
        # Compute correlation
        numerator = torch.mean(x_centered * torch.conj(y_centered))
        denominator = torch.sqrt(torch.mean(torch.abs(x_centered)**2) * 
                                torch.mean(torch.abs(y_centered)**2))
        
        correlation = numerator / (denominator + 1e-8)
        return correlation


class PDPLoss(nn.Module):
    """
    Power Delay Profile (PDP) Loss Functions
    
    Provides time-domain validation by comparing PDPs derived from CSI data.
    Supports MSE, correlation, delay, and hybrid PDP losses.
    """
    
    def __init__(self, loss_type: str = 'hybrid', fft_size: int = 1024,
                 normalize_pdp: bool = True, mse_weight: float = 0.5,
                 correlation_weight: float = 0.3, delay_weight: float = 0.2):
        """
        Initialize PDP loss function
        
        Args:
            loss_type: Type of PDP loss ('mse', 'correlation', 'delay', 'hybrid')
            fft_size: FFT size for PDP computation
            normalize_pdp: Whether to normalize PDPs before comparison
            mse_weight: Weight for MSE component in hybrid loss
            correlation_weight: Weight for correlation component in hybrid loss
            delay_weight: Weight for delay component in hybrid loss
        """
        super(PDPLoss, self).__init__()
        self.loss_type = loss_type
        self.fft_size = fft_size
        self.normalize_pdp = normalize_pdp
        self.mse_weight = mse_weight
        self.correlation_weight = correlation_weight
        self.delay_weight = delay_weight
        
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
            
        elif self.loss_type == 'correlation':
            # PDP Correlation Loss
            loss = self._compute_pdp_correlation_loss(pdp_pred, pdp_target)
            
        elif self.loss_type == 'delay':
            # Dominant Path Delay Loss
            loss = self._compute_delay_loss(pdp_pred, pdp_target)
            
        elif self.loss_type == 'hybrid':
            # Hybrid PDP Loss: MSE + Correlation + Delay
            mse_loss = F.mse_loss(pdp_pred, pdp_target)
            corr_loss = self._compute_pdp_correlation_loss(pdp_pred, pdp_target)
            delay_loss = self._compute_delay_loss(pdp_pred, pdp_target)
            
            loss = (self.mse_weight * mse_loss + 
                   self.correlation_weight * corr_loss + 
                   self.delay_weight * delay_loss)
            
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
    
    def _compute_pdp_correlation_loss(self, pdp_pred: torch.Tensor, 
                                     pdp_target: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation loss for PDPs
        """
        # Compute correlation coefficient
        pred_mean = torch.mean(pdp_pred)
        target_mean = torch.mean(pdp_target)
        
        pred_centered = pdp_pred - pred_mean
        target_centered = pdp_target - target_mean
        
        numerator = torch.sum(pred_centered * target_centered)
        denominator = torch.sqrt(torch.sum(pred_centered ** 2) * 
                                torch.sum(target_centered ** 2))
        
        if denominator < 1e-8:
            return torch.tensor(1.0, device=pdp_pred.device)
        
        correlation = numerator / denominator
        
        # Return 1 - |correlation|
        return 1.0 - torch.abs(correlation)
    
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
    
    Provides loss functions for comparing spatial spectrum matrices,
    including peak-aware losses and angular distribution losses.
    
    Note: Currently not used in the main training pipeline, but reserved
    for future spatial spectrum-based training scenarios.
    """
    
    def __init__(self, loss_type: str = 'mse', peak_weight: float = 2.0,
                 angular_smoothness_weight: float = 0.1):
        """
        Initialize spatial spectrum loss function
        
        Args:
            loss_type: Type of loss ('mse', 'mae', 'peak_aware', 'kl_divergence')
            peak_weight: Weight for peak regions in peak-aware loss
            angular_smoothness_weight: Weight for angular smoothness regularization
        """
        super(SpatialSpectrumLoss, self).__init__()
        self.loss_type = loss_type
        self.peak_weight = peak_weight
        self.angular_smoothness_weight = angular_smoothness_weight
        
    def forward(self, predicted_spectrum: torch.Tensor, target_spectrum: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute spatial spectrum loss
        
        Args:
            predicted_spectrum: Predicted spatial spectrum
                              Shape: (batch_size, theta_points, phi_points)
            target_spectrum: Target spatial spectrum
                           Shape: (batch_size, theta_points, phi_points)
            mask: Optional mask for selective loss computation
                  Shape: same as spectrum tensors
        
        Returns:
            loss: Computed loss value (scalar tensor)
        """
        if predicted_spectrum.shape != target_spectrum.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_spectrum.shape} vs target {target_spectrum.shape}")
        
        # Apply mask if provided
        if mask is not None:
            predicted_spectrum = predicted_spectrum * mask
            target_spectrum = target_spectrum * mask
        
        if self.loss_type == 'mse':
            # Standard MSE loss
            loss = F.mse_loss(predicted_spectrum, target_spectrum)
            
        elif self.loss_type == 'mae':
            # Mean Absolute Error
            loss = F.l1_loss(predicted_spectrum, target_spectrum)
            
        elif self.loss_type == 'peak_aware':
            # Peak-aware loss: higher weight for peak regions
            peak_mask = self._detect_peaks(target_spectrum)
            
            # Standard loss
            base_loss = F.mse_loss(predicted_spectrum, target_spectrum)
            
            # Peak region loss
            peak_loss = F.mse_loss(predicted_spectrum * peak_mask, 
                                  target_spectrum * peak_mask)
            
            loss = base_loss + self.peak_weight * peak_loss
            
        elif self.loss_type == 'kl_divergence':
            # KL divergence for probability distributions
            # Normalize to probability distributions
            pred_prob = self._normalize_to_prob(predicted_spectrum)
            target_prob = self._normalize_to_prob(target_spectrum)
            
            loss = F.kl_div(torch.log(pred_prob + 1e-8), target_prob, reduction='batchmean')
            
        elif self.loss_type == 'angular_weighted':
            # Angular-weighted loss considering spatial relationships
            angular_weights = self._compute_angular_weights(target_spectrum.shape[-2:])
            weighted_diff = (predicted_spectrum - target_spectrum)**2 * angular_weights
            loss = torch.mean(weighted_diff)
            
        elif self.loss_type == 'combined':
            # Combined loss with multiple components
            mse_loss = F.mse_loss(predicted_spectrum, target_spectrum)
            peak_loss = self._peak_aware_loss(predicted_spectrum, target_spectrum)
            smoothness_loss = self._angular_smoothness_loss(predicted_spectrum)
            
            loss = mse_loss + self.peak_weight * peak_loss + \
                   self.angular_smoothness_weight * smoothness_loss
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def _detect_peaks(self, spectrum: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
        """
        Detect peak regions in spatial spectrum
        
        Args:
            spectrum: Spatial spectrum tensor
            threshold: Threshold for peak detection (relative to max)
            
        Returns:
            peak_mask: Binary mask indicating peak regions
        """
        batch_size = spectrum.shape[0]
        peak_masks = []
        
        for b in range(batch_size):
            spec = spectrum[b]
            max_val = torch.max(spec)
            peak_mask = (spec > threshold * max_val).float()
            peak_masks.append(peak_mask)
        
        return torch.stack(peak_masks)
    
    def _normalize_to_prob(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Normalize spectrum to probability distribution
        
        Args:
            spectrum: Input spectrum tensor
            
        Returns:
            prob: Normalized probability distribution
        """
        # Ensure non-negative values
        spectrum_pos = torch.clamp(spectrum, min=1e-8)
        
        # Normalize to sum to 1
        prob = spectrum_pos / torch.sum(spectrum_pos, dim=(-2, -1), keepdim=True)
        
        return prob
    
    def _compute_angular_weights(self, shape: Tuple[int, int]) -> torch.Tensor:
        """
        Compute angular weights for spatial spectrum
        
        Args:
            shape: (theta_points, phi_points)
            
        Returns:
            weights: Angular weighting matrix
        """
        theta_points, phi_points = shape
        
        # Create angular grids
        theta_grid = torch.linspace(-np.pi/2, np.pi/2, theta_points)
        phi_grid = torch.linspace(0, 2*np.pi, phi_points)
        
        # Elevation weighting (higher weight near horizon)
        theta_weights = torch.cos(theta_grid).unsqueeze(1)  # (theta_points, 1)
        
        # Uniform azimuth weighting
        phi_weights = torch.ones(1, phi_points)  # (1, phi_points)
        
        # Combined weights
        weights = theta_weights * phi_weights  # (theta_points, phi_points)
        
        return weights
    
    def _peak_aware_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute peak-aware loss component
        """
        peak_mask = self._detect_peaks(target)
        peak_loss = F.mse_loss(predicted * peak_mask, target * peak_mask)
        return peak_loss
    
    def _angular_smoothness_loss(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Compute angular smoothness regularization loss
        """
        # Gradient in theta direction
        theta_grad = torch.diff(spectrum, dim=-2)
        theta_smoothness = torch.mean(theta_grad**2)
        
        # Gradient in phi direction
        phi_grad = torch.diff(spectrum, dim=-1)
        phi_smoothness = torch.mean(phi_grad**2)
        
        return theta_smoothness + phi_smoothness


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
        self.regularization_weight = config.get('regularization_weight', 0.01)
        
        # Initialize component losses with reasonable defaults
        csi_config = config.get('csi_loss', {})
        self.csi_loss = CSILoss(
            loss_type=csi_config.get('type', 'hybrid'),
            phase_weight=csi_config.get('phase_weight', 1.0),
            magnitude_weight=csi_config.get('magnitude_weight', 1.0),
            cmse_weight=csi_config.get('cmse_weight', 1.0),
            correlation_weight=csi_config.get('correlation_weight', 1.0)
        )
        
        # Initialize PDP loss
        pdp_config = config.get('pdp_loss', {})
        self.pdp_loss = PDPLoss(
            loss_type=pdp_config.get('type', 'hybrid'),
            fft_size=pdp_config.get('fft_size', 1024),
            normalize_pdp=pdp_config.get('normalize_pdp', True),
            mse_weight=pdp_config.get('mse_weight', 0.5),
            correlation_weight=pdp_config.get('correlation_weight', 0.3),
            delay_weight=pdp_config.get('delay_weight', 0.2)
        )
        

        
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
        
        # CSI loss (hybrid: CMSE + Correlation)
        if 'csi' in predictions and 'csi' in targets:
            csi_loss_val = self.csi_loss(
                predictions['csi'], 
                targets['csi'], 
                masks.get('csi')
            )
            total_loss = total_loss + self.csi_weight * csi_loss_val
            loss_components['csi_loss'] = csi_loss_val.item()
        
        # PDP loss (hybrid: MSE + Correlation + Delay)
        if 'csi' in predictions and 'csi' in targets and self.pdp_weight > 0:
            pdp_loss_val = self.pdp_loss(
                predictions['csi'], 
                targets['csi']
            )
            total_loss = total_loss + self.pdp_weight * pdp_loss_val
            loss_components['pdp_loss'] = pdp_loss_val.item()
        

        
        # Regularization losses
        if 'regularization' in predictions:
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






