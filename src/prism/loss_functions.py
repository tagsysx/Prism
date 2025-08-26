"""
Loss Functions for Prism: Neural Network-Based Electromagnetic Ray Tracing

This module provides specialized loss functions for CSI (Channel State Information) 
and spatial spectrum estimation tasks, all supporting automatic differentiation.

Classes:
- PrismLossFunction: Main loss function class with CSI and spatial spectrum losses
- CSILoss: Specialized CSI loss functions
- SpatialSpectrumLoss: Specialized spatial spectrum loss functions

All loss functions are designed to work with PyTorch tensors and support backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import numpy as np


class CSILoss(nn.Module):
    """
    CSI (Channel State Information) Loss Functions
    
    Provides various loss functions for comparing complex-valued CSI matrices,
    including magnitude, phase, and combined losses.
    """
    
    def __init__(self, loss_type: str = 'mse', phase_weight: float = 1.0, 
                 magnitude_weight: float = 1.0):
        """
        Initialize CSI loss function
        
        Args:
            loss_type: Type of loss ('mse', 'mae', 'complex_mse', 'magnitude_phase')
            phase_weight: Weight for phase component in combined losses
            magnitude_weight: Weight for magnitude component in combined losses
        """
        super(CSILoss, self).__init__()
        self.loss_type = loss_type
        self.phase_weight = phase_weight
        self.magnitude_weight = magnitude_weight
        
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


class SpatialSpectrumLoss(nn.Module):
    """
    Spatial Spectrum Loss Functions
    
    Provides loss functions for comparing spatial spectrum matrices,
    including peak-aware losses and angular distribution losses.
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
    
    Combines CSI and spatial spectrum losses with configurable weights
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
        self.csi_weight = config.get('csi_weight', 1.0)
        self.spectrum_weight = config.get('spectrum_weight', 1.0)
        self.regularization_weight = config.get('regularization_weight', 0.01)
        
        # Initialize component losses with reasonable defaults
        csi_config = config.get('csi_loss', {})
        self.csi_loss = CSILoss(
            loss_type=csi_config.get('type', 'mse'),
            phase_weight=csi_config.get('phase_weight', 1.0),
            magnitude_weight=csi_config.get('magnitude_weight', 1.0)
        )
        
        spectrum_config = config.get('spectrum_loss', {})
        self.spectrum_loss = SpatialSpectrumLoss(
            loss_type=spectrum_config.get('type', 'mse'),
            peak_weight=spectrum_config.get('peak_weight', 2.0),
            angular_smoothness_weight=spectrum_config.get('angular_smoothness_weight', 0.1)
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
                        - 'spectrum': Predicted spatial spectrum (optional)
            targets: Dictionary containing target values
                    - 'csi': Target CSI tensor
                    - 'spectrum': Target spatial spectrum (optional)
            masks: Optional masks for selective loss computation
        
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary of individual loss components
        """
        total_loss = 0.0
        loss_components = {}
        
        if masks is None:
            masks = {}
        
        # CSI loss
        if 'csi' in predictions and 'csi' in targets:
            csi_loss_val = self.csi_loss(
                predictions['csi'], 
                targets['csi'], 
                masks.get('csi')
            )
            total_loss += self.csi_weight * csi_loss_val
            loss_components['csi_loss'] = csi_loss_val.item()
        
        # Spatial spectrum loss
        if 'spectrum' in predictions and 'spectrum' in targets:
            spectrum_loss_val = self.spectrum_loss(
                predictions['spectrum'], 
                targets['spectrum'], 
                masks.get('spectrum')
            )
            total_loss += self.spectrum_weight * spectrum_loss_val
            loss_components['spectrum_loss'] = spectrum_loss_val.item()
        
        # Regularization losses
        if 'regularization' in predictions:
            reg_loss = predictions['regularization']
            total_loss += self.regularization_weight * reg_loss
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


# Utility functions for loss computation
def compute_csi_metrics(predicted_csi: torch.Tensor, target_csi: torch.Tensor) -> Dict[str, float]:
    """
    Compute various CSI evaluation metrics
    
    Args:
        predicted_csi: Predicted CSI tensor
        target_csi: Target CSI tensor
        
    Returns:
        metrics: Dictionary of computed metrics
    """
    with torch.no_grad():
        # MSE - compute manually for complex numbers
        diff = predicted_csi - target_csi
        mse = torch.mean(torch.abs(diff)**2).item()
        
        # MAE
        mae = torch.mean(torch.abs(predicted_csi - target_csi)).item()
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # Complex correlation
        csi_loss = CSILoss(loss_type='correlation')
        correlation = 1.0 - csi_loss._complex_correlation(predicted_csi, target_csi).abs().item()
        
        # Magnitude error
        pred_mag = torch.abs(predicted_csi)
        target_mag = torch.abs(target_csi)
        magnitude_error = F.mse_loss(pred_mag, target_mag).item()
        
        # Phase error
        pred_phase = torch.angle(predicted_csi)
        target_phase = torch.angle(target_csi)
        phase_diff = torch.remainder(pred_phase - target_phase + np.pi, 2*np.pi) - np.pi
        phase_error = torch.mean(phase_diff**2).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'correlation_loss': correlation,
        'magnitude_error': magnitude_error,
        'phase_error': phase_error
    }


def compute_spectrum_metrics(predicted_spectrum: torch.Tensor, 
                           target_spectrum: torch.Tensor) -> Dict[str, float]:
    """
    Compute spatial spectrum evaluation metrics
    
    Args:
        predicted_spectrum: Predicted spatial spectrum
        target_spectrum: Target spatial spectrum
        
    Returns:
        metrics: Dictionary of computed metrics
    """
    with torch.no_grad():
        # MSE
        mse = F.mse_loss(predicted_spectrum, target_spectrum).item()
        
        # MAE
        mae = F.l1_loss(predicted_spectrum, target_spectrum).item()
        
        # Peak detection accuracy
        spectrum_loss = SpatialSpectrumLoss()
        pred_peaks = spectrum_loss._detect_peaks(predicted_spectrum)
        target_peaks = spectrum_loss._detect_peaks(target_spectrum)
        peak_accuracy = (pred_peaks == target_peaks).float().mean().item()
        
        # Energy preservation
        pred_energy = torch.sum(predicted_spectrum, dim=(-2, -1))
        target_energy = torch.sum(target_spectrum, dim=(-2, -1))
        energy_error = F.mse_loss(pred_energy, target_energy).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'peak_accuracy': peak_accuracy,
        'energy_error': energy_error
    }


# Example configuration for loss functions
DEFAULT_LOSS_CONFIG = {
    'csi_weight': 1.0,
    'spectrum_weight': 1.0,
    'regularization_weight': 0.01,
    'csi_loss': {
        'type': 'magnitude_phase',  # 'mse', 'mae', 'complex_mse', 'magnitude_phase', 'correlation'
        'phase_weight': 1.0,
        'magnitude_weight': 1.0
    },
    'spectrum_loss': {
        'type': 'combined',  # 'mse', 'mae', 'peak_aware', 'kl_divergence', 'angular_weighted', 'combined'
        'peak_weight': 2.0,
        'angular_smoothness_weight': 0.1
    }
}
