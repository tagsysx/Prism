"""
CSI (Channel State Information) Loss Function

Provides subcarrier-precise loss for comparing complex-valued CSI matrices,
addressing the complex MSE vs per-subcarrier accuracy paradox.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np
import logging

# Get logger for this module
logger = logging.getLogger(__name__)


class CSILoss(nn.Module):
    """
    CSI (Channel State Information) Loss Function
    
    Provides subcarrier-precise loss for comparing complex-valued CSI matrices,
    addressing the complex MSE vs per-subcarrier accuracy paradox.
    """
    
    def __init__(self, phase_weight: float = 1.0, magnitude_weight: float = 1.0,
                 normalize_weights: bool = True):
        """
        Initialize CSI loss function
        
        Args:
            phase_weight: Weight for phase component in the loss
            magnitude_weight: Weight for magnitude component in the loss
            normalize_weights: Whether to normalize weights to sum to 1.0 for balanced scaling
        """
        super(CSILoss, self).__init__()
        self.normalize_weights = normalize_weights
        
        # Store original weights for reference
        self.original_phase_weight = phase_weight
        self.original_magnitude_weight = magnitude_weight
        
        # Apply weight normalization if requested
        if normalize_weights:
            total_weight = magnitude_weight + phase_weight
            if total_weight > 0:
                self.magnitude_weight = magnitude_weight / total_weight
                self.phase_weight = phase_weight / total_weight
            else:
                self.magnitude_weight = 0.5
                self.phase_weight = 0.5
        else:
            # Use original weights without normalization
            self.phase_weight = phase_weight
            self.magnitude_weight = magnitude_weight
        
    def forward(self, predicted_csi: torch.Tensor, target_csi: torch.Tensor) -> torch.Tensor:
        """
        Compute subcarrier-precise CSI loss between predicted and target CSI
        
        Args:
            predicted_csi: Predicted CSI tensor (complex) from selected subcarriers
                          Shape: (N,) - selected subcarriers, or any shape with selected data
            target_csi: Target CSI tensor (complex) from selected subcarriers
                       Shape: same as predicted_csi
        
        Returns:
            loss: Computed loss value (scalar tensor)
        """
        if predicted_csi.shape != target_csi.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_csi.shape} vs target {target_csi.shape}")
        
        # Ensure complex tensors with proper conversion
        if not predicted_csi.is_complex():
            if predicted_csi.dtype == torch.float32:
                # Convert float32 to complex64 (real part only, imaginary = 0)
                predicted_csi = predicted_csi + 1j * torch.zeros_like(predicted_csi)
            elif predicted_csi.dtype == torch.float64:
                # Convert float64 to complex128 (real part only, imaginary = 0)
                predicted_csi = predicted_csi + 1j * torch.zeros_like(predicted_csi)
            else:
                # Fallback: direct conversion
                predicted_csi = predicted_csi.to(torch.complex64)
                
        if not target_csi.is_complex():
            if target_csi.dtype == torch.float32:
                # Convert float32 to complex64 (real part only, imaginary = 0)
                target_csi = target_csi + 1j * torch.zeros_like(target_csi)
            elif target_csi.dtype == torch.float64:
                # Convert float64 to complex128 (real part only, imaginary = 0)
                target_csi = target_csi + 1j * torch.zeros_like(target_csi)
            else:
                # Fallback: direct conversion
                target_csi = target_csi.to(torch.complex64)
        
        # Subcarrier-precise loss: addresses the complex MSE vs per-subcarrier accuracy paradox
        pred_mag = torch.abs(predicted_csi)
        target_mag = torch.abs(target_csi)
        
        # 1. Per-subcarrier magnitude MSE (not averaged across subcarriers)
        subcarrier_mag_mse = torch.mean((pred_mag - target_mag)**2, dim=1)  # [num_subcarriers]
        subcarrier_mag_loss = torch.mean(subcarrier_mag_mse)  # Average across subcarriers
        
        # 2. Magnitude distribution preservation
        pred_mag_mean = torch.mean(pred_mag, dim=1)  # Mean per subcarrier
        target_mag_mean = torch.mean(target_mag, dim=1)
        pred_mag_std = torch.std(pred_mag, dim=1)    # Std per subcarrier  
        target_mag_std = torch.std(target_mag, dim=1)
        
        mean_distribution_loss = F.mse_loss(pred_mag_mean, target_mag_mean)
        std_distribution_loss = F.mse_loss(pred_mag_std, target_mag_std)
        
        # 3. Phase consistency per subcarrier
        pred_phase = torch.atan2(predicted_csi.imag, predicted_csi.real + 1e-8)
        target_phase = torch.atan2(target_csi.imag, target_csi.real + 1e-8)
        
        phase_diff = torch.remainder(pred_phase - target_phase + np.pi, 2*np.pi) - np.pi
        subcarrier_phase_mse = torch.mean(phase_diff**2, dim=1)  # [num_subcarriers]
        subcarrier_phase_loss = torch.mean(subcarrier_phase_mse)
        
        # 4. Complex correlation preservation (maintains complex relationship)
        pred_complex_norm = predicted_csi / (torch.abs(predicted_csi) + 1e-8)
        target_complex_norm = target_csi / (torch.abs(target_csi) + 1e-8)
        correlation_loss = torch.mean(torch.abs(pred_complex_norm - target_complex_norm)**2)
        
        # Combined loss emphasizing subcarrier-level accuracy
        loss = (self.magnitude_weight * subcarrier_mag_loss +
               0.5 * self.magnitude_weight * mean_distribution_loss +
               0.3 * self.magnitude_weight * std_distribution_loss +
               self.phase_weight * subcarrier_phase_loss +
               0.2 * correlation_loss)
        
        return loss
    
    def get_weight_info(self) -> Dict[str, float]:
        """
        Get information about the current weight configuration.
        
        Returns:
            Dictionary containing original and normalized weights
        """
        info = {
            'normalize_weights': self.normalize_weights,
        }
        
        # Add original weights
        info.update({
            'original_phase_weight': self.original_phase_weight,
            'original_magnitude_weight': self.original_magnitude_weight,
        })
        
        # Add current (possibly normalized) weights
        info.update({
            'current_phase_weight': self.phase_weight,
            'current_magnitude_weight': self.magnitude_weight,
        })
        
        # Add normalization info
        total_weight = self.magnitude_weight + self.phase_weight
        info['total_normalized_weight'] = total_weight
        info['weight_distribution'] = {
            'magnitude': f"{self.magnitude_weight:.3f} ({self.magnitude_weight/total_weight*100:.1f}%)",
            'phase': f"{self.phase_weight:.3f} ({self.phase_weight/total_weight*100:.1f}%)"
        }
        
        return info
    
    def print_weight_info(self):
        """Print weight configuration information for debugging."""
        info = self.get_weight_info()
        print(f"\n📊 CSI Loss Weight Configuration:")
        print(f"   Loss Type: subcarrier_precise")
        print(f"   Normalize Weights: {info['normalize_weights']}")
        
        if 'weight_distribution' in info:
            print(f"   Weight Distribution:")
            for component, weight_str in info['weight_distribution'].items():
                print(f"     {component}: {weight_str}")
        
        print(f"   Original Weights:")
        print(f"     magnitude: {info['original_magnitude_weight']}")
        print(f"     phase: {info['original_phase_weight']}")
        
        print(f"   Current Weights:")
        print(f"     magnitude: {info['current_magnitude_weight']:.6f}")
        print(f"     phase: {info['current_phase_weight']:.6f}")
    
    # _complex_correlation method removed - was incorrectly implemented
    # (mixed all subcarriers and antennas together)
