"""
CSI (Channel State Information) Loss Functions

Provides various loss functions for comparing complex-valued CSI matrices,
including magnitude, phase, correlation, and hybrid losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
import logging

# Get logger for this module
logger = logging.getLogger(__name__)


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
        
    def forward(self, predicted_csi: torch.Tensor, target_csi: torch.Tensor) -> torch.Tensor:
        """
        Compute CSI loss between predicted and target CSI
        
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
        
        # Ensure complex tensors
        if not predicted_csi.is_complex():
            predicted_csi = predicted_csi.to(torch.complex64)
        if not target_csi.is_complex():
            target_csi = target_csi.to(torch.complex64)
        
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
