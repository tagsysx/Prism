"""
Power Delay Profile (PDP) Loss Functions

Provides time-domain validation by comparing PDPs derived from CSI data.
Supports MSE, correlation, delay, and hybrid PDP losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict

# Get logger for this module
logger = logging.getLogger(__name__)


class PDPLoss(nn.Module):
    """
    Power Delay Profile (PDP) Loss Function
    
    Provides time-domain validation by comparing PDPs derived from CSI data.
    Converts CSI to PDP using IFFT and computes MSE loss between PDPs.
    """
    
    def __init__(self, fft_size: int = 1024, normalize_pdp: bool = True):
        """
        Initialize PDP loss function
        
        Args:
            fft_size: Size of FFT for PDP computation (default: 1024)
            normalize_pdp: Whether to normalize PDPs before loss computation (default: True)
        """
        super(PDPLoss, self).__init__()
        
        self.fft_size = int(fft_size)  # Ensure fft_size is always an integer
        self.normalize_pdp = normalize_pdp
        
        logger.info(f"PDP Loss initialized:")
        logger.info(f"  FFT size: {fft_size}")
        logger.info(f"  Normalize PDP: {normalize_pdp}")
        logger.info(f"  Loss computation: MSE between PDPs")
        
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute PDP loss between predicted and target CSI
        
        Args:
            predictions: Dictionary containing predicted values
                        - 'csi': Predicted CSI tensor [batch_size, num_subcarriers] (complex)
                        - Other keys are optional and not used in PDP loss
            targets: Dictionary containing target values (same structure as predictions)
        
        Returns:
            loss: Computed PDP MSE loss value (scalar tensor)
        """
        # Extract CSI tensors from dictionaries
        predicted_csi = predictions['csi']
        target_csi = targets['csi']
        
        # Validate input shapes
        if predicted_csi.shape != target_csi.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_csi.shape} vs target {target_csi.shape}")
        
        # Ensure complex tensors
        if not predicted_csi.is_complex():
            predicted_csi = torch.view_as_complex(
                torch.stack([predicted_csi, torch.zeros_like(predicted_csi)], dim=-1)
            )
                
        if not target_csi.is_complex():
            target_csi = torch.view_as_complex(
                torch.stack([target_csi, torch.zeros_like(target_csi)], dim=-1)
            )
        
        # Only support 2D input: [batch_size, num_subcarriers]
        if predicted_csi.dim() != 2:
            raise ValueError(f"Expected 2D input [batch_size, num_subcarriers], got {predicted_csi.dim()}D: {predicted_csi.shape}")
        
        batch_size, num_subcarriers = predicted_csi.shape
        
        # Convert CSI to PDP for each sample in the batch
        pdp_pred_batch = []
        pdp_target_batch = []
        
        for batch_idx in range(batch_size):
            # Extract CSI for this sample: [num_subcarriers]
            sample_pred_csi = predicted_csi[batch_idx, :]
            sample_target_csi = target_csi[batch_idx, :]
            
            # Convert CSI to PDP using configured FFT size
            pdp_pred = self._csi_to_pdp(sample_pred_csi)
            pdp_target = self._csi_to_pdp(sample_target_csi)
            
            # Normalize PDPs if required
            if self.normalize_pdp:
                pdp_pred = self._normalize_pdp(pdp_pred)
                pdp_target = self._normalize_pdp(pdp_target)
            
            pdp_pred_batch.append(pdp_pred)
            pdp_target_batch.append(pdp_target)
        
        # Stack PDPs into batch tensors
        pdp_pred_batch = torch.stack(pdp_pred_batch, dim=0)  # [batch_size, fft_size]
        pdp_target_batch = torch.stack(pdp_target_batch, dim=0)  # [batch_size, fft_size]
        
        # Compute MSE loss between PDPs
        mse_loss = F.mse_loss(pdp_pred_batch, pdp_target_batch)
        
        return mse_loss
    
    def _csi_to_pdp(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Convert CSI to Power Delay Profile using IFFT.
        
        Args:
            csi: Complex CSI tensor [num_subcarriers]
            
        Returns:
            pdp: Power Delay Profile [fft_size] (real-valued)
        """
        # Pad or truncate CSI to match FFT size
        num_subcarriers = csi.shape[0]
        
        if num_subcarriers < self.fft_size:
            # Zero-pad CSI to FFT size
            padded_csi = torch.zeros(self.fft_size, dtype=csi.dtype, device=csi.device)
            padded_csi[:num_subcarriers] = csi
        elif num_subcarriers > self.fft_size:
            # Truncate CSI to FFT size
            padded_csi = csi[:self.fft_size]
        else:
            padded_csi = csi
        
        # Apply IFFT to get impulse response
        impulse_response = torch.fft.ifft(padded_csi, n=self.fft_size)
        
        # Compute power delay profile (magnitude squared)
        pdp = torch.abs(impulse_response) ** 2
        
        return pdp
    
    def _normalize_pdp(self, pdp: torch.Tensor) -> torch.Tensor:
        """
        Normalize PDP to unit energy.
        
        Args:
            pdp: Power Delay Profile [fft_size]
            
        Returns:
            normalized_pdp: Normalized PDP [fft_size]
        """
        # Compute total energy
        total_energy = torch.sum(pdp)
        
        # Avoid division by zero
        if total_energy < 1e-12:
            return pdp
        
        # Normalize to unit energy
        normalized_pdp = pdp / total_energy
        
        return normalized_pdp