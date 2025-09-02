"""
Power Delay Profile (PDP) Loss Functions

Provides time-domain validation by comparing PDPs derived from CSI data.
Supports MSE, correlation, delay, and hybrid PDP losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Get logger for this module
logger = logging.getLogger(__name__)


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
