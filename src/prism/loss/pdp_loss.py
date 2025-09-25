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
    
    def __init__(self, loss_type: str = 'hybrid', fft_size: int = 2046,
                 normalize_pdp: bool = True, mse_weight: float = 0.7, 
                 delay_weight: float = 0.3):
        """
        Initialize PDP loss function
        
        Args:
            loss_type: Type of PDP loss ('mse', 'delay', 'hybrid')
                      Note: 'correlation' type is disabled due to incorrect implementation
            fft_size: FFT size for PDP computation
            normalize_pdp: Whether to normalize PDPs before comparison
            mse_weight: Weight for MSE component in hybrid loss
            delay_weight: Weight for delay component in hybrid loss
        """
        super(PDPLoss, self).__init__()
        self.loss_type = loss_type
        self.fft_size = int(fft_size)  # Ensure fft_size is always an integer
        self.normalize_pdp = normalize_pdp
        self.mse_weight = mse_weight
        self.delay_weight = delay_weight
        
        # Validate hybrid loss weights if applicable
        if loss_type == 'hybrid':
            if abs(self.mse_weight + self.delay_weight - 1.0) > 1e-6:
                logger.warning(f"PDP hybrid loss weights ({self.mse_weight:.3f} + {self.delay_weight:.3f} = {self.mse_weight + self.delay_weight:.3f}) do not sum to 1.0, consider normalizing for better interpretability")
        
    def forward(self, predicted_csi: torch.Tensor, target_csi: torch.Tensor) -> torch.Tensor:
        """
        Compute PDP loss between predicted and target CSI
        
        Args:
            predicted_csi: Predicted CSI tensor (complex)
                          Shape: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
            target_csi: Target CSI tensor (complex)
                       Shape: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
        
        Returns:
            loss: Computed PDP loss value (scalar tensor)
        """
        # Validate input shapes
        if predicted_csi.shape != target_csi.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_csi.shape} vs target {target_csi.shape}")
        
        # Ensure complex tensors with simplified conversion
        if not predicted_csi.is_complex():
            predicted_csi = torch.view_as_complex(
                torch.stack([predicted_csi, torch.zeros_like(predicted_csi)], dim=-1)
            )
                
        if not target_csi.is_complex():
            target_csi = torch.view_as_complex(
                torch.stack([target_csi, torch.zeros_like(target_csi)], dim=-1)
            )
        
        # Handle 4D input: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
        if predicted_csi.dim() == 4:
            batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = predicted_csi.shape
            
            # Compute PDP loss for each antenna pair and average
            total_loss = 0.0
            total_pairs = 0
            
            for bs_idx in range(num_bs_antennas):
                for ue_idx in range(num_ue_antennas):
                    # Extract CSI for this antenna pair: [batch_size, num_subcarriers]
                    antenna_pred = predicted_csi[:, bs_idx, ue_idx, :]  # [batch_size, num_subcarriers]
                    antenna_target = target_csi[:, bs_idx, ue_idx, :]    # [batch_size, num_subcarriers]
                    
                    # Compute PDP loss for this antenna pair across all samples
                    antenna_loss = self._compute_antenna_pair_pdp_loss(antenna_pred, antenna_target)
                    total_loss += antenna_loss
                    total_pairs += 1
            
            # Average loss across all antenna pairs
            avg_loss = total_loss / total_pairs if total_pairs > 0 else torch.tensor(0.0, device=predicted_csi.device)
            return avg_loss
        
        # Handle 1D input (legacy support): [num_subcarriers]
        elif predicted_csi.dim() == 1:
            # Compute PDPs
            pdp_pred = self._compute_pdp(predicted_csi)
            pdp_target = self._compute_pdp(target_csi)
            
            # Normalize PDPs if required
            if self.normalize_pdp:
                pdp_pred = self._normalize_pdp(pdp_pred)
                pdp_target = self._normalize_pdp(pdp_target)
            
            # Compute loss using the extracted method
            loss = self._compute_loss_from_pdps(pdp_pred, pdp_target)
            return loss
        else:
            raise ValueError(f"Unsupported input dimension: {predicted_csi.dim()}D. Expected 1D or 4D tensors.")
    
    def _compute_antenna_pair_pdp_loss(self, antenna_pred: torch.Tensor, antenna_target: torch.Tensor) -> torch.Tensor:
        """
        Compute PDP loss for a single antenna pair across all samples in the batch
        
        Args:
            antenna_pred: Predicted CSI for antenna pair [batch_size, num_subcarriers]
            antenna_target: Target CSI for antenna pair [batch_size, num_subcarriers]
        
        Returns:
            loss: Average PDP loss across all samples in the batch
        """
        batch_size = antenna_pred.shape[0]
        total_loss = 0.0
        
        for sample_idx in range(batch_size):
            # Extract CSI for this sample: [num_subcarriers]
            sample_pred = antenna_pred[sample_idx, :]
            sample_target = antenna_target[sample_idx, :]
            
            # Compute PDPs for this sample
            pdp_pred = self._compute_pdp(sample_pred)
            pdp_target = self._compute_pdp(sample_target)
            
            # Normalize PDPs if required
            if self.normalize_pdp:
                pdp_pred = self._normalize_pdp(pdp_pred)
                pdp_target = self._normalize_pdp(pdp_target)
            
            # Compute loss for this sample
            sample_loss = self._compute_loss_from_pdps(pdp_pred, pdp_target)
            total_loss += sample_loss
        
        # Average loss across all samples
        avg_loss = total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=antenna_pred.device)
        return avg_loss
    
    def _compute_loss_from_pdps(self, pdp_pred: torch.Tensor, pdp_target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss from PDP tensors
        
        Args:
            pdp_pred: Predicted PDP tensor
            pdp_target: Target PDP tensor
        
        Returns:
            loss: Computed loss value
        """
        if self.loss_type == 'mse':
            # PDP MSE Loss
            loss = F.mse_loss(pdp_pred, pdp_target)
            
        elif self.loss_type == 'delay':
            # Dominant Path Delay Loss
            loss = self._compute_delay_loss(pdp_pred, pdp_target)
            
        elif self.loss_type == 'hybrid':
            # Hybrid PDP Loss: MSE + Delay with customizable weights
            mse_loss = F.mse_loss(pdp_pred, pdp_target)
            delay_loss = self._compute_delay_loss(pdp_pred, pdp_target)
            
            # Use customizable weights instead of hardcoded values
            loss = (self.mse_weight * mse_loss + self.delay_weight * delay_loss)
            
        else:
            raise ValueError(f"Unknown PDP loss type: {self.loss_type}")
        
        return loss
    
    def _compute_pdp(self, csi_data: torch.Tensor) -> torch.Tensor:
        """
        Power Delay Profile computation using IFFT - each CSI sequence separately
        
        Args:
            csi_data: CSI data tensor (complex)
                     Shape: (..., N) - any shape with last dimension as subcarriers
        
        Returns:
            pdp: Power delay profile tensor
                Shape: (..., fft_size)
        """
        # Save original shape
        original_shape = csi_data.shape
        
        # Handle empty input edge case
        if csi_data.numel() == 0:
            return torch.zeros(original_shape[:-1] + (self.fft_size,), device=csi_data.device)
        
        # Zero-pad or truncate each CSI sequence to fft_size
        if csi_data.shape[-1] < self.fft_size:
            # Zero-pad each sequence
            pad_size = self.fft_size - csi_data.shape[-1]
            padded_csi = F.pad(csi_data, (0, pad_size), mode='constant', value=0)
        else:
            # Truncate each sequence
            padded_csi = csi_data[..., :self.fft_size]
        
        # Compute IFFT for each CSI sequence separately (along the last dimension)
        time_domain = torch.fft.ifft(padded_csi, dim=-1)
        pdp = torch.abs(time_domain) ** 2
        
        return pdp
    
    def _normalize_pdp(self, pdp: torch.Tensor) -> torch.Tensor:
        """
        Normalize PDP (energy normalization)
        """
        total_energy = torch.sum(pdp)
        if total_energy < 1e-8:
            return pdp
        return pdp / total_energy
    
    # _compute_pdp_correlation_loss method removed - was incorrectly implemented
    # (mixed all delay bins together)
    
    def _compute_delay_loss(self, pdp_pred: torch.Tensor, 
                           pdp_target: torch.Tensor) -> torch.Tensor:
        """
        Improved delay loss computation with adaptive temperature scaling
        """
        # Get device for tensor operations
        device = pdp_pred.device
        
        # Flatten all dimensions except the last one
        original_shape = pdp_pred.shape
        pdp_pred_flat = pdp_pred.reshape(-1, original_shape[-1])
        pdp_target_flat = pdp_target.reshape(-1, original_shape[-1])
        
        # Create delay indices with proper device placement
        indices = torch.arange(original_shape[-1], dtype=torch.float32, device=device)
        
        # Adaptive temperature scaling based on PDP dynamic range
        pred_range = pdp_pred_flat.max(dim=-1, keepdim=True)[0] - pdp_pred_flat.min(dim=-1, keepdim=True)[0]
        target_range = pdp_target_flat.max(dim=-1, keepdim=True)[0] - pdp_target_flat.min(dim=-1, keepdim=True)[0]
        
        # Avoid division by zero
        pred_range = torch.clamp(pred_range, min=1e-8)
        target_range = torch.clamp(target_range, min=1e-8)
        
        # Dynamic temperature parameters for better stability
        pred_temp = 10.0 / pred_range
        target_temp = 10.0 / target_range
        
        # Soft argmax with improved numerical stability using log-space computation
        pred_logits = pdp_pred_flat * pred_temp - torch.logsumexp(pdp_pred_flat * pred_temp, dim=-1, keepdim=True)
        pred_weights = torch.exp(pred_logits)
        
        target_logits = pdp_target_flat * target_temp - torch.logsumexp(pdp_target_flat * target_temp, dim=-1, keepdim=True)
        target_weights = torch.exp(target_logits)
        
        # Compute weighted delay indices
        pred_delay = torch.sum(indices * pred_weights, dim=-1)
        target_delay = torch.sum(indices * target_weights, dim=-1)
        
        # Normalized delay difference
        delay_diff = torch.abs(pred_delay - target_delay) / original_shape[-1]
        
        return torch.mean(delay_diff)
    
    def compute_pdp_loss(self, predicted_csi: torch.Tensor, target_csi: torch.Tensor) -> torch.Tensor:
        """
        Public interface for computing PDP loss - convenient for external testing
        
        Args:
            predicted_csi: Predicted CSI tensor (complex)
                          Shape: Any shape with last dimension as subcarriers
            target_csi: Target CSI tensor (complex)  
                       Shape: Same as predicted_csi
        
        Returns:
            loss: Computed PDP loss value (scalar tensor)
        """
        return self.forward(predicted_csi, target_csi)
    
    def compute_pdp_only(self, csi_data: torch.Tensor) -> torch.Tensor:
        """
        Public interface for computing PDP from CSI data only (without loss calculation)
        Useful for analysis and visualization
        
        Args:
            csi_data: CSI data tensor (complex)
                     Shape: Any shape with last dimension as subcarriers
        
        Returns:
            pdp: Power delay profile tensor
                Shape: Same as input except last dimension becomes fft_size
        """
        # Ensure complex tensor
        if not csi_data.is_complex():
            csi_data = torch.view_as_complex(
                torch.stack([csi_data, torch.zeros_like(csi_data)], dim=-1)
            )
        
        # Compute PDP
        pdp = self._compute_pdp(csi_data)
        
        # Normalize if required
        if self.normalize_pdp:
            pdp = self._normalize_pdp(pdp)
            
        return pdp