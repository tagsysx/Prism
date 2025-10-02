"""
Power Delay Profile (PDP) Loss Functions

Provides time-domain validation by comparing PDPs derived from CSI data.
Supports MSE, correlation, delay, and hybrid PDP losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Get logger for this module
logger = logging.getLogger(__name__)


class PDPLoss(nn.Module):
    """
    Power Delay Profile (PDP) Loss Function
    
    Provides time-domain validation by comparing PDPs derived from CSI data.
    Converts CSI to PDP using IFFT and computes loss between PDPs using various metrics.
    
    Supported loss types:
    - 'mse': Mean Squared Error (default)
    - 'mae': Mean Absolute Error  
    - 'cosine': Cosine similarity loss for pattern matching (range [0,1])
    """
    
    def __init__(self, fft_size: int = 1024, normalize_pdp: bool = True, loss_type: str = 'mse', 
                 debug_dir: Optional[str] = None, debug_sample_rate: float = 0.5):
        """
        Initialize PDP loss function
        
        Args:
            fft_size: Size of FFT for PDP computation (default: 1024)
            normalize_pdp: Whether to normalize PDPs before loss computation (default: True)
            loss_type: Type of loss ('mse', 'mae', 'cosine') (default: 'mse')
            debug_dir: Directory to save debug PDP plots (default: None, no plotting)
            debug_sample_rate: Probability to save debug plots (default: 0.5, 50%)
        """
        super(PDPLoss, self).__init__()
        
        self.fft_size = int(fft_size)  # Ensure fft_size is always an integer
        self.normalize_pdp = normalize_pdp
        self.loss_type = loss_type.lower()
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.debug_sample_rate = debug_sample_rate
        
        # Create debug directory if specified
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Debug plots will be saved to: {self.debug_dir}")
            logger.info(f"  Debug sample rate: {self.debug_sample_rate*100:.1f}%")
        
        # Validate loss type
        valid_types = ['mse', 'mae', 'cosine']
        if self.loss_type not in valid_types:
            raise ValueError(f"Invalid loss_type: {self.loss_type}. Must be one of {valid_types}")
        
        logger.info(f"PDP Loss initialized:")
        logger.info(f"  FFT size: {fft_size}")
        logger.info(f"  Normalize PDP: {normalize_pdp}")
        logger.info(f"  Loss type: {self.loss_type}")
        
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
                # Normalize predicted PDP
                total_energy_pred = torch.sum(pdp_pred)
                if total_energy_pred >= 1e-12:
                    pdp_pred = pdp_pred / total_energy_pred
                
                # Normalize target PDP
                total_energy_target = torch.sum(pdp_target)
                if total_energy_target >= 1e-12:
                    pdp_target = pdp_target / total_energy_target
            
            pdp_pred_batch.append(pdp_pred)
            pdp_target_batch.append(pdp_target)
        
        # Stack PDPs into batch tensors
        pdp_pred_batch = torch.stack(pdp_pred_batch, dim=0)  # [batch_size, fft_size]
        pdp_target_batch = torch.stack(pdp_target_batch, dim=0)  # [batch_size, fft_size]
        
        # Debug: randomly save PDP comparison plots
        if self.debug_dir and random.random() < self.debug_sample_rate:
            self._save_debug_plot(pdp_pred_batch, pdp_target_batch)
        
        # Compute loss based on specified type - per-CSI calculation
        if self.loss_type == 'mse':
            # Compute MSE loss per sample: [batch_size]
            loss_per_sample = torch.mean((pdp_pred_batch - pdp_target_batch) ** 2, dim=1)
            
        elif self.loss_type == 'mae':
            # Compute MAE loss per sample: [batch_size]
            loss_per_sample = torch.mean(torch.abs(pdp_pred_batch - pdp_target_batch), dim=1)
            
        elif self.loss_type == 'cosine':
            # Cosine similarity loss for PDP pattern matching - per sample
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            
            # Compute cosine similarity for each sample
            dot_product = torch.sum(pdp_pred_batch * pdp_target_batch, dim=1)  # [batch_size]
            pred_norm = torch.norm(pdp_pred_batch, dim=1) + eps  # [batch_size]
            target_norm = torch.norm(pdp_target_batch, dim=1) + eps  # [batch_size]
            
            cosine_similarity = dot_product / (pred_norm * target_norm)  # [batch_size]
            
            # Convert to loss: (1 - (1 + cosine) / 2) (range [0, 1], 0 = perfect similarity)
            # cosine=1 → loss=0 (best), cosine=0 → loss=0.5 (medium), cosine=-1 → loss=1 (worst)
            loss_per_sample = 1.0 - (1.0 + cosine_similarity) / 2.0  # [batch_size]
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Return mean loss for backward compatibility, but now computed per-CSI first
        loss = torch.mean(loss_per_sample)
        return loss
    
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
    
    def _save_debug_plot(self, pdp_pred_batch: torch.Tensor, pdp_target_batch: torch.Tensor):
        """
        Save debug plot comparing predicted and target PDPs
        
        Args:
            pdp_pred_batch: Predicted PDPs [batch_size, fft_size]
            pdp_target_batch: Target PDPs [batch_size, fft_size]
        """
        try:
            # Randomly select one sample from the batch
            batch_size = pdp_pred_batch.shape[0]
            sample_idx = random.randint(0, batch_size - 1)
            
            # Get PDPs for selected sample
            pdp_pred = pdp_pred_batch[sample_idx].detach().cpu().numpy()
            pdp_target = pdp_target_batch[sample_idx].detach().cpu().numpy()
            
            # Create delay axis (in samples)
            delay_samples = range(len(pdp_pred))
            
            # Compute similarity metrics
            mse = ((pdp_pred - pdp_target) ** 2).mean()
            mae = abs(pdp_pred - pdp_target).mean()
            
            # Cosine similarity
            dot_product = (pdp_pred * pdp_target).sum()
            pred_norm = (pdp_pred ** 2).sum() ** 0.5
            target_norm = (pdp_target ** 2).sum() ** 0.5
            cosine_sim = dot_product / (pred_norm * target_norm + 1e-8)
            cosine_loss = 1.0 - (1.0 + cosine_sim) / 2.0
            
            # Determine actual loss value based on loss type
            if self.loss_type == 'mse':
                actual_loss = mse
                loss_info = f"MSE Loss: {mse:.6f}"
            elif self.loss_type == 'mae':
                actual_loss = mae
                loss_info = f"MAE Loss: {mae:.6f}"
            elif self.loss_type == 'cosine':
                actual_loss = cosine_loss
                loss_info = f"Cosine Loss: {cosine_loss:.6f} (sim: {cosine_sim:.4f})"
            else:
                actual_loss = mse
                loss_info = f"Loss: {mse:.6f}"
            
            # Create plot
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Overlay comparison
            axes[0].plot(delay_samples, pdp_pred, 'b-', linewidth=2, label='Predicted', alpha=0.8)
            axes[0].plot(delay_samples, pdp_target, 'r--', linewidth=2, label='Target', alpha=0.8)
            axes[0].set_xlabel('Delay (samples)', fontsize=11)
            axes[0].set_ylabel('Power', fontsize=11)
            axes[0].set_title(f'PDP Comparison [{self.loss_type.upper()}] | {loss_info} | MSE: {mse:.6f}', 
                            fontsize=12, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Error plot
            error = pdp_pred - pdp_target
            axes[1].plot(delay_samples, error, 'g-', linewidth=1.5, label='Error (Pred - Target)')
            axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[1].set_xlabel('Delay (samples)', fontsize=11)
            axes[1].set_ylabel('Error', fontsize=11)
            axes[1].set_title('Prediction Error', fontsize=12, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'pdp_comparison_{timestamp}_sample{sample_idx}.png'
            filepath = self.debug_dir / filename
            
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"Saved debug PDP plot: {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug PDP plot: {e}")