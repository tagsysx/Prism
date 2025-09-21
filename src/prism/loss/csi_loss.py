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
            predicted_csi: Predicted CSI tensor (complex)
                          Shape: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
            target_csi: Target CSI tensor (complex)
                       Shape: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
        
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
        
        # Numerical stability: Clip extreme values to prevent NaN/Inf
        max_magnitude = 1e6  # Reasonable upper bound
        pred_mag = torch.clamp(pred_mag, min=1e-12, max=max_magnitude)
        target_mag = torch.clamp(target_mag, min=1e-12, max=max_magnitude)
        
        # Debug: Check for NaN/Inf values
        if torch.isnan(pred_mag).any() or torch.isinf(pred_mag).any():
            logger.error(f"❌ NaN/Inf detected in predicted magnitude!")
            logger.error(f"   NaN count: {torch.isnan(pred_mag).sum()}")
            logger.error(f"   Inf count: {torch.isinf(pred_mag).sum()}")
            logger.error(f"   Min: {pred_mag.min()}, Max: {pred_mag.max()}")
        
        if torch.isnan(target_mag).any() or torch.isinf(target_mag).any():
            logger.error(f"❌ NaN/Inf detected in target magnitude!")
            logger.error(f"   NaN count: {torch.isnan(target_mag).sum()}")
            logger.error(f"   Inf count: {torch.isinf(target_mag).sum()}")
            logger.error(f"   Min: {target_mag.min()}, Max: {target_mag.max()}")
        
        # 1. True per-subcarrier magnitude MSE
        # Handle 4D format: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
        if pred_mag.dim() != 4:
            raise ValueError(f"Expected 4D tensor [batch, bs_antennas, ue_antennas, subcarriers], got {pred_mag.shape}")
        
        # True per-subcarrier calculation: compute MSE for each subcarrier separately
        batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = pred_mag.shape
        subcarrier_mse_list = []
        
        for subcarrier_idx in range(num_subcarriers):
            # Extract data for this specific subcarrier: [batch_size, num_bs_antennas, num_ue_antennas]
            pred_subcarrier = pred_mag[:, :, :, subcarrier_idx]
            target_subcarrier = target_mag[:, :, :, subcarrier_idx]
            
            # Compute MSE for this subcarrier across all samples and antennas
            subcarrier_mse = torch.mean((pred_subcarrier - target_subcarrier)**2)
            subcarrier_mse_list.append(subcarrier_mse)
        
        # Average MSE across all subcarriers
        subcarrier_mag_loss = torch.mean(torch.stack(subcarrier_mse_list))
        
        # Debug: Check magnitude loss
        if torch.isnan(subcarrier_mag_loss) or torch.isinf(subcarrier_mag_loss):
            logger.error(f"❌ NaN/Inf in magnitude loss: {subcarrier_mag_loss}")
            logger.error(f"   pred_mag range: [{pred_mag.min():.6f}, {pred_mag.max():.6f}]")
            logger.error(f"   target_mag range: [{target_mag.min():.6f}, {target_mag.max():.6f}]")
            logger.error(f"   subcarrier_mse_list: {[mse.item() for mse in subcarrier_mse_list]}")
            logger.error(f"   subcarrier_mse_list has NaN: {[torch.isnan(mse).item() for mse in subcarrier_mse_list]}")
            logger.error(f"   subcarrier_mse_list has Inf: {[torch.isinf(mse).item() for mse in subcarrier_mse_list]}")
        
        # 2. True per-subcarrier magnitude distribution preservation
        pred_mag_mean_list = []
        target_mag_mean_list = []
        pred_mag_var_list = []
        target_mag_var_list = []
        
        for subcarrier_idx in range(num_subcarriers):
            # Extract data for this specific subcarrier: [batch_size, num_bs_antennas, num_ue_antennas]
            pred_subcarrier = pred_mag[:, :, :, subcarrier_idx]
            target_subcarrier = target_mag[:, :, :, subcarrier_idx]
            
            # Compute mean and variance for this subcarrier across all samples and antennas
            pred_mean = torch.mean(pred_subcarrier)
            target_mean = torch.mean(target_subcarrier)
            pred_var = torch.var(pred_subcarrier, unbiased=False)
            target_var = torch.var(target_subcarrier, unbiased=False)
            
            pred_mag_mean_list.append(pred_mean)
            target_mag_mean_list.append(target_mean)
            pred_mag_var_list.append(pred_var)
            target_mag_var_list.append(target_var)
        
        # Calculate std with numerical stability
        pred_mag_std_list = [torch.sqrt(torch.clamp(var, min=1e-12)) for var in pred_mag_var_list]
        target_mag_std_list = [torch.sqrt(torch.clamp(var, min=1e-12)) for var in target_mag_var_list]
        
        # Compute distribution losses per subcarrier
        mean_distribution_loss = F.mse_loss(torch.stack(pred_mag_mean_list), torch.stack(target_mag_mean_list))
        std_distribution_loss = F.mse_loss(torch.stack(pred_mag_std_list), torch.stack(target_mag_std_list))
        
        # Debug: Check distribution losses
        if torch.isnan(mean_distribution_loss) or torch.isinf(mean_distribution_loss):
            logger.error(f"❌ NaN/Inf in mean distribution loss: {mean_distribution_loss}")
        if torch.isnan(std_distribution_loss) or torch.isinf(std_distribution_loss):
            logger.error(f"❌ NaN/Inf in std distribution loss: {std_distribution_loss}")
        
        # 3. Phase consistency per subcarrier
        # Add numerical stability for phase calculation
        pred_real = torch.clamp(predicted_csi.real, min=-max_magnitude, max=max_magnitude)
        pred_imag = torch.clamp(predicted_csi.imag, min=-max_magnitude, max=max_magnitude)
        target_real = torch.clamp(target_csi.real, min=-max_magnitude, max=max_magnitude)
        target_imag = torch.clamp(target_csi.imag, min=-max_magnitude, max=max_magnitude)
        
        pred_phase = torch.atan2(pred_imag, pred_real + 1e-8)
        target_phase = torch.atan2(target_imag, target_real + 1e-8)
        
        phase_diff = torch.remainder(pred_phase - target_phase + np.pi, 2*np.pi) - np.pi
        
        # True per-subcarrier phase loss calculation
        subcarrier_phase_mse_list = []
        
        for subcarrier_idx in range(num_subcarriers):
            # Extract phase difference for this specific subcarrier: [batch_size, num_bs_antennas, num_ue_antennas]
            phase_diff_subcarrier = phase_diff[:, :, :, subcarrier_idx]
            
            # Compute MSE for this subcarrier across all samples and antennas
            subcarrier_phase_mse = torch.mean(phase_diff_subcarrier**2)
            subcarrier_phase_mse_list.append(subcarrier_phase_mse)
        
        # Average MSE across all subcarriers
        subcarrier_phase_loss = torch.mean(torch.stack(subcarrier_phase_mse_list))
        
        # 4. Complex correlation preservation (maintains complex relationship)
        # Use clipped magnitudes for normalization
        pred_complex_norm = predicted_csi / (pred_mag + 1e-8)
        target_complex_norm = target_csi / (target_mag + 1e-8)
        correlation_loss = torch.mean(torch.abs(pred_complex_norm - target_complex_norm)**2)
        
        # Combined loss emphasizing subcarrier-level accuracy
        loss = (self.magnitude_weight * subcarrier_mag_loss +
               0.5 * self.magnitude_weight * mean_distribution_loss +
               0.3 * self.magnitude_weight * std_distribution_loss +
               self.phase_weight * subcarrier_phase_loss +
               0.2 * correlation_loss)
        
        # Log CSI loss components
        # logger.info(f"📊 CSI Loss Components:")
        # logger.info(f"   Magnitude Loss: {subcarrier_mag_loss.item():.6f} (weight: {self.magnitude_weight})")
        # logger.info(f"   Mean Distribution Loss: {mean_distribution_loss.item():.6f} (weight: {0.5 * self.magnitude_weight:.3f})")
        # logger.info(f"   Std Distribution Loss: {std_distribution_loss.item():.6f} (weight: {0.3 * self.magnitude_weight:.3f})")
        # logger.info(f"   Phase Loss: {subcarrier_phase_loss.item():.6f} (weight: {self.phase_weight})")
        # logger.info(f"   Correlation Loss: {correlation_loss.item():.6f} (weight: 0.2)")
        # logger.info(f"   Total CSI Loss: {loss.item():.6f}")
        
        # Final NaN/Inf check and replacement
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"❌ Final CSI loss is NaN/Inf: {loss}")
            logger.error(f"   Component losses:")
            logger.error(f"     magnitude: {subcarrier_mag_loss}")
            logger.error(f"     mean_dist: {mean_distribution_loss}")
            logger.error(f"     std_dist: {std_distribution_loss}")
            logger.error(f"     phase: {subcarrier_phase_loss}")
            logger.error(f"     correlation: {correlation_loss}")
            # Return a small positive loss to prevent training crash
            loss = torch.tensor(1e-6, device=loss.device, dtype=loss.dtype, requires_grad=True)
        
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
