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
                 normalize_weights: bool = True, max_magnitude: float = 100.0):
        """
        Initialize CSI loss function
        
        Args:
            phase_weight: Weight for phase component in the loss
            magnitude_weight: Weight for magnitude component in the loss
            normalize_weights: Whether to normalize weights to sum to 1.0 for balanced scaling
            max_magnitude: Maximum magnitude value from config for numerical stability
        """
        super(CSILoss, self).__init__()
        self.normalize_weights = normalize_weights
        self.max_magnitude = max_magnitude
        
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
        # Use max_magnitude from config instead of hardcoded value
        pred_mag = torch.clamp(pred_mag, min=1e-12, max=self.max_magnitude)
        target_mag = torch.clamp(target_mag, min=1e-12, max=self.max_magnitude)
        
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
        
        # 1. Per-subcarrier magnitude MSE
        # Handle 4D format: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
        if pred_mag.dim() != 4:
            raise ValueError(f"Expected 4D tensor [batch, bs_antennas, ue_antennas, subcarriers], got {pred_mag.shape}")
        
        batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = pred_mag.shape
        
        # Per-antenna-pair magnitude loss: compute MSE for each antenna pair independently
        antenna_pair_mag_losses = []
        for b in range(batch_size):
            for bs in range(num_bs_antennas):
                for ue in range(num_ue_antennas):
                    # Get magnitude vectors for this antenna pair: [num_subcarriers]
                    pred_antenna = pred_mag[b, bs, ue, :]
                    target_antenna = target_mag[b, bs, ue, :]
                    
                    # Compute MSE for this antenna pair (all subcarriers)
                    antenna_mse = torch.mean((pred_antenna - target_antenna)**2)
                    antenna_pair_mag_losses.append(antenna_mse)
        
        # Average MSE across all antenna pairs
        subcarrier_mag_loss = torch.mean(torch.stack(antenna_pair_mag_losses))
        
        # 2. Per-antenna-pair magnitude distribution preservation
        antenna_pair_mean_losses = []
        antenna_pair_std_losses = []
        
        for b in range(batch_size):
            for bs in range(num_bs_antennas):
                for ue in range(num_ue_antennas):
                    # Get magnitude vectors for this antenna pair: [num_subcarriers]
                    pred_antenna = pred_mag[b, bs, ue, :]
                    target_antenna = target_mag[b, bs, ue, :]
                    
                    # Compute mean and std for this antenna pair
                    pred_mean = torch.mean(pred_antenna)
                    target_mean = torch.mean(target_antenna)
                    pred_std = torch.sqrt(torch.clamp(torch.var(pred_antenna, unbiased=False), min=1e-12))
                    target_std = torch.sqrt(torch.clamp(torch.var(target_antenna, unbiased=False), min=1e-12))
                    
                    # Compute distribution losses for this antenna pair
                    mean_loss = (pred_mean - target_mean)**2
                    std_loss = (pred_std - target_std)**2
                    
                    antenna_pair_mean_losses.append(mean_loss)
                    antenna_pair_std_losses.append(std_loss)
        
        # Average distribution losses across all antenna pairs
        mean_distribution_loss = torch.mean(torch.stack(antenna_pair_mean_losses))
        std_distribution_loss = torch.mean(torch.stack(antenna_pair_std_losses))
        
        # Debug: Check distribution losses
        if torch.isnan(mean_distribution_loss) or torch.isinf(mean_distribution_loss):
            logger.error(f"❌ NaN/Inf in mean distribution loss: {mean_distribution_loss}")
        if torch.isnan(std_distribution_loss) or torch.isinf(std_distribution_loss):
            logger.error(f"❌ NaN/Inf in std distribution loss: {std_distribution_loss}")
        
        # 3. Per-subcarrier phase consistency
        # Add numerical stability for phase calculation
        pred_phase = torch.atan2(predicted_csi.imag, predicted_csi.real + 1e-8)
        target_phase = torch.atan2(target_csi.imag, target_csi.real + 1e-8)
        
        phase_diff = torch.remainder(pred_phase - target_phase + np.pi, 2*np.pi) - np.pi
        
        # Per-antenna-pair phase loss: compute MSE for each antenna pair independently
        antenna_pair_phase_losses = []
        for b in range(batch_size):
            for bs in range(num_bs_antennas):
                for ue in range(num_ue_antennas):
                    # Get phase difference vector for this antenna pair: [num_subcarriers]
                    phase_diff_antenna = phase_diff[b, bs, ue, :]
                    
                    # Compute MSE for this antenna pair (all subcarriers)
                    antenna_phase_mse = torch.mean(phase_diff_antenna**2)
                    antenna_pair_phase_losses.append(antenna_phase_mse)
        
        # Average MSE across all antenna pairs
        subcarrier_phase_loss = torch.mean(torch.stack(antenna_pair_phase_losses))
        
        # 4. Per-antenna-pair complex correlation preservation (maintains complex relationship)
        # Use clipped magnitudes for normalization
        pred_complex_norm = predicted_csi / (pred_mag + 1e-8)
        target_complex_norm = target_csi / (target_mag + 1e-8)
        
        # Per-antenna-pair correlation loss: compute MSE for each antenna pair independently
        antenna_pair_correlation_losses = []
        for b in range(batch_size):
            for bs in range(num_bs_antennas):
                for ue in range(num_ue_antennas):
                    # Get normalized complex vectors for this antenna pair: [num_subcarriers]
                    pred_norm_antenna = pred_complex_norm[b, bs, ue, :]
                    target_norm_antenna = target_complex_norm[b, bs, ue, :]
                    
                    # Compute MSE for this antenna pair (all subcarriers)
                    antenna_corr_mse = torch.mean(torch.abs(pred_norm_antenna - target_norm_antenna)**2)
                    antenna_pair_correlation_losses.append(antenna_corr_mse)
        
        # Average MSE across all antenna pairs
        correlation_loss = torch.mean(torch.stack(antenna_pair_correlation_losses))
        
        # 5. Per-antenna-pair Jensen-Shannon Divergence for distribution similarity
        antenna_pair_js_losses = []
        for b in range(batch_size):
            for bs in range(num_bs_antennas):
                for ue in range(num_ue_antennas):
                    # Get CSI vectors for this antenna pair: [num_subcarriers]
                    pred_antenna = predicted_csi[b, bs, ue, :]
                    target_antenna = target_csi[b, bs, ue, :]
                    
                    # Compute JS divergence for this antenna pair (all subcarriers)
                    js_loss = self._compute_js_divergence(pred_antenna, target_antenna)
                    antenna_pair_js_losses.append(js_loss)
        
        # Average JS divergence loss across all antenna pairs
        js_divergence_loss = torch.mean(torch.stack(antenna_pair_js_losses))
        
        # Combined loss emphasizing subcarrier-level accuracy
        loss = (self.magnitude_weight * subcarrier_mag_loss +
               0.5 * self.magnitude_weight * mean_distribution_loss +
               0.3 * self.magnitude_weight * std_distribution_loss +
               self.phase_weight * subcarrier_phase_loss +
               0.2 * correlation_loss +
               0.1 * js_divergence_loss)  # Add JS divergence loss
        
        # Log CSI loss components
        logger.info(f"📊 CSI Loss Components:")
        logger.info(f"   Magnitude Loss: {subcarrier_mag_loss.item():.6f} (weight: {self.magnitude_weight})")
        logger.info(f"   Mean Distribution Loss: {mean_distribution_loss.item():.6f} (weight: {0.5 * self.magnitude_weight:.3f})")
        logger.info(f"   Std Distribution Loss: {std_distribution_loss.item():.6f} (weight: {0.3 * self.magnitude_weight:.3f})")
        logger.info(f"   Phase Loss: {subcarrier_phase_loss.item():.6f} (weight: {self.phase_weight})")
        logger.info(f"   Correlation Loss: {correlation_loss.item():.6f} (weight: 0.2)")
        logger.info(f"   JS Divergence Loss: {js_divergence_loss.item():.6f} (weight: 0.1)")
        logger.info(f"   Total CSI Loss: {loss.item():.6f}")
        
        # Final NaN/Inf check and replacement
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"❌ Final CSI loss is NaN/Inf: {loss}")
            logger.error(f"   Component losses:")
            logger.error(f"     magnitude: {subcarrier_mag_loss}")
            logger.error(f"     mean_dist: {mean_distribution_loss}")
            logger.error(f"     std_dist: {std_distribution_loss}")
            logger.error(f"     phase: {subcarrier_phase_loss}")
            logger.error(f"     correlation: {correlation_loss}")
            logger.error(f"     js_divergence: {js_divergence_loss}")
            # Return a small positive loss to prevent training crash
            loss = torch.tensor(1e-6, device=loss.device, dtype=loss.dtype, requires_grad=True)
        
        return loss
    
    def _compute_js_divergence(self, pred_antenna: torch.Tensor, target_antenna: torch.Tensor) -> torch.Tensor:
        """
        Compute Jensen-Shannon Divergence for a single antenna pair.
        
        Args:
            pred_antenna: Predicted CSI tensor [num_subcarriers]
            target_antenna: Target CSI tensor [num_subcarriers]
            
        Returns:
            JS divergence loss for this antenna pair
        """
        # Convert to amplitude distributions
        pred_amp = torch.abs(pred_antenna)
        target_amp = torch.abs(target_antenna)
        
        # Normalize to probability distributions
        pred_norm = pred_amp / (torch.sum(pred_amp) + 1e-12)
        target_norm = target_amp / (torch.sum(target_amp) + 1e-12)
        
        # Compute average distribution M = 0.5 * (P + Q)
        avg_dist = (pred_norm + target_norm) / 2.0
        
        # Compute KL divergences with numerical stability
        kl_pred = torch.sum(pred_norm * torch.log((pred_norm + 1e-12) / (avg_dist + 1e-12)))
        kl_target = torch.sum(target_norm * torch.log((target_norm + 1e-12) / (avg_dist + 1e-12)))
        
        # Compute JS divergence: JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        js_divergence = (kl_pred + kl_target) / 2.0
        
        # Convert JS divergence to loss (smaller is better)
        # JS divergence range: [0, ln(2)] ≈ [0, 0.693]
        # Scale to [0, 1] using: 2 * (1 - exp(-js_divergence))
        js_loss = 2.0 * (1.0 - torch.exp(-js_divergence))
        
        return js_loss
    
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
