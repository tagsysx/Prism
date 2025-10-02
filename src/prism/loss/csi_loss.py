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
import random
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Get logger for this module
logger = logging.getLogger(__name__)


class CSILoss(nn.Module):
    """
    CSI (Channel State Information) Loss Function
    
    Provides subcarrier-precise loss for comparing complex-valued CSI matrices,
    addressing the complex MSE vs per-subcarrier accuracy paradox.
    """
    
    def __init__(self, phase_weight: float = 1.0, magnitude_weight: float = 1.0,
                 normalize_weights: bool = True, max_magnitude: float = 100.0,
                 debug_dir: Optional[str] = None, debug_sample_rate: float = 0.5):
        """
        Initialize CSI loss function
        
        Args:
            phase_weight: Weight for phase component in the loss
            magnitude_weight: Weight for magnitude component in the loss
            normalize_weights: Whether to normalize weights to sum to 1.0 for balanced scaling
            max_magnitude: Maximum magnitude value from config for numerical stability
            debug_dir: Directory to save debug CSI plots (default: None, no plotting)
            debug_sample_rate: Probability to save debug plots (default: 0.5, 50%)
        """
        super(CSILoss, self).__init__()
        self.normalize_weights = normalize_weights
        self.max_magnitude = max_magnitude
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.debug_sample_rate = debug_sample_rate
        
        # Store original weights for reference
        self.original_phase_weight = phase_weight
        self.original_magnitude_weight = magnitude_weight
        
        # Create debug directory if specified
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"CSI Loss debug plots will be saved to: {self.debug_dir}")
            logger.info(f"CSI Loss debug sample rate: {self.debug_sample_rate*100:.1f}%")
        
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
        
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute subcarrier-precise CSI loss between predicted and target CSI
        
        Args:
            predictions: Dictionary containing predicted values
                        - 'csi': Predicted CSI tensor [batch_size, num_subcarriers] (complex)
                        - 'bs_antenna_indices': BS antenna indices [batch_size] (optional)
                        - 'ue_antenna_indices': UE antenna indices [batch_size] (optional)
                        - 'bs_positions': BS positions [batch_size, 3] (optional)
                        - 'ue_positions': UE positions [batch_size, 3] (optional)
            targets: Dictionary containing target values (same structure as predictions)
        
        Returns:
            loss: Computed loss value (scalar tensor)
        """
        # Extract CSI tensors from dictionaries
        predicted_csi = predictions['csi']
        target_csi = targets['csi']
        
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
        # Handle 2D format: [batch_size, num_subcarriers] - individual CSI samples
        if pred_mag.dim() != 2:
            raise ValueError(f"Expected 2D tensor [batch_size, num_subcarriers], got {pred_mag.shape}")
        
        batch_size, num_subcarriers = pred_mag.shape
        
        # Per-sample magnitude loss: compute MSE for each sample independently
        sample_mag_losses = []
        for b in range(batch_size):
            # Get magnitude vectors for this sample: [num_subcarriers]
            pred_sample = pred_mag[b, :]
            target_sample = target_mag[b, :]
            
            # Compute MSE for this sample (all subcarriers)
            sample_mse = torch.mean((pred_sample - target_sample)**2)
            sample_mag_losses.append(sample_mse)
        
        # Average MSE across all samples
        subcarrier_mag_loss = torch.mean(torch.stack(sample_mag_losses))
        
        # 2. Per-sample magnitude distribution preservation
        sample_mean_losses = []
        sample_std_losses = []
        
        for b in range(batch_size):
            # Get magnitude vectors for this sample: [num_subcarriers]
            pred_sample = pred_mag[b, :]
            target_sample = target_mag[b, :]
            
            # Compute mean and std for this sample
            pred_mean = torch.mean(pred_sample)
            target_mean = torch.mean(target_sample)
            pred_std = torch.sqrt(torch.clamp(torch.var(pred_sample, unbiased=False), min=1e-12))
            target_std = torch.sqrt(torch.clamp(torch.var(target_sample, unbiased=False), min=1e-12))
            
            # Compute distribution losses for this sample
            mean_loss = (pred_mean - target_mean)**2
            std_loss = (pred_std - target_std)**2
            
            sample_mean_losses.append(mean_loss)
            sample_std_losses.append(std_loss)
        
        # Average distribution losses across all samples
        mean_distribution_loss = torch.mean(torch.stack(sample_mean_losses))
        std_distribution_loss = torch.mean(torch.stack(sample_std_losses))
        
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
        
        # Per-sample phase loss: compute MSE for each sample independently
        sample_phase_losses = []
        for b in range(batch_size):
            # Get phase difference vector for this sample: [num_subcarriers]
            phase_diff_sample = phase_diff[b, :]
            
            # Compute MSE for this sample (all subcarriers)
            sample_phase_mse = torch.mean(phase_diff_sample**2)
            sample_phase_losses.append(sample_phase_mse)
        
        # Average MSE across all samples
        subcarrier_phase_loss = torch.mean(torch.stack(sample_phase_losses))
        
        # 4. Per-sample complex correlation preservation (maintains complex relationship)
        # Use clipped magnitudes for normalization
        pred_complex_norm = predicted_csi / (pred_mag + 1e-8)
        target_complex_norm = target_csi / (target_mag + 1e-8)
        
        # Per-sample correlation loss: compute MSE for each sample independently
        sample_correlation_losses = []
        for b in range(batch_size):
            # Get normalized complex vectors for this sample: [num_subcarriers]
            pred_norm_sample = pred_complex_norm[b, :]
            target_norm_sample = target_complex_norm[b, :]
            
            # Compute MSE for this sample (all subcarriers)
            sample_corr_mse = torch.mean(torch.abs(pred_norm_sample - target_norm_sample)**2)
            sample_correlation_losses.append(sample_corr_mse)
        
        # Average MSE across all samples
        correlation_loss = torch.mean(torch.stack(sample_correlation_losses))
        
        # 5. Per-sample Jensen-Shannon Divergence for distribution similarity
        sample_js_losses = []
        for b in range(batch_size):
            # Get CSI vectors for this sample: [num_subcarriers]
            pred_sample = predicted_csi[b, :]
            target_sample = target_csi[b, :]
            
            # Compute JS divergence for this sample (all subcarriers)
            js_loss = self._compute_js_divergence(pred_sample, target_sample)
            sample_js_losses.append(js_loss)
        
        # Average JS divergence loss across all samples
        js_divergence_loss = torch.mean(torch.stack(sample_js_losses))
        
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
        
        # Debug: randomly save CSI comparison plots
        if self.debug_dir and random.random() < self.debug_sample_rate:
            self._save_debug_plot(predicted_csi, target_csi, loss)
        
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
    
    def _save_debug_plot(self, pred_csi: torch.Tensor, target_csi: torch.Tensor, loss_value: torch.Tensor):
        """
        Save debug plot comparing predicted and target CSI
        
        Args:
            pred_csi: Predicted CSI [batch_size, num_subcarriers] (complex)
            target_csi: Target CSI [batch_size, num_subcarriers] (complex)
            loss_value: Computed loss value (scalar tensor)
        """
        try:
            # Randomly select one sample from the batch
            batch_size = pred_csi.shape[0]
            sample_idx = random.randint(0, batch_size - 1)
            
            # Get CSI for selected sample
            pred_sample = pred_csi[sample_idx].detach().cpu().numpy()
            target_sample = target_csi[sample_idx].detach().cpu().numpy()
            
            # Extract magnitude and phase
            pred_mag = np.abs(pred_sample)
            target_mag = np.abs(target_sample)
            pred_phase = np.angle(pred_sample)
            target_phase = np.angle(target_sample)
            
            # Create subcarrier indices
            num_subcarriers = len(pred_sample)
            subcarrier_indices = np.arange(num_subcarriers)
            
            # Compute metrics
            mag_mse = ((pred_mag - target_mag) ** 2).mean()
            mag_mae = np.abs(pred_mag - target_mag).mean()
            phase_mse = ((pred_phase - target_phase) ** 2).mean()
            phase_mae = np.abs(pred_phase - target_phase).mean()
            
            # Create plot with 4 subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Magnitude comparison
            axes[0, 0].plot(subcarrier_indices, pred_mag, 'b-', linewidth=2, label='Predicted', alpha=0.8)
            axes[0, 0].plot(subcarrier_indices, target_mag, 'r--', linewidth=2, label='Target', alpha=0.8)
            axes[0, 0].set_xlabel('Subcarrier Index', fontsize=11)
            axes[0, 0].set_ylabel('Magnitude', fontsize=11)
            axes[0, 0].set_title(f'CSI Magnitude Comparison | MAE: {mag_mae:.6f}, MSE: {mag_mse:.6f}',
                               fontsize=12, fontweight='bold')
            axes[0, 0].legend(fontsize=10)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Phase comparison
            axes[0, 1].plot(subcarrier_indices, pred_phase, 'b-', linewidth=2, label='Predicted', alpha=0.8)
            axes[0, 1].plot(subcarrier_indices, target_phase, 'r--', linewidth=2, label='Target', alpha=0.8)
            axes[0, 1].set_xlabel('Subcarrier Index', fontsize=11)
            axes[0, 1].set_ylabel('Phase (radians)', fontsize=11)
            axes[0, 1].set_title(f'CSI Phase Comparison | MAE: {phase_mae:.6f}, MSE: {phase_mse:.6f}',
                               fontsize=12, fontweight='bold')
            axes[0, 1].legend(fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Magnitude error
            mag_error = pred_mag - target_mag
            axes[1, 0].plot(subcarrier_indices, mag_error, 'g-', linewidth=1.5, label='Error (Pred - Target)')
            axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Subcarrier Index', fontsize=11)
            axes[1, 0].set_ylabel('Magnitude Error', fontsize=11)
            axes[1, 0].set_title('Magnitude Prediction Error', fontsize=12, fontweight='bold')
            axes[1, 0].legend(fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Phase error
            phase_error = pred_phase - target_phase
            # Wrap phase error to [-pi, pi]
            phase_error = np.arctan2(np.sin(phase_error), np.cos(phase_error))
            axes[1, 1].plot(subcarrier_indices, phase_error, 'm-', linewidth=1.5, label='Error (Pred - Target)')
            axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Subcarrier Index', fontsize=11)
            axes[1, 1].set_ylabel('Phase Error (radians)', fontsize=11)
            axes[1, 1].set_title('Phase Prediction Error', fontsize=12, fontweight='bold')
            axes[1, 1].legend(fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add overall title with loss information
            fig.suptitle(f'CSI Loss Debug | Total Loss: {loss_value.item():.6f} | Sample: {sample_idx}',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'csi_comparison_{timestamp}_sample{sample_idx}.png'
            filepath = self.debug_dir / filename
            
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"Saved debug CSI plot: {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug CSI plot: {e}")
    
    # _complex_correlation method removed - was incorrectly implemented
    # (mixed all subcarriers and antennas together)
