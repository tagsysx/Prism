"""
Main Loss Function Class for Prism Framework

Combines CSI and PDP losses with configurable weights
and provides a unified interface for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

from .csi_loss import CSILoss
from .pdp_loss import PDPLoss
from .ss_loss import SSLoss

# Get logger for this module
logger = logging.getLogger(__name__)

# Default configuration for loss functions
DEFAULT_LOSS_CONFIG = {
    'csi_weight': 0.7,
    'pdp_weight': 0.3,
    'regularization_weight': 0.01,
    'csi_loss': {
        'phase_weight': 1.0,
        'magnitude_weight': 1.0,
        'normalize_weights': True
    },
    'pdp_loss': {
        'type': 'hybrid',  # 'mse', 'delay', 'hybrid' (correlation disabled)
        'fft_size': 1024,
        'normalize_pdp': True
        # mse_weight and delay_weight now hardcoded in hybrid loss (0.7, 0.3)
        # correlation_weight removed - correlation loss disabled
    }
}


class LossFunction(nn.Module):
    """
    Main Loss Function Class for Prism Framework
    
    Combines CSI and PDP losses with configurable weights
    and provides a unified interface for training.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Prism loss function
        
        Args:
            config: Configuration dictionary containing loss parameters
        """
        super(LossFunction, self).__init__()
        
        # Extract configuration with reasonable defaults for loss weights
        # These are algorithm parameters, not critical system config
        self.csi_weight = config.get('csi_weight', 0.7)
        self.pdp_weight = config.get('pdp_weight', 0.3)
        self.spatial_spectrum_weight = config.get('spatial_spectrum_weight', 0.0)
        self.regularization_weight = config.get('regularization_weight', 0.01)
        
        # Initialize component losses with configuration
        csi_config = config.get('csi_loss', {})
        self.csi_enabled = csi_config.get('enabled', True)  # Default enabled for backward compatibility
        self.csi_loss = CSILoss(
            phase_weight=csi_config.get('phase_weight', 1.0),
            magnitude_weight=csi_config.get('magnitude_weight', 1.0),
            normalize_weights=csi_config.get('normalize_weights', True)
        )
        
        # Initialize PDP loss
        pdp_config = config.get('pdp_loss', {})
        self.pdp_enabled = pdp_config.get('enabled', True)  # Default enabled for backward compatibility
        self.pdp_loss = PDPLoss(
            loss_type=pdp_config.get('type', 'hybrid'),
            fft_size=pdp_config.get('fft_size', 1024),
            normalize_pdp=pdp_config.get('normalize_pdp', True)
            # mse_weight, correlation_weight, delay_weight parameters removed
        )
        
        # Initialize regularization loss enabled flag
        reg_config = config.get('regularization_loss', {})
        self.regularization_enabled = reg_config.get('enabled', True)  # Default enabled for backward compatibility
        
        # Initialize Spatial Spectrum loss (only if enabled and weight > 0)
        ssl_config = config.get('spatial_spectrum_loss', {})
        self.spatial_spectrum_enabled = ssl_config.get('enabled', False)
        self.ss_loss = None
        if self.spatial_spectrum_weight > 0 and self.spatial_spectrum_enabled:
            # Pass the full config to SSLoss (it needs base_station, user_equipment and training sections)
            full_config = {
                'base_station': config.get('base_station', {}), 
                'user_equipment': config.get('user_equipment', {}),
                'training': {'loss': {'spatial_spectrum_loss': ssl_config}}
            }
            self.ss_loss = SSLoss(full_config)
        
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
            targets: Dictionary containing target values
                    - 'csi': Target CSI tensor
            masks: Optional masks for selective loss computation
        
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary of individual loss components
        """
        # Initialize total_loss properly to maintain gradients
        # Use a tensor derived from predictions to ensure gradient flow, but ensure it's real
        total_loss = torch.real(predictions['csi'].sum()) * 0.0  # This maintains gradient connection to predictions
        loss_components = {}
        
        if masks is None:
            masks = {}
        
        # CSI loss (hybrid: CMSE + Magnitude + Phase) - use original 4D CSI tensors
        if ('csi' in predictions and 'csi' in targets and self.csi_enabled):
            # Use original 4D CSI tensors to preserve spatial structure
            csi_pred = predictions['csi']
            csi_target = targets['csi']
            
            # Use CSILoss class for comprehensive CSI loss calculation
            try:
                csi_loss_val = self.csi_loss(csi_pred, csi_target)
                total_loss = total_loss + self.csi_weight * csi_loss_val
                loss_components['csi_loss'] = csi_loss_val.item()
            except Exception as e:
                logger.error(f"❌ CSI loss computation failed: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Set CSI loss to a small value to prevent training crash
                csi_loss_val = torch.tensor(1e-6, device=csi_pred.device, requires_grad=True)
                total_loss = total_loss + self.csi_weight * csi_loss_val
                loss_components['csi_loss'] = csi_loss_val.item()
        else:
            loss_components['csi_loss'] = 0.0
        
        # PDP loss (hybrid: MSE + Delay) - use original 4D CSI tensors for frequency domain analysis
        if ('csi' in predictions and 'csi' in targets and 
            self.pdp_enabled and self.pdp_weight > 0):
            pdp_loss_val = self.pdp_loss(
                predictions['csi'], 
                targets['csi']
            )
            total_loss = total_loss + self.pdp_weight * pdp_loss_val
            loss_components['pdp_loss'] = pdp_loss_val.item()
        else:
            # PDP loss is disabled, set to 0
            loss_components['pdp_loss'] = 0.0
        
        # Spatial Spectrum loss
        if (self.ss_loss is not None and 
            self.spatial_spectrum_enabled and
            'csi' in predictions and 'csi' in targets and 
            self.spatial_spectrum_weight > 0):
            
            # SSLoss will handle CSI format conversion internally
            logger.info(f"🔍 Computing spatial spectrum loss with CSI shape: {predictions['csi'].shape}")
            spatial_loss_val = self.ss_loss(predictions['csi'], targets['csi'])
            total_loss = total_loss + self.spatial_spectrum_weight * spatial_loss_val
            loss_components['ss_loss'] = spatial_loss_val.item()
        
        # Regularization losses
        if ('regularization' in predictions and 
            self.regularization_enabled and self.regularization_weight > 0):
            reg_loss = predictions['regularization']
            total_loss = total_loss + self.regularization_weight * reg_loss
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
    
    def compute_and_visualize_ss_loss(self, predicted_csi: torch.Tensor, 
                                      target_csi: torch.Tensor,
                                      save_path: str, sample_idx: int = 0) -> Optional[Tuple[float, str]]:
        """
        Compute spatial spectrum loss and create visualization (for testing)
        
        Args:
            predicted_csi: Predicted CSI tensor
            target_csi: Target CSI tensor  
            save_path: Directory to save visualization
            sample_idx: Sample index to visualize
            
        Returns:
            (loss_value, plot_path) if spatial spectrum loss is enabled, None otherwise
        """
        if self.ss_loss is None:
            return None
            
        return self.ss_loss.compute_and_visualize_loss(
            predicted_csi, target_csi, save_path, sample_idx
        )
