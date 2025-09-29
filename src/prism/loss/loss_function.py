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
from .pas_loss import PASLoss
from .pas_loss2 import PAS2Loss

# Get logger for this module
logger = logging.getLogger(__name__)

# Default configuration for loss functions
DEFAULT_LOSS_CONFIG = {
    'csi_weight': 0.7,
    'pdp_weight': 0.3,
    'pas_weight': 0.1,
    'pas2_weight': 0.2,
    'regularization_weight': 0.01,
    'csi_loss': {
        'phase_weight': 1.0,
        'magnitude_weight': 1.0,
        'normalize_weights': True
    },
    'pdp_loss': {
        'loss_type': 'mse',  # 'mse', 'mae', 'cosine'
        'fft_size': 1024,
        'normalize_pdp': True
    },
    'pas_loss': {
        'enabled': False,
        'azimuth_divisions': 18,
        'elevation_divisions': 6,
        'normalize_pas': True,
        'type': 'mse',
        'weight_by_power': True
    },
    'pas_loss2': {
        'enabled': False,
        'freq_spatial_weight': 0.40,
        'phase_consistency_weight': 0.35,
        'angle_spectrum_weight': 0.25,
        'lambda_smooth': 0.1,
        'mu_eig': 0.1
    }
}


class LossFunction(nn.Module):
    """
    Main Loss Function Class for Prism Framework
    
    Combines CSI and PDP losses with configurable weights
    and provides a unified interface for training.
    """
    
    def __init__(self, config: Dict, full_config: Dict = None):
        """
        Initialize Prism loss function
        
        Args:
            config: Loss configuration dictionary containing loss parameters
            full_config: Full configuration dictionary (needed for PAS losses)
        """
        super(LossFunction, self).__init__()
        
        # Store full config for PAS losses
        self.full_config = full_config or config
        
        # Extract configuration with reasonable defaults for loss weights
        # These are algorithm parameters, not critical system config
        self.csi_weight = config.get('csi_weight', 0.7)
        self.pdp_weight = config.get('pdp_weight', 0.3)
        self.pas_weight = config.get('pas_weight', 0.1)
        self.pas2_weight = config.get('pas2_weight', 0.2)
        self.regularization_weight = config.get('regularization_weight', 0.01)
        
        # Initialize component losses with configuration
        csi_config = config.get('csi_loss', {})
        self.csi_enabled = csi_config.get('enabled', True)  # Default enabled for backward compatibility
        self.csi_loss = CSILoss(
            phase_weight=csi_config.get('phase_weight', 1.0),
            magnitude_weight=csi_config.get('magnitude_weight', 1.0),
            normalize_weights=csi_config.get('normalize_weights', True),
            max_magnitude=csi_config.get('max_magnitude', 100.0)
        )
        
        # Initialize PDP loss
        pdp_config = config.get('pdp_loss', {})
        self.pdp_enabled = pdp_config.get('enabled', True)  # Default enabled for backward compatibility
        
        # Extract PDP configuration with defaults
        pdp_fft_size = pdp_config.get('fft_size', 1024)
        pdp_normalize = pdp_config.get('normalize_pdp', True)
        
        # Warn about deprecated parameters
        if 'type' in pdp_config:
            logger.warning(f"PDP loss 'type' parameter is deprecated and will be ignored. PDP loss now only uses MSE.")
        if 'mse_weight' in pdp_config or 'delay_weight' in pdp_config:
            logger.warning(f"PDP loss weight parameters (mse_weight, delay_weight) are deprecated and will be ignored.")
        
        self.pdp_loss = PDPLoss(
            fft_size=pdp_fft_size,
            normalize_pdp=pdp_normalize,
            loss_type=pdp_config.get('loss_type', 'mse')
        )
        
        logger.info(f"PDP Loss configuration:")
        logger.info(f"  Enabled: {self.pdp_enabled}")
        logger.info(f"  Loss type: {pdp_config.get('loss_type', 'mse')}")
        logger.info(f"  FFT size: {pdp_fft_size}")
        logger.info(f"  Normalize PDP: {pdp_normalize}")
        
        # Initialize PAS loss
        pas_config = config.get('pas_loss', {})
        self.pas_enabled = pas_config.get('enabled', False)  # Default disabled
        
        if self.pas_enabled:
            # Extract required configs for PAS loss only when enabled
            bs_config = self.full_config.get('base_station', {})
            ue_config = self.full_config.get('user_equipment', {})
            
            if not bs_config:
                raise ValueError("Configuration must contain 'base_station' section for PAS loss")
            if not ue_config:
                raise ValueError("Configuration must contain 'user_equipment' section for PAS loss")
            
            # Extract debug directory from output configuration
            debug_dir = None
            if self.full_config and 'output' in self.full_config:
                output_config = self.full_config['output']
                if 'training' in output_config and 'debug_dir' in output_config['training']:
                    debug_dir = output_config['training']['debug_dir']
            
            self.pas_loss = PASLoss(
                bs_config=bs_config,
                ue_config=ue_config,
                azimuth_divisions=pas_config.get('azimuth_divisions', 18),
                elevation_divisions=pas_config.get('elevation_divisions', 6),
                normalize_pas=pas_config.get('normalize_pas', True),
                loss_type=pas_config.get('type', 'mse'),
                weight_by_power=pas_config.get('weight_by_power', True),
                debug_dir=debug_dir
            )
        else:
            self.pas_loss = None
        
        # Initialize PAS Loss 2 (Multi-subcarrier CSI Spatial-Frequency Loss)
        pas2_config = config.get('pas2_loss', {})
        self.pas2_enabled = pas2_config.get('enabled', False)  # Default disabled
        
        if self.pas2_enabled:
            # Extract required configs for PAS Loss 2 only when enabled
            bs_config = self.full_config.get('base_station', {})
            ue_config = self.full_config.get('user_equipment', {})
            
            if not bs_config:
                raise ValueError("Configuration must contain 'base_station' section for PAS Loss 2")
            if not ue_config:
                raise ValueError("Configuration must contain 'user_equipment' section for PAS Loss 2")
            
            # Extract antenna counts from configs
            num_bs_antennas = bs_config.get('num_antennas')
            num_ue_antennas = ue_config.get('num_ue_antennas')
            
            if num_bs_antennas is None:
                raise ValueError("bs_config must contain 'num_antennas'")
            if num_ue_antennas is None:
                raise ValueError("ue_config must contain 'num_ue_antennas'")
            
            self.pas2_loss = PAS2Loss(
                num_bs_antennas=num_bs_antennas,
                num_ue_antennas=num_ue_antennas,
                freq_spatial_weight=pas2_config.get('freq_spatial_weight', 0.40),
                phase_consistency_weight=pas2_config.get('phase_consistency_weight', 0.35),
                angle_spectrum_weight=pas2_config.get('angle_spectrum_weight', 0.25),
                lambda_smooth=pas2_config.get('lambda_smooth', 0.1),
                mu_eig=pas2_config.get('mu_eig', 0.1)
            )
            
            logger.info(f"✅ PAS Loss 2 configuration:")
            logger.info(f"  Enabled: {self.pas2_enabled}")
            logger.info(f"  Weight: {self.pas2_weight}")
            logger.info(f"  BS antennas: {num_bs_antennas}, UE antennas: {num_ue_antennas}")
            logger.info(f"  Loss weights: freq_spatial={pas2_config.get('freq_spatial_weight', 0.40):.2f}, phase={pas2_config.get('phase_consistency_weight', 0.35):.2f}, angle={pas2_config.get('angle_spectrum_weight', 0.25):.2f}")
        else:
            self.pas2_loss = None
        
        # Initialize regularization loss enabled flag
        reg_config = config.get('regularization_loss', {})
        self.regularization_enabled = reg_config.get('enabled', True)  # Default enabled for backward compatibility
        
        
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
                        - 'bs_antenna_indices': BS antenna indices [batch_size] (optional)
                        - 'ue_antenna_indices': UE antenna indices [batch_size] (optional)
                        - 'bs_positions': BS positions [batch_size, 3] (optional)
                        - 'ue_positions': UE positions [batch_size, 3] (optional)
            targets: Dictionary containing target values
                    - 'csi': Target CSI tensor
                    - 'bs_antenna_indices': BS antenna indices [batch_size] (optional)
                    - 'ue_antenna_indices': UE antenna indices [batch_size] (optional)
                    - 'bs_positions': BS positions [batch_size, 3] (optional)
                    - 'ue_positions': UE positions [batch_size, 3] (optional)
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
        
        # CSI loss (hybrid: CMSE + Magnitude + Phase)
        if ('csi' in predictions and 'csi' in targets and self.csi_enabled):
            # Use CSILoss class for comprehensive CSI loss calculation
            try:
                csi_loss_val = self.csi_loss(predictions, targets)
                weighted_csi_loss = self.csi_weight * csi_loss_val
                total_loss = total_loss + weighted_csi_loss
                loss_components['csi_loss'] = csi_loss_val.item()
                logger.info(f"🎯 Weighted CSI Loss: {weighted_csi_loss.item():.6f} (base: {csi_loss_val.item():.6f} × weight: {self.csi_weight})")
            except Exception as e:
                logger.error(f"❌ CSI loss computation failed: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Set CSI loss to a small value to prevent training crash
                csi_loss_val = torch.tensor(1e-6, device=predictions['csi'].device, requires_grad=True)
                weighted_csi_loss = self.csi_weight * csi_loss_val
                total_loss = total_loss + weighted_csi_loss
                loss_components['csi_loss'] = csi_loss_val.item()
        else:
            loss_components['csi_loss'] = 0.0
        
        # PDP loss (hybrid: MSE + Delay)
        if ('csi' in predictions and 'csi' in targets and 
            self.pdp_enabled and self.pdp_weight > 0):
            pdp_loss_val = self.pdp_loss(predictions, targets)
            weighted_pdp_loss = self.pdp_weight * pdp_loss_val
            total_loss = total_loss + weighted_pdp_loss
            loss_components['pdp_loss'] = pdp_loss_val.item()
            logger.info(f"🎯 Weighted PDP Loss: {weighted_pdp_loss.item():.6f} (base: {pdp_loss_val.item():.6f} × weight: {self.pdp_weight})")
        else:
            # PDP loss is disabled, set to 0
            loss_components['pdp_loss'] = 0.0
        
        # PAS loss (Power Angular Spectrum) - requires position information
        if ('csi' in predictions and 'csi' in targets and 
            'bs_positions' in predictions and 'ue_positions' in predictions and
            'bs_positions' in targets and 'ue_positions' in targets and
            self.pas_enabled and self.pas_weight > 0):
            try:
                pas_loss_val = self.pas_loss(predictions, targets)
                weighted_pas_loss = self.pas_weight * pas_loss_val
                total_loss = total_loss + weighted_pas_loss
                loss_components['pas_loss'] = pas_loss_val.item()
                logger.info(f"🎯 Weighted PAS Loss: {weighted_pas_loss.item():.6f} (base: {pas_loss_val.item():.6f} × weight: {self.pas_weight})")
            except Exception as e:
                logger.error(f"❌ PAS loss computation failed: {e}")
                logger.error(f"   Predictions keys: {list(predictions.keys())}")
                logger.error(f"   Targets keys: {list(targets.keys())}")
                if 'csi' in predictions:
                    logger.error(f"   Pred CSI shape: {predictions['csi'].shape}")
                if 'csi' in targets:
                    logger.error(f"   Target CSI shape: {targets['csi'].shape}")
                # Log full traceback for debugging
                import traceback
                logger.error(f"   Full traceback:\n{traceback.format_exc()}")
                loss_components['pas_loss'] = 0.0
        else:
            # PAS loss is disabled or missing position data, set to 0
            loss_components['pas_loss'] = 0.0
        
        # PAS Loss 2 (Multi-subcarrier CSI Spatial-Frequency Loss) - requires position and antenna information
        if self.pas2_enabled and self.pas2_weight > 0:
            # Check required data availability
            required_pred_keys = ['csi', 'bs_positions', 'ue_positions', 'bs_antenna_indices', 'ue_antenna_indices']
            required_target_keys = ['csi', 'bs_positions', 'ue_positions', 'bs_antenna_indices', 'ue_antenna_indices']
            
            missing_pred = [key for key in required_pred_keys if key not in predictions]
            missing_target = [key for key in required_target_keys if key not in targets]
            
            if missing_pred or missing_target:
                logger.warning(f"PAS2 loss skipped - missing data:")
                if missing_pred:
                    logger.warning(f"  Missing in predictions: {missing_pred}")
                if missing_target:
                    logger.warning(f"  Missing in targets: {missing_target}")
                logger.warning(f"  Available in predictions: {list(predictions.keys())}")
                logger.warning(f"  Available in targets: {list(targets.keys())}")
                loss_components['pas2_loss'] = 0.0
            else:
                try:
                    pas2_loss_val = self.pas2_loss(predictions, targets)
                    weighted_pas2_loss = self.pas2_weight * pas2_loss_val
                    total_loss = total_loss + weighted_pas2_loss
                    loss_components['pas2_loss'] = pas2_loss_val.item()
                    logger.info(f"🎯 Weighted PAS2 Loss: {weighted_pas2_loss.item():.6f} (base: {pas2_loss_val.item():.6f} × weight: {self.pas2_weight})")
                except Exception as e:
                    logger.warning(f"PAS2 loss computation failed: {e}")
                    import traceback
                    logger.warning(f"PAS2 loss traceback: {traceback.format_exc()}")
                    loss_components['pas2_loss'] = 0.0
        else:
            # PAS2 loss is disabled or missing required data, set to 0
            loss_components['pas2_loss'] = 0.0
        
        # Regularization losses
        if ('regularization' in predictions and 
            self.regularization_enabled and self.regularization_weight > 0):
            reg_loss = predictions['regularization']
            total_loss = total_loss + self.regularization_weight * reg_loss
            loss_components['regularization_loss'] = reg_loss.item()
        
        loss_components['total_loss'] = total_loss.item()
        self.loss_components = loss_components
        
        # Log total loss summary
        logger.info(f"🎯 Total Loss Summary: {total_loss.item():.6f}")
        logger.info(f"   CSI: {loss_components['csi_loss']:.6f}, PDP: {loss_components['pdp_loss']:.6f}, PAS: {loss_components['pas_loss']:.6f}, PAS2: {loss_components['pas2_loss']:.6f}")
        
        return total_loss, loss_components
    
    def get_loss_components(self) -> Dict[str, float]:
        """
        Get the most recent loss components
        
        Returns:
            loss_components: Dictionary of loss component values
        """
        return self.loss_components.copy()
    
