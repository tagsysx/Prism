"""
Prism Training Interface (Completely Rewritten)

This module provides a modernized training interface that integrates:
1. PrismNetwork for feature extraction from ray tracing
2. NNRayTracer for neural CSI prediction
3. Efficient subcarrier selection and CSI computation
4. Streamlined checkpoint management

This is a complete rewrite based on the new architecture - no legacy compatibility.
"""

import torch
import torch.nn as nn
import logging
import os
import json
import time
from typing import Dict, Optional, List, Tuple, Any, Union
import contextlib
from pathlib import Path
import numpy as np

from .networks.prism_network import PrismNetwork
from .tracers import LowRankRayTracer

logger = logging.getLogger(__name__)


# Custom exceptions for better error handling
class PrismTrainingError(Exception):
    """Base exception for Prism training interface errors."""
    pass


class ConfigurationError(PrismTrainingError):
    """Raised when configuration is invalid or missing."""
    pass


class RayTracingError(PrismTrainingError):
    """Raised when ray tracing operations fail."""
    pass


class PrismTrainingInterface(nn.Module):
    """
    Modernized training interface for Prism neural ray tracing system.
    
    Key Features:
    - Integration with neural network-based NNRayTracer
    - Two-stage architecture: PrismNetwork + TraceNetwork
    - Attention-based feature fusion
    - End-to-end learnable pipeline
    - Enhanced checkpoint system
    """
    
    def __init__(
        self,
        prism_network: PrismNetwork,
        config: Dict[str, Any],
        checkpoint_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.config = config
        self.prism_network = prism_network
        
        # Device setup
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Extract configuration sections
        self.system_config = config.get('system', {})
        self.training_config = config.get('training', {})
        
        self.ue_config = config.get('user_equipment', {})
        self.data_config = config.get('input', {})
        
        # LowRankTransformer configuration
        self.transformer_config = config.get('transformer', {})
        self.use_transformer_enhancement = self.transformer_config.get('use_enhancement', False)
        
        # CSI correction configuration
        calibration_config = self.data_config.get('calibration', {})
        self.reference_subcarrier_index = calibration_config.get('reference_subcarrier_index', 0)
        
        # Validate required configurations
        self._validate_config()
        
        # Initialize LowRankRayTracer for CSI prediction with configuration
        self.ray_tracer = LowRankRayTracer(prism_network=self.prism_network, config=self.config)
        
        # Move ray_tracer to device
        self.ray_tracer = self.ray_tracer.to(self.device)
        
        # Subcarrier configuration - we use base subcarriers from PrismNetwork
        # Note: virtual_subcarriers concept removed in new 2D architecture
        self.num_subcarriers = self.prism_network.num_subcarriers  # Base subcarriers per sample
        
        # UE antenna configuration - single antenna combinations processed per sample
        
        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('./checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.current_batch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # No subcarrier selection cache needed - we use all subcarriers
        
        logger.info(f"🚀 PrismTrainingInterface initialized")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - Subcarriers: {self.num_subcarriers} total (all used)")
        logger.info(f"   - Processing single antenna combinations per sample")
        logger.info(f"   - Checkpoint dir: {self.checkpoint_dir}")
    
    def _validate_config(self):
        """Validate required configuration parameters."""
        required_paths = [
            ('system', 'device'),
            # ue_antenna_count validation removed
        ]
        
        for path in required_paths:
            current = self.config
            try:
                for key in path:
                    current = current[key]
            except KeyError:
                raise ConfigurationError(f"Missing required configuration: {'.'.join(path)}")
        
        # Validate UE antenna count
        # ue_antenna_count validation removed - single antenna combinations processed per sample
        
        logger.debug("Configuration validation completed successfully")
    
    
    
    def forward(
        self, 
        ue_positions: torch.Tensor,     # [batch_size, 3]
        bs_positions: torch.Tensor,     # [batch_size, 3]
        bs_antenna_indices: torch.Tensor,  # [batch_size] - single BS antenna index per sample
        ue_antenna_indices: torch.Tensor,  # [batch_size] - single UE antenna index per sample
        return_intermediates: bool = False
    ) -> Dict[str, Union[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass for individual CSI samples.
        
        This method processes each sample individually where each sample represents
        a single antenna combination (one BS antenna + one UE antenna).
        
        Processing Pipeline:
        1. Sample-by-sample processing: PrismNetwork + RayTracing
        2. Direct CSI prediction for single antenna pair
        3. Return individual CSI predictions
        
        Args:
            ue_positions: UE positions tensor [batch_size, 3]
            bs_positions: BS positions tensor [batch_size, 3]
            bs_antenna_indices: BS antenna indices [batch_size] - single index per sample
            ue_antenna_indices: UE antenna indices [batch_size] - single index per sample
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Dictionary containing:
                - csi: [batch_size, num_subcarriers] - individual CSI predictions
                - subcarrier_selection: None (all subcarriers used, no selection)
                - intermediates: Ray tracing intermediates (if requested)
        """
        # Quick input validation
        if torch.isnan(ue_positions).any() or torch.isnan(bs_positions).any():
            raise ValueError("Input data contains NaN values")
        
        batch_size = ue_positions.shape[0]
        device = ue_positions.device
        
        # Performance timing
        start_time = time.time()
        logger.debug(f"Individual CSI sample forward pass: batch_size={batch_size}")
        
        # Initialize output containers
        batch_csi = []
        all_intermediates = [] if return_intermediates else None
        
        # Process each sample individually
        for sample_idx in range(batch_size):
            logger.debug(f"🔄 Processing sample {sample_idx + 1}/{batch_size}")
            
            # Extract single sample data
            single_ue_position = ue_positions[sample_idx]  # [3]
            single_bs_position = bs_positions[sample_idx]  # [3]
            single_bs_antenna_idx = bs_antenna_indices[sample_idx]  # scalar
            single_ue_antenna_idx = ue_antenna_indices[sample_idx]  # scalar
            
            # Process single sample
            sample_result = self._process_individual_csi_sample(
                ue_position=single_ue_position,
                bs_position=single_bs_position,
                bs_antenna_idx=single_bs_antenna_idx,
                ue_antenna_idx=single_ue_antenna_idx,
                return_intermediates=return_intermediates
                )
            
            batch_csi.append(sample_result['csi'])
            
            # Collect intermediates if requested
            if return_intermediates and 'intermediates' in sample_result:
                all_intermediates.append(sample_result['intermediates'])
            
            # Progress logging
            if batch_size > 4 and (sample_idx + 1) % max(1, batch_size // 4) == 0:
                logger.debug(f"✅ Completed {sample_idx + 1}/{batch_size} samples")
        
        # Combine results from all samples (CSI is already enhanced per sample)
        combined_csi = torch.stack(batch_csi, dim=0)  # [batch_size, num_subcarriers]
        
        # Note: CSI enhancement is now applied per sample in _process_individual_csi_sample
        logger.debug("CSI enhancement applied per sample during individual processing")
        
        # Prepare final outputs
        outputs = {
            'csi': combined_csi,  # Return enhanced CSI from individual sample processing
            'subcarrier_selection': None  # All subcarriers used
        }
        
        if return_intermediates and all_intermediates:
            outputs['intermediates'] = all_intermediates
        
        # Final performance logging
        total_time = time.time() - start_time
        logger.debug(f"🏁 Individual CSI forward pass completed in {total_time:.3f}s")
        logger.debug(f"   Output shape: {combined_csi.shape}")
        
        return outputs
    
    def _process_individual_csi_sample(
        self, 
        ue_position: torch.Tensor,
        bs_position: torch.Tensor,
        bs_antenna_idx: torch.Tensor,
        ue_antenna_idx: torch.Tensor,
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single individual CSI sample (one BS antenna + one UE antenna).
        
        Args:
            ue_position: UE position tensor of shape [3]
            bs_position: BS position tensor of shape [3]
            bs_antenna_idx: BS antenna index (scalar)
            ue_antenna_idx: UE antenna index (scalar)
            return_intermediates: Whether to return intermediate results
        
        Returns:
            Dictionary containing:
                - csi: [num_subcarriers] - individual CSI prediction
                - intermediates: Ray tracing intermediates (if requested)
        """
        device = next(self.prism_network.parameters()).device
        
        # Input validation
        if torch.isnan(ue_position).any() or torch.isnan(bs_position).any():
            raise ValueError("Input positions contain NaN values")
        
        # Prepare inputs for PrismNetwork
        ue_pos = ue_position.to(device)  # [3]
        bs_pos = bs_position.to(device)  # [3]
        
        # PrismNetwork forward pass
        prism_outputs = self.prism_network(
            bs_position=bs_pos, 
            ue_position=ue_pos, 
            bs_antenna_index=bs_antenna_idx.item(), 
            ue_antenna_index=ue_antenna_idx.item()
        )
        attenuation_vectors = prism_outputs['attenuation_vectors']  # (A*B, num_sampling_points, output_dim)
        radiation_vectors = prism_outputs['radiation_vectors']  # (A*B, num_sampling_points, output_dim)
        frequency_basis_vectors = prism_outputs['frequency_basis_vectors']  # (num_subcarriers, output_dim)
        
        # Ray tracing for single antenna pair
        ray_trace_results = self.ray_tracer.trace_rays(
            attenuation_vectors=attenuation_vectors,
            radiation_vectors=radiation_vectors,
            frequency_basis_vectors=frequency_basis_vectors
        )
        
        # Get CSI from ray tracing results
        csi = ray_trace_results['csi']  # [num_subcarriers]
        
        # Check for empty CSI results
        if csi.shape[0] == 0:
            logger.error(f"❌ Empty CSI from ray tracer: shape={csi.shape}")
            # Return zero CSI with expected subcarrier count
            num_subcarriers = self.prism_network.num_subcarriers
            zero_csi = torch.zeros(num_subcarriers, dtype=csi.dtype, device=csi.device)
            return {'csi': zero_csi}
        
        # Apply CSI enhancement for this specific BS-UE antenna pair
        enhanced_csi = self.prism_network.enhance_csi(csi, bs_antenna_idx.item(), ue_antenna_idx.item())
        
        # Note: No need to extract CSI for specific UE antenna since we process
        # individual antenna combinations directly (single sample = single antenna pair)
        # CSI already represents the prediction for the specific BS-UE antenna combination
        
        # Prepare outputs
        result = {'csi': enhanced_csi}
        
        if return_intermediates:
            result['intermediates'] = {
                'prism_outputs': prism_outputs,
                'ray_trace_results': ray_trace_results
            }
        
        return result
    
    def _csi_calibration(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Apply full CSI correction (both phase and amplitude normalization).

        This method applies both phase and amplitude correction by dividing all subcarriers
        by the reference subcarrier for each CSI independently. This aligns both phase and 
        amplitude to the reference subcarrier.

        Args:
            csi: CSI tensor [batch_size, num_subcarriers] - individual CSI samples
            
        Returns:
            corrected_csi: Corrected CSI tensor with same shape as input
        """
        # Ensure reference index is within bounds
        reference_idx = self.reference_subcarrier_index
        if reference_idx >= csi.shape[-1]:
            reference_idx = 0
        
        # Apply correction per CSI independently
        corrected_csi = csi.clone()
        batch_size, num_subcarriers = csi.shape
        
        for batch_idx in range(batch_size):
            # Get CSI for this sample
            sample_csi = csi[batch_idx, :]  # [num_subcarriers]
            
            reference_value = sample_csi[reference_idx]
            
            # Avoid division by zero
            if torch.abs(reference_value) < 1e-12:
                # If reference is too small, use identity correction
                corrected_csi[batch_idx, :] = sample_csi
            else:
                # Apply full correction: CSI / reference
                # This normalizes both phase and amplitude
                corrected_csi[batch_idx, :] = sample_csi / reference_value
        
        logger.debug(f"🔧 Applied full CSI calibration (phase and amplitude)")
        logger.debug(f"   Reference subcarrier index: {reference_idx}")
        logger.debug(f"   CSI shape: {csi.shape}")
        
        return corrected_csi
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], 
                    loss_function, validation_mode: bool = False) -> torch.Tensor:
        """
        Compute loss with consistent CSI calibration for both predictions and targets.
        
        This method ensures that both predicted and target CSI are calibrated using
        the same reference subcarrier before loss calculation, ensuring fair comparison.
        
        Args:
            predictions: Dictionary containing predicted values
                        - 'csi': Predicted CSI tensor [batch_size, num_subcarriers]
                        - 'bs_antenna_indices': BS antenna indices [batch_size]
                        - 'ue_antenna_indices': UE antenna indices [batch_size]
                        - 'bs_positions': BS positions [batch_size, 3]
                        - 'ue_positions': UE positions [batch_size, 3]
            targets: Dictionary containing target values
                    - 'csi': Target CSI tensor [batch_size, num_subcarriers]
                    - 'bs_antenna_indices': BS antenna indices [batch_size]
                    - 'ue_antenna_indices': UE antenna indices [batch_size]
                    - 'bs_positions': BS positions [batch_size, 3]
                    - 'ue_positions': UE positions [batch_size, 3]
            loss_function: Loss function to use for computation
            validation_mode: Whether this is validation (affects logging)
            
        Returns:
            loss: Computed loss value (scalar tensor)
        """
        # Apply consistent calibration to both predictions and targets CSI
        calibrated_predictions = self._csi_calibration(predictions['csi'])
        calibrated_targets = self._csi_calibration(targets['csi'])
        
        if not validation_mode:
            logger.debug("Applied consistent CSI calibration to predictions and targets")
        
        # Prepare data for loss function (update CSI with calibrated versions)
        pred_dict = predictions.copy()
        pred_dict['csi'] = calibrated_predictions
        
        target_dict = targets.copy()
        target_dict['csi'] = calibrated_targets
        
        # Compute loss using the loss function
        total_loss, loss_components = loss_function(pred_dict, target_dict)
        
        if not validation_mode:
            logger.debug(f"Loss computed: {total_loss.item():.6f}, components: {loss_components}")
        
        return total_loss
    
    def update_training_state(self, epoch: int, batch: int, loss: float):
        """Update training state for progress tracking."""
        self.current_epoch = epoch
        self.current_batch = batch
        
        # Update best loss
        if loss < self.best_loss:
            self.best_loss = loss
            logger.debug(f"🎯 New best loss: {self.best_loss:.6f}")
        
        # Add to history
        if hasattr(self, 'loss_history'):
            self.loss_history.append(loss)
        else:
            self.loss_history = [loss]

    def save_checkpoint(self, epoch: int, batch: int, optimizer_state: dict, scheduler_state: dict = None):
        """Save training checkpoint."""
        if not hasattr(self, 'checkpoint_dir') or self.checkpoint_dir is None:
            logger.warning("No checkpoint directory configured, skipping checkpoint save")
            return
            
        import os
        import torch
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'batch': batch,
            'best_loss': float(self.best_loss),
            'prism_network_state': self.prism_network.state_dict(),
            'optimizer_state': optimizer_state,
            'training_config': self.config,
            'model_info': {
                'num_subcarriers': self.num_subcarriers,
                # ue_antenna_count removed
            }
        }
        
        if scheduler_state is not None:
            checkpoint_data['scheduler_state'] = scheduler_state
            
        # Save checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}_batch_{batch}.pt')
        torch.save(checkpoint_data, checkpoint_path)
        
        # Also save as latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint_data, latest_path)
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str, load_training_state: bool = True, 
                       load_optimizer: bool = True, load_scheduler: bool = True) -> Dict[str, Any]:
        """Load training checkpoint."""
        import torch
        import os
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        logger.info(f"📂 Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint data
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        # Load PrismNetwork state
        if 'prism_network_state' in checkpoint_data:
            self.prism_network.load_state_dict(checkpoint_data['prism_network_state'])
            logger.info("✅ PrismNetwork state loaded")
        
        # Prepare return data
        loaded_states = {}
        
        # Load optimizer state if requested
        if load_optimizer and 'optimizer_state' in checkpoint_data:
            loaded_states['optimizer_state_dict'] = checkpoint_data['optimizer_state']
            logger.info("✅ Optimizer state prepared for loading")
        
        # Load scheduler state if requested
        if load_scheduler and 'scheduler_state' in checkpoint_data:
            loaded_states['scheduler_state_dict'] = checkpoint_data['scheduler_state']
            logger.info("✅ Scheduler state prepared for loading")
        
        # Load training state if requested
        if load_training_state:
            self.current_epoch = checkpoint_data.get('epoch', 0)
            self.current_batch = checkpoint_data.get('batch', 0)
            self.best_loss = checkpoint_data.get('best_loss', float('inf'))
            
            # Prepare checkpoint info for trainer
            loaded_states['checkpoint_info'] = {
                'epoch': self.current_epoch,
                'batch': self.current_batch,
                'best_loss': self.best_loss
            }
            logger.info(f"✅ Training state loaded: epoch={self.current_epoch}, batch={self.current_batch}, best_loss={self.best_loss:.6f}")
        
        logger.info("🎯 Checkpoint loading completed successfully")
        return loaded_states

    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        return {
            'training_state': {
                'current_epoch': self.current_epoch,
                'current_batch': self.current_batch,
                'best_loss': float(self.best_loss),
                'history_length': len(self.training_history)
            },
            'model_config': {
                'num_subcarriers': self.num_subcarriers,
                # ue_antenna_count removed,
                'subcarrier_usage': 'all_subcarriers'
            },
            'system_info': {
                'device': str(self.device),
                'checkpoint_dir': str(self.checkpoint_dir),
                'prism_network_type': type(self.prism_network).__name__
            },
            'ray_tracer_info': {
                'azimuth_divisions': self.ray_tracer.azimuth_divisions,
                'elevation_divisions': self.ray_tracer.elevation_divisions,
                'total_directions': self.ray_tracer.total_directions,
                'max_ray_length': self.ray_tracer.max_ray_length,
                'num_sampling_points': self.ray_tracer.num_sampling_points
            },
            'transformer_info': {
                'use_enhancement': False,
                'transformer_available': False,
                'transformer_config': None
            }
        }
    
    
    def _generate_uniform_directions(self) -> torch.Tensor:
        """
        Generate uniform ray directions (same as PrismNetwork).
        
        Returns:
            directions: (num_directions, 3) - Unit direction vectors
                       These are PURE direction vectors (unit length), NOT including BS position offset.
                       To get actual ray positions: P(t) = bs_position + direction * t
        """
        # Calculate angular resolutions
        azimuth_resolution = 2 * torch.pi / self.prism_network.azimuth_divisions  # 360° / azimuth_divisions
        elevation_resolution = torch.pi / 2 / self.prism_network.elevation_divisions
        
        # Create grid of angles
        i, j = torch.meshgrid(
            torch.arange(self.prism_network.azimuth_divisions, dtype=torch.float32),
            torch.arange(self.prism_network.elevation_divisions, dtype=torch.float32),
            indexing='xy'
        )
        phi = i * azimuth_resolution     # Azimuth: 0° to 360°
        theta = j * elevation_resolution  # Elevation: 0° to 90°
        
        # Convert spherical coordinates to Cartesian unit vectors
        x = torch.cos(theta) * torch.cos(phi)
        y = torch.cos(theta) * torch.sin(phi)
        z = torch.sin(theta)
        
        # Flatten and stack to create (num_directions, 3) tensor
        return torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)
    
    def _sample_positions_along_rays(self, 
                                   bs_position: torch.Tensor,
                                   directions: torch.Tensor,
                                   max_length: float,
                                   num_points: int) -> torch.Tensor:
        """
        Sample positions along rays (same as PrismNetwork).
        
        Args:
            bs_position: Base station position (3,)
            directions: Ray directions (num_directions, 3)
            max_length: Maximum ray length
            num_points: Number of sampling points per ray
            
        Returns:
            sampled_positions: (num_directions, num_points, 3)
        """
        device = bs_position.device
        
        # Generate t values
        t_values = torch.linspace(0, max_length, num_points, dtype=torch.float32, device=device)
        
        # Ray equation: P(t) = bs_position + direction * t
        # Broadcasting: [1,1,3] + [R,1,3] * [1,P,1] → [R,P,3]
        sampled_positions = bs_position.view(1, 1, 3) + directions.unsqueeze(1) * t_values.view(1, -1, 1)
        
        return sampled_positions
