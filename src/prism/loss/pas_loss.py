"""
PAS (Power Angular Spectrum) Loss Implementation

This module implements loss functions for Power Angular Spectrum (PAS) comparison
between predicted and target CSI data. The PAS represents the spatial distribution
of signal power across different angles of arrival/departure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from prism.utils.pas_utils import reorganize_data_as_mimo

logger = logging.getLogger(__name__)


class PASLoss(nn.Module):
    """
    Power Angular Spectrum (PAS) Loss for spatial spectrum comparison.
    
    This loss computes the difference between predicted and target PAS,
    which represents the spatial distribution of signal power. The loss combines
    traditional distance metrics (MSE, MAE, KL-div, JS-div) with cosine similarity
    to measure both magnitude and pattern similarity.
    
     Features:
     - Multiple loss types: MSE, MAE, KL divergence, Jensen-Shannon divergence, Cosine similarity
     - Cosine similarity loss for pattern matching (focuses on shape, range [0,1])
     - Power-based weighting for emphasizing high-power samples
     - Debug visualization with configurable save directory
    
    Args:
        bs_config: Base station configuration dictionary (must contain 'num_antennas')
        ue_config: User equipment configuration dictionary (must contain 'num_ue_antennas')
        azimuth_divisions: Number of azimuth angle divisions (default: 18)
        elevation_divisions: Number of elevation angle divisions (default: 6)
        normalize_pas: Whether to normalize PAS before loss computation (default: True)
        loss_type: Type of loss ('mse', 'mae', 'kl_div', 'js_div', 'cosine') (default: 'mse')
        weight_by_power: Whether to weight loss by power distribution (default: True)
        debug_dir: Directory path for saving debug spatial spectrum files (default: None)
    """
    
    def __init__(
        self,
        bs_config: dict,
        ue_config: dict,
        azimuth_divisions: int = 360,
        elevation_divisions: int = 90,
        normalize_pas: bool = True,
        loss_type: str = 'mse',
        weight_by_power: bool = True,
        center_freq: float = 3.5e9,
        subcarrier_spacing: float = 245.1e3,
        debug_dir: str = None,
        debug_sample_rate: float = 0.5
    ):
        super().__init__()
        
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.normalize_pas = normalize_pas
        self.loss_type = loss_type.lower()
        self.weight_by_power = weight_by_power
        self.center_freq = center_freq
        self.subcarrier_spacing = subcarrier_spacing
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.debug_sample_rate = debug_sample_rate
        
        # Validate required configs
        if not bs_config:
            raise ValueError("bs_config is required and cannot be empty")
        if not ue_config:
            raise ValueError("ue_config is required and cannot be empty")
        
        # Extract antenna information from configs
        self.num_bs_antennas = bs_config.get('num_antennas')
        self.num_ue_antennas = ue_config.get('num_ue_antennas')
        
        # Validate antenna counts
        if self.num_bs_antennas is None:
            raise ValueError("bs_config must contain 'num_antennas'")
        if self.num_ue_antennas is None:
            raise ValueError("ue_config must contain 'num_ue_antennas'")
        if self.num_bs_antennas <= 0:
            raise ValueError(f"num_bs_antennas must be positive, got {self.num_bs_antennas}")
        if self.num_ue_antennas <= 0:
            raise ValueError(f"num_ue_antennas must be positive, got {self.num_ue_antennas}")
        
        # Store configs for potential future use
        self.bs_config = bs_config
        self.ue_config = ue_config
        
        # Validate loss type
        valid_types = ['mse', 'mae', 'kl_div', 'js_div', 'cosine']
        if self.loss_type not in valid_types:
            raise ValueError(f"Invalid loss_type: {self.loss_type}. Must be one of {valid_types}")
        
        # Create debug directory if specified
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PASLoss initialized:")
        logger.info(f"  Azimuth divisions: {azimuth_divisions}")
        logger.info(f"  Elevation divisions: {elevation_divisions}")
        logger.info(f"  Normalize PAS: {normalize_pas}")
        logger.info(f"  Loss type: {loss_type}")
        logger.info(f"  Weight by power: {weight_by_power}")
        logger.info(f"  Center frequency: {center_freq/1e9:.2f} GHz")
        logger.info(f"  Subcarrier spacing: {subcarrier_spacing/1e3:.1f} kHz")
        logger.info(f"  Number of BS antennas: {self.num_bs_antennas}")
        logger.info(f"  Number of UE antennas: {self.num_ue_antennas}")
        if self.debug_dir:
            logger.info(f"  Debug plots will be saved to: {self.debug_dir}")
            logger.info(f"  Debug sample rate: {self.debug_sample_rate*100:.1f}%")
    
    def compute_loss(self, pred_pas: torch.Tensor, target_pas: torch.Tensor, total_antennas: int = 1) -> torch.Tensor:
        """
        Compute loss between predicted and target PAS.
        
        Args:
            pred_pas: Predicted PAS [batch_size, azimuth_divisions, elevation_divisions]
                     where batch_size now includes all individual antenna PAS
            target_pas: Target PAS [batch_size, azimuth_divisions, elevation_divisions]
                       where batch_size now includes all individual antenna PAS
            total_antennas: Total number of antennas (for normalization)
            
        Returns:
            Loss scalar
        """
        # Compute power weights first if needed
        if self.weight_by_power:
            target_power = torch.sum(target_pas, dim=(1, 2))  # [batch_size * total_antennas]
            power_weights = target_power / (torch.mean(target_power) + 1e-8)  # [batch_size * total_antennas]
        else:
            power_weights = None
        
        if self.loss_type == 'mse':
            if power_weights is not None:
                # Compute per-sample MSE loss and apply weights
                sample_losses = F.mse_loss(pred_pas, target_pas, reduction='none')  # [batch_size * total_antennas, azimuth_div, elevation_div]
                sample_losses = torch.mean(sample_losses, dim=(1, 2))  # [batch_size * total_antennas]
                weighted_losses = sample_losses * power_weights  # [batch_size * total_antennas]
                loss = torch.mean(weighted_losses)  # Scalar
            else:
                loss = F.mse_loss(pred_pas, target_pas)
            
        elif self.loss_type == 'mae':
            if power_weights is not None:
                # Compute per-sample MAE loss and apply weights
                sample_losses = F.l1_loss(pred_pas, target_pas, reduction='none')  # [batch_size * total_antennas, azimuth_div, elevation_div]
                sample_losses = torch.mean(sample_losses, dim=(1, 2))  # [batch_size * total_antennas]
                weighted_losses = sample_losses * power_weights  # [batch_size * total_antennas]
                loss = torch.mean(weighted_losses)  # Scalar
            else:
                loss = F.l1_loss(pred_pas, target_pas)
            
        elif self.loss_type == 'kl_div':
            # KL divergence (treat PAS as probability distributions)
            pred_pas_flat = pred_pas.view(pred_pas.shape[0], -1)
            target_pas_flat = target_pas.view(target_pas.shape[0], -1)
            
            # Add small epsilon for numerical stability
            pred_pas_flat = pred_pas_flat + 1e-8
            target_pas_flat = target_pas_flat + 1e-8
            
            # Normalize to probability distributions
            pred_pas_flat = pred_pas_flat / torch.sum(pred_pas_flat, dim=-1, keepdim=True)
            target_pas_flat = target_pas_flat / torch.sum(target_pas_flat, dim=-1, keepdim=True)
            
            loss = F.kl_div(torch.log(pred_pas_flat), target_pas_flat, reduction='batchmean')
            
            # Apply average weight for KL divergence
            if power_weights is not None:
                loss = loss * torch.mean(power_weights)
            
        elif self.loss_type == 'js_div':
            # Jensen-Shannon divergence
            pred_pas_flat = pred_pas.view(pred_pas.shape[0], -1)
            target_pas_flat = target_pas.view(target_pas.shape[0], -1)
            
            # Add small epsilon for numerical stability
            pred_pas_flat = pred_pas_flat + 1e-8
            target_pas_flat = target_pas_flat + 1e-8
            
            # Normalize to probability distributions
            pred_pas_flat = pred_pas_flat / torch.sum(pred_pas_flat, dim=-1, keepdim=True)
            target_pas_flat = target_pas_flat / torch.sum(target_pas_flat, dim=-1, keepdim=True)
            
            # Compute M = (P + Q) / 2
            m = (pred_pas_flat + target_pas_flat) / 2
            
            # JS divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            kl_pm = F.kl_div(torch.log(pred_pas_flat), m, reduction='batchmean')
            kl_qm = F.kl_div(torch.log(target_pas_flat), m, reduction='batchmean')
            loss = 0.5 * kl_pm + 0.5 * kl_qm
            
            # Apply average weight for JS divergence
            if power_weights is not None:
                loss = loss * torch.mean(power_weights)
            
        elif self.loss_type == 'cosine':
            # Cosine similarity loss for pattern matching
            # Flatten PAS patterns for cosine similarity computation
            pred_flat = pred_pas.view(pred_pas.shape[0], -1)  # [batch_size, azimuth_div * elevation_div]
            target_flat = target_pas.view(target_pas.shape[0], -1)  # [batch_size, azimuth_div * elevation_div]
            
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            
            # Compute cosine similarity for each sample
            # cosine_sim = (A · B) / (||A|| * ||B||)
            dot_product = torch.sum(pred_flat * target_flat, dim=1)  # [batch_size]
            pred_norm = torch.norm(pred_flat, dim=1) + eps  # [batch_size]
            target_norm = torch.norm(target_flat, dim=1) + eps  # [batch_size]
            
            cosine_similarity = dot_product / (pred_norm * target_norm)  # [batch_size]
            
            # Convert to loss: (1 - (1 + cosine) / 2) (range [0, 1], 0 = perfect similarity)
            # cosine=1 → loss=0 (best), cosine=0 → loss=0.5 (medium), cosine=-1 → loss=1 (worst)
            cosine_loss_per_sample = 1.0 - (1.0 + cosine_similarity) / 2.0  # [batch_size]
            
            loss = torch.mean(cosine_loss_per_sample)
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        combined_loss = loss
        
        # Debug: randomly save PAS comparison plots
        # Only save debug plots when we have meaningful spatial spectrum
        if (self.debug_dir and random.random() < self.debug_sample_rate and 
            (self.num_bs_antennas > 1 or self.num_ue_antennas > 1)):
            self._save_debug_plot(pred_pas, target_pas, loss, self.loss_type)
        
        # Normalize by total number of antennas to prevent loss scaling with antenna count
        # This ensures that the loss magnitude is comparable regardless of antenna configuration
        if total_antennas > 1:
            # Don't normalize if we only have 1 antenna total
            normalization_factor = 1.0 / total_antennas
            combined_loss = combined_loss * normalization_factor
            
        return combined_loss
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of PAS loss.
        
        Args:
            predictions: Dictionary containing predicted values
                        - 'csi': Predicted CSI tensor [batch_size, num_subcarriers] (complex)
                        - 'bs_positions': BS positions [batch_size, 3] (required)
                        - 'ue_positions': UE positions [batch_size, 3] (required)
                        - 'bs_antenna_indices': BS antenna indices [batch_size] (required)
                        - 'ue_antenna_indices': UE antenna indices [batch_size] (required)
            targets: Dictionary containing target values (same structure as predictions)
            
        Returns:
            loss: Computed PAS loss value (scalar tensor)
        """
        # Extract data from dictionaries
        pred_csi = predictions['csi']
        target_csi = targets['csi']
        bs_positions = predictions['bs_positions']
        ue_positions = predictions['ue_positions']
        
        # Extract antenna indices (now required for proper PAS computation)
        bs_antenna_indices = predictions.get('bs_antenna_indices', None)
        ue_antenna_indices = predictions.get('ue_antenna_indices', None)
        
        if bs_antenna_indices is None:
            raise ValueError("bs_antenna_indices is required for PAS loss computation")
        if ue_antenna_indices is None:
            raise ValueError("ue_antenna_indices is required for PAS loss computation")
        
        # Reorganize data for phase array processing
        
        # Process predicted CSI
        pred_csi_by_pos, unique_positions = reorganize_data_as_mimo(
            pred_csi, bs_positions, ue_positions, bs_antenna_indices, ue_antenna_indices,
            self.num_bs_antennas, self.num_ue_antennas
        )
        
        # Process target CSI (should have same unique positions)
        target_csi_by_pos, _ = reorganize_data_as_mimo(
            target_csi, bs_positions, ue_positions, bs_antenna_indices, ue_antenna_indices,
            self.num_bs_antennas, self.num_ue_antennas
        )
        
        # Compute spatial spectrum for each position pair
        pred_pas_list = []
        target_pas_list = []
        
        for pos_pair_idx, (bs_pos, ue_pos) in enumerate(unique_positions):
            # Get CSI data for this position pair
            pos_pred_csi = pred_csi_by_pos[pos_pair_idx]  # [num_bs_antennas, num_ue_antennas, num_subcarriers]
            pos_target_csi = target_csi_by_pos[pos_pair_idx]  # [num_bs_antennas, num_ue_antennas, num_subcarriers]
            
            # Compute spatial spectrum for this position pair
            pred_pas = self._compute_spatial_spectrum_for_position_pair(pos_pred_csi, bs_pos, ue_pos)
            target_pas = self._compute_spatial_spectrum_for_position_pair(pos_target_csi, bs_pos, ue_pos)
            
            pred_pas_list.append(pred_pas)
            target_pas_list.append(target_pas)
        
        # Stack PAS results
        pred_pas_batch = torch.stack(pred_pas_list, dim=0)  # [num_position_pairs, total_antennas, azimuth_divisions, elevation_divisions]
        target_pas_batch = torch.stack(target_pas_list, dim=0)  # [num_position_pairs, total_antennas, azimuth_divisions, elevation_divisions]
        
        # Reshape to treat each antenna's PAS as a separate sample for loss computation
        # This allows computing loss for each individual PAS
        batch_size, total_antennas, azimuth_div, elevation_div = pred_pas_batch.shape
        pred_pas_batch = pred_pas_batch.view(batch_size * total_antennas, azimuth_div, elevation_div)
        target_pas_batch = target_pas_batch.view(batch_size * total_antennas, azimuth_div, elevation_div)
        
        # Compute loss with normalization factor
        # Split BS and UE PAS for separate processing to avoid mixing zero and non-zero samples
        pas_loss = self._compute_split_pas_loss(pred_pas_batch, target_pas_batch, total_antennas)
        
        return pas_loss
    
    def _compute_split_pas_loss(self, pred_pas_batch: torch.Tensor, target_pas_batch: torch.Tensor, total_antennas: int) -> torch.Tensor:
        """
        Compute PAS loss by processing BS and UE PAS separately.
        
        This avoids the issue where single-antenna BS generates all-zero PAS samples
        that get mixed with meaningful UE PAS samples.
        
        Args:
            pred_pas_batch: [batch_size * total_antennas, azimuth_div, elevation_div]
            target_pas_batch: [batch_size * total_antennas, azimuth_div, elevation_div]
            total_antennas: Total number of antenna samples (num_ue_antennas + num_bs_antennas)
            
        Returns:
            Combined PAS loss
        """
        # Split the batch back into BS and UE components
        batch_size = pred_pas_batch.shape[0] // total_antennas
        azimuth_div, elevation_div = pred_pas_batch.shape[1], pred_pas_batch.shape[2]
        
        # Reshape back to [batch_size, total_antennas, azimuth_div, elevation_div]
        pred_pas_batch = pred_pas_batch.view(batch_size, total_antennas, azimuth_div, elevation_div)
        target_pas_batch = target_pas_batch.view(batch_size, total_antennas, azimuth_div, elevation_div)
        
        # Split BS and UE PAS based on antenna counts
        # BS PAS: first num_ue_antennas samples (from BS perspective)
        # UE PAS: last num_bs_antennas samples (from UE perspective)
        bs_pas_pred = pred_pas_batch[:, :self.num_ue_antennas, :, :]  # [batch_size, num_ue_antennas, azimuth_div, elevation_div]
        bs_pas_target = target_pas_batch[:, :self.num_ue_antennas, :, :]
        
        ue_pas_pred = pred_pas_batch[:, self.num_ue_antennas:, :, :]  # [batch_size, num_bs_antennas, azimuth_div, elevation_div]
        ue_pas_target = target_pas_batch[:, self.num_ue_antennas:, :, :]
        
        total_loss = torch.tensor(0.0, device=pred_pas_batch.device, requires_grad=True)
        bs_loss = None  # Initialize for later use
        
        # Process BS PAS (only if BS has multiple antennas)
        if self.num_bs_antennas > 1:
            bs_pas_pred_flat = bs_pas_pred.view(-1, azimuth_div, elevation_div)
            bs_pas_target_flat = bs_pas_target.view(-1, azimuth_div, elevation_div)
            bs_loss = self.compute_loss(bs_pas_pred_flat, bs_pas_target_flat, self.num_bs_antennas)
            total_loss = total_loss + bs_loss
            logger.debug(f"BS PAS loss: {bs_loss.item():.6f} (BS antennas: {self.num_bs_antennas})")
        else:
            logger.debug(f"Skipping BS PAS loss (single antenna: {self.num_bs_antennas})")
        
        # Process UE PAS (only if UE has multiple antennas)  
        if self.num_ue_antennas > 1:
            ue_pas_pred_flat = ue_pas_pred.view(-1, azimuth_div, elevation_div)
            ue_pas_target_flat = ue_pas_target.view(-1, azimuth_div, elevation_div)
            ue_loss = self.compute_loss(ue_pas_pred_flat, ue_pas_target_flat, self.num_ue_antennas)
            total_loss = total_loss + ue_loss
            logger.debug(f"UE PAS loss: {ue_loss.item():.6f} (UE antennas: {self.num_ue_antennas})")
        else:
            logger.debug(f"Skipping UE PAS loss (single antenna: {self.num_ue_antennas})")
        
        # Also save debug plot for BS PAS if it has multiple antennas
        if (self.num_bs_antennas > 1 and bs_loss is not None):
            # Debug plot for BS PAS is handled in compute_loss method
            pass
        
        return total_loss
    
    
    def _compute_spatial_spectrum_for_position_pair(self, csi_matrix: torch.Tensor, bs_pos: torch.Tensor, ue_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial spectrum for a single position pair using mimo_to_pas function.
        
        Args:
            csi_matrix: [num_bs_antennas, num_ue_antennas, num_subcarriers]
            bs_pos: [3] BS position
            ue_pos: [3] UE position
            
        Returns:
            all_pas: [total_antennas, azimuth_divisions, elevation_divisions] 
                    where total_antennas = num_ue_antennas + num_bs_antennas
        """
        from prism.utils.pas_utils import mimo_to_pas
        
        # Parse array shapes from configurations
        bs_array_config = self.bs_config.get('array_configuration', f'{self.num_bs_antennas}x1')
        ue_array_config = self.ue_config.get('array_configuration', f'{self.num_ue_antennas}x1')
        
        from prism.utils.pas_utils import parse_array_configuration
        bs_array_shape = parse_array_configuration(bs_array_config)
        ue_array_shape = parse_array_configuration(ue_array_config)
        
        # Use the common mimo_to_pas function
        pas_dict = mimo_to_pas(
            csi_matrix=csi_matrix,
            bs_array_shape=bs_array_shape,
            ue_array_shape=ue_array_shape,
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions,
            normalize_pas=self.normalize_pas,
            center_freq=self.center_freq,
            subcarrier_spacing=self.subcarrier_spacing
        )
        
        # Combine BS and UE PAS 
        # bs_pas: [num_ue_antennas, azimuth_divisions, elevation_divisions]
        # ue_pas: [num_bs_antennas, azimuth_divisions, elevation_divisions]
        bs_pas = pas_dict["bs"]  # [num_ue_antennas, azimuth_divisions, elevation_divisions]
        ue_pas = pas_dict["ue"]  # [num_bs_antennas, azimuth_divisions, elevation_divisions]
        
        # Return all PAS values instead of averaging
        # Concatenate BS and UE PAS along the first dimension to get all individual PAS
        # Total antennas = num_ue_antennas (from BS perspective) + num_bs_antennas (from UE perspective)
        all_pas = torch.cat([bs_pas, ue_pas], dim=0)  # [total_antennas, azimuth_divisions, elevation_divisions]
        
        return all_pas
    
    def _has_meaningful_pas_content(self, pred_pas: torch.Tensor, target_pas: torch.Tensor) -> bool:
        """
        Check if PAS batch contains meaningful (non-zero) spatial spectrum samples.
        
        For configurations like PolyU where BS has single antenna, most PAS samples
        will be zero. This method ensures we only save debug plots when there are
        sufficient non-zero samples worth visualizing.
        
        Args:
            pred_pas: Predicted PAS [batch_size, azimuth_divisions, elevation_divisions]
            target_pas: Target PAS (same shape as pred_pas)
            
        Returns:
            bool: True if batch contains meaningful PAS samples, False otherwise
        """
        try:
            batch_size = pred_pas.shape[0]
            
            # Count non-zero samples
            pred_nonzero_samples = 0
            target_nonzero_samples = 0
            
            for i in range(batch_size):
                # Check if this sample has any significant values
                pred_sample_energy = torch.sum(torch.abs(pred_pas[i])).item()
                target_sample_energy = torch.sum(torch.abs(target_pas[i])).item()
                
                if pred_sample_energy > 1e-8:
                    pred_nonzero_samples += 1
                if target_sample_energy > 1e-8:
                    target_nonzero_samples += 1
            
            # Calculate ratios
            pred_nonzero_ratio = pred_nonzero_samples / batch_size
            target_nonzero_ratio = target_nonzero_samples / batch_size
            
            # We consider the batch meaningful if at least 20% of samples are non-zero
            # This threshold helps avoid the PolyU case where 88.9% samples are zero
            meaningful_threshold = 0.2
            
            is_meaningful = (pred_nonzero_ratio >= meaningful_threshold or 
                           target_nonzero_ratio >= meaningful_threshold)
            
            if not is_meaningful:
                logger.debug(f"Skipping PAS debug plot - insufficient meaningful samples "
                           f"(pred: {pred_nonzero_samples}/{batch_size} = {pred_nonzero_ratio:.1%}, "
                           f"target: {target_nonzero_samples}/{batch_size} = {target_nonzero_ratio:.1%})")
            
            return is_meaningful
            
        except Exception as e:
            logger.warning(f"Error checking PAS content meaningfulness: {e}, defaulting to save")
            return True  # Default to saving if check fails
    
    def _save_debug_plot(self, pred_pas: torch.Tensor, target_pas: torch.Tensor, loss_value: torch.Tensor, loss_type: str):
        """
        Save debug plot comparing predicted and target PAS
        
        Args:
            pred_pas: Predicted PAS [batch_size, azimuth_divisions, elevation_divisions] or [batch_size, azimuth_divisions]
            target_pas: Target PAS (same shape as pred_pas)
            loss_value: Computed loss value (scalar tensor)
        """
        try:
            # Intelligently select a sample from the batch
            # Prefer non-zero samples for more meaningful visualization
            batch_size = pred_pas.shape[0]
            
            # Find non-zero samples
            non_zero_indices = []
            for i in range(batch_size):
                pred_energy = torch.sum(torch.abs(pred_pas[i])).item()
                target_energy = torch.sum(torch.abs(target_pas[i])).item()
                if pred_energy > 1e-8 or target_energy > 1e-8:
                    non_zero_indices.append(i)
            
            # Select sample index
            if non_zero_indices:
                # Prefer non-zero samples
                sample_idx = random.choice(non_zero_indices)
                logger.debug(f"Selected non-zero PAS sample {sample_idx} from {len(non_zero_indices)} non-zero samples")
            else:
                # Fallback to random selection if all samples are zero
                sample_idx = random.randint(0, batch_size - 1)
                logger.debug(f"All PAS samples are zero, selected random sample {sample_idx}")
            
            # Get PAS for selected sample
            pred_spectrum = pred_pas[sample_idx].detach().cpu().numpy()
            target_spectrum = target_pas[sample_idx].detach().cpu().numpy()
            
            # Compute metrics
            mse = ((pred_spectrum - target_spectrum) ** 2).mean()
            mae = np.abs(pred_spectrum - target_spectrum).mean()
            
            # Cosine similarity
            pred_flat = pred_spectrum.flatten()
            target_flat = target_spectrum.flatten()
            dot_product = (pred_flat * target_flat).sum()
            pred_norm = (pred_flat ** 2).sum() ** 0.5
            target_norm = (target_flat ** 2).sum() ** 0.5
            cosine_sim = dot_product / (pred_norm * target_norm + 1e-8)
            cosine_loss = 1.0 - (1.0 + cosine_sim) / 2.0
            
            # Determine actual loss value based on loss type
            if loss_type == 'mse':
                actual_loss_value = mse
                loss_info = f"MSE Loss: {mse:.6f}"
            elif loss_type == 'mae':
                actual_loss_value = mae
                loss_info = f"MAE Loss: {mae:.6f}"
            elif loss_type == 'cosine':
                actual_loss_value = cosine_loss
                loss_info = f"Cosine Loss: {cosine_loss:.6f} (sim: {cosine_sim:.4f})"
            elif loss_type in ['kl_div', 'js_div']:
                actual_loss_value = loss_value.item()
                loss_info = f"{loss_type.upper()} Loss: {loss_value.item():.6f}"
            else:
                actual_loss_value = loss_value.item()
                loss_info = f"Unknown Loss: {loss_value.item():.6f}"
            
            # Add antenna configuration information and loss details
            antenna_info = f"BS: {self.num_bs_antennas} ant, UE: {self.num_ue_antennas} ant"
            loss_type_info = f"Loss Type: {loss_type.upper()}"
            loss_value_info = f"Loss Value: {actual_loss_value:.6f}"
            config_info = f"[{antenna_info}] [{loss_type_info}] [{loss_value_info}] Sample {sample_idx}"
            
            # Check if 1D or 2D PAS
            is_1d = len(pred_spectrum.shape) == 1
            
            if is_1d:
                # 1D PAS: single line plot
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                
                azimuth_angles = np.linspace(0, 180, len(pred_spectrum))
                
                # Plot 1: Comparison
                axes[0].plot(azimuth_angles, pred_spectrum, 'b-', linewidth=2, label='Predicted', alpha=0.8)
                axes[0].plot(azimuth_angles, target_spectrum, 'r--', linewidth=2, label='Target', alpha=0.8)
                axes[0].set_xlabel('Azimuth Angle (degrees)', fontsize=11)
                axes[0].set_ylabel('Power', fontsize=11)
                axes[0].set_title(f'PAS Comparison | {config_info}\nMSE: {mse:.6f} | Additional Metrics: {loss_info}',
                                fontsize=11, fontweight='bold')
                axes[0].legend(fontsize=10)
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: Error
                error = pred_spectrum - target_spectrum
                axes[1].plot(azimuth_angles, error, 'g-', linewidth=1.5, label='Error (Pred - Target)')
                axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                axes[1].set_xlabel('Azimuth Angle (degrees)', fontsize=11)
                axes[1].set_ylabel('Error', fontsize=11)
                axes[1].set_title('Prediction Error', fontsize=12, fontweight='bold')
                axes[1].legend(fontsize=10)
                axes[1].grid(True, alpha=0.3)
                
            else:
                # 2D PAS: heatmaps
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Overall title with antenna configuration and loss information
                fig.suptitle(f'PAS Comparison | {config_info}\nAdditional Metrics: MSE: {mse:.6f}, MAE: {mae:.6f}', 
                           fontsize=12, fontweight='bold')
                
                # Plot 1: Predicted PAS
                im1 = axes[0, 0].imshow(pred_spectrum.T, aspect='auto', origin='lower', cmap='hot')
                axes[0, 0].set_title('Predicted PAS', fontsize=12, fontweight='bold')
                axes[0, 0].set_xlabel('Azimuth', fontsize=10)
                axes[0, 0].set_ylabel('Elevation', fontsize=10)
                plt.colorbar(im1, ax=axes[0, 0])
                
                # Plot 2: Target PAS
                im2 = axes[0, 1].imshow(target_spectrum.T, aspect='auto', origin='lower', cmap='hot')
                axes[0, 1].set_title('Target PAS', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel('Azimuth', fontsize=10)
                axes[0, 1].set_ylabel('Elevation', fontsize=10)
                plt.colorbar(im2, ax=axes[0, 1])
                
                # Plot 3: Difference
                diff = pred_spectrum - target_spectrum
                im3 = axes[1, 0].imshow(diff.T, aspect='auto', origin='lower', cmap='RdBu_r')
                axes[1, 0].set_title('Difference (Pred - Target)', fontsize=12, fontweight='bold')
                axes[1, 0].set_xlabel('Azimuth', fontsize=10)
                axes[1, 0].set_ylabel('Elevation', fontsize=10)
                plt.colorbar(im3, ax=axes[1, 0])
                
                # Plot 4: Metrics and Loss Information
                axes[1, 1].axis('off')
                metrics_text = f"""
PAS Loss Analysis
{'='*50}
LOSS FUNCTION DETAILS:
• Type: {loss_type.upper()}
• Current Loss Value: {actual_loss_value:.6f}

ANTENNA CONFIGURATION:
• BS Antennas: {self.num_bs_antennas}
• UE Antennas: {self.num_ue_antennas}

COMPARISON METRICS:
• MSE (Mean Squared Error): {mse:.6f}
• MAE (Mean Absolute Error): {mae:.6f}
• Cosine Similarity: {cosine_sim:.4f}
• Cosine Loss: {cosine_loss:.6f}

SPECTRUM STATISTICS:
• Pred Energy: {np.sum(np.abs(pred_spectrum)):.3e}
• Target Energy: {np.sum(np.abs(target_spectrum)):.3e}
• Max Pred Value: {np.max(pred_spectrum):.6f}
• Max Target Value: {np.max(target_spectrum):.6f}

SAMPLE INFO:
• Sample Index: {sample_idx}
• Spectrum Shape: {pred_spectrum.shape}
                """
                axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                               fontsize=9, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'pas_comparison_{timestamp}_sample{sample_idx}.png'
            filepath = self.debug_dir / filename
            
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"Saved debug PAS plot: {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug PAS plot: {e}")

def create_pas_loss(config: Dict) -> PASLoss:
    """
    Create PAS loss from configuration.
    
    Args:
        config: Full configuration dictionary (should be processed config with templates resolved)
        
    Returns:
        PASLoss instance
    """
    # Extract required configs
    bs_config = config.get('base_station', {})
    ue_config = config.get('user_equipment', {})
    # Try to get pas_config from training.loss.pas_loss first, then fallback to direct pas_loss
    pas_config = config.get('training', {}).get('loss', {}).get('pas_loss', {})
    if not pas_config:
        pas_config = config.get('pas_loss', {})
    ofdm_config = config.get('ofdm', {})
    
    if not bs_config:
        raise ValueError("Configuration must contain 'base_station' section for PAS loss")
    if not ue_config:
        raise ValueError("Configuration must contain 'user_equipment' section for PAS loss")
    
    # Get frequency parameters from OFDM config
    center_freq = float(ofdm_config.get('center_frequency', 3.5e9))
    subcarrier_spacing = float(ofdm_config.get('subcarrier_spacing', 245.1e3))
    
    # Get debug_dir from output.training.debug_dir first, then fallback to pas_loss.debug_dir
    debug_dir = None
    output_config = config.get('output', {})
    training_output_config = output_config.get('training', {})
    if 'debug_dir' in training_output_config:
        debug_dir = training_output_config['debug_dir']
    elif 'debug_dir' in pas_config:
        debug_dir = pas_config['debug_dir']
    
    return PASLoss(
        bs_config=bs_config,
        ue_config=ue_config,
        azimuth_divisions=pas_config.get('azimuth_divisions', 18),
        elevation_divisions=pas_config.get('elevation_divisions', 6),
        normalize_pas=pas_config.get('normalize_pas', True),
        loss_type=pas_config.get('type', 'mse'),
        weight_by_power=pas_config.get('weight_by_power', True),
        center_freq=center_freq,
        subcarrier_spacing=subcarrier_spacing,
        debug_dir=debug_dir,
        debug_sample_rate=pas_config.get('debug_sample_rate', 0.5)
    )

