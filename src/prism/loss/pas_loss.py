"""
PAS (Power Angular Spectrum) Loss Implementation

This module implements loss functions for Power Angular Spectrum (PAS) comparison
between predicted and target CSI data. The PAS represents the spatial distribution
of signal power across different angles of arrival/departure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import logging
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
        debug_dir: str = None
    ):
        super().__init__()
        
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.normalize_pas = normalize_pas
        self.loss_type = loss_type.lower()
        self.weight_by_power = weight_by_power
        self.center_freq = center_freq
        self.subcarrier_spacing = subcarrier_spacing
        self.debug_dir = debug_dir or "results/temp"  # Default fallback path
        
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
            
            # Convert to loss: 1 - abs(cosine_similarity) (range [0, 1], 0 = perfect similarity)
            cosine_loss_per_sample = 1.0 - torch.abs(cosine_similarity)  # [batch_size]
            
            # Apply power weights if provided
            if power_weights is not None:
                weighted_cosine_loss = cosine_loss_per_sample * power_weights  # [batch_size]
                loss = torch.mean(weighted_cosine_loss)
            else:
                loss = torch.mean(cosine_loss_per_sample)
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        combined_loss = loss
        
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
        
        # Debug: Save PAS data with 1% probability for inspection
        # Reshape back to original dimensions for visualization
        pred_pas_for_debug = pred_pas_batch.view(batch_size, total_antennas, azimuth_div, elevation_div)
        target_pas_for_debug = target_pas_batch.view(batch_size, total_antennas, azimuth_div, elevation_div)
        self._maybe_save_pas_debug(pred_pas_for_debug, target_pas_for_debug, unique_positions, pred_csi_by_pos, target_csi_by_pos)
        
        # Compute loss with normalization factor
        pas_loss = self.compute_loss(pred_pas_batch, target_pas_batch, total_antennas)
        
        return pas_loss
    
    
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
    
    def _maybe_save_pas_debug(self, pred_pas_batch: torch.Tensor, target_pas_batch: torch.Tensor, unique_positions: list, 
                             pred_csi_by_pos: list = None, target_csi_by_pos: list = None):
        """
        Save PAS data and visualization for debugging with 1% probability.
        
        Args:
            pred_pas_batch: [num_position_pairs, total_antennas, azimuth_divisions, elevation_divisions]
            target_pas_batch: [num_position_pairs, total_antennas, azimuth_divisions, elevation_divisions]
            unique_positions: List of (bs_pos, ue_pos) tuples
            pred_csi_by_pos: List of predicted CSI matrices [num_bs_antennas, num_ue_antennas, num_subcarriers]
            target_csi_by_pos: List of target CSI matrices [num_bs_antennas, num_ue_antennas, num_subcarriers]
        """
        import random
        import os
        import numpy as np
        from datetime import datetime
        
        # 1% probability to save debug data
        if random.random() < 0.05:
            try:
                # Use configured debug directory
                debug_dir = self.debug_dir
                os.makedirs(debug_dir, exist_ok=True)
                
                # Generate timestamp for unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                
                # Convert tensors to numpy for saving
                pred_pas_np = pred_pas_batch.detach().cpu().numpy()  # [num_position_pairs, total_antennas, azimuth_divisions, elevation_divisions]
                target_pas_np = target_pas_batch.detach().cpu().numpy()  # [num_position_pairs, total_antennas, azimuth_divisions, elevation_divisions]
                
                # Convert CSI data to numpy if available
                pred_csi_np = None
                target_csi_np = None
                if pred_csi_by_pos is not None:
                    pred_csi_np = [csi.detach().cpu().numpy() for csi in pred_csi_by_pos]
                if target_csi_by_pos is not None:
                    target_csi_np = [csi.detach().cpu().numpy() for csi in target_csi_by_pos]
                
                # Create visualization
                self._create_pas_visualization(pred_pas_np, target_pas_np, unique_positions, timestamp, debug_dir, pred_csi_np, target_csi_np)
                
                # Save data as .npz file
                save_path = os.path.join(debug_dir, f"pas_debug_{timestamp}.npz")
                positions_data = []
                for i, (bs_pos, ue_pos) in enumerate(unique_positions):
                    positions_data.append({
                        'position_pair_idx': i,
                        'bs_position': bs_pos.detach().cpu().numpy().tolist(),
                        'ue_position': ue_pos.detach().cpu().numpy().tolist()
                    })
                
                np.savez(
                    save_path,
                    pred_pas=pred_pas_np,
                    target_pas=target_pas_np,
                    positions=positions_data,
                    azimuth_divisions=self.azimuth_divisions,
                    elevation_divisions=self.elevation_divisions,
                    num_bs_antennas=self.num_bs_antennas,
                    num_ue_antennas=self.num_ue_antennas,
                    normalize_pas=self.normalize_pas,
                    center_freq=self.center_freq,
                    subcarrier_spacing=self.subcarrier_spacing
                )
                
                
            except Exception as e:
                # Don't let debug saving crash the training
                pass
    
    def _create_pas_visualization(self, pred_pas_np, target_pas_np, 
                                 unique_positions: list, timestamp: str, debug_dir: str,
                                 pred_csi_np: list = None, target_csi_np: list = None):
        """
        Create and save PAS visualization plots with CSI magnitude information.
        
        Args:
            pred_pas_np: Predicted PAS numpy array [num_position_pairs, total_antennas, azimuth_divisions, elevation_divisions]
            target_pas_np: Target PAS numpy array [num_position_pairs, total_antennas, azimuth_divisions, elevation_divisions]
            unique_positions: List of (bs_pos, ue_pos) tuples
            timestamp: Timestamp string for filename
            debug_dir: Directory to save plots
            pred_csi_np: List of predicted CSI matrices [num_bs_antennas, num_ue_antennas, num_subcarriers]
            target_csi_np: List of target CSI matrices [num_bs_antennas, num_ue_antennas, num_subcarriers]
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            import numpy as np
            import os
            matplotlib.use('Agg')  # Use non-interactive backend
            
            num_position_pairs, total_antennas, azimuth_div, elevation_div = pred_pas_np.shape
            
            # Create angle grids for plotting
            azimuth_angles = np.linspace(0, 360, self.azimuth_divisions)  # 0 to 360 degrees
            elevation_angles = np.linspace(0, 90, self.elevation_divisions)  # 0 to 90 degrees
            
            # Plot each position pair (limit to first 3 for clarity)
            for pos_idx in range(min(num_position_pairs, 3)):
                # Create larger figure with more subplots to include CSI magnitude
                if pred_csi_np is not None and target_csi_np is not None:
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    fig.suptitle(f'PAS & CSI Comparison - Position Pair {pos_idx} - {timestamp}', fontsize=16)
                else:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    fig.suptitle(f'PAS Comparison - Position Pair {pos_idx} - {timestamp}', fontsize=14)
                
                # Get PAS for this position pair: [total_antennas, azimuth_divisions, elevation_divisions]
                pred_pas_all_antennas = pred_pas_np[pos_idx]  # [total_antennas, azimuth_divisions, elevation_divisions]
                target_pas_all_antennas = target_pas_np[pos_idx]  # [total_antennas, azimuth_divisions, elevation_divisions]
                
                # Average across all antennas for visualization (you can also plot individual antennas if needed)
                pred_pas_single = np.mean(pred_pas_all_antennas, axis=0)  # [azimuth_divisions, elevation_divisions]
                target_pas_single = np.mean(target_pas_all_antennas, axis=0)  # [azimuth_divisions, elevation_divisions]
                
                if pred_csi_np is not None and target_csi_np is not None:
                    # 3x3 layout with CSI magnitude information
                    pred_csi_single = pred_csi_np[pos_idx]  # [num_bs_antennas, num_ue_antennas, num_subcarriers]
                    target_csi_single = target_csi_np[pos_idx]
                    
                    # Row 1: CSI Magnitude plots
                    # Predicted CSI magnitude (averaged over subcarriers)
                    pred_csi_mag = np.mean(np.abs(pred_csi_single), axis=2)  # [num_bs_antennas, num_ue_antennas]
                    im_csi1 = axes[0, 0].imshow(pred_csi_mag, aspect='auto', cmap='plasma')
                    axes[0, 0].set_title('Predicted CSI Magnitude\n(avg over subcarriers)')
                    axes[0, 0].set_xlabel('UE Antenna Index')
                    axes[0, 0].set_ylabel('BS Antenna Index')
                    plt.colorbar(im_csi1, ax=axes[0, 0])
                    
                    # Target CSI magnitude (averaged over subcarriers)
                    target_csi_mag = np.mean(np.abs(target_csi_single), axis=2)
                    im_csi2 = axes[0, 1].imshow(target_csi_mag, aspect='auto', cmap='plasma')
                    axes[0, 1].set_title('Target CSI Magnitude\n(avg over subcarriers)')
                    axes[0, 1].set_xlabel('UE Antenna Index')
                    axes[0, 1].set_ylabel('BS Antenna Index')
                    plt.colorbar(im_csi2, ax=axes[0, 1])
                    
                    # CSI magnitude difference
                    csi_mag_diff = pred_csi_mag - target_csi_mag
                    im_csi3 = axes[0, 2].imshow(csi_mag_diff, aspect='auto', cmap='RdBu_r')
                    axes[0, 2].set_title('CSI Magnitude Difference\n(Pred - Target)')
                    axes[0, 2].set_xlabel('UE Antenna Index')
                    axes[0, 2].set_ylabel('BS Antenna Index')
                    plt.colorbar(im_csi3, ax=axes[0, 2])
                    
                    # Row 2: PAS heatmaps
                    # Predicted PAS
                    im1 = axes[1, 0].imshow(pred_pas_single.T, aspect='auto', origin='lower', 
                                           extent=[azimuth_angles[0], azimuth_angles[-1], 
                                                  elevation_angles[0], elevation_angles[-1]],
                                           cmap='viridis')
                    axes[1, 0].set_title('Predicted PAS')
                    axes[1, 0].set_xlabel('Azimuth (degrees)')
                    axes[1, 0].set_ylabel('Elevation (degrees)')
                    plt.colorbar(im1, ax=axes[1, 0])
                    
                    # Target PAS
                    im2 = axes[1, 1].imshow(target_pas_single.T, aspect='auto', origin='lower',
                                           extent=[azimuth_angles[0], azimuth_angles[-1], 
                                                  elevation_angles[0], elevation_angles[-1]],
                                           cmap='viridis')
                    axes[1, 1].set_title('Target PAS')
                    axes[1, 1].set_xlabel('Azimuth (degrees)')
                    axes[1, 1].set_ylabel('Elevation (degrees)')
                    plt.colorbar(im2, ax=axes[1, 1])
                    
                    # PAS difference
                    pas_diff = pred_pas_single - target_pas_single
                    im3 = axes[1, 2].imshow(pas_diff.T, aspect='auto', origin='lower', cmap='RdBu_r',
                                           extent=[azimuth_angles[0], azimuth_angles[-1], 
                                                  elevation_angles[0], elevation_angles[-1]])
                    axes[1, 2].set_title('PAS Difference\n(Pred - Target)')
                    axes[1, 2].set_xlabel('Azimuth (degrees)')
                    axes[1, 2].set_ylabel('Elevation (degrees)')
                    plt.colorbar(im3, ax=axes[1, 2])
                    
                    # Row 3: 1D profiles and statistics
                    # Azimuth profiles
                    pred_azimuth = np.mean(pred_pas_single, axis=1)
                    target_azimuth = np.mean(target_pas_single, axis=1)
                    
                    axes[1, 0].plot(azimuth_angles, pred_azimuth, 'b-', label='Predicted', linewidth=2)
                    axes[1, 0].plot(azimuth_angles, target_azimuth, 'r--', label='Target', linewidth=2)
                    axes[1, 0].set_title('Azimuth Profile (avg over elevation)')
                    axes[1, 0].set_xlabel('Azimuth (degrees)')
                    axes[1, 0].set_ylabel('Power')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Elevation profiles
                    pred_elevation = np.mean(pred_pas_single, axis=0)
                    target_elevation = np.mean(target_pas_single, axis=0)
                    
                    axes[1, 1].plot(elevation_angles, pred_elevation, 'b-', label='Predicted', linewidth=2)
                    axes[1, 1].plot(elevation_angles, target_elevation, 'r--', label='Target', linewidth=2)
                    axes[1, 1].set_title('Elevation Profile (avg over azimuth)')
                    axes[1, 1].set_xlabel('Elevation (degrees)')
                    axes[1, 1].set_ylabel('Power')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Hide the unused subplot
                    axes[1, 2].set_visible(False)
                    
                    
                else:
                    # Original 2x2 layout when CSI data is not available
                    # Predicted PAS
                    im1 = axes[0, 0].imshow(pred_pas_single.T, aspect='auto', origin='lower', 
                                           extent=[azimuth_angles[0], azimuth_angles[-1], 
                                                  elevation_angles[0], elevation_angles[-1]],
                                           cmap='viridis')
                    axes[0, 0].set_title('Predicted PAS')
                    axes[0, 0].set_xlabel('Azimuth (degrees)')
                    axes[0, 0].set_ylabel('Elevation (degrees)')
                    plt.colorbar(im1, ax=axes[0, 0])
                    
                    # Target PAS
                    im2 = axes[0, 1].imshow(target_pas_single.T, aspect='auto', origin='lower',
                                           extent=[azimuth_angles[0], azimuth_angles[-1], 
                                                  elevation_angles[0], elevation_angles[-1]],
                                           cmap='viridis')
                    axes[0, 1].set_title('Target PAS')
                    axes[0, 1].set_xlabel('Azimuth (degrees)')
                    axes[0, 1].set_ylabel('Elevation (degrees)')
                    plt.colorbar(im2, ax=axes[0, 1])
                    
                    # 1D comparison - Azimuth profiles (averaged over elevation)
                    pred_azimuth = np.mean(pred_pas_single, axis=1)
                    target_azimuth = np.mean(target_pas_single, axis=1)
                    
                    axes[1, 0].plot(azimuth_angles, pred_azimuth, 'b-', label='Predicted', linewidth=2)
                    axes[1, 0].plot(azimuth_angles, target_azimuth, 'r--', label='Target', linewidth=2)
                    axes[1, 0].set_title('Azimuth Profile (avg over elevation)')
                    axes[1, 0].set_xlabel('Azimuth (degrees)')
                    axes[1, 0].set_ylabel('Power')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # 1D comparison - Elevation profiles (averaged over azimuth)
                    pred_elevation = np.mean(pred_pas_single, axis=0)
                    target_elevation = np.mean(target_pas_single, axis=0)
                    
                    axes[1, 1].plot(elevation_angles, pred_elevation, 'b-', label='Predicted', linewidth=2)
                    axes[1, 1].plot(elevation_angles, target_elevation, 'r--', label='Target', linewidth=2)
                    axes[1, 1].set_title('Elevation Profile (avg over azimuth)')
                    axes[1, 1].set_xlabel('Elevation (degrees)')
                    axes[1, 1].set_ylabel('Power')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                    
                
                plt.tight_layout()
                
                # Save plot
                plot_path = os.path.join(debug_dir, f"pas_comparison_{timestamp}_pos{pos_idx}.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()  # Close to free memory
                
                
        except Exception as e:
            pass
    

def create_pas_loss(config: Dict) -> PASLoss:
    """
    Create PAS loss from configuration.
    
    Args:
        config: Full configuration dictionary containing 'base_station', 'user_equipment', 'pas_loss', and 'ofdm' sections
        
    Returns:
        PASLoss instance
    """
    # Extract required configs
    bs_config = config.get('base_station', {})
    ue_config = config.get('user_equipment', {})
    pas_config = config.get('pas_loss', {})
    ofdm_config = config.get('ofdm', {})
    
    if not bs_config:
        raise ValueError("Configuration must contain 'base_station' section for PAS loss")
    if not ue_config:
        raise ValueError("Configuration must contain 'user_equipment' section for PAS loss")
    
    # Get frequency parameters from OFDM config
    center_freq = float(ofdm_config.get('center_frequency', 3.5e9))
    subcarrier_spacing = float(ofdm_config.get('subcarrier_spacing', 245.1e3))
    
    return PASLoss(
        bs_config=bs_config,
        ue_config=ue_config,
        azimuth_divisions=pas_config.get('azimuth_divisions', 18),
        elevation_divisions=pas_config.get('elevation_divisions', 6),
        normalize_pas=pas_config.get('normalize_pas', True),
        loss_type=pas_config.get('type', 'mse'),
        weight_by_power=pas_config.get('weight_by_power', True),
        center_freq=center_freq,
        subcarrier_spacing=subcarrier_spacing
    )

