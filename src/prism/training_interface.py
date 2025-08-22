"""
Prism Training Interface

This module provides a simplified training interface that integrates:
1. PrismNetwork for spatial feature extraction
2. BS-Centric ray tracing from each BS antenna
3. AntennaNetwork-guided direction selection
4. Antenna-specific subcarrier selection
5. CSI computation and loss calculation
6. Checkpoint support for training recovery
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging
import random
import os
import json

from .networks.prism_network import PrismNetwork
from .ray_tracer import DiscreteRayTracer

logger = logging.getLogger(__name__)

class PrismTrainingInterface(nn.Module):
    """
    Simplified training interface with BS-Centric ray tracing and checkpoint support.
    
    This interface handles:
    1. BS-Centric ray tracing from each BS antenna
    2. AntennaNetwork-guided direction selection
    3. Antenna-specific subcarrier selection
    4. CSI computation and loss calculation
    5. Training checkpoint and recovery
    """
    
    def __init__(
        self,
        prism_network: PrismNetwork,
        ray_tracer: DiscreteRayTracer,
        num_sampling_points: int = 64,
        scene_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        subcarrier_sampling_ratio: float = 0.3,
        checkpoint_dir: str = "checkpoints"
    ):
        super().__init__()
        self.prism_network = prism_network
        self.ray_tracer = ray_tracer
        self.num_sampling_points = num_sampling_points
        self.subcarrier_sampling_ratio = subcarrier_sampling_ratio
        self.checkpoint_dir = checkpoint_dir
        
        # Scene bounds for sampling point generation
        if scene_bounds is None:
            self.scene_min = torch.tensor([-50.0, -50.0, 0.0])
            self.scene_max = torch.tensor([50.0, 50.0, 30.0])
        else:
            self.scene_min, self.scene_max = scene_bounds
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training state for checkpoint recovery
        self.current_epoch = 0
        self.current_batch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Current selection state
        self.current_selection = None
        self.current_selection_mask = None
    
    def forward(
        self,
        ue_positions: torch.Tensor,
        bs_position: torch.Tensor,
        antenna_indices: torch.Tensor,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with BS-Centric ray tracing using integrated ray_tracer.
        
        Args:
            ue_positions: UE positions (batch_size, num_ue, 3)
            bs_position: Base station position (batch_size, 3)
            antenna_indices: BS antenna indices (batch_size, num_bs_antennas)
            return_intermediates: Whether to return intermediate outputs
            
        Returns:
            Dictionary containing CSI predictions and network outputs
        """
        batch_size = ue_positions.shape[0]
        num_ue = ue_positions.shape[1]
        num_bs_antennas = antenna_indices.shape[1]
        num_subcarriers = self.prism_network.num_subcarriers
        num_selected = int(num_subcarriers * self.subcarrier_sampling_ratio)
        
        # Step 1: Select subcarriers for each BS antenna
        selection_info = self._select_subcarriers_per_antenna(
            batch_size, num_ue, num_bs_antennas, num_subcarriers, num_selected
        )
        self.current_selection = selection_info['selected_indices']
        self.current_selection_mask = selection_info['selection_mask']
        
        # Step 2: BS-Centric ray tracing from each BS antenna using ray_tracer
        csi_predictions = torch.zeros(
            batch_size, num_bs_antennas, num_ue, num_selected, 
            dtype=torch.complex64, device=ue_positions.device
        )
        
        all_ray_results = []
        all_signal_strengths = []
        
        for bs_antenna_idx in range(num_bs_antennas):
            # Get antenna-specific embedding
            antenna_embedding = self.prism_network.antenna_codebook(
                antenna_indices[:, bs_antenna_idx]
            )
            
            # Process each batch item
            batch_ray_results = []
            batch_signal_strengths = []
            
            for b in range(batch_size):
                # Convert UE positions to list format for ray_tracer
                ue_pos_list = [ue_positions[b, u].cpu() for u in range(num_ue)]
                
                # Create subcarrier dictionary mapping UE positions to selected subcarriers
                selected_subcarriers = {}
                for u in range(num_ue):
                    ue_pos_tuple = tuple(ue_pos_list[u].tolist())
                    selected_subcarriers[ue_pos_tuple] = self.current_selection[b, bs_antenna_idx, u].tolist()
                
                # Use ray_tracer for signal accumulation with AntennaNetwork-guided directions
                ray_results = self.ray_tracer.accumulate_signals(
                    base_station_pos=bs_position[b].cpu(),
                    ue_positions=ue_pos_list,
                    selected_subcarriers=selected_subcarriers,
                    antenna_embedding=antenna_embedding[b].cpu()
                )
                
                batch_ray_results.append(ray_results)
                
                # Convert ray_tracer results to CSI predictions
                for u in range(num_ue):
                    ue_pos_tuple = tuple(ue_pos_list[u].tolist())
                    ue_selected_subcarriers = self.current_selection[b, bs_antenna_idx, u].tolist()
                    
                    for k_idx, k in enumerate(ue_selected_subcarriers):
                        if (ue_pos_tuple, k) in ray_results:
                            signal_strength = ray_results[(ue_pos_tuple, k)]
                            
                            # Convert signal strength to complex CSI
                            csi_value = self._signal_strength_to_csi(
                                signal_strength, bs_position[b], ue_positions[b, u], k
                            )
                            csi_predictions[b, bs_antenna_idx, u, k_idx] = csi_value
                        else:
                            # Fallback for missing results
                            csi_predictions[b, bs_antenna_idx, u, k_idx] = torch.complex(
                                torch.tensor(0.0, device=ue_positions.device), 
                                torch.tensor(0.0, device=ue_positions.device)
                            )
                
                batch_signal_strengths.append(ray_results)
            
            all_ray_results.append(batch_ray_results)
            all_signal_strengths.append(batch_signal_strengths)
        
        outputs = {
            'csi_predictions': csi_predictions,
            'ray_results': all_ray_results,
            'signal_strengths': all_signal_strengths,
            'subcarrier_selection': selection_info
        }
        
        if return_intermediates:
            outputs.update({'ray_tracer_results': all_ray_results})
        
        return outputs
    
    def _select_subcarriers_per_antenna(
        self, 
        batch_size: int, 
        num_ue: int, 
        num_bs_antennas: int,
        total_subcarriers: int,
        num_selected: int
    ) -> Dict[str, torch.Tensor]:
        """Select subcarriers for each BS antenna independently."""
        selected_indices = torch.zeros(batch_size, num_bs_antennas, num_ue, num_selected, dtype=torch.long)
        selection_mask = torch.zeros(batch_size, num_bs_antennas, num_ue, total_subcarriers, dtype=torch.bool)
        
        for b in range(batch_size):
            for bs_antenna in range(num_bs_antennas):
                for u in range(num_ue):
                    ue_selected = random.sample(range(total_subcarriers), num_selected)
                    selected_indices[b, bs_antenna, u] = torch.tensor(ue_selected)
                    selection_mask[b, bs_antenna, u, ue_selected] = True
        
        return {
            'selected_indices': selected_indices,
            'selection_mask': selection_mask,
            'num_selected': num_selected
        }
    

    
    def _compute_view_directions(
        self, 
        bs_position: torch.Tensor, 
        sampled_positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute view directions from BS to each sampling point."""
        view_directions = sampled_positions - bs_position.unsqueeze(1).unsqueeze(2)
        view_directions = view_directions / (torch.norm(view_directions, dim=-1, keepdim=True) + 1e-8)
        return view_directions
    
    def _signal_strength_to_csi(
        self,
        signal_strength: float,
        bs_pos: torch.Tensor,
        ue_pos: torch.Tensor,
        subcarrier_idx: int
    ) -> torch.complex64:
        """
        Convert ray_tracer signal strength to complex CSI value.
        
        Args:
            signal_strength: Signal strength from ray_tracer
            bs_pos: Base station position
            ue_pos: UE position
            subcarrier_idx: Subcarrier index
            
        Returns:
            Complex CSI value
        """
        # Calculate distance-based phase
        distance = torch.norm(ue_pos - bs_pos)
        phase = 2 * torch.pi * subcarrier_idx * distance / 100.0  # Normalized wavelength
        
        # Convert signal strength to complex CSI with phase
        amplitude = torch.sqrt(torch.tensor(signal_strength, dtype=torch.float32, device=bs_pos.device))
        csi_value = torch.complex(
            amplitude * torch.cos(phase), 
            amplitude * torch.sin(phase)
        )
        
        return csi_value.to(torch.complex64)
    

    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        loss_function: nn.Module
    ) -> torch.Tensor:
        """Compute loss for selected subcarriers."""
        if self.current_selection_mask is None:
            raise ValueError("No subcarrier selection available. Call forward() first.")
        
        batch_size, num_bs_antennas, num_ue, total_subcarriers = targets.shape
        num_selected = self.current_selection.shape[-1]
        
        selected_targets = torch.zeros(
            batch_size, num_bs_antennas, num_ue, num_selected,
            dtype=targets.dtype, device=targets.device
        )
        
        for b in range(batch_size):
            for bs_antenna in range(num_bs_antennas):
                for u in range(num_ue):
                    selected_indices = self.current_selection[b, bs_antenna, u]
                    # Ensure indices are within bounds
                    valid_indices = selected_indices[selected_indices < total_subcarriers]
                    if len(valid_indices) > 0:
                        selected_targets[b, bs_antenna, u, :len(valid_indices)] = targets[b, bs_antenna, u, valid_indices]
        
        return loss_function(predictions, selected_targets)
    
    def set_training_phase(self, phase: int):
        """Set training phase for curriculum learning."""
        if phase == 0:
            self.prism_network.azimuth_divisions = 8
            self.prism_network.elevation_divisions = 4
            self.prism_network.top_k_directions = 16
        elif phase == 1:
            self.prism_network.azimuth_divisions = 16
            self.prism_network.elevation_divisions = 8
            self.prism_network.top_k_directions = 32
        elif phase == 2:
            self.prism_network.azimuth_divisions = 36
            self.prism_network.elevation_divisions = 18
            self.prism_network.top_k_directions = 64
        else:
            raise ValueError(f"Invalid training phase: {phase}")
        
        logger.info(f"Training phase {phase} set: {self.prism_network.azimuth_divisions}×{self.prism_network.elevation_divisions}")
    
    # Checkpoint and recovery methods
    def save_checkpoint(self, filename: str = None):
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}_batch_{self.current_batch}.pt"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'prism_network_state_dict': self.prism_network.state_dict(),
            'current_epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'current_selection': self.current_selection,
            'current_selection_mask': self.current_selection_mask,
            'training_config': {
                'num_sampling_points': self.num_sampling_points,
                'subcarrier_sampling_ratio': self.subcarrier_sampling_ratio,
                'scene_bounds': (self.scene_min.tolist(), self.scene_max.tolist())
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save training info
        info_path = checkpoint_path.replace('.pt', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(self.get_training_info(), f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model states
        self.load_state_dict(checkpoint['model_state_dict'])
        self.prism_network.load_state_dict(checkpoint['prism_network_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['current_epoch']
        self.current_batch = checkpoint['current_batch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        self.current_selection = checkpoint['current_selection']
        self.current_selection_mask = checkpoint['current_selection_mask']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}, batch {self.current_batch}")
        logger.info(f"Best loss so far: {self.best_loss}")
    
    def update_training_state(self, epoch: int, batch: int, loss: float):
        """Update training state for checkpoint tracking."""
        self.current_epoch = epoch
        self.current_batch = batch
        
        if loss < self.best_loss:
            self.best_loss = loss
        
        self.training_history.append({
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'best_loss': self.best_loss
        })
    
    def get_training_info(self) -> Dict:
        """Get training interface information."""
        return {
            'num_sampling_points': self.num_sampling_points,
            'subcarrier_sampling_ratio': self.subcarrier_sampling_ratio,
            'scene_bounds': (self.scene_min.tolist(), self.scene_max.tolist()),
            'current_epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'best_loss': self.best_loss,
            'training_history_length': len(self.training_history),
            'checkpoint_dir': self.checkpoint_dir,
            'prism_network_config': {
                'azimuth_divisions': self.prism_network.azimuth_divisions,
                'elevation_divisions': self.prism_network.elevation_divisions,
                'top_k_directions': self.prism_network.top_k_directions
            }
        }
