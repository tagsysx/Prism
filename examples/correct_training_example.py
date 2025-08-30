#!/usr/bin/env python3
"""
Correct Prism Training Example

This example demonstrates the proper way to train Prism networks using:
1. PrismTrainingInterface for correct CSI computation
2. Specialized loss functions for complex-valued CSI
3. Proper integration of network outputs with ray tracing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import logging

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from prism import (
    PrismNetwork, 
    PrismTrainingInterface, 
    CPURayTracer,
    PrismLoss,
    FrequencyAwareLoss,
    CSIVirtualLinkLoss
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorrectPrismTrainer:
    """
    Correct trainer implementation that properly integrates:
    - PrismNetwork outputs
    - Ray tracing for CSI computation
    - Appropriate loss functions
    """
    
    def __init__(self, config: dict):
        """Initialize the correct trainer."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create PrismNetwork
        self.prism_network = PrismNetwork(
            num_subcarriers=config['num_subcarriers'],
            num_ue_antennas=config['num_ue_antennas'],
            num_bs_antennas=config['num_bs_antennas'],
            feature_dim=config['feature_dim'],
            antenna_embedding_dim=config['antenna_embedding_dim'],
            azimuth_divisions=config['azimuth_divisions'],
            elevation_divisions=config['elevation_divisions'],
            top_k_directions=config['top_k_directions'],
            complex_output=True
        ).to(self.device)
        
        # Create RayTracer
        self.ray_tracer = CPURayTracer(
            scene_size=200.0,  # Use scene_size instead of scene_bounds
            azimuth_divisions=config['azimuth_divisions'],
            elevation_divisions=config['elevation_divisions'],
            max_ray_length=config['max_ray_length']
        )
        
        # Create PrismTrainingInterface
        self.training_interface = PrismTrainingInterface(
            prism_network=self.prism_network,
            ray_tracer=self.ray_tracer,
            num_sampling_points=config['num_sampling_points'],
            scene_bounds=config['scene_bounds']
        ).to(self.device)
        
        # Create appropriate loss function
        if config['loss_type'] == 'frequency_aware':
            self.criterion = FrequencyAwareLoss(
                num_subcarriers=config['num_subcarriers'],
                frequency_emphasis=config['frequency_emphasis'],
                complex_handling='magnitude_phase'
            ).to(self.device)
        elif config['loss_type'] == 'csi_virtual_link':
            self.criterion = CSIVirtualLinkLoss(
                num_ue_antennas=config['num_ue_antennas'],
                num_subcarriers=config['num_subcarriers'],
                virtual_link_sampling=config['virtual_link_sampling'],
                sampling_ratio=config['sampling_ratio']
            ).to(self.device)
        else:
            self.criterion = PrismLoss(
                loss_type=config['loss_type'],
                complex_handling='magnitude_phase'
            ).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.training_interface.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        logger.info(f"Correct Prism trainer initialized on {self.device}")
        logger.info(f"Training interface info: {self.training_interface.get_training_info()}")
        logger.info(f"Loss function info: {self.criterion.get_loss_info()}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch using the correct training interface."""
        self.training_interface.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Extract batch data
            ue_positions = batch_data['ue_positions'].to(self.device)
            bs_position = batch_data['bs_position'].to(self.device)
            antenna_indices = batch_data['antenna_indices'].to(self.device)
            csi_targets = batch_data['csi_targets'].to(self.device)
            
            # Forward pass through training interface
            self.optimizer.zero_grad()
            
            try:
                # This is the key difference: use training interface instead of direct model
                outputs = self.training_interface(
                    ue_positions=ue_positions,
                    bs_position=bs_position,
                    antenna_indices=antenna_indices
                )
                
                # Extract CSI predictions (these are actual CSI values, not intermediate factors)
                csi_predictions = outputs['csi_predictions']
                
                # Compute loss between computed CSI and target CSI
                loss = self.criterion(csi_predictions, csi_targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.training_interface.parameters(), 
                    max_norm=1.0
                )
                
                # Update parameters
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}, Loss: {loss.item():.6f}")
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate using the correct training interface."""
        self.training_interface.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                ue_positions = batch_data['ue_positions'].to(self.device)
                bs_position = batch_data['bs_position'].to(self.device)
                antenna_indices = batch_data['antenna_indices'].to(self.device)
                csi_targets = batch_data['csi_targets'].to(self.device)
                
                try:
                    # Forward pass
                    outputs = self.training_interface(
                        ue_positions=ue_positions,
                        bs_position=bs_position,
                        antenna_indices=antenna_indices
                    )
                    
                    csi_predictions = outputs['csi_predictions']
                    
                    # Compute validation loss
                    loss = self.criterion(csi_predictions, csi_targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Validation error: {e}")
                    continue
        
        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_val_loss
    
    def save_checkpoint(self, filepath: str, epoch: int, train_loss: float, val_loss: float):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'training_interface_state_dict': self.training_interface.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")


def create_sample_data(batch_size: int, num_ue: int, num_subcarriers: int) -> dict:
    """Create sample training data."""
    # Generate random UE positions
    ue_positions = torch.randn(batch_size, num_ue, 3) * 50.0  # ±50m range
    
    # Generate random BS position
    bs_position = torch.randn(batch_size, 3) * 10.0  # ±10m range
    
    # Generate random antenna indices
    antenna_indices = torch.randint(0, 64, (batch_size, 64))  # 64 BS antennas
    
    # Generate random CSI targets (complex values)
    csi_targets = torch.complex(
        torch.randn(batch_size, num_ue, num_subcarriers),
        torch.randn(batch_size, num_ue, num_subcarriers)
    )
    
    return {
        'ue_positions': ue_positions,
        'bs_position': bs_position,
        'antenna_indices': antenna_indices,
        'csi_targets': csi_targets
    }


def main():
    """Main training example."""
    # Configuration
    config = {
        'num_subcarriers': 64,
        'num_ue_antennas': 4,
        'num_bs_antennas': 64,
        'feature_dim': 128,
        'antenna_embedding_dim': 64,
        'azimuth_divisions': 16,
        'elevation_divisions': 8,
        'top_k_directions': 32,
        'num_sampling_points': 64,
        'scene_bounds': ([-100, -100, 0], [100, 100, 50]),
        'max_ray_length': 200.0,
        'loss_type': 'frequency_aware',
        'frequency_emphasis': 'center',
        'virtual_link_sampling': 'random',
        'sampling_ratio': 0.5,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4
    }
    
    # Create trainer
    trainer = CorrectPrismTrainer(config)
    
    # Create sample data
    batch_size = 8
    sample_data = create_sample_data(
        batch_size, 
        config['num_ue_antennas'], 
        config['num_subcarriers']
    )
    
    # Create dataset and dataloader
    dataset = TensorDataset(
        sample_data['ue_positions'],
        sample_data['bs_position'],
        sample_data['antenna_indices'],
        sample_data['csi_targets']
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_loss = trainer.train_epoch(dataloader)
        
        # Validate
        val_loss = trainer.validate(dataloader)
        
        # Update learning rate
        trainer.scheduler.step(val_loss)
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(checkpoint_path, epoch + 1, train_loss, val_loss)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
