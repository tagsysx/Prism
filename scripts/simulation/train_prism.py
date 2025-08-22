#!/usr/bin/env python3
"""
Prism Network Training Script

This script trains the Prism neural network for electromagnetic ray tracing
using simulated data from Sionna. It implements the complete training pipeline
including data loading, model initialization, training loop, and checkpointing.
"""

import os
import sys
import argparse
import logging
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from prism.networks.prism_network import PrismNetwork
from prism.ray_tracer import DiscreteRayTracer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PrismTrainer:
    """Main trainer class for Prism network"""
    
    def __init__(self, config_path: str, data_path: str, output_dir: str):
        """Initialize trainer with configuration and data paths"""
        self.config_path = config_path
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and training components
        self._setup_model()
        self._setup_training()
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        
    def _load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
    
    def _setup_model(self):
        """Initialize Prism network model"""
        nn_config = self.config['neural_networks']
        
        # Create model with configuration from YAML
        self.model = PrismNetwork(
            num_subcarriers=nn_config['attenuation_decoder']['output_dim'],
            num_ue_antennas=nn_config['attenuation_decoder']['num_ue'],
            num_bs_antennas=nn_config['antenna_codebook']['num_antennas'],
            position_dim=nn_config['attenuation_network']['input_dim'],
            hidden_dim=nn_config['attenuation_network']['hidden_dim'],
            feature_dim=nn_config['attenuation_network']['feature_dim'],
            antenna_embedding_dim=nn_config['antenna_codebook']['embedding_dim'],
            use_antenna_codebook=nn_config['antenna_codebook']['learnable'],
            use_ipe_encoding=True,  # Enable IPE encoding for better performance
            azimuth_divisions=self.config['ray_tracing']['azimuth_divisions'],
            elevation_divisions=self.config['ray_tracing']['elevation_divisions'],
            top_k_directions=32,  # Top-K directions for importance sampling
            complex_output=True
        )
        
        self.model = self.model.to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
    def _setup_training(self):
        """Setup training hyperparameters and optimizers"""
        # Training hyperparameters
        self.batch_size = self.config['performance']['batch_size']
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.save_interval = 10
        
        # Loss function for complex-valued outputs
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        logger.info(f"Training setup: batch_size={self.batch_size}, lr={self.learning_rate}")
        
    def _load_data(self):
        """Load training data from HDF5 file"""
        logger.info(f"Loading training data from {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            # Load UE positions
            ue_positions = f['ue_positions'][:]
            logger.info(f"Loaded {len(ue_positions)} UE positions")
            
            # Load channel responses (CSI)
            csi_data = f['channel_responses'][:]
            logger.info(f"Loaded CSI data with shape: {csi_data.shape}")
            
            # Load BS position
            bs_position = f['bs_position'][:]
            logger.info(f"BS position: {bs_position}")
            
            # Load simulation parameters
            params = dict(f['simulation_params'].attrs)
            logger.info(f"Simulation parameters: {params}")
            
            # Check if this is split data
            if 'split_type' in f.attrs:
                split_info = dict(f.attrs)
                logger.info(f"Data split info: {split_info}")
        
        # Convert to tensors
        self.ue_positions = torch.tensor(ue_positions, dtype=torch.float32)
        self.csi_data = torch.tensor(csi_data, dtype=torch.complex64)
        self.bs_position = torch.tensor(bs_position, dtype=torch.float32)
        
        # Create dataset
        self.dataset = TensorDataset(self.ue_positions, self.csi_data)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Training data loaded: {len(self.dataset)} samples, batch_size={self.batch_size}")
        
    def _train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (ue_pos, csi_target) in enumerate(self.dataloader):
            ue_pos = ue_pos.to(self.device)
            csi_target = csi_target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                # Get model output
                csi_pred = self.model(ue_pos)
                
                # Compute loss (handle complex numbers)
                if csi_pred.dtype == torch.complex64:
                    # Convert complex to real for loss calculation
                    csi_pred_real = torch.cat([csi_pred.real, csi_pred.imag], dim=-1)
                    csi_target_real = torch.cat([csi_target.real, csi_target.imag], dim=-1)
                    loss = self.criterion(csi_pred_real, csi_target_real)
                else:
                    loss = self.criterion(csi_pred, csi_target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.dataloader)}, Loss: {loss.item():.6f}")
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _validate(self, epoch: int):
        """Validate model on a subset of data"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Use a subset for validation
        val_size = min(100, len(self.dataset))
        val_indices = torch.randperm(len(self.dataset))[:val_size]
        
        with torch.no_grad():
            for i in range(0, val_size, self.batch_size):
                batch_indices = val_indices[i:i+self.batch_size]
                ue_pos = self.ue_positions[batch_indices].to(self.device)
                csi_target = self.csi_data[batch_indices].to(self.device)
                
                try:
                    csi_pred = self.model(ue_pos)
                    
                    if csi_pred.dtype == torch.complex64:
                        csi_pred_real = torch.cat([csi_pred.real, csi_pred.imag], dim=-1)
                        csi_target_real = torch.cat([csi_target.real, csi_target.imag], dim=-1)
                        loss = self.criterion(csi_pred_real, csi_target_real)
                    else:
                        loss = self.criterion(csi_pred, csi_target)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Validation error: {e}")
                    continue
        
        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_val_loss
    
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_model_path)
            logger.info(f"Best model saved: {best_model_path}")
    
    def _log_metrics(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        """Log metrics to tensorboard"""
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Log model parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Load data
        self._load_data()
        
        # Training history
        train_losses = []
        val_losses = []
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_loss = self._train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self._validate(epoch)
            val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
            self._log_metrics(epoch, train_loss, val_loss, current_lr)
            
            # Save checkpoint
            if epoch % self.save_interval == 0 or epoch == self.num_epochs:
                self._save_checkpoint(epoch, train_loss, val_loss)
            
            # Early stopping check
            if current_lr < 1e-7:
                logger.info("Learning rate too low, stopping training")
                break
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': self.config
        }
        
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training curves
        self._plot_training_curves(train_losses, val_losses)
        
        logger.info("Training completed!")
        self.writer.close()
    
    def _plot_training_curves(self, train_losses: list, val_losses: list):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved: {plot_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Prism Network')
    parser.add_argument('--config', type=str, default='configs/ofdm-5g-sionna.yml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data HDF5 file')
    parser.add_argument('--output', type=str, default='results/training',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = PrismTrainer(args.config, args.data, args.output)
    trainer.train()

if __name__ == '__main__':
    main()
