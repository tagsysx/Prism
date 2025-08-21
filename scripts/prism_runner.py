#!/usr/bin/env python3
"""
Prism Runner: Main script for training and testing the Prism model.
Extends NeRF2 for wideband RF signals in OFDM scenarios.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from prism.model import PrismModel, PrismLoss, create_prism_model
from prism.dataloader import PrismDataset
from prism.renderer import PrismRenderer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrismTrainer:
    """
    Trainer class for the Prism model.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Create model
        self.model = create_prism_model(config['model']).to(device)
        
        # Create loss function
        self.criterion = PrismLoss(loss_type=config.get('loss_type', 'mse'))
        
        # Create optimizer
        # Ensure numeric types for optimizer parameters
        lr = float(config['training']['learning_rate'])
        weight_decay = float(config['training'].get('weight_decay', 1e-5))
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        # Ensure numeric types for scheduler parameters
        lr_step_size = int(config['training'].get('lr_step_size', 1000))
        lr_gamma = float(config['training'].get('lr_gamma', 0.9))
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_step_size,
            gamma=lr_gamma
        )
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move data to device
            positions = batch['positions'].to(self.device)
            ue_antennas = batch['ue_antennas'].to(self.device)
            bs_antennas = batch['bs_antennas'].to(self.device)
            additional_features = batch['additional_features'].to(self.device)
            targets = batch['subcarrier_responses'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(positions, ue_antennas, bs_antennas, additional_features)
            predictions = outputs['subcarrier_responses']
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Update learning rate
        self.scheduler.step()
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                positions = batch['positions'].to(self.device)
                ue_antennas = batch['ue_antennas'].to(self.device)
                bs_antennas = batch['bs_antennas'].to(self.device)
                additional_features = batch['additional_features'].to(self.device)
                targets = batch['subcarrier_responses'].to(self.device)
                
                # Forward pass
                outputs = self.model(positions, ue_antennas, bs_antennas, additional_features)
                predictions = outputs['subcarrier_responses']
                
                # Compute loss
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        logger.info(f"Checkpoint loaded from {filepath}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, checkpoint_dir: str = 'checkpoints'):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            checkpoint_dir: Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(os.path.join(checkpoint_dir, 'best_model.pth'))
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['training'].get('save_freq', 10) == 0:
                self.save_checkpoint(os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth'))
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def plot_training_curves(self, train_losses: list, val_losses: list):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

class PrismTester:
    """
    Tester class for the Prism model.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Create model
        self.model = create_prism_model(config['model']).to(device)
        
        # Create renderer
        self.renderer = PrismRenderer(config, device)
    
    def load_model(self, checkpoint_path: str):
        """Load trained model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {checkpoint_path}")
    
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Test the model.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                # Move data to device
                positions = batch['positions'].to(self.device)
                ue_antennas = batch['ue_antennas'].to(self.device)
                bs_antennas = batch['bs_antennas'].to(self.device)
                additional_features = batch['additional_features'].to(self.device)
                targets = batch['subcarrier_responses'].to(self.device)
                
                # Forward pass
                outputs = self.model(positions, ue_antennas, bs_antennas, additional_features)
                predictions = outputs['subcarrier_responses']
                
                # Compute metrics
                mse = torch.mean((predictions - targets) ** 2)
                mae = torch.mean(torch.abs(predictions - targets))
                
                total_mse += mse.item()
                total_mae += mae.item()
                num_batches += 1
                
                # Store for detailed analysis
                predictions_list.append(predictions.cpu())
                targets_list.append(targets.cpu())
        
        # Calculate average metrics
        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        
        # Store predictions and targets for analysis
        self.predictions = torch.cat(predictions_list, dim=0)
        self.targets = torch.cat(targets_list, dim=0)
        
        metrics = {
            'mse': avg_mse,
            'mae': avg_mae,
            'rmse': np.sqrt(avg_mse)
        }
        
        logger.info(f"Test Results: MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}, RMSE: {np.sqrt(avg_mse):.6f}")
        
        return metrics
    
    def analyze_subcarrier_performance(self):
        """Analyze performance across different subcarriers."""
        if not hasattr(self, 'predictions') or not hasattr(self, 'targets'):
            logger.error("No test results available. Run test() first.")
            return
        
        # Calculate per-subcarrier metrics
        mse_per_subcarrier = torch.mean((self.predictions - self.targets) ** 2, dim=0)
        mae_per_subcarrier = torch.mean(torch.abs(self.predictions - self.targets), dim=0)
        
        # Plot subcarrier performance
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(mse_per_subcarrier.numpy())
        plt.xlabel('Subcarrier Index')
        plt.ylabel('MSE')
        plt.title('MSE per Subcarrier')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(mae_per_subcarrier.numpy())
        plt.xlabel('Subcarrier Index')
        plt.ylabel('MAE')
        plt.title('MAE per Subcarrier')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('subcarrier_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Subcarrier performance analysis completed.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Prism: Wideband RF Neural Radiance Fields')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--dataset_type', type=str, default='ofdm',
                       choices=['ofdm', 'ble', 'rfid'],
                       help='Type of dataset')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for testing')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_dataset = PrismDataset(config['data'], split='train')
    val_dataset = PrismDataset(config['data'], split='val')
    test_dataset = PrismDataset(config['data'], split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                             shuffle=True, num_workers=config['data'].get('num_workers', 4))
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                           shuffle=False, num_workers=config['data'].get('num_workers', 4))
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=config['data'].get('num_workers', 4))
    
    if args.mode == 'train':
        # Training mode
        trainer = PrismTrainer(config, device)
        
        # Load checkpoint if specified
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        
        # Train the model
        train_losses, val_losses = trainer.train(
            train_loader, val_loader,
            num_epochs=config['training']['num_epochs'],
            checkpoint_dir=config['training'].get('checkpoint_dir', 'checkpoints')
        )
        
    elif args.mode == 'test':
        # Testing mode
        tester = PrismTester(config, device)
        
        # Load trained model
        if args.checkpoint:
            tester.load_model(args.checkpoint)
        else:
            logger.error("Checkpoint path required for testing mode")
            return
        
        # Test the model
        metrics = tester.test(test_loader)
        
        # Analyze subcarrier performance
        tester.analyze_subcarrier_performance()
        
        logger.info("Testing completed successfully!")

if __name__ == '__main__':
    main()
