#!/usr/bin/env python3
"""
Sionna 5G OFDM Data Runner for Prism
Complete training and evaluation pipeline using Sionna-generated simulation data.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from prism.utils.sionna_data_loader import SionnaDataLoader
from prism.model import PrismModel, create_prism_model, PrismLoss
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class SionnaTrainer:
    """Trainer class for Sionna data with Prism model."""
    
    def __init__(self, config: dict, device: str = 'cuda'):
        """Initialize the trainer."""
        self.config = config
        self.device = device
        
        # Initialize data loader
        self.data_loader = SionnaDataLoader(config)
        
        # Create data splits using config ratios
        data_config = config.get('data', {})
        train_ratio = data_config.get('train_ratio', 0.8)
        val_ratio = data_config.get('val_ratio', 0.0)
        test_ratio = data_config.get('test_ratio', 0.2)
        self.splits = self.data_loader.get_data_split(train_ratio, val_ratio, test_ratio)
        
        # Store splits in data_loader for batch loading
        self.data_loader.splits = self.splits
        
        # Initialize model
        self.model = create_prism_model(config).to(device)
        
        # Temporarily disable CSI processing to avoid device mismatch issues
        if hasattr(self.model, 'csi_processor'):
            self.model.csi_processor = None
            logger.info("CSI processing disabled for training")
        
        # Initialize loss function
        loss_config = config.get('loss', {})
        self.criterion = PrismLoss(loss_type=loss_config.get('loss_type', 'mse'))
        
        # Initialize optimizer
        training_config = config.get('training', {})
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 0.001),
            weight_decay=training_config.get('weight_decay', 1e-5)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=training_config.get('lr_step_size', 200),
            gamma=training_config.get('lr_gamma', 0.9)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, batch_size: int = 32) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Calculate number of batches
        train_data = self.splits['train']
        num_samples = len(train_data['indices'])
        num_batches_total = (num_samples + batch_size - 1) // batch_size
        
        progress_bar = tqdm(range(num_batches_total), desc=f"Epoch {self.current_epoch}")
        
        for batch_idx in progress_bar:
            # Get batch data
            batch = self.data_loader.get_batch('train', batch_size, batch_idx)
            
            # Move to device
            positions = batch['ue_positions'].to(self.device)
            channel_responses = batch['channel_responses'].to(self.device)
            
            # Create additional features (simplified)
            ue_antennas = torch.randn(positions.shape[0], self.config['model']['num_ue_antennas']).to(self.device)
            bs_antennas = torch.randn(positions.shape[0], self.config['model']['num_bs_antennas']).to(self.device)
            additional_features = torch.randn(positions.shape[0], 10).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(positions, ue_antennas, bs_antennas, additional_features)
            
            # Prepare targets (use channel responses as targets)
            # Reshape channel responses to match model output
            targets = channel_responses.view(positions.shape[0], -1)  # [batch_size, 408*4*64]
            predictions = outputs['subcarrier_responses']  # [batch_size, 408]
            
            # Calculate loss
            loss = self.criterion(
                predictions, 
                targets[:, :self.config['model']['num_subcarriers']],  # Use first 408 dimensions
                config=self.config
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('training', {}).get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, batch_size: int = 32) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        val_data = self.splits['val']
        num_samples = len(val_data['indices'])
        num_batches_total = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_batches_total):
                # Get batch data
                batch = self.data_loader.get_batch('val', batch_size, batch_idx)
                
                # Move to device
                positions = batch['ue_positions'].to(self.device)
                channel_responses = batch['channel_responses'].to(self.device)
                
                # Create additional features
                ue_antennas = torch.randn(positions.shape[0], self.config['model']['num_ue_antennas']).to(self.device)
                bs_antennas = torch.randn(positions.shape[0], self.config['model']['num_bs_antennas']).to(self.device)
                additional_features = torch.randn(positions.shape[0], 10).to(self.device)
                
                # Forward pass
                outputs = self.model(positions, ue_antennas, bs_antennas, additional_features)
                
                # Calculate loss
                targets = channel_responses.view(positions.shape[0], -1)
                predictions = outputs['subcarrier_responses']
                
                loss = self.criterion(
                    predictions, 
                    targets[:, :self.config['model']['num_subcarriers']],
                    config=self.config
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, num_epochs: int, batch_size: int = 32, save_dir: str = 'checkpoints'):
        """Train the model."""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Check if validation set exists
        has_validation = 'val' in self.splits and len(self.splits['val']['indices']) > 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch(batch_size)
            
            # Validation (if available)
            if has_validation:
                val_loss = self.validate(batch_size)
                self.val_losses.append(val_loss)
                
                # Log progress with validation
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                           f"Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, "
                           f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Save best model based on validation
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(save_path / 'best_model.pth')
                    logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                # No validation set, use training loss for best model
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                           f"Train Loss: {train_loss:.4f}, "
                           f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Save best model based on training loss
                if train_loss < self.best_val_loss:
                    self.best_val_loss = train_loss
                    self.save_checkpoint(save_path / 'best_model.pth')
                    logger.info(f"New best model saved with training loss: {train_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.get('training', {}).get('save_freq', 50) == 0:
                self.save_checkpoint(save_path / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save final model
        self.save_checkpoint(save_path / 'final_model.pth')
        logger.info("Training completed!")
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        logger.info(f"Checkpoint loaded from {filepath}")

class SionnaEvaluator:
    """Evaluator class for trained models."""
    
    def __init__(self, model: PrismModel, data_loader: SionnaDataLoader, device: str = 'cuda'):
        """Initialize the evaluator."""
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        # Create data splits using config ratios
        data_config = config.get('data', {})
        train_ratio = data_config.get('train_ratio', 0.8)
        val_ratio = data_config.get('val_ratio', 0.0)
        test_ratio = data_config.get('test_ratio', 0.2)
        self.splits = data_loader.get_data_split(train_ratio, val_ratio, test_ratio)
    
    def evaluate(self, split: str = 'test', batch_size: int = 32) -> dict:
        """Evaluate the model on specified split."""
        self.model.eval()
        
        split_data = self.splits[split]
        num_samples = len(split_data['indices'])
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        total_mse = 0.0
        total_mae = 0.0
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {split}"):
                # Get batch data
                batch = self.data_loader.get_batch(split, batch_size, batch_idx)
                
                # Move to device
                positions = batch['ue_positions'].to(self.device)
                channel_responses = batch['channel_responses'].to(self.device)
                
                # Create additional features
                ue_antennas = torch.randn(positions.shape[0], self.model.num_ue_antennas).to(self.device)
                bs_antennas = torch.randn(positions.shape[0], self.model.num_bs_antennas).to(self.device)
                additional_features = torch.randn(positions.shape[0], 10).to(self.device)
                
                # Forward pass
                outputs = self.model(positions, ue_antennas, bs_antennas, additional_features)
                
                # Prepare targets and predictions
                targets = channel_responses.view(positions.shape[0], -1)
                predictions = outputs['subcarrier_responses']
                
                # Calculate metrics
                mse = torch.mean((predictions - targets[:, :self.model.num_subcarriers]) ** 2)
                mae = torch.mean(torch.abs(predictions - targets[:, :self.model.num_subcarriers]))
                
                total_mse += mse.item()
                total_mae += mae.item()
                
                # Store for analysis
                predictions_list.append(predictions.cpu())
                targets_list.append(targets[:, :self.model.num_subcarriers].cpu())
        
        # Calculate average metrics
        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        
        # Compile results
        results = {
            'mse': avg_mse,
            'mae': avg_mae,
            'rmse': np.sqrt(avg_mse),
            'predictions': torch.cat(predictions_list, dim=0),
            'targets': torch.cat(targets_list, dim=0)
        }
        
        return results

def create_visualizations(trainer: SionnaTrainer, evaluator: SionnaEvaluator, save_dir: str = 'results'):
    """Create training and evaluation visualizations."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sionna 5G OFDM Training Results', fontsize=16)
    
    # Plot 1: Training and validation loss
    ax1 = axes[0, 0]
    epochs = range(1, len(trainer.train_losses) + 1)
    ax1.plot(epochs, trainer.train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, trainer.val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss comparison
    ax2 = axes[0, 1]
    ax2.bar(['Train', 'Validation'], [trainer.train_losses[-1], trainer.val_losses[-1]], 
             color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Loss')
    ax2.set_title('Final Loss Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate schedule
    ax3 = axes[1, 0]
    lr_schedule = [group['lr'] for group in trainer.optimizer.param_groups]
    ax3.plot(epochs, lr_schedule, 'g-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model parameters distribution
    ax4 = axes[1, 1]
    param_values = []
    for param in trainer.model.parameters():
        if param.requires_grad:
            param_values.extend(param.data.cpu().numpy().flatten())
    
    ax4.hist(param_values, bins=50, alpha=0.7, color='purple')
    ax4.set_xlabel('Parameter Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Model Parameter Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = save_path / 'training_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training visualizations saved to {output_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Sionna 5G OFDM Training Runner')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo'],
                       help='Mode: train, test, or demo')
    parser.add_argument('--config', type=str, default='configs/ofdm-5g-sionna.yml',
                       help='Configuration file path')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint file path for testing')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/sionna_5g',
                       help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='results/sionna_5g',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        return
    
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Check if Sionna integration is enabled
    if not config.get('sionna_integration', {}).get('enabled', False):
        print("Sionna integration not enabled in configuration.")
        print("Please enable sionna_integration.enabled in the config file.")
        return
    
    if args.mode == 'train':
        # Training mode
        print("=== Training Mode ===")
        
        # Initialize trainer
        trainer = SionnaTrainer(config, device=args.device)
        
        # Train the model
        trainer.train(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=args.save_dir
        )
        
        # Create visualizations
        evaluator = SionnaEvaluator(trainer.model, trainer.data_loader, device=args.device)
        create_visualizations(trainer, evaluator, save_dir=args.results_dir)
        
        print("Training completed successfully!")
        
    elif args.mode == 'test':
        # Testing mode
        print("=== Testing Mode ===")
        
        if not args.checkpoint:
            print("Please provide a checkpoint file for testing.")
            return
        
        # Initialize trainer and load checkpoint
        trainer = SionnaTrainer(config, device=args.device)
        trainer.load_checkpoint(args.checkpoint)
        
        # Initialize evaluator
        evaluator = SionnaEvaluator(trainer.model, trainer.data_loader, device=args.device)
        
        # Evaluate on test set
        results = evaluator.evaluate(split='test', batch_size=args.batch_size)
        
        print("\nTest Results:")
        print(f"  MSE: {results['mse']:.6f}")
        print(f"  MAE: {results['mae']:.6f}")
        print(f"  RMSE: {results['rmse']:.6f}")
        
        # Save results
        results_path = Path(args.results_dir)
        results_path.mkdir(exist_ok=True)
        
        torch.save(results, results_path / 'test_results.pt')
        print(f"Results saved to {results_path / 'test_results.pt'}")
        
    elif args.mode == 'demo':
        # Demo mode (original functionality)
        print("=== Demo Mode ===")
        
        # Initialize data loader
        data_loader = SionnaDataLoader(config)
        
        # Get data splits using config ratios
        data_config = config.get('data', {})
        train_ratio = data_config.get('train_ratio', 0.8)
        val_ratio = data_config.get('val_ratio', 0.0)
        test_ratio = data_config.get('test_ratio', 0.2)
        splits = data_loader.get_data_split(train_ratio, val_ratio, test_ratio)
        
        # Store splits in data_loader for batch loading
        data_loader.splits = splits
        
        # Get a sample batch
        batch = data_loader.get_batch('train', batch_size=8, batch_idx=0)
        
        # Initialize model
        model = create_prism_model(config)
        
        # Run forward pass
        positions = batch['ue_positions']
        ue_antennas = torch.randn(positions.shape[0], config['model']['num_ue_antennas'])
        bs_antennas = torch.randn(positions.shape[0], config['model']['num_bs_antennas'])
        additional_features = torch.randn(positions.shape[0], 10)
        
        model.eval()
        with torch.no_grad():
            outputs = model(positions, ue_antennas, bs_antennas, additional_features)
        
        print("\nDemo Results:")
        print(f"  Model outputs: {len(outputs)}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape}")
        
        print("\nDemo completed successfully!")

if __name__ == '__main__':
    main()
