#!/usr/bin/env python3
"""
Prism Runner: Main script for training and testing the Prism model.
Extends NeRF2 for wideband RF signals in OFDM scenarios.

This script provides a complete training and testing pipeline for the Prism model,
which is designed to handle wideband RF signals with multiple subcarriers in
OFDM communication systems. It includes data loading, model training, validation,
testing, and comprehensive visualization capabilities.

Key Features:
- Training pipeline with configurable hyperparameters
- Validation and testing with multiple metrics
- Checkpoint saving and loading
- Comprehensive visualization of results
- Support for different dataset types (OFDM, BLE, RFID)
- GPU acceleration support
- Configurable training parameters via YAML files
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

# Import Prism-specific modules
from prism.model import PrismModel, PrismLoss, create_prism_model
from prism.dataloader import PrismDataset
from prism.renderer import PrismRenderer

# Set up logging configuration for tracking training progress and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrismTrainer:
    """
    Trainer class for the Prism model.
    
    This class handles the complete training pipeline including:
    - Model initialization with configurable parameters
    - Training loop with gradient descent optimization
    - Validation during training
    - Checkpoint management
    - Learning rate scheduling
    - Gradient clipping for stability
    - Progress tracking and visualization
    
    The trainer is designed to work with the Prism model's unique architecture
    that processes wideband RF signals across multiple subcarriers.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        """
        Initialize the Prism trainer.
        
        Args:
            config: Configuration dictionary containing model and training parameters
            device: Device to run training on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device
        
        # Create and initialize the Prism model based on configuration
        # The model will be moved to the specified device (GPU/CPU)
        self.model = create_prism_model(config['model']).to(device)
        
        # Initialize the frequency-aware loss function
        # This loss function handles the multi-subcarrier nature of OFDM signals
        self.criterion = PrismLoss(loss_type=config.get('loss_type', 'mse'))
        
        # Initialize the Adam optimizer with configurable learning rate and weight decay
        # Adam is chosen for its adaptive learning rate and good convergence properties
        # Ensure numeric types for optimizer parameters to avoid type errors
        lr = float(config['training']['learning_rate'])
        weight_decay = float(config['training'].get('weight_decay', 1e-5))
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler for adaptive learning rate adjustment
        # StepLR reduces learning rate by a factor every specified number of steps
        # This helps with convergence and prevents overfitting
        # Ensure numeric types for scheduler parameters
        lr_step_size = int(config['training'].get('lr_step_size', 1000))
        lr_gamma = float(config['training'].get('lr_gamma', 0.9))
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_step_size,
            gamma=lr_gamma
        )
        
        # Initialize training state variables for tracking progress
        self.current_epoch = 0
        self.best_loss = float('inf')  # Track best validation loss for model selection
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the model for one complete epoch.
        
        This method implements the core training loop for a single epoch:
        - Iterates through all training batches
        - Performs forward pass through the Prism model
        - Computes frequency-aware loss across all subcarriers
        - Performs backward pass with gradient computation
        - Updates model parameters using the optimizer
        - Applies gradient clipping if configured
        - Updates learning rate scheduler
        
        Args:
            train_loader: DataLoader providing training batches with:
                - positions: 3D spatial coordinates [batch_size, 3]
                - ue_antennas: User equipment antenna features [batch_size, num_ue_antennas]
                - bs_antennas: Base station antenna features [batch_size, num_bs_antennas]
                - additional_features: RF-specific features [batch_size, 10]
                - subcarrier_responses: Ground truth responses [batch_size, num_subcarriers]
            
        Returns:
            Average training loss for the epoch (float)
        """
        # Set model to training mode to enable gradient computation and dropout
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create progress bar for monitoring training progress
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move all batch data to the specified device (GPU/CPU)
            # This ensures data and model are on the same device
            positions = batch['positions'].to(self.device)
            ue_antennas = batch['ue_antennas'].to(self.device)
            bs_antennas = batch['bs_antennas'].to(self.device)
            additional_features = batch['additional_features'].to(self.device)
            targets = batch['subcarrier_responses'].to(self.device)
            
            # Forward pass through the Prism model
            # The model processes wideband RF signals and outputs subcarrier responses
            self.optimizer.zero_grad()  # Clear gradients from previous iteration
            outputs = self.model(positions, ue_antennas, bs_antennas, additional_features)
            predictions = outputs['subcarrier_responses']
            
            # Compute frequency-aware loss across all subcarriers
            # This loss function handles the multi-frequency nature of OFDM signals
            loss = self.criterion(predictions, targets)
            
            # Backward pass: compute gradients with respect to model parameters
            loss.backward()
            
            # Apply gradient clipping if configured to prevent exploding gradients
            # This is especially important for deep networks and helps training stability
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            # Update model parameters using computed gradients
            self.optimizer.step()
            
            # Update training metrics for monitoring and logging
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar with current loss value
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Update learning rate according to scheduler policy
        # This helps with convergence and prevents overfitting
        self.scheduler.step()
        
        # Return average loss for the epoch
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model on the validation dataset.
        
        This method evaluates the model's performance on unseen data:
        - Runs inference without gradient computation (torch.no_grad())
        - Computes validation loss across all subcarriers
        - Provides unbiased estimate of model performance
        - Used for model selection and early stopping decisions
        
        Args:
            val_loader: DataLoader providing validation batches with same structure as training
            
        Returns:
            Average validation loss for the epoch (float)
        """
        # Set model to evaluation mode (disables dropout and batch normalization updates)
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Disable gradient computation for validation (saves memory and computation)
        with torch.no_grad():
            for batch in val_loader:
                # Move batch data to device
                positions = batch['positions'].to(self.device)
                ue_antennas = batch['ue_antennas'].to(self.device)
                bs_antennas = batch['bs_antennas'].to(self.device)
                additional_features = batch['additional_features'].to(self.device)
                targets = batch['subcarrier_responses'].to(self.device)
                
                # Forward pass to get predictions
                outputs = self.model(positions, ue_antennas, bs_antennas, additional_features)
                predictions = outputs['subcarrier_responses']
                
                # Compute validation loss
                loss = self.criterion(predictions, targets)
                
                # Accumulate loss for averaging
                total_loss += loss.item()
                num_batches += 1
        
        # Return average validation loss
        return total_loss / num_batches
    
    def save_checkpoint(self, filepath: str):
        """
        Save a complete model checkpoint including all training state.
        
        This method saves everything needed to resume training:
        - Current epoch number
        - Model parameters (weights and biases)
        - Optimizer state (momentum, learning rate history)
        - Scheduler state (learning rate schedule progress)
        - Best validation loss achieved
        - Configuration used for training
        
        Args:
            filepath: Path where the checkpoint file should be saved
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load a previously saved checkpoint to resume training.
        
        This method restores the complete training state:
        - Loads model parameters from saved state
        - Restores optimizer state for continued optimization
        - Restores scheduler state for learning rate management
        - Sets current epoch and best loss for continued tracking
        
        Args:
            filepath: Path to the checkpoint file to load
        """
        # Load checkpoint from file
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore model parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer state (important for maintaining momentum)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state (important for learning rate management)
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Checkpoint loaded from {filepath}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, checkpoint_dir: str = 'checkpoints'):
        """
        Main training loop that runs for the specified number of epochs.
        
        This method orchestrates the complete training process:
        - Runs training and validation for each epoch
        - Tracks training and validation losses
        - Saves best model based on validation performance
        - Saves regular checkpoints at specified intervals
        - Generates training curves visualization
        - Implements early stopping logic (can be extended)
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Total number of training epochs
            checkpoint_dir: Directory to save model checkpoints
            
        Returns:
            Tuple of (train_losses, val_losses) lists for analysis
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize lists to track training progress
        train_losses = []
        val_losses = []
        
        # Main training loop over all epochs
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase: update model parameters
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase: evaluate model performance
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            # Log progress for monitoring
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model based on validation loss
            # This implements model selection to prevent overfitting
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(os.path.join(checkpoint_dir, 'best_model.pth'))
            
            # Save regular checkpoints at specified intervals
            # This allows resuming training from any point
            if (epoch + 1) % self.config['training'].get('save_freq', 10) == 0:
                self.save_checkpoint(os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth'))
        
        # Generate training curves visualization for analysis
        self.plot_training_curves(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def plot_training_curves(self, train_losses: list, val_losses: list):
        """
        Generate and save training curves visualization.
        
        This method creates plots showing:
        - Training and validation loss over epochs
        - Helps identify overfitting/underfitting
        - Useful for hyperparameter tuning
        - Saves high-quality plots for reports/papers
        
        Args:
            train_losses: List of training losses for each epoch
            val_losses: List of validation losses for each epoch
        """
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
    Tester class for evaluating trained Prism models.
    
    This class handles model evaluation and analysis:
    - Loads trained models from checkpoints
    - Runs inference on test datasets
    - Computes comprehensive evaluation metrics
    - Generates detailed performance analysis
    - Creates visualizations for result interpretation
    
    The tester is designed to provide thorough evaluation of the Prism model's
    performance across different subcarriers and spatial positions.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        """
        Initialize the Prism tester.
        
        Args:
            config: Configuration dictionary containing model parameters
            device: Device to run testing on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device
        
        # Create model instance for testing
        self.model = create_prism_model(config['model']).to(device)
        
        # Create renderer for generating visualizations
        self.renderer = PrismRenderer(config, device)
    
    def load_model(self, checkpoint_path: str):
        """
        Load a trained model from checkpoint file.
        
        This method loads the model parameters from a saved checkpoint
        to enable testing and evaluation of the trained model.
        
        Args:
            checkpoint_path: Path to the checkpoint file containing trained model
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {checkpoint_path}")
    
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the trained model on the test dataset.
        
        This method performs comprehensive testing:
        - Runs inference on all test samples
        - Computes multiple evaluation metrics (MSE, MAE, RMSE)
        - Stores predictions and targets for detailed analysis
        - Provides overall performance summary
        
        Args:
            test_loader: DataLoader providing test batches with same structure as training
            
        Returns:
            Dictionary containing test metrics:
            - mse: Mean Squared Error
            - mae: Mean Absolute Error  
            - rmse: Root Mean Squared Error
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metric accumulators
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        # Lists to store predictions and targets for detailed analysis
        predictions_list = []
        targets_list = []
        
        # Disable gradient computation for testing
        with torch.no_grad():
            # Iterate through all test batches
            for batch in tqdm(test_loader, desc="Testing"):
                # Move batch data to device
                positions = batch['positions'].to(self.device)
                ue_antennas = batch['ue_antennas'].to(self.device)
                bs_antennas = batch['bs_antennas'].to(self.device)
                additional_features = batch['additional_features'].to(self.device)
                targets = batch['subcarrier_responses'].to(self.device)
                
                # Forward pass to get predictions
                outputs = self.model(positions, ue_antennas, bs_antennas, additional_features)
                predictions = outputs['subcarrier_responses']
                
                # Compute evaluation metrics for this batch
                mse = torch.mean((predictions - targets) ** 2)
                mae = torch.mean(torch.abs(predictions - targets))
                
                # Accumulate metrics
                total_mse += mse.item()
                total_mae += mae.item()
                num_batches += 1
                
                # Store predictions and targets for detailed analysis
                predictions_list.append(predictions.cpu())
                targets_list.append(targets.cpu())
        
        # Calculate average metrics across all test samples
        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        
        # Store predictions and targets as instance variables for analysis methods
        self.predictions = torch.cat(predictions_list, dim=0)
        self.targets = torch.cat(targets_list, dim=0)
        
        # Compile final metrics dictionary
        metrics = {
            'mse': avg_mse,
            'mae': avg_mae,
            'rmse': np.sqrt(avg_mse)
        }
        
        # Log test results
        logger.info(f"Test Results: MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}, RMSE: {np.sqrt(avg_mse):.6f}")
        
        return metrics
    
    def analyze_subcarrier_performance(self):
        """
        Analyze model performance across different subcarriers.
        
        This method provides detailed analysis of how well the model performs
        on different frequency components of the OFDM signal:
        - Computes per-subcarrier MSE and MAE
        - Identifies frequency bands where the model excels or struggles
        - Generates visualizations for performance analysis
        - Useful for understanding model behavior across the frequency spectrum
        
        This analysis is crucial for OFDM systems where different subcarriers
        may have different characteristics and importance.
        """
        # Check if test results are available
        if not hasattr(self, 'predictions') or not hasattr(self, 'targets'):
            logger.error("No test results available. Run test() first.")
            return
        
        # Calculate per-subcarrier performance metrics
        # This reveals frequency-dependent performance patterns
        mse_per_subcarrier = torch.mean((self.predictions - self.targets) ** 2, dim=0)
        mae_per_subcarrier = torch.mean(torch.abs(self.predictions - self.targets), dim=0)
        
        # Create comprehensive subcarrier performance visualization
        plt.figure(figsize=(12, 6))
        
        # Plot MSE per subcarrier
        plt.subplot(1, 2, 1)
        plt.plot(mse_per_subcarrier.numpy())
        plt.xlabel('Subcarrier Index')
        plt.ylabel('MSE')
        plt.title('MSE per Subcarrier')
        plt.grid(True)
        
        # Plot MAE per subcarrier
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
    """
    Main function that orchestrates the entire Prism training/testing pipeline.
    
    This function:
    - Parses command line arguments
    - Loads configuration from YAML files
    - Sets up data loaders for training/validation/testing
    - Initializes trainer or tester based on mode
    - Executes the requested operation (training or testing)
    - Handles error cases and provides user feedback
    
    The main function serves as the entry point for the Prism system and
    coordinates all the components for end-to-end operation.
    """
    # Set up command line argument parser
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
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Load configuration from YAML file
    # The config file contains all model and training parameters
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device for computation (GPU or CPU)
    # GPU acceleration significantly speeds up training and inference
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create data loaders for training, validation, and testing
    # Each dataset is split appropriately for machine learning best practices
    train_dataset = PrismDataset(config['data'], split='train')
    val_dataset = PrismDataset(config['data'], split='val')
    test_dataset = PrismDataset(config['data'], split='test')
    
    # Create DataLoader instances with appropriate batch sizes and workers
    # Shuffling is enabled for training to improve convergence
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                             shuffle=True, num_workers=config['data'].get('num_workers', 4))
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                           shuffle=False, num_workers=config['data'].get('num_workers', 4))
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=config['data'].get('num_workers', 4))
    
    if args.mode == 'train':
        # Training mode: initialize trainer and run training loop
        logger.info("Initializing Prism trainer...")
        trainer = PrismTrainer(config, device)
        
        # Load checkpoint if specified (for resuming training)
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
        
        # Execute training loop
        logger.info("Starting training...")
        train_losses, val_losses = trainer.train(
            train_loader, val_loader,
            num_epochs=config['training']['num_epochs'],
            checkpoint_dir=config['training'].get('checkpoint_dir', 'checkpoints')
        )
        
        logger.info("Training completed successfully!")
        
    elif args.mode == 'test':
        # Testing mode: initialize tester and run evaluation
        logger.info("Initializing Prism tester...")
        tester = PrismTester(config, device)
        
        # Load trained model from checkpoint
        if args.checkpoint:
            tester.load_model(args.checkpoint)
        else:
            logger.error("Checkpoint path required for testing mode")
            return
        
        # Execute testing and analysis
        logger.info("Starting testing...")
        metrics = tester.test(test_loader)
        
        # Perform detailed subcarrier performance analysis
        logger.info("Analyzing subcarrier performance...")
        tester.analyze_subcarrier_performance()
        
        logger.info("Testing completed successfully!")

if __name__ == '__main__':
    # Entry point for the script
    # This ensures the main function only runs when the script is executed directly
    main()
