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
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from prism.networks.prism_network import PrismNetwork
from prism.ray_tracer import DiscreteRayTracer
from prism.training_interface import PrismTrainingInterface

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
    """Main trainer class for Prism network using TrainingInterface"""
    
    def __init__(self, config_path: str, data_path: str, output_dir: str, resume_from: str = None):
        """Initialize trainer with configuration and data paths"""
        self.config_path = config_path
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from = resume_from
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and training components
        self._setup_model()
        self._setup_training()
        
        # Resume from checkpoint if specified
        if self.resume_from:
            self._resume_from_checkpoint()
        else:
            # Auto-detect checkpoint if available
            auto_checkpoint = self._auto_detect_checkpoint()
            if auto_checkpoint:
                print(f"🔄 Auto-resuming from checkpoint: {auto_checkpoint}")
                self.resume_from = auto_checkpoint
                self._resume_from_checkpoint()
            else:
                print("🆕 Starting fresh training (no checkpoints found)")
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        
        # Display checkpoint information
        self._display_checkpoint_info()
        
    def _load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
    
    def _setup_model(self):
        """Initialize Prism network model and TrainingInterface"""
        nn_config = self.config['neural_networks']
        rt_config = self.config['ray_tracing']
        
        # Create PrismNetwork with configuration from YAML
        self.prism_network = PrismNetwork(
            num_subcarriers=nn_config['attenuation_decoder']['output_dim'],
            num_ue_antennas=nn_config['attenuation_decoder']['num_ue'],
            num_bs_antennas=nn_config['antenna_codebook']['num_antennas'],
            position_dim=nn_config['attenuation_network']['input_dim'],
            hidden_dim=nn_config['attenuation_network']['hidden_dim'],
            feature_dim=nn_config['attenuation_network']['feature_dim'],
            antenna_embedding_dim=nn_config['antenna_codebook']['embedding_dim'],
            use_antenna_codebook=nn_config['antenna_codebook']['learnable'],
            use_ipe_encoding=True,  # Enable IPE encoding for better performance
            azimuth_divisions=rt_config['azimuth_divisions'],
            elevation_divisions=rt_config['elevation_divisions'],
            top_k_directions=32,  # Top-K directions for importance sampling
            complex_output=True
        )
        
        # Create DiscreteRayTracer
        self.ray_tracer = DiscreteRayTracer(
            azimuth_divisions=rt_config['azimuth_divisions'],
            elevation_divisions=rt_config['elevation_divisions'],
            max_ray_length=rt_config.get('max_ray_length', 100.0),
            scene_size=rt_config.get('scene_size', 200.0),
            device=self.device.type,
            signal_threshold=rt_config.get('signal_threshold', 1e-6),
            enable_early_termination=rt_config.get('enable_early_termination', True)
        )
        
        # Create PrismTrainingInterface
        self.model = PrismTrainingInterface(
            prism_network=self.prism_network,
            ray_tracer=self.ray_tracer,
            num_sampling_points=rt_config.get('spatial_sampling', 64),
            scene_bounds=rt_config.get('scene_bounds', None),
            subcarrier_sampling_ratio=rt_config.get('subcarrier_sampling_ratio', 0.3),
            checkpoint_dir=str(self.output_dir / 'checkpoints')
        )
        
        self.model = self.model.to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"TrainingInterface created with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
    def _setup_training(self):
        """Setup training hyperparameters and optimizers"""
        # Training hyperparameters
        self.batch_size = self.config['performance']['batch_size']
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.save_interval = 10
        
        # Loss function for complex-valued outputs
        self.criterion = nn.MSELoss()
        
        # Optimizer - now optimizing the TrainingInterface
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
        print(f"📂 Loading data from: {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            # Load UE positions
            ue_positions = f['ue_positions'][:]
            logger.info(f"Loaded {len(ue_positions)} UE positions")
            print(f"   📍 UE positions: {len(ue_positions)} samples")
            
            # Load channel responses (CSI)
            csi_data = f['channel_responses'][:]
            logger.info(f"Loaded CSI data with shape: {csi_data.shape}")
            print(f"   📡 CSI data: {csi_data.shape}")
            
            # Load BS position
            bs_position = f['bs_position'][:]
            logger.info(f"BS position: {bs_position}")
            print(f"   🏢 BS position: {bs_position}")
            
            # Load antenna indices if available
            if 'antenna_indices' in f:
                antenna_indices = f['antenna_indices'][:]
                logger.info(f"Loaded antenna indices with shape: {antenna_indices.shape}")
                print(f"   📡 Antenna indices: {antenna_indices.shape}")
            else:
                # Create default antenna indices if not available
                num_bs_antennas = csi_data.shape[1] if len(csi_data.shape) > 1 else 1
                antenna_indices = np.arange(num_bs_antennas)
                logger.info(f"Created default antenna indices: {antenna_indices}")
                print(f"   📡 Created default antenna indices: {len(antenna_indices)}")
            
            # Load simulation parameters
            params = dict(f['simulation_params'].attrs)
            logger.info(f"Simulation parameters: {params}")
            print(f"   ⚙️  Simulation parameters loaded")
            
            # Check if this is split data
            if 'split_type' in f.attrs:
                split_info = dict(f.attrs)
                logger.info(f"Data split info: {split_info}")
                print(f"   📊 Data split: {split_info.get('split_type', 'unknown')} ({split_info.get('num_samples', 'unknown')} samples)")
        
        # Convert to tensors
        self.ue_positions = torch.tensor(ue_positions, dtype=torch.float32)
        self.csi_data = torch.tensor(csi_data, dtype=torch.complex64)
        self.bs_position = torch.tensor(bs_position, dtype=torch.float32)
        self.antenna_indices = torch.tensor(antenna_indices, dtype=torch.long)
        
        # Create dataset with all required data
        self.dataset = TensorDataset(
            self.ue_positions, 
            self.bs_position.expand(len(ue_positions), -1),
            self.antenna_indices.expand(len(ue_positions), -1),
            self.csi_data
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Training data loaded: {len(self.dataset)} samples, batch_size={self.batch_size}")
        print(f"✅ Data loaded successfully!")
        print(f"   📊 Dataset size: {len(self.dataset)} samples")
        print(f"   🔄 Batch size: {self.batch_size}")
        print(f"   📦 Number of batches: {len(self.dataloader)}")
        print(f"   💾 Data types: UE (float32), CSI (complex64), BS (float32), Antenna (long)")
        
    def _train_epoch(self, epoch: int):
        """Train for one epoch using TrainingInterface"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        print(f"\n🔄 Training Epoch {epoch}")
        print(f"{'='*60}")
        
        for batch_idx, (ue_pos, bs_pos, antenna_idx, csi_target) in enumerate(self.dataloader):
            ue_pos = ue_pos.to(self.device)
            bs_pos = bs_pos.to(self.device)
            antenna_idx = antenna_idx.to(self.device)
            csi_target = csi_target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                # Use TrainingInterface forward pass
                outputs = self.model(
                    ue_positions=ue_pos,
                    bs_position=bs_pos,
                    antenna_indices=antenna_idx
                )
                
                # Extract CSI predictions
                csi_pred = outputs['csi_predictions']
                
                # Compute loss using TrainingInterface's loss computation
                loss = self.model.compute_loss(csi_pred, csi_target, self.criterion)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update training state in TrainingInterface
                self.model.update_training_state(epoch, batch_idx, loss.item())
                
                # Enhanced progress logging
                if batch_idx % 5 == 0 or batch_idx == len(self.dataloader) - 1:
                    progress = (batch_idx + 1) / len(self.dataloader) * 100
                    avg_loss_so_far = total_loss / num_batches
                    print(f"  📊 Batch {batch_idx+1:3d}/{len(self.dataloader):3d} ({progress:5.1f}%) | "
                          f"Loss: {loss.item():.6f} | Avg: {avg_loss_so_far:.6f}")
                    
                    # Log to tensorboard
                    self.writer.add_scalar(f'Loss/Batch_{epoch}', loss.item(), batch_idx)
                    
            except Exception as e:
                logger.error(f"❌ Error in batch {batch_idx}: {e}")
                print(f"  ❌ Batch {batch_idx+1} failed: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"✅ Epoch {epoch} Training Complete | Avg Loss: {avg_loss:.6f}")
        return avg_loss
    
    def _validate(self, epoch: int):
        """Validate model on a subset of data using TrainingInterface"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        print(f"🔍 Validating Epoch {epoch}...")
        
        # Use a subset for validation
        val_size = min(100, len(self.dataset))
        val_indices = torch.randperm(len(self.dataset))[:val_size]
        
        with torch.no_grad():
            for i in range(0, val_size, self.batch_size):
                batch_indices = val_indices[i:i+self.batch_size]
                ue_pos = self.ue_positions[batch_indices].to(self.device)
                bs_pos = self.bs_position.expand(len(batch_indices), -1).to(self.device)
                antenna_idx = self.antenna_indices.expand(len(batch_indices), -1).to(self.device)
                csi_target = self.csi_data[batch_indices].to(self.device)
                
                try:
                    # Use TrainingInterface forward pass
                    outputs = self.model(
                        ue_positions=ue_pos,
                        bs_position=bs_pos,
                        antenna_indices=antenna_idx
                    )
                    
                    csi_pred = outputs['csi_predictions']
                    loss = self.model.compute_loss(csi_pred, csi_target, self.criterion)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"❌ Validation error: {e}")
                    print(f"  ❌ Validation batch failed: {e}")
                    continue
        
        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"✅ Validation Complete | Val Loss: {avg_val_loss:.6f}")
        return avg_val_loss
    
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        """Save model checkpoint using TrainingInterface"""
        print(f"💾 Saving checkpoint for epoch {epoch}...")
        
        # Save TrainingInterface checkpoint
        self.model.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # Save additional training state
        training_state = {
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config,
            'train_losses': getattr(self, 'train_losses', []),
            'val_losses': getattr(self, 'val_losses', []),
            'best_val_loss': getattr(self, 'best_val_loss', float('inf')),
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.output_dir / f'training_state_epoch_{epoch}.pt'
        torch.save(training_state, checkpoint_path)
        logger.info(f"Training state saved: {checkpoint_path}")
        
        # Save best model
        if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = self.output_dir / 'best_model.pt'
            self.model.save_checkpoint('best_model.pt')
            torch.save(training_state, str(best_model_path).replace('.pt', '_state.pt'))
            logger.info(f"Best model saved: {best_model_path}")
            print(f"🏆 New best model saved! (Val Loss: {val_loss:.6f})")
        
        # Save latest checkpoint for resuming
        latest_checkpoint_path = self.output_dir / 'latest_checkpoint.pt'
        self.model.save_checkpoint('latest_checkpoint.pt')
        torch.save(training_state, str(latest_checkpoint_path).replace('.pt', '_state.pt'))
        logger.info(f"Latest checkpoint saved: {latest_checkpoint_path}")
        
        # Save emergency checkpoint every epoch for better recovery
        emergency_checkpoint_path = self.output_dir / 'emergency_checkpoint.pt'
        self.model.save_checkpoint('emergency_checkpoint.pt')
        torch.save(training_state, str(emergency_checkpoint_path).replace('.pt', '_state.pt'))
        
        print(f"✅ Checkpoint saved: Epoch {epoch}, Loss: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        # Clean up old checkpoints (keep last 5)
        self._cleanup_old_checkpoints()
    
    def _save_emergency_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        """Save emergency checkpoint for quick recovery"""
        try:
            # Save TrainingInterface emergency checkpoint
            self.model.save_checkpoint('emergency_checkpoint.pt')
            
            # Save minimal training state for emergency recovery
            emergency_state = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat(),
                'emergency': True
            }
            
            emergency_path = self.output_dir / 'emergency_checkpoint_state.pt'
            torch.save(emergency_state, emergency_path)
            
        except Exception as e:
            logger.warning(f"Emergency checkpoint failed: {e}")
    
    def _auto_detect_checkpoint(self):
        """Automatically detect the best checkpoint to resume from"""
        print("🔍 Auto-detecting checkpoints...")
        
        # Priority order for checkpoint detection
        checkpoint_candidates = [
            self.output_dir / 'emergency_checkpoint.pt',  # Most recent
            self.output_dir / 'latest_checkpoint.pt',    # Latest epoch
            self.output_dir / 'best_model.pt'            # Best performance
        ]
        
        for checkpoint_path in checkpoint_candidates:
            if checkpoint_path.exists():
                print(f"✅ Found checkpoint: {checkpoint_path}")
                return str(checkpoint_path)
        
        # Check for epoch-specific checkpoints
        checkpoint_dir = self.output_dir / 'checkpoints'
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if checkpoints:
                # Get the latest epoch checkpoint
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
                print(f"✅ Found epoch checkpoint: {latest_checkpoint}")
                return str(latest_checkpoint)
        
        print("❌ No checkpoints found")
        return None
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to save disk space"""
        # Clean up TrainingInterface checkpoints
        checkpoint_dir = Path(self.model.checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > 5:
            # Sort by epoch number and remove oldest
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for checkpoint in checkpoints[:-5]:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
        
        # Clean up training state files
        training_states = list(self.output_dir.glob('training_state_epoch_*.pt'))
        if len(training_states) > 5:
            training_states.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for state in training_states[:-5]:
                state.unlink()
                logger.info(f"Removed old training state: {state}")
    
    def _resume_from_checkpoint(self):
        """Resume training from a checkpoint using TrainingInterface"""
        logger.info(f"Resuming training from checkpoint: {self.resume_from}")
        
        if not os.path.exists(self.resume_from):
            logger.error(f"Checkpoint file not found: {self.resume_from}")
            raise FileNotFoundError(f"Checkpoint file not found: {self.resume_from}")
        
        try:
            # Load TrainingInterface checkpoint
            self.model.load_checkpoint(self.resume_from)
            logger.info("TrainingInterface checkpoint loaded successfully")
            
            # Load training state if available
            training_state_path = self.resume_from.replace('.pt', '_state.pt')
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path, map_location=self.device)
                
                # Load optimizer state
                self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
                logger.info("Optimizer state loaded successfully")
                
                # Load scheduler state
                self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
                logger.info("Scheduler state loaded successfully")
                
                # Load training state
                self.start_epoch = training_state['epoch'] + 1
                self.best_val_loss = training_state.get('val_loss', float('inf'))
                
                logger.info(f"Resuming from epoch {self.start_epoch}")
                logger.info(f"Best validation loss so far: {self.best_val_loss:.6f}")
                
                # Load training history if available
                if 'train_losses' in training_state:
                    self.train_losses = training_state['train_losses']
                    self.val_losses = training_state['val_losses']
                    logger.info(f"Loaded training history: {len(self.train_losses)} epochs")
                else:
                    self.train_losses = []
                    self.val_losses = []
            else:
                # Fallback to TrainingInterface state
                self.start_epoch = self.model.current_epoch + 1
                self.best_val_loss = self.model.best_loss
                self.train_losses = []
                self.val_losses = []
                logger.info(f"Resuming from TrainingInterface state: epoch {self.start_epoch}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
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
        
        # Log TrainingInterface specific metrics
        training_info = self.model.get_training_info()
        self.writer.add_scalar('Training/Current_Epoch', training_info['current_epoch'], epoch)
        self.writer.add_scalar('Training/Best_Loss', training_info['best_loss'], epoch)
        self.writer.add_scalar('Training/History_Length', training_info['training_history_length'], epoch)
    
    def train(self):
        """Main training loop using TrainingInterface"""
        logger.info("Starting training with TrainingInterface...")
        print("\n🚀 Starting Prism Network Training with TrainingInterface")
        print("=" * 80)
        
        # Load data
        print("📂 Loading training data...")
        self._load_data()
        print(f"✅ Data loaded: {len(self.dataset)} samples")
        
        # Initialize training state
        if hasattr(self, 'start_epoch'):
            start_epoch = self.start_epoch
            train_losses = self.train_losses
            val_losses = self.val_losses
            logger.info(f"Resuming training from epoch {start_epoch}")
            print(f"🔄 Resuming training from epoch {start_epoch}")
        else:
            start_epoch = 1
            train_losses = []
            val_losses = []
            logger.info("Starting training from epoch 1")
            print("🆕 Starting training from epoch 1")
        
        print(f"\n📈 Training Configuration:")
        print(f"   • Total epochs: {self.num_epochs}")
        print(f"   • Batch size: {self.batch_size}")
        print(f"   • Learning rate: {self.learning_rate:.2e}")
        print(f"   • Device: {self.device}")
        print(f"   • Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epochs + 1):
            epoch_start_time = time.time()
            
            print(f"\n{'='*80}")
            print(f"🎯 EPOCH {epoch}/{self.num_epochs}")
            print(f"{'='*80}")
            
            # Train
            train_loss = self._train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self._validate(epoch)
            val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Calculate timing
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            avg_epoch_time = total_time / (epoch - start_epoch + 1)
            eta = avg_epoch_time * (self.num_epochs - epoch)
            
            # Enhanced epoch summary
            print(f"\n📊 EPOCH {epoch} SUMMARY:")
            print(f"   • Training Loss: {train_loss:.6f}")
            print(f"   • Validation Loss: {val_loss:.6f}")
            print(f"   • Learning Rate: {current_lr:.2e}")
            print(f"   • Epoch Time: {epoch_time:.1f}s")
            print(f"   • Total Time: {total_time/3600:.1f}h")
            print(f"   • ETA: {eta/3600:.1f}h")
            
            # Log metrics
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
            self._log_metrics(epoch, train_loss, val_loss, current_lr)
            
            # Save checkpoint
            if epoch % self.save_interval == 0 or epoch == self.num_epochs:
                print(f"💾 Saving checkpoint for epoch {epoch}...")
                self._save_checkpoint(epoch, train_loss, val_loss)
                print(f"✅ Checkpoint saved successfully")
            
            # Save emergency checkpoint every epoch for better recovery
            if epoch % 1 == 0:  # Every epoch
                self._save_emergency_checkpoint(epoch, train_loss, val_loss)
            
            # Early stopping check
            if current_lr < 1e-7:
                print(f"⚠️  Learning rate too low ({current_lr:.2e}), stopping training")
                logger.info("Learning rate too low, stopping training")
                break
            
            print(f"{'='*80}")
        
        # Training completion summary
        total_training_time = time.time() - start_time
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"{'='*80}")
        print(f"   • Total epochs completed: {len(train_losses)}")
        print(f"   • Total training time: {total_training_time/3600:.2f} hours")
        print(f"   • Final training loss: {train_losses[-1]:.6f}")
        print(f"   • Final validation loss: {val_losses[-1]:.6f}")
        print(f"   • Best validation loss: {min(val_losses):.6f}")
        print(f"   • Results saved to: {self.output_dir}")
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': self.config,
            'training_interface_info': self.model.get_training_info(),
            'total_training_time': total_training_time,
            'final_epoch': len(train_losses)
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

    def _display_checkpoint_info(self):
        """Display information about available checkpoint files."""
        print("\n📁 Available Checkpoint Files:")
        print("=" * 20)
        
        # Display TrainingInterface checkpoints
        checkpoint_dir = Path(self.model.checkpoint_dir)
        if checkpoint_dir.exists():
            print(f"TrainingInterface Checkpoints (in {self.model.checkpoint_dir}):")
            checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if checkpoints:
                print(f"  - Latest: {checkpoints[-1]}")
                print(f"  - Total: {len(checkpoints)}")
            else:
                print("  - No TrainingInterface checkpoints found.")
        else:
            print(f"TrainingInterface Checkpoints directory not found: {self.model.checkpoint_dir}")

        # Display emergency checkpoint
        emergency_checkpoint_path = self.output_dir / 'emergency_checkpoint.pt'
        if emergency_checkpoint_path.exists():
            print(f"\nEmergency Checkpoint (in {self.output_dir}):")
            print(f"  - Path: {emergency_checkpoint_path}")
        else:
            print(f"\nEmergency Checkpoint (in {self.output_dir}):")
            print("  - Not found.")

        # Display latest checkpoint
        latest_checkpoint_path = self.output_dir / 'latest_checkpoint.pt'
        if latest_checkpoint_path.exists():
            print(f"\nLatest Checkpoint (in {self.output_dir}):")
            print(f"  - Path: {latest_checkpoint_path}")
        else:
            print(f"\nLatest Checkpoint (in {self.output_dir}):")
            print("  - Not found.")

        # Display best model
        best_model_path = self.output_dir / 'best_model.pt'
        if best_model_path.exists():
            print(f"\nBest Model (in {self.output_dir}):")
            print(f"  - Path: {best_model_path}")
        else:
            print(f"\nBest Model (in {self.output_dir}):")
            print("  - Not found.")

        print("=" * 20)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Prism Network using TrainingInterface')
    parser.add_argument('--config', type=str, default='configs/ofdm-5g-sionna.yml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data HDF5 file')
    parser.add_argument('--output', type=str, default='results/training',
                       help='Output directory for results')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = PrismTrainer(args.config, args.data, args.output, args.resume)
    trainer.train()

if __name__ == '__main__':
    main()
