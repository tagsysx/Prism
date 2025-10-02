#!/usr/bin/env python3
"""
Prism Neural Network Training Script

This script provides a general training pipeline for Prism neural network system,
capable of loading different datasets and configurations.

Features:
- Flexible data loading for various dataset formats
- Modern configuration loading with template processing
- Simplified training interface with new PrismNetwork
- Efficient memory management and GPU utilization
- Real-time progress monitoring
- Automatic checkpointing and resumption
- Comprehensive logging and error handling
"""

import os
import sys
import time
import logging
import argparse
import warnings
import h5py
import numpy as np
import torch

# Suppress the specific PyTorch autocast FutureWarning
warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*is deprecated.*")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
from datetime import datetime
import traceback
from typing import Dict, Any, Optional, Tuple, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prism.networks.prism_network import PrismNetwork
from prism.training_interface import PrismTrainingInterface
from prism.config_loader import ModernConfigLoader
from prism.loss.loss_function import LossFunction
from prism.data_utils import create_position_aware_dataloader
from base_runner import BaseRunner


class TrainingProgressMonitor:
    """Lightweight training progress monitor for real-time feedback."""
    
    def __init__(self, total_epochs: int, total_batches_per_epoch: int):
        self.total_epochs = total_epochs
        self.total_batches_per_epoch = total_batches_per_epoch
        self.start_time = time.time()
        self.epoch_start_time = None
        self.current_epoch = 0
        
    def start_epoch(self, epoch: int):
        """Start monitoring a new epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
        print(f"\n🔄 Epoch {epoch}/{self.total_epochs}")
        print(f"{'='*60}")
        print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
        
    def update_batch(self, batch_idx: int, loss: float):
        """Update batch progress."""
        progress = (batch_idx + 1) / self.total_batches_per_epoch * 100
        elapsed = time.time() - self.epoch_start_time
        
        # Simple progress indicator
        bar_length = 30
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r  📊 [{bar}] {progress:5.1f}% | Batch {batch_idx+1:3d}/{self.total_batches_per_epoch:3d} | "
              f"Loss: {loss:.6f} | Time: {elapsed:.1f}s", end="", flush=True)
        
    def end_epoch(self, avg_loss: float):
        """Complete epoch monitoring."""
        epoch_time = time.time() - self.epoch_start_time
        print(f"\n  ✅ Completed in {epoch_time:.1f}s | Average Loss: {avg_loss:.6f}")
        print(f"{'='*60}")


class PrismTrainer(BaseRunner):
    """
    General training pipeline for Prism neural network system.
    
    Flexible training implementation that can handle various dataset formats
    and configurations through the config file system.
    """
    
    def __init__(self, config_path: str, resume_from: Optional[str] = None, new_training: bool = False):
        """
        Initialize trainer with configuration path.
        
        Args:
            config_path: Path to YAML configuration file
            resume_from: Optional path to checkpoint for resuming training
            new_training: If True, clear previous training results
        """
        self.resume_from = resume_from
        self.new_training = new_training
        
        # Initialize base runner
        super().__init__(config_path)
        
        # Extract additional configurations
        self.training_config = self.config_loader.get_training_kwargs()
        self.output_paths = self.config_loader.get_output_paths()
        
        # Setup logging (requires output_paths)
        self._setup_logging()
        
        # Create output directories
        self.config_loader.ensure_output_directories()
        
        # Initialize components
        self.prism_network = None
        self.training_interface = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.current_batch = 0
        self.best_loss = float('inf')
        self.training_start_time = None
    
    def _clear_previous_results(self):
        """Clear previous training results."""
        import shutil
        
        output_paths = self.config_loader.get_output_paths()
        
        # Directories to clear
        dirs_to_clear = [
            'checkpoint_dir',
            'tensorboard_dir', 
            'models_dir',
            'log_dir',
            'results_dir',
            'plots_dir',
            'predictions_dir',
            'reports_dir'
        ]
        
        print("🧹 Clearing previous training results...")
        
        for dir_name in dirs_to_clear:
            dir_path = output_paths[dir_name]
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    print(f"   ✅ Cleared: {dir_path}")
                except Exception as e:
                    print(f"   ⚠️  Failed to clear {dir_path}: {e}")
        
        print("✅ Previous results cleared successfully")
        
        self.logger.info("🚀 PrismTrainer initialized successfully")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   Configuration: {self.config_path}")
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.output_paths['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.output_paths['log_file']
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging configured: {log_file}")
        
    def _create_model(self):
        """Create PrismNetwork model."""
        self.logger.info("🔧 Creating PrismNetwork...")
        
        # Get network configuration
        network_kwargs = self.config_loader.get_prism_network_kwargs()
        
        # Always use BS antenna count for training (regardless of orientation)
        num_antennas = network_kwargs['num_bs_antennas']
        self.logger.info(f"🔧 Using BS antenna array for training: {num_antennas} antennas")
        
        # Keep original antenna count in network kwargs
        # network_kwargs['num_bs_antennas'] = num_antennas  # Already correct
        
        # Create network
        self.prism_network = PrismNetwork(**network_kwargs).to(self.device)
        
        # Create training interface
        config_dict = {
            'system': {'device': str(self.device)},
            'training': self.config_loader.training.__dict__,
            'user_equipment': {
                'num_ue_antennas': self.config_loader.user_equipment.num_ue_antennas,
                # ue_antenna_count removed - single antenna combinations processed per sample
            },
            'input': {
                'subcarrier_sampling': {
                    'sampling_ratio': self.data_config['sampling_ratio'],
                    'sampling_method': self.data_config['sampling_method'],
                    'antenna_consistent': self.data_config['antenna_consistent']
                },
                'calibration': {
                    'enabled': self.data_config['calibration']['enabled'],
                    'reference_subcarrier_index': self.data_config['calibration']['reference_subcarrier_index']
                }
            },
            # Add tracing configuration from raw config
            'tracing': self.config_loader._processed_config.get('tracing', {})
        }
        
        self.training_interface = PrismTrainingInterface(
            prism_network=self.prism_network,
            config=config_dict,
            checkpoint_dir=self.output_paths['checkpoint_dir'],
            device=self.device
        )
        
        # Memory optimizations are no longer needed due to per-sample processing
        self.logger.info("🔧 Memory optimizations no longer needed - using per-sample processing")
        
        # Log model info
        model_info = self.prism_network.get_network_info()
        self.logger.info(f"✅ PrismNetwork created:")
        self.logger.info(f"   Total parameters: {model_info['total_parameters']:,}")
        self.logger.info(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
        self.logger.info(f"   Ray directions: {model_info['ray_directions']}")
        self.logger.info(f"   Subcarriers: {model_info['num_subcarriers']}")
        
    def _create_loss_function(self):
        """Create loss function."""
        self.logger.info("🔧 Creating loss function...")
        
        # Get loss configuration
        loss_config = self.config_loader.get_loss_functions_config()
        
        # Get full config for PAS loss
        full_config = self.config_loader._processed_config
        
        # Add debug_dir to loss configurations
        debug_dir = self.output_paths.get('debug_dir', None)
        
        if 'csi_loss' in loss_config:
            loss_config['csi_loss']['debug_dir'] = debug_dir
            # debug_sample_rate is now read from config file (default: 0.5 if not specified)
        
        if 'pdp_loss' in loss_config:
            loss_config['pdp_loss']['debug_dir'] = debug_dir
            # debug_sample_rate is now read from config file (default: 0.5 if not specified)
        
        if 'pas_loss' in loss_config:
            loss_config['pas_loss']['debug_dir'] = debug_dir
            # debug_sample_rate is now read from config file (default: 0.5 if not specified)
        
        # Create combined loss function with full config for PAS loss
        self.loss_function = LossFunction(config=loss_config, full_config=full_config)
        
        self.logger.info(f"✅ Loss function created with weights: CSI={loss_config.get('csi_weight', 0.7)}, "
                        f"PDP={loss_config.get('pdp_weight', 0.3)}, "
                        f"Spatial={loss_config.get('spatial_spectrum_weight', 0.0)}, "
                        f"Reg={loss_config.get('regularization_weight', 0.01)}")
        
    def _create_optimizer(self):
        """Create optimizer and scheduler."""
        self.logger.info("🔧 Creating optimizer...")
        
        # Create optimizer with different learning rates for different parameter groups
        # Main network parameters
        main_params = list(self.prism_network.parameters())
        
        # Ray tracer parameters (excluding prism_network to avoid duplication)
        ray_tracer_params = self.training_interface.ray_tracer.get_learnable_parameters()
        other_ray_tracer_params = []
        
        if ray_tracer_params:
            # Only include non-prism_network parameters from ray tracer
            for param_name, param_list in ray_tracer_params.items():
                if param_name != 'prism_network':  # Skip prism_network to avoid duplication
                    other_ray_tracer_params.extend(param_list)
            self.logger.info(f"🔧 Including ray_tracer parameters in optimizer: {list(ray_tracer_params.keys())}")
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': main_params, 'lr': self.training_config['learning_rate']},
            {'params': other_ray_tracer_params, 'lr': self.training_config['learning_rate']}
        ]
        
        self.optimizer = optim.Adam(
            param_groups,
            weight_decay=self.training_config['weight_decay']
        )
        
        # Create scheduler from config
        scheduler_config = getattr(self.config_loader.training, 'lr_scheduler', {})
        
        # Handle both dict and object cases
        if isinstance(scheduler_config, dict):
            scheduler_params = {
                'mode': scheduler_config.get('mode', 'min'),
                'factor': scheduler_config.get('factor', 0.7),
                'patience': scheduler_config.get('patience', 4),
                'threshold': scheduler_config.get('threshold', 0.0001),
                'threshold_mode': scheduler_config.get('threshold_mode', 'rel'),
                'cooldown': scheduler_config.get('cooldown', 1),
                'min_lr': scheduler_config.get('min_lr_plateau', 0.000005)
                # Note: 'verbose' parameter removed - deprecated in PyTorch. Use get_last_lr() instead.
            }
        else:
            scheduler_params = {
                'mode': getattr(scheduler_config, 'mode', 'min'),
                'factor': getattr(scheduler_config, 'factor', 0.7),
                'patience': getattr(scheduler_config, 'patience', 4),
                'threshold': getattr(scheduler_config, 'threshold', 0.0001),
                'threshold_mode': getattr(scheduler_config, 'threshold_mode', 'rel'),
                'cooldown': getattr(scheduler_config, 'cooldown', 1),
                'min_lr': getattr(scheduler_config, 'min_lr_plateau', 0.000005)
                # Note: 'verbose' parameter removed - deprecated in PyTorch. Use get_last_lr() instead.
            }
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            **scheduler_params
        )
        
        # Log scheduler configuration
        self.logger.info(f"✅ Scheduler created: ReduceLROnPlateau")
        self.logger.info(f"   Factor: {scheduler_params['factor']}")
        self.logger.info(f"   Patience: {scheduler_params['patience']}")
        self.logger.info(f"   Threshold: {scheduler_params['threshold']}")
        self.logger.info(f"   Min LR: {scheduler_params['min_lr']}")
        
        # Mixed precision disabled: incompatible with complex tensor operations in CSI processing
        self.scaler = None
        
        self.logger.info(f"✅ Optimizer created: Adam (lr={self.training_config['learning_rate']})")
        self.logger.info(f"   Mixed precision: disabled (incompatible with complex tensors)")
        
        # Note: Amplitude scaling handled by CSI enhancement network
        
    
    def _load_data(self):
        """Load and prepare training data based on configuration."""
        self.logger.info("🔧 Loading training data...")
        
        # Use base class data loading method
        ue_positions, bs_positions, bs_ant_indices, ue_ant_indices, csi_data = super()._load_data()
        
        # Prepare data split for training
        self._prepare_data_split(ue_positions, bs_positions, bs_ant_indices, ue_ant_indices, csi_data)
    def _prepare_data_split(self, ue_positions: torch.Tensor, bs_positions: torch.Tensor, 
                           bs_ant_indices: torch.Tensor, ue_ant_indices: torch.Tensor, csi_data: torch.Tensor):
        """Prepare train/validation data split for individual CSI samples"""
        self.logger.info(f"✅ Data loaded successfully:")
        self.logger.info(f"   UE positions: {ue_positions.shape}")
        self.logger.info(f"   BS positions: {bs_positions.shape}")
        self.logger.info(f"   BS antenna indices: {bs_ant_indices.shape}")
        self.logger.info(f"   UE antenna indices: {ue_ant_indices.shape}")
        self.logger.info(f"   CSI data: {csi_data.shape}")
        
        # Create train/validation split
        num_samples = ue_positions.shape[0]
        train_size = int(num_samples * self.data_config['train_ratio'])
        
        # Random split with fixed seed
        torch.manual_seed(self.data_config['random_seed'])
        indices = torch.randperm(num_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create datasets with logical order: (ue_pos, bs_pos, bs_ant_idx, ue_ant_idx, target_csi)
        train_dataset = TensorDataset(
            ue_positions[train_indices],
            bs_positions[train_indices],
            bs_ant_indices[train_indices],
            ue_ant_indices[train_indices],
            csi_data[train_indices]
        )
        
        val_dataset = TensorDataset(
            ue_positions[val_indices],
            bs_positions[val_indices],
            bs_ant_indices[val_indices],
            ue_ant_indices[val_indices],
            csi_data[val_indices]
        )
        
        # Create data loaders with position-aware batching if enabled
        use_position_aware = self.data_config.get('use_position_aware_loading', False)
        
        # Get metadata from the loaded data (passed from BaseRunner)
        metadata = getattr(self, '_loaded_metadata', {})
        
        if use_position_aware and 'position_pair_grouping' in metadata:
            self.logger.info("🔄 Creating position-aware data loaders")
            
            # Get position pair grouping information
            position_pair_grouping = metadata['position_pair_grouping']
            position_pair_indices = position_pair_grouping['group_indices']
            
            # Create position-aware data loaders
            self.train_loader = create_position_aware_dataloader(
                dataset=train_dataset,
                position_pair_indices=position_pair_indices[:len(train_dataset)],
                batch_size=self.training_config['batch_size'],
                shuffle=True,
                drop_last=False,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            self.val_loader = create_position_aware_dataloader(
                dataset=val_dataset,
                position_pair_indices=position_pair_indices[len(train_dataset):],
                batch_size=self.training_config['batch_size'],
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            self.logger.info(f"✅ Position-aware data loaders created:")
            self.logger.info(f"   Training batches: {len(self.train_loader)}")
            self.logger.info(f"   Validation batches: {len(self.val_loader)}")
            self.logger.info(f"   Batch size (position pairs): {self.training_config['batch_size']}")
            self.logger.info(f"   Samples per position pair: {position_pair_grouping['samples_per_group']}")
            
        else:
            self.logger.info("🔄 Creating standard data loaders")
            # Create standard data loaders
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=True,
                num_workers=0,  # Set to 0 for debugging
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            self.logger.info(f"✅ Standard data loaders created:")
            self.logger.info(f"   Training samples: {len(train_dataset)} ({len(self.train_loader)} batches)")
            self.logger.info(f"   Validation samples: {len(val_dataset)} ({len(self.val_loader)} batches)")
        
        # Validate that training interface subcarrier count matches data
        # CSI data is now 2D format: [batch, subcarriers]
        actual_subcarriers = csi_data.shape[1]
        if hasattr(self.training_interface, 'num_subcarriers') and self.training_interface.num_subcarriers != actual_subcarriers:
            self.logger.warning(f"⚠️ Subcarrier count mismatch: training_interface has {self.training_interface.num_subcarriers}, data has {actual_subcarriers}")
            self.logger.info(f"   Training interface expects {self.training_interface.num_subcarriers} subcarriers per UE antenna")
        
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        self.writer = SummaryWriter(self.output_paths['tensorboard_dir'])
        self.logger.info(f"✅ TensorBoard logging setup: {self.output_paths['tensorboard_dir']}")
        
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the checkpoint directory."""
        checkpoint_dir = self.output_paths['checkpoint_dir']
        if not os.path.exists(checkpoint_dir):
            return None
        
        # Find all .pt checkpoint files
        checkpoint_files = []
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pt') and file.startswith('checkpoint_epoch_'):
                full_path = os.path.join(checkpoint_dir, file)
                checkpoint_files.append((full_path, os.path.getmtime(full_path)))
        
        if not checkpoint_files:
            return None
        
        # Return the most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])[0]
        return latest_checkpoint

    def _load_checkpoint(self):
        """Load checkpoint if resuming training."""
        checkpoint_path = None
        
        # Priority 1: Explicitly specified checkpoint
        if self.resume_from and os.path.exists(self.resume_from):
            checkpoint_path = self.resume_from
            self.logger.info(f"🔄 Resuming training from specified checkpoint: {self.resume_from}")
        
        # Priority 2: Auto-find latest checkpoint
        elif not self.resume_from:
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                checkpoint_path = latest_checkpoint
                self.logger.info(f"🔄 Auto-resuming from latest checkpoint: {latest_checkpoint}")
            else:
                self.logger.info("🆕 No existing checkpoints found, starting fresh training")
                return
        
        if checkpoint_path:
            try:
                # Load checkpoint through training interface
                loaded_states = self.training_interface.load_checkpoint(
                    checkpoint_path,
                    load_training_state=True,
                    load_optimizer=True,
                    load_scheduler=True
                )
                
                # Restore optimizer and scheduler states
                if 'optimizer_state_dict' in loaded_states:
                    self.optimizer.load_state_dict(loaded_states['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in loaded_states:
                    self.scheduler.load_state_dict(loaded_states['scheduler_state_dict'])
                
                # Update training state
                checkpoint_info = loaded_states['checkpoint_info']
                self.current_epoch = checkpoint_info['epoch'] + 1  # Continue from next epoch
                self.best_loss = checkpoint_info['best_loss']
                
                self.logger.info(f"✅ Checkpoint loaded successfully:")
                self.logger.info(f"   Continuing from epoch: {self.current_epoch}")
                self.logger.info(f"   Previous best loss: {self.best_loss:.6f}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load checkpoint: {e}")
                self.logger.info("🆕 Starting fresh training instead")
                self.current_epoch = 0
                self.best_loss = float('inf')
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.prism_network.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Setup progress monitor
        monitor = TrainingProgressMonitor(self.training_config['num_epochs'], num_batches)
        monitor.start_epoch(epoch)
        
        for batch_idx, (ue_pos, bs_pos, bs_ant_idx, ue_ant_idx, target_csi) in enumerate(self.train_loader):
            # Data order: (inputs: ue_pos, bs_pos, bs_ant_idx, ue_ant_idx) -> (output: target_csi)
            
            # Show progress for each batch
            progress_percent = (batch_idx + 1) / num_batches * 100
            print(f"\r🔄 Training Batch {batch_idx + 1}/{num_batches} ({progress_percent:.1f}%)", end="", flush=True)
            
            # Note: Amplitude scaling handled by CSI enhancement network
            
            # Move data to device
            ue_pos = ue_pos.to(self.device)
            bs_pos = bs_pos.to(self.device)
            target_csi = target_csi.to(self.device)
            bs_ant_idx = bs_ant_idx.to(self.device)
            ue_ant_idx = ue_ant_idx.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            try:
                # Forward pass without mixed precision (complex tensors incompatible)
                # Forward pass through training interface
                outputs = self.training_interface(
                    ue_positions=ue_pos,
                    bs_positions=bs_pos,
                    bs_antenna_indices=bs_ant_idx,
                    ue_antenna_indices=ue_ant_idx
                )
                
                predictions = outputs['csi']
                
                # Prepare prediction and target dictionaries
                pred_dict = {
                    'csi': predictions,
                    'bs_antenna_indices': bs_ant_idx,
                    'ue_antenna_indices': ue_ant_idx,
                    'bs_positions': bs_pos,
                    'ue_positions': ue_pos
                }
                target_dict = {
                    'csi': target_csi,
                    'bs_antenna_indices': bs_ant_idx,
                    'ue_antenna_indices': ue_ant_idx,
                    'bs_positions': bs_pos,
                    'ue_positions': ue_pos
                }
                
                # Compute loss
                loss = self.training_interface.compute_loss(
                    predictions=pred_dict,
                    targets=target_dict,
                    loss_function=self.loss_function,
                    validation_mode=False
                )
                
                # Backward pass - standard FP32 training (mixed precision disabled for complex tensors)
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
                
                # Optimizer step
                if torch.isfinite(loss) and torch.isfinite(grad_norm):
                    self.optimizer.step()
                else:
                    self.logger.warning(f"⚠️ Non-finite values detected (grad_norm={grad_norm}, loss={loss.item()}), skipping step")
                
                # Update statistics
                batch_loss = loss.item()
                total_loss += batch_loss
                
                
                # Update training interface state
                self.training_interface.update_training_state(epoch, batch_idx, batch_loss)
                
                # Update progress
                monitor.update_batch(batch_idx, batch_loss)
                
                # Force memory cleanup after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                # Automatic checkpointing
                if (self.training_config['auto_checkpoint'] and 
                    batch_idx % self.training_config['checkpoint_frequency'] == 0 and 
                    batch_idx > 0):
                    
                    self.training_interface.save_checkpoint(
                        epoch=epoch,
                        batch=batch_idx,
                        optimizer_state=self.optimizer.state_dict(),
                        scheduler_state=self.scheduler.state_dict()
                    )
                
                # Log to TensorBoard
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Loss/Train/Batch', batch_loss, global_step)
                
                
            except Exception as e:
                self.logger.error(f"❌ Error in batch {batch_idx}: {e}")
                self.logger.error(f"   UE positions shape: {ue_pos.shape}")
                self.logger.error(f"   BS positions shape: {bs_pos.shape}")
                self.logger.error(f"   Target CSI shape: {target_csi.shape}")
                self.logger.error(f"   BS antenna indices shape: {bs_ant_idx.shape}")
                self.logger.error(f"   UE antenna indices shape: {ue_ant_idx.shape}")
                raise
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        monitor.end_epoch(avg_loss)
        
        # New line after progress display
        print()  # Move to next line after progress display
        
        return avg_loss
        
    def validate_epoch(self, epoch: int) -> float:
        """Validate for one epoch."""
        self.prism_network.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, (ue_pos, bs_pos, bs_ant_idx, ue_ant_idx, target_csi) in enumerate(self.val_loader):
                # Data order: (inputs: ue_pos, bs_pos, bs_ant_idx, ue_ant_idx) -> (output: target_csi)
                # Show progress for each batch
                progress_percent = (batch_idx + 1) / num_batches * 100
                print(f"\r🔍 Validation Batch {batch_idx + 1}/{num_batches} ({progress_percent:.1f}%)", end="", flush=True)
                
                # Note: Amplitude scaling handled by CSI enhancement network
                
                # Move data to device
                ue_pos = ue_pos.to(self.device)
                bs_pos = bs_pos.to(self.device)
                target_csi = target_csi.to(self.device)
                bs_ant_idx = bs_ant_idx.to(self.device)
                ue_ant_idx = ue_ant_idx.to(self.device)
                
                try:
                    # Forward pass
                    outputs = self.training_interface(
                        ue_positions=ue_pos,
                        bs_positions=bs_pos,
                        bs_antenna_indices=bs_ant_idx,
                        ue_antenna_indices=ue_ant_idx
                    )
                    
                    predictions = outputs['csi']
                    
                    # Prepare prediction and target dictionaries
                    pred_dict = {
                        'csi': predictions,
                        'bs_antenna_indices': bs_ant_idx,
                        'ue_antenna_indices': ue_ant_idx,
                        'bs_positions': bs_pos,
                        'ue_positions': ue_pos
                    }
                    target_dict = {
                        'csi': target_csi,
                        'bs_antenna_indices': bs_ant_idx,
                        'ue_antenna_indices': ue_ant_idx,
                        'bs_positions': bs_pos,
                        'ue_positions': ue_pos
                    }
                    
                    # Compute loss
                    loss = self.training_interface.compute_loss(
                        predictions=pred_dict,
                        targets=target_dict,
                        loss_function=self.loss_function,
                        validation_mode=True
                    )
                    
                    total_loss += loss.item()
                    
                    # ========================================
                    # CSI Amplitude Statistics for validation batch
                    # ========================================
                    # Calculate amplitude statistics for predictions and targets
                    pred_amplitude = torch.abs(predictions)
                    target_amplitude = torch.abs(target_csi)
                    
                    # Filter out zero values for meaningful statistics
                    pred_nonzero = pred_amplitude[pred_amplitude > 1e-8]
                    target_nonzero = target_amplitude[target_amplitude > 1e-8]
                    
                    if len(pred_nonzero) > 0 and len(target_nonzero) > 0:
                        pred_stats = {
                            'min': pred_nonzero.min().item(),
                            'max': pred_nonzero.max().item(),
                            'mean': pred_nonzero.mean().item(),
                            'std': pred_nonzero.std().item()
                        }
                        target_stats = {
                            'min': target_nonzero.min().item(),
                            'max': target_nonzero.max().item(),
                            'mean': target_nonzero.mean().item(),
                            'std': target_nonzero.std().item()
                        }
                        
                        # Log amplitude statistics
                        self.logger.info(f"📊 Validation Batch {batch_idx + 1} CSI Amplitude Statistics:")
                        self.logger.info(f"   Predictions: min={pred_stats['min']:.6f}, max={pred_stats['max']:.6f}, mean={pred_stats['mean']:.6f}, std={pred_stats['std']:.6f}")
                        self.logger.info(f"   Targets:     min={target_stats['min']:.6f}, max={target_stats['max']:.6f}, mean={target_stats['mean']:.6f}, std={target_stats['std']:.6f}")
                        
                        # Calculate amplitude ratio (how close predictions are to targets)
                        amplitude_ratio = pred_stats['mean'] / target_stats['mean'] if target_stats['mean'] > 0 else 0
                        self.logger.info(f"   Amplitude Ratio (pred/target): {amplitude_ratio:.4f}")
                        
                        # Log range coverage
                        pred_range = pred_stats['max'] - pred_stats['min']
                        target_range = target_stats['max'] - target_stats['min']
                        range_coverage = pred_range / target_range if target_range > 0 else 0
                        self.logger.info(f"   Range Coverage (pred/target): {range_coverage:.4f}")
                    else:
                        self.logger.warning(f"⚠️ Validation Batch {batch_idx + 1}: No valid amplitude data found")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️  Validation error in batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # New line after progress display
        print()  # Move to next line after progress display
        
        return avg_loss
    
    def _log_training_time_statistics(self, total_time: float):
        """Log detailed training time statistics."""
        # Calculate time components
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        # Format time strings
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"
        
        # Calculate additional statistics
        num_epochs = self.training_config.get('num_epochs', 0)
        avg_epoch_time = total_time / num_epochs if num_epochs > 0 else 0
        
        # Log detailed statistics
        self.logger.info("=" * 80)
        self.logger.info("🏁 TRAINING COMPLETED - TIME STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"⏱️  Total Training Time: {time_str} ({total_time:.1f} seconds)")
        self.logger.info(f"📊 Total Epochs: {num_epochs}")
        self.logger.info(f"⚡ Average Time per Epoch: {avg_epoch_time:.1f}s")
        self.logger.info(f"🎯 Best Validation Loss: {self.best_loss:.6f}")
        
        # Log start and end times
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.training_start_time))
        end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.logger.info(f"🕐 Training Started: {start_time_str}")
        self.logger.info(f"🕐 Training Ended: {end_time_str}")
        
        # Log efficiency metrics
        if hasattr(self, 'train_loader') and self.train_loader:
            total_batches = len(self.train_loader) * num_epochs
            avg_batch_time = total_time / total_batches if total_batches > 0 else 0
            self.logger.info(f"📈 Total Batches Processed: {total_batches}")
            self.logger.info(f"⚡ Average Time per Batch: {avg_batch_time:.3f}s")
        
        
        self.logger.info("=" * 80)
        
        # Also print to console for immediate visibility
        print("\n" + "=" * 80)
        print("🏁 TRAINING COMPLETED - TIME STATISTICS")
        print("=" * 80)
        print(f"⏱️  Total Training Time: {time_str} ({total_time:.1f} seconds)")
        print(f"📊 Total Epochs: {num_epochs}")
        print(f"⚡ Average Time per Epoch: {avg_epoch_time:.1f}s")
        print(f"🎯 Best Validation Loss: {self.best_loss:.6f}")
        print(f"🕐 Training Started: {start_time_str}")
        print(f"🕐 Training Ended: {end_time_str}")
        if hasattr(self, 'train_loader') and self.train_loader:
            total_batches = len(self.train_loader) * num_epochs
            avg_batch_time = total_time / total_batches if total_batches > 0 else 0
            print(f"📈 Total Batches Processed: {total_batches}")
            print(f"⚡ Average Time per Batch: {avg_batch_time:.3f}s")
        print("=" * 80)
        
    def train(self):
        """Main training loop."""
        self.logger.info("🚀 Starting Prism neural network training")
        self.training_start_time = time.time()
        
        # Clear previous results FIRST if new training
        if self.new_training:
            self._clear_previous_results()
            # Recreate directories after clearing
            self.config_loader.ensure_output_directories()
            # Re-setup logging after clearing (log file was deleted)
            self._setup_logging()
        
        # Initialize all components
        self._create_model()
        self._create_loss_function()
        self._create_optimizer()
        self._load_data()  # Load data based on configuration
        self._setup_tensorboard()
        self._load_checkpoint()
        
        # Training loop
        start_epoch = self.current_epoch
        num_epochs = self.training_config['num_epochs']
        
        self.logger.info(f"🔄 Training from epoch {start_epoch} to {num_epochs}")
        
        # Initialize epoch to handle early errors in finally block
        epoch = start_epoch - 1
        
        try:
            for epoch in range(start_epoch, num_epochs):
                epoch_start_time = time.time()
                
                # Training phase
                train_loss = self.train_epoch(epoch)
                
                # Validation phase
                val_loss = self.validate_epoch(epoch)
                
                # Learning rate scheduling
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                
                # Log learning rate changes (replaces deprecated 'verbose' parameter)
                if old_lr != new_lr:
                    self.logger.info(f"📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
                
                # Update best loss and save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    best_model_path = os.path.join(self.output_paths['models_dir'], 'best_model.pt')
                    torch.save(self.prism_network.state_dict(), best_model_path)
                    self.logger.info(f"🎯 New best model saved: {best_model_path}")
                
                # Epoch-level checkpointing
                if epoch % self.training_config['epoch_save_interval'] == 0:
                    self.training_interface.save_checkpoint(
                        epoch=epoch,
                        batch=0,  # Epoch-level checkpoint
                        optimizer_state=self.optimizer.state_dict(),
                        scheduler_state=self.scheduler.state_dict()
                    )
                
                # Logging
                epoch_time = time.time() - epoch_start_time
                self.logger.info(f"📊 Epoch {epoch} Summary:")
                self.logger.info(f"   Training Loss: {train_loss:.6f}")
                self.logger.info(f"   Validation Loss: {val_loss:.6f}")
                self.logger.info(f"   Best Loss: {self.best_loss:.6f}")
                self.logger.info(f"   Epoch Time: {epoch_time:.1f}s")
                self.logger.info(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Log ray tracer info if available
                if hasattr(self.training_interface, 'ray_tracer'):
                    self.logger.info(f"   Ray Tracer: {type(self.training_interface.ray_tracer).__name__}")
                
                # TensorBoard logging
                self.writer.add_scalar('Loss/Train/Epoch', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation/Epoch', val_loss, epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
                
                # Log ray tracer info to TensorBoard
                if hasattr(self.training_interface, 'ray_tracer'):
                    # Log ray tracer type as a scalar (converted to float for TensorBoard)
                    ray_tracer_type = hash(type(self.training_interface.ray_tracer).__name__) % 1000
                    self.writer.add_scalar('RayTracer/Type', ray_tracer_type, epoch)
                
        except KeyboardInterrupt:
            self.logger.info("⚠️  Training interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            # Final cleanup and training time statistics
            total_time = time.time() - self.training_start_time
            
            # Detailed training time statistics
            self._log_training_time_statistics(total_time)
            
            # Save final model
            final_model_path = os.path.join(self.output_paths['models_dir'], 'final_model.pt')
            torch.save(self.prism_network.state_dict(), final_model_path)
            self.logger.info(f"🎯 Final model saved: {final_model_path}")
            
            # Save final checkpoint
            self.training_interface.save_checkpoint(
                epoch=epoch,
                batch=-1,  # Final checkpoint marker
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict()
            )
            
            # Close TensorBoard writer
            if self.writer:
                self.writer.close()
            
            self.logger.info("✅ Prism training pipeline completed successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train Prism Neural Network')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--resume', help='Path to checkpoint for resuming training')
    parser.add_argument('--gpu', type=int, help='GPU device ID to use')
    parser.add_argument('--new', action='store_true', help='Start fresh training (clear previous results)')
    
    args = parser.parse_args()
    
    # Set GPU device if specified
    if args.gpu is not None:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            print(f"🔧 Using GPU device: {args.gpu}")
        else:
            print("⚠️  GPU requested but CUDA not available, using CPU")
    
    # Create and run trainer
    try:
        trainer = PrismTrainer(
            config_path=args.config,
            resume_from=args.resume,
            new_training=args.new
        )
        trainer.train()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
