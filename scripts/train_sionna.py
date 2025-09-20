#!/usr/bin/env python3
"""
Modern Prism Neural Network Training Script

This script provides a streamlined training pipeline for the Prism neural electromagnetic 
ray tracing system using the new simplified architecture and configuration structure.

Features:
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


class SionnaTrainer:
    """
    Sionna training pipeline for Prism neural network system.
    
    Simplified training implementation using the new PrismTrainingInterface
    and modern configuration loading without legacy compatibility concerns.
    """
    
    def __init__(self, config_path: str, resume_from: Optional[str] = None):
        """
        Initialize trainer with configuration path.
        
        Args:
            config_path: Path to YAML configuration file
            resume_from: Optional path to checkpoint for resuming training
        """
        self.config_path = config_path
        self.resume_from = resume_from
        
        # Load configuration using modern loader
        print("🔧 Loading configuration...")
        try:
            self.config_loader = ModernConfigLoader(config_path)
            print(f"✅ Configuration loaded from: {config_path}")
        except Exception as e:
            print(f"❌ FATAL ERROR: Failed to load configuration from {config_path}")
            print(f"   Error: {e}")
            print(f"   Please check your configuration file and ensure it exists and is valid.")
            raise
        
        # Extract key configurations first
        self.device = self.config_loader.get_device()
        self.training_config = self.config_loader.get_training_kwargs()
        self.output_paths = self.config_loader.get_output_paths()
        self.data_config = self.config_loader.get_data_loader_config()
        
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
        
        self.logger.info("🚀 SionnaTrainer initialized successfully")
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
        
        # Create network
        self.prism_network = PrismNetwork(**network_kwargs).to(self.device)
        
        # Create training interface
        config_dict = {
            'system': {'device': str(self.device)},
            'training': self.config_loader.training.__dict__,
            'user_equipment': {
                'target_antenna_index': self.data_config['ue_antenna_index']
            },
            'input': {
                'subcarrier_sampling': {
                    'sampling_ratio': self.data_config['sampling_ratio'],
                    'sampling_method': self.data_config['sampling_method'],
                    'antenna_consistent': self.data_config['antenna_consistent']
                }
            }
        }
        
        self.training_interface = PrismTrainingInterface(
            prism_network=self.prism_network,
            config=config_dict,
            checkpoint_dir=self.output_paths['checkpoint_dir'],
            device=self.device
        )
        
        # Apply memory optimizations if enabled
        if self.training_config.get('enable_gradient_checkpointing', False):
            self.logger.info("🚀 Applying ultra memory optimizations...")
            
            try:
                from prism.memory_optimizations import apply_memory_optimizations
                self.training_interface, self.prism_network = apply_memory_optimizations(
                    self.training_interface, 
                    self.prism_network, 
                    {'training': self.training_config}
                )
                self.logger.info("✅ Memory optimizations applied successfully")
            except ImportError as e:
                self.logger.warning(f"⚠️ Memory optimization module not available: {e}")
                self.logger.info("💡 Using standard memory optimization")
            except Exception as e:
                self.logger.error(f"❌ Failed to apply memory optimizations: {e}")
                self.logger.info("💡 Falling back to standard training")
        
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
        
        # Create combined loss function with unified config
        self.loss_function = LossFunction(config=loss_config)
        
        self.logger.info(f"✅ Loss function created with weights: CSI={loss_config.get('csi_weight', 0.7)}, "
                        f"PDP={loss_config.get('pdp_weight', 0.3)}, "
                        f"Spatial={loss_config.get('spatial_spectrum_weight', 0.0)}, "
                        f"Reg={loss_config.get('regularization_weight', 0.01)}")
        
    def _create_optimizer(self):
        """Create optimizer and scheduler."""
        self.logger.info("🔧 Creating optimizer...")
        
        # Create optimizer - include both prism_network and ray_tracer parameters
        all_parameters = list(self.prism_network.parameters())
        
        # Add ray_tracer learnable parameters
        ray_tracer_params = self.training_interface.ray_tracer.get_learnable_parameters()
        if ray_tracer_params:
            # Flatten the parameter lists
            for param_list in ray_tracer_params.values():
                all_parameters.extend(param_list)
            self.logger.info(f"🔧 Including ray_tracer parameters in optimizer: {list(ray_tracer_params.keys())}")
        
        self.optimizer = optim.Adam(
            all_parameters,
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=4,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=1,
            min_lr=0.000005
        )
        
        # Create GradScaler for mixed precision training with proper initialization
        if self.config_loader.prism_network.use_mixed_precision and torch.cuda.is_available():
            # Initialize GradScaler with conservative settings to avoid "No inf checks" error
            self.scaler = torch.amp.GradScaler(
                device='cuda',
                init_scale=2.**16,  # Start with moderate scale
                growth_factor=2.0,  # Conservative growth
                backoff_factor=0.5, # Conservative backoff
                growth_interval=2000  # Less frequent growth
            )
        else:
            self.scaler = None
        
        mixed_precision_status = "enabled" if self.config_loader.prism_network.use_mixed_precision else "disabled"
        self.logger.info(f"✅ Optimizer created: Adam (lr={self.training_config['learning_rate']})")
        self.logger.info(f"   Mixed precision: {mixed_precision_status}")
        
    def _load_data(self):
        """Load and prepare training data."""
        self.logger.info("🔧 Loading training data...")
        
        # Construct dataset path (updated for new HDF5 structure)
        dataset_path = "data/sionna/results/P300/ray_tracing_5g_simulation_P300.h5"
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Load data
        with h5py.File(dataset_path, 'r') as f:
            # Extract data from new HDF5 structure
            ue_positions = torch.from_numpy(f['data/ue_position'][:]).float()
            bs_position_single = torch.from_numpy(f['data/bs_position'][:]).float()
            channel_responses = torch.from_numpy(f['data/channel_responses'][:]).cfloat()
            
            # Current structure: [samples, subcarriers, ue_antennas, bs_antennas]
            # Need to convert to: [samples, bs_antennas, subcarriers, ue_antennas]
            csi_data = channel_responses.permute(0, 3, 1, 2)  # [samples, bs_antennas, subcarriers, ue_antennas]
            
            # Handle BS positions intelligently based on dataset structure
            num_samples = ue_positions.shape[0]
            
            # Check if BS positions are per-sample or single fixed position
            if bs_position_single.dim() == 1:
                # Single fixed BS position for all samples
                self.logger.info(f"📍 Fixed BS mode: Single BS position for all {num_samples} samples")
                bs_positions = bs_position_single.unsqueeze(0).expand(num_samples, -1)
                self._bs_mode = 'fixed'
            elif bs_position_single.dim() == 2 and bs_position_single.shape[0] == num_samples:
                # Per-sample BS positions
                self.logger.info(f"📍 Dynamic BS mode: {num_samples} BS positions for {num_samples} samples")
                bs_positions = bs_position_single
                self._bs_mode = 'dynamic'
            else:
                raise ValueError(f"Invalid BS position shape: {bs_position_single.shape}. "
                               f"Expected [3] for fixed BS or [{num_samples}, 3] for dynamic BS.")
            
            # Generate antenna indices (sequential 0 to num_antennas-1 for each sample)
            num_bs_antennas = csi_data.shape[1]
            antenna_indices = torch.arange(num_bs_antennas).unsqueeze(0).expand(num_samples, -1).long()
            
            # Extract specific UE antenna and remove UE dimension
            ue_antenna_idx = self.data_config['ue_antenna_index']
            if csi_data.dim() == 4:  # [samples, bs_antennas, subcarriers, ue_antennas]
                csi_data = csi_data[:, :, :, ue_antenna_idx]  # Remove UE dimension: [samples, bs_antennas, subcarriers]
            
            # Phase differential calibration to remove common phase offset
            self.logger.info("🔧 Applying phase differential calibration...")
            original_shape = csi_data.shape
            original_subcarriers = original_shape[2]
            
            # Method: Use first subcarrier as reference
            # Normalize all subcarriers by dividing by the first subcarrier
            # csi_norm[k] = csi[k] / csi[0] for k = 0, 1, ..., N-1
            # This removes absolute phase offset while preserving relative phase and amplitude relationships
            
            # Extract first subcarrier as reference (shape: [batch, antenna, 1])
            reference_subcarrier = csi_data[:, :, 0:1]  # Keep dimension for broadcasting
            
            # Avoid division by zero by adding small epsilon to reference
            epsilon = 1e-12
            reference_subcarrier_safe = reference_subcarrier + epsilon * torch.exp(1j * torch.angle(reference_subcarrier))
            
            # Normalize all subcarriers by the reference (first subcarrier)
            # This preserves all subcarriers (N subcarriers -> N subcarriers)
            # Complex division: csi[k] / ref = csi[k] * conj(ref) / |ref|^2
            csi_data = csi_data * torch.conj(reference_subcarrier_safe) / (torch.abs(reference_subcarrier_safe) ** 2)
            
            # Update CSI data to use first subcarrier normalization
            # Note: This preserves all subcarriers (N subcarriers -> N subcarriers)
            # Format: [samples, bs_antennas, subcarriers]
            
            self.logger.info(f"✅ Phase differential calibration applied:")
            self.logger.info(f"   Original subcarriers: {original_subcarriers}")
            self.logger.info(f"   Normalized subcarriers: {csi_data.shape[2]}")
            self.logger.info(f"   Final format: [samples, antennas, subcarriers] = {csi_data.shape}")
            self.logger.info(f"   Method: First subcarrier normalization (csi[k] / csi[0])")
            
        self.logger.info(f"✅ Data loaded from new HDF5 structure:")
        self.logger.info(f"   UE positions: {ue_positions.shape}")
        self.logger.info(f"   BS mode: {self._bs_mode}")
        if self._bs_mode == 'fixed':
            self.logger.info(f"   BS position (fixed): {bs_position_single}")
        else:
            self.logger.info(f"   BS positions (dynamic): {bs_positions.shape}")
        self.logger.info(f"   CSI data: {csi_data.shape} [samples, bs_antennas, subcarriers, ue_antennas]")
        self.logger.info(f"   Antenna indices (generated): {antenna_indices.shape}")
        self.logger.info(f"   Selected UE antenna: {ue_antenna_idx}")
        
        # Create train/validation split
        num_samples = ue_positions.shape[0]
        train_size = int(num_samples * self.data_config['train_ratio'])
        
        # Random split with fixed seed
        torch.manual_seed(self.data_config['random_seed'])
        indices = torch.randperm(num_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create datasets with logical order: (ue_pos, bs_pos, antenna_idx, target_csi)
        train_dataset = TensorDataset(
            ue_positions[train_indices],
            bs_positions[train_indices],
            antenna_indices[train_indices],
            csi_data[train_indices]
        )
        
        val_dataset = TensorDataset(
            ue_positions[val_indices],
            bs_positions[val_indices],
            antenna_indices[val_indices],
            csi_data[val_indices]
        )
        
        # Create data loaders
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
        
        self.logger.info(f"✅ Data loaders created:")
        self.logger.info(f"   Training samples: {len(train_dataset)} ({len(self.train_loader)} batches)")
        self.logger.info(f"   Validation samples: {len(val_dataset)} ({len(self.val_loader)} batches)")
        
        # Update training interface with actual subcarrier count after phase differential processing
        actual_subcarriers = csi_data.shape[2]  # Get actual subcarrier count from processed data
        if hasattr(self.training_interface, 'num_subcarriers') and self.training_interface.num_subcarriers != actual_subcarriers:
            self.logger.info(f"🔧 Updating training interface subcarrier count: {self.training_interface.num_subcarriers} → {actual_subcarriers}")
            self.training_interface.num_subcarriers = actual_subcarriers
            self.logger.info(f"   Now using all {actual_subcarriers} subcarriers (no selection needed)")
        
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
        
        for batch_idx, (ue_pos, bs_pos, antenna_idx, target_csi) in enumerate(self.train_loader):
            # Data order: (inputs: ue_pos, bs_pos, antenna_idx) -> (output: target_csi)
            # Clear cache periodically to prevent fragmentation
            if torch.cuda.is_available() and batch_idx % 20 == 0:
                torch.cuda.empty_cache()
            
            # Show progress for each batch
            progress_percent = (batch_idx + 1) / num_batches * 100
            print(f"\r🔄 Training Batch {batch_idx + 1}/{num_batches} ({progress_percent:.1f}%)", end="", flush=True)
            
            # Move data to device
            ue_pos = ue_pos.to(self.device)
            bs_pos = bs_pos.to(self.device)
            target_csi = target_csi.to(self.device)
            antenna_idx = antenna_idx.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            try:
                # Forward pass with mixed precision autocast
                with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                    # Forward pass through training interface (use optimized version if available)
                    if hasattr(self.training_interface, 'forward_memory_optimized'):
                        outputs = self.training_interface.forward_memory_optimized(
                            ue_positions=ue_pos,
                            bs_positions=bs_pos,
                            antenna_indices=antenna_idx
                        )
                    else:
                        outputs = self.training_interface(
                            ue_positions=ue_pos,
                            bs_positions=bs_pos,  # BS positions for each sample
                            antenna_indices=antenna_idx
                        )
                    
                    predictions = outputs['csi_predictions']
                    
                    # Compute loss
                    loss = self.training_interface.compute_loss(
                        predictions=predictions,
                        targets=target_csi,
                        loss_function=self.loss_function,
                        validation_mode=False
                    )
                
                # Backward pass with mixed precision support
                if self.scaler is not None:
                    # Mixed precision training - scale loss and compute gradients
                    scaled_loss = self.scaler.scale(loss)
                    scaled_loss.backward()
                    
                    # Gradient clipping with scaler
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
                    
                    # Check for infinite or NaN gradients
                    if torch.isfinite(grad_norm) and torch.isfinite(loss):
                        # Safe optimizer step
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Skip optimizer step for non-finite gradients
                        self.logger.warning(f"⚠️ Non-finite gradients detected (grad_norm={grad_norm}, loss={loss.item()}), skipping step")
                        self.scaler.update()  # Still update scaler to maintain state
                else:
                    # Standard FP32 training
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
                
                # Minimal memory monitoring for critical issues only
                if torch.cuda.is_available() and batch_idx % 50 == 0:
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    if peak > 70.0:
                        self.logger.warning(f"⚠️ CRITICAL: Peak memory {peak:.2f}GB approaching limit!")
                
                # Update training interface state
                self.training_interface.update_training_state(epoch, batch_idx, batch_loss)
                
                # Update progress
                monitor.update_batch(batch_idx, batch_loss)
                
                # Automatic checkpointing
                if (self.training_config['auto_checkpoint'] and 
                    batch_idx % self.training_config['checkpoint_frequency'] == 0 and 
                    batch_idx > 0):
                    
                    self.training_interface.save_checkpoint(
                        optimizer_state=self.optimizer.state_dict(),
                        scheduler_state=self.scheduler.state_dict()
                    )
                
                # Log to TensorBoard
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Loss/Train/Batch', batch_loss, global_step)
                
                # Aggressive memory cleanup between batches
                del outputs, predictions, loss
                del ue_pos, bs_pos, target_csi, antenna_idx  # Clear batch data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensure all operations complete
                
            except Exception as e:
                self.logger.error(f"❌ Error in batch {batch_idx}: {e}")
                self.logger.error(f"   UE positions shape: {ue_pos.shape}")
                self.logger.error(f"   BS positions shape: {bs_pos.shape}")
                self.logger.error(f"   Target CSI shape: {target_csi.shape}")
                self.logger.error(f"   Antenna indices shape: {antenna_idx.shape}")
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
            for batch_idx, (ue_pos, bs_pos, antenna_idx, target_csi) in enumerate(self.val_loader):
                # Data order: (inputs: ue_pos, bs_pos, antenna_idx) -> (output: target_csi)
                # Show progress for each batch
                progress_percent = (batch_idx + 1) / num_batches * 100
                print(f"\r🔍 Validation Batch {batch_idx + 1}/{num_batches} ({progress_percent:.1f}%)", end="", flush=True)
                
                # Move data to device
                ue_pos = ue_pos.to(self.device)
                bs_pos = bs_pos.to(self.device)
                target_csi = target_csi.to(self.device)
                antenna_idx = antenna_idx.to(self.device)
                
                try:
                    # Forward pass (use optimized version if available)
                    if hasattr(self.training_interface, 'forward_memory_optimized'):
                        outputs = self.training_interface.forward_memory_optimized(
                            ue_positions=ue_pos,
                            bs_positions=bs_pos,
                            antenna_indices=antenna_idx
                        )
                    else:
                        outputs = self.training_interface(
                            ue_positions=ue_pos,
                            bs_positions=bs_pos,  # BS positions for each sample
                            antenna_indices=antenna_idx
                        )
                    
                    predictions = outputs['csi_predictions']
                    
                    # Compute loss
                    loss = self.training_interface.compute_loss(
                        predictions=predictions,
                        targets=target_csi,
                        loss_function=self.loss_function,
                        validation_mode=True
                    )
                    
                    total_loss += loss.item()
                    
                except Exception as e:
                    self.logger.warning(f"⚠️  Validation error in batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # New line after progress display
        print()  # Move to next line after progress display
        
        return avg_loss
        
    def train(self):
        """Main training loop."""
        self.logger.info("🚀 Starting Prism neural network training")
        self.training_start_time = time.time()
        
        # Initialize all components
        self._create_model()
        self._create_loss_function()
        self._create_optimizer()
        self._load_data()
        self._setup_tensorboard()
        self._load_checkpoint()
        
        # Training loop
        start_epoch = self.current_epoch
        num_epochs = self.training_config['num_epochs']
        
        self.logger.info(f"🔄 Training from epoch {start_epoch} to {num_epochs}")
        
        try:
            for epoch in range(start_epoch, num_epochs):
                epoch_start_time = time.time()
                
                # Training phase
                train_loss = self.train_epoch(epoch)
                
                # Validation phase
                val_loss = self.validate_epoch(epoch)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Update best loss and save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    best_model_path = os.path.join(self.output_paths['models_dir'], 'best_model.pt')
                    torch.save(self.prism_network.state_dict(), best_model_path)
                    self.logger.info(f"🎯 New best model saved: {best_model_path}")
                
                # Epoch-level checkpointing
                if epoch % self.training_config['epoch_save_interval'] == 0:
                    self.training_interface.save_checkpoint(
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
            # Final cleanup
            total_time = time.time() - self.training_start_time
            self.logger.info(f"🏁 Training completed in {total_time:.1f}s ({total_time/3600:.2f}h)")
            
            # Save final model
            final_model_path = os.path.join(self.output_paths['models_dir'], 'final_model.pt')
            torch.save(self.prism_network.state_dict(), final_model_path)
            
            # Save final checkpoint
            self.training_interface.save_checkpoint(
                filename='final_checkpoint.pt',
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict()
            )
            
            # Close TensorBoard writer
            if self.writer:
                self.writer.close()
            
            self.logger.info("✅ Training pipeline completed successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train Prism Neural Network')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--resume', help='Path to checkpoint for resuming training')
    parser.add_argument('--gpu', type=int, help='GPU device ID to use')
    
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
        trainer = SionnaTrainer(
            config_path=args.config,
            resume_from=args.resume
        )
        trainer.train()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
