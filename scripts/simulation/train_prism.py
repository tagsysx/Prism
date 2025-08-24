#!/usr/bin/env python3
"""
Prism Network Training Script

This script trains the Prism neural network for electromagnetic ray tracing
using simulated data from Sionna. It implements the complete training pipeline
including data loading, model initialization, training loop, and checkpointing.
"""

import os
import sys
import time
import logging
import argparse
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import threading
from tqdm import tqdm
import psutil

# Optional imports
try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import json
from datetime import datetime
import traceback

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from prism.networks.prism_network import PrismNetwork
from prism.ray_tracer_cpu import CPURayTracer
from prism.ray_tracer_cuda import CUDARayTracer
from prism.training_interface import PrismTrainingInterface

class TrainingProgressMonitor:
    """Real-time training progress monitor with GPU utilization tracking"""
    
    def __init__(self, total_epochs, total_batches_per_epoch):
        self.total_epochs = total_epochs
        self.total_batches_per_epoch = total_batches_per_epoch
        self.start_time = time.time()
        self.epoch_start_time = None
        self.batch_start_time = None
        
        # Progress tracking
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches_completed = 0
        
        # Performance metrics
        self.epoch_times = []
        self.batch_times = []
        self.losses = []
        
        # GPU monitoring
        self.gpu_utilization_history = []
        self.memory_usage_history = []
        
    def start_epoch(self, epoch):
        """Start monitoring a new epoch"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.current_batch = 0
        
        print(f"\n🔄 Epoch {epoch}/{self.total_epochs}")
        print(f"{'='*60}")
        print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
        print(f"💡 Training is active - you'll see real-time updates below")
        print(f"{'='*60}")
        
    def start_batch(self, batch_idx):
        """Start monitoring a new batch"""
        self.current_batch = batch_idx
        self.batch_start_time = time.time()
        
        # Show simple status indicator for first few batches
        if batch_idx < 3:
            print(f"  🚀 Processing batch {batch_idx+1}...")
        
    def update_batch_progress(self, batch_idx, loss, total_batches_in_epoch):
        """Update batch progress with real-time information"""
        self.current_batch = batch_idx
        self.total_batches_completed += 1
        
        # Calculate progress
        epoch_progress = (batch_idx + 1) / total_batches_in_epoch * 100
        overall_progress = (self.total_batches_completed) / (self.total_epochs * self.total_batches_per_epoch) * 100
        
        # Calculate timing
        batch_time = time.time() - self.batch_start_time
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.start_time
        
        # Estimate remaining time
        if batch_idx > 0:
            avg_batch_time = epoch_time / (batch_idx + 1)
            remaining_batches = total_batches_in_epoch - batch_idx - 1
            remaining_epoch_time = remaining_batches * avg_batch_time
            remaining_total_time = (self.total_epochs - self.current_epoch - 1) * self.total_batches_per_epoch * avg_batch_time
        else:
            remaining_epoch_time = 0
            remaining_total_time = 0
        
        # Get GPU information
        gpu_info = self._get_gpu_info()
        memory_info = self._get_memory_info()
        
        # Create simple progress bar
        progress_bar_length = 30
        filled_length = int(progress_bar_length * epoch_progress / 100)
        progress_bar = '█' * filled_length + '░' * (progress_bar_length - filled_length)
        
        # Clear previous lines and show updated progress
        print(f"\r", end="", flush=True)
        print(f"  📊 [{progress_bar}] {epoch_progress:5.1f}% | Batch {batch_idx+1:3d}/{total_batches_in_epoch:3d}")
        print(f"    🎯 Overall: {overall_progress:5.1f}% | Loss: {loss:.6f}")
        print(f"    ⏱️  Batch: {batch_time:.2f}s | Epoch: {epoch_time:.1f}s | Total: {total_time:.1f}s")
        print(f"    🔍 GPU: {gpu_info} | Memory: {memory_info}")
        print(f"    ⏳ Remaining: Epoch {remaining_epoch_time:.1f}s | Total {remaining_total_time:.1f}s")
        
        # Store metrics
        self.batch_times.append(batch_time)
        self.losses.append(loss)
        self.gpu_utilization_history.append(gpu_info)
        self.memory_usage_history.append(memory_info)
        
        # Show heartbeat every 10 batches to prove training is alive
        if batch_idx % 10 == 0:
            current_time = time.strftime('%H:%M:%S')
            print(f"    💓 Heartbeat: {current_time} - Training is alive and running!")
        
    def end_epoch(self, avg_loss):
        """End epoch monitoring and show summary"""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        print(f"\n  ✅ Epoch {self.current_epoch} completed in {epoch_time:.1f}s")
        print(f"    📈 Average Loss: {avg_loss:.6f}")
        print(f"    🚀 Progress: {self.current_epoch}/{self.total_epochs} epochs ({self.current_epoch/self.total_epochs*100:.1f}%)")
        
        # Show performance summary
        if len(self.epoch_times) > 1:
            avg_epoch_time = np.mean(self.epoch_times[1:])  # Skip first epoch
            estimated_total_time = avg_epoch_time * (self.total_epochs - self.current_epoch)
            print(f"    ⏱️  Average epoch time: {avg_epoch_time:.1f}s")
            print(f"    🎯 Estimated completion: {estimated_total_time:.1f}s ({estimated_total_time/3600:.1f}h)")
        
        print(f"{'='*60}")
        
    def _get_gpu_info(self):
        """Get current GPU utilization information"""
        try:
            if not torch.cuda.is_available():
                return "CPU only"
            
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                # Get GPU utilization using nvidia-smi
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                         '--format=csv,noheader,nounits', '-i', str(i)], 
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        gpu_data = result.stdout.strip().split(', ')
                        if len(gpu_data) >= 3:
                            util, mem_used, mem_total = gpu_data[:3]
                            gpu_info.append(f"GPU{i}: {util}% | {mem_used}/{mem_total}MB")
                        else:
                            gpu_info.append(f"GPU{i}: N/A")
                    else:
                        gpu_info.append(f"GPU{i}: Error")
                except:
                    # Fallback to PyTorch memory info
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**2   # MB
                    gpu_info.append(f"GPU{i}: {memory_allocated:.0f}/{memory_reserved:.0f}MB")
            
            return " | ".join(gpu_info)
        except Exception as e:
            return f"Error: {e}"
    
    def _get_memory_info(self):
        """Get current memory usage information"""
        try:
            if not torch.cuda.is_available():
                return "CPU only"
            
            # Get main GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved_memory = torch.cuda.memory_reserved() / 1024**3   # GB
            
            return f"{allocated_memory:.1f}GB/{reserved_memory:.1f}GB/{total_memory:.1f}GB (alloc/reserved/total)"
        except Exception as e:
            return f"Error: {e}"
    
    def get_performance_summary(self):
        """Get overall training performance summary"""
        if not self.epoch_times:
            return "No training data available"
        
        total_time = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times)
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0
        avg_loss = np.mean(self.losses) if self.losses else 0
        
        return {
            'total_time': total_time,
            'avg_epoch_time': avg_epoch_time,
            'avg_batch_time': avg_batch_time,
            'avg_loss': avg_loss,
            'epochs_completed': len(self.epoch_times),
            'total_batches': self.total_batches_completed
        }

class PrismTrainer:
    """Main trainer class for Prism network using TrainingInterface"""
    
    def __init__(self, config_path: str, data_path: str, output_dir: str, resume_from: str = None):
        """Initialize trainer with configuration and data paths"""
        # Configure logging first
        logging.basicConfig(
            level=logging.WARNING,  # Changed from INFO to WARNING to show more important messages
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set specific logger levels for more detailed output
        self.logger.setLevel(logging.INFO)  # Set this logger to INFO level
        
        # Also enable warnings from other libraries
        logging.getLogger('torch').setLevel(logging.WARNING)
        logging.getLogger('torch.cuda').setLevel(logging.INFO)
        logging.getLogger('h5py').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        
        # Show warnings immediately
        import warnings
        warnings.filterwarnings('always', category=UserWarning)
        warnings.filterwarnings('always', category=DeprecationWarning)
        
        print("🔍 Logging configured: WARNING level enabled for system messages")
        print("📝 Training logger set to INFO level for detailed training output")
        
        # Display current logging configuration
        self._display_logging_config()
        
        self.config_path = config_path
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from = resume_from
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup device and multi-GPU configuration
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.num_gpus = torch.cuda.device_count()
            self.logger.info(f"CUDA available: {self.num_gpus} GPUs detected")
            
            # Check if multi-GPU is enabled in config
            if self.config.get('performance', {}).get('enable_distributed', False):
                self.use_multi_gpu = True
                self.logger.info(f"Multi-GPU training enabled with {self.num_gpus} GPUs")
            else:
                self.use_multi_gpu = False
                self.logger.info(f"Single GPU training on GPU 0")
        else:
            self.device = torch.device('cpu')
            self.num_gpus = 0
            self.use_multi_gpu = False
            self.logger.info("CUDA not available, using CPU")
        
        self.logger.info(f"Using device: {self.device}")
        
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
        
        # Progress monitor will be initialized after dataloader is created in _load_data()
        self.progress_monitor = None
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        
        # Display ray tracer information
        self._display_ray_tracer_info()
        
        # Display checkpoint information
        self._display_checkpoint_info()
    
    def _complex_mse_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Custom MSE loss for complex-valued tensors that always returns a tensor."""
        # Handle complex tensors by separating real and imaginary parts
        if predictions.is_complex():
            pred_real = predictions.real
            pred_imag = predictions.imag
        else:
            pred_real = predictions
            pred_imag = torch.zeros_like(predictions)
        
        if targets.is_complex():
            target_real = targets.real
            target_imag = targets.imag
        else:
            target_real = targets
            target_imag = torch.zeros_like(targets)
        
        # Compute MSE for real and imaginary parts
        real_loss = nn.functional.mse_loss(pred_real, target_real, reduction='mean')
        imag_loss = nn.functional.mse_loss(pred_imag, target_imag, reduction='mean')
        
        # Combine losses
        total_loss = real_loss + imag_loss
        
        # Ensure we return a tensor with gradients
        if not isinstance(total_loss, torch.Tensor):
            # Create tensor that maintains gradients
            total_loss = torch.tensor(total_loss, device=predictions.device, dtype=predictions.dtype, requires_grad=True)
        elif not total_loss.requires_grad:
            # If tensor exists but doesn't have gradients, ensure it does
            total_loss.requires_grad_(True)
        
        return total_loss
    
    def _load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set logging level from config
        log_level = config.get('logging', {}).get('log_level', 'INFO')
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)
        
        self.logger.info(f"Loaded configuration from {self.config_path}")
        self.logger.info(f"Logging level set to: {log_level}")
        return config
    
    def _setup_model(self):
        """Initialize Prism network model and TrainingInterface"""
        nn_config = self.config['neural_networks']
        rt_config = self.config['ray_tracing']
        
        # Create PrismNetwork with configuration from YAML
        self.logger.info(f"Creating PrismNetwork with:")
        self.logger.info(f"  num_subcarriers: {nn_config['attenuation_decoder']['output_dim']}")
        self.logger.info(f"  num_ue_antennas: {nn_config['attenuation_decoder']['num_ue_antennas']}")
        self.logger.info(f"  num_bs_antennas: {nn_config['antenna_codebook']['num_antennas']}")
        self.logger.info(f"  position_dim: {nn_config['attenuation_network']['input_dim']}")
        self.logger.info(f"  hidden_dim: {nn_config['attenuation_network']['hidden_dim']}")
        self.logger.info(f"  feature_dim: {nn_config['attenuation_network']['feature_dim']}")
        self.logger.info(f"  antenna_embedding_dim: {nn_config['antenna_codebook']['embedding_dim']}")
        
        self.prism_network = PrismNetwork(
            num_subcarriers=nn_config['attenuation_decoder']['output_dim'],
            num_ue_antennas=nn_config['attenuation_decoder']['num_ue_antennas'],
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
        
        self.logger.info(f"PrismNetwork created successfully")
        self.logger.info(f"  num_subcarriers: {self.prism_network.num_subcarriers}")
        self.logger.info(f"  num_ue_antennas: {self.prism_network.num_ue_antennas}")
        self.logger.info(f"  num_bs_antennas: {self.prism_network.num_bs_antennas}")
        
        # Create Ray Tracer with PrismNetwork for MLP-based direction selection
        # Read parallel processing configuration from config file
        parallel_config = self.config.get('performance', {})
        ray_tracer_config = self.config.get('ray_tracer_integration', {})
        
        # Check if CUDA ray tracer should be used
        use_cuda_ray_tracer = ray_tracer_config.get('use_cuda_ray_tracer', False)
        
        # Parallel processing settings with fallback to config values
        enable_parallel = parallel_config.get('enable_parallel_processing', True)
        max_workers = parallel_config.get('num_workers', 4)
        
        # Get ray tracing mode from training interface configuration
        training_interface_config = self.config.get('training_interface', {})
        ray_tracing_mode = training_interface_config.get('ray_tracing_mode', 'hybrid')
        
        # Configure parallel processing based on ray tracing mode
        if ray_tracing_mode == 'cuda':
            # CUDA mode: disable parallel processing to avoid conflicts
            enable_parallel = False
            self.logger.info("🔒 CUDA mode: parallel processing disabled to avoid device conflicts")
        elif ray_tracing_mode == 'cpu':
            # CPU mode: enable parallel processing for performance
            enable_parallel = ray_tracer_config.get('parallel_antenna_processing', True)
            self.logger.info("🚀 CPU mode: parallel processing enabled for performance")
        else:  # hybrid mode
            # Hybrid mode: use configured parallel processing
            enable_parallel = ray_tracer_config.get('parallel_antenna_processing', True)
            self.logger.info("⚖️  Hybrid mode: using configured parallel processing")
        
        # Override with ray_tracer_integration settings if available
        if 'parallel_antenna_processing' in ray_tracer_config:
            enable_parallel = ray_tracer_config['parallel_antenna_processing']
        if 'num_workers' in ray_tracer_config:
            max_workers = ray_tracer_config['num_workers']
        
        self.logger.info(f"Ray tracer configuration:")
        self.logger.info(f"  - Type: {'CUDA' if use_cuda_ray_tracer else 'CPU'}")
        self.logger.info(f"  - Ray tracing mode: {ray_tracing_mode}")
        self.logger.info(f"  - Parallel processing: {enable_parallel}")
        self.logger.info(f"  - Max workers: {max_workers}")
        
        # Create ray tracer based on configuration
        if use_cuda_ray_tracer and torch.cuda.is_available():
            self.logger.info("🚀 Using CUDA-accelerated ray tracer for maximum performance")
            self.ray_tracer = CUDARayTracer(
                azimuth_divisions=rt_config['azimuth_divisions'],
                elevation_divisions=rt_config['elevation_divisions'],
                max_ray_length=rt_config.get('max_ray_length', 100.0),
                scene_size=rt_config.get('scene_size', 200.0),
                device=self.device.type,
                prism_network=self.prism_network,  # Enable MLP-based direction selection
                signal_threshold=rt_config.get('signal_threshold', 1e-6),
                enable_early_termination=rt_config.get('enable_early_termination', True),
                uniform_samples=rt_config.get('uniform_samples', 128),
                resampled_points=rt_config.get('resampled_points', 64),
                enable_parallel_processing=enable_parallel,  # Configured based on mode
                max_workers=max_workers  # Read from config
            )
        else:
            if use_cuda_ray_tracer and not torch.cuda.is_available():
                self.logger.warning("⚠️  CUDA ray tracer requested but CUDA not available, falling back to CPU version")
            self.logger.info("💻 Using CPU ray tracer")
            self.ray_tracer = CPURayTracer(
                azimuth_divisions=rt_config['azimuth_divisions'],
                elevation_divisions=rt_config['elevation_divisions'],
                max_ray_length=rt_config.get('max_ray_length', 100.0),
                scene_size=rt_config.get('scene_size', 200.0),
                device=self.device.type,
                prism_network=self.prism_network,  # Enable MLP-based direction selection
                signal_threshold=rt_config.get('signal_threshold', 1e-6),
                enable_early_termination=rt_config.get('enable_early_termination', True),
                top_k_directions=rt_config.get('top_k_directions', None),  # Use configured K value
                enable_parallel_processing=enable_parallel,  # Configured based on mode
                max_workers=max_workers  # Read from config
            )
        
        # Get ray tracing mode from training interface configuration
        training_interface_config = self.config.get('training_interface', {})
        ray_tracing_mode = training_interface_config.get('ray_tracing_mode', 'hybrid')
        
        # Create PrismTrainingInterface with ray tracing mode
        self.model = PrismTrainingInterface(
            prism_network=self.prism_network,
            ray_tracer=self.ray_tracer,
            num_sampling_points=rt_config.get('spatial_sampling', 64),
            scene_bounds=rt_config.get('scene_bounds', None),
            subcarrier_sampling_ratio=training_interface_config.get('subcarrier_sampling_ratio', 0.3),
            checkpoint_dir=str(self.output_dir / 'checkpoints'),
            ray_tracing_mode=ray_tracing_mode
        )
        
        # Log configuration details
        subcarrier_ratio = training_interface_config.get('subcarrier_sampling_ratio', 0.3)
        total_subcarriers = self.prism_network.num_subcarriers
        selected_subcarriers = int(total_subcarriers * subcarrier_ratio)
        
        self.logger.info(f"Training interface created with ray_tracing_mode: {ray_tracing_mode}")
        self.logger.info(f"Subcarrier sampling: {subcarrier_ratio} ({subcarrier_ratio*100}%) = {selected_subcarriers}/{total_subcarriers} subcarriers")
        
        # Enable multi-GPU training if configured
        if self.use_multi_gpu and self.num_gpus > 1:
            self.logger.info(f"🚀 Wrapping model with DataParallel for {self.num_gpus} GPUs")
            # Store the original model for attribute access
            self.original_model = self.model
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.num_gpus)))
            self.logger.info(f"Multi-GPU model created successfully")
        else:
            self.logger.info("Single GPU training mode")
            self.original_model = self.model
        
        self.model = self.model.to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"TrainingInterface created with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
    def _setup_training(self):
        """Setup training hyperparameters and optimizers"""
        # Training hyperparameters
        self.batch_size = self.config['performance']['batch_size']
        
        # Scale batch size for multi-GPU training
        if self.use_multi_gpu and self.num_gpus > 1:
            original_batch_size = self.batch_size
            self.batch_size = self.batch_size * self.num_gpus
            self.logger.info(f"Multi-GPU batch size scaling: {original_batch_size} × {self.num_gpus} = {self.batch_size}")
        
        self.learning_rate = 1e-4
        self.num_epochs = self.config['training']['num_epochs']  # Read from config
        self.save_interval = 10
        
        # Deadlock detection settings
        self.batch_timeout = 600  # 10 minutes per batch
        self.progress_check_interval = 30  # Check progress every 30 seconds
        
        # Loss function for complex-valued outputs - ensure it returns tensors
        self.criterion = self._complex_mse_loss
        
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
        
        self.logger.info(f"Training setup: batch_size={self.batch_size}, lr={self.learning_rate}")
        
        # GPU monitoring setup
        self.gpu_monitoring_active = False
        self.gpu_monitor_thread = None
        
    def _get_gpu_utilization(self) -> str:
        """Get current GPU utilization and memory usage"""
        try:
            if not torch.cuda.is_available():
                return "CPU only"
            
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                gpu_info.append(f"GPU{i}: {memory_allocated:.1f}GB/{memory_reserved:.1f}GB")
            
            return " | ".join(gpu_info)
        except Exception as e:
            return f"Error: {e}"
    
    def _get_memory_usage(self) -> str:
        """Get current memory usage for the model"""
        try:
            if not torch.cuda.is_available():
                return "CPU only"
            
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            return f"{allocated_memory:.1f}GB/{total_memory:.1f}GB"
        except Exception as e:
            return f"Error: {e}"
    
    def _start_gpu_monitoring(self):
        """Start GPU monitoring thread"""
        if not torch.cuda.is_available():
            return
        
        self.gpu_monitoring_active = True
        print(f"🔍 GPU Monitoring Started | Devices: {torch.cuda.device_count()}")
        
    def _stop_gpu_monitoring(self):
        """Stop GPU monitoring"""
        self.gpu_monitoring_active = False
        
    def _check_training_progress(self, start_time: float, batch_idx: int, total_batches: int) -> bool:
        """Check if training is progressing normally or stuck"""
        elapsed_time = time.time() - start_time
        
        # Check if we're taking too long on a batch
        if elapsed_time > self.batch_timeout:
            self.logger.warning(f"⚠️  Batch {batch_idx} taking too long ({elapsed_time:.1f}s > {self.batch_timeout}s)")
            return False
        
        # Check if we're making reasonable progress
        if batch_idx > 0:
            avg_time_per_batch = elapsed_time / batch_idx
            estimated_total_time = avg_time_per_batch * total_batches
            if estimated_total_time > 3600:  # More than 1 hour
                self.logger.warning(f"⚠️  Slow training progress: estimated {estimated_total_time/3600:.1f} hours total")
        
        return True
        
    def _validate_data_integrity(self):
        """Validate HDF5 data integrity before training"""
        self.logger.info("Validating data integrity...")
        print("🔍 Validating data integrity...")
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                # Check if file is corrupted
                if f.id.get_access_plist().get_fclose_degree() == h5py.h5f.CLOSE_STRONG:
                    self.logger.info("HDF5 file uses strong closing, checking for corruption...")
                
                # Validate dataset shapes and types
                ue_positions = f['positions']['ue_positions']
                csi_data = f['channel_data']['channel_responses']
                bs_position = f['positions']['bs_position']
                
                print(f"   📊 Dataset validation:")
                print(f"      UE positions: {ue_positions.shape} - {ue_positions.dtype}")
                print(f"      CSI data: {csi_data.shape} - {csi_data.dtype}")
                print(f"      BS position: {bs_position.shape} - {bs_position.dtype}")
                
                # Check for NaN or infinite values
                ue_nan_count = np.isnan(ue_positions[:]).sum()
                csi_nan_count = np.isnan(csi_data[:]).sum()
                bs_nan_count = np.isnan(bs_position[:]).sum()
                
                ue_inf_count = np.isinf(ue_positions[:]).sum()
                csi_inf_count = np.isinf(csi_data[:]).sum()
                bs_inf_count = np.isinf(bs_position[:]).sum()
                
                print(f"   🔍 Data quality check:")
                print(f"      NaN values - UE: {ue_nan_count}, CSI: {csi_nan_count}, BS: {bs_nan_count}")
                print(f"      Inf values - UE: {ue_inf_count}, CSI: {csi_inf_count}, BS: {bs_inf_count}")
                
                if ue_nan_count > 0 or csi_nan_count > 0 or bs_nan_count > 0:
                    self.logger.warning(f"⚠️  Found NaN values in data: UE={ue_nan_count}, CSI={csi_nan_count}, BS={bs_nan_count}")
                    print(f"   ⚠️  Warning: Found NaN values in data")
                
                if ue_inf_count > 0 or csi_inf_count > 0 or bs_inf_count > 0:
                    self.logger.warning(f"⚠️  Found infinite values in data: UE={ue_inf_count}, CSI={csi_inf_count}, BS={bs_inf_count}")
                    print(f"   ⚠️  Warning: Found infinite values in data")
                
                # Check data ranges
                ue_min, ue_max = ue_positions[:].min(), ue_positions[:].max()
                csi_min, csi_max = csi_data[:].min(), csi_data[:].max()
                bs_min, bs_max = bs_position[:].min(), bs_position[:].max()
                
                print(f"   📏 Data ranges:")
                print(f"      UE positions: [{ue_min:.2f}, {ue_max:.2f}]")
                print(f"      CSI data: [{csi_min:.2e}, {csi_max:.2e}]")
                print(f"      BS position: [{bs_min:.2f}, {bs_max:.2f}]")
                
                # Validate data consistency
                if csi_data.shape[0] != ue_positions.shape[0]:
                    raise ValueError(f"Data mismatch: {csi_data.shape[0]} CSI samples vs {ue_positions.shape[0]} UE positions")
                
                print(f"   ✅ Data integrity validation passed")
                self.logger.info("Data integrity validation passed")
                
        except Exception as e:
            self.logger.error(f"Data integrity validation failed: {e}")
            print(f"   ❌ Data integrity validation failed: {e}")
            raise
    
    def _adjust_training_parameters(self):
        """Adjust training parameters if data loading issues are detected"""
        self.logger.info("Adjusting training parameters for stability...")
        print("⚙️  Adjusting training parameters for stability...")
        
        # Reduce batch size if it's too large
        original_batch_size = self.batch_size
        if self.batch_size > 4:
            self.batch_size = max(1, self.batch_size // 2)
            self.logger.info(f"Reduced batch size from {original_batch_size} to {self.batch_size}")
            print(f"   📦 Batch size reduced: {original_batch_size} → {self.batch_size}")
        
        # Reduce learning rate if it's too high
        original_lr = self.learning_rate
        if self.learning_rate > 1e-3:
            self.learning_rate = self.learning_rate / 2
            self.logger.info(f"Reduced learning rate from {original_lr:.2e} to {self.learning_rate:.2e}")
            print(f"   📚 Learning rate reduced: {original_lr:.2e} → {self.learning_rate:.2e}")
            
            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        
        print(f"   ✅ Training parameters adjusted for stability")
    
    def _load_data(self):
        """Load training data from HDF5 file"""
        self.logger.info(f"Loading training data from {self.data_path}")
        print(f"📂 Loading data from: {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            # Load UE positions from nested group
            ue_positions = f['positions']['ue_positions'][:]
            self.logger.info(f"Loaded {len(ue_positions)} UE positions")
            print(f"   📍 UE positions: {len(ue_positions)} samples")
            
            # Load channel responses (CSI) from nested group
            csi_data = f['channel_data']['channel_responses'][:]
            self.logger.info(f"Loaded CSI data with shape: {csi_data.shape}")
            print(f"   📡 CSI data: {csi_data.shape}")
            
            # Load BS position from nested group
            bs_position = f['positions']['bs_position'][:]
            self.logger.info(f"BS position: {bs_position}")
            print(f"   🏢 BS position: {bs_position}")
            
            # Load antenna indices if available
            if 'antenna_indices' in f:
                antenna_indices = f['antenna_indices'][:]
                self.logger.info(f"Loaded antenna indices with shape: {antenna_indices.shape}")
                print(f"   📡 Antenna indices: {len(antenna_indices)}")
            else:
                # Create default antenna indices if not available
                num_bs_antennas = csi_data.shape[3] if len(csi_data.shape) > 3 else 64  # Shape is (100, 408, 4, 64)
                antenna_indices = np.arange(num_bs_antennas)
                self.logger.info(f"Created default antenna indices: {len(antenna_indices)}")
                print(f"   📡 Created default antenna indices: {len(antenna_indices)}")
            
            # Load simulation parameters if available
            if 'simulation_config' in f and hasattr(f['simulation_config'], 'attrs'):
                params = dict(f['simulation_config'].attrs)
                self.logger.info(f"Simulation parameters: {params}")
                print(f"   ⚙️  Simulation parameters loaded")
            else:
                self.logger.info("No simulation parameters found")
                print(f"   ⚙️  No simulation parameters found")
            
            # Check if this is split data
            if 'split_type' in f.attrs:
                split_info = dict(f.attrs)
                self.logger.info(f"Data split info: {split_info}")
                print(f"   📊 Data split: {split_info.get('split_type', 'unknown')} ({split_info.get('num_samples', 'unknown')} samples)")
        
        # Convert to tensors with proper data types
        self.ue_positions = torch.tensor(ue_positions, dtype=torch.float32)
        self.csi_data = torch.tensor(csi_data, dtype=torch.complex64)
        self.bs_position = torch.tensor(bs_position, dtype=torch.float32)
        self.antenna_indices = torch.tensor(antenna_indices, dtype=torch.long)
        
        # Validate data shapes
        self.logger.info(f"Data validation:")
        self.logger.info(f"  UE positions: {self.ue_positions.shape} - {self.ue_positions.dtype}")
        self.logger.info(f"  CSI data: {self.csi_data.shape} - {self.csi_data.dtype}")
        self.logger.info(f"  BS position: {self.bs_position.shape} - {self.bs_position.dtype}")
        self.logger.info(f"  Antenna indices: {self.antenna_indices.shape} - {self.antenna_indices.dtype}")
        
        # Check for data consistency
        if self.csi_data.shape[0] != self.ue_positions.shape[0]:
            raise ValueError(f"Data mismatch: {self.csi_data.shape[0]} CSI samples vs {self.ue_positions.shape[0]} UE positions")
        if self.csi_data.shape[3] != self.antenna_indices.shape[0]:
            raise ValueError(f"Data mismatch: {self.csi_data.shape[3]} BS antennas vs {self.antenna_indices.shape[0]} antenna indices")
        
        # Create dataset with all required data
        self.dataset = TensorDataset(
            self.ue_positions, 
            self.bs_position.expand(len(ue_positions), -1),
            self.antenna_indices.expand(len(ue_positions), -1),
            self.csi_data
        )
        
        # Validate batch size
        if self.batch_size > len(self.dataset):
            self.logger.warning(f"Batch size ({self.batch_size}) is larger than dataset size ({len(self.dataset)}). Adjusting batch size.")
            self.batch_size = len(self.dataset)
            self.logger.info(f"Adjusted batch size to: {self.batch_size}")
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,  # Reduced from 4 to prevent StopIteration errors
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            drop_last=False  # Don't drop incomplete batches
        )
        
        self.logger.info(f"Training data loaded: {len(self.dataset)} samples, batch_size={self.batch_size}")
        print(f"✅ Data loaded successfully!")
        print(f"   📊 Dataset size: {len(self.dataset)} samples")
        print(f"   🔄 Batch size: {self.batch_size}")
        print(f"   📦 Number of batches: {len(self.dataloader)}")
        print(f"   💾 Data types: UE (float32), CSI (complex64), BS (float32), Antenna (long)")
        
        # Initialize progress monitor AFTER dataloader is created
        self.progress_monitor = TrainingProgressMonitor(
            total_epochs=self.config['training']['num_epochs'],
            total_batches_per_epoch=len(self.dataloader)
        )
        print(f"   📊 Progress monitor initialized for {len(self.dataloader)} batches per epoch")
        
    def _train_epoch(self, epoch: int):
        """Train for one epoch using TrainingInterface"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Start epoch monitoring if progress monitor is available
        if hasattr(self, 'progress_monitor') and self.progress_monitor is not None:
            self.progress_monitor.start_epoch(epoch)
        
        # Start progress monitoring
        epoch_start_time = time.time()
        
        try:
            for batch_idx, (ue_pos, bs_pos, antenna_idx, csi_target) in enumerate(self.dataloader):
                # Start batch monitoring if progress monitor is available
                if hasattr(self, 'progress_monitor') and self.progress_monitor is not None:
                    self.progress_monitor.start_batch(batch_idx)
                
                self.logger.debug(f"Processing batch {batch_idx}:")
                self.logger.debug(f"  ue_pos shape: {ue_pos.shape}, dtype: {ue_pos.dtype}")
                self.logger.debug(f"  bs_pos shape: {bs_pos.shape}, dtype: {bs_pos.dtype}")
                self.logger.debug(f"  antenna_idx shape: {antenna_idx.shape}, dtype: {antenna_idx.dtype}")
                self.logger.debug(f"  csi_target shape: {csi_target.shape}, dtype: {csi_target.shape}")
                
                # Validate batch data
                if torch.isnan(ue_pos).any() or torch.isnan(bs_pos).any() or torch.isnan(csi_target).any():
                    self.logger.warning(f"⚠️  Batch {batch_idx} contains NaN values, skipping...")
                    continue
                
                if torch.isinf(ue_pos).any() or torch.isinf(bs_pos).any() or torch.isinf(csi_target).any():
                    self.logger.warning(f"⚠️  Batch {batch_idx} contains infinite values, skipping...")
                    continue
                
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
                    try:
                        loss = self.model.compute_loss(csi_pred, csi_target, self.criterion)
                        
                        # Validate loss is a tensor
                        if not isinstance(loss, torch.Tensor):
                            self.logger.error(f"Loss computation returned non-tensor: {type(loss)} = {loss}")
                            raise ValueError(f"Loss must be a torch.Tensor, got {type(loss)}")
                        
                        # Ensure loss has requires_grad for backward pass
                        if not loss.requires_grad:
                            self.logger.warning("Loss tensor does not require gradients, this may cause issues")
                        
                        # Check for NaN or infinite values
                        if torch.isnan(loss) or torch.isinf(loss):
                            self.logger.error(f"Invalid loss value: {loss}")
                            raise ValueError(f"Loss contains NaN or infinite values: {loss}")
                        
                    except Exception as e:
                        self.logger.error(f"Loss computation failed: {e}")
                        self.logger.error(f"Shapes - csi_pred: {csi_pred.shape}, csi_target: {csi_target.shape}")
                        raise
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Update training state in TrainingInterface
                    self.model.update_training_state(epoch, batch_idx, loss.item())
                    
                    # Update progress monitor with real-time information if available
                    if hasattr(self, 'progress_monitor') and self.progress_monitor is not None:
                        self.progress_monitor.update_batch_progress(batch_idx, loss.item(), len(self.dataloader))
                    else:
                        # Fallback progress logging
                        if batch_idx % 5 == 0 or batch_idx == len(self.dataloader) - 1:
                            progress = (batch_idx + 1) / len(self.dataloader) * 100
                            avg_loss_so_far = total_loss / num_batches
                            print(f"  📊 Batch {batch_idx+1:3d}/{len(self.dataloader):3d} ({progress:5.1f}%) | "
                                  f"Loss: {loss.item():.6f} | Avg: {avg_loss_so_far:.6f}")
                    
                    # Check for potential deadlocks or slow progress
                    if not self._check_training_progress(epoch_start_time, batch_idx, len(self.dataloader)):
                        print(f"  ⚠️  Training progress check failed - consider restarting if stuck")
                    
                    # Log to tensorboard
                    self.writer.add_scalar(f'Loss/Batch_{epoch}', loss.item(), batch_idx)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in batch {batch_idx}: {e}")
                    print(f"  ❌ Batch {batch_idx} failed: {e}")
                    
                    # Try to continue with next batch instead of crashing
                    if batch_idx < len(self.dataloader) - 1:
                        print(f"  🔄 Continuing with next batch...")
                        continue
                    else:
                        print(f"  🛑 Last batch failed, ending epoch early")
                        break
                        
        except StopIteration as e:
            self.logger.error(f"❌ StopIteration error in epoch {epoch}: {e}")
            print(f"  ❌ StopIteration error: {e}")
            print(f"  🔍 This usually indicates a data loading issue")
            print(f"  💡 Try reducing batch_size or num_workers")
            
            # Try to recover by reinitializing the dataloader
            try:
                print(f"  🔄 Attempting to recover by reinitializing dataloader...")
                self.dataloader = DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=1,  # Use single worker for recovery
                    pin_memory=True,
                    persistent_workers=False,  # Disable persistent workers
                    drop_last=False
                )
                print(f"  ✅ Dataloader reinitialized successfully")
            except Exception as recovery_error:
                self.logger.error(f"Failed to recover dataloader: {recovery_error}")
                print(f"  ❌ Failed to recover: {recovery_error}")
                raise
                
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in epoch {epoch}: {e}")
            print(f"  ❌ Unexpected error: {e}")
            raise
        
        if num_batches == 0:
            self.logger.error(f"No successful batches in epoch {epoch}")
            raise RuntimeError(f"Epoch {epoch} failed - no successful batches")
        
        avg_loss = total_loss / num_batches
        self.logger.info(f"Epoch {epoch} completed: {num_batches} batches, avg loss: {avg_loss:.6f}")
        
        # End epoch monitoring if progress monitor is available
        if hasattr(self, 'progress_monitor') and self.progress_monitor is not None:
            self.progress_monitor.end_epoch(avg_loss)
        
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
                    try:
                        loss = self.model.compute_loss(csi_pred, csi_target, self.criterion)
                        
                        # Validate loss is a tensor
                        if not isinstance(loss, torch.Tensor):
                            self.logger.error(f"Validation loss computation returned non-tensor: {type(loss)} = {loss}")
                            raise ValueError(f"Validation loss must be a torch.Tensor, got {type(loss)}")
                        
                        # Check for NaN or infinite values
                        if torch.isnan(loss) or torch.isinf(loss):
                            self.logger.error(f"Invalid validation loss value: {loss}")
                            raise ValueError(f"Validation loss contains NaN or infinite values: {loss}")
                        
                    except Exception as e:
                        self.logger.error(f"Validation loss computation failed: {e}")
                        self.logger.error(f"Shapes - csi_pred: {csi_pred.shape}, csi_target: {csi_target.shape}")
                        raise
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Validation error: {e}")
                    self.logger.error(f"Full validation traceback:")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    print(f"  ❌ Validation batch failed: {e}")
                    print(f"  🔍 Check logs for full traceback")
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
        self.logger.info(f"Training state saved: {checkpoint_path}")
        
        # Save best model
        if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = self.output_dir / 'best_model.pt'
            self.model.save_checkpoint('best_model.pt')
            torch.save(training_state, str(best_model_path).replace('.pt', '_state.pt'))
            self.logger.info(f"Best model saved: {best_model_path}")
            print(f"🏆 New best model saved! (Val Loss: {val_loss:.6f})")
        
        # Save latest checkpoint for resuming
        latest_checkpoint_path = self.output_dir / 'latest_checkpoint.pt'
        self.model.save_checkpoint('latest_checkpoint.pt')
        torch.save(training_state, str(latest_checkpoint_path).replace('.pt', '_state.pt'))
        self.logger.info(f"Latest checkpoint saved: {latest_checkpoint_path}")
        
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
            self.logger.warning(f"Emergency checkpoint failed: {e}")
    
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
                self.logger.info(f"Removed old checkpoint: {checkpoint}")
        
        # Clean up training state files
        training_states = list(self.output_dir.glob('training_state_epoch_*.pt'))
        if len(training_states) > 5:
            training_states.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for state in training_states[:-5]:
                state.unlink()
                self.logger.info(f"Removed old training state: {state}")
    
    def _resume_from_checkpoint(self):
        """Resume training from a checkpoint using TrainingInterface"""
        self.logger.info(f"Resuming training from checkpoint: {self.resume_from}")
        
        if not os.path.exists(self.resume_from):
            self.logger.error(f"Checkpoint file not found: {self.resume_from}")
            raise FileNotFoundError(f"Checkpoint file not found: {self.resume_from}")
        
        try:
            # Load TrainingInterface checkpoint
            self.model.load_checkpoint(self.resume_from)
            self.logger.info("TrainingInterface checkpoint loaded successfully")
            
            # Load training state if available
            training_state_path = self.resume_from.replace('.pt', '_state.pt')
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path, map_location=self.device)
                
                # Load optimizer state
                self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
                self.logger.info("Optimizer state loaded successfully")
                
                # Load scheduler state
                self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
                self.logger.info("Scheduler state loaded successfully")
                
                # Load training state
                self.start_epoch = training_state['epoch'] + 1
                self.best_val_loss = training_state.get('val_loss', float('inf'))
                
                self.logger.info(f"Resuming from epoch {self.start_epoch}")
                self.logger.info(f"Best validation loss so far: {self.best_val_loss:.6f}")
                
                # Load training history if available
                if 'train_losses' in training_state:
                    self.train_losses = training_state['train_losses']
                    self.val_losses = training_state['val_losses']
                    self.logger.info(f"Loaded training history: {len(self.train_losses)} epochs")
                else:
                    self.train_losses = []
                    self.val_losses = []
            else:
                # Fallback to TrainingInterface state
                self.start_epoch = self.model.current_epoch + 1
                self.best_val_loss = self.model.best_loss
                self.train_losses = []
                self.val_losses = []
                self.logger.info(f"Resuming from TrainingInterface state: epoch {self.start_epoch}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
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
        self.logger.info("Starting training with TrainingInterface...")
        print("\n🚀 Starting Prism Network Training with TrainingInterface")
        print("=" * 80)
        
        # Load data
        print("📂 Loading training data...")
        self._validate_data_integrity()  # Validate data integrity first
        self._load_data()
        self._adjust_training_parameters()  # Adjust parameters for stability
        print(f"✅ Data loaded: {len(self.dataset)} samples")
        
        # Initialize training state
        if hasattr(self, 'start_epoch'):
            start_epoch = self.start_epoch
            train_losses = self.train_losses
            val_losses = self.val_losses
            self.logger.info(f"Resuming training from epoch {start_epoch}")
            print(f"🔄 Resuming training from epoch {start_epoch}")
        else:
            start_epoch = 1
            train_losses = []
            val_losses = []
            self.logger.info("Starting training from epoch 1")
            print("🆕 Starting training from epoch 1")
        
        print(f"\n📈 Training Configuration:")
        print(f"   • Total epochs: {self.num_epochs}")
        print(f"   • Batch size: {self.batch_size}")
        print(f"   • Learning rate: {self.learning_rate:.2e}")
        print(f"   • Device: {self.device}")
        print(f"   • Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Display GPU configuration
        if torch.cuda.is_available():
            print(f"   • Multi-GPU: {'Enabled' if self.use_multi_gpu else 'Disabled'}")
            print(f"   • GPU Count: {self.num_gpus}")
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   • GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
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
            self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
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
                self.logger.info("Learning rate too low, stopping training")
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
        
        # Display final training summary
        self._display_training_summary()
        
        self.logger.info("Training completed!")
        self.writer.close()
    
    def _display_training_summary(self):
        """Display comprehensive training summary with performance metrics"""
        print("\n" + "="*80)
        print("🎯 TRAINING COMPLETED - FINAL SUMMARY")
        print("="*80)
        
        # Get performance summary from progress monitor
        if hasattr(self, 'progress_monitor') and self.progress_monitor is not None:
            summary = self.progress_monitor.get_performance_summary()
            if isinstance(summary, dict):
                print(f"⏱️  Total Training Time: {summary['total_time']/3600:.2f} hours ({summary['total_time']:.1f}s)")
                print(f"📊 Epochs Completed: {summary['epochs_completed']}")
                print(f"🔄 Total Batches: {summary['total_batches']}")
                print(f"⚡ Average Epoch Time: {summary['avg_epoch_time']:.1f}s")
                print(f"🚀 Average Batch Time: {summary['avg_batch_time']:.2f}s")
                print(f"📈 Final Average Loss: {summary['avg_loss']:.6f}")
            else:
                print(f"📊 Performance Summary: {summary}")
        else:
            print("📊 Performance Summary: Progress monitor not available")
        
        # Show final model performance
        if hasattr(self, 'best_val_loss') and self.best_val_loss != float('inf'):
            print(f"🏆 Best Validation Loss: {self.best_val_loss:.6f}")
        
        # Show GPU utilization summary
        if torch.cuda.is_available():
            print(f"\n🔍 GPU Utilization Summary:")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({total_memory:.1f}GB)")
        
        # Show output files
        print(f"\n📁 Output Files:")
        print(f"   📊 TensorBoard logs: {self.output_dir}/tensorboard/")
        print(f"   💾 Checkpoints: {self.output_dir}/checkpoints/")
        print(f"   📈 Training plots: {self.output_dir}/")
        print(f"   📝 Training log: training.log")
        
        print("="*80)
        print("🎉 Training completed successfully!")
        print("="*80)
    
    def _display_ray_tracer_info(self):
        """Display information about the ray tracer configuration and performance."""
        print("\n🔍 Ray Tracer Configuration:")
        print("=" * 30)
        
        # Display ray tracer type and performance info
        if hasattr(self.ray_tracer, 'get_parallelization_stats'):
            try:
                stats = self.ray_tracer.get_parallelization_stats()
                print(f"  - Type: {'CUDA' if stats.get('cuda_enabled', False) else 'CPU'}")
                print(f"  - Parallel processing: {stats.get('parallel_processing_enabled', 'N/A')}")
                print(f"  - Processing mode: {stats.get('processing_mode', 'N/A')}")
                print(f"  - Max workers: {stats.get('max_workers', 'N/A')}")
                print(f"  - Total directions: {stats.get('total_directions', 'N/A')}")
            except Exception as e:
                print(f"  - Error getting parallelization stats: {e}")
        
        # Display CUDA-specific information if available
        if hasattr(self.ray_tracer, 'get_performance_info'):
            try:
                perf_info = self.ray_tracer.get_performance_info()
                print(f"  - Device: {perf_info.get('device', 'N/A')}")
                print(f"  - CUDA enabled: {perf_info.get('use_cuda', 'N/A')}")
                if perf_info.get('use_cuda', False):
                    print(f"  - CUDA device: {perf_info.get('cuda_device_name', 'N/A')}")
                    print(f"  - CUDA memory: {perf_info.get('cuda_memory_gb', 'N/A')} GB")
            except Exception as e:
                print(f"  - Error getting performance info: {e}")
        
        # Display ray count analysis if available
        if hasattr(self.ray_tracer, 'get_ray_count_analysis'):
            try:
                # Use default values for analysis
                num_bs = 1
                num_ue = 100
                num_subcarriers = 64
                analysis = self.ray_tracer.get_ray_count_analysis(num_bs, num_ue, num_subcarriers)
                print(f"  - Total rays (1 BS, 100 UE, 64 subcarriers): {analysis.get('total_rays', 'N/A'):,}")
                print(f"  - Ray count formula: {analysis.get('ray_count_formula', 'N/A')}")
            except Exception as e:
                print(f"  - Error getting ray count analysis: {e}")
        
        print("=" * 30)
    
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
        
        self.logger.info(f"Training curves saved: {plot_path}")

    def _display_checkpoint_info(self):
        """Display information about available checkpoint files."""
        print("\n📁 Available Checkpoint Files:")
        print("=" * 20)
        
        # Display TrainingInterface checkpoints
        checkpoint_dir = Path(self.original_model.checkpoint_dir)
        if checkpoint_dir.exists():
            print(f"TrainingInterface Checkpoints (in {self.original_model.checkpoint_dir}):")
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

    def _display_logging_config(self):
        """Display current logging configuration."""
        print("\n📝 Current Logging Configuration:")
        print("=" * 30)
        print(f"  - Root Logger Level: {logging.getLevelName(logging.getLogger().level)}")
        print(f"  - File Handler Level: {logging.getLevelName(logging.getLogger('training.log').level)}")
        print(f"  - Stream Handler Level: {logging.getLevelName(logging.getLogger().handlers[1].level)}")
        print("=" * 30)

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
