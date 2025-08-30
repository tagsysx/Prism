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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prism.networks.prism_network import PrismNetwork
from prism.ray_tracer_cpu import CPURayTracer
from prism.ray_tracer_cuda import CUDARayTracer
from prism.training_interface import PrismTrainingInterface
from prism.config_loader import ConfigLoader

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
        
        # Show status indicator for all batches
        print(f"  🚀 Processing training batch {batch_idx+1}...")
        
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
    
    def __init__(self, config_path: str, data_path: str = None, output_dir: str = None, resume_from: str = None, gpu_id: int = None):
        """Initialize trainer with configuration and optional data/output paths (will read from config if not provided)"""
        # Load config first to get logging configuration
        self.config_path = config_path
        self.gpu_id = gpu_id  # Store GPU ID for device setup
        
        try:
            config_loader = ConfigLoader(config_path)
            temp_config = config_loader.config
        except Exception as e:
            print(f"❌ FATAL ERROR: Failed to load configuration from {config_path}")
            print(f"   Error details: {str(e)}")
            print(f"   Please check your configuration file and ensure it exists and is valid.")
            sys.exit(1)
        
        # Get logging configuration from config file (under output.training section)
        output_config = temp_config.get('output', {})
        training_config = output_config.get('training', {})
        logging_config = training_config.get('logging', {})
        log_level_str = logging_config.get('log_level', 'INFO')
        
        # Get log file path from config (should always be available after ConfigLoader processing)
        log_file = logging_config.get('log_file')
        if not log_file:
            raise ValueError("log_file not found in configuration. Check your config file and ConfigLoader processing.")
        
        # Create log directory if it doesn't exist
        import os
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Convert string log level to logging constant
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
        # Configure logging with config file settings
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set specific logger levels for more detailed output
        self.logger.setLevel(log_level)
        
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
        
        # Store config path and load configuration first
        self.config_path = config_path
        
        # Load configuration using ConfigLoader to process template variables
        try:
            config_loader = ConfigLoader(config_path)
            self.config = config_loader.config
        except Exception as e:
            self.logger.error(f"❌ FATAL ERROR: Failed to reload configuration from {config_path}")
            self.logger.error(f"   Error details: {str(e)}")
            print(f"❌ FATAL ERROR: Configuration loading failed during trainer initialization")
            sys.exit(1)
        
        # Set data path and output directory from config if not provided
        try:
            # Import data utilities
            from prism.data_utils import check_dataset_compatibility
            
            # Check dataset configuration
            dataset_path, split_config = check_dataset_compatibility(self.config)
            
            # Use single dataset with split
            self.data_path = data_path or dataset_path
            self.split_config = split_config
            self.logger.info(f"Using single dataset with train/test split: {self.data_path}")
            self.logger.info(f"Split configuration: {self.split_config}")
                
        except KeyError as e:
            self.logger.error(f"❌ FATAL ERROR: Missing required configuration key: {e}")
            print(f"❌ FATAL ERROR: Configuration is missing required key: {e}")
            print(f"   Please check your configuration file structure.")
            sys.exit(1)
        
        # Build output directory from base_dir (new simplified structure)
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Use base_dir + training to construct output directory
            base_dir = self.config['output'].get('base_dir', 'results')
            self.output_dir = Path(base_dir) / 'training'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from = resume_from
        

        
        # Setup device and multi-GPU configuration with intelligent GPU selection
        device_config = self.config.get('system', {}).get('device', 'cuda')
        cuda_config = self.config.get('system', {}).get('cuda', {})
        
        if device_config == 'cuda' and torch.cuda.is_available():
            # GPU selection: use specified GPU ID or auto-select
            self.num_gpus = torch.cuda.device_count()
            
            if self.gpu_id is not None:
                # Use specified GPU ID
                if self.gpu_id >= self.num_gpus or self.gpu_id < 0:
                    self.logger.error(f"❌ Invalid GPU ID {self.gpu_id}. Available GPUs: 0-{self.num_gpus-1}")
                    raise ValueError(f"Invalid GPU ID {self.gpu_id}. Available GPUs: 0-{self.num_gpus-1}")
                selected_device_id = self.gpu_id
                self.logger.info(f"🎯 Using specified GPU: {selected_device_id}")
            else:
                # Auto-select best GPU
                selected_device_id = self._select_best_gpu()
                self.logger.info(f"🔍 Auto-selected GPU: {selected_device_id}")
            
            self.device = torch.device(f'cuda:{selected_device_id}')
            
            self.logger.info(f"🔍 GPU Selection Results:")
            self.logger.info(f"  • Total GPUs detected: {self.num_gpus}")
            self.logger.info(f"  • Selected GPU: {selected_device_id}")
            self.logger.info(f"  • GPU Name: {torch.cuda.get_device_name(selected_device_id)}")
            self.logger.info(f"  • GPU Memory: {torch.cuda.get_device_properties(selected_device_id).total_memory / 1024**3:.1f}GB")
            
            # Configure CUDA settings
            if cuda_config.get('benchmark_mode', True):
                torch.backends.cudnn.benchmark = True
                self.logger.info("CUDA benchmark mode enabled")
            
            if cuda_config.get('deterministic', False):
                torch.backends.cudnn.deterministic = True
                self.logger.info("CUDA deterministic mode enabled")
            
            # Check if multi-GPU is enabled in config
            if cuda_config.get('multi_gpu', False) and self.num_gpus > 1:
                self.use_multi_gpu = True
                self.available_gpus = self._get_available_gpus()
                self.logger.info(f"Multi-GPU training enabled with GPUs: {self.available_gpus}")
            else:
                self.use_multi_gpu = False
                self.available_gpus = [selected_device_id]
                self.logger.info(f"Single GPU training on GPU {selected_device_id}")
        else:
            self.device = torch.device('cpu')
            self.num_gpus = 0
            self.use_multi_gpu = False
            self.available_gpus = []
            if device_config == 'cuda':
                self.logger.warning("CUDA requested but not available, using CPU")
            else:
                self.logger.info("Using CPU as configured")
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model and training components
        self._setup_model()
        self._setup_training()
        self._setup_optimizer_and_scheduler()
        
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
        tensorboard_dir = self.config['output']['training']['tensorboard_dir']
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # Display ray tracer information
        self._display_ray_tracer_info()
        
        # Display checkpoint information
        self._display_checkpoint_info()
        
        # Display batch checkpoint settings
        self._display_batch_checkpoint_info()
    

    

    
    def _setup_model(self):
        """
        Initialize Prism network model and TrainingInterface using the cleaned configuration structure.
        """
        # Load configuration sections from the new cleaned structure
        nn_config = self.config.get('neural_networks', {})
        ray_tracing_config = self.config.get('ray_tracing', {})
        system_config = self.config.get('system', {})
        
        # Include logging configuration in system_config for TrainingInterface
        output_config = self.config.get('output', {})
        training_output_config = output_config.get('training', {})
        system_config['logging'] = training_output_config.get('logging', {})
        
        # Extract neural network sub-configurations
        attenuation_network = nn_config.get('attenuation_network', {})
        attenuation_decoder = nn_config.get('attenuation_decoder', {})
        antenna_codebook = nn_config.get('antenna_codebook', {})
        antenna_network = nn_config.get('antenna_network', {})
        radiance_network = nn_config.get('radiance_network', {})
        
        # Extract ray tracing sub-configurations
        angular_sampling = ray_tracing_config.get('angular_sampling', {})
        radial_sampling = ray_tracing_config.get('radial_sampling', {})
        subcarrier_sampling = ray_tracing_config.get('subcarrier_sampling', {})
        scene_bounds = ray_tracing_config.get('scene_bounds', {})

        # Log neural network configuration
        self.logger.info("🧠 Creating PrismNetwork with the following parameters:")
        self.logger.info(f"  📊 num_subcarriers: {attenuation_decoder.get('output_dim', 408)}")
        self.logger.info(f"  📡 num_bs_antennas: {antenna_codebook.get('num_antennas', 64)}")
        self.logger.info(f"  📱 position_dim: {attenuation_network.get('input_dim', 3)}")
        self.logger.info(f"  🔧 hidden_dim: {attenuation_network.get('hidden_dim', 256)}")
        self.logger.info(f"  ✨ feature_dim: {attenuation_network.get('feature_dim', 128)}")
        self.logger.info(f"  🎯 antenna_embedding_dim: {antenna_codebook.get('embedding_dim', 64)}")
        self.logger.info(f"  🔄 azimuth_divisions: {angular_sampling.get('azimuth_divisions', 18)}")
        self.logger.info(f"  📐 elevation_divisions: {angular_sampling.get('elevation_divisions', 9)}")

        # Get configuration sections
        base_station_config = self.config.get('base_station', {})
        ofdm_config = base_station_config.get('ofdm', {})
        user_equipment_config = self.config.get('user_equipment', {})
        
        # Create PrismNetwork instance using the cleaned configuration structure
        self.prism_network = PrismNetwork(
            num_subcarriers=ofdm_config.get('num_subcarriers', attenuation_decoder.get('output_dim', 408)),
            num_ue_antennas=user_equipment_config.get('num_ue_antennas', 4),
            num_bs_antennas=base_station_config.get('num_antennas', antenna_codebook.get('num_antennas', 64)),
            position_dim=attenuation_network.get('input_dim', 3),
            hidden_dim=attenuation_network.get('hidden_dim', 256),
            feature_dim=attenuation_network.get('feature_dim', 128),
            antenna_embedding_dim=base_station_config.get('antenna_embedding_dim', antenna_codebook.get('embedding_dim', 64)),
            use_antenna_codebook=antenna_codebook.get('learnable', True),
            use_ipe_encoding=True,  # Enable IPE encoding for better performance
            azimuth_divisions=angular_sampling.get('azimuth_divisions', 18),
            elevation_divisions=angular_sampling.get('elevation_divisions', 9),
            top_k_directions=angular_sampling.get('top_k_directions', 32),
            complex_output=True  # Enable complex output for CSI
        )
        
        # Log OFDM configuration usage
        self.logger.info(f"  📡 OFDM Configuration:")
        center_freq = float(ofdm_config.get('center_frequency', 3.5e9))
        bandwidth = float(ofdm_config.get('bandwidth', 100e6))
        num_subcarriers = int(ofdm_config.get('num_subcarriers', 408))
        subcarrier_spacing = float(ofdm_config.get('subcarrier_spacing', 245.1e3))
        
        self.logger.info(f"    • Center frequency: {center_freq/1e9:.1f} GHz")
        self.logger.info(f"    • Bandwidth: {bandwidth/1e6:.0f} MHz")
        self.logger.info(f"    • Subcarriers: {num_subcarriers}")
        self.logger.info(f"    • Subcarrier spacing: {subcarrier_spacing/1e3:.1f} kHz")
        self.logger.info(f"    • FFT size: {ofdm_config.get('fft_size', 512)}")
        self.logger.info(f"    • Guard carriers: {ofdm_config.get('num_guard_carriers', 52)}")

        self.logger.info("✅ PrismNetwork created successfully")
        self.logger.info(f"  📊 num_subcarriers: {self.prism_network.num_subcarriers}")
        self.logger.info(f"  📱 num_ue_antennas: {self.prism_network.num_ue_antennas}")
        self.logger.info(f"  📡 num_bs_antennas: {self.prism_network.num_bs_antennas}")

        # Create PrismTrainingInterface with the cleaned configuration structure
        checkpoint_dir = self.config['output']['training']['checkpoint_dir']
        self.logger.info(f"🔍 TRAIN_PRISM DEBUG: About to create TrainingInterface")
        self.logger.info(f"🔍 TRAIN_PRISM DEBUG: checkpoint_dir = {repr(checkpoint_dir)}")
        self.logger.info(f"🔍 TRAIN_PRISM DEBUG: checkpoint_dir type = {type(checkpoint_dir)}")
        print(f"🔍 TRAIN_PRISM DEBUG: checkpoint_dir = {repr(checkpoint_dir)}")
        
        self.model = PrismTrainingInterface(
            prism_network=self.prism_network,
            ray_tracing_config=ray_tracing_config,
            system_config=system_config,
            user_equipment_config=user_equipment_config,
            checkpoint_dir=checkpoint_dir
        )
        
        self.logger.info(f"🔍 TRAIN_PRISM DEBUG: TrainingInterface created")
        self.logger.info(f"🔍 TRAIN_PRISM DEBUG: self.model.checkpoint_dir = {repr(self.model.checkpoint_dir)}")
        print(f"🔍 TRAIN_PRISM DEBUG: self.model.checkpoint_dir = {repr(self.model.checkpoint_dir)}")
        
        # This section is redundant - the PrismNetwork was already created above
        # Just log the final configuration
        self.logger.info(f"PrismNetwork configuration completed:")
        
        self.logger.info(f"PrismNetwork created successfully")
        self.logger.info(f"  num_subcarriers: {self.prism_network.num_subcarriers}")
        self.logger.info(f"  num_ue_antennas: {self.prism_network.num_ue_antennas}")
        self.logger.info(f"  num_bs_antennas: {self.prism_network.num_bs_antennas}")
        
        # Create Ray Tracer with PrismNetwork for MLP-based direction selection
        # Note: PrismTrainingInterface will handle all ray tracing configuration internally
        
        # Note: PrismTrainingInterface will create its own ray tracer based on configuration
        self.logger.info("📡 Ray tracer will be created by PrismTrainingInterface based on configuration")
        
        # Note: PrismTrainingInterface will create its own ray tracer based on configuration
        self.logger.info("📡 Ray tracer will be created by PrismTrainingInterface based on configuration")
        
        # Get configuration sections for new structure
        ray_tracing_config = self.config.get('ray_tracing', {})
        system_config = self.config.get('system', {})
        
        # Get ray tracing mode from system config (new structure)
        ray_tracing_mode = system_config.get('ray_tracing_mode', 'hybrid')
        
        # Set PyTorch's current CUDA device to match training device
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            self.logger.info(f"🔧 Set PyTorch CUDA device to {self.device}")
        
        # Get checkpoint directory from config
        output_config = self.config.get('output', {})
        training_config = output_config.get('training', {})
        checkpoint_dir = training_config.get('checkpoint_dir')
        
        # Create PrismTrainingInterface with new simplified parameters
        self.model = PrismTrainingInterface(
            prism_network=self.prism_network,
            ray_tracing_config=ray_tracing_config,
            system_config=system_config,
            checkpoint_dir=checkpoint_dir,
            device=self.device
        )

        # Log configuration details
        subcarrier_sampling = ray_tracing_config.get('subcarrier_sampling', {})
        subcarrier_ratio = subcarrier_sampling.get('sampling_ratio', 0.1)
        total_subcarriers = self.prism_network.num_subcarriers
        selected_subcarriers = int(total_subcarriers * subcarrier_ratio)
        
        self.logger.info(f"Training interface created with ray_tracing_mode: {ray_tracing_mode}")
        self.logger.info(f"Ray tracing config: {ray_tracing_config.keys()}")
        self.logger.info(f"System config: {system_config.keys()}")
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
        # Training hyperparameters - batch_size will be calculated after data loading
        self.batch_size = None  # Will be set in _calculate_optimal_batch_size()
        
        # Multi-GPU setup (batch_size scaling will be done after calculation)
        # Scale batch size for multi-GPU training will be handled in _calculate_optimal_batch_size()
        
        self.learning_rate = float(self.config['training'].get('learning_rate', 1e-4))  # Read from config and convert to float
        self.num_epochs = self.config['training']['num_epochs']  # Read from config
        self.save_interval = self.config['training'].get('epoch_save_interval', 10)  # Read from config
        
        # Batch-level checkpoint settings
        self.batch_save_interval = self.config['training'].get('checkpoint_frequency', 10)  # Read from config
        self.enable_batch_checkpoints = self.config['training'].get('auto_checkpoint', True)  # Read from config
        
        # Debug logging for checkpoint configuration
        self.logger.info(f"Checkpoint configuration:")
        self.logger.info(f"  Batch checkpoints:")
        self.logger.info(f"    enable_batch_checkpoints: {self.enable_batch_checkpoints}")
        self.logger.info(f"    batch_save_interval: {self.batch_save_interval}")
        self.logger.info(f"    checkpoint_frequency from config: {self.config['training'].get('checkpoint_frequency', 'NOT SET')}")
        self.logger.info(f"    auto_checkpoint from config: {self.config['training'].get('auto_checkpoint', 'NOT SET')}")
        self.logger.info(f"  Epoch checkpoints:")
        self.logger.info(f"    epoch_save_interval: {self.save_interval}")
        self.logger.info(f"    epoch_save_interval from config: {self.config['training'].get('epoch_save_interval', 'NOT SET')}")
        
        # Deadlock detection settings
        self.batch_timeout = 600  # 10 minutes per batch
        self.progress_check_interval = 30  # Check progress every 30 seconds
        
        # Import and setup loss function
        from prism.loss_functions import PrismLossFunction, DEFAULT_LOSS_CONFIG
        
        # Create loss configuration
        loss_config = DEFAULT_LOSS_CONFIG.copy()
        # Override with any config-specific settings if needed
        training_config = self.config.get('training', {})
        if 'loss' in training_config:
            loss_config.update(training_config['loss'])
        
        # Initialize loss function
        self.criterion = PrismLossFunction(loss_config)
        
        self.logger.info(f"🎯 Initialized Hybrid CSI+PDP Loss Function:")
        self.logger.info(f"   CSI Weight: {loss_config['csi_weight']}")
        self.logger.info(f"   PDP Weight: {loss_config['pdp_weight']}")
        self.logger.info(f"   CSI Type: {loss_config['csi_loss']['type']}")
        self.logger.info(f"   PDP Type: {loss_config['pdp_loss']['type']}")
        
        # Setup mixed precision training
        mixed_precision_config = self.config.get('system', {}).get('mixed_precision', {})
        self.use_mixed_precision = mixed_precision_config.get('enabled', True)
        if self.use_mixed_precision and self.device.type == 'cuda':
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda', enabled=mixed_precision_config.get('grad_scaler_enabled', True))
            self.logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            if self.use_mixed_precision:
                self.logger.info("Mixed precision requested but not available (CPU mode)")
            else:
                self.logger.info("Mixed precision training disabled")
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and scheduler after model is created"""
        # Optimizer - now optimizing the TrainingInterface
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=float(self.config['training'].get('optimizer_params', {}).get('weight_decay', 1e-4))
        )
        
        # Learning rate scheduler
        lr_scheduler_config = self.config.get('training', {}).get('lr_scheduler', {})
        if lr_scheduler_config.get('enabled', True):
            scheduler_type = lr_scheduler_config.get('type', 'step')
            if scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=lr_scheduler_config.get('step_size', 30),
                    gamma=lr_scheduler_config.get('gamma', 0.1)
                )
                self.logger.info(f"StepLR scheduler enabled: step_size={lr_scheduler_config.get('step_size', 30)}, gamma={lr_scheduler_config.get('gamma', 0.1)}")
            elif scheduler_type == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=lr_scheduler_config.get('gamma', 0.5),
                    patience=lr_scheduler_config.get('patience', 5),
                    verbose=True
                )
                self.logger.info(f"ReduceLROnPlateau scheduler enabled: factor={lr_scheduler_config.get('gamma', 0.5)}, patience={lr_scheduler_config.get('patience', 5)}")
            else:
                self.scheduler = None
                self.logger.warning(f"Unknown scheduler type: {scheduler_type}, disabling scheduler")
        else:
            self.scheduler = None
            self.logger.info("Learning rate scheduler disabled")
        
        # Early stopping configuration
        early_stopping_config = self.config.get('training', {}).get('early_stopping', {})
        self.early_stopping_enabled = early_stopping_config.get('enabled', True)
        if self.early_stopping_enabled:
            self.early_stopping_patience = int(early_stopping_config.get('patience', 10))
            self.early_stopping_min_delta = float(early_stopping_config.get('min_delta', 1e-6))
            self.early_stopping_restore_best = early_stopping_config.get('restore_best_weights', True)
            self.early_stopping_counter = 0
            self.best_val_loss = float('inf')
            self.logger.info(f"Early stopping enabled: patience={self.early_stopping_patience}, min_delta={self.early_stopping_min_delta}")
        else:
            self.logger.info("Early stopping disabled")
    
    def _calculate_optimal_batch_size(self, total_samples: int) -> int:
        """
        Calculate optimal batch size based on total samples and batches per epoch
        
        Args:
            total_samples: Total number of training samples
            
        Returns:
            Optimal batch size
        """
        batches_per_epoch = self.config['training'].get('batches_per_epoch', 10)
        
        # Calculate base batch size
        base_batch_size = max(1, total_samples // batches_per_epoch)
        
        # Set reasonable limits for small datasets (≤10k samples)
        min_batch_size = 1
        max_batch_size = min(32, total_samples)  # Don't exceed 32 or total samples
        
        # Ensure batch size is within reasonable bounds
        optimal_batch_size = max(min_batch_size, min(base_batch_size, max_batch_size))
        
        # Scale for multi-GPU training
        if self.use_multi_gpu and self.num_gpus > 1:
            original_batch_size = optimal_batch_size
            optimal_batch_size = optimal_batch_size * self.num_gpus
            self.logger.info(f"Multi-GPU batch size scaling: {original_batch_size} × {self.num_gpus} = {optimal_batch_size}")
        
        self.logger.info(f"📊 Batch size calculation:")
        self.logger.info(f"   - Total samples: {total_samples}")
        self.logger.info(f"   - Batches per epoch: {batches_per_epoch}")
        self.logger.info(f"   - Calculated batch size: {optimal_batch_size}")
        self.logger.info(f"   - Expected samples per epoch: {optimal_batch_size * batches_per_epoch}")
        
        return optimal_batch_size
    
    def _select_best_gpu(self) -> int:
        """智能选择最佳可用GPU"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            return 0
        
        self.logger.info("🔍 Scanning available GPUs...")
        
        gpu_info = []
        for i in range(num_gpus):
            try:
                # 获取GPU基本信息
                props = torch.cuda.get_device_properties(i)
                name = torch.cuda.get_device_name(i)
                total_memory = props.total_memory / 1024**3  # GB
                
                # 获取当前内存使用情况
                torch.cuda.set_device(i)
                allocated_memory = torch.cuda.memory_allocated(i) / 1024**3  # GB
                reserved_memory = torch.cuda.memory_reserved(i) / 1024**3   # GB
                free_memory = total_memory - reserved_memory
                
                # 计算GPU利用率分数 (内存越多，使用率越低越好)
                memory_usage_ratio = reserved_memory / total_memory
                score = total_memory * (1 - memory_usage_ratio)  # 优先选择内存大且使用率低的GPU
                
                gpu_info.append({
                    'id': i,
                    'name': name,
                    'total_memory': total_memory,
                    'allocated_memory': allocated_memory,
                    'reserved_memory': reserved_memory,
                    'free_memory': free_memory,
                    'usage_ratio': memory_usage_ratio,
                    'score': score
                })
                
                self.logger.info(f"  GPU {i}: {name}")
                self.logger.info(f"    • Total Memory: {total_memory:.1f}GB")
                self.logger.info(f"    • Free Memory: {free_memory:.1f}GB")
                self.logger.info(f"    • Usage: {memory_usage_ratio*100:.1f}%")
                self.logger.info(f"    • Score: {score:.1f}")
                
            except Exception as e:
                self.logger.warning(f"  GPU {i}: Error getting info - {e}")
                gpu_info.append({
                    'id': i,
                    'score': -1  # 标记为不可用
                })
        
        # 选择得分最高的GPU
        best_gpu = max(gpu_info, key=lambda x: x['score'])
        selected_id = best_gpu['id']
        
        self.logger.info(f"✅ Selected GPU {selected_id} as the best option")
        return selected_id
    
    def _get_available_gpus(self) -> list:
        """获取所有可用GPU列表"""
        if not torch.cuda.is_available():
            return []
        
        available_gpus = []
        num_gpus = torch.cuda.device_count()
        
        for i in range(num_gpus):
            try:
                # 测试GPU是否可用
                torch.cuda.set_device(i)
                # 尝试分配一小块内存来测试GPU是否正常工作
                test_tensor = torch.zeros(1, device=f'cuda:{i}')
                del test_tensor
                available_gpus.append(i)
            except Exception as e:
                self.logger.warning(f"GPU {i} not available: {e}")
        
        return available_gpus
        
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
        
        # # Check if we're taking too long on a batch
        if elapsed_time > self.batch_timeout:
            self.logger.warning(f"⚠️  Batch {batch_idx} taking too long ({elapsed_time:.1f}s > {self.batch_timeout}s)")
            # return False
        
        # Check if we're making reasonable progress
        if batch_idx > 0:
            avg_time_per_batch = elapsed_time / batch_idx
            estimated_total_time = avg_time_per_batch * total_batches
            if estimated_total_time > 300:  # More than 1 hour
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
        
        # Use split-based data loading
        from prism.data_utils import load_and_split_data
        
        self.ue_positions, self.csi_data, self.bs_position, self.antenna_indices, metadata = load_and_split_data(
            dataset_path=self.data_path,
            train_ratio=self.split_config['train_ratio'],
            test_ratio=self.split_config['test_ratio'],
            random_seed=self.split_config['random_seed'],
            mode='train'
        )
        
        # Log split information
        print(f"   📊 Using train/test split mode")
        print(f"   🎲 Random seed: {self.split_config['random_seed']}")
        print(f"   📈 Train ratio: {self.split_config['train_ratio']}")
        print(f"   📉 Test ratio: {self.split_config['test_ratio']}")
        print(f"   📍 UE positions: {self.ue_positions.shape[0]} samples (training split)")
        print(f"   📡 CSI data: {self.csi_data.shape}")
        print(f"   🏢 BS position: {self.bs_position.shape}")
        print(f"   📡 Antenna indices: {len(self.antenna_indices)}")
        
        # Store metadata
        self.split_metadata = metadata
        
        # Validate data shapes
        self.logger.info(f"Data validation:")
        self.logger.info(f"  UE positions: {self.ue_positions.shape} - {self.ue_positions.dtype}")
        self.logger.info(f"  CSI data: {self.csi_data.shape} - {self.csi_data.dtype}")
        self.logger.info(f"  BS position: {self.bs_position.shape} - {self.bs_position.dtype}")
        self.logger.info(f"  Antenna indices: {self.antenna_indices.shape} - {self.antenna_indices.dtype}")
        
        # Calculate optimal batch size based on total samples
        total_samples = len(self.ue_positions)
        self.batch_size = self._calculate_optimal_batch_size(total_samples)
        
        # Check for data consistency
        if self.csi_data.shape[0] != self.ue_positions.shape[0]:
            raise ValueError(f"Data mismatch: {self.csi_data.shape[0]} CSI samples vs {self.ue_positions.shape[0]} UE positions")
        if self.csi_data.shape[3] != self.antenna_indices.shape[0]:
            raise ValueError(f"Data mismatch: {self.csi_data.shape[3]} BS antennas vs {self.antenna_indices.shape[0]} antenna indices")
        
        # Create dataset with all required data
        self.dataset = TensorDataset(
            self.ue_positions, 
            self.bs_position.expand(len(self.ue_positions), -1),
            self.antenna_indices.expand(len(self.ue_positions), -1),
            self.csi_data
        )
        
        # Validate batch size
        if self.batch_size > len(self.dataset):
            self.logger.warning(f"Batch size ({self.batch_size}) is larger than dataset size ({len(self.dataset)}). Adjusting batch size.")
            self.batch_size = len(self.dataset)
            self.logger.info(f"Adjusted batch size to: {self.batch_size}")
        
        # Get CPU configuration - use safer DataLoader settings to avoid "Broken pipe" errors
        cpu_config = self.config.get('system', {}).get('cpu', {})
        
        # Use num_workers=0 for CUDA to avoid multiprocessing issues that cause "Broken pipe"
        if self.device.type == 'cuda':
            num_workers = 0  # Single-threaded for CUDA to avoid process communication issues
            pin_memory = True
            persistent_workers = False
        else:
            num_workers = min(cpu_config.get('num_workers', 2), 2)  # Limit workers for stability
            pin_memory = False
            persistent_workers = False
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,  # Don't drop incomplete batches
            timeout=30 if num_workers > 0 else 0,  # Add timeout for worker processes
            multiprocessing_context='spawn' if num_workers > 0 else None  # Use spawn for better stability
        )
        self.logger.info(f"DataLoader created with {num_workers} workers (device: {self.device.type})")
        
        self.logger.info(f"Training data loaded: {len(self.dataset)} samples, batch_size={self.batch_size}")
        print(f"✅ Data loaded successfully!")
        print(f"   📊 Dataset size: {len(self.dataset)} samples")
        print(f"   🔄 Batch size: {self.batch_size}")
        print(f"   📦 Number of batches: {len(self.dataloader)}")
        print(f"   💾 Data types: UE (float32), CSI (complex64), BS (float32), Antenna (long)")
        
        # Get batches per epoch from config or use dataloader length
        self.batches_per_epoch = self.config['training'].get('batches_per_epoch', len(self.dataloader))
        
        # Initialize progress monitor AFTER dataloader is created
        self.progress_monitor = TrainingProgressMonitor(
            total_epochs=self.config['training']['num_epochs'],
            total_batches_per_epoch=self.batches_per_epoch
        )
        print(f"   📊 Progress monitor initialized for {self.batches_per_epoch} batches per epoch")
        print(f"   🔧 Configured batches per epoch: {self.batches_per_epoch} (dataloader has {len(self.dataloader)} batches)")
        
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
            # Create an iterator that can cycle through the dataloader if needed
            dataloader_iter = iter(self.dataloader)
            
            # Determine starting batch index for this epoch
            if epoch == self._current_start_epoch and hasattr(self, '_current_start_batch') and self._current_start_batch > 0:
                # If resuming in the middle of an epoch, start from the correct batch
                batch_start = self._current_start_batch
                self.logger.info(f"Resuming epoch {epoch} from batch {batch_start + 1}/{self.batches_per_epoch}")
                # Skip the batches that were already processed
                for _ in range(batch_start):
                    try:
                        next(dataloader_iter)
                    except StopIteration:
                        dataloader_iter = iter(self.dataloader)
                        next(dataloader_iter)
            else:
                # Normal case: start from first batch
                batch_start = 0
            
            for batch_idx in range(batch_start, self.batches_per_epoch):
                # Log batch start with clear progress information
                print(f"\n📦 BATCH {batch_idx + 1}/{self.batches_per_epoch} (Epoch {epoch})")
                print(f"{'='*60}")
                self.logger.info(f"Starting batch {batch_idx + 1}/{self.batches_per_epoch} in epoch {epoch}")
                
                try:
                    # Get next batch, cycling through dataloader if necessary
                    ue_pos, bs_pos, antenna_idx, csi_target = next(dataloader_iter)
                except StopIteration:
                    # If we've exhausted the dataloader, create a new iterator
                    dataloader_iter = iter(self.dataloader)
                    ue_pos, bs_pos, antenna_idx, csi_target = next(dataloader_iter)
                # Start batch monitoring if progress monitor is available
                if hasattr(self, 'progress_monitor') and self.progress_monitor is not None:
                    self.progress_monitor.start_batch(batch_idx)
                
                self.logger.debug(f"Processing training batch {batch_idx}:")
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
                    print(f"    🧠 Forward pass for training batch {batch_idx+1}...")
                    
                    # Use mixed precision if enabled
                    if self.use_mixed_precision and self.scaler is not None:
                        with torch.amp.autocast('cuda', enabled=self.config.get('system', {}).get('mixed_precision', {}).get('autocast_enabled', True)):
                            outputs = self.model(
                                ue_positions=ue_pos,
                                bs_position=bs_pos,
                                antenna_indices=antenna_idx
                            )
                    else:
                        # Regular forward pass
                        outputs = self.model(
                            ue_positions=ue_pos,
                            bs_position=bs_pos,
                            antenna_indices=antenna_idx
                        )
                    
                    # Extract CSI predictions and compute loss (common logic)
                    csi_pred = outputs['csi_predictions']
                    print(f"    🎯 Computing loss for batch {batch_idx+1}...")
                    loss = self.model.compute_loss(csi_pred, csi_target, self.criterion)
                    
                    print(f"    ✅ Forward pass completed for batch {batch_idx+1}")
                    
                    # Memory cleanup after forward pass
                    if hasattr(outputs, 'keys'):
                        for key in list(outputs.keys()):
                            if key not in ['csi_predictions']:  # Keep only essential outputs
                                del outputs[key]
                    torch.cuda.empty_cache()
                    
                    # Validate loss is a tensor
                    if not isinstance(loss, torch.Tensor):
                        self.logger.error(f"Loss computation returned non-tensor: {type(loss)} = {loss}")
                        raise ValueError(f"Loss must be a torch.Tensor, got {type(loss)}")
                    
                    # Ensure loss has requires_grad for backward pass
                    if not loss.requires_grad:
                        self.logger.error("❌ CRITICAL: Loss tensor does not require gradients!")
                        raise RuntimeError("Loss tensor must require gradients for proper training.")
                    
                    # Check for NaN or infinite values
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.error(f"Invalid loss value: {loss}")
                        raise ValueError(f"Loss contains NaN or infinite values: {loss}")
                    
                except Exception as e:
                    self.logger.error(f"Loss computation failed: {e}")
                    if 'csi_pred' in locals():
                        self.logger.error(f"Shapes - csi_pred: {csi_pred.shape}, csi_target: {csi_target.shape}")
                    else:
                        self.logger.error(f"csi_pred not available, csi_target: {csi_target.shape}")
                    raise
                
                # Backward pass
                print(f"    ⬅️  Backward pass for batch {batch_idx+1}...")
                if self.use_mixed_precision and self.scaler is not None:
                    # Mixed precision backward pass
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping with scaler
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    print(f"    📈 Optimizer step for batch {batch_idx+1}...")
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    print(f"    📈 Optimizer step for batch {batch_idx+1}...")
                    self.optimizer.step()
                
                # Store loss value before cleanup
                batch_loss = loss.item()
                total_loss += batch_loss
                num_batches += 1
                
                print(f"    ✅ Batch {batch_idx+1} completed! Loss: {batch_loss:.6f}")
                
                # Memory cleanup after backward pass
                del loss, csi_pred, csi_target, outputs
                torch.cuda.empty_cache()
                
                # Update progress monitoring
                if hasattr(self, 'progress_monitor') and self.progress_monitor is not None:
                    self.progress_monitor.update_batch_progress(batch_idx, batch_loss, self.batches_per_epoch)
                
                # Update training state in TrainingInterface
                self.model.update_training_state(epoch, batch_idx, batch_loss)
                
                # Save batch checkpoint if enabled and at the right interval
                checkpoint_condition = (self.enable_batch_checkpoints and 
                                      self.batch_save_interval > 0 and 
                                      (batch_idx + 1) % self.batch_save_interval == 0)
                
                # Debug logging for checkpoint decision (ENHANCED DEBUG)
                # ALWAYS show debug info to debug checkpoint issue
                self.logger.debug(f"🔍 CHECKPOINT DEBUG - Batch {batch_idx}:")
                self.logger.debug(f"  📊 batch_idx: {batch_idx}")
                self.logger.debug(f"  📊 batch_idx + 1: {batch_idx + 1}")
                self.logger.debug(f"  📊 batch_save_interval: {self.batch_save_interval}")
                self.logger.debug(f"  📊 (batch_idx + 1) % batch_save_interval: {(batch_idx + 1) % self.batch_save_interval}")
                self.logger.debug(f"  📊 enable_batch_checkpoints: {self.enable_batch_checkpoints}")
                self.logger.debug(f"  📊 checkpoint_condition: {checkpoint_condition}")
                print(f"🔍 CHECKPOINT DEBUG - Batch {batch_idx}: condition = {checkpoint_condition}")
                
                if checkpoint_condition:
                    self.logger.info(f"🎯 CHECKPOINT TRIGGERED! Saving checkpoint for batch {batch_idx + 1}")
                    print(f"🎯 CHECKPOINT TRIGGERED! Saving checkpoint for batch {batch_idx + 1}")
                    try:
                        self._save_batch_checkpoint(epoch, batch_idx + 1, batch_loss)
                        self.logger.info(f"✅ Checkpoint save completed for batch {batch_idx + 1}")
                        print(f"✅ Checkpoint save completed for batch {batch_idx + 1}")
                    except Exception as e:
                        self.logger.error(f"❌ Checkpoint save FAILED for batch {batch_idx + 1}: {e}")
                        print(f"❌ Checkpoint save FAILED for batch {batch_idx + 1}: {e}")
                        import traceback
                        self.logger.error(f"Full traceback: {traceback.format_exc()}")
                else:
                    self.logger.info(f"⏭️  No checkpoint for batch {batch_idx + 1} (condition not met)")
                    print(f"⏭️  No checkpoint for batch {batch_idx + 1} (condition not met)")
                
                # Log batch completion with epoch progress bar
                batch_progress = (batch_idx + 1) / self.batches_per_epoch * 100
                avg_loss_so_far = total_loss / num_batches
                
                # Create epoch progress bar
                progress_bar_length = 30
                filled_length = int(progress_bar_length * batch_progress / 100)
                progress_bar = '█' * filled_length + '░' * (progress_bar_length - filled_length)
                
                print(f"✅ BATCH {batch_idx + 1}/{self.batches_per_epoch} COMPLETED")
                print(f"   📊 Epoch Progress: [{progress_bar}] {batch_progress:5.1f}%")
                print(f"   📈 Loss: {batch_loss:.6f} | Avg Loss: {avg_loss_so_far:.6f}")
                self.logger.info(f"Batch {batch_idx + 1}/{self.batches_per_epoch} completed with loss: {batch_loss:.6f}")
                
                # Update progress monitor with real-time information if available
                if hasattr(self, 'progress_monitor') and self.progress_monitor is not None:
                    self.progress_monitor.update_batch_progress(batch_idx, batch_loss, self.batches_per_epoch)
                else:
                    # Fallback progress logging
                    if batch_idx % 5 == 0 or batch_idx == self.batches_per_epoch - 1:
                        progress = (batch_idx + 1) / self.batches_per_epoch * 100
                        avg_loss_so_far = total_loss / num_batches
                        print(f"  📊 Batch {batch_idx+1:3d}/{self.batches_per_epoch:3d} ({progress:5.1f}%) | "
                              f"Loss: {batch_loss:.6f} | Avg: {avg_loss_so_far:.6f}")
                
                # Check for potential deadlocks or slow progress
                if not self._check_training_progress(epoch_start_time, batch_idx, self.batches_per_epoch):
                    print(f"  ⚠️  Training progress check failed - consider restarting if stuck")
                
                # Log to tensorboard
                self.writer.add_scalar(f'Loss/Batch_{epoch}', batch_loss, batch_idx)
                    
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
                    num_workers=0,  # Use no workers for recovery to avoid "Broken pipe"
                    pin_memory=True if self.device.type == 'cuda' else False,
                    persistent_workers=False,  # Disable persistent workers
                    drop_last=False,
                    timeout=0,  # No timeout for single-threaded
                    multiprocessing_context=None  # No multiprocessing
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
        # Convert tensor to float for JSON serialization
        avg_loss_float = float(avg_loss.item()) if isinstance(avg_loss, torch.Tensor) else float(avg_loss)
        self.logger.info(f"Epoch {epoch} completed: {num_batches} batches, avg loss: {avg_loss_float:.6f}")
        
        # End epoch monitoring if progress monitor is available
        if hasattr(self, 'progress_monitor') and self.progress_monitor is not None:
            self.progress_monitor.end_epoch(avg_loss_float)
        
        return avg_loss_float
    
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
                        loss = self.model.compute_loss(csi_pred, csi_target, self.criterion, validation_mode=True)
                        
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
                    
                    total_loss += loss
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
        
        # Save TrainingInterface checkpoint with optimizer and scheduler states
        checkpoint_name = f'checkpoint_epoch_{epoch}.pt'
        self.model.save_checkpoint(checkpoint_name, 
                                  optimizer_state_dict=self.optimizer.state_dict(),
                                  scheduler_state_dict=self.scheduler.state_dict())
        self.logger.info(f"💾 TrainingInterface checkpoint saved: {checkpoint_name}")
        
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
        
        # Save epoch training state in checkpoints directory for consistency
        checkpoint_dir = Path(self.model.checkpoint_dir)
        checkpoint_path = checkpoint_dir / f'training_state_epoch_{epoch}.pt'
        torch.save(training_state, checkpoint_path)
        self.logger.info(f"💾 Training state saved: {checkpoint_path}")
        self.logger.info(f"📊 Training state includes: epoch={epoch}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Save best model
        if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
            # Store the previous best loss for logging before updating
            previous_best = getattr(self, 'best_val_loss', float('inf'))
            self.best_val_loss = val_loss
            
            models_dir = Path(self.config['output']['training']['models_dir'])
            models_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = models_dir / 'best_model.pt'
            
            # Save the best model directly to the models directory (not through checkpoint_dir)
            self.logger.info(f"🏆 Saving new best model (Val Loss: {val_loss:.6f} < {previous_best:.6f})")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'prism_network_state_dict': self.model.prism_network.state_dict(),
                'current_epoch': self.model.current_epoch,
                'current_batch': self.model.current_batch,
                'best_loss': self.model.best_loss,
                'training_history': self.model.training_history,
                'current_selection': self.model.current_selection,
                'current_selection_mask': self.model.current_selection_mask,
                'training_config': {
                    'num_sampling_points': self.model.num_sampling_points,
                    'subcarrier_sampling_ratio': self.model.subcarrier_sampling_ratio,
                    'scene_bounds': (self.model.scene_min.tolist(), self.model.scene_max.tolist())
                }
            }, str(best_model_path))
            self.logger.info(f"💾 Best model weights saved: {best_model_path}")
            
            # Save best model training state
            best_model_state_path = str(best_model_path).replace('.pt', '_state.pt')
            torch.save(training_state, best_model_state_path)
            self.logger.info(f"💾 Best model training state saved: {best_model_state_path}")
            print(f"🏆 New best model saved! (Val Loss: {val_loss:.6f})")
        else:
            self.logger.info(f"📊 No new best model (Val Loss: {val_loss:.6f} >= {self.best_val_loss:.6f})")
        
        # Save latest checkpoint for resuming in checkpoint directory
        checkpoint_dir = Path(self.model.checkpoint_dir)
        latest_checkpoint_path = checkpoint_dir / 'latest_checkpoint.pt'
        self.model.save_checkpoint('latest_checkpoint.pt',
                                  optimizer_state_dict=self.optimizer.state_dict(),
                                  scheduler_state_dict=self.scheduler.state_dict())
        self.logger.info(f"💾 Latest checkpoint saved: {latest_checkpoint_path}")
        
        # Save latest checkpoint training state
        latest_state_path = str(latest_checkpoint_path).replace('.pt', '_state.pt')
        torch.save(training_state, latest_state_path)
        self.logger.info(f"💾 Latest checkpoint training state saved: {latest_state_path}")
        
        # Save emergency checkpoint every epoch for better recovery in checkpoint directory
        emergency_checkpoint_path = checkpoint_dir / 'emergency_checkpoint.pt'
        self.model.save_checkpoint('emergency_checkpoint.pt',
                                  optimizer_state_dict=self.optimizer.state_dict(),
                                  scheduler_state_dict=self.scheduler.state_dict())
        self.logger.info(f"🚨 Emergency checkpoint saved: {emergency_checkpoint_path}")
        
        # Save emergency checkpoint training state
        emergency_state_path = str(emergency_checkpoint_path).replace('.pt', '_state.pt')
        torch.save(training_state, emergency_state_path)
        self.logger.info(f"🚨 Emergency checkpoint training state saved: {emergency_state_path}")
        
        print(f"✅ Checkpoint saved: Epoch {epoch}, Loss: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        # Clean up old checkpoints (keep last 5)
        self._cleanup_old_checkpoints()
    
    def _save_batch_checkpoint(self, epoch: int, batch_idx: int, batch_loss: float):
        """Save checkpoint after completing a batch"""
        self.logger.info(f"🔧 _save_batch_checkpoint called: epoch={epoch}, batch_idx={batch_idx}, loss={batch_loss}")
        print(f"🔧 _save_batch_checkpoint called: epoch={epoch}, batch_idx={batch_idx}, loss={batch_loss}")
        try:
            if not self.enable_batch_checkpoints:
                self.logger.warning(f"❌ Batch checkpoints disabled! enable_batch_checkpoints={self.enable_batch_checkpoints}")
                print(f"❌ Batch checkpoints disabled! enable_batch_checkpoints={self.enable_batch_checkpoints}")
                return
                
            # Get checkpoint directory from model (configured directory)
            checkpoint_dir = Path(self.model.checkpoint_dir)
            
            # Save TrainingInterface batch checkpoint with optimizer and scheduler states
            checkpoint_name = f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pt'
            self.model.save_checkpoint(checkpoint_name,
                                      optimizer_state_dict=self.optimizer.state_dict(),
                                      scheduler_state_dict=self.scheduler.state_dict())
            self.logger.info(f"💾 Batch TrainingInterface checkpoint saved: {checkpoint_name}")
            
            # Save batch training state to the same checkpoint directory
            batch_state = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'batch_loss': batch_loss,
                'learning_rate': self.learning_rate,
                'timestamp': datetime.now().isoformat(),
                'batch_checkpoint': True
            }
            
            batch_state_path = checkpoint_dir / f'training_state_epoch_{epoch}_batch_{batch_idx}.pt'
            torch.save(batch_state, batch_state_path)
            self.logger.info(f"💾 Batch training state saved: {batch_state_path}")
            
            # Update latest batch checkpoint in checkpoint directory
            latest_batch_path = checkpoint_dir / 'latest_batch_checkpoint.pt'
            self.model.save_checkpoint('latest_batch_checkpoint.pt',
                                      optimizer_state_dict=self.optimizer.state_dict(),
                                      scheduler_state_dict=self.scheduler.state_dict())
            self.logger.info(f"💾 Latest batch checkpoint updated: {latest_batch_path}")
            
            latest_batch_state_path = str(latest_batch_path).replace('.pt', '_state.pt')
            torch.save(batch_state, latest_batch_state_path)
            self.logger.info(f"💾 Latest batch checkpoint state updated: {latest_batch_state_path}")
            
            self.logger.info(f"Batch checkpoint saved: Epoch {epoch}, Batch {batch_idx}, Loss: {batch_loss:.6f}")
            print(f"💾 Batch checkpoint saved: E{epoch}B{batch_idx} (Loss: {batch_loss:.6f})")
            
        except Exception as e:
            self.logger.warning(f"Batch checkpoint failed: {e}")
    
    def _save_emergency_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        """Save emergency checkpoint for quick recovery"""
        try:
            # Save TrainingInterface emergency checkpoint
            self.model.save_checkpoint('emergency_checkpoint.pt',
                                      optimizer_state_dict=self.optimizer.state_dict(),
                                      scheduler_state_dict=self.scheduler.state_dict())
            self.logger.info(f"🚨 Emergency TrainingInterface checkpoint saved")
            
            # Save minimal training state for emergency recovery
            emergency_state = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat(),
                'emergency': True
            }
            
            checkpoint_dir = Path(self.model.checkpoint_dir)
            emergency_path = checkpoint_dir / 'emergency_checkpoint_state.pt'
            torch.save(emergency_state, emergency_path)
            self.logger.info(f"🚨 Emergency checkpoint state saved: {emergency_path}")
            
        except Exception as e:
            self.logger.warning(f"Emergency checkpoint failed: {e}")
    
    def _save_best_model(self, epoch: int, train_loss: float, val_loss: float):
        """Save the best model checkpoint to the correct models directory"""
        try:
            # Get the models directory (not checkpoint directory) - this is the correct location
            models_dir = Path(self.config['output']['training']['models_dir'])
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Save best model to models directory (primary location)
            best_model_path = models_dir / 'best_model.pt'
            best_model_data = {
                'model_state_dict': self.model.state_dict(),
                'prism_network_state_dict': self.model.prism_network.state_dict(),
                'current_epoch': self.model.current_epoch,
                'current_batch': self.model.current_batch,
                'best_loss': self.model.best_loss,
                'training_history': self.model.training_history,
                'current_selection': self.model.current_selection,
                'current_selection_mask': self.model.current_selection_mask,
                'training_config': {
                    'num_sampling_points': self.model.num_sampling_points,
                    'subcarrier_sampling_ratio': self.model.subcarrier_sampling_ratio,
                    'scene_bounds': (self.model.scene_min.tolist(), self.model.scene_max.tolist())
                }
            }
            torch.save(best_model_data, best_model_path)
            self.logger.info(f"🏆 Best model saved to models directory: {best_model_path}")
            
            # Save best model training state (with optimizer/scheduler info)
            best_model_state = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config,
                'timestamp': datetime.now().isoformat(),
                'is_best_model': True
            }
            
            best_model_state_path = models_dir / 'best_model_state.pt'
            torch.save(best_model_state, best_model_state_path)
            self.logger.info(f"🏆 Best model training state saved: {best_model_state_path}")
            
            # Also save a copy to checkpoint directory for backup (optional)
            checkpoint_dir = Path(self.model.checkpoint_dir)
            backup_best_path = checkpoint_dir / 'best_model_backup.pt'
            torch.save(best_model_data, backup_best_path)
            self.logger.info(f"🏆 Best model backup saved to checkpoint directory: {backup_best_path}")
            
        except Exception as e:
            self.logger.warning(f"Best model save failed: {e}")
    
    def _auto_detect_checkpoint(self):
        """Automatically detect the best checkpoint to resume from"""
        print("🔍 Auto-detecting checkpoints...")
        
        # Priority order for checkpoint detection - use checkpoint directory
        checkpoint_dir = Path(self.config['output']['training']['checkpoint_dir'])
        checkpoint_candidates = [
            checkpoint_dir / 'latest_batch_checkpoint.pt',  # Most recent batch
            checkpoint_dir / 'emergency_checkpoint.pt',     # Emergency checkpoint
            checkpoint_dir / 'latest_checkpoint.pt',        # Latest epoch
            Path(self.config['output']['training']['models_dir']) / 'best_model.pt'  # Best performance
        ]
        
        for checkpoint_path in checkpoint_candidates:
            if checkpoint_path.exists():
                print(f"✅ Found checkpoint: {checkpoint_path}")
                return str(checkpoint_path)
        
        # Check for batch-specific checkpoints
        checkpoint_dir = Path(self.config['output']['training']['checkpoint_dir'])
        if checkpoint_dir.exists():
            # First check for batch checkpoints (most recent)
            batch_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*_batch_*.pt'))
            if batch_checkpoints:
                # Get the latest batch checkpoint
                latest_batch_checkpoint = max(batch_checkpoints, key=lambda x: (
                    int(x.stem.split('_')[2]),  # epoch number
                    int(x.stem.split('_')[4])   # batch number
                ))
                print(f"✅ Found batch checkpoint: {latest_batch_checkpoint}")
                return str(latest_batch_checkpoint)
            
            # Then check for epoch checkpoints
            epoch_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            epoch_only_checkpoints = [cp for cp in epoch_checkpoints if '_batch_' not in cp.name]
            if epoch_only_checkpoints:
                # Get the latest epoch checkpoint
                latest_epoch_checkpoint = max(epoch_only_checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
                print(f"✅ Found epoch checkpoint: {latest_epoch_checkpoint}")
                return str(latest_epoch_checkpoint)
        
        print("❌ No checkpoints found")
        return None
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to save disk space"""
        # Clean up TrainingInterface epoch checkpoints
        checkpoint_dir = Path(self.model.checkpoint_dir)
        epoch_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        # Filter out batch checkpoints from epoch cleanup
        epoch_only_checkpoints = [cp for cp in epoch_checkpoints if '_batch_' not in cp.name]
        if len(epoch_only_checkpoints) > 5:
            # Sort by epoch number and remove oldest
            epoch_only_checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for checkpoint in epoch_only_checkpoints[:-5]:
                checkpoint.unlink()
                self.logger.info(f"Removed old epoch checkpoint: {checkpoint}")
        
        # Clean up batch checkpoints (keep last 20 batch checkpoints)
        batch_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*_batch_*.pt'))
        if len(batch_checkpoints) > 20:
            # Sort by epoch and batch number
            batch_checkpoints.sort(key=lambda x: (
                int(x.stem.split('_')[2]),  # epoch number
                int(x.stem.split('_')[4])   # batch number
            ))
            for checkpoint in batch_checkpoints[:-20]:
                checkpoint.unlink()
                self.logger.info(f"Removed old batch checkpoint: {checkpoint}")
        
        # Clean up epoch training state files in checkpoint directory
        epoch_training_states = list(checkpoint_dir.glob('training_state_epoch_*.pt'))
        # Filter out batch training states
        epoch_only_states = [ts for ts in epoch_training_states if '_batch_' not in ts.name]
        if len(epoch_only_states) > 5:
            epoch_only_states.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for state in epoch_only_states[:-5]:
                state.unlink()
                self.logger.info(f"Removed old epoch training state: {state}")
        
        # Clean up batch training state files in checkpoint directory (keep last 20)
        batch_training_states = list(checkpoint_dir.glob('training_state_epoch_*_batch_*.pt'))
        if len(batch_training_states) > 20:
            batch_training_states.sort(key=lambda x: (
                int(x.stem.split('_')[3]),  # epoch number
                int(x.stem.split('_')[5])   # batch number
            ))
            for state in batch_training_states[:-20]:
                state.unlink()
                self.logger.info(f"Removed old batch training state: {state}")
    
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
            
            # Try to load optimizer and scheduler states from main checkpoint first
            try:
                checkpoint = torch.load(self.resume_from, map_location=self.device)
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.logger.info("Optimizer state loaded from main checkpoint")
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    self.logger.info("Scheduler state loaded from main checkpoint")
            except Exception as e:
                self.logger.warning(f"Could not load optimizer/scheduler from main checkpoint: {e}")
                
                # Fallback: Load training state from separate file if available
                training_state_path = self.resume_from.replace('.pt', '_state.pt')
                if os.path.exists(training_state_path):
                    training_state = torch.load(training_state_path, map_location=self.device)
                    
                    # Load optimizer state
                    if 'optimizer_state_dict' in training_state:
                        self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
                        self.logger.info("Optimizer state loaded from separate state file")
                    
                    # Load scheduler state
                    if 'scheduler_state_dict' in training_state:
                        self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
                        self.logger.info("Scheduler state loaded from separate state file")
                
                # Load training state
                # Check if we're in the middle of an epoch or if the epoch is complete
                current_batch = training_state.get('batch', 0)
                current_epoch = training_state['epoch']
                batches_per_epoch = self.config['training'].get('batches_per_epoch', 10)
                
                if current_batch + 1 < batches_per_epoch:
                    # Still in the middle of an epoch
                    self.start_epoch = current_epoch  # Continue current epoch
                    self.start_batch = current_batch + 1  # Start from next batch
                else:
                    # Epoch is complete, start next epoch
                    self.start_epoch = current_epoch + 1  # Start next epoch
                    self.start_batch = 0  # Start from first batch of next epoch
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
                # Check if we're in the middle of an epoch or if the epoch is complete
                current_batch = getattr(self.model, 'current_batch', 0)
                current_epoch = self.model.current_epoch
                batches_per_epoch = self.config['training'].get('batches_per_epoch', 10)
                
                if current_batch + 1 < batches_per_epoch:
                    # Still in the middle of an epoch
                    self.start_epoch = current_epoch  # Continue current epoch
                    self.start_batch = current_batch + 1  # Start from next batch
                else:
                    # Epoch is complete, start next epoch
                    self.start_epoch = current_epoch + 1  # Start next epoch
                    self.start_batch = 0  # Start from first batch of next epoch
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
        self._load_data()
        print(f"✅ Data loaded: {len(self.dataset)} samples")
        
        # Initialize training state
        if hasattr(self, 'start_epoch'):
            start_epoch = self.start_epoch
            start_batch = getattr(self, 'start_batch', 0)
            train_losses = self.train_losses
            val_losses = self.val_losses
            if start_batch > 0:
                self.logger.info(f"Resuming training from epoch {start_epoch}, batch {start_batch + 1}")
                print(f"🔄 Resuming training from epoch {start_epoch}, batch {start_batch + 1}")
            else:
                self.logger.info(f"Resuming training from epoch {start_epoch}")
                print(f"🔄 Resuming training from epoch {start_epoch}")
        else:
            start_epoch = 1
            start_batch = 0
            train_losses = []
            val_losses = []
            self.logger.info("Starting training from epoch 1")
            print("🆕 Starting training from epoch 1")
        
        # Store as instance variables for use in _train_epoch
        self._current_start_epoch = start_epoch
        self._current_start_batch = start_batch
        
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
        
        # Adjust num_epochs if resuming from a later epoch
        if start_epoch > self.num_epochs:
            self.logger.warning(f"Resuming from epoch {start_epoch} but config only has {self.num_epochs} epochs")
            self.logger.warning(f"Extending training to complete at least {start_epoch} epochs")
            actual_num_epochs = max(self.num_epochs, start_epoch)
        else:
            actual_num_epochs = self.num_epochs
            
        for epoch in range(start_epoch, actual_num_epochs + 1):
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
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
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
            
            # Save checkpoint - ensure we always save when epoch_save_interval is 1
            should_save_checkpoint = (epoch % self.save_interval == 0 or 
                                    epoch == self.num_epochs or 
                                    self.save_interval == 1)  # Always save if interval is 1
            
            if should_save_checkpoint:
                print(f"💾 Saving checkpoint for epoch {epoch}...")
                self.logger.info(f"💾 Saving epoch checkpoint: epoch={epoch}, save_interval={self.save_interval}")
                self._save_checkpoint(epoch, train_loss, val_loss)
                print(f"✅ Checkpoint saved successfully")
            else:
                print(f"⏭️  Skipping checkpoint save for epoch {epoch} (save_interval={self.save_interval})")
                self.logger.info(f"⏭️  Skipping epoch checkpoint: epoch={epoch}, save_interval={self.save_interval}")
            
            # Save emergency checkpoint every epoch for better recovery
            if epoch % 1 == 0:  # Every epoch
                self._save_emergency_checkpoint(epoch, train_loss, val_loss)
            
            # Always save best model if this is the best validation loss so far
            if not hasattr(self, 'best_val_loss_ever'):
                self.best_val_loss_ever = float('inf')
            
            if val_loss < self.best_val_loss_ever:
                self.best_val_loss_ever = val_loss
                print(f"🏆 New best validation loss: {val_loss:.6f} - Saving best model...")
                self.logger.info(f"🏆 New best validation loss: {val_loss:.6f} - Saving best model")
                self._save_best_model(epoch, train_loss, val_loss)
                print(f"✅ Best model saved successfully")
            
            # Early stopping check
            if self.early_stopping_enabled:
                if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    print(f"🏆 New best validation loss: {val_loss:.6f}")
                else:
                    self.early_stopping_counter += 1
                    print(f"⏳ Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                    
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"🛑 Early stopping triggered after {self.early_stopping_patience} epochs without improvement")
                    self.logger.info(f"Early stopping triggered: no improvement for {self.early_stopping_patience} epochs")
                    break
            
            # Learning rate stopping check
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
        
        if train_losses:
            print(f"   • Final training loss: {train_losses[-1]:.6f}")
            print(f"   • Final validation loss: {val_losses[-1]:.6f}")
            print(f"   • Best validation loss: {min(val_losses):.6f}")
        else:
            print(f"   • No training epochs completed (resumed training may have finished immediately)")
        
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
        print(f"   📊 TensorBoard logs: {self.config['output']['training']['tensorboard_dir']}/")
        print(f"   💾 Checkpoints: {self.config['output']['training']['checkpoint_dir']}/")
        print(f"   🏆 Best models: {self.config['output']['training']['models_dir']}/")
        print(f"   📈 Training plots: {self.output_dir}/")
        print(f"   📝 Training log: training.log")
        
        print("="*80)
        print("🎉 Training completed successfully!")
        print("="*80)
    
    def _display_ray_tracer_info(self):
        """Display information about the ray tracer configuration and performance."""
        print("\n🔍 Ray Tracer Configuration:")
        print("=" * 30)
        
        # Display ray tracer type and performance info from TrainingInterface
        if hasattr(self.model, 'ray_tracer') and self.model.ray_tracer is not None:
            ray_tracer = self.model.ray_tracer
            
            # Display parallelization stats if available
            if hasattr(ray_tracer, 'get_parallelization_stats'):
                try:
                    stats = ray_tracer.get_parallelization_stats()
                    print(f"  - Type: {'CUDA' if stats.get('cuda_enabled', False) else 'CPU'}")
                    print(f"  - Parallel processing: {stats.get('parallel_processing_enabled', 'N/A')}")
                    print(f"  - Processing mode: {stats.get('processing_mode', 'N/A')}")
                    print(f"  - Max workers: {stats.get('max_workers', 'N/A')}")
                    print(f"  - Total directions: {stats.get('total_directions', 'N/A')}")
                except Exception as e:
                    print(f"  - Error getting parallelization stats: {e}")
            
            # Display CUDA-specific information if available
            if hasattr(ray_tracer, 'get_performance_info'):
                try:
                    perf_info = ray_tracer.get_performance_info()
                    print(f"  - Device: {perf_info.get('device', 'N/A')}")
                    print(f"  - CUDA enabled: {perf_info.get('use_cuda', 'N/A')}")
                    if perf_info.get('use_cuda', False):
                        print(f"  - CUDA device: {perf_info.get('cuda_device_name', 'N/A')}")
                        print(f"  - CUDA memory: {perf_info.get('cuda_memory_gb', 'N/A')} GB")
                except Exception as e:
                    print(f"  - Error getting performance info: {e}")
            
            # Display ray count analysis if available
            if hasattr(ray_tracer, 'get_ray_count_analysis'):
                try:
                    # Use default values for analysis
                    num_bs = 1
                    num_ue = 100
                    num_subcarriers = 64
                    analysis = ray_tracer.get_ray_count_analysis(num_bs, num_ue, num_subcarriers)
                    print(f"  - Total rays (1 BS, 100 UE, 64 subcarriers): {analysis.get('total_rays', 'N/A'):,}")
                    print(f"  - Ray count formula: {analysis.get('ray_count_formula', 'N/A')}")
                except Exception as e:
                    print(f"  - Error getting ray count analysis: {e}")
        else:
            print(f"  - Ray tracer information not available (managed by TrainingInterface)")
        
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

        # Display emergency checkpoint in checkpoint directory
        checkpoint_dir = Path(self.model.checkpoint_dir)
        emergency_checkpoint_path = checkpoint_dir / 'emergency_checkpoint.pt'
        if emergency_checkpoint_path.exists():
            print(f"\nEmergency Checkpoint (in {checkpoint_dir}):")
            print(f"  - Path: {emergency_checkpoint_path}")
        else:
            print(f"\nEmergency Checkpoint (in {checkpoint_dir}):")
            print("  - Not found.")
    
    def _display_batch_checkpoint_info(self):
        """Display batch checkpoint configuration information"""
        print(f"\n💾 Batch Checkpoint Configuration:")
        if self.enable_batch_checkpoints:
            print(f"   ✅ Batch checkpoints: ENABLED")
            print(f"   🔄 Save interval: Every {self.batch_save_interval} batches")
            print(f"   🧹 Cleanup: Keep last 20 batch checkpoints")
            print(f"   📝 Format: checkpoint_epoch_X_batch_Y.pt")
            
            # Show existing batch checkpoints
            checkpoint_dir = Path(self.model.checkpoint_dir)
            if checkpoint_dir.exists():
                batch_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*_batch_*.pt'))
                if batch_checkpoints:
                    print(f"   📊 Existing batch checkpoints: {len(batch_checkpoints)}")
                    latest_batch = max(batch_checkpoints, key=lambda x: (
                        int(x.stem.split('_')[2]),  # epoch number
                        int(x.stem.split('_')[4])   # batch number
                    ))
                    print(f"   📄 Latest: {latest_batch.name}")
                else:
                    print(f"   📊 No batch checkpoints found yet")
        else:
            print(f"   ❌ Batch checkpoints: DISABLED")
        print(f"   💡 This allows recovery from mid-epoch failures")

        # Display latest checkpoint in checkpoint directory
        checkpoint_dir = Path(self.model.checkpoint_dir)
        latest_checkpoint_path = checkpoint_dir / 'latest_checkpoint.pt'
        if latest_checkpoint_path.exists():
            print(f"\nLatest Checkpoint (in {checkpoint_dir}):")
            print(f"  - Path: {latest_checkpoint_path}")
        else:
            print(f"\nLatest Checkpoint (in {checkpoint_dir}):")
            print("  - Not found.")

        # Display best model
        models_dir = Path(self.config['output']['training']['models_dir'])
        best_model_path = models_dir / 'best_model.pt'
        if best_model_path.exists():
            print(f"\nBest Model (in {models_dir}):")
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
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file (required)')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID to use (e.g., 0, 1, 2). If not specified, will auto-select based on available memory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    # Create trainer and run training (data path and output directory from config)
    trainer = PrismTrainer(args.config, None, None, args.resume, args.gpu)
    trainer.train()

if __name__ == '__main__':
    main()
