#!/usr/bin/env python3
"""
Prism Network Testing Script

This script tests the trained Prism neural network for electromagnetic ray tracing.
It loads a trained model and evaluates its performance on test data, including
metrics calculation, visualization, and comparison with ground truth.
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
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import time
from typing import Dict, List, Tuple, Optional

# GPU monitoring imports
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism.networks.prism_network import PrismNetwork
from prism.config_loader import ConfigLoader
from prism.ray_tracer_cpu import CPURayTracer
from prism.ray_tracer_cuda import CUDARayTracer
from prism.training_interface import PrismTrainingInterface
from prism.data_utils import load_and_split_data, check_dataset_compatibility

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestingProgressMonitor:
    """Real-time testing progress monitor with GPU utilization tracking"""
    
    def __init__(self, total_samples: int, batch_size: int):
        """Initialize testing progress monitor"""
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.total_batches = (total_samples + batch_size - 1) // batch_size
        
        # Timing
        self.start_time = None
        self.batch_start_time = None
        self.batch_times = []
        
        # Progress tracking
        self.current_batch = 0
        self.processed_samples = 0
        
        # Performance metrics
        self.gpu_utilization_history = []
        self.memory_usage_history = []
        
        logger.info(f"🔍 Testing Progress Monitor initialized:")
        logger.info(f"   • Total samples: {self.total_samples:,}")
        logger.info(f"   • Batch size: {self.batch_size}")
        logger.info(f"   • Total batches: {self.total_batches}")
    
    def start_testing(self):
        """Start testing monitoring"""
        self.start_time = time.time()
        print(f"\n🧪 Starting CSI Inference Progress Monitoring")
        print(f"{'='*80}")
        print(f"📊 Total Samples: {self.total_samples:,}")
        print(f"📦 Batch Size: {self.batch_size}")
        print(f"🔄 Total Batches: {self.total_batches}")
        print(f"{'='*80}")
    
    def start_batch(self, batch_idx: int, batch_size: int):
        """Start batch processing"""
        self.current_batch = batch_idx + 1
        self.batch_start_time = time.time()
        self.processed_samples = min(batch_idx * self.batch_size + batch_size, self.total_samples)
        
        # Show batch start info
        progress_percent = (self.processed_samples / self.total_samples) * 100
        print(f"\n🔄 Batch {self.current_batch}/{self.total_batches} ({progress_percent:.1f}%)")
        print(f"    📦 Processing samples {batch_idx * self.batch_size + 1} to {self.processed_samples}")
    
    def update_batch_progress(self, batch_idx: int, batch_size: int):
        """Update batch progress with performance metrics"""
        batch_time = time.time() - self.batch_start_time
        total_time = time.time() - self.start_time
        
        # Calculate progress
        progress_percent = (self.processed_samples / self.total_samples) * 100
        
        # Calculate timing estimates
        avg_batch_time = np.mean(self.batch_times[-10:]) if len(self.batch_times) >= 10 else (total_time / self.current_batch)
        remaining_batches = self.total_batches - self.current_batch
        eta = avg_batch_time * remaining_batches
        
        # Get GPU info
        gpu_info = self._get_gpu_info()
        memory_info = self._get_memory_info()
        
        # Calculate throughput
        samples_per_second = self.processed_samples / total_time if total_time > 0 else 0
        
        # Display progress
        print(f"    ✅ Batch {self.current_batch}/{self.total_batches} completed in {batch_time:.1f}s")
        print(f"    📊 Progress: {progress_percent:.1f}% ({self.processed_samples:,}/{self.total_samples:,} samples)")
        print(f"    ⚡ Throughput: {samples_per_second:.1f} samples/s")
        print(f"    ⏱️  Timing: Batch {batch_time:.1f}s | Total {total_time:.1f}s | ETA {eta:.1f}s")
        print(f"    🔍 GPU: {gpu_info} | Memory: {memory_info}")
        
        # Store metrics
        self.batch_times.append(batch_time)
        self.gpu_utilization_history.append(gpu_info)
        self.memory_usage_history.append(memory_info)
        
        # Show heartbeat for longer batches
        if batch_time > 30:  # If batch takes more than 30 seconds
            current_time = time.strftime('%H:%M:%S')
            print(f"    💓 Heartbeat: {current_time} - Testing is alive and running!")
    
    def end_testing(self):
        """End testing monitoring and show summary"""
        total_time = time.time() - self.start_time
        
        print(f"\n✅ CSI Inference Completed!")
        print(f"{'='*80}")
        print(f"📊 Testing Summary:")
        print(f"   • Total samples processed: {self.processed_samples:,}")
        print(f"   • Total batches: {self.current_batch}")
        print(f"   • Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        
        if len(self.batch_times) > 0:
            avg_batch_time = np.mean(self.batch_times)
            avg_samples_per_second = self.processed_samples / total_time
            print(f"   • Average batch time: {avg_batch_time:.1f}s")
            print(f"   • Average throughput: {avg_samples_per_second:.1f} samples/s")
        
        # Show performance summary
        if len(self.batch_times) > 1:
            fastest_batch = min(self.batch_times)
            slowest_batch = max(self.batch_times)
            print(f"   • Fastest batch: {fastest_batch:.1f}s")
            print(f"   • Slowest batch: {slowest_batch:.1f}s")
        
        print(f"{'='*80}")
    
    def _get_gpu_info(self):
        """Get current GPU utilization information"""
        if not GPU_AVAILABLE or not torch.cuda.is_available():
            return "N/A"
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return f"{gpu.load*100:.1f}%"
            else:
                return "No GPU"
        except Exception:
            return "Error"
    
    def _get_memory_info(self):
        """Get current memory usage information"""
        if not torch.cuda.is_available():
            return "CPU"
        
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            return f"{memory_allocated:.1f}GB/{memory_reserved:.1f}GB"
        except Exception:
            return "Error"
    
    def get_performance_summary(self):
        """Get performance summary for final reporting"""
        if not self.batch_times:
            return "No performance data available"
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_time': total_time,
            'total_samples': self.processed_samples,
            'total_batches': self.current_batch,
            'avg_batch_time': np.mean(self.batch_times),
            'avg_throughput': self.processed_samples / total_time if total_time > 0 else 0,
            'fastest_batch': min(self.batch_times),
            'slowest_batch': max(self.batch_times),
            'gpu_utilization_avg': np.mean([float(gpu.replace('%', '')) for gpu in self.gpu_utilization_history if gpu != "N/A" and gpu != "Error"]) if self.gpu_utilization_history else 0
        }

class PrismTester:
    """Main tester class for Prism network using TrainingInterface"""
    
    def __init__(self, config_path: str, model_path: str = None, data_path: str = None, output_dir: str = None, gpu_id: int = None):
        """Initialize tester with configuration and optional model/data/output paths"""
        self.config_path = config_path
        self.gpu_id = gpu_id  # Store GPU ID for device setup
        
        # Load configuration first using ConfigLoader to process template variables
        try:
            config_loader = ConfigLoader(config_path)
            self.config = config_loader.config
            logger.info(f"Configuration loaded from: {config_path}")
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: Failed to load configuration from {config_path}")
            logger.error(f"   Error details: {str(e)}")
            raise
        
        # Setup proper logging after config is loaded
        try:
            self._setup_logging()
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: Failed to setup logging")
            logger.error(f"   Error details: {str(e)}")
            raise
        
        # Set model path, data path and output directory from config if not provided
        try:
            self.model_path = model_path or self.config['testing']['model_path']
            
            # Check dataset configuration
            dataset_path, split_config = check_dataset_compatibility(self.config)
            
            # Use single dataset with split
            self.data_path = data_path or dataset_path
            self.split_config = split_config
            logger.info(f"Using single dataset with train/test split: {self.data_path}")
            logger.info(f"Split configuration: {self.split_config}")
                
        except KeyError as e:
            logger.error(f"❌ FATAL ERROR: Missing required configuration key: {e}")
            raise
        
        # Build output directory from base_dir
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            base_dir = self.config['output'].get('base_dir', 'results')
            self.output_dir = Path(base_dir) / 'testing'
        
        # Create all necessary testing directories
        self._create_testing_directories()
        
        # Initialize device and model
        self.device = self._setup_device()
        self.model = None
        self.checkpoint_info = {}
        
        # Load model and data
        self._load_model()
        self._load_data()
    
    def _setup_logging(self):
        """Setup logging with proper file path from config for testing"""
        # Get testing logging configuration from config (under output.testing section)
        output_config = self.config.get('output', {})
        testing_config = output_config.get('testing', {})
        testing_logging_config = testing_config.get('logging', {})
        
        if not testing_logging_config:
            logger.error("❌ FATAL ERROR: No testing_logging configuration found in config")
            raise ValueError("testing_logging configuration is required")
        
        log_level_str = testing_logging_config.get('log_level', 'INFO')
        log_file = testing_logging_config.get('log_file')
        
        if not log_file:
            logger.error("❌ FATAL ERROR: No log_file found in testing_logging config")
            raise ValueError("log_file is required in testing_logging configuration")
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"📁 Created testing log directory: {log_dir}")
        
        # Convert string log level to logging constant
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
        # Clear any existing handlers to avoid duplicates
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)
        
        # Add file handler for testing
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        
        # Add to root logger
        root_logger.addHandler(file_handler)
        root_logger.setLevel(log_level)
        
        logger.info(f"🧪 Testing logging setup complete. Log file: {log_file}")
        logger.info(f"📊 Testing logging level: {log_level_str}")
        
        # Log testing configuration details
        logger.info("🔧 Testing logging configuration:")
        for key, value in testing_logging_config.items():
            logger.info(f"   • {key}: {value}")
    
    def _create_testing_directories(self):
        """Create all necessary testing output directories"""
        try:
            output_config = self.config.get('output', {})
            testing_config = output_config.get('testing', {})
            testing_logging_config = testing_config.get('logging', {})
            
            # Get all testing directory paths
            directories = {
                'results': testing_config.get('results_dir'),
                'plots': testing_config.get('plots_dir'),
                'predictions': testing_config.get('predictions_dir'),
                'reports': testing_config.get('reports_dir'),
                'logs': testing_logging_config.get('log_dir')
            }
            
            logger.info("📁 Creating testing output directories...")
            
            for dir_name, dir_path in directories.items():
                if not dir_path:
                    logger.error(f"❌ FATAL ERROR: No path configured for: {dir_name}")
                    raise ValueError(f"Missing directory path for {dir_name}")
                
                # Create directory if it doesn't exist
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"   ✅ Created: {dir_name} -> {dir_path}")
                else:
                    logger.info(f"   📁 Exists: {dir_name} -> {dir_path}")
            
            # Also create the main testing directory if it doesn't exist
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
                logger.info(f"   ✅ Created main testing directory: {self.output_dir}")
            
            logger.info("📁 Testing directories setup complete!")
            
        except Exception as e:
            logger.error(f"❌ Failed to create testing directories: {e}")
            raise
    
    def _setup_device(self):
        """Setup device for testing"""
        logger.info("🔍 Scanning available GPUs for testing...")
        
        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA not available, using CPU")
            return torch.device('cpu')
        
        # Get GPU information
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            gpu_free = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB, {gpu_free:.1f}GB free)")
        
        # GPU selection: use specified GPU ID or auto-select
        if self.gpu_id is not None:
            # Use specified GPU ID
            if self.gpu_id >= gpu_count or self.gpu_id < 0:
                logger.error(f"❌ Invalid GPU ID {self.gpu_id}. Available GPUs: 0-{gpu_count-1}")
                raise ValueError(f"Invalid GPU ID {self.gpu_id}. Available GPUs: 0-{gpu_count-1}")
            selected_gpu = self.gpu_id
            logger.info(f"🎯 Using specified GPU {selected_gpu} for testing")
        else:
            # Auto-select GPU 0 for testing
            selected_gpu = 0
            logger.info(f"🔍 Auto-selected GPU {selected_gpu} for testing")
        
        device = torch.device(f'cuda:{selected_gpu}')
        torch.cuda.set_device(device)
        
        logger.info("🔍 GPU Selection for Testing:")
        logger.info(f"  • Selected GPU: {selected_gpu}")
        logger.info(f"  • GPU Name: {torch.cuda.get_device_name(selected_gpu)}")
        logger.info(f"  • GPU Memory: {torch.cuda.get_device_properties(selected_gpu).total_memory / 1024**3:.1f}GB")
        logger.info(f"Using device: {device}")
        
        return device
    
    def _load_model(self):
        """Load trained Prism network model from TrainingInterface checkpoint"""
        logger.info(f"Loading TrainingInterface model from {self.model_path}")
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            error_msg = f"Model file not found: {self.model_path}"
            logger.error(f"❌ FATAL ERROR: {error_msg}")
            logger.error("Please ensure the model has been trained and the path is correct.")
            logger.error("Available model locations to check:")
            logger.error("  - results/sionna/training/models/best_model.pt")
            logger.error("  - results/sionna/training/checkpoints/")
            raise FileNotFoundError(error_msg)
        
        # Check if model file is readable
        if not os.access(self.model_path, os.R_OK):
            error_msg = f"Model file is not readable: {self.model_path}"
            logger.error(f"❌ FATAL ERROR: {error_msg}")
            logger.error("Please check file permissions.")
            raise PermissionError(error_msg)
        
        # Check file size (empty files are likely corrupted)
        file_size = os.path.getsize(self.model_path)
        if file_size == 0:
            error_msg = f"Model file is empty (0 bytes): {self.model_path}"
            logger.error(f"❌ FATAL ERROR: {error_msg}")
            logger.error("The model file appears to be corrupted or incomplete.")
            raise ValueError(error_msg)
        
        logger.info(f"Model file found: {self.model_path} ({file_size / 1024 / 1024:.1f} MB)")
        
        try:
            # Check if this is a TrainingInterface checkpoint
            if ('checkpoint_epoch_' in str(self.model_path) or 'best_model.pt' in str(self.model_path) or 
                'latest_checkpoint.pt' in str(self.model_path) or 'latest_batch_checkpoint.pt' in str(self.model_path)):
                # This is a TrainingInterface checkpoint
                self._load_training_interface_checkpoint()
            else:
                error_msg = "Unsupported checkpoint format. Only TrainingInterface checkpoints are supported."
                logger.error(f"❌ FATAL ERROR: {error_msg}")
                logger.error(f"Expected checkpoint file names: *best_model.pt, *checkpoint_epoch_*.pt, *latest_checkpoint.pt")
                raise ValueError(error_msg)
                
        except (FileNotFoundError, PermissionError, ValueError) as e:
            # Re-raise these specific errors without modification
            raise
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: Failed to load model: {e}")
            logger.error(f"Model path: {self.model_path}")
            logger.error(f"Error type: {type(e).__name__}")
            if hasattr(e, '__traceback__'):
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _load_training_interface_checkpoint(self):
        """Load TrainingInterface checkpoint"""
        try:
            # Create PrismNetwork and ray tracer first
            nn_config = self.config['neural_networks']
            ray_tracing_config = self.config['ray_tracing']
            ue_config = self.config.get('user_equipment', {})
            
            # Get angular sampling config
            angular_sampling = ray_tracing_config.get('angular_sampling', {})
            
            self.prism_network = PrismNetwork(
                num_subcarriers=nn_config['attenuation_decoder']['output_dim'],
                num_ue_antennas=ue_config.get('num_ue_antennas', 4),
                num_bs_antennas=nn_config['antenna_codebook']['num_antennas'],
                position_dim=nn_config['attenuation_network']['input_dim'],
                hidden_dim=nn_config['attenuation_network']['hidden_dim'],
                feature_dim=nn_config['attenuation_network']['feature_dim'],
                antenna_embedding_dim=nn_config['antenna_codebook']['embedding_dim'],
                use_antenna_codebook=nn_config['antenna_codebook']['learnable'],
                use_ipe_encoding=True,
                azimuth_divisions=angular_sampling.get('azimuth_divisions', 18),
                elevation_divisions=angular_sampling.get('elevation_divisions', 9),
                top_k_directions=angular_sampling.get('top_k_directions', 32),
                complex_output=True
            )
            
            # Get configuration sections
            ray_tracing_config = self.config.get('ray_tracing', {})
            system_config = self.config.get('system', {})
            
            # Use subcarrier sampling configuration from config file (no override for testing)
            subcarrier_sampling = ray_tracing_config.get('subcarrier_sampling', {})
            
            # Get ray tracing execution settings from system config
            ray_tracing_mode = system_config.get('ray_tracing_mode', 'cuda')
            fallback_to_cpu = system_config.get('fallback_to_cpu', True)
            
            logger.info(f"Ray tracer configuration:")
            logger.info(f"  - Ray tracing mode: {ray_tracing_mode}")
            logger.info(f"  - CUDA available: {torch.cuda.is_available()}")
            logger.info(f"  - Fallback to CPU: {fallback_to_cpu}")
            
            # Log subcarrier sampling configuration
            subcarrier_ratio = subcarrier_sampling.get('sampling_ratio', 1.0)
            total_subcarriers = self.prism_network.num_subcarriers
            selected_subcarriers = int(total_subcarriers * subcarrier_ratio)
            
            logger.info(f"🔧 Subcarrier sampling configuration for testing:")
            logger.info(f"   - Method: {subcarrier_sampling.get('method', 'random')}")
            logger.info(f"   - Sampling ratio: {subcarrier_ratio}")
            logger.info(f"   - Selected subcarriers: {selected_subcarriers}/{total_subcarriers}")
            
            # Calculate max ray length from scene bounds
            scene_bounds = ray_tracing_config.get('scene_bounds', {})
            max_ray_length = scene_bounds.get('max_ray_length', 200.0)
            logger.info(f"�� Calculated max_ray_length: {max_ray_length}m from scene bounds")
            
            # Create ray tracer based on mode
            if ray_tracing_mode == 'cuda' and torch.cuda.is_available():
                logger.info("🚀 Using CUDA-accelerated ray tracer")
                self.ray_tracer = CUDARayTracer(
                    max_ray_length=max_ray_length,
                    prism_network=self.prism_network
                )
            elif ray_tracing_mode == 'cpu' or not torch.cuda.is_available():
                logger.info("🖥️  Using CPU ray tracer")
                self.ray_tracer = CPURayTracer(
                    max_ray_length=max_ray_length,
                    prism_network=self.prism_network
                )
            else:
                logger.error("❌ FATAL ERROR: Invalid ray tracing mode")
                raise ValueError(f"Invalid ray tracing mode: {ray_tracing_mode}")
            
            # Create TrainingInterface
            training_config = self.config.get('training', {})
            checkpoint_dir = self.config['output']['training']['checkpoint_dir']
            
            self.model = PrismTrainingInterface(
                prism_network=self.prism_network,
                ray_tracing_config=ray_tracing_config,
                system_config=system_config,
                user_equipment_config=ue_config,
                checkpoint_dir=checkpoint_dir
            )
            
            # Load checkpoint with detailed error handling
            try:
                logger.info("Loading checkpoint file...")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                logger.info("Checkpoint file loaded successfully")
            except Exception as e:
                error_msg = f"Failed to load checkpoint file: {e}"
                logger.error(f"❌ FATAL ERROR: {error_msg}")
                logger.error("This could indicate:")
                logger.error("  - Corrupted checkpoint file")
                logger.error("  - Incompatible PyTorch version")
                logger.error("  - Insufficient memory")
                logger.error("  - Invalid checkpoint format")
                raise RuntimeError(error_msg) from e
            
            # Validate checkpoint structure
            if 'model_state_dict' not in checkpoint:
                error_msg = "Checkpoint file is missing 'model_state_dict'. This is not a valid TrainingInterface checkpoint."
                logger.error(f"❌ FATAL ERROR: {error_msg}")
                logger.error(f"Available keys in checkpoint: {list(checkpoint.keys())}")
                raise ValueError(error_msg)
            
            # Load model state with error handling
            try:
                logger.info("Loading model state dictionary...")
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Model state dictionary loaded successfully")
            except Exception as e:
                error_msg = f"Failed to load model state dictionary: {e}"
                logger.error(f"❌ FATAL ERROR: {error_msg}")
                logger.error("This could indicate:")
                logger.error("  - Model architecture mismatch")
                logger.error("  - Incompatible checkpoint version")
                logger.error("  - Missing or extra model parameters")
                raise RuntimeError(error_msg) from e
            
            self.checkpoint_info = checkpoint.get('checkpoint_info', {})
            
            # Set device for model and ray tracer with error handling
            try:
                logger.info(f"Moving model to device: {self.device}")
                self.model.to(self.device)
                self.ray_tracer.device = self.device
                logger.info(f"Set ray tracer device to: {self.device}")
                
                # Set CUDA device for ray tracer
                if hasattr(self.ray_tracer, 'cuda_device'):
                    self.ray_tracer.cuda_device = self.device
                    logger.info(f"Set CUDA device to: {self.device}")
                    
            except Exception as e:
                error_msg = f"Failed to move model to device {self.device}: {e}"
                logger.error(f"❌ FATAL ERROR: {error_msg}")
                logger.error("This could indicate:")
                logger.error("  - Insufficient GPU memory")
                logger.error("  - CUDA driver issues")
                logger.error("  - Device compatibility problems")
                raise RuntimeError(error_msg) from e
            
            # Log successful loading
            try:
                param_count = sum(p.numel() for p in self.model.parameters())
                logger.info(f"✅ TrainingInterface model loaded successfully with {param_count:,} parameters")
                logger.info(f"✅ Checkpoint info: {self.checkpoint_info}")
            except Exception as e:
                logger.warning(f"Could not count model parameters: {e}")
                logger.info(f"✅ TrainingInterface model loaded successfully")
                logger.info(f"✅ Checkpoint info: {self.checkpoint_info}")
            
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: Failed to load TrainingInterface checkpoint: {e}")
            logger.error(f"Checkpoint path: {self.model_path}")
            logger.error(f"Device: {self.device}")
            raise
    
    def _load_data(self):
        """Load test data from HDF5 file"""
        logger.info(f"Loading test data from {self.data_path}")
        
        try:
            # Use split-based data loading
            self.ue_positions, self.csi_target, self.bs_position, self.antenna_indices, metadata = load_and_split_data(
                dataset_path=self.data_path,
                train_ratio=self.split_config['train_ratio'],
                test_ratio=self.split_config['test_ratio'],
                random_seed=self.split_config['random_seed'],
                mode='test'
            )
            
            # Log split information
            logger.info(f"Using train/test split mode")
            logger.info(f"Random seed: {self.split_config['random_seed']}")
            logger.info(f"Train ratio: {self.split_config['train_ratio']}")
            logger.info(f"Test ratio: {self.split_config['test_ratio']}")
            logger.info(f"UE positions: {self.ue_positions.shape[0]} samples (testing split)")
            logger.info(f"CSI data: {self.csi_target.shape}")
            logger.info(f"BS position: {self.bs_position.shape}")
            logger.info(f"Antenna indices: {len(self.antenna_indices)}")
            
            # Store metadata
            self.split_metadata = metadata
            self.sim_params = metadata.get('simulation_params', {})
            
            # Move to device
            self.ue_positions = self.ue_positions.to(self.device)
            self.csi_target = self.csi_target.to(self.device)
            self.bs_position = self.bs_position.to(self.device)
            self.antenna_indices = self.antenna_indices.to(self.device)
            
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: Failed to load test data: {e}")
            raise
    
    def _plot_csi_magnitude_comparison(self, plots_dir: Path):
        """Plot CSI magnitude comparison between predicted and target"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Get sample indices for visualization
        sample_indices = [0, len(self.ue_positions)//2, -1]
        
        for i, sample_idx in enumerate(sample_indices):
            if sample_idx >= len(self.ue_positions):
                continue
                
            row = i // 2
            col = i % 2
            
            # Get data for this sample
            pred_mag = torch.abs(self.predictions[sample_idx]).numpy()
            target_mag = torch.abs(self.csi_target[sample_idx]).cpu().numpy()
            
            # Handle different data shapes - reshape if needed for visualization
            if pred_mag.ndim == 3:
                # If 3D (e.g., UEs, subcarriers, antennas), take mean over first dimension or reshape
                pred_mag = pred_mag.mean(axis=0) if pred_mag.shape[0] > 1 else pred_mag[0]
            if target_mag.ndim == 3:
                target_mag = target_mag.mean(axis=0) if target_mag.shape[0] > 1 else target_mag[0]
            
            # Plot magnitude comparison
            im1 = axes[row, col].imshow(pred_mag, aspect='auto', cmap='viridis')
            axes[row, col].set_title(f'Sample {sample_idx}: Predicted Magnitude')
            axes[row, col].set_xlabel('Antenna Index')
            axes[row, col].set_ylabel('Subcarrier Index')
            plt.colorbar(im1, ax=axes[row, col])
        
        plt.tight_layout()
        plot_path = plots_dir / 'csi_magnitude_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"CSI magnitude comparison plot saved: {plot_path}")
    
    def _plot_csi_phase_comparison(self, plots_dir: Path):
        """Plot CSI phase comparison between predicted and target"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Get sample indices for visualization
        sample_indices = [0, len(self.ue_positions)//2, -1]
        
        for i, sample_idx in enumerate(sample_indices):
            if sample_idx >= len(self.ue_positions):
                continue
                
            row = i // 2
            col = i % 2
            
            # Get data for this sample
            pred_phase = torch.angle(self.predictions[sample_idx]).numpy()
            target_phase = torch.angle(self.csi_target[sample_idx]).cpu().numpy()
            
            # Handle different data shapes - reshape if needed for visualization
            if pred_phase.ndim == 3:
                # If 3D (e.g., UEs, subcarriers, antennas), take mean over first dimension or reshape
                pred_phase = pred_phase.mean(axis=0) if pred_phase.shape[0] > 1 else pred_phase[0]
            if target_phase.ndim == 3:
                target_phase = target_phase.mean(axis=0) if target_phase.shape[0] > 1 else target_phase[0]
            
            # Plot phase comparison
            im1 = axes[row, col].imshow(pred_phase, aspect='auto', cmap='twilight')
            axes[row, col].set_title(f'Sample {sample_idx}: Predicted Phase')
            axes[row, col].set_xlabel('Antenna Index')
            axes[row, col].set_ylabel('Subcarrier Index')
            plt.colorbar(im1, ax=axes[row, col])
        
        plt.tight_layout()
        plot_path = plots_dir / 'csi_phase_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"CSI phase comparison plot saved: {plot_path}")
    
    def _plot_error_distribution(self, plots_dir: Path):
        """Plot error distribution with error bars per subcarrier"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate errors
        pred_np = self.predictions.numpy()
        target_np = self.csi_target.cpu().numpy()
        
        # Calculate errors for each sample
        mag_error = np.abs(np.abs(pred_np) - np.abs(target_np))
        pred_phase = np.angle(pred_np)
        target_phase = np.angle(target_np)
        phase_diff = pred_phase - target_phase
        phase_diff = np.angle(np.exp(1j * phase_diff))
        phase_error = np.abs(phase_diff)
        
        # Debug: Check data shape
        logger.info(f"Error data shape: {mag_error.shape}")
        
        # Calculate statistics per subcarrier
        # Shape is (samples, time_steps, subcarriers, antennas)
        logger.info(f"Processing error data with shape: {mag_error.shape}")
        
        if mag_error.ndim == 4:
            # Shape: (samples, time_steps, subcarriers, antennas)
            # Average over samples (axis=0), time_steps (axis=1), and antennas (axis=3)
            # Keep subcarriers (axis=2)
            mag_error_mean = np.mean(mag_error, axis=(0, 1, 3))  # Result shape: (subcarriers,)
            mag_error_std = np.std(mag_error, axis=(0, 1, 3))
            phase_error_mean = np.mean(phase_error, axis=(0, 1, 3))
            phase_error_std = np.std(phase_error, axis=(0, 1, 3))
            
        elif mag_error.ndim == 3:
            # Shape: (samples, subcarriers, antennas) or (samples, time_steps, subcarriers)
            # Assume subcarriers are in the middle dimension (axis=1)
            mag_error_mean = np.mean(mag_error, axis=(0, 2))  # Average over samples and antennas
            mag_error_std = np.std(mag_error, axis=(0, 2))
            phase_error_mean = np.mean(phase_error, axis=(0, 2))
            phase_error_std = np.std(phase_error, axis=(0, 2))
            
        elif mag_error.ndim == 2:
            # Shape: (samples, subcarriers)
            mag_error_mean = np.mean(mag_error, axis=0)  # Average over samples
            mag_error_std = np.std(mag_error, axis=0)
            phase_error_mean = np.mean(phase_error, axis=0)
            phase_error_std = np.std(phase_error, axis=0)
            
        else:
            # If 1D, treat as single subcarrier
            mag_error_mean = np.array([np.mean(mag_error)])
            mag_error_std = np.array([np.std(mag_error)])
            phase_error_mean = np.array([np.mean(phase_error)])
            phase_error_std = np.array([np.std(phase_error)])
        
        # Ensure we have arrays, not scalars
        mag_error_mean = np.atleast_1d(mag_error_mean)
        mag_error_std = np.atleast_1d(mag_error_std)
        phase_error_mean = np.atleast_1d(phase_error_mean)
        phase_error_std = np.atleast_1d(phase_error_std)
        
        logger.info(f"Processed error statistics - mag_error_mean shape: {mag_error_mean.shape}")
        
        # Create subcarrier indices
        subcarrier_indices = np.arange(len(mag_error_mean))
        
        # Plot magnitude error with error bars per subcarrier
        axes[0].errorbar(subcarrier_indices, mag_error_mean, yerr=mag_error_std, 
                        fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=6,
                        color='blue', ecolor='lightblue', alpha=0.8)
        axes[0].set_xlabel('Subcarrier Index')
        axes[0].set_ylabel('Magnitude Error')
        axes[0].set_title('Magnitude Error per Subcarrier (Mean ± Std)')
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics text
        overall_mag_mean = np.mean(mag_error_mean)
        overall_mag_std = np.mean(mag_error_std)
        axes[0].text(0.02, 0.98, f'Overall Mean: {overall_mag_mean:.4f}\\nAvg Std: {overall_mag_std:.4f}',
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot phase error with error bars per subcarrier
        axes[1].errorbar(subcarrier_indices, phase_error_mean, yerr=phase_error_std,
                        fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=6,
                        color='red', ecolor='lightcoral', alpha=0.8)
        axes[1].set_xlabel('Subcarrier Index')
        axes[1].set_ylabel('Phase Error (radians)')
        axes[1].set_title('Phase Error per Subcarrier (Mean ± Std)')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics text
        overall_phase_mean = np.mean(phase_error_mean)
        overall_phase_std = np.mean(phase_error_std)
        axes[1].text(0.02, 0.98, f'Overall Mean: {overall_phase_mean:.4f}\\nAvg Std: {overall_phase_std:.4f}',
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plot_path = plots_dir / 'error_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Error distribution plot saved: {plot_path}")
        logger.info(f"Magnitude error per subcarrier - Mean: {overall_mag_mean:.6f}, Avg Std: {overall_mag_std:.6f}")
        logger.info(f"Phase error per subcarrier - Mean: {overall_phase_mean:.6f}, Avg Std: {overall_phase_std:.6f}")
    
    def _plot_spatial_performance(self, plots_dir: Path):
        """Plot spatial performance map showing error by UE position"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate error per UE position
        pred_np = self.predictions.numpy()
        target_np = self.csi_target.cpu().numpy()
        
        # Calculate error per UE - need to handle different data shapes
        # First calculate magnitude and phase errors
        mag_error = np.abs(np.abs(pred_np) - np.abs(target_np))
        pred_phase = np.angle(pred_np)
        target_phase = np.angle(target_np)
        phase_diff = pred_phase - target_phase
        phase_diff = np.angle(np.exp(1j * phase_diff))
        phase_error = np.abs(phase_diff)
        
        # Average over all dimensions except the first (samples/UEs)
        # This handles any shape: (samples, ...) -> (samples,)
        axes_to_average = tuple(range(1, mag_error.ndim)) if mag_error.ndim > 1 else None
        if axes_to_average:
            mag_error_per_ue = np.mean(mag_error, axis=axes_to_average)
            phase_error_per_ue = np.mean(phase_error, axis=axes_to_average)
        else:
            mag_error_per_ue = mag_error
            phase_error_per_ue = phase_error
        
        # Get UE positions
        ue_pos_np = self.ue_positions.cpu().numpy()
        
        # Debug: Check dimensions
        logger.info(f"UE positions shape: {ue_pos_np.shape}")
        logger.info(f"Magnitude error shape: {mag_error_per_ue.shape}")
        logger.info(f"Phase error shape: {phase_error_per_ue.shape}")
        
        # Ensure error arrays match UE position count
        if mag_error_per_ue.shape != (len(ue_pos_np),):
            logger.warning(f"Dimension mismatch: mag_error shape {mag_error_per_ue.shape} vs {len(ue_pos_np)} UE positions")
            # If still 3D, we need to average over the extra dimensions
            if mag_error_per_ue.ndim > 1:
                mag_error_per_ue = np.mean(mag_error_per_ue, axis=tuple(range(1, mag_error_per_ue.ndim)))
                phase_error_per_ue = np.mean(phase_error_per_ue, axis=tuple(range(1, phase_error_per_ue.ndim)))
                logger.info(f"Averaged error arrays to shape: {mag_error_per_ue.shape}")
            # Take only the first N values to match UE count if still too long
            if len(mag_error_per_ue) > len(ue_pos_np):
                mag_error_per_ue = mag_error_per_ue[:len(ue_pos_np)]
                phase_error_per_ue = phase_error_per_ue[:len(ue_pos_np)]
        
        # Plot magnitude error
        scatter1 = axes[0].scatter(ue_pos_np[:, 0], ue_pos_np[:, 1], 
                                  c=mag_error_per_ue, cmap='viridis', s=50)
        axes[0].set_xlabel('X Position (m)')
        axes[0].set_ylabel('Y Position (m)')
        axes[0].set_title('Magnitude Error by UE Position')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Magnitude Error')
        
        # Plot phase error
        scatter2 = axes[1].scatter(ue_pos_np[:, 0], ue_pos_np[:, 1], 
                                  c=phase_error_per_ue, cmap='plasma', s=50)
        axes[1].set_xlabel('X Position (m)')
        axes[1].set_ylabel('Y Position (m)')
        axes[1].set_title('Phase Error by UE Position')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Phase Error (radians)')
        
        plt.tight_layout()
        plot_path = plots_dir / 'spatial_performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Spatial performance plot saved: {plot_path}")
    
    def _plot_subcarrier_performance(self, plots_dir: Path):
        """Plot performance across subcarriers"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        pred_np = self.predictions.numpy()
        target_np = self.csi_target.cpu().numpy()
        
        # Debug: Print shapes to understand the data structure
        logger.info(f"Predictions shape: {pred_np.shape}")
        logger.info(f"Target shape: {target_np.shape}")
        
        # Calculate error per subcarrier
        # First calculate the errors
        mag_error = np.abs(np.abs(pred_np) - np.abs(target_np))
        phase_error = np.abs(np.angle(pred_np) - np.angle(target_np))
        
        logger.info(f"Magnitude error shape: {mag_error.shape}")
        
        # We need to determine which axis represents subcarriers
        # Based on the error message, it seems like the shape is (samples, ?, subcarriers, antennas)
        # Let's assume the subcarrier dimension is axis=1 (second dimension)
        if mag_error.ndim == 4:
            # Shape: (samples, ?, subcarriers, antennas) -> average over samples, first dim, and antennas
            mag_error_per_subcarrier = np.mean(mag_error, axis=(0, 1, 3))
            phase_error_per_subcarrier = np.mean(phase_error, axis=(0, 1, 3))
        elif mag_error.ndim == 3:
            # Shape: (samples, subcarriers, antennas) -> average over samples and antennas
            mag_error_per_subcarrier = np.mean(mag_error, axis=(0, 2))
            phase_error_per_subcarrier = np.mean(phase_error, axis=(0, 2))
        elif mag_error.ndim == 2:
            # Average over samples (axis=0)
            mag_error_per_subcarrier = np.mean(mag_error, axis=0)
            phase_error_per_subcarrier = np.mean(phase_error, axis=0)
        else:
            # If 1D, use as is
            mag_error_per_subcarrier = mag_error
            phase_error_per_subcarrier = phase_error
        
        logger.info(f"Final mag_error_per_subcarrier shape: {mag_error_per_subcarrier.shape}")
        
        subcarriers = range(len(mag_error_per_subcarrier))
        
        # Magnitude error per subcarrier
        axes[0].plot(subcarriers, mag_error_per_subcarrier, 'b-', linewidth=2)
        axes[0].set_xlabel('Subcarrier Index')
        axes[0].set_ylabel('Average Magnitude Error')
        axes[0].set_title('Magnitude Error by Subcarrier')
        axes[0].grid(True, alpha=0.3)
        
        # Phase error per subcarrier
        axes[1].plot(subcarriers, phase_error_per_subcarrier, 'r-', linewidth=2)
        axes[1].set_xlabel('Subcarrier Index')
        axes[1].set_ylabel('Average Phase Error (radians)')
        axes[1].set_title('Phase Error by Subcarrier')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = plots_dir / 'subcarrier_performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Subcarrier performance plot saved: {plot_path}")
    
    def _plot_error_cdf(self, plots_dir: Path):
        """Plot CDF of magnitude and phase errors"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        pred_np = self.predictions.numpy()
        target_np = self.csi_target.cpu().numpy()
        
        # Calculate errors for all samples
        mag_error = np.abs(np.abs(pred_np) - np.abs(target_np))
        pred_phase = np.angle(pred_np)
        target_phase = np.angle(target_np)
        phase_diff = pred_phase - target_phase
        # Wrap phase difference to [-π, π]
        phase_diff = np.angle(np.exp(1j * phase_diff))
        phase_error = np.abs(phase_diff)
        
        # Flatten all errors for CDF calculation
        mag_error_flat = mag_error.flatten()
        phase_error_flat = phase_error.flatten()
        
        # Remove any NaN or infinite values
        mag_error_flat = mag_error_flat[np.isfinite(mag_error_flat)]
        phase_error_flat = phase_error_flat[np.isfinite(phase_error_flat)]
        
        # Calculate CDF for magnitude error
        mag_sorted = np.sort(mag_error_flat)
        mag_cdf = np.arange(1, len(mag_sorted) + 1) / len(mag_sorted)
        
        # Calculate CDF for phase error
        phase_sorted = np.sort(phase_error_flat)
        phase_cdf = np.arange(1, len(phase_sorted) + 1) / len(phase_sorted)
        
        # Plot magnitude error CDF
        axes[0].plot(mag_sorted, mag_cdf, 'b-', linewidth=2)
        axes[0].set_xlabel('Magnitude Error')
        axes[0].set_ylabel('Cumulative Probability')
        axes[0].set_title('CDF of Magnitude Error')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(left=0)
        
        # Add percentile lines
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            idx = int(p * len(mag_sorted) / 100)
            if idx < len(mag_sorted):
                axes[0].axvline(mag_sorted[idx], color='red', linestyle='--', alpha=0.7)
                axes[0].text(mag_sorted[idx], p/100, f'{p}%', rotation=90, 
                           verticalalignment='bottom', fontsize=8)
        
        # Plot phase error CDF
        axes[1].plot(phase_sorted, phase_cdf, 'r-', linewidth=2)
        axes[1].set_xlabel('Phase Error (radians)')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title('CDF of Phase Error')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(left=0)
        
        # Add percentile lines
        for p in percentiles:
            idx = int(p * len(phase_sorted) / 100)
            if idx < len(phase_sorted):
                axes[1].axvline(phase_sorted[idx], color='red', linestyle='--', alpha=0.7)
                axes[1].text(phase_sorted[idx], p/100, f'{p}%', rotation=90, 
                           verticalalignment='bottom', fontsize=8)
        
        plt.tight_layout()
        plot_path = plots_dir / 'error_cdf.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Error CDF plot saved: {plot_path}")
        
        # Log some statistics
        logger.info(f"Magnitude Error Statistics:")
        logger.info(f"  Mean: {np.mean(mag_error_flat):.6f}")
        logger.info(f"  Median: {np.median(mag_error_flat):.6f}")
        logger.info(f"  90th percentile: {np.percentile(mag_error_flat, 90):.6f}")
        logger.info(f"  95th percentile: {np.percentile(mag_error_flat, 95):.6f}")
        
        logger.info(f"Phase Error Statistics:")
        logger.info(f"  Mean: {np.mean(phase_error_flat):.6f} rad")
        logger.info(f"  Median: {np.median(phase_error_flat):.6f} rad")
        logger.info(f"  90th percentile: {np.percentile(phase_error_flat, 90):.6f} rad")
        logger.info(f"  95th percentile: {np.percentile(phase_error_flat, 95):.6f} rad")
    
    def _plot_pdp_analysis(self, plots_dir: Path):
        """Compute PDP for each CSI and plot CDF of PDP characteristics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        pred_np = self.predictions.numpy()
        target_np = self.csi_target.cpu().numpy()
        
        # Lists to store PDP characteristics
        pred_rms_delays = []
        target_rms_delays = []
        pred_mean_powers = []
        target_mean_powers = []
        pred_max_powers = []
        target_max_powers = []
        
        # Process each CSI sample
        for sample_idx in range(pred_np.shape[0]):
            # Get CSI for this sample
            pred_csi = pred_np[sample_idx]  # Shape depends on data structure
            target_csi = target_np[sample_idx]
            
            # Flatten spatial dimensions to get delay profiles
            # Assuming the second-to-last dimension represents delay/time
            if pred_csi.ndim >= 2:
                # Average over spatial dimensions to get power delay profile
                pred_pdp = np.mean(np.abs(pred_csi)**2, axis=tuple(range(pred_csi.ndim-1)))
                target_pdp = np.mean(np.abs(target_csi)**2, axis=tuple(range(target_csi.ndim-1)))
            else:
                pred_pdp = np.abs(pred_csi)**2
                target_pdp = np.abs(target_csi)**2
            
            # Normalize PDPs
            pred_pdp = pred_pdp / np.sum(pred_pdp) if np.sum(pred_pdp) > 0 else pred_pdp
            target_pdp = target_pdp / np.sum(target_pdp) if np.sum(target_pdp) > 0 else target_pdp
            
            # Calculate RMS delay spread
            delay_bins = np.arange(len(pred_pdp))
            
            # Predicted CSI statistics
            pred_mean_delay = np.sum(delay_bins * pred_pdp)
            pred_rms_delay = np.sqrt(np.sum((delay_bins - pred_mean_delay)**2 * pred_pdp))
            pred_rms_delays.append(pred_rms_delay)
            pred_mean_powers.append(np.mean(pred_pdp))
            pred_max_powers.append(np.max(pred_pdp))
            
            # Target CSI statistics
            target_mean_delay = np.sum(delay_bins * target_pdp)
            target_rms_delay = np.sqrt(np.sum((delay_bins - target_mean_delay)**2 * target_pdp))
            target_rms_delays.append(target_rms_delay)
            target_mean_powers.append(np.mean(target_pdp))
            target_max_powers.append(np.max(target_pdp))
        
        # Convert to numpy arrays
        pred_rms_delays = np.array(pred_rms_delays)
        target_rms_delays = np.array(target_rms_delays)
        pred_mean_powers = np.array(pred_mean_powers)
        target_mean_powers = np.array(target_mean_powers)
        pred_max_powers = np.array(pred_max_powers)
        target_max_powers = np.array(target_max_powers)
        
        # Remove any NaN or infinite values
        valid_mask = (np.isfinite(pred_rms_delays) & np.isfinite(target_rms_delays) & 
                     np.isfinite(pred_mean_powers) & np.isfinite(target_mean_powers) &
                     np.isfinite(pred_max_powers) & np.isfinite(target_max_powers))
        
        pred_rms_delays = pred_rms_delays[valid_mask]
        target_rms_delays = target_rms_delays[valid_mask]
        pred_mean_powers = pred_mean_powers[valid_mask]
        target_mean_powers = target_mean_powers[valid_mask]
        pred_max_powers = pred_max_powers[valid_mask]
        target_max_powers = target_max_powers[valid_mask]
        
        # Plot 1: CDF of RMS Delay Spread
        if len(pred_rms_delays) > 0:
            pred_rms_sorted = np.sort(pred_rms_delays)
            target_rms_sorted = np.sort(target_rms_delays)
            cdf_vals = np.arange(1, len(pred_rms_sorted) + 1) / len(pred_rms_sorted)
            
            axes[0, 0].plot(pred_rms_sorted, cdf_vals, 'b-', linewidth=2, label='Predicted')
            axes[0, 0].plot(target_rms_sorted, cdf_vals, 'r--', linewidth=2, label='Target')
            axes[0, 0].set_xlabel('RMS Delay Spread')
            axes[0, 0].set_ylabel('Cumulative Probability')
            axes[0, 0].set_title('CDF of RMS Delay Spread')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # Plot 2: CDF of Mean Power
        if len(pred_mean_powers) > 0:
            pred_mean_sorted = np.sort(pred_mean_powers)
            target_mean_sorted = np.sort(target_mean_powers)
            cdf_vals = np.arange(1, len(pred_mean_sorted) + 1) / len(pred_mean_sorted)
            
            axes[0, 1].plot(pred_mean_sorted, cdf_vals, 'b-', linewidth=2, label='Predicted')
            axes[0, 1].plot(target_mean_sorted, cdf_vals, 'r--', linewidth=2, label='Target')
            axes[0, 1].set_xlabel('Mean Power')
            axes[0, 1].set_ylabel('Cumulative Probability')
            axes[0, 1].set_title('CDF of Mean PDP Power')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # Plot 3: CDF of Peak Power
        if len(pred_max_powers) > 0:
            pred_max_sorted = np.sort(pred_max_powers)
            target_max_sorted = np.sort(target_max_powers)
            cdf_vals = np.arange(1, len(pred_max_sorted) + 1) / len(pred_max_sorted)
            
            axes[1, 0].plot(pred_max_sorted, cdf_vals, 'b-', linewidth=2, label='Predicted')
            axes[1, 0].plot(target_max_sorted, cdf_vals, 'r--', linewidth=2, label='Target')
            axes[1, 0].set_xlabel('Peak Power')
            axes[1, 0].set_ylabel('Cumulative Probability')
            axes[1, 0].set_title('CDF of Peak PDP Power')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # Plot 4: Scatter plot of RMS Delay Spread (Predicted vs Target)
        if len(pred_rms_delays) > 0:
            axes[1, 1].scatter(target_rms_delays, pred_rms_delays, alpha=0.6, s=20)
            
            # Add diagonal line for perfect prediction
            min_val = min(np.min(target_rms_delays), np.min(pred_rms_delays))
            max_val = max(np.max(target_rms_delays), np.max(pred_rms_delays))
            axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            axes[1, 1].set_xlabel('Target RMS Delay Spread')
            axes[1, 1].set_ylabel('Predicted RMS Delay Spread')
            axes[1, 1].set_title('RMS Delay Spread: Predicted vs Target')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
            
            # Calculate correlation coefficient
            if len(pred_rms_delays) > 1:
                correlation = np.corrcoef(target_rms_delays, pred_rms_delays)[0, 1]
                axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                               transform=axes[1, 1].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plot_path = plots_dir / 'pdp_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PDP analysis plot saved: {plot_path}")
        
        # Log PDP statistics
        if len(pred_rms_delays) > 0:
            logger.info(f"PDP Statistics:")
            logger.info(f"  RMS Delay Spread - Predicted mean: {np.mean(pred_rms_delays):.6f}")
            logger.info(f"  RMS Delay Spread - Target mean: {np.mean(target_rms_delays):.6f}")
            logger.info(f"  Mean Power - Predicted mean: {np.mean(pred_mean_powers):.6f}")
            logger.info(f"  Mean Power - Target mean: {np.mean(target_mean_powers):.6f}")
            logger.info(f"  Peak Power - Predicted mean: {np.mean(pred_max_powers):.6f}")
            logger.info(f"  Peak Power - Target mean: {np.mean(target_max_powers):.6f}")
        
        # Now compute and plot PDP differences
        self._plot_pdp_difference_pdf(plots_dir)
    
    def _plot_pdp_difference_pdf(self, plots_dir: Path):
        """Compute PDP for predicted and true CSI, calculate absolute difference, and plot PDF of PDP differences"""
        logger.info("Computing PDP differences and plotting PDF...")
        
        pred_np = self.predictions.numpy()
        target_np = self.csi_target.cpu().numpy()
        
        # Lists to store PDP data for all samples
        pred_pdp_list = []
        target_pdp_list = []
        pdp_diff_list = []
        
        # Process each CSI sample to compute PDP
        for sample_idx in range(pred_np.shape[0]):
            # Get CSI for this sample
            pred_csi = pred_np[sample_idx]  # Shape: (time_steps, subcarriers, antennas)
            target_csi = target_np[sample_idx]
            
            # Calculate PDP (Power Delay Profile) for each sample
            # Average over subcarriers and antennas to get power vs delay
            if pred_csi.ndim >= 2:
                # Average over all dimensions except the first (delay/time dimension)
                pred_pdp = np.mean(np.abs(pred_csi)**2, axis=tuple(range(1, pred_csi.ndim)))
                target_pdp = np.mean(np.abs(target_csi)**2, axis=tuple(range(1, target_csi.ndim)))
            else:
                pred_pdp = np.abs(pred_csi)**2
                target_pdp = np.abs(target_csi)**2
            
            # Store PDPs
            pred_pdp_list.append(pred_pdp)
            target_pdp_list.append(target_pdp)
            
            # Calculate absolute difference between PDPs
            pdp_diff = np.abs(pred_pdp - target_pdp)
            pdp_diff_list.append(pdp_diff)
        
        # Convert to numpy arrays
        pred_pdp_array = np.array(pred_pdp_list)  # Shape: (samples, delay_taps)
        target_pdp_array = np.array(target_pdp_list)
        pdp_diff_array = np.array(pdp_diff_list)
        
        # Flatten all PDP differences for PDF calculation
        pdp_diff_flat = pdp_diff_array.flatten()
        
        # Remove any NaN or infinite values
        pdp_diff_flat = pdp_diff_flat[np.isfinite(pdp_diff_flat)]
        
        if len(pdp_diff_flat) == 0:
            logger.warning("No valid PDP difference data found")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: PDF of PDP differences (histogram + KDE)
        from scipy import stats
        
        # Calculate histogram
        hist_counts, hist_bins, _ = axes[0, 0].hist(pdp_diff_flat, bins=50, density=True, 
                                                   alpha=0.7, color='skyblue', edgecolor='black')
        
        # Calculate and plot KDE (Kernel Density Estimation)
        if len(pdp_diff_flat) > 1:
            kde = stats.gaussian_kde(pdp_diff_flat)
            x_range = np.linspace(pdp_diff_flat.min(), pdp_diff_flat.max(), 200)
            kde_values = kde(x_range)
            axes[0, 0].plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
        
        axes[0, 0].set_xlabel('PDP Absolute Difference')
        axes[0, 0].set_ylabel('Probability Density')
        axes[0, 0].set_title('PDF of PDP Absolute Differences')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Add statistics text
        mean_diff = np.mean(pdp_diff_flat)
        std_diff = np.std(pdp_diff_flat)
        median_diff = np.median(pdp_diff_flat)
        axes[0, 0].text(0.65, 0.95, f'Mean: {mean_diff:.6f}\\nStd: {std_diff:.6f}\\nMedian: {median_diff:.6f}',
                       transform=axes[0, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: CDF of PDP differences
        pdp_diff_sorted = np.sort(pdp_diff_flat)
        cdf_values = np.arange(1, len(pdp_diff_sorted) + 1) / len(pdp_diff_sorted)
        
        axes[0, 1].plot(pdp_diff_sorted, cdf_values, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('PDP Absolute Difference')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].set_title('CDF of PDP Absolute Differences')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add percentile lines
        percentiles = [50, 90, 95, 99]
        colors = ['green', 'orange', 'red', 'purple']
        for p, color in zip(percentiles, colors):
            p_value = np.percentile(pdp_diff_flat, p)
            axes[0, 1].axvline(p_value, color=color, linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[0, 1].legend()
        
        # Plot 3: Average PDP comparison
        mean_pred_pdp = np.mean(pred_pdp_array, axis=0)
        mean_target_pdp = np.mean(target_pdp_array, axis=0)
        delay_taps = np.arange(len(mean_pred_pdp))
        
        axes[1, 0].plot(delay_taps, mean_pred_pdp, 'b-', linewidth=2, label='Predicted PDP')
        axes[1, 0].plot(delay_taps, mean_target_pdp, 'r--', linewidth=2, label='Target PDP')
        axes[1, 0].set_xlabel('Delay Tap Index')
        axes[1, 0].set_ylabel('Average Power')
        axes[1, 0].set_title('Average PDP Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: PDP difference vs delay tap
        mean_pdp_diff = np.mean(pdp_diff_array, axis=0)
        std_pdp_diff = np.std(pdp_diff_array, axis=0)
        
        axes[1, 1].plot(delay_taps, mean_pdp_diff, 'g-', linewidth=2, label='Mean Difference')
        axes[1, 1].fill_between(delay_taps, 
                               mean_pdp_diff - std_pdp_diff, 
                               mean_pdp_diff + std_pdp_diff, 
                               alpha=0.3, color='green', label='±1 Std Dev')
        axes[1, 1].set_xlabel('Delay Tap Index')
        axes[1, 1].set_ylabel('PDP Absolute Difference')
        axes[1, 1].set_title('PDP Difference vs Delay Tap')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plot_path = plots_dir / 'pdp_difference_pdf.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PDP difference PDF plot saved: {plot_path}")
        
        # Log detailed statistics
        logger.info(f"PDP Difference Statistics:")
        logger.info(f"  Total samples: {len(pdp_diff_flat)}")
        logger.info(f"  Mean difference: {mean_diff:.6f}")
        logger.info(f"  Std deviation: {std_diff:.6f}")
        logger.info(f"  Median difference: {median_diff:.6f}")
        logger.info(f"  Min difference: {np.min(pdp_diff_flat):.6f}")
        logger.info(f"  Max difference: {np.max(pdp_diff_flat):.6f}")
        for p in percentiles:
            p_value = np.percentile(pdp_diff_flat, p)
            logger.info(f"  {p}th percentile: {p_value:.6f}")
    
    def _evaluate_model(self) -> Dict[str, float]:
        """Perform CSI inference on test data using TrainingInterface"""
        logger.info("Performing CSI inference on test data...")
        
        try:
            self.model.eval()
            predictions = []
            
            # Process in batches to avoid memory issues (use config setting)
            batch_size = self.config['testing']['batch_size']
            num_samples = len(self.ue_positions)
            
            logger.info(f"Processing {num_samples} samples in batches of {batch_size}")
            
            # Initialize progress monitor
            progress_monitor = TestingProgressMonitor(num_samples, batch_size)
            progress_monitor.start_testing()
            
            with torch.no_grad():
                for i in range(0, num_samples, batch_size):
                    end_idx = min(i + batch_size, num_samples)
                    actual_batch_size = end_idx - i
                    batch_idx = i // batch_size
                    
                    # Start batch monitoring
                    progress_monitor.start_batch(batch_idx, actual_batch_size)
                    
                    batch_ue_pos = self.ue_positions[i:end_idx]
                    batch_bs_pos = self.bs_position.expand(actual_batch_size, -1)
                    batch_antenna_idx = self.antenna_indices.expand(actual_batch_size, -1)
                    
                    try:
                        # Use TrainingInterface forward pass for CSI prediction
                        outputs = self.model(
                            ue_positions=batch_ue_pos,
                            bs_position=batch_bs_pos,
                            antenna_indices=batch_antenna_idx
                        )
                        batch_predictions = outputs['csi_predictions']
                        
                        # Store predictions (move to CPU to save GPU memory)
                        predictions.append(batch_predictions.cpu())
                        
                        # Update progress monitor
                        progress_monitor.update_batch_progress(batch_idx, actual_batch_size)
                        
                        # Log progress (less frequent now due to progress monitor)
                        if (batch_idx + 1) % 5 == 0 or end_idx == num_samples:
                            logger.info(f"Processed batch {batch_idx + 1}/{(num_samples + batch_size - 1)//batch_size}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error in batch {batch_idx}: {e}")
                        raise
            
            # End progress monitoring
            progress_monitor.end_testing()
            
            # Store progress monitor for later use
            self.progress_monitor = progress_monitor
            
            if not predictions:
                raise RuntimeError("No successful predictions made")
            
            # Concatenate all predictions
            self.predictions = torch.cat(predictions, dim=0)
            
            # Log prediction statistics
            logger.info(f"CSI inference completed successfully")
            logger.info(f"Prediction tensor shape: {self.predictions.shape}")
            logger.info(f"Prediction tensor dtype: {self.predictions.dtype}")
            logger.info(f"Target tensor shape: {self.csi_target.shape}")
            logger.info(f"Target tensor dtype: {self.csi_target.dtype}")
            
            # Store predictions for later analysis
            self._store_predictions()
            
            # Return basic statistics (no loss calculation)
            return {
                'num_samples': num_samples,
                'prediction_shape': list(self.predictions.shape),
                'target_shape': list(self.csi_target.shape)
            }
            
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: CSI inference failed: {e}")
            raise
    
    def _store_predictions(self):
        """Store predicted CSI results for later analysis"""
        logger.info("Storing predicted CSI results...")
        
        try:
            # Create predictions directory
            predictions_dir = Path(self.config['output']['testing']['predictions_dir'])
            predictions_dir.mkdir(parents=True, exist_ok=True)
            
            # Save raw predictions and targets
            predictions_path = predictions_dir / 'csi_predictions_raw.npz'
            
            # Convert to numpy for storage
            predictions_np = self.predictions.numpy()
            targets_np = self.csi_target.cpu().numpy()
            ue_positions_np = self.ue_positions.cpu().numpy()
            bs_position_np = self.bs_position.cpu().numpy()
            antenna_indices_np = self.antenna_indices.cpu().numpy()
            
            # Save comprehensive prediction data
            np.savez_compressed(
                predictions_path,
                # CSI data
                csi_predictions=predictions_np,
                csi_targets=targets_np,
                # Position and antenna data
                ue_positions=ue_positions_np,
                bs_position=bs_position_np,
                antenna_indices=antenna_indices_np,
                # Metadata
                prediction_timestamp=datetime.now().isoformat(),
                model_path=str(self.model_path),
                data_path=str(self.data_path),
                checkpoint_info=self.checkpoint_info
            )
            
            logger.info(f"✅ CSI predictions stored successfully")
            logger.info(f"   📁 File: {predictions_path}")
            logger.info(f"   📊 Predictions shape: {predictions_np.shape}")
            logger.info(f"   📊 Targets shape: {targets_np.shape}")
            logger.info(f"   📊 UE positions shape: {ue_positions_np.shape}")
            logger.info(f"   💾 File size: {predictions_path.stat().st_size / 1024 / 1024:.1f} MB")
            
            # Also save a summary JSON file with metadata
            summary_path = predictions_dir / 'csi_predictions_summary.json'
            summary_data = {
                'prediction_timestamp': datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'data_path': str(self.data_path),
                'checkpoint_info': self.checkpoint_info,
                'data_shapes': {
                    'csi_predictions': list(predictions_np.shape),
                    'csi_targets': list(targets_np.shape),
                    'ue_positions': list(ue_positions_np.shape),
                    'bs_position': list(bs_position_np.shape),
                    'antenna_indices': list(antenna_indices_np.shape)
                },
                'data_types': {
                    'csi_predictions': str(predictions_np.dtype),
                    'csi_targets': str(targets_np.dtype),
                    'ue_positions': str(ue_positions_np.shape),
                    'bs_position': str(bs_position_np.dtype),
                    'antenna_indices': str(antenna_indices_np.dtype)
                },
                'simulation_parameters': self.sim_params
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            logger.info(f"   📋 Summary saved: {summary_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store predictions: {e}")
            raise
    
    def analyze_csi_predictions(self):
        """Analyze and compare predicted CSI with target CSI"""
        logger.info("🔍 Analyzing CSI predictions vs targets...")
        
        if not hasattr(self, 'predictions') or self.predictions is None:
            logger.error("❌ No predictions available for analysis. Run inference first.")
            raise ValueError("Predictions not available")
        
        try:
            # Calculate comprehensive metrics
            metrics = self._calculate_csi_metrics()
            
            # Create detailed visualizations
            self._create_csi_analysis_plots()
            
            # Save analysis results
            self._save_csi_analysis(metrics)
            
            logger.info("✅ CSI analysis completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ CSI analysis failed: {e}")
            raise
    
    def _calculate_csi_metrics(self) -> Dict[str, float]:
        """Calculate detailed CSI comparison metrics"""
        logger.info("Calculating CSI comparison metrics...")
        
        pred_np = self.predictions.numpy()
        target_np = self.csi_target.cpu().numpy()
        
        # Handle UE antenna dimension mismatch between predictions and targets
        if pred_np.shape[2] != target_np.shape[2]:
            logger.info(f"🔧 UE antenna dimension mismatch detected in CSI analysis:")
            logger.info(f"   Predictions UE antennas: {pred_np.shape[2]}")
            logger.info(f"   Targets UE antennas: {target_np.shape[2]}")
            
            if target_np.shape[2] > pred_np.shape[2]:
                # Use only the first UE antenna from targets to match predictions
                target_np = target_np[:, :, :pred_np.shape[2], :]
                logger.info(f"   ✅ Adjusted targets to use first {pred_np.shape[2]} UE antenna(s)")
                logger.info(f"   New target shape: {target_np.shape}")
            else:
                raise ValueError(f"Cannot match dimensions: targets have {target_np.shape[2]} UE antennas but predictions have {pred_np.shape[2]}")
        
        # Calculate magnitude and phase errors
        pred_mag = np.abs(pred_np)
        target_mag = np.abs(target_np)
        mag_error = np.abs(pred_mag - target_mag)
        
        pred_phase = np.angle(pred_np)
        target_phase = np.angle(target_np)
        phase_diff = pred_phase - target_phase
        phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrap to [-π, π]
        phase_error = np.abs(phase_diff)
        
        # Calculate MSE for comparison (optional)
        if pred_np.dtype == np.complex64 or pred_np.dtype == np.complex128:
            mse_real = np.mean((pred_np.real - target_np.real)**2)
            mse_imag = np.mean((pred_np.imag - target_np.imag)**2)
            mse_total = mse_real + mse_imag
        else:
            mse_total = np.mean((pred_np - target_np)**2)
        
        # Calculate correlation coefficients
        pred_flat = pred_np.flatten()
        target_flat = target_np.flatten()
        
        if pred_flat.dtype == np.complex64 or pred_flat.dtype == np.complex128:
            # For complex data, calculate correlation for magnitude and phase separately
            mag_corr = np.corrcoef(np.abs(pred_flat), np.abs(target_flat))[0, 1]
            phase_corr = np.corrcoef(np.angle(pred_flat), np.angle(target_flat))[0, 1]
        else:
            mag_corr = np.corrcoef(pred_flat, target_flat)[0, 1]
            phase_corr = 0.0
        
        # Calculate comprehensive metrics
        metrics = {
            # Basic errors
            'mse_total': float(mse_total),
            'rmse_total': float(np.sqrt(mse_total)),
            
            # Magnitude metrics
            'magnitude_mae': float(np.mean(mag_error)),
            'magnitude_mse': float(np.mean(mag_error**2)),
            'magnitude_rmse': float(np.sqrt(np.mean(mag_error**2))),
            'magnitude_max_error': float(np.max(mag_error)),
            'magnitude_correlation': float(mag_corr) if not np.isnan(mag_corr) else 0.0,
            
            # Phase metrics  
            'phase_mae': float(np.mean(phase_error)),
            'phase_mse': float(np.mean(phase_error**2)),
            'phase_rmse': float(np.sqrt(np.mean(phase_error**2))),
            'phase_max_error': float(np.max(phase_error)),
            'phase_correlation': float(phase_corr) if not np.isnan(phase_corr) else 0.0,
            
            # Percentile metrics
            'magnitude_error_50th': float(np.percentile(mag_error, 50)),
            'magnitude_error_90th': float(np.percentile(mag_error, 90)),
            'magnitude_error_95th': float(np.percentile(mag_error, 95)),
            'magnitude_error_99th': float(np.percentile(mag_error, 99)),
            
            'phase_error_50th': float(np.percentile(phase_error, 50)),
            'phase_error_90th': float(np.percentile(phase_error, 90)),
            'phase_error_95th': float(np.percentile(phase_error, 95)),
            'phase_error_99th': float(np.percentile(phase_error, 99)),
            
            # Data statistics
            'num_samples': int(pred_np.shape[0]),
            'prediction_shape': list(pred_np.shape),
            'target_shape': list(target_np.shape)
        }
        
        # Log key metrics
        logger.info("📊 Key CSI Comparison Metrics:")
        logger.info(f"   • Total RMSE: {metrics['rmse_total']:.6f}")
        logger.info(f"   • Magnitude RMSE: {metrics['magnitude_rmse']:.6f}")
        logger.info(f"   • Phase RMSE: {metrics['phase_rmse']:.6f}")
        logger.info(f"   • Magnitude Correlation: {metrics['magnitude_correlation']:.4f}")
        logger.info(f"   • Phase Correlation: {metrics['phase_correlation']:.4f}")
        
        return metrics
    
    def _create_csi_analysis_plots(self):
        """Create detailed CSI analysis plots"""
        logger.info("Creating CSI analysis plots...")
        
        # Use existing visualization methods
        self._visualize_results()
        
        logger.info("✅ CSI analysis plots created")
    
    def _save_csi_analysis(self, metrics: Dict[str, float]):
        """Save CSI analysis results"""
        logger.info("Saving CSI analysis results...")
        
        try:
            # Save analysis results
            results_dir = Path(self.config['output']['testing']['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)
            
            analysis_results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'data_path': str(self.data_path),
                'checkpoint_info': self.checkpoint_info,
                'csi_comparison_metrics': metrics,
                'simulation_parameters': self.sim_params
            }
            
            analysis_path = results_dir / 'csi_analysis_results.json'
            with open(analysis_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"✅ CSI analysis results saved: {analysis_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save CSI analysis: {e}")
            raise
    
    def inference_only(self):
        """Perform only CSI inference without analysis"""
        logger.info("🧪 PRISM NETWORK CSI INFERENCE ONLY")
        logger.info("=" * 80)
        logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"🔧 Configuration file: {self.config_path}")
        logger.info(f"🤖 Model path: {self.model_path}")
        logger.info(f"📊 Data path: {self.data_path}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Phase 1: CSI Inference Only
            logger.info("🔄 INFERENCE PHASE: CSI PREDICTION")
            logger.info("📝 Description: Running CSI prediction inference only")
            logger.info("-" * 60)
            inference_stats = self._evaluate_model()
            
            # Log inference statistics
            logger.info("📊 CSI INFERENCE STATISTICS:")
            logger.info("-" * 40)
            for stat_name, stat_value in inference_stats.items():
                logger.info(f"   • {stat_name}: {stat_value}")
            logger.info("-" * 40)
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Log completion
            logger.info("=" * 80)
            logger.info("✅ PRISM NETWORK CSI INFERENCE COMPLETED")
            logger.info("=" * 80)
            logger.info(f"⏱️  Total inference time: {total_time:.2f} seconds")
            logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("")
            logger.info("📋 INFERENCE SUMMARY:")
            logger.info(f"   • Total test samples: {len(self.ue_positions)}")
            logger.info(f"   • Model parameters: {sum(p.numel() for p in self.prism_network.parameters()):,}")
            logger.info(f"   • Predictions saved for later analysis")
            
            # Show performance summary from progress monitor
            if hasattr(self, 'progress_monitor') and self.progress_monitor is not None:
                perf_summary = self.progress_monitor.get_performance_summary()
                if isinstance(perf_summary, dict):
                    logger.info(f"   • Average batch time: {perf_summary['avg_batch_time']:.1f}s")
                    logger.info(f"   • Average throughput: {perf_summary['avg_throughput']:.1f} samples/s")
                    logger.info(f"   • Fastest batch: {perf_summary['fastest_batch']:.1f}s")
                    logger.info(f"   • Slowest batch: {perf_summary['slowest_batch']:.1f}s")
                    if perf_summary['gpu_utilization_avg'] > 0:
                        logger.info(f"   • Average GPU utilization: {perf_summary['gpu_utilization_avg']:.1f}%")
            
            for stat_name, stat_value in inference_stats.items():
                logger.info(f"   • {stat_name}: {stat_value}")
            logger.info("=" * 80)
            
            return inference_stats
            
        except Exception as e:
            logger.error(f"❌ CSI inference failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _visualize_results(self):
        """Create visualization plots"""
        logger.info("Creating visualizations...")
        
        try:
            # Create plots directory
            plots_dir = Path(self.config['output']['testing']['plots_dir'])
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot 1: CSI magnitude comparison
            self._plot_csi_magnitude_comparison(plots_dir)
            
            # Plot 2: CSI phase comparison
            self._plot_csi_phase_comparison(plots_dir)
            
            # Plot 3: Error distribution
            self._plot_error_distribution(plots_dir)
            
            # Plot 4: Spatial performance
            self._plot_spatial_performance(plots_dir)
            
            # Plot 5: Subcarrier performance
            self._plot_subcarrier_performance(plots_dir)
            
            # Plot 6: CDF of magnitude and phase errors
            self._plot_error_cdf(plots_dir)
            
            # Plot 7: PDP analysis and CDF
            self._plot_pdp_analysis(plots_dir)
            
            logger.info("✅ All visualizations created successfully")
            
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: Visualization creation failed: {e}")
            raise
    
    def _save_results(self, metrics: Dict[str, float]):
        """Save test results and metrics"""
        logger.info("Saving test results...")
        
        try:
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                """Convert numpy types to Python native types for JSON serialization"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            # Save metrics
            results = {
                'test_timestamp': datetime.now().isoformat(),
                'model_path': self.model_path,
                'data_path': self.data_path,
                'checkpoint_info': convert_numpy_types(self.checkpoint_info),
                'metrics': convert_numpy_types(metrics),
                'simulation_parameters': convert_numpy_types(self.sim_params)
            }
            
            # Save results
            results_dir = Path(self.config['output']['testing']['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)
            results_path = results_dir / 'test_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save predictions
            predictions_dir = Path(self.config['output']['testing']['predictions_dir'])
            predictions_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = predictions_dir / 'predictions.npz'
            np.savez_compressed(
                predictions_path,
                ue_positions=self.ue_positions.cpu().numpy(),
                csi_predictions=self.predictions.numpy(),
                csi_targets=self.csi_target.cpu().numpy(),
                bs_position=self.bs_position.cpu().numpy()
            )
            
            logger.info(f"Results saved to {self.output_dir}")
            logger.info(f"  - Metrics: {results_path}")
            logger.info(f"  - Predictions: {predictions_path}")
            
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: Failed to save results: {e}")
            raise
    
    def test(self):
        """Main testing function"""
        logger.info("🧪 PRISM NETWORK TESTING STARTED")
        logger.info("=" * 80)
        logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"🔧 Configuration file: {self.config_path}")
        logger.info(f"🤖 Model path: {self.model_path}")
        logger.info(f"📊 Data path: {self.data_path}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Phase 1: Model Loading
            logger.info("🔄 TESTING PHASE: MODEL LOADING")
            logger.info("📝 Description: Loading trained model and preparing for testing")
            logger.info("-" * 60)
            logger.info("✅ Model loaded successfully")
            
            # Phase 2: Data Loading
            logger.info("🔄 TESTING PHASE: DATA LOADING")
            logger.info("📝 Description: Loading and preparing test data")
            logger.info("-" * 60)
            logger.info("✅ Test data loaded successfully")
            
            # Phase 3: CSI Inference
            logger.info("🔄 TESTING PHASE: CSI INFERENCE")
            logger.info("📝 Description: Running CSI prediction inference")
            logger.info("-" * 60)
            inference_stats = self._evaluate_model()
            
            # Log inference statistics
            logger.info("📊 CSI INFERENCE STATISTICS:")
            logger.info("-" * 40)
            for stat_name, stat_value in inference_stats.items():
                logger.info(f"   • {stat_name}: {stat_value}")
            logger.info("-" * 40)
            logger.info("✅ CSI inference completed successfully")
            
            # Phase 4: CSI Analysis (Optional - can be run separately)
            logger.info("🔄 TESTING PHASE: CSI ANALYSIS")
            logger.info("📝 Description: Analyzing predicted CSI vs target CSI")
            logger.info("-" * 60)
            analysis_metrics = self.analyze_csi_predictions()
            
            # Log analysis metrics
            logger.info("📊 CSI ANALYSIS METRICS:")
            logger.info("-" * 40)
            for metric_name, metric_value in analysis_metrics.items():
                if isinstance(metric_value, float):
                    logger.info(f"   • {metric_name}: {metric_value:.6f}")
                else:
                    logger.info(f"   • {metric_name}: {metric_value}")
            logger.info("-" * 40)
            logger.info("✅ CSI analysis completed successfully")
            
            # Phase 5: Results Saving
            logger.info("🔄 TESTING PHASE: RESULTS SAVING")
            logger.info("📝 Description: Saving final test results and summary")
            logger.info("-" * 60)
            # Combine inference stats and analysis metrics
            combined_metrics = {**inference_stats, **analysis_metrics}
            self._save_results(combined_metrics)
            logger.info("✅ Results saved successfully")
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Log testing completion
            logger.info("=" * 80)
            logger.info("✅ PRISM NETWORK TESTING COMPLETED")
            logger.info("=" * 80)
            logger.info(f"⏱️  Total testing time: {total_time:.2f} seconds")
            logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("")
            logger.info("📋 TESTING SUMMARY:")
            logger.info(f"   • Total test samples: {len(self.ue_positions)}")
            logger.info(f"   • Model parameters: {sum(p.numel() for p in self.prism_network.parameters()):,}")
            logger.info(f"   • Best loss from checkpoint: {self.checkpoint_info.get('best_loss', 'N/A')}")
            
            # Show performance summary from progress monitor
            if hasattr(self, 'progress_monitor') and self.progress_monitor is not None:
                perf_summary = self.progress_monitor.get_performance_summary()
                if isinstance(perf_summary, dict):
                    logger.info(f"   • Average batch time: {perf_summary['avg_batch_time']:.1f}s")
                    logger.info(f"   • Average throughput: {perf_summary['avg_throughput']:.1f} samples/s")
                    logger.info(f"   • Fastest batch: {perf_summary['fastest_batch']:.1f}s")
                    logger.info(f"   • Slowest batch: {perf_summary['slowest_batch']:.1f}s")
                    if perf_summary['gpu_utilization_avg'] > 0:
                        logger.info(f"   • Average GPU utilization: {perf_summary['gpu_utilization_avg']:.1f}%")
            
            # Show key metrics from analysis
            if 'rmse_total' in combined_metrics:
                logger.info(f"   • Total RMSE: {combined_metrics['rmse_total']:.6f}")
            if 'magnitude_rmse' in combined_metrics:
                logger.info(f"   • Magnitude RMSE: {combined_metrics['magnitude_rmse']:.6f}")
            if 'phase_rmse' in combined_metrics:
                logger.info(f"   • Phase RMSE: {combined_metrics['phase_rmse']:.6f}")
            if 'magnitude_correlation' in combined_metrics:
                logger.info(f"   • Magnitude Correlation: {combined_metrics['magnitude_correlation']:.4f}")
            if 'phase_correlation' in combined_metrics:
                logger.info(f"   • Phase Correlation: {combined_metrics['phase_correlation']:.4f}")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Testing failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test Prism Network')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file (required)')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID to use (e.g., 0, 1, 2). If not specified, will auto-select based on available memory')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model checkpoint (optional, will read from config if not provided)')
    parser.add_argument('--inference-only', action='store_true',
                       help='Perform only CSI inference without analysis (faster, saves predictions for later analysis)')
    
    args = parser.parse_args()
    
    try:
        # Create tester and start testing
        logger.info("🚀 Initializing Prism Tester...")
        tester = PrismTester(args.config, args.model, None, None, args.gpu)
        logger.info("✅ Prism Tester initialized successfully")
        
        if args.inference_only:
            logger.info("🔍 Starting CSI inference only...")
            tester.inference_only()
            logger.info("✅ CSI inference completed successfully")
            logger.info("💡 Tip: Run without --inference-only flag to perform full analysis")
        else:
            logger.info("🧪 Starting full testing process...")
        tester.test()
        logger.info("✅ Full testing completed successfully")
        
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"❌ FATAL ERROR - Model file issue: {e}")
        logger.error("Please check:")
        logger.error("  1. Model file exists at the specified path")
        logger.error("  2. File has proper read permissions")
        logger.error("  3. Training has been completed successfully")
        sys.exit(1)
        
    except (ValueError, RuntimeError) as e:
        logger.error(f"❌ FATAL ERROR - Model loading failed: {e}")
        logger.error("Please check:")
        logger.error("  1. Model checkpoint is not corrupted")
        logger.error("  2. PyTorch version compatibility")
        logger.error("  3. Model architecture matches training configuration")
        logger.error("  4. Sufficient GPU/CPU memory available")
        sys.exit(1)
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Testing interrupted by user (Ctrl+C)")
        sys.exit(130)  # Standard exit code for SIGINT
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR - Unexpected error during testing: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == '__main__':
    main()
