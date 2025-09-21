#!/usr/bin/env python3
"""
Generic Prism Network Testing Script

This script tests the trained Prism neural network for electromagnetic ray tracing.
It can load different datasets and configurations based on the provided config file.

Features:
- Generic data loading for different dataset formats (Sionna, PolyU, Chrissy)
- Automatic dataset format detection and adaptation
- Modern configuration loading with template processing
- Comprehensive testing metrics (MSE, NMSE, CSI accuracy, etc.)
- Model loading from checkpoints
- Real-time progress monitoring
- Visualization capabilities
- Comprehensive logging and error handling
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
from prism.config_loader import ModernConfigLoader
from prism.training_interface import PrismTrainingInterface
from prism.loss.loss_function import LossFunction
from prism.loss.ss_loss import SSLoss

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
        print(f"\r🔄 Batch {self.current_batch:3d}/{self.total_batches:3d} "
              f"({progress_percent:5.1f}%) - Processing {batch_size} samples...", end="", flush=True)
    
    def end_batch(self, batch_idx: int):
        """End batch processing and collect metrics"""
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        
        # Collect GPU metrics if available
        if GPU_AVAILABLE and torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    self.gpu_utilization_history.append(gpu.load * 100)
                    self.memory_usage_history.append(gpu.memoryUtil * 100)
            except:
                pass  # Ignore GPU monitoring errors
        
        # Calculate statistics
        avg_batch_time = sum(self.batch_times) / len(self.batch_times)
        remaining_batches = self.total_batches - self.current_batch
        eta_seconds = remaining_batches * avg_batch_time
        
        progress_percent = (self.processed_samples / self.total_samples) * 100
        
        print(f"\r✅ Batch {self.current_batch:3d}/{self.total_batches:3d} "
              f"({progress_percent:5.1f}%) - {batch_time:.2f}s "
              f"(avg: {avg_batch_time:.2f}s, ETA: {eta_seconds/60:.1f}m)", end="", flush=True)
    
    def end_testing(self):
        """Complete testing and show summary"""
        total_time = time.time() - self.start_time
        avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        
        print(f"\n\n🎉 Testing completed successfully!")
        print(f"{'='*80}")
        print(f"📊 Testing Summary:")
        print(f"   • Total samples processed: {self.processed_samples:,}")
        print(f"   • Total time: {total_time:.1f}s ({total_time/60:.2f}m)")
        print(f"   • Average batch time: {avg_batch_time:.2f}s")
        print(f"   • Throughput: {self.processed_samples/total_time:.1f} samples/s")
        
        if self.gpu_utilization_history:
            avg_gpu_util = sum(self.gpu_utilization_history) / len(self.gpu_utilization_history)
            avg_mem_util = sum(self.memory_usage_history) / len(self.memory_usage_history)
            print(f"   • Average GPU utilization: {avg_gpu_util:.1f}%")
            print(f"   • Average GPU memory usage: {avg_mem_util:.1f}%")
        
        print(f"{'='*80}")


class PrismTester:
    """Generic tester class for Prism network using modern configuration
    
    This tester can handle different dataset formats and configurations:
    - Sionna: Ray tracing simulation data
    - PolyU: WiFi measurement data with BS position padding
    - Chrissy: Custom dataset format
    """
    
    def __init__(self, config_path: str, model_path: str = None, data_path: str = None, output_dir: str = None, gpu_id: int = None):
        """Initialize tester with configuration and optional model/data/output paths"""
        self.config_path = config_path
        self.gpu_id = gpu_id  # Store GPU ID for device setup
        
        # Load configuration first using ModernConfigLoader to process template variables
        try:
            self.config_loader = ModernConfigLoader(config_path)
            logger.info(f"Configuration loaded from: {config_path}")
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: Failed to load configuration from {config_path}")
            logger.error(f"   Error details: {str(e)}")
            raise
        
        # Note: Setup logging after output_dir is set
        
        # Set model path, data path and output directory from config if not provided
        try:
            # Use default model path from output configuration
            output_paths = self.config_loader.get_output_paths()
            default_model_path = os.path.join(output_paths['models_dir'], 'final_model.pt')
            self.model_path = model_path or default_model_path
            
            # Construct dataset path from new configuration structure
            data_config = self.config_loader.get_data_loader_config()
            dataset_path = data_config['dataset_path']
            self.data_path = data_path or dataset_path
            self.train_ratio = data_config['train_ratio']
            self.random_seed = data_config['random_seed']
            logger.info(f"Using dataset: {self.data_path}")
            logger.info(f"Train ratio: {self.train_ratio}, Random seed: {self.random_seed}")
                
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: Failed to extract configuration: {e}")
            raise
        
        # Build output directory from config base_dir + testing
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            output_paths = self.config_loader.get_output_paths()
            base_dir = output_paths.get('base_dir', 'results')  # Get base_dir from config
            self.output_dir = Path(base_dir) / 'testing'
        
        # Clean previous testing results and create directories first
        self._clean_and_create_testing_directories()
        
        # Setup proper logging after directories are created
        try:
            self._setup_logging()
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: Failed to setup logging")
            logger.error(f"   Error details: {str(e)}")
            raise
        
        # Initialize device and model
        self.device = self.config_loader.get_device()
        self.model = None
        self.checkpoint_info = {}
        
        # Load model and data
        self._load_model()
        self._load_data()
        
        # Initialize spatial spectrum loss for testing
        self._init_spatial_spectrum_loss()
    
        # Initialize PDP loss for testing
        self._init_pdp_loss()
    
    def _init_spatial_spectrum_loss(self):
        """Initialize spatial spectrum loss for testing metrics with high resolution"""
        self.ss_loss = None
        
        try:
            from prism.loss.ss_loss import SSLoss
            config = self.config_loader._processed_config.copy()  # Make a copy to avoid modifying original
            
            # Always enable spatial spectrum loss for testing visualization
            ss_config = config.get('training', {}).get('loss', {}).get('spatial_spectrum_loss', {})
            
            # Force enable spatial spectrum loss for testing
            if not ss_config.get('enabled', False):
                logger.info("🔧 Force enabling spatial spectrum loss for testing visualization...")
                config['training']['loss']['spatial_spectrum_loss']['enabled'] = True
            
            # Set high-resolution configuration for testing
            original_theta_range = ss_config.get('theta_range', [0, 5.0, 90.0])
            original_phi_range = ss_config.get('phi_range', [0.0, 10.0, 360.0])
            original_theta_points = int((original_theta_range[2] - original_theta_range[0]) / original_theta_range[1]) + 1
            original_phi_points = int((original_phi_range[2] - original_phi_range[0]) / original_phi_range[1]) + 1
            
            # 🔧 Override angle resolution for high-resolution testing visualization
            new_theta_range = [0.0, 1.0, 90.0]   # 1° resolution
            new_phi_range = [0.0, 2.0, 360.0]    # 2° resolution
            new_theta_points = int((new_theta_range[2] - new_theta_range[0]) / new_theta_range[1]) + 1
            new_phi_points = int((new_phi_range[2] - new_phi_range[0]) / new_phi_range[1]) + 1
            
            config['training']['loss']['spatial_spectrum_loss']['theta_range'] = new_theta_range
            config['training']['loss']['spatial_spectrum_loss']['phi_range'] = new_phi_range
            
            logger.info("🔧 Enhancing spatial spectrum resolution for testing:")
            logger.info(f"   📐 Theta (elevation): {original_theta_range[1]}° → {new_theta_range[1]}° step ({original_theta_points} → {new_theta_points} points)")
            logger.info(f"   📐 Phi (azimuth): {original_phi_range[1]}° → {new_phi_range[1]}° step ({original_phi_points} → {new_phi_points} points)")
            logger.info(f"   📊 Total grid: {original_theta_points}×{original_phi_points}={original_theta_points*original_phi_points} → {new_theta_points}×{new_phi_points}={new_theta_points*new_phi_points} points ({(new_theta_points*new_phi_points)/(original_theta_points*original_phi_points):.1f}x improvement)")
            
            self.ss_loss = SSLoss(config).to(self.device)
            logger.info("✅ Spatial spectrum loss initialized for testing with enhanced resolution")
                
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize spatial spectrum loss: {e}")
            self.ss_loss = None
    
    def _init_pdp_loss(self):
        """Initialize PDP loss for testing metrics - force enable for testing"""
        self.pdp_loss = None
        
        try:
            from prism.loss.pdp_loss import PDPLoss
            config = self.config_loader._processed_config.copy()  # Make a copy to avoid modifying original
            
            # Force enable PDP loss for testing
            pdp_config = config.get('training', {}).get('loss', {}).get('pdp_loss', {})
            if not pdp_config.get('enabled', False):
                logger.info("🔧 Force enabling PDP loss for testing metrics...")
                config['training']['loss']['pdp_loss']['enabled'] = True
            
            # Get OFDM configuration for FFT size
            ofdm_config = config.get('base_station', {}).get('ofdm', {})
            fft_size = ofdm_config.get('fft_size', 2046)
            
            # Initialize PDP loss with configuration
            self.pdp_loss = PDPLoss(
                loss_type=pdp_config.get('type', 'hybrid'),
                fft_size=fft_size,
                normalize_pdp=pdp_config.get('normalize_pdp', True),
                mse_weight=pdp_config.get('mse_weight', 0.7),
                delay_weight=pdp_config.get('delay_weight', 0.3)
            ).to(self.device)
            
            logger.info("✅ PDP loss initialized for testing")
                
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize PDP loss: {e}")
            self.pdp_loss = None
    
    def _setup_logging(self):
        """Setup logging with proper file path from config for testing"""
        # Create testing-specific log configuration from base config
        output_paths = self.config_loader.get_output_paths()
        
        # Create testing log directory and file
        testing_log_dir = self.output_dir / 'logs'
        testing_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = testing_log_dir / 'testing.log'
        
        log_level_str = 'INFO'
        
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
        
        logger.info(f"Testing logging configured: {log_file}")
    
    def _clean_and_create_testing_directories(self):
        """Clean previous testing results and create all necessary testing directories"""
        import shutil
        
        # Clean previous testing results if directory exists
        if self.output_dir.exists():
            logger.info(f"🧹 Cleaning previous testing results from: {self.output_dir}")
            try:
                shutil.rmtree(self.output_dir)
                logger.info("✅ Previous testing results cleaned successfully")
            except Exception as e:
                logger.warning(f"⚠️  Failed to clean previous results: {e}")
        
        # Create all necessary testing directories
        directories = [
            self.output_dir / 'logs',
            self.output_dir / 'plots',
            self.output_dir / 'metrics',
            self.output_dir / 'predictions'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Testing directory created: {directory}")
        
        logger.info(f"🗂️  Testing directories created under: {self.output_dir}")
    
    def _load_model(self):
        """Load trained Prism network model from TrainingInterface checkpoint"""
        logger.info(f"Loading TrainingInterface model from {self.model_path}")
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            error_msg = f"Model file not found: {self.model_path}"
            logger.error(f"❌ FATAL ERROR: {error_msg}")
            logger.error("Please ensure the model has been trained and the path is correct.")
            logger.error("Available model locations to check:")
            logger.error("  - results/sionna/training/models/final_model.pt")
            logger.error("  - results/sionna/training/checkpoints/")
            raise FileNotFoundError(error_msg)
        
        try:
            # Load using modern configuration and training interface
            self._load_training_interface_checkpoint()
                
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: Failed to load model: {e}")
            logger.error(f"Model path: {self.model_path}")
            logger.error(f"Error type: {type(e).__name__}")
            if hasattr(e, '__traceback__'):
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _load_training_interface_checkpoint(self):
        """Load TrainingInterface checkpoint using modern configuration"""
        try:
            # Create PrismNetwork using modern config loader
            network_kwargs = self.config_loader.get_prism_network_kwargs()
            self.prism_network = PrismNetwork(**network_kwargs).to(self.device)
            
            # Create training interface config
            data_config = self.config_loader.get_data_loader_config()
            config_dict = {
                'system': {'device': str(self.device)},
                'training': self.config_loader.training.__dict__,
                'user_equipment': {
                    'target_antenna_index': data_config['ue_antenna_index']
                },
                'input': {
                    'subcarrier_sampling': {
                        'sampling_ratio': data_config['sampling_ratio'],
                        'sampling_method': data_config['sampling_method'],
                        'antenna_consistent': data_config['antenna_consistent']
                    }
                }
            }
            
            # Create training interface
            output_paths = self.config_loader.get_output_paths()
            self.model = PrismTrainingInterface(
                prism_network=self.prism_network,
                config=config_dict,
                checkpoint_dir=output_paths['checkpoint_dir'],
                device=self.device
            )
            
            # Load checkpoint
            logger.info("Loading checkpoint file...")
            
            # Handle different checkpoint formats
            if self.model_path.endswith('.pt'):
                # This could be a simple model state dict or a full checkpoint
                # Try to load with weights_only=True first for security, fall back if needed
                try:
                    # First try the secure way
                    checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
                    logger.info("✅ Loaded checkpoint with weights_only=True (secure mode)")
                except Exception as secure_error:
                    # If that fails, try with weights_only=False for full checkpoint
                    logger.warning("⚠️  weights_only=True failed, trying full checkpoint loading...")
                    try:
                        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                        logger.info("✅ Loaded checkpoint with weights_only=False (contains metadata)")
                    except Exception as e:
                        logger.error(f"❌ Failed to load checkpoint from {self.model_path}: {e}")
                        logger.error(f"   Secure load error: {secure_error}")
                        raise
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Full checkpoint format from training interface
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.checkpoint_info = checkpoint.get('checkpoint_info', {})
                elif isinstance(checkpoint, dict):
                    # Simple state dict (probably just the PrismNetwork)
                    self.prism_network.load_state_dict(checkpoint)
                    self.checkpoint_info = {}
                else:
                    raise ValueError("Unknown checkpoint format")
            else:
                raise ValueError(f"Unsupported checkpoint format: {self.model_path}")
            
            logger.info("Model loaded successfully")
            
            # Set model to evaluation mode
            self.model.eval()
            self.prism_network.eval()
            
            # Log model information
            model_info = self.prism_network.get_network_info()
            logger.info(f"✅ Model loaded successfully:")
            logger.info(f"   Total parameters: {model_info['total_parameters']:,}")
            logger.info(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
            logger.info(f"   Ray directions: {model_info['ray_directions']}")
            logger.info(f"   Subcarriers: {model_info['num_subcarriers']}")
            
            if self.checkpoint_info:
                logger.info(f"   Checkpoint epoch: {self.checkpoint_info.get('epoch', 'Unknown')}")
                logger.info(f"   Checkpoint loss: {self.checkpoint_info.get('best_loss', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load training interface checkpoint: {e}")
            raise
    
    def _load_data(self):
        """Load and prepare testing data based on configuration"""
        logger.info("🔧 Loading testing data...")
        
        # Get dataset configuration
        data_config = self.config_loader.get_data_loader_config()
        dataset_path = data_config['dataset_path']
        
        logger.info(f"📊 Dataset configuration:")
        logger.info(f"   Path: {dataset_path}")
        
        # Check if the dataset path exists
        if not os.path.exists(dataset_path):
            logger.error(f"❌ Dataset not found: {dataset_path}")
            logger.error(f"   Please check the dataset path in your configuration file.")
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Auto-detect format and load data
        with h5py.File(dataset_path, 'r') as f:
            keys = list(f.keys())
            logger.info(f"📊 Dataset keys: {keys}")
            
            if 'data' not in keys:
                raise ValueError(f"Invalid dataset structure. Available keys: {keys}")
            
            data_keys = list(f['data'].keys())
            logger.info(f"📊 Data keys: {data_keys}")
            
            # Auto-detect data format based on available keys
            if 'ue_positions' in data_keys and 'bs_positions' in data_keys and 'csi' in data_keys:
                # All datasets now use the same key names - detect format by data dimensions
                logger.info("🔍 Detected unified format - analyzing data dimensions")
                ue_positions = torch.from_numpy(f['data/ue_positions'][:]).float()
                bs_positions_raw = torch.from_numpy(f['data/bs_positions'][:]).float()
                channel_responses = torch.from_numpy(f['data/csi'][:]).cfloat()
                
                logger.info(f"📊 Raw channel_responses shape: {channel_responses.shape}")
                logger.info(f"   Shape[1] (dim1): {channel_responses.shape[1]}")
                logger.info(f"   Shape[2] (dim2): {channel_responses.shape[2]}")
                logger.info(f"   Shape[3] (dim3): {channel_responses.shape[3]}")
                
                # Enhanced format detection based on CSI data dimensions and typical patterns
                # Check for typical subcarrier counts to help identify format
                dim1, dim2, dim3 = channel_responses.shape[1], channel_responses.shape[2], channel_responses.shape[3]
                
                # Common subcarrier counts for different datasets
                common_subcarriers = [64, 128, 256, 408, 512, 1024]
                
                # Identify which dimension is likely subcarriers
                subcarrier_dim = None
                for i, dim in enumerate([dim1, dim2, dim3], 1):
                    if dim in common_subcarriers:
                        subcarrier_dim = i
                        break
                
                logger.info(f"   Detected subcarrier dimension: {subcarrier_dim} (value: {[dim1, dim2, dim3][subcarrier_dim-1] if subcarrier_dim else 'unknown'})")
                
                # Format detection logic
                if subcarrier_dim == 3:  # Subcarriers in last dimension
                    if dim1 > dim2:
                        # PolyU format: [samples, ue_antennas, bs_antennas, subcarriers]
                        csi_data = channel_responses.permute(0, 2, 3, 1)
                        logger.info("   Detected PolyU format: [samples, ue_antennas, bs_antennas, subcarriers]")
                    else:
                        # Chrissy/Sionna format: [samples, bs_antennas, ue_antennas, subcarriers]
                        csi_data = channel_responses.permute(0, 1, 3, 2)
                        logger.info("   Detected Chrissy/Sionna format: [samples, bs_antennas, ue_antennas, subcarriers]")
                elif subcarrier_dim == 2:  # Subcarriers in middle dimension
                    # Unusual format, try to handle
                    csi_data = channel_responses.permute(0, 1, 2, 3)
                    logger.info("   Detected unusual format: subcarriers in middle dimension")
                else:
                    # Fallback to original logic
                    if dim1 > dim2:
                        csi_data = channel_responses.permute(0, 2, 3, 1)
                        logger.info("   Fallback: Detected PolyU format: [samples, ue_antennas, bs_antennas, subcarriers]")
                    else:
                        csi_data = channel_responses.permute(0, 1, 3, 2)
                        logger.info("   Fallback: Detected Chrissy/Sionna format: [samples, bs_antennas, ue_antennas, subcarriers]")
                
            else:
                raise ValueError(f"Unknown dataset format. Available keys: {data_keys}")
            
            logger.info(f"📊 Raw data shapes:")
            logger.info(f"   UE positions: {ue_positions.shape}")
            logger.info(f"   BS positions (raw): {bs_positions_raw.shape}")
            logger.info(f"   CSI data: {csi_data.shape}")
            
            # Handle BS positions using the generic expansion method
            num_samples = ue_positions.shape[0]
            bs_positions = self._expand_bs_positions_to_3d(bs_positions_raw, num_samples)
            
            # Determine antenna selection based on SSLoss configuration
            loss_config = self.config_loader.get_loss_functions_config()
            ssl_config = loss_config['spatial_spectrum_loss_config']
            array_type = ssl_config.array_type
            
            if array_type == 'ue':
                # Use UE antenna array - select all UE antennas, remove BS dimension
                logger.info("🔧 Using UE antenna array for spatial spectrum calculation")
                logger.info(f"   Original CSI data shape: {csi_data.shape}")
                if csi_data.dim() == 4:
                    # After permutation: [samples, bs_antennas, subcarriers, ue_antennas]
                    # Select all UE antennas, remove BS dimension
                    csi_data = csi_data[:, 0, :, :]  # [samples, subcarriers, ue_antennas]
                    csi_data = csi_data.permute(0, 2, 1)  # [samples, ue_antennas, subcarriers]
                    logger.info(f"   After UE selection: {csi_data.shape}")
                    num_antennas = csi_data.shape[1]  # UE antennas
                else:
                    num_antennas = csi_data.shape[1]  # Already processed
                antenna_indices = torch.arange(num_antennas).unsqueeze(0).expand(num_samples, -1).long()
            else:
                # Use BS antenna array - select specific UE antenna, remove UE dimension
                logger.info("🔧 Using BS antenna array for spatial spectrum calculation")
                ue_antenna_idx = data_config['ue_antenna_index']
                if csi_data.dim() == 4:
                    csi_data = csi_data[:, :, :, ue_antenna_idx]  # Remove UE dimension
                num_antennas = csi_data.shape[1]  # BS antennas
                antenna_indices = torch.arange(num_antennas).unsqueeze(0).expand(num_samples, -1).long()
            
            # Apply phase differential calibration
            self._apply_phase_calibration(csi_data)
            
        self._prepare_test_split(ue_positions, bs_positions, antenna_indices, csi_data)
    
    def _apply_phase_calibration(self, csi_data: torch.Tensor, reference_index: int = 0):
        """Apply phase differential calibration using configured reference subcarrier"""
        # Get phase calibration configuration
        data_config = self.config_loader.get_data_loader_config()
        phase_calibration_config = data_config.get('phase_calibration', {})
        enabled = phase_calibration_config.get('enabled', True)
        
        if not enabled:
            logger.info("🔧 Phase calibration disabled by configuration")
            return
        
        # Use configured reference subcarrier index
        if 'reference_subcarrier_index' in phase_calibration_config:
            reference_index = phase_calibration_config['reference_subcarrier_index']
        
        logger.info(f"🔧 Applying phase differential calibration using subcarrier {reference_index}...")
        logger.info(f"   Input CSI data shape: {csi_data.shape}")
        
        original_shape = csi_data.shape
        original_subcarriers = original_shape[2]
        
        # Validate reference index
        if reference_index >= original_subcarriers:
            logger.warning(f"⚠️ Reference subcarrier index {reference_index} >= total subcarriers {original_subcarriers}, using index 0")
            reference_index = 0
        
        # Extract reference subcarrier
        reference_subcarrier = csi_data[:, :, reference_index:reference_index+1]  # Keep dimension for broadcasting
        
        # Avoid division by zero
        epsilon = 1e-12
        reference_subcarrier_safe = reference_subcarrier + epsilon * torch.exp(1j * torch.angle(reference_subcarrier))
        
        # Normalize all subcarriers by the reference
        csi_data = csi_data * torch.conj(reference_subcarrier_safe) / (torch.abs(reference_subcarrier_safe) ** 2)
        
        logger.info(f"✅ Phase differential calibration applied:")
        logger.info(f"   Reference subcarrier: {reference_index}")
        logger.info(f"   Original subcarriers: {original_subcarriers}")
        logger.info(f"   Final format: [samples, antennas, subcarriers] = {csi_data.shape}")
        
    def _expand_bs_positions_to_3d(self, bs_positions_raw: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Expand BS positions from 1D to 3D coordinates"""
        logger.info(f"🔧 Processing BS positions...")
        logger.info(f"   Original shape: {bs_positions_raw.shape}")
        
        # Handle different BS position formats
        if bs_positions_raw.dim() == 1:
            # Single BS position for all samples
            if bs_positions_raw.shape[0] == 1:
                # Single value: expand to all samples
                bs_positions = bs_positions_raw.unsqueeze(0).expand(num_samples, -1)
                if bs_positions.shape[1] == 1:
                    # 1D case: convert to 3D coordinates (e.g., SSID values)
                    bs_positions_3d = torch.zeros(num_samples, 3)
                    bs_positions_3d[:, 0] = bs_positions[:, 0]  # X = original value
                    bs_positions_3d[:, 1] = 0.0  # Y = 0 (padded)
                    bs_positions_3d[:, 2] = 0.0  # Z = 0 (padded)
                    bs_positions = bs_positions_3d
            else:
                # Multiple values: one per sample
                if bs_positions_raw.shape[0] == num_samples:
                    if bs_positions_raw.shape[0] == num_samples and bs_positions_raw.shape[0] > 1:
                        # 1D case: convert to 3D coordinates
                        bs_positions = torch.zeros(num_samples, 3)
                        bs_positions[:, 0] = bs_positions_raw  # X = original values
                        bs_positions[:, 1] = 0.0  # Y = 0 (padded)
                        bs_positions[:, 2] = 0.0  # Z = 0 (padded)
                    else:
                        bs_positions = bs_positions_raw
                else:
                    raise ValueError(f"BS position count mismatch: {bs_positions_raw.shape[0]} vs {num_samples}")
        elif bs_positions_raw.dim() == 2:
            # 2D tensor: [samples, features]
            if bs_positions_raw.shape[0] == num_samples:
                if bs_positions_raw.shape[1] == 1:
                    # 1D case: convert to 3D coordinates (e.g., SSID values)
                    bs_positions = torch.zeros(num_samples, 3)
                    bs_positions[:, 0] = bs_positions_raw[:, 0]  # X = original values
                    bs_positions[:, 1] = 0.0  # Y = 0 (padded)
                    bs_positions[:, 2] = 0.0  # Z = 0 (padded)
                elif bs_positions_raw.shape[1] == 3:
                    # Already 3D coordinates
                    bs_positions = bs_positions_raw
                else:
                    raise ValueError(f"Unsupported BS position dimensions: {bs_positions_raw.shape[1]}")
            else:
                raise ValueError(f"BS position sample count mismatch: {bs_positions_raw.shape[0]} vs {num_samples}")
        else:
            raise ValueError(f"Unsupported BS position tensor dimensions: {bs_positions_raw.dim()}")
        
        logger.info(f"✅ BS position expansion completed:")
        logger.info(f"   Original shape: {bs_positions_raw.shape}")
        logger.info(f"   Expanded shape: {bs_positions.shape}")
        if bs_positions_raw.numel() > 0:
            logger.info(f"   Original range: [{bs_positions_raw.min():.1f}, {bs_positions_raw.max():.1f}]")
        logger.info(f"   X range: [{bs_positions[:, 0].min():.1f}, {bs_positions[:, 0].max():.1f}]")
        logger.info(f"   Y values: all {bs_positions[:, 1].unique().item():.1f}")
        logger.info(f"   Z values: all {bs_positions[:, 2].unique().item():.1f}")
        
        return bs_positions
        
    def _prepare_test_split(self, ue_positions: torch.Tensor, bs_positions: torch.Tensor, 
                           antenna_indices: torch.Tensor, csi_data: torch.Tensor):
        """Prepare test data split"""
        logger.info(f"✅ Data loaded successfully:")
        logger.info(f"   UE positions: {ue_positions.shape}")
        logger.info(f"   BS positions: {bs_positions.shape}")
        logger.info(f"   CSI data: {csi_data.shape}")
        logger.info(f"   Antenna indices: {antenna_indices.shape}")
        
        # Create test split (use the portion not used for training)
        num_samples = ue_positions.shape[0]
        train_size = int(num_samples * self.train_ratio)
        
        # Use same random split as training
        torch.manual_seed(self.random_seed)
        indices = torch.randperm(num_samples)
        test_indices = indices[train_size:]  # Use validation portion as test
        
        # Store test data
        self.test_ue_positions = ue_positions[test_indices]
        self.test_bs_positions = bs_positions[test_indices]
        self.test_csi_data = csi_data[test_indices]
        self.test_antenna_indices = antenna_indices[test_indices]
        
        logger.info(f"✅ Test data prepared:")
        logger.info(f"   Test samples: {len(test_indices)} (indices {train_size} to {num_samples-1})")
        logger.info(f"   Test UE positions: {self.test_ue_positions.shape}")
        logger.info(f"   Test CSI data: {self.test_csi_data.shape}")
    
    def test(self):
        """Run testing on the loaded model and data"""
        logger.info("🧪 Starting Prism network testing")
        
        # Test configuration
        test_batch_size = 1  # Use small batch size for testing
        num_test_samples = self.test_ue_positions.shape[0]
        
        # Initialize progress monitor
        monitor = TestingProgressMonitor(num_test_samples, test_batch_size)
        monitor.start_testing()
        
        # Results storage
        predictions = []
        targets = []
        
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, num_test_samples, test_batch_size):
                monitor.start_batch(i // test_batch_size, min(test_batch_size, num_test_samples - i))
                
                # Get batch data
                end_idx = min(i + test_batch_size, num_test_samples)
                batch_ue_pos = self.test_ue_positions[i:end_idx].to(self.device)
                batch_bs_pos = self.test_bs_positions[i:end_idx].to(self.device)
                batch_target_csi = self.test_csi_data[i:end_idx].to(self.device)
                batch_antenna_idx = self.test_antenna_indices[i:end_idx].to(self.device)
                
                try:
                    # Forward pass through training interface
                    outputs = self.model(
                        ue_positions=batch_ue_pos,
                        bs_positions=batch_bs_pos,
                        antenna_indices=batch_antenna_idx
                    )
                    
                    batch_predictions = outputs['csi']
                    
                    # Store results
                    predictions.append(batch_predictions.cpu())
                    targets.append(batch_target_csi.cpu())
                    
                    monitor.end_batch(i // test_batch_size)
                    
                except Exception as e:
                    logger.error(f"❌ Error in test batch {i // test_batch_size}: {e}")
                    continue
        
        monitor.end_testing()
        
        # Concatenate all results
        if predictions:
            all_predictions = torch.cat(predictions, dim=0)
            all_targets = torch.cat(targets, dim=0)
            
            # Calculate metrics
            self._calculate_metrics(all_predictions, all_targets)
            
            # Save results
            self._save_results(all_predictions, all_targets)
            
            # Generate comprehensive visualizations
            self._generate_all_visualizations(all_predictions, all_targets)
            
            logger.info("✅ Testing completed successfully")
        else:
            logger.error("❌ No successful predictions were made")
    
    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Calculate comprehensive testing metrics with proper phase wrapping handling"""
        logger.info("📊 Calculating testing metrics...")
        
        # Verify tensor shapes match (should be consistent with first subcarrier normalization)
        if predictions.shape != targets.shape:
            logger.warning(f"⚠️ Shape mismatch: pred {predictions.shape} vs target {targets.shape}")
            # This should not happen with first subcarrier normalization method
        
        # 1. Magnitude and Phase extraction
        pred_mag = torch.abs(predictions)
        target_mag = torch.abs(targets)
        pred_phase = torch.angle(predictions)  # Range: [-π, π]
        target_phase = torch.angle(targets)    # Range: [-π, π]
        
        # 2. Magnitude MSE
        magnitude_mse = torch.nn.functional.mse_loss(pred_mag, target_mag).item()
        magnitude_mae = torch.nn.functional.l1_loss(pred_mag, target_mag).item()
        
        # 3. Phase MSE with wrapping correction
        # Handle phase wrapping: find the shortest angular distance
        phase_diff = pred_phase - target_phase
        
        # Wrap phase differences to [-π, π] range
        phase_diff_wrapped = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        
        # Calculate phase MSE using wrapped differences
        phase_mse_wrapped = torch.mean(phase_diff_wrapped ** 2).item()
        phase_mae_wrapped = torch.mean(torch.abs(phase_diff_wrapped)).item()
        
        # 4. Spatial Spectrum Loss
        spatial_spectrum_loss = None
        if self.ss_loss is not None:
            try:
                # Move tensors to device and compute spatial spectrum loss
                pred_device = predictions.to(self.device)
                target_device = targets.to(self.device)
                
                with torch.no_grad():
                    spatial_spectrum_loss = self.ss_loss(pred_device, target_device).item()
                    
                logger.info(f"✅ Spatial spectrum loss computed: {spatial_spectrum_loss:.8f}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to compute spatial spectrum loss: {e}")
                spatial_spectrum_loss = None
        
        # 6. PDP Loss
        pdp_loss_value = None
        if self.pdp_loss is not None:
            try:
                # Move tensors to device
                pred_device = predictions.to(self.device)
                target_device = targets.to(self.device)
                
                # Flatten to [batch*antennas, subcarriers] for PDP computation
                batch_size, num_antennas, num_subcarriers = pred_device.shape
                pred_flat = pred_device.reshape(-1, num_subcarriers)
                target_flat = target_device.reshape(-1, num_subcarriers)
                
                with torch.no_grad():
                    # Call the public interface for PDP loss computation
                    pdp_loss_value = self.pdp_loss.compute_pdp_loss(pred_flat, target_flat).item()
                    
                logger.info(f"✅ PDP loss computed: {pdp_loss_value:.8f}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to compute PDP loss: {e}")
                pdp_loss_value = None
        
        # 7. Statistical analysis
        pred_mag_stats = {
            'min': torch.min(pred_mag).item(),
            'max': torch.max(pred_mag).item(),
            'mean': torch.mean(pred_mag).item(),
            'std': torch.std(pred_mag).item()
        }
        
        target_mag_stats = {
            'min': torch.min(target_mag).item(),
            'max': torch.max(target_mag).item(),
            'mean': torch.mean(target_mag).item(),
            'std': torch.std(target_mag).item()
        }
        
        phase_diff_stats = {
            'min': torch.min(phase_diff_wrapped).item(),
            'max': torch.max(phase_diff_wrapped).item(),
            'mean': torch.mean(phase_diff_wrapped).item(),
            'std': torch.std(phase_diff_wrapped).item()
        }
        
        # Log comprehensive metrics
        logger.info(f"📈 Comprehensive CSI Testing Metrics:")
        logger.info(f"   ═══════════════════════════════════════")
        logger.info(f"   📏 Magnitude MSE:         {magnitude_mse:.8f}")
        logger.info(f"   📏 Magnitude MAE:         {magnitude_mae:.8f}")
        logger.info(f"   ═══════════════════════════════════════")
        logger.info(f"   🌀 Phase MSE (wrapped):   {phase_mse_wrapped:.8f}")
        logger.info(f"   🌀 Phase MAE (wrapped):   {phase_mae_wrapped:.8f}")
        logger.info(f"   ═══════════════════════════════════════")
        if spatial_spectrum_loss is not None:
            logger.info(f"   🎯 Spatial Spectrum Loss: {spatial_spectrum_loss:.8f}")
        else:
            logger.info(f"   🎯 Spatial Spectrum Loss: Not computed")
        logger.info(f"   ═══════════════════════════════════════")
        if pdp_loss_value is not None:
            logger.info(f"   ⏰ PDP Loss:              {pdp_loss_value:.8f}")
        else:
            logger.info(f"   ⏰ PDP Loss:              Not computed")
        logger.info(f"   ═══════════════════════════════════════")
        logger.info(f"   📊 Prediction Magnitude Stats:")
        logger.info(f"      Min: {pred_mag_stats['min']:.6f}, Max: {pred_mag_stats['max']:.6f}")
        logger.info(f"      Mean: {pred_mag_stats['mean']:.6f}, Std: {pred_mag_stats['std']:.6f}")
        logger.info(f"   📊 Target Magnitude Stats:")
        logger.info(f"      Min: {target_mag_stats['min']:.6f}, Max: {target_mag_stats['max']:.6f}")
        logger.info(f"      Mean: {target_mag_stats['mean']:.6f}, Std: {target_mag_stats['std']:.6f}")
        logger.info(f"   📊 Phase Difference Stats (wrapped):")
        logger.info(f"      Min: {phase_diff_stats['min']:.6f}, Max: {phase_diff_stats['max']:.6f}")
        logger.info(f"      Mean: {phase_diff_stats['mean']:.6f}, Std: {phase_diff_stats['std']:.6f}")
        
        # Save comprehensive metrics
        metrics = {
            'magnitude_mse': magnitude_mse,
            'magnitude_mae': magnitude_mae,
            'phase_mse_wrapped': phase_mse_wrapped,
            'phase_mae_wrapped': phase_mae_wrapped,
            'spatial_spectrum_loss': spatial_spectrum_loss,
            'pdp_loss': pdp_loss_value,
            'prediction_magnitude_stats': pred_mag_stats,
            'target_magnitude_stats': target_mag_stats,
            'phase_difference_stats': phase_diff_stats,
            'num_samples': predictions.shape[0],
            'test_timestamp': datetime.now().isoformat()
        }
        
        metrics_file = self.output_dir / 'metrics' / 'test_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"💾 Metrics saved to: {metrics_file}")
    
    def _save_results(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Save testing results"""
        logger.info("💾 Saving testing results...")
        
        results_file = self.output_dir / 'predictions' / 'test_results.npz'
        
        np.savez_compressed(
            results_file,
            predictions=predictions.cpu().numpy(),
            targets=targets.cpu().numpy(),
            test_ue_positions=self.test_ue_positions.cpu().numpy(),
            test_bs_positions=self.test_bs_positions.cpu().numpy()
        )
        
        logger.info(f"💾 Results saved to: {results_file}")
    
    def _generate_all_visualizations(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Generate all required visualizations"""
        logger.info("🎨 Generating comprehensive visualizations...")
        
        try:
            # 1. Magnitude CDF and MAE visualization
            self._visualize_magnitude_cdf_mae(predictions, targets)
            
            # 2. Phase CDF and MAE visualization (with phase wrapping)
            self._visualize_phase_cdf_mae(predictions, targets)
            
            # 3. PDP CDF and loss CDF visualization
            self._visualize_pdp_cdf_loss(predictions, targets)
            
            # 4. Spatial spectrum loss CDF visualization
            self._visualize_spatial_spectrum_loss_cdf(predictions, targets)
            
            # 5. CSI comparison (10 samples, random antennas)
            self._visualize_csi_comparison_random(predictions, targets)
            
            # 6. Array spectrum comparison (10 samples, random frequency)
            self._visualize_array_spectrum_comparison(predictions, targets)
            
            logger.info("✅ All visualizations completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate visualizations: {e}")
            import traceback
            traceback.print_exc()
  
    
    
    def _visualize_magnitude_cdf_mae(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Visualize magnitude CDF and MAE comparison"""
        logger.info("📊 Generating magnitude CDF and MAE visualization...")
        
        try:
            # Extract magnitudes
            pred_mag = torch.abs(predictions).cpu().numpy()
            target_mag = torch.abs(targets).cpu().numpy()
            
            # Flatten for CDF computation
            pred_mag_flat = pred_mag.flatten()
            target_mag_flat = target_mag.flatten()
            
            # Compute CDFs
            pred_mag_cdf = self._compute_empirical_cdf(pred_mag_flat)
            target_mag_cdf = self._compute_empirical_cdf(target_mag_flat)
            
            # Compute MAE per sample
            mae_per_sample = np.mean(np.abs(pred_mag - target_mag), axis=(1, 2))
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle('Magnitude Analysis: CDF and MAE Distribution', fontsize=16, fontweight='bold')
            
            # Plot 1: CDF Comparison
            ax1.plot(pred_mag_cdf[0], pred_mag_cdf[1], 'r-', linewidth=2, label='Predicted Magnitude')
            ax1.plot(target_mag_cdf[0], target_mag_cdf[1], 'b-', linewidth=2, label='Target Magnitude')
            ax1.set_xlabel('Magnitude Value')
            ax1.set_ylabel('Cumulative Probability')
            ax1.set_title('Magnitude CDF Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Plot 2: MAE Distribution
            ax2.hist(mae_per_sample, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('MAE per Sample')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Magnitude MAE Distribution (Mean: {np.mean(mae_per_sample):.4f})')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plots_dir / 'magnitude_cdf_mae.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Magnitude CDF and MAE visualization saved: {plot_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create magnitude visualization: {e}")
            import traceback
            traceback.print_exc()
            
    
    def _visualize_phase_cdf_mae(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Visualize phase CDF and MAE comparison with phase wrapping correction"""
        logger.info("📊 Generating phase CDF and MAE visualization...")
        
        try:
            # Extract phases
            pred_phase = torch.angle(predictions).cpu().numpy()
            target_phase = torch.angle(targets).cpu().numpy()
            
            # Apply phase wrapping correction
            phase_diff = pred_phase - target_phase
            phase_diff_wrapped = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
            
            # Flatten for CDF computation
            pred_phase_flat = pred_phase.flatten()
            target_phase_flat = target_phase.flatten()
            phase_diff_flat = phase_diff_wrapped.flatten()
            
            # Compute CDFs
            pred_phase_cdf = self._compute_empirical_cdf(pred_phase_flat)
            target_phase_cdf = self._compute_empirical_cdf(target_phase_flat)
            
            # Compute MAE per sample (using wrapped phase difference)
            mae_per_sample = np.mean(np.abs(phase_diff_wrapped), axis=(1, 2))
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle('Phase Analysis: CDF and MAE Distribution (with wrapping)', fontsize=16, fontweight='bold')
            
            # Plot 1: CDF Comparison
            ax1.plot(pred_phase_cdf[0], pred_phase_cdf[1], 'r-', linewidth=2, label='Predicted Phase')
            ax1.plot(target_phase_cdf[0], target_phase_cdf[1], 'b-', linewidth=2, label='Target Phase')
            ax1.set_xlabel('Phase Value (rad)')
            ax1.set_ylabel('Cumulative Probability')
            ax1.set_title('Phase CDF Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            ax1.set_xlim(-np.pi, np.pi)
            
            # Plot 2: MAE Distribution (wrapped phase difference)
            ax2.hist(mae_per_sample, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax2.set_xlabel('Phase MAE per Sample (wrapped)')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Phase MAE Distribution (Mean: {np.mean(mae_per_sample):.4f})')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plots_dir / 'phase_cdf_mae.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Phase CDF and MAE visualization saved: {plot_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create phase visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_pdp_cdf_loss(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Visualize PDP CDF and loss CDF comparison"""
        logger.info("📊 Generating PDP CDF and loss visualization...")
        
        # Always create PDP visualization, regardless of whether PDP loss was used in training
        
        try:
            # Convert to numpy for basic PDP analysis
            pred_np = predictions.cpu().numpy()
            target_np = targets.cpu().numpy()
            
            # Compute simple power delay profiles (magnitude squared)
            pred_pdp = np.mean(np.abs(pred_np)**2, axis=(0, 1))  # Average across samples and antennas
            target_pdp = np.mean(np.abs(target_np)**2, axis=(0, 1))
            
            # Compute CDFs for PDP
            pred_pdp_cdf = self._compute_empirical_cdf(pred_pdp)
            target_pdp_cdf = self._compute_empirical_cdf(target_pdp)
            
            # Compute basic PDP differences as "loss"
            pdp_diff = np.abs(pred_pdp - target_pdp)
            pdp_loss_per_sample = pdp_diff  # Use PDP differences as loss metric
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle('PDP Analysis: CDF and Loss Distribution', fontsize=16, fontweight='bold')
            
            # Plot 1: PDP CDF Comparison
            ax1.plot(pred_pdp_cdf[0], pred_pdp_cdf[1], 'r-', linewidth=2, label='Predicted PDP')
            ax1.plot(target_pdp_cdf[0], target_pdp_cdf[1], 'b-', linewidth=2, label='Target PDP')
            ax1.set_xlabel('PDP Value')
            ax1.set_ylabel('Cumulative Probability')
            ax1.set_title('PDP CDF Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Plot 2: PDP Loss CDF
            pdp_loss_cdf = self._compute_empirical_cdf(pdp_loss_per_sample)
            ax2.plot(pdp_loss_cdf[0], pdp_loss_cdf[1], 'g-', linewidth=2, label='PDP Loss CDF')
            ax2.set_xlabel('PDP Loss Value')
            ax2.set_ylabel('Cumulative Probability')
            ax2.set_title(f'PDP Loss CDF (Mean: {np.mean(pdp_loss_per_sample):.4f})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save plot
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plots_dir / 'pdp_cdf_loss.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ PDP CDF and loss visualization saved: {plot_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create PDP visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_spatial_spectrum_loss_cdf(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Visualize spatial spectrum loss CDF"""
        logger.info("📊 Generating spatial spectrum loss CDF visualization...")
        
        # Always create spatial spectrum visualization, regardless of whether spatial spectrum loss was used in training
        
        try:
            # Use ss_loss to compute spatial spectrums
            if self.ss_loss is not None:
                try:
                    # Compute spatial spectrums using SSLoss
                    pred_spectrum = self.ss_loss.compute_spatial_spectrum(predictions)
                    target_spectrum = self.ss_loss.compute_spatial_spectrum(targets)
                    
                    # Compute spatial spectrum differences as "loss"
                    spectrum_diff = torch.abs(pred_spectrum - target_spectrum)
                    ss_losses = spectrum_diff.flatten().cpu().numpy()
                    
                    logger.info(f"✅ Computed spatial spectrums using SSLoss: {pred_spectrum.shape}")
                except ValueError as e:
                    # Handle antenna count mismatch
                    logger.warning(f"⚠️ SSLoss antenna mismatch: {e}")
                    logger.info("🔄 Falling back to basic magnitude analysis")
                    
                    pred_np = predictions.cpu().numpy()
                    target_np = targets.cpu().numpy()
                    
                    pred_mag = np.abs(pred_np)
                    target_mag = np.abs(target_np)
                    
                    mag_diff = np.mean(np.abs(pred_mag - target_mag), axis=0)
                    ss_losses = mag_diff.flatten()
            else:
                # Fallback to basic magnitude analysis if ss_loss not available
                logger.warning("⚠️ SSLoss not available, using basic magnitude analysis")
                pred_np = predictions.cpu().numpy()
                target_np = targets.cpu().numpy()
                
                pred_mag = np.abs(pred_np)
                target_mag = np.abs(target_np)
                
                mag_diff = np.mean(np.abs(pred_mag - target_mag), axis=0)
                ss_losses = mag_diff.flatten()
            
            # Compute CDF for spatial spectrum losses
            ss_loss_cdf = self._compute_empirical_cdf(ss_losses)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle('Spatial Spectrum Analysis: Spectrum Difference Distribution', fontsize=16, fontweight='bold')
            
            # Plot 1: Spatial Spectrum Difference CDF
            ax1.plot(ss_loss_cdf[0], ss_loss_cdf[1], 'g-', linewidth=2, label='Spatial Spectrum Difference')
            ax1.set_xlabel('Spectrum Difference Value')
            ax1.set_ylabel('Cumulative Probability')
            ax1.set_title('Spatial Spectrum Difference CDF')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Plot 2: Spatial Spectrum Difference Distribution
            ax2.hist(ss_losses, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Spectrum Difference')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Spatial Spectrum Difference Distribution (Mean: {np.mean(ss_losses):.4f})')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plots_dir / 'spatial_spectrum_loss_cdf.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Spatial spectrum loss CDF visualization saved: {plot_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create spatial spectrum loss visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_csi_comparison_random(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Visualize CSI comparison for 10 random samples with random antennas"""
        logger.info("📊 Generating CSI comparison visualization (10 samples, random antennas)...")
        
        try:
            # Randomly select 10 samples
            num_samples = predictions.shape[0]
            if num_samples < 10:
                logger.warning(f"⚠️ Only {num_samples} samples available, using all samples")
                selected_indices = list(range(num_samples))
            else:
                selected_indices = np.random.choice(num_samples, 10, replace=False)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 10, figsize=(20, 8))
            fig.suptitle('CSI Comparison: 10 Random Samples with Random Antennas', fontsize=16, fontweight='bold')
            
            for i, sample_idx in enumerate(selected_indices):
                # Get CSI data for this sample
                pred_csi = predictions[sample_idx]  # [num_antennas, num_subcarriers]
                target_csi = targets[sample_idx]   # [num_antennas, num_subcarriers]
                
                # Randomly select one antenna (different for each sample)
                num_antennas = pred_csi.shape[0]
                antenna_idx = np.random.randint(0, num_antennas)
                
                # Select CSI for the chosen antenna
                pred_csi_antenna = pred_csi[antenna_idx]  # [num_subcarriers]
                target_csi_antenna = target_csi[antenna_idx]  # [num_subcarriers]
                
                # Calculate amplitude and phase
                pred_amp = torch.abs(pred_csi_antenna).cpu().numpy()
                target_amp = torch.abs(target_csi_antenna).cpu().numpy()
                pred_phase = torch.angle(pred_csi_antenna).cpu().numpy()
                target_phase = torch.angle(target_csi_antenna).cpu().numpy()
                
                # Subcarrier indices
                subcarrier_indices = np.arange(len(pred_amp))
                
                # Plot amplitude comparison
                axes[0, i].plot(subcarrier_indices, pred_amp, 'b-', label='Predicted', linewidth=2)
                axes[0, i].plot(subcarrier_indices, target_amp, 'r--', label='Target', linewidth=2)
                axes[0, i].set_title(f'Sample {sample_idx}\nAntenna {antenna_idx}', fontsize=8, fontweight='bold')
                axes[0, i].set_ylabel('Amplitude', fontsize=8)
                axes[0, i].grid(True, alpha=0.3)
                axes[0, i].legend(fontsize=6)
                
                # Plot phase comparison
                axes[1, i].plot(subcarrier_indices, pred_phase, 'b-', label='Predicted', linewidth=2)
                axes[1, i].plot(subcarrier_indices, target_phase, 'r--', label='Target', linewidth=2)
                axes[1, i].set_xlabel('Subcarrier Index', fontsize=8)
                axes[1, i].set_ylabel('Phase (rad)', fontsize=8)
                axes[1, i].grid(True, alpha=0.3)
                axes[1, i].legend(fontsize=6)
                axes[1, i].set_ylim([-np.pi, np.pi])
            
            # Add row labels
            axes[0, 0].text(-0.15, 0.5, 'Amplitude', transform=axes[0, 0].transAxes, 
                            rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
            axes[1, 0].text(-0.15, 0.5, 'Phase', transform=axes[1, 0].transAxes, 
                            rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plots_dir / 'csi_comparison_random.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ CSI comparison visualization saved: {plot_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create CSI comparison visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_array_spectrum_comparison(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Visualize array spectrum comparison for 10 random samples at random frequency"""
        logger.info("�� Generating array spectrum comparison visualization...")
        
        try:
            # Randomly select 10 samples
            num_samples = predictions.shape[0]
            if num_samples < 10:
                logger.warning(f"⚠️ Only {num_samples} samples available, using all samples")
                selected_indices = list(range(num_samples))
            else:
                selected_indices = np.random.choice(num_samples, 10, replace=False)
            
            # Randomly select one frequency (same for all samples)
            num_subcarriers = predictions.shape[2]
            freq_idx = np.random.randint(0, num_subcarriers)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 10, figsize=(20, 8))
            fig.suptitle(f'Array Spectrum Comparison: 10 Random Samples at Frequency {freq_idx}', fontsize=16, fontweight='bold')
            
            for i, sample_idx in enumerate(selected_indices):
                # Get CSI data for this sample at the selected frequency
                pred_csi_freq = predictions[sample_idx, :, freq_idx]  # [num_antennas]
                target_csi_freq = targets[sample_idx, :, freq_idx]   # [num_antennas]
                
                # Calculate amplitude and phase
                pred_amp = torch.abs(pred_csi_freq).cpu().numpy()
                target_amp = torch.abs(target_csi_freq).cpu().numpy()
                pred_phase = torch.angle(pred_csi_freq).cpu().numpy()
                target_phase = torch.angle(target_csi_freq).cpu().numpy()
                
                # Antenna indices
                antenna_indices = np.arange(len(pred_amp))
                
                # Plot amplitude comparison
                axes[0, i].plot(antenna_indices, pred_amp, 'b-', label='Predicted', linewidth=2)
                axes[0, i].plot(antenna_indices, target_amp, 'r--', label='Target', linewidth=2)
                axes[0, i].set_title(f'Sample {sample_idx}', fontsize=8, fontweight='bold')
                axes[0, i].set_ylabel('Amplitude', fontsize=8)
                axes[0, i].grid(True, alpha=0.3)
                axes[0, i].legend(fontsize=6)
                
                # Plot phase comparison
                axes[1, i].plot(antenna_indices, pred_phase, 'b-', label='Predicted', linewidth=2)
                axes[1, i].plot(antenna_indices, target_phase, 'r--', label='Target', linewidth=2)
                axes[1, i].set_xlabel('Antenna Index', fontsize=8)
                axes[1, i].set_ylabel('Phase (rad)', fontsize=8)
                axes[1, i].grid(True, alpha=0.3)
                axes[1, i].legend(fontsize=6)
                axes[1, i].set_ylim([-np.pi, np.pi])
            
            # Add row labels
            axes[0, 0].text(-0.15, 0.5, 'Amplitude', transform=axes[0, 0].transAxes, 
                            rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
            axes[1, 0].text(-0.15, 0.5, 'Phase', transform=axes[1, 0].transAxes, 
                            rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plots_dir / 'array_spectrum_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Array spectrum comparison visualization saved: {plot_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create array spectrum comparison visualization: {e}")
            import traceback
            traceback.print_exc()

    def _compute_empirical_cdf(self, data: np.ndarray) -> tuple:
        """Compute empirical cumulative distribution function (CDF) for given data"""
        # Sort data and compute CDF
        sorted_data = np.sort(data)
        n = len(sorted_data)
        cdf_values = np.arange(1, n + 1) / n
        
        return sorted_data, cdf_values


def main():
    """Main entry point for generic testing."""
    parser = argparse.ArgumentParser(
        description='Generic Prism Neural Network Testing Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with Sionna configuration
  python test.py --config configs/sionna.yml
  
  # Test with PolyU configuration
  python test.py --config configs/polyu.yml
  
  # Test with Chrissy configuration
  python test.py --config configs/chrissy.yml
  
  # Test with custom model and output directory
  python test.py --config configs/sionna.yml --model path/to/model.pt --output results/custom_test
  
  # Test on specific GPU
  python test.py --config configs/sionna.yml --gpu 0
        """
    )
    parser.add_argument('--config', required=True, help='Path to configuration file (e.g., configs/sionna.yml)')
    parser.add_argument('--model', help='Path to trained model file (optional, will auto-detect if not provided)')
    parser.add_argument('--data', help='Path to test data file (optional, will use config path)')
    parser.add_argument('--output', help='Output directory for results (optional, will use config output dir)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    
    args = parser.parse_args()
    
    try:
        print(f"🚀 Starting Prism testing with configuration: {args.config}")
        
        tester = PrismTester(
            config_path=args.config,
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output,
            gpu_id=args.gpu
        )
        
        # Run testing
        tester.test()
        
        print("✅ Testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
