#!/usr/bin/env python3
"""
Prism Network Testing Script (Updated for Modern Configuration)

This script tests the trained Prism neural network for electromagnetic ray tracing.
It loads a trained model and evaluates its performance on test data, including
metrics calculation, visualization, and comparison with ground truth.

Updated to use ModernConfigLoader and new simplified architecture.
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
    """Main tester class for Prism network using modern configuration"""
    
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
            self.output_dir / 'predictions',
            self.output_dir / 'analysis',
            self.output_dir / 'spatial_spectrums',
            self.output_dir / 'pdp_analysis',
            self.output_dir / 'magnitude_phase_analysis'
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
        """Load and prepare testing data using new HDF5 structure"""
        logger.info("🔧 Loading testing data...")
        
        # Construct actual dataset file path (same as training)
        actual_dataset_path = "data/sionna/results/P300/ray_tracing_5g_simulation_P300.h5"
        
        if not os.path.exists(actual_dataset_path):
            raise FileNotFoundError(f"Dataset not found: {actual_dataset_path}")
        
        logger.info(f"Loading data from: {actual_dataset_path}")
        
        # Load data from new HDF5 structure
        with h5py.File(actual_dataset_path, 'r') as f:
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
                logger.info(f"📍 Fixed BS mode: Single BS position for all {num_samples} samples")
                bs_positions = bs_position_single.unsqueeze(0).expand(num_samples, -1)
                self._bs_mode = 'fixed'
            elif bs_position_single.dim() == 2 and bs_position_single.shape[0] == num_samples:
                # Per-sample BS positions
                logger.info(f"📍 Dynamic BS mode: {num_samples} BS positions for {num_samples} samples")
                bs_positions = bs_position_single
                self._bs_mode = 'dynamic'
            else:
                raise ValueError(f"Invalid BS position shape: {bs_position_single.shape}. "
                               f"Expected [3] for fixed BS or [{num_samples}, 3] for dynamic BS.")
            
            # Generate antenna indices (sequential 0 to num_antennas-1 for each sample)
            num_bs_antennas = csi_data.shape[1]
            antenna_indices = torch.arange(num_bs_antennas).unsqueeze(0).expand(num_samples, -1).long()
            
            # Extract specific UE antenna and remove UE dimension
            data_config = self.config_loader.get_data_loader_config()
            ue_antenna_idx = data_config['ue_antenna_index']
            if csi_data.dim() == 4:  # [samples, bs_antennas, subcarriers, ue_antennas]
                csi_data = csi_data[:, :, :, ue_antenna_idx]  # Remove UE dimension: [samples, bs_antennas, subcarriers]
            
            # Phase differential calibration to remove common phase offset
            logger.info("🔧 Applying phase differential calibration...")
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
            
            logger.info(f"✅ Phase differential calibration applied:")
            logger.info(f"   Original subcarriers: {original_subcarriers}")
            logger.info(f"   Normalized subcarriers: {csi_data.shape[2]}")
            logger.info(f"   Final format: [samples, antennas, subcarriers] = {csi_data.shape}")
            logger.info(f"   Method: First subcarrier normalization (csi[k] / csi[0])")
        
        logger.info(f"✅ Data loaded from new HDF5 structure:")
        logger.info(f"   UE positions: {ue_positions.shape}")
        logger.info(f"   BS mode: {self._bs_mode}")
        if self._bs_mode == 'fixed':
            logger.info(f"   BS position (fixed): {bs_position_single}")
        else:
            logger.info(f"   BS positions (dynamic): {bs_positions.shape}")
        logger.info(f"   CSI data: {csi_data.shape} [samples, bs_antennas, subcarriers, ue_antennas]")
        logger.info(f"   Antenna indices (generated): {antenna_indices.shape}")
        logger.info(f"   Selected UE antenna: {ue_antenna_idx}")
        
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
                    
                    batch_predictions = outputs['csi_predictions']
                    
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
            
            # Generate spatial spectrum data
            target_spectrums, pred_spectrums = self._compute_spatial_spectrum(all_predictions, all_targets)
            
            # Save spectrum data if available
            if target_spectrums is not None and pred_spectrums is not None and len(target_spectrums) > 0:
                self._save_spatial_spectrum_data(target_spectrums, pred_spectrums)
                # Generate spatial spectrum visualization
                self._visualize_spatial_spectrums(target_spectrums, pred_spectrums)
            
            # Generate and visualize PDP CDF
            self._compute_and_visualize_pdp_cdf(all_predictions, all_targets)
            
            # Generate magnitude and phase CDF analysis
            self._compute_and_visualize_magnitude_phase_cdf(all_predictions, all_targets)
            
            # Generate CSI amplitude and phase comparison plots
            self._visualize_csi_comparison(all_predictions, all_targets)
            
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
        
        # 1. Complex MSE (manual calculation for complex tensors)
        # PyTorch MSE doesn't support complex tensors directly
        complex_diff = predictions - targets
        complex_mse = torch.mean(torch.abs(complex_diff) ** 2).item()
        complex_mae = torch.mean(torch.abs(complex_diff)).item()
        
        # 2. Magnitude and Phase extraction
        pred_mag = torch.abs(predictions)
        target_mag = torch.abs(targets)
        pred_phase = torch.angle(predictions)  # Range: [-π, π]
        target_phase = torch.angle(targets)    # Range: [-π, π]
        
        # 3. Magnitude MSE
        magnitude_mse = torch.nn.functional.mse_loss(pred_mag, target_mag).item()
        magnitude_mae = torch.nn.functional.l1_loss(pred_mag, target_mag).item()
        
        # 4. Phase MSE with wrapping correction
        # Handle phase wrapping: find the shortest angular distance
        phase_diff = pred_phase - target_phase
        
        # Wrap phase differences to [-π, π] range
        phase_diff_wrapped = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        
        # Calculate phase MSE using wrapped differences
        phase_mse_wrapped = torch.mean(phase_diff_wrapped ** 2).item()
        phase_mae_wrapped = torch.mean(torch.abs(phase_diff_wrapped)).item()
        
        # 5. Spatial Spectrum Loss
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
        logger.info(f"   🔢 Complex MSE:           {complex_mse:.8f}")
        logger.info(f"   🔢 Complex MAE:           {complex_mae:.8f}")
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
            'complex_mse': complex_mse,
            'complex_mae': complex_mae,
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
    
  
    def _compute_and_visualize_pdp_cdf(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Generate and visualize PDP CDF comparison"""
        logger.info("📊 Generating PDP CDF visualization...")
        
        if self.pdp_loss is None:
            logger.warning("⚠️  PDP loss not available, skipping PDP CDF visualization")
            return
        
        try:
            # Verify tensor shapes match (should be consistent with first subcarrier normalization)
            if predictions.shape != targets.shape:
                logger.warning(f"⚠️ PDP tensor shape mismatch: pred {predictions.shape} vs target {targets.shape}")
                # This should not happen with first subcarrier normalization method
            
            # Move tensors to device
            pred_device = predictions.to(self.device)
            target_device = targets.to(self.device)
            
            # Reshape for PDP computation: [batch*antennas, subcarriers]
            batch_size, num_antennas, num_subcarriers = pred_device.shape
            pred_flat = pred_device.reshape(-1, num_subcarriers)
            target_flat = target_device.reshape(-1, num_subcarriers)
            
            # Compute PDPs using the public interface
            with torch.no_grad():
                pred_pdp = self.pdp_loss.compute_pdp_only(pred_flat).cpu().numpy()  # [N, fft_size]
                target_pdp = self.pdp_loss.compute_pdp_only(target_flat).cpu().numpy()
            
            # Compute CDFs
            pred_cdf = self._compute_cdf_from_pdp(pred_pdp)  # [N, fft_size]
            target_cdf = self._compute_cdf_from_pdp(target_pdp)
            
            # Create delay axis (in normalized delay bins)
            delay_axis = np.arange(pred_pdp.shape[1]) / pred_pdp.shape[1]  # Normalized from 0 to 1
            
            # Create visualization
            self._visualize_pdp_cdf_comparison(delay_axis, pred_cdf, target_cdf, pred_pdp, target_pdp)
            
            # Save PDP data
            self._save_pdp_cdf_data(delay_axis, pred_cdf, target_cdf, pred_pdp, target_pdp)
            
            logger.info("✅ PDP CDF visualization completed")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate PDP CDF visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_cdf_from_pdp(self, pdp: np.ndarray) -> np.ndarray:
        """Compute CDF from PDP data"""
        # Normalize PDPs so they sum to 1 (treat as probability distributions)
        pdp_normalized = pdp / (np.sum(pdp, axis=1, keepdims=True) + 1e-8)
        # Compute cumulative sum along delay dimension
        cdf = np.cumsum(pdp_normalized, axis=1)
        return cdf
    
    def _visualize_pdp_cdf_comparison(self, delay_axis: np.ndarray, pred_cdf: np.ndarray, 
                                     target_cdf: np.ndarray, pred_pdp: np.ndarray, target_pdp: np.ndarray):
        """Create PDP and CDF comparison plots"""
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Compute statistics across all samples
            pred_cdf_mean = np.mean(pred_cdf, axis=0)
            pred_cdf_std = np.std(pred_cdf, axis=0)
            target_cdf_mean = np.mean(target_cdf, axis=0)
            target_cdf_std = np.std(target_cdf, axis=0)
            
            pred_pdp_mean = np.mean(pred_pdp, axis=0)
            pred_pdp_std = np.std(pred_pdp, axis=0)
            target_pdp_mean = np.mean(target_pdp, axis=0)
            target_pdp_std = np.std(target_pdp, axis=0)
            
            # Plot 1: CDF Comparison (mean ± std)
            ax1.plot(delay_axis, target_cdf_mean, 'b-', linewidth=2, label='Ground Truth CDF')
            ax1.fill_between(delay_axis, target_cdf_mean - target_cdf_std, target_cdf_mean + target_cdf_std, 
                           alpha=0.3, color='blue')
            ax1.plot(delay_axis, pred_cdf_mean, 'r--', linewidth=2, label='Predicted CDF')
            ax1.fill_between(delay_axis, pred_cdf_mean - pred_cdf_std, pred_cdf_mean + pred_cdf_std, 
                           alpha=0.3, color='red')
            ax1.set_xlabel('Normalized Delay')
            ax1.set_ylabel('Cumulative Probability')
            ax1.set_title('PDP CDF Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Plot 2: PDP Comparison (mean ± std)
            ax2.plot(delay_axis, target_pdp_mean, 'b-', linewidth=2, label='Ground Truth PDP')
            ax2.fill_between(delay_axis, target_pdp_mean - target_pdp_std, target_pdp_mean + target_pdp_std, 
                           alpha=0.3, color='blue')
            ax2.plot(delay_axis, pred_pdp_mean, 'r--', linewidth=2, label='Predicted PDP')
            ax2.fill_between(delay_axis, pred_pdp_mean - pred_pdp_std, pred_pdp_mean + pred_pdp_std, 
                           alpha=0.3, color='red')
            ax2.set_xlabel('Normalized Delay')
            ax2.set_ylabel('Power')
            ax2.set_title('PDP Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            # Plot 3: CDF Difference
            cdf_diff = np.abs(pred_cdf_mean - target_cdf_mean)
            ax3.plot(delay_axis, cdf_diff, 'g-', linewidth=2)
            ax3.set_xlabel('Normalized Delay')
            ax3.set_ylabel('|CDF Difference|')
            ax3.set_title('CDF Absolute Difference')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Sample CDFs (first 10 samples)
            num_samples_to_plot = min(10, pred_cdf.shape[0])
            for i in range(num_samples_to_plot):
                alpha = 0.6 if i < 5 else 0.3
                ax4.plot(delay_axis, target_cdf[i], 'b-', alpha=alpha, linewidth=1)
                ax4.plot(delay_axis, pred_cdf[i], 'r--', alpha=alpha, linewidth=1)
            
            # Add legend for sample plot
            ax4.plot([], [], 'b-', label='Ground Truth (samples)')
            ax4.plot([], [], 'r--', label='Predicted (samples)')
            ax4.set_xlabel('Normalized Delay')
            ax4.set_ylabel('Cumulative Probability')
            ax4.set_title(f'Sample CDFs (first {num_samples_to_plot})')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save the plot
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            vis_save_path = plots_dir / 'pdp_cdf_comparison.png'
            plt.savefig(vis_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ PDP CDF visualization saved to: {vis_save_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create PDP CDF plots: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_pdp_cdf_data(self, delay_axis: np.ndarray, pred_cdf: np.ndarray, 
                          target_cdf: np.ndarray, pred_pdp: np.ndarray, target_pdp: np.ndarray):
        """Save PDP and CDF data to files"""
        try:
            # Create data directory
            data_dir = self.output_dir / 'pdp_analysis'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save raw data
            np.save(data_dir / 'delay_axis.npy', delay_axis)
            np.save(data_dir / 'predicted_cdf.npy', pred_cdf)
            np.save(data_dir / 'target_cdf.npy', target_cdf)
            np.save(data_dir / 'predicted_pdp.npy', pred_pdp)
            np.save(data_dir / 'target_pdp.npy', target_pdp)
            
            # Save metadata
            metadata = {
                'num_samples': pred_cdf.shape[0],
                'fft_size': pred_pdp.shape[1],
                'delay_resolution': 1.0 / pred_pdp.shape[1],
                'normalize_pdp': self.pdp_loss.normalize_pdp,
                'pdp_loss_type': self.pdp_loss.loss_type,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(data_dir / 'pdp_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 PDP CDF data saved to: {data_dir}")
            logger.info(f"   - Delay axis shape: {delay_axis.shape}")
            logger.info(f"   - CDF shapes: pred={pred_cdf.shape}, target={target_cdf.shape}")
            logger.info(f"   - PDP shapes: pred={pred_pdp.shape}, target={target_pdp.shape}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save PDP CDF data: {e}")
    
    def _compute_and_visualize_magnitude_phase_cdf(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Generate and visualize magnitude and phase CDF analysis"""
        logger.info("📊 Generating magnitude and phase CDF analysis...")
        
        try:
            # Verify tensor shapes match (should be consistent with first subcarrier normalization)
            if predictions.shape != targets.shape:
                logger.warning(f"⚠️ Magnitude/phase tensor shape mismatch: pred {predictions.shape} vs target {targets.shape}")
                # This should not happen with first subcarrier normalization method
            
            # Extract magnitude and phase
            pred_magnitude = torch.abs(predictions).cpu().numpy()
            target_magnitude = torch.abs(targets).cpu().numpy()
            pred_phase = torch.angle(predictions).cpu().numpy()
            target_phase = torch.angle(targets).cpu().numpy()
            
            # Flatten for CDF computation
            pred_mag_flat = pred_magnitude.flatten()
            target_mag_flat = target_magnitude.flatten()
            pred_phase_flat = pred_phase.flatten()
            target_phase_flat = target_phase.flatten()
            
            # Compute CDFs
            pred_mag_cdf = self._compute_empirical_cdf(pred_mag_flat)
            target_mag_cdf = self._compute_empirical_cdf(target_mag_flat)
            pred_phase_cdf = self._compute_empirical_cdf(pred_phase_flat)
            target_phase_cdf = self._compute_empirical_cdf(target_phase_flat)
            
            # Create visualization
            self._visualize_magnitude_phase_cdf_comparison(
                pred_mag_cdf, target_mag_cdf, pred_phase_cdf, target_phase_cdf,
                pred_mag_flat, target_mag_flat, pred_phase_flat, target_phase_flat
            )
            
            # Save CDF data
            self._save_magnitude_phase_cdf_data(
                pred_mag_cdf, target_mag_cdf, pred_phase_cdf, target_phase_cdf
            )
            
            logger.info("✅ Magnitude and phase CDF analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate magnitude/phase CDF analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_empirical_cdf(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute empirical CDF from data"""
        # Sort data
        sorted_data = np.sort(data)
        # Compute empirical CDF values
        n = len(sorted_data)
        cdf_values = np.arange(1, n + 1) / n
        return sorted_data, cdf_values
    
    def _visualize_magnitude_phase_cdf_comparison(self, pred_mag_cdf, target_mag_cdf, 
                                                 pred_phase_cdf, target_phase_cdf,
                                                 pred_mag_flat, target_mag_flat, 
                                                 pred_phase_flat, target_phase_flat):
        """Create magnitude and phase CDF comparison plots"""
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('CSI Magnitude and Phase CDF Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Magnitude CDF Comparison
            ax1.plot(pred_mag_cdf[0], pred_mag_cdf[1], 'r-', linewidth=2, label='Predicted Magnitude')
            ax1.plot(target_mag_cdf[0], target_mag_cdf[1], 'b-', linewidth=2, label='Target Magnitude')
            ax1.set_xlabel('Magnitude Value')
            ax1.set_ylabel('Cumulative Probability')
            ax1.set_title('Magnitude CDF Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, None)
            ax1.set_ylim(0, 1)
            
            # Plot 2: Phase CDF Comparison
            ax2.plot(pred_phase_cdf[0], pred_phase_cdf[1], 'r-', linewidth=2, label='Predicted Phase')
            ax2.plot(target_phase_cdf[0], target_phase_cdf[1], 'b-', linewidth=2, label='Target Phase')
            ax2.set_xlabel('Phase Value (radians)')
            ax2.set_ylabel('Cumulative Probability')
            ax2.set_title('Phase CDF Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-np.pi, np.pi)
            ax2.set_ylim(0, 1)
            
            # Plot 3: Magnitude Distribution (PDF approximation)
            ax3.hist(target_mag_flat, bins=100, alpha=0.7, density=True, label='Target Magnitude', color='blue')
            ax3.hist(pred_mag_flat, bins=100, alpha=0.7, density=True, label='Predicted Magnitude', color='red')
            ax3.set_xlabel('Magnitude Value')
            ax3.set_ylabel('Density')
            ax3.set_title('Magnitude Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, None)
            
            # Plot 4: Phase Distribution (PDF approximation)
            ax4.hist(target_phase_flat, bins=100, alpha=0.7, density=True, label='Target Phase', color='blue')
            ax4.hist(pred_phase_flat, bins=100, alpha=0.7, density=True, label='Predicted Phase', color='red')
            ax4.set_xlabel('Phase Value (radians)')
            ax4.set_ylabel('Density')
            ax4.set_title('Phase Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(-np.pi, np.pi)
            
            plt.tight_layout()
            
            # Save the plot
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            vis_save_path = plots_dir / 'magnitude_phase_cdf_comparison.png'
            plt.savefig(vis_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Magnitude/Phase CDF visualization saved to: {vis_save_path}")
            
            # Generate statistical comparison
            self._generate_magnitude_phase_statistics(
                pred_mag_flat, target_mag_flat, pred_phase_flat, target_phase_flat
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create magnitude/phase CDF plots: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_magnitude_phase_statistics(self, pred_mag, target_mag, pred_phase, target_phase):
        """Generate detailed statistical comparison"""
        try:
            # Compute Kolmogorov-Smirnov test statistics (simple approximation)
            from scipy import stats
            
            # KS test for magnitude (check for valid data)
            if len(pred_mag) > 10 and len(target_mag) > 10:
                ks_stat_mag, ks_p_mag = stats.ks_2samp(pred_mag, target_mag)
            else:
                ks_stat_mag, ks_p_mag = float('nan'), float('nan')
            
            # KS test for phase (handle circular nature)
            if len(pred_phase) > 10 and len(target_phase) > 10:
                ks_stat_phase, ks_p_phase = stats.ks_2samp(pred_phase, target_phase)
            else:
                ks_stat_phase, ks_p_phase = float('nan'), float('nan')
            
            # Create statistical summary figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Statistical summary text (handle NaN values safely)
            ks_mag_str = f"D={ks_stat_mag:.4f}, p={ks_p_mag:.4f}" if not np.isnan(ks_stat_mag) else "N/A (insufficient data)"
            ks_phase_str = f"D={ks_stat_phase:.4f}, p={ks_p_phase:.4f}" if not np.isnan(ks_stat_phase) else "N/A (insufficient data)"
            
            mag_pass = 'PASS' if not np.isnan(ks_p_mag) and ks_p_mag > 0.05 else 'FAIL' if not np.isnan(ks_p_mag) else 'N/A'
            phase_pass = 'PASS' if not np.isnan(ks_p_phase) and ks_p_phase > 0.05 else 'FAIL' if not np.isnan(ks_p_phase) else 'N/A'
            
            stats_text = f"""
CSI Statistical Analysis Summary

Magnitude Statistics:
• Target: μ={np.mean(target_mag):.4f}, σ={np.std(target_mag):.4f}
• Predicted: μ={np.mean(pred_mag):.4f}, σ={np.std(pred_mag):.4f}
• KS Test: {ks_mag_str}

Phase Statistics:
• Target: μ={np.mean(target_phase):.4f}, σ={np.std(target_phase):.4f}
• Predicted: μ={np.mean(pred_phase):.4f}, σ={np.std(pred_phase):.4f}
• KS Test: {ks_phase_str}

Distribution Similarity:
• Magnitude KS Test: {mag_pass} (p > 0.05)
• Phase KS Test: {phase_pass} (p > 0.05)
            """
            
            ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('CSI Statistical Analysis Summary', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # Save statistical summary
            plots_dir = self.output_dir / 'plots'
            stats_save_path = plots_dir / 'magnitude_phase_statistics.png'
            plt.savefig(stats_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Statistical summary saved to: {stats_save_path}")
            
        except ImportError:
            logger.warning("⚠️  scipy not available, skipping KS test statistics")
        except Exception as e:
            logger.error(f"❌ Failed to generate statistical comparison: {e}")
    
    def _save_magnitude_phase_cdf_data(self, pred_mag_cdf, target_mag_cdf, pred_phase_cdf, target_phase_cdf):
        """Save magnitude and phase CDF data to files"""
        try:
            # Create data directory
            data_dir = self.output_dir / 'magnitude_phase_analysis'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save CDF data
            np.save(data_dir / 'predicted_magnitude_cdf_x.npy', pred_mag_cdf[0])
            np.save(data_dir / 'predicted_magnitude_cdf_y.npy', pred_mag_cdf[1])
            np.save(data_dir / 'target_magnitude_cdf_x.npy', target_mag_cdf[0])
            np.save(data_dir / 'target_magnitude_cdf_y.npy', target_mag_cdf[1])
            
            np.save(data_dir / 'predicted_phase_cdf_x.npy', pred_phase_cdf[0])
            np.save(data_dir / 'predicted_phase_cdf_y.npy', pred_phase_cdf[1])
            np.save(data_dir / 'target_phase_cdf_x.npy', target_phase_cdf[0])
            np.save(data_dir / 'target_phase_cdf_y.npy', target_phase_cdf[1])
            
            # Save metadata
            metadata = {
                'magnitude_samples': len(pred_mag_cdf[0]),
                'phase_samples': len(pred_phase_cdf[0]),
                'magnitude_range': [float(pred_mag_cdf[0].min()), float(pred_mag_cdf[0].max())],
                'phase_range': [float(pred_phase_cdf[0].min()), float(pred_phase_cdf[0].max())],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(data_dir / 'cdf_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Magnitude/Phase CDF data saved to: {data_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save magnitude/phase CDF data: {e}")
    
    def _compute_spatial_spectrum(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Generate spatial spectrum data for 10 random test samples and return the spectrums"""
        logger.info("🎨 Computing spatial spectrum data...")
        
        if self.ss_loss is None:
            logger.warning("⚠️  Spatial spectrum loss not available, skipping spatial spectrum computation")
            return None, None
        
        try:
            # Randomly select 10 samples
            num_samples = predictions.shape[0]
            if num_samples < 10:
                logger.warning(f"⚠️  Only {num_samples} samples available, using all")
                selected_indices = list(range(num_samples))
            else:
                np.random.seed(42)  # For reproducible results
                selected_indices = np.random.choice(num_samples, 10, replace=False)
            
            target_spectrums = []
            pred_spectrums = []
            
            for sample_idx in selected_indices:
                # Get single sample CSI data
                target_sample = targets[sample_idx:sample_idx+1].to(self.device)  # [1, bs_antennas, subcarriers, ue_antennas]
                pred_sample = predictions[sample_idx:sample_idx+1].to(self.device)
                
                # Verify tensor shapes match (should be consistent with first subcarrier normalization)
                if pred_sample.shape != target_sample.shape:
                    logger.warning(f"⚠️ Sample shape mismatch: pred {pred_sample.shape} vs target {target_sample.shape}")
                    # This should not happen with first subcarrier normalization method
                
                # 🐛 DEBUG: Check input CSI data
                logger.info(f"🔍 Sample {sample_idx+1} INPUT CSI DEBUG:")
                logger.info(f"  📏 Target CSI shape: {target_sample.shape}")
                logger.info(f"  📏 Pred CSI shape: {pred_sample.shape}")
                target_mag = torch.abs(target_sample).cpu().numpy()
                pred_mag = torch.abs(pred_sample).cpu().numpy()
                logger.info(f"  📊 Target CSI magnitude: min={target_mag.min():.6f}, max={target_mag.max():.6f}, mean={target_mag.mean():.6f}")
                logger.info(f"  📊 Pred CSI magnitude: min={pred_mag.min():.6f}, max={pred_mag.max():.6f}, mean={pred_mag.mean():.6f}")
                
                try:
                    # Generate high-resolution spatial spectrum using pre-configured SS_loss
                    with torch.no_grad():
                        target_spectrum = self.ss_loss.compute_spatial_spectrum(target_sample)  # [1, theta_points, phi_points]
                        pred_spectrum = self.ss_loss.compute_spatial_spectrum(pred_sample)
                    
                    # Convert to numpy and remove batch dimension
                    target_spectrum_np = target_spectrum[0].cpu().numpy()  # [theta_points, phi_points]
                    pred_spectrum_np = pred_spectrum[0].cpu().numpy()
                    
                    # 🐛 DEBUG: Detailed spectrum statistics
                    logger.info(f"🔍 Sample {sample_idx+1} DEBUG:")
                    logger.info(f"  📏 Target spectrum shape: {target_spectrum_np.shape}")
                    logger.info(f"  📏 Pred spectrum shape: {pred_spectrum_np.shape}")
                    logger.info(f"  📊 Target stats: min={target_spectrum_np.min():.6f}, max={target_spectrum_np.max():.6f}, mean={target_spectrum_np.mean():.6f}")
                    logger.info(f"  📊 Pred stats: min={pred_spectrum_np.min():.6f}, max={pred_spectrum_np.max():.6f}, mean={pred_spectrum_np.mean():.6f}")
                    logger.info(f"  🔥 Target non-zero elements: {np.count_nonzero(target_spectrum_np)}/{target_spectrum_np.size}")
                    logger.info(f"  🔥 Pred non-zero elements: {np.count_nonzero(pred_spectrum_np)}/{pred_spectrum_np.size}")
                    
                    # Check for potential issues
                    if target_spectrum_np.max() < 1e-6:
                        logger.warning(f"⚠️  Sample {sample_idx+1}: Target spectrum is almost all zeros - likely invalid CSI data")
                        # Skip this sample for correlation calculation to avoid NaN
                        target_spectrums.append(None)
                        pred_spectrums.append(None)
                        continue
                    if pred_spectrum_np.max() < 1e-6:
                        logger.warning(f"⚠️  Sample {sample_idx+1}: Predicted spectrum is almost all zeros!")
                    if np.allclose(target_spectrum_np, pred_spectrum_np, atol=1e-8):
                        logger.warning(f"⚠️  Sample {sample_idx+1}: Target and predicted spectrums are identical - possible data leakage!")
                    
                    target_spectrums.append(target_spectrum_np)
                    pred_spectrums.append(pred_spectrum_np)
                    
                    # Log statistics for this sample (safe correlation calculation)
                    try:
                        target_flat = target_spectrum_np.flatten()
                        pred_flat = pred_spectrum_np.flatten()
                        
                        # Check for valid data before correlation
                        if len(target_flat) > 1 and np.std(target_flat) > 1e-8 and np.std(pred_flat) > 1e-8:
                            correlation = np.corrcoef(target_flat, pred_flat)[0, 1]
                            if np.isnan(correlation):
                                correlation = 0.0
                        else:
                            correlation = 0.0
                            
                        mse = np.mean((target_spectrum_np - pred_spectrum_np) ** 2)
                        logger.info(f"📊 Sample {sample_idx+1}: Corr={correlation:.3f}, MSE={mse:.3f}")
                    except Exception as e:
                        logger.warning(f"⚠️  Sample {sample_idx+1}: Failed to compute correlation: {e}")
                        logger.info(f"📊 Sample {sample_idx+1}: Corr=N/A, MSE=N/A")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to generate spatial spectrum for sample {sample_idx}: {e}")
                    # Add None placeholders
                    target_spectrums.append(None)
                    pred_spectrums.append(None)
            
            # Convert to numpy arrays, filtering out None values
            valid_targets = [s for s in target_spectrums if s is not None]
            valid_preds = [s for s in pred_spectrums if s is not None]
            
            if len(valid_targets) == 0:
                logger.error("❌ No valid spatial spectrum samples generated!")
                return None, None
                
            target_spectrums = np.array(valid_targets)
            pred_spectrums = np.array(valid_preds)
            
            logger.info(f"✅ Generated spatial spectrums for {len(target_spectrums)} samples")
            logger.info(f"   Spectrum shape: {target_spectrums[0].shape if len(target_spectrums) > 0 else 'N/A'}")
            
            return target_spectrums, pred_spectrums
            
        except Exception as e:
            logger.error(f"❌ Failed to generate spatial spectrum data: {e}")
            return None, None
    
    def _save_spatial_spectrum_data(self, target_spectrums: np.ndarray, pred_spectrums: np.ndarray):
        """Save spatial spectrum data to files"""
        try:
            # Create spectrum data directory
            spectrum_dir = self.output_dir / 'spatial_spectrums'
            spectrum_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as numpy arrays
            target_file = spectrum_dir / 'target_spatial_spectrums.npy'
            pred_file = spectrum_dir / 'predicted_spatial_spectrums.npy'
            
            np.save(target_file, target_spectrums)
            np.save(pred_file, pred_spectrums)
            
            # Save metadata
            metadata = {
                'num_samples': len(target_spectrums),
                'spectrum_shape': target_spectrums[0].shape if len(target_spectrums) > 0 else None,
                'theta_points': target_spectrums.shape[1] if len(target_spectrums) > 0 else 0,
                'phi_points': target_spectrums.shape[2] if len(target_spectrums) > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_file = spectrum_dir / 'spatial_spectrum_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Spatial spectrum data saved to: {spectrum_dir}")
            logger.info(f"   Target file: {target_file}")
            logger.info(f"   Predicted file: {pred_file}")
            logger.info(f"   Shape: {target_spectrums.shape}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save spatial spectrum data: {e}")
    
    def _visualize_spatial_spectrums(self, target_spectrums: np.ndarray, pred_spectrums: np.ndarray):
        """Generate comparison plots for spatial spectrums (10 samples in one figure)"""
        logger.info("🎨 Generating spatial spectrum comparison plots...")
        
        try:
            num_samples = target_spectrums.shape[0]
            samples_to_plot = min(10, num_samples)
            
            # Create a large figure with 10 rows and 2 columns
            fig, axes = plt.subplots(samples_to_plot, 2, figsize=(12, 4*samples_to_plot))
            
            # Handle single sample case
            if samples_to_plot == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(samples_to_plot):
                # Get data for this sample
                target_spectrum = target_spectrums[i]
                pred_spectrum = pred_spectrums[i]
                
                # 🔧 FIX: Improved color scaling for better visualization
                # Use logarithmic scale for better dynamic range visualization
                target_log = np.log10(target_spectrum + 1e-8)  # Add small epsilon to avoid log(0)
                pred_log = np.log10(pred_spectrum + 1e-8)
                
                # Find common color scale for log values
                vmin = min(target_log.min(), pred_log.min())
                vmax = max(target_log.max(), pred_log.max())
                
                # Use a more appropriate colormap for power spectra
                cmap = 'jet'  # Better for showing distinct peaks
                
                # Plot target spectrum (left column)
                im1 = axes[i, 0].imshow(target_log, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, origin='lower')
                axes[i, 0].set_title(f'Sample {i+1}: Ground Truth (log scale)')
                axes[i, 0].set_xlabel('Phi (azimuth) - 181 points, 2° res')
                axes[i, 0].set_ylabel('Theta (elevation) - 91 points, 1° res')
                
                # Set proper tick labels for high-resolution visualization
                # Phi axis: 0° to 360° with 181 points
                phi_ticks = np.linspace(0, 180, 7)  # Show every 60°
                phi_labels = ['0°', '60°', '120°', '180°', '240°', '300°', '360°']
                axes[i, 0].set_xticks(phi_ticks)
                axes[i, 0].set_xticklabels(phi_labels)
                
                # Theta axis: 0° to 90° with 91 points  
                theta_ticks = np.linspace(0, 90, 7)  # Show every 15°
                theta_labels = ['0°', '15°', '30°', '45°', '60°', '75°', '90°']
                axes[i, 0].set_yticks(theta_ticks)
                axes[i, 0].set_yticklabels(theta_labels)
                
                # Plot predicted spectrum (right column)
                im2 = axes[i, 1].imshow(pred_log, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, origin='lower')
                axes[i, 1].set_title(f'Sample {i+1}: Predicted (log scale)')
                axes[i, 1].set_xlabel('Phi (azimuth) - 181 points, 2° res')
                axes[i, 1].set_ylabel('Theta (elevation) - 91 points, 1° res')
                
                # Set same tick labels for predicted spectrum
                axes[i, 1].set_xticks(phi_ticks)
                axes[i, 1].set_xticklabels(phi_labels)
                axes[i, 1].set_yticks(theta_ticks)
                axes[i, 1].set_yticklabels(theta_labels)
                
                # Add colorbar to the right plot
                plt.colorbar(im2, ax=axes[i, 1])
                
                # Calculate and display metrics
                correlation = np.corrcoef(target_spectrum.flatten(), pred_spectrum.flatten())[0, 1]
                mse = np.mean((target_spectrum - pred_spectrum) ** 2)
                
                # Add metrics as text
                axes[i, 0].text(0.02, 0.98, f'Corr: {correlation:.3f}\\nMSE: {mse:.3f}', 
                               transform=axes[i, 0].transAxes, fontsize=10, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save the comparison plot
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            vis_save_path = plots_dir / 'spatial_spectrum_comparison.png'
            plt.savefig(vis_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Spatial spectrum visualization saved to: {vis_save_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create spatial spectrum visualization: {e}")
            import traceback
            traceback.print_exc()

    def _visualize_csi_comparison(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Visualize CSI amplitude and phase comparison for 5 random samples"""
        logger.info("📊 Generating CSI amplitude and phase comparison plots...")
        
        # Randomly select 5 samples
        num_samples = predictions.shape[0]
        if num_samples < 5:
            logger.warning(f"⚠️ Only {num_samples} samples available, using all samples")
            selected_indices = list(range(num_samples))
        else:
            selected_indices = np.random.choice(num_samples, 5, replace=False)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        fig.suptitle('CSI Amplitude and Phase Comparison (Predicted vs Target)', fontsize=16, fontweight='bold')
        
        for i, sample_idx in enumerate(selected_indices):
            # Get CSI data for this sample
            pred_csi = predictions[sample_idx]  # [num_antennas, num_subcarriers]
            target_csi = targets[sample_idx]   # [num_antennas, num_subcarriers]
            
            # Randomly select one antenna (same for both pred and target)
            num_antennas = pred_csi.shape[0]
            antenna_idx = np.random.randint(0, num_antennas)
            
            # Select CSI for the chosen antenna
            pred_csi_antenna = pred_csi[antenna_idx]  # [num_subcarriers]
            target_csi_antenna = target_csi[antenna_idx]  # [num_subcarriers]
            
            # Calculate amplitude and phase for the selected antenna
            pred_amp = torch.abs(pred_csi_antenna).cpu().numpy()
            target_amp = torch.abs(target_csi_antenna).cpu().numpy()
            pred_phase = torch.angle(pred_csi_antenna).cpu().numpy()
            target_phase = torch.angle(target_csi_antenna).cpu().numpy()
            
            # Subcarrier indices
            subcarrier_indices = np.arange(len(pred_amp))
            
            # Plot amplitude comparison
            axes[0, i].plot(subcarrier_indices, pred_amp, 'b-', label='Predicted', linewidth=2)
            axes[0, i].plot(subcarrier_indices, target_amp, 'r--', label='Target', linewidth=2)
            axes[0, i].set_title(f'Sample {sample_idx} (Antenna {antenna_idx})', fontsize=10, fontweight='bold')
            axes[0, i].set_ylabel('Amplitude', fontsize=10)
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend(fontsize=8)
            
            # Plot phase comparison
            axes[1, i].plot(subcarrier_indices, pred_phase, 'b-', label='Predicted', linewidth=2)
            axes[1, i].plot(subcarrier_indices, target_phase, 'r--', label='Target', linewidth=2)
            axes[1, i].set_xlabel('Subcarrier Index', fontsize=10)
            axes[1, i].set_ylabel('Phase (rad)', fontsize=10)
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend(fontsize=8)
            
            # Set y-axis limits for phase
            axes[1, i].set_ylim([-np.pi, np.pi])
        
        # Add row labels
        axes[0, 0].text(-0.1, 0.5, 'Amplitude', transform=axes[0, 0].transAxes, 
                        rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
        axes[1, 0].text(-0.1, 0.5, 'Phase', transform=axes[1, 0].transAxes, 
                        rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'plots' / 'csi_amplitude_phase_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 CSI amplitude and phase comparison plot saved: {plot_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test Prism Neural Network')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--model', help='Path to trained model file')
    parser.add_argument('--data', help='Path to test data file')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--gpu', type=int, help='GPU device ID to use')
    
    args = parser.parse_args()
    
    # Set GPU device if specified
    if args.gpu is not None:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            print(f"🔧 Using GPU device: {args.gpu}")
        else:
            print("⚠️  GPU requested but CUDA not available, using CPU")
    
    # Create and run tester
    try:
        tester = PrismTester(
            config_path=args.config,
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output,
            gpu_id=args.gpu
        )
        tester.test()
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
