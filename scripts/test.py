#!/usr/bin/env python3
"""
Generic Prism Network Testing Script

This script tests the trained Prism neural network for electromagnetic ray tracing.
It can load different datasets and configurations based on the provided config file.

Features:
- Generic data loading for different dataset formats (Sionna, PolyU, Chrissy)
- Automatic dataset format detection and adaptation
- Modern configuration loading with template processing
- CSI-focused testing metrics (magnitude, phase, CSI loss)
- Model loading from checkpoints
- Real-time progress monitoring
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
from base_runner import BaseRunner

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


class PrismTester(BaseRunner):
    """Generic tester class for Prism network using modern configuration
    
    This tester can handle different dataset formats and configurations:
    - Sionna: Ray tracing simulation data
    - PolyU: WiFi measurement data with BS position padding
    - Chrissy: Custom dataset format
    """
    
    def __init__(self, config_path: str, model_path: str = None, data_path: str = None, output_dir: str = None, gpu_id: int = None):
        """Initialize tester with configuration and optional model/data/output paths"""
        self.gpu_id = gpu_id  # Store GPU ID for device setup
        
        # Initialize base runner
        super().__init__(config_path)
        
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
        
        # Initialize CSI loss for testing
        self._init_csi_loss()
    
    
    def _init_csi_loss(self):
        """Initialize CSI loss for testing metrics"""
        self.csi_loss = None
        
        try:
            from prism.loss.csi_loss import CSILoss
            
            # Initialize CSI loss with default configuration
            self.csi_loss = CSILoss(
                phase_weight=1.0,
                magnitude_weight=1.0,
                normalize_weights=True
            ).to(self.device)
            
            logger.info("✅ CSI loss initialized for testing")
                
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize CSI loss: {e}")
            self.csi_loss = None
    
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
                    'num_ue_antennas': self.config_loader.user_equipment.num_ue_antennas,
                    'ue_antenna_count': self.config_loader.user_equipment.ue_antenna_count
                },
                'input': {
                    'subcarrier_sampling': {
                        'sampling_ratio': data_config['sampling_ratio'],
                        'sampling_method': data_config['sampling_method'],
                        'antenna_consistent': data_config['antenna_consistent']
                    }
                },
                'loss_functions': self.config_loader.training.loss.__dict__,
                'neural_networks': {
                    'prism_network': self.config_loader.prism_network.__dict__,
                    'attenuation_network': self.config_loader.attenuation_network.__dict__,
                    'radiance_network': self.config_loader.radiance_network.__dict__,
                    'antenna_codebook': self.config_loader.antenna_codebook.__dict__,
                },
                'output': self.config_loader.output.__dict__,
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
                    logger.warning("⚠️  Secure loading failed (expected for complex checkpoints)")
                    logger.info("🔓 Using compatibility mode (weights_only=False) - this is normal for Prism checkpoints")
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
        
        # Use base class data loading method
        ue_positions, bs_positions, antenna_indices, csi_data = super()._load_data()
        
        # Prepare test split for testing
        self._prepare_test_split(ue_positions, bs_positions, antenna_indices, csi_data)
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
                        bs_antenna_indices=batch_antenna_idx
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
            
            # Apply phase calibration to both predictions and targets
            logger.info("🔧 Applying phase calibration to CSI data...")
            predictions_calibrated = self.model._apply_phase_calibration_to_batch(all_predictions)
            targets_calibrated = self.model._apply_phase_calibration_to_batch(all_targets)
            
            logger.info("✅ Phase calibration completed")
            logger.info(f"   Original predictions shape: {all_predictions.shape}")
            logger.info(f"   Calibrated predictions shape: {predictions_calibrated.shape}")
            logger.info(f"   Original targets shape: {all_targets.shape}")
            logger.info(f"   Calibrated targets shape: {targets_calibrated.shape}")
            
            # Calculate metrics using calibrated CSI
            self._calculate_metrics(predictions_calibrated, targets_calibrated)
            
            # Save both original and calibrated results
            self._save_results(all_predictions, all_targets, predictions_calibrated, targets_calibrated)
            
            
            logger.info("✅ Testing completed successfully")
        else:
            logger.error("❌ No successful predictions were made")
    
    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Calculate CSI-focused testing metrics using phase-calibrated CSI"""
        logger.info("📊 Calculating CSI testing metrics...")
        
        # Verify tensor shapes match
        if predictions.shape != targets.shape:
            logger.warning(f"⚠️ Shape mismatch: pred {predictions.shape} vs target {targets.shape}")
        
        # 1. Magnitude and Phase extraction
        pred_mag = torch.abs(predictions)
        target_mag = torch.abs(targets)
        pred_phase = torch.angle(predictions)  # Range: [-π, π]
        target_phase = torch.angle(targets)    # Range: [-π, π]
        
        # 2. Magnitude MSE and MAE
        magnitude_mse = torch.nn.functional.mse_loss(pred_mag, target_mag).item()
        magnitude_mae = torch.nn.functional.l1_loss(pred_mag, target_mag).item()
        
        # 3. Phase MSE with wrapping correction
        phase_diff = pred_phase - target_phase
        phase_diff_wrapped = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        phase_mse_wrapped = torch.mean(phase_diff_wrapped ** 2).item()
        phase_mae_wrapped = torch.mean(torch.abs(phase_diff_wrapped)).item()
        
        # 4. CSI Loss
        csi_loss_value = None
        if self.csi_loss is not None:
            try:
                # Move tensors to device and compute CSI loss
                pred_device = predictions.to(self.device)
                target_device = targets.to(self.device)
                
                with torch.no_grad():
                    csi_loss_value = self.csi_loss(pred_device, target_device).item()
                    
                logger.info(f"✅ CSI loss computed: {csi_loss_value:.8f}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to compute CSI loss: {e}")
                csi_loss_value = None
        
        # 5. Statistical analysis
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
        
        # Log CSI-focused metrics
        logger.info(f"📈 CSI Testing Metrics:")
        logger.info(f"   ═══════════════════════════════════════")
        logger.info(f"   📏 Magnitude MSE:         {magnitude_mse:.8f}")
        logger.info(f"   📏 Magnitude MAE:         {magnitude_mae:.8f}")
        logger.info(f"   ═══════════════════════════════════════")
        logger.info(f"   🌀 Phase MSE (wrapped):   {phase_mse_wrapped:.8f}")
        logger.info(f"   🌀 Phase MAE (wrapped):   {phase_mae_wrapped:.8f}")
        logger.info(f"   ═══════════════════════════════════════")
        if csi_loss_value is not None:
            logger.info(f"   🎯 CSI Loss:              {csi_loss_value:.8f}")
        else:
            logger.info(f"   🎯 CSI Loss:              Not computed")
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
        
        # Save CSI-focused metrics
        metrics = {
            'magnitude_mse': magnitude_mse,
            'magnitude_mae': magnitude_mae,
            'phase_mse_wrapped': phase_mse_wrapped,
            'phase_mae_wrapped': phase_mae_wrapped,
            'csi_loss': csi_loss_value,
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
    
    def _save_results(self, predictions: torch.Tensor, targets: torch.Tensor, 
                      predictions_calibrated: torch.Tensor = None, targets_calibrated: torch.Tensor = None):
        """Save testing results including both original and calibrated CSI"""
        logger.info("💾 Saving testing results...")
        
        results_file = self.output_dir / 'predictions' / 'test_results.npz'
        
        # Prepare data dictionary
        save_data = {
            'predictions': predictions.cpu().numpy(),
            'targets': targets.cpu().numpy(),
            'test_ue_positions': self.test_ue_positions.cpu().numpy(),
            'test_bs_positions': self.test_bs_positions.cpu().numpy()
        }
        
        # Add calibrated CSI if provided
        if predictions_calibrated is not None and targets_calibrated is not None:
            save_data.update({
                'predictions_calibrated': predictions_calibrated.cpu().numpy(),
                'targets_calibrated': targets_calibrated.cpu().numpy()
            })
            logger.info("💾 Saving both original and calibrated CSI data")
        else:
            logger.info("💾 Saving original CSI data only")
        
        np.savez_compressed(results_file, **save_data)
        
        logger.info(f"💾 Results saved to: {results_file}")
        logger.info(f"   File contains: {list(save_data.keys())}")
    
    # End of Selection


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
