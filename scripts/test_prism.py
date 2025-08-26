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
from typing import Dict, List, Tuple, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from prism.networks.prism_network import PrismNetwork
from prism.config_loader import ConfigLoader
from prism.ray_tracer_cpu import CPURayTracer
from prism.ray_tracer_cuda import CUDARayTracer
from prism.training_interface import PrismTrainingInterface

# Configure logging (will be updated after config loading)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Only console initially, file handler added after config load
    ]
)
logger = logging.getLogger(__name__)

class PrismTester:
    """Main tester class for Prism network using TrainingInterface"""
    
    def __init__(self, config_path: str, model_path: str = None, data_path: str = None, output_dir: str = None):
        """Initialize tester with configuration and optional model/data/output paths (will read from config if not provided)"""
        self.config_path = config_path
        
        # Load configuration first using ConfigLoader to process template variables
        try:
            config_loader = ConfigLoader(config_path)
            self.config = config_loader.config
        except Exception as e:
            print(f"❌ FATAL ERROR: Failed to load configuration from {config_path}")
            print(f"   Error details: {str(e)}")
            print(f"   Please check your configuration file and ensure it exists and is valid.")
            sys.exit(1)
        
        # Setup proper logging after config is loaded
        try:
            self._setup_logging()
        except Exception as e:
            print(f"❌ FATAL ERROR: Failed to setup logging")
            print(f"   Error details: {str(e)}")
            print(f"   Please check your logging configuration.")
            sys.exit(1)
        
        # Set model path, data path and output directory from config if not provided
        try:
            self.model_path = model_path or self.config['testing']['model_path']
            
            # Import data utilities
            from src.prism.data_utils import check_dataset_compatibility
            
            # Check dataset configuration
            dataset_path, split_config = check_dataset_compatibility(self.config)
            
            # Use single dataset with split
            self.data_path = data_path or dataset_path
            self.split_config = split_config
            logger.info(f"Using single dataset with train/test split: {self.data_path}")
            logger.info(f"Split configuration: {self.split_config}")
                
        except KeyError as e:
            logger.error(f"❌ FATAL ERROR: Missing required configuration key: {e}")
            print(f"❌ FATAL ERROR: Configuration is missing required key: {e}")
            print(f"   Please check your configuration file structure.")
            sys.exit(1)
        
        # Build output directory from base_dir (new simplified structure)
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Use base_dir + testing to construct output directory
            base_dir = self.config['output'].get('base_dir', 'results')
            self.output_dir = Path(base_dir) / 'testing'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        
        logger.info(f"Loaded configuration from {config_path}")
        
        # Setup device with intelligent GPU selection
        device_config = self.config.get('system', {}).get('device', 'cuda')
        if device_config == 'cuda' and torch.cuda.is_available():
            selected_gpu = self._select_best_gpu()
            self.device = torch.device(f'cuda:{selected_gpu}')
            logger.info(f"🔍 GPU Auto-Selection for Testing:")
            logger.info(f"  • Selected GPU: {selected_gpu}")
            logger.info(f"  • GPU Name: {torch.cuda.get_device_name(selected_gpu)}")
            logger.info(f"  • GPU Memory: {torch.cuda.get_device_properties(selected_gpu).total_memory / 1024**3:.1f}GB")
        else:
            self.device = torch.device('cpu')
            if device_config == 'cuda':
                logger.warning("CUDA requested but not available, using CPU")
            else:
                logger.info("Using CPU as configured")
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and data
        self._load_model()
        self._load_data()
    
    def _select_best_gpu(self) -> int:
        """智能选择最佳可用GPU (与训练脚本相同的逻辑)"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            return 0
        
        logger.info("🔍 Scanning available GPUs for testing...")
        
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
                
                # 计算GPU利用率分数
                memory_usage_ratio = reserved_memory / total_memory
                score = total_memory * (1 - memory_usage_ratio)
                
                gpu_info.append({
                    'id': i,
                    'name': name,
                    'total_memory': total_memory,
                    'free_memory': free_memory,
                    'usage_ratio': memory_usage_ratio,
                    'score': score
                })
                
                logger.info(f"  GPU {i}: {name} ({total_memory:.1f}GB, {free_memory:.1f}GB free)")
                
            except Exception as e:
                logger.warning(f"  GPU {i}: Error getting info - {e}")
                gpu_info.append({'id': i, 'score': -1})
        
        # 选择得分最高的GPU
        best_gpu = max(gpu_info, key=lambda x: x['score'])
        selected_id = best_gpu['id']
        
        logger.info(f"✅ Selected GPU {selected_id} for testing")
        return selected_id
    
    def _setup_logging(self):
        """Setup logging with proper file path from config"""
        # Get logging configuration from config
        output_config = self.config.get('output', {})
        logging_config = output_config.get('logging', {})
        log_level_str = logging_config.get('log_level', 'INFO')
        
        # Get log file path from config (should always be available after ConfigLoader processing)
        log_file = logging_config.get('log_file')
        if not log_file:
            raise ValueError("log_file not found in configuration. Check your config file and ConfigLoader processing.")
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Convert string log level to logging constant
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
        # Add file handler to existing logger
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(log_level)
        
        logger.info(f"Logging setup complete. Log file: {log_file}")
        
    def _load_config(self):
        """Load configuration from YAML file using ConfigLoader"""
        try:
            config_loader = ConfigLoader(self.config_path)
            config = config_loader.config
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: Failed to load configuration from {self.config_path}")
            logger.error(f"   Error details: {str(e)}")
            print(f"❌ FATAL ERROR: Configuration loading failed")
            sys.exit(1)
    
    def _load_model(self):
        """Load trained Prism network model from TrainingInterface checkpoint"""
        logger.info(f"Loading TrainingInterface model from {self.model_path}")
        
        # Check if this is a TrainingInterface checkpoint
        if ('checkpoint_epoch_' in str(self.model_path) or 'best_model.pt' in str(self.model_path) or 
            'latest_checkpoint.pt' in str(self.model_path) or 'latest_batch_checkpoint.pt' in str(self.model_path)):
            # This is a TrainingInterface checkpoint
            self._load_training_interface_checkpoint()
        else:
            # This is a legacy checkpoint, try to load it
            self._load_legacy_checkpoint()
    
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
            
            # Get ray tracing execution settings from system config
            ray_tracing_mode = system_config.get('ray_tracing_mode', 'cuda')
            fallback_to_cpu = system_config.get('fallback_to_cpu', True)
            
            logger.info(f"Ray tracer configuration:")
            logger.info(f"  - Ray tracing mode: {ray_tracing_mode}")
            logger.info(f"  - CUDA available: {torch.cuda.is_available()}")
            logger.info(f"  - Fallback to CPU: {fallback_to_cpu}")
            
            # Calculate max ray length from scene bounds
            def calculate_max_ray_length(scene_bounds):
                """Calculate maximum ray length from scene bounds"""
                if 'min' in scene_bounds and 'max' in scene_bounds:
                    import numpy as np
                    min_bounds = np.array(scene_bounds['min'])
                    max_bounds = np.array(scene_bounds['max'])
                    # Calculate diagonal distance of the scene
                    diagonal = np.linalg.norm(max_bounds - min_bounds)
                    # Add some margin for safety
                    return diagonal * 1.2
                else:
                    # Fallback to default if scene bounds not properly configured
                    return 200.0
            
            # Get sampling configurations
            angular_sampling = ray_tracing_config.get('angular_sampling', {})
            spatial_sampling = ray_tracing_config.get('spatial_sampling', {})
            scene_bounds = ray_tracing_config.get('scene_bounds', {})
            max_ray_length = ray_tracing_config.get('max_ray_length', calculate_max_ray_length(scene_bounds))
            
            logger.info(f"📏 Calculated max_ray_length: {max_ray_length:.1f}m from scene bounds")
            
            # Create ray tracer based on configuration
            if ray_tracing_mode == 'cuda' and torch.cuda.is_available():
                logger.info("🚀 Using CUDA-accelerated ray tracer")
                mixed_precision_enabled = system_config.get('mixed_precision', {}).get('enabled', True)
                
                self.ray_tracer = CUDARayTracer(
                    azimuth_divisions=angular_sampling.get('azimuth_divisions', 18),
                    elevation_divisions=angular_sampling.get('elevation_divisions', 9),
                    max_ray_length=max_ray_length,
                    prism_network=self.prism_network,
                    signal_threshold=ray_tracing_config.get('signal_threshold', 1e-6),
                    enable_early_termination=ray_tracing_config.get('enable_early_termination', True),
                    uniform_samples=spatial_sampling.get('num_sampling_points', 64),
                    resampled_points=spatial_sampling.get('resampled_points', 32),
                    use_mixed_precision=mixed_precision_enabled
                )
            else:
                if ray_tracing_mode == 'cuda' and not torch.cuda.is_available():
                    if fallback_to_cpu:
                        logger.warning("⚠️  CUDA ray tracer requested but CUDA not available, falling back to CPU")
                    else:
                        raise RuntimeError("CUDA ray tracer requested but CUDA not available and fallback disabled")
                
                logger.info("💻 Using CPU ray tracer")
                cpu_config = system_config.get('cpu', {})
                
                self.ray_tracer = CPURayTracer(
                    azimuth_divisions=angular_sampling.get('azimuth_divisions', 18),
                    elevation_divisions=angular_sampling.get('elevation_divisions', 9),
                    max_ray_length=max_ray_length,
                    scene_bounds=ray_tracing_config.get('scene_bounds'),
                    prism_network=self.prism_network,
                    signal_threshold=ray_tracing_config.get('signal_threshold', 1e-6),
                    enable_early_termination=ray_tracing_config.get('enable_early_termination', True),
                    top_k_directions=angular_sampling.get('top_k_directions', 32),
                    max_workers=cpu_config.get('num_workers', 4),
                    uniform_samples=spatial_sampling.get('num_sampling_points', 64),
                    resampled_points=spatial_sampling.get('resampled_points', 32)
                )
            
            # Get checkpoint directory from config
            checkpoint_dir = self.config['output']['training']['checkpoint_dir']
            
            # Create TrainingInterface with simplified parameters
            self.model = PrismTrainingInterface(
                prism_network=self.prism_network,
                ray_tracer=self.ray_tracer,
                ray_tracing_config=ray_tracing_config,
                system_config=system_config,
                checkpoint_dir=checkpoint_dir
            )
            
            # Pass full config for logging fallback
            self.model._full_config = self.config
            
            # Load the checkpoint
            self.model.load_checkpoint(self.model_path)
            self.model = self.model.to(self.device)
            
            # Ensure ray tracer uses the same device as the model
            if hasattr(self.ray_tracer, 'device'):
                self.ray_tracer.device = self.device.type
                logger.info(f"Set ray tracer device to: {self.device.type}")
            
            # Also set CUDA device if available
            if torch.cuda.is_available():
                torch.cuda.set_device(self.device)
                logger.info(f"Set CUDA device to: {self.device}")
            
            self.model.eval()
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"TrainingInterface model loaded with {total_params:,} parameters")
            
            # Store checkpoint info from TrainingInterface
            training_info = self.model.get_training_info()
            self.checkpoint_info = {
                'epoch': training_info['current_epoch'],
                'best_loss': training_info['best_loss'],
                'training_history_length': training_info['training_history_length']
            }
            logger.info(f"TrainingInterface checkpoint info: {self.checkpoint_info}")
            
        except Exception as e:
            logger.error(f"Failed to load TrainingInterface checkpoint: {e}")
            raise
    
    def _load_legacy_checkpoint(self):
        """Load legacy checkpoint format"""
        logger.info("Attempting to load legacy checkpoint format...")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract configuration from checkpoint
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            logger.info("Using configuration from checkpoint")
        else:
            checkpoint_config = self.config
            logger.info("Using configuration from YAML file")
        
        # Create model with updated configuration mapping
        nn_config = checkpoint_config['neural_networks']
        self.model = PrismNetwork(
            num_subcarriers=nn_config['attenuation_decoder']['output_dim'],
            num_ue_antennas=nn_config['attenuation_decoder'].get('num_ue_antennas', 
                           nn_config['attenuation_decoder'].get('num_ue', 4)),  # Handle both old and new names
            num_bs_antennas=nn_config['antenna_codebook']['num_antennas'],
            position_dim=nn_config['attenuation_network']['input_dim'],
            hidden_dim=nn_config['attenuation_network']['hidden_dim'],
            feature_dim=nn_config['attenuation_network']['feature_dim'],
            antenna_embedding_dim=nn_config['antenna_codebook']['embedding_dim'],
            use_antenna_codebook=nn_config['antenna_codebook']['learnable'],
            use_ipe_encoding=True,
            azimuth_divisions=checkpoint_config['ray_tracing']['angular_sampling']['azimuth_divisions'],
            elevation_divisions=checkpoint_config['ray_tracing']['angular_sampling']['elevation_divisions'],
            top_k_directions=checkpoint_config['ray_tracing']['angular_sampling']['top_k_directions'],
            complex_output=True
        )
        
        # Load model weights
        # Load model state dict - handle potential prefix mismatch
        model_state_dict = checkpoint['model_state_dict']
        
        # Check if the checkpoint has 'prism_network.' prefix but our model doesn't
        if any(key.startswith('prism_network.') for key in model_state_dict.keys()):
            logger.info("Detected 'prism_network.' prefix in checkpoint, removing prefix...")
            # Remove 'prism_network.' prefix from all keys
            new_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith('prism_network.'):
                    new_key = key[len('prism_network.'):]  # Remove the prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            model_state_dict = new_state_dict
        
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Legacy model loaded with {total_params:,} parameters")
        
        # Store checkpoint info
        self.checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'train_loss': checkpoint.get('train_loss', 'Unknown'),
            'val_loss': checkpoint.get('val_loss', 'Unknown')
        }
        logger.info(f"Legacy checkpoint info: {self.checkpoint_info}")
        
    def _load_data(self):
        """Load test data from HDF5 file"""
        logger.info(f"Loading test data from {self.data_path}")
        
        # Use split-based data loading
        from src.prism.data_utils import load_and_split_data
        
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
        
    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate model performance on test data using TrainingInterface"""
        logger.info("Evaluating model performance...")
        
        self.model.eval()
        predictions = []
        losses = []
        
        # Process in batches to avoid memory issues (use config setting)
        batch_size = self.config['testing']['batch_size']
        num_samples = len(self.ue_positions)
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_ue_pos = self.ue_positions[i:end_idx]
                batch_bs_pos = self.bs_position.expand(end_idx - i, -1)
                batch_antenna_idx = self.antenna_indices.expand(end_idx - i, -1)
                batch_csi_target = self.csi_target[i:end_idx]
                
                try:
                    if hasattr(self.model, 'forward') and callable(getattr(self.model, 'forward')):
                        # This is a TrainingInterface
                        outputs = self.model(
                            ue_positions=batch_ue_pos,
                            bs_position=batch_bs_pos,
                            antenna_indices=batch_antenna_idx
                        )
                        batch_predictions = outputs['csi_predictions']
                    else:
                        # This is a legacy PrismNetwork
                        batch_predictions = self.model(batch_ue_pos)
                    
                    # Compute loss
                    if hasattr(self.model, 'compute_loss'):
                        # TrainingInterface loss computation
                        batch_loss = self.model.compute_loss(batch_predictions, batch_csi_target, nn.MSELoss())
                    else:
                        # Legacy loss computation
                        if batch_predictions.dtype == torch.complex64:
                            batch_predictions_real = torch.cat([batch_predictions.real, batch_predictions.imag], dim=-1)
                            batch_csi_target_real = torch.cat([batch_csi_target.real, batch_csi_target.imag], dim=-1)
                            batch_loss = nn.MSELoss()(batch_predictions_real, batch_csi_target_real)
                        else:
                            batch_loss = nn.MSELoss()(batch_predictions, batch_csi_target)
                    
                    predictions.append(batch_predictions.cpu())
                    losses.append(batch_loss.item())
                    
                except Exception as e:
                    logger.error(f"Error in batch {i//batch_size}: {e}")
                    continue
        
        if not predictions:
            raise RuntimeError("No successful predictions made")
        
        # Concatenate all predictions
        self.predictions = torch.cat(predictions, dim=0)
        avg_loss = np.mean(losses)
        
        logger.info(f"Evaluation completed. Average loss: {avg_loss:.6f}")
        return {'avg_loss': avg_loss}
    
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive performance metrics"""
        logger.info("Computing performance metrics...")
        
        # Ensure predictions and targets have the same shape
        if self.predictions.shape != self.csi_target.shape:
            logger.warning(f"Shape mismatch: predictions {self.predictions.shape} vs targets {self.csi_target.shape}")
            # Try to reshape if possible
            if len(self.predictions.shape) == len(self.csi_target.shape):
                min_shape = tuple(min(p, t) for p, t in zip(self.predictions.shape, self.csi_target.shape))
                self.predictions = self.predictions[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
                self.csi_target = self.csi_target[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
            else:
                logger.error("Cannot reshape predictions to match targets")
                return {}
        
        # Convert to numpy for metric computation
        pred_np = self.predictions.cpu().numpy()
        target_np = self.csi_target.cpu().numpy()
        
        # Complex MSE
        complex_mse = np.mean(np.abs(pred_np - target_np) ** 2)
        
        # Magnitude error
        pred_magnitude = np.abs(pred_np)
        target_magnitude = np.abs(target_np)
        magnitude_error = np.mean(np.abs(pred_magnitude - target_magnitude) ** 2)
        
        # Phase error
        pred_phase = np.angle(pred_np)
        target_phase = np.angle(target_np)
        phase_error = np.mean(np.abs(pred_phase - target_phase) ** 2)
        
        # Correlation coefficient
        pred_flat = pred_np.flatten()
        target_flat = target_np.flatten()
        correlation = np.corrcoef(pred_flat.real, target_flat.real)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # NMSE (Normalized Mean Square Error)
        nmse = complex_mse / (np.mean(np.abs(target_np) ** 2) + 1e-8)
        
        # SNR (Signal-to-Noise Ratio)
        signal_power = np.mean(np.abs(target_np) ** 2)
        noise_power = complex_mse
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        metrics = {
            'complex_mse': float(complex_mse),
            'magnitude_error': float(magnitude_error),
            'phase_error': float(phase_error),
            'correlation': float(correlation),
            'nmse': float(nmse),
            'snr_db': float(snr)
        }
        
        logger.info("Metrics computed:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.6f}")
        
        return metrics
    
    def _visualize_results(self):
        """Create visualizations of test results"""
        logger.info("Creating visualizations...")
        
        # 1. CSI Magnitude Comparison
        self._plot_csi_magnitude_comparison()
        
        # 2. CSI Phase Comparison
        self._plot_csi_phase_comparison()
        
        # 3. Error Distribution
        self._plot_error_distribution()
        
        # 4. Spatial Performance Map
        self._plot_spatial_performance()
        
        # 5. Subcarrier Performance
        self._plot_subcarrier_performance()
        
        logger.info("Visualizations completed")
    
    def _plot_csi_magnitude_comparison(self):
        """Plot CSI magnitude comparison between prediction and target"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample a few UE positions for visualization
        sample_indices = np.random.choice(len(self.ue_positions), min(4, len(self.ue_positions)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            pred_mag = np.abs(self.predictions[idx].numpy())
            target_mag = np.abs(self.csi_target[idx].cpu().numpy())
            
            # Handle multi-dimensional data by averaging over UE and BS antennas
            if pred_mag.ndim > 1:
                pred_mag = np.mean(pred_mag, axis=tuple(range(1, pred_mag.ndim)))
            if target_mag.ndim > 1:
                target_mag = np.mean(target_mag, axis=tuple(range(1, target_mag.ndim)))
            
            subcarriers = range(len(pred_mag))
            ax.plot(subcarriers, target_mag, 'b-', label='Target', linewidth=2)
            ax.plot(subcarriers, pred_mag, 'r--', label='Prediction', linewidth=2)
            
            ax.set_xlabel('Subcarrier Index')
            ax.set_ylabel('CSI Magnitude')
            ax.set_title(f'UE {idx}: CSI Magnitude Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots_dir = Path(self.config['output']['testing']['plots_dir'])
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / 'csi_magnitude_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"CSI magnitude comparison plot saved: {plot_path}")
    
    def _plot_csi_phase_comparison(self):
        """Plot CSI phase comparison between prediction and target"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        sample_indices = np.random.choice(len(self.ue_positions), min(4, len(self.ue_positions)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            pred_phase = np.angle(self.predictions[idx].numpy())
            target_phase = np.angle(self.csi_target[idx].cpu().numpy())
            
            # Handle multi-dimensional data by averaging over UE and BS antennas
            if pred_phase.ndim > 1:
                pred_phase = np.mean(pred_phase, axis=tuple(range(1, pred_phase.ndim)))
            if target_phase.ndim > 1:
                target_phase = np.mean(target_phase, axis=tuple(range(1, target_phase.ndim)))
            
            subcarriers = range(len(pred_phase))
            ax.plot(subcarriers, target_phase, 'b-', label='Target', linewidth=2)
            ax.plot(subcarriers, pred_phase, 'r--', label='Prediction', linewidth=2)
            
            ax.set_xlabel('Subcarrier Index')
            ax.set_ylabel('CSI Phase (radians)')
            ax.set_title(f'UE {idx}: CSI Phase Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots_dir = Path(self.config['output']['testing']['plots_dir'])
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / 'csi_phase_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"CSI phase comparison plot saved: {plot_path}")
    
    def _plot_error_distribution(self):
        """Plot error distribution for magnitude and phase"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Magnitude error
        pred_mag = np.abs(self.predictions.numpy())
        target_mag = np.abs(self.csi_target.cpu().numpy())
        mag_error = pred_mag - target_mag
        
        axes[0].hist(mag_error.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('Magnitude Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Magnitude Error Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Phase error
        pred_phase = np.angle(self.predictions.numpy())
        target_phase = np.angle(self.csi_target.cpu().numpy())
        phase_diff = pred_phase - target_phase
        phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrap to [-π, π]
        
        axes[1].hist(phase_diff.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1].set_xlabel('Phase Error (radians)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Phase Error Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots_dir = Path(self.config['output']['testing']['plots_dir'])
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / 'error_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Error distribution plot saved: {plot_path}")
    
    def _plot_spatial_performance(self):
        """Plot spatial performance map showing error by UE position"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate error per UE position
        pred_np = self.predictions.numpy()
        target_np = self.csi_target.cpu().numpy()
        
        # Magnitude error per UE
        mag_error_per_ue = np.mean(np.abs(np.abs(pred_np) - np.abs(target_np)), axis=1)
        
        # Phase error per UE
        pred_phase = np.angle(pred_np)
        target_phase = np.angle(target_np)
        phase_diff = pred_phase - target_phase
        phase_diff = np.angle(np.exp(1j * phase_diff))
        phase_error_per_ue = np.mean(np.abs(phase_diff), axis=1)
        
        # Get UE positions
        ue_pos_np = self.ue_positions.cpu().numpy()
        
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
        plots_dir = Path(self.config['output']['testing']['plots_dir'])
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / 'spatial_performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Spatial performance plot saved: {plot_path}")
    
    def _plot_subcarrier_performance(self):
        """Plot performance across subcarriers"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        pred_np = self.predictions.numpy()
        target_np = self.csi_target.cpu().numpy()
        
        # Calculate error per subcarrier
        mag_error_per_subcarrier = np.mean(np.abs(np.abs(pred_np) - np.abs(target_np)), axis=0)
        phase_error_per_subcarrier = np.mean(np.abs(np.angle(pred_np) - np.angle(target_np)), axis=0)
        
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
        plots_dir = Path(self.config['output']['testing']['plots_dir'])
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / 'subcarrier_performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Subcarrier performance plot saved: {plot_path}")
    
    def _save_results(self, metrics: Dict[str, float]):
        """Save test results and metrics"""
        logger.info("Saving test results...")
        
        # Save metrics
        results = {
            'test_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'data_path': self.data_path,
            'checkpoint_info': self.checkpoint_info,
            'metrics': metrics,
            'simulation_parameters': self.sim_params
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
    
    def test(self):
        """Main testing function"""
        logger.info("Starting Prism network testing...")
        
        try:
            # Evaluate model
            metrics = self._evaluate_model()
            
            # Create visualizations
            self._visualize_results()
            
            # Save results
            self._save_results(metrics)
            
            logger.info("Testing completed successfully!")
            
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test Prism Network')
    parser.add_argument('--config', type=str, default='configs/ofdm-5g-sionna.yml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model checkpoint (optional, will read from config if not provided)')


    
    args = parser.parse_args()
    
    # Create tester and start testing (model path, data path and output directory from config if not provided)
    tester = PrismTester(args.config, args.model, None, None)
    tester.test()

if __name__ == '__main__':
    main()
