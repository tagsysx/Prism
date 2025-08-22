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
from prism.ray_tracer import DiscreteRayTracer
from prism.training_interface import PrismTrainingInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PrismTester:
    """Main tester class for Prism network using TrainingInterface"""
    
    def __init__(self, config_path: str, model_path: str, data_path: str, output_dir: str):
        """Initialize tester with configuration, model, and data paths"""
        self.config_path = config_path
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and data
        self._load_model()
        self._load_data()
        
    def _load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
    
    def _load_model(self):
        """Load trained Prism network model from TrainingInterface checkpoint"""
        logger.info(f"Loading TrainingInterface model from {self.model_path}")
        
        # Check if this is a TrainingInterface checkpoint
        if 'checkpoint_epoch_' in str(self.model_path) or 'best_model.pt' in str(self.model_path) or 'latest_checkpoint.pt' in str(self.model_path):
            # This is a TrainingInterface checkpoint
            self._load_training_interface_checkpoint()
        else:
            # This is a legacy checkpoint, try to load it
            self._load_legacy_checkpoint()
    
    def _load_training_interface_checkpoint(self):
        """Load TrainingInterface checkpoint"""
        try:
            # Create PrismNetwork and DiscreteRayTracer first
            nn_config = self.config['neural_networks']
            rt_config = self.config['ray_tracing']
            
            self.prism_network = PrismNetwork(
                num_subcarriers=nn_config['attenuation_decoder']['output_dim'],
                num_ue_antennas=nn_config['attenuation_decoder']['num_ue'],
                num_bs_antennas=nn_config['antenna_codebook']['num_antennas'],
                position_dim=nn_config['attenuation_network']['input_dim'],
                hidden_dim=nn_config['attenuation_network']['hidden_dim'],
                feature_dim=nn_config['attenuation_network']['feature_dim'],
                antenna_embedding_dim=nn_config['antenna_codebook']['embedding_dim'],
                use_antenna_codebook=nn_config['antenna_codebook']['learnable'],
                use_ipe_encoding=True,
                azimuth_divisions=rt_config['azimuth_divisions'],
                elevation_divisions=rt_config['elevation_divisions'],
                top_k_directions=32,
                complex_output=True
            )
            
            self.ray_tracer = DiscreteRayTracer(
                scene_bounds=rt_config.get('scene_bounds', None),
                angular_divisions=(rt_config['azimuth_divisions'], rt_config['elevation_divisions']),
                spatial_sampling=rt_config.get('spatial_sampling', 64),
                gpu_acceleration=rt_config.get('gpu_acceleration', True)
            )
            
            # Create TrainingInterface
            self.model = PrismTrainingInterface(
                prism_network=self.prism_network,
                ray_tracer=self.ray_tracer,
                num_sampling_points=rt_config.get('spatial_sampling', 64),
                scene_bounds=rt_config.get('scene_bounds', None),
                subcarrier_sampling_ratio=rt_config.get('subcarrier_sampling_ratio', 0.3),
                checkpoint_dir=str(self.output_dir / 'temp_checkpoints')
            )
            
            # Load the checkpoint
            self.model.load_checkpoint(self.model_path)
            self.model = self.model.to(self.device)
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
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract configuration from checkpoint
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            logger.info("Using configuration from checkpoint")
        else:
            checkpoint_config = self.config
            logger.info("Using configuration from YAML file")
        
        # Create model with same configuration
        nn_config = checkpoint_config['neural_networks']
        self.model = PrismNetwork(
            num_subcarriers=nn_config['attenuation_decoder']['output_dim'],
            num_ue_antennas=nn_config['attenuation_decoder']['num_ue'],
            num_bs_antennas=nn_config['antenna_codebook']['num_antennas'],
            position_dim=nn_config['attenuation_network']['input_dim'],
            hidden_dim=nn_config['attenuation_network']['hidden_dim'],
            feature_dim=nn_config['attenuation_network']['feature_dim'],
            antenna_embedding_dim=nn_config['antenna_codebook']['embedding_dim'],
            use_antenna_codebook=nn_config['antenna_codebook']['learnable'],
            use_ipe_encoding=True,
            azimuth_divisions=checkpoint_config['ray_tracing']['azimuth_divisions'],
            elevation_divisions=checkpoint_config['ray_tracing']['elevation_divisions'],
            top_k_directions=32,
            complex_output=True
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
        
        with h5py.File(self.data_path, 'r') as f:
            # Load UE positions
            self.ue_positions = torch.tensor(f['ue_positions'][:], dtype=torch.float32)
            logger.info(f"Loaded {len(self.ue_positions)} UE positions")
            
            # Load channel responses (CSI)
            self.csi_target = torch.tensor(f['channel_responses'][:], dtype=torch.complex64)
            logger.info(f"Loaded CSI data with shape: {self.csi_target.shape}")
            
            # Load BS position
            self.bs_position = torch.tensor(f['bs_position'][:], dtype=torch.float32)
            logger.info(f"BS position: {self.bs_position}")
            
            # Load antenna indices if available
            if 'antenna_indices' in f:
                self.antenna_indices = torch.tensor(f['antenna_indices'][:], dtype=torch.long)
                logger.info(f"Loaded antenna indices with shape: {self.antenna_indices.shape}")
            else:
                # Create default antenna indices if not available
                num_bs_antennas = self.csi_target.shape[1] if len(self.csi_target.shape) > 1 else 1
                self.antenna_indices = torch.arange(num_bs_antennas)
                logger.info(f"Created default antenna indices: {self.antenna_indices}")
            
            # Load simulation parameters
            self.sim_params = dict(f['simulation_params'].attrs)
            logger.info(f"Simulation parameters: {self.sim_params}")
        
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
        
        # Process in batches to avoid memory issues
        batch_size = 64
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
            
            subcarriers = range(len(pred_mag))
            ax.plot(subcarriers, target_mag, 'b-', label='Target', linewidth=2)
            ax.plot(subcarriers, pred_mag, 'r--', label='Prediction', linewidth=2)
            
            ax.set_xlabel('Subcarrier Index')
            ax.set_ylabel('CSI Magnitude')
            ax.set_title(f'UE {idx}: CSI Magnitude Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'csi_magnitude_comparison.png'
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
            
            subcarriers = range(len(pred_phase))
            ax.plot(subcarriers, target_phase, 'b-', label='Target', linewidth=2)
            ax.plot(subcarriers, pred_phase, 'r--', label='Prediction', linewidth=2)
            
            ax.set_xlabel('Subcarrier Index')
            ax.set_ylabel('CSI Phase (radians)')
            ax.set_title(f'UE {idx}: CSI Phase Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'csi_phase_comparison.png'
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
        plot_path = self.output_dir / 'error_distribution.png'
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
        plot_path = self.output_dir / 'spatial_performance.png'
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
        plot_path = self.output_dir / 'subcarrier_performance.png'
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
        
        results_path = self.output_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save predictions
        predictions_path = self.output_dir / 'predictions.npz'
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
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data HDF5 file')
    parser.add_argument('--output', type=str, default='results/testing',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create tester and start testing
    tester = PrismTester(args.config, args.model, args.data, args.output)
    tester.test()

if __name__ == '__main__':
    main()
