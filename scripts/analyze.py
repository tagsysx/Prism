#!/usr/bin/env python3
"""
CSI Analysis Script for Prism Test Results

This script analyzes test results from the Prism neural network testing.
It performs detailed analysis of CSI predictions vs ground truth including:

1. Per-subcarrier amplitude and phase CDF comparison
2. Amplitude MAE error CDF and phase MAE error CDF (with phase wrapping)
3. PDP computation with 1024 FFT and delay CDF comparison
4. PDP MAE CDF analysis
5. Spatial spectrum analysis with Bartlett algorithm using NMSE-based accuracy

Features:
- Loads configuration file to auto-detect test results path
- Comprehensive CSI analysis with proper phase wrapping
- PDP analysis with configurable FFT size
- Detailed visualizations and statistical analysis
- Export results to JSON and plots
- Automatic path resolution from config files
- Parallel processing for spatial spectrum computation (significant speedup)
- Vectorized Bartlett algorithm implementation for improved performance
- NMSE-based accuracy calculation (most scientific and professional approach)
- Signal fidelity percentage measurement: Accuracy (%) = max(0, (1 - NMSE) × 100)
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import h5py
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
# Note: SSIM implementation will be added inline to avoid dependency issues

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism.config_loader import ModernConfigLoader


def compute_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: First image array
        img2: Second image array  
        data_range: Data range of the images
        
    Returns:
        SSIM value between 0 and 1
    """
    # Ensure arrays are the same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape: {img1.shape} vs {img2.shape}")
    
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Constants for SSIM computation
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    # Compute means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Compute variances and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    # Compute SSIM
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    
    ssim_value = numerator / denominator
    return float(ssim_value)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CSIAnalyzer:
    """
    Comprehensive CSI analysis for Prism test results
    
    Analyzes predicted vs ground truth CSI data with focus on:
    - Per-subcarrier amplitude and phase analysis
    - Error distribution analysis with proper phase wrapping
    - PDP analysis with FFT-based computation
    """
    
    def __init__(self, config_path: str, results_path: str = None, output_dir: str = None, fft_size: int = 2048, 
                 use_parallel: bool = True, num_workers: int = None):
        """
        Initialize CSI analyzer
        
        Args:
            config_path: Path to configuration file
            results_path: Path to test results (.npz file) - optional, will auto-detect from config
            output_dir: Output directory for analysis results
            fft_size: FFT size for PDP computation
            use_parallel: Whether to use parallel processing for spatial spectrum computation
            num_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.config_path = Path(config_path)
        self.fft_size = fft_size
        self.use_parallel = use_parallel
        self.num_workers = num_workers if num_workers is not None else max(1, cpu_count() - 1)
        
        # Load configuration
        self.config_loader = ModernConfigLoader(self.config_path)
        
        # Determine results path
        if results_path:
            self.results_path = Path(results_path)
        else:
            # Auto-detect from config
            output_paths = self.config_loader.get_output_paths()
            predictions_dir = output_paths['predictions_dir']
            self.results_path = Path(predictions_dir) / 'test_results.npz'
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to plots directory in testing results
            output_paths = self.config_loader.get_output_paths()
            plots_dir = output_paths['plots_dir']
            self.output_dir = Path(plots_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        
        # Load test results
        self._load_results()
        
        logger.info(f"🔍 CSI Analyzer initialized:")
        logger.info(f"   Config path: {self.config_path}")
        logger.info(f"   Results path: {self.results_path}")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"   FFT size for PDP: {self.fft_size}")
        logger.info(f"   Parallel processing: {self.use_parallel} ({self.num_workers} workers)")
        logger.info(f"   Data shape: {self.predictions.shape}")
    
    def _load_results(self):
        """Load test results from .npz file"""
        logger.info("📂 Loading test results...")
        
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        
        # Load data
        data = np.load(self.results_path)
        
        # Extract CSI data - preserve complex data type
        self.predictions = torch.from_numpy(data['predictions'])
        self.targets = torch.from_numpy(data['targets'])
        
        # Convert to complex if needed
        if not self.predictions.is_complex():
            # If data is real, assume it's already complex64/128 format
            if self.predictions.dtype == torch.float32:
                self.predictions = self.predictions.to(torch.complex64)
            elif self.predictions.dtype == torch.float64:
                self.predictions = self.predictions.to(torch.complex128)
        
        if not self.targets.is_complex():
            if self.targets.dtype == torch.float32:
                self.targets = self.targets.to(torch.complex64)
            elif self.targets.dtype == torch.float64:
                self.targets = self.targets.to(torch.complex128)
        
        # Extract position data (optional)
        if 'test_ue_positions' in data:
            self.ue_positions = torch.from_numpy(data['test_ue_positions']).float()
        else:
            self.ue_positions = None
            
        if 'test_bs_positions' in data:
            self.bs_positions = torch.from_numpy(data['test_bs_positions']).float()
        else:
            self.bs_positions = None
        
        logger.info(f"✅ Results loaded successfully:")
        logger.info(f"   Predictions shape: {self.predictions.shape}")
        logger.info(f"   Targets shape: {self.targets.shape}")
        logger.info(f"   Data type: {self.predictions.dtype}")
        # Handle complex data min/max calculation
        if self.predictions.is_complex():
            pred_abs = torch.abs(self.predictions)
            target_abs = torch.abs(self.targets)
            logger.info(f"   Predictions magnitude min/max: {pred_abs.min():.6f}/{pred_abs.max():.6f}")
            logger.info(f"   Targets magnitude min/max: {target_abs.min():.6f}/{target_abs.max():.6f}")
        else:
            logger.info(f"   Predictions min/max: {self.predictions.min():.6f}/{self.predictions.max():.6f}")
            logger.info(f"   Targets min/max: {self.targets.min():.6f}/{self.targets.max():.6f}")
        
        # Verify shapes match
        if self.predictions.shape != self.targets.shape:
            raise ValueError(f"Shape mismatch: predictions {self.predictions.shape} vs targets {self.targets.shape}")
    
    def analyze_csi(self):
        """Perform comprehensive CSI analysis"""
        logger.info("🧪 Starting comprehensive CSI analysis...")
        
        # 1. Per-subcarrier amplitude and phase analysis
        self._analyze_per_subcarrier_csi()
        
        # 2. Error distribution analysis
        self._analyze_error_distributions()
        
        # 3. PDP analysis
        self._analyze_pdp()
        
        # 4. Spectrum analysis
        self._analyze_spectrum()
        
        # 5. Random CSI samples comparison (skip spatial spectrum analysis)
        self._analyze_random_csi_samples()
        
        # 6. Generate summary report
        self._generate_summary_report()
        
        logger.info("✅ CSI analysis completed successfully!")
    
    def _analyze_per_subcarrier_csi(self):
        """Analyze per-subcarrier amplitude and phase distributions"""
        logger.info("📊 Analyzing per-subcarrier CSI distributions...")
        
        # Convert to complex tensors if needed
        if not self.predictions.is_complex():
            self.predictions = self.predictions + 1j * torch.zeros_like(self.predictions)
        if not self.targets.is_complex():
            self.targets = self.targets + 1j * torch.zeros_like(self.targets)
        
        # Extract amplitude and phase
        pred_amp = torch.abs(self.predictions)
        target_amp = torch.abs(self.targets)
        pred_phase = torch.angle(self.predictions)
        target_phase = torch.angle(self.targets)
        
        # Flatten for CDF computation (all subcarriers across all samples and antennas)
        pred_amp_flat = pred_amp.flatten().numpy()
        target_amp_flat = target_amp.flatten().numpy()
        pred_phase_flat = pred_phase.flatten().numpy()
        target_phase_flat = target_phase.flatten().numpy()
        
        # Compute CDFs
        pred_amp_cdf = self._compute_empirical_cdf(pred_amp_flat)
        target_amp_cdf = self._compute_empirical_cdf(target_amp_flat)
        pred_phase_cdf = self._compute_empirical_cdf(pred_phase_flat)
        target_phase_cdf = self._compute_empirical_cdf(target_phase_flat)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Per-Subcarrier CSI Analysis: Amplitude and Phase CDFs', fontsize=16, fontweight='bold')
        
        # Amplitude CDFs
        ax1.plot(pred_amp_cdf[0], pred_amp_cdf[1], 'r-', linewidth=2, label='Predicted Amplitude')
        ax1.plot(target_amp_cdf[0], target_amp_cdf[1], 'b-', linewidth=2, label='Target Amplitude')
        ax1.set_xlabel('Amplitude Value')
        ax1.set_ylabel('Cumulative Probability')
        ax1.set_title('Amplitude CDF Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Phase CDFs
        ax2.plot(pred_phase_cdf[0], pred_phase_cdf[1], 'r-', linewidth=2, label='Predicted Phase')
        ax2.plot(target_phase_cdf[0], target_phase_cdf[1], 'b-', linewidth=2, label='Target Phase')
        ax2.set_xlabel('Phase Value (rad)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Phase CDF Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(-np.pi, np.pi)
        
        # Amplitude distributions
        ax3.hist(pred_amp_flat, bins=50, alpha=0.7, color='red', label='Predicted', density=True)
        ax3.hist(target_amp_flat, bins=50, alpha=0.7, color='blue', label='Target', density=True)
        ax3.set_xlabel('Amplitude Value')
        ax3.set_ylabel('Density')
        ax3.set_title('Amplitude Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Phase distributions
        ax4.hist(pred_phase_flat, bins=50, alpha=0.7, color='red', label='Predicted', density=True)
        ax4.hist(target_phase_flat, bins=50, alpha=0.7, color='blue', label='Target', density=True)
        ax4.set_xlabel('Phase Value (rad)')
        ax4.set_ylabel('Density')
        ax4.set_title('Phase Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-np.pi, np.pi)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'per_subcarrier_csi_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Per-subcarrier CSI analysis saved: {plot_path}")
        
        # Store results for summary
        self.per_subcarrier_stats = {
            'pred_amp_mean': float(np.mean(pred_amp_flat)),
            'pred_amp_std': float(np.std(pred_amp_flat)),
            'target_amp_mean': float(np.mean(target_amp_flat)),
            'target_amp_std': float(np.std(target_amp_flat)),
            'pred_phase_mean': float(np.mean(pred_phase_flat)),
            'pred_phase_std': float(np.std(pred_phase_flat)),
            'target_phase_mean': float(np.mean(target_phase_flat)),
            'target_phase_std': float(np.std(target_phase_flat))
        }
    
    def _analyze_error_distributions(self):
        """Analyze amplitude MAE and phase MAE distributions"""
        logger.info("📊 Analyzing amplitude MAE and phase MAE distributions...")
        
        # Convert to complex tensors if needed
        if not self.predictions.is_complex():
            self.predictions = self.predictions + 1j * torch.zeros_like(self.predictions)
        if not self.targets.is_complex():
            self.targets = self.targets + 1j * torch.zeros_like(self.targets)
        
        batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = self.predictions.shape
        
        # Compute amplitude and phase MAE for each antenna pair
        amp_mae_values = []
        phase_mae_values = []
        
        for batch_idx in range(batch_size):
            for bs_idx in range(num_bs_antennas):
                for ue_idx in range(num_ue_antennas):
                    # Get amplitude and phase for this antenna pair
                    pred_amp = torch.abs(self.predictions[batch_idx, bs_idx, ue_idx, :])
                    target_amp = torch.abs(self.targets[batch_idx, bs_idx, ue_idx, :])
                    pred_phase = torch.angle(self.predictions[batch_idx, bs_idx, ue_idx, :])
                    target_phase = torch.angle(self.targets[batch_idx, bs_idx, ue_idx, :])
                    
                    # Compute amplitude MAE
                    amp_mae = torch.mean(torch.abs(pred_amp - target_amp)).item()
                    amp_mae_values.append(amp_mae)
                    
                    # Compute phase MAE with proper wrapping
                    phase_diff = pred_phase - target_phase
                    phase_diff_wrapped = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
                    phase_mae = torch.mean(torch.abs(phase_diff_wrapped)).item()
                    phase_mae_values.append(phase_mae)
        
        # Convert to numpy arrays
        amp_mae_values = np.array(amp_mae_values)
        phase_mae_values = np.array(phase_mae_values)
        
        # Compute CDFs for MAE distributions
        amp_mae_cdf = self._compute_empirical_cdf(amp_mae_values)
        phase_mae_cdf = self._compute_empirical_cdf(phase_mae_values)
        
        # Create visualization focused on MAE distributions
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CSI Error Analysis: Amplitude MAE and Phase MAE Distributions', fontsize=16, fontweight='bold')
        
        # Amplitude MAE CDF
        ax1.plot(amp_mae_cdf[0], amp_mae_cdf[1], 'r-', linewidth=2, label='Amplitude MAE')
        ax1.set_xlabel('Amplitude MAE')
        ax1.set_ylabel('Cumulative Probability')
        ax1.set_title('Amplitude MAE CDF')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Amplitude MAE Histogram
        ax2.hist(amp_mae_values, bins=50, alpha=0.7, color='red', edgecolor='black', density=True)
        ax2.set_xlabel('Amplitude MAE')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Amplitude MAE Distribution (Mean: {np.mean(amp_mae_values):.4f})')
        ax2.grid(True, alpha=0.3)
        
        # Phase MAE CDF
        ax3.plot(phase_mae_cdf[0], phase_mae_cdf[1], 'b-', linewidth=2, label='Phase MAE')
        ax3.set_xlabel('Phase MAE (rad)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Phase MAE CDF')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Phase MAE Histogram
        ax4.hist(phase_mae_values, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
        ax4.set_xlabel('Phase MAE (rad)')
        ax4.set_ylabel('Density')
        ax4.set_title(f'Phase MAE Distribution (Mean: {np.mean(phase_mae_values):.4f})')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f"""Error Statistics:
Amplitude MAE: {np.mean(amp_mae_values):.4f} ± {np.std(amp_mae_values):.4f}
Phase MAE: {np.mean(phase_mae_values):.4f} ± {np.std(phase_mae_values):.4f}
Median Amp MAE: {np.median(amp_mae_values):.4f}
Median Phase MAE: {np.median(phase_mae_values):.4f}"""
        
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'error_distribution_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Amplitude MAE and phase MAE analysis saved: {plot_path}")
        
        # Store results for summary
        self.error_stats = {
            'amp_mae_mean': float(np.mean(amp_mae_values)),
            'amp_mae_std': float(np.std(amp_mae_values)),
            'amp_mae_median': float(np.median(amp_mae_values)),
            'phase_mae_mean': float(np.mean(phase_mae_values)),
            'phase_mae_std': float(np.std(phase_mae_values)),
            'phase_mae_median': float(np.median(phase_mae_values)),
            'num_subcarriers': num_subcarriers
        }
    
    def _analyze_pdp(self):
        """Analyze Power Delay Profile (PDP) with FFT-based computation"""
        logger.info("📊 Analyzing Power Delay Profile (PDP)...")
        
        # Convert to complex tensors if needed
        if not self.predictions.is_complex():
            self.predictions = self.predictions + 1j * torch.zeros_like(self.predictions)
        if not self.targets.is_complex():
            self.targets = self.targets + 1j * torch.zeros_like(self.targets)
        
        # Compute PDP for all samples and antennas
        # Only support 4D CSI data format
        if len(self.predictions.shape) != 4:
            raise ValueError(f"Only 4D CSI data is supported. Current shape: {self.predictions.shape}. Expected: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]")
        
        batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = self.predictions.shape
        total_antennas = num_bs_antennas * num_ue_antennas
        
        # Initialize PDP arrays
        pred_pdp_all = []
        target_pdp_all = []
        
        logger.info(f"   Computing PDP for {batch_size} samples × {total_antennas} antennas...")
        
        processed_count = 0
        total_pdp_computations = batch_size * num_bs_antennas * num_ue_antennas
        
        for sample_idx in range(batch_size):
            # 4D format: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
            for bs_idx in range(num_bs_antennas):
                    for ue_idx in range(num_ue_antennas):
                        # Get CSI for this sample, BS antenna, and UE antenna
                        pred_csi = self.predictions[sample_idx, bs_idx, ue_idx, :].numpy()
                        target_csi = self.targets[sample_idx, bs_idx, ue_idx, :].numpy()
                        
                        pred_pdp = self._compute_pdp(pred_csi, self.fft_size)
                        target_pdp = self._compute_pdp(target_csi, self.fft_size)
                        
                        pred_pdp_all.append(pred_pdp)
                        target_pdp_all.append(target_pdp)
                        
                        processed_count += 1
                        if processed_count % 50 == 0:  # Progress update every 50 PDP computations
                            logger.info(f"     PDP computation progress: {processed_count}/{total_pdp_computations} ({processed_count/total_pdp_computations*100:.1f}%)")
        
        # Convert to numpy arrays
        pred_pdp_all = np.array(pred_pdp_all)
        target_pdp_all = np.array(target_pdp_all)
        
        logger.info(f"   PDP computed: {pred_pdp_all.shape}")
        
        # Compute PDP MAE
        pdp_mae = np.mean(np.abs(pred_pdp_all - target_pdp_all), axis=0)
        
        # Compute Global Similarity Metrics
        logger.info("   Computing global similarity metrics...")
        global_metrics, global_metrics_raw = self._compute_global_similarity_metrics(pred_pdp_all, target_pdp_all)
        
        # Compute CDFs
        pdp_mae_cdf = self._compute_empirical_cdf(pdp_mae)
        
        # Compute CDFs for global similarity metrics
        mse_cdf = self._compute_empirical_cdf(global_metrics_raw['mse_values'])
        rmse_cdf = self._compute_empirical_cdf(global_metrics_raw['rmse_values'])
        nmse_cdf = self._compute_empirical_cdf(global_metrics_raw['nmse_values'])
        cosine_sim_cdf = self._compute_empirical_cdf(global_metrics_raw['cosine_sim_values'])
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Power Delay Profile Analysis with Global Similarity Metrics (FFT Size: {self.fft_size})', fontsize=16, fontweight='bold')
        
        # MSE CDF
        ax1.plot(mse_cdf[0], mse_cdf[1], 'r-', linewidth=2, label='MSE')
        ax1.set_xlabel('MSE Value')
        ax1.set_ylabel('Cumulative Probability')
        ax1.set_title('MSE CDF')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # PDP MAE CDF
        ax2.plot(pdp_mae_cdf[0], pdp_mae_cdf[1], 'g-', linewidth=2, label='PDP MAE')
        ax2.set_xlabel('PDP MAE Value')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('PDP MAE CDF')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # RMSE CDF
        ax3.plot(rmse_cdf[0], rmse_cdf[1], 'g-', linewidth=2, label='RMSE')
        ax3.set_xlabel('RMSE Value')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('RMSE CDF')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Cosine Similarity CDF
        ax4.plot(cosine_sim_cdf[0], cosine_sim_cdf[1], 'purple', linewidth=2, label='Cosine Similarity')
        ax4.set_xlabel('Cosine Similarity Value')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cosine Similarity CDF')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(0, 1)
        
        # Add global similarity metrics text
        metrics_text = f"""Global Similarity Metrics:
MSE: {global_metrics['mse_mean']:.6f} ± {global_metrics['mse_std']:.6f}
RMSE: {global_metrics['rmse_mean']:.6f} ± {global_metrics['rmse_std']:.6f}
NMSE: {global_metrics['nmse_mean']:.6f} ± {global_metrics['nmse_std']:.6f}
Cosine Sim: {global_metrics['cosine_sim_mean']:.6f} ± {global_metrics['cosine_sim_std']:.6f}"""
        
        ax4.text(0.02, 0.98, metrics_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'pdp_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ PDP analysis saved: {plot_path}")
        
        # Create random PDP comparison
        self._plot_random_pdp_comparison(pred_pdp_all, target_pdp_all, batch_size, num_bs_antennas, num_ue_antennas)
        
        # Store results for summary
        self.pdp_stats = {
            'pdp_mae_mean': float(np.mean(pdp_mae)),
            'pdp_mae_std': float(np.std(pdp_mae)),
            'fft_size': self.fft_size,
            'global_similarity': global_metrics
        }
    
    def _analyze_spatial_spectra(self):
        """Analyze spatial spectra for BS and UE antenna arrays"""
        logger.info("📊 Analyzing spatial spectra...")
        
        # Convert to complex tensors if needed
        if not self.predictions.is_complex():
            self.predictions = self.predictions + 1j * torch.zeros_like(self.predictions)
        if not self.targets.is_complex():
            self.targets = self.targets + 1j * torch.zeros_like(self.targets)
        
        # Get antenna configuration from config
        bs_antenna_count = self.config_loader.num_bs_antennas
        ue_antenna_count = self.config_loader.ue_antenna_count
        
        logger.info(f"   BS antennas: {bs_antenna_count}, UE antennas: {ue_antenna_count}")
        
        # Determine CSI data shape - only support 4D CSI data
        if len(self.predictions.shape) != 4:
            raise ValueError(f"Only 4D CSI data is supported. Current shape: {self.predictions.shape}. Expected: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]")
        
        batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = self.predictions.shape
        
        logger.info(f"   CSI shape: {self.predictions.shape}")
        logger.info(f"   Detected: {num_bs_antennas} BS antennas, {num_ue_antennas} UE antennas")
        
        # Initialize results storage
        spatial_spectrum_stats = {}
        
        # Analyze BS spatial spectrum if BS antennas > 1
        if num_bs_antennas > 1:
            logger.info("   Computing BS spatial spectra accuracy (NMSE-based)...")
            bs_accuracy_values = self._compute_spatial_spectrum_accuracy(
                self.predictions, self.targets, 
                antenna_dim=1,  # BS antenna dimension
                antenna_count=num_bs_antennas,
                antenna_type="BS"
            )
            spatial_spectrum_stats['bs_accuracy'] = {
                'mean': float(np.mean(bs_accuracy_values)),
                'std': float(np.std(bs_accuracy_values)),
                'median': float(np.median(bs_accuracy_values)),
                'min': float(np.min(bs_accuracy_values)),
                'max': float(np.max(bs_accuracy_values)),
                'values': bs_accuracy_values.tolist()
            }
            logger.info(f"   BS Accuracy: {np.mean(bs_accuracy_values):.2f}% ± {np.std(bs_accuracy_values):.2f}%")
        
        # Analyze UE spatial spectrum if UE antennas > 1
        if num_ue_antennas > 1:
            logger.info("   Computing UE spatial spectra accuracy (NMSE-based)...")
            ue_accuracy_values = self._compute_spatial_spectrum_accuracy(
                self.predictions, self.targets,
                antenna_dim=2,  # UE antenna dimension
                antenna_count=num_ue_antennas,
                antenna_type="UE"
            )
            spatial_spectrum_stats['ue_accuracy'] = {
                'mean': float(np.mean(ue_accuracy_values)),
                'std': float(np.std(ue_accuracy_values)),
                'median': float(np.median(ue_accuracy_values)),
                'min': float(np.min(ue_accuracy_values)),
                'max': float(np.max(ue_accuracy_values)),
                'values': ue_accuracy_values.tolist()
            }
            logger.info(f"   UE Accuracy: {np.mean(ue_accuracy_values):.2f}% ± {np.std(ue_accuracy_values):.2f}%")
        
        # Create accuracy CDF plot
        self._plot_spatial_spectrum_accuracy_cdf(spatial_spectrum_stats)
        
        # Create random spatial spectrum comparison
        self._plot_random_spatial_spectra_comparison()
        
        # Create individual spatial spectrum comparisons
        self._plot_individual_spatial_spectra_comparisons()
        
        # Store results for summary
        self.spatial_spectrum_stats = spatial_spectrum_stats
        
        logger.info("✅ Spatial spectrum analysis completed!")
    
    def _compute_spatial_spectrum_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                          antenna_dim: int, antenna_count: int, antenna_type: str) -> np.ndarray:
        """
        Compute spatial spectrum accuracy using NMSE-based method with parallel processing.
        
        This implements the most scientific and professional approach:
        Accuracy (%) = max(0, (1 - NMSE) × 100)
        where NMSE = mean(|H_pred - H_gt|²) / mean(|H_gt|²)
        
        This method is widely recognized in academic and engineering communities as the most
        appropriate way to measure signal fidelity in spatial spectrum analysis.
        """
        
        batch_size = predictions.shape[0]
        num_subcarriers = predictions.shape[-1]
        total_samples = batch_size * num_subcarriers
        
        logger.info(f"🔍 Computing spatial spectrum accuracy (NMSE-based) for {antenna_type} antennas ({total_samples} samples)...")
        
        # Prepare arguments for parallel processing
        args_list = []
        for batch_idx in range(batch_size):
            for subcarrier_idx in range(num_subcarriers):
                # Extract CSI for this batch and subcarrier
                # 4D format: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
                if antenna_dim == 1:  # BS antennas
                    pred_csi = predictions[batch_idx, :, :, subcarrier_idx]  # [num_bs_antennas, num_ue_antennas]
                    target_csi = targets[batch_idx, :, :, subcarrier_idx]
                else:  # UE antennas
                    pred_csi = predictions[batch_idx, :, :, subcarrier_idx].T  # [num_ue_antennas, num_bs_antennas]
                    target_csi = targets[batch_idx, :, :, subcarrier_idx].T
                
                args_list.append((pred_csi, target_csi, antenna_dim, antenna_type, batch_idx, subcarrier_idx))
        
        # Use parallel processing if enabled and we have enough samples
        if hasattr(self, 'use_parallel') and self.use_parallel and len(args_list) > 4:
            logger.info(f"   Using parallel processing with {self.num_workers} workers...")
            
            # Suppress multiprocessing warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Use multiprocessing Pool
                with Pool(processes=self.num_workers) as pool:
                    # Create partial function with self bound
                    compute_func = partial(self._compute_single_spatial_spectrum_accuracy)
                    
                    # Process in chunks to avoid memory issues
                    chunk_size = max(1, len(args_list) // (self.num_workers * 4))
                    results = pool.map(compute_func, args_list, chunksize=chunk_size)
            
            # Extract accuracies from results
            accuracy_values = [result[0] for result in results]
            
        else:
            # Fallback to sequential processing
            logger.info(f"   Using sequential processing...")
            accuracy_values = []
            processed_samples = 0
            
            for args in args_list:
                processed_samples += 1
                if processed_samples % 100 == 0:
                    logger.info(f"   Processing sample {processed_samples}/{total_samples} for {antenna_type} antennas...")
                
                result = self._compute_single_spatial_spectrum_accuracy(args)
                accuracy_values.append(result[0])
        
        logger.info(f"✅ Completed accuracy computation for {antenna_type} antennas: {len(accuracy_values)} samples")
        logger.info(f"   Mean accuracy: {np.mean(accuracy_values):.2f}%, Std: {np.std(accuracy_values):.2f}%")
        return np.array(accuracy_values)
    
    def _plot_spatial_spectrum_accuracy_cdf(self, spatial_spectrum_stats: Dict):
        """
        Plot CDF of accuracy values for spatial spectra using NMSE-based method.
        
        This plot shows the cumulative distribution of spatial spectrum accuracy,
        which represents signal fidelity percentage based on the most scientific
        and professional NMSE-based approach.
        """
        logger.info("📊 Creating spatial spectrum accuracy CDF plot (NMSE-based)...")
        
        plt.figure(figsize=(12, 8))
        
        plot_count = 0
        if 'bs_accuracy' in spatial_spectrum_stats:
            plot_count += 1
        if 'ue_accuracy' in spatial_spectrum_stats:
            plot_count += 1
        
        if plot_count == 0:
            logger.warning("No spatial spectrum accuracy data to plot")
            return
        
        subplot_idx = 1
        
        # Plot BS Accuracy CDF
        if 'bs_accuracy' in spatial_spectrum_stats:
            plt.subplot(plot_count, 1, subplot_idx)
            bs_accuracy_values = spatial_spectrum_stats['bs_accuracy']['values']
            bs_accuracy_sorted = np.sort(bs_accuracy_values)
            bs_accuracy_cdf = np.arange(1, len(bs_accuracy_sorted) + 1) / len(bs_accuracy_sorted)
            
            plt.plot(bs_accuracy_sorted, bs_accuracy_cdf, 'b-', linewidth=2, label='BS Spatial Spectrum Accuracy')
            
            plt.xlabel('Accuracy (%)')
            plt.ylabel('Cumulative Probability')
            plt.title(f'BS Spatial Spectrum Accuracy CDF (NMSE-based)\nMean: {spatial_spectrum_stats["bs_accuracy"]["mean"]:.2f}%, Median: {spatial_spectrum_stats["bs_accuracy"]["median"]:.2f}%')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.ylim(0, 1)
            plt.xlim(0, 100)
            
            # Add percentile lines
            percentiles = [25, 50, 75, 90, 95]
            for p in percentiles:
                value = np.percentile(bs_accuracy_values, p)
                plt.axvline(value, color='gray', linestyle='--', alpha=0.5)
                plt.text(value, 0.1, f'{p}%', rotation=90, fontsize=8)
            
            subplot_idx += 1
        
        # Plot UE Accuracy CDF
        if 'ue_accuracy' in spatial_spectrum_stats:
            plt.subplot(plot_count, 1, subplot_idx)
            ue_accuracy_values = spatial_spectrum_stats['ue_accuracy']['values']
            ue_accuracy_sorted = np.sort(ue_accuracy_values)
            ue_accuracy_cdf = np.arange(1, len(ue_accuracy_sorted) + 1) / len(ue_accuracy_sorted)
            
            plt.plot(ue_accuracy_sorted, ue_accuracy_cdf, 'r-', linewidth=2, label='UE Spatial Spectrum Accuracy')
            
            plt.xlabel('Accuracy (%)')
            plt.ylabel('Cumulative Probability')
            plt.title(f'UE Spatial Spectrum Accuracy CDF (NMSE-based)\nMean: {spatial_spectrum_stats["ue_accuracy"]["mean"]:.2f}%, Median: {spatial_spectrum_stats["ue_accuracy"]["median"]:.2f}%')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.ylim(0, 1)
            plt.xlim(0, 100)
            
            # Add percentile lines
            percentiles = [25, 50, 75, 90, 95]
            for p in percentiles:
                value = np.percentile(ue_accuracy_values, p)
                plt.axvline(value, color='gray', linestyle='--', alpha=0.5)
                plt.text(value, 0.1, f'{p}%', rotation=90, fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'spatial_spectrum_accuracy_cdf.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Spatial spectrum accuracy CDF plot saved: {plot_path}")
    
    def _plot_random_spatial_spectra_comparison(self):
        """Plot random spatial spectra comparison (10 samples, left: ground truth, right: predicted)"""
        logger.info("📊 Creating random spatial spectra comparison...")
        
        # Convert to complex tensors if needed
        if not self.predictions.is_complex():
            self.predictions = self.predictions + 1j * torch.zeros_like(self.predictions)
        if not self.targets.is_complex():
            self.targets = self.targets + 1j * torch.zeros_like(self.targets)
        
        # Get antenna configuration
        bs_antenna_count = self.config_loader.num_bs_antennas
        ue_antenna_count = self.config_loader.ue_antenna_count
        
        # Determine CSI data shape - only support 4D CSI data
        if len(self.predictions.shape) != 4:
            raise ValueError(f"Only 4D CSI data is supported. Current shape: {self.predictions.shape}. Expected: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]")
        
        batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = self.predictions.shape
        
        # Randomly select 10 samples
        np.random.seed(42)  # For reproducibility
        selected_indices = np.random.choice(batch_size, size=min(10, batch_size), replace=False)
        logger.info(f"   Selected samples: {selected_indices}")
        
        # Determine which spatial spectra to plot
        plot_bs = num_bs_antennas > 1
        plot_ue = num_ue_antennas > 1
        
        if not plot_bs and not plot_ue:
            logger.warning("No multi-antenna arrays found for spatial spectrum comparison")
            return
        
        # Create figure with subplots
        num_plots = 0
        if plot_bs:
            num_plots += 1
        if plot_ue:
            num_plots += 1
            
        fig, axes = plt.subplots(num_plots, 1, figsize=(16, 4 * num_plots))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot BS spatial spectra if applicable
        if plot_bs:
            ax = axes[plot_idx]
            self._plot_spatial_spectra_for_antenna_type(
                ax, selected_indices, antenna_dim=1, antenna_type="BS",
                num_bs_antennas=num_bs_antennas, num_ue_antennas=num_ue_antennas,
                num_subcarriers=num_subcarriers
            )
            plot_idx += 1
        
        # Plot UE spatial spectra if applicable
        if plot_ue:
            ax = axes[plot_idx]
            self._plot_spatial_spectra_for_antenna_type(
                ax, selected_indices, antenna_dim=2, antenna_type="UE",
                num_bs_antennas=num_bs_antennas, num_ue_antennas=num_ue_antennas,
                num_subcarriers=num_subcarriers
            )
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'random_spatial_spectra_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Random spatial spectra comparison saved: {plot_path}")
    
    def _plot_spatial_spectra_for_antenna_type(self, ax, selected_indices, antenna_dim, antenna_type,
                                              num_bs_antennas, num_ue_antennas, num_subcarriers):
        """Plot spatial spectra for a specific antenna type"""
        
        # Create a grid of subplots within the main subplot
        n_samples = len(selected_indices)
        
        # Remove the original axis
        ax.remove()
        
        # Create subplot grid: 10 rows, 2 columns (left: GT, right: Pred)
        fig = plt.gcf()
        gs = fig.add_gridspec(n_samples, 2, hspace=0.3, wspace=0.1)
        
        for i, sample_idx in enumerate(selected_indices):
            # Left subplot: Ground Truth
            ax_gt = fig.add_subplot(gs[i, 0])
            # Right subplot: Predicted
            ax_pred = fig.add_subplot(gs[i, 1])
            
            # Select a representative subcarrier (middle one)
            subcarrier_idx = num_subcarriers // 2
            
            # Extract CSI for this sample and subcarrier
            # 4D format: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
            if antenna_dim == 1:  # BS antennas
                gt_csi = self.targets[sample_idx, :, :, subcarrier_idx]  # [num_bs_antennas, num_ue_antennas]
                pred_csi = self.predictions[sample_idx, :, :, subcarrier_idx]
            else:  # UE antennas
                gt_csi = self.targets[sample_idx, :, :, subcarrier_idx].T  # [num_ue_antennas, num_bs_antennas]
                pred_csi = self.predictions[sample_idx, :, :, subcarrier_idx].T
            
            # Compute spatial spectrum using Bartlett algorithm
            gt_spectrum = self._compute_bartlett_spatial_spectrum(gt_csi)
            pred_spectrum = self._compute_bartlett_spatial_spectrum(pred_csi)
            
            # Convert to numpy
            gt_spectrum_np = gt_spectrum.detach().cpu().numpy()
            pred_spectrum_np = pred_spectrum.detach().cpu().numpy()
            
            # Plot ground truth (Bartlett spatial spectrum is always 2D: azimuth × elevation)
            im_gt = ax_gt.imshow(gt_spectrum_np, cmap='viridis', aspect='auto')
            ax_gt.set_title(f'Sample {sample_idx} - GT', fontsize=8)
            ax_gt.set_ylabel('Elevation (deg)', fontsize=7)
            if i == n_samples - 1:  # Last row
                ax_gt.set_xlabel('Azimuth (deg)', fontsize=7)
            
            # Plot predicted (Bartlett spatial spectrum is always 2D: azimuth × elevation)
            im_pred = ax_pred.imshow(pred_spectrum_np, cmap='viridis', aspect='auto')
            ax_pred.set_title(f'Sample {sample_idx} - Pred', fontsize=8)
            ax_pred.set_ylabel('Elevation (deg)', fontsize=7)
            if i == n_samples - 1:  # Last row
                ax_pred.set_xlabel('Azimuth (deg)', fontsize=7)
            
            # Set consistent color scale
            vmin = min(gt_spectrum_np.min(), pred_spectrum_np.min())
            vmax = max(gt_spectrum_np.max(), pred_spectrum_np.max())
            im_gt.set_clim(vmin, vmax)
            im_pred.set_clim(vmin, vmax)
            
            # Set axis ticks for azimuth and elevation (36×5 resolution)
            azimuth_ticks = np.arange(0, gt_spectrum_np.shape[1], max(1, gt_spectrum_np.shape[1]//6))
            elevation_ticks = np.arange(0, gt_spectrum_np.shape[0], max(1, gt_spectrum_np.shape[0]//3))
            
            ax_gt.set_xticks(azimuth_ticks)
            ax_gt.set_xticklabels([f'{int(az*10)}' for az in azimuth_ticks], fontsize=6)
            ax_gt.set_yticks(elevation_ticks)
            ax_gt.set_yticklabels([f'{int(el*18)}' for el in elevation_ticks], fontsize=6)
            
            ax_pred.set_xticks(azimuth_ticks)
            ax_pred.set_xticklabels([f'{int(az*10)}' for az in azimuth_ticks], fontsize=6)
            ax_pred.set_yticks(elevation_ticks)
            ax_pred.set_yticklabels([f'{int(el*18)}' for el in elevation_ticks], fontsize=6)
            
            # Remove ticks for cleaner look
            ax_gt.tick_params(labelsize=6)
            ax_pred.tick_params(labelsize=6)
        
        # Add main title
        plt.suptitle(f'{antenna_type} Spatial Spectra Comparison (Subcarrier {num_subcarriers//2})', 
                    fontsize=12, y=0.98)
    
    def _plot_individual_spatial_spectra_comparisons(self):
        """Plot individual spatial spectrum comparisons (10 samples, each as separate figure)"""
        logger.info("📊 Creating individual spatial spectrum comparisons...")
        
        # Convert to complex tensors if needed
        if not self.predictions.is_complex():
            self.predictions = self.predictions + 1j * torch.zeros_like(self.predictions)
        if not self.targets.is_complex():
            self.targets = self.targets + 1j * torch.zeros_like(self.targets)
        
        # Get antenna configuration
        bs_antenna_count = self.config_loader.num_bs_antennas
        ue_antenna_count = self.config_loader.ue_antenna_count
        
        # Determine CSI data shape - only support 4D CSI data
        if len(self.predictions.shape) != 4:
            raise ValueError(f"Only 4D CSI data is supported. Current shape: {self.predictions.shape}. Expected: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]")
        
        batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = self.predictions.shape
        
        # Randomly select 10 samples
        np.random.seed(42)  # For reproducibility
        selected_indices = np.random.choice(batch_size, size=min(10, batch_size), replace=False)
        logger.info(f"   Selected samples: {selected_indices}")
        
        # Determine which spatial spectra to plot
        plot_bs = num_bs_antennas > 1
        plot_ue = num_ue_antennas > 1
        
        if not plot_bs and not plot_ue:
            logger.warning("No multi-antenna arrays found for spatial spectrum comparison")
            return
        
        # Create ss directory
        ss_dir = self.output_dir / 'ss'
        ss_dir.mkdir(exist_ok=True)
        
        # Plot individual comparisons
        if plot_bs:
            self._plot_individual_spatial_spectra_for_antenna_type(
                selected_indices, antenna_dim=1, antenna_type="BS",
                num_bs_antennas=num_bs_antennas, num_ue_antennas=num_ue_antennas,
                num_subcarriers=num_subcarriers, output_dir=ss_dir
            )
        
        if plot_ue:
            self._plot_individual_spatial_spectra_for_antenna_type(
                selected_indices, antenna_dim=2, antenna_type="UE",
                num_bs_antennas=num_bs_antennas, num_ue_antennas=num_ue_antennas,
                num_subcarriers=num_subcarriers, output_dir=ss_dir
            )
        
        logger.info(f"✅ Individual spatial spectrum comparisons saved to: {ss_dir}")
    
    def _plot_individual_spatial_spectra_for_antenna_type(self, selected_indices, antenna_dim, antenna_type,
                                                         num_bs_antennas, num_ue_antennas, num_subcarriers, output_dir):
        """Plot individual spatial spectra for a specific antenna type"""
        
        # Select a representative subcarrier (middle one)
        subcarrier_idx = num_subcarriers // 2
        
        # First pass: compute all spectra to determine global colorbar range
        all_spectra = []
        for sample_idx in selected_indices:
            # Extract CSI for this sample and subcarrier
            # 4D format: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
            if antenna_dim == 1:  # BS antennas
                gt_csi = self.targets[sample_idx, :, :, subcarrier_idx]  # [num_bs_antennas, num_ue_antennas]
                pred_csi = self.predictions[sample_idx, :, :, subcarrier_idx]
            else:  # UE antennas
                gt_csi = self.targets[sample_idx, :, :, subcarrier_idx].T  # [num_ue_antennas, num_bs_antennas]
                pred_csi = self.predictions[sample_idx, :, :, subcarrier_idx].T
            
            # Compute spatial spectrum using Bartlett algorithm
            gt_spectrum = self._compute_bartlett_spatial_spectrum(gt_csi)
            pred_spectrum = self._compute_bartlett_spatial_spectrum(pred_csi)
            
            all_spectra.extend([gt_spectrum, pred_spectrum])
        
        # Determine global colorbar range for normalization
        all_spectra_tensor = torch.stack(all_spectra)
        global_min = all_spectra_tensor.min().item()
        global_max = all_spectra_tensor.max().item()
        
        # Normalize all spectra to [0, 1] range
        normalized_spectra = []
        for spectrum in all_spectra:
            normalized_spectrum = (spectrum - global_min) / (global_max - global_min + 1e-8)
            normalized_spectra.append(normalized_spectrum)
        
        # Use [0, 1] as colorbar range for normalized data
        vmin, vmax = 0.0, 1.0
        
        # Second pass: create individual plots
        for i, sample_idx in enumerate(selected_indices):
            # Get normalized spectra (index 2*i for GT, 2*i+1 for Pred)
            gt_spectrum_norm = normalized_spectra[2*i]
            pred_spectrum_norm = normalized_spectra[2*i+1]
            
            # Convert to numpy
            gt_spectrum_np = gt_spectrum_norm.detach().cpu().numpy()
            pred_spectrum_np = pred_spectrum_norm.detach().cpu().numpy()
            
            # Create figure with side-by-side comparison
            fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot ground truth
            im_gt = ax_gt.imshow(gt_spectrum_np, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            ax_gt.set_title(f'Sample {sample_idx} - Ground Truth', fontsize=14, fontweight='bold')
            ax_gt.set_xlabel('Azimuth (deg)', fontsize=12)
            ax_gt.set_ylabel('Elevation (deg)', fontsize=12)
            
            # Plot predicted
            im_pred = ax_pred.imshow(pred_spectrum_np, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            ax_pred.set_title(f'Sample {sample_idx} - Predicted', fontsize=14, fontweight='bold')
            ax_pred.set_xlabel('Azimuth (deg)', fontsize=12)
            ax_pred.set_ylabel('Elevation (deg)', fontsize=12)
            
            # Set axis ticks for azimuth and elevation (36×5 resolution)
            azimuth_ticks = np.arange(0, gt_spectrum_np.shape[1], max(1, gt_spectrum_np.shape[1]//6))
            elevation_ticks = np.arange(0, gt_spectrum_np.shape[0], max(1, gt_spectrum_np.shape[0]//3))
            
            for ax in [ax_gt, ax_pred]:
                ax.set_xticks(azimuth_ticks)
                ax.set_xticklabels([f'{int(az*10)}' for az in azimuth_ticks], fontsize=10)
                ax.set_yticks(elevation_ticks)
                ax.set_yticklabels([f'{int(el*18)}' for el in elevation_ticks], fontsize=10)
            
            # Add colorbar
            cbar = plt.colorbar(im_pred, ax=[ax_gt, ax_pred], shrink=0.8, aspect=20)
            cbar.set_label('Normalized Spatial Spectrum Power', fontsize=12)
            
            # Add main title
            plt.suptitle(f'{antenna_type} Spatial Spectrum Comparison - Sample {sample_idx} (Subcarrier {subcarrier_idx})', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            plt.tight_layout()
            
            # Save individual plot
            plot_path = output_dir / f'{antenna_type.lower()}_spatial_spectrum_sample_{sample_idx:03d}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"   Saved: {plot_path}")
    
    def _compute_bartlett_spatial_spectrum(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial spectrum using Bartlett algorithm with vectorized operations.
        
        Args:
            csi: Complex CSI tensor [num_antennas_1, num_antennas_2] or [num_antennas]
            
        Returns:
            spatial_spectrum: Spatial spectrum tensor
        """
        # Get carrier frequency from config
        base_station_config = self.config_loader._processed_config.get('base_station', {})
        carrier_freq = base_station_config.get('carrier_frequency', 2.4e9)  # Default to 2.4 GHz
        
        # Calculate wavelength
        c = 3e8  # Speed of light in m/s
        wavelength = c / carrier_freq
        
        # Calculate antenna spacing (half wavelength)
        antenna_spacing = wavelength / 2
        
        if csi.dim() == 1:
            # 1D case: [num_antennas]
            num_antennas = csi.shape[0]
            
            # Define azimuth and elevation angles (36×5 = 180 angle combinations for optimal performance)
            azimuth_angles = torch.arange(0, 360, 10, dtype=torch.float32, device=csi.device)  # 0, 10, 20, ..., 350 (36 points)
            elevation_angles = torch.arange(0, 90, 18, dtype=torch.float32, device=csi.device)  # 0, 18, 36, 54, 72 (5 points)
            
            # Convert to radians
            azimuth_rad = torch.deg2rad(azimuth_angles)
            elevation_rad = torch.deg2rad(elevation_angles)
            
            # Create antenna positions for linear array
            antenna_positions = torch.arange(num_antennas, dtype=torch.float32, device=csi.device) * antenna_spacing
            
            # Vectorized computation: create all steering vectors at once
            # Shape: [num_azimuth, num_elevation, num_antennas]
            az_grid, el_grid, pos_grid = torch.meshgrid(
                azimuth_rad, elevation_rad, antenna_positions, indexing='ij'
            )
            
            # Compute steering vectors for all combinations
            # For linear array, steering vector depends on azimuth angle
            steering_vectors = torch.exp(1j * 2 * torch.pi * pos_grid * torch.cos(az_grid))
            
            # Compute Bartlett spectrum: |steering_vector^H * csi|^2
            # Shape: [num_azimuth, num_elevation]
            spatial_spectrum = torch.abs(torch.sum(steering_vectors.conj() * csi, dim=-1)) ** 2
            
            return spatial_spectrum
            
        elif csi.dim() == 2:
            # 2D case: [num_antennas_1, num_antennas_2]
            num_antennas_1, num_antennas_2 = csi.shape
            
            # Define azimuth and elevation angles (36×5 = 180 angle combinations for optimal performance)
            azimuth_angles = torch.arange(0, 360, 10, dtype=torch.float32, device=csi.device)  # 0, 10, 20, ..., 350 (36 points)
            elevation_angles = torch.arange(0, 90, 18, dtype=torch.float32, device=csi.device)  # 0, 18, 36, 54, 72 (5 points)
            
            # Convert to radians
            azimuth_rad = torch.deg2rad(azimuth_angles)
            elevation_rad = torch.deg2rad(elevation_angles)
            
            # Create antenna positions for 2D array (assuming rectangular grid)
            antenna_positions_1 = torch.arange(num_antennas_1, dtype=torch.float32, device=csi.device) * antenna_spacing
            antenna_positions_2 = torch.arange(num_antennas_2, dtype=torch.float32, device=csi.device) * antenna_spacing
            
            # Vectorized computation: create all steering matrices at once
            # Shape: [num_azimuth, num_elevation, num_antennas_1, num_antennas_2]
            az_grid, el_grid, pos1_grid, pos2_grid = torch.meshgrid(
                azimuth_rad, elevation_rad, antenna_positions_1, antenna_positions_2, indexing='ij'
            )
            
            # Compute steering vectors for both dimensions
            steering_vector_1 = torch.exp(1j * 2 * torch.pi * pos1_grid * torch.cos(az_grid) * torch.sin(el_grid))
            steering_vector_2 = torch.exp(1j * 2 * torch.pi * pos2_grid * torch.sin(az_grid) * torch.sin(el_grid))
            
            # Create 2D steering matrices
            steering_matrices = steering_vector_1 * steering_vector_2
            
            # Compute Bartlett spectrum: |trace(steering_matrix^H * csi)|^2
            # Shape: [num_azimuth, num_elevation]
            spatial_spectrum = torch.abs(torch.sum(steering_matrices.conj() * csi, dim=(-2, -1))) ** 2
            
            return spatial_spectrum
            
        else:
            raise ValueError(f"Unsupported CSI dimension for spatial spectrum: {csi.dim()}")
    
    def _compute_single_spatial_spectrum_accuracy(self, args):
        """
        Compute spatial spectrum accuracy using NMSE-based method (for parallel processing).
        
        This implements the most scientific and professional approach:
        Accuracy (%) = max(0, (1 - NMSE) × 100)
        where NMSE = mean(|H_pred - H_gt|²) / mean(|H_gt|²)
        
        Args:
            args: Tuple containing (pred_csi, target_csi, antenna_dim, antenna_type, sample_idx, subcarrier_idx)
            
        Returns:
            Tuple of (accuracy, sample_idx, subcarrier_idx, antenna_type)
        """
        pred_csi, target_csi, antenna_dim, antenna_type, sample_idx, subcarrier_idx = args
        
        try:
            # Compute spatial spectrum using Bartlett algorithm
            pred_spectrum = self._compute_bartlett_spatial_spectrum(pred_csi)
            target_spectrum = self._compute_bartlett_spatial_spectrum(target_csi)
            
            # Convert to numpy for computation
            pred_spectrum_np = pred_spectrum.detach().cpu().numpy()
            target_spectrum_np = target_spectrum.detach().cpu().numpy()
            
            # Compute NMSE-based accuracy
            # NMSE = mean(|H_pred - H_gt|²) / mean(|H_gt|²)
            error_power = np.mean(np.abs(pred_spectrum_np - target_spectrum_np) ** 2)
            signal_power = np.mean(np.abs(target_spectrum_np) ** 2)
            
            # Avoid division by zero
            if signal_power < 1e-12:
                nmse = 1.0  # Maximum error when signal power is negligible
            else:
                nmse = error_power / signal_power
            
            # Convert NMSE to accuracy percentage
            # Accuracy (%) = max(0, (1 - NMSE) × 100)
            accuracy = max(0.0, (1.0 - nmse) * 100.0)
            
            return (accuracy, sample_idx, subcarrier_idx, antenna_type)
            
        except Exception as e:
            logger.warning(f"Accuracy computation failed for {antenna_type} batch {sample_idx}, subcarrier {subcarrier_idx}: {e}")
            return (0.0, sample_idx, subcarrier_idx, antenna_type)  # Return 0% accuracy for failed cases
    
    def _analyze_random_csi_samples(self):
        """Analyze random CSI samples with random BS and UE antenna combinations"""
        logger.info("📊 Analyzing random CSI samples with random antenna combinations...")
        
        # Convert to complex tensors if needed
        if not self.predictions.is_complex():
            self.predictions = self.predictions + 1j * torch.zeros_like(self.predictions)
        if not self.targets.is_complex():
            self.targets = self.targets + 1j * torch.zeros_like(self.targets)
        
        # Get data dimensions - only support 4D CSI data
        if len(self.predictions.shape) != 4:
            raise ValueError(f"Only 4D CSI data is supported. Current shape: {self.predictions.shape}. Expected: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]")
        
        batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = self.predictions.shape
        
        # Randomly select one sample (different each time)
        sample_idx = np.random.choice(batch_size)
        
        # Randomly select 10 BS antenna and UE antenna combinations (completely random each time)
        bs_antenna_indices = np.random.choice(num_bs_antennas, size=10, replace=True)
        ue_antenna_indices = np.random.choice(num_ue_antennas, size=10, replace=True)
        
        # Log the random selections
        combinations = list(zip(bs_antenna_indices, ue_antenna_indices))
        unique_combinations = list(set(combinations))
        logger.info(f"   Unique antenna combinations: {len(unique_combinations)} out of 10")
        
        logger.info(f"   Selected sample: {sample_idx}")
        logger.info(f"   Selected BS antennas: {bs_antenna_indices}")
        logger.info(f"   Selected UE antennas: {ue_antenna_indices}")
        
        # Create CSI comparison plots (separate amplitude and phase)
        self._plot_random_antenna_amplitude_comparison(sample_idx, bs_antenna_indices, ue_antenna_indices, num_subcarriers)
        self._plot_random_antenna_phase_comparison(sample_idx, bs_antenna_indices, ue_antenna_indices, num_subcarriers)
        
        logger.info("✅ Random CSI samples analysis completed")
    
    def _plot_random_antenna_amplitude_comparison(self, sample_idx, bs_antenna_indices, ue_antenna_indices, num_subcarriers):
        """Plot amplitude comparison for random antenna combinations"""
        logger.info("📊 Creating random antenna amplitude comparison plots...")
        
        # Create subplots: 10 rows, 1 column for 10 plots
        fig, axes = plt.subplots(10, 1, figsize=(12, 25))
        fig.suptitle(f'Random CSI Amplitude Comparison - Sample {sample_idx} (Random BS & UE Antenna Combinations)',
                    fontsize=16, fontweight='bold')
        
        subcarrier_indices = np.arange(num_subcarriers)
        
        for i in range(10):
            ax = axes[i]
            
            bs_idx = bs_antenna_indices[i]
            ue_idx = ue_antenna_indices[i]
            
            # Get CSI data for this specific antenna combination
            pred_csi = self.predictions[sample_idx, bs_idx, ue_idx, :].numpy()
            target_csi = self.targets[sample_idx, bs_idx, ue_idx, :].numpy()
            
            # Plot amplitude
            ax.plot(subcarrier_indices, np.abs(pred_csi), 'r-', linewidth=2, label='Predicted', alpha=0.8)
            ax.plot(subcarrier_indices, np.abs(target_csi), 'b-', linewidth=2, label='Target', alpha=0.8)
            
            # Calculate error metrics
            amp_error = np.mean(np.abs(np.abs(pred_csi) - np.abs(target_csi)))
            
            # Set title and labels
            ax.set_title(f'BS{bs_idx}-UE{ue_idx} - Amplitude (MAE: {amp_error:.4f})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits for better visualization
            max_amp = max(np.abs(pred_csi).max(), np.abs(target_csi).max())
            ax.set_ylim(0, max_amp * 1.1)
            
            # Add MAE value to the plot
            ax.text(0.02, 0.95, f'MAE: {amp_error:.4f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   fontsize=8, verticalalignment='top')
        
        # Set x-axis label for the bottom plot
        axes[-1].set_xlabel('Subcarrier Index', fontsize=10)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'random_antenna_amplitude_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Random antenna amplitude comparison saved: {plot_path}")
    
    def _plot_random_antenna_phase_comparison(self, sample_idx, bs_antenna_indices, ue_antenna_indices, num_subcarriers):
        """Plot phase comparison for random antenna combinations with cosine-based MAE calculation"""
        logger.info("📊 Creating random antenna phase comparison plots...")
        
        # Create subplots: 10 rows, 1 column for 10 plots
        fig, axes = plt.subplots(10, 1, figsize=(12, 25))
        fig.suptitle(f'Random CSI Phase Comparison with Cosine-based MAE - Sample {sample_idx} (Random BS & UE Antenna Combinations)',
                    fontsize=16, fontweight='bold')
        
        subcarrier_indices = np.arange(num_subcarriers)
        
        for i in range(10):
            ax = axes[i]
            
            bs_idx = bs_antenna_indices[i]
            ue_idx = ue_antenna_indices[i]
            
            # Get CSI data for this specific antenna combination
            pred_csi = self.predictions[sample_idx, bs_idx, ue_idx, :].numpy()
            target_csi = self.targets[sample_idx, bs_idx, ue_idx, :].numpy()
            
            # Extract phases
            pred_phase = np.angle(pred_csi)
            target_phase = np.angle(target_csi)
            
            # Apply wrapping to predicted phase based on distance to target
            # For each predicted phase, choose the wrapped version closest to target
            pred_phase_wrapped = np.zeros_like(pred_phase)
            
            for i in range(len(pred_phase)):
                target_val = target_phase[i]
                pred_val = pred_phase[i]
                
                # Calculate all possible wrapped versions of predicted phase
                pred_0 = pred_val  # No wrapping
                pred_plus_2pi = pred_val + 2*np.pi  # Add 2π
                pred_minus_2pi = pred_val - 2*np.pi  # Subtract 2π
                
                # Find the wrapped version closest to target
                wrapped_versions = [pred_0, pred_plus_2pi, pred_minus_2pi]
                distances = [abs(wrapped - target_val) for wrapped in wrapped_versions]
                min_idx = np.argmin(distances)
                
                pred_phase_wrapped[i] = wrapped_versions[min_idx]
            
            # Calculate phase error using cosine-based method to avoid wrapping issues
            # Convert phases to unit vectors and compute angular distance
            pred_cos = np.cos(pred_phase_wrapped)
            pred_sin = np.sin(pred_phase_wrapped)
            target_cos = np.cos(target_phase)
            target_sin = np.sin(target_phase)
            
            # Compute cosine of angular difference
            cos_diff = pred_cos * target_cos + pred_sin * target_sin
            
            # Clamp to avoid numerical issues
            cos_diff = np.clip(cos_diff, -1.0, 1.0)
            
            # Compute angular distance using arccos
            angular_distance = np.arccos(cos_diff)
            
            # Plot phases with wrapped predicted phase
            ax.plot(subcarrier_indices, pred_phase_wrapped, 'r-', linewidth=2, label='Predicted (Wrapped)', alpha=0.8)
            ax.plot(subcarrier_indices, target_phase, 'b-', linewidth=2, label='Target', alpha=0.8)
            
            # Calculate phase error metrics using angular distance
            phase_error = np.mean(angular_distance)
            
            # Set title and labels
            ax.set_title(f'BS{bs_idx}-UE{ue_idx} - Phase Comparison with Cosine-based MAE (MAE: {phase_error:.4f})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Phase (rad)', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add MAE value to the plot
            ax.text(0.02, 0.95, f'MAE: {phase_error:.4f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   fontsize=8, verticalalignment='top')
        
        # Set x-axis label for the bottom plot
        axes[-1].set_xlabel('Subcarrier Index', fontsize=10)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'random_antenna_phase_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Random antenna phase comparison saved: {plot_path}")
    
    def _plot_amplitude_comparison(self, selected_indices, num_bs_antennas, num_ue_antennas, num_subcarriers):
        """Plot amplitude comparison for selected samples"""
        logger.info("📊 Creating amplitude comparison plot...")
        
        # Create figure with subplots (10 rows, 1 column)
        fig, axes = plt.subplots(10, 1, figsize=(12, 25))
        fig.suptitle('CSI Amplitude Comparison: 10 Random Samples', fontsize=16, fontweight='bold')
        
        for i, sample_idx in enumerate(selected_indices):
            # 4D format: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
            # Average across BS and UE antennas for this sample
            pred_amp = torch.abs(self.predictions[sample_idx]).mean(dim=(0, 1)).cpu().numpy()
            target_amp = torch.abs(self.targets[sample_idx]).mean(dim=(0, 1)).cpu().numpy()
            
            # Subcarrier indices
            subcarrier_indices = np.arange(len(pred_amp))
            
            # Plot amplitude comparison
            axes[i].plot(subcarrier_indices, pred_amp, 'b-', label='Predicted', linewidth=2)
            axes[i].plot(subcarrier_indices, target_amp, 'r--', label='Ground Truth', linewidth=2)
            axes[i].set_title(f'Sample {sample_idx}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Amplitude', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=8)
            
            # Add MAE value to the plot
            mae = np.mean(np.abs(pred_amp - target_amp))
            axes[i].text(0.02, 0.95, f'MAE: {mae:.4f}', transform=axes[i].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                        fontsize=8, verticalalignment='top')
        
        # Set x-axis label for the bottom plot
        axes[-1].set_xlabel('Subcarrier Index', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'random_csi_amplitude_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Amplitude comparison plot saved: {plot_path}")
    
    def _plot_phase_comparison(self, selected_indices, num_bs_antennas, num_ue_antennas, num_subcarriers):
        """Plot phase comparison for selected samples"""
        logger.info("📊 Creating phase comparison plot...")
        
        # Create figure with subplots (10 rows, 1 column)
        fig, axes = plt.subplots(10, 1, figsize=(12, 25))
        fig.suptitle('CSI Phase Comparison: 10 Random Samples', fontsize=16, fontweight='bold')
        
        for i, sample_idx in enumerate(selected_indices):
            # 4D format: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
            # Average across BS and UE antennas for this sample
            pred_phase = torch.angle(self.predictions[sample_idx]).mean(dim=(0, 1)).cpu().numpy()
            target_phase = torch.angle(self.targets[sample_idx]).mean(dim=(0, 1)).cpu().numpy()
            
            # Subcarrier indices
            subcarrier_indices = np.arange(len(pred_phase))
            
            # Plot phase comparison
            axes[i].plot(subcarrier_indices, pred_phase, 'b-', label='Predicted', linewidth=2)
            axes[i].plot(subcarrier_indices, target_phase, 'r--', label='Ground Truth', linewidth=2)
            axes[i].set_title(f'Sample {sample_idx}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Phase (rad)', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=8)
            axes[i].set_ylim([-np.pi, np.pi])
            
            # Add phase MAE value to the plot (with wrapping)
            phase_diff = pred_phase - target_phase
            phase_diff_wrapped = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
            phase_mae = np.mean(np.abs(phase_diff_wrapped))
            axes[i].text(0.02, 0.95, f'Phase MAE: {phase_mae:.4f}', transform=axes[i].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                        fontsize=8, verticalalignment='top')
        
        # Set x-axis label for the bottom plot
        axes[-1].set_xlabel('Subcarrier Index', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'random_csi_phase_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Phase comparison plot saved: {plot_path}")
    
    def _compute_pdp(self, csi: np.ndarray, fft_size: int) -> np.ndarray:
        """
        Compute Power Delay Profile from CSI using IFFT
        
        Args:
            csi: Complex CSI values
            fft_size: FFT size for IFFT computation
            
        Returns:
            PDP values
        """
        # Pad or truncate CSI to match FFT size
        if len(csi) < fft_size:
            # Zero-pad
            padded_csi = np.zeros(fft_size, dtype=complex)
            padded_csi[:len(csi)] = csi
        else:
            # Truncate
            padded_csi = csi[:fft_size]
        
        # Compute IFFT
        impulse_response = np.fft.ifft(padded_csi)
        
        # Compute PDP (power of impulse response)
        pdp = np.abs(impulse_response) ** 2
        
        return pdp
    
    def _compute_global_similarity_metrics(self, pred_pdp_all: np.ndarray, target_pdp_all: np.ndarray) -> dict:
        """
        Compute global similarity metrics for PDP comparison.
        
        This implements the most scientific and professional approach for global PDP similarity:
        - MSE (Mean Squared Error): Direct error measurement
        - RMSE (Root Mean Squared Error): MSE square root, same units as original data
        - NMSE (Normalized Mean Squared Error): Normalized by signal power, dimensionless
        - Cosine Similarity: Shape similarity independent of magnitude
        
        Args:
            pred_pdp_all: Predicted PDP arrays [N_samples, FFT_size]
            target_pdp_all: Ground truth PDP arrays [N_samples, FFT_size]
            
        Returns:
            Dictionary containing all global similarity metrics
        """
        logger.info("     Computing MSE, RMSE, NMSE, and Cosine Similarity...")
        
        mse_values = []
        rmse_values = []
        nmse_values = []
        cosine_sim_values = []
        
        total_samples = len(pred_pdp_all)
        
        for i in range(total_samples):
            pred_pdp = pred_pdp_all[i]
            target_pdp = target_pdp_all[i]
            
            # MSE: Mean Squared Error
            mse = np.mean((pred_pdp - target_pdp) ** 2)
            mse_values.append(mse)
            
            # RMSE: Root Mean Squared Error
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
            
            # NMSE: Normalized Mean Squared Error
            signal_power = np.mean(target_pdp ** 2)
            if signal_power < 1e-12:
                nmse = 1.0  # Avoid division by zero
            else:
                nmse = mse / signal_power
            nmse_values.append(nmse)
            
            # Cosine Similarity: Shape similarity independent of magnitude
            pred_norm = np.sqrt(np.sum(pred_pdp ** 2))
            target_norm = np.sqrt(np.sum(target_pdp ** 2))
            
            if pred_norm < 1e-12 or target_norm < 1e-12:
                cosine_sim = 0.0  # Avoid division by zero
            else:
                cosine_sim = np.sum(pred_pdp * target_pdp) / (pred_norm * target_norm)
            cosine_sim_values.append(cosine_sim)
        
        # Convert to numpy arrays
        mse_values = np.array(mse_values)
        rmse_values = np.array(rmse_values)
        nmse_values = np.array(nmse_values)
        cosine_sim_values = np.array(cosine_sim_values)
        
        # Compute statistics
        global_metrics = {
            'mse_mean': float(np.mean(mse_values)),
            'mse_std': float(np.std(mse_values)),
            'mse_median': float(np.median(mse_values)),
            'mse_min': float(np.min(mse_values)),
            'mse_max': float(np.max(mse_values)),
            
            'rmse_mean': float(np.mean(rmse_values)),
            'rmse_std': float(np.std(rmse_values)),
            'rmse_median': float(np.median(rmse_values)),
            'rmse_min': float(np.min(rmse_values)),
            'rmse_max': float(np.max(rmse_values)),
            
            'nmse_mean': float(np.mean(nmse_values)),
            'nmse_std': float(np.std(nmse_values)),
            'nmse_median': float(np.median(nmse_values)),
            'nmse_min': float(np.min(nmse_values)),
            'nmse_max': float(np.max(nmse_values)),
            
            'cosine_sim_mean': float(np.mean(cosine_sim_values)),
            'cosine_sim_std': float(np.std(cosine_sim_values)),
            'cosine_sim_median': float(np.median(cosine_sim_values)),
            'cosine_sim_min': float(np.min(cosine_sim_values)),
            'cosine_sim_max': float(np.max(cosine_sim_values))
        }
        
        logger.info(f"     MSE: {global_metrics['mse_mean']:.6f} ± {global_metrics['mse_std']:.6f}")
        logger.info(f"     RMSE: {global_metrics['rmse_mean']:.6f} ± {global_metrics['rmse_std']:.6f}")
        logger.info(f"     NMSE: {global_metrics['nmse_mean']:.6f} ± {global_metrics['nmse_std']:.6f}")
        logger.info(f"     Cosine Similarity: {global_metrics['cosine_sim_mean']:.6f} ± {global_metrics['cosine_sim_std']:.6f}")
        
        # Return both statistics and raw data for CDF computation
        global_metrics_raw = {
            'mse_values': mse_values,
            'rmse_values': rmse_values,
            'nmse_values': nmse_values,
            'cosine_sim_values': cosine_sim_values
        }
        
        return global_metrics, global_metrics_raw
    
    def _plot_random_pdp_comparison(self, pred_pdp_all: np.ndarray, target_pdp_all: np.ndarray, 
                                   batch_size: int, num_bs_antennas: int, num_ue_antennas: int):
        """Plot random PDP comparison for 10 CSI samples"""
        logger.info("📊 Creating random PDP comparison plots...")
        
        # Randomly select 10 samples
        np.random.seed(42)  # For reproducibility
        total_samples = len(pred_pdp_all)
        selected_indices = np.random.choice(total_samples, size=min(10, total_samples), replace=False)
        
        logger.info(f"   Selected PDP samples: {selected_indices}")
        
        # Create subplots: 10 rows, 1 column for 10 plots
        fig, axes = plt.subplots(10, 1, figsize=(12, 25))
        fig.suptitle(f'Random PDP Comparison: Predicted vs Ground Truth (FFT Size: {self.fft_size})',
                    fontsize=16, fontweight='bold')
        
        # Delay indices (time domain)
        delay_indices = np.arange(self.fft_size)
        
        for i, sample_idx in enumerate(selected_indices):
            ax = axes[i]
            
            # Get PDP data for this sample
            pred_pdp = pred_pdp_all[sample_idx]
            target_pdp = target_pdp_all[sample_idx]
            
            # Plot PDP comparison
            ax.plot(delay_indices, pred_pdp, 'r-', linewidth=2, label='Predicted PDP', alpha=0.8)
            ax.plot(delay_indices, target_pdp, 'b-', linewidth=2, label='Target PDP', alpha=0.8)
            
            # Calculate PDP MAE for this sample
            pdp_mae = np.mean(np.abs(pred_pdp - target_pdp))
            
            # Set title and labels
            ax.set_title(f'Sample {sample_idx} - PDP Comparison (MAE: {pdp_mae:.6f})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Power', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits for better visualization
            max_power = max(pred_pdp.max(), target_pdp.max())
            ax.set_ylim(0, max_power * 1.1)
            
            # Add MAE value to the plot
            ax.text(0.02, 0.95, f'MAE: {pdp_mae:.6f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   fontsize=8, verticalalignment='top')
            
            # Only show x-axis label on the bottom plot
            if i == 9:  # Last row
                ax.set_xlabel('Delay Index', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'pdp_random_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Random PDP comparison saved: {plot_path}")
    
    def _analyze_spectrum(self):
        """Analyze frequency spectrum of CSI data"""
        logger.info("📊 Analyzing frequency spectrum...")
        
        # Convert to complex tensors if needed
        if not self.predictions.is_complex():
            self.predictions = self.predictions + 1j * torch.zeros_like(self.predictions)
        if not self.targets.is_complex():
            self.targets = self.targets + 1j * torch.zeros_like(self.targets)
        
        batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = self.predictions.shape
        
        # Compute spectrum for all samples and antennas
        pred_spectrum_all = []
        target_spectrum_all = []
        
        logger.info(f"   Computing spectrum for {batch_size} samples × {num_bs_antennas * num_ue_antennas} antennas...")
        
        processed_count = 0
        total_spectrum_computations = batch_size * num_bs_antennas * num_ue_antennas
        
        for sample_idx in range(batch_size):
            for bs_idx in range(num_bs_antennas):
                for ue_idx in range(num_ue_antennas):
                    # Get CSI for this sample, BS antenna, and UE antenna
                    pred_csi = self.predictions[sample_idx, bs_idx, ue_idx, :].numpy()
                    target_csi = self.targets[sample_idx, bs_idx, ue_idx, :].numpy()
                    
                    # Compute spectrum using FFT
                    pred_spectrum = self._compute_spectrum(pred_csi)
                    target_spectrum = self._compute_spectrum(target_csi)
                    
                    pred_spectrum_all.append(pred_spectrum)
                    target_spectrum_all.append(target_spectrum)
                    
                    processed_count += 1
                    if processed_count % 50 == 0:  # Progress update every 50 spectrum computations
                        logger.info(f"     Spectrum computation progress: {processed_count}/{total_spectrum_computations} ({processed_count/total_spectrum_computations*100:.1f}%)")
        
        # Convert to numpy arrays
        pred_spectrum_all = np.array(pred_spectrum_all)
        target_spectrum_all = np.array(target_spectrum_all)
        
        logger.info(f"   Spectrum computed: {pred_spectrum_all.shape}")
        
        # Compute spectrum MAE
        spectrum_mae = np.mean(np.abs(pred_spectrum_all - target_spectrum_all), axis=0)
        
        # Create spectrum visualization
        self._plot_spectrum_analysis(pred_spectrum_all, target_spectrum_all, spectrum_mae)
        
        # Create random spectrum comparison
        self._plot_random_spectrum_comparison(pred_spectrum_all, target_spectrum_all, batch_size, num_bs_antennas, num_ue_antennas)
        
        logger.info("✅ Spectrum analysis completed!")
    
    def _compute_spectrum(self, csi: np.ndarray) -> np.ndarray:
        """
        Compute frequency spectrum from CSI using FFT
        
        Args:
            csi: Complex CSI values
            
        Returns:
            Spectrum values (magnitude)
        """
        # Compute FFT
        spectrum = np.fft.fft(csi)
        
        # Return magnitude spectrum
        return np.abs(spectrum)
    
    def _plot_spectrum_analysis(self, pred_spectrum_all: np.ndarray, target_spectrum_all: np.ndarray, spectrum_mae: np.ndarray):
        """Plot spectrum analysis with MAE distribution"""
        logger.info("📊 Creating spectrum analysis plots...")
        
        # Compute CDFs
        spectrum_mae_cdf = self._compute_empirical_cdf(spectrum_mae)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Frequency Spectrum Analysis', fontsize=16, fontweight='bold')
        
        # Spectrum MAE CDF
        ax1.plot(spectrum_mae_cdf[0], spectrum_mae_cdf[1], 'g-', linewidth=2, label='Spectrum MAE')
        ax1.set_xlabel('Spectrum MAE')
        ax1.set_ylabel('Cumulative Probability')
        ax1.set_title('Spectrum MAE CDF')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Spectrum MAE Histogram
        ax2.hist(spectrum_mae, bins=50, alpha=0.7, color='green', edgecolor='black', density=True)
        ax2.set_xlabel('Spectrum MAE')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Spectrum MAE Distribution (Mean: {np.mean(spectrum_mae):.4f})')
        ax2.grid(True, alpha=0.3)
        
        # Average spectrum comparison
        avg_pred_spectrum = np.mean(pred_spectrum_all, axis=0)
        avg_target_spectrum = np.mean(target_spectrum_all, axis=0)
        
        subcarrier_indices = np.arange(len(avg_pred_spectrum))
        ax3.plot(subcarrier_indices, avg_pred_spectrum, 'r-', linewidth=2, label='Average Predicted Spectrum')
        ax3.plot(subcarrier_indices, avg_target_spectrum, 'b-', linewidth=2, label='Average Target Spectrum')
        ax3.set_xlabel('Subcarrier Index')
        ax3.set_ylabel('Spectrum Magnitude')
        ax3.set_title('Average Frequency Spectrum Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Spectrum error per subcarrier
        ax4.plot(subcarrier_indices, spectrum_mae, 'purple', linewidth=2, label='Spectrum MAE per Subcarrier')
        ax4.set_xlabel('Subcarrier Index')
        ax4.set_ylabel('MAE')
        ax4.set_title('Spectrum MAE per Subcarrier')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'spectrum_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Spectrum analysis saved: {plot_path}")
    
    def _plot_random_spectrum_comparison(self, pred_spectrum_all: np.ndarray, target_spectrum_all: np.ndarray,
                                        batch_size: int, num_bs_antennas: int, num_ue_antennas: int):
        """Plot random spectrum comparison for 10 CSI samples"""
        logger.info("📊 Creating random spectrum comparison plots...")
        
        # Randomly select 10 samples
        np.random.seed(42)  # For reproducibility
        total_samples = len(pred_spectrum_all)
        selected_indices = np.random.choice(total_samples, size=min(10, total_samples), replace=False)
        
        logger.info(f"   Selected spectrum samples: {selected_indices}")
        
        # Create subplots: 10 rows, 1 column for 10 plots
        fig, axes = plt.subplots(10, 1, figsize=(12, 25))
        fig.suptitle('Random Spectrum Comparison: Predicted vs Ground Truth',
                    fontsize=16, fontweight='bold')
        
        # Subcarrier indices
        subcarrier_indices = np.arange(pred_spectrum_all.shape[1])
        
        for i, sample_idx in enumerate(selected_indices):
            ax = axes[i]
            
            # Get spectrum data for this sample
            pred_spectrum = pred_spectrum_all[sample_idx]
            target_spectrum = target_spectrum_all[sample_idx]
            
            # Plot spectrum comparison
            ax.plot(subcarrier_indices, pred_spectrum, 'r-', linewidth=2, label='Predicted Spectrum', alpha=0.8)
            ax.plot(subcarrier_indices, target_spectrum, 'b-', linewidth=2, label='Target Spectrum', alpha=0.8)
            
            # Calculate spectrum MAE for this sample
            spectrum_mae = np.mean(np.abs(pred_spectrum - target_spectrum))
            
            # Set title and labels
            ax.set_title(f'Sample {sample_idx} - Spectrum Comparison (MAE: {spectrum_mae:.4f})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Spectrum Magnitude', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits for better visualization
            max_spectrum = max(pred_spectrum.max(), target_spectrum.max())
            ax.set_ylim(0, max_spectrum * 1.1)
            
            # Add MAE value to the plot
            ax.text(0.02, 0.95, f'MAE: {spectrum_mae:.4f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   fontsize=8, verticalalignment='top')
            
            # Only show x-axis label on the bottom plot
            if i == 9:  # Last row
                ax.set_xlabel('Subcarrier Index', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'spectrum_random_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Random spectrum comparison saved: {plot_path}")
    
    def _compute_empirical_cdf(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute empirical cumulative distribution function (CDF) for given data"""
        # Sort data and compute CDF
        sorted_data = np.sort(data)
        n = len(sorted_data)
        cdf_values = np.arange(1, n + 1) / n
        
        return sorted_data, cdf_values
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        logger.info("📋 Generating summary report...")
        
        # Combine all statistics
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_info': {
                'predictions_shape': list(self.predictions.shape),
                'targets_shape': list(self.targets.shape),
                'data_type': str(self.predictions.dtype),
                'fft_size': self.fft_size
            },
            'per_subcarrier_stats': getattr(self, 'per_subcarrier_stats', {}),
            'error_stats': getattr(self, 'error_stats', {}),
            'pdp_stats': getattr(self, 'pdp_stats', {}),
            'spatial_spectrum_stats': getattr(self, 'spatial_spectrum_stats', {})
        }
        
        # Save summary to JSON
        summary_file = self.output_dir / 'metrics' / 'analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Summary report saved: {summary_file}")
        
        # Print summary to console
        self._print_summary(summary)
    
    def _print_summary(self, summary: Dict):
        """Print analysis summary to console"""
        print(f"\n{'='*80}")
        print(f"📊 CSI ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        # Data info
        data_info = summary['data_info']
        print(f"📋 Data Information:")
        print(f"   Predictions shape: {data_info['predictions_shape']}")
        print(f"   Targets shape: {data_info['targets_shape']}")
        print(f"   Data type: {data_info['data_type']}")
        print(f"   FFT size: {data_info['fft_size']}")
        
        # Per-subcarrier stats
        if 'per_subcarrier_stats' in summary and summary['per_subcarrier_stats']:
            stats = summary['per_subcarrier_stats']
            print(f"\n📈 Per-Subcarrier Statistics:")
            print(f"   Predicted Amplitude: Mean={stats['pred_amp_mean']:.6f}, Std={stats['pred_amp_std']:.6f}")
            print(f"   Target Amplitude:     Mean={stats['target_amp_mean']:.6f}, Std={stats['target_amp_std']:.6f}")
            print(f"   Predicted Phase:      Mean={stats['pred_phase_mean']:.6f}, Std={stats['pred_phase_std']:.6f}")
            print(f"   Target Phase:         Mean={stats['target_phase_mean']:.6f}, Std={stats['target_phase_std']:.6f}")
        
        # Error stats
        if 'error_stats' in summary and summary['error_stats']:
            stats = summary['error_stats']
            print(f"\n📊 Error Statistics:")
            print(f"   Amplitude MAE: Mean={stats['amp_mae_mean']:.6f}, Std={stats['amp_mae_std']:.6f}, Median={stats['amp_mae_median']:.6f}")
            print(f"   Phase MAE:     Mean={stats['phase_mae_mean']:.6f}, Std={stats['phase_mae_std']:.6f}, Median={stats['phase_mae_median']:.6f}")
        
        # PDP stats
        if 'pdp_stats' in summary and summary['pdp_stats']:
            stats = summary['pdp_stats']
            print(f"\n⏰ PDP Statistics (FFT Size: {stats['fft_size']}):")
            print(f"   PDP MAE:         Mean={stats['pdp_mae_mean']:.6f}, Std={stats['pdp_mae_std']:.6f}")
            
            # Global similarity metrics
            if 'global_similarity' in stats:
                gs = stats['global_similarity']
                print(f"   Global Similarity Metrics:")
                print(f"     MSE:           Mean={gs['mse_mean']:.6f}, Std={gs['mse_std']:.6f}")
                print(f"     RMSE:          Mean={gs['rmse_mean']:.6f}, Std={gs['rmse_std']:.6f}")
                print(f"     NMSE:          Mean={gs['nmse_mean']:.6f}, Std={gs['nmse_std']:.6f}")
                print(f"     Cosine Sim:    Mean={gs['cosine_sim_mean']:.6f}, Std={gs['cosine_sim_std']:.6f}")
        
        # Spatial spectrum stats
        if 'spatial_spectrum_stats' in summary and summary['spatial_spectrum_stats']:
            stats = summary['spatial_spectrum_stats']
            print(f"\n📡 Spatial Spectrum Statistics (NMSE-based Accuracy):")
            if 'bs_accuracy' in stats:
                bs_stats = stats['bs_accuracy']
                print(f"   BS Spatial Spectrum Accuracy: Mean={bs_stats['mean']:.2f}%, Std={bs_stats['std']:.2f}%, Median={bs_stats['median']:.2f}%")
                print(f"   BS Accuracy Range: [{bs_stats['min']:.2f}%, {bs_stats['max']:.2f}%]")
            if 'ue_accuracy' in stats:
                ue_stats = stats['ue_accuracy']
                print(f"   UE Spatial Spectrum Accuracy: Mean={ue_stats['mean']:.2f}%, Std={ue_stats['std']:.2f}%, Median={ue_stats['median']:.2f}%")
                print(f"   UE Accuracy Range: [{ue_stats['min']:.2f}%, {ue_stats['max']:.2f}%]")
        
        print(f"\n{'='*80}")
        print(f"✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"{'='*80}")


def main():
    """Main entry point for CSI analysis."""
    parser = argparse.ArgumentParser(
        description='CSI Analysis Script for Prism Test Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze test results using config file (auto-detect results path)
  python analyze.py --config configs/sionna.yml
  
  # Analyze with custom FFT size
  python analyze.py --config configs/sionna.yml --fft-size 2048
  
  # Analyze with custom output directory
  python analyze.py --config configs/sionna.yml --output results/custom_analysis
  
  # Analyze with explicit results path (override auto-detection)
  python analyze.py --config configs/sionna.yml --results path/to/custom_results.npz
  
  # Analyze with parallel processing disabled (for debugging)
  python analyze.py --config configs/sionna.yml --no-parallel
  
  # Analyze with custom number of parallel workers
  python analyze.py --config configs/sionna.yml --num-workers 8
  
  # Analyze PolyU results
  python analyze.py --config configs/polyu.yml
        """
    )
    parser.add_argument('--config', required=True, help='Path to configuration file (e.g., configs/sionna.yml)')
    parser.add_argument('--results', help='Path to test results (.npz file) - optional, will auto-detect from config')
    parser.add_argument('--output', help='Output directory for analysis results (optional)')
    parser.add_argument('--fft-size', type=int, default=2048, help='FFT size for PDP computation (default: 2048)')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing for spatial spectrum computation')
    parser.add_argument('--num-workers', type=int, help='Number of parallel workers (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    try:
        print(f"🚀 Starting CSI analysis with config: {args.config}")
        
        analyzer = CSIAnalyzer(
            config_path=args.config,
            results_path=args.results,
            output_dir=args.output,
            fft_size=args.fft_size,
            use_parallel=not args.no_parallel,
            num_workers=args.num_workers
        )
        
        # Run analysis
        analyzer.analyze_csi()
        
        print("✅ CSI analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ CSI analysis failed: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
