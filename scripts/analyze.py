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
- **GPU-accelerated spatial spectrum computation with significant speedup**
- **Vectorized Bartlett algorithm implementation optimized for GPU**
- **Memory-efficient batch processing to handle large datasets**
- NMSE-based accuracy calculation (most scientific and professional approach)
- Signal fidelity percentage measurement: Accuracy (%) = max(0, (1 - NMSE) × 100)
- Automatic device detection with fallback to CPU
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
# Note: matplotlib import removed - plot functionality moved to plot.py
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import h5py
import warnings
from scipy import ndimage

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism.config_loader import ModernConfigLoader


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
                 num_workers: int = None, device: str = None, gpu_id: int = None):
        """
        Initialize CSI analyzer
        
        Args:
            config_path: Path to configuration file
            results_path: Path to test results (.npz file) - optional, will auto-detect from config
            output_dir: Output directory for analysis results
            fft_size: FFT size for PDP computation
            num_workers: Number of parallel workers (deprecated, kept for compatibility)
            device: Device to use for computation ('cuda', 'cpu', or None for auto-detect)
            gpu_id: Specific GPU ID to use (e.g., 0, 1, 2). Only effective when device='cuda'
        """
        self.config_path = Path(config_path)
        self.fft_size = fft_size
        
        # Load configuration first
        self.config_loader = ModernConfigLoader(self.config_path)
        
        # Setup device for GPU computation
        if device is None:
            # Auto-detect device from config or use CUDA if available
            self.device = self._setup_device(gpu_id)
        else:
            if device == 'cuda' and gpu_id is not None:
                # Set specific GPU ID
                self.device = torch.device(f'cuda:{gpu_id}')
                logger.info(f"🎯 Using specific GPU: {gpu_id}")
            else:
                self.device = torch.device(device)
        
        logger.info(f"🔧 Using device: {self.device}")
        
        # Load spatial spectrum configuration
        testing_config = self.config_loader._processed_config.get('testing', {})
        ss_error_config = testing_config.get('ss_error', {})
        self.ignore_amplitude = ss_error_config.get('ignore_amplitude', False)  # Default to False
        self.ss_sample_count = ss_error_config.get('count', 500)  # Default to 500 samples
        
        # Load analysis configuration
        analysis_config = self.config_loader._processed_config.get('analysis', {})
        pas_config = analysis_config.get('pas', {})
        self.pas_enabled = pas_config.get('enabled', True)  # Default to True
        
        logger.info(f"📋 Ignore amplitude for spatial spectrum: {self.ignore_amplitude}")
        logger.info(f"📋 Spatial spectrum sample count: {self.ss_sample_count}")
        logger.info(f"📋 PAS analysis enabled: {self.pas_enabled}")
        logger.info(f"📋 Using GPU for spatial spectrum computation")
        
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
        # Create analysis directory in testing folder (not plots)
        self.analysis_dir = self.output_dir.parent / 'analysis'
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test results
        self._load_results()
        
        logger.info(f"🔍 CSI Analyzer initialized:")
        logger.info(f"   Config path: {self.config_path}")
        logger.info(f"   Results path: {self.results_path}")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"   FFT size for PDP: {self.fft_size}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Data shape: {self.predictions.shape}")
    
    def _setup_device(self, gpu_id: int = None) -> torch.device:
        """Setup computation device based on availability and configuration"""
        # Check if CUDA is available
        if torch.cuda.is_available():
            # Get device from config if available
            system_config = self.config_loader._processed_config.get('system', {})
            device_name = system_config.get('device', 'cuda')
            
            if device_name == 'cuda':
                if gpu_id is not None:
                    # Use specific GPU ID
                    device = torch.device(f'cuda:{gpu_id}')
                    logger.info(f"🎯 Using specific GPU: {gpu_id}")
                    logger.info(f"🚀 CUDA available, using GPU: {torch.cuda.get_device_name(gpu_id)}")
                    logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
                else:
                    # Use default GPU (GPU 0)
                    device = torch.device('cuda')
                    logger.info(f"🚀 CUDA available, using GPU: {torch.cuda.get_device_name()}")
                    logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                return device
        
        # Fallback to CPU
        logger.info("💻 Using CPU for computation")
        return torch.device('cpu')
    
    def _is_target_csi_zero(self, target_csi: torch.Tensor, threshold: float = 1e-12) -> bool:
        """
        Check if target CSI is effectively zero (no ground truth)
        
        Args:
            target_csi: Target CSI tensor
            threshold: Threshold for considering values as zero
            
        Returns:
            True if target CSI is effectively zero, False otherwise
        """
        if target_csi.is_complex():
            # For complex data, check magnitude
            magnitude = torch.abs(target_csi)
            return torch.all(magnitude < threshold).item()
        else:
            # For real data, check absolute values
            return torch.all(torch.abs(target_csi) < threshold).item()

    def _load_results(self):
        """Load test results from .npz file"""
        logger.info("📂 Loading test results...")
        
        # 输出详细的数据文件信息
        print(f"\n📂 正在加载预测结果数据:")
        print(f"{'='*60}")
        print(f"📄 数据文件路径: {self.results_path}")
        
        if not self.results_path.exists():
            print(f"❌ 错误: 数据文件不存在!")
            print(f"   请确保test.py脚本已经运行并生成了预测结果文件")
            print(f"   或者检查文件路径是否正确")
            print(f"{'='*60}")
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        
        # 显示文件信息
        file_size_mb = self.results_path.stat().st_size / 1024 / 1024
        print(f"📊 文件大小: {file_size_mb:.2f} MB")
        print(f"📊 文件格式: .npz (NumPy压缩格式)")
        print(f"{'='*60}")
        
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
        
        # Move data to device
        self.predictions = self.predictions.to(self.device)
        self.targets = self.targets.to(self.device)
        
        # Extract position data (optional)
        if 'test_ue_positions' in data:
            self.ue_positions = torch.from_numpy(data['test_ue_positions']).float().to(self.device)
        else:
            self.ue_positions = None
            
        if 'test_bs_positions' in data:
            self.bs_positions = torch.from_numpy(data['test_bs_positions']).float().to(self.device)
        else:
            self.bs_positions = None
        
        logger.info(f"✅ Results loaded successfully:")
        logger.info(f"   Predictions shape: {self.predictions.shape}")
        logger.info(f"   Targets shape: {self.targets.shape}")
        logger.info(f"   Data type: {self.predictions.dtype}")
        
        # 输出成功加载的详细信息
        print(f"✅ 数据加载成功!")
        print(f"📊 数据形状:")
        print(f"   - predictions: {self.predictions.shape}")
        print(f"   - targets: {self.targets.shape}")
        print(f"   - test_ue_positions: {self.ue_positions.shape if self.ue_positions is not None else 'None'}")
        print(f"   - test_bs_positions: {self.bs_positions.shape if self.bs_positions is not None else 'None'}")
        print(f"📊 数据类型: {self.predictions.dtype}")
        print(f"📊 数据来源: test.py脚本生成的预测结果")
        print(f"{'='*60}")
        
        # Check if CSI data appears to be calibrated
        # Calibrated CSI typically has reference subcarrier values close to 1+0j
        if self.predictions.is_complex():
            # Check if reference subcarrier (middle subcarrier) values are close to 1+0j
            middle_subcarrier = self.predictions.shape[-1] // 2
            ref_pred = self.predictions[:, :, :, middle_subcarrier]
            ref_target = self.targets[:, :, :, middle_subcarrier]
            
            pred_ref_mean = torch.mean(torch.abs(ref_pred)).item()
            target_ref_mean = torch.mean(torch.abs(ref_target)).item()
            
            logger.info(f"   CSI Calibration Status:")
            logger.info(f"     Reference subcarrier index: {middle_subcarrier}")
            logger.info(f"     Predicted reference magnitude mean: {pred_ref_mean:.6f}")
            logger.info(f"     Target reference magnitude mean: {target_ref_mean:.6f}")
            
            # Check if data appears calibrated (reference values close to 1.0)
            if 0.8 <= pred_ref_mean <= 1.2 and 0.8 <= target_ref_mean <= 1.2:
                logger.info(f"     ✅ CSI data appears to be CALIBRATED (reference values ~1.0)")
            else:
                logger.info(f"     ⚠️  CSI data may NOT be calibrated (reference values far from 1.0)")
        
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
        
        # 1. Error distribution analysis
        logger.info("Starting CSI error distribution analysis (amplitude and phase MAE)...")
        self._analyze_csi()
        
        # 2. PDP analysis with NMSE similarity CDF
        logger.info("Starting PDP analysis...")
        self._analyze_pdp()
        
        # 3. Spatial spectrum analysis (if enabled)
        if self.pas_enabled:
            logger.info("Starting spatial spectrum analysis...")
            self._analyze_spatial_spectra()
        else:
            logger.info("Skipping spatial spectrum analysis (disabled in config)")
            self.spatial_spectrum_stats = {}
        
        # 4. Generate summary report
        logger.info("Generating summary report...")
        self._generate_summary_report()
        
        logger.info("✅ CSI analysis completed successfully!")
    
    def _analyze_csi(self):
        """Analyze CSI data including error distributions and random samples"""
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
                    
                    # Skip if target CSI is zero (no ground truth)
                    if self._is_target_csi_zero(self.targets[batch_idx, bs_idx, ue_idx, :]):
                        logger.warning(f"Skipping CSI analysis for sample {batch_idx}, BS antenna {bs_idx}, UE antenna {ue_idx} - target CSI is zero (no ground truth)")
                        continue
                    
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
        amp_mae_cdf = _compute_empirical_cdf(amp_mae_values)
        phase_mae_cdf = _compute_empirical_cdf(phase_mae_values)
        
        # Note: Plot functionality has been moved to plot.py script
        # To create error distribution plots, use the CSIPlotter class in plot.py
        
        logger.info(f"✅ Amplitude MAE and phase MAE analysis completed")
        
        # Compute CDF values for amplitude and phase
        amp_sorted = np.sort(amp_mae_values)
        amp_cdf = np.arange(1, len(amp_sorted) + 1) / len(amp_sorted)
        
        phase_sorted = np.sort(phase_mae_values)
        phase_cdf = np.arange(1, len(phase_sorted) + 1) / len(phase_sorted)
        
        # Store results for summary
        self.error_stats = {
            'amp_mae_mean': float(np.mean(amp_mae_values)),
            'amp_mae_std': float(np.std(amp_mae_values)),
            'amp_mae_median': float(np.median(amp_mae_values)),
            'phase_mae_mean': float(np.mean(phase_mae_values)),
            'phase_mae_std': float(np.std(phase_mae_values)),
            'phase_mae_median': float(np.median(phase_mae_values)),
            'num_subcarriers': num_subcarriers,
            'total_csi_count': len(amp_mae_values),
            'all_amp_mae_values': amp_mae_values.tolist(),
            'all_phase_mae_values': phase_mae_values.tolist(),
            'amp_mae_cdf_values': amp_sorted.tolist(),
            'amp_mae_cdf_probabilities': amp_cdf.tolist(),
            'phase_mae_cdf_values': phase_sorted.tolist(),
            'phase_mae_cdf_probabilities': phase_cdf.tolist()
        }
        
        # Save detailed error data to JSON
        error_file = self.analysis_dir / 'detailed_csi_analysis.json'
        with open(error_file, 'w') as f:
            json.dump({
                'error_statistics': self.error_stats,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"✅ Detailed error analysis saved: {error_file}")
        logger.info(f"   Analyzed {len(amp_mae_values)} CSI samples")
        
        # Random CSI samples analysis
        logger.info("📊 Analyzing random CSI samples with random antenna combinations...")
        
        # Get data dimensions - only support 4D CSI data
        if len(self.predictions.shape) != 4:
            raise ValueError(f"Only 4D CSI data is supported. Current shape: {self.predictions.shape}. Expected: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]")
        
        batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = self.predictions.shape
        
        # Randomly select 50 BS-UE antenna combinations (increased from 20 for better diversity)
        # 使用时间戳作为随机种子，增加每次运行的多样性
        import time
        dynamic_seed = int(time.time() * 1000) % 10000  # 使用毫秒时间戳
        np.random.seed(dynamic_seed)
        logger.info(f"🔀 Using dynamic random seed: {dynamic_seed} for antenna sampling")
        demo_samples = []
        max_attempts = 100  # Maximum attempts to find valid samples
        attempts = 0
        
        while len(demo_samples) < 50 and attempts < max_attempts:
            sample_idx = np.random.choice(batch_size)
            bs_idx = np.random.choice(num_bs_antennas)
            ue_idx = np.random.choice(num_ue_antennas)
            
            # Get CSI for this combination
            pred_csi = self.predictions[sample_idx, bs_idx, ue_idx, :].cpu().numpy()
            target_csi = self.targets[sample_idx, bs_idx, ue_idx, :].cpu().numpy()
            
            # Skip if target CSI is zero (no ground truth)
            if self._is_target_csi_zero(torch.from_numpy(target_csi)):
                logger.warning(f"Skipping CSI sample attempt {attempts+1} - sample {sample_idx}, BS antenna {bs_idx}, UE antenna {ue_idx} - target CSI is zero (no ground truth)")
                attempts += 1
                continue
            
            demo_samples.append({
                'sample_idx': int(sample_idx),
                'bs_antenna_idx': int(bs_idx),
                'ue_antenna_idx': int(ue_idx),
                'predicted_csi_real': pred_csi.real.tolist(),
                'predicted_csi_imag': pred_csi.imag.tolist(),
                'target_csi_real': target_csi.real.tolist(),
                'target_csi_imag': target_csi.imag.tolist()
            })
            attempts += 1
        
        if len(demo_samples) < 50:
            logger.warning(f"Only found {len(demo_samples)} valid CSI samples out of {max_attempts} attempts. Some samples had zero target CSI.")
        
        # Save demo CSI samples to JSON
        demo_file = self.analysis_dir / 'demo_csi_samples.json'
        with open(demo_file, 'w') as f:
            json.dump({
                'demo_samples': demo_samples,
                'num_samples': len(demo_samples),
                'csi_shape': [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers],
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"✅ Demo CSI samples saved: {demo_file}")
        logger.info(f"   Saved {len(demo_samples)} CSI samples")
    
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
        valid_sample_info = []  # Store sample info for valid PDPs
        
        logger.info(f"   Computing PDP for {batch_size} samples × {total_antennas} antennas...")
        
        processed_count = 0
        total_pdp_computations = batch_size * num_bs_antennas * num_ue_antennas
        
        for sample_idx in range(batch_size):
            # 4D format: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
            for bs_idx in range(num_bs_antennas):
                    for ue_idx in range(num_ue_antennas):
                        # Get CSI for this sample, BS antenna, and UE antenna
                        pred_csi = self.predictions[sample_idx, bs_idx, ue_idx, :].cpu().numpy()
                        target_csi = self.targets[sample_idx, bs_idx, ue_idx, :].cpu().numpy()
                        
                        # Skip if target CSI is zero (no ground truth)
                        if self._is_target_csi_zero(self.targets[sample_idx, bs_idx, ue_idx, :]):
                            logger.warning(f"Skipping PDP analysis for sample {sample_idx}, BS antenna {bs_idx}, UE antenna {ue_idx} - target CSI is zero (no ground truth)")
                            continue
                        
                        pred_pdp = self._compute_pdp(pred_csi, self.fft_size)
                        target_pdp = self._compute_pdp(target_csi, self.fft_size)
                        
                        pred_pdp_all.append(pred_pdp)
                        target_pdp_all.append(target_pdp)
                        valid_sample_info.append({
                            'sample_idx': sample_idx,
                            'bs_idx': bs_idx,
                            'ue_idx': ue_idx
                        })
                        
                        processed_count += 1
                        if processed_count % 50 == 0:  # Progress update every 50 PDP computations
                            logger.info(f"     PDP computation progress: {processed_count}/{total_pdp_computations} ({processed_count/total_pdp_computations*100:.1f}%)")
        
        # Convert to numpy arrays
        pred_pdp_all = np.array(pred_pdp_all)
        target_pdp_all = np.array(target_pdp_all)
        
        logger.info(f"   PDP computed: {pred_pdp_all.shape}")
        
        # Compute PDP MAE
        pdp_mae = np.mean(np.abs(pred_pdp_all - target_pdp_all), axis=0)
        
        # Compute all similarity metrics in a single loop
        logger.info("   Computing all PDP similarity metrics...")
        cosine_similarity_values = []
        relative_error_similarity_values = []
        spectral_correlation_values = []
        log_spectral_distance_values = []
        bhattacharyya_coefficient_values = []
        jensen_shannon_divergence_values = []
        ssim_values = []
        nmse_values = []
        
        for i in range(len(pred_pdp_all)):
            pred_pdp = pred_pdp_all[i]
            target_pdp = target_pdp_all[i]
            
            # Cosine Similarity using traditional cosine similarity method
            pred_tensor = torch.from_numpy(pred_pdp).float()
            target_tensor = torch.from_numpy(target_pdp).float()
            cosine_similarity = self._compute_cosine_similarity(pred_tensor, target_tensor)
            cosine_similarity_values.append(cosine_similarity)
            
            # Relative Error Similarity
            relative_error = np.mean(np.abs(pred_pdp - target_pdp) / (np.abs(target_pdp) + 1e-12))
            # Map to [0, 1], smaller error = higher similarity
            # Use exponential decay to map relative error to similarity
            relative_error_similarity = np.exp(-relative_error)
            relative_error_similarity_values.append(relative_error_similarity)
            
            # Spectral Correlation Coefficient
            scc = _compute_spectral_correlation_coefficient(pred_pdp, target_pdp)
            spectral_correlation_values.append(scc)
            
            # Log Spectral Distance
            lsd = _compute_log_spectral_distance(pred_pdp, target_pdp)
            log_spectral_distance_values.append(lsd)
            
            # Bhattacharyya Coefficient
            bc = _compute_bhattacharyya_coefficient(pred_pdp, target_pdp)
            bhattacharyya_coefficient_values.append(bc)
            
            # Jensen-Shannon Divergence
            jsd = _compute_jensen_shannon_divergence(pred_pdp, target_pdp)
            jensen_shannon_divergence_values.append(jsd)
            
            # 1D SSIM for PDP
            ssim = self._compute_ssim_1d(pred_pdp, target_pdp)
            ssim_values.append(ssim)
            
            # Normalized Mean Squared Error (NMSE) for PDP using standard NMSE similarity method
            pred_tensor = torch.from_numpy(pred_pdp).float()
            target_tensor = torch.from_numpy(target_pdp).float()
            nmse_similarity = self._compute_nmse_similarity(pred_tensor, target_tensor)
            nmse_values.append(nmse_similarity)
        
        logger.info(f"✅ All PDP similarity metrics computed")
        
        # Note: Plot functionality has been moved to plot.py script
        # To create PDP comparison plots, use the CSIPlotter class in plot.py
        
        # Store basic PDP stats for summary
        self.pdp_stats = {
            'pdp_mae_mean': float(np.mean(pdp_mae)),
            'pdp_mae_std': float(np.std(pdp_mae)),
            'fft_size': self.fft_size,
            'total_pdp_samples': len(pred_pdp_all)
        }
        
        # Save detailed PDP similarity analysis to separate file
        logger.info("   Saving detailed PDP similarity analysis...")
        detailed_pdp_data = {
            'pdp_analysis_info': {
                'fft_size': self.fft_size,
                'total_pdp_samples': len(pred_pdp_all),
                'pdp_shape': pred_pdp_all.shape,
                'timestamp': datetime.now().isoformat()
            },
            'similarity_metrics': {
                'cosine_similarity': cosine_similarity_values,
                'relative_error_similarity': relative_error_similarity_values,
                'spectral_correlation': spectral_correlation_values,
                'log_spectral_distance': log_spectral_distance_values,
                'bhattacharyya_coefficient': bhattacharyya_coefficient_values,
                'jensen_shannon_divergence': jensen_shannon_divergence_values,
                'ssim': ssim_values,
                'nmse': nmse_values
            }
        }
        
        # Save detailed PDP analysis to JSON
        detailed_pdp_file = self.analysis_dir / 'detailed_pdp_analysis.json'
        with open(detailed_pdp_file, 'w') as f:
            json.dump(detailed_pdp_data, f, indent=2)
        
        logger.info(f"✅ Detailed PDP analysis saved: {detailed_pdp_file}")
        logger.info(f"   Saved {len(pred_pdp_all)} PDP similarity values for each metric")
        
        # Save 10 random PDP samples for demo
        logger.info("   Saving 10 random PDP samples for demo...")
        np.random.seed(42)  # For reproducibility
        
        # Ensure we don't try to select more samples than available
        num_demo_samples = min(10, len(pred_pdp_all))
        demo_pdp_indices = np.random.choice(len(pred_pdp_all), size=num_demo_samples, replace=False)
        
        demo_pdp_samples = []
        for i, idx in enumerate(demo_pdp_indices):
            # Use the stored sample information for valid PDPs
            sample_info = valid_sample_info[idx]
            
            demo_pdp_samples.append({
                'sample_idx': int(sample_info['sample_idx']),
                'bs_antenna_idx': int(sample_info['bs_idx']),
                'ue_antenna_idx': int(sample_info['ue_idx']),
                'predicted_pdp': pred_pdp_all[idx].tolist(),
                'target_pdp': target_pdp_all[idx].tolist(),
                'pdp_length': len(pred_pdp_all[idx])
            })
        
        # Save demo PDP samples to JSON
        demo_pdp_file = self.analysis_dir / 'demo_pdp_samples.json'
        with open(demo_pdp_file, 'w') as f:
            json.dump({
                'demo_pdp_samples': demo_pdp_samples,
                'num_samples': len(demo_pdp_samples),
                'pdp_shape': pred_pdp_all.shape,
                'fft_size': self.fft_size,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"✅ Demo PDP samples saved: {demo_pdp_file}")
        logger.info(f"   Saved {len(demo_pdp_samples)} PDP samples")
    
    def _analyze_spatial_spectra(self):
        """Analyze spatial spectra for BS and UE antenna arrays using proper array geometry"""
        logger.info("📊 Analyzing spatial spectra with proper array geometry...")
        
        # Convert to complex tensors if needed
        if not self.predictions.is_complex():
            self.predictions = self.predictions + 1j * torch.zeros_like(self.predictions)
        if not self.targets.is_complex():
            self.targets = self.targets + 1j * torch.zeros_like(self.targets)
        
        # Get antenna configuration from config
        bs_config = self.config_loader._processed_config['base_station']['antenna_array']
        ue_config = self.config_loader._processed_config['user_equipment']['antenna_array']
        ofdm_config = self.config_loader._processed_config['base_station']['ofdm']
        analysis_config = self.config_loader._processed_config.get('analysis', {}).get('pas', {})
        
        bs_antenna_count = self.config_loader.num_bs_antennas
        ue_antenna_count = self.config_loader.ue_antenna_count
        
        # Log detailed antenna structure information
        logger.info("📡 Antenna Structure Information:")
        logger.info(f"   BS antennas: {bs_antenna_count}, UE antennas: {ue_antenna_count}")
        logger.info(f"   BS array config: {bs_config['configuration']}")
        logger.info(f"   UE array config: {ue_config['configuration']}")
        # Calculate antenna spacing from center frequency for logging
        c = 3e8  # Speed of light in m/s
        wavelength = c / float(ofdm_config['center_frequency'])
        antenna_spacing = wavelength / 2
        
        logger.info(f"   BS spacing: x={antenna_spacing:.3f}m, y={antenna_spacing:.3f}m (calculated from {float(ofdm_config['center_frequency'])/1e9:.2f} GHz)")
        logger.info(f"   UE spacing: x={antenna_spacing:.3f}m, y={antenna_spacing:.3f}m (calculated from {float(ofdm_config['center_frequency'])/1e9:.2f} GHz)")
        
        # Log OFDM configuration
        logger.info("📡 OFDM Configuration:")
        logger.info(f"   Center frequency: {float(ofdm_config['center_frequency'])/1e9:.2f} GHz")
        logger.info(f"   Bandwidth: {float(ofdm_config['bandwidth'])/1e6:.1f} MHz")
        logger.info(f"   Subcarrier spacing: {float(ofdm_config['subcarrier_spacing'])/1e3:.1f} kHz")
        logger.info(f"   Number of subcarriers: {ofdm_config['num_subcarriers']}")
        
        # Log analysis parameters
        logger.info("📊 Analysis Parameters:")
        logger.info(f"   Ignore amplitude: {analysis_config['ignore_amplitude']}")
        logger.info(f"   Sample count: {analysis_config['count']}")
        
        # Determine CSI data shape - only support 4D CSI data
        if len(self.predictions.shape) != 4:
            raise ValueError(f"Only 4D CSI data is supported. Current shape: {self.predictions.shape}. Expected: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]")
        
        batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = self.predictions.shape
        
        # Log detailed CSI information
        logger.info("📊 CSI Data Information:")
        logger.info(f"   CSI shape: {self.predictions.shape}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   BS antennas: {num_bs_antennas}")
        logger.info(f"   UE antennas: {num_ue_antennas}")
        logger.info(f"   Subcarriers: {num_subcarriers}")
        logger.info(f"   Total CSI elements: {batch_size * num_bs_antennas * num_ue_antennas * num_subcarriers:,}")
        logger.info(f"   CSI data type: {self.predictions.dtype}")
        logger.info(f"   CSI memory size: {self.predictions.element_size() * self.predictions.nelement() / 1024**2:.2f} MB")
        
        # Initialize results storage
        spatial_spectrum_stats = {}
        
        # Analyze BS spatial spectrum if BS antennas > 1
        if num_bs_antennas > 1:
            logger.info("   Computing BS spatial spectra...")
            bs_results = self._compute_spatial_spectrum_analysis(
                self.predictions, self.targets, 
                antenna_type="BS",
                array_config=bs_config,
                ofdm_config=ofdm_config,
                analysis_config=analysis_config,
                batch_size=batch_size,
                num_antennas=num_ue_antennas,  # BS端：为每个UE天线计算空间谱
                num_subcarriers=num_subcarriers
            )
            spatial_spectrum_stats['bs_analysis'] = bs_results
            logger.info(f"   BS Analysis completed: {bs_results['total_samples']} samples")
        
        # Analyze UE spatial spectrum if UE antennas > 1
        if num_ue_antennas > 1:
            logger.info("   Computing UE spatial spectra...")
            ue_results = self._compute_spatial_spectrum_analysis(
                self.predictions, self.targets,
                antenna_type="UE",
                array_config=ue_config,
                ofdm_config=ofdm_config,
                analysis_config=analysis_config,
                batch_size=batch_size,
                num_antennas=num_bs_antennas,  # UE端：为每个BS天线计算空间谱
                num_subcarriers=num_subcarriers
            )
            spatial_spectrum_stats['ue_analysis'] = ue_results
            logger.info(f"   UE Analysis completed: {ue_results['total_samples']} samples")
        
        # Store results for summary
        self.spatial_spectrum_stats = spatial_spectrum_stats
        
        # Create unified PAS analysis file
        self._create_unified_pas_analysis()
        
        logger.info("✅ Spatial spectrum analysis completed!")
    
    def _create_unified_pas_analysis(self):
        """Create unified PAS analysis file with similarity metrics only"""
        logger.info("📊 Creating unified PAS analysis file...")
        
        unified_data = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'total_bs_samples': 0,
                'total_ue_samples': 0
            },
            'bs_analysis': {},
            'ue_analysis': {}
        }
        
        # Process BS analysis if available
        if 'bs_analysis' in self.spatial_spectrum_stats:
            bs_data = self.spatial_spectrum_stats['bs_analysis']
            unified_data['analysis_info']['total_bs_samples'] = bs_data['total_samples']
            
            # Load similarity data from file
            similarity_file = self.analysis_dir / 'detailed_spatial_spectrum_similarity_bs.json'
            if similarity_file.exists():
                with open(similarity_file, 'r') as f:
                    bs_similarity_data = json.load(f)
                    unified_data['bs_analysis'] = {
                        'antenna_type': 'BS',
                        'array_shape': bs_data['array_shape'],
                        'is_2d_array': bs_data['is_2d_array'],
                        'total_samples': bs_data['total_samples'],
                        'similarity_metrics': bs_similarity_data['similarity_metrics']
                    }
        
        # Process UE analysis if available
        if 'ue_analysis' in self.spatial_spectrum_stats:
            ue_data = self.spatial_spectrum_stats['ue_analysis']
            unified_data['analysis_info']['total_ue_samples'] = ue_data['total_samples']
            
            # Load similarity data from file
            similarity_file = self.analysis_dir / 'detailed_spatial_spectrum_similarity_ue.json'
            if similarity_file.exists():
                with open(similarity_file, 'r') as f:
                    ue_similarity_data = json.load(f)
                    unified_data['ue_analysis'] = {
                        'antenna_type': 'UE',
                        'array_shape': ue_data['array_shape'],
                        'is_2d_array': ue_data['is_2d_array'],
                        'total_samples': ue_data['total_samples'],
                        'similarity_metrics': ue_similarity_data['similarity_metrics']
                    }
        
        # Save unified PAS analysis
        unified_file = self.analysis_dir / 'detailed_pas_analysis.json'
        with open(unified_file, 'w') as f:
            json.dump(unified_data, f, indent=2)
        
        logger.info(f"✅ Unified PAS analysis saved: {unified_file}")
        
        # Create demo samples file
        self._create_demo_pas_samples()
    
    def _create_demo_pas_samples(self):
        """Create demo PAS samples file with 10 BS and 10 UE samples"""
        logger.info("📊 Creating demo PAS samples file...")
        
        demo_data = {
            'demo_info': {
                'timestamp': datetime.now().isoformat(),
                'bs_samples': 0,
                'ue_samples': 0
            },
            'bs_samples': [],
            'ue_samples': []
        }
        
        # Load BS demo samples
        bs_demo_file = self.analysis_dir / 'demo_pas_samples_bs.json'
        if bs_demo_file.exists():
            with open(bs_demo_file, 'r') as f:
                bs_demo_data = json.load(f)
                demo_data['bs_samples'] = bs_demo_data['demo_samples'][:10]  # Take first 10 samples
                demo_data['demo_info']['bs_samples'] = len(demo_data['bs_samples'])
        
        # Load UE demo samples
        ue_demo_file = self.analysis_dir / 'demo_pas_samples_ue.json'
        if ue_demo_file.exists():
            with open(ue_demo_file, 'r') as f:
                ue_demo_data = json.load(f)
                demo_data['ue_samples'] = ue_demo_data['demo_samples'][:10]  # Take first 10 samples
                demo_data['demo_info']['ue_samples'] = len(demo_data['ue_samples'])
        
        # Save unified demo file
        demo_file = self.analysis_dir / 'demo_pas_samples.json'
        with open(demo_file, 'w') as f:
            json.dump(demo_data, f, indent=2)
        
        logger.info(f"✅ Demo PAS samples saved: {demo_file}")
        logger.info(f"   BS samples: {demo_data['demo_info']['bs_samples']}")
        logger.info(f"   UE samples: {demo_data['demo_info']['ue_samples']}")
    
    def _compute_spatial_spectrum_analysis(self, predictions: torch.Tensor, targets: torch.Tensor,
                                         antenna_type: str, array_config: dict, ofdm_config: dict,
                                         analysis_config: dict, batch_size: int, num_antennas: int,
                                         num_subcarriers: int) -> dict:
        """Compute spatial spectrum analysis for given antenna type"""
        
        # Parse array configuration
        array_shape = self._parse_array_configuration(array_config['configuration'])
        
        # Get OFDM parameters
        center_freq = float(ofdm_config['center_frequency'])
        
        # Calculate antenna spacing from center frequency (half wavelength)
        c = 3e8  # Speed of light in m/s
        wavelength = c / center_freq
        antenna_spacing = wavelength / 2
        
        # Use calculated spacing for both x and y directions
        spacing_x = antenna_spacing
        spacing_y = antenna_spacing
        bandwidth = float(ofdm_config['bandwidth'])
        subcarrier_spacing = float(ofdm_config['subcarrier_spacing'])
        
        # Get analysis parameters
        ignore_amplitude = analysis_config['ignore_amplitude']
        max_samples = analysis_config['count']
        
        logger.info(f"   Array shape: {array_shape}, Spacing: ({spacing_x}, {spacing_y})")
        logger.info(f"   Center freq: {center_freq/1e9:.2f} GHz, Bandwidth: {bandwidth/1e6:.1f} MHz")
        
        # Determine if we need 1D or 2D spatial spectrum
        is_2d_array = len(array_shape) == 2 and array_shape[1] > 1
        
        # Generate all possible sample combinations: (batch_idx, antenna_idx, subcarrier_idx)
        all_samples = []
        for batch_idx in range(batch_size):
            for antenna_idx in range(num_antennas):
                for subcarrier_idx in range(num_subcarriers):
                    all_samples.append((batch_idx, antenna_idx, subcarrier_idx))
        
        # Randomly sample if we have more samples than configured limit
        if len(all_samples) > max_samples:
            np.random.seed(42)  # For reproducible results
            selected_samples = np.random.choice(len(all_samples), size=max_samples, replace=False)
            all_samples = [all_samples[i] for i in selected_samples]
            logger.info(f"   Randomly selected {max_samples} samples from {batch_size * num_antennas * num_subcarriers} total samples")
        
        # Process samples with GPU batch computation
        logger.info(f"   🚀 Starting GPU batch processing of spatial spectrum samples...")
        
        # Prepare batch data
        valid_samples = []
        pred_csi_list = []
        target_csi_list = []
        subcarrier_indices_list = []
        
        for batch_idx, antenna_idx, subcarrier_idx in all_samples:
            # Extract CSI for this sample
            if antenna_type == "BS":
                # BS端：提取到特定UE天线的信道向量
                # antenna_idx 是UE天线索引 (0-3)
                pred_csi = predictions[batch_idx, :, antenna_idx, subcarrier_idx]  # [num_bs_antennas]
                target_csi = targets[batch_idx, :, antenna_idx, subcarrier_idx]
            else:  # UE
                # UE端：提取到特定BS天线的信道向量
                # antenna_idx 是BS天线索引 (0-63)
                pred_csi = predictions[batch_idx, antenna_idx, :, subcarrier_idx]  # [num_ue_antennas]
                target_csi = targets[batch_idx, antenna_idx, :, subcarrier_idx]
            
            # Skip if target CSI is zero (no ground truth)
            if self._is_target_csi_zero(target_csi):
                logger.warning(f"Skipping spatial spectrum analysis for sample {batch_idx}, antenna {antenna_idx}, subcarrier {subcarrier_idx}, {antenna_type} antennas - target CSI is zero (no ground truth)")
                continue
            
            valid_samples.append((batch_idx, antenna_idx, subcarrier_idx))
            pred_csi_list.append(pred_csi)
            target_csi_list.append(target_csi)
            subcarrier_indices_list.append(subcarrier_idx)
        
        if not pred_csi_list:
            logger.warning(f"   No valid samples found for {antenna_type} spatial spectrum analysis!")
            return {
                'antenna_type': antenna_type,
                'array_shape': array_shape,
                'is_2d_array': is_2d_array,
                'total_samples': 0,
                'similarity_file': None
            }
        
        # Process in chunks to manage GPU memory
        chunk_size = 512  # Process 512 samples at a time to avoid GPU OOM
        total_samples = len(pred_csi_list)
        
        logger.info(f"   Processing {total_samples} valid samples in GPU batches of {chunk_size}...")
        
        # Initialize result storage
        pred_spatial_spectra_list = []
        target_spatial_spectra_list = []
        
        import time
        start_time = time.time()
        
        # Process in chunks
        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            chunk_indices = slice(chunk_start, chunk_end)
            
            logger.info(f"     Processing chunk {chunk_start//chunk_size + 1}/{(total_samples-1)//chunk_size + 1}: samples {chunk_start}-{chunk_end-1}")
            
            # Stack CSI data for this chunk
            pred_csi_chunk = torch.stack(pred_csi_list[chunk_indices], dim=0)  # [chunk_size, num_antennas]
            target_csi_chunk = torch.stack(target_csi_list[chunk_indices], dim=0)  # [chunk_size, num_antennas]
            subcarrier_indices_chunk = torch.tensor(subcarrier_indices_list[chunk_indices], device=self.device)  # [chunk_size]
            
            # Compute spatial spectra for this chunk
            pred_chunk_spectra = self._compute_spatial_spectrum_batch(
                pred_csi_chunk, antenna_type, array_shape, spacing_x, spacing_y,
                center_freq, subcarrier_spacing, subcarrier_indices_chunk, ignore_amplitude, is_2d_array
            )
            
            target_chunk_spectra = self._compute_spatial_spectrum_batch(
                target_csi_chunk, antenna_type, array_shape, spacing_x, spacing_y,
                center_freq, subcarrier_spacing, subcarrier_indices_chunk, ignore_amplitude, is_2d_array
            )
            
            # Store results
            pred_spatial_spectra_list.append(pred_chunk_spectra)
            target_spatial_spectra_list.append(target_chunk_spectra)
            
            # Clear GPU cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all chunks
        pred_spatial_spectra = torch.cat(pred_spatial_spectra_list, dim=0)
        target_spatial_spectra = torch.cat(target_spatial_spectra_list, dim=0)
        
        computation_time = time.time() - start_time
        logger.info(f"   ✅ GPU batch computation completed in {computation_time:.2f}s!")
        
        # Report GPU memory usage if available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            logger.info(f"   📊 GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        # Compute similarity metrics for all samples
        logger.info(f"   Computing similarity metrics for {len(pred_csi_list)} samples...")
        
        results = []
        for i in range(len(pred_csi_list)):
            batch_idx, antenna_idx, subcarrier_idx = valid_samples[i]
            
            # Get spatial spectra for this sample
            pred_spectrum = pred_spatial_spectra[i]  # [azimuth] or [azimuth, elevation]
            target_spectrum = target_spatial_spectra[i]  # [azimuth] or [azimuth, elevation]
            
            # Compute similarity metrics
            similarity_metrics = self._compute_spatial_spectrum_similarity_gpu(pred_spectrum, target_spectrum)
            
            # Calculate wavelength for this sample
            subcarrier_freq = center_freq + (subcarrier_idx - predictions.shape[-1]//2) * subcarrier_spacing
            wavelength = 3e8 / subcarrier_freq
            
            results.append({
                'batch_idx': batch_idx,
                'antenna_idx': antenna_idx,
                'subcarrier_idx': subcarrier_idx,
                'predicted_spectrum': pred_spectrum.cpu().numpy().tolist(),
                'target_spectrum': target_spectrum.cpu().numpy().tolist(),
                'similarity_metrics': similarity_metrics,
                'subcarrier_frequency': float(subcarrier_freq),
                'wavelength': float(wavelength),
                'antenna_type': antenna_type
            })
        
        logger.info(f"   Completed processing of {len(results)} spatial spectra")
        
        # Extract similarity values for separate storage
        cosine_similarity_values = []
        nmse_values = []
        ssim_values = []
        
        for sample in results:
            similarity_metrics = sample['similarity_metrics']
            cosine_similarity_values.append(similarity_metrics['cosine_similarity'])
            nmse_values.append(similarity_metrics['nmse'])
            ssim_values.append(similarity_metrics['ssim'])
        
        # Randomly select 5 samples for demo
        logger.info(f"   Selecting 5 random samples for {antenna_type} demo...")
        np.random.seed(42)  # For reproducibility
        demo_indices = np.random.choice(len(results), size=min(20, len(results)), replace=False)
        
        demo_samples = []
        for idx in demo_indices:
            sample = results[idx]
            demo_samples.append({
                'sample_idx': int(idx),
                'batch_idx': sample['batch_idx'],
                'antenna_idx': sample['antenna_idx'],
                'subcarrier_idx': sample['subcarrier_idx'],
                'predicted_spatial_spectrum': sample['predicted_spectrum'],
                'target_spatial_spectrum': sample['target_spectrum'],
                'similarity_metrics': sample['similarity_metrics'],
                'array_shape': array_shape,
                'is_2d_array': is_2d_array,
                'wavelength': sample['wavelength'],
                'subcarrier_frequency': sample['subcarrier_frequency'],
                'antenna_type': sample['antenna_type']
            })
        
        # Save demo samples to JSON
        demo_file = self.analysis_dir / f'demo_pas_samples_{antenna_type.lower()}.json'
        with open(demo_file, 'w') as f:
            json.dump({
                'antenna_type': antenna_type,
                'demo_samples': demo_samples,
                'num_demo_samples': len(demo_samples),
                'array_configuration': array_config,
                'array_shape': array_shape,
                'is_2d_array': is_2d_array,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"✅ Demo {antenna_type} spatial spectrum samples saved: {demo_file}")
        logger.info(f"   Saved {len(demo_samples)} demo samples")
        
        # Save detailed similarity analysis to separate file
        similarity_file = self.analysis_dir / f'detailed_spatial_spectrum_similarity_{antenna_type.lower()}.json'
        with open(similarity_file, 'w') as f:
            json.dump({
                'antenna_type': antenna_type,
                'array_configuration': array_config,
                'array_shape': array_shape,
                'is_2d_array': is_2d_array,
                'total_samples': len(results),
                'timestamp': datetime.now().isoformat(),
                'similarity_metrics': {
                    'cosine_similarity': cosine_similarity_values,
                    'nmse': nmse_values,
                    'ssim': ssim_values
                }
            }, f, indent=2)
        
        logger.info(f"✅ Detailed {antenna_type} spatial spectrum similarity values saved: {similarity_file}")
        
        # Note: Detailed spatial spectrum data is no longer saved separately
        # All similarity metrics are consolidated in detailed_pas_analysis.json
        
        return {
            'antenna_type': antenna_type,
            'array_shape': array_shape,
            'is_2d_array': is_2d_array,
            'total_samples': len(results),
            'similarity_file': str(similarity_file)
        }
    
    def _parse_array_configuration(self, config_str: str) -> tuple:
        """Parse antenna array configuration string (e.g., '8x8', '2x2', '1x4')"""
        try:
            parts = config_str.split('x')
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
            elif len(parts) == 1:
                return (int(parts[0]), 1)  # Linear array
            else:
                raise ValueError(f"Invalid array configuration: {config_str}")
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse array configuration '{config_str}': {e}")
    
    # NOTE: Legacy single sample processing functions removed - replaced with GPU batch processing
    # def _process_single_spatial_spectrum(...) - removed
    # def _compute_single_spatial_spectrum(...) - removed

    # NOTE: _compute_single_spatial_spectrum function removed - replaced with GPU batch processing
    
    def _compute_1d_spatial_spectrum(self, csi: torch.Tensor, array_shape: tuple, 
                                   spacing: float, wavelength: float) -> torch.Tensor:
        """Compute 1D spatial spectrum for linear array using 1D CSI vector"""
        num_antennas = array_shape[0]
        
        # Ensure csi is 1D vector
        if csi.dim() != 1:
            raise ValueError(f"Expected 1D CSI vector, got shape {csi.shape}")
        
        # Create angle grid for 1D (azimuth only)
        azimuth_angles = torch.linspace(0, 180, 180, dtype=torch.float32, device=csi.device)  # 0-180 degrees
        
        # Vectorized computation for all angles at once
        spatial_spectrum = self._compute_1d_spatial_spectrum_vectorized(
            csi, num_antennas, spacing, wavelength, azimuth_angles
        )
        
        return spatial_spectrum
    
    def _compute_1d_spatial_spectrum_vectorized(self, csi: torch.Tensor, num_antennas: int,
                                              spacing: float, wavelength: float, 
                                              azimuth_angles: torch.Tensor) -> torch.Tensor:
        """Vectorized 1D spatial spectrum computation for GPU"""
        # Convert angles to radians
        az_rad = azimuth_angles * torch.pi / 180.0  # [num_angles]
        
        # Create antenna positions (linear array along x-axis)
        antenna_positions = torch.arange(num_antennas, dtype=torch.float32, device=csi.device) * spacing  # [num_antennas]
        
        # Compute phase progression for all angles at once
        # az_rad: [num_angles], antenna_positions: [num_antennas]
        # Result: [num_angles, num_antennas]
        phase_progression = 2 * torch.pi * antenna_positions.unsqueeze(0) * torch.cos(az_rad).unsqueeze(1) / wavelength
        
        # Create steering vectors for all angles
        steering_vectors = torch.exp(1j * phase_progression)  # [num_angles, num_antennas]
        
        # Compute Bartlett beamformer output for all angles
        # csi: [num_antennas], steering_vectors: [num_angles, num_antennas]
        responses = torch.sum(csi.unsqueeze(0) * torch.conj(steering_vectors), dim=1)  # [num_angles]
        
        # Compute power spectrum
        spatial_spectrum = torch.abs(responses) ** 2
        
        return spatial_spectrum
    
    def _compute_2d_spatial_spectrum(self, csi: torch.Tensor, array_shape: tuple,
                                   spacing_x: float, spacing_y: float, wavelength: float) -> torch.Tensor:
        """Compute 2D spatial spectrum for planar array using 1D CSI vector"""
        num_antennas_x, num_antennas_y = array_shape
        
        # Ensure csi is 1D vector
        if csi.dim() != 1:
            raise ValueError(f"Expected 1D CSI vector, got shape {csi.shape}")
        
        # Create angle grid for 2D (azimuth and elevation)
        azimuth_angles = torch.linspace(0, 360, 181, dtype=torch.float32, device=csi.device)  # 0-360 degrees, 2-degree steps
        elevation_angles = torch.linspace(0, 90, 46, dtype=torch.float32, device=csi.device)  # 0-90 degrees, 2-degree steps
        
        # Vectorized computation for all angle pairs at once
        spatial_spectrum = self._compute_2d_spatial_spectrum_vectorized(
            csi, num_antennas_x, num_antennas_y, spacing_x, spacing_y, wavelength, 
            azimuth_angles, elevation_angles
        )
        
        return spatial_spectrum
    
    def _compute_2d_spatial_spectrum_vectorized(self, csi: torch.Tensor, num_antennas_x: int, num_antennas_y: int,
                                              spacing_x: float, spacing_y: float, wavelength: float,
                                              azimuth_angles: torch.Tensor, elevation_angles: torch.Tensor) -> torch.Tensor:
        """Vectorized 2D spatial spectrum computation for GPU"""
        # Convert angles to radians
        az_rad = azimuth_angles * torch.pi / 180.0  # [num_azimuth]
        el_rad = elevation_angles * torch.pi / 180.0  # [num_elevation]
        
        # Create antenna positions (planar array)
        antenna_positions_x = torch.arange(num_antennas_x, dtype=torch.float32, device=csi.device) * spacing_x  # [num_antennas_x]
        antenna_positions_y = torch.arange(num_antennas_y, dtype=torch.float32, device=csi.device) * spacing_y  # [num_antennas_y]
        
        # Create meshgrid for 2D positions
        pos_x, pos_y = torch.meshgrid(antenna_positions_x, antenna_positions_y, indexing='ij')  # [num_antennas_x, num_antennas_y]
        pos_x_flat = pos_x.flatten()  # [num_antennas]
        pos_y_flat = pos_y.flatten()  # [num_antennas]
        
        # Create angle meshgrids
        az_grid, el_grid = torch.meshgrid(az_rad, el_rad, indexing='ij')  # [num_azimuth, num_elevation]
        
        # Compute phase progression for all angle pairs at once
        # az_grid: [num_azimuth, num_elevation], el_grid: [num_azimuth, num_elevation]
        # pos_x_flat: [num_antennas], pos_y_flat: [num_antennas]
        # Result: [num_azimuth, num_elevation, num_antennas]
        phase_progression = 2 * torch.pi * (
            pos_x_flat.unsqueeze(0).unsqueeze(0) * torch.sin(el_grid).unsqueeze(2) * torch.cos(az_grid).unsqueeze(2) + 
            pos_y_flat.unsqueeze(0).unsqueeze(0) * torch.sin(el_grid).unsqueeze(2) * torch.sin(az_grid).unsqueeze(2)
        ) / wavelength
        
        # Create steering vectors for all angle pairs
        steering_vectors = torch.exp(1j * phase_progression)  # [num_azimuth, num_elevation, num_antennas]
        
        # Compute Bartlett beamformer output for all angle pairs
        # csi: [num_antennas], steering_vectors: [num_azimuth, num_elevation, num_antennas]
        responses = torch.sum(csi.unsqueeze(0).unsqueeze(0) * torch.conj(steering_vectors), dim=2)  # [num_azimuth, num_elevation]
        
        # Compute power spectrum
        spatial_spectrum = torch.abs(responses) ** 2
        
        return spatial_spectrum
    
    def _compute_spatial_spectrum_batch(self, csi_batch: torch.Tensor, antenna_type: str, 
                                      array_shape: tuple, spacing_x: float, spacing_y: float,
                                      center_freq: float, subcarrier_spacing: float, 
                                      subcarrier_indices: torch.Tensor, ignore_amplitude: bool,
                                      is_2d_array: bool) -> torch.Tensor:
        """
        Compute spatial spectrum for a batch of CSI samples on GPU
        
        Args:
            csi_batch: CSI tensor of shape [batch_size, num_antennas]
            antenna_type: "BS" or "UE"
            array_shape: Antenna array shape tuple
            spacing_x, spacing_y: Antenna spacing
            center_freq: Center frequency
            subcarrier_spacing: Subcarrier spacing
            subcarrier_indices: Subcarrier indices for each sample [batch_size]
            ignore_amplitude: Whether to ignore amplitude
            is_2d_array: Whether array is 2D
            
        Returns:
            spatial_spectra: Tensor of shape [batch_size, azimuth_angles, elevation_angles] or [batch_size, azimuth_angles]
        """
        batch_size = csi_batch.shape[0]
        device = csi_batch.device
        
        # Apply amplitude handling if needed
        if ignore_amplitude:
            csi_batch = torch.exp(1j * torch.angle(csi_batch))
        
        # Calculate wavelengths for each sample
        subcarrier_freqs = center_freq + (subcarrier_indices - csi_batch.shape[-1]//2) * subcarrier_spacing
        wavelengths = 3e8 / subcarrier_freqs  # [batch_size]
        
        if is_2d_array:
            # 2D spatial spectrum
            num_antennas_x, num_antennas_y = array_shape
            
            # Create angle grids
            azimuth_angles = torch.linspace(0, 360, 181, dtype=torch.float32, device=device)
            elevation_angles = torch.linspace(0, 90, 46, dtype=torch.float32, device=device)
            
            # Convert to radians
            az_rad = azimuth_angles * torch.pi / 180.0
            el_rad = elevation_angles * torch.pi / 180.0
            
            # Create antenna positions
            antenna_positions_x = torch.arange(num_antennas_x, dtype=torch.float32, device=device) * spacing_x
            antenna_positions_y = torch.arange(num_antennas_y, dtype=torch.float32, device=device) * spacing_y
            
            # Create meshgrids
            pos_x, pos_y = torch.meshgrid(antenna_positions_x, antenna_positions_y, indexing='ij')
            pos_x_flat = pos_x.flatten()
            pos_y_flat = pos_y.flatten()
            
            az_grid, el_grid = torch.meshgrid(az_rad, el_rad, indexing='ij')
            
            # Initialize output tensor
            spatial_spectra = torch.zeros(batch_size, len(azimuth_angles), len(elevation_angles), device=device)
            
            # Process each sample (can be further vectorized if needed)
            for i in range(batch_size):
                wavelength = wavelengths[i]
                
                # Compute phase progression for all angle pairs
                phase_progression = 2 * torch.pi * (
                    pos_x_flat.unsqueeze(0).unsqueeze(0) * torch.sin(el_grid).unsqueeze(2) * torch.cos(az_grid).unsqueeze(2) + 
                    pos_y_flat.unsqueeze(0).unsqueeze(0) * torch.sin(el_grid).unsqueeze(2) * torch.sin(az_grid).unsqueeze(2)
                ) / wavelength
                
                # Create steering vectors
                steering_vectors = torch.exp(1j * phase_progression)  # [num_azimuth, num_elevation, num_antennas]
                
                # Compute Bartlett beamformer output
                responses = torch.sum(csi_batch[i].unsqueeze(0).unsqueeze(0) * torch.conj(steering_vectors), dim=2)
                
                # Compute power spectrum
                spatial_spectra[i] = torch.abs(responses) ** 2
        else:
            # 1D spatial spectrum
            num_antennas = array_shape[0]
            
            # Create angle grid
            azimuth_angles = torch.linspace(0, 180, 180, dtype=torch.float32, device=device)
            az_rad = azimuth_angles * torch.pi / 180.0
            
            # Create antenna positions
            antenna_positions = torch.arange(num_antennas, dtype=torch.float32, device=device) * spacing_x
            
            # Initialize output tensor
            spatial_spectra = torch.zeros(batch_size, len(azimuth_angles), device=device)
            
            # Process each sample
            for i in range(batch_size):
                wavelength = wavelengths[i]
                
                # Compute phase progression for all angles
                phase_progression = 2 * torch.pi * antenna_positions.unsqueeze(0) * torch.cos(az_rad).unsqueeze(1) / wavelength
                
                # Create steering vectors
                steering_vectors = torch.exp(1j * phase_progression)  # [num_angles, num_antennas]
                
                # Compute Bartlett beamformer output
                responses = torch.sum(csi_batch[i].unsqueeze(0) * torch.conj(steering_vectors), dim=1)
                
                # Compute power spectrum
                spatial_spectra[i] = torch.abs(responses) ** 2
        
        return spatial_spectra
    
    def _compute_spatial_spectrum_similarity_gpu(self, pred_spectrum: torch.Tensor, 
                                               target_spectrum: torch.Tensor) -> dict:
        """Compute similarity metrics between predicted and target spatial spectra on GPU"""
        
        # Flatten spectra for computation
        pred_flat = pred_spectrum.flatten()
        target_flat = target_spectrum.flatten()
        
        # Normalize spectra using consistent normalization
        # Use the maximum sum across both predicted and target spectra for consistent normalization
        pred_sum = torch.sum(pred_flat)
        target_sum = torch.sum(target_flat)
        max_sum = torch.maximum(pred_sum, target_sum)
        pred_norm = pred_flat / (max_sum + 1e-12)
        target_norm = target_flat / (max_sum + 1e-12)
        
        # Compute various similarity metrics
        # 1. Cosine similarity using traditional cosine similarity method
        cosine_sim = self._compute_cosine_similarity(pred_spectrum, target_spectrum)
        
        # 2. Normalized Mean Squared Error (NMSE) using standard NMSE similarity method
        nmse = self._compute_nmse_similarity(pred_spectrum, target_spectrum)
        
        # 3. Structural Similarity Index (SSIM) - always use 2D version for spatial spectra
        if pred_spectrum.dim() == 2 and target_spectrum.dim() == 2:
            # Use 2D SSIM for 2D spatial spectra (azimuth x elevation)
            ssim = self._compute_ssim_2d(pred_spectrum, target_spectrum)
        elif pred_spectrum.dim() == 1 and target_spectrum.dim() == 1:
            # Use 1D SSIM for 1D spatial spectra
            pred_np = pred_norm.cpu().numpy()
            target_np = target_norm.cpu().numpy()
            ssim = self._compute_ssim_1d(pred_np, target_np)
        else:
            # Unsupported dimension combination
            raise ValueError(f"Unsupported spectrum dimensions: pred={pred_spectrum.dim()}D, target={target_spectrum.dim()}D. Only 1D and 2D spectra are supported.")
        
        return {
            'cosine_similarity': float(cosine_sim),
            'nmse': float(nmse),
            'ssim': float(ssim)
        }
    
    def _create_1d_steering_vector(self, azimuth: float, num_antennas: int, 
                                  spacing: float, wavelength: float, device: torch.device) -> torch.Tensor:
        """Create steering vector for 1D linear array"""
        # Convert angle to radians
        az_rad = (azimuth * torch.pi / 180.0).to(device)
        
        # Create antenna positions (linear array along x-axis)
        antenna_positions = torch.arange(num_antennas, dtype=torch.float32, device=device) * spacing
        
        # Compute phase progression
        phase_progression = 2 * torch.pi * antenna_positions * torch.cos(az_rad) / wavelength
        
        # Create steering vector
        steering_vector = torch.exp(1j * phase_progression)
        
        return steering_vector
    
    def _create_2d_steering_vector(self, azimuth: float, elevation: float, 
                                 num_antennas_x: int, num_antennas_y: int,
                                 spacing_x: float, spacing_y: float, wavelength: float,
                                 device: torch.device) -> torch.Tensor:
        """Create steering vector for 2D planar array"""
        # Convert angles to radians
        az_rad = (azimuth * torch.pi / 180.0).to(device)
        el_rad = (elevation * torch.pi / 180.0).to(device)
        
        # Create antenna positions (planar array)
        antenna_positions_x = torch.arange(num_antennas_x, dtype=torch.float32, device=device) * spacing_x
        antenna_positions_y = torch.arange(num_antennas_y, dtype=torch.float32, device=device) * spacing_y
        
        # Create meshgrid for 2D positions
        pos_x, pos_y = torch.meshgrid(antenna_positions_x, antenna_positions_y, indexing='ij')
        
        # Compute phase progression
        phase_progression = 2 * torch.pi * (
            pos_x * torch.sin(el_rad) * torch.cos(az_rad) + 
            pos_y * torch.sin(el_rad) * torch.sin(az_rad)
        ) / wavelength
        
        # Flatten to 1D for steering vector
        steering_vector = torch.exp(1j * phase_progression.flatten())
        
        return steering_vector
    
    
    def _compute_ssim_2d(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
        """
        Compute 2D Structural Similarity Index (SSIM) between two 2D tensors using custom implementation.
        
        This uses a custom SSIM implementation instead of pytorch-msssim library.
        Assumes pred and target are 2D tensors of shape [height, width] (e.g., [azimuth, elevation]).
        
        Args:
            pred: Predicted 2D tensor
            target: Target 2D tensor  
            window_size: Window size for SSIM computation (default: 11)
            sigma: Gaussian window sigma parameter (default: 1.5)
            
        Returns:
            SSIM value as float
        """
        # Ensure same shape
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
        
        # Normalize using consistent energy normalization (same as other similarity metrics)
        pred_sum = torch.sum(pred)
        target_sum = torch.sum(target)
        max_sum = torch.maximum(pred_sum, target_sum)
        pred_norm = pred / (max_sum + 1e-12)
        target_norm = target / (max_sum + 1e-12)
        
        # Custom 2D SSIM implementation
        # Convert to numpy for easier computation
        pred_np = pred_norm.cpu().numpy()
        target_np = target_norm.cpu().numpy()
        
        # Ensure window size is not larger than image dimensions
        height, width = pred_np.shape
        window_size = min(window_size, height, width)
        
        if window_size < 3:
            window_size = min(height, width)
        
        # Create Gaussian window
        def create_gaussian_window(window_size, sigma):
            """Create a 2D Gaussian window"""
            center = window_size // 2
            x, y = np.meshgrid(np.arange(window_size) - center, np.arange(window_size) - center)
            gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            return gaussian / np.sum(gaussian)
        
        gaussian_window = create_gaussian_window(window_size, sigma)
        
        # Compute local means using convolution
        from scipy import ndimage
        mu_pred = ndimage.convolve(pred_np, gaussian_window, mode='constant')
        mu_target = ndimage.convolve(target_np, gaussian_window, mode='constant')
        
        # Compute local variances and covariance
        mu_pred_sq = ndimage.convolve(pred_np**2, gaussian_window, mode='constant')
        mu_target_sq = ndimage.convolve(target_np**2, gaussian_window, mode='constant')
        mu_pred_target = ndimage.convolve(pred_np * target_np, gaussian_window, mode='constant')
        
        sigma_pred_sq = mu_pred_sq - mu_pred**2
        sigma_target_sq = mu_target_sq - mu_target**2
        sigma_pred_target = mu_pred_target - mu_pred * mu_target
        
        # SSIM constants
        c1 = 0.01 ** 2  # (0.01 * data_range) ** 2, where data_range = 1.0
        c2 = 0.03 ** 2  # (0.03 * data_range) ** 2, where data_range = 1.0
        
        # Compute SSIM
        numerator = (2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)
        denominator = (mu_pred**2 + mu_target**2 + c1) * (sigma_pred_sq + sigma_target_sq + c2)
        
        ssim_map = numerator / (denominator + 1e-12)
        ssim_value = np.mean(ssim_map)
        
        # Return as float
        return float(ssim_value)

    def _compute_nmse_similarity(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute NMSE similarity using maximum possible error normalization.
        
        Maximum Possible Error Normalization NMSE Similarity Definition:
        Similarity = 1 - MSE / (max(x_i²) + max(y_i²))
        
        where:
        - MSE = (1/n) * sum((x_i - y_i)²): mean squared error
        - max(x_i²): maximum squared value in predicted vector
        - max(y_i²): maximum squared value in target vector
        
        This gives a similarity value in range [0, 1] where:
        - 1.0 = perfect match (MSE = 0)
        - 0.0 = worst match (MSE = max(x_i²) + max(y_i²))
        - Higher values indicate better similarity
        
        Args:
            pred: Predicted values tensor
            target: Target values tensor
            
        Returns:
            Maximum possible error normalized NMSE similarity value in range [0, 1]
        """
        # Flatten tensors for computation
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Compute MSE
        mse = torch.mean((pred_flat - target_flat) ** 2)
        
        # Compute maximum squared values
        pred_max_squared = torch.max(pred_flat ** 2)
        target_max_squared = torch.max(target_flat ** 2)
        max_possible_error = pred_max_squared + target_max_squared
        
        # Avoid division by zero
        if max_possible_error < 1e-12:
            return 1.0  # Perfect similarity if no variation
        
        # Compute maximum possible error normalized similarity
        similarity = 1.0 - (mse / max_possible_error)
        
        # Clip to valid range [0, 1]
        similarity = torch.clamp(similarity, 0.0, 1.0)
        
        return float(similarity)

    def _compute_cosine_similarity(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute traditional cosine similarity with Min-Max normalization preprocessing.
        
        Steps:
        1. Apply Min-Max normalization to both vectors: (x - min) / (max - min)
        2. Compute cosine similarity: cosine_similarity = (A · B) / (||A|| × ||B||)
        
        where:
        - A · B is the dot product of normalized vectors A and B
        - ||A|| and ||B|| are the L2 norms (Euclidean norms) of the normalized vectors
        
        This gives a similarity value in range [-1, 1] where:
        - 1.0 = perfect positive correlation (same direction)
        - 0.0 = orthogonal vectors (no correlation)
        - -1.0 = perfect negative correlation (opposite direction)
        - Values are normalized to [0, 1] for consistency: (cosine_sim + 1) / 2
        
        Args:
            pred: Predicted values tensor
            target: Target values tensor
            
        Returns:
            Cosine similarity value normalized to range [0, 1]
        """
        # Flatten tensors for computation
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Min-Max normalization for predicted vector
        pred_min = torch.min(pred_flat)
        pred_max = torch.max(pred_flat)
        if pred_max - pred_min > 1e-12:
            pred_normalized = (pred_flat - pred_min) / (pred_max - pred_min)
        else:
            # If all values are the same, set to 0.5 (middle of [0,1] range)
            pred_normalized = torch.full_like(pred_flat, 0.5)
        
        # Min-Max normalization for target vector
        target_min = torch.min(target_flat)
        target_max = torch.max(target_flat)
        if target_max - target_min > 1e-12:
            target_normalized = (target_flat - target_min) / (target_max - target_min)
        else:
            # If all values are the same, set to 0.5 (middle of [0,1] range)
            target_normalized = torch.full_like(target_flat, 0.5)
        
        # Compute dot product of normalized vectors
        dot_product = torch.dot(pred_normalized, target_normalized)
        
        # Compute L2 norms of normalized vectors
        pred_norm = torch.norm(pred_normalized)
        target_norm = torch.norm(target_normalized)
        
        # Avoid division by zero
        if pred_norm < 1e-12 or target_norm < 1e-12:
            if torch.allclose(pred_normalized, target_normalized, atol=1e-12):
                return 1.0  # Perfect similarity if both normalized vectors are equal
            else:
                return 0.0  # No similarity if one is zero vector but not equal
        
        # Compute cosine similarity
        cosine_sim = dot_product / (pred_norm * target_norm)
        
        # Normalize to [0, 1] range: (cosine_sim + 1) / 2
        # This maps [-1, 1] to [0, 1] where 0.5 means orthogonal vectors
        normalized_similarity = (cosine_sim + 1.0) / 2.0
        
        # Clip to valid range [0, 1] (should be unnecessary but for safety)
        normalized_similarity = torch.clamp(normalized_similarity, 0.0, 1.0)
        
        return float(normalized_similarity)

    def _compute_ssim_1d(self, pred_pdp: np.ndarray, target_pdp: np.ndarray) -> float:
        """
        Compute 1D Structural Similarity Index (SSIM) between two 1D PDPs.
        
        This implements 1D SSIM specifically for Power Delay Profiles (PDPs).
        PDPs are 1D signals representing power vs delay, so we use 1D SSIM computation.
        
        Args:
            pred_pdp: Predicted PDP as 1D numpy array
            target_pdp: Target PDP as 1D numpy array
            
        Returns:
            SSIM value as float in [0, 1] where 1 is most similar
        """
        # Ensure same length
        if len(pred_pdp) != len(target_pdp):
            raise ValueError(f"Length mismatch: {len(pred_pdp)} vs {len(target_pdp)}")
        
        # Normalize PDPs using maximum value normalization only
        pred_max = np.max(pred_pdp)
        target_max = np.max(target_pdp)
        
        # Use the maximum value across both sequences for consistent normalization
        global_max = max(pred_max, target_max) + 1e-8
        
        pred_norm = pred_pdp / global_max
        target_norm = target_pdp / global_max
        
        # 1D SSIM computation
        # Use a sliding window approach for 1D SSIM
        window_size = min(11, len(pred_pdp))  # Window size, capped by signal length
        if window_size < 3:
            window_size = len(pred_pdp)  # Use full signal if too short
        
        # Compute local means using convolution
        kernel = np.ones(window_size) / window_size
        
        mu_pred = np.convolve(pred_norm, kernel, mode='valid')
        mu_target = np.convolve(target_norm, kernel, mode='valid')
        
        # Compute local variances and covariance
        mu_pred_padded = np.pad(mu_pred, (window_size//2, window_size//2), mode='edge')
        mu_target_padded = np.pad(mu_target, (window_size//2, window_size//2), mode='edge')
        
        # Ensure same length
        min_len = min(len(pred_norm), len(mu_pred_padded))
        pred_norm = pred_norm[:min_len]
        target_norm = target_norm[:min_len]
        mu_pred_padded = mu_pred_padded[:min_len]
        mu_target_padded = mu_target_padded[:min_len]
        
        sigma_pred_sq = np.mean((pred_norm - mu_pred_padded) ** 2)
        sigma_target_sq = np.mean((target_norm - mu_target_padded) ** 2)
        sigma_pred_target = np.mean((pred_norm - mu_pred_padded) * (target_norm - mu_target_padded))
        
        # SSIM constants
        c1 = 0.01 ** 2  # (0.01 * data_range) ** 2, where data_range = 1.0
        c2 = 0.03 ** 2  # (0.03 * data_range) ** 2, where data_range = 1.0
        
        # Compute SSIM
        numerator = (2 * np.mean(mu_pred_padded) * np.mean(mu_target_padded) + c1) * (2 * sigma_pred_target + c2)
        denominator = (np.mean(mu_pred_padded) ** 2 + np.mean(mu_target_padded) ** 2 + c1) * (sigma_pred_sq + sigma_target_sq + c2)
        
        ssim_value = numerator / (denominator + 1e-12)
        
        return float(ssim_value)
    
    def _compute_pdp(self, csi: np.ndarray, fft_size: int) -> np.ndarray:
        """Compute Power Delay Profile (PDP) from CSI using IFFT"""
        # Pad CSI to fft_size
        if len(csi) < fft_size:
            padded_csi = np.zeros(fft_size, dtype=complex)
            padded_csi[:len(csi)] = csi
        else:
            padded_csi = csi[:fft_size]
        
        # Compute IFFT to get time domain
        time_domain = np.fft.ifft(padded_csi)
        
        # Compute power delay profile
        pdp = np.abs(time_domain) ** 2
        
        # Energy normalization: divide by sum
        total_energy = np.sum(pdp)
        if total_energy < 1e-8:
            return pdp
        pdp = pdp / total_energy
        
        return pdp
    
    def _compute_global_similarity_metrics(self, pred_pdp_all: np.ndarray, target_pdp_all: np.ndarray) -> tuple:
        """Compute global similarity metrics for PDP comparison"""
        mse_values = []
        rmse_values = []
        nmse_values = []
        cosine_sim_values = []
        
        for i in range(len(pred_pdp_all)):
            pred_pdp = pred_pdp_all[i]
            target_pdp = target_pdp_all[i]
            
            # MSE
            mse = np.mean((pred_pdp - target_pdp) ** 2)
            mse_values.append(mse)
            
            # RMSE
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
            
            # NMSE using standard NMSE similarity method
            pred_tensor = torch.from_numpy(pred_pdp).float()
            target_tensor = torch.from_numpy(target_pdp).float()
            nmse = self._compute_nmse_similarity(pred_tensor, target_tensor)
            nmse_values.append(nmse)
            
            # Cosine similarity using traditional cosine similarity method
            cosine_sim = self._compute_cosine_similarity(pred_tensor, target_tensor)
            
            cosine_sim_values.append(cosine_sim)
        
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
        
        global_metrics_raw = {
            'mse_values': np.array(mse_values),
            'rmse_values': np.array(rmse_values),
            'nmse_values': np.array(nmse_values),
            'cosine_sim_values': np.array(cosine_sim_values)
        }
        
        return global_metrics, global_metrics_raw
    
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
        summary_file = self.analysis_dir / 'analysis_summary.json'
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
            
            # PDP similarity metrics - show summary info
            if 'total_pdp_samples' in stats:
                print(f"   Total PDP samples analyzed: {stats['total_pdp_samples']}")
                print(f"   Detailed similarity values saved to: detailed_pdp_analysis.json")
        
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


def _compute_empirical_cdf(data: np.ndarray) -> tuple:
    """Compute empirical CDF for given data"""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y = np.arange(1, n + 1) / n
    return sorted_data, y


def _compute_spectral_correlation_coefficient(pred_pdp: np.ndarray, target_pdp: np.ndarray) -> float:
    """Compute Spectral Correlation Coefficient (SCC) between two PDPs"""
    pred_flat = pred_pdp.flatten()
    target_flat = target_pdp.flatten()
    
    # Compute correlation coefficient
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    # Handle NaN case
    if np.isnan(correlation):
        return 0.0
    
    # Map from [-1, 1] to [0, 1] where 1 is most similar
    return (correlation + 1.0) / 2.0


def _compute_log_spectral_distance(pred_pdp: np.ndarray, target_pdp: np.ndarray) -> float:
    """Compute Log Spectral Distance (LSD) using maximum possible error normalization"""
    pred_flat = pred_pdp.flatten()
    target_flat = target_pdp.flatten()
    
    # Compute MSE
    mse = np.mean((pred_flat - target_flat) ** 2)
    
    # Compute maximum squared values
    pred_max_squared = np.max(pred_flat ** 2)
    target_max_squared = np.max(target_flat ** 2)
    max_possible_error = pred_max_squared + target_max_squared
    
    # Avoid division by zero
    if max_possible_error < 1e-12:
        return 1.0  # Perfect similarity if no variation
    
    # Compute maximum possible error normalized similarity
    similarity = 1.0 - (mse / max_possible_error)
    
    # Clip to valid range [0, 1]
    similarity = np.clip(similarity, 0.0, 1.0)
    
    return float(similarity)


def _compute_bhattacharyya_coefficient(pred_pdp: np.ndarray, target_pdp: np.ndarray) -> float:
    """Compute Bhattacharyya Coefficient (BC) between two PDPs"""
    pred_flat = pred_pdp.flatten()
    target_flat = target_pdp.flatten()
    
    # Normalize to make them probability distributions
    pred_norm = pred_flat / (np.sum(pred_flat) + 1e-10)
    target_norm = target_flat / (np.sum(target_flat) + 1e-10)
    
    # Compute Bhattacharyya coefficient
    bc = np.sum(np.sqrt(pred_norm * target_norm))
    
    # BC is already in [0, 1] where 1 is most similar
    return bc


def _compute_jensen_shannon_divergence(pred_pdp: np.ndarray, target_pdp: np.ndarray) -> float:
    """Compute Jensen-Shannon Divergence (JSD) using maximum possible error normalization"""
    pred_flat = pred_pdp.flatten()
    target_flat = target_pdp.flatten()
    
    # Compute MSE
    mse = np.mean((pred_flat - target_flat) ** 2)
    
    # Compute maximum squared values
    pred_max_squared = np.max(pred_flat ** 2)
    target_max_squared = np.max(target_flat ** 2)
    max_possible_error = pred_max_squared + target_max_squared
    
    # Avoid division by zero
    if max_possible_error < 1e-12:
        return 1.0  # Perfect similarity if no variation
    
    # Compute maximum possible error normalized similarity
    similarity = 1.0 - (mse / max_possible_error)
    
    # Clip to valid range [0, 1]
    similarity = np.clip(similarity, 0.0, 1.0)
    
    return float(similarity)


def main():
    """Main entry point for CSI analysis."""
    parser = argparse.ArgumentParser(
        description='CSI Analysis Script for Prism Test Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze test results using config file (auto-detect results path)
  python analyze.py --config configs/sionna.yml
  
  # Analyze with GPU acceleration (default auto-detect)
  python analyze.py --config configs/sionna.yml --device cuda
  
  # Force CPU computation (useful for debugging)
  python analyze.py --config configs/sionna.yml --device cpu
  
  # Analyze with custom FFT size
  python analyze.py --config configs/sionna.yml --fft-size 2048
  
  # Analyze with custom output directory
  python analyze.py --config configs/sionna.yml --output results/custom_analysis
  
  # Analyze with explicit results path (override auto-detection)
  python analyze.py --config configs/sionna.yml --results path/to/custom_results.npz
  
  # Analyze PolyU results with GPU acceleration
  python analyze.py --config configs/polyu.yml --device cuda
  
  # Analyze with specific GPU ID
  python analyze.py --config configs/polyu.yml --device cuda --gpu 0
  
  # Analyze on GPU 1 specifically
  python analyze.py --config configs/polyu.yml --device cuda --gpu 1
        """
    )
    parser.add_argument('--config', required=True, help='Path to configuration file (e.g., configs/sionna.yml)')
    parser.add_argument('--results', help='Path to test results (.npz file) - optional, will auto-detect from config')
    parser.add_argument('--output', help='Output directory for analysis results (optional)')
    parser.add_argument('--fft-size', type=int, default=2048, help='FFT size for PDP computation (default: 2048)')
    parser.add_argument('--num-workers', type=int, help='Number of parallel workers (deprecated, kept for compatibility)')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='auto', help='Device to use for computation (default: auto)')
    parser.add_argument('--gpu', type=int, default=None, help='Specific GPU ID to use (e.g., 0, 1, 2). Only effective when --device cuda')
    
    args = parser.parse_args()
    
    try:
        print(f"🚀 Starting CSI analysis with config: {args.config}")
        
        # 显示数据文件读取位置信息
        print(f"\n📁 数据文件读取位置信息:")
        print(f"{'='*60}")
        
        # 创建临时配置加载器来获取路径信息
        temp_config_loader = ModernConfigLoader(args.config)
        output_paths = temp_config_loader.get_output_paths()
        
        # 显示预测结果文件路径
        if args.results:
            results_file = Path(args.results)
            print(f"📄 手动指定的预测结果文件: {results_file}")
        else:
            # 自动检测路径
            predictions_dir = output_paths['predictions_dir']
            results_file = Path(predictions_dir) / 'test_results.npz'
            print(f"📄 自动检测的预测结果文件: {results_file}")
        
        # 显示输出目录
        if args.output:
            output_dir = Path(args.output)
            print(f"📂 手动指定的输出目录: {output_dir}")
        else:
            plots_dir = output_paths['plots_dir']
            output_dir = Path(plots_dir)
            print(f"📂 自动检测的输出目录: {output_dir}")
        
        # 显示分析目录
        analysis_dir = output_dir.parent / 'analysis'
        print(f"📂 分析结果目录: {analysis_dir}")
        
        print(f"📊 文件格式: .npz (NumPy压缩格式)")
        print(f"📊 文件内容: predictions, targets, test_ue_positions, test_bs_positions")
        print(f"🔧 分析功能: CSI误差分析、PDP分析、空间谱分析")
        print(f"{'='*60}")
        print(f"💡 提示: 此文件由test.py脚本生成并保存")
        print(f"   test.py → {results_file} → analyze.py")
        print(f"{'='*60}\n")
        
        device = None if args.device == 'auto' else args.device
        analyzer = CSIAnalyzer(
            config_path=args.config,
            results_path=args.results,
            output_dir=args.output,
            fft_size=args.fft_size,
            num_workers=args.num_workers,
            device=device,
            gpu_id=args.gpu
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
