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
import pdb
import warnings
from scipy import ndimage
from tqdm import tqdm
# SSIM implementation moved to similarity.py

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism.config_loader import ModernConfigLoader
from similarity import Similarity


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
    
    def __init__(self, config_path: str, results_path: str = None, output_dir: str = None, 
                 device: str = None, gpu_id: int = None):
        """
        Initialize CSI analyzer
        
        Args:
            config_path: Path to configuration file
            results_path: Path to test results (.npz file) - optional, will auto-detect from config if None
            output_dir: Output directory for analysis results - optional, will auto-detect from config if None
            device: Device to use for computation ('cuda', 'cpu', or None for auto-detect)
            gpu_id: Specific GPU ID to use (e.g., 0, 1, 2). Only effective when device='cuda'
            
        Configuration Parameters (read from config file):
            - fft_size: analysis.pdp.fft_size (default: 2048 if not specified)
            - PAS settings: analysis.pas.* (count, enabled, etc.)
            - Other analysis parameters from config
            
        Parameter Sources:
            - results_path: parameter > auto-detect from config
            - output_dir: parameter > auto-detect from config  
            - device/gpu_id: from parameters only (not from config)
            - All other settings: from configuration file
        """
        self.config_path = Path(config_path)
        
        # Load configuration first
        self.config_loader = ModernConfigLoader(self.config_path)
        
        # Load analysis configuration
        analysis_config = self.config_loader._processed_config.get('analysis', {})
        
        # Get fft_size from config (with default fallback)
        self.fft_size = analysis_config.get('pdp', {}).get('fft_size', 512)
        logger.info(f"Using FFT size from config: {self.fft_size}")
        
        # Setup device for GPU computation (from parameters only)
        if device is None:
            # Auto-detect device from system or use CUDA if available
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
        
        # Load PAS configuration from already loaded analysis_config
        pas_config = analysis_config.get('pas', {})
        self.pas_enabled = pas_config.get('enabled', True)  # Default to True
        
        # Load calibration configuration
        calibration_config = self.config_loader._processed_config.get('input', {}).get('calibration', {})
        self.reference_subcarrier_index = calibration_config.get('reference_subcarrier_index', None)
        
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
        
        # Clean existing analysis files before creating new ones
        self._clean_analysis_directory()
        
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
    
    def _clean_analysis_directory(self):
        """Clean all existing files in the analysis directory before generating new ones"""
        if self.analysis_dir.exists():
            import shutil
            
            logger.info(f"🧹 Cleaning analysis directory: {self.analysis_dir}")
            
            # Count files to be deleted
            total_files = 0
            for item in self.analysis_dir.rglob('*'):
                if item.is_file():
                    total_files += 1
            
            if total_files > 0:
                logger.info(f"   Removing {total_files} existing analysis files...")
                
                # Remove all contents but keep the directory structure
                for item in self.analysis_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                
                logger.info(f"   ✅ Cleaned {total_files} files from analysis directory")
            else:
                logger.info(f"   📁 Analysis directory is already empty")
        else:
            logger.info(f"   📁 Analysis directory does not exist yet, will be created")
    
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
            # Use reference subcarrier index from config, fallback to middle subcarrier
            if self.reference_subcarrier_index is not None:
                ref_subcarrier_idx = self.reference_subcarrier_index
                logger.info(f"   Using reference subcarrier index from config: {ref_subcarrier_idx}")
            else:
                ref_subcarrier_idx = self.predictions.shape[-1] // 2
                logger.info(f"   No reference subcarrier index in config, using middle subcarrier: {ref_subcarrier_idx}")
            
            ref_pred = self.predictions[:, :, :, ref_subcarrier_idx]
            ref_target = self.targets[:, :, :, ref_subcarrier_idx]
            
            pred_ref_mean = torch.mean(torch.abs(ref_pred)).item()
            target_ref_mean = torch.mean(torch.abs(ref_target)).item()
            
            logger.info(f"   CSI Calibration Status:")
            logger.info(f"     Reference subcarrier index: {ref_subcarrier_idx}")
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
            logger.info("Starting PAS (Power Angular Spectrum) analysis...")
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
        amp_mae_cdf = Similarity.compute_empirical_cdf(amp_mae_values)
        phase_mae_cdf = Similarity.compute_empirical_cdf(phase_mae_values)
        
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
        
        while len(demo_samples) < 20 and attempts < max_attempts:
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
        
        if len(demo_samples) < 20:
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
    
    def _compute_pdp(self, csi_data, fft_size: int) -> torch.Tensor:
        """
        Power Delay Profile computation using IFFT
        
        Args:
            csi_data: CSI data (complex) - can be numpy array or torch tensor
                     Shape: (..., N) - any shape with last dimension as subcarriers
            fft_size: FFT size for PDP computation
        
        Returns:
            pdp: Power delay profile tensor
                Shape: (..., fft_size)
        """
        # Convert to torch tensor if needed
        if isinstance(csi_data, np.ndarray):
            csi_data = torch.from_numpy(csi_data).to(self.device)
        elif not isinstance(csi_data, torch.Tensor):
            csi_data = torch.tensor(csi_data, device=self.device)
        
        # Ensure it's on the correct device
        if hasattr(csi_data, 'to'):
            csi_data = csi_data.to(self.device)
        
        # Save original shape
        original_shape = csi_data.shape
        
        # Handle empty input edge case
        if csi_data.numel() == 0:
            return torch.zeros(original_shape[:-1] + (fft_size,), device=csi_data.device)
        
        # Zero-pad or truncate each CSI sequence to fft_size
        if csi_data.shape[-1] < fft_size:
            # Zero-pad each sequence
            pad_size = fft_size - csi_data.shape[-1]
            padded_csi = torch.nn.functional.pad(csi_data, (0, pad_size), mode='constant', value=0)
        else:
            # Truncate each sequence
            padded_csi = csi_data[..., :fft_size]
        
        # Compute IFFT for each CSI sequence separately (along the last dimension)
        time_domain = torch.fft.ifft(padded_csi, dim=-1)
        pdp = torch.abs(time_domain) ** 2
        
        return pdp

    
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
        
        # Convert to numpy arrays (move to CPU first if on GPU)
        pred_pdp_all = [pdp.cpu() if pdp.is_cuda else pdp for pdp in pred_pdp_all]
        target_pdp_all = [pdp.cpu() if pdp.is_cuda else pdp for pdp in target_pdp_all]
        pred_pdp_all = np.array([pdp.numpy() for pdp in pred_pdp_all])
        target_pdp_all = np.array([pdp.numpy() for pdp in target_pdp_all])
        
        logger.info(f"   PDP computed: {pred_pdp_all.shape}")
        
        # Compute PDP MAE
        pdp_mae = np.mean(np.abs(pred_pdp_all - target_pdp_all), axis=0)
        
        # Compute all PDP (Power Delay Profile) similarity metrics in a single loop
        logger.info("   Computing PDP Similarity metrics...")
        cosine_similarity_values = []
        relative_error_similarity_values = []
        spectral_correlation_values = []
        log_spectral_distance_values = []
        bhattacharyya_coefficient_values = []
        jensen_shannon_divergence_values = []
        ssim_values = []
        nmse_values = []
        
        for i in tqdm(range(len(pred_pdp_all)), desc="Computing PDP Similarity", unit="samples"):
            pred_pdp = pred_pdp_all[i]
            target_pdp = target_pdp_all[i]
            
            # Cosine Similarity using traditional cosine similarity method
            pred_tensor = torch.from_numpy(pred_pdp).float()
            target_tensor = torch.from_numpy(target_pdp).float()
            cosine_similarity = Similarity.compute_cosine_similarity(pred_tensor, target_tensor)
            cosine_similarity_values.append(cosine_similarity)
            
            # Relative Error Similarity
            relative_error = float(np.mean(np.abs(pred_pdp - target_pdp) / (np.abs(target_pdp) + 1e-12)))
            # Map to [0, 1], smaller error = higher similarity
            # Use exponential decay to map relative error to similarity
            relative_error_similarity = float(np.exp(-relative_error))
            relative_error_similarity_values.append(relative_error_similarity)
            
            # Spectral Correlation Coefficient
            scc = Similarity.compute_spectral_correlation_coefficient(pred_pdp, target_pdp)
            spectral_correlation_values.append(scc)
            
            # Log Spectral Distance
            lsd = Similarity.compute_log_spectral_distance(pred_pdp, target_pdp)
            log_spectral_distance_values.append(lsd)
            
            # Bhattacharyya Coefficient
            bc = Similarity.compute_bhattacharyya_coefficient(pred_pdp, target_pdp)
            bhattacharyya_coefficient_values.append(bc)
            
            # Jensen-Shannon Divergence
            jsd = Similarity.compute_jensen_shannon_divergence(pred_pdp, target_pdp)
            jensen_shannon_divergence_values.append(jsd)
            
            # 1D SSIM for PDP
            pred_pdp_tensor = torch.from_numpy(pred_pdp).float().to(self.device)
            target_pdp_tensor = torch.from_numpy(target_pdp).float().to(self.device)
            ssim = Similarity.compute_ssim_1d(pred_pdp_tensor, target_pdp_tensor)
            ssim_values.append(ssim)
            
            # Normalized Mean Squared Error (NMSE) for PDP using standard NMSE similarity method
            pred_tensor = torch.from_numpy(pred_pdp).float()
            target_tensor = torch.from_numpy(target_pdp).float()
            nmse_similarity = Similarity.compute_nmse_similarity(pred_tensor, target_tensor)
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
                'pdp_shape': list(pred_pdp_all.shape),  # Convert shape to list
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
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(detailed_pdp_file, 'w') as f:
            json.dump(detailed_pdp_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"✅ Detailed PDP analysis saved: {detailed_pdp_file}")
        logger.info(f"   Saved {len(pred_pdp_all)} PDP similarity values for each metric")
        
        # Save 20 random PDP samples for demo
        logger.info("   Saving 20 random PDP samples for demo...")
        np.random.seed(42)  # For reproducibility
        
        # Ensure we don't try to select more samples than available
        num_demo_samples = min(20, len(pred_pdp_all))
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
                'pdp_shape': list(pred_pdp_all.shape),  # Convert shape to list
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
        bs_config = self.config_loader._processed_config['base_station']
        ue_config = self.config_loader._processed_config['user_equipment']
        ofdm_config = self.config_loader._processed_config['base_station']['ofdm']
        analysis_config = self.config_loader._processed_config.get('analysis', {}).get('pas', {})
        
        bs_antenna_count = self.config_loader.num_bs_antennas
        # ue_antenna_count removed - single antenna combinations processed per sample
        
        # Log detailed antenna structure information
        logger.info("📡 Antenna Structure Information:")
        logger.info(f"   BS antennas: {bs_antenna_count}, processing single antenna combinations")
        logger.info(f"   BS array config: {bs_config['antenna_array']['configuration']}")
        logger.info(f"   UE array config: {ue_config['antenna_array']['configuration']}")
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
        
        # Analyze spatial spectrum from both BS and UE perspectives if both have multiple antennas
        if num_bs_antennas > 1 or num_ue_antennas > 1:
            logger.info("   Computing dual-perspective spatial spectra...")
            dual_results = self._compute_spatial_spectrum_analysis(
                self.predictions, self.targets, 
                analysis_config=analysis_config,
                bs_config=bs_config,
                ue_config=ue_config
            )
            spatial_spectrum_stats['dual_perspective_analysis'] = dual_results
            logger.info(f"   Dual-perspective Analysis completed: {dual_results['total_samples']} samples")
            logger.info(f"     BS samples: {dual_results['bs_samples']}, UE samples: {dual_results['ue_samples']}")
        else:
            logger.warning("   Skipping spatial spectrum analysis: both BS and UE need multiple antennas")
        
        # Store results for summary
        self.spatial_spectrum_stats = spatial_spectrum_stats
        
        # Create unified PAS analysis file
        self._create_unified_pas_analysis()
        
        logger.info("✅ Spatial spectrum analysis completed!")
    
    def _create_unified_pas_analysis(self):
        """Create unified PAS analysis file from dual-perspective data"""
        logger.info("📊 Creating unified PAS analysis file...")
        
        # Check if dual-perspective analysis exists
        if 'dual_perspective_analysis' not in self.spatial_spectrum_stats:
            logger.warning("No dual-perspective analysis found, creating empty unified PAS analysis")
            unified_data = {
                'analysis_info': {
                    'timestamp': datetime.now().isoformat(),
                    'total_bs_samples': 0,
                    'total_ue_samples': 0
                },
                'bs_analysis': {},
                'ue_analysis': {}
            }
        else:
            dual_data = self.spatial_spectrum_stats['dual_perspective_analysis']
            
            # Create unified format compatible with plot script
            unified_data = {
                'analysis_info': {
                    'timestamp': datetime.now().isoformat(),
                    'total_bs_samples': dual_data.get('bs_samples', 0),
                    'total_ue_samples': dual_data.get('ue_samples', 0)
                },
                'bs_analysis': {},
                'ue_analysis': {}
            }
            
            # Load similarity metrics from the detailed file if it exists
            similarity_file = self.analysis_dir / 'detailed_pas_analysis.json'
            if similarity_file.exists():
                try:
                    with open(similarity_file, 'r') as f:
                        existing_data = json.load(f)
                        if 'similarity_metrics' in existing_data:
                            # Use the same similarity metrics for both BS and UE sections for compatibility
                            similarity_metrics = existing_data['similarity_metrics']
                            
                            if dual_data.get('bs_samples', 0) > 0:
                                unified_data['bs_analysis'] = {
                                    'total_samples': dual_data.get('bs_samples', 0),
                                    'similarity_metrics': similarity_metrics
                                }
                            
                            if dual_data.get('ue_samples', 0) > 0:
                                unified_data['ue_analysis'] = {
                                    'total_samples': dual_data.get('ue_samples', 0),
                                    'similarity_metrics': similarity_metrics
                                }
                except Exception as e:
                    logger.warning(f"Could not load existing similarity data: {e}")
        
        # Save unified PAS analysis
        unified_file = self.analysis_dir / 'detailed_pas_analysis.json'
        with open(unified_file, 'w') as f:
            json.dump(unified_data, f, indent=2)
        
        logger.info(f"✅ Unified PAS analysis saved: {unified_file}")
        logger.info(f"   Total BS samples: {unified_data['analysis_info']['total_bs_samples']}")
        logger.info(f"   Total UE samples: {unified_data['analysis_info']['total_ue_samples']}")
        
        # Demo samples are already created in _compute_spatial_spectrum_analysis
    
    def _compute_spatial_spectrum_analysis(self, predictions: torch.Tensor, targets: torch.Tensor,
                                         analysis_config: dict, bs_config: dict, ue_config: dict) -> dict:
        """
        Compute dual-perspective spatial spectrum analysis using BS and UE configurations
        
        Args:
            predictions: Predicted CSI tensor [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
            targets: Target CSI tensor [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
            analysis_config: Analysis configuration dictionary containing:
                           - 'ignore_amplitude': Whether to ignore amplitude in spatial spectrum (bool)
                           - 'count': Maximum number of samples to analyze (int)
            bs_config: Base station configuration dictionary containing:
                      - 'array_configuration': Array configuration string (e.g., '4x1', '8x8')
                      - 'ofdm': OFDM configuration with center_frequency, bandwidth, subcarrier_spacing
                      - 'num_antennas': Number of BS antennas
            ue_config: User equipment configuration dictionary containing:
                      - 'array_configuration': Array configuration string (e.g., '2x1', '4x4')
                      - 'num_ue_antennas': Number of UE antennas
        
        Returns:
            dict: Analysis results containing:
                - 'bs_array_shape': BS array shape tuple
                - 'ue_array_shape': UE array shape tuple  
                - 'total_samples': Total number of samples processed
                - 'bs_samples': Number of BS perspective samples
                - 'ue_samples': Number of UE perspective samples
                - 'similarity_file': Path to detailed similarity analysis file
                - 'similarity_metrics': Statistical summary of similarity metrics
                - 'skipped_reason': Reason for skipping analysis (if applicable)
        
        Note:
            This function performs dual-perspective spatial spectrum analysis using MIMO CSI data.
            It computes Power Angular Spectrum (PAS) from both BS and UE perspectives and calculates
            similarity metrics including cosine similarity, NMSE, and SSIM.
            All parameters (num_antennas, num_subcarriers, OFDM config) are extracted from bs_config and ue_config.
        """
        
        # Extract parameters from configurations
        if not bs_config or not ue_config:
            raise ValueError("Both bs_config and ue_config are required for dual-perspective spatial spectrum analysis")
        
        # Get antenna counts from configurations
        num_bs_antennas = bs_config.get('num_antennas')
        num_ue_antennas = ue_config.get('num_ue_antennas')
        
        if num_bs_antennas is None:
            raise ValueError("bs_config must contain 'num_antennas'")
        if num_ue_antennas is None:
            raise ValueError("ue_config must contain 'num_ue_antennas'")
        
        # Check if we have enough antennas for spatial spectrum analysis
        if num_bs_antennas <= 1 and num_ue_antennas <= 1:
            logger.warning(f"⚠️  Skipping dual-perspective spatial spectrum analysis: BS has {num_bs_antennas} antenna(s), UE has {num_ue_antennas} antenna(s). Spatial spectrum requires antenna arrays (>1 antenna).")
            return {
                'bs_array_shape': (num_bs_antennas,),
                'ue_array_shape': (num_ue_antennas,),
                'total_samples': 0,
                'similarity_file': None,
                'skipped_reason': f'Insufficient antennas: BS={num_bs_antennas}, UE={num_ue_antennas} (need >1 for spatial spectrum)'
            }
        
        # Get OFDM parameters from bs_config
        ofdm_config = bs_config.get('ofdm', {})
        if not ofdm_config:
            raise ValueError("bs_config must contain 'ofdm' configuration")
        
        # Get number of subcarriers from OFDM configuration
        num_subcarriers = ofdm_config.get('num_subcarriers')
        if num_subcarriers is None:
            raise ValueError("ofdm_config must contain 'num_subcarriers'")
        
        # Validate that data shape matches configuration
        data_subcarriers = predictions.shape[-1]
        if data_subcarriers != num_subcarriers:
            raise ValueError(f"Data subcarrier count mismatch: data has {data_subcarriers} subcarriers but OFDM config specifies {num_subcarriers}. Please check your configuration or data.")
        
        # Parse array configuration from bs_config (dual-perspective analysis always uses both BS and UE configs)
        from prism.utils.pas_utils import parse_array_configuration
        
        # Get OFDM parameters from bs_config
        center_freq = float(ofdm_config['center_frequency'])
        bandwidth = float(ofdm_config['bandwidth'])
        subcarrier_spacing = float(ofdm_config['subcarrier_spacing'])
        subcarrier_indices = torch.arange(num_subcarriers, device=self.device)
        subcarrier_freqs = center_freq + (subcarrier_indices - num_subcarriers//2) * subcarrier_spacing
        wavelengths = 3e8 / subcarrier_freqs  # [num_subcarriers]
        
        # Get analysis parameters
        ignore_amplitude = analysis_config['ignore_amplitude']
        max_samples = analysis_config['count']
        
        logger.info(f"   Center freq: {center_freq/1e9:.2f} GHz, Bandwidth: {bandwidth/1e6:.1f} MHz")
        
        # Get number of position pairs to process
        num_positions = predictions.shape[0]
        
        # Apply sampling limit to position pairs
        position_indices = list(range(num_positions))
        if len(position_indices) > max_samples:
            np.random.seed(42)  # For reproducible results
            selected_positions = np.random.choice(len(position_indices), size=max_samples, replace=False)
            position_indices = [position_indices[i] for i in selected_positions]
            logger.info(f"   Randomly selected {max_samples} position pairs from {num_positions} total position pairs")
        
        # Process samples with GPU batch computation
        logger.info(f"   🚀 Starting GPU batch processing of PAS samples...")
        
        # Prepare batch data - directly use 4D MIMO format since data is already organized by position pairs
        pred_pas_list = []
        target_pas_list = []
        
        # Parse array configurations (already validated above)
        
        from prism.utils.pas_utils import parse_array_configuration
        bs_array_shape = parse_array_configuration(bs_config['antenna_array'].get('configuration', '4x1'))
        ue_array_shape = parse_array_configuration(ue_config['antenna_array'].get('configuration', '2x1'))
        
        logger.info(f"   Processing {len(position_indices)} position pairs directly from 4D MIMO format...")
        
        # Process each selected position pair directly - data is already in MIMO format
        for pos_idx in position_indices:
            # Extract MIMO CSI for this position pair
            pred_csi_mimo = predictions[pos_idx, :, :, :]  # [num_bs_antennas, num_ue_antennas, num_subcarriers]
            target_csi_mimo = targets[pos_idx, :, :, :]
            
            # Skip if target CSI is zero (no ground truth)
            if torch.all(torch.abs(target_csi_mimo) < 1e-10):
                continue
            
            # Get position data - must be available for spatial spectrum analysis
            if self.bs_positions is None:
                raise ValueError("BS positions are required for spatial spectrum analysis but not found in test data")
            if self.ue_positions is None:
                raise ValueError("UE positions are required for spatial spectrum analysis but not found in test data")
            if pos_idx >= len(self.bs_positions):
                raise ValueError(f"Position index {pos_idx} exceeds BS positions length {len(self.bs_positions)}")
            if pos_idx >= len(self.ue_positions):
                raise ValueError(f"Position index {pos_idx} exceeds UE positions length {len(self.ue_positions)}")
                
            bs_pos = self.bs_positions[pos_idx]
            ue_pos = self.ue_positions[pos_idx]
            
            # Directly call mimo_to_pas for PAS computation
            from prism.utils.pas_utils import mimo_to_pas
            
            # Get azimuth_only setting from config
            azimuth_only = analysis_config.get('azimuth_only', False)
            
            pred_pas_dict = mimo_to_pas(
                csi_matrix=pred_csi_mimo,
                bs_array_shape=bs_array_shape,
                ue_array_shape=ue_array_shape,
                azimuth_divisions=361,  # High resolution for analysis
                elevation_divisions=91,
                normalize_pas=True,
                center_freq=center_freq,
                subcarrier_spacing=subcarrier_spacing,
                azimuth_only=azimuth_only
            )
            target_pas_dict = mimo_to_pas(
                csi_matrix=target_csi_mimo,
                bs_array_shape=bs_array_shape,
                ue_array_shape=ue_array_shape,
                azimuth_divisions=361,  # High resolution for analysis
                elevation_divisions=91,
                normalize_pas=True,
                center_freq=center_freq,
                subcarrier_spacing=subcarrier_spacing,
                azimuth_only=azimuth_only
            )
            
            # Save all PAS data without averaging - preserve full information
            pred_pas_list.append({
                'pos_idx': pos_idx,  # Position pair index
                'bs': pred_pas_dict["bs"],  # [num_ue_antennas, azimuth_divisions, elevation_divisions]
                'ue': pred_pas_dict["ue"]   # [num_bs_antennas, azimuth_divisions, elevation_divisions]
            })
            target_pas_list.append({
                'pos_idx': pos_idx,  # Position pair index
                'bs': target_pas_dict["bs"],  # [num_ue_antennas, azimuth_divisions, elevation_divisions]
                'ue': target_pas_dict["ue"]   # [num_bs_antennas, azimuth_divisions, elevation_divisions]
            })
        
        logger.info(f"   Processing {len(pred_pas_list)} valid samples directly (no chunking)...")
        
        # Compute PAS (Power Angular Spectrum) similarity metrics for all samples
        logger.info(f"   Computing PAS Similarity metrics for {len(pred_pas_list)} samples...")
        
        results = []
        for i in tqdm(range(len(pred_pas_list)), desc="Computing PAS Similarity", unit="samples"):
            pred_pas_data = pred_pas_list[i]
            target_pas_data = target_pas_list[i]
            
            # Process BS perspective PAS (skip if BS has only 1 antenna)
            bs_results = []
            if num_bs_antennas > 1:
                pred_bs_pas = pred_pas_data['bs']  # [num_ue_antennas, azimuth_divisions, elevation_divisions]
                target_bs_pas = target_pas_data['bs']
                
                for ue_ant_idx in range(pred_bs_pas.shape[0]):
                    pred_spectrum = pred_bs_pas[ue_ant_idx]  # [azimuth_divisions, elevation_divisions]
                    target_spectrum = target_bs_pas[ue_ant_idx]
                    similarity_metrics = self._compute_spatial_spectrum_similarity(pred_spectrum, target_spectrum)
                    
                    bs_results.append({
                        'ue_antenna_idx': ue_ant_idx,
                        'predicted_spectrum': pred_spectrum.cpu().numpy().tolist(),
                        'target_spectrum': target_spectrum.cpu().numpy().tolist(),
                        'similarity_metrics': similarity_metrics
                    })
            
            # Process UE perspective PAS (skip if UE has only 1 antenna)
            ue_results = []
            if num_ue_antennas > 1:
                pred_ue_pas = pred_pas_data['ue']  # [num_bs_antennas, azimuth_divisions, elevation_divisions]
                target_ue_pas = target_pas_data['ue']
                
                for bs_ant_idx in range(pred_ue_pas.shape[0]):
                    pred_spectrum = pred_ue_pas[bs_ant_idx]  # [azimuth_divisions, elevation_divisions]
                    target_spectrum = target_ue_pas[bs_ant_idx]
                    similarity_metrics = self._compute_spatial_spectrum_similarity(pred_spectrum, target_spectrum)
                    
                    ue_results.append({
                        'bs_antenna_idx': bs_ant_idx,
                'predicted_spectrum': pred_spectrum.cpu().numpy().tolist(),
                'target_spectrum': target_spectrum.cpu().numpy().tolist(),
                        'similarity_metrics': similarity_metrics
                    })
            
            results.append({
                'sample_idx': i,
                'pos_idx': pred_pas_data['pos_idx'],  # Original position pair index
                'bs_perspective': bs_results,  # All UE antennas from BS perspective
                'ue_perspective': ue_results,  # All BS antennas from UE perspective
                'center_frequency': float(center_freq),
                'wavelength_range': [float(wavelengths.min()), float(wavelengths.max())],
                'num_subcarriers': num_subcarriers,
                'note': 'Full PAS data preserved - no averaging applied'
            })
        
        logger.info(f"   Completed processing of {len(results)} spatial spectra")
        
        # Extract similarity values for separate storage - collect from all antennas
        cosine_similarity_values = []
        nmse_values = []
        ssim_values = []
        
        for sample in results:
            # Collect from BS perspective (all UE antennas) - only if BS has multiple antennas
            if sample['bs_perspective']:  # Will be empty if BS has only 1 antenna
                for bs_result in sample['bs_perspective']:
                    similarity_metrics = bs_result['similarity_metrics']
            cosine_similarity_values.append(similarity_metrics['cosine_similarity'])
            nmse_values.append(similarity_metrics['nmse'])
            ssim_values.append(similarity_metrics['ssim'])
        
            # Collect from UE perspective (all BS antennas) - only if UE has multiple antennas
            if sample['ue_perspective']:  # Will be empty if UE has only 1 antenna
                for ue_result in sample['ue_perspective']:
                    similarity_metrics = ue_result['similarity_metrics']
                    cosine_similarity_values.append(similarity_metrics['cosine_similarity'])
                    nmse_values.append(similarity_metrics['nmse'])
                    ssim_values.append(similarity_metrics['ssim'])
        
        # Randomly select samples and convert to BS/UE separated format
        logger.info(f"   Selecting random samples for demo...")
        np.random.seed(42)  # For reproducibility
        
        # Collect all BS and UE perspective data
        all_bs_data = []
        all_ue_data = []
        
        for sample in results:
            sample_idx = sample['sample_idx']
            pos_idx = sample['pos_idx']
            
            # Collect BS perspective data (only if BS has multiple antennas)
            if num_bs_antennas > 1:
                for bs_data in sample['bs_perspective']:
                    all_bs_data.append({
                        'sample_idx': sample_idx,
                        'pos_idx': pos_idx,
                        'ue_antenna_idx': bs_data.get('ue_antenna_idx', 0),
                        'predicted_spatial_spectrum': bs_data['predicted_spectrum'],
                        'target_spatial_spectrum': bs_data['target_spectrum'],
                        'similarity_metrics': bs_data['similarity_metrics']
                    })
            
            # Collect UE perspective data (only if UE has multiple antennas)
            if num_ue_antennas > 1:
                for ue_data in sample['ue_perspective']:
                    all_ue_data.append({
                        'sample_idx': sample_idx,
                        'pos_idx': pos_idx,
                        'bs_antenna_idx': ue_data.get('bs_antenna_idx', 0),
                        'predicted_spatial_spectrum': ue_data['predicted_spectrum'],
                        'target_spatial_spectrum': ue_data['target_spectrum'],
                        'similarity_metrics': ue_data['similarity_metrics']
                    })
        
        # Randomly select up to 20 samples from each perspective
        import random
        selected_bs = []
        selected_ue = []
        
        if all_bs_data and num_bs_antennas > 1:
            selected_bs = random.sample(all_bs_data, min(20, len(all_bs_data)))
            logger.info(f"   Selected {len(selected_bs)} BS perspective samples from {len(all_bs_data)} available")
        
        if all_ue_data and num_ue_antennas > 1:
            selected_ue = random.sample(all_ue_data, min(20, len(all_ue_data)))
            logger.info(f"   Selected {len(selected_ue)} UE perspective samples from {len(all_ue_data)} available")
        
        # Save demo samples in BS/UE separated format
        demo_file = self.analysis_dir / 'demo_pas_samples.json'
        with open(demo_file, 'w') as f:
            json.dump({
                'demo_info': {
                    'timestamp': datetime.now().isoformat(),
                    'bs_samples': len(selected_bs),
                    'ue_samples': len(selected_ue),
                    'note': f'BS antennas: {num_bs_antennas}, UE antennas: {num_ue_antennas}'
                },
                'bs_samples': selected_bs,
                'ue_samples': selected_ue
            }, f, indent=2)
        
        logger.info(f"✅ Demo PAS samples saved: {demo_file}")
        logger.info(f"   BS samples: {len(selected_bs)}, UE samples: {len(selected_ue)}")
        
        # Save detailed similarity analysis to separate file
        similarity_file = self.analysis_dir / 'detailed_pas_analysis.json'
        with open(similarity_file, 'w') as f:
            json.dump({
                'bs_array_shape': bs_array_shape,
                'ue_array_shape': ue_array_shape,
                'total_samples': len(results),
                'total_bs_antenna_samples': sum(len(s['bs_perspective']) for s in results),
                'total_ue_antenna_samples': sum(len(s['ue_perspective']) for s in results),
                'center_frequency': float(center_freq),
                'wavelength_range': [float(wavelengths.min()), float(wavelengths.max())],
                'num_subcarriers': num_subcarriers,
                'timestamp': datetime.now().isoformat(),
                'note': f'BS antennas: {num_bs_antennas}, UE antennas: {num_ue_antennas}. Skipped perspectives with single antennas.',
                'similarity_metrics': {
                    'cosine_similarity': cosine_similarity_values,
                    'nmse': nmse_values,
                    'ssim': ssim_values
                } if cosine_similarity_values else {
                    'cosine_similarity': [],
                    'nmse': [],
                    'ssim': [],
                    'note': 'No similarity values computed - both BS and UE have single antennas'
                }
            }, f, indent=2)
        
        logger.info(f"✅ Detailed dual-perspective spatial spectrum similarity values saved: {similarity_file}")
        
        # Note: Detailed spatial spectrum data is no longer saved separately
        # All similarity metrics are consolidated in detailed_pas_analysis.json
        
        # Calculate statistics only if we have similarity values
        similarity_stats = {}
        if cosine_similarity_values:
            similarity_stats = {
                'cosine_similarity': {
                    'mean': float(np.mean(cosine_similarity_values)),
                    'std': float(np.std(cosine_similarity_values)),
                    'min': float(np.min(cosine_similarity_values)),
                    'max': float(np.max(cosine_similarity_values))
                },
                'nmse': {
                    'mean': float(np.mean(nmse_values)),
                    'std': float(np.std(nmse_values)),
                    'min': float(np.min(nmse_values)),
                    'max': float(np.max(nmse_values))
                },
                'ssim': {
                    'mean': float(np.mean(ssim_values)),
                    'std': float(np.std(ssim_values)),
                    'min': float(np.min(ssim_values)),
                    'max': float(np.max(ssim_values))
                }
            }
        else:
            logger.warning("   No similarity values computed (both BS and UE have single antennas)")
            similarity_stats = {
                'cosine_similarity': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                'nmse': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                'ssim': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            }
        
        return {
            'bs_array_shape': bs_array_shape,
            'ue_array_shape': ue_array_shape,
            'total_samples': len(results),
            'bs_samples': sum(len(s['bs_perspective']) for s in results),
            'ue_samples': sum(len(s['ue_perspective']) for s in results),
            'similarity_file': str(similarity_file),
            'similarity_metrics': similarity_stats
        }
    
    

    def _compute_spatial_spectrum_similarity(self, pred_spectrum: torch.Tensor, 
                                           target_spectrum: torch.Tensor) -> dict:
        """Compute similarity metrics between predicted and target spatial spectra on GPU"""
        
        # Compute various similarity metrics
        # 1. Cosine similarity using traditional cosine similarity method (with internal Min-Max normalization)
        cosine_sim = Similarity.compute_cosine_similarity(pred_spectrum, target_spectrum)
        
        # 2. Normalized Mean Squared Error (NMSE) using standard NMSE similarity method (no internal normalization)
        nmse = Similarity.compute_nmse_similarity(pred_spectrum, target_spectrum)
        
        # 3. Structural Similarity Index (SSIM) - always use 2D version for spatial spectra
        if pred_spectrum.dim() == 2 and target_spectrum.dim() == 2:
            # Use 2D SSIM for 2D spatial spectra (azimuth x elevation) (with internal energy normalization)
            ssim = Similarity.compute_ssim_2d(pred_spectrum, target_spectrum)
        elif pred_spectrum.dim() == 1 and target_spectrum.dim() == 1:
            # Use 1D SSIM for 1D spatial spectra (with internal normalization)
            ssim = Similarity.compute_ssim_1d(pred_spectrum, target_spectrum)
        else:
            # Unsupported dimension combination
            raise ValueError(f"Unsupported spectrum dimensions: pred={pred_spectrum.dim()}D, target={target_spectrum.dim()}D. Only 1D and 2D spectra are supported.")
        
        return {
            'cosine_similarity': float(cosine_sim),
            'nmse': float(nmse),
            'ssim': float(ssim)
        }
    
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
            print(f"\n📡 PDP Statistics:")
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
    parser.add_argument('--fft-size', type=int, default=2048, help='DEPRECATED: FFT size now read from config analysis.pdp.fft_size')
    parser.add_argument('--num-workers', type=int, help='DEPRECATED: Number of parallel workers (no longer used)')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='auto', help='Device to use for computation (default: auto)')
    parser.add_argument('--gpu', type=int, default=None, help='Specific GPU ID to use (e.g., 0, 1, 2). Only effective when --device cuda')
    
    args = parser.parse_args()
    
    # Show deprecation warnings for unused parameters
    if args.fft_size != 2048:
        print("⚠️  WARNING: --fft-size parameter is deprecated. FFT size is now read from config file (analysis.pdp.fft_size)")
    if args.num_workers is not None:
        print("⚠️  WARNING: --num-workers parameter is deprecated and no longer used")
    
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
