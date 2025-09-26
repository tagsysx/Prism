#!/usr/bin/env python3
"""
Plot Script for Prism Test Results Analysis

This script provides comprehensive visualization capabilities for CSI analysis results including:

1. CSI MAE CDF plots (amplitude and phase)
2. Demo CSI samples visualization (amplitude and phase comparison)
3. Demo PAS samples visualization (spatial spectrum comparison)
4. Demo PDP samples visualization
5. PDP similarity metrics CDF plots
6. PAS similarity metrics CDF plots

Features:
- Comprehensive visualization of CSI analysis results
- High-quality plots with proper formatting
- Support for various analysis metrics
- Configurable plot parameters
"""

import os
import sys
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import json

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

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CSIPlotter:
    """
    Comprehensive CSI Analysis Plotter
    
    This class provides all plotting functionality for CSI analysis results.
    It can generate various types of plots including CDF plots, comparison plots,
    and spatial spectrum visualizations.
    """
    
    def __init__(self, analysis_dir: str, output_dir: str = None):
        """
        Initialize the CSI Plotter
        
        Args:
            analysis_dir: Path to the analysis directory containing JSON files
            output_dir: Path to save plots (defaults to analysis_dir parent/plots)
        """
        self.analysis_dir = Path(analysis_dir)
        self.output_dir = Path(output_dir) if output_dir else self.analysis_dir.parent / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'pas').mkdir(exist_ok=True)
        
        logger.info(f"📊 CSI Plotter initialized:")
        logger.info(f"   Analysis directory: {self.analysis_dir}")
        logger.info(f"   Output directory: {self.output_dir}")
    
    def plot_csi_mae_cdf(self):
        """Plot CSI distribution analysis with 4 subplots: amplitude CDF, phase CDF, amplitude MAE CDF, phase MAE CDF"""
        logger.info("📊 Plotting CSI MAE CDF...")
        
        # Load detailed CSI analysis
        csi_file = self.analysis_dir / 'detailed_csi_analysis.json'
        if not csi_file.exists():
            logger.error(f"❌ CSI analysis file not found: {csi_file}")
            return
        
        with open(csi_file, 'r') as f:
            data = json.load(f)
        
        amp_mae_values = np.array(data['error_statistics']['all_amp_mae_values'])
        phase_mae_values = np.array(data['error_statistics']['all_phase_mae_values'])
        
        # Use precomputed CDF values if available
        if 'amp_mae_cdf_values' in data['error_statistics'] and 'amp_mae_cdf_probabilities' in data['error_statistics']:
            amp_sorted = np.array(data['error_statistics']['amp_mae_cdf_values'])
            amp_cdf = np.array(data['error_statistics']['amp_mae_cdf_probabilities'])
            phase_sorted = np.array(data['error_statistics']['phase_mae_cdf_values'])
            phase_cdf = np.array(data['error_statistics']['phase_mae_cdf_probabilities'])
        else:
            # Fallback: compute CDF values
            amp_sorted = np.sort(amp_mae_values)
            amp_cdf = np.arange(1, len(amp_sorted) + 1) / len(amp_sorted)
            phase_sorted = np.sort(phase_mae_values)
            phase_cdf = np.arange(1, len(phase_sorted) + 1) / len(phase_sorted)
        
        # Load demo CSI samples for amplitude and phase CDF
        demo_file = self.analysis_dir / 'demo_csi_samples.json'
        if not demo_file.exists():
            logger.error(f"❌ Demo CSI samples file not found: {demo_file}")
            return
        
        with open(demo_file, 'r') as f:
            demo_data = json.load(f)
        
        # Extract all amplitude and phase values from demo samples
        pred_amplitudes = []
        target_amplitudes = []
        pred_phases = []
        target_phases = []
        
        for sample in demo_data['demo_samples']:
            pred_real = np.array(sample['predicted_csi_real'])
            pred_imag = np.array(sample['predicted_csi_imag'])
            target_real = np.array(sample['target_csi_real'])
            target_imag = np.array(sample['target_csi_imag'])
            
            # Compute amplitudes
            pred_amp = np.sqrt(pred_real**2 + pred_imag**2)
            target_amp = np.sqrt(target_real**2 + target_imag**2)
            
            # Compute phases
            pred_phase = np.angle(pred_real + 1j * pred_imag)
            target_phase = np.angle(target_real + 1j * target_imag)
            
            pred_amplitudes.extend(pred_amp)
            target_amplitudes.extend(target_amp)
            pred_phases.extend(pred_phase)
            target_phases.extend(target_phase)
        
        # Convert to numpy arrays
        pred_amplitudes = np.array(pred_amplitudes)
        target_amplitudes = np.array(target_amplitudes)
        pred_phases = np.array(pred_phases)
        target_phases = np.array(target_phases)
        
        # Compute CDFs for amplitudes and phases
        pred_amp_sorted = np.sort(pred_amplitudes)
        pred_amp_cdf = np.arange(1, len(pred_amp_sorted) + 1) / len(pred_amp_sorted)
        target_amp_sorted = np.sort(target_amplitudes)
        target_amp_cdf = np.arange(1, len(target_amp_sorted) + 1) / len(target_amp_sorted)
        
        pred_phase_sorted = np.sort(pred_phases)
        pred_phase_cdf = np.arange(1, len(pred_phase_sorted) + 1) / len(pred_phase_sorted)
        target_phase_sorted = np.sort(target_phases)
        target_phase_cdf = np.arange(1, len(target_phase_sorted) + 1) / len(target_phase_sorted)
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Amplitude CDF comparison
        ax1 = axes[0, 0]
        ax1.plot(pred_amp_sorted, pred_amp_cdf, 'b-', linewidth=2, label=f'Predicted (Mean: {np.mean(pred_amplitudes):.4f})')
        ax1.plot(target_amp_sorted, target_amp_cdf, 'r-', linewidth=2, label=f'Target (Mean: {np.mean(target_amplitudes):.4f})')
        ax1.set_xlabel('Amplitude', fontsize=12)
        ax1.set_ylabel('Cumulative Probability', fontsize=12)
        ax1.set_title('CSI Amplitude CDF', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Phase CDF comparison
        ax2 = axes[0, 1]
        ax2.plot(pred_phase_sorted, pred_phase_cdf, 'b-', linewidth=2, label=f'Predicted (Mean: {np.mean(pred_phases):.4f})')
        ax2.plot(target_phase_sorted, target_phase_cdf, 'r-', linewidth=2, label=f'Target (Mean: {np.mean(target_phases):.4f})')
        ax2.set_xlabel('Phase (radians)', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.set_title('CSI Phase CDF', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Amplitude MAE CDF
        ax3 = axes[1, 0]
        ax3.plot(amp_sorted, amp_cdf, 'g-', linewidth=2, label=f'Amplitude MAE (Mean: {np.mean(amp_mae_values):.4f})')
        ax3.set_xlabel('MAE Value', fontsize=12)
        ax3.set_ylabel('Cumulative Probability', fontsize=12)
        ax3.set_title('Amplitude MAE CDF', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, np.max(amp_mae_values) * 1.1)
        
        # Plot 4: Phase MAE CDF
        ax4 = axes[1, 1]
        ax4.plot(phase_sorted, phase_cdf, 'm-', linewidth=2, label=f'Phase MAE (Mean: {np.mean(phase_mae_values):.4f})')
        ax4.set_xlabel('MAE Value', fontsize=12)
        ax4.set_ylabel('Cumulative Probability', fontsize=12)
        ax4.set_title('Phase MAE CDF', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, np.max(phase_mae_values) * 1.1)
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / 'csi_mae_cdf.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ CSI MAE CDF plot saved: {output_file}")
    
    def plot_demo_csi_samples(self):
        """Plot demo CSI samples: amplitude and phase comparison - separate PNG for each sample"""
        logger.info("📊 Plotting demo CSI samples...")
        
        # Load demo CSI samples
        demo_file = self.analysis_dir / 'demo_csi_samples.json'
        if not demo_file.exists():
            logger.error(f"❌ Demo CSI samples file not found: {demo_file}")
            return
        
        with open(demo_file, 'r') as f:
            data = json.load(f)
        
        samples = data['demo_samples']
        num_samples = len(samples)
        
        # Create csi subdirectory
        csi_dir = self.output_dir / 'csi'
        csi_dir.mkdir(exist_ok=True)
        
        # Plot each sample separately
        for i, sample in enumerate(samples):
            logger.info(f"   Creating CSI sample {i+1}/{num_samples}...")
            
            # Create individual figure for each sample with amplitude and phase subplots
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # Extract sample information
            sample_idx = sample.get('sample_idx', i)
            bs_antenna_idx = sample.get('bs_antenna_idx', 'N/A')
            ue_antenna_idx = sample.get('ue_antenna_idx', 'N/A')
            
            pred_real = np.array(sample['predicted_csi_real'])
            pred_imag = np.array(sample['predicted_csi_imag'])
            target_real = np.array(sample['target_csi_real'])
            target_imag = np.array(sample['target_csi_imag'])
            
            # Plot amplitude comparison
            ax_amp = axes[0]
            pred_amp = np.sqrt(pred_real**2 + pred_imag**2)
            target_amp = np.sqrt(target_real**2 + target_imag**2)
            
            subcarriers = np.arange(len(pred_amp))
            ax_amp.plot(subcarriers, pred_amp, 'b-', linewidth=2, label='Predicted')
            ax_amp.plot(subcarriers, target_amp, 'r--', linewidth=2, label='Target')
            
            ax_amp.set_xlabel('Subcarrier Index', fontsize=12)
            ax_amp.set_ylabel('Amplitude', fontsize=12)
            ax_amp.set_title(f'CSI Amplitude Comparison\nSample {sample_idx}, BS Ant {bs_antenna_idx}, UE Ant {ue_antenna_idx}', 
                           fontsize=14, fontweight='bold')
            ax_amp.legend(fontsize=11)
            ax_amp.grid(True, alpha=0.3)
            
            # Plot phase comparison with wrapping consideration
            ax_phase = axes[1]
            pred_phase = np.angle(pred_real + 1j * pred_imag)
            target_phase = np.angle(target_real + 1j * target_imag)
            
            # Apply phase wrapping to predicted phase to minimize distance to target
            pred_phase_wrapped = np.copy(pred_phase)
            for j in range(len(pred_phase)):
                phase_diff = pred_phase[j] - target_phase[j]
                # Wrap to [-π, π] range
                phase_diff_wrapped = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
                # Choose the wrapped version if it's closer to target
                if abs(phase_diff_wrapped) < abs(phase_diff):
                    pred_phase_wrapped[j] = target_phase[j] + phase_diff_wrapped
            
            ax_phase.plot(subcarriers, pred_phase_wrapped, 'b-', linewidth=2, label='Predicted (wrapped)')
            ax_phase.plot(subcarriers, target_phase, 'r--', linewidth=2, label='Target')
            
            ax_phase.set_xlabel('Subcarrier Index', fontsize=12)
            ax_phase.set_ylabel('Phase (radians)', fontsize=12)
            ax_phase.set_title(f'CSI Phase Comparison\nSample {sample_idx}, BS Ant {bs_antenna_idx}, UE Ant {ue_antenna_idx}', 
                             fontsize=14, fontweight='bold')
            ax_phase.legend(fontsize=11)
            ax_phase.grid(True, alpha=0.3)
            
            # Plot cos(phase) comparison - COMMENTED OUT
            # ax_cos_phase = axes[2]
            # pred_cos_phase = np.cos(pred_phase)
            # target_cos_phase = np.cos(target_phase)
            # 
            # ax_cos_phase.plot(subcarriers, pred_cos_phase, 'b-', linewidth=2, label='Predicted cos(phase)')
            # ax_cos_phase.plot(subcarriers, target_cos_phase, 'r--', linewidth=2, label='Target cos(phase)')
            # 
            # ax_cos_phase.set_xlabel('Subcarrier Index', fontsize=12)
            # ax_cos_phase.set_ylabel('cos(Phase)', fontsize=12)
            # ax_cos_phase.set_title(f'CSI cos(Phase) Comparison\nSample {sample_idx}, BS Ant {bs_antenna_idx}, UE Ant {ue_antenna_idx}', 
            #                      fontsize=14, fontweight='bold')
            # ax_cos_phase.legend(fontsize=11)
            # ax_cos_phase.grid(True, alpha=0.3)
            # ax_cos_phase.set_ylim([-1.1, 1.1])  # cos(phase) range is [-1, 1]
            
            plt.tight_layout()
            
            # Save individual plot with sample information
            output_file = csi_dir / f'csi_sample_{sample_idx}_bs_{bs_antenna_idx}_ue_{ue_antenna_idx}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"   ✅ CSI sample {i+1} saved: {output_file}")
        
        logger.info(f"✅ Demo CSI samples plots saved to: {csi_dir}")
    
    def plot_demo_pas_samples(self):
        """Plot demo PAS samples: spatial spectrum comparison"""
        logger.info("📊 Plotting demo PAS samples...")
        
        # Load demo PAS samples
        demo_file = self.analysis_dir / 'demo_pas_samples.json'
        if not demo_file.exists():
            logger.error(f"❌ Demo PAS samples file not found: {demo_file}")
            return
        
        with open(demo_file, 'r') as f:
            data = json.load(f)
        
        bs_samples = data['bs_samples']
        ue_samples = data['ue_samples']
        
        # Plot BS samples
        for i, sample in enumerate(bs_samples):
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            pred_spectrum = np.array(sample['predicted_spatial_spectrum'])
            target_spectrum = np.array(sample['target_spatial_spectrum'])
            
            # Extract sample information
            sample_idx = sample.get('sample_idx', i)
            batch_idx = sample.get('batch_idx', 'N/A')
            subcarrier_idx = sample.get('subcarrier_idx', 'N/A')
            
            # Create angle arrays for 2D spatial spectrum
            azimuth_angles = np.linspace(0, 360, pred_spectrum.shape[0])  # 0-360 degrees, 2-degree steps
            elevation_angles = np.linspace(0, 90, pred_spectrum.shape[1])  # 0-90 degrees, 2-degree steps
            
            # Plot predicted spectrum
            im1 = axes[0].imshow(pred_spectrum, cmap='viridis', aspect='auto',
                               extent=[0, 360, 0, 90], origin='lower')
            axes[0].set_title(f'BS Predicted Spatial Spectrum\nSample {sample_idx}, Batch {batch_idx}, Subcarrier {subcarrier_idx}', 
                            fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Azimuth Angle (degrees)', fontsize=10)
            axes[0].set_ylabel('Elevation Angle (degrees)', fontsize=10)
            axes[0].set_xticks(np.arange(0, 361, 45))  # Every 45 degrees (0, 45, 90, 135, 180, 225, 270, 315, 360)
            axes[0].set_yticks(np.arange(0, 91, 15))   # Every 15 degrees (0, 15, 30, 45, 60, 75, 90)
            axes[0].grid(True, alpha=0.3, linestyle='--')
            plt.colorbar(im1, ax=axes[0], label='Power (dB)')
            
            # Plot target spectrum
            im2 = axes[1].imshow(target_spectrum, cmap='viridis', aspect='auto',
                               extent=[0, 360, 0, 90], origin='lower')
            axes[1].set_title(f'BS Target Spatial Spectrum\nSample {sample_idx}, Batch {batch_idx}, Subcarrier {subcarrier_idx}', 
                            fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Azimuth Angle (degrees)', fontsize=10)
            axes[1].set_ylabel('Elevation Angle (degrees)', fontsize=10)
            axes[1].set_xticks(np.arange(0, 361, 45))  # Every 45 degrees
            axes[1].set_yticks(np.arange(0, 91, 15))   # Every 15 degrees
            axes[1].grid(True, alpha=0.3, linestyle='--')
            plt.colorbar(im2, ax=axes[1], label='Power (dB)')
            
            plt.tight_layout()
            
            # Save plot with actual sample information
            output_file = self.output_dir / 'pas' / f'bs_spatial_spectrum_sample_{sample_idx}_batch_{batch_idx}_subcarrier_{subcarrier_idx}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot UE samples
        for i, sample in enumerate(ue_samples):
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            pred_spectrum = np.array(sample['predicted_spatial_spectrum'])
            target_spectrum = np.array(sample['target_spatial_spectrum'])
            
            # Extract sample information
            sample_idx = sample.get('sample_idx', i)
            batch_idx = sample.get('batch_idx', 'N/A')
            subcarrier_idx = sample.get('subcarrier_idx', 'N/A')
            
            # Create angle arrays for 2D spatial spectrum
            azimuth_angles = np.linspace(0, 360, pred_spectrum.shape[0])  # 0-360 degrees, 2-degree steps
            elevation_angles = np.linspace(0, 90, pred_spectrum.shape[1])  # 0-90 degrees, 2-degree steps
            
            # Plot predicted spectrum
            im1 = axes[0].imshow(pred_spectrum, cmap='viridis', aspect='auto',
                               extent=[0, 360, 0, 90], origin='lower')
            axes[0].set_title(f'UE Predicted Spatial Spectrum\nSample {sample_idx}, Batch {batch_idx}, Subcarrier {subcarrier_idx}', 
                            fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Azimuth Angle (degrees)', fontsize=10)
            axes[0].set_ylabel('Elevation Angle (degrees)', fontsize=10)
            axes[0].set_xticks(np.arange(0, 361, 45))  # Every 45 degrees (0, 45, 90, 135, 180, 225, 270, 315, 360)
            axes[0].set_yticks(np.arange(0, 91, 15))   # Every 15 degrees (0, 15, 30, 45, 60, 75, 90)
            axes[0].grid(True, alpha=0.3, linestyle='--')
            plt.colorbar(im1, ax=axes[0], label='Power (dB)')
            
            # Plot target spectrum
            im2 = axes[1].imshow(target_spectrum, cmap='viridis', aspect='auto',
                               extent=[0, 360, 0, 90], origin='lower')
            axes[1].set_title(f'UE Target Spatial Spectrum\nSample {sample_idx}, Batch {batch_idx}, Subcarrier {subcarrier_idx}', 
                            fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Azimuth Angle (degrees)', fontsize=10)
            axes[1].set_ylabel('Elevation Angle (degrees)', fontsize=10)
            axes[1].set_xticks(np.arange(0, 361, 45))  # Every 45 degrees
            axes[1].set_yticks(np.arange(0, 91, 15))   # Every 15 degrees
            axes[1].grid(True, alpha=0.3, linestyle='--')
            plt.colorbar(im2, ax=axes[1], label='Power (dB)')
            
            plt.tight_layout()
            
            # Save plot with actual sample information
            output_file = self.output_dir / 'pas' / f'ue_spatial_spectrum_sample_{sample_idx}_batch_{batch_idx}_subcarrier_{subcarrier_idx}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"✅ Demo PAS samples plots saved to: {self.output_dir / 'pas'}")
    
    def plot_demo_pdp_samples(self):
        """Plot demo PDP samples: PDP comparison - separate PNG for each sample"""
        logger.info("📊 Plotting demo PDP samples...")
        
        # Load demo PDP samples
        demo_file = self.analysis_dir / 'demo_pdp_samples.json'
        if not demo_file.exists():
            logger.error(f"❌ Demo PDP samples file not found: {demo_file}")
            return
        
        with open(demo_file, 'r') as f:
            data = json.load(f)
        
        samples = data['demo_pdp_samples']
        num_samples = len(samples)
        
        # Create pdp subdirectory
        pdp_dir = self.output_dir / 'pdp'
        pdp_dir.mkdir(exist_ok=True)
        
        # Plot each sample separately
        for i, sample in enumerate(samples):
            logger.info(f"   Creating PDP sample {i+1}/{num_samples}...")
            
            # Create individual figure for each sample
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            pred_pdp = np.array(sample['predicted_pdp'])
            target_pdp = np.array(sample['target_pdp'])
            
            delay_bins = np.arange(len(pred_pdp))
            
            ax.plot(delay_bins, pred_pdp, 'b-', linewidth=2, label='Predicted')
            ax.plot(delay_bins, target_pdp, 'r--', linewidth=2, label='Target')
            
            ax.set_xlabel('Delay Bin', fontsize=12)
            ax.set_ylabel('Power', fontsize=12)
            ax.set_title(f'PDP Sample {i+1}\nSample {sample["sample_idx"]}, BS Ant {sample["bs_antenna_idx"]}, UE Ant {sample["ue_antenna_idx"]}', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save individual plot with sample information
            output_file = pdp_dir / f'pdp_sample_{sample["sample_idx"]}_bs_{sample["bs_antenna_idx"]}_ue_{sample["ue_antenna_idx"]}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"   ✅ PDP sample {i+1} saved: {output_file}")
        
        logger.info(f"✅ Demo PDP samples plots saved to: {pdp_dir}")
    
    def plot_pdp_similarity_cdf(self):
        """Plot PDP similarity metrics CDF - separate subplots for each metric in one PNG"""
        logger.info("📊 Plotting PDP similarity metrics CDF...")
        
        # Load detailed PDP analysis
        pdp_file = self.analysis_dir / 'detailed_pdp_analysis.json'
        if not pdp_file.exists():
            logger.error(f"❌ PDP analysis file not found: {pdp_file}")
            return
        
        with open(pdp_file, 'r') as f:
            data = json.load(f)
        
        similarity_metrics = data['similarity_metrics']
        
        # Filter out relative_error_similarity and log_spectral_distance, add SSIM and NMSE
        colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown']
        metrics = ['cosine_similarity', 'spectral_correlation', 'bhattacharyya_coefficient', 'jensen_shannon_divergence', 'ssim', 'nmse']
        labels = ['Cosine Similarity', 'Spectral Correlation', 'Bhattacharyya Coefficient', 'Jensen-Shannon Divergence', 'SSIM', 'NMSE Similarity']
        
        # Create figure with 2x3 subplots (2 rows x 3 columns for 6 metrics)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()  # Flatten to 1D array for easier indexing
        
        # Plot each metric in separate subplot
        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            logger.info(f"   Creating {label} CDF subplot...")
            
            ax = axes[i]
            values = np.array(similarity_metrics[metric])
            sorted_values = np.sort(values)
            cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            
            # Calculate mean and 20th percentile
            mean_value = np.mean(values)
            percentile_20 = np.percentile(values, 20)
            
            ax.plot(sorted_values, cdf, color=color, linewidth=2, 
                   label=f'{label}\nMean: {mean_value:.4f}\n20th %ile: {percentile_20:.4f}')
            
            ax.set_xlabel('Similarity Value', fontsize=10)
            ax.set_ylabel('Cumulative Probability', fontsize=10)
            ax.set_title(f'{label} CDF', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # All 6 subplots are used for 6 metrics
        
        plt.tight_layout()
        
        # Save single plot with all metrics
        output_file = self.output_dir / 'pdp_similarity_cdf.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ PDP similarity CDF plot saved: {output_file}")
    
    def plot_pas_similarity_cdf(self):
        """Plot PAS similarity metrics CDF - separate subplots for each metric in one PNG"""
        logger.info("📊 Plotting PAS similarity metrics CDF...")
        
        # Load detailed PAS analysis
        pas_file = self.analysis_dir / 'detailed_pas_analysis.json'
        if not pas_file.exists():
            logger.error(f"❌ PAS analysis file not found: {pas_file}")
            return
        
        with open(pas_file, 'r') as f:
            data = json.load(f)
        
        metrics = ['cosine_similarity', 'ssim', 'nmse']
        labels = ['Cosine Similarity', 'SSIM', 'NMSE']
        colors = ['blue', 'green', 'red']
        
        # Create figure with 3x2 subplots (3 metrics x 2 antenna types)
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot each metric
        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            logger.info(f"   Creating {label} CDF subplot...")
            
            # Plot BS analysis
            if 'bs_analysis' in data and data['bs_analysis']:
                bs_metrics = data['bs_analysis']['similarity_metrics']
                ax_bs = axes[i, 0]
                
                values = np.array(bs_metrics[metric])
                sorted_values = np.sort(values)
                cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
                
                # Calculate mean and 20th percentile
                mean_value = np.mean(values)
                percentile_20 = np.percentile(values, 20)
                
                ax_bs.plot(sorted_values, cdf, color=color, linewidth=2, 
                          label=f'BS {label}\nMean: {mean_value:.4f}\n20th %ile: {percentile_20:.4f}')
                
                ax_bs.set_xlabel('Similarity Value', fontsize=12)
                ax_bs.set_ylabel('Cumulative Probability', fontsize=12)
                ax_bs.set_title(f'BS {label} CDF', fontsize=14, fontweight='bold')
                # Dynamic x-axis range: start from actual min, end at 1 (all metrics are now 0-1 similarity)
                min_val = np.min(values)
                ax_bs.set_xlim(min_val, 1.0)
                ax_bs.legend(fontsize=9)
                ax_bs.grid(True, alpha=0.3)
            
            # Plot UE analysis
            if 'ue_analysis' in data and data['ue_analysis']:
                ue_metrics = data['ue_analysis']['similarity_metrics']
                ax_ue = axes[i, 1]
                
                values = np.array(ue_metrics[metric])
                sorted_values = np.sort(values)
                cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
                
                # Calculate mean and 20th percentile
                mean_value = np.mean(values)
                percentile_20 = np.percentile(values, 20)
                
                ax_ue.plot(sorted_values, cdf, color=color, linewidth=2, 
                          label=f'UE {label}\nMean: {mean_value:.4f}\n20th %ile: {percentile_20:.4f}')
                
                ax_ue.set_xlabel('Similarity Value', fontsize=12)
                ax_ue.set_ylabel('Cumulative Probability', fontsize=12)
                ax_ue.set_title(f'UE {label} CDF', fontsize=14, fontweight='bold')
                # Dynamic x-axis range: start from actual min, end at 1 (all metrics are now 0-1 similarity)
                min_val = np.min(values)
                ax_ue.set_xlim(min_val, 1.0)
                ax_ue.legend(fontsize=9)
                ax_ue.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save single plot with all metrics
        output_file = self.output_dir / 'pas_similarity_cdf.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ PAS similarity CDF plot saved: {output_file}")
    
    def generate_all_plots(self):
        """Generate all plots"""
        logger.info("🎨 Generating all CSI analysis plots...")
        
        try:
            self.plot_csi_mae_cdf()
            self.plot_pdp_similarity_cdf()
            self.plot_pas_similarity_cdf()
            self.plot_demo_csi_samples()
            self.plot_demo_pas_samples()
            self.plot_demo_pdp_samples()

            
            logger.info("✅ All plots generated successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error generating plots: {e}")
            raise


def main():
    """Main function to run the plotting script"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate CSI analysis plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots using config file (auto-detect paths)
  python plot.py --config configs/chrissy.yml
  
  # Generate plots with explicit analysis directory
  python plot.py --analysis-dir results/chrissy/testing/analysis
  
  # Generate plots with custom output directory
  python plot.py --analysis-dir results/chrissy/testing/analysis --output-dir custom/plots
  
  # Generate plots using config file with custom output directory
  python plot.py --config configs/chrissy.yml --output-dir custom/plots
        """
    )
    
    # Add mutually exclusive group for config vs analysis-dir
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str, 
                      help='Path to configuration file (e.g., configs/chrissy.yml)')
    group.add_argument('--analysis-dir', type=str,
                      help='Path to analysis directory containing JSON files')
    
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Path to save plots (defaults to config-based or analysis-dir parent/plots)')
    
    args = parser.parse_args()
    
    # Determine analysis and output directories
    if args.config:
        # Use config file to determine paths
        logger.info(f"📋 Loading configuration from: {args.config}")
        
        config_loader = ModernConfigLoader(args.config)
        output_paths = config_loader.get_output_paths()
        
        # Get analysis directory from config
        analysis_dir = Path(output_paths['plots_dir']).parent / 'analysis'
        
        # Use provided output-dir or default from config
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = output_paths['plots_dir']
        
        logger.info(f"📂 Analysis directory: {analysis_dir}")
        logger.info(f"📁 Output directory: {output_dir}")
        
    else:
        # Use explicit analysis directory
        analysis_dir = args.analysis_dir
        
        # Use provided output-dir or default to analysis-dir parent/plots
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = str(Path(analysis_dir).parent / 'plots')
        
        logger.info(f"📂 Analysis directory: {analysis_dir}")
        logger.info(f"📁 Output directory: {output_dir}")
    
    # Create plotter and generate all plots
    plotter = CSIPlotter(analysis_dir, output_dir)
    plotter.generate_all_plots()


if __name__ == "__main__":
    main()
