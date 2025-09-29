#!/usr/bin/env python3
"""
Base Runner Class for Prism Network Training and Testing

This module provides a base class that extracts common functionality
between training and testing scripts, including:
- Configuration loading
- Data loading and preprocessing
- Network configuration updates
- Common utility methods

This reduces code duplication and ensures consistency between train.py and test.py.
"""

import os
import sys
import logging
import h5py
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism.networks.prism_network import PrismNetwork
from prism.config_loader import ModernConfigLoader
from prism.data_utils import load_and_split_by_csi, load_and_split_by_position_and_csi, create_position_aware_dataloader


class BaseRunner:
    """
    Base class for Prism network runners (training and testing).
    
    Provides common functionality for:
    - Configuration loading and validation
    - Data loading and preprocessing
    - Network configuration updates
    - Common utility methods
    """
    
    def __init__(self, config_path: str):
        """
        Initialize base runner with configuration path.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        
        # Load configuration using modern loader
        try:
            self.config_loader = ModernConfigLoader(config_path)
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.info(f"Configuration loaded from: {config_path}")
        except Exception as e:
            print(f"❌ FATAL ERROR: Failed to load configuration from {config_path}")
            print(f"   Error: {e}")
            raise
        
        # Extract key configurations
        self.device = self.config_loader.get_device()
        self.data_config = self.config_loader.get_data_loader_config()
        
        # Initialize common attributes
        self.num_subcarriers = None
        
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and prepare data based on configuration using load_and_split_by_csi.
        
        Returns:
            Tuple of (ue_positions, bs_positions, antenna_indices, csi_data)
        """
        self.logger.info("🔧 Loading data using load_and_split_by_csi...")
        
        # Get dataset configuration
        dataset_path = self.data_config['dataset_path']
        
        self.logger.info(f"📊 Dataset configuration:")
        self.logger.info(f"   Path: {dataset_path}")
        
        # Get split configuration
        train_ratio = self.data_config.get('train_ratio', 0.8)
        test_ratio = self.data_config.get('test_ratio', 0.2)
        random_seed = self.data_config.get('random_seed', 42)
        
        # Get subcarrier sampling configuration
        sampling_ratio = self.data_config.get('sampling_ratio', 1.0)
        sampling_method = self.data_config.get('sampling_method', 'uniform')
        antenna_consistent = self.data_config.get('antenna_consistent', True)
        
        self.logger.info(f"📊 Split configuration:")
        self.logger.info(f"   Train ratio: {train_ratio}")
        self.logger.info(f"   Test ratio: {test_ratio}")
        self.logger.info(f"   Random seed: {random_seed}")
        self.logger.info(f"📊 Subcarrier sampling configuration:")
        self.logger.info(f"   Sampling ratio: {sampling_ratio}")
        self.logger.info(f"   Sampling method: {sampling_method}")
        self.logger.info(f"   Antenna consistent: {antenna_consistent}")
        
        # Check if the dataset path exists
        if not os.path.exists(dataset_path):
            self.logger.error(f"❌ Dataset not found: {dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Determine mode based on class type
        if hasattr(self, '__class__') and 'Train' in self.__class__.__name__:
            mode = 'train'
        else:
            mode = 'test'
        
        self.logger.info(f"📊 Loading data in {mode} mode")
        
        # Check if position-aware loading is enabled
        use_position_aware = self.data_config.get('use_position_aware_loading', False)
        
        if use_position_aware:
            self.logger.info("🔄 Using position-aware data loading")
            # Use load_and_split_by_position_and_csi to load data
            bs_positions, ue_positions, bs_ant_indices, ue_ant_indices, csi_data, metadata = load_and_split_by_position_and_csi(
                dataset_path=dataset_path,
                train_ratio=train_ratio,
                test_ratio=test_ratio,
                random_seed=random_seed,
                mode=mode,
                sampling_ratio=sampling_ratio,
                sampling_method=sampling_method,
                antenna_consistent=antenna_consistent
            )
        else:
            self.logger.info("🔄 Using standard data loading")
            # Use load_and_split_by_csi to load data
            bs_positions, ue_positions, bs_ant_indices, ue_ant_indices, csi_data, metadata = load_and_split_by_csi(
                dataset_path=dataset_path,
                train_ratio=train_ratio,
                test_ratio=test_ratio,
                random_seed=random_seed,
                mode=mode,
                sampling_ratio=sampling_ratio,
                sampling_method=sampling_method,
                antenna_consistent=antenna_consistent
            )
        
        self.logger.info(f"📊 Loaded data shapes:")
        self.logger.info(f"   BS positions: {bs_positions.shape}")
        self.logger.info(f"   UE positions: {ue_positions.shape}")
        self.logger.info(f"   BS antenna indices: {bs_ant_indices.shape}")
        self.logger.info(f"   UE antenna indices: {ue_ant_indices.shape}")
        self.logger.info(f"   CSI data: {csi_data.shape}")
        
        # Debug: Check for NaN values in positions
        if torch.isnan(ue_positions).any():
            self.logger.error(f"❌ NaN detected in UE positions")
            self.logger.error(f"   UE positions shape: {ue_positions.shape}")
            self.logger.error(f"   UE positions min/max: {ue_positions.min().item():.6f}/{ue_positions.max().item():.6f}")
            self.logger.error(f"   NaN count: {torch.isnan(ue_positions).sum().item()}")
            raise ValueError("NaN detected in UE positions")
        
        if torch.isnan(bs_positions).any():
            self.logger.error(f"❌ NaN detected in BS positions")
            self.logger.error(f"   BS positions shape: {bs_positions.shape}")
            self.logger.error(f"   BS positions min/max: {bs_positions.min().item():.6f}/{bs_positions.max().item():.6f}")
            self.logger.error(f"   NaN count: {torch.isnan(bs_positions).sum().item()}")
            raise ValueError("NaN detected in BS positions")
        
        # Store metadata for use in subclasses
        self._loaded_metadata = metadata
        
        return ue_positions, bs_positions, bs_ant_indices, ue_ant_indices, csi_data
    
    
    def _validate_network_config_consistency(self, required_subcarriers: int):
        """Validate that network configuration matches data requirements."""
        self.logger.info(f"🔧 Validating network configuration consistency...")
        self.logger.info(f"   Required subcarriers per UE antenna from data: {required_subcarriers}")
        
        # Get expected subcarriers from configuration
        base_subcarriers = self.config_loader._processed_config.get('base_station', {}).get('ofdm', {}).get('num_subcarriers', None)
        if base_subcarriers is None:
            raise ValueError("❌ Configuration error: num_subcarriers not specified in base_station.ofdm section")
        # ue_antenna_count removed - single antenna combinations processed per sample
        expected_subcarriers_per_ue = base_subcarriers
        expected_total_subcarriers = base_subcarriers  # Use base subcarriers directly
        
        self.logger.info(f"   Expected subcarriers per UE antenna from config: {expected_subcarriers_per_ue}")
        self.logger.info(f"   Expected total subcarriers from config: {expected_total_subcarriers}")
        self.logger.info(f"   Base subcarriers: {base_subcarriers}")
        self.logger.info(f"   Processing single antenna combinations per sample")
        
        # Validate subcarriers per UE antenna (not total subcarriers)
        if required_subcarriers != expected_subcarriers_per_ue:
            raise ValueError(
                f"❌ Configuration mismatch!\n"
                f"   Data requires: {required_subcarriers} subcarriers per UE antenna\n"
                f"   Config expects: {expected_subcarriers_per_ue} subcarriers per UE antenna\n"
                f"   Total subcarriers: {required_subcarriers} (data) vs {expected_total_subcarriers} (config)\n"
                f"   Please update your configuration file to match the data requirements."
            )
        
        self.logger.info(f"✅ Network configuration validation passed")
    
    def _apply_subcarrier_sampling(
        self, 
        csi_data: torch.Tensor, 
        sampling_ratio: float, 
        sampling_method: str, 
        antenna_consistent: bool
    ) -> torch.Tensor:
        """
        Apply subcarrier sampling to CSI data.
        
        Args:
            csi_data: CSI tensor [batch, bs_antennas, ue_antennas, subcarriers]
            sampling_ratio: Ratio of subcarriers to sample (0.0 to 1.0)
            sampling_method: 'uniform' or 'random'
            antenna_consistent: Use same indices for all antennas
            
        Returns:
            Sampled CSI tensor [batch, bs_antennas, ue_antennas, sampled_subcarriers]
        """
        import numpy as np
        
        batch_size, bs_antennas, ue_antennas, original_subcarriers = csi_data.shape
        num_sampled_subcarriers = max(1, int(original_subcarriers * sampling_ratio))
        
        # Set random seed for reproducible sampling
        random_seed = self.data_config.get('random_seed', 42)
        np.random.seed(random_seed)
        
        if sampling_method == 'uniform':
            # Uniform sampling: evenly spaced subcarriers
            subcarrier_indices = np.linspace(0, original_subcarriers - 1, 
                                           num_sampled_subcarriers, dtype=int)
        elif sampling_method == 'random':
            # Random sampling: randomly selected subcarriers
            subcarrier_indices = np.sort(np.random.choice(original_subcarriers, 
                                                        num_sampled_subcarriers, 
                                                        replace=False))
        else:
            raise ValueError(f"Unsupported sampling_method: {sampling_method}. Use 'uniform' or 'random'")
        
        # Apply subcarrier sampling
        csi_data_sampled = csi_data[:, :, :, subcarrier_indices]
        
        # Update internal subcarrier count
        self.num_subcarriers = num_sampled_subcarriers
        
        self.logger.info(f"🔧 Subcarrier sampling applied:")
        self.logger.info(f"   Method: {sampling_method}")
        self.logger.info(f"   Sampling ratio: {sampling_ratio}")
        self.logger.info(f"   Original subcarriers: {original_subcarriers}")
        self.logger.info(f"   Sampled subcarriers: {num_sampled_subcarriers}")
        self.logger.info(f"   Selected indices: {subcarrier_indices[:10]}{'...' if len(subcarrier_indices) > 10 else ''}")
        self.logger.info(f"   Original CSI shape: {csi_data.shape}")
        self.logger.info(f"   Sampled CSI shape: {csi_data_sampled.shape}")
        
        return csi_data_sampled
