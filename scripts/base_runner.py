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
        Load and prepare data based on configuration.
        
        Returns:
            Tuple of (ue_positions, bs_positions, antenna_indices, csi_data)
        """
        self.logger.info("🔧 Loading data...")
        
        # Get dataset configuration
        dataset_path = self.data_config['dataset_path']
        
        self.logger.info(f"📊 Dataset configuration:")
        self.logger.info(f"   Path: {dataset_path}")
        
        # Get subcarrier sampling configuration
        sampling_ratio = self.data_config.get('sampling_ratio', 1.0)
        sampling_method = self.data_config.get('sampling_method', 'uniform')
        antenna_consistent = self.data_config.get('antenna_consistent', True)
        
        self.logger.info(f"📊 Subcarrier sampling configuration:")
        self.logger.info(f"   Sampling ratio: {sampling_ratio}")
        self.logger.info(f"   Sampling method: {sampling_method}")
        self.logger.info(f"   Antenna consistent: {antenna_consistent}")
        
        # Check if the dataset path exists
        if not os.path.exists(dataset_path):
            self.logger.error(f"❌ Dataset not found: {dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Auto-detect format and load data
        with h5py.File(dataset_path, 'r') as f:
            keys = list(f.keys())
            self.logger.info(f"📊 Dataset keys: {keys}")
            
            if 'data' not in keys:
                raise ValueError(f"Invalid dataset structure. Available keys: {keys}")
            
            data_keys = list(f['data'].keys())
            self.logger.info(f"📊 Data keys: {data_keys}")
            
            # Auto-detect data format based on available keys
            if 'ue_positions' in data_keys and 'bs_positions' in data_keys and 'csi' in data_keys:
                # All datasets now use the same key names - detect format by data dimensions
                self.logger.info("🔍 Detected unified format - analyzing data dimensions")
                ue_positions = torch.from_numpy(f['data/ue_positions'][:]).float()
                bs_positions_raw = torch.from_numpy(f['data/bs_positions'][:]).float()
                channel_responses = torch.from_numpy(f['data/csi'][:]).cfloat()
                
                self.logger.info(f"📊 Raw channel_responses shape: {channel_responses.shape}")
                self.logger.info(f"   Shape[1] (dim1): {channel_responses.shape[1]}")
                self.logger.info(f"   Shape[2] (dim2): {channel_responses.shape[2]}")
                self.logger.info(f"   Shape[3] (dim3): {channel_responses.shape[3]}")
                
                # CSI data format is fixed: [batch_size, num_bs_antennas, num_ue_antennas, subcarriers]
                # Validate that data matches expected format
                dim1, dim2, dim3 = channel_responses.shape[1], channel_responses.shape[2], channel_responses.shape[3]
                
                # Get expected subcarrier count from configuration
                expected_subcarriers = self.config_loader._processed_config.get('base_station', {}).get('ofdm', {}).get('num_subcarriers', None)
                if expected_subcarriers is None:
                    raise ValueError("❌ Configuration error: num_subcarriers not specified in base_station.ofdm section")
                
                # Validate that subcarriers are in the last dimension (dim3)
                if dim3 != expected_subcarriers:
                    raise ValueError(
                        f"❌ Data format error: Expected subcarriers ({expected_subcarriers}) in last dimension, "
                        f"but found {dim3}. Data shape: {channel_responses.shape}. "
                        f"Expected format: [samples, bs_antennas, ue_antennas, subcarriers]"
                    )
                
                self.logger.info(f"   Validated subcarriers: {dim3} (matches config: {expected_subcarriers})")
                
                # Data should already be in correct format: [samples, bs_antennas, ue_antennas, subcarriers]
                # No permutation needed - just verify the format
                csi_data = channel_responses.permute(0, 1, 2, 3)  # Identity permutation for clarity
                self.logger.info("   Format: [samples, bs_antennas, ue_antennas, subcarriers]")
                
            else:
                raise ValueError(f"Unknown dataset format. Available keys: {data_keys}")
            
            self.logger.info(f"📊 Raw data shapes:")
            self.logger.info(f"   UE positions: {ue_positions.shape}")
            self.logger.info(f"   BS positions (raw): {bs_positions_raw.shape}")
            self.logger.info(f"   CSI data: {csi_data.shape}")
            
            # Handle BS positions using the generic expansion method
            num_samples = ue_positions.shape[0]
            bs_positions = self._expand_bs_positions_to_3d(bs_positions_raw, num_samples)
            
            # Always use BS antenna array (consistent with training)
            self.logger.info("🔧 Using BS antenna array")
            num_bs_antennas = csi_data.shape[1]
            antenna_indices = torch.arange(num_bs_antennas).unsqueeze(0).expand(num_samples, -1).long()
            
            # Handle UE antenna configuration
            ue_antenna_count = self.data_config.get('ue_antenna_count', 1)
            batch_size, bs_antennas, ue_antennas, subcarriers = csi_data.shape
            
            if ue_antenna_count == 1:
                # Use single UE antenna (antenna 0) - keep 4D format for CSI network
                csi_data = csi_data[:, :, 0:1, :]  # Keep UE dimension but use only antenna 0
                self.logger.info(f"🔧 Using single UE antenna (index 0) - keeping 4D format")
            elif ue_antenna_count > 1:
                # Use multiple UE antennas - keep them separate
                if ue_antenna_count > ue_antennas:
                    raise ValueError(f"Requested {ue_antenna_count} UE antennas but only {ue_antennas} available")
                
                # Select first N UE antennas and keep dimensions separate
                csi_data = csi_data[:, :, :ue_antenna_count, :]  # Keep the correct dimension order
                
                # Keep 4D format: [batch, bs_antennas, ue_antennas, subcarriers]
                # Don't reshape - maintain UE antenna dimension separation
                
                self.logger.info(f"🔧 Using {ue_antenna_count} UE antennas (keeping dimensions separate)")
                self.logger.info(f"   Final CSI format: [batch, bs_antennas, ue_antennas, subcarriers]")
                self.logger.info(f"   CSI shape: {csi_data.shape}")
                
                # Update subcarrier count to total subcarriers (not multiplied by UE antennas)
                self.num_subcarriers = subcarriers
                self.logger.info(f"   Subcarriers per UE antenna: {subcarriers}")
                self.logger.info(f"   Total subcarriers: {self.num_subcarriers}")
                
                # Validate that network configuration matches data requirements
                self._validate_network_config_consistency(self.num_subcarriers)
            else:
                raise ValueError(f"Invalid ue_antenna_count: {ue_antenna_count}. Must be >= 1")
            
            # 🚀 Apply subcarrier sampling if configured
            if sampling_ratio < 1.0:
                csi_data = self._apply_subcarrier_sampling(
                    csi_data, sampling_ratio, sampling_method, antenna_consistent
                )
            
            # Phase differential calibration is handled by the network
            # No additional calibration needed here
            
        return ue_positions, bs_positions, antenna_indices, csi_data
    
    def _expand_bs_positions_to_3d(self, bs_positions_raw: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Expand BS positions from 1D to 3D coordinates"""
        self.logger.info(f"🔧 Processing BS positions...")
        self.logger.info(f"   Original shape: {bs_positions_raw.shape}")
        
        # Handle different BS position formats
        if bs_positions_raw.dim() == 1:
            # Single BS position for all samples
            if bs_positions_raw.shape[0] == 1:
                # Single value: expand to all samples
                bs_positions = bs_positions_raw.unsqueeze(0).expand(num_samples, -1)
                if bs_positions.shape[1] == 1:
                    # 1D case: convert to 3D coordinates (e.g., SSID values)
                    bs_positions_3d = torch.zeros(num_samples, 3)
                    bs_positions_3d[:, 0] = bs_positions[:, 0]  # X coordinate
                    bs_positions_3d[:, 1] = 0.0  # Y coordinate (default)
                    bs_positions_3d[:, 2] = 0.0  # Z coordinate (default)
                    bs_positions = bs_positions_3d
                    self.logger.info(f"   Converted 1D BS positions to 3D coordinates")
                else:
                    # Already 3D, just expand
                    bs_positions = bs_positions.expand(num_samples, -1)
            else:
                # Multiple BS positions: expand each to all samples
                bs_positions = bs_positions_raw.unsqueeze(0).expand(num_samples, -1)
                if bs_positions.shape[1] == 1:
                    # 1D case: convert to 3D coordinates
                    bs_positions_3d = torch.zeros(num_samples, 3)
                    bs_positions_3d[:, 0] = bs_positions[:, 0]  # X coordinate
                    bs_positions_3d[:, 1] = 0.0  # Y coordinate (default)
                    bs_positions_3d[:, 2] = 0.0  # Z coordinate (default)
                    bs_positions = bs_positions_3d
                    self.logger.info(f"   Converted 1D BS positions to 3D coordinates")
        elif bs_positions_raw.dim() == 2:
            # Already 2D: [num_samples, coordinates]
            if bs_positions_raw.shape[1] == 1:
                # 1D case: convert to 3D coordinates
                bs_positions_3d = torch.zeros(num_samples, 3)
                bs_positions_3d[:, 0] = bs_positions_raw[:, 0]  # X coordinate
                bs_positions_3d[:, 1] = 0.0  # Y coordinate (default)
                bs_positions_3d[:, 2] = 0.0  # Z coordinate (default)
                bs_positions = bs_positions_3d
                self.logger.info(f"   Converted 1D BS positions to 3D coordinates")
            elif bs_positions_raw.shape[1] == 3:
                # Already 3D coordinates
                bs_positions = bs_positions_raw
                self.logger.info(f"   BS positions already in 3D format")
            else:
                # Other 2D format: assume first 3 columns are coordinates
                bs_positions = bs_positions_raw[:, :3]
                self.logger.info(f"   Extracted first 3 columns as 3D coordinates")
        else:
            raise ValueError(f"Unsupported BS position format: {bs_positions_raw.shape}")
        
        self.logger.info(f"✅ BS position expansion completed:")
        self.logger.info(f"   Original shape: {bs_positions_raw.shape}")
        self.logger.info(f"   Expanded shape: {bs_positions.shape}")
        self.logger.info(f"   Original range: [{bs_positions_raw.min():.1f}, {bs_positions_raw.max():.1f}]")
        self.logger.info(f"   X range: [{bs_positions[:, 0].min():.1f}, {bs_positions[:, 0].max():.1f}]")
        self.logger.info(f"   Y values: all {bs_positions[:, 1].unique().item():.1f}")
        self.logger.info(f"   Z values: all {bs_positions[:, 2].unique().item():.1f}")
        
        return bs_positions
    
    def _validate_network_config_consistency(self, required_subcarriers: int):
        """Validate that network configuration matches data requirements."""
        self.logger.info(f"🔧 Validating network configuration consistency...")
        self.logger.info(f"   Required subcarriers per UE antenna from data: {required_subcarriers}")
        
        # Get expected subcarriers from configuration
        base_subcarriers = self.config_loader._processed_config.get('base_station', {}).get('ofdm', {}).get('num_subcarriers', None)
        if base_subcarriers is None:
            raise ValueError("❌ Configuration error: num_subcarriers not specified in base_station.ofdm section")
        ue_antenna_count = self.config_loader._processed_config.get('user_equipment', {}).get('ue_antenna_count', 1)
        expected_subcarriers_per_ue = base_subcarriers
        expected_total_subcarriers = base_subcarriers * ue_antenna_count
        
        self.logger.info(f"   Expected subcarriers per UE antenna from config: {expected_subcarriers_per_ue}")
        self.logger.info(f"   Expected total subcarriers from config: {expected_total_subcarriers}")
        self.logger.info(f"   Base subcarriers: {base_subcarriers}")
        self.logger.info(f"   UE antenna count: {ue_antenna_count}")
        
        # Validate subcarriers per UE antenna (not total subcarriers)
        if required_subcarriers != expected_subcarriers_per_ue:
            raise ValueError(
                f"❌ Configuration mismatch!\n"
                f"   Data requires: {required_subcarriers} subcarriers per UE antenna\n"
                f"   Config expects: {expected_subcarriers_per_ue} subcarriers per UE antenna\n"
                f"   Total subcarriers: {required_subcarriers * ue_antenna_count} (data) vs {expected_total_subcarriers} (config)\n"
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
