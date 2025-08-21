"""
Sionna Data Loader for Prism
Loads and preprocesses Sionna-generated 5G OFDM simulation data.
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class SionnaDataLoader:
    """
    Data loader for Sionna-generated 5G OFDM simulation data.
    
    This loader handles HDF5 files containing:
    - Channel responses: (num_positions, num_subcarriers, num_ue_ant, num_bs_ant)
    - Path losses: (num_positions, num_subcarriers)
    - Delays: (num_positions, num_subcarriers)
    - UE positions: (num_positions, 3)
    - BS position: (3,)
    - Simulation configuration
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Sionna data loader.
        
        Args:
            config: Configuration dictionary containing Sionna integration settings
        """
        self.config = config
        self.sionna_config = config.get('sionna_integration', {})
        
        # Extract data file path
        self.data_file = config['data']['data_dir']
        self.data_type = config['data']['data_type']
        
        # Extract model parameters
        self.num_subcarriers = config['model']['num_subcarriers']
        self.num_ue_antennas = config['model']['num_ue_antennas']
        self.num_bs_antennas = config['model']['num_bs_antennas']
        self.position_dim = config['model']['position_dim']
        
        # Extract Sionna parameters
        self.carrier_frequency = float(self.sionna_config.get('carrier_frequency', 3.5e9))
        self.bandwidth = float(self.sionna_config.get('bandwidth', 100e6))
        self.fft_size = int(self.sionna_config.get('fft_size', 512))
        
        # HDF5 structure mapping
        self.hdf5_structure = self.sionna_config.get('hdf5_structure', {})
        
        # Preprocessing flags
        self.enable_frequency_normalization = self.sionna_config.get('enable_frequency_normalization', True)
        self.enable_spatial_normalization = self.sionna_config.get('enable_spatial_normalization', True)
        self.enable_channel_estimation = self.sionna_config.get('enable_channel_estimation', True)
        
        # Validate configuration
        self._validate_config()
        
        # Load data
        self._load_data()
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        if not Path(self.data_file).exists():
            raise FileNotFoundError(f"Sionna data file not found: {self.data_file}")
        
        if self.data_type != 'sionna_hdf5':
            raise ValueError(f"Expected data_type 'sionna_hdf5', got '{self.data_type}'")
        
        logger.info(f"Sionna data loader initialized with:")
        logger.info(f"  Data file: {self.data_file}")
        logger.info(f"  Subcarriers: {self.num_subcarriers}")
        logger.info(f"  UE antennas: {self.num_ue_antennas}")
        logger.info(f"  BS antennas: {self.num_bs_antennas}")
        logger.info(f"  Carrier frequency: {self.carrier_frequency/1e9:.1f} GHz")
        logger.info(f"  Bandwidth: {self.bandwidth/1e6:.0f} MHz")
    
    def _load_data(self):
        """Load data from HDF5 file."""
        logger.info(f"Loading Sionna data from {self.data_file}...")
        
        with h5py.File(self.data_file, 'r') as f:
            # Load channel responses
            self.channel_responses = f[self.hdf5_structure['channel_responses']][:]
            
            # Load path losses
            self.path_losses = f[self.hdf5_structure['path_losses']][:]
            
            # Load delays
            self.delays = f[self.hdf5_structure['delays']][:]
            
            # Load positions
            self.ue_positions = f[self.hdf5_structure['ue_positions']][:]
            self.bs_position = f[self.hdf5_structure['bs_position']][:]
            
            # Load simulation configuration
            self.simulation_config = dict(f[self.hdf5_structure['simulation_config']].attrs)
            
            # Load metadata
            if 'metadata' in f:
                self.metadata = dict(f['metadata'].attrs)
            else:
                self.metadata = {}
        
        # Validate data shapes
        self._validate_data_shapes()
        
        # Preprocess data
        self._preprocess_data()
        
        logger.info("Sionna data loaded successfully!")
        logger.info(f"  Channel responses: {self.channel_responses.shape}")
        logger.info(f"  Path losses: {self.path_losses.shape}")
        logger.info(f"  Delays: {self.delays.shape}")
        logger.info(f"  UE positions: {self.ue_positions.shape}")
        logger.info(f"  BS position: {self.bs_position.shape}")
    
    def _validate_data_shapes(self):
        """Validate that data shapes match expected configuration."""
        expected_shapes = {
            'channel_responses': (self.config['data']['num_samples'], self.num_subcarriers, 
                                self.num_ue_antennas, self.num_bs_antennas),
            'path_losses': (self.config['data']['num_samples'], self.num_subcarriers),
            'delays': (self.config['data']['num_samples'], self.num_subcarriers),
            'ue_positions': (self.config['data']['num_samples'], self.position_dim),
            'bs_position': (self.position_dim,)
        }
        
        actual_shapes = {
            'channel_responses': self.channel_responses.shape,
            'path_losses': self.path_losses.shape,
            'delays': self.delays.shape,
            'ue_positions': self.ue_positions.shape,
            'bs_position': self.bs_position.shape
        }
        
        for key, expected in expected_shapes.items():
            if actual_shapes[key] != expected:
                raise ValueError(f"Shape mismatch for {key}: expected {expected}, got {actual_shapes[key]}")
    
    def _preprocess_data(self):
        """Preprocess the loaded data."""
        logger.info("Preprocessing Sionna data...")
        
        # Convert to torch tensors
        self.channel_responses = torch.from_numpy(self.channel_responses).float()
        self.path_losses = torch.from_numpy(self.path_losses).float()
        self.delays = torch.from_numpy(self.delays).float()
        self.ue_positions = torch.from_numpy(self.ue_positions).float()
        self.bs_position = torch.from_numpy(self.bs_position).float()
        
        # Frequency normalization
        if self.enable_frequency_normalization:
            self._normalize_frequency()
        
        # Spatial normalization
        if self.enable_spatial_normalization:
            self._normalize_spatial()
        
        # Channel estimation enhancement
        if self.enable_channel_estimation:
            self._enhance_channel_estimation()
        
        logger.info("Data preprocessing completed!")
    
    def _normalize_frequency(self):
        """Normalize frequency-dependent data."""
        # Normalize channel responses across subcarriers
        for i in range(self.channel_responses.shape[0]):  # For each position
            for j in range(self.channel_responses.shape[2]):  # For each UE antenna
                for k in range(self.channel_responses.shape[3]):  # For each BS antenna
                    channel_slice = self.channel_responses[i, :, j, k]
                    mean_val = torch.mean(torch.abs(channel_slice))
                    if mean_val > 0:
                        self.channel_responses[i, :, j, k] = channel_slice / mean_val
        
        # Normalize path losses
        for i in range(self.path_losses.shape[0]):
            mean_path_loss = torch.mean(self.path_losses[i, :])
            if mean_path_loss > 0:
                self.path_losses[i, :] = self.path_losses[i, :] / mean_path_loss
        
        logger.info("Frequency normalization applied")
    
    def _normalize_spatial(self):
        """Normalize spatial coordinates."""
        # Normalize UE positions relative to BS
        ue_positions_relative = self.ue_positions - self.bs_position.unsqueeze(0)
        
        # Calculate spatial scale factor (max distance)
        max_distance = torch.max(torch.norm(ue_positions_relative, dim=1))
        if max_distance > 0:
            ue_positions_relative = ue_positions_relative / max_distance
        
        # Update positions
        self.ue_positions = ue_positions_relative
        self.bs_position = torch.zeros_like(self.bs_position)  # BS at origin
        
        logger.info("Spatial normalization applied")
    
    def _enhance_channel_estimation(self):
        """Enhance channel estimation with additional features."""
        # Calculate channel quality metrics
        channel_magnitudes = torch.abs(self.channel_responses)
        channel_phases = torch.angle(self.channel_responses)
        
        # Calculate SNR-like metric
        signal_power = torch.mean(channel_magnitudes**2, dim=(2, 3))  # Average across antennas
        noise_power = torch.var(channel_magnitudes, dim=(2, 3))       # Variance as noise estimate
        snr_metric = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        
        # Store enhanced metrics
        self.channel_quality = {
            'magnitudes': channel_magnitudes,
            'phases': channel_phases,
            'snr_metric': snr_metric,
            'path_losses': self.path_losses,
            'delays': self.delays
        }
        
        logger.info("Channel estimation enhancement applied")
    
    def get_data_split(self, train_ratio: float = 0.8, val_ratio: float = 0.0, 
                       test_ratio: float = 0.2, random_seed: int = 42) -> Dict[str, Dict]:
        """
        Split data into train/validation/test sets.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio (can be 0.0 for no validation set)
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing train/val/test splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Set random seed
        torch.manual_seed(random_seed)
        
        # Calculate split indices
        num_samples = self.channel_responses.shape[0]
        indices = torch.randperm(num_samples)
        
        train_end = int(train_ratio * num_samples)
        val_end = int((train_ratio + val_ratio) * num_samples)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end] if val_ratio > 0 else torch.tensor([])
        test_indices = indices[val_end:]
        
        # Create splits
        splits = {}
        
        # Always create train split
        splits['train'] = {
            'channel_responses': self.channel_responses[train_indices],
            'path_losses': self.path_losses[train_indices],
            'delays': self.delays[train_indices],
            'ue_positions': self.ue_positions[train_indices],
            'bs_position': self.bs_position,
            'indices': train_indices
        }
        
        # Create validation split only if val_ratio > 0
        if val_ratio > 0:
            splits['val'] = {
                'channel_responses': self.channel_responses[val_indices],
                'path_losses': self.path_losses[val_indices],
                'delays': self.delays[val_indices],
                'ue_positions': self.ue_positions[val_indices],
                'bs_position': self.bs_position,
                'indices': val_indices
            }
        
        # Always create test split
        splits['test'] = {
            'channel_responses': self.channel_responses[test_indices],
            'path_losses': self.path_losses[test_indices],
            'delays': self.delays[test_indices],
            'ue_positions': self.ue_positions[test_indices],
            'bs_position': self.bs_position,
            'indices': test_indices
        }
        
        logger.info(f"Data split created:")
        logger.info(f"  Train: {len(train_indices)} samples")
        if val_ratio > 0:
            logger.info(f"  Validation: {len(val_indices)} samples")
        else:
            logger.info(f"  Validation: 0 samples (disabled)")
        logger.info(f"  Test: {len(test_indices)} samples")
        
        return splits
    
    def get_batch(self, split: str, batch_size: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a batch of data from the specified split.
        
        Args:
            split: Split name ('train', 'val', 'test')
            batch_size: Batch size
            batch_idx: Batch index
            
        Returns:
            Dictionary containing batch data
        """
        if not hasattr(self, 'splits'):
            raise ValueError("Data splits not created. Call get_data_split() first.")
        
        if split not in self.splits:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(self.splits.keys())}")
        
        split_data = self.splits[split]
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(split_data['indices']))
        
        batch = {}
        for key, value in split_data.items():
            if key != 'indices':
                if key == 'bs_position':
                    # BS position is the same for all samples
                    batch[key] = value.unsqueeze(0).expand(end_idx - start_idx, -1)
                else:
                    batch[key] = value[start_idx:end_idx]
        
        return batch
    
    def get_statistics(self) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Get statistical information about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'num_samples': self.channel_responses.shape[0],
            'num_subcarriers': self.num_subcarriers,
            'num_ue_antennas': self.num_ue_antennas,
            'num_bs_antennas': self.num_bs_antennas,
            'position_dim': self.position_dim,
            'carrier_frequency': self.carrier_frequency,
            'bandwidth': self.bandwidth,
            'fft_size': self.fft_size
        }
        
        # Channel statistics
        if hasattr(self, 'channel_quality'):
            stats.update({
                'mean_channel_magnitude': torch.mean(self.channel_quality['magnitudes']).item(),
                'std_channel_magnitude': torch.std(self.channel_quality['magnitudes']).item(),
                'mean_snr_metric': torch.mean(self.channel_quality['snr_metric']).item(),
                'std_snr_metric': torch.std(self.channel_quality['snr_metric']).item()
            })
        
        # Position statistics
        ue_distances = torch.norm(self.ue_positions, dim=1)
        stats.update({
            'mean_ue_distance': torch.mean(ue_distances).item(),
            'max_ue_distance': torch.max(ue_distances).item(),
            'min_ue_distance': torch.min(ue_distances).item()
        })
        
        return stats
    
    def export_to_torch(self, output_path: str):
        """
        Export processed data to PyTorch format.
        
        Args:
            output_path: Output file path
        """
        export_data = {
            'channel_responses': self.channel_responses,
            'path_losses': self.path_losses,
            'delays': self.delays,
            'ue_positions': self.ue_positions,
            'bs_position': self.bs_position,
            'simulation_config': self.simulation_config,
            'metadata': self.metadata,
            'config': self.config
        }
        
        if hasattr(self, 'channel_quality'):
            export_data['channel_quality'] = self.channel_quality
        
        torch.save(export_data, output_path)
        logger.info(f"Data exported to {output_path}")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.channel_responses.shape[0]
    
    def __getitem__(self, idx):
        """Get a single sample by index."""
        return {
            'channel_responses': self.channel_responses[idx],
            'path_losses': self.path_losses[idx],
            'delays': self.delays[idx],
            'ue_positions': self.ue_positions[idx],
            'bs_position': self.bs_position
        }
