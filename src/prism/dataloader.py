"""
Data loader for Prism: Wideband RF Neural Radiance Fields.
Handles OFDM datasets with multiple subcarriers and MIMO configurations.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class PrismDataset(Dataset):
    """
    Dataset class for Prism model training and testing.
    Supports various OFDM configurations with wideband RF signals.
    """
    
    def __init__(self, config: Dict, split: str = 'train'):
        """
        Initialize the Prism dataset.
        
        Args:
            config: Configuration dictionary containing dataset parameters
            split: Dataset split ('train', 'val', 'test')
        """
        self.config = config
        self.split = split
        
        # Dataset parameters
        self.data_dir = config['data_dir']
        self.num_subcarriers = config.get('num_subcarriers', 1024)
        self.num_ue_antennas = config.get('num_ue_antennas', 2)
        self.num_bs_antennas = config.get('num_bs_antennas', 4)
        self.position_dim = config.get('position_dim', 3)
        
        # Load dataset
        self.data = self._load_dataset()
        
        # Split data
        self.indices = self._get_split_indices()
        
        logger.info(f"Loaded {split} dataset with {len(self.indices)} samples")
    
    def _load_dataset(self) -> Dict[str, np.ndarray]:
        """
        Load the dataset from files.
        
        Returns:
            Dictionary containing dataset arrays
        """
        data = {}
        
        # Load positions (3D coordinates)
        positions_file = os.path.join(self.data_dir, 'positions.npy')
        if os.path.exists(positions_file):
            data['positions'] = np.load(positions_file)
        else:
            # Generate synthetic positions if file doesn't exist
            logger.warning("Positions file not found, generating synthetic data")
            data['positions'] = self._generate_synthetic_positions()
        
        # Load UE antenna features
        ue_antennas_file = os.path.join(self.data_dir, 'ue_antennas.npy')
        if os.path.exists(ue_antennas_file):
            data['ue_antennas'] = np.load(ue_antennas_file)
        else:
            data['ue_antennas'] = self._generate_synthetic_ue_antennas()
        
        # Load BS antenna features
        bs_antennas_file = os.path.join(self.data_dir, 'bs_antennas.npy')
        if os.path.exists(bs_antennas_file):
            data['bs_antennas'] = np.load(bs_antennas_file)
        else:
            data['bs_antennas'] = self._generate_synthetic_bs_antennas()
        
        # Load additional RF features
        additional_features_file = os.path.join(self.data_dir, 'additional_features.npy')
        if os.path.exists(additional_features_file):
            data['additional_features'] = np.load(additional_features_file)
        else:
            data['additional_features'] = self._generate_synthetic_additional_features()
        
        # Load subcarrier responses (ground truth)
        subcarrier_file = os.path.join(self.data_dir, 'subcarrier_responses.npy')
        if os.path.exists(subcarrier_file):
            data['subcarrier_responses'] = np.load(subcarrier_file)
        else:
            data['subcarrier_responses'] = self._generate_synthetic_subcarrier_responses()
        
        # Validate data dimensions
        self._validate_data_dimensions(data)
        
        return data
    
    def _generate_synthetic_positions(self) -> np.ndarray:
        """Generate synthetic 3D positions."""
        num_samples = self.config.get('num_samples', 10000)
        # Generate positions in a 10x10x3 meter room
        positions = np.random.uniform(
            low=[0, 0, 0],
            high=[10, 10, 3],
            size=(num_samples, self.position_dim)
        )
        return positions.astype(np.float32)
    
    def _generate_synthetic_ue_antennas(self) -> np.ndarray:
        """Generate synthetic UE antenna features."""
        num_samples = self.config.get('num_samples', 10000)
        # Generate random antenna features (normalized)
        ue_antennas = np.random.normal(
            loc=0.0, scale=1.0,
            size=(num_samples, self.num_ue_antennas)
        )
        return ue_antennas.astype(np.float32)
    
    def _generate_synthetic_bs_antennas(self) -> np.ndarray:
        """Generate synthetic BS antenna features."""
        num_samples = self.config.get('num_samples', 10000)
        # Generate random antenna features (normalized)
        bs_antennas = np.random.normal(
            loc=0.0, scale=1.0,
            size=(num_samples, self.num_bs_antennas)
        )
        return bs_antennas.astype(np.float32)
    
    def _generate_synthetic_additional_features(self) -> np.ndarray:
        """Generate synthetic additional RF features."""
        num_samples = self.config.get('num_samples', 10000)
        # Generate 10 additional RF features (frequency, power, etc.)
        additional_features = np.random.normal(
            loc=0.0, scale=1.0,
            size=(num_samples, 10)
        )
        return additional_features.astype(np.float32)
    
    def _generate_synthetic_subcarrier_responses(self) -> np.ndarray:
        """Generate synthetic subcarrier responses."""
        num_samples = self.config.get('num_samples', 10000)
        
        # Generate realistic subcarrier responses with frequency-dependent characteristics
        subcarrier_responses = np.zeros((num_samples, self.num_subcarriers), dtype=np.float32)
        
        for i in range(num_samples):
            # Base response with frequency-dependent decay
            base_response = np.exp(-np.arange(self.num_subcarriers) / (self.num_subcarriers * 0.3))
            
            # Add random variations
            random_variation = np.random.normal(0, 0.1, self.num_subcarriers)
            
            # Add multipath effects (simplified)
            multipath = 0.3 * np.exp(-np.arange(self.num_subcarriers) / (self.num_subcarriers * 0.1))
            
            subcarrier_responses[i] = base_response + random_variation + multipath
        
        return subcarrier_responses
    
    def _validate_data_dimensions(self, data: Dict[str, np.ndarray]):
        """Validate that all data arrays have consistent dimensions."""
        num_samples = len(data['positions'])
        
        expected_shapes = {
            'positions': (num_samples, self.position_dim),
            'ue_antennas': (num_samples, self.num_ue_antennas),
            'bs_antennas': (num_samples, self.num_bs_antennas),
            'additional_features': (num_samples, 10),
            'subcarrier_responses': (num_samples, self.num_subcarriers)
        }
        
        for key, expected_shape in expected_shapes.items():
            if data[key].shape != expected_shape:
                raise ValueError(f"Data shape mismatch for {key}: "
                               f"expected {expected_shape}, got {data[key].shape}")
    
    def _get_split_indices(self) -> List[int]:
        """Get indices for the specified dataset split."""
        total_samples = len(self.data['positions'])
        
        # Split ratios
        train_ratio = self.config.get('train_ratio', 0.7)
        val_ratio = self.config.get('val_ratio', 0.15)
        test_ratio = self.config.get('test_ratio', 0.15)
        
        # Calculate split boundaries
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        if self.split == 'train':
            return list(range(0, train_end))
        elif self.split == 'val':
            return list(range(train_end, val_end))
        elif self.split == 'test':
            return list(range(val_end, total_samples))
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the sample data
        """
        data_idx = self.indices[idx]
        
        sample = {
            'positions': torch.from_numpy(self.data['positions'][data_idx]).float(),
            'ue_antennas': torch.from_numpy(self.data['ue_antennas'][data_idx]).float(),
            'bs_antennas': torch.from_numpy(self.data['bs_antennas'][data_idx]).float(),
            'additional_features': torch.from_numpy(self.data['additional_features'][data_idx]).float(),
            'subcarrier_responses': torch.from_numpy(self.data['subcarrier_responses'][data_idx]).float()
        }
        
        return sample
    
    def get_subcarrier_statistics(self) -> Dict[str, np.ndarray]:
        """
        Get statistics for each subcarrier across the dataset.
        
        Returns:
            Dictionary containing mean and std for each subcarrier
        """
        responses = self.data['subcarrier_responses'][self.indices]
        
        stats = {
            'mean': np.mean(responses, axis=0),
            'std': np.std(responses, axis=0),
            'min': np.min(responses, axis=0),
            'max': np.max(responses, axis=0)
        }
        
        return stats
    
    def normalize_data(self, method: str = 'standard'):
        """
        Normalize the dataset.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
        """
        if method == 'standard':
            # Standard normalization (zero mean, unit variance)
            for key in ['positions', 'ue_antennas', 'bs_antennas', 'additional_features']:
                data = self.data[key][self.indices]
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                std[std == 0] = 1.0  # Avoid division by zero
                
                self.data[key] = (self.data[key] - mean) / std
                
        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            for key in ['positions', 'ue_antennas', 'bs_antennas', 'additional_features']:
                data = self.data[key][self.indices]
                min_val = np.min(data, axis=0)
                max_val = np.max(data, axis=0)
                range_val = max_val - min_val
                range_val[range_val == 0] = 1.0  # Avoid division by zero
                
                self.data[key] = (self.data[key] - min_val) / range_val
                
        elif method == 'robust':
            # Robust normalization using median and IQR
            for key in ['positions', 'ue_antennas', 'bs_antennas', 'additional_features']:
                data = self.data[key][self.indices]
                median = np.median(data, axis=0)
                q75, q25 = np.percentile(data, [75, 25], axis=0)
                iqr = q75 - q25
                iqr[iqr == 0] = 1.0  # Avoid division by zero
                
                self.data[key] = (self.data[key] - median) / iqr
                
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        logger.info(f"Applied {method} normalization to dataset")

class PrismDataLoader:
    """
    Convenience class for creating data loaders with common configurations.
    """
    
    @staticmethod
    def create_loaders(config: Dict, batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Args:
            config: Configuration dictionary
            batch_size: Batch size for all loaders
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_dataset = PrismDataset(config, split='train')
        val_dataset = PrismDataset(config, split='val')
        test_dataset = PrismDataset(config, split='test')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
