"""
Data utilities for Prism training and testing.

This module provides utilities for loading and splitting datasets.
"""

import numpy as np
import torch
import h5py
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_and_split_data(
    dataset_path: str,
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    mode: str = 'train',
    target_antenna_index: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Load data from HDF5 file and split into train/test sets.
    
    Args:
        dataset_path: Path to the HDF5 dataset file
        train_ratio: Ratio of data to use for training (0.0 to 1.0)
        test_ratio: Ratio of data to use for testing (0.0 to 1.0)
        random_seed: Random seed for reproducible splits
        mode: 'train' or 'test' - which split to return
        target_antenna_index: Which UE antenna to extract (0-based index)
        
    Returns:
        Tuple of (ue_positions, csi_data, bs_position, antenna_indices, metadata)
        Note: csi_data will have shape (samples, subcarriers, 1, bs_antennas) for the selected antenna
        
    Note:
        train_ratio + test_ratio can be < 1.0 to use only a subset of data
    """
    logger.info(f"Loading dataset from {dataset_path}")
    logger.info(f"Split configuration: train_ratio={train_ratio}, test_ratio={test_ratio}, seed={random_seed}")
    logger.info(f"Target UE antenna index: {target_antenna_index}")
    
    # Validate parameters
    if train_ratio < 0 or train_ratio > 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if test_ratio < 0 or test_ratio > 1:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
    if train_ratio + test_ratio > 1:
        raise ValueError(f"train_ratio + test_ratio cannot exceed 1.0, got {train_ratio + test_ratio}")
    if mode not in ['train', 'test']:
        raise ValueError(f"mode must be 'train' or 'test', got {mode}")
    
    with h5py.File(dataset_path, 'r') as f:
        # Load all data
        ue_positions = f['positions']['ue_positions'][:]
        csi_data = f['channel_data']['channel_responses'][:]
        bs_position = f['positions']['bs_position'][:]
        
        # Load antenna indices if available
        if 'antenna_indices' in f:
            antenna_indices = f['antenna_indices'][:]
        else:
            # Create default antenna indices
            num_bs_antennas = csi_data.shape[3] if len(csi_data.shape) > 3 else 64
            antenna_indices = np.arange(num_bs_antennas)
            logger.info(f"Created default antenna indices: {len(antenna_indices)}")
        
        # Load metadata
        metadata = {}
        if 'simulation_config' in f and hasattr(f['simulation_config'], 'attrs'):
            metadata['simulation_params'] = dict(f['simulation_config'].attrs)
        
        # Add file attributes to metadata
        if hasattr(f, 'attrs'):
            for key, value in f.attrs.items():
                metadata[key] = value
    
    # Get total number of samples
    num_samples = len(ue_positions)
    logger.info(f"Total samples in dataset: {num_samples}")
    
    # Validate data consistency
    if csi_data.shape[0] != num_samples:
        raise ValueError(f"Data mismatch: {csi_data.shape[0]} CSI samples vs {num_samples} UE positions")
    
    # Validate target antenna index
    num_ue_antennas = csi_data.shape[2] if len(csi_data.shape) > 2 else 1
    if target_antenna_index >= num_ue_antennas:
        raise ValueError(f"target_antenna_index ({target_antenna_index}) >= available UE antennas ({num_ue_antennas})")
    
    logger.info(f"Dataset has {num_ue_antennas} UE antennas, extracting antenna {target_antenna_index}")
    
    # Extract only the target antenna data to reduce memory usage
    # Original shape: (samples, subcarriers, ue_antennas, bs_antennas)
    # New shape: (samples, subcarriers, 1, bs_antennas)
    csi_data = csi_data[:, :, target_antenna_index:target_antenna_index+1, :]
    logger.info(f"Extracted CSI data shape: {csi_data.shape}")
    
    # Set random seed for reproducible splits
    np.random.seed(random_seed)
    
    # Create indices for splitting
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    
    # Create splits
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]
    
    logger.info(f"Split sizes: train={len(train_indices)}, test={len(test_indices)}")
    
    # Select the appropriate split
    if mode == 'train':
        selected_indices = train_indices
        logger.info(f"Returning training split with {len(selected_indices)} samples")
    else:  # mode == 'test'
        selected_indices = test_indices
        logger.info(f"Returning testing split with {len(selected_indices)} samples")
    
    # Extract data for the selected split
    split_ue_positions = ue_positions[selected_indices]
    split_csi_data = csi_data[selected_indices]
    
    # Convert to tensors
    ue_positions_tensor = torch.tensor(split_ue_positions, dtype=torch.float32)
    csi_data_tensor = torch.tensor(split_csi_data, dtype=torch.complex64)
    bs_position_tensor = torch.tensor(bs_position, dtype=torch.float32)
    antenna_indices_tensor = torch.tensor(antenna_indices, dtype=torch.long)
    
    # Add split information to metadata
    metadata.update({
        'split_mode': mode,
        'split_random_seed': random_seed,
        'split_train_ratio': train_ratio,
        'split_test_ratio': test_ratio,
        'split_num_samples': len(selected_indices),
        'split_total_samples': num_samples,
        'split_indices': selected_indices.tolist()
    })
    
    logger.info(f"Data loading completed successfully")
    logger.info(f"  UE positions: {ue_positions_tensor.shape} - {ue_positions_tensor.dtype}")
    logger.info(f"  CSI data: {csi_data_tensor.shape} - {csi_data_tensor.dtype}")
    logger.info(f"  BS position: {bs_position_tensor.shape} - {bs_position_tensor.dtype}")
    logger.info(f"  Antenna indices: {antenna_indices_tensor.shape} - {antenna_indices_tensor.dtype}")
    
    return ue_positions_tensor, csi_data_tensor, bs_position_tensor, antenna_indices_tensor, metadata


def check_dataset_compatibility(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Check dataset configuration and return appropriate data path and split config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (dataset_path, split_config)
        
    Raises:
        ValueError: If configuration is invalid or missing required keys
    """
    input_config = config.get('input', {})
    
    # Check if new single dataset configuration is available
    if 'dataset_path' in input_config and input_config['dataset_path']:
        dataset_path = input_config['dataset_path']
        split_config = input_config.get('split', {})
        
        # Validate split configuration
        if 'random_seed' not in split_config:
            split_config['random_seed'] = 42  # Default seed
        if 'train_ratio' not in split_config:
            split_config['train_ratio'] = 0.8  # Default train ratio
        if 'test_ratio' not in split_config:
            split_config['test_ratio'] = 0.2  # Default test ratio
            
        logger.info(f"Using single dataset configuration: {dataset_path}")
        logger.info(f"Split config: {split_config}")
        
        return dataset_path, split_config
    

    
    else:
        raise ValueError("No valid dataset configuration found. Please provide 'dataset_path' with split configuration in your config file.")
