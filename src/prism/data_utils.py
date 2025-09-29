"""
Data utilities for Prism training and testing.

This module provides utilities for loading and splitting datasets.
"""

import numpy as np
import torch
import h5py
import logging
from typing import Tuple, Dict, Any, Optional, Iterator, List
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)

def load_and_split_by_pos(
    dataset_path: str,
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    mode: str = 'train',
    target_antenna_index: int = 0,
    sampling_ratio: float = 1.0,
    sampling_method: str = 'uniform',
    antenna_consistent: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Load data from HDF5 file and split into train/test sets by position.
    
    Args:
        dataset_path: Path to the HDF5 dataset file
        train_ratio: Ratio of data to use for training (0.0 to 1.0)
        test_ratio: Ratio of data to use for testing (0.0 to 1.0)
        random_seed: Random seed for reproducible splits
        mode: 'train' or 'test' - which split to return
        target_antenna_index: Which UE antenna to extract (0-based index)
        sampling_ratio: Ratio of subcarriers to sample (0.0 to 1.0)
        sampling_method: 'uniform' or 'random' - how to sample subcarriers
        antenna_consistent: If True, use same subcarrier indices for all UE antennas
        
    Returns:
        Tuple of (ue_positions, csi_data, bs_position, antenna_indices, metadata)
        Note: csi_data will have shape (samples, sampled_subcarriers, 1, bs_antennas) for the selected antenna
        
    Note:
        train_ratio + test_ratio can be < 1.0 to use only a subset of data
        Subcarrier sampling reduces memory usage and computation time
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
        ue_positions = f['data']['ue_positions'][:]
        csi_data = f['data']['csi'][:]
        bs_position = f['data']['bs_positions'][:]
        
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
    
    # 🚀 Subcarrier sampling for memory optimization
    original_num_subcarriers = csi_data.shape[1]
    if sampling_ratio < 1.0:
        num_sampled_subcarriers = max(1, int(original_num_subcarriers * sampling_ratio))
        
        # Set random seed for reproducible subcarrier sampling
        np.random.seed(random_seed)
        
        if sampling_method == 'uniform':
            # Uniform sampling: evenly spaced subcarriers
            subcarrier_indices = np.linspace(0, original_num_subcarriers - 1, 
                                           num_sampled_subcarriers, dtype=int)
        elif sampling_method == 'random':
            # Random sampling: randomly selected subcarriers
            subcarrier_indices = np.sort(np.random.choice(original_num_subcarriers, 
                                                        num_sampled_subcarriers, 
                                                        replace=False))
        else:
            raise ValueError(f"Unsupported sampling_method: {sampling_method}. Use 'uniform' or 'random'")
        
        # Apply subcarrier sampling
        csi_data = csi_data[:, subcarrier_indices, :, :]
        
        logger.info(f"🔧 Subcarrier sampling applied:")
        logger.info(f"   Method: {sampling_method}")
        logger.info(f"   Sampling ratio: {sampling_ratio}")
        logger.info(f"   Original subcarriers: {original_num_subcarriers}")
        logger.info(f"   Sampled subcarriers: {num_sampled_subcarriers}")
        logger.info(f"   Selected indices: {subcarrier_indices[:10]}{'...' if len(subcarrier_indices) > 10 else ''}")
        logger.info(f"   Final CSI shape: {csi_data.shape}")
        
        # Add sampling info to metadata
        metadata['subcarrier_sampling'] = {
            'enabled': True,
            'method': sampling_method,
            'ratio': sampling_ratio,
            'original_count': original_num_subcarriers,
            'sampled_count': num_sampled_subcarriers,
            'indices': subcarrier_indices.tolist(),
            'antenna_consistent': antenna_consistent
        }
    else:
        logger.info(f"🔧 No subcarrier sampling (ratio={sampling_ratio})")
        metadata['subcarrier_sampling'] = {
            'enabled': False,
            'original_count': original_num_subcarriers,
            'sampled_count': original_num_subcarriers
        }
    
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


def load_and_split_by_csi(
    dataset_path: str,
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    mode: str = 'train',
    sampling_ratio: float = 1.0,
    sampling_method: str = 'uniform',
    antenna_consistent: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Load data from HDF5 file and split into train/test sets by CSI.
    
    This function creates individual samples for each antenna combination to reduce memory usage.
    All datasets now have unified format: bs_positions (N, 3), ue_positions (N, 3), csi (N, bs_ant, ue_ant, subcarriers)
    
    Args:
        dataset_path: Path to the HDF5 dataset file
        train_ratio: Ratio of data to use for training (0.0 to 1.0)
        test_ratio: Ratio of data to use for testing (0.0 to 1.0)
        random_seed: Random seed for reproducible splits
        mode: 'train' or 'test' - which split to return
        sampling_ratio: Ratio of subcarriers to sample (0.0 to 1.0)
        sampling_method: 'uniform' or 'random' - how to sample subcarriers
        antenna_consistent: If True, use same subcarrier indices for all UE antennas
        
    Returns:
        Tuple of (bs_positions, ue_positions, bs_antenna_indices, ue_antenna_indices, csi_values, metadata)
        Each sample contains: [bs_position, ue_position, bs_antenna_index, ue_antenna_index, csi_value]
        
    Note:
        This method creates individual samples for each antenna combination to reduce memory usage.
        All datasets now have consistent 3D coordinate format for bs_positions.
    """
    logger.info(f"Loading dataset from {dataset_path} with CSI-based splitting")
    logger.info(f"Split configuration: train_ratio={train_ratio}, test_ratio={test_ratio}, seed={random_seed}")
    
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
        # Load all data - now all datasets have unified format
        ue_positions = f['data']['ue_positions'][:]  # (N, 3)
        csi_data = f['data']['csi'][:]               # (N, bs_ant, ue_ant, subcarriers)
        bs_positions = f['data']['bs_positions'][:]  # (N, 3) - now unified 3D format
        
        # Load metadata
        metadata = {}
        if 'metadata' in f:
            metadata_group = f['metadata']
            if hasattr(metadata_group, 'attrs'):
                metadata.update(dict(metadata_group.attrs))
            
            # Load config metadata
            if 'config' in metadata_group:
                config_group = metadata_group['config']
                if hasattr(config_group, 'attrs'):
                    metadata['config'] = dict(config_group.attrs)
        
        # Add file attributes to metadata
        if hasattr(f, 'attrs'):
            for key, value in f.attrs.items():
                metadata[key] = value
    
    # Get data dimensions
    num_samples = len(ue_positions)
    
    # CSI data format: [samples, bs_antennas, ue_antennas, subcarriers]
    num_bs_antennas = csi_data.shape[1]
    num_ue_antennas = csi_data.shape[2]
    num_subcarriers = csi_data.shape[3]
    
    logger.info(f"Data dimensions: {num_samples} samples, {num_bs_antennas} BS antennas, {num_ue_antennas} UE antennas, {num_subcarriers} subcarriers")
    logger.info(f"BS positions shape: {bs_positions.shape} (unified 3D format)")
    logger.info(f"UE positions shape: {ue_positions.shape}")
    
    # Apply relative position normalization to prevent numerical instability
    # Use the first BS position as origin for consistent relative coordinates
    origin = bs_positions[0].copy()  # Take first BS position as origin
    logger.info(f"🔄 Applying relative position normalization with origin: {origin}")
    
    # Convert to relative coordinates
    bs_positions_relative = bs_positions - origin
    ue_positions_relative = ue_positions - origin
    
    # Log position ranges before and after normalization
    logger.info(f"📊 Position normalization results:")
    logger.info(f"   Original BS range: [{np.min(bs_positions):.2f}, {np.max(bs_positions):.2f}]")
    logger.info(f"   Original UE range: [{np.min(ue_positions):.2f}, {np.max(ue_positions):.2f}]")
    logger.info(f"   Relative BS range: [{np.min(bs_positions_relative):.2f}, {np.max(bs_positions_relative):.2f}]")
    logger.info(f"   Relative UE range: [{np.min(ue_positions_relative):.2f}, {np.max(ue_positions_relative):.2f}]")
    
    # Update position arrays with normalized values
    bs_positions = bs_positions_relative
    ue_positions = ue_positions_relative
    
    # Store normalization info in metadata for consistent test processing
    metadata['position_normalization'] = {
        'origin': origin.tolist(),
        'method': 'relative_to_first_bs',
        'applied': True
    }
    
    # Apply subcarrier sampling
    if sampling_ratio < 1.0:
        num_sampled_subcarriers = int(num_subcarriers * sampling_ratio)
        
        if sampling_method == 'uniform':
            # Uniform sampling
            step = num_subcarriers // num_sampled_subcarriers
            sampled_indices = np.arange(0, num_subcarriers, step)[:num_sampled_subcarriers]
        elif sampling_method == 'random':
            # Random sampling
            np.random.seed(random_seed)
            sampled_indices = np.sort(np.random.choice(num_subcarriers, num_sampled_subcarriers, replace=False))
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        # Sample subcarriers - CSI data format: [samples, bs_antennas, ue_antennas, subcarriers]
        csi_data = csi_data[:, :, :, sampled_indices]
        logger.info(f"Subcarrier sampling: {num_subcarriers} -> {len(sampled_indices)} subcarriers")
    else:
        sampled_indices = np.arange(num_subcarriers)
    
    # Create position-based split (80% train, 20% test)
    np.random.seed(random_seed)
    position_indices = np.random.permutation(num_samples)
    train_size = int(num_samples * train_ratio)
    
    if mode == 'train':
        selected_position_indices = position_indices[:train_size]
    else:  # test
        selected_position_indices = position_indices[train_size:]
    
    logger.info(f"Position-based split: {len(selected_position_indices)} positions for {mode} mode")
    
    # Create individual samples for each antenna combination
    # Format: [position, bs_antenna_index, ue_antenna_index, csi_value]
    positions_list = []
    bs_antenna_indices_list = []
    ue_antenna_indices_list = []
    csi_values_list = []
    
    for pos_idx in selected_position_indices:
        ue_pos = ue_positions[pos_idx]
        bs_pos = bs_positions[pos_idx]
        
        # Debug: Check for NaN in original positions
        if np.isnan(ue_pos).any():
            logger.error(f"❌ NaN detected in UE position at index {pos_idx}")
            logger.error(f"   UE position: {ue_pos}")
            raise ValueError(f"NaN detected in UE position at index {pos_idx}")
        
        if np.isnan(bs_pos).any():
            logger.error(f"❌ NaN detected in BS position at index {pos_idx}")
            logger.error(f"   BS position: {bs_pos}")
            raise ValueError(f"NaN detected in BS position at index {pos_idx}")
        
        # For each UE antenna and BS antenna combination
        for ue_ant_idx in range(num_ue_antennas):
            for bs_ant_idx in range(num_bs_antennas):
                # Get CSI values for this antenna combination
                csi_values = csi_data[pos_idx, bs_ant_idx, ue_ant_idx, :]
                
                # Create sample: [position, bs_antenna_index, ue_antenna_index, csi_values]
                combined_pos = np.concatenate([ue_pos, bs_pos])  # Combine UE and BS positions
                
                # Debug: Check for NaN in combined position
                if np.isnan(combined_pos).any():
                    logger.error(f"❌ NaN detected in combined position at pos_idx={pos_idx}, ue_ant={ue_ant_idx}, bs_ant={bs_ant_idx}")
                    logger.error(f"   UE position: {ue_pos}")
                    logger.error(f"   BS position: {bs_pos}")
                    logger.error(f"   Combined position: {combined_pos}")
                    raise ValueError(f"NaN detected in combined position")
                
                positions_list.append(combined_pos)
                bs_antenna_indices_list.append(bs_ant_idx)
                ue_antenna_indices_list.append(ue_ant_idx)
                csi_values_list.append(csi_values)
    
    # Convert to tensors
    positions_tensor = torch.tensor(np.array(positions_list), dtype=torch.float32)
    bs_antenna_indices_tensor = torch.tensor(bs_antenna_indices_list, dtype=torch.long)
    ue_antenna_indices_tensor = torch.tensor(ue_antenna_indices_list, dtype=torch.long)
    csi_values_tensor = torch.tensor(np.array(csi_values_list), dtype=torch.complex64)
    
    # Debug: Check for NaN values in positions
    if torch.isnan(positions_tensor).any():
        logger.error(f"❌ NaN detected in positions_tensor")
        logger.error(f"   Positions shape: {positions_tensor.shape}")
        logger.error(f"   Positions min/max: {positions_tensor.min().item():.6f}/{positions_tensor.max().item():.6f}")
        logger.error(f"   NaN count: {torch.isnan(positions_tensor).sum().item()}")
        logger.error(f"   First few positions: {positions_tensor[:5]}")
        raise ValueError("NaN detected in positions_tensor")
    
    # Keep individual CSI samples format (each sample contains one CSI value)
    # Split positions back to UE and BS positions
    ue_positions_final = positions_tensor[:, :3]  # First 3 elements are UE position
    bs_positions_final = positions_tensor[:, 3:6]  # Next 3 elements are BS position
    
    # Keep individual antenna indices (one per sample)
    bs_ant_indices = bs_antenna_indices_tensor  # Shape: [num_samples]
    ue_ant_indices = ue_antenna_indices_tensor  # Shape: [num_samples]
    
    # Keep individual CSI values (one per sample)
    csi_data_final = csi_values_tensor  # Shape: [num_samples, subcarriers]
    
    # Add split information to metadata
    metadata.update({
        'split_mode': mode,
        'split_random_seed': random_seed,
        'split_train_ratio': train_ratio,
        'split_test_ratio': test_ratio,
        'split_num_positions': len(selected_position_indices),
        'split_total_positions': num_samples,
        'split_position_indices': selected_position_indices.tolist(),
        'num_samples_per_position': num_ue_antennas * num_bs_antennas,
        'total_samples': len(positions_list),
        'sampled_subcarriers': len(sampled_indices),
        'sampling_method': sampling_method,
        'antenna_consistent': antenna_consistent
    })
    
    logger.info(f"CSI-based data loading completed successfully")
    logger.info(f"  BS positions: {bs_positions_final.shape} - {bs_positions_final.dtype}")
    logger.info(f"  UE positions: {ue_positions_final.shape} - {ue_positions_final.dtype}")
    logger.info(f"  BS antenna indices: {bs_ant_indices.shape} - {bs_ant_indices.dtype}")
    logger.info(f"  UE antenna indices: {ue_ant_indices.shape} - {ue_ant_indices.dtype}")
    logger.info(f"  CSI data: {csi_data_final.shape} - {csi_data_final.dtype}")
    logger.info(f"  Total samples: {len(positions_list)} (positions × UE_antennas × BS_antennas)")
    
    return bs_positions_final, ue_positions_final, bs_ant_indices, ue_ant_indices, csi_data_final, metadata


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


def load_and_split_by_position_and_csi(
    dataset_path: str,
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    mode: str = 'train',
    sampling_ratio: float = 1.0,
    sampling_method: str = 'uniform',
    antenna_consistent: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Load data from HDF5 file and split into train/test sets by position pairs.
    
    This function ensures that all antenna combinations for the same position pair
    (bs_position, ue_position) are grouped together in the same batch.
    Batch size represents the number of position pairs, not individual samples.
    
    Args:
        dataset_path: Path to the HDF5 dataset file
        train_ratio: Ratio of position pairs to use for training (0.0 to 1.0)
        test_ratio: Ratio of position pairs to use for testing (0.0 to 1.0)
        random_seed: Random seed for reproducible splits
        mode: 'train' or 'test' - which split to return
        sampling_ratio: Ratio of subcarriers to sample (0.0 to 1.0)
        sampling_method: 'uniform' or 'random' - how to sample subcarriers
        antenna_consistent: If True, use same subcarrier indices for all UE antennas
        
    Returns:
        Tuple of (bs_positions, ue_positions, bs_antenna_indices, ue_antenna_indices, csi_values, metadata)
        Each sample contains: [bs_position, ue_position, bs_antenna_index, ue_antenna_index, csi_value]
        
    Note:
        This method groups samples by position pairs to ensure all antenna combinations
        for the same position pair are processed together in the same batch.
    """
    logger.info(f"Loading dataset from {dataset_path} with position-pair-based splitting")
    logger.info(f"Split configuration: train_ratio={train_ratio}, test_ratio={test_ratio}, seed={random_seed}")
    
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
        ue_positions = f['data']['ue_positions'][:]
        bs_positions = f['data']['bs_positions'][:]
        csi_data = f['data']['csi'][:]
        
        # Load metadata
        metadata = {}
        if 'simulation_config' in f and hasattr(f['simulation_config'], 'attrs'):
            metadata['simulation_params'] = dict(f['simulation_config'].attrs)
        
        # Add file attributes to metadata
        if hasattr(f, 'attrs'):
            for key, value in f.attrs.items():
                metadata[key] = value
    
    # Get dimensions
    num_samples = len(ue_positions)
    num_bs_antennas = csi_data.shape[1]
    num_ue_antennas = csi_data.shape[2]
    num_subcarriers = csi_data.shape[3]
    
    logger.info(f"Dataset dimensions:")
    logger.info(f"  Total position pairs: {num_samples}")
    logger.info(f"  BS antennas: {num_bs_antennas}")
    logger.info(f"  UE antennas: {num_ue_antennas}")
    logger.info(f"  Subcarriers: {num_subcarriers}")
    
    # Validate data consistency
    if csi_data.shape[0] != num_samples:
        raise ValueError(f"Data mismatch: {csi_data.shape[0]} CSI samples vs {num_samples} UE positions")
    
    # Apply subcarrier sampling
    if sampling_ratio < 1.0:
        num_sampled_subcarriers = int(num_subcarriers * sampling_ratio)
        
        if sampling_method == 'uniform':
            # Uniform sampling
            step = num_subcarriers // num_sampled_subcarriers
            sampled_indices = np.arange(0, num_subcarriers, step)[:num_sampled_subcarriers]
        elif sampling_method == 'random':
            # Random sampling
            np.random.seed(random_seed)
            sampled_indices = np.sort(np.random.choice(num_subcarriers, num_sampled_subcarriers, replace=False))
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        # Sample subcarriers - CSI data format: [samples, bs_antennas, ue_antennas, subcarriers]
        csi_data = csi_data[:, :, :, sampled_indices]
        logger.info(f"Subcarrier sampling: {num_subcarriers} -> {len(sampled_indices)} subcarriers")
    else:
        sampled_indices = np.arange(num_subcarriers)
    
    # Create position-pair-based split
    np.random.seed(random_seed)
    position_indices = np.random.permutation(num_samples)
    train_size = int(num_samples * train_ratio)
    
    if mode == 'train':
        selected_position_indices = position_indices[:train_size]
    else:  # test
        selected_position_indices = position_indices[train_size:]
    
    logger.info(f"Position-pair-based split: {len(selected_position_indices)} position pairs for {mode} mode")
    
    # Create grouped samples for each position pair
    # Each position pair will have all antenna combinations grouped together
    positions_list = []
    bs_antenna_indices_list = []
    ue_antenna_indices_list = []
    csi_values_list = []
    position_pair_indices = []  # Track which position pair each sample belongs to
    
    for group_idx, pos_idx in enumerate(selected_position_indices):
        ue_pos = ue_positions[pos_idx]
        bs_pos = bs_positions[pos_idx]
        
        # Debug: Check for NaN in original positions
        if np.isnan(ue_pos).any():
            logger.error(f"❌ NaN detected in UE position at index {pos_idx}")
            logger.error(f"   UE position: {ue_pos}")
            raise ValueError(f"NaN detected in UE position at index {pos_idx}")
        
        if np.isnan(bs_pos).any():
            logger.error(f"❌ NaN detected in BS position at index {pos_idx}")
            logger.error(f"   BS position: {bs_pos}")
            raise ValueError(f"NaN detected in BS position at index {pos_idx}")
        
        # For each antenna combination in this position pair
        for ue_ant_idx in range(num_ue_antennas):
            for bs_ant_idx in range(num_bs_antennas):
                # Get CSI values for this antenna combination
                csi_values = csi_data[pos_idx, bs_ant_idx, ue_ant_idx, :]
                
                # Create sample: [position, bs_antenna_index, ue_antenna_index, csi_values]
                combined_pos = np.concatenate([ue_pos, bs_pos])  # Combine UE and BS positions
                
                # Debug: Check for NaN in combined position
                if np.isnan(combined_pos).any():
                    logger.error(f"❌ NaN detected in combined position at pos_idx={pos_idx}, ue_ant={ue_ant_idx}, bs_ant={bs_ant_idx}")
                    logger.error(f"   UE position: {ue_pos}")
                    logger.error(f"   BS position: {bs_pos}")
                    logger.error(f"   Combined position: {combined_pos}")
                    raise ValueError(f"NaN detected in combined position")
                
                positions_list.append(combined_pos)
                bs_antenna_indices_list.append(bs_ant_idx)
                ue_antenna_indices_list.append(ue_ant_idx)
                csi_values_list.append(csi_values)
                position_pair_indices.append(group_idx)  # Track position pair group
    
    # Convert to tensors
    positions_tensor = torch.tensor(np.array(positions_list), dtype=torch.float32)
    bs_antenna_indices_tensor = torch.tensor(bs_antenna_indices_list, dtype=torch.long)
    ue_antenna_indices_tensor = torch.tensor(ue_antenna_indices_list, dtype=torch.long)
    csi_values_tensor = torch.tensor(np.array(csi_values_list), dtype=torch.complex64)
    position_pair_indices_tensor = torch.tensor(position_pair_indices, dtype=torch.long)
    
    # Debug: Check for NaN values in positions
    if torch.isnan(positions_tensor).any():
        logger.error(f"❌ NaN detected in positions_tensor")
        logger.error(f"   Positions shape: {positions_tensor.shape}")
        logger.error(f"   Positions min/max: {positions_tensor.min().item():.6f}/{positions_tensor.max().item():.6f}")
        logger.error(f"   NaN count: {torch.isnan(positions_tensor).sum().item()}")
        logger.error(f"   First few positions: {positions_tensor[:5]}")
        raise ValueError("NaN detected in positions_tensor")
    
    # Split positions back to UE and BS positions
    ue_positions_final = positions_tensor[:, :3]  # First 3 elements are UE position
    bs_positions_final = positions_tensor[:, 3:6]  # Next 3 elements are BS position
    
    # Add position pair grouping information to metadata
    metadata.update({
        'split_mode': mode,
        'split_random_seed': random_seed,
        'split_train_ratio': train_ratio,
        'split_test_ratio': test_ratio,
        'split_num_position_pairs': len(selected_position_indices),
        'split_total_position_pairs': num_samples,
        'split_position_indices': selected_position_indices.tolist(),
        'num_antenna_combinations_per_pair': num_ue_antennas * num_bs_antennas,
        'total_samples': len(positions_list),
        'sampled_subcarriers': len(sampled_indices),
        'sampling_method': sampling_method,
        'antenna_consistent': antenna_consistent,
        'position_pair_grouping': {
            'enabled': True,
            'group_indices': position_pair_indices_tensor.tolist(),
            'num_groups': len(selected_position_indices),
            'samples_per_group': num_ue_antennas * num_bs_antennas
        }
    })
    
    logger.info(f"Position-pair-based data loading completed successfully")
    logger.info(f"  BS positions: {bs_positions_final.shape} - {bs_positions_final.dtype}")
    logger.info(f"  UE positions: {ue_positions_final.shape} - {ue_positions_final.dtype}")
    logger.info(f"  BS antenna indices: {bs_antenna_indices_tensor.shape} - {bs_antenna_indices_tensor.dtype}")
    logger.info(f"  UE antenna indices: {ue_antenna_indices_tensor.shape} - {ue_antenna_indices_tensor.dtype}")
    logger.info(f"  CSI data: {csi_values_tensor.shape} - {csi_values_tensor.dtype}")
    logger.info(f"  Total samples: {len(positions_list)} (position_pairs × UE_antennas × BS_antennas)")
    logger.info(f"  Position pairs: {len(selected_position_indices)}")
    logger.info(f"  Samples per position pair: {num_ue_antennas * num_bs_antennas}")
    logger.info(f"  ✅ Position pair grouping enabled - all antenna combinations for same position pair are grouped together")
    
    return bs_positions_final, ue_positions_final, bs_antenna_indices_tensor, ue_antenna_indices_tensor, csi_values_tensor, metadata


class PositionPairBatchSampler(Sampler):
    """
    Custom batch sampler that groups samples by position pairs.
    
    This sampler ensures that all antenna combinations for the same position pair
    are processed together in the same batch. Batch size represents the number of
    position pairs, not individual samples.
    
    Args:
        position_pair_indices: List of position pair group indices for each sample
        batch_size: Number of position pairs per batch (not individual samples)
        shuffle: Whether to shuffle position pairs between epochs
        drop_last: Whether to drop the last incomplete batch
    """
    
    def __init__(
        self,
        position_pair_indices: List[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.position_pair_indices = position_pair_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Get unique position pair groups
        self.position_pair_groups = list(set(position_pair_indices))
        self.num_position_pairs = len(self.position_pair_groups)
        
        # Calculate samples per position pair
        samples_per_pair = len(position_pair_indices) // self.num_position_pairs
        self.samples_per_position_pair = samples_per_pair
        
        logger.info(f"PositionPairBatchSampler initialized:")
        logger.info(f"  Total position pairs: {self.num_position_pairs}")
        logger.info(f"  Samples per position pair: {self.samples_per_position_pair}")
        logger.info(f"  Batch size (position pairs): {self.batch_size}")
        logger.info(f"  Shuffle: {self.shuffle}")
        logger.info(f"  Drop last: {self.drop_last}")
        
        # Validate batch size
        if self.batch_size > self.num_position_pairs:
            logger.warning(f"Batch size ({self.batch_size}) > number of position pairs ({self.num_position_pairs})")
            logger.warning("Will use all position pairs in each batch")
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches of sample indices.
        
        Each batch contains all antenna combinations for the selected position pairs.
        """
        # Get position pair groups
        position_pairs = self.position_pair_groups.copy()
        
        # Shuffle position pairs if requested
        if self.shuffle:
            import random
            random.shuffle(position_pairs)
        
        # Create batches of position pairs
        for i in range(0, len(position_pairs), self.batch_size):
            batch_position_pairs = position_pairs[i:i + self.batch_size]
            
            # Skip incomplete batch if drop_last is True
            if self.drop_last and len(batch_position_pairs) < self.batch_size:
                break
            
            # Get all sample indices for these position pairs
            batch_sample_indices = []
            for pos_pair_idx in batch_position_pairs:
                # Find all samples belonging to this position pair
                for sample_idx, group_idx in enumerate(self.position_pair_indices):
                    if group_idx == pos_pair_idx:
                        batch_sample_indices.append(sample_idx)
            
            # Sort sample indices within each position pair for consistency
            batch_sample_indices.sort()
            
            logger.debug(f"Batch {i // self.batch_size + 1}: {len(batch_position_pairs)} position pairs, {len(batch_sample_indices)} samples")
            yield batch_sample_indices
    
    def __len__(self) -> int:
        """Return the number of batches."""
        if self.drop_last:
            return self.num_position_pairs // self.batch_size
        else:
            return (self.num_position_pairs + self.batch_size - 1) // self.batch_size


def create_position_aware_dataloader(
    dataset: torch.utils.data.Dataset,
    position_pair_indices: List[int],
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with position-aware batch sampling.
    
    Args:
        dataset: PyTorch dataset
        position_pair_indices: List of position pair group indices for each sample
        batch_size: Number of position pairs per batch
        shuffle: Whether to shuffle position pairs between epochs
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader with position-aware batch sampling
    """
    # Create position-aware batch sampler
    batch_sampler = PositionPairBatchSampler(
        position_pair_indices=position_pair_indices,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    
    # Create DataLoader with custom batch sampler
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Position-aware DataLoader created:")
    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Number of batches: {len(batch_sampler)}")
    logger.info(f"  Batch size (position pairs): {batch_size}")
    logger.info(f"  Samples per batch: {batch_size * batch_sampler.samples_per_position_pair}")
    
    return dataloader
