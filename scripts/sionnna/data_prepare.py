#!/usr/bin/env python3
"""
Data Preparation Script for Prism Training

This script prepares the Sionna simulation data by splitting it into
training (80%) and testing (20%) sets, and saves them as separate HDF5 files.
"""

import os
import sys
import argparse
import logging
import h5py
import numpy as np
from pathlib import Path
import random
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def split_data(data_path: str, output_dir: str, train_ratio: float = 0.8, random_seed: int = 42):
    """
    Split the Sionna simulation data into training and testing sets
    
    Args:
        data_path: Path to the original HDF5 data file
        output_dir: Directory to save the split datasets
        train_ratio: Ratio of training data (default: 0.8 for 80%)
        random_seed: Random seed for reproducible splits
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from {data_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Training ratio: {train_ratio:.1%}")
    
    # Load the original data with flexible key handling
    with h5py.File(data_path, 'r') as f:
        # Try different possible key structures for sionna data
        if 'positions' in f:
            # Sionna data structure
            ue_positions = f['positions/ue_positions'][:]
            bs_position = f['positions/bs_position'][:]
            logger.info("Using sionna data structure (nested positions)")
        elif 'ue_positions' in f:
            # Direct key structure
            ue_positions = f['ue_positions'][:]
            bs_position = f['bs_position'][:]
            logger.info("Using direct key structure")
        else:
            raise ValueError("Could not find UE positions in the data file")
        
        # Try different possible key structures for channel data
        if 'channel_data' in f and 'channel_responses' in f['channel_data']:
            # Sionna data structure
            channel_responses = f['channel_data/channel_responses'][:]
            logger.info("Using sionna channel data structure (nested)")
        elif 'channel_responses' in f:
            # Direct key structure
            channel_responses = f['channel_responses'][:]
            logger.info("Using direct channel data structure")
        else:
            raise ValueError("Could not find channel responses in the data file")
        
        # Load simulation parameters if available
        sim_params = {}
        if 'simulation_config' in f:
            sim_params = dict(f['simulation_config'].attrs)
            logger.info("Loaded simulation config parameters")
        elif 'simulation_params' in f:
            sim_params = dict(f['simulation_params'].attrs)
            logger.info("Loaded simulation params")
        
        # Load additional data if available
        additional_data = {}
        if 'channel_data' in f:
            for key in f['channel_data'].keys():
                if key != 'channel_responses':  # Already loaded
                    additional_data[key] = f[f'channel_data/{key}'][:]
                    logger.info(f"Loaded additional data: {key} with shape {additional_data[key].shape}")
        
        # Create antenna indices if not present
        if 'antenna_indices' not in additional_data:
            # Create default antenna indices based on channel response shape
            if len(channel_responses.shape) >= 3:
                num_bs_antennas = channel_responses.shape[-1]  # Last dimension
                antenna_indices = np.arange(num_bs_antennas)
                additional_data['antenna_indices'] = antenna_indices
                logger.info(f"Created default antenna indices: {antenna_indices}")
        
        logger.info(f"Original data shape:")
        logger.info(f"  UE positions: {ue_positions.shape}")
        logger.info(f"  Channel responses: {channel_responses.shape}")
        logger.info(f"  BS position: {bs_position.shape}")
        if additional_data:
            for key, data in additional_data.items():
                logger.info(f"  {key}: {data.shape}")
    
    # Calculate split indices
    num_samples = len(ue_positions)
    num_train = int(num_samples * train_ratio)
    num_test = num_samples - num_train
    
    logger.info(f"Data split:")
    logger.info(f"  Total samples: {num_samples}")
    logger.info(f"  Training samples: {num_train} ({train_ratio:.1%})")
    logger.info(f"  Testing samples: {num_test} ({1-train_ratio:.1%})")
    
    # Create random permutation for shuffling
    indices = np.random.permutation(num_samples)
    
    # Split indices
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    
    # Split data
    train_ue_positions = ue_positions[train_indices]
    train_channel_responses = channel_responses[train_indices]
    
    test_ue_positions = ue_positions[test_indices]
    test_channel_responses = channel_responses[test_indices]
    
    # Split additional data if present
    train_additional = {}
    test_additional = {}
    for key, data in additional_data.items():
        if len(data.shape) > 0 and data.shape[0] == num_samples:
            train_additional[key] = data[train_indices]
            test_additional[key] = data[test_indices]
        else:
            # Data doesn't have sample dimension, copy as is
            train_additional[key] = data
            test_additional[key] = data
    
    # Save training data
    train_file_path = output_path / 'train_data.h5'
    logger.info(f"Saving training data to {train_file_path}")
    
    with h5py.File(train_file_path, 'w') as f:
        f.create_dataset('ue_positions', data=train_ue_positions)
        f.create_dataset('channel_responses', data=train_channel_responses)
        f.create_dataset('bs_position', data=bs_position)
        
        # Save additional data
        for key, data in train_additional.items():
            f.create_dataset(key, data=data)
        
        # Create simulation parameters group
        sim_group = f.create_group('simulation_params')
        for key, value in sim_params.items():
            sim_group.attrs[key] = value
        
        # Add split information
        f.attrs['split_type'] = 'train'
        f.attrs['num_samples'] = num_train
        f.attrs['split_ratio'] = train_ratio
        f.attrs['split_timestamp'] = datetime.now().isoformat()
        f.attrs['original_file'] = str(data_path)
    
    # Save testing data
    test_file_path = output_path / 'test_data.h5'
    logger.info(f"Saving testing data to {test_file_path}")
    
    with h5py.File(test_file_path, 'w') as f:
        f.create_dataset('ue_positions', data=test_ue_positions)
        f.create_dataset('channel_responses', data=test_channel_responses)
        f.create_dataset('bs_position', data=bs_position)
        
        # Save additional data
        for key, data in test_additional.items():
            f.create_dataset(key, data=data)
        
        # Create simulation parameters group
        sim_group = f.create_group('simulation_params')
        for key, value in sim_params.items():
            sim_group.attrs[key] = value
        
        # Add split information
        f.attrs['split_type'] = 'test'
        f.attrs['num_samples'] = num_test
        f.attrs['split_ratio'] = 1 - train_ratio
        f.attrs['split_timestamp'] = datetime.now().isoformat()
        f.attrs['original_file'] = str(data_path)
    
    # Save split indices for reproducibility
    indices_file_path = output_path / 'split_indices.npz'
    np.savez_compressed(
        indices_file_path,
        train_indices=train_indices,
        test_indices=test_indices,
        random_seed=random_seed,
        train_ratio=train_ratio
    )
    
    # Create summary file
    summary_file_path = output_path / 'split_summary.txt'
    with open(summary_file_path, 'w') as f:
        f.write("Data Split Summary\n")
        f.write("==================\n\n")
        f.write(f"Original data file: {data_path}\n")
        f.write(f"Split timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Random seed: {random_seed}\n")
        f.write(f"Training ratio: {train_ratio:.1%}\n\n")
        
        f.write("Data Shapes:\n")
        f.write(f"  UE positions: {ue_positions.shape}\n")
        f.write(f"  Channel responses: {channel_responses.shape}\n")
        f.write(f"  BS position: {bs_position.shape}\n")
        if additional_data:
            for key, data in additional_data.items():
                f.write(f"  {key}: {data.shape}\n")
        
        f.write(f"\nSplit Results:\n")
        f.write(f"  Total samples: {num_samples}\n")
        f.write(f"  Training samples: {num_train}\n")
        f.write(f"  Testing samples: {num_test}\n")
        
        f.write(f"\nOutput Files:\n")
        f.write(f"  Training data: {train_file_path}\n")
        f.write(f"  Testing data: {test_file_path}\n")
        f.write(f"  Split indices: {indices_file_path}\n")
        f.write(f"  Summary: {summary_file_path}\n")
    
    logger.info(f"Data split completed successfully!")
    logger.info(f"Training data: {train_file_path}")
    logger.info(f"Testing data: {test_file_path}")
    logger.info(f"Summary: {summary_file_path}")
    
    return str(train_file_path), str(test_file_path)

def verify_split(train_file: str, test_file: str):
    """
    Verify that the split data files are valid and contain the expected data
    
    Args:
        train_file: Path to training data file
        test_file: Path to testing data file
    """
    logger.info("Verifying split data files...")
    
    # Verify training file
    with h5py.File(train_file, 'r') as f:
        train_ue = f['ue_positions'][:]
        train_csi = f['channel_responses'][:]
        train_bs = f['bs_position'][:]
        train_attrs = dict(f.attrs)
        
        logger.info(f"Training file verification:")
        logger.info(f"  UE positions: {train_ue.shape}")
        logger.info(f"  Channel responses: {train_csi.shape}")
        logger.info(f"  BS position: {train_bs.shape}")
        logger.info(f"  Split type: {train_attrs.get('split_type', 'unknown')}")
        logger.info(f"  Num samples: {train_attrs.get('num_samples', 'unknown')}")
    
    # Verify testing file
    with h5py.File(test_file, 'r') as f:
        test_ue = f['ue_positions'][:]
        test_csi = f['channel_responses'][:]
        test_bs = f['bs_position'][:]
        test_attrs = dict(f.attrs)
        
        logger.info(f"Testing file verification:")
        logger.info(f"  UE positions: {test_ue.shape}")
        logger.info(f"  Channel responses: {test_csi.shape}")
        logger.info(f"  BS position: {test_bs.shape}")
        logger.info(f"  Split type: {test_attrs.get('split_type', 'unknown')}")
        logger.info(f"  Num samples: {test_attrs.get('num_samples', 'unknown')}")
    
    # Verify no overlap
    total_train = len(train_ue)
    total_test = len(test_ue)
    logger.info(f"Total samples: {total_train + total_test}")
    logger.info(f"Split verification: {total_train} train + {total_test} test = {total_train + total_test}")
    
    logger.info("Data verification completed successfully!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Prepare Sionna simulation data for training')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to Sionna simulation data HDF5 file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for split datasets')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training data ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits (default: 42)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify the split data files after creation')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.data):
        logger.error(f"Input data file not found: {args.data}")
        sys.exit(1)
    
    try:
        # Split the data
        train_file, test_file = split_data(
            args.data, args.output, args.train_ratio, args.seed
        )
        
        # Verify if requested
        if args.verify:
            verify_split(train_file, test_file)
        
        logger.info("Data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
