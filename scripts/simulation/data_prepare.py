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
    
    # Load the original data
    with h5py.File(data_path, 'r') as f:
        # Load all data
        ue_positions = f['ue_positions'][:]
        channel_responses = f['channel_responses'][:]
        bs_position = f['bs_position'][:]
        
        # Load simulation parameters
        sim_params = dict(f['simulation_params'].attrs)
        
        logger.info(f"Original data shape:")
        logger.info(f"  UE positions: {ue_positions.shape}")
        logger.info(f"  Channel responses: {channel_responses.shape}")
        logger.info(f"  BS position: {bs_position.shape}")
    
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
    
    # Save training data
    train_file_path = output_path / 'train_data.h5'
    logger.info(f"Saving training data to {train_file_path}")
    
    with h5py.File(train_file_path, 'w') as f:
        f.create_dataset('ue_positions', data=train_ue_positions)
        f.create_dataset('channel_responses', data=train_channel_responses)
        f.create_dataset('bs_position', data=bs_position)
        
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
        f.write(f"Random seed: {random_seed}\n\n")
        f.write(f"Total samples: {num_samples}\n")
        f.write(f"Training ratio: {train_ratio:.1%}\n")
        f.write(f"Training samples: {num_train}\n")
        f.write(f"Testing samples: {num_test}\n\n")
        f.write(f"Training file: {train_file_path}\n")
        f.write(f"Testing file: {test_file_path}\n")
        f.write(f"Indices file: {indices_file_path}\n")
    
    logger.info("Data split completed successfully!")
    logger.info(f"Training data: {train_file_path}")
    logger.info(f"Testing data: {test_file_path}")
    logger.info(f"Split summary: {summary_file_path}")
    
    return str(train_file_path), str(test_file_path)

def verify_split(train_path: str, test_path: str):
    """Verify that the split data is correct"""
    logger.info("Verifying data split...")
    
    with h5py.File(train_path, 'r') as f_train:
        train_samples = f_train['ue_positions'].shape[0]
        train_attrs = dict(f_train.attrs)
    
    with h5py.File(test_path, 'r') as f_test:
        test_samples = f_test['ue_positions'].shape[0]
        test_attrs = dict(f_test.attrs)
    
    total_split = train_samples + test_samples
    
    logger.info(f"Verification results:")
    logger.info(f"  Training samples: {train_samples}")
    logger.info(f"  Testing samples: {test_samples}")
    logger.info(f"  Total split samples: {total_split}")
    logger.info(f"  Training ratio: {train_samples/total_split:.1%}")
    logger.info(f"  Testing ratio: {test_samples/total_split:.1%}")
    
    # Check if the split is correct
    if abs(train_samples/total_split - 0.8) < 0.01:
        logger.info("✓ Training ratio is approximately 80%")
    else:
        logger.warning("⚠ Training ratio deviates from 80%")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Prepare Sionna data for training/testing')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to original Sionna HDF5 data file')
    parser.add_argument('--output', type=str, default='data/split',
                       help='Output directory for split datasets')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of training data (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify the split data after preparation')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.data):
        logger.error(f"Input data file not found: {args.data}")
        sys.exit(1)
    
    try:
        # Split the data
        train_path, test_path = split_data(
            args.data, 
            args.output, 
            args.train_ratio, 
            args.seed
        )
        
        # Verify if requested
        if args.verify:
            verify_split(train_path, test_path)
        
        logger.info("Data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
