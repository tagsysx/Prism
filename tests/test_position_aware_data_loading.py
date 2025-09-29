"""
Test script for position-aware data loading functionality.

This script tests the load_and_split_by_position_and_csi function and
PositionPairBatchSampler to ensure they work correctly.
"""

import os
import sys
import numpy as np
import torch
import h5py
import tempfile
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism.data_utils import (
    load_and_split_by_position_and_csi,
    PositionPairBatchSampler,
    create_position_aware_dataloader
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_dataset(file_path: str, num_positions: int = 20, num_bs_antennas: int = 2, num_ue_antennas: int = 4, num_subcarriers: int = 64):
    """
    Create a test HDF5 dataset for testing position-aware data loading.
    
    Args:
        file_path: Path to save the test dataset
        num_positions: Number of position pairs
        num_bs_antennas: Number of BS antennas
        num_ue_antennas: Number of UE antennas
        num_subcarriers: Number of subcarriers
    """
    logger.info(f"Creating test dataset: {file_path}")
    
    # Create test data
    ue_positions = np.random.randn(num_positions, 3) * 100  # Random UE positions
    bs_positions = np.random.randn(num_positions, 3) * 10   # Random BS positions
    
    # Create complex CSI data
    csi_real = np.random.randn(num_positions, num_bs_antennas, num_ue_antennas, num_subcarriers)
    csi_imag = np.random.randn(num_positions, num_bs_antennas, num_ue_antennas, num_subcarriers)
    csi_data = csi_real + 1j * csi_imag
    
    # Create HDF5 file
    with h5py.File(file_path, 'w') as f:
        # Create data group
        data_group = f.create_group('data')
        data_group.create_dataset('ue_positions', data=ue_positions)
        data_group.create_dataset('bs_positions', data=bs_positions)
        data_group.create_dataset('csi', data=csi_data)
        
        # Add metadata
        f.attrs['dataset_name'] = 'test_position_aware_dataset'
        f.attrs['num_positions'] = num_positions
        f.attrs['num_bs_antennas'] = num_bs_antennas
        f.attrs['num_ue_antennas'] = num_ue_antennas
        f.attrs['num_subcarriers'] = num_subcarriers
    
    logger.info(f"✅ Test dataset created successfully")
    logger.info(f"   Positions: {num_positions}")
    logger.info(f"   BS antennas: {num_bs_antennas}")
    logger.info(f"   UE antennas: {num_ue_antennas}")
    logger.info(f"   Subcarriers: {num_subcarriers}")
    logger.info(f"   Total antenna combinations: {num_positions * num_bs_antennas * num_ue_antennas}")


def test_position_aware_data_loading():
    """Test the position-aware data loading functionality."""
    logger.info("🧪 Testing position-aware data loading...")
    
    # Create temporary test dataset
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        test_dataset_path = tmp_file.name
    
    try:
        # Create test dataset
        create_test_dataset(test_dataset_path, num_positions=20, num_bs_antennas=2, num_ue_antennas=4, num_subcarriers=64)
        
        # Test train data loading
        logger.info("\n📊 Testing train data loading...")
        bs_positions_train, ue_positions_train, bs_ant_indices_train, ue_ant_indices_train, csi_data_train, metadata_train = load_and_split_by_position_and_csi(
            dataset_path=test_dataset_path,
            train_ratio=0.8,
            test_ratio=0.2,
            random_seed=42,
            mode='train'
        )
        
        # Test test data loading
        logger.info("\n📊 Testing test data loading...")
        bs_positions_test, ue_positions_test, bs_ant_indices_test, ue_ant_indices_test, csi_data_test, metadata_test = load_and_split_by_position_and_csi(
            dataset_path=test_dataset_path,
            train_ratio=0.8,
            test_ratio=0.2,
            random_seed=42,
            mode='test'
        )
        
        # Verify data shapes and properties
        logger.info("\n🔍 Verifying data properties...")
        
        # Check train data
        expected_train_positions = 20 * 0.8  # 16 position pairs
        expected_train_samples = expected_train_positions * 2 * 4  # 16 * 2 * 4 = 128 samples
        
        assert bs_positions_train.shape[0] == expected_train_samples, f"Expected {expected_train_samples} train samples, got {bs_positions_train.shape[0]}"
        assert ue_positions_train.shape[0] == expected_train_samples, f"Expected {expected_train_samples} train samples, got {ue_positions_train.shape[0]}"
        assert csi_data_train.shape[0] == expected_train_samples, f"Expected {expected_train_samples} train samples, got {csi_data_train.shape[0]}"
        assert csi_data_train.shape[1] == 64, f"Expected 64 subcarriers, got {csi_data_train.shape[1]}"
        
        # Check test data
        expected_test_positions = 20 * 0.2  # 4 position pairs
        expected_test_samples = expected_test_positions * 2 * 4  # 4 * 2 * 4 = 32 samples
        
        assert bs_positions_test.shape[0] == expected_test_samples, f"Expected {expected_test_samples} test samples, got {bs_positions_test.shape[0]}"
        assert ue_positions_test.shape[0] == expected_test_samples, f"Expected {expected_test_samples} test samples, got {ue_positions_test.shape[0]}"
        assert csi_data_test.shape[0] == expected_test_samples, f"Expected {expected_test_samples} test samples, got {csi_data_test.shape[0]}"
        
        # Check metadata
        assert 'position_pair_grouping' in metadata_train, "Missing position_pair_grouping in train metadata"
        assert 'position_pair_grouping' in metadata_test, "Missing position_pair_grouping in test metadata"
        
        position_grouping_train = metadata_train['position_pair_grouping']
        position_grouping_test = metadata_test['position_pair_grouping']
        
        assert position_grouping_train['enabled'] == True, "Position grouping not enabled in train data"
        assert position_grouping_test['enabled'] == True, "Position grouping not enabled in test data"
        assert position_grouping_train['num_groups'] == expected_train_positions, f"Expected {expected_train_positions} train groups, got {position_grouping_train['num_groups']}"
        assert position_grouping_test['num_groups'] == expected_test_positions, f"Expected {expected_test_positions} test groups, got {position_grouping_test['num_groups']}"
        
        logger.info("✅ Position-aware data loading test passed!")
        
        return metadata_train, metadata_test
        
    finally:
        # Clean up temporary file
        if os.path.exists(test_dataset_path):
            os.unlink(test_dataset_path)


def test_position_pair_batch_sampler():
    """Test the PositionPairBatchSampler functionality."""
    logger.info("\n🧪 Testing PositionPairBatchSampler...")
    
    # Create test position pair indices
    # Simulate 4 position pairs, each with 8 antenna combinations (2 BS × 4 UE)
    position_pair_indices = []
    for pos_idx in range(4):
        for _ in range(8):  # 8 antenna combinations per position pair
            position_pair_indices.append(pos_idx)
    
    logger.info(f"Created test position pair indices: {len(position_pair_indices)} samples")
    logger.info(f"Position pair distribution: {np.bincount(position_pair_indices)}")
    
    # Test batch sampler
    batch_sampler = PositionPairBatchSampler(
        position_pair_indices=position_pair_indices,
        batch_size=2,  # 2 position pairs per batch
        shuffle=True,
        drop_last=False
    )
    
    # Test batch generation
    batches = list(batch_sampler)
    logger.info(f"Generated {len(batches)} batches")
    
    # Verify batch properties
    for i, batch in enumerate(batches):
        logger.info(f"Batch {i+1}: {len(batch)} samples")
        
        # Check that samples in each batch belong to the same position pairs
        batch_position_pairs = set()
        for sample_idx in batch:
            batch_position_pairs.add(position_pair_indices[sample_idx])
        
        logger.info(f"  Position pairs in batch: {sorted(batch_position_pairs)}")
        
        # Verify that each batch contains complete position pairs
        for pos_pair in batch_position_pairs:
            pos_pair_samples = [idx for idx in batch if position_pair_indices[idx] == pos_pair]
            expected_samples_per_pair = 8  # 2 BS × 4 UE
            assert len(pos_pair_samples) == expected_samples_per_pair, f"Position pair {pos_pair} should have {expected_samples_per_pair} samples, got {len(pos_pair_samples)}"
    
    logger.info("✅ PositionPairBatchSampler test passed!")


def test_position_aware_dataloader():
    """Test the create_position_aware_dataloader function."""
    logger.info("\n🧪 Testing create_position_aware_dataloader...")
    
    # Create a simple test dataset
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size
            self.data = torch.randn(size, 10)  # Random data
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # Create test dataset and position pair indices
    dataset_size = 32  # 4 position pairs × 8 antenna combinations
    test_dataset = TestDataset(dataset_size)
    
    position_pair_indices = []
    for pos_idx in range(4):
        for _ in range(8):
            position_pair_indices.append(pos_idx)
    
    # Create position-aware dataloader
    dataloader = create_position_aware_dataloader(
        dataset=test_dataset,
        position_pair_indices=position_pair_indices,
        batch_size=2,  # 2 position pairs per batch
        shuffle=True,
        drop_last=False
    )
    
    # Test dataloader
    logger.info(f"DataLoader created with {len(dataloader)} batches")
    
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        logger.info(f"Batch {batch_count}: shape {batch.shape}")
        
        # Verify batch size (should be 2 position pairs × 8 samples = 16 samples)
        expected_batch_size = 2 * 8  # 2 position pairs × 8 samples per pair
        assert batch.shape[0] == expected_batch_size, f"Expected batch size {expected_batch_size}, got {batch.shape[0]}"
    
    logger.info(f"Processed {batch_count} batches")
    logger.info("✅ create_position_aware_dataloader test passed!")


def test_position_consistency():
    """Test that samples from the same position pair have consistent positions."""
    logger.info("\n🧪 Testing position consistency...")
    
    # Create temporary test dataset
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        test_dataset_path = tmp_file.name
    
    try:
        # Create test dataset with known positions
        num_positions = 6
        num_bs_antennas = 2
        num_ue_antennas = 3
        
        # Create deterministic positions
        ue_positions = np.array([[i, i+1, i+2] for i in range(num_positions)], dtype=np.float32)
        bs_positions = np.array([[i*10, i*10+1, i*10+2] for i in range(num_positions)], dtype=np.float32)
        
        # Create CSI data
        csi_real = np.random.randn(num_positions, num_bs_antennas, num_ue_antennas, 64)
        csi_imag = np.random.randn(num_positions, num_bs_antennas, num_ue_antennas, 64)
        csi_data = csi_real + 1j * csi_imag
        
        # Save test dataset
        with h5py.File(test_dataset_path, 'w') as f:
            data_group = f.create_group('data')
            data_group.create_dataset('ue_positions', data=ue_positions)
            data_group.create_dataset('bs_positions', data=bs_positions)
            data_group.create_dataset('csi', data=csi_data)
        
        # Load data
        bs_positions_loaded, ue_positions_loaded, bs_ant_indices, ue_ant_indices, csi_data_loaded, metadata = load_and_split_by_position_and_csi(
            dataset_path=test_dataset_path,
            train_ratio=1.0,  # Use all data
            test_ratio=0.0,
            random_seed=42,
            mode='train'
        )
        
        # Get position pair grouping information
        position_pair_grouping = metadata['position_pair_grouping']
        group_indices = position_pair_grouping['group_indices']
        
        # Verify position consistency within each group
        for group_idx in range(num_positions):
            # Find all samples belonging to this group
            group_samples = [i for i, g_idx in enumerate(group_indices) if g_idx == group_idx]
            
            if len(group_samples) > 0:
                # Check that all samples in the group have the same UE and BS positions
                group_ue_positions = ue_positions_loaded[group_samples]
                group_bs_positions = bs_positions_loaded[group_samples]
                
                # All UE positions in the group should be identical
                ue_pos_unique = torch.unique(group_ue_positions, dim=0)
                assert ue_pos_unique.shape[0] == 1, f"Group {group_idx} has inconsistent UE positions"
                
                # All BS positions in the group should be identical
                bs_pos_unique = torch.unique(group_bs_positions, dim=0)
                assert bs_pos_unique.shape[0] == 1, f"Group {group_idx} has inconsistent BS positions"
                
                logger.info(f"Group {group_idx}: {len(group_samples)} samples, UE pos: {ue_pos_unique[0].numpy()}, BS pos: {bs_pos_unique[0].numpy()}")
        
        logger.info("✅ Position consistency test passed!")
        
    finally:
        # Clean up temporary file
        if os.path.exists(test_dataset_path):
            os.unlink(test_dataset_path)


def main():
    """Run all tests."""
    logger.info("🚀 Starting position-aware data loading tests...")
    
    try:
        # Test 1: Position-aware data loading
        metadata_train, metadata_test = test_position_aware_data_loading()
        
        # Test 2: PositionPairBatchSampler
        test_position_pair_batch_sampler()
        
        # Test 3: Position-aware DataLoader
        test_position_aware_dataloader()
        
        # Test 4: Position consistency
        test_position_consistency()
        
        logger.info("\n🎉 All tests passed successfully!")
        logger.info("✅ Position-aware data loading functionality is working correctly")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
