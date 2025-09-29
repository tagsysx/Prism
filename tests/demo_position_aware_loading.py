"""
Demo script showing the difference between standard and position-aware data loading.

This script demonstrates how position-aware loading ensures that all antenna
combinations for the same position pair are grouped together in batches.
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
    load_and_split_by_csi,
    load_and_split_by_position_and_csi,
    create_position_aware_dataloader
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_dataset(file_path: str):
    """Create a small demo dataset for visualization."""
    logger.info(f"Creating demo dataset: {file_path}")
    
    # Create 4 position pairs with distinct positions
    num_positions = 4
    num_bs_antennas = 2
    num_ue_antennas = 3
    num_subcarriers = 16  # Smaller for demo
    
    # Create distinct positions for each position pair
    ue_positions = np.array([
        [0, 0, 0],    # Position pair 0
        [10, 10, 10], # Position pair 1
        [20, 20, 20], # Position pair 2
        [30, 30, 30]  # Position pair 3
    ], dtype=np.float32)
    
    bs_positions = np.array([
        [0, 0, 0],    # Position pair 0
        [5, 5, 5],    # Position pair 1
        [15, 15, 15], # Position pair 2
        [25, 25, 25]  # Position pair 3
    ], dtype=np.float32)
    
    # Create CSI data
    csi_real = np.random.randn(num_positions, num_bs_antennas, num_ue_antennas, num_subcarriers)
    csi_imag = np.random.randn(num_positions, num_bs_antennas, num_ue_antennas, num_subcarriers)
    csi_data = csi_real + 1j * csi_imag
    
    # Create HDF5 file
    with h5py.File(file_path, 'w') as f:
        data_group = f.create_group('data')
        data_group.create_dataset('ue_positions', data=ue_positions)
        data_group.create_dataset('bs_positions', data=bs_positions)
        data_group.create_dataset('csi', data=csi_data)
        
        f.attrs['dataset_name'] = 'demo_position_dataset'
    
    logger.info(f"✅ Demo dataset created:")
    logger.info(f"   Position pairs: {num_positions}")
    logger.info(f"   BS antennas: {num_bs_antennas}")
    logger.info(f"   UE antennas: {num_ue_antennas}")
    logger.info(f"   Total samples: {num_positions * num_bs_antennas * num_ue_antennas}")


def demo_standard_data_loading(dataset_path: str):
    """Demonstrate standard data loading behavior."""
    logger.info("\n📊 Standard Data Loading Demo:")
    logger.info("=" * 50)
    
    # Load data using standard method
    bs_positions, ue_positions, bs_ant_indices, ue_ant_indices, csi_data, metadata = load_and_split_by_csi(
        dataset_path=dataset_path,
        train_ratio=1.0,
        test_ratio=0.0,
        random_seed=42,
        mode='train'
    )
    
    logger.info(f"Loaded {len(ue_positions)} samples")
    logger.info("\nSample distribution (first 12 samples):")
    logger.info("Sample | UE Position    | BS Position    | BS Ant | UE Ant")
    logger.info("-" * 65)
    
    for i in range(min(12, len(ue_positions))):
        ue_pos = ue_positions[i].numpy()
        bs_pos = bs_positions[i].numpy()
        bs_ant = bs_ant_indices[i].item()
        ue_ant = ue_ant_indices[i].item()
        
        logger.info(f"{i:6d} | [{ue_pos[0]:3.0f},{ue_pos[1]:3.0f},{ue_pos[2]:3.0f}] | [{bs_pos[0]:3.0f},{bs_pos[1]:3.0f},{bs_pos[2]:3.0f}] | {bs_ant:6d} | {ue_ant:6d}")
    
    # Show how samples are mixed
    logger.info("\n🔍 Analysis:")
    unique_ue_positions = torch.unique(ue_positions, dim=0)
    logger.info(f"Unique UE positions: {len(unique_ue_positions)}")
    logger.info("Note: Samples from different position pairs are mixed together!")


def demo_position_aware_data_loading(dataset_path: str):
    """Demonstrate position-aware data loading behavior."""
    logger.info("\n📊 Position-Aware Data Loading Demo:")
    logger.info("=" * 50)
    
    # Load data using position-aware method
    bs_positions, ue_positions, bs_ant_indices, ue_ant_indices, csi_data, metadata = load_and_split_by_position_and_csi(
        dataset_path=dataset_path,
        train_ratio=1.0,
        test_ratio=0.0,
        random_seed=42,
        mode='train'
    )
    
    logger.info(f"Loaded {len(ue_positions)} samples")
    
    # Get position pair grouping information
    position_pair_grouping = metadata['position_pair_grouping']
    group_indices = position_pair_grouping['group_indices']
    
    logger.info(f"\nPosition pair grouping:")
    logger.info(f"  Number of groups: {position_pair_grouping['num_groups']}")
    logger.info(f"  Samples per group: {position_pair_grouping['samples_per_group']}")
    
    logger.info("\nSample distribution (grouped by position pairs):")
    logger.info("Group | Sample | UE Position    | BS Position    | BS Ant | UE Ant")
    logger.info("-" * 75)
    
    # Show samples grouped by position pair
    for group_idx in range(position_pair_grouping['num_groups']):
        group_samples = [i for i, g_idx in enumerate(group_indices) if g_idx == group_idx]
        
        logger.info(f"Group {group_idx}:")
        for sample_idx in group_samples:
            ue_pos = ue_positions[sample_idx].numpy()
            bs_pos = bs_positions[sample_idx].numpy()
            bs_ant = bs_ant_indices[sample_idx].item()
            ue_ant = ue_ant_indices[sample_idx].item()
            
            logger.info(f"       | {sample_idx:6d} | [{ue_pos[0]:3.0f},{ue_pos[1]:3.0f},{ue_pos[2]:3.0f}] | [{bs_pos[0]:3.0f},{bs_pos[1]:3.0f},{bs_pos[2]:3.0f}] | {bs_ant:6d} | {ue_ant:6d}")
    
    logger.info("\n✅ Analysis:")
    logger.info("✅ All samples from the same position pair are grouped together!")
    logger.info("✅ Each group contains all antenna combinations for that position pair!")


def demo_position_aware_batching(dataset_path: str):
    """Demonstrate position-aware batching behavior."""
    logger.info("\n📊 Position-Aware Batching Demo:")
    logger.info("=" * 50)
    
    # Load data using position-aware method
    bs_positions, ue_positions, bs_ant_indices, ue_ant_indices, csi_data, metadata = load_and_split_by_position_and_csi(
        dataset_path=dataset_path,
        train_ratio=1.0,
        test_ratio=0.0,
        random_seed=42,
        mode='train'
    )
    
    # Create a simple dataset
    class DemoDataset(torch.utils.data.Dataset):
        def __init__(self, positions, csi_data):
            self.positions = positions
            self.csi_data = csi_data
        
        def __len__(self):
            return len(self.positions)
        
        def __getitem__(self, idx):
            return {
                'position': self.positions[idx],
                'csi': self.csi_data[idx]
            }
    
    # Create dataset
    dataset = DemoDataset(ue_positions, csi_data)
    
    # Get position pair grouping information
    position_pair_grouping = metadata['position_pair_grouping']
    group_indices = position_pair_grouping['group_indices']
    
    # Create position-aware dataloader with batch_size=2 (2 position pairs per batch)
    dataloader = create_position_aware_dataloader(
        dataset=dataset,
        position_pair_indices=group_indices,
        batch_size=2,  # 2 position pairs per batch
        shuffle=True,
        drop_last=False
    )
    
    logger.info(f"Created DataLoader with {len(dataloader)} batches")
    logger.info(f"Batch size: 2 position pairs = {2 * position_pair_grouping['samples_per_group']} samples per batch")
    
    # Process batches
    logger.info("\nBatch processing:")
    logger.info("Batch | Position Pairs | Sample Indices | UE Positions")
    logger.info("-" * 70)
    
    for batch_idx, batch in enumerate(dataloader):
        batch_positions = batch['position']
        batch_csi = batch['csi']
        
        # Get unique UE positions in this batch
        unique_positions = torch.unique(batch_positions, dim=0)
        
        logger.info(f"{batch_idx:5d} | {len(unique_positions):13d} | {len(batch_positions):13d} | {unique_positions.numpy().tolist()}")
    
    logger.info("\n✅ Analysis:")
    logger.info("✅ Each batch contains complete position pairs!")
    logger.info("✅ No position pair is split across multiple batches!")


def main():
    """Run the demo."""
    logger.info("🚀 Position-Aware Data Loading Demo")
    logger.info("=" * 60)
    
    # Create temporary demo dataset
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        demo_dataset_path = tmp_file.name
    
    try:
        # Create demo dataset
        create_demo_dataset(demo_dataset_path)
        
        # Demo 1: Standard data loading
        demo_standard_data_loading(demo_dataset_path)
        
        # Demo 2: Position-aware data loading
        demo_position_aware_data_loading(demo_dataset_path)
        
        # Demo 3: Position-aware batching
        demo_position_aware_batching(demo_dataset_path)
        
        logger.info("\n🎉 Demo completed successfully!")
        logger.info("✅ Position-aware data loading ensures proper grouping of position pairs!")
        
    finally:
        # Clean up temporary file
        if os.path.exists(demo_dataset_path):
            os.unlink(demo_dataset_path)


if __name__ == "__main__":
    main()
