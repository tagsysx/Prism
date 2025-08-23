#!/usr/bin/env python3
"""
Test using real training data and pipeline to reproduce the 'float' object is not iterable error.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import traceback
import logging
import h5py
import numpy as np

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(lineno)d:%(message)s')

def test_real_training_data():
    """Test with real training data to reproduce the error."""
    print("🔍 Testing with real training data...")
    
    try:
        from prism.networks.prism_network import PrismNetwork
        from prism.ray_tracer import DiscreteRayTracer  
        from prism.training_interface import PrismTrainingInterface
        
        print("✓ Successfully imported all components")
        
        # Check if training data exists
        data_path = "results/complete_pipeline/data_split/train_data.h5"
        if not Path(data_path).exists():
            print(f"❌ Training data not found at: {data_path}")
            print("Please run the data generation pipeline first.")
            return False
        
        print(f"✓ Found training data at: {data_path}")
        
        # Load a small sample of real training data
        with h5py.File(data_path, 'r') as f:
            print("📊 Available datasets:")
            for key in f.keys():
                if hasattr(f[key], 'shape'):
                    print(f"   {key}: {f[key].shape} - {f[key].dtype}")
                else:
                    print(f"   {key}: {type(f[key])}")
            
            # Load a small sample of real training data
            # The issue: ue_positions is (80, 3) but we need (batch_size, num_ue, 3)
            # Let's create proper 3D positions for testing
            ue_positions = f['ue_positions'][:1, :]      # (1, 3) -> single UE position
            bs_position = f['bs_position'][:]             # (3,) -> base station position
            antenna_indices = f['antenna_indices'][:]     # (64,) -> all antennas
            
            print(f"\n📊 Data shapes after loading:")
            print(f"   ue_positions: {ue_positions.shape}")
            print(f"   bs_position: {bs_position.shape}")
            print(f"   antenna_indices: {antenna_indices.shape}")
            
            # Create proper 3D UE positions for testing
            # We need (batch_size, num_ue, 3) where num_ue=3
            ue_positions_3d = np.zeros((1, 3, 3))  # (1, 3, 3) - 1 batch, 3 UEs, 3 coordinates
            ue_positions_3d[0, 0, :] = ue_positions[0, :]  # First UE gets the loaded position
            ue_positions_3d[0, 1, :] = ue_positions[0, :] + np.array([10, 10, 0])  # Second UE offset
            ue_positions_3d[0, 2, :] = ue_positions[0, :] + np.array([20, 20, 0])  # Third UE offset
            
            print(f"   ue_positions_3d: {ue_positions_3d.shape}")
            print(f"   Sample UE positions:")
            for u in range(3):
                print(f"     UE {u}: {ue_positions_3d[0, u, :]}")
            
            # Convert to tensors
            ue_positions_tensor = torch.from_numpy(ue_positions_3d).float()  # (1, 3, 3)
            bs_positions_tensor = torch.from_numpy(bs_position).float()   # (3,) -> needs to be (1, 3)
            antenna_indices_tensor = torch.from_numpy(antenna_indices).long()  # (64,) -> needs to be (1, 64)
            
            print(f"\n📊 Loaded real data shapes:")
            print(f"   ue_positions_tensor: {ue_positions_tensor.shape}")
            print(f"   bs_position: {bs_positions_tensor.shape}")
            print(f"   antenna_indices: {antenna_indices_tensor.shape}")
            
            # Check data types and values
            print(f"\n🔍 Data validation:")
            print(f"   ue_positions dtype: {ue_positions_tensor.dtype}, min: {ue_positions_tensor.min():.3f}, max: {ue_positions_tensor.max():.3f}")
            print(f"   bs_position dtype: {bs_position.dtype}, min: {bs_position.min():.3f}, max: {bs_position.max():.3f}")
            print(f"   antenna_indices dtype: {antenna_indices.dtype}, min: {antenna_indices.min()}, max: {antenna_indices.max()}")
            
            # Check for any NaN or infinite values
            print(f"\n🔍 Data quality check:")
            print(f"   ue_positions has NaN: {np.isnan(ue_positions_tensor).any()}")
            print(f"   ue_positions has inf: {np.isinf(ue_positions_tensor).any()}")
            print(f"   bs_position has NaN: {np.isnan(bs_position).any()}")
            print(f"   bs_position has inf: {np.isinf(bs_position).any()}")
            
            # Reshape to match expected format
            bs_positions_tensor = bs_positions_tensor.unsqueeze(0)  # (3,) -> (1, 3)
            antenna_indices_tensor = antenna_indices_tensor.unsqueeze(0)  # (64,) -> (1, 64)
        
        # Create model with EXACT same parameters as real training
        prism_network = PrismNetwork(
            num_subcarriers=408,  # Same as real training
            num_ue_antennas=4,    # Same as real training  
            num_bs_antennas=64,   # Same as real training
            position_dim=3,
            hidden_dim=256,       # Same as real training
            feature_dim=128,      # Same as real training
            antenna_embedding_dim=64,  # Same as real training
            use_antenna_codebook=True,
            use_ipe_encoding=True,
            azimuth_divisions=36,  # Same as real training
            elevation_divisions=18, # Same as real training
            top_k_directions=32,
            complex_output=True
        )
        print("✓ Created PrismNetwork with real training parameters")
        
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=36,  # Same as real training
            elevation_divisions=18, # Same as real training
            max_ray_length=200.0,
            scene_size=200.0,
            device='cpu'
        )
        print("✓ Created DiscreteRayTracer")
        
        training_interface = PrismTrainingInterface(
            prism_network=prism_network,
            ray_tracer=ray_tracer,
            num_sampling_points=64,  # Same as real training
            subcarrier_sampling_ratio=0.3,
            checkpoint_dir="test_checkpoints"
        )
        print("✓ Created PrismTrainingInterface")
        
        # Convert numpy arrays to torch tensors and reshape to expected format
        ue_positions_tensor = torch.from_numpy(ue_positions_3d).float()  # (1, 3, 3)
        bs_positions_tensor = torch.from_numpy(bs_position).float()   # (3,) -> needs to be (1, 3)
        antenna_indices_tensor = torch.from_numpy(antenna_indices).long()  # (64,) -> needs to be (1, 64)
        
        # Reshape to match expected format
        bs_positions_tensor = bs_positions_tensor.unsqueeze(0)  # (3,) -> (1, 3)
        antenna_indices_tensor = antenna_indices_tensor.unsqueeze(0)  # (64,) -> (1, 64)
        
        print(f"\n🔄 Running forward pass with REAL training data...")
        print(f"   Input tensor shapes:")
        print(f"     ue_positions_tensor: {ue_positions_tensor.shape} - {ue_positions_tensor.dtype}")
        print(f"     bs_positions_tensor: {bs_positions_tensor.shape} - {bs_positions_tensor.dtype}")
        print(f"     antenna_indices_tensor: {antenna_indices_tensor.shape} - {antenna_indices_tensor.dtype}")
        
        # Forward pass - this should reproduce the error
        outputs = training_interface(
            ue_positions=ue_positions_tensor,
            bs_position=bs_positions_tensor,
            antenna_indices=antenna_indices_tensor
        )
        print("✅ Forward pass completed successfully!")
        
        print(f"\n📋 Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape} - {value.dtype}")
            else:
                print(f"   {key}: {type(value)}")
        
        # Test loss computation with real targets
        print(f"\n💰 Testing loss computation with real targets...")
        csi_predictions = outputs['csi_predictions']
        csi_targets_tensor = torch.from_numpy(channel_responses).float() # Changed from csi_targets to channel_responses
        
        print(f"   csi_predictions: {csi_predictions.shape} - {csi_predictions.dtype}")
        print(f"   csi_targets_tensor: {csi_targets_tensor.shape} - {csi_targets_tensor.dtype}")
        
        def test_loss_fn(pred, target):
            return torch.nn.functional.mse_loss(pred.real, target.real) + torch.nn.functional.mse_loss(pred.imag, target.imag)
        
        loss = training_interface.compute_loss(csi_predictions, csi_targets_tensor, test_loss_fn)
        print(f"✅ Loss computation successful: {loss.item():.6f}")
        
        print("\n🎉 Real training data test passed! The error is not reproducible with this data.")
        return True
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_training_data()
    exit(0 if success else 1)
