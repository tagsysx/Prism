#!/usr/bin/env python3
"""
Test to reproduce the error with multiple antennas (64) like in real training.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import traceback
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(lineno)d:%(message)s')

def test_multi_antenna_scenario():
    """Test the exact scenario that fails in training: 64 antennas."""
    print("🔍 Testing multi-antenna scenario (64 antennas)...")
    
    try:
        from prism.networks.prism_network import PrismNetwork
        from prism.ray_tracer import DiscreteRayTracer  
        from prism.training_interface import PrismTrainingInterface
        
        print("✓ Successfully imported all components")
        
        # Create model with EXACT same parameters as real training
        prism_network = PrismNetwork(
            num_subcarriers=408,  # Same as real training
            num_ue_antennas=4,    # Same as real training  
            num_bs_antennas=64,   # Same as real training - THIS IS THE KEY DIFFERENCE
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
        print("✓ Created PrismNetwork with 64 antennas")
        
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
        
        # Create data with EXACT same shapes as real training
        batch_size = 1  # Start small but with 64 antennas
        num_ue = 1
        num_bs_antennas = 64  # THIS IS THE KEY - 64 antennas like real training
        
        ue_positions = torch.randn(batch_size, num_ue, 3)
        bs_position = torch.randn(batch_size, 3) 
        antenna_indices = torch.randint(0, 64, (batch_size, num_bs_antennas))  # 64 antennas
        
        print(f"📊 Test data shapes (matching real training):")
        print(f"   ue_positions: {ue_positions.shape}")
        print(f"   bs_position: {bs_position.shape}")
        print(f"   antenna_indices: {antenna_indices.shape}")
        print(f"   num_bs_antennas: {num_bs_antennas}")
        
        # Forward pass - this should reproduce the error
        print("🔄 Running forward pass with 64 antennas...")
        outputs = training_interface(
            ue_positions=ue_positions,
            bs_position=bs_position,
            antenna_indices=antenna_indices
        )
        print("✅ Forward pass completed successfully!")
        
        print(f"📋 Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
            else:
                print(f"   {key}: {type(value)}")
        
        print("\n🎉 Multi-antenna test passed! The issue is not in antenna count.")
        return True
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_antenna_scenario()
    exit(0 if success else 1)
