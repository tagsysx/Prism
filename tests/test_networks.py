#!/usr/bin/env python3
"""
Test script for Prism networks.

This script tests that all network components can be imported and created correctly.
"""

import sys
import torch
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all network components can be imported."""
    print("Testing imports...")
    
    try:
        from prism.networks import (
            AttenuationNetwork,
            AttenuationDecoder,
            AntennaEmbeddingCodebook,
            AntennaNetwork,
            RadianceNetwork,
            PrismNetwork
        )
        print("✓ All network classes imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_network_creation():
    """Test that all networks can be created."""
    print("\nTesting network creation...")
    
    try:
        from prism.networks import (
            AttenuationNetwork,
            AttenuationDecoder,
            AntennaEmbeddingCodebook,
            AntennaNetwork,
            RadianceNetwork,
            PrismNetwork
        )
        
        # Test AttenuationNetwork
        atten_net = AttenuationNetwork(
            input_dim=63,
            output_dim=128,
            complex_output=True
        )
        print("✓ AttenuationNetwork created successfully")
        
        # Test AttenuationDecoder
        atten_decoder = AttenuationDecoder(
            feature_dim=128,
            num_ue_antennas=4,
            num_subcarriers=64,
            complex_output=True
        )
        print("✓ AttenuationDecoder created successfully")
        
        # Test AntennaEmbeddingCodebook
        antenna_codebook = AntennaEmbeddingCodebook(
            num_bs_antennas=64,
            embedding_dim=64
        )
        print("✓ AntennaEmbeddingCodebook created successfully")
        
        # Test AntennaNetwork
        ant_net = AntennaNetwork(
            antenna_embedding_dim=64,
            azimuth_divisions=16,
            elevation_divisions=8
        )
        print("✓ AntennaNetwork created successfully")
        
        # Test RadianceNetwork
        radiance_net = RadianceNetwork(
            ue_position_dim=63,
            view_direction_dim=63,
            feature_dim=128,
            antenna_embedding_dim=64,
            num_ue_antennas=4,
            num_subcarriers=64,
            complex_output=True
        )
        print("✓ RadianceNetwork created successfully")
        
        # Test PrismNetwork
        prism_net = PrismNetwork(
            num_subcarriers=64,
            num_ue_antennas=4,
            num_bs_antennas=64,
            feature_dim=128,
            antenna_embedding_dim=64
        )
        print("✓ PrismNetwork created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Network creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test that networks can perform forward passes."""
    print("\nTesting forward passes...")
    
    try:
        from prism.networks import PrismNetwork
        
        # Create a simple PrismNetwork
        prism_net = PrismNetwork(
            num_subcarriers=16,
            num_ue_antennas=2,
            num_bs_antennas=8,
            feature_dim=64,
            antenna_embedding_dim=32,
            azimuth_divisions=8,
            elevation_divisions=4,
            top_k_directions=8
        )
        
        # Create sample data
        batch_size = 2
        num_voxels = 10
        
        sampled_positions = torch.randn(batch_size, num_voxels, 3)
        ue_positions = torch.randn(batch_size, 3)
        view_directions = torch.randn(batch_size, 3)
        antenna_indices = torch.randint(0, 8, (batch_size,))
        
        # Forward pass
        with torch.no_grad():
            outputs = prism_net(
                sampled_positions=sampled_positions,
                ue_positions=ue_positions,
                view_directions=view_directions,
                antenna_indices=antenna_indices
            )
        
        print("✓ Forward pass completed successfully")
        print(f"   Output keys: {list(outputs.keys())}")
        
        # Check output shapes
        expected_shapes = {
            'attenuation_factors': (batch_size, num_voxels, 2, 16),
            'radiation_factors': (batch_size, 2, 16),
            'directional_importance': (batch_size, 8, 4),
            'top_k_directions': (batch_size, 8, 2),
            'top_k_importance': (batch_size, 8)
        }
        
        for key, expected_shape in expected_shapes.items():
            if key in outputs:
                actual_shape = outputs[key].shape
                if actual_shape == expected_shape:
                    print(f"   ✓ {key}: {actual_shape}")
                else:
                    print(f"   ✗ {key}: expected {expected_shape}, got {actual_shape}")
            else:
                print(f"   ✗ {key}: missing from outputs")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_network_info():
    """Test that networks provide information about their configuration."""
    print("\nTesting network information...")
    
    try:
        from prism.networks import PrismNetwork
        
        prism_net = PrismNetwork(
            num_subcarriers=32,
            num_ue_antennas=2,
            num_bs_antennas=16,
            feature_dim=64,
            antenna_embedding_dim=32
        )
        
        # Get network info
        network_info = prism_net.get_network_info()
        config = prism_net.get_config()
        
        print("✓ Network information retrieved successfully")
        print(f"   Total parameters: {network_info['total_parameters']}")
        print(f"   Trainable parameters: {network_info['trainable_parameters']}")
        print(f"   Configuration keys: {list(config.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Network info error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Prism Networks Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_network_creation,
        test_forward_pass,
        test_network_info
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Prism networks are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
