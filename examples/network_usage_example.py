#!/usr/bin/env python3
"""
Example script demonstrating the usage of Prism networks.

This script shows how to:
1. Create and configure the individual network components
2. Use the integrated PrismNetwork
3. Process sample data through the networks
4. Analyze network outputs and performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path to import prism modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism.networks import (
    AttenuationNetwork,
    AttenuationDecoder,
    AntennaEmbeddingCodebook,
    AntennaNetwork,
    RadianceNetwork,
    PrismNetwork,
    PrismNetworkConfig
)


def create_sample_data(batch_size=4, num_voxels=100, num_antennas=8):
    """Create sample data for testing the networks."""
    
    # Sample voxel positions in 3D space
    sampled_positions = torch.randn(batch_size, num_voxels, 3)
    
    # UE positions
    ue_positions = torch.randn(batch_size, 3)
    
    # Viewing directions (normalized)
    view_directions = torch.randn(batch_size, 3)
    view_directions = F.normalize(view_directions, p=2, dim=-1)
    
    # Antenna indices
    antenna_indices = torch.randint(0, num_antennas, (batch_size,))
    
    return {
        'sampled_positions': sampled_positions,
        'ue_positions': ue_positions,
        'view_directions': view_directions,
        'antenna_indices': antenna_indices
    }


def test_individual_networks():
    """Test individual network components."""
    print("Testing individual network components...")
    
    batch_size = 4
    feature_dim = 128
    num_ue_antennas = 4
    num_subcarriers = 64
    num_bs_antennas = 64
    antenna_embedding_dim = 64
    
    # 1. Test AttenuationNetwork
    print("\n1. Testing AttenuationNetwork...")
    atten_net = AttenuationNetwork(
        input_dim=63,  # IPE-encoded 3D position
        output_dim=feature_dim,
        complex_output=True
    )
    
    # Create sample input (IPE-encoded positions)
    sample_positions = torch.randn(batch_size, 63)
    features = atten_net(sample_positions)
    print(f"   Input shape: {sample_positions.shape}")
    print(f"   Output shape: {features.shape}")
    print(f"   Output is complex: {features.is_complex()}")
    
    # 2. Test AttenuationDecoder
    print("\n2. Testing AttenuationDecoder...")
    atten_decoder = AttenuationDecoder(
        feature_dim=feature_dim,
        num_ue_antennas=num_ue_antennas,
        num_subcarriers=num_subcarriers,
        complex_output=True
    )
    
    # Convert complex features to real for the decoder
    if features.is_complex():
        features_real = torch.abs(features)
    else:
        features_real = features
    
    attenuation_factors = atten_decoder(features_real)
    print(f"   Input shape: {features.shape}")
    print(f"   Output shape: {attenuation_factors.shape}")
    print(f"   Output is complex: {attenuation_factors.is_complex()}")
    
    # 3. Test AntennaEmbeddingCodebook
    print("\n3. Testing AntennaEmbeddingCodebook...")
    antenna_codebook = AntennaEmbeddingCodebook(
        num_bs_antennas=num_bs_antennas,
        embedding_dim=antenna_embedding_dim
    )
    
    antenna_indices = torch.randint(0, num_bs_antennas, (batch_size,))
    antenna_embeddings = antenna_codebook(antenna_indices)
    print(f"   Antenna indices: {antenna_indices}")
    print(f"   Embedding shape: {antenna_embeddings.shape}")
    print(f"   Total parameters: {antenna_codebook.get_total_parameters()}")
    
    # 4. Test AntennaNetwork
    print("\n4. Testing AntennaNetwork...")
    antenna_net = AntennaNetwork(
        antenna_embedding_dim=antenna_embedding_dim,
        azimuth_divisions=16,
        elevation_divisions=8
    )
    
    directional_importance = antenna_net(antenna_embeddings)
    print(f"   Input shape: {antenna_embeddings.shape}")
    print(f"   Output shape: {directional_importance.shape}")
    
    # Get top-K directions
    top_k_indices, top_k_importance = antenna_net.get_top_k_directions(directional_importance, k=8)
    print(f"   Top-K indices shape: {top_k_indices.shape}")
    print(f"   Top-K importance shape: {top_k_importance.shape}")
    
    # 5. Test RadianceNetwork
    print("\n5. Testing RadianceNetwork...")
    radiance_net = RadianceNetwork(
        ue_position_dim=63,
        view_direction_dim=63,
        feature_dim=feature_dim,
        antenna_embedding_dim=antenna_embedding_dim,
        num_ue_antennas=num_ue_antennas,
        num_subcarriers=num_subcarriers,
        complex_output=True
    )
    
    # Create sample inputs
    ue_positions = torch.randn(batch_size, 63)
    view_directions = torch.randn(batch_size, 63)
    
    # Convert complex features to real for the radiance network
    if features.is_complex():
        features_real = torch.abs(features)
    else:
        features_real = features
    
    radiation_factors = radiance_net(ue_positions, view_directions, features_real, antenna_embeddings)
    print(f"   UE positions shape: {ue_positions.shape}")
    print(f"   View directions shape: {view_directions.shape}")
    print(f"   Features shape: {features.shape}")
    print(f"   Antenna embeddings shape: {antenna_embeddings.shape}")
    print(f"   Output shape: {radiation_factors.shape}")
    print(f"   Output is complex: {radiation_factors.is_complex()}")
    
    print("\nIndividual network tests completed successfully!")


def test_integrated_network():
    """Test the integrated PrismNetwork."""
    print("\nTesting integrated PrismNetwork...")
    
    # Create configuration
    config = PrismNetworkConfig(
        num_subcarriers=64,
        num_ue_antennas=4,
        num_bs_antennas=64,
        feature_dim=128,
        antenna_embedding_dim=64,
        azimuth_divisions=16,
        elevation_divisions=8,
        top_k_directions=32,
        complex_output=True
    )
    
    # Create the integrated network
    prism_net = PrismNetwork(**config.to_dict())
    
    # Get network information
    network_info = prism_net.get_network_info()
    print(f"Network configuration:")
    for key, value in network_info.items():
        print(f"   {key}: {value}")
    
    # Create sample data
    sample_data = create_sample_data(batch_size=2, num_voxels=50, num_antennas=64)
    
    # Forward pass
    print(f"\nInput data shapes:")
    for key, value in sample_data.items():
        print(f"   {key}: {value.shape}")
    
    # Process through network
    outputs = prism_net(
        sampled_positions=sample_data['sampled_positions'],
        ue_positions=sample_data['ue_positions'],
        view_directions=sample_data['view_directions'],
        antenna_indices=sample_data['antenna_indices'],
        return_intermediates=True
    )
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
            if value.is_complex():
                print(f"     (complex tensor)")
        else:
            print(f"   {key}: {type(value)}")
    
    print("\nIntegrated network test completed successfully!")


def test_network_training():
    """Test network training with a simple loss function."""
    print("\nTesting network training...")
    
    # Create a smaller network for training test
    config = PrismNetworkConfig(
        num_subcarriers=16,
        num_ue_antennas=2,
        num_bs_antennas=8,
        feature_dim=64,
        antenna_embedding_dim=32,
        azimuth_divisions=8,
        elevation_divisions=4,
        top_k_directions=8,
        complex_output=True
    )
    
    prism_net = PrismNetwork(**config.to_dict())
    
    # Create optimizer
    optimizer = torch.optim.Adam(prism_net.parameters(), lr=0.001)
    
    # Create sample data
    sample_data = create_sample_data(batch_size=2, num_voxels=20, num_antennas=8)
    
    # Training loop
    num_epochs = 5
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = prism_net(
            sampled_positions=sample_data['sampled_positions'],
            ue_positions=sample_data['ue_positions'],
            view_directions=sample_data['view_directions'],
            antenna_indices=sample_data['antenna_indices']
        )
        
        # Simple loss function (example)
        # In practice, you would use a more sophisticated loss based on your specific requirements
        loss = torch.mean(torch.abs(outputs['attenuation_factors'])) + \
               torch.mean(torch.abs(outputs['radiation_factors']))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    print("Training test completed successfully!")


def main():
    """Main function to run all tests."""
    print("Prism Network Usage Example")
    print("=" * 50)
    
    try:
        # Test individual networks
        test_individual_networks()
        
        # Test integrated network
        test_integrated_network()
        
        # Test training
        test_network_training()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("The Prism networks are working correctly.")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
