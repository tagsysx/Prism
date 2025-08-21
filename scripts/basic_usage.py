#!/usr/bin/env python3
"""
Basic usage example for Prism: Wideband RF Neural Radiance Fields.
This example demonstrates how to use the Prism model for OFDM scenarios.
"""

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from the prism package
from prism.model import PrismModel, PrismLoss
from prism.utils import create_ofdm_processor, create_mimo_processor

def main():
    """Main function demonstrating basic Prism usage."""
    
    # Configuration for WiFi-like scenario (52 subcarriers)
    config = {
        'model': {
            'num_subcarriers': 52,
            'num_ue_antennas': 2,
            'num_bs_antennas': 4,
            'position_dim': 3,
            'hidden_dim': 256
        },
        'ofdm': {
            'center_frequency': 2.4e9,
            'bandwidth': 20e6,
            'subcarrier_spacing': 312.5e3,
            'cyclic_prefix': 0.25,
            'pilot_density': 0.15,
            'mimo_type': 'spatial_multiplexing',
            'precoding': 'zero_forcing',
            'detection': 'maximum_likelihood'
        }
    }
    
    print("=== Prism: Wideband RF Neural Radiance Fields ===")
    print(f"Configuration: {config['model']['num_subcarriers']} subcarriers")
    print(f"UE antennas: {config['model']['num_ue_antennas']}")
    print(f"BS antennas: {config['model']['num_bs_antennas']}")
    print()
    
    # Create model
    print("Creating Prism model...")
    model = PrismModel(
        num_subcarriers=config['model']['num_subcarriers'],
        num_ue_antennas=config['model']['num_ue_antennas'],
        num_bs_antennas=config['model']['num_bs_antennas'],
        position_dim=config['model']['position_dim'],
        hidden_dim=config['model']['hidden_dim']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Create sample input data
    print("Generating sample input data...")
    batch_size = 16
    
    # Generate random positions in a 10x10x3 meter room
    positions = torch.rand(batch_size, config['model']['position_dim']) * torch.tensor([10.0, 10.0, 3.0])
    
    # Generate random UE antenna features
    ue_antennas = torch.randn(batch_size, config['model']['num_ue_antennas'])
    
    # Generate random BS antenna features
    bs_antennas = torch.randn(batch_size, config['model']['num_bs_antennas'])
    
    # Generate additional RF features
    additional_features = torch.randn(batch_size, 10)
    
    print(f"Input shapes:")
    print(f"  Positions: {positions.shape}")
    print(f"  UE antennas: {ue_antennas.shape}")
    print(f"  BS antennas: {bs_antennas.shape}")
    print(f"  Additional features: {additional_features.shape}")
    print()
    
    # Forward pass
    print("Running forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(positions, ue_antennas, bs_antennas, additional_features)
    
    print(f"Output shapes:")
    print(f"  Subcarrier responses: {outputs['subcarrier_responses'].shape}")
    print(f"  MIMO channel: {outputs['mimo_channel'].shape}")
    print(f"  Attenuation features: {outputs['attenuation_features'].shape}")
    print(f"  Radiance features: {outputs['radiance_features'].shape}")
    print()
    
    # Create OFDM processor
    print("Creating OFDM signal processor...")
    ofdm_processor = create_ofdm_processor(config)
    
    print(f"OFDM parameters:")
    print(f"  Center frequency: {ofdm_processor.center_frequency / 1e9:.2f} GHz")
    print(f"  Bandwidth: {ofdm_processor.bandwidth / 1e6:.1f} MHz")
    print(f"  Subcarrier spacing: {ofdm_processor.subcarrier_spacing / 1e3:.1f} kHz")
    print(f"  Pilot subcarriers: {len(ofdm_processor.pilot_indices)}")
    print(f"  Data subcarriers: {len(ofdm_processor.data_indices)}")
    print()
    
    # Create MIMO processor
    print("Creating MIMO channel processor...")
    mimo_processor = create_mimo_processor(config)
    
    # Generate sample MIMO channel
    mimo_channel = mimo_processor.generate_mimo_channel_matrix(
        config['model']['num_subcarriers'],
        correlation=0.3
    )
    
    print(f"MIMO channel shape: {mimo_channel.shape}")
    print(f"Channel capacity (SNR=10dB): {np.mean(mimo_processor.calculate_channel_capacity(mimo_channel, 10)):.2f} bits/s/Hz")
    print()
    
    # Test loss function
    print("Testing loss function...")
    criterion = PrismLoss(loss_type='mse')
    
    # Generate random targets
    targets = torch.randn(batch_size, config['model']['num_subcarriers'])
    
    # Calculate loss
    loss = criterion(outputs['subcarrier_responses'], targets)
    print(f"Loss value: {loss.item():.6f}")
    print()
    
    # Visualize subcarrier responses
    print("Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot subcarrier responses for first sample
    sample_idx = 0
    subcarrier_indices = np.arange(config['model']['num_subcarriers'])
    
    ax1.plot(subcarrier_indices, outputs['subcarrier_responses'][sample_idx].numpy(), 'b-', linewidth=2, label='Predicted')
    ax1.plot(subcarrier_indices, targets[sample_idx].numpy(), 'r--', linewidth=2, label='Target')
    ax1.set_xlabel('Subcarrier Index')
    ax1.set_ylabel('Response Magnitude')
    ax1.set_title('Subcarrier Responses (Sample 1)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot MIMO channel matrix for first sample
    mimo_channel_reshaped = outputs['mimo_channel'][sample_idx].view(
        config['model']['num_ue_antennas'], 
        config['model']['num_bs_antennas']
    )
    
    im = ax2.imshow(mimo_channel_reshaped.numpy(), cmap='viridis', aspect='auto')
    ax2.set_xlabel('BS Antenna Index')
    ax2.set_ylabel('UE Antenna Index')
    ax2.set_title('MIMO Channel Matrix (Sample 1)')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('prism_basic_usage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved to 'prism_basic_usage.png'")
    print()
    
    # Demonstrate subcarrier-specific operations
    print("Demonstrating subcarrier-specific operations...")
    
    # Get response for specific subcarrier
    subcarrier_idx = 25  # Middle subcarrier
    specific_response = model.get_subcarrier_response(
        subcarrier_idx,
        positions=positions, 
        ue_antennas=ue_antennas,
        bs_antennas=bs_antennas,
        additional_features=additional_features
    )
    
    print(f"Response for subcarrier {subcarrier_idx}: {specific_response.shape}")
    
    # Get MIMO channel matrix
    mimo_matrix = model.get_mimo_channel(
        positions=positions,
        ue_antennas=ue_antennas,
        bs_antennas=bs_antennas,
        additional_features=additional_features
    )
    
    print(f"MIMO channel matrix shape: {mimo_matrix.shape}")
    print()
    
    print("=== Basic usage demonstration completed successfully! ===")
    print("\nNext steps:")
    print("1. Train the model using: python prism_runner.py --mode train --config configs/ofdm-wifi.yml")
            print("2. Train with 5G features: python prism_runner.py --mode train --config configs/ofdm-5g-sionna.yml")
    print("3. Test the model using: python prism_runner.py --mode test --config configs/ofdm-wifi.yml --checkpoint path/to/checkpoint")
            print("4. Test with 5G features: python prism_runner.py --mode test --config configs/ofdm-5g-sionna.yml --checkpoint path/to/checkpoint")
    print("5. Explore different configurations in the configs/ directory")
    print("6. Modify the model architecture in model.py for your specific needs")

if __name__ == '__main__':
    main()
