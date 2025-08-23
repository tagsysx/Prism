#!/usr/bin/env python3
"""
Example: Training with Configuration File

This example demonstrates how to use the updated TrainingInterface
with ray_tracer integration and configuration file support.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import logging
from unittest.mock import Mock

from prism.config_loader import load_config
from prism.training_interface import PrismTrainingInterface
from prism.ray_tracer_cpu import CPURayTracer
from prism.networks.prism_network import PrismNetwork

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_prism_network(config_loader):
    """Create a mock PrismNetwork for demonstration."""
    nn_config = config_loader.get_neural_network_config()
    
    mock_network = Mock(spec=PrismNetwork)
    mock_network.num_subcarriers = nn_config.get('attenuation_decoder', {}).get('output_dim', 408)
    mock_network.azimuth_divisions = 8
    mock_network.elevation_divisions = 4
    mock_network.top_k_directions = 16
    
    # Mock antenna codebook
    mock_network.antenna_codebook = Mock()
    mock_network.antenna_codebook.return_value = torch.randn(2, 128)  # batch_size=2
    
    # Mock antenna network
    mock_network.antenna_network = Mock()
    mock_network.antenna_network.return_value = torch.randn(2, 8, 4)
    mock_network.antenna_network.get_top_k_directions = Mock()
    mock_network.antenna_network.get_top_k_directions.return_value = (
        torch.randint(0, 8, (2, 16, 2)),  # top_k_indices
        torch.rand(2, 16)  # importance scores
    )
    
    return mock_network


def main():
    """Main example function."""
    print("=== Training with Configuration File Example ===\n")
    
    # 1. Load configuration
    config_path = "configs/ofdm-5g-sionna.yml"
    print(f"Loading configuration from: {config_path}")
    
    try:
        config_loader = load_config(config_path)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        print("Please make sure you're running from the project root directory.")
        return
    
    # 2. Validate configuration
    print("Validating configuration...")
    if not config_loader.validate_config():
        print("Configuration validation failed!")
        return
    print("✓ Configuration is valid\n")
    
    # 3. Get device
    device = config_loader.get_device()
    print(f"Using device: {device}\n")
    
    # 4. Create ray_tracer with configuration
    print("Creating ray_tracer with configuration...")
    rt_kwargs = config_loader.create_ray_tracer_kwargs()
    rt_kwargs['device'] = device  # Ensure device is set
    print(f"Ray tracer parameters: {rt_kwargs}")
    
    ray_tracer = CPURayTracer(**rt_kwargs)
    print("✓ Ray tracer created\n")
    
    # 5. Create mock PrismNetwork
    print("Creating mock PrismNetwork...")
    prism_network = create_mock_prism_network(config_loader)
    print("✓ Mock PrismNetwork created\n")
    
    # 6. Create TrainingInterface with configuration
    print("Creating TrainingInterface with configuration...")
    ti_kwargs = config_loader.create_training_interface_kwargs()
    print(f"TrainingInterface parameters: {ti_kwargs}")
    
    training_interface = PrismTrainingInterface(
        prism_network=prism_network,
        ray_tracer=ray_tracer,
        **ti_kwargs
    )
    print("✓ TrainingInterface created\n")
    
    # 7. Set up curriculum learning
    phases = config_loader.get_curriculum_learning_phases()
    if phases:
        print("Setting up curriculum learning...")
        for i, phase_config in enumerate(phases):
            print(f"  Phase {i}: {phase_config}")
        
        # Set initial phase
        training_interface.set_training_phase(0)
        print("✓ Initial training phase set\n")
    
    # 8. Create sample data
    print("Creating sample training data...")
    batch_size = 2
    num_ue = 3
    num_bs_antennas = 2
    
    ue_positions = torch.randn(batch_size, num_ue, 3, device=device)
    bs_position = torch.zeros(batch_size, 3, device=device)
    antenna_indices = torch.randint(0, 4, (batch_size, num_bs_antennas), device=device)
    
    print(f"Sample data shapes:")
    print(f"  UE positions: {ue_positions.shape}")
    print(f"  BS position: {bs_position.shape}")
    print(f"  Antenna indices: {antenna_indices.shape}\n")
    
    # 9. Mock ray_tracer.accumulate_signals for demonstration
    def mock_accumulate_signals(base_station_pos, ue_positions, selected_subcarriers, antenna_embedding):
        results = {}
        for ue_pos in ue_positions:
            ue_pos_tuple = tuple(ue_pos.tolist())
            if isinstance(selected_subcarriers, dict) and ue_pos_tuple in selected_subcarriers:
                subcarriers = selected_subcarriers[ue_pos_tuple]
                for k in subcarriers:
                    results[(ue_pos_tuple, k)] = 0.5 + 0.1 * k  # Mock signal strength
        return results
    
    ray_tracer.accumulate_signals = mock_accumulate_signals
    
    # 10. Run forward pass
    print("Running forward pass with ray_tracer integration...")
    try:
        outputs = training_interface.forward(
            ue_positions=ue_positions,
            bs_position=bs_position,
            antenna_indices=antenna_indices,
            return_intermediates=True
        )
        
        print("✓ Forward pass completed successfully")
        print(f"Output keys: {list(outputs.keys())}")
        print(f"CSI predictions shape: {outputs['csi_predictions'].shape}")
        print(f"CSI predictions dtype: {outputs['csi_predictions'].dtype}\n")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return
    
    # 11. Test loss computation
    print("Testing loss computation...")
    try:
        # Create mock targets
        csi_shape = outputs['csi_predictions'].shape
        targets = torch.randn(*csi_shape, dtype=torch.complex64, device=device)
        
        # Compute loss with complex-aware loss function
        def complex_mse_loss(predictions, targets):
            """MSE loss for complex tensors."""
            real_loss = nn.functional.mse_loss(predictions.real, targets.real)
            imag_loss = nn.functional.mse_loss(predictions.imag, targets.imag)
            return real_loss + imag_loss
        
        loss = training_interface.compute_loss(outputs['csi_predictions'], targets, complex_mse_loss)
        
        print(f"✓ Loss computation successful: {loss.item():.6f}\n")
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        return
    
    # 12. Test checkpoint functionality
    print("Testing checkpoint functionality...")
    try:
        # Update training state
        training_interface.update_training_state(epoch=1, batch=10, loss=loss.item())
        
        # Save checkpoint
        checkpoint_config = config_loader.get_checkpoint_config()
        checkpoint_dir = checkpoint_config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_file = "example_checkpoint.pt"
        training_interface.save_checkpoint(checkpoint_file)
        
        print(f"✓ Checkpoint saved: {checkpoint_dir}/{checkpoint_file}")
        
        # Test loading (create new interface and load)
        new_training_interface = PrismTrainingInterface(
            prism_network=prism_network,
            ray_tracer=ray_tracer,
            **ti_kwargs
        )
        
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        new_training_interface.load_checkpoint(checkpoint_path)
        
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Loaded epoch: {new_training_interface.current_epoch}")
        print(f"  Loaded batch: {new_training_interface.current_batch}")
        print(f"  Loaded loss: {new_training_interface.best_loss:.6f}\n")
        
    except Exception as e:
        print(f"✗ Checkpoint functionality failed: {e}")
        return
    
    # 13. Test curriculum learning phase transitions
    if phases:
        print("Testing curriculum learning phase transitions...")
        try:
            for phase_idx in range(len(phases)):
                training_interface.set_training_phase(phase_idx)
                phase_config = phases[phase_idx]
                
                print(f"  Phase {phase_idx}:")
                print(f"    Azimuth divisions: {prism_network.azimuth_divisions}")
                print(f"    Elevation divisions: {prism_network.elevation_divisions}")
                print(f"    Top-K directions: {prism_network.top_k_directions}")
            
            print("✓ Curriculum learning phase transitions successful\n")
            
        except Exception as e:
            print(f"✗ Curriculum learning failed: {e}")
            return
    
    # 14. Display training info
    print("Training interface information:")
    training_info = training_interface.get_training_info()
    for key, value in training_info.items():
        print(f"  {key}: {value}")
    
    print("\n=== Example completed successfully! ===")
    
    # 15. Display configuration summary
    print("\nConfiguration Summary:")
    print(f"  Training Interface enabled: {config_loader.get_training_interface_config().get('enabled', False)}")
    print(f"  Ray tracer integration enabled: {config_loader.get_ray_tracer_config().get('enabled', False)}")
    print(f"  Curriculum learning enabled: {phases is not None}")
    print(f"  Checkpoint directory: {config_loader.get_checkpoint_config().get('checkpoint_dir', 'checkpoints')}")
    print(f"  Subcarrier sampling ratio: {config_loader.get_training_interface_config().get('subcarrier_sampling_ratio', 0.3)}")


if __name__ == "__main__":
    main()
