#!/usr/bin/env python3
"""
Example script demonstrating the Prism discrete electromagnetic ray tracing system.

This example shows how to:
1. Initialize the ray tracing system
2. Set up MLP-based direction sampling
3. Perform ray tracing with antenna embeddings
4. Process RF signals for virtual links
"""

import torch
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prism.ray_tracer import DiscreteRayTracer, BaseStation, UserEquipment, VoxelGrid, Environment
from prism.mlp_direction_sampler import create_mlp_direction_sampler
from prism.rf_signal_processor import RFSignalProcessor

def main():
    """Main example function."""
    print("Prism Discrete Electromagnetic Ray Tracing Example")
    print("=" * 50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Initialize the ray tracing system
    print("\n1. Initializing ray tracing system...")
    ray_tracer = DiscreteRayTracer(
        azimuth_divisions=36,
        elevation_divisions=18,
        max_ray_length=100.0,
        device=device
    )
    
    # 2. Create environment with voxel grid
    print("2. Setting up environment...")
    voxel_grid = VoxelGrid(
        grid_size=(100, 100, 25),
        voxel_size=1.0,
        device=device
    )
    environment = Environment(voxel_grid, device=device)
    
    # 3. Create base station and user equipment
    print("3. Setting up base station and UE...")
    base_station = BaseStation(
        position=torch.tensor([0.0, 0.0, 0.0], device=device),
        num_antennas=4,
        device=device
    )
    
    ue_positions = [
        [10.0, 5.0, 1.5],
        [15.0, -3.0, 1.5],
        [8.0, 12.0, 1.5],
        [-5.0, 8.0, 1.5]
    ]
    
    ue_devices = [UserEquipment(torch.tensor(pos, device=device), device=device) 
                  for pos in ue_positions]
    
    # 4. Initialize MLP direction sampler
    print("4. Setting up MLP direction sampler...")
    mlp_sampler = create_mlp_direction_sampler(
        azimuth_divisions=36,
        elevation_divisions=18,
        hidden_dim=256,
        num_hidden_layers=2
    )
    
    # 5. Initialize RF signal processor
    print("5. Setting up RF signal processor...")
    rf_processor = RFSignalProcessor(
        total_subcarriers=64,
        sampling_ratio=0.5,
        frequency_band=2.4e9,
        device=device
    )
    
    # 6. Perform ray tracing
    print("6. Performing ray tracing...")
    
    # Get antenna embedding for the first antenna
    antenna_embedding = base_station.get_antenna_embedding(antenna_idx=0)
    
    # Select subcarriers for each UE
    selected_subcarriers = rf_processor.subcarrier_selector.select_subcarriers(ue_positions)
    
    # Perform adaptive ray tracing using MLP
    accumulated_signals = ray_tracer.adaptive_ray_tracing(
        base_station_pos=base_station.position,
        antenna_embedding=antenna_embedding,
        ue_positions=ue_positions,
        selected_subcarriers=selected_subcarriers,
        mlp_model=mlp_sampler
    )
    
    # 7. Process virtual links
    print("7. Processing virtual links...")
    
    # Simulate ray tracing results for processing
    ray_tracing_results = {}
    for ue_pos in ue_positions:
        ue_pos_key = tuple(ue_pos)
        ue_subcarriers = selected_subcarriers[ue_pos_key]
        
        for subcarrier_idx in ue_subcarriers:
            # Create a simple ray path (in practice, this would come from ray tracing)
            ray_path = torch.tensor([
                base_station.position,
                torch.tensor(ue_pos, device=device)
            ], device=device)
            
            ray_tracing_results[(ue_pos_key, subcarrier_idx)] = ray_path
    
    # Process virtual links
    virtual_link_results = rf_processor.process_virtual_links(
        base_station_pos=base_station.position,
        ue_positions=ue_positions,
        antenna_embedding=antenna_embedding,
        ray_tracing_results=ray_tracing_results
    )
    
    # 8. Display results
    print("8. Results:")
    print(f"   - Total directions: {ray_tracer.total_directions}")
    print(f"   - UE positions: {len(ue_positions)}")
    print(f"   - Subcarriers per UE: {len(list(selected_subcarriers.values())[0])}")
    print(f"   - Virtual links processed: {len(virtual_link_results)}")
    
    # Calculate sampling efficiency
    test_embedding = torch.randn(1, 128, device=device)
    efficiency = mlp_sampler.get_sampling_efficiency(test_embedding, 36, 18)
    print(f"   - MLP sampling efficiency: {efficiency:.3f}")
    
    # Display some signal strengths
    print("\n   Sample signal strengths:")
    for i, ((ue_pos, subcarrier), result) in enumerate(list(virtual_link_results.items())[:3]):
        print(f"     UE{ue_pos}, Subcarrier{subcarrier}: {result['signal_strength']:.6f}")
    
    print("\nRay tracing example completed successfully!")
    
    return {
        'ray_tracer': ray_tracer,
        'mlp_sampler': mlp_sampler,
        'rf_processor': rf_processor,
        'virtual_link_results': virtual_link_results
    }

if __name__ == "__main__":
    try:
        results = main()
        print(f"\nExample completed with {len(results['virtual_link_results'])} virtual links")
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()
