#!/usr/bin/env python3
"""
Example: Basic Ray Tracing with Prism

This example demonstrates basic ray tracing functionality
using the Prism system components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np

from prism.ray_tracer_cpu import CPURayTracer
from prism.ray_tracer_base import Ray


def main():
    """Main example function."""
    print("Prism Discrete Electromagnetic Ray Tracing Example")
    print("=" * 50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Initialize the ray tracing system
    print("\n1. Initializing ray tracing system...")
    ray_tracer = CPURayTracer(
        azimuth_divisions=8,
        elevation_divisions=4,
        max_ray_length=50.0,
        scene_size=100.0,
        device=device
    )
    
    # 2. Create base station and user equipment
    print("2. Setting up base station and UE...")
    base_station = BaseStation(
        position=torch.tensor([0.0, 0.0, 0.0], device=device),
        num_antennas=2,
        device=device
    )
    
    ue_positions = [
        [10.0, 5.0, 1.5],
        [15.0, -3.0, 1.5],
        [8.0, 12.0, 1.5]
    ]
    
    ue_devices = [UserEquipment(torch.tensor(pos, device=device), device=device) 
                  for pos in ue_positions]
    
    # 3. Perform ray tracing with integrated ray_tracer
    print("3. Performing ray tracing with integrated ray_tracer...")
    
    # Get antenna embedding for the first antenna
    antenna_embedding = base_station.get_antenna_embedding(antenna_idx=0)
    
    # Define subcarriers for demonstration (in practice, this would come from TrainingInterface)
    selected_subcarriers = [0, 1, 2, 3]  # 4 subcarriers
    
    # Perform ray tracing using the integrated ray_tracer
    accumulated_signals = ray_tracer.accumulate_signals(
        base_station_pos=base_station.position,
        ue_positions=ue_positions,
        selected_subcarriers=selected_subcarriers,
        antenna_embedding=antenna_embedding
    )
    
    # 4. Display results
    print("4. Results:")
    print(f"   - Total directions: {ray_tracer.total_directions}")
    print(f"   - UE positions: {len(ue_positions)}")
    print(f"   - Subcarriers: {len(selected_subcarriers)}")
    print(f"   - Virtual links processed: {len(accumulated_signals)}")
    
    # Display some signal strengths
    print("\n   Sample signal strengths:")
    for i, ((ue_pos, subcarrier), signal_strength) in enumerate(list(accumulated_signals.items())[:3]):
        print(f"     UE{ue_pos}, Subcarrier{subcarrier}: {signal_strength:.6f}")
    
    print("\nRay tracing example completed successfully!")
    
    return {
        'ray_tracer': ray_tracer,
        'accumulated_signals': accumulated_signals
    }


if __name__ == "__main__":
    try:
        results = main()
        print(f"\nExample completed with {len(results['accumulated_signals'])} virtual links")
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()
