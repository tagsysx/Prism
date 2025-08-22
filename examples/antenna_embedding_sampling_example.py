#!/usr/bin/env python3
"""
Antenna Embedding-based Direction Sampling Example

This example demonstrates how the updated ray tracer uses antenna embedding C
to perform MLP-based direction sampling, as specified in the design document.
"""

import torch
import numpy as np
from typing import List, Dict

# Import Prism components
from prism import (
    DiscreteRayTracer,
    BaseStation,
    UserEquipment,
    PrismNetwork,
    AntennaEmbeddingCodebook
)

def main():
    """Demonstrate antenna embedding-based direction sampling."""
    print("=== Antenna Embedding-based Direction Sampling Example ===\n")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Create PrismNetwork with AntennaNetwork for direction sampling
    print("\n1. Creating PrismNetwork with AntennaNetwork...")
    prism_network = PrismNetwork(
        num_subcarriers=64,
        num_ue_antennas=4,
        num_bs_antennas=8,
        azimuth_divisions=16,
        elevation_divisions=8,
        top_k_directions=32,
        antenna_embedding_dim=64
    ).to(device)
    
    # 2. Create DiscreteRayTracer with PrismNetwork
    print("2. Creating DiscreteRayTracer with integrated PrismNetwork...")
    ray_tracer = DiscreteRayTracer(
        azimuth_divisions=16,
        elevation_divisions=8,
        max_ray_length=100.0,
        prism_network=prism_network,
        device=device
    )
    
    # 3. Create base station and UE
    print("3. Setting up base station and user equipment...")
    bs_pos = torch.tensor([0.0, 0.0, 10.0], device=device)
    base_station = BaseStation(bs_pos, device=device)
    
    ue_positions = [
        torch.tensor([20.0, 15.0, 2.0], device=device),
        torch.tensor([-10.0, 25.0, 2.0], device=device)
    ]
    
    # 4. Get antenna embedding from codebook
    print("4. Getting antenna embedding from codebook...")
    antenna_indices = torch.tensor([0], dtype=torch.long, device=device)  # Use first antenna
    antenna_embedding = prism_network.antenna_codebook(antenna_indices)[0]  # Shape: (64,)
    
    print(f"   Antenna embedding shape: {antenna_embedding.shape}")
    print(f"   Antenna embedding norm: {torch.norm(antenna_embedding):.4f}")
    
    # 5. Demonstrate MLP-based direction sampling
    print("\n5. Demonstrating MLP-based direction sampling...")
    
    # Get directional importance matrix from AntennaNetwork
    with torch.no_grad():
        directional_importance = prism_network.antenna_network(antenna_embedding.unsqueeze(0))
        print(f"   Directional importance shape: {directional_importance.shape}")
        
        # Get top-K directions
        top_k_directions, top_k_importance = prism_network.antenna_network.get_top_k_directions(
            directional_importance, k=8
        )
        
        print(f"   Selected {top_k_directions.shape[1]} important directions:")
        selected_dirs = top_k_directions[0]  # First batch element
        selected_importance = top_k_importance[0]  # First batch element
        
        for i in range(min(5, selected_dirs.shape[0])):  # Show first 5
            phi_idx, theta_idx = selected_dirs[i]
            importance = selected_importance[i]
            phi_angle = phi_idx * (2 * np.pi / 16)  # Convert to radians
            theta_angle = theta_idx * (np.pi / 8)    # Convert to radians
            print(f"     Direction {i+1}: φ={phi_angle:.2f}rad, θ={theta_angle:.2f}rad, importance={importance:.4f}")
    
    # 6. Perform ray tracing with antenna embedding-based sampling
    print("\n6. Performing ray tracing with antenna embedding-based sampling...")
    
    # Select subcarriers
    selected_subcarriers = ray_tracer.select_subcarriers(64, 0.25)  # 25% sampling
    print(f"   Selected {len(selected_subcarriers)} subcarriers: {selected_subcarriers[:5]}...")
    
    # Accumulate signals using MLP-based direction sampling
    accumulated_signals = ray_tracer.accumulate_signals(
        base_station_pos=bs_pos,
        ue_positions=ue_positions,
        selected_subcarriers={0: selected_subcarriers},  # Map UE index to subcarriers
        antenna_embedding=antenna_embedding
    )
    
    print(f"   Computed signals for {len(accumulated_signals)} virtual links")
    
    # 7. Compare with fallback method (all directions)
    print("\n7. Comparing with fallback method (all directions)...")
    
    # Temporarily disable network to force fallback
    original_network = ray_tracer.prism_network
    ray_tracer.prism_network = None
    
    fallback_signals = ray_tracer.accumulate_signals(
        base_station_pos=bs_pos,
        ue_positions=ue_positions,
        selected_subcarriers={0: selected_subcarriers},
        antenna_embedding=antenna_embedding
    )
    
    # Restore network
    ray_tracer.prism_network = original_network
    
    print(f"   Fallback method computed signals for {len(fallback_signals)} virtual links")
    
    # 8. Analyze computational efficiency
    print("\n8. Computational efficiency analysis:")
    total_directions = 16 * 8  # azimuth × elevation
    selected_directions = 8   # top-K
    efficiency_gain = total_directions / selected_directions
    
    print(f"   Total possible directions: {total_directions}")
    print(f"   MLP-selected directions: {selected_directions}")
    print(f"   Computational efficiency gain: {efficiency_gain:.1f}x")
    
    # 9. Ray count analysis
    print("\n9. Ray count analysis:")
    ray_analysis = ray_tracer.get_ray_count_analysis(
        num_bs=1, num_ue=len(ue_positions), num_subcarriers=len(selected_subcarriers)
    )
    
    for key, value in ray_analysis.items():
        if key != 'ray_count_formula':
            print(f"   {key}: {value}")
    print(f"   {ray_analysis['ray_count_formula']}")
    
    print("\n=== Example completed successfully! ===")
    print("\nKey improvements:")
    print("✓ Antenna embedding C is now used for direction sampling")
    print("✓ AntennaNetwork (MLP) selects important directions automatically")
    print("✓ Computational efficiency improved through selective ray tracing")
    print("✓ Importance-based sampling implemented for ray points")

if __name__ == "__main__":
    main()
