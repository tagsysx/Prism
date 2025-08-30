#!/usr/bin/env python3
"""
Importance-Based Sampling Example

This example demonstrates the updated ray tracer with importance-based sampling
that implements the two-stage approach:
1. Uniform sampling with weight computation
2. Importance-based resampling based on computed weights
"""

import torch
import numpy as np
from src.prism.ray_tracer_cpu import CPURayTracer
from src.prism.ray_tracer_base import Ray
from src.prism.networks import PrismNetwork, PrismNetworkConfig

def main():
    """Demonstrate importance-based sampling in ray tracing."""
    print("=== Importance-Based Sampling Example ===\n")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create a simple PrismNetwork for demonstration
    config = PrismNetworkConfig(
        num_ue=1,
        num_subcarriers=4,
        num_bs=1,
        num_antennas=1
    )
    prism_network = PrismNetwork(config).to(device)
    print("✓ PrismNetwork created")
    
    # Create ray tracer with importance sampling
    ray_tracer = CPURayTracer(
        azimuth_divisions=8,
        elevation_divisions=4,
        prism_network=prism_network,
        device=device
    )
    print("✓ CPURayTracer created with importance sampling")
    
    # Create test scenario
    base_station = BaseStation(torch.tensor([0.0, 0.0, 0.0]))
    ue = UserEquipment(torch.tensor([10.0, 5.0, 2.0]))
    
    # Create a ray from base station to UE
    ray_direction = ue.position - base_station.position
    ray_direction = ray_direction / torch.norm(ray_direction)
    ray = Ray(base_station.position, ray_direction)
    
    print(f"\nBase Station Position: {base_station.position}")
    print(f"UE Position: {ue.position}")
    print(f"Ray Direction: {ray.direction}")
    
    # Test importance-based sampling
    print("\n--- Testing Importance-Based Sampling ---")
    
    # Create dummy antenna embedding
    antenna_embedding = torch.randn(64, device=device)
    
    # Test ray tracing with importance sampling
    try:
        signal_strength = ray_tracer._discrete_radiance_ray_tracing(
            ray=ray,
            ue_pos=ue.position,
            subcarrier_idx=0,
            antenna_embedding=antenna_embedding
        )
        
        print(f"✓ Signal strength computed: {signal_strength:.6f}")
        print("✓ Importance-based sampling completed successfully")
        
    except Exception as e:
        print(f"✗ Error during importance-based sampling: {e}")
        print("This might be due to network configuration or tensor shape issues")
    
    # Test the individual importance sampling components
    print("\n--- Testing Individual Components ---")
    
    try:
        # Test uniform sampling
        uniform_positions = ray_tracer._sample_ray_points(ray, ue.position, 128)
        print(f"✓ Uniform sampling: {len(uniform_positions)} points generated")
        
        # Test importance weight computation (with dummy attenuation)
        dummy_attenuation = torch.randn(128, device=device) + 1j * torch.randn(128, device=device)
        importance_weights = ray_tracer._compute_importance_weights(dummy_attenuation)
        print(f"✓ Importance weights computed: shape {importance_weights.shape}")
        print(f"  Weight sum: {torch.sum(importance_weights):.6f}")
        print(f"  Weight range: [{torch.min(importance_weights):.6f}, {torch.max(importance_weights):.6f}]")
        
        # Test importance-based resampling
        resampled_positions = ray_tracer._importance_based_resampling(
            uniform_positions, importance_weights, 64
        )
        print(f"✓ Importance-based resampling: {len(resampled_positions)} points selected")
        
        print("\n✓ All importance sampling components working correctly!")
        
    except Exception as e:
        print(f"✗ Error testing components: {e}")
    
    print("\n=== Example Completed ===")

if __name__ == "__main__":
    main()
