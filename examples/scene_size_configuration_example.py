#!/usr/bin/env python3
"""
Scene Size Configuration Example

This example demonstrates how to configure and use scene size D in the ray tracer,
which affects ray length limits, sampling, and environment boundaries.
"""

import torch
import numpy as np
from typing import List, Dict

# Import Prism components
from prism import (
    CPURayTracer,
    BaseStation,
    UserEquipment,
    PrismNetwork
)

def main():
    """Demonstrate scene size configuration and its effects."""
    print("=== Scene Size Configuration Example ===\n")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Create ray tracer with different scene sizes
    print("\n1. Creating ray tracers with different scene sizes...")
    
    # Small scene (indoor environment)
    small_scene_tracer = CPURayTracer(
        azimuth_divisions=16,
        elevation_divisions=8,
        max_ray_length=50.0,
        scene_size=100.0,  # 100m x 100m x 100m indoor environment
        device=device
    )
    
    # Medium scene (urban environment)
    medium_scene_tracer = CPURayTracer(
        azimuth_divisions=32,
        elevation_divisions=16,
        max_ray_length=200.0,
        scene_size=500.0,  # 500m x 500m x 500m urban environment
        device=device
    )
    
    # Large scene (rural environment)
    large_scene_tracer = CPURayTracer(
        azimuth_divisions=64,
        elevation_divisions=32,
        max_ray_length=1000.0,
        scene_size=2000.0,  # 2km x 2km x 2km rural environment
        device=device
    )
    
    print("✓ Created ray tracers with different scene sizes")
    
    # 2. Display scene configurations
    print("\n2. Scene configurations:")
    
    tracers = [
        ("Small (Indoor)", small_scene_tracer),
        ("Medium (Urban)", medium_scene_tracer),
        ("Large (Rural)", large_scene_tracer)
    ]
    
    for name, tracer in tracers:
        config = tracer.get_scene_config()
        print(f"\n   {name} Scene:")
        print(f"     Scene size: {config['scene_size']}m")
        print(f"     Boundaries: [{config['scene_min']:.1f}, {config['scene_max']:.1f}]³")
        print(f"     Max ray length: {config['max_ray_length']}m")
        print(f"     Direction grid: {config['azimuth_divisions']}×{config['elevation_divisions']}")
    
    # 3. Test position validation
    print("\n3. Testing position validation...")
    
    test_positions = [
        torch.tensor([0.0, 0.0, 0.0]),      # Center
        torch.tensor([25.0, 30.0, 15.0]),   # Within small scene
        torch.tensor([100.0, 150.0, 200.0]), # Outside small scene
        torch.tensor([200.0, 250.0, 300.0]), # Within medium scene
        torch.tensor([1000.0, 1200.0, 800.0]) # Within large scene
    ]
    
    for i, pos in enumerate(test_positions):
        small_valid = small_scene_tracer.is_position_in_scene(pos)
        medium_valid = medium_scene_tracer.is_position_in_scene(pos)
        large_valid = large_scene_tracer.is_position_in_scene(pos)
        
        print(f"   Position {i+1} {pos.tolist()}:")
        print(f"     Small scene: {'✓' if small_valid else '✗'}")
        print(f"     Medium scene: {'✓' if medium_valid else '✗'}")
        print(f"     Large scene: {'✓' if large_valid else '✗'}")
    
    # 4. Dynamic scene size adjustment
    print("\n4. Dynamic scene size adjustment...")
    
    # Start with small scene
    tracer = CPURayTracer(
        azimuth_divisions=16,
        elevation_divisions=8,
        max_ray_length=50.0,
        scene_size=100.0,
        device=device
    )
    
    print(f"   Initial scene size: {tracer.get_scene_size()}m")
    
    # Expand scene for outdoor deployment
    tracer.update_scene_size(300.0)
    print(f"   Updated scene size: {tracer.get_scene_size()}m")
    print(f"   New boundaries: {tracer.get_scene_bounds()}")
    
    # Expand further for rural deployment
    tracer.update_scene_size(1000.0)
    print(f"   Final scene size: {tracer.get_scene_size()}m")
    print(f"   Max ray length: {tracer.max_ray_length}m")
    
    # 5. Demonstrate scene size impact on ray tracing
    print("\n5. Scene size impact on ray tracing...")
    
    # Create base station and UE
    bs_pos = torch.tensor([0.0, 0.0, 10.0], device=device)
    ue_positions = [
        torch.tensor([50.0, 60.0, 2.0], device=device),   # Within small scene
        torch.tensor([150.0, 200.0, 2.0], device=device), # Outside small, within medium
        torch.tensor([500.0, 600.0, 2.0], device=device)  # Outside medium, within large
    ]
    
    # Test ray tracing with different scene sizes
    for scene_name, scene_tracer in tracers:
        print(f"\n   Testing {scene_name} scene:")
        
        for i, ue_pos in enumerate(ue_positions):
            in_scene = scene_tracer.is_position_in_scene(ue_pos)
            distance = torch.norm(ue_pos - bs_pos).item()
            
            print(f"     UE {i+1} at distance {distance:.1f}m: {'✓' if in_scene else '✗'}")
            
            if in_scene:
                # Test ray sampling
                ray = scene_tracer.Ray(bs_pos, ue_pos - bs_pos, device=device)
                sampled_positions = scene_tracer._sample_ray_points(ray, ue_pos, num_samples=10)
                print(f"       Sampled {len(sampled_positions)} valid positions")
            else:
                print(f"       Position outside scene boundaries")
    
    # 6. Performance implications
    print("\n6. Performance implications of scene size:")
    
    for name, tracer in tracers:
        config = tracer.get_scene_config()
        total_directions = config['azimuth_divisions'] * config['elevation_divisions']
        scene_volume = config['scene_size'] ** 3
        
        print(f"\n   {name} Scene:")
        print(f"     Total directions: {total_directions:,}")
        print(f"     Scene volume: {scene_volume:,.0f} m³")
        print(f"     Directions per m³: {total_directions / scene_volume:.2f}")
    
    print("\n=== Example completed successfully! ===")
    print("\nKey insights:")
    print("✓ Scene size D is now a configurable parameter")
    print("✓ Affects ray length limits and sampling boundaries")
    print("✓ Enables different deployment scenarios (indoor/urban/rural)")
    print("✓ Supports dynamic scene size adjustment")
    print("✓ Validates positions against scene boundaries")

if __name__ == "__main__":
    main()
