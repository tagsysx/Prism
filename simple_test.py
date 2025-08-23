#!/usr/bin/env python3
"""
Simple test to verify accumulate_signals fix.
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prism.ray_tracer_cuda import CUDARayTracer

def main():
    print("Simple test for accumulate_signals...")
    
    # Create tracer
    tracer = CUDARayTracer(
        azimuth_divisions=4,  # Very small for testing
        elevation_divisions=2,
        max_ray_length=50.0,
        scene_size=100.0,
        device='cpu'
    )
    
    # Test data
    base_station_pos = torch.tensor([0.0, 0.0, 0.0])
    ue_positions = [torch.tensor([10.0, 0.0, 0.0]), torch.tensor([0.0, 10.0, 0.0])]
    selected_subcarriers = [0, 1]  # 2 subcarriers
    antenna_embedding = torch.randn(128)
    
    print(f"Testing with {len(ue_positions)} UEs and {len(selected_subcarriers)} subcarriers")
    print(f"Total directions: {tracer.azimuth_divisions * tracer.elevation_divisions}")
    print(f"Expected results: {len(ue_positions) * len(selected_subcarriers)} = {len(ue_positions) * len(selected_subcarriers)}")
    
    # Test accumulate_signals
    try:
        accumulated = tracer.accumulate_signals(
            base_station_pos, ue_positions, selected_subcarriers, antenna_embedding
        )
        
        print(f"Actual results: {len(accumulated)}")
        print(f"Keys: {list(accumulated.keys())}")
        
        if len(accumulated) == len(ue_positions) * len(selected_subcarriers):
            print("✓ SUCCESS: accumulate_signals working correctly!")
            return True
        else:
            print("✗ FAILED: Wrong number of results")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
