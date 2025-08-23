#!/usr/bin/env python3
"""
Test script for the updated CUDA ray tracer.
This script verifies that the updated ray_tracer_cuda.py has the same interface and functionality as ray_tracer.py.
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prism.ray_tracer_cuda import CUDARayTracer, Ray, BaseStation, UserEquipment

def test_basic_functionality():
    """Test basic functionality of the updated CUDA ray tracer."""
    print("Testing basic functionality...")
    
    # Test initialization
    try:
        tracer = CUDARayTracer(
            azimuth_divisions=36,
            elevation_divisions=18,
            max_ray_length=100.0,
            scene_size=200.0,
            device='cpu',  # Use CPU for testing
            signal_threshold=1e-6,
            enable_early_termination=True
        )
        print("✓ CUDA ray tracer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize CUDA ray tracer: {e}")
        return False
    
    # Test scene configuration
    scene_config = tracer.get_scene_config()
    expected_keys = ['scene_size', 'scene_min', 'scene_max', 'max_ray_length', 'azimuth_divisions', 'elevation_divisions']
    for key in expected_keys:
        if key not in scene_config:
            print(f"✗ Missing scene config key: {key}")
            return False
    print("✓ Scene configuration working correctly")
    
    # Test direction vector generation
    try:
        directions = tracer.generate_direction_vectors()
        expected_shape = (36 * 18, 3)  # (azimuth * elevation, 3)
        if directions.shape != expected_shape:
            print(f"✗ Direction vectors shape mismatch: expected {expected_shape}, got {directions.shape}")
            return False
        print("✓ Direction vectors generated correctly")
    except Exception as e:
        print(f"✗ Failed to generate direction vectors: {e}")
        return False
    
    # Test position validation
    test_positions = [
        torch.tensor([0.0, 0.0, 0.0]),  # Center
        torch.tensor([100.0, 100.0, 100.0]),  # Within bounds
        torch.tensor([200.0, 200.0, 200.0]),  # Outside bounds
    ]
    
    expected_results = [True, True, False]
    for pos, expected in zip(test_positions, expected_results):
        result = tracer.is_position_in_scene(pos)
        if result != expected:
            print(f"✗ Position validation failed for {pos}: expected {expected}, got {result}")
            return False
    print("✓ Position validation working correctly")
    
    return True

def test_ray_tracing():
    """Test ray tracing functionality."""
    print("\nTesting ray tracing...")
    
    tracer = CUDARayTracer(
        azimuth_divisions=12,  # Smaller for faster testing
        elevation_divisions=6,
        max_ray_length=50.0,
        scene_size=100.0,
        device='cpu'
    )
    
    # Test data
    base_station_pos = torch.tensor([0.0, 0.0, 0.0])
    ue_positions = [torch.tensor([10.0, 0.0, 0.0]), torch.tensor([0.0, 10.0, 0.0])]
    selected_subcarriers = [0, 1, 2]  # Test with list format
    antenna_embedding = torch.randn(128)
    
    # Test single ray tracing
    try:
        direction = (0, 0)  # First direction
        results = tracer.trace_ray(
            base_station_pos, direction, ue_positions, selected_subcarriers, antenna_embedding
        )
        
        if not isinstance(results, dict):
            print("✗ trace_ray should return a dictionary")
            return False
        
        # Check that we have results for each UE-subcarrier combination
        expected_keys = 6  # 2 UEs × 3 subcarriers
        if len(results) != expected_keys:
            print(f"✗ Expected {expected_keys} results, got {len(results)}")
            return False
        
        print("✓ Single ray tracing working correctly")
    except Exception as e:
        print(f"✗ Single ray tracing failed: {e}")
        return False
    
    # Test signal accumulation
    try:
        print(f"Debug: Testing with {len(ue_positions)} UEs and {len(selected_subcarriers)} subcarriers")
        print(f"Debug: UE positions: {[tuple(pos.tolist()) for pos in ue_positions]}")
        print(f"Debug: Selected subcarriers: {selected_subcarriers}")
        
        accumulated = tracer.accumulate_signals(
            base_station_pos, ue_positions, selected_subcarriers, antenna_embedding
        )
        
        if not isinstance(accumulated, dict):
            print("✗ accumulate_signals should return a dictionary")
            return False
        
        # Debug: print the actual results
        print(f"Debug: accumulate_signals returned {len(accumulated)} results")
        print(f"Debug: First few keys: {list(accumulated.keys())[:5]}")
        print(f"Debug: Expected UE-subcarrier combinations: {expected_keys}")
        
        # Check if the keys are in the expected format
        if len(accumulated) > 0:
            first_key = list(accumulated.keys())[0]
            print(f"Debug: First key type: {type(first_key)}")
            print(f"Debug: First key content: {first_key}")
            if isinstance(first_key, tuple) and len(first_key) == 2:
                print(f"Debug: First key UE part type: {type(first_key[0])}")
                print(f"Debug: First key subcarrier part type: {type(first_key[1])}")
        
        # Should have results for each UE-subcarrier combination
        if len(accumulated) != expected_keys:
            print(f"✗ Expected {expected_keys} accumulated results, got {len(accumulated)}")
            print(f"Debug: This suggests the fallback method is being used and tracing all directions")
            print(f"Debug: Total directions: {tracer.azimuth_divisions * tracer.elevation_divisions}")
            print(f"Debug: Directions × UE-subcarrier combinations = {tracer.azimuth_divisions * tracer.elevation_divisions * expected_keys}")
            
            # Check if this is actually the expected behavior
            if len(accumulated) == tracer.azimuth_divisions * tracer.elevation_divisions * expected_keys:
                print("Debug: This might be the correct behavior - returning results for each direction")
                print("Debug: But accumulate_signals should accumulate across directions")
                return False
            return False
        
        print("✓ Signal accumulation working correctly")
    except Exception as e:
        print(f"✗ Signal accumulation failed: {e}")
        return False
    
    return True

def test_auxiliary_classes():
    """Test the auxiliary classes (Ray, BaseStation, UserEquipment)."""
    print("\nTesting auxiliary classes...")
    
    # Test Ray class
    try:
        origin = torch.tensor([0.0, 0.0, 0.0])
        direction = torch.tensor([1.0, 0.0, 0.0])
        ray = Ray(origin, direction, max_length=100.0, device='cpu')
        
        if not torch.allclose(ray.direction, torch.tensor([1.0, 0.0, 0.0])):
            print("✗ Ray direction not normalized correctly")
            return False
        
        print("✓ Ray class working correctly")
    except Exception as e:
        print(f"✗ Ray class failed: {e}")
        return False
    
    # Test BaseStation class
    try:
        bs = BaseStation(position=torch.tensor([1.0, 2.0, 3.0]), num_antennas=2, device='cpu')
        
        if bs.num_antennas != 2:
            print("✗ BaseStation antenna count incorrect")
            return False
        
        embedding = bs.get_antenna_embedding(0)
        if embedding.shape != (128,):
            print(f"✗ BaseStation embedding shape incorrect: {embedding.shape}")
            return False
        
        print("✓ BaseStation class working correctly")
    except Exception as e:
        print(f"✗ BaseStation class failed: {e}")
        return False
    
    # Test UserEquipment class
    try:
        ue = UserEquipment(position=torch.tensor([5.0, 5.0, 5.0]), device='cpu')
        
        if not torch.allclose(ue.position, torch.tensor([5.0, 5.0, 5.0])):
            print("✗ UserEquipment position not set correctly")
            return False
        
        print("✓ UserEquipment class working correctly")
    except Exception as e:
        print(f"✗ UserEquipment class failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Testing updated CUDA ray tracer...")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_ray_tracing,
        test_auxiliary_classes
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ Test {test.__name__} failed")
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The updated CUDA ray tracer is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
