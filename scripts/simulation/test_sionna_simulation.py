#!/usr/bin/env python3
"""
Test script for Sionna simulation setup
Verifies that all dependencies are available and basic functionality works
"""

import sys
import numpy as np

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import h5py
        print(f"✓ h5py {h5py.__version__}")
    except ImportError as e:
        print(f"✗ h5py import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA not available (will use CPU)")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
        if tf.config.list_physical_devices('GPU'):
            print(f"  GPU available: {len(tf.config.list_physical_devices('GPU'))} devices")
        else:
            print("  GPU not available (will use CPU)")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import sionna
        print(f"✓ Sionna {sionna.__version__}")
    except ImportError as e:
        print(f"✗ Sionna import failed: {e}")
        print("  Install with: pip install sionna")
        return False
    
    return True

def test_basic_functionality():
    """Test basic Sionna functionality"""
    print("\nTesting basic Sionna functionality...")
    
    try:
        from sionna.channel import UMi
        from sionna.rt import Scene
        
        # Test channel model creation
        channel_model = UMi(
            carrier_frequency=3.5e9,
            o2i_model="low",
            ut_array="3gpp-3d",
            bs_array="3gpp-3d"
        )
        print("✓ UMi channel model created successfully")
        
        # Test scene creation
        scene = Scene("test_scene")
        print("✓ Scene created successfully")
        
        # Test basic channel calculation
        bs_pos = np.array([[0, 0, 25]])
        ut_pos = np.array([[100, 100, 1.5]])
        
        h = channel_model(
            bs_positions=bs_pos,
            ut_positions=ut_pos,
            bs_orientations=np.array([[0, 0, 0]]),
            ut_orientations=np.array([[0, 0, 0]])
        )
        
        print(f"✓ Channel calculation successful, shape: {h.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_data_generation():
    """Test data generation functions"""
    print("\nTesting data generation...")
    
    try:
        # Test UE position generation
        num_positions = 10  # Small number for testing
        
        # Define deployment area
        area_size = 500.0
        bs_position = np.array([area_size/2, area_size/2, 25.0])
        
        # Generate UE positions
        ue_positions = []
        min_distance = 50.0
        
        np.random.seed(42)
        
        for i in range(num_positions):
            while True:
                x = np.random.uniform(0, area_size)
                y = np.random.uniform(0, area_size)
                z = np.random.uniform(1.5, 2.0)
                
                pos = np.array([x, y, z])
                distance = np.linalg.norm(pos[:2] - bs_position[:2])
                
                if distance >= min_distance:
                    ue_positions.append(pos)
                    break
        
        ue_positions = np.array(ue_positions)
        
        print(f"✓ Generated {len(ue_positions)} UE positions")
        print(f"  BS position: {bs_position}")
        print(f"  UE positions shape: {ue_positions.shape}")
        
        # Test minimum distance constraint
        distances = [np.linalg.norm(pos[:2] - bs_position[:2]) for pos in ue_positions]
        min_actual_distance = min(distances)
        
        if min_actual_distance >= min_distance:
            print(f"✓ Minimum distance constraint satisfied: {min_actual_distance:.1f}m >= {min_distance}m")
        else:
            print(f"✗ Minimum distance constraint violated: {min_actual_distance:.1f}m < {min_distance}m")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=== Sionna Simulation Setup Test ===\n")
    
    # Test 1: Package imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing packages.")
        sys.exit(1)
    
    # Test 2: Basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality test failed. Sionna may not be properly installed.")
        sys.exit(1)
    
    # Test 3: Data generation
    if not test_data_generation():
        print("\n❌ Data generation test failed.")
        sys.exit(1)
    
    print("\n✅ All tests passed! Sionna simulation is ready to use.")
    print("\nNext steps:")
    print("1. Run the full simulation: python scripts/sionna_simulation.py")
    print("2. Check the generated data in the 'data/' directory")
    print("3. View the visualization plots")
    print("4. Use the data with your Prism models")

if __name__ == "__main__":
    main()
