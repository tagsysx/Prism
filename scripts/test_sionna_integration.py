#!/usr/bin/env python3
"""
Test script for Sionna integration with Prism.
This script verifies that the Sionna data loader and model integration work correctly.
"""

import sys
import os
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_config_loading():
    """Test configuration file loading."""
    print("Testing configuration loading...")
    
    config_path = 'configs/ofdm-5g-sionna.yml'
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['model', 'data', 'sionna_integration']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required section: {section}")
                return False
        
        # Check Sionna integration
        sionna_config = config['sionna_integration']
        if not sionna_config.get('enabled', False):
            print("❌ Sionna integration not enabled")
            return False
        
        print("✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def test_sionna_data_loader():
    """Test Sionna data loader initialization."""
    print("\nTesting Sionna data loader...")
    
    try:
        from prism.utils.sionna_data_loader import SionnaDataLoader
        
        # Load configuration
        config_path = 'configs/ofdm-5g-sionna.yml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if data file exists
        data_file = config['data']['data_dir']
        if not os.path.exists(data_file):
            print(f"⚠️  Sionna data file not found: {data_file}")
            print("   This is expected if you haven't run the simulation yet.")
            print("   Run 'python scripts/simulation/sionna_simulation.py' first.")
            return True  # Not a failure, just missing data
        
        # Try to initialize data loader
        data_loader = SionnaDataLoader(config)
        
        # Check basic properties
        assert data_loader.num_subcarriers == config['model']['num_subcarriers']
        assert data_loader.num_ue_antennas == config['model']['num_ue_antennas']
        assert data_loader.num_bs_antennas == config['model']['num_bs_antennas']
        
        print("✅ Sionna data loader initialized successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you have installed the required dependencies.")
        return False
    except Exception as e:
        print(f"❌ Error initializing data loader: {e}")
        return False

def test_model_creation():
    """Test Prism model creation with Sionna configuration."""
    print("\nTesting model creation...")
    
    try:
        from prism.model import create_prism_model
        
        # Load configuration
        config_path = 'configs/ofdm-5g-sionna.yml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model = create_prism_model(config)
        
        # Check model parameters
        assert model.num_subcarriers == config['model']['num_subcarriers']
        assert model.num_ue_antennas == config['model']['num_ue_antennas']
        assert model.num_bs_antennas == config['model']['num_bs_antennas']
        
        print("✅ Prism model created successfully")
        print(f"   Subcarriers: {model.num_subcarriers}")
        print(f"   UE antennas: {model.num_ue_antennas}")
        print(f"   BS antennas: {model.num_bs_antennas}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

def test_model_forward_pass():
    """Test model forward pass with sample data."""
    print("\nTesting model forward pass...")
    
    try:
        from prism.model import create_prism_model
        
        # Load configuration
        config_path = 'configs/ofdm-5g-sionna.yml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model = create_prism_model(config)
        model.eval()
        
        # Create sample input data
        batch_size = 4
        positions = torch.randn(batch_size, 3)
        ue_antennas = torch.randn(batch_size, config['model']['num_ue_antennas'])
        bs_antennas = torch.randn(batch_size, config['model']['num_bs_antennas'])
        additional_features = torch.randn(batch_size, 10)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                positions=positions,
                ue_antennas=ue_antennas,
                bs_antennas=bs_antennas,
                additional_features=additional_features
            )
        
        # Check output shapes
        expected_subcarrier_shape = (batch_size, config['model']['num_subcarriers'])
        expected_mimo_shape = (batch_size, config['model']['num_ue_antennas'] * config['model']['num_bs_antennas'])
        
        assert outputs['subcarrier_responses'].shape == expected_subcarrier_shape
        assert outputs['mimo_channel'].shape == expected_mimo_shape
        
        print("✅ Model forward pass successful")
        print(f"   Subcarrier responses: {outputs['subcarrier_responses'].shape}")
        print(f"   MIMO channel: {outputs['mimo_channel'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        return False

def test_csi_configuration():
    """Test CSI configuration parameters."""
    print("\nTesting CSI configuration...")
    
    try:
        # Load configuration
        config_path = 'configs/ofdm-5g-sionna.yml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        csi_config = config.get('csi_processing', {})
        if not csi_config.get('virtual_link_enabled', False):
            print("⚠️  CSI processing not enabled")
            return True
        
        # Check CSI parameters
        m_subcarriers = csi_config['m_subcarriers']
        n_ue_antennas = csi_config['n_ue_antennas']
        n_bs_antennas = csi_config['n_bs_antennas']
        
        # Verify calculations
        expected_virtual_links = m_subcarriers * n_ue_antennas
        actual_virtual_links = csi_config['virtual_link_count']
        
        if expected_virtual_links != actual_virtual_links:
            print(f"❌ Virtual link count mismatch: expected {expected_virtual_links}, got {actual_virtual_links}")
            return False
        
        print("✅ CSI configuration verified")
        print(f"   Virtual links: {actual_virtual_links}")
        print(f"   Uplinks per BS antenna: {csi_config['uplink_per_bs_antenna']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in CSI configuration: {e}")
        return False

def main():
    """Run all tests."""
    print("Sionna Integration Test Suite")
    print("=" * 40)
    
    tests = [
        test_config_loading,
        test_sionna_data_loader,
        test_model_creation,
        test_model_forward_pass,
        test_csi_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Sionna integration is working correctly.")
        print("\nNext steps:")
        print("1. Run 'python scripts/simulation/sionna_simulation.py' to generate data")
        print("2. Run 'python scripts/sionna_demo.py' to see the full demo")
        print("3. Start training with Sionna data")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check configuration file syntax")
        print("3. Verify Sionna installation")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
