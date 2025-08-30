#!/usr/bin/env python3
"""Test script to verify parallel processing configuration reading"""

import yaml
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_config_reading():
    """Test reading parallel processing configuration from config file"""
    
    print("🧪 Testing Parallel Processing Configuration Reading...")
    print("=" * 60)
    
    # Load configuration
    config_path = "configs/ofdm-5g-sionna.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Configuration loaded from: {config_path}")
    print()
    
    # Check performance section
    print("📊 Performance Section:")
    performance_config = config.get('performance', {})
    print(f"  • enable_parallel_processing: {performance_config.get('enable_parallel_processing', 'NOT_FOUND')}")
    print(f"  • num_workers: {performance_config.get('num_workers', 'NOT_FOUND')}")
    print(f"  • use_multiprocessing: {performance_config.get('use_multiprocessing', 'NOT_FOUND')}")
    print(f"  • enable_distributed: {performance_config.get('enable_distributed', 'NOT_FOUND')}")
    print()
    
    # Check ray_tracer_integration section
    print("🎯 Ray Tracer Integration Section:")
    ray_tracer_config = config.get('ray_tracer_integration', {})
    print(f"  • parallel_antenna_processing: {ray_tracer_config.get('parallel_antenna_processing', 'NOT_FOUND')}")
    print(f"  • num_workers: {ray_tracer_config.get('num_workers', 'NOT_FOUND')}")
    print(f"  • batch_processing: {ray_tracer_config.get('batch_processing', 'NOT_FOUND')}")
    print(f"  • cpu_offload: {ray_tracer_config.get('cpu_offload', 'NOT_FOUND')}")
    print()
    
    # Check ray_tracing section
    print("🌍 Ray Tracing Section:")
    ray_tracing_config = config.get('ray_tracing', {})
    print(f"  • enabled: {ray_tracing_config.get('enabled', 'NOT_FOUND')}")
    print(f"  • top_k_directions: {ray_tracing_config.get('top_k_directions', 'NOT_FOUND')}")
    print(f"  • azimuth_divisions: {ray_tracing_config.get('azimuth_divisions', 'NOT_FOUND')}")
    print(f"  • elevation_divisions: {ray_tracing_config.get('elevation_divisions', 'NOT_FOUND')}")
    print()
    
    # Simulate the configuration reading logic from train_prism.py
    print("🔧 Simulating Configuration Reading Logic:")
    
    # Parallel processing settings with fallback to config values
    enable_parallel = performance_config.get('enable_parallel_processing', True)
    max_workers = performance_config.get('num_workers', 4)
    use_multiprocessing = performance_config.get('use_multiprocessing', False)
    
    # Override with ray_tracer_integration settings if available
    if 'parallel_antenna_processing' in ray_tracer_config:
        enable_parallel = ray_tracer_config['parallel_antenna_processing']
    if 'num_workers' in ray_tracer_config:
        max_workers = ray_tracer_config['num_workers']
    
    print(f"  • Final enable_parallel_processing: {enable_parallel}")
    print(f"  • Final max_workers: {max_workers}")
    print(f"  • Final use_multiprocessing: {use_multiprocessing}")
    print()
    
    # Check if configuration is correct
    print("✅ Configuration Validation:")
    if enable_parallel:
        print(f"  ✅ Parallel processing is ENABLED")
    else:
        print(f"  ❌ Parallel processing is DISABLED")
    
    if max_workers >= 8:
        print(f"  ✅ Sufficient workers for antenna-level parallelization: {max_workers}")
    else:
        print(f"  ⚠️  Limited workers for parallelization: {max_workers}")
    
    if not use_multiprocessing:
        print(f"  ✅ Using threading (better compatibility)")
    else:
        print(f"  ⚠️  Using multiprocessing (may have compatibility issues)")
    
    print()
    print("🎉 Configuration test completed!")

if __name__ == "__main__":
    test_config_reading()
