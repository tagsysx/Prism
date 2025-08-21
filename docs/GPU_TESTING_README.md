# GPU Ray Tracing Testing Guide

## Overview

This document explains how to test the GPU-accelerated ray tracing functionality to ensure the CUDA implementation is working correctly.

## Test Files

### 1. **Comprehensive Test Suite** (`tests/test_gpu_ray_tracer.py`)
- Full pytest-based test suite
- Tests all GPU ray tracing components
- Includes performance benchmarking
- Virtual link ray tracing tests

### 2. **Simple Test Script** (`test_gpu_ray_tracing.py`)
- Quick verification without pytest
- Standalone script for basic testing
- Performance comparison tests
- Easy to run and debug

## Prerequisites

### CUDA Requirements
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Required packages
torch>=1.9.0
numpy>=1.21.0
pytest>=6.0.0  # For comprehensive tests
```

### GPU Requirements
- **NVIDIA GPU**: Compute Capability 7.0+
- **GPU Memory**: 2GB+ for testing
- **Driver Version**: CUDA 11.0+ compatible

## Running Tests

### Option 1: Simple Test Script (Recommended for Quick Testing)

```bash
# Run the simple test script
python test_gpu_ray_tracing.py
```

**Output Example:**
```
🧪 GPU Ray Tracing Test Suite
============================================================

==================== Basic GPU Ray Tracing ====================
🚀 Testing GPU Ray Tracing...
==================================================
✅ CUDA available: NVIDIA GeForce RTX 3080
✅ CUDA version: 11.8
✅ GPU Memory: 10.0 GB

✅ GPU ray tracing modules imported successfully
🔧 Creating GPU Ray Tracer...
✅ GPU Ray Tracer created on cuda
🏗️ Creating test environment...
✅ Test environment created with obstacles
📍 Creating test source positions...
✅ Created 10 source positions on cuda
📡 Testing ray generation...
✅ Generated 72 rays from single source
📦 Testing batch ray generation: torch.Size([10, 72, 3]) origins, torch.Size([10, 72, 3]) directions
🔍 Testing basic ray tracing...
✅ Basic ray tracing completed: 5 ray paths
📦 Testing batch ray tracing...
✅ Batch ray tracing completed: 5 ray paths
⚡ Testing GPU optimized ray tracing...
✅ GPU optimized tracing completed: 360 ray paths
✅ GPU Memory Used: 0.15 GB
🗺️ Testing spatial analysis...
✅ Spatial analysis completed: 360 total points
📊 Testing channel estimation...
✅ Channel estimation completed: torch.Size([5]) path losses
📈 Testing performance metrics...
✅ Performance metrics: NVIDIA GeForce RTX 3080
✅ GPU Memory: 0.15 GB allocated
✅ Batch Size: 64
✅ Mixed Precision: True
🧹 Testing memory optimization...
✅ Memory optimization: 0.15 GB → 0.12 GB

🎉 All GPU ray tracing tests passed successfully!

==================== Virtual Link Ray Tracing ====================
🔗 Testing Virtual Link Ray Tracing...
==================================================
✅ GPU Ray Tracer created for virtual links
✅ Urban environment created
✅ Created 20 UE positions
🔍 Tracing rays for virtual links...
✅ Virtual link ray tracing completed: 1440 rays
✅ Expected rays: 1440
✅ GPU Memory Used: 0.18 GB
✅ Total virtual links: 128
✅ Sampled virtual links: 32
✅ Sampling ratio: 25.0%

🎉 Virtual link ray tracing test passed!

==================== Performance Comparison ====================
⚡ Testing Performance Comparison...
==================================================
📊 Benchmarking GPU performance...
✅ GPU Performance: 1280 rays/sec
✅ GPU Time: 0.025 seconds
✅ GPU Total Rays: 320
📊 Benchmarking CPU performance...
✅ CPU Performance: 64 rays/sec
✅ CPU Time: 5.000 seconds
✅ CPU Total Rays: 320
✅ GPU Speedup: 20.0x

🎉 Performance comparison test passed!

============================================================
📋 TEST SUMMARY
============================================================
✅ PASS Basic GPU Ray Tracing
✅ PASS Virtual Link Ray Tracing
✅ PASS Performance Comparison

📊 Results: 3/3 tests passed

🎉 All tests passed! GPU ray tracing is working correctly.
```

### Option 2: Comprehensive Pytest Suite

```bash
# Install pytest if not already installed
pip install pytest

# Run all GPU tests
pytest tests/test_gpu_ray_tracer.py -v

# Run specific test class
pytest tests/test_gpu_ray_tracer.py::TestGPURayTracer -v

# Run specific test method
pytest tests/test_gpu_ray_tracer.py::TestGPURayTracer::test_gpu_ray_tracer_creation -v

# Run with detailed output
pytest tests/test_gpu_ray_tracer.py -v -s --tb=short
```

## Test Categories

### 1. **Basic GPU Ray Tracing Tests**
- GPU ray tracer creation
- Ray generation (single and batch)
- Basic ray tracing
- Batch ray tracing
- GPU optimized ray tracing

### 2. **Advanced Functionality Tests**
- Spatial analysis
- Channel estimation
- Performance metrics
- Device management
- Memory optimization

### 3. **Performance Tests**
- Performance benchmarking
- CPU vs GPU comparison
- Speedup calculation

### 4. **Virtual Link Tests**
- Virtual link ray tracing
- Sampling verification
- Memory usage monitoring

## Test Configuration

### Test Environment Settings
```python
config = {
    'ray_tracing': {
        'gpu_acceleration': True,
        'batch_size': 64,           # Smaller for testing
        'azimuth_samples': 12,      # Reduced for testing
        'elevation_samples': 6,     # Reduced for testing
        'points_per_ray': 32,       # Reduced for testing
        'gpu_memory_fraction': 0.5, # Use less memory for testing
        'mixed_precision': True
    }
}
```

### Test Data Sizes
- **Source positions**: 10-20 positions (small for testing)
- **Angular resolution**: 12×6 angles (reduced from 36×18)
- **Spatial points**: 32 points per ray (reduced from 64)
- **Batch size**: 64 (reduced from 512)

## Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**
```bash
# Reduce test parameters
config = {
    'ray_tracing': {
        'batch_size': 32,           # Reduce from 64
        'azimuth_samples': 8,       # Reduce from 12
        'elevation_samples': 4,     # Reduce from 6
        'gpu_memory_fraction': 0.3  # Reduce from 0.5
    }
}
```

#### 2. **Import Errors**
```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or add to script
sys.path.append(str(Path(__file__).parent / 'src'))
```

#### 3. **Device Mismatch Errors**
```python
# Ensure all tensors are on the same device
source_positions = source_positions.to('cuda')
environment = environment.to_device('cuda')

# Check device consistency
print(f"Source device: {source_positions.device}")
print(f"Environment device: {environment.device}")
```

#### 4. **Performance Test Failures**
```python
# Increase number of runs for more stable results
gpu_results = benchmark_ray_tracing_performance(gpu_tracer, source_positions, env, num_runs=5)

# Use larger test data for more accurate measurements
source_positions = torch.randn(50, 3, device='cuda')  # Increase from 10
```

## Expected Results

### Performance Benchmarks
- **GPU Performance**: 1000-5000+ rays/sec (depending on GPU)
- **CPU Performance**: 50-200 rays/sec (depending on CPU)
- **GPU Speedup**: 10-100x (depending on hardware)

### Memory Usage
- **GPU Memory**: 0.1-1.0 GB (depending on test size)
- **Memory Efficiency**: Should use <50% of available GPU memory

### Accuracy
- **Ray Count**: Should match expected calculations
- **Virtual Links**: Should correctly sample K=64 from total
- **Device Consistency**: All operations should use correct device

## Advanced Testing

### Custom Test Scenarios
```python
# Test with different materials
config['ray_tracing']['materials'] = {
    'glass': {'permittivity': 3.8, 'conductivity': 0.0},
    'metal': {'permittivity': 1.0, 'conductivity': 1e7}
}

# Test with different environments
env = Environment(device='cuda')
env.add_obstacle(Building([-100, -100, 0], [100, 100, 100], 'concrete', device='cuda'))

# Test with different UE configurations
config['csi_processing'] = {
    'm_subcarriers': 1024,
    'n_ue_antennas': 8,
    'sample_size': 128
}
```

### Stress Testing
```python
# Large-scale testing
source_positions = torch.randn(1000, 3, device='cuda')
config['ray_tracing']['batch_size'] = 256
config['ray_tracing']['azimuth_samples'] = 36
config['ray_tracing']['elevation_samples'] = 18
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: GPU Ray Tracing Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install torch torchvision
        pip install pytest numpy
    - name: Run GPU tests
      run: |
        python test_gpu_ray_tracing.py
```

## Conclusion

The GPU ray tracing tests provide comprehensive verification of:

- ✅ **CUDA Implementation**: Full GPU acceleration support
- ✅ **Performance**: Significant speedup over CPU
- ✅ **Functionality**: All ray tracing features working
- ✅ **Memory Management**: Efficient GPU memory usage
- ✅ **Virtual Links**: Correct sampling and processing
- ✅ **Device Management**: Proper CUDA device handling

Run the tests regularly to ensure GPU ray tracing continues to work correctly as the codebase evolves.
