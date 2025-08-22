# Sionna 5G OFDM Simulation Guide

This guide explains how to use the NVIDIA Sionna-based 5G OFDM simulation script to generate realistic channel data.

## Overview

The simulation script (`scripts/sionna_simulation.py`) creates a 5G communication scenario with:
- **1 Base Station (BS)** with 64 antennas
- **100 User Equipment (UE)** positions with 4 antennas each
- **100 MHz bandwidth** with 408 subcarriers
- **3.5 GHz carrier frequency** (mid-band 5G)

## Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for faster simulation)
- At least 8GB RAM

### Python Dependencies
Install the required packages:
```bash
pip install -r requirements_sionna.txt
```

**Note**: Sionna requires both PyTorch and TensorFlow. Make sure you have compatible versions installed.

## Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Carrier Frequency | 3.5 GHz | Mid-band 5G frequency |
| Bandwidth | 100 MHz | OFDM system bandwidth |
| Subcarriers | 408 | Number of OFDM subcarriers |
| BS Antennas | 64 | Massive MIMO base station |
| UE Antennas | 4 | User equipment antennas |
| FFT Size | 512 | OFDM FFT size |
| Deployment Area | 500m × 500m | Simulation coverage area |
| UE Height | 1.5-2.0m | Realistic UE positioning |
| BS Height | 25m | Typical BS tower height |

## Running the Simulation

### Basic Usage
```bash
cd /Users/Young/Projects/Prism
python scripts/sionna_simulation.py
```

### Expected Output
The script will:
1. Create a 500m × 500m deployment area
2. Position 1 BS at the center with 64 antennas
3. Generate 100 UE positions with minimum 50m distance from BS
4. Simulate channel responses for each UE position
5. Save results to HDF5 format
6. Generate visualization plots

### Output Files
- **Data**: `data/sionna_5g_simulation.h5`
- **Visualization**: `data/sionna_simulation_results.png`

## Data Structure

The HDF5 file contains the following data:

### Simulation Configuration
- Carrier frequency, bandwidth, subcarrier parameters
- Antenna configurations
- OFDM parameters

### Positions
- BS position (3D coordinates)
- UE positions (100 × 3 array)

### Channel Data
- **Channel Responses**: `(100, 408, 4, 64)` - Complex channel matrices
- **Path Losses**: `(100, 408)` - Frequency-dependent path loss
- **Delays**: `(100, 408)` - Channel delay information

### Metadata
- Simulation date and time
- Sionna version
- Dataset shapes and information

## Channel Model

The simulation uses Sionna's **UMi (Urban Microcell)** channel model with:
- 3GPP-3D antenna patterns
- Outdoor-to-indoor (O2I) modeling
- Frequency-dependent fading across subcarriers
- Realistic path loss and delay characteristics

## Visualization

The script generates four plots:
1. **UE and BS Positions**: Top-down view of deployment
2. **Path Loss Distribution**: Histogram of average path losses
3. **Channel Magnitude**: Subcarrier response magnitude for first UE
4. **Channel Phase**: Subcarrier response phase for first UE

## Customization

### Modifying Parameters
Edit the `create_simulation_scenario()` function to change:
- Carrier frequency
- Bandwidth
- Number of subcarriers
- Antenna configurations

### Changing UE Positions
Modify `generate_ue_positions()` to:
- Change deployment area size
- Adjust minimum distance from BS
- Modify UE height range
- Use different positioning strategies

### Channel Model Selection
Replace UMi with other Sionna models:
- **UMa**: Urban Macrocell (larger cells)
- **sub6GHz**: General sub-6GHz modeling
- **Custom**: Build your own channel model

## Performance Considerations

### Memory Usage
- **Channel responses**: ~200 MB for 100 UE positions
- **Total memory**: ~500 MB during simulation
- **Output file**: ~200-300 MB HDF5 file

### Simulation Time
- **Without GPU**: ~10-30 minutes
- **With GPU**: ~2-5 minutes
- **Progress tracking**: Updates every 10 UE positions

### Optimization Tips
1. Use GPU acceleration when available
2. Reduce number of UE positions for faster testing
3. Use smaller FFT sizes for development
4. Enable parallel processing for multiple simulations

## Troubleshooting

### Common Issues

**Sionna Import Error**
```bash
pip install sionna
# or
conda install -c conda-forge sionna
```

**CUDA Issues**
- Ensure CUDA toolkit is installed
- Check PyTorch/TensorFlow CUDA compatibility
- Use CPU-only mode if GPU unavailable

**Memory Errors**
- Reduce number of UE positions
- Use smaller FFT sizes
- Close other applications

### Getting Help
- Check Sionna documentation: https://nvlabs.github.io/sionna/
- Verify CUDA installation: `nvidia-smi`
- Check Python environment: `python --version`

## Integration with Prism

The generated data can be used with the Prism framework:
1. Load HDF5 data using `h5py`
2. Extract channel responses and positions
3. Use as training data for neural network models
4. Compare with theoretical channel models

## Example Data Loading

```python
import h5py

# Load simulation data
with h5py.File('data/sionna_5g_simulation.h5', 'r') as f:
    # Extract channel responses
    channel_responses = f['channel_data/channel_responses'][:]
    
    # Extract positions
    ue_positions = f['positions/ue_positions'][:]
    bs_position = f['positions/bs_position'][:]
    
    # Extract configuration
    config = dict(f['simulation_config'].attrs)

print(f"Channel shape: {channel_responses.shape}")
print(f"UE positions: {ue_positions.shape}")
```

## Next Steps

After running the simulation:
1. Analyze the generated data structure
2. Visualize channel characteristics
3. Use data for Prism model training
4. Compare with theoretical models
5. Extend simulation for different scenarios
