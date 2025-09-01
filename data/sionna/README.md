# Sionna Data Generator

This directory contains scripts for generating synthetic 5G OFDM channel data for Prism training.

## Quick Start

### Basic Usage
```bash
# Generate data for 100 UE positions (most common)
python generator.py -n 100

# Generate data for 300 UE positions
python generator.py -n 300

# Generate data for 500 UE positions
python generator.py -n 500
```

### Custom Output Path
```bash
# Default behavior: creates Pxxx folder in current directory
python generator.py -n 100
# Creates: ./P100/ with data files inside

# Specify custom output directory
python generator.py -n 100 --output_path /path/to/custom/output

# Use relative path
python generator.py -n 100 --output_path ./my_simulation

# Use absolute path
python generator.py -n 100 --output_path /home/user/data/simulation
```

**Note**: If `--output_path` is not specified, the script automatically creates a folder named `Pxxx` (where xxx is the number of positions) relative to the script location. For example, running `python generator.py -n 100` creates a `P100` folder with all output files inside.

## Installation

### Install Dependencies
```bash
# Install Sionna and dependencies
bash install_sionna.sh

# Or install manually
pip install -r requirements_sionna.txt
```

## Command Line Options

### Required Arguments
- `-n, --num` : Number of UE positions to generate (required)

### Optional Arguments
- `--output_path` : Output directory (default: auto-generated `Pxxx` folder where xxx is number of positions)
- `--area_size` : Coverage area size in meters (default: 500)
- `--bs_height` : Base station height in meters (default: 25.0)
- `--center_frequency` : Center frequency in Hz (default: 3.5e9)
- `--num_subcarriers` : Number of OFDM subcarriers (default: 408)
- `--num_bs_antennas` : Number of BS antennas (default: 64)
- `--num_ue_antennas` : Number of UE antennas (default: 4)
- `--random_seed` : Random seed for reproducibility (default: 42)
- `--no_plots` : Disable visualization plots

### Examples
```bash
# Basic generation with 200 positions
python generator.py -n 200

# Custom area size and BS height
python generator.py -n 100 --area_size 1000 --bs_height 30

# Different frequency band (2.4 GHz)
python generator.py -n 100 --center_frequency 2.4e9

# More antennas configuration
python generator.py -n 100 --num_bs_antennas 128 --num_ue_antennas 8

# Disable plots for faster generation
python generator.py -n 100 --no_plots

# Custom seed for different random data
python generator.py -n 100 --random_seed 123
```

## Output Files

For each generation, the script creates:

### Data File
- **Filename**: `synthetic_5g_simulation_Pxxx.h5` (where xxx is number of positions)
- **Format**: HDF5 with the following structure:
  ```
  ├── simulation_config/     # Simulation parameters (stored as HDF5 attributes)
  │   ├── center_frequency   # Center frequency in Hz
  │   ├── bandwidth          # Total bandwidth in Hz
  │   ├── num_subcarriers    # Number of OFDM subcarriers
  │   ├── subcarrier_spacing # Subcarrier spacing in Hz
  │   ├── num_ue_antennas    # Number of UE antennas
  │   ├── num_bs_antennas    # Number of BS antennas
  │   ├── area_size          # Coverage area size in meters
  │   ├── bs_height          # Base station height in meters
  │   ├── shadowing_std      # Shadowing standard deviation in dB
  │   └── multipath_delay_spread # Multipath delay spread in seconds
  ├── positions/
  │   ├── bs_position       # Base station position (3,)
  │   └── ue_positions      # UE positions (N, 3)
  ├── channel_data/
  │   ├── channel_responses # Complex channel matrices (N, K, 4, 64)
  │   ├── path_losses      # Path losses in dB (N, K)
  │   └── delays           # Channel delays (N, K)
  └── metadata/            # Generation info and timestamps (stored as HDF5 attributes)
      ├── simulation_date   # ISO format timestamp
      ├── num_ue_positions  # Total number of UE positions
      ├── generator_version # Script version
      ├── channel_responses_shape # Shape of channel data
      ├── path_losses_shape # Shape of path loss data
      └── delays_shape      # Shape of delay data
  ```

### Visualization
- **Filename**: `synthetic_simulation_results.png`
- **Content**: 
  - UE and BS positions plot
  - Path loss distribution histogram
  - Channel magnitude and phase for first UE

## Configuration

### Default Parameters
```python
# Basic Parameters
num_positions: 300                    # Number of UE positions
output_path: 'Pxxx'                   # Auto-generated: Pxxx folder (xxx = num_positions)

# Area Configuration  
area_size: 500                        # Coverage area (500m × 500m)
bs_height: 25.0                       # Base station height
ue_height_min: 1.0                    # Minimum UE height
ue_height_max: 3.0                    # Maximum UE height

# 5G OFDM Parameters
center_frequency: 3.5e9               # 3.5 GHz (mid-band 5G)
subcarrier_spacing: 30e3              # 30 kHz subcarrier spacing
num_subcarriers: 408                  # Number of subcarriers
num_ue_antennas: 4                    # UE antennas (4×4 MIMO)
num_bs_antennas: 64                   # BS antennas (massive MIMO)

# Channel Model
shadowing_std: 8.0                    # Shadowing standard deviation (dB)
multipath_delay_spread: 50e-9         # Multipath delay spread (ns)

# Other
random_seed: 42                       # For reproducible results
create_plots: True                    # Generate visualization plots
plot_dpi: 300                         # Plot resolution
```

## Data Usage

### Loading Data in Python
```python
import h5py
import numpy as np

# Load generated data
with h5py.File('P100/synthetic_5g_simulation_P100.h5', 'r') as f:
    # Load channel data
    channel_responses = f['channel_data/channel_responses'][:]  # (100, 408, 4, 64)
    path_losses = f['channel_data/path_losses'][:]              # (100, 408)
    delays = f['channel_data/delays'][:]                        # (100, 408)
    
    # Load positions
    ue_positions = f['positions/ue_positions'][:]               # (100, 3)
    bs_position = f['positions/bs_position'][:]                 # (3,)
    
    # Load simulation configuration (stored as HDF5 attributes)
    config = dict(f['simulation_config'].attrs)
    print("Simulation Configuration:")
    print(f"  Center frequency: {config['center_frequency']/1e9:.1f} GHz")
    print(f"  Bandwidth: {config['bandwidth']/1e6:.1f} MHz")
    print(f"  Subcarriers: {config['num_subcarriers']}")
    print(f"  BS antennas: {config['num_bs_antennas']}")
    print(f"  UE antennas: {config['num_ue_antennas']}")
    print(f"  Coverage area: {config['area_size']} m")
    print(f"  BS height: {config['bs_height']} m")
    
    # Load metadata (also stored as HDF5 attributes)
    metadata = dict(f['metadata'].attrs)
    print(f"\nGeneration Info:")
    print(f"  Generated on: {metadata['simulation_date']}")
    print(f"  Generator version: {metadata['generator_version']}")
    print(f"  Total UE positions: {metadata['num_ue_positions']}")
    
    print(f"\nData Shapes:")
    print(f"  Channel responses: {channel_responses.shape}")
    print(f"  Path losses: {path_losses.shape}")
    print(f"  Delays: {delays.shape}")
```

### Accessing Specific Configuration Values
```python
# Access individual configuration parameters
with h5py.File('P100/synthetic_5g_simulation_P100.h5', 'r') as f:
    # Method 1: Access via attributes dictionary
    center_freq = f['simulation_config'].attrs['center_frequency']
    num_antennas = f['simulation_config'].attrs['num_bs_antennas']
    
    # Method 2: Get all config at once
    all_config = dict(f['simulation_config'].attrs)
    
    # Method 3: Check available configuration keys
    config_keys = list(f['simulation_config'].attrs.keys())
    print("Available configuration parameters:", config_keys)
    
    # Method 4: Access metadata
    generation_date = f['metadata'].attrs['simulation_date']
    data_shapes = {
        'channel_responses': f['metadata'].attrs['channel_responses_shape'],
        'path_losses': f['metadata'].attrs['path_losses_shape'],
        'delays': f['metadata'].attrs['delays_shape']
    }
```

### Integration with Prism
```python
# Use with Prism training pipeline
from prism import PrismDataLoader

# Load data for training
data_loader = PrismDataLoader('P100/synthetic_5g_simulation_P100.h5')
train_data, test_data = data_loader.split(train_ratio=0.8)
```

## Common Use Cases

### 1. Small Dataset (Testing)
```bash
python generator.py -n 50 --output_path ./test_data
```

### 2. Standard Dataset (Training)
```bash
python generator.py -n 300
```

### 3. Large Dataset (Production)
```bash
python generator.py -n 1000 --output_path ./large_dataset
```

### 4. Custom Scenario (Urban Dense)
```bash
python generator.py -n 200 --area_size 200 --bs_height 40 --num_bs_antennas 128
```

### 5. Different Frequency Band (mmWave)
```bash
python generator.py -n 100 --center_frequency 28e9 --area_size 100
```

## Performance Notes

- **Generation Time**: ~1-2 seconds per 100 positions
- **Memory Usage**: ~10MB per 100 positions
- **File Size**: ~50MB per 100 positions (HDF5 compressed)

## Troubleshooting

### Common Issues

1. **Import Error**: Install dependencies with `bash install_sionna.sh`
2. **Permission Error**: Check write permissions for output directory
3. **Memory Error**: Reduce number of positions or use `--no_plots`
4. **Slow Generation**: Use SSD storage and sufficient RAM

### Getting Help
```bash
# Show all available options
python generator.py --help

# Check script version and configuration
python generator.py -n 1 --no_plots  # Quick test run
```

## File Structure
```
data/sionna/
├── generator.py              # Main data generation script
├── install_sionna.sh         # Installation script
├── requirements_sionna.txt   # Python dependencies
├── README.md                # This file
├── P100/                    # Generated data (100 positions)
│   ├── synthetic_5g_simulation_P100.h5
│   └── synthetic_simulation_results.png
└── P300/                    # Generated data (300 positions)
    ├── synthetic_5g_simulation_P300.h5
    └── synthetic_simulation_results.png
```

---

**Note**: This generator creates synthetic data for development and testing. For production use, consider using real channel measurements or more sophisticated channel models.
