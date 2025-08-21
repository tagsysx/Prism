# Sionna 5G OFDM Integration Guide

This guide explains how to integrate NVIDIA Sionna-generated 5G OFDM simulation data with the Prism framework.

## Overview

The Sionna integration allows you to:
- Load Sionna-generated HDF5 simulation data
- Preprocess and normalize the data for neural network training
- Integrate with Prism models for 5G OFDM channel modeling
- Process CSI virtual links and perform advanced analysis
- Export processed data for further use

## Prerequisites

### 1. Install Sionna
```bash
# Install Sionna and dependencies
cd scripts/simulation
./install_sionna.sh

# Or manually install
pip install sionna
pip install h5py
```

### 2. Generate Sionna Data
```bash
# Run the 5G OFDM simulation
cd scripts/simulation
python sionna_simulation.py
```

This will create:
- `data/sionna_5g_simulation.h5` - HDF5 data file
- `data/sionna_simulation_results.png` - Visualization plots

## Configuration

### Update `configs/ofdm-5g-sionna.yml`

The configuration has been updated to support Sionna data:

```yaml
# Sionna Integration Configuration
sionna_integration:
  enabled: true
  
  # Sionna simulation parameters
  carrier_frequency: 3.5e9      # 3.5 GHz (mid-band 5G)
  bandwidth: 100e6              # 100 MHz bandwidth
  fft_size: 512                 # OFDM FFT size
  
  # Channel model configuration
  channel_model: 'UMi'          # Urban Microcell channel model
  enable_3d_antenna_patterns: true
  enable_o2i_modeling: true     # Outdoor-to-Indoor modeling
  
  # Data file structure
  hdf5_structure:
    channel_responses: 'channel_data/channel_responses'  # Shape: (100, 408, 4, 64)
    path_losses: 'channel_data/path_losses'             # Shape: (100, 408)
    delays: 'channel_data/delays'                       # Shape: (100, 408)
    ue_positions: 'positions/ue_positions'              # Shape: (100, 3)
    bs_position: 'positions/bs_position'                # Shape: (3,)
    simulation_config: 'simulation_config'               # Simulation parameters
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_subcarriers` | 408 | Number of OFDM subcarriers |
| `num_ue_antennas` | 4 | UE antenna count |
| `num_bs_antennas` | 64 | BS antenna count (Massive MIMO) |
| `data_dir` | `data/sionna_5g_simulation.h5` | Sionna HDF5 file path |
| `data_type` | `sionna_hdf5` | Data format identifier |

## Usage

### 1. Basic Data Loading

```python
from prism.utils.sionna_data_loader import SionnaDataLoader
import yaml

# Load configuration
with open('configs/ofdm-5g-sionna.yml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize data loader
data_loader = SionnaDataLoader(config)

# Get dataset statistics
stats = data_loader.get_statistics()
print(f"Dataset: {stats['num_samples']} samples, {stats['num_subcarriers']} subcarriers")
```

### 2. Data Splitting

```python
# Create train/validation/test splits
splits = data_loader.get_data_split(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Access training data
train_data = splits['train']
print(f"Training samples: {train_data['channel_responses'].shape}")
```

### 3. Batch Loading

```python
# Get batches for training
batch_size = 32
for batch_idx in range(num_batches):
    batch = data_loader.get_batch('train', batch_size, batch_idx)
    
    # Access batch data
    positions = batch['ue_positions']           # [batch_size, 3]
    channel_responses = batch['channel_responses']  # [batch_size, 408, 4, 64]
    path_losses = batch['path_losses']         # [batch_size, 408]
```

### 4. Model Integration

```python
from prism.model import create_prism_model

# Create Prism model
model = create_prism_model(config)

# Prepare input data
positions = batch['ue_positions']
ue_antennas = torch.randn(positions.shape[0], 4)      # UE antenna features
bs_antennas = torch.randn(positions.shape[0], 64)     # BS antenna features
additional_features = torch.randn(positions.shape[0], 10)

# Forward pass
outputs = model(
    positions=positions,
    ue_antennas=ue_antennas,
    bs_antennas=bs_antennas,
    additional_features=additional_features
)

print(f"Subcarrier responses: {outputs['subcarrier_responses'].shape}")
print(f"MIMO channel: {outputs['mimo_channel'].shape}")
```

## Data Structure

### HDF5 File Organization

```
sionna_5g_simulation.h5
├── simulation_config/          # Simulation parameters
│   ├── carrier_frequency
│   ├── bandwidth
│   ├── num_subcarriers
│   └── ...
├── positions/                  # Spatial coordinates
│   ├── bs_position            # [3,] - BS coordinates
│   └── ue_positions           # [100, 3] - UE coordinates
├── channel_data/              # Channel information
│   ├── channel_responses      # [100, 408, 4, 64] - Complex channel matrices
│   ├── path_losses            # [100, 408] - Frequency-dependent path loss
│   └── delays                 # [100, 408] - Channel delay information
└── metadata/                  # Additional information
    ├── simulation_date
    ├── sionna_version
    └── ...
```

### Data Shapes

| Data Type | Shape | Description |
|-----------|-------|-------------|
| `channel_responses` | `(100, 408, 4, 64)` | Complex channel matrices |
| `path_losses` | `(100, 408)` | Path loss per subcarrier |
| `delays` | `(100, 408)` | Channel delay per subcarrier |
| `ue_positions` | `(100, 3)` | UE 3D coordinates |
| `bs_position` | `(3,)` | BS 3D coordinates |

## Preprocessing Features

### 1. Frequency Normalization
- Normalizes channel responses across subcarriers
- Ensures consistent signal levels across frequency bands
- Applied per UE-BS antenna pair

### 2. Spatial Normalization
- Normalizes UE positions relative to BS
- Scales coordinates to unit range
- Places BS at origin (0, 0, 0)

### 3. Channel Enhancement
- Calculates channel quality metrics
- Computes SNR-like measurements
- Extracts magnitude and phase information

## CSI Virtual Link Processing

### Virtual Link Calculation
```python
# Reshape channel data to virtual links
# [batch_size, 408, 4, 64] -> [batch_size, 1632, 64]
virtual_links = channel_responses.view(batch_size, -1, num_bs_antennas)

# 1632 = 408 subcarriers × 4 UE antennas
```

### Link Quality Metrics
- **Link Strength**: Magnitude of channel response
- **Link Quality**: SNR-like metric (signal power / noise power)
- **Spatial Diversity**: Variation across BS antennas
- **Frequency Diversity**: Variation across subcarriers

## Running the Demo

### 1. Basic Demo
```bash
# Run the Sionna integration demo
python scripts/sionna_demo.py
```

### 2. Expected Output
```
=== Sionna Data Loading Demonstration ===
Initializing Sionna data loader...
Loading Sionna data from data/sionna_5g_simulation.h5...
Data preprocessing completed!

Dataset Statistics:
  num_samples: 100
  num_subcarriers: 408
  num_ue_antennas: 4
  num_bs_antennas: 64
  carrier_frequency: 3.5000 GHz
  bandwidth: 100.0000 MHz
```

### 3. Visualization
The demo creates comprehensive plots:
- UE and BS positions
- Channel magnitude distribution
- Path loss distribution
- Virtual link analysis
- Subcarrier response curves

## Advanced Usage

### 1. Custom Data Preprocessing
```python
# Disable specific preprocessing
config['sionna_integration']['enable_frequency_normalization'] = False
config['sionna_integration']['enable_spatial_normalization'] = False

# Reinitialize loader
data_loader = SionnaDataLoader(config)
```

### 2. Export Processed Data
```python
# Export to PyTorch format
data_loader.export_to_torch('data/processed_sionna_5g_data.pt')

# Load exported data
processed_data = torch.load('data/processed_sionna_5g_data.pt')
```

### 3. Custom HDF5 Structure
```python
# Modify HDF5 structure mapping
config['sionna_integration']['hdf5_structure'] = {
    'channel_responses': 'custom/path/to/channels',
    'path_losses': 'custom/path/to/losses',
    # ... other mappings
}
```

## Troubleshooting

### Common Issues

**1. File Not Found**
```bash
Error: Sionna data file not found: data/sionna_5g_simulation.h5
```
**Solution**: Run `python scripts/simulation/sionna_simulation.py` first

**2. Shape Mismatch**
```bash
Error: Shape mismatch for channel_responses: expected (100, 408, 4, 64), got (100, 1024, 2, 4)
```
**Solution**: Update configuration to match your Sionna simulation parameters

**3. Import Errors**
```bash
ModuleNotFoundError: No module named 'sionna'
```
**Solution**: Install Sionna using `./scripts/simulation/install_sionna.sh`

### Performance Optimization

**1. GPU Acceleration**
```python
# Enable GPU acceleration in config
config['sionna_integration']['enable_gpu_acceleration'] = True
```

**2. Memory Optimization**
```python
# Reduce batch size for large datasets
batch_size = 16  # Instead of 32 or 64
```

**3. Data Caching**
```python
# Export processed data to avoid reprocessing
data_loader.export_to_torch('cached_data.pt')
```

## Integration with Training

### 1. Training Loop
```python
# Create data splits
splits = data_loader.get_data_split()

# Training loop
for epoch in range(num_epochs):
    for batch_idx in range(num_batches):
        batch = data_loader.get_batch('train', batch_size, batch_idx)
        
        # Forward pass
        outputs = model(batch)
        
        # Calculate loss
        loss = loss_function(outputs, batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

### 2. Validation
```python
# Validation loop
model.eval()
with torch.no_grad():
    for batch_idx in range(val_batches):
        batch = data_loader.get_batch('val', batch_size, batch_idx)
        outputs = model(batch)
        
        # Calculate validation metrics
        val_loss = loss_function(outputs, batch)
```

## Next Steps

1. **Extend to Different Bands**: Modify Sionna simulation for different frequency bands
2. **Real-world Integration**: Compare with measured channel data
3. **Advanced Channel Models**: Implement more sophisticated channel models
4. **Performance Analysis**: Benchmark against theoretical models
5. **Multi-scenario Training**: Train on diverse deployment scenarios

## References

- [Sionna Documentation](https://nvlabs.github.io/sionna/)
- [5G NR Specifications](https://www.3gpp.org/specifications-technologies/5g-specifications)
- [Prism Framework Documentation](./README.md)
- [OFDM Fundamentals](./OFDM_BASICS.md)
