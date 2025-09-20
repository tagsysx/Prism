# PolyU Compacus WiFi Data Documentation

## Overview
`data.h5` is an optimized HDF5 format file containing PolyU Compacus WiFi simulation data. This file has been restructured and optimized for wireless communication research, featuring CSI (Channel State Information) data, UE positions, BS SSID information, and comprehensive metadata configuration.

## File Information
- **File Name**: `data.h5`
- **File Type**: Polyu WiFi
- **Version**: 2.0
- **Description**: PolyU Compacus WiFi data
- **Total Size**: ~7.53 MB
- **Data Points**: 913 positions

## File Structure
The HDF5 file contains two main groups:
- **`/metadata`**: Contains file metadata and configuration parameters
- **`/data`**: Contains the actual simulation data

### Complete Structure
```
📁 data.h5/
├── 📁 metadata/
│   ├── 📋 description: "PolyU Compacus WiFi data"
│   ├── 📋 file_type: "Polyu WiFi"
│   ├── 📋 version: "2.0"
│   └── 📁 config/ (16 attributes)
│       ├── 📋 bandwidth: 20000000
│       ├── 📋 bs_antenna_configuration: "1x1"
│       ├── 📋 bs_positions_description: "BS SSID data for each position"
│       ├── 📋 bs_positions_dimensions: "(position, bs_ssid)"
│       ├── 📋 bs_positions_shape: "(913, 1)"
│       ├── 📋 center_frequency: 2400000000
│       ├── 📋 csi_description: "Channel State Information (CSI) data"
│       ├── 📋 csi_dimensions: "(position, ue_antenna_index, bs_antenna_index, subcarrier_index)"
│       ├── 📋 csi_shape: "(913, 8, 1, 64)"
│       ├── 📋 num_samples: 913
│       ├── 📋 num_subcarriers: 64
│       ├── 📋 subcarrier_spacing: 312500
│       ├── 📋 ue_antenna_configuration: "2x4"
│       ├── 📋 ue_positions_description: "UE position coordinates (x, y, z)"
│       ├── 📋 ue_positions_dimensions: "(position, coordinates)"
│       └── 📋 ue_positions_shape: "(913, 3)"
│
└── 📁 data/
    ├── 📊 bs_positions: (913, 1) float64
    ├── 📊 csi: (913, 8, 1, 64) complex128
    └── 📊 ue_positions: (913, 3) float64
```

## Dataset Details

### 1. CSI Data (`/data/csi`)
- **Shape**: (913, 8, 1, 64)
- **Data Type**: complex128
- **Size**: 467,456 elements (~7.13 MB)
- **Dimensions**: 
  - **Position**: 913 UE positions
  - **UE Antenna Index**: 8 UE antennas
  - **BS Antenna Index**: 1 BS antenna
  - **Subcarrier Index**: 64 subcarriers
- **Description**: Channel State Information (CSI) data containing complex channel responses
- **Frequency**: 2.4 GHz center frequency
- **Bandwidth**: 20 MHz
- **Subcarrier Spacing**: 312.5 kHz

### 2. BS Positions (`/data/bs_positions`)
- **Shape**: (913, 1)
- **Data Type**: float64
- **Size**: 913 elements (~0.01 MB)
- **Dimensions**:
  - **Position**: 913 UE positions
  - **BS SSID**: BS SSID identifier for each position
- **Description**: BS SSID data for each position
- **Content**: AP ID values (0.0 - 2.0, 3 unique APs)

### 3. UE Positions (`/data/ue_positions`)
- **Shape**: (913, 3)
- **Data Type**: float64
- **Size**: 2,739 elements (~0.02 MB)
- **Dimensions**:
  - **Position**: 913 UE positions
  - **Coordinates**: (x, y, z) coordinates
- **Description**: UE position coordinates (x, y, z)
- **Content**: 
  - X coordinates: 679.44 - 894.97
  - Y coordinates: -2601.71 - -2329.68
  - Z coordinates: -24.88 - 61.37

## Configuration Parameters

### Frequency and Bandwidth
- **Center Frequency**: 2,400,000,000 Hz (2.4 GHz)
- **Bandwidth**: 20,000,000 Hz (20 MHz)
- **Subcarrier Spacing**: 312,500 Hz (312.5 kHz)

### Antenna Configuration
- **BS Antenna Configuration**: 1x1 (1 BS antenna)
- **UE Antenna Configuration**: 2x4 (8 UE antennas)

### Data Parameters
- **Number of Samples**: 913
- **Number of Subcarriers**: 64

## Data Access Examples

### Python Access
```python
import h5py
import numpy as np

# Open the HDF5 file
with h5py.File('data.h5', 'r') as f:
    # Access metadata
    metadata = dict(f['/metadata'].attrs)
    config = dict(f['/metadata/config'].attrs)
    
    print("File Description:", metadata['description'])
    print("File Type:", metadata['file_type'])
    print("Version:", metadata['version'])
    
    # Access datasets
    csi = f['/data/csi'][:]
    bs_positions = f['/data/bs_positions'][:]
    ue_positions = f['/data/ue_positions'][:]
    
    # Print shapes and sample data
    print("CSI Shape:", csi.shape)
    print("BS Positions Shape:", bs_positions.shape)
    print("UE Positions Shape:", ue_positions.shape)
    
    # Access CSI for specific position
    position_idx = 0
    csi_position = csi[position_idx, :, :, :]  # (8, 1, 64)
    
    # Access UE coordinates for specific position
    ue_coords = ue_positions[position_idx, :]  # [x, y, z]
    
    # Access BS SSID for specific position
    bs_ssid = bs_positions[position_idx, 0]  # scalar value
```

### CSI Data Usage
```python
# Get CSI for position i
csi_i = csi[i, :, :, :]  # (8, 1, 64)

# Get CSI for position i, UE antenna j
csi_ij = csi[i, j, :, :]  # (1, 64)

# Get CSI for position i, UE antenna j, subcarrier k
csi_ijk = csi[i, j, 0, k]  # complex number

# Get CSI magnitude for all positions
csi_magnitude = np.abs(csi)  # (913, 8, 1, 64)

# Get CSI phase for all positions
csi_phase = np.angle(csi)  # (913, 8, 1, 64)
```

### Position Data Usage
```python
# Get UE coordinates for position i
ue_coords_i = ue_positions[i, :]  # [x, y, z]

# Get BS SSID for position i
bs_ssid_i = bs_positions[i, 0]  # scalar value

# Get all unique BS SSIDs
unique_bs_ssids = np.unique(bs_positions.flatten())

# Get positions for specific BS SSID
bs_ssid_mask = bs_positions.flatten() == target_ssid
positions_for_bs = ue_positions[bs_ssid_mask]
```

## Data Statistics

### CSI Statistics
- **Magnitude Range**: 0.000000 - 47.539457
- **Magnitude Mean**: 9.290332
- **Magnitude Standard Deviation**: 6.781997
- **Data Size**: 7.13 MB

### BS Position Statistics
- **AP ID Range**: 0.0 - 2.0
- **Unique AP Count**: 3

### UE Position Statistics
- **X Coordinate Range**: 679.44 - 894.97
- **Y Coordinate Range**: -2601.71 - -2329.68
- **Z Coordinate Range**: -24.88 - 61.37

## File Optimization Features

### 1. Standardized Naming
- Uses CSI (Channel State Information) standard terminology
- Clear dimension descriptions for all datasets
- Self-documenting configuration parameters

### 2. Optimized Structure
- Separated BS SSID data from UE coordinates
- Eliminated redundant data dimensions
- Clean metadata organization

### 3. Comprehensive Documentation
- Complete dimension descriptions
- Data type specifications
- Usage examples and statistics

## Dependencies
- **Python**: 3.x
- **h5py**: For HDF5 file operations
- **numpy**: For numerical operations

## Installation
```bash
pip install h5py numpy
```

## Notes
- **Data Consistency**: This file has been verified to be consistent with the original `simulation_data.h5`
- **Optimization**: The file structure has been optimized for research use
- **Documentation**: All parameters are self-documenting with clear descriptions
- **Standards**: Uses industry-standard CSI terminology and HDF5 best practices

## Verification
A verification script is available at `.temp/verify_data_consistency.py` to ensure data integrity between `data.h5` and `simulation_data.h5`.

## File History
This file represents the final optimized version of PolyU Compacus WiFi data, incorporating:
1. CSI standard terminology
2. Optimized data structure
3. Comprehensive metadata
4. Complete dimension descriptions
5. Self-documenting configuration