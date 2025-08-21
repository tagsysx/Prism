# Simulation Directory

This directory contains all simulation-related scripts, configurations, and documentation for the Prism project using NVIDIA Sionna.

## 📁 Directory Structure

```
scripts/simulation/
├── README.md                           # This file - Main simulation overview
├── requirements_sionna.txt             # Python dependencies for Sionna
├── install_sionna.sh                  # Automated installation script
├── test_sionna_simulation.py          # Test script to verify setup
├── sionna_simulation.py               # Generic 5G OFDM simulation
├── sionna_simulation_china_mobile_n41.py  # China Mobile n41 band simulation
├── README_SIONNA.md                   # General Sionna simulation guide
├── README_CHINA_MOBILE_N41.md         # n41 band specific documentation
└── sionna_simulation_guide.md         # Detailed technical guide
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd scripts/simulation
./install_sionna.sh
```

### 2. Test the Setup
```bash
python test_sionna_simulation.py
```

### 3. Run Simulations

#### Generic 5G Simulation
```bash
python sionna_simulation.py
```



## 📋 Available Simulations

### **Generic 5G OFDM Simulation** (`sionna_simulation.py`)
- **Frequency**: 3.5 GHz (mid-band 5G)
- **Bandwidth**: 100 MHz
- **Subcarriers**: 408
- **Antennas**: 64 BS, 4 UE
- **Coverage**: 500m × 500m
- **Use Case**: General 5G research and development



## 🔧 Configuration Files

The simulations use configuration files located in the `configs/` directory:
- `configs/ofdm-wifi.yml` - WiFi-like OFDM configuration
- `configs/ofdm-wideband.yml` - Ultra-wideband OFDM configuration
- `configs/china-mobile-n41.yml` - China Mobile n41 band configuration

## 📊 Output Data

### **Data Files**
- **Generic 5G**: `data/sionna_5g_simulation.h5`


### **Visualizations**
- **Generic 5G**: `data/sionna_simulation_results.png`


### **Data Structure**
```
Channel Responses: (100, N, 4, 64) - Complex matrices
Path Losses:      (100, N)          - Frequency-dependent attenuation
Delays:           (100, N)          - Channel delay information
Positions:        (100, 3)          - UE and BS coordinates
```

Where `N` is the number of subcarriers (408 for generic 5G).

## 🎯 Key Features

### **Common Features**
- NVIDIA Sionna-based channel modeling
- Realistic urban environment simulation
- Massive MIMO support (64×4 antenna configuration)
- OFDM with configurable subcarriers
- HDF5 data export for easy integration
- Comprehensive visualization plots



## 🔄 Integration with Prism

### **Training with Generated Data**
```bash
# Train with generic 5G data
python ../prism_runner.py --mode train --config ../../configs/ofdm-wideband.yml


```

### **Data Loading Example**
```python
import h5py

# Load simulation data
with h5py.File('data/sionna_5g_simulation.h5', 'r') as f:
    channel_responses = f['channel_data/channel_responses'][:]
    ue_positions = f['positions/ue_positions'][:]
    bs_position = f['positions/bs_position'][:]
```

## 🛠️ Customization

### **Modifying Simulation Parameters**
1. Edit the simulation script files directly
2. Modify configuration files in `configs/` directory
3. Adjust channel model parameters in the scripts
4. Change deployment area and UE positioning

### **Adding New Bands**
1. Copy an existing simulation script
2. Update frequency, bandwidth, and subcarrier parameters
3. Modify channel model characteristics
4. Update visualization and analysis functions

## 🔍 Troubleshooting

### **Common Issues**
- **Import errors**: Run `./install_sionna.sh`
- **CUDA issues**: Check GPU compatibility
- **Memory errors**: Reduce number of UE positions
- **Simulation time**: Use GPU acceleration

### **Getting Help**
- Check individual README files for specific simulations
- Review `sionna_simulation_guide.md` for technical details
- Consult Sionna documentation: https://nvlabs.github.io/sionna/
- Check configuration files for parameter explanations

## 📚 Documentation

- **`README_SIONNA.md`**: General Sionna simulation overview

- **`sionna_simulation_guide.md`**: Comprehensive technical guide
- **Configuration files**: Detailed parameter explanations

## 🚀 Next Steps

1. **Run Simulations**: Execute the simulation scripts to generate data
2. **Analyze Results**: Review channel characteristics and performance metrics
3. **Train Models**: Use generated data with Prism neural networks
4. **Extend Scenarios**: Create additional frequency band configurations
5. **Performance Analysis**: Evaluate system capacity and coverage

---

**Note**: All simulations are designed to work with the Prism framework and generate realistic channel data for neural network training and analysis.
