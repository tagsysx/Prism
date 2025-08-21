# China Mobile n41 Band (2.5 GHz) Simulation for Prism

This directory contains NVIDIA Sionna-based 5G OFDM simulation scripts specifically configured for **China Mobile's n41 band** deployment.

## 🎯 n41 Band Specifications

### **Frequency & Bandwidth**
- **Band**: n41 (2.5 GHz)
- **Carrier Frequency**: 2.5 GHz
- **Bandwidth**: 100 MHz
- **Subcarrier Spacing**: 30 kHz
- **Number of Subcarriers**: 273
- **Cyclic Prefix**: 7%

### **Antenna Configuration**
- **Base Station**: 64 antennas (Massive MIMO)
- **User Equipment**: 4 antennas
- **MIMO Type**: Spatial Multiplexing
- **Precoding**: Zero Forcing
- **Detection**: Maximum Likelihood

### **Coverage & Environment**
- **Deployment Type**: Urban Microcell
- **Coverage Area**: 400m × 400m
- **BS Height**: 30m (urban deployment)
- **UE Height**: 1.5-2.0m
- **Channel Model**: UMi (Urban Microcell)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Run the installation script
./scripts/install_sionna.sh

# Or manually activate virtual environment and install
source venv/bin/activate
pip install -r requirements_sionna.txt
```

### 2. Test the Setup
```bash
python scripts/test_sionna_simulation.py
```

### 3. Run n41 Band Simulation
```bash
# Run China Mobile n41 band simulation
python scripts/sionna_simulation_china_mobile_n41.py

# Or use the configuration file
python scripts/prism_runner.py --mode train --config configs/china-mobile-n41.yml
```

## 📊 Why n41 Band?

### **Advantages over 3.5 GHz (n78)**
1. **Better Building Penetration**: 2.5 GHz has lower penetration loss than 3.5 GHz
2. **Lower Path Loss**: Reduced free-space path loss in urban environments
3. **Urban Coverage**: Optimized for dense urban deployment scenarios
4. **Balanced Performance**: Good trade-off between coverage and capacity

### **China Mobile Deployment Strategy**
- **Primary Use**: Urban microcell deployment
- **Coverage Enhancement**: Indoor and outdoor coverage optimization
- **Capacity Management**: High-density user scenarios
- **Network Planning**: Strategic placement in urban centers

## 🔧 Technical Details

### **OFDM Parameters**
```yaml
# n41 band specific configuration
carrier_frequency: 2.5 GHz
bandwidth: 100 MHz
subcarriers: 273
subcarrier_spacing: 30 kHz
cyclic_prefix: 7%
pilot_density: 10%
```

### **Channel Characteristics**
- **Path Loss Exponent**: 3.5 (urban environment)
- **Shadowing Standard Deviation**: 8 dB
- **RMS Delay Spread**: 0.5 μs
- **Maximum Doppler Shift**: 50 Hz
- **Multipath Components**: 6 (urban environment)

### **Simulation Features**
- **Realistic Urban Environment**: UMi channel model with 3GPP-3D antenna patterns
- **Frequency-Dependent Fading**: Subcarrier-specific channel responses
- **SNR & Capacity Analysis**: Performance metrics for each UE position
- **Comprehensive Visualization**: 6-panel analysis plots

## 📁 Output Files

### **Data Files**
- **HDF5 Dataset**: `data/china_mobile_n41_simulation.h5`
- **Configuration**: `configs/china-mobile-n41.yml`
- **Visualization**: `data/china_mobile_n41_simulation_results.png`

### **Data Structure**
```
Channel Responses: (100, 273, 4, 64) - Complex matrices
Path Losses:      (100, 273)          - Frequency-dependent attenuation
Delays:           (100, 273)          - Channel delay information
SNR Values:       (100, 273)          - Signal-to-noise ratios
Capacity Values:  (100, 273)          - Shannon capacity per subcarrier
UE Positions:     (100, 3)            - 3D coordinates
```

## 🎨 Visualization Features

The simulation generates comprehensive visualizations:

1. **UE and BS Positions**: Top-down view of n41 band deployment
2. **Path Loss Distribution**: Histogram of average path losses
3. **SNR Distribution**: Signal-to-noise ratio analysis
4. **Channel Magnitude**: Subcarrier response magnitude for first UE
5. **Channel Capacity**: Shannon capacity across subcarriers
6. **Capacity Distribution**: Overall system capacity analysis

## 🔄 Integration with Prism

### **Training Configuration**
```bash
# Train Prism model with n41 band data
python scripts/prism_runner.py --mode train --config configs/china-mobile-n41.yml
```

### **Data Loading Example**
```python
import h5py

# Load n41 band simulation data
with h5py.File('data/china_mobile_n41_simulation.h5', 'r') as f:
    # Extract channel responses
    channel_responses = f['channel_data/channel_responses'][:]
    
    # Extract positions
    ue_positions = f['positions/ue_positions'][:]
    bs_position = f['positions/bs_position'][:]
    
    # Extract n41 band metrics
    snr_values = f['n41_metrics/snr_values'][:]
    capacity_values = f['n41_metrics/capacity_values'][:]
    
    # Extract configuration
    config = dict(f['simulation_config'].attrs)

print(f"n41 Band: {config['band']}")
print(f"Frequency: {config['carrier_frequency']/1e9:.1f} GHz")
print(f"Subcarriers: {config['num_subcarriers']}")
```

## 📈 Performance Characteristics

### **n41 Band vs Generic 5G**
| Metric | n41 Band (2.5 GHz) | Generic 5G (3.5 GHz) |
|--------|---------------------|----------------------|
| **Building Penetration** | Better | Lower |
| **Urban Coverage** | 400m × 400m | 500m × 500m |
| **Path Loss** | Lower | Higher |
| **Subcarriers** | 273 | 408 |
| **Subcarrier Spacing** | 30 kHz | 30 kHz |
| **Typical Use** | Urban Microcell | Mid-band 5G |

### **Expected Results**
- **Coverage**: 20% better building penetration than 3.5 GHz
- **Capacity**: Optimized for urban dense deployment
- **Reliability**: Lower path loss variation across subcarriers
- **Efficiency**: Better performance in urban environments

## 🛠️ Customization

### **Modifying Parameters**
Edit `scripts/sionna_simulation_china_mobile_n41.py`:
- Change carrier frequency within n41 band range
- Adjust coverage area dimensions
- Modify UE positioning constraints
- Update channel model parameters

### **Configuration File**
Modify `configs/china-mobile-n41.yml`:
- Adjust model architecture
- Change training parameters
- Update OFDM settings
- Modify environment characteristics

## 🔍 Troubleshooting

### **Common Issues**
- **Import errors**: Ensure Sionna is properly installed
- **Memory issues**: Reduce batch size for 64 BS antennas
- **CUDA problems**: Check GPU compatibility with PyTorch/TensorFlow
- **Simulation time**: Use GPU acceleration for faster results

### **Performance Tips**
1. **GPU Acceleration**: Use CUDA-compatible GPU for faster simulation
2. **Memory Management**: Monitor memory usage with 64 BS antennas
3. **Batch Processing**: Use appropriate batch sizes for your hardware
4. **Parallel Processing**: Consider running multiple simulations in parallel

## 📚 References

### **3GPP Standards**
- **TS 38.101-1**: User Equipment (UE) radio transmission and reception
- **TS 38.211**: Physical channels and modulation
- **TS 38.214**: Physical layer procedures for data

### **China Mobile Specifications**
- **n41 Band Allocation**: 2.5 GHz frequency range
- **Deployment Guidelines**: Urban microcell specifications
- **Performance Requirements**: Coverage and capacity targets

### **Technical Resources**
- **Sionna Documentation**: https://nvlabs.github.io/sionna/
- **5G NR Standards**: 3GPP specifications
- **China Mobile Technical Papers**: Operator-specific research

## 🚀 Next Steps

After running the n41 band simulation:

1. **Analyze Results**: Review channel characteristics and performance metrics
2. **Train Prism Models**: Use generated data for neural network training
3. **Compare with Theory**: Validate against 3GPP channel models
4. **Extend Scenarios**: Create additional n41 band configurations
5. **Performance Analysis**: Evaluate system capacity and coverage

## 📞 Support

For questions about the n41 band simulation:
- Check the detailed guide: `docs/sionna_simulation_guide.md`
- Review configuration options in `configs/china-mobile-n41.yml`
- Test basic functionality with `scripts/test_sionna_simulation.py`
- Consult Sionna documentation for advanced features

---

**Note**: This simulation is specifically optimized for China Mobile's n41 band deployment characteristics and may require adjustments for other operators or frequency bands.
