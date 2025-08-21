# Sionna 5G OFDM Simulation for Prism

This directory contains NVIDIA Sionna-based 5G OFDM simulation scripts to generate realistic channel data for the Prism project.

## Quick Start

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

### 3. Run the Simulation
```bash
python scripts/sionna_simulation.py
```

## Simulation Specifications

- **Base Station**: 1 BS with 64 antennas (Massive MIMO)
- **User Equipment**: 100 UE positions with 4 antennas each
- **Bandwidth**: 100 MHz
- **Subcarriers**: 408
- **Carrier Frequency**: 3.5 GHz (mid-band 5G)
- **Channel Model**: UMi (Urban Microcell)
- **Coverage Area**: 500m × 500m

## Output Files

- **Data**: `data/sionna_5g_simulation.h5` (HDF5 format)
- **Visualization**: `data/sionna_simulation_results.png`
- **Channel Responses**: Shape `(100, 408, 4, 64)` - Complex matrices
- **Positions**: UE and BS coordinates
- **Path Losses**: Frequency-dependent attenuation

## Files Description

- `scripts/sionna_simulation.py` - Main simulation script
- `scripts/test_sionna_simulation.py` - Test script to verify setup
- `scripts/install_sionna.sh` - Installation helper script
- `requirements_sionna.txt` - Python dependencies
- `docs/sionna_simulation_guide.md` - Detailed documentation

## Requirements

- Python 3.8+
- NVIDIA Sionna
- PyTorch + TensorFlow
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

## Integration with Prism

The generated data can be used directly with Prism models:
1. Load HDF5 data using `h5py`
2. Extract channel responses and positions
3. Use as training data for neural networks
4. Compare with theoretical models

## Troubleshooting

- **Import errors**: Run `./scripts/install_sionna.sh`
- **CUDA issues**: Check GPU drivers and PyTorch/TensorFlow compatibility
- **Memory errors**: Reduce number of UE positions or use smaller FFT sizes

For detailed information, see `docs/sionna_simulation_guide.md`.
