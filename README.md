# Prism: Wideband RF Neural Radiance Fields for OFDM Communication

## Overview

Prism extends the NeRF2 architecture to handle wideband RF signals in Orthogonal Frequency-Division Multiplexing (OFDM) scenarios. Unlike the original NeRF2 system which was limited to narrowband signals, Prism introduces the **RF Prism Module** to decompose global features into distinct subcarrier components, enabling precise modeling of frequency-dependent RF signal behaviors.

## Key Features

- **Wideband RF Support**: Handles thousands of subcarriers (e.g., 1024 subcarriers in ultra-wideband scenarios)
- **Multi-Antenna MIMO**: Supports UE with N_UE antennas and Base Station with N_BS antennas
- **Frequency-Aware Processing**: RF Prism Module with C channels for C subcarriers
- **OFDM-Optimized**: Specifically designed for orthogonal frequency-division multiplexing systems
- **Independent Subcarrier Training**: Ray marching with independent error back-propagation per subcarrier

## Architecture

### RF Prism Module
The core innovation of Prism is the RF Prism Module, a multi-channel MLP that:
- Decomposes global features from attenuation and radiance networks
- Processes each subcarrier independently with 256-dimensional layers
- Maintains modulation characteristics across the frequency spectrum
- Enables precise interference management and signal optimization

### Network Structure
```
Input: RF Signal + Position → Attenuation Network → Radiance Network → RF Prism Module → C Subcarrier Outputs
```

## Project Structure

```
Prism/
├── 📁 src/prism/                     # Main package source code
│   ├── 📄 model.py                  # Core model architecture
│   ├── 📄 dataloader.py             # Data loading and processing
│   ├── 📄 renderer.py               # Visualization and rendering
│   └── 📁 utils/                    # OFDM signal processing utilities
│
├── 📁 scripts/                       # Executable scripts
│   ├── 📄 prism_runner.py           # Main training/testing runner
│   ├── 📄 basic_usage.py            # Basic usage demonstration
│   └── 📁 simulation/               # Sionna-based 5G simulations
│       ├── 📄 sionna_simulation.py  # Generic 5G OFDM simulation
│       ├── 📄 sionna_simulation_china_mobile_n41.py  # China Mobile n41 band
│       └── 📄 README.md             # Simulation overview
│
├── 📁 configs/                       # Configuration files
│   ├── 📄 ofdm-wideband.yml         # 1024 subcarriers configuration
│   ├── 📄 ofdm-wifi.yml             # 52 subcarriers configuration
│   └── 📄 china-mobile-n41.yml      # China Mobile n41 band configuration
│
├── 📁 docs/                          # Documentation
├── 📁 tests/                         # Test suite
├── 📁 data/                          # Data directory
├── 📁 checkpoints/                   # Model checkpoints
├── 📁 results/                       # Output results
└── 📁 logs/                          # Training logs
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (for GPU acceleration)
- NumPy, SciPy, Matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/tagsysx/Prism.git
cd Prism

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install with development tools
pip install -e ".[dev]"
```

## Usage

### Training

```bash
# Train for OFDM scenario with C subcarriers
python scripts/prism_runner.py --mode train --config configs/ofdm-wideband.yml --dataset_type ofdm --gpu 0

# Train for WiFi-like scenario with 52 subcarriers
python scripts/prism_runner.py --mode train --config configs/ofdm-wifi.yml --dataset_type ofdm --gpu 0
```

### Inference

```bash
# Test the trained model
python scripts/prism_runner.py --mode test --config configs/ofdm-wideband.yml --dataset_type ofdm --gpu 0 --checkpoint path/to/checkpoint
```

### Basic Example

```bash
# Run basic demonstration
python scripts/basic_usage.py
```

### Simulations

```bash
# Run 5G OFDM simulations with Sionna
cd scripts/simulation

# Install Sionna dependencies
./install_sionna.sh

# Test setup
python test_sionna_simulation.py

# Run generic 5G simulation
python sionna_simulation.py

# Run China Mobile n41 band simulation
python sionna_simulation_china_mobile_n41.py
```

For detailed simulation documentation, see `scripts/simulation/README.md`.

## Configuration

The system supports various OFDM configurations:
- **Subcarrier Count**: Configurable from 52 (WiFi) to 1024+ (ultra-wideband)
- **Antenna Configuration**: Flexible N_UE × N_BS MIMO setups
- **Frequency Bands**: Adaptable to different communication standards

## Loss Function

The training employs a frequency-aware loss function:

```
L = |h₁ - h̃₁|² + |h₂ - h̃₂|² + ... + |h_C - h̃_C|²
```

Where each subcarrier's error is computed independently, preserving orthogonality and enabling tailored parameter adjustments.

## Applications

- **5G/6G MIMO Channel Prediction**: Ultra-wideband channel modeling
- **WiFi CSI Enhancement**: Multi-carrier signal optimization
- **Indoor Localization**: Frequency-dependent positioning accuracy
- **Interference Management**: Subcarrier-specific optimization
- **Multi-Path Analysis**: Frequency-dependent propagation modeling

## Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src/prism
```

### Code Quality
```bash
# Format code
make format

# Lint code
make lint

# Run full development cycle
make dev-cycle
```

### Building
```bash
# Build package
make build

# Clean build artifacts
make clean
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{prism2024,
    title={Prism: Wideband RF Neural Radiance Fields for OFDM Communication},
    author={Your Name},
    journal={arXiv preprint},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on NeRF2: Neural Radio-Frequency Radiance Fields
- Inspired by the need for wideband RF signal processing
- Built for next-generation communication systems

## Contributing

We welcome contributions! Please see our contributing guidelines for more details.

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.
