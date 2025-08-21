# Prism: Wideband RF Neural Radiance Fields for OFDM Communication

## Overview

Prism extends the NeRF2 architecture to handle wideband RF signals in Orthogonal Frequency-Division Multiplexing (OFDM) scenarios. Unlike the original NeRF2 system which was limited to narrowband signals, Prism introduces the **RF Prism Module** to decompose global features into distinct subcarrier components, enabling precise modeling of frequency-dependent RF signal behaviors.

## Key Features

- **Wideband RF Support**: Handles thousands of subcarriers (e.g., 1024 subcarriers in ultra-wideband scenarios)
- **Multi-Antenna MIMO**: Supports UE with N_UE antennas and Base Station with N_BS antennas
- **Frequency-Aware Processing**: RF Prism Module with C channels for C subcarriers
- **OFDM-Optimized**: Specifically designed for orthogonal frequency-division multiplexing systems
- **Independent Subcarrier Training**: Ray marching with independent error back-propagation per subcarrier
- **CSI Virtual Link Modeling**: Treats M×N_UE uplink channel combinations as virtual links for enhanced channel modeling
- **Advanced Ray Tracing**: Comprehensive ray tracing with configurable azimuth/elevation sampling and spatial point sampling

## Architecture

### RF Prism Module
The core innovation of Prism is the RF Prism Module, a multi-channel MLP that:
- Decomposes global features from attenuation and radiance networks
- Processes each subcarrier independently with 256-dimensional layers
- Maintains modulation characteristics across the frequency spectrum
- Enables precise interference management and signal optimization

### CSI Virtual Link Architecture
Prism introduces a novel approach to MIMO channel modeling:
- **Virtual Link Concept**: Each M×N_UE uplink channel combination is treated as a single virtual link
- **Enhanced Channel Modeling**: Base station antennas receive M×N_UE uplink signals per antenna
- **Improved Spatial Resolution**: Better representation of complex multi-path environments
- **Smart Sampling**: Randomly samples K virtual links per antenna for computational efficiency
- **Scalable Architecture**: Configurable parameters for different deployment scenarios

### Smart Sampling for Computational Efficiency
The system implements intelligent sampling strategies to handle large numbers of virtual links:
- **Random Sampling**: Randomly selects K virtual links from M×N_UE combinations
- **Configurable Sample Size**: Adjustable K parameter (default: K=64) for different accuracy requirements
- **Batch Diversity**: Each batch uses different random seeds for diverse sampling
- **Performance Optimization**: Reduces computational complexity from O(M×N_UE) to O(K)
- **Quality Preservation**: Maintains representative statistics while reducing processing time

### Advanced Ray Tracing System
The system implements sophisticated ray tracing capabilities:
- **Multi-Angle Sampling**: 36 azimuth × 18 elevation angle combinations for comprehensive coverage
- **Spatial Point Sampling**: 64 sampling points per ray for high-resolution spatial modeling
- **Independent Antenna Tracking**: 6 antennas with independent ray tracing
- **Configurable Parameters**: Easily adjustable sampling densities for different accuracy requirements
- **Performance Optimization**: Efficient algorithms for real-time ray tracing applications

### Network Structure
```
Input: RF Signal + Position → Attenuation Network → Radiance Network → RF Prism Module → C Subcarrier Outputs
                    ↓
            CSI Virtual Link Processing → Ray Tracing Engine → Enhanced Channel Prediction
```

## Project Structure

```
Prism/
├── 📁 src/prism/                     # Main package source code
│   ├── 📄 model.py                  # Core model architecture
│   ├── 📄 dataloader.py             # Data loading and processing
│   ├── 📄 renderer.py               # Visualization and rendering
│   ├── 📄 csi_processor.py          # CSI virtual link processing
│   ├── 📄 ray_tracer.py             # Advanced ray tracing engine
│   └── 📁 utils/                    # OFDM signal processing utilities
│
├── 📁 scripts/                       # Executable scripts
│   ├── 📄 prism_runner.py           # Main training/testing runner
│   ├── 📄 basic_usage.py            # Basic usage demonstration
│   ├── 📄 csi_demo.py               # CSI virtual link demonstration
│   ├── 📄 ray_tracing_demo.py       # Ray tracing demonstration
│   └── 📁 simulation/               # Sionna-based 5G simulations
│       ├── 📄 sionna_simulation.py  # Generic 5G OFDM simulation
│       └── 📄 README.md             # Simulation overview
│
├── 📁 configs/                       # Configuration files
│   ├── 📄 ofdm-5g-sionna.yml       # 5G OFDM with CSI virtual links and ray tracing
│   ├── 📄 ofdm-wideband.yml         # 1024 subcarriers configuration
│   └── 📄 ofdm-wifi.yml             # 52 subcarriers configuration
│
├── 📁 docs/                          # Documentation
│   ├── 📄 csi_architecture.md       # CSI virtual link architecture
│   ├── 📄 ray_tracing_guide.md      # Ray tracing implementation guide
│   └── 📄 advanced_features.md      # Advanced features documentation
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
- Additional ray tracing libraries (configurable)

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

# Install ray tracing dependencies (optional)
pip install -e ".[ray-tracing]"
```

## Usage

### Training

```bash
# Train for OFDM scenario with C subcarriers
python scripts/prism_runner.py --mode train --config configs/ofdm-wideband.yml --dataset_type ofdm --gpu 0

# Train for WiFi-like scenario with 52 subcarriers
python scripts/prism_runner.py --mode train --config configs/ofdm-wifi.yml --dataset_type ofdm --gpu 0

# Train with 5G OFDM (CSI virtual links + ray tracing)
python scripts/prism_runner.py --mode train --config configs/ofdm-5g-sionna.yml --dataset_type ofdm --gpu 0
```

### Inference

```bash
# Test the trained model
python scripts/prism_runner.py --mode test --config configs/ofdm-wideband.yml --dataset_type ofdm --gpu 0 --checkpoint path/to/checkpoint

# Test with 5G OFDM features
python scripts/prism_runner.py --mode test --config configs/ofdm-5g-sionna.yml --dataset_type ofdm --gpu 0 --checkpoint path/to/checkpoint
```

### CSI Virtual Link Demonstration

```bash
# Run CSI virtual link demonstration
python scripts/csi_demo.py --config configs/csi-virtual-link.yml

# Demonstrate M×N_UE uplink combinations
python scripts/csi_demo.py --config configs/csi-virtual-link.yml --show_virtual_links
```

### Ray Tracing Demonstration

```bash
# Run ray tracing demonstration
python scripts/ray_tracing_demo.py --config configs/ray-tracing.yml

# Customize ray tracing parameters
python scripts/ray_tracing_demo.py --config configs/ray-tracing.yml --azimuth_samples 72 --elevation_samples 16 --points_per_ray 128
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
```

For detailed simulation documentation, see `scripts/simulation/README.md`.

## Configuration

The system supports various OFDM configurations:
- **Subcarrier Count**: Configurable from 52 (WiFi) to 1024+ (ultra-wideband)
- **Antenna Configuration**: Flexible N_UE × N_BS MIMO setups
- **Frequency Bands**: Adaptable to different communication standards

### 5G OFDM Configuration (ofdm-5g-sionna.yml)

```yaml
# CSI Virtual Link Processing
csi_processing:
  virtual_link_enabled: true
  m_subcarriers: 1024          # Number of subcarriers
  n_ue_antennas: 2             # Number of UE antennas
  n_bs_antennas: 4             # Number of BS antennas
  virtual_link_count: 2048     # M × N_UE = 1024 × 2
  uplink_per_bs_antenna: 2048  # M × N_UE uplinks per BS antenna
  
  # Smart Sampling for Computational Efficiency
  enable_random_sampling: true
  sample_size: 64              # Sample K=64 virtual links per antenna
  sampling_strategy: 'random'  # Random sampling strategy

# Advanced Ray Tracing
ray_tracing:
  enabled: true
  azimuth_samples: 36          # Number of azimuth angles
  elevation_samples: 18         # Number of elevation angles
  points_per_ray: 64           # Spatial sampling points per ray
  total_angle_combinations: 648 # 36 × 18 = 648 angle combinations
  total_spatial_points: 41472   # 648 × 64 = 41,472 spatial samples
```

### Ray Tracing Configuration

```yaml
ray_tracing:
  enabled: true
  azimuth_samples: 36          # Number of azimuth angles (configurable)
  elevation_samples: 18         # Number of elevation angles (configurable)
  points_per_ray: 64           # Spatial sampling points per ray (configurable)
  spatial_resolution: 0.1      # Spatial resolution in meters
  angle_resolution: 10          # Angular resolution in degrees
  max_ray_length: 100.0        # Maximum ray length in meters
  reflection_order: 3           # Maximum reflection order
  diffraction_enabled: true     # Enable diffraction effects
  scattering_enabled: true      # Enable scattering effects
```

## Advanced Features

### CSI Virtual Link Processing

The CSI virtual link concept treats each M×N_UE uplink channel combination as a single virtual link:

- **Enhanced Channel Modeling**: Each base station antenna receives M×N_UE uplink signals
- **Improved Spatial Resolution**: Better representation of complex multi-path environments
- **Scalable Architecture**: Configurable parameters for different deployment scenarios
- **Performance Optimization**: Efficient processing of large channel matrices

### Advanced Ray Tracing

The ray tracing system provides comprehensive spatial modeling:

- **Multi-Angle Coverage**: 36 azimuth × 18 elevation = 648 angle combinations
- **High-Resolution Sampling**: 64 spatial points per ray for detailed path analysis
- **Configurable Parameters**: Easily adjustable for different accuracy requirements
- **Advanced Effects**: Support for reflection, diffraction, and scattering
- **Real-Time Processing**: Optimized algorithms for practical applications

## Loss Function

The training employs a frequency-aware loss function with CSI virtual link enhancement:

```
L = |h₁ - h̃₁|² + |h₂ - h̃₂|² + ... + |h_C - h̃_C|² + λ_CSI × L_CSI + λ_ray × L_ray
```

Where:
- Each subcarrier's error is computed independently
- L_CSI represents CSI virtual link consistency
- L_ray represents ray tracing accuracy
- λ_CSI and λ_ray are weighting parameters

## Applications

- **5G/6G MIMO Channel Prediction**: Ultra-wideband channel modeling with CSI virtual links
- **WiFi CSI Enhancement**: Multi-carrier signal optimization and ray tracing
- **Indoor Localization**: Frequency-dependent positioning with spatial ray analysis
- **Interference Management**: Subcarrier-specific optimization and spatial interference modeling
- **Multi-Path Analysis**: Frequency-dependent propagation modeling with ray tracing
- **Network Planning**: Advanced ray tracing for optimal base station placement
- **Channel Estimation**: Enhanced CSI processing for improved link quality

## Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src/prism

# Run CSI-specific tests
python -m pytest tests/test_csi_processor.py -v

# Run ray tracing tests
python -m pytest tests/test_ray_tracer.py -v
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

## Future Improvements

The system is designed with extensibility in mind:

- **Configurable Ray Tracing**: All ray tracing parameters are configurable for easy optimization
- **Modular CSI Processing**: CSI virtual link processing can be enhanced with additional algorithms
- **Advanced Ray Tracing**: Support for additional electromagnetic effects and materials
- **Performance Optimization**: GPU-accelerated ray tracing and parallel processing
- **Integration Capabilities**: Easy integration with existing RF simulation tools

## Citation

If you find this work useful, please cite:

```bibtex
@article{prism2024,
    title={Prism: Wideband RF Neural Radiance Fields for OFDM Communication with Advanced CSI Processing and Ray Tracing},
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
- Enhanced with CSI virtual link concepts and advanced ray tracing

## Contributing

We welcome contributions! Please see our contributing guidelines for more details.

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.
