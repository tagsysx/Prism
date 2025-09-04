# Prism: Frequency-Aware Neural Radio-Frequency Radiance Fields

A PyTorch-based implementation of wideband RF neural radiance fields for OFDM communication systems, combining discrete electromagnetic ray tracing with neural network-based optimization.

## Overview

Prism implements an efficient voxel-based ray tracing approach that combines discrete radiance field modeling with advanced optimization strategies to achieve both accuracy and computational efficiency. The system is designed for RF signal strength computation in wireless communication scenarios, particularly for OFDM systems with MIMO antenna configurations.

## 🚀 Key Features

### Core Capabilities
- **🧠 Neural Radiance Fields**: Advanced neural network-based RF signal modeling with complex-valued outputs
- **📡 OFDM Communication Support**: Optimized for wideband OFDM systems with multi-subcarrier processing
- **⚡ GPU Acceleration**: CUDA support with automatic fallback for high-performance computations
- **🏗️ Modular Architecture**: Clean, extensible design for different use cases and antenna configurations

### Advanced Loss Functions
- **📊 Multi-Domain Validation**: CSI (frequency), PDP (time), and Spatial Spectrum (space) losses
- **🎯 Hybrid Loss System**: Combines magnitude, phase, and complex MSE for optimal training
- **📐 Spatial Spectrum Analysis**: Bartlett beamformer-based DOA estimation and validation
- **⚖️ Multi-Objective Training**: Configurable loss weights for different validation aspects

### Latest Features (2025)
- **🔄 Automatic GPU Selection**: No manual GPU configuration needed
- **📈 Enhanced Training Interface**: Comprehensive training pipeline with real-time monitoring
- **🎨 Visualization Support**: Automatic generation of spatial spectrum and CSI comparison plots
- **⚙️ Template Configuration**: Dynamic path resolution and flexible configuration system
- **🔧 Comprehensive Testing**: Complete testing pipeline with performance analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/tagsysx/Prism.git
cd Prism

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 🚀 Quick Start

### Generating Synthesized Dataset

Generate synthetic channel data using the Sionna-compatible data generator:

```bash
# Most concise command - generate 300 positions
python data/sionna/generator.py --n 300

# Generate custom dataset with specific parameters
python data/sionna/generator.py \
    --num_positions 300 \
    --num_subcarriers 408 \
    --num_ue_antennas 1 \
    --num_bs_antennas 64 \
    --output_file data/sionna/synthetic_data.h5

# Generate dataset with specific environment parameters
python data/sionna/generator.py \
    --carrier_frequency 3.5e9 \
    --bandwidth 1.224e7 \
    --scenario "urban_macro" \
    --output_file data/sionna/urban_dataset.h5
```

The generator creates HDF5 files with:
- **Channel responses** for each UE position and subcarrier
- **UE positions** in 3D coordinates
- **BS antenna array geometry** (8x8 configuration)
- **Frequency domain information** (subcarrier frequencies)
- **Spatial correlation** based on antenna geometry and propagation paths

### Training a Model

```bash
# Most concise training command
python scripts/train_prism.py --config configs/sionna.yml

# Train with custom parameters
python scripts/train_prism.py \
    --config configs/sionna.yml \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001
```

### Testing a Trained Model

```bash
# Most concise testing command
python scripts/test_prism.py --config configs/sionna.yml

# Test with specific checkpoint and output directory
python scripts/test_prism.py \
    --config configs/sionna.yml \
    --checkpoint results/sionna/training/models/best_model.pt \
    --output_dir results/sionna/testing
```

### Python API Usage

```python
from prism.training_interface import PrismTrainingInterface
from prism.loss import LossFunction
import yaml

# Load configuration
with open('configs/sionna.yml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize training interface
trainer = PrismTrainingInterface(config)

# Create loss function with multi-domain validation
loss_config = {
    'csi_weight': 0.7,           # Frequency domain
    'pdp_weight': 0.3,           # Time domain  
    'spatial_spectrum_weight': 0.1,  # Spatial domain
    'csi_loss': {'type': 'hybrid'},
    'pdp_loss': {'type': 'hybrid'},
    'spatial_spectrum_loss': {
        'enabled': True,
        'algorithm': 'bartlett',
        'fusion_method': 'average'
    }
}
loss_fn = LossFunction(loss_config)

# Train the model
trainer.train(loss_function=loss_fn)
```

## 📁 Project Structure

```
Prism/
├── src/prism/                    # Main source code
│   ├── networks/                 # Neural network components
│   │   ├── prism_network.py     # Main integrated network
│   │   ├── attenuation_network.py
│   │   ├── radiance_network.py
│   │   └── antenna_codebook.py
│   ├── loss/                     # Loss function system
│   │   ├── csi_loss.py          # CSI loss functions
│   │   ├── pdp_loss.py          # PDP loss functions
│   │   ├── spatial_spectrum_loss.py  # Spatial spectrum loss
│   │   └── prism_loss_function.py    # Main loss combiner
│   ├── ray_tracer_*.py          # Ray tracing implementations
│   ├── training_interface.py    # Training pipeline
│   └── spatial_spectrum.py      # Spatial spectrum utilities
├── docs/                         # Comprehensive documentation
│   ├── LOSS_FUNCTIONS.md       # Loss function system guide
│   ├── NETWORK_DESIGN.md       # Architecture documentation
│   ├── TRAINING_DESIGN.md      # Training methodology
│   ├── SPATIAL_SPECTRUM.md     # Spatial spectrum theory
│   └── RAY_TRACING_DESIGN.md   # Ray tracing implementation
├── configs/                      # Configuration files
│   ├── sionna.yml              # Main configuration
│   └── README.md               # Configuration guide
├── scripts/                      # Training and testing scripts
│   ├── train_prism.py          # Training script
│   └── test_prism.py           # Testing script
├── tests/                        # Comprehensive test suite
├── results/                      # Training and testing outputs
│   └── sionna/
│       ├── training/           # Training results and checkpoints
│       └── testing/            # Testing results and plots
└── data/                         # Dataset and data generation
    └── sionna/                 # Synthetic data generator
```

## ⚙️ Configuration

The system uses YAML configuration files with template variables and dynamic path resolution:

```yaml
# Example configuration structure
neural_networks:
  num_subcarriers: 408
  num_ue_antennas: 1  # Single antenna processing
  num_bs_antennas: 64

base_station:
  antenna_array:
    configuration: "8x8"
    element_spacing: "half_wavelength"
  ofdm:
    center_frequency: 3.5e9
    bandwidth: 1.224e7
    num_subcarriers: 408

training:
  loss:
    csi_weight: 0.7
    pdp_weight: 0.3
    spatial_spectrum_weight: 0.1
    spatial_spectrum_loss:
      enabled: true
      algorithm: 'bartlett'
      theta_range: [0, 5, 90]    # degrees
      phi_range: [0, 10, 360]    # degrees
```

See the [Configuration Guide](configs/README.md) for detailed parameter explanations.

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Loss Functions Guide](docs/LOSS_FUNCTIONS.md)** - Complete loss function system documentation
- **[Network Design](docs/NETWORK_DESIGN.md)** - Architecture and component details
- **[Training Design](docs/TRAINING_DESIGN.md)** - Training methodology and best practices
- **[Spatial Spectrum](docs/SPATIAL_SPECTRUM.md)** - Spatial spectrum analysis theory
- **[Ray Tracing Design](docs/RAY_TRACING_DESIGN.md)** - Ray tracing implementation details
- **[Configuration Guide](configs/README.md)** - Complete configuration reference

## 🛠️ Development

### Setting up development environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_networks.py          # Network component tests
pytest tests/test_ray_tracer.py        # Ray tracing tests
pytest tests/test_training_interface.py # Training pipeline tests

# Run with coverage
pytest tests/ --cov=src/prism --cov-report=html
```

### Performance Testing

```bash
# Test CUDA ray tracing performance
python tests/test_cuda_ray_tracer.py

# Test training performance
python tests/test_ray_tracing_performance.py

# Test multi-antenna processing
python tests/test_multi_antenna.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Prism in your research, please cite:

```bibtex
@ARTICLE{zhao2025prism,
  author={Zhao, Xiaopeng and An, Zhenlin and Pan, Qingrui and Yang, Lei},
  journal={IEEE Transactions on Mobile Computing},
  title={Frequency-Aware Neural Radio-Frequency Radiance Fields},
  year={2025}
}
```

## 🔧 Performance & Hardware Requirements

### Recommended System Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX 3080+ recommended)
- **RAM**: 16GB+ (32GB for large datasets)
- **Storage**: 50GB+ free space for datasets and results
- **Python**: 3.8+ with PyTorch 1.12+

### Performance Benchmarks
- **Training Speed**: ~2-3 minutes/epoch on RTX 4090 (100 positions)
- **Memory Usage**: ~8-12GB GPU memory during training
- **Inference Speed**: ~50ms per position prediction

## 🆘 Support & Community

### Getting Help
- **📖 Documentation**: Start with the [docs/](docs/) directory
- **🐛 Bug Reports**: Open an issue on [GitHub Issues](https://github.com/tagsysx/Prism/issues)
- **💬 Discussions**: Join [GitHub Discussions](https://github.com/tagsysx/Prism/discussions)
- **📧 Contact**: Reach out to the development team

### Common Issues
- **CUDA Issues**: Check [CUDA Installation Guide](docs/INSTALLATION.md#cuda-setup)
- **Memory Errors**: See [Performance Optimization](configs/README.md#performance-optimization-guide)
- **Configuration Problems**: Refer to [Configuration Guide](configs/README.md)
