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

### Latest Features (V3.0.0 - 2025)
- **🚀 GPU-Accelerated Analysis**: CUDA support with automatic device detection and fallback
- **⚡ Vectorized Spatial Spectrum**: GPU batch processing with 512-sample chunks for optimal performance
- **🎨 Comprehensive Plotting**: Dedicated plotting script with 6 types of analysis visualizations
- **📊 Enhanced PAS Analysis**: Increased sample count from 5 to 10 for better statistical analysis
- **🔄 Automatic GPU Selection**: No manual GPU configuration needed
- **📈 Enhanced Training Interface**: Comprehensive training pipeline with real-time monitoring
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

### Complete Workflow: Train → Test → Analyze

**1. Training a Model:**
```bash
# Basic training with default config
python scripts/train_prism.py --config configs/sionna.yml

# Training with custom parameters
python scripts/train_prism.py \
    --config configs/sionna.yml \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --output_dir results/my_experiment
```

**2. Testing the Trained Model:**
```bash
# Test with latest checkpoint
python scripts/test_prism.py --config configs/sionna.yml

# Test with specific checkpoint
python scripts/test_prism.py \
    --config configs/sionna.yml \
    --checkpoint results/sionna/training/models/best_model.pt \
    --output_dir results/sionna/testing
```

**3. Analyzing Results:**
```bash
# Basic analysis with GPU acceleration (recommended)
python scripts/analyze.py --config configs/sionna.yml --device cuda

# Analysis with explicit results path
python scripts/analyze.py \
    --results results/sionna/testing/results.npz \
    --output_dir results/sionna/analysis \
    --device cuda

# CPU analysis (for debugging)
python scripts/analyze.py --config configs/sionna.yml --device cpu
```

**4. Generating Plots:**
```bash
# Generate all analysis plots using config file
python scripts/plot.py --config configs/sionna.yml

# Generate plots with explicit analysis directory
python scripts/plot.py --analysis-dir results/sionna/testing/analysis

# Generate plots with custom output directory
python scripts/plot.py \
    --config configs/sionna.yml \
    --output-dir custom/plots
```

### Generating Synthesized Dataset

Generate synthetic channel data using the Sionna-compatible data generator:

**Most concise command - generate 300 positions:**
```bash
python data/sionna/generator.py --n 300
```

**Generate custom dataset with specific parameters:**
```bash
python data/sionna/generator.py \
    --num_positions 300 \
    --num_subcarriers 408 \
    --num_ue_antennas 1 \
    --num_bs_antennas 64 \
    --output_file data/sionna/synthetic_data.h5
```

**Generate dataset with specific environment parameters:**
```bash
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

**Most concise training command:**
```bash
python scripts/train_prism.py --config configs/sionna.yml
```

**Train with custom parameters:**
```bash
python scripts/train_prism.py \
    --config configs/sionna.yml \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001
```

### Testing a Trained Model

**Most concise testing command:**
```bash
python scripts/test_prism.py --config configs/sionna.yml
```

**Test with specific checkpoint and output directory:**
```bash
python scripts/test_prism.py \
    --config configs/sionna.yml \
    --checkpoint results/sionna/training/models/best_model.pt \
    --output_dir results/sionna/testing
```

### Analyzing Results

**Most concise analysis command (GPU-accelerated):**
```bash
python scripts/analyze.py --config configs/sionna.yml --device cuda
```

**Analyze with custom parameters:**
```bash
python scripts/analyze.py \
    --results results/sionna/testing/results.npz \
    --output_dir results/sionna/analysis \
    --device cuda \
    --fft-size 2048
```

**Analyze with specific configuration:**
```bash
python scripts/analyze.py \
    --config configs/sionna.yml \
    --output_dir results/sionna/analysis \
    --device auto
```

### 🎨 Plotting and Visualization

Prism includes a comprehensive plotting system that generates high-quality visualizations for CSI analysis results. The plotting functionality is separated into a dedicated script for better organization and performance.

#### Available Plot Types

1. **CSI MAE CDF Plots**: Amplitude and phase distribution analysis
2. **Demo CSI Samples**: Individual sample comparisons (amplitude, phase, cos(phase))
3. **Demo PAS Samples**: Spatial spectrum comparisons (BS and UE antennas)
4. **Demo PDP Samples**: Power delay profile comparisons
5. **PDP Similarity CDF**: Similarity metrics distribution (cosine, NMSE, SSIM)
6. **PAS Similarity CDF**: Spatial spectrum similarity analysis

#### Plot Generation Commands

**Generate all plots using config file:**
```bash
python scripts/plot.py --config configs/sionna.yml
```

**Generate plots with explicit analysis directory:**
```bash
python scripts/plot.py --analysis-dir results/sionna/testing/analysis
```

**Generate plots with custom output directory:**
```bash
python scripts/plot.py \
    --config configs/sionna.yml \
    --output-dir custom/plots
```

#### Plot Output Structure

```
results/sionna/testing/plots/
├── csi_mae_cdf.png              # CSI amplitude/phase analysis
├── pdp_similarity_cdf.png       # PDP similarity metrics
├── pas_similarity_cdf.png       # PAS similarity metrics
├── csi_samples/                 # Individual CSI sample plots
│   ├── csi_sample_0.png
│   ├── csi_sample_1.png
│   └── ...
├── pas/                         # PAS spatial spectrum plots
│   ├── bs_spatial_spectrum_sample_0_batch_0_subcarrier_50.png
│   ├── ue_spatial_spectrum_sample_0_batch_0_subcarrier_50.png
│   └── ...
└── pdp/                         # PDP comparison plots
    ├── pdp_sample_0_bs_0_ue_0.png
    ├── pdp_sample_1_bs_0_ue_0.png
    └── ...
```

#### Plot Features

- **High-Quality Output**: 300 DPI PNG files with professional formatting
- **Comprehensive Metrics**: Cosine similarity, NMSE, SSIM analysis
- **Statistical Analysis**: CDF plots for distribution analysis
- **Sample Visualization**: Individual sample comparisons for detailed inspection
- **Spatial Spectrum**: 2D spatial spectrum visualization for antenna analysis
- **Automatic Organization**: Structured output directories for easy navigation

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

## 📋 Command Line Reference

### Training Script (`scripts/train_prism.py`)

```bash
python scripts/train_prism.py [OPTIONS]

Options:
  --config CONFIG_FILE     Path to configuration file (required)
  --epochs N               Number of training epochs
  --batch_size N           Training batch size
  --learning_rate LR       Learning rate
  --output_dir DIR         Output directory for results
  --checkpoint_dir DIR     Directory to save checkpoints
  --resume CHECKPOINT      Resume training from checkpoint
  --device DEVICE          Device to use (cuda/cpu)
```

### Testing Script (`scripts/test_prism.py`)

```bash
python scripts/test_prism.py [OPTIONS]

Options:
  --config CONFIG_FILE     Path to configuration file (required)
  --checkpoint CHECKPOINT  Path to model checkpoint
  --output_dir DIR         Output directory for results
  --num_samples N         Number of samples to test
  --device DEVICE          Device to use (cuda/cpu)
```

### Analysis Script (`scripts/analyze.py`)

```bash
python scripts/analyze.py [OPTIONS]

Options:
  --config CONFIG_FILE     Path to configuration file (required)
  --results RESULTS_FILE   Path to results.npz file (optional, auto-detect from config)
  --output_dir DIR         Output directory for analysis
  --device DEVICE          Device to use (cuda/cpu/auto, default: auto)
  --fft-size N            FFT size for PDP computation (default: 2048)
  --num-workers N         Number of parallel workers (deprecated, kept for compatibility)
```

### Plotting Script (`scripts/plot.py`)

```bash
python scripts/plot.py [OPTIONS]

Options:
  --config CONFIG_FILE     Path to configuration file (required if --analysis-dir not used)
  --analysis-dir DIR       Path to analysis directory containing JSON files
  --output-dir DIR         Path to save plots (defaults to config-based or analysis-dir parent/plots)
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

## 💡 Usage Examples

### Example 1: Complete Training Pipeline

```bash
# 1. Generate synthetic data
python data/sionna/generator.py --n 1000

# 2. Train the model
python scripts/train_prism.py \
    --config configs/sionna.yml \
    --epochs 50 \
    --batch_size 16 \
    --output_dir results/experiment_1

# 3. Test the model
python scripts/test_prism.py \
    --config configs/sionna.yml \
    --checkpoint results/experiment_1/training/models/best_model.pt \
    --output_dir results/experiment_1/testing

# 4. Analyze results with GPU acceleration
python scripts/analyze.py \
    --config configs/sionna.yml \
    --output_dir results/experiment_1/analysis \
    --device cuda

# 5. Generate comprehensive plots
python scripts/plot.py \
    --config configs/sionna.yml \
    --output-dir results/experiment_1/plots
```

### Example 2: Using Different Datasets

```bash
# Train on Chrissy dataset
python scripts/train_prism.py --config configs/chrissy.yml

# Train on PolyU dataset  
python scripts/train_prism.py --config configs/polyu.yml

# Test with specific dataset
python scripts/test_prism.py \
    --config configs/chrissy.yml \
    --checkpoint results/chrissy/training/models/best_model.pt
```

### Example 3: Performance Analysis

```bash
# Analyze with GPU acceleration for large datasets
python scripts/analyze.py \
    --config configs/sionna.yml \
    --device cuda \
    --fft-size 2048

# Generate comprehensive plots for analysis
python scripts/plot.py \
    --config configs/sionna.yml \
    --output-dir results/sionna/plots

# CPU analysis (for debugging or systems without GPU)
python scripts/analyze.py \
    --config configs/sionna.yml \
    --device cpu
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Loss Functions Guide](docs/LOSS_FUNCTIONS.md)** - Complete loss function system documentation
- **[Network Design](docs/NETWORK_DESIGN.md)** - Architecture and component details
- **[Training Design](docs/TRAINING_DESIGN.md)** - Training methodology and best practices
- **[Spatial Spectrum](docs/SPATIAL_SPECTRUM.md)** - Spatial spectrum analysis theory
- **[Ray Tracing Design](docs/RAY_TRACING_DESIGN.md)** - Ray tracing implementation details
- **[Configuration Guide](configs/README.md)** - Complete configuration reference

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
- **GPU Analysis**: ~10x speedup on NVIDIA A100 with vectorized spatial spectrum computation
- **Batch Processing**: 512-sample chunks for optimal GPU memory utilization
- **Plot Generation**: High-quality 300 DPI plots with comprehensive metrics

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
