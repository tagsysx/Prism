# Prism: Discrete Electromagnetic Ray Tracing System

A PyTorch-based implementation of discrete electromagnetic ray tracing with MLP-based direction sampling for RF signal strength computation.

## Overview

Prism implements an efficient voxel-based ray tracing approach that combines discrete radiance field modeling with advanced optimization strategies to achieve both accuracy and computational efficiency. The core concept is that the system computes **RF signal strength** at the base station's antenna from each direction, which includes both amplitude and phase information of the electromagnetic wave.

## Key Features

- **Discrete Electromagnetic Ray Tracing**: Efficient voxel-based ray tracing system
- **MLP-Based Direction Sampling**: Intelligent direction selection using trained neural networks
- **Antenna Embedding Integration**: Support for antenna-specific radiation patterns
- **Subcarrier Optimization**: Intelligent subcarrier sampling for computational efficiency
- **GPU Acceleration**: CUDA support for high-performance ray tracing
- **Modular Architecture**: Clean, extensible design for different use cases

## Architecture

### Core Components

1. **DiscreteRayTracer**: Main ray tracing engine implementing the design specifications
2. **MLPDirectionSampler**: Neural network for intelligent direction sampling
3. **RFSignalProcessor**: Signal strength calculation and subcarrier management
4. **BaseStation & UserEquipment**: Network entity representations
5. **VoxelGrid & Environment**: Spatial environment modeling

### MLP Direction Sampling

The system implements an intelligent direction sampling strategy using a shallow Multi-Layer Perceptron (MLP):

- **Input**: Base station's antenna embedding parameter C (128D vector)
- **Architecture**: 2-3 fully connected layers with ReLU activation
- **Output**: A×B binary indicator matrix M_ij for direction selection
- **Training**: Supervised learning on historical ray tracing data

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Prism

# Install dependencies
pip install torch numpy

# Run the example
python examples/ray_tracing_example.py
```

## Quick Start

```python
from prism.ray_tracer import DiscreteRayTracer, BaseStation
from prism.mlp_direction_sampler import create_mlp_direction_sampler
from prism.rf_signal_processor import RFSignalProcessor

# Initialize ray tracer
ray_tracer = DiscreteRayTracer(
    azimuth_divisions=36,
    elevation_divisions=18,
    max_ray_length=100.0
)

# Create base station
base_station = BaseStation(num_antennas=4)

# Set up MLP direction sampler
mlp_sampler = create_mlp_direction_sampler(
    azimuth_divisions=36,
    elevation_divisions=18
)

# Perform ray tracing
signals = ray_tracer.adaptive_ray_tracing(
    base_station_pos=base_station.position,
    antenna_embedding=base_station.get_antenna_embedding(),
    ue_positions=[[10, 5, 1.5], [15, -3, 1.5]],
    selected_subcarriers={},
    mlp_model=mlp_sampler
)
```

## Configuration

The system uses YAML configuration files for different scenarios:

- **`configs/ofdm-5g-sionna.yml`**: 5G OFDM with 408 subcarriers
- **`configs/ofdm-wideband.yml`**: Wideband with 1024 subcarriers  
- **`configs/ofdm-wifi.yml`**: WiFi with 64 subcarriers

### Key Configuration Parameters

```yaml
ray_tracing:
  azimuth_divisions: 36      # Azimuth divisions A
  elevation_divisions: 18    # Elevation divisions B
  max_ray_length: 100.0      # Maximum ray length

mlp_direction_sampling:
  input_dim: 128             # Antenna embedding dimension
  hidden_dim: 256            # Hidden layer dimension
  target_efficiency: 0.3     # Target sampling efficiency

rf_signal_processing:
  total_subcarriers: 408     # Total subcarriers K
  sampling_ratio: 0.3        # Subcarrier sampling ratio α
```

## Performance Considerations

### Ray Tracing Independence

**Important**: Ray tracing operations are **independent** of each other, enabling efficient parallelization:

- **CUDA/GPU Computing**: Utilize GPU parallelization for massive ray tracing workloads
- **Multi-threading**: Implement multi-threaded processing for CPU-based acceleration
- **Distributed Computing**: Scale across multiple compute nodes for large-scale deployments

### Computational Efficiency

- **Parallel processing**: Ray tracing can be parallelized across directions and UEs
- **Memory management**: Efficient caching of voxel properties and ray intersection results
- **Early termination**: Stop ray tracing when signal strength falls below threshold

## Examples

See `examples/ray_tracing_example.py` for a complete working example that demonstrates:

1. System initialization
2. Environment setup
3. MLP direction sampling
4. Ray tracing execution
5. RF signal processing
6. Results analysis

## Design Document

For detailed technical specifications, see `docs/RAY_TRACING_DESIGN.md` which covers:

- Core design principles
- Ray tracing process
- RF signal computation
- MLP direction sampling implementation
- Performance considerations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use Prism in your research, please cite:

```bibtex
@software{prism_ray_tracing,
  title={Prism: Discrete Electromagnetic Ray Tracing System},
  author={Prism Project Team},
  year={2024},
  url={https://github.com/your-repo/prism}
}
```
