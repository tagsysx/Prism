# Prism Networks Module

This module implements the neural network components for the Prism discrete electromagnetic ray tracing system, as described in the design documentation.

## Architecture Overview

The Prism system consists of four main neural network components that work together to model electromagnetic wave propagation:

1. **AttenuationNetwork**: Encodes spatial position information into compact feature representations
2. **AttenuationDecoder**: Converts spatial features into attenuation factors for all UE antenna channels
3. **AntennaEmbeddingCodebook**: Provides learnable antenna-specific embeddings
4. **AntennaNetwork**: Generates directional importance indicators for efficient ray tracing sampling
5. **RadianceNetwork**: Processes UE position, viewing direction, and spatial features for radiation modeling

## Network Components

### 1. AttenuationNetwork

**Purpose**: Encode spatial position information into a compact feature representation

**Input**: 
- Sampling point position (3D coordinates) - IPE-encoded

**Output**: 
- Single 128-dimensional feature vector (configurable dimension)

**Architecture**: 
- Similar to Standard NeRF density network architecture with 8 layers and shortcuts
- Input: IPE-encoded 3D → Hidden: 256D → Output: 128D
- Outputs complex-valued features for RF signal modeling

### 2. AttenuationDecoder

**Purpose**: Convert 128D features into N_UE × K attenuation factors

**Input**: 
- 128-dimensional feature vector from AttenuationNetwork

**Output**: 
- N_UE × K attenuation values for all UE antenna channels
- Complex-valued outputs for accurate RF signal modeling

**Architecture**: 
- Single network processing all UE antenna channels
- Network: 128D → 256D → 256D → N_UE × K

### 3. AntennaEmbeddingCodebook

**Purpose**: Provide learnable antenna-specific embeddings

**Structure**:
- Codebook Size: N_BS learnable embeddings
- Embedding Dimension: 64-dimensional learnable vectors
- Total Parameters: N_BS × 64 learnable parameters

**Features**:
- Antenna diversity: Each antenna learns unique radiation characteristics
- Efficient storage: Compact 64D representation per antenna
- Easy extension: Can add new antennas by extending the codebook

### 4. AntennaNetwork

**Purpose**: Process antenna embeddings to generate directional importance indicators

**Input**: 
- 64-dimensional antenna embedding from the antenna codebook

**Output**: 
- A × B directional importance matrix (indicator matrix)
- Each element indicates the importance of a specific direction

**Architecture**: 
- Shallow network for efficient processing
- Input: 64D antenna embedding → Hidden: 128D → Output: A × B importance values
- Supports top-K directional sampling for efficiency

### 5. RadianceNetwork

**Purpose**: Process UE position, viewing direction, spatial features, and antenna embeddings

**Input**: 
- UE position (3D coordinates) - IPE-encoded
- Viewing direction (3D vector) - IPE-encoded
- 128-dimensional feature vector from AttenuationNetwork
- Antenna embedding from codebook

**Output**: 
- N_UE × K radiation values for all UE antenna channels
- Complex values representing radiation characteristics

**Architecture**: 
- Structure similar to the color subnetwork in standard NeRF
- Single network processing all UE antenna channels
- Incorporates spatial features from AttenuationNetwork

## Integrated Network

### PrismNetwork

The main integrated network that combines all components:

```python
from prism.networks import PrismNetwork, PrismNetworkConfig

# Create configuration
config = PrismNetworkConfig(
    num_subcarriers=64,
    num_ue_antennas=4,
    num_bs_antennas=64,
    feature_dim=128,
    antenna_embedding_dim=64,
    azimuth_divisions=16,
    elevation_divisions=8,
    top_k_directions=32,
    complex_output=True
)

# Create network
prism_net = PrismNetwork(**config.to_dict())

# Forward pass
outputs = prism_net(
    sampled_positions=sampled_positions,
    ue_positions=ue_positions,
    view_directions=view_directions,
    antenna_indices=antenna_indices
)
```

## Usage Examples

### Individual Networks

```python
from prism.networks import AttenuationNetwork, AttenuationDecoder

# AttenuationNetwork
atten_net = AttenuationNetwork(
    input_dim=63,  # IPE-encoded 3D position
    output_dim=128,
    complex_output=True
)

# AttenuationDecoder
atten_decoder = AttenuationDecoder(
    feature_dim=128,
    num_ue_antennas=4,
    num_subcarriers=64,
    complex_output=True
)
```

### Antenna Codebook

```python
from prism.networks import AntennaEmbeddingCodebook

antenna_codebook = AntennaEmbeddingCodebook(
    num_bs_antennas=64,
    embedding_dim=64
)

# Get embeddings for specific antennas
antenna_indices = torch.tensor([0, 1, 2])
embeddings = antenna_codebook(antenna_indices)
```

### Directional Sampling

```python
from prism.networks import AntennaNetwork

antenna_net = AntennaNetwork(
    antenna_embedding_dim=64,
    azimuth_divisions=16,
    elevation_divisions=8
)

# Get directional importance
directional_importance = antenna_net(antenna_embeddings)

# Get top-K directions for efficient sampling
top_k_indices, top_k_importance = antenna_net.get_top_k_directions(
    directional_importance, k=32
)
```

## Configuration

Each network component has a corresponding configuration class that allows easy customization:

```python
from prism.networks import AttenuationNetworkConfig

config = AttenuationNetworkConfig(
    input_dim=63,
    hidden_dim=256,
    output_dim=128,
    num_layers=8,
    use_shortcuts=True,
    activation="relu",
    complex_output=True
)

network = AttenuationNetwork(**config.to_dict())
```

## Key Features

- **Complex-valued outputs**: All networks support complex-valued outputs for accurate RF signal modeling
- **Configurable architecture**: Easy to modify network dimensions and parameters
- **Efficient processing**: Single networks process all UE antenna channels simultaneously
- **Directional sampling**: AntennaNetwork enables efficient top-K directional sampling
- **Antenna diversity**: Each BS antenna learns unique radiation characteristics
- **Modular design**: Components can be used independently or integrated

## Performance Considerations

- **Memory efficiency**: Compact 128D feature representation instead of K × N_UE × 128D
- **Computational efficiency**: Single networks instead of K × N_UE independent networks
- **Directional sampling**: Reduces computational complexity from A × B to K directions
- **Batch processing**: Supports batch processing for multiple antennas and voxels

## Integration with Ray Tracing

The networks are designed to integrate seamlessly with the existing ray tracing infrastructure:

- **Voxel-based processing**: Works with discrete voxel grids
- **Directional sampling**: Integrates with MLP-based direction sampling
- **RF signal processing**: Outputs compatible with existing signal processing modules

For more details, see the main design documentation and the example usage script.
