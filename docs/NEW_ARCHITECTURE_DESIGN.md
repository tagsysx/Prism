# New Prism Network Architecture Design

## Overview

This document describes the redesigned Prism network architecture that addresses the computational efficiency issues of the previous design while maintaining the virtual link concept for OFDM communication systems.

## Key Design Principles

1. **Eliminate Independent Subcarrier Channels**: Replace the previous design where each subcarrier had independent MLP channels
2. **Shared Feature Encoding**: Use a single AttenuationNetwork to encode spatial information into a compact feature vector
3. **Efficient Decoding**: Use configurable multi-channel MLPs to decode features into attenuation and radiation factors
4. **Configurable Parameters**: Support different UE antenna counts (N_UE) and subcarrier counts (M)

## Network Architecture

### 1. AttenuationNetwork

**Purpose**: Encode spatial position information into a compact feature representation

**Input**: 
- Sampling point position (3D coordinates)

**Output**: 
- Single 128-dimensional feature vector (configurable dimension)

**Architecture**: 
- 8-layer MLP with ReLU activation
- Input: 3D → Hidden: 256D → Output: 128D

**Key Benefits**:
- Single network instead of M×N_UE independent networks
- Compact 128D representation instead of M×N_UE×128D
- Maintains spatial encoding capabilities

### 2. Attenuation Decoder

**Purpose**: Convert 128D features into M×N_UE attenuation factors

**Input**: 
- 128-dimensional feature vector from AttenuationNetwork

**Output**: 
- N_UE channels, each outputting M attenuation factors
- Total: N_UE × M attenuation values

**Architecture**: 
- N_UE independent 3-layer MLPs
- Each channel: 128D → 256D → 256D → M
- Output: Complex values representing attenuation factors

**Key Benefits**:
- Configurable N_UE channels
- Efficient processing of M subcarriers per channel
- Maintains per-UE-antenna processing

### 3. RadianceNetwork

**Purpose**: Process UE position, viewing direction, and spatial features to output radiation characteristics

**Input**: 
- UE position (3D coordinates)
- Viewing direction (3D vector)
- 128-dimensional feature vector from AttenuationNetwork

**Output**: 
- N_UE channels, each outputting M radiation factors
- Total: N_UE × M radiation values

**Architecture**: 
- N_UE independent channels
- Each channel processes: [UE_pos, view_dir, 128D_features] → M radiation values
- Output: Complex values representing radiation characteristics

**Key Benefits**:
- Independent processing for each UE antenna
- Incorporates spatial features from AttenuationNetwork
- Configurable output dimensions

## Parameter Configuration

### Core Parameters

```yaml
model:
  num_subcarriers: 408          # M: Number of subcarriers
  num_ue_antennas: 4            # N_UE: Number of UE antennas
  num_bs_antennas: 64           # N_BS: Number of BS antennas
  position_dim: 3               # 3D position coordinates
  hidden_dim: 256               # Hidden layer dimension
  feature_dim: 128              # Feature vector dimension (configurable)
```

### Virtual Link Configuration

```yaml
virtual_links:
  total_count: 1632             # M × N_UE = 408 × 4
  attenuation_factors: 1632     # N_UE × M attenuation values
  radiation_factors: 1632       # N_UE × M radiation values
  feature_dimension: 128        # Compact spatial encoding
```

## Data Flow

```
1. Spatial Position (3D)
   ↓
2. AttenuationNetwork
   ↓
3. 128D Feature Vector
   ↓
4. Attenuation Decoder (N_UE channels)
   ↓
5. N_UE × M Attenuation Factors
   
6. UE Position + View Direction + 128D Features
   ↓
7. RadianceNetwork (N_UE channels)
   ↓
8. N_UE × M Radiation Factors
```

## Computational Benefits

### Previous Design
- **Parameters**: ~81M (2482 layers)
- **Memory**: High (M×N_UE×128D features)
- **Computation**: O(M×N_UE) independent forward passes

### New Design
- **Parameters**: Significantly reduced
- **Memory**: Low (single 128D feature vector)
- **Computation**: O(1) feature encoding + O(N_UE) channel processing

### Efficiency Improvements
- **Parameter Reduction**: ~90% reduction in model parameters
- **Memory Efficiency**: ~99% reduction in feature memory usage
- **Training Speed**: Significantly faster training and inference
- **Scalability**: Easy to adjust N_UE and M parameters

## Implementation Notes

1. **Feature Dimension**: 128D is configurable, can be adjusted based on complexity requirements
2. **Channel Independence**: Each UE antenna channel processes independently for better parallelization
3. **Complex Output**: Both attenuation and radiation factors are complex numbers for RF signal modeling
4. **Configurable Architecture**: Easy to modify N_UE and M without changing core network structure

## Future Extensions

1. **Attention Mechanisms**: Can add attention layers for better feature processing
2. **Multi-Scale Features**: Can incorporate features at different spatial scales
3. **Dynamic Channels**: Can implement dynamic channel allocation based on signal strength
4. **Adaptive Features**: Can implement adaptive feature dimension based on complexity requirements
