# Prism Network Architecture Design

## Overview

This document describes the redesigned Prism network architecture that addresses the computational efficiency issues of the previous design while maintaining the virtual link concept for OFDM communication systems.

## Key Design Principles

This project extends $\text{NeRF}^2$ to a $K$-subcarrier wideband OFDM communication system. We consider a single base station equipped with an $N_{\text{BS}}$-element antenna array, and a user equipment (UE) equipped with an $N_{\text{UE}}$-element antenna array. The goal is to estimate the $K$-dimensional CSI values received at each base station antenna.

Specifically, we enhance the AttenuationNetwork by introducing an $N_{\text{UE}}$-channel AttenuationDecoder. The AttenuationNetwork takes as input the position of a sampled point $P_\text{v}$ and produces a 128-dimensional feature vector $F$. Each channel of the AttenuationDecoder processes the attenuation feature $F$ and outputs $K$ complex-valued CSI values, representing the attenuation factors at the sampled points. As a result, the output of the decoder is a $N_{\text{UE}}\times K$ matrix of complex-valued outputs.

To model the radiation pattern of each base station antenna, we construct an $N_{\text{BS}}$-entry antenna codebook, where each entry is a 64-dimensional learnable embedding corresponding to one antenna. The RadianceNetwork accepts the antenna embedding $C_i$ of the $i^\text{th}$ antenna, the viewing direction $\omega$, and the UE position $P_{\text{UE}}$. These inputs are combined to produce a $N_{\text{UE}} \times K$ matrix of complex-valued outputs, capturing the radiation-dependent CSI observed at the UE.

**Note**: All spatial inputs (sampled point position $P_\text{v}$, viewing direction $\omega$, and UE position $P_{\text{UE}}$) are IPE-encoded before being processed by the networks.

## Network Architecture

### 1. AttenuationNetwork

**Purpose**: Encode spatial position information into a compact feature representation

**Input**: 
- Sampling point position (3D coordinates) - **IPE-encoded**

**Output**: 
- Single 128-dimensional feature vector (configurable dimension)

**Architecture**: 
- Similar to Standard NeRF density network architecture with 8 layers and shortcuts
- Input: IPE-encoded 3D → Hidden: 256D → Output: 128D
- **Note**: Outputs complex-valued features for RF signal modeling

**Key Benefits**:
- Single network instead of $K \times N_{\text{UE}}$ independent networks
- Compact 128D representation instead of $K \times N_{\text{UE}} \times 128D$
- Maintains spatial encoding capabilities
- Standard NeRF architecture ensures proven performance and stability

### 2. Attenuation Decoder

**Purpose**: Convert 128D features into $N_{\text{UE}} \times K$ attenuation factors

**Input**: 
- 128-dimensional feature vector from AttenuationNetwork

**Output**: 
- $N_{\text{UE}} \times K$ attenuation values for all UE antenna channels
- Total: $N_{\text{UE}} \times K$ attenuation values
- **Complex-valued outputs** for accurate RF signal modeling

**Architecture**: 
- Single network processing all UE antenna channels
- Network: 128D → 256D → 256D → $N_{\text{UE}} \times K$
- Output: Complex values representing attenuation factors

**Key Benefits**:
- Direct output of all $N_{\text{UE}} \times K$ channel values
- Efficient processing of all UE antenna channels simultaneously
- Maintains virtual link concept for OFDM communication
- **Complex-valued outputs** capture both magnitude and phase information

### 3. RadianceNetwork

**Purpose**: Process UE position, viewing direction, spatial features, and antenna-specific embeddings to output radiation characteristics

**Input**: 
- UE position (3D coordinates) - **IPE-encoded**
- Viewing direction (3D vector) - **IPE-encoded**
- 128-dimensional feature vector from AttenuationNetwork
- **Antenna embedding index** (to select from BS antenna codebook)

**Output**: 
- $N_{\text{UE}} \times K$ radiation values for all UE antenna channels
- Total: $N_{\text{UE}} \times K$ radiation values

**Architecture**: 
- Structure similar to the color subnetwork in standard NeRF
- Single network processing all UE antenna channels
- Each channel processes: [IPE-encoded UE_pos, IPE-encoded view_dir, 128D_features, 64D_antenna_embedding] → $N_{\text{UE}} \times K$ radiation values
- **Antenna Codebook**: Learnable 64-dimensional embeddings for each BS antenna
- Output: Complex values representing radiation characteristics

**Key Benefits**:
- Direct output of all $N_{\text{UE}} \times K$ channel radiation values
- Incorporates spatial features from AttenuationNetwork
- **Antenna-specific radiation patterns**: Each BS antenna has unique learnable embedding
- **Flexible antenna selection**: Can handle different antenna configurations
- Maintains virtual link concept for complete OFDM system modeling
- **Standard NeRF color subnetwork architecture** ensures proven performance and stability

### 4. Antenna Embedding Codebook

**Purpose**: Provide learnable antenna-specific embeddings to capture unique radiation patterns of each BS antenna

**Structure**:
- **Codebook Size**: $N_{\text{BS}}$ learnable embeddings ($N_{\text{BS}}$ = 64 for typical configurations)
- **Embedding Dimension**: 64-dimensional learnable vectors
- **Total Parameters**: $N_{\text{BS}}$ × 64 = 4,096 learnable parameters

**Implementation**:
- **Lookup Table**: Indexed by antenna ID (0 to $N_{\text{BS}}$-1)
- **Learnable Parameters**: Each embedding is a trainable 64D vector
- **Initialization**: Random initialization or pre-trained embeddings
- **Gradient Flow**: Full gradient updates during training

**Key Features**:
- **Antenna Diversity**: Each antenna learns unique radiation characteristics
- **Efficient Storage**: Compact 64D representation per antenna
- **Easy Extension**: Can add new antennas by extending the codebook
- **Transfer Learning**: Pre-trained embeddings can be fine-tuned for new scenarios

## Parameter Configuration

### Core Parameters

```yaml
model:
  num_subcarriers: K             # K: Number of subcarriers
  num_ue_antennas: N_UE         # N_{\text{UE}}: Number of UE antennas
  num_bs_antennas: N_BS         # N_{\text{BS}}: Number of BS antennas
  position_dim: 3               # 3D position coordinates
  hidden_dim: 256               # Hidden layer dimension
  feature_dim: 128              # Feature vector dimension (configurable)
  antenna_embedding_dim: 64     # Antenna embedding dimension
  use_antenna_codebook: true    # Enable antenna-specific embeddings
  use_ipe_encoding: true        # Enable IPE encoding for spatial inputs
```

### Virtual Link Configuration

```yaml
virtual_links:
  total_count: N_UE × K         # N_{\text{UE}} × K total channels
  attenuation_factors: N_UE × K # N_{\text{UE}} × K attenuation values
  radiation_factors: N_UE × K   # N_{\text{UE}} × K radiation values
  feature_dimension: 128        # Compact spatial encoding
```

## Data Flow

```
1. Spatial Position (3D) → IPE Encoding
   ↓
2. AttenuationNetwork
   ↓
3. 128D Feature Vector
   ↓
4. Attenuation Decoder (Single Network)
   ↓
5. N_{\text{UE}} × K Attenuation Factors (All UE Antenna Channels)
   
6. IPE-encoded UE Position + IPE-encoded View Direction + 128D Features + Antenna Index
   ↓
7. Antenna Embedding Codebook (N_{\text{BS}} × 64D)
   ↓
8. 64D Antenna Embedding
   ↓
9. RadianceNetwork (Single Network)
   ↓
10. N_{\text{UE}} × K Radiation Factors (All UE Antenna Channels)
```




## Implementation Notes

1. **Feature Dimension**: 128D is configurable, can be adjusted based on complexity requirements
2. **Channel Independence**: Each UE antenna channel processes independently for better parallelization
3. **Complex Output**: Both attenuation and radiation factors are complex numbers for RF signal modeling
4. **Configurable Architecture**: Easy to modify $N_{\text{UE}}$ and $K$ without changing core network structure
5. **Antenna Codebook**: $N_{\text{BS}} \times 64D$ learnable embeddings for antenna-specific radiation patterns
6. **Codebook Management**: Efficient lookup table implementation with gradient flow to all embeddings
7. **IPE Encoding**: All spatial inputs (positions and viewing directions) are IPE-encoded for better spatial representation learning

