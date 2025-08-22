# Ray Tracing Design Document

## Overview

This document outlines the design and implementation of the discrete electromagnetic ray tracing system for the Prism project. The system implements an efficient voxel-based ray tracing approach that combines discrete radiance field modeling with advanced optimization strategies to achieve both accuracy and computational efficiency.

The core concept is that the system computes **RF signal strength** $S_{\text{ray}}$ at the base station's antenna from each direction, which includes both amplitude and phase information of the electromagnetic wave. A key architectural advantage is that **ray tracing operations are independent**, enabling efficient parallelization using CUDA or multi-threading for significant performance acceleration.

## 1. Core Design

### 1.1 Base Station Configuration

The ray tracing system is centered around base stations (BS) with configurable locations:
- **Default configuration**: Base station positioned at the origin $(0, 0, 0)$
- **Customizable positioning**: Base station location $P_{\text{BS}}$ can be configured for different deployment scenarios
- **Multi-antenna support**: Each base station can have multiple antennas for MIMO operations

### 1.2 Directional Space Division

The antenna's directional space is discretized into a structured grid:
- **Grid dimensions**: $A \times B$ directions
- **Azimuth divisions**: $A$ divisions covering the horizontal plane
- **Elevation divisions**: $B$ divisions covering the vertical plane
- **Angular resolution**: $\Delta \phi = \frac{2\pi}{A}$ (azimuth) and $\Delta \theta = \frac{\pi}{B}$ (elevation)

## 2. Ray Tracing Process

### 2.1 Ray Definition and Parameterization

For each direction $(\phi_i, \theta_j)$ in the $A \times B$ grid:

```math
\text{Ray}(\phi_i, \theta_j): P_{\text{BS}} + \omega_{ij} \cdot t
```

Where:
- $P_{\text{BS}}$: Base station position
- $\omega_{ij}$: Unit direction vector for direction $(\phi_i, \theta_j)$
- $t$: Distance parameter along the ray
- $\omega_{ij} = [\sin(\theta_j)\cos(\phi_i), \sin(\theta_j)\sin(\phi_i), \cos(\theta_j)]$

### 2.2 Energy Tracing Process

For each antenna of the base station, the system traces RF energy along all $A \times B$ directions:

1. **Directional initialization**: Initialize ray parameters for each direction
2. **Energy propagation**: Trace electromagnetic energy along each ray using the discrete radiance field model
3. **Attenuation modeling**: Apply material-dependent attenuation coefficients at each voxel intersection
4. **Energy accumulation**: Compute cumulative energy received at user equipment (UE) locations

### 2.3 Advanced Ray Tracing Techniques

The system implements two key optimization strategies (as detailed in the Technical Specification):

#### 2.3.1 Importance-Based Sampling
- **Adaptive sampling density**: Concentrates computational resources in high-attenuation regions
- **Non-uniform discretization**: Optimizes sampling based on material properties
- **Efficiency improvement**: Reduces ineffective computation in low-attenuation areas

#### 2.3.2 Pyramid Ray Tracing
- **Spatial subdivision**: Divides directional space into pyramidal regions
- **Hierarchical sampling**: Implements multi-level sampling strategy
- **Monte Carlo integration**: Improves accuracy within truncated cone regions

## 3. RF Signal Computation

### 3.1 Ray Count Analysis

The total number of rays in the system is substantial:

```math
N_{\text{total}} = N_{\text{BS}} \times A \times B \times N_{\text{UE}} \times K
```

Where:
- $N_{\text{BS}}$: Number of base stations
- $A \times B$: Directional grid dimensions
- $N_{\text{UE}}$: Number of user equipment devices
- $K$: Number of subcarriers in the frequency domain

### 3.2 Subcarrier Sampling Optimization

To manage computational complexity, the system implements intelligent subcarrier sampling:

- **Random selection**: For each UE, randomly select $K' < K$ subcarriers
- **Sampling ratio**: $K' = \alpha \cdot K$ where $\alpha \in (0, 1)$ is the sampling factor
- **Reduced complexity**: Effective ray count becomes $N_{\text{BS}} \times A \times B \times N_{\text{UE}} \times K'$

### 3.3 RF Signal Computation

The ray tracer computes RF signal strength at each UE location from each direction:

```math
S_{\text{ray}}(\phi_i, \theta_j, P_{\text{UE}}, f_k, C)
```

Where:
- $S_{\text{ray}}(\phi_i, \theta_j, P_{\text{UE}}, f_k, C)$: Complex RF signal strength received at UE position $P_{\text{UE}}$ from direction $(\phi_i, \theta_j)$ at frequency $f_k$ with base station antenna embedding $C$
- $C$: Base station's antenna embedding parameter that characterizes the antenna's radiation pattern and properties

### 3.4 Virtual Link Computation

The ray tracer computes accumulated RF signal strength for the optimized set of virtual links:

```math
S_{\text{accumulated}}(P_{\text{UE}}, f_k, C) = \sum_{i=1}^{A} \sum_{j=1}^{B} S_{\text{ray}}(\phi_i, \theta_j, P_{\text{UE}}, f_k, C)
```

Where:
- $P_{\text{UE}}$: UE position
- $f_k$: Selected subcarrier frequency
- $C$: Base station's antenna embedding parameter
- $S_{\text{ray}}(\phi_i, \theta_j, P_{\text{UE}}, f_k, C)$: RF signal strength received from direction $(\phi_i, \theta_j)$ at frequency $f_k$ with antenna embedding $C$

## 4. Implementation Details

**Implementation Note**: Since ray tracing operations are independent, the system is designed to leverage parallel computing architectures. Consider implementing CUDA kernels or multi-threaded processing for optimal performance.

### 4.1 Ray Tracing Algorithm

```python
def trace_ray(base_station_pos, direction, ue_positions, selected_subcarriers, antenna_embedding):
    """
    Trace RF signal along a single ray direction
    
    Args:
        base_station_pos: Base station position P_BS
        direction: Direction vector (phi, theta)
        ue_positions: List of UE positions
        selected_subcarriers: Randomly selected subcarrier indices
        antenna_embedding: Base station's antenna embedding parameter C
    
    Returns:
        Dictionary mapping (ue_pos, subcarrier) to received RF signal strength
    """
    results = {}
    
    for ue_pos in ue_positions:
        for subcarrier_idx in selected_subcarriers:
            # Apply importance-based sampling along ray with antenna embedding
            signal_strength = importance_based_ray_tracing(
                base_station_pos, direction, ue_pos, subcarrier_idx, antenna_embedding
            )
            results[(ue_pos, subcarrier_idx)] = signal_strength
    
    return results
```

### 4.2 Subcarrier Selection Strategy

```python
def select_subcarriers(num_total_subcarriers, sampling_ratio):
    """
    Randomly select a subset of subcarriers for each UE
    
    Args:
        num_total_subcarriers: Total number of available subcarriers K
        sampling_ratio: Fraction of subcarriers to select α
    
    Returns:
        List of selected subcarrier indices
    """
    num_selected = int(num_total_subcarriers * sampling_ratio)
    return random.sample(range(num_total_subcarriers), num_selected)
```

### 4.3 RF Signal Accumulation

```python
def accumulate_signals(base_station_pos, ue_positions, selected_subcarriers, antenna_embedding):
    """
    Accumulate RF signals from all directions for all UEs and selected subcarriers
    
    Args:
        base_station_pos: Base station position
        ue_positions: List of UE positions
        selected_subcarriers: Dictionary mapping UE to selected subcarriers
        antenna_embedding: Base station's antenna embedding parameter C
    
    Returns:
        Accumulated signal strength matrix for all virtual links
    """
    accumulated_signals = {}
    
    # Iterate through all A x B directions
    for phi in range(num_azimuth_divisions):
        for theta in range(num_elevation_divisions):
            direction = (phi, theta)
            
            # Trace ray for this direction with antenna embedding
            ray_results = trace_ray(
                base_station_pos, direction, ue_positions, selected_subcarriers, antenna_embedding
            )
            
            # Accumulate signals for each virtual link
            for (ue_pos, subcarrier), signal_strength in ray_results.items():
                if (ue_pos, subcarrier) not in accumulated_signals:
                    accumulated_signals[(ue_pos, subcarrier)] = 0
                accumulated_signals[(ue_pos, subcarrier)] += signal_strength
    
    return accumulated_signals
```

## 5. Performance Considerations

### 5.1 Ray Tracing Independence and Parallelization

**Important Notice**: The tracing of rays are **independent** of each other. Each ray can be processed independently without dependencies on other rays. This makes the system highly suitable for parallel computing acceleration.

**Recommended Acceleration Strategies**:
- **CUDA/GPU Computing**: Utilize GPU parallelization for massive ray tracing workloads
- **Multi-threading**: Implement multi-threaded processing for CPU-based acceleration
- **Distributed Computing**: Scale across multiple compute nodes for large-scale deployments

### 5.2 Computational Efficiency

- **Parallel processing**: Ray tracing can be parallelized across directions and UEs
- **Memory management**: Efficient caching of voxel properties and ray intersection results
- **Early termination**: Stop ray tracing when signal strength falls below threshold

### 5.2 Accuracy vs. Speed Trade-offs

- **Sampling ratio selection**: Balance between computational cost and frequency resolution
- **Directional resolution**: Trade-off between angular accuracy and ray count
- **Voxel size**: Balance between spatial resolution and memory usage

### 5.3 Scalability

- **Multi-base station support**: Extensible architecture for cellular networks
- **Dynamic UE positioning**: Support for mobile user equipment
- **Frequency domain flexibility**: Adaptable to different wireless standards

## 6. Integration with Discrete Radiance Field Model

The ray tracing system integrates seamlessly with the discrete radiance field model:

1. **Voxel interaction**: Rays intersect with voxels to determine material properties
2. **Attenuation modeling**: Complex attenuation coefficients applied at each intersection
3. **Signal propagation**: Exponential decay model for cumulative attenuation
4. **Radiation calculation**: Direction-dependent voxel radiation properties

## 7. Future Enhancements

### 7.1 Advanced Optimization Techniques

- **Machine learning-based directional sampling**: Set up a MLP to learn which directions should be sampled based on the antenna embedding $C$.


---

*This document describes the ray tracing system design for the Prism project. For implementation details and technical specifications, refer to the Technical Specification document.*

