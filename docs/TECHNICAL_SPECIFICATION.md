# Discrete Radiance Field Model Technical Specification

## Overview

This document provides a detailed description of the technical implementation details of the Discrete Radiance Field Model in the Prism project. This model achieves efficient electromagnetic wave propagation modeling and ray tracing by discretizing 3D scenes into a finite number of voxels.

## 1. Discrete Radiance Field Model Architecture

### 1.1 Voxelized Scene Representation

In the discrete representation, 3D scenes are divided into a finite collection of voxels:

$\{P_v^i\}_{i=1}^N$

Each voxel $P_v^i$ represents a localized volume element, serving as a radiation source with two key properties:
- **Attenuation coefficient** $\rho(P_v^i)$: Describes the electromagnetic wave attenuation characteristics within the voxel
- **Radiation properties** $S(P_v^i,\omega)$: Describes the radiation intensity of the voxel in direction $\omega$

### 1.2 Neural Network Modeling

To model these properties, we use two multi-layer perceptrons (MLPs):

#### (1) AttenuationNetwork ($f_\theta$)
$f_\theta\!\left(\text{IPE}(P_v^i)\right) \to \big(\rho(P_v^i), \mathcal{F}(P_v^i)\big)$

where IPE is the Integrated Positional Encoding (IPE). 

**Function**:
- Input: Position encoding $\text{IPE}(P_v^i)$ of voxel position
- Output: Attenuation coefficient $\rho(P_v^i)$ and 256-dimensional latent features $\mathcal{F}(P_v^i)$

#### (2) RadiationNetwork ($f_\psi$)
$f_\psi\!\left(\mathcal{F}(P_v^i), \text{PE}(\omega), \text{PE}(P_{\text{TX}})\right) \to S(P_v^i,\omega)$

where $\text{PE}(\cdot)$ is the standard positional encoding. 

**Function**:
- Input: Voxel features, direction encoding, transmitter position encoding
- Output: Direction-dependent voxel radiation intensity $S(P_v^i,\omega)$

### 1.3 Positional Encoding

#### 1.3.1 Standard Positional Encoding

The standard positional encoding function $\mathrm{PE}(\cdot)$ transforms input coordinates into high-dimensional representations, enhancing the model's expressive power:

$\mathrm{PE}(x) = [\sin(2^0 \pi x), \cos(2^0 \pi x), \sin(2^1 \pi x), \cos(2^1 \pi x), \ldots, \sin(2^{L-1} \pi x), \cos(2^{L-1} \pi x)]$

This encoding allows the network to learn high-frequency functions by mapping input coordinates to a higher dimensional space.

#### 1.3.2 Integrated Positional Encoding (IPE)

The Integrated Positional Encoding function $\mathrm{IPE}(\cdot)$ extends the standard encoding to handle spatial regions rather than single points. For a 3D point $\mathbf{x} \sim \mathcal{N}(\mu, \Sigma)$, the IPE is the expected value of the positional encoding:

$\text{IPE}(\mu, \Sigma) = \left( \mathbb{E}[\sin(2^k \pi \mathbf{x})], \mathbb{E}[\cos(2^k \pi \mathbf{x})] \right)_{k=0}^{L-1}$

Where:
- $\mu$: Mean position of the Gaussian (center of the conical frustum)
- $\Sigma$: Covariance matrix modeling the spatial extent

The expectation for each dimension (e.g., x-coordinate) is:

$\mathbb{E}[\sin(2^k \pi x)] = \sin(2^k \pi \mu_x) \cdot \exp\left(-\frac{1}{2} (2^k \pi)^2 \Sigma_{xx}\right)$
$\mathbb{E}[\cos(2^k \pi x)] = \cos(2^k \pi \mu_x) \cdot \exp\left(-\frac{1}{2} (2^k \pi)^2 \Sigma_{xx}\right)$

The exponential term $\exp\left(-\frac{1}{2} (2^k \pi)^2 \Sigma_{xx}\right)$ acts as a low-pass filter, attenuating high frequencies for larger regions (higher variance), which prevents aliasing and enables efficient modeling of spatial volumes.

## 2. Attenuation Coefficient Modeling

### 2.1 Complex Attenuation Representation

The attenuation coefficient $\rho(P_v^i)$ is determined by local material properties and represented in complex form:

$\rho(P_v^i) = \Delta A + j \Delta \phi$

Where:
- $\Delta A$: Amplitude attenuation (dB value per voxel step length)
- $\Delta \phi$: Phase shift (radian value per voxel step length)

### 2.2 Physical Significance

- **Amplitude attenuation**: Reflects energy loss of electromagnetic waves during propagation
- **Phase shift**: Reflects phase changes of electromagnetic waves during propagation
- **Complex representation**: Simultaneously considers amplitude and phase information, conforming to the physical characteristics of electromagnetic wave propagation

## 3. Discrete Electromagnetic Ray Tracing

### 3.1 Ray Discretization

Given a ray starting from receiver position $P_{\text{RX}}$ with direction $\omega$, we discretize it into $M$ uniform sampling points:

$P_v^k = P_{\text{RX}} + k\Delta t \cdot \omega, \quad k=1,\dots,M$

Where $\Delta t$ is the step size.

### 3.2 Received Radiation Calculation

The radiation intensity received from direction $\omega$ is approximated as:

$S(P_{\text{RX}}, \omega) \approx \sum_{k=1}^M \exp\!\left(-\sum_{j=1}^{k-1} \rho(P_v^j)\,\Delta t\right) \rho(P_v^k) S(P_v^k, -\omega) \,\Delta t$

#### Formula Analysis

**Term (1) - Cumulative Attenuation**:
$\exp\!\left(-\sum_{j=1}^{k-1} \rho(P_v^j)\,\Delta t\right)$

- Represents the cumulative attenuation from the receiver to before the k-th voxel
- Exponential decay model conforms to the physical laws of electromagnetic wave propagation

**Term (2) - Local Radiation Contribution**:
$\rho(P_v^k) S(P_v^k, -\omega)$

- Represents the local radiation contribution of the k-th voxel
- Direction is $-\omega$ (opposite to the incident direction)

### 3.3 Total Received Signal

By summing over all sampling directions, we obtain the total received signal:

$S_{\Omega}(P_{\text{RX}}) \approx \sum_{\omega \in \Omega} S(P_{\text{RX}}, \omega)\,\Delta \omega$

Where $\Omega$ is the set of sampling directions, and $\Delta \omega$ is the directional sampling interval.

## 4. Training Objective Function

### 4.1 Signal Loss Function

Training loss is computed by comparing predicted aggregated signals with true measurements:

```math
\mathcal{L}_{\text{signal}} = \sum_{P_{\text{RX}}\sim \mathcal{D}} \left\|S_{\Omega}(P_{\text{RX}}) - \widehat{S}_{\Omega}(P_{\text{RX}}) \right\|_2$
```

Where:
- $\mathcal{D}$: Training dataset
- $\widehat{S}_{\Omega}(P_{\text{RX}})$: True measurement values
- $\|\cdot\|_2$: L2 norm

### 4.2 Loss Function Characteristics

- **Physical consistency**: Loss function is based on the physical laws of electromagnetic wave propagation
- **End-to-end training**: The entire model can be trained end-to-end
- **Numerical stability**: Discretization ensures numerical computation stability

## 5. Implementation Details

### 5.1 Numerical Computation Optimization

```python
# Pseudocode example
def compute_received_radiance(receiver_pos, direction, voxels, step_size):
    accumulated_radiance = 0
    cumulative_attenuation = 0
    
    for k in range(num_steps):
        voxel_pos = receiver_pos + k * step_size * direction
        voxel = get_voxel_at_position(voxel_pos)
        
        # Calculate cumulative attenuation
        if k > 0:
            cumulative_attenuation += voxel.attenuation * step_size
        
        # Calculate local radiation contribution
        local_radiance = voxel.attenuation * voxel.radiance(-direction)
        
        # Apply attenuation and accumulate
        attenuated_radiance = local_radiance * exp(-cumulative_attenuation)
        accumulated_radiance += attenuated_radiance * step_size
    
    return accumulated_radiance
```

### 5.2 Memory Optimization Strategies

- **Voxel caching**: Cache computed voxel properties
- **Batch processing**: Process multiple ray directions simultaneously
- **Sparse representation**: Use sparse storage for empty voxels

## 6. Physical Interpretability

### 6.1 Electromagnetic Wave Propagation Model

This discrete model maintains the physical interpretability of electromagnetic wave propagation:

1. **Attenuation accumulation**: Exponential decay model conforms to Beer-Lambert law
2. **Phase propagation**: Complex attenuation coefficients consider phase information
3. **Directionality**: Voxel radiation has directional dependence

### 6.2 Correspondence with Continuous Models

When voxel dimensions approach zero, the discrete model converges to the continuous radiance field model:

$\lim_{\Delta t \to 0} S(P_{\text{RX}}, \omega) = \int_0^L \exp\!\left(-\int_0^s \rho(s')\,ds'\right) \rho(s) S(s, -\omega)\,ds$

## 7. Performance Analysis

### 7.1 Computational Complexity

- **Time complexity**: $O(M \cdot N)$, where $M$ is the number of ray steps and $N$ is the number of voxels
- **Space complexity**: $O(N)$, mainly storing voxel properties and features

### 7.2 Accuracy vs. Efficiency Trade-offs

- **Step size selection**: Smaller step sizes improve accuracy but increase computation
- **Voxel resolution**: Higher resolution improves accuracy but increases memory requirements
- **Directional sampling**: More directional sampling improves angular resolution but increases computation

## 8. Optimization Strategies

### 8.1 Importance Sampling Optimization

#### 8.1.1 Discrete Ray Tracing Approximation

For practical implementation, continuous ray tracing integration is approximated by discretization. Specifically, directional rays over the interval $[0, D]$ (where $D$ represents maximum depth) are uniformly divided into $K$ segments, producing $K$ voxels at positions $\{P_{\text{v}}(t_1), P_{\text{v}}(t_2), \ldots, P_{\text{v}}(t_K)\}$. The received signal can be approximated as:

$S\big(P_{\text{RX}}, \omega\big) \approx \sum_{k=1}^{K} \exp\!\left(-\sum_{j=1}^{k-1} \rho(P_{\text{v}}(t_j)) \Delta t \right) \big(1 - e^{-\rho(P_{\text{v}}(t_k)) \Delta t}\big) S(P_{\text{v}}(t_k), -\omega)$

Where $\Delta t = D/K$.

#### 8.1.2 Non-uniform Sampling Strategy

Wireless propagation is inherently non-uniform—signals suffer significant attenuation in dense materials (such as concrete walls) but experience negligible loss in open air or sparse vegetation. To leverage this property, we adopt an importance sampling strategy, implementing non-uniform sampling along each ray in the RF domain.

**Coarse sampling stage**:
- Uniformly sample $K$ positions $\{P_{\text{v}}(t_k)\}_{k=1}^K$ along each propagation path
- Estimate attenuation factors $\hat{\rho}(P_{\text{v}}(t_k))$
- Use the real part $\beta_k=\Re(\hat{\rho}(P_{\text{v}}(t_k)))$ to construct sampling distribution

**Importance weight calculation**:
The unnormalized importance weight for the k-th segment is:

$w_k = \big(1 - e^{-\beta_k \Delta t}\big)\, \exp\!\Big(-\!\!\sum_{j<k}\beta_j\,\Delta t\Big)$

Weights $\{w_k\}$ are normalized to form a piecewise constant probability density function (PDF) along the ray.

**Fine sampling stage**:
- Resample rays by allocating more points to high-weight segments
- Draw $K$ stratified samples $u_i \sim \mathcal{U}[0,1]$
- For each $u_i$, determine corresponding position $t'_i$ such that $\mathrm{CDF}(t'_i) = u_i$
- Update segment lengths as $\Delta t_i = t'_i - t'_{i-1}$

### 8.2 Pyramid Sampling Optimization

#### 8.2.1 Directional Space Discretization

Continuous aggregated received signals in all directions are discretized into $L$ sampling directions:

$S_{\Omega}(P_{\text{RX}}) \approx \sum_{l=1}^{L} S(P_{\text{RX}}, \omega_l)$

#### 8.2.2 Pyramid Structure

The antenna-centered directional space is divided into $L = A \times B$ quadrilateral pyramids, where:
- $A$: Number of divisions for azimuth angle ($\phi$)
- $B$: Number of divisions for elevation angle ($\theta$)
- Each pyramid features: angular resolution $\Delta \theta = \pi/A$ and $\Delta \phi = 2\pi/B$
- Pyramid bases are located on a spherical shell with radius $D$

#### 8.2.3 Hierarchical Sampling Strategy

For a given direction:
1. **Radial importance sampling**: Place $K$ samples along the central axis
2. **Pyramid subdivision**: Subdivide the corresponding pyramid into $K$ consecutive truncated cones (truncated pyramids)
3. **Monte Carlo sampling**: Within each truncated cone, randomly draw $M$ uniformly distributed points
4. **Truncated cone level properties**: Truncated cone level attenuation factors, attenuation features, and radiation are defined as averages of these sampling points
5. **Ray tracing**: Averages participate in the ray tracing process

### 8.3 Optimization Effects

#### 8.3.1 Computational Efficiency Improvement

- **Importance sampling**: Concentrates computational resources in high-attenuation regions, reducing ineffective computation
- **Pyramid sampling**: Avoids tracing along infinitesimal rays, improving numerical stability
- **Adaptive sampling**: Dynamically adjusts sampling density based on scene characteristics

#### 8.3.2 Accuracy Preservation

- **Non-uniform sampling**: Maintains high sampling density in high-attenuation regions
- **Hierarchical strategy**: Ensures uniform coverage of directional space
- **Monte Carlo integration**: Improves integration accuracy within truncated cones

#### 8.3.3 Memory Optimization

- **Sparse representation**: Use sparse storage for transparent regions
- **Hierarchical caching**: Cache computation results at different levels
- **Batch processing**: Process multiple pyramids and rays simultaneously

## 9. Application Scenarios

### 9.1 Indoor Propagation Modeling

- Electromagnetic wave propagation inside buildings
- Multipath effect modeling
- Signal strength prediction

### 9.2 Wireless Communication Systems

- 5G/6G network planning
- Antenna array optimization
- Interference analysis

### 9.3 Radar Systems

- Target detection and tracking
- Clutter modeling
- Signal processing algorithm validation

## 10. Future Improvement Directions

### 10.1 Algorithm Optimization

- **Adaptive step size**: Dynamically adjust step size based on scene complexity
- **Parallel computation**: Utilize GPU parallelization for ray tracing
- **Hierarchical representation**: Multi-resolution voxel representation

### 10.2 Model Extensions

- **Time-varying scenarios**: Support dynamic scene modeling
- **Multi-frequency modeling**: Consider frequency dependence
- **Polarization effects**: Include electromagnetic wave polarization information

## 11. Summary

The discrete radiance field model achieves efficient electromagnetic wave propagation modeling by discretizing continuous 3D scenes into voxel grids. This model offers the following advantages:

1. **Physical accuracy**: Maintains the physical laws of electromagnetic wave propagation
2. **Computational efficiency**: Discretization enables efficient numerical computation
3. **Interpretability**: Each voxel's contribution has clear physical meaning
4. **Scalability**: Easy to integrate into larger systems

This model provides a powerful technical foundation for indoor propagation modeling, wireless communication system design, and radar system analysis.

---

*This document is based on the technical implementation of the Prism project. For questions or suggestions, please contact the development team.*



