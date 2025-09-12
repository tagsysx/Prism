# Ray Tracing Design Document

## Overview

This document outlines the design and implementation of the discrete electromagnetic ray tracing system for the Prism project. The system implements an efficient voxel-based ray tracing approach that combines discrete radiance field modeling with advanced optimization strategies to achieve both accuracy and computational efficiency.

The core concept is that the system computes **complex RF signals** $S_{\text{ray}}$ at the base station's antenna from each direction, which includes both amplitude and phase information of the electromagnetic wave. **Critical Design Principle**: All ray tracing computations maintain complex number representation throughout the entire pipeline - from neural network outputs to final signal accumulation. Only during loss computation are complex signals converted to real values for MSE calculation. A key architectural advantage is that **ray tracing operations are independent**, enabling efficient parallelization using CUDA or multi-threading for significant performance acceleration.

## 1. Core Design

### 1.1 Base Station Configuration

The ray tracing system is centered around base stations (BS) with configurable locations:
- **Default configuration**: Base station positioned at the origin $(0, 0, 0)$
- **Customizable positioning**: Base station location $P_{\text{BS}}$ can be configured for different deployment scenarios
- **Multi-antenna support**: Each base station can have multiple antennas for MIMO operations

### 1.2 Ray Tracing Principle

**Critical Concept**: Ray tracing is performed from the BS antenna outward in all directions, **not** from UE to BS or based on UE distance:

- **Ray Origin**: Always starts from BS antenna position (configurable, defaults to $(0, 0, 0)$)
- **Ray Direction**: Fixed directional vectors in the $A \times B$ grid
- **Ray Length**: Fixed maximum length (`max_ray_length`), independent of UE positions
- **UE Role**: UE positions are **only** used as inputs to the RadianceNetwork to determine how sampling points radiate toward different UEs
- **Sampling Independence**: Ray sampling points are determined solely by the ray direction and maximum length, not by UE locations

### 1.3 Directional Space Division

The antenna's directional space is discretized into a structured grid:
- **Grid dimensions**: $A \times B$ directions
- **Azimuth divisions**: $A$ divisions covering the horizontal plane
- **Elevation divisions**: $B$ divisions covering the vertical plane
- **Angular resolution**: $\Delta \phi = \frac{2\pi}{A}$ (azimuth: 0° to 360°) and $\Delta \theta = \frac{\pi}{B}$ (elevation: -90° to +90°)

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
- $\omega_{ij} = [\cos(\theta_j)\cos(\phi_i), \cos(\theta_j)\sin(\phi_i), \sin(\theta_j)]$ where $\theta_j$ is elevation angle from -90° to +90°

### 2.2 BS-Centric Radiation Pattern

**Core Principle**: The ray tracing system adopts a **BS antenna-centric radiation approach**, where electromagnetic waves are traced from the base station antenna as the central radiation source outward into its entire directional space.

**Radiation Geometry**:
- **Central Source**: BS antenna positioned at $P_{BS}$ serves as the electromagnetic radiation center
- **Omnidirectional Coverage**: Rays are cast in all $A \times B$ directions covering the complete spherical space around the antenna
- **Fixed Ray Length**: All rays extend to a maximum distance `max_ray_length`, independent of UE positions
- **Uniform Angular Sampling**: Directional space is uniformly discretized to ensure comprehensive coverage

**Physical Significance**:
```
BS Antenna (Center) → Ray Direction 1 → Sampling Points → Signal Propagation
                   → Ray Direction 2 → Sampling Points → Signal Propagation  
                   → Ray Direction 3 → Sampling Points → Signal Propagation
                   → ...
                   → Ray Direction A×B → Sampling Points → Signal Propagation
```

**Key Design Decisions**:
1. **Ray Origin**: Always starts from BS antenna position $P_{BS}$
2. **Direction Independence**: Ray directions are predetermined by the $A \times B$ grid, not influenced by UE locations
3. **Comprehensive Coverage**: Every direction in the antenna's radiation pattern is systematically sampled
4. **UE Role**: UE positions serve only as radiation targets for the RadianceNetwork, not as ray endpoints

### 2.3 View Direction Calculation

**Unified Implementation**: All ray tracers use a consistent view direction calculation implemented in the base class:


**Physical Interpretation**:
- **Direction Definition**: From each sampling point toward the BS antenna (radiation center)
- **RadianceNetwork Usage**: Determines how electromagnetic energy radiates from sampling points toward the central receiving antenna
- **Consistency**: All ray tracer implementations (CPU, CUDA) use the same unified calculation
- **Simplification**: BS antenna position is treated as a point source (actual antenna array geometry is simplified)

### 2.4 Energy Tracing Process

For each antenna of the base station, the system traces RF energy along all $A \times B$ directions:

1. **Directional initialization**: Initialize ray parameters for each direction
2. **Energy propagation**: Trace electromagnetic energy along each ray using the discrete radiance field model
3. **Attenuation modeling**: Apply material-dependent attenuation coefficients at each voxel intersection
4. **Energy accumulation**: Compute cumulative energy received at user equipment (UE) locations

### 2.3 Uniform Sampling Strategy

The system uses uniform sampling along rays for computational efficiency and simplicity:

#### 2.3.1 Uniform Ray Sampling

**Sampling Process**:
The system implements uniform sampling along each ray direction:


**Benefits of Uniform Sampling**:
- **Simplicity**: Straightforward implementation without complex weight calculations
- **Predictable performance**: Consistent computational cost across different scenarios
- **Numerical stability**: Avoids potential issues with importance weight computation
- **Memory efficiency**: No need to store and compute importance weights

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

### 3.3 Complex RF Signal Computation

The ray tracer computes **complex RF signals** using the discrete radiance field model as specified in SPECIFICATION.md. **All computations preserve complex number representation** to maintain both amplitude and phase information throughout the ray tracing process. For each ray direction, the complex signal is computed using the precise formula:

```math
S(P_{\text{RX}}, \omega) \approx \sum_{k=1}^{K} \exp\!\left(-\sum_{j=1}^{k-1} \rho(P_{\text{v}}(t_j)) \Delta t_j \right) \big(1 - e^{-\rho(P_{\text{v}}(t_k)) \Delta t_k}\big) S(P_{\text{v}}(t_k), -\omega)
```

Where:
- $P_{\text{RX}}$: Receiver (UE) position
- $\omega$: Ray direction from receiver toward transmitter
- $K$: Number of voxels along the ray
- $\rho(P_{\text{v}}(t_k))$: Complex attenuation coefficient at voxel $k$ (from AttenuationDecoder)
- $S(P_{\text{v}}(t_k), -\omega)$: Radiance at voxel $k$ in direction $-\omega$ (from RadianceNetwork)
- $\Delta t_k = t_k - t_{k-1}$: Path length through voxel $k$

#### 3.3.1 Neural Network Integration

The ray tracing system integrates with four neural networks:

1. **AttenuationNetwork**: $f_\theta(\text{IPE}(P_v)) \to \mathcal{F}(P_v)$ (128D features)
2. **AttenuationDecoder**: $f_\delta(\mathcal{F}(P_v)) \to \rho(P_v)$ (complex attenuation coefficients)
3. **RadianceNetwork**: $f_\psi(\mathcal{F}(P_v), \text{IPE}(P_{\text{UE}}), \text{IPE}(\omega), C) \to S(P_v, \omega)$ (complex radiance values)
4. **AntennaNetwork**: $f_\alpha(C) \to M_{ij}$ (directional importance matrix for top-K sampling)

### 3.4 Complex Number Preservation Throughout Ray Tracing

**Critical Implementation Requirement**: The ray tracing system maintains complex number representation at every stage of computation to preserve both amplitude and phase information of electromagnetic waves.

#### 3.4.1 Complex Signal Flow


#### 3.4.2 Physical Significance

**Why Complex Numbers Are Essential**:
- **Amplitude Information**: `|z|` represents signal strength/power
- **Phase Information**: `arg(z)` represents wave phase/timing
- **Coherent Superposition**: Complex addition correctly models wave interference
- **Frequency Domain**: Natural representation for OFDM subcarriers
- **Channel State Information**: CSI inherently complex-valued in wireless systems



### 3.5 Virtual Link Computation

The ray tracer computes accumulated RF signal strength for the optimized set of virtual links:

```math
S_{\text{accumulated}}(P_{\text{UE}}, f_k, C) = \sum_{i=1}^{A} \sum_{j=1}^{B} S_{\text{ray}}(\phi_i, \theta_j, P_{\text{UE}}, f_k, C)
```

Where:
- $P_{\text{UE}}$: UE position
- $f_k$: Selected subcarrier frequency
- $C$: Base station's antenna embedding parameter
- $S_{\text{ray}}(\phi_i, \theta_j, P_{\text{UE}}, f_k, C)$: RF signal strength received from direction $(\phi_i, \theta_j)$ at frequency $f_k$ with antenna embedding $C$

## 4. Low-Rank Factorization in Ray Tracing

### 4.1 Factorized Ray Tracing Theory

Now, we investigate whether the ray tracing procedure itself can be factorized in a manner consistent with the low-rank representations of attenuation and radiance. Assuming both the attenuation coefficient $\rho_f(P)$ and radiance $S_f(P, \omega)$ admit separable low-rank forms, we examine how this structure propagates through the rendering equation.

To this end, we extend the discrete ray-tracing formula by explicitly introducing frequency dependence through $\rho_f$ and $S_f$:

$$\small\label{eqn:simple-ray-tracing}
\begin{aligned}
S_f\big(P_{\mathrm{RX}}, \omega\big) 
&= \sum_{k=1}^{K} 
   e^{-\sum_{j=1}^{k-1} \rho_f(P_j)\,\Delta t}
   \big(1-e^{-\rho_f(P_k)\,\Delta t}\big)\,
   S_f(P_k, \omega) \\[4pt]
&\approx \sum_{k=1}^{K} 
   \Big(1 - \sum_{j=1}^{k-1} \rho_f(P_j)\,\Delta t\Big)\,
   \rho_f(P_k)\, S_f(P_k, \omega)\,\Delta t \\[4pt]
&= \sum_{k=1}^{K} H_f(P_k)\,\rho_f(P_k)\, S_f(P_k, \omega)\,\Delta t,
\end{aligned}
$$

where the first-order Taylor approximations $(1-e^{-x}) \approx x$ and $e^{-x}\approx 1-x$ are applied to linearize the exponential attenuation terms, yielding a form that is more amenable to factorization analysis. The channel coefficient from the sample $k$ to the receiver is defined as:

$$\small
H_f(P_k) \;=\; 
\exp\!\Big(-\sum_{j=1}^{k-1} \rho_f(P_j)\,\Delta t\Big)
\;\approx\; 1 - \sum_{j=1}^{k-1} \rho_f(P_j)\,\Delta t.
$$

where $k=2,4,\dots, K$.

### 4.2 Channel Factorization

Let us first study the factorizability of the channel coefficient $H_f(P_k)$ under the first-order Taylor approximation:

$$\small
\begin{aligned}
	H_f(P_k) &=  1 - \sum_{j=1}^{k-1} \rho_f(P_j) \Delta t.
\end{aligned}
$$

Substituting the low-rank attenuation representation from Eqn.~\ref{eqn:low-rank-rho} into the summation yields:

$$\small
\begin{aligned}
\sum_{j=1}^{k-1} \rho_f(P_j)\,\Delta t &= \sum_{j=1}^{k-1} \left( \sum_{r=1}^R U^\rho_r(P_j)^* V_r(f) \right) \Delta t \\
&= \sum_{r=1}^R V_r(f) \left( \sum_{j=1}^{k-1} U^\rho_r(P_j)^* \Delta t \right).
\end{aligned}
$$

We now define a frequency-independent cumulative spatial vector $\widehat{U}^\rho(P_k)$ that aggregates attenuation contributions along the ray path:

$$\footnotesize
\widehat{U}^\rho(P_k)^* = \left[ \underbrace{\sum_{j=1}^{k-1} U^\rho_1(P_j)^* \Delta t}_{\widehat{U}^\rho_1 (P_k)^*}, \underbrace{\sum_{j=1}^{k-1} U^\rho_2(P_j)^* \Delta t}_{\widehat{U}^\rho_2 (P_k)^*}, \dots, \underbrace{\sum_{j=1}^{k-1} U^\rho_R(P_j)^* \Delta t}_{\widehat{U}^\rho_R (P_k)^*} \right]^\top.
$$

This vector captures the integrated spatial attenuation of all $R$ latent components from voxel $P_k$ to the RX. Crucially, $\widehat{U}^\rho(P_k)$ depends solely on scene geometry and material properties and is independent of frequency $f$, enabling its precomputation during ray marching for efficient reuse across all frequencies.

Consequently, the sum can now be expressed compactly using the Hermitian form:

$$\small
\sum_{j=1}^{k-1} \rho_f(P_j)\,\Delta t = \langle \widehat{U}^\rho(P_k), V(f)\rangle = \sum_{r=1}^R \widehat{U}^\rho_r(P_k)^* V_r(f),
$$

where the Hermitian transpose ensures proper complex conjugation for the inner product operation. Finally, the channel coefficient is factorized to:

$$\label{eqn:h-factor}
H_f (P_k)  \approx 1 - \langle \widehat{U}^\rho(P_k), V(f)\rangle .
$$

Consequently, evaluating $H_f(P_k)$ for any frequency reduces to a simple inner product between frequency-independent spatial feature accumulations and frequency-dependent spectral basis.

### 4.3 Ray-Tracing Factorization

Revisiting Eqn.~\ref{eqn:simple-ray-tracing}, we observe that all key terms—$H_f$, $\rho_f$, and $S_f$—are now expressed in factorizable form. By substituting Eqn.~\ref{eqn:h-factor}, Eqn.~\ref{eqn:low-rank-rho}, and Eqn.~\ref{eqn:low-rank-s} into Eqn.~\ref{eqn:simple-ray-tracing}, the ray-tracing equation simplifies into a compact expression where the frequency dependence is fully separated:

$$\footnotesize
\boxed{S_f(P_{\mathrm{RX}}, \omega) \approx
\left\langle \boldsymbol{\mathcal{U}}^{(1)}(\omega),\ \boldsymbol{\mathcal{V}}^{(1)}(f) \right\rangle
\left\langle \boldsymbol{\mathcal{U}}^{(2)}(\omega),\ \boldsymbol{\mathcal{V}}^{(2)}(f) \right\rangle}
$$

where

$$\small
\begin{cases}
\boldsymbol{\mathcal{U}}^{(1)}(\omega) = \sum\limits_{k=1}^{K} \big(U^S(P_k, -\omega) \otimes U^\rho(P_k)\big) \Delta t  \\
\boldsymbol{\mathcal{U}}^{(2)}(\omega) = -\sum\limits_{k=1}^{K} \big(U^S(P_k, -\omega) \otimes U^\rho(P_k) \otimes \widehat{U}^\rho(P_k)\big) \Delta t 
\end{cases}
$$

and

$$\small
\begin{cases}
\boldsymbol{\mathcal{V}}^{(1)}(f) &= V(f) \otimes V(f)  \\
\boldsymbol{\mathcal{V}}^{(2)}(f) &= V(f) \otimes V(f) \otimes V(f) 
\end{cases}
$$

Here, $\boldsymbol{\mathcal{U}^{(1)}}$ and $\boldsymbol{\mathcal{U}}^{(2)}$ are the consolidated frequency-agnostic spatial tensors that aggregate all path-dependent scene interactions, while $\boldsymbol{\mathcal{V}}^{(1)}$ and $\boldsymbol{\mathcal{V}}^{(2)}$ are the spectral tensors that combine frequency components of different orders. For clarity, the detailed derivation of this result is shown in Appendix~\ref{appendix:ray-tracing}.

Conceptually, this factorization shows that ray tracing reduces to weighted combinations of spatial feature components (i.e., $U^\rho(P_k)$, $U^S(P_k, -\omega)$, and $\widehat{U}^\rho(P_k)$) paired with spectral basis functions $V(f)$. Since $\boldsymbol{\mathcal{U}}$ are frequency independent, they can be precomputed once. Evaluating the received signal at a new frequency $f$ then requires only combining the precomputed $\boldsymbol{\mathcal{V}}$ with the corresponding $V(f)$, achieving the elegant goal of **"one trace, all tones!"**

## 5. Integration with Discrete Radiance Field Model

The ray tracing system integrates seamlessly with the discrete radiance field model:

1. **Voxel interaction**: Rays intersect with voxels to determine material properties
2. **Attenuation modeling**: Complex attenuation coefficients applied at each intersection
3. **Signal propagation**: Exponential decay model for cumulative attenuation
4. **Radiation calculation**: Direction-dependent voxel radiation properties
5. **Low-rank factorization**: Frequency-independent spatial features enable efficient multi-frequency computation

