# Training Design Document

## Overview

This document outlines the training methodology and implementation for the Prism model, which extends NeRF2 architecture to handle wideband RF signals in Orthogonal Frequency-Division Multiplexing (OFDM) scenarios. The training system is designed to efficiently train neural networks that can model complex RF signal propagation through 3D environments with multiple subcarriers and MIMO antenna configurations.

## 1. Training Architecture

### 1.1 Model Components for Training

The Prism model consists of four main neural network components that are trained simultaneously:

#### 1.1.1 AttenuationNetwork
- **Purpose**: Encodes spatial position information into compact 128-dimensional feature representations
- **Architecture**: 8-layer MLP with ReLU activation
- **Input**: 3D position coordinates (x, y, z)
- **Output**: 128-dimensional feature vector

#### 1.1.2 AttenuationDecoder
- **Purpose**: Converts 128D spatial features into `N_BS × N_UE` attenuation factors
- **Architecture**: `N_UE` independent 3-layer MLPs
- **Input**: 128D features from AttenuationNetwork
- **Output**: Attenuation factors for each BS-UE antenna combination

#### 1.1.3 RadianceNetwork
- **Purpose**: Processes UE position, viewing direction, and spatial features
- **Architecture**: 8-layer MLP with ReLU activation
- **Input**: UE position, viewing direction, and spatial features
- **Output**: Radiance values for signal propagation modeling

#### 1.1.4 AntennaNetwork
- **Purpose**: Process antenna embeddings to generate directional importance indicators for efficient ray tracing
- **Architecture**: Shallow network (64D → 128D → directional importance values)
- **Input**: 64-dimensional antenna embedding from antenna codebook
- **Output**: Directional importance matrix indicating importance of each direction
- **Key Features**: 
  - Enables efficient directional sampling for computational efficiency
  - Guides ray tracing to focus on antenna-specific important directions

## 2. Training Pipeline

### 2.1 Data Loading and Preprocessing

The training system handles complex multi-dimensional data including:

- **Spatial positions**: 3D coordinates for base stations, UEs, and sampling points
- **Subcarrier data**: Multiple subcarrier frequencies for OFDM signals
- **Antenna configurations**: MIMO setups with multiple UE and BS antennas
- **Ray tracing data**: AntennaNetwork-guided directional sampling and spatial point sampling

### 2.2 Training Loop Implementation

The training process follows this workflow:

1. **Data Loading**: Load training samples with proper tensor formatting
2. **Forward Pass**: Process data through all four network components
3. **BS-Centric Ray Tracing**: Start ray tracing from each BS antenna
4. **AntennaNetwork Direction Selection**: Use AntennaNetwork to suggest important directions for ray tracing
5. **Subcarrier Sampling**: Randomly select `K' < K` subcarriers per antenna for computational efficiency
6. **CSI Prediction**: Calculate predicted CSI for selected subcarriers on each BS antenna
7. **Loss Computation**: Compute MSE between predicted CSI and ground truth CSI from real measurements
8. **Backward Pass**: Compute gradients and update model parameters
9. **Validation**: Periodic validation to monitor training progress

## 3. Loss Functions and Optimization

### 3.1 CSI Loss Functions

The choice of loss function is critical for training neural networks to predict complex-valued Channel State Information (CSI). This section presents several loss function approaches, from simple to sophisticated, for optimizing CSI prediction accuracy.

#### 3.1.1 Complex MSE

**The most commonly used and direct approach** that simultaneously optimizes both magnitude and phase components.

```math
\mathcal{L}_{\text{CMSE}} = \frac{1}{N} \sum_{i=1}^{N} \left\| \mathbf{H}_{\text{pred}}^{(i)} - \mathbf{H}_{\text{true}}^{(i)} \right\|_F^2 = \frac{1}{N} \sum_{i=1}^{N} \left( |\Delta \mathbf{H}_i| \right)^2
```

where `Δ H_i = H_pred^(i) - H_true^(i)` is the complex error.

**Advantages:**
- Simple computation with clear physical meaning (Euclidean distance between complex vectors)
- Naturally handles both magnitude and phase optimization
- Stable gradients and well-behaved optimization

**Disadvantages:**
- Equal weighting for magnitude and phase errors may not be optimal for all applications

#### 3.1.2 Magnitude + Phase Loss

Separate handling of magnitude and phase components with adjustable weighting.

```math
\mathcal{L}_{\text{Mag+Phase}} = \alpha \cdot \frac{1}{N} \sum_{i=1}^{N} \left( |\mathbf{H}_{\text{pred}}^{(i)}| - |\mathbf{H}_{\text{true}}^{(i)}| \right)^2 + \beta \cdot \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{\text{Phase}}^{(i)}
```

The **phase loss `L_Phase` is critical** and can be defined in two ways:

**Option 1: Corrected Phase Difference MSE** (handles `2π` wrapping):
```math
\mathcal{L}_{\text{Phase}}^{(i)} = \left( \arctan2\left( \sin(\Delta\theta_i), \cos(\Delta\theta_i) \right) \right)^2
```
where `Δθ_i = ∠ H_pred^(i) - ∠ H_true^(i)`.

**Option 2: Complex Cosine Similarity** (more stable):
```math
\mathcal{L}_{\text{Phase}}^{(i)} = 1 - \frac{ | \langle \mathbf{H}_{\text{pred}}^{(i)}, \mathbf{H}_{\text{true}}^{(i)} \rangle | }{ |\mathbf{H}_{\text{pred}}^{(i)}| \cdot |\mathbf{H}_{\text{true}}^{(i)}| } = 1 - |\cos(\Delta\theta_i)|
```

**Advantages:**
- Flexible weighting between magnitude and phase importance
- Suitable for applications where phase accuracy is critical (e.g., beamforming)

**Disadvantages:**
- Requires manual tuning of hyperparameters `α` and `β`

#### 3.1.3 Correlation-Based Loss

Focuses on structural similarity between CSI vectors rather than absolute errors.

```math
\mathcal{L}_{\text{Corr}} = 1 - \frac{ | \langle \mathbf{H}_{\text{pred}}, \mathbf{H}_{\text{true}} \rangle | }{ \|\mathbf{H}_{\text{pred}}\|_F \cdot \|\mathbf{H}_{\text{true}}\|_F } = 1 - \frac{ | \sum_{i=1}^{N} \mathbf{H}_{\text{pred}}^{(i)} \cdot (\mathbf{H}_{\text{true}}^{(i)})^* | }{ \sqrt{ \sum |\mathbf{H}_{\text{pred}}^{(i)}|^2 } \cdot \sqrt{ \sum |\mathbf{H}_{\text{true}}^{(i)}|^2 } }
```

**Advantages:**
- Invariant to global magnitude scaling (same loss for `H_pred` and `k · H_pred`)
- Emphasizes **structural similarity**, directly related to communication performance metrics
- Robust to amplitude variations while preserving phase relationships

**Disadvantages:**
- May need combination with other losses if absolute magnitude values are important

#### 3.1.4 Hybrid CSI Loss Function (Recommended)

For optimal performance, we recommend combining the strengths of multiple approaches:

```math
\mathcal{L}_{\text{Total}} = \lambda_1 \cdot \mathcal{L}_{\text{CMSE}} + \lambda_2 \cdot \mathcal{L}_{\text{Corr}}
```

**Rationale:**
- `L_CMSE` ensures **absolute value accuracy**
- `L_Corr` ensures **structural similarity**, critical for application performance
- The combination provides both stability and effectiveness
- Typically `λ₁ = λ₂ = 1` works well in practice

### 3.2 PDP Loss Functions

Power Delay Profile (PDP) loss functions provide a time-domain perspective for validating CSI predictions by comparing the delay-domain characteristics of predicted and true CSI. This approach is particularly valuable for ensuring that the predicted CSI maintains correct temporal structure, which is crucial for applications like positioning and beamforming.

#### 3.2.1 Core Computation Flow

The PDP loss computation follows this workflow for comparing **predicted CSI** and **true CSI** (both on randomly selected subcarriers):

1. **Zero Padding**: Pad both predicted and true CSI to the same complete frequency sequence (e.g., 1024 points)
2. **PDP Calculation**: Apply IFFT to zero-padded frequency data and compute power delay profile
   - `PDP_pred = |IFFT(CSI_pred,padded)|²`
   - `PDP_true = |IFFT(CSI_true,padded)|²`
3. **PDP Comparison**: Calculate differences between the two PDPs as loss function or evaluation metric

```mermaid
flowchart TD
    A[Predicted CSI<br/>Random Subcarriers] --> B[Frequency Domain<br/>Zero Padding]
    C[True CSI<br/>Same Subcarriers] --> D[Frequency Domain<br/>Zero Padding]
    
    B --> E[Compute PDP<br/>IFFT + Power]
    D --> F[Compute PDP<br/>IFFT + Power]
    
    E --> G[Compare PDP Differences]
    F --> G
    
    G --> H[Calculate PDP Loss]
```

#### 3.2.2 Mean Squared Error (MSE) PDP Loss

The most direct approach, computing MSE between PDPs at each delay bin:
```math
\mathcal{L}_{\text{PDP,MSE}} = \frac{1}{M} \sum_{m=1}^{M} \left( \text{PDP}_{\text{pred,norm}}[m] - \text{PDP}_{\text{true,norm}}[m] \right)^2
```

**Advantages:**
- Simple computation and implementation
- Direct comparison of delay-domain characteristics
- Well-suited for applications requiring precise delay profile matching

**Disadvantages:**
- Sensitive to normalization choices
- May be affected by zero-padding artifacts

#### 3.2.3 Correlation-Based PDP Loss

Focuses on shape similarity rather than absolute values:
```math
\rho = \text{corrcoef}(\text{PDP}_{\text{pred}}, \text{PDP}_{\text{true}})
```
```math
\mathcal{L}_{\text{PDP,corr}} = 1 - \rho
```
where ideally `ρ = 1` and `L_PDP,corr = 0`.

**Advantages:**
- Robust to absolute power scaling differences
- Emphasizes structural similarity in delay domain
- Less sensitive to normalization artifacts

**Disadvantages:**
- May ignore important absolute timing information
- Requires careful handling of noise and artifacts

#### 3.2.4 Dominant Path Feature Loss

Extracts and compares key characteristics:
- **Dominant path delay error**: `L_delay = |argmax(PDP_pred) - argmax(PDP_true)|`
- **RMS delay spread error**: `L_spread = |σ_pred - σ_true|`
- **Dominant path power ratio error**: `L_power = |max(PDP_pred)/∑PDP_pred - max(PDP_true)/∑PDP_true|`

**Advantages:**
- Focuses on physically meaningful parameters
- Directly relates to communication performance metrics
- Robust to detailed shape variations

**Disadvantages:**
- May miss important multipath structure details
- Requires domain expertise for parameter selection

#### 3.2.5 Hybrid PDP Loss Function (Recommended)

For optimal performance, we recommend combining multiple PDP loss approaches:

```math
\mathcal{L}_{\text{PDP,total}} = \alpha \cdot \mathcal{L}_{\text{PDP,MSE}} + \beta \cdot \mathcal{L}_{\text{PDP,corr}} + \gamma \cdot \mathcal{L}_{\text{delay}}
```

**Rationale:**
- `L_PDP,MSE` ensures **detailed delay profile accuracy**
- `L_PDP,corr` ensures **structural similarity** in time domain
- `L_delay` ensures **dominant path timing accuracy**
- The combination provides comprehensive time-domain validation
- Typically `α = 0.5`, `β = 0.3`, `γ = 0.2` works well in practice

**Advantages:**
- Comprehensive time-domain validation
- Balances detailed accuracy with structural correctness
- Robust to various types of prediction errors

**Disadvantages:**
- Requires tuning of multiple hyperparameters
- Higher computational cost than individual methods

**Implementation Considerations:**

**Normalization Requirements:**
Before comparison, PDPs must be **normalized** to the same total energy or peak energy to focus on shape rather than absolute power:
- Peak normalization: `PDP_norm = PDP / max(PDP)`
- Energy normalization: `PDP_norm = PDP / ∑(PDP)`

**Subcarrier Alignment:**
Ensure that predicted and true CSI come from **identical subcarrier indices** with identical zero-padding patterns.

**Handling Artifacts:**
Zero-padding introduces sidelobe effects in both predicted and true PDPs. The loss remains valid as long as:
- Both PDPs exhibit consistent artifact patterns
- Relative positions and strengths of true multipath peaks are preserved

**Physical Significance:**
PDP loss functions offer several advantages over frequency-domain comparisons:
- **Time-domain validation**: Ensures predicted CSI maintains correct **delay-domain structure**
- **Physical meaningfulness**: Directly relates to multipath propagation characteristics
- **Application relevance**: Critical for positioning, beamforming, and channel modeling applications
- **Robustness**: Less sensitive to individual subcarrier errors while emphasizing overall temporal structure

**Recommendation:** Use PDP loss as a complementary validation metric alongside frequency-domain CSI losses. This dual-domain approach ensures both numerical accuracy and physical correctness of the predicted channel characteristics.

### 3.3 Loss Function Selection Guide

| Application Focus | Recommended Loss Function |
|:-----------------|:-------------------------|
| **General Purpose** | **Complex MSE** (`L_CMSE`) |
| **Phase Accuracy Critical** | **Magnitude + Phase Loss** (`L_Mag+Phase`) |
| **Beamforming/Correlation** | **Correlation Loss** (`L_Corr`) |
| **Best Performance** | **Hybrid CSI Loss** (`L_CMSE + L_Corr`) |
| **Time-domain Validation** | **PDP Loss** (`L_PDP`) |
| **Comprehensive Validation** | **Hybrid PDP Loss** (`L_PDP,total`) |
| **Production Use** | **Hybrid CSI+PDP Loss** (`L_total`) |


