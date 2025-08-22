# Training Design Document

## Overview

This document outlines the training methodology and implementation for the Prism model, which extends NeRF2 architecture to handle wideband RF signals in Orthogonal Frequency-Division Multiplexing (OFDM) scenarios. The training system is designed to efficiently train neural networks that can model complex RF signal propagation through 3D environments with multiple subcarriers and MIMO antenna configurations.

## 1. Training Architecture

### 1.1 Model Components for Training

The Prism model consists of three main neural network components that are trained simultaneously:

#### 1.1.1 AttenuationNetwork
- **Purpose**: Encodes spatial position information into compact 128-dimensional feature representations
- **Architecture**: 8-layer MLP with ReLU activation
- **Input**: 3D position coordinates (x, y, z)
- **Output**: 128-dimensional feature vector
- **Training**: Learns to extract meaningful spatial features that capture environmental characteristics

#### 1.1.2 AttenuationDecoder
- **Purpose**: Converts 128D spatial features into M×N_UE attenuation factors
- **Architecture**: N_UE independent 3-layer MLPs
- **Input**: 128D features from AttenuationNetwork
- **Output**: Attenuation factors for each subcarrier and UE antenna combination
- **Training**: Learns to decode spatial features into frequency-dependent attenuation patterns

#### 1.1.3 RadianceNetwork
- **Purpose**: Processes UE position, viewing direction, and spatial features
- **Architecture**: 8-layer MLP with ReLU activation
- **Input**: UE position, viewing direction, and spatial features
- **Output**: Radiance values for signal propagation modeling
- **Training**: Learns to model directional signal characteristics

#### 1.1.4 AntNetwork
- **Purpose**: Process antenna embeddings to generate directional importance indicators for efficient directional sampling
- **Architecture**: Shallow network (64D → 128D → A×B importance values)
- **Input**: 64-dimensional antenna embedding from antenna codebook
- **Output**: A×B directional importance matrix indicating importance of each direction
- **Training**: Learns antenna-specific directional preferences and importance patterns
- **Key Features**: 
  - Enables top-K directional sampling for computational efficiency
  - Reduces directional space from A×B to K important directions
  - Lightweight processing with minimal computational overhead
  - Configurable angular resolution (A azimuth × B elevation divisions)

## 2. Training Pipeline

### 2.1 Data Loading and Preprocessing

The training system handles complex multi-dimensional data including:

- **Spatial positions**: 3D coordinates for base stations, UEs, and sampling points
- **Frequency data**: Multiple subcarrier frequencies for OFDM signals
- **Antenna configurations**: MIMO setups with multiple UE and BS antennas
- **Ray tracing data**: Directional sampling and spatial point sampling

### 2.2 Training Loop Implementation

The training process follows this workflow:

1. **Data Loading**: Load training samples with proper tensor formatting
2. **Forward Pass**: Process data through all three network components
3. **Loss Computation**: Calculate frequency-aware loss between predictions and targets
4. **Backward Pass**: Compute gradients and update model parameters
5. **Validation**: Periodic validation to monitor training progress

## 3. Loss Functions and Optimization

### 3.1 Frequency-Aware Loss Function

The PrismLoss class implements specialized loss functions for multi-subcarrier RF signals:

- **MSE Loss**: Standard mean squared error for signal reconstruction
- **Frequency-Weighted Loss**: Apply different weights to different subcarriers
- **Complex MSE Loss**: Handle magnitude and phase components separately

### 3.2 Optimization Strategy

**Adam Optimizer**: Adaptive learning rate optimization with configurable parameters

**Learning Rate Scheduling**: Step-based or cosine annealing scheduling

**Gradient Clipping**: Prevents exploding gradients for training stability

**Weight Decay**: L2 regularization to prevent overfitting

## 4. Training Configuration

### 4.1 Configuration File Structure

Training parameters are configured through YAML configuration files with sections for:

- **Basic Parameters**: Number of epochs, batch size, learning rate
- **Optimization Settings**: Optimizer type, scheduler, gradient clipping
- **Loss Configuration**: Loss type and frequency weights
- **Training Strategies**: Early stopping, checkpoint saving, validation intervals

### 4.2 Hyperparameter Tuning

**Learning Rate**: 1e-4 (empirically determined for stable convergence)

**Batch Size**: 32-64 (balanced between memory constraints and training efficiency)

**Network Architecture**: 256 hidden dimensions, 8 layers (optimal for spatial encoding)

## 5. Advanced Training Features

### 5.1 CSI Virtual Link Processing

The training system incorporates CSI virtual link processing for enhanced channel modeling:

- **Virtual Link Creation**: Generate M×N_UE uplink combinations
- **Random Sampling**: Efficiently sample K virtual links for computational efficiency
- **Batch Diversity**: Different random seeds ensure diverse sampling across batches

### 5.2 Ray Tracing Integration

Ray tracing is integrated into the training process for realistic signal propagation modeling:

- **Multi-Angle Sampling**: 36 azimuth × 18 elevation angle combinations
- **Spatial Point Sampling**: 64 sampling points per ray for high-resolution modeling
- **Material Properties**: Environment-specific attenuation and reflection modeling

### 5.3 AntNetwork Training

The AntNetwork is trained to learn antenna-specific directional importance patterns:

- **Directional Importance Learning**: Train network to predict which directions are most important for each antenna
- **Top-K Sampling Training**: Learn to identify K most critical directions from A×B directional grid
- **Antenna-Specific Patterns**: Each antenna learns its own directional preferences during training
- **Efficiency Optimization**: Training focuses on reducing computational complexity while maintaining accuracy
- **Integration with Ray Tracing**: Trained AntNetwork guides ray tracing to focus on important directions

## 6. Training Monitoring and Visualization

### 6.1 TensorBoard Integration

Comprehensive training monitoring through TensorBoard:

- **Loss Tracking**: Training and validation loss across epochs
- **Learning Rate**: Monitor learning rate schedule changes
- **Frequency Metrics**: Subcarrier-specific MSE and correlation
- **Spatial Accuracy**: Position and direction prediction accuracy
- **AntNetwork Metrics**: Directional importance prediction accuracy and top-K sampling efficiency
- **Antenna-Specific Performance**: Individual antenna directional importance learning progress

## 7. Training Strategies and Best Practices

### 7.1 Curriculum Learning

Progressive training strategy for complex RF environments:

- **Stage 0**: Simple environments with few obstacles
- **Stage 1**: Medium complexity with moderate obstacles
- **Stage 2**: Complex environments with dense obstacles

**AntNetwork Curriculum**:
- **Phase 1**: Train with coarse directional grid (e.g., 8×4 directions)
- **Phase 2**: Increase to medium resolution (e.g., 16×8 directions)
- **Phase 3**: Full resolution training (e.g., 36×18 directions)
- **Progressive K**: Gradually increase top-K sampling from small to optimal values

### 7.2 Regularization Techniques

Advanced regularization for preventing overfitting:

- **Weight Decay**: L2 regularization for parameter constraints
- **Dropout**: Random feature masking during training
- **Batch Normalization**: Training stability and convergence improvement

## 8. Performance Optimization

### 8.1 Memory Management

Efficient memory usage during training:

- **Gradient Checkpointing**: Memory efficiency for large models
- **Mixed Precision**: Reduced memory usage with FP16 training
- **Efficient Data Loading**: Optimized DataLoader with prefetching

## 9. Training Results and Evaluation

### 9.1 Training Metrics

Key metrics tracked during training:

- **Loss Functions**: Training and validation loss across epochs
- **Frequency Accuracy**: Subcarrier-specific MSE and correlation
- **Spatial Accuracy**: Position and direction prediction accuracy
- **Channel Modeling**: CSI virtual link accuracy and interference cancellation
- **AntNetwork Performance**: Directional importance prediction accuracy and top-K sampling efficiency
- **Directional Sampling**: Reduction in computational complexity through intelligent direction selection

## 10. Future Enhancements

### 10.1 Advanced Training Techniques

- **Meta-learning**: Adapt to new environments with few-shot learning
- **Adversarial Training**: Improve robustness against environmental variations
- **Reinforcement Learning**: Optimize ray tracing strategies dynamically
- **Federated Learning**: Train across distributed RF environments

### 10.2 Scalability Improvements

- **Distributed Training**: Multi-node training for large-scale models
- **Model Parallelism**: Split large models across multiple GPUs
- **Data Parallelism**: Process multiple batches simultaneously
- **Pipeline Parallelism**: Overlap computation and communication

---

*This document describes the comprehensive training design for the Prism project. For implementation details and technical specifications, refer to the source code and configuration files.*
