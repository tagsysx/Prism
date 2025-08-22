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
- **Purpose**: Converts 128D spatial features into $N_\text{BS} \times N_\text{UE}$ attenuation factors
- **Architecture**: $N_\text{UE}$ independent 3-layer MLPs
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
5. **Subcarrier Sampling**: Randomly select $K'<K$ subcarriers per antenna for computational efficiency
6. **CSI Prediction**: Calculate predicted CSI for selected subcarriers on each BS antenna
7. **Loss Computation**: Compute MSE between predicted CSI and ground truth CSI from real measurements
8. **Backward Pass**: Compute gradients and update model parameters
9. **Validation**: Periodic validation to monitor training progress

## 3. Loss Functions and Optimization

### 3.1 Frequency-Aware Loss Function

The PrismLoss class implements specialized loss functions for multi-subcarrier RF signals:

- **BS-Centric Ray Tracing**: Ray tracing starts from each BS antenna as the center
- **AntennaNetwork-Guided Direction Selection**: Rays are traced along directions suggested by the AntennaNetwork
- **Subcarrier Sampling**: For each antenna, K' subcarriers are randomly selected to reduce computational complexity
- **CSI Computation**: CSI is calculated for selected subcarriers on each BS antenna
- **MSE Loss**: Mean squared error between predicted CSI and ground truth CSI from real measurements

### 3.2 Optimization Strategy

- **Adam Optimizer**: Adaptive learning rate optimization with configurable parameters
- **Learning Rate Scheduling**: Step-based or cosine annealing scheduling
- **Gradient Clipping**: Prevents exploding gradients for training stability
- **Weight Decay**: L2 regularization to prevent overfitting

## 4. Training Configuration

### 4.1 Configuration File Structure

Training parameters are configured through YAML configuration files with sections for:

- **Basic Parameters**: Number of epochs, batch size, learning rate
- **Optimization Settings**: Optimizer type, scheduler, gradient clipping
- **Loss Configuration**: Loss type and subcarrier weights
- **Training Strategies**: Early stopping, checkpoint saving, validation intervals

### 4.2 Hyperparameter Tuning

- **Learning Rate**: 1e-4 (empirically determined for stable convergence)
- **Batch Size**: 32-64 (balanced between memory constraints and training efficiency)
- **Network Architecture**: 256 hidden dimensions, 8 layers (optimal for spatial encoding)

## 5. Training Strategies

### 5.1 Curriculum Learning

Progressive training strategy for complex RF environments:

- **Stage 0**: Simple environments with few obstacles (20% of training)
- **Stage 1**: Medium complexity with moderate obstacles (30% of training)
- **Stage 2**: Complex environments with dense obstacles (50% of training)

### 5.2 Regularization Techniques

- **Weight Decay**: L2 regularization (default: 1e-4)
- **Dropout**: Random feature masking (0.1 for early layers, 0.2 for later layers)
- **Batch Normalization**: Training stability and convergence improvement

## 6. Performance Optimization

### 6.1 Memory Management

- **Gradient Checkpointing**: Memory efficiency for large models (30-50% memory savings)
- **Mixed Precision**: FP16 training for memory efficiency (40-60% reduction)
- **Efficient Data Loading**: Optimized DataLoader with prefetching

### 6.2 Computational Optimization

- **Model Parallelism**: Split large models across multiple GPUs
- **Data Parallelism**: Process multiple batches simultaneously
- **Pipeline Parallelism**: Overlap computation and communication

## 7. Training Results and Evaluation

### 7.1 Training Metrics

Key metrics tracked during training:

- **Loss Functions**: Training and validation loss across epochs
- **CSI Accuracy**: MSE between predicted CSI and ground truth CSI for selected subcarriers on each BS antenna
- **Spatial Accuracy**: Position and direction prediction accuracy
- **AntennaNetwork Performance**: Directional importance prediction accuracy and top-K directional sampling efficiency
- **Subcarrier Sampling Efficiency**: Impact of K' subcarrier selection on computational complexity and accuracy

### 7.2 Evaluation Metrics

- **MSE Loss**: Mean squared error between predicted and ground truth CSI
- **Correlation Coefficient**: Linear correlation between predictions and ground truth
- **Directional Accuracy**: AntennaNetwork directional importance prediction accuracy
- **Computational Efficiency**: Training and inference performance metrics

## 8. Future Enhancements

### 8.1 Advanced Training Techniques

- **Meta-learning**: Adapt to new environments with few-shot learning
- **Adversarial Training**: Improve robustness against environmental variations
- **Reinforcement Learning**: Optimize ray tracing strategies dynamically
- **Federated Learning**: Train across distributed RF environments

### 8.2 Scalability Improvements

- **Distributed Training**: Multi-node training for large-scale models
- **Model Parallelism**: Split large models across multiple GPUs
- **Data Parallelism**: Process multiple batches simultaneously

---

*This document describes the comprehensive training design for the Prism project. For implementation details and technical specifications, refer to the source code and configuration files.*
