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
- **Training**: Learns to extract meaningful spatial features that capture environmental characteristics

#### 1.1.2 AttenuationDecoder
- **Purpose**: Converts 128D spatial features into $N_\text{BS} \times N_\text{UE}$ attenuation factors
- **Architecture**: $N_\text{UE}$ independent 3-layer MLPs
- **Input**: 128D features from AttenuationNetwork
- **Output**: Attenuation factors for each BS-UE antenna combination
- **Training**: Learns to decode spatial features into antenna-dependent attenuation patterns

#### 1.1.3 RadianceNetwork
- **Purpose**: Processes UE position, viewing direction, and spatial features
- **Architecture**: 8-layer MLP with ReLU activation
- **Input**: UE position, viewing direction, and spatial features
- **Output**: Radiance values for signal propagation modeling
- **Training**: Learns to model directional signal characteristics

#### 1.1.4 AntennaNetwork
- **Purpose**: Process antenna embeddings to generate directional importance indicators for efficient ray tracing
- **Architecture**: Shallow network (64D → 128D → directional importance values)
- **Input**: 64-dimensional antenna embedding from antenna codebook
- **Output**: Directional importance matrix indicating importance of each direction
- **Training**: Learns antenna-specific directional preferences and importance patterns
- **Key Features**: 
  - Enables efficient directional sampling for computational efficiency
  - Reduces computational complexity by focusing on important directions
  - Lightweight processing with minimal computational overhead
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
- **Subcarrier Filtering**: Unselected subcarriers are ignored in the loss computation to focus computational resources on sampled frequencies

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
- **Loss Configuration**: Loss type and subcarrier weights
- **Training Strategies**: Early stopping, checkpoint saving, validation intervals

### 4.2 Hyperparameter Tuning

**Learning Rate**: 1e-4 (empirically determined for stable convergence)

**Batch Size**: 32-64 (balanced between memory constraints and training efficiency)

**Network Architecture**: 256 hidden dimensions, 8 layers (optimal for spatial encoding)

## 5. Advanced Training Features

### 5.1 CSI Virtual Link Processing

The training system incorporates CSI virtual link processing for enhanced channel modeling:

- **Virtual Link Creation**: Generate $N_\text{UE} \times K$ uplink combinations
- **Random Sampling**: Efficiently sample $N_\text{UE} \times K'$ where $K' < K$ subcarriers are randomly selected
- **Batch Diversity**: Different random seeds ensure diverse sampling across batches
- **Link Selection Strategy**: 
  - Random selection with replacement for training diversity
  - Stratified sampling to ensure coverage across frequency bands
  - Adaptive sampling based on training progress and loss convergence

### 5.2 Ray Tracing Integration

Ray tracing is integrated into the training process for realistic signal propagation modeling:

- **Multi-Angle Sampling**: Tracing the rays suggested by the AntennaNetwork
- **Spatial Point Sampling**: M (configurable, 128 by default) sampling points per ray for high-resolution modeling
- **Material Properties**: Environment-specific attenuation and reflection modeling
- **Ray Tracing Parameters**:
  - **Max Ray Depth**: Configurable reflection/refraction depth (default: 3)
  - **Ray Bouncing**: Support for specular and diffuse reflections
  - **Atmospheric Effects**: Path loss, shadowing, and multipath modeling
  - **Environment Mapping**: 3D scene representation with material properties

### 5.3 AntennaNetwork Training

The AntennaNetwork is trained to learn antenna-specific directional importance patterns:

- **Directional Importance Learning**: Train network to predict which directions are most important for each antenna
- **Directional Sampling Training**: Learn to identify critical directions for efficient ray tracing
- **Antenna-Specific Patterns**: Each antenna learns its own directional preferences during training
- **Efficiency Optimization**: Training focuses on reducing computational complexity while maintaining accuracy
- **Integration with Ray Tracing**: Trained AntennaNetwork guides ray tracing to focus on important directions
- **Training Objectives**:
  - **Directional Importance Loss**: MSE between predicted and actual directional importance
  - **Sampling Efficiency Loss**: Penalty for missing important directions
  - **Computational Cost Loss**: Encourage sparse directional sampling

## 6. Training Monitoring and Visualization

### 6.1 TensorBoard Integration

Comprehensive training monitoring through TensorBoard:

- **Loss Tracking**: Training and validation loss across epochs
- **Learning Rate**: Monitor learning rate schedule changes
- **CSI Metrics**: MSE between predicted and ground truth CSI for selected subcarriers
- **Spatial Accuracy**: Position and direction prediction accuracy
- **AntennaNetwork Metrics**: Directional importance prediction accuracy and top-K directional sampling efficiency
- **Subcarrier Sampling**: Performance impact of K' subcarrier selection strategy
- **BS-Centric Metrics**: Per-base-station antenna CSI prediction accuracy
- **Computational Efficiency**: Trade-off between subcarrier sampling and prediction accuracy

### 6.2 Advanced Monitoring Features

- **Real-time Loss Visualization**: Live plotting of training curves
- **Gradient Flow Analysis**: Monitor gradient magnitudes and vanishing/exploding gradients
- **Parameter Distribution**: Track weight and bias distributions across layers
- **Activation Patterns**: Visualize intermediate layer activations
- **Memory Usage**: Monitor GPU memory consumption and optimization opportunities

## 7. Training Strategies and Best Practices

### 7.1 Curriculum Learning

Progressive training strategy for complex RF environments:

- **Stage 0**: Simple environments with few obstacles
  - **Duration**: 20% of total training epochs
  - **Complexity**: Single BS, single UE, no obstacles
  - **Objectives**: Basic spatial encoding and CSI prediction
- **Stage 1**: Medium complexity with moderate obstacles
  - **Duration**: 30% of total training epochs
  - **Complexity**: Multiple BSs, multiple UEs, simple obstacles
  - **Objectives**: Multi-antenna coordination and basic ray tracing
- **Stage 2**: Complex environments with dense obstacles
  - **Duration**: 50% of total training epochs
  - **Complexity**: Full MIMO setup, complex environment, advanced ray tracing
  - **Objectives**: Full system optimization and fine-tuning

**AntennaNetwork Curriculum**:
- **Phase 1**: Train with coarse directional sampling for basic directional importance learning
  - **Resolution**: 8×4 directional grid
  - **Top-K**: K=16 directions
  - **Duration**: 25% of AntennaNetwork training
- **Phase 2**: Increase directional resolution for more detailed importance patterns
  - **Resolution**: 16×8 directional grid
  - **Top-K**: K=32 directions
  - **Duration**: 35% of AntennaNetwork training
- **Phase 3**: Full resolution training for optimal directional importance prediction
  - **Resolution**: 36×18 directional grid
  - **Top-K**: K=64 directions
  - **Duration**: 40% of AntennaNetwork training

### 7.2 Regularization Techniques

Advanced regularization for preventing overfitting:

- **Weight Decay**: L2 regularization for parameter constraints
  - **Default Value**: 1e-4
  - **Adaptive Scaling**: Increase with training progress
- **Dropout**: Random feature masking during training
  - **Rate**: 0.1 for early layers, 0.2 for later layers
  - **Spatial Dropout**: Apply to spatial feature maps
- **Batch Normalization**: Training stability and convergence improvement
  - **Momentum**: 0.9 for moving average updates
  - **Epsilon**: 1e-5 for numerical stability

### 7.3 Advanced Training Techniques

- **Gradient Accumulation**: Effective batch size increase for memory-constrained scenarios
  - **Accumulation Steps**: 4-8 steps per update
  - **Effective Batch Size**: 128-256 samples
- **Mixed Precision Training**: FP16 training for memory efficiency
  - **Dynamic Loss Scaling**: Automatic scaling factor adjustment
  - **Gradient Clipping**: Prevent gradient overflow in FP16
- **Learning Rate Warmup**: Gradual learning rate increase
  - **Warmup Epochs**: 5-10% of total training
  - **Warmup Strategy**: Linear or cosine warmup

## 8. Performance Optimization

### 8.1 Memory Management

Efficient memory usage during training:

- **Gradient Checkpointing**: Memory efficiency for large models
  - **Checkpoint Frequency**: Every 2-4 layers
  - **Memory Reduction**: 30-50% memory savings
- **Mixed Precision**: Reduced memory usage with FP16 training
  - **Memory Savings**: 40-60% reduction
  - **Speed Improvement**: 20-30% faster training
- **Efficient Data Loading**: Optimized DataLoader with prefetching
  - **Num Workers**: 4-8 worker processes
  - **Prefetch Factor**: 2-4 batches ahead
  - **Pin Memory**: GPU memory pinning for faster transfer

### 8.2 Computational Optimization

- **Model Parallelism**: Split large models across multiple GPUs
  - **Layer Distribution**: Distribute layers across devices
  - **Communication Overhead**: Minimize inter-device communication
- **Data Parallelism**: Process multiple batches simultaneously
  - **Synchronization**: AllReduce for gradient synchronization
  - **Batch Distribution**: Even distribution across devices
- **Pipeline Parallelism**: Overlap computation and communication
  - **Micro-batching**: Small batches for pipeline stages
  - **Bubble Minimization**: Reduce pipeline bubbles

## 9. Training Results and Evaluation

### 9.1 Training Metrics

Key metrics tracked during training:

- **Loss Functions**: Training and validation loss across epochs
- **CSI Accuracy**: （计算BS梅根天线在随机选取的载波上的CSI，然后和对应的ground truth对比计算MSE作为主要loss）
- **Spatial Accuracy**: Position and direction prediction accuracy
- **Channel Modeling**: CSI prediction accuracy on sampled subcarriers
- **AntennaNetwork Performance**: Directional importance prediction accuracy and top-K directional sampling efficiency
- **Subcarrier Sampling Efficiency**: Impact of K' subcarrier selection on computational complexity and accuracy

### 9.2 Evaluation Metrics

- **MSE Loss**: Mean squared error between predicted and ground truth CSI
  - **Per-Antenna MSE**: Individual antenna performance
  - **Per-Subcarrier MSE**: Frequency-specific accuracy
  - **Overall MSE**: System-wide performance
- **Correlation Coefficient**: Linear correlation between predictions and ground truth
  - **Pearson Correlation**: Linear relationship strength
  - **Spearman Correlation**: Rank correlation for non-linear relationships
- **Directional Accuracy**: AntennaNetwork directional importance prediction accuracy
  - **Top-K Hit Rate**: Percentage of important directions correctly identified
  - **Directional MSE**: Error in directional importance values
- **Computational Efficiency**: Training and inference performance metrics
  - **Training Time**: Time per epoch and total training time
  - **Memory Usage**: Peak and average memory consumption
  - **Inference Speed**: Time per prediction and throughput

### 9.3 Model Convergence Analysis

- **Loss Convergence**: Training and validation loss trends
  - **Convergence Rate**: Epochs to reach target loss
  - **Overfitting Detection**: Validation loss divergence
  - **Stability Metrics**: Loss variance and oscillations
- **Parameter Convergence**: Weight and bias convergence patterns
  - **Parameter Norms**: L1/L2 norms across layers
  - **Gradient Norms**: Gradient magnitude trends
  - **Learning Rate Impact**: Effect of LR scheduling on convergence

## 10. Future Enhancements

### 10.1 Advanced Training Techniques

- **Meta-learning**: Adapt to new environments with few-shot learning
  - **Model-Agnostic Meta-Learning (MAML)**: Fast adaptation to new scenarios
  - **Reptile**: Simplified meta-learning for RF environments
  - **Few-shot CSI Prediction**: Learn from limited training data
- **Adversarial Training**: Improve robustness against environmental variations
  - **Adversarial Examples**: Generate challenging training samples
  - **Robustness Metrics**: Performance under adversarial conditions
  - **Defense Mechanisms**: Adversarial training strategies
- **Reinforcement Learning**: Optimize ray tracing strategies dynamically
  - **Policy Learning**: Learn optimal ray tracing policies
  - **Reward Functions**: Design rewards for efficient sampling
  - **Multi-agent RL**: Coordinate multiple BSs and UEs
- **Federated Learning**: Train across distributed RF environments
  - **Privacy Preservation**: Local training with model aggregation
  - **Communication Efficiency**: Reduce inter-node communication
  - **Heterogeneous Data**: Handle diverse environment characteristics

### 10.2 Scalability Improvements

- **Distributed Training**: Multi-node training for large-scale models
  - **Node Communication**: Efficient inter-node communication protocols
  - **Load Balancing**: Dynamic workload distribution
  - **Fault Tolerance**: Handle node failures gracefully
- **Model Parallelism**: Split large models across multiple GPUs
  - **Layer Distribution**: Optimal layer placement across devices
  - **Memory Optimization**: Minimize memory fragmentation
  - **Communication Overlap**: Hide communication behind computation
- **Data Parallelism**: Process multiple batches simultaneously
  - **Gradient Synchronization**: Efficient gradient aggregation
  - **Batch Size Scaling**: Scale batch size with number of devices
  - **Data Sharding**: Distribute data across multiple nodes
- **Pipeline Parallelism**: Overlap computation and communication
  - **Micro-batch Optimization**: Optimal micro-batch sizes
  - **Bubble Reduction**: Minimize pipeline idle time
  - **Memory Management**: Efficient memory usage across pipeline stages

### 10.3 Advanced Optimization Techniques

- **Neural Architecture Search (NAS)**: Automatically discover optimal network architectures
  - **Search Space**: Define architecture search space
  - **Search Strategy**: Efficient search algorithms
  - **Performance Prediction**: Predict architecture performance
- **Knowledge Distillation**: Transfer knowledge from larger models
  - **Teacher-Student Training**: Large model guides small model
  - **Knowledge Transfer**: Transfer learned representations
  - **Model Compression**: Reduce model size while maintaining performance
- **Quantization**: Reduce model precision for efficiency
  - **Post-training Quantization**: Quantize trained models
  - **Quantization-aware Training**: Train with quantization constraints
  - **Mixed Precision**: Dynamic precision selection

---

*This document describes the comprehensive training design for the Prism project. For implementation details and technical specifications, refer to the source code and configuration files.*
