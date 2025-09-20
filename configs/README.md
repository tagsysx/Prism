# Prism Configuration Guide

## Overview
This document explains the configuration parameters for the Prism neural network-based electromagnetic ray tracing system. The configuration has been completely reorganized and updated with the latest features including automatic GPU selection, improved data handling, enhanced CUDA acceleration, and comprehensive testing capabilities.

### Latest Updates (January 2025)
- ✅ **Automatic GPU Selection**: No manual GPU configuration needed
- ✅ **Unified Data Handling**: Single dataset with automatic train/test splitting
- ✅ **Enhanced CUDA Support**: Improved CUDA ray tracing with fallback mechanisms
- ✅ **Comprehensive Testing**: Complete testing pipeline with visualization
- ✅ **Performance Optimizations**: Reduced computational complexity for faster training
- ✅ **Template Variables**: Dynamic path resolution in configuration files
- ✅ **Spatial Spectrum Loss**: Advanced loss function for beamforming and DOA applications
- ✅ **Multi-objective Training**: Configurable loss weights for CSI, PDP, and spatial spectrum components
- ✅ **Visualization Support**: Automatic generation of spatial spectrum comparison plots

## Table of Contents
1. [Key Concepts Clarification](#key-concepts-clarification)
2. [Neural Network Configuration](#neural-network-configuration)
3. [Base Station Configuration](#base-station-configuration)
4. [User Equipment Configuration](#user-equipment-configuration)
5. [Ray Tracing Configuration](#ray-tracing-configuration)
6. [System Configuration](#system-configuration)
7. [Training Configuration](#training-configuration)
8. [Testing Configuration](#testing-configuration)
9. [Input Configuration](#input-configuration)
10. [Output Configuration](#output-configuration)
11. [Spatial Spectrum Loss Configuration](#spatial-spectrum-loss-configuration)
12. [Performance Optimization Guide](#performance-optimization-guide)
13. [CUDA Acceleration Guide](#cuda-acceleration-guide)
14. [Recent Updates and Improvements](#recent-updates-and-improvements-january-2025)
15. [Configuration Examples](#configuration-examples)
16. [Troubleshooting Guide](#troubleshooting-guide)

## Key Concepts Clarification

### UE (User Equipment) Terminology
To avoid confusion, we use specific terminology:

- **UE Device**: A single user equipment device (e.g., a smartphone)
- **UE Antennas**: Multiple antennas on a single UE device
- **UE Positions**: Different spatial locations where the same UE device is placed

### Data Structure
Our simulation data represents:
- **1 UE Device** with **4 UE Antennas** placed at **100 different positions**
- Each position has complete MIMO channel responses: `(100, 408, 4, 64)`
  - 100 positions
  - 408 subcarriers  
  - 4 UE antennas
  - 64 BS antennas

## Configuration Parameters

### Neural Networks Section
```yaml
neural_networks:
  enabled: true                       # Enable neural network-based processing
  
  attenuation_decoder:
    num_ue_antennas: 4                # Number of antennas per UE device (NOT number of UE devices)
    output_dim: 408                   # Number of subcarriers
  
  radiance_network:
    num_ue_antennas: 4                # Number of antennas per UE device (NOT number of UE devices)
    output_dim: 408                   # Number of subcarriers
```

### User Equipment Section
```yaml
user_equipment:
  num_ue_antennas: 4                  # Number of antennas per UE device
  antenna_config: '4x64'              # 4 UE antennas × 64 BS antennas
  position_range: [-150.0, 150.0]     # UE position range in meters
  height_range: [1.0, 2.0]            # UE height range in meters
  # Note: Number of UE positions is determined from actual training data
```

## Important Notes

1. **`num_ue_antennas`** refers to the number of antennas on a single UE device
2. **Number of UE positions** is determined from the actual training data, not from config
3. **Each training sample represents one UE position**, not multiple UE devices
4. **The model learns position-dependent channel characteristics** for a single UE device

## Data Flow
```
Training Data: (batch_size, 408, 4, 64)
├── batch_size: Number of UE positions in this batch
├── 408: Number of subcarriers
├── 4: Number of UE antennas per device
└── 64: Number of BS antennas
```

## Common Misconceptions

❌ **Wrong**: "We have 100 UE devices"
✅ **Correct**: "We have 1 UE device at 100 different positions"

❌ **Wrong**: "num_ue: 100 means 100 devices"
✅ **Correct**: "100 positions determined from training data"

❌ **Wrong**: "num_ue: 4 means 4 devices"
✅ **Correct**: "num_ue_antennas: 4 means 4 antennas per device"

---

## Ray Tracing Configuration

### Physical Parameters
```yaml
ray_tracing:
  max_ray_length: 200.0              # Maximum ray length in meters
  signal_threshold: 1e-6             # Signal strength threshold for early termination
  enable_early_termination: true     # Enable early termination optimization
```

**Description**: Defines the physical boundaries and signal processing parameters for ray tracing.

### Scene Configuration
```yaml
ray_tracing:
  scene_bounds:
    min: [-100.0, -100.0, 0.0]       # Scene minimum bounds [x, y, z]
    max: [100.0, 100.0, 30.0]        # Scene maximum bounds [x, y, z]
```

**Description**: Defines the 3D simulation environment boundaries.

### Angular Sampling
```yaml
ray_tracing:
  angular_sampling:
    azimuth_divisions: 18            # Number of azimuth divisions (0° to 360°)
    elevation_divisions: 9           # Number of elevation divisions (0° to 90°)
    total_directions: 162            # 18 × 9 = 162 angle combinations
    top_k_directions: 32             # Number of top-K directions to select
```

**Description**: Controls the angular resolution of ray tracing. Higher values provide more accurate direction sampling but increase computation time.

### Spatial Sampling
```yaml
ray_tracing:
  radial_sampling:
    num_sampling_points: 64          # Number of sampling points per ray (uniform_samples)
    resampled_points: 32             # Second stage: importance-based resampled points
```

**Description**: Two-stage importance sampling for efficient spatial point selection along rays.

### Subcarrier Sampling
```yaml
ray_tracing:
  subcarrier_sampling:
    sampling_ratio: 0.1              # Ratio of subcarriers to select (10%)
    antenna_specific_selection: true  # Enable antenna-specific subcarrier selection
```

**Description**: Controls subcarrier selection for efficient OFDM processing.

---

## Neural Network Configuration

### Attenuation Network
```yaml
neural_networks:
  attenuation_network:
    input_dim: 3                  # 3D spatial position
    hidden_dim: 256               # Hidden layer dimension
    num_hidden_layers: 8          # Number of hidden layers
    feature_dim: 128              # Output feature dimension
    activation: 'relu'            # Activation function
    use_shortcut: true            # Enable shortcut connections
```

**Description**: Extracts spatial features from 3D positions for attenuation prediction.

### Attenuation Decoder
```yaml
neural_networks:
  attenuation_decoder:
    input_dim: 128                # Feature dimension from AttenuationNetwork
    hidden_dim: 256               # Hidden layer dimension
    num_hidden_layers: 3          # Number of hidden layers
    output_dim: 408               # Number of subcarriers K
    num_ue_antennas: 4            # Number of UE antennas per device
    activation: 'relu'            # Activation function
```

**Description**: Decodes spatial features to predict attenuation for each subcarrier and UE antenna.

### Antenna Codebook
```yaml
neural_networks:
  antenna_codebook:
    num_antennas: 64              # Number of BS antennas N_BS
    embedding_dim: 64             # Antenna embedding dimension
    learnable: true               # Learnable embeddings
```

**Description**: Learnable embeddings for each base station antenna.

### Antenna Network (MLP-based Direction Sampling)
```yaml
neural_networks:
  antenna_network:
    input_dim: 64                 # Antenna embedding dimension
    hidden_dim: 128               # Hidden layer dimension
    num_hidden_layers: 2          # Number of hidden layers
    output_dim: 162               # A × B directions (18 × 9 = 162)
    activation: 'relu'            # Activation function
    dropout_rate: 0.1             # Dropout rate for regularization
```

**Description**: MLP that selects important ray directions based on antenna characteristics.

### Radiance Network
```yaml
neural_networks:
  radiance_network:
    ue_pos_dim: 3                 # UE position dimension
    view_dir_dim: 3               # Viewing direction dimension
    spatial_feature_dim: 128      # Spatial feature dimension
    antenna_embedding_dim: 64     # Antenna embedding dimension
    hidden_dim: 256               # Hidden layer dimension
    num_hidden_layers: 4          # Number of hidden layers
    output_dim: 408               # Number of subcarriers K
    activation: 'relu'            # Activation function
```

**Description**: Predicts radiance (signal strength) for each subcarrier and UE antenna. Note that `num_ue_antennas` is now specified in the `attenuation_decoder` section for consistency.

---

## Base Station Configuration

### Basic Configuration
```yaml
base_station:
  default_position: [0.0, 0.0, 0.0]  # Base station at origin
  num_antennas: 64                   # Number of BS antennas (matches actual data)
  antenna_embedding_dim: 64          # Antenna embedding dimension
  antenna_type: 'mimo'               # MIMO antenna configuration
  polarization: 'dual'               # Dual polarization support
```

**Description**: Configures base station properties including position, number of antennas, and antenna characteristics.

### OFDM System Parameters
```yaml
base_station:
  ofdm:
    center_frequency: 3.5e9          # Center frequency in Hz (3.5 GHz - mid-band 5G)
    bandwidth: 100.0e6               # Bandwidth in Hz (100 MHz - 5G NR standard)
    num_subcarriers: 408             # Total number of subcarriers
    subcarrier_spacing: 245.1e3      # Subcarrier spacing in Hz (100MHz/408 ≈ 245.1 kHz)
    guard_band_ratio: 0.1            # Guard band ratio (10% of bandwidth)
    fft_size: 512                    # OFDM FFT size
    num_guard_carriers: 52           # Number of guard carriers ((512-408)/2)
```

**Description**: 5G NR OFDM system parameters for realistic wireless communication simulation.

### Antenna Array Configuration
```yaml
base_station:
  antenna_array:
    configuration: '8x8'              # Antenna array configuration (M x N)
    element_spacing: 'half_wavelength' # Element spacing type
    custom_spacing: null              # Custom spacing in meters (if not half_wavelength)
    beamforming_enabled: true         # Enable beamforming capabilities
```

**Description**: Configures the physical antenna array layout and beamforming capabilities.

---

## User Equipment Configuration

```yaml
user_equipment:
  num_ue_antennas: 4                 # Number of UE antennas per device
  antenna_config: '4x64'             # 4 UE antennas per device, 64 BS antennas
  position_range: [-150.0, 150.0]    # UE position range in meters
  height_range: [1.0, 2.0]           # UE height range in meters
```

**Description**: Configures user equipment properties and position distribution.

---

## System Configuration

### Computational Settings
```yaml
system:
  device: 'cuda'                     # Primary device for computation ('cuda' or 'cpu')
  batch_size: 2                      # Global batch size for all operations
  gpu_memory_fraction: 0.8           # GPU memory usage fraction
```

**Description**: Core system settings for device selection and memory management.

### Ray Tracing Execution Mode
```yaml
system:
  ray_tracing_mode: 'cuda'           # Ray tracing mode: 'cuda', 'cpu', or 'hybrid'
  # - 'cuda': Pure CUDA acceleration (fastest, neural networks + ray tracing on GPU)
  # - 'cpu': Pure CPU with multiprocessing (stable, reliable)
  # - 'hybrid': Neural networks on CUDA, ray tracing on CPU (balanced)
  fallback_to_cpu: true              # Fallback to CPU if CUDA not available
```

**Description**: Controls the execution mode for ray tracing operations with automatic fallback capabilities.

### CUDA-Specific Settings (Automatic GPU Selection)
```yaml
system:
  cuda:
    # GPU selection is now automatic - system will scan and select the best available GPU
    # No need to manually specify device_id or gpu_ids
    optimization_level: 'O2'         # CUDA optimization level ('O0', 'O1', 'O2', 'O3')
    benchmark_mode: true             # Enable CUDA benchmark mode for optimal performance
    deterministic: false             # Disable deterministic mode for better performance
    multi_gpu: false                 # Enable/disable multi-GPU CUDA operations
    memory_pool: true                # Enable CUDA memory pool for better memory management
    auto_select_gpu: true            # Enable automatic GPU selection (default: true)
```

**Description**: 
- **Automatic GPU Selection**: System automatically detects and selects the best available GPU
- **No Manual Configuration**: No need to specify device IDs or GPU lists
- **Intelligent Selection**: Chooses GPU based on memory availability and compute capability
- **Fallback Support**: Automatically falls back to CPU if no suitable GPU is found

### CPU-Specific Settings
```yaml
system:
  cpu:
    num_workers: 4                   # Number of worker processes for CPU ray tracer
```

**Description**: CPU-specific configuration for multiprocessing ray tracing.

### Mixed Precision Configuration
```yaml
system:
  mixed_precision:
    enabled: true                    # Enable/disable mixed precision globally
    autocast_enabled: true           # Enable autocast for forward pass
    grad_scaler_enabled: true        # Enable gradient scaler for training
    loss_scale: "dynamic"            # "dynamic" or fixed number (e.g., 2048)
```

**Description**: Mixed precision training configuration for improved performance and memory efficiency.

### Spatial Spectrum Configuration
```yaml
system:
  spatial_spectrum:
    enabled: true                    # Enable spatial spectrum calculation
    default_algorithm: 'bartlett'    # Default spatial spectrum algorithm
    default_fusion_method: 'average' # Default subcarrier fusion method
    angle_resolution:
      theta_points: 60               # Number of elevation angle points
      phi_points: 120                # Number of azimuth angle points
      theta_range: [-60, 60]         # Elevation angle range in degrees
      phi_range: [0, 360]            # Azimuth angle range in degrees
    peak_detection:
      num_peaks: 3                   # Number of peaks to detect
      min_distance: 5                # Minimum distance between peaks (pixels)
    subcarrier_selection:
      enabled: true                  # Enable subcarrier selection for spatial spectrum
      selection_method: 'uniform'    # 'uniform', 'weighted', or 'custom'
      num_selected: 64               # Number of subcarriers to use (64 out of 408)
      frequency_weighting: true      # Apply frequency-dependent weighting
```

**Description**: 5G NR specific spatial spectrum analysis configuration for beamforming and direction finding.



---

## Training Configuration

### Basic Training Parameters
```yaml
training:
  learning_rate: 1e-4                # Initial learning rate
  num_epochs: 1                      # Number of training epochs
  batches_per_epoch: 50              # Number of batches per epoch
```

**Description**: Core training parameters for model optimization.

### Checkpoint and Recovery
```yaml
training:
  auto_checkpoint: true              # Enable automatic checkpointing
  checkpoint_frequency: 10           # Save checkpoint every N batches (every 10 batches)
```

**Description**: Automatic checkpointing for training recovery and model saving.

### Loss Function Configuration
```yaml
training:
  loss:
    # Overall loss weights
    csi_weight: 0.7                  # Weight for CSI prediction loss
    pdp_weight: 300.0                # Weight for PDP loss (increased to balance scale difference)
    spatial_spectrum_weight: 0.0     # Weight for spatial spectrum loss (disabled by default)
    regularization_weight: 0.01      # Weight for regularization terms
    
    # CSI Loss Configuration
    csi_loss:
      phase_weight: 1.0              # Weight for phase component
      magnitude_weight: 1.0           # Weight for magnitude component
      normalize_weights: true         # Whether to normalize weights to sum to 1.0
    
    # PDP Loss Configuration  
    pdp_loss:
      type: 'hybrid'                 # Loss type: 'mse', 'delay', 'hybrid'
      fft_size: 512                  # FFT size for PDP computation
      normalize_pdp: true            # Whether to normalize PDPs before comparison
    
    # Spatial Spectrum Loss Configuration
    spatial_spectrum_loss:
      enabled: false                 # Enable/disable spatial spectrum loss (disabled by default)
      weight: 0.1                    # Weight for spatial spectrum loss component
      algorithm: 'bartlett'          # Spatial spectrum estimation algorithm ('bartlett', 'capon', 'music')
      fusion_method: 'average'       # Multi-subcarrier fusion method ('average', 'max')
      theta_range: [-60.0, 2.0, 60.0]   # Elevation angle range [min, step, max] in degrees
      phi_range: [0.0, 2.0, 360.0]      # Azimuth angle range [min, step, max] in degrees
```

**Description**: Comprehensive loss function configuration:

- **CSI Loss**: Subcarrier-precise loss addressing complex MSE vs per-subcarrier accuracy paradox for accurate channel prediction
- **PDP Loss**: Power Delay Profile loss for time-domain validation using FFT-based analysis
- **Spatial Spectrum Loss**: Optional spatial spectrum loss for direction-of-arrival and beamforming applications
- **Multi-objective Training**: Configurable weights allow balancing different loss components based on application requirements

**Spatial Spectrum Loss Features**:
- **Bartlett Beamforming**: Currently supports Bartlett spatial spectrum estimation
- **Configurable Angular Resolution**: Adjustable elevation and azimuth angle sampling
- **Multi-subcarrier Fusion**: Averages spatial spectrums across all subcarriers
- **Visualization Support**: Generates comparison plots during testing for analysis

### Optimizer Configuration
```yaml
training:
  optimizer: 'adam'                  # Optimizer type
  optimizer_params:
    beta1: 0.9                       # Adam beta1 parameter
    beta2: 0.999                     # Adam beta2 parameter
    weight_decay: 1e-5               # Weight decay for regularization
```

**Description**: Optimizer selection and hyperparameter configuration.

### Learning Rate Scheduling
```yaml
training:
  lr_scheduler:
    enabled: true                    # Enable learning rate scheduling
    type: 'reduce_on_plateau'        # Scheduler type
    
    # ReduceLROnPlateau parameters (recommended for adaptive training)
    mode: 'min'                      # Monitor validation loss minimum
    factor: 0.7                      # Reduce LR by 30% each time
    patience: 4                      # Wait 4 epochs before reducing LR
    threshold: 1e-4                  # Minimum improvement threshold
    threshold_mode: 'rel'            # Relative threshold mode
    cooldown: 1                      # Cooldown period after LR reduction
    min_lr_plateau: 5e-6             # Minimum learning rate limit
    verbose: true                    # Print LR reduction messages
    
    # StepLR parameters (alternative fixed-step scheduling)
    step_size: 30                    # Step size for StepLR
    gamma: 0.1                       # Multiplicative factor for LR decay
```

**Description**: Learning rate scheduling for improved training convergence. The system supports multiple scheduler types:

- **`reduce_on_plateau`** (Recommended): Automatically reduces learning rate when validation loss stops improving. Best for handling training instability and loss fluctuations.
- **`step`**: Reduces learning rate at fixed intervals. Suitable when you know the optimal decay schedule.
- **`plateau`**: Alias for `reduce_on_plateau` for backward compatibility.

**Supported Scheduler Types**:
- `reduce_on_plateau`: Adaptive scheduling based on validation loss
- `step`: Fixed-step learning rate decay
- `plateau`: Backward compatibility alias for `reduce_on_plateau`

**Parameter Details**:
- `mode`: 'min' for loss minimization, 'max' for metric maximization
- `factor`: Multiplication factor for learning rate reduction (0.0 < factor < 1.0)
- `patience`: Number of epochs to wait before reducing learning rate
- `threshold`: Minimum change to qualify as improvement
- `threshold_mode`: 'rel' for relative threshold, 'abs' for absolute
- `cooldown`: Number of epochs to wait before resuming normal operation
- `min_lr_plateau`: Lower bound on the learning rate
- `verbose`: Whether to print messages when learning rate is reduced

### Early Stopping
```yaml
training:
  early_stopping:
    enabled: true                    # Enable early stopping
    patience: 10                     # Number of epochs to wait for improvement
    min_delta: 1e-6                  # Minimum change to qualify as improvement
    restore_best_weights: true       # Restore best weights on early stopping
```

**Description**: Early stopping mechanism to prevent overfitting and save training time.

---

## Testing Configuration

### Basic Testing Parameters
```yaml
testing:
  batch_size: 1                      # Testing batch size (can be different from training)
```

**Description**: Core testing parameters for model evaluation.

### Model Configuration
```yaml
testing:
  model_path: "results/sionna/training/models/best_model.pt"    # Path to trained model for testing
  # Alternative model paths (uncomment to use):
  # model_path: "results/sionna/training/checkpoints/checkpoint_epoch_1_batch_39.pt"  # Specific epoch/batch
  # model_path: "results/sionna/training/models/best_model.pt"  # Best model
```

**Description**: 
- **Best Model Path**: Uses the best performing model saved during training
- **Checkpoint Support**: Can load specific epoch/batch checkpoints for analysis
- **Template Variable Support**: Paths use the same base directory structure as training

### Evaluation Metrics
```yaml
testing:
  metrics:
    - 'mse'                          # Mean Squared Error
    - 'mae'                          # Mean Absolute Error
    - 'rmse'                         # Root Mean Squared Error
    - 'complex_correlation'          # Complex correlation coefficient
```

**Description**: Metrics to compute during model evaluation.

### Output Configuration
```yaml
testing:
  save_predictions: true             # Save model predictions
  save_intermediate_results: true    # Save intermediate ray tracing results
  save_visualizations: true          # Save result visualizations
```

**Description**: Controls what outputs to save during testing.

---

## Input Configuration

### Unified Data Handling (New Approach)
```yaml
input:
  # Single dataset configuration for train/test split
  dataset_path: "data/sionna/sionna_5g_simulation.h5"    # Single HDF5 file containing all data
  
  # Train/test split configuration
  split:
    random_seed: 42                    # Random seed for reproducible splits
    train_ratio: 0.2                   # Training data ratio (0.0 to 1.0)
    test_ratio: 0.1                    # Testing data ratio (0.0 to 1.0)
    # Note: train_ratio + test_ratio can be < 1.0 to use only subset of data
```

**Description**: 
- **Unified approach**: Single dataset file with automatic train/test splitting
- **Flexible ratios**: Configure training and testing data proportions
- **Reproducible splits**: Fixed random seed ensures consistent data splits
- **Subset support**: Use only a portion of data for faster experimentation

### Legacy Data Paths (Still Supported)
```yaml
input:
  # Alternative: Separate training and testing files
  training_data: "data/sionna/sionna_5g_training.h5"     # Training data HDF5 file
  testing_data: "data/sionna/sionna_5g_testing.h5"       # Testing data HDF5 file
```

**Note**: The system automatically detects which configuration approach is used and handles data loading accordingly.

---

## Spatial Spectrum Loss Configuration

The Spatial Spectrum Loss is an advanced loss function that computes spatial spectrum from CSI data and compares predicted vs target spatial spectrums. This is particularly useful for beamforming, direction-of-arrival estimation, and spatial channel modeling applications.

### Basic Configuration

```yaml
training:
  loss:
    spatial_spectrum_weight: 0.1       # Weight for spatial spectrum loss (0.0 = disabled)
    
    spatial_spectrum_loss:
      enabled: true                    # Enable/disable spatial spectrum loss
      algorithm: 'bartlett'            # Spatial spectrum estimation algorithm
      fusion_method: 'average'         # Multi-subcarrier fusion method
      theta_range: [-60.0, 2.0, 60.0] # Elevation angle range [min, step, max] in degrees
      phi_range: [0.0, 2.0, 360.0]    # Azimuth angle range [min, step, max] in degrees
```

### Required Base Station Configuration

The Spatial Spectrum Loss requires specific base station antenna array configuration:

```yaml
base_station:
  antenna_array:
    configuration: '8x8'              # Antenna array configuration (M x N)
    element_spacing: 'half_wavelength' # Element spacing type
    custom_spacing: null              # Custom spacing in meters (if not half_wavelength)
  
  ofdm:
    center_frequency: 3.5e9           # Center frequency in Hz (required for wavelength calculation)
    bandwidth: 100.0e6                # Bandwidth in Hz (required for subcarrier frequencies)
    num_subcarriers: 408              # Total number of subcarriers
```

### Algorithm Options

**Currently Supported:**
- `'bartlett'`: Bartlett beamforming (classical beamforming)

**Future Support:**
- `'capon'`: Capon beamforming (adaptive beamforming)
- `'music'`: MUSIC algorithm (subspace-based method)

### Fusion Methods

- `'average'`: Average spatial spectrums across all subcarriers (recommended)
- `'max'`: Take maximum values across subcarriers

### Angular Resolution Configuration

The `theta_range` and `phi_range` parameters use the format `[min, step, max]` in degrees:

```yaml
theta_range: [-60.0, 2.0, 60.0]     # Elevation: -60° to +60° with 2° steps (61 points)
phi_range: [0.0, 2.0, 360.0]        # Azimuth: 0° to 360° with 2° steps (181 points)
```

**Performance Considerations:**
- Higher angular resolution (smaller step size) increases computation time
- Typical configurations:
  - **Fast**: `theta_range: [-60.0, 5.0, 60.0]`, `phi_range: [0.0, 10.0, 360.0]`
  - **Balanced**: `theta_range: [-60.0, 2.0, 60.0]`, `phi_range: [0.0, 2.0, 360.0]`
  - **High-res**: `theta_range: [-90.0, 1.0, 90.0]`, `phi_range: [0.0, 1.0, 360.0]`

### Usage Examples

#### Enable for Beamforming Applications
```yaml
training:
  loss:
    csi_weight: 0.6                   # Reduce CSI weight slightly
    pdp_weight: 200.0                 # Maintain PDP weight
    spatial_spectrum_weight: 0.2      # Add spatial spectrum component
    
    spatial_spectrum_loss:
      enabled: true
      algorithm: 'bartlett'
      fusion_method: 'average'
      theta_range: [-45.0, 2.0, 45.0]  # Focus on main lobe region
      phi_range: [0.0, 2.0, 360.0]
```

#### Disable Spatial Spectrum Loss (Default)
```yaml
training:
  loss:
    spatial_spectrum_weight: 0.0      # Set weight to 0 to disable
    
    spatial_spectrum_loss:
      enabled: false                  # Explicitly disable
```

### Testing and Visualization

When enabled, the Spatial Spectrum Loss provides visualization capabilities during testing:

```python
# In testing code
result = loss_function.compute_and_visualize_spatial_spectrum_loss(
    predicted_csi, target_csi, 
    save_path='results/sionna/testing/plots',
    sample_idx=0
)

if result:
    loss_value, plot_path = result
    print(f"Spatial spectrum loss: {loss_value:.6f}")
    print(f"Comparison plot saved to: {plot_path}")
```

**Generated Visualizations:**
- Side-by-side comparison of predicted vs target spatial spectrums
- Color-coded power levels with elevation/azimuth axes
- Loss value and algorithm information in plot title
- Automatic saving to `results/sionna/testing/plots/` directory

### Performance Impact

**Computational Complexity:**
- **Training**: Adds significant computation per batch (depends on angular resolution)
- **Memory**: Moderate additional GPU memory usage
- **Time**: Approximately 2-5x slower training depending on configuration

**Optimization Tips:**
1. **Start with coarse resolution** during initial training
2. **Use smaller subcarrier counts** for faster prototyping
3. **Enable only when spatial characteristics are important** for your application
4. **Monitor GPU memory usage** with high-resolution configurations

### Troubleshooting

**Common Issues:**

1. **"Configuration must contain 'base_station.antenna_array.configuration'"**
   - Ensure antenna array configuration is properly set in base_station section

2. **"BS antennas doesn't match array config"**
   - Verify `num_antennas` matches `M × N` from antenna array configuration

3. **High memory usage**
   - Reduce angular resolution (increase step size)
   - Use smaller batch sizes
   - Consider reducing number of subcarriers for testing

4. **Slow training**
   - Use coarser angular resolution during initial experiments
   - Enable only for final training runs
   - Consider using CPU ray tracing mode for debugging

---

## Output Configuration

### Base Directories with Template Variables
```yaml
output:
  # Base directories (all paths relative to project root)
  base_dir: "results/sionna"                           # Base results directory
```

**Description**: Base directory for all output files, relative to project root.

### Training Output Directories (Template Variables)
```yaml
output:
  # Training output directories (using template variables)
  training:
    checkpoint_dir: "{{base_dir}}/training/checkpoints"    # Directory for saving checkpoints
    tensorboard_dir: "{{base_dir}}/training/tensorboard"   # TensorBoard logs directory
    models_dir: "{{base_dir}}/training/models"             # Saved models directory
    
    # Training logging configuration
    logging:
      log_level: 'INFO'                 # Training logging level
      log_dir: "{{base_dir}}/training/logs"                # Training logs directory
      log_file: "{{base_dir}}/training/logs/training.log"  # Training log file path
```

**Description**: 
- **Template Variables**: Use `{{base_dir}}` for dynamic path resolution
- **Automatic Creation**: Directories are created automatically during execution
- **Centralized Logging**: Dedicated logging configuration for training

### Testing Output Directories (Template Variables)
```yaml
output:
  # Testing output directories (using template variables)
  testing:
    results_dir: "{{base_dir}}/testing/results"      # Test results and metrics
    plots_dir: "{{base_dir}}/testing/plots"          # Visualization plots
    predictions_dir: "{{base_dir}}/testing/predictions"  # Model predictions
    reports_dir: "{{base_dir}}/testing/reports"      # Test reports directory
    
    # Testing logging configuration
    logging:
      log_level: 'INFO'                  # Testing logging level
      log_dir: "{{base_dir}}/testing/logs"             # Testing logs directory
      log_file: "{{base_dir}}/testing/logs/testing.log"  # Testing log file path
```

**Description**: 
- **Organized Structure**: Separate directories for different types of test outputs
- **Template Variables**: Dynamic path resolution based on base directory
- **Comprehensive Logging**: Dedicated logging for testing operations

### File Formats and Compression
```yaml
output:
  format: 'hdf5'                     # Output file format
  compression_level: 6               # Compression level for output files
```

**Description**: File format and compression settings for efficient storage.

### What to Save
```yaml
output:
  save_results: true                 # Save ray tracing results
  save_training_outputs: true        # Save training interface outputs
  save_ray_tracer_results: true      # Save ray tracer intermediate results
  save_csi_predictions: true         # Save CSI predictions
```

**Description**: Controls which intermediate and final results to save.

### Checkpoint Configuration
```yaml
output:
  checkpoint_format: 'pytorch'       # Checkpoint file format
  save_optimizer_state: true         # Save optimizer state in checkpoints
  save_training_history: true        # Save training history
```

**Description**: Checkpoint saving configuration for training recovery.

### Logging Configuration
```yaml
output:
  log_level: 'INFO'                  # Logging level
  log_file: "training.log"           # Main log file (in project root)
  enable_progress_bar: true          # Enable progress bars
  log_ray_tracer_stats: true         # Log ray tracer statistics
  log_training_metrics: true         # Log training metrics
```

**Description**: Logging configuration for monitoring training progress and debugging.

---

## Performance Optimization Guide

### Overview
This section explains the performance optimizations implemented in the current configuration to achieve faster training while maintaining acceptable accuracy.

### Training Speed Optimization

#### Problem Analysis
The first epoch was taking too long (16+ minutes) due to excessive computational complexity in ray tracing.

#### Root Cause
- **Original configuration**: 36×18 = 648 directions × 64 spatial points = 41,472 samples per ray
- **Fallback method**: Always used `_accumulate_signals_fallback` which iterates through ALL directions
- **No MLP direction selection**: `prism_network` was `None`, forcing brute-force computation

#### Optimization Strategy

##### 1. Reduced Angular Resolution
```yaml
# BEFORE (Slow)
azimuth_divisions: 36      # 10° resolution
elevation_divisions: 18    # 10° resolution
total_directions: 648      # 36 × 18 = 648

# AFTER (Fast - 75% reduction)
azimuth_divisions: 18      # 20° resolution
elevation_divisions: 9     # 20° resolution  
total_directions: 162      # 18 × 9 = 162
```

##### 2. Reduced Spatial Sampling
```yaml
# BEFORE (Slow)
uniform_samples: 128       # 128 uniform points
resampled_points: 64       # 64 resampled points
total_spatial_points: 41472 # 648 × 64

# AFTER (Fast - 75% reduction)
uniform_samples: 64        # 64 uniform points
resampled_points: 32       # 32 resampled points
total_spatial_points: 10368 # 162 × 32
```

##### 3. Updated Neural Network Dimensions
```yaml
# AntennaNetwork output dimension must match total_directions
antenna_network:
  output_dim: 162          # Updated from 648 to match new direction count
```

#### Expected Performance Improvement

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Directions** | 648 | 162 | **75% faster** |
| **Spatial points** | 41,472 | 10,368 | **75% faster** |
| **Total samples** | 26,765,856 | 1,679,616 | **93.7% faster** |

#### Trade-offs

##### ✅ **Advantages**
- **Training speed**: 4-5x faster training
- **Memory usage**: Significantly reduced
- **GPU efficiency**: Better utilization

##### ⚠️ **Trade-offs**
- **Angular resolution**: Reduced from 10° to 20° (still acceptable for most applications)
- **Spatial precision**: Reduced from 64 to 32 points per ray
- **Model accuracy**: May be slightly lower but should still be sufficient

#### Validation
- **Angular coverage**: 20° resolution is still adequate for MIMO beamforming
- **Spatial sampling**: 32 points per ray provides good signal reconstruction
- **Training stability**: Reduced complexity should improve convergence

#### Future Improvements
1. **Enable MLP direction selection**: Ensure `prism_network` is properly initialized
2. **Dynamic resolution**: Start with low resolution, increase during training
3. **Curriculum learning**: Gradually increase complexity as training progresses

---

## Top-K Directions Configuration

### Overview
The `top_k_directions` parameter can now be read from the configuration file, replacing the previously hardcoded K value calculation. This provides more flexible control over direction selection.

### Configuration Parameters

#### In `configs/ofdm-5g-sionna.yml`
```yaml
ray_tracing:
  # MLP-based direction selection
  top_k_directions: 32            # Number of top-K directions to select (configurable)
```

### How It Works

#### 1. Configuration Priority
- **If `top_k_directions` is set**: Use the configured value
- **If not set**: Use default formula `min(32, total_directions // 4)`

#### 2. Default Formula Examples

| Configuration | Total Directions | Default K Value | Sampling Rate |
|---------------|------------------|-----------------|---------------|
| 8×4 = 32 | 32 | min(32, 32÷4) = **8** | 25% |
| 16×8 = 128 | 128 | min(32, 128÷4) = **32** | 25% |
| 18×9 = 162 | 162 | min(32, 162÷4) = **32** | 19.8% |
| 36×18 = 648 | 648 | min(32, 648÷4) = **32** | 4.9% |

#### 3. Custom Configuration Examples

```yaml
# High-precision training
ray_tracing:
  azimuth_divisions: 36
  elevation_divisions: 18
  top_k_directions: 64    # Select 64 directions for higher precision

# Fast training
ray_tracing:
  azimuth_divisions: 18
  elevation_divisions: 9
  top_k_directions: 16    # Select 16 directions for faster speed
```

### Performance Impact

#### Accuracy vs Speed Trade-off

- **Higher K values**:
  - ✅ Higher ray tracing precision
  - ❌ Slower training speed
  
- **Lower K values**:
  - ✅ Faster training speed
  - ❌ May reduce ray tracing precision

#### Recommended Configuration

| Training Phase | Recommended K Value | Description |
|----------------|---------------------|-------------|
| Initial training | 16-32 | Fast convergence, establish basic model |
| Mid-training | 32-48 | Balance precision and speed |
| Final training | 48-64 | High precision for final optimization |

### Curriculum Learning Integration

The `curriculum_learning` section in the configuration file already includes `top_k_directions` settings for different phases:

```yaml
curriculum_learning:
  phases:
    - phase: 0
      top_k_directions: 16    # Initial phase: 16 directions
    - phase: 1  
      top_k_directions: 32    # Mid phase: 32 directions
    - phase: 2
      top_k_directions: 64    # Final phase: 64 directions
```

**Note**: The curriculum learning functionality currently requires further development to automatically use these configurations.

### Summary

✅ **Configurable K values**: No longer hardcoded, adjustable via configuration file  
✅ **Backward compatibility**: Automatically uses default formula when not set  
✅ **Flexible control**: Supports different precision requirements for different training phases  
✅ **Performance optimization**: Can adjust precision vs speed balance based on hardware capabilities  

---

## Recent Updates and Improvements (January 2025)

### Major Configuration Enhancements

The configuration system has been completely overhauled with significant improvements:

#### ✅ **Key New Features**
1. **Automatic GPU Selection**: Intelligent GPU detection and selection
2. **Template Variables**: Dynamic path resolution with `{{base_dir}}` syntax
3. **Unified Data Handling**: Single dataset with automatic train/test splitting
4. **Enhanced Testing Pipeline**: Comprehensive testing with visualization
5. **Improved CUDA Support**: Better CUDA acceleration with fallback mechanisms
6. **Centralized Logging**: Dedicated logging configuration for training and testing

#### 🚀 **Performance Improvements**
- **Reduced Complexity**: Optimized angular and spatial sampling for faster training
- **Smart Memory Management**: CUDA memory pool and automatic memory optimization
- **Efficient Data Loading**: Streamlined data pipeline with configurable split ratios
- **Parallel Processing**: Enhanced CPU multiprocessing support

#### 🔧 **System Enhancements**
- **Automatic Hardware Detection**: No manual GPU configuration needed
- **Robust Fallback Mechanisms**: Seamless CPU fallback when CUDA unavailable
- **Flexible Ray Tracing Modes**: 'cuda', 'cpu', and 'hybrid' execution modes
- **Advanced Subcarrier Sampling**: Improved OFDM subcarrier selection strategies

#### 📊 **Configuration Benefits**
- **Simplified Setup**: Minimal manual configuration required
- **Better Organization**: Logical grouping with clear documentation
- **Production Ready**: Robust error handling and recovery mechanisms
- **Scalable Architecture**: Support for various hardware configurations

## CUDA Acceleration Guide

### Overview
The Prism system now supports CUDA-accelerated ray tracing for maximum performance. CUDA acceleration can provide significant speedup for ray tracing operations.

### Performance Improvements
- **Small workloads (100-1000 rays)**: 2-5x faster
- **Medium workloads (1000-10000 rays)**: 5-15x faster  
- **Large workloads (10000+ rays)**: 15-50x faster

### Enabling CUDA Acceleration
To enable CUDA acceleration, set the following in your configuration:

```yaml
ray_tracer_integration:
  use_cuda_ray_tracer: true            # Enable CUDA-accelerated ray tracer
  cuda_fallback_to_cpu: true           # Fallback to CPU if CUDA not available
  cuda_memory_fraction: 0.8            # Use 80% of GPU memory
```

### CUDA Configuration Options

#### Basic CUDA Settings
```yaml
ray_tracer_integration:
  use_cuda_ray_tracer: true            # Enable CUDA-accelerated ray tracer (recommended for GPU systems)
  cuda_fallback_to_cpu: true           # Automatic fallback behavior when CUDA is not available
  cuda_memory_fraction: 0.8            # GPU memory usage fraction for CUDA operations
```

**Description**:
- **use_cuda_ray_tracer**: Controls whether to use CUDARayTracer for GPU acceleration
- **cuda_fallback_to_cpu**: Automatically switch to CPU ray tracer if CUDA fails
- **cuda_memory_fraction**: Range 0.1 to 1.0 (10% to 100% of GPU memory)

#### Advanced CUDA Settings
```yaml
performance:
  cuda_device_id: 0                    # CUDA device ID to use for ray tracing operations
  cuda_optimization_level: 'O2'        # CUDA compilation optimization level
  cuda_benchmark_mode: true            # Enable CUDA benchmark mode for performance tuning
  cuda_deterministic: false            # CUDA deterministic mode for reproducible results
  cuda_memory_pool: true               # Enable CUDA memory pool for better memory management
```

**Description**:
- **cuda_device_id**: 0 for first GPU, 1 for second GPU, -1 for auto-select
- **cuda_optimization_level**: 'O0' (no optimization) to 'O3' (maximum optimization)
- **cuda_benchmark_mode**: Measure and report CUDA kernel performance
- **cuda_deterministic**: Ensure reproducible results (slower performance)
- **cuda_memory_pool**: Efficient memory allocation for long-running sessions

### Automatic Behavior
The system will automatically:
1. **Detect CUDA availability** when `use_cuda_ray_tracer: true`
2. **Use CUDARayTracer** when CUDA is available
3. **Fall back to DiscreteRayTracer** (CPU) if CUDA fails
4. **Optimize memory usage** and performance based on your settings

### Requirements
- **CUDA-capable GPU** with sufficient memory
- **PyTorch with CUDA support** installed
- **CUDA toolkit** compatible with your PyTorch version

---

## Configuration Examples

### Example 1: High-Performance CUDA Configuration (2025)
```yaml
# System configuration for maximum performance with automatic GPU selection
system:
  device: 'cuda'
  batch_size: 4
  ray_tracing_mode: 'cuda'
  fallback_to_cpu: true  # Always enable fallback for robustness
  gpu_memory_fraction: 0.9
  
  cuda:
    # No manual GPU selection needed - automatic detection
    auto_select_gpu: true
    optimization_level: 'O3'
    benchmark_mode: true
    deterministic: false
    memory_pool: true

# Ray tracing with high resolution
ray_tracing:
  angular_sampling:
    azimuth_divisions: 36
    elevation_divisions: 18
    top_k_directions: 64
  
  radial_sampling:
    num_sampling_points: 128
    resampled_points: 64

# Unified data handling
input:
  dataset_path: "data/sionna/sionna_5g_simulation.h5"
  split:
    train_ratio: 0.8
    test_ratio: 0.2
    random_seed: 42

# Template-based output paths
output:
  base_dir: "results/sionna"
  training:
    models_dir: "{{base_dir}}/training/models"
    checkpoint_dir: "{{base_dir}}/training/checkpoints"
```

### Example 2: Balanced Performance Configuration (2025)
```yaml
# System configuration for balanced performance and stability
system:
  device: 'cuda'
  batch_size: 2
  ray_tracing_mode: 'hybrid'  # Neural networks on GPU, ray tracing on CPU
  fallback_to_cpu: true
  gpu_memory_fraction: 0.8
  
  cuda:
    auto_select_gpu: true  # Automatic GPU selection
    optimization_level: 'O2'
    benchmark_mode: true
    deterministic: false
    memory_pool: true
  
  cpu:
    num_workers: 4

# Ray tracing with medium resolution (current default)
ray_tracing:
  angular_sampling:
    azimuth_divisions: 18
    elevation_divisions: 9
    top_k_directions: 32
  
  radial_sampling:
    num_sampling_points: 64
    resampled_points: 32
  
  subcarrier_sampling:
    sampling_ratio: 0.01  # Use 1% of subcarriers for efficiency
    sampling_method: 'random'
    antenna_consistent: true

# Unified data with smaller split for faster experimentation
input:
  dataset_path: "data/sionna/sionna_5g_simulation.h5"
  split:
    train_ratio: 0.2  # Use only 20% for training
    test_ratio: 0.1   # Use only 10% for testing
    random_seed: 42

# Template-based paths with organized structure
output:
  base_dir: "results/sionna"
  training:
    checkpoint_dir: "{{base_dir}}/training/checkpoints"
    models_dir: "{{base_dir}}/training/models"
    logging:
      log_file: "{{base_dir}}/training/logs/training.log"
  testing:
    results_dir: "{{base_dir}}/testing/results"
    plots_dir: "{{base_dir}}/testing/plots"
```

### Example 3: CPU-Only Configuration
```yaml
# System configuration for CPU-only execution
system:
  device: 'cpu'
  batch_size: 1
  ray_tracing_mode: 'cpu'
  fallback_to_cpu: true
  
  cpu:
    num_workers: 8

# Ray tracing with lower resolution for CPU efficiency
ray_tracing:
  angular_sampling:
    azimuth_divisions: 12
    elevation_divisions: 6
    top_k_directions: 16
  
  radial_sampling:
    num_sampling_points: 32
    resampled_points: 16
```

### Example 4: Development/Testing Configuration
```yaml
# System configuration for development and testing
system:
  device: 'cuda'
  batch_size: 1
  ray_tracing_mode: 'hybrid'
  fallback_to_cpu: true
  gpu_memory_fraction: 0.6
  
  cuda:
    device_id: 0
    optimization_level: 'O1'
    benchmark_mode: true
    deterministic: true  # Reproducible results for testing
  
  mixed_precision:
    enabled: false  # Disable for debugging

# Training configuration for quick testing
training:
  learning_rate: 1e-3
  num_epochs: 1
  batches_per_epoch: 10
  auto_checkpoint: true
  checkpoint_frequency: 5
```

### Example 5: Production Training Configuration
```yaml
# System configuration for production training
system:
  device: 'cuda'
  batch_size: 8
  ray_tracing_mode: 'cuda'
  fallback_to_cpu: true
  gpu_memory_fraction: 0.85
  
  cuda:
    device_id: 0
    optimization_level: 'O2'
    benchmark_mode: false
    deterministic: false
    memory_pool: true
  
  mixed_precision:
    enabled: true
    autocast_enabled: true
    grad_scaler_enabled: true
    loss_scale: "dynamic"

# Training configuration for production
training:
  learning_rate: 1e-4
  num_epochs: 100
  batches_per_epoch: 200
  auto_checkpoint: true
  checkpoint_frequency: 50
  
  early_stopping:
    enabled: true
    patience: 20
    min_delta: 1e-7
```

### Example 6: Multi-GPU Configuration (2025)
```yaml
# System configuration for multi-GPU setup with automatic selection
system:
  device: 'cuda'
  batch_size: 16
  ray_tracing_mode: 'cuda'
  fallback_to_cpu: true
  gpu_memory_fraction: 0.8
  
  cuda:
    auto_select_gpu: true  # Automatic GPU selection
    multi_gpu: true        # Enable multi-GPU if available
    optimization_level: 'O2'
    benchmark_mode: true
    memory_pool: true

# High-throughput data configuration
input:
  dataset_path: "data/sionna/sionna_5g_simulation.h5"
  split:
    train_ratio: 0.8  # Use most data for training
    test_ratio: 0.2
    random_seed: 42

# Production training settings
training:
  learning_rate: 1e-4
  num_epochs: 100
  batches_per_epoch: 200
  auto_checkpoint: true
  checkpoint_frequency: 50
```

---

## Troubleshooting Guide

### Common Configuration Issues

#### 1. CUDA Not Available
**Problem**: System falls back to CPU even when GPU is available
```
WARNING: CUDA not available, falling back to CPU
```

**Solutions**:
```yaml
# Check CUDA configuration
system:
  device: 'cuda'
  fallback_to_cpu: true  # Ensure fallback is enabled
  cuda:
    auto_select_gpu: true  # Let system choose best GPU
```

**Debugging Steps**:
1. Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Check GPU memory: `nvidia-smi`
3. Enable debug logging: `log_level: 'DEBUG'`

#### 2. Template Variable Errors
**Problem**: Path template variables not resolved
```
ERROR: Path {{base_dir}}/training/models not found
```

**Solution**: Ensure ConfigLoader processes template variables correctly
```yaml
output:
  base_dir: "results/sionna"  # Must be defined before use
  training:
    models_dir: "{{base_dir}}/training/models"  # Correct syntax
```

#### 3. Data Loading Issues
**Problem**: Dataset not found or split configuration errors
```
ERROR: Dataset file not found: data/sionna/sionna_5g_simulation.h5
```

**Solutions**:
```yaml
# Check data path and split configuration
input:
  dataset_path: "data/sionna/sionna_5g_simulation.h5"  # Verify file exists
  split:
    train_ratio: 0.2  # Ensure ratios are valid (0.0 to 1.0)
    test_ratio: 0.1
    random_seed: 42   # For reproducible splits
```

#### 4. Memory Issues
**Problem**: Out of memory errors during training
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```yaml
system:
  batch_size: 1  # Reduce batch size
  gpu_memory_fraction: 0.6  # Use less GPU memory
  
ray_tracing:
  radial_sampling:
    num_sampling_points: 32  # Reduce sampling points
    resampled_points: 16
  
  subcarrier_sampling:
    sampling_ratio: 0.005  # Use fewer subcarriers
```

#### 5. Performance Issues
**Problem**: Training is too slow
```
INFO: Epoch 1 taking longer than expected...
```

**Solutions**:
```yaml
# Optimize for speed
ray_tracing:
  angular_sampling:
    azimuth_divisions: 12    # Reduce angular resolution
    elevation_divisions: 6
    top_k_directions: 16     # Use fewer directions
  
  radial_sampling:
    num_sampling_points: 32  # Reduce spatial sampling
    resampled_points: 16
  
  subcarrier_sampling:
    sampling_ratio: 0.005    # Use minimal subcarriers for testing
```

### Best Practices

#### 1. Development vs Production
```yaml
# Development (fast iteration)
training:
  num_epochs: 2
  batches_per_epoch: 5
  checkpoint_frequency: 1

input:
  split:
    train_ratio: 0.1  # Use small subset
    test_ratio: 0.05

# Production (full training)
training:
  num_epochs: 100
  batches_per_epoch: 200
  checkpoint_frequency: 50

input:
  split:
    train_ratio: 0.8  # Use most data
    test_ratio: 0.2
```

#### 2. Hardware-Specific Optimization
```yaml
# High-end GPU (RTX 4090, A100)
system:
  batch_size: 8
  gpu_memory_fraction: 0.9
ray_tracing:
  top_k_directions: 64

# Mid-range GPU (RTX 3070, RTX 4070)
system:
  batch_size: 4
  gpu_memory_fraction: 0.8
ray_tracing:
  top_k_directions: 32

# Low-end GPU or CPU
system:
  batch_size: 1
  ray_tracing_mode: 'cpu'
ray_tracing:
  top_k_directions: 16
```

#### 3. Monitoring and Debugging
```yaml
output:
  training:
    logging:
      log_level: 'DEBUG'  # Enable detailed logging
  testing:
    logging:
      log_level: 'INFO'   # Standard logging for testing

training:
  auto_checkpoint: true
  checkpoint_frequency: 10  # Frequent checkpoints for debugging
```
