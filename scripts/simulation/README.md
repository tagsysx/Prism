# Simulation Directory

This directory contains all simulation-related scripts, configurations, and documentation for the Prism project using NVIDIA Sionna.

## 📁 Directory Structure

```
scripts/simulation/
├── README.md                           # This file - Main simulation overview
├── requirements_sionna.txt             # Python dependencies for Sionna
├── install_sionna.sh                  # Automated installation script
├── data_generator.py                  # Sionna simulation data generator
├── data_prepare.py                    # Data preparation and splitting script
├── train_prism.py                     # Prism network training script
├── test_prism.py                      # Prism network testing script
├── run_training_pipeline.py           # Complete training pipeline
├── README_SIONNA.md                   # General Sionna simulation guide
└── sionna_simulation_guide.md         # Detailed technical guide
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd scripts/simulation
./install_sionna.sh
```

### 2. Generate Simulation Data
```bash
python data_generator.py
```

### 3. Run Complete Training Pipeline (Recommended)
```bash
python run_training_pipeline.py \
    --data data/sionna_5g_simulation.h5 \
    --config ../../configs/ofdm-5g-sionna.yml \
    --output results/complete_pipeline
```

## 📋 Available Simulations

### **5G OFDM Data Generation** (`data_generator.py`)
- **Frequency**: 3.5 GHz (mid-band 5G)
- **Bandwidth**: 100 MHz
- **Subcarriers**: 408
- **Antennas**: 64 BS, 4 UE
- **Coverage**: 500m × 500m
- **Use Case**: Generate training data for Prism neural networks



## 🔧 Configuration Files

The simulations use configuration files located in the `configs/` directory:
- `configs/ofdm-wifi.yml` - WiFi-like OFDM configuration
- `configs/ofdm-wideband.yml` - Ultra-wideband OFDM configuration
- `configs/ofdm-5g-sionna.yml` - 5G OFDM configuration for Sionna

## 📊 Output Data

### **Data Files**
- **5G OFDM**: `data/sionna_5g_simulation.h5`

### **Visualizations**
- **5G OFDM**: `data/sionna_simulation_results.png`


### **Data Structure**
```
Channel Responses: (100, 408, 4, 64) - Complex matrices
Path Losses:      (100, 408)          - Frequency-dependent attenuation
Delays:           (100, 408)          - Channel delay information
Positions:        (100, 3)            - UE and BS coordinates
```

Where 100 = number of UE positions, 408 = number of subcarriers.

## 🎯 Key Features

### **Common Features**
- NVIDIA Sionna-based channel modeling
- Realistic urban environment simulation
- Massive MIMO support (64×4 antenna configuration)
- OFDM with configurable subcarriers
- HDF5 data export for easy integration
- Comprehensive visualization plots
- Neural network-based ray tracing (Prism framework)

### **Neural Network Ray Tracing** 🚧
- **Status**: Implemented in Prism framework
- **Features**: Neural network-based electromagnetic ray tracing
- **Integration**: Works with Sionna-generated training data
- **Testing**: Use training and testing scripts for validation



## 🔄 Integration with Prism

### **Training with Generated Data**
```bash
# Train with 5G OFDM data
python ../prism_runner.py --mode train --config ../../configs/ofdm-5g-sionna.yml
```

### **Data Loading Example**
```python
import h5py

# Load simulation data
with h5py.File('data/sionna_5g_simulation.h5', 'r') as f:
    channel_responses = f['channel_data/channel_responses'][:]
    ue_positions = f['positions/ue_positions'][:]
    bs_position = f['positions/bs_position'][:]
    path_losses = f['channel_data/path_losses'][:]
    delays = f['channel_data/delays'][:]
```

## 🎯 Complete Training Workflow

### **1. Data Generation** 📊

Generate simulation data using Sionna:

```bash
# Generate 5G OFDM simulation data
python data_generator.py

# Output file: data/sionna_5g_simulation.h5
```

**Data Format**:
- `ue_positions`: UE position data (N, 3)
- `channel_responses`: Channel response data (N, K, 4, 64) - complex
- `bs_position`: Base station position (3,)
- `path_losses`: Path loss data (N, K)
- `delays`: Channel delay data (N, K)
- `simulation_config`: Simulation parameter dictionary

Where N = number of UEs (100), K = number of subcarriers (408)

### **2. Data Preparation** ✂️

Split data into training set (80%) and test set (20%):

```bash
# Data preparation and splitting
python data_prepare.py \
    --data data/sionna_5g_simulation.h5 \
    --output data/split \
    --train-ratio 0.8 \
    --seed 42 \
    --verify
```

**Output Files**:
- `data/split/train_data.h5` - Training data
- `data/split/test_data.h5` - Test data
- `data/split/split_summary.txt` - Split summary

### **3. Model Training** 🚀

Train Prism neural network:

```bash
# Train model
python train_prism.py \
    --config ../../configs/ofdm-5g-sionna.yml \
    --data data/split/train_data.h5 \
    --output results/training

# Resume training from checkpoint
python train_prism.py \
    --config ../../configs/ofdm-5g-sionna.yml \
    --data data/split/train_data.h5 \
    --output results/training \
    --resume results/training/latest_checkpoint.pt
```

**Training Features**:
- ✅ CUDA acceleration support
- ✅ Automatic device detection (GPU/CPU)
- ✅ Mixed precision training
- ✅ Learning rate scheduling
- ✅ Early stopping mechanism
- ✅ TensorBoard monitoring
- ✅ Automatic checkpoint saving
- ✅ Checkpoint resume training
- ✅ Smart checkpoint management

**Output Results**:
- Model checkpoint files
- Training logs and metrics
- TensorBoard logs
- Training curve plots

### **4. Model Testing** 🧪

Test the trained model:

```bash
# Test model
python test_prism.py \
    --config ../../configs/ofdm-5g-sionna.yml \
    --model results/training/best_model.pt \
    --data data/split/test_data.h5 \
    --output results/testing
```

**Test Metrics**:
- Complex MSE
- Magnitude error
- Phase error
- Correlation coefficient
- NMSE (Normalized Mean Square Error)
- SNR (Signal-to-Noise Ratio)

**Visualization Results**:
- CSI magnitude and phase comparison
- Error distribution plots
- Spatial performance maps
- Subcarrier performance analysis

### **5. Complete Pipeline** 🔄

Run complete workflow with one command:

```bash
python run_training_pipeline.py \
    --data data/sionna_5g_simulation.h5 \
    --config ../../configs/ofdm-5g-sionna.yml \
    --output results/complete_pipeline
```

Automatically executes: Data preparation → Training → Testing → Report generation

## 📊 Training Configuration

Configuration file `configs/ofdm-5g-sionna.yml` contains:

- **Neural Network Architecture**: Hidden layer dimensions, activation functions, regularization
- **Ray Tracing Configuration**: Angular divisions, spatial sampling, GPU acceleration
- **Performance Settings**: Batch size, learning rate, optimizer
- **Output Options**: Log level, save format, visualization

## 🔧 Performance Optimization

### **GPU Acceleration**
- Automatic CUDA detection and fallback
- Mixed precision training (FP16/FP32)
- GPU memory management optimization
- Batch processing parallelization

### **Training Optimization**
- AdamW optimizer + weight decay
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping to prevent explosion
- Early stopping to avoid overfitting

### **Checkpoint Resume Training** 🔄
- Automatically save latest checkpoint (`latest_checkpoint.pt`)
- Save best model (`best_model.pt`)
- Smart cleanup of old checkpoints (keep last 5)
- Complete restoration of training state (model, optimizer, scheduler)
- Automatically continue from last interrupted epoch
- Maintain training history and best validation loss records

**Checkpoint files contain**:
- Model weights and state
- Optimizer state (momentum, gradients, etc.)
- Learning rate scheduler state
- Training progress (epoch, loss, etc.)
- Training history records

## 📈 Monitoring & Visualization

### **TensorBoard Monitoring**
```bash
tensorboard --logdir results/complete_pipeline/training/tensorboard
```

### **Training Curves**
- Training/validation loss curves
- Learning rate change curves
- Parameter distribution histograms
- Gradient distribution monitoring

## 🚨 Troubleshooting

### **Common Issues**
1. **Insufficient memory**: Reduce batch_size or enable gradient accumulation
2. **Training not converging**: Adjust learning rate or check data quality
3. **GPU errors**: Check CUDA version compatibility
4. **Slow data loading**: Increase num_workers or use SSD
5. **Checkpoint resume failure**: Check checkpoint file integrity and version compatibility
6. **Training interruption**: Use `--resume` parameter to continue from latest checkpoint

### **Log Files**
- `training_pipeline.log`: Complete pipeline logs
- `training.log`: Training process logs
- `testing.log`: Testing process logs
- `data_preparation.log`: Data preparation logs

## 🎯 Next Steps

1. **Run complete pipeline**: Use `run_training_pipeline.py`
2. **Monitor training**: Observe training progress through TensorBoard
3. **Analyze results**: Review test results and visualization charts
4. **Tune parameters**: Adjust network architecture and hyperparameters based on performance
5. **Deploy model**: Use trained model for inference

## 🛠️ Customization

### **Modifying Simulation Parameters**
1. Edit the data generation script (`data_generator.py`) directly
2. Modify configuration files in `configs/` directory
3. Adjust channel model parameters in the scripts
4. Change deployment area and UE positioning

### **Adding New Bands**
1. Copy the existing data generation script
2. Update frequency, bandwidth, and subcarrier parameters
3. Modify channel model characteristics
4. Update visualization and analysis functions

## 🔍 Troubleshooting

### **Common Issues**
- **Import errors**: Run `./install_sionna.sh`
- **CUDA issues**: Check GPU compatibility
- **Memory errors**: Reduce number of UE positions
- **Simulation time**: Use GPU acceleration

### **Getting Help**
- Check individual README files for specific simulations
- Review `sionna_simulation_guide.md` for technical details
- Consult Sionna documentation: https://nvlabs.github.io/sionna/
- Check configuration files for parameter explanations

## 📚 Documentation

- **`README_SIONNA.md`**: General Sionna simulation overview
- **`sionna_simulation_guide.md`**: Comprehensive technical guide
- **Configuration files**: Detailed parameter explanations



---

**Note**: The data generation and training pipeline are designed to work with the Prism framework and generate realistic channel data for neural network training and analysis. All scripts support both English and Chinese environments.
