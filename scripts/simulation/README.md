# Simulation Directory

This directory contains all simulation-related scripts, configurations, and documentation for the Prism project using NVIDIA Sionna.

## 📁 Directory Structure

```
scripts/simulation/
├── README.md                           # This file - Main simulation overview
├── requirements_sionna.txt             # Python dependencies for Sionna
├── install_sionna.sh                  # Automated installation script
├── test_sionna_simulation.py          # Test script to verify setup
├── sionna_simulation.py               # Generic 5G OFDM simulation
├── sionna_generator.py                # Sionna simulation data generator
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

### 2. Test the Setup
```bash
python test_sionna_simulation.py
```

### 3. Run Simulations

#### Generic 5G Simulation
```bash
python sionna_simulation.py
```

### 4. Complete Training Pipeline (Recommended)
```bash
python run_training_pipeline.py \
    --data data/sionna_simulation.h5 \
    --config ../../configs/ofdm-5g-sionna.yml \
    --output results/complete_pipeline
```

## 📋 Available Simulations

### **Generic 5G OFDM Simulation** (`sionna_simulation.py`)
- **Frequency**: 3.5 GHz (mid-band 5G)
- **Bandwidth**: 100 MHz
- **Subcarriers**: 408
- **Antennas**: 64 BS, 4 UE
- **Coverage**: 500m × 500m
- **Use Case**: General 5G research and development



## 🔧 Configuration Files

The simulations use configuration files located in the `configs/` directory:
- `configs/ofdm-wifi.yml` - WiFi-like OFDM configuration
- `configs/ofdm-wideband.yml` - Ultra-wideband OFDM configuration
- `configs/ofdm-5g-sionna.yml` - 5G OFDM configuration for Sionna

## 📊 Output Data

### **Data Files**
- **Generic 5G**: `data/sionna_5g_simulation.h5`

### **Visualizations**
- **Generic 5G**: `data/sionna_simulation_results.png`


### **Data Structure**
```
Channel Responses: (100, N, 4, 64) - Complex matrices
Path Losses:      (100, N)          - Frequency-dependent attenuation
Delays:           (100, N)          - Channel delay information
Positions:        (100, 3)          - UE and BS coordinates
```

Where `N` is the number of subcarriers (408 for generic 5G).

## 🎯 Key Features

### **Common Features**
- NVIDIA Sionna-based channel modeling
- Realistic urban environment simulation
- Massive MIMO support (64×4 antenna configuration)
- OFDM with configurable subcarriers
- HDF5 data export for easy integration
- Comprehensive visualization plots
- GPU-accelerated ray tracing support (experimental)

### **GPU Ray Tracing** 🚧
- **Status**: Under development (see `TODO.MD`)
- **Features**: GPU-accelerated RF signal propagation modeling
- **Integration**: Works with existing Sionna simulations
- **Testing**: Use `../test_gpu_ray_tracer.py` for validation



## 🔄 Integration with Prism

### **Training with Generated Data**
```bash
# Train with generic 5G data
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
```

## 🎯 Complete Training Workflow

### **1. Data Generation** 📊

使用Sionna生成模拟数据：

```bash
# 生成5G OFDM模拟数据
python sionna_generator.py

# 输出文件：data/sionna_5g_simulation.h5
```

**数据格式**：
- `ue_positions`: UE位置数据 (N, 3)
- `channel_responses`: 信道响应数据 (N, K) - 复数
- `bs_position`: 基站位置 (3,)
- `simulation_params`: 模拟参数字典

其中N=UE数量，K=子载波数量(408)

### **2. Data Preparation** ✂️

将数据分割为训练集(80%)和测试集(20%)：

```bash
# 数据准备和分割
python data_prepare.py \
    --data data/sionna_5g_simulation.h5 \
    --output data/split \
    --train-ratio 0.8 \
    --seed 42 \
    --verify
```

**输出文件**：
- `data/split/train_data.h5` - 训练数据
- `data/split/test_data.h5` - 测试数据
- `data/split/split_summary.txt` - 分割摘要

### **3. Model Training** 🚀

训练Prism神经网络：

```bash
# 训练模型
python train_prism.py \
    --config ../../configs/ofdm-5g-sionna.yml \
    --data data/split/train_data.h5 \
    --output results/training
```

**训练特性**：
- ✅ CUDA加速支持
- ✅ 自动设备检测(GPU/CPU)
- ✅ 混合精度训练
- ✅ 学习率调度
- ✅ 早停机制
- ✅ TensorBoard监控
- ✅ 自动检查点保存

**输出结果**：
- 模型检查点文件
- 训练日志和指标
- TensorBoard日志
- 训练曲线图

### **4. Model Testing** 🧪

测试训练好的模型：

```bash
# 测试模型
python test_prism.py \
    --config ../../configs/ofdm-5g-sionna.yml \
    --model results/training/best_model.pt \
    --data data/split/test_data.h5 \
    --output results/testing
```

**测试指标**：
- 复数MSE
- 幅度误差
- 相位误差
- 相关性系数
- NMSE (归一化均方误差)
- SNR (信噪比)

**可视化结果**：
- CSI幅度和相位对比
- 误差分布图
- 空间性能图
- 子载波性能分析

### **5. Complete Pipeline** 🔄

一键运行完整流程：

```bash
python run_training_pipeline.py \
    --data data/sionna_5g_simulation.h5 \
    --config ../../configs/ofdm-5g-sionna.yml \
    --output results/complete_pipeline
```

自动执行：数据准备 → 训练 → 测试 → 报告生成

## 📊 Training Configuration

配置文件 `configs/ofdm-5g-sionna.yml` 包含：

- **神经网络架构**：隐藏层维度、激活函数、正则化
- **射线追踪配置**：角度分割、空间采样、GPU加速
- **性能设置**：批处理大小、学习率、优化器
- **输出选项**：日志级别、保存格式、可视化

## 🔧 Performance Optimization

### **GPU加速**
- 自动CUDA检测和回退
- 混合精度训练(FP16/FP32)
- GPU内存管理优化
- 批处理并行处理

### **训练优化**
- AdamW优化器 + 权重衰减
- 学习率调度(ReduceLROnPlateau)
- 梯度裁剪防止爆炸
- 早停机制避免过拟合

## 📈 Monitoring & Visualization

### **TensorBoard监控**
```bash
tensorboard --logdir results/complete_pipeline/training/tensorboard
```

### **训练曲线**
- 训练/验证损失曲线
- 学习率变化曲线
- 参数分布直方图
- 梯度分布监控

## 🚨 Troubleshooting

### **常见问题**
1. **内存不足**: 减少batch_size或启用梯度累积
2. **训练不收敛**: 调整学习率或检查数据质量
3. **GPU错误**: 检查CUDA版本兼容性
4. **数据加载慢**: 增加num_workers或使用SSD

### **日志文件**
- `training_pipeline.log`: 完整流程日志
- `training.log`: 训练过程日志
- `testing.log`: 测试过程日志
- `data_preparation.log`: 数据准备日志

## 🎯 Next Steps

1. **运行完整流程**: 使用 `run_training_pipeline.py`
2. **监控训练**: 通过TensorBoard观察训练进度
3. **分析结果**: 查看测试结果和可视化图表
4. **调优参数**: 根据性能调整网络架构和超参数
5. **部署模型**: 将训练好的模型用于推理

## 🛠️ Customization

### **Modifying Simulation Parameters**
1. Edit the simulation script files directly
2. Modify configuration files in `configs/` directory
3. Adjust channel model parameters in the scripts
4. Change deployment area and UE positioning

### **Adding New Bands**
1. Copy an existing simulation script
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

## 🚀 Next Steps

1. **Run Simulations**: Execute the simulation scripts to generate data
2. **Analyze Results**: Review channel characteristics and performance metrics
3. **Train Models**: Use generated data with Prism neural networks
4. **Extend Scenarios**: Create additional frequency band configurations
5. **Performance Analysis**: Evaluate system capacity and coverage

---

**Note**: All simulations are designed to work with the Prism framework and generate realistic channel data for neural network training and analysis.
