# Prism 故障排除指南

## 概述

本文档提供Prism系统常见问题的诊断和解决方案，帮助用户快速解决安装、配置、训练和推理过程中遇到的问题。

## 目录

1. [安装问题](#安装问题)
2. [CUDA和GPU问题](#cuda和gpu问题)
3. [配置问题](#配置问题)
4. [训练问题](#训练问题)
5. [内存问题](#内存问题)
6. [性能问题](#性能问题)
7. [数据问题](#数据问题)
8. [网络架构问题](#网络架构问题)
9. [损失函数问题](#损失函数问题)
10. [可视化问题](#可视化问题)

---

## 安装问题

### 问题: 依赖包安装失败

**症状:**
```bash
ERROR: Could not find a version that satisfies the requirement torch>=1.12.0
```

**解决方案:**
```bash
# 1. 更新pip
pip install --upgrade pip

# 2. 使用清华源安装PyTorch
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 安装其他依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 开发模式安装
pip install -e .
```

### 问题: 导入模块失败

**症状:**
```python
ModuleNotFoundError: No module named 'prism'
```

**解决方案:**
```bash
# 确保在项目根目录
cd /path/to/Prism

# 开发模式安装
pip install -e .

# 或者添加到Python路径
export PYTHONPATH="${PYTHONPATH}:/path/to/Prism/src"
```

### 问题: HDF5依赖问题

**症状:**
```bash
ImportError: libhdf5.so.103: cannot open shared object file
```

**解决方案:**
```bash
# Ubuntu/Debian
sudo apt-get install libhdf5-dev

# CentOS/RHEL
sudo yum install hdf5-devel

# macOS
brew install hdf5

# 重新安装h5py
pip uninstall h5py
pip install h5py --no-binary=h5py
```

---

## CUDA和GPU问题

### 问题: CUDA不可用

**症状:**
```python
RuntimeError: CUDA not available
```

**诊断:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

**解决方案:**
```bash
# 1. 检查NVIDIA驱动
nvidia-smi

# 2. 检查CUDA安装
nvcc --version

# 3. 重新安装PyTorch (CUDA版本)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 如果没有GPU，使用CPU配置
# 在配置文件中设置:
system:
  device: 'cpu'
  force_cpu: true
```

### 问题: GPU内存不足

**症状:**
```python
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案:**
```yaml
# 1. 减少批大小
training:
  batch_size: 8  # 从32减少到8

# 2. 启用梯度检查点
training:
  gradient_checkpointing: true

# 3. 使用混合精度
training:
  use_amp: true

# 4. 减少空间采样
ray_tracing:
  spatial_sampling:
    num_samples_per_ray: 32  # 从64减少到32
```

### 问题: CUDA版本不匹配

**症状:**
```python
RuntimeError: The NVIDIA driver on your system is too old
```

**解决方案:**
```bash
# 1. 更新NVIDIA驱动
sudo apt update
sudo apt install nvidia-driver-525

# 2. 或者使用兼容的PyTorch版本
pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# 3. 重启系统
sudo reboot
```

---

## 配置问题

### 问题: 配置文件解析错误

**症状:**
```python
yaml.scanner.ScannerError: mapping values are not allowed here
```

**解决方案:**
```yaml
# 检查YAML语法
# 错误示例:
training:
  batch_size = 32  # 错误: 使用了等号

# 正确示例:
training:
  batch_size: 32   # 正确: 使用冒号

# 检查缩进 (必须使用空格，不能使用Tab)
neural_networks:
  num_subcarriers: 408
  num_ue_antennas: 1
```

### 问题: 必需参数缺失

**症状:**
```python
ValueError: Configuration must contain 'base_station.antenna_array.configuration'
```

**解决方案:**
```yaml
# 确保所有必需参数都存在
base_station:
  antenna_array:
    configuration: "8x8"           # 必需
    element_spacing: "half_wavelength"  # 必需
  ofdm:
    center_frequency: 3.5e9        # 必需
    bandwidth: 1.224e7             # 必需
    num_subcarriers: 408           # 必需

training:
  loss:
    spatial_spectrum_loss:
      algorithm: 'bartlett'        # 必需
      fusion_method: 'average'     # 必需
      theta_range: [0, 5, 90]      # 必需
      phi_range: [0, 10, 360]      # 必需
```

### 问题: 路径解析错误

**症状:**
```python
FileNotFoundError: [Errno 2] No such file or directory: '${PROJECT_ROOT}/data'
```

**解决方案:**
```yaml
# 使用绝对路径或正确的相对路径
input:
  dataset_path: "data/sionna/P300/P300.h5"  # 相对于项目根目录

# 或者使用绝对路径
input:
  dataset_path: "/home/user/Prism/data/sionna/P300/P300.h5"

# 确保文件存在
ls -la data/sionna/P300/P300.h5
```

---

## 训练问题

### 问题: 训练损失不收敛

**症状:**
- 损失值保持很高或震荡
- 损失值变为NaN或inf

**诊断:**
```python
# 检查学习率
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

# 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm().item():.6f}")
```

**解决方案:**
```yaml
# 1. 调整学习率
training:
  learning_rate: 0.0001  # 从0.001减少到0.0001

# 2. 使用学习率调度器
training:
  scheduler:
    type: 'cosine'
    T_max: 100

# 3. 添加梯度裁剪
training:
  gradient_clipping: 1.0

# 4. 调整损失权重
training:
  loss:
    csi_weight: 0.8
    pdp_weight: 0.2
    spatial_spectrum_weight: 0.0  # 先关闭空间频谱损失
```

### 问题: 训练速度过慢

**症状:**
- 每个epoch耗时过长
- GPU利用率低

**解决方案:**
```yaml
# 1. 增加批大小 (如果内存允许)
training:
  batch_size: 32

# 2. 使用数据并行
training:
  use_data_parallel: true

# 3. 优化射线追踪
ray_tracing:
  implementation: 'cuda'  # 使用CUDA加速
  spatial_sampling:
    num_samples_per_ray: 64  # 适中的采样数

# 4. 启用编译优化
training:
  compile_model: true  # PyTorch 2.0+
```

### 问题: 梯度爆炸或消失

**症状:**
```python
RuntimeError: Function 'AddmmBackward' returned nan values in its 0th output.
```

**解决方案:**
```yaml
# 1. 梯度裁剪
training:
  gradient_clipping: 1.0

# 2. 降低学习率
training:
  learning_rate: 0.0001

# 3. 使用更稳定的优化器
training:
  optimizer: 'adamw'
  weight_decay: 0.01

# 4. 检查网络初始化
neural_networks:
  initialization: 'xavier_uniform'
```

---

## 内存问题

### 问题: 系统内存不足

**症状:**
```python
MemoryError: Unable to allocate array
```

**解决方案:**
```yaml
# 1. 减少数据加载
input:
  max_samples: 1000  # 限制样本数量

# 2. 使用数据流式加载
training:
  dataloader:
    num_workers: 2
    pin_memory: false

# 3. 减少缓存
training:
  cache_data: false
```

### 问题: GPU显存泄漏

**症状:**
- 训练过程中GPU内存持续增长
- 最终导致OOM错误

**解决方案:**
```python
# 1. 显式清理缓存
import torch
torch.cuda.empty_cache()

# 2. 使用上下文管理器
with torch.no_grad():
    predictions = model(inputs)

# 3. 及时删除大张量
del large_tensor
torch.cuda.empty_cache()

# 4. 检查循环引用
import gc
gc.collect()
```

---

## 性能问题

### 问题: 推理速度慢

**症状:**
- 单次预测耗时过长
- 批处理效率低

**解决方案:**
```python
# 1. 使用eval模式
model.eval()

# 2. 禁用梯度计算
with torch.no_grad():
    predictions = model(inputs)

# 3. 使用JIT编译
model = torch.jit.script(model)

# 4. 批量预测
batch_predictions = model(batch_inputs)
```

### 问题: 射线追踪性能瓶颈

**症状:**
- 射线追踪占用大量计算时间

**解决方案:**
```yaml
# 1. 优化采样策略
ray_tracing:
  spatial_sampling:
    adaptive_sampling: true
    min_samples: 32
    max_samples: 128

# 2. 使用方向性采样
ray_tracing:
  directional_sampling:
    enabled: true
    top_k_directions: 32

# 3. 并行化设置
ray_tracing:
  parallel_processing:
    num_threads: 8
```

---

## 数据问题

### 问题: 数据加载失败

**症状:**
```python
OSError: Unable to open file (file signature not found)
```

**解决方案:**
```bash
# 1. 检查文件完整性
ls -la data/sionna/P300/P300.h5
file data/sionna/P300/P300.h5

# 2. 重新生成数据
cd data/sionna
python generator.py -n 300

# 3. 检查文件权限
chmod 644 data/sionna/P300/P300.h5
```

### 问题: 数据维度不匹配

**症状:**
```python
RuntimeError: Expected 4D tensor, got 3D tensor
```

**解决方案:**
```python
# 检查数据形状
print(f"CSI shape: {csi_data.shape}")
print(f"Expected: (samples, subcarriers, ue_antennas, bs_antennas)")

# 修正维度
if len(csi_data.shape) == 3:
    csi_data = csi_data.unsqueeze(2)  # 添加UE天线维度
```

### 问题: 数据类型错误

**症状:**
```python
RuntimeError: Expected tensor to be complex, got float
```

**解决方案:**
```python
# 转换为复数类型
if not csi_data.is_complex():
    csi_data = csi_data.to(torch.complex64)

# 或者在配置中指定
neural_networks:
  complex_output: true
```

---

## 网络架构问题

### 问题: 网络参数不匹配

**症状:**
```python
RuntimeError: size mismatch for attenuation_network.layers.0.weight
```

**解决方案:**
```yaml
# 检查网络配置一致性
neural_networks:
  num_subcarriers: 408      # 必须与数据一致
  num_ue_antennas: 1        # 必须与数据一致  
  num_bs_antennas: 64       # 必须与数据一致

# 如果加载预训练模型，确保配置匹配
```

### 问题: 天线嵌入维度错误

**症状:**
```python
IndexError: index 64 is out of range for dimension 0 with size 64
```

**解决方案:**
```yaml
# 确保天线索引在有效范围内
base_station:
  antenna_array:
    configuration: "8x8"  # 8*8 = 64个天线，索引0-63

# 检查天线索引
neural_networks:
  num_bs_antennas: 64  # 必须匹配天线阵列配置
```

---

## 损失函数问题

### 问题: 空间频谱损失为零

**症状:**
- 空间频谱损失始终为0
- 相关可视化图表为空

**解决方案:**
```yaml
# 1. 确保启用空间频谱损失
training:
  loss:
    spatial_spectrum_weight: 0.1  # 必须大于0
    spatial_spectrum_loss:
      enabled: true               # 必须为true

# 2. 检查配置完整性
base_station:
  antenna_array:
    configuration: "8x8"
    element_spacing: "half_wavelength"
  ofdm:
    center_frequency: 3.5e9
    num_subcarriers: 408

# 3. 检查角度范围
training:
  loss:
    spatial_spectrum_loss:
      theta_range: [0, 5, 90]     # [min, step, max] in degrees
      phi_range: [0, 10, 360]     # [min, step, max] in degrees
```

### 问题: 相位损失异常

**症状:**
- 相位损失值异常大或为NaN

**解决方案:**
```python
# 检查CSI数据中的零值
zero_count = torch.sum(torch.abs(csi_data) < 1e-10)
print(f"Zero CSI count: {zero_count}")

# 相位计算会自动添加小量避免零值
# 如果仍有问题，增加小量值
pred_phase = torch.angle(predicted_csi + 1e-6)  # 从1e-8增加到1e-6
```

### 问题: PDP损失计算失败

**症状:**
```python
RuntimeError: FFT input must be at least 1D
```

**解决方案:**
```yaml
# 检查FFT大小设置
training:
  loss:
    pdp_loss:
      fft_size: 1024  # 必须是2的幂次

# 确保CSI数据不为空
# 检查子载波数量
base_station:
  ofdm:
    num_subcarriers: 408  # 必须与数据匹配
```

---

## 可视化问题

### 问题: 图表生成失败

**症状:**
```python
ImportError: No module named 'matplotlib'
```

**解决方案:**
```bash
# 安装可视化依赖
pip install matplotlib seaborn

# 如果是服务器环境，设置后端
export MPLBACKEND=Agg
```

### 问题: 空间频谱图表为空

**症状:**
- 生成的图表没有内容
- 颜色条范围异常

**解决方案:**
```python
# 检查频谱数据
print(f"Spectrum range: [{spectrum.min():.6f}, {spectrum.max():.6f}]")
print(f"Spectrum shape: {spectrum.shape}")

# 如果数据全为零，检查CSI输入
print(f"CSI magnitude range: [{torch.abs(csi).min():.6f}, {torch.abs(csi).max():.6f}]")
```

---

## 调试技巧

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或者在配置中设置
system:
  log_level: 'DEBUG'
```

### 使用断点调试

```python
import pdb
pdb.set_trace()  # 在关键位置设置断点

# 或者使用IPython
import IPython
IPython.embed()
```

### 监控资源使用

```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控内存使用
htop

# 监控磁盘空间
df -h
```

### 性能分析

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # 运行代码
    output = model(input_data)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## 获取帮助

### 收集诊断信息

运行以下脚本收集系统信息：

```python
import torch
import sys
import platform

print("=== 系统信息 ===")
print(f"Python版本: {sys.version}")
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")

print("\n=== Prism配置 ===")
# 添加配置信息
```

### 提交问题报告

在GitHub Issues中包含以下信息：

1. **问题描述**: 详细描述遇到的问题
2. **重现步骤**: 提供可重现问题的最小示例
3. **错误信息**: 完整的错误堆栈跟踪
4. **系统信息**: 使用上述脚本收集的信息
5. **配置文件**: 相关的配置文件内容
6. **预期行为**: 描述期望的正确行为

### 社区支持

- **GitHub Discussions**: 一般问题和讨论
- **GitHub Issues**: Bug报告和功能请求
- **文档**: 查阅完整文档获取更多信息

---

*文档版本: v1.0*  
*最后更新: 2025年1月*  
*维护者: Prism项目团队*
