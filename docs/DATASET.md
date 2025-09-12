# 数据集格式规范

本文档描述了Prism项目中使用的数据集格式和文件结构规范。

## 📋 概述

Prism项目使用HDF5格式存储5G OFDM信道数据，包括信道状态信息(CSI)、位置信息和仿真配置参数。数据集主要用于训练和测试基于深度学习的信道预测模型。

## 🗂️ HDF5文件结构

### 文件格式规范

所有数据文件采用HDF5格式（`.h5`扩展名），具有以下标准化的层次结构：

```
dataset.h5
├── simulation_config/          # 仿真配置组
│   ├── @center_frequency      # 中心频率 (Hz)
│   ├── @num_bs_antennas       # 基站天线数量
│   ├── @num_ue_antennas       # 用户设备天线数量
│   ├── @num_subcarriers       # OFDM子载波数量
│   └── @subcarrier_spacing    # 子载波间隔 (Hz)
├── positions/                  # 位置信息组
│   ├── bs_position            # 基站位置 [x, y, z] (米)
│   └── ue_positions           # 用户设备位置 [N, 3] (米)
├── channel_data/              # 信道数据组
│   ├── channel_responses      # 信道响应矩阵 [N, S, U, B]
│   ├── path_losses           # 路径损耗 [N, S] (dB)
│   └── delays                # 传播延迟 [N, S] (秒)
└── metadata/                  # 元数据组
    ├── @simulation_date      # 仿真日期
    ├── @generator_version    # 生成器版本
    └── @fixed_spatial_phase  # 空间相位修正标志
```

### 数据维度说明

- **N**: 用户设备位置数量
- **S**: OFDM子载波数量  
- **U**: 用户设备天线数量
- **B**: 基站天线数量

## 📊 Sionna数据集参考

### 基本参数

以`P300.h5`数据集为例，展示标准的数据集参数配置：

| 参数 | 值 | 说明 |
|------|----|----- |
| **位置数量** | 300 | UE位置采样点数量 |
| **中心频率** | 3.5 GHz | 5G NR频段 |
| **子载波数量** | 408 | OFDM子载波数 |
| **子载波间隔** | 30 kHz | 5G NR标准间隔 |
| **基站天线** | 64 | 8×8天线阵列 |
| **UE天线** | 4 | 多天线配置 |

### 数据集详细结构

#### 1. 仿真配置 (`simulation_config/`)

```python
# 属性 (Attributes)
center_frequency: 3500000000.0    # 3.5 GHz
num_bs_antennas: 64              # 基站天线数
num_ue_antennas: 4               # UE天线数  
num_subcarriers: 408             # 子载波数
subcarrier_spacing: 30000.0      # 30 kHz
```

#### 2. 位置信息 (`positions/`)

```python
# 基站位置 (固定)
bs_position: [0.0, 0.0, 25.0]   # [x, y, z] 米
                                 # 位于原点，高度25米

# UE位置 (随机分布)
ue_positions: shape=(300, 3)     # [N, 3] 米
# 示例位置: [-62.73, -224.16, 1.34]
# X范围: [-247.47, 245.03] 米
# Y范围: [-244.58, 249.86] 米  
# Z范围: [1.01, 2.99] 米 (地面高度)
```

#### 3. 信道数据 (`channel_data/`)

```python
# 信道响应矩阵 (复数)
channel_responses: shape=(300, 408, 4, 64)
dtype: complex128
# [位置, 子载波, UE天线, BS天线]
# 数值范围: 实部[-6.79e-4, 7.27e-4], 虚部[-7.16e-4, 6.92e-4]

# 路径损耗
path_losses: shape=(300, 408)
dtype: float64
# [位置, 子载波] (dB)

# 传播延迟  
delays: shape=(300, 408)
dtype: float64
# [位置, 子载波] (秒)
```

#### 4. 元数据 (`metadata/`)

```python
# 属性 (Attributes)
simulation_date: "2025-09-03T10:54:16"  # ISO格式时间戳
generator_version: "1.0.0"              # 生成器版本
fixed_spatial_phase: True               # 空间相位修正标志
```

## 🔧 数据生成

### 生成命令

```bash
# 生成300个位置的数据集
python data/sionna/generator.py --n 300

# 指定输出路径
python data/sionna/generator.py --num_positions 300 --output_path custom_dataset
```

### 配置参数

生成器支持以下主要配置参数：

```python
DEFAULT_CONFIG = {
    # 基本参数
    'num_positions': 300,           # 位置数量
    'area_size': 500,              # 仿真区域大小 (米)
    'bs_height': 25.0,             # 基站高度 (米)
    'ue_height_min': 1.0,          # UE最小高度 (米)
    'ue_height_max': 3.0,          # UE最大高度 (米)
    
    # 5G OFDM参数
    'center_frequency': 3.5e9,      # 中心频率 (Hz)
    'subcarrier_spacing': 30e3,     # 子载波间隔 (Hz)
    'num_subcarriers': 408,         # 子载波数量
    'num_ue_antennas': 4,          # UE天线数
    'num_bs_antennas': 64,         # BS天线数
    
    # 信道模型参数
    'shadowing_std': 8.0,          # 阴影衰落标准差 (dB)
    'multipath_delay_spread': 50e-9, # 多径延迟扩展 (秒)
}
```

## 📖 数据加载

### Python加载示例

```python
import h5py
import numpy as np

# 加载数据集
with h5py.File('data/sionna/P300/P300.h5', 'r') as f:
    # 读取信道响应
    csi_data = f['channel_data/channel_responses'][:]  # (300, 408, 4, 64)
    
    # 读取位置信息
    ue_positions = f['positions/ue_positions'][:]      # (300, 3)
    bs_position = f['positions/bs_position'][:]        # (3,)
    
    # 读取配置参数
    center_freq = f['simulation_config'].attrs['center_frequency']
    num_antennas = f['simulation_config'].attrs['num_bs_antennas']
    
    # 读取元数据
    sim_date = f['metadata'].attrs['simulation_date']
    is_fixed = f['metadata'].attrs.get('fixed_spatial_phase', False)

print(f"数据集形状: {csi_data.shape}")
print(f"中心频率: {center_freq/1e9:.1f} GHz")
print(f"基站天线数: {num_antennas}")
print(f"空间相位已修正: {is_fixed}")
```

### 数据预处理

```python
# CSI数据预处理
def preprocess_csi(csi_data):
    """预处理CSI数据"""
    # 计算幅度和相位
    amplitude = np.abs(csi_data)
    phase = np.angle(csi_data)
    
    # 归一化
    amplitude_norm = amplitude / np.max(amplitude)
    
    return amplitude_norm, phase

# 位置数据预处理  
def preprocess_positions(ue_positions, bs_position):
    """预处理位置数据"""
    # 计算相对位置
    relative_positions = ue_positions - bs_position
    
    # 计算距离
    distances = np.linalg.norm(relative_positions, axis=1)
    
    return relative_positions, distances
```

## 🎯 数据质量验证

### 基本检查

```python
def validate_dataset(file_path):
    """验证数据集完整性"""
    with h5py.File(file_path, 'r') as f:
        # 检查必需的组
        required_groups = ['simulation_config', 'positions', 'channel_data', 'metadata']
        for group in required_groups:
            assert group in f, f"缺少必需组: {group}"
        
        # 检查数据维度一致性
        csi_shape = f['channel_data/channel_responses'].shape
        pos_shape = f['positions/ue_positions'].shape
        
        assert csi_shape[0] == pos_shape[0], "位置数量与CSI数据不匹配"
        
        # 检查复数数据
        csi_data = f['channel_data/channel_responses']
        assert np.iscomplexobj(csi_data), "CSI数据必须为复数类型"
        
        print("✅ 数据集验证通过")

# 使用示例
validate_dataset('data/sionna/P300/P300.h5')
```

### 空间相位验证

```python
def check_spatial_correlation(csi_data, ue_positions):
    """检查空间相关性"""
    # 计算第一个子载波的空间谱
    h_first = csi_data[:, 0, 0, :]  # [N, B]
    
    # 计算相位差异
    phases = np.angle(h_first)
    phase_std = np.std(phases, axis=1)
    
    print(f"相位标准差范围: [{np.min(phase_std):.3f}, {np.max(phase_std):.3f}]")
    
    # 检查是否存在空间相关性
    if np.mean(phase_std) < 0.1:
        print("⚠️  警告: 空间相位相关性较低，可能需要修正")
    else:
        print("✅ 空间相位相关性正常")
```

## 📁 文件命名规范

### 标准命名格式

```
P{N}.h5          # 标准数据集，N为位置数量
P{N}_fixed.h5    # 空间相位修正版本
P{N}_backup.h5   # 备份文件
```

### 示例文件

```
data/sionna/P300/
├── P300.h5                 # 主数据文件
├── P300_fixed.h5          # 相位修正版本  
├── P300_original_backup.h5 # 原始备份
└── P300.png               # 可视化图表
```

## 🚀 最佳实践

### 1. 数据生成建议

- **位置数量**: 建议300-1000个位置用于训练
- **区域大小**: 500米×500米适合城市场景
- **天线配置**: 64天线(8×8)提供良好的空间分辨率
- **频率设置**: 3.5GHz符合5G标准

### 2. 存储优化

- 使用HDF5压缩减少文件大小
- 合理选择数据类型（complex128 vs complex64）
- 分批加载大数据集避免内存溢出

### 3. 版本控制

- 保留原始数据备份
- 记录数据修正历史
- 使用元数据标记数据版本

## 🔍 故障排除

### 常见问题

1. **文件损坏**: 使用`h5py.is_hdf5()`检查文件完整性
2. **维度不匹配**: 验证数据生成参数一致性
3. **内存不足**: 使用数据分块加载
4. **相位问题**: 检查空间相位修正标志

### 调试工具

```python
# HDF5文件结构查看器
def inspect_h5_file(file_path):
    """检查HDF5文件结构"""
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            indent = '  ' * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f'{indent}{name}/ (Group)')
            elif isinstance(obj, h5py.Dataset):
                print(f'{indent}{name}: {obj.shape}, {obj.dtype}')
        
        f.visititems(print_structure)
```

---

**注意**: 本文档基于Sionna数据生成器v1.0.0，如有更新请及时同步文档内容。
