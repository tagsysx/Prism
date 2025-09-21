# chrissy数据H5文件结构说明


## chrissy原始数据格式

### 输入文件
- **data.npy**: `(20, 1, 64, 4, 408, 1)`
  - 20个测量位置
  - 1个基站
  - 64个基站天线
  - 4个UE天线（使用所有4个）
  - 408个子载波
  - 1个UE设备
- **pos.npy**: `(20, 3)`
  - 20个位置的XYZ坐标

### 数据规格
- **中心频率**: 3.7GHz
- **带宽**: 100MHz
- **子载波间隔**: 240kHz
- **基站天线配置**: 64x1
- **UE天线配置**: 4x1

## H5文件结构

```
chrissy_data.h5/
├── metadata/                    # 元数据组
│   ├── description             # "chrissy CSI data"
│   ├── file_type               # "chrissy CSI"
│   ├── version                 # "1.0"
│   └── config/                 # 配置参数组
│       ├── bandwidth           # 100000000 (100MHz)
│       ├── bs_antenna_configuration  # "64x1"
│       ├── bs_positions_description  # "BS position coordinates (x, y, z) for each sample"
│       ├── bs_positions_dimensions   # "(position, coordinates)"
│       ├── bs_positions_shape        # "(20, 3)"
│       ├── center_frequency          # 3700000000 (3.7GHz)
│       ├── csi_description           # "Channel State Information (CSI) data from chrissy dataset"
│       ├── csi_dimensions            # "(position, bs_antenna_index, ue_antenna_index, subcarrier_index)"
│       ├── csi_shape                 # "(20, 64, 4, 408)"
│       ├── num_samples               # 20
│       ├── num_subcarriers           # 408
│       ├── subcarrier_spacing        # 240000 (240kHz)
│       ├── ue_antenna_configuration  # "4x1"
│       ├── ue_positions_description  # "UE position coordinates (x, y, z)"
│       ├── ue_positions_dimensions   # "(position, coordinates)"
│       ├── ue_positions_shape        # "(20, 3)"
│       ├── data_source               # "chrissy WiFi CSI dataset"
│       ├── original_data_format      # "(20, 1, 64, 4, 408, 1)"
│       └── processing_note           # "All 4 UE antennas used, dimensions reordered to (position, bs_antenna, ue_antenna, subcarrier)"
└── data/                        # 数据组
    ├── bs_positions             # (20, 3) - 基站位置坐标
    ├── csi                      # (20, 64, 4, 408) - 信道状态信息
    └── ue_positions             # (20, 3) - UE位置坐标
```

## 数据维度详解

### 1. bs_positions (基站位置)
- **形状**: `(20, 3)`
- **数据类型**: `float64`
- **描述**: 每个样本对应的基站位置坐标
- **内容**: 所有20个样本使用相同的基站坐标 `[x, y, z]`

### 2. csi (信道状态信息)
- **形状**: `(20, 64, 4, 408)`
- **数据类型**: `complex128`
- **描述**: 信道状态信息数据，维度顺序为(position, bs_antenna, ue_antenna, subcarrier)
- **维度含义**:
  - 第1维 (20): 位置索引
  - 第2维 (64): 基站天线索引
  - 第3维 (4): UE天线索引（使用所有4个）
  - 第4维 (408): 子载波索引

### 3. ue_positions (UE位置)
- **形状**: `(20, 3)`
- **数据类型**: `float64`
- **描述**: 每个测量位置的UE坐标
- **内容**: 20个不同位置的XYZ坐标

## 配置参数说明

| 参数 | 值 | 说明 |
|------|----|----- |
| bandwidth | 100000000 | 带宽100MHz |
| center_frequency | 3700000000 | 中心频率3.7GHz |
| subcarrier_spacing | 240000 | 子载波间隔240kHz |
| bs_antenna_configuration | "64x1" | 64个基站天线 |
| ue_antenna_configuration | "4x1" | 使用所有4个UE天线 |
| num_samples | 20 | 样本数量 |
| num_subcarriers | 408 | 子载波数量 |
