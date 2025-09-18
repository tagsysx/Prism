# simulation_data.h5 说明文档

## 概述
`simulation_data.h5` 是一个 HDF5 格式文件，用于存储无线通信场景的仿真数据，包括信道响应(channel_responses)、用户设备位置(ue_position)、基站位置(bs_position)以及仿真配置参数。数据以层次结构组织，存储在 HDF5 的组和数据集中，适用于 5G 信道建模或射线追踪仿真等应用。

## 文件生成
该文件由py脚本生成，依赖以下库：
- Python 3.x
- `h5py`（用于操作 HDF5 文件）
- `numpy`（用于处理数组）
- `logging`（用于记录日志）

### 变量格式
- `channel_responses`：包含信道响应数据，预期形状为 `(num_data_size, num_ue_antennas, num_bs_antennas, num_subcarriers)`，（例如300个数据,接收阵列天线数为8，基站天线数为1，子载波数为64，则shape应为300×8×1×64）。
- `ue_position`：包含 UE 位置数据，预期形状为 `(num_positions, 3)`，num_positions表示数据数量，应和channel_responses的num_data_size相等且顺序对应，后一维度包含x、y、z坐标。
- `bs_position`：包含基站的x，y，z坐标，预期形状为`(3,)`

## 文件结构
HDF5 文件包含两个主要组：
- **`/antenna`**：存储配置参数（以属性形式）。
- **`/data`**：存储仿真数据（以数据集形式）。

### 详细结构
- **组：`/antenna`**
  - 属性：包含脚本中 `DEFAULT_CONFIG` 的所有配置参数：
    - `num_positions`：UE 位置数量（例如 300）。
    - `output_path`：仿真输出路径（例如 "../sionna/simulation"）。
    - `area_size`：仿真区域大小（例如 500）。
    - `bs_height`：基站高度（例如 25.0）。
    - `ue_height_min`：UE 最小高度（例如 1.0）。
    - `ue_height_max`：UE 最大高度（例如 3.0）。
    - `center_frequency`：中心频率，单位 Hz（例如 2.4e9，即 2.4 GHz）。
    - `subcarrier_spacing`：子载波间隔，单位 Hz（例如 312.5e3，即 312.5 kHz）。
    - `num_subcarriers`：子载波数量（例如 64）。
    - `num_ue_antennas`：UE 天线数量（例如 8）。
    - `num_bs_antennas`：BS 天线数量（例如 1）。
    - `bs_antenna_pattern`：基站天线配置（例如 "1_dual_pol"）。
    - `bs_antenna_rows`：基站天线行数（例如 1）。
    - `bs_antenna_cols`：基站天线列数（例如 1）。
    - `max_depth`：射线追踪最大深度（例如 5）。
    - `num_samples`：射线追踪采样数（例如 100000）。
    - `random_seed`：随机种子，确保可重复性（例如 42）。
    - `create_plots`：是否生成绘图（例如 true）。
    - `plot_dpi`：绘图分辨率（例如 300）。
    - `preview_scene`：是否生成场景预览图像（例如 true）。

- **组：`/data`**
  - 数据集：
    - `bs_position`：基站位置，1D NumPy 数组，形状 `(3,)`，包含 x、y、z 坐标（例如 [250.0, 250.0, 25.0]）。
    - `ue_position`：UE位置，2D NumPy 数组，形状 `(num_positions, 3)`，包含每个 UE 的 x、y、z 坐标。
    - `channel_responses`：信道响应数据，4D NumPy 数组，形状 `(num_positions, num_ue_antennas, num_bs_antennas, num_subcarriers)`（例如 300×8×1×64），数据类型为 complex128。

## 访问数据
使用 Python 的 `h5py` 库可以读取 `simulation_data.h5` 文件。以下是一个示例脚本，用于加载和检查数据：

```python
import h5py
import numpy as np

# 打开 HDF5 文件
with h5py.File('simulation_data.h5', 'r') as f:
    # 访问配置
    config = dict(f['antenna'].attrs)
    print("配置参数：", config)

    # 访问数据集
    bs_position = f['data/bs_position'][:]
    ue_position = f['data/ue_position'][:]
    channel_responses = f['data/channel_responses'][:]

    # 打印形状和样本数据
    print("基站位置：", bs_position)
    print("UE 位置形状：", ue_position.shape)
    print("信道响应形状：", channel_responses.shape)
    print("样本信道响应（第一个位置、第一个 UE 天线、第一个 BS 天线、第一个子载波）：", 
          channel_responses[0, 0, 0, 0])
```

### 预期输出
假设 `num_positions=300`, `num_ue_antennas=8`, `num_bs_antennas=1`, `num_subcarriers=64`，输出可能如下：
```
配置参数：{'num_positions': 300, 'output_path': '../sionna/simulation', ...}
基站位置：[250. 250.  25.]
UE 位置形状：(300, 3)
信道响应形状：(300, 8, 1, 64)
样本信道响应（第一个位置、第一个 UE 天线、第一个 BS 天线、第一个子载波）：(1.23+4.56j)
```

## 注意事项
- **依赖库**：确保安装 `h5py` 和 `numpy`（`pip install h5py numpy`）。
- **未使用数据**：脚本包含 `path_losses` 和 `delays` 的占位符，但当前未保存（设为 `None`）。
- **自定义配置**：如需修改配置参数，编辑脚本 `DEFAULT_CONFIG` 字典。
- **错误日志**：脚本使用 `logging` 模块记录消息，运行时检查控制台输出的错误信息。
