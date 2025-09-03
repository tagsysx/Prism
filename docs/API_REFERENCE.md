# Prism API 参考文档

## 概述

本文档提供Prism系统所有主要API的详细参考，包括类、函数、参数说明和使用示例。

## 目录

1. [训练接口 (Training Interface)](#训练接口)
2. [网络组件 (Network Components)](#网络组件)
3. [损失函数 (Loss Functions)](#损失函数)
4. [射线追踪 (Ray Tracing)](#射线追踪)
5. [数据工具 (Data Utilities)](#数据工具)
6. [配置系统 (Configuration)](#配置系统)

---

## 训练接口

### PrismTrainingInterface

主要的训练接口类，提供完整的训练管道。

```python
class PrismTrainingInterface(nn.Module):
    """
    Prism训练接口，集成所有训练组件
    """
    
    def __init__(self, config: Dict[str, Any])
```

#### 参数
- **config** (`Dict[str, Any]`): 完整的配置字典

#### 主要方法

##### `train(loss_function: nn.Module, epochs: int = None) -> Dict[str, Any]`

执行模型训练。

**参数:**
- `loss_function`: 损失函数实例
- `epochs`: 训练轮数 (可选，使用配置中的值)

**返回:**
- `Dict[str, Any]`: 训练结果和统计信息

**示例:**
```python
from prism.training_interface import PrismTrainingInterface
from prism.loss import LossFunction

# 初始化训练接口
trainer = PrismTrainingInterface(config)

# 创建损失函数
loss_fn = LossFunction(config['training']['loss'])

# 开始训练
results = trainer.train(loss_function=loss_fn, epochs=100)
print(f"Final loss: {results['final_loss']:.6f}")
```

##### `forward(ue_positions: torch.Tensor, selected_subcarriers: torch.Tensor) -> Dict[str, torch.Tensor]`

前向传播预测。

**参数:**
- `ue_positions`: UE位置张量 `(batch_size, 3)`
- `selected_subcarriers`: 选定的子载波索引 `(batch_size, num_selected)`

**返回:**
- `Dict[str, torch.Tensor]`: 预测结果字典

---

## 网络组件

### PrismNetwork

集成的主网络，包含所有子网络组件。

```python
class PrismNetwork(nn.Module):
    """
    Prism主网络，集成所有组件
    """
    
    def __init__(
        self,
        num_subcarriers: int,
        num_ue_antennas: int,
        num_bs_antennas: int,
        feature_dim: int = 128,
        antenna_embedding_dim: int = 64,
        **kwargs
    )
```

#### 参数
- **num_subcarriers** (`int`): 子载波数量
- **num_ue_antennas** (`int`): UE天线数量
- **num_bs_antennas** (`int`): 基站天线数量
- **feature_dim** (`int`, 默认128): 特征维度
- **antenna_embedding_dim** (`int`, 默认64): 天线嵌入维度

### AttenuationNetwork

空间位置编码网络。

```python
class AttenuationNetwork(nn.Module):
    """
    将空间位置编码为紧凑特征表示
    """
    
    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 8,
        use_shortcuts: bool = True,
        complex_output: bool = True
    )
```

#### 参数
- **input_dim** (`int`): 输入维度 (IPE编码后的3D位置)
- **hidden_dim** (`int`): 隐藏层维度
- **output_dim** (`int`): 输出特征维度
- **num_layers** (`int`): 网络层数
- **use_shortcuts** (`bool`): 是否使用跳跃连接
- **complex_output** (`bool`): 是否输出复数值

### AttenuationDecoder

衰减解码器，将特征转换为衰减因子。

```python
class AttenuationDecoder(nn.Module):
    """
    将128D特征转换为N_UE × K衰减因子
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_ue_antennas: int,
        num_subcarriers: int,
        complex_output: bool = True
    )
```

### AntennaEmbeddingCodebook

天线嵌入码本。

```python
class AntennaEmbeddingCodebook(nn.Module):
    """
    为每个基站天线提供可学习的嵌入向量
    """
    
    def __init__(
        self,
        num_bs_antennas: int,
        embedding_dim: int = 64
    )
    
    def forward(self, antenna_indices: torch.Tensor) -> torch.Tensor:
        """
        获取指定天线的嵌入向量
        
        参数:
            antenna_indices: 天线索引 (batch_size,) 或 (batch_size, num_antennas)
            
        返回:
            embeddings: 天线嵌入 (batch_size, embedding_dim) 或 (batch_size, num_antennas, embedding_dim)
        """
```

### AntennaNetwork

天线网络，生成方向重要性指示器。

```python
class AntennaNetwork(nn.Module):
    """
    处理天线嵌入生成方向重要性指示器
    """
    
    def __init__(
        self,
        antenna_embedding_dim: int = 64,
        azimuth_divisions: int = 16,
        elevation_divisions: int = 8,
        hidden_dim: int = 128
    )
    
    def get_top_k_directions(
        self, 
        directional_importance: torch.Tensor, 
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取前K个重要方向
        
        参数:
            directional_importance: 方向重要性矩阵 (batch_size, azimuth, elevation)
            k: 选择的方向数量
            
        返回:
            top_k_indices: 前K个方向的索引 (batch_size, k)
            top_k_importance: 前K个方向的重要性值 (batch_size, k)
        """
```

### RadianceNetwork

辐射网络，处理辐射特性。

```python
class RadianceNetwork(nn.Module):
    """
    处理UE位置、观察方向和空间特征
    """
    
    def __init__(
        self,
        ue_pos_dim: int = 63,
        view_dir_dim: int = 27,
        feature_dim: int = 128,
        antenna_embedding_dim: int = 64,
        num_ue_antennas: int = 4,
        num_subcarriers: int = 64,
        hidden_dim: int = 256,
        complex_output: bool = True
    )
```

---

## 损失函数

### LossFunction

主损失函数，组合多个子损失。

```python
class LossFunction(nn.Module):
    """
    Prism主损失函数，组合CSI、PDP和空间频谱损失
    """
    
    def __init__(self, config: Dict[str, Any])
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算组合损失
        
        参数:
            predictions: 预测值字典
                - 'csi': 预测的CSI张量
                - 'traced_csi': 追踪的CSI张量 (可选)
            targets: 目标值字典
                - 'csi': 目标CSI张量
                - 'traced_csi': 目标追踪CSI张量 (可选)
            masks: 掩码字典 (可选)
            
        返回:
            total_loss: 总损失值
            loss_components: 各组件损失值字典
        """
```

### CSILoss

CSI损失函数。

```python
class CSILoss(nn.Module):
    """
    CSI (信道状态信息) 损失函数
    """
    
    def __init__(
        self, 
        loss_type: str = 'mse', 
        phase_weight: float = 1.0, 
        magnitude_weight: float = 1.0, 
        cmse_weight: float = 1.0
    )
```

#### 支持的损失类型
- `'mse'`: 标准复数MSE损失
- `'mae'`: 复数平均绝对误差
- `'complex_mse'`: 分离实部虚部的MSE
- `'magnitude_phase'`: 分离幅度相位损失
- `'hybrid'`: 混合损失 (推荐)

### PDPLoss

功率延迟分布损失函数。

```python
class PDPLoss(nn.Module):
    """
    功率延迟分布 (PDP) 损失函数
    """
    
    def __init__(
        self, 
        loss_type: str = 'hybrid', 
        fft_size: int = 1024,
        normalize_pdp: bool = True
    )
```

#### 支持的损失类型
- `'mse'`: PDP MSE损失
- `'delay'`: 主径延迟损失
- `'hybrid'`: 混合损失 (推荐)

### SSLoss

空间频谱损失函数。

```python
class SSLoss(nn.Module):
    """
    空间频谱损失函数，基于Bartlett波束形成器
    """
    
    def __init__(self, config: Dict[str, Any])
    
    def compute_and_visualize_loss(
        self, 
        predicted_csi: torch.Tensor, 
        target_csi: torch.Tensor,
        save_path: str, 
        sample_idx: int = 0
    ) -> Tuple[float, str]:
        """
        计算损失并生成可视化
        
        参数:
            predicted_csi: 预测CSI
            target_csi: 目标CSI
            save_path: 保存路径
            sample_idx: 样本索引
            
        返回:
            loss_value: 损失值
            plot_path: 图表保存路径
        """
```

---

## 射线追踪

### RayTracerBase

射线追踪基类。

```python
class RayTracerBase(nn.Module):
    """
    射线追踪基础类
    """
    
    def __init__(self, config: Dict[str, Any])
    
    def trace_rays(
        self,
        bs_position: torch.Tensor,
        ue_positions: torch.Tensor,
        selected_subcarriers: torch.Tensor,
        antenna_indices: torch.Tensor
    ) -> Dict[Tuple[Tuple[float, float, float], int], torch.Tensor]:
        """
        执行射线追踪
        
        参数:
            bs_position: 基站位置 (3,)
            ue_positions: UE位置 (batch_size, 3)
            selected_subcarriers: 选定子载波 (batch_size, num_selected)
            antenna_indices: 天线索引 (num_antennas,)
            
        返回:
            ray_results: 射线追踪结果字典
        """
```

### RayTracerCUDA

CUDA加速射线追踪。

```python
class RayTracerCUDA(RayTracerBase):
    """
    CUDA加速射线追踪实现
    """
    
    def __init__(self, config: Dict[str, Any])
```

### RayTracerCPU

CPU射线追踪实现。

```python
class RayTracerCPU(RayTracerBase):
    """
    CPU射线追踪实现
    """
    
    def __init__(self, config: Dict[str, Any])
```

---

## 数据工具

### load_and_split_data

数据加载和分割函数。

```python
def load_and_split_data(
    dataset_path: str,
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    mode: str = 'train',
    target_antenna_index: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    从HDF5文件加载数据并分割为训练/测试集
    
    参数:
        dataset_path: HDF5数据集文件路径
        train_ratio: 训练集比例 (0.0 到 1.0)
        test_ratio: 测试集比例 (0.0 到 1.0)
        random_seed: 随机种子
        mode: 'train' 或 'test' - 返回哪个分割
        target_antenna_index: 目标UE天线索引 (0-based)
        
    返回:
        ue_positions: UE位置 (samples, 3)
        csi_data: CSI数据 (samples, subcarriers, 1, bs_antennas)
        bs_position: 基站位置 (3,)
        antenna_indices: 天线索引 (num_antennas,)
        metadata: 元数据字典
    """
```

---

## 配置系统

### ConfigLoader

配置加载器。

```python
class ConfigLoader:
    """
    配置文件加载器，支持模板变量和动态路径解析
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        参数:
            config_path: 配置文件路径
            
        返回:
            config: 解析后的配置字典
        """
    
    @staticmethod
    def resolve_template_variables(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析配置中的模板变量
        
        参数:
            config: 原始配置字典
            
        返回:
            resolved_config: 解析后的配置字典
        """
```

---

## 使用示例

### 完整训练流程

```python
import yaml
import torch
from prism.training_interface import PrismTrainingInterface
from prism.loss import LossFunction
from prism.data_utils import load_and_split_data

# 1. 加载配置
with open('configs/sionna.yml', 'r') as f:
    config = yaml.safe_load(f)

# 2. 加载数据
ue_positions, csi_data, bs_position, antenna_indices, metadata = load_and_split_data(
    dataset_path='data/sionna/P300/P300.h5',
    mode='train',
    target_antenna_index=0
)

# 3. 初始化训练接口
trainer = PrismTrainingInterface(config)

# 4. 创建损失函数
loss_config = {
    'csi_weight': 0.7,
    'pdp_weight': 0.3,
    'spatial_spectrum_weight': 0.1,
    'csi_loss': {'type': 'hybrid'},
    'pdp_loss': {'type': 'hybrid'},
    'spatial_spectrum_loss': {
        'enabled': True,
        'algorithm': 'bartlett',
        'fusion_method': 'average',
        'theta_range': [0, 5, 90],
        'phi_range': [0, 10, 360]
    }
}
loss_fn = LossFunction(loss_config)

# 5. 开始训练
results = trainer.train(loss_function=loss_fn, epochs=100)

# 6. 保存模型
torch.save(trainer.state_dict(), 'models/trained_model.pt')
```

### 模型推理

```python
# 加载训练好的模型
trainer.load_state_dict(torch.load('models/trained_model.pt'))
trainer.eval()

# 进行预测
with torch.no_grad():
    predictions = trainer.forward(
        ue_positions=test_ue_positions,
        selected_subcarriers=test_subcarriers
    )
    
    predicted_csi = predictions['csi']
    print(f"Predicted CSI shape: {predicted_csi.shape}")
```

### 损失函数可视化

```python
# 计算并可视化空间频谱损失
if hasattr(loss_fn, 'spatial_spectrum_loss') and loss_fn.spatial_spectrum_loss is not None:
    loss_value, plot_path = loss_fn.compute_and_visualize_spatial_spectrum_loss(
        predicted_csi=predicted_csi,
        target_csi=target_csi,
        save_path="./results/spatial_spectrum/",
        sample_idx=0
    )
    print(f"Spatial spectrum loss: {loss_value:.6f}")
    print(f"Visualization saved to: {plot_path}")
```

---

## 错误处理

### 常见异常

#### `ValueError`
- 配置参数缺失或无效
- 数据维度不匹配
- 不支持的损失类型

#### `RuntimeError`
- CUDA内存不足
- 网络前向传播失败
- 射线追踪计算错误

#### `FileNotFoundError`
- 配置文件或数据文件不存在
- 模型检查点文件缺失

### 异常处理示例

```python
try:
    trainer = PrismTrainingInterface(config)
    results = trainer.train(loss_function=loss_fn)
except ValueError as e:
    print(f"配置错误: {e}")
except RuntimeError as e:
    print(f"运行时错误: {e}")
    # 可能需要减少批大小或使用CPU
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
```

---

## 性能优化建议

### 内存优化
- 使用适当的批大小 (通常8-32)
- 启用梯度检查点 (`gradient_checkpointing=True`)
- 使用混合精度训练 (`use_amp=True`)

### 计算优化
- 优先使用CUDA射线追踪
- 合理设置空间采样密度
- 使用预计算的天线位置

### 配置优化
```yaml
# 推荐的性能配置
training:
  batch_size: 16
  gradient_checkpointing: true
  use_amp: true
  
ray_tracing:
  implementation: 'cuda'  # 或 'cpu' 如果没有GPU
  spatial_sampling:
    num_samples_per_ray: 64
    
system:
  device: 'auto'  # 自动选择最佳设备
```

---

*文档版本: v1.0*  
*最后更新: 2025年1月*  
*维护者: Prism项目团队*
