# Top-K Directions Configuration

## 概述

`top_k_directions` 参数现在可以从配置文件中读取，替代了之前硬编码的K值计算。这提供了更灵活的方向选择控制。

## 配置参数

### 在 `configs/ofdm-5g-sionna.yml` 中

```yaml
ray_tracing:
  # ... 其他参数 ...
  
  # MLP-based direction selection
  top_k_directions: 32            # Number of top-K directions to select (configurable)
  # 代码实现：DiscreteRayTracer使用此参数，替代硬编码的K值
```

## 工作原理

### 1. 配置优先级

- **如果设置了 `top_k_directions`**：使用配置的值
- **如果没有设置**：使用默认公式 `min(32, total_directions // 4)`

### 2. 默认公式示例

| 配置 | 方向总数 | 默认K值 | 采样率 |
|------|----------|---------|--------|
| 8×4 = 32 | 32 | min(32, 32÷4) = **8** | 25% |
| 16×8 = 128 | 128 | min(32, 128÷4) = **32** | 25% |
| 18×9 = 162 | 162 | min(32, 162÷4) = **32** | 19.8% |
| 36×18 = 648 | 648 | min(32, 648÷4) = **32** | 4.9% |

### 3. 自定义配置示例

```yaml
# 高精度训练
ray_tracing:
  azimuth_divisions: 36
  elevation_divisions: 18
  top_k_directions: 64    # 选择64个方向，提高精度

# 快速训练
ray_tracing:
  azimuth_divisions: 18
  elevation_divisions: 9
  top_k_directions: 16    # 选择16个方向，加快速度
```

## 代码实现

### DiscreteRayTracer 构造函数

```python
def __init__(self, 
             azimuth_divisions: int = 36,
             elevation_divisions: int = 18,
             # ... 其他参数 ...
             top_k_directions: int = None):
    """
    Args:
        # ... 其他参数 ...
        top_k_directions: Number of top-K directions to select for MLP-based sampling 
                         (if None, uses default formula)
    """
    # Set top-K directions for MLP-based sampling
    if top_k_directions is not None:
        self.top_k_directions = top_k_directions
        logger.info(f"Using configured top-K directions: {self.top_k_directions}")
    else:
        # Default formula: min(32, total_directions // 4)
        self.top_k_directions = min(32, (azimuth_divisions * elevation_divisions) // 4)
        logger.info(f"Using default top-K formula: min(32, {azimuth_divisions * elevation_divisions} // 4) = {self.top_k_directions}")
```

### 训练脚本中的使用

```python
# 在 scripts/simulation/train_prism.py 中
self.ray_tracer = DiscreteRayTracer(
    azimuth_divisions=rt_config['azimuth_divisions'],
    elevation_divisions=rt_config['elevation_divisions'],
    # ... 其他参数 ...
    top_k_directions=rt_config.get('top_k_directions', None)  # 使用配置的K值
)
```

## 性能影响

### 精度 vs 速度权衡

- **更高的K值**：
  - ✅ 更高的射线追踪精度
  - ❌ 更慢的训练速度
  
- **更低的K值**：
  - ✅ 更快的训练速度
  - ❌ 可能降低射线追踪精度

### 推荐配置

| 训练阶段 | 推荐K值 | 说明 |
|----------|---------|------|
| 初始训练 | 16-32 | 快速收敛，建立基础模型 |
| 中期训练 | 32-48 | 平衡精度和速度 |
| 最终训练 | 48-64 | 高精度，用于最终优化 |

## 课程学习集成

配置文件中的 `curriculum_learning` 部分已经包含了不同阶段的 `top_k_directions` 设置：

```yaml
curriculum_learning:
  phases:
    - phase: 0
      top_k_directions: 16    # 初始阶段：16个方向
    - phase: 1  
      top_k_directions: 32    # 中期阶段：32个方向
    - phase: 2
      top_k_directions: 64    # 最终阶段：64个方向
```

**注意**：目前课程学习功能需要进一步开发才能自动使用这些配置。

## 验证

运行以下命令验证配置是否正确：

```bash
# 检查配置文件语法
python -c "import yaml; yaml.safe_load(open('configs/ofdm-5g-sionna.yml'))"

# 启动训练（会显示使用的K值）
python scripts/simulation/train_prism.py --config configs/ofdm-5g-sionna.yml --data results/complete_pipeline/data_split/train_data.h5 --output results/complete_pipeline/training
```

## 总结

✅ **配置化K值**：不再硬编码，可通过配置文件调整  
✅ **向后兼容**：未设置时自动使用默认公式  
✅ **灵活控制**：支持不同训练阶段的不同精度需求  
✅ **性能优化**：可根据硬件能力调整精度vs速度平衡  

现在你可以通过修改配置文件中的 `top_k_directions` 参数来精确控制MLP方向选择的数量，而无需修改代码！
