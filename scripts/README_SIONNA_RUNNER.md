# Sionna Runner - 5G OFDM Training Pipeline

`scripts/sionna_runner.py` 是一个完整的训练和评估运行器，专门用于使用Sionna生成的5G OFDM数据进行Prism模型训练。

## 功能特性

### 🚀 **训练模式 (Train Mode)**
- 完整的训练循环
- 自动验证
- 学习率调度
- 梯度裁剪
- 检查点保存
- 训练可视化

### 🧪 **测试模式 (Test Mode)**
- 加载训练好的模型
- 在测试集上评估
- 计算MSE、MAE、RMSE指标
- 保存评估结果

### 🎯 **演示模式 (Demo Mode)**
- 快速验证模型和数据加载
- 运行前向传播
- 检查输出形状

## 使用方法

### 基本用法

```bash
# 训练模式
python scripts/sionna_runner.py --mode train --epochs 100 --batch_size 32

# 测试模式
python scripts/sionna_runner.py --mode test --checkpoint checkpoints/sionna_5g/best_model.pth

# 演示模式
python scripts/sionna_runner.py --mode demo
```

### 完整参数

```bash
python scripts/sionna_runner.py \
    --mode train \
    --config configs/ofdm-5g-sionna.yml \
    --epochs 200 \
    --batch_size 64 \
    --device cuda \
    --save_dir checkpoints/sionna_5g \
    --results_dir results/sionna_5g
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | `train` | 运行模式: `train`, `test`, `demo` |
| `--config` | str | `configs/ofdm-5g-sionna.yml` | 配置文件路径 |
| `--checkpoint` | str | `None` | 测试模式的检查点文件路径 |
| `--epochs` | int | `100` | 训练轮数 |
| `--batch_size` | int | `32` | 批次大小 |
| `--device` | str | `cuda` | 设备类型 (`cuda`/`cpu`) |
| `--save_dir` | str | `checkpoints/sionna_5g` | 检查点保存目录 |
| `--results_dir` | str | `results/sionna_5g` | 结果保存目录 |

## 训练流程

### 1. 数据加载
- 自动加载Sionna HDF5数据
- 数据预处理和归一化
- 训练/验证/测试集分割

### 2. 模型初始化
- 创建Prism模型
- 配置损失函数 (PrismLoss)
- 初始化优化器 (Adam)
- 设置学习率调度器

### 3. 训练循环
```python
for epoch in range(num_epochs):
    # 训练一个epoch
    train_loss = trainer.train_epoch(batch_size)
    
    # 验证
    val_loss = trainer.validate(batch_size)
    
    # 学习率调度
    scheduler.step()
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        save_checkpoint('best_model.pth')
```

### 4. 损失计算
使用 `PrismLoss` 类计算频率感知损失：

```python
# 主要损失: 子载波响应预测
loss = criterion(predictions, targets, config=config)

# 额外损失组件 (如果启用):
# - CSI虚拟链路损失 (权重: 0.3)
# - 射线追踪损失 (权重: 0.2)  
# - 空间一致性损失 (权重: 0.1)
```

## Loss函数详解

### PrismLoss 架构

```python
class PrismLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        # 支持 MSE 和 L1 损失
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
    
    def forward(self, predictions, targets, weights=None, 
                csi_targets=None, ray_tracing_targets=None, config=None):
        # 1. 计算每个子载波的独立损失
        per_subcarrier_loss = self.criterion(predictions, targets)
        
        # 2. 应用可选的子载波权重
        if weights is not None:
            per_subcarrier_loss *= weights.unsqueeze(0)
        
        # 3. 汇总所有子载波损失
        total_loss = torch.sum(per_subcarrier_loss)
        
        # 4. 添加CSI损失 (如果启用)
        if csi_targets is not None and config:
            csi_weight = config['loss'].get('csi_loss_weight', 0)
            if csi_weight > 0:
                csi_loss = self._compute_csi_loss(predictions, csi_targets)
                total_loss += csi_weight * csi_loss
        
        # 5. 添加射线追踪损失 (如果启用)
        if ray_tracing_targets is not None and config:
            ray_weight = config['loss'].get('ray_tracing_loss_weight', 0)
            if ray_weight > 0:
                ray_loss = self._compute_ray_tracing_loss(predictions, ray_tracing_targets)
                total_loss += ray_weight * ray_loss
        
        return total_loss
```

### 损失组件权重

配置文件中的损失权重设置：

```yaml
loss:
  loss_type: 'mse'                    # 主要损失类型
  frequency_weighting: true           # 启用频率相关权重
  low_freq_weight: 1.0               # 低频子载波权重
  high_freq_weight: 1.2              # 高频子载波权重
  
  # 高级损失组件
  csi_loss_weight: 0.3               # CSI虚拟链路损失权重
  ray_tracing_loss_weight: 0.2       # 射线追踪损失权重
  spatial_consistency_weight: 0.1    # 空间一致性损失权重
  
  # 损失平衡
  enable_loss_balancing: true        # 启用损失平衡
  adaptive_weight_adjustment: true   # 自适应权重调整
```

## 输出文件

### 训练输出
- `checkpoints/sionna_5g/best_model.pth` - 最佳模型
- `checkpoints/sionna_5g/final_model.pth` - 最终模型
- `checkpoints/sionna_5g/checkpoint_epoch_N.pth` - 定期检查点

### 结果输出
- `results/sionna_5g/training_results.png` - 训练可视化
- `results/sionna_5g/test_results.pt` - 测试结果

## 使用示例

### 1. 开始训练
```bash
# 训练100个epoch，批次大小32
python scripts/sionna_runner.py --mode train --epochs 100 --batch_size 32
```

### 2. 继续训练
```bash
# 从检查点继续训练
python scripts/sionna_runner.py --mode train --epochs 200 --batch_size 64
```

### 3. 测试模型
```bash
# 测试最佳模型
python scripts/sionna_runner.py --mode test --checkpoint checkpoints/sionna_5g/best_model.pth
```

### 4. 快速演示
```bash
# 验证设置
python scripts/sionna_runner.py --mode demo
```

## 故障排除

### 常见问题

**1. CUDA内存不足**
```bash
# 减少批次大小
python scripts/sionna_runner.py --mode train --batch_size 16
```

**2. 数据文件未找到**
```bash
# 先运行Sionna仿真
cd scripts/simulation
python sionna_simulation.py
```

**3. 配置错误**
```bash
# 检查配置文件
python scripts/test_sionna_integration.py
```

### 性能优化

**1. GPU加速**
```bash
# 确保使用CUDA
python scripts/sionna_runner.py --device cuda
```

**2. 批次大小调优**
```bash
# 根据GPU内存调整批次大小
python scripts/sionna_runner.py --batch_size 64  # 或更大
```

**3. 混合精度训练**
```yaml
# 在配置文件中启用
training:
  enable_mixed_precision: true
```

## 下一步

1. **训练模型**: 使用 `--mode train` 开始训练
2. **监控进度**: 查看训练日志和损失曲线
3. **评估性能**: 使用 `--mode test` 评估模型
4. **分析结果**: 查看生成的可视化图表
5. **调优参数**: 根据结果调整配置参数

## 相关文件

- `configs/ofdm-5g-sionna.yml` - 主配置文件
- `src/prism/utils/sionna_data_loader.py` - 数据加载器
- `src/prism/model.py` - Prism模型和损失函数
- `scripts/test_sionna_integration.py` - 集成测试
- `docs/SIONNA_INTEGRATION.md` - 详细集成文档
