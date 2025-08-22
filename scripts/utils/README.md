# Scripts Utils

这个文件夹包含了Prism项目的实用工具脚本，用于训练监控、检查点管理和后台训练等任务。

## 工具列表

### 1. 后台训练脚本 (`run_training_background.sh`)

使用screen会话在后台运行Prism训练，支持断开连接后重新连接。

**使用方法：**
```bash
# 使用默认参数
./scripts/utils/run_training_background.sh

# 自定义参数
./scripts/utils/run_training_background.sh --config configs/custom.yml --epochs 200 --batch-size 64 --device cuda --session my_training

# 查看帮助
./scripts/utils/run_training_background.sh --help
```

**参数说明：**
- `--config FILE`: 配置文件路径 (默认: configs/ofdm-5g-sionna.yml)
- `--epochs N`: 训练轮数 (默认: 100)
- `--batch-size N`: 批次大小 (默认: 32)
- `--device DEVICE`: 设备类型 (默认: cuda)
- `--session NAME`: Screen会话名称 (默认: prism_training)

**Screen会话管理：**
```bash
# 重新连接到训练会话
screen -r prism_training

# 列出所有会话
screen -ls

# 从会话中分离 (在会话内)
Ctrl+A 然后按 D

# 终止会话
screen -S prism_training -X quit
```

### 2. 训练监控脚本 (`monitor_training.py`)

实时监控训练进程的状态、资源使用情况和训练进度。

**使用方法：**
```bash
# 使用默认监控间隔 (30秒)
python scripts/utils/monitor_training.py

# 自定义监控间隔
python scripts/utils/monitor_training.py --interval 60
```

**监控信息包括：**
- 训练进程状态 (PID, CPU使用率, 内存使用)
- GPU使用情况 (如果nvidia-smi可用)
- 训练进度 (当前epoch, 最佳损失)
- 最新检查点信息
- 测试结果 (如果可用)

### 3. 检查点读取脚本 (`read_checkpoint.py`)

读取并显示PyTorch检查点文件的详细信息。

**使用方法：**
```bash
# 读取检查点文件
python scripts/utils/read_checkpoint.py ../../checkpoints/sionna_5g/best_model.pth

# 读取最新的epoch检查点
python scripts/utils/read_checkpoint.py ../../checkpoints/sionna_5g/checkpoint_epoch_50.pth
```

**显示信息包括：**
- 基本训练信息 (epoch, 损失)
- 训练和验证损失历史
- 模型参数信息
- 优化器和调度器状态
- 训练配置参数

## 典型工作流程

### 1. 启动后台训练
```bash
# 启动训练
./scripts/utils/run_training_background.sh --epochs 100 --batch-size 32

# 确认训练已启动
screen -ls
```

### 2. 监控训练进度
```bash
# 在另一个终端中监控训练
python scripts/utils/monitor_training.py --interval 30
```

### 3. 检查训练状态
```bash
# 重新连接到训练会话查看详细日志
screen -r prism_training

# 或者读取最新检查点
python scripts/utils/read_checkpoint.py ../../checkpoints/sionna_5g/best_model.pth
```

### 4. 训练完成后
```bash
# 查看最终结果
python scripts/utils/read_checkpoint.py ../../checkpoints/sionna_5g/best_model.pth

# 清理screen会话 (如果需要)
screen -S prism_training -X quit
```

## 注意事项

1. **路径问题**: 这些脚本假设从项目根目录运行，或者使用相对路径访问checkpoints和results目录。

2. **依赖要求**: 
   - `screen` 命令 (后台训练脚本会自动安装)
   - `nvidia-smi` (GPU监控，可选)
   - PyTorch (检查点读取)

3. **权限问题**: 确保脚本有执行权限：
   ```bash
   chmod +x scripts/utils/run_training_background.sh
   ```

4. **会话管理**: 避免创建重复的screen会话名称，脚本会自动检查并提示。

## 故障排除

### 训练无法启动
- 检查conda环境是否正确激活
- 确认配置文件路径正确
- 检查CUDA设备可用性

### 监控脚本无响应
- 确认训练进程正在运行
- 检查文件路径是否正确
- 验证检查点文件是否存在

### Screen会话问题
- 使用 `screen -ls` 查看所有会话
- 使用 `screen -r <session_name>` 重新连接
- 使用 `screen -S <session_name> -X quit` 强制终止会话
