# Prism工程迁移信息

## 打包信息
- 打包时间: Friday, August 29, 2025 PM10:23:02 HKT
- 源服务器: tagsys-sever3
- 用户: young
- Python版本: Python 3.8.20
- CUDA版本: 535.183.01

## 环境依赖
- Python 3.8+
- PyTorch (支持CUDA)
- 其他依赖见 requirements.txt

## 重要文件说明
- `configs/`: 训练配置文件
- `src/prism/`: 核心代码
- `scripts/train_prism.py`: 主训练脚本
- `data/sionna/`: 训练数据
- `results/`: 训练结果和检查点

## 迁移后需要做的事情
1. 安装Python环境和依赖
2. 检查CUDA版本兼容性
3. 验证数据文件完整性
4. 运行测试确保环境正常
5. 根据新服务器GPU配置调整configs中的设置

## 已知问题
- 原服务器RTX 4090 24GB显存不足
- 需要更大显存的GPU (推荐40GB+)
- 训练过程中存在CUDA索引越界错误需要修复

## 联系信息
如有问题请联系原开发者
