#!/usr/bin/env python3
"""
读取PyTorch检查点文件的脚本
"""

import torch
import sys
from pathlib import Path

def read_checkpoint(checkpoint_path):
    """读取检查点文件并显示内容"""
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"=== 检查点文件: {checkpoint_path} ===")
        print()
        
        # 显示基本信息
        print("📊 基本信息:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  最佳验证损失: {checkpoint.get('best_val_loss', 'N/A')}")
        
        # 显示训练损失历史
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            print(f"  训练损失历史: {len(train_losses)} 个epoch")
            if train_losses:
                print(f"    最新损失: {train_losses[-1]:.6f}")
                print(f"    最佳损失: {min(train_losses):.6f}")
        
        # 显示验证损失历史
        if 'val_losses' in checkpoint:
            val_losses = checkpoint['val_losses']
            print(f"  验证损失历史: {len(val_losses)} 个epoch")
            if val_losses:
                print(f"    最新损失: {val_losses[-1]:.6f}")
                print(f"    最佳损失: {min(val_losses):.6f}")
        
        print()
        
        # 显示模型信息
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print("🏗️  模型信息:")
            print(f"  模型参数数量: {len(model_state)} 层")
            
            # 显示前几层的参数形状
            print("  前5层参数形状:")
            for i, (name, param) in enumerate(model_state.items()):
                if i < 5:
                    print(f"    {name}: {param.shape}")
                else:
                    break
            if len(model_state) > 5:
                print(f"    ... 还有 {len(model_state) - 5} 层")
        
        print()
        
        # 显示优化器信息
        if 'optimizer_state_dict' in checkpoint:
            print("⚙️  优化器信息:")
            print("  优化器状态已保存")
        
        # 显示学习率调度器信息
        if 'scheduler_state_dict' in checkpoint:
            print("📈  学习率调度器信息:")
            print("  调度器状态已保存")
        
        print()
        
        # 显示配置信息
        if 'config' in checkpoint:
            config = checkpoint['config']
            print("⚙️  训练配置:")
            if 'training' in config:
                training_config = config['training']
                print(f"  学习率: {training_config.get('learning_rate', 'N/A')}")
                print(f"  批次大小: {training_config.get('batch_size', 'N/A')}")
                print(f"  权重衰减: {training_config.get('weight_decay', 'N/A')}")
            
            if 'model' in config:
                model_config = config['model']
                print(f"  子载波数量: {model_config.get('num_subcarriers', 'N/A')}")
                print(f"  UE天线数量: {model_config.get('num_ue_antennas', 'N/A')}")
                print(f"  BS天线数量: {model_config.get('num_bs_antennas', 'N/A')}")
        
        print()
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 读取检查点文件失败: {e}")
        print("可能的原因:")
        print("  1. 文件正在被写入（训练进行中）")
        print("  2. 文件损坏")
        print("  3. 文件格式不兼容")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python scripts/utils/read_checkpoint.py <检查点文件路径>")
        print("示例: python scripts/utils/read_checkpoint.py ../../checkpoints/sionna_5g/best_model.pth")
        return
    
    checkpoint_path = Path(sys.argv[1])
    
    if not checkpoint_path.exists():
        print(f"❌ 文件不存在: {checkpoint_path}")
        return
    
    read_checkpoint(checkpoint_path)

if __name__ == '__main__':
    main()
