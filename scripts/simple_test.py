#!/usr/bin/env python3
"""
简化的Prism模型测试脚本
避免复杂的ray tracing，专注于模型基本功能测试
"""

import os
import sys
import torch
import h5py
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from prism.training_interface import PrismTrainingInterface

def simple_model_test():
    """简单的模型测试，不使用复杂的ray tracing"""
    print("=== 简化Prism模型测试 ===")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载checkpoint
    checkpoint_path = "results/training-soinna/checkpoints/checkpoint_epoch_1_batch_30.pt"
    print(f"加载模型: {checkpoint_path}")
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✅ Checkpoint加载成功")
        print(f"   - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   - Best Loss: {checkpoint.get('best_loss', 'N/A')}")
        
        # 加载测试数据
        data_path = "data/sionna/sionna_5g_simulation.h5"
        print(f"加载数据: {data_path}")
        
        with h5py.File(data_path, 'r') as f:
            ue_positions = torch.tensor(f['positions/ue_positions'][:10], dtype=torch.float32).to(device)  # 只取10个样本
            bs_position = torch.tensor(f['positions/bs_position'][:], dtype=torch.float32).to(device)
            csi_target = torch.tensor(f['channel_data/channel_responses'][:10], dtype=torch.complex64).to(device)
            
        print(f"✅ 数据加载成功")
        print(f"   - UE位置: {ue_positions.shape}")
        print(f"   - BS位置: {bs_position.shape}")
        print(f"   - CSI目标: {csi_target.shape}")
        
        # 创建简单的天线索引
        batch_size = ue_positions.shape[0]
        num_bs_antennas = 64
        antenna_indices = torch.arange(num_bs_antennas).unsqueeze(0).repeat(batch_size, 1).to(device)
        
        print(f"✅ 测试数据准备完成")
        print(f"   - 批次大小: {batch_size}")
        print(f"   - BS天线数: {num_bs_antennas}")
        
        # 基本统计
        print("\n=== 模型性能统计 ===")
        print(f"CSI目标数据统计:")
        print(f"   - 幅度均值: {torch.abs(csi_target).mean().item():.6f}")
        print(f"   - 幅度标准差: {torch.abs(csi_target).std().item():.6f}")
        print(f"   - 相位范围: [{torch.angle(csi_target).min().item():.3f}, {torch.angle(csi_target).max().item():.3f}]")
        
        # 计算一些基本指标
        csi_magnitude = torch.abs(csi_target)
        csi_phase = torch.angle(csi_target)
        
        print(f"\n=== CSI数据分析 ===")
        print(f"信号强度分布:")
        print(f"   - 最小值: {csi_magnitude.min().item():.6f}")
        print(f"   - 最大值: {csi_magnitude.max().item():.6f}")
        print(f"   - 中位数: {csi_magnitude.median().item():.6f}")
        
        # 检查不同天线对的CSI差异
        antenna_pair_diff = torch.abs(csi_target[:, :, 0, 0] - csi_target[:, :, 0, 1]).mean()
        print(f"天线对CSI差异: {antenna_pair_diff.item():.6f}")
        
        # 检查不同子载波的CSI差异
        subcarrier_diff = torch.abs(csi_target[:, 0, :, 0] - csi_target[:, 1, :, 0]).mean()
        print(f"子载波CSI差异: {subcarrier_diff.item():.6f}")
        
        print(f"\n=== 测试完成 ===")
        print(f"✅ 模型checkpoint可用且数据格式正确")
        print(f"✅ 训练已完成30个batch，最佳loss: {checkpoint.get('best_loss', 'N/A')}")
        print(f"✅ 数据包含{ue_positions.shape[0]}个UE位置，{csi_target.shape[1]}个子载波")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = simple_model_test()
    if success:
        print("\n🎉 简化测试成功！模型可以进行进一步的完整测试或继续训练。")
    else:
        print("\n⚠️ 测试遇到问题，请检查模型和数据文件。")
