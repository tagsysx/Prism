#!/usr/bin/env python3
"""
测试修复后的accumulate_signals方法
验证是否使用超优化算法
"""

import torch
import time
import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prism.ray_tracer_cuda import CUDARayTracer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fixed_accumulate_signals():
    """测试修复后的accumulate_signals方法"""
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return
    
    logger.info(f"🚀 测试修复后的accumulate_signals方法 on {torch.cuda.get_device_name()}")
    
    # 创建射线追踪器 - 模拟你的实际配置
    ray_tracer = CUDARayTracer(
        azimuth_divisions=64,      # 64个方位角
        elevation_divisions=32,    # 32个仰角
        max_ray_length=100.0,
        scene_size=200.0,
        device='cuda',
        uniform_samples=128,
        enable_parallel_processing=True,
        max_workers=2  # 模拟你的2个worker配置
    )
    
    # 测试数据
    base_station_pos = torch.tensor([0.0, 0.0, 10.0], device='cuda')
    
    # 4个UE位置
    ue_positions = [
        torch.tensor([25.0, 0.0, 1.5], device='cuda'),
        torch.tensor([50.0, 25.0, 1.5], device='cuda'),
        torch.tensor([-30.0, 40.0, 1.5], device='cuda'),
        torch.tensor([0.0, -60.0, 1.5], device='cuda'),
    ]
    
    # 8个子载波 per UE
    selected_subcarriers = {}
    for ue_pos in ue_positions:
        ue_key = tuple(ue_pos.tolist())
        selected_subcarriers[ue_key] = list(range(8))
    
    antenna_embedding = torch.randn(8, 128, device='cuda')
    
    # 计算预期射线数量
    total_directions = ray_tracer.azimuth_divisions * ray_tracer.elevation_divisions
    expected_rays = total_directions * len(ue_positions) * 8
    
    logger.info(f"📊 测试配置:")
    logger.info(f"   - 方向: {total_directions} ({ray_tracer.azimuth_divisions}×{ray_tracer.elevation_divisions})")
    logger.info(f"   - UE位置: {len(ue_positions)}")
    logger.info(f"   - 每UE子载波: 8")
    logger.info(f"   - 预期射线: {expected_rays:,}")
    logger.info(f"   - 最大worker: {ray_tracer.max_workers}")
    
    # 测试1: 使用accumulate_signals (应该调用超优化方法)
    logger.info(f"\n{'='*60}")
    logger.info("🔍 测试1: accumulate_signals方法 (应该使用超优化算法)")
    logger.info(f"{'='*60}")
    
    torch.cuda.empty_cache()
    start_time = time.time()
    
    try:
        # 这个方法现在应该调用我们的超优化算法
        results = ray_tracer.accumulate_signals(
            base_station_pos, ue_positions, selected_subcarriers, antenna_embedding
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        actual_results = len(results)
        logger.info(f"✅ accumulate_signals完成!")
        logger.info(f"   - 时间: {total_time:.2f}s ({total_time/60:.2f} 分钟)")
        logger.info(f"   - 结果数量: {actual_results}")
        
        if total_time < 1:
            logger.info("🎉 成功! 在1秒内完成!")
        elif total_time < 60:
            logger.info("✅ 很好! 在1分钟内完成!")
        else:
            logger.info("⚠️ 仍然需要更多优化")
            
    except Exception as e:
        logger.error(f"❌ accumulate_signals失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 直接调用超优化方法对比
    logger.info(f"\n{'='*60}")
    logger.info("🔍 测试2: 直接调用超优化方法对比")
    logger.info(f"{'='*60}")
    
    torch.cuda.empty_cache()
    start_time = time.time()
    
    try:
        # 直接调用超优化方法
        direction_vectors = ray_tracer.generate_direction_vectors()
        results = ray_tracer.trace_rays_pytorch_gpu_ultra_optimized(
            base_station_pos, direction_vectors, ue_positions,
            selected_subcarriers, antenna_embedding
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        actual_rays = len(results)
        rays_per_second = actual_rays / total_time
        
        logger.info(f"✅ 直接超优化方法完成!")
        logger.info(f"   - 时间: {total_time:.2f}s ({total_time/60:.2f} 分钟)")
        logger.info(f"   - 处理射线: {actual_rays:,}")
        logger.info(f"   - 性能: {rays_per_second:,.0f} 射线/秒")
        
        if total_time < 1:
            logger.info("🎉 成功! 在1秒内完成!")
        elif total_time < 60:
            logger.info("✅ 很好! 在1分钟内完成!")
        else:
            logger.info("⚠️ 仍然需要更多优化")
            
    except Exception as e:
        logger.error(f"❌ 直接超优化方法失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("🚀 开始测试修复后的accumulate_signals方法")
    logger.info("=" * 80)
    
    test_fixed_accumulate_signals()
    
    logger.info("\n🏁 测试完成!")
    logger.info("=" * 80)
