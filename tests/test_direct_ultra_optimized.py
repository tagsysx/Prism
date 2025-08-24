#!/usr/bin/env python3
"""
直接测试超优化射线追踪算法
绕过其他代码路径，直接调用超优化方法
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

def test_direct_ultra_optimized():
    """直接测试超优化算法，绕过其他代码路径"""
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return
    
    logger.info(f"🚀 直接测试超优化算法 on {torch.cuda.get_device_name()}")
    
    # 创建射线追踪器
    ray_tracer = CUDARayTracer(
        azimuth_divisions=8,       # 8个方位角
        elevation_divisions=4,     # 4个仰角
        max_ray_length=100.0,
        scene_size=200.0,
        device='cuda',
        uniform_samples=64,
        enable_parallel_processing=True,
        max_workers=4
    )
    
    # 测试数据 - 模拟你的2048射线场景
    base_station_pos = torch.tensor([0.0, 0.0, 10.0], device='cuda')
    
    # 2个UE位置
    ue_positions = [
        torch.tensor([25.0, 0.0, 1.5], device='cuda'),
        torch.tensor([50.0, 25.0, 1.5], device='cuda'),
    ]
    
    # 32个子载波 per UE = 8×4×2×32 = 2048 射线
    selected_subcarriers = {}
    for ue_pos in ue_positions:
        ue_key = tuple(ue_pos.tolist())
        selected_subcarriers[ue_key] = list(range(32))
    
    antenna_embeddings = torch.randn(32, 128, device='cuda')
    
    # 计算预期射线数量
    total_directions = ray_tracer.azimuth_divisions * ray_tracer.elevation_divisions
    expected_rays = total_directions * len(ue_positions) * 32
    
    logger.info(f"📊 测试配置:")
    logger.info(f"   - 方向: {total_directions} ({ray_tracer.azimuth_divisions}×{ray_tracer.elevation_divisions})")
    logger.info(f"   - UE位置: {len(ue_positions)}")
    logger.info(f"   - 每UE子载波: 32")
    logger.info(f"   - 预期射线: {expected_rays:,}")
    
    # 方法1: 直接调用超优化方法
    logger.info(f"\n{'='*60}")
    logger.info("🔍 方法1: 直接调用超优化方法")
    logger.info(f"{'='*60}")
    
    torch.cuda.empty_cache()
    start_time = time.time()
    
    try:
        # 直接调用超优化方法，绕过trace_rays()
        results = ray_tracer.trace_rays_pytorch_gpu_ultra_optimized(
            base_station_pos, ray_tracer.generate_direction_vectors(),
            ue_positions, selected_subcarriers, antenna_embeddings
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
            logger.info("🎉 成功! 2048射线在1秒内完成!")
        elif total_time < 60:
            logger.info("✅ 很好! 2048射线在1分钟内完成!")
        else:
            logger.info("⚠️ 需要更多优化")
            
    except Exception as e:
        logger.error(f"❌ 直接超优化方法失败: {e}")
    
    # 方法2: 通过trace_rays()调用
    logger.info(f"\n{'='*60}")
    logger.info("🔍 方法2: 通过trace_rays()调用")
    logger.info(f"{'='*60}")
    
    torch.cuda.empty_cache()
    start_time = time.time()
    
    try:
        # 通过trace_rays()调用，应该自动选择超优化版本
        results = ray_tracer.trace_rays(
            base_station_pos, ue_positions, selected_subcarriers, antenna_embeddings
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        actual_rays = len(results)
        rays_per_second = actual_rays / total_time
        
        logger.info(f"✅ trace_rays()方法完成!")
        logger.info(f"   - 时间: {total_time:.2f}s ({total_time/60:.2f} 分钟)")
        logger.info(f"   - 处理射线: {actual_rays:,}")
        logger.info(f"   - 性能: {rays_per_second:,.0f} 射线/秒")
        
        if total_time < 1:
            logger.info("🎉 成功! 2048射线在1秒内完成!")
        elif total_time < 60:
            logger.info("✅ 很好! 2048射线在1分钟内完成!")
        else:
            logger.info("⚠️ 需要更多优化")
            
    except Exception as e:
        logger.error(f"❌ trace_rays()方法失败: {e}")
    
    # 方法3: 强制使用超优化版本
    logger.info(f"\n{'='*60}")
    logger.info("🔍 方法3: 强制使用超优化版本")
    logger.info(f"{'='*60}")
    
    # 临时禁用CUDA kernel，强制使用超优化版本
    original_use_cuda = ray_tracer.use_cuda
    ray_tracer.use_cuda = False  # 强制使用PyTorch GPU版本
    
    torch.cuda.empty_cache()
    start_time = time.time()
    
    try:
        results = ray_tracer.trace_rays(
            base_station_pos, ue_positions, selected_subcarriers, antenna_embeddings
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        actual_rays = len(results)
        rays_per_second = actual_rays / total_time
        
        logger.info(f"✅ 强制超优化版本完成!")
        logger.info(f"   - 时间: {total_time:.2f}s ({total_time/60:.2f} 分钟)")
        logger.info(f"   - 处理射线: {actual_rays:,}")
        logger.info(f"   - 性能: {rays_per_second:,.0f} 射线/秒")
        
        if total_time < 1:
            logger.info("🎉 成功! 2048射线在1秒内完成!")
        elif total_time < 60:
            logger.info("✅ 很好! 2048射线在1分钟内完成!")
        else:
            logger.info("⚠️ 需要更多优化")
            
    except Exception as e:
        logger.error(f"❌ 强制超优化版本失败: {e}")
    finally:
        # 恢复原始设置
        ray_tracer.use_cuda = original_use_cuda

if __name__ == "__main__":
    logger.info("🚀 开始直接超优化算法测试")
    logger.info("=" * 80)
    
    test_direct_ultra_optimized()
    
    logger.info("\n🏁 直接测试完成!")
    logger.info("=" * 80)
