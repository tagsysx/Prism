#!/usr/bin/env python3
"""
测试清理后的CUDARayTracer
验证是否只使用CUDA实现，没有线程池或multiprocessing
"""

import torch
import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prism.ray_tracer_cuda import CUDARayTracer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cuda_only_ray_tracing():
    """测试CUDARayTracer是否只使用CUDA实现"""
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return
    
    logger.info(f"🚀 测试清理后的CUDARayTracer on {torch.cuda.get_device_name()}")
    
    # 创建CUDARayTracer - 注意这些参数现在被忽略
    ray_tracer = CUDARayTracer(
        azimuth_divisions=18,
        elevation_divisions=9,
        max_ray_length=200.0,
        scene_size=200.0,
        device='cuda',
        uniform_samples=64,
        enable_parallel_processing=True,  # 这个参数现在被忽略
        max_workers=8,                   # 这个参数现在被忽略
        use_multiprocessing=True         # 这个参数现在被忽略
    )
    
    logger.info(f"✅ CUDARayTracer创建成功!")
    logger.info(f"   - ray_tracer类型: {type(ray_tracer).__name__}")
    logger.info(f"   - 设备: {ray_tracer.device}")
    logger.info(f"   - 使用CUDA: {ray_tracer.use_cuda}")
    logger.info(f"   - 启用并行处理: {ray_tracer.enable_parallel_processing}")
    logger.info(f"   - 使用多进程: {ray_tracer.use_multiprocessing}")
    logger.info(f"   - 最大worker数: {ray_tracer.max_workers}")
    
    # 验证这些参数被正确忽略
    if ray_tracer.enable_parallel_processing == False and ray_tracer.use_multiprocessing == False and ray_tracer.max_workers == 0:
        logger.info("🎉 成功! 所有CPU并行参数都被正确忽略!")
    else:
        logger.warning("⚠️ 某些CPU并行参数没有被正确忽略")
    
    # 测试数据
    base_station_pos = torch.tensor([0.0, 0.0, 10.0], device='cuda')
    
    # 2个UE位置
    ue_positions = [
        torch.tensor([25.0, 0.0, 1.5], device='cuda'),
        torch.tensor([50.0, 25.0, 1.5], device='cuda'),
    ]
    
    # 8个子载波 per UE
    selected_subcarriers = {}
    for ue_pos in ue_positions:
        ue_key = tuple(ue_pos.tolist())
        selected_subcarriers[ue_key] = list(range(8))
    
    antenna_embeddings = torch.randn(8, 128, device='cuda')
    
    # 计算预期射线数量
    total_directions = ray_tracer.azimuth_divisions * ray_tracer.elevation_divisions
    expected_rays = total_directions * len(ue_positions) * 8
    
    logger.info(f"📊 测试配置:")
    logger.info(f"   - 方向: {total_directions} ({ray_tracer.azimuth_divisions}×{ray_tracer.elevation_divisions})")
    logger.info(f"   - UE位置: {len(ue_positions)}")
    logger.info(f"   - 每UE子载波: 8")
    logger.info(f"   - 预期射线: {expected_rays:,}")
    
    # 测试1: 使用accumulate_signals (应该调用超优化方法)
    logger.info(f"\n{'='*60}")
    logger.info("🔍 测试1: accumulate_signals方法")
    logger.info(f"{'='*60}")
    
    torch.cuda.empty_cache()
    start_time = time.time()
    
    try:
        # 这个方法现在应该调用我们的超优化算法
        results = ray_tracer.accumulate_signals(
            base_station_pos, ue_positions, selected_subcarriers, antenna_embeddings
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
    
    # 测试2: 直接调用超优化方法
    logger.info(f"\n{'='*60}")
    logger.info("🔍 测试2: 直接调用超优化方法")
    logger.info(f"{'='*60}")
    
    torch.cuda.empty_cache()
    start_time = time.time()
    
    try:
        # 直接调用超优化方法
        direction_vectors = ray_tracer.generate_direction_vectors()
        results = ray_tracer.trace_rays_pytorch_gpu_ultra_optimized(
            base_station_pos, direction_vectors, ue_positions,
            selected_subcarriers, antenna_embeddings
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
    
    # 测试3: 检查是否有线程池相关的导入
    logger.info(f"\n{'='*60}")
    logger.info("🔍 测试3: 检查代码清理")
    logger.info(f"{'='*60}")
    
    # 检查源代码中是否还有线程池相关的代码
    source_file = "src/prism/ray_tracer_cuda.py"
    if os.path.exists(source_file):
        with open(source_file, 'r') as f:
            source_content = f.read()
        
        # 检查是否还有线程池相关的代码
        thread_pool_mentions = source_content.count("ThreadPoolExecutor")
        multiprocessing_mentions = source_content.count("multiprocessing")
        max_workers_mentions = source_content.count("max_workers")
        
        logger.info(f"📊 代码清理检查:")
        logger.info(f"   - ThreadPoolExecutor引用: {thread_pool_mentions}")
        logger.info(f"   - multiprocessing引用: {multiprocessing_mentions}")
        logger.info(f"   - max_workers引用: {max_workers_mentions}")
        
        if thread_pool_mentions == 0 and multiprocessing_mentions == 0:
            logger.info("🎉 成功! 所有线程池和多进程代码都被清理!")
        else:
            logger.warning("⚠️ 仍有线程池或多进程相关代码")
    else:
        logger.warning("⚠️ 无法找到源代码文件进行检查")

if __name__ == "__main__":
    import time
    
    logger.info("🚀 开始测试清理后的CUDARayTracer")
    logger.info("=" * 80)
    
    test_cuda_only_ray_tracing()
    
    logger.info("\n🏁 测试完成!")
    logger.info("=" * 80)
