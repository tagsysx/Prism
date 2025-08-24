#!/usr/bin/env python3
"""
测试修复后的training_interface
验证是否正确根据ray_tracing_mode选择ray_tracer
"""

import torch
import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prism.training_interface import TrainingInterface
from prism.prism_network import PrismNetwork

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def test_training_interface_modes():
    """测试不同ray_tracing_mode下的ray_tracer选择"""
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available, cannot test CUDA mode")
        return
    
    logger.info("🚀 测试training_interface的ray_tracer选择逻辑")
    
    # 创建PrismNetwork (简化版本)
    try:
        prism_network = PrismNetwork(
            num_subcarriers=408,
            num_ue_antennas=4,
            num_bs_antennas=64
        )
        logger.info("✅ PrismNetwork created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create PrismNetwork: {e}")
        return
    
    # 测试1: CUDA模式
    logger.info(f"\n{'='*60}")
    logger.info("🔍 测试1: ray_tracing_mode = 'cuda'")
    logger.info(f"{'='*60}")
    
    try:
        training_interface = TrainingInterface(
            prism_network=prism_network,
            ray_tracing_mode='cuda'  # 不传入ray_tracer，让它自动创建
        )
        
        logger.info(f"✅ CUDA模式创建成功!")
        logger.info(f"   - ray_tracing_mode: {training_interface.ray_tracing_mode}")
        logger.info(f"   - ray_tracer类型: {type(training_interface.ray_tracer).__name__}")
        
        # 验证是否使用了CUDARayTracer
        if 'CUDARayTracer' in str(type(training_interface.ray_tracer)):
            logger.info("🎉 成功使用CUDARayTracer!")
        else:
            logger.warning("⚠️ 没有使用CUDARayTracer")
            
    except Exception as e:
        logger.error(f"❌ CUDA模式创建失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: CPU模式
    logger.info(f"\n{'='*60}")
    logger.info("🔍 测试2: ray_tracing_mode = 'cpu'")
    logger.info(f"{'='*60}")
    
    try:
        training_interface = TrainingInterface(
            prism_network=prism_network,
            ray_tracing_mode='cpu'
        )
        
        logger.info(f"✅ CPU模式创建成功!")
        logger.info(f"   - ray_tracing_mode: {training_interface.ray_tracing_mode}")
        logger.info(f"   - ray_tracer类型: {type(training_interface.ray_tracer).__name__}")
        
        # 验证是否使用了CPURayTracer
if 'CPURayTracer' in str(type(training_interface.ray_tracer)):
    logger.info("🎉 成功使用CPURayTracer!")
else:
    logger.warning("⚠️ 没有使用CPURayTracer")
            
    except Exception as e:
        logger.error(f"❌ CPU模式创建失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试3: Hybrid模式
    logger.info(f"\n{'='*60}")
    logger.info("🔍 测试3: ray_tracing_mode = 'hybrid'")
    logger.info(f"{'='*60}")
    
    try:
        training_interface = TrainingInterface(
            prism_network=prism_network,
            ray_tracing_mode='hybrid'
        )
        
        logger.info(f"✅ Hybrid模式创建成功!")
        logger.info(f"   - ray_tracing_mode: {training_interface.ray_tracing_mode}")
        logger.info(f"   - ray_tracer类型: {type(training_interface.ray_tracer).__name__}")
        
        # Hybrid模式应该优先使用CUDA
        if 'CUDARayTracer' in str(type(training_interface.ray_tracer)):
            logger.info("🎉 Hybrid模式成功使用CUDARayTracer!")
        elif 'CPURayTracer' in str(type(training_interface.ray_tracer)):
    logger.info("💻 Hybrid模式fallback到CPURayTracer")
        else:
            logger.warning("⚠️ Hybrid模式使用了未知的ray_tracer类型")
            
    except Exception as e:
        logger.error(f"❌ Hybrid模式创建失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试4: 传入预创建的ray_tracer
    logger.info(f"\n{'='*60}")
    logger.info("🔍 测试4: 传入预创建的ray_tracer")
    logger.info(f"{'='*60}")
    
    try:
        # 创建CUDARayTracer
        from prism.ray_tracer_cuda import CUDARayTracer
        cuda_ray_tracer = CUDARayTracer(
            azimuth_divisions=18,
            elevation_divisions=9,
            max_ray_length=200.0,
            scene_size=200.0,
            device='cuda',
            uniform_samples=64
        )
        
        training_interface = TrainingInterface(
            prism_network=prism_network,
            ray_tracer=cuda_ray_tracer,  # 传入预创建的ray_tracer
            ray_tracing_mode='cuda'
        )
        
        logger.info(f"✅ 预创建ray_tracer模式成功!")
        logger.info(f"   - ray_tracing_mode: {training_interface.ray_tracing_mode}")
        logger.info(f"   - ray_tracer类型: {type(training_interface.ray_tracer).__name__}")
        
        # 验证是否使用了传入的ray_tracer
        if training_interface.ray_tracer is cuda_ray_tracer:
            logger.info("🎉 成功使用传入的ray_tracer!")
        else:
            logger.warning("⚠️ 没有使用传入的ray_tracer")
            
    except Exception as e:
        logger.error(f"❌ 预创建ray_tracer模式失败: {e}")
        import traceback
        traceback.print_exc()

def test_ray_tracing_performance():
    """测试ray tracing性能"""
    
    if not torch.cuda.is_available():
        return
    
    logger.info(f"\n{'='*60}")
    logger.info("🚀 测试ray tracing性能")
    logger.info(f"{'='*60}")
    
    try:
        # 创建PrismNetwork
        prism_network = PrismNetwork(
            num_subcarriers=408,
            num_ue_antennas=4,
            num_bs_antennas=64
        )
        
        # 创建CUDA模式的training_interface
        training_interface = TrainingInterface(
            prism_network=prism_network,
            ray_tracing_mode='cuda'
        )
        
        logger.info(f"✅ 创建了ray_tracing_mode='cuda'的training_interface")
        logger.info(f"   - ray_tracer类型: {type(training_interface.ray_tracer).__name__}")
        
        # 测试数据
        batch_size = 2
        ue_positions = torch.randn(batch_size, 3, device='cuda') * 50  # 随机UE位置
        bs_position = torch.tensor([0.0, 0.0, 10.0], device='cuda')   # BS位置
        antenna_indices = torch.randint(0, 64, (batch_size, 4), device='cuda')  # 天线索引
        
        logger.info(f"📊 测试数据:")
        logger.info(f"   - batch_size: {batch_size}")
        logger.info(f"   - ue_positions: {ue_positions.shape}")
        logger.info(f"   - bs_position: {bs_position.shape}")
        logger.info(f"   - antenna_indices: {antenna_indices.shape}")
        
        # 运行forward方法
        logger.info("🚀 开始运行forward方法...")
        import time
        start_time = time.time()
        
        try:
            results = training_interface(ue_positions, bs_position, antenna_indices)
            end_time = time.time()
            
            logger.info(f"✅ Forward方法完成!")
            logger.info(f"   - 时间: {end_time - start_time:.2f}秒")
            logger.info(f"   - 结果类型: {type(results)}")
            logger.info(f"   - 结果键: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
            
        except Exception as e:
            logger.error(f"❌ Forward方法失败: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        logger.error(f"❌ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("🚀 开始测试修复后的training_interface")
    logger.info("=" * 80)
    
    # 测试ray_tracer选择逻辑
    test_training_interface_modes()
    
    # 测试ray tracing性能
    test_ray_tracing_performance()
    
    logger.info("\n🏁 测试完成!")
    logger.info("=" * 80)
