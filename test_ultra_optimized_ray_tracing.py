#!/usr/bin/env python3
"""
Ultra-optimized ray tracing performance test.
This script tests the new algorithmic optimizations for maximum speed.
"""

import torch
import time
import logging
import sys
import os
import gc
import math

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prism.ray_tracer_cuda import CUDARayTracer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ultra_optimization():
    """Test ultra-optimized ray tracing performance."""
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Cannot test GPU performance.")
        return
    
    logger.info(f"🚀 Testing ULTRA-OPTIMIZED ray tracing on {torch.cuda.get_device_name()}")
    
    # Test configuration similar to real usage
    ray_tracer = CUDARayTracer(
        azimuth_divisions=36,      # 36 azimuth divisions
        elevation_divisions=18,    # 18 elevation divisions
        max_ray_length=100.0,     # 100m max ray length
        scene_size=200.0,         # 200m scene size
        device='cuda',            # Use CUDA
        uniform_samples=128,      # Will be adaptively reduced
        enable_parallel_processing=True,
        max_workers=4
    )
    
    # Create test data for 2048 rays scenario
    base_station_pos = torch.tensor([0.0, 0.0, 10.0], device='cuda')
    
    # Create UE positions at various distances
    ue_positions = [
        torch.tensor([10.0, 0.0, 1.5], device='cuda'),    # Close UE
        torch.tensor([50.0, 0.0, 1.5], device='cuda'),    # Medium distance
        torch.tensor([80.0, 0.0, 1.5], device='cuda'),    # Far UE
        torch.tensor([0.0, 30.0, 1.5], device='cuda'),    # Different angle
    ]
    
    # Create selected subcarriers (typical scenario)
    selected_subcarriers = {}
    for i, ue_pos in enumerate(ue_positions):
        ue_key = tuple(ue_pos.tolist())
        selected_subcarriers[ue_key] = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 subcarriers per UE
    
    # Create antenna embeddings
    antenna_embeddings = torch.randn(8, 128, device='cuda')
    
    # Calculate expected rays
    total_directions = ray_tracer.azimuth_divisions * ray_tracer.elevation_divisions
    total_rays = total_directions * len(ue_positions) * 8  # 8 subcarriers
    
    logger.info(f"📊 Test Configuration:")
    logger.info(f"   - Directions: {total_directions} (36×18)")
    logger.info(f"   - UE positions: {len(ue_positions)}")
    logger.info(f"   - Subcarriers per UE: 8")
    logger.info(f"   - Expected total rays: {total_rays:,}")
    
    # Test different optimization levels
    test_methods = [
        ("Original Advanced", "trace_rays_pytorch_gpu"),
        ("Ultra Optimized", "trace_rays_pytorch_gpu_ultra_optimized")
    ]
    
    results_comparison = {}
    
    for method_name, method_func in test_methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Testing: {method_name}")
        logger.info(f"{'='*60}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get the method
        trace_method = getattr(ray_tracer, method_func)
        
        # Warm up
        logger.info("🔥 Warming up...")
        for _ in range(2):
            _ = trace_method(
                base_station_pos, ray_tracer.generate_direction_vectors(),
                ue_positions, selected_subcarriers, antenna_embeddings
            )
        
        # Performance test
        torch.cuda.empty_cache()
        start_time = time.time()
        
        try:
            results = trace_method(
                base_station_pos, ray_tracer.generate_direction_vectors(),
                ue_positions, selected_subcarriers, antenna_embeddings
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate performance metrics
            actual_rays = len(results)
            rays_per_second = actual_rays / total_time
            
            logger.info(f"✅ {method_name} completed!")
            logger.info(f"   - Time: {total_time:.4f}s")
            logger.info(f"   - Rays processed: {actual_rays:,}")
            logger.info(f"   - Performance: {rays_per_second:,.0f} rays/second")
            logger.info(f"   - Performance: {rays_per_second/1000:.1f}k rays/second")
            
            # Estimate time for 2048 rays
            if actual_rays > 0:
                estimated_2048_time = 2048 * total_time / actual_rays
                logger.info(f"   - Estimated time for 2048 rays: {estimated_2048_time:.2f}s")
                
                if estimated_2048_time < 60:
                    logger.info(f"   - 🎉 EXCELLENT! Under 1 minute for 2048 rays!")
                elif estimated_2048_time < 300:
                    logger.info(f"   - ✅ GOOD! Under 5 minutes for 2048 rays!")
                else:
                    logger.info(f"   - ⚠️  Still needs improvement for 2048 rays")
            
            results_comparison[method_name] = {
                'time': total_time,
                'rays': actual_rays,
                'rays_per_second': rays_per_second,
                'estimated_2048_time': estimated_2048_time if actual_rays > 0 else float('inf')
            }
            
        except Exception as e:
            logger.error(f"❌ {method_name} failed: {e}")
            results_comparison[method_name] = {
                'time': float('inf'),
                'rays': 0,
                'rays_per_second': 0,
                'estimated_2048_time': float('inf')
            }
    
    # Performance comparison
    logger.info(f"\n{'='*60}")
    logger.info("📈 Performance Comparison")
    logger.info(f"{'='*60}")
    
    if len(results_comparison) >= 2:
        methods = list(results_comparison.keys())
        original = results_comparison[methods[0]]
        optimized = results_comparison[methods[1]]
        
        if original['time'] > 0 and optimized['time'] > 0:
            speedup = original['time'] / optimized['time']
            logger.info(f"🚀 Speedup: {speedup:.2f}x faster")
            
            time_improvement = original['estimated_2048_time'] - optimized['estimated_2048_time']
            logger.info(f"⏰ Time saved for 2048 rays: {time_improvement:.2f} seconds")
            
            if optimized['estimated_2048_time'] < 60:
                logger.info("🎯 TARGET ACHIEVED: 2048 rays in under 1 minute!")
            elif optimized['estimated_2048_time'] < 300:
                logger.info("👍 GOOD PROGRESS: 2048 rays in under 5 minutes!")

def test_real_2048_scenario():
    """Test a realistic 2048 ray scenario."""
    
    if not torch.cuda.is_available():
        return
    
    logger.info(f"\n{'='*60}")
    logger.info("🎯 Testing Real 2048 Ray Scenario")
    logger.info(f"{'='*60}")
    
    # Create a scenario that results in approximately 2048 rays
    # 32 directions × 2 UE × 32 subcarriers = 2048 rays
    ray_tracer = CUDARayTracer(
        azimuth_divisions=8,       # 8 azimuth divisions
        elevation_divisions=4,     # 4 elevation divisions  
        max_ray_length=100.0,
        scene_size=200.0,
        device='cuda',
        uniform_samples=64,        # Reduced samples
    )
    
    base_station_pos = torch.tensor([0.0, 0.0, 10.0], device='cuda')
    
    # 2 UE positions
    ue_positions = [
        torch.tensor([25.0, 0.0, 1.5], device='cuda'),
        torch.tensor([50.0, 25.0, 1.5], device='cuda'),
    ]
    
    # 32 subcarriers per UE to get close to 2048 total rays
    # 8×4 = 32 directions × 2 UE × 32 subcarriers = 2048 rays
    selected_subcarriers = {}
    for ue_pos in ue_positions:
        ue_key = tuple(ue_pos.tolist())
        selected_subcarriers[ue_key] = list(range(32))
    
    antenna_embeddings = torch.randn(32, 128, device='cuda')
    
    total_directions = ray_tracer.azimuth_divisions * ray_tracer.elevation_divisions
    expected_rays = total_directions * len(ue_positions) * 32
    
    logger.info(f"📊 Real 2048 Ray Test:")
    logger.info(f"   - Directions: {total_directions} ({ray_tracer.azimuth_divisions}×{ray_tracer.elevation_divisions})")
    logger.info(f"   - UE positions: {len(ue_positions)}")
    logger.info(f"   - Subcarriers per UE: 32")
    logger.info(f"   - Expected rays: {expected_rays:,}")
    
    # Test ultra-optimized version
    torch.cuda.empty_cache()
    start_time = time.time()
    
    try:
        results = ray_tracer.trace_rays_pytorch_gpu_ultra_optimized(
            base_station_pos, ray_tracer.generate_direction_vectors(),
            ue_positions, selected_subcarriers, antenna_embeddings
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        actual_rays = len(results)
        rays_per_second = actual_rays / total_time
        
        logger.info(f"✅ Real scenario completed!")
        logger.info(f"   - Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        logger.info(f"   - Rays processed: {actual_rays:,}")
        logger.info(f"   - Performance: {rays_per_second:,.0f} rays/second")
        
        if total_time < 60:
            logger.info("🎉 SUCCESS! Under 1 minute for ~2048 rays!")
        elif total_time < 300:
            logger.info("✅ GOOD! Under 5 minutes for ~2048 rays!")
        else:
            logger.info("⚠️  Still needs more optimization")
            
    except Exception as e:
        logger.error(f"❌ Real scenario test failed: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting Ultra-Optimized Ray Tracing Performance Test")
    logger.info("=" * 80)
    
    # Test optimization comparison
    test_ultra_optimization()
    
    # Test real 2048 ray scenario
    test_real_2048_scenario()
    
    logger.info("\n🏁 Ultra-optimization test completed!")
    logger.info("=" * 80)
