#!/usr/bin/env python3
"""
Advanced performance test script for optimized CUDA ray tracing.
This script tests the new advanced optimizations including mixed precision,
memory management, and CUDA version compatibility fixes.
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

def test_advanced_optimizations():
    """Test advanced optimizations and performance improvements."""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Cannot test GPU performance.")
        return
    
    logger.info(f"🚀 Testing ADVANCED OPTIMIZED CUDA ray tracing on {torch.cuda.get_device_name()}")
    
    # Test different optimization levels
    test_configs = [
        {
            'name': 'Standard (36×18)',
            'azimuth': 36, 'elevation': 18, 'ue': 4, 'subcarriers': 8,
            'samples': 128, 'expected_rays': 20_736
        },
        {
            'name': 'High Resolution (72×36)',
            'azimuth': 72, 'elevation': 36, 'ue': 8, 'subcarriers': 16,
            'samples': 64, 'expected_rays': 331_776
        },
        {
            'name': 'Ultra High (144×72)',
            'azimuth': 144, 'elevation': 72, 'ue': 16, 'subcarriers': 32,
            'samples': 32, 'expected_rays': 5_308_416
        }
    ]
    
    for config in test_configs:
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 Testing Configuration: {config['name']}")
        logger.info(f"{'='*80}")
        
        # Initialize ray tracer with advanced optimizations
        ray_tracer = CUDARayTracer(
            azimuth_divisions=config['azimuth'],
            elevation_divisions=config['elevation'],
            max_ray_length=100.0,
            scene_size=200.0,
            device='cuda',
            uniform_samples=config['samples'],
            enable_parallel_processing=True,
            max_workers=4
        )
        
        # Create test data
        base_station_pos = torch.tensor([0.0, 0.0, 10.0], device='cuda')
        
        # Create multiple UE positions
        ue_positions = []
        for i in range(config['ue']):
            angle = 2 * math.pi * i / config['ue']
            x = 50.0 * math.cos(angle)
            y = 50.0 * math.sin(angle)
            ue_positions.append(torch.tensor([x, y, 1.5], device='cuda'))
        
        # Create selected subcarriers mapping
        selected_subcarriers = {}
        for i, ue_pos in enumerate(ue_positions):
            ue_key = tuple(ue_pos.tolist())
            selected_subcarriers[ue_key] = list(range(config['subcarriers']))
        
        # Create antenna embeddings
        antenna_embeddings = torch.randn(config['subcarriers'], 128, device='cuda')
        
        # Calculate total rays
        total_directions = ray_tracer.azimuth_divisions * ray_tracer.elevation_divisions
        total_rays = total_directions * config['ue'] * config['subcarriers']
        
        logger.info(f"📊 Configuration Details:")
        logger.info(f"   - Directions: {total_directions:,} ({config['azimuth']}×{config['elevation']})")
        logger.info(f"   - UE positions: {config['ue']}")
        logger.info(f"   - Subcarriers per UE: {config['subcarriers']}")
        logger.info(f"   - Samples per ray: {config['samples']}")
        logger.info(f"   - Total rays: {total_rays:,}")
        
        # Memory usage before test
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1024**3
        
        # Performance test with multiple runs
        num_runs = 3
        times = []
        
        for run in range(num_runs):
            logger.info(f"\n🏃 Run {run + 1}/{num_runs}")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Warm up GPU
            if run == 0:
                logger.info("🔥 Warming up GPU...")
                for _ in range(2):
                    _ = ray_tracer.trace_rays_pytorch_gpu(
                        base_station_pos, ray_tracer.generate_direction_vectors(),
                        ue_positions, selected_subcarriers, antenna_embeddings
                    )
            
            # Performance test
            start_time = time.time()
            
            try:
                results = ray_tracer.trace_rays_pytorch_gpu(
                    base_station_pos, ray_tracer.generate_direction_vectors(),
                    ue_positions, selected_subcarriers, antenna_embeddings
                )
                
                end_time = time.time()
                total_time = end_time - start_time
                times.append(total_time)
                
                # Calculate performance metrics
                rays_per_second = total_rays / total_time
                rays_per_second_k = rays_per_second / 1000
                
                logger.info(f"   ✅ Completed in {total_time:.4f}s")
                logger.info(f"   🎯 Performance: {rays_per_second:,.0f} rays/second ({rays_per_second_k:.1f}k/s)")
                
                # Verify results
                if len(results) == total_rays:
                    logger.info(f"   ✅ All {total_rays:,} rays processed successfully")
                else:
                    logger.warning(f"   ⚠️  Expected {total_rays:,} results, got {len(results)}")
                
            except Exception as e:
                logger.error(f"   ❌ Run {run + 1} failed: {e}")
                times.append(float('inf'))
        
        # Memory usage after test
        memory_after = torch.cuda.memory_allocated() / 1024**3
        memory_used = memory_after - memory_before
        
        # Performance analysis
        valid_times = [t for t in times if t != float('inf')]
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            min_time = min(valid_times)
            max_time = max(valid_times)
            
            logger.info(f"\n📈 Performance Summary:")
            logger.info(f"   - Average time: {avg_time:.4f}s")
            logger.info(f"   - Best time: {min_time:.4f}s")
            logger.info(f"   - Worst time: {max_time:.4f}s")
            logger.info(f"   - Average performance: {total_rays/avg_time/1000:.1f}k rays/second")
            
            # Performance rating
            if avg_time < 1.0:
                rating = "🎉 EXCELLENT"
            elif avg_time < 5.0:
                rating = "✅ VERY GOOD"
            elif avg_time < 15.0:
                rating = "👍 GOOD"
            else:
                rating = "⚠️  NEEDS IMPROVEMENT"
            
            logger.info(f"   - Rating: {rating}")
        
        logger.info(f"\n💾 Memory Usage:")
        logger.info(f"   - Memory before: {memory_before:.2f} GB")
        logger.info(f"   - Memory after: {memory_after:.2f} GB")
        logger.info(f"   - Memory used: {memory_used:.2f} GB")
        
        # Clean up
        del results, antenna_embeddings, ue_positions
        torch.cuda.empty_cache()
        gc.collect()

def test_mixed_precision():
    """Test mixed precision performance."""
    logger.info(f"\n{'='*80}")
    logger.info("🧪 Testing Mixed Precision Performance")
    logger.info(f"{'='*80}")
    
    if not torch.cuda.is_available():
        return
    
    # Test with and without mixed precision
    configs = [
        {'mixed_precision': False, 'name': 'FP32 Only'},
        {'mixed_precision': True, 'name': 'Mixed Precision'}
    ]
    
    for config in configs:
        logger.info(f"\n🔍 Testing: {config['name']}")
        
        # Initialize ray tracer
        ray_tracer = CUDARayTracer(
            azimuth_divisions=36,
            elevation_divisions=18,
            device='cuda',
            uniform_samples=64
        )
        
        # Set mixed precision
        ray_tracer.use_mixed_precision = config['mixed_precision']
        
        # Create test data
        base_station_pos = torch.tensor([0.0, 0.0, 10.0], device='cuda')
        ue_positions = [torch.tensor([50.0, 0.0, 1.5], device='cuda')]
        selected_subcarriers = {tuple(ue_positions[0].tolist()): [0, 1, 2, 3]}
        antenna_embeddings = torch.randn(4, 128, device='cuda')
        
        # Performance test
        torch.cuda.empty_cache()
        start_time = time.time()
        
        try:
            results = ray_tracer.trace_rays_pytorch_gpu(
                base_station_pos, ray_tracer.generate_direction_vectors(),
                ue_positions, selected_subcarriers, antenna_embeddings
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            total_rays = len(results)
            rays_per_second = total_rays / total_time
            
            logger.info(f"   ✅ Completed in {total_time:.4f}s")
            logger.info(f"   🎯 Performance: {rays_per_second:,.0f} rays/second")
            
        except Exception as e:
            logger.error(f"   ❌ Failed: {e}")

def test_memory_efficiency():
    """Test memory efficiency optimizations."""
    logger.info(f"\n{'='*80}")
    logger.info("💾 Testing Memory Efficiency")
    logger.info(f"{'='*80}")
    
    if not torch.cuda.is_available():
        return
    
    # Test with different memory settings
    ray_tracer = CUDARayTracer(
        azimuth_divisions=72,
        elevation_divisions=36,
        device='cuda',
        uniform_samples=32
    )
    
    # Create large test data
    base_station_pos = torch.tensor([0.0, 0.0, 10.0], device='cuda')
    ue_positions = [torch.tensor([50.0, 0.0, 1.5], device='cuda') for _ in range(16)]
    selected_subcarriers = {tuple(ue_pos.tolist()): list(range(32)) for ue_pos in ue_positions}
    antenna_embeddings = torch.randn(32, 128, device='cuda')
    
    # Monitor memory usage
    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated() / 1024**3
    
    logger.info(f"Memory before test: {memory_before:.2f} GB")
    
    try:
        results = ray_tracer.trace_rays_pytorch_gpu(
            base_station_pos, ray_tracer.generate_direction_vectors(),
            ue_positions, selected_subcarriers, antenna_embeddings
        )
        
        memory_after = torch.cuda.memory_allocated() / 1024**3
        memory_peak = torch.cuda.max_memory_allocated() / 1024**3
        
        logger.info(f"Memory after test: {memory_after:.2f} GB")
        logger.info(f"Peak memory usage: {memory_peak:.2f} GB")
        logger.info(f"Memory efficiency: {(memory_after - memory_before):.2f} GB")
        
        total_rays = len(results)
        logger.info(f"Total rays processed: {total_rays:,}")
        
    except Exception as e:
        logger.error(f"Memory test failed: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting ADVANCED CUDA Ray Tracing Performance Test")
    logger.info("=" * 80)
    
    # Run advanced optimization tests
    test_advanced_optimizations()
    
    # Run mixed precision tests
    test_mixed_precision()
    
    # Run memory efficiency tests
    test_memory_efficiency()
    
    logger.info("\n🏁 Advanced performance test completed!")
    logger.info("=" * 80)
