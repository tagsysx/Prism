#!/usr/bin/env python3
"""
Performance test script for CUDA ray tracing to verify parallelization.
This script tests the performance improvement after fixing the serial processing issue.
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

def test_ray_tracing_performance():
    """Test ray tracing performance to verify parallelization."""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Cannot test GPU performance.")
        return
    
    logger.info(f"🚀 Testing CUDA ray tracing performance on {torch.cuda.get_device_name()}")
    
    # Initialize ray tracer with realistic parameters
    ray_tracer = CUDARayTracer(
        azimuth_divisions=36,      # 36 azimuth divisions
        elevation_divisions=18,    # 18 elevation divisions
        max_ray_length=100.0,     # 100m max ray length
        scene_size=200.0,         # 200m scene size
        device='cuda',            # Use CUDA
        uniform_samples=128,      # 128 samples per ray
        enable_parallel_processing=True,
        max_workers=4
    )
    
    # Test parameters
    num_ue = 4                    # 4 UE positions
    num_subcarriers = 8           # 8 subcarriers per UE
    
    # Create test data
    base_station_pos = torch.tensor([0.0, 0.0, 10.0], device='cuda')
    
    # Create multiple UE positions
    ue_positions = [
        torch.tensor([50.0, 0.0, 1.5], device='cuda'),
        torch.tensor([-50.0, 0.0, 1.5], device='cuda'),
        torch.tensor([0.0, 50.0, 1.5], device='cuda'),
        torch.tensor([0.0, -50.0, 1.5], device='cuda')
    ]
    
    # Create selected subcarriers mapping
    selected_subcarriers = {}
    for i, ue_pos in enumerate(ue_positions):
        ue_key = tuple(ue_pos.tolist())
        selected_subcarriers[ue_key] = list(range(num_subcarriers))
    
    # Create antenna embeddings
    antenna_embeddings = torch.randn(num_subcarriers, 128, device='cuda')
    
    # Calculate total rays
    total_directions = ray_tracer.azimuth_divisions * ray_tracer.elevation_divisions
    total_rays = total_directions * num_ue * num_subcarriers
    
    logger.info(f"📊 Test Configuration:")
    logger.info(f"   - Total directions: {total_directions} (36×18)")
    logger.info(f"   - UE positions: {num_ue}")
    logger.info(f"   - Subcarriers per UE: {num_subcarriers}")
    logger.info(f"   - Total rays: {total_rays:,}")
    logger.info(f"   - Expected time (old serial): ~{total_rays * 0.7:.1f} seconds")
    logger.info(f"   - Expected time (new parallel): ~{total_rays * 0.01:.1f} seconds")
    
    # Warm up GPU
    logger.info("🔥 Warming up GPU...")
    for _ in range(3):
        _ = ray_tracer.trace_rays_pytorch_gpu(
            base_station_pos, ray_tracer.generate_direction_vectors(),
            ue_positions, selected_subcarriers, antenna_embeddings
        )
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Performance test
    logger.info("🚀 Starting performance test...")
    start_time = time.time()
    
    try:
        results = ray_tracer.trace_rays_pytorch_gpu(
            base_station_pos, ray_tracer.generate_direction_vectors(),
            ue_positions, selected_subcarriers, antenna_embeddings
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate performance metrics
        rays_per_second = total_rays / total_time
        speedup_vs_serial = (total_rays * 0.7) / total_time  # Assuming 0.7s per ray in old version
        
        logger.info(f"✅ Performance test completed!")
        logger.info(f"   - Total time: {total_time:.4f} seconds")
        logger.info(f"   - Rays processed: {total_rays:,}")
        logger.info(f"   - Processing rate: {rays_per_second:,.0f} rays/second")
        logger.info(f"   - Speedup vs old serial: {speedup_vs_serial:.1f}x")
        
        # Performance analysis
        if total_time < 1.0:
            logger.info("🎉 EXCELLENT! Ray tracing is now properly parallelized!")
            logger.info("   - Processing time is orders of magnitude faster than serial version")
        elif total_time < 10.0:
            logger.info("✅ GOOD! Significant performance improvement achieved!")
            logger.info("   - Processing time is much faster than serial version")
        else:
            logger.warning("⚠️  Performance improvement is limited.")
            logger.warning("   - May need further optimization")
        
        # Verify results
        logger.info(f"📋 Results verification:")
        logger.info(f"   - Number of results: {len(results)}")
        logger.info(f"   - Expected results: {total_rays}")
        
        if len(results) == total_rays:
            logger.info("✅ All rays processed successfully!")
        else:
            logger.warning(f"⚠️  Expected {total_rays} results, got {len(results)}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return None

def test_scalability():
    """Test scalability with different numbers of rays."""
    
    if not torch.cuda.is_available():
        return
    
    logger.info("📈 Testing scalability...")
    
    # Test different configurations
    test_configs = [
        (18, 9, 2, 4),    # Small: 162 directions, 2 UE, 4 subcarriers = 1,296 rays
        (36, 18, 4, 8),   # Medium: 648 directions, 4 UE, 8 subcarriers = 20,736 rays
        (72, 36, 8, 16),  # Large: 2,592 directions, 8 UE, 16 subcarriers = 331,776 rays
    ]
    
    for azimuth, elevation, num_ue, num_subcarriers in test_configs:
        logger.info(f"\n🔍 Testing configuration: {azimuth}×{elevation} × {num_ue} UE × {num_subcarriers} subcarriers")
        
        # Initialize ray tracer
        ray_tracer = CUDARayTracer(
            azimuth_divisions=azimuth,
            elevation_divisions=elevation,
            device='cuda',
            uniform_samples=64  # Reduce samples for faster testing
        )
        
        # Create test data
        base_station_pos = torch.tensor([0.0, 0.0, 10.0], device='cuda')
        ue_positions = [torch.tensor([50.0, 0.0, 1.5], device='cuda') for _ in range(num_ue)]
        
        selected_subcarriers = {}
        for ue_pos in ue_positions:
            ue_key = tuple(ue_pos.tolist())
            selected_subcarriers[ue_key] = list(range(num_subcarriers))
        
        antenna_embeddings = torch.randn(num_subcarriers, 128, device='cuda')
        
        # Calculate total rays
        total_directions = azimuth * elevation
        total_rays = total_directions * num_ue * num_subcarriers
        
        # Performance test
        start_time = time.time()
        try:
            results = ray_tracer.trace_rays_pytorch_gpu(
                base_station_pos, ray_tracer.generate_direction_vectors(),
                ue_positions, selected_subcarriers, antenna_embeddings
            )
            end_time = time.time()
            total_time = end_time - start_time
            
            rays_per_second = total_rays / total_time
            logger.info(f"   ✅ Completed: {total_rays:,} rays in {total_time:.3f}s ({rays_per_second:,.0f} rays/s)")
            
        except Exception as e:
            logger.error(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting CUDA Ray Tracing Performance Test")
    logger.info("=" * 60)
    
    # Run main performance test
    results = test_ray_tracing_performance()
    
    if results:
        logger.info("\n" + "=" * 60)
        # Run scalability test
        test_scalability()
    
    logger.info("\n🏁 Performance test completed!")
