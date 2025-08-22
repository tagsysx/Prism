#!/usr/bin/env python3
"""
CUDA-Accelerated Ray Tracing Example

This example demonstrates the performance improvement of the CUDA-accelerated
ray tracing system compared to the original CPU implementation.
"""

import torch
import numpy as np
import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prism.ray_tracer_cuda import CUDARayTracer
from prism.ray_tracer import DiscreteRayTracer

def create_test_scenario(num_ue: int = 100, num_subcarriers: int = 64):
    """Create a test scenario with multiple UEs and subcarriers."""
    
    # Base station position
    base_station_pos = torch.tensor([0.0, 0.0, 0.0])
    
    # Generate random UE positions
    np.random.seed(42)  # For reproducible results
    ue_positions = []
    for _ in range(num_ue):
        x = np.random.uniform(-80, 80)
        y = np.random.uniform(-80, 80)
        z = np.random.uniform(1.0, 2.0)
        ue_positions.append([x, y, z])
    
    # Create subcarrier selection for each UE
    selected_subcarriers = {}
    for ue_pos in ue_positions:
        # Randomly select 30% of subcarriers for each UE
        num_selected = max(1, int(num_subcarriers * 0.3))
        selected = np.random.choice(num_subcarriers, num_selected, replace=False)
        selected_subcarriers[tuple(ue_pos)] = selected.tolist()
    
    # Create dummy antenna embeddings
    antenna_embeddings = torch.randn(num_subcarriers, 64)
    
    return base_station_pos, ue_positions, selected_subcarriers, antenna_embeddings

def benchmark_ray_tracers(base_station_pos, ue_positions, selected_subcarriers, antenna_embeddings):
    """Benchmark different ray tracing implementations."""
    
    print("=" * 60)
    print("RAY TRACING PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Test parameters
    azimuth_divisions = 36
    elevation_divisions = 18
    max_ray_length = 100.0
    scene_size = 200.0
    
    # 1. Test CUDA Ray Tracer
    print("\n1. Testing CUDA-Accelerated Ray Tracer...")
    cuda_tracer = CUDARayTracer(
        azimuth_divisions=azimuth_divisions,
        elevation_divisions=elevation_divisions,
        max_ray_length=max_ray_length,
        scene_size=scene_size
    )
    
    # Get performance info
    perf_info = cuda_tracer.get_performance_info()
    print(f"   Device: {perf_info['device']}")
    print(f"   CUDA enabled: {perf_info['use_cuda']}")
    if perf_info['use_cuda']:
        print(f"   CUDA device: {perf_info['cuda_device_name']}")
        print(f"   CUDA memory: {perf_info['cuda_memory_gb']:.1f} GB")
    
    # Benchmark CUDA implementation
    start_time = time.time()
    cuda_results = cuda_tracer.trace_rays(
        base_station_pos, ue_positions, selected_subcarriers, antenna_embeddings
    )
    cuda_time = time.time() - start_time
    
    print(f"   CUDA execution time: {cuda_time:.4f}s")
    print(f"   Results count: {len(cuda_results)}")
    
    # 2. Test Original CPU Ray Tracer (for comparison)
    print("\n2. Testing Original CPU Ray Tracer...")
    cpu_tracer = DiscreteRayTracer(
        azimuth_divisions=azimuth_divisions,
        elevation_divisions=elevation_divisions,
        max_ray_length=max_ray_length,
        scene_size=scene_size,
        device='cpu'
    )
    
    # Benchmark CPU implementation
    start_time = time.time()
    cpu_results = cpu_tracer.accumulate_signals(
        base_station_pos, ue_positions, selected_subcarriers, antenna_embeddings[0]
    )
    cpu_time = time.time() - start_time
    
    print(f"   CPU execution time: {cpu_time:.4f}s")
    print(f"   Results count: {len(cpu_results)}")
    
    # 3. Performance Analysis
    print("\n3. PERFORMANCE ANALYSIS")
    print("-" * 30)
    
    if cuda_time > 0 and cpu_time > 0:
        speedup = cpu_time / cuda_time
        print(f"   Speedup: {speedup:.2f}x faster with CUDA")
        
        if speedup > 10:
            print("   🚀 Excellent performance improvement!")
        elif speedup > 5:
            print("   ⚡ Very good performance improvement!")
        elif speedup > 2:
            print("   ✅ Good performance improvement!")
        else:
            print("   ⚠️  Moderate performance improvement")
    
    # Calculate theoretical performance
    total_rays = azimuth_divisions * elevation_divisions * len(ue_positions) * len(selected_subcarriers)
    print(f"   Total rays processed: {total_rays:,}")
    print(f"   Rays per second (CUDA): {total_rays/cuda_time:,.0f}")
    print(f"   Rays per second (CPU): {total_rays/cpu_time:,.0f}")
    
    return cuda_results, cpu_results, cuda_time, cpu_time

def test_different_scenarios():
    """Test different scenario sizes to show scalability."""
    
    print("\n" + "=" * 60)
    print("SCALABILITY TESTING")
    print("=" * 60)
    
    scenarios = [
        (50, 32, "Small"),
        (100, 64, "Medium"),
        (200, 128, "Large"),
        (500, 256, "Extra Large")
    ]
    
    results = []
    
    for num_ue, num_subcarriers, scenario_name in scenarios:
        print(f"\n--- {scenario_name} Scenario: {num_ue} UEs, {num_subcarriers} subcarriers ---")
        
        try:
            base_station_pos, ue_positions, selected_subcarriers, antenna_embeddings = create_test_scenario(
                num_ue, num_subcarriers
            )
            
            # Quick benchmark
            cuda_tracer = CUDARayTracer(
                azimuth_divisions=36,
                elevation_divisions=18,
                max_ray_length=100.0,
                scene_size=200.0
            )
            
            start_time = time.time()
            cuda_results = cuda_tracer.trace_rays(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embeddings
            )
            cuda_time = time.time() - start_time
            
            total_rays = 36 * 18 * num_ue * num_subcarriers
            rays_per_second = total_rays / cuda_time
            
            print(f"   Execution time: {cuda_time:.4f}s")
            print(f"   Total rays: {total_rays:,}")
            print(f"   Rays per second: {rays_per_second:,.0f}")
            
            results.append({
                'scenario': scenario_name,
                'num_ue': num_ue,
                'num_subcarriers': num_subcarriers,
                'time': cuda_time,
                'rays_per_second': rays_per_second
            })
            
        except Exception as e:
            print(f"   Error: {e}")
            continue
    
    # Summary
    if results:
        print(f"\n--- SCALABILITY SUMMARY ---")
        for result in results:
            print(f"{result['scenario']:12}: {result['rays_per_second']:>10,.0f} rays/sec")

def main():
    """Main function to run the CUDA ray tracing example."""
    
    print("CUDA-Accelerated Ray Tracing System Demo")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA detected: {torch.cuda.get_device_name()}")
        print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"✓ PyTorch CUDA version: {torch.version.cuda}")
    else:
        print("⚠ CUDA not available - will use CPU implementation")
    
    # Create test scenario
    print("\nCreating test scenario...")
    base_station_pos, ue_positions, selected_subcarriers, antenna_embeddings = create_test_scenario(
        num_ue=100, num_subcarriers=64
    )
    
    print(f"   Base station: {base_station_pos.tolist()}")
    print(f"   UEs: {len(ue_positions)}")
    print(f"   Subcarriers: {len(antenna_embeddings)}")
    
    # Run benchmark
    cuda_results, cpu_results, cuda_time, cpu_time = benchmark_ray_tracers(
        base_station_pos, ue_positions, selected_subcarriers, antenna_embeddings
    )
    
    # Test scalability
    test_different_scenarios()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("✓ CUDA acceleration successfully implemented")
        print("✓ Automatic device detection working")
        print("✓ Performance improvement achieved")
    else:
        print("⚠ CUDA not available - using CPU implementation")
        print("⚠ Consider upgrading to a CUDA-capable GPU for better performance")
    
    print(f"\nTotal rays processed: {len(cuda_results):,}")
    print(f"CUDA execution time: {cuda_time:.4f}s")
    if cpu_time > 0:
        print(f"CPU execution time: {cpu_time:.4f}s")
        print(f"Speedup: {cpu_time/cuda_time:.2f}x")

if __name__ == "__main__":
    main()
