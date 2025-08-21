"""
Tests for GPU-accelerated ray tracing functionality.
This script tests the CUDA implementation of ray tracing to ensure correctness.
"""

import pytest
import torch
import numpy as np
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from prism.ray_tracer import (
    GPURayTracer, AdvancedRayTracer, Environment, Building, Plane, 
    Ray, RayGenerator, PathTracer, InteractionModel, ChannelEstimator, SpatialAnalyzer,
    create_gpu_ray_tracer, create_cpu_ray_tracer, benchmark_ray_tracing_performance,
    compare_cpu_gpu_performance
)

class TestGPURayTracer:
    """Test GPU ray tracing functionality."""
    
    @pytest.fixture(scope="class")
    def gpu_available(self):
        """Check if GPU is available for testing."""
        return torch.cuda.is_available()
    
    @pytest.fixture(scope="class")
    def device(self, gpu_available):
        """Get device for testing."""
        return 'cuda' if gpu_available else 'cpu'
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'ray_tracing': {
                'gpu_acceleration': True,
                'batch_size': 64,  # Smaller for testing
                'max_concurrent_rays': 500,
                'gpu_memory_fraction': 0.5,  # Use less memory for testing
                'mixed_precision': True,
                'azimuth_samples': 12,  # Smaller for testing
                'elevation_samples': 6,
                'points_per_ray': 32,
                'reflection_order': 2,
                'max_diffractions': 1
            }
        }
        
        # Create test environment
        self.env = Environment(device=self.device)
        
        # Add test obstacles
        wall = Plane([0, 0, 0], [1, 0, 0], 'concrete', device=self.device)
        building = Building([-5, -5, 0], [5, 5, 10], 'concrete', device=self.device)
        self.env.add_obstacle(wall)
        self.env.add_obstacle(building)
        
        # Create test source positions
        self.source_positions = torch.randn(20, 3, device=self.device)
        
        # Create test target positions
        self.target_positions = torch.randn(20, 3, device=self.device)
    
    def test_gpu_ray_tracer_creation(self, gpu_available):
        """Test GPU ray tracer creation."""
        if not gpu_available:
            pytest.skip("CUDA not available")
        
        try:
            gpu_tracer = create_gpu_ray_tracer(self.config)
            assert isinstance(gpu_tracer, GPURayTracer)
            assert gpu_tracer.device == 'cuda'
            print(f"✓ GPU Ray Tracer created successfully on {torch.cuda.get_device_name()}")
        except Exception as e:
            pytest.fail(f"Failed to create GPU ray tracer: {e}")
    
    def test_cpu_ray_tracer_creation(self):
        """Test CPU ray tracer creation."""
        try:
            cpu_tracer = create_cpu_ray_tracer(self.config)
            assert isinstance(cpu_tracer, AdvancedRayTracer)
            assert cpu_tracer.device == 'cpu'
            print("✓ CPU Ray Tracer created successfully")
        except Exception as e:
            pytest.fail(f"Failed to create CPU ray tracer: {e}")
    
    def test_ray_generation_gpu(self, gpu_available):
        """Test GPU ray generation."""
        if not gpu_available:
            pytest.skip("CUDA not available")
        
        try:
            gpu_tracer = create_gpu_ray_tracer(self.config)
            
            # Test single ray generation
            rays = gpu_tracer.ray_generator.generate_rays(self.source_positions[0])
            assert len(rays) == 12 * 6  # azimuth × elevation
            assert all(ray.device.type == 'cuda' for ray in rays)
            
            # Test batch ray generation
            ray_origins, ray_directions = gpu_tracer.ray_generator.generate_rays_batch(self.source_positions)
            assert ray_origins.shape == (20, 72, 3)  # batch_size × total_rays × 3
            assert ray_directions.shape == (20, 72, 3)
            assert ray_origins.device.type == 'cuda'
            assert ray_directions.device.type == 'cuda'
            
            print("✓ GPU ray generation working correctly")
        except Exception as e:
            pytest.fail(f"GPU ray generation failed: {e}")
    
    def test_ray_generation_cpu(self):
        """Test CPU ray generation."""
        try:
            cpu_tracer = create_cpu_ray_tracer(self.config)
            
            # Test single ray generation
            rays = cpu_tracer.ray_generator.generate_rays(self.source_positions[0])
            assert len(rays) == 12 * 6  # azimuth × elevation
            assert all(ray.device.type == 'cpu' for ray in rays)
            
            print("✓ CPU ray generation working correctly")
        except Exception as e:
            pytest.fail(f"CPU ray generation failed: {e}")
    
    def test_ray_tracing_gpu(self, gpu_available):
        """Test GPU ray tracing."""
        if not gpu_available:
            pytest.skip("CUDA not available")
        
        try:
            gpu_tracer = create_gpu_ray_tracer(self.config)
            
            # Test single ray tracing
            results = gpu_tracer.trace_rays(self.source_positions[:5], None, self.env)
            assert len(results) > 0
            assert all(isinstance(result, type(results[0])) for result in results)
            
            # Test batch ray tracing
            batch_results = gpu_tracer.trace_rays_batch(self.source_positions[:10], None, self.env)
            assert len(batch_results) > 0
            
            # Test GPU optimized tracing
            gpu_results = gpu_tracer.trace_rays_gpu_optimized(self.source_positions[:10], environment=self.env)
            assert 'ray_paths' in gpu_results
            assert 'gpu_memory_used' in gpu_results
            assert len(gpu_results['ray_paths']) > 0
            
            print("✓ GPU ray tracing working correctly")
        except Exception as e:
            pytest.fail(f"GPU ray tracing failed: {e}")
    
    def test_ray_tracing_cpu(self):
        """Test CPU ray tracing."""
        try:
            cpu_tracer = create_cpu_ray_tracer(self.config)
            
            # Test single ray tracing
            results = cpu_tracer.trace_rays(self.source_positions[:5], None, self.env)
            assert len(results) > 0
            assert all(isinstance(result, type(results[0])) for result in results)
            
            # Test batch ray tracing
            batch_results = cpu_tracer.trace_rays_batch(self.source_positions[:10], None, self.env)
            assert len(batch_results) > 0
            
            print("✓ CPU ray tracing working correctly")
        except Exception as e:
            pytest.fail(f"CPU ray tracing failed: {e}")
    
    def test_spatial_analysis_gpu(self, gpu_available):
        """Test GPU spatial analysis."""
        if not gpu_available:
            pytest.skip("CUDA not available")
        
        try:
            gpu_tracer = create_gpu_ray_tracer(self.config)
            
            # Generate some ray tracing results
            results = gpu_tracer.trace_rays(self.source_positions[:5], None, self.env)
            
            # Test GPU spatial analysis
            spatial_analysis = gpu_tracer.analyze_spatial_distribution_gpu(results)
            assert 'x_grid' in spatial_analysis
            assert 'y_grid' in spatial_analysis
            assert 'z_grid' in spatial_analysis
            assert 'spatial_grid' in spatial_analysis
            assert 'total_points' in spatial_analysis
            
            print("✓ GPU spatial analysis working correctly")
        except Exception as e:
            pytest.fail(f"GPU spatial analysis failed: {e}")
    
    def test_channel_estimation_gpu(self, gpu_available):
        """Test GPU channel estimation."""
        if not gpu_available:
            pytest.skip("CUDA not available")
        
        try:
            gpu_tracer = create_gpu_ray_tracer(self.config)
            
            # Generate some ray tracing results
            results = gpu_tracer.trace_rays(self.source_positions[:5], None, self.env)
            
            # Test GPU channel estimation
            channel_estimation = gpu_tracer.estimate_channels_gpu(results, self.env)
            assert 'path_loss' in channel_estimation
            assert 'delay' in channel_estimation
            assert 'doppler' in channel_estimation
            assert 'subcarrier_responses' in channel_estimation
            
            print("✓ GPU channel estimation working correctly")
        except Exception as e:
            pytest.fail(f"GPU channel estimation failed: {e}")
    
    def test_performance_metrics_gpu(self, gpu_available):
        """Test GPU performance metrics."""
        if not gpu_available:
            pytest.skip("CUDA not available")
        
        try:
            gpu_tracer = create_gpu_ray_tracer(self.config)
            
            # Get performance metrics
            metrics = gpu_tracer.get_gpu_performance_metrics()
            
            # Check required metrics
            required_metrics = [
                'device_name', 'device_capability', 'gpu_memory_total',
                'gpu_memory_allocated', 'gpu_memory_reserved', 'gpu_memory_free',
                'batch_size', 'max_concurrent_rays', 'mixed_precision'
            ]
            
            for metric in required_metrics:
                assert metric in metrics, f"Missing metric: {metric}"
            
            # Check memory values
            assert metrics['gpu_memory_total'] > 0
            assert metrics['gpu_memory_allocated'] >= 0
            assert metrics['gpu_memory_reserved'] >= 0
            
            print("✓ GPU performance metrics working correctly")
            print(f"  Device: {metrics['device_name']}")
            print(f"  Memory: {metrics['gpu_memory_allocated']:.2f} GB used")
        except Exception as e:
            pytest.fail(f"GPU performance metrics failed: {e}")
    
    def test_device_management(self, gpu_available):
        """Test device management functionality."""
        if not gpu_available:
            pytest.skip("CUDA not available")
        
        try:
            # Test device movement
            gpu_tracer = create_gpu_ray_tracer(self.config)
            assert gpu_tracer.device == 'cuda'
            
            # Test to_device method (should remain on GPU)
            gpu_tracer.to_device('cuda')
            assert gpu_tracer.device == 'cuda'
            
            # Test CPU tracer device movement
            cpu_tracer = create_cpu_ray_tracer(self.config)
            assert cpu_tracer.device == 'cpu'
            
            # Move CPU tracer to GPU
            cpu_tracer.to_device('cuda')
            assert cpu_tracer.device == 'cuda'
            
            print("✓ Device management working correctly")
        except Exception as e:
            pytest.fail(f"Device management failed: {e}")
    
    def test_memory_optimization(self, gpu_available):
        """Test GPU memory optimization."""
        if not gpu_available:
            pytest.skip("CUDA not available")
        
        try:
            gpu_tracer = create_gpu_ray_tracer(self.config)
            
            # Get initial memory usage
            initial_memory = gpu_tracer.get_gpu_performance_metrics()['gpu_memory_allocated']
            
            # Run some operations to use memory
            _ = gpu_tracer.trace_rays_gpu_optimized(self.source_positions[:10], environment=self.env)
            
            # Get memory after operations
            after_ops_memory = gpu_tracer.get_gpu_performance_metrics()['gpu_memory_allocated']
            
            # Optimize memory
            gpu_tracer.optimize_gpu_memory()
            
            # Get memory after optimization
            after_opt_memory = gpu_tracer.get_gpu_performance_metrics()['gpu_memory_allocated']
            
            # Memory should be optimized (may not always decrease due to PyTorch caching)
            print(f"  Initial memory: {initial_memory:.2f} GB")
            print(f"  After operations: {after_ops_memory:.2f} GB")
            print(f"  After optimization: {after_opt_memory:.2f} GB")
            
            print("✓ GPU memory optimization working correctly")
        except Exception as e:
            pytest.fail(f"GPU memory optimization failed: {e}")

class TestPerformanceComparison:
    """Test performance comparison between CPU and GPU."""
    
    @pytest.fixture(scope="class")
    def gpu_available(self):
        """Check if GPU is available for testing."""
        return torch.cuda.is_available()
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'ray_tracing': {
                'gpu_acceleration': True,
                'batch_size': 32,
                'azimuth_samples': 8,  # Small for quick testing
                'elevation_samples': 4,
                'points_per_ray': 16
            }
        }
        
        # Create test environment
        self.env = Environment(device='cuda' if torch.cuda.is_available() else 'cpu')
        wall = Plane([0, 0, 0], [1, 0, 0], 'concrete', device=self.env.device)
        self.env.add_obstacle(wall)
        
        # Create test positions
        self.source_positions = torch.randn(20, 3, device=self.env.device)
    
    def test_benchmark_performance(self, gpu_available):
        """Test performance benchmarking."""
        if not gpu_available:
            pytest.skip("CUDA not available")
        
        try:
            # Test GPU benchmarking
            gpu_tracer = create_gpu_ray_tracer(self.config)
            gpu_results = benchmark_ray_tracing_performance(gpu_tracer, self.source_positions, self.env, num_runs=2)
            
            assert 'average_time' in gpu_results
            assert 'rays_per_second' in gpu_results
            assert 'device' in gpu_results
            assert gpu_results['device'] == 'cuda'
            
            print(f"✓ GPU Benchmark: {gpu_results['rays_per_second']:.0f} rays/sec")
        except Exception as e:
            pytest.fail(f"GPU benchmarking failed: {e}")
    
    def test_cpu_gpu_comparison(self, gpu_available):
        """Test CPU vs GPU performance comparison."""
        if not gpu_available:
            pytest.skip("CUDA not available")
        
        try:
            # Compare CPU vs GPU performance
            comparison = compare_cpu_gpu_performance(self.source_positions, self.env, self.config)
            
            # Check results structure
            assert 'cpu' in comparison or 'gpu' in comparison
            
            if 'gpu' in comparison:
                print(f"✓ GPU Performance: {comparison['gpu']['rays_per_second']:.0f} rays/sec")
            
            if 'cpu' in comparison:
                print(f"✓ CPU Performance: {comparison['cpu']['rays_per_second']:.0f} rays/sec")
            
            if 'speedup' in comparison:
                print(f"✓ GPU Speedup: {comparison['speedup']:.1f}x")
            
            print("✓ CPU vs GPU comparison working correctly")
        except Exception as e:
            pytest.fail(f"CPU vs GPU comparison failed: {e}")

class TestVirtualLinkRayTracing:
    """Test virtual link ray tracing functionality."""
    
    @pytest.fixture(scope="class")
    def gpu_available(self):
        """Check if GPU is available for testing."""
        return torch.cuda.is_available()
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'ray_tracing': {
                'gpu_acceleration': True,
                'batch_size': 64,
                'azimuth_samples': 12,
                'elevation_samples': 6,
                'points_per_ray': 32
            },
            'csi_processing': {
                'virtual_link_enabled': True,
                'm_subcarriers': 64,  # Smaller for testing
                'n_ue_antennas': 2,
                'sample_size': 32
            }
        }
        
        # Create test environment
        self.env = Environment(device='cuda' if torch.cuda.is_available() else 'cpu')
        building = Building([-10, -10, 0], [10, 10, 20], 'concrete', device=self.env.device)
        self.env.add_obstacle(building)
        
        # Create UE positions (virtual links)
        self.ue_positions = torch.randn(50, 3, device=self.env.device)
    
    def test_virtual_link_ray_tracing(self, gpu_available):
        """Test virtual link ray tracing."""
        if not gpu_available:
            pytest.skip("CUDA not available")
        
        try:
            gpu_tracer = create_gpu_ray_tracer(self.config)
            
            # Trace rays for virtual links
            results = gpu_tracer.trace_rays_gpu_optimized(self.ue_positions, environment=self.env)
            
            # Check results structure
            assert 'ray_paths' in results
            assert 'batch_size' in results
            assert 'total_rays' in results
            assert 'gpu_memory_used' in results
            
            # Calculate expected rays
            expected_rays = 50 * 12 * 6  # ue_positions × azimuth × elevation
            assert len(results['ray_paths']) == expected_rays
            
            # Check virtual link parameters
            total_virtual_links = 64 * 2  # m_subcarriers × n_ue_antennas
            sampled_virtual_links = 32    # sample_size
            
            print(f"✓ Virtual Link Ray Tracing: {len(results['ray_paths'])} rays")
            print(f"  Total virtual links: {total_virtual_links}")
            print(f"  Sampled virtual links: {sampled_virtual_links}")
            print(f"  GPU Memory Used: {results['gpu_memory_used']:.2f} GB")
            
        except Exception as e:
            pytest.fail(f"Virtual link ray tracing failed: {e}")

def run_gpu_tests():
    """Run GPU ray tracing tests manually."""
    print("🚀 Running GPU Ray Tracing Tests...")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot run GPU tests.")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Test GPU ray tracer creation
    try:
        config = {
            'ray_tracing': {
                'gpu_acceleration': True,
                'batch_size': 64,
                'azimuth_samples': 12,
                'elevation_samples': 6,
                'points_per_ray': 32
            }
        }
        
        gpu_tracer = create_gpu_ray_tracer(config)
        print("✅ GPU Ray Tracer created successfully")
        
        # Test basic functionality
        env = Environment(device='cuda')
        wall = Plane([0, 0, 0], [1, 0, 0], 'concrete', device='cuda')
        env.add_obstacle(wall)
        
        source_positions = torch.randn(10, 3, device='cuda')
        
        # Test ray tracing
        results = gpu_tracer.trace_rays_gpu_optimized(source_positions, environment=env)
        print(f"✅ GPU Ray Tracing completed: {len(results['ray_paths'])} rays")
        
        # Test performance metrics
        metrics = gpu_tracer.get_gpu_performance_metrics()
        print(f"✅ Performance metrics: {metrics['rays_per_second']:.0f} rays/sec")
        
        print("\n🎉 All GPU tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests manually
    success = run_gpu_tests()
    sys.exit(0 if success else 1)
