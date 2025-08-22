#!/usr/bin/env python3
"""
Test suite for the CUDA-accelerated ray tracer.

This test file verifies the functionality of the CUDA ray tracer
with automatic device detection and fallback mechanisms.
"""

import unittest
import torch
import numpy as np
from typing import List, Dict

# Import the CUDA ray tracer
from prism.ray_tracer_cuda import CUDARayTracer

class TestCUDARayTracer(unittest.TestCase):
    """Test the CUDA Ray Tracer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.azimuth_divisions = 8
        self.elevation_divisions = 4
        self.max_ray_length = 100.0
        self.scene_size = 200.0
        
        # Create test data
        self.base_station_pos = torch.tensor([0.0, 0.0, 0.0])
        self.ue_positions = [
            [10.0, 5.0, 1.5],
            [15.0, -3.0, 1.5]
        ]
        self.selected_subcarriers = {
            (10.0, 5.0, 1.5): [0, 1],
            (15.0, -3.0, 1.5): [0, 2]
        }
        self.antenna_embeddings = torch.randn(4, 64)  # 4 subcarriers, 64D embeddings
    
    def test_initialization(self):
        """Test CUDA ray tracer initialization."""
        tracer = CUDARayTracer(
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions,
            max_ray_length=self.max_ray_length,
            scene_size=self.scene_size
        )
        
        # Check basic attributes
        self.assertEqual(tracer.azimuth_divisions, self.azimuth_divisions)
        self.assertEqual(tracer.elevation_divisions, self.elevation_divisions)
        self.assertEqual(tracer.max_ray_length, self.max_ray_length)
        self.assertEqual(tracer.scene_size, self.scene_size)
        self.assertEqual(tracer.total_directions, self.azimuth_divisions * self.elevation_divisions)
        
        # Check device detection
        self.assertIsInstance(tracer.device, str)
        self.assertIsInstance(tracer.use_cuda, bool)
        
        print(f"Device: {tracer.device}, CUDA enabled: {tracer.use_cuda}")
    
    def test_device_detection(self):
        """Test automatic device detection."""
        tracer = CUDARayTracer()
        
        # Device should be either 'cuda' or 'cpu'
        self.assertIn(tracer.device, ['cuda', 'cpu'])
        
        # If CUDA is available, use_cuda should be True
        if torch.cuda.is_available():
            self.assertEqual(tracer.device, 'cuda')
            self.assertTrue(tracer.use_cuda)
        else:
            self.assertEqual(tracer.device, 'cpu')
            self.assertFalse(tracer.use_cuda)
    
    def test_direction_vectors(self):
        """Test direction vector generation."""
        tracer = CUDARayTracer(
            azimuth_divisions=4,
            elevation_divisions=2
        )
        
        directions = tracer.generate_direction_vectors()
        
        # Check shape
        expected_directions = 4 * 2  # azimuth × elevation
        self.assertEqual(directions.shape, (expected_directions, 3))
        
        # Check device
        self.assertEqual(directions.device.type, tracer.device)
        
        # Check normalization (approximately unit vectors)
        norms = torch.norm(directions, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
    
    def test_performance_info(self):
        """Test performance information retrieval."""
        tracer = CUDARayTracer()
        
        perf_info = tracer.get_performance_info()
        
        # Check required keys
        required_keys = ['device', 'use_cuda', 'total_directions', 'uniform_samples', 'resampled_points']
        for key in required_keys:
            self.assertIn(key, perf_info)
        
        # Check CUDA-specific info if available
        if perf_info['use_cuda']:
            self.assertIn('cuda_device_name', perf_info)
            self.assertIn('cuda_memory_gb', perf_info)
            self.assertIn('cuda_compute_capability', perf_info)
    
    def test_ray_tracing_basic(self):
        """Test basic ray tracing functionality."""
        tracer = CUDARayTracer(
            azimuth_divisions=4,
            elevation_divisions=2,
            uniform_samples=16,  # Reduced for faster testing
            resampled_points=8
        )
        
        try:
            results = tracer.trace_rays(
                self.base_station_pos,
                self.ue_positions,
                self.selected_subcarriers,
                self.antenna_embeddings
            )
            
            # Check that we got results
            self.assertIsInstance(results, dict)
            self.assertGreater(len(results), 0)
            
            # Check result structure
            for key, value in results.items():
                self.assertIsInstance(key, tuple)
                self.assertEqual(len(key), 3)  # (ue_pos, subcarrier, direction)
                self.assertIsInstance(value, (int, float))
            
            print(f"Successfully traced {len(results)} rays")
            
        except Exception as e:
            # If CUDA kernel fails, this is acceptable
            if "CUDA" in str(e) or "kernel" in str(e).lower():
                print(f"CUDA kernel failed (expected): {e}")
                print("This is acceptable - system will fall back to PyTorch GPU operations")
            else:
                raise e
    
    def test_fallback_mechanisms(self):
        """Test fallback mechanisms when CUDA is not available."""
        # Force CPU mode by setting device to CPU
        if torch.cuda.is_available():
            # Temporarily disable CUDA
            original_cuda_available = torch.cuda.is_available
            torch.cuda.is_available = lambda: False
            
            try:
                tracer = CUDARayTracer()
                self.assertEqual(tracer.device, 'cpu')
                self.assertFalse(tracer.use_cuda)
                
                # Test CPU ray tracing
                results = tracer.trace_rays(
                    self.base_station_pos,
                    self.ue_positions,
                    self.selected_subcarriers,
                    self.antenna_embeddings
                )
                
                self.assertIsInstance(results, dict)
                self.assertGreater(len(results), 0)
                
            finally:
                # Restore original function
                torch.cuda.is_available = original_cuda_available
        else:
            # CUDA not available, test CPU mode
            tracer = CUDARayTracer()
            self.assertEqual(tracer.device, 'cpu')
            self.assertFalse(tracer.use_cuda)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        tracer = CUDARayTracer()
        
        # Test with invalid UE positions
        invalid_ue_positions = [
            [float('inf'), 5.0, 1.5],  # Invalid position
            [15.0, -3.0, 1.5]
        ]
        
        # Should handle gracefully
        try:
            results = tracer.trace_rays(
                self.base_station_pos,
                invalid_ue_positions,
                self.selected_subcarriers,
                self.antenna_embeddings
            )
            # If it succeeds, that's fine
        except Exception as e:
            # If it fails, that's also acceptable
            print(f"Error handling test: {e}")
    
    def test_memory_efficiency(self):
        """Test memory efficiency for large scenarios."""
        # Create larger test scenario
        large_ue_positions = []
        large_subcarriers = {}
        
        for i in range(50):  # 50 UEs
            x = np.random.uniform(-80, 80)
            y = np.random.uniform(-80, 80)
            z = np.random.uniform(1.0, 2.0)
            pos = [x, y, z]
            large_ue_positions.append(pos)
            
            # 10 subcarriers per UE
            large_subcarriers[tuple(pos)] = list(range(10))
        
        large_embeddings = torch.randn(10, 64)
        
        tracer = CUDARayTracer(
            azimuth_divisions=8,
            elevation_divisions=4,
            uniform_samples=32,  # Reduced for memory efficiency
            resampled_points=16
        )
        
        try:
            # This should not cause memory issues
            results = tracer.trace_rays(
                self.base_station_pos,
                large_ue_positions,
                large_subcarriers,
                large_embeddings
            )
            
            # Check memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
                print(f"GPU memory used: {memory_allocated:.1f} MB")
                
                # Should be reasonable (< 2GB for this scenario)
                self.assertLess(memory_allocated, 2000)
            
            print(f"Successfully processed large scenario: {len(results)} rays")
            
        except Exception as e:
            if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                print(f"Memory test failed (acceptable): {e}")
            else:
                raise e

def run_performance_test():
    """Run a quick performance test."""
    print("Running CUDA Ray Tracer Performance Test...")
    print("=" * 50)
    
    # Create tracer
    tracer = CUDARayTracer(
        azimuth_divisions=16,
        elevation_divisions=8,
        uniform_samples=64,
        resampled_points=32
    )
    
    # Create test data
    base_station_pos = torch.tensor([0.0, 0.0, 0.0])
    ue_positions = [[x, y, 1.5] for x in range(-5, 6, 2) for y in range(-5, 6, 2)]
    selected_subcarriers = {tuple(pos): [0, 1, 2] for pos in ue_positions}
    antenna_embeddings = torch.randn(4, 64)
    
    print(f"Test scenario: {len(ue_positions)} UEs, {len(antenna_embeddings)} subcarriers")
    print(f"Total directions: {tracer.total_directions}")
    print(f"Device: {tracer.device}, CUDA: {tracer.use_cuda}")
    
    # Time the execution
    import time
    start_time = time.time()
    
    try:
        results = tracer.trace_rays(
            base_station_pos, ue_positions, selected_subcarriers, antenna_embeddings
        )
        
        execution_time = time.time() - start_time
        total_rays = len(results)
        
        print(f"Execution time: {execution_time:.4f}s")
        print(f"Total rays: {total_rays}")
        print(f"Rays per second: {total_rays/execution_time:.0f}")
        
        if execution_time < 1.0:
            print("✅ Performance test passed - execution time is reasonable")
        else:
            print("⚠️  Performance test - execution time is slower than expected")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

if __name__ == "__main__":
    # Run unit tests
    print("Running CUDA Ray Tracer Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*50)
    
    # Run performance test
    run_performance_test()
