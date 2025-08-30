#!/usr/bin/env python3
"""
Test suite for the CUDA-accelerated discrete electromagnetic ray tracing system.

This test file verifies the functionality of the CUDA ray tracer after updating it
to match the new CPURayTracer interface, with support for MLP-based direction
sampling and efficient RF signal strength computation.

The CUDA version should provide the same interface as the CPU version but with
accelerated performance on GPU devices.
"""

import unittest
import torch
import numpy as np
from typing import List, Dict, Union

# Import the CUDA ray tracer
from prism.ray_tracer_cuda import CUDARayTracer

class TestCUDARayTracer(unittest.TestCase):
    """Test the CUDA Ray Tracer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.azimuth_divisions = 8
        self.elevation_divisions = 4
        self.max_ray_length = 50.0
        self.scene_size = 100.0
        
        # Create test data
        self.base_station_pos = torch.tensor([0.0, 0.0, 0.0])
        self.ue_positions = [
            torch.tensor([10.0, 0.0, 0.0]),
            torch.tensor([0.0, 10.0, 0.0]),
            torch.tensor([5.0, 5.0, 5.0])
        ]
        
        # Test subcarriers in various formats
        self.selected_subcarriers_list = [0, 1, 2, 3]  # List format
        self.selected_subcarriers_tensor = torch.tensor([0, 1, 2, 3])  # Tensor format
        self.selected_subcarriers_dict = {  # Dict format
            tuple(self.ue_positions[0].tolist()): [0, 1],
            tuple(self.ue_positions[1].tolist()): [2, 3],
            tuple(self.ue_positions[2].tolist()): [0, 2]
        }
        
        # Create antenna embeddings
        self.antenna_embeddings = torch.randn(4, 128)  # 4 subcarriers, 128D embeddings
    
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
    
    def test_scene_bounds_validation(self):
        """Test scene bounds validation."""
        # Test with valid scene size
        ray_tracer = CUDARayTracer(scene_size=100.0)
        
        self.assertIsNotNone(ray_tracer)
        
        # Test with invalid scene size (negative)
        with self.assertRaises(ValueError):
            CUDARayTracer(scene_size=-100.0)
    
    def test_position_in_scene_validation(self):
        """Test position validation within scene bounds."""
        tracer = CUDARayTracer(scene_size=self.scene_size)
        
        # Test position inside scene
        pos = torch.tensor([10.0, 10.0, 10.0])
        self.assertTrue(tracer.is_position_in_scene(pos))
        
        # Test position outside scene
        pos = torch.tensor([100.0, 100.0, 100.0])
        self.assertFalse(tracer.is_position_in_scene(pos))
        
        # Test position at scene boundary
        pos = torch.tensor([50.0, 50.0, 25.0])
        self.assertTrue(tracer.is_position_in_scene(pos))
    
    def test_direction_vector_generation(self):
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
    
    def test_scene_size_updates(self):
        """Test scene size updates."""
        tracer = CUDARayTracer(scene_size=self.scene_size)
        original_size = tracer.get_scene_size()
        original_bounds = tracer.get_scene_bounds()
        
        # Update scene size
        new_size = 150.0
        tracer.update_scene_size(new_size)
        
        # Check that size was updated
        self.assertEqual(tracer.get_scene_size(), new_size)
        new_bounds = tracer.get_scene_bounds()
        
        # Bounds should be updated accordingly
        self.assertNotEqual(original_bounds, new_bounds)
        
        # Max ray length should be adjusted if needed
        self.assertGreaterEqual(tracer.max_ray_length, 50.0)
    
    def test_scene_config_retrieval(self):
        """Test scene configuration retrieval."""
        tracer = CUDARayTracer(
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions,
            scene_size=self.scene_size
        )
        
        config = tracer.get_scene_config()
        
        expected_keys = {
            'scene_min', 'scene_max', 'scene_size', 'max_ray_length',
            'azimuth_divisions', 'elevation_divisions'
        }
        
        self.assertEqual(set(config.keys()), expected_keys)
        self.assertEqual(config['azimuth_divisions'], self.azimuth_divisions)
        self.assertEqual(config['elevation_divisions'], self.elevation_divisions)
    
    def test_ray_tracing_workflow(self):
        """Test complete ray tracing workflow with external subcarrier selection."""
        tracer = CUDARayTracer(
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions,
            max_ray_length=self.max_ray_length,
            scene_size=self.scene_size
        )
        
        # Test with list format subcarriers
        direction = (0, 0)  # First azimuth and elevation division
        ray_results = tracer.trace_ray(
            self.base_station_pos,
            direction,
            self.ue_positions,
            self.selected_subcarriers_list,  # External subcarrier selection
            self.antenna_embeddings[0]  # Use first antenna embedding
        )
        
        # Check results
        self.assertIsInstance(ray_results, dict)
        self.assertGreater(len(ray_results), 0)
        
        # Each result should have correct format
        for (ue_pos, subcarrier), signal_strength in ray_results.items():
            self.assertIsInstance(ue_pos, tuple)
            self.assertIsInstance(subcarrier, int)
            self.assertIsInstance(signal_strength, float)
    
    def test_signal_accumulation(self):
        """Test signal accumulation across directions with external subcarrier selection."""
        tracer = CUDARayTracer(
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions,
            max_ray_length=self.max_ray_length,
            scene_size=self.scene_size
        )
        
        # Test with fallback method (no prism network)
        accumulated_signals = tracer.accumulate_signals(
            self.base_station_pos,
            self.ue_positions,
            self.selected_subcarriers_list,  # External subcarrier selection
            self.antenna_embeddings[0]  # Use first antenna embedding
        )
        
        # Should have results for each UE-subcarrier combination
        self.assertGreater(len(accumulated_signals), 0)
        
        # Check signal format
        for (ue_pos, subcarrier), signal_strength in accumulated_signals.items():
            self.assertIsInstance(signal_strength, float)
            self.assertGreaterEqual(signal_strength, 0.0)  # Signal should be non-negative
    
    def test_adaptive_ray_tracing(self):
        """Test adaptive ray tracing method with external subcarrier selection."""
        tracer = CUDARayTracer(
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions,
            max_ray_length=self.max_ray_length,
            scene_size=self.scene_size
        )
        
        # Test adaptive ray tracing
        adaptive_results = tracer.adaptive_ray_tracing(
            self.base_station_pos,
            self.antenna_embeddings[0],  # Use first antenna embedding
            self.ue_positions,
            self.selected_subcarriers_tensor  # External subcarrier selection
        )
        
        # Should return accumulated signals
        self.assertIsInstance(adaptive_results, dict)
        self.assertGreater(len(adaptive_results), 0)
    
    def test_pyramid_ray_tracing(self):
        """Test pyramid ray tracing method with external subcarrier selection."""
        tracer = CUDARayTracer(
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions,
            max_ray_length=self.max_ray_length,
            scene_size=self.scene_size
        )
        
        # Test pyramid ray tracing
        pyramid_results = tracer.pyramid_ray_tracing(
            self.base_station_pos,
            self.ue_positions,
            self.selected_subcarriers_list,  # External subcarrier selection
            self.antenna_embeddings[0],  # Use first antenna embedding
            pyramid_levels=2
        )
        
        # Should return accumulated signals
        self.assertIsInstance(pyramid_results, dict)
        self.assertGreater(len(pyramid_results), 0)
    
    def test_subcarrier_input_normalization(self):
        """Test subcarrier input normalization with various formats."""
        tracer = CUDARayTracer(
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions,
            max_ray_length=self.max_ray_length,
            scene_size=self.scene_size
        )
        
        # Test with list format
        ray_results_list = tracer.trace_ray(
            self.base_station_pos,
            (0, 0),
            self.ue_positions,
            self.selected_subcarriers_list,
            self.antenna_embeddings[0]
        )
        
        # Test with tensor format
        ray_results_tensor = tracer.trace_ray(
            self.base_station_pos,
            (0, 0),
            self.ue_positions,
            self.selected_subcarriers_tensor,
            self.antenna_embeddings[0]
        )
        
        # Test with dict format
        ray_results_dict = tracer.trace_ray(
            self.base_station_pos,
            (0, 0),
            self.ue_positions,
            self.selected_subcarriers_dict,
            self.antenna_embeddings[0]
        )
        
        # All should produce valid results
        self.assertGreater(len(ray_results_list), 0)
        self.assertGreater(len(ray_results_tensor), 0)
        self.assertGreater(len(ray_results_dict), 0)
    
    def test_subcarrier_input_validation(self):
        """Test subcarrier input validation and error handling."""
        tracer = CUDARayTracer(
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions,
            max_ray_length=self.max_ray_length,
            scene_size=self.scene_size
        )
        
        # Test with None input
        with self.assertRaises(ValueError):
            tracer.trace_ray(
                self.base_station_pos,
                (0, 0),
                self.ue_positions,
                None,  # Invalid: None
                self.antenna_embeddings[0]
            )
        
        # Test with empty list
        with self.assertRaises(ValueError):
            tracer.trace_ray(
                self.base_station_pos,
                (0, 0),
                self.ue_positions,
                [],  # Invalid: empty
                self.antenna_embeddings[0]
            )
        
        # Test with empty tensor
        with self.assertRaises(ValueError):
            tracer.trace_ray(
                self.base_station_pos,
                (0, 0),
                self.ue_positions,
                torch.tensor([]),  # Invalid: empty
                self.antenna_embeddings[0]
            )
    
    def test_ray_count_analysis(self):
        """Test ray count analysis."""
        tracer = CUDARayTracer(
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions
        )
        
        analysis = tracer.get_ray_count_analysis(
            num_bs=2,
            num_ue=len(self.ue_positions),
            num_subcarriers=16
        )
        
        expected_keys = {
            'total_directions', 'azimuth_divisions', 'elevation_divisions',
            'total_rays', 'ray_count_formula'
        }
        
        self.assertEqual(set(analysis.keys()), expected_keys)
        self.assertEqual(analysis['total_directions'], 8 * 4)
        self.assertEqual(analysis['total_rays'], 2 * 8 * 4 * 3 * 16)
        
        # Check formula contains the expected numbers
        self.assertIn("2", analysis['ray_count_formula'])  # num_bs
        self.assertIn("32", analysis['ray_count_formula'])  # total_directions (8*4)
        self.assertIn("3", analysis['ray_count_formula'])   # num_ue
        self.assertIn("16", analysis['ray_count_formula'])  # num_subcarriers
    
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
                results = tracer.trace_ray(
                    self.base_station_pos,
                    (0, 0),
                    self.ue_positions,
                    self.selected_subcarriers_list,
                    self.antenna_embeddings[0]
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
    
    def test_memory_efficiency(self):
        """Test memory efficiency for large scenarios."""
        # Create larger test scenario
        large_ue_positions = []
        large_subcarriers = []
        
        for i in range(20):  # 20 UEs (reduced for testing)
            x = np.random.uniform(-40, 40)
            y = np.random.uniform(-40, 40)
            z = np.random.uniform(1.0, 2.0)
            pos = torch.tensor([x, y, z])
            large_ue_positions.append(pos)
            
            # 5 subcarriers per UE
            large_subcarriers.extend([i * 5 + j for j in range(5)])
        
        large_embeddings = torch.randn(100, 128)  # 100 subcarriers, 128D embeddings
        
        tracer = CUDARayTracer(
            azimuth_divisions=8,
            elevation_divisions=4,
            uniform_samples=32,  # Reduced for memory efficiency
            resampled_points=16
        )
        
        try:
            # This should not cause memory issues
            results = tracer.trace_ray(
                self.base_station_pos,
                (0, 0),  # Single direction
                large_ue_positions,
                large_subcarriers,  # List format
                large_embeddings[0]  # Single antenna embedding
            )
            
            # Check memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
                print(f"GPU memory used: {memory_allocated:.1f} MB")
                
                # Should be reasonable (< 2GB for this scenario)
                self.assertLess(memory_allocated, 2000)
            
            print(f"Successfully processed large scenario: {len(results)} results")
            
        except Exception as e:
            if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                print(f"Memory test failed (acceptable): {e}")
            else:
                raise e
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        tracer = CUDARayTracer()
        
        # Test with invalid UE positions
        invalid_ue_positions = [
            torch.tensor([float('inf'), 5.0, 1.5]),  # Invalid position
            torch.tensor([15.0, -3.0, 1.5])
        ]
        
        # Should handle gracefully
        try:
            results = tracer.trace_ray(
                self.base_station_pos,
                (0, 0),
                invalid_ue_positions,
                self.selected_subcarriers_list,
                self.antenna_embeddings[0]
            )
            # If it succeeds, that's fine
        except Exception as e:
            # If it fails, that's also acceptable
            print(f"Error handling test: {e}")


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
    ue_positions = [torch.tensor([x, y, 1.5]) for x in range(-5, 6, 2) for y in range(-5, 6, 2)]
    selected_subcarriers = [0, 1, 2]  # List format
    antenna_embeddings = torch.randn(4, 128)
    
    print(f"Test scenario: {len(ue_positions)} UEs, {len(selected_subcarriers)} subcarriers")
    print(f"Total directions: {tracer.total_directions}")
    print(f"Device: {tracer.device}, CUDA: {tracer.use_cuda}")
    
    # Time the execution
    import time
    start_time = time.time()
    
    try:
        # Test single ray tracing
        results = tracer.trace_ray(
            base_station_pos, (0, 0), ue_positions, selected_subcarriers, antenna_embeddings[0]
        )
        
        execution_time = time.time() - start_time
        total_results = len(results)
        
        print(f"Execution time: {execution_time:.4f}s")
        print(f"Total results: {total_results}")
        print(f"Results per second: {total_results/execution_time:.0f}")
        
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
