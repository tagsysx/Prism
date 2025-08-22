#!/usr/bin/env python3
"""
Test suite for the cleaned ray_tracer module.

This test file verifies the functionality of the simplified ray tracer
after removing unused classes and methods.
"""

import unittest
import torch
import numpy as np
from typing import List, Dict

# Import the cleaned ray tracer components
from prism.ray_tracer import (
    Ray,
    BaseStation,
    UserEquipment,
    DiscreteRayTracer
)

class TestRay(unittest.TestCase):
    """Test the Ray class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.origin = torch.tensor([0.0, 0.0, 0.0])
        self.direction = torch.tensor([1.0, 0.0, 0.0])
        self.max_length = 100.0
    
    def test_ray_creation(self):
        """Test Ray object creation."""
        ray = Ray(self.origin, self.direction, self.max_length, self.device)
        
        self.assertEqual(ray.device, self.device)
        self.assertEqual(ray.max_length, self.max_length)
        self.assertTrue(torch.allclose(ray.origin, self.origin))
        self.assertTrue(torch.allclose(ray.direction, torch.tensor([1.0, 0.0, 0.0])))
    
    def test_ray_normalization(self):
        """Test that ray direction is properly normalized."""
        # Test with non-normalized direction
        non_normalized = torch.tensor([3.0, 4.0, 0.0])
        ray = Ray(self.origin, non_normalized, self.max_length, self.device)
        
        # Should be normalized to unit vector
        expected_normalized = torch.tensor([0.6, 0.8, 0.0])
        self.assertTrue(torch.allclose(ray.direction, expected_normalized, atol=1e-6))
    
    def test_ray_edge_cases(self):
        """Test edge cases for ray creation."""
        # Test with zero direction vector
        zero_direction = torch.tensor([0.0, 0.0, 0.0])
        ray = Ray(self.origin, zero_direction, self.max_length, self.device)
        
        # Should handle zero direction gracefully
        self.assertTrue(torch.allclose(ray.direction, zero_direction))
        
        # Test with very small direction vector
        small_direction = torch.tensor([1e-12, 1e-12, 1e-12])
        ray = Ray(self.origin, small_direction, self.max_length, self.device)
        
        # Should handle small direction gracefully
        self.assertTrue(torch.allclose(ray.direction, small_direction))


class TestBaseStation(unittest.TestCase):
    """Test the BaseStation class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.position = torch.tensor([10.0, 20.0, 30.0])
        self.num_antennas = 4
    
    def test_base_station_creation(self):
        """Test BaseStation object creation."""
        bs = BaseStation(self.position, self.num_antennas, self.device)
        
        self.assertEqual(bs.device, self.device)
        self.assertEqual(bs.num_antennas, self.num_antennas)
        self.assertTrue(torch.allclose(bs.position, self.position))
        self.assertEqual(bs.antenna_embeddings.shape, (self.num_antennas, 128))
    
    def test_base_station_default_position(self):
        """Test BaseStation with default position."""
        bs = BaseStation(num_antennas=2, device=self.device)
        
        expected_position = torch.tensor([0.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(bs.position, expected_position))
        self.assertEqual(bs.num_antennas, 2)
    
    def test_antenna_embedding_retrieval(self):
        """Test antenna embedding retrieval."""
        bs = BaseStation(self.position, self.num_antennas, self.device)
        
        # Test first antenna
        embedding_0 = bs.get_antenna_embedding(0)
        self.assertEqual(embedding_0.shape, (128,))
        
        # Test second antenna
        embedding_1 = bs.get_antenna_embedding(1)
        self.assertEqual(embedding_1.shape, (128,))
        
        # Test that different antennas have different embeddings
        self.assertFalse(torch.allclose(embedding_0, embedding_1))


class TestUserEquipment(unittest.TestCase):
    """Test the UserEquipment class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.position = torch.tensor([5.0, 15.0, 2.0])
    
    def test_user_equipment_creation(self):
        """Test UserEquipment object creation."""
        ue = UserEquipment(self.position, self.device)
        
        self.assertTrue(torch.allclose(ue.position, self.position))
    
    def test_position_cloning(self):
        """Test that position is properly cloned and detached."""
        ue = UserEquipment(self.position, self.device)
        
        # Modify original position
        original_position = self.position.clone()
        self.position[0] = 999.0
        
        # UE position should remain unchanged
        self.assertTrue(torch.allclose(ue.position, original_position))
        self.assertFalse(torch.allclose(ue.position, self.position))


class TestDiscreteRayTracer(unittest.TestCase):
    """Test the DiscreteRayTracer class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.azimuth_divisions = 16
        self.elevation_divisions = 8
        self.max_ray_length = 100.0
        self.scene_size = 200.0
    
    def test_ray_tracer_creation(self):
        """Test DiscreteRayTracer object creation."""
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions,
            max_ray_length=self.max_ray_length,
            scene_size=self.scene_size,
            device=self.device
        )
        
        self.assertEqual(ray_tracer.device, self.device)
        self.assertEqual(ray_tracer.azimuth_divisions, self.azimuth_divisions)
        self.assertEqual(ray_tracer.elevation_divisions, self.elevation_divisions)
        self.assertEqual(ray_tracer.max_ray_length, self.max_ray_length)
        self.assertEqual(ray_tracer.scene_size, self.scene_size)
        self.assertEqual(ray_tracer.total_directions, self.azimuth_divisions * self.elevation_divisions)
    
    def test_scene_boundaries(self):
        """Test scene boundary calculations."""
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=8,
            elevation_divisions=4,
            scene_size=100.0,
            device=self.device
        )
        
        expected_min = -50.0
        expected_max = 50.0
        
        self.assertEqual(ray_tracer.scene_min, expected_min)
        self.assertEqual(ray_tracer.scene_max, expected_max)
    
    def test_scene_validation(self):
        """Test scene configuration validation."""
        # Test with valid parameters
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=8,
            elevation_divisions=4,
            scene_size=100.0,
            max_ray_length=50.0,
            device=self.device
        )
        
        # Should not raise any errors
        self.assertIsNotNone(ray_tracer)
        
        # Test with invalid scene size
        with self.assertRaises(ValueError):
            DiscreteRayTracer(
                azimuth_divisions=8,
                elevation_divisions=4,
                scene_size=-100.0,  # Invalid negative size
                device=self.device
            )
        
        # Test with invalid divisions
        with self.assertRaises(ValueError):
            DiscreteRayTracer(
                azimuth_divisions=-1,  # Invalid negative divisions
                elevation_divisions=4,
                scene_size=100.0,
                device=self.device
            )
    
    def test_position_validation(self):
        """Test position validation within scene."""
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=8,
            elevation_divisions=4,
            scene_size=100.0,
            device=self.device
        )
        
        # Test positions within scene
        valid_positions = [
            torch.tensor([0.0, 0.0, 0.0]),      # Center
            torch.tensor([25.0, 30.0, 15.0]),   # Within bounds
            torch.tensor([-40.0, -45.0, -30.0]) # Within bounds
        ]
        
        for pos in valid_positions:
            self.assertTrue(ray_tracer.is_position_in_scene(pos))
        
        # Test positions outside scene
        invalid_positions = [
            torch.tensor([60.0, 70.0, 80.0]),   # Outside bounds
            torch.tensor([-60.0, -70.0, -80.0]), # Outside bounds
            torch.tensor([100.0, 0.0, 0.0])     # On boundary (should be valid)
        ]
        
        for pos in invalid_positions[:-1]:  # Exclude boundary case
            self.assertFalse(ray_tracer.is_position_in_scene(pos))
    
    def test_direction_vector_generation(self):
        """Test direction vector generation."""
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=4,
            elevation_divisions=2,
            device=self.device
        )
        
        directions = ray_tracer.generate_direction_vectors()
        
        # Should generate 4 * 2 = 8 direction vectors
        self.assertEqual(directions.shape, (8, 3))
        
        # Each direction should be a unit vector
        for i in range(directions.shape[0]):
            norm = torch.norm(directions[i])
            self.assertAlmostEqual(norm.item(), 1.0, places=6)
    
    def test_ray_sampling(self):
        """Test ray point sampling."""
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=8,
            elevation_divisions=4,
            scene_size=100.0,
            max_ray_length=50.0,
            device=self.device
        )
        
        # Create a ray
        origin = torch.tensor([0.0, 0.0, 10.0])
        direction = torch.tensor([1.0, 0.0, 0.0])
        ray = Ray(origin, direction, max_length=50.0, device=self.device)
        
        # Sample points along ray
        ue_pos = torch.tensor([25.0, 0.0, 10.0])
        sampled_positions = ray_tracer._sample_ray_points(ray, ue_pos, num_samples=10)
        
        # Should have valid positions
        self.assertGreater(len(sampled_positions), 0)
        self.assertLessEqual(len(sampled_positions), 10)
        
        # All positions should be within scene bounds
        for pos in sampled_positions:
            self.assertTrue(ray_tracer.is_position_in_scene(pos))
    
    def test_importance_weight_computation(self):
        """Test importance weight computation."""
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=8,
            elevation_divisions=4,
            device=self.device
        )
        
        # Create sample attenuation factors
        attenuation_factors = torch.tensor([0.1, 0.5, 1.0, 0.3, 0.8])
        
        weights = ray_tracer._compute_importance_weights(attenuation_factors)
        
        # Weights should sum to 1
        self.assertAlmostEqual(torch.sum(weights).item(), 1.0, places=6)
        
        # Higher attenuation should have higher weights
        self.assertGreater(weights[2], weights[0])  # 1.0 > 0.1
        self.assertGreater(weights[4], weights[1])  # 0.8 > 0.5
    
    def test_importance_based_resampling(self):
        """Test importance-based resampling."""
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=8,
            elevation_divisions=4,
            device=self.device
        )
        
        # Create uniform positions and weights
        uniform_positions = torch.randn(20, 3)
        importance_weights = torch.softmax(torch.randn(20), dim=0)
        
        # Resample to 10 positions
        resampled = ray_tracer._importance_based_resampling(
            uniform_positions, importance_weights, num_samples=10
        )
        
        # Should have correct number of samples
        self.assertEqual(resampled.shape, (10, 3))
        
        # All resampled positions should be from original set
        for pos in resampled:
            # Check if this position exists in original set (with tolerance)
            found = False
            for orig_pos in uniform_positions:
                if torch.allclose(pos, orig_pos, atol=1e-6):
                    found = True
                    break
            self.assertTrue(found, f"Position {pos} not found in original set")
    
    def test_scene_size_update(self):
        """Test dynamic scene size update."""
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=8,
            elevation_divisions=4,
            scene_size=100.0,
            max_ray_length=50.0,
            device=self.device
        )
        
        original_size = ray_tracer.get_scene_size()
        original_bounds = ray_tracer.get_scene_bounds()
        
        # Update scene size
        new_size = 200.0
        ray_tracer.update_scene_size(new_size)
        
        # Check updates
        self.assertEqual(ray_tracer.get_scene_size(), new_size)
        new_bounds = ray_tracer.get_scene_bounds()
        self.assertNotEqual(original_bounds, new_bounds)
        
        # Check that max ray length was adjusted if necessary
        if original_size < new_size:
            self.assertGreaterEqual(ray_tracer.max_ray_length, 50.0)
    
    def test_scene_config_retrieval(self):
        """Test scene configuration retrieval."""
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=16,
            elevation_divisions=8,
            scene_size=150.0,
            max_ray_length=75.0,
            device=self.device
        )
        
        config = ray_tracer.get_scene_config()
        
        expected_keys = {
            'scene_size', 'scene_min', 'scene_max', 
            'max_ray_length', 'azimuth_divisions', 'elevation_divisions'
        }
        
        self.assertEqual(set(config.keys()), expected_keys)
        self.assertEqual(config['scene_size'], 150.0)
        self.assertEqual(config['max_ray_length'], 75.0)
        self.assertEqual(config['azimuth_divisions'], 16)
        self.assertEqual(config['elevation_divisions'], 8)


class TestRayTracerIntegration(unittest.TestCase):
    """Test integration between ray tracer components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.ray_tracer = DiscreteRayTracer(
            azimuth_divisions=8,
            elevation_divisions=4,
            scene_size=100.0,
            max_ray_length=50.0,
            device=self.device
        )
        
        self.base_station = BaseStation(
            position=torch.tensor([0.0, 0.0, 10.0]),
            num_antennas=2,
            device=self.device
        )
        
        self.ue_positions = [
            torch.tensor([20.0, 15.0, 2.0]),
            torch.tensor([-10.0, 25.0, 2.0])
        ]
    
    def test_ray_tracing_workflow(self):
        """Test complete ray tracing workflow."""
        # Get antenna embedding
        antenna_embedding = self.base_station.get_antenna_embedding(0)
        
        # Select subcarriers
        selected_subcarriers = self.ray_tracer.select_subcarriers(64, 0.25)
        
        # Trace ray for a specific direction
        direction = (0, 0)  # First azimuth and elevation division
        ray_results = self.ray_tracer.trace_ray(
            self.base_station.position,
            direction,
            self.ue_positions,
            selected_subcarriers,
            antenna_embedding
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
        """Test signal accumulation across directions."""
        antenna_embedding = self.base_station.get_antenna_embedding(0)
        selected_subcarriers = self.ray_tracer.select_subcarriers(64, 0.25)
        
        # Test with fallback method (no prism network)
        accumulated_signals = self.ray_tracer.accumulate_signals(
            self.base_station.position,
            self.ue_positions,
            {0: selected_subcarriers},
            antenna_embedding
        )
        
        # Should have results for each UE-subcarrier combination
        self.assertGreater(len(accumulated_signals), 0)
        
        # Check signal format
        for (ue_pos, subcarrier), signal_strength in accumulated_signals.items():
            self.assertIsInstance(signal_strength, float)
            self.assertGreaterEqual(signal_strength, 0.0)  # Signal should be non-negative
    
    def test_adaptive_ray_tracing(self):
        """Test adaptive ray tracing method."""
        antenna_embedding = self.base_station.get_antenna_embedding(0)
        selected_subcarriers = self.ray_tracer.select_subcarriers(64, 0.25)
        
        # Test adaptive ray tracing
        adaptive_results = self.ray_tracer.adaptive_ray_tracing(
            self.base_station.position,
            antenna_embedding,
            self.ue_positions,
            {0: selected_subcarriers}
        )
        
        # Should return accumulated signals
        self.assertIsInstance(adaptive_results, dict)
        self.assertGreater(len(adaptive_results), 0)
    
    def test_pyramid_ray_tracing(self):
        """Test pyramid ray tracing method."""
        antenna_embedding = self.base_station.get_antenna_embedding(0)
        selected_subcarriers = self.ray_tracer.select_subcarriers(64, 0.25)
        
        # Test pyramid ray tracing
        pyramid_results = self.ray_tracer.pyramid_ray_tracing(
            self.base_station.position,
            self.ue_positions,
            {0: selected_subcarriers},
            antenna_embedding,
            pyramid_levels=2
        )
        
        # Should return accumulated signals
        self.assertIsInstance(pyramid_results, dict)
        self.assertGreater(len(pyramid_results), 0)
    
    def test_ray_count_analysis(self):
        """Test ray count analysis."""
        analysis = self.ray_tracer.get_ray_count_analysis(
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
        self.assertEqual(analysis['total_rays'], 2 * 8 * 4 * 2 * 16)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestRay,
        TestBaseStation,
        TestUserEquipment,
        TestDiscreteRayTracer,
        TestRayTracerIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
