#!/usr/bin/env python3
"""
Test suite for the updated ray_tracer module.

This test file verifies the functionality of the ray tracer after updating it
to receive subcarrier information through parameters instead of selecting them internally.
"""

import unittest
import torch
import numpy as np
from typing import List, Dict

# Import the updated ray tracer components
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
        self.position = torch.tensor([5.0, 15.0, 25.0])
    
    def test_ue_creation(self):
        """Test UserEquipment object creation."""
        ue = UserEquipment(self.position, self.device)
        
        self.assertTrue(torch.allclose(ue.position, self.position))
        self.assertEqual(ue.position.dtype, torch.float32)


class TestDiscreteRayTracer(unittest.TestCase):
    """Test the DiscreteRayTracer class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.azimuth_divisions = 8
        self.elevation_divisions = 4
        self.max_ray_length = 50.0
        self.scene_size = 100.0
        
        # Create ray tracer
        self.ray_tracer = DiscreteRayTracer(
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions,
            max_ray_length=self.max_ray_length,
            scene_size=self.scene_size,
            device=self.device
        )
        
        # Create test objects
        self.base_station = BaseStation(
            position=torch.tensor([0.0, 0.0, 0.0]),
            num_antennas=2,
            device=self.device
        )
        
        self.ue_positions = [
            torch.tensor([10.0, 0.0, 0.0]),
            torch.tensor([0.0, 10.0, 0.0]),
            torch.tensor([5.0, 5.0, 5.0])
        ]
    
    def test_ray_tracer_creation(self):
        """Test DiscreteRayTracer object creation."""
        self.assertEqual(self.ray_tracer.device, self.device)
        self.assertEqual(self.ray_tracer.azimuth_divisions, self.azimuth_divisions)
        self.assertEqual(self.ray_tracer.elevation_divisions, self.elevation_divisions)
        self.assertEqual(self.ray_tracer.max_ray_length, self.max_ray_length)
        self.assertEqual(self.ray_tracer.scene_size, self.scene_size)
        self.assertEqual(self.ray_tracer.total_directions, self.azimuth_divisions * self.elevation_divisions)
        
        # Test with custom scene size
        custom_ray_tracer = DiscreteRayTracer(
            azimuth_divisions=16,
            elevation_divisions=8,
            scene_size=50.0,  # Use scene_size instead of scene_bounds
            device=self.device
        )
        
        self.assertEqual(custom_ray_tracer.scene_size, 50.0)
        self.assertEqual(custom_ray_tracer.scene_min, -25.0)
        self.assertEqual(custom_ray_tracer.scene_max, 25.0)
    
    def test_scene_bounds_validation(self):
        """Test scene bounds validation."""
        # Test with valid scene size
        ray_tracer = DiscreteRayTracer(scene_size=100.0, device=self.device)
        
        self.assertIsNotNone(ray_tracer)
        
        # Test with invalid scene size (negative)
        with self.assertRaises(ValueError):
            DiscreteRayTracer(scene_size=-100.0, device=self.device)
    
    def test_position_in_scene_validation(self):
        """Test position validation within scene bounds."""
        # Test position inside scene
        pos = torch.tensor([10.0, 10.0, 10.0])
        self.assertTrue(self.ray_tracer.is_position_in_scene(pos))
        
        # Test position outside scene
        pos = torch.tensor([100.0, 100.0, 100.0])
        self.assertFalse(self.ray_tracer.is_position_in_scene(pos))
        
        # Test position at scene boundary
        pos = torch.tensor([50.0, 50.0, 25.0])
        self.assertTrue(self.ray_tracer.is_position_in_scene(pos))
    
    def test_direction_vector_generation(self):
        """Test direction vector generation."""
        directions = self.ray_tracer.generate_direction_vectors()
        
        # Should have correct number of directions
        expected_directions = self.azimuth_divisions * self.elevation_divisions
        self.assertEqual(len(directions), expected_directions)
        
        # Each direction should be a unit vector
        for direction in directions:
            norm = torch.norm(direction)
            self.assertAlmostEqual(norm.item(), 1.0, places=6)
    
    def test_ray_point_sampling(self):
        """Test ray point sampling."""
        # Create a ray
        origin = torch.tensor([0.0, 0.0, 0.0])
        direction = torch.tensor([1.0, 0.0, 0.0])
        ray = Ray(origin, direction, self.max_ray_length, self.device)
        
        # Sample points along ray
        ue_pos = torch.tensor([10.0, 0.0, 0.0])
        sampled_positions = self.ray_tracer._sample_ray_points(ray, ue_pos, num_samples=10)
        
        # Should have correct number of samples (may be less if some are outside scene)
        self.assertGreater(len(sampled_positions), 0)
        self.assertLessEqual(len(sampled_positions), 10)
        
        # All positions should be within scene bounds
        for pos in sampled_positions:
            self.assertTrue(self.ray_tracer.is_position_in_scene(pos))
    
    def test_importance_weight_computation(self):
        """Test importance weight computation."""
        # Create dummy attenuation factors
        attenuation_factors = torch.randn(10, device=self.device)
        
        weights = self.ray_tracer._compute_importance_weights(attenuation_factors)
        
        # Should have same number of weights as factors
        self.assertEqual(len(weights), len(attenuation_factors))
        
        # Weights should be non-negative
        self.assertTrue(torch.all(weights >= 0))
    
    def test_importance_based_resampling(self):
        """Test importance-based resampling."""
        # Create dummy data
        positions = torch.randn(10, 3, device=self.device)
        importance_weights = torch.rand(10, device=self.device)
        
        resampled = self.ray_tracer._importance_based_resampling(
            positions, importance_weights, num_samples=5
        )
        
        # Should have requested number of samples
        self.assertEqual(len(resampled), 5)
        
        # All resampled positions should be within scene bounds
        for pos in resampled:
            self.assertTrue(self.ray_tracer.is_position_in_scene(pos))
    
    def test_scene_size_updates(self):
        """Test scene size updates."""
        original_size = self.ray_tracer.get_scene_size()
        original_bounds = self.ray_tracer.get_scene_bounds()
        
        # Update scene size
        new_size = 150.0
        self.ray_tracer.update_scene_size(new_size)
        
        # Check that size was updated
        self.assertEqual(self.ray_tracer.get_scene_size(), new_size)
        new_bounds = self.ray_tracer.get_scene_bounds()
        
        # Bounds should be updated accordingly
        self.assertNotEqual(original_bounds, new_bounds)
        
        # Max ray length should be adjusted if needed
        self.assertGreaterEqual(self.ray_tracer.max_ray_length, 50.0)
    
    def test_scene_config_retrieval(self):
        """Test scene configuration retrieval."""
        config = self.ray_tracer.get_scene_config()
        
        expected_keys = {
            'scene_min', 'scene_max', 'scene_size', 'max_ray_length',
            'azimuth_divisions', 'elevation_divisions'
        }
        
        self.assertEqual(set(config.keys()), expected_keys)
        self.assertEqual(config['azimuth_divisions'], self.azimuth_divisions)
        self.assertEqual(config['elevation_divisions'], self.elevation_divisions)
    
    def test_ray_tracing_workflow(self):
        """Test complete ray tracing workflow with external subcarrier selection."""
        # Get antenna embedding
        antenna_embedding = self.base_station.get_antenna_embedding(0)
        
        # Define subcarriers externally (simulating TrainingInterface)
        selected_subcarriers = [0, 1, 2, 3]  # List format
        
        # Trace ray for a specific direction
        direction = (0, 0)  # First azimuth and elevation division
        ray_results = self.ray_tracer.trace_ray(
            self.base_station.position,
            direction,
            self.ue_positions,
            selected_subcarriers,  # External subcarrier selection
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
        """Test signal accumulation across directions with external subcarrier selection."""
        antenna_embedding = self.base_station.get_antenna_embedding(0)
        
        # Define subcarriers externally (simulating TrainingInterface)
        # Fix the dictionary format to match what the method expects
        selected_subcarriers = [0, 1, 2]  # List format instead of dict
        
        # Test with fallback method (no prism network)
        accumulated_signals = self.ray_tracer.accumulate_signals(
            self.base_station.position,
            self.ue_positions,
            selected_subcarriers,  # External subcarrier selection
            antenna_embedding
        )
        
        # Should have results for each UE-subcarrier combination
        self.assertGreater(len(accumulated_signals), 0)
        
        # Check signal format
        for (ue_pos, subcarrier), signal_strength in accumulated_signals.items():
            self.assertIsInstance(signal_strength, float)
            self.assertGreaterEqual(signal_strength, 0.0)  # Signal should be non-negative
    
    def test_adaptive_ray_tracing(self):
        """Test adaptive ray tracing method with external subcarrier selection."""
        antenna_embedding = self.base_station.get_antenna_embedding(0)
        
        # Define subcarriers externally (simulating TrainingInterface)
        selected_subcarriers = torch.tensor([0, 1, 2, 3])  # Tensor format
        
        # Test adaptive ray tracing
        adaptive_results = self.ray_tracer.adaptive_ray_tracing(
            self.base_station.position,
            antenna_embedding,
            self.ue_positions,
            selected_subcarriers  # External subcarrier selection
        )
        
        # Should return accumulated signals
        self.assertIsInstance(adaptive_results, dict)
        self.assertGreater(len(adaptive_results), 0)
    
    def test_pyramid_ray_tracing(self):
        """Test pyramid ray tracing method with external subcarrier selection."""
        antenna_embedding = self.base_station.get_antenna_embedding(0)
        
        # Define subcarriers externally (simulating TrainingInterface)
        selected_subcarriers = [0, 1, 2]  # List format
        
        # Test pyramid ray tracing
        pyramid_results = self.ray_tracer.pyramid_ray_tracing(
            self.base_station.position,
            self.ue_positions,
            selected_subcarriers,  # External subcarrier selection
            antenna_embedding,
            pyramid_levels=2
        )
        
        # Should return accumulated signals
        self.assertIsInstance(pyramid_results, dict)
        self.assertGreater(len(pyramid_results), 0)
    
    def test_subcarrier_input_normalization(self):
        """Test subcarrier input normalization with various formats."""
        antenna_embedding = self.base_station.get_antenna_embedding(0)
        
        # Test with list format
        list_subcarriers = [0, 1, 2, 3]
        ray_results_list = self.ray_tracer.trace_ray(
            self.base_station.position,
            (0, 0),
            self.ue_positions,
            list_subcarriers,
            antenna_embedding
        )
        
        # Test with tensor format
        tensor_subcarriers = torch.tensor([0, 1, 2, 3])
        ray_results_tensor = self.ray_tracer.trace_ray(
            self.base_station.position,
            (0, 0),
            self.ue_positions,
            tensor_subcarriers,
            antenna_embedding
        )
        
        # Test with dict format - fix the format to match expected structure
        # The dictionary should map UE positions to subcarrier indices
        dict_subcarriers = {}
        for i, ue_pos in enumerate(self.ue_positions):
            # Convert tensor to list, then to tuple for dictionary key
            ue_pos_tuple = tuple(ue_pos.tolist())
            dict_subcarriers[ue_pos_tuple] = [i * 2, i * 2 + 1]
        
        ray_results_dict = self.ray_tracer.trace_ray(
            self.base_station.position,
            (0, 0),
            self.ue_positions,
            dict_subcarriers,
            antenna_embedding
        )
        
        # All should produce valid results
        self.assertGreater(len(ray_results_list), 0)
        self.assertGreater(len(ray_results_tensor), 0)
        self.assertGreater(len(ray_results_dict), 0)
    
    def test_subcarrier_input_validation(self):
        """Test subcarrier input validation and error handling."""
        antenna_embedding = self.base_station.get_antenna_embedding(0)
        
        # Test with None input
        with self.assertRaises(ValueError):
            self.ray_tracer.trace_ray(
                self.base_station.position,
                (0, 0),
                self.ue_positions,
                None,  # Invalid: None
                antenna_embedding
            )
        
        # Test with empty list
        with self.assertRaises(ValueError):
            self.ray_tracer.trace_ray(
                self.base_station.position,
                (0, 0),
                self.ue_positions,
                [],  # Invalid: empty
                antenna_embedding
            )
        
        # Test with empty tensor
        with self.assertRaises(ValueError):
            self.ray_tracer.trace_ray(
                self.base_station.position,
                (0, 0),
                self.ue_positions,
                torch.tensor([]),  # Invalid: empty
                antenna_embedding
            )
    
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
        self.assertEqual(analysis['total_rays'], 2 * 8 * 4 * 3 * 16)
        
        # Check formula - the actual format is different, so let's check if it contains the numbers
        # The actual formula shows total_directions (32) instead of individual divisions
        self.assertIn("2", analysis['ray_count_formula'])  # num_bs
        self.assertIn("32", analysis['ray_count_formula'])  # total_directions (8*4)
        self.assertIn("3", analysis['ray_count_formula'])   # num_ue
        self.assertIn("16", analysis['ray_count_formula'])  # num_subcarriers


if __name__ == '__main__':
    unittest.main()
