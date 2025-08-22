#!/usr/bin/env python3
"""
Test suite for TrainingInterface with integrated ray_tracer.

This test file verifies the functionality of the TrainingInterface after integrating
the ray_tracer for actual ray tracing and signal strength computation.
"""

import unittest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock

# Import the components
from prism.training_interface import PrismTrainingInterface
from prism.ray_tracer import DiscreteRayTracer
from prism.networks.prism_network import PrismNetwork


class TestTrainingInterfaceIntegration(unittest.TestCase):
    """Test the integrated TrainingInterface with ray_tracer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.num_ue = 3
        self.num_bs_antennas = 2
        self.num_subcarriers = 64
        
        # Create mock PrismNetwork
        self.mock_prism_network = Mock(spec=PrismNetwork)
        self.mock_prism_network.num_subcarriers = self.num_subcarriers
        self.mock_prism_network.azimuth_divisions = 8
        self.mock_prism_network.elevation_divisions = 4
        self.mock_prism_network.top_k_directions = 16
        
        # Mock antenna codebook
        self.mock_prism_network.antenna_codebook = Mock()
        self.mock_prism_network.antenna_codebook.return_value = torch.randn(self.batch_size, 128)
        
        # Mock antenna network
        self.mock_prism_network.antenna_network = Mock()
        self.mock_prism_network.antenna_network.return_value = torch.randn(self.batch_size, 8, 4)
        self.mock_prism_network.antenna_network.get_top_k_directions = Mock()
        self.mock_prism_network.antenna_network.get_top_k_directions.return_value = (
            torch.randint(0, 8, (self.batch_size, 16, 2)),  # top_k_indices
            torch.rand(self.batch_size, 16)  # importance scores
        )
        
        # Create ray tracer
        self.ray_tracer = DiscreteRayTracer(
            azimuth_divisions=8,
            elevation_divisions=4,
            max_ray_length=50.0,
            scene_size=100.0,
            device=self.device
        )
        
        # Create training interface
        self.training_interface = PrismTrainingInterface(
            prism_network=self.mock_prism_network,
            ray_tracer=self.ray_tracer,
            num_sampling_points=32,
            subcarrier_sampling_ratio=0.3
        )
        
        # Test data
        self.ue_positions = torch.randn(self.batch_size, self.num_ue, 3)
        self.bs_position = torch.zeros(self.batch_size, 3)
        self.antenna_indices = torch.randint(0, 4, (self.batch_size, self.num_bs_antennas))
    
    def test_forward_pass_integration(self):
        """Test forward pass with ray_tracer integration."""
        # Mock ray_tracer.accumulate_signals to return predictable results
        def mock_accumulate_signals(base_station_pos, ue_positions, selected_subcarriers, antenna_embedding):
            results = {}
            for ue_pos in ue_positions:
                ue_pos_tuple = tuple(ue_pos.tolist())
                if isinstance(selected_subcarriers, dict) and ue_pos_tuple in selected_subcarriers:
                    subcarriers = selected_subcarriers[ue_pos_tuple]
                    for k in subcarriers:
                        results[(ue_pos_tuple, k)] = 0.5 + 0.1 * k  # Mock signal strength
                elif isinstance(selected_subcarriers, list):
                    for k in selected_subcarriers:
                        results[(ue_pos_tuple, k)] = 0.5 + 0.1 * k
            return results
        
        self.ray_tracer.accumulate_signals = Mock(side_effect=mock_accumulate_signals)
        
        # Run forward pass
        outputs = self.training_interface.forward(
            ue_positions=self.ue_positions,
            bs_position=self.bs_position,
            antenna_indices=self.antenna_indices,
            return_intermediates=True
        )
        
        # Check outputs
        self.assertIn('csi_predictions', outputs)
        self.assertIn('ray_results', outputs)
        self.assertIn('signal_strengths', outputs)
        self.assertIn('subcarrier_selection', outputs)
        self.assertIn('ray_tracer_results', outputs)
        
        # Check CSI predictions shape
        csi_predictions = outputs['csi_predictions']
        expected_shape = (self.batch_size, self.num_bs_antennas, self.num_ue, 
                         int(self.num_subcarriers * 0.3))
        self.assertEqual(csi_predictions.shape, expected_shape)
        self.assertEqual(csi_predictions.dtype, torch.complex64)
        
        # Verify ray_tracer was called
        self.assertTrue(self.ray_tracer.accumulate_signals.called)
        call_count = self.ray_tracer.accumulate_signals.call_count
        expected_calls = self.batch_size * self.num_bs_antennas
        self.assertEqual(call_count, expected_calls)
    
    def test_signal_strength_to_csi_conversion(self):
        """Test signal strength to CSI conversion."""
        signal_strength = 0.8
        bs_pos = torch.tensor([0.0, 0.0, 0.0])
        ue_pos = torch.tensor([10.0, 0.0, 0.0])
        subcarrier_idx = 5
        
        csi_value = self.training_interface._signal_strength_to_csi(
            signal_strength, bs_pos, ue_pos, subcarrier_idx
        )
        
        # Check that result is complex
        self.assertEqual(csi_value.dtype, torch.complex64)
        
        # Check that amplitude is reasonable
        amplitude = torch.abs(csi_value)
        expected_amplitude = torch.sqrt(torch.tensor(signal_strength))
        self.assertAlmostEqual(amplitude.item(), expected_amplitude.item(), places=5)
    
    def test_subcarrier_selection_format(self):
        """Test that subcarrier selection format is compatible with ray_tracer."""
        selection_info = self.training_interface._select_subcarriers_per_antenna(
            self.batch_size, self.num_ue, self.num_bs_antennas, self.num_subcarriers, 20
        )
        
        # Check selection format
        self.assertIn('selected_indices', selection_info)
        self.assertIn('selection_mask', selection_info)
        self.assertIn('num_selected', selection_info)
        
        selected_indices = selection_info['selected_indices']
        self.assertEqual(selected_indices.shape, (self.batch_size, self.num_bs_antennas, self.num_ue, 20))
        
        # Test conversion to dictionary format for ray_tracer
        for b in range(self.batch_size):
            for bs_antenna in range(self.num_bs_antennas):
                ue_pos_list = [self.ue_positions[b, u].cpu() for u in range(self.num_ue)]
                selected_subcarriers = {}
                
                for u in range(self.num_ue):
                    ue_pos_tuple = tuple(ue_pos_list[u].tolist())
                    selected_subcarriers[ue_pos_tuple] = selected_indices[b, bs_antenna, u].tolist()
                
                # Verify dictionary format
                self.assertEqual(len(selected_subcarriers), self.num_ue)
                for ue_pos_tuple, subcarriers in selected_subcarriers.items():
                    self.assertIsInstance(ue_pos_tuple, tuple)
                    self.assertIsInstance(subcarriers, list)
                    self.assertEqual(len(subcarriers), 20)
    
    def test_loss_computation_with_ray_tracer_results(self):
        """Test loss computation with ray_tracer integrated results."""
        # Mock ray_tracer results
        def mock_accumulate_signals(base_station_pos, ue_positions, selected_subcarriers, antenna_embedding):
            results = {}
            for ue_pos in ue_positions:
                ue_pos_tuple = tuple(ue_pos.tolist())
                if isinstance(selected_subcarriers, dict) and ue_pos_tuple in selected_subcarriers:
                    subcarriers = selected_subcarriers[ue_pos_tuple]
                    for k in subcarriers:
                        results[(ue_pos_tuple, k)] = 0.5 + 0.1 * k
            return results
        
        self.ray_tracer.accumulate_signals = Mock(side_effect=mock_accumulate_signals)
        
        # Run forward pass
        outputs = self.training_interface.forward(
            ue_positions=self.ue_positions,
            bs_position=self.bs_position,
            antenna_indices=self.antenna_indices
        )
        
        # Create mock targets with correct shape (full subcarrier dimension)
        targets = torch.randn(
            self.batch_size, self.num_bs_antennas, self.num_ue, self.num_subcarriers,
            dtype=torch.complex64
        )
        
        # Test loss computation with complex-aware loss function
        def complex_mse_loss(predictions, targets):
            """MSE loss for complex tensors."""
            real_loss = nn.functional.mse_loss(predictions.real, targets.real)
            imag_loss = nn.functional.mse_loss(predictions.imag, targets.imag)
            return real_loss + imag_loss
        
        loss = self.training_interface.compute_loss(
            outputs['csi_predictions'], targets, complex_mse_loss
        )
        
        # Check loss
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreaterEqual(loss.item(), 0.0)  # Non-negative loss
    
    def test_checkpoint_functionality_with_ray_tracer(self):
        """Test checkpoint save/load functionality with ray_tracer integration."""
        import tempfile
        import os
        
        # Update training state
        self.training_interface.update_training_state(epoch=5, batch=100, loss=0.123)
        
        # Save checkpoint (skip this test due to Mock serialization issues)
        # In real usage, PrismNetwork would be a proper nn.Module that can be serialized
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                self.training_interface.checkpoint_dir = temp_dir
                checkpoint_file = "test_checkpoint.pt"
                self.training_interface.save_checkpoint(checkpoint_file)
                
                # If we get here, checkpoint saving worked
                checkpoint_path = os.path.join(temp_dir, checkpoint_file)
                self.assertTrue(os.path.exists(checkpoint_path))
                
        except Exception as e:
            # Expected to fail with Mock objects due to serialization
            self.assertIn("pickle", str(e).lower())
            
        # Test that training state was updated correctly
        self.assertEqual(self.training_interface.current_epoch, 5)
        self.assertEqual(self.training_interface.current_batch, 100)
        self.assertAlmostEqual(self.training_interface.best_loss, 0.123, places=6)
    
    def test_training_phase_configuration(self):
        """Test training phase configuration affects ray_tracer indirectly."""
        # Test different training phases
        for phase in [0, 1, 2]:
            self.training_interface.set_training_phase(phase)
            
            # Verify PrismNetwork configuration was updated
            if phase == 0:
                self.assertEqual(self.mock_prism_network.azimuth_divisions, 8)
                self.assertEqual(self.mock_prism_network.elevation_divisions, 4)
                self.assertEqual(self.mock_prism_network.top_k_directions, 16)
            elif phase == 1:
                self.assertEqual(self.mock_prism_network.azimuth_divisions, 16)
                self.assertEqual(self.mock_prism_network.elevation_divisions, 8)
                self.assertEqual(self.mock_prism_network.top_k_directions, 32)
            elif phase == 2:
                self.assertEqual(self.mock_prism_network.azimuth_divisions, 36)
                self.assertEqual(self.mock_prism_network.elevation_divisions, 18)
                self.assertEqual(self.mock_prism_network.top_k_directions, 64)
    
    def test_training_info_includes_ray_tracer_config(self):
        """Test that training info includes relevant ray_tracer configuration."""
        info = self.training_interface.get_training_info()
        
        # Check basic info
        self.assertIn('num_sampling_points', info)
        self.assertIn('subcarrier_sampling_ratio', info)
        self.assertIn('scene_bounds', info)
        self.assertIn('prism_network_config', info)
        
        # Check values
        self.assertEqual(info['num_sampling_points'], 32)
        self.assertEqual(info['subcarrier_sampling_ratio'], 0.3)
        self.assertIsInstance(info['scene_bounds'], tuple)
        self.assertEqual(len(info['scene_bounds']), 2)  # (min, max)


if __name__ == '__main__':
    unittest.main()
