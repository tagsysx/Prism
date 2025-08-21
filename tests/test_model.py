"""
Tests for the Prism model components.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism.model import (
    PrismModel,
    PrismLoss,
    RFPrismModule,
    AttenuationNetwork,
    RadianceNetwork
)

class TestRFPrismModule:
    """Test the RF Prism Module."""
    
    def test_initialization(self):
        """Test RF Prism Module initialization."""
        module = RFPrismModule(input_dim=512, num_subcarriers=1024, hidden_dim=256)
        
        assert module.num_subcarriers == 1024
        assert module.hidden_dim == 256
        assert len(module.layer1) == 1024
        assert len(module.layer2) == 1024
        assert len(module.output_layers) == 1024
    
    def test_forward_pass(self):
        """Test RF Prism Module forward pass."""
        module = RFPrismModule(input_dim=512, num_subcarriers=64, hidden_dim=256)
        
        batch_size = 8
        input_tensor = torch.randn(batch_size, 512)
        
        output = module(input_tensor)
        
        assert output.shape == (batch_size, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestAttenuationNetwork:
    """Test the Attenuation Network."""
    
    def test_initialization(self):
        """Test Attenuation Network initialization."""
        network = AttenuationNetwork(input_dim=100, hidden_dim=256, num_layers=8)
        
        assert len(network.network) == 8
        assert network.network[0].in_features == 100
        assert network.network[0].out_features == 256
    
    def test_forward_pass(self):
        """Test Attenuation Network forward pass."""
        network = AttenuationNetwork(input_dim=100, hidden_dim=256, num_layers=4)
        
        batch_size = 16
        input_tensor = torch.randn(batch_size, 100)
        
        output = network(input_tensor)
        
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestRadianceNetwork:
    """Test the Radiance Network."""
    
    def test_initialization(self):
        """Test Radiance Network initialization."""
        network = RadianceNetwork(input_dim=100, hidden_dim=256, num_layers=8)
        
        assert len(network.network) == 8
        assert network.network[0].in_features == 100
        assert network.network[0].out_features == 256
    
    def test_forward_pass(self):
        """Test Radiance Network forward pass."""
        network = RadianceNetwork(input_dim=100, hidden_dim=256, num_layers=4)
        
        batch_size = 16
        input_tensor = torch.randn(batch_size, 100)
        
        output = network(input_tensor)
        
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestPrismModel:
    """Test the main Prism Model."""
    
    def test_initialization(self):
        """Test Prism Model initialization."""
        model = PrismModel(
            num_subcarriers=64,
            num_ue_antennas=2,
            num_bs_antennas=4,
            position_dim=3,
            hidden_dim=256
        )
        
        assert model.num_subcarriers == 64
        assert model.num_ue_antennas == 2
        assert model.num_bs_antennas == 4
        assert model.position_dim == 3
        assert model.hidden_dim == 256
    
    def test_forward_pass(self):
        """Test Prism Model forward pass."""
        model = PrismModel(
            num_subcarriers=32,
            num_ue_antennas=2,
            num_bs_antennas=4,
            position_dim=3,
            hidden_dim=128
        )
        
        batch_size = 8
        
        # Create sample inputs
        positions = torch.randn(batch_size, 3)
        ue_antennas = torch.randn(batch_size, 2)
        bs_antennas = torch.randn(batch_size, 4)
        additional_features = torch.randn(batch_size, 10)
        
        # Forward pass
        outputs = model(positions, ue_antennas, bs_antennas, additional_features)
        
        # Check output shapes
        assert outputs['subcarrier_responses'].shape == (batch_size, 32)
        assert outputs['mimo_channel'].shape == (batch_size, 8)  # 2 * 4
        assert outputs['attenuation_features'].shape == (batch_size, 128)
        assert outputs['radiance_features'].shape == (batch_size, 128)
        
        # Check for valid outputs
        for key, value in outputs.items():
            assert not torch.isnan(value).any()
            assert not torch.isinf(value).any()
    
    def test_subcarrier_response(self):
        """Test getting specific subcarrier response."""
        model = PrismModel(
            num_subcarriers=16,
            num_ue_antennas=1,
            num_bs_antennas=2,
            position_dim=3,
            hidden_dim=64
        )
        
        batch_size = 4
        
        # Create sample inputs
        positions = torch.randn(batch_size, 3)
        ue_antennas = torch.randn(batch_size, 1)
        bs_antennas = torch.randn(batch_size, 2)
        additional_features = torch.randn(batch_size, 10)
        
        # Get response for specific subcarrier
        subcarrier_idx = 5
        response = model.get_subcarrier_response(
            subcarrier_idx,
            positions=positions,
            ue_antennas=ue_antennas,
            bs_antennas=bs_antennas,
            additional_features=additional_features
        )
        
        assert response.shape == (batch_size, 1)
        assert not torch.isnan(response).any()
    
    def test_mimo_channel(self):
        """Test getting MIMO channel matrix."""
        model = PrismModel(
            num_subcarriers=16,
            num_ue_antennas=2,
            num_bs_antennas=3,
            position_dim=3,
            hidden_dim=64
        )
        
        batch_size = 4
        
        # Create sample inputs
        positions = torch.randn(batch_size, 3)
        ue_antennas = torch.randn(batch_size, 2)
        bs_antennas = torch.randn(batch_size, 3)
        additional_features = torch.randn(batch_size, 10)
        
        # Get MIMO channel matrix
        mimo_matrix = model.get_mimo_channel(
            positions=positions,
            ue_antennas=ue_antennas,
            bs_antennas=bs_antennas,
            additional_features=additional_features
        )
        
        assert mimo_matrix.shape == (batch_size, 2, 3)
        assert not torch.isnan(mimo_matrix).any()

class TestPrismLoss:
    """Test the Prism Loss function."""
    
    def test_initialization(self):
        """Test Prism Loss initialization."""
        loss_fn = PrismLoss(loss_type='mse')
        assert loss_fn.loss_type == 'mse'
        
        loss_fn = PrismLoss(loss_type='l1')
        assert loss_fn.loss_type == 'l1'
    
    def test_invalid_loss_type(self):
        """Test that invalid loss type raises error."""
        with pytest.raises(ValueError, match="Unsupported loss type"):
            PrismLoss(loss_type='invalid')
    
    def test_mse_loss(self):
        """Test MSE loss computation."""
        loss_fn = PrismLoss(loss_type='mse')
        
        batch_size = 8
        num_subcarriers = 16
        
        predictions = torch.randn(batch_size, num_subcarriers)
        targets = torch.randn(batch_size, num_subcarriers)
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_l1_loss(self):
        """Test L1 loss computation."""
        loss_fn = PrismLoss(loss_type='l1')
        
        batch_size = 8
        num_subcarriers = 16
        
        predictions = torch.randn(batch_size, num_subcarriers)
        targets = torch.randn(batch_size, num_subcarriers)
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_weighted_loss(self):
        """Test weighted loss computation."""
        loss_fn = PrismLoss(loss_type='mse')
        
        batch_size = 8
        num_subcarriers = 16
        
        predictions = torch.randn(batch_size, num_subcarriers)
        targets = torch.randn(batch_size, num_subcarriers)
        weights = torch.ones(num_subcarriers)
        
        loss = loss_fn(predictions, targets, weights)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)

if __name__ == '__main__':
    pytest.main([__file__])
