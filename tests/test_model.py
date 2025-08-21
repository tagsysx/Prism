"""
Tests for the Prism model components.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism.model import (
    PrismModel,
    PrismLoss,
    AttenuationNetwork,
    AttenuationDecoder,
    RadianceNetwork
)

class TestAttenuationDecoder:
    """Test the Attenuation Decoder."""
    
    def test_initialization(self):
        """Test Attenuation Decoder initialization."""
        decoder = AttenuationDecoder(feature_dim=128, num_subcarriers=408, num_ue_antennas=4, hidden_dim=256)
        
        assert decoder.feature_dim == 128
        assert decoder.num_subcarriers == 408
        assert decoder.num_ue_antennas == 4
        assert decoder.hidden_dim == 256
        assert len(decoder.channels) == 4
    
    def test_forward_pass(self):
        """Test Attenuation Decoder forward pass."""
        decoder = AttenuationDecoder(feature_dim=128, num_subcarriers=64, num_ue_antennas=2, hidden_dim=256)
        
        batch_size = 8
        input_features = torch.randn(batch_size, 128)
        
        output = decoder(input_features)
        
        assert output.shape == (batch_size, 2, 64)  # [batch_size, num_ue_antennas, num_subcarriers]
        assert output.is_complex()
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestAttenuationNetwork:
    """Test the Attenuation Network."""
    
    def test_initialization(self):
        """Test Attenuation Network initialization."""
        network = AttenuationNetwork(input_dim=3, hidden_dim=256, feature_dim=128)
        
        assert len(network.network) == 9  # 9 total layers: 1 input + 7 hidden + 1 output
        assert network.network[0].in_features == 3
        assert network.network[0].out_features == 256
        assert network.network[-1].out_features == 128
    
    def test_forward_pass(self):
        """Test Attenuation Network forward pass."""
        network = AttenuationNetwork(input_dim=3, hidden_dim=256, feature_dim=128)
        
        batch_size = 16
        input_tensor = torch.randn(batch_size, 3)  # 3D positions
        
        output = network(input_tensor)
        
        assert output.shape == (batch_size, 128)  # Feature dimension
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestRadianceNetwork:
    """Test the Radiance Network."""
    
    def test_initialization(self):
        """Test Radiance Network initialization."""
        network = RadianceNetwork(position_dim=3, view_dim=3, feature_dim=128, 
                                num_subcarriers=408, num_ue_antennas=4, hidden_dim=256)
        
        assert len(network.channels) == 4  # N_UE channels
        input_dim = 3 + 3 + 128  # position + view + features
        assert network.channels[0][0].in_features == input_dim
    
    def test_forward_pass(self):
        """Test Radiance Network forward pass."""
        network = RadianceNetwork(position_dim=3, view_dim=3, feature_dim=128, 
                                num_subcarriers=64, num_ue_antennas=2, hidden_dim=256)
        
        batch_size = 16
        ue_positions = torch.randn(batch_size, 3)
        view_directions = torch.randn(batch_size, 3)
        spatial_features = torch.randn(batch_size, 128)
        
        output = network(ue_positions, view_directions, spatial_features)
        
        assert output.shape == (batch_size, 2, 64)  # [batch_size, num_ue_antennas, num_subcarriers]
        assert output.is_complex()
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
            hidden_dim=256,
            feature_dim=128
        )
        
        assert model.num_subcarriers == 64
        assert model.num_ue_antennas == 2
        assert model.num_bs_antennas == 4
        assert model.position_dim == 3
        assert model.hidden_dim == 256
        assert model.feature_dim == 128
    
    def test_forward_pass(self):
        """Test Prism Model forward pass."""
        model = PrismModel(
            num_subcarriers=32,
            num_ue_antennas=2,
            num_bs_antennas=4,
            position_dim=3,
            hidden_dim=128,
            feature_dim=64
        )
        
        batch_size = 8
        
        # Create sample inputs
        positions = torch.randn(batch_size, 3)
        ue_positions = torch.randn(batch_size, 3)
        view_directions = torch.randn(batch_size, 3)
        
        # Forward pass
        outputs = model(positions, ue_positions, view_directions)
        
        # Check output shapes
        assert outputs['spatial_features'].shape == (batch_size, 64)
        assert outputs['attenuation_factors'].shape == (batch_size, 2, 32)  # [batch_size, num_ue_antennas, num_subcarriers]
        assert outputs['radiation_factors'].shape == (batch_size, 2, 32)   # [batch_size, num_ue_antennas, num_subcarriers]
        assert outputs['mimo_channel'].shape == (batch_size, 2, 4)        # [batch_size, num_ue_antennas, num_bs_antennas]
        
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
            hidden_dim=64,
            feature_dim=32
        )
        
        batch_size = 4
        
        # Create sample inputs
        positions = torch.randn(batch_size, 3)
        ue_positions = torch.randn(batch_size, 3)
        view_directions = torch.randn(batch_size, 3)
        
        # Get MIMO channel matrix
        mimo_matrix = model.get_mimo_channel(
            positions=positions,
            ue_positions=ue_positions,
            view_directions=view_directions
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
        try:
            PrismLoss(loss_type='invalid')
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
    
    def test_mse_loss(self):
        """Test MSE loss computation."""
        loss_fn = PrismLoss(loss_type='mse')
        
        batch_size = 8
        num_ue_antennas = 2
        num_subcarriers = 16
        
        # Create complex predictions and targets
        predictions = {
            'attenuation_factors': torch.randn(batch_size, num_ue_antennas, num_subcarriers) + 1j * torch.randn(batch_size, num_ue_antennas, num_subcarriers),
            'radiation_factors': torch.randn(batch_size, num_ue_antennas, num_subcarriers) + 1j * torch.randn(batch_size, num_ue_antennas, num_subcarriers)
        }
        targets = {
            'attenuation_factors': torch.randn(batch_size, num_ue_antennas, num_subcarriers) + 1j * torch.randn(batch_size, num_ue_antennas, num_subcarriers),
            'radiation_factors': torch.randn(batch_size, num_ue_antennas, num_subcarriers) + 1j * torch.randn(batch_size, num_ue_antennas, num_subcarriers)
        }
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_l1_loss(self):
        """Test L1 loss computation."""
        loss_fn = PrismLoss(loss_type='l1')
        
        batch_size = 8
        num_ue_antennas = 2
        num_subcarriers = 16
        
        # Create complex predictions and targets
        predictions = {
            'attenuation_factors': torch.randn(batch_size, num_ue_antennas, num_subcarriers) + 1j * torch.randn(batch_size, num_ue_antennas, num_subcarriers),
            'radiation_factors': torch.randn(batch_size, num_ue_antennas, num_subcarriers) + 1j * torch.randn(batch_size, num_ue_antennas, num_subcarriers)
        }
        targets = {
            'attenuation_factors': torch.randn(batch_size, num_ue_antennas, num_subcarriers) + 1j * torch.randn(batch_size, num_ue_antennas, num_subcarriers),
            'radiation_factors': torch.randn(batch_size, num_ue_antennas, num_subcarriers) + 1j * torch.randn(batch_size, num_ue_antennas, num_subcarriers)
        }
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_weighted_loss(self):
        """Test weighted loss computation."""
        loss_fn = PrismLoss(loss_type='mse')
        
        batch_size = 8
        num_ue_antennas = 2
        num_subcarriers = 16
        
        # Create complex predictions and targets
        predictions = {
            'attenuation_factors': torch.randn(batch_size, num_ue_antennas, num_subcarriers) + 1j * torch.randn(batch_size, num_ue_antennas, num_subcarriers),
            'radiation_factors': torch.randn(batch_size, num_ue_antennas, num_subcarriers) + 1j * torch.randn(batch_size, num_ue_antennas, num_subcarriers)
        }
        targets = {
            'attenuation_factors': torch.randn(batch_size, num_ue_antennas, num_subcarriers) + 1j * torch.randn(batch_size, num_ue_antennas, num_subcarriers),
            'radiation_factors': torch.randn(batch_size, num_ue_antennas, num_subcarriers) + 1j * torch.randn(batch_size, num_ue_antennas, num_subcarriers)
        }
        weights = torch.ones(num_subcarriers)
        
        loss = loss_fn(predictions, targets, weights)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)

if __name__ == '__main__':
    # Run all tests
    print("Running Prism Model Tests...")
    print("=" * 50)
    
    # Test AttenuationDecoder
    print("\n1. Testing AttenuationDecoder...")
    test_decoder = TestAttenuationDecoder()
    test_decoder.test_initialization()
    test_decoder.test_forward_pass()
    print("✓ AttenuationDecoder tests passed")
    
    # Test AttenuationNetwork
    print("\n2. Testing AttenuationNetwork...")
    test_atten = TestAttenuationNetwork()
    test_atten.test_initialization()
    test_atten.test_forward_pass()
    print("✓ AttenuationNetwork tests passed")
    
    # Test RadianceNetwork
    print("\n3. Testing RadianceNetwork...")
    test_rad = TestRadianceNetwork()
    test_rad.test_initialization()
    test_rad.test_forward_pass()
    print("✓ RadianceNetwork tests passed")
    
    # Test PrismModel
    print("\n4. Testing PrismModel...")
    test_model = TestPrismModel()
    test_model.test_initialization()
    test_model.test_forward_pass()
    test_model.test_mimo_channel()
    print("✓ PrismModel tests passed")
    
    # Test PrismLoss
    print("\n5. Testing PrismLoss...")
    test_loss = TestPrismLoss()
    test_loss.test_initialization()
    test_loss.test_invalid_loss_type()
    test_loss.test_mse_loss()
    test_loss.test_l1_loss()
    test_loss.test_weighted_loss()
    print("✓ PrismLoss tests passed")
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed successfully!")
