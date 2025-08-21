import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class RFPrismModule(nn.Module):
    """
    RF Prism Module: Multi-channel MLP for decomposing global features into subcarrier components.
    This is the core innovation that enables wideband RF signal processing.
    """
    def __init__(self, input_dim: int, num_subcarriers: int, hidden_dim: int = 256):
        super(RFPrismModule, self).__init__()
        self.num_subcarriers = num_subcarriers
        self.hidden_dim = hidden_dim
        
        # Multi-channel MLP with C channels for C subcarriers
        self.layer1 = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_subcarriers)
        ])
        self.layer2 = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_subcarriers)
        ])
        
        # Output layer for each subcarrier
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_subcarriers)
        ])
        
        # Activation functions
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RF Prism Module.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, num_subcarriers]
        """
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(self.num_subcarriers):
            # Process each subcarrier independently
            h = self.layer1[i](x)
            h = self.activation(h)
            h = self.layer2[i](h)
            h = self.activation(h)
            h = self.output_layers[i](h)
            outputs.append(h)
        
        # Stack outputs for all subcarriers
        return torch.cat(outputs, dim=1)  # [batch_size, num_subcarriers]

class AttenuationNetwork(nn.Module):
    """
    Attenuation Network: Predicts signal attenuation based on position and environment.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 8):
        super(AttenuationNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.network = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.network:
            x = self.activation(layer(x))
        return x

class RadianceNetwork(nn.Module):
    """
    Radiance Network: Predicts signal radiance characteristics.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 8):
        super(RadianceNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.network = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.network:
            x = self.activation(layer(x))
        return x

class PrismModel(nn.Module):
    """
    Prism: Wideband RF Neural Radiance Fields for OFDM Communication.
    
    This model extends NeRF2 to handle wideband RF signals with:
    - M subcarriers (configurable from 52 to 1024+)
    - N_UE antennas at the User Equipment
    - N_BS antennas at the Base Station
    - RF Prism Module for subcarrier decomposition
    """
    
    def __init__(self, 
                 num_subcarriers: int = 1024,
                 num_ue_antennas: int = 2,
                 num_bs_antennas: int = 4,
                 position_dim: int = 3,
                 hidden_dim: int = 256):
        super(PrismModel, self).__init__()
        
        self.num_subcarriers = num_subcarriers
        self.num_ue_antennas = num_ue_antennas
        self.num_bs_antennas = num_bs_antennas
        self.position_dim = position_dim
        self.hidden_dim = hidden_dim
        
        # Input dimension: position + UE antenna + BS antenna + additional features
        input_dim = position_dim + num_ue_antennas + num_bs_antennas + 10  # 10 for additional RF features
        
        # Core networks
        self.attenuation_net = AttenuationNetwork(input_dim, hidden_dim)
        self.radiance_net = RadianceNetwork(input_dim, hidden_dim)
        
        # RF Prism Module: decomposes global features into subcarrier components
        prism_input_dim = hidden_dim * 2  # Concatenated features from both networks
        self.rf_prism = RFPrismModule(prism_input_dim, num_subcarriers, hidden_dim)
        
        # MIMO channel matrix output
        self.mimo_output = nn.Linear(num_subcarriers, num_ue_antennas * num_bs_antennas)
        
    def forward(self, positions: torch.Tensor, ue_antennas: torch.Tensor, 
                bs_antennas: torch.Tensor, additional_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Prism model.
        
        Args:
            positions: Position tensor [batch_size, position_dim]
            ue_antennas: UE antenna features [batch_size, num_ue_antennas]
            bs_antennas: BS antenna features [batch_size, num_bs_antennas]
            additional_features: Additional RF features [batch_size, 10]
            
        Returns:
            Dictionary containing:
            - subcarrier_responses: [batch_size, num_subcarriers]
            - mimo_channel: [batch_size, num_ue_antennas * num_bs_antennas]
            - attenuation_features: [batch_size, hidden_dim]
            - radiance_features: [batch_size, hidden_dim]
        """
        # Concatenate input features
        x = torch.cat([positions, ue_antennas, bs_antennas, additional_features], dim=1)
        
        # Extract features from core networks
        attenuation_features = self.attenuation_net(x)
        radiance_features = self.radiance_net(x)
        
        # Concatenate features for RF Prism Module
        combined_features = torch.cat([attenuation_features, radiance_features], dim=1)
        
        # Decompose into subcarrier components
        subcarrier_responses = self.rf_prism(combined_features)
        
        # Generate MIMO channel matrix
        mimo_channel = self.mimo_output(subcarrier_responses)
        
        return {
            'subcarrier_responses': subcarrier_responses,
            'mimo_channel': mimo_channel,
            'attenuation_features': attenuation_features,
            'radiance_features': radiance_features
        }
    
    def get_subcarrier_response(self, subcarrier_idx: int, **kwargs) -> torch.Tensor:
        """
        Get response for a specific subcarrier.
        
        Args:
            subcarrier_idx: Index of the subcarrier (0 to num_subcarriers-1)
            **kwargs: Same arguments as forward method
            
        Returns:
            Response for the specified subcarrier
        """
        outputs = self.forward(**kwargs)
        return outputs['subcarrier_responses'][:, subcarrier_idx:subcarrier_idx+1]
    
    def get_mimo_channel(self, **kwargs) -> torch.Tensor:
        """
        Get the MIMO channel matrix.
        
        Args:
            **kwargs: Same arguments as forward method
            
        Returns:
            MIMO channel matrix
        """
        outputs = self.forward(**kwargs)
        return outputs['mimo_channel'].view(-1, self.num_ue_antennas, self.num_bs_antennas)

class PrismLoss(nn.Module):
    """
    Loss function for Prism model training.
    Implements the frequency-aware loss function with independent subcarrier optimization.
    """
    
    def __init__(self, loss_type: str = 'mse'):
        super(PrismLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the frequency-aware loss.
        
        Args:
            predictions: Predicted subcarrier responses [batch_size, num_subcarriers]
            targets: Ground truth subcarrier responses [batch_size, num_subcarriers]
            weights: Optional weights for each subcarrier [num_subcarriers]
            
        Returns:
            Total loss value
        """
        # Compute loss for each subcarrier independently
        per_subcarrier_loss = self.criterion(predictions, targets)  # [batch_size, num_subcarriers]
        
        # Apply weights if provided
        if weights is not None:
            per_subcarrier_loss = per_subcarrier_loss * weights.unsqueeze(0)
        
        # Sum across all subcarriers (maintaining independence)
        total_loss = torch.sum(per_subcarrier_loss)
        
        return total_loss

def create_prism_model(config: Dict) -> PrismModel:
    """
    Factory function to create a Prism model from configuration.
    
    Args:
        config: Configuration dictionary containing model parameters
        
    Returns:
        Configured PrismModel instance
    """
    return PrismModel(
        num_subcarriers=config.get('num_subcarriers', 1024),
        num_ue_antennas=config.get('num_ue_antennas', 2),
        num_bs_antennas=config.get('num_bs_antennas', 4),
        position_dim=config.get('position_dim', 3),
        hidden_dim=config.get('hidden_dim', 256)
    )
