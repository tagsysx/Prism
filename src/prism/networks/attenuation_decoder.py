"""
AttenuationDecoder: Converts 128D features into N_UE × K attenuation factors.

This network processes the 128-dimensional feature vector from AttenuationNetwork
and outputs N_UE × K attenuation values for all UE antenna channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AttenuationDecoder(nn.Module):
    """
    AttenuationDecoder: Converts 128D features into N_UE × K attenuation factors.
    
    Architecture: Single network processing all UE antenna channels.
    Network: 128D → 256D → 256D → N_UE × K
    Output: Complex values representing attenuation factors.
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        num_ue_antennas: int = 4,
        num_subcarriers: int = 64,
        num_layers: int = 3,
        activation: str = "relu",
        complex_output: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_ue_antennas = num_ue_antennas
        self.num_subcarriers = num_subcarriers
        self.num_layers = num_layers
        self.complex_output = complex_output
        self.dropout = dropout
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layers
        self._build_network()
        
    def _build_network(self):
        """Build the network architecture."""
        
        layers = []
        
        # Input layer: feature_dim → hidden_dim
        layers.append(nn.Linear(self.feature_dim, self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.Dropout(self.dropout))
        
        # Hidden layers: hidden_dim → hidden_dim
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.Dropout(self.dropout))
        
        # Output layer: hidden_dim → num_ue_antennas * num_subcarriers
        if self.complex_output:
            # For complex output, we need 2 * num_ue_antennas * num_subcarriers
            output_dim = 2 * self.num_ue_antennas * self.num_subcarriers
        else:
            output_dim = self.num_ue_antennas * self.num_subcarriers
            
        layers.append(nn.Linear(self.hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the AttenuationDecoder.
        
        Args:
            features: Input tensor of shape (batch_size, feature_dim) from AttenuationNetwork
            
        Returns:
            Output tensor of shape (batch_size, num_ue_antennas, num_subcarriers) 
            or (batch_size, num_ue_antennas, num_subcarriers, 2) for complex
        """
        batch_size = features.shape[0]
        
        # Process through network
        output = self.network(features)
        
        if self.complex_output:
            # Reshape to (batch_size, num_ue_antennas, num_subcarriers, 2)
            output = output.view(batch_size, self.num_ue_antennas, self.num_subcarriers, 2)
            # Convert to complex tensor with explicit dtype to avoid ComplexHalf warning
            real_part = output[..., 0].to(torch.float32)
            imag_part = output[..., 1].to(torch.float32)
            output = torch.complex(real_part, imag_part)
        else:
            # Reshape to (batch_size, num_ue_antennas, num_subcarriers)
            output = output.view(batch_size, self.num_ue_antennas, self.num_subcarriers)
        
        return output
    
    def get_output_shape(self) -> Tuple[int, int]:
        """Get the output shape (num_ue_antennas, num_subcarriers)."""
        return (self.num_ue_antennas, self.num_subcarriers)
    
    def is_complex(self) -> bool:
        """Check if the network outputs complex values."""
        return self.complex_output


class AttenuationDecoderConfig:
    """Configuration class for AttenuationDecoder."""
    
    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        num_ue_antennas: int = 4,
        num_subcarriers: int = 64,
        num_layers: int = 3,
        activation: str = "relu",
        complex_output: bool = True,
        dropout: float = 0.1
    ):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_ue_antennas = num_ue_antennas
        self.num_subcarriers = num_subcarriers
        self.num_layers = num_layers
        self.activation = activation
        self.complex_output = complex_output
        self.dropout = dropout
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'num_ue_antennas': self.num_ue_antennas,
            'num_subcarriers': self.num_subcarriers,
            'num_layers': self.num_layers,
            'activation': self.activation,
            'complex_output': self.complex_output,
            'dropout': self.dropout
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AttenuationDecoderConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
