"""
FrequencyNetwork: Generates R-dimensional complex frequency basis from pre-encoded frequency.

This shallow MLP takes a 63-dimensional pre-encoded frequency f and outputs an R-dimensional 
complex frequency basis vector that captures frequency-dependent characteristics for RF signal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class FrequencyNetwork(nn.Module):
    """
    FrequencyNetwork: Shallow MLP for frequency-dependent complex basis generation.
    
    Takes a 63-dimensional pre-encoded frequency f and outputs an R-dimensional complex frequency basis.
    The complex frequency basis captures frequency-dependent characteristics for RF signal modeling.
    
    Architecture: Shallow MLP with 2-3 layers
    Input: 63D pre-encoded frequency → Hidden: 128D → Output: R-dimensional complex basis
    """
    
    def __init__(
        self,
        input_dim: int = 63,  # 63-dimensional pre-encoded frequency
        hidden_dim: int = 128,
        output_dim: int = 32,  # R-dimensional complex frequency basis
        num_layers: int = 3,  # Shallow MLP
        activation: str = "relu",
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        # Store network parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layers
        self._build_network()
        
    def _build_network(self):
        """Build the shallow MLP architecture."""
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(self.hidden_dim))
        
        # Hidden layers (shallow - typically 1-2 hidden layers)
        for i in range(self.num_layers - 2):  # -2 for input and output layers
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(self.hidden_dim))
        
        # Output layer - R-dimensional complex frequency basis (2*R for real/imag parts)
        layers.append(nn.Linear(self.hidden_dim, 2 * self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
    
    def forward(self, encoded_frequency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FrequencyNetwork.
        
        Args:
            encoded_frequency: Pre-encoded frequency tensor of shape (batch_size, 63)
            
        Returns:
            R-dimensional frequency basis of shape (batch_size, output_dim)
        """
        # Pass through shallow MLP
        frequency_basis = self.network[0](encoded_frequency)  # Input layer
        
        # Apply activation and layer norm if used
        if self.use_layer_norm:
            frequency_basis = self.network[1](frequency_basis)
        frequency_basis = self.activation(frequency_basis)
        
        # Process through remaining layers
        layer_idx = 2 if self.use_layer_norm else 1
        for i in range(1, self.num_layers - 1):  # Hidden layers
            frequency_basis = self.network[layer_idx](frequency_basis)
            layer_idx += 1
            
            if self.use_layer_norm:
                frequency_basis = self.network[layer_idx](frequency_basis)
                layer_idx += 1
            
            frequency_basis = self.activation(frequency_basis)
        
        # Output layer (no activation)
        frequency_basis = self.network[layer_idx](frequency_basis)
        
        return frequency_basis
    
    def get_output_dim(self) -> int:
        """Get the output dimension (R)."""
        return self.output_dim
    
    def get_input_info(self) -> dict:
        """Get information about input requirements."""
        return {
            'expects_encoded_frequency': True,
            'input_dim': self.input_dim,
            'input_shape': '(batch_size, 63)',
            'description': '63-dimensional pre-encoded frequency vector'
        }
    
    def get_network_info(self) -> dict:
        """Get network architecture information."""
        return {
            'network_type': 'shallow_mlp',
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'use_layer_norm': self.use_layer_norm,
            'activation': self.activation.__name__ if hasattr(self.activation, '__name__') else str(self.activation)
        }


class FrequencyNetworkConfig:
    """Configuration class for FrequencyNetwork."""
    
    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 128,
        output_dim: int = 32,
        num_layers: int = 3,
        activation: str = "relu",
        use_layer_norm: bool = True
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.use_layer_norm = use_layer_norm
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'activation': self.activation,
            'use_layer_norm': self.use_layer_norm
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'FrequencyNetworkConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
