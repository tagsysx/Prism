"""
AttenuationNetwork: Encodes spatial position information into compact feature representations.

This network takes IPE-encoded 3D positions and outputs 128-dimensional feature vectors
that capture spatial information for RF signal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AttenuationNetwork(nn.Module):
    """
    AttenuationNetwork: Encodes spatial position information into compact feature representations.
    
    Architecture: Similar to Standard NeRF density network with 8 layers and shortcuts.
    Input: IPE-encoded 3D → Hidden: 256D → Output: 128D
    Outputs complex-valued features for RF signal modeling.
    """
    
    def __init__(
        self,
        input_dim: int = 63,  # IPE-encoded 3D position (21 frequencies * 3 dimensions)
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 8,
        use_shortcuts: bool = True,
        activation: str = "relu",
        complex_output: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_shortcuts = use_shortcuts
        self.complex_output = complex_output
        
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
        """Build the network architecture with optional shortcuts."""
        
        # Input layer
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Hidden layers with shortcuts
        self.hidden_layers = nn.ModuleList()
        for i in range(self.num_layers - 2):  # -2 because we have input and output layers
            layer = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.hidden_layers.append(layer)
            
            # Add shortcut connections if enabled
            if self.use_shortcuts and i > 0:
                shortcut = nn.Linear(self.hidden_dim, self.hidden_dim)
                self.hidden_layers.append(shortcut)
        
        # Output layer
        if self.complex_output:
            # For complex output, we output 2 * output_dim (real and imaginary parts)
            self.output_layer = nn.Linear(self.hidden_dim, 2 * self.output_dim)
        else:
            self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
            
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers - 1)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the AttenuationNetwork.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) containing IPE-encoded positions
            
        Returns:
            Output tensor of shape (batch_size, output_dim) or (batch_size, output_dim, 2) for complex
        """
        batch_size = x.shape[0]
        
        # Input layer
        h = self.input_layer(x)
        h = self.layer_norms[0](h)
        h = self.activation(h)
        
        # Hidden layers with shortcuts
        shortcut_idx = 0
        for i, layer in enumerate(self.hidden_layers):
            if self.use_shortcuts and i > 0 and shortcut_idx < len(self.hidden_layers) // 2:
                # Apply shortcut connection
                shortcut = self.hidden_layers[shortcut_idx + len(self.hidden_layers) // 2]
                h_shortcut = shortcut(h)
                h = h + h_shortcut
                shortcut_idx += 1
            
            h = layer(h)
            if i + 1 < len(self.layer_norms):
                h = self.layer_norms[i + 1](h)
            h = self.activation(h)
        
        # Output layer
        output = self.output_layer(h)
        
        if self.complex_output:
            # Reshape to (batch_size, output_dim, 2) for complex representation
            output = output.view(batch_size, self.output_dim, 2)
            # Convert to complex tensor with explicit dtype to avoid ComplexHalf warning
            real_part = output[..., 0].to(torch.float32)
            imag_part = output[..., 1].to(torch.float32)
            output = torch.complex(real_part, imag_part)
        
        return output
    
    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        return self.output_dim
    
    def is_complex(self) -> bool:
        """Check if the network outputs complex values."""
        return self.complex_output


class AttenuationNetworkConfig:
    """Configuration class for AttenuationNetwork."""
    
    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 8,
        use_shortcuts: bool = True,
        activation: str = "relu",
        complex_output: bool = True
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_shortcuts = use_shortcuts
        self.activation = activation
        self.complex_output = complex_output
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'use_shortcuts': self.use_shortcuts,
            'activation': self.activation,
            'complex_output': self.complex_output
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AttenuationNetworkConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
