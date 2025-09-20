"""
AttenuationNetwork: Computes attenuation coefficients and feature vectors for RF signal modeling.

Given a 3D voxel location P, the network outputs:
1. An attenuation coefficient ρ(P) = ln(ΔA) + j*Δφ (R-dimensional complex vector)
2. A feature vector F(P) for additional spatial information

The attenuation factor ρ(P) = ln(ΔA*e^(j*Δφ)) captures amplitude loss (ΔA in dB/m) 
and phase rotation (Δφ in rad/m) caused by the voxel's material properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AttenuationNetwork(nn.Module):
    """
    AttenuationNetwork: Computes attenuation coefficients and feature vectors for RF signal modeling.
    
    Given a 3D voxel location P with positional encoding PE(P), outputs:
    1. Attenuation coefficient ρ(P) = ln(ΔA) + j*Δφ (R-dimensional complex vector)
    2. Feature vector F(P) for additional spatial information
    
    Architecture: Similar to Standard NeRF density network with 8 layers and shortcuts.
    Input: PE-encoded 3D → Hidden: 256D → Output: ρ(P) + F(P)
    """
    
    def __init__(
        self,
        input_dim: int = 63,  # PE-encoded 3D position (21 frequencies * 3 dimensions)
        hidden_dim: int = 256,
        feature_dim: int = 128,  # Dimension of feature vector F(P)
        output_dim: int = 32,  # R-dimensional attenuation coefficient ρ(P)
        num_layers: int = 8,
        use_shortcuts: bool = True,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_shortcuts = use_shortcuts
        
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
        
        # Output layers - separate heads for attenuation coefficient and feature vector
        # Attenuation coefficient ρ(P): R-dimensional complex vector (2*R for real/imag parts)
        self.attenuation_head = nn.Linear(self.hidden_dim, 2 * self.output_dim)
        
        # Feature vector F(P): feature_dim dimensional real vector
        self.feature_head = nn.Linear(self.hidden_dim, self.feature_dim)
            
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers - 1)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both attenuation coefficients and feature vectors.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) - PE-encoded positions
            
        Returns:
            Tuple of (attenuation_coefficients, feature_vectors)
            - attenuation_coefficients: shape (batch_size, output_dim)
            - feature_vectors: shape (batch_size, feature_dim)
        """
        # Existing network processing
        h = self.input_layer(x)
        h = self.activation(h)
        
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.activation(h)
        
        # Compute separate outputs
        attenuation_raw = self.attenuation_head(h)  # Shape: [batch_size, 2 * output_dim]
        features = self.feature_head(h)             # Shape: [batch_size, feature_dim]
        
        # Convert attenuation from real-valued [2*R] to complex [R]
        batch_size = attenuation_raw.shape[0]
        attenuation_reshaped = attenuation_raw.view(batch_size, self.output_dim, 2)
        real_part = attenuation_reshaped[..., 0]  # Shape: [batch_size, output_dim]
        imag_part = attenuation_reshaped[..., 1]  # Shape: [batch_size, output_dim]
        # Use alternative complex tensor creation to avoid PyTorch compatibility issues
        # Ensure float32 dtype for numerical stability
        real_part = real_part.to(torch.float32)
        imag_part = imag_part.to(torch.float32)
        attenuation = real_part + 1j * imag_part  # Shape: [batch_size, output_dim] - Complex64
        
        
        return attenuation, features
    
    def get_feature_dim(self) -> int:
        """Get the feature vector dimension."""
        return self.feature_dim
    
    def get_attenuation_dim(self) -> int:
        """Get the attenuation coefficient dimension."""
        return self.output_dim
    
    def get_output_info(self) -> dict:
        """Get information about network outputs."""
        return {
            'feature_dim': self.feature_dim,
            'attenuation_dim': self.output_dim,
            'attenuation_is_complex': True,
            'feature_is_complex': False
        }


class AttenuationNetworkConfig:
    """Configuration class for AttenuationNetwork."""
    
    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 256,
        feature_dim: int = 128,
        output_dim: int = 32,
        num_layers: int = 8,
        use_shortcuts: bool = True,
        activation: str = "relu"
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_shortcuts = use_shortcuts
        self.activation = activation
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'use_shortcuts': self.use_shortcuts,
            'activation': self.activation
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AttenuationNetworkConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

