"""
RadianceNetwork: Processes UE position, viewing direction, spatial features, and antenna embeddings.

This network outputs N_UE × K radiation values for all UE antenna channels,
capturing the radiation-dependent CSI observed at the UE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RadianceNetwork(nn.Module):
    """
    RadianceNetwork: Processes UE position, viewing direction, spatial features, and antenna embeddings.
    
    Architecture: Structure similar to the color subnetwork in standard NeRF.
    Single network processing all UE antenna channels.
    Each channel processes: [IPE-encoded UE_pos, IPE-encoded view_dir, 128D_features, 64D_antenna_embedding] 
    → N_UE × K radiation values
    Output: Complex values representing radiation characteristics.
    """
    
    def __init__(
        self,
        ue_position_dim: int = 63,  # IPE-encoded 3D position
        view_direction_dim: int = 63,  # IPE-encoded 3D direction
        feature_dim: int = 128,  # From AttenuationNetwork
        antenna_embedding_dim: int = 64,  # From antenna codebook
        hidden_dim: int = 256,
        num_ue_antennas: int = 4,
        num_subcarriers: int = 64,
        num_layers: int = 4,
        activation: str = "relu",
        complex_output: bool = True,
        dropout: float = 0.1,
        use_skip_connections: bool = True
    ):
        super().__init__()
        
        self.ue_position_dim = ue_position_dim
        self.view_direction_dim = view_direction_dim
        self.feature_dim = feature_dim
        self.antenna_embedding_dim = antenna_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_ue_antennas = num_ue_antennas
        self.num_subcarriers = num_subcarriers
        self.num_layers = num_layers
        self.use_skip_connections = use_skip_connections
        self.complex_output = complex_output
        self.dropout = dropout
        
        # Calculate input dimension
        self.input_dim = ue_position_dim + view_direction_dim + feature_dim + antenna_embedding_dim
        
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
        """Build the network architecture with optional skip connections."""
        
        # Input layer: concatenated input → hidden_dim
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        
        # Hidden layers with skip connections
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        
        for i in range(self.num_layers - 2):
            layer = nn.Linear(self.hidden_dim, self.hidden_dim)
            norm = nn.LayerNorm(self.hidden_dim)
            self.hidden_layers.append(layer)
            self.hidden_norms.append(norm)
        
        # Output layer: hidden_dim → num_ue_antennas * num_subcarriers
        if self.complex_output:
            # For complex output, we need 2 * num_ue_antennas * num_subcarriers
            output_dim = 2 * self.num_ue_antennas * self.num_subcarriers
        else:
            output_dim = self.num_ue_antennas * self.num_subcarriers
            
        self.output_layer = nn.Linear(self.hidden_dim, output_dim)
        
        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(self.dropout) for _ in range(self.num_layers - 1)
        ])
        
    def forward(
        self, 
        ue_positions: torch.Tensor,
        view_directions: torch.Tensor,
        features: torch.Tensor,
        antenna_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the RadianceNetwork.
        
        Args:
            ue_positions: IPE-encoded UE positions of shape (batch_size, ue_position_dim)
            view_directions: IPE-encoded viewing directions of shape (batch_size, view_direction_dim)
            features: 128D feature vectors from AttenuationNetwork of shape (batch_size, feature_dim)
            antenna_embeddings: Antenna embeddings of shape (batch_size, num_antennas, antenna_embedding_dim)
                               Note: In practice, num_antennas=1 (single BS antenna per ray tracing call)
            
        Returns:
            Radiation values of shape (batch_size, num_ue_antennas, num_subcarriers)
            Note: Single BS antenna processing - no multi-antenna dimension needed
        """
        batch_size, num_antennas = antenna_embeddings.shape[:2]
        
        # Since we always process one BS antenna at a time, num_antennas should be 1
        # Extract the single antenna embedding
        if num_antennas == 1:
            antenna_embedding_single = antenna_embeddings[:, 0, :]  # (batch_size, antenna_embedding_dim)
        else:
            # Fallback: take mean if multiple antennas (shouldn't happen in practice)
            antenna_embedding_single = antenna_embeddings.mean(dim=1)  # (batch_size, antenna_embedding_dim)
        
        # Concatenate inputs: [ue_pos, view_dir, features, antenna_embedding]
        x = torch.cat([ue_positions, view_directions, features, antenna_embedding_single], dim=-1)
        
        # Process through single antenna path
        output = self._process_single_antenna(x)
        
        return output
    
    def _process_single_antenna(self, x: torch.Tensor) -> torch.Tensor:
        """Process input for single antenna case."""
        batch_size = x.shape[0]
        
        # Input layer
        h = self.input_layer(x)
        h = self.input_norm(h)
        h = self.activation(h)
        h = self.dropout_layers[0](h)
        
        # Hidden layers with skip connections
        for i, (layer, norm, dropout) in enumerate(zip(self.hidden_layers, self.hidden_norms, self.dropout_layers[1:])):
            if self.use_skip_connections and i > 0:
                h = h + self.hidden_layers[i-1](h)  # Skip connection
            
            h = layer(h)
            h = norm(h)
            h = self.activation(h)
            h = dropout(h)
        
        # Output layer
        output = self.output_layer(h)
        
        if self.complex_output:
            # Reshape to (batch_size, num_ue_antennas, num_subcarriers, 2)
            output = output.view(batch_size, self.num_ue_antennas, self.num_subcarriers, 2)
            # Convert to complex tensor
            output = torch.complex(output[..., 0], output[..., 1])
        else:
            # Reshape to (batch_size, num_ue_antennas, num_subcarriers)
            output = output.view(batch_size, self.num_ue_antennas, self.num_subcarriers)
        
        return output
    
    def _process_multiple_antennas(self, x: torch.Tensor) -> torch.Tensor:
        """Process input for multiple antennas case."""
        batch_size, num_antennas = x.shape[:2]
        
        # Reshape to (batch_size * num_antennas, total_input_dim) for processing
        x_flat = x.view(-1, x.shape[-1])
        
        # Process through network
        output_flat = self._process_single_antenna(x_flat)
        
        # Reshape back to (batch_size, num_antennas, ...)
        if self.complex_output:
            output = output_flat.view(batch_size, num_antennas, self.num_ue_antennas, self.num_subcarriers)
        else:
            output = output_flat.view(batch_size, num_antennas, self.num_ue_antennas, self.num_subcarriers)
        
        return output
    
    def get_output_shape(self) -> Tuple[int, int]:
        """Get the output shape (num_ue_antennas, num_subcarriers)."""
        return (self.num_ue_antennas, self.num_subcarriers)
    
    def is_complex(self) -> bool:
        """Check if the network outputs complex values."""
        return self.complex_output
    
    def get_input_dimensions(self) -> dict:
        """Get the input dimensions for each component."""
        return {
            'ue_position_dim': self.ue_position_dim,
            'view_direction_dim': self.view_direction_dim,
            'feature_dim': self.feature_dim,
            'antenna_embedding_dim': self.antenna_embedding_dim,
            'total_input_dim': self.input_dim
        }


class RadianceNetworkConfig:
    """Configuration class for RadianceNetwork."""
    
    def __init__(
        self,
        ue_position_dim: int = 63,
        view_direction_dim: int = 63,
        feature_dim: int = 128,
        antenna_embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_ue_antennas: int = 4,
        num_subcarriers: int = 64,
        num_layers: int = 4,
        activation: str = "relu",
        complex_output: bool = True,
        dropout: float = 0.1,
        use_skip_connections: bool = True
    ):
        self.ue_position_dim = ue_position_dim
        self.view_direction_dim = view_direction_dim
        self.feature_dim = feature_dim
        self.antenna_embedding_dim = antenna_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_ue_antennas = num_ue_antennas
        self.num_subcarriers = num_subcarriers
        self.num_layers = num_layers
        self.activation = activation
        self.complex_output = complex_output
        self.dropout = dropout
        self.use_skip_connections = use_skip_connections
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'ue_position_dim': self.ue_position_dim,
            'view_direction_dim': self.view_direction_dim,
            'feature_dim': self.feature_dim,
            'antenna_embedding_dim': self.antenna_embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_ue_antennas': self.num_ue_antennas,
            'num_subcarriers': self.num_subcarriers,
            'num_layers': self.num_layers,
            'activation': self.activation,
            'complex_output': self.complex_output,
            'dropout': self.dropout,
            'use_skip_connections': self.use_skip_connections
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'RadianceNetworkConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
