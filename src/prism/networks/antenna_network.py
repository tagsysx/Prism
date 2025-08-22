"""
AntennaNetwork: Processes antenna embeddings to generate directional importance indicators.

This network generates A × B directional importance matrices for efficient
directional sampling in ray tracing applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AntennaNetwork(nn.Module):
    """
    AntennaNetwork: Processes antenna embeddings to generate directional importance indicators.
    
    Architecture: Shallow network for efficient processing.
    Input: 64D antenna embedding → Hidden: 128D → Output: A × B importance values
    Activation: Softmax or sigmoid to normalize importance scores
    Output Shape: A × B matrix where A and B are configurable directional divisions
    """
    
    def __init__(
        self,
        antenna_embedding_dim: int = 64,
        hidden_dim: int = 128,
        azimuth_divisions: int = 16,
        elevation_divisions: int = 8,
        num_layers: int = 2,
        activation: str = "relu",
        importance_activation: str = "softmax",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.antenna_embedding_dim = antenna_embedding_dim
        self.hidden_dim = hidden_dim
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.num_layers = num_layers
        self.importance_activation = importance_activation
        self.dropout = dropout
        
        # Calculate output dimensions
        self.output_dim = azimuth_divisions * elevation_divisions
        
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
        
        # Input layer: antenna_embedding_dim → hidden_dim
        layers.append(nn.Linear(self.antenna_embedding_dim, self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.Dropout(self.dropout))
        
        # Hidden layers: hidden_dim → hidden_dim
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.Dropout(self.dropout))
        
        # Output layer: hidden_dim → output_dim (A × B)
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, antenna_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the AntennaNetwork.
        
        Args:
            antenna_embeddings: Input tensor of shape (batch_size, antenna_embedding_dim)
                               or (batch_size, num_antennas, antenna_embedding_dim)
            
        Returns:
            Directional importance matrix of shape (batch_size, azimuth_divisions, elevation_divisions)
            or (batch_size, num_antennas, azimuth_divisions, elevation_divisions)
        """
        batch_size = antenna_embeddings.shape[0]
        
        # Process through network
        if antenna_embeddings.dim() == 2:
            # Single antenna per batch element
            output = self.network(antenna_embeddings)
            # Reshape to (batch_size, azimuth_divisions, elevation_divisions)
            output = output.view(batch_size, self.azimuth_divisions, self.elevation_divisions)
        else:
            # Multiple antennas per batch element
            num_antennas = antenna_embeddings.shape[1]
            # Reshape to (batch_size * num_antennas, antenna_embedding_dim) for processing
            flat_embeddings = antenna_embeddings.view(-1, self.antenna_embedding_dim)
            output = self.network(flat_embeddings)
            # Reshape to (batch_size, num_antennas, azimuth_divisions, elevation_divisions)
            output = output.view(batch_size, num_antennas, self.azimuth_divisions, self.elevation_divisions)
        
        # Apply importance activation
        if self.importance_activation == "softmax":
            # Apply softmax across the last two dimensions (azimuth and elevation)
            output = F.softmax(output.view(*output.shape[:-2], -1), dim=-1)
            output = output.view(*output.shape[:-1], self.azimuth_divisions, self.elevation_divisions)
        elif self.importance_activation == "sigmoid":
            output = torch.sigmoid(output)
        elif self.importance_activation == "tanh":
            output = torch.tanh(output)
        else:
            raise ValueError(f"Unsupported importance activation: {self.importance_activation}")
        
        return output
    
    def get_top_k_directions(
        self, 
        importance_matrix: torch.Tensor, 
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-K most important directions from the importance matrix.
        
        Args:
            importance_matrix: Directional importance matrix from forward pass
            k: Number of top directions to select
            
        Returns:
            Tuple of (top_k_indices, top_k_importance_scores)
        """
        batch_size = importance_matrix.shape[0]
        
        if importance_matrix.dim() == 3:
            # Single antenna: (batch_size, azimuth_divisions, elevation_divisions)
            # Flatten to (batch_size, azimuth_divisions * elevation_divisions)
            flat_importance = importance_matrix.view(batch_size, -1)
            
            # Get top-k indices and values
            top_k_values, top_k_indices = torch.topk(flat_importance, k, dim=-1)
            
            # Convert flat indices to 2D indices
            azimuth_indices = top_k_indices // self.elevation_divisions
            elevation_indices = top_k_indices % self.elevation_divisions
            
            # Stack to get (batch_size, k, 2) tensor
            top_k_indices_2d = torch.stack([azimuth_indices, elevation_indices], dim=-1)
            
            return top_k_indices_2d, top_k_values
            
        else:
            # Multiple antennas: (batch_size, num_antennas, azimuth_divisions, elevation_divisions)
            num_antennas = importance_matrix.shape[1]
            # Flatten to (batch_size, num_antennas, azimuth_divisions * elevation_divisions)
            flat_importance = importance_matrix.view(batch_size, num_antennas, -1)
            
            # Get top-k indices and values
            top_k_values, top_k_indices = torch.topk(flat_importance, k, dim=-1)
            
            # Convert flat indices to 2D indices
            azimuth_indices = top_k_indices // self.elevation_divisions
            elevation_indices = top_k_indices % self.elevation_divisions
            
            # Stack to get (batch_size, num_antennas, k, 2) tensor
            top_k_indices_2d = torch.stack([azimuth_indices, elevation_indices], dim=-1)
            
            return top_k_indices_2d, top_k_values
    
    def get_directional_resolution(self) -> Tuple[int, int]:
        """Get the directional resolution (azimuth_divisions, elevation_divisions)."""
        return (self.azimuth_divisions, self.elevation_divisions)
    
    def get_output_shape(self) -> Tuple[int, int]:
        """Get the output shape (azimuth_divisions, elevation_divisions)."""
        return (self.azimuth_divisions, self.elevation_divisions)


class AntennaNetworkConfig:
    """Configuration class for AntennaNetwork."""
    
    def __init__(
        self,
        antenna_embedding_dim: int = 64,
        hidden_dim: int = 128,
        azimuth_divisions: int = 16,
        elevation_divisions: int = 8,
        num_layers: int = 2,
        activation: str = "relu",
        importance_activation: str = "softmax",
        dropout: float = 0.1
    ):
        self.antenna_embedding_dim = antenna_embedding_dim
        self.hidden_dim = hidden_dim
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.num_layers = num_layers
        self.activation = activation
        self.importance_activation = importance_activation
        self.dropout = dropout
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'antenna_embedding_dim': self.antenna_embedding_dim,
            'hidden_dim': self.hidden_dim,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions,
            'num_layers': self.num_layers,
            'activation': self.activation,
            'importance_activation': self.importance_activation,
            'dropout': self.dropout
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AntennaNetworkConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
