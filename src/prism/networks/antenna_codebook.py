"""
AntennaEmbeddingCodebook: Provides learnable antenna-specific embeddings.

This module implements a lookup table of learnable 64-dimensional embeddings
for each BS antenna to capture unique radiation characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AntennaEmbeddingCodebook(nn.Module):
    """
    AntennaEmbeddingCodebook: Learnable antenna-specific embeddings.
    
    Structure:
    - Codebook Size: N_BS learnable embeddings (N_BS = 64 for typical configurations)
    - Embedding Dimension: 64-dimensional learnable vectors
    - Total Parameters: N_BS × 64 = 4,096 learnable parameters
    
    Implementation:
    - Lookup Table: Indexed by antenna ID (0 to N_BS-1)
    - Learnable Parameters: Each embedding is a trainable 64D vector
    - Initialization: Random initialization or pre-trained embeddings
    - Gradient Flow: Full gradient updates during training
    """
    
    def __init__(
        self,
        num_bs_antennas: int = 64,
        embedding_dim: int = 64,
        initialization: str = "normal",
        std: float = 0.1,
        normalize: bool = False
    ):
        super().__init__()
        
        self.num_bs_antennas = num_bs_antennas
        self.embedding_dim = embedding_dim
        self.initialization = initialization
        self.std = std
        self.normalize = normalize
        
        # Create learnable embeddings
        self._create_embeddings()
        
    def _create_embeddings(self):
        """Create and initialize the learnable embeddings."""
        
        # Initialize embeddings based on specified method
        if self.initialization == "normal":
            embeddings = torch.randn(self.num_bs_antennas, self.embedding_dim) * self.std
        elif self.initialization == "uniform":
            embeddings = torch.rand(self.num_bs_antennas, self.embedding_dim) * 2 * self.std - self.std
        elif self.initialization == "xavier":
            embeddings = torch.randn(self.num_bs_antennas, self.embedding_dim)
            nn.init.xavier_uniform_(embeddings)
        elif self.initialization == "kaiming":
            embeddings = torch.randn(self.num_bs_antennas, self.embedding_dim)
            nn.init.kaiming_uniform_(embeddings)
        else:
            raise ValueError(f"Unsupported initialization: {self.initialization}")
        
        # Register as learnable parameter
        self.register_parameter('embeddings', nn.Parameter(embeddings))
        
    def forward(self, antenna_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to retrieve antenna embeddings.
        
        Args:
            antenna_indices: Tensor of antenna indices of shape (batch_size,) or (batch_size, num_antennas)
                           Values should be in range [0, num_bs_antennas-1]
            
        Returns:
            Antenna embeddings of shape (batch_size, embedding_dim) or (batch_size, num_antennas, embedding_dim)
        """
        batch_size = antenna_indices.shape[0]
        
        # Handle single antenna vs multiple antennas
        if antenna_indices.dim() == 1:
            # Single antenna per batch element
            embeddings = F.embedding(antenna_indices, self.embeddings)
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings
        else:
            # Multiple antennas per batch element
            num_antennas = antenna_indices.shape[1]
            # Reshape to (batch_size * num_antennas,) for embedding lookup
            flat_indices = antenna_indices.view(-1)
            embeddings = F.embedding(flat_indices, self.embeddings)
            # Reshape back to (batch_size, num_antennas, embedding_dim)
            embeddings = embeddings.view(batch_size, num_antennas, self.embedding_dim)
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings
    
    def get_embedding(self, antenna_id: int) -> torch.Tensor:
        """
        Get embedding for a specific antenna ID.
        
        Args:
            antenna_id: Antenna ID (0 to num_bs_antennas-1)
            
        Returns:
            Embedding tensor of shape (embedding_dim,)
        """
        if not 0 <= antenna_id < self.num_bs_antennas:
            raise ValueError(f"Antenna ID {antenna_id} out of range [0, {self.num_bs_antennas-1}]")
        
        embedding = self.embeddings[antenna_id]
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=0)
        return embedding
    
    def get_all_embeddings(self) -> torch.Tensor:
        """
        Get all antenna embeddings.
        
        Returns:
            All embeddings tensor of shape (num_bs_antennas, embedding_dim)
        """
        embeddings = self.embeddings
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
    
    def update_embedding(self, antenna_id: int, new_embedding: torch.Tensor):
        """
        Update embedding for a specific antenna ID.
        
        Args:
            antenna_id: Antenna ID to update
            new_embedding: New embedding tensor of shape (embedding_dim,)
        """
        if not 0 <= antenna_id < self.num_bs_antennas:
            raise ValueError(f"Antenna ID {antenna_id} out of range [0, {self.num_bs_antennas-1}]")
        
        if new_embedding.shape != (self.embedding_dim,):
            raise ValueError(f"New embedding shape {new_embedding.shape} doesn't match expected {(self.embedding_dim,)}")
        
        with torch.no_grad():
            self.embeddings[antenna_id] = new_embedding
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def get_num_antennas(self) -> int:
        """Get the number of BS antennas."""
        return self.num_bs_antennas
    
    def get_total_parameters(self) -> int:
        """Get the total number of learnable parameters."""
        return self.num_bs_antennas * self.embedding_dim


class AntennaEmbeddingCodebookConfig:
    """Configuration class for AntennaEmbeddingCodebook."""
    
    def __init__(
        self,
        num_bs_antennas: int = 64,
        embedding_dim: int = 64,
        initialization: str = "normal",
        std: float = 0.1,
        normalize: bool = False
    ):
        self.num_bs_antennas = num_bs_antennas
        self.embedding_dim = embedding_dim
        self.initialization = initialization
        self.std = std
        self.normalize = normalize
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'num_bs_antennas': self.num_bs_antennas,
            'embedding_dim': self.embedding_dim,
            'initialization': self.initialization,
            'std': self.std,
            'normalize': self.normalize
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AntennaEmbeddingCodebookConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
