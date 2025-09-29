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
    AntennaEmbeddingCodebook: Learnable BS-UE antenna pair embeddings.
    
    Structure:
    - Codebook Size: N_BS × N_UE learnable embeddings for each BS-UE antenna pair
    - Embedding Dimension: 64-dimensional learnable vectors
    - Total Parameters: N_BS × N_UE × 64 learnable parameters
    
    Implementation:
    - Lookup Table: Indexed by (bs_antenna_id, ue_antenna_id) pairs
    - Learnable Parameters: Each embedding is a trainable 64D vector for each antenna pair
    - Initialization: Random initialization or pre-trained embeddings
    - Gradient Flow: Full gradient updates during training
    """
    
    def __init__(
        self,
        num_bs_antennas: int = 64,
        num_ue_antennas: int = 4,
        embedding_dim: int = 64,
        initialization: str = "normal",
        std: float = 0.1,
        normalize: bool = False
    ):
        super().__init__()
        
        self.num_bs_antennas = num_bs_antennas
        self.num_ue_antennas = num_ue_antennas
        self.embedding_dim = embedding_dim
        self.initialization = initialization
        self.std = std
        self.normalize = normalize
        
        # Create learnable embeddings for BS-UE antenna pairs
        self._create_embeddings()
        
    def _create_embeddings(self):
        """Create and initialize the learnable embeddings for BS-UE antenna pairs."""
        
        # Initialize embeddings based on specified method
        # Shape: (num_bs_antennas, num_ue_antennas, embedding_dim)
        if self.initialization == "normal":
            embeddings = torch.randn(self.num_bs_antennas, self.num_ue_antennas, self.embedding_dim) * self.std
        elif self.initialization == "uniform":
            embeddings = torch.rand(self.num_bs_antennas, self.num_ue_antennas, self.embedding_dim) * 2 * self.std - self.std
        elif self.initialization == "xavier":
            embeddings = torch.randn(self.num_bs_antennas, self.num_ue_antennas, self.embedding_dim)
            nn.init.xavier_uniform_(embeddings)
        elif self.initialization == "kaiming":
            embeddings = torch.randn(self.num_bs_antennas, self.num_ue_antennas, self.embedding_dim)
            nn.init.kaiming_uniform_(embeddings)
        else:
            raise ValueError(f"Unsupported initialization: {self.initialization}")
        
        # Register as learnable parameter
        self.register_parameter('embeddings', nn.Parameter(embeddings))
        
    def forward(self, bs_antenna_indices: torch.Tensor, ue_antenna_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to retrieve BS-UE antenna pair embeddings.
        
        Args:
            bs_antenna_indices: BS antenna indices of shape (batch_size,)
                              Values should be in range [0, num_bs_antennas-1]
            ue_antenna_indices: UE antenna indices of shape (batch_size,)
                              Values should be in range [0, num_ue_antennas-1]
            
        Returns:
            Antenna pair embeddings of shape (batch_size, embedding_dim)
            Each row contains the embedding for the corresponding BS-UE antenna pair
        """
        # Validate input shapes
        if bs_antenna_indices.shape != ue_antenna_indices.shape:
            raise ValueError(f"BS and UE antenna indices must have same shape: "
                           f"{bs_antenna_indices.shape} vs {ue_antenna_indices.shape}")
        
        # Ensure inputs are 1D
        if bs_antenna_indices.dim() != 1:
            raise ValueError(f"Expected 1D tensors, got BS indices shape: {bs_antenna_indices.shape}")
        
        # Validate antenna indices ranges
        if torch.any(bs_antenna_indices >= self.num_bs_antennas) or torch.any(bs_antenna_indices < 0):
            raise ValueError(f"BS antenna indices out of range [0, {self.num_bs_antennas-1}]")
        if torch.any(ue_antenna_indices >= self.num_ue_antennas) or torch.any(ue_antenna_indices < 0):
            raise ValueError(f"UE antenna indices out of range [0, {self.num_ue_antennas-1}]")
        
        # Direct indexing for specific BS-UE antenna pairs
        embeddings = self.embeddings[bs_antenna_indices, ue_antenna_indices]
        
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        return embeddings
    
    def get_embedding(self, bs_antenna_id: int, ue_antenna_id: int) -> torch.Tensor:
        """
        Get embedding for a specific BS-UE antenna pair.
        
        Args:
            bs_antenna_id: BS antenna ID (0 to num_bs_antennas-1)
            ue_antenna_id: UE antenna ID (0 to num_ue_antennas-1)
            
        Returns:
            Embedding tensor of shape (embedding_dim,)
        """
        if not 0 <= bs_antenna_id < self.num_bs_antennas:
            raise ValueError(f"BS antenna ID {bs_antenna_id} out of range [0, {self.num_bs_antennas-1}]")
        if not 0 <= ue_antenna_id < self.num_ue_antennas:
            raise ValueError(f"UE antenna ID {ue_antenna_id} out of range [0, {self.num_ue_antennas-1}]")
        
        embedding = self.embeddings[bs_antenna_id, ue_antenna_id]
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=0)
        return embedding
    
    def get_all_embeddings(self) -> torch.Tensor:
        """
        Get all BS-UE antenna pair embeddings.
        
        Returns:
            All embeddings tensor of shape (num_bs_antennas, num_ue_antennas, embedding_dim)
        """
        embeddings = self.embeddings
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
    
    def update_embedding(self, bs_antenna_id: int, ue_antenna_id: int, new_embedding: torch.Tensor):
        """
        Update embedding for a specific BS-UE antenna pair.
        
        Args:
            bs_antenna_id: BS antenna ID to update
            ue_antenna_id: UE antenna ID to update
            new_embedding: New embedding tensor of shape (embedding_dim,)
        """
        if not 0 <= bs_antenna_id < self.num_bs_antennas:
            raise ValueError(f"BS antenna ID {bs_antenna_id} out of range [0, {self.num_bs_antennas-1}]")
        if not 0 <= ue_antenna_id < self.num_ue_antennas:
            raise ValueError(f"UE antenna ID {ue_antenna_id} out of range [0, {self.num_ue_antennas-1}]")
        
        if new_embedding.shape != (self.embedding_dim,):
            raise ValueError(f"New embedding shape {new_embedding.shape} doesn't match expected {(self.embedding_dim,)}")
        
        with torch.no_grad():
            self.embeddings[bs_antenna_id, ue_antenna_id] = new_embedding
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def get_num_bs_antennas(self) -> int:
        """Get the number of BS antennas."""
        return self.num_bs_antennas
    
    def get_num_ue_antennas(self) -> int:
        """Get the number of UE antennas."""
        return self.num_ue_antennas
    
    def get_total_parameters(self) -> int:
        """Get the total number of learnable parameters."""
        return self.num_bs_antennas * self.num_ue_antennas * self.embedding_dim


class AntennaEmbeddingCodebookConfig:
    """Configuration class for AntennaEmbeddingCodebook."""
    
    def __init__(
        self,
        num_bs_antennas: int = 64,
        num_ue_antennas: int = 4,
        embedding_dim: int = 64,
        initialization: str = "normal",
        std: float = 0.1,
        normalize: bool = False
    ):
        self.num_bs_antennas = num_bs_antennas
        self.num_ue_antennas = num_ue_antennas
        self.embedding_dim = embedding_dim
        self.initialization = initialization
        self.std = std
        self.normalize = normalize
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'num_bs_antennas': self.num_bs_antennas,
            'num_ue_antennas': self.num_ue_antennas,
            'embedding_dim': self.embedding_dim,
            'initialization': self.initialization,
            'std': self.std,
            'normalize': self.normalize
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AntennaEmbeddingCodebookConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
