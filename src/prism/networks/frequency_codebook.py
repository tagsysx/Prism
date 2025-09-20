"""
FrequencyCodebook: Provides learnable frequency-specific basis vectors.

This module implements a lookup table of learnable R-dimensional complex basis vectors
for each subcarrier to capture frequency-dependent characteristics for RF signal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class FrequencyCodebook(nn.Module):
    """
    FrequencyCodebook: Learnable frequency-specific basis vectors.
    
    Structure:
    - Codebook Size: N_f learnable embeddings (N_f = 408 for typical configurations)
    - Embedding Dimension: R-dimensional complex vectors (R = 32 typically)
    - Total Parameters: N_f × R × 2 = 26,112 learnable parameters (for N_f=408, R=32)
    
    Implementation:
    - Lookup Table: Indexed by subcarrier ID (0 to N_f-1)
    - Learnable Parameters: Each basis vector is a trainable R-dimensional complex vector
    - Initialization: Random initialization with proper complex number initialization
    - Gradient Flow: Full gradient updates during training
    """
    
    def __init__(
        self,
        num_subcarriers: int = 408,
        basis_dim: int = 32,  # R dimension
        initialization: str = "complex_normal",
        std: float = 0.1,
        normalize: bool = False
    ):
        super().__init__()
        
        self.num_subcarriers = num_subcarriers
        self.basis_dim = basis_dim
        self.initialization = initialization
        self.std = std
        self.normalize = normalize
        
        # Create learnable frequency basis vectors
        self._create_basis_vectors()
        
    def _create_basis_vectors(self):
        """Create and initialize the learnable frequency basis vectors."""
        
        if self.initialization == "complex_normal":
            # Initialize with complex normal distribution
            # Real and imaginary parts are independent normal distributions
            real_part = torch.randn(self.num_subcarriers, self.basis_dim) * self.std
            imag_part = torch.randn(self.num_subcarriers, self.basis_dim) * self.std
            self.frequency_basis = nn.Parameter(real_part + 1j * imag_part)
            
        elif self.initialization == "uniform":
            # Initialize with uniform distribution in complex plane
            real_part = torch.rand(self.num_subcarriers, self.basis_dim) * 2 * self.std - self.std
            imag_part = torch.rand(self.num_subcarriers, self.basis_dim) * 2 * self.std - self.std
            self.frequency_basis = nn.Parameter(real_part + 1j * imag_part)
            
        elif self.initialization == "xavier":
            # Xavier initialization for complex numbers
            fan_in = self.basis_dim
            fan_out = self.basis_dim
            std = math.sqrt(2.0 / (fan_in + fan_out))
            real_part = torch.randn(self.num_subcarriers, self.basis_dim) * std
            imag_part = torch.randn(self.num_subcarriers, self.basis_dim) * std
            self.frequency_basis = nn.Parameter(real_part + 1j * imag_part)
            
        else:
            raise ValueError(f"Unknown initialization method: {self.initialization}")
    
    def forward(self, subcarrier_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to retrieve frequency basis vectors.
        
        Args:
            subcarrier_indices: Optional tensor of subcarrier indices to retrieve.
                              If None, returns all frequency basis vectors.
                              Shape: (num_selected_subcarriers,) or None
        
        Returns:
            frequency_basis_vectors: Complex frequency basis vectors
                                   Shape: (num_selected_subcarriers, basis_dim) or (num_subcarriers, basis_dim)
        """
        if subcarrier_indices is not None:
            # Retrieve specific subcarrier basis vectors
            selected_basis = self.frequency_basis[subcarrier_indices]
        else:
            # Return all frequency basis vectors
            selected_basis = self.frequency_basis
        
        # Apply normalization if requested
        if self.normalize:
            # L2 normalize each frequency basis vector
            eps = 1e-12
            norm = torch.sqrt(torch.sum(torch.real(selected_basis * torch.conj(selected_basis)), dim=1, keepdim=True) + eps)
            selected_basis = selected_basis / norm
        
        return selected_basis
    
    def get_basis_vector(self, subcarrier_idx: int) -> torch.Tensor:
        """
        Get a specific frequency basis vector by index.
        
        Args:
            subcarrier_idx: Index of the subcarrier (0 to num_subcarriers-1)
        
        Returns:
            basis_vector: Complex basis vector for the specified subcarrier
                         Shape: (basis_dim,)
        """
        if subcarrier_idx < 0 or subcarrier_idx >= self.num_subcarriers:
            raise ValueError(f"Subcarrier index {subcarrier_idx} out of range [0, {self.num_subcarriers-1}]")
        
        basis_vector = self.frequency_basis[subcarrier_idx]
        
        if self.normalize:
            eps = 1e-12
            norm = torch.sqrt(torch.sum(torch.real(basis_vector * torch.conj(basis_vector))) + eps)
            basis_vector = basis_vector / norm
        
        return basis_vector
    
    def get_all_basis_vectors(self) -> torch.Tensor:
        """
        Get all frequency basis vectors.
        
        Returns:
            all_basis: All frequency basis vectors
                      Shape: (num_subcarriers, basis_dim)
        """
        return self.forward()
    
    def get_codebook_info(self) -> dict:
        """Get information about the frequency codebook."""
        return {
            'num_subcarriers': self.num_subcarriers,
            'basis_dim': self.basis_dim,
            'total_parameters': self.num_subcarriers * self.basis_dim * 2,  # Complex numbers
            'initialization': self.initialization,
            'normalize': self.normalize,
            'parameter_shape': self.frequency_basis.shape
        }
    
    def get_device(self) -> torch.device:
        """Get the device of the frequency codebook."""
        return next(self.parameters()).device
    
    @property
    def device(self) -> torch.device:
        """Device property for compatibility."""
        return self.get_device()


# Export
__all__ = ['FrequencyCodebook']
