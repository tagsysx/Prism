"""
Positional Encoder: Traditional positional encoding for spatial coordinates.

This module implements the standard positional encoding used in NeRF and similar models,
which transforms 3D coordinates into high-dimensional representations using sinusoidal functions.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoder(nn.Module):
    """
    Traditional Positional Encoding for spatial coordinates.
    
    Transforms input coordinates into high-dimensional representations using
    sinusoidal functions at different frequencies, as used in NeRF.
    
    PE(x) = [sin(2^0 π x), cos(2^0 π x), sin(2^1 π x), cos(2^1 π x), ..., 
             sin(2^(L-1) π x), cos(2^(L-1) π x)]
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        num_frequencies: int = 10,
        include_input: bool = True,
        log_sampling: bool = True
    ):
        """
        Initialize the positional encoder.
        
        Args:
            input_dim: Input dimension (typically 3 for 3D coordinates)
            num_frequencies: Number of frequency levels (L in the formula)
            include_input: Whether to include the original input coordinates
            log_sampling: Whether to use logarithmic frequency sampling
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.log_sampling = log_sampling
        
        # Calculate output dimension
        # Each frequency level contributes 2 * input_dim (sin + cos for each dimension)
        self.encoding_dim = 2 * num_frequencies * input_dim
        if include_input:
            self.encoding_dim += input_dim
        
        # Pre-compute frequency bands
        if log_sampling:
            # Logarithmic sampling: 2^0, 2^1, 2^2, ..., 2^(L-1)
            freq_bands = 2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
        else:
            # Linear sampling: 1, 2, 3, ..., L
            freq_bands = torch.linspace(1.0, num_frequencies, num_frequencies)
        
        # Register as buffer (non-trainable parameter)
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input coordinates.
        
        Args:
            x: Input coordinates of shape (..., input_dim)
            
        Returns:
            Encoded coordinates of shape (..., encoding_dim)
        """
        # Store original shape for later reshaping
        original_shape = x.shape[:-1]
        input_dim = x.shape[-1]
        
        # Flatten all dimensions except the last one
        x_flat = x.view(-1, input_dim)
        
        # Initialize output list
        encoded_parts = []
        
        # Include original input if requested
        if self.include_input:
            encoded_parts.append(x_flat)
        
        # Apply sinusoidal encoding for each frequency
        for freq in self.freq_bands:
            # Scale coordinates by frequency
            scaled_x = freq * math.pi * x_flat
            
            # Apply sin and cos
            encoded_parts.append(torch.sin(scaled_x))
            encoded_parts.append(torch.cos(scaled_x))
        
        # Concatenate all encoded parts
        encoded = torch.cat(encoded_parts, dim=-1)
        
        # Reshape back to original shape (except last dimension)
        output_shape = original_shape + (self.encoding_dim,)
        encoded = encoded.view(output_shape)
        
        return encoded
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the encoded coordinates."""
        return self.encoding_dim
    
    def __repr__(self) -> str:
        return (f"PositionalEncoder(input_dim={self.input_dim}, "
                f"num_frequencies={self.num_frequencies}, "
                f"include_input={self.include_input}, "
                f"output_dim={self.encoding_dim})")


def create_positional_encoder(
    input_dim: int = 3,
    num_frequencies: int = 10,
    include_input: bool = True
) -> PositionalEncoder:
    """
    Factory function to create a positional encoder with common settings.
    
    Args:
        input_dim: Input dimension (3 for 3D coordinates)
        num_frequencies: Number of frequency levels
        include_input: Whether to include original coordinates
        
    Returns:
        Configured PositionalEncoder instance
    """
    return PositionalEncoder(
        input_dim=input_dim,
        num_frequencies=num_frequencies,
        include_input=include_input,
        log_sampling=True
    )


# Common configurations
def create_position_encoder() -> PositionalEncoder:
    """Create encoder for 3D positions (10 frequencies)."""
    return create_positional_encoder(input_dim=3, num_frequencies=10, include_input=True)


def create_direction_encoder() -> PositionalEncoder:
    """Create encoder for 3D directions (4 frequencies, typically fewer for directions)."""
    return create_positional_encoder(input_dim=3, num_frequencies=4, include_input=True)
