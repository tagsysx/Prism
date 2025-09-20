"""
CSI Enhancement Network using Transformer

This module implements a Transformer-based network that enhances CSI (Channel State Information)
by learning spatial and frequency correlations across antennas and subcarriers.

Key Features:
- Multi-head self-attention for spatial correlations
- Frequency-aware processing for subcarrier correlations
- Complex number handling (real/imaginary parts)
- Positional encoding for spatial relationships
- Memory-efficient processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class CSIEncoder(nn.Module):
    """Encoder for CSI data with spatial and frequency awareness."""
    
    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Complex number processing: separate real and imaginary parts
        self.complex_projection = nn.Linear(2, d_model)  # [real, imag] -> d_model
        
        # Spatial and frequency embeddings
        self.spatial_embedding = nn.Embedding(64, d_model)  # For 64 antennas
        self.frequency_embedding = nn.Embedding(408, d_model)  # For 408 subcarriers
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Encode CSI data.
        
        Args:
            csi: Complex CSI tensor [batch_size, num_antennas, num_subcarriers]
            
        Returns:
            Encoded features [batch_size, num_antennas, num_subcarriers, d_model]
        """
        batch_size, num_antennas, num_subcarriers = csi.shape
        device = csi.device
        
        # Extract real and imaginary parts
        real_part = csi.real  # [batch_size, num_antennas, num_subcarriers]
        imag_part = csi.imag  # [batch_size, num_antennas, num_subcarriers]
        
        # Stack real and imaginary parts
        complex_features = torch.stack([real_part, imag_part], dim=-1)  # [batch_size, num_antennas, num_subcarriers, 2]
        
        # Project to model dimension
        x = self.complex_projection(complex_features)  # [batch_size, num_antennas, num_subcarriers, d_model]
        
        # Add spatial embeddings
        spatial_indices = torch.arange(num_antennas, device=device).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_antennas, num_subcarriers)
        spatial_emb = self.spatial_embedding(spatial_indices)  # [batch_size, num_antennas, num_subcarriers, d_model]
        
        # Add frequency embeddings
        freq_indices = torch.arange(num_subcarriers, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_antennas, num_subcarriers)
        freq_emb = self.frequency_embedding(freq_indices)  # [batch_size, num_antennas, num_subcarriers, d_model]
        
        # Combine features
        x = x + spatial_emb + freq_emb
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x


class CSIDecoder(nn.Module):
    """Decoder for enhanced CSI data."""
    
    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Output projection back to complex numbers
        self.output_projection = nn.Linear(d_model, 2)  # d_model -> [real, imag]
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode enhanced features back to CSI.
        
        Args:
            x: Enhanced features [batch_size, num_antennas, num_subcarriers, d_model]
            
        Returns:
            Enhanced CSI [batch_size, num_antennas, num_subcarriers]
        """
        # Apply layer normalization
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Project to complex numbers
        complex_features = self.output_projection(x)  # [batch_size, num_antennas, num_subcarriers, 2]
        
        # Convert back to complex tensor
        real_part = complex_features[..., 0]  # [batch_size, num_antennas, num_subcarriers]
        imag_part = complex_features[..., 1]   # [batch_size, num_antennas, num_subcarriers]
        
        enhanced_csi = torch.complex(real_part, imag_part)  # [batch_size, num_antennas, num_subcarriers]
        
        return enhanced_csi


class CSITransformerLayer(nn.Module):
    """Single transformer layer for CSI enhancement."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_rate)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer layer.
        
        Args:
            x: Input features [batch_size, num_antennas, num_subcarriers, d_model]
            
        Returns:
            Enhanced features [batch_size, num_antennas, num_subcarriers, d_model]
        """
        batch_size, num_antennas, num_subcarriers, d_model = x.shape
        
        # Reshape for attention: [batch_size, num_antennas * num_subcarriers, d_model]
        x_flat = x.view(batch_size, num_antennas * num_subcarriers, d_model)
        
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x_flat)
        x_flat = self.norm2(x_flat + ffn_output)
        
        # Reshape back to original shape
        x = x_flat.view(batch_size, num_antennas, num_subcarriers, d_model)
        
        return x


class CSINetwork(nn.Module):
    """
    CSI Enhancement Network using Transformer.
    
    This network enhances CSI by learning spatial and frequency correlations
    across antennas and subcarriers using multi-head self-attention.
    
    Architecture:
    1. Input: Complex CSI [batch_size, num_antennas, num_subcarriers]
    2. Encoding: Convert to real features + spatial/frequency embeddings
    3. Transformer: Multi-head self-attention layers
    4. Decoding: Convert back to enhanced complex CSI
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout_rate: float = 0.1,
        num_antennas: int = 64,
        num_subcarriers: int = 408,
        smoothing_weight: float = 0.1,
        magnitude_constraint: bool = True,
        max_magnitude: float = 5.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.smoothing_weight = smoothing_weight
        self.magnitude_constraint = magnitude_constraint
        self.max_magnitude = max_magnitude
        
        # Encoder
        self.encoder = CSIEncoder(d_model, dropout_rate)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            CSITransformerLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])
        
        # Decoder
        self.decoder = CSIDecoder(d_model, dropout_rate)
        
        logger.info(f"CSINetwork initialized:")
        logger.info(f"  - d_model: {d_model}")
        logger.info(f"  - n_layers: {n_layers}")
        logger.info(f"  - n_heads: {n_heads}")
        logger.info(f"  - d_ff: {d_ff}")
        logger.info(f"  - dropout_rate: {dropout_rate}")
        logger.info(f"  - num_antennas: {num_antennas}")
        logger.info(f"  - num_subcarriers: {num_subcarriers}")
    
    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CSI enhancement network.
        
        Args:
            csi: Input CSI tensor [batch_size, num_antennas, num_subcarriers]
            
        Returns:
            enhanced_csi: Enhanced CSI tensor [batch_size, num_antennas, num_subcarriers]
        """
        # Validate input
        if not torch.is_complex(csi):
            raise ValueError("Input CSI must be complex tensor")
        
        batch_size, num_antennas, num_subcarriers = csi.shape
        
        if num_antennas != self.num_antennas:
            raise ValueError(f"Expected {self.num_antennas} antennas, got {num_antennas}")
        if num_subcarriers != self.num_subcarriers:
            raise ValueError(f"Expected {self.num_subcarriers} subcarriers, got {num_subcarriers}")
        
        logger.debug(f"🔧 CSINetwork processing: {csi.shape}")
        
        # Encode CSI
        x = self.encoder(csi)  # [batch_size, num_antennas, num_subcarriers, d_model]
        
        # Apply transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            logger.debug(f"   Transformer layer {i+1}: {x.shape}")
        
        # Decode to enhanced CSI
        enhanced_csi = self.decoder(x)  # [batch_size, num_antennas, num_subcarriers]
        
        # Apply magnitude constraint to reduce vibrations
        if self.magnitude_constraint:
            enhanced_csi = self._apply_magnitude_constraint(enhanced_csi)
        
        logger.debug(f"✅ CSINetwork enhancement completed: {enhanced_csi.shape}")
        
        return enhanced_csi
    
    def _apply_magnitude_constraint(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Apply magnitude constraint to reduce excessive vibrations.
        
        Args:
            csi: Complex CSI tensor [batch_size, num_antennas, num_subcarriers]
            
        Returns:
            constrained_csi: CSI with constrained magnitude
        """
        # Get magnitude and phase
        magnitude = torch.abs(csi)
        phase = torch.angle(csi)
        
        # Apply soft constraint: gradually reduce magnitude if too high
        constrained_magnitude = torch.clamp(magnitude, max=self.max_magnitude)
        
        # Apply smoothing to reduce vibrations
        if self.smoothing_weight > 0:
            # Smooth across subcarriers (frequency domain)
            constrained_magnitude = self._smooth_frequency(constrained_magnitude)
            
            # Smooth across antennas (spatial domain) 
            constrained_magnitude = self._smooth_spatial(constrained_magnitude)
        
        # Reconstruct complex CSI
        constrained_csi = constrained_magnitude * torch.exp(1j * phase)
        
        return constrained_csi
    
    def _smooth_frequency(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Apply frequency domain smoothing."""
        # Simple moving average across subcarriers
        kernel_size = 3
        padding = kernel_size // 2
        
        # Apply 1D convolution along subcarrier dimension
        smoothed = F.conv1d(
            magnitude.unsqueeze(1),  # Add channel dimension
            torch.ones(1, 1, kernel_size, device=magnitude.device) / kernel_size,
            padding=padding
        ).squeeze(1)
        
        return smoothed
    
    def _smooth_spatial(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Apply spatial domain smoothing."""
        # Simple moving average across antennas
        kernel_size = 3
        padding = kernel_size // 2
        
        # Apply 1D convolution along antenna dimension
        smoothed = F.conv1d(
            magnitude.transpose(1, 2).unsqueeze(1),  # Transpose and add channel dimension
            torch.ones(1, 1, kernel_size, device=magnitude.device) / kernel_size,
            padding=padding
        ).squeeze(1).transpose(1, 2)
        
        return smoothed
    
    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'num_antennas': self.num_antennas,
            'num_subcarriers': self.num_subcarriers,
            'total_parameters': total_params
        }
