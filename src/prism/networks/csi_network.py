"""
CSI Enhancement Network using Transformer

This module implements a Transformer-based network that enhances CSI (Channel State Information)
by learning frequency correlations within each antenna pair (BS-UE).

Key Features:
- Multi-head self-attention within each antenna pair
- Frequency-aware processing for subcarrier correlations
- Complex number handling (real/imaginary parts)
- Positional encoding for spatial relationships
- Per-antenna-pair processing to avoid cross-antenna interference
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
        
        # Use learnable parameters instead of embeddings for spatial and frequency encoding
        self.spatial_projection = nn.Linear(1, d_model)  # Project antenna index to d_model
        self.frequency_projection = nn.Linear(1, d_model)  # Project subcarrier index to d_model
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Encode CSI data.
        
        Args:
            csi: Complex CSI tensor [batch_size, bs_antennas, ue_antennas, num_subcarriers] (4D only)
            
        Returns:
            Encoded features [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model] (5D)
        """
        device = csi.device
        
        # Only accept 4D input format
        if csi.dim() != 4:
            raise ValueError(f"CSI tensor must be 4D. Got {csi.dim()}D tensor with shape {csi.shape}. Expected: [batch_size, bs_antennas, ue_antennas, num_subcarriers]")
        
        # 4D format: [batch_size, bs_antennas, ue_antennas, num_subcarriers]
        batch_size, bs_antennas, ue_antennas, num_subcarriers = csi.shape
        
        # Extract real and imaginary parts
        real_part = csi.real  # [batch_size, bs_antennas, ue_antennas, num_subcarriers]
        imag_part = csi.imag  # [batch_size, bs_antennas, ue_antennas, num_subcarriers]
        
        # Stack real and imaginary parts
        complex_features = torch.stack([real_part, imag_part], dim=-1)  # [batch_size, bs_antennas, ue_antennas, num_subcarriers, 2]
        
        # Project to model dimension
        x = self.complex_projection(complex_features)  # [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model]
        
        # Add spatial encodings for BS antennas (using learnable projection)
        bs_spatial_indices = torch.arange(bs_antennas, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(batch_size, bs_antennas, ue_antennas, num_subcarriers).unsqueeze(-1)
        bs_spatial_emb = self.spatial_projection(bs_spatial_indices)  # [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model]
        
        # Add frequency encodings (using learnable projection)
        freq_indices = torch.arange(num_subcarriers, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, bs_antennas, ue_antennas, num_subcarriers).unsqueeze(-1)
        freq_emb = self.frequency_projection(freq_indices)  # [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model]
        
        # Combine features
        x = x + bs_spatial_emb + freq_emb
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
            x: Enhanced features [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model] (5D only)
            
        Returns:
            Enhanced CSI [batch_size, bs_antennas, ue_antennas, num_subcarriers] (4D)
        """
        # Only accept 5D input format
        if x.dim() != 5:
            raise ValueError(f"Enhanced features tensor must be 5D. Got {x.dim()}D tensor with shape {x.shape}. Expected: [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model]")
        
        # Apply layer normalization
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Project to complex numbers
        complex_features = self.output_projection(x)  # [batch_size, bs_antennas, ue_antennas, num_subcarriers, 2]
        
        # Convert back to complex tensor
        real_part = complex_features[..., 0]  # [batch_size, bs_antennas, ue_antennas, num_subcarriers]
        imag_part = complex_features[..., 1]   # [batch_size, bs_antennas, ue_antennas, num_subcarriers]
        
        enhanced_csi = torch.complex(real_part, imag_part)
        
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
            x: Input features [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model] (5D only)
            
        Returns:
            Enhanced features [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model] (5D)
        """
        # Only accept 5D input format
        if x.dim() != 5:
            raise ValueError(f"Input features tensor must be 5D. Got {x.dim()}D tensor with shape {x.shape}. Expected: [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model]")
        
        # 5D format: [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model]
        batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model = x.shape
        
        # Process each antenna pair separately to avoid cross-antenna interference
        enhanced_outputs = []
        
        for bs_idx in range(bs_antennas):
            for ue_idx in range(ue_antennas):
                # Extract CSI for this specific antenna pair: [batch_size, num_subcarriers, d_model]
                antenna_csi = x[:, bs_idx, ue_idx, :, :]  # [batch_size, num_subcarriers, d_model]
                
                # Self-attention within this antenna pair only
                attn_output, _ = self.self_attention(antenna_csi, antenna_csi, antenna_csi)
                antenna_csi = self.norm1(antenna_csi + self.dropout(attn_output))
                
                # Feed-forward with residual connection
                ffn_output = self.ffn(antenna_csi)
                antenna_csi = self.norm2(antenna_csi + ffn_output)
                
                enhanced_outputs.append(antenna_csi)
        
        # Reshape back to original format
        # Stack all antenna pairs: [batch_size, bs_antennas * ue_antennas, num_subcarriers, d_model]
        stacked_output = torch.stack(enhanced_outputs, dim=1)  # [batch_size, bs_antennas * ue_antennas, num_subcarriers, d_model]
        
        # Reshape to original 5D format: [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model]
        x = stacked_output.view(batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model)
        
        return x


class CSINetwork(nn.Module):
    """
    CSI Enhancement Network using Transformer.
    
    This network enhances CSI by learning frequency correlations within each antenna pair
    using multi-head self-attention. Each antenna pair (BS-UE) is processed independently
    to avoid cross-antenna interference.
    
    Architecture:
    1. Input: Complex CSI [batch_size, bs_antennas, ue_antennas, num_subcarriers]
    2. Encoding: Convert to real features + spatial/frequency embeddings
    3. Transformer: Multi-head self-attention within each antenna pair
    4. Decoding: Convert back to enhanced complex CSI
    
    Key Feature: Per-antenna-pair processing - each BS-UE antenna pair is enhanced independently
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout_rate: float = 0.1,
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
        self.smoothing_weight = smoothing_weight
        self.magnitude_constraint = magnitude_constraint
        self.max_magnitude = max_magnitude
        
        # Encoder (no max dimension constraints)
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
        logger.info(f"  - smoothing_weight: {smoothing_weight}")
        logger.info(f"  - magnitude_constraint: {magnitude_constraint}")
    
    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CSI enhancement network.
        
        Args:
            csi: Input CSI tensor [batch_size, bs_antennas, ue_antennas, num_subcarriers] (4D only)
            
        Returns:
            enhanced_csi: Enhanced CSI tensor [batch_size, bs_antennas, ue_antennas, num_subcarriers] (4D)
        """
        # Validate input
        if not torch.is_complex(csi):
            raise ValueError("Input CSI must be complex tensor")
        
        # Only accept 4D input format
        if csi.dim() != 4:
            raise ValueError(f"CSI tensor must be 4D. Got {csi.dim()}D tensor with shape {csi.shape}. Expected: [batch_size, bs_antennas, ue_antennas, num_subcarriers]")
        
        # 4D format: [batch_size, bs_antennas, ue_antennas, num_subcarriers]
        batch_size, bs_antennas, ue_antennas, num_subcarriers = csi.shape
        
        # No dimension validation - accept any size
        
        logger.debug(f"🔧 CSINetwork processing: {csi.shape}")
        
        # Encode CSI
        x = self.encoder(csi)  # [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model]
        
        # Apply transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            logger.debug(f"   Transformer layer {i+1}: {x.shape}")
        
        # Decode to enhanced CSI
        enhanced_csi = self.decoder(x)  # [batch_size, bs_antennas, ue_antennas, num_subcarriers]
        
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
        
        # Handle different input shapes
        if magnitude.dim() == 3:  # [batch_size, num_antennas, num_subcarriers]
            # Reshape to [batch_size * num_antennas, 1, num_subcarriers] for conv1d
            batch_size, num_antennas, num_subcarriers = magnitude.shape
            magnitude_reshaped = magnitude.view(batch_size * num_antennas, 1, num_subcarriers)
            
            # Apply 1D convolution along subcarrier dimension
            smoothed = F.conv1d(
                magnitude_reshaped,
                torch.ones(1, 1, kernel_size, device=magnitude.device) / kernel_size,
                padding=padding
            )
            
            # Reshape back to original shape
            smoothed = smoothed.view(batch_size, num_antennas, num_subcarriers)
        else:
            # For other shapes, use simple moving average
            smoothed = magnitude
        
        return smoothed
    
    def _smooth_spatial(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Apply spatial domain smoothing."""
        # For single antenna case, no spatial smoothing needed
        if magnitude.shape[1] == 1:
            return magnitude
        
        # Simple moving average across antennas
        kernel_size = 3
        padding = kernel_size // 2
        
        # Handle different input shapes
        if magnitude.dim() == 3:  # [batch_size, num_antennas, num_subcarriers]
            # Reshape to [batch_size * num_subcarriers, 1, num_antennas] for conv1d
            batch_size, num_antennas, num_subcarriers = magnitude.shape
            magnitude_reshaped = magnitude.transpose(1, 2).contiguous().view(batch_size * num_subcarriers, 1, num_antennas)
            
            # Apply 1D convolution along antenna dimension
            smoothed = F.conv1d(
                magnitude_reshaped,
                torch.ones(1, 1, kernel_size, device=magnitude.device) / kernel_size,
                padding=padding
            )
            
            # Reshape back to original shape
            smoothed = smoothed.view(batch_size, num_subcarriers, num_antennas).transpose(1, 2)
        else:
            # For other shapes, no smoothing
            smoothed = magnitude
        
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
