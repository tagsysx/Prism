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
        self.spatial_projection = nn.Linear(1, d_model)  # Project BS antenna index to d_model
        self.ue_projection = nn.Linear(1, d_model)  # Project UE antenna index to d_model
        self.frequency_projection = nn.Linear(1, d_model)  # Project subcarrier index to d_model
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights for better amplitude learning
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better amplitude learning."""
        # Initialize complex projection with larger variance for amplitude learning
        nn.init.xavier_uniform_(self.complex_projection.weight, gain=2.0)  # Larger gain for amplitude
        nn.init.zeros_(self.complex_projection.bias)
        
        # Initialize spatial, UE, and frequency projections with moderate variance
        nn.init.xavier_uniform_(self.spatial_projection.weight, gain=1.0)
        nn.init.zeros_(self.spatial_projection.bias)
        
        nn.init.xavier_uniform_(self.ue_projection.weight, gain=1.0)
        nn.init.zeros_(self.ue_projection.bias)
        
        nn.init.xavier_uniform_(self.frequency_projection.weight, gain=1.0)
        nn.init.zeros_(self.frequency_projection.bias)
        
        logger.debug("🔧 CSIEncoder weights initialized with enhanced amplitude learning strategy")
        
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
        
        # Add UE antenna encodings (using learnable projection)
        ue_indices = torch.arange(ue_antennas, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(batch_size, bs_antennas, ue_antennas, num_subcarriers).unsqueeze(-1)
        ue_emb = self.ue_projection(ue_indices)  # [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model]
        
        # Add frequency encodings (using learnable projection)
        freq_indices = torch.arange(num_subcarriers, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, bs_antennas, ue_antennas, num_subcarriers).unsqueeze(-1)
        freq_emb = self.frequency_projection(freq_indices)  # [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model]
        
        # Combine features
        x = x + bs_spatial_emb + ue_emb + freq_emb
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x


class CSIDecoder(nn.Module):
    """Decoder for enhanced CSI data with per-subcarrier scaling."""
    
    def __init__(self, d_model: int, dropout_rate: float = 0.1, num_subcarriers: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_subcarriers = num_subcarriers
        
        # Output projection back to complex numbers
        self.output_projection = nn.Linear(d_model, 2)  # d_model -> [real, imag]
        
        # Per-subcarrier learnable scaling factors
        # Each subcarrier learns its own amplitude scaling factor
        self.subcarrier_scalers = nn.Parameter(torch.ones(num_subcarriers))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights for better amplitude learning
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with standard Xavier initialization."""
        # Standard Xavier initialization (no special amplitude scaling)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=1.0)
        nn.init.zeros_(self.output_projection.bias)
        
        # Initialize per-subcarrier scaling factors with small random values
        # This allows each subcarrier to learn its own optimal scaling
        nn.init.normal_(self.subcarrier_scalers, mean=1.0, std=0.1)
        
        logger.debug("🔧 CSIDecoder weights initialized with per-subcarrier scaling factors")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode enhanced features back to CSI with per-subcarrier scaling.
        
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
        
        # Apply per-subcarrier scaling factors
        # subcarrier_scalers: [num_subcarriers] -> broadcast to all dimensions
        scaling_factors = torch.abs(self.subcarrier_scalers) + 0.1  # Ensure positive scaling
        real_part = real_part * scaling_factors
        imag_part = imag_part * scaling_factors
        
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
            nn.GELU(),  # GELU allows negative values, better for complex data
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_rate)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Initialize weights for better amplitude learning
        self._init_weights()
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def _init_weights(self):
        """Initialize weights for better amplitude learning."""
        # Initialize FFN layers with larger variance for amplitude learning
        for module in self.ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.5)  # Moderate gain for FFN
                nn.init.zeros_(module.bias)
        
        logger.debug("🔧 CSITransformerLayer weights initialized with enhanced amplitude learning strategy")
        
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
        num_subcarriers: int = 64,
        smoothing_weight: float = 0.0,
        smoothing_type: str = "phase_preserve"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.num_subcarriers = num_subcarriers
        self.smoothing_weight = smoothing_weight
        self.smoothing_type = smoothing_type
        
        # Encoder (no max dimension constraints)
        self.encoder = CSIEncoder(d_model, dropout_rate)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            CSITransformerLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])
        
        # Decoder with per-subcarrier scaling
        self.decoder = CSIDecoder(d_model, dropout_rate, num_subcarriers)
        
        logger.info(f"CSINetwork initialized:")
        logger.info(f"  - d_model: {d_model}")
        logger.info(f"  - n_layers: {n_layers}")
        logger.info(f"  - n_heads: {n_heads}")
        logger.info(f"  - d_ff: {d_ff}")
        logger.info(f"  - dropout_rate: {dropout_rate}")
        logger.info(f"  - num_subcarriers: {num_subcarriers}")
        logger.info(f"  - smoothing_weight: {smoothing_weight}")
        logger.info(f"  - smoothing_type: {smoothing_type}")
        logger.info(f"  - per-subcarrier scaling: enabled")
    
    def forward(self, csi: torch.Tensor, max_magnitude: float = 100.0) -> torch.Tensor:
        """
        Forward pass of CSI enhancement network.
        
        Args:
            csi: Input CSI tensor [batch_size, bs_antennas, ue_antennas, num_subcarriers] (4D only)
            max_magnitude: Maximum allowed magnitude value from config
            
        Returns:
            enhanced_csi: Enhanced CSI tensor [batch_size, bs_antennas, ue_antennas, num_subcarriers] (4D)
        """
        # Validate input
        if not torch.is_complex(csi):
            raise ValueError("Input CSI must be complex tensor")
        
        # Only accept 4D input format
        if csi.dim() != 4:
            raise ValueError(f"CSI tensor must be 4D. Got {csi.dim()}D tensor with shape {csi.shape}. Expected: [batch_size, bs_antennas, ue_antennas, num_subcarriers]")
        
        # Get actual number of subcarriers from input tensor
        batch_size, bs_antennas, ue_antennas, actual_num_subcarriers = csi.shape
        
        # Check if we need to resize subcarrier scalers
        if actual_num_subcarriers != self.num_subcarriers:
            logger.debug(f"🔧 Resizing subcarrier scalers from {self.num_subcarriers} to {actual_num_subcarriers}")
            # Resize subcarrier scalers to match actual subcarrier count
            if actual_num_subcarriers > self.num_subcarriers:
                # Expand: pad with ones
                padding = torch.ones(actual_num_subcarriers - self.num_subcarriers, device=csi.device)
                self.subcarrier_scalers = nn.Parameter(torch.cat([self.subcarrier_scalers, padding]))
            else:
                # Shrink: take first actual_num_subcarriers elements
                self.subcarrier_scalers = nn.Parameter(self.subcarrier_scalers[:actual_num_subcarriers])
            self.num_subcarriers = actual_num_subcarriers
        
        # 4D format: [batch_size, bs_antennas, ue_antennas, num_subcarriers]
        batch_size, bs_antennas, ue_antennas, num_subcarriers = csi.shape
        
        # No dimension validation - accept any size
        
        logger.debug(f"🔧 CSINetwork processing: {csi.shape}")
        
        # Debug: Log input CSI amplitude statistics
        input_amplitude = torch.abs(csi)
        input_nonzero = input_amplitude[input_amplitude > 1e-8]
        if len(input_nonzero) > 0:
            logger.debug(f"📊 Input CSI Amplitude Stats: min={input_nonzero.min().item():.6f}, max={input_nonzero.max().item():.6f}, mean={input_nonzero.mean().item():.6f}")
        else:
            logger.warning("⚠️ Input CSI has no valid amplitude data")
        
        # Normalize input CSI to [0, max_magnitude] range
        # Always normalize to ensure consistent range
        if len(input_nonzero) > 0:
            # Use 95th percentile for robust normalization (avoid extreme outliers)
            input_max = torch.quantile(input_nonzero, 0.95)
            
            if input_max > max_magnitude:
                # Scale to [0, max_magnitude] range
                norm_factor = input_max / max_magnitude
                csi_normalized = csi / norm_factor
                
            else:
                csi_normalized = csi
        else:
            csi_normalized = csi
        
        # Encode CSI (use normalized version)
        x = self.encoder(csi_normalized)  # [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model]
        
        # Apply transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            logger.debug(f"   Transformer layer {i+1}: {x.shape}")
        
        # Apply frequency domain smoothing to features before decoding (if enabled)
        if self.smoothing_weight > 0:
            x = self._apply_feature_smoothing(x)
        
        # Decode to enhanced CSI
        enhanced_csi = self.decoder(x)  # [batch_size, bs_antennas, ue_antennas, num_subcarriers]
        
        # Apply magnitude constraint to reduce vibrations (always enabled)
        enhanced_csi = self._apply_magnitude_constraint(enhanced_csi, max_magnitude)
        
        # Debug: Log output CSI amplitude statistics
        output_amplitude = torch.abs(enhanced_csi)
        output_nonzero = output_amplitude[output_amplitude > 1e-8]
        if len(output_nonzero) > 0:
            logger.debug(f"📊 Output CSI Amplitude Stats: min={output_nonzero.min().item():.6f}, max={output_nonzero.max().item():.6f}, mean={output_nonzero.mean().item():.6f}")
        else:
            logger.warning("⚠️ Output CSI has no valid amplitude data")
        
        logger.debug(f"✅ CSINetwork enhancement completed: {enhanced_csi.shape}")
        
        return enhanced_csi
    
    def _apply_magnitude_constraint(self, csi: torch.Tensor, max_magnitude: float = 100.0) -> torch.Tensor:
        """
        Apply magnitude constraint to reduce excessive vibrations.
        
        Args:
            csi: Complex CSI tensor [batch_size, num_antennas, num_subcarriers]
            max_magnitude: Maximum allowed magnitude value
            
        Returns:
            constrained_csi: CSI with constrained magnitude
        """
        # Get magnitude and phase
        magnitude = torch.abs(csi)
        phase = torch.angle(csi)
        
        # Apply magnitude constraint: clamp to max_magnitude
        constrained_magnitude = torch.clamp(magnitude, max=max_magnitude)
        
        # Reconstruct complex CSI
        constrained_csi = constrained_magnitude * torch.exp(1j * phase)
        
        return constrained_csi
    
    def _apply_feature_smoothing(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency domain smoothing to features before decoding.
        
        Args:
            features: Feature tensor [batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model]
            
        Returns:
            smoothed_features: Smoothed feature tensor
        """
        if self.smoothing_weight <= 0:
            return features
            
        logger.debug(f"🔧 Applying feature smoothing: weight={self.smoothing_weight}, type={self.smoothing_type}")
        
        batch_size, bs_antennas, ue_antennas, num_subcarriers, d_model = features.shape
        
        # Apply smoothing along frequency dimension for each feature dimension
        smoothed_features = torch.zeros_like(features)
        
        for b in range(batch_size):
            for bs in range(bs_antennas):
                for ue in range(ue_antennas):
                    for d in range(d_model):
                        # Extract frequency response for this feature dimension
                        freq_response = features[b, bs, ue, :, d]  # [num_subcarriers]
                        
                        # Apply smoothing based on smoothing_type
                        if self.smoothing_type == "phase_preserve":
                            # For phase_preserve, apply moderate smoothing to all features
                            smoothed_response = self._smooth_frequency_domain(
                                freq_response.unsqueeze(0).unsqueeze(0).unsqueeze(0), 
                                self.smoothing_weight * 0.5  # Reduced smoothing for features
                            ).squeeze()
                        elif self.smoothing_type == "complex":
                            # For complex, apply full smoothing to all features
                            smoothed_response = self._smooth_frequency_domain(
                                freq_response.unsqueeze(0).unsqueeze(0).unsqueeze(0), 
                                self.smoothing_weight
                            ).squeeze()
                        elif self.smoothing_type == "magnitude_only":
                            # For magnitude_only, apply light smoothing to all features
                            smoothed_response = self._smooth_frequency_domain(
                                freq_response.unsqueeze(0).unsqueeze(0).unsqueeze(0), 
                                self.smoothing_weight * 0.3  # Light smoothing for features
                            ).squeeze()
                        else:
                            # Default: apply moderate smoothing
                            smoothed_response = self._smooth_frequency_domain(
                                freq_response.unsqueeze(0).unsqueeze(0).unsqueeze(0), 
                                self.smoothing_weight * 0.5
                            ).squeeze()
                        
                        smoothed_features[b, bs, ue, :, d] = smoothed_response
        
        logger.debug(f"✅ Feature smoothing completed with type: {self.smoothing_type}")
        return smoothed_features
    
    def _smooth_frequency_domain(self, signal: torch.Tensor, smoothing_weight: float) -> torch.Tensor:
        """
        Apply frequency domain smoothing using moving average.
        
        Args:
            signal: Input signal tensor [batch_size, bs_antennas, ue_antennas, num_subcarriers]
            smoothing_weight: Smoothing strength (0-1)
            
        Returns:
            smoothed_signal: Smoothed signal tensor
        """
        batch_size, bs_antennas, ue_antennas, num_subcarriers = signal.shape
        
        # Create smoothing kernel (moving average)
        kernel_size = max(3, int(smoothing_weight * num_subcarriers * 0.1))  # Adaptive kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        # Create 1D smoothing kernel
        kernel = torch.ones(kernel_size, device=signal.device, dtype=signal.dtype) / kernel_size
        
        # Apply smoothing along frequency dimension
        smoothed_signal = torch.zeros_like(signal)
        
        for b in range(batch_size):
            for bs in range(bs_antennas):
                for ue in range(ue_antennas):
                    # Extract frequency response for this antenna pair
                    freq_response = signal[b, bs, ue, :]  # [num_subcarriers]
                    
                    # Apply 1D convolution for smoothing
                    padded_response = torch.nn.functional.pad(
                        freq_response.unsqueeze(0).unsqueeze(0), 
                        (kernel_size // 2, kernel_size // 2), 
                        mode='reflect'
                    )
                    
                    smoothed_response = torch.nn.functional.conv1d(
                        padded_response, 
                        kernel.unsqueeze(0).unsqueeze(0)
                    ).squeeze()
                    
                    smoothed_signal[b, bs, ue, :] = smoothed_response
        
        return smoothed_signal
    
    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'num_subcarriers': self.num_subcarriers,
            'smoothing_weight': self.smoothing_weight,
            'smoothing_type': self.smoothing_type,
            'total_parameters': total_params
        }