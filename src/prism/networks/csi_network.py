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


class JointRoPELayer(nn.Module):
    """Joint RoPE encoding layer for frequency, BS antenna, and UE antenna dimensions."""
    
    def __init__(self, d_model: int, max_freq: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq
        
        # Complex projection layer
        self.complex_projection = nn.Linear(1, d_model, dtype=torch.complex64)
        
        # Learnable scaling parameters for different dimensions
        self.freq_scale = nn.Parameter(torch.ones(1))
        self.bs_scale = nn.Parameter(torch.ones(1))
        self.ue_scale = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for complex projection."""
        nn.init.xavier_uniform_(self.complex_projection.weight, gain=1.0)
        nn.init.zeros_(self.complex_projection.bias)
        
        logger.debug("🔧 JointRoPELayer weights initialized")
        
    def forward(self, csi: torch.Tensor, bs_antenna_index: int, ue_antenna_index: int) -> torch.Tensor:
        """
        Apply joint RoPE encoding to complex CSI.
        
        Args:
            csi: Complex CSI tensor [batch_size, num_subcarriers] (complex)
            bs_antenna_index: BS antenna index
            ue_antenna_index: UE antenna index
            
        Returns:
            Encoded complex features [batch_size, num_subcarriers, d_model] (complex)
        """
        batch_size, num_subcarriers = csi.shape
        device = csi.device
        
        # 1. Project complex CSI to model dimension
        x = self.complex_projection(csi.unsqueeze(-1))  # [batch_size, num_subcarriers, d_model] (complex)
        
        # 2. Calculate joint position encoding
        # Frequency dimension encoding
        freq_indices = torch.arange(num_subcarriers, device=device, dtype=torch.float32)
        freq_angles = freq_indices * (2 * torch.pi / self.max_freq) * self.freq_scale
        
        # BS antenna dimension encoding
        bs_angles = torch.full((batch_size, num_subcarriers), 
                              float(bs_antenna_index * (2 * torch.pi / self.max_freq) * self.bs_scale), 
                              device=device, dtype=torch.float32)
        
        # UE antenna dimension encoding
        ue_angles = torch.full((batch_size, num_subcarriers), 
                              float(ue_antenna_index * (2 * torch.pi / self.max_freq) * self.ue_scale), 
                              device=device, dtype=torch.float32)
        
        # 3. Joint angle encoding
        total_angles = freq_angles.unsqueeze(0) + bs_angles + ue_angles  # [batch_size, num_subcarriers]
        
        # 4. Apply complex rotation
        rotation_matrix = torch.exp(1j * total_angles.unsqueeze(-1))  # [batch_size, num_subcarriers, 1]
        encoded_x = x * rotation_matrix  # [batch_size, num_subcarriers, d_model] (complex)
        
        return encoded_x


class CSIEncoder(nn.Module):
    """Encoder for CSI data with spatial and frequency awareness."""
    
    def __init__(self, d_model: int, dropout_rate: float = 0.1, max_freq: int = 10000):
        super().__init__()
        self.d_model = d_model
        
        # Joint RoPE encoding layer
        self.rope_encoder = JointRoPELayer(d_model, max_freq)
        
        # Complex to real projection layer
        self.complex_to_real_projection = nn.Linear(2 * d_model, d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for complex to real projection."""
        nn.init.xavier_uniform_(self.complex_to_real_projection.weight, gain=1.0)
        nn.init.zeros_(self.complex_to_real_projection.bias)
        
        logger.debug("🔧 CSIEncoder weights initialized")
        
        
    def forward(self, csi: torch.Tensor, bs_antenna_index: int, ue_antenna_index: int) -> torch.Tensor:
        """
        Encode CSI data with joint RoPE encoding.
        
        Args:
            csi: Complex CSI tensor [batch_size, num_subcarriers] (complex)
            bs_antenna_index: BS antenna index
            ue_antenna_index: UE antenna index
            
        Returns:
            Encoded real features [batch_size, num_subcarriers, d_model] (real)
        """
        # Only accept 2D input format
        if csi.dim() != 2:
            raise ValueError(f"CSI tensor must be 2D. Got {csi.dim()}D tensor with shape {csi.shape}. Expected: [batch_size, num_subcarriers]")
        
        # 1. 复数RoPE编码
        complex_encoded = self.rope_encoder(csi, bs_antenna_index, ue_antenna_index)  # [batch_size, num_subcarriers, d_model] (complex)
        
        # 2. 复数转实数：分离实部和虚部
        real_part = complex_encoded.real  # [batch_size, num_subcarriers, d_model]
        imag_part = complex_encoded.imag  # [batch_size, num_subcarriers, d_model]
        
        # 3. 拼接实部和虚部
        real_features = torch.cat([real_part, imag_part], dim=-1)  # [batch_size, num_subcarriers, 2*d_model]
        
        # 4. 投影回d_model维度
        x = self.complex_to_real_projection(real_features)  # [batch_size, num_subcarriers, d_model]
        
        # 5. 层归一化和dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x  # 实数特征，供Transformer使用


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
            x: Enhanced features [batch_size, num_subcarriers, d_model]
            
        Returns:
            Enhanced CSI [batch_size, num_subcarriers]
        """
        # Only accept 3D input format
        if x.dim() != 3:
            raise ValueError(f"Enhanced features tensor must be 3D. Got {x.dim()}D tensor with shape {x.shape}. Expected: [batch_size, num_subcarriers, d_model]")
        
        # Apply layer normalization
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Project to complex numbers
        complex_features = self.output_projection(x)  # [batch_size, num_subcarriers, 2]
        
        # Convert back to complex tensor
        real_part = complex_features[..., 0]  # [batch_size, num_subcarriers]
        imag_part = complex_features[..., 1]  # [batch_size, num_subcarriers]
        
        # Apply per-subcarrier scaling factors
        # subcarrier_scalers: [num_subcarriers] -> broadcast to batch dimension
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
            x: Input features [batch_size, num_subcarriers, d_model]
            
        Returns:
            Enhanced features [batch_size, num_subcarriers, d_model]
        """
        # Only accept 3D input format
        if x.dim() != 3:
            raise ValueError(f"Input features tensor must be 3D. Got {x.dim()}D tensor with shape {x.shape}. Expected: [batch_size, num_subcarriers, d_model]")
        
        # 3D format: [batch_size, num_subcarriers, d_model]
        # Direct processing for single antenna pair
        
        # Self-attention across subcarriers
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
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
        smoothing_type: str = "phase_preserve",
        max_freq: int = 10000
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
        
        # Encoder with RoPE encoding
        self.encoder = CSIEncoder(d_model, dropout_rate, max_freq)
        
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
    
    def _resize_subcarrier_scalers(self, actual_num_subcarriers: int, device: torch.device):
        """Resize subcarrier scalers to match actual subcarrier count."""
        if actual_num_subcarriers > self.num_subcarriers:
            # Expand: pad with ones
            padding = torch.ones(actual_num_subcarriers - self.num_subcarriers, device=device)
            self.decoder.subcarrier_scalers = nn.Parameter(torch.cat([self.decoder.subcarrier_scalers, padding]))
        else:
            # Shrink: take first actual_num_subcarriers elements
            self.decoder.subcarrier_scalers = nn.Parameter(self.decoder.subcarrier_scalers[:actual_num_subcarriers])
        
        # Update both CSINetwork and CSIDecoder num_subcarriers
        self.num_subcarriers = actual_num_subcarriers
        self.decoder.num_subcarriers = actual_num_subcarriers
    
    def forward(self, csi: torch.Tensor, bs_antenna_index: int, ue_antenna_index: int, max_magnitude: float = 100.0) -> torch.Tensor:
        """
        Forward pass of CSI enhancement network for single sample.
        
        Args:
            csi: Input CSI tensor [num_subcarriers] - single CSI sample
            bs_antenna_index: BS antenna index (single integer)
            ue_antenna_index: UE antenna index (single integer)
            max_magnitude: Maximum allowed magnitude value from config
            
        Returns:
            enhanced_csi: Enhanced CSI tensor [num_subcarriers]
        """
        # Validate input
        if not torch.is_complex(csi):
            raise ValueError("Input CSI must be complex tensor")
        
        # Only accept 1D input format for single sample
        if csi.dim() != 1:
            raise ValueError(f"CSI tensor must be 1D [num_subcarriers]. Got {csi.dim()}D tensor with shape {csi.shape}")
        
        actual_num_subcarriers = csi.shape[0]
        # Add batch dimension for processing
        csi = csi.unsqueeze(0)  # [1, num_subcarriers]
        batch_size = 1
        
        # Check if we need to resize subcarrier scalers
        if actual_num_subcarriers != self.num_subcarriers:
            logger.debug(f"🔧 Resizing subcarrier scalers from {self.num_subcarriers} to {actual_num_subcarriers}")
            self._resize_subcarrier_scalers(actual_num_subcarriers, csi.device)
        
        logger.debug(f"🔧 CSINetwork processing: {csi.shape}")
        
        # Debug: Log input CSI amplitude statistics
        input_amplitude = torch.abs(csi)
        input_nonzero = input_amplitude[input_amplitude > 1e-8]
        if len(input_nonzero) > 0:
            logger.debug(f"📊 Input CSI Amplitude Stats: min={input_nonzero.min().item():.6f}, max={input_nonzero.max().item():.6f}, mean={input_nonzero.mean().item():.6f}")
        else:
            logger.warning("⚠️ Input CSI has no valid amplitude data")
        
        # Normalize input CSI to [0, max_magnitude] range
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
        
        # Encode CSI with antenna indices (2D input)
        x = self.encoder(csi_normalized, bs_antenna_index, ue_antenna_index)  # [batch_size, num_subcarriers, d_model]
        
        # Apply transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            logger.debug(f"   Transformer layer {i+1}: {x.shape}")
        
        # Decode to enhanced CSI (2D output)
        enhanced_csi = self.decoder(x)  # [batch_size, num_subcarriers]
        
        # Apply frequency domain smoothing to CSI output (if enabled)
        if self.smoothing_weight > 0:
            enhanced_csi = self._apply_csi_smoothing(enhanced_csi)
        
        # Apply magnitude constraint to reduce vibrations (always enabled)
        enhanced_csi = self._apply_magnitude_constraint(enhanced_csi, max_magnitude)
        
        # Debug: Log output CSI amplitude statistics
        output_amplitude = torch.abs(enhanced_csi)
        output_nonzero = output_amplitude[output_amplitude > 1e-8]
        if len(output_nonzero) > 0:
            logger.debug(f"📊 Output CSI Amplitude Stats: min={output_nonzero.min().item():.6f}, max={output_nonzero.max().item():.6f}, mean={output_nonzero.mean().item():.6f}")
        else:
            logger.warning("⚠️ Output CSI has no valid amplitude data")
        
        # Remove batch dimension for single sample output
        enhanced_csi = enhanced_csi.squeeze(0)  # [num_subcarriers]
        
        logger.debug(f"✅ CSINetwork enhancement completed: {enhanced_csi.shape}")
        
        return enhanced_csi
    
    def _apply_magnitude_constraint(self, csi: torch.Tensor, max_magnitude: float = 100.0) -> torch.Tensor:
        """
        Apply magnitude constraint to reduce excessive vibrations.
        
        Args:
            csi: Complex CSI tensor [batch_size, num_subcarriers]
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
    
    def _apply_csi_smoothing(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency domain smoothing to complex CSI output.
        
        Args:
            csi: Complex CSI tensor [batch_size, num_subcarriers]
            
        Returns:
            smoothed_csi: Smoothed complex CSI tensor
        """
        if self.smoothing_weight <= 0:
            return csi
            
        logger.debug(f"🔧 Applying CSI smoothing: weight={self.smoothing_weight}, type={self.smoothing_type}")
        
        batch_size, num_subcarriers = csi.shape
        
        # Apply smoothing along frequency dimension for each batch
        smoothed_csi = torch.zeros_like(csi)
        
        for b in range(batch_size):
            # Extract CSI for this batch: [num_subcarriers]
            csi_signal = csi[b, :]  # [num_subcarriers]
            
            # Apply smoothing based on smoothing_type
            if self.smoothing_type == "phase_preserve":
                # For phase_preserve, smooth magnitude but preserve phase structure
                magnitude = torch.abs(csi_signal)
                phase = torch.angle(csi_signal)
                
                # Smooth only magnitude
                smoothed_magnitude = self._smooth_1d_signal(magnitude, self.smoothing_weight * 0.7)
                smoothed_csi[b, :] = smoothed_magnitude * torch.exp(1j * phase)
                
            elif self.smoothing_type == "complex":
                # For complex, smooth both real and imaginary parts
                real_part = csi_signal.real
                imag_part = csi_signal.imag
                
                smoothed_real = self._smooth_1d_signal(real_part, self.smoothing_weight)
                smoothed_imag = self._smooth_1d_signal(imag_part, self.smoothing_weight)
                smoothed_csi[b, :] = torch.complex(smoothed_real, smoothed_imag)
                
            elif self.smoothing_type == "magnitude_only":
                # For magnitude_only, smooth magnitude and keep original phase
                magnitude = torch.abs(csi_signal)
                phase = torch.angle(csi_signal)
                
                smoothed_magnitude = self._smooth_1d_signal(magnitude, self.smoothing_weight * 0.5)
                smoothed_csi[b, :] = smoothed_magnitude * torch.exp(1j * phase)
                
            else:
                # Default: smooth both parts with moderate strength
                real_part = csi_signal.real
                imag_part = csi_signal.imag
                
                smoothed_real = self._smooth_1d_signal(real_part, self.smoothing_weight * 0.6)
                smoothed_imag = self._smooth_1d_signal(imag_part, self.smoothing_weight * 0.6)
                smoothed_csi[b, :] = torch.complex(smoothed_real, smoothed_imag)
        
        logger.debug(f"✅ CSI smoothing completed with type: {self.smoothing_type}")
        return smoothed_csi
    
    def _smooth_1d_signal(self, signal: torch.Tensor, smoothing_weight: float) -> torch.Tensor:
        """
        Apply 1D smoothing to a signal using moving average.
        
        Args:
            signal: 1D signal tensor [num_subcarriers]
            smoothing_weight: Smoothing strength (0-1)
            
        Returns:
            smoothed_signal: Smoothed 1D signal tensor
        """
        num_subcarriers = signal.shape[0]
        
        # Create smoothing kernel (moving average)
        kernel_size = max(3, int(smoothing_weight * num_subcarriers * 0.1))  # Adaptive kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        # Create moving average kernel
        kernel = torch.ones(kernel_size, device=signal.device) / kernel_size
        
        # Apply 1D convolution for smoothing
        signal_padded = torch.nn.functional.pad(signal.unsqueeze(0).unsqueeze(0), 
                                               (kernel_size//2, kernel_size//2), 
                                               mode='reflect')
        smoothed = torch.nn.functional.conv1d(signal_padded, kernel.unsqueeze(0).unsqueeze(0), padding=0)
        
        return smoothed.squeeze()
    
    
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