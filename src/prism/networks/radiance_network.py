"""
RadianceNetwork: Processes UE position, viewing direction, spatial features, and antenna embeddings.

This network outputs N_UE × K radiation values for all UE antenna channels,
capturing the radiation-dependent CSI observed at the UE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class RadianceNetwork(nn.Module):
    """
    RadianceNetwork: Unified processing of UE position, viewing direction, spatial features, and antenna embeddings.
    
    Architecture: Structure similar to the color subnetwork in standard NeRF.
    Unified network processing for both single and multi-antenna cases:
    - Single antenna: num_antennas=1 (special case)
    - Multi-antenna: num_antennas>1 (general case)
    
    Each antenna processes: [Linear-encoded UE_pos, Linear-encoded view_dir, 128D_features, 64D_antenna_embedding] 
    → N_BS × N_UE × K radiation values
    Output: Complex values representing radiation characteristics with antenna dimension.
    """
    
    def __init__(
        self,
        ue_position_dim: int = 16,  # Linear projection of 3D position to N-dimensional features
        view_direction_dim: int = 16,  # Linear projection of 3D direction to N-dimensional features
        feature_dim: int = 128,  # From AttenuationNetwork
        antenna_embedding_dim: int = 64,  # From antenna codebook
        hidden_dim: int = 256,
        num_ue_antennas: int = 1,  # Fixed to 1 UE antenna
        num_subcarriers: int = 64,
        num_layers: int = 4,
        activation: str = "relu",
        complex_output: bool = True,
        dropout: float = 0.1,
        use_skip_connections: bool = True,
        output_dim: int = 32  # Output dimension for ray tracing (R dimension)
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
        self.output_dim = output_dim  # Store output_dim for ray tracing
        
        # Optimize chunk_size for memory efficiency
        self.chunk_size = 512  # Increased for better GPU utilization
        self.use_gradient_checkpointing = True  # Enable gradient checkpointing
        
        # Linear projection layers for encoding raw 3D inputs
        self.ue_position_encoder = nn.Linear(3, ue_position_dim)    # UE position: 3D -> N-dimensional
        self.view_direction_encoder = nn.Linear(3, view_direction_dim)  # View direction: 3D -> N-dimensional
        
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
        
        # Output layer: hidden_dim → output_dim (fixed dimension for ray tracing)
        # The output_dim should match AttenuationNetwork's output_dim for ray tracing
        if self.complex_output:
            # For complex output, we need 2 * output_dim
            output_dim = 2 * self.output_dim  # Use configurable output_dim for ray tracing
        else:
            output_dim = self.output_dim  # Use configurable output_dim for ray tracing
            
        self.output_layer = nn.Linear(self.hidden_dim, output_dim)
        
        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(self.dropout) for _ in range(self.num_layers - 1)
        ])
        
        # Initialize weights properly to prevent NaN
        self._initialize_weights()
        
    def forward(
        self, 
        ue_positions: torch.Tensor,
        view_directions: torch.Tensor,
        features: torch.Tensor,
        antenna_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the RadianceNetwork with chunking for memory optimization.
        
        Handles single BS-UE antenna pair embeddings.
        
        Args:
            ue_positions: Raw UE positions of shape (batch_size, 3) - will be linearly encoded to N-dimensional
            view_directions: IPE-encoded viewing directions of shape (batch_size, view_direction_dim)
            features: 128D feature vectors from AttenuationNetwork of shape (batch_size, feature_dim)
            antenna_embeddings: BS-UE antenna pair embeddings of shape (batch_size, antenna_embedding_dim)
            
        Returns:
            Radiation values of shape (batch_size, output_dim)
        """
        batch_size = antenna_embeddings.shape[0]
        
        # Validate antenna embeddings shape
        if antenna_embeddings.dim() != 2:
            raise ValueError(f"Expected 2D antenna embeddings (batch_size, embedding_dim), got shape: {antenna_embeddings.shape}")
        
        # No need to expand antenna embeddings - they're already in the right format
        
        # Debug: Check input shapes to catch expand errors
        logger.debug(f"🔍 RadianceNetwork forward debug:")
        logger.debug(f"   - ue_positions shape: {ue_positions.shape}")
        logger.debug(f"   - view_directions shape: {view_directions.shape}")
        logger.debug(f"   - features shape: {features.shape}")
        logger.debug(f"   - antenna_embeddings shape: {antenna_embeddings.shape}")
        logger.debug(f"   - batch_size: {batch_size}")
        
        # Encode inputs using linear projections (3D -> N-dimensional)
        if ue_positions.shape[-1] != 3:
            raise ValueError(f"Expected UE positions to have 3 dimensions (x,y,z), but got {ue_positions.shape[-1]}")
        if view_directions.shape[-1] != 3:
            raise ValueError(f"Expected view directions to have 3 dimensions (dx,dy,dz), but got {view_directions.shape[-1]}")
        
        # Linear encoding: (batch_size, 3) -> (batch_size, encoded_dim)
        ue_positions_encoded = self.ue_position_encoder(ue_positions)
        view_directions_encoded = self.view_direction_encoder(view_directions)

        
        # Concatenate inputs: [ue_pos, view_dir, features, antenna_embedding]
        # Shape: (batch_size, total_input_dim)
        x = torch.cat([
            ue_positions_encoded, 
            view_directions_encoded, 
            features, 
            antenna_embeddings
        ], dim=-1)
        
        # Input validation passed - position normalization ensures numerical stability
        
        # Process through network with chunking
        # x is already in the right shape: (batch_size, total_input_dim)
        x_flat = x
        
        # If input is small, process directly
        if x_flat.shape[0] <= self.chunk_size:
            if self.training and self.use_gradient_checkpointing:
                raw_output = checkpoint.checkpoint(
                    self._process_unified_antennas, x_flat, use_reentrant=False
                )
            else:
                raw_output = self._process_unified_antennas(x_flat)
        else:
            # Chunk processing for large inputs with memory cleanup
            raw_outputs = []
            for i in range(0, x_flat.shape[0], self.chunk_size):
                chunk = x_flat[i:i + self.chunk_size]
                
                if self.training and self.use_gradient_checkpointing:
                    raw_output_chunk = checkpoint.checkpoint(
                        self._process_unified_antennas, chunk, use_reentrant=False
                    )
                else:
                    raw_output_chunk = self._process_unified_antennas(chunk)
                
                raw_outputs.append(raw_output_chunk)
                
                # Clear cache after every few chunks
                if (i // self.chunk_size + 1) % 4 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Concatenate all raw chunks
            raw_output = torch.cat(raw_outputs, dim=0)
        
        # Now perform reshape and complex conversion on the full raw output
        # Raw output shape: (batch_size, output_dim)
        if self.complex_output:
            # For complex output, output_dim = 2 * self.output_dim (configurable dimension for ray tracing)
            # Reshape to (batch_size, self.output_dim, 2)
            raw_output = raw_output.view(batch_size, self.output_dim, 2)
            # Convert to complex tensor with float32 for numerical stability
            real_part = raw_output[..., 0].to(torch.float32)
            imag_part = raw_output[..., 1].to(torch.float32)
            # Use float32 complex tensors for numerical stability
            output = real_part + 1j * imag_part  # Complex64
            # Output shape: (batch_size, output_dim)
        else:
            # Output shape: (batch_size, output_dim)
            output = raw_output
        
        return output
    
    def _initialize_weights(self):
        """Initialize network weights to prevent numerical instability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot initialization for better numerical stability
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def _process_unified_antennas(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unified BS-UE antenna pair processing method. Processes flattened input and returns raw output before reshape.
        
        Args:
            x: Flattened input tensor of shape (flattened_batch_size, total_input_dim)
               where flattened_batch_size = batch_size * num_bs_antennas * num_ue_antennas (or chunk thereof)
            
        Returns:
            Raw output tensor of shape (flattened_batch_size, output_dim)
               where output_dim = 2 * self.output_dim if complex_output else self.output_dim
        """
        # NaN debugging removed - issue resolved with position normalization
        
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
        
        # Output layer - return raw output without reshape
        raw_output = self.output_layer(h)
        
        return raw_output
    
    def get_output_shape(self) -> Tuple[int, int, int]:
        """
        Get the output shape for BS-UE antenna pair processing.
        
        Returns:
            Tuple of (num_bs_antennas, num_ue_antennas, output_dim)
            Note: num_bs_antennas and num_ue_antennas are determined at runtime
        """
        return (-1, -1, self.output_dim)  # -1 indicates variable antenna dimensions
    
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
        ue_position_dim: int = 16,  # Linear projection of 3D position to N-dimensional features
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
        use_skip_connections: bool = True,
        output_dim: int = 64  # R-dimensional output for low-rank factorization
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
        self.output_dim = output_dim
    
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
