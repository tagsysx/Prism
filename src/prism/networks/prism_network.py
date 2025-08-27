"""
PrismNetwork: Main integrated network combining all four network components.

This module integrates:
1. AttenuationNetwork: Encodes spatial position information
2. AttenuationDecoder: Converts features to attenuation factors
3. AntennaEmbeddingCodebook: Provides antenna-specific embeddings
4. AntNetwork: Generates directional importance indicators
5. RadianceNetwork: Processes inputs for radiation modeling
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple, Dict, Any, List

# Setup logger
logger = logging.getLogger(__name__)

from .attenuation_network import AttenuationNetwork, AttenuationNetworkConfig
from .attenuation_decoder import AttenuationDecoder, AttenuationDecoderConfig
from .antenna_codebook import AntennaEmbeddingCodebook, AntennaEmbeddingCodebookConfig
from .antenna_network import AntennaNetwork, AntennaNetworkConfig
from .radiance_network import RadianceNetwork, RadianceNetworkConfig
from .positional_encoder import PositionalEncoder, create_position_encoder, create_direction_encoder


class PrismNetwork(nn.Module):
    """
    PrismNetwork: Main integrated network for discrete electromagnetic ray tracing.
    
    This network implements the complete Prism architecture as described in the design docs:
    - Takes spatial positions and outputs CSI values for OFDM communication systems
    - Processes N_BS antenna array and N_UE UE antenna array
    - Outputs K-dimensional CSI values for each base station antenna
    """
    
    def __init__(
        self,
        num_subcarriers: int = 64,
        num_ue_antennas: int = 4,
        num_bs_antennas: int = 64,
        position_dim: int = 3,
        hidden_dim: int = 256,
        feature_dim: int = 128,
        antenna_embedding_dim: int = 64,
        use_antenna_codebook: bool = True,
        use_ipe_encoding: bool = True,
        azimuth_divisions: int = 16,
        elevation_divisions: int = 8,
        top_k_directions: int = 32,
        complex_output: bool = True,
        use_mixed_precision: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Store configuration
        self.num_subcarriers = num_subcarriers
        self.num_ue_antennas = num_ue_antennas
        self.num_bs_antennas = num_bs_antennas
        self.position_dim = position_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.antenna_embedding_dim = antenna_embedding_dim
        self.use_antenna_codebook = use_antenna_codebook
        self.use_ipe_encoding = use_ipe_encoding
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.top_k_directions = top_k_directions
        self.complex_output = complex_output
        
        # Configure mixed precision from parameter
        self.use_mixed_precision = use_mixed_precision
        self.attenuation_use_mixed_precision = use_mixed_precision
        self.antenna_use_mixed_precision = use_mixed_precision
        self.radiance_use_mixed_precision = use_mixed_precision
        
        # Initialize positional encoders
        if use_ipe_encoding:
            # Use traditional PE encoding instead of IPE for now
            self.position_encoder = create_position_encoder()  # 10 frequencies, include_input=True
            self.direction_encoder = create_position_encoder()  # Use same as position (10 frequencies) for consistency
            
            # Calculate PE encoding dimensions
            self.pe_position_dim = self.position_encoder.get_output_dim()  # 3 + 2*10*3 = 63
            self.pe_direction_dim = self.direction_encoder.get_output_dim()  # 3 + 2*10*3 = 63
        else:
            self.position_encoder = None
            self.direction_encoder = None
            self.pe_position_dim = position_dim
            self.pe_direction_dim = position_dim
        
        # Build network components
        self._build_networks()
        
    def _build_networks(self):
        """Build all network components."""
        
        # 1. AttenuationNetwork: Encode spatial position information
        self.attenuation_network = AttenuationNetwork(
            input_dim=self.pe_position_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.feature_dim,
            complex_output=self.complex_output
        )
        
        # 2. AttenuationDecoder: Convert features to attenuation factors
        self.attenuation_decoder = AttenuationDecoder(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_ue_antennas=self.num_ue_antennas,
            num_subcarriers=self.num_subcarriers,
            complex_output=self.complex_output
        )
        
        # 3. AntennaEmbeddingCodebook: Antenna-specific embeddings
        if self.use_antenna_codebook:
            self.antenna_codebook = AntennaEmbeddingCodebook(
                num_bs_antennas=self.num_bs_antennas,
                embedding_dim=self.antenna_embedding_dim
            )
        else:
            self.antenna_codebook = None
        
        # 4. AntennaNetwork: Directional importance indicators
        self.antenna_network = AntennaNetwork(
            antenna_embedding_dim=self.antenna_embedding_dim,
            hidden_dim=self.hidden_dim // 2,  # Smaller for efficiency
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions
        )
        
        # 5. RadianceNetwork: Radiation modeling
        self.radiance_network = RadianceNetwork(
            ue_position_dim=self.pe_position_dim,
            view_direction_dim=self.pe_direction_dim,
            feature_dim=self.feature_dim,
            antenna_embedding_dim=self.antenna_embedding_dim,
            hidden_dim=self.hidden_dim,
            num_ue_antennas=self.num_ue_antennas,
            num_subcarriers=self.num_subcarriers,
            complex_output=self.complex_output
        )
        
        # Initialize network weights for better training
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization for better training."""
        logger.info("🔧 Initializing PrismNetwork weights with Xavier/Glorot initialization...")
        
        # Initialize all linear layers with Xavier/Glorot initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                logger.debug(f"   ✓ Initialized {module.__class__.__name__} with Xavier uniform")
            
            elif isinstance(module, nn.LayerNorm):
                # Initialize LayerNorm with standard values
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                logger.debug(f"   ✓ Initialized {module.__class__.__name__} with standard values")
        
        # Initialize antenna codebook embeddings if present
        if hasattr(self, 'antenna_codebook') and self.antenna_codebook is not None:
            if hasattr(self.antenna_codebook, 'embedding'):
                nn.init.xavier_uniform_(self.antenna_codebook.embedding.weight)
                logger.debug(f"   ✓ Initialized AntennaEmbeddingCodebook with Xavier uniform")
        
        logger.info("✅ PrismNetwork weight initialization completed")
        
    def forward(
        self,
        sampled_positions: torch.Tensor,
        ue_positions: torch.Tensor,
        view_directions: torch.Tensor,
        antenna_indices: torch.Tensor,
        selected_subcarriers: Optional[List[int]] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete Prism network.
        
        Args:
            sampled_positions: Voxel positions for attenuation modeling (batch_size, num_voxels, 3)
            ue_positions: UE positions (batch_size, 3) or (batch_size, num_antennas, 3)
            view_directions: Viewing directions (batch_size, 3) or (batch_size, num_antennas, 3)
            antenna_indices: BS antenna indices (batch_size,) or (batch_size, num_antennas)
            return_intermediates: Whether to return intermediate outputs
            
        Returns:
            Dictionary containing final outputs and optionally intermediate values
        """
        batch_size = sampled_positions.shape[0]
        num_voxels = sampled_positions.shape[1]
        
        # Enable mixed precision for forward pass if configured
        use_autocast = torch.cuda.is_available() and self.use_mixed_precision
        
        with torch.amp.autocast('cuda', enabled=use_autocast):
            # 1. AttenuationNetwork: Encode spatial positions
            # Reshape to (batch_size * num_voxels, 3) for processing
            positions_flat = sampled_positions.view(-1, 3)
            
            # Apply positional encoding if enabled
            if self.use_ipe_encoding and self.position_encoder is not None:
                # Use traditional PE encoding
                encoded_positions = self.position_encoder(positions_flat)
            else:
                encoded_positions = positions_flat
            
            # Get spatial features
            features = self.attenuation_network(encoded_positions)
            # Reshape back to (batch_size, num_voxels, feature_dim)
            features = features.view(batch_size, num_voxels, self.feature_dim)
            
            # Keep complex features throughout the computation - DO NOT convert to real
            # Complex features preserve both magnitude and phase information
            
            # 2. AttenuationDecoder: Get attenuation factors
            # Process each voxel's features - handle complex features properly
            attenuation_factors = []
            for i in range(num_voxels):
                voxel_features = features[:, i, :]  # (batch_size, feature_dim)
            
                # If features are complex, process real and imaginary parts separately
                if voxel_features.is_complex():
                    # Process real and imaginary parts through the decoder
                    real_features = voxel_features.real
                    imag_features = voxel_features.imag
                    
                    real_attenuation = self.attenuation_decoder(real_features)
                    imag_attenuation = self.attenuation_decoder(imag_features)
                    
                    # Combine to form complex attenuation
                    if self.attenuation_decoder.is_complex():
                        # If decoder outputs complex, combine properly
                        voxel_attenuation = real_attenuation + 1j * imag_attenuation
                    else:
                        # If decoder outputs real, create complex from real/imag parts
                        voxel_attenuation = torch.complex(real_attenuation, imag_attenuation)
                else:
                    # Features are real, process directly
                    voxel_attenuation = self.attenuation_decoder(voxel_features)
                
                # Add this voxel's attenuation factor to the list
                attenuation_factors.append(voxel_attenuation)
        
        # Stack attenuation factors: (batch_size, num_voxels, num_ue_antennas, num_subcarriers)
        attenuation_factors = torch.stack(attenuation_factors, dim=1)
        
        # 3. AntennaEmbeddingCodebook: Get antenna embeddings (now unified to 3D output)
        if self.antenna_codebook is not None:
            antenna_embeddings = self.antenna_codebook(antenna_indices)  # Always returns (batch_size, num_antennas, embedding_dim)
        else:
            # Fallback: create random embeddings in unified 3D format
            if antenna_indices.dim() == 1:
                # Single antenna case: create (batch_size, 1, embedding_dim)
                antenna_embeddings = torch.randn(batch_size, 1, self.antenna_embedding_dim, device=antenna_indices.device)
            else:
                # Multiple antennas case: create (batch_size, num_antennas, embedding_dim)
                num_antennas = antenna_indices.shape[1]
                antenna_embeddings = torch.randn(batch_size, num_antennas, self.antenna_embedding_dim, device=antenna_indices.device)
        
        # 4. AntennaNetwork: Get directional importance
        directional_importance = self.antenna_network(antenna_embeddings)
        
        # Get top-K directions for efficient sampling
        top_k_indices, top_k_importance = self.antenna_network.get_top_k_directions(
            directional_importance, self.top_k_directions
        )
        
        # 5. RadianceNetwork: Get radiation factors
        # Apply positional encoding to UE positions and view directions if enabled
        if self.use_ipe_encoding and self.position_encoder is not None and self.direction_encoder is not None:
            # Use traditional PE encoding
            encoded_ue_positions = self.position_encoder(ue_positions)
            encoded_view_directions = self.direction_encoder(view_directions)
        else:
            encoded_ue_positions = ue_positions
            encoded_view_directions = view_directions
        
        # Get radiation factors
        # Convert complex features to real for RadianceNetwork input
        mean_features = features.mean(dim=1)  # Use mean features across voxels
        if torch.is_complex(mean_features):
            # Convert complex features to real by taking magnitude
            mean_features = torch.abs(mean_features)
        
        radiation_factors = self.radiance_network(
            encoded_ue_positions,
            encoded_view_directions,
            mean_features,
            antenna_embeddings
        )
        
        # Debug: Check the computed factors
        logger.debug(f"🔍 PrismNetwork forward outputs:")
        logger.debug(f"   - attenuation_factors shape: {attenuation_factors.shape}")
        logger.debug(f"   - radiation_factors shape: {radiation_factors.shape}")
        logger.debug(f"   - attenuation_factors abs max: {torch.abs(attenuation_factors).max() if attenuation_factors.numel() > 0 else 'N/A'}")
        logger.debug(f"   - radiation_factors abs max: {torch.abs(radiation_factors).max() if radiation_factors.numel() > 0 else 'N/A'}")
        logger.debug(f"   - attenuation_factors sample: {attenuation_factors[:2, :2, :2, :2] if attenuation_factors.numel() > 0 else 'empty'}")

        
        # Prepare outputs
        outputs = {
            'attenuation_factors': attenuation_factors,
            'radiation_factors': radiation_factors,
            'directional_importance': directional_importance,
            'top_k_directions': top_k_indices,
            'top_k_importance': top_k_importance
        }
        
        if return_intermediates:
            outputs.update({
                'spatial_features': features,
                'antenna_embeddings': antenna_embeddings
            })
        
        return outputs
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the network architecture."""
        return {
            'num_subcarriers': self.num_subcarriers,
            'num_ue_antennas': self.num_ue_antennas,
            'num_bs_antennas': self.num_bs_antennas,
            'feature_dim': self.feature_dim,
            'antenna_embedding_dim': self.antenna_embedding_dim,
            'directional_resolution': (self.azimuth_divisions, self.elevation_divisions),
            'top_k_directions': self.top_k_directions,
            'complex_output': self.complex_output,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the network configuration."""
        return {
            'num_subcarriers': self.num_subcarriers,
            'num_ue_antennas': self.num_ue_antennas,
            'num_bs_antennas': self.num_bs_antennas,
            'position_dim': self.position_dim,
            'hidden_dim': self.hidden_dim,
            'feature_dim': self.feature_dim,
            'antenna_embedding_dim': self.antenna_embedding_dim,
            'use_antenna_codebook': self.use_antenna_codebook,
            'use_ipe_encoding': self.use_ipe_encoding,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions,
            'top_k_directions': self.top_k_directions,
            'complex_output': self.complex_output
        }


class PrismNetworkConfig:
    """Configuration class for PrismNetwork."""
    
    def __init__(
        self,
        num_subcarriers: int = 64,
        num_ue_antennas: int = 4,
        num_bs_antennas: int = 64,
        position_dim: int = 3,
        hidden_dim: int = 256,
        feature_dim: int = 128,
        antenna_embedding_dim: int = 64,
        use_antenna_codebook: bool = True,
        use_ipe_encoding: bool = True,
        azimuth_divisions: int = 16,
        elevation_divisions: int = 8,
        top_k_directions: int = 32,
        complex_output: bool = True
    ):
        self.num_subcarriers = num_subcarriers
        self.num_ue_antennas = num_ue_antennas
        self.num_bs_antennas = num_bs_antennas
        self.position_dim = position_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.antenna_embedding_dim = antenna_embedding_dim
        self.use_antenna_codebook = use_antenna_codebook
        self.use_ipe_encoding = use_ipe_encoding
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.top_k_directions = top_k_directions
        self.complex_output = complex_output
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'num_subcarriers': self.num_subcarriers,
            'num_ue_antennas': self.num_ue_antennas,
            'num_bs_antennas': self.num_bs_antennas,
            'position_dim': self.position_dim,
            'hidden_dim': self.hidden_dim,
            'feature_dim': self.feature_dim,
            'antenna_embedding_dim': self.antenna_embedding_dim,
            'use_antenna_codebook': self.use_antenna_codebook,
            'use_ipe_encoding': self.use_ipe_encoding,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions,
            'top_k_directions': self.top_k_directions,
            'complex_output': self.complex_output
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PrismNetworkConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
