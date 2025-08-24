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
from typing import Optional, Tuple, Dict, Any

from .attenuation_network import AttenuationNetwork, AttenuationNetworkConfig
from .attenuation_decoder import AttenuationDecoder, AttenuationDecoderConfig
from .antenna_codebook import AntennaEmbeddingCodebook, AntennaEmbeddingCodebookConfig
from .antenna_network import AntennaNetwork, AntennaNetworkConfig
from .radiance_network import RadianceNetwork, RadianceNetworkConfig


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
        
        # Calculate IPE encoding dimensions
        if use_ipe_encoding:
            # IPE encoding: 21 frequencies * 3 dimensions = 63
            self.ipe_position_dim = 63
            self.ipe_direction_dim = 63
        else:
            self.ipe_position_dim = position_dim
            self.ipe_direction_dim = position_dim
        
        # Build network components
        self._build_networks()
        
    def _build_networks(self):
        """Build all network components."""
        
        # 1. AttenuationNetwork: Encode spatial position information
        self.attenuation_network = AttenuationNetwork(
            input_dim=self.ipe_position_dim,
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
            ue_position_dim=self.ipe_position_dim,
            view_direction_dim=self.ipe_direction_dim,
            feature_dim=self.feature_dim,
            antenna_embedding_dim=self.antenna_embedding_dim,
            hidden_dim=self.hidden_dim,
            num_ue_antennas=self.num_ue_antennas,
            num_subcarriers=self.num_subcarriers,
            complex_output=self.complex_output
        )
        
    def forward(
        self,
        sampled_positions: torch.Tensor,
        ue_positions: torch.Tensor,
        view_directions: torch.Tensor,
        antenna_indices: torch.Tensor,
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
        
        # 1. AttenuationNetwork: Encode spatial positions
        # Reshape to (batch_size * num_voxels, 3) for processing
        positions_flat = sampled_positions.view(-1, 3)
        
        # Apply IPE encoding if enabled
        if self.use_ipe_encoding:
            # Note: In practice, you would implement IPE encoding here
            # For now, we'll expand 3D positions to match the expected input dimension
            # This is a placeholder - in practice you would implement proper IPE encoding
            if positions_flat.shape[-1] == 3:
                # Expand 3D positions to 63D (placeholder for IPE encoding)
                encoded_positions = positions_flat.repeat(1, 21)  # 21 frequencies * 3 dimensions
            else:
                encoded_positions = positions_flat
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
                voxel_attenuation = self.attenuation_decoder(voxel_features)
                
            attenuation_factors.append(voxel_attenuation)
        
        # Stack attenuation factors: (batch_size, num_voxels, num_ue_antennas, num_subcarriers)
        attenuation_factors = torch.stack(attenuation_factors, dim=1)
        
        # 3. AntennaEmbeddingCodebook: Get antenna embeddings
        if self.antenna_codebook is not None:
            antenna_embeddings = self.antenna_codebook(antenna_indices)
        else:
            # Fallback: create random embeddings
            if antenna_indices.dim() == 1:
                antenna_embeddings = torch.randn(batch_size, self.antenna_embedding_dim)
            else:
                num_antennas = antenna_indices.shape[1]
                antenna_embeddings = torch.randn(batch_size, num_antennas, self.antenna_embedding_dim)
        
        # 4. AntennaNetwork: Get directional importance
        directional_importance = self.antenna_network(antenna_embeddings)
        
        # Get top-K directions for efficient sampling
        top_k_indices, top_k_importance = self.antenna_network.get_top_k_directions(
            directional_importance, self.top_k_directions
        )
        
        # 5. RadianceNetwork: Get radiation factors
        # Apply IPE encoding to UE positions and view directions if enabled
        if self.use_ipe_encoding:
            # Note: In practice, you would implement IPE encoding here
            # For now, we'll expand 3D positions/directions to match the expected input dimension
            # This is a placeholder - in practice you would implement proper IPE encoding
            if ue_positions.shape[-1] == 3:
                encoded_ue_positions = ue_positions.repeat(1, 21)  # 21 frequencies * 3 dimensions
            else:
                encoded_ue_positions = ue_positions
                
            if view_directions.shape[-1] == 3:
                encoded_view_directions = view_directions.repeat(1, 21)  # 21 frequencies * 3 dimensions
            else:
                encoded_view_directions = view_directions
        else:
            encoded_ue_positions = ue_positions
            encoded_view_directions = view_directions
        
        # Get radiation factors
        radiation_factors = self.radiance_network(
            encoded_ue_positions,
            encoded_view_directions,
            features.mean(dim=1),  # Use mean features across voxels
            antenna_embeddings
        )
        
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
