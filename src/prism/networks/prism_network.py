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
        # Network-specific configurations
        attenuation_network_config: dict = None,
        attenuation_decoder_config: dict = None,
        antenna_codebook_config: dict = None,
        antenna_network_config: dict = None,
        radiance_network_config: dict = None,
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
        
        # Store network-specific configurations with defaults
        self.attenuation_network_config = attenuation_network_config or {}
        self.attenuation_decoder_config = attenuation_decoder_config or {}
        self.antenna_codebook_config = antenna_codebook_config or {}
        self.antenna_network_config = antenna_network_config or {}
        self.radiance_network_config = radiance_network_config or {}
        
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
            hidden_dim=self.attenuation_network_config.get('hidden_dim', self.hidden_dim),
            output_dim=self.attenuation_network_config.get('feature_dim', self.feature_dim),
            num_layers=self.attenuation_network_config.get('num_hidden_layers', 8),
            activation=self.attenuation_network_config.get('activation', 'relu'),
            use_shortcuts=self.attenuation_network_config.get('use_shortcut', True),
            complex_output=self.complex_output
        )
        
        # 2. AttenuationDecoder: Convert features to attenuation factors
        self.attenuation_decoder = AttenuationDecoder(
            feature_dim=self.attenuation_decoder_config.get('input_dim', self.feature_dim),
            hidden_dim=self.attenuation_decoder_config.get('hidden_dim', self.hidden_dim),
            num_layers=self.attenuation_decoder_config.get('num_hidden_layers', 3),
            activation=self.attenuation_decoder_config.get('activation', 'relu'),
            num_ue_antennas=self.attenuation_decoder_config.get('num_ue_antennas', self.num_ue_antennas),
            num_subcarriers=self.attenuation_decoder_config.get('output_dim', self.num_subcarriers),
            complex_output=self.complex_output
        )
        
        # 3. AntennaEmbeddingCodebook: Antenna-specific embeddings
        if self.use_antenna_codebook:
            self.antenna_codebook = AntennaEmbeddingCodebook(
                num_bs_antennas=self.antenna_codebook_config.get('num_antennas', self.num_bs_antennas),
                embedding_dim=self.antenna_codebook_config.get('embedding_dim', self.antenna_embedding_dim)
            )
        else:
            self.antenna_codebook = None
        
        # 4. AntennaNetwork: Directional importance indicators
        self.antenna_network = AntennaNetwork(
            antenna_embedding_dim=self.antenna_network_config.get('input_dim', self.antenna_embedding_dim),
            hidden_dim=self.antenna_network_config.get('hidden_dim', self.hidden_dim // 2),
            azimuth_divisions=self.azimuth_divisions,
            elevation_divisions=self.elevation_divisions,
            num_layers=self.antenna_network_config.get('num_hidden_layers', 2),
            activation=self.antenna_network_config.get('activation', 'relu'),
            dropout=self.antenna_network_config.get('dropout_rate', 0.1)
        )
        
        # 5. RadianceNetwork: Radiation modeling
        self.radiance_network = RadianceNetwork(
            ue_position_dim=self.radiance_network_config.get('ue_pos_dim', self.pe_position_dim),
            view_direction_dim=self.radiance_network_config.get('view_dir_dim', self.pe_direction_dim),
            feature_dim=self.radiance_network_config.get('spatial_feature_dim', self.feature_dim),
            antenna_embedding_dim=self.radiance_network_config.get('antenna_embedding_dim', self.antenna_embedding_dim),
            hidden_dim=self.radiance_network_config.get('hidden_dim', self.hidden_dim),
            num_layers=self.radiance_network_config.get('num_hidden_layers', 4),
            activation=self.radiance_network_config.get('activation', 'relu'),
            num_ue_antennas=self.radiance_network_config.get('num_ue_antennas', self.num_ue_antennas),
            num_subcarriers=self.radiance_network_config.get('output_dim', self.num_subcarriers),
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
        Forward pass through the complete Prism network with unified antenna processing.
        
        This method handles both single and multi-antenna queries uniformly:
        - Single antenna: antenna_indices shape (batch_size,) -> treated as (batch_size, 1)
        - Multi-antenna: antenna_indices shape (batch_size, num_antennas)
        
        Args:
            sampled_positions: Voxel positions for attenuation modeling (batch_size, num_voxels, 3)
            ue_positions: UE positions (batch_size, 3) or (batch_size, num_antennas, 3)
            view_directions: Viewing directions (batch_size, 3) or (batch_size, num_antennas, 3)
            antenna_indices: BS antenna indices (batch_size,) or (batch_size, num_antennas)
            selected_subcarriers: Optional list of subcarrier indices
            return_intermediates: Whether to return intermediate outputs
            
        Returns:
            Dictionary containing outputs with antenna dimension:
            - All outputs have shape (..., num_antennas, ...)
            - For single antenna case, num_antennas=1
        """
        batch_size = sampled_positions.shape[0]
        
        # Normalize antenna_indices to always have antenna dimension
        if len(antenna_indices.shape) == 1:
            # Single antenna case: (batch_size,) -> (batch_size, 1)
            antenna_indices = antenna_indices.unsqueeze(1)
            is_single_antenna = True
        else:
            is_single_antenna = False
            
        num_antennas = antenna_indices.shape[1]
        logger.debug(f"🔍 Processing {num_antennas} antenna(s) (single_mode={is_single_antenna})")
        
        # Enable mixed precision for forward pass if configured
        use_autocast = torch.cuda.is_available() and self.use_mixed_precision
        
        # Use unified multi-antenna processing (single antenna is just num_antennas=1)
        outputs = self._forward_unified_antenna_processing(
            sampled_positions, ue_positions, view_directions, antenna_indices,
            selected_subcarriers, return_intermediates, use_autocast
        )
        
        # For single antenna case, optionally squeeze the antenna dimension for backward compatibility
        if is_single_antenna:
            # Keep antenna dimension for consistency, but could squeeze if needed
            pass
            
        return outputs
    
    def _process_single_antenna(
        self,
        sampled_positions: torch.Tensor,
        ue_positions: torch.Tensor,
        view_directions: torch.Tensor,
        antenna_indices: torch.Tensor,
        selected_subcarriers: Optional[List[int]],
        return_intermediates: bool,
        use_autocast: bool
    ) -> Dict[str, torch.Tensor]:
        """Internal method for single antenna forward pass (original implementation)."""
        batch_size = sampled_positions.shape[0]
        num_voxels = sampled_positions.shape[1]
        
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
                        # If decoder outputs real, create complex from real/imag parts with explicit dtype
                        real_part = real_attenuation.to(torch.float32)
                        imag_part = imag_attenuation.to(torch.float32)
                        voxel_attenuation = torch.complex(real_part, imag_part)
                else:
                    # Features are real, process directly
                    voxel_attenuation = self.attenuation_decoder(voxel_features)
                
                # Add this voxel's attenuation factor to the list
                attenuation_factors.append(voxel_attenuation)
        
        # Stack attenuation factors: (batch_size, num_voxels, num_ue_antennas, num_subcarriers)
        attenuation_factors = torch.stack(attenuation_factors, dim=1)
        
        # Apply subcarrier selection if specified
        if selected_subcarriers is not None:
            # selected_subcarriers contains the indices of subcarriers to use
            attenuation_factors = attenuation_factors[:, :, :, selected_subcarriers]
            logger.debug(f"🔍 Applied subcarrier selection: {len(selected_subcarriers)} subcarriers")
            logger.debug(f"🔍 Attenuation factors shape after selection: {attenuation_factors.shape}")
        
        # 3. AntennaEmbeddingCodebook: Get antenna embeddings
        if self.antenna_codebook is not None:
            # antenna_indices is 1D (batch_size,) for single antenna processing
            antenna_embeddings = self.antenna_codebook(antenna_indices.unsqueeze(1))  # Make it (batch_size, 1)
            antenna_embeddings = antenna_embeddings.squeeze(1)  # Back to (batch_size, embedding_dim)
        else:
            # Fallback: create random embeddings for single antenna
            antenna_embeddings = torch.randn(batch_size, self.antenna_embedding_dim, device=antenna_indices.device)
        
        # 4. AntennaNetwork: Get directional importance
        # AntennaNetwork expects (batch_size, num_antennas, embedding_dim)
        directional_importance = self.antenna_network(antenna_embeddings.unsqueeze(1))
        
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
        
        # RadianceNetwork now expects unified antenna processing format
        # Convert antenna_embeddings from (batch_size, embedding_dim) to (batch_size, 1, embedding_dim)
        antenna_embeddings_unified = antenna_embeddings.unsqueeze(1)  # Add antenna dimension
        
        radiation_factors = self.radiance_network(
            encoded_ue_positions,
            encoded_view_directions,
            mean_features,
            antenna_embeddings_unified  # (batch_size, 1, embedding_dim) for single antenna
        )
        
        # RadianceNetwork now returns (batch_size, num_antennas, num_ue_antennas, num_subcarriers)
        # For single antenna case, squeeze the antenna dimension: (batch_size, 1, num_ue_antennas, num_subcarriers) -> (batch_size, num_ue_antennas, num_subcarriers)
        radiation_factors = radiation_factors.squeeze(1)
        
        # Apply subcarrier selection if specified
        if selected_subcarriers is not None:
            # selected_subcarriers contains the indices of subcarriers to use
            radiation_factors = radiation_factors[:, :, selected_subcarriers]
            logger.debug(f"🔍 Applied subcarrier selection to radiation factors: {len(selected_subcarriers)} subcarriers")
            logger.debug(f"🔍 Radiation factors shape after selection: {radiation_factors.shape}")
        
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
    
    def _forward_unified_antenna_processing(
        self,
        sampled_positions: torch.Tensor,
        ue_positions: torch.Tensor,
        view_directions: torch.Tensor,
        antenna_indices: torch.Tensor,
        selected_subcarriers: Optional[List[int]],
        return_intermediates: bool,
        use_autocast: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Unified antenna processing method that handles both single and multi-antenna cases.
        
        This method processes each antenna individually and then stacks the results.
        Single antenna case is just a special case where num_antennas=1.
        """
        batch_size = sampled_positions.shape[0]
        num_antennas = antenna_indices.shape[1]
        
        logger.debug(f"🔍 Unified processing: {batch_size} samples × {num_antennas} antennas")
        
        # Initialize aggregated outputs
        aggregated_outputs = {
            'attenuation_factors': [],
            'radiation_factors': [],
            'directional_importance': [],
            'top_k_directions': [],
            'top_k_importance': []
        }
        
        # If return_intermediates is requested, also aggregate intermediate results
        if return_intermediates:
            aggregated_outputs.update({
                'spatial_features': [],
                'antenna_embeddings': []
            })
        
        # Process each antenna serially
        for antenna_idx in range(num_antennas):
            # Extract single antenna indices for this iteration
            single_antenna_indices = antenna_indices[:, antenna_idx]  # (batch_size,)
            
            # Process this single antenna
            single_antenna_output = self._process_single_antenna(
                sampled_positions=sampled_positions,
                ue_positions=ue_positions,
                view_directions=view_directions,
                antenna_indices=single_antenna_indices,
                selected_subcarriers=selected_subcarriers,
                return_intermediates=return_intermediates,
                use_autocast=use_autocast
            )
            
            # Aggregate the outputs
            for key in ['attenuation_factors', 'radiation_factors', 'directional_importance', 
                       'top_k_directions', 'top_k_importance']:
                if key in single_antenna_output:
                    aggregated_outputs[key].append(single_antenna_output[key])
            
            # Aggregate intermediate outputs if requested
            if return_intermediates:
                for key in ['spatial_features', 'antenna_embeddings']:
                    if key in single_antenna_output:
                        aggregated_outputs[key].append(single_antenna_output[key])
        
        # Stack the results along the antenna dimension (dim=1)
        final_outputs = {}
        logger.debug(f"🔍 Stacking results for {len(aggregated_outputs)} keys")
        
        for key, value_list in aggregated_outputs.items():
            if not value_list:
                continue
                
            try:
                final_outputs[key] = self._stack_antenna_outputs(key, value_list, num_antennas)
                logger.debug(f"✅ Stacked {key}: {final_outputs[key].shape}")
            except Exception as e:
                logger.error(f"❌ Failed to stack {key}: {e}")
                final_outputs[key] = value_list
        
        return final_outputs
    
    def _stack_antenna_outputs(self, key: str, value_list: List[torch.Tensor], num_antennas: int) -> torch.Tensor:
        """
        Stack antenna outputs along the antenna dimension.
        
        Args:
            key: Output key name
            value_list: List of tensors from each antenna
            num_antennas: Number of antennas
            
        Returns:
            Stacked tensor with antenna dimension at dim=1
        """
        if key == 'attenuation_factors':
            # Special handling for attenuation_factors which has shape (batch, 64, ue_antennas, subcarriers)
            # We need to select the correct antenna index from each tensor
            stacked_tensors = []
            for antenna_idx, tensor in enumerate(value_list):
                # Select the specific antenna from the 64 BS antennas
                antenna_tensor = tensor[:, antenna_idx:antenna_idx+1, :, :]  # Keep dim for concatenation
                stacked_tensors.append(antenna_tensor)
            return torch.cat(stacked_tensors, dim=1)
            
        elif key in ['directional_importance', 'top_k_directions', 'top_k_importance']:
            # These tensors have an extra singleton dimension that needs to be squeezed first
            squeezed_tensors = []
            for tensor in value_list:
                # Remove the singleton dimension at position 1
                squeezed = tensor.squeeze(1)
                squeezed_tensors.append(squeezed)
            return torch.stack(squeezed_tensors, dim=1)
            
        else:
            # Standard stacking for radiation_factors and other tensors
            return torch.stack(value_list, dim=1)
    
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
