"""
PrismNetwork: Main integrated network combining all four network components.

This module integrates:
1. AttenuationNetwork: Encodes spatial position information
2. AttenuationDecoder: Converts features to attenuation factors
3. AntennaEmbeddingCodebook: Provides antenna-specific embeddings
4. AntennaNetwork: Generates directional importance indicators
5. RadianceNetwork: Processes inputs for radiation modeling
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple, Dict, Any, List

# Setup logger
logger = logging.getLogger(__name__)

from .attenuation_network import AttenuationNetwork, AttenuationNetworkConfig
from .frequency_network import FrequencyNetwork, FrequencyNetworkConfig
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
        num_subcarriers: int = 408,
        num_bs_antennas: int = 64,
        feature_dim: int = 128,
        antenna_embedding_dim: int = 64,
        azimuth_divisions: int = 36,
        elevation_divisions: int = 10,
        use_ipe_encoding: bool = True,
        use_mixed_precision: bool = False,
        # Ray tracing configuration
        max_ray_length: float = 250.0,
        num_sampling_points: int = 64,
        # Network-specific configurations (simplified)
        attenuation_network_config: dict = None,
        frequency_network_config: dict = None,
        radiance_network_config: dict = None,
        **kwargs
    ):
        super().__init__()
        
        # Store essential configuration
        self.num_subcarriers = num_subcarriers
        self.num_bs_antennas = num_bs_antennas
        self.feature_dim = feature_dim
        self.antenna_embedding_dim = antenna_embedding_dim
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.use_ipe_encoding = use_ipe_encoding
        self.use_mixed_precision = use_mixed_precision
        
        # Ray tracing configuration
        self.max_ray_length = max_ray_length
        self.num_sampling_points = num_sampling_points
        
        # Fixed parameters for ray tracing
        self.position_dim = 3  # Always 3D coordinates
        self.use_antenna_codebook = True  # Always use antenna codebook
        self.complex_output = True  # Always use complex output for RF signals
        
        # Store simplified network configurations
        self.attenuation_network_config = attenuation_network_config or {}
        self.frequency_network_config = frequency_network_config or {}
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
            self.pe_position_dim = self.position_dim
            self.pe_direction_dim = self.position_dim
        
        # Build network components
        self._build_networks()
        
    def _build_networks(self):
        """Build essential network components for ray tracing."""
        
        # Default hidden dimension
        default_hidden_dim = 256
        
        # 1. AttenuationNetwork: Compute attenuation coefficients and feature vectors
        self.attenuation_network = AttenuationNetwork(
            input_dim=self.pe_position_dim,
            hidden_dim=self.attenuation_network_config.get('hidden_dim', default_hidden_dim),
            feature_dim=self.attenuation_network_config.get('feature_dim', self.feature_dim),
            output_dim=self.attenuation_network_config.get('output_dim', 32),
            num_layers=self.attenuation_network_config.get('num_layers', 8),
            activation=self.attenuation_network_config.get('activation', 'relu'),
            use_shortcuts=self.attenuation_network_config.get('use_shortcuts', True)
        )
        
        # 2. FrequencyNetwork: Generate R-dimensional frequency basis
        self.frequency_network = FrequencyNetwork(
            input_dim=self.frequency_network_config.get('input_dim', 63),  # PE-encoded frequency
            hidden_dim=self.frequency_network_config.get('hidden_dim', 128),
            output_dim=self.frequency_network_config.get('output_dim', 32),  # R-dimensional
            num_layers=self.frequency_network_config.get('num_layers', 3),
            activation=self.frequency_network_config.get('activation', 'relu'),
            use_layer_norm=self.frequency_network_config.get('use_layer_norm', True)
        )
        
        # 3. AntennaEmbeddingCodebook: Antenna-specific embeddings (always enabled)
        self.antenna_codebook = AntennaEmbeddingCodebook(
            num_bs_antennas=self.num_bs_antennas,
            embedding_dim=self.antenna_embedding_dim
        )
        
        # 4. RadianceNetwork: Radiation modeling
        self.radiance_network = RadianceNetwork(
            ue_position_dim=self.radiance_network_config.get('ue_pos_dim', self.pe_position_dim),
            view_direction_dim=self.radiance_network_config.get('view_dir_dim', self.pe_direction_dim),
            feature_dim=self.radiance_network_config.get('spatial_feature_dim', self.feature_dim),
            antenna_embedding_dim=self.radiance_network_config.get('antenna_embedding_dim', self.antenna_embedding_dim),
            hidden_dim=self.radiance_network_config.get('hidden_dim', default_hidden_dim),
            num_layers=self.radiance_network_config.get('num_hidden_layers', 4),
            activation=self.radiance_network_config.get('activation', 'relu'),
            num_ue_antennas=1,  # Always 1 for single UE processing
            num_subcarriers=self.radiance_network_config.get('output_dim', self.num_subcarriers),
            complex_output=self.complex_output
        )
        
        # Initialize network weights for better training
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        logger.info("🔧 Initializing PrismNetwork weights...")
        
        # Initialize all linear layers with Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize antenna codebook embeddings
        if hasattr(self.antenna_codebook, 'embedding'):
            nn.init.xavier_uniform_(self.antenna_codebook.embedding.weight)
        
        logger.info("✅ PrismNetwork weight initialization completed")
    
    def _generate_uniform_directions(self) -> torch.Tensor:
        """
        Generate A×B uniform direction vectors for ray tracing.
        
        Returns:
            directions: (A*B, 3) - Unit direction vectors covering spherical space
        """
        import math
        
        directions = []
        
        # Get angular sampling parameters from configuration
        azimuth_divisions = self.azimuth_divisions      # A = 36
        elevation_divisions = self.elevation_divisions  # B = 10
        
        # Calculate angular resolutions
        azimuth_resolution = 2 * math.pi / azimuth_divisions      # 360° / 36 = 10°
        elevation_resolution = math.pi / 2 / elevation_divisions  # 90° / 10 = 9°
        
        for i in range(azimuth_divisions):
            for j in range(elevation_divisions):
                # Azimuth: 0° to 360° (0 to 2π)
                phi = i * azimuth_resolution
                # Elevation: 0° to 90° (0 to π/2)
                theta = j * elevation_resolution
                
                # Convert spherical coordinates to Cartesian unit vectors
                # x = cos(elevation) * cos(azimuth)
                # y = cos(elevation) * sin(azimuth)  
                # z = sin(elevation)
                x = math.cos(theta) * math.cos(phi)
                y = math.cos(theta) * math.sin(phi)
                z = math.sin(theta)
                
                directions.append([x, y, z])
        
        return torch.tensor(directions, dtype=torch.float32)
    
    def _sample_points_along_ray(
        self, 
        bs_position: torch.Tensor,
        direction: torch.Tensor,
        max_length: float,
        num_points: int
    ) -> torch.Tensor:
        """
        Sample points uniformly along a ray from BS position.
        
        Args:
            bs_position: Base station position (3,)
            direction: Unit direction vector (3,)
            max_length: Maximum ray length in meters
            num_points: Number of sampling points along the ray
            
        Returns:
            sampled_positions: (num_points, 3) - Voxel coordinates along the ray
        """
        # Generate uniform sampling distances along the ray
        # t ∈ [0, max_length] with num_points samples
        t_values = torch.linspace(0, max_length, num_points, dtype=torch.float32, device=bs_position.device)
        
        # Ray equation: P(t) = bs_position + direction * t
        # Expand dimensions for broadcasting: (num_points, 1) * (1, 3) + (1, 3)
        sampled_positions = bs_position.unsqueeze(0) + direction.unsqueeze(0) * t_values.unsqueeze(1)
        
        return sampled_positions  # (num_points, 3)
        
    def forward(
        self,
        bs_position: torch.Tensor,
        ue_position: torch.Tensor,
        antenna_index: int,
        selected_subcarriers: Optional[List[int]] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete Prism network with ray tracing approach.
        
        Generates rays from BS antenna in A×B uniform directions, samples voxels along each ray,
        and processes them through the neural networks to get attenuation and radiation vectors.
        
        Args:
            bs_position: Base station position (3,) - Single BS position coordinates
            ue_position: User equipment position (3,) - Single UE position coordinates  
            antenna_index: BS antenna index (int) - Single antenna to process
            selected_subcarriers: Optional list of subcarrier indices for frequency basis vectors
            return_intermediates: Whether to return intermediate outputs
            
        Returns:
            Dictionary containing outputs with specific dimensions:
            
            Core outputs:
            - 'attenuation_vectors': (A*B, num_sampling_points, output_dim)
                Complex attenuation coefficients for each voxel along each ray direction
            - 'radiation_vectors': (A*B, num_sampling_points, output_dim)  
                Complex radiation factors for each voxel along each ray direction
            - 'frequency_basis_vectors': (num_selected_subcarriers, output_dim)
                Complex frequency basis vectors for selected subcarriers from FrequencyNetwork
            - 'sampled_positions': (A*B, num_sampling_points, 3)
                Voxel coordinates for all ray directions
            - 'directions': (A*B, 3)
                Unit direction vectors for all rays
                
            Intermediate outputs (if return_intermediates=True):
            - 'spatial_features': (A*B, num_sampling_points, feature_dim)
                Spatial feature vectors from AttenuationNetwork
            - 'antenna_embedding': (1, antenna_embedding_dim)
                Antenna embedding vector from AntennaEmbeddingCodebook
            
        Raises:
            ValueError: If any input tensor doesn't have the required dimensions
        """
        # Validate input tensor dimensions
        if len(bs_position.shape) != 1 or bs_position.shape[0] != 3:
            raise ValueError(f"bs_position must have shape (3,), but got shape {bs_position.shape}.")
        
        if len(ue_position.shape) != 1 or ue_position.shape[0] != 3:
            raise ValueError(f"ue_position must have shape (3,), but got shape {ue_position.shape}.")
        
        if not isinstance(antenna_index, int):
            raise ValueError(f"antenna_index must be an integer, but got {type(antenna_index)}.")
        
        logger.debug(f"🔍 Processing antenna {antenna_index} from BS at {bs_position} to UE at {ue_position}")
        
        # Get ray tracing parameters from network configuration
        max_ray_length = self.max_ray_length
        num_sampling_points = self.num_sampling_points
        
        # Enable mixed precision for forward pass if configured
        use_autocast = torch.cuda.is_available() and self.use_mixed_precision
        
        with torch.amp.autocast('cuda', enabled=use_autocast):
            # 1. Generate uniform ray directions (A×B directions)
            directions = self._generate_uniform_directions()  # (A*B, 3)
            directions = directions.to(bs_position.device)
            num_directions = directions.shape[0]  # A*B = 36*10 = 360
            
            logger.debug(f"🔍 Generated {num_directions} uniform ray directions")
            
            # 2. Get antenna embedding from codebook
            if self.use_antenna_codebook and self.antenna_codebook is not None:
                antenna_embedding = self.antenna_codebook(torch.tensor([antenna_index], device=bs_position.device))
                antenna_embedding = antenna_embedding.squeeze(0)  # (antenna_embedding_dim,)
            else:
                # Fallback: random embedding for testing
                antenna_embedding = torch.randn(self.antenna_embedding_dim, device=bs_position.device)
            
            logger.debug(f"🔍 Retrieved antenna embedding with shape {antenna_embedding.shape}")
            
            # Initialize output tensors
            attenuation_vectors = []
            radiation_vectors = []
            all_sampled_positions = []
            all_spatial_features = []
            
            # 3. Process each ray direction
            for dir_idx in range(num_directions):
                direction = directions[dir_idx]  # (3,)
                
                # 3.1 Sample points along this ray
                sampled_positions = self._sample_points_along_ray(
                    bs_position, direction, max_ray_length, num_sampling_points
                )  # (num_sampling_points, 3)
                
                # 3.2 & 3.3: Process each voxel individually
                # Apply IPE encoding to UE position and negative direction (once per ray)
                neg_direction = -direction  # Viewing direction is opposite to ray direction
                
                if self.use_ipe_encoding and self.position_encoder is not None and self.direction_encoder is not None:
                    encoded_ue_position = self.position_encoder(ue_position)  # (encoded_dim,)
                    encoded_neg_direction = self.direction_encoder(neg_direction)  # (encoded_dim,)
                else:
                    encoded_ue_position = ue_position  # (3,)
                    encoded_neg_direction = neg_direction  # (3,)
                
                # Process each voxel position individually
                voxel_attenuation_coeffs = []
                voxel_radiation_vectors = []
                voxel_features_list = []
                
                for voxel_idx in range(num_sampling_points):
                    voxel_position = sampled_positions[voxel_idx]  # (3,)
                    
                    # 3.2.1 AttenuationNetwork: Process single voxel position
                    if self.use_ipe_encoding and self.position_encoder is not None:
                        encoded_voxel_position = self.position_encoder(voxel_position)  # (encoded_dim,)
                    else:
                        encoded_voxel_position = voxel_position  # (3,)
                    
                    # AttenuationNetwork expects batch dimensions
                    attenuation_coeff, voxel_features = self.attenuation_network(encoded_voxel_position.unsqueeze(0).unsqueeze(0))
                    attenuation_coeff = attenuation_coeff.squeeze(0).squeeze(0)  # (output_dim,)
                    voxel_features = voxel_features.squeeze(0).squeeze(0)  # (feature_dim,)
                    
                    # 3.2.2 Convert complex features to real for RadianceNetwork input
                    real_voxel_features = voxel_features
                    if torch.is_complex(real_voxel_features):
                        real_voxel_features = torch.abs(real_voxel_features)
                    
                    # 3.3 RadianceNetwork: Process single voxel radiation
                    # RadianceNetwork expects batch dimensions
                    radiation_factor = self.radiance_network(
                        encoded_ue_position.unsqueeze(0).unsqueeze(0),    # (1, 1, encoded_dim)
                        encoded_neg_direction.unsqueeze(0).unsqueeze(0),  # (1, 1, encoded_dim)
                        real_voxel_features.unsqueeze(0).unsqueeze(0),    # (1, 1, feature_dim)
                        antenna_embedding.unsqueeze(0).unsqueeze(0)       # (1, 1, embedding_dim)
                    )
                    radiation_factor = radiation_factor.squeeze(0).squeeze(0)  # (output_dim,)
                    
                    # Store results for this voxel
                    voxel_attenuation_coeffs.append(attenuation_coeff)
                    voxel_radiation_vectors.append(radiation_factor)
                    voxel_features_list.append(voxel_features)  # Store original features for intermediates
                
                # Stack results for all voxels in this ray
                attenuation_coeff = torch.stack(voxel_attenuation_coeffs, dim=0)  # (num_sampling_points, output_dim)
                radiation_vector = torch.stack(voxel_radiation_vectors, dim=0)    # (num_sampling_points, output_dim)
                features = torch.stack(voxel_features_list, dim=0)               # (num_sampling_points, feature_dim)
                
                # Store results for this direction
                attenuation_vectors.append(attenuation_coeff)
                radiation_vectors.append(radiation_vector)
                all_sampled_positions.append(sampled_positions)
                all_spatial_features.append(features)
            
            # 4. Stack results from all directions
            attenuation_vectors = torch.stack(attenuation_vectors, dim=0)  # (A*B, num_sampling_points, output_dim)
            radiation_vectors = torch.stack(radiation_vectors, dim=0)      # (A*B, num_sampling_points, output_dim)
            
            logger.debug(f"🔍 Processed {num_directions} ray directions with shapes: "
                        f"attenuation={attenuation_vectors.shape}, radiation={radiation_vectors.shape}")
            
            # 5. FrequencyNetwork: Generate frequency basis vectors
            if selected_subcarriers is not None:
                # Normalize subcarrier indices to [0, 1] range
                normalized_frequencies = torch.tensor(
                    selected_subcarriers, dtype=torch.float32, device=bs_position.device
                ) / self.num_subcarriers
                
                # Apply IPE encoding to frequencies
                if self.use_ipe_encoding and self.position_encoder is not None:
                    encoded_frequencies = self.position_encoder(normalized_frequencies.unsqueeze(-1))
                else:
                    encoded_frequencies = normalized_frequencies.unsqueeze(-1)
                
                # Get frequency basis vectors
                frequency_basis_vectors = self.frequency_network(encoded_frequencies)
                logger.debug(f"🔍 Generated frequency basis vectors for {len(selected_subcarriers)} selected subcarriers")
            else:
                # Generate for all subcarriers
                all_subcarrier_indices = torch.arange(self.num_subcarriers, dtype=torch.float32, device=bs_position.device)
                normalized_frequencies = all_subcarrier_indices / self.num_subcarriers
                
                if self.use_ipe_encoding and self.position_encoder is not None:
                    encoded_frequencies = self.position_encoder(normalized_frequencies.unsqueeze(-1))
                else:
                    encoded_frequencies = normalized_frequencies.unsqueeze(-1)
                
                frequency_basis_vectors = self.frequency_network(encoded_frequencies)
                logger.debug(f"🔍 Generated frequency basis vectors for all {self.num_subcarriers} subcarriers")
            
            # 6. Prepare core outputs (always included)
            all_sampled_positions = torch.stack(all_sampled_positions, dim=0)  # (A*B, num_sampling_points, 3)
            
            outputs = {
                'attenuation_vectors': attenuation_vectors,        # (A*B, num_sampling_points, output_dim)
                'radiation_vectors': radiation_vectors,            # (A*B, num_sampling_points, output_dim)
                'frequency_basis_vectors': frequency_basis_vectors, # (num_selected_subcarriers, output_dim)
                'sampled_positions': all_sampled_positions,       # (A*B, num_sampling_points, 3)
                'directions': directions                           # (A*B, 3)
            }
            
            # Add intermediate outputs if requested
            if return_intermediates:
                all_spatial_features = torch.stack(all_spatial_features, dim=0)    # (A*B, num_sampling_points, feature_dim)
                
                outputs.update({
                    'spatial_features': all_spatial_features,     # (A*B, num_sampling_points, feature_dim)
                    'antenna_embedding': antenna_embedding.unsqueeze(0)  # (1, antenna_embedding_dim)
                })
            
            logger.debug(f"🔍 PrismNetwork ray tracing forward completed with output keys: {list(outputs.keys())}")
            return outputs
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the network architecture."""
        return {
            'num_subcarriers': self.num_subcarriers,
            'num_bs_antennas': self.num_bs_antennas,
            'feature_dim': self.feature_dim,
            'antenna_embedding_dim': self.antenna_embedding_dim,
            'ray_directions': self.azimuth_divisions * self.elevation_divisions,
            'directional_resolution': (self.azimuth_divisions, self.elevation_divisions),
            'max_ray_length': self.max_ray_length,
            'num_sampling_points': self.num_sampling_points,
            'use_ipe_encoding': self.use_ipe_encoding,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the essential network configuration."""
        return {
            'num_subcarriers': self.num_subcarriers,
            'num_bs_antennas': self.num_bs_antennas,
            'feature_dim': self.feature_dim,
            'antenna_embedding_dim': self.antenna_embedding_dim,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions,
            'max_ray_length': self.max_ray_length,
            'num_sampling_points': self.num_sampling_points,
            'use_ipe_encoding': self.use_ipe_encoding,
            'use_mixed_precision': self.use_mixed_precision
        }


class PrismNetworkConfig:
    """Simplified configuration class for PrismNetwork."""
    
    def __init__(
        self,
        num_subcarriers: int = 408,
        num_bs_antennas: int = 64,
        feature_dim: int = 128,
        antenna_embedding_dim: int = 64,
        azimuth_divisions: int = 36,
        elevation_divisions: int = 10,
        max_ray_length: float = 250.0,
        num_sampling_points: int = 64,
        use_ipe_encoding: bool = True,
        use_mixed_precision: bool = False
    ):
        self.num_subcarriers = num_subcarriers
        self.num_bs_antennas = num_bs_antennas
        self.feature_dim = feature_dim
        self.antenna_embedding_dim = antenna_embedding_dim
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.max_ray_length = max_ray_length
        self.num_sampling_points = num_sampling_points
        self.use_ipe_encoding = use_ipe_encoding
        self.use_mixed_precision = use_mixed_precision
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'num_subcarriers': self.num_subcarriers,
            'num_bs_antennas': self.num_bs_antennas,
            'feature_dim': self.feature_dim,
            'antenna_embedding_dim': self.antenna_embedding_dim,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions,
            'max_ray_length': self.max_ray_length,
            'num_sampling_points': self.num_sampling_points,
            'use_ipe_encoding': self.use_ipe_encoding,
            'use_mixed_precision': self.use_mixed_precision
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PrismNetworkConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
