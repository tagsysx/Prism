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
import contextlib
from typing import Optional, Tuple, Dict, Any, List

# Setup logger
logger = logging.getLogger(__name__)

from .attenuation_network import AttenuationNetwork, AttenuationNetworkConfig
from .frequency_codebook import FrequencyCodebook
from .antenna_codebook import AntennaEmbeddingCodebook, AntennaEmbeddingCodebookConfig
from .radiance_network import RadianceNetwork, RadianceNetworkConfig
from .positional_encoder import PositionalEncoder, create_position_encoder, create_direction_encoder
from .csi_network import CSINetwork


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
        num_subcarriers: int = 408,  # Base subcarriers (will be dynamically updated)
        num_bs_antennas: int = 64,
        num_ue_antennas: int = 4,  # Number of UE antennas
        feature_dim: int = 128,
        antenna_embedding_dim: int = 64,
        azimuth_divisions: int = 36,
        elevation_divisions: int = 10,
        # Ray tracing configuration
        max_ray_length: float = 250.0,
        num_sampling_points: int = 64,
        # Network-specific configurations (simplified)
        attenuation_network_config: dict = None,
        frequency_codebook_config: dict = None,
        radiance_network_config: dict = None,
        # CSI network configuration (always enabled)
        csi_network_config: dict = None,
        **kwargs
    ):
        super().__init__()
        
        # Store essential configuration
        self.num_subcarriers = num_subcarriers
        self.num_bs_antennas = num_bs_antennas
        self.num_ue_antennas = num_ue_antennas
        self.feature_dim = feature_dim
        self.antenna_embedding_dim = antenna_embedding_dim
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        
        # Single antenna CSI processing - no virtual subcarriers needed
        
        # Ray tracing configuration
        self.max_ray_length = max_ray_length
        self.num_sampling_points = num_sampling_points
        
        # Fixed parameters for ray tracing
        self.position_dim = 3  # Always 3D coordinates
        
        # Store simplified network configurations
        self.attenuation_network_config = attenuation_network_config or {}
        self.frequency_codebook_config = frequency_codebook_config or {}
        self.radiance_network_config = radiance_network_config or {}
        
        
        # CSI network configuration (always enabled)
        self.csi_network_config = csi_network_config or {}
        
        # Initialize positional encoders
        # PE encoding for spatial positions (voxels) and view directions
        self.position_encoder = create_position_encoder()  # 10 frequencies, include_input=True
        self.direction_encoder = create_direction_encoder()  # 4 frequencies for directions
        # Create separate encoder for 1D frequency inputs (subcarrier indices)
        self.frequency_encoder = PositionalEncoder(input_dim=1, num_frequencies=10, include_input=True, log_sampling=True)
        
        # Calculate PE encoding dimensions
        self.pe_position_dim = self.position_encoder.get_output_dim()  # 3 + 2*10*3 = 63 (for voxels)
        self.pe_direction_dim = self.direction_encoder.get_output_dim()  # 3 + 2*4*3 = 27 (for view directions)
        self.pe_frequency_dim = self.frequency_encoder.get_output_dim()  # 1 + 2*10*1 = 21
        
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
        
        # 2. FrequencyCodebook: Learnable frequency basis vectors
        self.frequency_codebook = FrequencyCodebook(
            num_subcarriers=self.num_subcarriers,  # Use actual subcarriers for single antenna processing
            basis_dim=self.frequency_codebook_config.get('basis_dim', 32),  # R-dimensional
            initialization=self.frequency_codebook_config.get('initialization', 'complex_normal'),
            std=self.frequency_codebook_config.get('std', 0.1),
            normalize=self.frequency_codebook_config.get('normalize', False)
        )
        
        # 3. AntennaEmbeddingCodebook: BS-UE antenna pair embeddings (always enabled)
        self.antenna_codebook = AntennaEmbeddingCodebook(
            num_bs_antennas=self.num_bs_antennas,
            num_ue_antennas=self.num_ue_antennas,
            embedding_dim=self.antenna_embedding_dim
        )
        
        # 4. RadianceNetwork: Radiation modeling
        self.radiance_network = RadianceNetwork(
            ue_position_dim=self.radiance_network_config.get('ue_position_dim', 16),  # From config, default 16
            view_direction_dim=self.radiance_network_config.get('view_direction_dim', self.pe_direction_dim),
            feature_dim=self.radiance_network_config.get('feature_dim', self.feature_dim),
            antenna_embedding_dim=self.radiance_network_config.get('antenna_embedding_dim', self.antenna_embedding_dim),
            hidden_dim=self.radiance_network_config.get('hidden_dim', default_hidden_dim),
            num_layers=self.radiance_network_config.get('num_layers', 4),
            activation=self.radiance_network_config.get('activation', 'relu'),
            num_subcarriers=self.num_subcarriers,
            output_dim=self.radiance_network_config.get('output_dim', 32),
            complex_output=True
        )
        
        # 5. CSINetwork: Enhance CSI using Transformer (always enabled)
        # Use num_subcarriers from CSI network config (real subcarriers: 408)
        self.csi_network = CSINetwork(
            d_model=self.csi_network_config.get('d_model', 128),
            n_layers=self.csi_network_config.get('n_layers', 2),
            n_heads=self.csi_network_config.get('n_heads', 8),
            d_ff=self.csi_network_config.get('d_ff', 512),
            dropout_rate=self.csi_network_config.get('dropout_rate', 0.1),
            num_subcarriers=self.num_subcarriers,
            smoothing_weight=self.csi_network_config.get('smoothing_weight', 0.0),
            smoothing_type=self.csi_network_config.get('smoothing_type', 'phase_preserve')
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
        Generate A×B uniform direction vectors for ray tracing (vectorized).
        
        Returns:
            directions: (A*B, 3) - Unit direction vectors covering spherical space
                       These are PURE direction vectors (unit length), NOT including BS position offset.
                       To get actual ray positions: P(t) = bs_position + direction * t
        """
        # Calculate angular resolutions
        azimuth_resolution = 2 * torch.pi / self.azimuth_divisions      # 360° / azimuth_divisions
        elevation_resolution = torch.pi / 2 / self.elevation_divisions  # 90° / elevation_divisions
        
        # Create grid of angles using PyTorch (vectorized)
        i, j = torch.meshgrid(
            torch.arange(self.azimuth_divisions, dtype=torch.float32),
            torch.arange(self.elevation_divisions, dtype=torch.float32),
            indexing='xy'
        )
        phi = i * azimuth_resolution     # Azimuth: 0° to 360° (0 to 2π)
        theta = j * elevation_resolution  # Elevation: 0° to 90° (0 to π/2)
        
        # Convert spherical coordinates to Cartesian unit vectors (vectorized)
        # x = cos(elevation) * cos(azimuth)
        # y = cos(elevation) * sin(azimuth)  
        # z = sin(elevation)
        x = torch.cos(theta) * torch.cos(phi)
        y = torch.cos(theta) * torch.sin(phi)
        z = torch.sin(theta)
        
        # Flatten and stack to create (A*B, 3) tensor
        return torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)
    
    def _get_antenna_embedding(self, bs_antenna_index: int, ue_antenna_index: int, device: torch.device) -> torch.Tensor:
        """
        Get single BS-UE antenna pair embedding with proper device handling.
        
        Args:
            bs_antenna_index: BS antenna index (single integer)
            ue_antenna_index: UE antenna index (single integer)
            device: Target device for the embedding
            
        Returns:
            embedding: Tensor of shape (1, antenna_embedding_dim)
        """
        if self.antenna_codebook is not None:
            # Create single-element tensors for the indices
            bs_indices = torch.tensor([bs_antenna_index], dtype=torch.long, device=device)
            ue_indices = torch.tensor([ue_antenna_index], dtype=torch.long, device=device)
            
            return self.antenna_codebook(bs_indices, ue_indices)
        else:
            # Fallback: random embedding
            return torch.randn(1, self.antenna_embedding_dim, device=device)

    def _sample_rays_batch(
        self,
        bs_position: torch.Tensor,
        directions: torch.Tensor,
        max_length: float,
        num_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch sample points along all rays (vectorized).
        
        Args:
            bs_position: Base station position (3,)
            directions: Ray directions (num_rays, 3)
            max_length: Maximum ray length
            num_points: Number of sampling points per ray
            
        Returns:
            sampled_positions: (num_rays, num_points, 3)
            flat_positions: (num_rays * num_points, 3)
        """
        t_values = torch.linspace(
            0, max_length, num_points, 
            dtype=torch.float32, device=bs_position.device
        )
        
        # Ray equation: P(t) = bs_position + direction * t
        # Using broadcasting: [1,1,3] + [R,1,3] * [1,P,1] → [R,P,3]
        sampled_positions = bs_position.view(1, 1, 3) + directions.unsqueeze(1) * t_values.view(1, -1, 1)
        return sampled_positions, sampled_positions.reshape(-1, 3)


    def _encode_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode positions with proper complex handling.
        
        Args:
            positions: (N, 3) tensor of positions
            
        Returns:
            encoded: (N, encoded_dim) tensor
        """
        # For spatial positions (voxels), use PE encoding
        encoded = self.position_encoder(positions)
        return torch.view_as_real(encoded).flatten(-2) if torch.is_complex(encoded) else encoded

    def _prepare_radiance_inputs(
        self,
        ue_pos: torch.Tensor,
        view_dirs: torch.Tensor,
        features: torch.Tensor,
        antenna_embeddings: torch.Tensor,
        num_rays: int,
        num_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for RadianceNetwork with complex feature handling.
        
        Args:
            ue_pos: Raw UE position (3,) - will be encoded in RadianceNetwork
            view_dirs: Raw view directions (num_rays, 3) - will be encoded in RadianceNetwork
            features: Voxel features (num_rays*num_points, feature_dim)
            antenna_embeddings: BS-UE antenna pair embeddings (1, num_bs_antennas, num_ue_antennas, embedding_dim)
            num_rays: Number of rays
            num_points: Number of points per ray
            
        Returns:
            Tuple of inputs for RadianceNetwork
        """
        # Features from AttenuationNetwork are always real-valued
        real_features = features
        
        # Expand tensors using memory-efficient operations
        ue_pos_expanded = ue_pos.expand(num_rays * num_points, -1)
        view_dirs_expanded = view_dirs.repeat_interleave(num_points, dim=0)
        
        # Expand antenna embeddings to match the number of sampling points
        # antenna_embeddings: (1, antenna_embedding_dim)
        # Need to expand to: (num_rays * num_points, antenna_embedding_dim)
        antenna_embeddings_expanded = antenna_embeddings.expand(num_rays * num_points, -1)
        
        # Return tensors in the expected format for RadianceNetwork
        return (
            ue_pos_expanded,
            view_dirs_expanded,
            real_features,
            antenna_embeddings_expanded
        )
        
    def forward(
        self,
        bs_position: torch.Tensor,
        ue_position: torch.Tensor,
        bs_antenna_index: int,
        ue_antenna_index: int
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete Prism network with chunked ray tracing approach.
        
        Generates rays from BS antenna in A×B uniform directions, processes them in chunks
        to reduce memory usage, and outputs attenuation and radiation vectors.
        
        Args:
            bs_position: Base station position (3,) - Single BS position coordinates
            ue_position: User equipment position (3,) - Single UE position coordinates  
            bs_antenna_index: BS antenna index (int) - Single BS antenna to process
            ue_antenna_index: UE antenna index (int) - Single UE antenna to process
            
        Returns:
            Dictionary containing outputs with specific dimensions:
            
            Core outputs:
            - 'attenuation_vectors': (A*B, num_sampling_points, output_dim)
                Complex attenuation coefficients for each voxel along each ray direction
            - 'radiation_vectors': (A*B, num_sampling_points, output_dim)  
                Complex radiation factors for each voxel along each ray direction
            - 'frequency_basis_vectors': (num_subcarriers, output_dim)
                Complex frequency basis vectors for all subcarriers from FrequencyNetwork
            - 'sampled_positions': (A*B, num_sampling_points, 3)
                Complete 3D coordinates for all sampling points along all ray directions
            - 'directions': (A*B, 3)
                Direction vectors with BS position offset added (unit_directions + bs_position)
            
        Raises:
            ValueError: If any input tensor doesn't have the required dimensions
        """
        # Validate input tensor dimensions
        if len(bs_position.shape) != 1 or bs_position.shape[0] != 3:
            raise ValueError(f"bs_position must have shape (3,), but got shape {bs_position.shape}.")
        
        if len(ue_position.shape) != 1 or ue_position.shape[0] != 3:
            raise ValueError(f"ue_position must have shape (3,), but got shape {ue_position.shape}.")
        
        if not isinstance(bs_antenna_index, int):
            raise ValueError(f"bs_antenna_index must be an integer, but got {type(bs_antenna_index)}.")
        
        if not isinstance(ue_antenna_index, int):
            raise ValueError(f"ue_antenna_index must be an integer, but got {type(ue_antenna_index)}.")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🔍 Processing BS antenna {bs_antenna_index} and UE antenna {ue_antenna_index} from BS at {bs_position} to UE at {ue_position}")
        
        # Get device and parameters
        device = bs_position.device
        num_rays = self.azimuth_divisions * self.elevation_divisions
        num_points = self.num_sampling_points
        
        # Forward pass
        # 1. Generate uniform ray directions
        directions = self._generate_uniform_directions().to(device)
        
        # 2. Get BS-UE antenna pair embedding (once for all chunks)
        # Get single BS-UE antenna pair embedding
        antenna_embedding = self._get_antenna_embedding(bs_antenna_index, ue_antenna_index, device)
        
        # 3. UE position will be encoded in RadianceNetwork using linear projector
        
        # 4. Process all rays at once
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🚀 Processing all {num_rays} rays")
        
        # Sample points along all rays
        sampled_positions, flat_positions = self._sample_rays_batch(
            bs_position, directions, self.max_ray_length, num_points
        )
        
        # Encode all positions
        encoded_positions = self._encode_positions(flat_positions)
        
        # AttenuationNetwork: process all points
        attenuation_vectors_flat, spatial_features_flat = self.attenuation_network(encoded_positions)
        
        # Reshape outputs
        attenuation_vectors = attenuation_vectors_flat.view(num_rays, num_points, -1)
        spatial_features = spatial_features_flat.view(num_rays, num_points, -1)
        
        # View directions (negative for incoming rays)
        view_dirs = -directions
        
        # RadianceNetwork: process all points
        spatial_features_flat = spatial_features.view(-1, spatial_features.shape[-1])
        radiance_inputs = self._prepare_radiance_inputs(
            ue_position, view_dirs, spatial_features_flat, antenna_embedding, num_rays, num_points
        )
        
        radiation_vectors_flat = self.radiance_network(*radiance_inputs)
        radiation_vectors = radiation_vectors_flat.view(num_rays, num_points, -1)
        
        logger.debug(f"✅ Completed ray processing: {attenuation_vectors.shape}")
            
        # 5. FrequencyCodebook: Retrieve frequency basis vectors for all subcarriers
        frequency_basis_vectors = self.frequency_codebook()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🔍 Retrieved frequency basis vectors for all {self.num_subcarriers} subcarriers")
            
        # 6. Prepare outputs
        outputs = {
            'attenuation_vectors': attenuation_vectors,        # (A*B, num_sampling_points, output_dim)
            'radiation_vectors': radiation_vectors,            # (A*B, num_sampling_points, output_dim)
            'frequency_basis_vectors': frequency_basis_vectors, # (num_subcarriers, output_dim)
            'sampled_positions': sampled_positions,           # (A*B, num_sampling_points, 3)
            'directions': directions + bs_position.unsqueeze(0)  # (A*B, 3) - Directions with BS position offset
        }
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🔍 PrismNetwork forward completed with output keys: {list(outputs.keys())}")
        return outputs
    
    def enhance_csi(self, csi: torch.Tensor, bs_antenna_index: int, ue_antenna_index: int) -> torch.Tensor:
        """
        Enhance CSI using CSINetwork for single sample.
        
        Args:
            csi: Input CSI tensor [num_subcarriers] - single CSI sample
            bs_antenna_index: BS antenna index (single integer)
            ue_antenna_index: UE antenna index (single integer)
            
        Returns:
            enhanced_csi: Enhanced CSI tensor [num_subcarriers] - single enhanced CSI sample
        """
        logger.debug(f"🔧 Applying CSI enhancement: {csi.shape}")
        
        # CSINetwork now processes single sample with antenna indices
        max_magnitude = self.csi_network_config.get('max_magnitude', 100.0)
        enhanced_csi = self.csi_network(csi, bs_antenna_index, ue_antenna_index, max_magnitude)
        
        logger.debug(f"✅ CSI enhancement completed: {enhanced_csi.shape}")
        return enhanced_csi
    
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
            'use_csi_network': True,  # Always enabled
            'individual_antenna_processing': True,  # New architecture feature
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the essential network configuration."""
        return {
            'num_subcarriers': self.num_subcarriers,
            'num_bs_antennas': self.num_bs_antennas,
            'num_ue_antennas': self.num_ue_antennas,
            'feature_dim': self.feature_dim,
            'antenna_embedding_dim': self.antenna_embedding_dim,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions,
            'max_ray_length': self.max_ray_length,
            'num_sampling_points': self.num_sampling_points,
            'use_csi_network': True  # Always enabled
        }


class PrismNetworkConfig:
    """Simplified configuration class for PrismNetwork."""
    
    def __init__(
        self,
        num_subcarriers: int = 408,
        num_bs_antennas: int = 64,
        num_ue_antennas: int = 4,
        feature_dim: int = 128,
        antenna_embedding_dim: int = 64,
        azimuth_divisions: int = 36,
        elevation_divisions: int = 10,
        max_ray_length: float = 250.0,
        num_sampling_points: int = 64
    ):
        self.num_subcarriers = num_subcarriers
        self.num_bs_antennas = num_bs_antennas
        self.num_ue_antennas = num_ue_antennas
        self.feature_dim = feature_dim
        self.antenna_embedding_dim = antenna_embedding_dim
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.max_ray_length = max_ray_length
        self.num_sampling_points = num_sampling_points
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'num_subcarriers': self.num_subcarriers,
            'num_bs_antennas': self.num_bs_antennas,
            'num_ue_antennas': self.num_ue_antennas,
            'feature_dim': self.feature_dim,
            'antenna_embedding_dim': self.antenna_embedding_dim,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions,
            'max_ray_length': self.max_ray_length,
            'num_sampling_points': self.num_sampling_points,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PrismNetworkConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
