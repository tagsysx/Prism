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
        self.feature_dim = feature_dim
        self.antenna_embedding_dim = antenna_embedding_dim
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.use_ipe_encoding = use_ipe_encoding
        self.use_mixed_precision = use_mixed_precision
        
        # Virtual subcarriers for multi-UE antenna scenarios
        # This allows flexible usage of subcarriers based on UE antenna configuration
        self.num_virtual_subcarriers = self.num_subcarriers
        
        # Ray tracing configuration
        self.max_ray_length = max_ray_length
        self.num_sampling_points = num_sampling_points
        
        # Fixed parameters for ray tracing
        self.position_dim = 3  # Always 3D coordinates
        self.use_antenna_codebook = True  # Always use antenna codebook
        self.complex_output = True  # Always use complex output for RF signals
        
        # Store simplified network configurations
        self.attenuation_network_config = attenuation_network_config or {}
        self.frequency_codebook_config = frequency_codebook_config or {}
        self.radiance_network_config = radiance_network_config or {}
        
        
        # CSI network configuration (always enabled)
        self.csi_network_config = csi_network_config or {}
        
        # Initialize positional encoders
        if use_ipe_encoding:
            # Use traditional PE encoding instead of IPE for now
            self.position_encoder = create_position_encoder()  # 10 frequencies, include_input=True
            self.direction_encoder = create_position_encoder()  # Use same as position (10 frequencies) for consistency
            # Create separate encoder for 1D frequency inputs (subcarrier indices)
            self.frequency_encoder = PositionalEncoder(input_dim=1, num_frequencies=10, include_input=True, log_sampling=True)
            
            # Calculate PE encoding dimensions
            self.pe_position_dim = self.position_encoder.get_output_dim()  # 3 + 2*10*3 = 63
            self.pe_direction_dim = self.direction_encoder.get_output_dim()  # 3 + 2*10*3 = 63
            self.pe_frequency_dim = self.frequency_encoder.get_output_dim()  # 1 + 2*10*1 = 21
        else:
            self.position_encoder = None
            self.direction_encoder = None
            self.frequency_encoder = None
            self.pe_position_dim = self.position_dim
            self.pe_direction_dim = self.position_dim
            self.pe_frequency_dim = 1
        
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
            num_subcarriers=self.num_virtual_subcarriers,  # Use virtual subcarriers (1632)
            basis_dim=self.frequency_codebook_config.get('basis_dim', 32),  # R-dimensional
            initialization=self.frequency_codebook_config.get('initialization', 'complex_normal'),
            std=self.frequency_codebook_config.get('std', 0.1),
            normalize=self.frequency_codebook_config.get('normalize', False)
        )
        
        # 3. AntennaEmbeddingCodebook: Antenna-specific embeddings (always enabled)
        self.antenna_codebook = AntennaEmbeddingCodebook(
            num_bs_antennas=self.num_bs_antennas,
            embedding_dim=self.antenna_embedding_dim
        )
        
        # 4. RadianceNetwork: Radiation modeling
        self.radiance_network = RadianceNetwork(
            ue_position_dim=self.radiance_network_config.get('ue_position_dim', self.pe_position_dim),
            view_direction_dim=self.radiance_network_config.get('view_direction_dim', self.pe_direction_dim),
            feature_dim=self.radiance_network_config.get('feature_dim', self.feature_dim),
            antenna_embedding_dim=self.radiance_network_config.get('antenna_embedding_dim', self.antenna_embedding_dim),
            hidden_dim=self.radiance_network_config.get('hidden_dim', default_hidden_dim),
            num_layers=self.radiance_network_config.get('num_layers', 4),
            activation=self.radiance_network_config.get('activation', 'relu'),
            num_subcarriers=self.radiance_network_config.get('output_dim', self.num_subcarriers),
            complex_output=self.complex_output
        )
        
        # 5. LowRankTransformer has been removed
        
        # 6. CSINetwork: Enhance CSI using Transformer (always enabled)
        # Use num_subcarriers from CSI network config (real subcarriers: 408)
        self.csi_network = CSINetwork(
            d_model=self.csi_network_config.get('d_model', 128),
            n_layers=self.csi_network_config.get('n_layers', 2),
            n_heads=self.csi_network_config.get('n_heads', 8),
            d_ff=self.csi_network_config.get('d_ff', 512),
            dropout_rate=self.csi_network_config.get('dropout_rate', 0.1),
            num_subcarriers=self.csi_network_config.get('num_subcarriers', 64),
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
    
    def set_virtual_subcarriers(self, num_ue_antennas: int):
        """
        Set virtual subcarriers based on UE antenna configuration.
        
        Args:
            num_ue_antennas: Number of UE antennas
        """
        self.num_virtual_subcarriers = self.num_subcarriers * num_ue_antennas
        logger.info(f"🔧 Set virtual subcarriers: {self.num_subcarriers} * {num_ue_antennas} = {self.num_virtual_subcarriers}")
        
        # Update FrequencyCodebook to use virtual subcarriers
        if hasattr(self, 'frequency_codebook'):
            # Create new FrequencyCodebook with virtual subcarriers
            from .frequency_codebook import FrequencyCodebook
            self.frequency_codebook = FrequencyCodebook(
                num_subcarriers=self.num_virtual_subcarriers,
                basis_dim=self.frequency_codebook_config.get('basis_dim', 32),
                initialization=self.frequency_codebook_config.get('initialization', 'complex_normal'),
                std=self.frequency_codebook_config.get('std', 0.1),
                normalize=self.frequency_codebook_config.get('normalize', False)
            ).to(next(self.parameters()).device)
            logger.info(f"🔧 Updated FrequencyCodebook to use {self.num_virtual_subcarriers} virtual subcarriers")
    
    def _generate_uniform_directions(self) -> torch.Tensor:
        """
        Generate A×B uniform direction vectors for ray tracing (vectorized).
        
        Returns:
            directions: (A*B, 3) - Unit direction vectors covering spherical space
                       These are PURE direction vectors (unit length), NOT including BS position offset.
                       To get actual ray positions: P(t) = bs_position + direction * t
        """
        # Calculate angular resolutions
        azimuth_resolution = 2 * torch.pi / self.azimuth_divisions      # 360° / 36 = 10°
        elevation_resolution = torch.pi / 2 / self.elevation_divisions  # 90° / 10 = 9°
        
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
    
    def _get_antenna_embedding(self, index: int, device: torch.device) -> torch.Tensor:
        """
        Get antenna embedding with proper device handling.
        
        Args:
            index: Antenna index
            device: Target device for the embedding
            
        Returns:
            embedding: (antenna_embedding_dim,) tensor
        """
        if self.use_antenna_codebook and self.antenna_codebook is not None:
            antenna_tensor = torch.tensor([index], dtype=torch.long).to(device)
            return self.antenna_codebook(antenna_tensor).squeeze(0)
        return torch.randn(self.antenna_embedding_dim, device=device)

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

    def _precompute_encodings(
        self,
        ue_position: torch.Tensor,
        directions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute UE position and direction encodings.
        
        Args:
            ue_position: UE position (3,)
            directions: View directions (num_rays, 3)
            
        Returns:
            encoded_ue_position: (encoded_dim,)
            encoded_directions: (num_rays, encoded_dim)
        """
        if self.use_ipe_encoding and self.position_encoder is not None and self.direction_encoder is not None:
            encoded_ue = self.position_encoder(ue_position)
            encoded_dirs = self.direction_encoder(directions)
            
            # Handle complex encodings
            if torch.is_complex(encoded_ue):
                encoded_ue = torch.view_as_real(encoded_ue).flatten()
            if torch.is_complex(encoded_dirs):
                encoded_dirs = torch.view_as_real(encoded_dirs).flatten(-2)
                
            return encoded_ue, encoded_dirs
        return ue_position, directions

    def _encode_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode positions with proper complex handling.
        
        Args:
            positions: (N, 3) tensor of positions
            
        Returns:
            encoded: (N, encoded_dim) tensor
        """
        if self.use_ipe_encoding and self.position_encoder is not None:
            encoded = self.position_encoder(positions)
            return torch.view_as_real(encoded).flatten(-2) if torch.is_complex(encoded) else encoded
        return positions

    def _prepare_radiance_inputs(
        self,
        ue_pos: torch.Tensor,
        view_dirs: torch.Tensor,
        features: torch.Tensor,
        antenna_emb: torch.Tensor,
        num_rays: int,
        num_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for RadianceNetwork with complex feature handling.
        
        Args:
            ue_pos: UE position encoding (encoded_dim,)
            view_dirs: View directions encoding (num_rays, encoded_dim)
            features: Voxel features (num_rays*num_points, feature_dim)
            antenna_emb: Antenna embedding (embedding_dim,)
            num_rays: Number of rays
            num_points: Number of points per ray
            
        Returns:
            Tuple of inputs for RadianceNetwork
        """
        # Handle complex features: preserve both magnitude and phase
        if torch.is_complex(features):
            magnitude = torch.abs(features)
            phase = torch.angle(features)
            real_features = torch.cat([magnitude, phase], dim=-1)
        else:
            real_features = features
        
        # Expand tensors using memory-efficient operations
        ue_pos_expanded = ue_pos.expand(num_rays * num_points, -1)
        view_dirs_expanded = view_dirs.repeat_interleave(num_points, dim=0)
        antenna_emb_expanded = antenna_emb.expand(num_rays * num_points, -1)
        
        # Return tensors in the expected format for RadianceNetwork
        return (
            ue_pos_expanded,
            view_dirs_expanded,
            real_features,
            antenna_emb_expanded.unsqueeze(1)  # Only antenna_embeddings needs extra dimension
        )
        
    def forward(
        self,
        bs_position: torch.Tensor,
        ue_position: torch.Tensor,
        antenna_index: int,
        selected_subcarriers: Optional[List[int]] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete Prism network with chunked ray tracing approach.
        
        Generates rays from BS antenna in A×B uniform directions, processes them in chunks
        to reduce memory usage, and outputs attenuation and radiation vectors.
        
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
                Complete 3D coordinates for all sampling points along all ray directions
            - 'directions': (A*B, 3)
                Direction vectors with BS position offset added (unit_directions + bs_position)
                
            Intermediate outputs (if return_intermediates=True):
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
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🔍 Processing antenna {antenna_index} from BS at {bs_position} to UE at {ue_position}")
        
        # Get device and parameters
        device = bs_position.device
        num_rays = self.azimuth_divisions * self.elevation_divisions
        num_points = self.num_sampling_points
        
        # Calculate optimal chunk size based on available memory
        if torch.cuda.is_available() and device.type == 'cuda':
            try:
                free_memory = torch.cuda.mem_get_info(device)[0]
                # More aggressive estimate: ~20MB per ray (reduced from 50MB)
                memory_per_ray = 20 * 1024 * 1024
                chunk_size = max(50, min(200, int(free_memory * 0.5 / memory_per_ray)))
            except Exception:
                chunk_size = 100  # More aggressive fallback
        else:
            chunk_size = 100  # CPU or unknown device
        
        logger.debug(f"🔁 Processing {num_rays} rays in chunks of {chunk_size} (estimated memory optimization: ~75%)")
        
        # Enable mixed precision for forward pass if configured
        use_autocast = torch.cuda.is_available() and self.use_mixed_precision
        
        with torch.amp.autocast('cuda', enabled=use_autocast):
            # 1. Generate uniform ray directions
            directions = self._generate_uniform_directions().to(device)
            
            # 2. Get antenna embedding (once for all chunks)
            antenna_embedding = self._get_antenna_embedding(antenna_index, device)
            
            # 3. Precompute UE position encoding (once for all chunks)
            encoded_ue_position, _ = self._precompute_encodings(ue_position, torch.zeros(1, 3, device=device))
            
            # 4. Process rays in chunks to reduce memory usage
            attenuation_list = []
            radiation_list = []
            sampled_list = []
            
            for chunk_start in range(0, num_rays, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_rays)
                chunk_dirs = directions[chunk_start:chunk_end]
                chunk_num_rays = chunk_end - chunk_start
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"📦 Processing chunk {chunk_start//chunk_size + 1}/{(num_rays + chunk_size - 1)//chunk_size}: rays {chunk_start}-{chunk_end-1}")
                
                # Sample points along current chunk of rays
                chunk_sampled, chunk_flat = self._sample_rays_batch(
                    bs_position, chunk_dirs, self.max_ray_length, num_points
                )
                
                # Encode positions for current chunk
                chunk_encoded_positions = self._encode_positions(chunk_flat)
                
                # AttenuationNetwork: process chunk
                with torch.amp.autocast('cuda', enabled=False) if use_autocast else contextlib.nullcontext():
                    chunk_attenuation, chunk_spatial = self.attenuation_network(chunk_encoded_positions)
                
                # Reshape chunk outputs
                chunk_attenuation = chunk_attenuation.view(chunk_num_rays, num_points, -1)
                chunk_spatial = chunk_spatial.view(chunk_num_rays, num_points, -1)
                
                # Encode directions for current chunk
                _, chunk_encoded_dirs = self._precompute_encodings(ue_position, -chunk_dirs)
                
                # RadianceNetwork: process chunk
                chunk_spatial_flat = chunk_spatial.view(-1, chunk_spatial.shape[-1])
                chunk_radiance_inputs = self._prepare_radiance_inputs(
                    encoded_ue_position,
                    chunk_encoded_dirs,
                    chunk_spatial_flat,
                    antenna_embedding,
                    chunk_num_rays,
                    num_points
                )
                
                chunk_radiation = self.radiance_network(*chunk_radiance_inputs).squeeze(1)
                chunk_radiation = chunk_radiation.view(chunk_num_rays, num_points, -1)
                
                # Collect chunk results
                attenuation_list.append(chunk_attenuation)
                radiation_list.append(chunk_radiation)
                sampled_list.append(chunk_sampled)
                
                # Free intermediate tensors and cache
                del chunk_flat, chunk_encoded_positions, chunk_spatial_flat, chunk_radiance_inputs
                if chunk_start > 0:  # Keep cache for first chunk
                    torch.cuda.empty_cache()
            
            # 5. Concatenate all chunks
            attenuation_vectors = torch.cat(attenuation_list, dim=0)
            radiation_vectors = torch.cat(radiation_list, dim=0)
            sampled_positions = torch.cat(sampled_list, dim=0)
            
            logger.debug(f"✅ Completed chunked ray processing: {len(attenuation_list)} chunks, total shape: {attenuation_vectors.shape}")
            
            # 6. LowRankTransformer has been removed
            
        # 7. FrequencyCodebook: Retrieve frequency basis vectors
        if selected_subcarriers is not None:
            # Convert to tensor for indexing
            subcarrier_indices = torch.tensor(selected_subcarriers, dtype=torch.long, device=device)
            frequency_basis_vectors = self.frequency_codebook(subcarrier_indices)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🔍 Retrieved frequency basis vectors for {len(selected_subcarriers)} selected subcarriers")
        else:
            # Get all frequency basis vectors - use virtual subcarriers for multi-UE antenna scenarios
            frequency_basis_vectors = self.frequency_codebook()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🔍 Retrieved frequency basis vectors for all {self.num_virtual_subcarriers} virtual subcarriers")
            
            # 8. Prepare outputs
            outputs = {
                'attenuation_vectors': attenuation_vectors,        # (A*B, num_sampling_points, output_dim)
                'radiation_vectors': radiation_vectors,            # (A*B, num_sampling_points, output_dim)
                'frequency_basis_vectors': frequency_basis_vectors, # (num_selected_subcarriers, output_dim)
                'sampled_positions': sampled_positions,           # (A*B, num_sampling_points, 3)
                'directions': directions + bs_position.unsqueeze(0)  # (A*B, 3) - Directions with BS position offset
            }
            
            # Add intermediate outputs if requested
            if return_intermediates:
                outputs.update({
                    'antenna_embedding': antenna_embedding.unsqueeze(0)  # (1, antenna_embedding_dim)
                })
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🔍 PrismNetwork chunked forward completed with output keys: {list(outputs.keys())}")
            return outputs
    
    def enhance_csi(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Enhance CSI using CSINetwork (always enabled).
        
        Args:
            csi: Input CSI tensor [batch_size, bs_antennas, ue_antennas, num_subcarriers] (4D)
            
        Returns:
            enhanced_csi: Enhanced CSI tensor [batch_size, bs_antennas, ue_antennas, num_subcarriers] (4D)
        """
        logger.debug(f"🔧 Applying CSI enhancement: {csi.shape}")
        max_magnitude = self.csi_network_config.get('max_magnitude', 100.0)
        enhanced_csi = self.csi_network(csi, max_magnitude)
        logger.debug(f"✅ CSI enhancement completed: {enhanced_csi.shape}")
        return enhanced_csi
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the network architecture."""
        return {
            'num_subcarriers': self.num_subcarriers,
            'num_virtual_subcarriers': self.num_virtual_subcarriers,
            'num_bs_antennas': self.num_bs_antennas,
            'feature_dim': self.feature_dim,
            'antenna_embedding_dim': self.antenna_embedding_dim,
            'ray_directions': self.azimuth_divisions * self.elevation_divisions,
            'directional_resolution': (self.azimuth_divisions, self.elevation_divisions),
            'max_ray_length': self.max_ray_length,
            'num_sampling_points': self.num_sampling_points,
            'use_ipe_encoding': self.use_ipe_encoding,
            'use_csi_network': True,  # Always enabled
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the essential network configuration."""
        return {
            'num_subcarriers': self.num_subcarriers,
            'num_virtual_subcarriers': self.num_virtual_subcarriers,
            'num_bs_antennas': self.num_bs_antennas,
            'feature_dim': self.feature_dim,
            'antenna_embedding_dim': self.antenna_embedding_dim,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions,
            'max_ray_length': self.max_ray_length,
            'num_sampling_points': self.num_sampling_points,
            'use_ipe_encoding': self.use_ipe_encoding,
            'use_mixed_precision': self.use_mixed_precision,
            'use_csi_network': True  # Always enabled
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
