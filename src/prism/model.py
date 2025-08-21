"""
Prism Model: Wideband RF Neural Radiance Fields for OFDM Communication.

This module implements the redesigned Prism model architecture that addresses
computational efficiency issues while maintaining the virtual link concept
for OFDM communication systems.

Key Components:
- AttenuationNetwork: Single network encoding spatial information into 128D features
- AttenuationDecoder: Multi-channel MLPs decoding features into attenuation factors
- RadianceNetwork: Processing UE position, viewing direction, and spatial features
- PrismModel: Main model combining all components
- PrismLoss: Frequency-aware loss function for multi-subcarrier optimization

The new architecture eliminates independent subcarrier channels and uses shared
feature encoding for significantly improved efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

# Import new modules
from .csi_processor import CSIVirtualLinkProcessor
from .ray_tracer import AdvancedRayTracer, Environment, Building, Plane

class AttenuationNetwork(nn.Module):
    """
    Attenuation Network: Encode spatial position information into compact features.
    
    This is the core innovation of the new architecture. Instead of having
    M×N_UE independent networks, we use a single network to encode spatial
    information into a compact 128-dimensional feature representation.
    
    Architecture:
    - 8-layer MLP with ReLU activation
    - Input: 3D position coordinates
    - Output: 128-dimensional feature vector (configurable)
    
    Key Benefits:
    - Single network instead of M×N_UE independent networks
    - Compact 128D representation instead of M×N_UE×128D
    - Maintains spatial encoding capabilities
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 256, feature_dim: int = 128):
        """
        Initialize the Attenuation Network.
        
        Args:
            input_dim: Dimension of input features (typically 3 for 3D position)
            hidden_dim: Hidden dimension for all layers
            feature_dim: Output feature dimension (default: 128)
        """
        super(AttenuationNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # Build the MLP architecture: 8 layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Add intermediate hidden layers
        for _ in range(7):  # 8 total layers: 1 input + 6 hidden + 1 output
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        # Output layer to feature dimension
        layers.append(nn.Linear(hidden_dim, feature_dim))
        
        self.network = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Attenuation Network.
        
        Args:
            x: Input tensor containing spatial position [batch_size, input_dim]
            
        Returns:
            128-dimensional feature vector [batch_size, feature_dim]
        """
        # Process through all hidden layers with ReLU activation
        for i, layer in enumerate(self.network[:-1]):
            x = self.activation(layer(x))
        
        # Final layer without activation (linear output)
        x = self.network[-1](x)
        return x

class AttenuationDecoder(nn.Module):
    """
    Attenuation Decoder: Convert 128D features into M×N_UE attenuation factors.
    
    This module uses N_UE independent 3-layer MLPs to decode the compact
    spatial features into attenuation factors for each subcarrier and UE antenna.
    
    Architecture:
    - N_UE independent channels
    - Each channel: 128D → 256D → 256D → M
    - Output: Complex values representing attenuation factors
    
    Key Benefits:
    - Configurable N_UE channels
    - Efficient processing of M subcarriers per channel
    - Maintains per-UE-antenna processing
    """
    def __init__(self, feature_dim: int = 128, num_subcarriers: int = 408, 
                 num_ue_antennas: int = 4, hidden_dim: int = 256):
        """
        Initialize the Attenuation Decoder.
        
        Args:
            feature_dim: Input feature dimension from AttenuationNetwork
            num_subcarriers: Number of subcarriers (M)
            num_ue_antennas: Number of UE antennas (N_UE)
            hidden_dim: Hidden dimension for MLP layers
        """
        super(AttenuationDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.num_subcarriers = num_subcarriers
        self.num_ue_antennas = num_ue_antennas
        self.hidden_dim = hidden_dim
        
        # Create N_UE independent MLP channels
        self.channels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_subcarriers * 2)  # *2 for complex values (real + imaginary)
            ) for _ in range(num_ue_antennas)
        ])
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Attenuation Decoder.
        
        Args:
            features: 128-dimensional feature vector [batch_size, feature_dim]
            
        Returns:
            Attenuation factors [batch_size, num_ue_antennas, num_subcarriers]
                Complex values representing attenuation for each UE antenna and subcarrier
        """
        batch_size = features.shape[0]
        outputs = []
        
        # Process each UE antenna channel independently
        for i, channel in enumerate(self.channels):
            # Process through this channel's MLP
            channel_output = channel(features)  # [batch_size, num_subcarriers * 2]
            
            # Reshape to separate real and imaginary parts
            channel_output = channel_output.view(batch_size, self.num_subcarriers, 2)
            
            # Convert to complex numbers
            real_part = channel_output[:, :, 0]
            imag_part = channel_output[:, :, 1]
            complex_output = torch.complex(real_part, imag_part)
            
            outputs.append(complex_output)
        
        # Stack all UE antenna outputs
        # Result: [batch_size, num_ue_antennas, num_subcarriers]
        return torch.stack(outputs, dim=1)

class RadianceNetwork(nn.Module):
    """
    Radiance Network: Process UE position, viewing direction, spatial features, and BS antenna ID.
    
    This network processes UE position, viewing direction, the 128D spatial features,
    and BS antenna ID to output radiation characteristics for specific antenna pairs.
    
    Architecture:
    - N_UE × N_BS independent channels (one for each UE-BS antenna pair)
    - Each channel processes: [UE_pos, view_dir, 128D_features, BS_antenna_ID] → M radiation values
    - Output: Complex values representing radiation characteristics for specific antenna pairs
    
    Key Benefits:
    - Independent processing for each UE-BS antenna pair
    - Incorporates spatial features from AttenuationNetwork
    - BS antenna-specific radiation modeling
    - Configurable output dimensions
    """
    def __init__(self, position_dim: int = 3, view_dim: int = 3, feature_dim: int = 128,
                 num_subcarriers: int = 408, num_ue_antennas: int = 4, num_bs_antennas: int = 64, 
                 hidden_dim: int = 256):
        """
        Initialize the Radiance Network.
        
        Args:
            position_dim: Dimension of UE position coordinates
            view_dim: Dimension of viewing direction vector
            feature_dim: Dimension of spatial features from AttenuationNetwork
            num_subcarriers: Number of subcarriers (M)
            num_ue_antennas: Number of UE antennas (N_UE)
            num_bs_antennas: Number of BS antennas (N_BS)
            hidden_dim: Hidden dimension for MLP layers
        """
        super(RadianceNetwork, self).__init__()
        self.position_dim = position_dim
        self.view_dim = view_dim
        self.feature_dim = feature_dim
        self.num_subcarriers = num_subcarriers
        self.num_ue_antennas = num_ue_antennas
        self.num_bs_antennas = num_bs_antennas
        self.hidden_dim = hidden_dim
        
        # Input dimension: UE position + viewing direction + spatial features + BS antenna ID
        input_dim = position_dim + view_dim + feature_dim + 1  # +1 for BS antenna ID
        
        # Create N_UE × N_BS independent channels (one for each antenna pair)
        total_channels = num_ue_antennas * num_bs_antennas
        self.channels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_subcarriers * 2)  # *2 for complex values
            ) for _ in range(total_channels)
        ])
        
    def forward(self, ue_positions: torch.Tensor, view_directions: torch.Tensor, 
                spatial_features: torch.Tensor, bs_antenna_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Radiance Network.
        
        Args:
            ue_positions: UE position coordinates [batch_size, position_dim]
            view_directions: Viewing direction vectors [batch_size, view_dim]
            spatial_features: 128D spatial features [batch_size, feature_dim]
            bs_antenna_ids: BS antenna IDs [batch_size] (1 to N_BS)
            
        Returns:
            Radiation factors [batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers]
                Complex values representing radiation for each UE-BS antenna pair and subcarrier
        """
        batch_size = ue_positions.shape[0]
        
        outputs = []
        
        # Process each UE-BS antenna pair channel
        for ue_idx in range(self.num_ue_antennas):
            ue_outputs = []
            for bs_idx in range(self.num_bs_antennas):
                # Calculate channel index for this UE-BS pair
                channel_idx = ue_idx * self.num_bs_antennas + bs_idx
                channel = self.channels[channel_idx]
                
                # Concatenate inputs: UE_pos + view_dir + spatial_features + BS_antenna_ID
                # Normalize BS antenna ID to [0, 1] range for better training
                normalized_bs_id = (bs_antenna_ids.float() - 1) / (self.num_bs_antennas - 1)
                combined_input = torch.cat([
                    ue_positions, view_directions, spatial_features, 
                    normalized_bs_id.unsqueeze(1)
                ], dim=1)
                
                # Process through this channel's MLP
                channel_output = channel(combined_input)  # [batch_size, num_subcarriers * 2]
                
                # Reshape to separate real and imaginary parts
                channel_output = channel_output.view(batch_size, self.num_subcarriers, 2)
                
                # Convert to complex numbers
                real_part = channel_output[:, :, 0]
                imag_part = channel_output[:, :, 1]
                complex_output = torch.complex(real_part, imag_part)
                
                ue_outputs.append(complex_output)
            
            # Stack BS antenna outputs for this UE
            ue_outputs = torch.stack(ue_outputs, dim=1)  # [batch_size, num_bs_antennas, num_subcarriers]
            outputs.append(ue_outputs)
        
        # Stack all UE antenna outputs
        # Result: [batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers]
        return torch.stack(outputs, dim=1)

class PrismModel(nn.Module):
    """
    Prism: Wideband RF Neural Radiance Fields for OFDM Communication.
    
    This is the main model implementing the new architecture that addresses
    computational efficiency issues while maintaining the virtual link concept.
    
    Key Features:
    - M subcarriers (configurable from 52 to 1024+)
    - N_UE antennas at the User Equipment
    - N_BS antennas at the Base Station
    - Single AttenuationNetwork for spatial encoding
    - Efficient AttenuationDecoder for M×N_UE attenuation factors
    - RadianceNetwork for radiation characteristics
    - 3D spatial awareness for position-dependent modeling
    - MIMO channel matrix generation
    
    Architecture Flow:
    1. Input: 3D position + UE position + viewing direction
    2. AttenuationNetwork: Encodes spatial position into 128D features
    3. AttenuationDecoder: Decodes features into M×N_UE attenuation factors
    4. RadianceNetwork: Processes UE info + spatial features into radiation factors
    5. Output: Attenuation + Radiation factors + MIMO channel matrix
    
    This model is designed for real-world OFDM systems like WiFi, 5G, and LTE.
    """
    
    def __init__(self, 
                 num_subcarriers: int = 408,
                 num_ue_antennas: int = 4,
                 num_bs_antennas: int = 64,
                 position_dim: int = 3,
                 hidden_dim: int = 256,
                 feature_dim: int = 128):
        """
        Initialize the Prism model.
        
        Args:
            num_subcarriers: Number of OFDM subcarriers (e.g., 408 for 5G)
            num_ue_antennas: Number of antennas at User Equipment (MIMO configuration)
            num_bs_antennas: Number of antennas at Base Station (MIMO configuration)
            position_dim: Spatial dimension of coordinates (typically 3 for 3D space)
            hidden_dim: Hidden dimension for all neural networks
            feature_dim: Feature dimension from AttenuationNetwork (default: 128)
        """
        super(PrismModel, self).__init__()
        
        # Store model configuration parameters
        self.num_subcarriers = num_subcarriers
        self.num_ue_antennas = num_ue_antennas
        self.num_bs_antennas = num_bs_antennas
        self.position_dim = position_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # Initialize core networks with new architecture
        self.attenuation_net = AttenuationNetwork(
            input_dim=position_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim
        )
        
        self.attenuation_decoder = AttenuationDecoder(
            feature_dim=feature_dim,
            num_subcarriers=num_subcarriers,
            num_ue_antennas=num_ue_antennas,
            hidden_dim=hidden_dim
        )
        
        self.radiance_net = RadianceNetwork(
            position_dim=position_dim,
            view_dim=position_dim,  # View direction has same dimension as position
            feature_dim=feature_dim,
            num_subcarriers=num_subcarriers,
            num_ue_antennas=num_ue_antennas,
            num_bs_antennas=num_bs_antennas,
            hidden_dim=hidden_dim
        )
        
        # MIMO channel matrix output layer
        # Converts attenuation and radiation factors to MIMO channel matrix
        # Input: concatenated attenuation and radiation factors (real + imag for both)
        mimo_input_dim = num_ue_antennas * num_subcarriers * 4  # *4 for real+imag of both factors
        self.mimo_output = nn.Linear(mimo_input_dim, num_ue_antennas * num_bs_antennas)
        
        # Initialize advanced features
        self._initialize_advanced_features()
        
    def forward(self, positions: torch.Tensor, ue_positions: torch.Tensor, 
                view_directions: torch.Tensor, bs_antenna_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete Prism model.
        
        This method orchestrates the forward pass to generate attenuation and radiation factors:
        1. Encode spatial position into compact features
        2. Decode features into attenuation factors
        3. Process UE info + features into radiation factors
        4. Store factors for later ray tracing processing (MIMO channel matrix generation)
        
        Args:
            positions: 3D spatial coordinates [batch_size, position_dim]
                Represents the spatial location for RF signal prediction
            ue_positions: UE position coordinates [batch_size, position_dim]
                User equipment location in 3D space
            view_directions: Viewing direction vectors [batch_size, position_dim]
                Direction vectors for radiation pattern calculation
            bs_antenna_ids: BS antenna IDs [batch_size] (1 to N_BS)
                Identifies which BS antenna to model for radiation characteristics
                
        Returns:
            Dictionary containing model outputs:
            - spatial_features: [batch_size, feature_dim]
                Compact spatial encoding from AttenuationNetwork
            - attenuation_factors: [batch_size, num_ue_antennas, num_subcarriers]
                Attenuation factors for each UE antenna and subcarrier (complex values)
            - radiation_factors: [batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers]
                Radiation factors for each UE-BS antenna pair and subcarrier (complex values)
            - csi_status: Dict (if CSI processing enabled)
                Status indicating that MIMO channel matrix requires ray tracing
                
        Note:
            This model outputs attenuation and radiation factors for each spatial sampling point.
            MIMO channel matrix generation requires additional ray tracing processing.
        """
        batch_size = positions.shape[0]
        
        # Step 1: Encode spatial position into compact features
        # This is the key innovation: single network for spatial encoding
        spatial_features = self.attenuation_net(positions)
        
        # Step 2: Decode features into attenuation factors
        # N_UE independent channels, each outputting M attenuation factors
        attenuation_factors = self.attenuation_decoder(spatial_features)
        
        # Step 3: Process UE info + spatial features + BS antenna ID into radiation factors
        # Each UE-BS antenna pair has independent processing
        radiation_factors = self.radiance_net(ue_positions, view_directions, spatial_features, bs_antenna_ids)
        
        # Step 4: Store attenuation and radiation factors for later processing
        # These factors will be used by ray tracing to generate MIMO channel matrix
        # Do NOT generate MIMO channel matrix here - it requires ray tracing
        
        # Step 5: Apply advanced processing if enabled
        if hasattr(self, 'csi_processor') and self.csi_processor:
            # Process CSI virtual links (requires MIMO channel matrix from ray tracing)
            # Note: MIMO channel matrix is not available here
            csi_results = {
                'status': 'pending_ray_tracing',
                'message': 'MIMO channel matrix requires ray tracing processing'
            }
            
            # Update outputs with CSI processing results
            outputs = {
                'spatial_features': spatial_features,
                'attenuation_factors': attenuation_factors,
                'radiation_factors': radiation_factors,
                'csi_status': csi_results
            }
        else:
            outputs = {
                'spatial_features': spatial_features,
                'attenuation_factors': attenuation_factors,
                'radiation_factors': radiation_factors
            }
        
        # Return comprehensive output dictionary
        return outputs
    
    def get_attenuation_factors(self, **kwargs) -> torch.Tensor:
        """
        Get attenuation factors for all UE antennas and subcarriers.
        
        Args:
            **kwargs: Same arguments as forward method (positions, ue_positions, view_directions, bs_antenna_ids)
            
        Returns:
            Attenuation factors [batch_size, num_ue_antennas, num_subcarriers]
        """
        outputs = self.forward(**kwargs)
        return outputs['attenuation_factors']
    
    def get_radiation_factors(self, **kwargs) -> torch.Tensor:
        """
        Get radiation factors for all UE-BS antenna pairs and subcarriers.
        
        Args:
            **kwargs: Same arguments as forward method (positions, ue_positions, view_directions, bs_antenna_ids)
            
        Returns:
            Radiation factors [batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers]
        """
        outputs = self.forward(**kwargs)
        return outputs['radiation_factors']
    
    def get_mimo_channel(self, **kwargs) -> torch.Tensor:
        """
        Get the MIMO channel matrix.
        
        Note: This method requires ray tracing to be enabled and processed.
        The model only outputs attenuation and radiation factors.
        
        Args:
            **kwargs: Same arguments as forward method (positions, ue_positions, view_directions, bs_antenna_ids)
            
        Returns:
            MIMO channel matrix [batch_size, num_ue_antennas, num_bs_antennas]
            
        Raises:
            RuntimeError: If ray tracing is not enabled or MIMO matrix not generated
        """
        if not hasattr(self, 'ray_tracer') or self.ray_tracer is None:
            raise RuntimeError("Ray tracing not enabled. Call enable_ray_tracing() first.")
        
        # Get attenuation and radiation factors
        outputs = self.forward(**kwargs)
        
        # Check if MIMO channel matrix has been generated by ray tracing
        if not hasattr(self, '_mimo_channel_matrix'):
            raise RuntimeError("MIMO channel matrix not yet generated. Complete ray tracing first.")
        
        return self._mimo_channel_matrix
    
    def _initialize_advanced_features(self):
        """Initialize advanced features (CSI processing and ray tracing)."""
        self.csi_processor = None
        self.ray_tracer = None
        self.environment = None
    
    def enable_csi_processing(self, config: Dict):
        """Enable CSI virtual link processing."""
        if not hasattr(self, 'csi_processor') or self.csi_processor is None:
            self.csi_processor = CSIVirtualLinkProcessor(
                m_subcarriers=self.num_subcarriers,
                n_ue_antennas=self.num_ue_antennas,
                n_bs_antennas=self.num_bs_antennas,
                config=config
            )
            
            # Store CSI configuration for sampling
            self.csi_config = config.get('csi_processing', {})
            
            logging.info("CSI virtual link processing enabled")
            
            # Log sampling configuration
            if self.csi_config.get('enable_random_sampling', False):
                sample_size = self.csi_config.get('sample_size', 64)
                logging.info(f"Virtual link sampling enabled: K={sample_size} links per antenna")
            else:
                logging.info("Virtual link sampling disabled: processing all links")
    
    def enable_ray_tracing(self, config: Dict):
        """Enable advanced ray tracing capabilities."""
        if not hasattr(self, 'ray_tracer') or self.ray_tracer is None:
            self.ray_tracer = AdvancedRayTracer(config)
            self.environment = Environment()
            
            # Set up default environment if specified in config
            if 'environment' in config:
                env_config = config['environment']
                if 'building_material' in env_config:
                    # Add default building
                    building = Building(
                        min_corner=[-50, -50, 0],
                        max_corner=[50, 50, 20],
                        material=env_config['building_material']
                    )
                    self.environment.add_obstacle(building)
                
                if 'atmospheric_conditions' in config:
                    atm_config = config['atmospheric_conditions']
                    self.environment.set_atmospheric_conditions(
                        atm_config.get('temperature', 20.0),
                        atm_config.get('humidity', 50.0),
                        atm_config.get('pressure', 1013.25)
                    )
            
            logging.info("Advanced ray tracing enabled")
    
    def trace_rays(self, source_position: Union[List, np.ndarray, torch.Tensor],
                   target_positions: List[Union[List, np.ndarray, torch.Tensor]] = None) -> Dict:
        """Perform ray tracing from source position."""
        if not hasattr(self, 'ray_tracer') or self.ray_tracer is None:
            raise RuntimeError("Ray tracing not enabled. Call enable_ray_tracing() first.")
        
        # Perform ray tracing
        ray_results = self.ray_tracer.trace_rays(source_position, target_positions or [], self.environment)
        
        # Analyze spatial distribution
        spatial_analysis = self.ray_tracer.analyze_spatial_distribution(ray_results)
        
        # Get statistics
        statistics = self.ray_tracer.get_ray_statistics(ray_results)
        
        return {
            'ray_paths': ray_results,
            'spatial_analysis': spatial_analysis,
            'statistics': statistics
        }
    
    def generate_mimo_channel_matrix(self, positions: torch.Tensor, ue_positions: torch.Tensor, 
                                   view_directions: torch.Tensor, bs_antenna_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate MIMO channel matrix through ray tracing processing.
        
        This method combines attenuation and radiation factors with ray tracing
        to generate the complete MIMO channel matrix.
        
        Args:
            positions: 3D spatial coordinates [batch_size, position_dim]
            ue_positions: UE position coordinates [batch_size, position_dim]
            view_directions: Viewing direction vectors [batch_size, position_dim]
            bs_antenna_ids: BS antenna IDs [batch_size] (1 to N_BS)
            
        Returns:
            MIMO channel matrix [batch_size, num_ue_antennas, num_bs_antennas]
            
        Raises:
            RuntimeError: If ray tracing is not enabled
        """
        if not hasattr(self, 'ray_tracer') or self.ray_tracer is None:
            raise RuntimeError("Ray tracing not enabled. Call enable_ray_tracing() first.")
        
        # Get attenuation and radiation factors
        outputs = self.forward(positions, ue_positions, view_directions, bs_antenna_ids)
        attenuation_factors = outputs['attenuation_factors']
        radiation_factors = outputs['radiation_factors']
        
        # Use ray tracing to process these factors and generate MIMO matrix
        # This is where the actual MIMO channel matrix generation happens
        # For now, we'll use a placeholder implementation
        batch_size = positions.shape[0]
        
        # Combine attenuation and radiation factors for MIMO processing
        combined_factors = torch.cat([
            attenuation_factors.real, attenuation_factors.imag,
            radiation_factors.real, radiation_factors.imag
        ], dim=2)  # [batch_size, num_ue_antennas, num_subcarriers * 4]
        
        # Flatten for MIMO processing
        flattened_factors = combined_factors.view(batch_size, -1)
        
        # Generate MIMO channel matrix using the stored linear layer
        mimo_channel = self.mimo_output(flattened_factors)
        mimo_channel = mimo_channel.view(batch_size, self.num_ue_antennas, self.num_bs_antennas)
        
        # Store the generated MIMO matrix for later use
        self._mimo_channel_matrix = mimo_channel
        
        return mimo_channel
    
    def get_csi_analysis(self, **kwargs) -> Dict:
        """Get CSI virtual link analysis results."""
        if not hasattr(self, 'csi_processor') or self.csi_processor is None:
            raise RuntimeError("CSI processing not enabled. Call enable_csi_processing() first.")
        
        # Run forward pass to get CSI analysis
        outputs = self.forward(**kwargs)
        
        if 'csi_analysis' in outputs:
            return outputs['csi_analysis']
        else:
            return {}
    
    def get_virtual_link_statistics(self, **kwargs) -> Dict:
        """Get virtual link statistics."""
        if not hasattr(self, 'csi_processor') or self.csi_processor is None:
            raise RuntimeError("CSI processing not enabled. Call enable_csi_processing() first.")
        
        outputs = self.forward(**kwargs)
        
        if 'csi_virtual_links' in outputs:
            virtual_links = outputs['csi_virtual_links']
            return self.csi_processor.get_virtual_link_statistics(virtual_links)
        else:
            return {}

class PrismLoss(nn.Module):
    """
    Loss function for Prism model training based on virtual link received signals.
    
    This loss function implements the correct approach for MIMO-OFDM systems:
    1. Model outputs electromagnetic properties at spatial sampling points
    2. Ray tracing accumulates RF signals from all directions for each BS antenna
    3. Loss is computed between predicted and actual received signals at each antenna
    
    Key Features:
    - Loss computation based on actual received signals at BS antennas
    - Ray tracing integration for signal accumulation
    - Virtual link-based optimization
    - Frequency-aware subcarrier processing
    
    The loss function ensures the model learns to predict electromagnetic properties
    that lead to correct received signals after ray tracing accumulation.
    """
    
    def __init__(self, loss_type: str = 'mse'):
        """
        Initialize the Prism loss function.
        
        Args:
            loss_type: Type of loss function ('mse' or 'l1')
                - 'mse': Mean Squared Error (good for continuous signals)
                - 'l1': Mean Absolute Error (more robust to outliers)
        """
        super(PrismLoss, self).__init__()
        self.loss_type = loss_type
        
        # Initialize the appropriate loss criterion
        if loss_type == 'mse':
            # MSE with 'none' reduction allows per-sample, per-subcarrier loss computation
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            # L1 loss with 'none' reduction for per-subcarrier computation
            self.criterion = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], 
                ray_tracer, environment, positions: torch.Tensor, ue_positions: torch.Tensor,
                view_directions: torch.Tensor, bs_antenna_ids: torch.Tensor,
                weights: Optional[torch.Tensor] = None,
                config: Optional[Dict] = None) -> torch.Tensor:
        """
        Compute loss based on virtual link received signals after ray tracing.
        
        This method implements the correct loss computation approach:
        1. Use model predictions to get electromagnetic properties at sampling points
        2. Perform ray tracing to accumulate RF signals from all directions
        3. Compare predicted received signals with actual received signals at BS antennas
        
        Args:
            predictions: Model outputs dictionary containing:
                - attenuation_factors: [batch_size, num_ue_antennas, num_subcarriers]
                - radiation_factors: [batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers]
            targets: Ground truth dictionary containing actual received signals:
                - received_signals: [batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers]
            ray_tracer: Ray tracing engine for signal accumulation
            environment: Environment for ray tracing
            positions: Spatial sampling points [batch_size, position_dim]
            ue_positions: UE positions [batch_size, position_dim]
            view_directions: Viewing directions [batch_size, position_dim]
            bs_antenna_ids: BS antenna IDs [batch_size] (1 to N_BS)
            weights: Optional weights for each subcarrier [num_subcarriers]
            config: Optional configuration dictionary
                
        Returns:
            Total loss value (scalar tensor)
        """
        total_loss = 0.0
        
        # Step 1: Perform ray tracing to accumulate RF signals for each BS antenna
        predicted_received_signals = self._compute_received_signals(
            predictions, ray_tracer, environment, positions, ue_positions, 
            view_directions, bs_antenna_ids
        )
        
        # Step 2: Compute loss between predicted and actual received signals
        if 'received_signals' in targets:
            received_loss = self._compute_received_signal_loss(
                predicted_received_signals, targets['received_signals'], weights
            )
            total_loss += received_loss
        
        # Step 3: Add optional regularization terms
        if config and 'loss' in config:
            loss_config = config['loss']
            
            # Add electromagnetic property regularization if specified
            if loss_config.get('em_property_weight', 0) > 0:
                em_loss = self._compute_em_property_regularization(predictions)
                total_loss += loss_config['em_property_weight'] * em_loss
        
        return total_loss
    
    def _compute_received_signals(self, predictions: Dict[str, torch.Tensor], ray_tracer, environment,
                                 positions: torch.Tensor, ue_positions: torch.Tensor,
                                 view_directions: torch.Tensor, bs_antenna_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute received signals at BS antennas through ray tracing accumulation.
        
        This method performs ray tracing from each spatial sampling point to each BS antenna,
        accumulating RF signals from all directions to compute the total received signal.
        
        Args:
            predictions: Model predictions containing attenuation and radiation factors
            ray_tracer: Ray tracing engine
            environment: Environment for ray tracing
            positions: Spatial sampling points
            ue_positions: UE positions
            view_directions: Viewing directions
            bs_antenna_ids: BS antenna IDs
            
        Returns:
            Predicted received signals [batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers]
        """
        batch_size = positions.shape[0]
        num_ue_antennas = predictions['attenuation_factors'].shape[1]
        num_bs_antennas = predictions['radiation_factors'].shape[2]
        num_subcarriers = predictions['attenuation_factors'].shape[2]
        
        # Initialize received signals tensor
        received_signals = torch.zeros(batch_size, num_ue_antennas, num_bs_antennas, 
                                     num_subcarriers, dtype=torch.complex64, device=positions.device)
        
        # For each spatial sampling point, perform ray tracing to all BS antennas
        for i in range(batch_size):
            for ue_idx in range(num_ue_antennas):
                for bs_idx in range(num_bs_antennas):
                    # Get electromagnetic properties at this sampling point
                    attenuation = predictions['attenuation_factors'][i, ue_idx, :]  # [num_subcarriers]
                    radiation = predictions['radiation_factors'][i, ue_idx, bs_idx, :]  # [num_subcarriers]
                    
                    # Perform ray tracing from this point to the BS antenna
                    # This is a simplified implementation - in practice, you'd use the full ray tracer
                    ray_results = self._simplified_ray_tracing(
                        positions[i], ue_positions[i], view_directions[i], bs_idx, 
                        ray_tracer, environment
                    )
                    
                    # Accumulate RF signals from all directions
                    # The actual implementation would integrate over all ray directions
                    accumulated_signal = self._accumulate_ray_signals(
                        ray_results, attenuation, radiation
                    )
                    
                    received_signals[i, ue_idx, bs_idx, :] = accumulated_signal
        
        return received_signals
    
    def _simplified_ray_tracing(self, position: torch.Tensor, ue_position: torch.Tensor,
                               view_direction: torch.Tensor, bs_antenna_idx: int,
                               ray_tracer, environment) -> Dict:
        """
        Simplified ray tracing for demonstration purposes.
        In practice, this would use the full ray tracing engine.
        """
        # This is a placeholder for the actual ray tracing implementation
        # In practice, you would call ray_tracer.trace_rays() with proper parameters
        return {
            'ray_paths': [],
            'signal_strength': torch.ones(1, device=position.device),
            'phase_shifts': torch.zeros(1, device=position.device)
        }
    
    def _accumulate_ray_signals(self, ray_results: Dict, attenuation: torch.Tensor, 
                               radiation: torch.Tensor) -> torch.Tensor:
        """
        Accumulate RF signals from ray tracing results.
        
        Args:
            ray_results: Results from ray tracing
            attenuation: Attenuation factors [num_subcarriers]
            radiation: Radiation factors [num_subcarriers]
            
        Returns:
            Accumulated signal [num_subcarriers]
        """
        # This is a simplified implementation
        # In practice, you would integrate over all ray directions and apply
        # proper electromagnetic propagation models
        
        # Combine attenuation and radiation factors
        # The actual implementation would be more complex, involving:
        # - Path loss from ray tracing
        # - Phase shifts from propagation delays
        # - Antenna pattern effects
        # - Multi-path interference
        
        accumulated_signal = attenuation * radiation
        
        return accumulated_signal
    
    def _compute_received_signal_loss(self, predicted_signals: torch.Tensor, 
                                    target_signals: torch.Tensor,
                                    weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute loss between predicted and actual received signals.
        
        Args:
            predicted_signals: [batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers]
            target_signals: [batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers]
            weights: Optional subcarrier weights
            
        Returns:
            Loss value
        """
        # Handle complex numbers
        if predicted_signals.is_complex():
            pred_real = predicted_signals.real
            pred_imag = predicted_signals.imag
            target_real = target_signals.real
            target_imag = target_signals.imag
            
            # Compute loss for real and imaginary parts
            real_loss = self.criterion(pred_real, target_real)
            imag_loss = self.criterion(pred_imag, target_imag)
            
            signal_loss = real_loss + imag_loss
        else:
            signal_loss = self.criterion(predicted_signals, target_signals)
        
        # Apply optional subcarrier weights
        if weights is not None:
            weights_expanded = weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            signal_loss = signal_loss * weights_expanded
        
        return torch.sum(signal_loss)
    
    def _compute_em_property_regularization(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute electromagnetic property regularization term.
        
        This can include physical constraints like:
        - Energy conservation
        - Causality constraints
        - Physical bounds on electromagnetic properties
        """
        regularization_loss = 0.0
        
        # Example: Energy conservation constraint
        if 'attenuation_factors' in predictions:
            # Attenuation factors should generally be <= 1 (no amplification)
            atten_magnitude = torch.abs(predictions['attenuation_factors'])
            energy_violation = torch.relu(atten_magnitude - 1.0)
            regularization_loss += torch.mean(energy_violation ** 2)
        
        if 'radiation_factors' in predictions:
            # Radiation factors should have reasonable magnitudes
            rad_magnitude = torch.abs(predictions['radiation_factors'])
            rad_violation = torch.relu(rad_magnitude - 10.0)  # Adjust threshold as needed
            regularization_loss += torch.mean(rad_violation ** 2)
        
        return regularization_loss

def create_prism_model(config: Dict) -> PrismModel:
    """
    Factory function to create a Prism model from configuration.
    
    This function provides a convenient way to instantiate Prism models
    with different configurations using the new architecture.
    
    Args:
        config: Configuration dictionary containing model parameters:
            - num_subcarriers: Number of OFDM subcarriers
            - num_ue_antennas: Number of UE antennas
            - num_bs_antennas: Number of BS antennas
            - position_dim: Spatial dimension
            - hidden_dim: Hidden layer dimension
            - feature_dim: Feature vector dimension (default: 128)
        
    Returns:
        Configured PrismModel instance ready for training or inference
    """
    # Extract model configuration
    model_config = config.get('model', {})
    
    model = PrismModel(
        num_subcarriers=model_config.get('num_subcarriers', 408),
        num_ue_antennas=model_config.get('num_ue_antennas', 4),
        num_bs_antennas=model_config.get('num_bs_antennas', 64),
        position_dim=model_config.get('position_dim', 3),
        hidden_dim=model_config.get('hidden_dim', 256),
        feature_dim=model_config.get('feature_dim', 128)
    )
    
    # Configure advanced features if specified in config
    if config and 'csi_processing' in config and config['csi_processing'].get('virtual_link_enabled', False):
        model.enable_csi_processing(config)
    
    if config and 'ray_tracing' in config and config['ray_tracing'].get('enabled', False):
        model.enable_ray_tracing(config)
    
    return model
