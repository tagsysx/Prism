"""
Prism Model: Wideband RF Neural Radiance Fields for OFDM Communication.

This module implements the core Prism model architecture that extends NeRF2
to handle wideband RF signals with multiple subcarriers. The model is designed
specifically for OFDM communication systems and includes:

Key Components:
- RFPrismModule: Multi-channel MLP for subcarrier decomposition
- AttenuationNetwork: Predicts signal attenuation based on position/environment
- RadianceNetwork: Predicts signal radiance characteristics
- PrismModel: Main model combining all components
- PrismLoss: Frequency-aware loss function for multi-subcarrier optimization

The architecture enables processing of wideband RF signals with:
- Configurable number of subcarriers (52 to 1024+)
- MIMO antenna configurations
- 3D spatial awareness
- Frequency-dependent signal modeling
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

class RFPrismModule(nn.Module):
    """
    RF Prism Module: Multi-channel MLP for decomposing global features into subcarrier components.
    
    This is the core innovation that enables wideband RF signal processing.
    Instead of using a single network for all subcarriers, this module uses
    separate MLP channels for each subcarrier, allowing independent learning
    of frequency-specific characteristics.
    
    Architecture:
    - C independent MLP channels (one per subcarrier)
    - Each channel has 2 hidden layers with ReLU activation
    - Outputs are concatenated to form subcarrier responses
    
    This design is inspired by the multi-frequency nature of OFDM signals
    where different subcarriers may have different propagation characteristics
    and channel responses.
    """
    def __init__(self, input_dim: int, num_subcarriers: int, hidden_dim: int = 256):
        """
        Initialize the RF Prism Module.
        
        Args:
            input_dim: Dimension of input features (concatenated attenuation + radiance features)
            num_subcarriers: Number of subcarriers to model (e.g., 52 for WiFi, 1024 for 5G)
            hidden_dim: Hidden dimension for each MLP channel
        """
        super(RFPrismModule, self).__init__()
        self.num_subcarriers = num_subcarriers
        self.hidden_dim = hidden_dim
        
        # Create independent MLP channels for each subcarrier
        # This allows each subcarrier to learn its own frequency-specific features
        self.layer1 = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_subcarriers)
        ])
        self.layer2 = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_subcarriers)
        ])
        
        # Output layer for each subcarrier (produces scalar response)
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_subcarriers)
        ])
        
        # ReLU activation for non-linearity
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RF Prism Module.
        
        This method processes input features through C independent MLP channels,
        where C is the number of subcarriers. Each channel learns to predict
        the response for a specific subcarrier frequency.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
                Contains concatenated features from attenuation and radiance networks
                
        Returns:
            Output tensor of shape [batch_size, num_subcarriers]
                Each column represents the response for a specific subcarrier
        """
        batch_size = x.shape[0]
        outputs = []
        
        # Process each subcarrier independently through its dedicated MLP channel
        for i in range(self.num_subcarriers):
            # First hidden layer
            h = self.layer1[i](x)
            h = self.activation(h)
            
            # Second hidden layer
            h = self.layer2[i](h)
            h = self.activation(h)
            
            # Output layer (produces scalar response for this subcarrier)
            h = self.output_layers[i](h)
            outputs.append(h)
        
        # Concatenate outputs from all subcarrier channels
        # Result: [batch_size, num_subcarriers] tensor
        return torch.cat(outputs, dim=1)

class AttenuationNetwork(nn.Module):
    """
    Attenuation Network: Predicts signal attenuation based on position and environment.
    
    This network models how RF signals attenuate as they propagate through space.
    It takes into account:
    - 3D spatial position (x, y, z coordinates)
    - Antenna configurations (UE and BS antenna features)
    - Environmental factors (encoded in additional features)
    
    The network uses a deep MLP architecture to learn complex non-linear
    relationships between spatial position and signal attenuation.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 8):
        """
        Initialize the Attenuation Network.
        
        Args:
            input_dim: Dimension of input features (position + antenna + environment)
            hidden_dim: Hidden dimension for all layers
            num_layers: Number of hidden layers in the network
        """
        super(AttenuationNetwork, self).__init__()
        
        # Build the MLP architecture
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Add intermediate hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.network = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Attenuation Network.
        
        Args:
            x: Input tensor containing position, antenna, and environment features
            
        Returns:
            Hidden representation of attenuation characteristics
        """
        # Process through all hidden layers with ReLU activation
        for layer in self.network:
            x = self.activation(layer(x))
        return x

class RadianceNetwork(nn.Module):
    """
    Radiance Network: Predicts signal radiance characteristics.
    
    This network models the directional and intensity characteristics of RF signals.
    It captures:
    - Signal strength variations with direction
    - Antenna radiation patterns
    - Multi-path effects and reflections
    - Frequency-dependent radiation characteristics
    
    Similar to the Attenuation Network, it uses a deep MLP architecture
    to learn complex spatial relationships in RF signal propagation.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 8):
        """
        Initialize the Radiance Network.
        
        Args:
            input_dim: Dimension of input features (position + antenna + environment)
            hidden_dim: Hidden dimension for all layers
            num_layers: Number of hidden layers in the network
        """
        super(RadianceNetwork, self).__init__()
        
        # Build the MLP architecture (identical to Attenuation Network)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Add intermediate hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.network = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Radiance Network.
        
        Args:
            x: Input tensor containing position, antenna, and environment features
            
        Returns:
            Hidden representation of radiance characteristics
        """
        # Process through all hidden layers with ReLU activation
        for layer in self.network:
            x = self.activation(layer(x))
        return x

class PrismModel(nn.Module):
    """
    Prism: Wideband RF Neural Radiance Fields for OFDM Communication.
    
    This is the main model that combines all components to create a comprehensive
    RF signal model. It extends NeRF2 concepts to handle wideband RF signals with:
    
    Key Features:
    - M subcarriers (configurable from 52 to 1024+)
    - N_UE antennas at the User Equipment
    - N_BS antennas at the Base Station
    - RF Prism Module for subcarrier decomposition
    - 3D spatial awareness for position-dependent modeling
    - MIMO channel matrix generation
    
    Architecture Flow:
    1. Input: 3D position + antenna features + RF environment features
    2. Attenuation Network: Learns spatial attenuation patterns
    3. Radiance Network: Learns directional radiation patterns
    4. RF Prism Module: Decomposes features into subcarrier responses
    5. Output: Subcarrier responses + MIMO channel matrix
    
    This model is designed for real-world OFDM systems like WiFi, 5G, and LTE.
    """
    
    def __init__(self, 
                 num_subcarriers: int = 1024,
                 num_ue_antennas: int = 2,
                 num_bs_antennas: int = 4,
                 position_dim: int = 3,
                 hidden_dim: int = 256):
        """
        Initialize the Prism model.
        
        Args:
            num_subcarriers: Number of OFDM subcarriers (e.g., 52 for WiFi, 1024 for 5G)
            num_ue_antennas: Number of antennas at User Equipment (MIMO configuration)
            num_bs_antennas: Number of antennas at Base Station (MIMO configuration)
            position_dim: Spatial dimension of coordinates (typically 3 for 3D space)
            hidden_dim: Hidden dimension for all neural networks
        """
        super(PrismModel, self).__init__()
        
        # Store model configuration parameters
        self.num_subcarriers = num_subcarriers
        self.num_ue_antennas = num_ue_antennas
        self.num_bs_antennas = num_bs_antennas
        self.position_dim = position_dim
        self.hidden_dim = hidden_dim
        
        # Calculate total input dimension
        # Input includes: position + UE antennas + BS antennas + additional RF features
        input_dim = position_dim + num_ue_antennas + num_bs_antennas + 10  # 10 for additional RF features
        
        # Initialize core networks
        # These networks learn spatial and environmental RF characteristics
        self.attenuation_net = AttenuationNetwork(input_dim, hidden_dim)
        self.radiance_net = RadianceNetwork(input_dim, hidden_dim)
        
        # RF Prism Module: decomposes global features into subcarrier components
        # Input dimension is concatenated features from both networks
        prism_input_dim = hidden_dim * 2  # Concatenated features from both networks
        self.rf_prism = RFPrismModule(prism_input_dim, num_subcarriers, hidden_dim)
        
        # MIMO channel matrix output layer
        # Converts subcarrier responses to MIMO channel matrix
        # Output shape: [batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas]
        self.mimo_output = nn.Linear(1, num_ue_antennas * num_bs_antennas)
        
        # Initialize advanced features
        self._initialize_advanced_features()
        
    def forward(self, positions: torch.Tensor, ue_antennas: torch.Tensor, 
                bs_antennas: torch.Tensor, additional_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete Prism model.
        
        This method orchestrates the entire forward pass:
        1. Concatenates all input features
        2. Processes through attenuation and radiance networks
        3. Combines features for subcarrier decomposition
        4. Generates subcarrier responses and MIMO channel matrix
        
        Args:
            positions: 3D spatial coordinates [batch_size, position_dim]
                Represents the spatial location for RF signal prediction
            ue_antennas: UE antenna features [batch_size, num_ue_antennas]
                Contains antenna-specific characteristics and configurations
            bs_antennas: BS antenna features [batch_size, num_bs_antennas]
                Contains base station antenna characteristics
            additional_features: Additional RF features [batch_size, 10]
                Environmental factors, frequency, power, etc.
            
        Returns:
            Dictionary containing model outputs:
            - subcarrier_responses: [batch_size, num_subcarriers]
                Frequency response for each subcarrier
            - mimo_channel: [batch_size, num_ue_antennas * num_bs_antennas]
                Flattened MIMO channel matrix
            - attenuation_features: [batch_size, hidden_dim]
                Learned attenuation characteristics
            - radiance_features: [batch_size, hidden_dim]
                Learned radiance characteristics
        """
        # Step 1: Concatenate all input features into a single tensor
        # This creates a comprehensive feature vector for RF modeling
        x = torch.cat([positions, ue_antennas, bs_antennas, additional_features], dim=1)
        
        # Step 2: Extract features from core networks
        # These networks learn spatial and environmental RF characteristics
        attenuation_features = self.attenuation_net(x)
        radiance_features = self.radiance_net(x)
        
        # Step 3: Concatenate features for RF Prism Module
        # The prism module needs both attenuation and radiance information
        combined_features = torch.cat([attenuation_features, radiance_features], dim=1)
        
        # Step 4: Decompose into subcarrier components
        # This is the key innovation: frequency-specific signal modeling
        subcarrier_responses = self.rf_prism(combined_features)
        
        # Step 5: Generate MIMO channel matrix for each subcarrier
        # Process each subcarrier through the MIMO output layer
        batch_size = positions.shape[0]
        mimo_channels = []
        
        for i in range(self.num_subcarriers):
            # Extract features for this subcarrier
            subcarrier_features = subcarrier_responses[:, i:i+1]  # [batch_size, 1]
            # Generate MIMO matrix for this subcarrier
            mimo_matrix = self.mimo_output(subcarrier_features)  # [batch_size, num_ue_antennas * num_bs_antennas]
            mimo_channels.append(mimo_matrix)
        
        # Stack all subcarrier MIMO matrices
        mimo_channel = torch.stack(mimo_channels, dim=1)  # [batch_size, num_subcarriers, num_ue_antennas * num_bs_antennas]
        
        # Step 6: Apply advanced processing if enabled
        if hasattr(self, 'csi_processor') and self.csi_processor:
            # Process CSI virtual links
            mimo_channel_reshaped = mimo_channel.view(
                batch_size, 
                self.num_subcarriers, 
                self.num_ue_antennas, 
                self.num_bs_antennas
            )
            
            # Get sampling configuration
            sample_size = None
            if hasattr(self, 'csi_config') and self.csi_config:
                if self.csi_config.get('enable_random_sampling', False):
                    sample_size = self.csi_config.get('sample_size', 64)
            
            csi_results = self.csi_processor.process_virtual_links(
                mimo_channel_reshaped, positions, additional_features, sample_size
            )
            
            # Update outputs with CSI processing results
            outputs = {
                'subcarrier_responses': subcarrier_responses,
                'mimo_channel': mimo_channel,
                'attenuation_features': attenuation_features,
                'radiance_features': radiance_features,
                'csi_virtual_links': csi_results.get('processed_virtual_links', mimo_channel_reshaped),
                'csi_analysis': csi_results
            }
        else:
            outputs = {
                'subcarrier_responses': subcarrier_responses,
                'mimo_channel': mimo_channel,
                'attenuation_features': attenuation_features,
                'radiance_features': radiance_features
            }
        
        # Return comprehensive output dictionary
        return outputs
    
    def get_subcarrier_response(self, subcarrier_idx: int, **kwargs) -> torch.Tensor:
        """
        Get response for a specific subcarrier.
        
        This utility method allows extracting the response for a single
        subcarrier, which is useful for:
        - Frequency-specific analysis
        - Debugging individual subcarrier performance
        - Selective subcarrier processing
        
        Args:
            subcarrier_idx: Index of the subcarrier (0 to num_subcarriers-1)
            **kwargs: Same arguments as forward method (positions, ue_antennas, etc.)
            
        Returns:
            Response tensor for the specified subcarrier [batch_size, 1]
        """
        # Run full forward pass and extract specific subcarrier
        outputs = self.forward(**kwargs)
        return outputs['subcarrier_responses'][:, subcarrier_idx:subcarrier_idx+1]
    
    def get_mimo_channel(self, **kwargs) -> torch.Tensor:
        """
        Get the MIMO channel matrix in proper shape.
        
        This utility method returns the MIMO channel matrix reshaped
        from flattened form to proper 3D tensor for easier manipulation.
        
        Args:
            **kwargs: Same arguments as forward method (positions, ue_antennas, etc.)
            
        Returns:
            MIMO channel matrix [batch_size, num_ue_antennas, num_bs_antennas]
        """
        # Run forward pass and reshape MIMO channel output
        outputs = self.forward(**kwargs)
        return outputs['mimo_channel'].view(-1, self.num_ue_antennas, self.num_bs_antennas)
    
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
    Loss function for Prism model training.
    
    This loss function implements frequency-aware optimization for multi-subcarrier
    OFDM signals. It's designed to handle the unique characteristics of wideband
    RF signals where different subcarriers may have different importance or
    characteristics.
    
    Key Features:
    - Independent subcarrier loss computation
    - Configurable loss types (MSE, L1)
    - Optional subcarrier weighting
    - Maintains frequency independence
    
    The loss function ensures that the model learns to predict each subcarrier
    independently while maintaining overall signal quality.
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
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                weights: Optional[torch.Tensor] = None,
                csi_targets: Optional[torch.Tensor] = None,
                ray_tracing_targets: Optional[torch.Tensor] = None,
                config: Optional[Dict] = None) -> torch.Tensor:
        """
        Compute the frequency-aware loss.
        
        This method computes loss independently for each subcarrier and then
        combines them. This approach is crucial for OFDM systems where:
        - Different subcarriers may have different importance
        - Frequency-dependent characteristics need independent optimization
        - Overall signal quality depends on all subcarriers
        
        Args:
            predictions: Predicted subcarrier responses [batch_size, num_subcarriers]
                Model outputs for each subcarrier frequency
            targets: Ground truth subcarrier responses [batch_size, num_subcarriers]
                True subcarrier responses from measurements or simulation
            weights: Optional weights for each subcarrier [num_subcarriers]
                Allows prioritizing certain subcarriers (e.g., pilot subcarriers)
                
        Returns:
            Total loss value (scalar tensor)
        """
        # Step 1: Compute loss for each subcarrier independently
        # Result: [batch_size, num_subcarriers] tensor of per-subcarrier losses
        per_subcarrier_loss = self.criterion(predictions, targets)
        
        # Step 2: Apply optional subcarrier weights if provided
        # This allows frequency-dependent importance weighting
        if weights is not None:
            # Expand weights to match batch dimension for broadcasting
            per_subcarrier_loss = per_subcarrier_loss * weights.unsqueeze(0)
        
        # Step 3: Sum across all subcarriers while maintaining independence
        # This gives equal importance to all subcarriers (unless weighted)
        total_loss = torch.sum(per_subcarrier_loss)
        
        # Step 4: Add CSI virtual link loss if enabled
        if csi_targets is not None and config and 'loss' in config:
            loss_config = config['loss']
            if loss_config.get('csi_loss_weight', 0) > 0:
                csi_loss = self._compute_csi_loss(predictions, csi_targets)
                total_loss += loss_config['csi_loss_weight'] * csi_loss
        
        # Step 5: Add ray tracing loss if enabled
        if ray_tracing_targets is not None and config and 'loss' in config:
            loss_config = config['loss']
            if loss_config.get('ray_tracing_loss_weight', 0) > 0:
                ray_loss = self._compute_ray_tracing_loss(predictions, ray_tracing_targets)
                total_loss += loss_config['ray_tracing_loss_weight'] * ray_loss
        
        return total_loss
    
    def _compute_csi_loss(self, predictions: torch.Tensor, csi_targets: torch.Tensor) -> torch.Tensor:
        """Compute CSI virtual link loss."""
        # Simple MSE loss for CSI targets
        return torch.mean((predictions - csi_targets) ** 2)
    
    def _compute_ray_tracing_loss(self, predictions: torch.Tensor, ray_tracing_targets: torch.Tensor) -> torch.Tensor:
        """Compute ray tracing loss."""
        # Simple MSE loss for ray tracing targets
        return torch.mean((predictions - ray_tracing_targets) ** 2)

def create_prism_model(config: Dict) -> PrismModel:
    """
    Factory function to create a Prism model from configuration.
    
    This function provides a convenient way to instantiate Prism models
    with different configurations. It's useful for:
    - Experimentation with different model sizes
    - Configuration-based model creation
    - Easy hyperparameter tuning
    
    Args:
        config: Configuration dictionary containing model parameters:
            - num_subcarriers: Number of OFDM subcarriers
            - num_ue_antennas: Number of UE antennas
            - num_bs_antennas: Number of BS antennas
            - position_dim: Spatial dimension
            - hidden_dim: Hidden layer dimension
        
    Returns:
        Configured PrismModel instance ready for training or inference
    """
    # Extract model configuration
    model_config = config.get('model', {})
    
    model = PrismModel(
        num_subcarriers=model_config.get('num_subcarriers', 1024),
        num_ue_antennas=model_config.get('num_ue_antennas', 2),
        num_bs_antennas=model_config.get('num_bs_antennas', 4),
        position_dim=model_config.get('position_dim', 3),
        hidden_dim=model_config.get('hidden_dim', 256)
    )
    
    # Configure advanced features if specified in config
    if config and 'csi_processing' in config and config['csi_processing'].get('virtual_link_enabled', False):
        model.enable_csi_processing(config)
    
    if config and 'ray_tracing' in config and config['ray_tracing'].get('enabled', False):
        model.enable_ray_tracing(config)
    
    return model
