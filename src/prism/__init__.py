"""
Prism: Discrete Electromagnetic Ray Tracing System

A PyTorch-based implementation of discrete electromagnetic ray tracing
with MLP-based direction sampling for RF signal strength computation.
"""

__version__ = "2.0.0"
__author__ = "Prism Project Team"
__email__ = "contact@prism-project.org"

from .tracers import (
    NaiveRayTracer,
    NNRayTracer,
    LowRankRayTracer
)

# from .spatial_spectrum import (
#     csi_to_spatial_spectrum,
#     calculate_spatial_spectrum,
#     calculate_bartlett_spectrum,
#     calculate_capon_spectrum,
#     calculate_music_spectrum,
#     plot_spatial_spectrum,
#     find_peak_directions,
#     generate_steering_vector,
#     fuse_subcarrier_spectrums
# )

from .loss import (
    LossFunction,
    CSILoss,
    PDPLoss,
    SSLoss,
    DEFAULT_LOSS_CONFIG
)

from .networks import (
    AttenuationNetwork,
    AntennaEmbeddingCodebook,
    AntennaNetwork,
    RadianceNetwork,
    PrismNetwork,
    AttenuationNetworkConfig,
    AntennaEmbeddingCodebookConfig,
    AntennaNetworkConfig,
    RadianceNetworkConfig,
    PrismNetworkConfig
)

from .training_interface import (
    PrismTrainingInterface
)



__all__ = [
    # Core ray tracing
    'RayTracer',
    'LowRankRayTracer',
    
    # Spatial spectrum estimation
    'csi_to_spatial_spectrum',
    'calculate_spatial_spectrum',
    'calculate_bartlett_spectrum',
    'calculate_capon_spectrum',
    'calculate_music_spectrum',
    'plot_spatial_spectrum',
    'find_peak_directions',
    'generate_steering_vector',
    'fuse_subcarrier_spectrums',
    
    # Loss functions
    'LossFunction',
    'CSILoss',
    'PDPLoss',
    'SSLoss',
    'DEFAULT_LOSS_CONFIG',
    
    # Neural networks
    'AttenuationNetwork',
    'AntennaEmbeddingCodebook',
    'AntennaNetwork',
    'RadianceNetwork',
    'PrismNetwork',
    'AttenuationNetworkConfig',
    'AntennaEmbeddingCodebookConfig',
    'AntennaNetworkConfig',
    'RadianceNetworkConfig',
    'PrismNetworkConfig',
    
    # Training interface
    'PrismTrainingInterface',
    

    
    # Version info
    '__version__',
    '__author__',
    '__email__'
]
