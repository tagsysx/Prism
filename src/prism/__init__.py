"""
Prism: Discrete Electromagnetic Ray Tracing System

A PyTorch-based implementation of discrete electromagnetic ray tracing
with MLP-based direction sampling for RF signal strength computation.
"""

__version__ = "2.0.0"
__author__ = "Prism Project Team"
__email__ = "contact@prism-project.org"

from .ray_tracer import (
    DiscreteRayTracer,
    Ray,
    BaseStation,
    UserEquipment
)

from .ray_tracer_cuda import (
    CUDARayTracer
)



from .networks import (
    AttenuationNetwork,
    AttenuationDecoder,
    AntennaEmbeddingCodebook,
    AntennaNetwork,
    RadianceNetwork,
    PrismNetwork,
    AttenuationNetworkConfig,
    AttenuationDecoderConfig,
    AntennaEmbeddingCodebookConfig,
    AntennaNetworkConfig,
    RadianceNetworkConfig,
    PrismNetworkConfig
)

from .training_interface import (
    PrismTrainingInterface
)

from .loss_functions import (
    PrismLoss,
    FrequencyAwareLoss,
    CSIVirtualLinkLoss
)

__all__ = [
    # Core ray tracing
    'DiscreteRayTracer',
    'Ray',
    'BaseStation',
    'UserEquipment',
    
    # CUDA-accelerated ray tracing
    'CUDARayTracer',
    

    
    # Neural networks
    'AttenuationNetwork',
    'AttenuationDecoder',
    'AntennaEmbeddingCodebook',
    'AntennaNetwork',
    'RadianceNetwork',
    'PrismNetwork',
    'AttenuationNetworkConfig',
    'AttenuationDecoderConfig',
    'AntennaEmbeddingCodebookConfig',
    'AntennaNetworkConfig',
    'RadianceNetworkConfig',
    'PrismNetworkConfig',
    
    # Training interface
    'PrismTrainingInterface',
    
    # Loss functions
    'PrismLoss',
    'FrequencyAwareLoss',
    'CSIVirtualLinkLoss',
    
    # Version info
    '__version__',
    '__author__',
    '__email__'
]
