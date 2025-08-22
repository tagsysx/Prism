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



from .rf_signal_processor import (
    RFSignalProcessor,
    SignalStrengthCalculator,
    SubcarrierSelector
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



__all__ = [
    # Core ray tracing
    'DiscreteRayTracer',
    'Ray',
    'RayIntersection',
    'RayPath',
    'BaseStation',
    'UserEquipment',
    
    # CUDA-accelerated ray tracing
    'CUDARayTracer',
    

    
    # RF signal processing
    'RFSignalProcessor',
    'SignalStrengthCalculator',
    'SubcarrierSelector',
    
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
    

    
    # Version info
    '__version__',
    '__author__',
    '__email__'
]
