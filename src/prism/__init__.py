"""
Prism: Wideband RF Neural Radiance Fields for OFDM Communication

A PyTorch-based implementation extending NeRF2 for wideband RF signals
in Orthogonal Frequency-Division Multiplexing (OFDM) scenarios.
"""

__version__ = "0.1.0"
__author__ = "Prism Project Team"
__email__ = "contact@prism-project.org"

from .model import (
    PrismModel,
    PrismLoss,
    RFPrismModule,
    AttenuationNetwork,
    RadianceNetwork,
    create_prism_model
)

from .dataloader import (
    PrismDataset,
    PrismDataLoader
)

from .renderer import PrismRenderer

# Advanced features
from .csi_processor import CSIVirtualLinkProcessor
from .ray_tracer import (
    AdvancedRayTracer, Environment, Building, Plane, 
    Ray, RayGenerator, PathTracer
)

__all__ = [
    # Core models
    'PrismModel',
    'PrismLoss',
    'RFPrismModule',
    'AttenuationNetwork',
    'RadianceNetwork',
    'create_prism_model',
    
    # Data handling
    'PrismDataset',
    'PrismDataLoader',
    
    # Visualization
    'PrismRenderer',
    
    # Advanced features
    'CSIVirtualLinkProcessor',
    'AdvancedRayTracer',
    'Environment',
    'Building',
    'Plane',
    'Ray',
    'RayGenerator',
    'PathTracer',
    
    # Version info
    '__version__',
    '__author__',
    '__email__'
]
