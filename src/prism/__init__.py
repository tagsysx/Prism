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
    RayIntersection,
    RayPath,
    BaseStation,
    UserEquipment,
    Environment,
    VoxelGrid
)

from .mlp_direction_sampler import (
    MLPDirectionSampler,
    DirectionSamplingConfig
)

from .rf_signal_processor import (
    RFSignalProcessor,
    SignalStrengthCalculator,
    SubcarrierSelector
)

from .utils import (
    RayTracingUtils,
    SignalProcessingUtils,
    GeometryUtils
)

__all__ = [
    # Core ray tracing
    'DiscreteRayTracer',
    'Ray',
    'RayIntersection',
    'RayPath',
    'BaseStation',
    'UserEquipment',
    'Environment',
    'VoxelGrid',
    
    # MLP direction sampling
    'MLPDirectionSampler',
    'DirectionSamplingConfig',
    
    # RF signal processing
    'RFSignalProcessor',
    'SignalStrengthCalculator',
    'SubcarrierSelector',
    
    # Utilities
    'RayTracingUtils',
    'SignalProcessingUtils',
    'GeometryUtils',
    
    # Version info
    '__version__',
    '__author__',
    '__email__'
]
