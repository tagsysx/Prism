"""
Loss Functions for Prism: Neural Network-Based Electromagnetic Ray Tracing

This module provides specialized loss functions for CSI (Channel State Information) 
and spatial spectrum estimation tasks, all supporting automatic differentiation.

Classes:
- PrismLossFunction: Main loss function class with CSI and PDP losses
- CSILoss: Specialized CSI loss functions
- PDPLoss: Power Delay Profile loss functions
- SpatialSpectrumLoss: Spatial spectrum loss functions

All loss functions are designed to work with PyTorch tensors and support backpropagation.
"""

from .csi_loss import CSILoss
from .pdp_loss import PDPLoss
from .ss_loss import SSLoss
from .loss_function import LossFunction, DEFAULT_LOSS_CONFIG

__all__ = [
    'CSILoss',
    'PDPLoss', 
    'SSLoss',
    'LossFunction',
    'DEFAULT_LOSS_CONFIG'
]
