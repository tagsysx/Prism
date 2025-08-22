"""
Geometry utility functions for the Prism system.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
import logging
import math

logger = logging.getLogger(__name__)

class GeometryUtils:
    """Utility functions for geometric operations."""
    
    @staticmethod
    def spherical_to_cartesian(azimuth: float, elevation: float) -> Tuple[float, float, float]:
        """Convert spherical coordinates to Cartesian coordinates."""
        x = math.sin(elevation) * math.cos(azimuth)
        y = math.sin(elevation) * math.sin(azimuth)
        z = math.cos(elevation)
        return x, y, z
    
    @staticmethod
    def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float]:
        """Convert Cartesian coordinates to spherical coordinates."""
        azimuth = math.atan2(y, x)
        elevation = math.acos(z / math.sqrt(x*x + y*y + z*z))
        return azimuth, elevation
    
    @staticmethod
    def calculate_distance(point1: torch.Tensor, point2: torch.Tensor) -> float:
        """Calculate Euclidean distance between two points."""
        return torch.norm(point2 - point1).item()
    
    @staticmethod
    def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
        """Normalize a vector to unit length."""
        norm = torch.norm(vector)
        if norm < 1e-10:
            return vector
        return vector / norm
