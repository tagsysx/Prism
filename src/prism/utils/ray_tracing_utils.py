"""
Ray tracing utility functions for the Prism system.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class RayTracingUtils:
    """Utility functions for ray tracing operations."""
    
    @staticmethod
    def calculate_ray_statistics(ray_results: List[Dict]) -> Dict:
        """Calculate statistics for ray tracing results."""
        if not ray_results:
            return {}
        
        total_rays = len(ray_results)
        signal_strengths = [result.get('signal_strength', 0.0) for result in ray_results]
        
        return {
            'total_rays': total_rays,
            'mean_signal_strength': np.mean(signal_strengths),
            'std_signal_strength': np.std(signal_strengths),
            'max_signal_strength': np.max(signal_strengths),
            'min_signal_strength': np.min(signal_strengths)
        }
    
    @staticmethod
    def optimize_direction_sampling(azimuth_divisions: int, elevation_divisions: int,
                                  target_efficiency: float = 0.3) -> int:
        """Calculate optimal number of directions to sample."""
        total_directions = azimuth_divisions * elevation_divisions
        optimal_directions = int(total_directions * target_efficiency)
        return max(1, optimal_directions)
    
    @staticmethod
    def validate_ray_parameters(azimuth_divisions: int, elevation_divisions: int,
                              max_ray_length: float) -> bool:
        """Validate ray tracing parameters."""
        if azimuth_divisions <= 0 or elevation_divisions <= 0:
            logger.error("Azimuth and elevation divisions must be positive")
            return False
        
        if max_ray_length <= 0:
            logger.error("Maximum ray length must be positive")
            return False
        
        return True
