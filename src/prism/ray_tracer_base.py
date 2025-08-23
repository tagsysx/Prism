"""
RayTracer Base Interface

Defines the common interface for all ray tracer implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class Ray:
    """Represents a single ray for ray tracing."""
    
    def __init__(self, origin: torch.Tensor, direction: torch.Tensor, max_length: float = 100.0, device: str = 'cpu'):
        """
        Initialize a ray.
        
        Args:
            origin: Ray origin point [3]
            direction: Ray direction vector [3]
            max_length: Maximum ray length
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.device = device
        self.origin = origin.clone().detach().to(dtype=torch.float32, device=device)
        self.direction = self._normalize(direction.clone().detach().to(dtype=torch.float32, device=device))
        self.max_length = max_length
    
    def _normalize(self, vector: torch.Tensor) -> torch.Tensor:
        """Normalize direction vector."""
        norm = torch.norm(vector)
        if norm < 1e-10:
            return vector
        return vector / norm

class RayTracer(ABC):
    """
    Abstract base class for ray tracer implementations.
    
    This interface defines the common methods that all ray tracer implementations
    must provide, ensuring consistency across different backends (CPU, CUDA, Hybrid).
    """
    
    def __init__(self, 
                 azimuth_divisions: int,
                 elevation_divisions: int,
                 max_ray_length: float,
                 scene_size: float,
                 device: str = 'cpu',
                 uniform_samples: int = 128,
                 resampled_points: int = 64):
        """
        Initialize the ray tracer with common parameters.
        
        Args:
            azimuth_divisions: Number of azimuth divisions (0° to 360°)
            elevation_divisions: Number of elevation divisions (-90° to +90°)
            max_ray_length: Maximum ray length in meters
            scene_size: Scene size in meters (cubic environment)
            device: Device to run computations on
            uniform_samples: Number of uniform samples per ray
            resampled_points: Number of resampled points per ray
        """
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.max_ray_length = max_ray_length
        self.scene_size = scene_size
        self.device = device
        self.uniform_samples = uniform_samples
        self.resampled_points = resampled_points
        
        # Calculate derived parameters
        self.azimuth_resolution = 2 * 3.14159 / azimuth_divisions
        self.elevation_resolution = 3.14159 / elevation_divisions
        self.total_directions = azimuth_divisions * elevation_divisions
        
        # Scene boundaries
        self.scene_min = -scene_size / 2.0
        self.scene_max = scene_size / 2.0
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"RayTracer initialized: {azimuth_divisions}×{elevation_divisions} = {self.total_directions} directions")
        logger.info(f"Scene: {scene_size}m, Max ray length: {max_ray_length}m")
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.scene_size <= 0:
            raise ValueError(f"Scene size must be positive, got {self.scene_size}")
        
        if self.max_ray_length > self.scene_size:
            logger.warning(f"Max ray length ({self.max_ray_length}m) exceeds scene size ({self.scene_size}m)")
            self.max_ray_length = min(self.max_ray_length, self.scene_size)
            logger.info(f"Adjusted max ray length to {self.max_ray_length}m")
        
        if self.azimuth_divisions <= 0 or self.elevation_divisions <= 0:
            raise ValueError("Azimuth and elevation divisions must be positive")
    
    @abstractmethod
    def trace_ray(self, 
                  base_station_pos: torch.Tensor,
                  direction: Tuple[int, int],
                  ue_positions: List[torch.Tensor],
                  selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                  antenna_embedding: torch.Tensor) -> Dict:
        """
        Trace RF signal along a single ray direction.
        
        Args:
            base_station_pos: Base station position
            direction: Direction indices (phi_idx, theta_idx)
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information
            antenna_embedding: Base station's antenna embedding parameter C
        
        Returns:
            Dictionary mapping (ue_pos, subcarrier) to signal strength
        """
        pass
    
    @abstractmethod
    def trace_rays(self, 
                   base_station_pos: torch.Tensor,
                   directions: torch.Tensor,
                   ue_positions: List[torch.Tensor],
                   selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                   antenna_embedding: torch.Tensor) -> Dict:
        """
        Trace RF signals along multiple ray directions.
        
        Args:
            base_station_pos: Base station position
            directions: Direction vectors [num_directions, 3]
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information
            antenna_embedding: Base station's antenna embedding parameter C
        
        Returns:
            Dictionary mapping (ue_pos, subcarrier, direction_idx) to signal strength
        """
        pass
    
    @abstractmethod
    def accumulate_signals(self, 
                          base_station_pos: torch.Tensor,
                          ue_positions: List[torch.Tensor],
                          selected_subcarriers: Union[Dict, torch.Tensor, List[int]],
                          antenna_embedding: torch.Tensor) -> Dict:
        """
        Accumulate signals from all directions using MLP-based direction selection.
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Subcarrier information
            antenna_embedding: Base station's antenna embedding parameter C
        
        Returns:
            Accumulated signal strength matrix for all virtual links
        """
        pass
    
    def is_position_in_scene(self, position: torch.Tensor) -> bool:
        """
        Check if a position is within the scene boundaries.
        
        Args:
            position: 3D position tensor [x, y, z]
        
        Returns:
            True if position is within scene boundaries
        """
        if position.dim() == 1:
            position = position.unsqueeze(0)
        
        # Check if all coordinates are within bounds
        scene_bounds = self.scene_size / 2.0
        in_bounds = torch.all(
            (position >= -scene_bounds) & (position <= scene_bounds), 
            dim=1
        )
        
        return in_bounds.all().item()
    
    def get_scene_bounds(self) -> Tuple[float, float]:
        """Get scene boundaries."""
        scene_bounds = self.scene_size / 2.0
        return -scene_bounds, scene_bounds
    
    def get_scene_size(self) -> float:
        """Get scene size."""
        return self.scene_size
    
    def update_scene_size(self, new_scene_size: float):
        """
        Update scene size and related parameters.
        
        Args:
            new_scene_size: New scene size in meters
        """
        if new_scene_size <= 0:
            raise ValueError(f"Scene size must be positive, got {new_scene_size}")
        
        self.scene_size = new_scene_size
        
        # Adjust max ray length if necessary
        if self.max_ray_length > new_scene_size:
            self.max_ray_length = new_scene_size
            logger.info(f"Adjusted max ray length to {self.max_ray_length}m")
        
        # Update scene boundaries
        self.scene_min = -new_scene_size / 2.0
        self.scene_max = new_scene_size / 2.0
        
        logger.info(f"Updated scene size to {new_scene_size}m")
    
    def get_scene_config(self) -> Dict[str, float]:
        """Get complete scene configuration."""
        scene_bounds = self.scene_size / 2.0
        return {
            'scene_size': self.scene_size,
            'scene_min': -scene_bounds,
            'scene_max': scene_bounds,
            'max_ray_length': self.max_ray_length,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions
        }
    
    def get_performance_info(self) -> Dict:
        """Get performance information and device capabilities."""
        info = {
            'device': self.device,
            'total_directions': self.total_directions,
            'uniform_samples': self.uniform_samples,
            'resampled_points': self.resampled_points,
            'implementation': self.__class__.__name__
        }
        
        return info
    
    def generate_direction_vectors(self) -> torch.Tensor:
        """
        Generate direction vectors for all azimuth and elevation combinations.
        
        Returns:
            Direction vectors tensor [total_directions, 3]
        """
        import math
        
        direction_vectors = []
        for phi_idx in range(self.azimuth_divisions):
            for theta_idx in range(self.elevation_divisions):
                phi = phi_idx * self.azimuth_resolution
                theta = theta_idx * self.elevation_resolution
                
                # Calculate direction vector
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)
                
                direction_vectors.append([x, y, z])
        
        return torch.tensor(direction_vectors, dtype=torch.float32, device=self.device)
