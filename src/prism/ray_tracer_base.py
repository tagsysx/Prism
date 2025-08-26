"""
RayTracer Base Interface

Defines the common interface for all ray tracer implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
import torch
import logging
import math

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
                 azimuth_divisions: int = 18,
                 elevation_divisions: int = 9,
                 max_ray_length: float = 200.0,
                 scene_bounds: Optional[Dict[str, List[float]]] = None,
                 device: str = 'cpu',
                 prism_network = None,
                 signal_threshold: float = 1e-6,
                 enable_early_termination: bool = True,
                 top_k_directions: int = 32,
                 uniform_samples: int = 64,
                 resampled_points: int = 32,
                 # Backward compatibility
                 scene_size: Optional[float] = None):
        """
        Initialize the ray tracer with common parameters.
        
        Args:
            azimuth_divisions: Number of azimuth divisions (0° to 360°)
            elevation_divisions: Number of elevation divisions (-90° to +90°)
            max_ray_length: Maximum ray length in meters
            scene_bounds: Scene boundaries as {'min': [x,y,z], 'max': [x,y,z]}
            device: Device to run computations on
            prism_network: PrismNetwork instance for getting attenuation and radiance properties
            signal_threshold: Minimum signal strength threshold for early termination
            enable_early_termination: Enable early termination optimization
            top_k_directions: Number of top-K directions to select for MLP-based sampling
            uniform_samples: Number of uniform samples per ray
            resampled_points: Number of resampled points per ray
            scene_size: [DEPRECATED] Use scene_bounds instead
        """
        # Basic ray tracing parameters
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.max_ray_length = max_ray_length
        self.device = device
        self.uniform_samples = uniform_samples
        self.resampled_points = resampled_points
        
        # Neural network and signal processing parameters
        self.prism_network = prism_network
        self.signal_threshold = signal_threshold
        self.enable_early_termination = enable_early_termination
        self.top_k_directions = top_k_directions
        
        # Handle scene_bounds vs scene_size (backward compatibility)
        if scene_bounds is not None:
            self.scene_bounds = scene_bounds
            self.scene_min = torch.tensor(scene_bounds['min'], dtype=torch.float32)
            self.scene_max = torch.tensor(scene_bounds['max'], dtype=torch.float32)
            # Calculate equivalent scene_size for backward compatibility
            scene_dimensions = self.scene_max - self.scene_min
            self.scene_size = torch.max(scene_dimensions).item()
        elif scene_size is not None:
            # Convert scene_size to scene_bounds (deprecated path)
            logger.warning("scene_size parameter is deprecated, use scene_bounds instead")
            half_size = scene_size / 2.0
            self.scene_bounds = {
                'min': [-half_size, -half_size, -half_size],
                'max': [half_size, half_size, half_size]
            }
            self.scene_min = torch.tensor([-half_size, -half_size, -half_size], dtype=torch.float32)
            self.scene_max = torch.tensor([half_size, half_size, half_size], dtype=torch.float32)
            self.scene_size = scene_size
        else:
            # Default scene bounds from config
            self.scene_bounds = {
                'min': [-100.0, -100.0, 0.0],
                'max': [100.0, 100.0, 30.0]
            }
            self.scene_min = torch.tensor([-100.0, -100.0, 0.0], dtype=torch.float32)
            self.scene_max = torch.tensor([100.0, 100.0, 30.0], dtype=torch.float32)
            scene_dimensions = self.scene_max - self.scene_min
            self.scene_size = torch.max(scene_dimensions).item()
        
        # Calculate derived parameters
        self.azimuth_resolution = 2 * 3.14159 / azimuth_divisions  # 0° to 360°
        self.elevation_resolution = 3.14159 / elevation_divisions   # -90° to +90° (π range)
        self.total_directions = azimuth_divisions * elevation_divisions
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"RayTracer initialized: {azimuth_divisions}×{elevation_divisions} = {self.total_directions} directions")
        logger.info(f"Scene: {self.scene_size}m, Max ray length: {max_ray_length}m")
    
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
        
        # Check if all coordinates are within bounds using scene_bounds
        in_bounds = torch.all(
            (position >= self.scene_min) & (position <= self.scene_max), 
            dim=1
        )
        
        return in_bounds.all().item()
    
    def get_scene_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get scene boundaries."""
        return self.scene_min, self.scene_max
    
    def get_scene_size(self) -> float:
        """Get scene size."""
        return self.scene_size
    
    def update_scene_bounds(self, new_scene_bounds: Dict[str, List[float]]):
        """
        Update scene bounds and related parameters.
        
        Args:
            new_scene_bounds: New scene bounds as {'min': [x,y,z], 'max': [x,y,z]}
        """
        self.scene_bounds = new_scene_bounds
        self.scene_min = torch.tensor(new_scene_bounds['min'], dtype=torch.float32)
        self.scene_max = torch.tensor(new_scene_bounds['max'], dtype=torch.float32)
        
        # Update scene_size for backward compatibility
        scene_dimensions = self.scene_max - self.scene_min
        self.scene_size = torch.max(scene_dimensions).item()
        
        # Adjust max ray length if necessary
        scene_diagonal = torch.norm(scene_dimensions).item()
        if self.max_ray_length > scene_diagonal:
            self.max_ray_length = scene_diagonal
            logger.info(f"Adjusted max ray length to {self.max_ray_length}m")
        
        logger.info(f"Updated scene bounds: min={new_scene_bounds['min']}, max={new_scene_bounds['max']}")
    
    def get_scene_config(self) -> Dict:
        """Get complete scene configuration."""
        return {
            'scene_bounds': self.scene_bounds,
            'scene_min': self.scene_min.tolist() if hasattr(self.scene_min, 'tolist') else self.scene_min,
            'scene_max': self.scene_max.tolist() if hasattr(self.scene_max, 'tolist') else self.scene_max,
            'max_ray_length': self.max_ray_length,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions,
            'total_directions': self.total_directions,
            # Backward compatibility
            'scene_size': self.scene_size
        }
    
    def calculate_max_ray_length_from_bounds(self, scene_bounds: Dict[str, List[float]], margin: float = 1.2) -> float:
        """
        Calculate maximum ray length from scene bounds with safety margin.
        
        Args:
            scene_bounds: Scene bounds dictionary
            margin: Safety margin multiplier (default 1.2 = 20% margin)
            
        Returns:
            Calculated max ray length
        """
        if 'min' in scene_bounds and 'max' in scene_bounds:
            import numpy as np
            min_bounds = np.array(scene_bounds['min'])
            max_bounds = np.array(scene_bounds['max'])
            # Calculate diagonal distance of the scene
            diagonal = np.linalg.norm(max_bounds - min_bounds)
            # Add safety margin
            return diagonal * margin
        else:
            # Fallback to default if scene bounds not properly configured
            return 200.0
    
    def generate_direction_vectors(self) -> torch.Tensor:
        """
        Generate unit direction vectors for all azimuth×elevation directions.
        
        Returns:
            Tensor of shape [total_directions, 3] containing unit direction vectors
        """
        directions = []
        
        for phi_idx in range(self.azimuth_divisions):
            for theta_idx in range(self.elevation_divisions):
                # Convert indices to angles
                phi = phi_idx * self.azimuth_resolution  # 0 to 2π
                theta = (theta_idx * self.elevation_resolution) - (math.pi / 2)  # -π/2 to π/2
                
                # Convert spherical to Cartesian coordinates
                x = math.cos(theta) * math.cos(phi)
                y = math.cos(theta) * math.sin(phi)
                z = math.sin(theta)
                
                directions.append([x, y, z])
        
        return torch.tensor(directions, dtype=torch.float32, device=self.device)
    
    def get_ray_tracer_stats(self) -> Dict:
        """
        Get basic ray tracer statistics.
        
        Returns:
            Dictionary with ray tracer statistics
        """
        return {
            'device': self.device,
            'total_directions': self.total_directions,
            'azimuth_divisions': self.azimuth_divisions,
            'elevation_divisions': self.elevation_divisions,
            'top_k_directions': self.top_k_directions,
            'max_ray_length': self.max_ray_length,
            'uniform_samples': self.uniform_samples,
            'resampled_points': self.resampled_points,
            'signal_threshold': self.signal_threshold,
            'enable_early_termination': self.enable_early_termination,
            'scene_bounds': self.scene_bounds,
            'scene_size': self.scene_size
        }
    
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
                # Azimuth: 0° to 360° (0 to 2π)
                phi = phi_idx * self.azimuth_resolution
                # Elevation: -90° to +90° (-π/2 to +π/2)
                theta = (theta_idx * self.elevation_resolution) - (3.14159 / 2)
                
                # Calculate direction vector using spherical coordinates
                # x = cos(elevation) * cos(azimuth)
                # y = cos(elevation) * sin(azimuth)  
                # z = sin(elevation)
                x = math.cos(theta) * math.cos(phi)
                y = math.cos(theta) * math.sin(phi)
                z = math.sin(theta)
                
                direction_vectors.append([x, y, z])
        
        return torch.tensor(direction_vectors, dtype=torch.float32, device=self.device)
    

    
    def _ensure_complex_accumulation(self, accumulated_signals: Dict, key: tuple, signal_strength):
        """Helper function to ensure proper complex signal accumulation."""
        if key not in accumulated_signals:
            # Initialize with complex zero
            accumulated_signals[key] = torch.tensor(0.0 + 0.0j, dtype=torch.complex64)
        
        # Convert signal_strength to tensor if it's a scalar
        if not isinstance(signal_strength, torch.Tensor):
            signal_strength = torch.tensor(signal_strength, dtype=torch.complex64)
        
        # Ensure signal_strength is complex
        if not torch.is_complex(signal_strength):
            signal_strength = torch.complex(signal_strength, torch.tensor(0.0))
        
        accumulated_signals[key] += signal_strength
    

    

    
    def _compute_view_directions(self, 
                               sampled_positions: torch.Tensor,
                               bs_position: torch.Tensor) -> torch.Tensor:
        """
        Compute view directions from sampled positions to BS antenna.
        
        In wireless communication, the view direction represents the direction from
        each sampling point to the BS antenna (transmitter). This is used by the
        RadianceNetwork to compute directional radiation patterns.
        
        Args:
            sampled_positions: Sampled positions along the ray (num_samples, 3)
            bs_position: BS antenna position (3,) or (1, 3)
        
        Returns:
            View directions from sampled positions to BS (num_samples, 3)
        """
        # Ensure bs_position has correct shape
        if bs_position.dim() == 1:
            bs_position = bs_position.unsqueeze(0)  # (1, 3)
        
        # Compute directions from sampled positions to BS antenna
        # Direction: sampled_position -> BS_antenna
        view_directions = bs_position - sampled_positions  # (num_samples, 3)
        
        # Normalize directions
        view_directions = view_directions / (torch.norm(view_directions, dim=1, keepdim=True) + 1e-8)
        
        return view_directions
    
