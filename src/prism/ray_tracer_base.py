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
    
    def _compute_importance_weights(self, attenuation_factors: torch.Tensor, delta_t: float = None) -> torch.Tensor:
        """
        Compute importance weights based on attenuation factors using the formula from SPECIFICATION.md 8.1.2:
        w_k = (1 - e^(-β_k * Δt)) * exp(-Σ_{j<k} β_j * Δt)
        
        Args:
            attenuation_factors: Complex attenuation factors from uniform sampling (num_samples,)
            delta_t: Step size along the ray (if None, use default based on max_ray_length)
        
        Returns:
            Importance weights for resampling (num_samples,)
        """
        if delta_t is None:
            delta_t = self.max_ray_length / len(attenuation_factors)
        
        # Extract real part β_k = Re(ρ(P_v(t_k))) for importance weight calculation
        beta_k = torch.real(attenuation_factors)  # (num_samples,)
        
        # Ensure non-negative values for physical validity
        beta_k = torch.clamp(beta_k, min=0.0)
        
        # Calculate importance weights using the correct formula:
        # w_k = (1 - e^(-β_k * Δt)) * exp(-Σ_{j<k} β_j * Δt)
        
        # Term 1: (1 - e^(-β_k * Δt)) - local absorption probability
        local_absorption = 1.0 - torch.exp(-beta_k * delta_t)  # (num_samples,)
        
        # Term 2: exp(-Σ_{j<k} β_j * Δt) - cumulative transmission up to point k
        # Calculate cumulative sum of β_j for j < k
        cumulative_beta = torch.cumsum(beta_k, dim=0)  # (num_samples,)
        # Shift to get sum for j < k (exclude current k)
        cumulative_beta_prev = torch.cat([torch.zeros(1, device=beta_k.device), cumulative_beta[:-1]])
        cumulative_transmission = torch.exp(-cumulative_beta_prev * delta_t)  # (num_samples,)
        
        # Combine terms: w_k = local_absorption * cumulative_transmission
        importance_weights = local_absorption * cumulative_transmission  # (num_samples,)
        
        # Add small epsilon to avoid zero weights and numerical issues
        importance_weights = importance_weights + 1e-8
        
        # Normalize weights to sum to 1 for proper probability distribution
        total_weight = torch.sum(importance_weights)
        if total_weight > 1e-8:
            importance_weights = importance_weights / total_weight
        else:
            # Fallback to uniform weights if all weights are near zero
            importance_weights = torch.ones_like(importance_weights) / len(importance_weights)
        
        return importance_weights
    
    def _importance_based_resampling(self, 
                                   uniform_positions: torch.Tensor,
                                   importance_weights: torch.Tensor,
                                   num_samples: int) -> torch.Tensor:
        """
        Perform importance-based resampling based on computed weights.
        
        Args:
            uniform_positions: Uniformly sampled positions (num_uniform_samples, 3)
            importance_weights: Importance weights for each position (num_uniform_samples,)
            num_samples: Number of samples to select
        
        Returns:
            Resampled positions based on importance (num_samples, 3)
        """
        num_uniform_samples = uniform_positions.shape[0]
        
        if num_samples >= num_uniform_samples:
            # If we want more samples than available, return all with repetition
            return uniform_positions
        
        # Use importance sampling to select positions
        # Higher weight positions have higher probability of being selected
        selected_indices = torch.multinomial(importance_weights, num_samples, replacement=True)
        
        # Get resampled positions
        resampled_positions = uniform_positions[selected_indices]
        
        return resampled_positions
    
    def _ensure_complex_accumulation(self, accumulated_signals: Dict, key: tuple, signal_strength: torch.Tensor):
        """Helper function to ensure proper complex signal accumulation."""
        if key not in accumulated_signals:
            # Initialize with complex zero
            accumulated_signals[key] = torch.tensor(0.0 + 0.0j, dtype=torch.complex64)
        
        # Ensure signal_strength is complex
        if not torch.is_complex(signal_strength):
            signal_strength = torch.complex(signal_strength, torch.tensor(0.0))
        
        accumulated_signals[key] += signal_strength
    
    def _sample_ray_points(self, ray: Ray, ue_pos: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Sample points along the ray for discrete radiance field computation.
        Ray tracing is from BS antenna (ray.origin) in the given direction.
        UE position is only used as input to RadianceNetwork, not for ray length calculation.
        
        Args:
            ray: Ray object (from BS antenna)
            ue_pos: UE position (used only for RadianceNetwork input)
            num_samples: Number of sample points
        
        Returns:
            Sampled positions along the ray
        """
        # Sample points along the ray from BS antenna up to max_ray_length
        # No need to consider UE position for ray length calculation
        ray_length = self.max_ray_length
        
        # Sample points along the ray from BS antenna
        t_values = torch.linspace(0, ray_length, num_samples, device=self.device)
        sampled_positions = ray.origin.unsqueeze(0) + t_values.unsqueeze(1) * ray.direction.unsqueeze(0)
        
        # Filter out points outside scene boundaries
        valid_mask = self.is_position_in_scene(sampled_positions)
        if not valid_mask.any():
            # If no valid positions, return empty tensor
            return torch.empty(0, 3, device=self.device)
        
        # Return only valid positions
        valid_positions = sampled_positions[valid_mask]
        
        # Ensure we have at least some samples
        if len(valid_positions) < num_samples // 2:
            logger.warning(f"Only {len(valid_positions)} valid positions out of {num_samples} requested")
        
        return valid_positions
    
    def _simple_distance_model(self, 
                              ray: Ray,
                              ue_pos: torch.Tensor,
                              subcarrier_idx: int,
                              antenna_embedding: torch.Tensor) -> float:
        """
        Simple distance-based model as fallback when neural network is not available.
        This model simulates ray tracing by sampling points along the ray and computing
        signal contributions from sampling points to BS antenna.
        
        Args:
            ray: Ray object (from BS antenna)
            ue_pos: UE position (used for radiation direction calculation)
            subcarrier_idx: Subcarrier index
            antenna_embedding: Antenna embedding parameter
        
        Returns:
            Computed signal strength using simple model
        """
        # Sample points along the ray from BS antenna
        num_samples = 32  # Simple model uses fewer samples
        ray_length = self.max_ray_length
        t_values = torch.linspace(0, ray_length, num_samples, device=self.device)
        sampled_positions = ray.origin.unsqueeze(0) + t_values.unsqueeze(1) * ray.direction.unsqueeze(0)
        
        # Filter points within scene boundaries
        valid_mask = self.is_position_in_scene(sampled_positions)
        if not valid_mask.any():
            return 0.0
        
        valid_positions = sampled_positions[valid_mask]
        valid_t_values = t_values[valid_mask]
        
        # Calculate signal contributions from each sampling point
        total_signal = 0.0
        step_size = ray_length / num_samples
        cumulative_attenuation = 1.0
        
        for i, (pos, t) in enumerate(zip(valid_positions, valid_t_values)):
            # Distance from sampling point to BS antenna (ray origin)
            distance_to_bs = t.item()  # Distance along ray from BS antenna
            
            # Distance from sampling point to UE (for radiation calculation)
            distance_to_ue = torch.norm(pos - ue_pos).item()
            
            # Local attenuation based on distance from BS antenna
            local_attenuation = 0.1 * torch.exp(-distance_to_bs / 30.0)  # Attenuation coefficient
            
            # Radiation factor based on distance to UE (closer UE = stronger radiation)
            radiation_factor = torch.exp(-distance_to_ue / 40.0)  # Radiation strength
            
            # Apply antenna embedding influence
            antenna_factor = torch.norm(antenna_embedding) / math.sqrt(128)  # Normalize to [0, 1]
            
            # Apply frequency-dependent effects
            frequency_factor = 1.0 / (1.0 + 0.1 * subcarrier_idx)
            
            # Local signal contribution: (1 - e^(-ρΔt)) × S
            local_absorption = 1.0 - torch.exp(-local_attenuation * step_size)
            local_contribution = local_absorption * radiation_factor * antenna_factor * frequency_factor
            
            # Apply cumulative attenuation and accumulate
            total_signal += cumulative_attenuation * local_contribution
            
            # Update cumulative attenuation for next sample
            cumulative_attenuation *= torch.exp(-local_attenuation * step_size)
            
            # Early termination if signal becomes negligible
            if cumulative_attenuation < 1e-6:
                break
        
        return float(total_signal)