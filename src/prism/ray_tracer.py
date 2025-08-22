"""
Discrete Electromagnetic Ray Tracing System for Prism

This module implements the discrete electromagnetic ray tracing system as described
in the design document, with support for MLP-based direction sampling and
efficient RF signal strength computation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class RayIntersection:
    """Data class for ray intersection information."""
    point: torch.Tensor
    distance: float
    normal: torch.Tensor
    material: str
    interaction_type: str

@dataclass
class RayPath:
    """Data class for complete ray path information."""
    origin: torch.Tensor
    direction: torch.Tensor
    path_points: List[torch.Tensor]
    interactions: List[str]
    materials: List[str]
    total_length: float
    final_point: torch.Tensor

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
        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)
        self.direction = self._normalize(torch.tensor(direction, dtype=torch.float32, device=device))
        self.max_length = max_length
        self.path_points = [self.origin.clone()]
        self.interactions = []
        self.materials = []
        self.current_length = 0.0
    
    def _normalize(self, vector: torch.Tensor) -> torch.Tensor:
        """Normalize direction vector."""
        norm = torch.norm(vector)
        if norm < 1e-10:
            return vector
        return vector / norm
    
    def add_path_point(self, point: torch.Tensor, interaction_type: str = None, material: str = None):
        """Add a point along the ray path."""
        self.path_points.append(point.clone())
        if interaction_type:
            self.interactions.append(interaction_type)
        if material:
            self.materials.append(material)
        
        # Update current length
        if len(self.path_points) > 1:
            segment_length = torch.norm(self.path_points[-1] - self.path_points[-2])
            self.current_length += segment_length
    
    def get_spatial_samples(self, num_points: int = 64) -> torch.Tensor:
        """Generate spatial samples along the ray path."""
        if len(self.path_points) < 2:
            return torch.empty(0, 3, device=self.device)
        
        samples = []
        
        # Interpolate between path points
        for i in range(len(self.path_points) - 1):
            start = self.path_points[i]
            end = self.path_points[i + 1]
            
            # Generate samples between start and end
            segment_samples = torch.linspace(0, 1, num_points // len(self.path_points) + 1, device=self.device)
            for t in segment_samples:
                sample_point = start + t * (end - start)
                samples.append(sample_point)
        
        # Ensure we have exactly num_points samples
        if len(samples) > num_points:
            samples = samples[:num_points]
        elif len(samples) < num_points:
            # Pad with the last point if needed
            while len(samples) < num_points:
                samples.append(samples[-1] if samples else self.origin)
        
        return torch.stack(samples)

class BaseStation:
    """Represents a base station with configurable location and antennas."""
    
    def __init__(self, position: torch.Tensor = None, num_antennas: int = 1, device: str = 'cpu'):
        """
        Initialize base station.
        
        Args:
            position: Base station position [3], defaults to origin (0, 0, 0)
            num_antennas: Number of antennas at this base station
            device: Device to run computations on
        """
        self.device = device
        self.position = torch.tensor([0.0, 0.0, 0.0], device=device) if position is None else torch.tensor(position, device=device)
        self.num_antennas = num_antennas
        self.antenna_embeddings = torch.randn(num_antennas, 128, device=device)  # 128D antenna embedding
    
    def get_antenna_embedding(self, antenna_idx: int = 0) -> torch.Tensor:
        """Get antenna embedding parameter C for the specified antenna."""
        return self.antenna_embeddings[antenna_idx]

class UserEquipment:
    """Represents user equipment at a specific location."""
    
    def __init__(self, position: torch.Tensor, device: str = 'cpu'):
        """
        Initialize user equipment.
        
        Args:
            position: UE position [3]
            device: Device to run computations on
        """
        self.device = device
        self.position = torch.tensor(position, dtype=torch.float32, device=device)

class VoxelGrid:
    """Represents a voxel grid for discrete radiance field modeling."""
    
    def __init__(self, grid_size: Tuple[int, int, int], voxel_size: float, device: str = 'cpu'):
        """
        Initialize voxel grid.
        
        Args:
            grid_size: Grid dimensions (nx, ny, nz)
            voxel_size: Size of each voxel in meters
            device: Device to run computations on
        """
        self.device = device
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.materials = torch.randint(0, 5, grid_size, device=device)  # 5 material types
        self.attenuation_coeffs = torch.rand(grid_size, device=device)  # Random attenuation coefficients
        
    def get_voxel_properties(self, position: torch.Tensor) -> Tuple[int, float]:
        """Get material type and attenuation coefficient for a given position."""
        # Convert position to voxel indices
        indices = ((position / self.voxel_size) + torch.tensor(self.grid_size, device=self.device) / 2).long()
        indices = torch.clamp(indices, 0, torch.tensor(self.grid_size, device=self.device) - 1)
        
        material = self.materials[indices[0], indices[1], indices[2]]
        attenuation = self.attenuation_coeffs[indices[0], indices[1], indices[2]]
        
        return material.item(), attenuation.item()

class Environment:
    """Represents the environment with buildings and obstacles."""
    
    def __init__(self, voxel_grid: VoxelGrid, device: str = 'cpu'):
        """
        Initialize environment.
        
        Args:
            voxel_grid: Voxel grid for the environment
            device: Device to run computations on
        """
        self.device = device
        self.voxel_grid = voxel_grid
        self.buildings = []
    
    def add_building(self, building):
        """Add a building to the environment."""
        self.buildings.append(building)
    
    def get_material_at_position(self, position: torch.Tensor) -> Tuple[int, float]:
        """Get material properties at a given position."""
        return self.voxel_grid.get_voxel_properties(position)

class DiscreteRayTracer:
    """Discrete electromagnetic ray tracer implementing the design document specifications."""
    
    def __init__(self, 
                 azimuth_divisions: int = 36,
                 elevation_divisions: int = 18,
                 max_ray_length: float = 100.0,
                 device: str = 'cpu'):
        """
        Initialize discrete ray tracer.
        
        Args:
            azimuth_divisions: Number of azimuth divisions A (0° to 360°)
            elevation_divisions: Number of elevation divisions B (-90° to +90°)
            max_ray_length: Maximum ray length in meters
            device: Device to run computations on
        """
        self.device = device
        self.azimuth_divisions = azimuth_divisions
        self.elevation_divisions = elevation_divisions
        self.max_ray_length = max_ray_length
        
        # Calculate angular resolutions
        self.azimuth_resolution = 2 * math.pi / azimuth_divisions
        self.elevation_resolution = math.pi / elevation_divisions
        
        # Total number of directions
        self.total_directions = azimuth_divisions * elevation_divisions
        
        logger.info(f"Initialized ray tracer with {azimuth_divisions}x{elevation_divisions} = {self.total_directions} directions")
    
    def generate_direction_vectors(self) -> torch.Tensor:
        """Generate unit direction vectors for all A×B directions."""
        directions = []
        
        for i in range(self.azimuth_divisions):
            for j in range(self.elevation_divisions):
                phi = i * self.azimuth_resolution  # Azimuth angle
                theta = j * self.elevation_resolution  # Elevation angle
                
                # Convert to Cartesian coordinates
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)
                
                directions.append([x, y, z])
        
        return torch.tensor(directions, dtype=torch.float32, device=self.device)
    
    def trace_ray(self, 
                  base_station_pos: torch.Tensor,
                  direction: Tuple[int, int],
                  ue_positions: List[torch.Tensor],
                  selected_subcarriers: Dict,
                  antenna_embedding: torch.Tensor) -> Dict:
        """
        Trace RF signal along a single ray direction.
        
        Args:
            base_station_pos: Base station position P_BS
            direction: Direction indices (phi_idx, theta_idx)
            ue_positions: List of UE positions
            selected_subcarriers: Dictionary mapping UE to selected subcarriers
            antenna_embedding: Base station's antenna embedding parameter C
        
        Returns:
            Dictionary mapping (ue_pos, subcarrier) to received RF signal strength
        """
        phi_idx, theta_idx = direction
        
        # Convert indices to angles
        phi = phi_idx * self.azimuth_resolution
        theta = theta_idx * self.elevation_resolution
        
        # Create direction vector
        direction_vector = torch.tensor([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta)
        ], dtype=torch.float32, device=self.device)
        
        # Create ray
        ray = Ray(base_station_pos, direction_vector, self.max_ray_length, self.device)
        
        results = {}
        
        for ue_pos in ue_positions:
            ue_pos_tensor = torch.tensor(ue_pos, dtype=torch.float32, device=self.device)
            
            for subcarrier_idx in selected_subcarriers:
                # Apply importance-based sampling along ray with antenna embedding
                signal_strength = self._importance_based_ray_tracing(
                    ray, ue_pos_tensor, subcarrier_idx, antenna_embedding
                )
                results[(tuple(ue_pos), subcarrier_idx)] = signal_strength
        
        return results
    
    def _importance_based_ray_tracing(self, 
                                    ray: Ray,
                                    ue_pos: torch.Tensor,
                                    subcarrier_idx: int,
                                    antenna_embedding: torch.Tensor) -> float:
        """
        Apply importance-based sampling for ray tracing.
        
        Args:
            ray: Ray object
            ue_pos: UE position
            subcarrier_idx: Subcarrier index
            antenna_embedding: Antenna embedding parameter
        
        Returns:
            Computed signal strength
        """
        # Simple importance-based sampling: more samples near high-attenuation regions
        num_samples = 64
        samples = ray.get_spatial_samples(num_samples)
        
        if len(samples) == 0:
            return 0.0
        
        # Calculate distance-based attenuation
        distances = torch.norm(samples - ray.origin, dim=1)
        
        # Simple exponential decay model
        base_attenuation = 1.0 / (1.0 + distances)
        
        # Apply antenna embedding influence (simplified)
        antenna_factor = torch.norm(antenna_embedding) / math.sqrt(128)  # Normalize to [0, 1]
        
        # Combine factors
        signal_strength = torch.mean(base_attenuation) * antenna_factor
        
        return signal_strength.item()
    
    def accumulate_signals(self, 
                          base_station_pos: torch.Tensor,
                          ue_positions: List[torch.Tensor],
                          selected_subcarriers: Dict,
                          antenna_embedding: torch.Tensor) -> Dict:
        """
        Accumulate RF signals from all directions for all UEs and selected subcarriers.
        
        Args:
            base_station_pos: Base station position
            ue_positions: List of UE positions
            selected_subcarriers: Dictionary mapping UE to selected subcarriers
            antenna_embedding: Base station's antenna embedding parameter C
        
        Returns:
            Accumulated signal strength matrix for all virtual links
        """
        accumulated_signals = {}
        
        # Iterate through all A × B directions
        for phi in range(self.azimuth_divisions):
            for theta in range(self.elevation_divisions):
                direction = (phi, theta)
                
                # Trace ray for this direction with antenna embedding
                ray_results = self.trace_ray(
                    base_station_pos, direction, ue_positions, selected_subcarriers, antenna_embedding
                )
                
                # Accumulate signals for each virtual link
                for (ue_pos, subcarrier), signal_strength in ray_results.items():
                    if (ue_pos, subcarrier) not in accumulated_signals:
                        accumulated_signals[(ue_pos, subcarrier)] = 0.0
                    accumulated_signals[(ue_pos, subcarrier)] += signal_strength
        
        return accumulated_signals
    
    def adaptive_ray_tracing(self, 
                           base_station_pos: torch.Tensor,
                           antenna_embedding: torch.Tensor,
                           ue_positions: List[torch.Tensor],
                           selected_subcarriers: Dict,
                           mlp_model) -> Dict:
        """
        Perform ray tracing only on MLP-selected directions.
        
        Args:
            base_station_pos: Base station position
            antenna_embedding: Base station's antenna embedding parameter C
            ue_positions: List of UE positions
            selected_subcarriers: Dictionary mapping UE to selected subcarriers
            mlp_model: Trained MLP model for direction sampling
        
        Returns:
            Accumulated signal strength for selected directions only
        """
        # Get direction indicators from MLP
        direction_indicators = self._mlp_direction_sampling(antenna_embedding, mlp_model)
        
        accumulated_signals = {}
        
        # Only trace rays for directions indicated by MLP
        for phi in range(self.azimuth_divisions):
            for theta in range(self.elevation_divisions):
                if direction_indicators[phi, theta] == 1:
                    direction = (phi, theta)
                    
                    # Trace ray for this selected direction
                    ray_results = self.trace_ray(
                        base_station_pos, direction, ue_positions, 
                        selected_subcarriers, antenna_embedding
                    )
                    
                    # Accumulate signals
                    for (ue_pos, subcarrier), signal_strength in ray_results.items():
                        if (ue_pos, subcarrier) not in accumulated_signals:
                            accumulated_signals[(ue_pos, subcarrier)] = 0.0
                        accumulated_signals[(ue_pos, subcarrier)] += signal_strength
        
        return accumulated_signals
    
    def _mlp_direction_sampling(self, antenna_embedding: torch.Tensor, mlp_model) -> torch.Tensor:
        """
        Use trained MLP to determine which directions to trace.
        
        Args:
            antenna_embedding: Base station's antenna embedding parameter C
            mlp_model: Trained MLP model for direction sampling
        
        Returns:
            A x B binary indicator matrix M_ij
        """
        # Forward pass through MLP
        raw_output = mlp_model(antenna_embedding.unsqueeze(0))  # Add batch dimension
        
        # Reshape to A x B matrix
        output_matrix = raw_output.view(self.azimuth_divisions, self.elevation_divisions)
        
        # Apply sigmoid and threshold to get binary indicators
        threshold = 0.5
        indicator_matrix = (torch.sigmoid(output_matrix) > threshold).int()
        
        return indicator_matrix
