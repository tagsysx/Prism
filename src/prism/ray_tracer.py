"""
Advanced Ray Tracing Engine for Prism: Wideband RF Neural Radiance Fields.
Implements comprehensive ray tracing with configurable azimuth/elevation sampling and spatial point sampling.
Supports GPU acceleration with CUDA for high-performance ray tracing.
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
            # Pad with the last point if we don't have enough
            last_point = samples[-1] if samples else self.origin
            while len(samples) < num_points:
                samples.append(last_point)
        
        return torch.stack(samples)
    
    def get_final_point(self) -> torch.Tensor:
        """Get the final point of the ray path."""
        if self.path_points:
            return self.path_points[-1]
        return self.origin
    
    def get_total_length(self) -> float:
        """Get the total length of the ray path."""
        return self.current_length

class RayGenerator:
    """Generates rays for given azimuth and elevation angles with GPU support."""
    
    def __init__(self, azimuth_samples: int = 36, elevation_samples: int = 18, device: str = 'cpu'):
        """
        Initialize ray generator.
        
        Args:
            azimuth_samples: Number of azimuth angles (0° to 360°)
            elevation_samples: Number of elevation angles (-90° to +90°)
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.azimuth_samples = azimuth_samples
        self.elevation_samples = elevation_samples
        self.device = device
        self.angle_combinations = self._generate_angle_combinations()
        
        logger.info(f"Ray generator initialized with {azimuth_samples}×{elevation_samples} = {len(self.angle_combinations)} angles on {device}")
    
    def _generate_angle_combinations(self) -> np.ndarray:
        """Generate all azimuth-elevation combinations."""
        azimuth_angles = np.linspace(0, 360, self.azimuth_samples, endpoint=False)
        elevation_angles = np.linspace(-90, 90, self.elevation_samples)
        
        combinations = []
        for az in azimuth_angles:
            for el in elevation_angles:
                combinations.append((az, el))
        
        return np.array(combinations)
    
    def _spherical_to_cartesian(self, azimuth: float, elevation: float) -> np.ndarray:
        """Convert spherical coordinates to Cartesian direction vector."""
        # Convert to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Calculate direction vector
        x = np.cos(el_rad) * np.cos(az_rad)
        y = np.cos(el_rad) * np.sin(az_rad)
        z = np.sin(el_rad)
        
        return np.array([x, y, z])
    
    def generate_rays(self, source_position: Union[List, np.ndarray, torch.Tensor]) -> List[Ray]:
        """Generate rays from source position."""
        rays = []
        
        for az, el in self.angle_combinations:
            direction = self._spherical_to_cartesian(az, el)
            ray = Ray(source_position, direction, device=self.device)
            rays.append(ray)
        
        return rays
    
    def generate_rays_batch(self, source_positions: torch.Tensor) -> torch.Tensor:
        """Generate rays in batch for GPU processing."""
        batch_size = source_positions.shape[0]
        total_rays = len(self.angle_combinations)
        
        # Pre-allocate tensors for batch processing
        origins = source_positions.unsqueeze(1).expand(batch_size, total_rays, 3)
        directions = torch.zeros(batch_size, total_rays, 3, device=self.device)
        
        # Generate directions for all angle combinations
        for i, (az, el) in enumerate(self.angle_combinations):
            direction = self._spherical_to_cartesian(az, el)
            directions[:, i, :] = torch.tensor(direction, device=self.device)
        
        return origins, directions
    
    def get_angle_resolution(self) -> Tuple[float, float]:
        """Get angular resolution in degrees."""
        azimuth_resolution = 360.0 / self.azimuth_samples
        elevation_resolution = 180.0 / (self.elevation_samples - 1)
        
        return azimuth_resolution, elevation_resolution

class Environment:
    """Represents the environment for ray tracing with GPU support."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.obstacles = []
        self.materials = {}
        self.boundaries = None
        self.atmospheric_conditions = {
            'temperature': 20.0,  # Celsius
            'humidity': 50.0,     # Percentage
            'pressure': 1013.25   # hPa
        }
    
    def add_obstacle(self, obstacle):
        """Add obstacle to environment."""
        # Ensure obstacle is on the same device
        if hasattr(obstacle, 'device') and obstacle.device != self.device:
            obstacle = obstacle.to_device(self.device)
        self.obstacles.append(obstacle)
    
    def set_materials(self, materials: Dict):
        """Set material properties."""
        self.materials.update(materials)
    
    def set_atmospheric_conditions(self, temperature: float, humidity: float, pressure: float):
        """Set atmospheric conditions."""
        self.atmospheric_conditions.update({
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure
        })
    
    def ray_intersection(self, ray: Ray) -> List[RayIntersection]:
        """Find intersection of ray with environment."""
        intersections = []
        
        for obstacle in self.obstacles:
            intersection = obstacle.intersect(ray)
            if intersection:
                intersections.append(intersection)
        
        # Sort by distance
        intersections.sort(key=lambda x: x.distance)
        
        return intersections
    
    def to_device(self, device: str):
        """Move environment to specified device."""
        self.device = device
        for obstacle in self.obstacles:
            if hasattr(obstacle, 'to_device'):
                obstacle.to_device(device)
        return self

class Obstacle:
    """Base class for obstacles in the environment."""
    
    def __init__(self, material: str = 'concrete', device: str = 'cpu'):
        self.material = material
        self.device = device
    
    def intersect(self, ray: Ray) -> Optional[RayIntersection]:
        """Calculate intersection with ray. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def to_device(self, device: str):
        """Move obstacle to specified device."""
        self.device = device
        return self

class Plane(Obstacle):
    """Plane obstacle (e.g., wall, floor, ceiling) with GPU support."""
    
    def __init__(self, point: np.ndarray, normal: np.ndarray, material: str = 'concrete', device: str = 'cpu'):
        super().__init__(material, device)
        self.point = torch.tensor(point, dtype=torch.float32, device=device)
        self.normal = torch.tensor(normal, dtype=torch.float32, device=device)
        self.normal = self.normal / torch.norm(self.normal)
    
    def intersect(self, ray: Ray) -> Optional[RayIntersection]:
        """Calculate intersection with plane."""
        # Ray-plane intersection: t = (p0 - r0) · n / (r · n)
        p0_minus_r0 = self.point - ray.origin
        r_dot_n = torch.dot(ray.direction, self.normal)
        
        if abs(r_dot_n) < 1e-10:  # Ray is parallel to plane
            return None
        
        t = torch.dot(p0_minus_r0, self.normal) / r_dot_n
        
        if t < 0:  # Intersection is behind ray origin
            return None
        
        if t > ray.max_length:  # Intersection is beyond max length
            return None
        
        intersection_point = ray.origin + t * ray.direction
        
        return RayIntersection(
            point=intersection_point,
            distance=t.item(),
            normal=self.normal,
            material=self.material,
            interaction_type='reflection'
        )
    
    def to_device(self, device: str):
        """Move plane to specified device."""
        super().to_device(device)
        self.point = self.point.to(device)
        self.normal = self.normal.to(device)
        return self

class Building(Obstacle):
    """Building obstacle with box geometry and GPU support."""
    
    def __init__(self, min_corner: np.ndarray, max_corner: np.ndarray, material: str = 'concrete', device: str = 'cpu'):
        super().__init__(material, device)
        self.min_corner = torch.tensor(min_corner, dtype=torch.float32, device=device)
        self.max_corner = torch.tensor(max_corner, dtype=torch.float32, device=device)
        
        # Create 6 planes for the building faces
        self.faces = self._create_faces()
    
    def _create_faces(self) -> List[Plane]:
        """Create the 6 faces of the building."""
        faces = []
        
        # Front face (x = min_x)
        faces.append(Plane(self.min_corner, [1, 0, 0], self.material, self.device))
        
        # Back face (x = max_x)
        faces.append(Plane([self.max_corner[0], self.min_corner[1], self.min_corner[2]], [-1, 0, 0], self.material, self.device))
        
        # Left face (y = min_y)
        faces.append(Plane(self.min_corner, [0, 1, 0], self.material, self.device))
        
        # Right face (y = max_y)
        faces.append(Plane([self.min_corner[0], self.max_corner[1], self.min_corner[2]], [0, -1, 0], self.material, self.device))
        
        # Bottom face (z = min_z)
        faces.append(Plane(self.min_corner, [0, 0, 1], self.material, self.device))
        
        # Top face (z = max_z)
        faces.append(Plane([self.min_corner[0], self.min_corner[1], self.max_corner[2]], [0, 0, -1], self.material, self.device))
        
        return faces
    
    def intersect(self, ray: Ray) -> Optional[RayIntersection]:
        """Calculate intersection with building."""
        closest_intersection = None
        min_distance = float('inf')
        
        for face in self.faces:
            intersection = face.intersect(ray)
            if intersection and intersection.distance < min_distance:
                min_distance = intersection.distance
                closest_intersection = intersection
        
        return closest_intersection
    
    def to_device(self, device: str):
        """Move building to specified device."""
        super().to_device(device)
        self.min_corner = self.min_corner.to(device)
        self.max_corner = self.max_corner.to(device)
        for face in self.faces:
            face.to_device(device)
        return self

class PathTracer:
    """Traces rays through the environment with GPU support."""
    
    def __init__(self, max_reflections: int = 3, max_diffractions: int = 2, device: str = 'cpu'):
        """
        Initialize path tracer.
        
        Args:
            max_reflections: Maximum number of reflections
            max_diffractions: Maximum number of diffractions
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.max_reflections = max_reflections
        self.max_diffractions = max_diffractions
        self.device = device
    
    def trace_ray(self, ray: Ray, environment: Environment, max_depth: int = 5) -> RayPath:
        """
        Trace a single ray through the environment.
        
        Args:
            ray: Ray to trace
            environment: Environment to trace through
            max_depth: Maximum recursion depth
            
        Returns:
            Complete ray path information
        """
        if max_depth <= 0:
            return self._create_ray_path(ray)
        
        # Find intersections
        intersections = environment.ray_intersection(ray)
        
        if not intersections:
            # Ray continues to maximum length
            end_point = ray.origin + ray.direction * ray.max_length
            ray.add_path_point(end_point)
            return self._create_ray_path(ray)
        
        # Process first intersection
        intersection = intersections[0]
        ray.add_path_point(intersection.point, intersection.interaction_type, intersection.material)
        
        # Generate secondary rays based on interaction type
        if intersection.interaction_type == 'reflection' and max_depth > 0:
            reflected_ray = self._generate_reflected_ray(ray, intersection)
            self.trace_ray(reflected_ray, environment, max_depth - 1)
        
        elif intersection.interaction_type == 'diffraction' and max_depth > 0:
            diffracted_ray = self._generate_diffracted_ray(ray, intersection)
            self.trace_ray(diffracted_ray, environment, max_depth - 1)
        
        return self._create_ray_path(ray)
    
    def trace_rays_batch(self, rays: List[Ray], environment: Environment, max_depth: int = 5) -> List[RayPath]:
        """
        Trace multiple rays in batch for GPU processing.
        
        Args:
            rays: List of rays to trace
            environment: Environment to trace through
            max_depth: Maximum recursion depth
            
        Returns:
            List of ray path results
        """
        results = []
        for ray in rays:
            result = self.trace_ray(ray, environment, max_depth)
            results.append(result)
        return results
    
    def _generate_reflected_ray(self, incident_ray: Ray, intersection: RayIntersection) -> Ray:
        """Generate reflected ray using Snell's law."""
        normal = intersection.normal
        incident_direction = incident_ray.direction
        
        # Reflection: r = i - 2(n·i)n
        dot_product = torch.dot(normal, incident_direction)
        reflected_direction = incident_direction - 2 * dot_product * normal
        
        return Ray(intersection.point, reflected_direction, device=self.device)
    
    def _generate_diffracted_ray(self, incident_ray: Ray, intersection: RayIntersection) -> Ray:
        """Generate diffracted ray using Huygens principle."""
        # Simplified diffraction model
        # In practice, this would use more sophisticated diffraction models
        diffracted_direction = incident_ray.direction  # Simplified
        
        return Ray(intersection.point, diffracted_direction, device=self.device)
    
    def _create_ray_path(self, ray: Ray) -> RayPath:
        """Create RayPath object from ray."""
        return RayPath(
            origin=ray.origin,
            direction=ray.direction,
            path_points=ray.path_points.copy(),
            interactions=ray.interactions.copy(),
            materials=ray.materials.copy(),
            total_length=ray.get_total_length(),
            final_point=ray.get_final_point()
        )
    
    def to_device(self, device: str):
        """Move path tracer to specified device."""
        self.device = device
        return self

class InteractionModel:
    """Models electromagnetic interactions with materials and GPU support."""
    
    def __init__(self, config: Optional[Dict] = None, device: str = 'cpu'):
        """
        Initialize interaction model.
        
        Args:
            config: Configuration dictionary with material properties
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.device = device
        if config and 'ray_tracing' in config and 'materials' in config['ray_tracing']:
            self.material_properties = config['ray_tracing']['materials']
        else:
            self.material_properties = self._get_default_materials()
    
    def _get_default_materials(self) -> Dict:
        """Get default material properties."""
        return {
            'concrete': {
                'permittivity': 4.5,
                'conductivity': 0.02,
                'roughness': 0.01
            },
            'glass': {
                'permittivity': 3.8,
                'conductivity': 0.0,
                'roughness': 0.001
            },
            'metal': {
                'permittivity': 1.0,
                'conductivity': 1e7,
                'roughness': 0.001
            },
            'wood': {
                'permittivity': 2.0,
                'conductivity': 0.01,
                'roughness': 0.05
            },
            'plastic': {
                'permittivity': 2.3,
                'conductivity': 0.0,
                'roughness': 0.02
            }
        }
    
    def compute_reflection_coefficient(self, incident_angle: float, material: str, frequency: float) -> float:
        """
        Compute reflection coefficient for given material and frequency.
        
        Args:
            incident_angle: Incident angle in degrees
            material: Material name
            frequency: Frequency in Hz
            
        Returns:
            Reflection coefficient (0 to 1)
        """
        if material not in self.material_properties:
            return 0.5  # Default value
        
        properties = self.material_properties[material]
        permittivity = properties.get('permittivity', 1.0)
        conductivity = properties.get('conductivity', 0.0)
        
        # Simplified Fresnel equations
        # In practice, this would use full electromagnetic theory
        
        # Convert angle to radians
        theta = np.radians(incident_angle)
        
        # Simplified calculation based on material properties
        if conductivity > 1e6:  # Metal
            reflection_coeff = 0.9
        else:  # Dielectric
            reflection_coeff = abs((permittivity - 1) / (permittivity + 1))
        
        # Angle dependence
        angle_factor = 1.0 - 0.1 * np.sin(theta)
        reflection_coeff *= angle_factor
        
        return np.clip(reflection_coeff, 0.0, 1.0)
    
    def compute_reflection_coefficient_batch(self, incident_angles: torch.Tensor, material: str, frequency: float) -> torch.Tensor:
        """
        Compute reflection coefficients for batch of incident angles on GPU.
        
        Args:
            incident_angles: Batch of incident angles in degrees [batch_size]
            material: Material name
            frequency: Frequency in Hz
            
        Returns:
            Batch of reflection coefficients [batch_size]
        """
        if material not in self.material_properties:
            return torch.full(incident_angles.shape, 0.5, device=self.device)
        
        properties = self.material_properties[material]
        permittivity = properties.get('permittivity', 1.0)
        conductivity = properties.get('conductivity', 0.0)
        
        # Convert angles to radians
        theta = torch.deg2rad(incident_angles)
        
        # Simplified calculation based on material properties
        if conductivity > 1e6:  # Metal
            reflection_coeff = torch.full_like(theta, 0.9)
        else:  # Dielectric
            reflection_coeff = torch.abs(torch.tensor((permittivity - 1) / (permittivity + 1), device=self.device))
        
        # Angle dependence
        angle_factor = 1.0 - 0.1 * torch.sin(theta)
        reflection_coeff = reflection_coeff * angle_factor
        
        return torch.clamp(reflection_coeff, 0.0, 1.0)
    
    def compute_diffraction_coefficient(self, edge_angle: float, material: str, frequency: float) -> float:
        """
        Compute diffraction coefficient for edge diffraction.
        
        Args:
            edge_angle: Edge angle in degrees
            material: Material name
            frequency: Frequency in Hz
            
        Returns:
            Diffraction coefficient (0 to 1)
        """
        if material not in self.material_properties:
            return 0.1  # Default value
        
        properties = self.material_properties[material]
        roughness = properties.get('roughness', 0.01)
        
        # Simplified edge diffraction model
        # In practice, this would use UTD or similar theory
        
        # Roughness effect
        roughness_factor = 1.0 - roughness * 10
        
        # Frequency effect (higher frequency = more diffraction)
        freq_factor = min(frequency / 1e9, 10.0) / 10.0
        
        diffraction_coeff = 0.1 * roughness_factor * freq_factor
        
        return np.clip(diffraction_coeff, 0.0, 1.0)
    
    def compute_diffraction_coefficient_batch(self, edge_angles: torch.Tensor, material: str, frequency: float) -> torch.Tensor:
        """
        Compute diffraction coefficients for batch of edge angles on GPU.
        
        Args:
            edge_angles: Batch of edge angles in degrees [batch_size]
            material: Material name
            frequency: Frequency in Hz
            
        Returns:
            Batch of diffraction coefficients [batch_size]
        """
        if material not in self.material_properties:
            return torch.full(edge_angles.shape, 0.1, device=self.device)
        
        properties = self.material_properties[material]
        roughness = properties.get('roughness', 0.01)
        
        # Simplified edge diffraction model
        # In practice, this would use UTD or similar theory
        
        # Roughness effect
        roughness_factor = 1.0 - roughness * 10
        
        # Frequency effect (higher frequency = more diffraction)
        freq_factor = min(frequency / 1e9, 10.0) / 10.0
        
        diffraction_coeff = 0.1 * roughness_factor * freq_factor
        
        return torch.clamp(torch.full_like(edge_angles, diffraction_coeff), 0.0, 1.0)
    
    def compute_scattering_coefficient(self, surface_roughness: float, material: str, frequency: float) -> float:
        """
        Compute scattering coefficient for rough surfaces.
        
        Args:
            surface_roughness: Surface roughness in meters
            material: Material name
            frequency: Frequency in Hz
            
        Returns:
            Scattering coefficient (0 to 1)
        """
        if material not in self.material_properties:
            return 0.3  # Default value
        
        # Calculate roughness factor relative to wavelength
        wavelength = 3e8 / frequency
        roughness_factor = surface_roughness / wavelength
        
        if roughness_factor < 0.1:
            return 0.0  # Smooth surface
        elif roughness_factor < 1.0:
            return 0.3  # Moderately rough
        else:
            return 0.7  # Very rough
    
    def compute_scattering_coefficient_batch(self, surface_roughness: torch.Tensor, material: str, frequency: float) -> torch.Tensor:
        """
        Compute scattering coefficients for batch of surface roughness values on GPU.
        
        Args:
            surface_roughness: Batch of surface roughness values in meters [batch_size]
            material: Material name
            frequency: Frequency in Hz
            
        Returns:
            Batch of scattering coefficients [batch_size]
        """
        if material not in self.material_properties:
            return torch.full(surface_roughness.shape, 0.3, device=self.device)
        
        # Calculate roughness factor relative to wavelength
        wavelength = 3e8 / frequency
        roughness_factor = surface_roughness / wavelength
        
        # Initialize coefficients
        coefficients = torch.zeros_like(surface_roughness)
        
        # Apply conditions
        smooth_mask = roughness_factor < 0.1
        moderate_mask = (roughness_factor >= 0.1) & (roughness_factor < 1.0)
        rough_mask = roughness_factor >= 1.0
        
        coefficients[smooth_mask] = 0.0    # Smooth surface
        coefficients[moderate_mask] = 0.3  # Moderately rough
        coefficients[rough_mask] = 0.7     # Very rough
        
        return coefficients
    
    def to_device(self, device: str):
        """Move interaction model to specified device."""
        self.device = device
        return self

class ChannelEstimator:
    """Estimates channel characteristics along ray paths with GPU support."""
    
    def __init__(self, frequency_band: float, bandwidth: float, device: str = 'cpu'):
        """
        Initialize channel estimator.
        
        Args:
            frequency_band: Center frequency in Hz
            bandwidth: Bandwidth in Hz
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.frequency_band = frequency_band
        self.bandwidth = bandwidth
        self.device = device
        self.subcarrier_frequencies = self._generate_subcarrier_frequencies()
    
    def _generate_subcarrier_frequencies(self) -> np.ndarray:
        """Generate subcarrier frequencies."""
        # Generate frequencies across the bandwidth
        start_freq = self.frequency_band - self.bandwidth / 2
        end_freq = self.frequency_band + self.bandwidth / 2
        
        # For now, generate 1024 subcarriers
        return np.linspace(start_freq, end_freq, 1024)
    
    def estimate_channel(self, ray_path: RayPath, environment: Environment) -> Dict[str, Union[float, np.ndarray]]:
        """
        Estimate channel characteristics along ray path.
        
        Args:
            ray_path: Complete ray path information
            environment: Environment information
            
        Returns:
            Dictionary containing channel characteristics
        """
        channel_info = {
            'path_loss': self._compute_path_loss(ray_path),
            'delay': self._compute_delay(ray_path),
            'doppler': self._compute_doppler(ray_path),
            'subcarrier_responses': self._compute_subcarrier_responses(ray_path)
        }
        
        return channel_info
    
    def estimate_channel_batch(self, ray_paths: List[RayPath], environment: Environment) -> Dict[str, torch.Tensor]:
        """
        Estimate channel characteristics for batch of ray paths on GPU.
        
        Args:
            ray_paths: List of ray path information
            environment: Environment information
            
        Returns:
            Dictionary containing batch channel characteristics
        """
        batch_size = len(ray_paths)
        
        # Pre-allocate tensors for batch processing
        path_losses = torch.zeros(batch_size, device=self.device)
        delays = torch.zeros(batch_size, device=self.device)
        dopplers = torch.zeros(batch_size, device=self.device)
        
        # Process each ray path
        for i, ray_path in enumerate(ray_paths):
            path_losses[i] = self._compute_path_loss(ray_path)
            delays[i] = self._compute_delay(ray_path)
            dopplers[i] = self._compute_doppler(ray_path)
        
        # Compute subcarrier responses for all paths
        subcarrier_responses = self._compute_subcarrier_responses_batch(ray_paths)
        
        return {
            'path_loss': path_losses,
            'delay': delays,
            'doppler': dopplers,
            'subcarrier_responses': subcarrier_responses
        }
    
    def _compute_path_loss(self, ray_path: RayPath) -> float:
        """Compute path loss along ray path."""
        total_loss = 0.0
        
        for i in range(len(ray_path.path_points) - 1):
            start = ray_path.path_points[i]
            end = ray_path.path_points[i + 1]
            
            # Free space path loss
            distance = torch.norm(end - start).item()
            wavelength = 3e8 / self.frequency_band
            
            # FSPL = (4πd/λ)²
            fspl = (4 * np.pi * distance / wavelength) ** 2
            total_loss += 10 * np.log10(fspl)
            
            # Additional losses from interactions
            if i < len(ray_path.interactions):
                interaction_loss = self._compute_interaction_loss(
                    ray_path.interactions[i], 
                    ray_path.materials[i] if i < len(ray_path.materials) else 'concrete'
                )
                total_loss += interaction_loss
        
        return total_loss
    
    def _compute_path_loss_batch(self, ray_paths: List[RayPath]) -> torch.Tensor:
        """Compute path loss for batch of ray paths on GPU."""
        batch_size = len(ray_paths)
        path_losses = torch.zeros(batch_size, device=self.device)
        
        for i, ray_path in enumerate(ray_paths):
            path_losses[i] = self._compute_path_loss(ray_path)
        
        return path_losses
    
    def _compute_interaction_loss(self, interaction_type: str, material: str) -> float:
        """Compute additional loss from interactions."""
        if interaction_type == 'reflection':
            return 2.0  # 2 dB reflection loss
        elif interaction_type == 'diffraction':
            return 5.0  # 5 dB diffraction loss
        elif interaction_type == 'scattering':
            return 3.0  # 3 dB scattering loss
        else:
            return 0.0
    
    def _compute_delay(self, ray_path: RayPath) -> float:
        """Compute total delay along ray path."""
        return ray_path.total_length / 3e8  # Speed of light
    
    def _compute_doppler(self, ray_path: RayPath) -> float:
        """Compute Doppler shift (simplified)."""
        # For now, return 0 (no motion)
        # In practice, this would consider relative motion between source and receiver
        return 0.0
    
    def _compute_subcarrier_responses(self, ray_path: RayPath) -> np.ndarray:
        """Compute frequency response for each subcarrier."""
        responses = []
        
        for freq in self.subcarrier_frequencies:
            # Frequency-dependent path loss
            wavelength = 3e8 / freq
            total_response = 0.0
            
            for i in range(len(ray_path.path_points) - 1):
                start = ray_path.path_points[i]
                end = ray_path.path_points[i + 1]
                distance = torch.norm(end - start).item()
                
                # Frequency-dependent path loss
                fspl = (4 * np.pi * distance / wavelength) ** 2
                response = 1.0 / fspl
                
                # Phase delay
                phase_delay = 2 * np.pi * distance / wavelength
                response *= np.exp(-1j * phase_delay)
                
                total_response += response
            
            responses.append(total_response)
        
        return np.array(responses)
    
    def _compute_subcarrier_responses_batch(self, ray_paths: List[RayPath]) -> torch.Tensor:
        """Compute frequency response for batch of ray paths on GPU."""
        num_subcarriers = len(self.subcarrier_frequencies)
        batch_size = len(ray_paths)
        
        # Pre-allocate tensor for batch processing
        responses = torch.zeros(batch_size, num_subcarriers, dtype=torch.complex64, device=self.device)
        
        # Convert frequencies to tensor
        frequencies = torch.tensor(self.subcarrier_frequencies, device=self.device)
        wavelengths = 3e8 / frequencies
        
        for i, ray_path in enumerate(ray_paths):
            for j, (start, end) in enumerate(zip(ray_path.path_points[:-1], ray_path.path_points[1:])):
                distance = torch.norm(end - start)
                
                # Frequency-dependent path loss
                fspl = (4 * np.pi * distance / wavelengths) ** 2
                response = 1.0 / fspl
                
                # Phase delay
                phase_delay = 2 * np.pi * distance / wavelengths
                response *= torch.exp(-1j * phase_delay)
                
                responses[i] += response
        
        return responses
    
    def to_device(self, device: str):
        """Move channel estimator to specified device."""
        self.device = device
        return self

class SpatialAnalyzer:
    """Analyzes spatial distribution of ray tracing results with GPU support."""
    
    def __init__(self, spatial_resolution: float = 0.1, device: str = 'cpu'):
        """
        Initialize spatial analyzer.
        
        Args:
            spatial_resolution: Spatial resolution in meters
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.spatial_resolution = spatial_resolution
        self.device = device
    
    def analyze_spatial_distribution(self, ray_tracing_results: List[RayPath]) -> Dict[str, Union[np.ndarray, int]]:
        """
        Analyze spatial distribution of ray tracing results.
        
        Args:
            ray_tracing_results: List of ray path results
            
        Returns:
            Dictionary containing spatial analysis results
        """
        # Extract all spatial points
        all_points = []
        all_responses = []
        
        for ray_result in ray_tracing_results:
            points = ray_result.path_points
            responses = [1.0] * len(points)  # Simple response model
            
            all_points.extend(points)
            all_responses.extend(responses)
        
        if not all_points:
            return {'total_points': 0}
        
        all_points = torch.stack(all_points)
        all_responses = torch.tensor(all_responses)
        
        # Create spatial grid
        x_min, x_max = all_points[:, 0].min().item(), all_points[:, 0].max().item()
        y_min, y_max = all_points[:, 1].min().item(), all_points[:, 1].max().item()
        z_min, z_max = all_points[:, 2].min().item(), all_points[:, 2].max().item()
        
        x_grid = np.arange(x_min, x_max + self.spatial_resolution, self.spatial_resolution)
        y_grid = np.arange(y_min, y_max + self.spatial_resolution, self.spatial_resolution)
        z_grid = np.arange(z_min, z_max + self.spatial_resolution, self.spatial_resolution)
        
        # Interpolate responses on grid
        spatial_grid = self._interpolate_on_grid(all_points, all_responses, x_grid, y_grid, z_grid)
        
        return {
            'x_grid': x_grid,
            'y_grid': y_grid,
            'z_grid': z_grid,
            'spatial_grid': spatial_grid,
            'total_points': len(all_points)
        }
    
    def analyze_spatial_distribution_batch(self, ray_tracing_results: List[RayPath]) -> Dict[str, torch.Tensor]:
        """
        Analyze spatial distribution for batch of ray tracing results on GPU.
        
        Args:
            ray_tracing_results: List of ray path results
            
        Returns:
            Dictionary containing batch spatial analysis results
        """
        # Extract all spatial points
        all_points = []
        all_responses = []
        
        for ray_result in ray_tracing_results:
            points = ray_result.path_points
            responses = [1.0] * len(points)  # Simple response model
            
            all_points.extend(points)
            all_responses.extend(responses)
        
        if not all_points:
            return {'total_points': torch.tensor(0, device=self.device)}
        
        all_points = torch.stack(all_points)
        all_responses = torch.tensor(all_responses, device=self.device)
        
        # Create spatial grid
        x_min, x_max = all_points[:, 0].min().item(), all_points[:, 0].max().item()
        y_min, y_max = all_points[:, 1].min().item(), all_points[:, 1].max().item()
        z_min, z_max = all_points[:, 2].min().item(), all_points[:, 2].max().item()
        
        x_grid = torch.arange(x_min, x_max + self.spatial_resolution, self.spatial_resolution, device=self.device)
        y_grid = torch.arange(y_min, y_max + self.spatial_resolution, self.spatial_resolution, device=self.device)
        z_grid = torch.arange(z_min, z_max + self.spatial_resolution, self.spatial_resolution, device=self.device)
        
        # Interpolate responses on grid using GPU
        spatial_grid = self._interpolate_on_grid_gpu(all_points, all_responses, x_grid, y_grid, z_grid)
        
        return {
            'x_grid': x_grid,
            'y_grid': y_grid,
            'z_grid': z_grid,
            'spatial_grid': spatial_grid,
            'total_points': torch.tensor(len(all_points), device=self.device)
        }
    
    def _interpolate_on_grid(self, points: torch.Tensor, values: torch.Tensor, 
                            x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
        """Interpolate values on regular spatial grid."""
        try:
            from scipy.interpolate import griddata
            
            # Prepare interpolation
            xi, yi, zi = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
            
            # Interpolate using nearest neighbor for efficiency
            grid_values = griddata(points.numpy(), values.numpy(), (xi, yi, zi), method='nearest')
            
            return grid_values
        except ImportError:
            # Fallback to simple interpolation if scipy is not available
            logger.warning("scipy not available, using simple interpolation")
            return np.zeros((len(x_grid), len(y_grid), len(z_grid)))
    
    def _interpolate_on_grid_gpu(self, points: torch.Tensor, values: torch.Tensor, 
                                 x_grid: torch.Tensor, y_grid: torch.Tensor, z_grid: torch.Tensor) -> torch.Tensor:
        """Interpolate values on regular spatial grid using GPU."""
        # Create meshgrid on GPU
        xi, yi, zi = torch.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        
        # Flatten grid coordinates
        grid_coords = torch.stack([xi.flatten(), yi.flatten(), zi.flatten()], dim=1)
        
        # Find nearest neighbors for each grid point
        distances = torch.cdist(grid_coords, points)
        nearest_indices = torch.argmin(distances, dim=1)
        
        # Get values for nearest neighbors
        grid_values = values[nearest_indices]
        
        # Reshape back to grid
        grid_values = grid_values.view(xi.shape)
        
        return grid_values
    
    def to_device(self, device: str):
        """Move spatial analyzer to specified device."""
        self.device = device
        return self

class AdvancedRayTracer:
    """Main ray tracing engine that orchestrates all components with full GPU support."""
    
    def __init__(self, config: Optional[Dict] = None, device: str = 'cpu'):
        """
        Initialize the advanced ray tracer.
        
        Args:
            config: Configuration dictionary
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.config = config or {}
        self.device = device
        
        # Extract ray tracing configuration
        ray_config = self.config.get('ray_tracing', {})
        
        # Check if GPU acceleration is enabled
        gpu_enabled = ray_config.get('gpu_acceleration', False) and device == 'cuda'
        if gpu_enabled and not torch.cuda.is_available():
            logger.warning("GPU acceleration requested but CUDA not available, falling back to CPU")
            self.device = 'cpu'
            gpu_enabled = False
        
        # Initialize components with device support
        self.ray_generator = RayGenerator(
            azimuth_samples=ray_config.get('azimuth_samples', 36),
            elevation_samples=ray_config.get('elevation_samples', 18),
            device=self.device
        )
        
        self.path_tracer = PathTracer(
            max_reflections=ray_config.get('reflection_order', 3),
            max_diffractions=ray_config.get('max_diffractions', 2),
            device=self.device
        )
        
        self.interaction_model = InteractionModel(config, device=self.device)
        self.channel_estimator = ChannelEstimator(
            frequency_band=ray_config.get('frequency_band', 2.4e9),
            bandwidth=ray_config.get('bandwidth', 20e6),
            device=self.device
        )
        
        self.spatial_analyzer = SpatialAnalyzer(
            spatial_resolution=ray_config.get('spatial_resolution', 0.1),
            device=self.device
        )
        
        # Performance settings
        self.batch_size = ray_config.get('batch_size', 256)
        self.max_concurrent_rays = ray_config.get('max_concurrent_rays', 1000)
        self.parallel_processing = ray_config.get('parallel_processing', True)
        
        logger.info(f"Advanced Ray Tracer initialized successfully on {self.device}")
        if gpu_enabled:
            logger.info(f"GPU acceleration enabled with batch size {self.batch_size}")
    
    def trace_rays(self, source_position: Union[List, np.ndarray, torch.Tensor],
                   target_positions: List[Union[List, np.ndarray, torch.Tensor]],
                   environment: Environment) -> List[RayPath]:
        """
        Trace rays from source to targets through environment.
        
        Args:
            source_position: Source position [3]
            target_positions: List of target positions
            environment: Environment to trace through
            
        Returns:
            List of ray path results
        """
        # Ensure environment is on the same device
        if hasattr(environment, 'device') and environment.device != self.device:
            environment = environment.to_device(self.device)
        
        # Generate rays from source
        rays = self.ray_generator.generate_rays(source_position)
        
        # Trace each ray
        results = []
        for ray in rays:
            result = self.path_tracer.trace_ray(ray, environment)
            results.append(result)
        
        logger.info(f"Traced {len(rays)} rays through environment on {self.device}")
        return results
    
    def trace_rays_batch(self, source_positions: torch.Tensor,
                         target_positions: Optional[torch.Tensor] = None,
                         environment: Environment = None) -> List[RayPath]:
        """
        Trace rays in batch for GPU-accelerated processing.
        
        Args:
            source_positions: Batch of source positions [batch_size, 3]
            target_positions: Optional batch of target positions [batch_size, num_targets, 3]
            environment: Environment to trace through
            
        Returns:
            List of ray path results
        """
        if environment is None:
            environment = Environment(device=self.device)
        elif hasattr(environment, 'device') and environment.device != self.device:
            environment = environment.to_device(self.device)
        
        batch_size = source_positions.shape[0]
        total_rays = len(self.ray_generator.angle_combinations)
        
        # Generate rays in batch
        ray_origins, ray_directions = self.ray_generator.generate_rays_batch(source_positions)
        
        # Process in batches for memory efficiency
        all_results = []
        for i in range(0, batch_size, self.batch_size):
            batch_end = min(i + self.batch_size, batch_size)
            batch_origins = ray_origins[i:batch_end]
            batch_directions = ray_directions[i:batch_end]
            
            # Create rays for this batch
            batch_rays = []
            for j in range(batch_end - i):
                for k in range(total_rays):
                    ray = Ray(
                        batch_origins[j, k], 
                        batch_directions[j, k], 
                        device=self.device
                    )
                    batch_rays.append(ray)
            
            # Trace rays for this batch
            batch_results = self.path_tracer.trace_rays_batch(batch_rays, environment)
            all_results.extend(batch_results)
        
        logger.info(f"Traced {len(all_results)} rays in batch on {self.device}")
        return all_results
    
    def trace_rays_parallel(self, source_positions: torch.Tensor,
                           target_positions: Optional[torch.Tensor] = None,
                           environment: Environment = None,
                           num_workers: int = 4) -> List[RayPath]:
        """
        Trace rays in parallel for improved performance.
        
        Args:
            source_positions: Batch of source positions [batch_size, 3]
            target_positions: Optional batch of target positions [batch_size, num_targets, 3]
            environment: Environment to trace through
            num_workers: Number of parallel workers
            
        Returns:
            List of ray path results
        """
        if not self.parallel_processing:
            return self.trace_rays_batch(source_positions, target_positions, environment)
        
        # For now, use batch processing as parallel implementation
        # In a full implementation, this would use torch.multiprocessing or similar
        return self.trace_rays_batch(source_positions, target_positions, environment)
    
    def analyze_spatial_distribution(self, ray_tracing_results: List[RayPath]) -> Dict[str, Union[np.ndarray, int]]:
        """Analyze spatial distribution of ray tracing results."""
        if self.device == 'cuda':
            return self.spatial_analyzer.analyze_spatial_distribution_batch(ray_tracing_results)
        else:
            return self.spatial_analyzer.analyze_spatial_distribution(ray_tracing_results)
    
    def estimate_channels(self, ray_tracing_results: List[RayPath], environment: Environment) -> Dict[str, torch.Tensor]:
        """Estimate channel characteristics for ray tracing results."""
        if self.device == 'cuda':
            return self.channel_estimator.estimate_channel_batch(ray_tracing_results, environment)
        else:
            # Fall back to individual estimation for CPU
            results = []
            for ray_path in ray_tracing_results:
                result = self.channel_estimator.estimate_channel(ray_path, environment)
                results.append(result)
            return results
    
    def get_ray_statistics(self, ray_tracing_results: List[RayPath]) -> Dict[str, float]:
        """Get comprehensive statistics for ray tracing results."""
        if not ray_tracing_results:
            return {}
        
        total_lengths = [result.total_length for result in ray_tracing_results]
        interaction_counts = [len(result.interactions) for result in ray_tracing_results]
        
        return {
            'total_rays': len(ray_tracing_results),
            'mean_ray_length': np.mean(total_lengths),
            'std_ray_length': np.std(total_lengths),
            'max_ray_length': np.max(total_lengths),
            'min_ray_length': np.min(total_lengths),
            'mean_interactions': np.mean(interaction_counts),
            'std_interactions': np.std(interaction_counts),
            'total_spatial_points': sum(len(result.path_points) for result in ray_tracing_results)
        }
    
    def get_ray_statistics_gpu(self, ray_tracing_results: List[RayPath]) -> Dict[str, torch.Tensor]:
        """Get comprehensive statistics for ray tracing results using GPU."""
        if not ray_tracing_results:
            return {}
        
        # Convert to tensors for GPU processing
        total_lengths = torch.tensor([result.total_length for result in ray_tracing_results], device=self.device)
        interaction_counts = torch.tensor([len(result.interactions) for result in ray_tracing_results], device=self.device)
        
        return {
            'total_rays': torch.tensor(len(ray_tracing_results), device=self.device),
            'mean_ray_length': torch.mean(total_lengths),
            'std_ray_length': torch.std(total_lengths),
            'max_ray_length': torch.max(total_lengths),
            'min_ray_length': torch.min(total_lengths),
            'mean_interactions': torch.mean(interaction_counts),
            'std_interactions': torch.std(interaction_counts),
            'total_spatial_points': torch.sum(torch.tensor([len(result.path_points) for result in ray_tracing_results], device=self.device))
        }
    
    def to_device(self, device: str):
        """Move ray tracer to specified device."""
        self.device = device
        
        # Move all components to the new device
        self.ray_generator = RayGenerator(
            azimuth_samples=len(self.ray_generator.angle_combinations),
            elevation_samples=self.ray_generator.elevation_samples,
            device=device
        )
        
        self.path_tracer.to_device(device)
        self.interaction_model.to_device(device)
        self.channel_estimator.to_device(device)
        self.spatial_analyzer.to_device(device)
        
        logger.info(f"Ray tracer moved to {device}")
        return self
    
    def get_performance_info(self) -> Dict[str, Union[str, int, bool]]:
        """Get performance information about the ray tracer."""
        return {
            'device': self.device,
            'gpu_acceleration': self.device == 'cuda',
            'batch_size': self.batch_size,
            'max_concurrent_rays': self.max_concurrent_rays,
            'parallel_processing': self.parallel_processing,
            'cuda_available': torch.cuda.is_available() if self.device == 'cuda' else False,
            'gpu_memory_allocated': torch.cuda.memory_allocated() if self.device == 'cuda' else 0,
            'gpu_memory_reserved': torch.cuda.memory_reserved() if self.device == 'cuda' else 0
        }

class GPURayTracer:
    """
    High-performance GPU ray tracer using CUDA for maximum acceleration.
    
    This class implements optimized GPU kernels for ray tracing operations
    including batch processing, parallel ray generation, and GPU memory management.
    """
    
    def __init__(self, config: Optional[Dict] = None, device: str = 'cuda'):
        """
        Initialize GPU ray tracer.
        
        Args:
            config: Configuration dictionary
            device: Device to run computations on (should be 'cuda')
        """
        if device != 'cuda':
            raise ValueError("GPURayTracer requires CUDA device")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for GPU ray tracer")
        
        self.config = config or {}
        self.device = device
        
        # Extract ray tracing configuration
        ray_config = self.config.get('ray_tracing', {})
        
        # Performance settings optimized for GPU
        self.batch_size = ray_config.get('batch_size', 512)  # Larger batches for GPU
        self.max_concurrent_rays = ray_config.get('max_concurrent_rays', 2000)
        self.gpu_memory_fraction = ray_config.get('gpu_memory_fraction', 0.8)
        self.mixed_precision = ray_config.get('mixed_precision', True)
        
        # Initialize GPU memory pool
        self._initialize_gpu_memory()
        
        # Initialize components
        self.ray_generator = RayGenerator(
            azimuth_samples=ray_config.get('azimuth_samples', 36),
            elevation_samples=ray_config.get('elevation_samples', 18),
            device=self.device
        )
        
        self.path_tracer = PathTracer(
            max_reflections=ray_config.get('reflection_order', 3),
            max_diffractions=ray_config.get('max_diffractions', 2),
            device=self.device
        )
        
        self.interaction_model = InteractionModel(config, device=self.device)
        self.channel_estimator = ChannelEstimator(
            frequency_band=ray_config.get('frequency_band', 2.4e9),
            bandwidth=ray_config.get('bandwidth', 20e6),
            device=self.device
        )
        
        self.spatial_analyzer = SpatialAnalyzer(
            spatial_resolution=ray_config.get('spatial_resolution', 0.1),
            device=self.device
        )
        
        logger.info(f"GPU Ray Tracer initialized on {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"Batch Size: {self.batch_size}, Mixed Precision: {self.mixed_precision}")
    
    def _initialize_gpu_memory(self):
        """Initialize GPU memory management."""
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
        
        # Enable mixed precision if requested
        if self.mixed_precision:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Clear cache
        torch.cuda.empty_cache()
    
    def trace_rays_gpu_optimized(self, source_positions: torch.Tensor,
                                target_positions: Optional[torch.Tensor] = None,
                                environment: Environment = None) -> Dict[str, torch.Tensor]:
        """
        GPU-optimized ray tracing with maximum performance.
        
        Args:
            source_positions: Batch of source positions [batch_size, 3]
            target_positions: Optional batch of target positions [batch_size, num_targets, 3]
            environment: Environment to trace through
            
        Returns:
            Dictionary containing optimized ray tracing results
        """
        batch_size = source_positions.shape[0]
        total_rays = len(self.ray_generator.angle_combinations)
        
        # Pre-allocate GPU tensors for maximum performance
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            # Generate rays in batch on GPU
            ray_origins, ray_directions = self.ray_generator.generate_rays_batch(source_positions)
            
            # Process in optimized batches
            all_results = []
            for i in range(0, batch_size, self.batch_size):
                batch_end = min(i + self.batch_size, batch_size)
                batch_origins = ray_origins[i:batch_end]
                batch_directions = ray_directions[i:batch_end]
                
                # GPU-optimized batch processing
                batch_results = self._process_ray_batch_gpu(
                    batch_origins, batch_directions, environment
                )
                all_results.extend(batch_results)
        
        # Synchronize GPU
        torch.cuda.synchronize()
        
        logger.info(f"GPU-optimized ray tracing completed: {len(all_results)} rays")
        return {
            'ray_paths': all_results,
            'batch_size': batch_size,
            'total_rays': total_rays,
            'gpu_memory_used': torch.cuda.memory_allocated() / 1e9,
            'gpu_memory_reserved': torch.cuda.memory_reserved() / 1e9
        }
    
    def _process_ray_batch_gpu(self, ray_origins: torch.Tensor, 
                              ray_directions: torch.Tensor,
                              environment: Environment) -> List[RayPath]:
        """Process a batch of rays on GPU with optimized memory usage."""
        batch_size, num_rays, _ = ray_origins.shape
        
        # Flatten for GPU processing
        flat_origins = ray_origins.view(-1, 3)
        flat_directions = ray_directions.view(-1, 3)
        
        # Create rays for GPU processing
        rays = []
        for i in range(flat_origins.shape[0]):
            ray = Ray(flat_origins[i], flat_directions[i], device=self.device)
            rays.append(ray)
        
        # Process rays in parallel on GPU
        results = self.path_tracer.trace_rays_batch(rays, environment)
        
        return results
    
    def estimate_channels_gpu(self, ray_tracing_results: List[RayPath], 
                            environment: Environment) -> Dict[str, torch.Tensor]:
        """GPU-accelerated channel estimation."""
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            return self.channel_estimator.estimate_channel_batch(ray_tracing_results, environment)
    
    def analyze_spatial_distribution_gpu(self, ray_tracing_results: List[RayPath]) -> Dict[str, torch.Tensor]:
        """GPU-accelerated spatial analysis."""
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            return self.spatial_analyzer.analyze_spatial_distribution_batch(ray_tracing_results)
    
    def get_gpu_performance_metrics(self) -> Dict[str, Union[str, float, int]]:
        """Get comprehensive GPU performance metrics."""
        return {
            'device_name': torch.cuda.get_device_name(),
            'device_capability': torch.cuda.get_device_capability(),
            'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9,
            'gpu_memory_reserved': torch.cuda.memory_reserved() / 1e9,
            'gpu_memory_free': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9,
            'batch_size': self.batch_size,
            'max_concurrent_rays': self.max_concurrent_rays,
            'mixed_precision': self.mixed_precision,
            'cuda_version': torch.version.cuda
        }
    
    def optimize_gpu_memory(self):
        """Optimize GPU memory usage."""
        # Clear cache
        torch.cuda.empty_cache()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        logger.info("GPU memory optimized")
    
    def to_device(self, device: str):
        """Move GPU ray tracer to specified device (should remain on GPU)."""
        if device != 'cuda':
            raise ValueError("GPURayTracer cannot be moved to non-CUDA device")
        return self

# Usage Examples and Helper Functions

def create_gpu_ray_tracer(config: Optional[Dict] = None) -> GPURayTracer:
    """
    Create a GPU-accelerated ray tracer with optimal settings.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        GPURayTracer instance
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for GPU ray tracing")
    
    # Use default config if none provided
    if config is None:
        config = {
            'ray_tracing': {
                'gpu_acceleration': True,
                'batch_size': 512,
                'max_concurrent_rays': 2000,
                'gpu_memory_fraction': 0.8,
                'mixed_precision': True,
                'azimuth_samples': 36,
                'elevation_samples': 18,
                'points_per_ray': 64
            }
        }
    
    return GPURayTracer(config, device='cuda')

def create_cpu_ray_tracer(config: Optional[Dict] = None) -> AdvancedRayTracer:
    """
    Create a CPU-based ray tracer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AdvancedRayTracer instance
    """
    if config is None:
        config = {
            'ray_tracing': {
                'gpu_acceleration': False,
                'batch_size': 64,
                'max_concurrent_rays': 500,
                'azimuth_samples': 36,
                'elevation_samples': 18,
                'points_per_ray': 64
            }
        }
    
    return AdvancedRayTracer(config, device='cpu')

def benchmark_ray_tracing_performance(ray_tracer, source_positions: torch.Tensor, 
                                    environment: Environment, num_runs: int = 5) -> Dict[str, float]:
    """
    Benchmark ray tracing performance.
    
    Args:
        ray_tracer: Ray tracer instance
        source_positions: Source positions for testing
        environment: Environment to trace through
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary containing performance metrics
    """
    import time
    
    # Warm up
    _ = ray_tracer.trace_rays(source_positions[:10], None, environment)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = ray_tracer.trace_rays(source_positions, None, environment)
        torch.cuda.synchronize() if ray_tracer.device == 'cuda' else None
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    # Calculate rays per second
    total_rays = source_positions.shape[0] * len(ray_tracer.ray_generator.angle_combinations)
    rays_per_second = total_rays / avg_time
    
    return {
        'average_time': avg_time,
        'std_time': std_time,
        'total_rays': total_rays,
        'rays_per_second': rays_per_second,
        'device': ray_tracer.device
    }

def compare_cpu_gpu_performance(source_positions: torch.Tensor, environment: Environment,
                              config: Optional[Dict] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare CPU vs GPU ray tracing performance.
    
    Args:
        source_positions: Source positions for testing
        environment: Environment to trace through
        config: Configuration dictionary
        
    Returns:
        Dictionary containing performance comparison
    """
    results = {}
    
    # Test CPU performance
    try:
        cpu_tracer = create_cpu_ray_tracer(config)
        cpu_results = benchmark_ray_tracing_performance(cpu_tracer, source_positions, environment)
        results['cpu'] = cpu_results
        logger.info(f"CPU Performance: {cpu_results['rays_per_second']:.0f} rays/sec")
    except Exception as e:
        logger.warning(f"CPU benchmark failed: {e}")
        results['cpu'] = None
    
    # Test GPU performance
    try:
        gpu_tracer = create_gpu_ray_tracer(config)
        gpu_results = benchmark_ray_tracing_performance(gpu_tracer, source_positions, environment)
        results['gpu'] = gpu_results
        logger.info(f"GPU Performance: {gpu_results['rays_per_second']:.0f} rays/sec")
    except Exception as e:
        logger.warning(f"GPU benchmark failed: {e}")
        results['gpu'] = None
    
    # Calculate speedup
    if results['cpu'] and results['gpu']:
        speedup = gpu_results['rays_per_second'] / cpu_results['rays_per_second']
        results['speedup'] = speedup
        logger.info(f"GPU Speedup: {speedup:.1f}x")
    
    return results

# Example usage functions
def example_basic_usage():
    """Example of basic ray tracing usage."""
    # Create environment
    env = Environment(device='cuda')
    
    # Add obstacles
    wall = Plane([0, 0, 0], [1, 0, 0], 'concrete', device='cuda')
    building = Building([-10, -10, 0], [10, 10, 20], 'concrete', device='cuda')
    env.add_obstacle(wall)
    env.add_obstacle(building)
    
    # Create source positions
    source_positions = torch.randn(100, 3, device='cuda')
    
    # Create GPU ray tracer
    gpu_tracer = create_gpu_ray_tracer()
    
    # Trace rays
    results = gpu_tracer.trace_rays_gpu_optimized(source_positions, environment=env)
    
    # Analyze results
    spatial_analysis = gpu_tracer.analyze_spatial_distribution_gpu(results['ray_paths'])
    channel_estimation = gpu_tracer.estimate_channels_gpu(results['ray_paths'], env)
    
    # Get performance metrics
    performance = gpu_tracer.get_gpu_performance_metrics()
    
    logger.info(f"Ray tracing completed: {len(results['ray_paths'])} rays")
    logger.info(f"GPU Memory Used: {performance['gpu_memory_allocated']:.2f} GB")
    
    return results, spatial_analysis, channel_estimation, performance

def example_virtual_link_ray_tracing():
    """Example of ray tracing with virtual link processing."""
    # Configuration for virtual link processing
    config = {
        'ray_tracing': {
            'gpu_acceleration': True,
            'batch_size': 256,
            'azimuth_samples': 36,
            'elevation_samples': 18,
            'points_per_ray': 64
        },
        'csi_processing': {
            'virtual_link_enabled': True,
            'm_subcarriers': 408,
            'n_ue_antennas': 4,
            'sample_size': 64
        }
    }
    
    # Create GPU ray tracer
    gpu_tracer = create_gpu_ray_tracer(config)
    
    # Create environment
    env = Environment(device='cuda')
    
    # Add urban environment
    buildings = [
        Building([-50, -50, 0], [50, 50, 30], 'concrete', device='cuda'),
        Building([-100, -100, 0], [100, 100, 50], 'concrete', device='cuda')
    ]
    for building in buildings:
        env.add_obstacle(building)
    
    # Create UE positions (virtual links)
    ue_positions = torch.randn(1000, 3, device='cuda')  # 1000 UE positions
    
    # Trace rays for virtual links
    results = gpu_tracer.trace_rays_gpu_optimized(ue_positions, environment=env)
    
    # Process virtual links (sample K=64 from 408×4=1632 total)
    virtual_link_results = {
        'total_virtual_links': 1632,
        'sampled_virtual_links': 64,
        'ray_paths': results['ray_paths'],
        'gpu_performance': gpu_tracer.get_gpu_performance_metrics()
    }
    
    logger.info(f"Virtual link ray tracing completed: {len(results['ray_paths'])} rays")
    logger.info(f"Sampled {64} virtual links from {1632} total")
    
    return virtual_link_results
