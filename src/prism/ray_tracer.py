"""
Advanced Ray Tracing Engine for Prism: Wideband RF Neural Radiance Fields.
Implements comprehensive ray tracing with configurable azimuth/elevation sampling and spatial point sampling.
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
    
    def __init__(self, origin: torch.Tensor, direction: torch.Tensor, max_length: float = 100.0):
        """
        Initialize a ray.
        
        Args:
            origin: Ray origin point [3]
            direction: Ray direction vector [3]
            max_length: Maximum ray length
        """
        self.origin = torch.tensor(origin, dtype=torch.float32)
        self.direction = self._normalize(torch.tensor(direction, dtype=torch.float32))
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
            return torch.empty(0, 3)
        
        samples = []
        
        # Interpolate between path points
        for i in range(len(self.path_points) - 1):
            start = self.path_points[i]
            end = self.path_points[i + 1]
            
            # Generate samples between start and end
            segment_samples = torch.linspace(0, 1, num_points // len(self.path_points) + 1)
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
    """Generates rays for given azimuth and elevation angles."""
    
    def __init__(self, azimuth_samples: int = 36, elevation_samples: int = 18):
        """
        Initialize ray generator.
        
        Args:
            azimuth_samples: Number of azimuth angles (0° to 360°)
            elevation_samples: Number of elevation angles (-90° to +90°)
        """
        self.azimuth_samples = azimuth_samples
        self.elevation_samples = elevation_samples
        self.angle_combinations = self._generate_angle_combinations()
        
        logger.info(f"Ray generator initialized with {azimuth_samples}×{elevation_samples} = {len(self.angle_combinations)} angles")
    
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
            ray = Ray(source_position, direction)
            rays.append(ray)
        
        return rays
    
    def get_angle_resolution(self) -> Tuple[float, float]:
        """Get angular resolution in degrees."""
        azimuth_resolution = 360.0 / self.azimuth_samples
        elevation_resolution = 180.0 / (self.elevation_samples - 1)
        
        return azimuth_resolution, elevation_resolution

class Environment:
    """Represents the environment for ray tracing."""
    
    def __init__(self):
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

class Obstacle:
    """Base class for obstacles in the environment."""
    
    def __init__(self, material: str = 'concrete'):
        self.material = material
    
    def intersect(self, ray: Ray) -> Optional[RayIntersection]:
        """Calculate intersection with ray. Must be implemented by subclasses."""
        raise NotImplementedError

class Plane(Obstacle):
    """Plane obstacle (e.g., wall, floor, ceiling)."""
    
    def __init__(self, point: np.ndarray, normal: np.ndarray, material: str = 'concrete'):
        super().__init__(material)
        self.point = torch.tensor(point, dtype=torch.float32)
        self.normal = torch.tensor(normal, dtype=torch.float32)
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

class Building(Obstacle):
    """Building obstacle with box geometry."""
    
    def __init__(self, min_corner: np.ndarray, max_corner: np.ndarray, material: str = 'concrete'):
        super().__init__(material)
        self.min_corner = torch.tensor(min_corner, dtype=torch.float32)
        self.max_corner = torch.tensor(max_corner, dtype=torch.float32)
        
        # Create 6 planes for the building faces
        self.faces = self._create_faces()
    
    def _create_faces(self) -> List[Plane]:
        """Create the 6 faces of the building."""
        faces = []
        
        # Front face (x = min_x)
        faces.append(Plane(self.min_corner, [1, 0, 0], self.material))
        
        # Back face (x = max_x)
        faces.append(Plane([self.max_corner[0], self.min_corner[1], self.min_corner[2]], [-1, 0, 0], self.material))
        
        # Left face (y = min_y)
        faces.append(Plane(self.min_corner, [0, 1, 0], self.material))
        
        # Right face (y = max_y)
        faces.append(Plane([self.min_corner[0], self.max_corner[1], self.min_corner[2]], [0, -1, 0], self.material))
        
        # Bottom face (z = min_z)
        faces.append(Plane(self.min_corner, [0, 0, 1], self.material))
        
        # Top face (z = max_z)
        faces.append(Plane([self.min_corner[0], self.min_corner[1], self.max_corner[2]], [0, 0, -1], self.material))
        
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

class PathTracer:
    """Traces rays through the environment."""
    
    def __init__(self, max_reflections: int = 3, max_diffractions: int = 2):
        """
        Initialize path tracer.
        
        Args:
            max_reflections: Maximum number of reflections
            max_diffractions: Maximum number of diffractions
        """
        self.max_reflections = max_reflections
        self.max_diffractions = max_diffractions
    
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
    
    def _generate_reflected_ray(self, incident_ray: Ray, intersection: RayIntersection) -> Ray:
        """Generate reflected ray using Snell's law."""
        normal = intersection.normal
        incident_direction = incident_ray.direction
        
        # Reflection: r = i - 2(n·i)n
        dot_product = torch.dot(normal, incident_direction)
        reflected_direction = incident_direction - 2 * dot_product * normal
        
        return Ray(intersection.point, reflected_direction)
    
    def _generate_diffracted_ray(self, incident_ray: Ray, intersection: RayIntersection) -> Ray:
        """Generate diffracted ray using Huygens principle."""
        # Simplified diffraction model
        # In practice, this would use more sophisticated diffraction models
        diffracted_direction = incident_ray.direction  # Simplified
        
        return Ray(intersection.point, diffracted_direction)
    
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

class InteractionModel:
    """Models electromagnetic interactions with materials."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize interaction model.
        
        Args:
            config: Configuration dictionary with material properties
        """
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

class ChannelEstimator:
    """Estimates channel characteristics along ray paths."""
    
    def __init__(self, frequency_band: float, bandwidth: float):
        """
        Initialize channel estimator.
        
        Args:
            frequency_band: Center frequency in Hz
            bandwidth: Bandwidth in Hz
        """
        self.frequency_band = frequency_band
        self.bandwidth = bandwidth
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

class SpatialAnalyzer:
    """Analyzes spatial distribution of ray tracing results."""
    
    def __init__(self, spatial_resolution: float = 0.1):
        """
        Initialize spatial analyzer.
        
        Args:
            spatial_resolution: Spatial resolution in meters
        """
        self.spatial_resolution = spatial_resolution
    
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

class AdvancedRayTracer:
    """Main ray tracing engine that orchestrates all components."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the advanced ray tracer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Extract ray tracing configuration
        ray_config = self.config.get('ray_tracing', {})
        
        # Initialize components
        self.ray_generator = RayGenerator(
            azimuth_samples=ray_config.get('azimuth_samples', 36),
            elevation_samples=ray_config.get('elevation_samples', 18)
        )
        
        self.path_tracer = PathTracer(
            max_reflections=ray_config.get('reflection_order', 3),
            max_diffractions=ray_config.get('max_diffractions', 2)
        )
        
        self.interaction_model = InteractionModel(config)
        self.channel_estimator = ChannelEstimator(
            frequency_band=ray_config.get('frequency_band', 2.4e9),
            bandwidth=ray_config.get('bandwidth', 20e6)
        )
        
        self.spatial_analyzer = SpatialAnalyzer(
            spatial_resolution=ray_config.get('spatial_resolution', 0.1)
        )
        
        logger.info("Advanced Ray Tracer initialized successfully")
    
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
        # Generate rays from source
        rays = self.ray_generator.generate_rays(source_position)
        
        # Trace each ray
        results = []
        for ray in rays:
            result = self.path_tracer.trace_ray(ray, environment)
            results.append(result)
        
        logger.info(f"Traced {len(rays)} rays through environment")
        return results
    
    def analyze_spatial_distribution(self, ray_tracing_results: List[RayPath]) -> Dict[str, Union[np.ndarray, int]]:
        """Analyze spatial distribution of ray tracing results."""
        return self.spatial_analyzer.analyze_spatial_distribution(ray_tracing_results)
    
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
