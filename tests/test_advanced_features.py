"""
Tests for advanced features: CSI virtual links and ray tracing.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from prism.csi_processor import CSIVirtualLinkProcessor
from prism.ray_tracer import (
    AdvancedRayTracer, Environment, Building, Plane, 
    Ray, RayGenerator, PathTracer
)
from prism.model import create_prism_model

class TestCSIVirtualLinkProcessor:
    """Test CSI virtual link processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'csi_processing': {
                'virtual_link_enabled': True,
                'enable_interference_cancellation': True,
                'enable_channel_estimation': True,
                'enable_spatial_filtering': True,
                'enable_frequency_correlation': True,
                'enable_spatial_correlation': True,
                'enable_temporal_correlation': True
            }
        }
        
        self.processor = CSIVirtualLinkProcessor(
            m_subcarriers=64,  # Smaller for testing
            n_ue_antennas=2,
            n_bs_antennas=4,
            config=self.config
        )
        
        # Create sample data
        self.batch_size = 8
        self.channel_matrix = torch.randn(
            self.batch_size, 64, 2, 4
        )
        self.positions = torch.randn(self.batch_size, 3)
        self.additional_features = torch.randn(self.batch_size, 10)
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.m_subcarriers == 64
        assert self.processor.n_ue_antennas == 2
        assert self.processor.n_bs_antennas == 4
        assert self.processor.virtual_link_count == 128  # 64 * 2
        assert self.processor.uplink_per_bs_antenna == 128
    
    def test_virtual_link_processing(self):
        """Test virtual link processing."""
        results = self.processor.process_virtual_links(
            self.channel_matrix,
            self.positions,
            self.additional_features
        )
        
        # Check basic outputs
        assert 'virtual_links' in results
        assert 'original_channel_matrix' in results
        assert 'virtual_link_count' in results
        assert 'uplink_per_bs_antenna' in results
        
        # Check shapes
        assert results['virtual_links'].shape == (self.batch_size, 128, 4)
        assert results['original_channel_matrix'].shape == self.channel_matrix.shape
        
        # Check advanced outputs
        assert 'interference_cancelled_links' in results
        assert 'spatially_filtered_links' in results
        assert 'channel_estimates' in results
        assert 'frequency_correlation' in results
        assert 'spatial_correlation' in results
        assert 'temporal_correlation' in results
    
    def test_virtual_link_sampling(self):
        """Test virtual link sampling functionality."""
        sample_size = 32  # Sample K=32 links
        
        results = self.processor.process_virtual_links(
            self.channel_matrix,
            self.positions,
            self.additional_features,
            sample_size=sample_size
        )
        
        # Check sampling outputs
        assert 'sampled_indices' in results
        assert 'sampled_virtual_link_count' in results
        
        # Check shapes
        assert results['virtual_links'].shape == (self.batch_size, sample_size, 4)
        assert results['sampled_indices'].shape == (self.batch_size, sample_size)
        assert results['sampled_virtual_link_count'] == sample_size
        
        # Check that sampled indices are within valid range
        assert torch.all(results['sampled_indices'] >= 0)
        assert torch.all(results['sampled_indices'] < 128)
        
        # Check that all indices in a batch are unique
        for i in range(self.batch_size):
            batch_indices = results['sampled_indices'][i]
            assert len(torch.unique(batch_indices)) == sample_size
    
    def test_virtual_link_analysis(self):
        """Test virtual link analysis."""
        # Process virtual links first
        results = self.processor.process_virtual_links(
            self.channel_matrix,
            self.positions,
            self.additional_features
        )
        
        # Analyze virtual links
        analysis = self.processor.analyze_virtual_links(results['virtual_links'])
        
        # Check analysis outputs
        assert 'link_strengths' in analysis
        assert 'link_quality' in analysis
        assert 'spatial_diversity' in analysis
        assert 'frequency_diversity' in analysis
        
        # Check shapes
        assert analysis['link_strengths'].shape == (self.batch_size, 128)
        assert analysis['link_quality'].shape == (self.batch_size, 128)
    
    def test_statistics(self):
        """Test statistics computation."""
        # Process virtual links first
        results = self.processor.process_virtual_links(
            self.channel_matrix,
            self.positions,
            self.additional_features
        )
        
        # Get statistics
        stats = self.processor.get_virtual_link_statistics(results['virtual_links'])
        
        # Check statistics
        assert 'mean_strength' in stats
        assert 'std_strength' in stats
        assert 'max_strength' in stats
        assert 'min_strength' in stats
        assert 'mean_quality' in stats
        assert 'std_quality' in stats
        assert 'spatial_diversity' in stats
        assert 'frequency_diversity' in stats
        
        # Check that statistics are reasonable
        assert 0 <= stats['spatial_diversity'] <= 1
        assert 0 <= stats['frequency_diversity'] <= 1

class TestRayTracing:
    """Test ray tracing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'ray_tracing': {
                'enabled': True,
                'azimuth_samples': 8,  # Smaller for testing
                'elevation_samples': 4,
                'points_per_ray': 16,
                'reflection_order': 2,
                'max_diffractions': 1,
                'spatial_resolution': 0.5
            }
        }
        
        self.environment = Environment()
        
        # Add a simple building
        building = Building(
            min_corner=[-10, -10, 0],
            max_corner=[10, 10, 10],
            material='concrete'
        )
        self.environment.add_obstacle(building)
    
    def test_ray_generator(self):
        """Test ray generation."""
        generator = RayGenerator(8, 4)
        
        assert generator.azimuth_samples == 8
        assert generator.elevation_samples == 4
        assert len(generator.angle_combinations) == 32  # 8 * 4
        
        # Test ray generation
        source_position = [0, 0, 5]
        rays = generator.generate_rays(source_position)
        
        assert len(rays) == 32
        
        # Check first ray
        first_ray = rays[0]
        assert torch.allclose(first_ray.origin, torch.tensor(source_position))
        assert torch.norm(first_ray.direction) == pytest.approx(1.0, abs=1e-6)
    
    def test_ray_intersection(self):
        """Test ray-obstacle intersection."""
        # Create a simple plane
        plane = Plane([0, 0, 5], [0, 0, 1], 'concrete')
        
        # Create a ray pointing upward
        ray = Ray([0, 0, 0], [0, 0, 1])
        
        # Test intersection
        intersection = plane.intersect(ray)
        
        assert intersection is not None
        assert intersection.distance == pytest.approx(5.0, abs=1e-6)
        assert intersection.material == 'concrete'
        assert intersection.interaction_type == 'reflection'
    
    def test_building_intersection(self):
        """Test building intersection."""
        # Create a ray pointing toward building
        ray = Ray([0, 0, 5], [1, 0, 0])
        
        # Test intersection
        intersection = self.environment.ray_intersection(ray)
        
        assert len(intersection) > 0
        assert intersection[0].material == 'concrete'
    
    def test_path_tracing(self):
        """Test path tracing."""
        tracer = PathTracer(max_reflections=2, max_diffractions=1)
        
        # Create a ray
        ray = Ray([0, 0, 5], [1, 0, 0])
        
        # Trace the ray
        result = tracer.trace_ray(ray, self.environment)
        
        assert result is not None
        assert len(result.path_points) > 1
        assert result.total_length > 0
    
    def test_advanced_ray_tracer(self):
        """Test the complete advanced ray tracer."""
        ray_tracer = AdvancedRayTracer(self.config)
        
        # Test ray tracing
        source_position = [0, 0, 5]
        target_positions = [[10, 0, 5], [-10, 0, 5]]
        
        ray_results = ray_tracer.trace_rays(
            source_position, target_positions, self.environment
        )
        
        assert len(ray_results) > 0
        
        # Test spatial analysis
        spatial_analysis = ray_tracer.analyze_spatial_distribution(ray_results)
        assert 'total_points' in spatial_analysis
        
        # Test statistics
        statistics = ray_tracer.get_ray_statistics(ray_results)
        assert 'total_rays' in statistics
        assert statistics['total_rays'] > 0

class TestIntegratedModel:
    """Test integrated model with advanced features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'model': {
                'num_subcarriers': 64,
                'num_ue_antennas': 2,
                'num_bs_antennas': 4,
                'position_dim': 3,
                'hidden_dim': 128
            },
            'csi_processing': {
                'virtual_link_enabled': True
            },
            'ray_tracing': {
                'enabled': True,
                'azimuth_samples': 8,
                'elevation_samples': 4,
                'points_per_ray': 16
            }
        }
    
    def test_model_creation(self):
        """Test model creation with advanced features."""
        model = create_prism_model(self.config)
        
        # Check that advanced features are enabled
        assert hasattr(model, 'csi_processor')
        assert model.csi_processor is not None
        assert hasattr(model, 'ray_tracer')
        assert model.ray_tracer is not None
    
    def test_csi_processing_integration(self):
        """Test CSI processing integration in model."""
        model = create_prism_model(self.config)
        
        # Create sample data
        batch_size = 4
        positions = torch.randn(batch_size, 3)
        ue_antennas = torch.randn(batch_size, 2)
        bs_antennas = torch.randn(batch_size, 4)
        additional_features = torch.randn(batch_size, 10)
        
        # Run forward pass
        with torch.no_grad():
            outputs = model(
                positions=positions,
                ue_antennas=ue_antennas,
                bs_antennas=bs_antennas,
                additional_features=additional_features
            )
        
        # Check that CSI outputs are present
        assert 'csi_virtual_links' in outputs
        assert 'csi_analysis' in outputs
        
        # Check shapes
        assert outputs['csi_virtual_links'].shape == (batch_size, 128, 4)
    
    def test_ray_tracing_integration(self):
        """Test ray tracing integration in model."""
        model = create_prism_model(self.config)
        
        # Test ray tracing
        source_position = [0, 0, 5]
        ray_results = model.trace_rays(source_position)
        
        assert 'ray_paths' in ray_results
        assert 'spatial_analysis' in ray_results
        assert 'statistics' in ray_results
        
        # Check that we have ray paths
        assert len(ray_results['ray_paths']) > 0
    
    def test_virtual_link_statistics(self):
        """Test virtual link statistics from model."""
        model = create_prism_model(self.config)
        
        # Create sample data
        batch_size = 4
        positions = torch.randn(batch_size, 3)
        ue_antennas = torch.randn(batch_size, 2)
        bs_antennas = torch.randn(batch_size, 4)
        additional_features = torch.randn(batch_size, 10)
        
        # Get virtual link statistics
        stats = model.get_virtual_link_statistics(
            positions=positions,
            ue_antennas=ue_antennas,
            bs_antennas=bs_antennas,
            additional_features=additional_features
        )
        
        # Check that we have statistics
        assert 'mean_strength' in stats
        assert 'spatial_diversity' in stats
        assert 'frequency_diversity' in stats

if __name__ == '__main__':
    pytest.main([__file__])
