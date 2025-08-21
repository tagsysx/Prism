"""
CSI Virtual Link Processor for Prism: Wideband RF Neural Radiance Fields.
Handles M×N_UE uplink channel combinations as virtual links for enhanced channel modeling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class CSIVirtualLinkProcessor:
    """
    CSI Virtual Link Processor that treats each M×N_UE uplink channel combination
    as a single virtual link for enhanced MIMO channel modeling.
    
    This processor enables:
    - Enhanced channel modeling for complex MIMO scenarios
    - Improved spatial resolution in multi-path environments
    - Scalable architecture for different deployment scenarios
    - Efficient processing of large channel matrices
    """
    
    def __init__(self, 
                 m_subcarriers: int = 1024,
                 n_ue_antennas: int = 2,
                 n_bs_antennas: int = 4,
                 config: Optional[Dict] = None):
        """
        Initialize the CSI Virtual Link Processor.
        
        Args:
            m_subcarriers: Number of subcarriers
            n_ue_antennas: Number of UE antennas
            n_bs_antennas: Number of BS antennas
            config: Configuration dictionary
        """
        self.m_subcarriers = m_subcarriers
        self.n_ue_antennas = n_ue_antennas
        self.n_bs_antennas = n_bs_antennas
        
        # Calculate virtual link parameters
        self.virtual_link_count = m_subcarriers * n_ue_antennas
        self.uplink_per_bs_antenna = m_subcarriers * n_ue_antennas
        
        # Load configuration
        if config and 'csi_processing' in config:
            self.config = config['csi_processing']
        else:
            self.config = self._get_default_config()
        
        # Initialize processing components
        self._initialize_components()
        
        logger.info(f"CSI Virtual Link Processor initialized:")
        logger.info(f"  M subcarriers: {m_subcarriers}")
        logger.info(f"  N_UE antennas: {n_ue_antennas}")
        logger.info(f"  N_BS antennas: {n_bs_antennas}")
        logger.info(f"  Virtual links: {self.virtual_link_count}")
        logger.info(f"  Uplinks per BS antenna: {self.uplink_per_bs_antenna}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for CSI processing."""
        return {
            'virtual_link_enabled': True,
            'enable_interference_cancellation': True,
            'enable_channel_estimation': True,
            'enable_spatial_filtering': True,
            'batch_processing': True,
            'gpu_acceleration': True,
            'memory_optimization': True,
            'enable_frequency_correlation': True,
            'enable_spatial_correlation': True,
            'enable_temporal_correlation': True
        }
    
    def _initialize_components(self):
        """Initialize processing components based on configuration."""
        # Interference cancellation module
        if self.config.get('enable_interference_cancellation', False):
            self.interference_canceller = InterferenceCanceller(
                self.virtual_link_count,
                self.n_bs_antennas
            )
        
        # Channel estimation module
        if self.config.get('enable_channel_estimation', False):
            self.channel_estimator = ChannelEstimator(
                self.virtual_link_count,
                self.n_bs_antennas
            )
        
        # Spatial filtering module
        if self.config.get('enable_spatial_filtering', False):
            self.spatial_filter = SpatialFilter(
                self.virtual_link_count,
                self.n_bs_antennas
            )
        
        # Correlation analysis modules
        if self.config.get('enable_frequency_correlation', False):
            self.frequency_correlator = FrequencyCorrelator(self.m_subcarriers)
        
        if self.config.get('enable_spatial_correlation', False):
            self.spatial_correlator = SpatialCorrelator(self.n_ue_antennas, self.n_bs_antennas)
        
        if self.config.get('enable_temporal_correlation', False):
            self.temporal_correlator = TemporalCorrelator()
    
    def process_virtual_links(self, 
                             channel_matrix: torch.Tensor,
                             positions: Optional[torch.Tensor] = None,
                             additional_features: Optional[torch.Tensor] = None,
                             sample_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Process M×N_UE uplink channel combinations as virtual links.
        
        Args:
            channel_matrix: Input channel matrix [batch_size, m_subcarriers, n_ue_antennas, n_bs_antennas]
            positions: 3D positions [batch_size, 3] (optional)
            additional_features: Additional features [batch_size, feature_dim] (optional)
            
        Returns:
            Dictionary containing processed virtual links and analysis results
        """
        batch_size = channel_matrix.shape[0]
        
        # Initialize results dictionary first
        results = {
            'virtual_links': None,
            'original_channel_matrix': channel_matrix,
            'virtual_link_count': self.virtual_link_count,
            'uplink_per_bs_antenna': self.uplink_per_bs_antenna
        }
        
        # Reshape channel matrix to virtual link format
        # [batch_size, m_subcarriers, n_ue_antennas, n_bs_antennas] -> [batch_size, virtual_link_count, n_bs_antennas]
        virtual_links = channel_matrix.view(batch_size, self.virtual_link_count, self.n_bs_antennas)
        
        # Apply random sampling if specified
        if sample_size is not None and sample_size < self.virtual_link_count:
            virtual_links, sampled_indices = self._sample_virtual_links(virtual_links, sample_size)
            results['sampled_indices'] = sampled_indices
            results['sampled_virtual_link_count'] = sample_size
        else:
            results['sampled_indices'] = None
            results['sampled_virtual_link_count'] = self.virtual_link_count
        
        # Update virtual_links in results
        results['virtual_links'] = virtual_links
        
        # Apply interference cancellation if enabled
        if hasattr(self, 'interference_canceller'):
            virtual_links = self.interference_canceller.cancel_interference(virtual_links)
            results['interference_cancelled_links'] = virtual_links
        
        # Apply spatial filtering if enabled
        if hasattr(self, 'spatial_filter'):
            virtual_links = self.spatial_filter.apply_filter(virtual_links, positions, additional_features)
            results['spatially_filtered_links'] = virtual_links
        
        # Perform channel estimation if enabled
        if hasattr(self, 'channel_estimator'):
            channel_estimates = self.channel_estimator.estimate_channel(virtual_links)
            results['channel_estimates'] = channel_estimates
        
        # Perform correlation analysis if enabled
        if hasattr(self, 'frequency_correlator'):
            freq_correlation = self.frequency_correlator.analyze_correlation(channel_matrix)
            results['frequency_correlation'] = freq_correlation
        
        if hasattr(self, 'spatial_correlator'):
            spatial_correlation = self.spatial_correlator.analyze_correlation(channel_matrix)
            results['spatial_correlation'] = spatial_correlation
        
        if hasattr(self, 'temporal_correlator'):
            temporal_correlation = self.temporal_correlator.analyze_correlation(virtual_links)
            results['temporal_correlation'] = temporal_correlation
        
        # Update final virtual links
        results['processed_virtual_links'] = virtual_links
        
        return results
    
    def analyze_virtual_links(self, virtual_links: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze characteristics of virtual links.
        
        Args:
            virtual_links: Processed virtual links [batch_size, virtual_link_count, n_bs_antennas]
            
        Returns:
            Dictionary containing analysis results
        """
        batch_size = virtual_links.shape[0]
        
        # Calculate link strength statistics
        link_strengths = torch.norm(virtual_links, dim=-1)  # [batch_size, virtual_link_count]
        
        # Calculate link quality metrics
        link_quality = self._calculate_link_quality(virtual_links)
        
        # Calculate spatial diversity
        spatial_diversity = self._calculate_spatial_diversity(virtual_links)
        
        # Calculate frequency diversity
        frequency_diversity = self._calculate_frequency_diversity(virtual_links)
        
        return {
            'link_strengths': link_strengths,
            'link_quality': link_quality,
            'spatial_diversity': spatial_diversity,
            'frequency_diversity': frequency_diversity,
            'mean_link_strength': torch.mean(link_strengths, dim=1),
            'std_link_strength': torch.std(link_strengths, dim=1),
            'max_link_strength': torch.max(link_strengths, dim=1)[0],
            'min_link_strength': torch.min(link_strengths, dim=1)[0]
        }
    
    def _calculate_link_quality(self, virtual_links: torch.Tensor) -> torch.Tensor:
        """Calculate quality metrics for each virtual link."""
        # Signal-to-noise ratio approximation
        signal_power = torch.mean(torch.abs(virtual_links) ** 2, dim=-1)
        noise_power = torch.var(torch.abs(virtual_links), dim=-1)
        
        # Avoid division by zero
        noise_power = torch.clamp(noise_power, min=1e-10)
        snr = 10 * torch.log10(signal_power / noise_power)
        
        # Normalize to [0, 1] range
        quality = torch.sigmoid(snr / 10)  # Normalize SNR to reasonable range
        
        return quality
    
    def _calculate_spatial_diversity(self, virtual_links: torch.Tensor) -> torch.Tensor:
        """Calculate spatial diversity across BS antennas."""
        # Calculate correlation between BS antennas
        bs_correlations = []
        
        for i in range(self.n_bs_antennas):
            for j in range(i + 1, self.n_bs_antennas):
                corr = torch.corrcoef(
                    virtual_links[:, :, i].flatten(),
                    virtual_links[:, :, j].flatten()
                )
                bs_correlations.append(corr[0, 1])
        
        # Spatial diversity is inversely proportional to correlation
        spatial_diversity = 1.0 - torch.mean(torch.stack(bs_correlations))
        
        return spatial_diversity
    
    def _calculate_frequency_diversity(self, virtual_links: torch.Tensor) -> torch.Tensor:
        """Calculate frequency diversity across subcarriers."""
        # Reshape to [batch_size, m_subcarriers, n_ue_antennas, n_bs_antennas]
        reshaped = virtual_links.view(
            virtual_links.shape[0], 
            self.m_subcarriers, 
            self.n_ue_antennas, 
            self.n_bs_antennas
        )
        
        # Calculate frequency correlation
        freq_correlations = []
        
        for i in range(self.m_subcarriers // 2):
            for j in range(i + 1, min(i + 51, self.m_subcarriers)):  # Limit correlation range
                corr = torch.corrcoef(
                    reshaped[:, i, :, :].flatten(),
                    reshaped[:, j, :, :].flatten()
                )
                freq_correlations.append(corr[0, 1])
        
        # Frequency diversity is inversely proportional to correlation
        frequency_diversity = 1.0 - torch.mean(torch.stack(freq_correlations))
        
        return frequency_diversity
    
    def _sample_virtual_links(self, virtual_links: torch.Tensor, sample_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly sample K virtual links from the total virtual links.
        
        Args:
            virtual_links: Full virtual links tensor [batch_size, virtual_link_count, n_bs_antennas]
            sample_size: Number of virtual links to sample (K)
            
        Returns:
            Tuple of (sampled_virtual_links, sampled_indices)
        """
        batch_size, virtual_link_count, n_bs_antennas = virtual_links.shape
        
        # Generate random indices for sampling
        # Use different random seed for each batch to ensure diversity
        sampled_indices = []
        sampled_virtual_links = []
        
        for i in range(batch_size):
            # Generate random indices for this batch
            indices = torch.randperm(virtual_link_count)[:sample_size]
            sampled_indices.append(indices)
            
            # Sample virtual links for this batch
            batch_virtual_links = virtual_links[i, indices, :]  # [sample_size, n_bs_antennas]
            sampled_virtual_links.append(batch_virtual_links)
        
        # Stack results
        sampled_indices = torch.stack(sampled_indices)  # [batch_size, sample_size]
        sampled_virtual_links = torch.stack(sampled_virtual_links)  # [batch_size, sample_size, n_bs_antennas]
        
        return sampled_virtual_links, sampled_indices
    
    def get_virtual_link_statistics(self, virtual_links: torch.Tensor) -> Dict[str, float]:
        """
        Get comprehensive statistics for virtual links.
        
        Args:
            virtual_links: Virtual links tensor
            
        Returns:
            Dictionary containing statistical measures
        """
        with torch.no_grad():
            # Basic statistics
            mean_strength = torch.mean(virtual_links).item()
            std_strength = torch.std(virtual_links).item()
            max_strength = torch.max(virtual_links).item()
            min_strength = torch.min(virtual_links).item()
            
            # Link quality statistics
            link_quality = self._calculate_link_quality(virtual_links)
            mean_quality = torch.mean(link_quality).item()
            std_quality = torch.std(link_quality).item()
            
            # Diversity statistics
            spatial_diversity = self._calculate_spatial_diversity(virtual_links).item()
            frequency_diversity = self._calculate_frequency_diversity(virtual_links).item()
            
            return {
                'mean_strength': mean_strength,
                'std_strength': std_strength,
                'max_strength': max_strength,
                'min_strength': min_strength,
                'mean_quality': mean_quality,
                'std_quality': std_quality,
                'spatial_diversity': spatial_diversity,
                'frequency_diversity': frequency_diversity
            }


class InterferenceCanceller(nn.Module):
    """Module for canceling interference in virtual links."""
    
    def __init__(self, virtual_link_count: int, n_bs_antennas: int):
        super().__init__()
        self.virtual_link_count = virtual_link_count
        self.n_bs_antennas = n_bs_antennas
        
        # Interference cancellation network
        self.interference_net = nn.Sequential(
            nn.Linear(virtual_link_count * n_bs_antennas, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, virtual_link_count * n_bs_antennas)
        )
    
    def forward(self, virtual_links: torch.Tensor) -> torch.Tensor:
        """Cancel interference in virtual links."""
        batch_size = virtual_links.shape[0]
        
        # Flatten for processing
        flattened = virtual_links.view(batch_size, -1)
        
        # Apply interference cancellation
        interference_cancelled = self.interference_net(flattened)
        
        # Reshape back
        return interference_cancelled.view(batch_size, self.virtual_link_count, self.n_bs_antennas)
    
    def cancel_interference(self, virtual_links: torch.Tensor) -> torch.Tensor:
        """Cancel interference using the trained network."""
        return self.forward(virtual_links)


class ChannelEstimator(nn.Module):
    """Module for enhanced channel estimation."""
    
    def __init__(self, virtual_link_count: int, n_bs_antennas: int):
        super().__init__()
        self.virtual_link_count = virtual_link_count
        self.n_bs_antennas = n_bs_antennas
        
        # Channel estimation network
        self.estimation_net = nn.Sequential(
            nn.Linear(virtual_link_count * n_bs_antennas, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, virtual_link_count * n_bs_antennas)
        )
    
    def forward(self, virtual_links: torch.Tensor) -> torch.Tensor:
        """Estimate enhanced channel characteristics."""
        batch_size = virtual_links.shape[0]
        
        # Flatten for processing
        flattened = virtual_links.view(batch_size, -1)
        
        # Apply channel estimation
        enhanced_channel = self.estimation_net(flattened)
        
        # Reshape back
        return enhanced_channel.view(batch_size, self.virtual_link_count, self.n_bs_antennas)
    
    def estimate_channel(self, virtual_links: torch.Tensor) -> torch.Tensor:
        """Estimate channel using the trained network."""
        return self.forward(virtual_links)


class SpatialFilter(nn.Module):
    """Module for spatial filtering of virtual links."""
    
    def __init__(self, virtual_link_count: int, n_bs_antennas: int):
        super().__init__()
        self.virtual_link_count = virtual_link_count
        self.n_bs_antennas = n_bs_antennas
        
        # Spatial filtering network
        self.spatial_net = nn.Sequential(
            nn.Linear(virtual_link_count * n_bs_antennas + 3, 512),  # +3 for position
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, virtual_link_count * n_bs_antennas)
        )
    
    def forward(self, virtual_links: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply spatial filtering to virtual links."""
        batch_size = virtual_links.shape[0]
        
        # Flatten virtual links
        flattened = virtual_links.view(batch_size, -1)
        
        # Concatenate with position information
        if positions is not None:
            combined = torch.cat([flattened, positions], dim=1)
        else:
            # Use zero positions if not provided
            zero_positions = torch.zeros(batch_size, 3, device=virtual_links.device)
            combined = torch.cat([flattened, zero_positions], dim=1)
        
        # Apply spatial filtering
        filtered = self.spatial_net(combined)
        
        # Reshape back
        return filtered.view(batch_size, self.virtual_link_count, self.n_bs_antennas)
    
    def apply_filter(self, virtual_links: torch.Tensor, 
                    positions: Optional[torch.Tensor] = None,
                    additional_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply spatial filter to virtual links."""
        return self.forward(virtual_links, positions)


class FrequencyCorrelator:
    """Analyze frequency correlation across subcarriers."""
    
    def __init__(self, m_subcarriers: int):
        self.m_subcarriers = m_subcarriers
    
    def analyze_correlation(self, channel_matrix: torch.Tensor) -> torch.Tensor:
        """Analyze frequency correlation in channel matrix."""
        # Calculate correlation between adjacent subcarriers
        correlations = []
        
        for i in range(self.m_subcarriers - 1):
            corr = torch.corrcoef(
                channel_matrix[:, i, :, :].flatten(),
                channel_matrix[:, i + 1, :, :].flatten()
            )
            correlations.append(corr[0, 1])
        
        return torch.stack(correlations)


class SpatialCorrelator:
    """Analyze spatial correlation across antennas."""
    
    def __init__(self, n_ue_antennas: int, n_bs_antennas: int):
        self.n_ue_antennas = n_ue_antennas
        self.n_bs_antennas = n_bs_antennas
    
    def analyze_correlation(self, channel_matrix: torch.Tensor) -> torch.Tensor:
        """Analyze spatial correlation in channel matrix."""
        # Calculate correlation between different antenna pairs
        correlations = []
        
        # UE antenna correlations
        for i in range(self.n_ue_antennas):
            for j in range(i + 1, self.n_ue_antennas):
                corr = torch.corrcoef(
                    channel_matrix[:, :, i, :].flatten(),
                    channel_matrix[:, :, j, :].flatten()
                )
                correlations.append(corr[0, 1])
        
        # BS antenna correlations
        for i in range(self.n_bs_antennas):
            for j in range(i + 1, self.n_bs_antennas):
                corr = torch.corrcoef(
                    channel_matrix[:, :, :, i].flatten(),
                    channel_matrix[:, :, :, j].flatten()
                )
                correlations.append(corr[0, 1])
        
        return torch.stack(correlations)


class TemporalCorrelator:
    """Analyze temporal correlation across time instances."""
    
    def __init__(self):
        pass
    
    def analyze_correlation(self, virtual_links: torch.Tensor) -> torch.Tensor:
        """Analyze temporal correlation in virtual links."""
        # For now, return a simple temporal correlation measure
        # In practice, this would analyze correlation across time instances
        
        # Calculate temporal stability (variance across batch)
        temporal_stability = torch.var(virtual_links, dim=0)
        
        # Convert to correlation-like measure (inverse of variance)
        temporal_correlation = 1.0 / (1.0 + temporal_stability)
        
        return temporal_correlation
