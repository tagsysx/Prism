"""
RF Signal Processing for Discrete Electromagnetic Ray Tracing

This module implements RF signal processing components for the discrete
electromagnetic ray tracing system, including signal strength calculation,
subcarrier selection, and virtual link processing.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import random
import math

logger = logging.getLogger(__name__)

class SubcarrierSelector:
    """Intelligent subcarrier selection for computational efficiency."""
    
    def __init__(self, total_subcarriers: int, sampling_ratio: float = 0.5):
        self.total_subcarriers = total_subcarriers
        self.sampling_ratio = sampling_ratio
        self.num_selected = int(total_subcarriers * sampling_ratio)
        
        logger.info(f"Subcarrier selector: {self.num_selected}/{total_subcarriers} subcarriers (α={sampling_ratio})")
    
    def select_subcarriers(self, ue_positions: List) -> Dict:
        """Randomly select a subset of subcarriers for each UE."""
        selected_subcarriers = {}
        
        for ue_pos in ue_positions:
            ue_subcarriers = random.sample(range(self.total_subcarriers), self.num_selected)
            selected_subcarriers[tuple(ue_pos)] = ue_subcarriers
        
        return selected_subcarriers

class SignalStrengthCalculator:
    """Calculator for RF signal strength based on ray tracing results."""
    
    def __init__(self, frequency_band: float = 2.4e9, device: str = 'cpu'):
        self.frequency_band = frequency_band
        self.device = device
        self.wavelength = 3e8 / frequency_band
    
    def calculate_signal_strength(self, ray_path: torch.Tensor,
                                antenna_embedding: torch.Tensor,
                                material_properties: Dict,
                                subcarrier_frequency: float) -> float:
        """Calculate RF signal strength for a given ray path."""
        if len(ray_path) < 2:
            return 0.0
        
        # Calculate path loss
        path_loss = self._calculate_path_loss(ray_path, subcarrier_frequency)
        
        # Calculate antenna factor
        antenna_factor = self._calculate_antenna_factor(antenna_embedding)
        
        # Combine factors
        signal_strength = antenna_factor / path_loss
        
        return signal_strength.item()
    
    def _calculate_path_loss(self, ray_path: torch.Tensor, frequency: float) -> torch.Tensor:
        """Calculate free space path loss along the ray path."""
        distances = torch.norm(ray_path[1:] - ray_path[:-1], dim=1)
        total_distance = torch.sum(distances)
        
        wavelength = 3e8 / frequency
        path_loss = (4 * math.pi * total_distance / wavelength) ** 2
        
        return path_loss
    
    def _calculate_antenna_factor(self, antenna_embedding: torch.Tensor) -> float:
        """Calculate antenna factor from embedding parameter."""
        antenna_norm = torch.norm(antenna_embedding)
        antenna_factor = antenna_norm / math.sqrt(antenna_embedding.numel())
        antenna_factor = math.tanh(antenna_factor * 2)
        
        return antenna_factor

class RFSignalProcessor:
    """Main RF signal processor for the ray tracing system."""
    
    def __init__(self, total_subcarriers: int = 408, sampling_ratio: float = 0.5,
                 frequency_band: float = 2.4e9, device: str = 'cpu'):
        self.device = device
        self.subcarrier_selector = SubcarrierSelector(total_subcarriers, sampling_ratio)
        self.signal_calculator = SignalStrengthCalculator(frequency_band, device)
    
    def process_virtual_links(self, base_station_pos: torch.Tensor,
                            ue_positions: List[torch.Tensor],
                            antenna_embedding: torch.Tensor,
                            ray_tracing_results: Dict) -> Dict:
        """Process virtual links for RF signal computation."""
        selected_subcarriers = self.subcarrier_selector.select_subcarriers(ue_positions)
        
        virtual_link_results = {}
        
        for ue_pos in ue_positions:
            ue_pos_key = tuple(ue_pos)
            ue_subcarriers = selected_subcarriers[ue_pos_key]
            
            for subcarrier_idx in ue_subcarriers:
                if (ue_pos_key, subcarrier_idx) in ray_tracing_results:
                    ray_path = ray_tracing_results[(ue_pos_key, subcarrier_idx)]
                    
                    signal_strength = self.signal_calculator.calculate_signal_strength(
                        ray_path, antenna_embedding, {}, 2.4e9
                    )
                    
                    virtual_link_results[(ue_pos_key, subcarrier_idx)] = {
                        'signal_strength': signal_strength,
                        'ray_path': ray_path
                    }
        
        return virtual_link_results
