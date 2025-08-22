"""
Signal processing utility functions for the Prism system.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class SignalProcessingUtils:
    """Utility functions for signal processing operations."""
    
    @staticmethod
    def calculate_snr(signal_power: float, noise_power: float) -> float:
        """Calculate Signal-to-Noise Ratio (SNR) in dB."""
        if noise_power <= 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def calculate_path_loss(distance: float, frequency: float) -> float:
        """Calculate free space path loss in dB."""
        wavelength = 3e8 / frequency
        path_loss_db = 20 * np.log10(4 * np.pi * distance / wavelength)
        return path_loss_db
    
    @staticmethod
    def optimize_subcarrier_sampling(total_subcarriers: int, 
                                   target_complexity: float) -> int:
        """Calculate optimal number of subcarriers to sample."""
        optimal_subcarriers = int(total_subcarriers * target_complexity)
        return max(1, optimal_subcarriers)
