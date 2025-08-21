"""
Utility functions for OFDM signal processing in Prism.
Provides tools for working with wideband RF signals and subcarrier management.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class OFDMSignalProcessor:
    """
    Utility class for OFDM signal processing operations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the OFDM signal processor.
        
        Args:
            config: Configuration dictionary containing OFDM parameters
        """
        self.config = config
        
        # Extract OFDM parameters
        self.num_subcarriers = config['model']['num_subcarriers']
        self.center_frequency = config['ofdm']['center_frequency']
        self.bandwidth = config['ofdm']['bandwidth']
        self.subcarrier_spacing = config['ofdm']['subcarrier_spacing']
        self.cyclic_prefix = config['ofdm']['cyclic_prefix']
        self.pilot_density = config['ofdm']['pilot_density']
        
        # Calculate frequency array
        self.frequencies = self._calculate_frequencies()
        
        # Calculate pilot subcarrier indices
        self.pilot_indices = self._calculate_pilot_indices()
        
        # Calculate data subcarrier indices
        self.data_indices = self._calculate_data_indices()
    
    def _calculate_frequencies(self) -> np.ndarray:
        """Calculate frequency values for each subcarrier."""
        start_freq = self.center_frequency - self.bandwidth / 2
        end_freq = self.center_frequency + self.bandwidth / 2
        
        frequencies = np.linspace(start_freq, end_freq, self.num_subcarriers)
        return frequencies
    
    def _calculate_pilot_indices(self) -> np.ndarray:
        """Calculate indices of pilot subcarriers."""
        num_pilots = int(self.num_subcarriers * self.pilot_density)
        pilot_indices = np.linspace(0, self.num_subcarriers - 1, num_pilots, dtype=int)
        return pilot_indices
    
    def _calculate_data_indices(self) -> np.ndarray:
        """Calculate indices of data subcarriers."""
        all_indices = np.arange(self.num_subcarriers)
        data_indices = np.setdiff1d(all_indices, self.pilot_indices)
        return data_indices
    
    def generate_pilot_sequence(self, length: Optional[int] = None) -> np.ndarray:
        """
        Generate pilot sequence for channel estimation.
        
        Args:
            length: Length of pilot sequence (default: number of pilot subcarriers)
            
        Returns:
            Pilot sequence array
        """
        if length is None:
            length = len(self.pilot_indices)
        
        # Generate random QPSK pilot symbols
        pilot_sequence = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=length)
        pilot_sequence = pilot_sequence / np.sqrt(2)  # Normalize power
        
        return pilot_sequence
    
    def apply_frequency_selective_channel(self, signal: np.ndarray, 
                                        channel_response: np.ndarray) -> np.ndarray:
        """
        Apply frequency-selective channel to OFDM signal.
        
        Args:
            signal: Input OFDM signal [num_subcarriers]
            channel_response: Channel frequency response [num_subcarriers]
            
        Returns:
            Channel-affected signal
        """
        # Element-wise multiplication in frequency domain
        output_signal = signal * channel_response
        
        # Add noise (simplified)
        noise_power = 0.01  # Adjust based on SNR requirements
        noise = np.random.normal(0, np.sqrt(noise_power/2), output_signal.shape) + \
                1j * np.random.normal(0, np.sqrt(noise_power/2), output_signal.shape)
        
        return output_signal + noise
    
    def estimate_channel_from_pilots(self, received_pilots: np.ndarray,
                                   transmitted_pilots: np.ndarray) -> np.ndarray:
        """
        Estimate channel response from pilot measurements.
        
        Args:
            received_pilots: Received pilot signals
            transmitted_pilots: Transmitted pilot signals
            
        Returns:
            Estimated channel response for all subcarriers
        """
        # Calculate channel response at pilot positions
        pilot_channel = received_pilots / transmitted_pilots
        
        # Interpolate to all subcarriers
        channel_response = np.interp(
            np.arange(self.num_subcarriers),
            self.pilot_indices,
            np.abs(pilot_channel)
        )
        
        return channel_response
    
    def calculate_snr_per_subcarrier(self, signal_power: np.ndarray,
                                   noise_power: float) -> np.ndarray:
        """
        Calculate SNR for each subcarrier.
        
        Args:
            signal_power: Signal power per subcarrier
            noise_power: Noise power
            
        Returns:
            SNR per subcarrier in dB
        """
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        return snr_db
    
    def apply_power_allocation(self, subcarrier_responses: np.ndarray,
                             power_constraint: float = 1.0) -> np.ndarray:
        """
        Apply water-filling power allocation across subcarriers.
        
        Args:
            subcarrier_responses: Channel responses per subcarrier
            power_constraint: Total power constraint
            
        Returns:
            Power allocation per subcarrier
        """
        # Sort subcarriers by channel quality (descending)
        sorted_indices = np.argsort(subcarrier_responses)[::-1]
        sorted_responses = subcarrier_responses[sorted_indices]
        
        # Water-filling algorithm
        num_subcarriers = len(subcarrier_responses)
        power_allocation = np.zeros(num_subcarriers)
        
        # Find water level
        water_level = 0
        for i in range(num_subcarriers):
            # Calculate required power for this subcarrier
            required_power = 1.0 / sorted_responses[i] - 1.0 / sorted_responses[0]
            
            if required_power <= power_constraint:
                water_level = 1.0 / sorted_responses[i]
                break
        
        # Allocate power
        for i in range(num_subcarriers):
            if i < len(sorted_responses):
                power_allocation[sorted_indices[i]] = max(0, water_level - 1.0 / sorted_responses[i])
        
        # Normalize to meet power constraint
        total_power = np.sum(power_allocation)
        if total_power > 0:
            power_allocation = power_allocation * (power_constraint / total_power)
        
        return power_allocation

class MIMOChannelProcessor:
    """
    Utility class for MIMO channel processing operations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the MIMO channel processor.
        
        Args:
            config: Configuration dictionary containing MIMO parameters
        """
        self.config = config
        
        # Extract MIMO parameters
        self.num_ue_antennas = config['model']['num_ue_antennas']
        self.num_bs_antennas = config['model']['num_bs_antennas']
        self.mimo_type = config['ofdm']['mimo_type']
        self.precoding = config['ofdm']['precoding']
        self.detection = config['ofdm']['detection']
    
    def generate_mimo_channel_matrix(self, num_subcarriers: int,
                                   correlation: float = 0.3) -> np.ndarray:
        """
        Generate MIMO channel matrix for multiple subcarriers.
        
        Args:
            num_subcarriers: Number of subcarriers
            correlation: Antenna correlation coefficient
            
        Returns:
            MIMO channel matrix [num_subcarriers, num_ue_antennas, num_bs_antennas]
        """
        # Generate uncorrelated channel coefficients
        channel_matrix = np.random.normal(
            0, 1/np.sqrt(2),
            size=(num_subcarriers, self.num_ue_antennas, self.num_bs_antennas)
        ) + 1j * np.random.normal(
            0, 1/np.sqrt(2),
            size=(num_subcarriers, self.num_ue_antennas, self.num_bs_antennas)
        )
        
        # Apply antenna correlation
        if correlation > 0:
            correlation_matrix_ue = self._create_correlation_matrix(self.num_ue_antennas, correlation)
            correlation_matrix_bs = self._create_correlation_matrix(self.num_bs_antennas, correlation)
            
            for k in range(num_subcarriers):
                channel_matrix[k] = np.sqrt(correlation_matrix_ue) @ channel_matrix[k] @ np.sqrt(correlation_matrix_bs)
        
        return channel_matrix
    
    def _create_correlation_matrix(self, num_antennas: int, correlation: float) -> np.ndarray:
        """Create antenna correlation matrix."""
        correlation_matrix = np.eye(num_antennas)
        for i in range(num_antennas):
            for j in range(num_antennas):
                if i != j:
                    correlation_matrix[i, j] = correlation ** abs(i - j)
        return correlation_matrix
    
    def apply_precoding(self, signal: np.ndarray, channel_matrix: np.ndarray,
                       precoding_type: str = 'zero_forcing') -> np.ndarray:
        """
        Apply precoding to the transmitted signal.
        
        Args:
            signal: Input signal [num_subcarriers, num_ue_antennas]
            channel_matrix: MIMO channel matrix
            precoding_type: Type of precoding
            
        Returns:
            Precoded signal [num_subcarriers, num_bs_antennas]
        """
        num_subcarriers = signal.shape[0]
        precoded_signal = np.zeros((num_subcarriers, self.num_bs_antennas), dtype=complex)
        
        for k in range(num_subcarriers):
            H = channel_matrix[k]  # [num_ue_antennas, num_bs_antennas]
            
            if precoding_type == 'zero_forcing':
                # Zero-forcing precoding
                W = H.conj().T @ np.linalg.inv(H @ H.conj().T)
            elif precoding_type == 'matched_filter':
                # Matched filter precoding
                W = H.conj().T
            else:
                # No precoding
                W = np.eye(self.num_bs_antennas)
            
            # Normalize precoding matrix
            W = W / np.sqrt(np.trace(W @ W.conj().T))
            
            # Apply precoding
            precoded_signal[k] = W.conj().T @ signal[k]
        
        return precoded_signal
    
    def calculate_channel_capacity(self, channel_matrix: np.ndarray,
                                 snr: float) -> np.ndarray:
        """
        Calculate channel capacity for each subcarrier.
        
        Args:
            channel_matrix: MIMO channel matrix
            snr: Signal-to-noise ratio
            
        Returns:
            Channel capacity per subcarrier in bits/s/Hz
        """
        num_subcarriers = channel_matrix.shape[0]
        capacity = np.zeros(num_subcarriers)
        
        for k in range(num_subcarriers):
            H = channel_matrix[k]
            
            # Calculate singular values
            singular_values = np.linalg.svd(H, compute_uv=False)
            
            # Calculate capacity using water-filling
            for sv in singular_values:
                if sv > 0:
                    capacity[k] += np.log2(1 + snr * sv**2)
        
        return capacity

def create_ofdm_processor(config: Dict) -> OFDMSignalProcessor:
    """
    Factory function to create an OFDM signal processor.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured OFDMSignalProcessor instance
    """
    return OFDMSignalProcessor(config)

def create_mimo_processor(config: Dict) -> MIMOChannelProcessor:
    """
    Factory function to create a MIMO channel processor.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured MIMOChannelProcessor instance
    """
    return MIMOChannelProcessor(config)
