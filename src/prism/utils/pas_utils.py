"""
Power Angular Spectrum (PAS) Utilities

This module provides utilities for computing spatial spectra and power angular spectra
from CSI data using beamforming techniques.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def _compute_pas(csi_batch: torch.Tensor, array_shape: tuple, wavelengths: torch.Tensor, 
                 ignore_amplitude: bool = False, azimuth_divisions: int = 361, 
                 elevation_divisions: int = 91) -> torch.Tensor:
    """
    Internal function to compute Power Angular Spectrum (PAS) for a batch of CSI samples on GPU
    
    This is the core implementation that computes PAS for each subcarrier separately, 
    then averages across subcarriers to get a robust PAS estimate.
    
    Note: This is an internal function. Use compute_pas() instead for the public API.
    
    Args:
        csi_batch: CSI tensor of shape [batch_size, num_antennas, num_subcarriers]
        array_shape: Antenna array shape tuple (length 1 for linear array, length 2 for planar array)
        wavelengths: Wavelength tensor of shape [num_subcarriers] containing wavelength for each subcarrier
        ignore_amplitude: Whether to ignore amplitude
        azimuth_divisions: Number of azimuth angle divisions (default: 361 for 0-360° with 1° resolution)
        elevation_divisions: Number of elevation angle divisions (default: 91 for 0-90° with 1° resolution)
        
    Returns:
        pas: Tensor of shape [batch_size, azimuth_angles, elevation_angles] for planar arrays
             or [batch_size, azimuth_angles] for linear arrays
             Averaged across all subcarriers for robust estimation
    
    Note:
        Antenna spacing is automatically set to half wavelength (λ/2) for optimal array performance.
        Each subcarrier uses its specific wavelength for accurate phase calculations.
        Array type is automatically determined from array_shape length.
    """
    batch_size, num_antennas, num_subcarriers = csi_batch.shape
    device = csi_batch.device
    
    # Validate wavelengths tensor
    if wavelengths.shape[0] != num_subcarriers:
        raise ValueError(f"Wavelengths tensor shape {wavelengths.shape} doesn't match num_subcarriers {num_subcarriers}")
    
    # Ensure wavelengths are on the same device
    wavelengths = wavelengths.to(device)
    
    # Apply amplitude handling if needed
    if ignore_amplitude:
        csi_batch = torch.exp(1j * torch.angle(csi_batch))
    
    # Phase alignment: normalize by the first antenna to remove common phase offset
    # This ensures that the first antenna has phase 0, improving spatial spectrum accuracy
    first_antenna_phase = torch.angle(csi_batch[:, 0:1, :])  # [batch_size, 1, num_subcarriers]
    csi_batch = csi_batch * torch.exp(-1j * first_antenna_phase)  # Align all antennas to first antenna phase
    
    # Determine array type from array_shape length
    is_2d_array = len(array_shape) == 2
    
    if is_2d_array:
        # 2D spatial spectrum
        num_antennas_x, num_antennas_y = array_shape
        
        # Create angle grids
        azimuth_angles = torch.linspace(0, 360, azimuth_divisions, dtype=torch.float32, device=device)
        elevation_angles = torch.linspace(0, 90, elevation_divisions, dtype=torch.float32, device=device)
        
        # Convert to radians
        az_rad = azimuth_angles * torch.pi / 180.0
        el_rad = elevation_angles * torch.pi / 180.0
        
        
        az_grid, el_grid = torch.meshgrid(az_rad, el_rad, indexing='ij')
        
        # Initialize output tensor
        pas_batch = torch.zeros(batch_size, len(azimuth_angles), len(elevation_angles), device=device)
        
        # Process each sample
        for i in range(batch_size):
            # Initialize accumulator for this sample
            sample_spectrum_sum = torch.zeros(len(azimuth_angles), len(elevation_angles), device=device)
            
            # Process each subcarrier
            for subcarrier_idx in range(num_subcarriers):
                # Get CSI and wavelength for this subcarrier
                csi_subcarrier = csi_batch[i, :, subcarrier_idx]  # [num_antennas]
                wavelength_subcarrier = wavelengths[subcarrier_idx]  # scalar
                
                # Calculate antenna spacing for this subcarrier (half wavelength)
                spacing_x = wavelength_subcarrier / 2
                spacing_y = wavelength_subcarrier / 2
                
                # Update antenna positions for this subcarrier
                antenna_positions_x = torch.arange(num_antennas_x, dtype=torch.float32, device=device) * spacing_x
                antenna_positions_y = torch.arange(num_antennas_y, dtype=torch.float32, device=device) * spacing_y
                
                # Create meshgrids for this subcarrier
                pos_x, pos_y = torch.meshgrid(antenna_positions_x, antenna_positions_y, indexing='ij')
                pos_x_flat = pos_x.flatten()
                pos_y_flat = pos_y.flatten()
                
                # Compute phase progression for all angle pairs
                phase_progression = 2 * torch.pi * (
                    pos_x_flat.unsqueeze(0).unsqueeze(0) * torch.sin(el_grid).unsqueeze(2) * torch.cos(az_grid).unsqueeze(2) + 
                    pos_y_flat.unsqueeze(0).unsqueeze(0) * torch.sin(el_grid).unsqueeze(2) * torch.sin(az_grid).unsqueeze(2)
                ) / wavelength_subcarrier
                
                # Create steering vectors
                steering_vectors = torch.exp(1j * phase_progression)  # [num_azimuth, num_elevation, num_antennas]
                
                # Compute Bartlett beamformer output
                responses = torch.sum(csi_subcarrier.unsqueeze(0).unsqueeze(0) * torch.conj(steering_vectors), dim=2)
                
                # Compute power spectrum for this subcarrier
                subcarrier_spectrum = torch.abs(responses) ** 2
                
                # Accumulate
                sample_spectrum_sum += subcarrier_spectrum
            
            # Average across subcarriers
            pas_batch[i] = sample_spectrum_sum / num_subcarriers
            
    else:
        # 1D spatial spectrum
        num_antennas = array_shape[0]
        
        # Create angle grid
        azimuth_angles = torch.linspace(0, 180, azimuth_divisions, dtype=torch.float32, device=device)
        az_rad = azimuth_angles * torch.pi / 180.0
        
        
        # Initialize output tensor
        pas_batch = torch.zeros(batch_size, len(azimuth_angles), device=device)
        
        # Process each sample
        for i in range(batch_size):
            # Initialize accumulator for this sample
            sample_spectrum_sum = torch.zeros(len(azimuth_angles), device=device)
            
            # Process each subcarrier
            for subcarrier_idx in range(num_subcarriers):
                # Get CSI and wavelength for this subcarrier
                csi_subcarrier = csi_batch[i, :, subcarrier_idx]  # [num_antennas]
                wavelength_subcarrier = wavelengths[subcarrier_idx]  # scalar
                
                # Calculate antenna spacing for this subcarrier (half wavelength)
                spacing_x = wavelength_subcarrier / 2
                
                # Update antenna positions for this subcarrier
                antenna_positions = torch.arange(num_antennas, dtype=torch.float32, device=device) * spacing_x
                
                # Compute phase progression for all angles
                phase_progression = 2 * torch.pi * antenna_positions.unsqueeze(0) * torch.cos(az_rad).unsqueeze(1) / wavelength_subcarrier
                
                # Create steering vectors
                steering_vectors = torch.exp(1j * phase_progression)  # [num_angles, num_antennas]
                
                # Compute Bartlett beamformer output
                responses = torch.sum(csi_subcarrier.unsqueeze(0) * torch.conj(steering_vectors), dim=1)
                
                # Compute power spectrum for this subcarrier
                subcarrier_spectrum = torch.abs(responses) ** 2
                
                # Accumulate
                sample_spectrum_sum += subcarrier_spectrum
            
            # Average across subcarriers
            pas_batch[i] = sample_spectrum_sum / num_subcarriers
    
    return pas_batch


def compute_pas(csi_batch: torch.Tensor, array_shape: tuple, wavelengths: torch.Tensor,
                ignore_amplitude: bool = False, azimuth_divisions: int = 361,
                elevation_divisions: int = 91, azimuth_only: bool = False) -> torch.Tensor:
    """
    Compute Power Angular Spectrum (PAS) for a batch of CSI samples on GPU
    
    This is the main public API for computing spatial spectrum from CSI data.
    Supports both 1D linear arrays and 2D planar arrays, with flexible output options.
    
    Args:
        csi_batch: CSI tensor of shape [batch_size, num_antennas, num_subcarriers]
        array_shape: Antenna array shape tuple (length 1 for linear array, length 2 for planar array)
        wavelengths: Wavelength tensor of shape [num_subcarriers] containing wavelength for each subcarrier
        ignore_amplitude: Whether to ignore amplitude (default: False)
        azimuth_divisions: Number of azimuth angle divisions (default: 361 for 0-360° with 1° resolution)
        elevation_divisions: Number of elevation angle divisions (default: 91 for 0-90° with 1° resolution)
        azimuth_only: If True, returns only azimuth spectrum [batch_size, azimuth_divisions]
                     If False, returns full spectrum based on array type (default: False)
        
    Returns:
        When azimuth_only=False (default):
            - For 1D arrays: [batch_size, azimuth_divisions]
            - For 2D arrays: [batch_size, azimuth_divisions, elevation_divisions]
        When azimuth_only=True:
            - Always returns: [batch_size, azimuth_divisions]
            - For 2D arrays: integrates over elevation to get azimuth energy
                         
    Note:
        - For 1D arrays: Uses azimuth range 0-180° (linear array assumption)
        - For 2D arrays: Uses azimuth range 0-360° and elevation 0-90°
        - When azimuth_only=True for 2D arrays, elevation energy is summed into azimuth
        - Antenna spacing is automatically set to half wavelength (λ/2)
        - Each subcarrier uses its specific wavelength for accurate phase calculations
    
    Example:
        >>> csi = torch.randn(32, 8, 64, dtype=torch.complex64)  # 32 samples, 8 antennas, 64 subcarriers
        >>> wavelengths = torch.ones(64) * 0.125  # wavelength in meters
        >>> 
        >>> # For 1D array (8 antennas in a line) - full spectrum
        >>> pas_1d = compute_pas(csi, (8,), wavelengths)
        >>> print(pas_1d.shape)  # [32, 361]
        >>> 
        >>> # For 2D array (2x4 planar array) - full 2D spectrum
        >>> pas_2d = compute_pas(csi, (2, 4), wavelengths)
        >>> print(pas_2d.shape)  # [32, 361, 91]
        >>> 
        >>> # For 2D array - azimuth only
        >>> pas_azimuth = compute_pas(csi, (2, 4), wavelengths, azimuth_only=True)
        >>> print(pas_azimuth.shape)  # [32, 361]
    """
    is_2d_array = len(array_shape) == 2
    
    if azimuth_only:
        # Azimuth-only mode: always return [batch_size, azimuth_divisions]
        if is_2d_array:
            # For 2D planar arrays: compute full PAS then integrate over elevation
            # This gives us the total energy at each azimuth angle across all elevation angles
            pas_full = _compute_pas(
                csi_batch, 
                array_shape, 
                wavelengths, 
                ignore_amplitude=ignore_amplitude,
                azimuth_divisions=azimuth_divisions, 
                elevation_divisions=elevation_divisions
            )
            # Integrate over elevation dimension to get azimuth energy
            # pas_full shape: [batch_size, azimuth_divisions, elevation_divisions]
            azimuth_spectrum = torch.sum(pas_full, dim=2)  # [batch_size, azimuth_divisions]
            return azimuth_spectrum
        else:
            # For 1D linear arrays: directly compute azimuth spectrum
            return _compute_pas(
                csi_batch, 
                array_shape, 
                wavelengths, 
                ignore_amplitude=ignore_amplitude,
                azimuth_divisions=azimuth_divisions,
                elevation_divisions=elevation_divisions
            )
    else:
        # Full spectrum mode: return based on array type
        return _compute_pas(
            csi_batch, 
            array_shape, 
            wavelengths, 
            ignore_amplitude=ignore_amplitude,
            azimuth_divisions=azimuth_divisions,
            elevation_divisions=elevation_divisions
        )


def reorganize_data_as_mimo(csi: torch.Tensor,
                                   bs_positions: torch.Tensor, ue_positions: torch.Tensor,
                                   bs_antenna_indices: torch.Tensor, ue_antenna_indices: torch.Tensor,
                                   num_bs_antennas: int, num_ue_antennas: int):
    """
    Reorganize CSI data into MIMO format for spatial processing.
    
    This function takes batch data where each sample represents one antenna pair at one position pair,
    and reorganizes it into MIMO matrices where we have complete antenna array data for each position pair.
    This is essential for MIMO processing such as spatial spectrum computation.
    
    Processing Logic:
    1. Identify unique position pairs (BS position, UE position combinations)
    2. For each unique position pair, create a CSI matrix [num_bs_antennas, num_ue_antennas, num_subcarriers]
    3. Fill the CSI matrices using the antenna indices to place each sample in the correct position
    4. Return organized MIMO data ready for spatial processing
    
    Args:
        csi: CSI data [batch_size, num_subcarriers]
        bs_positions: BS positions [batch_size, 3]
        ue_positions: UE positions [batch_size, 3]
        bs_antenna_indices: BS antenna indices [batch_size]
        ue_antenna_indices: UE antenna indices [batch_size]
        num_bs_antennas: Total number of BS antennas
        num_ue_antennas: Total number of UE antennas
        
    Returns:
        csi_by_pos: List of tensors [num_bs_antennas, num_ue_antennas, num_subcarriers]
        unique_positions: List of (bs_pos, ue_pos) tuples
    """
    batch_size = csi.shape[0]
    num_subcarriers = csi.shape[1]
    device = csi.device
    
    # Step 1: Find unique position pairs (keep on GPU as much as possible)
    # Convert positions to hashable format for uniqueness check
    position_pairs = []
    position_tensors = []  # Keep original tensors for later use
    
    for i in range(batch_size):
        bs_pos_tuple = tuple(bs_positions[i].cpu().numpy().round(6))  # Round for numerical stability
        ue_pos_tuple = tuple(ue_positions[i].cpu().numpy().round(6))
        pos_pair = (bs_pos_tuple, ue_pos_tuple)
        position_pairs.append(pos_pair)
        position_tensors.append((bs_positions[i], ue_positions[i]))
    
    # Find unique position pairs
    unique_position_pairs = list(set(position_pairs))
    
    # Step 2: Create mapping from position pair to index
    pos_pair_to_idx = {pos_pair: idx for idx, pos_pair in enumerate(unique_position_pairs)}
    
    # Step 3: Initialize CSI arrays for each position pair
    csi_by_pos = []
    
    for _ in unique_position_pairs:
        csi_pos = torch.zeros(num_bs_antennas, num_ue_antennas, num_subcarriers, 
                            dtype=csi.dtype, device=device)
        csi_by_pos.append(csi_pos)
    
    # Step 4: Fill CSI arrays using antenna indices (avoid repeated CPU conversion)
    for i in range(batch_size):
        pos_pair = position_pairs[i]  # Use pre-computed position pair
        pos_pair_idx = pos_pair_to_idx[pos_pair]
        
        bs_ant_idx = bs_antenna_indices[i].item()
        ue_ant_idx = ue_antenna_indices[i].item()
        
        csi_by_pos[pos_pair_idx][bs_ant_idx, ue_ant_idx, :] = csi[i, :]
    
    # Step 5: Convert unique position pairs back to GPU tensors
    unique_positions = []
    for pos_pair in unique_position_pairs:
        bs_pos, ue_pos = pos_pair
        bs_pos_tensor = torch.tensor(bs_pos, device=device, dtype=torch.float32)
        ue_pos_tensor = torch.tensor(ue_pos, device=device, dtype=torch.float32)
        unique_positions.append((bs_pos_tensor, ue_pos_tensor))
    
    return csi_by_pos, unique_positions


def mimo_to_pas(csi_matrix: torch.Tensor, bs_array_shape: tuple, ue_array_shape: tuple,
                azimuth_divisions: int = 36, elevation_divisions: int = 9,
                normalize_pas: bool = True, center_freq: float = 3.5e9, 
                subcarrier_spacing: float = 245.1e3, azimuth_only: bool = False) -> dict:
    """
    Convert MIMO CSI matrix to Power Angular Spectrum (PAS) using spatial spectrum computation.
    
    This function takes a MIMO CSI matrix and computes the spatial spectrum from both BS and UE perspectives,
    then combines them into a unified PAS representation.
    
    Args:
        csi_matrix: MIMO CSI data [num_bs_antennas, num_ue_antennas, num_subcarriers]
        bs_array_shape: Base station antenna array shape tuple (e.g., (8,) for linear, (4, 4) for planar)
        ue_array_shape: User equipment antenna array shape tuple (e.g., (2,) for linear, (2, 2) for planar)
        azimuth_divisions: Number of azimuth angle divisions for PAS
        elevation_divisions: Number of elevation angle divisions for PAS
        normalize_pas: Whether to normalize the PAS to unit energy
        center_freq: Center frequency in Hz (default: 3.5 GHz)
        subcarrier_spacing: Subcarrier spacing in Hz (default: 245.1 kHz)
        azimuth_only: If True, returns only azimuth spectrum (integrates over elevation for 2D arrays)
        
    Returns:
        pas_dict: Dictionary containing PAS from both perspectives
                 {"bs": bs_pas, "ue": ue_pas}
                 When azimuth_only=False:
                   bs_pas: [num_ue_antennas, azimuth_divisions, elevation_divisions] if BS has >1 antennas, else zeros
                   ue_pas: [num_bs_antennas, azimuth_divisions, elevation_divisions] if UE has >1 antennas, else zeros
                 When azimuth_only=True:
                   bs_pas: [num_ue_antennas, azimuth_divisions] if BS has >1 antennas, else zeros
                   ue_pas: [num_bs_antennas, azimuth_divisions] if UE has >1 antennas, else zeros
        
    Note:
        - Always computes PAS from both BS and UE perspectives
        - If BS has single antenna, bs_pas will be zeros
        - If UE has single antenna, ue_pas will be zeros
        - Each perspective uses its respective antenna array configuration
        - When azimuth_only=True for 2D arrays, elevation energy is summed into azimuth
    """
    device = csi_matrix.device
    num_bs_antennas, num_ue_antennas, num_subcarriers = csi_matrix.shape
    
    # Calculate wavelengths for all subcarriers
    c = 3e8  # Speed of light
    subcarrier_indices = torch.arange(num_subcarriers, device=device)
    subcarrier_freqs = center_freq + (subcarrier_indices - num_subcarriers//2) * subcarrier_spacing
    wavelengths = c / subcarrier_freqs  # [num_subcarriers]
    
    # Compute PAS from both BS and UE perspectives
    
    # Initialize result dictionary
    pas_dict = {}
    
    # 1. Compute BS perspective PAS
    if num_bs_antennas > 1:
        # From BS perspective: each BS antenna sees signals from all UE antennas
        # We treat each UE antenna as a separate "batch" sample
        bs_csi_batch = csi_matrix.permute(1, 0, 2)  # [num_ue_antennas, num_bs_antennas, num_subcarriers]
        
        bs_pas_results = compute_pas(
            bs_csi_batch, bs_array_shape, wavelengths, ignore_amplitude=False,
            azimuth_divisions=azimuth_divisions, elevation_divisions=elevation_divisions,
            azimuth_only=azimuth_only
        )
        
        # Keep all UE antenna results
        bs_pas = bs_pas_results
        
        # Ensure proper output format based on azimuth_only flag
        if not azimuth_only and len(bs_pas.shape) == 2:
            # 1D case without azimuth_only: expand to 2D
            bs_pas = bs_pas.unsqueeze(2).expand(-1, -1, elevation_divisions)  # [num_ue_antennas, azimuth_divisions, elevation_divisions]
        
        # Normalize each UE antenna's PAS if requested (avoid inplace operations)
        if normalize_pas:
            normalized_pas_list = []
            for ue_idx in range(num_ue_antennas):
                pas_slice = bs_pas[ue_idx]
                total_energy = torch.sum(pas_slice)
                if total_energy > 1e-12:
                    normalized_pas_list.append(pas_slice / total_energy)
                else:
                    normalized_pas_list.append(pas_slice)
            bs_pas = torch.stack(normalized_pas_list, dim=0)
    else:
        # Single BS antenna - return zeros with appropriate shape
        if azimuth_only:
            bs_pas = torch.zeros(num_ue_antennas, azimuth_divisions, device=device)
        else:
            bs_pas = torch.zeros(num_ue_antennas, azimuth_divisions, elevation_divisions, device=device)
    
    pas_dict["bs"] = bs_pas
    
    # 2. Compute UE perspective PAS
    if num_ue_antennas > 1:
        # From UE perspective: each UE antenna sees signals from all BS antennas
        # We treat each BS antenna as a separate "batch" sample
        # For each BS antenna, we have CSI to all UE antennas: [num_ue_antennas, num_subcarriers]
        ue_csi_batch = csi_matrix  # [num_bs_antennas, num_ue_antennas, num_subcarriers]
        
        ue_pas_results = compute_pas(
            ue_csi_batch, ue_array_shape, wavelengths, ignore_amplitude=False,
            azimuth_divisions=azimuth_divisions, elevation_divisions=elevation_divisions,
            azimuth_only=azimuth_only
        )
        
        # Keep all BS antenna results
        ue_pas = ue_pas_results
        
        # Ensure proper output format based on azimuth_only flag
        if not azimuth_only and len(ue_pas.shape) == 2:
            # 1D case without azimuth_only: expand to 2D
            ue_pas = ue_pas.unsqueeze(2).expand(-1, -1, elevation_divisions)  # [num_bs_antennas, azimuth_divisions, elevation_divisions]
        
        # Normalize each BS antenna's PAS if requested (avoid inplace operations)
        if normalize_pas:
            normalized_pas_list = []
            for bs_idx in range(num_bs_antennas):
                pas_slice = ue_pas[bs_idx]
                total_energy = torch.sum(pas_slice)
                if total_energy > 1e-12:
                    normalized_pas_list.append(pas_slice / total_energy)
                else:
                    normalized_pas_list.append(pas_slice)
            ue_pas = torch.stack(normalized_pas_list, dim=0)
    else:
        # Single UE antenna - return zeros with appropriate shape
        if azimuth_only:
            ue_pas = torch.zeros(num_bs_antennas, azimuth_divisions, device=device)
        else:
            ue_pas = torch.zeros(num_bs_antennas, azimuth_divisions, elevation_divisions, device=device)
    
    pas_dict["ue"] = ue_pas
    
    return pas_dict


def parse_array_configuration(config_str: str) -> tuple:
    """
    Parse antenna array configuration string (e.g., '8x8', '2x2', '1x4')
    
    Args:
        config_str: Configuration string like '8x8', '2x4', '64', etc.
        
    Returns:
        tuple: (num_antennas_x, num_antennas_y) for 2D array or (num_antennas, 1) for 1D array
        
    Raises:
        ValueError: If configuration string is invalid
    """
    try:
        parts = config_str.split('x')
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
        elif len(parts) == 1:
            return (int(parts[0]), 1)  # Linear array
        else:
            raise ValueError(f"Invalid array configuration: {config_str}")
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse array configuration '{config_str}': {e}")
