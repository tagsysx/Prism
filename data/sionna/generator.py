#!/usr/bin/env python3
"""
Synthetic Data Generator for Prism Training

This script generates synthetic 5G OFDM channel data without requiring Sionna's
ray tracing functionality, avoiding LLVM dependencies.

Usage:
    python synthetic_data_generator.py -n 300 --output_path ../sionna/P300
"""

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
import argparse

# ============================================================================
# CONFIGURATION SECTION - Default Values
# ============================================================================

# Simulation Configuration
DEFAULT_CONFIG = {
    # Basic Parameters
    'num_positions': 300,                    # Number of UE positions to generate
    'output_path': '../sionna/simulation',         # Default output path (Pxxx format where xxx is num_positions)
    
    # Area Configuration
    'area_size': 500,                        # Size of the square area in meters
    'bs_height': 25.0,                       # Base station height in meters
    'ue_height_min': 1.0,                    # Minimum UE height in meters
    'ue_height_max': 3.0,                    # Maximum UE height in meters
    
    # 5G OFDM Parameters
    'center_frequency': 3.5e9,               # Center frequency in Hz (3.5 GHz)
    'subcarrier_spacing': 30e3,              # Subcarrier spacing in Hz (30 kHz)
    'num_subcarriers': 408,                  # Number of OFDM subcarriers
    'num_ue_antennas': 4,                    # Number of UE antennas
    'num_bs_antennas': 64,                   # Number of BS antennas
    
    # Channel Model Parameters
    'shadowing_std': 8.0,                    # Shadowing standard deviation in dB
    'multipath_delay_spread': 50e-9,         # Multipath delay spread in seconds
    
    # Random Seed
    'random_seed': 42,                       # Random seed for reproducibility
    
    # Visualization
    'create_plots': True,                    # Whether to create visualization plots
    'plot_dpi': 300,                         # Plot resolution
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_ue_positions(config):
    """
    Generate random UE positions in a square area
    
    Args:
        config: Configuration dictionary containing simulation parameters
    
    Returns:
        ue_positions: Array of UE positions (num_positions, 3)
        bs_position: Base station position (3,)
    """
    num_positions = config['num_positions']
    area_size = config['area_size']
    ue_height_min = config['ue_height_min']
    ue_height_max = config['ue_height_max']
    bs_height = config['bs_height']
    
    logger.info(f"Generating {num_positions} UE positions in {area_size}x{area_size}m area")
    
    # Set random seed for reproducibility
    np.random.seed(config['random_seed'])
    
    # UE positions: random in square area, height between min-max
    ue_positions = np.zeros((num_positions, 3))
    ue_positions[:, 0] = np.random.uniform(-area_size/2, area_size/2, num_positions)  # X
    ue_positions[:, 1] = np.random.uniform(-area_size/2, area_size/2, num_positions)  # Y
    ue_positions[:, 2] = np.random.uniform(ue_height_min, ue_height_max, num_positions)  # Z (height)
    
    # BS position: center of area, configurable height
    bs_position = np.array([0.0, 0.0, bs_height])
    
    logger.info(f"Generated {num_positions} UE positions")
    logger.info(f"BS position: {bs_position}")
    
    return ue_positions, bs_position

def generate_synthetic_channels(ue_positions, bs_position, config):
    """
    Generate synthetic channel responses based on distance and frequency
    
    Args:
        ue_positions: UE positions (num_positions, 3)
        bs_position: BS position (3,)
        config: Configuration dictionary containing simulation parameters
    
    Returns:
        channel_responses: Complex channel responses (num_positions, num_subcarriers, num_ue_antennas, num_bs_antennas)
        path_losses: Path losses in dB (num_positions, num_subcarriers)
        delays: Channel delays in seconds (num_positions, num_subcarriers)
    """
    num_positions = len(ue_positions)
    num_subcarriers = config['num_subcarriers']
    num_ue_antennas = config['num_ue_antennas']
    num_bs_antennas = config['num_bs_antennas']
    center_freq = config['center_frequency']
    subcarrier_spacing = config['subcarrier_spacing']
    shadowing_std = config['shadowing_std']
    multipath_delay_spread = config['multipath_delay_spread']
    
    logger.info(f"Generating synthetic channels for {num_positions} positions")
    logger.info(f"Channel dimensions: ({num_positions}, {num_subcarriers}, {num_ue_antennas}, {num_bs_antennas})")
    
    # Calculate distances
    distances = np.linalg.norm(ue_positions - bs_position, axis=1)
    
    # Initialize arrays
    channel_responses = np.zeros((num_positions, num_subcarriers, num_ue_antennas, num_bs_antennas), dtype=complex)
    path_losses = np.zeros((num_positions, num_subcarriers))
    delays = np.zeros((num_positions, num_subcarriers))
    
    # Frequency parameters
    frequencies = center_freq + np.arange(num_subcarriers) * subcarrier_spacing
    
    for pos_idx in range(num_positions):
        distance = distances[pos_idx]
        
        # Path loss model (simplified free space + shadowing)
        free_space_loss = 20 * np.log10(distance) + 20 * np.log10(frequencies / 1e9) + 32.45
        shadowing = np.random.normal(0, shadowing_std, num_subcarriers)
        path_loss_db = free_space_loss + shadowing
        path_losses[pos_idx, :] = path_loss_db
        
        # Convert to linear scale
        path_loss_linear = 10 ** (-path_loss_db / 20)
        
        # Delay (time of flight + multipath)
        time_of_flight = distance / 3e8  # Speed of light
        multipath_delays = np.random.exponential(multipath_delay_spread, num_subcarriers)
        delays[pos_idx, :] = time_of_flight + multipath_delays
        
        # Generate channel matrix for each subcarrier
        for sc_idx in range(num_subcarriers):
            # Rayleigh fading channel
            h_real = np.random.normal(0, 1, (num_ue_antennas, num_bs_antennas))
            h_imag = np.random.normal(0, 1, (num_ue_antennas, num_bs_antennas))
            h_complex = (h_real + 1j * h_imag) / np.sqrt(2)
            
            # Apply path loss
            h_complex *= path_loss_linear[sc_idx]
            
            # Apply frequency-dependent phase shift
            phase_shift = np.exp(-1j * 2 * np.pi * frequencies[sc_idx] * delays[pos_idx, sc_idx])
            h_complex *= phase_shift
            
            channel_responses[pos_idx, sc_idx, :, :] = h_complex
        
        if (pos_idx + 1) % 50 == 0:
            logger.info(f"Generated channels for {pos_idx + 1}/{num_positions} positions")
    
    logger.info("Channel generation completed")
    return channel_responses, path_losses, delays

def save_simulation_data(channel_responses, path_losses, delays, ue_positions, bs_position, config, output_file):
    """Save simulation data to HDF5 file"""
    
    logger.info(f"Saving simulation data to {output_file}...")
    
    # Create scenario info from config
    scenario = {
        'center_frequency': config['center_frequency'],
        'bandwidth': config['subcarrier_spacing'] * config['num_subcarriers'],
        'num_subcarriers': config['num_subcarriers'],
        'subcarrier_spacing': config['subcarrier_spacing'],
        'num_ue_antennas': config['num_ue_antennas'],
        'num_bs_antennas': config['num_bs_antennas'],
        'simulation_type': 'synthetic_5g_ofdm',
        'area_size': config['area_size'],
        'bs_height': config['bs_height'],
        'shadowing_std': config['shadowing_std'],
        'multipath_delay_spread': config['multipath_delay_spread']
    }
    
    with h5py.File(output_file, 'w') as f:
        # Create groups
        f.create_group('simulation_config')
        f.create_group('positions')
        f.create_group('channel_data')
        f.create_group('metadata')
        
        # Save simulation configuration
        config_group = f['simulation_config']
        for key, value in scenario.items():
            config_group.attrs[key] = value
        
        # Save positions
        pos_group = f['positions']
        pos_group.create_dataset('bs_position', data=bs_position)
        pos_group.create_dataset('ue_positions', data=ue_positions)
        
        # Save channel data
        channel_group = f['channel_data']
        channel_group.create_dataset('channel_responses', data=channel_responses)
        channel_group.create_dataset('path_losses', data=path_losses)
        channel_group.create_dataset('delays', data=delays)
        
        # Save metadata
        meta_group = f['metadata']
        meta_group.attrs['simulation_date'] = datetime.now().isoformat()
        meta_group.attrs['num_ue_positions'] = len(ue_positions)
        meta_group.attrs['generator_version'] = '1.0.0'
        
        # Dataset info
        meta_group.attrs['channel_responses_shape'] = channel_responses.shape
        meta_group.attrs['path_losses_shape'] = path_losses.shape
        meta_group.attrs['delays_shape'] = delays.shape
    
    logger.info(f"Data saved successfully!")

def create_visualizations(channel_responses, path_losses, delays, ue_positions, bs_position, config, output_dir):
    """Create visualization plots"""
    
    if not config['create_plots']:
        logger.info("Skipping visualization creation (disabled in config)")
        return
    
    logger.info("Creating visualizations...")
    
    num_positions = config['num_positions']
    num_subcarriers = config['num_subcarriers']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Synthetic 5G OFDM Simulation Results ({num_positions} positions)', fontsize=16)
    
    # Plot 1: UE positions
    ax1 = axes[0, 0]
    ax1.scatter(ue_positions[:, 0], ue_positions[:, 1], c='blue', alpha=0.6, s=20, label='UE Positions')
    ax1.scatter(bs_position[0], bs_position[1], c='red', s=100, marker='^', label='BS')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title(f'UE and BS Positions ({num_positions} UEs)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Path loss distribution
    ax2 = axes[0, 1]
    mean_path_loss = np.mean(path_losses, axis=1)
    ax2.hist(mean_path_loss, bins=30, alpha=0.7, color='green')
    ax2.set_xlabel('Average Path Loss (dB)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Path Loss Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Channel magnitude for first UE position
    ax3 = axes[1, 0]
    ue_idx = 0
    channel_mag = np.abs(channel_responses[ue_idx, :, 0, 0])
    subcarrier_indices = np.arange(num_subcarriers)
    ax3.plot(subcarrier_indices, channel_mag, 'b-', linewidth=1)
    ax3.set_xlabel('Subcarrier Index')
    ax3.set_ylabel('Channel Magnitude')
    ax3.set_title(f'Channel Magnitude (UE {ue_idx})')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Channel phase for first UE position
    ax4 = axes[1, 1]
    channel_phase = np.angle(channel_responses[ue_idx, :, 0, 0])
    ax4.plot(subcarrier_indices, channel_phase, 'r-', linewidth=1)
    ax4.set_xlabel('Subcarrier Index')
    ax4.set_ylabel('Channel Phase (radians)')
    ax4.set_title(f'Channel Phase (UE {ue_idx})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_plot = output_dir / 'synthetic_simulation_results.png'
    plt.savefig(output_plot, dpi=config['plot_dpi'], bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to {output_plot}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Synthetic 5G OFDM Data Generator for Prism Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-n', '--num', 
        type=int, 
        required=True,
        help='Number of UE positions to generate'
    )
    
    parser.add_argument(
        '--output_path', 
        type=str, 
        default=None,
        help='Output path for saving data (default: ../sionna/Pxxx where xxx is num_positions)'
    )
    
    parser.add_argument(
        '--area_size', 
        type=float, 
        default=DEFAULT_CONFIG['area_size'],
        help='Size of the square area in meters'
    )
    
    parser.add_argument(
        '--bs_height', 
        type=float, 
        default=DEFAULT_CONFIG['bs_height'],
        help='Base station height in meters'
    )
    
    parser.add_argument(
        '--center_frequency', 
        type=float, 
        default=DEFAULT_CONFIG['center_frequency'],
        help='Center frequency in Hz'
    )
    
    parser.add_argument(
        '--num_subcarriers', 
        type=int, 
        default=DEFAULT_CONFIG['num_subcarriers'],
        help='Number of OFDM subcarriers'
    )
    
    parser.add_argument(
        '--num_bs_antennas', 
        type=int, 
        default=DEFAULT_CONFIG['num_bs_antennas'],
        help='Number of BS antennas'
    )
    
    parser.add_argument(
        '--num_ue_antennas', 
        type=int, 
        default=DEFAULT_CONFIG['num_ue_antennas'],
        help='Number of UE antennas'
    )
    
    parser.add_argument(
        '--random_seed', 
        type=int, 
        default=DEFAULT_CONFIG['random_seed'],
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--no_plots', 
        action='store_true',
        help='Disable visualization plots'
    )
    
    return parser.parse_args()

def main():
    """Main simulation function"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create configuration from defaults and command line arguments
    config = DEFAULT_CONFIG.copy()
    config['num_positions'] = args.num
    config['area_size'] = args.area_size
    config['bs_height'] = args.bs_height
    config['center_frequency'] = args.center_frequency
    config['num_subcarriers'] = args.num_subcarriers
    config['num_bs_antennas'] = args.num_bs_antennas
    config['num_ue_antennas'] = args.num_ue_antennas
    config['random_seed'] = args.random_seed
    config['create_plots'] = not args.no_plots
    
    # Set output path
    if args.output_path is None:
        config['output_path'] = f"../sionna/P{args.num}"
    else:
        config['output_path'] = args.output_path
    
    print(f"=== Synthetic 5G OFDM Data Generation ({config['num_positions']} positions) ===")
    print()
    
    # Display configuration
    print("Configuration:")
    print(f"  Number of positions: {config['num_positions']}")
    print(f"  Output path: {config['output_path']}")
    print(f"  Area size: {config['area_size']} m")
    print(f"  BS height: {config['bs_height']} m")
    print(f"  Center frequency: {config['center_frequency']/1e9:.1f} GHz")
    print(f"  Subcarriers: {config['num_subcarriers']}")
    print(f"  BS antennas: {config['num_bs_antennas']}")
    print(f"  UE antennas: {config['num_ue_antennas']}")
    print(f"  Random seed: {config['random_seed']}")
    print(f"  Create plots: {config['create_plots']}")
    print()
    
    # Create data directory
    data_dir = Path(config['output_path'])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate UE positions
    print("Step 1: Generating UE positions...")
    ue_positions, bs_position = generate_ue_positions(config)
    
    # Step 2: Generate synthetic channel responses
    print("Step 2: Generating synthetic channel responses...")
    channel_responses, path_losses, delays = generate_synthetic_channels(
        ue_positions, bs_position, config
    )
    
    # Step 3: Save data
    output_file = data_dir / f"synthetic_5g_simulation_P{config['num_positions']}.h5"
    print(f"Step 3: Saving data to {output_file}...")
    save_simulation_data(
        channel_responses, path_losses, delays, 
        ue_positions, bs_position, config, output_file
    )
    
    # Step 4: Create visualizations
    print("Step 4: Creating visualizations...")
    create_visualizations(
        channel_responses, path_losses, delays, 
        ue_positions, bs_position, config, data_dir
    )
    
    print("=== Data generation completed successfully! ===")
    print(f"Output files:")
    print(f"  - Data: {output_file}")
    if config['create_plots']:
        print(f"  - Plot: {data_dir}/synthetic_simulation_results.png")
    print()
    print("Data structure:")
    print(f"  - Channel responses: {channel_responses.shape}")
    print(f"  - Path losses: {path_losses.shape}")
    print(f"  - Delays: {delays.shape}")
    print(f"  - UE positions: {ue_positions.shape}")
    print(f"  - Total positions: {ue_positions.shape[0]}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Data generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        raise
