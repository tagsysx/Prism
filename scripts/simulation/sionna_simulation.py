#!/usr/bin/env python3
"""
5G OFDM Simulation using NVIDIA Sionna
Simulates 1 BS with 64 antennas, UE with 4 antennas
100MHz bandwidth, 408 subcarriers, 100 UE positions
"""

import numpy as np
import h5py
import os
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Sionna imports
try:
    import sionna
    from sionna.channel import sub6GHz, UMi, UMa
    from sionna.channel import gen_single_sector_topology
    from sionna.rt import load_scene, RadioMaterial, Scene
    from sionna.utils import si
    from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator
    from sionna.mimo import StreamManagement
    from sionna.channel import OFDMChannel
    from sionna.rt import load_scene, RadioMaterial, Scene
    print(f"Sionna version: {sionna.__version__}")
except ImportError:
    print("Sionna not found. Please install with: pip install sionna")
    exit(1)

def create_simulation_scenario():
    """Create 5G simulation scenario with Sionna"""
    
    # 5G NR parameters
    carrier_frequency = 3.5e9  # 3.5 GHz
    bandwidth = 100e6  # 100 MHz
    num_subcarriers = 408
    subcarrier_spacing = bandwidth / num_subcarriers
    
    # Antenna configuration
    num_bs_antennas = 64  # BS antennas
    num_ue_antennas = 4   # UE antennas
    
    # OFDM parameters
    fft_size = 512
    num_guard_carriers = (fft_size - num_subcarriers) // 2
    
    print(f"=== 5G OFDM Simulation Configuration ===")
    print(f"Carrier frequency: {carrier_frequency/1e9:.1f} GHz")
    print(f"Bandwidth: {bandwidth/1e6:.0f} MHz")
    print(f"Subcarriers: {num_subcarriers}")
    print(f"Subcarrier spacing: {subcarrier_spacing/1e3:.1f} kHz")
    print(f"BS antennas: {num_bs_antennas}")
    print(f"UE antennas: {num_ue_antennas}")
    print(f"FFT size: {fft_size}")
    print()
    
    return {
        'carrier_frequency': carrier_frequency,
        'bandwidth': bandwidth,
        'num_subcarriers': num_subcarriers,
        'subcarrier_spacing': subcarrier_spacing,
        'num_bs_antennas': num_bs_antennas,
        'num_ue_antennas': num_ue_antennas,
        'fft_size': fft_size,
        'num_guard_carriers': num_guard_carriers
    }

def generate_ue_positions(num_positions=100):
    """Generate 100 UE positions in a realistic deployment area"""
    
    # Define deployment area (500m x 500m)
    area_size = 500.0
    
    # BS position at center
    bs_position = np.array([area_size/2, area_size/2, 25.0])  # 25m height
    
    # Generate UE positions with minimum distance from BS
    ue_positions = []
    min_distance = 50.0  # Minimum 50m from BS
    
    np.random.seed(42)  # For reproducible results
    
    for i in range(num_positions):
        while True:
            # Random position in area
            x = np.random.uniform(0, area_size)
            y = np.random.uniform(0, area_size)
            z = np.random.uniform(1.5, 2.0)  # UE height 1.5-2m
            
            pos = np.array([x, y, z])
            distance = np.linalg.norm(pos[:2] - bs_position[:2])
            
            if distance >= min_distance:
                ue_positions.append(pos)
                break
    
    print(f"Generated {len(ue_positions)} UE positions")
    print(f"BS position: {bs_position}")
    print(f"UE positions range: X[{min([p[0] for p in ue_positions]):.1f}, {max([p[0] for p in ue_positions]):.1f}]")
    print(f"                Y[{min([p[1] for p in ue_positions]):.1f}, {max([p[1] for p in ue_positions]):.1f}]")
    print(f"                Z[{min([p[2] for p in ue_positions]):.1f}, {max([p[2] for p in ue_positions]):.1f}]")
    print()
    
    return np.array(ue_positions), bs_position

def simulate_channel_responses(scenario, ue_positions, bs_position):
    """Simulate channel responses for all UE positions"""
    
    # Create Sionna scene
    scene = Scene("5G_Simulation_Scene")
    
    # Add BS and UE to scene
    scene.add_bs("BS", bs_position, num_bs_antennas)
    
    # Create channel model (UMi for urban microcell)
    channel_model = UMi(
        carrier_frequency=scenario['carrier_frequency'],
        o2i_model="low",
        ut_array="3gpp-3d",
        bs_array="3gpp-3d"
    )
    
    # Initialize arrays for storing results
    num_positions = len(ue_positions)
    num_subcarriers = scenario['num_subcarriers']
    num_bs_ant = scenario['num_bs_antennas']
    num_ue_ant = scenario['num_ue_antennas']
    
    # Channel responses for each subcarrier
    channel_responses = np.zeros((num_positions, num_subcarriers, num_ue_ant, num_bs_ant), dtype=np.complex128)
    
    # Path loss and delay for each position
    path_losses = np.zeros((num_positions, num_subcarriers))
    delays = np.zeros((num_positions, num_subcarriers))
    
    print("Simulating channel responses...")
    
    for i, ue_pos in enumerate(ue_positions):
        if i % 10 == 0:
            print(f"Progress: {i}/{num_positions}")
        
        # Add UE to scene
        scene.add_ue(f"UE_{i}", ue_pos, num_ue_ant)
        
        # Calculate channel
        h = channel_model(
            bs_positions=bs_position.reshape(1, 3),
            ut_positions=ue_pos.reshape(1, 3),
            bs_orientations=np.array([[0, 0, 0]]),
            ut_orientations=np.array([[0, 0, 0]])
        )
        
        # Extract channel matrix
        h_matrix = h.numpy()[0, 0]  # Shape: (num_ue_ant, num_bs_ant, num_paths)
        
        # For simplicity, use the first path (LOS or strongest path)
        if h_matrix.shape[2] > 0:
            h_main = h_matrix[:, :, 0]  # Shape: (num_ue_ant, num_bs_ant)
            
            # Apply frequency-dependent fading across subcarriers
            for sc in range(num_subcarriers):
                # Simple frequency-dependent phase shift
                phase_shift = 2 * np.pi * sc * scenario['subcarrier_spacing'] * 1e-6
                h_freq = h_main * np.exp(1j * phase_shift)
                
                channel_responses[i, sc] = h_freq
                
                # Calculate path loss (simplified)
                path_losses[i, sc] = np.mean(np.abs(h_freq)**2)
                delays[i, sc] = np.angle(h_freq[0, 0]) / (2 * np.pi * scenario['subcarrier_spacing'])
        
        # Remove UE from scene for next iteration
        scene.remove_ue(f"UE_{i}")
    
    print("Channel simulation completed!")
    print()
    
    return channel_responses, path_losses, delays

def save_simulation_data(channel_responses, path_losses, delays, ue_positions, bs_position, scenario, output_file):
    """Save simulation data to HDF5 file"""
    
    print(f"Saving simulation data to {output_file}...")
    
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
        meta_group.attrs['sionna_version'] = sionna.__version__
        
        # Dataset info
        meta_group.attrs['channel_responses_shape'] = channel_responses.shape
        meta_group.attrs['path_losses_shape'] = path_losses.shape
        meta_group.attrs['delays_shape'] = delays.shape
    
    print(f"Data saved successfully!")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print()

def create_visualizations(channel_responses, path_losses, delays, ue_positions, bs_position, scenario):
    """Create visualization plots"""
    
    print("Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('5G OFDM Simulation Results', fontsize=16)
    
    # Plot 1: UE positions
    ax1 = axes[0, 0]
    ax1.scatter(ue_positions[:, 0], ue_positions[:, 1], c='blue', alpha=0.6, s=20, label='UE Positions')
    ax1.scatter(bs_position[0], bs_position[1], c='red', s=100, marker='^', label='BS')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('UE and BS Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Path loss distribution
    ax2 = axes[0, 1]
    mean_path_loss = np.mean(path_losses, axis=1)
    ax2.hist(mean_path_loss, bins=20, alpha=0.7, color='green')
    ax2.set_xlabel('Average Path Loss')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Path Loss Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Channel magnitude for first UE position
    ax3 = axes[1, 0]
    ue_idx = 0
    channel_mag = np.abs(channel_responses[ue_idx, :, 0, 0])
    subcarrier_indices = np.arange(scenario['num_subcarriers'])
    ax3.plot(subcarrier_indices, channel_mag, 'b-', linewidth=2)
    ax3.set_xlabel('Subcarrier Index')
    ax3.set_ylabel('Channel Magnitude')
    ax3.set_title(f'Channel Magnitude (UE {ue_idx})')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Channel phase for first UE position
    ax4 = axes[1, 1]
    channel_phase = np.angle(channel_responses[ue_idx, :, 0, 0])
    ax4.plot(subcarrier_indices, channel_phase, 'r-', linewidth=2)
    ax4.set_xlabel('Subcarrier Index')
    ax4.set_ylabel('Channel Phase (radians)')
    ax4.set_title(f'Channel Phase (UE {ue_idx})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_plot = 'data/sionna_simulation_results.png'
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_plot}")
    print()

def main():
    """Main simulation function"""
    
    print("=== 5G OFDM Simulation using NVIDIA Sionna ===")
    print()
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Step 1: Create simulation scenario
    scenario = create_simulation_scenario()
    
    # Step 2: Generate UE positions
    ue_positions, bs_position = generate_ue_positions(num_positions=100)
    
    # Step 3: Simulate channel responses
    channel_responses, path_losses, delays = simulate_channel_responses(
        scenario, ue_positions, bs_position
    )
    
    # Step 4: Save data
    output_file = data_dir / "sionna_5g_simulation.h5"
    save_simulation_data(
        channel_responses, path_losses, delays, 
        ue_positions, bs_position, scenario, output_file
    )
    
    # Step 5: Create visualizations
    create_visualizations(
        channel_responses, path_losses, delays, 
        ue_positions, bs_position, scenario
    )
    
    print("=== Simulation completed successfully! ===")
    print(f"Output files:")
    print(f"  - Data: {output_file}")
    print(f"  - Plot: data/sionna_simulation_results.png")
    print()
    print("Data structure:")
    print(f"  - Channel responses: {channel_responses.shape}")
    print(f"  - Path losses: {path_losses.shape}")
    print(f"  - Delays: {delays.shape}")
    print(f"  - UE positions: {ue_positions.shape}")

if __name__ == "__main__":
    main()
