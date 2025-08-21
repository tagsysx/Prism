#!/usr/bin/env python3
"""
China Mobile n41 Band 5G OFDM Simulation using NVIDIA Sionna
Simulates 1 BS with 64 antennas, UE with 4 antennas
n41 band: 2.5 GHz, 100 MHz bandwidth, 273 subcarriers
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

def create_china_mobile_n41_scenario():
    """Create China Mobile n41 band simulation scenario with Sionna"""
    
    # China Mobile n41 band parameters (2.5 GHz)
    carrier_frequency = 2.5e9  # 2.5 GHz (n41 band)
    bandwidth = 100e6  # 100 MHz (n41 band standard)
    num_subcarriers = 273  # n41 band: 100 MHz / 30 kHz = 273 subcarriers
    subcarrier_spacing = 30e3  # 30 kHz (n41 band standard)
    
    # Antenna configuration
    num_bs_antennas = 64  # BS antennas
    num_ue_antennas = 4   # UE antennas
    
    # OFDM parameters for n41 band
    fft_size = 512
    num_guard_carriers = (fft_size - num_subcarriers) // 2
    
    # n41 band specific parameters
    cyclic_prefix_ratio = 0.07  # 7% cyclic prefix for n41
    pilot_density = 0.1  # 10% pilot subcarriers
    
    print(f"=== China Mobile n41 Band 5G OFDM Simulation ===")
    print(f"Band: n41 (2.5 GHz)")
    print(f"Carrier frequency: {carrier_frequency/1e9:.1f} GHz")
    print(f"Bandwidth: {bandwidth/1e6:.0f} MHz")
    print(f"Subcarriers: {num_subcarriers}")
    print(f"Subcarrier spacing: {subcarrier_spacing/1e3:.0f} kHz")
    print(f"Cyclic prefix ratio: {cyclic_prefix_ratio:.1%}")
    print(f"Pilot density: {pilot_density:.1%}")
    print(f"BS antennas: {num_bs_antennas}")
    print(f"UE antennas: {num_ue_antennas}")
    print(f"FFT size: {fft_size}")
    print()
    
    return {
        'band': 'n41',
        'carrier_frequency': carrier_frequency,
        'bandwidth': bandwidth,
        'num_subcarriers': num_subcarriers,
        'subcarrier_spacing': subcarrier_spacing,
        'num_bs_antennas': num_bs_antennas,
        'num_ue_antennas': num_ue_antennas,
        'fft_size': fft_size,
        'num_guard_carriers': num_guard_carriers,
        'cyclic_prefix_ratio': cyclic_prefix_ratio,
        'pilot_density': pilot_density
    }

def generate_ue_positions_china_mobile(num_positions=100):
    """Generate 100 UE positions for China Mobile urban deployment"""
    
    # Define deployment area (typical urban microcell coverage)
    area_size = 400.0  # 400m x 400m for n41 band urban coverage
    
    # BS position at center (typical urban deployment)
    bs_position = np.array([area_size/2, area_size/2, 30.0])  # 30m height for urban
    
    # Generate UE positions with realistic constraints for n41 band
    ue_positions = []
    min_distance = 30.0  # Minimum 30m from BS (n41 band has shorter range)
    max_distance = 200.0  # Maximum 200m (n41 band urban coverage)
    
    np.random.seed(42)  # For reproducible results
    
    for i in range(num_positions):
        attempts = 0
        while attempts < 100:  # Prevent infinite loops
            # Random position in area
            x = np.random.uniform(0, area_size)
            y = np.random.uniform(0, area_size)
            z = np.random.uniform(1.5, 2.0)  # UE height 1.5-2m
            
            pos = np.array([x, y, z])
            distance = np.linalg.norm(pos[:2] - bs_position[:2])
            
            if min_distance <= distance <= max_distance:
                ue_positions.append(pos)
                break
            attempts += 1
        
        if attempts >= 100:
            # Fallback: place UE at valid distance
            angle = 2 * np.pi * i / num_positions
            distance = min_distance + (max_distance - min_distance) * 0.5
            x = bs_position[0] + distance * np.cos(angle)
            y = bs_position[1] + distance * np.sin(angle)
            z = np.random.uniform(1.5, 2.0)
            ue_positions.append(np.array([x, y, z]))
    
    print(f"Generated {len(ue_positions)} UE positions for China Mobile n41")
    print(f"BS position: {bs_position}")
    print(f"Coverage area: {area_size}m × {area_size}m")
    print(f"UE distance range: {min_distance:.1f}m - {max_distance:.1f}m")
    print(f"UE positions range: X[{min([p[0] for p in ue_positions]):.1f}, {max([p[0] for p in ue_positions]):.1f}]")
    print(f"                Y[{min([p[1] for p in ue_positions]):.1f}, {max([p[1] for p in ue_positions]):.1f}]")
    print(f"                Z[{min([p[2] for p in ue_positions]):.1f}, {max([p[2] for p in ue_positions]):.1f}]")
    print()
    
    return np.array(ue_positions), bs_position

def simulate_channel_responses_n41(scenario, ue_positions, bs_position):
    """Simulate channel responses for China Mobile n41 band"""
    
    # Create Sionna scene for n41 band urban environment
    scene = Scene("China_Mobile_n41_Urban_Scene")
    
    # Add BS and UE to scene
    scene.add_bs("BS", bs_position, scenario['num_bs_antennas'])
    
    # Create channel model optimized for n41 band (2.5 GHz urban environment)
    channel_model = UMi(
        carrier_frequency=scenario['carrier_frequency'],
        o2i_model="low",  # Low outdoor-to-indoor penetration for n41
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
    
    # n41 band specific metrics
    snr_values = np.zeros((num_positions, num_subcarriers))
    capacity_values = np.zeros((num_positions, num_subcarriers))
    
    print("Simulating channel responses for China Mobile n41 band...")
    
    for i, ue_pos in enumerate(ue_positions):
        if i % 10 == 0:
            print(f"Progress: {i}/{num_positions}")
        
        # Add UE to scene
        scene.add_ue(f"UE_{i}", ue_pos, num_ue_ant)
        
        # Calculate channel for n41 band
        h = channel_model(
            bs_positions=bs_position.reshape(1, 3),
            ut_positions=ue_pos.reshape(1, 3),
            bs_orientations=np.array([[0, 0, 0]]),
            ut_orientations=np.array([[0, 0, 0]])
        )
        
        # Extract channel matrix
        h_matrix = h.numpy()[0, 0]  # Shape: (num_ue_ant, num_bs_ant, num_paths)
        
        # For n41 band, consider multiple paths for better realism
        if h_matrix.shape[2] > 0:
            # Use strongest path for main channel
            h_main = h_matrix[:, :, 0]  # Shape: (num_ue_ant, num_bs_ant)
            
            # Apply frequency-dependent fading across subcarriers for n41 band
            for sc in range(num_subcarriers):
                # n41 band specific frequency-dependent effects
                # 2.5 GHz has different propagation characteristics than 3.5 GHz
                freq_offset = (sc - num_subcarriers//2) * scenario['subcarrier_spacing']
                relative_freq = freq_offset / scenario['carrier_frequency']
                
                # Phase shift with frequency-dependent delay
                phase_shift = 2 * np.pi * freq_offset * 1e-6
                h_freq = h_main * np.exp(1j * phase_shift)
                
                # Add n41 band specific fading (2.5 GHz has less penetration loss)
                # but more multipath due to urban environment
                multipath_factor = 1 + 0.3 * np.random.normal(0, 1) * np.exp(-sc/50)
                h_freq = h_freq * multipath_factor
                
                channel_responses[i, sc] = h_freq
                
                # Calculate path loss for n41 band
                path_losses[i, sc] = np.mean(np.abs(h_freq)**2)
                delays[i, sc] = np.angle(h_freq[0, 0]) / (2 * np.pi * scenario['subcarrier_spacing'])
                
                # Calculate SNR and capacity for n41 band
                noise_power = 1e-10  # Thermal noise at 2.5 GHz
                signal_power = path_losses[i, sc]
                snr_values[i, sc] = 10 * np.log10(signal_power / noise_power)
                
                # Shannon capacity for n41 band
                capacity_values[i, sc] = np.log2(1 + signal_power / noise_power)
        
        # Remove UE from scene for next iteration
        scene.remove_ue(f"UE_{i}")
    
    print("n41 band channel simulation completed!")
    print()
    
    return channel_responses, path_losses, delays, snr_values, capacity_values

def save_n41_simulation_data(channel_responses, path_losses, delays, snr_values, capacity_values, 
                            ue_positions, bs_position, scenario, output_file):
    """Save China Mobile n41 band simulation data to HDF5 file"""
    
    print(f"Saving n41 band simulation data to {output_file}...")
    
    with h5py.File(output_file, 'w') as f:
        # Create groups
        f.create_group('simulation_config')
        f.create_group('positions')
        f.create_group('channel_data')
        f.create_group('n41_metrics')
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
        
        # Save n41 band specific metrics
        n41_group = f['n41_metrics']
        n41_group.create_dataset('snr_values', data=snr_values)
        n41_group.create_dataset('capacity_values', data=capacity_values)
        
        # Save metadata
        meta_group = f['metadata']
        meta_group.attrs['simulation_date'] = datetime.now().isoformat()
        meta_group.attrs['band'] = 'n41'
        meta_group.attrs['operator'] = 'China Mobile'
        meta_group.attrs['num_ue_positions'] = len(ue_positions)
        meta_group.attrs['sionna_version'] = sionna.__version__
        
        # Dataset info
        meta_group.attrs['channel_responses_shape'] = channel_responses.shape
        meta_group.attrs['path_losses_shape'] = path_losses.shape
        meta_group.attrs['delays_shape'] = delays.shape
        meta_group.attrs['snr_values_shape'] = snr_values.shape
        meta_group.attrs['capacity_values_shape'] = capacity_values.shape
    
    print(f"n41 band data saved successfully!")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print()

def create_n41_visualizations(channel_responses, path_losses, delays, snr_values, capacity_values,
                             ue_positions, bs_position, scenario):
    """Create visualizations for China Mobile n41 band simulation"""
    
    print("Creating n41 band visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('China Mobile n41 Band (2.5 GHz) Simulation Results', fontsize=16)
    
    # Plot 1: UE positions
    ax1 = axes[0, 0]
    ax1.scatter(ue_positions[:, 0], ue_positions[:, 1], c='blue', alpha=0.6, s=20, label='UE Positions')
    ax1.scatter(bs_position[0], bs_position[1], c='red', s=100, marker='^', label='BS')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('UE and BS Positions (n41 Band)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Path loss distribution
    ax2 = axes[0, 1]
    mean_path_loss = np.mean(path_losses, axis=1)
    ax2.hist(mean_path_loss, bins=20, alpha=0.7, color='green')
    ax2.set_xlabel('Average Path Loss')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Path Loss Distribution (n41 Band)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: SNR distribution
    ax3 = axes[0, 2]
    mean_snr = np.mean(snr_values, axis=1)
    ax3.hist(mean_snr, bins=20, alpha=0.7, color='orange')
    ax3.set_xlabel('Average SNR (dB)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('SNR Distribution (n41 Band)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Channel magnitude for first UE position
    ax4 = axes[1, 0]
    ue_idx = 0
    channel_mag = np.abs(channel_responses[ue_idx, :, 0, 0])
    subcarrier_indices = np.arange(scenario['num_subcarriers'])
    ax4.plot(subcarrier_indices, channel_mag, 'b-', linewidth=2)
    ax4.set_xlabel('Subcarrier Index')
    ax4.set_ylabel('Channel Magnitude')
    ax4.set_title(f'Channel Magnitude (UE {ue_idx}, n41 Band)')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Channel capacity for first UE position
    ax5 = axes[1, 1]
    capacity_ue = capacity_values[ue_idx, :]
    ax5.plot(subcarrier_indices, capacity_ue, 'r-', linewidth=2)
    ax5.set_xlabel('Subcarrier Index')
    ax5.set_ylabel('Channel Capacity (bits/s/Hz)')
    ax5.set_title(f'Channel Capacity (UE {ue_idx}, n41 Band)')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Capacity distribution
    ax6 = axes[1, 2]
    mean_capacity = np.mean(capacity_values, axis=1)
    ax6.hist(mean_capacity, bins=20, alpha=0.7, color='purple')
    ax6.set_xlabel('Average Capacity (bits/s/Hz)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Capacity Distribution (n41 Band)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_plot = 'data/china_mobile_n41_simulation_results.png'
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"n41 band visualization saved to {output_plot}")
    print()

def main():
    """Main simulation function for China Mobile n41 band"""
    
    print("=== China Mobile n41 Band 5G OFDM Simulation using NVIDIA Sionna ===")
    print("Band: n41 (2.5 GHz)")
    print("Operator: China Mobile")
    print("Environment: Urban Microcell")
    print()
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Step 1: Create n41 band simulation scenario
    scenario = create_china_mobile_n41_scenario()
    
    # Step 2: Generate UE positions for n41 band
    ue_positions, bs_position = generate_ue_positions_china_mobile(num_positions=100)
    
    # Step 3: Simulate channel responses for n41 band
    channel_responses, path_losses, delays, snr_values, capacity_values = simulate_channel_responses_n41(
        scenario, ue_positions, bs_position
    )
    
    # Step 4: Save n41 band data
    output_file = data_dir / "china_mobile_n41_simulation.h5"
    save_n41_simulation_data(
        channel_responses, path_losses, delays, snr_values, capacity_values,
        ue_positions, bs_position, scenario, output_file
    )
    
    # Step 5: Create n41 band visualizations
    create_n41_visualizations(
        channel_responses, path_losses, delays, snr_values, capacity_values,
        ue_positions, bs_position, scenario
    )
    
    print("=== China Mobile n41 Band Simulation completed successfully! ===")
    print(f"Output files:")
    print(f"  - Data: {output_file}")
    print(f"  - Plot: data/china_mobile_n41_simulation_results.png")
    print()
    print("n41 Band Data Structure:")
    print(f"  - Channel responses: {channel_responses.shape}")
    print(f"  - Path losses: {path_losses.shape}")
    print(f"  - Delays: {delays.shape}")
    print(f"  - SNR values: {snr_values.shape}")
    print(f"  - Capacity values: {capacity_values.shape}")
    print(f"  - UE positions: {ue_positions.shape}")
    print()
    print("n41 Band Characteristics:")
    print(f"  - Frequency: 2.5 GHz (better penetration than 3.5 GHz)")
    print(f"  - Bandwidth: 100 MHz (standard n41 allocation)")
    print(f"  - Subcarriers: 273 (30 kHz spacing)")
    print(f"  - Urban coverage: 400m × 400m")
    print(f"  - Typical use: Dense urban deployment")

if __name__ == "__main__":
    main()
