#!/usr/bin/env python3
"""
Ray Tracing Data Generator for 5G OFDM Channel Simulation
Based on campus environment using Sionna ray tracing

This script generates realistic 5G OFDM channel data using Sionna's ray tracing
in a campus environment.

天线配置说明:
=============

基站天线 (Base Station):
- 配置: 4行 × 8列 双极化阵列
- 水平间距: 0.5λ (标准配置)
- 垂直间距: 0.68λ (标准配置)
- 极化方式: 交叉极化 (+45°/-45°)
- 通道数: 64个 (4×8=32个物理天线 × 2极化 = 64个通道)
- 天线方向图: TR38901 (5G NR标准)
- 设计特点: 双极化实现空间复用

UE天线 (User Equipment):
- 配置: 商用手机双天线 (1×2线性排列) 双极化
- 天线类型: 主天线 + 分集天线，各自双极化
- 天线间距: 0.7λ (手机内部空间限制)
- 极化方式: 交叉极化 (提高分集性能)
- 方向图: 全向 (iso) - 模拟手机天线特性
- 通道数: 4个 (2个物理天线 × 2极化 = 4个通道)

MIMO配置: 64×4 (64个基站通道 × 4个UE通道)

Usage:
    python signal_tracking_simulation.py -n 300 --output_path ../sionna/P300
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from datetime import datetime
import time
import logging
import argparse

import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    # Basic Parameters
    'num_positions': 300,
    'output_path': '../sionna/simulation',
    
    # Area Configuration
    'area_size': 500,
    'bs_height': 25.0,
    'ue_height_min': 1.0,
    'ue_height_max': 3.0,
    
    # 5G OFDM Parameters
    'center_frequency': 3.5e9,  # 3.5 GHz
    'subcarrier_spacing': 30e3,  # 30 kHz
    'num_subcarriers': 408,
    'num_ue_antennas': 4,  # 商用手机1×2双极化 = 4通道
    'num_bs_antennas': 64,  # 4行8列双极化 = 64通道
    
    # Base Station Antenna Configuration
    # 4行8列双极化阵列，每个物理位置产生2个极化通道
    'bs_antenna_pattern': '4x8_dual_pol',
    'bs_antenna_rows': 4,
    'bs_antenna_cols': 8,
    
    # Ray Tracing Parameters
    'max_depth': 5,
    'num_samples': int(1e5),
    
    # Misc
    'random_seed': 42,
    'create_plots': True,
    'plot_dpi': 300,
    'preview_scene': True,  # Whether to create scene preview image
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RayTracingDataGenerator:
    def __init__(self, config=None):
        """Initialize the ray tracing data generator."""
        self.config = config if config is not None else DEFAULT_CONFIG
        
        # Set random seed for reproducibility
        np.random.seed(self.config['random_seed'])
        sionna.config.seed = self.config['random_seed']
        
        logger.info("Initialized Ray Tracing Data Generator")
        
    def load_campus_scene(self):
        """Load the campus environment scene and configure antennas."""
        scene_path = "./polyu1.xml"
        
        logger.info(f"Loading campus scene from: {scene_path}")
        self.scene = load_scene(scene_path)
        
        # Configure base station antenna array
        # 4行8列双极化阵列：物理阵子4行×8列，双极化产生64个信道
        bs_rows = self.config['bs_antenna_rows']  # 4行
        bs_cols = self.config['bs_antenna_cols']  # 8列
        
        # 双极化配置：每个物理位置产生2个极化通道
        actual_antennas = bs_rows * bs_cols * 2  # 4×8×2 = 64个极化通道
        expected_channels = 64  # 实际64个通道
        
        if expected_channels != self.config['num_bs_antennas']:
            logger.warning(f"Updating num_bs_antennas from {self.config['num_bs_antennas']} to {expected_channels}")
            self.config['num_bs_antennas'] = expected_channels
        
        # 基站天线阵列配置
        self.scene.tx_array = PlanarArray(
            num_rows=bs_rows,  # 4行
            num_cols=bs_cols,  # 8列  
            vertical_spacing=0.68,      # 垂直间距: 0.68λ (标准配置)
            horizontal_spacing=0.5,     # 水平间距: 0.5λ
            pattern="tr38901",          # 5G标准天线方向图
            polarization="cross"        # 交叉极化 (+45°/-45°) 实现双极化效果
        )
        
        # 记录实际创建的天线数量
        logger.info(f"BS antenna array: {bs_rows}×{bs_cols} with cross polarization")
        logger.info(f"Expected BS channels: {bs_rows * bs_cols * 2} = {expected_channels}")
        
        # Configure UE antenna array (商用手机)
        # 商用手机1×2双极化配置：2个物理天线，每个双极化产生4通道
        # 使用1行2列确保2个独立的物理天线位置，每个位置双极化
        ue_rows = 1  # 1行
        ue_cols = 2  # 2列，2个物理天线位置
        
        # 商用手机天线配置 - 1×2排列，双极化产生4通道
        self.scene.rx_array = PlanarArray(
            num_rows=ue_rows,  # 1行
            num_cols=ue_cols,  # 2列，确保2个独立的物理位置
            vertical_spacing=0.5,    # 垂直间距（此配置下不起作用）
            horizontal_spacing=0.7,  # 水平间距0.7λ，手机内天线间距
            pattern="iso",           # 全向天线方向图(手机天线特性)
            polarization="cross"     # 交叉极化，每个物理位置产生2个极化通道
        )
        
        # 记录实际创建的UE天线数量
        logger.info(f"UE antenna array: {ue_rows}×{ue_cols} with cross polarization")
        logger.info(f"Expected UE channels: {ue_rows * ue_cols * 2} = {self.config['num_ue_antennas']}")
        
        # Set carrier frequency
        self.scene.frequency = self.config['center_frequency']
        
        logger.info("Campus scene loaded successfully")
        
    def preview_scene(self, output_dir):
        """Create a preview image of the campus scene."""
        if not self.config['preview_scene']:
            return
            
        logger.info("Creating scene preview...")
        
        # Create a camera for scene visualization
        camera = Camera(
            name="preview_cam",
            position=[55, 55, 40],  # Position above the campus area
            look_at=[55, 55, 0]     # Look down at the campus center
        )
        self.scene.add(camera)
        
        # Render the scene
        try:
            # Sionna's render method returns a matplotlib figure
            try:
                # Method 1: Try with camera name
                fig = self.scene.render("preview_cam")
            except:
                # Method 2: Try with camera object
                fig = self.scene.render(camera)
            
            # Add title to the existing figure
            fig.suptitle('Campus Scene Preview', fontsize=16)
            
            # Save the preview image
            preview_path = output_dir / 'campus_scene_preview.png'
            fig.savefig(preview_path, dpi=self.config['plot_dpi'], bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Scene preview saved to {preview_path}")
            
        except Exception as e:
            logger.warning(f"Could not create scene preview: {e}")
            logger.info("Scene preview is optional and simulation will continue without it")
        finally:
            # Remove camera
            try:
                self.scene.remove("preview_cam")
            except:
                pass
        
    def generate_ue_positions(self):
        """Generate random UE positions in campus area."""
        num_positions = self.config['num_positions']
        ue_height_min = self.config['ue_height_min']
        ue_height_max = self.config['ue_height_max']
        
        logger.info(f"Generating {num_positions} UE positions")
        
        # UE positions distributed in campus coordinates
        ue_positions = np.zeros((num_positions, 3))
        ue_positions[:, 0] = np.random.uniform(10, 100, num_positions)  # X
        ue_positions[:, 1] = np.random.uniform(10, 100, num_positions)  # Y  
        ue_positions[:, 2] = np.random.uniform(ue_height_min, ue_height_max, num_positions)  # Z
        
        # Base station position (known working position in campus scene)
        bs_position = np.array([8.5, 21, 27])
        
        return ue_positions, bs_position
        
    def generate_ray_tracing_channels(self, ue_positions, bs_position):
        """Generate channel responses using Sionna ray tracing."""
        num_positions = len(ue_positions)
        num_subcarriers = self.config['num_subcarriers']
        num_ue_antennas = self.config['num_ue_antennas']
        num_bs_antennas = self.config['num_bs_antennas']
        subcarrier_spacing = self.config['subcarrier_spacing']
        
        logger.info(f"Generating ray tracing channels for {num_positions} positions")
        logger.info(f"Channel dimensions: ({num_positions}, {num_subcarriers}, {num_ue_antennas}, {num_bs_antennas})")
        
        # Initialize output arrays
        channel_responses = np.zeros((num_positions, num_subcarriers, num_ue_antennas, num_bs_antennas), dtype=complex)
        path_losses = np.zeros((num_positions, num_subcarriers))
        delays = np.zeros((num_positions, num_subcarriers))
        
        # Setup base station
        self.tx = Transmitter(name="BS", position=bs_position)
        self.scene.add(self.tx)
        
        # Generate frequency array for OFDM
        freq_array = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
        
        for pos_idx in range(num_positions):
            if (pos_idx + 1) % 50 == 0:
                logger.info(f"Processing position {pos_idx + 1}/{num_positions}")
            
            ue_pos = ue_positions[pos_idx]
            
            # Create receiver at UE position
            rx_name = f"UE_{pos_idx}"
            if rx_name in self.scene.receivers:
                self.scene.remove(rx_name)
                
            rx = Receiver(name=rx_name, position=ue_pos, orientation=[0,0,0])
            self.scene.add(rx)
            
            # Point transmitter towards receiver
            self.tx.look_at(rx)
            
            # Compute propagation paths using ray tracing
            paths = self.scene.compute_paths(
                max_depth=self.config['max_depth'],
                num_samples=self.config['num_samples']
            )
            
            # Convert to OFDM channel responses
            ofdm_channel = cir_to_ofdm_channel(
                frequencies=freq_array,
                a=paths.a,
                tau=paths.tau,
                normalize=True
            )
            
            # Extract channel matrix properly
            # ofdm_channel shape: [batch, num_tx, num_rx, num_time_symbols, num_ofdm_symbols, subcarriers]
            ofdm_numpy = ofdm_channel.numpy()
            
            # 调试：打印实际的维度
            if pos_idx == 0:
                logger.info(f"OFDM channel shape: {ofdm_numpy.shape}")
                logger.info(f"Expected MIMO: TX={num_bs_antennas}, RX={num_ue_antennas}")
                logger.info(f"Actual TX antennas: {ofdm_numpy.shape[1] if len(ofdm_numpy.shape) > 1 else 'N/A'}")
                logger.info(f"Actual RX antennas: {ofdm_numpy.shape[2] if len(ofdm_numpy.shape) > 2 else 'N/A'}")
                
            # Sionna返回格式分析: (1, 1, 4, 1, 64, 1, 408)
            # 很可能是: [batch, ?, num_rx, ?, num_tx, ?, subcarriers]
            # 其中 num_rx=4 (UE天线), num_tx=64 (BS天线), subcarriers=408
            if len(ofdm_numpy.shape) == 7:
                # 7维格式: 分析每个维度的含义
                # 形状 (1, 1, 4, 1, 64, 1, 408)
                # 维度 0: batch=1
                # 维度 1: ? =1 
                # 维度 2: num_rx=4 (UE天线)
                # 维度 3: ? =1
                # 维度 4: num_tx=64 (BS天线)  
                # 维度 5: ? =1
                # 维度 6: subcarriers=408
                
                # 提取: [num_tx, num_rx, subcarriers]
                h_mimo = ofdm_numpy[0, 0, :, 0, :, 0, :]  # [num_rx, num_tx, subcarriers] -> 需要转置
                h_mimo = np.transpose(h_mimo, (1, 0, 2))  # [num_tx, num_rx, subcarriers]
                
                if pos_idx == 0:
                    logger.info(f"Using 7D format extraction with transpose")
                    logger.info(f"Raw extracted shape before transpose: {ofdm_numpy[0, 0, :, 0, :, 0, :].shape}")
                    logger.info(f"Final h_mimo shape after transpose: {h_mimo.shape}")
                    
            elif len(ofdm_numpy.shape) >= 6:
                # 标准格式：取第一个batch, 第一个时间符号, 第一个OFDM符号
                h_mimo = ofdm_numpy[0, :, :, 0, 0, :]  # [num_tx, num_rx, subcarriers]
                if pos_idx == 0:
                    logger.info(f"Using 6D format extraction")
            else:
                logger.error(f"Unexpected OFDM shape with {len(ofdm_numpy.shape)} dimensions")
                # 改为填充零值而不是continue，确保不跳过处理
                h_mimo = np.zeros((num_bs_antennas, num_ue_antennas, num_subcarriers), dtype=complex)
                if pos_idx == 0:
                    logger.warning(f"Using zero-filled fallback for unexpected shape")
                
            if pos_idx == 0:
                logger.info(f"Extracted h_mimo shape: {h_mimo.shape}")
                logger.info(f"Actual dimensions: TX={h_mimo.shape[0]}, RX={h_mimo.shape[1]}, Subcarriers={h_mimo.shape[2]}")
                
                # 检查是否需要调整配置以匹配实际天线数量
                actual_tx = h_mimo.shape[0]
                actual_rx = h_mimo.shape[1]
                
                if actual_tx != num_bs_antennas:
                    logger.warning(f"BS antenna mismatch: expected {num_bs_antennas}, got {actual_tx}")
                    logger.warning(f"Updating num_bs_antennas to {actual_tx}")
                    num_bs_antennas = actual_tx
                    self.config['num_bs_antennas'] = actual_tx
                    
                if actual_rx != num_ue_antennas:
                    logger.warning(f"UE antenna mismatch: expected {num_ue_antennas}, got {actual_rx}")
                    logger.warning(f"Updating num_ue_antennas to {actual_rx}")
                    num_ue_antennas = actual_rx
                    self.config['num_ue_antennas'] = actual_rx
                    
                    # 重新分配 channel_responses 数组
                    logger.info(f"Reallocating channel_responses array with new dimensions")
                    new_shape = (num_positions, num_subcarriers, num_ue_antennas, num_bs_antennas)
                    channel_responses = np.zeros(new_shape, dtype=complex)
                
                logger.info(f"Updated MIMO configuration: {num_bs_antennas}×{num_ue_antennas}")
                for rx_idx in range(min(4, h_mimo.shape[1])):
                    power = np.mean(np.abs(h_mimo[:, rx_idx, 0])**2)
                    logger.info(f"  RX antenna {rx_idx}: average power = {power:.2e}")
                
            # 转换为所需格式 [subcarriers, ue_antennas, bs_antennas]
            for sc_idx in range(min(num_subcarriers, h_mimo.shape[2])):
                # 提取该子载波的信道矩阵 [num_tx, num_rx]
                h_matrix = h_mimo[:, :, sc_idx]  # [bs_antennas, ue_antennas]
                
                # 现在维度应该匹配，但为了安全起见还是检查一下
                if h_matrix.shape[0] != num_bs_antennas or h_matrix.shape[1] != num_ue_antennas:
                    if pos_idx == 0 and sc_idx == 0:
                        logger.error(f"Critical dimension mismatch after adjustment:")
                        logger.error(f"h_matrix shape: {h_matrix.shape}")
                        logger.error(f"Expected: [{num_bs_antennas}, {num_ue_antennas}]")
                        logger.error(f"This should not happen after the dynamic adjustment above")
                    
                    # 简单的截断或填充策略
                    h_correct = np.zeros((num_bs_antennas, num_ue_antennas), dtype=complex)
                    
                    # 计算实际可复制的维度
                    copy_tx = min(h_matrix.shape[0], num_bs_antennas)
                    copy_rx = min(h_matrix.shape[1], num_ue_antennas)
                    
                    if copy_tx > 0 and copy_rx > 0:
                        h_correct[:copy_tx, :copy_rx] = h_matrix[:copy_tx, :copy_rx]
                    
                    h_matrix = h_correct
                
                # Store in format [ue_antennas, bs_antennas]
                channel_responses[pos_idx, sc_idx, :, :] = h_matrix.T
                
                # Calculate path loss for this subcarrier
                channel_power = np.abs(h_matrix)**2
                path_loss_linear = np.mean(channel_power)
                if path_loss_linear > 0:
                    path_losses[pos_idx, sc_idx] = -10 * np.log10(path_loss_linear)
                else:
                    path_losses[pos_idx, sc_idx] = 150.0
                
                # Calculate delay for this subcarrier
                tau_array = paths.tau.numpy()[0, 0, 0, :]
                if len(tau_array) > 0:
                    delays[pos_idx, sc_idx] = np.mean(tau_array)
                else:
                    distance = np.linalg.norm(ue_pos - bs_position)
                    delays[pos_idx, sc_idx] = distance / 3e8
            
            # Remove receiver for next iteration
            self.scene.remove(rx_name)
            
        logger.info("Ray tracing channel generation completed")
        return channel_responses, path_losses, delays
        
    def save_simulation_data(self, channel_responses, path_losses, delays, ue_positions, bs_position, output_file):
        """Save simulation data to HDF5 file."""
        logger.info(f"Saving simulation data to {output_file}...")
        
        with h5py.File(output_file, 'w') as f:
            # Create groups
            config_group = f.create_group('antenna')
            data_group = f.create_group('data')
            
            # Save configuration
            for key, value in self.config.items():
                config_group.attrs[key] = value
            
            # Save positions
            data_group.create_dataset('bs_position', data=bs_position)
            data_group.create_dataset('ue_position', data=ue_positions)
            
            # Save channel data
            data_group.create_dataset('channel_responses', data=channel_responses)
        
        logger.info("Data saved successfully!")
        
    def save_tensor_data(self, channel_responses, ue_positions, bs_position, output_file):
        """Save channel data in tensor format [数量, 基站天线, UE天线, 子载波]."""
        logger.info(f"Saving tensor data to {output_file}...")
        
        # Transpose to required format: [positions, bs_antennas, ue_antennas, subcarriers]
        # Input: [positions, subcarriers, ue_antennas, bs_antennas]
        # Output: [positions, bs_antennas, ue_antennas, subcarriers]
        tensor_data = np.transpose(channel_responses, (0, 3, 2, 1))
        
        # Save as numpy file
        numpy_file = str(output_file).replace('.t', '.npy')
        np.savez(numpy_file, 
                 channel_responses=tensor_data,
                 ue_positions=ue_positions,
                 bs_position=bs_position)
        
        # Save as text file with basic info
        with open(output_file, 'w') as f:
            f.write(f"# Tensor data format: [数量({tensor_data.shape[0]}), 基站天线({tensor_data.shape[1]}), UE天线({tensor_data.shape[2]}), 子载波({tensor_data.shape[3]})]\n")
            f.write(f"# Data shape: {tensor_data.shape}\n")
            f.write(f"# MIMO配置: 64×4 (64个基站天线 × 4个UE天线)\n")
            f.write(f"# BS position: [{bs_position[0]:.6f}, {bs_position[1]:.6f}, {bs_position[2]:.6f}]\n")
            f.write(f"# Use corresponding .npy.npz file for actual data\n")
        
        logger.info(f"Tensor data saved: shape {tensor_data.shape}")
        return tensor_data

    def create_visualizations(self, channel_responses, path_losses, delays, ue_positions, bs_position, output_dir):
        """Create visualization plots."""
        if not self.config['create_plots']:
            return
        
        logger.info("Creating visualizations...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Ray Tracing 5G OFDM Simulation Results ({len(ue_positions)} positions)', fontsize=16)
        
        # UE positions
        ax1 = axes[0]
        ax1.scatter(ue_positions[:, 0], ue_positions[:, 1], c='blue', alpha=0.6, s=20, label='UE Positions')
        ax1.scatter(bs_position[0], bs_position[1], c='red', s=100, marker='^', label='BS')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('UE and BS Positions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Channel magnitude for all UE positions
        ax2 = axes[1]
        num_positions = len(ue_positions)
        colors = plt.cm.viridis(np.linspace(0, 1, num_positions))
        
        for pos_idx in range(num_positions):
            # 使用第一个UE天线和第一个BS天线的信道
            channel_mag = np.abs(channel_responses[pos_idx, :, 0, 0])
            ax2.plot(channel_mag, color=colors[pos_idx], alpha=0.7, linewidth=0.8)
        
        ax2.set_xlabel('Subcarrier Index')
        ax2.set_ylabel('Channel Magnitude')
        ax2.set_title(f'Channel Magnitude (All {num_positions} UE Positions)')
        ax2.grid(True, alpha=0.3)
        
        # Channel phase for all UE positions
        ax3 = axes[2]
        for pos_idx in range(num_positions):
            # 使用第一个UE天线和第一个BS天线的信道
            channel_phase = np.angle(channel_responses[pos_idx, :, 0, 0])
            ax3.plot(channel_phase, color=colors[pos_idx], alpha=0.7, linewidth=0.8)
        
        ax3.set_xlabel('Subcarrier Index')
        ax3.set_ylabel('Channel Phase (radians)')
        ax3.set_title(f'Channel Phase (All {num_positions} UE Positions)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_plot = output_dir / 'ray_tracing_simulation_results.png'
        plt.savefig(output_plot, dpi=self.config['plot_dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_plot}")
        
    def run_simulation(self, output_path):
        """Run the complete ray tracing simulation pipeline."""
        logger.info("=== Starting Ray Tracing 5G OFDM Data Generation ===")
        start_time = time.time()
        
        # Load scene and setup
        self.load_campus_scene()
        
        # Create scene preview
        self.preview_scene(output_path)
        
        # Generate positions
        ue_positions, bs_position = self.generate_ue_positions()
        
        # Generate channels
        channel_responses, path_losses, delays = self.generate_ray_tracing_channels(ue_positions, bs_position)
        
        # Save data
        output_file = output_path / f"ray_tracing_5g_simulation_P{self.config['num_positions']}.h5"
        self.save_simulation_data(channel_responses, path_losses, delays, ue_positions, bs_position, output_file)
        
        # Save tensor data
        tensor_output_file = output_path / f"ray_tracing_5g_simulation_P{self.config['num_positions']}.t"
        tensor_data = self.save_tensor_data(channel_responses, ue_positions, bs_position, tensor_output_file)
        
        # Create visualizations
        self.create_visualizations(channel_responses, path_losses, delays, ue_positions, bs_position, output_path)
        
        simulation_time = time.time() - start_time
        
        logger.info("=== Simulation completed successfully! ===")
        logger.info(f"Time: {simulation_time:.2f} seconds")
        logger.info(f"Output files: {output_file}, {tensor_output_file}")
        logger.info(f"Channel shape: {channel_responses.shape}")
        logger.info(f"Tensor format: [数量={tensor_data.shape[0]}, 基站天线={tensor_data.shape[1]}, UE天线={tensor_data.shape[2]}, 子载波={tensor_data.shape[3]}]")
        logger.info(f"MIMO配置验证: 基站{tensor_data.shape[1]}天线 × UE{tensor_data.shape[2]}天线 = {tensor_data.shape[1]}×{tensor_data.shape[2]} MIMO")
        
        return output_path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Ray Tracing 5G OFDM Data Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-n', '--num', type=int, required=True, help='Number of UE positions')
    parser.add_argument('--output_path', type=str, default='./sionna_results', help='Output path')
    parser.add_argument('--area_size', type=float, default=DEFAULT_CONFIG['area_size'], help='Area size (m)')
    parser.add_argument('--bs_height', type=float, default=DEFAULT_CONFIG['bs_height'], help='BS height (m)')
    parser.add_argument('--center_frequency', type=float, default=DEFAULT_CONFIG['center_frequency'], help='Center frequency (Hz)')
    parser.add_argument('--num_subcarriers', type=int, default=DEFAULT_CONFIG['num_subcarriers'], help='Number of subcarriers')
    parser.add_argument('--num_bs_antennas', type=int, default=DEFAULT_CONFIG['num_bs_antennas'], help='Number of BS antennas')
    parser.add_argument('--num_ue_antennas', type=int, default=DEFAULT_CONFIG['num_ue_antennas'], help='Number of UE antennas')
    parser.add_argument('--bs_antenna_pattern', type=str, default=DEFAULT_CONFIG['bs_antenna_pattern'], help='BS antenna pattern')
    parser.add_argument('--bs_antenna_rows', type=int, default=DEFAULT_CONFIG['bs_antenna_rows'], help='BS antenna rows')
    parser.add_argument('--bs_antenna_cols', type=int, default=DEFAULT_CONFIG['bs_antenna_cols'], help='BS antenna columns')
    parser.add_argument('--random_seed', type=int, default=DEFAULT_CONFIG['random_seed'], help='Random seed')
    parser.add_argument('--no_plots', action='store_true', help='Disable plots')
    parser.add_argument('--no_scene_preview', action='store_true', help='Disable scene preview')
    
    return parser.parse_args()

def main():
    """Main simulation function."""
    args = parse_arguments()
    
    # Create configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        'num_positions': args.num,
        'area_size': args.area_size,
        'bs_height': args.bs_height,
        'center_frequency': args.center_frequency,
        'num_subcarriers': args.num_subcarriers,
        'num_bs_antennas': args.num_bs_antennas,
        'num_ue_antennas': args.num_ue_antennas,
        'bs_antenna_pattern': args.bs_antenna_pattern,
        'bs_antenna_rows': args.bs_antenna_rows,
        'bs_antenna_cols': args.bs_antenna_cols,
        'random_seed': args.random_seed,
        'create_plots': not args.no_plots,
        'preview_scene': not args.no_scene_preview,
        'output_path': args.output_path
    })
    
    print(f"=== Ray Tracing 5G OFDM Data Generation ({config['num_positions']} positions) ===")
    print(f"基站天线: {config['bs_antenna_pattern']} ({config['bs_antenna_rows']}行×{config['bs_antenna_cols']}列双极化)")
    print(f"基站通道: {config['num_bs_antennas']}个 (32物理天线×2极化)")
    print(f"UE天线: 商用手机1×2双极化 {config['num_ue_antennas']}通道 (2物理天线×2极化)")
    print(f"MIMO配置: {config['num_bs_antennas']}×{config['num_ue_antennas']}")
    print(f"Output path: {config['output_path']}")
    print()
    
    # Create output directory
    data_dir = Path(config['output_path'])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Run simulation
    generator = RayTracingDataGenerator(config)
    output_dir = generator.run_simulation(data_dir)
    
    print(f"\nSimulation completed! Results saved to: {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise