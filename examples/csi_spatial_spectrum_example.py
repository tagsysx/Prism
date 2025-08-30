#!/usr/bin/env python3
"""
CSI空间谱生成示例

本示例展示了如何使用Prism项目生成的CSI数据来计算空间谱，
包括Bartlett、CAPON和MUSIC等不同的波束形成算法。

作者: Prism项目团队
日期: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import sys
import os

# 添加Prism源代码路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from prism.ray_tracer_cpu import RayTracerCPU
    from prism.config_loader import ConfigLoader
    PRISM_AVAILABLE = True
except ImportError:
    print("警告: 无法导入Prism模块，将使用模拟数据")
    PRISM_AVAILABLE = False


class CSISpatialSpectrum:
    """CSI空间谱计算类"""
    
    def __init__(self, M, N, freq, dx=None, dy=None):
        """
        初始化CSI空间谱计算器
        
        参数:
        M: 天线阵列x方向大小
        N: 天线阵列y方向大小
        freq: 载波频率 (Hz)
        dx, dy: 天线间距，默认为半波长间距
        """
        self.M = M
        self.N = N
        self.freq = freq
        self.wavelength = 3e8 / freq
        self.dx = dx if dx is not None else 0.5 * self.wavelength
        self.dy = dy if dy is not None else 0.5 * self.wavelength
        
    def calculate_steering_vector(self, theta, phi):
        """
        计算导向矢量
        
        参数:
        theta: 俯仰角 (弧度)
        phi: 方位角 (弧度)
        
        返回:
        a: 导向矢量，形状为 (M*N, 1)
        """
        a = np.zeros((self.M, self.N), dtype=complex)
        
        for m in range(self.M):
            for n in range(self.N):
                phase_shift = 2 * np.pi * (
                    (m * self.dx * np.sin(theta) * np.cos(phi)) / self.wavelength +
                    (n * self.dy * np.sin(theta) * np.sin(phi)) / self.wavelength
                )
                a[m, n] = np.exp(-1j * phase_shift)
        
        return a.reshape(-1, 1)
    
    def bartlett_beamformer(self, csi, theta_grid, phi_grid):
        """
        Bartlett波束形成器
        
        参数:
        csi: CSI矩阵，形状为 (M*N, T)
        theta_grid: 俯仰角网格
        phi_grid: 方位角网格
        
        返回:
        P_spectrum: 空间谱矩阵
        """
        T = csi.shape[1]
        R_xx = (csi @ csi.conj().T) / T
        
        P_spectrum = np.zeros((len(theta_grid), len(phi_grid)))
        
        for i, theta in enumerate(theta_grid):
            for j, phi in enumerate(phi_grid):
                a_vec = self.calculate_steering_vector(theta, phi)
                power = np.real(a_vec.conj().T @ R_xx @ a_vec)
                P_spectrum[i, j] = power.squeeze()
                
        return P_spectrum
    
    def capon_beamformer(self, csi, theta_grid, phi_grid, reg_factor=1e-6):
        """
        CAPON波束形成器 (MVDR)
        
        参数:
        csi: CSI矩阵，形状为 (M*N, T)
        theta_grid: 俯仰角网格
        phi_grid: 方位角网格
        reg_factor: 正则化因子，防止矩阵奇异
        
        返回:
        P_spectrum: 空间谱矩阵
        """
        T = csi.shape[1]
        R_xx = (csi @ csi.conj().T) / T
        
        # 添加正则化项
        R_xx += reg_factor * np.eye(R_xx.shape[0])
        R_xx_inv = np.linalg.inv(R_xx)
        
        P_spectrum = np.zeros((len(theta_grid), len(phi_grid)))
        
        for i, theta in enumerate(theta_grid):
            for j, phi in enumerate(phi_grid):
                a_vec = self.calculate_steering_vector(theta, phi)
                denominator = np.real(a_vec.conj().T @ R_xx_inv @ a_vec)
                P_spectrum[i, j] = 1.0 / (denominator + 1e-10)
                
        return P_spectrum
    
    def music_algorithm(self, csi, theta_grid, phi_grid, num_sources):
        """
        MUSIC算法
        
        参数:
        csi: CSI矩阵，形状为 (M*N, T)
        theta_grid: 俯仰角网格
        phi_grid: 方位角网格
        num_sources: 信源数量
        
        返回:
        P_spectrum: 空间谱矩阵
        """
        T = csi.shape[1]
        R_xx = (csi @ csi.conj().T) / T
        
        # 特征值分解
        eigenvals, eigenvecs = eigh(R_xx)
        
        # 按特征值降序排列
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # 噪声子空间
        noise_subspace = eigenvecs[:, num_sources:]
        
        P_spectrum = np.zeros((len(theta_grid), len(phi_grid)))
        
        for i, theta in enumerate(theta_grid):
            for j, phi in enumerate(phi_grid):
                a_vec = self.calculate_steering_vector(theta, phi)
                denominator = np.real(a_vec.conj().T @ noise_subspace @ noise_subspace.conj().T @ a_vec)
                P_spectrum[i, j] = 1.0 / (denominator + 1e-10)
                
        return P_spectrum


def generate_simulated_csi(N, num_sources=2, snr_db=20, num_snapshots=100):
    """
    生成模拟CSI数据
    
    参数:
    N: 天线阵列大小
    num_sources: 信源数量
    snr_db: 信噪比 (dB)
    num_snapshots: 快拍数量
    
    返回:
    csi: 模拟的CSI数据
    true_angles: 真实的信号到达角度
    """
    freq = 5e9  # 5 GHz
    spectrum_calc = CSISpatialSpectrum(N, freq)
    
    # 随机生成信源角度
    np.random.seed(42)
    true_thetas = np.random.uniform(-np.pi/4, np.pi/4, num_sources)
    true_phis = np.random.uniform(0, 2*np.pi, num_sources)
    true_angles = list(zip(true_thetas, true_phis))
    
    # 生成信号
    signals = np.random.randn(num_sources, num_snapshots) + 1j * np.random.randn(num_sources, num_snapshots)
    
    # 构建阵列流形矩阵
    A = np.zeros((N*N, num_sources), dtype=complex)
    for i, (theta, phi) in enumerate(true_angles):
        A[:, i] = spectrum_calc.calculate_steering_vector(theta, phi).squeeze()
    
    # 接收信号
    received_signals = A @ signals
    
    # 添加噪声
    noise_power = 10**(-snr_db/10)
    noise = np.sqrt(noise_power/2) * (np.random.randn(N*N, num_snapshots) + 
                                     1j * np.random.randn(N*N, num_snapshots))
    
    csi = received_signals + noise
    
    return csi, true_angles


def load_prism_csi_data(config_path):
    """
    从Prism项目加载CSI数据
    
    参数:
    config_path: 配置文件路径
    
    返回:
    csi: CSI数据
    """
    if not PRISM_AVAILABLE:
        print("Prism模块不可用，使用模拟数据")
        return None
    
    try:
        # 加载配置
        config = ConfigLoader.load_config(config_path)
        
        # 创建射线追踪器
        ray_tracer = RayTracerCPU(config)
        
        # 这里应该根据实际的Prism API来获取CSI数据
        # 以下是示例代码，需要根据实际情况调整
        print("注意: 这是示例代码，需要根据实际Prism API调整")
        
        # 示例：假设我们有UE位置和BS配置
        ue_positions = np.array([[10.0, 5.0, 1.5]])  # 示例UE位置
        
        # 执行射线追踪（这需要根据实际API调整）
        # results = ray_tracer.trace_rays(ue_positions)
        # csi = results['csi']  # 提取CSI数据
        
        return None  # 暂时返回None，需要实际实现
        
    except Exception as e:
        print(f"加载Prism数据时出错: {e}")
        return None


def plot_spatial_spectrum_comparison(spectrums, titles, theta_grid, phi_grid, true_angles=None):
    """
    比较不同算法的空间谱结果
    
    参数:
    spectrums: 空间谱列表
    titles: 标题列表
    theta_grid: 俯仰角网格
    phi_grid: 方位角网格
    true_angles: 真实角度（可选）
    """
    fig, axes = plt.subplots(1, len(spectrums), figsize=(5*len(spectrums), 4))
    if len(spectrums) == 1:
        axes = [axes]
    
    for i, (spectrum, title) in enumerate(zip(spectrums, titles)):
        # 转换为dB
        spectrum_db = 10 * np.log10(spectrum + 1e-10)
        
        im = axes[i].imshow(spectrum_db, aspect='auto', origin='lower',
                           extent=[np.degrees(phi_grid[0]), np.degrees(phi_grid[-1]), 
                                  np.degrees(theta_grid[0]), np.degrees(theta_grid[-1])],
                           cmap='viridis')
        
        axes[i].set_xlabel('方位角 φ (度)')
        axes[i].set_ylabel('俯仰角 θ (度)')
        axes[i].set_title(title)
        axes[i].grid(True, alpha=0.3)
        
        # 标记真实角度
        if true_angles is not None:
            for theta, phi in true_angles:
                axes[i].plot(np.degrees(phi), np.degrees(theta), 'r*', markersize=15, 
                           markeredgecolor='white', markeredgewidth=2, label='真实位置')
            if i == 0:  # 只在第一个子图显示图例
                axes[i].legend()
        
        plt.colorbar(im, ax=axes[i], label='功率 (dB)')
    
    plt.tight_layout()
    plt.show()


def find_peaks(spectrum, theta_grid, phi_grid, num_peaks=2):
    """
    在空间谱中找到峰值
    
    参数:
    spectrum: 空间谱矩阵
    theta_grid: 俯仰角网格
    phi_grid: 方位角网格
    num_peaks: 要找的峰值数量
    
    返回:
    peaks: 峰值位置列表 [(theta, phi), ...]
    """
    # 找到最大值位置
    flat_spectrum = spectrum.flatten()
    peak_indices = np.argpartition(flat_spectrum, -num_peaks)[-num_peaks:]
    peak_indices = peak_indices[np.argsort(flat_spectrum[peak_indices])[::-1]]
    
    peaks = []
    for idx in peak_indices:
        i, j = np.unravel_index(idx, spectrum.shape)
        theta_est = theta_grid[i]
        phi_est = phi_grid[j]
        peaks.append((theta_est, phi_est))
    
    return peaks


def main():
    """主函数"""
    print("CSI空间谱生成示例")
    print("=" * 50)
    
    # 参数设置
    N = 8  # 8×8天线阵列
    freq = 5e9  # 5 GHz
    num_sources = 2
    snr_db = 20
    num_snapshots = 100
    
    # 角度网格
    A, B = 60, 120  # 角度分辨率
    theta_range = np.linspace(-np.pi/4, np.pi/4, A)  # -45°到45°
    phi_range = np.linspace(0, 2*np.pi, B)  # 完整360°
    
    print(f"天线阵列: {N}×{N}")
    print(f"载波频率: {freq/1e9:.1f} GHz")
    print(f"信源数量: {num_sources}")
    print(f"信噪比: {snr_db} dB")
    print(f"快拍数量: {num_snapshots}")
    print(f"角度分辨率: {A}×{B}")
    
    # 创建空间谱计算器
    spectrum_calc = CSISpatialSpectrum(N, freq)
    
    # 生成或加载CSI数据
    print("\n正在生成CSI数据...")
    
    # 尝试从Prism加载数据
    config_path = "../configs/ofdm-5g-sionna.yml"
    if os.path.exists(config_path):
        csi_data = load_prism_csi_data(config_path)
    else:
        csi_data = None
    
    # 如果无法从Prism加载，使用模拟数据
    if csi_data is None:
        print("使用模拟CSI数据")
        csi_data, true_angles = generate_simulated_csi(N, num_sources, snr_db, num_snapshots)
        print(f"真实信号方向:")
        for i, (theta, phi) in enumerate(true_angles):
            print(f"  信源 {i+1}: θ={np.degrees(theta):.1f}°, φ={np.degrees(phi):.1f}°")
    else:
        true_angles = None
        print("使用Prism生成的CSI数据")
    
    # 计算不同算法的空间谱
    print("\n正在计算空间谱...")
    
    print("  - Bartlett波束形成器...")
    bartlett_spectrum = spectrum_calc.bartlett_beamformer(csi_data, theta_range, phi_range)
    
    print("  - CAPON波束形成器...")
    capon_spectrum = spectrum_calc.capon_beamformer(csi_data, theta_range, phi_range)
    
    if num_snapshots > num_sources:  # MUSIC需要足够的快拍数
        print("  - MUSIC算法...")
        music_spectrum = spectrum_calc.music_algorithm(csi_data, theta_range, phi_range, num_sources)
        spectrums = [bartlett_spectrum, capon_spectrum, music_spectrum]
        titles = ['Bartlett波束形成器', 'CAPON波束形成器', 'MUSIC算法']
    else:
        spectrums = [bartlett_spectrum, capon_spectrum]
        titles = ['Bartlett波束形成器', 'CAPON波束形成器']
    
    # 绘制结果
    print("\n正在绘制结果...")
    plot_spatial_spectrum_comparison(spectrums, titles, theta_range, phi_range, true_angles)
    
    # 峰值检测
    print("\n峰值检测结果:")
    for i, (spectrum, title) in enumerate(zip(spectrums, titles)):
        peaks = find_peaks(spectrum, theta_range, phi_range, num_sources)
        print(f"\n{title}:")
        for j, (theta_est, phi_est) in enumerate(peaks):
            print(f"  峰值 {j+1}: θ={np.degrees(theta_est):.1f}°, φ={np.degrees(phi_est):.1f}°")
            
            # 如果有真实角度，计算误差
            if true_angles is not None and j < len(true_angles):
                theta_true, phi_true = true_angles[j]
                theta_error = np.degrees(abs(theta_est - theta_true))
                phi_error = np.degrees(abs(phi_est - phi_true))
                print(f"    误差: Δθ={theta_error:.1f}°, Δφ={phi_error:.1f}°")
    
    print("\n示例完成！")


if __name__ == "__main__":
    main()
