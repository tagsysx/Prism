"""
CSI空间谱计算模块

本模块实现了基于信道状态信息（CSI）的空间谱估计功能，支持：
- 均匀平面阵列（UPA）导向矢量计算
- 多种空间谱估计算法（Bartlett、CAPON、MUSIC）
- 多子载波频谱融合
- 2D空间谱可视化

主要函数：
- calculate_spatial_spectrum: 计算单子载波空间谱
- calculate_multicarrier_spectrum: 计算多子载波融合空间谱
- generate_steering_vector: 生成导向矢量
- plot_spatial_spectrum: 绘制空间谱热力图
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
import warnings


def generate_steering_vector(
    theta: float, 
    phi: float, 
    M: int, 
    N: int, 
    dx: float, 
    dy: float, 
    wavelength: float
) -> np.ndarray:
    """
    计算均匀平面阵列（UPA）的导向矢量
    
    参数:
    theta: 俯仰角 (弧度)，范围 [-π/2, π/2]
    phi: 方位角 (弧度)，范围 [0, 2π]
    M: 天线阵列x方向大小（行数）
    N: 天线阵列y方向大小（列数）
    dx: x方向天线间距 (米)
    dy: y方向天线间距 (米)
    wavelength: 信号波长 (米)
    
    返回:
    a: 导向矢量，形状为 (M*N, 1)
    """
    a = np.zeros((M, N), dtype=complex)
    
    for m in range(M):  # x方向索引
        for n in range(N):  # y方向索引
            phase_shift = 2 * np.pi * (
                (m * dx * np.sin(theta) * np.cos(phi)) / wavelength +
                (n * dy * np.sin(theta) * np.sin(phi)) / wavelength
            )
            a[m, n] = np.exp(-1j * phase_shift)
    
    return a.reshape(-1, 1)  # 展平为 (M*N, 1) 向量


def calculate_bartlett_spectrum(
    csi: np.ndarray,
    M: int,
    N: int,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    wavelength: float,
    dx: Optional[float] = None,
    dy: Optional[float] = None
) -> np.ndarray:
    """
    使用Bartlett波束形成器计算空间谱
    
    参数:
    csi: CSI矩阵，形状为 (M*N, T)，T为快拍数
    M: 天线阵列x方向大小
    N: 天线阵列y方向大小
    theta_grid: 俯仰角网格，1D数组
    phi_grid: 方位角网格，1D数组
    wavelength: 信号波长 (米)
    dx, dy: 天线间距，默认为半波长间距
    
    返回:
    P_spectrum: 空间谱矩阵，形状为 (len(theta_grid), len(phi_grid))
    """
    T = csi.shape[1]
    
    # 默认半波长间距
    if dx is None:
        dx = 0.5 * wavelength
    if dy is None:
        dy = 0.5 * wavelength
    
    # 计算协方差矩阵
    R_xx = (csi @ csi.conj().T) / T  # 形状 (M*N, M*N)
    
    # 初始化空间谱
    P_spectrum = np.zeros((len(theta_grid), len(phi_grid)))
    
    # 角度网格扫描
    for i, theta in enumerate(theta_grid):
        for j, phi in enumerate(phi_grid):
            # 构建导向矢量
            a_vec = generate_steering_vector(theta, phi, M, N, dx, dy, wavelength)
            
            # Bartlett波束形成
            power = np.real(a_vec.conj().T @ R_xx @ a_vec)
            P_spectrum[i, j] = power.squeeze()
    
    return P_spectrum


def calculate_capon_spectrum(
    csi: np.ndarray,
    M: int,
    N: int,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    wavelength: float,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    regularization: float = 1e-6
) -> np.ndarray:
    """
    使用CAPON波束形成器计算空间谱（MVDR）
    
    参数:
    csi: CSI矩阵，形状为 (M*N, T)
    M: 天线阵列x方向大小
    N: 天线阵列y方向大小
    theta_grid: 俯仰角网格
    phi_grid: 方位角网格
    wavelength: 信号波长
    dx, dy: 天线间距
    regularization: 正则化参数，用于矩阵求逆的数值稳定性
    
    返回:
    P_spectrum: 空间谱矩阵
    """
    T = csi.shape[1]
    
    if dx is None:
        dx = 0.5 * wavelength
    if dy is None:
        dy = 0.5 * wavelength
    
    # 计算协方差矩阵
    R_xx = (csi @ csi.conj().T) / T
    
    # 添加正则化项确保矩阵可逆
    R_xx_reg = R_xx + regularization * np.eye(R_xx.shape[0])
    
    # 计算协方差矩阵的逆
    try:
        R_xx_inv = np.linalg.inv(R_xx_reg)
    except np.linalg.LinAlgError:
        warnings.warn("协方差矩阵求逆失败，使用伪逆")
        R_xx_inv = np.linalg.pinv(R_xx_reg)
    
    # 初始化空间谱
    P_spectrum = np.zeros((len(theta_grid), len(phi_grid)))
    
    # 角度网格扫描
    for i, theta in enumerate(theta_grid):
        for j, phi in enumerate(phi_grid):
            a_vec = generate_steering_vector(theta, phi, M, N, dx, dy, wavelength)
            
            # CAPON波束形成
            denominator = a_vec.conj().T @ R_xx_inv @ a_vec
            if np.abs(denominator) > 1e-10:
                power = 1.0 / np.real(denominator)
            else:
                power = 0.0
            P_spectrum[i, j] = power.squeeze()
    
    return P_spectrum


def calculate_music_spectrum(
    csi: np.ndarray,
    M: int,
    N: int,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    wavelength: float,
    num_sources: int,
    dx: Optional[float] = None,
    dy: Optional[float] = None
) -> np.ndarray:
    """
    使用MUSIC算法计算空间谱
    
    参数:
    csi: CSI矩阵，形状为 (M*N, T)
    M: 天线阵列x方向大小
    N: 天线阵列y方向大小
    theta_grid: 俯仰角网格
    phi_grid: 方位角网格
    wavelength: 信号波长
    num_sources: 信号源数量
    dx, dy: 天线间距
    
    返回:
    P_spectrum: 空间谱矩阵
    """
    T = csi.shape[1]
    MN = M * N
    
    if dx is None:
        dx = 0.5 * wavelength
    if dy is None:
        dy = 0.5 * wavelength
    
    # 计算协方差矩阵
    R_xx = (csi @ csi.conj().T) / T
    
    # 特征值分解
    eigenvals, eigenvecs = np.linalg.eigh(R_xx)
    
    # 按特征值降序排列
    sorted_indices = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[sorted_indices]
    eigenvecs = eigenvecs[:, sorted_indices]
    
    # 分离信号子空间和噪声子空间
    signal_subspace = eigenvecs[:, :num_sources]
    noise_subspace = eigenvecs[:, num_sources:]
    
    # 初始化空间谱
    P_spectrum = np.zeros((len(theta_grid), len(phi_grid)))
    
    # 角度网格扫描
    for i, theta in enumerate(theta_grid):
        for j, phi in enumerate(phi_grid):
            a_vec = generate_steering_vector(theta, phi, M, N, dx, dy, wavelength)
            
            # MUSIC算法
            denominator = a_vec.conj().T @ noise_subspace @ noise_subspace.conj().T @ a_vec
            if np.abs(denominator) > 1e-10:
                power = 1.0 / np.real(denominator)
            else:
                power = 0.0
            P_spectrum[i, j] = power.squeeze()
    
    return P_spectrum


def fuse_subcarrier_spectrums(
    spectrums: List[np.ndarray], 
    method: str = 'average', 
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    融合多个子载波的空间谱
    
    参数:
    spectrums: 子载波空间谱列表，每个元素形状为 (A, B)
    method: 融合方法 ('average', 'weighted', 'max', 'coherent')
    weights: 权重数组，仅在method='weighted'时使用
    
    返回:
    fused_spectrum: 融合后的空间谱，形状为 (A, B)
    """
    spectrums = np.array(spectrums)  # 形状 (K, A, B)
    
    if method == 'average':
        return np.mean(spectrums, axis=0)
    elif method == 'weighted':
        if weights is None:
            weights = np.ones(len(spectrums)) / len(spectrums)
        weights = np.array(weights).reshape(-1, 1, 1)  # 广播形状 (K, 1, 1)
        return np.sum(weights * spectrums, axis=0)
    elif method == 'max':
        return np.max(spectrums, axis=0)
    elif method == 'coherent':
        # 相干融合：对功率谱进行几何平均
        return np.exp(np.mean(np.log(spectrums + 1e-10), axis=0))
    else:
        raise ValueError(f"未知的融合方法: {method}")


def calculate_spatial_spectrum(
    csi_data: Union[np.ndarray, List[np.ndarray]],
    antenna_config: str,
    center_freq: float,
    bandwidth: float,
    subcarrier_indices: List[int],
    theta_range: Tuple[float, float, int],
    phi_range: Tuple[float, float, int],
    algorithm: str = 'bartlett',
    fusion_method: str = 'average',
    weights: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[np.ndarray, List[np.ndarray], dict]:
    """
    从CSI数据计算空间谱的主函数
    
    参数:
    csi_data: CSI数据，可以是：
              - 形状为 (M*N, T, K) 的3D数组，其中K是子载波数
              - 子载波CSI列表 [csi_1, csi_2, ..., csi_K]
    antenna_config: 天线配置字符串，如 "8x8" 表示8行8列
    center_freq: 中心频率 (Hz)
    bandwidth: 带宽 (Hz)
    subcarrier_indices: 子载波索引列表
    theta_range: 俯仰角范围 (min, max, num_points)
    phi_range: 方位角范围 (min, min, num_points)
    algorithm: 空间谱估计算法 ('bartlett', 'capon', 'music')
    fusion_method: 多子载波融合方法
    weights: 融合权重
    **kwargs: 其他参数，如num_sources（MUSIC算法需要）
    
    返回:
    fused_spectrum: 融合后的空间谱
    individual_spectrums: 各子载波的空间谱列表
    metadata: 计算元数据
    """
    # 解析天线配置
    M, N = map(int, antenna_config.split('x'))
    
    # 计算波长和天线间距
    wavelength = 3e8 / center_freq
    dx = dy = 0.5 * wavelength  # 半波长间距
    
    # 生成角度网格
    theta_min, theta_max, theta_points = theta_range
    phi_min, phi_max, phi_points = phi_range
    
    theta_grid = np.linspace(theta_min, theta_max, theta_points)
    phi_grid = np.linspace(phi_min, phi_max, phi_points)
    
    # 生成子载波频率
    K = len(subcarrier_indices)
    frequencies = []
    for idx in subcarrier_indices:
        # 假设子载波均匀分布在带宽内
        freq_offset = (idx - K//2) * (bandwidth / K)
        freq = center_freq + freq_offset
        frequencies.append(freq)
    
    # 处理CSI数据格式
    if isinstance(csi_data, np.ndarray):
        if csi_data.ndim == 3:  # (M*N, T, K)
            csi_subcarriers = [csi_data[:, :, k] for k in range(K)]
        else:
            raise ValueError("CSI数据必须是3D数组 (M*N, T, K) 或子载波列表")
    else:
        csi_subcarriers = csi_data
    
    # 验证数据维度
    for i, csi_k in enumerate(csi_subcarriers):
        if csi_k.shape[0] != M * N:
            raise ValueError(f"子载波 {i} 的天线维度不匹配: 期望 {M*N}, 实际 {csi_k.shape[0]}")
    
    # 计算每个子载波的空间谱
    individual_spectrums = []
    
    for k, (csi_k, freq_k) in enumerate(zip(csi_subcarriers, frequencies)):
        wavelength_k = 3e8 / freq_k
        
        if algorithm == 'bartlett':
            spectrum_k = calculate_bartlett_spectrum(
                csi_k, M, N, theta_grid, phi_grid, wavelength_k, dx, dy
            )
        elif algorithm == 'capon':
            spectrum_k = calculate_capon_spectrum(
                csi_k, M, N, theta_grid, phi_grid, wavelength_k, dx, dy
            )
        elif algorithm == 'music':
            if 'num_sources' not in kwargs:
                raise ValueError("MUSIC算法需要指定num_sources参数")
            spectrum_k = calculate_music_spectrum(
                csi_k, M, N, theta_grid, phi_grid, wavelength_k, 
                kwargs['num_sources'], dx, dy
            )
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        individual_spectrums.append(spectrum_k)
    
    # 融合空间谱
    fused_spectrum = fuse_subcarrier_spectrums(
        individual_spectrums, fusion_method, weights
    )
    
    # 构建元数据
    metadata = {
        'antenna_config': antenna_config,
        'M': M,
        'N': N,
        'center_freq': center_freq,
        'bandwidth': bandwidth,
        'subcarrier_indices': subcarrier_indices,
        'frequencies': frequencies,
        'theta_grid': theta_grid,
        'phi_grid': phi_grid,
        'algorithm': algorithm,
        'fusion_method': fusion_method,
        'dx': dx,
        'dy': dy,
        'wavelength': wavelength
    }
    
    return fused_spectrum, individual_spectrums, metadata


def plot_spatial_spectrum(
    spectrum: np.ndarray,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    title: str = "2D空间谱",
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    绘制空间谱热力图
    
    参数:
    spectrum: 空间谱矩阵
    theta_grid: 俯仰角网格
    phi_grid: 方位角网格
    title: 图像标题
    save_path: 保存路径（可选）
    show_plot: 是否显示图像
    """
    plt.figure(figsize=(12, 8))
    
    # 转换为dB并绘制
    spectrum_db = 10 * np.log10(spectrum + 1e-10)  # 添加小值避免log(0)
    
    # 创建网格
    phi_deg = np.degrees(phi_grid)
    theta_deg = np.degrees(theta_grid)
    
    # 绘制热力图
    im = plt.imshow(spectrum_db, aspect='auto', origin='lower',
                    extent=[phi_deg[0], phi_deg[-1], theta_deg[0], theta_deg[-1]],
                    cmap='viridis')
    
    plt.colorbar(im, label='功率 (dB)')
    plt.xlabel('方位角 φ (度)')
    plt.ylabel('俯仰角 θ (度)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # 找到峰值位置
    max_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    max_theta = theta_deg[max_idx[0]]
    max_phi = phi_deg[max_idx[1]]
    max_power = spectrum_db[max_idx[0], max_idx[1]]
    
    # 标记峰值
    plt.plot(max_phi, max_theta, 'r*', markersize=15, label=f'峰值: ({max_theta:.1f}°, {max_phi:.1f}°)')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    print(f"空间谱峰值位置: θ={max_theta:.1f}°, φ={max_phi:.1f}°, 功率={max_power:.1f} dB")


def find_peak_directions(
    spectrum: np.ndarray,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    num_peaks: int = 3,
    min_distance: int = 5
) -> List[Tuple[float, float, float]]:
    """
    在空间谱中找到多个峰值方向
    
    参数:
    spectrum: 空间谱矩阵
    theta_grid: 俯仰角网格
    phi_grid: 方位角网格
    num_peaks: 要找到的峰值数量
    min_distance: 峰值之间的最小距离（像素）
    
    返回:
    peaks: 峰值列表，每个元素为 (theta, phi, power)
    """
    from scipy.ndimage import maximum_filter
    from scipy.ndimage import generate_binary_structure, binary_erosion
    
    # 使用最大滤波器找到局部最大值
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(spectrum, footprint=neighborhood) == spectrum
    
    # 移除边界
    eroded = binary_erosion(local_max, structure=neighborhood, border_value=0)
    
    # 找到峰值坐标
    peak_coords = np.where(eroded)
    peak_values = spectrum[peak_coords]
    
    # 按功率排序
    sorted_indices = np.argsort(peak_values)[::-1]
    peak_coords = (peak_coords[0][sorted_indices], peak_coords[1][sorted_indices])
    peak_values = peak_values[sorted_indices]
    
    # 选择峰值，确保它们之间有足够距离
    selected_peaks = []
    for i in range(len(peak_coords[0])):
        if len(selected_peaks) >= num_peaks:
            break
            
        current_theta_idx = peak_coords[0][i]
        current_phi_idx = peak_coords[1][i]
        
        # 检查与已选峰值的距离
        too_close = False
        for selected_theta_idx, selected_phi_idx, _ in selected_peaks:
            distance = np.sqrt((current_theta_idx - selected_theta_idx)**2 + 
                             (current_phi_idx - selected_phi_idx)**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            theta = theta_grid[current_theta_idx]
            phi = phi_grid[current_phi_idx]
            power = peak_values[i]
            selected_peaks.append((theta, phi, power))
    
    return selected_peaks


# 便捷函数
def csi_to_spatial_spectrum(
    csi_data: Union[np.ndarray, List[np.ndarray]],
    antenna_config: str = "8x8",
    center_freq: float = 5e9,
    bandwidth: float = 20e6,
    subcarrier_indices: Optional[List[int]] = None,
    theta_range: Tuple[float, float, int] = (-np.pi/4, np.pi/4, 60),
    phi_range: Tuple[float, float, int] = (0, 2*np.pi, 120),
    algorithm: str = 'bartlett',
    fusion_method: str = 'average',
    plot_result: bool = True,
    **kwargs
) -> Tuple[np.ndarray, dict]:
    """
    从CSI数据计算空间谱的便捷函数
    
    参数:
    csi_data: CSI数据
    antenna_config: 天线配置，如 "8x8"
    center_freq: 中心频率 (Hz)
    bandwidth: 带宽 (Hz)
    subcarrier_indices: 子载波索引，默认为所有子载波
    theta_range: 俯仰角范围 (min, max, num_points)
    phi_range: 方位角范围 (min, max, num_points)
    algorithm: 算法选择
    fusion_method: 融合方法
    plot_result: 是否绘制结果
    **kwargs: 其他参数
    
    返回:
    fused_spectrum: 融合后的空间谱
    metadata: 计算元数据
    """
    # 如果没有指定子载波索引，使用默认值
    if subcarrier_indices is None:
        if isinstance(csi_data, np.ndarray):
            K = csi_data.shape[2]
        else:
            K = len(csi_data)
        subcarrier_indices = list(range(K))
    
    # 计算空间谱
    fused_spectrum, individual_spectrums, metadata = calculate_spatial_spectrum(
        csi_data=csi_data,
        antenna_config=antenna_config,
        center_freq=center_freq,
        bandwidth=bandwidth,
        subcarrier_indices=subcarrier_indices,
        theta_range=theta_range,
        phi_range=phi_range,
        algorithm=algorithm,
        fusion_method=fusion_method,
        **kwargs
    )
    
    # 绘制结果
    if plot_result:
        title = f"空间谱 - {antenna_config}阵列, {algorithm}算法, {fusion_method}融合"
        plot_spatial_spectrum(
            fused_spectrum, 
            metadata['theta_grid'], 
            metadata['phi_grid'], 
            title=title
        )
    
    return fused_spectrum, metadata
