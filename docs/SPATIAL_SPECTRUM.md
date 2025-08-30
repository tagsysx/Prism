# CSI空间谱生成技术文档

## 概述

本文档详细描述了如何通过信道状态信息（Channel State Information, CSI）生成空间谱的理论基础、数学模型和实现方法。空间谱估计是无线通信系统中的关键技术，用于确定信号的到达方向（Direction of Arrival, DoA），在波束形成、干扰抑制和定位服务中具有重要应用。

对于**M×N接收天线阵列**、**K个子载波**和维度为**A×B的空间角度谱**，本文档提供了完整的计算方法和代码实现，包括多子载波频谱融合技术。

## 1. 理论基础

### 1.1 核心思想

空间谱估计的核心是**空间扫描**技术。我们构建一个与来自特定方向 `(θ, φ)` 的信号完全匹配的**导向矢量（Steering Vector）** `a(θ, φ)`，然后计算接收信号 `X`（即CSI）与该导向矢量的相关性。相关性越高，说明该方向存在信号源的概率越大。

### 1.2 物理模型

对于远场窄带信号，其波达方向由以下角度参数定义：
- **俯仰角** `θ` (elevation)：信号与水平面的夹角，范围 [-π/2, π/2]
- **方位角** `φ` (azimuth)：信号在水平面的投影与参考方向的夹角，范围 [0, 2π]

## 2. 数学模型

### 2.1 导向矢量

**导向矢量** `a(θ, φ)` 是一个 `M×N × 1` 的复数向量，表示平面波在M×N天线阵列上的响应。其第 `k` 个元素的计算公式为：

```math
a_k(θ, φ) = \exp(-j \cdot 2π \cdot f \cdot τ_k(θ, φ))
```

其中 `τ_k(θ, φ)` 是波前到达第 `k` 个天线元素相对于阵列参考点（通常是阵列中心）的时间延迟。

### 2.2 均匀平面阵列（UPA）导向矢量

对于**均匀平面阵列**，假设阵元在x-y平面上，间距分别为 `d_x` 和 `d_y`，则第 `(m, n)` 个天线元素的导向矢量为：

```math
a_{m,n}(θ, φ) = \exp\left(-j \cdot 2π \cdot \left(\frac{m \cdot d_x \cdot \sin θ \cdot \cos φ}{λ} + \frac{n \cdot d_y \cdot \sin θ \cdot \sin φ}{λ}\right)\right)
```

其中：
- `λ` 是信号波长
- `m` 是天线在x方向的索引，范围 [0, M-1]
- `n` 是天线在y方向的索引，范围 [0, N-1]

将整个 `M × N` 阵列的响应按行优先顺序排列成一个列向量，就得到了完整的导向矢量 `a(θ, φ)`。

### 2.3 空间谱计算公式

#### Bartlett波束形成器

最经典和常用的方法是**Bartlett Beamformer**（也称为傅里叶波束形成或常规波束形成）。其空间谱 `P(θ, φ)` 的计算公式为：

```math
P(θ, φ) = \mathbf{a}^H(θ, φ) \mathbf{R}_{xx} \mathbf{a}(θ, φ)
```

其中：
- `a(θ, φ)` 是 `M×N × 1` 的导向矢量
- `a^H(θ, φ)` 是 `a(θ, φ)` 的共轭转置（Hermitian Transpose）
- `R_xx` 是接收信号的协方差矩阵，维度为 `M×N × M×N`

#### 协方差矩阵计算

协方差矩阵的计算公式为：

```math
\mathbf{R}_{xx} = \mathbb{E}[\mathbf{X} \mathbf{X}^H] \approx \frac{1}{T} \sum_{t=1}^{T} \mathbf{X}_t \mathbf{X}_t^H
```

其中：
- `X` 是 `M×N × T × K` 的CSI矩阵（包含K个子载波）
- `T` 是快拍数（样本数）
- `K` 是子载波数量
- 对于单个CSI快拍，`T=1`，则 `R_xx = X X^H`

**物理意义**：该公式本质上计算了接收信号在某个方向上的**能量**。如果 `(θ, φ)` 方向确实有信号源，那么 `a(θ, φ)` 会与信号子空间匹配，输出功率 `P(θ, φ)` 就会很大。

### 2.4 多子载波频谱融合

在OFDM系统中，我们有K个子载波，每个子载波具有不同的频率 `f_k`。对于第 `k` 个子载波，其导向矢量为：

```math
a_k(θ, φ) = \exp\left(-j \cdot 2π \cdot \left(\frac{m \cdot d_x \cdot \sin θ \cdot \cos φ}{λ_k} + \frac{n \cdot d_y \cdot \sin θ \cdot \sin φ}{λ_k}\right)\right)
```

其中 `λ_k = c/f_k` 是第 `k` 个子载波的波长。

#### 频谱融合方法

**方法1：简单平均融合**
```math
P_{\text{avg}}(θ, φ) = \frac{1}{K} \sum_{k=1}^{K} P_k(θ, φ)
```

**方法2：加权融合**
```math
P_{\text{weighted}}(θ, φ) = \sum_{k=1}^{K} w_k \cdot P_k(θ, φ)
```
其中 `w_k` 是第 `k` 个子载波的权重，通常基于信噪比或信号强度确定。

**方法3：最大值融合**
```math
P_{\text{max}}(θ, φ) = \max_{k=1,\ldots,K} P_k(θ, φ)
```

**方法4：相干融合**
对于相干融合，我们将所有子载波的CSI数据合并后再计算空间谱：
```math
\mathbf{X}_{\text{combined}} = [\mathbf{X}_1, \mathbf{X}_2, \ldots, \mathbf{X}_K]
```
然后使用合并后的数据计算协方差矩阵和空间谱。

## 3. 算法实现

### 3.1 计算步骤

给定：
- CSI矩阵 `X` (维度 `M×N × T × K`，包含K个子载波)
- 要估计的角度范围：`θ ∈ [θ_min, θ_max]`（A个点）, `φ ∈ [φ_min, φ_max]`（B个点）

#### 单子载波处理步骤

**步骤1：计算协方差矩阵**
对于第 `k` 个子载波：
```
R_xx_k = (X_k * X_k.H) / T
```
（`.H` 表示共轭转置）

**步骤2：初始化空间谱矩阵**
创建一个全零矩阵 `P_k`，维度为 `A × B`，用于存储每个角度对的功率。

**步骤3：角度网格扫描**
```
for i in range(A):  # 遍历俯仰角 θ_i
    for j in range(B):  # 遍历方位角 φ_j
        # 计算导向矢量（考虑子载波频率）
        a_ij_k = a_k(θ_i, φ_j)  # 维度 M×N × 1
        
        # 计算该方向的功率
        P_k[i, j] = real(a_ij_k.H @ R_xx_k @ a_ij_k)
```

#### 多子载波融合步骤

**步骤4：频谱融合**
根据选择的融合方法合并各子载波的空间谱：

```python
# 方法1：平均融合
P_fused = np.mean([P_1, P_2, ..., P_K], axis=0)

# 方法2：加权融合
P_fused = np.sum([w_k * P_k for k, P_k in enumerate(spectrums)], axis=0)

# 方法3：最大值融合
P_fused = np.maximum.reduce([P_1, P_2, ..., P_K])

# 方法4：相干融合
X_combined = np.concatenate([X_1, X_2, ..., X_K], axis=1)
R_xx_combined = (X_combined @ X_combined.conj().T) / (T * K)
# 然后使用R_xx_combined计算空间谱
```

**步骤5：可视化**
将融合后的 `P_fused` 矩阵以热力图的形式绘制出来，峰值位置即对应信号源的来波方向。

### 3.2 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt

def calculate_steering_vector(theta, phi, N, dx, dy, wavelength):
    """
    计算导向矢量
    
    参数:
    theta: 俯仰角 (弧度)
    phi: 方位角 (弧度)
    N: 天线阵列大小 (N×N)
    dx, dy: x和y方向的天线间距
    wavelength: 信号波长
    
    返回:
    a: 导向矢量，形状为 (N*N, 1)
    """
    a = np.zeros((N, N), dtype=complex)
    
    for m in range(N):  # x方向索引
        for n in range(N):  # y方向索引
            phase_shift = 2 * np.pi * (
                (m * dx * np.sin(theta) * np.cos(phi)) / wavelength +
                (n * dy * np.sin(theta) * np.sin(phi)) / wavelength
            )
            a[m, n] = np.exp(-1j * phase_shift)
    
    return a.reshape(-1, 1)  # 展平为 (N*N, 1) 向量

def calculate_spatial_spectrum(csi, N, theta_grid, phi_grid, freq, dx=None, dy=None):
    """
    计算空间谱
    
    参数:
    csi: CSI矩阵，形状为 (N*N, T)
    N: 天线阵列大小 (N×N)
    theta_grid: 俯仰角网格，1D数组，包含A个角度值
    phi_grid: 方位角网格，1D数组，包含B个角度值
    freq: 载波频率 (Hz)
    dx, dy: 天线间距，默认为半波长间距
    
    返回:
    P_spectrum: 空间谱矩阵，形状为 (A, B)
    """
    T = csi.shape[1]
    wavelength = 3e8 / freq
    
    # 默认半波长间距
    if dx is None:
        dx = 0.5 * wavelength
    if dy is None:
        dy = 0.5 * wavelength
    
    # 步骤1：估计协方差矩阵
    R_xx = (csi @ csi.conj().T) / T  # 形状 (N*N, N*N)
    
    # 步骤2：初始化空间谱
    P_spectrum = np.zeros((len(theta_grid), len(phi_grid)))
    
    # 步骤3：角度网格扫描
    for i, theta in enumerate(theta_grid):
        for j, phi in enumerate(phi_grid):
            # 构建导向矢量 a(theta, phi)
            a_vec = calculate_steering_vector(theta, phi, N, dx, dy, wavelength)
            
            # Bartlett波束形成
            power = np.real(a_vec.conj().T @ R_xx @ a_vec)  # 形状 (1, 1)
            P_spectrum[i, j] = power.squeeze()
            
    return P_spectrum

def plot_spatial_spectrum(spectrum, theta_grid, phi_grid, title="2D空间谱 (Bartlett波束形成器)"):
    """
    绘制空间谱
    
    参数:
    spectrum: 空间谱矩阵
    theta_grid: 俯仰角网格
    phi_grid: 方位角网格
    title: 图像标题
    """
    plt.figure(figsize=(12, 8))
    
    # 转换为dB并绘制
    spectrum_db = 10 * np.log10(spectrum + 1e-10)  # 添加小值避免log(0)
    
    plt.imshow(spectrum_db, aspect='auto', origin='lower',
               extent=[np.degrees(phi_grid[0]), np.degrees(phi_grid[-1]), 
                      np.degrees(theta_grid[0]), np.degrees(theta_grid[-1])],
               cmap='viridis')
    
    plt.colorbar(label='功率 (dB)')
    plt.xlabel('方位角 φ (度)')
    plt.ylabel('俯仰角 θ (度)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

# 使用示例
def example_usage():
    """
    空间谱计算示例
    """
    # 参数设置
    M, N = 8, 16  # 8×16天线阵列
    A, B = 90, 180  # 90个俯仰角，180个方位角
    freq = 5e9  # 5 GHz载波频率
    
    # 角度网格
    theta_range = np.linspace(-np.pi/3, np.pi/3, A)  # -60°到60°
    phi_range = np.linspace(0, 2*np.pi, B)  # 完整360°
    
    # 生成模拟CSI数据（单快拍）
    # 在实际应用中，这里应该是从无线通信系统获得的真实CSI数据
    np.random.seed(42)  # 为了结果可重现
    csi_data = np.random.randn(M*N, 1) + 1j * np.random.randn(M*N, 1)
    
    # 添加一个强信号源（模拟）
    signal_theta, signal_phi = np.pi/6, np.pi/4  # 30°俯仰角，45°方位角
    wavelength = 3e8 / freq
    dx = dy = 0.5 * wavelength
    
    # 生成该方向的导向矢量并添加强信号
    signal_steering = calculate_steering_vector(signal_theta, signal_phi, M, N, dx, dy, wavelength)
    signal_power = 10  # 信号功率
    csi_data += signal_power * signal_steering * (np.random.randn() + 1j * np.random.randn())
    
    # 计算空间谱
    spectrum = calculate_spatial_spectrum(
        csi=csi_data,
        M=M,
        N=N,
        theta_grid=theta_range,
        phi_grid=phi_range,
        freq=freq,
        dx=dx,
        dy=dy
    )
    
    # 绘制结果
    plot_spatial_spectrum(spectrum, theta_range, phi_range)
    
    # 找到峰值位置
    max_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    estimated_theta = theta_range[max_idx[0]]
    estimated_phi = phi_range[max_idx[1]]
    
    print(f"真实信号方向: θ={np.degrees(signal_theta):.1f}°, φ={np.degrees(signal_phi):.1f}°")
    print(f"估计信号方向: θ={np.degrees(estimated_theta):.1f}°, φ={np.degrees(estimated_phi):.1f}°")
    
    return spectrum

def calculate_subcarrier_spectrum(csi_k, M, N, theta_grid, phi_grid, freq_k, dx=None, dy=None):
    """
    计算单个子载波的空间谱
    
    参数:
    csi_k: 第k个子载波的CSI矩阵，形状为 (M*N, T)
    M: 天线阵列x方向大小
    N: 天线阵列y方向大小
    theta_grid: 俯仰角网格，1D数组，包含A个角度值
    phi_grid: 方位角网格，1D数组，包含B个角度值
    freq_k: 第k个子载波频率 (Hz)
    dx, dy: 天线间距，默认为半波长间距
    
    返回:
    P_spectrum: 空间谱矩阵，形状为 (A, B)
    """
    T = csi_k.shape[1]
    wavelength = 3e8 / freq_k
    
    # 默认半波长间距
    if dx is None:
        dx = 0.5 * wavelength
    if dy is None:
        dy = 0.5 * wavelength
    
    # 步骤1：估计协方差矩阵
    R_xx = (csi_k @ csi_k.conj().T) / T  # 形状 (M*N, M*N)
    
    # 步骤2：初始化空间谱
    P_spectrum = np.zeros((len(theta_grid), len(phi_grid)))
    
    # 步骤3：角度网格扫描
    for i, theta in enumerate(theta_grid):
        for j, phi in enumerate(phi_grid):
            # 构建导向矢量 a(theta, phi)，需要更新为M×N
            a = np.zeros((M, N), dtype=complex)
            for m in range(M):
                for n in range(N):
                    phase_shift = 2 * np.pi * (
                        (m * dx * np.sin(theta) * np.cos(phi)) / wavelength +
                        (n * dy * np.sin(theta) * np.sin(phi)) / wavelength
                    )
                    a[m, n] = np.exp(-1j * phase_shift)
            a_vec = a.reshape(-1, 1)
            
            # Bartlett波束形成
            power = np.real(a_vec.conj().T @ R_xx @ a_vec)  # 形状 (1, 1)
            P_spectrum[i, j] = power.squeeze()
            
    return P_spectrum

def fuse_subcarrier_spectrums(spectrums, method='average', weights=None):
    """
    融合多个子载波的空间谱
    
    参数:
    spectrums: 子载波空间谱列表，每个元素形状为 (A, B)
    method: 融合方法 ('average', 'weighted', 'max')
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
    else:
        raise ValueError(f"Unknown fusion method: {method}")

def calculate_multicarrier_spectrum(csi_data, M, N, theta_grid, phi_grid, frequencies, 
                                  dx=None, dy=None, fusion_method='average', weights=None):
    """
    计算多子载波融合空间谱
    
    参数:
    csi_data: CSI数据，形状为 (M*N, T, K) 或列表 [csi_1, csi_2, ..., csi_K]
    M: 天线阵列x方向大小
    N: 天线阵列y方向大小
    theta_grid: 俯仰角网格
    phi_grid: 方位角网格
    frequencies: 子载波频率数组，长度为K
    dx, dy: 天线间距
    fusion_method: 融合方法
    weights: 融合权重
    
    返回:
    fused_spectrum: 融合后的空间谱
    individual_spectrums: 各子载波的空间谱列表
    """
    K = len(frequencies)
    individual_spectrums = []
    
    # 处理CSI数据格式
    if isinstance(csi_data, list):
        csi_subcarriers = csi_data
    elif csi_data.ndim == 3:  # (M*N, T, K)
        csi_subcarriers = [csi_data[:, :, k] for k in range(K)]
    else:  # (M*N, T*K) - 需要重新整形
        T_total = csi_data.shape[1]
        T_per_subcarrier = T_total // K
        csi_subcarriers = []
        for k in range(K):
            start_idx = k * T_per_subcarrier
            end_idx = (k + 1) * T_per_subcarrier
            csi_subcarriers.append(csi_data[:, start_idx:end_idx])
    
    # 计算每个子载波的空间谱
    for k, freq_k in enumerate(frequencies):
        spectrum_k = calculate_subcarrier_spectrum(
            csi_subcarriers[k], M, N, theta_grid, phi_grid, freq_k, dx, dy
        )
        individual_spectrums.append(spectrum_k)
    
    # 融合空间谱
    fused_spectrum = fuse_subcarrier_spectrums(individual_spectrums, fusion_method, weights)
    
    return fused_spectrum, individual_spectrums

# 多子载波示例
def multicarrier_example():
    """
    多子载波空间谱计算示例
    """
    # 参数设置
    M, N = 8, 16  # 8×16天线阵列
    K = 64  # 64个子载波
    A, B = 60, 120  # 角度分辨率
    freq_center = 5e9  # 5 GHz中心频率
    bandwidth = 20e6  # 20 MHz带宽
    
    # 生成子载波频率
    frequencies = np.linspace(freq_center - bandwidth/2, freq_center + bandwidth/2, K)
    
    # 角度网格
    theta_range = np.linspace(-np.pi/4, np.pi/4, A)  # -45°到45°
    phi_range = np.linspace(0, 2*np.pi, B)  # 完整360°
    
    # 生成模拟多子载波CSI数据
    np.random.seed(42)
    csi_subcarriers = []
    
    # 真实信号方向
    true_theta, true_phi = np.pi/6, np.pi/4  # 30°俯仰角，45°方位角
    
    for k, freq_k in enumerate(frequencies):
        # 为每个子载波生成CSI
        csi_k = np.random.randn(M*N, 1) + 1j * np.random.randn(M*N, 1)
        
        # 添加来自特定方向的信号
        wavelength_k = 3e8 / freq_k
        dx = dy = 0.5 * wavelength_k
        
        # 计算该子载波的导向矢量
        a = np.zeros((M, N), dtype=complex)
        for m in range(M):
            for n in range(N):
                phase_shift = 2 * np.pi * (
                    (m * dx * np.sin(true_theta) * np.cos(true_phi)) / wavelength_k +
                    (n * dy * np.sin(true_theta) * np.sin(true_phi)) / wavelength_k
                )
                a[m, n] = np.exp(-1j * phase_shift)
        
        signal_steering = a.reshape(-1, 1)
        signal_power = 10  # 信号功率
        csi_k += signal_power * signal_steering * (np.random.randn() + 1j * np.random.randn())
        
        csi_subcarriers.append(csi_k)
    
    # 计算多子载波融合空间谱
    print("正在计算多子载波空间谱...")
    
    # 测试不同融合方法
    fusion_methods = ['average', 'max', 'weighted']
    
    for method in fusion_methods:
        print(f"  - {method}融合方法...")
        
        # 对于加权融合，使用基于频率的权重
        if method == 'weighted':
            # 中心频率权重更高
            weights = np.exp(-((frequencies - freq_center) / (bandwidth/4))**2)
            weights = weights / np.sum(weights)
        else:
            weights = None
        
        fused_spectrum, individual_spectrums = calculate_multicarrier_spectrum(
            csi_subcarriers, M, N, theta_range, phi_range, frequencies,
            fusion_method=method, weights=weights
        )
        
        # 找到峰值位置
        max_idx = np.unravel_index(np.argmax(fused_spectrum), fused_spectrum.shape)
        estimated_theta = theta_range[max_idx[0]]
        estimated_phi = phi_range[max_idx[1]]
        
        print(f"    估计方向: θ={np.degrees(estimated_theta):.1f}°, φ={np.degrees(estimated_phi):.1f}°")
        
        # 计算误差
        theta_error = np.degrees(abs(estimated_theta - true_theta))
        phi_error = np.degrees(abs(estimated_phi - true_phi))
        print(f"    误差: Δθ={theta_error:.1f}°, Δφ={phi_error:.1f}°")
    
    print(f"真实信号方向: θ={np.degrees(true_theta):.1f}°, φ={np.degrees(true_phi):.1f}°")
    
    return fused_spectrum, individual_spectrums

if __name__ == "__main__":
    # 运行单子载波示例
    spectrum = example_usage()
    
    # 运行多子载波示例
    print("\n" + "="*60)
    print("多子载波空间谱融合示例")
    print("="*60)
    multicarrier_example()
```

## 4. 高级方法

### 4.1 CAPON波束形成器

CAPON波束形成器（也称为最小方差无失真响应，MVDR）提供更高的分辨率：

```math
P_{\text{CAPON}}(θ, φ) = \frac{1}{\mathbf{a}^H(θ, φ) \mathbf{R}_{xx}^{-1} \mathbf{a}(θ, φ)}
```

**优点**：
- 更高的角度分辨率
- 更好的旁瓣抑制

**缺点**：
- 需要矩阵求逆，计算复杂度高
- 对噪声和模型误差敏感

### 4.2 MUSIC算法

MUSIC（Multiple Signal Classification）基于子空间分解，分辨率极高：

```math
P_{\text{MUSIC}}(θ, φ) = \frac{1}{\mathbf{a}^H(θ, φ) \mathbf{E}_n \mathbf{E}_n^H \mathbf{a}(θ, φ)}
```

其中 `E_n` 是噪声子空间的特征向量矩阵。

**优点**：
- 超分辨率性能
- 理论上可以分辨任意接近的信号源

**缺点**：
- 需要准确估计信源数量
- 计算复杂度很高
- 对模型误差非常敏感

## 5. 实际应用考虑

### 5.1 天线阵列校准

在实际应用中，天线阵列的响应可能与理论模型存在偏差，需要进行校准：

1. **相位校准**：补偿天线间的相位误差
2. **幅度校准**：补偿天线间的增益差异
3. **位置校准**：补偿天线位置的偏差

### 5.2 多径效应

在实际无线环境中，信号会经历多径传播，导致：
- 多个峰值出现在空间谱中
- 需要多径分离算法
- 可能需要时域处理结合

### 5.3 计算优化

对于大型天线阵列和高分辨率角度网格：
- 使用FFT加速计算
- 并行化处理
- 稀疏处理技术

## 6. 与Prism项目的集成

### 6.1 CSI数据获取

在Prism项目中，CSI数据可以通过以下方式获得：
- 从射线追踪仿真结果中提取
- 使用训练好的神经网络模型预测
- 从实际硬件测量中获得

### 6.2 应用场景

空间谱估计在Prism项目中的应用包括：
- **波束形成优化**：确定最佳发射方向
- **干扰源定位**：识别和抑制干扰信号
- **信道建模验证**：验证射线追踪模型的准确性
- **自适应天线系统**：动态调整天线方向图

### 6.3 性能评估

可以使用以下指标评估空间谱估计性能：
- **角度估计精度**：估计角度与真实角度的误差
- **分辨率**：能够分辨的最小角度间隔
- **检测概率**：正确检测信号源的概率
- **虚警概率**：错误检测信号源的概率

## 7. 总结

本文档详细介绍了基于CSI的空间谱生成方法，包括：

1. **理论基础**：导向矢量、协方差矩阵和空间扫描原理
2. **数学模型**：Bartlett、CAPON和MUSIC等不同算法
3. **实现方法**：完整的Python代码示例
4. **实际考虑**：校准、多径效应和计算优化
5. **项目集成**：与Prism项目的结合应用

对于大部分应用场景，**Bartlett方法**因其简单性和稳定性而被广泛采用。当需要更高分辨率时，可以考虑CAPON或MUSIC算法，但需要权衡计算复杂度和鲁棒性。

在实际部署时，建议根据具体的应用需求、计算资源和精度要求选择合适的算法，并进行充分的仿真验证和实测验证。
