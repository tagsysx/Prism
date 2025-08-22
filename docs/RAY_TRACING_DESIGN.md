# Ray Tracing Design Document

## Overview

This document outlines the design and implementation of the discrete electromagnetic ray tracing system for the Prism project. The system implements an efficient voxel-based ray tracing approach that combines discrete radiance field modeling with advanced optimization strategies to achieve both accuracy and computational efficiency.

## 1. System Architecture

### 1.1 Core Components

The ray tracing system consists of three main components:

1. **Voxel Scene Representation**: Discretized 3D scene into voxel grid
2. **Neural Network Models**: Attenuation and radiation property networks (see [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md))
3. **Ray Tracing Engine**: Discrete ray marching with optimization strategies

### 1.2 Data Flow

```
Scene Geometry → Voxelization → Neural Networks → Ray Tracing → Signal Output
     ↓              ↓              ↓              ↓           ↓
  3D Models    Voxel Grid    Attenuation/     Optimized    Received
              + Features     Radiation        Sampling     Signals
```

## 2. Voxel Scene Representation

### 2.1 Voxel Structure

Each voxel contains:

```python
class Voxel:
    position: Vector3D          # Center position (x, y, z)
    size: float                 # Voxel edge length
    attenuation_coeff: Complex  # ρ = ΔA + jΔφ
    features: Tensor[256]       # Latent features F(P_v)
    material_id: int            # Material type identifier
    occupancy: float            # Occupancy probability
```

### 2.2 Scene Voxelization

```python
def voxelize_scene(scene_geometry, voxel_size):
    """
    Convert 3D scene into uniform voxel grid
    
    Args:
        scene_geometry: 3D mesh or point cloud
        voxel_size: Size of each voxel cube
    
    Returns:
        voxel_grid: 3D array of voxels
        bounding_box: Scene boundaries
    """
    # Calculate grid dimensions
    min_coords, max_coords = get_scene_bounds(scene_geometry)
    grid_dims = calculate_grid_dimensions(min_coords, max_coords, voxel_size)
    
    # Initialize voxel grid
    voxel_grid = initialize_voxel_grid(grid_dims)
    
    # Assign geometry to voxels
    for geometry in scene_geometry:
        affected_voxels = find_affected_voxels(geometry, voxel_grid)
        for voxel in affected_voxels:
            update_voxel_properties(voxel, geometry)
    
    return voxel_grid, (min_coords, max_coords)
```

## 3. Neural Network Models

The neural network architecture for the Prism project is fully specified in the [Architecture Design Document](ARCHITECTURE_DESIGN.md). This document covers:

- **AttenuationNetwork**: Spatial position encoding with 128D feature output
- **Attenuation Decoder**: Conversion to N_UE × K attenuation factors  
- **RadianceNetwork**: Radiation pattern modeling with antenna-specific embeddings
- **Antenna Embedding Codebook**: Learnable antenna representations

For complete implementation details, network architectures, and parameter configurations, please refer to [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md).

The ray tracing system described in this document integrates with these pre-defined neural networks to perform electromagnetic wave propagation modeling.

### 3.3 Positional Encoding

```python
class PositionalEncoding:
    def __init__(self, num_frequencies=10):
        self.num_frequencies = num_frequencies
    
    def encode(self, x):
        """Standard positional encoding"""
        encodings = []
        for i in range(self.num_frequencies):
            encodings.extend([
                torch.sin(2**i * torch.pi * x),
                torch.cos(2**i * torch.pi * x)
            ])
        return torch.cat(encodings, dim=-1)
    
    def integrated_encode(self, mean, covariance):
        """Integrated positional encoding for spatial regions"""
        encodings = []
        for i in range(self.num_frequencies):
            freq = 2**i * torch.pi
            # Exponential attenuation based on variance
            attenuation = torch.exp(-0.5 * (freq**2) * covariance)
            
            encodings.extend([
                torch.sin(freq * mean) * attenuation,
                torch.cos(freq * mean) * attenuation
            ])
        return torch.cat(encodings, dim=-1)
```

## 4. Discrete Ray Tracing Engine

The ray tracing engine implements the core electromagnetic wave propagation modeling based on the discrete radiance field approach. It integrates with the neural networks specified in [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) to compute attenuation factors and radiation patterns.

### 4.1 Ray Tracer Core Design

The ray tracer operates on the principle of discrete ray marching along direction $\omega$ from base station position $P_{\text{BS}}$, accumulating energy contributions from voxels along the ray path.

#### 4.1.1 Single-Ray Tracing Function

**Purpose**: Trace a single ray in direction $\omega$ and compute received energy from that direction.

**Input Parameters**:
- $P_{\text{BS}}$: Base station location (3D coordinates)
- $\text{antenna\_index}$: Index of the base station antenna (0 to $N_{\text{BS}}-1$)
- $\text{subcarrier\_index}$: Index of the subcarrier (0 to $K-1$)
- $\omega$: Ray tracing direction vector (normalized 3D vector)
- $D$: Scene depth (maximum tracing distance)

**Output**: 
- Received energy from direction $\omega$ for the specified antenna and subcarrier

**Algorithm**:
```python
def trace_single_ray(self, bs_position, antenna_index, subcarrier_index, direction, scene_depth):
    """
    Trace a single ray and compute received energy
    
    Args:
        bs_position: P_BS - Base station 3D position
        antenna_index: Base station antenna index (0 to N_BS-1)
        subcarrier_index: Subcarrier index (0 to K-1)
        direction: ω - Ray direction vector (normalized)
        scene_depth: D - Maximum tracing distance
    
    Returns:
        received_energy: Complex energy value received from this direction
    """
    # Stage 1: Uniform sampling for importance estimation
    uniform_samples = self._generate_uniform_samples(bs_position, direction, scene_depth)
    importance_weights = self._compute_importance_weights(uniform_samples, antenna_index, subcarrier_index)
    
    # Stage 2: Importance-based resampling using CDF
    fine_samples = self._resample_with_importance(importance_weights, uniform_samples)
    
    # Stage 3: Energy accumulation along the ray
    total_energy = self._accumulate_energy(fine_samples, antenna_index, subcarrier_index, direction)
    
    return total_energy
```

#### 4.1.2 Importance-Based Sampling Implementation

**Stage 1: Uniform Sampling and Weight Computation**

```python
def _generate_uniform_samples(self, bs_position, direction, scene_depth):
    """Generate uniform samples along ray for importance estimation"""
    num_coarse_samples = self.config.get('coarse_samples', 100)
    step_size = scene_depth / num_coarse_samples
    
    samples = []
    for i in range(num_coarse_samples):
        # Sample position: P_BS + ω * t
        t = i * step_size
        sample_pos = bs_position + t * direction
        samples.append({
            'position': sample_pos,
            'distance': t,
            'step_size': step_size
        })
    
    return samples

def _compute_importance_weights(self, samples, antenna_index, subcarrier_index):
    """Compute importance weights based on attenuation factors"""
    weights = []
    
    for sample in samples:
        # Query AttenuationNetwork for attenuation factor
        attenuation_factor = self._query_attenuation_network(
            sample['position'], antenna_index, subcarrier_index
        )
        
        # Compute importance weight based on attenuation
        # Higher attenuation = higher importance for sampling
        weight = self._compute_sample_weight(attenuation_factor, sample['step_size'])
        weights.append(weight)
    
    # Normalize weights to form probability distribution
    weights = torch.tensor(weights)
    weights = weights / torch.sum(weights)
    
    return weights
```

**Stage 2: CDF-Based Resampling**

```python
def _resample_with_importance(self, importance_weights, uniform_samples):
    """Resample ray using importance weights and CDF"""
    num_fine_samples = self.config.get('fine_samples', 200)
    
    # Compute cumulative distribution function
    cdf = torch.cumsum(importance_weights, dim=0)
    
    fine_samples = []
    for i in range(num_fine_samples):
        # Generate random sample u ~ U[0,1]
        u = torch.rand(1)
        
        # Find corresponding position using inverse CDF
        sample_idx = torch.searchsorted(cdf, u)
        sample_idx = torch.clamp(sample_idx, 0, len(uniform_samples) - 1)
        
        # Interpolate position between uniform samples
        if sample_idx == 0:
            sample_pos = uniform_samples[0]['position']
        else:
            # Linear interpolation between samples
            alpha = (u - cdf[sample_idx-1]) / (cdf[sample_idx] - cdf[sample_idx-1])
            pos_prev = uniform_samples[sample_idx-1]['position']
            pos_curr = uniform_samples[sample_idx]['position']
            sample_pos = pos_prev + alpha * (pos_curr - pos_prev)
        
        fine_samples.append({
            'position': sample_pos,
            'importance_weight': importance_weights[sample_idx]
        })
    
    return fine_samples
```

**Stage 3: Energy Accumulation**

```python
def _accumulate_energy(self, fine_samples, antenna_index, subcarrier_index, direction):
    """Accumulate energy contributions along the ray"""
    accumulated_energy = 0.0
    cumulative_attenuation = 0.0
    
    for i, sample in enumerate(fine_samples):
        # Query neural networks for this sample point
        attenuation_factor = self._query_attenuation_network(
            sample['position'], antenna_index, subcarrier_index
        )
        
        radiation_factor = self._query_radiation_network(
            sample['position'], direction, antenna_index, subcarrier_index
        )
        
        # Apply cumulative attenuation from previous samples
        if i > 0:
            cumulative_attenuation += attenuation_factor * sample['step_size']
        
        # Compute local energy contribution
        local_energy = attenuation_factor * radiation_factor
        
        # Apply attenuation and accumulate
        attenuated_energy = local_energy * torch.exp(-cumulative_attenuation)
        accumulated_energy += attenuated_energy * sample['step_size']
    
    return accumulated_energy
```

#### 4.1.3 Neural Network Integration

The ray tracer integrates with the neural networks specified in [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md):

```python
def _query_attenuation_network(self, position, antenna_index, subcarrier_index):
    """Query AttenuationNetwork for attenuation factors"""
    # Apply IPE encoding to position
    ipe_features = self.positional_encoding.integrated_encode(position)
    
    # Query AttenuationNetwork (outputs 128D features)
    features = self.attenuation_network(ipe_features)
    
    # Query Attenuation Decoder for specific antenna and subcarrier
    # Output: N_UE × K attenuation factors
    attenuation_factors = self.attenuation_decoder(features)
    
    # Extract specific antenna and subcarrier
    return attenuation_factors[antenna_index, subcarrier_index]

def _query_radiation_network(self, position, direction, antenna_index, subcarrier_index):
    """Query RadianceNetwork for radiation patterns"""
    # Apply IPE encoding to position and direction
    pos_ipe = self.positional_encoding.integrated_encode(position)
    dir_pe = self.positional_encoding.encode(direction)
    
    # Get antenna embedding from codebook
    antenna_embedding = self.antenna_codebook[antenna_index]
    
    # Query RadianceNetwork
    # Input: [IPE_pos, PE_direction, 128D_features, 64D_antenna_embedding]
    radiation_output = self.radiance_network(
        pos_ipe, dir_pe, self.current_features, antenna_embedding
    )
    
    # Output: N_UE × K radiation factors
    return radiation_output[antenna_index, subcarrier_index]
```

### 4.2 Multiple-Ray Tracing

**Purpose**: Trace multiple rays in parallel for different antennas, subcarriers, and directions.

**Input Parameters**:
- $P_{\text{BS}}$: Base station location
- $\text{antenna\_indices}$: Array of antenna indices to trace
- $\text{subcarrier\_indices}$: Array of subcarrier indices to trace  
- $\text{directions}$: Array of ray directions
- $D$: Scene depth

**Output**: 
- Matrix of received energies: $[\text{num\_antennas}] \times [\text{num\_subcarriers}] \times [\text{num\_directions}]$

**Implementation**:

```python
def trace_multiple_rays(self, bs_position, antenna_indices, subcarrier_indices, directions, scene_depth):
    """
    Trace multiple rays in parallel
    
    Args:
        bs_position: Base station 3D position
        antenna_indices: Array of antenna indices [num_antennas]
        subcarrier_indices: Array of subcarrier indices [num_subcarriers]
        directions: Array of ray directions [num_directions, 3]
        scene_depth: Maximum tracing distance
    
    Returns:
        energy_matrix: [num_antennas, num_subcarriers, num_directions] complex energy values
    """
    num_antennas = len(antenna_indices)
    num_subcarriers = len(subcarrier_indices)
    num_directions = len(directions)
    
    # Initialize output matrix
    energy_matrix = torch.zeros(
        (num_antennas, num_subcarriers, num_directions), 
        dtype=torch.complex64
    )
    
    # Process in batches for efficiency
    batch_size = self.config.get('batch_size', 100)
    
    for ant_idx in range(0, num_antennas, batch_size):
        ant_end = min(ant_idx + batch_size, num_antennas)
        
        for sub_idx in range(0, num_subcarriers, batch_size):
            sub_end = min(sub_idx + batch_size, num_subcarriers)
            
            for dir_idx in range(0, num_directions, batch_size):
                dir_end = min(dir_idx + batch_size, num_directions)
                
                # Process batch
                batch_energies = self._trace_ray_batch(
                    bs_position,
                    antenna_indices[ant_idx:ant_end],
                    subcarrier_indices[sub_idx:sub_end],
                    directions[dir_idx:dir_end],
                    scene_depth
                )
                
                # Store results
                energy_matrix[ant_idx:ant_end, sub_idx:sub_end, dir_idx:dir_end] = batch_energies
    
    return energy_matrix

def _trace_ray_batch(self, bs_position, antenna_batch, subcarrier_batch, direction_batch, scene_depth):
    """Process a batch of rays in parallel"""
    batch_energies = []
    
    # Vectorized processing for the batch
    for antenna_idx in antenna_batch:
        for subcarrier_idx in subcarrier_batch:
            for direction in direction_batch:
                energy = self.trace_single_ray(
                    bs_position, antenna_idx, subcarrier_idx, direction, scene_depth
                )
                batch_energies.append(energy)
    
    return torch.stack(batch_energies).reshape(len(antenna_batch), len(subcarrier_batch), len(direction_batch))
```

### 4.3 Configuration Parameters

```python
RAY_TRACER_CONFIG = {
    # Sampling parameters
    'coarse_samples': 100,      # Number of uniform samples for importance estimation
    'fine_samples': 200,        # Number of importance-based samples for energy computation
    'batch_size': 100,          # Batch size for multiple-ray tracing
    
    # Neural network integration
    'use_attenuation_cache': True,    # Cache attenuation network outputs
    'use_radiation_cache': True,      # Cache radiation network outputs
    'cache_size': 10000,             # Maximum cache size
    
    # Performance optimization
    'enable_parallel_processing': True,  # Enable parallel ray tracing
    'num_worker_threads': 4,            # Number of worker threads
    'gpu_acceleration': True            # Enable GPU acceleration if available
}
```

## 5. Signal Aggregation and Training

### 5.1 Multi-Directional Signal Aggregation

The signal aggregation process computes the total received energy at the base station by tracing rays in multiple directions and aggregating the results across all antennas and subcarriers.

```python
def aggregate_signals(self, bs_position, antenna_indices, subcarrier_indices, scene_depth):
    """
    Aggregate signals from all directions for multiple antennas and subcarriers
    
    Args:
        bs_position: Base station 3D position
        antenna_indices: Array of antenna indices to process
        subcarrier_indices: Array of subcarrier indices to process
        scene_depth: Maximum tracing distance
    
    Returns:
        total_signals: [num_antennas, num_subcarriers] aggregated signal matrix
    """
    # Generate directional samples using pyramid structure
    if self.pyramid_sampling_enabled:
        directions = self.pyramid_sampler.generate_directional_samples(bs_position)
    else:
        directions = self._generate_uniform_directions()
    
    # Trace multiple rays in parallel
    energy_matrix = self.trace_multiple_rays(
        bs_position, antenna_indices, subcarrier_indices, directions, scene_depth
    )
    
    # Aggregate across directions
    # Shape: [num_antennas, num_subcarriers, num_directions] -> [num_antennas, num_subcarriers]
    total_signals = torch.sum(energy_matrix, dim=2) / len(directions)
    
    return total_signals

def _generate_uniform_directions(self):
    """Generate uniform directional samples using Fibonacci sphere sampling"""
    num_directions = self.config.get('num_directions', 100)
    directions = []
    
    phi = torch.pi * (3 - torch.sqrt(5))  # Golden angle
    
    for i in range(num_directions):
        y = 1 - (i / (num_directions - 1)) * 2
        radius = torch.sqrt(1 - y * y)
        
        theta = phi * i
        
        x = torch.cos(theta) * radius
        z = torch.sin(theta) * radius
        
        directions.append(torch.tensor([x, y, z]))
    
    return directions
```

### 5.2 Training Loss Function

The training loss compares predicted aggregated signals with ground truth measurements across all antennas and subcarriers:

```python
class RayTracingLoss(nn.Module):
    def __init__(self, loss_type='l2'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'l2':
            self.criterion = nn.MSELoss()
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, predicted_signals, target_signals):
        """
        Calculate training loss for multiple antennas and subcarriers
        
        Args:
            predicted_signals: Model predictions [batch_size, num_antennas, num_subcarriers]
            target_signals: Ground truth measurements [batch_size, num_antennas, num_subcarriers]
        
        Returns:
            loss: Training loss value
        """
        if self.loss_type == 'l2':
            return self.criterion(predicted_signals, target_signals)
        elif self.loss_type == 'l1':
            return self.criterion(predicted_signals, target_signals)
        
        # Complex signal handling for RF modeling
        if torch.is_complex(predicted_signals) or torch.is_complex(target_signals):
            # Separate real and imaginary parts
            pred_real = torch.real(predicted_signals)
            pred_imag = torch.imag(predicted_signals)
            target_real = torch.real(target_signals)
            target_imag = torch.imag(target_signals)
            
            real_loss = self.criterion(pred_real, target_real)
            imag_loss = self.criterion(pred_imag, target_imag)
            
            return real_loss + imag_loss
```

### 5.3 Training Data Structure

The training data should be organized to support the multi-antenna, multi-subcarrier architecture:

```python
class RayTracingDataset:
    def __init__(self, data_path, num_antennas, num_subcarriers):
        self.data_path = data_path
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        
    def __getitem__(self, index):
        """
        Get training sample
        
        Returns:
            sample: Dictionary containing:
                - bs_position: Base station position [3]
                - antenna_indices: Antenna indices to process
                - subcarrier_indices: Subcarrier indices to process
                - target_signals: Ground truth signals [num_antennas, num_subcarriers]
                - scene_depth: Maximum tracing distance
        """
        # Load training data
        data = torch.load(f"{self.data_path}/sample_{index}.pt")
        
        return {
            'bs_position': data['bs_position'],
            'antenna_indices': data['antenna_indices'],
            'subcarrier_indices': data['subcarrier_indices'],
            'target_signals': data['target_signals'],
            'scene_depth': data['scene_depth']
        }
```

## 6. Performance Optimization

### 6.1 Memory Management

```python
class VoxelCache:
    def __init__(self, max_cache_size=10000):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_count = {}
    
    def get_voxel(self, position):
        """Get voxel from cache or compute if not cached"""
        pos_key = tuple(position.tolist())
        
        if pos_key in self.cache:
            self.access_count[pos_key] += 1
            return self.cache[pos_key]
        
        # Compute voxel properties
        voxel = self._compute_voxel_properties(position)
        
        # Add to cache
        self._add_to_cache(pos_key, voxel)
        
        return voxel
    
    def _add_to_cache(self, key, voxel):
        """Add voxel to cache with LRU eviction"""
        if len(self.cache) >= self.max_cache_size:
            # Remove least recently used
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = voxel
        self.access_count[key] = 1
```

### 6.2 Batch Processing

```python
def batch_ray_tracing(self, receiver_positions, directions, transmitter_positions):
    """Process multiple rays in batches for efficiency"""
    batch_size = receiver_positions.shape[0]
    num_directions = directions.shape[1]
    
    # Reshape for batch processing
    receiver_pos_batch = receiver_positions.unsqueeze(1).expand(-1, num_directions, -1)
    tx_pos_batch = transmitter_positions.unsqueeze(1).expand(-1, num_directions, -1)
    
    # Flatten for processing
    receiver_pos_flat = receiver_pos_batch.reshape(-1, 3)
    directions_flat = directions.reshape(-1, 3)
    tx_pos_flat = tx_pos_batch.reshape(-1, 3)
    
    # Process in batches
    batch_size_flat = 1000  # Adjust based on memory constraints
    signals = []
    
    for i in range(0, len(receiver_pos_flat), batch_size_flat):
        batch_end = min(i + batch_size_flat, len(receiver_pos_flat))
        
        batch_signals = self._process_ray_batch(
            receiver_pos_flat[i:batch_end],
            directions_flat[i:batch_end],
            tx_pos_flat[i:batch_end]
        )
        signals.append(batch_signals)
    
    # Reshape back to original dimensions
    signals = torch.cat(signals, dim=0)
    return signals.reshape(batch_size, num_directions)
```

### 6.3 GPU Acceleration

```python
class GPURayTracer:
    def __init__(self, device='cuda'):
        self.device = device
        self.voxel_grid = None
        self.attenuation_net = None
        self.radiation_net = None
    
    def to_device(self, voxel_grid, attenuation_net, radiation_net):
        """Move models and data to GPU"""
        self.voxel_grid = voxel_grid.to(self.device)
        self.attenuation_net = attenuation_net.to(self.device)
        self.radiation_net = radiation_net.to(self.device)
    
    def trace_ray_gpu(self, receiver_pos, direction, transmitter_pos):
        """GPU-accelerated ray tracing"""
        # Ensure inputs are on GPU
        receiver_pos = receiver_pos.to(self.device)
        direction = direction.to(self.device)
        transmitter_pos = transmitter_pos.to(self.device)
        
        # Use CUDA kernels for voxel access and ray marching
        return self._cuda_ray_trace(receiver_pos, direction, transmitter_pos)
```

## 7. Configuration and Parameters

### 7.1 System Configuration

```python
RAY_TRACING_CONFIG = {
    # Sampling parameters
    'num_ray_steps': 100,
    'step_size': 0.1,
    'coarse_samples': 50,
    'fine_samples': 200,
    
    # Directional sampling
    'azimuth_divisions': 36,
    'elevation_divisions': 18,
    'num_directions': 100,
    
    # Optimization flags
    'importance_sampling': True,
    'pyramid_sampling': True,
    'gpu_acceleration': True,
    
    # Memory management
    'max_cache_size': 10000,
    'batch_size': 1000,
    
    # Training parameters
    'loss_type': 'l2',
    'learning_rate': 1e-4,
    'num_epochs': 1000
}
```

### 7.2 Performance Tuning

```python
def tune_performance(config, scene_complexity):
    """Automatically tune parameters based on scene complexity"""
    
    # Adjust step size based on scene scale
    if scene_complexity == 'low':
        config['step_size'] = 0.2
        config['num_ray_steps'] = 50
    elif scene_complexity == 'high':
        config['step_size'] = 0.05
        config['num_ray_steps'] = 200
    
    # Adjust directional sampling based on accuracy requirements
    if config.get('high_accuracy', False):
        config['azimuth_divisions'] = 72
        config['elevation_divisions'] = 36
        config['num_directions'] = 200
    
    # Memory optimization for large scenes
    if scene_complexity == 'high':
        config['max_cache_size'] = 50000
        config['batch_size'] = 500
    
    return config
```

## 8. Integration and Usage

### 8.1 Main Interface

```python
class PrismRayTracer:
    def __init__(self, config=None):
        self.config = config or RAY_TRACING_CONFIG
        self.voxel_grid = None
        
        # Neural networks (specified in ARCHITECTURE_DESIGN.md)
        self.attenuation_network = None
        self.attenuation_decoder = None
        self.radiance_network = None
        self.antenna_codebook = None
        
        # Ray tracing components
        self.ray_tracer = None
        self.pyramid_sampler = None
        self.positional_encoding = None
        
        self._initialize_networks()
        self._initialize_ray_tracer()
    
    def load_scene(self, scene_file):
        """Load and voxelize 3D scene"""
        scene_geometry = load_scene_geometry(scene_file)
        self.voxel_grid, bounding_box = voxelize_scene(
            scene_geometry, self.config['step_size']
        )
        return bounding_box
    
    def train(self, training_data):
        """Train the neural networks"""
        # Training implementation using the loss function from section 5.2
        pass
    
    def predict_signals(self, bs_position, antenna_indices, subcarrier_indices, scene_depth):
        """
        Predict received signals for given base station position
        
        Args:
            bs_position: Base station 3D position
            antenna_indices: Array of antenna indices to process
            subcarrier_indices: Array of subcarrier indices to process
            scene_depth: Maximum tracing distance
        
        Returns:
            predicted_signals: [num_antennas, num_subcarriers] signal matrix
        """
        return self.ray_tracer.aggregate_signals(
            bs_position, antenna_indices, subcarrier_indices, scene_depth
        )
    
    def trace_single_ray(self, bs_position, antenna_index, subcarrier_index, direction, scene_depth):
        """
        Trace a single ray for debugging and analysis
        
        Args:
            bs_position: Base station 3D position
            antenna_index: Specific antenna index
            subcarrier_index: Specific subcarrier index
            direction: Ray direction vector
            scene_depth: Maximum tracing distance
        
        Returns:
            received_energy: Energy received from this specific direction
        """
        return self.ray_tracer.trace_single_ray(
            bs_position, antenna_index, subcarrier_index, direction, scene_depth
        )
    
    def visualize_ray_path(self, bs_position, direction, scene_depth, antenna_index=0, subcarrier_index=0):
        """Visualize ray path for debugging"""
        # Visualization implementation showing:
        # - Ray path from base station
        # - Sample points along the ray
        # - Importance weights at each sample
        # - Energy accumulation
        pass
```

### 8.2 Example Usage

```python
# Initialize ray tracer
config = RAY_TRACING_CONFIG.copy()
config['high_accuracy'] = True

ray_tracer = PrismRayTracer(config)

# Load scene
bounding_box = ray_tracer.load_scene('office_building.obj')

# Train model (if needed)
if training_data_available:
    ray_tracer.train(training_data)

# Example 1: Single ray tracing for debugging
bs_position = torch.tensor([0, 0, 0])  # Base station at origin
antenna_index = 0                      # First antenna
subcarrier_index = 0                   # First subcarrier
direction = torch.tensor([1, 0, 0])   # Ray in +x direction
scene_depth = 100.0                    # Maximum tracing distance

single_ray_energy = ray_tracer.trace_single_ray(
    bs_position, antenna_index, subcarrier_index, direction, scene_depth
)
print(f"Single ray energy: {single_ray_energy}")

# Example 2: Multiple ray tracing for signal prediction
bs_position = torch.tensor([5, 5, 5])           # Base station position
antenna_indices = [0, 1, 2, 3]                 # 4 antennas
subcarrier_indices = [0, 1, 2, 3, 4]           # 5 subcarriers
scene_depth = 150.0                             # Maximum tracing distance

predicted_signals = ray_tracer.predict_signals(
    bs_position, antenna_indices, subcarrier_indices, scene_depth
)
print(f"Predicted signals shape: {predicted_signals.shape}")
print(f"Predicted signals: {predicted_signals}")

# Example 3: Batch processing for multiple base station positions
bs_positions = torch.tensor([
    [0, 0, 0],    # Base station 1
    [10, 0, 0],   # Base station 2
    [0, 10, 0],   # Base station 3
    [10, 10, 0]   # Base station 4
])

all_signals = []
for bs_pos in bs_positions:
    signals = ray_tracer.predict_signals(
        bs_pos, antenna_indices, subcarrier_indices, scene_depth
    )
    all_signals.append(signals)

all_signals = torch.stack(all_signals)
print(f"All signals shape: {all_signals.shape}")  # [4, 4, 5] = [num_bs, num_antennas, num_subcarriers]

# Example 4: Visualization for debugging
ray_tracer.visualize_ray_path(
    bs_position=torch.tensor([0, 0, 0]),
    direction=torch.tensor([1, 1, 0]),
    scene_depth=100.0,
    antenna_index=0,
    subcarrier_index=0
)
```

## 9. Testing and Validation

### 9.1 Unit Tests

```python
def test_voxel_properties():
    """Test voxel property computation"""
    voxel = Voxel(position=torch.tensor([0, 0, 0]), size=1.0)
    assert voxel.position.shape == (3,)
    assert voxel.size == 1.0

def test_ray_tracing_basic():
    """Test basic ray tracing functionality"""
    # Test implementation
    pass

def test_importance_sampling():
    """Test importance sampling optimization"""
    # Test implementation
    pass
```

### 9.2 Performance Benchmarks

```python
def benchmark_ray_tracing():
    """Benchmark ray tracing performance"""
    import time
    
    # Setup test scene
    ray_tracer = PrismRayTracer()
    ray_tracer.load_scene('test_scene.obj')
    
    # Benchmark uniform sampling
    start_time = time.time()
    uniform_signals = ray_tracer.predict_signals_uniform(receiver_pos, tx_pos)
    uniform_time = time.time() - start_time
    
    # Benchmark importance sampling
    start_time = time.time()
    importance_signals = ray_tracer.predict_signals_importance(receiver_pos, tx_pos)
    importance_time = time.time() - start_time
    
    print(f"Uniform sampling: {uniform_time:.3f}s")
    print(f"Importance sampling: {importance_time:.3f}s")
    print(f"Speedup: {uniform_time/importance_time:.2f}x")
```

## 10. Future Enhancements

### 10.1 Planned Improvements

1. **Adaptive Voxelization**: Dynamic voxel size based on scene complexity
2. **Multi-scale Ray Tracing**: Hierarchical ray tracing for different detail levels
3. **Real-time Rendering**: GPU-optimized real-time visualization
4. **Machine Learning Integration**: End-to-end learning of scene representations

### 10.2 Research Directions

1. **Neural Radiance Fields**: Integration with NeRF-based scene representations
2. **Physics-informed Networks**: Incorporation of Maxwell's equations
3. **Multi-frequency Modeling**: Support for different frequency bands
4. **Dynamic Scenes**: Real-time updates for moving objects

---

*This document provides the complete design for implementing the discrete electromagnetic ray tracing system in the Prism project. The implementation combines efficient voxel-based ray tracing with advanced optimization strategies to achieve both accuracy and computational efficiency.*
