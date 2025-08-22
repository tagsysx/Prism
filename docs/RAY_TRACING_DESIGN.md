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

### 4.1 Core Ray Tracing Algorithm

```python
class DiscreteRayTracer:
    def __init__(self, voxel_grid, attenuation_net, radiation_net, config):
        self.voxel_grid = voxel_grid
        self.attenuation_net = attenuation_net
        self.radiation_net = radiation_net
        self.config = config
        
        # Optimization parameters
        self.importance_sampling_enabled = config.get('importance_sampling', True)
        self.pyramid_sampling_enabled = config.get('pyramid_sampling', True)
        self.num_ray_steps = config.get('num_ray_steps', 100)
        self.step_size = config.get('step_size', 0.1)
    
    def trace_ray(self, receiver_pos, direction, transmitter_pos):
        """
        Trace a single ray from receiver position in given direction
        
        Args:
            receiver_pos: Receiver 3D position
            direction: Ray direction vector (normalized)
            transmitter_pos: Transmitter 3D position
        
        Returns:
            received_signal: Complex signal strength
        """
        if self.importance_sampling_enabled:
            return self._trace_ray_with_importance_sampling(
                receiver_pos, direction, transmitter_pos
            )
        else:
            return self._trace_ray_uniform_sampling(
                receiver_pos, direction, transmitter_pos
            )
    
    def _trace_ray_uniform_sampling(self, receiver_pos, direction, transmitter_pos):
        """Uniform sampling ray tracing"""
        accumulated_signal = 0.0
        cumulative_attenuation = 0.0
        
        for step in range(self.num_ray_steps):
            # Calculate voxel position
            voxel_pos = receiver_pos + step * self.step_size * direction
            
            # Get voxel at position
            voxel = self._get_voxel_at_position(voxel_pos)
            if voxel is None:
                continue
            
            # Calculate local contribution
            local_signal = self._calculate_local_signal(
                voxel, -direction, transmitter_pos
            )
            
            # Apply cumulative attenuation
            if step > 0:
                cumulative_attenuation += voxel.attenuation_coeff * self.step_size
            
            attenuated_signal = local_signal * torch.exp(-cumulative_attenuation)
            accumulated_signal += attenuated_signal * self.step_size
        
        return accumulated_signal
```

### 4.2 Importance Sampling Implementation

```python
def _trace_ray_with_importance_sampling(self, receiver_pos, direction, transmitter_pos):
    """Importance sampling ray tracing with non-uniform sampling"""
    
    # Stage 1: Coarse uniform sampling for importance estimation
    coarse_positions = self._generate_coarse_samples(receiver_pos, direction)
    coarse_attenuations = self._estimate_coarse_attenuations(coarse_positions)
    
    # Stage 2: Calculate importance weights
    importance_weights = self._calculate_importance_weights(coarse_attenuations)
    
    # Stage 3: Fine importance-based sampling
    fine_positions = self._generate_fine_samples(importance_weights, coarse_positions)
    
    # Stage 4: Ray tracing with fine sampling
    return self._trace_with_fine_samples(fine_positions, direction, transmitter_pos)

def _generate_coarse_samples(self, receiver_pos, direction):
    """Generate K uniform samples along ray"""
    positions = []
    for k in range(self.config['coarse_samples']):
        pos = receiver_pos + k * self.config['coarse_step_size'] * direction
        positions.append(pos)
    return positions

def _estimate_coarse_attenuations(self, positions):
    """Estimate attenuation factors at coarse positions"""
    attenuations = []
    for pos in positions:
        voxel = self._get_voxel_at_position(pos)
        if voxel:
            # Use real part of attenuation for importance calculation
            beta = torch.real(voxel.attenuation_coeff)
            attenuations.append(beta)
        else:
            attenuations.append(0.0)
    return torch.tensor(attenuations)

def _calculate_importance_weights(self, attenuations):
    """Calculate importance weights for non-uniform sampling"""
    weights = []
    cumulative_attenuation = 0.0
    
    for k, beta in enumerate(attenuations):
        # Calculate segment weight
        segment_weight = (1 - torch.exp(-beta * self.config['coarse_step_size'])) * \
                        torch.exp(-cumulative_attenuation)
        
        weights.append(segment_weight)
        cumulative_attenuation += beta * self.config['coarse_step_size']
    
    # Normalize weights
    weights = torch.tensor(weights)
    weights = weights / torch.sum(weights)
    
    return weights

def _generate_fine_samples(self, importance_weights, coarse_positions):
    """Generate fine samples based on importance weights"""
    # Use inverse CDF sampling
    cdf = torch.cumsum(importance_weights, dim=0)
    
    fine_positions = []
    for i in range(self.config['fine_samples']):
        # Generate random sample
        u = torch.rand(1)
        
        # Find corresponding position using CDF
        idx = torch.searchsorted(cdf, u)
        idx = torch.clamp(idx, 0, len(coarse_positions) - 1)
        
        # Interpolate position
        if idx == 0:
            pos = coarse_positions[0]
        else:
            alpha = (u - cdf[idx-1]) / (cdf[idx] - cdf[idx-1])
            pos = coarse_positions[idx-1] + alpha * (coarse_positions[idx] - coarse_positions[idx-1])
        
        fine_positions.append(pos)
    
    return fine_positions
```

### 4.3 Pyramid Sampling Implementation

```python
class PyramidSampler:
    def __init__(self, config):
        self.azimuth_divisions = config.get('azimuth_divisions', 36)  # A
        self.elevation_divisions = config.get('elevation_divisions', 18)  # B
        self.max_depth = config.get('max_depth', 100.0)
        self.samples_per_pyramid = config.get('samples_per_pyramid', 10)
    
    def generate_directional_samples(self, receiver_pos):
        """Generate directional samples using pyramid structure"""
        directions = []
        
        for a in range(self.azimuth_divisions):
            for b in range(self.elevation_divisions):
                # Calculate pyramid boundaries
                phi_min = 2 * torch.pi * a / self.azimuth_divisions
                phi_max = 2 * torch.pi * (a + 1) / self.azimuth_divisions
                theta_min = torch.pi * b / self.elevation_divisions
                theta_max = torch.pi * (b + 1) / self.elevation_divisions
                
                # Generate samples within pyramid
                pyramid_directions = self._sample_pyramid(
                    phi_min, phi_max, theta_min, theta_max
                )
                directions.extend(pyramid_directions)
        
        return directions
    
    def _sample_pyramid(self, phi_min, phi_max, theta_min, theta_max):
        """Sample directions within a single pyramid"""
        directions = []
        
        for _ in range(self.samples_per_pyramid):
            # Uniform sampling within pyramid
            phi = phi_min + torch.rand(1) * (phi_max - phi_min)
            theta = theta_min + torch.rand(1) * (theta_max - theta_min)
            
            # Convert to Cartesian coordinates
            direction = torch.tensor([
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta)
            ])
            
            directions.append(direction)
        
        return directions
```

## 5. Signal Aggregation and Training

### 5.1 Multi-Directional Signal Aggregation

```python
def aggregate_signals(self, receiver_pos, transmitter_pos):
    """Aggregate signals from all directions"""
    if self.pyramid_sampling_enabled:
        directions = self.pyramid_sampler.generate_directional_samples(receiver_pos)
    else:
        directions = self._generate_uniform_directions()
    
    total_signal = 0.0
    for direction in directions:
        signal = self.trace_ray(receiver_pos, direction, transmitter_pos)
        total_signal += signal
    
    return total_signal / len(directions)

def _generate_uniform_directions(self):
    """Generate uniform directional samples"""
    # Fibonacci sphere sampling for uniform distribution
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
        Calculate training loss
        
        Args:
            predicted_signals: Model predictions [batch_size, num_receivers]
            target_signals: Ground truth measurements [batch_size, num_receivers]
        
        Returns:
            loss: Training loss value
        """
        if self.loss_type == 'l2':
            return self.criterion(predicted_signals, target_signals)
        elif self.loss_type == 'l1':
            return self.criterion(predicted_signals, target_signals)
        
        # Complex signal handling
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
        self.attenuation_net = None
        self.radiation_net = None
        self.ray_tracer = None
        
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
        # Training implementation
        pass
    
    def predict_signals(self, receiver_positions, transmitter_positions):
        """Predict received signals for given positions"""
        signals = []
        
        for rx_pos, tx_pos in zip(receiver_positions, transmitter_positions):
            signal = self.ray_tracer.aggregate_signals(rx_pos, tx_pos)
            signals.append(signal)
        
        return torch.stack(signals)
    
    def visualize_ray_path(self, receiver_pos, direction, transmitter_pos):
        """Visualize ray path for debugging"""
        # Visualization implementation
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

# Predict signals
receiver_positions = torch.tensor([[0, 0, 0], [10, 0, 0], [0, 10, 0]])
transmitter_positions = torch.tensor([[5, 5, 5]] * 3)

predicted_signals = ray_tracer.predict_signals(
    receiver_positions, transmitter_positions
)

print(f"Predicted signals: {predicted_signals}")
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
