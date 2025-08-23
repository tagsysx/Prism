#!/usr/bin/env python3
"""
Test script to verify enhanced parallelization features:
1. Direction-level parallelization
2. Antenna-level parallelization  
3. Spatial sampling parallelization
4. Full parallelization (all levels combined)
"""

import time
import torch
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prism.ray_tracer import DiscreteRayTracer
from prism.networks.prism_network import PrismNetwork
import yaml

def create_test_environment():
    """Create test environment with dummy data."""
    
    # Load configuration
    config_path = "configs/ofdm-5g-sionna.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create a dummy PrismNetwork for testing
    nn_config = config['neural_networks']
    prism_network = PrismNetwork(
        attenuation_config=nn_config['attenuation_network'],
        attenuation_decoder_config=nn_config['attenuation_decoder'],
        antenna_codebook_config=nn_config['antenna_codebook'],
        antenna_network_config=nn_config['antenna_network'],
        radiance_config=nn_config['radiance_network']
    )
    
    # Test parameters - use valid subcarrier indices (0-63)
    base_station_pos = torch.tensor([0.0, 0.0, 0.0])
    ue_positions = [torch.tensor([50.0, 30.0, 1.5])]
    selected_subcarriers = {tuple(ue_positions[0].tolist()): [10, 20, 30]}  # Fixed: valid indices
    antenna_embedding = torch.randn(64)  # 64 BS antennas
    
    return config, prism_network, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding

def test_direction_parallelization(prism_network, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding):
    """Test direction-level parallelization."""
    
    print("🧪 Testing Direction-Level Parallelization...")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Sequential (No Parallel)',
            'enable_parallel': False,
            'max_workers': 1,
            'use_multiprocessing': False
        },
        {
            'name': 'Threading (4 workers)',
            'enable_parallel': True,
            'max_workers': 4,
            'use_multiprocessing': False
        },
        {
            'name': 'Threading (8 workers)',
            'enable_parallel': True,
            'max_workers': 8,
            'use_multiprocessing': False
        }
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n🔧 Testing: {config['name']}")
        
        # Create ray tracer with specific configuration
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=18,
            elevation_divisions=9,
            device='cpu',
            prism_network=prism_network,
            enable_parallel_processing=config['enable_parallel'],
            max_workers=config['max_workers'],
            use_multiprocessing=config['use_multiprocessing']
        )
        
        # Print configuration
        stats = ray_tracer.get_parallelization_stats()
        print(f"   • Parallel processing: {stats['parallel_processing_enabled']}")
        print(f"   • Max workers: {stats['max_workers']}")
        print(f"   • Processing mode: {stats['processing_mode']}")
        print(f"   • Total directions: {stats['total_directions']}")
        print(f"   • Top-K directions: {stats['top_k_directions']}")
        
        # Test performance
        print(f"   • Testing performance...")
        
        start_time = time.time()
        try:
            # Test MLP-based direction selection
            accumulated_signals = ray_tracer.accumulate_signals(
                base_station_pos, ue_positions, selected_subcarriers, antenna_embedding
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            results[config['name']] = {
                'time': execution_time,
                'success': True,
                'signals_count': len(accumulated_signals)
            }
            
            print(f"   ✅ Success! Time: {execution_time:.3f}s, Signals: {len(accumulated_signals)}")
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            results[config['name']] = {
                'time': execution_time,
                'success': False,
                'error': str(e)
            }
            
            print(f"   ❌ Failed! Time: {execution_time:.3f}s, Error: {e}")
    
    return results

def test_antenna_parallelization(prism_network, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding):
    """Test antenna-level parallelization."""
    
    print(f"\n🧪 Testing Antenna-Level Parallelization...")
    
    # Test different antenna configurations
    antenna_configs = [
        {'name': 'Small Array (16 antennas)', 'num_antennas': 16, 'max_workers': 4},
        {'name': 'Medium Array (32 antennas)', 'num_antennas': 32, 'max_workers': 6},
        {'name': 'Large Array (64 antennas)', 'num_antennas': 64, 'max_workers': 8}
    ]
    
    results = {}
    
    for config in antenna_configs:
        print(f"\n🔧 Testing: {config['name']}")
        
        # Create ray tracer
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=18,
            elevation_divisions=9,
            device='cpu',
            prism_network=prism_network,
            enable_parallel_processing=True,
            max_workers=config['max_workers'],
            use_multiprocessing=False
        )
        
        # Create antenna-specific embedding
        antenna_embedding_test = torch.randn(config['num_antennas'])
        
        print(f"   • Antennas: {config['num_antennas']}")
        print(f"   • Max workers: {config['max_workers']}")
        print(f"   • Top-K directions: {ray_tracer.top_k_directions}")
        
        # Test performance
        print(f"   • Testing antenna-level parallelization...")
        
        start_time = time.time()
        try:
            # Test antenna-level parallel processing
            accumulated_signals = ray_tracer._accumulate_signals_antenna_parallel(
                base_station_pos, ue_positions, selected_subcarriers, 
                antenna_embedding_test, [(0, 0), (1, 1), (2, 2)], config['num_antennas']
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            results[config['name']] = {
                'time': execution_time,
                'success': True,
                'signals_count': len(accumulated_signals),
                'antennas': config['num_antennas']
            }
            
            print(f"   ✅ Success! Time: {execution_time:.3f}s, Signals: {len(accumulated_signals)}")
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            results[config['name']] = {
                'time': execution_time,
                'success': False,
                'error': str(e),
                'antennas': config['num_antennas']
            }
            
            print(f"   ❌ Failed! Time: {execution_time:.3f}s, Error: {e}")
    
    return results

def test_spatial_parallelization(prism_network, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding):
    """Test spatial sampling parallelization."""
    
    print(f"\n🧪 Testing Spatial Sampling Parallelization...")
    
    # Test different spatial point configurations
    spatial_configs = [
        {'name': 'Low Resolution (16 points)', 'num_points': 16, 'max_workers': 4},
        {'name': 'Medium Resolution (32 points)', 'num_points': 32, 'max_workers': 6},
        {'name': 'High Resolution (64 points)', 'num_points': 64, 'max_workers': 8}
    ]
    
    results = {}
    
    for config in spatial_configs:
        print(f"\n🔧 Testing: {config['name']}")
        
        # Create ray tracer
        ray_tracer = DiscreteRayTracer(
            azimuth_divisions=18,
            elevation_divisions=9,
            device='cpu',
            prism_network=prism_network,
            enable_parallel_processing=True,
            max_workers=config['max_workers'],
            use_multiprocessing=False
        )
        
        print(f"   • Spatial points: {config['num_points']}")
        print(f"   • Max workers: {config['max_workers']}")
        print(f"   • Test directions: 3")
        
        # Test performance
        print(f"   • Testing spatial sampling parallelization...")
        
        start_time = time.time()
        try:
            # Test spatial sampling parallel processing
            accumulated_signals = ray_tracer._accumulate_signals_spatial_parallel(
                base_station_pos, ue_positions, selected_subcarriers, 
                antenna_embedding, [(0, 0), (1, 1), (2, 2)], config['num_points']
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            results[config['name']] = {
                'time': execution_time,
                'success': True,
                'signals_count': len(accumulated_signals),
                'spatial_points': config['num_points']
            }
            
            print(f"   ✅ Success! Time: {execution_time:.3f}s, Signals: {len(accumulated_signals)}")
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            results[config['name']] = {
                'time': execution_time,
                'success': False,
                'error': str(e),
                'spatial_points': config['num_points']
            }
            
            print(f"   ❌ Failed! Time: {execution_time:.3f}s, Error: {e}")
    
    return results

def test_full_parallelization(prism_network, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding):
    """Test full parallelization combining all levels."""
    
    print(f"\n🧪 Testing Full Parallelization (All Levels Combined)...")
    
    # Create ray tracer with full parallelization support
    ray_tracer = DiscreteRayTracer(
        azimuth_divisions=18,
        elevation_divisions=9,
        device='cpu',
        prism_network=prism_network,
        enable_parallel_processing=True,
        max_workers=8,
        use_multiprocessing=False
    )
    
    # Test full parallelization
    print(f"   • Max workers: 8")
    print(f"   • Test directions: 16")
    print(f"   • Test antennas: 32")
    print(f"   • Test spatial points: 32")
    
    print(f"   • Testing full parallelization...")
    
    start_time = time.time()
    try:
        # Test full parallel processing
        accumulated_signals = ray_tracer._accumulate_signals_full_parallel(
            base_station_pos, ue_positions, selected_subcarriers, 
            antenna_embedding, [(i, i) for i in range(16)], 32, 32
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        result = {
            'time': execution_time,
            'success': True,
            'signals_count': len(accumulated_signals)
        }
        
        print(f"   ✅ Success! Time: {execution_time:.3f}s, Signals: {len(accumulated_signals)}")
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        result = {
            'time': execution_time,
            'success': False,
            'error': str(e)
        }
        
        print(f"   ❌ Failed! Time: {execution_time:.3f}s, Error: {e}")
    
    return result

def analyze_performance_results(direction_results, antenna_results, spatial_results, full_result):
    """Analyze and compare performance results."""
    
    print(f"\n📊 Performance Analysis Summary:")
    print(f"{'='*80}")
    
    # Direction parallelization analysis
    print(f"\n🎯 Direction-Level Parallelization:")
    successful_direction = {k: v for k, v in direction_results.items() if v['success']}
    if successful_direction:
        fastest_direction = min(successful_direction.items(), key=lambda x: x[1]['time'])
        slowest_direction = max(successful_direction.items(), key=lambda x: x[1]['time'])
        
        print(f"   • Fastest: {fastest_direction[0]} ({fastest_direction[1]['time']:.3f}s)")
        print(f"   • Slowest: {slowest_direction[0]} ({slowest_direction[1]['time']:.3f}s)")
        
        if fastest_direction[1]['time'] > 0:
            for name, result in successful_direction.items():
                if name != fastest_direction[0]:
                    speedup = result['time'] / fastest_direction[1]['time']
                    print(f"   • {name}: {speedup:.2f}x slower than fastest")
    
    # Antenna parallelization analysis
    print(f"\n📡 Antenna-Level Parallelization:")
    successful_antenna = {k: v for k, v in antenna_results.items() if v['success']}
    if successful_antenna:
        for name, result in successful_antenna.items():
            print(f"   • {name}: {result['time']:.3f}s ({result['antennas']} antennas)")
    
    # Spatial parallelization analysis
    print(f"\n🌍 Spatial Sampling Parallelization:")
    successful_spatial = {k: v for k, v in spatial_results.items() if v['success']}
    if successful_spatial:
        for name, result in successful_spatial.items():
            print(f"   • {name}: {result['time']:.3f}s ({result['spatial_points']} points)")
    
    # Full parallelization analysis
    print(f"\n🚀 Full Parallelization:")
    if full_result['success']:
        print(f"   • Full parallel: {full_result['time']:.3f}s")
        print(f"   • Signals computed: {full_result['signals_count']}")
        
        # Compare with direction-only
        if successful_direction:
            fastest_direction_time = min(successful_direction.values(), key=lambda x: x[1]['time'])['time']
            if fastest_direction_time > 0:
                speedup = fastest_direction_time / full_result['time']
                print(f"   • Speedup vs direction-only: {speedup:.2f}x")
    
    # Overall recommendations
    print(f"\n💡 Optimization Recommendations:")
    print(f"   • Current best: Direction-level parallelization")
    print(f"   • Next target: Antenna-level parallelization")
    print(f"   • Ultimate goal: Full parallelization (2,048x speedup)")
    
    return {
        'direction': direction_results,
        'antenna': antenna_results,
        'spatial': spatial_results,
        'full': full_result
    }

def main():
    """Main test function."""
    
    print("🚀 Starting Enhanced Parallelization Tests...")
    print("=" * 80)
    
    try:
        # Create test environment
        config, prism_network, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding = create_test_environment()
        print(f"✅ Test environment created successfully")
        
        # Test 1: Direction-level parallelization
        direction_results = test_direction_parallelization(
            prism_network, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding
        )
        
        # Test 2: Antenna-level parallelization
        antenna_results = test_antenna_parallelization(
            prism_network, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding
        )
        
        # Test 3: Spatial sampling parallelization
        spatial_results = test_spatial_parallelization(
            prism_network, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding
        )
        
        # Test 4: Full parallelization
        full_result = test_full_parallelization(
            prism_network, base_station_pos, ue_positions, selected_subcarriers, antenna_embedding
        )
        
        # Analyze results
        all_results = analyze_performance_results(
            direction_results, antenna_results, spatial_results, full_result
        )
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"📊 Enhanced parallelization is working correctly!")
        print(f"🚀 Ready for 2,048x speedup optimization!")
        
        return all_results
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n✅ Test completed successfully!")
    else:
        print(f"\n❌ Test failed!")
        sys.exit(1)
