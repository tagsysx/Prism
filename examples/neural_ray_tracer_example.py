#!/usr/bin/env python3
"""
Example: Neural Network Ray Tracer (NNRayTracer) Usage

Demonstrates the new NNRayTracer which uses PrismNetwork as a TraceNetwork
for direct CSI prediction, bypassing traditional ray tracing calculations.
"""

import sys
import torch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism.networks.prism_network import PrismNetwork
from prism.networks.trace_network import TraceNetwork, TraceNetworkConfig
from prism.tracers import NaiveRayTracer, NNRayTracer

def compare_ray_tracers():
    """Compare NaiveRayTracer vs NNRayTracer performance and results."""
    
    # Create a simple PrismNetwork configuration
    prism_network = PrismNetwork(
        num_subcarriers=64,
        num_bs_antennas=16,
        feature_dim=128,
        azimuth_divisions=36,
        elevation_divisions=10,
        max_ray_length=250.0,
        num_sampling_points=64
    )
    
    # Create TraceNetwork
    trace_config = TraceNetworkConfig(
        R_dim=32,           # Should match PrismNetwork's output_dim
        hidden_dim=256,
        num_attention_heads=8,
        dropout_rate=0.1
    )
    trace_network = TraceNetwork(trace_config)
    
    # Initialize both ray tracers
    naive_tracer = NaiveRayTracer(prism_network)
    nn_tracer = NNRayTracer(prism_network, trace_network)  # Uses separate PrismNetwork and TraceNetwork
    
    # Test positions
    bs_position = torch.tensor([0.0, 0.0, 10.0])  # Base station at height 10m
    ue_positions = [
        torch.tensor([50.0, 50.0, 1.5]),   # UE 1
        torch.tensor([100.0, 0.0, 1.5]),   # UE 2
    ]
    antenna_index = 0
    selected_subcarriers = [0, 16, 32, 48]  # Sample subcarriers
    
    print("🚀 Testing Ray Tracers Comparison")
    print("=" * 50)
    
    # Test NaiveRayTracer
    print("\n📡 Testing NaiveRayTracer (Traditional Method):")
    try:
        with torch.no_grad():
            naive_results = naive_tracer.trace_rays(
                base_station_pos=bs_position,
                ue_positions=ue_positions,
                antenna_index=antenna_index,
                selected_subcarriers=selected_subcarriers
            )
        print(f"✅ NaiveRayTracer Result Shape: {naive_results.shape}")
        print(f"   Sample values: {naive_results[0, :2]}")
    except Exception as e:
        print(f"❌ NaiveRayTracer Error: {e}")
    
    # Test NNRayTracer
    print("\n🧠 Testing NNRayTracer (Neural Network Method):")
    try:
        with torch.no_grad():
            nn_results = nn_tracer.trace_rays(
                base_station_pos=bs_position,
                ue_positions=ue_positions,
                antenna_index=antenna_index,
                selected_subcarriers=selected_subcarriers
            )
        print(f"✅ NNRayTracer Result Shape: {nn_results.shape}")
        print(f"   Sample values: {nn_results[0, :2]}")
    except Exception as e:
        print(f"❌ NNRayTracer Error: {e}")
    
    print("\n🔍 Architecture Comparison:")
    print("NaiveRayTracer:")
    print("  - Uses explicit ray generation and tracing")
    print("  - Implements mathematical ray tracing formulas")
    print("  - Complex signal accumulation calculations")
    print("  - More computationally intensive")
    print("\nNNRayTracer:")
    print("  - Two-stage neural network approach")
    print("  - PrismNetwork extracts ray tracing features")
    print("  - TraceNetwork predicts CSI from features")
    print("  - Attention-based feature fusion")
    print("  - End-to-end learnable pipeline")

def demonstrate_nn_raytracer_features():
    """Demonstrate specific features of NNRayTracer."""
    
    print("\n🎯 NNRayTracer Feature Demonstration")
    print("=" * 40)
    
    # Create PrismNetwork for feature extraction
    prism_network = PrismNetwork(
        num_subcarriers=128,
        num_bs_antennas=64,
        feature_dim=256,
        azimuth_divisions=72,
        elevation_divisions=20
    )
    
    # Create TraceNetwork for CSI prediction
    trace_config = TraceNetworkConfig(
        R_dim=32,
        hidden_dim=512,
        num_attention_heads=8,
        num_transformer_layers=3
    )
    trace_network = TraceNetwork(trace_config)
    
    # Initialize NNRayTracer
    nn_tracer = NNRayTracer(prism_network, trace_network)
    
    print(f"\n📊 NNRayTracer Configuration:")
    print(f"   - TraceNetwork Type: {type(nn_tracer.trace_network).__name__}")
    print(f"   - Subcarriers: {nn_tracer.num_subcarriers}")
    print(f"   - BS Antennas: {nn_tracer.num_bs_antennas}")
    print(f"   - Angular Resolution: {nn_tracer.azimuth_divisions}×{nn_tracer.elevation_divisions}")
    print(f"   - Total Directions: {nn_tracer.total_directions}")
    
    # Test with multiple scenarios
    scenarios = [
        {
            'name': 'Single UE',
            'bs_pos': torch.tensor([0.0, 0.0, 30.0]),
            'ue_positions': [torch.tensor([200.0, 100.0, 1.5])],
            'antenna_idx': 5
        },
        {
            'name': 'Multiple UEs',
            'bs_pos': torch.tensor([100.0, 100.0, 25.0]),
            'ue_positions': [
                torch.tensor([150.0, 150.0, 1.5]),
                torch.tensor([50.0, 150.0, 1.5]),
                torch.tensor([150.0, 50.0, 1.5])
            ],
            'antenna_idx': 10
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🎭 Scenario: {scenario['name']}")
        try:
            with torch.no_grad():
                results = nn_tracer.trace_rays(
                    base_station_pos=scenario['bs_pos'],
                    ue_positions=scenario['ue_positions'],
                    antenna_index=scenario['antenna_idx']
                )
            print(f"   ✅ Success: CSI shape {results.shape}")
            print(f"   📈 CSI magnitude range: [{results.abs().min():.4f}, {results.abs().max():.4f}]")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🧠 Neural Ray Tracer Example")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Run comparisons
    compare_ray_tracers()
    
    # Demonstrate features
    demonstrate_nn_raytracer_features()
    
    print("\n✨ Example completed!")
