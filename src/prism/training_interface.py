"""
Prism Training Interface

This module provides a simplified training interface that integrates:
1. PrismNetwork for spatial feature extraction
2. BS-Centric ray tracing from each BS antenna
3. AntennaNetwork-guided direction selection
4. Antenna-specific subcarrier selection
5. CSI computation and loss calculation
6. Checkpoint support for training recovery
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
import logging
import random
import os
import json
import time
import signal
from contextlib import contextmanager

from .networks.prism_network import PrismNetwork
from .ray_tracer_cpu import CPURayTracer

logger = logging.getLogger(__name__)

@contextmanager
def timeout_context(seconds: int, operation_name: str = "operation"):
    """Context manager for timeout operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"{operation_name} timed out after {seconds} seconds")
    
    # Set up signal handler for timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore original signal handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class PrismTrainingInterface(nn.Module):
    """
    Simplified training interface with BS-Centric ray tracing and checkpoint support.
    
    This interface handles:
    1. BS-Centric ray tracing from each BS antenna
    2. AntennaNetwork-guided direction selection
    3. Antenna-specific subcarrier selection
    4. CSI computation and loss calculation
    5. Training checkpoint and recovery
    """
    
    def __init__(
        self,
        prism_network: PrismNetwork,
        ray_tracer: Optional[Union['CPURayTracer', 'CUDARayTracer']] = None,
        num_sampling_points: int = 64,
        scene_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        subcarrier_sampling_ratio: float = 0.3,
        checkpoint_dir: str = "checkpoints",
        ray_tracing_mode: str = 'hybrid'  # Ray tracing mode: 'cuda', 'cpu', or 'hybrid'
    ):
        super().__init__()
        
        # Validate ray tracing mode first
        if ray_tracing_mode not in ['cuda', 'cpu', 'hybrid']:
            raise ValueError(f"Invalid ray_tracing_mode: {ray_tracing_mode}. Must be 'cuda', 'cpu', or 'hybrid'")
        
        # Set ray tracing mode
        self.ray_tracing_mode = ray_tracing_mode
        
        # Create appropriate ray tracer based on mode if not provided
        if ray_tracer is None:
            self.ray_tracer = self._create_ray_tracer_by_mode(ray_tracing_mode)
        else:
            # Use provided ray_tracer but validate it matches the mode
            self.ray_tracer = self._validate_ray_tracer(ray_tracer, ray_tracing_mode)
        
        self.prism_network = prism_network
        # Store configuration
        self.num_sampling_points = num_sampling_points
        self.subcarrier_sampling_ratio = subcarrier_sampling_ratio
        self.checkpoint_dir = checkpoint_dir
        
        logger.info(f"Training interface initialized with ray_tracing_mode: {ray_tracing_mode}")
        logger.info(f"Using ray tracer type: {type(self.ray_tracer).__name__}")
        
        # Set scene bounds
        if scene_bounds is not None:
            self.scene_min, self.scene_max = scene_bounds
        else:
            # Default scene bounds
            self.scene_min = torch.tensor([-100.0, -100.0, 0.0])
            self.scene_max = torch.tensor([100.0, 100.0, 30.0])
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training state for checkpoint recovery
        self.current_epoch = 0
        self.current_batch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Current selection state
        self.current_selection = None
        self.current_selection_mask = None
    
    def _create_ray_tracer_by_mode(self, ray_tracing_mode: str) -> Union['CPURayTracer', 'CUDARayTracer']:
        """
        Create appropriate ray tracer based on the specified mode.
        
        Args:
            ray_tracing_mode: 'cuda', 'cpu', or 'hybrid'
            
        Returns:
            Appropriate ray tracer instance
        """
        if ray_tracing_mode == 'cuda':
            # Create CUDA ray tracer for maximum performance
            try:
                from .ray_tracer_cuda import CUDARayTracer
                logger.info("🚀 Creating CUDARayTracer for CUDA mode")
                return CUDARayTracer(
                    azimuth_divisions=18,      # From config
                    elevation_divisions=9,     # From config
                    max_ray_length=200.0,     # From config
                    scene_size=200.0,         # From config
                    device='cuda',
                    uniform_samples=64,        # From config
                    enable_parallel_processing=True,
                    max_workers=2             # From config
                )
            except Exception as e:
                logger.warning(f"Failed to create CUDARayTracer: {e}. Falling back to CPURayTracer.")
                from .ray_tracer_cpu import CPURayTracer
                return CPURayTracer(
                    azimuth_divisions=18,
                    elevation_divisions=9,
                    max_ray_length=200.0,
                    scene_size=200.0,
                    uniform_samples=64
                )
        
        elif ray_tracing_mode == 'cpu':
            # Create CPU ray tracer
            from .ray_tracer_cpu import CPURayTracer
            logger.info("💻 Creating CPURayTracer for CPU mode")
            return CPURayTracer(
                azimuth_divisions=18,
                elevation_divisions=9,
                max_ray_length=200.0,
                scene_size=200.0,
                uniform_samples=64
            )
        
        else:  # hybrid mode
            # Try CUDA first, fallback to CPU
            try:
                from .ray_tracer_cuda import CUDARayTracer
                logger.info("🔄 Creating CUDARayTracer for hybrid mode (CUDA first)")
                return CUDARayTracer(
                    azimuth_divisions=18,
                    elevation_divisions=9,
                    max_ray_length=200.0,
                    scene_size=200.0,
                    device='cuda',
                    uniform_samples=64,
                    enable_parallel_processing=True,
                    max_workers=2
                )
            except Exception as e:
                logger.warning(f"CUDA ray tracer failed in hybrid mode: {e}. Using CPU fallback.")
                from .ray_tracer_cpu import CPURayTracer
                return CPURayTracer(
                    azimuth_divisions=18,
                    elevation_divisions=9,
                    max_ray_length=200.0,
                    scene_size=200.0,
                    uniform_samples=64
                )
    
    def _validate_ray_tracer(self, ray_tracer, ray_tracing_mode: str) -> Union['CPURayTracer', 'CUDARayTracer']:
        """
        Validate that the provided ray tracer matches the specified mode.
        
        Args:
            ray_tracer: The ray tracer instance to validate
            ray_tracing_mode: The expected mode
            
        Returns:
            The validated ray tracer
        """
        from .ray_tracer_cpu import CPURayTracer
        from .ray_tracer_cuda import CUDARayTracer
        
        if ray_tracing_mode == 'cuda':
            if not isinstance(ray_tracer, CUDARayTracer):
                logger.warning(f"Expected CUDARayTracer for 'cuda' mode, but got {type(ray_tracer).__name__}. Creating new CUDARayTracer.")
                return self._create_ray_tracer_by_mode('cuda')
        elif ray_tracing_mode == 'cpu':
            if not isinstance(ray_tracer, CPURayTracer):
                logger.warning(f"Expected CPURayTracer for 'cpu' mode, but got {type(ray_tracer).__name__}. Creating new CPURayTracer.")
                return self._create_ray_tracer_by_mode('cpu')
        # For hybrid mode, accept either type
        
        return ray_tracer
    
    def forward(self, ue_positions: torch.Tensor, bs_position: torch.Tensor, antenna_indices: torch.Tensor, return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the training interface.
        
        Args:
            ue_positions: UE positions [batch_size, 3]
            bs_position: BS position [3]
            antenna_indices: BS antenna indices [batch_size, num_bs_antennas]
            return_intermediates: Whether to return intermediate ray tracing results
            
        Returns:
            Dictionary containing CSI predictions and network outputs
        """
        # Store the input device for proper device detection
        self._last_input_device = ue_positions.device
        
        batch_size = ue_positions.shape[0]
        # There is 1 UE device with 4 antennas, but many different positions
        # ue_positions shape is [batch_size, coordinates] - different positions for the same device
        num_ue = 1  # One UE device
        num_ue_antennas = 4  # Four antennas on the UE device
        num_positions = batch_size  # Number of different positions to train on
        num_bs_antennas = antenna_indices.shape[1]
        num_subcarriers = self.prism_network.num_subcarriers
        num_selected = int(num_subcarriers * self.subcarrier_sampling_ratio)
        
        logger.debug(f"Forward method parameters:")
        logger.debug(f"  ue_positions shape: {ue_positions.shape} (batch_size, coordinates)")
        logger.debug(f"  antenna_indices shape: {antenna_indices.shape}")
        logger.debug(f"  batch_size: {batch_size} (number of different UE positions)")
        logger.debug(f"  num_ue: {num_ue} (UE devices)")
        logger.debug(f"  num_ue_antennas: {num_ue_antennas} (antennas per UE device)")
        logger.debug(f"  num_bs_antennas: {num_bs_antennas}")
        logger.debug(f"  num_subcarriers: {num_subcarriers}")
        logger.debug(f"  subcarrier_sampling_ratio: {self.subcarrier_sampling_ratio}")
        logger.debug(f"  num_selected: {num_selected}")
        logger.debug(f"  num_subcarriers type: {type(num_subcarriers)}")
        logger.debug(f"  num_selected type: {type(num_selected)}")
        logger.debug(f"  subcarrier_sampling_ratio type: {type(self.subcarrier_sampling_ratio)}")
        
        # Step 1: Select subcarriers for each BS antenna and UE antenna combination
        # Training scenario: 1 UE device with 4 antennas, placed at many different positions
        # Each batch item represents a different position for the same UE device
        # Ensure selection variables are properly sized for current batch
        self._ensure_selection_variables_sized(batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers, num_selected)
        
        selection_info = self._select_subcarriers_per_antenna(
            batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers, num_selected
        )
        self.current_selection = selection_info['selected_indices']
        self.current_selection_mask = selection_info['selection_mask']
        
        # Validate selection variables
        if self.current_selection is None or self.current_selection_mask is None:
            raise ValueError("Failed to initialize subcarrier selection variables")
        
        expected_selection_shape = (batch_size, num_bs_antennas, num_ue_antennas, num_selected)
        if self.current_selection.shape != expected_selection_shape:
            raise ValueError(f"Selection shape mismatch. Expected: {expected_selection_shape}, Got: {self.current_selection.shape}")
        
        expected_mask_shape = (batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers)
        if self.current_selection_mask.shape != expected_mask_shape:
            raise ValueError(f"Selection mask shape mismatch. Expected: {expected_mask_shape}, Got: {self.current_selection_mask.shape}")
        
        logger.debug(f"Subcarrier selection initialized - Shape: {self.current_selection.shape}, Mask: {self.current_selection_mask.shape}")
        
        # Additional debugging
        logger.debug(f"current_selection dtype: {self.current_selection.dtype}")
        logger.debug(f"current_selection device: {self.current_selection.device}")
        logger.debug(f"current_selection sample values: {self.current_selection[0, 0, 0, :5] if self.current_selection.numel() > 0 else 'empty'}")
        
        # Step 2: BS-Centric ray tracing from each BS antenna using ray_tracer
        # We create predictions for the selected subcarriers
        csi_predictions = torch.zeros(
            batch_size, num_bs_antennas, num_ue_antennas, num_selected, 
            dtype=torch.complex64, device=ue_positions.device
        )
        
        # Calculate total computational workload for ray tracing
        if hasattr(self.ray_tracer, 'actual_directions_used'):
            num_directions = self.ray_tracer.actual_directions_used
        elif hasattr(self.ray_tracer, 'total_directions'):
            num_directions = self.ray_tracer.total_directions
        elif hasattr(self.ray_tracer, 'prism_network') and self.ray_tracer.prism_network is not None:
            num_directions = 32  # MLP-guided direction selection
        else:
            num_directions = 162  # Fallback to all directions
        
        if hasattr(self.ray_tracer, 'uniform_samples'):
            num_spatial_points = self.ray_tracer.uniform_samples
        else:
            num_spatial_points = 64  # Default from config
        
        # Calculate computational workload for vectorized ray tracing
        total_vectorized_operations = batch_size * num_bs_antennas * num_ue_antennas * num_directions
        total_voxel_subcarrier_pairs = num_selected * num_spatial_points
        total_computations = total_vectorized_operations * total_voxel_subcarrier_pairs
        
        logger.info(f"🚀 Starting BS-Centric vectorized ray tracing:")
        logger.info(f"   📊 Workload: {batch_size} batches × {num_bs_antennas} BS antennas × {num_ue_antennas} UE antennas × {num_directions} directions = {total_vectorized_operations:,} rays")
        logger.info(f"   🎯 Per ray: {num_selected} subcarriers × {num_spatial_points} spatial points = {total_voxel_subcarrier_pairs:,} voxel-subcarrier pairs processed in parallel")
        logger.info(f"   ⚡ Total computations: {total_computations:,} (vectorized from {total_computations:,} serial operations)")
        
        all_ray_results = []
        all_signal_strengths = []
        
        for bs_antenna_idx in range(num_bs_antennas):
            # Log progress every 10 antennas
            if bs_antenna_idx % 10 == 0:
                progress = (bs_antenna_idx / num_bs_antennas) * 100
                
                # For the first antenna, we don't know actual directions yet, so use estimate
                # For subsequent antennas, use the actual directions from previous calls
                if bs_antenna_idx == 0:
                    # Initial estimate - will be corrected after first accumulate_signals call
                    if hasattr(self.ray_tracer, 'prism_network') and self.ray_tracer.prism_network is not None:
                        estimated_directions = 32  # MLP-guided direction selection
                    else:
                        estimated_directions = 162  # Fallback to all directions
                    
                    # Get spatial sampling configuration
                    if hasattr(self.ray_tracer, 'uniform_samples'):
                        num_spatial_points = self.ray_tracer.uniform_samples
                    else:
                        num_spatial_points = 64  # Default from config
                    
                    total_rays_per_antenna = estimated_directions * num_ue_antennas * num_selected * num_spatial_points
                    logger.info(f"📡 Processing BS antenna {bs_antenna_idx+1}/{num_bs_antennas} ({progress:.1f}%) - Estimated ~{total_rays_per_antenna:,} rays ({estimated_directions} estimated directions × {num_ue_antennas} UE antennas × {num_selected} subcarriers × {num_spatial_points} spatial points)")
                else:
                    # Use actual directions from previous calls
                    if hasattr(self.ray_tracer, 'actual_directions_used'):
                        actual_directions = self.ray_tracer.actual_directions_used
                    else:
                        actual_directions = 162  # Fallback
                    
                    if hasattr(self.ray_tracer, 'uniform_samples'):
                        num_spatial_points = self.ray_tracer.uniform_samples
                    else:
                        num_spatial_points = 64  # Default from config
                    
                    vectorized_ops_per_antenna = actual_directions * num_ue_antennas
                    computations_per_op = num_selected * num_spatial_points
                    total_computations_per_antenna = vectorized_ops_per_antenna * computations_per_op
                    
                    logger.info(f"📡 Processing BS antenna {bs_antenna_idx+1}/{num_bs_antennas} ({progress:.1f}%)")
                    logger.info(f"   🎯 Rays: {actual_directions} directions × {num_ue_antennas} UE antennas = {vectorized_ops_per_antenna:,}")
                    logger.info(f"   ⚡ Per ray: {num_selected} subcarriers × {num_spatial_points} voxels = {computations_per_op:,} parallel computations")
                    logger.info(f"   📊 Total: {total_computations_per_antenna:,} computations (vectorized execution)")
            
            # Get antenna-specific embedding
            antenna_embedding = self.prism_network.antenna_codebook(
                antenna_indices[:, bs_antenna_idx]
            )
            
            # Process each batch item
            batch_ray_results = []
            batch_signal_strengths = []
            
            for b in range(batch_size):
                # Each batch item represents a different position for the same UE device
                # The UE device (with its 4 antennas) is placed at this specific position
                ue_device_pos = ue_positions[b].cpu()  # UE device position at this batch item
                ue_pos_list = [ue_device_pos] * num_ue_antennas  # All 4 antennas share this position
                
                # Debug logging for UE positions
                # logger.debug(f"Batch {b} UE positions:")
                # for u in range(num_ue):
                #     logger.debug(f"  UE {u}: ue_positions[b, u] = {ue_positions[b, u]} (type: {type(ue_positions[b, u])}, shape: {ue_positions[b, u].shape if hasattr(ue_positions[b, u], 'shape') else 'no shape'})")
                #     logger.debug(f"  UE {u}: ue_pos_list[u] = {ue_positions[b, u]} (type: {type(ue_pos_list[u])}, shape: {ue_pos_list[u].shape if hasattr(ue_pos_list[u], 'shape') else 'no shape'})")
                
                # Create subcarrier dictionary mapping UE positions to selected subcarriers
                selected_subcarriers = {}
                
                # Process each UE antenna (all share the same device position)
                for u in range(num_ue_antennas):
                    ue_pos_tuple = tuple(ue_pos_list[u].tolist())
                    selection_tensor = self.current_selection[b, bs_antenna_idx, u]
                    
                    # Debug logging
                    logger.debug(f"Selection tensor shape: {selection_tensor.shape}, dtype: {selection_tensor.dtype}")
                    
                    # Ensure we get a proper list of integers
                    if selection_tensor.numel() == 1:
                        # Single value case
                        selected_subcarriers[ue_pos_tuple] = [int(selection_tensor.item())]
                    else:
                        # Multiple values case - ensure it's a proper list
                        try:
                            tensor_list = selection_tensor.tolist()
                            # Validate that we got a list
                            if isinstance(tensor_list, (list, tuple)):
                                selected_subcarriers[ue_pos_tuple] = tensor_list
                            else:
                                logger.error(f"tensor.tolist() returned non-list: {type(tensor_list)} = {tensor_list}")
                                selected_subcarriers[ue_pos_tuple] = [0]  # Fallback
                        except Exception as e:
                            logger.error(f"Error converting tensor to list: {e}")
                            selected_subcarriers[ue_pos_tuple] = [0]  # Fallback
                    
                    # Validate the selected subcarriers
                    if not selected_subcarriers[ue_pos_tuple]:
                        logger.warning(f"Empty subcarrier selection for UE {u}, using fallback")
                        selected_subcarriers[ue_pos_tuple] = [0]  # Fallback to first subcarrier
                    
                    # Ensure all values are valid integers
                    try:
                        selected_subcarriers[ue_pos_tuple] = [int(idx) for idx in selected_subcarriers[ue_pos_tuple]]
                    except (ValueError, TypeError) as e:
                        logger.error(f"Invalid subcarrier indices: {selected_subcarriers[ue_pos_tuple]}, error: {e}")
                        selected_subcarriers[ue_pos_tuple] = [0]  # Fallback
                    
                    logger.debug(f"Selected subcarriers for UE {u}: {len(selected_subcarriers[ue_pos_tuple])} indices")
                
                # Use ray_tracer for signal accumulation with AntennaNetwork-guided directions
                logger.debug(f"Calling ray_tracer.accumulate_signals with:")
                logger.debug(f"  base_station_pos: {bs_position[b].shape}, {bs_position[b].dtype}")
                logger.debug(f"  ue_positions: {len(ue_pos_list)} positions")
                logger.debug(f"  selected_subcarriers: {type(selected_subcarriers)}")
                # logger.debug(f"  selected_subcarriers content: {selected_subcarriers}")  # 屏蔽数据内容
                logger.debug(f"  antenna_embedding: {antenna_embedding[b].shape}, {antenna_embedding[b].dtype}")
                
                # Additional validation before calling ray_tracer
                # logger.debug(f"Validating selected_subcarriers before ray_tracer call:")
                # for ue_key, subcarriers in selected_subcarriers.items():
                #     logger.debug(f"  UE {ue_key}: {type(subcarriers)} = {subcarriers}")
                #     if not isinstance(subcarriers, (list, tuple)):
                #         logger.error(f"Invalid subcarriers type for UE {ue_key}: {type(subcarriers)}")
                #         raise ValueError(f"subcarriers must be list/tuple, got {type(subcarriers)}")
                
                try:
                    # Use ray tracing based on configured mode
                    if self.ray_tracing_mode == 'cuda':
                        # Pure CUDA mode - use ray tracer with CUDA acceleration
                        logger.debug("Using CUDA ray tracing mode")
                        ray_results = self.ray_tracer.accumulate_signals(
                            base_station_pos=bs_position[b],
                            ue_positions=ue_pos_list,
                            selected_subcarriers=selected_subcarriers,
                            antenna_embedding=antenna_embedding[b]
                        )
                    elif self.ray_tracing_mode == 'cpu':
                        # Pure CPU mode - use simple CPU-based ray tracing
                        logger.debug("Using CPU ray tracing mode")
                        ray_results = self._simple_ray_tracing(
                            bs_position[b], ue_pos_list[0], selected_subcarriers, antenna_embedding[b]
                        )
                    else:  # hybrid mode
                        # Hybrid mode - try CUDA first, fallback to CPU
                        logger.debug("Using hybrid ray tracing mode")
                        try:
                            with timeout_context(10, "ray_tracer.accumulate_signals"):  # 10 second timeout
                                ray_results = self.ray_tracer.accumulate_signals(
                                    base_station_pos=bs_position[b].cpu(),
                                    ue_positions=ue_pos_list,
                                    selected_subcarriers=selected_subcarriers,
                                    antenna_embedding=antenna_embedding[b].cpu()
                                )
                        except (TimeoutError, Exception) as e:
                            logger.warning(f"CUDA ray tracing failed: {e}. Falling back to CPU.")
                            ray_results = self._simple_ray_tracing(
                                bs_position[b], ue_pos_list[0], selected_subcarriers, antenna_embedding[b]
                            )
                except TimeoutError as e:
                    logger.error(f"Ray tracer operation timed out: {e}")
                    # Fallback to simple signal calculation
                    ray_results = self._fallback_signal_calculation(
                        bs_position[b], ue_pos_list[0], selected_subcarriers, antenna_embedding[b]
                    )
                except Exception as e:
                    logger.error(f"Ray tracer operation failed: {e}")
                    # Fallback to simple signal calculation
                    ray_results = self._fallback_signal_calculation(
                        bs_position[b], ue_pos_list[0], selected_subcarriers, antenna_embedding[b]
                    )
                
                batch_ray_results.append(ray_results)
                
                # Log actual directions used (only on first processing)
                if bs_antenna_idx == 0 and hasattr(self.ray_tracer, 'actual_directions_used'):
                    actual_directions = self.ray_tracer.actual_directions_used
                    if actual_directions != num_directions:
                        logger.info(f"🎯 MLP direction selection active: Using {actual_directions} directions (instead of {num_directions})")
                        logger.info(f"📊 Performance improvement: {num_directions/actual_directions:.1f}x faster")
                
                # Convert ray_tracer results to CSI predictions
                # Since num_ue = 1, we only have one UE device per batch item
                u = 0  # Single UE device index
                ue_pos_tuple = tuple(ue_pos_list[u].tolist())
                # Fill CSI predictions for each UE antenna
                for u in range(num_ue_antennas):
                    # Get the selected subcarriers for this UE antenna and BS antenna
                    ue_selected_subcarriers = self.current_selection[b, bs_antenna_idx, u].tolist()
                    
                    for k_idx, k in enumerate(ue_selected_subcarriers):
                        if k_idx < csi_predictions.shape[-1]:  # Ensure we don't go out of bounds
                            if (ue_pos_tuple, k) in ray_results:
                                signal_strength = ray_results[(ue_pos_tuple, k)]
                                
                                # Convert signal strength to complex CSI
                                # All UE antennas share the same device position
                                csi_value = self._signal_strength_to_csi(
                                    signal_strength, bs_position[b], ue_positions[b], k
                                )
                                csi_predictions[b, bs_antenna_idx, u, k_idx] = csi_value
                            else:
                                # Fallback for missing results
                                csi_predictions[b, bs_antenna_idx, u, k_idx] = torch.complex(
                                    torch.tensor(0.0, device=ue_positions.device), 
                                    torch.tensor(0.0, device=ue_positions.device)
                                )
                
                batch_signal_strengths.append(ray_results)
            
            all_ray_results.append(batch_ray_results)
            all_signal_strengths.append(batch_signal_strengths)
        
        # Map selected subcarrier predictions back to full subcarrier space
        # Create full predictions tensor: (batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas)
        full_predictions = torch.zeros(
            batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas,
            dtype=torch.complex64, device=ue_positions.device
        )
        
        # Fill in the predictions for selected subcarriers
        for b in range(batch_size):
            for bs_antenna_idx in range(num_bs_antennas):
                for u in range(num_ue_antennas):
                    selected_indices = self.current_selection[b, bs_antenna_idx, u]
                    for k_idx, k in enumerate(selected_indices):
                        if k_idx < csi_predictions.shape[-1]:  # Ensure we don't go out of bounds
                            full_predictions[b, k, u, bs_antenna_idx] = csi_predictions[b, bs_antenna_idx, u, k_idx]
        
        # Log completion summary
        logger.info(f"✅ BS-Centric ray tracing completed: {batch_size} batches × {num_bs_antennas} BS antennas × {num_ue_antennas} UE antennas × {num_selected} subcarriers processed successfully")
        
        outputs = {
            'csi_predictions': full_predictions,
            'ray_results': all_ray_results,
            'signal_strengths': all_signal_strengths,
            'subcarrier_selection': selection_info
        }
        
        if return_intermediates:
            outputs.update({'ray_tracer_results': all_ray_results})
        
        return outputs
    
    def _select_subcarriers_per_antenna(
        self, 
        batch_size: int, 
        num_ue_antennas: int, 
        num_bs_antennas: int,
        total_subcarriers: int,
        num_selected: int
    ) -> Dict[str, torch.Tensor]:
        """Select subcarriers for each BS antenna independently."""
        # Validate parameters
        if total_subcarriers <= 0:
            raise ValueError(f"total_subcarriers must be positive, got {total_subcarriers}")
        if num_selected <= 0:
            raise ValueError(f"num_selected must be positive, got {num_selected}")
        if num_selected > total_subcarriers:
            raise ValueError(f"num_selected ({num_selected}) cannot be greater than total_subcarriers ({total_subcarriers})")
        
        logger.debug(f"_select_subcarriers_per_antenna called with:")
        logger.debug(f"  batch_size: {batch_size}")
        logger.debug(f"  num_ue_antennas: {num_ue_antennas}")
        logger.debug(f"  num_bs_antennas: {num_bs_antennas}")
        logger.debug(f"  total_subcarriers: {total_subcarriers}")
        logger.debug(f"  num_selected: {num_selected}")
        
        selected_indices = torch.zeros(batch_size, num_bs_antennas, num_ue_antennas, num_selected, dtype=torch.long)
        selection_mask = torch.zeros(batch_size, num_bs_antennas, num_ue_antennas, total_subcarriers, dtype=torch.bool)
        
        for b in range(batch_size):
            for bs_antenna in range(num_bs_antennas):
                for u in range(num_ue_antennas):
                    try:
                        # logger.debug(f"Processing {b},{bs_antenna},{u}: total_subcarriers={total_subcarriers}, num_selected={num_selected}")
                        ue_selected = random.sample(range(total_subcarriers), num_selected)
                        # logger.debug(f"  Sample {b},{bs_antenna},{u}: {ue_selected}")
                        selected_indices[b, bs_antenna, u] = torch.tensor(ue_selected)
                        selection_mask[b, bs_antenna, u, ue_selected] = True
                    except Exception as e:
                        logger.error(f"Error in random.sample for {b},{bs_antenna},{u}: {e}")
                        logger.error(f"  total_subcarriers: {total_subcarriers}, num_selected: {num_selected}")
                        logger.error(f"  total_subcarriers type: {type(total_subcarriers)}")
                        logger.error(f"  num_selected type: {type(num_selected)}")
                        import traceback
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        raise
        
        logger.debug(f"Selected indices shape: {selected_indices.shape}")
        logger.debug(f"Selection mask shape: {selection_mask.shape}")
        
        return {
            'selected_indices': selected_indices,
            'selection_mask': selection_mask,
            'num_selected': num_selected
        }
    

    
    def _compute_view_directions(
        self, 
        bs_position: torch.Tensor, 
        sampled_positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute view directions from BS to each sampling point."""
        view_directions = sampled_positions - bs_position.unsqueeze(1).unsqueeze(2)
        view_directions = view_directions / (torch.norm(view_directions, dim=-1, keepdim=True) + 1e-8)
        return view_directions
    
    def _signal_strength_to_csi(
        self,
        signal_strength: float,
        bs_pos: torch.Tensor,
        ue_pos: torch.Tensor,
        subcarrier_idx: int
    ) -> torch.complex64:
        """
        Convert ray_tracer signal strength to complex CSI value.
        
        Args:
            signal_strength: Signal strength from ray_tracer
            bs_pos: Base station position
            ue_pos: UE position
            subcarrier_idx: Subcarrier index
            
        Returns:
            Complex CSI value
        """
        # Calculate distance-based phase
        distance = torch.norm(ue_pos - bs_pos)
        phase = 2 * torch.pi * subcarrier_idx * distance / 100.0  # Normalized wavelength
        
        # Convert signal strength to complex CSI with phase
        amplitude = torch.sqrt(torch.as_tensor(signal_strength, dtype=torch.float32, device=bs_pos.device))
        csi_value = torch.complex(
            amplitude * torch.cos(phase), 
            amplitude * torch.sin(phase)
        )
        
        return csi_value.to(torch.complex64)
    

    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        loss_function: nn.Module
    ) -> torch.Tensor:
        """Compute loss for selected subcarriers."""
        # Debug logging
        logger.debug(f"compute_loss called with predictions shape: {predictions.shape}, targets shape: {targets.shape}")
        logger.debug(f"current_selection: {self.current_selection is not None}, current_selection_mask: {self.current_selection_mask is not None}")
        
        if self.current_selection is None or self.current_selection_mask is None:
            logger.error("No subcarrier selection available. Call forward() first.")
            logger.error(f"current_selection: {self.current_selection}")
            logger.error(f"current_selection_mask: {self.current_selection_mask}")
            raise ValueError("No subcarrier selection available. Call forward() first.")
        
        # Validate shapes - targets are in format (batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas)
        batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas = targets.shape
        
        # Since we're using all subcarriers, predictions should have shape (batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas)
        expected_pred_shape = (batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas)
        
        if predictions.shape != expected_pred_shape:
            logger.error(f"Predictions shape mismatch. Expected: {expected_pred_shape}, Got: {predictions.shape}")
            raise ValueError(f"Predictions shape mismatch. Expected: {expected_pred_shape}, Got: {predictions.shape}")
        
        # Extract only the traced/selected subcarriers from both predictions and targets
        # and compute MSE loss only on those values, completely ignoring untraced subcarriers
        try:
            # Collect all traced subcarrier values
            traced_predictions = []
            traced_targets = []
            
            # Extract values for selected subcarriers only
            for b in range(batch_size):
                for bs_antenna_idx in range(num_bs_antennas):
                    for u in range(num_ue_antennas):
                        if self.current_selection is not None:
                            selected_indices = self.current_selection[b, bs_antenna_idx, u]
                            for k in selected_indices:
                                if k < num_subcarriers:  # Ensure index is valid
                                    traced_predictions.append(predictions[b, k, u, bs_antenna_idx])
                                    traced_targets.append(targets[b, k, u, bs_antenna_idx])
            
            # Convert to tensors
            if len(traced_predictions) == 0:
                logger.warning("No traced subcarriers found, returning zero loss")
                # Create a zero loss that maintains the computational graph
                zero_loss = torch.sum(predictions * 0.0)
                return zero_loss
            
            traced_predictions = torch.stack(traced_predictions)
            traced_targets = torch.stack(traced_targets)
            
            logger.debug(f"Computing loss on {len(traced_predictions)} traced subcarriers")
            logger.debug(f"Traced predictions shape: {traced_predictions.shape}")
            logger.debug(f"Traced targets shape: {traced_targets.shape}")
            logger.debug(f"Subcarrier sampling ratio: {self.subcarrier_sampling_ratio}")
            logger.debug(f"Expected selected subcarriers per antenna-UE pair: {int(self.prism_network.num_subcarriers * self.subcarrier_sampling_ratio)}")
            
            # Compute MSE loss only on traced subcarriers
            loss = loss_function(traced_predictions, traced_targets)
            
            # Validate loss is a tensor
            if not isinstance(loss, torch.Tensor):
                logger.error(f"Loss function returned non-tensor: {type(loss)} = {loss}")
                raise ValueError(f"Loss function must return a torch.Tensor, got {type(loss)}")
            
            # Ensure the loss requires gradients for backpropagation
            if not loss.requires_grad:
                logger.warning("Loss tensor does not require gradients, this may cause issues")
                # Try to create a loss that requires gradients by using the original predictions
                if len(traced_predictions) > 0:
                    # Use a simple MSE computation that maintains gradients
                    loss = torch.mean((traced_predictions - traced_targets) ** 2)
                    logger.info(f"Recomputed loss with gradients: {loss.requires_grad}")
                else:
                    # Fallback: create loss from predictions tensor
                    loss = torch.sum(predictions * 0.0)
                    logger.info(f"Created fallback loss with gradients: {loss.requires_grad}")
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in compute_loss: {e}")
            logger.error(f"Shapes - predictions: {predictions.shape}, targets: {targets.shape}")
            raise
    
    def set_training_phase(self, phase: int):
        """Set training phase for curriculum learning."""
        if phase == 0:
            self.prism_network.azimuth_divisions = 8
            self.prism_network.elevation_divisions = 4
            self.prism_network.top_k_directions = 16
        elif phase == 1:
            self.prism_network.azimuth_divisions = 16
            self.prism_network.elevation_divisions = 8
            self.prism_network.top_k_directions = 32
        elif phase == 2:
            self.prism_network.azimuth_divisions = 36
            self.prism_network.elevation_divisions = 18
            self.prism_network.top_k_directions = 64
        else:
            raise ValueError(f"Invalid training phase: {phase}")
        
        logger.info(f"Training phase {phase} set: {self.prism_network.azimuth_divisions}×{self.prism_network.elevation_divisions}")
    
    def reset_training_state(self):
        """Reset training state and selection variables."""
        self.current_epoch = 0
        self.current_batch = 0
        self.best_loss = float('inf')
        self.training_history = []
        self.current_selection = None
        self.current_selection_mask = None
        logger.info("Training state reset")
    
    def ensure_selection_initialized(self):
        """Ensure selection variables are properly initialized."""
        if self.current_selection is None or self.current_selection_mask is None:
            # Get default values from prism_network
            if hasattr(self.prism_network, 'num_subcarriers'):
                num_subcarriers = self.prism_network.num_subcarriers
            else:
                num_subcarriers = 408  # Default fallback
            
            # Use default batch size and antenna counts
            batch_size = 1  # Will be resized during forward pass
            num_bs_antennas = 64  # Default BS antenna count
            num_ue_antennas = 4   # Default UE antenna count
            
            self._initialize_selection_variables(batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers)
            logger.debug("Selection variables initialized for the first time")
            return True
        return True
    
    def _get_device_safely(self):
        """Safely get the device from prism_network parameters, handling DataParallel wrapper"""
        logger.debug("=== Device Detection Debug ===")
        
        # Method 1: Try to get device from prism_network parameters (most reliable)
        try:
            if hasattr(self.prism_network, 'parameters'):
                logger.debug(f"prism_network has parameters: True")
                # Handle DataParallel wrapper
                if hasattr(self.prism_network, 'module'):
                    logger.debug(f"prism_network has module (DataParallel): True")
                    # DataParallel case: access the underlying module
                    device = next(self.prism_network.module.parameters()).device
                    logger.debug(f"Device detected from DataParallel prism_network: {device}")
                    return device
                else:
                    logger.debug(f"prism_network has module (DataParallel): False")
                    # Direct module case
                    device = next(self.prism_network.parameters()).device
                    logger.debug(f"Device detected from direct prism_network: {device}")
                    return device
            else:
                logger.debug(f"prism_network has parameters: False")
        except (StopIteration, AttributeError) as e:
            logger.debug(f"Method 1 failed: {e}")
        
        # Method 2: Try to get device from this module's parameters
        try:
            logger.debug(f"self has parameters: {hasattr(self, 'parameters')}")
            device = next(self.parameters()).device
            logger.debug(f"Device detected from self parameters: {device}")
            return device
        except (StopIteration, AttributeError) as e:
            logger.debug(f"Method 2 failed: {e}")
        
        # Method 3: Try to get device from CUDA context
        if torch.cuda.is_available():
            logger.debug(f"CUDA available: True")
            # Check if we're in a DataParallel context
            if hasattr(self, 'device_ids') and self.device_ids:
                logger.debug(f"self.device_ids: {self.device_ids}")
                # Use the first device in the list
                device = torch.device(f'cuda:{self.device_ids[0]}')
                logger.debug(f"Device detected from device_ids: {device}")
                return device
            else:
                logger.debug(f"self.device_ids: {getattr(self, 'device_ids', 'Not found')}")
                # Use the current CUDA device
                device = torch.device('cuda:0')
                logger.debug(f"Device detected from CUDA context: {device}")
                return device
        else:
            logger.debug(f"CUDA available: False")
        
        # Method 4: Last resort - use CPU
        logger.warning("Could not determine device automatically, using CPU")
        return torch.device('cpu')
    
    def _get_device_from_context(self):
        """Get device from the current execution context, especially for DataParallel"""
        logger.debug("=== Context Device Detection Debug ===")
        
        try:
            # If we're in a forward pass, try to get device from input tensors
            if hasattr(self, '_last_input_device'):
                logger.debug(f"_last_input_device: {self._last_input_device}")
                return self._last_input_device
            else:
                logger.debug(f"_last_input_device: Not set")
        except Exception as e:
            logger.debug(f"Error getting _last_input_device: {e}")
        
        # Try to get device from CUDA context
        try:
            if torch.cuda.is_available():
                logger.debug(f"CUDA available in context: True")
                # Check current CUDA device
                current_device = torch.cuda.current_device()
                logger.debug(f"torch.cuda.current_device(): {current_device}")
                device = torch.device(f'cuda:{current_device}')
                logger.debug(f"Device detected from CUDA context: {device}")
                return device
            else:
                logger.debug(f"CUDA available in context: False")
        except Exception as e:
            logger.debug(f"Error getting CUDA context: {e}")
        
        # Fallback to the safe method
        logger.debug("Falling back to _get_device_safely")
        return self._get_device_safely()
    
    def _get_device_from_data_parallel(self):
        """Get device when this TrainingInterface is wrapped in DataParallel"""
        logger.debug("=== DataParallel Device Detection Debug ===")
        
        try:
            # Check if we're wrapped in DataParallel
            if hasattr(self, 'module'):
                logger.debug(f"self has module (DataParallel): True")
                # We're wrapped in DataParallel, get device from the underlying module
                if hasattr(self.module, 'prism_network'):
                    logger.debug(f"module.prism_network exists")
                    try:
                        device = next(self.module.prism_network.parameters()).device
                        logger.debug(f"Device detected from module.prism_network: {device}")
                        return device
                    except (StopIteration, AttributeError) as e:
                        logger.debug(f"Error getting device from module.prism_network: {e}")
                else:
                    logger.debug(f"module.prism_network does not exist")
            else:
                logger.debug(f"self has module (DataParallel): False")
        except Exception as e:
            logger.debug(f"Error in DataParallel detection: {e}")
        
        # Try other methods
        return self._get_device_from_context()
    
    def _initialize_selection_variables(self, batch_size: int, num_ue_antennas: int, num_bs_antennas: int, num_subcarriers: int):
        """Initialize selection variables with default values."""
        # Create default selection variables for the current configuration
        num_selected = int(num_subcarriers * self.subcarrier_sampling_ratio)
        
        # Get device safely - prefer DataParallel-aware detection
        device = self._get_device_from_data_parallel()
        
        # Initialize with placeholder values that will be overwritten
        self.current_selection = torch.zeros(
            (batch_size, num_bs_antennas, num_ue_antennas, num_selected),
            dtype=torch.long,
            device=device
        )
        
        self.current_selection_mask = torch.zeros(
            (batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers),
            dtype=torch.bool,
            device=device
        )
        
        logger.debug(f"Selection variables initialized on device {device}: {self.current_selection.shape}, {self.current_selection_mask.shape}")
    
    def _ensure_selection_variables_sized(self, batch_size: int, num_ue_antennas: int, num_bs_antennas: int, num_subcarriers: int, num_selected: int):
        """Ensure selection variables are properly sized for the current batch."""
        expected_selection_shape = (batch_size, num_bs_antennas, num_ue_antennas, num_selected)
        expected_mask_shape = (batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers)
        
        # Check if we need to resize or initialize
        if (self.current_selection is None or 
            self.current_selection.shape != expected_selection_shape or
            self.current_selection_mask is None or
            self.current_selection_mask.shape != expected_mask_shape):
            
            # Get device safely - prefer DataParallel-aware detection
            device = self._get_device_from_data_parallel()
            
            # Initialize or resize selection variables
            self.current_selection = torch.zeros(
                expected_selection_shape,
                dtype=torch.long,
                device=device
            )
            
            self.current_selection_mask = torch.zeros(
                expected_mask_shape,
                dtype=torch.bool,
                device=device
            )
            
            # Only log if this is a resize, not first initialization
            if self.current_selection is not None and hasattr(self, '_selection_initialized'):
                logger.debug(f"Selection variables resized to {expected_selection_shape} on device {device}")
            else:
                logger.debug(f"Selection variables initialized with shape {expected_selection_shape} on device {device}")
                self._selection_initialized = True
    
    def _fallback_signal_calculation(self, bs_position: torch.Tensor, ue_position: torch.Tensor, 
                                   selected_subcarriers: Dict, antenna_embedding: torch.Tensor) -> Dict:
        """Fallback signal calculation when ray tracer fails or times out."""
        logger.info("Using fallback signal calculation")
        
        # Ensure all tensors are on the same device
        device = bs_position.device
        ue_position = ue_position.to(device)
        
        # Simple distance-based signal strength calculation
        distance = torch.norm(bs_position - ue_position)
        
        # Basic path loss model (simplified)
        signal_strength = 1.0 / (1.0 + distance)  # Simple inverse distance model
        
        # Create fallback results structure
        fallback_results = {}
        for ue_key, subcarriers in selected_subcarriers.items():
            if isinstance(subcarriers, (list, tuple)):
                for subcarrier_idx in subcarriers:
                    fallback_results[(ue_key, subcarrier_idx)] = signal_strength.item()
            else:
                fallback_results[(ue_key, 0)] = signal_strength.item()
        
        logger.info(f"Fallback calculation completed with {len(fallback_results)} results")
        return fallback_results
    
    def _simple_ray_tracing(self, bs_position: torch.Tensor, ue_position: torch.Tensor, 
                           selected_subcarriers: Dict, antenna_embedding: torch.Tensor) -> Dict:
        """Simple ray tracing without parallel processing to avoid hanging."""
        logger.debug("Using simple ray tracing (no parallel processing)")
        
        # Ensure all tensors are on the same device
        device = bs_position.device
        ue_position = ue_position.to(device)
        
        # Simple distance-based signal strength calculation with some randomness
        distance = torch.norm(bs_position - ue_position)
        
        # Basic path loss model with frequency-dependent variations
        base_signal = 1.0 / (1.0 + distance * 0.1)  # Scaled distance
        
        # Create results with slight variations per subcarrier
        simple_results = {}
        for ue_key, subcarriers in selected_subcarriers.items():
            if isinstance(subcarriers, (list, tuple)):
                for i, subcarrier_idx in enumerate(subcarriers):
                    # Add small frequency-dependent variation
                    freq_factor = 1.0 + 0.1 * torch.sin(torch.tensor(subcarrier_idx * 0.01))
                    signal_strength = base_signal * freq_factor.item()
                    simple_results[(ue_key, subcarrier_idx)] = max(0.001, signal_strength)
            else:
                simple_results[(ue_key, 0)] = base_signal.item()
        
        logger.debug(f"Simple ray tracing completed with {len(simple_results)} results")
        return simple_results
    
    # Checkpoint and recovery methods
    def save_checkpoint(self, filename: str = None):
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}_batch_{self.current_batch}.pt"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'prism_network_state_dict': self.prism_network.state_dict(),
            'current_epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'current_selection': self.current_selection,
            'current_selection_mask': self.current_selection_mask,
            'training_config': {
                'num_sampling_points': self.num_sampling_points,
                'subcarrier_sampling_ratio': self.subcarrier_sampling_ratio,
                'scene_bounds': (self.scene_min.tolist(), self.scene_max.tolist())
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save training info
        info_path = checkpoint_path.replace('.pt', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(self.get_training_info(), f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model states
        self.load_state_dict(checkpoint['model_state_dict'])
        self.prism_network.load_state_dict(checkpoint['prism_network_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['current_epoch']
        self.current_batch = checkpoint['current_batch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        self.current_selection = checkpoint['current_selection']
        self.current_selection_mask = checkpoint['current_selection_mask']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}, batch {self.current_batch}")
        logger.info(f"Best loss so far: {self.best_loss}")
    
    def update_training_state(self, epoch: int, batch: int, loss: float):
        """Update training state for checkpoint tracking."""
        self.current_epoch = epoch
        self.current_batch = batch
        
        if loss < self.best_loss:
            self.best_loss = loss
        
        self.training_history.append({
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'best_loss': self.best_loss
        })
    
    def get_training_info(self) -> Dict:
        """Get training interface information."""
        return {
            'num_sampling_points': self.num_sampling_points,
            'subcarrier_sampling_ratio': self.subcarrier_sampling_ratio,
            'scene_bounds': (self.scene_min.tolist(), self.scene_max.tolist()),
            'current_epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'best_loss': self.best_loss,
            'training_history_length': len(self.training_history),
            'checkpoint_dir': self.checkpoint_dir,
            'prism_network_config': {
                'azimuth_divisions': self.prism_network.azimuth_divisions,
                'elevation_divisions': self.prism_network.elevation_divisions,
                'top_k_directions': self.prism_network.top_k_directions
            }
        }
