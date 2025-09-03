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
from typing import Dict, Optional, Tuple
import logging
import random
import os
import json
import time
import sys
import numpy as np


from .networks.prism_network import PrismNetwork
from .ray_tracer_cpu import CPURayTracer

logger = logging.getLogger(__name__)
# Note: Logger configuration is handled by the main training script
# No need to add handlers here to avoid duplicate logging



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
        ray_tracing_config: Optional[dict] = None,
        system_config: Optional[dict] = None,
        user_equipment_config: Optional[dict] = None,
        checkpoint_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        # Store configuration
        self.ray_tracing_config = ray_tracing_config or {}
        self.system_config = system_config or {}
        self.user_equipment_config = user_equipment_config or {}
        
        # Extract target antenna index from user equipment config
        self.target_antenna_index = self.user_equipment_config.get('target_antenna_index', 0)
        logger.info(f"🎯 Target UE antenna index: {self.target_antenna_index}")
        
        # Set device first - needed for ray tracer creation
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Set logger level from system config - use fallback if missing
        log_level = self.system_config.get('logging', {}).get('log_level', 'INFO')
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Set prism_network first - needed for ray tracer creation
        self.prism_network = prism_network
        
        # Get ray tracing mode from system config and create ray tracer
        try:
            ray_tracing_mode = self.system_config['ray_tracing_mode']
            # Validate ray tracing mode
            if ray_tracing_mode not in ['cuda', 'cpu', 'hybrid']:
                raise ValueError(f"Invalid ray_tracing_mode: {ray_tracing_mode}. Must be 'cuda', 'cpu', or 'hybrid'")
            # Set ray tracing mode
            self.ray_tracing_mode = ray_tracing_mode
            
            # Create ray tracer based on mode
            self.ray_tracer = self._create_ray_tracer_by_mode(ray_tracing_mode)
        except KeyError:
            logger.error("❌ FATAL: Missing required config 'system.ray_tracing_mode'")
            logger.error("   Please check your configuration file.")
            sys.exit(1)
        except Exception as e:
            error_msg = f"❌ FATAL: Failed to create ray tracer: {e}"
            logger.error(error_msg)
            logger.error(f"   Mode: {ray_tracing_mode}")
            print(error_msg)
            print(f"   Mode: {ray_tracing_mode}")
            sys.exit(1)
        
        # Get configuration parameters from config dictionaries - fail fast if missing
        try:
            radial_sampling = self.ray_tracing_config['radial_sampling']
        except KeyError:
            error_msg = "❌ FATAL: Missing required config 'ray_tracing.radial_sampling'"
            logger.error(error_msg)
            logger.error("   Please check your configuration file.")
            print(error_msg)
            print("   Please check your configuration file.")
            sys.exit(1)
        
        try:
            subcarrier_sampling = self.ray_tracing_config['subcarrier_sampling']
        except KeyError:
            error_msg = "❌ FATAL: Missing required config 'ray_tracing.subcarrier_sampling'"
            logger.error(error_msg)
            logger.error("   Please check your configuration file.")
            print(error_msg)
            print("   Please check your configuration file.")
            sys.exit(1)
        
        # Store configuration parameters - fail fast if missing
        try:
            self.num_sampling_points = radial_sampling['num_sampling_points']
        except KeyError:
            error_msg = "❌ FATAL: Missing required config 'ray_tracing.radial_sampling.num_sampling_points'"
            logger.error(error_msg)
            print(error_msg)
            raise ValueError("Missing required configuration: ray_tracing.radial_sampling.num_sampling_points")
        
        try:
            self.subcarrier_sampling_ratio = subcarrier_sampling['sampling_ratio']
        except KeyError:
            error_msg = "❌ FATAL: Missing required config 'ray_tracing.subcarrier_sampling.sampling_ratio'"
            logger.error(error_msg)
            print(error_msg)
            raise ValueError("Missing required configuration: ray_tracing.subcarrier_sampling.sampling_ratio")
        
        # Required parameters - fail fast if missing
        try:
            self.subcarrier_sampling_method = subcarrier_sampling['sampling_method']
        except KeyError:
            error_msg = "❌ FATAL: Missing required config 'ray_tracing.subcarrier_sampling.sampling_method'"
            logger.error(error_msg)
            logger.error("   Please check your configuration file.")
            print(error_msg)
            print("   Please check your configuration file.")
            sys.exit(1)
        
        try:
            self.antenna_consistent = subcarrier_sampling['antenna_consistent']
        except KeyError:
            error_msg = "❌ FATAL: Missing required config 'ray_tracing.subcarrier_sampling.antenna_consistent'"
            logger.error(error_msg)
            logger.error("   Please check your configuration file.")
            print(error_msg)
            print("   Please check your configuration file.")
            sys.exit(1)
        
        # Validate additional ray tracing configuration parameters needed for tracer creation
        try:
            angular_sampling = self.ray_tracing_config['angular_sampling']
            self.ray_tracing_config['max_ray_length']
            self.ray_tracing_config['signal_threshold']
            self.ray_tracing_config['enable_early_termination']
        except KeyError as e:
            error_msg = f"❌ FATAL: Missing required config 'ray_tracing.{e.args[0]}'"
            logger.error(error_msg)
            logger.error("   Please check your configuration file.")
            print(error_msg)
            print("   Please check your configuration file.")
            sys.exit(1)
        
        # Validate system configuration parameters needed for tracer creation
        try:
            self.system_config['mixed_precision']
            self.system_config['cpu']
        except KeyError as e:
            error_msg = f"❌ FATAL: Missing required config 'system.{e.args[0]}'"
            logger.error(error_msg)
            logger.error("   Please check your configuration file.")
            print(error_msg)
            print("   Please check your configuration file.")
            sys.exit(1)
        
        # Calculate and log training ray count if configuration is available
        self._log_training_ray_count()

        # Initialize checkpoint directory
        if not checkpoint_dir or not checkpoint_dir.strip():
            error_msg = "❌ FATAL: checkpoint_dir not provided"
            logger.error(error_msg)
            print(error_msg)
            sys.exit(1)
        
        self.checkpoint_dir = checkpoint_dir.strip()
        logger.info(f"✅ Checkpoint directory: {self.checkpoint_dir}")
        
        logger.info(f"Training interface initialized with ray_tracing_mode: {ray_tracing_mode}")
        
        # Set scene bounds from ray_tracing_config
        try:
            scene_bounds_config = self.ray_tracing_config['scene_bounds']
            self.scene_min = torch.tensor(scene_bounds_config['min'], dtype=torch.float32)
            self.scene_max = torch.tensor(scene_bounds_config['max'], dtype=torch.float32)
        except KeyError as e:
            logger.error(f"❌ FATAL: Missing required config 'ray_tracing.scene_bounds.{e.args[0] if e.args else 'scene_bounds'}'")
            sys.exit(1)
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state for checkpoint recovery
        self.current_epoch = 0
        self.current_batch = 0
        self.best_loss = float('inf')
        self.training_history = []
        self.current_selection = None
        self.current_selection_mask = None
    
    def _create_ray_tracer_by_mode(self, ray_tracing_mode: str):
        """
        Create appropriate ray tracer based on the specified mode.
        
        Args:
            ray_tracing_mode: 'cuda', 'cpu', or 'hybrid'
            
        Returns:
            Appropriate ray tracer instance
        """
        # Extract common configuration parameters
        try:
            angular_sampling = self.ray_tracing_config['angular_sampling']
            radial_sampling = self.ray_tracing_config['radial_sampling']
            mixed_precision = self.system_config['mixed_precision']
            cpu_config = self.system_config['cpu']
        except KeyError as e:
            logger.error(f"❌ FATAL: Missing required config section: {e.args[0]}")
            sys.exit(1)
        
        # Common parameters for both ray tracers
        try:
            # Get max_ray_length with fallback to scene bounds calculation
            max_ray_length = self.ray_tracing_config.get('max_ray_length')
            if max_ray_length is None:
                # Calculate from scene bounds if not provided
                scene_bounds = self.ray_tracing_config['scene_bounds']
                min_bounds = np.array(scene_bounds['min'])
                max_bounds = np.array(scene_bounds['max'])
                diagonal = np.linalg.norm(max_bounds - min_bounds)
                max_ray_length = diagonal * 1.2  # Add 20% margin
                logger.info(f"📏 Calculated max_ray_length: {max_ray_length:.1f}m from scene bounds")
            
            common_params = {
                'azimuth_divisions': angular_sampling['azimuth_divisions'],
                'elevation_divisions': angular_sampling['elevation_divisions'],
                'max_ray_length': max_ray_length,
                'scene_bounds': self.ray_tracing_config['scene_bounds'],
                'prism_network': self.prism_network,
                'signal_threshold': self.ray_tracing_config['signal_threshold'],
                'enable_early_termination': self.ray_tracing_config['enable_early_termination'],
                'top_k_directions': angular_sampling['top_k_directions'],
                'uniform_samples': radial_sampling['num_sampling_points'],
                'resampled_points': radial_sampling['resampled_points']
            }
        except KeyError as e:
            logger.error(f"❌ FATAL: Missing required config parameter: {e.args[0]}")
            sys.exit(1)
        
        def create_cuda_ray_tracer():
            """Helper function to create CUDA ray tracer."""
            from .ray_tracer_cuda import CUDARayTracer
            try:
                use_mixed_precision = mixed_precision['enabled']
            except KeyError:
                logger.error("❌ FATAL: Missing required config 'system.mixed_precision.enabled'")
                logger.error("   Please check your configuration file.")
                sys.exit(1)
            
            # Get GPU memory fraction from system config
            gpu_memory_fraction = self.system_config.get('gpu_memory_fraction', 0.6)
            
            # Create CUDA-specific parameters
            cuda_params = {
                'use_mixed_precision': use_mixed_precision,
                'gpu_memory_fraction': gpu_memory_fraction,
                'device': str(self.device),  # Pass the specific CUDA device
                'prism_network': self.prism_network,  # Pass the neural network
                **common_params
            }
            
            logger.info(f"🚀 Creating CUDARayTracer with {len(cuda_params)} parameters")
            logger.info(f"   - prism_network: {type(self.prism_network).__name__}")
            logger.info(f"   - direction batch size: auto-calculated based on hardware")
            return CUDARayTracer(**cuda_params)
        
        def create_cpu_ray_tracer():
            """Helper function to create CPU ray tracer."""
            from .ray_tracer_cpu import CPURayTracer
            try:
                max_workers = cpu_config['num_workers']
            except KeyError:
                logger.error("❌ FATAL: Missing required config 'system.cpu.num_workers'")
                logger.error("   Please check your configuration file.")
                sys.exit(1)
            
            # Create CPU-specific parameters
            cpu_params = {
                'max_workers': max_workers,
                'prism_network': self.prism_network,  # Pass the neural network
                **common_params
            }
            
            logger.info(f"💻 Creating CPURayTracer with {len(cpu_params)} parameters")
            logger.info(f"   - prism_network: {type(self.prism_network).__name__}")
            return CPURayTracer(**cpu_params)
        
        # Create ray tracer based on mode
        if ray_tracing_mode == 'cuda':
            logger.info("🚀 Creating CUDARayTracer for CUDA mode")
            try:
                return create_cuda_ray_tracer()
            except Exception as e:
                logger.error(f"Failed to create CUDARayTracer: {e}. Falling back to CPURayTracer.")
                raise RuntimeError(f"Failed to create CUDARayTracer: {e}")
        
        elif ray_tracing_mode == 'cpu':
            logger.info("💻 Creating CPURayTracer for CPU mode")
            return create_cpu_ray_tracer()
        
        else:  # hybrid mode
            logger.info("🔄 Creating ray tracer for hybrid mode (CUDA first, CPU fallback)")
            try:
                return create_cuda_ray_tracer()
            except Exception as e:
                logger.error(f"CUDA ray tracer failed in hybrid mode: {e}. No fallback will be used.")
                raise RuntimeError(f"Failed to create CUDA ray tracer in hybrid mode: {e}")
    
 
    
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
        device = ue_positions.device  # Use input device directly
        
        # Extract dimensions
        batch_size = ue_positions.shape[0]
        # Single UE antenna processing (always 1 after data extraction)
        num_ue_antennas = 1
        num_bs_antennas = antenna_indices.shape[1]
        num_subcarriers = self.prism_network.num_subcarriers
        num_selected = int(num_subcarriers * self.subcarrier_sampling_ratio)
        
        logger.info(f"🚀 Forward pass: {batch_size} samples × {num_bs_antennas} BS antennas × {num_ue_antennas} UE antennas × {num_selected} selected subcarriers")
        
        # Step 1: Clean up previous tensors to prevent memory leaks
        if hasattr(self, 'current_selection'):
            del self.current_selection
        if hasattr(self, 'current_selection_mask'):
            del self.current_selection_mask
        torch.cuda.empty_cache()
        
        # Initialize current_selection attributes directly to ensure they exist
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
        
        # Step 2: Generate actual subcarrier selection
        selection_info = self._select_subcarriers(batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers, num_selected)
        self.current_selection = selection_info['selected_indices']
        self.current_selection_mask = selection_info['selection_mask']
        
        # Validate selection variables
        self._validate_selection_variables(batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers, num_selected)
        
        # Step 2: Initialize CSI predictions tensor
        csi_predictions = torch.zeros(
            batch_size, num_bs_antennas, num_ue_antennas, num_selected, 
            dtype=torch.complex64, device=ue_positions.device
        )
        
        # Step 3: Process all BS antennas using unified interface
        print(f"    🔄 Processing {num_bs_antennas} BS antennas...")
        logger.info(f"📡 Processing {num_bs_antennas} BS antennas using unified ray tracer interface")
        
        # Process all batches - ray tracer will automatically detect single vs multi-antenna
        all_ray_results = []
        all_signals = []
        
        for b in range(batch_size):
            if batch_size > 1:
                batch_progress = (b / batch_size) * 100
                print(f"        🔹 Processing batch {b+1}/{batch_size} ({batch_progress:.0f}%)")
            
            # Prepare UE positions (all antennas share the same device position)
            ue_device_pos = ue_positions[b].cpu()
            ue_pos_list = [ue_device_pos] * num_ue_antennas
            
            # Create subcarrier dictionary (same for all antennas due to antenna_consistent=True)
            selected_subcarriers = self._create_subcarrier_dict_unified(b, num_ue_antennas, ue_pos_list)
            
            # Perform ray tracing - ray tracer will detect multi-antenna automatically
            print(f"        ⚡ Ray tracing for batch {b+1}...")
            print(f"           📡 Processing {len(ue_pos_list)} UEs with {num_bs_antennas} BS antennas")
            
            # Log UE positions for this batch
            for i, ue_pos in enumerate(ue_pos_list):
                if isinstance(ue_pos, torch.Tensor):
                    coords = ue_pos.cpu().numpy() if ue_pos.is_cuda else ue_pos.numpy()
                    print(f"           📍 UE {i+1}: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}]")
            
            # Call CUDA ray tracer interface
            ray_results = self.ray_tracer.accumulate_signals(
                base_station_pos=bs_position,
                ue_positions=ue_pos_list,
                selected_subcarriers=selected_subcarriers,
                antenna_indices=antenna_indices[b]  # All antenna indices for this batch
            )
            
            # Log ray tracing results
            if num_bs_antennas > 1:
                logger.info(f"🔍 Multi-antenna ray tracing completed with results for {num_bs_antennas} antennas")
            else:
                logger.info(f"🔍 Single-antenna ray tracing completed")
            
            all_ray_results.append(ray_results)
            all_signals.append(ray_results)
        
        # Update CSI predictions using unified format
        self._update_csi_predictions_unified(
            csi_predictions, all_ray_results, 
            batch_size, num_bs_antennas, num_ue_antennas, num_selected, ue_positions
        )
        
        # Step 4: Map selected subcarrier predictions back to full subcarrier space
        full_predictions = self._create_full_predictions(
            csi_predictions, batch_size, num_bs_antennas, num_ue_antennas, 
            num_subcarriers, ue_positions.device
        )
        
        logger.info(f"✅ Forward pass completed successfully")
        
        # Prepare outputs
        outputs = {
            'csi_predictions': full_predictions,
            'ray_results': all_ray_results,
            'signals': all_signals,
            'subcarrier_selection': selection_info
        }
        
        if return_intermediates:
            outputs['ray_tracer_results'] = all_ray_results
        
        return outputs
    
    def _validate_selection_variables(self, batch_size: int, num_bs_antennas: int, num_ue_antennas: int, num_subcarriers: int, num_selected: int):
        """Validate selection variables have correct shapes."""
        if self.current_selection is None or self.current_selection_mask is None:
            raise ValueError("Failed to initialize subcarrier selection variables")
        
        expected_selection_shape = (batch_size, num_bs_antennas, num_ue_antennas, num_selected)
        if self.current_selection.shape != expected_selection_shape:
            raise ValueError(f"Selection shape mismatch. Expected: {expected_selection_shape}, Got: {self.current_selection.shape}")
        
        expected_mask_shape = (batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers)
        if self.current_selection_mask.shape != expected_mask_shape:
            raise ValueError(f"Selection mask shape mismatch. Expected: {expected_mask_shape}, Got: {self.current_selection_mask.shape}")
    
    def _process_antenna_batches(self, bs_antenna_idx: int, batch_size: int, num_ue_antennas: int, num_selected: int,
                                ue_positions: torch.Tensor, bs_position: torch.Tensor, current_antenna_indices: torch.Tensor):
        """Process all batches for a specific BS antenna."""
        batch_ray_results = []
        batch_signals = []
        
        for b in range(batch_size):
            if batch_size > 1 and b % max(1, batch_size // 4) == 0:  # Show progress for larger batches
                batch_progress = (b / batch_size) * 100
                print(f"        🔹 UE Sub-batch {b+1}/{batch_size} ({batch_progress:.0f}%)")
            
            # Prepare UE positions (all antennas share the same device position)
            ue_device_pos = ue_positions[b].cpu()
            ue_pos_list = [ue_device_pos] * num_ue_antennas
            
            # Create subcarrier dictionary
            selected_subcarriers = self._create_subcarrier_dict(b, bs_antenna_idx, num_ue_antennas, ue_pos_list)
            
            # Perform ray tracing with detailed logging
            num_subcarriers = sum(len(v) if hasattr(v, '__len__') else 1 for v in selected_subcarriers.values())
            print(f"        ⚡ Ray tracing for UE sub-batch {b+1}...")
            print(f"           📡 Processing {len(ue_pos_list)} UEs with {num_subcarriers} subcarriers")
            
            # Log UE positions for this sub-batch
            for i, ue_pos in enumerate(ue_pos_list):
                if isinstance(ue_pos, torch.Tensor):
                    coords = ue_pos.cpu().numpy() if ue_pos.is_cuda else ue_pos.numpy()
                    print(f"           📍 UE {i+1}: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}]")
            
            # Log selected subcarriers structure
            logger.debug(f"🔍 Selected subcarriers structure:")
            for ue_key, subcarriers in selected_subcarriers.items():
                logger.debug(f"   - UE {ue_key}: {subcarriers}")
            
            ray_results = self._perform_ray_tracing(b, bs_position, ue_pos_list, selected_subcarriers, current_antenna_indices)
            
            # Log ray tracing results
            logger.info(f"🔍 Ray tracing completed with {len(ray_results)} results")
            logger.debug(f"🔍 Result keys: {list(ray_results.keys())[:10]}...")
            
            batch_ray_results.append(ray_results)
            batch_signals.append(ray_results)
        
        return batch_ray_results, batch_signals
    
    def _create_subcarrier_dict_unified(self, batch_idx: int, num_ue_antennas: int, ue_pos_list: list):
        """Create subcarrier dictionary for unified processing (same for all antennas)."""
        selected_subcarriers = {}
        
        # Use the first antenna's selection (they should all be the same due to antenna_consistent=True)
        for u in range(num_ue_antennas):
            ue_pos_tuple = tuple(ue_pos_list[u].tolist())
            selection_tensor = self.current_selection[batch_idx, 0, u]  # Use antenna 0 as reference
            
            # Convert tensor to list safely
            try:
                if selection_tensor.numel() == 1:
                    selected_subcarriers[ue_pos_tuple] = [int(selection_tensor.item())]
                else:
                    tensor_list = selection_tensor.tolist()
                    if isinstance(tensor_list, (list, tuple)):
                        selected_subcarriers[ue_pos_tuple] = [int(idx) for idx in tensor_list]
                    else:
                        logger.warning(f"⚠️ Fallback: Using default subcarrier [0] for UE at position {ue_pos_tuple}")
                        selected_subcarriers[ue_pos_tuple] = [0]  # Fallback
            except Exception as e:
                logger.warning(f"Error converting subcarrier selection: {e}, using fallback")
                selected_subcarriers[ue_pos_tuple] = [0]  # Fallback
            
            # Ensure we have valid subcarriers
            if not selected_subcarriers[ue_pos_tuple]:
                logger.warning(f"⚠️ Using fallback subcarrier [0] for UE at position {ue_pos_tuple}")
                selected_subcarriers[ue_pos_tuple] = [0]
        
        return selected_subcarriers
    
    def _update_csi_predictions_unified(self, csi_predictions: torch.Tensor, all_ray_results: list,
                                      batch_size: int, num_bs_antennas: int, num_ue_antennas: int, 
                                      num_selected: int, ue_positions: torch.Tensor):
        """Update CSI predictions using unified ray tracer results."""
        for b, ray_results in enumerate(all_ray_results):
            if isinstance(ray_results, dict):
                if num_bs_antennas == 1:
                    # Single antenna: results are in standard format
                    self._update_single_antenna_csi_unified(
                        csi_predictions, ray_results, b, 0,  # antenna_idx=0 for single antenna
                        num_ue_antennas, num_selected, ue_positions
                    )
                else:
                    # Multiple antennas: results have antenna keys
                    for antenna_idx in range(num_bs_antennas):
                        antenna_key = f"antenna_{antenna_idx}"
                        if antenna_key in ray_results:
                            antenna_results = ray_results[antenna_key]
                            self._update_single_antenna_csi_unified(
                                csi_predictions, antenna_results, b, antenna_idx,
                                num_ue_antennas, num_selected, ue_positions
                            )
                        else:
                            logger.warning(f"⚠️ Missing results for {antenna_key} in batch {b}")
            else:
                logger.warning(f"⚠️ Unexpected ray_results type for batch {b}: {type(ray_results)}")
    
    def _update_single_antenna_csi_unified(self, csi_predictions: torch.Tensor, ray_results: dict, 
                                         batch_idx: int, antenna_idx: int, num_ue_antennas: int, 
                                         num_selected: int, ue_positions: torch.Tensor):
        """Update CSI predictions for a single antenna from unified results."""
        ue_device_pos = ue_positions[batch_idx].cpu()
        
        for u in range(num_ue_antennas):
            ue_pos_tuple = tuple(ue_device_pos.tolist())
            
            # Update CSI predictions - ray_results format is {(ue_pos_tuple, subcarrier): signal}
            for s_idx, subcarrier_idx in enumerate(self.current_selection[batch_idx, antenna_idx, u]):
                if s_idx < num_selected:
                    result_key = (ue_pos_tuple, int(subcarrier_idx.item()))
                    
                    if result_key in ray_results:
                        signal = ray_results[result_key]
                        if isinstance(signal, (complex, torch.Tensor)):
                            csi_predictions[batch_idx, antenna_idx, u, s_idx] = signal
                        else:
                            logger.warning(f"⚠️ Invalid signal type: {type(signal)} for key {result_key}")
                    else:
                        logger.warning(f"⚠️ Missing ray result for key {result_key}")
                        logger.debug(f"   Available keys: {list(ray_results.keys())[:5]}...")  # Show first 5 keys for debugging
    
    def _create_subcarrier_dict(self, batch_idx: int, bs_antenna_idx: int, num_ue_antennas: int, ue_pos_list: list):
        """Create subcarrier dictionary for ray tracing."""
        selected_subcarriers = {}
        
        for u in range(num_ue_antennas):
            ue_pos_tuple = tuple(ue_pos_list[u].tolist())
            selection_tensor = self.current_selection[batch_idx, bs_antenna_idx, u]
            
            # Convert tensor to list safely
            try:
                if selection_tensor.numel() == 1:
                    selected_subcarriers[ue_pos_tuple] = [int(selection_tensor.item())]
                else:
                    tensor_list = selection_tensor.tolist()
                    if isinstance(tensor_list, (list, tuple)):
                        selected_subcarriers[ue_pos_tuple] = [int(idx) for idx in tensor_list]
                    else:
                        logger.warning(f"⚠️ Fallback: Using default subcarrier [0] for UE at position {ue_pos_tuple}")
                        selected_subcarriers[ue_pos_tuple] = [0]  # Fallback
            except Exception as e:
                logger.warning(f"Error converting subcarrier selection: {e}, using fallback")
                selected_subcarriers[ue_pos_tuple] = [0]  # Fallback
            
            # Ensure we have valid subcarriers
            if not selected_subcarriers[ue_pos_tuple]:
                logger.warning(f"⚠️ Using fallback subcarrier [0] for UE at position {ue_pos_tuple}")
                selected_subcarriers[ue_pos_tuple] = [0]
        
        return selected_subcarriers
    
    def _perform_ray_tracing(self, batch_idx: int, bs_position: torch.Tensor, ue_pos_list: list, 
                           selected_subcarriers: dict, current_antenna_indices: torch.Tensor):
        """Perform ray tracing based on the configured mode."""
        try:
            return self.ray_tracer.accumulate_signals(
                base_station_pos=bs_position[batch_idx],
                ue_positions=ue_pos_list,
                selected_subcarriers=selected_subcarriers,
                antenna_indices=current_antenna_indices[batch_idx:batch_idx+1]  # Keep as 1D tensor
            )
        except Exception as e:
            logger.error(f"Ray tracing failed: {e}")
            raise RuntimeError(f"Ray tracing failed: {e}")
    
    def _update_csi_predictions(self, csi_predictions: torch.Tensor, batch_ray_results: list, 
                              bs_antenna_idx: int, batch_size: int, num_ue_antennas: int, num_selected: int, ue_positions: torch.Tensor):
        """Update CSI predictions with ray tracing results."""
        zero_csi_count = 0  # Track zero CSI predictions
        missing_results_count = 0  # Track missing ray tracing results
        total_expected = 0  # Track total expected predictions
        
        for b in range(batch_size):
            ray_results = batch_ray_results[b]
            
            # Create UE position tuple for lookup - MUST match exactly how it was created in _create_subcarrier_dict
            # Use the same logic: ue_positions[b].cpu() and create tuple
            ue_device_pos = ue_positions[b].cpu()
            ue_pos_tuple = tuple(ue_device_pos.tolist())
            
            # Debug: Log the exact key creation process
            logger.debug(f"🔍 Creating lookup key for batch {b}:")
            logger.debug(f"   - ue_positions[b]: {ue_positions[b]}")
            logger.debug(f"   - ue_device_pos: {ue_device_pos}")
            logger.debug(f"   - ue_pos_tuple: {ue_pos_tuple}")
            logger.debug(f"   - ray_results keys: {list(ray_results.keys())[:5]}")
            
            # Debug: Log what we're looking for vs what we have
            logger.debug(f"🔍 Looking for results with UE position: {ue_pos_tuple}")
            logger.debug(f"🔍 Available ray_results keys: {list(ray_results.keys())[:10]}...")  # Show first 10 keys
            
            # Debug: Show the actual structure of ray_results
            logger.debug(f"🔍 ray_results structure analysis:")
            logger.debug(f"   - Total keys: {len(ray_results)}")
            logger.debug(f"   - First 5 keys: {list(ray_results.keys())[:5]}")
            logger.debug(f"   - Key types: {[type(key) for key in list(ray_results.keys())[:5]]}")
            logger.debug(f"   - Key lengths: {[len(key) if hasattr(key, '__len__') else 'scalar' for key in list(ray_results.keys())[:5]]}")
            
            # Show sample values
            if ray_results:
                sample_key = list(ray_results.keys())[0]
                sample_value = ray_results[sample_key]
                logger.debug(f"🔍 Sample key-value pair:")
                logger.debug(f"   - Key: {sample_key} (type: {type(sample_key)})")
                logger.debug(f"   - Value: {sample_value} (type: {type(sample_value)})")
                if hasattr(sample_value, 'dtype'):
                    logger.debug(f"   - Value dtype: {sample_value.dtype}")
                if hasattr(sample_value, 'shape'):
                    logger.debug(f"   - Value shape: {sample_value.shape}")
            
            for u in range(num_ue_antennas):
                # Get the selected subcarriers for this UE antenna and BS antenna
                ue_selected_subcarriers = self.current_selection[b, bs_antenna_idx, u].tolist()
                logger.debug(f"🔍 DEBUG: Selected subcarriers for batch {b}, BS antenna {bs_antenna_idx}, UE antenna {u}: {ue_selected_subcarriers}")
                
                for k_idx, k in enumerate(ue_selected_subcarriers):
                    if k_idx < num_selected:
                        total_expected += 1
                        
                        # Look for results in ray_results dictionary
                        expected_key = (ue_pos_tuple, k)
                        if expected_key in ray_results:
                            rf_signal = ray_results[expected_key]
                            # RF signal is always complex, convert to complex64 if needed
                            csi_value = rf_signal.to(torch.complex64)
                            csi_predictions[b, bs_antenna_idx, u, k_idx] = csi_value
                            
                            # Debug: Log the actual signal values
                            logger.debug(f"🔍 Found signal for key {expected_key}:")
                            logger.debug(f"   - rf_signal: {rf_signal}")
                            logger.debug(f"   - csi_value: {csi_value}")
                            logger.debug(f"   - abs(csi_value): {torch.abs(csi_value)}")
                            
                            # Check if CSI is zero and count
                            if torch.abs(csi_value) < 1e-10:  # Small threshold for numerical zero
                                zero_csi_count += 1
                                logger.warning(f"⚠️  Zero CSI detected for key {expected_key}")
                                
                        else:
                            # Log error and raise exception to stop training
                            missing_results_count += 1
                            logger.error(f"❌ Missing ray tracing result for subcarrier {k} at UE position {ue_pos_tuple}")
                            logger.error(f"   - Expected key: {expected_key}")
                            logger.error(f"   - Available keys with same UE position: {[key for key in ray_results.keys() if key[0] == ue_pos_tuple][:5]}")
                            logger.error(f"   - Total available keys: {len(ray_results)}")
                            logger.error(f"   - First 10 available keys: {list(ray_results.keys())[:10]}")
                            
                            # Check if this is a subcarrier selection issue
                            logger.error(f"   - This suggests a subcarrier selection inconsistency between _create_subcarrier_dict and ray tracer")
                            logger.error(f"   - The ray tracer did not calculate results for all selected subcarriers")
                            
                            # Raise error to stop training and force investigation
                            error_msg = f"Missing ray tracing result for batch {b}, BS antenna {bs_antenna_idx}, UE antenna {u}, subcarrier {k}"
                            raise RuntimeError(error_msg)
        
        # Log summary statistics
        total_predictions = batch_size * num_ue_antennas * num_selected
        zero_percentage = (zero_csi_count / total_predictions) * 100 if total_predictions > 0 else 0
        missing_percentage = (missing_results_count / total_predictions) * 100 if total_predictions > 0 else 0
        
        logger.info(f"📊 BS Antenna {bs_antenna_idx} CSI Update Summary:")
        logger.info(f"   - Total expected: {total_expected}")
        logger.info(f"   - Zero CSI: {zero_csi_count} ({zero_percentage:.1f}%)")
        logger.info(f"   - Missing results: {missing_results_count} ({missing_percentage:.1f}%)")
        logger.info(f"   - Successful updates: {total_expected - missing_results_count - zero_csi_count}")
        
        # Validate subcarrier selection consistency
        if missing_results_count > 0:
            logger.error(f"❌ SUBCARRIER SELECTION INCONSISTENCY DETECTED!")
            logger.error(f"   - The ray tracer did not calculate results for {missing_results_count} selected subcarriers")
            logger.error(f"   - This indicates a bug in the subcarrier selection or ray tracing logic")
            logger.error(f"   - Check _create_subcarrier_dict and ray_tracer.accumulate_signals for consistency")
    
    
    def _create_full_predictions(self, csi_predictions: torch.Tensor, batch_size: int, num_bs_antennas: int, 
                               num_ue_antennas: int, num_subcarriers: int, device: torch.device):
        """Map selected subcarrier predictions back to full subcarrier space."""
        full_predictions = torch.zeros(
            batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas,
            dtype=torch.complex64, device=device
        )
        
        # Fill in the predictions for selected subcarriers
        for b in range(batch_size):
            for bs_antenna_idx in range(num_bs_antennas):
                for u in range(num_ue_antennas):
                    selected_indices = self.current_selection[b, bs_antenna_idx, u]
                    for k_idx, k in enumerate(selected_indices):
                        if k_idx < csi_predictions.shape[-1]:
                            full_predictions[b, k, u, bs_antenna_idx] = csi_predictions[b, bs_antenna_idx, u, k_idx]
        
        # Log the final shape for debugging
        logger.info(f"📊 Final CSI predictions shape: {full_predictions.shape} - {full_predictions.dtype}")
        
        return full_predictions
    
    def _select_subcarriers(
        self, 
        batch_size: int, 
        num_ue_antennas: int, 
        num_bs_antennas: int,
        total_subcarriers: int,
        num_selected: int
    ) -> Dict[str, torch.Tensor]:
        """Select subcarriers based on configured sampling method.
        
        If antenna_consistent=True: All BS antennas use same subcarriers for each UE
        If antenna_consistent=False: Each BS antenna selects independently
        """
        # Validate parameters
        if total_subcarriers <= 0:
            raise ValueError(f"total_subcarriers must be positive, got {total_subcarriers}")
        if num_selected <= 0:
            raise ValueError(f"num_selected must be positive, got {num_selected}")
        if num_selected > total_subcarriers:
            raise ValueError(f"num_selected ({num_selected}) cannot be greater than total_subcarriers ({total_subcarriers})")
        
        logger.debug(f"_select_subcarriers called with:")
        logger.debug(f"  batch_size: {batch_size}")
        logger.debug(f"  num_ue_antennas: {num_ue_antennas}")
        logger.debug(f"  num_bs_antennas: {num_bs_antennas}")
        logger.debug(f"  total_subcarriers: {total_subcarriers}")
        logger.debug(f"  num_selected: {num_selected}")
        logger.debug(f"  sampling_method: {self.subcarrier_sampling_method}")
        logger.debug(f"  antenna_consistent: {self.antenna_consistent}")
        
        selected_indices = torch.zeros(batch_size, num_bs_antennas, num_ue_antennas, num_selected, dtype=torch.long)
        selection_mask = torch.zeros(batch_size, num_bs_antennas, num_ue_antennas, total_subcarriers, dtype=torch.bool)
        
        # Select subcarriers ONCE per batch to ensure consistency and avoid parallelization issues
        for b in range(batch_size):
            try:
                # Select subcarriers for this batch (same for all UEs and BS antennas in the batch)
                if self.subcarrier_sampling_method == 'uniform':
                    # Uniform sampling: evenly spaced subcarriers
                    step = total_subcarriers // num_selected
                    batch_selected = [i * step for i in range(num_selected)]
                    # Ensure we don't exceed bounds
                    batch_selected = [min(idx, total_subcarriers - 1) for idx in batch_selected]
                else:  # 'random' (default)
                    # Random sampling: randomly selected subcarriers
                    batch_selected = random.sample(range(total_subcarriers), num_selected)
                
                logger.debug(f"Batch {b}: selected subcarriers {batch_selected} (method: {self.subcarrier_sampling_method})")
                
                # Apply the SAME subcarrier selection to ALL UEs and BS antennas in this batch
                for u in range(num_ue_antennas):
                    for bs_antenna in range(num_bs_antennas):
                        selected_indices[b, bs_antenna, u] = torch.tensor(batch_selected, dtype=torch.long)
                        selection_mask[b, bs_antenna, u, batch_selected] = True
                        
            except Exception as e:
                logger.error(f"Error in subcarrier selection for batch {b}: {e}")
                logger.error(f"  total_subcarriers: {total_subcarriers}, num_selected: {num_selected}")
                logger.error(f"  sampling_method: {self.subcarrier_sampling_method}")
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

    def _extract_selected_subcarriers(self, csi_data: torch.Tensor, selection_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract selected subcarriers from full CSI data based on selection mask
        
        Args:
            csi_data: Full CSI tensor (batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas)
            selection_mask: Boolean mask (batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers)
            
        Returns:
            selected_csi: CSI tensor with only selected subcarriers 
                         (batch_size, num_selected_subcarriers, num_ue_antennas, num_bs_antennas)
        """
        batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas = csi_data.shape
        device = csi_data.device
        
        # Get the number of selected subcarriers (should be consistent across batch)
        num_selected = torch.sum(selection_mask[0, 0, 0, :]).item()
        
        # Initialize output tensor
        selected_csi = torch.zeros(
            (batch_size, num_selected, num_ue_antennas, num_bs_antennas), 
            dtype=csi_data.dtype, 
            device=device
        )
        
        # Extract selected subcarriers for each batch sample
        for b in range(batch_size):
            # Get selected subcarrier indices for this batch (should be same across antennas)
            # Use the first antenna pair as reference
            selected_indices = torch.where(selection_mask[b, 0, 0, :])[0]
            
            # Extract selected subcarriers
            selected_csi[b] = csi_data[b, selected_indices, :, :]
        
        return selected_csi
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        loss_function: nn.Module,
        validation_mode: bool = False
    ) -> torch.Tensor:
        """Compute loss for selected subcarriers."""
        # Debug logging
        logger.debug(f"compute_loss called with predictions shape: {predictions.shape}, targets shape: {targets.shape}")
        
        # Validate shapes - targets are in format (batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas)
        batch_size, num_subcarriers, target_ue_antennas, num_bs_antennas = targets.shape
        pred_batch_size, pred_subcarriers, pred_ue_antennas, pred_bs_antennas = predictions.shape
        
        # Handle UE antenna dimension mismatch between predictions and targets FIRST
        if target_ue_antennas != pred_ue_antennas:
            logger.info(f"🔧 UE antenna dimension mismatch detected:")
            logger.info(f"   Predictions UE antennas: {pred_ue_antennas}")
            logger.info(f"   Targets UE antennas: {target_ue_antennas}")
            
            # Since we now extract single antenna during data loading, this should rarely happen
            # But keep the logic for backward compatibility
            if target_ue_antennas > pred_ue_antennas and pred_ue_antennas == 1:
                # Use only the first UE antenna from targets to match predictions
                targets = targets[:, :, :pred_ue_antennas, :]
                logger.info(f"   ✅ Adjusted targets to use first {pred_ue_antennas} UE antenna(s)")
                logger.info(f"   New targets shape: {targets.shape}")
                # Update target dimensions after adjustment
                target_ue_antennas = pred_ue_antennas
            else:
                raise ValueError(f"Cannot match dimensions: targets have {target_ue_antennas} UE antennas but predictions have {pred_ue_antennas}")
        
        # Now validate that all dimensions match
        expected_pred_shape = (batch_size, num_subcarriers, target_ue_antennas, num_bs_antennas)
        
        if predictions.shape != expected_pred_shape:
            logger.error(f"Predictions shape mismatch. Expected: {expected_pred_shape}, Got: {predictions.shape}")
            raise ValueError(f"Predictions shape mismatch. Expected: {expected_pred_shape}, Got: {predictions.shape}")
        
        # Check if current_selection attributes exist, if not, initialize them
        # Note: This initialization happens AFTER UE antenna dimension adjustment
        if not hasattr(self, 'current_selection') or self.current_selection is None:
            logger.warning("current_selection not found, initializing with default values")
            # Use the adjusted target dimensions (after UE antenna matching)
            num_selected = int(num_subcarriers * self.subcarrier_sampling_ratio)
            self._initialize_selection_variables(batch_size, target_ue_antennas, num_bs_antennas, num_subcarriers)
            # Create default selection for all antennas
            selection_info = self._select_subcarriers(batch_size, target_ue_antennas, num_bs_antennas, num_subcarriers, num_selected)
            self.current_selection = selection_info['selected_indices']
            self.current_selection_mask = selection_info['selection_mask']
        
        logger.debug(f"current_selection: {self.current_selection is not None}, current_selection_mask: {self.current_selection_mask is not None}")
        
        if self.current_selection is None or self.current_selection_mask is None:
            logger.error("No subcarrier selection available even after initialization.")
            logger.error(f"current_selection: {self.current_selection}")
            logger.error(f"current_selection_mask: {self.current_selection_mask}")
            raise ValueError("No subcarrier selection available even after initialization.")
        
        # Extract only the traced/selected subcarriers from both predictions and targets
        # and compute MSE loss only on those values, completely ignoring untraced subcarriers
        try:
            # Collect all traced subcarrier values
            traced_predictions = []
            traced_targets = []
            
            # Extract values for selected subcarriers only
            zero_csi_count = 0  # Track zero CSI predictions
            for b in range(batch_size):
                for bs_antenna_idx in range(num_bs_antennas):
                    for u in range(target_ue_antennas):
                        if self.current_selection is not None:
                            selected_indices = self.current_selection[b, bs_antenna_idx, u]
                            for k in selected_indices:
                                k = int(k)  # Ensure k is an integer for indexing
                                if k < num_subcarriers:  # Ensure index is valid
                                    csi_value = predictions[b, k, u, bs_antenna_idx]
                                    # Check if CSI absolute value is zero
                                    if torch.abs(csi_value) < 1e-10:
                                        zero_csi_count += 1
                                        logger.warning(f"⚠️  Zero CSI detected: batch={b}, bs_antenna={bs_antenna_idx}, ue_antenna={u}, subcarrier={k}")
                                    
                                    traced_predictions.append(csi_value)
                                    traced_targets.append(targets[b, k, u, bs_antenna_idx])
            
            # Log warning if too many zero CSI values
            if zero_csi_count > 0:
                total_selected = len(traced_predictions)
                zero_percentage = (zero_csi_count / total_selected) * 100
                logger.warning(f"⚠️  {zero_csi_count}/{total_selected} ({zero_percentage:.1f}%) selected subcarriers have zero CSI")
                if zero_percentage > 50:
                    logger.error(f"❌ CRITICAL: More than 50% of selected subcarriers have zero CSI!")
                    logger.error(f"   This indicates serious issues with ray tracing or model predictions")
            
            # Convert to tensors
            if len(traced_predictions) == 0:
                logger.warning("No traced subcarriers found, returning zero loss")
                # Create a zero loss that maintains the computational graph
                zero_loss = torch.sum(predictions * 0.0)
                return zero_loss
            
            traced_predictions = torch.stack(traced_predictions)
            traced_targets = torch.stack(traced_targets)
            
            # Check for invalid values in predictions and targets
            pred_has_nan = torch.isnan(traced_predictions).any()
            pred_has_inf = torch.isinf(traced_predictions).any()
            target_has_nan = torch.isnan(traced_targets).any()
            target_has_inf = torch.isinf(traced_targets).any()
            
            logger.info(f"🔍 CSI Validation:")
            logger.info(f"   - Predictions: NaN={pred_has_nan}, Inf={pred_has_inf}")
            logger.info(f"   - Targets: NaN={target_has_nan}, Inf={target_has_inf}")
            logger.info(f"   - Pred range: [{torch.abs(traced_predictions).min():.6f}, {torch.abs(traced_predictions).max():.6f}]")
            logger.info(f"   - Target range: [{torch.abs(traced_targets).min():.6f}, {torch.abs(traced_targets).max():.6f}]")
            
            if pred_has_nan or pred_has_inf:
                logger.error(f"❌ CRITICAL: Invalid values in predictions!")
                logger.error(f"   - NaN count: {torch.isnan(traced_predictions).sum()}")
                logger.error(f"   - Inf count: {torch.isinf(traced_predictions).sum()}")
                # Replace invalid values with zeros to prevent NaN loss
                traced_predictions = torch.where(torch.isnan(traced_predictions) | torch.isinf(traced_predictions), 
                                               torch.zeros_like(traced_predictions), traced_predictions)
                logger.warning("⚠️  Replaced invalid prediction values with zeros")
            
            if target_has_nan or target_has_inf:
                logger.error(f"❌ CRITICAL: Invalid values in targets!")
                logger.error(f"   - NaN count: {torch.isnan(traced_targets).sum()}")
                logger.error(f"   - Inf count: {torch.isinf(traced_targets).sum()}")
            
            logger.debug(f"Computing loss on {len(traced_predictions)} traced subcarriers")
            logger.debug(f"Traced predictions shape: {traced_predictions.shape}")
            logger.debug(f"Traced targets shape: {traced_targets.shape}")
            logger.debug(f"Subcarrier sampling ratio: {self.subcarrier_sampling_ratio}")
            logger.debug(f"Expected selected subcarriers per antenna-UE pair: {int(self.prism_network.num_subcarriers * self.subcarrier_sampling_ratio)}")
            
            # Create selected CSI tensors for loss computation
            # Extract selected subcarriers based on current_selection_mask
            selected_predictions = self._extract_selected_subcarriers(predictions, self.current_selection_mask)
            selected_targets = self._extract_selected_subcarriers(targets, self.current_selection_mask)
            
            # Prepare data for loss functions
            predictions_dict = {'csi': selected_predictions}
            targets_dict = {'csi': selected_targets}
            
            # Also pass the traced values for backward compatibility with CSI loss component
            predictions_dict['traced_csi'] = traced_predictions
            targets_dict['traced_csi'] = traced_targets
            
            # Call loss function without masks (data is already selected)
            loss, loss_components = loss_function(predictions_dict, targets_dict)
            
            # Log detailed loss components
            logger.info(f"🔍 Hybrid CSI+PDP+Spatial Loss computed:")
            logger.info(f"   Total Loss: {loss.item():.6f}")
            if 'csi_loss' in loss_components:
                csi_raw = loss_components['csi_loss']
                csi_weighted = loss_function.csi_weight * csi_raw
                logger.info(f"   CSI Loss: {csi_raw:.6f} (raw) → {csi_weighted:.6f} (weighted × {loss_function.csi_weight})")
            if 'pdp_loss' in loss_components:
                pdp_raw = loss_components['pdp_loss']
                pdp_weighted = loss_function.pdp_weight * pdp_raw
                logger.info(f"   PDP Loss: {pdp_raw:.6f} (raw) → {pdp_weighted:.6f} (weighted × {loss_function.pdp_weight})")
            if 'ss_loss' in loss_components:
                spatial_raw = loss_components['ss_loss']
                spatial_weighted = loss_function.spatial_spectrum_weight * spatial_raw
                logger.info(f"   SS Loss: {spatial_raw:.6f} (raw) → {spatial_weighted:.6f} (weighted × {loss_function.spatial_spectrum_weight})")
            
            # Final validation of computed loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"❌ CRITICAL: Loss computation resulted in invalid value: {loss}")
                logger.error(f"   - Loss type: {type(loss)}")
                logger.error(f"   - Loss requires_grad: {loss.requires_grad}")
                logger.error(f"   - Predictions stats: mean={torch.abs(traced_predictions).mean():.6f}, std={torch.abs(traced_predictions).std():.6f}")
                logger.error(f"   - Targets stats: mean={torch.abs(traced_targets).mean():.6f}, std={torch.abs(traced_targets).std():.6f}")
                # Return a small positive loss to prevent training crash
                loss = torch.tensor(1e-6, requires_grad=True, device=predictions.device)
                logger.warning("⚠️  Replaced invalid loss with small positive value")
            
            # Validate loss is a tensor
            if not isinstance(loss, torch.Tensor):
                logger.error(f"Loss function returned non-tensor: {type(loss)} = {loss}")
                raise ValueError(f"Loss function must return a torch.Tensor, got {type(loss)}")
            
            # Ensure the loss requires gradients for backpropagation (skip check in validation mode)
            if not validation_mode and not loss.requires_grad:
                logger.error("❌ CRITICAL: Loss tensor does not require gradients!")
                logger.error(f"   - Loss type: {type(loss)}")
                logger.error(f"   - Loss device: {loss.device}")
                logger.error(f"   - Predictions requires_grad: {traced_predictions.requires_grad}")
                logger.error(f"   - Targets requires_grad: {traced_targets.requires_grad}")
                raise RuntimeError("Loss tensor must require gradients for proper training. This indicates a bug in the loss function.")
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in compute_loss: {e}")
            logger.error(f"Shapes - predictions: {predictions.shape}, targets: {targets.shape}")
            logger.error(f"current_selection type: {type(self.current_selection)}")
            logger.error(f"current_selection shape: {self.current_selection.shape if self.current_selection is not None else 'None'}")
            logger.error(f"current_selection dtype: {self.current_selection.dtype if self.current_selection is not None else 'None'}")
            
            # Add detailed traceback
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _get_device(self):
        """Get the device for tensors, handling DataParallel wrapper."""
        try:
            # Try to get device from prism_network parameters
            if hasattr(self.prism_network, 'parameters'):
                # Handle DataParallel wrapper
                if hasattr(self.prism_network, 'module'):
                    return next(self.prism_network.module.parameters()).device
                else:
                    return next(self.prism_network.parameters()).device
        except (StopIteration, AttributeError):
            pass
        
        # Fallback to CUDA if available, otherwise CPU
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')
    
    def _initialize_selection_variables(self, batch_size: int, num_ue_antennas: int, num_bs_antennas: int, num_subcarriers: int):
        """Initialize selection variables with default values."""
        # Create default selection variables for the current configuration
        num_selected = int(num_subcarriers * self.subcarrier_sampling_ratio)
        
        # Get device safely
        device = self._get_device()
        
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
            
            # Get device safely
            device = self._get_device()
            
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
    

    # Checkpoint and recovery methods
    def save_checkpoint(self, filename: str = None, optimizer_state_dict: dict = None, scheduler_state_dict: dict = None):
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
        
        # Add optimizer and scheduler states if provided
        if optimizer_state_dict is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state_dict
        if scheduler_state_dict is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state_dict
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 TrainingInterface checkpoint saved: {checkpoint_path}")
        logger.info(f"📊 Checkpoint includes: epoch={self.current_epoch}, batch={self.current_batch}, best_loss={self.best_loss:.6f}")
        
        # Save training info
        info_path = checkpoint_path.replace('.pt', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(self.get_training_info(), f, indent=2)
        logger.info(f"📝 Training info saved: {info_path}")
    
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
            'best_loss': float(self.best_loss.item()) if isinstance(self.best_loss, torch.Tensor) else float(self.best_loss),
            'training_history_length': len(self.training_history),
            'checkpoint_dir': self.checkpoint_dir,
            'prism_network_config': {
                'azimuth_divisions': self.prism_network.azimuth_divisions,
                'elevation_divisions': self.prism_network.elevation_divisions,
                'top_k_directions': self.prism_network.top_k_directions
            }
        }
    
    def _log_training_ray_count(self):
        """Calculate and log the total number of rays for training based on configuration."""
        try:
            # Get configuration values
            angular_sampling = self.ray_tracing_config.get('angular_sampling', {})
            radial_sampling = self.ray_tracing_config.get('radial_sampling', {})
            subcarrier_sampling = self.ray_tracing_config.get('subcarrier_sampling', {})
            
            # Use default training configuration since we don't have access to training config
            num_epochs = 2  # Default value
            batches_per_epoch = 5  # Default value
            
            # Get ray tracing configuration values
            top_k_directions = angular_sampling.get('top_k_directions', 32)
            azimuth_divisions = angular_sampling.get('azimuth_divisions', 18)
            elevation_divisions = angular_sampling.get('elevation_divisions', 9)
            num_sampling_points = radial_sampling.get('num_sampling_points', 64)
            resampled_points = radial_sampling.get('resampled_points', 32)
            
            # Get subcarrier configuration
            sampling_ratio = subcarrier_sampling.get('sampling_ratio', 0.01)
            num_subcarriers = getattr(self.prism_network, 'num_subcarriers', 408)
            
            # Get system configuration
            num_bs_antennas = self.system_config.get('base_station', {}).get('num_antennas', 64)
            num_ue_antennas = self.user_equipment_config.get('num_ue_antennas', 1)
            
            # Calculate ray counts
            rays_per_direction = num_sampling_points + resampled_points
            rays_per_batch = top_k_directions * rays_per_direction * num_subcarriers * sampling_ratio
            rays_per_epoch = batches_per_epoch * rays_per_batch
            total_rays_training = num_epochs * rays_per_epoch
            
            # Calculate total possible directions
            total_possible_directions = azimuth_divisions * elevation_divisions
            
            # Log detailed breakdown
            logger.info("=" * 80)
            logger.info("📊 TRAINING RAY COUNT ANALYSIS")
            logger.info("=" * 80)
            logger.info(f"🎯 Training Configuration:")
            logger.info(f"   • Number of epochs: {num_epochs}")
            logger.info(f"   • Batches per epoch: {batches_per_epoch}")
            logger.info(f"   • Total batches: {num_epochs * batches_per_epoch}")
            logger.info(f"")
            logger.info(f"🔍 Ray Tracing Configuration:")
            logger.info(f"   • Top-K directions: {top_k_directions}")
            logger.info(f"   • Total possible directions: {total_possible_directions} ({azimuth_divisions} × {elevation_divisions})")
            logger.info(f"   • Sampling points per ray: {num_sampling_points} uniform + {resampled_points} resampled = {rays_per_direction}")
            logger.info(f"   • Total subcarriers: {num_subcarriers}")
            logger.info(f"   • Subcarrier sampling ratio: {sampling_ratio:.3f} ({sampling_ratio*100:.1f}%)")
            logger.info(f"   • BS antennas: {num_bs_antennas}")
            logger.info(f"   • UE antennas per device: {num_ue_antennas}")
            logger.info(f"")
            logger.info(f"🧮 Ray Count Calculations:")
            logger.info(f"   • Rays per direction: {rays_per_direction:,}")
            logger.info(f"   • Rays per batch: {rays_per_batch:,.0f}")
            logger.info(f"   • Rays per epoch: {rays_per_epoch:,.0f}")
            logger.info(f"   • Total rays for training: {total_rays_training:,.0f}")
            logger.info(f"")
            logger.info(f"⚡ Performance Notes:")
            logger.info(f"   • CUDA acceleration will process {top_k_directions} directions per batch")
            logger.info(f"   • Each direction processes {num_subcarriers * sampling_ratio:.1f} subcarriers")
            logger.info(f"   • Total computational complexity: O({total_rays_training:,.0e})")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.warning(f"⚠️  Could not calculate training ray count: {e}")
            logger.info("Using default ray count logging")
    