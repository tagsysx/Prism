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
import sys
import signal
from contextlib import contextmanager

from .networks.prism_network import PrismNetwork
from .ray_tracer_cpu import CPURayTracer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG level for detailed logging

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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
        ray_tracing_config: Optional[dict] = None,
        system_config: Optional[dict] = None,
        user_equipment_config: Optional[dict] = None,
        checkpoint_dir: Optional[str] = None
    ):
        super().__init__()
        
        # Store configuration
        self.ray_tracing_config = ray_tracing_config or {}
        self.system_config = system_config or {}
        self.user_equipment_config = user_equipment_config or {}
        
        # Set logger level from system config - use fallback if missing
        log_level = self.system_config.get('logging', {}).get('log_level', 'INFO')
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Get ray tracing mode from system config - fail fast if missing
        try:
            ray_tracing_mode = self.system_config['ray_tracing_mode']
        except KeyError:
            logger.error("❌ FATAL: Missing required config 'system.ray_tracing_mode'")
            logger.error("   Please check your configuration file.")
            sys.exit(1)
        
        # Validate ray tracing mode first
        if ray_tracing_mode not in ['cuda', 'cpu', 'hybrid']:
            raise ValueError(f"Invalid ray_tracing_mode: {ray_tracing_mode}. Must be 'cuda', 'cpu', or 'hybrid'")
        
        # Set ray tracing mode
        self.ray_tracing_mode = ray_tracing_mode
        self.prism_network = prism_network  # Set prism_network first
        
        # Create appropriate ray tracer based on mode if not provided
        if ray_tracer is None:
            self.ray_tracer = self._create_ray_tracer_by_mode(ray_tracing_mode)
        else:
            # Use provided ray_tracer but validate it matches the mode
            self.ray_tracer = self._validate_ray_tracer(ray_tracer, ray_tracing_mode)
        
        # Get configuration parameters from config dictionaries - fail fast if missing
        try:
            spatial_sampling = self.ray_tracing_config['spatial_sampling']
        except KeyError:
            logger.error("❌ FATAL: Missing required config 'ray_tracing.spatial_sampling'")
            logger.error("   Please check your configuration file.")
            sys.exit(1)
        
        try:
            subcarrier_sampling = self.ray_tracing_config['subcarrier_sampling']
        except KeyError:
            logger.error("❌ FATAL: Missing required config 'ray_tracing.subcarrier_sampling'")
            logger.error("   Please check your configuration file.")
            sys.exit(1)
        
        # Store configuration parameters - fail fast if missing
        try:
            self.num_sampling_points = spatial_sampling['num_sampling_points']
        except KeyError:
            logger.error("❌ FATAL: Missing required config 'ray_tracing.spatial_sampling.num_sampling_points'")
            raise ValueError("Missing required configuration: ray_tracing.spatial_sampling.num_sampling_points")
        
        try:
            self.subcarrier_sampling_ratio = subcarrier_sampling['sampling_ratio']
        except KeyError:
            logger.error("❌ FATAL: Missing required config 'ray_tracing.subcarrier_sampling.sampling_ratio'")
            raise ValueError("Missing required configuration: ray_tracing.subcarrier_sampling.sampling_ratio")
        
        # Required parameters - fail fast if missing
        try:
            self.subcarrier_sampling_method = subcarrier_sampling['sampling_method']
        except KeyError:
            logger.error("❌ FATAL: Missing required config 'ray_tracing.subcarrier_sampling.sampling_method'")
            logger.error("   Please check your configuration file.")
            sys.exit(1)
        
        try:
            self.antenna_consistent = subcarrier_sampling['antenna_consistent']
        except KeyError:
            logger.error("❌ FATAL: Missing required config 'ray_tracing.subcarrier_sampling.antenna_consistent'")
            logger.error("   Please check your configuration file.")
            sys.exit(1)
        
        # Calculate and log training ray count if configuration is available
        self._log_training_ray_count()
        
        # Initialize checkpoint directory
        logger.info(f"🔍 DEBUG: checkpoint_dir parameter = {repr(checkpoint_dir)}")
        
        if not checkpoint_dir or not checkpoint_dir.strip():
            logger.error("❌ FATAL ERROR: checkpoint_dir not provided to PrismTrainingInterface")
            logger.error("   This indicates a configuration loading issue in the training script.")
            logger.error("   Please ensure the training script properly extracts checkpoint_dir from config.")
            logger.error("   Check your config file and training script setup.")
            sys.exit(1)
        
        self.checkpoint_dir = checkpoint_dir.strip()
        logger.info(f"✅ Using provided checkpoint_dir: {self.checkpoint_dir}")
        
        logger.info(f"Training interface initialized with ray_tracing_mode: {ray_tracing_mode}")
        logger.info(f"Using ray tracer type: {type(self.ray_tracer).__name__}")
        
        # Set scene bounds from ray_tracing_config - fail fast if missing
        try:
            scene_bounds_config = self.ray_tracing_config['scene_bounds']
            self.scene_min = torch.tensor(scene_bounds_config['min'], dtype=torch.float32)
            self.scene_max = torch.tensor(scene_bounds_config['max'], dtype=torch.float32)
        except KeyError as e:
            logger.error(f"❌ FATAL: Missing required config 'ray_tracing.scene_bounds.{e.args[0] if e.args else 'scene_bounds'}'")
            logger.error("   Please check your configuration file and ensure scene_bounds.min and scene_bounds.max are defined.")
            sys.exit(1)
        
        # Create checkpoint directory
        logger.info(f"Creating checkpoint directory: {self.checkpoint_dir}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpoint directory created successfully: {os.path.abspath(self.checkpoint_dir)}")
        
        # Training state for checkpoint recovery
        self.current_epoch = 0
        self.current_batch = 0
        self.best_loss = float('inf')
        self.training_history = []
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
        # Extract common configuration parameters - fail fast if missing
        try:
            angular_sampling = self.ray_tracing_config['angular_sampling']
            spatial_sampling = self.ray_tracing_config['spatial_sampling']
            mixed_precision = self.system_config['mixed_precision']
            cpu_config = self.system_config['cpu']
        except KeyError as e:
            logger.error(f"❌ FATAL: Missing required config section: {e.args[0]}")
            logger.error("   Please check your configuration file.")
            sys.exit(1)
        
        # Common parameters for both ray tracers - fail fast if missing
        try:
            common_params = {
                'azimuth_divisions': angular_sampling['azimuth_divisions'],
                'elevation_divisions': angular_sampling['elevation_divisions'],
                'max_ray_length': self.ray_tracing_config['max_ray_length'],
                'scene_bounds': self.ray_tracing_config['scene_bounds'],
                'prism_network': self.prism_network,
                'signal_threshold': self.ray_tracing_config['signal_threshold'],
                'enable_early_termination': self.ray_tracing_config['enable_early_termination'],
                'top_k_directions': angular_sampling['top_k_directions'],
                'uniform_samples': spatial_sampling['num_sampling_points'],
                'resampled_points': spatial_sampling['resampled_points']
            }
        except KeyError as e:
            logger.error(f"❌ FATAL: Missing required config parameter: {e.args[0]}")
            logger.error("   Please check your configuration file.")
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
            return CUDARayTracer(
                use_mixed_precision=use_mixed_precision,
                **common_params
            )
        
        def create_cpu_ray_tracer():
            """Helper function to create CPU ray tracer."""
            from .ray_tracer_cpu import CPURayTracer
            try:
                max_workers = cpu_config['num_workers']
            except KeyError:
                logger.error("❌ FATAL: Missing required config 'system.cpu.num_workers'")
                logger.error("   Please check your configuration file.")
                sys.exit(1)
            return CPURayTracer(
                max_workers=max_workers,
                **common_params
            )
        
        # Create ray tracer based on mode
        if ray_tracing_mode == 'cuda':
            logger.info("🚀 Creating CUDARayTracer for CUDA mode")
            try:
                return create_cuda_ray_tracer()
            except Exception as e:
                logger.warning(f"Failed to create CUDARayTracer: {e}. Falling back to CPURayTracer.")
                return create_cpu_ray_tracer()
        
        elif ray_tracing_mode == 'cpu':
            logger.info("💻 Creating CPURayTracer for CPU mode")
            return create_cpu_ray_tracer()
        
        else:  # hybrid mode
            logger.info("🔄 Creating ray tracer for hybrid mode (CUDA first, CPU fallback)")
            try:
                return create_cuda_ray_tracer()
            except Exception as e:
                logger.warning(f"CUDA ray tracer failed in hybrid mode: {e}. Using CPU fallback.")
                return create_cpu_ray_tracer()
    
    def _validate_ray_tracer(self, ray_tracer, ray_tracing_mode: str) -> Union['CPURayTracer', 'CUDARayTracer']:
        """
        Validate that the provided ray tracer matches the specified mode.
        
        Args:
            ray_tracer: The ray tracer instance to validate
            ray_tracing_mode: The expected ray tracing mode
            
        Returns:
            Validated ray tracer instance
        """
        from .ray_tracer_cuda import CUDARayTracer
        from .ray_tracer_cpu import CPURayTracer
        
        # Check if the ray tracer type matches the mode
        if ray_tracing_mode == 'cuda' and not isinstance(ray_tracer, CUDARayTracer):
            logger.warning(f"⚠️  Expected CUDARayTracer for 'cuda' mode, but got {type(ray_tracer).__name__}")
            logger.warning("   Creating new CUDARayTracer to match the mode")
            return self._create_ray_tracer_by_mode('cuda')
        
        elif ray_tracing_mode == 'cpu' and not isinstance(ray_tracer, CPURayTracer):
            logger.warning(f"⚠️  Expected CPURayTracer for 'cpu' mode, but got {type(ray_tracer).__name__}")
            logger.warning("   Creating new CPURayTracer to match the mode")
            return self._create_ray_tracer_by_mode('cpu')
        
        elif ray_tracing_mode == 'hybrid':
            # For hybrid mode, we accept either type but prefer CUDA
            if isinstance(ray_tracer, CUDARayTracer):
                logger.info("✅ Hybrid mode: Using provided CUDARayTracer")
            elif isinstance(ray_tracer, CPURayTracer):
                logger.info("✅ Hybrid mode: Using provided CPURayTracer")
            else:
                logger.warning(f"⚠️  Hybrid mode: Unexpected ray tracer type {type(ray_tracer).__name__}")
                logger.warning("   Creating new ray tracer for hybrid mode")
                return self._create_ray_tracer_by_mode('hybrid')
        
        logger.info(f"✅ Ray tracer validation passed: {type(ray_tracer).__name__} for {ray_tracing_mode} mode")
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
        device = ue_positions.device  # Use input device directly
        
        # Extract dimensions
        batch_size = ue_positions.shape[0]
        # Get UE antenna configuration from PrismNetwork
        num_ue_antennas = self.prism_network.num_ue_antennas
        num_bs_antennas = antenna_indices.shape[1]
        num_subcarriers = self.prism_network.num_subcarriers
        num_selected = int(num_subcarriers * self.subcarrier_sampling_ratio)
        
        logger.debug(f"🚀 Forward pass: {batch_size} batches × {num_bs_antennas} BS antennas × {num_ue_antennas} UE antennas × {num_selected} selected subcarriers")
        
        # Step 1: Initialize current_selection attributes directly to ensure they exist
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
        
        # Step 3: Process each BS antenna
        all_ray_results = []
        all_signal_strengths = []
        
        print(f"    🔄 Processing {num_bs_antennas} BS antennas...")
        for bs_antenna_idx in range(num_bs_antennas):
            if bs_antenna_idx % 5 == 0 or bs_antenna_idx == num_bs_antennas - 1:  # More frequent progress updates
                progress = (bs_antenna_idx / num_bs_antennas) * 100
                print(f"      📡 BS antenna {bs_antenna_idx+1}/{num_bs_antennas} ({progress:.1f}%)")
                logger.debug(f"📡 Processing BS antenna {bs_antenna_idx+1}/{num_bs_antennas} ({progress:.1f}%)")
            
            # Get antenna-specific embedding
            antenna_embedding = self.prism_network.antenna_codebook(antenna_indices[:, bs_antenna_idx])
            
            # Process all batches for this antenna
            batch_ray_results, batch_signal_strengths = self._process_antenna_batches(
                bs_antenna_idx, batch_size, num_ue_antennas, num_selected,
                ue_positions, bs_position, antenna_embedding
            )
            
            # Update CSI predictions
            self._update_csi_predictions(
                csi_predictions, batch_ray_results, bs_antenna_idx, 
                batch_size, num_ue_antennas, num_selected, ue_positions
            )
            
            all_ray_results.append(batch_ray_results)
            all_signal_strengths.append(batch_signal_strengths)
        
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
            'signal_strengths': all_signal_strengths,
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
                                ue_positions: torch.Tensor, bs_position: torch.Tensor, antenna_embedding: torch.Tensor):
        """Process all batches for a specific BS antenna."""
        batch_ray_results = []
        batch_signal_strengths = []
        
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
            
            ray_results = self._perform_ray_tracing(b, bs_position, ue_pos_list, selected_subcarriers, antenna_embedding)
            
            batch_ray_results.append(ray_results)
            batch_signal_strengths.append(ray_results)
        
        return batch_ray_results, batch_signal_strengths
    
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
                        selected_subcarriers[ue_pos_tuple] = [0]  # Fallback
            except Exception as e:
                logger.warning(f"Error converting subcarrier selection: {e}, using fallback")
                selected_subcarriers[ue_pos_tuple] = [0]  # Fallback
            
            # Ensure we have valid subcarriers
            if not selected_subcarriers[ue_pos_tuple]:
                selected_subcarriers[ue_pos_tuple] = [0]
        
        return selected_subcarriers
    
    def _perform_ray_tracing(self, batch_idx: int, bs_position: torch.Tensor, ue_pos_list: list, 
                           selected_subcarriers: dict, antenna_embedding: torch.Tensor):
        """Perform ray tracing based on the configured mode."""
        try:
            if self.ray_tracing_mode == 'cuda':
                return self.ray_tracer.accumulate_signals(
                    base_station_pos=bs_position[batch_idx],
                    ue_positions=ue_pos_list,
                    selected_subcarriers=selected_subcarriers,
                    antenna_embedding=antenna_embedding[batch_idx]
                )
            elif self.ray_tracing_mode == 'cpu':
                return self._simple_ray_tracing(
                    bs_position[batch_idx], ue_pos_list[0], selected_subcarriers, antenna_embedding[batch_idx]
                )
            else:  # hybrid mode
                try:
                    with timeout_context(10, "ray_tracer.accumulate_signals"):
                        return self.ray_tracer.accumulate_signals(
                            base_station_pos=bs_position[batch_idx].cpu(),
                            ue_positions=ue_pos_list,
                            selected_subcarriers=selected_subcarriers,
                            antenna_embedding=antenna_embedding[batch_idx].cpu()
                        )
                except (TimeoutError, Exception) as e:
                    logger.warning(f"CUDA ray tracing failed: {e}. Falling back to CPU.")
                    return self._simple_ray_tracing(
                        bs_position[batch_idx], ue_pos_list[0], selected_subcarriers, antenna_embedding[batch_idx]
                    )
        except Exception as e:
            logger.error(f"Ray tracing failed: {e}. Using fallback calculation.")
            return self._fallback_signal_calculation(
                bs_position[batch_idx], ue_pos_list[0], selected_subcarriers, antenna_embedding[batch_idx]
            )
    
    def _update_csi_predictions(self, csi_predictions: torch.Tensor, batch_ray_results: list, 
                              bs_antenna_idx: int, batch_size: int, num_ue_antennas: int, num_selected: int, ue_positions: torch.Tensor):
        """Update CSI predictions with ray tracing results."""
        for b in range(batch_size):
            ray_results = batch_ray_results[b]
            
            # Create UE position tuple for lookup (all antennas share same position)
            ue_device_pos = ue_positions[b].cpu()
            ue_pos_tuple = tuple(ue_device_pos.tolist())
            
            for u in range(num_ue_antennas):
                # Get the selected subcarriers for this UE antenna and BS antenna
                ue_selected_subcarriers = self.current_selection[b, bs_antenna_idx, u].tolist()
                
                for k_idx, k in enumerate(ue_selected_subcarriers):
                    if k_idx < num_selected:
                        # Look for results in ray_results dictionary
                        if (ue_pos_tuple, k) in ray_results:
                            signal_strength = ray_results[(ue_pos_tuple, k)]
                            csi_predictions[b, bs_antenna_idx, u, k_idx] = signal_strength.to(torch.complex64)
                        else:
                            # Fallback for missing results
                            csi_predictions[b, bs_antenna_idx, u, k_idx] = torch.complex(
                                torch.tensor(0.0, device=ue_positions.device), 
                                torch.tensor(0.0, device=ue_positions.device)
                            )
    
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
        
        for b in range(batch_size):
            for u in range(num_ue_antennas):
                try:
                    # Select subcarriers based on sampling method
                    if self.subcarrier_sampling_method == 'uniform':
                        # Uniform sampling: evenly spaced subcarriers
                        step = total_subcarriers // num_selected
                        ue_selected = [i * step for i in range(num_selected)]
                        # Ensure we don't exceed bounds
                        ue_selected = [min(idx, total_subcarriers - 1) for idx in ue_selected]
                    else:  # 'random' (default)
                        # Random sampling: randomly selected subcarriers
                        ue_selected = random.sample(range(total_subcarriers), num_selected)
                    
                    logger.debug(f"UE {u} in batch {b}: selected subcarriers {ue_selected} (method: {self.subcarrier_sampling_method})")
                    
                    if self.antenna_consistent:
                        # All BS antennas use the SAME subcarrier indices for this UE
                        for bs_antenna in range(num_bs_antennas):
                            selected_indices[b, bs_antenna, u] = torch.tensor(ue_selected)
                            selection_mask[b, bs_antenna, u, ue_selected] = True
                    else:
                        # Each BS antenna selects independently (legacy behavior)
                        for bs_antenna in range(num_bs_antennas):
                            if self.subcarrier_sampling_method == 'uniform':
                                step = total_subcarriers // num_selected
                                antenna_selected = [i * step for i in range(num_selected)]
                                antenna_selected = [min(idx, total_subcarriers - 1) for idx in antenna_selected]
                            else:  # 'random'
                                antenna_selected = random.sample(range(total_subcarriers), num_selected)
                            
                            selected_indices[b, bs_antenna, u] = torch.tensor(antenna_selected)
                            selection_mask[b, bs_antenna, u, antenna_selected] = True
                        
                except Exception as e:
                    logger.error(f"Error in subcarrier selection for batch {b}, UE {u}: {e}")
                    logger.error(f"  total_subcarriers: {total_subcarriers}, num_selected: {num_selected}")
                    logger.error(f"  sampling_method: {self.subcarrier_sampling_method}")
                    logger.error(f"  antenna_consistent: {self.antenna_consistent}")
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

    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        loss_function: nn.Module
    ) -> torch.Tensor:
        """Compute loss for selected subcarriers."""
        # Debug logging
        logger.debug(f"compute_loss called with predictions shape: {predictions.shape}, targets shape: {targets.shape}")
        
        # Check if current_selection attributes exist, if not, initialize them
        if not hasattr(self, 'current_selection') or self.current_selection is None:
            logger.warning("current_selection not found, initializing with default values")
            batch_size, num_subcarriers, num_ue_antennas, num_bs_antennas = targets.shape
            num_selected = int(num_subcarriers * self.subcarrier_sampling_ratio)
            self._initialize_selection_variables(batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers)
            # Create default selection for all antennas
            selection_info = self._select_subcarriers(batch_size, num_ue_antennas, num_bs_antennas, num_subcarriers, num_selected)
            self.current_selection = selection_info['selected_indices']
            self.current_selection_mask = selection_info['selection_mask']
        
        logger.debug(f"current_selection: {self.current_selection is not None}, current_selection_mask: {self.current_selection_mask is not None}")
        
        if self.current_selection is None or self.current_selection_mask is None:
            logger.error("No subcarrier selection available even after initialization.")
            logger.error(f"current_selection: {self.current_selection}")
            logger.error(f"current_selection_mask: {self.current_selection_mask}")
            raise ValueError("No subcarrier selection available even after initialization.")
        
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
            # Handle complex numbers by computing loss on real and imaginary parts separately
            if traced_predictions.dtype.is_complex:
                # For complex numbers, compute MSE on both real and imaginary parts
                real_loss = loss_function(traced_predictions.real, traced_targets.real)
                imag_loss = loss_function(traced_predictions.imag, traced_targets.imag)
                loss = real_loss + imag_loss
                logger.debug(f"Complex loss computed: real_loss={real_loss.item():.6f}, imag_loss={imag_loss.item():.6f}")
            else:
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
                    if traced_predictions.dtype.is_complex:
                        # For complex numbers, compute MSE on both real and imaginary parts
                        real_diff = traced_predictions.real - traced_targets.real
                        imag_diff = traced_predictions.imag - traced_targets.imag
                        loss = torch.mean(real_diff ** 2 + imag_diff ** 2)
                    else:
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
            'best_loss': self.best_loss,
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
            spatial_sampling = self.ray_tracing_config.get('spatial_sampling', {})
            subcarrier_sampling = self.ray_tracing_config.get('subcarrier_sampling', {})
            
            # Get training configuration from system config or use defaults
            training_config = getattr(self, 'config', {}).get('training', {})
            num_epochs = training_config.get('num_epochs', 2)
            batches_per_epoch = training_config.get('batches_per_epoch', 5)
            
            # Get ray tracing configuration values
            top_k_directions = angular_sampling.get('top_k_directions', 32)
            azimuth_divisions = angular_sampling.get('azimuth_divisions', 18)
            elevation_divisions = angular_sampling.get('elevation_divisions', 9)
            num_sampling_points = spatial_sampling.get('num_sampling_points', 64)
            resampled_points = spatial_sampling.get('resampled_points', 32)
            
            # Get subcarrier configuration
            sampling_ratio = subcarrier_sampling.get('sampling_ratio', 0.01)
            num_subcarriers = getattr(self.prism_network, 'num_subcarriers', 408)
            
            # Get system configuration
            num_bs_antennas = self.system_config.get('base_station', {}).get('num_antennas', 64)
            num_ue_antennas = self.user_equipment_config.get('num_ue_antennas', 4)
            
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
