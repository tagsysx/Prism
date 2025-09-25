"""
Prism Training Interface (Completely Rewritten)

This module provides a modernized training interface that integrates:
1. PrismNetwork for feature extraction from ray tracing
2. NNRayTracer for neural CSI prediction
3. Efficient subcarrier selection and CSI computation
4. Streamlined checkpoint management

This is a complete rewrite based on the new architecture - no legacy compatibility.
"""

import torch
import torch.nn as nn
import logging
import os
import json
import time
from typing import Dict, Optional, List, Tuple, Any, Union
import contextlib
from pathlib import Path
import numpy as np

from .networks.prism_network import PrismNetwork
from .tracers import LowRankRayTracer

logger = logging.getLogger(__name__)


# Custom exceptions for better error handling
class PrismTrainingError(Exception):
    """Base exception for Prism training interface errors."""
    pass


class ConfigurationError(PrismTrainingError):
    """Raised when configuration is invalid or missing."""
    pass


class RayTracingError(PrismTrainingError):
    """Raised when ray tracing operations fail."""
    pass


class PrismTrainingInterface(nn.Module):
    """
    Modernized training interface for Prism neural ray tracing system.
    
    Key Features:
    - Integration with neural network-based NNRayTracer
    - Two-stage architecture: PrismNetwork + TraceNetwork
    - Attention-based feature fusion
    - End-to-end learnable pipeline
    - Enhanced checkpoint system
    """
    
    def __init__(
        self,
        prism_network: PrismNetwork,
        config: Dict[str, Any],
        checkpoint_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.config = config
        self.prism_network = prism_network
        
        # Device setup
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Extract configuration sections
        self.system_config = config.get('system', {})
        self.training_config = config.get('training', {})
        
        # Get mixed precision setting from PrismNetwork
        self.use_mixed_precision = getattr(prism_network, 'use_mixed_precision', False)
        self.ue_config = config.get('user_equipment', {})
        self.data_config = config.get('input', {})
        
        # LowRankTransformer configuration
        self.transformer_config = config.get('transformer', {})
        self.use_transformer_enhancement = self.transformer_config.get('use_enhancement', False)
        
        # CSI correction configuration
        calibration_config = self.data_config.get('calibration', {})
        self.reference_subcarrier_index = calibration_config.get('reference_subcarrier_index', 0)
        
        # Validate required configurations
        self._validate_config()
        
        # Initialize LowRankRayTracer for CSI prediction
        self.ray_tracer = LowRankRayTracer(prism_network=self.prism_network)
        
        # Move ray_tracer to device
        self.ray_tracer = self.ray_tracer.to(self.device)
        
        # Subcarrier configuration - we use base subcarriers from PrismNetwork
        # Set virtual subcarriers based on UE antenna count
        self.prism_network.set_virtual_subcarriers(self.ue_config.get('ue_antenna_count', 1))
        # Set subcarrier configuration
        self.num_subcarriers = self.prism_network.num_subcarriers  # Base subcarriers per UE antenna (64)
        self.num_virtual_subcarriers = self.prism_network.num_virtual_subcarriers  # Total subcarriers (512)
        
        # UE antenna configuration
        self.ue_antenna_count = self.ue_config.get('ue_antenna_count', 1)
        
        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('./checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.current_batch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # No subcarrier selection cache needed - we use all subcarriers
        
        logger.info(f"🚀 PrismTrainingInterface initialized")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - Subcarriers: {self.num_subcarriers} total (all used)")
        logger.info(f"   - UE antenna count: {self.ue_antenna_count}")
        logger.info(f"   - Checkpoint dir: {self.checkpoint_dir}")
    
    def _validate_config(self):
        """Validate required configuration parameters."""
        required_paths = [
            ('system', 'device'),
            ('user_equipment', 'ue_antenna_count')
        ]
        
        for path in required_paths:
            current = self.config
            try:
                for key in path:
                    current = current[key]
            except KeyError:
                raise ConfigurationError(f"Missing required configuration: {'.'.join(path)}")
        
        # Validate UE antenna count
        ue_antenna_count = self.config.get('user_equipment', {}).get('ue_antenna_count', 1)
        if ue_antenna_count < 1:
            raise ConfigurationError(f"Invalid ue_antenna_count: {ue_antenna_count}. Must be >= 1")
        
        logger.debug("Configuration validation completed successfully")
    
    def _debug_check_tensors(self, stage_name: str, tensors: Dict[str, torch.Tensor]):
        """Debug utility to check tensors for NaN/Inf values and report statistics."""
        logger.info(f"🔍 Debug check at stage: {stage_name}")
        
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                logger.info(f"   {name}: Not a tensor (type: {type(tensor)})")
                continue
                
            # Basic tensor info
            nan_count = torch.isnan(tensor).sum().item() if tensor.dtype.is_floating_point or tensor.dtype.is_complex else 0
            inf_count = torch.isinf(tensor).sum().item() if tensor.dtype.is_floating_point or tensor.dtype.is_complex else 0
            total_elements = tensor.numel()
            
            logger.info(f"   {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            logger.info(f"     Total elements: {total_elements:,}")
            
            if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
                logger.info(f"     NaN count: {nan_count:,} ({nan_count/total_elements*100:.2f}%)")
                logger.info(f"     Inf count: {inf_count:,} ({inf_count/total_elements*100:.2f}%)")
                
                # Get finite value statistics
                finite_mask = torch.isfinite(tensor)
                finite_count = finite_mask.sum().item()
                logger.info(f"     Finite count: {finite_count:,} ({finite_count/total_elements*100:.2f}%)")
                
                if finite_count > 0:
                    finite_values = tensor[finite_mask]
                    if tensor.dtype.is_complex:
                        real_part = finite_values.real
                        imag_part = finite_values.imag
                        logger.info(f"     Real: min={real_part.min():.6f}, max={real_part.max():.6f}, mean={real_part.mean():.6f}")
                        logger.info(f"     Imag: min={imag_part.min():.6f}, max={imag_part.max():.6f}, mean={imag_part.mean():.6f}")
                    else:
                        logger.info(f"     Values: min={finite_values.min():.6f}, max={finite_values.max():.6f}, mean={finite_values.mean():.6f}")
                
                # Raise error if we find invalid values
                if nan_count > 0 or inf_count > 0:
                    error_msg = f"❌ Invalid values found in {name} at stage '{stage_name}': {nan_count} NaN, {inf_count} Inf"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                # For integer/boolean tensors, show value range
                logger.info(f"     Values: min={tensor.min():.6f}, max={tensor.max():.6f}")
    
    def _debug_check_model_weights(self):
        """Check model weights for NaN/Inf values."""
        logger.info("🔍 Checking model weights for NaN/Inf values...")
        
        total_params = 0
        nan_params = 0
        inf_params = 0
        
        for name, param in self.prism_network.named_parameters():
            if param.requires_grad:
                param_nan = torch.isnan(param).sum().item()
                param_inf = torch.isinf(param).sum().item()
                total_params += param.numel()
                nan_params += param_nan
                inf_params += param_inf
                
                if param_nan > 0 or param_inf > 0:
                    logger.error(f"❌ Invalid weights in {name}: {param_nan} NaN, {param_inf} Inf")
                    raise ValueError(f"Model weights contain invalid values in {name}")
        
        logger.info(f"✅ Model weights check passed: {total_params:,} parameters, all finite")
    
    def forward(
        self, 
        ue_positions: torch.Tensor,     # [batch_size, 3]
        bs_positions: torch.Tensor,     # [batch_size, 3]
        bs_antenna_indices: torch.Tensor,  # [batch_size, num_bs_antennas]
        return_intermediates: bool = False
    ) -> Dict[str, Union[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass with sample-by-sample processing for memory optimization.
        
        This method processes each sample individually to minimize memory usage, then applies
        batch CSI enhancement to all antenna pairs at once for efficiency.
        
        Processing Pipeline:
        1. Sample-by-sample processing: PrismNetwork + RayTracing
        2. Collect raw CSI from all BS antennas
        3. Batch CSI enhancement: Apply CSINetwork to all antenna pairs simultaneously
        4. Combine results and return final CSI predictions
        
        Args:
            ue_positions: UE positions tensor [batch_size, 3]
            bs_positions: BS positions tensor [batch_size, 3]
            bs_antenna_indices: BS antenna indices [batch_size, num_bs_antennas]
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Dictionary containing:
                - csi: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers_per_ue] (4D)
                - subcarrier_selection: None (all subcarriers used, no selection)
                - intermediates: Ray tracing intermediates (if requested)
        """
        # Quick input validation
        if torch.isnan(ue_positions).any() or torch.isnan(bs_positions).any():
            raise ValueError("Input data contains NaN values")
        
        batch_size = ue_positions.shape[0]
        num_bs_antennas = bs_antenna_indices.shape[1]
        device = ue_positions.device
        
        # Performance timing
        start_time = time.time()
        logger.debug(f"Sample-by-sample forward pass: batch_size={batch_size}, num_bs_antennas={num_bs_antennas}")
        
        # Initialize output containers
        batch_results = []
        all_intermediates = [] if return_intermediates else None
        
        # Process each sample individually to minimize memory usage
        for sample_idx in range(batch_size):
            logger.debug(f"🔄 Processing sample {sample_idx + 1}/{batch_size}")
            
            # Extract single sample data
            single_ue_position = ue_positions[sample_idx]  # [3]
            single_bs_position = bs_positions[sample_idx]  # [3]
            single_bs_antenna_indices = bs_antenna_indices[sample_idx].unsqueeze(0)  # [1, num_bs_antennas]
            
            # Process single sample with configurable mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_mixed_precision):
                sample_result = self._process_single_sample(
                    ue_position=single_ue_position,
                    bs_position=single_bs_position,
                    bs_antenna_indices=single_bs_antenna_indices,
                    return_intermediates=return_intermediates
                )
            
            batch_results.append(sample_result)
            
            # Collect intermediates if requested
            if return_intermediates and 'intermediates' in sample_result:
                all_intermediates.append(sample_result['intermediates'])
            
            # Memory cleanup after each sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Progress logging
            if batch_size > 4 and (sample_idx + 1) % max(1, batch_size // 4) == 0:
                memory_used = torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
                logger.debug(f"✅ Completed {sample_idx + 1}/{batch_size} samples, GPU memory: {memory_used:.1f} MB")
        
        # Combine results from all samples
        combined_csi = torch.cat([result['csi'] for result in batch_results], dim=0)
        # All samples use all subcarriers (no selection), so subcarrier_selection is None
        combined_selection = None
        
        # Apply CSI calibration (phase and amplitude normalization)
        combined_csi = self._csi_calibration(combined_csi)
        logger.debug("Applied CSI calibration (phase and amplitude normalization)")
        
        # Prepare final outputs
        outputs = {
            'csi': combined_csi,
            'subcarrier_selection': combined_selection
        }
        
        if return_intermediates and all_intermediates:
            outputs['intermediates'] = all_intermediates
        
        # Final performance logging
        total_time = time.time() - start_time
        logger.debug(f"🏁 Sample-by-sample forward pass completed in {total_time:.3f}s")
        logger.debug(f"   Output shape: {combined_csi.shape}")
        
        # Final memory check
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB
            logger.debug(f"   Final GPU memory: {final_memory:.1f} MB")
        
        # Aggressive cleanup of intermediate results
        del batch_results
        if all_intermediates:
            del all_intermediates
        
        # Final memory cleanup before returning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return outputs
    
    def _csi_calibration(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Apply full CSI correction (both phase and amplitude normalization).

        This method applies both phase and amplitude correction by dividing all subcarriers
        by the reference subcarrier for each CSI independently. This aligns both phase and 
        amplitude to the reference subcarrier.

        Args:
            csi: CSI tensor [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
            
        Returns:
            corrected_csi: Corrected CSI tensor with same shape as input
        """
        # Ensure reference index is within bounds
        reference_idx = self.reference_subcarrier_index
        if reference_idx >= csi.shape[-1]:
            reference_idx = 0
        
        # Apply correction per CSI independently
        corrected_csi = csi.clone()
        batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers = csi.shape
        
        for batch_idx in range(batch_size):
            for bs_idx in range(num_bs_antennas):
                for ue_idx in range(num_ue_antennas):
                    # Get CSI for this specific antenna pair
                    antenna_csi = csi[batch_idx, bs_idx, ue_idx, :]  # [num_subcarriers]
                    
                    # Get reference subcarrier value
                    reference_value = antenna_csi[reference_idx]
                    
                    # Avoid division by zero
                    if torch.abs(reference_value) < 1e-12:
                        # If reference is too small, use identity correction
                        corrected_csi[batch_idx, bs_idx, ue_idx, :] = antenna_csi
                    else:
                        # Apply full correction: CSI / reference
                        # This normalizes both phase and amplitude
                        corrected_csi[batch_idx, bs_idx, ue_idx, :] = antenna_csi / reference_value
        
        logger.debug(f"🔧 Applied full CSI calibration (phase and amplitude)")
        logger.debug(f"   Reference subcarrier index: {reference_idx}")
        logger.debug(f"   CSI shape: {csi.shape}")
        
        return corrected_csi
    
    def _process_single_sample(
        self, 
        ue_position: torch.Tensor,
        bs_position: torch.Tensor,
        bs_antenna_indices: torch.Tensor,
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single sample to minimize memory usage.
        
        This method implements the complete CSI prediction pipeline for a single sample:
        1. Input validation and preprocessing
        2. Frequency basis vector pre-computation (shared across antennas)
        3. BS antenna batch processing to reduce memory pressure
        4. For each BS antenna:
           a. PrismNetwork forward pass (spatial factors)
           b. Ray tracing with LowRankRayTracer
           c. CSI enhancement for all UE antennas of this BS antenna
           d. Memory cleanup
        5. Combine all BS antenna predictions
        6. Shape normalization and final formatting
        
        Key Optimization: CSI enhancement is applied per-BS-antenna to minimize memory usage,
        processing all UE antennas for each BS antenna together to avoid OOM errors.
        
        Args:
            ue_position: UE position tensor of shape [3]
            bs_position: BS position tensor of shape [3]
            bs_antenna_indices: BS antenna indices tensor of shape [1, num_bs_antennas]
            return_intermediates: Whether to return intermediate results
        
        Returns:
            Dictionary containing:
                - csi: [1, num_bs_antennas, num_ue_antennas, num_subcarriers_per_ue] (4D)
                - subcarrier_selection: None (all subcarriers used)
                - intermediates: Ray tracing intermediates (if requested)
        """
        device = next(self.prism_network.parameters()).device
        
        # ========================================
        # STEP 1: Input validation and preprocessing
        # ========================================
        # Basic validation - only check for invalid inputs (keep lightweight)
        if torch.isnan(ue_position).any() or torch.isnan(bs_position).any():
            raise ValueError("Input positions contain NaN values")
        # Note: Position values can be large (e.g., Chrissy dataset has values up to 3M+)
        # This is normal for real-world coordinate systems, so we don't warn about large values
        
        # No subcarrier selection needed - we predict all subcarriers
        
        # Initialize sample results
        sample_csi = []
        intermediates = [] if return_intermediates else None
        
        # ========================================
        # STEP 2: Memory optimization setup
        # ========================================
        # MEMORY OPTIMIZATION: Process BS antennas in smaller batches
        # Note: Even though CSI enhancement is now batched, we still need BS antenna batching
        # because PrismNetwork forward pass and Ray tracing are memory-intensive operations
        # that need to be processed individually for each BS antenna
        num_bs_antennas = bs_antenna_indices.shape[1]  # Number of BS antennas
        antenna_batch_size = self.training_config.get('antenna_batch_size', 8)  # Default to 8 BS antennas per batch
        
        logger.debug(f"📡 Processing {num_bs_antennas} BS antennas in batches of {antenna_batch_size}")
        logger.debug(f"📡 UE antenna configuration: {self.ue_antenna_count} UE antennas per BS antenna")
        
        # ========================================
        # STEP 3: Frequency basis vector pre-computation
        # ========================================
        # Pre-compute frequency basis vectors (shared across all antennas)
        # This avoids redundant computation for each antenna
        start_time = time.time()
        frequency_basis_vectors = self.prism_network.frequency_codebook()
        freq_compute_time = time.time() - start_time
        logger.debug(f"🔧 Pre-computed frequency basis vectors: {frequency_basis_vectors.shape} in {freq_compute_time:.4f}s")
        logger.debug(f"🔧 FrequencyCodebook num_subcarriers: {self.prism_network.frequency_codebook.num_subcarriers}")
        logger.debug(f"🔧 PrismNetwork num_virtual_subcarriers: {self.prism_network.num_virtual_subcarriers}")
        
        # ========================================
        # STEP 4: BS Antenna batch processing loop
        # ========================================
        # Process BS antennas in batches to reduce memory pressure
        for batch_start in range(0, num_bs_antennas, antenna_batch_size):
            batch_end = min(batch_start + antenna_batch_size, num_bs_antennas)
            batch_size_actual = batch_end - batch_start
            
            logger.debug(f"  📦 Processing BS antenna batch {batch_start//antenna_batch_size + 1}: BS antennas {batch_start}-{batch_end-1}")
            logger.debug(f"🔍 Memory at start of BS antenna batch {batch_start//antenna_batch_size + 1}")
            if torch.cuda.is_available():
                logger.debug(f"   GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
            
            # ========================================
            # STEP 4a: Individual BS antenna processing
            # ========================================
            # Process each BS antenna in current batch
            for bs_antenna_idx in range(batch_start, batch_end):
                # Get current BS antenna index
                current_bs_antenna_indices = bs_antenna_indices[:, bs_antenna_idx:bs_antenna_idx+1]  # [1, 1]
                bs_antenna_index_int = current_bs_antenna_indices.squeeze().item()  # Convert to int
                
                # ========================================
                # STEP 4a.1: PrismNetwork forward pass
                # ========================================
                # Call PrismNetwork.forward() for spatial factors only (frequency basis pre-computed)
                logger.debug(f"🔍 Memory before PrismNetwork forward - BS antenna {bs_antenna_idx}")
                if torch.cuda.is_available():
                    logger.debug(f"   GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
                
                prism_outputs = self.prism_network.forward(
                    bs_position=bs_position,
                    ue_position=ue_position,
                    antenna_index=bs_antenna_index_int,  # BS antenna index
                    selected_subcarriers=None,
                    return_intermediates=return_intermediates
                )
                
                logger.debug(f"✅ PrismNetwork forward completed - BS antenna {bs_antenna_idx}")
                if torch.cuda.is_available():
                    logger.debug(f"   GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
                
                # Extract spatial parameters from PrismNetwork output
                attenuation_vectors = prism_outputs['attenuation_vectors']    # (num_directions, num_points, R)
                radiation_vectors = prism_outputs['radiation_vectors']        # (num_directions, num_points, R)
                sampled_positions = prism_outputs['sampled_positions']        # (num_directions, num_points, 3)
                directions = prism_outputs['directions']                      # (num_directions, 3)
                # Note: frequency_basis_vectors already pre-computed above
                
                # ========================================
                # STEP 4a.2: Ray tracing with LowRankRayTracer
                # ========================================
                # Ray tracing with LowRankRayTracer using pre-computed frequency basis
                logger.debug(f"🔍 Memory before ray tracing - BS antenna {bs_antenna_idx}")
                if torch.cuda.is_available():
                    logger.debug(f"   GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
                
                logger.debug(f"🔧 Ray tracing inputs: frequency_basis_vectors.shape={frequency_basis_vectors.shape}")
                ray_trace_results = self.ray_tracer.trace_rays(
                    attenuation_vectors=attenuation_vectors,      # (num_directions, num_points, R)
                    radiation_vectors=radiation_vectors,          # (num_directions, num_points, R)
                    frequency_basis_vectors=frequency_basis_vectors  # (num_subcarriers, R) - pre-computed
                )
                
                logger.debug(f"✅ Ray tracing completed - BS antenna {bs_antenna_idx}")
                if torch.cuda.is_available():
                    logger.debug(f"   GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
                
                # Quick validation for ray tracing results
                if 'csi' in ray_trace_results:
                    csi_pred = ray_trace_results['csi']
                    if torch.isnan(csi_pred).any():
                        raise ValueError(f"NaN detected in ray trace output for BS antenna {bs_antenna_idx}")
                    if torch.isinf(csi_pred).any():
                        raise ValueError(f"Inf detected in ray trace output for BS antenna {bs_antenna_idx}")
                
                # Collect intermediate results if requested (lightweight)
                if return_intermediates:
                    intermediates.append({
                        'bs_antenna_idx': bs_antenna_idx,
                        'bs_antenna_index_int': bs_antenna_index_int,
                        'prism_outputs_keys': list(prism_outputs.keys()),  # Only keys, not full data
                        'ray_trace_results_keys': list(ray_trace_results.keys())  # Only keys, not full data
                    })
                
                # Extract predictions for this BS antenna
                bs_antenna_predictions = ray_trace_results['csi']  # [num_virtual_subcarriers] from ray tracing
                
                # Debug: Check bs_antenna_predictions shape
                logger.debug(f"🔍 bs_antenna_predictions shape: {bs_antenna_predictions.shape}")
                logger.debug(f"🔍 ray_trace_results keys: {list(ray_trace_results.keys())}")
                
                # ========================================
                # STEP 4a.3: Store raw CSI for batch enhancement
                # ========================================
                # Ray tracing returns [num_virtual_subcarriers], we need to convert to [num_ue_antennas, subcarriers_per_ue]
                # where num_virtual_subcarriers = num_subcarriers × num_ue_antennas
                
                # Ensure bs_antenna_predictions is 1D: [num_virtual_subcarriers]
                if bs_antenna_predictions.dim() > 1:
                    bs_antenna_predictions = bs_antenna_predictions.squeeze()  # Remove extra dimensions
                    logger.debug(f"🔧 Squeezed bs_antenna_predictions to 1D: {bs_antenna_predictions.shape}")
                
                # Calculate subcarriers per UE antenna
                total_subcarriers = bs_antenna_predictions.shape[0]  # num_virtual_subcarriers
                subcarriers_per_ue = total_subcarriers // self.ue_antenna_count
                
                if total_subcarriers % self.ue_antenna_count != 0:
                    raise ValueError(f"Cannot evenly divide {total_subcarriers} subcarriers among {self.ue_antenna_count} UE antennas")
                
                logger.debug(f"🔧 Total subcarriers: {total_subcarriers}, UE antennas: {self.ue_antenna_count}, Subcarriers per UE: {subcarriers_per_ue}")
                
                # Reshape to [num_ue_antennas, subcarriers_per_ue]
                bs_antenna_predictions_reshaped = bs_antenna_predictions.view(self.ue_antenna_count, subcarriers_per_ue)
                logger.debug(f"🔧 Reshaped to [num_ue_antennas, subcarriers_per_ue]: {bs_antenna_predictions_reshaped.shape}")
                
                # ========================================
                # STEP 4a.3: Apply CSI enhancement to all UE antennas for this BS antenna
                # ========================================
                logger.debug(f"🔍 Memory before CSI enhancement - BS antenna {bs_antenna_idx}")
                if torch.cuda.is_available():
                    logger.debug(f"   GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
                
                # Convert to 4D format: [1, 1, num_ue_antennas, subcarriers_per_ue]
                bs_antenna_csi_4d = bs_antenna_predictions_reshaped.unsqueeze(0).unsqueeze(0)
                logger.debug(f"🔧 CSI enhancement input shape: {bs_antenna_csi_4d.shape}")
                
                # Apply CSI enhancement to all UE antennas for this BS antenna at once
                enhanced_bs_antenna_csi = self.prism_network.enhance_csi(bs_antenna_csi_4d)
                logger.debug(f"🔧 CSI enhancement output shape: {enhanced_bs_antenna_csi.shape}")
                
                logger.debug(f"✅ CSI enhancement completed - BS antenna {bs_antenna_idx}")
                if torch.cuda.is_available():
                    logger.debug(f"   GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
                
                # Store enhanced CSI for this BS antenna (4D format)
                sample_csi.append(enhanced_bs_antenna_csi.clone())  # Clone to avoid reference issues
                
                # Clear intermediate variables to free memory
                del bs_antenna_csi_4d, enhanced_bs_antenna_csi
                
                # ========================================
                # STEP 4a.4: Memory cleanup for this BS antenna
                # ========================================
                # Immediate memory cleanup for this BS antenna
                del prism_outputs, attenuation_vectors, radiation_vectors, sampled_positions, directions
                del ray_trace_results, bs_antenna_predictions
            
            # Memory cleanup after each BS antenna batch
            logger.debug(f"🔍 Memory at end of BS antenna batch {batch_start//antenna_batch_size + 1}")
            if torch.cuda.is_available():
                logger.debug(f"   GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.debug(f"✅ BS antenna batch {batch_start//antenna_batch_size + 1} completed and memory cleaned")
            if torch.cuda.is_available():
                logger.debug(f"   GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
        
        # ========================================
        # STEP 5: Combine all BS antenna predictions and apply batch CSI enhancement
        # ========================================
        logger.debug(f"🔍 Memory before combining all BS antenna predictions")
        if torch.cuda.is_available():
            logger.debug(f"   GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
        
        # Combine predictions from all BS antennas for this sample
        # Concatenate along BS antenna dimension: [1, num_bs_antennas, num_ue_antennas, subcarriers_per_ue]
        # Each BS antenna prediction is [1, 1, num_ue_antennas, subcarriers_per_ue], we need to concatenate them
        
        # Debug: Check sample_csi
        logger.debug(f"📊 sample_csi length: {len(sample_csi)} (should equal {num_bs_antennas})")
        if len(sample_csi) > 0:
            logger.debug(f"📊 First BS antenna prediction shape: {sample_csi[0].shape}")
            logger.debug(f"📊 All BS antenna prediction shapes: {[pred.shape for pred in sample_csi[:3]]}")
        
        if len(sample_csi) == 0:
            logger.error("❌ No CSI predictions generated!")
            # Create dummy prediction: [1, 1, num_ue_antennas, subcarriers_per_ue] (4D format)
            subcarriers_per_ue = self.num_subcarriers // self.ue_antenna_count
            dummy_prediction = torch.zeros(1, 1, self.ue_antenna_count, subcarriers_per_ue, device=device, dtype=torch.complex64)
            sample_csi = [dummy_prediction]
        
        # Each BS antenna prediction is [1, 1, num_ue_antennas, subcarriers_per_ue] (4D)
        # We need to stack them to get [1, num_bs_antennas, num_ue_antennas, subcarriers_per_ue]
        
        # Stack BS antenna predictions: [1, num_bs_antennas, num_ue_antennas, subcarriers_per_ue]
        stacked_csi = torch.cat(sample_csi, dim=1)  # Concatenate along BS antenna dimension
        logger.debug(f"📊 Stacked CSI shape: {stacked_csi.shape}")
        
        # Already in correct 4D format: [1, num_bs_antennas, num_ue_antennas, subcarriers_per_ue]
        combined_csi = stacked_csi
        logger.debug(f"📊 Combined CSI shape: {combined_csi.shape}")
        
        logger.debug(f"✅ All BS antenna predictions combined successfully")
        if torch.cuda.is_available():
            logger.debug(f"   GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
        
        # ========================================
        # STEP 5.1: CSI enhancement already applied per-antenna-pair in BS antenna loop
        # ========================================
        # CSI enhancement has already been applied to each antenna pair individually
        # in the BS antenna processing loop above, so no additional enhancement needed here
        logger.debug(f"✅ CSI enhancement already completed per-antenna-pair in BS antenna loop")
        
        # ========================================
        # STEP 6: Final CSI formatting and validation
        # ========================================
        # CSI enhancement has already been applied per-antenna-pair in the BS antenna loop above
        # Now we just need to validate and format the final result
        
        logger.debug(f"📊 combined_csi shape: {combined_csi.shape}")
        predicted_subcarriers_per_ue = combined_csi.shape[3]
        expected_subcarriers_per_ue = self.num_subcarriers  # Base subcarriers per UE antenna (already calculated)
        
        # Validate subcarrier count matches expected
        if predicted_subcarriers_per_ue != expected_subcarriers_per_ue:
            logger.warning(f"⚠️ Subcarrier count mismatch: predicted {predicted_subcarriers_per_ue} vs expected {expected_subcarriers_per_ue}")
            if predicted_subcarriers_per_ue > expected_subcarriers_per_ue:
                # Truncate predictions to match target subcarrier count
                combined_csi = combined_csi[:, :, :, :expected_subcarriers_per_ue]
                logger.debug(f"📐 Truncated predictions from {predicted_subcarriers_per_ue} to {expected_subcarriers_per_ue} subcarriers")
            else:
                # Pad with zeros if needed
                padding_size = expected_subcarriers_per_ue - predicted_subcarriers_per_ue
                padding = torch.zeros(combined_csi.shape[0], combined_csi.shape[1], combined_csi.shape[2], padding_size, 
                                    dtype=combined_csi.dtype, device=combined_csi.device)
                combined_csi = torch.cat([combined_csi, padding], dim=3)
                logger.debug(f"📐 Padded predictions from {predicted_subcarriers_per_ue} to {expected_subcarriers_per_ue} subcarriers")
        
        logger.debug(f"📐 Final CSI shape: {combined_csi.shape} (format: [batch, bs_antennas, ue_antennas, subcarriers_per_ue])")
        
        # ========================================
        # STEP 7: Prepare final result and cleanup
        # ========================================
        # Prepare result for this sample
        result = {
            'csi': combined_csi,
            'subcarrier_selection': None  # All subcarriers used, no selection needed
        }
        
        if return_intermediates:
            result['intermediates'] = intermediates
        
        # Cleanup intermediate data to free memory
        del sample_csi
        if intermediates:
            del intermediates
            
        return result
    
    
    

    def compute_loss(
        self, 
        predictions: torch.Tensor,  # [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
        targets: torch.Tensor,      # [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
        loss_function: nn.Module,
        validation_mode: bool = False
    ) -> torch.Tensor:
        """
        Compute loss on all subcarriers using vectorized operations.
        
        Since we now predict all subcarriers, loss is computed on the entire tensor
        for maximum accuracy and simplified processing.
        
        Args:
            predictions: Model predictions [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
            targets: Ground truth targets [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
            loss_function: Loss function module (should accept dict input with 'predictions' and 'targets' keys)
            validation_mode: If True, additional gradient checks are skipped
            
        Returns:
            loss: Computed loss tensor with gradient tracking (unless validation_mode=True)
            
        Raises:
            ValueError: If input shapes don't match or loss function returns invalid type
        """
        # Debug: Log tensor shapes and statistics for troubleshooting
        logger.info(f"🔍 Loss computation - predictions shape: {predictions.shape}, targets shape: {targets.shape}")
        
        # Log tensor statistics to understand the magnitude
        # Handle complex tensors by using magnitude
        if torch.is_complex(predictions) or torch.is_complex(targets):
            pred_magnitude = torch.abs(predictions)
            target_magnitude = torch.abs(targets)
            logger.info(f"📊 Prediction stats (magnitude): min={pred_magnitude.min():.6f}, max={pred_magnitude.max():.6f}, mean={pred_magnitude.mean():.6f}")
            logger.info(f"📊 Target stats (magnitude): min={target_magnitude.min():.6f}, max={target_magnitude.max():.6f}, mean={target_magnitude.mean():.6f}")
        else:
            logger.info(f"📊 Prediction stats: min={predictions.min():.6f}, max={predictions.max():.6f}, mean={predictions.mean():.6f}")
            logger.info(f"📊 Target stats: min={targets.min():.6f}, max={targets.max():.6f}, mean={targets.mean():.6f}")
        
        # Validate input shapes
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
        
        # Validate tensor shape: [batch_size, num_bs_antennas, num_ue_antennas, num_subcarriers]
        if predictions.dim() != 4:
            raise ValueError(f"Expected 4D tensor [batch, bs_antennas, ue_antennas, subcarriers], got {predictions.shape}")
        
        batch_size = predictions.shape[0]
        num_bs_antennas = predictions.shape[1]
        num_ue_antennas = predictions.shape[2]
        num_subcarriers = predictions.shape[3]
        device = predictions.device
        
        # ========================================
        # STEP 1: Check for zero targets and create valid mask
        # ========================================
        # Create mask for non-zero targets (valid CSI samples)
        if torch.is_complex(targets):
            # For complex tensors, check if magnitude is non-zero
            target_magnitude = torch.abs(targets)
            valid_mask = target_magnitude > 1e-12  # Threshold for "non-zero"
        else:
            # For real tensors, check if absolute value is non-zero
            valid_mask = torch.abs(targets) > 1e-12
        
        # Count valid and invalid samples
        total_samples = targets.numel()
        valid_samples = valid_mask.sum().item()
        invalid_samples = total_samples - valid_samples
        
        logger.info(f"🔍 Target validation: {valid_samples:,} valid samples, {invalid_samples:,} zero/invalid samples")
        logger.info(f"📊 Valid sample ratio: {valid_samples/total_samples*100:.2f}%")
        
        # Check if we have any valid samples to compute loss on
        if valid_samples == 0:
            logger.warning("⚠️ All target CSI values are zero - skipping loss computation")
            # Return a small positive loss to prevent training crash
            return torch.tensor(1e-6, requires_grad=True, device=device)
        
        # If we have some invalid samples, log detailed information
        if invalid_samples > 0:
            logger.warning(f"⚠️ Found {invalid_samples:,} zero/invalid target samples ({invalid_samples/total_samples*100:.2f}%)")
            logger.warning("   These samples will be excluded from loss computation")
            
            # Log which dimensions have zero targets for debugging
            zero_batch_count = (target_magnitude.sum(dim=(1,2,3)) == 0).sum().item() if torch.is_complex(targets) else (torch.abs(targets).sum(dim=(1,2,3)) == 0).sum().item()
            zero_bs_count = (target_magnitude.sum(dim=(0,2,3)) == 0).sum().item() if torch.is_complex(targets) else (torch.abs(targets).sum(dim=(0,2,3)) == 0).sum().item()
            zero_ue_count = (target_magnitude.sum(dim=(0,1,3)) == 0).sum().item() if torch.is_complex(targets) else (torch.abs(targets).sum(dim=(0,1,3)) == 0).sum().item()
            
            logger.info(f"🔍 Zero target analysis:")
            logger.info(f"   Zero batches: {zero_batch_count}/{batch_size}")
            logger.info(f"   Zero BS antennas: {zero_bs_count}/{num_bs_antennas}")
            logger.info(f"   Zero UE antennas: {zero_ue_count}/{num_ue_antennas}")
        
        # Apply valid mask to both predictions and targets
        valid_predictions = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        logger.info(f"📊 Computing loss on {valid_samples:,} valid samples (filtered from {total_samples:,} total)")
        
        # ========================================
        # STEP 2: Check for invalid values in valid samples only
        # ========================================
        # Check for invalid values in the filtered valid samples
        if torch.isnan(valid_predictions).any() or torch.isinf(valid_predictions).any():
            # Detailed analysis for debugging
            nan_count = torch.isnan(valid_predictions).sum().item()
            inf_count = torch.isinf(valid_predictions).sum().item()
            finite_count = torch.isfinite(valid_predictions).sum().item()
            total_valid_elements = valid_predictions.numel()
            
            # Get statistics of finite values
            finite_mask = torch.isfinite(valid_predictions)
            if finite_mask.any():
                finite_values = valid_predictions[finite_mask]
                if torch.is_complex(valid_predictions):
                    finite_real = finite_values.real
                    finite_imag = finite_values.imag
                    stats_msg = (f"Finite values stats - Real: min={finite_real.min():.6f}, max={finite_real.max():.6f}, "
                               f"mean={finite_real.mean():.6f} | Imag: min={finite_imag.min():.6f}, "
                               f"max={finite_imag.max():.6f}, mean={finite_imag.mean():.6f}")
                else:
                    stats_msg = (f"Finite values stats - min={finite_values.min():.6f}, "
                               f"max={finite_values.max():.6f}, mean={finite_values.mean():.6f}")
            else:
                stats_msg = "No finite values found!"
            
            error_msg = (
                f"❌ Invalid values detected in valid predictions tensor!\n"
                f"   Valid samples: {valid_samples:,} out of {total_samples:,} total\n"
                f"   Valid elements: {total_valid_elements:,}\n"
                f"   NaN count: {nan_count:,} ({nan_count/total_valid_elements*100:.2f}%)\n"
                f"   Inf count: {inf_count:,} ({inf_count/total_valid_elements*100:.2f}%)\n"
                f"   Finite count: {finite_count:,} ({finite_count/total_valid_elements*100:.2f}%)\n"
                f"   {stats_msg}\n"
                f"   This indicates a serious numerical issue in the model forward pass.\n"
                f"   Check for: 1) Gradient explosion, 2) Division by zero, 3) Log of negative numbers,\n"
                f"   4) Network weight initialization, 5) Learning rate too high."
            )
            
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # ========================================
        # STEP 3: Compute loss on valid samples only
        # ========================================
        try:
            # Create filtered tensors with same shape as original for loss function compatibility
            # We need to maintain the original 4D structure for the loss function
            filtered_predictions = predictions.clone()
            filtered_targets = targets.clone()
            
            # Set invalid samples to zero (they won't contribute to loss)
            filtered_predictions[~valid_mask] = 0
            filtered_targets[~valid_mask] = 0
            
            logger.info(f"📊 Using filtered 4D CSI format [batch, bs_antennas, ue_antennas, subcarriers] for loss calculation")
            logger.info(f"📊 Computing loss on {valid_samples:,} valid samples out of {total_samples:,} total")
                
            # Apply CSI calibration to filtered predictions and targets
            predictions_calibrated = self._csi_calibration(filtered_predictions)
            targets_calibrated = self._csi_calibration(filtered_targets)
            
            # Prepare dictionary format for LossFunction
            # All losses now use phase-calibrated tensors to preserve spatial structure
            predictions_dict = {
                'csi': predictions_calibrated  # Phase-calibrated predictions (with zeros for invalid samples)
            }
            targets_dict = {
                'csi': targets_calibrated  # Phase-calibrated targets (with zeros for invalid samples)
            }
            
            # Call loss function and get detailed components
            loss, loss_components = loss_function(predictions_dict, targets_dict)
            
            # Log loss computation details
            logger.info(f"📊 Loss computed on {valid_samples:,} valid samples")
            if hasattr(loss_function, 'get_loss_components') and loss_components:
                logger.info(f"📊 Loss components: {loss_components}")
                
            # Log PDP loss status
            if hasattr(loss_function, 'pdp_enabled') and loss_function.pdp_enabled:
                logger.info(f"📊 PDP loss enabled with weight: {loss_function.pdp_weight}")
                if 'pdp_loss' in loss_components:
                    logger.info(f"📊 PDP loss value: {loss_components['pdp_loss']:.6f}")
            else:
                logger.info(f"📊 PDP loss disabled")
                
        except Exception as e:
            logger.error(f"Loss computation failed: {e}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Valid samples: {valid_samples:,} out of {total_samples:,} total")
            logger.error(f"Predictions dict keys: {list(predictions_dict.keys())}")
            logger.error(f"Targets dict keys: {list(targets_dict.keys())}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return a small positive loss to prevent training crash
            loss = torch.tensor(1e-6, requires_grad=True, device=device)
            
        # ========================================
        # STEP 4: Validate loss and return
        # ========================================
        # Validate loss
        if not isinstance(loss, torch.Tensor):
            raise ValueError(f"Loss function must return a torch.Tensor, got {type(loss)}")
            
        if not validation_mode and not loss.requires_grad:
            logger.warning("Loss tensor does not require gradients - this may affect training")
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Invalid loss value: {loss}")
            loss = torch.tensor(1e-6, requires_grad=True, device=device)
        
        # Final logging with valid sample information
        logger.info(f"✅ Loss computed: {loss.item():.6f}")
        logger.info(f"📊 Loss computed on {valid_samples:,} valid samples out of {total_samples:,} total ({valid_samples/total_samples*100:.2f}%)")
        
        # Clean up temporary variables
        del valid_predictions, valid_targets, filtered_predictions, filtered_targets
        
        return loss
    
    def save_checkpoint(
        self,
        filename: Optional[str] = None,
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
        additional_info: Optional[Dict] = None
    ):
        """Save training checkpoint with essential state."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"checkpoint_epoch_{self.current_epoch}_batch_{self.current_batch}_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': self.state_dict(),
            'prism_network_state_dict': self.prism_network.state_dict(),
            'training_state': {
                'epoch': self.current_epoch,
                'batch': self.current_batch,
                'best_loss': self.best_loss,
                'history': self.training_history
            },
            'config': self.config,
            'subcarrier_config': {
                'num_subcarriers': self.num_subcarriers,
                'note': 'All subcarriers used - no selection'
            },
            'timestamp': time.time()
        }
        
        # Transformer has been removed
        
        # Add optional states
        if optimizer_state is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer_state
        if scheduler_state is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler_state
        if additional_info is not None:
            checkpoint_data['additional_info'] = additional_info
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save human-readable info
        info_path = checkpoint_path.with_suffix('.json')
        info_data = {
            'epoch': self.current_epoch,
            'batch': self.current_batch,
            'best_loss': float(self.best_loss),
            'timestamp': checkpoint_data['timestamp'],
            'filename': filename,
            'config_summary': {
                'num_subcarriers': self.num_subcarriers,
                'subcarrier_usage': 'all_subcarriers',
                'device': str(self.device)
            }
        }
        
        with open(info_path, 'w') as f:
            json.dump(info_data, f, indent=2)
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        logger.info(f"📊 State: epoch={self.current_epoch}, batch={self.current_batch}, best_loss={self.best_loss:.6f}")
    
    
    def load_checkpoint(
        self, 
        checkpoint_path: str, 
        load_training_state: bool = True,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint and return auxiliary states.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_training_state: Whether to load training state
            load_optimizer: Whether to return optimizer state
            load_scheduler: Whether to return scheduler state
            
        Returns:
            Dictionary with loaded states
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint with security considerations
        try:
            # First try with weights_only=True for security
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            logger.info("🔒 Checkpoint loaded with weights_only=True (secure mode)")
        except Exception as e:
            # Fallback to weights_only=False for compatibility with complex checkpoint formats
            logger.warning(f"⚠️  Secure loading failed (expected for complex checkpoints): {str(e)[:100]}...")
            logger.info("🔓 Using compatibility mode (weights_only=False) - this is normal for Prism checkpoints")
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model states
        self.load_state_dict(checkpoint_data['model_state_dict'])
        self.prism_network.load_state_dict(checkpoint_data['prism_network_state_dict'])
        
        # Transformer has been removed
        
        # Load training state if requested
        if load_training_state and 'training_state' in checkpoint_data:
            training_state = checkpoint_data['training_state']
            self.current_epoch = training_state.get('epoch', 0)
            self.current_batch = training_state.get('batch', 0)
            self.best_loss = training_state.get('best_loss', float('inf'))
            self.training_history = training_state.get('history', [])
        
        # Prepare return data
        loaded_states = {
            'checkpoint_info': {
                'epoch': self.current_epoch,
                'batch': self.current_batch,
                'best_loss': self.best_loss,
                'timestamp': checkpoint_data.get('timestamp')
            }
        }
        
        # Add optional states to return
        if load_optimizer and 'optimizer_state_dict' in checkpoint_data:
            loaded_states['optimizer_state_dict'] = checkpoint_data['optimizer_state_dict']
        
        if load_scheduler and 'scheduler_state_dict' in checkpoint_data:
            loaded_states['scheduler_state_dict'] = checkpoint_data['scheduler_state_dict']
        
        if 'additional_info' in checkpoint_data:
            loaded_states['additional_info'] = checkpoint_data['additional_info']
        
        logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")
        logger.info(f"📊 Resumed at epoch {self.current_epoch}, batch {self.current_batch}")
        
        return loaded_states
    
    def update_training_state(self, epoch: int, batch: int, loss: float):
        """Update training state for progress tracking."""
        self.current_epoch = epoch
        self.current_batch = batch
        
        # Update best loss
        if loss < self.best_loss:
            self.best_loss = loss
            logger.debug(f"🎯 New best loss: {self.best_loss:.6f}")
        
        # Add to history
        self.training_history.append({
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'best_loss': self.best_loss,
            'timestamp': time.time()
        })
        
        # Trim history if too long (keep last 1000 entries)
        if len(self.training_history) > 1000:
            self.training_history = self.training_history[-1000:]
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        return {
            'training_state': {
                'current_epoch': self.current_epoch,
                'current_batch': self.current_batch,
                'best_loss': float(self.best_loss),
                'history_length': len(self.training_history)
            },
            'model_config': {
                'num_subcarriers': self.num_subcarriers,
                'ue_antenna_count': self.ue_antenna_count,
                'subcarrier_usage': 'all_subcarriers'
            },
            'system_info': {
                'device': str(self.device),
                'checkpoint_dir': str(self.checkpoint_dir),
                'prism_network_type': type(self.prism_network).__name__
            },
            'ray_tracer_info': {
                'azimuth_divisions': self.ray_tracer.azimuth_divisions,
                'elevation_divisions': self.ray_tracer.elevation_divisions,
                'total_directions': self.ray_tracer.total_directions,
                'max_ray_length': self.ray_tracer.max_ray_length,
                'num_sampling_points': self.ray_tracer.num_sampling_points
            },
            'transformer_info': {
                'use_enhancement': False,
                'transformer_available': False,
                'transformer_config': None
            }
        }
    
    
    def _generate_uniform_directions(self) -> torch.Tensor:
        """
        Generate uniform ray directions (same as PrismNetwork).
        
        Returns:
            directions: (num_directions, 3) - Unit direction vectors
                       These are PURE direction vectors (unit length), NOT including BS position offset.
                       To get actual ray positions: P(t) = bs_position + direction * t
        """
        # Calculate angular resolutions
        azimuth_resolution = 2 * torch.pi / self.prism_network.azimuth_divisions
        elevation_resolution = torch.pi / 2 / self.prism_network.elevation_divisions
        
        # Create grid of angles
        i, j = torch.meshgrid(
            torch.arange(self.prism_network.azimuth_divisions, dtype=torch.float32),
            torch.arange(self.prism_network.elevation_divisions, dtype=torch.float32),
            indexing='xy'
        )
        phi = i * azimuth_resolution     # Azimuth: 0° to 360°
        theta = j * elevation_resolution  # Elevation: 0° to 90°
        
        # Convert spherical coordinates to Cartesian unit vectors
        x = torch.cos(theta) * torch.cos(phi)
        y = torch.cos(theta) * torch.sin(phi)
        z = torch.sin(theta)
        
        # Flatten and stack to create (num_directions, 3) tensor
        return torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)
    
    def _sample_positions_along_rays(self, 
                                   bs_position: torch.Tensor,
                                   directions: torch.Tensor,
                                   max_length: float,
                                   num_points: int) -> torch.Tensor:
        """
        Sample positions along rays (same as PrismNetwork).
        
        Args:
            bs_position: Base station position (3,)
            directions: Ray directions (num_directions, 3)
            max_length: Maximum ray length
            num_points: Number of sampling points per ray
            
        Returns:
            sampled_positions: (num_directions, num_points, 3)
        """
        device = bs_position.device
        
        # Generate t values
        t_values = torch.linspace(0, max_length, num_points, dtype=torch.float32, device=device)
        
        # Ray equation: P(t) = bs_position + direction * t
        # Broadcasting: [1,1,3] + [R,1,3] * [1,P,1] → [R,P,3]
        sampled_positions = bs_position.view(1, 1, 3) + directions.unsqueeze(1) * t_values.view(1, -1, 1)
        
        return sampled_positions
