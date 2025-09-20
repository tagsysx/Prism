"""
Memory Optimization Module for Prism Neural Ray Tracing System

This module provides enhanced memory optimization strategies to support larger batch sizes
through aggressive memory management, micro-chunking, and gradient checkpointing.

Key Features:
- Ultra memory-optimized forward passes
- Progressive chunking and cleanup
- Gradient checkpointing integration
- Dynamic memory monitoring
- Micro-batching strategies
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)


class MemoryOptimizedMixin:
    """
    Mixin class providing memory optimization methods for PrismTrainingInterface.
    
    This class provides enhanced memory management strategies that can be applied
    to existing training interfaces to support larger batch sizes.
    """
    
    def forward_memory_optimized(
        self, 
        ue_positions: torch.Tensor,     # [batch_size, 3]
        bs_positions: torch.Tensor,     # [batch_size, 3]
        antenna_indices: torch.Tensor,  # [batch_size, num_bs_antennas]
        return_intermediates: bool = False
    ) -> Dict[str, Union[torch.Tensor, Dict[str, Any]]]:
        """
        Ultra memory-optimized forward pass with progressive chunking.
        
        Memory optimization strategies:
        1. Process samples in micro-batches
        2. Sub-chunk antenna processing
        3. Aggressive intermediate cleanup
        4. Dynamic memory monitoring
        
        Args:
            ue_positions: UE positions [batch_size, 3]
            bs_positions: BS positions [batch_size, 3]
            antenna_indices: Antenna indices [batch_size, num_bs_antennas]
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Dictionary with optimized CSI predictions and metadata
        """
        # Get configuration parameters
        training_config = getattr(self, 'training_config', {})
        ue_chunk_size = training_config.get('ue_chunk_size', 1)
        enable_checkpointing = training_config.get('enable_gradient_checkpointing', False)
        
        batch_size = ue_positions.shape[0]
        device = ue_positions.device
        
        logger.debug(f"🎯 Ultra memory-optimized forward: batch_size={batch_size}, ue_chunk_size={ue_chunk_size}")
        
        # Monitor initial memory
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated(device) / 1024**2
            logger.debug(f"💾 Initial GPU memory: {initial_memory:.1f} MB")
        
        all_results = []
        all_intermediates = [] if return_intermediates else None
        
        # Process UE samples in micro-chunks
        for chunk_start in range(0, batch_size, ue_chunk_size):
            chunk_end = min(chunk_start + ue_chunk_size, batch_size)
            chunk_size_actual = chunk_end - chunk_start
            
            logger.debug(f"🔄 Processing UE chunk {chunk_start//ue_chunk_size + 1}: samples {chunk_start}-{chunk_end-1}")
            
            # Extract chunk data
            chunk_ue_pos = ue_positions[chunk_start:chunk_end]
            chunk_bs_pos = bs_positions[chunk_start:chunk_end]
            chunk_antenna_idx = antenna_indices[chunk_start:chunk_end]
            
            # Process chunk with gradient checkpointing if enabled
            if enable_checkpointing and self.training:
                chunk_results = torch.utils.checkpoint.checkpoint(
                    self._process_ue_chunk_optimized,
                    chunk_ue_pos,
                    chunk_bs_pos,
                    chunk_antenna_idx,
                    return_intermediates,
                    use_reentrant=False
                )
            else:
                chunk_results = self._process_ue_chunk_optimized(
                    chunk_ue_pos,
                    chunk_bs_pos,
                    chunk_antenna_idx,
                    return_intermediates
                )
            
            all_results.append(chunk_results)
            
            if return_intermediates and 'intermediates' in chunk_results:
                all_intermediates.append(chunk_results['intermediates'])
            
            # Aggressive cleanup after each chunk
            del chunk_ue_pos, chunk_bs_pos, chunk_antenna_idx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
                # Memory monitoring
                current_memory = torch.cuda.memory_allocated(device) / 1024**2
                logger.debug(f"💾 Post-chunk GPU memory: {current_memory:.1f} MB")
        
        # Combine results efficiently
        combined_csi = torch.cat([result['csi_predictions'] for result in all_results], dim=0)
        combined_selection = all_results[0]['subcarrier_selection']  # Same for all
        
        # Final cleanup of intermediate results
        del all_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare output
        outputs = {
            'csi_predictions': combined_csi,
            'subcarrier_selection': combined_selection
        }
        
        if return_intermediates and all_intermediates:
            outputs['intermediates'] = all_intermediates
        
        return outputs
    
    def _process_ue_chunk_optimized(
        self,
        ue_positions: torch.Tensor,     # [chunk_size, 3]
        bs_positions: torch.Tensor,     # [chunk_size, 3] 
        antenna_indices: torch.Tensor,  # [chunk_size, num_bs_antennas]
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Process a chunk of UE samples with micro-antenna batching.
        
        Args:
            ue_positions: UE positions for chunk [chunk_size, 3]
            bs_positions: BS positions for chunk [chunk_size, 3]
            antenna_indices: Antenna indices for chunk [chunk_size, num_bs_antennas]
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Dictionary with combined CSI predictions for the chunk
        """
        chunk_size = ue_positions.shape[0]
        device = ue_positions.device
        
        chunk_results = []
        
        # Process each sample in the chunk
        for sample_idx in range(chunk_size):
            # Extract single sample
            single_ue_pos = ue_positions[sample_idx]
            single_bs_pos = bs_positions[sample_idx]
            single_antenna_idx = antenna_indices[sample_idx].unsqueeze(0)
            
            # Process with micro-antenna batching
            sample_result = self._process_single_sample_micro_batched(
                ue_position=single_ue_pos,
                bs_position=single_bs_pos,
                antenna_indices=single_antenna_idx,
                return_intermediates=return_intermediates
            )
            
            chunk_results.append(sample_result)
            
            # Micro cleanup
            del single_ue_pos, single_bs_pos, single_antenna_idx
            if sample_idx % 2 == 1 and torch.cuda.is_available():  # Every 2 samples
                torch.cuda.empty_cache()
        
        # Combine chunk results
        combined_csi = torch.cat([result['csi_predictions'] for result in chunk_results], dim=0)
        combined_selection = chunk_results[0]['subcarrier_selection']
        
        # Cleanup
        del chunk_results
        
        return {
            'csi_predictions': combined_csi,
            'subcarrier_selection': combined_selection
        }
    
    def _process_single_sample_micro_batched(
        self,
        ue_position: torch.Tensor,      # [3]
        bs_position: torch.Tensor,      # [3]
        antenna_indices: torch.Tensor,  # [1, num_bs_antennas]
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Process single sample with micro-antenna batching for ultra-low memory usage.
        
        Args:
            ue_position: Single UE position [3]
            bs_position: Single BS position [3]
            antenna_indices: Antenna indices [1, num_bs_antennas]
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Dictionary with CSI predictions for the sample
        """
        # Get configuration
        training_config = getattr(self, 'training_config', {})
        antenna_batch_size = training_config.get('antenna_batch_size', 4)  # Smaller batches
        
        num_antennas = antenna_indices.shape[1]
        device = ue_position.device
        
        logger.debug(f"🔋 Micro-batched processing: {num_antennas} antennas in batches of {antenna_batch_size}")
        
        all_csi_results = []
        subcarrier_selection = None
        
        # Process antennas in micro-batches
        for batch_start in range(0, num_antennas, antenna_batch_size):
            batch_end = min(batch_start + antenna_batch_size, num_antennas)
            batch_antenna_indices = antenna_indices[:, batch_start:batch_end]
            batch_size_actual = batch_end - batch_start
            
            logger.debug(f"  🔋 Micro-antenna batch: antennas {batch_start}-{batch_end-1} ({batch_size_actual} antennas)")
            
            # Process antenna micro-batch
            batch_csi_results = []
            
            for i in range(batch_size_actual):
                antenna_idx = batch_antenna_indices[0, i].item()
                
                # Trace single antenna with memory optimization
                trace_result = self.ray_tracer.trace_low_rank_csi(
                    ue_position=ue_position,
                    bs_position=bs_position,
                    antenna_index=antenna_idx,
                    return_intermediates=return_intermediates
                )
                
                batch_csi_results.append(trace_result['csi_matrix'])
                
                if subcarrier_selection is None:
                    subcarrier_selection = trace_result.get('subcarrier_selection')
                
                # Ultra-aggressive cleanup per antenna
                if torch.cuda.is_available() and i % 2 == 1:  # Every 2 antennas
                    torch.cuda.empty_cache()
            
            # Stack micro-batch results
            if batch_csi_results:
                batch_stacked = torch.stack(batch_csi_results, dim=0)  # [batch_size_actual, num_subcarriers, 1]
                all_csi_results.append(batch_stacked)
            
            # Cleanup micro-batch
            del batch_csi_results, batch_antenna_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Combine all antenna results
        if all_csi_results:
            combined_csi = torch.cat(all_csi_results, dim=0)  # [num_antennas, num_subcarriers, 1]
            combined_csi = combined_csi.unsqueeze(0)  # [1, num_antennas, num_subcarriers, 1]
        else:
            # Fallback
            combined_csi = torch.zeros(1, num_antennas, self.num_subcarriers, 1, 
                                     dtype=torch.complex64, device=device)
        
        # Final cleanup
        del all_csi_results
        
        return {
            'csi_predictions': combined_csi,
            'subcarrier_selection': subcarrier_selection
        }


class MemoryOptimizedPrismNetworkMixin:
    """
    Mixin class providing memory optimization methods for PrismNetwork.
    
    This class provides enhanced memory management for ray processing and
    network forward passes to support larger batch sizes.
    """
    
    def forward_ultra_memory_optimized(
        self,
        bs_position: torch.Tensor,
        ue_position: torch.Tensor,
        antenna_index: int,
        selected_subcarriers: Optional[List[int]] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Ultra memory-optimized forward pass with micro-ray chunking.
        
        Args:
            bs_position: Base station position [3]
            ue_position: User equipment position [3]
            antenna_index: Antenna index
            selected_subcarriers: Optional subcarrier selection
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Dictionary with network outputs
        """
        # Get memory optimization parameters
        ray_chunk_size = getattr(self, 'ray_chunk_size', 20)  # Smaller chunks
        
        device = bs_position.device
        num_rays = self.azimuth_divisions * self.elevation_divisions
        
        logger.debug(f"🚀 Ultra memory-optimized network: {num_rays} rays in chunks of {ray_chunk_size}")
        
        # Monitor memory
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated(device) / 1024**2
            logger.debug(f"💾 Network initial memory: {initial_memory:.1f} MB")
        
        # Generate all ray directions first (minimal memory)
        directions = self._generate_ray_directions(device)
        
        # Process rays in ultra-small chunks
        all_attenuation_vectors = []
        all_radiation_vectors = []
        all_sampled_positions = []
        
        for chunk_start in range(0, num_rays, ray_chunk_size):
            chunk_end = min(chunk_start + ray_chunk_size, num_rays)
            chunk_directions = directions[chunk_start:chunk_end]
            
            logger.debug(f"🔄 Ray chunk {chunk_start//ray_chunk_size + 1}: rays {chunk_start}-{chunk_end-1}")
            
            # Process chunk with minimal memory footprint
            chunk_result = self._process_ray_chunk_minimal(
                bs_position=bs_position,
                ue_position=ue_position,
                antenna_index=antenna_index,
                directions=chunk_directions,
                return_intermediates=return_intermediates
            )
            
            all_attenuation_vectors.append(chunk_result['attenuation_vectors'])
            all_radiation_vectors.append(chunk_result['radiation_vectors'])
            all_sampled_positions.append(chunk_result['sampled_positions'])
            
            # Aggressive per-chunk cleanup
            del chunk_directions, chunk_result
            if torch.cuda.is_available() and chunk_start % (ray_chunk_size * 3) == 0:  # Every 3 chunks
                torch.cuda.empty_cache()
        
        # Combine results
        combined_attenuation = torch.cat(all_attenuation_vectors, dim=0)
        combined_radiation = torch.cat(all_radiation_vectors, dim=0)
        combined_positions = torch.cat(all_sampled_positions, dim=0)
        
        # Cleanup intermediate lists
        del all_attenuation_vectors, all_radiation_vectors, all_sampled_positions
        
        # Generate frequency basis (once per call)
        frequency_basis = self._generate_frequency_basis(selected_subcarriers, device)
        
        # Final output
        outputs = {
            'attenuation_vectors': combined_attenuation,
            'radiation_vectors': combined_radiation,
            'frequency_basis_vectors': frequency_basis,
            'sampled_positions': combined_positions,
            'directions': directions
        }
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(device) / 1024**2
            logger.debug(f"💾 Network final memory: {final_memory:.1f} MB")
        
        return outputs
    
    def _process_ray_chunk_minimal(
        self,
        bs_position: torch.Tensor,
        ue_position: torch.Tensor,
        antenna_index: int,
        directions: torch.Tensor,  # [chunk_size, 3]
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process a small chunk of rays with minimal memory allocation.
        
        Args:
            bs_position: Base station position [3]
            ue_position: User equipment position [3]
            antenna_index: Antenna index
            directions: Ray directions [chunk_size, 3]
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Dictionary with processed ray results
        """
        chunk_size = directions.shape[0]
        device = directions.device
        
        # Sample positions along rays (minimal allocation)
        sampled_positions = self._sample_positions_along_rays(
            bs_position=bs_position,
            directions=directions,
            max_length=self.max_ray_length,
            num_points=self.num_sampling_points
        )
        
        # Process through networks with micro-point batching
        attenuation_vectors = self._process_attenuation_micro_batched(
            sampled_positions=sampled_positions
        )
        
        radiation_vectors = self._process_radiance_micro_batched(
            ue_position=ue_position,
            directions=directions,
            features=attenuation_vectors,
            antenna_index=antenna_index
        )
        
        return {
            'attenuation_vectors': attenuation_vectors,
            'radiation_vectors': radiation_vectors,
            'sampled_positions': sampled_positions
        }
    
    def _process_attenuation_micro_batched(
        self,
        sampled_positions: torch.Tensor  # [chunk_size, num_points, 3]
    ) -> torch.Tensor:
        """
        Process attenuation with micro-point batching.
        
        Args:
            sampled_positions: Sampled positions [chunk_size, num_points, 3]
            
        Returns:
            Attenuation features tensor
        """
        chunk_size, num_points, _ = sampled_positions.shape
        device = sampled_positions.device
        
        # Micro-batch size for point processing
        point_batch_size = 16  # Very small batches
        
        all_features = []
        
        # Flatten for point-wise processing
        flat_positions = sampled_positions.view(-1, 3)  # [chunk_size * num_points, 3]
        total_points = flat_positions.shape[0]
        
        # Process points in micro-batches
        for point_start in range(0, total_points, point_batch_size):
            point_end = min(point_start + point_batch_size, total_points)
            point_batch = flat_positions[point_start:point_end]
            
            # Process through attenuation network
            features = self.attenuation_network(point_batch)
            all_features.append(features)
            
            # Micro cleanup
            del point_batch
            if point_start % (point_batch_size * 4) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Combine and reshape
        combined_features = torch.cat(all_features, dim=0)
        reshaped_features = combined_features.view(chunk_size, num_points, -1)
        
        # Cleanup
        del all_features, flat_positions, combined_features
        
        return reshaped_features


def apply_memory_optimizations(training_interface, prism_network, config):
    """
    Apply ultra memory optimizations to existing components.
    
    This function patches the training interface and prism network with
    memory-optimized methods to support larger batch sizes.
    
    Args:
        training_interface: PrismTrainingInterface instance
        prism_network: PrismNetwork instance
        config: Configuration dictionary
        
    Returns:
        Tuple of (optimized_training_interface, optimized_prism_network)
    """
    logger.info("🚀 Applying ultra memory optimizations...")
    
    # Create enhanced classes using multiple inheritance
    class OptimizedTrainingInterface(training_interface.__class__, MemoryOptimizedMixin):
        pass
    
    class OptimizedPrismNetwork(prism_network.__class__, MemoryOptimizedPrismNetworkMixin):
        pass
    
    # Replace the class of existing instances
    training_interface.__class__ = OptimizedTrainingInterface
    prism_network.__class__ = OptimizedPrismNetwork
    
    # Set ray chunk size from config
    ray_chunk_size = config.get('training', {}).get('ray_chunk_size', 20)
    prism_network.ray_chunk_size = ray_chunk_size
    
    logger.info(f"✅ Ultra memory optimizations applied:")
    logger.info(f"   - Ray chunk size: {ray_chunk_size}")
    logger.info(f"   - UE chunk processing enabled")
    logger.info(f"   - Micro-antenna batching enabled")
    logger.info(f"   - Gradient checkpointing: {config.get('training', {}).get('enable_gradient_checkpointing', False)}")
    
    return training_interface, prism_network


# Convenience function for easy integration
def enable_memory_optimizations(training_interface, prism_network, config=None):
    """
    Convenience function to enable memory optimizations.
    
    Args:
        training_interface: PrismTrainingInterface instance
        prism_network: PrismNetwork instance
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (optimized_training_interface, optimized_prism_network)
    """
    if config is None:
        config = {'training': {}}
    
    return apply_memory_optimizations(training_interface, prism_network, config)
