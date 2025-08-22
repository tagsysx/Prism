"""
Prism Loss Functions

This module implements specialized loss functions for training Prism networks
that handle multi-subcarrier RF signals and complex-valued CSI predictions.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class PrismLoss(nn.Module):
    """
    Specialized loss function for Prism networks that handles:
    1. Complex-valued CSI predictions
    2. Multi-subcarrier frequency-aware loss
    3. Magnitude and phase components separately
    4. Frequency-dependent weighting
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        frequency_weights: Optional[torch.Tensor] = None,
        magnitude_weight: float = 1.0,
        phase_weight: float = 0.5,
        complex_handling: str = 'magnitude_phase'
    ):
        """
        Initialize Prism loss function.
        
        Args:
            loss_type: Type of loss ('mse', 'l1', 'huber')
            frequency_weights: Optional weights for different subcarriers
            magnitude_weight: Weight for magnitude loss component
            phase_weight: Weight for phase loss component
            complex_handling: How to handle complex numbers ('magnitude_phase', 'real_imag', 'magnitude_only')
        """
        super().__init__()
        self.loss_type = loss_type
        self.magnitude_weight = magnitude_weight
        self.phase_weight = phase_weight
        self.complex_handling = complex_handling
        
        # Set up base loss function
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.base_loss = nn.L1Loss(reduction='none')
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss(reduction='none', delta=1.0)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # Frequency weights
        if frequency_weights is not None:
            self.register_buffer('frequency_weights', frequency_weights)
        else:
            self.frequency_weights = None
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss between predicted and target CSI values.
        
        Args:
            predictions: Predicted CSI values (batch_size, num_ue, num_subcarriers) or complex
            targets: Target CSI values with same shape as predictions
            return_components: Whether to return individual loss components
            
        Returns:
            Total loss or dictionary with loss components
        """
        # Ensure inputs are complex
        if not torch.is_complex(predictions):
            predictions = self._convert_to_complex(predictions)
        if not torch.is_complex(targets):
            targets = self._convert_to_complex(targets)
        
        # Handle complex numbers based on configuration
        if self.complex_handling == 'magnitude_phase':
            loss = self._magnitude_phase_loss(predictions, targets)
        elif self.complex_handling == 'real_imag':
            loss = self._real_imag_loss(predictions, targets)
        elif self.complex_handling == 'magnitude_only':
            loss = self._magnitude_only_loss(predictions, targets)
        else:
            raise ValueError(f"Unsupported complex handling: {self.complex_handling}")
        
        # Apply frequency weights if specified
        if self.frequency_weights is not None:
            loss = self._apply_frequency_weights(loss)
        
        # Compute final loss
        total_loss = loss.mean()
        
        if return_components:
            return {
                'total_loss': total_loss,
                'magnitude_loss': loss.mean() if self.complex_handling == 'magnitude_only' else None,
                'phase_loss': None,  # Would need separate computation
                'frequency_weighted_loss': loss if self.frequency_weights is not None else None
            }
        
        return total_loss
    
    def _magnitude_phase_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss using magnitude and phase components separately.
        
        Args:
            predictions: Complex predictions
            targets: Complex targets
            
        Returns:
            Combined loss tensor
        """
        # Extract magnitude and phase
        pred_magnitude = torch.abs(predictions)
        target_magnitude = torch.abs(targets)
        
        pred_phase = torch.angle(predictions)
        target_phase = torch.angle(targets)
        
        # Compute magnitude loss
        magnitude_loss = self.base_loss(pred_magnitude, target_magnitude)
        
        # Compute phase loss (handle phase wrapping)
        phase_diff = torch.atan2(torch.sin(pred_phase - target_phase), torch.cos(pred_phase - target_phase))
        phase_loss = self.base_loss(phase_diff, torch.zeros_like(phase_diff))
        
        # Combine losses
        combined_loss = (
            self.magnitude_weight * magnitude_loss + 
            self.phase_weight * phase_loss
        )
        
        return combined_loss
    
    def _real_imag_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss using real and imaginary components.
        
        Args:
            predictions: Complex predictions
            targets: Complex targets
            
        Returns:
            Loss tensor
        """
        # Convert to real/imaginary representation
        pred_real_imag = torch.cat([predictions.real, predictions.imag], dim=-1)
        target_real_imag = torch.cat([targets.real, targets.imag], dim=-1)
        
        # Compute loss
        loss = self.base_loss(pred_real_imag, target_real_imag)
        
        return loss
    
    def _magnitude_only_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss using only magnitude components.
        
        Args:
            predictions: Complex predictions
            targets: Complex targets
            
        Returns:
            Loss tensor
        """
        # Extract magnitudes
        pred_magnitude = torch.abs(predictions)
        target_magnitude = torch.abs(targets)
        
        # Compute loss
        loss = self.base_loss(pred_magnitude, target_magnitude)
        
        return loss
    
    def _apply_frequency_weights(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency-dependent weights to the loss.
        
        Args:
            loss: Loss tensor (batch_size, num_ue, num_subcarriers)
            
        Returns:
            Weighted loss tensor
        """
        # Ensure frequency weights match the subcarrier dimension
        if self.frequency_weights.dim() == 1:
            # Expand to match loss dimensions
            weights = self.frequency_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, num_subcarriers)
        else:
            weights = self.frequency_weights
        
        # Apply weights
        weighted_loss = loss * weights
        
        return weighted_loss
    
    def _convert_to_complex(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert real tensor to complex if needed.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Complex tensor
        """
        if tensor.dim() == 3 and tensor.shape[-1] == 2:
            # Assume last dimension is [real, imag]
            return torch.complex(tensor[..., 0], tensor[..., 1])
        elif tensor.dim() == 3:
            # Assume real-valued, convert to complex with zero imaginary part
            return torch.complex(tensor, torch.zeros_like(tensor))
        else:
            raise ValueError(f"Cannot convert tensor of shape {tensor.shape} to complex")
    
    def get_loss_info(self) -> Dict[str, any]:
        """Get information about the loss function configuration."""
        return {
            'loss_type': self.loss_type,
            'complex_handling': self.complex_handling,
            'magnitude_weight': self.magnitude_weight,
            'phase_weight': self.phase_weight,
            'frequency_weights': self.frequency_weights is not None,
            'frequency_weights_shape': list(self.frequency_weights.shape) if self.frequency_weights is not None else None
        }


class FrequencyAwareLoss(PrismLoss):
    """
    Frequency-aware loss function that emphasizes certain subcarriers.
    Useful for OFDM systems where different subcarriers have different importance.
    """
    
    def __init__(
        self,
        center_frequency: float = 2.4e9,  # 2.4 GHz
        bandwidth: float = 20e6,  # 20 MHz
        num_subcarriers: int = 64,
        frequency_emphasis: str = 'center',  # 'center', 'edges', 'custom'
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.center_frequency = center_frequency
        self.bandwidth = bandwidth
        self.num_subcarriers = num_subcarriers
        self.frequency_emphasis = frequency_emphasis
        
        # Generate frequency weights
        self.frequency_weights = self._generate_frequency_weights()
    
    def _generate_frequency_weights(self) -> torch.Tensor:
        """
        Generate frequency-dependent weights.
        
        Returns:
            Frequency weights tensor
        """
        if self.frequency_emphasis == 'center':
            # Emphasize center subcarriers
            weights = torch.ones(self.num_subcarriers)
            center_idx = self.num_subcarriers // 2
            for i in range(self.num_subcarriers):
                distance_from_center = abs(i - center_idx)
                weights[i] = 1.0 + 0.5 * torch.exp(-distance_from_center / (self.num_subcarriers / 8))
        
        elif self.frequency_emphasis == 'edges':
            # Emphasize edge subcarriers
            weights = torch.ones(self.num_subcarriers)
            for i in range(self.num_subcarriers):
                distance_from_center = abs(i - self.num_subcarriers // 2)
                weights[i] = 1.0 + 0.3 * (distance_from_center / (self.num_subcarriers // 2))
        
        elif self.frequency_emphasis == 'custom':
            # Custom frequency response (e.g., low-pass filter)
            weights = torch.ones(self.num_subcarriers)
            for i in range(self.num_subcarriers):
                normalized_freq = (i - self.num_subcarriers // 2) / (self.num_subcarriers // 2)
                weights[i] = 1.0 / (1.0 + 0.1 * normalized_freq**2)
        
        else:
            weights = torch.ones(self.num_subcarriers)
        
        # Normalize weights
        weights = weights / weights.mean()
        
        return weights


class CSIVirtualLinkLoss(PrismLoss):
    """
    Loss function specifically designed for CSI virtual link processing.
    Handles the M×N_UE uplink combinations as described in the design document.
    """
    
    def __init__(
        self,
        num_ue_antennas: int,
        num_subcarriers: int,
        virtual_link_sampling: str = 'random',  # 'random', 'all', 'importance'
        sampling_ratio: float = 0.5,  # Ratio of virtual links to sample
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_ue_antennas = num_ue_antennas
        self.num_subcarriers = num_subcarriers
        self.virtual_link_sampling = virtual_link_sampling
        self.sampling_ratio = sampling_ratio
        
        # Total number of virtual links
        self.total_virtual_links = num_ue_antennas * num_subcarriers
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss for CSI virtual links with optional sampling.
        
        Args:
            predictions: Predicted CSI (batch_size, num_ue, num_subcarriers)
            targets: Target CSI (batch_size, num_ue, num_subcarriers)
            return_components: Whether to return loss components
            
        Returns:
            Loss value or components
        """
        batch_size = predictions.shape[0]
        
        if self.virtual_link_sampling == 'random':
            # Random sampling of virtual links for efficiency
            loss = self._random_sampling_loss(predictions, targets, batch_size)
        elif self.virtual_link_sampling == 'all':
            # Use all virtual links
            loss = super().forward(predictions, targets, return_components)
        elif self.virtual_link_sampling == 'importance':
            # Importance-based sampling
            loss = self._importance_sampling_loss(predictions, targets, batch_size)
        else:
            raise ValueError(f"Unsupported virtual link sampling: {self.virtual_link_sampling}")
        
        return loss
    
    def _random_sampling_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute loss using random sampling of virtual links.
        
        Args:
            predictions: Predicted CSI
            targets: Target CSI
            batch_size: Batch size
            
        Returns:
            Sampled loss value
        """
        # Sample virtual links
        num_samples = int(self.total_virtual_links * self.sampling_ratio)
        
        # Generate random indices for sampling
        ue_indices = torch.randint(0, self.num_ue_antennas, (num_samples,))
        subcarrier_indices = torch.randint(0, self.num_subcarriers, (num_samples,))
        
        # Extract sampled values
        pred_sampled = predictions[:, ue_indices, subcarrier_indices]
        target_sampled = targets[:, ue_indices, subcarrier_indices]
        
        # Compute loss on sampled data
        loss = super().forward(pred_sampled, target_sampled)
        
        return loss
    
    def _importance_sampling_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute loss using importance-based sampling of virtual links.
        
        Args:
            predictions: Predicted CSI
            targets: Target CSI
            batch_size: Batch size
            
        Returns:
            Importance-weighted loss value
        """
        # Compute importance weights based on target magnitude
        target_magnitude = torch.abs(targets)
        importance_weights = target_magnitude / (target_magnitude.sum() + 1e-8)
        
        # Sample based on importance
        num_samples = int(self.total_virtual_links * self.sampling_ratio)
        
        # Flatten importance weights for sampling
        flat_importance = importance_weights.view(batch_size, -1)
        
        # Sample indices based on importance
        sampled_indices = torch.multinomial(flat_importance, num_samples, replacement=False)
        
        # Convert flat indices to 2D indices
        ue_indices = sampled_indices // self.num_subcarriers
        subcarrier_indices = sampled_indices % self.num_subcarriers
        
        # Extract sampled values
        pred_sampled = predictions[torch.arange(batch_size).unsqueeze(1), ue_indices, subcarrier_indices]
        target_sampled = targets[torch.arange(batch_size).unsqueeze(1), ue_indices, subcarrier_indices]
        
        # Compute loss on sampled data
        loss = super().forward(pred_sampled, target_sampled)
        
        return loss
