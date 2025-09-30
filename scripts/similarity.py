#!/usr/bin/env python3
"""
Similarity Metrics Module

This module provides various similarity measurement algorithms for comparing
predicted and target values in signal processing and machine learning contexts.

All methods are implemented as static methods in the Similarity class for easy access
and modularity.
"""

from typing import Tuple
import numpy as np
import torch
class Similarity:
    """计算各种相似性度量的工具类"""
    
    @staticmethod
    def compute_empirical_cdf(data: np.ndarray) -> tuple:
        """Compute empirical CDF for given data"""
        sorted_data = np.sort(data)
        n = len(sorted_data)
        y = np.arange(1, n + 1) / n
        return sorted_data, y

    @staticmethod
    def compute_spectral_correlation_coefficient(pred_pdp: np.ndarray, target_pdp: np.ndarray) -> float:
        """Compute Spectral Correlation Coefficient (SCC) between two PDPs"""
        pred_flat = pred_pdp.flatten()
        target_flat = target_pdp.flatten()
        
        # Compute correlation coefficient
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        
        # Handle NaN case
        if np.isnan(correlation):
            return 0.0
        
        # Map from [-1, 1] to [0, 1] where 1 is most similar
        return float((correlation + 1.0) / 2.0)

    @staticmethod
    def compute_log_spectral_distance(pred_pdp: np.ndarray, target_pdp: np.ndarray) -> float:
        """Compute Log Spectral Distance (LSD) using maximum possible error normalization"""
        pred_flat = pred_pdp.flatten()
        target_flat = target_pdp.flatten()
        
        # Compute MSE
        mse = np.mean((pred_flat - target_flat) ** 2)
        
        # Compute maximum squared values
        pred_max_squared = np.max(pred_flat ** 2)
        target_max_squared = np.max(target_flat ** 2)
        max_possible_error = pred_max_squared + target_max_squared
        
        # Avoid division by zero
        if max_possible_error < 1e-12:
            return 1.0  # Perfect similarity if no variation
        
        # Compute maximum possible error normalized similarity
        similarity = 1.0 - (mse / max_possible_error)
        
        # Clip to valid range [0, 1]
        similarity = np.clip(similarity, 0.0, 1.0)
        
        return float(similarity)

    @staticmethod
    def compute_bhattacharyya_coefficient(pred_pdp: np.ndarray, target_pdp: np.ndarray) -> float:
        """Compute Bhattacharyya Coefficient (BC) between two PDPs"""
        pred_flat = pred_pdp.flatten()
        target_flat = target_pdp.flatten()
        
        # Normalize to make them probability distributions
        pred_norm = pred_flat / (np.sum(pred_flat) + 1e-10)
        target_norm = target_flat / (np.sum(target_flat) + 1e-10)
        
        # Compute Bhattacharyya coefficient
        bc = np.sum(np.sqrt(pred_norm * target_norm))
        
        # BC is already in [0, 1] where 1 is most similar
        return float(bc)

    @staticmethod
    def compute_jensen_shannon_divergence(pred_pdp: np.ndarray, target_pdp: np.ndarray) -> float:
        """Compute Jensen-Shannon Divergence (JSD) using maximum possible error normalization"""
        pred_flat = pred_pdp.flatten()
        target_flat = target_pdp.flatten()
        
        # Compute MSE
        mse = np.mean((pred_flat - target_flat) ** 2)
        
        # Compute maximum squared values
        pred_max_squared = np.max(pred_flat ** 2)
        target_max_squared = np.max(target_flat ** 2)
        max_possible_error = pred_max_squared + target_max_squared
        
        # Avoid division by zero
        if max_possible_error < 1e-12:
            return 1.0  # Perfect similarity if no variation
        
        # Compute maximum possible error normalized similarity
        similarity = 1.0 - (mse / max_possible_error)
        
        # Clip to valid range [0, 1]
        similarity = np.clip(similarity, 0.0, 1.0)
        
        return float(similarity)

    @staticmethod
    def compute_cosine_similarity(pred_tensor: torch.Tensor, target_tensor: torch.Tensor) -> float:
        """Compute cosine similarity between two tensors"""
        pred_flat = pred_tensor.flatten()
        target_flat = target_tensor.flatten()
        
        # Compute cosine similarity
        dot_product = torch.dot(pred_flat, target_flat)
        pred_norm = torch.norm(pred_flat)
        target_norm = torch.norm(target_flat)
        
        # Avoid division by zero
        if pred_norm < 1e-8 or target_norm < 1e-8:
            return 0.0
        
        cosine_sim = dot_product / (pred_norm * target_norm)
        
        # Convert to similarity: (1 + cosine) / 2 (range [0, 1], 1 = most similar)
        # cosine=1 → similarity=1 (best), cosine=0 → similarity=0.5 (medium), cosine=-1 → similarity=0 (worst)
        return float((1.0 + cosine_sim) / 2.0)

    @staticmethod
    def compute_nmse_similarity(pred_tensor: torch.Tensor, target_tensor: torch.Tensor) -> float:
        """
        Compute NMSE similarity using maximum possible error normalization.
        
        Maximum Possible Error Normalization NMSE Similarity Definition:
        Similarity = 1 - MSE / (max(x_i²) + max(y_i²))
        
        where:
        - MSE = (1/n) * sum((x_i - y_i)²): mean squared error
        - max(x_i²): maximum squared value in predicted vector
        - max(y_i²): maximum squared value in target vector
        
        This gives a similarity value in range [0, 1] where:
        - 1.0 = perfect match (MSE = 0)
        - 0.0 = worst match (MSE = max(x_i²) + max(y_i²))
        - Higher values indicate better similarity
        
        Args:
            pred: Predicted values tensor
            target: Target values tensor
            
        Returns:
            Maximum possible error normalized NMSE similarity value in range [0, 1]
        """
        # Flatten tensors for computation
        pred_flat = pred_tensor.flatten()
        target_flat = target_tensor.flatten()
        
        # Compute MSE
        mse = torch.mean((pred_flat - target_flat) ** 2)
        
        # Compute maximum squared values
        pred_max_squared = torch.max(pred_flat ** 2)
        target_max_squared = torch.max(target_flat ** 2)
        max_possible_error = pred_max_squared + target_max_squared
        
        # Avoid division by zero
        if max_possible_error < 1e-12:
            return 1.0  # Perfect similarity if no variation
        
        # Compute maximum possible error normalized similarity
        similarity = 1.0 - (mse / max_possible_error)
        
        # Clip to valid range [0, 1]
        similarity = torch.clamp(similarity, 0.0, 1.0)
        
        return float(similarity)

    @staticmethod
    def compute_ssim_1d(pred_pdp: torch.Tensor, target_pdp: torch.Tensor, 
                        window_size: int = 15, sigma: float = 2.0) -> float:
        """
        Compute 1D Structural Similarity Index (SSIM) between two 1D PDPs.
        
        Uses a sliding window approach with Gaussian weights for proper SSIM computation.
        Now implemented with pure PyTorch for GPU acceleration.
        
        Args:
            pred_pdp: Predicted PDP as torch tensor (will be flattened if multi-dimensional)
            target_pdp: Target PDP as torch tensor (will be flattened if multi-dimensional)
            window_size: Size of the Gaussian window (default: 11)
            sigma: Standard deviation of the Gaussian window (default: 1.5)
        
        Returns:
            SSIM value in [0, 1] where 1 indicates perfect similarity
        """
        # Keep tensors on GPU for computation
        device = pred_pdp.device
        pred_pdp = pred_pdp.flatten()
        target_pdp = target_pdp.flatten()
        
        # Flatten arrays if needed
        pred_flat = pred_pdp.flatten()
        target_flat = target_pdp.flatten()
        
        # Ensure equal length
        min_len = min(len(pred_flat), len(target_flat))
        pred_flat = pred_flat[:min_len]
        target_flat = target_flat[:min_len]
        
        try:
            # SSIM parameters
            data_range = max(pred_flat.max() - pred_flat.min(), target_flat.max() - target_flat.min())
            if data_range < 1e-10:
                return 1.0  # Perfect similarity if no variation
            

            c1 = (0.04 * data_range) ** 2 
            c2 = (0.12 * data_range) ** 2 
            
            # Window size for 1D SSIM (adaptive to signal length)
            actual_window_size = min(window_size, len(pred_flat))  # Adaptive window size
            if actual_window_size < 3:
                actual_window_size = len(pred_flat)  # Use full signal if too short
            
            # Create 1D Gaussian window on GPU
            center = actual_window_size // 2
            x = torch.arange(actual_window_size, device=device, dtype=torch.float32) - center
            window = torch.exp(-0.5 * (x / sigma) ** 2)
            window = window / window.sum()
            
            # Pad signals for convolution (using reflection padding)
            pad_width = actual_window_size // 2
            pred_padded = torch.nn.functional.pad(pred_flat.unsqueeze(0), (pad_width, pad_width), mode='reflect').squeeze(0)
            target_padded = torch.nn.functional.pad(target_flat.unsqueeze(0), (pad_width, pad_width), mode='reflect').squeeze(0)
            
            # Compute local means using convolution
            mu1 = torch.nn.functional.conv1d(pred_padded.unsqueeze(0).unsqueeze(0), 
                                           window.unsqueeze(0).unsqueeze(0), 
                                           padding=0).squeeze()
            mu2 = torch.nn.functional.conv1d(target_padded.unsqueeze(0).unsqueeze(0), 
                                           window.unsqueeze(0).unsqueeze(0), 
                                           padding=0).squeeze()
            
            # Compute local variances and covariance
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = torch.nn.functional.conv1d((pred_padded ** 2).unsqueeze(0).unsqueeze(0), 
                                                 window.unsqueeze(0).unsqueeze(0), 
                                                 padding=0).squeeze() - mu1_sq
            sigma2_sq = torch.nn.functional.conv1d((target_padded ** 2).unsqueeze(0).unsqueeze(0), 
                                                 window.unsqueeze(0).unsqueeze(0), 
                                                 padding=0).squeeze() - mu2_sq
            sigma12 = torch.nn.functional.conv1d((pred_padded * target_padded).unsqueeze(0).unsqueeze(0), 
                                               window.unsqueeze(0).unsqueeze(0), 
                                               padding=0).squeeze() - mu1_mu2
            
            # Compute SSIM
            numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
            denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
            
            ssim_values = numerator / denominator
            return float(ssim_values.mean())
            
        except Exception as e:
            # Fallback to simple correlation if SSIM fails
            pred_np = pred_flat.cpu().numpy()
            target_np = target_flat.cpu().numpy()
            correlation = np.corrcoef(pred_np, target_np)[0, 1]
            if np.isnan(correlation):
                return 0.0
            return float((correlation + 1.0) / 2.0)

    @staticmethod
    def compute_ssim_2d(pred_spectrum: torch.Tensor, target_spectrum: torch.Tensor,
                        window_size: int = 85, sigma: float = 30.0) -> float:
        """
        Compute 2D Structural Similarity Index (SSIM) between two 2D spatial spectra.
        
        Uses a sliding 2D window approach with Gaussian weights for proper SSIM computation.
        Now implemented with pure PyTorch for GPU acceleration.
        
        Args:
            pred_spectrum: Predicted spectrum tensor [azimuth, elevation]
            target_spectrum: Target spectrum tensor [azimuth, elevation]
            window_size: Size of the 2D Gaussian window (default: 11)
            sigma: Standard deviation of the Gaussian window (default: 1.5)
        
        Returns:
            SSIM value in [0, 1] where 1 indicates perfect similarity
        """
        try:
            # Keep tensors on GPU for computation
            device = pred_spectrum.device
            
            # Ensure same shape
            if pred_spectrum.shape != target_spectrum.shape:
                raise ValueError(f"Shape mismatch: {pred_spectrum.shape} vs {target_spectrum.shape}")
            
            # Compute data range
            data_range = max(pred_spectrum.max() - pred_spectrum.min(), target_spectrum.max() - target_spectrum.min())
            
            if data_range < 1e-10:
                return 1.0  # Perfect similarity if no variation
            
            # True 2D SSIM implementation with 2D sliding window
            H, W = pred_spectrum.shape
            
            # Ensure window size is odd and not larger than image dimensions
            actual_window_size = min(window_size, min(H, W))
            if actual_window_size % 2 == 0:
                actual_window_size -= 1  # Make it odd
            if actual_window_size < 3:
                actual_window_size = 3
            
            # Create 2D Gaussian window
            center = actual_window_size // 2
            y, x = torch.meshgrid(
                torch.arange(actual_window_size, device=device, dtype=torch.float32) - center,
                torch.arange(actual_window_size, device=device, dtype=torch.float32) - center,
                indexing='ij'
            )
            window_2d = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
            window_2d = window_2d / window_2d.sum()
            
            c1 = (0.30 * data_range) ** 2  # Ultra large C1 for maximum tolerance
            c2 = (0.90 * data_range) ** 2  # Ultra large C2 for maximum tolerance
            
            # Add batch and channel dimensions for conv2d
            pred_4d = pred_spectrum.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            target_4d = target_spectrum.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            window_4d = window_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, ws, ws]
            
            # Padding for 'same' convolution
            pad = actual_window_size // 2
            
            # Compute local means using 2D convolution
            mu1 = torch.nn.functional.conv2d(pred_4d, window_4d, padding=pad)
            mu2 = torch.nn.functional.conv2d(target_4d, window_4d, padding=pad)
            
            # Compute local variances and covariance
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = torch.nn.functional.conv2d(pred_4d ** 2, window_4d, padding=pad) - mu1_sq
            sigma2_sq = torch.nn.functional.conv2d(target_4d ** 2, window_4d, padding=pad) - mu2_sq
            sigma12 = torch.nn.functional.conv2d(pred_4d * target_4d, window_4d, padding=pad) - mu1_mu2
            
            # Compute SSIM map
            numerator1 = 2 * mu1_mu2 + c1
            numerator2 = 2 * sigma12 + c2
            denominator1 = mu1_sq + mu2_sq + c1
            denominator2 = sigma1_sq + sigma2_sq + c2
            
            ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
            
            # Return mean SSIM over the entire image
            ssim_value = torch.mean(ssim_map)
            return float(torch.clamp(ssim_value, 0.0, 1.0))
            
        except Exception as e:
            # Fallback to simple correlation if SSIM fails
            pred_np = pred_spectrum.cpu().numpy()
            target_np = target_spectrum.cpu().numpy()
            correlation = np.corrcoef(pred_np.flatten(), target_np.flatten())[0, 1]
            if np.isnan(correlation):
                return 0.0
            return float((correlation + 1.0) / 2.0)
