"""
PAS Loss 2 (Multi-subcarrier CSI Spatial-Frequency Loss) Implementation

This module implements advanced loss functions for multi-subcarrier CSI data
in MIMO systems, focusing on spatial-frequency joint characteristics without
using spectrograms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import logging
from prism.utils.pas_utils import reorganize_data_as_mimo

logger = logging.getLogger(__name__)


class PAS2Loss(nn.Module):
    """
    Multi-subcarrier CSI Spatial-Frequency Loss for MIMO systems.
    
    This loss implements spatial-frequency joint loss functions for CSI data
    containing multiple subcarriers, focusing on frequency domain correlations
    and spatial characteristics without using spectrograms.
    
    Args:
        num_bs_antennas: Number of base station antennas
        num_ue_antennas: Number of user equipment antennas
        freq_spatial_weight: Weight for frequency-spatial correlation loss (default: 0.40)
        phase_consistency_weight: Weight for phase consistency loss (default: 0.35)
        angle_spectrum_weight: Weight for angle spectrum loss (default: 0.25)
        lambda_smooth: Smoothing regularization parameter (default: 0.1)
        mu_eig: Eigenvalue gradient regularization parameter (default: 0.1)
    """
    
    def __init__(
        self,
        num_bs_antennas: int,
        num_ue_antennas: int,
        freq_spatial_weight: float = 0.40,
        phase_consistency_weight: float = 0.35,
        angle_spectrum_weight: float = 0.25,
        lambda_smooth: float = 0.1,
        mu_eig: float = 0.1
    ):
        super().__init__()
        
        # Validate antenna counts
        if num_bs_antennas <= 0:
            raise ValueError(f"num_bs_antennas must be positive, got {num_bs_antennas}")
        if num_ue_antennas <= 0:
            raise ValueError(f"num_ue_antennas must be positive, got {num_ue_antennas}")
        
        # Store antenna counts
        self.num_bs_antennas = num_bs_antennas
        self.num_ue_antennas = num_ue_antennas
        
        # Loss weights
        self.freq_spatial_weight = freq_spatial_weight
        self.phase_consistency_weight = phase_consistency_weight
        self.angle_spectrum_weight = angle_spectrum_weight
        
        # Regularization parameters
        self.lambda_smooth = lambda_smooth
        self.mu_eig = mu_eig
        
        # Validate weights sum to 1.0
        total_weight = freq_spatial_weight + phase_consistency_weight + angle_spectrum_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Loss weights must sum to 1.0, got {total_weight}")
        
        logger.info(f"PAS2Loss initialized:")
        logger.info(f"  Number of BS antennas: {self.num_bs_antennas}")
        logger.info(f"  Number of UE antennas: {self.num_ue_antennas}")
        logger.info(f"  Loss weights: freq_spatial={freq_spatial_weight:.2f}, phase={phase_consistency_weight:.2f}, angle={angle_spectrum_weight:.2f}")
        logger.info(f"  Regularization: lambda_smooth={lambda_smooth}, mu_eig={mu_eig}")
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of PASLoss2.
        
        Args:
            predictions: Dictionary containing predicted values
                        - 'csi': Predicted CSI tensor [batch_size, num_subcarriers] (complex)
                        - 'bs_positions': BS positions [batch_size, 3] (required)
                        - 'ue_positions': UE positions [batch_size, 3] (required)
                        - 'bs_antenna_indices': BS antenna indices [batch_size] (required)
                        - 'ue_antenna_indices': UE antenna indices [batch_size] (required)
            targets: Dictionary containing target values (same structure as predictions)
            
        Returns:
            loss: Combined loss value (scalar tensor)
        """
        # Check required data availability (similar to loss_function.py diagnostics)
        required_keys = ['csi', 'bs_positions', 'ue_positions', 'bs_antenna_indices', 'ue_antenna_indices']
        
        missing_pred = [key for key in required_keys if key not in predictions]
        missing_target = [key for key in required_keys if key not in targets]
        
        if missing_pred or missing_target:
            error_msg = "PAS2 loss computation failed - missing required data:"
            if missing_pred:
                error_msg += f"\n  Missing in predictions: {missing_pred}"
            if missing_target:
                error_msg += f"\n  Missing in targets: {missing_target}"
            error_msg += f"\n  Available in predictions: {list(predictions.keys())}"
            error_msg += f"\n  Available in targets: {list(targets.keys())}"
            raise ValueError(error_msg)
        
        # Extract data from dictionaries
        pred_csi = predictions['csi']
        target_csi = targets['csi']
        bs_positions = predictions['bs_positions']
        ue_positions = predictions['ue_positions']
        
        # Extract antenna indices (required for proper CSI reorganization)
        bs_antenna_indices = predictions['bs_antenna_indices']
        ue_antenna_indices = predictions['ue_antenna_indices']
        
        # Reorganize data as MIMO CSI tensors (similar to PAS loss approach)
        pred_csi_mimo, unique_positions = reorganize_data_as_mimo(
            pred_csi, bs_positions, ue_positions, bs_antenna_indices, ue_antenna_indices,
            self.num_bs_antennas, self.num_ue_antennas
        )
        
        target_csi_mimo, _ = reorganize_data_as_mimo(
            target_csi, bs_positions, ue_positions, bs_antenna_indices, ue_antenna_indices,
            self.num_bs_antennas, self.num_ue_antennas
        )
        
        # Stack MIMO CSI tensors: [num_position_pairs, M, N, S]
        # where M=num_bs_antennas, N=num_ue_antennas, S=num_subcarriers
        H_pred = torch.stack(pred_csi_mimo, dim=0)  # [num_pos, M, N, S]
        H_true = torch.stack(target_csi_mimo, dim=0)  # [num_pos, M, N, S]
        
        # Compute individual loss components
        L_freq_spatial = self._compute_freq_spatial_correlation_loss(H_pred, H_true)
        L_phase_consistency = self._compute_enhanced_freq_phase_consistency_loss(H_pred, H_true)
        L_angle_spectrum = self._compute_ue_angle_spectrum_loss(H_pred, H_true)
        
        # Log individual PAS2 loss components
        logger.info(f"📊 PAS2 Loss Components:")
        logger.info(f"   Freq-Spatial Loss: {L_freq_spatial.item():.6f} (weight: {self.freq_spatial_weight})")
        logger.info(f"   Phase Consistency Loss: {L_phase_consistency.item():.6f} (weight: {self.phase_consistency_weight})")
        logger.info(f"   Angle Spectrum Loss: {L_angle_spectrum.item():.6f} (weight: {self.angle_spectrum_weight})")
        
        # Combine losses with specified weights
        total_loss = (self.freq_spatial_weight * L_freq_spatial + 
                     self.phase_consistency_weight * L_phase_consistency + 
                     self.angle_spectrum_weight * L_angle_spectrum)
        
        logger.info(f"   Total PAS2 Loss: {total_loss.item():.6f}")
        
        return total_loss
    
    def _compute_freq_spatial_correlation_loss(self, H_pred: torch.Tensor, H_true: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency-spatial joint correlation loss.
        
        L_freq_spatial = ∑_{k,l} ||R_pred[k,l] - R_true[k,l]||_F² / ∑_{k,l} ||R_true[k,l]||_F²
        where R_pred[k,l] = H_pred[:,:,k] × H_pred[:,:,l]ᴴ
        
        For single BS antenna case (M=1), use frequency correlation across UE antennas instead.
        
        Args:
            H_pred: Predicted CSI [num_pos, M, N, S]
            H_true: True CSI [num_pos, M, N, S]
            
        Returns:
            Normalized frequency-spatial correlation loss
        """
        num_pos, M, N, S = H_pred.shape
        
        # Handle single BS antenna case differently
        if M == 1:
            logger.info("Single BS antenna detected, using frequency correlation across UE antennas")
            return self._compute_freq_correlation_single_bs(H_pred, H_true)
        
        total_loss = 0.0
        total_norm = 0.0
        
        for pos_idx in range(num_pos):
            H_p = H_pred[pos_idx]  # [M, N, S]
            H_t = H_true[pos_idx]  # [M, N, S]
            
            pos_loss = 0.0
            pos_norm = 0.0
            
            # Compute correlation matrices for all subcarrier pairs
            for k in range(S):
                for l in range(S):
                    # R[k,l] = H[:,:,k] × H[:,:,l]ᴴ
                    R_pred_kl = torch.matmul(H_p[:, :, k], torch.conj(H_p[:, :, l]).transpose(-2, -1))  # [M, M]
                    R_true_kl = torch.matmul(H_t[:, :, k], torch.conj(H_t[:, :, l]).transpose(-2, -1))  # [M, M]
                    
                    # Frobenius norm squared
                    diff_norm_sq = torch.norm(R_pred_kl - R_true_kl, p='fro') ** 2
                    true_norm_sq = torch.norm(R_true_kl, p='fro') ** 2
                    
                    pos_loss += diff_norm_sq
                    pos_norm += true_norm_sq
            
            total_loss += pos_loss
            total_norm += pos_norm
        
        # Normalize by total true correlation energy
        if total_norm > 1e-12:
            return total_loss / total_norm
        else:
            logger.warning("Total correlation norm is too small, returning zero loss")
            return torch.tensor(0.0, device=H_pred.device, dtype=H_pred.dtype)
    
    def _compute_freq_correlation_single_bs(self, H_pred: torch.Tensor, H_true: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency correlation for single BS antenna case.
        
        Uses direct frequency correlation: C[k,l] = <H[:,k], H[:,l]> (inner product across UE antennas)
        
        Args:
            H_pred: Predicted CSI [num_pos, 1, N, S]
            H_true: True CSI [num_pos, 1, N, S]
            
        Returns:
            Frequency correlation loss
        """
        num_pos, M, N, S = H_pred.shape
        
        total_loss = 0.0
        
        for pos_idx in range(num_pos):
            H_p = H_pred[pos_idx, 0, :, :]  # [N, S] - squeeze BS dimension
            H_t = H_true[pos_idx, 0, :, :]  # [N, S]
            
            pos_loss = 0.0
            
            # Compute frequency correlation matrix for each position
            # C[k,l] = <H[:,k], H[:,l]> = sum over UE antennas
            C_pred = torch.zeros(S, S, dtype=H_p.dtype, device=H_p.device)
            C_true = torch.zeros(S, S, dtype=H_t.dtype, device=H_t.device)
            
            for k in range(S):
                for l in range(S):
                    # Inner product across UE antennas
                    C_pred[k, l] = torch.sum(torch.conj(H_p[:, k]) * H_p[:, l])
                    C_true[k, l] = torch.sum(torch.conj(H_t[:, k]) * H_t[:, l])
            
            # Compute Frobenius norm of difference
            correlation_diff = torch.norm(C_pred - C_true, p='fro') ** 2
            
            # Normalize by the Frobenius norm of true correlation matrix
            correlation_norm = torch.norm(C_true, p='fro') ** 2
            
            if correlation_norm > 1e-12:
                pos_loss = correlation_diff / correlation_norm
            else:
                # If true correlation is too small, use unnormalized difference
                pos_loss = correlation_diff
            
            total_loss += pos_loss
        
        # Average over positions
        return total_loss / num_pos
    
    def _compute_enhanced_freq_phase_consistency_loss(self, H_pred: torch.Tensor, H_true: torch.Tensor) -> torch.Tensor:
        """
        Compute enhanced frequency phase consistency loss.
        
        Combines subcarrier phase consistency and eigenvalue trajectory losses:
        L_phase = L_phase_consistency + μ * L_eig_trace
        
        Args:
            H_pred: Predicted CSI [num_pos, M, N, S]
            H_true: True CSI [num_pos, M, N, S]
            
        Returns:
            Enhanced phase consistency loss
        """
        # 1. Subcarrier phase consistency loss
        L_phase_consistency = self._compute_subcarrier_phase_consistency_loss(H_pred, H_true)
        
        # 2. Eigenvalue trajectory loss
        L_eig_trace = self._compute_eigenvalue_trajectory_loss(H_pred, H_true)
        
        # Combine with regularization
        return L_phase_consistency + self.mu_eig * L_eig_trace
    
    def _compute_subcarrier_phase_consistency_loss(self, H_pred: torch.Tensor, H_true: torch.Tensor) -> torch.Tensor:
        """
        Compute subcarrier phase consistency loss.
        
        L_phase_consistency = ∑_s ||exp(j·ΔΦ_pred[s]) - exp(j·ΔΦ_true[s])||_F²
        where ΔΦ[s] = ∠H[:,:,s+1] - ∠H[:,:,s]
        
        Args:
            H_pred: Predicted CSI [num_pos, M, N, S]
            H_true: True CSI [num_pos, M, N, S]
            
        Returns:
            Phase consistency loss
        """
        num_pos, M, N, S = H_pred.shape
        
        if S < 2:
            # Need at least 2 subcarriers for phase difference
            return torch.tensor(0.0, device=H_pred.device, dtype=H_pred.dtype)
        
        total_loss = 0.0
        
        for pos_idx in range(num_pos):
            H_p = H_pred[pos_idx]  # [M, N, S]
            H_t = H_true[pos_idx]  # [M, N, S]
            
            pos_loss = 0.0
            
            for s in range(S - 1):
                # Phase differences between adjacent subcarriers
                phase_p_s = torch.angle(H_p[:, :, s])      # [M, N]
                phase_p_s1 = torch.angle(H_p[:, :, s + 1]) # [M, N]
                phase_t_s = torch.angle(H_t[:, :, s])      # [M, N]
                phase_t_s1 = torch.angle(H_t[:, :, s + 1]) # [M, N]
                
                delta_phi_pred = phase_p_s1 - phase_p_s  # [M, N]
                delta_phi_true = phase_t_s1 - phase_t_s  # [M, N]
                
                # Convert to complex exponentials
                exp_delta_pred = torch.exp(1j * delta_phi_pred)  # [M, N]
                exp_delta_true = torch.exp(1j * delta_phi_true)  # [M, N]
                
                # Frobenius norm squared of difference
                diff_norm_sq = torch.norm(exp_delta_pred - exp_delta_true, p='fro') ** 2
                pos_loss += diff_norm_sq
            
            total_loss += pos_loss
        
        # Average over positions and subcarrier pairs
        return total_loss / (num_pos * (S - 1))
    
    def _compute_eigenvalue_trajectory_loss(self, H_pred: torch.Tensor, H_true: torch.Tensor) -> torch.Tensor:
        """
        Compute eigenvalue trajectory loss.
        
        L_eig_trace = ||Λ_pred - Λ_true||_F² + μ·||∇_sΛ_pred - ∇_sΛ_true||_F²
        where Λ[s] = eig(H[:,:,s] × H[:,:,s]ᴴ)
        
        For single BS antenna case (M=1), use singular values of H instead.
        
        Args:
            H_pred: Predicted CSI [num_pos, M, N, S]
            H_true: True CSI [num_pos, M, N, S]
            
        Returns:
            Eigenvalue trajectory loss
        """
        num_pos, M, N, S = H_pred.shape
        
        # Handle single BS antenna case differently
        if M == 1:
            logger.info("Single BS antenna detected, using singular values for eigenvalue trajectory")
            return self._compute_singular_value_trajectory_single_bs(H_pred, H_true)
        
        total_loss = 0.0
        
        for pos_idx in range(num_pos):
            H_p = H_pred[pos_idx]  # [M, N, S]
            H_t = H_true[pos_idx]  # [M, N, S]
            
            # Compute eigenvalues for each subcarrier
            eig_pred = []
            eig_true = []
            
            for s in range(S):
                # Compute H × Hᴴ for each subcarrier
                HH_pred = torch.matmul(H_p[:, :, s], torch.conj(H_p[:, :, s]).transpose(-2, -1))  # [M, M]
                HH_true = torch.matmul(H_t[:, :, s], torch.conj(H_t[:, :, s]).transpose(-2, -1))  # [M, M]
                
                # Compute eigenvalues (real-valued for Hermitian matrices)
                eig_p = torch.linalg.eigvals(HH_pred).real  # [M]
                eig_t = torch.linalg.eigvals(HH_true).real  # [M]
                
                # Sort eigenvalues for consistent comparison
                eig_p_sorted, _ = torch.sort(eig_p, descending=True)
                eig_t_sorted, _ = torch.sort(eig_t, descending=True)
                
                eig_pred.append(eig_p_sorted)
                eig_true.append(eig_t_sorted)
            
            # Stack eigenvalue matrices: [S, M]
            Lambda_pred = torch.stack(eig_pred, dim=0)  # [S, M]
            Lambda_true = torch.stack(eig_true, dim=0)  # [S, M]
            
            # 1. Absolute eigenvalue difference
            eig_diff_loss = torch.norm(Lambda_pred - Lambda_true, p='fro') ** 2
            
            # 2. Eigenvalue gradient difference (if S >= 2)
            eig_grad_loss = 0.0
            if S >= 2:
                # Compute gradients along subcarrier dimension
                grad_pred = Lambda_pred[1:] - Lambda_pred[:-1]  # [S-1, M]
                grad_true = Lambda_true[1:] - Lambda_true[:-1]  # [S-1, M]
                eig_grad_loss = torch.norm(grad_pred - grad_true, p='fro') ** 2
            
            pos_loss = eig_diff_loss + self.mu_eig * eig_grad_loss
            total_loss += pos_loss
        
        # Average over positions
        return total_loss / num_pos
    
    def _compute_singular_value_trajectory_single_bs(self, H_pred: torch.Tensor, H_true: torch.Tensor) -> torch.Tensor:
        """
        Compute singular value trajectory for single BS antenna case.
        
        Uses singular values of H[0,:,s] for each subcarrier s.
        
        Args:
            H_pred: Predicted CSI [num_pos, 1, N, S]
            H_true: True CSI [num_pos, 1, N, S]
            
        Returns:
            Singular value trajectory loss
        """
        num_pos, M, N, S = H_pred.shape
        
        total_loss = 0.0
        
        for pos_idx in range(num_pos):
            H_p = H_pred[pos_idx, 0, :, :]  # [N, S] - squeeze BS dimension
            H_t = H_true[pos_idx, 0, :, :]  # [N, S]
            
            # Compute singular values for each subcarrier
            sv_pred = []
            sv_true = []
            
            for s in range(S):
                # Get channel vector for subcarrier s
                h_pred = H_p[:, s:s+1]  # [N, 1]
                h_true = H_t[:, s:s+1]  # [N, 1]
                
                # Compute singular values (which is just the magnitude for vectors)
                sv_p = torch.linalg.norm(h_pred, dim=0)  # [1]
                sv_t = torch.linalg.norm(h_true, dim=0)  # [1]
                
                sv_pred.append(sv_p)
                sv_true.append(sv_t)
            
            # Stack singular value vectors: [S, 1]
            SV_pred = torch.stack(sv_pred, dim=0)  # [S, 1]
            SV_true = torch.stack(sv_true, dim=0)  # [S, 1]
            
            # 1. Absolute singular value difference
            sv_diff_loss = torch.norm(SV_pred - SV_true, p='fro') ** 2
            
            # 2. Singular value gradient difference (if S >= 2)
            sv_grad_loss = 0.0
            if S >= 2:
                # Compute gradients along subcarrier dimension
                grad_pred = SV_pred[1:] - SV_pred[:-1]  # [S-1, 1]
                grad_true = SV_true[1:] - SV_true[:-1]  # [S-1, 1]
                sv_grad_loss = torch.norm(grad_pred - grad_true, p='fro') ** 2
            
            pos_loss = sv_diff_loss + self.mu_eig * sv_grad_loss
            total_loss += pos_loss
        
        # Average over positions
        return total_loss / num_pos
    
    def _compute_ue_angle_spectrum_loss(self, H_pred: torch.Tensor, H_true: torch.Tensor) -> torch.Tensor:
        """
        Compute UE angle spectrum loss using frequency domain smoothing.
        
        L_smooth_angle = D_KL(p_true_avg || p_pred_avg) + λ·|D_pred - D_true|
        where p_avg(θ) = (1/S)∑_s |FFT(H[:,:,s], axis=0)|²
        
        For single BS antenna case (M=1), use FFT along UE antenna dimension instead.
        
        Args:
            H_pred: Predicted CSI [num_pos, M, N, S]
            H_true: True CSI [num_pos, M, N, S]
            
        Returns:
            UE angle spectrum loss
        """
        num_pos, M, N, S = H_pred.shape
        
        # Handle single BS antenna case differently
        if M == 1:
            logger.info("Single BS antenna detected, using FFT along UE antenna dimension for angle spectrum")
            return self._compute_ue_angle_spectrum_single_bs(H_pred, H_true)
        
        total_loss = 0.0
        
        # Memory-efficient computation for multi-BS antenna case
        # Reshape for batch FFT: [num_pos, M, N, S] -> [num_pos*S, M, N]
        H_p_reshaped = H_pred.permute(0, 3, 1, 2).reshape(num_pos * S, M, N)  # [num_pos*S, M, N]
        H_t_reshaped = H_true.permute(0, 3, 1, 2).reshape(num_pos * S, M, N)  # [num_pos*S, M, N]
        
        # Clear original tensors to save memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Batch FFT along BS antenna dimension (dim=1)
        fft_pred_batch = torch.fft.fft(H_p_reshaped, dim=1)  # [num_pos*S, M, N]
        fft_true_batch = torch.fft.fft(H_t_reshaped, dim=1)  # [num_pos*S, M, N]
        
        # Clear reshaped tensors immediately after FFT
        del H_p_reshaped, H_t_reshaped
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Power spectrum (averaged over UE antennas)
        power_pred_batch = torch.mean(torch.abs(fft_pred_batch) ** 2, dim=2)  # [num_pos*S, M]
        power_true_batch = torch.mean(torch.abs(fft_true_batch) ** 2, dim=2)  # [num_pos*S, M]
        
        # Clear FFT results immediately after power computation
        del fft_pred_batch, fft_true_batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Reshape back to [num_pos, S, M]
        spectra_pred_all = power_pred_batch.reshape(num_pos, S, M)  # [num_pos, S, M]
        spectra_true_all = power_true_batch.reshape(num_pos, S, M)  # [num_pos, S, M]
        
        # Clear power batch tensors
        del power_pred_batch, power_true_batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Process each position (still need loop for KL divergence, but much faster now)
        for pos_idx in range(num_pos):
            spectra_pred = spectra_pred_all[pos_idx]  # [S, M]
            spectra_true = spectra_true_all[pos_idx]  # [S, M]
            
            # Average angle spectra across subcarriers
            p_pred_avg = torch.mean(spectra_pred, dim=0)  # [M]
            p_true_avg = torch.mean(spectra_true, dim=0)  # [M]
            
            # Normalize to probability distributions
            p_pred_avg = p_pred_avg / (torch.sum(p_pred_avg) + 1e-12)
            p_true_avg = p_true_avg / (torch.sum(p_true_avg) + 1e-12)
            
            # Add small epsilon for numerical stability
            p_pred_avg = p_pred_avg + 1e-12
            p_true_avg = p_true_avg + 1e-12
            
            # KL divergence: D_KL(p_true || p_pred) with normalization
            kl_loss = torch.sum(p_true_avg * torch.log(p_true_avg / p_pred_avg))
            # Normalize KL divergence by the number of bins to keep it in reasonable range
            kl_loss = kl_loss / len(p_pred_avg)
            
            # Frequency domain smoothness measure with normalization
            if S > 1:
                D_pred = torch.sum(torch.abs(spectra_pred[1:] - spectra_pred[:-1]) ** 2)
                D_true = torch.sum(torch.abs(spectra_true[1:] - spectra_true[:-1]) ** 2)
                # Normalize by the number of transitions and average power
                avg_power_pred = torch.mean(spectra_pred)
                avg_power_true = torch.mean(spectra_true)
                D_pred = D_pred / ((S - 1) * avg_power_pred + 1e-12)
                D_true = D_true / ((S - 1) * avg_power_true + 1e-12)
                smoothness_loss = torch.abs(D_pred - D_true)
            else:
                smoothness_loss = 0.0
            
            pos_loss = kl_loss + self.lambda_smooth * smoothness_loss
            total_loss += pos_loss
        
        # Clear final intermediate tensors
        del spectra_pred_all, spectra_true_all
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Average over positions
        return total_loss / num_pos
    
    def _compute_ue_angle_spectrum_single_bs(self, H_pred: torch.Tensor, H_true: torch.Tensor) -> torch.Tensor:
        """
        Compute UE angle spectrum for single BS antenna case.
        
        Uses FFT along UE antenna dimension: FFT(H[0,:,s], axis=0)
        
        Args:
            H_pred: Predicted CSI [num_pos, 1, N, S]
            H_true: True CSI [num_pos, 1, N, S]
            
        Returns:
            UE angle spectrum loss
        """
        num_pos, M, N, S = H_pred.shape
        
        # Check if UE has enough antennas for meaningful FFT
        if N == 1:
            logger.warning("Both BS and UE have single antennas, angle spectrum loss may not be meaningful")
            # Return a simple magnitude-based loss instead
            return self._compute_magnitude_loss_single_antenna(H_pred, H_true)
        
        total_loss = 0.0
        
        # Memory-efficient computation with immediate cleanup
        H_p_all = H_pred[:, 0, :, :]  # [num_pos, N, S] - squeeze BS dimension
        H_t_all = H_true[:, 0, :, :]  # [num_pos, N, S]
        
        # Batch FFT along UE antenna dimension for all positions and subcarriers
        # Reshape to [num_pos * S, N] for batch FFT
        H_p_reshaped = H_p_all.transpose(1, 2).reshape(num_pos * S, N)  # [num_pos*S, N]
        H_t_reshaped = H_t_all.transpose(1, 2).reshape(num_pos * S, N)  # [num_pos*S, N]
        
        # Clear intermediate tensors to save memory
        del H_p_all, H_t_all
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Batch FFT computation - much faster than individual FFTs
        fft_pred_batch = torch.fft.fft(H_p_reshaped, dim=1)  # [num_pos*S, N]
        fft_true_batch = torch.fft.fft(H_t_reshaped, dim=1)  # [num_pos*S, N]
        
        # Clear reshaped tensors immediately after FFT
        del H_p_reshaped, H_t_reshaped
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Power spectrum
        power_pred_batch = torch.abs(fft_pred_batch) ** 2  # [num_pos*S, N]
        power_true_batch = torch.abs(fft_true_batch) ** 2  # [num_pos*S, N]
        
        # Clear FFT results immediately after power computation
        del fft_pred_batch, fft_true_batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Reshape back to [num_pos, S, N]
        spectra_pred_all = power_pred_batch.reshape(num_pos, S, N)  # [num_pos, S, N]
        spectra_true_all = power_true_batch.reshape(num_pos, S, N)  # [num_pos, S, N]
        
        # Clear power batch tensors
        del power_pred_batch, power_true_batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Process each position (still need loop for KL divergence, but much faster now)
        for pos_idx in range(num_pos):
            spectra_pred = spectra_pred_all[pos_idx]  # [S, N]
            spectra_true = spectra_true_all[pos_idx]  # [S, N]
            
            # Average angle spectra across subcarriers
            p_pred_avg = torch.mean(spectra_pred, dim=0)  # [N]
            p_true_avg = torch.mean(spectra_true, dim=0)  # [N]
            
            # Normalize to probability distributions
            p_pred_avg = p_pred_avg / (torch.sum(p_pred_avg) + 1e-12)
            p_true_avg = p_true_avg / (torch.sum(p_true_avg) + 1e-12)
            
            # Add small epsilon for numerical stability
            p_pred_avg = p_pred_avg + 1e-12
            p_true_avg = p_true_avg + 1e-12
            
            # KL divergence: D_KL(p_true || p_pred) with normalization
            kl_loss = torch.sum(p_true_avg * torch.log(p_true_avg / p_pred_avg))
            # Normalize KL divergence by the number of bins to keep it in reasonable range
            kl_loss = kl_loss / len(p_pred_avg)
            
            # Frequency domain smoothness measure with normalization
            if S > 1:
                D_pred = torch.sum(torch.abs(spectra_pred[1:] - spectra_pred[:-1]) ** 2)
                D_true = torch.sum(torch.abs(spectra_true[1:] - spectra_true[:-1]) ** 2)
                # Normalize by the number of transitions and average power
                avg_power_pred = torch.mean(spectra_pred)
                avg_power_true = torch.mean(spectra_true)
                D_pred = D_pred / ((S - 1) * avg_power_pred + 1e-12)
                D_true = D_true / ((S - 1) * avg_power_true + 1e-12)
                smoothness_loss = torch.abs(D_pred - D_true)
            else:
                smoothness_loss = 0.0
            
            pos_loss = kl_loss + self.lambda_smooth * smoothness_loss
            total_loss += pos_loss
        
        # Clear final intermediate tensors
        del spectra_pred_all, spectra_true_all
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Average over positions
        return total_loss / num_pos
    
    def _compute_magnitude_loss_single_antenna(self, H_pred: torch.Tensor, H_true: torch.Tensor) -> torch.Tensor:
        """
        Compute simple magnitude-based loss for single BS and UE antenna case.
        
        Args:
            H_pred: Predicted CSI [num_pos, 1, 1, S]
            H_true: True CSI [num_pos, 1, 1, S]
            
        Returns:
            Magnitude-based loss
        """
        num_pos, M, N, S = H_pred.shape
        
        total_loss = 0.0
        
        for pos_idx in range(num_pos):
            H_p = H_pred[pos_idx, 0, 0, :]  # [S] - squeeze spatial dimensions
            H_t = H_true[pos_idx, 0, 0, :]  # [S]
            
            # Compute magnitude spectra
            mag_pred = torch.abs(H_p)  # [S]
            mag_true = torch.abs(H_t)  # [S]
            
            # Normalize to probability distributions
            p_pred = mag_pred / (torch.sum(mag_pred) + 1e-12)
            p_true = mag_true / (torch.sum(mag_true) + 1e-12)
            
            # Add small epsilon for numerical stability
            p_pred = p_pred + 1e-12
            p_true = p_true + 1e-12
            
            # KL divergence: D_KL(p_true || p_pred) with normalization
            kl_loss = torch.sum(p_true * torch.log(p_true / p_pred))
            # Normalize KL divergence by the number of subcarriers
            kl_loss = kl_loss / S
            
            # Frequency domain smoothness measure with normalization
            if S > 1:
                D_pred = torch.sum(torch.abs(mag_pred[1:] - mag_pred[:-1]) ** 2)
                D_true = torch.sum(torch.abs(mag_true[1:] - mag_true[:-1]) ** 2)
                # Normalize by the number of transitions and average magnitude
                avg_mag_pred = torch.mean(mag_pred)
                avg_mag_true = torch.mean(mag_true)
                D_pred = D_pred / ((S - 1) * avg_mag_pred + 1e-12)
                D_true = D_true / ((S - 1) * avg_mag_true + 1e-12)
                smoothness_loss = torch.abs(D_pred - D_true)
            else:
                smoothness_loss = 0.0
            
            pos_loss = kl_loss + self.lambda_smooth * smoothness_loss
            total_loss += pos_loss
        
        # Average over positions
        return total_loss / num_pos


def create_pas2_loss(config: Dict) -> PAS2Loss:
    """
    Create PAS2Loss from configuration.
    
    Args:
        config: Full configuration dictionary containing 'base_station', 'user_equipment', and 'pas2_loss' sections
        
    Returns:
        PAS2Loss instance
    """
    # Extract required configs
    bs_config = config.get('base_station', {})
    ue_config = config.get('user_equipment', {})
    pas2_config = config.get('pas2_loss', {})
    
    if not bs_config:
        raise ValueError("Configuration must contain 'base_station' section for PAS2Loss")
    if not ue_config:
        raise ValueError("Configuration must contain 'user_equipment' section for PAS2Loss")
    
    # Extract antenna counts from configs
    num_bs_antennas = bs_config.get('num_antennas')
    num_ue_antennas = ue_config.get('num_ue_antennas')
    
    if num_bs_antennas is None:
        raise ValueError("bs_config must contain 'num_antennas'")
    if num_ue_antennas is None:
        raise ValueError("ue_config must contain 'num_ue_antennas'")
    
    return PAS2Loss(
        num_bs_antennas=num_bs_antennas,
        num_ue_antennas=num_ue_antennas,
        freq_spatial_weight=pas2_config.get('freq_spatial_weight', 0.40),
        phase_consistency_weight=pas2_config.get('phase_consistency_weight', 0.35),
        angle_spectrum_weight=pas2_config.get('angle_spectrum_weight', 0.25),
        lambda_smooth=pas2_config.get('lambda_smooth', 0.1),
        mu_eig=pas2_config.get('mu_eig', 0.1)
    )
