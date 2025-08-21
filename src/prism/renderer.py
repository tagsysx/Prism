"""
Renderer for Prism: Wideband RF Neural Radiance Fields.
Handles visualization and rendering of OFDM signals across multiple subcarriers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class PrismRenderer:
    """
    Renderer class for visualizing Prism model outputs and OFDM signals.
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        """
        Initialize the Prism renderer.
        
        Args:
            config: Configuration dictionary
            device: Device to run computations on
        """
        self.config = config
        self.device = device
        
        # Rendering parameters
        self.num_subcarriers = config['model'].get('num_subcarriers', 1024)
        self.num_ue_antennas = config['model'].get('num_ue_antennas', 2)
        self.num_bs_antennas = config['model'].get('num_bs_antennas', 4)
        
        # Color maps for visualization
        self.subcarrier_colors = plt.cm.viridis(np.linspace(0, 1, min(self.num_subcarriers, 100)))
        
    def render_subcarrier_responses(self, predictions: torch.Tensor, targets: torch.Tensor,
                                  save_path: str = 'subcarrier_responses.png'):
        """
        Render subcarrier responses comparison.
        
        Args:
            predictions: Predicted subcarrier responses [batch_size, num_subcarriers]
            targets: Ground truth subcarrier responses [batch_size, num_subcarriers]
            save_path: Path to save the visualization
        """
        # Convert to numpy for plotting
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Sample a few examples for visualization
        num_examples = min(5, len(pred_np))
        sample_indices = np.random.choice(len(pred_np), num_examples, replace=False)
        
        fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3 * num_examples))
        if num_examples == 1:
            axes = [axes]
        
        for i, idx in enumerate(sample_indices):
            ax = axes[i]
            
            # Plot predictions vs targets
            subcarrier_indices = np.arange(self.num_subcarriers)
            ax.plot(subcarrier_indices, target_np[idx], 'b-', label='Ground Truth', linewidth=2)
            ax.plot(subcarrier_indices, pred_np[idx], 'r--', label='Prediction', linewidth=2)
            
            ax.set_xlabel('Subcarrier Index')
            ax.set_ylabel('Response Magnitude')
            ax.set_title(f'Sample {i+1}: Subcarrier Responses')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Subcarrier responses visualization saved to {save_path}")
    
    def render_mimo_channel(self, mimo_channel: torch.Tensor, save_path: str = 'mimo_channel.png'):
        """
        Render MIMO channel matrix visualization.
        
        Args:
            mimo_channel: MIMO channel matrix [batch_size, num_ue_antennas, num_bs_antennas]
            save_path: Path to save the visualization
        """
        # Convert to numpy for plotting
        mimo_np = mimo_channel.detach().cpu().numpy()
        
        # Sample a few examples
        num_examples = min(4, len(mimo_np))
        sample_indices = np.random.choice(len(mimo_np), num_examples, replace=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(sample_indices):
            if i >= num_examples:
                break
                
            ax = axes[i]
            channel_matrix = mimo_np[idx]
            
            # Create heatmap
            im = ax.imshow(channel_matrix, cmap='viridis', aspect='auto')
            ax.set_xlabel('BS Antenna Index')
            ax.set_ylabel('UE Antenna Index')
            ax.set_title(f'Sample {i+1}: MIMO Channel Matrix')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for i in range(num_examples, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"MIMO channel visualization saved to {save_path}")
    
    def render_frequency_spectrum(self, subcarrier_responses: torch.Tensor,
                                 frequencies: Optional[np.ndarray] = None,
                                 save_path: str = 'frequency_spectrum.png'):
        """
        Render frequency spectrum visualization.
        
        Args:
            subcarrier_responses: Subcarrier responses [batch_size, num_subcarriers]
            frequencies: Frequency values for each subcarrier (optional)
            save_path: Path to save the visualization
        """
        # Convert to numpy for plotting
        responses_np = subcarrier_responses.detach().cpu().numpy()
        
        # Generate frequencies if not provided
        if frequencies is None:
            # Assume WiFi-like frequency range (2.4 GHz band)
            center_freq = 2.4e9  # 2.4 GHz
            bandwidth = 20e6     # 20 MHz
            frequencies = np.linspace(center_freq - bandwidth/2, center_freq + bandwidth/2, self.num_subcarriers)
        
        # Sample a few examples
        num_examples = min(3, len(responses_np))
        sample_indices = np.random.choice(len(responses_np), num_examples, replace=False)
        
        fig, axes = plt.subplots(num_examples, 1, figsize=(12, 4 * num_examples))
        if num_examples == 1:
            axes = [axes]
        
        for i, idx in enumerate(sample_indices):
            ax = axes[i]
            
            # Plot frequency spectrum
            ax.plot(frequencies / 1e9, responses_np[idx], 'b-', linewidth=2)
            ax.fill_between(frequencies / 1e9, responses_np[idx], alpha=0.3, color='blue')
            
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel('Response Magnitude')
            ax.set_title(f'Sample {i+1}: Frequency Spectrum')
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits
            ax.set_xlim(frequencies.min() / 1e9, frequencies.max() / 1e9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Frequency spectrum visualization saved to {save_path}")
    
    def render_3d_position_heatmap(self, positions: torch.Tensor, responses: torch.Tensor,
                                   subcarrier_idx: int = 0, save_path: str = 'position_heatmap.png'):
        """
        Render 3D position heatmap for a specific subcarrier.
        
        Args:
            positions: 3D positions [batch_size, 3]
            responses: Subcarrier responses [batch_size, num_subcarriers]
            subcarrier_idx: Index of subcarrier to visualize
            save_path: Path to save the visualization
        """
        # Convert to numpy for plotting
        pos_np = positions.detach().cpu().numpy()
        resp_np = responses.detach().cpu().numpy()
        
        # Get responses for the specified subcarrier
        subcarrier_responses = resp_np[:, subcarrier_idx]
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot with color mapping
        scatter = ax.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2],
                           c=subcarrier_responses, cmap='viridis', s=50, alpha=0.7)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(f'3D Position Heatmap - Subcarrier {subcarrier_idx}')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Response Magnitude')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"3D position heatmap saved to {save_path}")
    
    def render_training_progress(self, train_losses: List[float], val_losses: List[float],
                                save_path: str = 'training_progress.png'):
        """
        Render training progress visualization.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            save_path: Path to save the visualization
        """
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(12, 8))
        
        # Plot training and validation losses
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot loss ratio
        plt.subplot(2, 2, 2)
        loss_ratio = [v/t for t, v in zip(train_losses, val_losses)]
        plt.plot(epochs, loss_ratio, 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Val Loss / Train Loss')
        plt.title('Loss Ratio')
        plt.grid(True, alpha=0.3)
        
        # Plot loss difference
        plt.subplot(2, 2, 3)
        loss_diff = [v - t for t, v in zip(train_losses, val_losses)]
        plt.plot(epochs, loss_diff, 'm-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Val Loss - Train Loss')
        plt.title('Loss Difference')
        plt.grid(True, alpha=0.3)
        
        # Plot learning curves in log scale
        plt.subplot(2, 2, 4)
        plt.semilogy(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.semilogy(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Learning Curves (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training progress visualization saved to {save_path}")
    
    def render_subcarrier_comparison(self, predictions: torch.Tensor, targets: torch.Tensor,
                                   save_path: str = 'subcarrier_comparison.png'):
        """
        Render detailed subcarrier comparison visualization.
        
        Args:
            predictions: Predicted subcarrier responses [batch_size, num_subcarriers]
            targets: Ground truth subcarrier responses [batch_size, num_subcarriers]
            save_path: Path to save the visualization
        """
        # Convert to numpy for plotting
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Calculate per-subcarrier metrics
        mse_per_subcarrier = np.mean((pred_np - target_np) ** 2, axis=0)
        mae_per_subcarrier = np.mean(np.abs(pred_np - target_np), axis=0)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Overall comparison
        axes[0, 0].plot(mse_per_subcarrier, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Subcarrier Index')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].set_title('MSE per Subcarrier')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. MAE comparison
        axes[0, 1].plot(mae_per_subcarrier, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Subcarrier Index')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('MAE per Subcarrier')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        errors = pred_np - target_np
        axes[0, 2].hist(errors.flatten(), bins=50, alpha=0.7, color='green')
        axes[0, 2].set_xlabel('Prediction Error')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Error Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Sample responses (first few samples)
        num_samples = min(3, len(pred_np))
        for i in range(num_samples):
            axes[1, 0].plot(target_np[i], 'b-', alpha=0.7, label=f'Target {i+1}' if i == 0 else "")
            axes[1, 0].plot(pred_np[i], 'r--', alpha=0.7, label=f'Pred {i+1}' if i == 0 else "")
        axes[1, 0].set_xlabel('Subcarrier Index')
        axes[1, 0].set_ylabel('Response Magnitude')
        axes[1, 0].set_title('Sample Responses')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Correlation plot
        axes[1, 1].scatter(target_np.flatten(), pred_np.flatten(), alpha=0.1, s=1)
        axes[1, 1].plot([target_np.min(), target_np.max()], [target_np.min(), target_np.max()], 'r--', linewidth=2)
        axes[1, 1].set_xlabel('Ground Truth')
        axes[1, 1].set_ylabel('Prediction')
        axes[1, 1].set_title('Prediction vs Ground Truth')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Subcarrier statistics
        target_std = np.std(target_np, axis=0)
        pred_std = np.std(pred_np, axis=0)
        axes[1, 2].plot(target_std, 'b-', label='Target Std', linewidth=2)
        axes[1, 2].plot(pred_std, 'r-', label='Pred Std', linewidth=2)
        axes[1, 2].set_xlabel('Subcarrier Index')
        axes[1, 2].set_ylabel('Standard Deviation')
        axes[1, 2].set_title('Variability per Subcarrier')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Subcarrier comparison visualization saved to {save_path}")
    
    def create_animation(self, predictions: torch.Tensor, targets: torch.Tensor,
                        save_path: str = 'subcarrier_animation.gif'):
        """
        Create animation of subcarrier responses over time.
        
        Args:
            predictions: Predicted subcarrier responses [batch_size, num_subcarriers]
            targets: Ground truth subcarrier responses [batch_size, num_subcarriers]
            save_path: Path to save the animation
        """
        try:
            import matplotlib.animation as animation
        except ImportError:
            logger.warning("matplotlib.animation not available, skipping animation creation")
            return
        
        # Convert to numpy for plotting
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Sample a few examples
        num_examples = min(3, len(pred_np))
        sample_indices = np.random.choice(len(pred_np), num_examples, replace=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def animate(frame):
            ax.clear()
            
            # Plot current frame
            for i, idx in enumerate(sample_indices):
                alpha = 0.3 + 0.4 * (i / num_examples)
                ax.plot(target_np[idx], 'b-', alpha=alpha, linewidth=2, label=f'Target {i+1}' if frame == 0 else "")
                ax.plot(pred_np[idx], 'r--', alpha=alpha, linewidth=2, label=f'Pred {i+1}' if frame == 0 else "")
            
            ax.set_xlabel('Subcarrier Index')
            ax.set_ylabel('Response Magnitude')
            ax.set_title(f'Subcarrier Responses - Frame {frame+1}')
            ax.grid(True, alpha=0.3)
            
            if frame == 0:
                ax.legend()
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=10, interval=500, repeat=True)
        
        # Save animation
        anim.save(save_path, writer='pillow', fps=2)
        plt.close()
        
        logger.info(f"Subcarrier animation saved to {save_path}")
    
    def render_all_visualizations(self, model_outputs: Dict[str, torch.Tensor],
                                 targets: torch.Tensor, positions: torch.Tensor,
                                 train_losses: Optional[List[float]] = None,
                                 val_losses: Optional[List[float]] = None,
                                 output_dir: str = 'visualizations'):
        """
        Render all available visualizations.
        
        Args:
            model_outputs: Dictionary containing model outputs
            targets: Ground truth targets
            positions: 3D positions
            train_losses: Training losses (optional)
            val_losses: Validation losses (optional)
            output_dir: Directory to save visualizations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract outputs
        subcarrier_responses = model_outputs['subcarrier_responses']
        mimo_channel = model_outputs['mimo_channel']
        
        # Render all visualizations
        self.render_subcarrier_responses(
            subcarrier_responses, targets,
            os.path.join(output_dir, 'subcarrier_responses.png')
        )
        
        self.render_mimo_channel(
            mimo_channel,
            os.path.join(output_dir, 'mimo_channel.png')
        )
        
        self.render_frequency_spectrum(
            subcarrier_responses,
            save_path=os.path.join(output_dir, 'frequency_spectrum.png')
        )
        
        self.render_3d_position_heatmap(
            positions, subcarrier_responses,
            save_path=os.path.join(output_dir, 'position_heatmap.png')
        )
        
        self.render_subcarrier_comparison(
            subcarrier_responses, targets,
            save_path=os.path.join(output_dir, 'subcarrier_comparison.png')
        )
        
        if train_losses and val_losses:
            self.render_training_progress(
                train_losses, val_losses,
                save_path=os.path.join(output_dir, 'training_progress.png')
            )
        
        logger.info(f"All visualizations saved to {output_dir}")
