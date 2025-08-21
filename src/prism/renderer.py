"""
Renderer for Prism: Wideband RF Neural Radiance Fields.
Handles visualization and rendering of OFDM signals across multiple subcarriers.

This module provides comprehensive visualization capabilities for the Prism model,
enabling researchers and engineers to:
- Analyze model performance across different subcarriers
- Visualize MIMO channel characteristics
- Generate frequency spectrum plots
- Create 3D spatial heatmaps
- Monitor training progress
- Compare predictions vs ground truth
- Generate publication-ready visualizations

The renderer is designed to handle the unique characteristics of wideband RF signals
and provide insights into both the model's learning process and its final performance.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logging for the renderer module
logger = logging.getLogger(__name__)

class PrismRenderer:
    """
    Renderer class for visualizing Prism model outputs and OFDM signals.
    
    This class provides a comprehensive suite of visualization methods
    specifically designed for wideband RF signals and neural radiance fields.
    It handles:
    - Subcarrier response visualization
    - MIMO channel matrix analysis
    - Frequency spectrum plotting
    - 3D spatial heatmaps
    - Training progress monitoring
    - Performance comparison plots
    - Animation generation
    
    The renderer is optimized for OFDM systems with multiple subcarriers
    and provides both qualitative and quantitative analysis tools.
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        """
        Initialize the Prism renderer.
        
        This method sets up the renderer with configuration parameters
        and prepares color maps for consistent visualization across
        different plotting functions.
        
        Args:
            config: Configuration dictionary containing model parameters
                - model.num_subcarriers: Number of OFDM subcarriers
                - model.num_ue_antennas: Number of UE antennas
                - model.num_bs_antennas: Number of BS antennas
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device
        
        # Extract model configuration parameters for rendering
        self.num_subcarriers = config['model'].get('num_subcarriers', 1024)
        self.num_ue_antennas = config['model'].get('num_ue_antennas', 2)
        self.num_bs_antennas = config['model'].get('num_bs_antennas', 4)
        
        # Create color maps for consistent visualization
        # Viridis provides good color differentiation and is colorblind-friendly
        # Limit to 100 colors to avoid overwhelming visualizations
        self.subcarrier_colors = plt.cm.viridis(np.linspace(0, 1, min(self.num_subcarriers, 100)))
        
    def render_subcarrier_responses(self, predictions: torch.Tensor, targets: torch.Tensor,
                                  save_path: str = 'subcarrier_responses.png'):
        """
        Render subcarrier responses comparison between predictions and ground truth.
        
        This method creates side-by-side comparisons of predicted vs actual
        subcarrier responses, which is crucial for:
        - Understanding model performance across the frequency spectrum
        - Identifying frequency bands where the model excels or struggles
        - Debugging frequency-specific prediction errors
        - Validating the model's ability to capture OFDM characteristics
        
        Args:
            predictions: Predicted subcarrier responses [batch_size, num_subcarriers]
                Model outputs for each subcarrier frequency
            targets: Ground truth subcarrier responses [batch_size, num_subcarriers]
                True subcarrier responses from measurements or simulation
            save_path: Path where the visualization should be saved
                Supports common image formats (PNG, JPG, PDF, SVG)
        """
        # Convert PyTorch tensors to NumPy arrays for matplotlib plotting
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Sample a few examples for visualization to avoid cluttered plots
        # This provides a representative view without overwhelming detail
        num_examples = min(5, len(pred_np))
        sample_indices = np.random.choice(len(pred_np), num_examples, replace=False)
        
        # Create subplot layout for multiple examples
        fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3 * num_examples))
        if num_examples == 1:
            axes = [axes]  # Handle single example case
        
        # Plot each sampled example
        for i, idx in enumerate(sample_indices):
            ax = axes[i]
            
            # Create subcarrier index array for x-axis
            subcarrier_indices = np.arange(self.num_subcarriers)
            
            # Plot ground truth (solid blue line) and predictions (dashed red line)
            ax.plot(subcarrier_indices, target_np[idx], 'b-', label='Ground Truth', linewidth=2)
            ax.plot(subcarrier_indices, pred_np[idx], 'r--', label='Prediction', linewidth=2)
            
            # Set plot labels and title
            ax.set_xlabel('Subcarrier Index')
            ax.set_ylabel('Response Magnitude')
            ax.set_title(f'Sample {i+1}: Subcarrier Responses')
            ax.legend()
            ax.grid(True, alpha=0.3)  # Semi-transparent grid for readability
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save high-quality plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Subcarrier responses visualization saved to {save_path}")
    
    def render_mimo_channel(self, mimo_channel: torch.Tensor, save_path: str = 'mimo_channel.png'):
        """
        Render MIMO channel matrix visualization.
        
        This method creates heatmap visualizations of the MIMO channel matrix,
        which is essential for understanding:
        - Spatial diversity characteristics
        - Antenna coupling effects
        - Channel correlation patterns
        - MIMO system performance
        
        The visualization shows how signals propagate between different
        antenna pairs across the spatial domain.
        
        Args:
            mimo_channel: MIMO channel matrix [batch_size, num_ue_antennas, num_bs_antennas]
                Contains complex channel coefficients between antenna pairs
            save_path: Path where the visualization should be saved
        """
        # Convert PyTorch tensor to NumPy array for plotting
        mimo_np = mimo_channel.detach().cpu().numpy()
        
        # Sample a few examples for visualization
        num_examples = min(4, len(mimo_np))
        sample_indices = np.random.choice(len(mimo_np), num_examples, replace=False)
        
        # Create 2x2 subplot grid for multiple examples
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()  # Flatten for easier indexing
        
        # Plot each sampled MIMO channel matrix
        for i, idx in enumerate(sample_indices):
            if i >= num_examples:
                break
                
            ax = axes[i]
            channel_matrix = mimo_np[idx]
            
            # Create heatmap using imshow
            # Viridis colormap provides good contrast for channel magnitude visualization
            im = ax.imshow(channel_matrix, cmap='viridis', aspect='auto')
            ax.set_xlabel('BS Antenna Index')
            ax.set_ylabel('UE Antenna Index')
            ax.set_title(f'Sample {i+1}: MIMO Channel Matrix')
            
            # Add colorbar to show magnitude scale
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots for clean appearance
        for i in range(num_examples, 4):
            axes[i].set_visible(False)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"MIMO channel visualization saved to {save_path}")
    
    def render_frequency_spectrum(self, subcarrier_responses: torch.Tensor,
                                 frequencies: Optional[np.ndarray] = None,
                                 save_path: str = 'frequency_spectrum.png'):
        """
        Render frequency spectrum visualization.
        
        This method creates frequency-domain plots showing how the model
        predicts signal strength across the frequency spectrum. This is
        crucial for:
        - Understanding frequency-selective fading
        - Analyzing bandwidth utilization
        - Validating OFDM signal characteristics
        - Identifying frequency-dependent performance patterns
        
        Args:
            subcarrier_responses: Subcarrier responses [batch_size, num_subcarriers]
                Model predictions for each subcarrier frequency
            frequencies: Frequency values for each subcarrier (optional)
                If not provided, assumes WiFi-like 2.4 GHz band with 20 MHz bandwidth
            save_path: Path where the visualization should be saved
        """
        # Convert PyTorch tensor to NumPy array for plotting
        responses_np = subcarrier_responses.detach().cpu().numpy()
        
        # Generate frequencies if not provided
        if frequencies is None:
            # Assume WiFi-like frequency range (2.4 GHz band)
            # These parameters can be customized for different OFDM systems
            center_freq = 2.4e9  # 2.4 GHz center frequency
            bandwidth = 20e6     # 20 MHz bandwidth
            frequencies = np.linspace(center_freq - bandwidth/2, center_freq + bandwidth/2, self.num_subcarriers)
        
        # Sample a few examples for visualization
        num_examples = min(3, len(responses_np))
        sample_indices = np.random.choice(len(responses_np), num_examples, replace=False)
        
        # Create subplot layout for multiple examples
        fig, axes = plt.subplots(num_examples, 1, figsize=(12, 4 * num_examples))
        if num_examples == 1:
            axes = [axes]  # Handle single example case
        
        # Plot each sampled frequency spectrum
        for i, idx in enumerate(sample_indices):
            ax = axes[i]
            
            # Plot frequency spectrum with filled area for better visualization
            ax.plot(frequencies / 1e9, responses_np[idx], 'b-', linewidth=2)
            ax.fill_between(frequencies / 1e9, responses_np[idx], alpha=0.3, color='blue')
            
            # Set plot labels and title
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel('Response Magnitude')
            ax.set_title(f'Sample {i+1}: Frequency Spectrum')
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits to focus on the frequency band of interest
            ax.set_xlim(frequencies.min() / 1e9, frequencies.max() / 1e9)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Frequency spectrum visualization saved to {save_path}")
    
    def render_3d_position_heatmap(self, positions: torch.Tensor, responses: torch.Tensor,
                                   subcarrier_idx: int = 0, save_path: str = 'position_heatmap.png'):
        """
        Render 3D position heatmap for a specific subcarrier.
        
        This method creates 3D spatial visualizations showing how signal
        strength varies across 3D space for a particular subcarrier.
        This is essential for understanding:
        - Spatial propagation patterns
        - Shadowing effects
        - Multi-path characteristics
        - Coverage area analysis
        
        Args:
            positions: 3D positions [batch_size, 3]
                Spatial coordinates (x, y, z) for each sample
            responses: Subcarrier responses [batch_size, num_subcarriers]
                Signal responses for all subcarriers
            subcarrier_idx: Index of subcarrier to visualize
                Allows focusing on specific frequency components
            save_path: Path where the visualization should be saved
        """
        # Convert PyTorch tensors to NumPy arrays for plotting
        pos_np = positions.detach().cpu().numpy()
        resp_np = responses.detach().cpu().numpy()
        
        # Extract responses for the specified subcarrier
        subcarrier_responses = resp_np[:, subcarrier_idx]
        
        # Create 3D figure and subplot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create 3D scatter plot with color mapping
        # Color represents signal strength at each spatial position
        scatter = ax.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2],
                           c=subcarrier_responses, cmap='viridis', s=50, alpha=0.7)
        
        # Set axis labels and title
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(f'3D Position Heatmap - Subcarrier {subcarrier_idx}')
        
        # Add colorbar to show signal strength scale
        plt.colorbar(scatter, ax=ax, label='Response Magnitude')
        
        # Save high-quality 3D plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"3D position heatmap saved to {save_path}")
    
    def render_training_progress(self, train_losses: List[float], val_losses: List[float],
                                save_path: str = 'training_progress.png'):
        """
        Render comprehensive training progress visualization.
        
        This method creates a multi-panel plot showing various aspects
        of the training process, including:
        - Training and validation loss curves
        - Loss ratio analysis
        - Loss difference trends
        - Log-scale learning curves
        
        This comprehensive view helps identify:
        - Overfitting/underfitting patterns
        - Convergence behavior
        - Learning rate effectiveness
        - Training stability issues
        
        Args:
            train_losses: List of training losses for each epoch
            val_losses: List of validation losses for each epoch
            save_path: Path where the visualization should be saved
        """
        # Create epoch index array
        epochs = range(1, len(train_losses) + 1)
        
        # Create 2x2 subplot grid for comprehensive analysis
        plt.figure(figsize=(12, 8))
        
        # Panel 1: Training and validation loss curves
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Panel 2: Loss ratio (validation/training)
        # Values > 1 indicate potential overfitting
        plt.subplot(2, 2, 2)
        loss_ratio = [v/t for t, v in zip(train_losses, val_losses)]
        plt.plot(epochs, loss_ratio, 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Val Loss / Train Loss')
        plt.title('Loss Ratio')
        plt.grid(True, alpha=0.3)
        
        # Panel 3: Loss difference (validation - training)
        # Positive values indicate validation loss > training loss
        plt.subplot(2, 2, 3)
        loss_diff = [v - t for t, v in zip(train_losses, val_losses)]
        plt.plot(epochs, loss_diff, 'm-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Val Loss - Train Loss')
        plt.title('Loss Difference')
        plt.grid(True, alpha=0.3)
        
        # Panel 4: Log-scale learning curves
        # Useful for identifying convergence patterns
        plt.subplot(2, 2, 4)
        plt.semilogy(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.semilogy(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Learning Curves (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training progress visualization saved to {save_path}")
    
    def render_subcarrier_comparison(self, predictions: torch.Tensor, targets: torch.Tensor,
                                   save_path: str = 'subcarrier_comparison.png'):
        """
        Render detailed subcarrier comparison visualization.
        
        This method creates a comprehensive 6-panel analysis showing:
        - Per-subcarrier MSE and MAE
        - Error distribution analysis
        - Sample response comparisons
        - Prediction vs ground truth correlation
        - Subcarrier variability analysis
        
        This detailed view is essential for understanding:
        - Frequency-dependent model performance
        - Error patterns across the spectrum
        - Model bias and variance characteristics
        - Subcarrier-specific optimization needs
        
        Args:
            predictions: Predicted subcarrier responses [batch_size, num_subcarriers]
            targets: Ground truth subcarrier responses [batch_size, num_subcarriers]
            save_path: Path where the visualization should be saved
        """
        # Convert PyTorch tensors to NumPy arrays for plotting
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Calculate per-subcarrier performance metrics
        # These reveal frequency-dependent performance patterns
        mse_per_subcarrier = np.mean((pred_np - target_np) ** 2, axis=0)
        mae_per_subcarrier = np.mean(np.abs(pred_np - target_np), axis=0)
        
        # Create comprehensive 2x3 subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Panel 1: MSE per subcarrier
        axes[0, 0].plot(mse_per_subcarrier, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Subcarrier Index')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].set_title('MSE per Subcarrier')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Panel 2: MAE per subcarrier
        axes[0, 1].plot(mae_per_subcarrier, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Subcarrier Index')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('MAE per Subcarrier')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Panel 3: Error distribution histogram
        errors = pred_np - target_np
        axes[0, 2].hist(errors.flatten(), bins=50, alpha=0.7, color='green')
        axes[0, 2].set_xlabel('Prediction Error')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Error Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Panel 4: Sample response comparisons
        num_samples = min(3, len(pred_np))
        for i in range(num_samples):
            axes[1, 0].plot(target_np[i], 'b-', alpha=0.7, label=f'Target {i+1}' if i == 0 else "")
            axes[1, 0].plot(pred_np[i], 'r--', alpha=0.7, label=f'Pred {i+1}' if i == 0 else "")
        axes[1, 0].set_xlabel('Subcarrier Index')
        axes[1, 0].set_ylabel('Response Magnitude')
        axes[1, 0].set_title('Sample Responses')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Panel 5: Correlation plot (prediction vs ground truth)
        axes[1, 1].scatter(target_np.flatten(), pred_np.flatten(), alpha=0.1, s=1)
        # Add perfect prediction line for reference
        axes[1, 1].plot([target_np.min(), target_np.max()], [target_np.min(), target_np.max()], 'r--', linewidth=2)
        axes[1, 1].set_xlabel('Ground Truth')
        axes[1, 1].set_ylabel('Prediction')
        axes[1, 1].set_title('Prediction vs Ground Truth')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Panel 6: Subcarrier variability analysis
        target_std = np.std(target_np, axis=0)
        pred_std = np.std(pred_np, axis=0)
        axes[1, 2].plot(target_std, 'b-', label='Target Std', linewidth=2)
        axes[1, 2].plot(pred_std, 'r-', label='Pred Std', linewidth=2)
        axes[1, 2].set_xlabel('Subcarrier Index')
        axes[1, 2].set_ylabel('Standard Deviation')
        axes[1, 2].set_title('Variability per Subcarrier')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Subcarrier comparison visualization saved to {save_path}")
    
    def create_animation(self, predictions: torch.Tensor, targets: torch.Tensor,
                        save_path: str = 'subcarrier_animation.gif'):
        """
        Create animation of subcarrier responses over time.
        
        This method generates animated visualizations showing how
        subcarrier responses evolve, which is useful for:
        - Demonstrating model behavior
        - Creating presentation materials
        - Analyzing temporal patterns
        - Visualizing training convergence
        
        Note: Requires matplotlib.animation module
        
        Args:
            predictions: Predicted subcarrier responses [batch_size, num_subcarriers]
            targets: Ground truth subcarrier responses [batch_size, num_subcarriers]
            save_path: Path where the animation should be saved
                Supports GIF format for easy sharing and viewing
        """
        # Try to import animation module
        try:
            import matplotlib.animation as animation
        except ImportError:
            logger.warning("matplotlib.animation not available, skipping animation creation")
            return
        
        # Convert PyTorch tensors to NumPy arrays for plotting
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Sample a few examples for animation
        num_examples = min(3, len(pred_np))
        sample_indices = np.random.choice(len(pred_np), num_examples, replace=False)
        
        # Create figure and subplot for animation
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define animation function
        def animate(frame):
            ax.clear()
            
            # Plot current frame with multiple examples
            for i, idx in enumerate(sample_indices):
                # Vary alpha for visual distinction between examples
                alpha = 0.3 + 0.4 * (i / num_examples)
                ax.plot(target_np[idx], 'b-', alpha=alpha, linewidth=2, label=f'Target {i+1}' if frame == 0 else "")
                ax.plot(pred_np[idx], 'r--', alpha=alpha, linewidth=2, label=f'Pred {i+1}' if frame == 0 else "")
            
            # Set plot properties
            ax.set_xlabel('Subcarrier Index')
            ax.set_ylabel('Response Magnitude')
            ax.set_title(f'Subcarrier Responses - Frame {frame+1}')
            ax.grid(True, alpha=0.3)
            
            # Add legend only on first frame
            if frame == 0:
                ax.legend()
        
        # Create animation object
        # 10 frames with 500ms intervals, repeating
        anim = animation.FuncAnimation(fig, animate, frames=10, interval=500, repeat=True)
        
        # Save animation as GIF
        anim.save(save_path, writer='pillow', fps=2)
        plt.close()
        
        logger.info(f"Subcarrier animation saved to {save_path}")
    
    def render_all_visualizations(self, model_outputs: Dict[str, torch.Tensor],
                                 targets: torch.Tensor, positions: torch.Tensor,
                                 train_losses: Optional[List[float]] = None,
                                 val_losses: Optional[List[float]] = None,
                                 output_dir: str = 'visualizations'):
        """
        Render all available visualizations in batch.
        
        This method provides a convenient way to generate all visualization
        types at once, creating a comprehensive analysis package. It's useful for:
        - Final model evaluation
        - Report generation
        - Presentation preparation
        - Comprehensive performance analysis
        
        Args:
            model_outputs: Dictionary containing model outputs:
                - subcarrier_responses: Predicted subcarrier responses
                - mimo_channel: MIMO channel matrix
                - attenuation_features: Attenuation network features
                - radiance_features: Radiance network features
            targets: Ground truth targets for comparison
            positions: 3D spatial positions for spatial analysis
            train_losses: Training loss history (optional)
            val_losses: Validation loss history (optional)
            output_dir: Directory to save all visualizations
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract outputs from model_outputs dictionary
        subcarrier_responses = model_outputs['subcarrier_responses']
        mimo_channel = model_outputs['mimo_channel']
        
        # Render all available visualizations
        logger.info("Generating comprehensive visualizations...")
        
        # 1. Subcarrier responses comparison
        self.render_subcarrier_responses(
            subcarrier_responses, targets,
            os.path.join(output_dir, 'subcarrier_responses.png')
        )
        
        # 2. MIMO channel visualization
        self.render_mimo_channel(
            mimo_channel,
            os.path.join(output_dir, 'mimo_channel.png')
        )
        
        # 3. Frequency spectrum analysis
        self.render_frequency_spectrum(
            subcarrier_responses,
            save_path=os.path.join(output_dir, 'frequency_spectrum.png')
        )
        
        # 4. 3D spatial heatmap
        self.render_3d_position_heatmap(
            positions, subcarrier_responses,
            save_path=os.path.join(output_dir, 'position_heatmap.png')
        )
        
        # 5. Detailed subcarrier comparison
        self.render_subcarrier_comparison(
            subcarrier_responses, targets,
            save_path=os.path.join(output_dir, 'subcarrier_comparison.png')
        )
        
        # 6. Training progress (if available)
        if train_losses and val_losses:
            self.render_training_progress(
                train_losses, val_losses,
                save_path=os.path.join(output_dir, 'training_progress.png')
            )
        
        # 7. Animation (optional, may take longer)
        try:
            self.create_animation(
                subcarrier_responses, targets,
                os.path.join(output_dir, 'subcarrier_animation.gif')
            )
        except Exception as e:
            logger.warning(f"Animation creation failed: {e}")
        
        logger.info(f"All visualizations saved to {output_dir}")
        logger.info("Visualization package includes:")
        logger.info("- Subcarrier response comparisons")
        logger.info("- MIMO channel analysis")
        logger.info("- Frequency spectrum plots")
        logger.info("- 3D spatial heatmaps")
        logger.info("- Performance analysis plots")
        if train_losses and val_losses:
            logger.info("- Training progress analysis")
        logger.info("- Animated visualizations")
