"""
MLP-Based Direction Sampling for Ray Tracing

This module implements the intelligent direction sampling strategy using a shallow
Multi-Layer Perceptron (MLP) to optimize ray tracing efficiency as described
in the design document.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DirectionSamplingConfig:
    """Configuration for MLP direction sampling."""
    input_dim: int = 128  # Antenna embedding dimension
    hidden_dim: int = 256  # Hidden layer dimension
    num_hidden_layers: int = 2  # Number of hidden layers (2-3 as per design)
    output_dim: int = 648  # A × B directions (36 × 18 = 648)
    activation: str = 'relu'  # Activation function
    dropout_rate: float = 0.1  # Dropout rate for regularization
    threshold: float = 0.5  # Threshold for binary indicators

class MLPDirectionSampler(nn.Module):
    """
    MLP-based direction sampler for intelligent ray tracing optimization.
    
    This module implements the design document specification for MLP-based
    direction sampling that automatically learns optimal direction selection
    based on antenna embedding parameters.
    
    Architecture:
    - Input layer: Accepts antenna embedding parameter C (128D)
    - Hidden layers: 2-3 fully connected layers with ReLU activation
    - Output layer: Produces A×B indicator matrix M_ij
    - Activation: Sigmoid activation followed by thresholding
    """
    
    def __init__(self, config: DirectionSamplingConfig):
        """
        Initialize MLP direction sampler.
        
        Args:
            config: Configuration for the MLP architecture
        """
        super(MLPDirectionSampler, self).__init__()
        self.config = config
        
        # Build the MLP architecture
        layers = []
        
        # Input layer
        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout_rate))
        
        # Hidden layers
        for _ in range(config.num_hidden_layers - 1):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"MLP Direction Sampler initialized with {config.num_hidden_layers} hidden layers")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, antenna_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            antenna_embedding: Base station's antenna embedding parameter C [batch_size, 128]
        
        Returns:
            Raw MLP output [batch_size, A×B] before thresholding
        """
        return self.network(antenna_embedding)
    
    def get_direction_indicators(self, antenna_embedding: torch.Tensor, 
                                azimuth_divisions: int, elevation_divisions: int) -> torch.Tensor:
        """
        Get binary direction indicators for ray tracing.
        
        Args:
            antenna_embedding: Base station's antenna embedding parameter C
            azimuth_divisions: Number of azimuth divisions A
            elevation_divisions: Number of elevation divisions B
        
        Returns:
            Binary indicator matrix M_ij [batch_size, A, B] where each element M_ij ∈ {0, 1}
        """
        # Forward pass through MLP
        raw_output = self.forward(antenna_embedding)
        
        # Reshape to [batch_size, A, B]
        batch_size = raw_output.shape[0]
        output_matrix = raw_output.view(batch_size, azimuth_divisions, elevation_divisions)
        
        # Apply sigmoid and threshold to get binary indicators
        sigmoid_output = torch.sigmoid(output_matrix)
        indicator_matrix = (sigmoid_output > self.config.threshold).int()
        
        return indicator_matrix
    
    def get_sampling_efficiency(self, antenna_embedding: torch.Tensor,
                               azimuth_divisions: int, elevation_divisions: int) -> float:
        """
        Calculate sampling efficiency (fraction of selected directions).
        
        Args:
            antenna_embedding: Base station's antenna embedding parameter C
            azimuth_divisions: Number of azimuth divisions A
            elevation_divisions: Number of elevation divisions B
        
        Returns:
            Sampling efficiency as a fraction between 0 and 1
        """
        indicators = self.get_direction_indicators(antenna_embedding, azimuth_divisions, elevation_divisions)
        
        # Calculate fraction of selected directions
        total_directions = azimuth_divisions * elevation_divisions
        selected_directions = torch.sum(indicators, dim=(1, 2))
        efficiency = selected_directions.float() / total_directions
        
        return efficiency.mean().item()
    
    def get_selected_directions(self, antenna_embedding: torch.Tensor,
                               azimuth_divisions: int, elevation_divisions: int) -> List[Tuple[int, int]]:
        """
        Get list of selected direction indices.
        
        Args:
            antenna_embedding: Base station's antenna embedding parameter C
            azimuth_divisions: Number of azimuth divisions A
            elevation_divisions: Number of elevation divisions B
        
        Returns:
            List of (phi_idx, theta_idx) tuples for selected directions
        """
        indicators = self.get_direction_indicators(antenna_embedding, azimuth_divisions, elevation_divisions)
        
        # For single batch, get the first (and only) sample
        if indicators.dim() == 3:
            indicators = indicators[0]  # Remove batch dimension
        
        selected_directions = []
        for phi in range(azimuth_divisions):
            for theta in range(elevation_divisions):
                if indicators[phi, theta] == 1:
                    selected_directions.append((phi, theta))
        
        return selected_directions
    
    def train_on_data(self, training_data: List[Tuple[torch.Tensor, torch.Tensor]],
                      validation_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                      num_epochs: int = 100,
                      learning_rate: float = 0.001,
                      batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Train the MLP on historical ray tracing data.
        
        Args:
            training_data: List of (antenna_embedding, optimal_directions) pairs
            validation_data: Optional validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
        
        Returns:
            Dictionary containing training and validation losses
        """
        self.train()
        
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
        
        # Training history
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(training_data), batch_size):
                batch_data = training_data[i:i + batch_size]
                
                # Prepare batch
                batch_embeddings = torch.stack([data[0] for data in batch_data])
                batch_targets = torch.stack([data[1] for data in batch_data])
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.forward(batch_embeddings)
                loss = criterion(outputs, batch_targets.float())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            if validation_data is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_batches = 0
                    
                    for i in range(0, len(validation_data), batch_size):
                        batch_data = validation_data[i:i + batch_size]
                        
                        batch_embeddings = torch.stack([data[0] for data in batch_data])
                        batch_targets = torch.stack([data[1] for data in batch_data])
                        
                        outputs = self.forward(batch_embeddings)
                        loss = criterion(outputs, batch_targets.float())
                        
                        val_loss += loss.item()
                        val_batches += 1
                    
                    avg_val_loss = val_loss / val_batches
                    val_losses.append(avg_val_loss)
                
                self.train()
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {filepath}")

def create_mlp_direction_sampler(azimuth_divisions: int = 36,
                                elevation_divisions: int = 18,
                                hidden_dim: int = 256,
                                num_hidden_layers: int = 2) -> MLPDirectionSampler:
    """
    Create an MLP direction sampler with default configuration.
    
    Args:
        azimuth_divisions: Number of azimuth divisions A
        elevation_divisions: Number of elevation divisions B
        hidden_dim: Hidden layer dimension
        num_hidden_layers: Number of hidden layers
    
    Returns:
        Configured MLPDirectionSampler instance
    """
    config = DirectionSamplingConfig(
        input_dim=128,  # Antenna embedding dimension
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        output_dim=azimuth_divisions * elevation_divisions
    )
    
    return MLPDirectionSampler(config)

def generate_training_data(num_samples: int = 1000,
                          input_dim: int = 128,
                          output_dim: int = 648,
                          sparsity: float = 0.3) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate synthetic training data for MLP training.
    
    Args:
        num_samples: Number of training samples
        input_dim: Input dimension (antenna embedding)
        output_dim: Output dimension (A×B directions)
        sparsity: Fraction of directions that should be selected (0-1)
    
    Returns:
        List of (antenna_embedding, optimal_directions) pairs
    """
    training_data = []
    
    for _ in range(num_samples):
        # Generate random antenna embedding
        antenna_embedding = torch.randn(input_dim)
        
        # Generate optimal direction indicators with specified sparsity
        optimal_directions = torch.zeros(output_dim)
        num_selected = int(output_dim * sparsity)
        selected_indices = torch.randperm(output_dim)[:num_selected]
        optimal_directions[selected_indices] = 1.0
        
        training_data.append((antenna_embedding, optimal_directions))
    
    return training_data

# Example usage
def example_mlp_training():
    """Example of training the MLP direction sampler."""
    # Create MLP sampler
    mlp_sampler = create_mlp_direction_sampler(
        azimuth_divisions=36,
        elevation_divisions=18,
        hidden_dim=256,
        num_hidden_layers=2
    )
    
    # Generate training data
    training_data = generate_training_data(
        num_samples=1000,
        input_dim=128,
        output_dim=648,
        sparsity=0.3
    )
    
    # Split into training and validation
    split_idx = int(0.8 * len(training_data))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    # Train the model
    training_history = mlp_sampler.train_on_data(
        training_data=train_data,
        validation_data=val_data,
        num_epochs=50,
        learning_rate=0.001,
        batch_size=32
    )
    
    # Test the trained model
    test_embedding = torch.randn(1, 128)
    indicators = mlp_sampler.get_direction_indicators(test_embedding, 36, 18)
    efficiency = mlp_sampler.get_sampling_efficiency(test_embedding, 36, 18)
    selected_dirs = mlp_sampler.get_selected_directions(test_embedding, 36, 18)
    
    logger.info(f"Training completed. Sampling efficiency: {efficiency:.3f}")
    logger.info(f"Selected {len(selected_dirs)} directions out of 648 total")
    
    return mlp_sampler, training_history
