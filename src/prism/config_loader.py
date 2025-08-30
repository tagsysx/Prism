"""
Configuration Loader for Prism Training Interface

This module provides utilities to load and parse configuration files
for the Prism ray tracing system with the new cleaned configuration structure.
"""

import yaml
import os
import re
from typing import Dict, Any, Optional, Tuple
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader for Prism training system with updated structure."""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file and process template variables."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Process template variables
        config = self._process_template_variables(config)
        
        logger.info(f"Configuration loaded from: {self.config_path}")
        return config
    
    def _process_template_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process template variables in configuration.
        
        Supports variables like:
        - {{base_dir}} - base results directory
        - {{training_dir}} - training output directory
        - {{testing_dir}} - testing output directory
        - {{checkpoint_dir}} - checkpoint directory
        - {{tensorboard_dir}} - tensorboard directory
        - {{models_dir}} - models directory
        - {{logs_dir}} - logs directory
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Configuration with template variables resolved
        """
        # Define template variables
        template_vars = self._get_template_variables(config)
        
        # Recursively process all string values in config
        processed_config = self._replace_variables_recursive(config, template_vars)
        
        # Add backward compatibility: add output_dir to training and testing sections
        processed_config = self._add_backward_compatibility(processed_config, template_vars)
        
        logger.info(f"Template variables processed: {list(template_vars.keys())}")
        return processed_config
    
    def _add_backward_compatibility(self, config: Dict[str, Any], template_vars: Dict[str, str]) -> Dict[str, Any]:
        """Add backward compatibility fields that may be expected by existing code."""
        # Add output_dir to training section for backward compatibility
        if 'output' in config and 'training' in config['output']:
            if 'output_dir' not in config['output']['training']:
                config['output']['training']['output_dir'] = template_vars['training_dir']
        
        # Add output_dir to testing section for backward compatibility
        if 'output' in config and 'testing' in config['output']:
            if 'output_dir' not in config['output']['testing']:
                config['output']['testing']['output_dir'] = template_vars['testing_dir']
        
        return config
    
    def _get_template_variables(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Get template variables and their values."""
        template_vars = {}
        
        # Get base directory (default to 'results')
        base_dir = config.get('output', {}).get('base_dir', 'results')
        template_vars['base_dir'] = base_dir
        
        # Define training and testing output directories
        template_vars['training_dir'] = f"{base_dir}/training"
        template_vars['testing_dir'] = f"{base_dir}/testing"
        
        # Define common subdirectories for training
        template_vars['checkpoint_dir'] = f"{template_vars['training_dir']}/checkpoints"
        template_vars['tensorboard_dir'] = f"{template_vars['training_dir']}/tensorboard"
        template_vars['models_dir'] = f"{template_vars['training_dir']}/models"
        template_vars['logs_dir'] = f"{template_vars['training_dir']}/logs"
        
        return template_vars
    
    def _replace_variables_recursive(self, obj: Any, template_vars: Dict[str, str]) -> Any:
        """Recursively replace template variables in configuration object."""
        if isinstance(obj, dict):
            return {key: self._replace_variables_recursive(value, template_vars) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_variables_recursive(item, template_vars) 
                   for item in obj]
        elif isinstance(obj, str):
            return self._replace_variables_in_string(obj, template_vars)
        else:
            return obj
    
    def _replace_variables_in_string(self, text: str, template_vars: Dict[str, str]) -> str:
        """Replace template variables in a string."""
        # Pattern to match {{variable_name}}
        pattern = r'\{\{(\w+)\}\}'
        
        def replace_var(match):
            var_name = match.group(1)
            if var_name in template_vars:
                return template_vars[var_name]
            else:
                logger.warning(f"Unknown template variable: {var_name}")
                return match.group(0)  # Return original if not found
        
        return re.sub(pattern, replace_var, text)
    
    # ==================== Main Configuration Sections ====================
    
    def get_neural_networks_config(self) -> Dict[str, Any]:
        """Get neural networks configuration."""
        return self.config.get('neural_networks', {})
    
    def get_base_station_config(self) -> Dict[str, Any]:
        """Get base station configuration."""
        return self.config.get('base_station', {})
    
    def get_user_equipment_config(self) -> Dict[str, Any]:
        """Get user equipment configuration."""
        return self.config.get('user_equipment', {})
    
    def get_ray_tracing_config(self) -> Dict[str, Any]:
        """Get ray tracing configuration (unified section)."""
        return self.config.get('ray_tracing', {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration (includes device, performance, etc.)."""
        return self.config.get('system', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})
    
    def get_testing_config(self) -> Dict[str, Any]:
        """Get testing configuration."""
        return self.config.get('testing', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config.get('output', {})
    
    # ==================== Specific Configuration Getters ====================
    
    def get_angular_sampling_config(self) -> Dict[str, Any]:
        """Get angular sampling configuration."""
        ray_tracing_config = self.get_ray_tracing_config()
        return ray_tracing_config.get('angular_sampling', {})
    
    def get_radial_sampling_config(self) -> Dict[str, Any]:
        """Get radial sampling configuration."""
        ray_tracing_config = self.get_ray_tracing_config()
        return ray_tracing_config.get('radial_sampling', {})
    
    def get_subcarrier_sampling_config(self) -> Dict[str, Any]:
        """Get subcarrier sampling configuration."""
        ray_tracing_config = self.get_ray_tracing_config()
        return ray_tracing_config.get('subcarrier_sampling', {})
    
    def get_scene_bounds_config(self) -> Dict[str, Any]:
        """Get scene bounds configuration."""
        ray_tracing_config = self.get_ray_tracing_config()
        return ray_tracing_config.get('scene_bounds', {})
    
    def get_mixed_precision_config(self) -> Dict[str, Any]:
        """Get mixed precision configuration."""
        system_config = self.get_system_config()
        return system_config.get('mixed_precision', {})
    
    def get_cuda_config(self) -> Dict[str, Any]:
        """Get CUDA-specific configuration."""
        system_config = self.get_system_config()
        return system_config.get('cuda', {})
    
    def get_cpu_config(self) -> Dict[str, Any]:
        """Get CPU-specific configuration."""
        system_config = self.get_system_config()
        return system_config.get('cpu', {})
    
    # ==================== Device and Performance ====================
    
    def get_device(self) -> str:
        """Get the primary device configuration."""
        system_config = self.get_system_config()
        device = system_config.get('device', 'cpu')
        
        # Validate device availability
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        return device
    
    def get_ray_tracing_mode(self) -> str:
        """Get ray tracing execution mode."""
        system_config = self.get_system_config()
        return system_config.get('ray_tracing_mode', 'hybrid')
    
    def is_mixed_precision_enabled(self) -> bool:
        """Check if mixed precision is globally enabled."""
        mixed_precision_config = self.get_mixed_precision_config()
        return mixed_precision_config.get('enabled', False)
    
    def should_fallback_to_cpu(self) -> bool:
        """Check if should fallback to CPU when CUDA fails."""
        system_config = self.get_system_config()
        return system_config.get('fallback_to_cpu', True)
    
    # ==================== Ray Tracer Configuration Creation ====================
    
    def create_ray_tracer_config(self) -> Dict[str, Any]:
        """
        Create ray tracing configuration dictionary for TrainingInterface.
        
        Returns:
            Dictionary containing ray tracing configuration
        """
        ray_tracing_config = self.get_ray_tracing_config()
        
        # Return the entire ray_tracing section as it's already well-structured
        return ray_tracing_config
    
    def create_system_config(self) -> Dict[str, Any]:
        """
        Create system configuration dictionary for TrainingInterface.
        
        Returns:
            Dictionary containing system configuration
        """
        system_config = self.get_system_config()
        
        # Return the entire system section
        return system_config
    
    def create_cpu_ray_tracer_kwargs(self) -> Dict[str, Any]:
        """
        Create keyword arguments for CPURayTracer initialization.
        
        Returns:
            Dictionary of keyword arguments for CPURayTracer
        """
        ray_tracing_config = self.get_ray_tracing_config()
        angular_sampling = self.get_angular_sampling_config()
        radial_sampling = self.get_radial_sampling_config()
        cpu_config = self.get_cpu_config()
        scene_bounds = self.get_scene_bounds_config()
        
        kwargs = {
            # CPU-specific parameters
            'max_workers': cpu_config.get('num_workers', 4),
            
            # Common parameters
            'azimuth_divisions': angular_sampling.get('azimuth_divisions', 18),
            'elevation_divisions': angular_sampling.get('elevation_divisions', 9),
            'max_ray_length': ray_tracing_config.get('max_ray_length', 200.0),
            'scene_bounds': scene_bounds,
            'signal_threshold': ray_tracing_config.get('signal_threshold', 1e-6),
            'enable_early_termination': ray_tracing_config.get('enable_early_termination', True),
            'top_k_directions': angular_sampling.get('top_k_directions', 32),
            'uniform_samples': radial_sampling.get('num_sampling_points', 64),
            'resampled_points': radial_sampling.get('resampled_points', 32)
        }
        
        return kwargs
    
    def create_cuda_ray_tracer_kwargs(self) -> Dict[str, Any]:
        """
        Create keyword arguments for CUDARayTracer initialization.
        
        Returns:
            Dictionary of keyword arguments for CUDARayTracer
        """
        ray_tracing_config = self.get_ray_tracing_config()
        angular_sampling = self.get_angular_sampling_config()
        radial_sampling = self.get_radial_sampling_config()
        mixed_precision = self.get_mixed_precision_config()
        scene_bounds = self.get_scene_bounds_config()
        
        kwargs = {
            # CUDA-specific parameters
            'use_mixed_precision': mixed_precision.get('enabled', True),
            
            # Common parameters
            'azimuth_divisions': angular_sampling.get('azimuth_divisions', 18),
            'elevation_divisions': angular_sampling.get('elevation_divisions', 9),
            'max_ray_length': ray_tracing_config.get('max_ray_length', 200.0),
            'scene_bounds': scene_bounds,
            'signal_threshold': ray_tracing_config.get('signal_threshold', 1e-6),
            'enable_early_termination': ray_tracing_config.get('enable_early_termination', True),
            'top_k_directions': angular_sampling.get('top_k_directions', 32),
            'uniform_samples': radial_sampling.get('num_sampling_points', 64),
            'resampled_points': radial_sampling.get('resampled_points', 32)
        }
        
        return kwargs
    
    # ==================== Neural Network Configuration ====================
    
    def get_prism_network_config(self) -> Dict[str, Any]:
        """Get PrismNetwork configuration."""
        nn_config = self.get_neural_networks_config()
        
        # Combine all network configurations for PrismNetwork
        prism_config = {
            'attenuation_network': nn_config.get('attenuation_network', {}),
            'attenuation_decoder': nn_config.get('attenuation_decoder', {}),
            'antenna_codebook': nn_config.get('antenna_codebook', {}),
            'antenna_network': nn_config.get('antenna_network', {}),
            'radiance_network': nn_config.get('radiance_network', {})
        }
        
        return prism_config
    
    def get_network_mixed_precision_config(self, network_name: str) -> bool:
        """Get mixed precision configuration for a specific network."""
        nn_config = self.get_neural_networks_config()
        network_config = nn_config.get(network_name, {})
        
        # Check network-specific setting first, then fall back to global setting
        network_specific = network_config.get('use_mixed_precision', None)
        if network_specific is not None:
            return network_specific
        
        # Fall back to global mixed precision setting
        return self.is_mixed_precision_enabled()
    
    # ==================== Scene and Bounds Utilities ====================
    
    def get_scene_bounds_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get scene bounds as PyTorch tensors.
        
        Returns:
            Tuple of (scene_min, scene_max) tensors
        """
        scene_bounds = self.get_scene_bounds_config()
        
        if 'min' not in scene_bounds or 'max' not in scene_bounds:
            # Default scene bounds
            scene_min = torch.tensor([-100.0, -100.0, 0.0], dtype=torch.float32)
            scene_max = torch.tensor([100.0, 100.0, 30.0], dtype=torch.float32)
        else:
            scene_min = torch.tensor(scene_bounds['min'], dtype=torch.float32)
            scene_max = torch.tensor(scene_bounds['max'], dtype=torch.float32)
        
        return scene_min, scene_max
    
    def calculate_max_ray_length(self, margin: float = 1.2) -> float:
        """
        Calculate maximum ray length from scene bounds.
        
        Args:
            margin: Safety margin multiplier
            
        Returns:
            Maximum ray length in meters
        """
        scene_bounds = self.get_scene_bounds_config()
        
        if 'min' not in scene_bounds or 'max' not in scene_bounds:
            # Use configured max_ray_length or default
            ray_tracing_config = self.get_ray_tracing_config()
            return ray_tracing_config.get('max_ray_length', 200.0)
        
        # Calculate diagonal distance of the scene
        scene_min = np.array(scene_bounds['min'])
        scene_max = np.array(scene_bounds['max'])
        scene_size = scene_max - scene_min
        diagonal_length = np.sqrt(np.sum(scene_size**2))
        
        return diagonal_length * margin
    
    # ==================== Training Configuration ====================
    
    def get_checkpoint_config(self) -> Dict[str, Any]:
        """Get checkpoint configuration."""
        training_config = self.get_training_config()
        output_config = self.get_output_config()
        
        checkpoint_config = {}
        
        # From training config
        if 'checkpoint_dir' in training_config:
            checkpoint_config['checkpoint_dir'] = training_config['checkpoint_dir']
        
        if 'auto_checkpoint' in training_config:
            checkpoint_config['auto_checkpoint'] = training_config['auto_checkpoint']
        
        if 'checkpoint_frequency' in training_config:
            checkpoint_config['checkpoint_frequency'] = training_config['checkpoint_frequency']
        
        # From output config
        if 'checkpoint_format' in output_config:
            checkpoint_config['checkpoint_format'] = output_config['checkpoint_format']
        
        if 'save_optimizer_state' in output_config:
            checkpoint_config['save_optimizer_state'] = output_config['save_optimizer_state']
        
        if 'save_training_history' in output_config:
            checkpoint_config['save_training_history'] = output_config['save_training_history']
        
        return checkpoint_config
    
    # ==================== Validation ====================
    
    def validate_config(self) -> bool:
        """
        Validate configuration for consistency and completeness.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required sections
            required_sections = ['neural_networks', 'ray_tracing', 'system']
            for section in required_sections:
                if section not in self.config:
                    logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Check neural networks config
            nn_config = self.get_neural_networks_config()
            if not nn_config.get('enabled', False):
                logger.warning("Neural networks are disabled in configuration")
            
            # Validate scene bounds
            scene_bounds = self.get_scene_bounds_config()
            if scene_bounds:
                if 'min' not in scene_bounds or 'max' not in scene_bounds:
                    logger.error("Scene bounds must specify both 'min' and 'max'")
                    return False
                
                scene_min = scene_bounds['min']
                scene_max = scene_bounds['max']
                if len(scene_min) != 3 or len(scene_max) != 3:
                    logger.error("Scene bounds must be 3D coordinates")
                    return False
                
                if any(scene_min[i] >= scene_max[i] for i in range(3)):
                    logger.error("Scene minimum bounds must be less than maximum bounds")
                    return False
            
            # Validate ray tracing mode
            ray_tracing_mode = self.get_ray_tracing_mode()
            valid_modes = ['cuda', 'cpu', 'hybrid']
            if ray_tracing_mode not in valid_modes:
                logger.error(f"Invalid ray_tracing_mode: {ray_tracing_mode}. Must be one of {valid_modes}")
                return False
            
            # Validate device configuration
            device = self.get_device()
            if device not in ['cuda', 'cpu']:
                logger.error(f"Invalid device: {device}. Must be 'cuda' or 'cpu'")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    # ==================== Utility Methods ====================
    
    def save_config(self, output_path: str):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save the configuration
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {output_path}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dictionary containing configuration summary
        """
        summary = {
            'config_file': self.config_path,
            'neural_networks_enabled': self.get_neural_networks_config().get('enabled', False),
            'device': self.get_device(),
            'ray_tracing_mode': self.get_ray_tracing_mode(),
            'mixed_precision_enabled': self.is_mixed_precision_enabled(),
            'scene_bounds': self.get_scene_bounds_config(),
            'angular_sampling': self.get_angular_sampling_config(),
            'radial_sampling': self.get_radial_sampling_config(),
            'subcarrier_sampling': self.get_subcarrier_sampling_config()
        }
        
        return summary


def load_config(config_path: str) -> ConfigLoader:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path)


# Example usage
if __name__ == "__main__":
    # Example of how to use the updated configuration loader
    config_loader = load_config("configs/ofdm-5g-sionna-clean.yml")
    
    # Validate configuration
    if config_loader.validate_config():
        print("Configuration is valid")
        
        # Get configuration summary
        summary = config_loader.get_config_summary()
        print(f"Configuration summary: {summary}")
        
        # Get ray tracer configurations
        ray_tracing_config = config_loader.create_ray_tracer_config()
        system_config = config_loader.create_system_config()
        
        print(f"Ray tracing config keys: {list(ray_tracing_config.keys())}")
        print(f"System config keys: {list(system_config.keys())}")
        
        # Get ray tracer kwargs
        cpu_kwargs = config_loader.create_cpu_ray_tracer_kwargs()
        cuda_kwargs = config_loader.create_cuda_ray_tracer_kwargs()
        
        print(f"CPU ray tracer kwargs: {list(cpu_kwargs.keys())}")
        print(f"CUDA ray tracer kwargs: {list(cuda_kwargs.keys())}")
        
    else:
        print("Configuration validation failed")