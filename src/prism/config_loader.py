"""
Configuration Loader for Prism Training Interface

This module provides utilities to load and parse configuration files
for the TrainingInterface and ray_tracer integration.
"""

import yaml
import os
from typing import Dict, Any, Optional, Tuple
import torch
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader for Prism training system."""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from: {self.config_path}")
        return config
    
    def get_training_interface_config(self) -> Dict[str, Any]:
        """Get TrainingInterface configuration."""
        return self.config.get('training_interface', {})
    
    def get_ray_tracer_config(self) -> Dict[str, Any]:
        """Get ray_tracer configuration."""
        # Merge ray_tracing and ray_tracer_integration configs
        ray_tracing_config = self.config.get('ray_tracing', {})
        integration_config = self.config.get('ray_tracer_integration', {})
        
        # Integration config overrides ray_tracing config
        merged_config = {**ray_tracing_config, **integration_config}
        return merged_config
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})
    
    def get_neural_network_config(self) -> Dict[str, Any]:
        """Get neural network configuration."""
        return self.config.get('neural_networks', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.config.get('performance', {})
    
    def get_mixed_precision_config(self) -> Dict[str, Any]:
        """Get mixed precision configuration."""
        performance_config = self.get_performance_config()
        return performance_config.get('mixed_precision', {})
    
    def is_mixed_precision_enabled(self) -> bool:
        """Check if mixed precision is globally enabled."""
        mixed_precision_config = self.get_mixed_precision_config()
        return mixed_precision_config.get('enabled', False)
    
    def get_network_mixed_precision_config(self, network_name: str) -> bool:
        """Get mixed precision configuration for a specific network."""
        nn_config = self.get_neural_network_config()
        network_config = nn_config.get(network_name, {})
        
        # Check network-specific setting first, then fall back to global setting
        network_specific = network_config.get('use_mixed_precision', None)
        if network_specific is not None:
            return network_specific
        
        # Fall back to global mixed precision setting
        return self.is_mixed_precision_enabled()
    
    def get_ray_tracer_mixed_precision_config(self, ray_tracer_type: str = None) -> bool:
        """Get mixed precision configuration for ray tracers."""
        ray_config = self.get_ray_tracer_config()
        
        # Check ray tracer specific setting first
        if ray_tracer_type and ray_tracer_type in ray_config:
            tracer_config = ray_config.get(ray_tracer_type, {})
            tracer_specific = tracer_config.get('use_mixed_precision', None)
            if tracer_specific is not None:
                return tracer_specific
        
        # Check general ray tracing mixed precision setting
        general_setting = ray_config.get('use_mixed_precision', None)
        if general_setting is not None:
            return general_setting
        
        # Fall back to global mixed precision setting
        return self.is_mixed_precision_enabled()
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config.get('output', {})
    
    def create_training_interface_kwargs(self) -> Dict[str, Any]:
        """
        Create keyword arguments for TrainingInterface initialization.
        
        Returns:
            Dictionary of keyword arguments for TrainingInterface
        """
        ti_config = self.get_training_interface_config()
        
        kwargs = {}
        
        # Basic parameters
        if 'num_sampling_points' in ti_config:
            kwargs['num_sampling_points'] = ti_config['num_sampling_points']
        
        if 'subcarrier_sampling_ratio' in ti_config:
            kwargs['subcarrier_sampling_ratio'] = ti_config['subcarrier_sampling_ratio']
        
        if 'checkpoint_dir' in ti_config:
            kwargs['checkpoint_dir'] = ti_config['checkpoint_dir']
        
        # Scene bounds
        if 'scene_bounds' in ti_config:
            scene_bounds = ti_config['scene_bounds']
            if 'min' in scene_bounds and 'max' in scene_bounds:
                scene_min = torch.tensor(scene_bounds['min'])
                scene_max = torch.tensor(scene_bounds['max'])
                kwargs['scene_bounds'] = (scene_min, scene_max)
        
        return kwargs
    
    def create_ray_tracer_kwargs(self) -> Dict[str, Any]:
        """
        Create keyword arguments for CPURayTracer initialization.
        
        Returns:
            Dictionary of keyword arguments for CPURayTracer
        """
        rt_config = self.get_ray_tracer_config()
        perf_config = self.get_performance_config()
        
        kwargs = {}
        
        # Angular divisions
        if 'azimuth_divisions' in rt_config:
            kwargs['azimuth_divisions'] = rt_config['azimuth_divisions']
        
        if 'elevation_divisions' in rt_config:
            kwargs['elevation_divisions'] = rt_config['elevation_divisions']
        
        # Physical parameters
        if 'max_ray_length' in rt_config:
            kwargs['max_ray_length'] = rt_config['max_ray_length']
        
        if 'scene_size' in rt_config:
            kwargs['scene_size'] = rt_config['scene_size']
        
        # Performance parameters
        if 'device' in perf_config:
            kwargs['device'] = perf_config['device']
        
        # Signal processing parameters
        if 'signal_threshold' in rt_config:
            kwargs['signal_threshold'] = rt_config['signal_threshold']
        
        if 'enable_early_termination' in rt_config:
            kwargs['enable_early_termination'] = rt_config['enable_early_termination']
        
        return kwargs
    
    def get_curriculum_learning_phases(self) -> Optional[list]:
        """
        Get curriculum learning phase configurations.
        
        Returns:
            List of phase configurations or None if disabled
        """
        ti_config = self.get_training_interface_config()
        curriculum_config = ti_config.get('curriculum_learning', {})
        
        if not curriculum_config.get('enabled', False):
            return None
        
        return curriculum_config.get('phases', [])
    
    def get_checkpoint_config(self) -> Dict[str, Any]:
        """Get checkpoint configuration."""
        ti_config = self.get_training_interface_config()
        output_config = self.get_output_config()
        
        checkpoint_config = {}
        
        # From training_interface config
        if 'checkpoint_dir' in ti_config:
            checkpoint_config['checkpoint_dir'] = ti_config['checkpoint_dir']
        
        if 'auto_checkpoint' in ti_config:
            checkpoint_config['auto_checkpoint'] = ti_config['auto_checkpoint']
        
        if 'checkpoint_frequency' in ti_config:
            checkpoint_config['checkpoint_frequency'] = ti_config['checkpoint_frequency']
        
        # From output config
        if 'checkpoint_format' in output_config:
            checkpoint_config['checkpoint_format'] = output_config['checkpoint_format']
        
        if 'save_optimizer_state' in output_config:
            checkpoint_config['save_optimizer_state'] = output_config['save_optimizer_state']
        
        if 'save_training_history' in output_config:
            checkpoint_config['save_training_history'] = output_config['save_training_history']
        
        return checkpoint_config
    
    def get_csi_computation_config(self) -> Dict[str, Any]:
        """Get CSI computation configuration."""
        ti_config = self.get_training_interface_config()
        return ti_config.get('csi_computation', {})
    
    def validate_config(self) -> bool:
        """
        Validate configuration for consistency and completeness.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required sections
            required_sections = ['training_interface', 'ray_tracer_integration', 'neural_networks']
            for section in required_sections:
                if section not in self.config:
                    logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Check TrainingInterface config
            ti_config = self.get_training_interface_config()
            if not ti_config.get('enabled', False):
                logger.warning("TrainingInterface is disabled in configuration")
            
            # Check ray_tracer integration
            rt_config = self.get_ray_tracer_config()
            if not rt_config.get('enabled', False):
                logger.warning("Ray tracer integration is disabled in configuration")
            
            # Validate scene bounds
            if 'scene_bounds' in ti_config:
                scene_bounds = ti_config['scene_bounds']
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
            
            # Validate curriculum learning phases
            phases = self.get_curriculum_learning_phases()
            if phases:
                for i, phase in enumerate(phases):
                    required_phase_keys = ['phase', 'azimuth_divisions', 'elevation_divisions', 'top_k_directions']
                    for key in required_phase_keys:
                        if key not in phase:
                            logger.error(f"Missing key '{key}' in curriculum learning phase {i}")
                            return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_device(self) -> str:
        """Get the device configuration."""
        perf_config = self.get_performance_config()
        device = perf_config.get('device', 'cpu')
        
        # Validate device availability
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        return device
    
    def save_config(self, output_path: str):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save the configuration
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {output_path}")


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
    # Example of how to use the configuration loader
    config_loader = load_config("configs/ofdm-5g-sionna.yml")
    
    # Validate configuration
    if config_loader.validate_config():
        print("Configuration is valid")
        
        # Get TrainingInterface kwargs
        ti_kwargs = config_loader.create_training_interface_kwargs()
        print(f"TrainingInterface kwargs: {ti_kwargs}")
        
        # Get ray_tracer kwargs
        rt_kwargs = config_loader.create_ray_tracer_kwargs()
        print(f"Ray tracer kwargs: {rt_kwargs}")
        
        # Get curriculum learning phases
        phases = config_loader.get_curriculum_learning_phases()
        print(f"Curriculum learning phases: {phases}")
    else:
        print("Configuration validation failed")
