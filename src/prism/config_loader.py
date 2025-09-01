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
        
        # Validate configuration consistency
        self._validate_config_consistency(config)
        
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
    
    def _validate_config_consistency(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration consistency and report any conflicts.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            SystemExit: If critical configuration conflicts are found
        """
        logger.info("🔍 Starting configuration consistency validation...")
        
        errors = []
        warnings = []
        
        try:
            # Extract configuration sections
            neural_networks = config.get('neural_networks', {})
            base_station = config.get('base_station', {})
            user_equipment = config.get('user_equipment', {})
            ray_tracing = config.get('ray_tracing', {})
            training = config.get('training', {})
            
            # 1. Validate angular sampling consistency
            self._validate_angular_sampling(ray_tracing, neural_networks, errors)
            
            # 2. Validate subcarrier consistency
            self._validate_subcarrier_consistency(base_station, neural_networks, errors)
            
            # 3. Validate antenna consistency
            self._validate_antenna_consistency(base_station, neural_networks, user_equipment, errors)
            
            # 4. Validate embedding dimension consistency
            self._validate_embedding_consistency(base_station, neural_networks, errors)
            
            # 5. Validate feature dimension consistency
            self._validate_feature_consistency(neural_networks, errors)
            
            # 6. Validate OFDM parameter consistency
            self._validate_ofdm_consistency(base_station, warnings)
            
            # 7. Validate antenna array consistency
            self._validate_antenna_array_consistency(base_station, warnings)
            
            # 8. Validate training parameter consistency
            self._validate_training_consistency(training, warnings)
            
        except Exception as e:
            errors.append(f"Configuration validation failed with exception: {e}")
        
        # Report results
        if warnings:
            logger.warning("⚠️  Configuration warnings found:")
            for warning in warnings:
                logger.warning(f"   - {warning}")
        
        if errors:
            logger.error("❌ Critical configuration errors found:")
            for error in errors:
                logger.error(f"   - {error}")
            logger.error("🛑 System will exit due to configuration errors")
            raise SystemExit(1)
        else:
            logger.info("✅ Configuration consistency validation passed")
    
    def _validate_angular_sampling(self, ray_tracing: Dict, neural_networks: Dict, errors: list) -> None:
        """Validate angular sampling consistency."""
        try:
            angular_sampling = ray_tracing.get('angular_sampling', {})
            antenna_network = neural_networks.get('antenna_network', {})
            
            azimuth_divisions = int(angular_sampling.get('azimuth_divisions', 0))
            elevation_divisions = int(angular_sampling.get('elevation_divisions', 0))
            total_directions = int(angular_sampling.get('total_directions', 0))
            antenna_output_dim = int(antenna_network.get('output_dim', 0))
            
            # Check azimuth × elevation = total_directions
            expected_total = azimuth_divisions * elevation_divisions
            if expected_total != total_directions:
                errors.append(
                    f"Angular sampling mismatch: azimuth_divisions({azimuth_divisions}) × "
                    f"elevation_divisions({elevation_divisions}) = {expected_total}, "
                    f"but total_directions = {total_directions}"
                )
            
            # Check antenna network output dimension matches total directions
            if antenna_output_dim != total_directions:
                errors.append(
                    f"Antenna network output dimension mismatch: "
                    f"antenna_network.output_dim({antenna_output_dim}) ≠ "
                    f"angular_sampling.total_directions({total_directions})"
                )
        except (ValueError, TypeError) as e:
            errors.append(f"Invalid angular sampling parameter types: {e}")
    
    def _validate_subcarrier_consistency(self, base_station: Dict, neural_networks: Dict, errors: list) -> None:
        """Validate subcarrier number consistency."""
        try:
            ofdm = base_station.get('ofdm', {})
            attenuation_decoder = neural_networks.get('attenuation_decoder', {})
            radiance_network = neural_networks.get('radiance_network', {})
            
            ofdm_subcarriers = int(ofdm.get('num_subcarriers', 0))
            decoder_output = int(attenuation_decoder.get('output_dim', 0))
            radiance_output = int(radiance_network.get('output_dim', 0))
            
            if not (ofdm_subcarriers == decoder_output == radiance_output):
                errors.append(
                    f"Subcarrier number mismatch: "
                    f"ofdm.num_subcarriers({ofdm_subcarriers}) ≠ "
                    f"attenuation_decoder.output_dim({decoder_output}) ≠ "
                    f"radiance_network.output_dim({radiance_output})"
                )
        except (ValueError, TypeError) as e:
            errors.append(f"Invalid subcarrier parameter types: {e}")
    
    def _validate_antenna_consistency(self, base_station: Dict, neural_networks: Dict, 
                                    user_equipment: Dict, errors: list) -> None:
        """Validate antenna number consistency."""
        try:
            # BS antenna consistency
            bs_antennas = int(base_station.get('num_antennas', 0))
            codebook_antennas = int(neural_networks.get('antenna_codebook', {}).get('num_antennas', 0))
            
            if bs_antennas != codebook_antennas:
                errors.append(
                    f"BS antenna number mismatch: "
                    f"base_station.num_antennas({bs_antennas}) ≠ "
                    f"antenna_codebook.num_antennas({codebook_antennas})"
                )
            
            # UE antenna consistency (neural networks should both use 1 for single antenna processing)
            decoder_ue = int(neural_networks.get('attenuation_decoder', {}).get('num_ue_antennas', 0))
            radiance_ue = int(neural_networks.get('radiance_network', {}).get('num_ue_antennas', 0))
            
            if decoder_ue != 1 or radiance_ue != 1:
                errors.append(
                    f"UE antenna configuration error: "
                    f"attenuation_decoder.num_ue_antennas({decoder_ue}) and "
                    f"radiance_network.num_ue_antennas({radiance_ue}) must both be 1 for single antenna processing"
                )
        except (ValueError, TypeError) as e:
            errors.append(f"Invalid antenna parameter types: {e}")
    
    def _validate_embedding_consistency(self, base_station: Dict, neural_networks: Dict, errors: list) -> None:
        """Validate embedding dimension consistency."""
        try:
            bs_embedding = int(base_station.get('antenna_embedding_dim', 0))
            codebook_embedding = int(neural_networks.get('antenna_codebook', {}).get('embedding_dim', 0))
            antenna_input = int(neural_networks.get('antenna_network', {}).get('input_dim', 0))
            radiance_embedding = int(neural_networks.get('radiance_network', {}).get('antenna_embedding_dim', 0))
            
            if not (bs_embedding == codebook_embedding == antenna_input == radiance_embedding):
                errors.append(
                    f"Antenna embedding dimension mismatch: "
                    f"base_station.antenna_embedding_dim({bs_embedding}) ≠ "
                    f"antenna_codebook.embedding_dim({codebook_embedding}) ≠ "
                    f"antenna_network.input_dim({antenna_input}) ≠ "
                    f"radiance_network.antenna_embedding_dim({radiance_embedding})"
                )
        except (ValueError, TypeError) as e:
            errors.append(f"Invalid embedding parameter types: {e}")
    
    def _validate_feature_consistency(self, neural_networks: Dict, errors: list) -> None:
        """Validate feature dimension consistency."""
        try:
            attenuation_feature = int(neural_networks.get('attenuation_network', {}).get('feature_dim', 0))
            decoder_input = int(neural_networks.get('attenuation_decoder', {}).get('input_dim', 0))
            radiance_feature = int(neural_networks.get('radiance_network', {}).get('spatial_feature_dim', 0))
            
            if not (attenuation_feature == decoder_input == radiance_feature):
                errors.append(
                    f"Feature dimension mismatch: "
                    f"attenuation_network.feature_dim({attenuation_feature}) ≠ "
                    f"attenuation_decoder.input_dim({decoder_input}) ≠ "
                    f"radiance_network.spatial_feature_dim({radiance_feature})"
                )
        except (ValueError, TypeError) as e:
            errors.append(f"Invalid feature parameter types: {e}")
    
    def _validate_ofdm_consistency(self, base_station: Dict, warnings: list) -> None:
        """Validate OFDM parameter consistency."""
        ofdm = base_station.get('ofdm', {})
        
        try:
            bandwidth = float(ofdm.get('bandwidth', 0))
            num_subcarriers = int(ofdm.get('num_subcarriers', 0))
            subcarrier_spacing = float(ofdm.get('subcarrier_spacing', 0))
            fft_size = int(ofdm.get('fft_size', 0))
            num_guard_carriers = int(ofdm.get('num_guard_carriers', 0))
        except (ValueError, TypeError):
            warnings.append("Invalid OFDM parameter types")
            return
        
        # Check subcarrier spacing
        if bandwidth > 0 and num_subcarriers > 0 and subcarrier_spacing > 0:
            expected_spacing = bandwidth / num_subcarriers
            spacing_error = abs(subcarrier_spacing - expected_spacing) / expected_spacing
            if spacing_error > 0.01:  # 1% tolerance
                warnings.append(
                    f"OFDM subcarrier spacing mismatch: "
                    f"configured({subcarrier_spacing:.1f} Hz) vs "
                    f"calculated({expected_spacing:.1f} Hz), "
                    f"error: {spacing_error*100:.2f}%"
                )
        
        # Check guard carriers
        if fft_size > 0 and num_subcarriers > 0:
            expected_guards = (fft_size - num_subcarriers) / 2
            if abs(num_guard_carriers - expected_guards) > 0.5:
                warnings.append(
                    f"OFDM guard carriers mismatch: "
                    f"configured({num_guard_carriers}) vs "
                    f"calculated({expected_guards})"
                )
    
    def _validate_antenna_array_consistency(self, base_station: Dict, warnings: list) -> None:
        """Validate antenna array configuration consistency."""
        num_antennas = base_station.get('num_antennas', 0)
        antenna_array = base_station.get('antenna_array', {})
        configuration = antenna_array.get('configuration', '')
        
        # Parse antenna array configuration (e.g., '8x8')
        if 'x' in configuration:
            try:
                parts = configuration.split('x')
                if len(parts) == 2:
                    rows, cols = int(parts[0]), int(parts[1])
                    expected_antennas = rows * cols
                    if expected_antennas != num_antennas:
                        warnings.append(
                            f"Antenna array configuration mismatch: "
                            f"configuration({configuration}) = {expected_antennas} antennas, "
                            f"but num_antennas = {num_antennas}"
                        )
            except (ValueError, IndexError):
                warnings.append(f"Invalid antenna array configuration format: {configuration}")
    
    def _validate_training_consistency(self, training: Dict, warnings: list) -> None:
        """Validate training parameter consistency."""
        try:
            batches_per_epoch = int(training.get('batches_per_epoch', 0))
            gradient_accumulation_steps = int(training.get('gradient_accumulation_steps', 1))
            
            effective_batch_size = batches_per_epoch * gradient_accumulation_steps
            
            # Check for reasonable effective batch size
            if effective_batch_size > 1000:
                warnings.append(
                    f"Very large effective batch size: "
                    f"batches_per_epoch({batches_per_epoch}) × "
                    f"gradient_accumulation_steps({gradient_accumulation_steps}) = "
                    f"{effective_batch_size}"
                )
            elif effective_batch_size < 4:
                warnings.append(
                    f"Very small effective batch size: "
                    f"batches_per_epoch({batches_per_epoch}) × "
                    f"gradient_accumulation_steps({gradient_accumulation_steps}) = "
                    f"{effective_batch_size}"
                )
        except (ValueError, TypeError):
            warnings.append("Invalid training parameter types")


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