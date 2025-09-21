"""
Modern Configuration Loader for Prism Neural Ray Tracing System

Completely redesigned configuration loader that matches the current simplified
configuration structure without legacy compatibility concerns.
"""

import yaml
import os
import re
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field
import torch

# Import network configuration classes
from .networks.attenuation_network import AttenuationNetworkConfig
from .networks.radiance_network import RadianceNetworkConfig
from .networks.antenna_codebook import AntennaEmbeddingCodebookConfig
from .networks.prism_network import PrismNetworkConfig

logger = logging.getLogger(__name__)


@dataclass
class CSILossConfig:
    """CSI Loss configuration."""
    enabled: bool = True
    phase_weight: float = 1.0
    magnitude_weight: float = 1.0
    normalize_weights: bool = True


@dataclass
class PDPLossConfig:
    """PDP Loss configuration."""
    enabled: bool = True
    type: str = 'delay'
    fft_size: int = 2046
    normalize_pdp: bool = True
    mse_weight: float = 0.7
    delay_weight: float = 0.3


@dataclass
class SpatialSpectrumLossConfig:
    """Spatial Spectrum Loss configuration."""
    enabled: bool = True
    algorithm: str = 'bartlett'
    fusion_method: str = 'average'
    loss_type: str = 'ssim'
    orientation: str = 'bs'  # 'bs' for BS antenna array, 'ue' for UE antenna array, 'orientation' for orientation-based array
    theta_range: List[float] = field(default_factory=lambda: [0, 5.0, 90.0])
    phi_range: List[float] = field(default_factory=lambda: [0.0, 10.0, 360.0])
    ssim_window_size: int = 11
    ssim_k1: float = 0.01
    ssim_k2: float = 0.03


@dataclass
class LossConfig:
    """Overall loss configuration."""
    csi_weight: float = 0.7
    pdp_weight: float = 300.0
    spatial_spectrum_weight: float = 50.0
    regularization_weight: float = 0.01
    csi_loss: CSILossConfig = field(default_factory=CSILossConfig)
    pdp_loss: PDPLossConfig = field(default_factory=PDPLossConfig)
    spatial_spectrum_loss: SpatialSpectrumLossConfig = field(default_factory=SpatialSpectrumLossConfig)


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 6
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    auto_checkpoint: bool = True
    checkpoint_frequency: int = 5
    epoch_save_interval: int = 1
    scaling_factor: float = 1e-6
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class PhaseCalibrationConfig:
    """Phase calibration configuration."""
    enabled: bool = True
    reference_subcarrier_index: int = 0


@dataclass
class DataConfig:
    """Data configuration."""
    enabled: bool = True
    dataset_path: str = "data/sionna"
    random_seed: int = 42
    train_ratio: float = 0.8
    test_ratio: float = 0.2
    sampling_ratio: float = 0.5
    sampling_method: str = 'uniform'
    antenna_consistent: bool = True
    phase_calibration: PhaseCalibrationConfig = field(default_factory=PhaseCalibrationConfig)


@dataclass
class UserEquipmentConfig:
    """User Equipment configuration."""
    num_ue_antennas: int = 1
    ue_antenna_count: int = 1  # Number of UE antennas to use (1=use antenna 0, 2=use antennas 0,1, 3=use antennas 0,1,2, etc.)


@dataclass
class OutputConfig:
    """Output configuration."""
    base_dir: str = "results/sionna"
    format: str = 'hdf5'
    compression_level: int = 6
    save_results: bool = True
    save_training_outputs: bool = True
    save_ray_tracer_results: bool = True
    save_csi_predictions: bool = True
    checkpoint_format: str = 'pytorch'
    save_optimizer_state: bool = True
    save_training_history: bool = True


class ModernConfigLoader:
    """
    Modern configuration loader for Prism system.
    
    This loader is designed specifically for the current simplified configuration
    structure and does not maintain backward compatibility with legacy configurations.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Load raw configuration
        self._raw_config = self._load_yaml()
        
        # Process template variables
        self._processed_config = self._process_templates(self._raw_config)
        
        # Parse into structured configurations
        self._parse_configurations()
        
        # Validate configuration
        self._validate()
        
        logger.info(f"Configuration loaded successfully from: {self.config_path}")
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise IOError(f"Failed to read configuration file: {e}")
    
    def _process_templates(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process template variables in configuration."""
        # Get base directory for template processing
        base_dir = config.get('output', {}).get('base_dir', 'results/sionna')
        
        # Define template variables
        template_vars = {
            'base_dir': base_dir,
            'training_dir': f"{base_dir}/training",
            'testing_dir': f"{base_dir}/testing"
        }
        
        # Recursively replace template variables
        return self._replace_templates_recursive(config, template_vars)
    
    def _replace_templates_recursive(self, obj: Any, template_vars: Dict[str, str]) -> Any:
        """Recursively replace template variables in configuration."""
        if isinstance(obj, dict):
            return {key: self._replace_templates_recursive(value, template_vars) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_templates_recursive(item, template_vars) 
                   for item in obj]
        elif isinstance(obj, str):
            # Replace {{variable}} patterns
            pattern = r'\{\{(\w+)\}\}'
            def replace_var(match):
                var_name = match.group(1)
                return template_vars.get(var_name, match.group(0))
            return re.sub(pattern, replace_var, obj)
        else:
            return obj
    
    def _parse_configurations(self) -> None:
        """Parse configuration into structured dataclasses."""
        # Neural network configurations
        nn_config = self._processed_config.get('neural_networks', {})
        if not nn_config:
            raise ValueError("❌ Missing required 'neural_networks' configuration section")
        
        # Validate required neural network sections
        required_nn_sections = ['prism_network', 'attenuation_network', 'radiance_network', 'antenna_codebook']
        for section in required_nn_sections:
            if section not in nn_config:
                raise ValueError(f"❌ Missing required neural network section: '{section}'")
        
        # Calculate derived parameters from base configuration
        base_station = self._processed_config.get('base_station', {})
        user_equipment = self._processed_config.get('user_equipment', {})
        
        self.num_bs_antennas = base_station.get('num_antennas', 64)
        self.base_subcarriers = base_station.get('ofdm', {}).get('num_subcarriers', 64)
        self.ue_antenna_count = user_equipment.get('ue_antenna_count', 1)
        self.total_subcarriers = self.base_subcarriers * self.ue_antenna_count
        
        # Parse network configurations with derived parameters
        # 1. PrismNetwork - remove derived parameters that should come from base config
        prism_config = nn_config['prism_network'].copy()
        # Remove parameters that should be derived from base_station
        prism_config.pop('num_bs_antennas', None)
        prism_config.pop('num_subcarriers', None)
        self.prism_network = PrismNetworkConfig(**prism_config)
        
        # 2. AttenuationNetwork - no derived parameters needed
        self.attenuation_network = AttenuationNetworkConfig(**nn_config['attenuation_network'])
        
        # 3. RadianceNetwork - no derived parameters needed (output_dim is fixed)
        self.radiance_network = RadianceNetworkConfig(**nn_config['radiance_network'])
        
        # 4. AntennaCodebook - needs num_bs_antennas
        antenna_config = nn_config['antenna_codebook'].copy()
        antenna_config['num_bs_antennas'] = self.num_bs_antennas
        self.antenna_codebook = AntennaEmbeddingCodebookConfig(**antenna_config)
        
        # 5. FrequencyCodebook - store config with derived num_subcarriers
        freq_config = nn_config.get('frequency_codebook', {}).copy()
        freq_config['num_subcarriers'] = self.total_subcarriers
        self.frequency_codebook_config = freq_config
        
        # 6. CSINetwork - store config with derived max_antennas and max_subcarriers
        csi_config = nn_config.get('csi_network', {}).copy()
        csi_config['max_antennas'] = self.num_bs_antennas
        csi_config['max_subcarriers'] = self.total_subcarriers
        self.csi_network_config = csi_config
        
        # Training configuration
        training_config = self._processed_config.get('training', {})
        if not training_config:
            raise ValueError("❌ Missing required 'training' configuration section")
        
        loss_config = training_config.get('loss', {})
        if not loss_config:
            raise ValueError("❌ Missing required 'loss' configuration in training section")
        
        # Parse loss configurations
        csi_loss = self._parse_dataclass(
            CSILossConfig, loss_config.get('csi_loss', {})
        )
        
        pdp_loss = self._parse_dataclass(
            PDPLossConfig, loss_config.get('pdp_loss', {})
        )
        
        spatial_spectrum_loss = self._parse_dataclass(
            SpatialSpectrumLossConfig, loss_config.get('spatial_spectrum_loss', {})
        )
        
        loss = self._parse_dataclass(
            LossConfig, loss_config, {
                'csi_loss': csi_loss,
                'pdp_loss': pdp_loss,
                'spatial_spectrum_loss': spatial_spectrum_loss
            }
        )
        
        self.training = self._parse_dataclass(
            TrainingConfig, training_config, {'loss': loss}
        )
        
        # Data configuration - support both 'data' and 'input' keys
        data_config = self._processed_config.get('data', self._processed_config.get('input', {}))
        if not data_config:
            raise ValueError("❌ Missing required 'data' or 'input' configuration section")
        
        # Validate required data parameters
        required_data_params = ['dataset_path', 'train_ratio']
        for param in required_data_params:
            if param not in data_config:
                raise ValueError(f"❌ Missing required data parameter: '{param}'")
        
        subcarrier_config = data_config.get('subcarrier_sampling', {})
        phase_calibration_config = data_config.get('phase_calibration', {})
        
        # Parse phase calibration config
        phase_calibration = self._parse_dataclass(
            PhaseCalibrationConfig, phase_calibration_config
        )
        
        # Get user_equipment configuration
        user_equipment_config = self._processed_config.get('user_equipment', {})
        
        # Parse data config with phase calibration
        self.data = self._parse_dataclass(
            DataConfig, {**data_config, **subcarrier_config}, 
            {'phase_calibration': phase_calibration}
        )
        
        # User Equipment configuration
        self.user_equipment = self._parse_dataclass(
            UserEquipmentConfig, user_equipment_config
        )
        
        # Output configuration
        self.output = self._parse_dataclass(
            OutputConfig, self._processed_config.get('output', {})
        )
    
    def _parse_dataclass(self, dataclass_type, config_dict: Dict[str, Any], 
                        overrides: Optional[Dict[str, Any]] = None) -> Any:
        """Parse configuration dictionary into dataclass."""
        # Get dataclass fields
        field_names = {f.name for f in dataclass_type.__dataclass_fields__.values()}
        
        # Filter config to only include valid fields
        filtered_config = {k: v for k, v in config_dict.items() if k in field_names}
        
        # Apply overrides
        if overrides:
            filtered_config.update(overrides)
        
        try:
            return dataclass_type(**filtered_config)
        except TypeError as e:
            logger.error(f"Failed to create {dataclass_type.__name__}: {e}")
            logger.error(f"Available fields: {field_names}")
            logger.error(f"Provided config: {filtered_config}")
            raise
    
    def _validate(self) -> None:
        """Validate configuration consistency."""
        errors = []
        
        # Validate neural network dimensions
        if self.prism_network.feature_dim != self.attenuation_network.feature_dim:
            errors.append(
                f"Feature dimension mismatch: prism_network.feature_dim({self.prism_network.feature_dim}) "
                f"!= attenuation_network.feature_dim({self.attenuation_network.feature_dim})"
            )
        
        if self.prism_network.antenna_embedding_dim != self.antenna_codebook.embedding_dim:
            errors.append(
                f"Antenna embedding dimension mismatch: "
                f"prism_network.antenna_embedding_dim({self.prism_network.antenna_embedding_dim}) "
                f"!= antenna_codebook.embedding_dim({self.antenna_codebook.embedding_dim})"
            )
        
        # Validate BS antenna count consistency (using derived parameters)
        if self.num_bs_antennas != self.antenna_codebook.num_bs_antennas:
            errors.append(
                f"BS antenna count mismatch: "
                f"base_station.num_antennas({self.num_bs_antennas}) "
                f"!= antenna_codebook.num_bs_antennas({self.antenna_codebook.num_bs_antennas})"
            )
        
        # Validate ray tracing dimensions
        total_directions = self.prism_network.azimuth_divisions * self.prism_network.elevation_divisions
        if total_directions <= 0:
            errors.append(
                f"Invalid ray directions: azimuth_divisions({self.prism_network.azimuth_divisions}) "
                f"* elevation_divisions({self.prism_network.elevation_divisions}) = {total_directions}"
            )
        
        # Validate R-dimensional consistency across networks
        # Note: frequency_network has been removed, validation simplified
        
        # RadianceNetwork output_dim is passed as num_subcarriers in PrismNetwork
        radiance_output_dim = getattr(self.radiance_network, 'output_dim', None)
        if radiance_output_dim and radiance_output_dim != self.attenuation_network.output_dim:
            errors.append(
                f"R-dimension mismatch: radiance_network.output_dim({radiance_output_dim}) "
                f"!= attenuation_network.output_dim({self.attenuation_network.output_dim})"
            )
        
        # Validate training configuration
        if self.training.batch_size <= 0:
            errors.append(f"Invalid batch size: {self.training.batch_size}")
        
        if self.training.learning_rate <= 0:
            errors.append(f"Invalid learning rate: {self.training.learning_rate}")
        
        # Report errors
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    # ==================== Configuration Access Methods ====================
    
    def get_device(self) -> torch.device:
        """Get the computation device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            logger.warning("CUDA not available, using CPU")
            return torch.device('cpu')
    
    def get_prism_network_kwargs(self) -> Dict[str, Any]:
        """Get PrismNetwork initialization arguments."""
        # Get CSI network configuration from neural_networks section
        neural_networks_config = self._processed_config.get('neural_networks', {})
        if not neural_networks_config:
            raise ValueError("❌ Missing required 'neural_networks' configuration section")
        
        csi_network_config = neural_networks_config.get('csi_network', {})
        if not csi_network_config:
            raise ValueError("❌ Missing required 'csi_network' configuration in neural_networks section")
        
        # Validate required CSI network parameters
        required_csi_params = ['d_model', 'n_layers', 'n_heads', 'd_ff', 'dropout_rate']
        for param in required_csi_params:
            if param not in csi_network_config:
                raise ValueError(f"❌ Missing required CSI network parameter: '{param}'")
        
        return {
            'num_subcarriers': self.base_subcarriers,  # 每个UE天线的子载波数
            'num_bs_antennas': self.num_bs_antennas,     # 从基础配置获取
            'feature_dim': self.prism_network.feature_dim,
            'antenna_embedding_dim': self.prism_network.antenna_embedding_dim,
            'azimuth_divisions': self.prism_network.azimuth_divisions,
            'elevation_divisions': self.prism_network.elevation_divisions,
            'max_ray_length': self.prism_network.max_ray_length,
            'num_sampling_points': self.prism_network.num_sampling_points,
            'use_ipe_encoding': self.prism_network.use_ipe_encoding,
            'use_mixed_precision': self.prism_network.use_mixed_precision,
            'attenuation_network_config': self.attenuation_network.__dict__,
            'radiance_network_config': self.radiance_network.__dict__,
            'antenna_codebook_config': self.antenna_codebook.__dict__,
            'frequency_codebook_config': self.frequency_codebook_config,
            # Add CSI network configuration
            'use_csi_network': True,  # Always enable CSI network when config exists
            'csi_network_config': csi_network_config,
        }
    
    def get_training_kwargs(self) -> Dict[str, Any]:
        """Get training configuration arguments."""
        return {
            'learning_rate': self.training.learning_rate,
            'weight_decay': self.training.weight_decay,
            'num_epochs': self.training.num_epochs,
            'batch_size': self.training.batch_size,
            'gradient_accumulation_steps': self.training.gradient_accumulation_steps,
            'auto_checkpoint': self.training.auto_checkpoint,
            'checkpoint_frequency': self.training.checkpoint_frequency,
            'epoch_save_interval': self.training.epoch_save_interval,
        }
    
    def get_loss_functions_config(self) -> Dict[str, Any]:
        """Get loss functions configuration."""
        return {
            'csi_weight': self.training.loss.csi_weight,
            'pdp_weight': self.training.loss.pdp_weight,
            'spatial_spectrum_weight': self.training.loss.spatial_spectrum_weight,
            'regularization_weight': self.training.loss.regularization_weight,
            'csi_loss_config': self.training.loss.csi_loss,
            'pdp_loss_config': self.training.loss.pdp_loss,
            'spatial_spectrum_loss_config': self.training.loss.spatial_spectrum_loss,
        }
    
    def get_data_loader_config(self) -> Dict[str, Any]:
        """Get data loader configuration."""
        return {
            'dataset_path': self.data.dataset_path,
            'ue_antenna_count': self.user_equipment.ue_antenna_count,  # Get from user_equipment config
            'random_seed': self.data.random_seed,
            'train_ratio': self.data.train_ratio,
            'test_ratio': self.data.test_ratio,
            'sampling_ratio': self.data.sampling_ratio,
            'sampling_method': self.data.sampling_method,
            'antenna_consistent': self.data.antenna_consistent,
            'phase_calibration': {
                'enabled': self.data.phase_calibration.enabled,
                'reference_subcarrier_index': self.data.phase_calibration.reference_subcarrier_index,
            },
        }
    
    def get_output_paths(self) -> Dict[str, str]:
        """Get output directory paths."""
        base_dir = self.output.base_dir
        return {
            'base_dir': base_dir,
            'checkpoint_dir': f"{base_dir}/training/checkpoints",
            'tensorboard_dir': f"{base_dir}/training/tensorboard",
            'models_dir': f"{base_dir}/training/models",
            'log_dir': f"{base_dir}/training/logs",
            'log_file': f"{base_dir}/training/logs/training.log",
            'results_dir': f"{base_dir}/testing/results",
            'plots_dir': f"{base_dir}/testing/plots",
            'predictions_dir': f"{base_dir}/testing/predictions",
            'reports_dir': f"{base_dir}/testing/reports",
        }
    
    def create_loss_functions(self) -> Dict[str, Any]:
        """Create loss function instances."""
        from .loss.csi_loss import CSILoss
        from .loss.pdp_loss import PDPLoss
        from .loss.ss_loss import SSLoss
        
        loss_functions = {}
        
        # CSI Loss
        if self.training.loss.csi_loss.enabled:
            loss_functions['csi_loss'] = CSILoss(
                phase_weight=self.training.loss.csi_loss.phase_weight,
                magnitude_weight=self.training.loss.csi_loss.magnitude_weight,
                normalize_weights=self.training.loss.csi_loss.normalize_weights,
            )
        
        # PDP Loss
        if self.training.loss.pdp_loss.enabled:
            loss_functions['pdp_loss'] = PDPLoss(
                loss_type=self.training.loss.pdp_loss.type,
                fft_size=self.training.loss.pdp_loss.fft_size,
                normalize_pdp=self.training.loss.pdp_loss.normalize_pdp,
                mse_weight=self.training.loss.pdp_loss.mse_weight,
                delay_weight=self.training.loss.pdp_loss.delay_weight,
            )
        
        # Spatial Spectrum Loss (requires full config including base_station)
        if self.training.loss.spatial_spectrum_loss.enabled:
            # Check if required base_station config exists
            if 'base_station' in self._processed_config:
                try:
                    loss_functions['spatial_spectrum_loss'] = SSLoss(self._processed_config)
                except Exception as e:
                    logger.warning(f"Failed to create SSLoss: {e}. Skipping spatial spectrum loss.")
            else:
                logger.warning("SSLoss requires base_station configuration. Skipping spatial spectrum loss.")
        
        return loss_functions
    
    def ensure_output_directories(self) -> None:
        """Create output directories if they don't exist."""
        paths = self.get_output_paths()
        
        for path_name, path in paths.items():
            if path_name.endswith('_dir'):
                Path(path).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {path}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            'config_file': str(self.config_path),
            'prism_network': {
                'num_subcarriers': self.prism_network.num_subcarriers,
                'num_bs_antennas': self.prism_network.num_bs_antennas,
                'feature_dim': self.prism_network.feature_dim,
                'total_directions': self.prism_network.azimuth_divisions * self.prism_network.elevation_divisions,
                'max_ray_length': self.prism_network.max_ray_length,
                'num_sampling_points': self.prism_network.num_sampling_points,
            },
            'training': {
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'num_epochs': self.training.num_epochs,
                'gradient_accumulation_steps': self.training.gradient_accumulation_steps,
            },
            'loss_weights': {
                'csi_weight': self.training.loss.csi_weight,
                'pdp_weight': self.training.loss.pdp_weight,
                'spatial_spectrum_weight': self.training.loss.spatial_spectrum_weight,
            },
            'data': {
                'dataset_path': self.data.dataset_path,
                'train_ratio': self.data.train_ratio,
                'sampling_ratio': self.data.sampling_ratio,
            },
            'output': {
                'base_dir': self.output.base_dir,
                'format': self.output.format,
            }
        }
    
    def save_processed_config(self, output_path: Union[str, Path]) -> None:
        """Save the processed configuration to a file."""
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._processed_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Processed configuration saved to: {output_path}")


# Convenience function for loading configuration
def load_config(config_path: Union[str, Path]) -> ModernConfigLoader:
    """
    Load configuration using the modern config loader.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        ModernConfigLoader instance
    """
    return ModernConfigLoader(config_path)


# Compatibility alias for existing code
ConfigLoader = ModernConfigLoader


# Example usage
if __name__ == "__main__":
    import sys
    
    # Load configuration
    config_file = "configs/sionna.yml" if len(sys.argv) < 2 else sys.argv[1]
    
    try:
        config = load_config(config_file)
        
        print("✅ Configuration loaded successfully!")
        print("\n📊 Configuration Summary:")
        summary = config.get_config_summary()
        for section, details in summary.items():
            print(f"  {section}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    print(f"    {key}: {value}")
            else:
                print(f"    {details}")
        
        print(f"\n🎯 Total ray directions: {config.prism_network.azimuth_divisions * config.prism_network.elevation_divisions}")
        print(f"📡 Output dimension (low-rank): {config.attenuation_network.output_dim}")
        print(f"🔧 Device: {config.get_device()}")
        
        # Ensure output directories exist
        config.ensure_output_directories()
        print("📁 Output directories created/verified")
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        sys.exit(1)
