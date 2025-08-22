#!/usr/bin/env python3
"""
Complete Training Pipeline for Prism Network

This script runs the complete training pipeline:
1. Prepare data (split into train/test sets)
2. Train the Prism network
3. Test the trained model
4. Generate comprehensive results

Usage:
    python run_training_pipeline.py --data <sionna_data.h5> --config <config.yml>
"""

import os
import sys
import argparse
import logging
import subprocess
import time
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Complete training pipeline for Prism network"""
    
    def __init__(self, config_path: str, data_path: str, output_dir: str):
        """Initialize the training pipeline"""
        self.config_path = config_path
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup paths
        self.data_split_dir = self.output_dir / 'data_split'
        self.training_dir = self.output_dir / 'training'
        self.testing_dir = self.output_dir / 'testing'
        
        logger.info(f"Training pipeline initialized:")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Data: {data_path}")
        logger.info(f"  Output: {output_dir}")
    
    def _load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {self.config_path}")
        return config
    
    def prepare_data(self):
        """Step 1: Prepare and split the data"""
        logger.info("=" * 60)
        logger.info("STEP 1: Preparing and splitting data")
        logger.info("=" * 60)
        
        cmd = [
            sys.executable, 'scripts/simulation/data_prepare.py',
            '--data', self.data_path,
            '--output', str(self.data_split_dir),
            '--train-ratio', '0.8',
            '--seed', '42',
            '--verify'
        ]
        
        logger.info(f"Running data preparation: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Data preparation completed successfully")
            logger.info(f"Output: {result.stdout}")
            
            # Get the paths to split data
            self.train_data_path = self.data_split_dir / 'train_data.h5'
            self.test_data_path = self.data_split_dir / 'test_data.h5'
            
            if not self.train_data_path.exists() or not self.test_data_path.exists():
                raise FileNotFoundError("Split data files not found")
            
            logger.info(f"Training data: {self.train_data_path}")
            logger.info(f"Testing data: {self.test_data_path}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Data preparation failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            raise
    
    def train_model(self):
        """Step 2: Train the Prism network"""
        logger.info("=" * 60)
        logger.info("STEP 2: Training Prism network")
        logger.info("=" * 60)
        
        # Check if there's a latest checkpoint to resume from
        latest_checkpoint = self.training_dir / 'latest_checkpoint.pt'
        resume_arg = []
        if latest_checkpoint.exists():
            resume_arg = ['--resume', str(latest_checkpoint)]
            logger.info(f"Found checkpoint to resume from: {latest_checkpoint}")
        
        cmd = [
            sys.executable, 'scripts/simulation/train_prism.py',
            '--config', self.config_path,
            '--data', str(self.train_data_path),
            '--output', str(self.training_dir)
        ] + resume_arg
        
        logger.info(f"Running training: {' '.join(cmd)}")
        
        try:
            # Run training (this may take a long time)
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            training_time = time.time() - start_time
            
            logger.info("Training completed successfully")
            logger.info(f"Training time: {training_time/3600:.2f} hours")
            logger.info(f"Output: {result.stdout}")
            
            # Find the best model
            best_model_path = self.training_dir / 'best_model.pt'
            if best_model_path.exists():
                self.best_model_path = best_model_path
                logger.info(f"Best model saved: {self.best_model_path}")
            else:
                # Look for the latest checkpoint
                checkpoints = list(self.training_dir.glob('checkpoint_epoch_*.pt'))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
                    self.best_model_path = latest_checkpoint
                    logger.info(f"Using latest checkpoint: {self.best_model_path}")
                else:
                    raise FileNotFoundError("No trained model found")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
    
    def test_model(self):
        """Step 3: Test the trained model"""
        logger.info("=" * 60)
        logger.info("STEP 3: Testing trained model")
        logger.info("=" * 60)
        
        cmd = [
            sys.executable, 'scripts/simulation/test_prism.py',
            '--config', self.config_path,
            '--model', str(self.best_model_path),
            '--data', str(self.test_data_path),
            '--output', str(self.testing_dir)
        ]
        
        logger.info(f"Running testing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Testing completed successfully")
            logger.info(f"Output: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Testing failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Testing error: {e}")
            raise
    
    def generate_summary(self):
        """Step 4: Generate comprehensive summary"""
        logger.info("=" * 60)
        logger.info("STEP 4: Generating summary report")
        logger.info("=" * 60)
        
        summary_path = self.output_dir / 'training_pipeline_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("Prism Network Training Pipeline Summary\n")
            f.write("=====================================\n\n")
            
            f.write(f"Pipeline completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Config file: {self.config_path}\n")
            f.write(f"  Original data: {self.data_path}\n")
            f.write(f"  Output directory: {self.output_dir}\n\n")
            
            f.write("Data Preparation:\n")
            f.write(f"  Training data: {self.train_data_path}\n")
            f.write(f"  Testing data: {self.test_data_path}\n")
            f.write(f"  Split ratio: 80% training, 20% testing\n\n")
            
            f.write("Training:\n")
            f.write(f"  Training directory: {self.training_dir}\n")
            f.write(f"  Best model: {self.best_model_path}\n\n")
            
            f.write("Testing:\n")
            f.write(f"  Testing directory: {self.testing_dir}\n")
            f.write(f"  Test results: {self.testing_dir / 'test_results.json'}\n")
            f.write(f"  Predictions: {self.testing_dir / 'predictions.npz'}\n\n")
            
            f.write("Generated Files:\n")
            f.write("  - Training checkpoints and logs\n")
            f.write("  - Training curves and metrics\n")
            f.write("  - Test results and visualizations\n")
            f.write("  - Model predictions\n")
            f.write("  - Comprehensive performance analysis\n\n")
            
            f.write("Next Steps:\n")
            f.write("  1. Review training curves in tensorboard\n")
            f.write("  2. Analyze test results and visualizations\n")
            f.write("  3. Fine-tune hyperparameters if needed\n")
            f.write("  4. Deploy model for inference\n")
        
        logger.info(f"Summary report generated: {summary_path}")
    
    def run_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting Prism Network Training Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Prepare data
            self.prepare_data()
            
            # Step 2: Train model
            self.train_model()
            
            # Step 3: Test model
            self.test_model()
            
            # Step 4: Generate summary
            self.generate_summary()
            
            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"All results saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error("TRAINING PIPELINE FAILED!")
            logger.error("=" * 60)
            logger.error(f"Error: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run complete Prism training pipeline')
    parser.add_argument('--config', type=str, default='configs/ofdm-5g-sionna.yml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to Sionna simulation data HDF5 file')
    parser.add_argument('--output', type=str, default='results/complete_pipeline',
                       help='Output directory for all results')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)
    
    # Create and run pipeline
    pipeline = TrainingPipeline(args.config, args.data, args.output)
    pipeline.run_pipeline()

if __name__ == '__main__':
    main()
