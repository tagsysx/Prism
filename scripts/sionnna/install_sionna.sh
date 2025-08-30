#!/bin/bash

# Sionna Installation Script for Prism Project
# This script helps install NVIDIA Sionna and its dependencies

echo "=== Sionna Installation Script for Prism ==="
echo "This script will install Sionna and required dependencies"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python version: $PYTHON_VERSION"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install pip first."
    exit 1
fi

echo "✓ pip3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (CPU version for compatibility)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install TensorFlow
echo "Installing TensorFlow..."
pip install tensorflow

# Install other dependencies
echo "Installing other dependencies..."
pip install numpy h5py matplotlib

# Install Sionna
echo "Installing Sionna..."
pip install sionna

# Verify installation
echo ""
echo "=== Verifying Installation ==="
python3 -c "
try:
    import sionna
    print(f'✓ Sionna {sionna.__version__} installed successfully')
except ImportError as e:
    print(f'❌ Sionna installation failed: {e}')
    exit(1)

try:
    import torch
    print(f'✓ PyTorch {torch.__version__} installed successfully')
except ImportError as e:
    print(f'❌ PyTorch installation failed: {e}')
    exit(1)

try:
    import tensorflow as tf
    print(f'✓ TensorFlow {tf.__version__} installed successfully')
except ImportError as e:
    print(f'❌ TensorFlow installation failed: {e}')
    exit(1)

print('\\n✅ All packages installed successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Installation Complete! ==="
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Test the setup: python scripts/test_sionna_simulation.py"
    echo "3. Run the simulation: python scripts/sionna_simulation.py"
    echo ""
    echo "Note: The virtual environment must be activated each time you work with this project."
else
    echo ""
    echo "❌ Installation verification failed. Please check the error messages above."
    exit 1
fi
