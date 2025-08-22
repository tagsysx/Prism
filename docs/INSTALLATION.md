# Installation Guide

This guide will help you install Prism: Wideband RF Neural Radiance Fields for OFDM Communication.

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### **Installation Options**

#### **Option 1: Install from Source (Recommended for Development)**

```bash
# Clone the repository
git clone https://github.com/tagsysx/Prism.git
cd Prism

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

#### **Option 2: Install with Development Dependencies**

```bash
# Install with all development tools
pip install -e ".[dev]"
```

#### **Option 3: Install with Documentation Dependencies**

```bash
# Install with documentation building tools
pip install -e ".[docs]"
```

#### **Option 4: Install with Jupyter Support**

```bash
# Install with Jupyter notebook support
pip install -e ".[notebooks]"
```

## 🔧 **Detailed Installation Steps**

### **Step 1: System Requirements**

#### **Ubuntu/Debian**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git
sudo apt install build-essential python3-dev
```

#### **macOS**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python3 git
```

#### **Windows**
- Download and install Python from [python.org](https://python.org)
- Download and install Git from [git-scm.com](https://git-scm.com)

### **Step 2: Clone Repository**

```bash
git clone https://github.com/tagsysx/Prism.git
cd Prism
```

### **Step 3: Create Virtual Environment**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Unix/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### **Step 4: Install Dependencies**

```bash
# Upgrade pip
pip install --upgrade pip

# Install the package
pip install -e .

# Or install with specific extras
pip install -e ".[dev,docs,notebooks]"
```

## 📦 **Dependencies**

### **Core Dependencies**
- **PyTorch** (>=1.9.0): Deep learning framework
- **NumPy** (>=1.21.0): Numerical computing
- **SciPy** (>=1.7.0): Scientific computing
- **Matplotlib** (>=3.4.0): Plotting and visualization
- **PyYAML** (>=5.4.0): Configuration file parsing
- **tqdm** (>=4.62.0): Progress bars
- **TensorBoard** (>=2.7.0): Training visualization

### **Optional Dependencies**

#### **Development Tools**
- **pytest** (>=6.0): Testing framework
- **black** (>=21.0): Code formatting
- **flake8** (>=3.8): Code linting
- **mypy** (>=0.800): Type checking
- **pre-commit** (>=2.0): Git hooks

#### **Documentation**
- **Sphinx** (>=4.0): Documentation generator
- **sphinx-rtd-theme** (>=1.0): Read the Docs theme
- **myst-parser** (>=0.15): Markdown parsing

#### **Jupyter Support**
- **jupyter** (>=1.0): Jupyter notebook
- **ipykernel** (>=6.0): IPython kernel

## 🐳 **Docker Installation**

### **Build Docker Image**

```bash
# Build the Docker image
docker build -t prism-rf .

# Run the container
docker run -it --gpus all -v $(pwd):/workspace prism-rf
```

### **Docker Compose**

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down
```

## 🔍 **Verification**

### **Test Installation**

```bash
# Run basic tests
python -m pytest tests/ -v

# Run demo
python scripts/basic_usage.py

# Check package installation
python -c "import prism; print(prism.__version__)"
```

### **Check GPU Support (if applicable)**

```bash
# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Import Errors**
```bash
# Make sure you're in the correct directory
cd Prism

# Reinstall the package
pip uninstall prism-rf -y
pip install -e .
```

#### **CUDA Issues**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### **Permission Issues**
```bash
# Use user installation
pip install --user -e .

# Or fix permissions
sudo chown -R $USER:$USER .
```

#### **Memory Issues**
```bash
# Reduce batch size in configuration files
# Check configs/ofdm-wifi.yml and configs/ofdm-wideband.yml
```

### **Getting Help**

- Check the [Troubleshooting Guide](advanced/troubleshooting.md)
- Search [GitHub Issues](https://github.com/tagsysx/Prism/issues)
- Ask on [GitHub Discussions](https://github.com/tagsysx/Prism/discussions)

## 🔄 **Updating**

### **Update from Source**

```bash
# Pull latest changes
git pull origin main

# Reinstall package
pip install -e . --force-reinstall
```

### **Update Dependencies**

```bash
# Update all dependencies
pip install --upgrade -r requirements.txt

# Or update specific packages
pip install --upgrade torch torchvision
```

## 📚 **Next Steps**

After successful installation:

1. **Read the [Quick Start Guide](quickstart.md)**
2. **Explore [Configuration Options](user_guide/configuration.md)**
3. **Try the [Basic Examples](examples/basic_usage.md)**
4. **Check out [Training Guide](user_guide/training.md)**

## 🤝 **Contributing**

If you encounter installation issues or want to contribute:

1. Check existing [GitHub Issues](https://github.com/tagsysx/Prism/issues)
2. Create a new issue with detailed error information
3. Follow our [Contributing Guidelines](development/contributing.md)

---

*For more help, please refer to the [main documentation](README.md) or contact the maintainers.*
