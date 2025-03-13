#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Catch errors in pipelines

# Define the virtual environment directory
VENV_DIR=".venv"

# Remove any existing virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create a new virtual environment using uv
echo "Creating a new virtual environment..."
uv venv --python=3.10 "$VENV_DIR"

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Verify Python version
python -V

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}\nCUDA available: {torch.cuda.is_available()}\nCUDA version: {torch.version.cuda}')"

# Install additional dependencies
echo "Installing additional dependencies..."
uv pip install numpy accelerate==1.4.0 bitsandbytes==0.45.3 xformers==0.0.28.post3 peft==0.14.0 trl==0.15.2

# Install Unsloth from GitHub
echo "Installing Unsloth..."
uv pip install git+https://github.com/unslothai/unsloth.git

# Ensure setuptools is installed to prevent import errors
uv pip install setuptools

# Install Unsloth Zoo
uv pip install unsloth_zoo==2025.2.7

# Verify Unsloth installation
python -c "from unsloth import FastLanguageModel; print('Unsloth import successful!')"

echo "Installation complete!"

ENTRYPOINT ["/bin/bash"]