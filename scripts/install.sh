#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Catch errors in pipelines

# Init Git modules, init uv
git submodule update --init --recursive
curl -LsSf https://astral.sh/uv/install.sh | sh

# Define the virtual environment directory
VENV_DIR=".venv"

# Remove any existing virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create a new virtual environment using uv
echo "Creating a new virtual environment..."
uv venv --python=3.12 "$VENV_DIR"

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Verify Python version
python -V

echo "CUDA version is"
nvidia-smi | grep "CUDA Version" | awk '{print $9}'

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
uv pip install "torch>=2.5" -i https://download.pytorch.org/whl/nightly/cu128

# Verify PyTorch installation
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}\nCUDA available: {torch.cuda.is_available()}\nCUDA version: {torch.version.cuda}')"

# Install additional dependencies
echo "Installing additional dependencies..."
uv pip install numpy "accelerate>=1.3.0" bitsandbytes==0.45.3 peft==0.14.0 trl==0.15.2

# Install Unsloth from GitHub
echo "Installing from uv.lock"

# Ensure setuptools is installed to prevent import errors
uv add setuptools

uv sync
uv sync --group dev
uv sync --group remote 

# Install Unsloth Zoo
# uv pip install --upgrade --force-reinstall unsloth==2025.3.10 unsloth_zoo
# uv pip install "unsloth[cu124-ampere-torch251] @ git+https://github.com/unslothai/unsloth.git" --no-build-isolation
# uv pip install --upgrade "xformers<=0.0.27"
# uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu124

echo "Running Unsloth"

uv pip show unsloth trl xformers unsloth_zoo

# Verify Unsloth installation
uv run python -c "from unsloth import FastLanguageModel; print('Unsloth import successful')"

uv run python -m devtools print-code >> .venv/lib/python3.12/site-packages/sitecustomize.py

echo "🚀 Installation complete"