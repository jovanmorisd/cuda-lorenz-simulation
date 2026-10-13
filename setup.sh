#!/bin/bash

# Setup script for AI-HPC Lorenz Project
# Verifies dependencies and prepares the environment

set -e

echo "========================================="
echo "AI-HPC Lorenz Project - Setup Script"
echo "========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Python 3 found: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check required Python packages
echo ""
echo "Checking Python packages..."
REQUIRED_PACKAGES=("numpy" "torch" "matplotlib" "sklearn" "jupyter")

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $pkg is installed"
    else
        echo -e "${YELLOW}⚠${NC} $pkg not found. Install with: pip install $pkg"
    fi
done

# Check C++ compiler
echo ""
echo "Checking C++ compiler..."
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n 1)
    echo -e "${GREEN}✓${NC} g++ found: $GCC_VERSION"
else
    echo -e "${YELLOW}⚠${NC} g++ not found. Install with: sudo apt install build-essential"
fi

# Check OpenMP
echo ""
echo "Checking OpenMP..."
if g++ -fopenmp -dM -E - </dev/null | grep -q "_OPENMP"; then
    echo -e "${GREEN}✓${NC} OpenMP is available"
else
    echo -e "${YELLOW}⚠${NC} OpenMP not available. Install with: sudo apt install libomp-dev"
fi

# Check CUDA
echo ""
echo "Checking CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $(NF-1)}')
    echo -e "${GREEN}✓${NC} NVCC found: CUDA $CUDA_VERSION"
    
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓${NC} NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name --format=csv,noheader
    fi
else
    echo -e "${YELLOW}⚠${NC} CUDA (nvcc) not found. GPU support will be skipped."
    echo "   To enable: Download CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
fi

# Create directories
echo ""
echo "Creating project directories..."
mkdir -p data models plots report
echo -e "${GREEN}✓${NC} Directories created"

# Install Python dependencies
echo ""
echo "Would you like to install Python dependencies? (y/n)"
read -r response
if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo -e "${GREEN}✓${NC} Dependencies installed"
fi

# Build C++ components
echo ""
echo "Would you like to compile C++/CUDA components? (y/n)"
read -r response
if [[ "$response" == "y" || "$response" == "Y" ]]; then
    make clean
    make all
fi

echo ""
echo "========================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Review README.md for detailed documentation"
echo "2. Start the Jupyter notebook:"
echo "   jupyter notebook lorenz_hpc.ipynb"
echo "3. Run all cells to execute the complete pipeline"
echo ""
