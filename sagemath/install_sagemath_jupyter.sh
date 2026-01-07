#!/bin/bash
#
# SageMath & Jupyter Installation Script for Ubuntu
# This script installs Miniconda, SageMath, and Jupyter
#
# Usage: bash install_sagemath_jupyter.sh
#

set -e  # Exit on error

echo "========================================================================"
echo "SageMath & Jupyter Installation Script for Ubuntu"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Ubuntu/Debian
if [ ! -f /etc/os-release ]; then
    echo -e "${RED}Error: Cannot determine OS version${NC}"
    exit 1
fi

source /etc/os-release
echo "Detected OS: $NAME $VERSION"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if conda is already installed
if command -v conda &> /dev/null; then
    print_warning "Conda is already installed at: $(which conda)"
    read -p "Do you want to skip Miniconda installation? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SKIP_MINICONDA=true
    else
        SKIP_MINICONDA=false
    fi
else
    SKIP_MINICONDA=false
fi

# Step 1: Install Miniconda
if [ "$SKIP_MINICONDA" = false ]; then
    echo ""
    echo "Step 1: Installing Miniconda..."
    echo "----------------------------------------"
    
    MINICONDA_DIR="$HOME/miniconda3"
    MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
    
    if [ -d "$MINICONDA_DIR" ]; then
        print_warning "Miniconda directory already exists: $MINICONDA_DIR"
        read -p "Remove and reinstall? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$MINICONDA_DIR"
        else
            print_status "Using existing Miniconda installation"
        fi
    fi
    
    if [ ! -d "$MINICONDA_DIR" ]; then
        # Download Miniconda
        echo "Downloading Miniconda..."
        wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_INSTALLER"
        
        # Install Miniconda
        echo "Installing Miniconda to $MINICONDA_DIR..."
        bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_DIR"
        
        # Clean up installer
        rm -f "$MINICONDA_INSTALLER"
        
        print_status "Miniconda installed successfully"
    fi
    
    # Initialize conda
    echo "Initializing conda for bash..."
    eval "$($MINICONDA_DIR/bin/conda shell.bash hook)"
    $MINICONDA_DIR/bin/conda init bash
    
    print_status "Conda initialized"
else
    print_status "Skipping Miniconda installation"
    # Make sure conda is in PATH
    if [ -f "$HOME/miniconda3/bin/conda" ]; then
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    fi
fi

# Source bashrc to get conda in PATH
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc" 2>/dev/null || true
fi

# Make sure conda command is available
if ! command -v conda &> /dev/null; then
    if [ -f "$HOME/miniconda3/bin/conda" ]; then
        export PATH="$HOME/miniconda3/bin:$PATH"
    else
        print_error "Conda installation failed or not found"
        exit 1
    fi
fi

# Step 2: Configure conda
echo ""
echo "Step 2: Configuring conda..."
echo "----------------------------------------"

# Accept Terms of Service if needed
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Add conda-forge channel
echo "Adding conda-forge channel..."
conda config --add channels conda-forge
conda config --set channel_priority strict

print_status "Conda configured with conda-forge channel"

# Step 3: Create sage environment
echo ""
echo "Step 3: Creating SageMath environment..."
echo "----------------------------------------"

# Check if sage environment already exists
if conda env list | grep -q "^sage "; then
    print_warning "Sage environment already exists"
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing sage environment..."
        conda env remove -n sage -y
    else
        print_status "Using existing sage environment"
        SKIP_SAGE_INSTALL=true
    fi
else
    SKIP_SAGE_INSTALL=false
fi

if [ "$SKIP_SAGE_INSTALL" = false ]; then
    echo "Creating conda environment 'sage' with SageMath..."
    echo "This may take 10-20 minutes..."
    
    conda create -n sage sage python=3.11 -y
    
    print_status "SageMath environment created successfully"
else
    print_status "Skipping sage environment creation"
fi

# Step 4: Verify installation
echo ""
echo "Step 4: Verifying installation..."
echo "----------------------------------------"

# Activate sage environment and check version
eval "$(conda shell.bash hook)"
conda activate sage

echo "Checking SageMath version..."
SAGE_VERSION=$(python -c "from sage.version import version; print(version)" 2>&1)
if [ $? -eq 0 ]; then
    print_status "SageMath version: $SAGE_VERSION"
else
    print_error "Failed to import SageMath"
    exit 1
fi

echo "Checking Jupyter version..."
JUPYTER_VERSION=$(jupyter --version 2>&1 | grep "jupyter core" | awk '{print $3}')
if [ $? -eq 0 ]; then
    print_status "Jupyter core version: $JUPYTER_VERSION"
else
    print_error "Jupyter not found"
    exit 1
fi

echo "Checking ipywidgets..."
python -c "import ipywidgets" 2>&1
if [ $? -eq 0 ]; then
    print_status "ipywidgets installed (required for interactive plots)"
else
    print_warning "ipywidgets not found, installing..."
    conda install ipywidgets -y
fi

# Step 5: Test SageMath
echo ""
echo "Step 5: Testing SageMath..."
echo "----------------------------------------"

TEST_OUTPUT=$(python -c "
from sage.all import *
x = var('x')
result = integrate(x**2, x)
print(f'Test: ∫x² dx = {result}')
print('Prime factorization of 2024:', factor(2024))
" 2>&1)

if [ $? -eq 0 ]; then
    echo "$TEST_OUTPUT"
    print_status "SageMath is working correctly!"
else
    print_error "SageMath test failed"
    echo "$TEST_OUTPUT"
    exit 1
fi

# Step 6: Create convenience scripts
echo ""
echo "Step 6: Creating convenience scripts..."
echo "----------------------------------------"

SAGE_BIN="$HOME/miniconda3/envs/sage/bin"

# Create alias suggestions file
cat > "$HOME/.sagemath_aliases" << 'EOF'
# SageMath & Jupyter aliases
# Add to your ~/.bashrc: source ~/.sagemath_aliases

alias sage-activate='conda activate sage'
alias sage-cmd='$HOME/miniconda3/envs/sage/bin/sage'
alias sage-python='$HOME/miniconda3/envs/sage/bin/python'
alias sage-jupyter='conda activate sage && jupyter notebook'
alias sage-lab='conda activate sage && jupyter lab'
EOF

print_status "Created alias file: ~/.sagemath_aliases"

# Step 7: Summary
echo ""
echo "========================================================================"
echo "Installation Complete!"
echo "========================================================================"
echo ""
echo "SageMath Information:"
echo "  - Version: $SAGE_VERSION"
echo "  - Environment: sage"
echo "  - Python: $(conda run -n sage python --version)"
echo "  - Location: $HOME/miniconda3/envs/sage"
echo ""
echo "Quick Start Commands:"
echo "  1. Activate environment:"
echo "     conda activate sage"
echo ""
echo "  2. Start SageMath (interactive):"
echo "     sage"
echo ""
echo "  3. Start Jupyter Notebook:"
echo "     jupyter notebook"
echo ""
echo "  4. Start Jupyter Lab:"
echo "     jupyter lab"
echo ""
echo "  5. Run Python with SageMath:"
echo "     python your_script.py"
echo ""
echo "Convenience Aliases (optional):"
echo "  Add to your ~/.bashrc:"
echo "    source ~/.sagemath_aliases"
echo ""
echo "  Then use:"
echo "    sage-activate    # Activate sage environment"
echo "    sage-cmd         # Run sage command"
echo "    sage-jupyter     # Start Jupyter notebook"
echo ""
echo "Next Steps:"
echo "  1. Close and reopen your terminal (or run: source ~/.bashrc)"
echo "  2. Activate sage: conda activate sage"
echo "  3. Try the tutorials in this directory!"
echo ""
echo "For VS Code integration:"
echo "  1. Install Jupyter extension (ms-toolsai.jupyter)"
echo "  2. Open a .ipynb file"
echo "  3. Select kernel: $SAGE_BIN/python"
echo ""
print_status "All done! Happy computing with SageMath! 🎉"
echo ""
