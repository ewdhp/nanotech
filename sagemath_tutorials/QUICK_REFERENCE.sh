#!/bin/bash
#
# Quick Reference - SageMath & Jupyter Commands
# Usage: source this file or read for command reference
#

# Installation
echo "=== INSTALLATION ==="
echo "bash install_sagemath_jupyter.sh    # Run automated installation"
echo ""

# Environment Management
echo "=== ENVIRONMENT ==="
echo "conda activate sage                 # Activate SageMath environment"
echo "conda deactivate                    # Deactivate environment"
echo "conda info --envs                   # List all environments"
echo ""

# SageMath Usage
echo "=== SAGEMATH ==="
echo "sage                                # Start interactive SageMath"
echo "sage --version                      # Check SageMath version"
echo "sage --notebook=jupyter             # Start Jupyter with SageMath kernel"
echo ""

# Jupyter Usage
echo "=== JUPYTER ==="
echo "jupyter notebook                    # Start Jupyter Notebook (browser)"
echo "jupyter lab                         # Start Jupyter Lab (browser)"
echo "jupyter notebook --help             # Show help"
echo ""

# Python with SageMath
echo "=== PYTHON SCRIPTS ==="
echo "python script.py                    # Run Python script with SageMath"
echo "python sage_demo.py                 # Run basic demo"
echo "python sage_parameters_demo.py      # Create parameter plots"
echo ""

# VS Code
echo "=== VS CODE ==="
echo "code file.ipynb                     # Open notebook in VS Code"
echo "# Then select kernel: sage environment"
echo ""

# Example Code
echo "=== EXAMPLE CODE ==="
cat << 'EOF'

# In Python script or notebook:
from sage.all import *

# Symbolic math
x = var('x')
expr = x**2 + 3*x + 2
print(integrate(expr, x))

# Plotting
p = plot(sin(x), (x, -pi, pi))
p.save('plot.png')

# Interactive (in Jupyter)
from sage.repl.ipython_kernel.interact import interact

@interact
def f(a=(1, 10)):
    plot(a * sin(x), (x, -2*pi, 2*pi)).show()
EOF

echo ""
echo "=== TUTORIALS ==="
echo "test_vscode_interactive.ipynb       # Quick test notebook"
echo "sage_interactive_plots.ipynb        # Full interactive examples"
echo ""

echo "=== DOCUMENTATION ==="
echo "README.md                           # This directory overview"
echo "SAGEMATH_USAGE.md                   # SageMath usage guide"
echo "INTERACTIVE_PLOTS_GUIDE.md          # Interactive plots guide"
echo "VSCODE_JUPYTER_SETUP.md             # VS Code setup guide"
echo ""

echo "For more help: https://doc.sagemath.org/"
