# SageMath & Jupyter Tutorials

This directory contains all SageMath and Jupyter notebook tutorials and examples.

## 📚 Contents

### 🚀 Quick Start
- **[VSCODE_JUPYTER_SETUP.md](VSCODE_JUPYTER_SETUP.md)** - Setup guide for VS Code + Jupyter
- **[SAGEMATH_USAGE.md](SAGEMATH_USAGE.md)** - How to use SageMath (terminal vs Python)

### 📓 Jupyter Notebooks
- **[test_vscode_interactive.ipynb](test_vscode_interactive.ipynb)** - Quick test with sliders (start here!)
- **[sage_interactive_plots.ipynb](sage_interactive_plots.ipynb)** - Full collection of interactive examples

### 🐍 Python Scripts
- **[sage_demo.py](sage_demo.py)** - Basic SageMath demo (symbolic math, number theory, plotting)
- **[sage_interactive_demo.py](sage_interactive_demo.py)** - Interactive plot definitions
- **[sage_parameters_demo.py](sage_parameters_demo.py)** - Parameter variation visualizations

### � Installation
- **[install_sagemath_jupyter.sh](install_sagemath_jupyter.sh)** - Automated installation script for Ubuntu

### �📖 Guides
- **[INTERACTIVE_PLOTS_GUIDE.md](INTERACTIVE_PLOTS_GUIDE.md)** - Complete guide to interactive plots with sliders

## 🎯 Getting Started

### 0. Installation (First Time Only)

**Automated Installation:**
```bash
# Run the installation script
bash install_sagemath_jupyter.sh
```

This script will:
- ✅ Install Miniconda (if not already installed)
- ✅ Create `sage` conda environment
- ✅ Install SageMath 10.7 with Jupyter
- ✅ Configure everything automatically
- ✅ Run tests to verify installation

**Manual Installation:** See [VSCODE_JUPYTER_SETUP.md](VSCODE_JUPYTER_SETUP.md)

### 1. First Time Setup
```bash
# Activate SageMath environment
conda activate sage

# Install VS Code Jupyter extension
# Extension ID: ms-toolsai.jupyter
```

### 2. Run Your First Interactive Notebook
```bash
# Open in VS Code
code test_vscode_interactive.ipynb

# Or start Jupyter in browser
jupyter notebook
```

### 3. Try the Demos
```bash
# Run basic demo
python sage_demo.py

# Create parameter variation plots
python sage_parameters_demo.py
```

## 📊 What You'll Learn

- ✅ Symbolic mathematics (calculus, algebra)
- ✅ Number theory computations
- ✅ 2D and 3D plotting
- ✅ Interactive plots with sliders
- ✅ Parametric curves
- ✅ Using SageMath as Python library

## 🔧 Requirements

- SageMath 10.7 (installed via Conda)
- Jupyter (included with SageMath)
- VS Code with Jupyter extension (optional but recommended)

## 🎨 Examples Included

### Interactive Sliders
- Sine wave explorer (amplitude, frequency, phase)
- Polynomial root finder
- Lissajous curves
- 3D surface explorer
- Taylor series approximation

### Static Visualizations
- Parameter effects on functions
- Multiple function comparison
- Damped oscillations
- Power functions
- Decay rates

## 💡 Tips

- **Start with** `test_vscode_interactive.ipynb` to verify setup
- **Use VS Code** for best development experience
- **Read guides** for detailed explanations
- **Experiment** with the sliders to understand functions

## 📞 Quick Help

**Issue**: No kernel found
- **Solution**: Select `/home/ewd/miniconda3/envs/sage/bin/python`

**Issue**: Sliders don't appear
- **Solution**: Import `from sage.repl.ipython_kernel.interact import interact`

**Issue**: Plots don't display
- **Solution**: Use `show(p)` instead of just `p`

---

**Happy exploring!** 🚀
