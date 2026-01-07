# SageMath Usage Guide

## Installation Location
- **Conda environment**: `~/miniconda3/envs/sage`
- **SageMath version**: 10.7
- **Python version**: 3.11

## Two Ways to Use SageMath

### Method 1: Interactive Terminal (sage command)

Activate the conda environment first:
```bash
conda activate sage
```

Then use SageMath interactively:
```bash
sage
```

Or run directly without activation:
```bash
/home/ewd/miniconda3/envs/sage/bin/sage
```

This opens an interactive shell where you can use `^` for exponentiation (SageMath syntax):
```python
sage: x^2 + 3*x + 2
sage: factor(x^2 + 3*x + 2)
sage: integrate(sin(x), x)
```

### Method 2: Python Script (as a library)

Run SageMath as a Python library:
```bash
conda activate sage
python your_script.py
```

Or directly:
```bash
/home/ewd/miniconda3/envs/sage/bin/python your_script.py
```

**Important**: In Python scripts, use `**` for exponentiation (Python syntax):
```python
from sage.all import *

x = var('x')
expr = x**2 + 3*x + 2  # Use ** not ^
print(factor(expr))
```

## Plotting Examples

### 2D Plots (Save to File)
```python
from sage.all import *

x = var('x')

# Simple plot
p = plot(sin(x), (x, -2*pi, 2*pi), 
         title='Sine Wave',
         color='blue',
         gridlines=True)
p.save('output.png')

# Multiple functions
p = plot([sin(x), cos(x)], (x, -pi, pi),
         legend_label=['sin', 'cos'],
         color=['blue', 'red'])
p.save('trig.png')

# Parametric plot
t = var('t')
p = parametric_plot([cos(t), sin(t)], (t, 0, 2*pi))
p.save('circle.png')
```

### 3D Plots
```python
from sage.all import *

x, y = var('x y')

# 3D surface
p = plot3d(sin(sqrt(x**2 + y**2)), (x, -5, 5), (y, -5, 5))
p.save('surface.png')
```

### Interactive Plots (requires display)
```python
# In interactive sage terminal
sage: plot(sin(x), (x, -2*pi, 2*pi)).show()  # Opens in browser/viewer
```

## Key Differences

| Aspect | Interactive `sage` | Python Script |
|--------|-------------------|---------------|
| Exponentiation | `x^2` | `x**2` |
| Import needed | No | `from sage.all import *` |
| Variable declaration | Automatic | `x = var('x')` |
| Plot display | `.show()` | `.save('file.png')` |

## Demo Script

See `sage_demo.py` for a complete example covering:
- Symbolic mathematics
- Number theory
- Linear algebra
- 2D and 3D plotting

Run it with:
```bash
/home/ewd/miniconda3/envs/sage/bin/python sage_demo.py
```

## Quick Start Aliases

Add to your `~/.bashrc`:
```bash
alias sage='/home/ewd/miniconda3/envs/sage/bin/sage'
alias sage-python='/home/ewd/miniconda3/envs/sage/bin/python'
```

Then use:
```bash
sage              # Interactive mode
sage-python script.py  # Run Python script
```
