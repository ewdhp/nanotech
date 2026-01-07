# SageMath Interactive Plots - Quick Start Guide

## 📊 What Are Interactive Plots?

Interactive plots let you adjust function parameters with **sliders** in real-time!
Instead of creating multiple static plots, you get one plot with interactive controls.

## 🎯 Three Ways to Create Interactive Plots

### Method 1: Jupyter Notebook (BEST - True Sliders!)

#### Step 1: Install Jupyter
```bash
conda activate sage
conda install jupyter ipywidgets -y
```

#### Step 2: Start Jupyter Notebook
```bash
cd /home/ewd/github/nanotech
jupyter notebook
```

#### Step 3: Open the Interactive Notebook
- Open `sage_interactive_plots.ipynb` in Jupyter
- Run each cell
- **Play with the sliders!** 🎚️

#### Example - Interactive Sine Wave:
```python
from sage.all import *
from sage.repl.ipython_kernel.interact import interact

@interact
def plot_sine(amplitude=(1, 10, 0.5), frequency=(0.5, 5, 0.25)):
    """Move sliders to change amplitude and frequency!"""
    x = var('x')
    p = plot(amplitude * sin(frequency * x), (x, -2*pi, 2*pi),
             title=f'y = {amplitude}·sin({frequency}x)',
             gridlines=True, thickness=2)
    show(p)
```

**Result**: You get sliders above the plot that update it in real-time! ✨

---

### Method 2: Static Parameter Variations (No Sliders)

If you can't use Jupyter, create plots showing different parameter values:

```bash
/home/ewd/miniconda3/envs/sage/bin/python sage_parameters_demo.py
```

**Result**: Creates 8 PNG files showing how different parameters affect functions:
- `param_amplitude.png` - Different amplitudes
- `param_frequency.png` - Different frequencies  
- `param_phase.png` - Different phase shifts
- `param_quadratic.png` - Different coefficients
- `param_decay.png` - Different decay rates
- `param_damping.png` - Different damping factors
- `param_powers.png` - Different powers
- `param_lissajous.png` - Different parametric ratios

---

### Method 3: SageMath Native Notebook

```bash
conda activate sage
sage --notebook=jupyter
```

This starts Jupyter with SageMath kernel pre-configured.

---

## 📝 Interactive Plot Examples

### 1. Sine Wave Explorer
```python
@interact
def sine_wave(amplitude=(1, 10), frequency=(0.5, 5), phase=(0, 2*pi)):
    x = var('x')
    plot(amplitude * sin(frequency * x + phase), (x, -2*pi, 2*pi)).show()
```
**Controls**: 3 sliders (amplitude, frequency, phase)

### 2. Polynomial Root Finder
```python
@interact
def quadratic(a=(-3, 3), b=(-5, 5), c=(-5, 5)):
    x = var('x')
    p = plot(a*x**2 + b*x + c, (x, -6, 6), ymin=-15, ymax=15)
    # Add roots as red dots
    roots = solve(a*x**2 + b*x + c == 0, x)
    for r in roots:
        p += point((r.rhs(), 0), size=80, color='red')
    show(p)
```
**Controls**: 3 sliders (a, b, c) - see roots move as you change coefficients!

### 3. Lissajous Curves
```python
@interact
def lissajous(a=(1, 8), b=(1, 8), delta=(0, 2*pi)):
    t = var('t')
    parametric_plot([sin(a*t + delta), sin(b*t)], 
                   (t, 0, 2*pi), aspect_ratio=1).show()
```
**Controls**: 3 sliders - create beautiful patterns!

### 4. Function Comparison
```python
@interact
def compare(n=(1, 10)):
    x = var('x')
    p = Graphics()
    for i in range(1, n+1):
        p += plot(x**i, (x, -2, 2), legend_label=f'x^{i}')
    show(p)
```
**Controls**: 1 slider - show powers from x¹ to xⁿ

### 5. 3D Surface Explorer
```python
@interact
def surface(func=['sin(sqrt(x^2+y^2))', 'cos(x)*sin(y)', 'x^2-y^2']):
    x, y = var('x y')
    f = sage_eval(func, locals={'x': x, 'y': y})
    plot3d(f, (x, -3, 3), (y, -3, 3)).show()
```
**Controls**: Dropdown menu - switch between 3D surfaces!

---

## 🎮 Control Types

### Sliders (continuous values)
```python
@interact
def f(x=(0, 10, 0.5)):  # min=0, max=10, step=0.5
    pass
```

### Integer Sliders
```python
@interact  
def f(n=(1, 10)):  # integers from 1 to 10
    pass
```

### Dropdown Menu
```python
@interact
def f(option=['choice1', 'choice2', 'choice3']):
    pass
```

### Checkbox
```python
@interact
def f(show_grid=True):  # Boolean checkbox
    pass
```

### Text Input
```python
@interact
def f(expr=input_box('sin(x)', label='Function:')):
    pass
```

---

## 📁 Files in This Directory

| File | Description |
|------|-------------|
| `sage_interactive_plots.ipynb` | Jupyter notebook with interactive sliders |
| `sage_interactive_demo.py` | Python script with interactive definitions |
| `sage_parameters_demo.py` | Creates static plots showing parameter effects |
| `param_*.png` | 8 static plots showing parameter variations |

---

## 🚀 Quick Start Commands

```bash
# Activate SageMath environment
conda activate sage

# Launch Jupyter for interactive plots
jupyter notebook

# Or launch SageMath notebook
sage --notebook=jupyter

# Create static parameter plots
python sage_parameters_demo.py
```

---

## 💡 Tips

1. **Jupyter is best** - You get real interactive sliders
2. **Start simple** - Begin with 1-2 sliders, add more as needed
3. **Use ranges wisely** - Set min/max that make sense for your function
4. **Add titles** - Help users understand what parameters do
5. **Combine plots** - Use `Graphics()` to overlay multiple plots

---

## 🎓 Learn More

- SageMath Documentation: https://doc.sagemath.org/
- Interact Tutorial: https://doc.sagemath.org/html/en/prep/Quickstarts/Interact.html
- More examples in `sage_interactive_plots.ipynb`

**Enjoy exploring mathematics interactively!** 🎉
