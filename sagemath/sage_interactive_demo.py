#!/usr/bin/env python
"""
SageMath Interactive Plots Demo
Shows how to create interactive plots with sliders for function parameters
Run this in Jupyter Notebook for best experience, or use sage --notebook
"""

from sage.all import *
from sage.repl.ipython_kernel.interact import interact

print("=" * 70)
print("SageMath Interactive Plots Demo")
print("=" * 70)

# Example 1: Interactive plot with sliders (works in Jupyter/sage notebook)
print("\n1. INTERACTIVE PLOT FUNCTION (for Jupyter notebook):")
print("   This creates sliders to adjust parameters in real-time\n")

@interact
def plot_sin_wave(amplitude=(1, 10, 0.5), frequency=(0.5, 5, 0.25), phase=(0, 2*pi, 0.1)):
    """
    Interactive sine wave plot
    - amplitude: controls the height
    - frequency: controls how many cycles
    - phase: controls horizontal shift
    """
    x = var('x')
    p = plot(amplitude * sin(frequency * x + phase), (x, -2*pi, 2*pi),
             title=f'y = {amplitude}·sin({frequency}·x + {phase:.2f})',
             gridlines=True,
             thickness=2)
    p.show()

# Example 2: Parametric function explorer
print("\n2. PARAMETRIC CURVE EXPLORER:")

@interact
def lissajous_curve(a=(1, 5, 1), b=(1, 5, 1), delta=(0, 2*pi, 0.1)):
    """
    Interactive Lissajous curves
    x = sin(a·t + delta)
    y = sin(b·t)
    """
    t = var('t')
    p = parametric_plot([sin(a*t + delta), sin(b*t)], (t, 0, 2*pi),
                       color='red', thickness=2,
                       title=f'Lissajous: a={a}, b={b}, δ={delta:.2f}')
    p.show()

# Example 3: 3D surface explorer
print("\n3. 3D SURFACE EXPLORER:")

@interact
def surface_plot(function=['sin(sqrt(x^2 + y^2))', 'cos(x)*sin(y)', 'x^2 - y^2', 'exp(-(x^2+y^2))']):
    """
    Interactive 3D surface plots
    """
    x, y = var('x y')
    # Convert string to expression
    f = sage_eval(function, locals={'x': x, 'y': y})
    p = plot3d(f, (x, -3, 3), (y, -3, 3),
              color='blue', opacity=0.8)
    p.show()

# Example 4: Polynomial explorer
print("\n4. POLYNOMIAL ROOTS EXPLORER:")

@interact
def polynomial_plot(a=(-2, 2, 0.1), b=(-2, 2, 0.1), c=(-2, 2, 0.1)):
    """
    Explore quadratic polynomial: ax² + bx + c
    """
    x = var('x')
    poly = a*x**2 + b*x + c
    
    # Plot the polynomial
    p = plot(poly, (x, -5, 5), 
            ymin=-10, ymax=10,
            title=f'y = {a}x² + {b}x + {c}',
            gridlines=True)
    
    # Add roots if they exist
    try:
        roots = solve(poly == 0, x)
        for root in roots:
            p += point((root.rhs(), 0), size=50, color='red')
    except:
        pass
    
    p.show()

# Example 5: Multiple function comparison
print("\n5. FUNCTION COMPARISON:")

@interact
def compare_functions(n=(1, 10, 1)):
    """
    Compare x^n for different values of n
    """
    x = var('x')
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    p = Graphics()
    
    for i in range(1, min(n+1, 6)):
        p += plot(x**i, (x, -2, 2), 
                 color=colors[i-1] if i <= len(colors) else 'black',
                 legend_label=f'x^{i}',
                 ymin=-5, ymax=5)
    
    p.show()

print("\n" + "=" * 70)
print("HOW TO USE:")
print("=" * 70)
print("\nOPTION 1 - Jupyter Notebook (Recommended):")
print("  1. Install jupyter in sage environment:")
print("     conda activate sage")
print("     conda install jupyter")
print("  2. Start Jupyter:")
print("     jupyter notebook")
print("  3. Create new notebook, run this code - you'll see sliders!")
print("\nOPTION 2 - SageMath Notebook:")
print("  sage --notebook=jupyter")
print("\nOPTION 3 - Terminal (saves plots to files):")
print("  Just run this script - plots saved to PNG files")
print("=" * 70)

# If running as script (not in notebook), create static examples
if not hasattr(sys, 'ps1') and 'IPython' not in sys.modules:
    print("\n⚠ Running as script - creating static plot examples...")
    print("  (For interactive sliders, use Jupyter notebook)\n")
    
    x = var('x')
    
    # Example with different amplitudes
    p1 = Graphics()
    for amp in [1, 2, 3, 4, 5]:
        p1 += plot(amp * sin(x), (x, -2*pi, 2*pi),
                  legend_label=f'amplitude={amp}',
                  thickness=2)
    p1.save('interactive_amplitude.png')
    print("✓ Saved: interactive_amplitude.png (amplitude variations)")
    
    # Example with different frequencies
    p2 = Graphics()
    for freq in [0.5, 1, 1.5, 2, 2.5]:
        p2 += plot(sin(freq * x), (x, -2*pi, 2*pi),
                  legend_label=f'freq={freq}',
                  thickness=2)
    p2.save('interactive_frequency.png')
    print("✓ Saved: interactive_frequency.png (frequency variations)")
    
    # Lissajous curves
    t = var('t')
    p3 = Graphics()
    for a, b in [(1,1), (1,2), (2,3), (3,4), (4,5)]:
        p3 += parametric_plot([sin(a*t), sin(b*t)], (t, 0, 2*pi),
                             legend_label=f'a={a},b={b}',
                             thickness=2)
    p3.save('interactive_lissajous.png')
    print("✓ Saved: interactive_lissajous.png (Lissajous curves)")
    
    print("\n✓ Static examples created!")
    print("  For REAL interactive sliders, use Jupyter notebook!")
