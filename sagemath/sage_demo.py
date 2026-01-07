#!/usr/bin/env python
"""
SageMath Demo Script
Shows how to use SageMath as a Python library for computations and plotting
"""

from sage.all import *

print("=" * 60)
print("SageMath Python Script Demo")
print("=" * 60)

# 1. Symbolic Mathematics
print("\n1. SYMBOLIC MATHEMATICS:")
x = var('x')
expr = x**3 - 3*x**2 + 2*x - 5
print(f"Expression: {expr}")
print(f"Derivative: {diff(expr, x)}")
print(f"Integral: {integrate(expr, x)}")

# 2. Number Theory
print("\n2. NUMBER THEORY:")
print(f"Prime factorization of 2024: {factor(2024)}")
print(f"Is 2027 prime? {is_prime(2027)}")
print(f"Next prime after 2024: {next_prime(2024)}")

# 3. Linear Algebra
print("\n3. LINEAR ALGEBRA:")
A = matrix([[1, 2], [3, 4]])
print(f"Matrix A:\n{A}")
print(f"Determinant: {A.det()}")
print(f"Eigenvalues: {A.eigenvalues()}")

# 4. Plotting - Save to file
print("\n4. PLOTTING:")
print("Creating plots...")

# Plot 1: Simple function plot
p1 = plot(sin(x), (x, -2*pi, 2*pi), 
          title='Sine Function',
          color='blue',
          gridlines=True)
p1.save('sage_plot_sine.png')
print("✓ Saved: sage_plot_sine.png")

# Plot 2: Multiple functions
p2 = plot([sin(x), cos(x), tan(x)], (x, -pi, pi),
          legend_label=['sin(x)', 'cos(x)', 'tan(x)'],
          ymin=-2, ymax=2,
          title='Trigonometric Functions',
          gridlines=True)
p2.save('sage_plot_trig.png')
print("✓ Saved: sage_plot_trig.png")

# Plot 3: 3D surface plot
var('y')
p3 = plot3d(sin(sqrt(x**2 + y**2)), (x, -5, 5), (y, -5, 5),
            color='blue')
p3.save('sage_plot_3d.png')
print("✓ Saved: sage_plot_3d.png")

# Plot 4: Parametric plot
t = var('t')
p4 = parametric_plot([cos(t), sin(t)], (t, 0, 2*pi),
                     color='red',
                     thickness=2,
                     title='Unit Circle')
p4.save('sage_plot_circle.png')
print("✓ Saved: sage_plot_circle.png")

print("\n" + "=" * 60)
print("Demo complete! Check the PNG files for plots.")
print("=" * 60)
