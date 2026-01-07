#!/usr/bin/env python
"""
Simple Interactive Plot Demo - Shows parameter variations
Creates multiple plots showing how changing parameters affects the function
"""

from sage.all import *

print("=" * 70)
print("Interactive Function Parameter Visualization")
print("=" * 70)

# Example 1: Sine wave with varying amplitude
print("\n1. AMPLITUDE EFFECT: y = A·sin(x)")
print("   Creating plot showing different amplitudes...")

x = var('x')
p1 = Graphics()
amplitudes = [0.5, 1, 2, 3, 5]
colors = ['lightblue', 'blue', 'green', 'orange', 'red']

for amp, color in zip(amplitudes, colors):
    p1 += plot(amp * sin(x), (x, -2*pi, 2*pi),
              color=color,
              thickness=2,
              legend_label=f'A = {amp}')

p1.set_legend_options(loc='upper right')
p1.axes_labels(['x', 'y'])
p1.save('param_amplitude.png', figsize=10)
print("   ✓ Saved: param_amplitude.png")

# Example 2: Sine wave with varying frequency
print("\n2. FREQUENCY EFFECT: y = sin(ω·x)")
print("   Creating plot showing different frequencies...")

p2 = Graphics()
frequencies = [0.5, 1, 2, 3, 4]

for freq, color in zip(frequencies, colors):
    p2 += plot(sin(freq * x), (x, -2*pi, 2*pi),
              color=color,
              thickness=2,
              legend_label=f'ω = {freq}')

p2.set_legend_options(loc='upper right')
p2.axes_labels(['x', 'y'])
p2.save('param_frequency.png', figsize=10)
print("   ✓ Saved: param_frequency.png")

# Example 3: Sine wave with varying phase
print("\n3. PHASE SHIFT EFFECT: y = sin(x + φ)")
print("   Creating plot showing different phase shifts...")

p3 = Graphics()
phases = [0, pi/4, pi/2, pi, 3*pi/2]
phase_labels = ['0', 'π/4', 'π/2', 'π', '3π/2']

for phase, label, color in zip(phases, phase_labels, colors):
    p3 += plot(sin(x + phase), (x, -2*pi, 2*pi),
              color=color,
              thickness=2,
              legend_label=f'φ = {label}')

p3.set_legend_options(loc='upper right')
p3.axes_labels(['x', 'y'])
p3.save('param_phase.png', figsize=10)
print("   ✓ Saved: param_phase.png")

# Example 4: Polynomial with varying coefficients
print("\n4. QUADRATIC COEFFICIENT: y = ax²")
print("   Creating plot showing different leading coefficients...")

p4 = Graphics()
coefficients = [-2, -1, -0.5, 0.5, 1, 2]

for coef in coefficients:
    color = 'red' if coef < 0 else 'blue'
    alpha = abs(coef) / 2
    p4 += plot(coef * x**2, (x, -3, 3),
              color=color,
              thickness=2,
              alpha=min(alpha, 1),
              legend_label=f'a = {coef}',
              ymin=-10, ymax=10)

p4.set_legend_options(loc='best', font_size=9)
p4.axes_labels(['x', 'y'])
p4.save('param_quadratic.png', figsize=10)
print("   ✓ Saved: param_quadratic.png")

# Example 5: Exponential decay with varying rates
print("\n5. DECAY RATE: y = e^(-kx)")
print("   Creating plot showing different decay rates...")

p5 = Graphics()
decay_rates = [0.1, 0.3, 0.5, 1, 2]

for k, color in zip(decay_rates, colors):
    p5 += plot(exp(-k * x), (x, 0, 5),
              color=color,
              thickness=2,
              legend_label=f'k = {k}')

p5.set_legend_options(loc='upper right')
p5.axes_labels(['x', 'y'])
p5.save('param_decay.png', figsize=10)
print("   ✓ Saved: param_decay.png")

# Example 6: Damped oscillation
print("\n6. DAMPING FACTOR: y = e^(-dx)·sin(x)")
print("   Creating plot showing different damping factors...")

p6 = Graphics()
damping = [0, 0.1, 0.2, 0.4, 0.8]

for d, color in zip(damping, colors):
    p6 += plot(exp(-d * x) * sin(x), (x, 0, 4*pi),
              color=color,
              thickness=2,
              legend_label=f'd = {d}')

p6.set_legend_options(loc='upper right')
p6.axes_labels(['x', 'y'])
p6.save('param_damping.png', figsize=10)
print("   ✓ Saved: param_damping.png")

# Example 7: Power functions
print("\n7. POWER FUNCTIONS: y = x^n")
print("   Creating plot showing different powers...")

p7 = Graphics()
powers = [1, 2, 3, 4, 5]

for n, color in zip(powers, colors):
    p7 += plot(x**n, (x, -1.5, 1.5),
              color=color,
              thickness=2,
              legend_label=f'n = {n}',
              ymin=-2, ymax=2)

p7.set_legend_options(loc='upper left')
p7.axes_labels(['x', 'y'])
p7.save('param_powers.png', figsize=10)
print("   ✓ Saved: param_powers.png")

# Example 8: Lissajous curves (parametric)
print("\n8. LISSAJOUS CURVES: x = sin(at), y = sin(bt)")
print("   Creating parametric plots with different ratios...")

t = var('t')
p8 = Graphics()
ratios = [(1,1), (1,2), (2,3), (3,4), (3,5)]

for (a, b), color in zip(ratios, colors):
    p8 += parametric_plot([sin(a*t), sin(b*t)], (t, 0, 2*pi),
                         color=color,
                         thickness=2,
                         legend_label=f'a:b = {a}:{b}')

p8.set_legend_options(loc='best')
p8.axes_labels(['x', 'y'])
p8.set_aspect_ratio(1)
p8.save('param_lissajous.png', figsize=8)
print("   ✓ Saved: param_lissajous.png")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nAll parameter variation plots created!")
print("\nThese static plots show how parameters affect functions.")
print("For INTERACTIVE sliders in real-time, use Jupyter:")
print("  1. conda activate sage")
print("  2. jupyter notebook")
print("  3. Open: sage_interactive_plots.ipynb")
print("\nOr use SageMath notebook:")
print("  sage --notebook=jupyter")
print("=" * 70)
