"""
Heisenberg Uncertainty Principle Demonstration
==============================================
Demonstrates the Heisenberg uncertainty principle: ΔxΔp ≥ ℏ/2

This script shows:
1. Mathematical calculations of position-momentum uncertainty
2. Wave packet examples with different widths
3. Fourier transform relationship between position and momentum space
4. Visualization of the uncertainty relationship

Author: ewdhp
Date: November 6, 2025
"""

import numpy as np
import matplotlib
# Try to use an interactive backend
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        pass  # Use whatever is available
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.fft import fft, fftfreq, fftshift

# Physical constants
h = 6.626e-34  # Planck constant (J·s)
hbar = h / (2 * np.pi)  # Reduced Planck constant (J·s)
m_e = 9.109e-31  # Electron mass (kg)


def gaussian_wave_packet(x, x0, sigma_x, k0=0):
    """
    Create a Gaussian wave packet in position space
    
    Parameters:
    -----------
    x : array
        Position array
    x0 : float
        Center position
    sigma_x : float
        Position uncertainty (standard deviation)
    k0 : float
        Central wave number
    
    Returns:
    --------
    array : Complex wave packet
    """
    normalization = (2 * np.pi * sigma_x**2)**(-0.25)
    gaussian = np.exp(-(x - x0)**2 / (4 * sigma_x**2))
    plane_wave = np.exp(1j * k0 * x)
    return normalization * gaussian * plane_wave


def momentum_space_gaussian(p, p0, sigma_p):
    """
    Gaussian wave packet in momentum space
    
    Parameters:
    -----------
    p : array
        Momentum array
    p0 : float
        Central momentum
    sigma_p : float
        Momentum uncertainty (standard deviation)
    
    Returns:
    --------
    array : Momentum space wave function
    """
    normalization = (2 * np.pi * sigma_p**2)**(-0.25)
    return normalization * np.exp(-(p - p0)**2 / (4 * sigma_p**2))


def calculate_uncertainty(values, probability):
    """
    Calculate standard deviation (uncertainty)
    
    Parameters:
    -----------
    values : array
        Position or momentum values
    probability : array
        Probability distribution
    
    Returns:
    --------
    tuple : (mean, standard_deviation)
    """
    # Normalize probability
    probability = probability / np.trapezoid(probability, values)
    
    # Calculate mean
    mean = np.trapezoid(values * probability, values)
    
    # Calculate variance
    variance = np.trapezoid((values - mean)**2 * probability, values)
    
    return mean, np.sqrt(variance)


def fourier_transform_wavefunction(x, psi_x):
    """
    Compute momentum space wavefunction via Fourier transform
    
    Parameters:
    -----------
    x : array
        Position array
    psi_x : array
        Position space wavefunction
    
    Returns:
    --------
    tuple : (momentum array, momentum space wavefunction)
    """
    dx = x[1] - x[0]
    psi_k = fftshift(fft(fftshift(psi_x))) * dx / np.sqrt(2 * np.pi)
    k = fftshift(fftfreq(len(x), dx)) * 2 * np.pi
    p = hbar * k
    return p, psi_k


# ============================================================================
# CALCULATIONS
# ============================================================================

print("=" * 75)
print("HEISENBERG UNCERTAINTY PRINCIPLE DEMONSTRATION")
print("=" * 75)
print()
print("Fundamental Principle: ΔxΔp ≥ ℏ/2")
print(f"where ℏ = {hbar:.3e} J·s")
print(f"Minimum uncertainty product: ℏ/2 = {hbar/2:.3e} J·s")
print("=" * 75)
print()

# Define position space
x = np.linspace(-50e-9, 50e-9, 2000)  # -50 to 50 nm
dx = x[1] - x[0]

# Test different wave packet widths
sigma_x_values = [2e-9, 5e-9, 10e-9, 20e-9]  # Different position uncertainties (nm)
colors = ['#E63946', '#F77F00', '#06AED5', '#073B4C']

results = []

print("\nWAVE PACKET ANALYSIS:")
print("-" * 75)

for i, sigma_x in enumerate(sigma_x_values):
    # Create wave packet
    psi_x = gaussian_wave_packet(x, x0=0, sigma_x=sigma_x, k0=0)
    prob_x = np.abs(psi_x)**2
    
    # Calculate position uncertainty
    x_mean, delta_x = calculate_uncertainty(x, prob_x)
    
    # Get momentum space representation via Fourier transform
    p, psi_p = fourier_transform_wavefunction(x, psi_x)
    prob_p = np.abs(psi_p)**2
    
    # Calculate momentum uncertainty
    p_mean, delta_p = calculate_uncertainty(p, prob_p)
    
    # Calculate uncertainty product
    uncertainty_product = delta_x * delta_p
    ratio_to_minimum = uncertainty_product / (hbar / 2)
    
    results.append({
        'sigma_x': sigma_x,
        'x': x,
        'psi_x': psi_x,
        'prob_x': prob_x,
        'delta_x': delta_x,
        'p': p,
        'psi_p': psi_p,
        'prob_p': prob_p,
        'delta_p': delta_p,
        'uncertainty_product': uncertainty_product,
        'ratio': ratio_to_minimum
    })
    
    print(f"\nWave Packet {i+1} (σ_x = {sigma_x*1e9:.1f} nm):")
    print(f"  Position uncertainty (Δx):     {delta_x*1e9:.4f} nm")
    print(f"  Momentum uncertainty (Δp):     {delta_p:.4e} kg·m/s")
    print(f"  Uncertainty product (ΔxΔp):    {uncertainty_product:.4e} J·s")
    print(f"  Minimum allowed (ℏ/2):         {hbar/2:.4e} J·s")
    print(f"  Ratio to minimum:              {ratio_to_minimum:.4f}")
    print(f"  Satisfies HUP: {'✓ YES' if uncertainty_product >= hbar/2 else '✗ NO'}")

# Additional examples with different particles
print("\n" + "=" * 75)
print("UNCERTAINTY IN DIFFERENT SCENARIOS:")
print("-" * 75)

# Example 1: Electron in an atom
print("\n1. Electron in Hydrogen Atom (Bohr radius ~ 0.53 Å):")
delta_x_atom = 0.53e-10  # meters
delta_p_min = hbar / (2 * delta_x_atom)
velocity = delta_p_min / m_e
energy = 0.5 * m_e * velocity**2

print(f"   Position uncertainty: {delta_x_atom*1e10:.2f} Å")
print(f"   Minimum momentum uncertainty: {delta_p_min:.4e} kg·m/s")
print(f"   Corresponding velocity uncertainty: {velocity:.4e} m/s")
print(f"   Kinetic energy scale: {energy/1.602e-19:.2f} eV")

# Example 2: Electron in quantum dot
print("\n2. Electron Confined in Quantum Dot (5 nm diameter):")
delta_x_qd = 5e-9 / 2  # radius
delta_p_qd = hbar / (2 * delta_x_qd)
velocity_qd = delta_p_qd / m_e
energy_qd = 0.5 * m_e * velocity_qd**2

print(f"   Position uncertainty: {delta_x_qd*1e9:.2f} nm")
print(f"   Minimum momentum uncertainty: {delta_p_qd:.4e} kg·m/s")
print(f"   Velocity uncertainty: {velocity_qd:.4e} m/s")
print(f"   Kinetic energy scale: {energy_qd/1.602e-19:.4f} eV")

# Example 3: Measuring electron position precisely
print("\n3. Precisely Measuring Electron Position (Δx = 0.1 nm):")
delta_x_precise = 0.1e-9
delta_p_precise = hbar / (2 * delta_x_precise)
velocity_precise = delta_p_precise / m_e
energy_precise = 0.5 * m_e * velocity_precise**2

print(f"   Position uncertainty: {delta_x_precise*1e10:.2f} Å")
print(f"   Minimum momentum uncertainty: {delta_p_precise:.4e} kg·m/s")
print(f"   Velocity uncertainty: {velocity_precise:.4e} m/s")
print(f"   Kinetic energy uncertainty: {energy_precise/1.602e-19:.2f} eV")

# Example 4: Macroscopic object
print("\n4. Macroscopic Object (1 μm precision, 1 mg mass):")
delta_x_macro = 1e-6
mass_macro = 1e-6  # kg
delta_p_macro = hbar / (2 * delta_x_macro)
velocity_macro = delta_p_macro / mass_macro

print(f"   Position uncertainty: {delta_x_macro*1e6:.1f} μm")
print(f"   Minimum momentum uncertainty: {delta_p_macro:.4e} kg·m/s")
print(f"   Velocity uncertainty: {velocity_macro:.4e} m/s")
print(f"   (Completely negligible for macroscopic objects!)")

print()
print("=" * 75)


# ============================================================================
# VISUALIZATIONS
# ============================================================================

# Set up the figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.35)

# Main title
fig.suptitle('Heisenberg Uncertainty Principle: ΔxΔp ≥ ℏ/2', 
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Position space wave functions (top left)
ax1 = fig.add_subplot(gs[0, 0])
for i, result in enumerate(results):
    ax1.plot(result['x'] * 1e9, np.real(result['psi_x']), 
            color=colors[i], linewidth=2, alpha=0.7,
            label=f'σ={result["sigma_x"]*1e9:.0f} nm')
ax1.set_xlabel('Position x (nm)', fontsize=11)
ax1.set_ylabel('Re[ψ(x)]', fontsize=11)
ax1.set_title('Wave Functions in Position Space', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.5)

# Plot 2: Position probability distributions (top middle)
ax2 = fig.add_subplot(gs[0, 1])
for i, result in enumerate(results):
    ax2.fill_between(result['x'] * 1e9, result['prob_x'], 
                     color=colors[i], alpha=0.6,
                     label=f'Δx={result["delta_x"]*1e9:.2f} nm')
ax2.set_xlabel('Position x (nm)', fontsize=11)
ax2.set_ylabel('|ψ(x)|²', fontsize=11)
ax2.set_title('Position Probability Density', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Momentum probability distributions (top right)
ax3 = fig.add_subplot(gs[0, 2])
for i, result in enumerate(results):
    # Filter momentum space to reasonable range
    mask = np.abs(result['p']) < 5e-24
    ax3.fill_between(result['p'][mask] * 1e24, result['prob_p'][mask] * 1e-24, 
                     color=colors[i], alpha=0.6,
                     label=f'Δp={result["delta_p"]*1e24:.2f}×10⁻²⁴')
ax3.set_xlabel('Momentum p (×10⁻²⁴ kg·m/s)', fontsize=11)
ax3.set_ylabel('|φ(p)|² (×10²⁴)', fontsize=11)
ax3.set_title('Momentum Probability Density', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Narrow wave packet example (second row left)
ax4 = fig.add_subplot(gs[1, 0])
narrow_result = results[0]  # Narrowest packet
ax4_twin = ax4.twinx()

line1 = ax4.plot(narrow_result['x'] * 1e9, np.real(narrow_result['psi_x']), 
                'b-', linewidth=2, label='Re[ψ(x)]')
line2 = ax4.fill_between(narrow_result['x'] * 1e9, narrow_result['prob_x'], 
                         color='blue', alpha=0.3, label='|ψ(x)|²')
ax4.set_xlabel('Position x (nm)', fontsize=11)
ax4.set_ylabel('Wave Function Amplitude', fontsize=11)

ax4.set_title(f'Narrow Wave Packet: Δx={narrow_result["delta_x"]*1e9:.2f} nm (Localized)', 
             fontsize=11, fontweight='bold')
ax4.legend(fontsize=9, loc='upper right')
ax4.grid(True, alpha=0.3)

# Plot 5: Wide wave packet example (second row middle)
ax5 = fig.add_subplot(gs[1, 1])

wide_result = results[-1]  # Widest packet
ax5.plot(wide_result['x'] * 1e9, np.real(wide_result['psi_x']), 
        'b-', linewidth=2, label='Re[ψ(x)]')
ax5.fill_between(wide_result['x'] * 1e9, wide_result['prob_x'], 
                 color='blue', alpha=0.3, label='|ψ(x)|²')
ax5.set_xlabel('Position x (nm)', fontsize=11)
ax5.set_ylabel('Wave Function Amplitude', fontsize=11)

ax5.set_title(f'Wide Wave Packet: Δx={wide_result["delta_x"]*1e9:.2f} nm (Delocalized)', 
             fontsize=11, fontweight='bold')
ax5.legend(fontsize=9, loc='upper right')
ax5.grid(True, alpha=0.3)

# Plot 6: Uncertainty product comparison (second row right)
ax6 = fig.add_subplot(gs[1, 2])
sigma_x_plot = [r['sigma_x'] * 1e9 for r in results]
uncertainty_products = [r['uncertainty_product'] for r in results]
ratios = [r['ratio'] for r in results]

bars = ax6.bar(range(len(results)), ratios, color=colors, alpha=0.7, edgecolor='black')
ax6.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Minimum (ℏ/2)')
ax6.set_xlabel('Wave Packet', fontsize=11)
ax6.set_ylabel('ΔxΔp / (ℏ/2)', fontsize=11)
ax6.set_title('Uncertainty Product Comparison', fontsize=12, fontweight='bold')
ax6.set_xticks(range(len(results)))
ax6.set_xticklabels([f'σ={s:.0f}nm' for s in sigma_x_plot], fontsize=9)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, ratio) in enumerate(zip(bars, ratios)):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{ratio:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 7: Position uncertainty vs Momentum uncertainty (third row left)
ax7 = fig.add_subplot(gs[2, 0])
delta_x_values = [r['delta_x'] * 1e9 for r in results]
delta_p_values = [r['delta_p'] * 1e24 for r in results]

ax7.plot(delta_x_values, delta_p_values, 'o-', markersize=10, 
        linewidth=2, color='#06AED5', markeredgecolor='black')

# Plot theoretical minimum curve
delta_x_theory = np.linspace(min(delta_x_values) * 0.5, max(delta_x_values) * 1.5, 100)
delta_p_theory = (hbar / 2) / (delta_x_theory * 1e-9) * 1e24

ax7.plot(delta_x_theory, delta_p_theory, '--', linewidth=2, 
        color='red', label='Minimum: Δp = ℏ/(2Δx)')
ax7.fill_between(delta_x_theory, 0, delta_p_theory, alpha=0.2, color='red',
                label='Forbidden region')

ax7.set_xlabel('Position Uncertainty Δx (nm)', fontsize=11)
ax7.set_ylabel('Momentum Uncertainty Δp (×10⁻²⁴ kg·m/s)', fontsize=11)
ax7.set_title('Position-Momentum Uncertainty Trade-off', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# Plot 8: Conceptual illustration (third row middle)
ax8 = fig.add_subplot(gs[2, 1])
ax8.text(0.5, 0.9, 'Heisenberg Uncertainty Principle', 
        ha='center', va='top', fontsize=13, fontweight='bold', transform=ax8.transAxes)
ax8.text(0.5, 0.75, 'ΔxΔp ≥ ℏ/2', 
        ha='center', va='top', fontsize=20, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        transform=ax8.transAxes)

implications = [
    '• Precise position → Uncertain momentum',
    '• Precise momentum → Uncertain position',
    '• Fundamental limit of nature',
    '• Not due to measurement limitations',
    '• Consequence of wave nature'
]

y_pos = 0.55
for text in implications:
    ax8.text(0.1, y_pos, text, ha='left', va='top', fontsize=10,
            transform=ax8.transAxes)
    y_pos -= 0.12

ax8.text(0.5, 0.05, f'ℏ = {hbar:.3e} J·s', 
        ha='center', va='bottom', fontsize=9, style='italic',
        transform=ax8.transAxes)
ax8.axis('off')

# Plot 9: Wave packet evolution illustration (third row right)
ax9 = fig.add_subplot(gs[2, 2])
x_small = np.linspace(-30e-9, 30e-9, 500)
times = [0, 0.3, 0.6, 1.0]
alphas = [1.0, 0.7, 0.5, 0.3]

for t, alpha in zip(times, alphas):
    sigma_t = results[1]['sigma_x'] * np.sqrt(1 + (hbar * t / (2 * m_e * results[1]['sigma_x']**2))**2)
    psi_t = gaussian_wave_packet(x_small, 0, sigma_t, k0=0)
    ax9.plot(x_small * 1e9, np.abs(psi_t)**2, linewidth=2, alpha=alpha,
            label=f't={t:.1f} (relative)')

ax9.set_xlabel('Position x (nm)', fontsize=11)
ax9.set_ylabel('|ψ(x,t)|²', fontsize=11)
ax9.set_title('Wave Packet Spreading Over Time', fontsize=12, fontweight='bold')
ax9.legend(fontsize=9)
ax9.grid(True, alpha=0.3)

# Plot 10: Energy-time uncertainty analogy (bottom left)
ax10 = fig.add_subplot(gs[3, 0])
delta_t = np.logspace(-15, -9, 100)  # Time scales from fs to µs
delta_E_min = hbar / (2 * delta_t) / 1.602e-19  # Convert to eV

ax10.loglog(delta_t * 1e15, delta_E_min, linewidth=3, color='#E63946')
ax10.fill_between(delta_t * 1e15, delta_E_min, 1e10, alpha=0.2, color='#E63946',
                 label='Allowed region')
ax10.set_xlabel('Time Interval Δt (fs)', fontsize=11)
ax10.set_ylabel('Minimum Energy Uncertainty ΔE (eV)', fontsize=11)
ax10.set_title('Energy-Time Uncertainty: ΔEΔt ≥ ℏ/2', fontsize=12, fontweight='bold')
ax10.legend(fontsize=9)
ax10.grid(True, alpha=0.3, which='both')

# Add some reference points
points = [
    (1e-15, hbar/(2*1e-15)/1.602e-19, 'Attosecond pulse'),
    (1e-12, hbar/(2*1e-12)/1.602e-19, 'Femtosecond laser'),
    (1e-9, hbar/(2*1e-9)/1.602e-19, 'Nanosecond')
]
for t, E, label in points:
    ax10.plot(t*1e15, E, 'ko', markersize=8)
    ax10.annotate(label, (t*1e15, E), xytext=(10, 10), 
                 textcoords='offset points', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

# Plot 11: Quantum vs Classical comparison (bottom middle)
ax11 = fig.add_subplot(gs[3, 1])
scenarios = ['Electron\nin atom', 'Electron in\nQD', 'Precise\nmeasurement', 'Macroscopic\nobject']
delta_x_scenarios = [0.53e-10, 2.5e-9, 0.1e-9, 1e-6]
delta_p_scenarios = [hbar/(2*dx) for dx in delta_x_scenarios]

x_pos = np.arange(len(scenarios))
bars = ax11.bar(x_pos, [dp*1e24 for dp in delta_p_scenarios], 
               color=['#E63946', '#F77F00', '#06AED5', '#073B4C'], 
               alpha=0.7, edgecolor='black')

ax11.set_ylabel('Minimum Δp (×10⁻²⁴ kg·m/s)', fontsize=11)
ax11.set_title('Momentum Uncertainty in Different Scenarios', fontsize=12, fontweight='bold')
ax11.set_xticks(x_pos)
ax11.set_xticklabels(scenarios, fontsize=9)
ax11.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, dp) in enumerate(zip(bars, delta_p_scenarios)):
    height = bar.get_height()
    ax11.text(bar.get_x() + bar.get_width()/2., height,
            f'{dp*1e24:.2e}', ha='center', va='bottom', fontsize=8, rotation=0)

# Plot 12: Summary and key insights (bottom right)
ax12 = fig.add_subplot(gs[3, 2])
ax12.text(0.5, 0.95, 'Key Insights', 
         ha='center', va='top', fontsize=13, fontweight='bold', 
         transform=ax12.transAxes)

insights = [
    '1. Position & momentum cannot both\n   be known precisely',
    
    '2. Narrower position spread →\n   Wider momentum spread',
    
    '3. Wave packets spread over time\n   due to momentum uncertainty',
    
    '4. Fundamental quantum limit,\n   not measurement error',
    
    '5. Energy-time uncertainty:\n   ΔEΔt ≥ ℏ/2',
    
    '6. Essential for atomic stability\n   and quantum behavior'
]

y_pos = 0.80
for text in insights:
    ax12.text(0.05, y_pos, text, ha='left', va='top', fontsize=9,
             transform=ax12.transAxes, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))
    y_pos -= 0.14

ax12.axis('off')

# Display the plots
print("\nDisplaying interactive plots...")
print("Close the plot window to continue.\n")
plt.show(block=True)

print("\n" + "=" * 75)
print("VISUALIZATION COMPLETE")
print("=" * 75)
print("\nKey Observations from Plots:")
print("• Narrow wave packets (small Δx) → Wide momentum distributions (large Δp)")
print("• Wide wave packets (large Δx) → Narrow momentum distributions (small Δp)")
print("• All wave packets satisfy ΔxΔp ≥ ℏ/2")
print("• Gaussian wave packets achieve the minimum: ΔxΔp = ℏ/2")
print("• Uncertainty is a fundamental property of wave mechanics")
print("=" * 75)
