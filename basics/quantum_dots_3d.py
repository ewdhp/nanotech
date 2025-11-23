"""
Quantum Dots - 3D Quantum Confinement
======================================
Demonstrates quantum confinement in all three dimensions (0D structure).
Particles are confined in a sphere, leading to discrete energy levels.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.special import spherical_jn
from scipy.optimize import brentq

# Physical constants
h = 6.626e-34  # Planck's constant (J·s)
hbar = h / (2 * np.pi)  # Reduced Planck's constant
m_e = 9.109e-31  # Electron mass (kg)
eV = 1.602e-19  # Electron volt to Joules

print("="*70)
print("QUANTUM DOTS - 3D CONFINEMENT")
print("="*70)
print("\nQuantum dots confine charge carriers in all three dimensions,")
print("creating discrete atomic-like energy levels.\n")

# Parameters for quantum dot
radius = 5e-9  # 5 nm radius
m_eff = 0.067 * m_e  # Effective mass (GaAs electron)

print(f"Quantum Dot Parameters:")
print(f"  Radius: {radius*1e9:.1f} nm")
print(f"  Effective mass: {m_eff/m_e:.3f} m_e")
print(f"  Material: GaAs (example)")

# Energy levels for 3D infinite spherical well
# E_nl = (ℏ²/2m) * (α_nl/R)²
# where α_nl are zeros of spherical Bessel functions

def find_bessel_zeros(l, num_zeros=5):
    """Find zeros of spherical Bessel function j_l"""
    zeros = []
    # Known approximate locations of zeros
    x = 0.1
    search_range = 50  # Look up to this value
    
    while len(zeros) < num_zeros and x < search_range:
        # Look for sign changes
        try:
            y1 = spherical_jn(l, x)
            y2 = spherical_jn(l, x + 0.5)
            
            if y1 * y2 < 0:  # Sign change detected
                zero = brentq(lambda t: spherical_jn(l, t), x, x + 0.5)
                if zero > 0.1 and not any(abs(zero - z) < 0.01 for z in zeros):
                    zeros.append(zero)
        except:
            pass
        x += 0.1
    
    return sorted(zeros)[:num_zeros]

# Calculate first few energy levels
print("\n" + "="*70)
print("ENERGY LEVELS (in eV):")
print("="*70)

energy_levels = []
states = []

for l in range(4):  # Angular momentum quantum numbers
    zeros = find_bessel_zeros(l, num_zeros=5)
    for i, alpha_nl in enumerate(zeros[:3], start=1):
        E = (hbar**2 / (2 * m_eff)) * (alpha_nl / radius)**2
        E_eV = E / eV
        energy_levels.append(E_eV)
        states.append(f"n={i}, l={l}")
        print(f"  State (n={i}, l={l}): E = {E_eV:.4f} eV (α = {alpha_nl:.3f})")

# Sort energy levels
sorted_indices = np.argsort(energy_levels)
energy_levels = [energy_levels[i] for i in sorted_indices[:8]]
states = [states[i] for i in sorted_indices[:8]]

print("\n" + "="*70)
print("QUANTUM CONFINEMENT ENERGY:")
print("="*70)
print(f"Ground state energy: {energy_levels[0]:.4f} eV")
print(f"First excited state: {energy_levels[1]:.4f} eV")
print(f"Energy gap: {energy_levels[1] - energy_levels[0]:.4f} eV")

# Size dependence
print("\n" + "="*70)
print("SIZE DEPENDENCE (E ∝ 1/R²):")
print("="*70)

radii = np.array([2, 3, 5, 7, 10]) * 1e-9  # nm to m
ground_state_energies = []

for R in radii:
    alpha_10 = find_bessel_zeros(0, num_zeros=1)[0]
    E = (hbar**2 / (2 * m_eff)) * (alpha_10 / R)**2
    ground_state_energies.append(E / eV)
    print(f"  R = {R*1e9:.0f} nm: E₀ = {E/eV:.4f} eV")

# Plotting
print("\n" + "="*70)
print("GENERATING PLOTS...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Energy level diagram
ax1 = axes[0, 0]
for i, (E, state) in enumerate(zip(energy_levels, states)):
    ax1.hlines(E, 0, 1, colors='blue', linewidth=2)
    ax1.text(1.05, E, f'{state}: {E:.3f} eV', va='center', fontsize=9)
ax1.set_xlim(-0.1, 1.5)
ax1.set_ylabel('Energy (eV)', fontsize=12)
ax1.set_title('Energy Level Diagram\n(Quantum Dot, R=5nm)', fontsize=12, fontweight='bold')
ax1.set_xticks([])
ax1.grid(True, alpha=0.3)

# Plot 2: Size dependence
ax2 = axes[0, 1]
ax2.plot(radii*1e9, ground_state_energies, 'o-', color='red', linewidth=2, markersize=8)
ax2.set_xlabel('Quantum Dot Radius (nm)', fontsize=12)
ax2.set_ylabel('Ground State Energy (eV)', fontsize=12)
ax2.set_title('Energy vs Size\n(E ∝ 1/R²)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Radial wave functions
ax3 = axes[1, 0]
r = np.linspace(0, radius, 1000)
for l in range(3):
    alpha = find_bessel_zeros(l, num_zeros=1)[0]
    psi = spherical_jn(l, alpha * r / radius)
    psi_normalized = psi / np.sqrt(np.trapezoid(r**2 * psi**2, r))
    ax3.plot(r*1e9, psi_normalized, label=f'l={l}', linewidth=2)
ax3.set_xlabel('Radius (nm)', fontsize=12)
ax3.set_ylabel('Radial Wave Function ψ(r)', fontsize=12)
ax3.set_title('Radial Wave Functions (n=1)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axvline(radius*1e9, color='red', linestyle='--', alpha=0.5, label='Dot boundary')

# Plot 4: Probability density
ax4 = axes[1, 1]
for l in range(3):
    alpha = find_bessel_zeros(l, num_zeros=1)[0]
    psi = spherical_jn(l, alpha * r / radius)
    prob_density = 4 * np.pi * r**2 * psi**2
    prob_density_norm = prob_density / np.max(prob_density)
    ax4.plot(r*1e9, prob_density_norm, label=f'l={l}', linewidth=2)
ax4.set_xlabel('Radius (nm)', fontsize=12)
ax4.set_ylabel('Radial Probability Density (normalized)', fontsize=12)
ax4.set_title('Radial Probability Distribution', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ewd/github/ewdhp/nanotech/quantum_dots_3d.png', dpi=300, bbox_inches='tight')
print("✓ Saved plot: quantum_dots_3d.png")
plt.close()

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)
print("• Quantum dots exhibit 3D confinement (0D structure)")
print("• Energy levels are discrete and atomic-like")
print("• Ground state energy increases as R decreases (E ∝ 1/R²)")
print("• Quantum dots are also called 'artificial atoms'")
print("• Applications: LEDs, solar cells, biological imaging")
print("="*70)
