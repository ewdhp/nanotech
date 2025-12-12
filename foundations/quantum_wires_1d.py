"""
Quantum Wires - 1D Quantum Confinement
=======================================
Demonstrates quantum confinement in two dimensions (1D structure).
Particles are confined in x-y plane but free to move along z-axis.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
h = 6.626e-34  # Planck's constant (J·s)
hbar = h / (2 * np.pi)  # Reduced Planck's constant
m_e = 9.109e-31  # Electron mass (kg)
eV = 1.602e-19  # Electron volt to Joules

print("="*70)
print("QUANTUM WIRES - 1D CONFINEMENT")
print("="*70)
print("\nQuantum wires confine charge carriers in two dimensions (x and y),")
print("while allowing free motion along the wire (z-axis).\n")

# Parameters for quantum wire (rectangular cross-section)
L_x = 10e-9  # Width: 10 nm
L_y = 10e-9  # Height: 10 nm
m_eff = 0.067 * m_e  # Effective mass (GaAs electron)

print(f"Quantum Wire Parameters:")
print(f"  Cross-section: {L_x*1e9:.1f} nm × {L_y*1e9:.1f} nm")
print(f"  Effective mass: {m_eff/m_e:.3f} m_e")
print(f"  Material: GaAs (example)")

# Energy levels for 2D infinite square well (x-y plane)
# E_nm = (ℏ²π²/2m) * (n²/L_x² + m²/L_y²)
# Plus kinetic energy in z-direction: E_z = ℏ²k_z²/(2m)

def energy_subband(n, m, L_x, L_y, m_eff):
    """Calculate subband energy for quantum wire"""
    return (hbar**2 * np.pi**2 / (2 * m_eff)) * (n**2 / L_x**2 + m**2 / L_y**2)

print("\n" + "="*70)
print("SUBBAND ENERGIES (Bottom of each subband):")
print("="*70)

subbands = []
n_max = 3
for n in range(1, n_max + 1):
    for m in range(1, n_max + 1):
        E = energy_subband(n, m, L_x, L_y, m_eff)
        E_eV = E / eV
        subbands.append((n, m, E_eV))
        print(f"  (n={n}, m={m}): E = {E_eV:.4f} eV")

# Sort by energy
subbands.sort(key=lambda x: x[2])

print("\n" + "="*70)
print("SORTED SUBBAND ENERGIES:")
print("="*70)
for i, (n, m, E) in enumerate(subbands[:6]):
    print(f"  Subband {i+1} (n={n}, m={m}): {E:.4f} eV")

# 1D Density of States for each subband
# ρ_1D(E) = (1/π) * √(m/(2ℏ²)) * Σ 1/√(E - E_nm)

def dos_1d_subband(E, E_nm, m_eff):
    """1D DOS for a single subband"""
    if E <= E_nm:
        return 0
    return (1/np.pi) * np.sqrt(m_eff / (2 * hbar**2)) * 1 / np.sqrt((E - E_nm) * eV)

# Calculate total DOS
print("\n" + "="*70)
print("1D DENSITY OF STATES:")
print("="*70)

E_range = np.linspace(subbands[0][2], subbands[5][2] + 0.1, 1000)
dos_total = np.zeros_like(E_range)

for E_val in E_range:
    for n, m, E_nm in subbands[:6]:
        dos_total[np.where(E_range == E_val)[0][0]] += dos_1d_subband(E_val, E_nm, m_eff)

dos_total_converted = dos_total / eV * 1e-9  # Convert to states/(eV·nm)

print(f"DOS at E = {subbands[0][2] + 0.05:.3f} eV: {dos_total_converted[50]:.2e} states/(eV·nm)")
print("Note: DOS has 1/√E singularities at each subband edge")

# Wave functions
def wavefunction_2d(n, m, x, y, L_x, L_y):
    """Normalized 2D wave function for rectangular quantum wire"""
    return (2 / np.sqrt(L_x * L_y)) * np.sin(n * np.pi * x / L_x) * np.sin(m * np.pi * y / L_y)

# Size dependence
print("\n" + "="*70)
print("SIZE DEPENDENCE (E ∝ 1/L²):")
print("="*70)

widths = np.array([5, 10, 15, 20, 30]) * 1e-9  # nm
ground_energies = []

for L in widths:
    E = energy_subband(1, 1, L, L, m_eff)
    ground_energies.append(E / eV)
    print(f"  L = {L*1e9:.0f} nm: E₁₁ = {E/eV:.4f} eV")

# Plotting
print("\n" + "="*70)
print("GENERATING PLOTS...")
print("="*70)

fig = plt.figure(figsize=(15, 10))

# Plot 1: Subband energy diagram
ax1 = plt.subplot(2, 3, 1)
for i, (n, m, E) in enumerate(subbands[:8]):
    color = plt.cm.viridis(i/8)
    ax1.hlines(E, 0, 1, colors=color, linewidth=3)
    ax1.text(1.05, E, f'({n},{m}): {E:.3f} eV', va='center', fontsize=9)

ax1.set_xlim(-0.1, 1.8)
ax1.set_ylabel('Energy (eV)', fontsize=11)
ax1.set_title('Subband Energy Diagram', fontsize=12, fontweight='bold')
ax1.set_xticks([])
ax1.grid(True, alpha=0.3)

# Plot 2: 1D Density of States
ax2 = plt.subplot(2, 3, 2)
ax2.plot(E_range, dos_total_converted, linewidth=2, color='blue')

for n, m, E_nm in subbands[:6]:
    ax2.axvline(E_nm, color='red', linestyle='--', alpha=0.5, linewidth=1)

ax2.set_xlabel('Energy (eV)', fontsize=11)
ax2.set_ylabel('DOS (states/(eV·nm))', fontsize=11)
ax2.set_title('1D Density of States\n(1/√E singularities)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, np.max(dos_total_converted[np.isfinite(dos_total_converted)]) * 1.2)
ax2.grid(True, alpha=0.3)

# Plot 3: Size dependence
ax3 = plt.subplot(2, 3, 3)
ax3.plot(widths*1e9, ground_energies, 'o-', color='red', linewidth=2, markersize=8)
ax3.set_xlabel('Wire Width/Height (nm)', fontsize=11)
ax3.set_ylabel('Ground Subband Energy (eV)', fontsize=11)
ax3.set_title('Energy vs Wire Size\n(E ∝ 1/L²)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Wave function for ground state (n=1, m=1)
ax4 = plt.subplot(2, 3, 4, projection='3d')
x = np.linspace(0, L_x, 50)
y = np.linspace(0, L_y, 50)
X, Y = np.meshgrid(x, y)
psi_11 = wavefunction_2d(1, 1, X, Y, L_x, L_y)

surf = ax4.plot_surface(X*1e9, Y*1e9, psi_11, cmap='viridis', alpha=0.8)
ax4.set_xlabel('x (nm)', fontsize=10)
ax4.set_ylabel('y (nm)', fontsize=10)
ax4.set_zlabel('ψ(x,y)', fontsize=10)
ax4.set_title('Wave Function (n=1, m=1)', fontsize=11, fontweight='bold')

# Plot 5: Wave function for first excited state (n=1, m=2 or n=2, m=1)
ax5 = plt.subplot(2, 3, 5, projection='3d')
psi_12 = wavefunction_2d(1, 2, X, Y, L_x, L_y)

surf = ax5.plot_surface(X*1e9, Y*1e9, psi_12, cmap='plasma', alpha=0.8)
ax5.set_xlabel('x (nm)', fontsize=10)
ax5.set_ylabel('y (nm)', fontsize=10)
ax5.set_zlabel('ψ(x,y)', fontsize=10)
ax5.set_title('Wave Function (n=1, m=2)', fontsize=11, fontweight='bold')

# Plot 6: Probability density for ground state
ax6 = plt.subplot(2, 3, 6)
prob_11 = psi_11**2
contour = ax6.contourf(X*1e9, Y*1e9, prob_11, levels=20, cmap='hot')
plt.colorbar(contour, ax=ax6, label='|ψ|²')
ax6.set_xlabel('x (nm)', fontsize=11)
ax6.set_ylabel('y (nm)', fontsize=11)
ax6.set_title('Probability Density (n=1, m=1)', fontsize=12, fontweight='bold')
ax6.set_aspect('equal')

plt.tight_layout()
plt.savefig('/home/ewd/github/ewdhp/nanotech/quantum_wires_1d.png', dpi=300, bbox_inches='tight')
print("✓ Saved plot: quantum_wires_1d.png")
plt.close()

print("\n" + "="*70)
print("DISPERSION RELATION:")
print("="*70)
print("\nFor each subband (n,m), the dispersion relation along z is:")
print("  E(k_z) = E_nm + (ℏ²k_z²)/(2m*)")
print("\nwhere E_nm is the subband bottom energy")
print(f"  E₁₁ = {subbands[0][2]:.4f} eV (ground state)")
print(f"  E₁₂/E₂₁ = {subbands[1][2]:.4f} eV (first excited)")

# Calculate effective Fermi velocity for lowest subband
k_z = 1e9  # Example k-vector (1/nm)
E_kinetic = (hbar**2 * k_z**2) / (2 * m_eff) / eV
print(f"\nAt k_z = {k_z*1e-9:.1f} nm⁻¹:")
print(f"  Additional kinetic energy: {E_kinetic:.4f} eV")
print(f"  Total energy: {subbands[0][2] + E_kinetic:.4f} eV")

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)
print("• Quantum wires exhibit 2D confinement (1D structure)")
print("• Electrons are confined in x-y plane, free along z-axis")
print("• Multiple subbands with energies E_nm ∝ (n²/L_x² + m²/L_y²)")
print("• DOS has 1/√E singularities at subband edges")
print("• Ballistic transport possible along wire direction")
print("• Applications: Single-electron transistors, quantum computing")
print("="*70)
