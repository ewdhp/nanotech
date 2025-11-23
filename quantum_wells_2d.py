"""
Quantum Wells - 2D Quantum Confinement
=======================================
Demonstrates quantum confinement in one dimension (2D structure).
Particles are free to move in x-y plane but confined in z-direction.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Physical constants
h = 6.626e-34  # Planck's constant (J·s)
hbar = h / (2 * np.pi)  # Reduced Planck's constant
m_e = 9.109e-31  # Electron mass (kg)
eV = 1.602e-19  # Electron volt to Joules

print("="*70)
print("QUANTUM WELLS - 2D CONFINEMENT")
print("="*70)
print("\nQuantum wells confine charge carriers in one dimension (z-axis),")
print("while allowing free motion in the x-y plane.\n")

# Parameters for quantum well
L_z = 10e-9  # Well width: 10 nm
m_eff = 0.067 * m_e  # Effective mass (GaAs electron)
V0 = 0.3 * eV  # Potential barrier height (300 meV)

print(f"Quantum Well Parameters:")
print(f"  Well width (L_z): {L_z*1e9:.1f} nm")
print(f"  Effective mass: {m_eff/m_e:.3f} m_e")
print(f"  Barrier height: {V0/eV:.2f} eV")
print(f"  Material: GaAs/AlGaAs (example)")

# Energy levels for infinite square well in z-direction
# E_n = (ℏ²π²/2m) * (n²/L_z²)

def energy_level_infinite(n, L, m):
    """Calculate energy level for infinite square well"""
    return (hbar**2 * np.pi**2 / (2 * m)) * (n**2 / L**2)

print("\n" + "="*70)
print("ENERGY LEVELS (Infinite Well Approximation):")
print("="*70)

n_max = 5
energy_levels = []
for n in range(1, n_max + 1):
    E = energy_level_infinite(n, L_z, m_eff)
    E_eV = E / eV
    energy_levels.append(E_eV)
    print(f"  n = {n}: E = {E_eV:.4f} eV")

print(f"\nEnergy spacing (n=1 to n=2): {energy_levels[1] - energy_levels[0]:.4f} eV")
print(f"Energy spacing (n=2 to n=3): {energy_levels[2] - energy_levels[1]:.4f} eV")

# 2D Density of States
# For each subband n, DOS is constant: ρ_2D = m*/(πℏ²)

def dos_2d(m_eff):
    """2D density of states (per unit area per unit energy)"""
    return m_eff / (np.pi * hbar**2)

dos = dos_2d(m_eff)
print("\n" + "="*70)
print("DENSITY OF STATES:")
print("="*70)
print(f"2D DOS (per subband): {dos*eV*1e-4:.2e} states/(eV·cm²)")
print("Note: DOS is constant (step function) for each subband")

# Wave functions for infinite well
def wavefunction_infinite(n, z, L):
    """Normalized wave function for infinite square well"""
    return np.sqrt(2/L) * np.sin(n * np.pi * z / L)

# Size dependence
print("\n" + "="*70)
print("SIZE DEPENDENCE (E ∝ 1/L²):")
print("="*70)

well_widths = np.array([5, 10, 15, 20, 30]) * 1e-9  # nm
ground_energies = []

for L in well_widths:
    E = energy_level_infinite(1, L, m_eff)
    ground_energies.append(E / eV)
    print(f"  L_z = {L*1e9:.0f} nm: E₁ = {E/eV:.4f} eV")

# Plotting
print("\n" + "="*70)
print("GENERATING PLOTS...")
print("="*70)

fig = plt.figure(figsize=(14, 10))

# Plot 1: Energy level diagram with wave functions
ax1 = plt.subplot(2, 3, 1)
z = np.linspace(0, L_z, 1000)

colors = plt.cm.viridis(np.linspace(0, 0.8, n_max))
for n in range(1, n_max + 1):
    E = energy_levels[n-1]
    psi = wavefunction_infinite(n, z, L_z)
    # Scale and shift wave function for visualization
    psi_scaled = psi * L_z * 1e9 / 4 + E
    ax1.plot(z*1e9, psi_scaled, color=colors[n-1], linewidth=2, label=f'n={n}')
    ax1.hlines(E, 0, L_z*1e9, colors=colors[n-1], linestyle='--', alpha=0.5)

ax1.set_xlabel('Position z (nm)', fontsize=11)
ax1.set_ylabel('Energy (eV)', fontsize=11)
ax1.set_title('Energy Levels & Wave Functions', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, L_z*1e9)

# Plot 2: Probability densities
ax2 = plt.subplot(2, 3, 2)
for n in range(1, min(4, n_max + 1)):
    psi = wavefunction_infinite(n, z, L_z)
    prob = psi**2
    ax2.plot(z*1e9, prob, linewidth=2, label=f'n={n}', color=colors[n-1])

ax2.set_xlabel('Position z (nm)', fontsize=11)
ax2.set_ylabel('Probability Density |ψ|²', fontsize=11)
ax2.set_title('Probability Distributions', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, L_z*1e9)

# Plot 3: Size dependence
ax3 = plt.subplot(2, 3, 3)
ax3.plot(well_widths*1e9, ground_energies, 'o-', color='red', 
         linewidth=2, markersize=8)
ax3.set_xlabel('Well Width L_z (nm)', fontsize=11)
ax3.set_ylabel('Ground State Energy E₁ (eV)', fontsize=11)
ax3.set_title('Energy vs Well Width\n(E ∝ 1/L²)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Energy level spacing
ax4 = plt.subplot(2, 3, 4)
n_values = np.arange(1, n_max + 1)
ax4.plot(n_values, energy_levels, 'o-', linewidth=2, markersize=8, color='blue')
ax4.set_xlabel('Quantum Number n', fontsize=11)
ax4.set_ylabel('Energy E_n (eV)', fontsize=11)
ax4.set_title('Energy vs Quantum Number\n(E ∝ n²)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: 2D Density of States (step function)
ax5 = plt.subplot(2, 3, 5)
E_range = np.linspace(0, energy_levels[3], 1000)
dos_total = np.zeros_like(E_range)

for i, E_n in enumerate(energy_levels[:4]):
    dos_total[E_range >= E_n] += dos * eV * 1e-4  # Convert to states/(eV·cm²)

ax5.plot(E_range, dos_total, linewidth=2, color='green')
for E_n in energy_levels[:4]:
    ax5.axvline(E_n, color='red', linestyle='--', alpha=0.5, linewidth=1)

ax5.set_xlabel('Energy (eV)', fontsize=11)
ax5.set_ylabel('DOS (states/(eV·cm²))', fontsize=11)
ax5.set_title('2D Density of States\n(Step Function)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Schematic of quantum well structure
ax6 = plt.subplot(2, 3, 6)
z_full = np.linspace(-5e-9, 15e-9, 1000)
V = np.zeros_like(z_full)
V[z_full < 0] = V0/eV
V[z_full > L_z] = V0/eV

ax6.fill_between(z_full*1e9, 0, V, alpha=0.3, color='gray', label='Barrier')
ax6.fill_between(z_full*1e9, 0, V, where=(z_full >= 0) & (z_full <= L_z), 
                  alpha=0.3, color='lightblue', label='Well')

# Add first 3 energy levels
for i, E in enumerate(energy_levels[:3]):
    if E < V0/eV:
        ax6.hlines(E, 0, L_z*1e9, colors=colors[i], linewidth=2, label=f'E_{i+1}')

ax6.set_xlabel('Position z (nm)', fontsize=11)
ax6.set_ylabel('Energy (eV)', fontsize=11)
ax6.set_title('Quantum Well Potential', fontsize=12, fontweight='bold')
ax6.set_xlim(-5, 15)
ax6.set_ylim(0, V0/eV*1.1)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ewd/github/ewdhp/nanotech/quantum_wells_2d.png', dpi=300, bbox_inches='tight')
print("✓ Saved plot: quantum_wells_2d.png")
plt.close()

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)
print("• Quantum wells exhibit 1D confinement (2D structure)")
print("• Electrons are free in x-y plane, quantized in z-direction")
print("• Energy levels: E_n ∝ n²/L_z²")
print("• DOS is constant per subband (step function)")
print("• Applications: Lasers, LEDs, HEMTs, quantum cascade lasers")
print("="*70)
