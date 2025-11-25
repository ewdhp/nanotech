"""
Blue-shift in Absorption with Decreasing Size
==============================================
Demonstrates quantum confinement effects and the blue-shift phenomenon
in semiconductor nanoparticles (quantum dots) as size decreases.

Theory:
-------
As nanoparticle radius decreases below the exciton Bohr radius:
1. Quantum confinement increases
2. Effective bandgap increases: E_g(R) = E_g(bulk) + ℏ²π²/(2μR²) - 1.8e²/(εR)
3. Absorption edge shifts to shorter wavelengths (higher energy)
4. This is the "blue-shift" - moving toward blue/UV in spectrum
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Physical constants
h = 6.626e-34  # Planck's constant (J·s)
hbar = h / (2 * np.pi)  # Reduced Planck's constant
c = 3.0e8  # Speed of light (m/s)
eV = 1.602e-19  # Electron volt to Joules
m_e = 9.109e-31  # Electron mass (kg)
e = 1.602e-19  # Elementary charge (C)
epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)

print("="*70)
print("BLUE-SHIFT IN ABSORPTION WITH DECREASING SIZE")
print("="*70)
print("\nQuantum confinement in semiconductor nanoparticles causes the")
print("absorption edge to shift to higher energies (shorter wavelengths)")
print("as particle size decreases.\n")

# Material parameters (CdSe - a common quantum dot material)
material = "CdSe"
E_g_bulk = 1.74  # Bulk bandgap (eV) at room temperature
m_e_eff = 0.13 * m_e  # Effective electron mass
m_h_eff = 0.45 * m_e  # Effective hole mass
epsilon_r = 10.0  # Relative permittivity

# Reduced effective mass
mu = (m_e_eff * m_h_eff) / (m_e_eff + m_h_eff)

# Exciton Bohr radius (characteristic confinement length)
a_B = (4 * np.pi * epsilon_0 * epsilon_r * hbar**2) / (mu * e**2)

print(f"Material: {material}")
print(f"Bulk bandgap: {E_g_bulk:.2f} eV")
print(f"Electron effective mass: {m_e_eff/m_e:.2f} m_e")
print(f"Hole effective mass: {m_h_eff/m_e:.2f} m_e")
print(f"Exciton Bohr radius: {a_B*1e9:.2f} nm")
print(f"Relative permittivity: {epsilon_r:.1f}")

def effective_bandgap(R, E_g_bulk, mu, epsilon_r):
    """
    Calculate size-dependent bandgap using Brus equation (simplified)
    
    E_g(R) = E_g(bulk) + (ℏ²π²)/(2μR²) - 1.8e²/(εR) + smaller terms
    
    First term: bulk bandgap
    Second term: confinement energy (quantum size effect)
    Third term: Coulomb interaction (attractive, reduces gap)
    """
    # Confinement energy contribution
    E_confinement = (hbar**2 * np.pi**2) / (2 * mu * R**2) / eV
    
    # Coulomb correction (attractive interaction)
    epsilon = epsilon_r * epsilon_0
    E_coulomb = -1.8 * e**2 / (4 * np.pi * epsilon * R) / eV
    
    # Total effective bandgap
    E_g_eff = E_g_bulk + E_confinement + E_coulomb
    
    return E_g_eff, E_confinement, E_coulomb

def energy_to_wavelength(E_eV):
    """Convert energy in eV to wavelength in nm"""
    return (h * c) / (E_eV * eV) * 1e9

def wavelength_to_color(wavelength_nm):
    """Approximate color from wavelength for visualization"""
    if wavelength_nm < 380:
        return 'UV', '#9400D3'  # Violet for UV
    elif wavelength_nm < 450:
        return 'Violet', '#9400D3'
    elif wavelength_nm < 495:
        return 'Blue', '#0000FF'
    elif wavelength_nm < 570:
        return 'Green', '#00FF00'
    elif wavelength_nm < 590:
        return 'Yellow', '#FFFF00'
    elif wavelength_nm < 620:
        return 'Orange', '#FF7F00'
    elif wavelength_nm < 750:
        return 'Red', '#FF0000'
    else:
        return 'IR', '#8B0000'  # Dark red for IR

# Calculate size-dependent properties
print("\n" + "="*70)
print("SIZE-DEPENDENT BANDGAP AND ABSORPTION:")
print("="*70)

radii_nm = np.array([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0])  # nm
radii = radii_nm * 1e-9  # Convert to meters

bandgaps = []
wavelengths = []
confinement_energies = []
coulomb_energies = []

for R_nm, R in zip(radii_nm, radii):
    E_g, E_conf, E_coul = effective_bandgap(R, E_g_bulk, mu, epsilon_r)
    lambda_nm = energy_to_wavelength(E_g)
    color_name, color_code = wavelength_to_color(lambda_nm)
    
    bandgaps.append(E_g)
    wavelengths.append(lambda_nm)
    confinement_energies.append(E_conf)
    coulomb_energies.append(E_coul)
    
    print(f"R = {R_nm:5.1f} nm: E_g = {E_g:.3f} eV, λ = {lambda_nm:6.1f} nm ({color_name})")

# Bulk material
lambda_bulk = energy_to_wavelength(E_g_bulk)
color_bulk, _ = wavelength_to_color(lambda_bulk)
print(f"\nBulk material: E_g = {E_g_bulk:.2f} eV, λ = {lambda_bulk:.1f} nm ({color_bulk})")

# Blue-shift analysis
print("\n" + "="*70)
print("BLUE-SHIFT ANALYSIS:")
print("="*70)

smallest_R = radii_nm[0]
largest_R = radii_nm[-1]
blue_shift_energy = bandgaps[0] - bandgaps[-1]
blue_shift_wavelength = wavelengths[-1] - wavelengths[0]

print(f"Smallest QD (R = {smallest_R} nm): E_g = {bandgaps[0]:.3f} eV, λ = {wavelengths[0]:.1f} nm")
print(f"Largest QD (R = {largest_R} nm): E_g = {bandgaps[-1]:.3f} eV, λ = {wavelengths[-1]:.1f} nm")
print(f"Energy shift: {blue_shift_energy:.3f} eV (increase)")
print(f"Wavelength shift: {blue_shift_wavelength:.1f} nm (decrease - BLUE SHIFT)")
print(f"\nAs size decreases from {largest_R} to {smallest_R} nm:")
print(f"  • Absorption edge shifts {abs(blue_shift_wavelength):.1f} nm toward BLUE/UV")
print(f"  • Bandgap increases by {blue_shift_energy:.3f} eV ({blue_shift_energy/E_g_bulk*100:.1f}%)")

# Quantum confinement regime
print("\n" + "="*70)
print("CONFINEMENT REGIME:")
print("="*70)

for R_nm, R in zip(radii_nm, radii):
    regime_ratio = R / a_B
    if regime_ratio < 1:
        regime = "Strong confinement"
    elif regime_ratio < 2:
        regime = "Moderate confinement"
    else:
        regime = "Weak confinement"
    print(f"R = {R_nm:5.1f} nm: R/a_B = {regime_ratio:.2f} ({regime})")

# Generate plots
print("\n" + "="*70)
print("GENERATING PLOTS...")
print("="*70)

fig = plt.figure(figsize=(16, 10))

# Plot 1: Bandgap vs size
ax1 = plt.subplot(2, 3, 1)
ax1.plot(radii_nm, bandgaps, 'o-', linewidth=2, markersize=8, color='darkblue', label='Effective bandgap')
ax1.axhline(E_g_bulk, color='red', linestyle='--', linewidth=2, label=f'Bulk ({E_g_bulk} eV)')
ax1.axvline(a_B*1e9, color='gray', linestyle=':', linewidth=2, alpha=0.7, label=f'Bohr radius ({a_B*1e9:.1f} nm)')
ax1.set_xlabel('Quantum Dot Radius (nm)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Bandgap Energy (eV)', fontsize=12, fontweight='bold')
ax1.set_title('Bandgap vs Size\n(Quantum Confinement Effect)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, max(radii_nm)*1.1)

# Plot 2: Wavelength vs size (showing blue-shift)
ax2 = plt.subplot(2, 3, 2)
ax2.plot(radii_nm, wavelengths, 's-', linewidth=2, markersize=8, color='purple', label='Absorption edge')
ax2.axhline(lambda_bulk, color='red', linestyle='--', linewidth=2, label=f'Bulk ({lambda_bulk:.0f} nm)')

# Add color regions
ax2.axhspan(380, 450, alpha=0.1, color='violet', label='Violet')
ax2.axhspan(450, 495, alpha=0.1, color='blue')
ax2.axhspan(495, 570, alpha=0.1, color='green')
ax2.axhspan(570, 590, alpha=0.1, color='yellow')
ax2.axhspan(590, 620, alpha=0.1, color='orange')
ax2.axhspan(620, 750, alpha=0.1, color='red')

ax2.set_xlabel('Quantum Dot Radius (nm)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Absorption Wavelength (nm)', fontsize=12, fontweight='bold')
ax2.set_title('Absorption Edge vs Size\n(Blue-shift with Decreasing Size)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, max(radii_nm)*1.1)
ax2.invert_yaxis()  # Smaller wavelength (blue) at top

# Plot 3: Energy contributions
ax3 = plt.subplot(2, 3, 3)
ax3.plot(radii_nm, confinement_energies, 'o-', linewidth=2, markersize=6, 
         color='green', label='Confinement (+)')
ax3.plot(radii_nm, np.abs(coulomb_energies), 's-', linewidth=2, markersize=6, 
         color='orange', label='Coulomb (−)')
ax3.plot(radii_nm, np.array(confinement_energies) + np.array(coulomb_energies), 
         '^-', linewidth=2, markersize=6, color='blue', label='Net shift')
ax3.set_xlabel('Quantum Dot Radius (nm)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Energy Contribution (eV)', fontsize=12, fontweight='bold')
ax3.set_title('Energy Contributions to Bandgap\n(Brus Equation Terms)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Plot 4: Absorption spectra (simulated)
ax4 = plt.subplot(2, 3, 4)
wavelength_range = np.linspace(300, 800, 1000)

for i, (R_nm, E_g) in enumerate(zip([2.0, 3.0, 5.0, 10.0], 
                                      [bandgaps[radii_nm.tolist().index(r)] for r in [2.0, 3.0, 5.0, 10.0]])):
    lambda_edge = energy_to_wavelength(E_g)
    # Simulate absorption: sharp onset at bandgap edge
    absorption = 1 / (1 + np.exp(-(wavelength_range - lambda_edge) / 20))
    absorption = 1 - absorption  # Invert for absorption (high below edge)
    ax4.plot(wavelength_range, absorption + i*0.3, linewidth=2, 
             label=f'R = {R_nm} nm ({E_g:.2f} eV)')

ax4.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Absorption (a.u., offset)', fontsize=12, fontweight='bold')
ax4.set_title('Simulated Absorption Spectra\n(Blue-shift Visible)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(300, 800)

# Plot 5: Size quantization scaling
ax5 = plt.subplot(2, 3, 5)
# Confinement energy scales as 1/R²
R_theory = np.linspace(1, 20, 100) * 1e-9
E_theory = (hbar**2 * np.pi**2) / (2 * mu * R_theory**2) / eV

ax5.loglog(radii_nm, confinement_energies, 'o', markersize=10, 
           color='darkgreen', label='Calculated')
ax5.loglog(R_theory*1e9, E_theory, '--', linewidth=2, 
           color='lightgreen', label='E ∝ 1/R² (theory)')
ax5.set_xlabel('Quantum Dot Radius (nm)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Confinement Energy (eV)', fontsize=12, fontweight='bold')
ax5.set_title('Confinement Energy Scaling\n(E ∝ 1/R²)', fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, which='both')

# Plot 6: Color visualization
ax6 = plt.subplot(2, 3, 6)
# Show approximate emission colors
y_pos = np.arange(len(radii_nm))
colors_hex = []

for wl in wavelengths:
    _, color = wavelength_to_color(wl)
    colors_hex.append(color)

ax6.barh(y_pos, bandgaps, color=colors_hex, alpha=0.7, edgecolor='black', linewidth=1.5)
ax6.set_yticks(y_pos)
ax6.set_yticklabels([f'{r:.1f} nm' for r in radii_nm])
ax6.set_xlabel('Bandgap Energy (eV)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Quantum Dot Radius', fontsize=12, fontweight='bold')
ax6.set_title('Size-Tunable Emission Color\n(Approximate)', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

# Add wavelength labels
for i, (E, wl) in enumerate(zip(bandgaps, wavelengths)):
    color_name, _ = wavelength_to_color(wl)
    ax6.text(E + 0.05, i, f'{wl:.0f} nm\n{color_name}', 
             va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ewd/github/ewdhp/nanotech/basics/blue_shift_absorption.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved plot: blue_shift_absorption.png")
plt.close()

print("\n" + "="*70)
print("PHYSICAL INTERPRETATION:")
print("="*70)
print("""
1. QUANTUM CONFINEMENT:
   • When R < exciton Bohr radius, electrons/holes are squeezed
   • Heisenberg uncertainty: Δx↓ → Δp↑ → kinetic energy↑
   • Higher kinetic energy → larger effective bandgap

2. BLUE-SHIFT MECHANISM:
   • Smaller QD → larger bandgap → higher photon energy needed
   • E = hc/λ, so higher E → smaller λ (toward blue/UV)
   • This is why small QDs appear blue/violet, large ones red

3. SIZE-DEPENDENT COLOR:
   • CdSe QDs: 2 nm (blue) → 6 nm (red)
   • Tunability: precise color control by size
   • Applications: displays, LEDs, biological imaging

4. SCALING LAW:
   • Confinement energy ∝ 1/R²
   • Strong size dependence in quantum regime (R < a_B)
   • Bulk behavior recovered for R >> a_B
""")

print("="*70)
print("APPLICATIONS:")
print("="*70)
print("""
• Quantum Dot Displays (QLED TVs): size-tuned pure colors
• Biological Imaging: size-coded multi-color labels
• Solar Cells: multiple exciton generation
• LEDs: narrow emission, high efficiency
• Photodetectors: tunable spectral response
• Quantum Computing: defined energy levels for qubits
""")
print("="*70)
