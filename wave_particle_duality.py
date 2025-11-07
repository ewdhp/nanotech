"""
Wave-Particle Duality Demonstration
====================================
Demonstrates De Broglie wavelength calculations and visualizations
for photons and matter waves.

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

# Physical constants
h = 6.626e-34  # Planck constant (J·s)
c = 3.0e8      # Speed of light (m/s)
m_e = 9.109e-31  # Electron mass (kg)
m_p = 1.673e-27  # Proton mass (kg)


def de_broglie_wavelength(momentum):
    """
    Calculate De Broglie wavelength: λ = h/p
    
    Parameters:
    -----------
    momentum : float
        Momentum in kg·m/s
    
    Returns:
    --------
    float : De Broglie wavelength in meters
    """
    return h / momentum


def photon_wavelength(energy):
    """
    Calculate photon wavelength from energy: λ = hc/E
    
    Parameters:
    -----------
    energy : float
        Photon energy in Joules
    
    Returns:
    --------
    float : Wavelength in meters
    """
    return (h * c) / energy


def matter_wave_velocity(mass, wavelength):
    """
    Calculate velocity from De Broglie wavelength
    
    Parameters:
    -----------
    mass : float
        Particle mass in kg
    wavelength : float
        De Broglie wavelength in meters
    
    Returns:
    --------
    float : Velocity in m/s
    """
    return h / (mass * wavelength)


def wave_function(x, wavelength, amplitude=1.0, phase=0):
    """
    Generate a matter wave function
    
    Parameters:
    -----------
    x : array
        Position array
    wavelength : float
        De Broglie wavelength
    amplitude : float
        Wave amplitude
    phase : float
        Phase offset
    
    Returns:
    --------
    array : Wave function values
    """
    k = 2 * np.pi / wavelength  # Wave number
    return amplitude * np.sin(k * x + phase)


# ============================================================================
# CALCULATIONS
# ============================================================================

print("=" * 70)
print("WAVE-PARTICLE DUALITY DEMONSTRATION")
print("=" * 70)
print()

# 1. Photon Examples
print("1. PHOTON EXAMPLES (Electromagnetic Radiation)")
print("-" * 70)

photon_energies = {
    "Radio wave (1 MHz)": 6.626e-28,  # J
    "Microwave (10 GHz)": 6.626e-24,
    "Visible light (green)": 3.5e-19,
    "UV light": 1.0e-18,
    "X-ray": 1.6e-15,
    "Gamma ray": 1.6e-13
}

for name, energy in photon_energies.items():
    wavelength = photon_wavelength(energy)
    frequency = energy / h
    momentum = energy / c
    
    # Format wavelength for readability
    if wavelength >= 1:
        wl_str = f"{wavelength:.2e} m"
    elif wavelength >= 1e-3:
        wl_str = f"{wavelength*1e3:.2f} mm"
    elif wavelength >= 1e-6:
        wl_str = f"{wavelength*1e6:.2f} μm"
    elif wavelength >= 1e-9:
        wl_str = f"{wavelength*1e9:.2f} nm"
    else:
        wl_str = f"{wavelength*1e12:.4f} pm"
    
    print(f"\n{name}:")
    print(f"  Energy:     {energy:.2e} J")
    print(f"  Frequency:  {frequency:.2e} Hz")
    print(f"  Wavelength: {wl_str}")
    print(f"  Momentum:   {momentum:.2e} kg·m/s")

print()

# 2. Matter Wave Examples
print("2. MATTER WAVE EXAMPLES (Particles)")
print("-" * 70)

# Electron at different velocities
print("\nElectron:")
electron_velocities = [1e5, 1e6, 1e7]  # m/s

for v in electron_velocities:
    momentum = m_e * v
    wavelength = de_broglie_wavelength(momentum)
    energy_kinetic = 0.5 * m_e * v**2
    
    print(f"\n  Velocity: {v:.2e} m/s ({v/c*100:.4f}% of c)")
    print(f"  Momentum: {momentum:.2e} kg·m/s")
    print(f"  De Broglie wavelength: {wavelength*1e9:.4f} nm ({wavelength*1e10:.4f} Å)")
    print(f"  Kinetic energy: {energy_kinetic:.2e} J ({energy_kinetic/1.602e-19:.2f} eV)")

# Proton
print("\n\nProton:")
proton_velocity = 1e6  # m/s
momentum = m_p * proton_velocity
wavelength = de_broglie_wavelength(momentum)

print(f"  Velocity: {proton_velocity:.2e} m/s")
print(f"  De Broglie wavelength: {wavelength*1e12:.4f} pm ({wavelength*1e10:.4f} Å)")

# Macroscopic object (baseball)
print("\n\nBaseball (0.145 kg at 40 m/s):")
baseball_mass = 0.145  # kg
baseball_velocity = 40  # m/s
momentum = baseball_mass * baseball_velocity
wavelength = de_broglie_wavelength(momentum)

print(f"  Velocity: {baseball_velocity} m/s")
print(f"  De Broglie wavelength: {wavelength:.2e} m")
print(f"  (Extremely small - quantum effects negligible!)")

print()
print("=" * 70)


# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# Color scheme
colors = {
    'electron': '#2E86AB',
    'photon': '#A23B72',
    'wave': '#F18F01',
    'particle': '#C73E1D'
}

# Plot 1: De Broglie wavelength vs velocity for different particles
ax1 = fig.add_subplot(gs[0, :])
velocities = np.logspace(3, 8, 100)  # 1e3 to 1e8 m/s

wavelength_electron = h / (m_e * velocities)
wavelength_proton = h / (m_p * velocities)

ax1.loglog(velocities, wavelength_electron * 1e9, 
          label='Electron', linewidth=2, color=colors['electron'])
ax1.loglog(velocities, wavelength_proton * 1e9, 
          label='Proton', linewidth=2, color=colors['photon'])
ax1.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='Atomic size (~0.1 nm)')
ax1.axhline(0.154, color='gray', linestyle=':', alpha=0.5, label='X-ray (Cu Kα)')
ax1.set_xlabel('Velocity (m/s)', fontsize=12)
ax1.set_ylabel('De Broglie Wavelength (nm)', fontsize=12)
ax1.set_title('De Broglie Wavelength vs Particle Velocity', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')

# Plot 2: Electron matter wave at different energies
ax2 = fig.add_subplot(gs[1, 0])
x = np.linspace(0, 20e-9, 1000)  # 20 nm range

energies_eV = [10, 50, 100]  # eV
for E_eV in energies_eV:
    E_J = E_eV * 1.602e-19  # Convert to Joules
    v = np.sqrt(2 * E_J / m_e)
    p = m_e * v
    wavelength = de_broglie_wavelength(p)
    
    psi = wave_function(x, wavelength)
    ax2.plot(x * 1e9, psi, label=f'{E_eV} eV (λ={wavelength*1e10:.2f} Å)', linewidth=2)

ax2.set_xlabel('Position (nm)', fontsize=11)
ax2.set_ylabel('Wave Function ψ(x)', fontsize=11)
ax2.set_title('Electron Matter Waves at Different Energies', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', linewidth=0.5)

# Plot 3: Photon wavelengths across electromagnetic spectrum
ax3 = fig.add_subplot(gs[1, 1])
em_spectrum = [
    ('Radio', 1e3, 1e0),
    ('Microwave', 1e0, 1e-3),
    ('Infrared', 1e-3, 7e-7),
    ('Visible', 7e-7, 4e-7),
    ('UV', 4e-7, 1e-8),
    ('X-ray', 1e-8, 1e-11),
    ('Gamma', 1e-11, 1e-14)
]

y_pos = np.arange(len(em_spectrum))
wavelengths_max = [item[1] for item in em_spectrum]
wavelengths_min = [item[2] for item in em_spectrum]
labels = [item[0] for item in em_spectrum]

# Create horizontal bar chart
for i, (label, wl_max, wl_min) in enumerate(em_spectrum):
    color = plt.cm.viridis(i / len(em_spectrum))
    ax3.barh(i, np.log10(wl_max) - np.log10(wl_min), 
            left=np.log10(wl_min), height=0.8, color=color, alpha=0.7)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(labels, fontsize=10)
ax3.set_xlabel('log₁₀(Wavelength in meters)', fontsize=11)
ax3.set_title('Electromagnetic Spectrum', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Probability density for electron
ax4 = fig.add_subplot(gs[2, 0])
x = np.linspace(0, 10e-9, 1000)
wavelength = 1e-9  # 1 nm
psi = wave_function(x, wavelength)
probability_density = psi**2

ax4.fill_between(x * 1e9, probability_density, alpha=0.6, color=colors['electron'])
ax4.plot(x * 1e9, psi, 'r--', alpha=0.5, linewidth=1.5, label='Wave function ψ(x)')
ax4.set_xlabel('Position (nm)', fontsize=11)
ax4.set_ylabel('Amplitude', fontsize=11)
ax4.set_title('Wave Function and Probability Density |ψ(x)|²', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.axhline(0, color='black', linewidth=0.5)

# Plot 5: Energy-wavelength relationship
ax5 = fig.add_subplot(gs[2, 1])
energies_eV = np.logspace(-2, 6, 100)  # 0.01 eV to 1 MeV
energies_J = energies_eV * 1.602e-19

# Photon wavelength
wavelengths_photon = (h * c) / energies_J

# Electron De Broglie wavelength (non-relativistic)
velocities = np.sqrt(2 * energies_J / m_e)
wavelengths_electron = h / (m_e * velocities)

ax5.loglog(energies_eV, wavelengths_photon * 1e9, 
          label='Photon', linewidth=2.5, color=colors['photon'])
ax5.loglog(energies_eV, wavelengths_electron * 1e9, 
          label='Electron (matter wave)', linewidth=2.5, color=colors['electron'])
ax5.set_xlabel('Energy (eV)', fontsize=11)
ax5.set_ylabel('Wavelength (nm)', fontsize=11)
ax5.set_title('Energy vs Wavelength: Photons and Electrons', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, which='both')

plt.suptitle('Wave-Particle Duality: De Broglie Wavelength Analysis', 
            fontsize=16, fontweight='bold', y=0.995)

# Display the plots
print("\nDisplaying interactive plots...")
print("Close the plot window to continue.\n")
plt.show(block=True)

print("\n" + "=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)
print("• De Broglie wavelength λ = h/p relates wave and particle properties")
print("• Lighter particles (electrons) have longer wavelengths at same velocity")
print("• Quantum effects become significant when λ ≈ size of system")
print("• Macroscopic objects have negligible De Broglie wavelengths")
print("• Both photons and matter exhibit wave-particle duality")
print("=" * 70)
