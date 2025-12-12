#!/usr/bin/env python3
"""
Discrete Electronic States in Quantum Dots
===========================================

This script demonstrates:
1. Quantum confinement effects in nanoparticles
2. Size-dependent energy levels (discrete electronic states)
3. Blue-shift in absorption with decreasing particle size

Theory:
- Particle in a 3D spherical box (quantum dot)
- Energy levels: E_n = (n²π²ℏ²)/(2m*r²)
- Smaller radius → Higher energy → Blue-shift
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Physical constants
h_bar = 1.054571817e-34  # Reduced Planck constant (J·s)
e_charge = 1.602176634e-19  # Elementary charge (C)
m_e = 9.10938356e-31  # Electron mass (kg)

# Material parameters (CdSe quantum dot example)
m_eff_electron = 0.13 * m_e  # Effective mass of electron
m_eff_hole = 0.45 * m_e  # Effective mass of hole
E_gap_bulk = 1.74  # Bulk bandgap (eV) for CdSe


def quantum_confinement_energy(n, L, m_eff):
    """
    Calculate energy level for particle in a 1D box
    
    Parameters:
    -----------
    n : int
        Quantum number (1, 2, 3, ...)
    L : float
        Box dimension (meters)
    m_eff : float
        Effective mass (kg)
    
    Returns:
    --------
    float : Energy in Joules
    """
    return (n**2 * np.pi**2 * h_bar**2) / (2 * m_eff * L**2)


def quantum_dot_energy_levels(radius, n_max=5):
    """
    Calculate energy levels for quantum dot (spherical confinement)
    
    Parameters:
    -----------
    radius : float
        Quantum dot radius (meters)
    n_max : int
        Maximum quantum number to calculate
    
    Returns:
    --------
    dict : Energy levels for electrons and holes
    """
    # Diameter as effective box length
    L = 2 * radius
    
    # Calculate energy levels
    E_electron = np.array([quantum_confinement_energy(n, L, m_eff_electron) 
                           for n in range(1, n_max + 1)])
    E_hole = np.array([quantum_confinement_energy(n, L, m_eff_hole) 
                       for n in range(1, n_max + 1)])
    
    # Convert to eV
    E_electron_eV = E_electron / e_charge
    E_hole_eV = E_hole / e_charge
    
    return {
        'electron': E_electron_eV,
        'hole': E_hole_eV,
        'radius': radius
    }


def first_excitation_energy(radius):
    """
    Calculate first excitation energy (effective bandgap) for quantum dot
    
    E_eff = E_gap_bulk + E_confinement_electron + E_confinement_hole
    """
    L = 2 * radius
    E_e = quantum_confinement_energy(1, L, m_eff_electron) / e_charge
    E_h = quantum_confinement_energy(1, L, m_eff_hole) / e_charge
    
    return E_gap_bulk + E_e + E_h


def wavelength_from_energy(E_eV):
    """Convert energy (eV) to wavelength (nm)"""
    # E = hc/λ  →  λ = hc/E
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 2.99792458e8    # Speed of light (m/s)
    
    wavelength_m = (h * c) / (E_eV * e_charge)
    return wavelength_m * 1e9  # Convert to nm


def plot_energy_levels(radii_nm):
    """
    Plot discrete energy levels for different quantum dot sizes
    """
    fig, axes = plt.subplots(1, len(radii_nm), figsize=(15, 6), sharey=True)
    
    if len(radii_nm) == 1:
        axes = [axes]
    
    colors_e = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
    colors_h = plt.cm.Reds(np.linspace(0.4, 0.9, 5))
    
    for idx, r_nm in enumerate(radii_nm):
        ax = axes[idx]
        r_m = r_nm * 1e-9
        
        # Calculate energy levels
        levels = quantum_dot_energy_levels(r_m, n_max=5)
        
        # Plot bulk bandgap
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, label='Valence Band')
        ax.axhline(y=E_gap_bulk, color='gray', linestyle='--', linewidth=2, label='Conduction Band')
        
        # Plot electron levels (above conduction band)
        for i, E_e in enumerate(levels['electron']):
            y_pos = E_gap_bulk + E_e
            ax.plot([0.2, 0.8], [y_pos, y_pos], 'b-', linewidth=3, 
                   color=colors_e[i], label=f'e: n={i+1}' if idx == 0 else '')
            ax.text(0.85, y_pos, f'n={i+1}', va='center', fontsize=9, color='blue')
        
        # Plot hole levels (below valence band)
        for i, E_h in enumerate(levels['hole']):
            y_pos = -E_h
            ax.plot([0.2, 0.8], [y_pos, y_pos], 'r-', linewidth=3,
                   color=colors_h[i], label=f'h: n={i+1}' if idx == 0 else '')
            ax.text(0.85, y_pos, f'n={i+1}', va='center', fontsize=9, color='red')
        
        # Arrow showing first transition
        E_eff = first_excitation_energy(r_m)
        ax.annotate('', xy=(0.5, E_gap_bulk + levels['electron'][0]), 
                   xytext=(0.5, -levels['hole'][0]),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax.text(0.52, E_eff/2, f'{E_eff:.2f} eV', fontsize=10, color='green', weight='bold')
        
        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.5, 5)
        ax.set_xlabel(f'r = {r_nm} nm', fontsize=12, weight='bold')
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        
        if idx == 0:
            ax.set_ylabel('Energy (eV)', fontsize=12, weight='bold')
    
    axes[0].legend(loc='upper left', fontsize=8)
    plt.suptitle('Discrete Electronic States in Quantum Dots\n(Blue-shift with decreasing size)', 
                 fontsize=14, weight='bold')
    plt.tight_layout()
    return fig


def plot_absorption_spectra():
    """
    Plot absorption spectra showing blue-shift with decreasing size
    """
    radii_nm = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    radii_m = radii_nm * 1e-9
    
    # Calculate effective bandgaps
    E_eff = np.array([first_excitation_energy(r) for r in radii_m])
    wavelengths = wavelength_from_energy(E_eff)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Simulated absorption spectra
    wavelength_range = np.linspace(300, 700, 1000)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(radii_nm)))
    
    for i, (r_nm, lambda_peak, color) in enumerate(zip(radii_nm, wavelengths, colors)):
        # Gaussian-like absorption peak
        sigma = 20  # Peak width
        absorption = np.exp(-((wavelength_range - lambda_peak)**2) / (2 * sigma**2))
        
        ax1.plot(wavelength_range, absorption + i*0.3, color=color, 
                linewidth=2, label=f'{r_nm} nm')
        ax1.fill_between(wavelength_range, i*0.3, absorption + i*0.3, 
                         color=color, alpha=0.3)
    
    ax1.set_xlabel('Wavelength (nm)', fontsize=12, weight='bold')
    ax1.set_ylabel('Absorption (a.u., offset)', fontsize=12, weight='bold')
    ax1.set_title('Absorption Spectra: Blue-shift with Decreasing Size', 
                  fontsize=14, weight='bold')
    ax1.legend(title='QD Radius', loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(300, 700)
    
    # Add visible spectrum background
    for wl, color_bg in zip([380, 450, 495, 570, 590, 620, 750],
                            ['violet', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red']):
        if wl < 750:
            ax1.axvspan(wl, min(wl + 70, 750), alpha=0.1, color=color_bg)
    
    # Plot 2: Size vs Energy/Wavelength relationship
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(radii_nm, E_eff, 'bo-', linewidth=2, markersize=8, 
                     label='Effective Bandgap')
    ax2.axhline(y=E_gap_bulk, color='gray', linestyle='--', linewidth=2, 
                label='Bulk Bandgap')
    
    line2 = ax2_twin.plot(radii_nm, wavelengths, 'rs-', linewidth=2, markersize=8,
                          label='Absorption Peak')
    
    ax2.set_xlabel('Quantum Dot Radius (nm)', fontsize=12, weight='bold')
    ax2.set_ylabel('Energy (eV)', fontsize=12, weight='bold', color='b')
    ax2_twin.set_ylabel('Wavelength (nm)', fontsize=12, weight='bold', color='r')
    
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Size-Dependent Optical Properties (CdSe Quantum Dots)', 
                  fontsize=14, weight='bold')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', fontsize=10)
    
    # Add annotations
    ax2.annotate('Quantum\nConfinement', xy=(1.5, E_eff[1]), xytext=(1.0, 3.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', weight='bold')
    
    plt.tight_layout()
    return fig


def print_size_dependent_properties():
    """
    Print table of size-dependent properties
    """
    print("\n" + "="*80)
    print("QUANTUM CONFINEMENT: SIZE-DEPENDENT PROPERTIES OF CdSe QUANTUM DOTS")
    print("="*80)
    print(f"\nBulk CdSe Bandgap: {E_gap_bulk} eV")
    print(f"Bulk Absorption Edge: {wavelength_from_energy(E_gap_bulk):.1f} nm\n")
    
    radii_nm = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    print(f"{'Radius (nm)':<12} {'Diameter (nm)':<14} {'E_eff (eV)':<12} "
          f"{'λ_abs (nm)':<12} {'Color':<15} {'Confinement (eV)':<18}")
    print("-" * 95)
    
    for r_nm in radii_nm:
        r_m = r_nm * 1e-9
        d_nm = 2 * r_nm
        E_eff = first_excitation_energy(r_m)
        wavelength = wavelength_from_energy(E_eff)
        confinement = E_eff - E_gap_bulk
        
        # Determine color
        if wavelength < 450:
            color = "Violet/Blue"
        elif wavelength < 495:
            color = "Blue"
        elif wavelength < 570:
            color = "Green"
        elif wavelength < 590:
            color = "Yellow"
        elif wavelength < 620:
            color = "Orange"
        else:
            color = "Red"
        
        print(f"{r_nm:<12.1f} {d_nm:<14.1f} {E_eff:<12.3f} "
              f"{wavelength:<12.1f} {color:<15} {confinement:<18.3f}")
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS:")
    print("="*80)
    print("1. SMALLER quantum dots → HIGHER effective bandgap (Blue-shift)")
    print("2. Confinement energy ∝ 1/r² (inverse square relationship)")
    print("3. Color changes from RED (large) → BLUE (small)")
    print("4. Discrete electronic states (quantized energy levels)")
    print("5. Tunable optical properties by controlling particle size")
    print("="*80 + "\n")


def main():
    """
    Main function to demonstrate discrete electronic states
    """
    print("\n🔬 DISCRETE ELECTRONIC STATES IN QUANTUM DOTS 🔬\n")
    
    # Print theoretical background
    print("THEORY:")
    print("-------")
    print("• Quantum confinement occurs when particle size ~ de Broglie wavelength")
    print("• Energy levels become discrete (quantized) rather than continuous")
    print("• Energy increases as size decreases: E ∝ 1/r²")
    print("• This causes a blue-shift in absorption spectra\n")
    
    # Print numerical results
    print_size_dependent_properties()
    
    # Generate plots
    print("Generating visualizations...")
    
    # Plot 1: Discrete energy levels for different sizes
    fig1 = plot_energy_levels([1.5, 2.5, 4.0])
    plt.savefig('discrete_energy_levels.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: discrete_energy_levels.png")
    
    # Plot 2: Absorption spectra and size dependence
    fig2 = plot_absorption_spectra()
    plt.savefig('absorption_blue_shift.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: absorption_blue_shift.png")
    
    plt.show()
    
    print("\n✨ Simulation complete! ✨\n")


if __name__ == "__main__":
    main()
