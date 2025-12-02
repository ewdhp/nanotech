"""
Planck Units – Natural Units of Measurement

Planck introduced natural units based on fundamental constants:
- Planck length:  l_P = √(ℏ G / c³) ≈ 1.616 × 10⁻³⁵ m
- Planck mass:    m_P = √(ℏ c / G)  ≈ 2.176 × 10⁻⁸ kg
- Planck time:    t_P = l_P / c     ≈ 5.391 × 10⁻⁴⁴ s
- Planck energy:  E_P = m_P c²      ≈ 1.956 × 10⁹ J
- Planck temperature: T_P = E_P / k ≈ 1.417 × 10³² K

These units emerge when setting ℏ = c = G = k_B = 1 and represent
scales where quantum gravity effects become important.

Demonstration:
- Calculate all Planck units.
- Compare them to everyday and atomic scales.
- Visualize the hierarchy of scales.
"""

import numpy as np
import matplotlib.pyplot as plt

# Fundamental constants
H_BAR = 1.054571817e-34  # ℏ (J·s)
C = 2.99792458e8         # speed of light (m/s)
G = 6.67430e-11          # gravitational constant (m³/(kg·s²))
K_B = 1.380649e-23       # Boltzmann constant (J/K)

# Planck units
L_PLANCK = np.sqrt(H_BAR * G / C**3)
M_PLANCK = np.sqrt(H_BAR * C / G)
T_PLANCK = L_PLANCK / C
E_PLANCK = M_PLANCK * C**2
TEMP_PLANCK = E_PLANCK / K_B


def main():
    print("=== Planck Units (Natural Units) ===\n")
    print(f"Planck length   l_P = √(ℏ G / c³) = {L_PLANCK:.3e} m")
    print(f"Planck mass     m_P = √(ℏ c / G)  = {M_PLANCK:.3e} kg")
    print(f"Planck time     t_P = l_P / c     = {T_PLANCK:.3e} s")
    print(f"Planck energy   E_P = m_P c²      = {E_PLANCK:.3e} J = {E_PLANCK / 1.602e-19:.3e} eV")
    print(f"Planck temp.    T_P = E_P / k_B   = {TEMP_PLANCK:.3e} K\n")

    # Comparisons
    print("Comparisons to familiar scales:")
    print(f"  Planck length / proton radius ≈ {L_PLANCK / 1e-15:.2e}")
    print(f"  Planck mass / electron mass   ≈ {M_PLANCK / 9.109e-31:.2e}")
    print(f"  Planck time / atomic time     ≈ {T_PLANCK / 1e-18:.2e}")
    print(f"  Planck energy / rest energy of proton ≈ {E_PLANCK / (1.673e-27 * C**2):.2e}\n")

    # Visualization: hierarchy of scales
    scales_length = {
        "Planck length": L_PLANCK,
        "Proton radius": 1e-15,
        "Atom (Bohr radius)": 5.29e-11,
        "Virus": 1e-7,
        "Human hair": 1e-4,
        "Human": 2,
        "Earth radius": 6.371e6,
        "Observable universe": 8.8e26,
    }

    scales_mass = {
        "Electron": 9.109e-31,
        "Proton": 1.673e-27,
        "Planck mass": M_PLANCK,
        "Dust particle": 1e-9,
        "Grain of sand": 1e-6,
        "Human": 70,
        "Earth": 5.972e24,
        "Sun": 1.989e30,
    }

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # (a) Length scales
    names_l = list(scales_length.keys())
    values_l = [scales_length[n] for n in names_l]
    colors_l = ['red' if 'Planck' in n else 'blue' for n in names_l]

    axes[0].barh(names_l, np.log10(values_l), color=colors_l, edgecolor='black')
    axes[0].set_xlabel("log₁₀(Length / m)")
    axes[0].set_title("Hierarchy of Length Scales (Planck length highlighted)")
    axes[0].grid(True, alpha=0.3, axis='x')

    # (b) Mass scales
    names_m = list(scales_mass.keys())
    values_m = [scales_mass[n] for n in names_m]
    colors_m = ['red' if 'Planck' in n else 'green' for n in names_m]

    axes[1].barh(names_m, np.log10(values_m), color=colors_m, edgecolor='black')
    axes[1].set_xlabel("log₁₀(Mass / kg)")
    axes[1].set_title("Hierarchy of Mass Scales (Planck mass highlighted)")
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()

    print("Planck units represent the scale where quantum mechanics and general relativity")
    print("both become essential. They are 'natural' units where ℏ = c = G = k_B = 1.\n")


if __name__ == "__main__":
    main()
