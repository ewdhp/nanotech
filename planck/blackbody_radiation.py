"""
Planck's Law for Black-Body Radiation (1900)

Classical theories (Rayleigh-Jeans law, Wien's law) failed to correctly
describe the spectrum of thermal radiation from a black body.

Planck's law resolved this:

    B(λ, T) = (2 h c²) / λ⁵  ×  1 / (exp(hc / λkT) - 1)

where:
- λ is wavelength
- T is temperature (K)
- h is Planck's constant
- c is speed of light
- k is Boltzmann's constant

This derivation required the quantum hypothesis (E = hν) and solved the
"ultraviolet catastrophe" — classical theory predicted infinite energy
at short wavelengths, which Planck's law correctly avoided.

Demonstration:
- Plot Planck's law for different temperatures.
- Compare with Rayleigh-Jeans (classical) prediction.
- Show Wien's displacement law.
"""

import numpy as np
import matplotlib.pyplot as plt

H = 6.62607015e-34   # J·s
C = 2.99792458e8     # m/s
K_B = 1.380649e-23   # J/K


def planck_law(wavelength, T):
    """
    Planck's law: spectral radiance B(λ, T).
    Returns in units of W/(m² sr m).
    """
    lambda_m = wavelength * 1e-9  # convert nm to m
    numerator = 2 * H * C**2
    denominator = lambda_m**5 * (np.exp((H * C) / (lambda_m * K_B * T)) - 1)
    return numerator / denominator


def rayleigh_jeans(wavelength, T):
    """
    Rayleigh-Jeans law (classical approximation):
    B(λ, T) ≈ (2 c k T) / λ⁴
    Valid only for long wavelengths.
    """
    lambda_m = wavelength * 1e-9
    return (2 * C * K_B * T) / lambda_m**4


def wien_displacement(T):
    """
    Wien's displacement law: λ_max T = b
    where b ≈ 2.898 × 10⁻³ m·K
    """
    b = 2.897771955e-3  # m·K
    return (b / T) * 1e9  # return in nm


def main():
    # Wavelength range (nm)
    wavelengths = np.linspace(100, 3000, 500)

    # Temperatures (K)
    T1 = 3000  # incandescent bulb
    T2 = 5800  # Sun's surface
    T3 = 8000  # hot star

    # Planck's law
    B1 = planck_law(wavelengths, T1)
    B2 = planck_law(wavelengths, T2)
    B3 = planck_law(wavelengths, T3)

    # Rayleigh-Jeans (for comparison at T = 5800 K)
    B_RJ = rayleigh_jeans(wavelengths, T2)

    # Wien's displacement
    lambda_max_1 = wien_displacement(T1)
    lambda_max_2 = wien_displacement(T2)
    lambda_max_3 = wien_displacement(T3)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Planck's law for different temperatures
    axes[0].plot(wavelengths, B1 * 1e-13, 'r-', label=f'T = {T1} K')
    axes[0].plot(wavelengths, B2 * 1e-13, 'orange', label=f'T = {T2} K (Sun)')
    axes[0].plot(wavelengths, B3 * 1e-13, 'b-', label=f'T = {T3} K')
    axes[0].axvline(lambda_max_1, color='r', linestyle='--', alpha=0.5, label=f'λ_max = {lambda_max_1:.0f} nm')
    axes[0].axvline(lambda_max_2, color='orange', linestyle='--', alpha=0.5, label=f'λ_max = {lambda_max_2:.0f} nm')
    axes[0].axvline(lambda_max_3, color='b', linestyle='--', alpha=0.5, label=f'λ_max = {lambda_max_3:.0f} nm')
    axes[0].set_xlabel("Wavelength λ (nm)")
    axes[0].set_ylabel("Spectral radiance B(λ,T) (×10¹³ W/(m²·sr·m))")
    axes[0].set_title("Planck's Law: Black-Body Radiation Spectrum")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(100, 3000)

    # (b) Planck vs. Rayleigh-Jeans (ultraviolet catastrophe)
    axes[1].plot(wavelengths, B2 * 1e-13, 'orange', linewidth=2, label=f'Planck (T = {T2} K)')
    axes[1].plot(wavelengths, B_RJ * 1e-13, 'k--', linewidth=2, label='Rayleigh-Jeans (classical)')
    axes[1].set_xlabel("Wavelength λ (nm)")
    axes[1].set_ylabel("Spectral radiance (×10¹³ W/(m²·sr·m))")
    axes[1].set_title("Planck's Law vs. Classical Rayleigh-Jeans\n(Ultraviolet Catastrophe Resolved)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(100, 1500)
    axes[1].set_ylim(0, np.max(B2[wavelengths < 1500]) * 1e-13 * 1.2)

    plt.tight_layout()
    plt.show()

    print("=== Planck's Black-Body Radiation Law ===")
    print("B(λ, T) = (2 h c²) / λ⁵  ×  1 / (exp(hc / λkT) - 1)\n")
    print(f"Sun's surface temperature T ≈ {T2} K:")
    print(f"  Peak wavelength λ_max = {lambda_max_2:.0f} nm (Wien's displacement law)")
    print(f"  This is in the visible spectrum (green-yellow).\n")
    print("Classical Rayleigh-Jeans law predicts infinite energy at short wavelengths")
    print("(ultraviolet catastrophe). Planck's quantum hypothesis resolved this.\n")


if __name__ == "__main__":
    main()
