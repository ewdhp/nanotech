"""
Planck's Constant h

Planck's constant is a fundamental physical constant that relates the energy
of a photon to its frequency:

    E = h ν

Value: h = 6.62607015 × 10⁻³⁴ J·s (exact, by definition since 2019)

It appears in:
- Quantum mechanics (Schrödinger equation, uncertainty principle)
- Photoelectric effect
- Compton scattering
- De Broglie wavelength

Demonstration:
- Calculate photon energies for different frequencies (radio to gamma rays).
- Show the photoelectric effect energy relationship.
- Plot energy vs. frequency (linear with slope h).
"""

import numpy as np
import matplotlib.pyplot as plt

H = 6.62607015e-34  # J·s
C = 2.99792458e8    # m/s
EV = 1.602176634e-19  # J per eV


def photon_energy(frequency):
    """E = h ν in Joules."""
    return H * frequency


def photon_energy_from_wavelength(wavelength_m):
    """E = h c / λ in Joules."""
    return H * C / wavelength_m


def main():
    # Electromagnetic spectrum frequencies
    spectrum = {
        "Radio": 1e6,           # 1 MHz
        "Microwave": 1e10,      # 10 GHz
        "Infrared": 1e13,       # ~30 THz
        "Visible (red)": 4.3e14,
        "Visible (violet)": 7.5e14,
        "UV": 1e15,
        "X-ray": 1e18,
        "Gamma-ray": 1e20,
    }

    print("=== Planck's Constant and Photon Energies ===")
    print(f"h = {H:.3e} J·s\n")
    print(f"{'Type':<20} {'Frequency (Hz)':<15} {'Energy (J)':<15} {'Energy (eV)':<15}")
    print("-" * 70)

    freqs = []
    energies_J = []
    names = []

    for name, freq in spectrum.items():
        E_J = photon_energy(freq)
        E_eV = E_J / EV
        print(f"{name:<20} {freq:<15.2e} {E_J:<15.3e} {E_eV:<15.3e}")
        freqs.append(freq)
        energies_J.append(E_J)
        names.append(name)

    print()

    # Plot E vs. ν
    freqs_plot = np.logspace(6, 20, 100)
    energies_plot = photon_energy(freqs_plot)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) E vs ν (log-log)
    axes[0].loglog(freqs_plot, energies_plot, 'b-', linewidth=2, label='E = h ν')
    axes[0].scatter(freqs, energies_J, c='red', s=100, zorder=5, edgecolors='k', label='EM spectrum points')
    for i, name in enumerate(names):
        axes[0].annotate(name, (freqs[i], energies_J[i]), fontsize=8, ha='right', alpha=0.7)
    axes[0].set_xlabel("Frequency ν (Hz)")
    axes[0].set_ylabel("Energy E (J)")
    axes[0].set_title("Photon Energy vs. Frequency\nE = h ν (slope = h)")
    axes[0].legend()
    axes[0].grid(True, which='both', alpha=0.3)

    # (b) Photoelectric effect illustration
    # Work function for a typical metal (e.g., sodium)
    work_function_eV = 2.28  # eV
    work_function_J = work_function_eV * EV

    freq_threshold = work_function_J / H
    freqs_photo = np.linspace(0, 2 * freq_threshold, 200)
    KE_max = np.maximum(photon_energy(freqs_photo) - work_function_J, 0)

    axes[1].plot(freqs_photo * 1e-14, KE_max / EV, 'g-', linewidth=2)
    axes[1].axvline(freq_threshold * 1e-14, color='r', linestyle='--', label=f'Threshold ν₀ = {freq_threshold:.2e} Hz')
    axes[1].axhline(0, color='k', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel("Frequency ν (×10¹⁴ Hz)")
    axes[1].set_ylabel("Max kinetic energy of ejected electrons (eV)")
    axes[1].set_title(f"Photoelectric Effect\nKE_max = h ν - Φ (Φ = {work_function_eV:.2f} eV)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Key insight:")
    print("Planck's constant h connects the wave (frequency ν) and particle (energy E)")
    print("properties of light. It is central to quantum mechanics.\n")


if __name__ == "__main__":
    main()
