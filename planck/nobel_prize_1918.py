"""
Max Planck: Nobel Prize in Physics 1918

Awarded "in recognition of the services he rendered to the advancement of
Physics by his discovery of energy quanta."

The Nobel Prize recognized Planck's revolutionary quantum hypothesis (1900),
which resolved the black-body radiation problem and founded quantum physics.

Key achievements leading to the prize:
- Derivation of Planck's radiation law
- Introduction of the quantum concept (E = hν)
- Foundation for all subsequent quantum mechanics

Demonstration:
- Reproduce the "ultraviolet catastrophe" problem.
- Show how Planck's quantum hypothesis solved it.
- Celebrate the impact on modern physics.
"""

import numpy as np
import matplotlib.pyplot as plt

H = 6.62607015e-34
C = 2.99792458e8
K_B = 1.380649e-23


def planck_law(wavelength_nm, T):
    """Planck's law in W/(m²·sr·m)."""
    lam = wavelength_nm * 1e-9
    return (2 * H * C**2) / (lam**5 * (np.exp((H * C) / (lam * K_B * T)) - 1))


def rayleigh_jeans(wavelength_nm, T):
    """Rayleigh-Jeans (classical) law."""
    lam = wavelength_nm * 1e-9
    return (2 * C * K_B * T) / lam**4


def main():
    print("=== Max Planck: Nobel Prize in Physics 1918 ===\n")
    print('Awarded "in recognition of the services he rendered to the advancement')
    print('of Physics by his discovery of energy quanta."\n')

    T = 5000  # K (typical temperature)
    wavelengths = np.linspace(100, 3000, 500)

    B_planck = planck_law(wavelengths, T)
    B_rj = rayleigh_jeans(wavelengths, T)

    # Plot the ultraviolet catastrophe
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(wavelengths, B_planck * 1e-13, 'b-', linewidth=3, label="Planck's Law (quantum)")
    ax.plot(wavelengths, B_rj * 1e-13, 'r--', linewidth=2, label="Rayleigh-Jeans (classical)")

    # Shade the UV region
    ax.axvspan(100, 400, alpha=0.2, color='purple', label='UV region (catastrophe)')

    ax.set_xlabel("Wavelength λ (nm)", fontsize=12)
    ax.set_ylabel("Spectral radiance B(λ,T) (×10¹³ W/(m²·sr·m))", fontsize=12)
    ax.set_title(f"The Ultraviolet Catastrophe and Planck's Solution\nT = {T} K", fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(100, 3000)
    ax.set_ylim(0, np.max(B_planck * 1e-13) * 1.2)

    # Add annotation
    ax.annotate(
        'Classical theory predicts\ninfinite energy at short λ\n("ultraviolet catastrophe")',
        xy=(200, B_rj[50] * 1e-13), xytext=(500, B_rj[50] * 1e-13 * 0.5),
        arrowprops=dict(arrowstyle='->', lw=2, color='red'),
        fontsize=10, color='red', weight='bold',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7)
    )

    ax.annotate(
        'Planck\'s quantum hypothesis\nresolves the catastrophe',
        xy=(250, B_planck[75] * 1e-13), xytext=(800, B_planck[75] * 1e-13 * 1.5),
        arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
        fontsize=10, color='blue', weight='bold',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7)
    )

    plt.tight_layout()
    plt.show()

    print("\nImpact of Planck's discovery:")
    print("  - Founded quantum physics (1900)")
    print("  - Enabled development of quantum mechanics (1920s)")
    print("  - Led to technologies: semiconductors, lasers, MRI, quantum computing")
    print("  - Changed our understanding of nature at the fundamental level\n")
    print("Planck's constant h is now one of the seven defining constants of the SI system (since 2019).\n")


if __name__ == "__main__":
    main()
