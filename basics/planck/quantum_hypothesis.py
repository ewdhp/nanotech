"""
Planck's Quantum Hypothesis (1900)

Classical physics assumed energy could be absorbed or emitted continuously
in any amount. Planck revolutionized this by proposing that electromagnetic
energy is quantized:

    E = n * h * ν

where:
- n is an integer (0, 1, 2, ...)
- h is Planck's constant ≈ 6.626 × 10⁻³⁴ J·s
- ν is the frequency of radiation

This discrete, "quantum" nature of energy was a radical departure and
marked the birth of quantum physics.

Demonstration:
- Show energy levels for different frequencies.
- Compare continuous vs. quantized energy absorption.
- Visualize the "quantum jump" concept.
"""

import numpy as np
import matplotlib.pyplot as plt

H = 6.62607015e-34  # Planck's constant (J·s)

def quantum_energy_levels(frequency, n_max=10):
    """
    Compute quantized energy levels E_n = n * h * ν for n = 0, 1, ..., n_max.
    """
    n = np.arange(0, n_max + 1)
    E = n * H * frequency
    return n, E


def main():
    # Frequencies corresponding to visible light
    freq_red = 4.3e14    # Hz (red light)
    freq_green = 5.5e14  # Hz (green light)
    freq_violet = 7.5e14 # Hz (violet light)

    n_max = 15

    # Compute energy levels
    n_red, E_red = quantum_energy_levels(freq_red, n_max)
    n_green, E_green = quantum_energy_levels(freq_green, n_max)
    n_violet, E_violet = quantum_energy_levels(freq_violet, n_max)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Energy level diagrams for different frequencies
    axes[0].stem(n_red, E_red * 1e19, linefmt='r-', markerfmt='ro', basefmt=" ", label=f'Red (ν = {freq_red:.1e} Hz)')
    axes[0].stem(n_green, E_green * 1e19, linefmt='g-', markerfmt='go', basefmt=" ", label=f'Green (ν = {freq_green:.1e} Hz)')
    axes[0].stem(n_violet, E_violet * 1e19, linefmt='b-', markerfmt='bo', basefmt=" ", label=f'Violet (ν = {freq_violet:.1e} Hz)')
    axes[0].set_xlabel("Quantum number n")
    axes[0].set_ylabel("Energy E (×10⁻¹⁹ J)")
    axes[0].set_title("Quantized Energy Levels: E = n h ν\n(Planck's Quantum Hypothesis)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (b) Continuous vs. quantized comparison
    freq_demo = 5e14  # Hz
    n_demo, E_demo = quantum_energy_levels(freq_demo, n_max=20)
    
    # Classical (continuous) prediction
    E_continuous = np.linspace(0, np.max(E_demo), 200)
    
    axes[1].plot(E_continuous * 1e19, np.ones_like(E_continuous), 'k-', linewidth=3, alpha=0.3, label='Classical (continuous)')
    axes[1].stem(E_demo * 1e19, np.ones_like(E_demo), linefmt='r-', markerfmt='ro', basefmt=" ", label='Quantum (discrete)')
    axes[1].set_xlabel("Energy (×10⁻¹⁹ J)")
    axes[1].set_yticks([])
    axes[1].set_title(f"Classical vs. Quantum Energy for ν = {freq_demo:.1e} Hz")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()

    print("=== Planck's Quantum Hypothesis ===")
    print("Energy is quantized: E = n h ν")
    print(f"Planck's constant h = {H:.3e} J·s\n")
    print("For red light (ν ≈ 4.3×10¹⁴ Hz):")
    print(f"  E₁ (n=1) = {E_red[1]:.3e} J = {E_red[1] * 1e19:.3f} × 10⁻¹⁹ J")
    print(f"  E₂ (n=2) = {E_red[2]:.3e} J = {E_red[2] * 1e19:.3f} × 10⁻¹⁹ J\n")
    print("Key insight: Energy can only be absorbed/emitted in discrete packets (quanta),")
    print("not continuously as classical physics assumed. This was revolutionary!\n")


if __name__ == "__main__":
    main()
