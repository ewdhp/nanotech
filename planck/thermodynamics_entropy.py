"""
Planck's Contributions to Thermodynamics: Entropy and Equilibrium

Planck made significant contributions to thermodynamic theory, particularly:

1. Clarifying the concept of entropy S and its statistical interpretation.
2. Deriving the entropy of black-body radiation.
3. Understanding irreversibility and the second law of thermodynamics.

Key formula (Boltzmann-Planck entropy):
    S = k_B ln(Ω)

where Ω is the number of microstates corresponding to a macrostate.

Demonstration:
- Calculate entropy for a simple two-state system.
- Show entropy of mixing (irreversibility).
- Plot entropy vs. temperature for an ideal gas.
"""

import numpy as np
import matplotlib.pyplot as plt

K_B = 1.380649e-23  # Boltzmann constant (J/K)
N_A = 6.02214076e23  # Avogadro's number


def boltzmann_entropy(omega):
    """S = k_B ln(Ω)"""
    return K_B * np.log(omega)


def entropy_ideal_gas(T, V, n_moles):
    """
    Sackur-Tetrode equation for entropy of an ideal monatomic gas.
    Simplified form: S ∝ n [ ln(V T^(3/2)) + const ]
    """
    # Using a simplified form for demonstration
    # Actual Sackur-Tetrode is more complex
    m = 4e-26  # mass of helium atom (kg)
    h = 6.626e-34
    const = (2 * np.pi * m * K_B / h**2)**(3/2)
    S = n_moles * N_A * K_B * (np.log(V * const * T**(3/2) / (n_moles * N_A)) + 5/2)
    return S


def main():
    print("=== Planck's Thermodynamics: Entropy and Irreversibility ===\n")

    # Example 1: Two-state system (e.g., spin up/down)
    N_particles = 10
    print(f"Example 1: System of {N_particles} particles, each with 2 states (spin up/down)")
    print(f"Total microstates Ω = 2^{N_particles} = {2**N_particles}")
    S_two_state = boltzmann_entropy(2**N_particles)
    print(f"Entropy S = k_B ln(Ω) = {S_two_state:.3e} J/K")
    print(f"         S/k_B = ln({2**N_particles}) = {np.log(2**N_particles):.2f}\n")

    # Example 2: Entropy of mixing (irreversibility)
    print("Example 2: Entropy of mixing two ideal gases")
    V_total = 2.0  # m³
    n_gas1 = 1.0   # moles
    n_gas2 = 1.0   # moles
    T = 300  # K

    # Initial entropy (gases separated)
    S_initial = entropy_ideal_gas(T, V_total/2, n_gas1) + entropy_ideal_gas(T, V_total/2, n_gas2)
    
    # Final entropy (gases mixed)
    S_final = entropy_ideal_gas(T, V_total, n_gas1 + n_gas2)
    
    Delta_S_mix = S_final - S_initial
    print(f"ΔS_mixing = {Delta_S_mix:.2f} J/K (always positive → irreversible!)\n")

    # Example 3: Entropy vs. temperature for ideal gas
    T_range = np.linspace(100, 1000, 100)
    V = 1.0  # m³
    n = 1.0  # moles
    S_range = [entropy_ideal_gas(T, V, n) for T in T_range]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Entropy vs. Temperature
    axes[0].plot(T_range, S_range, 'b-', linewidth=2)
    axes[0].set_xlabel("Temperature T (K)")
    axes[0].set_ylabel("Entropy S (J/K)")
    axes[0].set_title(f"Entropy of Ideal Gas (n = {n} mol, V = {V} m³)\nS increases with T (3rd law: S → 0 as T → 0)")
    axes[0].grid(True, alpha=0.3)

    # (b) Microstates illustration
    N_coins = np.arange(1, 21)
    Omega_coins = 2**N_coins
    S_coins = K_B * np.log(Omega_coins)

    axes[1].semilogy(N_coins, Omega_coins, 'ro-', label='Microstates Ω = 2^N')
    axes[1].set_xlabel("Number of particles N")
    axes[1].set_ylabel("Microstates Ω (log scale)")
    axes[1].set_title("Exponential Growth of Microstates\n(two-state system)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Key insights from Planck's thermodynamics:")
    print("  - Entropy S = k_B ln(Ω) connects microscopic states to macroscopic properties.")
    print("  - Irreversibility (2nd law): processes tend toward higher entropy.")
    print("  - Equilibrium is the state of maximum entropy for isolated systems.\n")


if __name__ == "__main__":
    main()
