"""
Thermodynamic Entropy Theory Demonstration

This script illustrates key concepts of thermodynamic entropy:
1. Non-conserved state function
2. Spontaneous processes and irreversibility
3. Entropy never decreases in isolated systems
4. Arrow of time and perpetual motion impossibility
5. Energy waste and work limitations

References:
- Second Law of Thermodynamics
- Clausius Inequality
- Boltzmann Entropy Formula

Key Concepts Covered:

Non-conserved state function - Unlike energy, entropy is not conserved
Spontaneous processes - Systems evolve toward higher entropy
Second Law - Entropy never decreases in isolated systems (ΔS ≥ 0)
Irreversibility & Arrow of Time - Time's direction defined by entropy increase
Perpetual motion impossibility - Efficiency always < 100%
Waste heat - Energy degradation limits work output
Features:

8 detailed visualizations showing entropy behavior
Quantitative examples (isothermal expansion, heat transfer, Carnot cycle, etc.)
Boltzmann entropy formula (S = k_B·ln(W))
Clausius inequality demonstrations
Real-world examples of irreversible processes
The script can be run directly to see both numerical calculations and graphical 
demonstrations of how entropy governs thermodynamic processes and establishes 
Sthe arrow of time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import boltzmann
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)


class ThermodynamicSystem:
    """Model for demonstrating entropy behavior in thermodynamic systems"""
    
    def __init__(self, temperature, num_particles=1000):
        self.T = temperature
        self.N = num_particles
        self.k_B = k_B
        
    def clausius_entropy_change(self, Q_rev, T):
        """
        Clausius definition of entropy change
        dS = dQ_rev / T
        
        For reversible heat transfer
        """
        return Q_rev / T
    
    def boltzmann_entropy(self, W):
        """
        Boltzmann's entropy formula
        S = k_B * ln(W)
        
        W: number of microstates
        """
        return self.k_B * np.log(W)
    
    def entropy_production(self, Q, T_hot, T_cold):
        """
        Calculate entropy production for irreversible heat transfer
        
        ΔS_total = ΔS_system + ΔS_surroundings
        For irreversible process: ΔS_total > 0
        """
        # Entropy change of hot reservoir
        dS_hot = -Q / T_hot
        # Entropy change of cold reservoir
        dS_cold = Q / T_cold
        # Total entropy change (always positive for spontaneous process)
        dS_total = dS_hot + dS_cold
        
        return dS_total, dS_hot, dS_cold


def demonstrate_entropy_increase():
    """
    Demonstrate that entropy never decreases in isolated systems
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Thermodynamic Entropy: Non-Conserved State Function', 
                 fontsize=16, fontweight='bold')
    
    # 1. Free Expansion - Irreversible Process
    ax1 = axes[0, 0]
    V_initial = np.linspace(1, 2, 100)
    V_final = np.linspace(2, 4, 100)
    n = 1  # mole
    R = 8.314  # J/(mol·K)
    
    # Entropy change for free expansion (irreversible)
    S_initial = np.zeros_like(V_initial)
    S_change = R * np.log(V_final / V_initial[0])
    
    ax1.plot(V_initial, S_initial, 'b-', linewidth=2, label='Initial State')
    ax1.axhline(y=R * np.log(2), color='r', linestyle='--', 
                linewidth=2, label=f'Final State (ΔS = R·ln(2) > 0)')
    ax1.fill_between(V_initial, 0, R * np.log(2), alpha=0.3, color='red')
    ax1.set_xlabel('Volume (V/V₀)', fontsize=12)
    ax1.set_ylabel('Entropy Change (J/K)', fontsize=12)
    ax1.set_title('Free Expansion: ΔS > 0 (Irreversible)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(1.5, R * np.log(2) * 0.5, 'Spontaneous\nProcess', 
             fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 2. Heat Transfer - Entropy Production
    ax2 = axes[0, 1]
    T_hot = np.linspace(400, 300, 100)
    T_cold = np.linspace(200, 300, 100)
    Q = 1000  # Joules
    
    system = ThermodynamicSystem(300)
    entropy_total = []
    
    for i in range(len(T_hot)):
        if T_hot[i] > T_cold[i]:
            dS_tot, _, _ = system.entropy_production(Q, T_hot[i], T_cold[i])
            entropy_total.append(dS_tot)
        else:
            entropy_total.append(0)
    
    time = np.linspace(0, 10, 100)
    ax2.plot(time, entropy_total, 'r-', linewidth=2.5, label='Total Entropy (ΔS_total)')
    ax2.fill_between(time, 0, entropy_total, alpha=0.3, color='red')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (arbitrary units)', fontsize=12)
    ax2.set_ylabel('Entropy Production (J/K)', fontsize=12)
    ax2.set_title('Heat Transfer: ΔS_universe ≥ 0 (2nd Law)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(5, max(entropy_total) * 0.7, 'Arrow of Time →\nIrreversible', 
             fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 3. Boltzmann Entropy - Microstate Distribution
    ax3 = axes[1, 0]
    # Number of microstates for different macrostates
    n_particles = 20
    macrostates = np.arange(0, n_particles + 1)
    # Binomial distribution - number of ways to arrange particles
    from scipy.special import comb
    microstates = comb(n_particles, macrostates)
    entropy_boltzmann = k_B * np.log(microstates + 1)  # +1 to avoid log(0)
    
    ax3.bar(macrostates, microstates, alpha=0.7, color='blue', label='Microstates (W)')
    ax3.set_xlabel('Macrostate (particles in left half)', fontsize=12)
    ax3.set_ylabel('Number of Microstates (W)', fontsize=12)
    ax3.set_title('Boltzmann Entropy: S = k_B·ln(W)', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Mark most probable state
    max_idx = np.argmax(microstates)
    ax3.axvline(x=macrostates[max_idx], color='r', linestyle='--', 
                linewidth=2, label='Maximum Entropy State')
    ax3.text(max_idx, max(microstates) * 0.8, f'Most Probable\n(Highest S)', 
             fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 4. Carnot Cycle - Maximum Efficiency
    ax4 = axes[1, 1]
    T_h = 500  # Hot reservoir (K)
    T_c_range = np.linspace(100, 499, 100)
    
    # Carnot efficiency
    eta_carnot = 1 - (T_c_range / T_h)
    
    # Entropy change in Carnot cycle (reversible)
    Q_h = 1000  # Heat absorbed from hot reservoir
    Q_c = Q_h * (T_c_range / T_h)
    W_out = Q_h - Q_c  # Work output
    
    ax4.plot(T_c_range, eta_carnot * 100, 'g-', linewidth=2.5, 
             label=f'Carnot Efficiency (T_h = {T_h} K)')
    ax4.fill_between(T_c_range, 0, eta_carnot * 100, alpha=0.3, color='green')
    ax4.axhline(y=100, color='r', linestyle='--', linewidth=2, 
                label='Impossible (100% - Perpetual Motion)')
    ax4.set_xlabel('Cold Reservoir Temperature (K)', fontsize=12)
    ax4.set_ylabel('Maximum Efficiency (%)', fontsize=12)
    ax4.set_title('No Perpetual Motion: η < 100%', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.text(300, 85, 'Energy lost\nas waste heat', 
             fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
    
    plt.tight_layout()
    return fig


def demonstrate_irreversibility():
    """
    Show irreversible processes and arrow of time
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Irreversibility and Arrow of Time', fontsize=16, fontweight='bold')
    
    # 1. Gas Mixing (Irreversible)
    ax1 = axes[0, 0]
    time = np.linspace(0, 10, 100)
    # Entropy increases during mixing
    S_mixing = 8.314 * (1 - np.exp(-time/2))  # Asymptotic approach to equilibrium
    
    ax1.plot(time, S_mixing, 'b-', linewidth=2.5, label='Entropy (mixing)')
    ax1.axhline(y=8.314, color='r', linestyle='--', alpha=0.5, label='Equilibrium')
    ax1.arrow(8, 1, 0, 5, head_width=0.3, head_length=0.3, fc='red', ec='red')
    ax1.text(8.5, 3.5, 'Arrow of\nTime', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Entropy (J/K·mol)', fontsize=12)
    ax1.set_title('Gas Mixing: Spontaneous & Irreversible', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(2, 7, 'Reverse process\nnever observed!', 
             fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 2. Temperature Equilibration
    ax2 = axes[0, 1]
    T1_initial, T2_initial = 400, 200
    T_equilibrium = (T1_initial + T2_initial) / 2
    
    T1 = T1_initial - (T1_initial - T_equilibrium) * (1 - np.exp(-time/2))
    T2 = T2_initial + (T_equilibrium - T2_initial) * (1 - np.exp(-time/2))
    
    ax2.plot(time, T1, 'r-', linewidth=2.5, label='Hot Object')
    ax2.plot(time, T2, 'b-', linewidth=2.5, label='Cold Object')
    ax2.axhline(y=T_equilibrium, color='purple', linestyle='--', 
                linewidth=2, label='Equilibrium')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Temperature (K)', fontsize=12)
    ax2.set_title('Heat Flow: Hot → Cold (Never Reverses)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.annotate('Spontaneous\nDirection', xy=(5, 350), xytext=(7, 370),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold')
    
    # 3. Work to Heat Conversion (Irreversible)
    ax3 = axes[1, 0]
    work_input = np.linspace(0, 100, 100)
    heat_output = work_input  # 100% conversion
    entropy_increase = work_input / 300  # At T = 300 K
    
    ax3.plot(work_input, heat_output, 'orange', linewidth=2.5, 
             label='Work → Heat (100% efficient)')
    ax3.plot(work_input, work_input * 0.4, 'blue', linestyle='--', 
             linewidth=2.5, label='Heat → Work (< 100% efficient)')
    ax3.fill_between(work_input, work_input * 0.4, work_input, 
                     alpha=0.3, color='red', label='Lost as waste heat')
    ax3.set_xlabel('Work Input (J)', fontsize=12)
    ax3.set_ylabel('Useful Output (J)', fontsize=12)
    ax3.set_title('Work-Heat Asymmetry: Limits System Performance', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Entropy Timeline - Universe Evolution
    ax4 = axes[1, 1]
    cosmic_time = np.linspace(0, 100, 1000)
    # Entropy of universe increases monotonically
    S_universe = 1e23 * (1 + cosmic_time)  # Simplified model
    
    ax4.semilogy(cosmic_time, S_universe, 'purple', linewidth=3)
    ax4.set_xlabel('Cosmic Time (arbitrary units)', fontsize=12)
    ax4.set_ylabel('Entropy of Universe (J/K)', fontsize=12)
    ax4.set_title('Universal Entropy: Always Increasing', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    
    # Mark key epochs
    ax4.axvline(x=10, color='red', linestyle='--', alpha=0.5)
    ax4.text(10, 1.5e23, 'Now', fontsize=10, rotation=90)
    ax4.axvline(x=90, color='orange', linestyle='--', alpha=0.5)
    ax4.text(90, 1.5e23, 'Heat Death?', fontsize=10, rotation=90)
    
    ax4.annotate('Arrow of Time →', xy=(50, 5e23), fontsize=14, 
                fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    return fig


def calculate_entropy_examples():
    """
    Numerical examples of entropy calculations
    """
    print("=" * 70)
    print("THERMODYNAMIC ENTROPY: Quantitative Examples")
    print("=" * 70)
    
    system = ThermodynamicSystem(300)
    
    # Example 1: Isothermal Expansion
    print("\n1. ISOTHERMAL EXPANSION (Reversible)")
    print("-" * 70)
    V_i, V_f = 1.0, 2.0  # m³
    n, R = 1, 8.314  # mol, J/(mol·K)
    T = 300  # K
    
    Q_rev = n * R * T * np.log(V_f / V_i)
    dS = system.clausius_entropy_change(Q_rev, T)
    
    print(f"   Initial Volume: {V_i} m³")
    print(f"   Final Volume: {V_f} m³")
    print(f"   Temperature: {T} K")
    print(f"   Heat absorbed (reversible): {Q_rev:.2f} J")
    print(f"   Entropy change: ΔS = {dS:.3f} J/K")
    print(f"   Result: ΔS > 0 (Entropy increases)")
    
    # Example 2: Irreversible Heat Transfer
    print("\n2. IRREVERSIBLE HEAT TRANSFER")
    print("-" * 70)
    T_hot, T_cold = 400, 200  # K
    Q = 1000  # J
    
    dS_total, dS_hot, dS_cold = system.entropy_production(Q, T_hot, T_cold)
    
    print(f"   Hot reservoir: {T_hot} K")
    print(f"   Cold reservoir: {T_cold} K")
    print(f"   Heat transferred: {Q} J")
    print(f"   ΔS_hot = {dS_hot:.3f} J/K (decreases)")
    print(f"   ΔS_cold = {dS_cold:.3f} J/K (increases)")
    print(f"   ΔS_total = {dS_total:.3f} J/K")
    print(f"   Result: ΔS_total > 0 (2nd Law satisfied)")
    
    # Example 3: Carnot Engine
    print("\n3. CARNOT ENGINE (Maximum Efficiency)")
    print("-" * 70)
    T_h, T_c = 500, 300  # K
    Q_h = 1000  # J
    
    eta_carnot = 1 - (T_c / T_h)
    W_max = eta_carnot * Q_h
    Q_c = Q_h - W_max
    
    print(f"   Hot reservoir: {T_h} K")
    print(f"   Cold reservoir: {T_c} K")
    print(f"   Heat input: {Q_h} J")
    print(f"   Maximum efficiency: η = {eta_carnot * 100:.1f}%")
    print(f"   Work output: {W_max:.2f} J")
    print(f"   Waste heat: {Q_c:.2f} J")
    print(f"   Result: η < 100% (Perpetual motion impossible)")
    
    # Entropy check for Carnot cycle
    dS_cycle = Q_h / T_h - Q_c / T_c
    print(f"   Net entropy change (cycle): {dS_cycle:.6f} J/K ≈ 0")
    print(f"   (Reversible cycle: ΔS = 0)")
    
    # Example 4: Boltzmann Entropy
    print("\n4. BOLTZMANN ENTROPY")
    print("-" * 70)
    W_ordered = 1  # Only one way to be perfectly ordered
    W_disordered = 10**20  # Many ways to be disordered
    
    S_ordered = system.boltzmann_entropy(W_ordered)
    S_disordered = system.boltzmann_entropy(W_disordered)
    
    print(f"   Ordered state microstates: W = {W_ordered}")
    print(f"   S_ordered = k_B·ln({W_ordered}) = {S_ordered:.2e} J/K")
    print(f"   Disordered state microstates: W = {W_disordered:.2e}")
    print(f"   S_disordered = k_B·ln({W_disordered:.2e}) = {S_disordered:.2e} J/K")
    print(f"   ΔS = {S_disordered - S_ordered:.2e} J/K")
    print(f"   Result: Systems spontaneously move toward disorder")
    
    # Example 5: Free Expansion (Joule Expansion)
    print("\n5. FREE EXPANSION (Irreversible - Joule Expansion)")
    print("-" * 70)
    n, R = 1, 8.314
    V_initial, V_final = 1.0, 3.0
    
    # Free expansion: Q = 0, W = 0, but ΔS > 0!
    dS_free = n * R * np.log(V_final / V_initial)
    
    print(f"   Initial volume: {V_initial} m³")
    print(f"   Final volume: {V_final} m³")
    print(f"   Heat transfer: Q = 0 J")
    print(f"   Work done: W = 0 J")
    print(f"   Entropy change: ΔS = {dS_free:.3f} J/K")
    print(f"   Result: Entropy increases even with no heat transfer!")
    print(f"   This is IRREVERSIBLE - gas never spontaneously contracts")
    
    print("\n" + "=" * 70)
    print("KEY CONCLUSIONS:")
    print("=" * 70)
    print("1. Entropy is NOT conserved (unlike energy)")
    print("2. Isolated systems: ΔS ≥ 0 (equality only for reversible)")
    print("3. Spontaneous processes increase total entropy")
    print("4. Perpetual motion machines are impossible")
    print("5. Arrow of time: entropy defines temporal direction")
    print("6. Work → Heat (100% efficient), but Heat → Work (< 100%)")
    print("=" * 70)


def main():
    """
    Main demonstration of thermodynamic entropy theory
    """
    print("\n" + "=" * 70)
    print("THERMODYNAMIC ENTROPY THEORY DEMONSTRATION")
    print("Non-Conserved State Function & Arrow of Time")
    print("=" * 70 + "\n")
    
    # Numerical calculations
    calculate_entropy_examples()
    
    print("\nGenerating visualizations...")
    
    # Create figures
    fig1 = demonstrate_entropy_increase()
    fig2 = demonstrate_irreversibility()
    
    print("\n✓ Visualizations created successfully!")
    print("\nKey Physical Insights:")
    print("  • Entropy increases in spontaneous processes")
    print("  • Time reversal would decrease entropy (never observed)")
    print("  • Waste heat limits system efficiency")
    print("  • Universe evolves toward maximum entropy state")
    
    plt.show()


if __name__ == "__main__":
    main()
