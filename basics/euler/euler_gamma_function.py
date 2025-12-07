"""
Euler's Gamma Function Γ(n)

A generalization of the factorial function to real and complex numbers.

Definition:
Γ(n) = ∫₀^∞ t^(n-1) e^(-t) dt  for Re(n) > 0

Key Properties:
1. Γ(n+1) = n·Γ(n) (functional equation)
2. Γ(n+1) = n! for non-negative integers n
3. Γ(1) = 1
4. Γ(1/2) = √π
5. Γ(n)·Γ(1-n) = π/sin(πn) (reflection formula)

Special Values:
• Γ(1) = 1
• Γ(2) = 1
• Γ(3) = 2
• Γ(4) = 6
• Γ(5) = 24
• Γ(1/2) = √π ≈ 1.772
• Γ(3/2) = √π/2
• Γ(-1/2) = -2√π

Applications:
• Probability distributions (Beta, Gamma, Chi-squared)
• Complex analysis (Riemann zeta function)
• Number theory
• Physics (quantum field theory, statistical mechanics)
• Combinatorics (generalized binomial coefficients)
• Differential equations

Historical Note:
Introduced by Leonhard Euler (1729) to interpolate factorials.
Later studied extensively by Gauss, Legendre, and Weierstrass.

Related Functions:
• Beta function: B(x,y) = Γ(x)·Γ(y)/Γ(x+y)
• Digamma function: ψ(x) = d/dx[ln(Γ(x))]
• Pochhammer symbol: (x)_n = Γ(x+n)/Γ(x)

Author: Leonhard Euler (1707-1783)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.integrate import quad
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)


def gamma_integral(n, n_samples=100000):
    """
    Compute Gamma function using integral definition
    Γ(n) = ∫₀^∞ t^(n-1) e^(-t) dt
    
    Uses numerical integration for demonstration
    """
    def integrand(t, n):
        if t == 0:
            return 0 if n > 0 else np.inf
        return t**(n-1) * np.exp(-t)
    
    # Use adaptive quadrature
    result, error = quad(lambda t: integrand(t, n), 0, 50)
    return result, error


def demonstrate_gamma_definition():
    """
    Demonstrate the definition and basic properties
    """
    print("=" * 80)
    print("EULER'S GAMMA FUNCTION Γ(n)")
    print("=" * 80)
    
    print("\nDefinition: Γ(n) = ∫₀^∞ t^(n-1) e^(-t) dt  for Re(n) > 0")
    
    print("\n1. FACTORIAL PROPERTY: Γ(n+1) = n!")
    print("-" * 80)
    print(f"{'n':>5s}  {'n!':>15s}  {'Γ(n+1) scipy':>20s}  {'Γ(n+1) integral':>20s}  {'Difference':>15s}")
    print("-" * 80)
    
    for n in range(0, 11):
        factorial_val = np.math.factorial(n)
        gamma_scipy = special.gamma(n + 1)
        gamma_integral_val, _ = gamma_integral(n + 1)
        diff = abs(gamma_scipy - factorial_val)
        
        print(f"{n:5d}  {factorial_val:15.6e}  {gamma_scipy:20.10f}  "
              f"{gamma_integral_val:20.10f}  {diff:15.2e}")
    
    print("\n2. FUNCTIONAL EQUATION: Γ(n+1) = n·Γ(n)")
    print("-" * 80)
    print(f"{'n':>8s}  {'Γ(n)':>20s}  {'n·Γ(n)':>20s}  {'Γ(n+1)':>20s}  {'Difference':>15s}")
    print("-" * 80)
    
    test_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    for n in test_values:
        gamma_n = special.gamma(n)
        n_times_gamma_n = n * gamma_n
        gamma_n_plus_1 = special.gamma(n + 1)
        diff = abs(n_times_gamma_n - gamma_n_plus_1)
        
        print(f"{n:8.1f}  {gamma_n:20.10f}  {n_times_gamma_n:20.10f}  "
              f"{gamma_n_plus_1:20.10f}  {diff:15.2e}")


def demonstrate_special_values():
    """
    Show special values of the Gamma function
    """
    print("\n" + "=" * 80)
    print("SPECIAL VALUES")
    print("=" * 80)
    
    print("\n1. INTEGER VALUES")
    print("-" * 80)
    
    special_int = [
        (1, "0! = 1"),
        (2, "1! = 1"),
        (3, "2! = 2"),
        (4, "3! = 6"),
        (5, "4! = 24"),
        (6, "5! = 120"),
        (10, "9! = 362880"),
    ]
    
    for n, description in special_int:
        value = special.gamma(n)
        print(f"   Γ({n:2d}) = {value:15.6f}  ({description})")
    
    print("\n2. HALF-INTEGER VALUES")
    print("-" * 80)
    print("   Important: Γ(1/2) = √π")
    
    sqrt_pi = np.sqrt(np.pi)
    gamma_half = special.gamma(0.5)
    
    print(f"   Γ(1/2) = {gamma_half:.15f}")
    print(f"   √π     = {sqrt_pi:.15f}")
    print(f"   Difference: {abs(gamma_half - sqrt_pi):.2e}")
    
    print("\n   Other half-integer values:")
    
    half_integers = [
        (1.5, "Γ(3/2) = √π/2"),
        (2.5, "Γ(5/2) = 3√π/4"),
        (3.5, "Γ(7/2) = 15√π/8"),
        (-0.5, "Γ(-1/2) = -2√π"),
        (-1.5, "Γ(-3/2) = 4√π/3"),
    ]
    
    for n, description in half_integers:
        value = special.gamma(n)
        print(f"   Γ({n:5.1f}) = {value:15.10f}  ({description})")
    
    print("\n3. REFLECTION FORMULA: Γ(z)·Γ(1-z) = π/sin(πz)")
    print("-" * 80)
    
    test_z = [0.3, 0.5, 0.7, 1.2, 1.5, 2.3]
    
    for z in test_z:
        lhs = special.gamma(z) * special.gamma(1 - z)
        rhs = np.pi / np.sin(np.pi * z)
        diff = abs(lhs - rhs)
        
        print(f"   z = {z:.1f}: Γ(z)·Γ(1-z) = {lhs:12.6f}, "
              f"π/sin(πz) = {rhs:12.6f}, diff = {diff:.2e}")
    
    print("\n4. DUPLICATION FORMULA: Γ(z)·Γ(z+1/2) = √π·2^(1-2z)·Γ(2z)")
    print("-" * 80)
    
    for z in [1.0, 1.5, 2.0, 2.5, 3.0]:
        lhs = special.gamma(z) * special.gamma(z + 0.5)
        rhs = np.sqrt(np.pi) * (2 ** (1 - 2*z)) * special.gamma(2*z)
        diff = abs(lhs - rhs)
        
        print(f"   z = {z:.1f}: LHS = {lhs:15.6f}, RHS = {rhs:15.6f}, diff = {diff:.2e}")


def demonstrate_beta_function():
    """
    Show relationship between Gamma and Beta functions
    """
    print("\n" + "=" * 80)
    print("BETA FUNCTION RELATIONSHIP")
    print("=" * 80)
    
    print("\nDefinition: B(x,y) = Γ(x)·Γ(y)/Γ(x+y)")
    print("Also: B(x,y) = ∫₀¹ t^(x-1)·(1-t)^(y-1) dt")
    
    print("\n" + "-" * 80)
    print(f"{'x':>6s}  {'y':>6s}  {'Γ(x)·Γ(y)/Γ(x+y)':>22s}  {'scipy.beta(x,y)':>20s}  {'Difference':>15s}")
    print("-" * 80)
    
    test_pairs = [(1, 1), (2, 2), (3, 3), (1.5, 2.5), (2.5, 3.5), (0.5, 0.5)]
    
    for x, y in test_pairs:
        gamma_ratio = special.gamma(x) * special.gamma(y) / special.gamma(x + y)
        beta_val = special.beta(x, y)
        diff = abs(gamma_ratio - beta_val)
        
        print(f"{x:6.1f}  {y:6.1f}  {gamma_ratio:22.15f}  {beta_val:20.15f}  {diff:15.2e}")


def demonstrate_applications():
    """
    Show practical applications
    """
    print("\n" + "=" * 80)
    print("APPLICATIONS OF GAMMA FUNCTION")
    print("=" * 80)
    
    print("\n1. PROBABILITY DISTRIBUTIONS")
    print("-" * 80)
    print("   Gamma Distribution PDF:")
    print("   f(x; α, β) = (β^α/Γ(α))·x^(α-1)·e^(-βx)")
    
    print("\n   Chi-squared Distribution (k degrees of freedom):")
    print("   f(x; k) = (1/(2^(k/2)·Γ(k/2)))·x^(k/2-1)·e^(-x/2)")
    
    print("\n   Student's t-distribution involves Γ((ν+1)/2) and Γ(ν/2)")
    
    print("\n2. FACTORIAL INTERPOLATION")
    print("-" * 80)
    print("   Γ extends n! to non-integer values:")
    
    for x in [0.5, 1.5, 2.5, 3.7, 4.2, 5.8]:
        gamma_val = special.gamma(x + 1)
        print(f"   ({x:.1f})! = Γ({x+1:.1f}) = {gamma_val:.10f}")
    
    print("\n3. RIEMANN ZETA FUNCTION")
    print("-" * 80)
    print("   ζ(s) = (1/Γ(s))·∫₀^∞ (t^(s-1))/(e^t - 1) dt")
    print("   The Gamma function appears in integral representations of ζ")
    
    print("\n4. STIRLING'S APPROXIMATION")
    print("-" * 80)
    print("   For large n: Γ(n) ≈ √(2π/n)·(n/e)^n")
    print("   Or: n! ≈ √(2πn)·(n/e)^n")
    
    print(f"\n   {'n':>5s}  {'Γ(n+1) exact':>20s}  {'Stirling approx':>20s}  {'Relative Error':>15s}")
    print("   " + "-" * 70)
    
    for n in [5, 10, 20, 50, 100]:
        exact = special.gamma(n + 1)
        stirling = np.sqrt(2 * np.pi * n) * (n / np.e) ** n
        rel_error = abs(exact - stirling) / exact
        
        print(f"   {n:5d}  {exact:20.6e}  {stirling:20.6e}  {rel_error:15.2e}")


def create_visualizations():
    """
    Create comprehensive visualizations
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Gamma function for positive reals
    ax1 = plt.subplot(3, 3, 1)
    x = np.linspace(0.01, 5, 1000)
    y = special.gamma(x)
    
    ax1.plot(x, y, 'b-', linewidth=2.5, label='Γ(x)')
    
    # Mark integer values
    for n in range(1, 6):
        gamma_n = special.gamma(n)
        ax1.plot(n, gamma_n, 'ro', markersize=10)
        ax1.text(n, gamma_n + 2, f'{n-1}!', ha='center', fontsize=9)
    
    # Mark Γ(1/2)
    ax1.plot(0.5, special.gamma(0.5), 'gs', markersize=12)
    ax1.text(0.5, special.gamma(0.5) + 2, '√π', ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('Γ(x)', fontsize=11)
    ax1.set_title('Gamma Function (Positive x)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 30])
    
    # 2. Gamma function including negative values
    ax2 = plt.subplot(3, 3, 2)
    x_full = np.linspace(-4.5, 5, 5000)
    y_full = np.zeros_like(x_full)
    
    # Compute Gamma avoiding poles at non-positive integers
    for i, xi in enumerate(x_full):
        if xi <= 0 and xi == int(xi):
            y_full[i] = np.nan
        else:
            try:
                y_full[i] = special.gamma(xi)
            except:
                y_full[i] = np.nan
    
    ax2.plot(x_full, y_full, 'b-', linewidth=1.5)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    
    # Mark poles
    for n in range(-4, 1):
        ax2.axvline(x=n, color='r', linestyle='--', alpha=0.3)
        ax2.text(n, 0, f'{n}', ha='center', va='bottom', fontsize=9, color='red')
    
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('Γ(x)', fontsize=11)
    ax2.set_title('Gamma Function (Full Real Line)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-5, 5])
    ax2.set_xlim([-4.5, 5])
    
    # 3. Log-Gamma function
    ax3 = plt.subplot(3, 3, 3)
    x_log = np.linspace(0.1, 10, 1000)
    y_log = special.loggamma(x_log)
    
    ax3.plot(x_log, y_log, 'g-', linewidth=2.5, label='ln(Γ(x))')
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('ln(Γ(x))', fontsize=11)
    ax3.set_title('Log-Gamma Function', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Factorial comparison
    ax4 = plt.subplot(3, 3, 4)
    n_vals = np.arange(0, 11)
    factorial_vals = [np.math.factorial(n) for n in n_vals]
    gamma_vals = [special.gamma(n + 1) for n in n_vals]
    
    ax4.semilogy(n_vals, factorial_vals, 'ro-', linewidth=2, markersize=8, 
                label='n!')
    ax4.semilogy(n_vals, gamma_vals, 'bx--', linewidth=2, markersize=10,
                label='Γ(n+1)', alpha=0.7)
    ax4.set_xlabel('n', fontsize=11)
    ax4.set_ylabel('Value (log scale)', fontsize=11)
    ax4.set_title('Factorial vs Gamma: Γ(n+1) = n!', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, which='both')
    
    # 5. Half-integer values
    ax5 = plt.subplot(3, 3, 5)
    half_int_x = np.arange(0.5, 6, 0.5)
    half_int_y = special.gamma(half_int_x)
    
    ax5.plot(half_int_x, half_int_y, 'mo-', linewidth=2, markersize=8)
    
    # Highlight multiples of √π
    sqrt_pi = np.sqrt(np.pi)
    ax5.axhline(y=sqrt_pi, color='orange', linestyle='--', alpha=0.5, 
               label=f'√π ≈ {sqrt_pi:.3f}')
    ax5.axhline(y=sqrt_pi/2, color='cyan', linestyle='--', alpha=0.5,
               label=f'√π/2 ≈ {sqrt_pi/2:.3f}')
    
    ax5.set_xlabel('x', fontsize=11)
    ax5.set_ylabel('Γ(x)', fontsize=11)
    ax5.set_title('Half-Integer Values', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Digamma function (derivative of log-gamma)
    ax6 = plt.subplot(3, 3, 6)
    x_digamma = np.linspace(0.1, 5, 1000)
    y_digamma = special.digamma(x_digamma)
    
    ax6.plot(x_digamma, y_digamma, 'purple', linewidth=2.5, label='ψ(x) = d/dx[ln(Γ(x))]')
    ax6.axhline(y=0, color='k', linewidth=0.5)
    ax6.set_xlabel('x', fontsize=11)
    ax6.set_ylabel('ψ(x)', fontsize=11)
    ax6.set_title('Digamma Function', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Beta function
    ax7 = plt.subplot(3, 3, 7)
    x_beta = np.linspace(0.1, 5, 100)
    y_beta = np.linspace(0.1, 5, 100)
    X_beta, Y_beta = np.meshgrid(x_beta, y_beta)
    Z_beta = np.zeros_like(X_beta)
    
    for i in range(len(x_beta)):
        for j in range(len(y_beta)):
            Z_beta[j, i] = special.beta(X_beta[j, i], Y_beta[j, i])
    
    contour = ax7.contourf(X_beta, Y_beta, Z_beta, levels=30, cmap='viridis')
    plt.colorbar(contour, ax=ax7, label='B(x,y)')
    ax7.set_xlabel('x', fontsize=11)
    ax7.set_ylabel('y', fontsize=11)
    ax7.set_title('Beta Function B(x,y)', fontsize=12, fontweight='bold')
    
    # 8. Stirling approximation
    ax8 = plt.subplot(3, 3, 8)
    n_stirling = np.arange(1, 21)
    exact_vals = special.gamma(n_stirling + 1)
    stirling_vals = np.sqrt(2 * np.pi * n_stirling) * (n_stirling / np.e) ** n_stirling
    
    ax8.semilogy(n_stirling, exact_vals, 'bo-', linewidth=2, markersize=6, label='Γ(n+1) exact')
    ax8.semilogy(n_stirling, stirling_vals, 'r^--', linewidth=2, markersize=6, 
                label='Stirling approx')
    ax8.set_xlabel('n', fontsize=11)
    ax8.set_ylabel('Value (log scale)', fontsize=11)
    ax8.set_title("Stirling's Approximation", fontsize=12, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3, which='both')
    
    # 9. Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary = f"""
GAMMA FUNCTION Γ(n)
═══════════════════════════

Definition:
───────────
Γ(n) = ∫₀^∞ t^(n-1)e^(-t) dt

Key Properties:
───────────────
• Γ(n+1) = n·Γ(n)
• Γ(n+1) = n! for n∈ℕ
• Γ(1) = 1
• Γ(1/2) = √π

Special Values:
───────────────
• Γ(1) = 1
• Γ(2) = 1
• Γ(3) = 2
• Γ(4) = 6
• Γ(5) = 24

Formulas:
─────────
• Reflection:
  Γ(z)·Γ(1-z) = π/sin(πz)
  
• Duplication:
  Γ(z)·Γ(z+1/2) = 
    √π·2^(1-2z)·Γ(2z)

• Stirling:
  n! ≈ √(2πn)·(n/e)^n

Applications:
─────────────
• Probability distributions
• Complex analysis
• Number theory
• Physics
═══════════════════════════
    """
    
    ax9.text(0.05, 0.95, summary, transform=ax9.transAxes,
            fontsize=8.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle("Euler's Gamma Function: Generalization of the Factorial", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def main():
    """
    Main demonstration
    """
    print("\n" + "=" * 80)
    print(" EULER'S GAMMA FUNCTION Γ(n)")
    print(" Generalization of the Factorial")
    print("=" * 80)
    
    # Definition and properties
    demonstrate_gamma_definition()
    
    # Special values
    demonstrate_special_values()
    
    # Beta function
    demonstrate_beta_function()
    
    # Applications
    demonstrate_applications()
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    
    # Create visualizations
    fig = create_visualizations()
    
    print("\n✓ Visualizations created successfully!")
    print("\nKey Insights:")
    print("  • Gamma function extends factorial to all real and complex numbers")
    print("  • Γ(n+1) = n! for non-negative integers")
    print("  • Γ(1/2) = √π is a beautiful special value")
    print("  • Essential in probability, statistics, and complex analysis")
    print("  • Connects to many other special functions")
    
    plt.show()


if __name__ == "__main__":
    main()
