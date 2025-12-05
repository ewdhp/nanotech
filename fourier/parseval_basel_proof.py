"""
Basel Problem Proof using Fourier Series and Parseval's Identity

This script demonstrates the proof that ∑(1/n²) = π²/6 using:
1. Fourier series expansion of f(x) = x on [-π, π]
2. Parseval's identity relating Fourier coefficients to L² norm
3. Direct calculation showing the connection to the Basel sum

Mathematical Framework:
----------------------
For f(x) = x on [-π, π]:

Fourier series: f(x) = a₀/2 + ∑[aₙcos(nx) + bₙsin(nx)]

Fourier coefficients:
  a₀ = (1/π)∫₋π^π x dx = 0
  aₙ = (1/π)∫₋π^π x·cos(nx) dx = 0  (odd function × even function)
  bₙ = (1/π)∫₋π^π x·sin(nx) dx = 2(-1)^(n+1)/n

Complex form: f(x) = ∑ cₙe^(inx) where cₙ = (aₙ - ibₙ)/2

Parseval's Identity:
  ∑|cₙ|² = (1/2π)∫₋π^π |f(x)|² dx

For f(x) = x:
  ∑|cₙ|² = (1/2π)∫₋π^π x² dx = (1/2π)[x³/3]₋π^π = π²/3

From Fourier coefficients:
  |c₀|² = 0
  |cₙ|² = bₙ²/4 = 1/n² for n ≠ 0

Therefore:
  ∑_{n=-∞}^∞ |cₙ|² = 2∑_{n=1}^∞ 1/n² = π²/3
  
Thus: ∑_{n=1}^∞ 1/n² = π²/6  ✓

Reference: Fourier Analysis, Parseval's theorem
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad
from scipy.special import factorial

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def fourier_coefficient_bn(n, numerical=False):
    """
    Calculate bₙ coefficient for f(x) = x on [-π, π]
    
    Analytical: bₙ = (1/π)∫₋π^π x·sin(nx) dx = 2(-1)^(n+1)/n
    
    Parameters:
    -----------
    n : int
        Fourier coefficient index
    numerical : bool
        If True, compute numerically; if False, use analytical formula
    
    Returns:
    --------
    float : bₙ coefficient
    """
    if n == 0:
        return 0.0
    
    if numerical:
        # Numerical integration
        integrand = lambda x: x * np.sin(n * x)
        result, _ = quad(integrand, -np.pi, np.pi)
        return result / np.pi
    else:
        # Analytical formula: bₙ = 2(-1)^(n+1)/n
        return 2.0 * ((-1) ** (n + 1)) / n


def fourier_coefficient_an(n, numerical=False):
    """
    Calculate aₙ coefficient for f(x) = x on [-π, π]
    
    Since f(x) = x is odd and cos(nx) is even, the product is odd,
    so the integral over symmetric interval is 0.
    
    Analytical: aₙ = 0 for all n
    """
    if numerical:
        integrand = lambda x: x * np.cos(n * x)
        result, _ = quad(integrand, -np.pi, np.pi)
        return result / np.pi
    else:
        return 0.0


def fourier_coefficient_cn(n):
    """
    Complex Fourier coefficient: cₙ = (aₙ - ibₙ)/2
    
    For f(x) = x:
    - a₀ = 0
    - aₙ = 0 for all n
    - bₙ = 2(-1)^(n+1)/n
    
    So: c₀ = 0
        cₙ = -ibₙ/2 = -i·(-1)^(n+1)/n for n ≠ 0
    """
    if n == 0:
        return 0 + 0j
    
    bn = fourier_coefficient_bn(n)
    an = 0.0
    return (an - 1j * bn) / 2


def fourier_series_reconstruction(x, n_terms):
    """
    Reconstruct f(x) = x using Fourier series with n_terms
    
    f(x) ≈ ∑_{n=1}^{N} bₙ sin(nx)
         = ∑_{n=1}^{N} [2(-1)^(n+1)/n] sin(nx)
    
    Parameters:
    -----------
    x : array-like
        Points to evaluate
    n_terms : int
        Number of Fourier terms
    
    Returns:
    --------
    array : Reconstructed function values
    """
    result = np.zeros_like(x, dtype=float)
    
    for n in range(1, n_terms + 1):
        bn = fourier_coefficient_bn(n)
        result += bn * np.sin(n * x)
    
    return result


def parseval_identity_lhs(n_max):
    """
    Left-hand side of Parseval's identity: ∑|cₙ|²
    
    For f(x) = x, cₙ = -i·(-1)^(n+1)/n
    So |cₙ|² = 1/n²
    
    Therefore: ∑_{n=-∞}^∞ |cₙ|² = 2∑_{n=1}^∞ 1/n²
    """
    # Sum from -n_max to n_max, excluding n=0
    total = 0.0
    for n in range(-n_max, n_max + 1):
        if n == 0:
            continue
        cn = fourier_coefficient_cn(n)
        total += abs(cn) ** 2
    
    return total


def parseval_identity_rhs():
    """
    Right-hand side of Parseval's identity: (1/2π)∫₋π^π x² dx
    
    Analytical: (1/2π)[x³/3]₋π^π = (1/2π)·(2π³/3) = π²/3
    """
    # Numerical verification
    integrand = lambda x: x ** 2
    integral, _ = quad(integrand, -np.pi, np.pi)
    return integral / (2 * np.pi)


def basel_sum_partial(n_terms):
    """
    Compute partial sum: ∑_{n=1}^N 1/n²
    """
    return sum(1.0 / (n ** 2) for n in range(1, n_terms + 1))


def demonstrate_parseval_proof():
    """
    Step-by-step demonstration of the proof
    """
    print("=" * 80)
    print("BASEL PROBLEM PROOF VIA FOURIER SERIES & PARSEVAL'S IDENTITY")
    print("=" * 80)
    
    print("\n1. FOURIER SERIES OF f(x) = x on [-π, π]")
    print("-" * 80)
    print("   f(x) = x is an odd function")
    print("   Therefore: a₀ = 0, aₙ = 0 (cosine terms vanish)")
    print("   Only sine terms remain:")
    print()
    
    print("   Computing bₙ = (1/π)∫₋π^π x·sin(nx) dx:")
    for n in [1, 2, 3, 4, 5]:
        bn_analytical = fourier_coefficient_bn(n, numerical=False)
        bn_numerical = fourier_coefficient_bn(n, numerical=True)
        print(f"     b_{n} = {bn_analytical:10.6f} (analytical) | {bn_numerical:10.6f} (numerical)")
    
    print("\n   Pattern: bₙ = 2(-1)^(n+1)/n")
    print("   So: f(x) = ∑_{n=1}^∞ [2(-1)^(n+1)/n] sin(nx)")
    
    print("\n2. COMPLEX FOURIER COEFFICIENTS")
    print("-" * 80)
    print("   In complex form: f(x) = ∑_{n=-∞}^∞ cₙe^(inx)")
    print("   where cₙ = (aₙ - ibₙ)/2")
    print()
    print("   For f(x) = x:")
    print("     c₀ = 0")
    print("     cₙ = -ibₙ/2 = -i·(-1)^(n+1)/n  (n ≠ 0)")
    print()
    
    print("   Computing |cₙ|²:")
    for n in [1, 2, 3, 4, 5]:
        cn = fourier_coefficient_cn(n)
        mag_squared = abs(cn) ** 2
        expected = 1.0 / (n ** 2)
        print(f"     |c_{n}|² = {mag_squared:.10f} = 1/{n}² = {expected:.10f}")
    
    print("\n3. PARSEVAL'S IDENTITY")
    print("-" * 80)
    print("   Parseval's identity states:")
    print("     ∑_{n=-∞}^∞ |cₙ|² = (1/2π)∫₋π^π |f(x)|² dx")
    print()
    
    # Right-hand side (analytical)
    rhs_analytical = (np.pi ** 2) / 3
    rhs_numerical = parseval_identity_rhs()
    
    print("   Right-hand side (RHS):")
    print(f"     (1/2π)∫₋π^π x² dx = (1/2π)[x³/3]₋π^π")
    print(f"                        = (1/2π)·(2π³/3)")
    print(f"                        = π²/3")
    print(f"                        = {rhs_analytical:.15f} (analytical)")
    print(f"                        = {rhs_numerical:.15f} (numerical)")
    print()
    
    # Left-hand side (sum of |cₙ|²)
    print("   Left-hand side (LHS):")
    print("     ∑_{n=-∞}^∞ |cₙ|²")
    print()
    
    for N in [10, 50, 100, 500, 1000]:
        lhs = parseval_identity_lhs(N)
        print(f"     Using n ∈ [-{N}, {N}]: LHS = {lhs:.15f}")
    
    print()
    print(f"   As N → ∞, LHS → {rhs_analytical:.15f}")
    
    print("\n4. CONNECTION TO BASEL PROBLEM")
    print("-" * 80)
    print("   Since |cₙ|² = 1/n² for n ≠ 0:")
    print()
    print("     ∑_{n=-∞}^∞ |cₙ|² = ∑_{n=-∞}^{-1} 1/n² + ∑_{n=1}^∞ 1/n²")
    print("                      = 2∑_{n=1}^∞ 1/n²  (by symmetry)")
    print()
    print("   From Parseval's identity:")
    print("     2∑_{n=1}^∞ 1/n² = π²/3")
    print()
    print("   Therefore:")
    print("     ∑_{n=1}^∞ 1/n² = π²/6")
    print()
    
    exact = (np.pi ** 2) / 6
    print(f"   π²/6 = {exact:.15f}")
    print()
    
    print("   Verification with partial sums:")
    for N in [10, 100, 1000, 10000, 100000]:
        partial = basel_sum_partial(N)
        error = abs(partial - exact)
        print(f"     ∑_{{n=1}}^{{{N:6d}}} 1/n² = {partial:.15f}  (error: {error:.2e})")
    
    print("\n" + "=" * 80)
    print("CONCLUSION: ∑_{n=1}^∞ 1/n² = π²/6  ✓ PROVEN")
    print("=" * 80)


def create_visualizations():
    """
    Create comprehensive visualizations of the proof
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Fourier Series Reconstruction
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.linspace(-np.pi, np.pi, 1000)
    
    ax1.plot(x, x, 'k-', linewidth=3, label='f(x) = x', zorder=5)
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, n_terms in enumerate([1, 3, 5, 10, 50]):
        y_approx = fourier_series_reconstruction(x, n_terms)
        alpha = 0.4 + 0.12 * i
        ax1.plot(x, y_approx, '--', linewidth=2, alpha=alpha, 
                label=f'N = {n_terms} terms', color=colors[i % len(colors)])
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Fourier Series Reconstruction: f(x) = ∑ bₙsin(nx)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-np.pi, np.pi])
    ax1.axhline(y=0, color='gray', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linewidth=0.5)
    
    # 2. Fourier Coefficients bₙ
    ax2 = fig.add_subplot(gs[0, 2])
    n_range = np.arange(1, 21)
    bn_vals = [fourier_coefficient_bn(n) for n in n_range]
    
    ax2.stem(n_range, bn_vals, basefmt=' ', linefmt='blue', markerfmt='bo')
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    ax2.set_xlabel('n', fontsize=12)
    ax2.set_ylabel('bₙ', fontsize=12)
    ax2.set_title('Fourier Coefficients: bₙ = 2(-1)^(n+1)/n', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. |cₙ|² values
    ax3 = fig.add_subplot(gs[1, 0])
    n_range_extended = np.arange(1, 51)
    cn_mag_squared = [abs(fourier_coefficient_cn(n)) ** 2 for n in n_range_extended]
    expected_vals = [1.0 / (n ** 2) for n in n_range_extended]
    
    ax3.scatter(n_range_extended, cn_mag_squared, s=40, alpha=0.7, 
               label='|cₙ|² (computed)', color='blue', zorder=3)
    ax3.plot(n_range_extended, expected_vals, 'r--', linewidth=2, 
            label='1/n² (theory)', alpha=0.7)
    ax3.set_xlabel('n', fontsize=12)
    ax3.set_ylabel('|cₙ|²', fontsize=12)
    ax3.set_title('Complex Coefficients: |cₙ|² = 1/n²', 
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Parseval LHS Convergence
    ax4 = fig.add_subplot(gs[1, 1])
    N_vals = np.arange(1, 201)
    lhs_vals = [parseval_identity_lhs(N) for N in N_vals]
    rhs_exact = (np.pi ** 2) / 3
    
    ax4.plot(N_vals, lhs_vals, 'b-', linewidth=2, label='∑|cₙ|² (LHS)')
    ax4.axhline(y=rhs_exact, color='r', linestyle='--', linewidth=2, 
               label=f'π²/3 = {rhs_exact:.6f}')
    ax4.fill_between(N_vals, lhs_vals, rhs_exact, alpha=0.2, color='yellow')
    ax4.set_xlabel('N (sum from -N to N)', fontsize=12)
    ax4.set_ylabel('Partial Sum', fontsize=12)
    ax4.set_title("Parseval's Identity: ∑|cₙ|² → π²/3", 
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Basel Sum Convergence
    ax5 = fig.add_subplot(gs[1, 2])
    N_basel = np.arange(1, 1001)
    basel_partials = [basel_sum_partial(n) for n in N_basel]
    basel_exact = (np.pi ** 2) / 6
    
    ax5.plot(N_basel, basel_partials, 'g-', linewidth=2, label='∑ 1/n²')
    ax5.axhline(y=basel_exact, color='r', linestyle='--', linewidth=2, 
               label=f'π²/6 = {basel_exact:.6f}')
    ax5.fill_between(N_basel, basel_partials, basel_exact, alpha=0.2, color='cyan')
    ax5.set_xlabel('N', fontsize=12)
    ax5.set_ylabel('Partial Sum', fontsize=12)
    ax5.set_title('Basel Problem: ∑ 1/n² → π²/6', 
                  fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Error in Fourier Reconstruction
    ax6 = fig.add_subplot(gs[2, 0])
    x_test = np.linspace(-np.pi, np.pi, 500)
    N_terms_range = [1, 2, 3, 5, 10, 20, 50, 100]
    
    for N in N_terms_range:
        y_approx = fourier_series_reconstruction(x_test, N)
        error = np.abs(y_approx - x_test)
        ax6.semilogy(x_test, error + 1e-10, linewidth=1.5, label=f'N={N}', alpha=0.7)
    
    ax6.set_xlabel('x', fontsize=12)
    ax6.set_ylabel('|error|', fontsize=12)
    ax6.set_title('Reconstruction Error (Log Scale)', 
                  fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8, ncol=2)
    ax6.grid(True, alpha=0.3, which='both')
    
    # 7. Gibbs Phenomenon at Discontinuities
    ax7 = fig.add_subplot(gs[2, 1])
    x_zoom = np.linspace(-0.5, 0.5, 500)
    
    ax7.plot(x_zoom, x_zoom, 'k-', linewidth=3, label='f(x) = x', zorder=5)
    
    for N in [5, 10, 20, 50]:
        y_zoom = fourier_series_reconstruction(x_zoom, N)
        ax7.plot(x_zoom, y_zoom, '--', linewidth=2, alpha=0.6, label=f'N={N}')
    
    ax7.set_xlabel('x', fontsize=12)
    ax7.set_ylabel('f(x)', fontsize=12)
    ax7.set_title('Convergence Detail (Near Origin)', 
                  fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # 8. Summary Box
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary = f"""
PROOF SUMMARY
═════════════════════════════

1. Function: f(x) = x on [-π, π]

2. Fourier Series:
   f(x) = ∑ bₙsin(nx)
   bₙ = 2(-1)^(n+1)/n

3. Complex Coefficients:
   cₙ = -ibₙ/2
   |cₙ|² = 1/n²

4. Parseval's Identity:
   ∑|cₙ|² = (1/2π)∫x² dx
   
5. Evaluate RHS:
   (1/2π)∫₋π^π x² dx = π²/3

6. Evaluate LHS:
   ∑_{{n=-∞}}^∞ |cₙ|² = 2∑_{{n=1}}^∞ 1/n²

7. Equate:
   2∑ 1/n² = π²/3

8. RESULT:
   ∑_{{n=1}}^∞ 1/n² = π²/6 ✓
   
   π²/6 = {basel_exact:.10f}
═════════════════════════════
    """
    
    ax8.text(0.05, 0.95, summary, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle("Basel Problem Proof via Fourier Series & Parseval's Identity", 
                 fontsize=16, fontweight='bold', y=0.998)
    
    return fig


def main():
    """
    Main demonstration
    """
    print("\n")
    
    # Step-by-step proof
    demonstrate_parseval_proof()
    
    print("\n\nGenerating visualizations...\n")
    
    # Create visualizations
    fig = create_visualizations()
    
    print("✓ Visualizations created successfully!")
    print("\nKey Insights:")
    print("  • Fourier series of f(x) = x uses only sine terms (odd function)")
    print("  • Parseval's identity connects Fourier coefficients to L² norm")
    print("  • The sum ∑|cₙ|² = π²/3 immediately gives Basel result")
    print("  • This proof is more elementary than Euler's infinite product method")
    
    plt.show()


if __name__ == "__main__":
    main()
