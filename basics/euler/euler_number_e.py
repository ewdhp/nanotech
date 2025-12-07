"""
Euler's Number e ≈ 2.71828182845904523536...

The base of natural logarithms and one of the most important constants in mathematics.

Definitions:
1. Limit definition: e = lim_{n→∞} (1 + 1/n)^n
2. Series definition: e = ∑_{n=0}^∞ 1/n! = 1 + 1/1! + 1/2! + 1/3! + ...
3. Derivative definition: d/dx[e^x] = e^x (unique function that is its own derivative)
4. Compound interest: lim_{n→∞} (1 + r/n)^n = e^r

Properties:
• e is irrational (proven by Euler, 1737)
• e is transcendental (proven by Hermite, 1873)
• e = 2.718281828459045...
• ln(e) = 1 (by definition)
• e^(iπ) = -1 (Euler's identity)

Applications:
• Calculus (natural logarithm, derivatives, integrals)
• Probability & Statistics (exponential, normal distributions)
• Compound interest and growth/decay
• Physics (radioactive decay, RC circuits)
• Complex analysis
• Number theory

Historical Note:
Euler didn't discover e (it was known before), but he:
- Chose the letter 'e' (possibly for "exponential")
- Proved it was irrational
- Calculated it to many decimal places
- Showed its fundamental importance in mathematics

Author: First studied by Jacob Bernoulli (1683) in compound interest
       Extensively developed by Leonhard Euler (1707-1783)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import factorial
from decimal import Decimal, getcontext

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)


def compute_e_limit_definition(n_max=1000000):
    """
    Compute e using limit definition: e = lim_{n→∞} (1 + 1/n)^n
    """
    print("=" * 80)
    print("EULER'S NUMBER e: LIMIT DEFINITION")
    print("=" * 80)
    
    print("\nDefinition: e = lim_{n→∞} (1 + 1/n)^n")
    print("\nConvergence to e:")
    print(f"{'n':>12s}  {'(1 + 1/n)^n':>25s}  {'Error':>15s}")
    print("-" * 80)
    
    e_exact = np.e
    
    n_values = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    results = []
    
    for n in n_values:
        if n <= 1000000:
            approx = (1 + 1/n)**n
        else:
            # Use logarithm for very large n to avoid overflow
            approx = np.exp(n * np.log(1 + 1/n))
        
        error = abs(approx - e_exact)
        results.append((n, approx, error))
        print(f"{n:12d}  {approx:25.15f}  {error:15.2e}")
    
    print(f"\nExact value: e = {e_exact:.15f}")
    
    return results


def compute_e_series_definition(n_terms=50):
    """
    Compute e using series: e = ∑_{n=0}^∞ 1/n!
    """
    print("\n" + "=" * 80)
    print("EULER'S NUMBER e: SERIES DEFINITION")
    print("=" * 80)
    
    print("\nDefinition: e = ∑_{n=0}^∞ 1/n! = 1 + 1 + 1/2 + 1/6 + 1/24 + ...")
    print("\nPartial sums:")
    print(f"{'n':>5s}  {'1/n!':>25s}  {'Partial Sum':>25s}  {'Error':>15s}")
    print("-" * 80)
    
    e_exact = np.e
    partial_sum = 0
    
    for n in range(min(n_terms, 21)):  # Show first 21 terms in detail
        term = 1 / np.math.factorial(n)
        partial_sum += term
        error = abs(partial_sum - e_exact)
        
        if n < 21:
            print(f"{n:5d}  {term:25.15e}  {partial_sum:25.15f}  {error:15.2e}")
    
    # Show convergence for larger n
    if n_terms > 21:
        print("  ...")
        for n in [25, 30, 40, 50]:
            if n <= n_terms:
                partial_sum = sum(1/np.math.factorial(k) for k in range(n+1))
                error = abs(partial_sum - e_exact)
                print(f"{n:5d}  {'...':>25s}  {partial_sum:25.15f}  {error:15.2e}")
    
    print(f"\nExact value: e = {e_exact:.15f}")
    
    # High precision computation
    print("\n" + "-" * 80)
    print("High-precision computation (100 decimal places):")
    getcontext().prec = 105
    
    e_decimal = Decimal(0)
    for n in range(100):
        e_decimal += Decimal(1) / Decimal(factorial(n, exact=True))
    
    print(f"e = {e_decimal}")


def demonstrate_properties():
    """
    Demonstrate key properties of e
    """
    print("\n" + "=" * 80)
    print("KEY PROPERTIES OF e")
    print("=" * 80)
    
    print("\n1. DERIVATIVE PROPERTY")
    print("-" * 80)
    print("   d/dx[e^x] = e^x")
    print("   This is the UNIQUE function that equals its own derivative!")
    
    # Numerical verification
    def numerical_derivative(f, x, h=1e-8):
        return (f(x + h) - f(x - h)) / (2 * h)
    
    x_test = 2.0
    f = lambda x: np.exp(x)
    analytical = np.exp(x_test)
    numerical = numerical_derivative(f, x_test)
    
    print(f"\n   At x = {x_test}:")
    print(f"   e^x = {analytical:.15f}")
    print(f"   d/dx[e^x] = {numerical:.15f}")
    print(f"   Difference: {abs(analytical - numerical):.2e}")
    
    print("\n2. INTEGRAL PROPERTY")
    print("-" * 80)
    print("   ∫e^x dx = e^x + C")
    print("   ∫₀¹ e^x dx = e - 1")
    
    integral_exact = np.e - 1
    integral_numerical = np.trapz(np.exp(np.linspace(0, 1, 10000)), 
                                  np.linspace(0, 1, 10000))
    print(f"\n   Exact: e - 1 = {integral_exact:.15f}")
    print(f"   Numerical: {integral_numerical:.15f}")
    print(f"   Difference: {abs(integral_exact - integral_numerical):.2e}")
    
    print("\n3. LOGARITHM PROPERTY")
    print("-" * 80)
    print("   ln(e) = 1 (by definition)")
    print("   e^(ln(x)) = x")
    print("   ln(e^x) = x")
    
    for x in [1, 2, 5, 10, 100]:
        result1 = np.exp(np.log(x))
        result2 = np.log(np.exp(x))
        print(f"\n   x = {x:3d}: e^(ln(x)) = {result1:.10f}, ln(e^x) = {result2:.10f}")
    
    print("\n4. IRRATIONALITY")
    print("-" * 80)
    print("   e cannot be expressed as a ratio of integers")
    print("   Proven by Euler in 1737")
    print("   Continued fraction: e = [2; 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, ...]")
    
    print("\n5. TRANSCENDENCE")
    print("-" * 80)
    print("   e is not the root of any polynomial with integer coefficients")
    print("   Proven by Charles Hermite in 1873")
    print("   This is a stronger property than irrationality")
    
    print("\n6. RELATIONSHIP TO π")
    print("-" * 80)
    print("   e^(iπ) + 1 = 0 (Euler's identity)")
    print("   e^(iπ) = -1")
    
    result = np.exp(1j * np.pi)
    print(f"\n   e^(iπ) = {result}")
    print(f"   e^(iπ) + 1 = {result + 1}")
    print(f"   |e^(iπ) + 1| = {abs(result + 1):.2e}")


def demonstrate_applications():
    """
    Show practical applications of e
    """
    print("\n" + "=" * 80)
    print("APPLICATIONS OF e")
    print("=" * 80)
    
    print("\n1. COMPOUND INTEREST")
    print("-" * 80)
    print("   Formula: A = P(1 + r/n)^(nt)")
    print("   Continuous compounding: A = Pe^(rt)")
    
    P = 1000  # Principal
    r = 0.05  # 5% annual rate
    t = 10    # 10 years
    
    print(f"\n   Principal: ${P}, Rate: {r*100}%, Time: {t} years")
    print(f"\n   {'Compounding':20s}  {'n':>10s}  {'Final Amount':>15s}")
    print("   " + "-" * 60)
    
    for freq, n in [("Annually", 1), ("Quarterly", 4), ("Monthly", 12), 
                    ("Daily", 365), ("Hourly", 8760), ("Continuous", None)]:
        if n is None:
            A = P * np.exp(r * t)
            print(f"   {freq:20s}  {'∞':>10s}  ${A:14.2f}")
        else:
            A = P * (1 + r/n)**(n*t)
            print(f"   {freq:20s}  {n:10d}  ${A:14.2f}")
    
    print("\n2. EXPONENTIAL GROWTH/DECAY")
    print("-" * 80)
    print("   Growth: N(t) = N₀·e^(kt), k > 0")
    print("   Decay: N(t) = N₀·e^(-kt), k > 0")
    
    # Population growth
    N0 = 1000
    k = 0.1
    t_vals = [0, 5, 10, 15, 20]
    
    print(f"\n   Population growth (N₀={N0}, k={k}):")
    for t in t_vals:
        N = N0 * np.exp(k * t)
        print(f"   t = {t:2d} years: N = {N:10.1f}")
    
    # Radioactive decay (Half-life)
    print("\n   Radioactive decay (Half-life = 5 years):")
    half_life = 5
    k_decay = np.log(2) / half_life
    
    for t in [0, 5, 10, 15, 20]:
        N = N0 * np.exp(-k_decay * t)
        print(f"   t = {t:2d} years: N = {N:10.1f} ({N/N0*100:.1f}% remaining)")
    
    print("\n3. NORMAL DISTRIBUTION")
    print("-" * 80)
    print("   PDF: f(x) = (1/σ√(2π))·e^(-(x-μ)²/(2σ²))")
    print("   The bell curve used throughout statistics")
    
    print("\n4. PROBABILITY - Derangements")
    print("-" * 80)
    print("   P(no element in original position) → 1/e as n → ∞")
    
    for n in [5, 10, 20, 50, 100]:
        # Derangement formula: D(n) ≈ n!/e
        p_approx = 1 / np.e
        print(f"   n = {n:3d}: P(derangement) ≈ {p_approx:.10f}")
    
    print(f"\n   Limit: 1/e = {1/np.e:.15f}")


def demonstrate_number_theory():
    """
    Show number theory properties
    """
    print("\n" + "=" * 80)
    print("NUMBER THEORY PROPERTIES")
    print("=" * 80)
    
    print("\n1. CONTINUED FRACTION EXPANSION")
    print("-" * 80)
    print("   e = [2; 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, 1, 1, 10, ...]")
    print("   Pattern: [2; 1, 2k, 1] for k = 1, 2, 3, ...")
    
    # First few convergents
    convergents = [
        (2, 1),      # 2/1
        (3, 1),      # 3/1
        (8, 3),      # 8/3
        (11, 4),     # 11/4
        (19, 7),     # 19/7
        (87, 32),    # 87/32
        (106, 39),   # 106/39
        (193, 71),   # 193/71
    ]
    
    print("\n   Convergents (increasingly accurate rational approximations):")
    print(f"   {'p/q':>10s}  {'Value':>20s}  {'Error':>15s}")
    print("   " + "-" * 60)
    
    for p, q in convergents:
        value = p / q
        error = abs(value - np.e)
        print(f"   {p:4d}/{q:<4d}  {value:20.15f}  {error:15.2e}")
    
    print("\n2. REPRESENTATIONS")
    print("-" * 80)
    print("   e = ∑_{n=0}^∞ 1/n!")
    print("   e = lim_{n→∞} (1 + 1/n)^n")
    print("   e = lim_{n→∞} (n/ln(n))^(1/n) = 1/ln(lim_{n→∞} n^(1/n))")
    
    print("\n3. SPECIAL VALUES")
    print("-" * 80)
    
    special = [
        ("e^0", 0, "1"),
        ("e^1", 1, "e"),
        ("e^(-1)", -1, "1/e"),
        ("e^(iπ)", complex(0, np.pi), "-1"),
        ("e^(2πi)", complex(0, 2*np.pi), "1"),
    ]
    
    for expr, x, description in special:
        if isinstance(x, complex):
            result = np.exp(x)
        else:
            result = np.exp(x)
        print(f"   {expr:10s} = {result:20s}  ({description})")


def create_visualizations():
    """
    Create comprehensive visualizations
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Limit convergence
    ax1 = plt.subplot(3, 3, 1)
    n_vals = np.logspace(0, 7, 1000)
    limit_vals = (1 + 1/n_vals)**n_vals
    
    ax1.semilogx(n_vals, limit_vals, 'b-', linewidth=2)
    ax1.axhline(y=np.e, color='r', linestyle='--', linewidth=2, label=f'e = {np.e:.6f}')
    ax1.fill_between(n_vals, limit_vals, np.e, alpha=0.3, color='yellow')
    ax1.set_xlabel('n', fontsize=11)
    ax1.set_ylabel('(1 + 1/n)^n', fontsize=11)
    ax1.set_title('Limit Definition Convergence', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Series convergence
    ax2 = plt.subplot(3, 3, 2)
    n_series = np.arange(0, 21)
    partial_sums = np.cumsum([1/np.math.factorial(n) for n in n_series])
    
    ax2.plot(n_series, partial_sums, 'go-', linewidth=2, markersize=6)
    ax2.axhline(y=np.e, color='r', linestyle='--', linewidth=2, label='e')
    ax2.fill_between(n_series, partial_sums, np.e, alpha=0.3, color='cyan')
    ax2.set_xlabel('n (number of terms)', fontsize=11)
    ax2.set_ylabel('∑ 1/k!', fontsize=11)
    ax2.set_title('Series Definition: ∑1/n!', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Exponential function
    ax3 = plt.subplot(3, 3, 3)
    x = np.linspace(-2, 3, 1000)
    y = np.exp(x)
    
    ax3.plot(x, y, 'b-', linewidth=2.5, label='y = e^x')
    ax3.plot(x, x, 'g--', linewidth=2, label='y = x (tangent at x=0)', alpha=0.7)
    ax3.plot([0, 1], [1, np.e], 'ro', markersize=10)
    ax3.text(0, 1, '  (0, 1)', fontsize=10)
    ax3.text(1, np.e, f'  (1, e)', fontsize=10)
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.axvline(x=0, color='k', linewidth=0.5)
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('y', fontsize=11)
    ax3.set_title('Exponential Function y = e^x', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-0.5, 20])
    
    # 4. Natural logarithm
    ax4 = plt.subplot(3, 3, 4)
    x_ln = np.linspace(0.01, 5, 1000)
    y_ln = np.log(x_ln)
    
    ax4.plot(x_ln, y_ln, 'r-', linewidth=2.5, label='y = ln(x)')
    ax4.plot([1, np.e], [0, 1], 'go', markersize=10)
    ax4.text(1, 0, '(1, 0)  ', fontsize=10, ha='right')
    ax4.text(np.e, 1, f'  (e, 1)', fontsize=10)
    ax4.axhline(y=0, color='k', linewidth=0.5)
    ax4.axvline(x=0, color='k', linewidth=0.5)
    ax4.axvline(x=np.e, color='orange', linestyle='--', alpha=0.5, label='x = e')
    ax4.set_xlabel('x', fontsize=11)
    ax4.set_ylabel('y', fontsize=11)
    ax4.set_title('Natural Logarithm y = ln(x)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Compound interest comparison
    ax5 = plt.subplot(3, 3, 5)
    t = np.linspace(0, 20, 1000)
    P, r = 1000, 0.05
    
    compounds = {
        'Annual (n=1)': P * (1 + r/1)**(1*t),
        'Monthly (n=12)': P * (1 + r/12)**(12*t),
        'Daily (n=365)': P * (1 + r/365)**(365*t),
        'Continuous': P * np.exp(r*t)
    }
    
    colors = ['blue', 'green', 'orange', 'red']
    for (label, amount), color in zip(compounds.items(), colors):
        linestyle = '--' if 'Continuous' in label else '-'
        linewidth = 3 if 'Continuous' in label else 2
        ax5.plot(t, amount, linestyle=linestyle, linewidth=linewidth, 
                label=label, color=color)
    
    ax5.set_xlabel('Time (years)', fontsize=11)
    ax5.set_ylabel('Amount ($)', fontsize=11)
    ax5.set_title(f'Compound Interest (P=${P}, r={r*100}%)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Exponential growth and decay
    ax6 = plt.subplot(3, 3, 6)
    t_exp = np.linspace(0, 10, 1000)
    
    ax6.plot(t_exp, np.exp(0.3*t_exp), 'b-', linewidth=2, label='Growth: e^(0.3t)')
    ax6.plot(t_exp, np.exp(-0.3*t_exp), 'r-', linewidth=2, label='Decay: e^(-0.3t)')
    ax6.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Time t', fontsize=11)
    ax6.set_ylabel('N(t)', fontsize=11)
    ax6.set_title('Exponential Growth & Decay', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # 7. Normal distribution
    ax7 = plt.subplot(3, 3, 7)
    x_norm = np.linspace(-4, 4, 1000)
    
    for sigma in [0.5, 1.0, 1.5]:
        pdf = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-x_norm**2 / (2*sigma**2))
        ax7.plot(x_norm, pdf, linewidth=2, label=f'σ = {sigma}')
    
    ax7.set_xlabel('x', fontsize=11)
    ax7.set_ylabel('Probability Density', fontsize=11)
    ax7.set_title('Normal Distribution (μ=0)', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # 8. Convergence comparison
    ax8 = plt.subplot(3, 3, 8)
    n_comp = np.logspace(0, 6, 100)
    
    limit_method = np.abs((1 + 1/n_comp)**n_comp - np.e)
    
    # Series method
    series_errors = []
    for n in n_comp:
        n_int = int(n)
        series_sum = sum(1/np.math.factorial(k) for k in range(min(n_int, 170)))
        series_errors.append(abs(series_sum - np.e))
    
    ax8.loglog(n_comp, limit_method, 'b-', linewidth=2, label='Limit method')
    ax8.loglog(n_comp, series_errors, 'r-', linewidth=2, label='Series method')
    ax8.set_xlabel('n', fontsize=11)
    ax8.set_ylabel('|Error|', fontsize=11)
    ax8.set_title('Convergence Rate Comparison', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3, which='both')
    
    # 9. Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary = f"""
EULER'S NUMBER e
═══════════════════════════

e ≈ 2.71828182845904523536...

Definitions:
────────────
• lim(1 + 1/n)^n as n→∞
• ∑1/n! from n=0 to ∞
• d/dx[e^x] = e^x

Properties:
───────────
• Irrational
• Transcendental
• ln(e) = 1
• e^(iπ) = -1

Applications:
─────────────
• Compound interest
• Population growth
• Radioactive decay
• Normal distribution
• Calculus
• Differential equations

Special Values:
───────────────
• e^0 = 1
• e^1 = e
• e^(-1) = 1/e ≈ 0.368
• e^(ln(x)) = x
• ln(e^x) = x
═══════════════════════════
    """
    
    ax9.text(0.05, 0.95, summary, transform=ax9.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle("Euler's Number e: The Base of Natural Logarithms", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def main():
    """
    Main demonstration
    """
    print("\n" + "=" * 80)
    print(" EULER'S NUMBER e ≈ 2.71828182845904523536...")
    print(" The Base of Natural Logarithms")
    print("=" * 80)
    
    # Compute using different methods
    compute_e_limit_definition()
    compute_e_series_definition()
    
    # Properties
    demonstrate_properties()
    
    # Applications
    demonstrate_applications()
    
    # Number theory
    demonstrate_number_theory()
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    
    # Create visualizations
    fig = create_visualizations()
    
    print("\n✓ Visualizations created successfully!")
    print("\nKey Insights:")
    print("  • e is the unique number where d/dx[e^x] = e^x")
    print("  • Appears naturally in continuous growth/decay processes")
    print("  • Foundation of calculus and analysis")
    print("  • Connects to compound interest and probability")
    print("  • One of the most important constants in mathematics")
    
    plt.show()


if __name__ == "__main__":
    main()
