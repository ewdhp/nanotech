"""
Euler's Derivation of π²/6 (Basel Problem Solution)

This script demonstrates Leonhard Euler's famous solution to the Basel problem:
    ∑(1/n²) from n=1 to ∞ = π²/6

Euler's approach (1734):
1. Start with Taylor series: sin(x)/x = 1 - x²/3! + x⁴/5! - x⁶/7! + ...
2. Treat this as a polynomial with roots at x = ±π, ±2π, ±3π, ...
3. Factor as infinite product: sin(x)/x = (1 - x²/π²)(1 - x²/4π²)(1 - x²/9π²)...
4. Compare coefficients of x² to get: -1/3! = -1/π² - 1/4π² - 1/9π² - ...
5. Therefore: 1 + 1/4 + 1/9 + 1/16 + ... = π²/6

Note: Euler's reasoning was later justified by Weierstrass (factorization theorem)

Reference:
- Euler, Leonhard (1734). "De summis serierum reciprocarum"
- Weierstrass factorization theorem (1876)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import seaborn as sns
from fractions import Fraction

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def taylor_sin_over_x(x, n_terms=10):
    """
    Taylor series expansion of sin(x)/x
    
    sin(x)/x = 1 - x²/3! + x⁴/5! - x⁶/7! + ...
    
    Parameters:
    -----------
    x : float or array
        Input value(s)
    n_terms : int
        Number of terms in the series
    
    Returns:
    --------
    result : float or array
        Value of sin(x)/x approximation
    """
    result = 0
    for n in range(n_terms):
        # Term: (-1)^n * x^(2n) / (2n+1)!
        term = ((-1)**n * x**(2*n)) / factorial(2*n + 1)
        result += term
    return result


def euler_infinite_product(x, n_terms=10):
    """
    Euler's infinite product representation of sin(x)/x
    
    sin(x)/x = ∏(1 - x²/(n²π²)) for n = 1, 2, 3, ...
    
    This was Euler's key insight! He treated sin(x)/x as a polynomial
    with roots at x = ±nπ for all integers n ≠ 0.
    
    Parameters:
    -----------
    x : float or array
        Input value(s)
    n_terms : int
        Number of factors in the product
    
    Returns:
    --------
    result : float or array
        Value of the infinite product approximation
    """
    result = 1.0
    for n in range(1, n_terms + 1):
        factor = 1 - (x**2) / (n**2 * np.pi**2)
        result *= factor
    return result


def compute_basel_sum(n_terms):
    """
    Compute partial sum of Basel series: ∑(1/n²)
    
    Parameters:
    -----------
    n_terms : int
        Number of terms to sum
    
    Returns:
    --------
    partial_sum : float
        Sum of first n_terms
    """
    return np.sum(1.0 / np.arange(1, n_terms + 1)**2)


def euler_coefficient_comparison():
    """
    Demonstrate Euler's coefficient comparison method
    
    Taylor series:   sin(x)/x = 1 - x²/3! + x⁴/5! - ...
    Infinite product: sin(x)/x = (1 - x²/π²)(1 - x²/4π²)(1 - x²/9π²)...
    
    Expanding the product and comparing coefficients of x²:
    Taylor: coefficient = -1/6
    Product: coefficient = -(1/π² + 1/4π² + 1/9π² + ...)
    
    Therefore: 1/6 = 1/π² + 1/4π² + 1/9π² + ...
    Multiply by π²: π²/6 = 1 + 1/4 + 1/9 + ...
    """
    print("=" * 80)
    print("EULER'S COEFFICIENT COMPARISON METHOD")
    print("=" * 80)
    
    # Taylor series coefficient of x²
    taylor_coeff = -1.0 / factorial(3)
    print(f"\nTaylor series: sin(x)/x = 1 - x²/3! + x⁴/5! - ...")
    print(f"Coefficient of x² in Taylor series: {taylor_coeff}")
    print(f"                                   = -1/6 = {taylor_coeff:.10f}")
    
    # Infinite product expansion
    print(f"\nInfinite product: sin(x)/x = (1 - x²/π²)(1 - x²/4π²)(1 - x²/9π²)...")
    print(f"\nExpanding the product:")
    print(f"  = 1 - x²(1/π² + 1/4π² + 1/9π² + ...) + higher order terms")
    
    # Show partial sums
    print(f"\nCoefficient of x² in infinite product:")
    partial_products = []
    for n in [5, 10, 50, 100, 500]:
        coeff = -sum(1.0 / (k**2 * np.pi**2) for k in range(1, n + 1))
        partial_products.append((n, coeff))
        print(f"  Using {n:3d} terms: {coeff:.10f}")
    
    print(f"\nThey match! Therefore:")
    print(f"  -1/6 = -(1/π² + 1/4π² + 1/9π² + ...)")
    print(f"   1/6 =  (1/π² + 1/4π² + 1/9π² + ...)")
    print(f"   1/6 = (1/π²)(1 + 1/4 + 1/9 + ...)")
    print(f"\nMultiplying both sides by π²:")
    print(f"  π²/6 = 1 + 1/4 + 1/9 + 1/16 + ...")
    print(f"  π²/6 = ∑(1/n²) from n=1 to ∞")
    
    return partial_products


def verify_basel_numerically():
    """
    Verify Euler's result numerically by computing partial sums
    This is how Euler gained confidence in his result!
    """
    print("\n" + "=" * 80)
    print("NUMERICAL VERIFICATION (As Euler Did)")
    print("=" * 80)
    
    pi_squared_over_6 = (np.pi**2) / 6
    print(f"\nExact value: π²/6 = {pi_squared_over_6:.15f}")
    print(f"\nPartial sums of ∑(1/n²):")
    print(f"{'Terms':>10} {'Partial Sum':>20} {'Error':>20} {'% Error':>15}")
    print("-" * 80)
    
    convergence_data = []
    for n in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]:
        partial_sum = compute_basel_sum(n)
        error = abs(partial_sum - pi_squared_over_6)
        percent_error = (error / pi_squared_over_6) * 100
        convergence_data.append((n, partial_sum, error, percent_error))
        print(f"{n:10d} {partial_sum:20.15f} {error:20.15e} {percent_error:14.10f}%")
    
    print(f"\nConclusion: The series converges to π²/6 ≈ {pi_squared_over_6:.10f}")
    return convergence_data


def demonstrate_visualizations():
    """
    Create comprehensive visualizations of Euler's method
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Taylor Series vs Infinite Product
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.linspace(-3*np.pi, 3*np.pi, 1000)
    x_nonzero = x[x != 0]
    
    # Actual sin(x)/x
    y_exact = np.sin(x_nonzero) / x_nonzero
    
    # Taylor approximation
    y_taylor = taylor_sin_over_x(x_nonzero, n_terms=15)
    
    # Euler's infinite product
    y_euler = euler_infinite_product(x_nonzero, n_terms=20)
    
    ax1.plot(x_nonzero / np.pi, y_exact, 'k-', linewidth=2.5, label='sin(x)/x (exact)', zorder=3)
    ax1.plot(x_nonzero / np.pi, y_taylor, 'b--', linewidth=2, label='Taylor series (15 terms)', alpha=0.7)
    ax1.plot(x_nonzero / np.pi, y_euler, 'r--', linewidth=2, label='Euler product (20 terms)', alpha=0.7)
    
    # Mark zeros
    zeros = np.array([-3, -2, -1, 1, 2, 3])
    ax1.plot(zeros, np.zeros_like(zeros), 'ro', markersize=10, label='Roots at ±nπ', zorder=4)
    
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('x/π', fontsize=12)
    ax1.set_ylabel('sin(x)/x', fontsize=12)
    ax1.set_title("Euler's Insight: sin(x)/x as Infinite Product with Roots at ±nπ", 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.5, 1.2])
    
    # 2. Convergence of Basel Series
    ax2 = fig.add_subplot(gs[0, 2])
    n_values = np.arange(1, 1001)
    partial_sums = np.array([compute_basel_sum(n) for n in n_values])
    pi_squared_over_6 = (np.pi**2) / 6
    
    ax2.plot(n_values, partial_sums, 'b-', linewidth=2, label='Partial sums')
    ax2.axhline(y=pi_squared_over_6, color='r', linestyle='--', linewidth=2, 
                label=f'π²/6 = {pi_squared_over_6:.6f}')
    ax2.fill_between(n_values, partial_sums, pi_squared_over_6, alpha=0.3, color='yellow')
    ax2.set_xlabel('Number of terms', fontsize=11)
    ax2.set_ylabel('∑(1/n²)', fontsize=11)
    ax2.set_title('Convergence to π²/6', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Error Analysis (Log Scale)
    ax3 = fig.add_subplot(gs[1, 0])
    errors = np.abs(partial_sums - pi_squared_over_6)
    ax3.semilogy(n_values, errors, 'r-', linewidth=2)
    ax3.set_xlabel('Number of terms', fontsize=11)
    ax3.set_ylabel('Absolute Error', fontsize=11)
    ax3.set_title('Convergence Error (Log Scale)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.text(500, 1e-2, 'Slow convergence:\nO(1/n)', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 4. Individual Terms of Series
    ax4 = fig.add_subplot(gs[1, 1])
    n_range = np.arange(1, 51)
    terms = 1.0 / n_range**2
    ax4.bar(n_range, terms, color='blue', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('n', fontsize=11)
    ax4.set_ylabel('1/n²', fontsize=11)
    ax4.set_title('Individual Terms: 1/n²', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations for first few terms
    for i in range(min(5, len(n_range))):
        height = terms[i]
        ax4.text(n_range[i], height + 0.02, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=8)
    
    # 5. Coefficient Comparison
    ax5 = fig.add_subplot(gs[1, 2])
    n_factors = np.arange(1, 101)
    taylor_x2_coeff = -1.0 / 6
    product_coeffs = [-sum(1.0/(k**2 * np.pi**2) for k in range(1, n+1)) for n in n_factors]
    
    ax5.plot(n_factors, product_coeffs, 'g-', linewidth=2, label='Product expansion')
    ax5.axhline(y=taylor_x2_coeff, color='r', linestyle='--', linewidth=2, 
                label=f'Taylor: -1/6')
    ax5.fill_between(n_factors, product_coeffs, taylor_x2_coeff, alpha=0.3, color='cyan')
    ax5.set_xlabel('Number of factors', fontsize=11)
    ax5.set_ylabel('Coefficient of x²', fontsize=11)
    ax5.set_title('Coefficient Matching', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Comparison with Related Sums
    ax6 = fig.add_subplot(gs[2, 0])
    exponents = [1, 2, 3, 4]
    exact_values = [np.nan, np.pi**2/6, 1.202057, np.pi**4/90]  # ζ(1) diverges
    colors = ['red', 'green', 'blue', 'orange']
    
    for i, (exp, exact, color) in enumerate(zip(exponents, exact_values, colors)):
        if exp == 1:
            # Harmonic series diverges
            n = np.arange(1, 101)
            partial = np.cumsum(1.0 / n**exp)
            ax6.plot(n, partial, color=color, linewidth=2, label=f'∑1/n (diverges)', linestyle='--')
        else:
            n = np.arange(1, 101)
            partial = np.cumsum(1.0 / n**exp)
            ax6.plot(n, partial, color=color, linewidth=2, label=f'∑1/n^{exp} → {exact:.4f}')
            ax6.axhline(y=exact, color=color, linestyle=':', alpha=0.5)
    
    ax6.set_xlabel('Number of terms', fontsize=11)
    ax6.set_ylabel('Partial sum', fontsize=11)
    ax6.set_title('Related Zeta Functions: ζ(s) = ∑1/nˢ', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 6])
    
    # 7. Euler Product Convergence
    ax7 = fig.add_subplot(gs[2, 1])
    x_test = 1.0  # Test at x = 1
    n_prod_terms = np.arange(1, 51)
    product_values = [euler_infinite_product(x_test, n) for n in n_prod_terms]
    exact_value = np.sin(x_test) / x_test
    
    ax7.plot(n_prod_terms, product_values, 'purple', linewidth=2, marker='o', 
             markersize=4, label='Product approximation')
    ax7.axhline(y=exact_value, color='red', linestyle='--', linewidth=2, 
                label=f'sin(1)/1 = {exact_value:.6f}')
    ax7.fill_between(n_prod_terms, product_values, exact_value, alpha=0.3, color='lavender')
    ax7.set_xlabel('Number of product factors', fontsize=11)
    ax7.set_ylabel('Value at x=1', fontsize=11)
    ax7.set_title('Infinite Product Convergence', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # 8. Historical Timeline
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    timeline_text = """
    HISTORICAL CONTEXT
    ══════════════════════════════════
    
    1650s: Pietro Mengoli poses the
           Basel Problem
    
    1734:  Leonhard Euler solves it!
           ∑(1/n²) = π²/6
           
           Method: Infinite product
           representation of sin(x)
           
           "Bold assumption" about
           infinite series
    
    1741:  Euler announces result
    
    1821:  Cauchy begins rigorous
           analysis of series
    
    1876:  Karl Weierstrass proves
           Euler's method is valid
           (Factorization theorem)
    
    ══════════════════════════════════
    Euler was RIGHT for 142 years
    before the proof was rigorous!
    """
    
    ax8.text(0.1, 0.95, timeline_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle("Euler's Solution to the Basel Problem: ∑(1/n²) = π²/6", 
                 fontsize=18, fontweight='bold', y=0.995)
    
    return fig


def show_first_terms_explicitly():
    """
    Show the first few terms of the Basel series explicitly
    """
    print("\n" + "=" * 80)
    print("BASEL SERIES - FIRST TERMS")
    print("=" * 80)
    
    print("\n∑(1/n²) = 1/1² + 1/2² + 1/3² + 1/4² + 1/5² + ...")
    print("\nFirst 10 terms:")
    
    total = 0
    for n in range(1, 11):
        term = 1.0 / n**2
        total += term
        frac = Fraction(1, n**2)
        print(f"  n={n:2d}: 1/{n:2d}² = {frac} = {term:.10f}  (running sum: {total:.10f})")
    
    print(f"\nSum of first 10 terms: {total:.15f}")
    print(f"π²/6                 : {(np.pi**2)/6:.15f}")
    print(f"Difference           : {abs(total - (np.pi**2)/6):.15e}")


def demonstrate_weierstrass_justification():
    """
    Explain why Weierstrass's theorem justifies Euler's approach
    """
    print("\n" + "=" * 80)
    print("WEIERSTRASS FACTORIZATION THEOREM - WHY EULER WAS RIGHT")
    print("=" * 80)
    
    print("""
The Weierstrass Factorization Theorem (1876) states:

    Any entire function f(z) can be represented as an infinite product
    involving its zeros.
    
For sin(x):
    • sin(x) is entire (analytic everywhere in ℂ)
    • Zeros at x = nπ for all integers n
    
Therefore, sin(x) CAN be written as:

    sin(x) = x ∏(1 - x²/(n²π²))  for n = 1, 2, 3, ...
             n=1

Dividing by x:

    sin(x)/x = ∏(1 - x²/(n²π²))
               n=1

This JUSTIFIES Euler's bold assumption from 1734!

Euler's Intuition vs. Rigor:
    • Euler (1734): "It seems reasonable to treat sin(x)/x like a polynomial"
    • Mathematicians: "You can't just do that with infinite series!"
    • Weierstrass (1876): "Actually, he was right. Here's the proof."
    
Timeline: Euler was vindicated 142 years later!
    """)


def main():
    """
    Main demonstration of Euler's derivation
    """
    print("\n" + "=" * 80)
    print("EULER'S SOLUTION TO THE BASEL PROBLEM (1734)")
    print("Derivation of ∑(1/n²) = π²/6")
    print("=" * 80)
    
    # Show first terms explicitly
    show_first_terms_explicitly()
    
    # Demonstrate coefficient comparison
    euler_coefficient_comparison()
    
    # Numerical verification
    convergence_data = verify_basel_numerically()
    
    # Explain Weierstrass justification
    demonstrate_weierstrass_justification()
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    
    # Create visualizations
    fig = demonstrate_visualizations()
    
    print("\n✓ All visualizations created successfully!")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
1. EULER'S GENIUS: Treated infinite series like polynomials
   
2. METHOD: Match coefficients between two representations
   • Taylor: sin(x)/x = 1 - x²/3! + ...
   • Product: sin(x)/x = (1 - x²/π²)(1 - x²/4π²)...
   
3. RESULT: Coefficient of x² → π²/6 = ∑(1/n²)

4. CONFIDENCE: Numerical verification gave Euler certainty

5. VINDICATION: Weierstrass proved it rigorous 142 years later!

6. LESSON: Mathematical intuition can precede rigorous proof
    """)
    
    print("=" * 80)
    
    plt.show()


if __name__ == "__main__":
    main()
