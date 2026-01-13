#!/usr/bin/env python3
"""
Euler-Maclaurin Formula with Bernoulli Numbers
===============================================

The Euler-Maclaurin formula provides a powerful connection between sums and integrals,
with correction terms involving Bernoulli numbers.

Standard form:
    Σ(k=a to b) f(k) = ∫(a to b) f(x)dx + [f(a) + f(b)]/2 
                       + Σ(k=1 to p) [B_(2k)/(2k)!] * [f^(2k-1)(b) - f^(2k-1)(a)]
                       + R_p

Where:
- B_(2k) are Bernoulli numbers
- f^(2k-1) denotes the (2k-1)th derivative
- R_p is the remainder term

Applications:
1. Approximating sums by integrals with high precision
2. Deriving asymptotic expansions (e.g., Stirling's formula)
3. Computing definite integrals numerically
4. Analyzing convergence of series
5. Number theory (Riemann zeta function, etc.)

The formula shows that the difference between a sum and an integral can be
expressed exactly using Bernoulli numbers.
"""

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from math import factorial, pi, log, sqrt, exp
from typing import Callable, List, Tuple
from scipy import integrate


def bernoulli_number(n: int) -> Fraction:
    """
    Calculate the nth Bernoulli number using recursive formula.
    
    Args:
        n: Index of the Bernoulli number (n >= 0)
    
    Returns:
        The nth Bernoulli number as a Fraction
    """
    if n == 0:
        return Fraction(1)
    if n == 1:
        return Fraction(-1, 2)
    if n > 1 and n % 2 == 1:
        return Fraction(0)
    
    B = [Fraction(0)] * (n + 1)
    B[0] = Fraction(1)
    
    for m in range(1, n + 1):
        sum_val = Fraction(0)
        for k in range(m):
            binom = factorial(m + 1) // (factorial(k) * factorial(m + 1 - k))
            sum_val += binom * B[k]
        B[m] = -sum_val / (m + 1)
    
    return B[n]


def numerical_derivative(f: Callable, x: float, n: int, h: float = 1e-5) -> float:
    """
    Compute the nth derivative of f at x numerically using finite differences.
    
    Args:
        f: Function to differentiate
        x: Point at which to evaluate derivative
        n: Order of derivative
        h: Step size for finite differences
    
    Returns:
        Approximate nth derivative
    """
    if n == 0:
        return f(x)
    elif n == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    elif n == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
    else:
        # Use recursive finite difference
        df = lambda t: numerical_derivative(f, t, n - 1, h)
        return (df(x + h) - df(x - h)) / (2 * h)


def euler_maclaurin_sum(f: Callable, a: int, b: int, num_terms: int = 5) -> float:
    """
    Approximate a sum using the Euler-Maclaurin formula.
    
    Args:
        f: Function to sum
        a: Start of summation
        b: End of summation
        num_terms: Number of Bernoulli correction terms to use
    
    Returns:
        Approximate value of Σ(k=a to b) f(k)
    """
    # Integral term
    integral, _ = integrate.quad(f, a, b)
    
    # Endpoint terms
    endpoint = (f(a) + f(b)) / 2
    
    # Bernoulli correction terms
    correction = 0.0
    for k in range(1, num_terms + 1):
        B_2k = float(bernoulli_number(2 * k))
        fact_2k = factorial(2 * k)
        
        # Derivatives at endpoints
        deriv_b = numerical_derivative(f, b, 2 * k - 1)
        deriv_a = numerical_derivative(f, a, 2 * k - 1)
        
        correction += (B_2k / fact_2k) * (deriv_b - deriv_a)
    
    return integral + endpoint + correction


def direct_sum(f: Callable, a: int, b: int) -> float:
    """
    Calculate the sum directly.
    
    Args:
        f: Function to sum
        a: Start of summation
        b: End of summation
    
    Returns:
        Exact value of Σ(k=a to b) f(k)
    """
    return sum(f(k) for k in range(a, b + 1))


def stirling_approximation_em(n: int, num_terms: int = 5) -> float:
    """
    Calculate log(n!) using Euler-Maclaurin formula (Stirling's approximation).
    
    log(n!) = Σ(k=1 to n) log(k)
    
    Args:
        n: Factorial argument
        num_terms: Number of Bernoulli terms
    
    Returns:
        Approximation of log(n!)
    """
    if n <= 0:
        return 0.0
    
    # Using Euler-Maclaurin on log function
    f = lambda x: np.log(x) if x > 0 else 0
    return euler_maclaurin_sum(f, 1, n, num_terms)


def stirling_formula(n: int) -> float:
    """
    Stirling's formula: ln(n!) ≈ n*ln(n) - n + 0.5*ln(2πn)
    
    Args:
        n: Factorial argument
    
    Returns:
        Stirling approximation of log(n!)
    """
    if n <= 0:
        return 0.0
    return n * log(n) - n + 0.5 * log(2 * pi * n)


def demonstrate_euler_maclaurin():
    """
    Demonstrate the Euler-Maclaurin formula with various examples.
    """
    print("=" * 80)
    print("EULER-MACLAURIN FORMULA WITH BERNOULLI NUMBERS")
    print("=" * 80)
    
    print("\n1. First few Bernoulli numbers used in the formula:")
    print("-" * 40)
    for k in range(1, 8):
        B_2k = bernoulli_number(2 * k)
        print(f"B_{2*k:2d} = {str(B_2k):>20s} = {float(B_2k):>20.15f}")
    
    print("\n2. Example: Σ(k=1 to n) k² (sum of squares)")
    print("-" * 40)
    print("Exact formula: n(n+1)(2n+1)/6")
    print()
    print(f"{'n':>6s} {'Direct':>15s} {'E-M (3 terms)':>15s} {'E-M (5 terms)':>15s} {'Error':>12s}")
    print("-" * 70)
    
    for n in [10, 50, 100, 500, 1000]:
        f = lambda x: x**2
        direct = direct_sum(f, 1, n)
        em_3 = euler_maclaurin_sum(f, 1, n, num_terms=3)
        em_5 = euler_maclaurin_sum(f, 1, n, num_terms=5)
        error = abs(direct - em_5)
        
        print(f"{n:>6d} {direct:>15.2f} {em_3:>15.2f} {em_5:>15.2f} {error:>12.2e}")
    
    print("\n3. Example: Σ(k=1 to n) k³ (sum of cubes)")
    print("-" * 40)
    print("Exact formula: [n(n+1)/2]²")
    print()
    print(f"{'n':>6s} {'Direct':>15s} {'E-M (3 terms)':>15s} {'E-M (5 terms)':>15s} {'Error':>12s}")
    print("-" * 70)
    
    for n in [10, 50, 100, 500, 1000]:
        f = lambda x: x**3
        direct = direct_sum(f, 1, n)
        em_3 = euler_maclaurin_sum(f, 1, n, num_terms=3)
        em_5 = euler_maclaurin_sum(f, 1, n, num_terms=5)
        error = abs(direct - em_5)
        
        print(f"{n:>6d} {direct:>15.2f} {em_3:>15.2f} {em_5:>15.2f} {error:>12.2e}")
    
    print("\n4. Example: Harmonic numbers H_n = Σ(k=1 to n) 1/k")
    print("-" * 40)
    print("Asymptotic: H_n ≈ ln(n) + γ (where γ is Euler-Mascheroni constant)")
    print()
    print(f"{'n':>6s} {'Direct':>15s} {'E-M (5 terms)':>15s} {'ln(n) + γ':>15s} {'EM Error':>12s}")
    print("-" * 75)
    
    gamma = 0.5772156649015329  # Euler-Mascheroni constant
    for n in [10, 50, 100, 500, 1000]:
        f = lambda x: 1/x if x > 0 else 0
        direct = direct_sum(f, 1, n)
        em_5 = euler_maclaurin_sum(f, 1, n, num_terms=5)
        asymptotic = log(n) + gamma
        error = abs(direct - em_5)
        
        print(f"{n:>6d} {direct:>15.8f} {em_5:>15.8f} {asymptotic:>15.8f} {error:>12.2e}")
    
    print("\n5. Stirling's Formula via Euler-Maclaurin")
    print("-" * 40)
    print("log(n!) = Σ(k=1 to n) log(k)")
    print("Stirling: log(n!) ≈ n*log(n) - n + 0.5*log(2πn)")
    print()
    print(f"{'n':>6s} {'Exact log(n!)':>18s} {'E-M Formula':>18s} {'Stirling':>18s} {'EM Error':>12s}")
    print("-" * 78)
    
    for n in [10, 20, 50, 100]:
        exact = sum(log(k) for k in range(1, n + 1))
        em = stirling_approximation_em(n, num_terms=5)
        stirling = stirling_formula(n)
        error = abs(exact - em)
        
        print(f"{n:>6d} {exact:>18.10f} {em:>18.10f} {stirling:>18.10f} {error:>12.2e}")


def demonstrate_correction_terms():
    """
    Show how Bernoulli correction terms improve accuracy.
    """
    print("\n6. Effect of increasing Bernoulli correction terms:")
    print("-" * 40)
    print("Example: Σ(k=1 to 100) 1/k²  (related to ζ(2) = π²/6)")
    print()
    
    n = 100
    f = lambda x: 1/x**2 if x > 0 else 0
    direct = direct_sum(f, 1, n)
    
    print(f"Direct sum: {direct:.15f}")
    print(f"ζ(2) - Σ(k>100) 1/k² ≈ {direct:.15f}")
    print()
    print(f"{'Terms':>6s} {'E-M Result':>20s} {'Error':>15s}")
    print("-" * 45)
    
    for num_terms in range(0, 11):
        em = euler_maclaurin_sum(f, 1, n, num_terms=num_terms)
        error = abs(direct - em)
        print(f"{num_terms:>6d} {em:>20.15f} {error:>15.2e}")


def plot_euler_maclaurin_visualizations():
    """
    Visualize the Euler-Maclaurin approximations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sum vs Integral visualization
    ax1 = axes[0, 0]
    n = 20
    x_discrete = np.arange(1, n + 1)
    y_discrete = x_discrete**2
    x_continuous = np.linspace(1, n, 300)
    y_continuous = x_continuous**2
    
    ax1.bar(x_discrete, y_discrete, width=0.8, alpha=0.6, label='Discrete sum', color='blue')
    ax1.plot(x_continuous, y_continuous, 'r-', linewidth=2, label='Integral')
    ax1.fill_between(x_continuous, 0, y_continuous, alpha=0.2, color='red')
    ax1.set_xlabel('k', fontsize=12)
    ax1.set_ylabel('k²', fontsize=12)
    ax1.set_title('Euler-Maclaurin: Sum vs Integral', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error vs number of terms
    ax2 = axes[0, 1]
    n = 100
    f = lambda x: x**2
    direct = direct_sum(f, 1, n)
    
    num_terms_range = range(0, 11)
    errors = []
    for nt in num_terms_range:
        em = euler_maclaurin_sum(f, 1, n, num_terms=nt)
        error = abs(direct - em)
        errors.append(error)
    
    ax2.semilogy(num_terms_range, errors, 'go-', markersize=8, linewidth=2)
    ax2.set_xlabel('Number of Bernoulli correction terms', fontsize=12)
    ax2.set_ylabel('Absolute error', fontsize=12)
    ax2.set_title('Convergence with Bernoulli Terms (Σk²)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Harmonic numbers approximation
    ax3 = axes[1, 0]
    n_vals = np.arange(10, 501, 10)
    direct_vals = []
    em_vals = []
    asymptotic_vals = []
    gamma = 0.5772156649015329
    
    for n in n_vals:
        f = lambda x: 1/x if x > 0 else 0
        direct_vals.append(direct_sum(f, 1, n))
        em_vals.append(euler_maclaurin_sum(f, 1, n, num_terms=5))
        asymptotic_vals.append(log(n) + gamma)
    
    ax3.plot(n_vals, direct_vals, 'b-', linewidth=2, label='Direct sum H_n')
    ax3.plot(n_vals, em_vals, 'r--', linewidth=2, label='Euler-Maclaurin')
    ax3.plot(n_vals, asymptotic_vals, 'g:', linewidth=2, label='ln(n) + γ')
    ax3.set_xlabel('n', fontsize=12)
    ax3.set_ylabel('H_n', fontsize=12)
    ax3.set_title('Harmonic Numbers: E-M Approximation', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Stirling's approximation
    ax4 = axes[1, 1]
    n_vals = np.arange(5, 101, 5)
    exact_vals = []
    em_vals = []
    stirling_vals = []
    
    for n in n_vals:
        exact_vals.append(sum(log(k) for k in range(1, n + 1)))
        em_vals.append(stirling_approximation_em(n, num_terms=5))
        stirling_vals.append(stirling_formula(n))
    
    ax4.plot(n_vals, exact_vals, 'b-', linewidth=2, label='Exact log(n!)')
    ax4.plot(n_vals, em_vals, 'r--', linewidth=2, label='Euler-Maclaurin')
    ax4.plot(n_vals, stirling_vals, 'g:', linewidth=2, label="Stirling's formula")
    ax4.set_xlabel('n', fontsize=12)
    ax4.set_ylabel('log(n!)', fontsize=12)
    ax4.set_title("Stirling's Approximation via E-M", fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('euler_maclaurin_formula.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'euler_maclaurin_formula.png'")
    plt.show()


def demonstrate_zeta_connection():
    """
    Show connection to Riemann zeta function.
    """
    print("\n7. Connection to Riemann Zeta Function:")
    print("-" * 40)
    print("For ζ(s) = Σ(k=1 to ∞) 1/k^s, we can use E-M to estimate the tail:")
    print()
    
    for s in [2, 3, 4]:
        # Sum first 100 terms directly
        n = 100
        f = lambda x: 1/x**s if x > 0 else 0
        partial_sum = direct_sum(f, 1, n)
        
        # Estimate tail using E-M from n+1 to large N
        tail_estimate = euler_maclaurin_sum(f, n + 1, 1000, num_terms=5) - partial_sum
        
        # Known values
        if s == 2:
            exact = pi**2 / 6
        elif s == 3:
            exact = 1.2020569  # Apéry's constant (approximate)
        elif s == 4:
            exact = pi**4 / 90
        
        zeta_estimate = partial_sum + tail_estimate
        error = abs(zeta_estimate - exact)
        
        print(f"ζ({s}) ≈ {zeta_estimate:.10f}  (exact: {exact:.10f}, error: {error:.2e})")


def main():
    """
    Main function to demonstrate Euler-Maclaurin formula.
    """
    demonstrate_euler_maclaurin()
    demonstrate_correction_terms()
    demonstrate_zeta_connection()
    
    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print("=" * 80)
    plot_euler_maclaurin_visualizations()
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("1. Euler-Maclaurin bridges discrete sums and continuous integrals")
    print("2. Bernoulli numbers provide exact correction terms")
    print("3. More Bernoulli terms → higher accuracy (up to a point)")
    print("4. Used to derive Stirling's approximation for factorials")
    print("5. Essential tool for asymptotic analysis in number theory")
    print("6. Shows deep connection between analysis and number theory via Bernoulli numbers")
    print("=" * 80)


if __name__ == "__main__":
    main()
