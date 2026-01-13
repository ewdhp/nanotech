#!/usr/bin/env python3
"""
Riemann Zeta Function at Even Integers via Bernoulli Numbers
=============================================================

This script demonstrates the remarkable connection between Bernoulli numbers
and the Riemann zeta function at even positive integers.

The fundamental formula:
    ζ(2n) = (-1)^(n+1) * (2π)^(2n) * B_(2n) / (2 * (2n)!)

This provides exact closed-form expressions for ζ(2), ζ(4), ζ(6), etc.

Examples:
- ζ(2) = π²/6                    (Basel problem)
- ζ(4) = π⁴/90
- ζ(6) = π⁶/945
- ζ(8) = π⁸/9450
- ζ(10) = π¹⁰/93555

The Riemann zeta function is defined as:
    ζ(s) = Σ(n=1 to ∞) 1/n^s    for Re(s) > 1

At even integers, Bernoulli numbers provide exact rational multiples of
powers of π.
"""

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from math import factorial, pi
from typing import List, Tuple


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


def zeta_even_from_bernoulli(n: int) -> float:
    """
    Calculate ζ(2n) using Bernoulli numbers.
    
    Formula: ζ(2n) = (-1)^(n+1) * (2π)^(2n) * B_(2n) / (2 * (2n)!)
    
    Args:
        n: Positive integer (computes ζ(2n))
    
    Returns:
        The value of ζ(2n)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    B_2n = float(bernoulli_number(2 * n))
    sign = (-1) ** (n + 1)
    numerator = sign * (2 * pi) ** (2 * n) * B_2n
    denominator = 2 * factorial(2 * n)
    
    return numerator / denominator


def zeta_even_exact_formula(n: int) -> Tuple[Fraction, float]:
    """
    Calculate ζ(2n) as an exact rational multiple of π^(2n).
    
    Returns both the rational coefficient and the numerical value.
    
    Args:
        n: Positive integer (computes ζ(2n))
    
    Returns:
        Tuple of (coefficient as Fraction, numerical value)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    B_2n = bernoulli_number(2 * n)
    sign = (-1) ** (n + 1)
    
    # ζ(2n) = sign * (2^(2n)) * π^(2n) * B_(2n) / (2 * (2n)!)
    numerator = sign * (2 ** (2 * n)) * B_2n
    denominator = 2 * factorial(2 * n)
    
    coefficient = numerator / denominator
    numerical_value = float(coefficient) * (pi ** (2 * n))
    
    return coefficient, numerical_value


def zeta_direct_sum(s: int, max_terms: int = 100000) -> float:
    """
    Calculate ζ(s) by direct summation.
    
    ζ(s) = Σ(n=1 to ∞) 1/n^s
    
    Args:
        s: The argument of the zeta function
        max_terms: Maximum number of terms to sum
    
    Returns:
        Approximate value of ζ(s)
    """
    return sum(1 / n**s for n in range(1, max_terms + 1))


def demonstrate_zeta_bernoulli_relation():
    """
    Demonstrate the connection between Bernoulli numbers and ζ(2n).
    """
    print("=" * 80)
    print("RIEMANN ZETA FUNCTION AT EVEN INTEGERS VIA BERNOULLI NUMBERS")
    print("=" * 80)
    
    print("\nFormula: ζ(2n) = (-1)^(n+1) * (2π)^(2n) * B_(2n) / (2 * (2n)!)")
    
    print("\n1. First few even Bernoulli numbers:")
    print("-" * 40)
    for n in range(0, 11):
        if n % 2 == 0:
            B_n = bernoulli_number(n)
            print(f"B_{n:2d} = {str(B_n):>15s} = {float(B_n):>15.10f}")
    
    print("\n2. Zeta function values from Bernoulli numbers:")
    print("-" * 40)
    print(f"{'n':>4s} {'ζ(n)':>6s} {'Bernoulli':>18s} {'Direct Sum':>18s} {'Error':>12s}")
    print("-" * 40)
    
    for n in range(1, 13):
        zeta_bern = zeta_even_from_bernoulli(n)
        zeta_sum = zeta_direct_sum(2 * n, max_terms=100000)
        error = abs(zeta_bern - zeta_sum)
        print(f"{2*n:>4d} {'ζ('+str(2*n)+')':>6s} {zeta_bern:>18.12f} {zeta_sum:>18.12f} {error:>12.2e}")
    
    print("\n3. Exact formulas as rational multiples of π^(2n):")
    print("-" * 40)
    print("ζ(2n) = coefficient × π^(2n)")
    print()
    
    for n in range(1, 11):
        coeff, value = zeta_even_exact_formula(n)
        print(f"ζ({2*n:2d}) = {str(coeff):>20s} × π^{2*n:2d} = {value:>15.10f}")
    
    print("\n4. Famous special cases:")
    print("-" * 40)
    
    # ζ(2) = π²/6
    coeff_2, val_2 = zeta_even_exact_formula(1)
    print(f"ζ(2)  = {str(coeff_2):>15s} × π² = π²/6     = {val_2:.12f}")
    print(f"      = {pi**2 / 6:.12f}")
    
    # ζ(4) = π⁴/90
    coeff_4, val_4 = zeta_even_exact_formula(2)
    print(f"ζ(4)  = {str(coeff_4):>15s} × π⁴ = π⁴/90    = {val_4:.12f}")
    print(f"      = {pi**4 / 90:.12f}")
    
    # ζ(6) = π⁶/945
    coeff_6, val_6 = zeta_even_exact_formula(3)
    print(f"ζ(6)  = {str(coeff_6):>15s} × π⁶ = π⁶/945   = {val_6:.12f}")
    print(f"      = {pi**6 / 945:.12f}")
    
    # ζ(8) = π⁸/9450
    coeff_8, val_8 = zeta_even_exact_formula(4)
    print(f"ζ(8)  = {str(coeff_8):>15s} × π⁸ = π⁸/9450  = {val_8:.12f}")
    print(f"      = {pi**8 / 9450:.12f}")
    
    print("\n5. Extracting π from ζ(2n):")
    print("-" * 40)
    print("Given ζ(2n), we can solve for π:")
    print()
    
    for n in range(1, 6):
        zeta_val = zeta_even_from_bernoulli(n)
        coeff, _ = zeta_even_exact_formula(n)
        # π^(2n) = ζ(2n) / coeff
        pi_power = zeta_val / float(coeff)
        pi_extracted = pi_power ** (1 / (2 * n))
        error = abs(pi_extracted - pi)
        
        print(f"From ζ({2*n:2d}): π = {pi_extracted:.12f}  (error: {error:.2e})")


def analyze_convergence():
    """
    Analyze how quickly the direct sum converges to the Bernoulli formula.
    """
    print("\n6. Convergence analysis of direct summation:")
    print("-" * 40)
    print("Number of terms needed to match Bernoulli formula accuracy\n")
    
    for n in [1, 2, 3, 4, 5]:
        target = zeta_even_from_bernoulli(n)
        print(f"ζ({2*n}) = {target:.12f}")
        
        for num_terms in [100, 1000, 10000, 100000]:
            approx = zeta_direct_sum(2 * n, max_terms=num_terms)
            error = abs(approx - target)
            print(f"  {num_terms:6d} terms: {approx:.12f}  (error: {error:.2e})")
        print()


def plot_zeta_visualizations():
    """
    Create visualizations of the zeta function and its Bernoulli relation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Zeta values at even integers
    ax1 = axes[0, 0]
    n_vals = np.arange(1, 16)
    zeta_vals = [zeta_even_from_bernoulli(n) for n in n_vals]
    
    ax1.semilogy(2 * n_vals, zeta_vals, 'bo-', markersize=8, linewidth=2, label='ζ(2n)')
    ax1.axhline(y=1, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='y = 1')
    ax1.set_xlabel('n', fontsize=12)
    ax1.set_ylabel('ζ(n)', fontsize=12)
    ax1.set_title('Riemann Zeta Function at Even Integers', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Convergence of partial sums for different ζ(2n)
    ax2 = axes[0, 1]
    max_n = 100
    n_terms = np.arange(1, max_n + 1)
    
    for s in [2, 4, 6, 8]:
        partial_sums = np.cumsum(1 / n_terms**s)
        target = zeta_even_from_bernoulli(s // 2)
        ax2.plot(n_terms, partial_sums, linewidth=2, label=f'ζ({s}) → {target:.6f}')
    
    ax2.set_xlabel('Number of terms', fontsize=12)
    ax2.set_ylabel('Partial sum', fontsize=12)
    ax2.set_title('Convergence of Direct Summation', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error in direct summation (log-log scale)
    ax3 = axes[1, 0]
    n_terms_log = np.logspace(1, 4, 50, dtype=int)
    
    for s in [2, 4, 6, 8]:
        errors = []
        target = zeta_even_from_bernoulli(s // 2)
        for n in n_terms_log:
            approx = zeta_direct_sum(s, max_terms=n)
            error = abs(approx - target)
            errors.append(error)
        ax3.loglog(n_terms_log, errors, 'o-', linewidth=2, markersize=4, label=f'ζ({s})')
    
    ax3.set_xlabel('Number of terms', fontsize=12)
    ax3.set_ylabel('Absolute error', fontsize=12)
    ax3.set_title('Convergence Error (log-log scale)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Coefficients (as fractions of 1/π^(2n))
    ax4 = axes[1, 1]
    n_vals = np.arange(1, 13)
    coefficients = []
    
    for n in n_vals:
        coeff, _ = zeta_even_exact_formula(n)
        # We want to show the denominator of ζ(2n)/π^(2n)
        coefficients.append(float(coeff))
    
    ax4.semilogy(2 * n_vals, coefficients, 'mo-', markersize=8, linewidth=2)
    ax4.set_xlabel('n (for ζ(n))', fontsize=12)
    ax4.set_ylabel('Coefficient (ζ(n)/π^n)', fontsize=12)
    ax4.set_title('Rational Coefficients in ζ(2n) = coeff × π^(2n)', 
                  fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('riemann_zeta_bernoulli.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'riemann_zeta_bernoulli.png'")
    plt.show()


def demonstrate_functional_equation():
    """
    Show the connection between ζ(s) and ζ(1-s) using Bernoulli numbers.
    """
    print("\n7. Functional equation of the Riemann zeta function:")
    print("-" * 40)
    print("ζ(s) = 2^s * π^(s-1) * sin(πs/2) * Γ(1-s) * ζ(1-s)")
    print("\nFor even positive integers, this connects to Bernoulli numbers.")
    print("At negative integers: ζ(-n) = -B_(n+1)/(n+1)")
    print()
    
    for n in range(0, 6):
        B_n_plus_1 = bernoulli_number(n + 1)
        zeta_negative = -B_n_plus_1 / (n + 1)
        print(f"ζ(-{n}) = -B_{n+1}/{n+1} = {str(zeta_negative):>15s} = {float(zeta_negative):>12.8f}")


def main():
    """
    Main function to demonstrate Riemann zeta at even integers.
    """
    demonstrate_zeta_bernoulli_relation()
    analyze_convergence()
    demonstrate_functional_equation()
    
    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print("=" * 80)
    plot_zeta_visualizations()
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("1. Bernoulli numbers give EXACT values for ζ(2n) in closed form")
    print("2. All ζ(2n) are rational multiples of π^(2n)")
    print("3. The direct sum Σ 1/n^s converges slowly, Bernoulli formula is exact")
    print("4. This connects discrete number theory (Bernoulli) to continuous analysis (π)")
    print("5. The formula fails at odd integers - ζ(3), ζ(5), etc. remain mysterious!")
    print("6. Euler discovered this remarkable connection in the 18th century")
    print("=" * 80)


if __name__ == "__main__":
    main()
