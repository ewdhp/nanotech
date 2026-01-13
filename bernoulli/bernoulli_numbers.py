#!/usr/bin/env python3
"""
Bernoulli Numbers and Their Relation to Pi

The Bernoulli numbers B_n are a sequence of rational numbers with deep connections
to number theory, analysis, and the value of π. They appear in:
- Taylor series expansions of trigonometric and hyperbolic functions
- Riemann zeta function values at even positive integers: ζ(2n) = (-1)^(n+1) * (2π)^(2n) * B_(2n) / (2(2n)!)
- Euler-Maclaurin formula
- Basel problem: ζ(2) = π²/6 = -B_2 * (2π)²/(2*2!)

The Bernoulli numbers are defined through the generating function:
    t/(e^t - 1) = Σ(n≥0) B_n * t^n / n!

Key properties:
- B_0 = 1
- B_1 = -1/2 (or +1/2 depending on convention)
- B_n = 0 for all odd n > 1
- B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, B_8 = -1/30, ...
"""

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from typing import List, Tuple
from math import factorial, pi


def bernoulli_number(n: int) -> Fraction:
    """
    Calculate the nth Bernoulli number using the recursive formula.
    
    Uses the formula:
    B_n = -1/(n+1) * Σ(k=0 to n-1) C(n+1,k) * B_k
    
    where C(n,k) is the binomial coefficient.
    
    Args:
        n: The index of the Bernoulli number (n >= 0)
    
    Returns:
        The nth Bernoulli number as a Fraction
    """
    if n == 0:
        return Fraction(1)
    if n == 1:
        return Fraction(-1, 2)  # Using B_1 = -1/2 convention
    if n > 1 and n % 2 == 1:
        return Fraction(0)  # All odd Bernoulli numbers (except B_1) are zero
    
    # Use dynamic programming to compute Bernoulli numbers
    B = [Fraction(0)] * (n + 1)
    B[0] = Fraction(1)
    
    for m in range(1, n + 1):
        sum_val = Fraction(0)
        for k in range(m):
            # Binomial coefficient C(m+1, k)
            binom = factorial(m + 1) // (factorial(k) * factorial(m + 1 - k))
            sum_val += binom * B[k]
        B[m] = -sum_val / (m + 1)
    
    return B[n]


def compute_bernoulli_numbers(max_n: int) -> List[Tuple[int, Fraction]]:
    """
    Compute Bernoulli numbers up to B_n.
    
    Args:
        max_n: Maximum index
    
    Returns:
        List of tuples (index, Bernoulli number)
    """
    results = []
    for n in range(max_n + 1):
        B_n = bernoulli_number(n)
        results.append((n, B_n))
    return results


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


def pi_from_zeta2() -> float:
    """
    Calculate π using the Basel problem: ζ(2) = π²/6.
    
    ζ(2) = 1 + 1/4 + 1/9 + 1/16 + ... = π²/6
    
    Using Bernoulli numbers: ζ(2) = -B_2 * (2π)² / (2 * 2!)
    Therefore: π² = -6 * B_2 * (2π)² / (2 * 2!) 
    Solving for π requires the actual series calculation.
    
    Returns:
        Approximate value of π from the series
    """
    # Calculate ζ(2) numerically
    zeta_2 = sum(1 / n**2 for n in range(1, 10000))
    pi_approx = np.sqrt(6 * zeta_2)
    return pi_approx


def demonstrate_zeta_relations():
    """
    Demonstrate the relationship between Bernoulli numbers and π
    through Riemann zeta function values.
    """
    print("=" * 80)
    print("BERNOULLI NUMBERS AND THEIR RELATION TO π")
    print("=" * 80)
    
    print("\n1. First few Bernoulli numbers:")
    print("-" * 40)
    bernoulli_list = compute_bernoulli_numbers(16)
    for n, B_n in bernoulli_list:
        if n <= 1 or n % 2 == 0:  # Only show B_0, B_1, and even indices
            print(f"B_{n:2d} = {B_n:>15s} = {float(B_n):>15.10f}")
    
    print("\n2. Riemann Zeta Function at Even Integers:")
    print("-" * 40)
    print("Formula: ζ(2n) = (-1)^(n+1) * (2π)^(2n) * B_(2n) / (2 * (2n)!)")
    print()
    
    for n in range(1, 7):
        zeta_val = zeta_even_from_bernoulli(n)
        # Direct computation for verification
        direct_sum = sum(1 / k**(2*n) for k in range(1, 10000))
        print(f"ζ({2*n:2d}) = {zeta_val:>15.10f}  (direct sum: {direct_sum:.10f})")
    
    print("\n3. Basel Problem - The Famous π²/6 Result:")
    print("-" * 40)
    print("ζ(2) = 1 + 1/4 + 1/9 + 1/16 + ... = π²/6")
    
    B_2 = float(bernoulli_number(2))
    print(f"\nB_2 = {B_2}")
    print(f"ζ(2) from Bernoulli = {zeta_even_from_bernoulli(1):.10f}")
    print(f"π²/6 = {pi**2 / 6:.10f}")
    print(f"Actual π = {pi:.10f}")
    
    # Calculate π from the series
    pi_from_series = pi_from_zeta2()
    print(f"π from ζ(2) series = {pi_from_series:.10f}")
    print(f"Error: {abs(pi - pi_from_series):.2e}")
    
    print("\n4. Expressing π² in terms of Bernoulli numbers:")
    print("-" * 40)
    print("From ζ(2) = (-1)^2 * (2π)² * B_2 / (2 * 2!)")
    print("We get: ζ(2) = 4π² * B_2 / 4 = π² * B_2")
    print(f"Since B_2 = 1/6, we have: ζ(2) = π²/6")
    print(f"Therefore: π² = 6 * ζ(2)")
    
    print("\n5. Other powers of π from Bernoulli numbers:")
    print("-" * 40)
    for n in range(2, 6):
        # ζ(2n) = (-1)^(n+1) * (2π)^(2n) * B_(2n) / (2 * (2n)!)
        # Rearranging: π^(2n) = (-1)^(n+1) * ζ(2n) * 2 * (2n)! / (2^(2n) * B_(2n))
        B_2n = float(bernoulli_number(2 * n))
        zeta_val = zeta_even_from_bernoulli(n)
        
        # Calculate π^(2n) from the formula
        sign = (-1) ** (n + 1)
        pi_power_calc = sign * zeta_val * 2 * factorial(2 * n) / ((2 ** (2 * n)) * B_2n)
        pi_power_actual = pi ** (2 * n)
        
        print(f"π^{2*n:2d} = {pi_power_actual:>15.2f}  (from Bernoulli: {pi_power_calc:>15.2f})")


def plot_bernoulli_convergence():
    """
    Visualize how partial sums of 1/n² converge to π²/6 using Bernoulli relation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Bernoulli numbers (absolute values, log scale)
    ax1 = axes[0, 0]
    indices = [i for i in range(0, 21) if i == 0 or i % 2 == 0]
    bernoulli_vals = [abs(float(bernoulli_number(i))) for i in indices]
    
    ax1.semilogy(indices, bernoulli_vals, 'bo-', markersize=8, linewidth=2)
    ax1.set_xlabel('n', fontsize=12)
    ax1.set_ylabel('|B_n|', fontsize=12)
    ax1.set_title('Bernoulli Numbers (absolute values, log scale)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Convergence of ζ(2) to π²/6
    ax2 = axes[0, 1]
    n_terms = np.arange(1, 1001)
    partial_sums = np.cumsum(1 / n_terms**2)
    target = pi**2 / 6
    
    ax2.plot(n_terms, partial_sums, 'b-', linewidth=2, label='Partial sum of 1/n²')
    ax2.axhline(y=target, color='r', linestyle='--', linewidth=2, label=f'π²/6 = {target:.6f}')
    ax2.set_xlabel('Number of terms', fontsize=12)
    ax2.set_ylabel('Partial sum', fontsize=12)
    ax2.set_title('Convergence of ζ(2) to π²/6 (Basel Problem)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1000)
    
    # Plot 3: Error in convergence (log scale)
    ax3 = axes[1, 0]
    errors = np.abs(partial_sums - target)
    ax3.loglog(n_terms, errors, 'g-', linewidth=2)
    ax3.set_xlabel('Number of terms', fontsize=12)
    ax3.set_ylabel('|Error|', fontsize=12)
    ax3.set_title('Error in approximating π²/6 (log-log scale)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: ζ(2n) values and their relationship with π
    ax4 = axes[1, 1]
    n_vals = np.arange(1, 11)
    zeta_vals = [zeta_even_from_bernoulli(n) for n in n_vals]
    
    ax4.semilogy(2 * n_vals, zeta_vals, 'mo-', markersize=8, linewidth=2, label='ζ(2n)')
    ax4.axhline(y=1, color='k', linestyle=':', alpha=0.5, label='y=1')
    ax4.set_xlabel('n (for ζ(n))', fontsize=12)
    ax4.set_ylabel('ζ(n)', fontsize=12)
    ax4.set_title('Riemann Zeta Function at Even Integers', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bernoulli_pi_relations.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'bernoulli_pi_relations.png'")
    plt.show()


def tangent_series_relation():
    """
    Demonstrate the relation between Bernoulli numbers and tan(x)/x series,
    which involves π.
    """
    print("\n6. Bernoulli Numbers in Tangent Series:")
    print("-" * 40)
    print("tan(x) = Σ(n≥1) [(-1)^(n-1) * 2^(2n) * (2^(2n) - 1) * B_(2n) * x^(2n-1)] / (2n)!")
    print("\nAt x = π/4, we have tan(π/4) = 1")
    
    # Calculate tan(π/4) using first few terms of Bernoulli series
    x = pi / 4
    terms = []
    for n in range(1, 8):
        B_2n = float(bernoulli_number(2 * n))
        sign = (-1) ** (n - 1)
        numerator = sign * (2 ** (2 * n)) * ((2 ** (2 * n)) - 1) * B_2n * (x ** (2 * n - 1))
        denominator = factorial(2 * n)
        term = numerator / denominator
        terms.append(term)
    
    approximation = sum(terms)
    print(f"\ntan(π/4) ≈ {approximation:.10f} (using {len(terms)} terms)")
    print(f"Actual value: {np.tan(x):.10f}")
    print(f"Error: {abs(np.tan(x) - approximation):.2e}")


def main():
    """
    Main function to demonstrate Bernoulli numbers and their relation to π.
    """
    demonstrate_zeta_relations()
    tangent_series_relation()
    
    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print("=" * 80)
    plot_bernoulli_convergence()
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("1. Bernoulli numbers provide exact formulas for ζ(2n) involving π")
    print("2. The Basel problem ζ(2) = π²/6 is a special case with B_2 = 1/6")
    print("3. Higher zeta values give π^4, π^6, π^8, etc., all in terms of Bernoulli numbers")
    print("4. Bernoulli numbers appear in series expansions of trigonometric functions")
    print("5. These connections reveal deep links between discrete (Bernoulli numbers)")
    print("   and continuous (π, transcendental functions) mathematics")
    print("=" * 80)


if __name__ == "__main__":
    main()
