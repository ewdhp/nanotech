#!/usr/bin/env python3
"""
Bernoulli Numbers in Taylor Series Expansions
==============================================

This script demonstrates how Bernoulli numbers appear in the Taylor series
expansions of trigonometric and hyperbolic functions.

Key formulas:
1. tan(x) = Σ(n≥1) [(-1)^(n-1) * 2^(2n) * (2^(2n) - 1) * B_(2n) * x^(2n-1)] / (2n)!
2. cot(x) = 1/x - Σ(n≥1) [2^(2n) * B_(2n) * x^(2n-1)] / (2n)!
3. tanh(x) = Σ(n≥1) [2^(2n) * (2^(2n) - 1) * B_(2n) * x^(2n-1)] / (2n)!
4. x/sinh(x) = Σ(n≥0) [2 * (1 - 2^(2n)) * B_(2n) * x^(2n)] / (2n)!
5. x*coth(x) = Σ(n≥0) [2^(2n) * B_(2n) * x^(2n)] / (2n)!

The generating function for Bernoulli numbers is:
    t/(e^t - 1) = Σ(n≥0) B_n * t^n / n!
"""

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from math import factorial, pi


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


def tan_series(x: float, terms: int = 10) -> float:
    """
    Calculate tan(x) using Bernoulli numbers.
    
    tan(x) = Σ(n≥1) [(-1)^(n-1) * 2^(2n) * (2^(2n) - 1) * B_(2n) * x^(2n-1)] / (2n)!
    
    Args:
        x: Input value in radians
        terms: Number of terms to use in series
    
    Returns:
        Approximation of tan(x)
    """
    result = 0.0
    for n in range(1, terms + 1):
        B_2n = float(bernoulli_number(2 * n))
        sign = (-1) ** (n - 1)
        numerator = sign * (2 ** (2 * n)) * ((2 ** (2 * n)) - 1) * B_2n * (x ** (2 * n - 1))
        denominator = factorial(2 * n)
        result += numerator / denominator
    return result


def cot_series(x: float, terms: int = 10) -> float:
    """
    Calculate cot(x) using Bernoulli numbers.
    
    cot(x) = 1/x - Σ(n≥1) [2^(2n) * B_(2n) * x^(2n-1)] / (2n)!
    
    Args:
        x: Input value in radians (must be non-zero)
        terms: Number of terms to use in series
    
    Returns:
        Approximation of cot(x)
    """
    result = 1.0 / x
    for n in range(1, terms + 1):
        B_2n = float(bernoulli_number(2 * n))
        numerator = (2 ** (2 * n)) * B_2n * (x ** (2 * n - 1))
        denominator = factorial(2 * n)
        result -= numerator / denominator
    return result


def tanh_series(x: float, terms: int = 10) -> float:
    """
    Calculate tanh(x) using Bernoulli numbers.
    
    tanh(x) = Σ(n≥1) [2^(2n) * (2^(2n) - 1) * B_(2n) * x^(2n-1)] / (2n)!
    
    Args:
        x: Input value
        terms: Number of terms to use in series
    
    Returns:
        Approximation of tanh(x)
    """
    result = 0.0
    for n in range(1, terms + 1):
        B_2n = float(bernoulli_number(2 * n))
        numerator = (2 ** (2 * n)) * ((2 ** (2 * n)) - 1) * B_2n * (x ** (2 * n - 1))
        denominator = factorial(2 * n)
        result += numerator / denominator
    return result


def x_over_sinh_series(x: float, terms: int = 10) -> float:
    """
    Calculate x/sinh(x) using Bernoulli numbers.
    
    x/sinh(x) = Σ(n≥0) [2 * (1 - 2^(2n)) * B_(2n) * x^(2n)] / (2n)!
    
    Args:
        x: Input value
        terms: Number of terms to use in series
    
    Returns:
        Approximation of x/sinh(x)
    """
    result = 0.0
    for n in range(terms + 1):
        B_2n = float(bernoulli_number(2 * n))
        numerator = 2 * (1 - (2 ** (2 * n))) * B_2n * (x ** (2 * n))
        denominator = factorial(2 * n)
        result += numerator / denominator
    return result


def x_coth_series(x: float, terms: int = 10) -> float:
    """
    Calculate x*coth(x) using Bernoulli numbers.
    
    x*coth(x) = Σ(n≥0) [2^(2n) * B_(2n) * x^(2n)] / (2n)!
    
    Args:
        x: Input value
        terms: Number of terms to use in series
    
    Returns:
        Approximation of x*coth(x)
    """
    result = 0.0
    for n in range(terms + 1):
        B_2n = float(bernoulli_number(2 * n))
        numerator = (2 ** (2 * n)) * B_2n * (x ** (2 * n))
        denominator = factorial(2 * n)
        result += numerator / denominator
    return result


def demonstrate_taylor_series():
    """
    Demonstrate Taylor series expansions with Bernoulli numbers.
    """
    print("=" * 80)
    print("BERNOULLI NUMBERS IN TAYLOR SERIES EXPANSIONS")
    print("=" * 80)
    
    print("\n1. First few Bernoulli numbers:")
    print("-" * 40)
    for n in range(0, 11):
        if n <= 1 or n % 2 == 0:
            B_n = bernoulli_number(n)
            print(f"B_{n:2d} = {str(B_n):>12s} = {float(B_n):>12.8f}")
    
    # Test values
    x_vals = [pi/6, pi/4, pi/3, 0.5, 1.0]
    
    print("\n2. Tangent Function - tan(x):")
    print("-" * 40)
    print(f"{'x':>10s} {'Bernoulli':>15s} {'NumPy':>15s} {'Error':>15s}")
    for x in x_vals:
        if abs(x) < 1.5:  # tan series converges for |x| < π/2
            approx = tan_series(x, terms=12)
            actual = np.tan(x)
            error = abs(approx - actual)
            print(f"{x:>10.6f} {approx:>15.10f} {actual:>15.10f} {error:>15.2e}")
    
    print("\n3. Cotangent Function - cot(x):")
    print("-" * 40)
    print(f"{'x':>10s} {'Bernoulli':>15s} {'NumPy':>15s} {'Error':>15s}")
    for x in x_vals:
        if x > 0.1:  # Avoid division by very small numbers
            approx = cot_series(x, terms=12)
            actual = 1.0 / np.tan(x)
            error = abs(approx - actual)
            print(f"{x:>10.6f} {approx:>15.10f} {actual:>15.10f} {error:>15.2e}")
    
    print("\n4. Hyperbolic Tangent - tanh(x):")
    print("-" * 40)
    print(f"{'x':>10s} {'Bernoulli':>15s} {'NumPy':>15s} {'Error':>15s}")
    for x in [0.5, 1.0, 1.5, 2.0]:
        approx = tanh_series(x, terms=12)
        actual = np.tanh(x)
        error = abs(approx - actual)
        print(f"{x:>10.6f} {approx:>15.10f} {actual:>15.10f} {error:>15.2e}")
    
    print("\n5. x/sinh(x) Function:")
    print("-" * 40)
    print(f"{'x':>10s} {'Bernoulli':>15s} {'Direct':>15s} {'Error':>15s}")
    for x in [0.5, 1.0, 1.5, 2.0]:
        approx = x_over_sinh_series(x, terms=12)
        actual = x / np.sinh(x)
        error = abs(approx - actual)
        print(f"{x:>10.6f} {approx:>15.10f} {actual:>15.10f} {error:>15.2e}")
    
    print("\n6. x*coth(x) Function:")
    print("-" * 40)
    print(f"{'x':>10s} {'Bernoulli':>15s} {'Direct':>15s} {'Error':>15s}")
    for x in [0.5, 1.0, 1.5, 2.0]:
        approx = x_coth_series(x, terms=12)
        actual = x / np.tanh(x)
        error = abs(approx - actual)
        print(f"{x:>10.6f} {approx:>15.10f} {actual:>15.10f} {error:>15.2e}")


def plot_taylor_series_comparisons():
    """
    Visualize the Taylor series approximations vs actual functions.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: tan(x)
    ax1 = axes[0, 0]
    x_range = np.linspace(-1.4, 1.4, 200)
    y_actual = np.tan(x_range)
    y_approx = [tan_series(x, terms=10) for x in x_range]
    
    ax1.plot(x_range, y_actual, 'b-', linewidth=2, label='tan(x) - actual')
    ax1.plot(x_range, y_approx, 'r--', linewidth=2, label='Bernoulli series')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('tan(x) via Bernoulli Numbers', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 5)
    
    # Plot 2: cot(x)
    ax2 = axes[0, 1]
    x_range = np.linspace(0.2, 3.0, 200)
    y_actual = 1.0 / np.tan(x_range)
    y_approx = [cot_series(x, terms=10) for x in x_range]
    
    ax2.plot(x_range, y_actual, 'b-', linewidth=2, label='cot(x) - actual')
    ax2.plot(x_range, y_approx, 'r--', linewidth=2, label='Bernoulli series')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title('cot(x) via Bernoulli Numbers', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-2, 5)
    
    # Plot 3: tanh(x)
    ax3 = axes[0, 2]
    x_range = np.linspace(-2, 2, 200)
    y_actual = np.tanh(x_range)
    y_approx = [tanh_series(x, terms=10) for x in x_range]
    
    ax3.plot(x_range, y_actual, 'b-', linewidth=2, label='tanh(x) - actual')
    ax3.plot(x_range, y_approx, 'r--', linewidth=2, label='Bernoulli series')
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('y', fontsize=11)
    ax3.set_title('tanh(x) via Bernoulli Numbers', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: x/sinh(x)
    ax4 = axes[1, 0]
    x_range = np.linspace(0.1, 3, 200)
    y_actual = x_range / np.sinh(x_range)
    y_approx = [x_over_sinh_series(x, terms=10) for x in x_range]
    
    ax4.plot(x_range, y_actual, 'b-', linewidth=2, label='x/sinh(x) - actual')
    ax4.plot(x_range, y_approx, 'r--', linewidth=2, label='Bernoulli series')
    ax4.set_xlabel('x', fontsize=11)
    ax4.set_ylabel('y', fontsize=11)
    ax4.set_title('x/sinh(x) via Bernoulli Numbers', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: x*coth(x)
    ax5 = axes[1, 1]
    x_range = np.linspace(0.1, 3, 200)
    y_actual = x_range / np.tanh(x_range)
    y_approx = [x_coth_series(x, terms=10) for x in x_range]
    
    ax5.plot(x_range, y_actual, 'b-', linewidth=2, label='x*coth(x) - actual')
    ax5.plot(x_range, y_approx, 'r--', linewidth=2, label='Bernoulli series')
    ax5.set_xlabel('x', fontsize=11)
    ax5.set_ylabel('y', fontsize=11)
    ax5.set_title('x*coth(x) via Bernoulli Numbers', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Convergence analysis for tan(x)
    ax6 = axes[1, 2]
    x_test = pi / 4
    num_terms = range(1, 16)
    errors = []
    for n in num_terms:
        approx = tan_series(x_test, terms=n)
        actual = np.tan(x_test)
        errors.append(abs(approx - actual))
    
    ax6.semilogy(num_terms, errors, 'go-', linewidth=2, markersize=8)
    ax6.set_xlabel('Number of terms', fontsize=11)
    ax6.set_ylabel('Absolute error', fontsize=11)
    ax6.set_title(f'Convergence of tan(π/4) Bernoulli Series', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('taylor_series_bernoulli.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'taylor_series_bernoulli.png'")
    plt.show()


def main():
    """
    Main function to demonstrate Bernoulli numbers in Taylor series.
    """
    demonstrate_taylor_series()
    
    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print("=" * 80)
    plot_taylor_series_comparisons()
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("1. Bernoulli numbers provide exact coefficients for Taylor series")
    print("2. Both trigonometric (tan, cot) and hyperbolic (tanh, coth) functions")
    print("   have Bernoulli number representations")
    print("3. The series converge rapidly within their radius of convergence")
    print("4. These expansions connect discrete number theory to continuous analysis")
    print("5. The generating function t/(e^t - 1) unifies all these representations")
    print("=" * 80)


if __name__ == "__main__":
    main()
