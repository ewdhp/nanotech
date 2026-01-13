#!/usr/bin/env python3
"""
Babbage Difference Engine Simulation
=====================================

This script simulates Charles Babbage's Difference Engine, demonstrating
the method of finite differences for computing polynomial functions.

The Difference Engine uses only addition (no multiplication or division)
to compute successive values of polynomial functions through cascading
differences.

Features:
- Full Difference Engine simulation
- Polynomial evaluation using method of differences
- Step-by-step visualization of the algorithm
- Support for arbitrary degree polynomials
- Historical table generation (logarithms, squares, etc.)
- Interactive demonstrations

Author: Simulation of Babbage's 1822-1871 designs
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from fractions import Fraction
import pandas as pd


class DifferenceEngine:
    """
    Simulation of Babbage's Difference Engine.
    
    The engine computes polynomial values using the method of finite
    differences, requiring only addition operations.
    """
    
    def __init__(self, degree: int, initial_values: List[float] = None):
        """
        Initialize the Difference Engine.
        
        Args:
            degree: Degree of the polynomial
            initial_values: Initial values [P(x0), Δ¹P(x0), Δ²P(x0), ..., ΔⁿP(x0)]
                           If None, will be set later
        """
        self.degree = degree
        self.columns = degree + 1  # Column 0 is P(x), columns 1..n are differences
        
        if initial_values is not None:
            if len(initial_values) != self.columns:
                raise ValueError(f"Need {self.columns} initial values for degree {degree}")
            self.values = list(initial_values)
        else:
            self.values = [0.0] * self.columns
        
        self.history = [self.values.copy()]
        self.step_count = 0
    
    def set_from_polynomial(self, coefficients: List[float], x0: float = 0):
        """
        Initialize the engine from polynomial coefficients.
        
        Args:
            coefficients: [a0, a1, a2, ...] for P(x) = a0 + a1*x + a2*x² + ...
            x0: Starting value of x
        """
        # Compute initial values P(x0), Δ¹P(x0), Δ²P(x0), ...
        poly = lambda x: sum(c * x**i for i, c in enumerate(coefficients))
        
        self.values[0] = poly(x0)
        
        # Compute differences
        for k in range(1, self.columns):
            # Δᵏ = Δᵏ⁻¹(x0+1) - Δᵏ⁻¹(x0)
            diff_values = []
            for x in range(int(x0), int(x0) + k + 2):
                val = poly(x)
                diff_values.append(val)
            
            # Compute k-th difference
            current = diff_values
            for _ in range(k):
                current = [current[i+1] - current[i] for i in range(len(current)-1)]
            
            self.values[k] = current[0]
        
        self.history = [self.values.copy()]
        self.step_count = 0
    
    def step(self) -> float:
        """
        Execute one step of the Difference Engine (one turn of the crank).
        
        This computes the next value of the polynomial by cascading additions
        from the highest difference down to P(x).
        
        Returns:
            The new value of P(x)
        """
        # Cascade additions from right to left (high order to low order)
        for i in range(self.columns - 1, 0, -1):
            self.values[i-1] += self.values[i]
        
        self.step_count += 1
        self.history.append(self.values.copy())
        
        return self.values[0]
    
    def compute_n_steps(self, n: int) -> List[float]:
        """
        Compute n steps and return all P(x) values.
        
        Args:
            n: Number of steps to compute
        
        Returns:
            List of P(x) values
        """
        results = [self.values[0]]
        for _ in range(n):
            results.append(self.step())
        return results
    
    def get_current_state(self) -> dict:
        """
        Get current state of all columns.
        
        Returns:
            Dictionary with column labels and values
        """
        state = {'P(x)': self.values[0]}
        for i in range(1, self.columns):
            state[f'Δ^{i}P'] = self.values[i]
        state['Step'] = self.step_count
        return state
    
    def print_state(self):
        """Print current state of the engine in tabular format."""
        print(f"\nStep {self.step_count}:")
        print("-" * 60)
        for i in range(self.columns):
            if i == 0:
                label = "P(x)"
            else:
                label = f"Δ^{i}P"
            print(f"  Column {i} ({label:>6s}): {self.values[i]:15.6f}")
    
    def get_difference_table(self, n_steps: int = 10) -> pd.DataFrame:
        """
        Generate a difference table showing the computation.
        
        Args:
            n_steps: Number of steps to compute
        
        Returns:
            DataFrame with difference table
        """
        # Store original state
        original_values = self.values.copy()
        original_step = self.step_count
        
        # Compute steps
        table_data = []
        table_data.append([self.step_count] + self.values.copy())
        
        for _ in range(n_steps):
            self.step()
            table_data.append([self.step_count] + self.values.copy())
        
        # Restore original state
        self.values = original_values
        self.step_count = original_step
        
        # Create DataFrame
        columns = ['Step', 'P(x)'] + [f'Δ^{i}P' for i in range(1, self.columns)]
        df = pd.DataFrame(table_data, columns=columns)
        
        return df


def compute_finite_differences(values: List[float]) -> List[List[float]]:
    """
    Compute finite difference table for a sequence of values.
    
    Args:
        values: Sequence of function values
    
    Returns:
        List of difference levels (each level is a list)
    """
    differences = [values]
    current = values
    
    while len(current) > 1:
        next_diff = [current[i+1] - current[i] for i in range(len(current)-1)]
        differences.append(next_diff)
        current = next_diff
        
        # Stop if all differences are effectively zero
        if all(abs(d) < 1e-10 for d in next_diff):
            break
    
    return differences


def demonstrate_simple_polynomial():
    """
    Demonstrate the engine computing a simple polynomial: P(x) = x²
    """
    print("=" * 80)
    print("DEMONSTRATION 1: Computing P(x) = x²")
    print("=" * 80)
    
    print("\nPolynomial: P(x) = x²")
    print("Starting at x = 0")
    print("\nInitializing Difference Engine...")
    
    # For P(x) = x², coefficients are [0, 0, 1]
    engine = DifferenceEngine(degree=2)
    engine.set_from_polynomial([0, 0, 1], x0=0)
    
    print("\nInitial state (x = 0):")
    engine.print_state()
    
    print("\nComputing next 10 values by turning the crank...")
    print("\n" + "-" * 80)
    print(f"{'Step':>6s} {'x':>6s} {'P(x)=x²':>12s} {'Δ¹P':>12s} {'Δ²P':>12s}")
    print("-" * 80)
    
    print(f"{0:>6d} {0:>6d} {0:>12.1f} {1:>12.1f} {2:>12.1f}")
    
    for i in range(1, 11):
        result = engine.step()
        state = engine.get_current_state()
        print(f"{i:>6d} {i:>6d} {state['P(x)']:>12.1f} {state['Δ^1P']:>12.1f} {state['Δ^2P']:>12.1f}")
    
    print("\nObservations:")
    print("1. Δ²P remains constant at 2 (second difference of x² is always 2)")
    print("2. Δ¹P increases by 2 each step (1, 3, 5, 7, 9, ...)")
    print("3. P(x) gives perfect squares (0, 1, 4, 9, 16, 25, ...)")
    print("4. Only addition was used—no multiplication!")


def demonstrate_cubic_polynomial():
    """
    Demonstrate computing a cubic polynomial: P(x) = x³ - 2x² + x
    """
    print("\n" + "=" * 80)
    print("DEMONSTRATION 2: Computing P(x) = x³ - 2x² + x")
    print("=" * 80)
    
    print("\nPolynomial: P(x) = x³ - 2x² + x")
    print("Coefficients: [0, 1, -2, 1]")
    print("Starting at x = 0")
    
    # Coefficients: [a0, a1, a2, a3] for P(x) = a0 + a1*x + a2*x² + a3*x³
    engine = DifferenceEngine(degree=3)
    engine.set_from_polynomial([0, 1, -2, 1], x0=0)
    
    print("\nGenerating difference table...")
    df = engine.get_difference_table(n_steps=10)
    print("\n" + df.to_string(index=False))
    
    print("\nObservations:")
    print("1. Δ³P remains constant at 6 (third difference of cubic is constant)")
    print("2. Each level of differences is computed by simple addition")
    print("3. The pattern cascades from right to left in each step")


def demonstrate_squares_table():
    """
    Generate a table of squares like Babbage intended.
    """
    print("\n" + "=" * 80)
    print("DEMONSTRATION 3: Table of Squares (Historical Application)")
    print("=" * 80)
    
    print("\nGenerating a table of squares from 1 to 50")
    print("This was a typical use case in Babbage's era")
    
    engine = DifferenceEngine(degree=2)
    engine.set_from_polynomial([0, 0, 1], x0=1)
    
    print("\n" + "-" * 50)
    print(f"{'n':>10s} {'n²':>15s} {'n² (check)':>15s}")
    print("-" * 50)
    
    # Print initial value
    print(f"{1:>10d} {int(engine.values[0]):>15d} {1**2:>15d}")
    
    # Compute next values
    for n in range(2, 51):
        result = engine.step()
        actual = n**2
        print(f"{n:>10d} {int(result):>15d} {actual:>15d}")
    
    print("\nResult: Perfect squares computed without any multiplication!")


def demonstrate_method_of_differences():
    """
    Explain and visualize the method of finite differences.
    """
    print("\n" + "=" * 80)
    print("THEORY: The Method of Finite Differences")
    print("=" * 80)
    
    print("\nFor any polynomial P(x) of degree n:")
    print("  Δ⁰P(x) = P(x)")
    print("  Δ¹P(x) = P(x+1) - P(x)")
    print("  Δ²P(x) = Δ¹P(x+1) - Δ¹P(x)")
    print("  ...")
    print("  ΔⁿP(x) = constant")
    print("  Δⁿ⁺¹P(x) = 0")
    
    print("\nExample with P(x) = x² + 3x + 2:")
    
    # Generate values
    x_vals = list(range(0, 8))
    p_vals = [x**2 + 3*x + 2 for x in x_vals]
    
    # Compute differences
    diffs = compute_finite_differences(p_vals)
    
    print("\n" + "-" * 70)
    print(f"{'x':>6s} {'P(x)':>10s} {'Δ¹P':>10s} {'Δ²P':>10s} {'Δ³P':>10s}")
    print("-" * 70)
    
    for i in range(len(x_vals)):
        row = f"{x_vals[i]:>6d} {diffs[0][i]:>10.0f}"
        if i < len(diffs[1]):
            row += f" {diffs[1][i]:>10.0f}"
        if i < len(diffs[2]):
            row += f" {diffs[2][i]:>10.0f}"
        if i < len(diffs[3]) if len(diffs) > 3 else False:
            row += f" {diffs[3][i]:>10.0f}"
        print(row)
    
    print("\nNotice:")
    print("- Δ²P is constant (= 2), confirming this is a degree 2 polynomial")
    print("- We can compute the next P(x) by adding back the differences")
    print("- This is exactly how the Difference Engine works!")


def demonstrate_logarithm_approximation():
    """
    Demonstrate computing logarithms using polynomial approximation.
    """
    print("\n" + "=" * 80)
    print("DEMONSTRATION 4: Computing Logarithms (Historical Application)")
    print("=" * 80)
    
    print("\nLogarithms were crucial for navigation and astronomy in the 1800s.")
    print("The Difference Engine could compute them using polynomial approximation.")
    
    print("\nFor x near 1, we can approximate:")
    print("  ln(1+x) ≈ x - x²/2 + x³/3 - x⁴/4 + ...")
    print("\nTruncating to a polynomial: ln(1+x) ≈ x - x²/2 + x³/3")
    
    # Coefficients for ln(1+x) ≈ x - x²/2 + x³/3
    # P(x) = 0 + 1*x - 0.5*x² + 0.333*x³
    engine = DifferenceEngine(degree=3)
    engine.set_from_polynomial([0, 1, -0.5, 1/3], x0=0)
    
    print("\nComputing values:")
    print("\n" + "-" * 70)
    print(f"{'x':>10s} {'ln(1+x) approx':>20s} {'ln(1+x) actual':>20s} {'Error':>15s}")
    print("-" * 70)
    
    x = 0
    for i in range(11):
        if i > 0:
            approx = engine.step()
        else:
            approx = engine.values[0]
        
        x_val = x + i * 0.1
        actual = np.log(1 + x_val) if x_val > -1 else np.nan
        error = abs(approx - actual) if not np.isnan(actual) else np.nan
        
        print(f"{x_val:>10.2f} {approx:>20.10f} {actual:>20.10f} {error:>15.2e}")


def visualize_difference_engine():
    """
    Create visualizations of the Difference Engine operation.
    """
    print("\n" + "=" * 80)
    print("VISUALIZATION: Difference Engine Operation")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Values of P(x) = x² over steps
    ax1 = axes[0, 0]
    engine = DifferenceEngine(degree=2)
    engine.set_from_polynomial([0, 0, 1], x0=0)
    
    steps = list(range(21))
    values = engine.compute_n_steps(20)
    
    ax1.plot(steps, values, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Step (x)', fontsize=12)
    ax1.set_ylabel('P(x) = x²', fontsize=12)
    ax1.set_title('Computing Squares with Difference Engine', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference columns evolution
    ax2 = axes[0, 1]
    engine2 = DifferenceEngine(degree=2)
    engine2.set_from_polynomial([0, 0, 1], x0=0)
    
    p_vals = [engine2.values[0]]
    d1_vals = [engine2.values[1]]
    d2_vals = [engine2.values[2]]
    
    for _ in range(20):
        engine2.step()
        p_vals.append(engine2.values[0])
        d1_vals.append(engine2.values[1])
        d2_vals.append(engine2.values[2])
    
    ax2.plot(steps, p_vals, 'b-', linewidth=2, label='P(x)')
    ax2.plot(steps, d1_vals, 'r--', linewidth=2, label='Δ¹P')
    ax2.plot(steps, d2_vals, 'g:', linewidth=3, label='Δ²P (constant)')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Column Value', fontsize=12)
    ax2.set_title('All Columns Evolution (P(x) = x²)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cubic polynomial
    ax3 = axes[1, 0]
    engine3 = DifferenceEngine(degree=3)
    engine3.set_from_polynomial([0, 1, -2, 1], x0=0)
    
    cubic_vals = engine3.compute_n_steps(20)
    
    ax3.plot(steps, cubic_vals, 'mo-', linewidth=2, markersize=6)
    ax3.set_xlabel('Step (x)', fontsize=12)
    ax3.set_ylabel('P(x) = x³ - 2x² + x', fontsize=12)
    ax3.set_title('Computing Cubic Polynomial', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot 4: Difference table heatmap
    ax4 = axes[1, 1]
    engine4 = DifferenceEngine(degree=3)
    engine4.set_from_polynomial([0, 0, 0, 1], x0=0)
    
    # Build difference table
    table = []
    for _ in range(15):
        table.append(engine4.values.copy())
        engine4.step()
    
    table = np.array(table).T
    im = ax4.imshow(table, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax4.set_xlabel('Step', fontsize=12)
    ax4.set_ylabel('Column (Difference Level)', fontsize=12)
    ax4.set_title('Difference Table Heatmap (P(x) = x³)', fontsize=13, fontweight='bold')
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(['P(x)', 'Δ¹P', 'Δ²P', 'Δ³P'])
    plt.colorbar(im, ax=ax4, label='Value')
    
    plt.tight_layout()
    plt.savefig('difference_engine_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'difference_engine_visualization.png'")
    plt.show()


def simulate_analytical_engine_concept():
    """
    Conceptual simulation of the Analytical Engine's capabilities.
    """
    print("\n" + "=" * 80)
    print("ANALYTICAL ENGINE CONCEPT: General Computation")
    print("=" * 80)
    
    print("\nThe Analytical Engine went beyond polynomial computation.")
    print("It could perform arbitrary sequences of operations.")
    
    print("\nExample: Computing Fibonacci numbers")
    print("Algorithm:")
    print("  1. Store: a = 0, b = 1")
    print("  2. Loop:")
    print("     - temp = a + b  (Mill performs addition)")
    print("     - a = b         (Store updates)")
    print("     - b = temp      (Store updates)")
    print("     - Output b")
    
    print("\n" + "-" * 40)
    print(f"{'n':>5s} {'Fibonacci(n)':>15s}")
    print("-" * 40)
    
    a, b = 0, 1
    print(f"{0:>5d} {a:>15d}")
    print(f"{1:>5d} {b:>15d}")
    
    for n in range(2, 21):
        a, b = b, a + b
        print(f"{n:>5d} {b:>15d}")
    
    print("\nThis simple example shows the power of programmability:")
    print("- Variables stored in 'Store'")
    print("- Operations performed by 'Mill'")
    print("- Loop control by operation cards")
    print("- General algorithm, not limited to polynomials!")


def ada_lovelace_algorithm():
    """
    Demonstrate Ada Lovelace's algorithm for Bernoulli numbers.
    """
    print("\n" + "=" * 80)
    print("ADA LOVELACE'S ALGORITHM: Computing Bernoulli Numbers")
    print("=" * 80)
    
    print("\nAda Lovelace's 1843 'Note G' described an algorithm to compute")
    print("Bernoulli numbers on the Analytical Engine—the first published")
    print("computer algorithm in history!")
    
    print("\nBernoulli numbers appear in many mathematical formulas.")
    print("They can be computed recursively:")
    
    def bernoulli(n):
        """Compute nth Bernoulli number."""
        if n == 0:
            return Fraction(1)
        if n == 1:
            return Fraction(-1, 2)
        if n > 1 and n % 2 == 1:
            return Fraction(0)
        
        from math import factorial
        B = [Fraction(0)] * (n + 1)
        B[0] = Fraction(1)
        
        for m in range(1, n + 1):
            sum_val = Fraction(0)
            for k in range(m):
                binom = factorial(m + 1) // (factorial(k) * factorial(m + 1 - k))
                sum_val += binom * B[k]
            B[m] = -sum_val / (m + 1)
        
        return B[n]
    
    print("\n" + "-" * 50)
    print(f"{'n':>5s} {'B_n (exact)':>25s} {'B_n (decimal)':>20s}")
    print("-" * 50)
    
    for n in [0, 1, 2, 4, 6, 8, 10]:
        B_n = bernoulli(n)
        print(f"{n:>5d} {str(B_n):>25s} {float(B_n):>20.10f}")
    
    print("\nLovelace's insight: The Analytical Engine could manipulate")
    print("symbols (not just numbers), making it a general-purpose computer!")


def main():
    """
    Main demonstration of the Babbage machine simulations.
    """
    print("\n" + "=" * 80)
    print("BABBAGE'S DIFFERENCE ENGINE AND ANALYTICAL ENGINE")
    print("Simulation and Demonstration")
    print("=" * 80)
    
    print("\nThis program simulates Charles Babbage's revolutionary computing")
    print("machines from the 1800s, demonstrating the method of finite")
    print("differences and early concepts of programmable computation.")
    
    # Run demonstrations
    demonstrate_simple_polynomial()
    demonstrate_cubic_polynomial()
    demonstrate_method_of_differences()
    demonstrate_squares_table()
    demonstrate_logarithm_approximation()
    simulate_analytical_engine_concept()
    ada_lovelace_algorithm()
    
    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print("=" * 80)
    visualize_difference_engine()
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
1. The Difference Engine computed polynomials using ONLY addition
2. Method of differences: nth difference of degree-n polynomial is constant
3. Cascading additions from high-order to low-order differences
4. Historical applications: mathematical tables, navigation, astronomy
5. Analytical Engine: first general-purpose programmable computer design
6. Ada Lovelace: first computer programmer (Bernoulli algorithm, 1843)
7. Modern vindication: Difference Engine No. 2 built successfully in 1991
8. Legacy: Concepts foundational to all modern computing

"The Analytical Engine weaves algebraical patterns just as the
 Jacquard loom weaves flowers and leaves." — Ada Lovelace
    """)
    print("=" * 80)
    
    print("\n✓ Simulation complete! See README.md for detailed theory.")


if __name__ == "__main__":
    main()
