"""
Riemann Zeta Function — Theory & Demonstrations (Python)

This script demonstrates key aspects of the Riemann zeta function ζ(s):
- Dirichlet series definition for Re(s) > 1
- Alternating Dirichlet eta function and analytic continuation relation
- Euler product for Re(s) > 1 (via primes)
- Special values (ζ(2) = π²/6 and trivial zeros at negative even integers)
- Numerical verification & convergence plots
- (Optional, if `mpmath` installed) analytic continuation, complex evaluation,
  and locating the first few nontrivial zeros on the critical line

Usage:
    python riemann_zeta_theory.py

Notes:
- If `mpmath` is not installed, advanced complex-plane features are disabled,
  but basic real-valued demonstrations still work.

Dependencies (recommended):
  - numpy
  - matplotlib
  - seaborn
  - mpmath (optional, strongly recommended for complex analysis)

"""

from __future__ import annotations
import math
import cmath
import sys
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import mpmath for analytic continuation and root-finding
try:
    import mpmath as mp
    MP_AVAILABLE = True
except Exception:
    mp = None  # type: ignore
    MP_AVAILABLE = False

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


# ------------------------- Elementary definitions -------------------------

def zeta_dirichlet(s: complex, n_terms: int = 100000) -> complex:
    """
    Dirichlet series definition of ζ(s) = ∑_{n=1}^∞ n^{-s}.
    Use only when Re(s) > 1 (converges slowly as Re(s) → 1+).
    n_terms controls the truncation.
    """
    total = 0 + 0j
    for n in range(1, n_terms + 1):
        total += n ** (-s)
    return total


def zeta_eta_via_dirichlet_eta(s: complex, n_terms: int = 200000) -> complex:
    """
    Compute Dirichlet eta function η(s) = ∑_{n=1}^∞ (-1)^{n-1} n^{-s}
    and use the relation ζ(s) = η(s) / (1 - 2^{1-s}) to extend to Re(s)>0,
    except where denominator vanishes.
    This is a classic way Euler used to analytically continue ζ.
    """
    eta = 0 + 0j
    for n in range(1, n_terms + 1):
        eta += ((-1) ** (n - 1)) * (n ** (-s))
    denom = 1 - 2 ** (1 - s)
    if abs(denom) < 1e-16:
        return complex('nan')
    return eta / denom


# ------------------------ Euler product (approx) --------------------------

def primes_sieve(limit: int) -> List[int]:
    """Simple Sieve of Eratosthenes to generate primes up to `limit`."""
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start: limit + 1: step] = b"\x00" * (((limit - start) // step) + 1)
    return [i for i, is_prime in enumerate(sieve) if is_prime]


def euler_product_zeta(s: complex, primes: List[int]) -> complex:
    """
    Approximate Euler product: ζ(s) ≈ ∏_{p prime} (1 - p^{-s})^{-1}
    Valid for Re(s) > 1; using a finite list of primes yields an approximation.
    """
    prod = 1 + 0j
    for p in primes:
        prod *= 1.0 / (1 - p ** (-s))
    return prod


# ----------------------- Special value: ζ(2) example ---------------------

def basel_partial_sum(n_terms: int) -> float:
    """Compute partial sum ∑_{n=1}^{N} 1/n² to illustrate convergence to π²/6."""
    return sum(1.0 / (n * n) for n in range(1, n_terms + 1))


# --------------------------- Visualization helpers -----------------------

def plot_basel_convergence(n_max: int = 2000) -> None:
    n_vals = np.arange(1, n_max + 1)
    partials = np.cumsum(1.0 / (n_vals ** 2))
    exact = math.pi ** 2 / 6

    plt.figure()
    plt.plot(n_vals, partials, label="Partial sums ∑1/n²")
    plt.axhline(y=exact, color='r', linestyle='--', label=f'π²/6 = {exact:.6f}')
    plt.xlabel('Number of terms (N)')
    plt.ylabel('Sum')
    plt.title('Convergence of the Basel Series: ∑ 1/n² → π²/6')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_dirichlet_vs_euler(s_real: float = 1.5, n_series: int = 10000, prime_limit: int = 100000) -> None:
    """
    Compare truncated Dirichlet series and Euler product for a real s>1.
    """
    s = s_real
    series_vals = [sum(1.0 / (n ** s) for n in range(1, N + 1)) for N in [10, 100, 1000, n_series]]

    # sample primes (small limit)
    primes = primes_sieve(5000)  # enough primes for rough product
    product_vals = [euler_product_zeta(s, primes[:k]) for k in [10, 50, 200, len(primes)]]

    plt.figure()
    plt.plot([10, 100, 1000, n_series], series_vals, 'o-', label='Dirichlet partial sums')
    plt.plot([10, 50, 200, len(primes)], product_vals, 's--', label='Euler product partials')
    plt.xlabel('Number of terms / primes used')
    plt.ylabel(f'Approx ζ({s})')
    plt.title(f'Comparing Dirichlet series and Euler product (s={s})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


# ----------------- Complex-plane demonstrations (mpmath) -----------------

def mpmath_demo() -> None:
    """
    If `mpmath` is available, do analytic continuation demos and locate first
    few nontrivial zeros (approximations) using findroot.
    """
    if not MP_AVAILABLE:
        print("mpmath not available — skipping complex-plane demos.\nTo enable, install 'mpmath' and rerun.")
        return

    mp.mp.dps = 50  # high precision

    print("\nUsing mpmath for analytic continuation and root-finding:\n")

    # Evaluate some known special values
    print(f"ζ(2) (mpmath): {mp.zeta(2)} vs π²/6 = {mp.pi()**2/6}")
    print(f"ζ(0) = {mp.zeta(0)}  (expect -1/2)")
    print(f"ζ(-2) = {mp.zeta(-2)}  (trivial zero)\n")

    # Evaluate on critical line and show modulus plot over a region
    re_range = np.linspace(0.0, 1.0, 201)
    im_range = np.linspace(0, 40, 401)
    Re, Im = np.meshgrid(re_range, im_range)
    Z = np.zeros_like(Re, dtype=float)

    print('Computing |ζ(s)| on a grid (this may take a few seconds)...')
    for i in range(Re.shape[0]):
        for j in range(Re.shape[1]):
            s = Re[i, j] + 1j * Im[i, j]
            Z[i, j] = abs(mp.zeta(s))

    plt.figure(figsize=(10, 6))
    plt.contourf(Re, Im, np.log(Z + 1e-30), levels=60, cmap='viridis')
    plt.colorbar(label='log |ζ(s)|')
    plt.xlabel('Re(s)')
    plt.ylabel('Im(s)')
    plt.title('Log-modulus of ζ(s) in the critical strip 0 ≤ Re(s) ≤ 1, 0 ≤ Im(s) ≤ 40')
    plt.tight_layout()

    # Locate first few nontrivial zeros using known starting heuristics
    # Known approximate imaginary parts of first zeros: 14.134725141, 21.022039639, 25.010857580
    initial_imag = [14.134725141, 21.022039639, 25.010857580]
    zeros = []
    print('\nAttempting to refine a few nontrivial zeros near the critical line s=1/2+it')
    for t0 in initial_imag:
        try:
            root = mp.findroot(lambda z: mp.zeta(z), mp.mpc(0.5, t0))
            zeros.append(root)
            print(f'  Found zero: {root}')
        except Exception as e:
            print(f'  findroot failed near t={t0}: {e}')

    # Mark zeros on the contour plot (if any found)
    if zeros:
        zs = np.array([[z.real, z.imag] for z in zeros], dtype=float)
        plt.scatter(zs[:, 0], zs[:, 1], color='red', s=50, marker='x', label='approx zeros')
        plt.legend()

    plt.show()


# ------------------------------- Main -----------------------------------

def main() -> None:
    print('\nRiemann Zeta Function — Theory Demonstration')
    print('--------------------------------------------\n')

    # 1) Basel problem demonstration
    print('1) Basel problem: ζ(2) = ∑ 1/n² = π²/6')
    for N in [10, 50, 100, 1000, 10000]:
        approx = basel_partial_sum(N)
        print(f'  Partial sum N={N:6d} → {approx:.12f}  (error {abs(approx - math.pi**2/6):.2e})')
    plot_basel_convergence(2000)

    # 2) Dirichlet series vs Euler product for a real s>1
    print('\n2) Compare Dirichlet partial sums and Euler product for s=1.5')
    plot_dirichlet_vs_euler(s_real=1.5, n_series=5000)

    # 3) Show that Dirichlet eta relation can be used for analytic continuation
    print('\n3) Dirichlet eta relation: ζ(s) = η(s) / (1 - 2^{1-s})')
    s_test = 0.5
    approx_eta = zeta_eta_via_dirichlet_eta(s_test, n_terms=200000)
    print(f'  Using alternating series at s={s_test}: ζ(s) ≈ {approx_eta}')

    # 4) Euler product approximation
    print('\n4) Euler product (approximation, Re(s)>1)')
    primes = primes_sieve(5000)
    euler_approx = euler_product_zeta(1.5, primes)
    print(f'  Euler-product approximation ζ(1.5) with {len(primes)} primes ≈ {euler_approx}')

    # 5) mpmath analytic continuation & zeros (if available)
    if MP_AVAILABLE:
        mpmath_demo()
    else:
        print('\nmpmath not installed — skip complex-plane demos. To enable advanced demos, install mpmath:')
        print('    pip install mpmath')

    plt.show()


if __name__ == '__main__':
    main()
