Riemann Zeta Function — Theory & Demo
====================================

Files:
- `riemann_zeta_theory.py` — Interactive script demonstrating the zeta function.
- `requirements.txt` — Python packages recommended for full functionality.

What the script shows:
- Dirichlet series definition (converges for Re(s)>1).
- Dirichlet eta function and the relation ζ(s) = η(s)/(1-2^{1-s}) used for analytic continuation.
- Euler product representation (approximate, uses primes).
- Numerical verification of ζ(2) = π²/6 (Basel problem).
- Visualizations: convergence plots and (if `mpmath` is installed) complex-plane modulus plots and attempts to refine some nontrivial zeros.

Run instructions:
1. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r nanotech/riemann/requirements.txt
```

3. Run the demo:

```bash
python nanotech/riemann/riemann_zeta_theory.py
```

Notes:
- `mpmath` is optional; without it the script still runs real-valued demonstrations (Basel series, Euler product, etc.), but complex analytic continuation and root-finding features are disabled.
- Some visualizations (complex-plane contour) can be slow for high resolution; adjust grid density inside the script if needed.

If you'd like, I can:
- Add a small unit-test harness to verify a few special values.
- Add interactive widgets for exploring the ζ(s) surface.
- Increase precision and reliability of zero-finding and save results to a CSV.
