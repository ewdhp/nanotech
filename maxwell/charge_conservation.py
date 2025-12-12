"""
Charge Conservation – Emerging from Maxwell's Equations

Taking the divergence of Ampère-Maxwell law:
    ∇×B = μ₀ J + μ₀ ε₀ ∂E/∂t

and using ∇·(∇×B) = 0 (vector identity), plus Gauss's law ∇·E = ρ/ε₀, yields:

    ∂ρ/∂t + ∇·J = 0

This is the *continuity equation* for charge: charge cannot be created or
destroyed, only moved around. It emerges naturally from Maxwell's equations.

We demonstrate this with a simple 2D simulation:
- Start with a localized charge distribution ρ(x,y,t).
- Allow a current density J(x,y,t) to flow (e.g., diffusion-like flow).
- Numerically verify that ∂ρ/∂t + ∇·J ≈ 0.
- Show that total charge ∫ ρ dA remains constant.

For simplicity, we use:
    ∂ρ/∂t = -∇·J
with J = -D ∇ρ (a diffusive current), which automatically satisfies continuity.
"""

import numpy as np
import matplotlib.pyplot as plt


def divergence_2d(Jx, Jy, dx, dy):
    """Compute ∇·J."""
    divJ = np.zeros_like(Jx)
    divJ[1:-1, 1:-1] = (
        (Jx[2:, 1:-1] - Jx[:-2, 1:-1]) / (2*dx)
        + (Jy[1:-1, 2:] - Jy[1:-1, :-2]) / (2*dy)
    )
    return divJ


def gradient_2d(f, dx, dy):
    """Compute ∇f = (∂f/∂x, ∂f/∂y)."""
    dfdx = np.zeros_like(f)
    dfdy = np.zeros_like(f)
    dfdx[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2*dx)
    dfdy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2*dy)
    return dfdx, dfdy


def main():
    # Grid
    L = 2.0
    N = 128
    xs = np.linspace(-L, L, N)
    ys = np.linspace(-L, L, N)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys)

    # Initial charge distribution (Gaussian)
    sigma = 0.4
    rho = np.exp(-(X**2 + Y**2) / (2*sigma**2))

    # Diffusion coefficient for current J = -D ∇ρ
    D = 0.05
    dt = 0.001
    n_steps = 500

    total_charge = []
    times = []

    # Time evolution
    for step in range(n_steps):
        # Current: J = -D ∇ρ
        drhodx, drhody = gradient_2d(rho, dx, dy)
        Jx = -D * drhodx
        Jy = -D * drhody

        # ∇·J
        divJ = divergence_2d(Jx, Jy, dx, dy)

        # Update ρ: ∂ρ/∂t = -∇·J
        rho -= dt * divJ

        # Track total charge (should be conserved)
        Q_total = np.sum(rho) * dx * dy
        total_charge.append(Q_total)
        times.append(step * dt)

    # Plot final state and charge conservation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Final charge distribution
    im = axes[0].pcolormesh(X, Y, rho, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=axes[0], label=r"$\rho(x,y,t)$")
    axes[0].set_title(f"Charge density after diffusion\nt = {n_steps*dt:.3f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal", "box")

    # (b) Total charge vs time
    axes[1].plot(times, total_charge, 'b-', linewidth=2)
    axes[1].set_title("Total charge conservation: ∫ ρ dA vs. time")
    axes[1].set_xlabel("Time (arbitrary units)")
    axes[1].set_ylabel("Total charge Q")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    print("=== Charge Conservation ===")
    print("Continuity equation: ∂ρ/∂t + ∇·J = 0")
    print("This emerges from Maxwell's equations and ensures charge is conserved.")
    print(f"Initial total charge: {total_charge[0]:.6f}")
    print(f"Final total charge:   {total_charge[-1]:.6f}")
    print(f"Relative change:      {abs(total_charge[-1] - total_charge[0]) / total_charge[0] * 100:.3e} %\n")


if __name__ == "__main__":
    main()
