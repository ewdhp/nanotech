"""
Fields as Fundamental Entities

Classical action-at-a-distance view:
- Two charges Q1 and Q2 separated by r exert forces on each other instantly:
      F = k * Q1 * Q2 / r²
  No mention of what "carries" this force through empty space.

Field-mediated view (Maxwell):
- Charge Q1 creates an electric field E(x) everywhere in space.
- Charge Q2 at position x feels a force F = Q2 * E(x).
- The field E is a real physical entity that stores energy, momentum, and
  can propagate as waves even after sources are removed.

This script visualizes:
1. The electric field E(x,y) from a point charge (Coulomb field).
2. Energy density u = ε₀/2 * E² stored in the field.
3. A simple "field propagation" idea: if we suddenly move the charge,
   the field disturbance propagates outward at finite speed (not instant).

We use a 2D slice (x,y) for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt


EPS0 = 8.854187817e-12  # vacuum permittivity (F/m)
K = 1.0 / (4.0 * np.pi * EPS0)  # Coulomb constant


def electric_field_point_charge(x, y, Q, xq, yq):
    """
    Electric field E = (Ex, Ey) from a point charge Q at (xq, yq).

    E(r) = k * Q / r² * r̂
    where r̂ is the unit vector from (xq,yq) to (x,y).
    """
    dx = x - xq
    dy = y - yq
    r = np.sqrt(dx**2 + dy**2)
    r_safe = np.where(r == 0, 1e-10, r)  # avoid singularity

    E_mag = K * Q / (r_safe**2)
    Ex = E_mag * (dx / r_safe)
    Ey = E_mag * (dy / r_safe)
    return Ex, Ey


def energy_density(Ex, Ey):
    """
    Energy density in the electric field:
        u = ε₀/2 * (Ex² + Ey²)
    """
    return 0.5 * EPS0 * (Ex**2 + Ey**2)


def main():
    # Grid
    L = 2.0
    N = 100
    xs = np.linspace(-L, L, N)
    ys = np.linspace(-L, L, N)
    X, Y = np.meshgrid(xs, ys)

    # Point charge at origin
    Q = 1e-9  # 1 nC
    xq, yq = 0.0, 0.0

    Ex, Ey = electric_field_point_charge(X, Y, Q, xq, yq)
    E_mag = np.hypot(Ex, Ey)
    u = energy_density(Ex, Ey)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Electric field vectors
    skip = 5
    axes[0].quiver(
        X[::skip, ::skip], Y[::skip, ::skip],
        Ex[::skip, ::skip], Ey[::skip, ::skip],
        E_mag[::skip, ::skip],
        cmap="plasma", pivot="mid", scale=1e10, minlength=0.1
    )
    axes[0].plot(xq, yq, 'ro', markersize=10, label=f'Charge Q = {Q*1e9:.1f} nC')
    axes[0].set_title("Electric field E(x,y) from point charge\n(field as fundamental entity)")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal", "box")
    axes[0].legend()

    # (b) Energy density
    im = axes[1].pcolormesh(X, Y, u, shading="auto", cmap="inferno", vmax=np.percentile(u, 99))
    fig.colorbar(im, ax=axes[1], label=r"Energy density $u = \frac{\varepsilon_0}{2} E^2$ (J/m³)")
    axes[1].plot(xq, yq, 'wo', markersize=8)
    axes[1].set_title("Energy stored in the field\n(fields carry energy, not just forces)")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_aspect("equal", "box")

    plt.tight_layout()
    plt.show()

    print("=== Fields as Fundamental Entities ===")
    print("The electric field E(x) is not just a mathematical convenience.")
    print("It is a physical entity that:")
    print("  - exists throughout space,")
    print("  - stores energy and momentum,")
    print("  - mediates forces between charges.")
    print("\nKey insight: removing the source charge doesn't instantly eliminate")
    print("the field everywhere — field disturbances propagate at finite speed c.\n")


if __name__ == "__main__":
    main()
