"""
Symmetries in Maxwell's Equations

Maxwell's equations reveal deep symmetries:

1. **Duality between E and B**:
   In vacuum (ρ=0, J=0), Maxwell's equations are nearly symmetric under:
       E → c B,   B → -E/c
   (with appropriate sign conventions). This is called electromagnetic duality.

2. **Lorentz invariance** (spacetime symmetry):
   E and B mix together when you change reference frames. What looks like
   a pure electric field in one frame can have both E and B components in
   another moving frame. They form a single "electromagnetic field tensor" F^{μν}.

3. **Gauge symmetry**:
   The potentials (φ, A) from which E and B are derived have freedom:
       A → A + ∇χ,  φ → φ - ∂χ/∂t
   leaves E and B unchanged. This gauge freedom underlies quantum electrodynamics.

Here, we demonstrate *electromagnetic duality* in a simple scenario:
- Show a configuration with E field only (static charge).
- Show a configuration with B field only (steady current loop).
- Highlight the structural similarity in the equations.

We also illustrate how a moving charge creates both E and B fields (Lorentz mix).
"""

import numpy as np
import matplotlib.pyplot as plt


EPS0 = 8.854187817e-12
MU0 = 4e-7 * np.pi
C = 3e8


def electric_field_point_charge(x, y, Q, xq, yq):
    """E field from point charge."""
    dx = x - xq
    dy = y - yq
    r = np.sqrt(dx**2 + dy**2)
    r_safe = np.where(r < 1e-10, 1e-10, r)
    k = 1.0 / (4 * np.pi * EPS0)
    E_mag = k * Q / (r_safe**2)
    Ex = E_mag * (dx / r_safe)
    Ey = E_mag * (dy / r_safe)
    return Ex, Ey


def magnetic_field_wire(x, y, I, xw, yw):
    """
    B field from an infinite wire at (xw, yw) carrying current I along z.
    B_θ = (μ₀ I) / (2π r), direction: circular around wire.
    In 2D (x,y), Bz points out of or into page, but we represent the
    in-plane "circulation" as Bx, By components for visualization.

    Actually, for a wire along z, B is purely azimuthal:
        B_x = - (μ₀ I / 2π r²) * (y - yw)
        B_y =   (μ₀ I / 2π r²) * (x - xw)
    This is perpendicular to the radial direction in the x-y plane.
    """
    dx = x - xw
    dy = y - yw
    r = np.sqrt(dx**2 + dy**2)
    r_safe = np.where(r < 1e-10, 1e-10, r)
    B_mag = (MU0 * I) / (2 * np.pi * r_safe**2)
    # Azimuthal direction in x-y: (-dy, dx)
    Bx = B_mag * (-dy)
    By = B_mag * dx
    return Bx, By


def main():
    L = 2.0
    N = 80
    xs = np.linspace(-L, L, N)
    ys = np.linspace(-L, L, N)
    X, Y = np.meshgrid(xs, ys)

    Q = 1e-9  # 1 nC
    I = 1.0   # 1 A

    # (a) E field from charge at origin
    Ex_charge, Ey_charge = electric_field_point_charge(X, Y, Q, 0, 0)
    E_mag_charge = np.hypot(Ex_charge, Ey_charge)

    # (b) B field from wire at origin
    Bx_wire, By_wire = magnetic_field_wire(X, Y, I, 0, 0)
    B_mag_wire = np.hypot(Bx_wire, By_wire)

    # (c) Moving charge creates both E and B (simplified illustration)
    # A charge Q moving with velocity v creates:
    #   E ~ (usual Coulomb pattern)
    #   B ~ (v × E)/c² in the non-relativistic limit
    # For demo, show a charge at (0.5, 0) moving in +x direction at v ~ 0.1c
    v = 0.1 * C
    xq_moving = 0.5
    yq_moving = 0.0
    Ex_moving, Ey_moving = electric_field_point_charge(X, Y, Q, xq_moving, yq_moving)
    # Approximate B from moving charge (non-relativistic): B ≈ (v × E) / c²
    # For charge moving in +x, v = (v, 0, 0), E in x-y plane → B has z-component
    # We can represent the in-plane "circulation" effect qualitatively.
    # Simplification: Bx ~ v * Ey / c², By ~ -v * Ex / c²
    Bx_moving = (v / C**2) * Ey_moving
    By_moving = -(v / C**2) * Ex_moving
    B_mag_moving = np.hypot(Bx_moving, By_moving)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Row 1: E from static charge, B from steady current
    skip = 4

    # (0,0) E field
    axes[0, 0].quiver(
        X[::skip, ::skip], Y[::skip, ::skip],
        Ex_charge[::skip, ::skip], Ey_charge[::skip, ::skip],
        E_mag_charge[::skip, ::skip],
        cmap="plasma", pivot="mid", scale=1e10, minlength=0.1
    )
    axes[0, 0].plot(0, 0, 'ro', markersize=10)
    axes[0, 0].set_title("(a) E field from static charge Q")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_aspect("equal", "box")

    # (0,1) B field from wire
    axes[0, 1].quiver(
        X[::skip, ::skip], Y[::skip, ::skip],
        Bx_wire[::skip, ::skip], By_wire[::skip, ::skip],
        B_mag_wire[::skip, ::skip],
        cmap="viridis", pivot="mid", scale=1e-5, minlength=0.1
    )
    axes[0, 1].plot(0, 0, 'ko', markersize=10, label='Wire (I along z)')
    axes[0, 1].set_title("(b) B field from steady current I")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].set_aspect("equal", "box")
    axes[0, 1].legend()

    # (0,2) Duality note
    axes[0, 2].text(
        0.5, 0.5,
        "Electromagnetic Duality:\n\n"
        "In vacuum (ρ=0, J=0),\n"
        "Maxwell's equations are nearly\n"
        "symmetric under:\n\n"
        "   E → cB,  B → -E/c\n\n"
        "(with sign conventions).\n\n"
        "This reveals deep symmetry\n"
        "between electric and magnetic fields.",
        ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )
    axes[0, 2].axis("off")

    # Row 2: Moving charge (Lorentz mixing of E and B)
    # (1,0) E from moving charge
    axes[1, 0].quiver(
        X[::skip, ::skip], Y[::skip, ::skip],
        Ex_moving[::skip, ::skip], Ey_moving[::skip, ::skip],
        np.hypot(Ex_moving[::skip, ::skip], Ey_moving[::skip, ::skip]),
        cmap="plasma", pivot="mid", scale=1e10, minlength=0.1
    )
    axes[1, 0].plot(xq_moving, yq_moving, 'ro', markersize=10, label=f'Q moving at v={v/C:.2f}c')
    axes[1, 0].arrow(xq_moving, yq_moving, 0.3, 0, head_width=0.1, head_length=0.05, fc='red', ec='red')
    axes[1, 0].set_title("(c) E field from moving charge")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    axes[1, 0].set_aspect("equal", "box")
    axes[1, 0].legend()

    # (1,1) B field induced by moving charge
    axes[1, 1].quiver(
        X[::skip, ::skip], Y[::skip, ::skip],
        Bx_moving[::skip, ::skip], By_moving[::skip, ::skip],
        B_mag_moving[::skip, ::skip],
        cmap="viridis", pivot="mid", scale=1e-6, minlength=0.1
    )
    axes[1, 1].plot(xq_moving, yq_moving, 'ro', markersize=10)
    axes[1, 1].arrow(xq_moving, yq_moving, 0.3, 0, head_width=0.1, head_length=0.05, fc='red', ec='red')
    axes[1, 1].set_title("(d) B field induced by moving charge")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    axes[1, 1].set_aspect("equal", "box")

    # (1,2) Lorentz mixing note
    axes[1, 2].text(
        0.5, 0.5,
        "Lorentz Invariance:\n\n"
        "A moving charge creates both\n"
        "E and B fields.\n\n"
        "In special relativity, E and B\n"
        "are components of a single\n"
        "electromagnetic field tensor F^{μν}.\n\n"
        "Changing reference frames mixes\n"
        "E and B, revealing their unity.",
        ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5)
    )
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

    print("=== Symmetries in Maxwell's Equations ===")
    print("1. Electromagnetic duality: E and B are interchangeable in vacuum.")
    print("2. Lorentz invariance: E and B mix under frame transformations.")
    print("3. Gauge symmetry: potentials (φ, A) have freedom that leaves E, B unchanged.\n")
    print("These symmetries unify space, time, electricity, and magnetism.\n")


if __name__ == "__main__":
    main()
