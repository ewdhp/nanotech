"""
Tensor fields in nanotechnology – simple stress tensor example.

We model a spherical nanoparticle (radius R) embedded in a matrix.
The matrix is subjected to uniaxial tension σ0 along z.
Near the nanoparticle, the stress field is perturbed.

We define a *tensor field* σ(x): ℝ³ → ℝ^{3×3}, i.e. at each point x we have
a 3×3 stress tensor. For simplicity, we approximate the perturbation using
a radially decaying amplification factor around the particle.

This is *not* a full elasticity solution – it's a pedagogical example to:

- Show what a tensor field is (here, a rank-2 stress tensor).
- Compute scalar invariants (trace, von Mises) from the tensor field.
- Visualize a 2D slice of the field, which is common in nano-mechanics
  (e.g., stress around nanoinclusions, nanovoids, etc.).
"""

import numpy as np
import matplotlib.pyplot as plt


def stress_tensor_field(x, y, z, R=10e-9, sigma0=1.0):
    """
    Return 3x3 Cauchy stress tensor σ at point (x,y,z).

    Base state: uniaxial tension along z:
        σ_base = diag(0, 0, sigma0)

    Nanoparticle at origin with radius R.
    We amplify stress near the particle using a simple factor:

        f(r) = 1 + A * exp( -(r-R)^2 / (2 * w^2) )

    where:
        r = sqrt(x^2 + y^2 + z^2)
        A > 0 is the max amplification;
        w controls how fast the perturbation decays away from the surface.
    """
    r = np.sqrt(x**2 + y**2 + z**2)

    A = 3.0       # max amplification
    w = 0.5 * R   # shell width

    # Handle r as array or scalar; r==0 is fine (inside particle, masked later)
    f = 1.0 + A * np.exp(-((r - R) ** 2) / (2.0 * w**2))

    sigma = np.zeros((3, 3), dtype=float)
    sigma[2, 2] = sigma0 * f
    return sigma


def von_mises_from_sigma(sigma):
    """
    Compute von Mises equivalent stress from a 3x3 symmetric stress tensor.

    σ_vm = sqrt(3/2 * s_ij s_ij)
    with deviatoric stress s_ij = σ_ij - 1/3 * tr(σ) δ_ij.
    """
    tr = np.trace(sigma)
    identity = np.eye(3)
    s = sigma - (tr / 3.0) * identity
    s2 = np.tensordot(s, s, axes=2)
    return np.sqrt(1.5 * s2)


def demo_point_calculations():
    """
    Print example tensor components and invariants at selected points.
    """
    R = 10e-9
    sigma0 = 1.0

    points = {
        "far_field": (0.0, 0.0, 5.0 * R),
        "near_axis_surface": (0.0, 0.0, 1.2 * R),
        "equator_surface": (1.0 * R, 0.0, 0.0),
    }

    print("=== Sample stress tensor field evaluations ===")
    print(f"Nanoparticle radius R = {R:.3e} m, base σ0 = {sigma0:.2f} (arb. units)\n")

    for name, (x, y, z) in points.items():
        sigma = stress_tensor_field(x, y, z, R=R, sigma0=sigma0)
        vm = von_mises_from_sigma(sigma)
        tr = np.trace(sigma)
        print(f"Point '{name}': (x,y,z) = ({x:.3e}, {y:.3e}, {z:.3e})")
        print("  σ (stress tensor) =")
        with np.printoptions(precision=3, suppress=True):
            print(" ", sigma)
        print(f"  trace(σ)        = {tr:.3f}")
        print(f"  von Mises σ_vm  = {vm:.3f}\n")


def plot_subplots():
    """
    Create a single window with multiple meaningful plots:

    (a) 2D x–z slice of von Mises stress (y = 0).
    (b) 2D x–z slice of σ_zz (normal stress along loading direction).
    (c) 1D radial profile of σ_zz along the loading axis (x=0, z>0).

    This mimics common nano-mechanics post-processing:
    - Comparing scalar invariant vs. a single tensor component.
    - Looking at radial stress concentration around a nanoparticle.
    """
    R = 10e-9
    sigma0 = 1.0

    n = 200
    xs = np.linspace(-4 * R, 4 * R, n)
    zs = np.linspace(-4 * R, 4 * R, n)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros_like(X)

    vm_field = np.zeros_like(X)
    szz_field = np.zeros_like(X)

    for i in range(n):
        for j in range(n):
            sigma = stress_tensor_field(X[i, j], Y[i, j], Z[i, j], R=R, sigma0=sigma0)
            vm_field[i, j] = von_mises_from_sigma(sigma)
            szz_field[i, j] = sigma[2, 2]  # σ_zz component

    R_grid = np.sqrt(X**2 + Y**2 + Z**2)
    vm_field = np.ma.masked_where(R_grid < R, vm_field)
    szz_field = np.ma.masked_where(R_grid < R, szz_field)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) von Mises field
    im0 = axes[0].pcolormesh(X / R, Z / R, vm_field, shading="auto", cmap="viridis")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("Von Mises stress\n(x–z slice, y=0)")
    axes[0].set_xlabel("x / R")
    axes[0].set_ylabel("z / R")
    axes[0].set_aspect("equal", "box")
    circle0 = plt.Circle((0, 0), 1.0, color="white", fill=False, linewidth=1.0)
    axes[0].add_patch(circle0)

    # (b) σ_zz field
    im1 = axes[1].pcolormesh(X / R, Z / R, szz_field, shading="auto", cmap="plasma")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("σ_zz (loading direction)\n(x–z slice, y=0)")
    axes[1].set_xlabel("x / R")
    axes[1].set_ylabel("z / R")
    axes[1].set_aspect("equal", "box")
    circle1 = plt.Circle((0, 0), 1.0, color="white", fill=False, linewidth=1.0)
    axes[1].add_patch(circle1)

    # (c) radial profile of σ_zz along loading axis (x=0, z≥0)
    z_line = np.linspace(0.5 * R, 4.0 * R, 200)  # start slightly outside center
    szz_line = []
    for z in z_line:
        sigma = stress_tensor_field(0.0, 0.0, z, R=R, sigma0=sigma0)
        szz_line.append(sigma[2, 2])
    szz_line = np.array(szz_line)

    axes[2].plot(z_line / R, szz_line, label=r"$\sigma_{zz}$ (x=0, y=0)")
    axes[2].axvline(1.0, color="k", linestyle="--", linewidth=1.0, label="particle surface (r=R)")
    axes[2].set_xlabel("z / R (along loading axis)")
    axes[2].set_ylabel(r"$\sigma_{zz}$ (arb. units)")
    axes[2].set_title(r"Radial profile of $\sigma_{zz}$")
    axes[2].legend(loc="best")

    plt.tight_layout()
    plt.show()


def main():
    demo_point_calculations()
    print("Generating tensor-field subplots (von Mises, σ_zz, radial profile)...")
    plot_subplots()


if __name__ == "__main__":
    main()
