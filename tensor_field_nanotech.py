"""
Tensor, vector and scalar fields in nanotechnology – nanoparticle example.

What is a *field*?
- A field assigns a quantity to every point in space (and sometimes time).
  In this script we consider fields in 3D space, evaluated on a 2D slice.

We illustrate three kinds of fields that routinely appear in nano‑mechanics
and nano‑materials modeling:

1. Scalar field  φ(x)      : ℝ³ → ℝ
   - One number per point.
   - Examples in nanotech: temperature field in a device, potential energy
     landscape for an atom or molecule, concentration of a species.
   - Here: φ(r) is an artificial "interaction potential" that peaks near the
     nanoparticle surface and decays away. It only stores *magnitude*.

2. Vector field  u(x)      : ℝ³ → ℝ³
   - A 3‑component quantity (direction + magnitude) per point.
   - Examples in nanotech: displacement field in a deformed nanowire,
     electric field around a charged nanoparticle, heat flux.
   - Here: u(x) points radially away from the nanoparticle center, with
     magnitude modulated by φ(r). It encodes both *where* and *how strongly*
     the "displacement" acts.

3. Tensor field  σ(x)      : ℝ³ → ℝ^{3×3}
   - A 3×3 matrix per point (rank‑2 tensor). In continuum mechanics, σ is
     the Cauchy stress tensor, describing how internal forces act on all
     possible surface orientations at that point.
   - Examples in nanotech: stress distribution in a nanocomposite,
     anisotropic elastic response of 2D materials, strain in quantum dots.
   - Here: σ(x) is a simplified uniaxial stress tensor amplified near the
     nanoparticle. We visualize it via the scalar *von Mises* stress, a
     common invariant used to summarize the intensity of a stress tensor.

The goal of this script is not to be a full mechanical model, but to give a
computationally simple, visual example of:
- a scalar field,
- a vector field, and
- a tensor field
defined on the same physical configuration: a spherical nanoparticle
embedded in a matrix under uniaxial loading.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------- tensor field: stress tensor σ(x) ----------

def stress_tensor_field(x, y, z, R=10e-9, sigma0=1.0):
    """
    Return 3x3 Cauchy stress tensor σ at point (x,y,z).

    Base state: uniaxial tension along z:
        σ_base = diag(0, 0, sigma0)

    Nanoparticle at origin with radius R.
    We amplify stress near the particle using a simple Gaussian shell factor:

        f(r) = 1 + A * exp( -(r-R)^2 / (2 * w^2) )

    where:
        r = sqrt(x^2 + y^2 + z^2)
        A > 0 is the max amplification;
        w controls how fast the perturbation decays away from the surface.
    """
    r = np.sqrt(x**2 + y**2 + z**2)

    A = 3.0       # max amplification
    w = 0.5 * R   # shell width

    f = 1.0 + A * np.exp(-((r - R) ** 2) / (2.0 * w**2))

    sigma = np.zeros((3, 3), dtype=float)
    sigma[2, 2] = sigma0 * f   # only σ_zz non-zero (simplified uniaxial)
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


# ---------- scalar field φ(x) ----------

def scalar_field_phi(x, y, z, R=10e-9):
    """
    Simple scalar field around the nanoparticle.

    Interpreted as an "interaction potential" that is strongest at the
    nanoparticle surface and decays away:

        φ(r) = exp( -(r-R)^2 / (2 * w^2) )

    This highlights that a scalar field assigns ONE number to each point.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    w = 0.5 * R
    phi = np.exp(-((r - R) ** 2) / (2.0 * w**2))
    return phi


# ---------- vector field u(x) ----------

def vector_field_u(x, y, z, R=10e-9):
    """
    Simple vector field around the nanoparticle.

    Interpreted as a displacement (or force) direction pointing
    radially outward from the nanoparticle center:

        direction = r̂ = (x, y, z) / r
        magnitude = φ(r) from scalar_field_phi

    We return a 3-component vector u = φ(r) * r̂.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero (at the exact center)
    r_safe = np.where(r == 0, 1.0, r)

    phi = scalar_field_phi(x, y, z, R=R)
    ux = phi * x / r_safe
    uy = phi * y / r_safe
    uz = phi * z / r_safe
    return np.array([ux, uy, uz])


# ---------- demos / plotting ----------

def demo_point_calculations():
    """
    Print example scalar, vector and tensor field values at selected points.
    """
    R = 10e-9
    sigma0 = 1.0

    points = {
        "far_field": (0.0, 0.0, 5.0 * R),
        "near_axis_surface": (0.0, 0.0, 1.2 * R),
        "equator_surface": (1.0 * R, 0.0, 0.0),
    }

    print("=== Sample field evaluations (scalar, vector, tensor) ===")
    print(f"Nanoparticle radius R = {R:.3e} m, base σ0 = {sigma0:.2f} (arb. units)\n")

    for name, (x, y, z) in points.items():
        phi = scalar_field_phi(x, y, z, R=R)
        u = vector_field_u(x, y, z, R=R)
        sigma = stress_tensor_field(x, y, z, R=R, sigma0=sigma0)
        vm = von_mises_from_sigma(sigma)

        print(f"Point '{name}': (x,y,z) = ({x:.3e}, {y:.3e}, {z:.3e})")
        print(f"  Scalar field φ      = {phi:.3f}")
        print(f"  Vector field u      = [{u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f}]")
        print("  Tensor field σ (3x3) =")
        with np.printoptions(precision=3, suppress=True):
            print(" ", sigma)
        print(f"  von Mises σ_vm      = {vm:.3f}\n")


def plot_scalar_vector_tensor_slice():
    """
    Create a single window with three subplots on an x–z slice (y=0):

    (a) Scalar field φ(x,z).
    (b) Vector field u(x,z) projected to x–z plane (ux, uz).
    (c) Tensor field visualized via von Mises stress σ_vm(x,z).

    This emphasizes the difference between:
    - scalar field: 1 number per point
    - vector field: direction + magnitude per point
    - tensor field: multi-component object, here reduced to scalar invariant.
    """
    R = 10e-9
    sigma0 = 1.0

    n = 40  # keep it moderate so quiver is readable
    xs = np.linspace(-4 * R, 4 * R, n)
    zs = np.linspace(-4 * R, 4 * R, n)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros_like(X)

    # Containers
    phi_field = np.zeros_like(X)
    vm_field = np.zeros_like(X)
    ux_field = np.zeros_like(X)
    uz_field = np.zeros_like(X)

    for i in range(n):
        for j in range(n):
            x = X[i, j]
            y = Y[i, j]
            z = Z[i, j]

            phi = scalar_field_phi(x, y, z, R=R)
            u = vector_field_u(x, y, z, R=R)
            sigma = stress_tensor_field(x, y, z, R=R, sigma0=sigma0)

            phi_field[i, j] = phi
            ux_field[i, j] = u[0]
            uz_field[i, j] = u[2]
            vm_field[i, j] = von_mises_from_sigma(sigma)

    R_grid = np.sqrt(X**2 + Y**2 + Z**2)

    # Mask inside nanoparticle for the scalar/tensor maps
    phi_field = np.ma.masked_where(R_grid < R, phi_field)
    vm_field = np.ma.masked_where(R_grid < R, vm_field)

    # Normalize vector field for quiver (only direction, magnitude ~ phi)
    mag = np.sqrt(ux_field**2 + uz_field**2)
    mag_safe = np.where(mag == 0, 1.0, mag)
    ux_norm = ux_field / mag_safe
    uz_norm = uz_field / mag_safe

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) Scalar field
    im0 = axes[0].pcolormesh(X / R, Z / R, phi_field, shading="auto", cmap="viridis")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("Scalar field φ(x,z)\n(potential-like)")
    axes[0].set_xlabel("x / R")
    axes[0].set_ylabel("z / R")
    axes[0].set_aspect("equal", "box")
    circle0 = plt.Circle((0, 0), 1.0, color="white", fill=False, linewidth=1.0)
    axes[0].add_patch(circle0)

    # (b) Vector field (quiver)
    skip = 2  # subsample for readability
    axes[1].quiver(
        (X / R)[::skip, ::skip],
        (Z / R)[::skip, ::skip],
        ux_norm[::skip, ::skip],
        uz_norm[::skip, ::skip],
        phi_field[::skip, ::skip],  # color vectors by φ magnitude
        cmap="plasma",
        pivot="mid",
        scale=20,
        minlength=0.1,
    )
    axes[1].set_title("Vector field u(x,z)\n(radial, colored by φ)")
    axes[1].set_xlabel("x / R")
    axes[1].set_ylabel("z / R")
    axes[1].set_aspect("equal", "box")
    circle1 = plt.Circle((0, 0), 1.0, color="black", fill=False, linewidth=1.0)
    axes[1].add_patch(circle1)

    # (c) Tensor field invariant (von Mises)
    im2 = axes[2].pcolormesh(X / R, Z / R, vm_field, shading="auto", cmap="magma")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Tensor field σ(x,z)\nshown as von Mises σ_vm")
    axes[2].set_xlabel("x / R")
    axes[2].set_ylabel("z / R")
    axes[2].set_aspect("equal", "box")
    circle2 = plt.Circle((0, 0), 1.0, color="white", fill=False, linewidth=1.0)
    axes[2].add_patch(circle2)

    plt.tight_layout()
    plt.show()


def main():
    demo_point_calculations()
    print("Generating scalar/vector/tensor field slice plots...")
    plot_scalar_vector_tensor_slice()


if __name__ == "__main__":
    main()
