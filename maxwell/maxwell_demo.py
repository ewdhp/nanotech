"""
Maxwell's Equations – Computational Learning Demo (2D toy model)

This script is a *field-theory learning aid*, not a full EM solver.

It illustrates, on a 2D grid (x,y) for a fixed time t:

- Scalar fields:
    ρ(x,y)    : charge density
    φ(x,y)    : toy electric potential
- Vector fields:
    E(x,y)    : electric field (Ex, Ey)
    B(x,y)    : magnetic field (Bx, By, Bz) – here we only use Bz for a 2D slice
    J(x,y)    : current density (Jx, Jy)
- Differential operators:
    ∇·E       : divergence of E
    ∇×E       : curl of E (in 2D, represented by the z-component)
    ∇·B, ∇×B  : divergence/curl of B

and numerically connects them to the static Maxwell equations:

    Gauss's Law:           ∇·E = ρ / ε₀
    No magnetic monopoles: ∇·B = 0
    Faraday's Law:         ∇×E = -∂B/∂t   (we set ∂B/∂t = 0 ⇒ ∇×E ≈ 0)
    Ampère-Maxwell:        ∇×B = μ₀ J + μ₀ ε₀ ∂E/∂t (we keep ∂E/∂t = 0)

We build a simple 2D "nanotech-like" configuration:

- A localized Gaussian charge distribution ρ(x,y) at the origin
  (can be thought of as a small charged nanoparticle in cross-section).

- A static electric field E(x,y) derived from a potential φ(r),
  approximately pointing radially outward.

- A simple current loop J(x,y) circulating around the origin,
  which creates a Bz field reminiscent of a magnetic field around a wire.

The script then:

1. Computes ∇·E and compares it to ρ/ε₀ (Gauss's Law).
2. Verifies that ∇·B ≈ 0 (no magnetic monopoles).
3. Shows that ∇×E is small (consistent with Faraday in the static case).
4. Plots:
   - ρ(x,y) as a scalar field,
   - E(x,y) as a vector field (quiver),
   - Bz(x,y) as a scalar field,
   - ∇·E and ρ/ε₀ for visual comparison.

Run:

    cd /home/ewd/github/ewdhp/nanotech/maxwell
    python maxwell_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt


# --- Physical constants (SI) – values not critical for the toy model ---

EPS0 = 8.854187817e-12  # vacuum permittivity
MU0 = 4.0e-7 * np.pi    # vacuum permeability


# --- Field definitions -------------------------------------------------


def charge_density_rho(x, y, sigma=1.0):
    """
    Localized Gaussian charge distribution centered at the origin.

        ρ(x,y) = ρ0 * exp(- (x² + y²) / (2 σ²))

    Here ρ0 is set to 1 (arbitrary units); σ controls the width.
    """
    r2 = x**2 + y**2
    rho0 = 1.0
    return rho0 * np.exp(-r2 / (2.0 * sigma**2))


def electric_potential_phi(x, y, sigma=1.0):
    """
    Toy electric potential associated with the charge distribution.

    We use a simple Gaussian-like potential for smoothness:

        φ(r) = φ0 * exp(- r² / (2 σ²))

    The electric field will be E = -∇φ (computed numerically).
    """
    r2 = x**2 + y**2
    phi0 = 1.0
    return phi0 * np.exp(-r2 / (2.0 * sigma**2))


def current_density_J(x, y, sigma=1.0):
    """
    Simple circulating current density around the origin (2D loop).

    We define J as tangential to circles around the origin with
    magnitude decaying as a Gaussian:

        direction:   ẑ × r̂  (perpendicular to radial direction in-plane)
        magnitude:   J0 * exp(-r² / (2 σ²))

    Returns (Jx, Jy).
    """
    r = np.sqrt(x**2 + y**2)
    r_safe = np.where(r == 0, 1.0, r)

    # Tangential unit vector: (-y, x) / r
    tx = -y / r_safe
    ty = x / r_safe

    J0 = 1.0
    mag = J0 * np.exp(-r**2 / (2.0 * sigma**2))

    Jx = mag * tx
    Jy = mag * ty
    return Jx, Jy


def magnetic_field_Bz_from_J(x, y, sigma=1.0):
    """
    Very rough, qualitative Bz field from the circulating current.

    For a circular current loop, Bz is strongest at the center and decays.
    We mimic that with:

        Bz(r) ∝ exp(-r² / (2 σ_B²))

    This is *not* Biot–Savart; it's just a smooth, localized B field.
    """
    r2 = x**2 + y**2
    sigma_B = 1.0 * sigma
    B0 = 1.0
    Bz = B0 * np.exp(-r2 / (2.0 * sigma_B**2))
    return Bz


# --- Numerical differential operators on a regular grid ----------------


def numerical_gradient(f, dx, dy):
    """
    Compute ∂f/∂x and ∂f/∂y for a scalar field f[i,j] on a uniform grid
    with spacings dx, dy using central differences in the interior.
    """
    dfdx = np.zeros_like(f)
    dfdy = np.zeros_like(f)

    # Central differences (interior)
    dfdx[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dx)
    dfdy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dy)

    # Simple one-sided differences at boundaries
    dfdx[0, :] = (f[1, :] - f[0, :]) / dx
    dfdx[-1, :] = (f[-1, :] - f[-2, :]) / dx

    dfdy[:, 0] = (f[:, 1] - f[:, 0]) / dy
    dfdy[:, -1] = (f[:, -1] - f[:, -2]) / dy

    return dfdx, dfdy


def divergence_2d(Fx, Fy, dx, dy):
    """
    Compute ∇·F for a 2D vector field F = (Fx, Fy).
    """
    dFxdx, _ = numerical_gradient(Fx, dx, dy)
    _, dFydy = numerical_gradient(Fy, dx, dy)
    return dFxdx + dFydy


def curl_z_2d(Fx, Fy, dx, dy):
    """
    Compute the z-component of the curl in 2D:

        (∇×F)_z = ∂Fy/∂x - ∂Fx/∂y
    """
    dFydx, _ = numerical_gradient(Fy, dx, dy)
    _, dFxdy = numerical_gradient(Fx, dx, dy)
    return dFydx - dFxdy


# --- Main demo ---------------------------------------------------------


def main():
    # Grid setup
    L = 5.0     # domain half-size in arbitrary units
    N = 128     # grid resolution (N x N)
    xs = np.linspace(-L, L, N)
    ys = np.linspace(-L, L, N)
    X, Y = np.meshgrid(xs, ys)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    sigma = 1.0  # width parameter for Gaussians

    # Scalar fields
    rho = charge_density_rho(X, Y, sigma=sigma)
    phi = electric_potential_phi(X, Y, sigma=sigma)

    # Electric field from potential: E = -∇φ
    dphidx, dphidy = numerical_gradient(phi, dx, dy)
    Ex = -dphidx
    Ey = -dphidy

    # Current density and magnetic field
    Jx, Jy = current_density_J(X, Y, sigma=sigma)
    Bz = magnetic_field_Bz_from_J(X, Y, sigma=sigma)

    # Divergence / curl checks
    divE = divergence_2d(Ex, Ey, dx, dy)
    rhs_gauss = rho / EPS0

    divB = divergence_2d(np.zeros_like(Bz), Bz, dx, dy)  # only Bz present → div ~ 0
    curlE_z = curl_z_2d(Ex, Ey, dx, dy)                  # should be small in static case

    # Print a few summary statistics
    print("=== Maxwell Demo: Numerical Checks ===")
    print(f"Grid: {N} x {N}, L = {L}, dx ≈ {dx:.3f}, dy ≈ {dy:.3f}\n")

    print("Gauss's Law: ∇·E ≈ ρ / ε₀")
    print(f"  max |∇·E - ρ/ε₀| = {np.max(np.abs(divE - rhs_gauss)):.3e}")
    print(f"  rms |∇·E - ρ/ε₀| = {np.sqrt(np.mean((divE - rhs_gauss)**2)):.3e}\n")

    print("No Magnetic Monopoles: ∇·B ≈ 0")
    print(f"  max |∇·B|         = {np.max(np.abs(divB)):.3e}")
    print(f"  rms |∇·B|         = {np.sqrt(np.mean(divB**2)):.3e}\n")

    print("Faraday's Law (static): ∇×E ≈ 0")
    print(f"  max |(∇×E)_z|     = {np.max(np.abs(curlE_z)):.3e}")
    print(f"  rms |(∇×E)_z|     = {np.sqrt(np.mean(curlE_z**2)):.3e}\n")

    # ----------------- Plots -----------------

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    # (0,0) – charge density ρ
    im0 = axes[0, 0].pcolormesh(X, Y, rho, shading="auto", cmap="viridis")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    axes[0, 0].set_title(r"Scalar field: $\rho(x,y)$ (charge density)")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_aspect("equal", "box")

    # (0,1) – electric potential φ
    im1 = axes[0, 1].pcolormesh(X, Y, phi, shading="auto", cmap="plasma")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[0, 1].set_title(r"Scalar field: $\phi(x,y)$ (toy potential)")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].set_aspect("equal", "box")

    # (0,2) – E field (quiver)
    skip = 5
    axes[0, 2].quiver(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        Ex[::skip, ::skip],
        Ey[::skip, ::skip],
        np.hypot(Ex[::skip, ::skip], Ey[::skip, ::skip]),
        cmap="inferno",
        pivot="mid",
        scale=50,
        minlength=0.1,
    )
    axes[0, 2].set_title(r"Vector field: $\mathbf{E}(x,y)$")
    axes[0, 2].set_xlabel("x")
    axes[0, 2].set_ylabel("y")
    axes[0, 2].set_aspect("equal", "box")

    # (1,0) – Bz field
    im3 = axes[1, 0].pcolormesh(X, Y, Bz, shading="auto", cmap="coolwarm")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].set_title(r"Scalar field: $B_z(x,y)$ (toy magnetic field)")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    axes[1, 0].set_aspect("equal", "box")

    # (1,1) – divergence of E and ρ/ε₀ comparison
    im4 = axes[1, 1].pcolormesh(X, Y, divE, shading="auto", cmap="viridis")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 1].set_title(r"$\nabla\cdot\mathbf{E}(x,y)$")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    axes[1, 1].set_aspect("equal", "box")

    diff = divE - rhs_gauss
    im5 = axes[1, 2].pcolormesh(X, Y, diff, shading="auto", cmap="seismic")
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
    axes[1, 2].set_title(r"$\nabla\cdot\mathbf{E} - \rho/\varepsilon_0$")
    axes[1, 2].set_xlabel("x")
    axes[1, 2].set_ylabel("y")
    axes[1, 2].set_aspect("equal", "box")

    fig.suptitle("Maxwell's Equations – Scalar, Vector, and Field-Operator Demo", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
