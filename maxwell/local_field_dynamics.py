"""
Local Field Dynamics – Fields Evolve via Partial Differential Equations

Maxwell's equations are *local* PDEs: the rate of change of a field at point x
depends only on the field and its derivatives at x and nearby points.

Example: Faraday's Law
    ∂B/∂t = -∇×E

The time derivative of B at (x,y,z,t) depends on the spatial curl of E at the
same location. No "action at a distance" — information propagates locally.

For demonstration, we use a simpler diffusion-like PDE to show the concept:

    ∂u/∂t = D * ∇²u

where u(x,y,t) is a scalar field and D is a diffusion coefficient.
This is analogous to how disturbances in fields spread out over time.

We simulate this on a 2D grid with an initial localized "pulse" and watch it
diffuse according to local dynamics (each grid point only "talks" to neighbors).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def laplacian_2d(u, dx, dy):
    """
    Compute ∇²u = ∂²u/∂x² + ∂²u/∂y² using finite differences.
    """
    lapl = np.zeros_like(u)
    lapl[1:-1, 1:-1] = (
        (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
        + (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )
    return lapl


def main():
    # Grid
    L = 2.0
    N = 128
    xs = np.linspace(-L, L, N)
    ys = np.linspace(-L, L, N)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys)

    # Initial condition: Gaussian pulse at origin
    sigma = 0.2
    u = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    # Diffusion coefficient
    D = 0.1
    dt = 0.001  # time step (must satisfy stability: dt < dx²/(4*D))
    n_steps = 300

    # Time evolution
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(X, Y, u, shading="auto", cmap="hot", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Field amplitude u(x,y,t)")
    ax.set_title("Local Field Dynamics: Diffusion via ∂u/∂t = D ∇²u\nt = 0.000")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", "box")

    def update(frame):
        nonlocal u
        for _ in range(10):  # multiple sub-steps per frame for smoother animation
            lapl = laplacian_2d(u, dx, dy)
            u += dt * D * lapl
        im.set_array(u.ravel())
        ax.set_title(f"Local Field Dynamics: Diffusion via ∂u/∂t = D ∇²u\nt = {frame*10*dt:.3f}")
        return [im]

    anim = FuncAnimation(fig, update, frames=n_steps//10, interval=30, blit=True)
    plt.show()

    print("=== Local Field Dynamics ===")
    print("Fields evolve based on *local* information encoded in PDEs.")
    print("Here, ∂u/∂t = D ∇²u means:")
    print("  - The rate of change at each point depends on the field's curvature")
    print("    at that point (computed from neighboring grid points).")
    print("  - No instantaneous global changes — diffusion spreads gradually.\n")
    print("Maxwell's equations similarly encode local evolution of E and B fields.\n")


if __name__ == "__main__":
    main()
