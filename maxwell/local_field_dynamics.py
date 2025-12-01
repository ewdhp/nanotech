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

KEY CONCEPT: At each time step, u(x,y,t+dt) is computed ONLY from u and its
spatial derivatives at (x,y,t) and immediate neighbors. This is the essence of
*local* field dynamics encoded in PDEs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def laplacian_2d(u, dx, dy):
    """
    Compute ∇²u = ∂²u/∂x² + ∂²u/∂y² using finite differences.
    
    This represents local information: the curvature at each point
    computed from neighboring grid points.
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
    dt = 0.0005  # time step (must satisfy stability: dt < dx²/(4*D))
    n_steps = 400
    frames_to_show = 80  # number of animation frames

    print("=== Local Field Dynamics ===")
    print("Simulating ∂u/∂t = D ∇²u on a 2D grid.")
    print("Each grid point evolves based ONLY on its neighbors (local dynamics).")
    print(f"Grid: {N}×{N}, dt = {dt:.4f}, D = {D:.2f}")
    print(f"Total time steps: {n_steps}, showing {frames_to_show} frames.\n")

    # Time evolution with animation
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(X, Y, u, shading="auto", cmap="hot", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Field amplitude u(x,y,t)")
    ax.set_title("Local Field Dynamics: ∂u/∂t = D ∇²u\nt = 0.000")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", "box")

    # Add text annotation explaining locality
    ax.text(
        0.02, 0.98,
        "Each point updates based\nONLY on nearby neighbors\n(local PDE)",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7)
    )

    steps_per_frame = n_steps // frames_to_show

    def update(frame):
        nonlocal u
        for _ in range(steps_per_frame):
            lapl = laplacian_2d(u, dx, dy)
            u += dt * D * lapl
        
        im.set_array(u.ravel())
        current_time = frame * steps_per_frame * dt
        ax.set_title(f"Local Field Dynamics: ∂u/∂t = D ∇²u\nt = {current_time:.3f}")
        return [im]

    anim = FuncAnimation(
        fig, update, frames=frames_to_show,
        interval=50, blit=True, repeat=True
    )

    plt.tight_layout()
    plt.show()  # Explicitly show the animation

    print("Animation complete.")
    print("\nKey takeaway:")
    print("  - The field evolved smoothly from a localized pulse to a diffuse distribution.")
    print("  - At each time step, u(x,y) changed based on ∇²u at (x,y), which depends")
    print("    only on u at neighboring grid points.")
    print("  - This is LOCAL field dynamics: no instantaneous global influence.")
    print("  - Maxwell's equations similarly encode local evolution of E and B fields.\n")


if __name__ == "__main__":
    main()
