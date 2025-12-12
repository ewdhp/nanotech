"""
Wave Solutions – Electromagnetic Waves Traveling at Speed c

In vacuum (no charges or currents), Maxwell's equations combine to give wave
equations for E and B:

    ∇²E - (1/c²) ∂²E/∂t² = 0
    ∇²B - (1/c²) ∂²B/∂t² = 0

where c = 1 / √(μ₀ ε₀) ≈ 3×10⁸ m/s.

These predict that disturbances in E and B propagate as waves at speed c.

For simplicity, we simulate a 1D wave equation:

    ∂²u/∂t² = c² ∂²u/∂x²

with initial conditions representing a localized pulse. We watch it propagate.

Then we show a plane-wave solution:
    E_y(x,t) = E₀ sin(k x - ω t)
    B_z(x,t) = (E₀/c) sin(k x - ω t)

with ω = c k, illustrating E ⊥ B and both perpendicular to propagation direction.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


C = 3e8  # speed of light (m/s)
EPS0 = 8.854187817e-12
MU0 = 4e-7 * np.pi


def wave_1d_step(u, u_prev, dx, dt, c):
    """
    Time-step the 1D wave equation ∂²u/∂t² = c² ∂²u/∂x² using finite differences.

    u_next[i] = 2*u[i] - u_prev[i] + (c*dt/dx)² * (u[i+1] - 2*u[i] + u[i-1])
    """
    r = (c * dt / dx)**2
    u_next = np.zeros_like(u)
    u_next[1:-1] = (
        2*u[1:-1] - u_prev[1:-1]
        + r * (u[2:] - 2*u[1:-1] + u[:-2])
    )
    # boundary: simple fixed ends
    u_next[0] = 0
    u_next[-1] = 0
    return u_next


def main():
    # --- Part 1: 1D pulse propagation ---
    L = 10.0  # domain length (m)
    N = 500
    xs = np.linspace(0, L, N)
    dx = xs[1] - xs[0]
    dt = 0.5 * dx / C  # CFL condition for stability

    # Initial Gaussian pulse at x=2
    x0 = 2.0
    sigma = 0.3
    u = np.exp(-((xs - x0)**2) / (2*sigma**2))
    u_prev = u.copy()

    n_steps = 300

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # (a) 1D wave propagation
    line1, = axes[0].plot(xs, u, 'b-', linewidth=2)
    axes[0].set_xlim(0, L)
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].set_title("1D Wave Propagation: ∂²u/∂t² = c² ∂²u/∂x²\nt = 0.000 ns")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("u(x,t)")
    axes[0].grid(True)

    def update_wave(frame):
        nonlocal u, u_prev
        for _ in range(5):
            u_next = wave_1d_step(u, u_prev, dx, dt, C)
            u_prev = u.copy()
            u = u_next
        line1.set_ydata(u)
        axes[0].set_title(f"1D Wave Propagation: ∂²u/∂t² = c² ∂²u/∂x²\nt = {frame*5*dt*1e9:.3f} ns")
        return [line1]

    anim1 = FuncAnimation(fig, update_wave, frames=n_steps//5, interval=30, blit=True)

    # --- Part 2: Plane wave E_y(x,t), B_z(x,t) ---
    # k and ω chosen so ω = c k (dispersion relation for EM waves)
    wavelength = 2.0  # meters
    k = 2 * np.pi / wavelength
    omega = C * k
    E0 = 1.0  # arbitrary amplitude

    x_plane = np.linspace(0, 3*wavelength, 200)
    t_frames = np.linspace(0, 2*np.pi/omega, 60)  # one full period

    Ey_init = E0 * np.sin(k * x_plane)
    Bz_init = (E0 / C) * np.sin(k * x_plane)

    line2, = axes[1].plot(x_plane, Ey_init, 'r-', label=r'$E_y(x,t)$')
    line3, = axes[1].plot(x_plane, Bz_init * 1e8, 'b--', label=r'$B_z(x,t) \times 10^8$ (scaled)')
    axes[1].set_xlim(0, 3*wavelength)
    axes[1].set_ylim(-1.5*E0, 1.5*E0)
    axes[1].set_title(r"Plane Wave: $E_y = E_0 \sin(kx - \omega t)$, $B_z = (E_0/c) \sin(kx - \omega t)$" + "\nt = 0.000 ns")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("Field amplitude")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    frame_idx = [0]

    def update_plane(frame):
        t = t_frames[frame_idx[0] % len(t_frames)]
        Ey = E0 * np.sin(k * x_plane - omega * t)
        Bz = (E0 / C) * np.sin(k * x_plane - omega * t)
        line2.set_ydata(Ey)
        line3.set_ydata(Bz * 1e8)
        axes[1].set_title(
            r"Plane Wave: $E_y = E_0 \sin(kx - \omega t)$, $B_z = (E_0/c) \sin(kx - \omega t)$"
            + f"\nt = {t*1e9:.3f} ns"
        )
        frame_idx[0] += 1
        return [line2, line3]

    anim2 = FuncAnimation(fig, update_plane, frames=len(t_frames), interval=50, blit=True)

    plt.tight_layout()
    plt.show()

    print("=== Wave Solutions ===")
    print(f"Speed of light c = 1 / √(μ₀ ε₀) = {C:.3e} m/s")
    print("Maxwell's equations predict that E and B propagate as waves at c.")
    print("Top plot: arbitrary 1D pulse moving at c.")
    print("Bottom plot: sinusoidal plane wave with E ⊥ B, both ⊥ propagation direction.\n")


if __name__ == "__main__":
    main()
