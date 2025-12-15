"""
Electric Potential - Scalar Field and Energy

Electric potential represents potential energy per unit charge.

Key Formula: V = k_e * q / r

where:
- V: Electric potential in Volts (V)
- q: Source charge in Coulombs (C)
- r: Distance from source in meters (m)

Important Relations:
- E = -∇V (Electric field is negative gradient of potential)
- W = q₀ * ΔV (Work done moving charge through potential difference)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

K_E = 8.987551787e9  # Coulomb's constant (N⋅m²/C²)


def electric_potential(q, r):
    """
    Calculate electric potential from point charge.
    
    V = k_e * q / r
    
    Returns: Potential in Volts (V)
    """
    if r == 0:
        raise ValueError("Distance cannot be zero")
    return K_E * q / r


def potential_energy(q1, q2, r):
    """
    Calculate potential energy of two charges.
    
    U = k_e * q₁ * q₂ / r
    """
    if r == 0:
        raise ValueError("Distance cannot be zero")
    return K_E * q1 * q2 / r


def work_done(q, V_initial, V_final):
    """
    Calculate work done moving charge through potential difference.
    
    W = q * (V_final - V_initial)
    """
    return q * (V_final - V_initial)


def visualize_potential_field():
    """Visualize electric potential around a point charge."""
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    q = 1e-9  # 1 nC
    
    # Calculate potential at each point
    R = np.sqrt(X**2 + Y**2)
    R[R < 0.1] = 0.1  # Avoid singularity
    V = K_E * q / R
    
    fig = plt.figure(figsize=(12, 5))
    
    # 2D contour plot
    ax1 = fig.add_subplot(121)
    contour = ax1.contourf(X, Y, V, levels=20, cmap='coolwarm')
    ax1.plot(0, 0, 'ko', markersize=10, label='Charge')
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)
    ax1.set_title('Electric Potential (2D)', fontsize=12, fontweight='bold')
    ax1.axis('equal')
    plt.colorbar(contour, ax=ax1, label='V (Volts)')
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, V, cmap='coolwarm', alpha=0.8)
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_zlabel('V (V)')
    ax2.set_title('Electric Potential (3D)', fontsize=12, fontweight='bold')
    plt.colorbar(surf, ax=ax2, shrink=0.5)
    
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    print("ELECTRIC POTENTIAL - Scalar Field and Energy")
    print("=" * 60)
    
    # Example 1: Potential from point charge
    q = 1e-6  # 1 μC
    r = 0.1   # 0.1 m
    V = electric_potential(q, r)
    
    print(f"\nExample 1: Potential from point charge")
    print(f"Charge: q = {q*1e6:.1f} μC")
    print(f"Distance: r = {r} m")
    print(f"Potential: V = {V:.2f} V")
    
    # Example 2: Potential energy
    q1, q2 = 1e-6, -2e-6
    r = 0.1
    U = potential_energy(q1, q2, r)
    
    print(f"\nExample 2: Potential energy")
    print(f"Charges: q₁ = {q1*1e6:.1f} μC, q₂ = {q2*1e6:.1f} μC")
    print(f"Distance: r = {r} m")
    print(f"Energy: U = {U:.4f} J")
    
    # Example 3: Work done
    q_test = 1e-9  # 1 nC test charge
    V_i, V_f = 100, 50  # Volts
    W = work_done(q_test, V_i, V_f)
    
    print(f"\nExample 3: Work done moving charge")
    print(f"Test charge: q = {q_test*1e9:.1f} nC")
    print(f"Initial potential: V_i = {V_i} V")
    print(f"Final potential: V_f = {V_f} V")
    print(f"Work done: W = {W*1e9:.2f} nJ")
    
    plt = visualize_potential_field()
    plt.savefig('electric_potential.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'electric_potential.png'")
    plt.show()
