"""
Electric Field Theory and Applications
=======================================

This script covers the most important concepts of electric fields including:
- Electric field definition and properties
- Point charges and Coulomb's law
- Electric field lines and visualization
- Electric potential and potential energy
- Gauss's law and electric flux
- Superposition principle
- Dipoles and field patterns
- Uniform fields and capacitors

Theory:
-------
Electric Field E at a point is defined as the force per unit charge:
    E = F/q = (1/4πε₀) × (Q/r²) r̂

Where:
    ε₀ = 8.854×10⁻¹² C²/(N·m²) (permittivity of free space)
    Q = source charge (C)
    r = distance from charge (m)
    r̂ = unit vector in direction from Q to the point

Key Concepts:
------------
1. Electric Field Strength: E (N/C or V/m)
2. Electric Potential: V = -∫E·dl (Volts)
3. Electric Flux: Φ = ∫E·dA (N·m²/C)
4. Gauss's Law: Φ = Q_enclosed/ε₀
5. Superposition: E_total = Σ E_i
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.integrate import odeint


# Physical constants
EPSILON_0 = 8.854187817e-12  # C²/(N·m²) - Permittivity of free space
K_E = 1 / (4 * np.pi * EPSILON_0)  # Coulomb's constant ≈ 8.99×10⁹ N·m²/C²


def electric_field_point_charge(Q, r_source, r_field):
    """
    Calculate electric field from a point charge using Coulomb's law.
    
    E = (1/4πε₀) × (Q/r²) r̂
    
    Parameters:
    -----------
    Q : float
        Charge (Coulombs)
    r_source : array-like, shape (3,)
        Position of the charge (m)
    r_field : array-like, shape (N, 3) or (3,)
        Position(s) where field is calculated (m)
    
    Returns:
    --------
    E : ndarray
        Electric field vector(s) (N/C)
    """
    r_source = np.array(r_source)
    r_field = np.array(r_field)
    
    # Handle single point or array of points
    if r_field.ndim == 1:
        r_field = r_field.reshape(1, -1)
        single_point = True
    else:
        single_point = False
    
    # Vector from source to field point
    r_vec = r_field - r_source
    
    # Distance
    r_mag = np.linalg.norm(r_vec, axis=1, keepdims=True)
    
    # Avoid division by zero
    r_mag[r_mag == 0] = np.inf
    
    # Unit vector
    r_hat = r_vec / r_mag
    
    # Electric field magnitude
    E_mag = K_E * Q / (r_mag ** 2)
    
    # Electric field vector
    E = E_mag * r_hat
    
    if single_point:
        return E[0]
    return E


def electric_field_multiple_charges(charges, positions, r_field):
    """
    Calculate electric field from multiple point charges (superposition).
    
    E_total = Σ E_i
    
    Parameters:
    -----------
    charges : array-like
        Array of charges (Coulombs)
    positions : array-like, shape (N_charges, 3)
        Positions of charges (m)
    r_field : array-like, shape (N_points, 3) or (3,)
        Position(s) where field is calculated (m)
    
    Returns:
    --------
    E_total : ndarray
        Total electric field vector(s) (N/C)
    """
    charges = np.array(charges)
    positions = np.array(positions)
    
    E_total = np.zeros_like(r_field)
    
    for Q, r_source in zip(charges, positions):
        E_total += electric_field_point_charge(Q, r_source, r_field)
    
    return E_total


def electric_potential_point_charge(Q, r_source, r_field):
    """
    Calculate electric potential from a point charge.
    
    V = (1/4πε₀) × (Q/r)
    
    Parameters:
    -----------
    Q : float
        Charge (Coulombs)
    r_source : array-like, shape (3,)
        Position of charge (m)
    r_field : array-like, shape (N, 3) or (3,)
        Position(s) where potential is calculated (m)
    
    Returns:
    --------
    V : float or ndarray
        Electric potential (Volts)
    """
    r_source = np.array(r_source)
    r_field = np.array(r_field)
    
    if r_field.ndim == 1:
        r_field = r_field.reshape(1, -1)
        single_point = True
    else:
        single_point = False
    
    # Distance
    r_vec = r_field - r_source
    r_mag = np.linalg.norm(r_vec, axis=1)
    
    # Avoid division by zero
    r_mag[r_mag == 0] = np.inf
    
    # Electric potential
    V = K_E * Q / r_mag
    
    if single_point:
        return V[0]
    return V


def electric_dipole_field(p, r_dipole, r_field):
    """
    Calculate electric field from a dipole (far field approximation).
    
    For a dipole with moment p = q×d:
    E ≈ (1/4πε₀) × (1/r³) × [3(p·r̂)r̂ - p]
    
    Parameters:
    -----------
    p : array-like, shape (3,)
        Dipole moment vector (C·m)
    r_dipole : array-like, shape (3,)
        Position of dipole center (m)
    r_field : array-like, shape (N, 3) or (3,)
        Position(s) where field is calculated (m)
    
    Returns:
    --------
    E : ndarray
        Electric field vector(s) (N/C)
    """
    p = np.array(p)
    r_dipole = np.array(r_dipole)
    r_field = np.array(r_field)
    
    if r_field.ndim == 1:
        r_field = r_field.reshape(1, -1)
        single_point = True
    else:
        single_point = False
    
    # Vector from dipole to field point
    r_vec = r_field - r_dipole
    r_mag = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r_hat = r_vec / r_mag
    
    # Dipole field formula
    p_dot_r = np.sum(p * r_hat, axis=1, keepdims=True)
    E = (K_E / r_mag**3) * (3 * p_dot_r * r_hat - p)
    
    if single_point:
        return E[0]
    return E


def demonstrate_coulomb_law():
    """Demonstrate Coulomb's law and electric field calculations."""
    print("=" * 70)
    print("COULOMB'S LAW AND ELECTRIC FIELD")
    print("=" * 70)
    
    # Example 1: Single point charge
    print("\n1. Electric field from a single point charge")
    print("-" * 70)
    Q = 1e-9  # 1 nC
    r_source = np.array([0, 0, 0])
    r_field = np.array([0.1, 0, 0])  # 10 cm away
    
    E = electric_field_point_charge(Q, r_source, r_field)
    V = electric_potential_point_charge(Q, r_source, r_field)
    
    print(f"Charge: Q = {Q*1e9:.1f} nC at {r_source}")
    print(f"Field point: r = {r_field} m")
    print(f"Distance: |r| = {np.linalg.norm(r_field - r_source):.3f} m")
    print(f"Electric field: E = {E} N/C")
    print(f"Magnitude: |E| = {np.linalg.norm(E):.3f} N/C")
    print(f"Electric potential: V = {V:.3f} V")
    
    # Example 2: Electric dipole
    print("\n2. Electric dipole")
    print("-" * 70)
    q = 1e-9  # 1 nC
    d = 0.01  # 1 cm separation
    
    # Two charges
    charges = [q, -q]
    positions = np.array([[-d/2, 0, 0], [d/2, 0, 0]])
    r_field = np.array([0, 0.1, 0])  # Point on perpendicular bisector
    
    E_dipole = electric_field_multiple_charges(charges, positions, r_field)
    
    print(f"Dipole: +{q*1e9:.1f} nC at {positions[0]}, -{q*1e9:.1f} nC at {positions[1]}")
    print(f"Separation: d = {d*100:.1f} cm")
    print(f"Dipole moment: p = qd = {q*d*1e12:.2f} pC·m")
    print(f"Field point: r = {r_field} m")
    print(f"Electric field: E = {E_dipole} N/C")
    print(f"Magnitude: |E| = {np.linalg.norm(E_dipole):.3e} N/C")
    
    # Example 3: Multiple charges
    print("\n3. Superposition principle - multiple charges")
    print("-" * 70)
    charges = [1e-9, -2e-9, 1.5e-9]  # nC
    positions = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]])
    r_field = np.array([0.05, 0.05, 0])
    
    E_total = electric_field_multiple_charges(charges, positions, r_field)
    
    print(f"Charges: {[f'{q*1e9:.1f} nC' for q in charges]}")
    print(f"Positions: {positions.tolist()} m")
    print(f"Field point: r = {r_field} m")
    print(f"Total electric field: E = {E_total} N/C")
    print(f"Magnitude: |E| = {np.linalg.norm(E_total):.3f} N/C")


def demonstrate_gauss_law():
    """Demonstrate Gauss's law applications."""
    print("\n" + "=" * 70)
    print("GAUSS'S LAW")
    print("=" * 70)
    print("\nΦ = ∮ E·dA = Q_enclosed/ε₀")
    
    # Example 1: Spherical charge distribution
    print("\n1. Uniform spherical charge distribution")
    print("-" * 70)
    Q_total = 1e-9  # 1 nC
    R = 0.05  # 5 cm radius
    
    # Field outside sphere (r > R)
    r_outside = 0.1  # 10 cm
    E_outside = K_E * Q_total / r_outside**2
    
    # Field inside sphere (r < R)
    r_inside = 0.03  # 3 cm
    E_inside = K_E * Q_total * r_inside / R**3
    
    print(f"Total charge: Q = {Q_total*1e9:.1f} nC")
    print(f"Sphere radius: R = {R*100:.1f} cm")
    print(f"\nOutside sphere (r = {r_outside*100:.1f} cm > R):")
    print(f"  E = Q/(4πε₀r²) = {E_outside:.3f} N/C")
    print(f"\nInside sphere (r = {r_inside*100:.1f} cm < R):")
    print(f"  E = Qr/(4πε₀R³) = {E_inside:.3f} N/C")
    
    # Example 2: Infinite line charge
    print("\n2. Infinite line charge")
    print("-" * 70)
    lambda_charge = 1e-9  # 1 nC/m
    r = 0.05  # 5 cm from line
    
    E_line = lambda_charge / (2 * np.pi * EPSILON_0 * r)
    
    print(f"Linear charge density: λ = {lambda_charge*1e9:.1f} nC/m")
    print(f"Distance from line: r = {r*100:.1f} cm")
    print(f"Electric field: E = λ/(2πε₀r) = {E_line:.3f} N/C")
    
    # Example 3: Infinite plane
    print("\n3. Infinite plane of charge")
    print("-" * 70)
    sigma = 1e-6  # 1 μC/m²
    
    E_plane = sigma / (2 * EPSILON_0)
    
    print(f"Surface charge density: σ = {sigma*1e6:.1f} μC/m²")
    print(f"Electric field: E = σ/(2ε₀) = {E_plane:.3e} N/C")
    print("(Field is uniform, independent of distance!)")


def visualize_field_lines_2d():
    """Visualize electric field lines in 2D."""
    print("\n" + "=" * 70)
    print("GENERATING 2D ELECTRIC FIELD VISUALIZATIONS")
    print("=" * 70)
    
    # Create grid
    x = np.linspace(-0.3, 0.3, 30)
    y = np.linspace(-0.3, 0.3, 30)
    X, Y = np.meshgrid(x, y)
    
    # Configuration 1: Single positive charge
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Single positive charge
    ax = axes[0, 0]
    Q1 = 1e-9
    pos1 = np.array([0, 0, 0])
    
    E_field = np.zeros((len(y), len(x), 2))
    for i, yi in enumerate(y):
        for j, xi in enumerate(x):
            r = np.array([xi, yi, 0])
            E = electric_field_point_charge(Q1, pos1, r)
            E_field[i, j] = E[:2]
    
    Ex, Ey = E_field[:, :, 0], E_field[:, :, 1]
    magnitude = np.sqrt(Ex**2 + Ey**2)
    
    ax.streamplot(X, Y, Ex, Ey, density=2, linewidth=1, color=magnitude, 
                  cmap='plasma', arrowsize=1.5)
    ax.plot(pos1[0], pos1[1], 'ro', markersize=15, label=f'+{Q1*1e9:.0f} nC')
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_title('Single Positive Charge', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 2. Electric dipole
    ax = axes[0, 1]
    Q2 = 1e-9
    pos_plus = np.array([-0.05, 0, 0])
    pos_minus = np.array([0.05, 0, 0])
    
    E_field = np.zeros((len(y), len(x), 2))
    for i, yi in enumerate(y):
        for j, xi in enumerate(x):
            r = np.array([xi, yi, 0])
            E = electric_field_multiple_charges([Q2, -Q2], [pos_plus, pos_minus], r)
            E_field[i, j] = E[:2]
    
    Ex, Ey = E_field[:, :, 0], E_field[:, :, 1]
    magnitude = np.sqrt(Ex**2 + Ey**2)
    
    ax.streamplot(X, Y, Ex, Ey, density=2, linewidth=1, color=magnitude, 
                  cmap='plasma', arrowsize=1.5)
    ax.plot(pos_plus[0], pos_plus[1], 'ro', markersize=15, label=f'+{Q2*1e9:.0f} nC')
    ax.plot(pos_minus[0], pos_minus[1], 'bo', markersize=15, label=f'-{Q2*1e9:.0f} nC')
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_title('Electric Dipole', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 3. Two positive charges
    ax = axes[1, 0]
    Q3 = 1e-9
    pos1 = np.array([-0.07, 0, 0])
    pos2 = np.array([0.07, 0, 0])
    
    E_field = np.zeros((len(y), len(x), 2))
    for i, yi in enumerate(y):
        for j, xi in enumerate(x):
            r = np.array([xi, yi, 0])
            E = electric_field_multiple_charges([Q3, Q3], [pos1, pos2], r)
            E_field[i, j] = E[:2]
    
    Ex, Ey = E_field[:, :, 0], E_field[:, :, 1]
    magnitude = np.sqrt(Ex**2 + Ey**2)
    
    ax.streamplot(X, Y, Ex, Ey, density=2, linewidth=1, color=magnitude, 
                  cmap='plasma', arrowsize=1.5)
    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'ro', markersize=15, 
            label=f'+{Q3*1e9:.0f} nC each')
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_title('Two Positive Charges (Repulsion)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 4. Quadrupole
    ax = axes[1, 1]
    Q4 = 1e-9
    d = 0.08
    charges = [Q4, -Q4, Q4, -Q4]
    positions = np.array([[d/2, d/2, 0], [-d/2, d/2, 0], 
                          [-d/2, -d/2, 0], [d/2, -d/2, 0]])
    
    E_field = np.zeros((len(y), len(x), 2))
    for i, yi in enumerate(y):
        for j, xi in enumerate(x):
            r = np.array([xi, yi, 0])
            E = electric_field_multiple_charges(charges, positions, r)
            E_field[i, j] = E[:2]
    
    Ex, Ey = E_field[:, :, 0], E_field[:, :, 1]
    magnitude = np.sqrt(Ex**2 + Ey**2)
    
    ax.streamplot(X, Y, Ex, Ey, density=2, linewidth=1, color=magnitude, 
                  cmap='plasma', arrowsize=1.5)
    for i, (q, pos) in enumerate(zip(charges, positions)):
        color = 'r' if q > 0 else 'b'
        sign = '+' if q > 0 else '-'
        ax.plot(pos[0], pos[1], f'{color}o', markersize=12)
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_title('Quadrupole Configuration', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('electric_field_lines.png', dpi=150, bbox_inches='tight')
    print("\nSaved: electric_field_lines.png")
    
    return fig


def visualize_potential_3d():
    """Visualize electric potential in 3D."""
    print("\n" + "=" * 70)
    print("GENERATING 3D POTENTIAL VISUALIZATION")
    print("=" * 70)
    
    # Create grid
    x = np.linspace(-0.2, 0.2, 50)
    y = np.linspace(-0.2, 0.2, 50)
    X, Y = np.meshgrid(x, y)
    
    # Electric dipole
    Q = 1e-9
    pos_plus = np.array([-0.05, 0, 0])
    pos_minus = np.array([0.05, 0, 0])
    
    V_total = np.zeros_like(X)
    for i in range(len(y)):
        for j in range(len(x)):
            r = np.array([X[i, j], Y[i, j], 0])
            V_plus = electric_potential_point_charge(Q, pos_plus, r)
            V_minus = electric_potential_point_charge(-Q, pos_minus, r)
            V_total[i, j] = V_plus + V_minus
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X, Y, V_total, cmap='RdBu_r', alpha=0.8, 
                            vmin=-100, vmax=100)
    ax1.set_xlabel('x (m)', fontsize=10)
    ax1.set_ylabel('y (m)', fontsize=10)
    ax1.set_zlabel('V (Volts)', fontsize=10)
    ax1.set_title('Electric Potential (3D)', fontsize=11, fontweight='bold')
    plt.colorbar(surf, ax=ax1, shrink=0.5, label='Potential (V)')
    
    # Contour plot
    ax2 = fig.add_subplot(132)
    contours = ax2.contour(X, Y, V_total, levels=20, cmap='RdBu_r', 
                           vmin=-100, vmax=100)
    ax2.clabel(contours, inline=True, fontsize=8)
    ax2.plot(pos_plus[0], pos_plus[1], 'ro', markersize=12, label='+Q')
    ax2.plot(pos_minus[0], pos_minus[1], 'bo', markersize=12, label='-Q')
    ax2.set_xlabel('x (m)', fontsize=10)
    ax2.set_ylabel('y (m)', fontsize=10)
    ax2.set_title('Equipotential Lines', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Filled contour
    ax3 = fig.add_subplot(133)
    cf = ax3.contourf(X, Y, V_total, levels=50, cmap='RdBu_r', 
                      vmin=-100, vmax=100)
    ax3.plot(pos_plus[0], pos_plus[1], 'ro', markersize=12, label='+Q')
    ax3.plot(pos_minus[0], pos_minus[1], 'bo', markersize=12, label='-Q')
    ax3.set_xlabel('x (m)', fontsize=10)
    ax3.set_ylabel('y (m)', fontsize=10)
    ax3.set_title('Potential Distribution', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.set_aspect('equal')
    plt.colorbar(cf, ax=ax3, label='Potential (V)')
    
    plt.tight_layout()
    plt.savefig('electric_potential.png', dpi=150, bbox_inches='tight')
    print("\nSaved: electric_potential.png")
    
    return fig


def main():
    """Main function demonstrating electric field concepts."""
    print("\n" + "=" * 70)
    print("ELECTRIC FIELD THEORY AND APPLICATIONS")
    print("=" * 70)
    print(f"\nCoulomb's constant: k = 1/(4πε₀) = {K_E:.3e} N·m²/C²")
    print(f"Permittivity of free space: ε₀ = {EPSILON_0:.3e} C²/(N·m²)")
    
    # Demonstrate concepts
    demonstrate_coulomb_law()
    demonstrate_gauss_law()
    
    # Visualizations
    try:
        print("\nGenerating visualizations...")
        fig1 = visualize_field_lines_2d()
        fig2 = visualize_potential_3d()
        
        print("\n" + "=" * 70)
        print("All visualizations complete!")
        print("Saved files:")
        print("  - electric_field_lines.png")
        print("  - electric_potential.png")
        print("\nDisplaying interactive plots...")
        print("Close the plot windows to exit.")
        print("=" * 70)
        plt.show()
        
    except Exception as e:
        print(f"\nVisualization error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
