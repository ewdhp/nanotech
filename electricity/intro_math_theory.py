"""
Electricity Foundations - Introduction to Mathematical Theory

This module provides a clear introduction to the most important mathematical
concepts underlying electricity and electrostatics.

Key Mathematical Concepts:
1. Coulomb's Law - Force between charges
2. Electric Field - Vector field representation
3. Electric Potential - Scalar field and energy
4. Gauss's Law - Flux and charge relationships
5. Superposition Principle - Linear field addition
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Physical Constants
EPSILON_0 = 8.854187817e-12  # Permittivity of free space (F/m)
K_E = 8.987551787e9          # Coulomb's constant k = 1/(4πε₀) (N⋅m²/C²)


def coulombs_law(q1, q2, r):
    """
    Coulomb's Law: F = k * |q1*q2| / r²
    
    The fundamental law describing the force between two point charges.
    
    Mathematical Form:
        F = k_e * (q1 * q2) / r²
        where k_e = 1/(4πε₀) ≈ 8.99 × 10⁹ N⋅m²/C²
    
    Parameters:
    -----------
    q1, q2 : float
        Charges in Coulombs (C)
    r : float
        Distance between charges in meters (m)
    
    Returns:
    --------
    float : Force in Newtons (N)
            Positive = repulsive, Negative = attractive
    """
    if r == 0:
        raise ValueError("Distance cannot be zero")
    
    force = K_E * (q1 * q2) / (r ** 2)
    return force


def electric_field_point_charge(q, r):
    """
    Electric Field from a Point Charge: E = k*q / r²
    
    The electric field E is a vector field that describes the force per unit
    charge at any point in space.
    
    Mathematical Definition:
        E = F/q₀ = k * q / r²  (magnitude)
        Direction: radially outward from positive, inward to negative charge
    
    Parameters:
    -----------
    q : float
        Source charge in Coulombs (C)
    r : float
        Distance from charge in meters (m)
    
    Returns:
    --------
    float : Electric field magnitude in N/C or V/m
    """
    if r == 0:
        raise ValueError("Distance cannot be zero")
    
    E = K_E * q / (r ** 2)
    return E


def electric_potential_point_charge(q, r):
    """
    Electric Potential from Point Charge: V = k*q / r
    
    The electric potential is a scalar field representing potential energy
    per unit charge. The negative gradient of potential gives the electric field:
    E = -∇V
    
    Mathematical Form:
        V(r) = k * q / r
        
    Work-Energy Relationship:
        W = q₀ * (V_final - V_initial)
    
    Parameters:
    -----------
    q : float
        Source charge in Coulombs (C)
    r : float
        Distance from charge in meters (m)
    
    Returns:
    --------
    float : Electric potential in Volts (V)
    """
    if r == 0:
        raise ValueError("Distance cannot be zero")
    
    V = K_E * q / r
    return V


def electric_field_vector(q, position, source_position):
    """
    Vector Electric Field: E = k*q*(r - r₀)/|r - r₀|³
    
    Complete vector form of the electric field at position r due to
    a charge q at position r₀.
    
    Mathematical Form:
        E(r) = k*q * (r - r₀) / |r - r₀|³
    
    Parameters:
    -----------
    q : float
        Source charge (C)
    position : array-like
        Observation point [x, y, z] (m)
    source_position : array-like
        Source charge location [x₀, y₀, z₀] (m)
    
    Returns:
    --------
    numpy.array : Electric field vector [Ex, Ey, Ez] in V/m
    """
    r = np.array(position)
    r0 = np.array(source_position)
    
    displacement = r - r0
    distance = np.linalg.norm(displacement)
    
    if distance == 0:
        raise ValueError("Position cannot coincide with source charge")
    
    # E = k*q * (r - r₀) / |r - r₀|³
    E_vector = K_E * q * displacement / (distance ** 3)
    return E_vector


def gauss_law_enclosed_charge(E_flux):
    """
    Gauss's Law: ∮ E·dA = Q_enclosed / ε₀
    
    One of Maxwell's equations, relating the electric flux through a closed
    surface to the enclosed charge.
    
    Integral Form:
        ∮ E·dA = Q_enclosed / ε₀
        
    Differential Form:
        ∇·E = ρ / ε₀
        
    where ρ is charge density
    
    Parameters:
    -----------
    E_flux : float
        Total electric flux through closed surface (V⋅m)
    
    Returns:
    --------
    float : Enclosed charge in Coulombs (C)
    """
    Q_enclosed = EPSILON_0 * E_flux
    return Q_enclosed


def superposition_principle(charges, positions, observation_point):
    """
    Superposition Principle for Electric Fields
    
    The total electric field from multiple charges is the vector sum
    of individual fields:
    
    E_total = E₁ + E₂ + E₃ + ... = Σ E_i
    
    This linearity is fundamental to electromagnetism.
    
    Parameters:
    -----------
    charges : array-like
        List of charges [q1, q2, ...] (C)
    positions : array-like
        List of positions [[x1,y1,z1], [x2,y2,z2], ...] (m)
    observation_point : array-like
        Point where field is calculated [x, y, z] (m)
    
    Returns:
    --------
    numpy.array : Total electric field vector [Ex, Ey, Ez] in V/m
    """
    E_total = np.zeros(3)
    
    for q, pos in zip(charges, positions):
        E_i = electric_field_vector(q, observation_point, pos)
        E_total += E_i
    
    return E_total


def visualize_electric_field_lines():
    """
    Visualize electric field lines for a dipole configuration
    
    This demonstrates the vector nature of electric fields and
    the superposition principle.
    """
    # Create a 2D grid
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    
    # Dipole: positive charge at (+1, 0) and negative at (-1, 0)
    q_pos = 1e-9  # +1 nC
    q_neg = -1e-9  # -1 nC
    pos_charge = np.array([1.0, 0, 0])
    neg_charge = np.array([-1.0, 0, 0])
    
    # Calculate field at each point
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.array([X[j, i], Y[j, i], 0])
            
            # Skip points too close to charges
            if np.linalg.norm(point - pos_charge) < 0.3 or \
               np.linalg.norm(point - neg_charge) < 0.3:
                Ex[j, i] = 0
                Ey[j, i] = 0
                continue
            
            E_field = superposition_principle(
                [q_pos, q_neg],
                [pos_charge, neg_charge],
                point
            )
            Ex[j, i] = E_field[0]
            Ey[j, i] = E_field[1]
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Normalize for better visualization
    magnitude = np.sqrt(Ex**2 + Ey**2)
    magnitude[magnitude == 0] = 1  # Avoid division by zero
    
    plt.streamplot(X, Y, Ex/magnitude, Ey/magnitude, 
                   color=np.log10(magnitude + 1e-10), 
                   cmap='plasma', linewidth=1.5, density=1.5)
    
    # Mark charges
    plt.plot(1, 0, 'ro', markersize=15, label='Positive charge (+)')
    plt.plot(-1, 0, 'bo', markersize=15, label='Negative charge (−)')
    
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('y (m)', fontsize=12)
    plt.title('Electric Field Lines - Electric Dipole', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.colorbar(label='log₁₀(|E|)')
    
    return plt


def demonstrate_key_concepts():
    """
    Demonstrate the five key mathematical concepts in electricity
    """
    print("=" * 70)
    print("ELECTRICITY FOUNDATIONS - Key Mathematical Concepts")
    print("=" * 70)
    
    # 1. Coulomb's Law
    print("\n1. COULOMB'S LAW")
    print("-" * 70)
    q1, q2, r = 1e-6, -2e-6, 0.1  # 1 μC, -2 μC, 0.1 m apart
    force = coulombs_law(q1, q2, r)
    print(f"   Charges: q₁ = {q1*1e6:.1f} μC, q₂ = {q2*1e6:.1f} μC")
    print(f"   Distance: r = {r} m")
    print(f"   Force: F = {force:.4f} N")
    print(f"   → {'Attractive' if force < 0 else 'Repulsive'} force")
    
    # 2. Electric Field
    print("\n2. ELECTRIC FIELD")
    print("-" * 70)
    q, r = 1e-6, 0.1  # 1 μC at 0.1 m
    E = electric_field_point_charge(q, r)
    print(f"   Source charge: q = {q*1e6:.1f} μC")
    print(f"   Distance: r = {r} m")
    print(f"   Electric field: E = {E:.2f} N/C")
    print(f"   → Field points {'away from' if q > 0 else 'towards'} charge")
    
    # 3. Electric Potential
    print("\n3. ELECTRIC POTENTIAL")
    print("-" * 70)
    V = electric_potential_point_charge(q, r)
    print(f"   Source charge: q = {q*1e6:.1f} μC")
    print(f"   Distance: r = {r} m")
    print(f"   Potential: V = {V:.2f} V")
    print(f"   → Work to bring +1 C from ∞ to r = {V:.2f} J")
    
    # 4. Gauss's Law
    print("\n4. GAUSS'S LAW")
    print("-" * 70)
    Q = 1e-9  # 1 nC
    # For a spherical surface around point charge: Flux = Q/ε₀
    flux = Q / EPSILON_0
    Q_calc = gauss_law_enclosed_charge(flux)
    print(f"   Enclosed charge: Q = {Q*1e9:.1f} nC")
    print(f"   Electric flux: Φ = {flux:.4e} V⋅m")
    print(f"   Calculated Q from flux: {Q_calc*1e9:.1f} nC")
    print(f"   → Gauss's law verified: ∮ E·dA = Q/ε₀")
    
    # 5. Superposition Principle
    print("\n5. SUPERPOSITION PRINCIPLE")
    print("-" * 70)
    charges = [1e-9, -1e-9, 2e-9]  # nC
    positions = [[1, 0, 0], [-1, 0, 0], [0, 1, 0]]  # meters
    obs_point = [0, 0, 0]  # origin
    E_total = superposition_principle(charges, positions, obs_point)
    E_mag = np.linalg.norm(E_total)
    print(f"   Charges: {[f'{q*1e9:.1f} nC' for q in charges]}")
    print(f"   Positions: {positions} (m)")
    print(f"   Observation point: {obs_point} (m)")
    print(f"   Total field E = [{E_total[0]:.2f}, {E_total[1]:.2f}, {E_total[2]:.2f}] V/m")
    print(f"   Magnitude: |E| = {E_mag:.2f} V/m")
    print(f"   → Linear superposition of individual fields")
    
    print("\n" + "=" * 70)
    print("KEY MATHEMATICAL RELATIONSHIPS:")
    print("=" * 70)
    print("  • Coulomb's Law:    F = k_e * q₁q₂/r²")
    print("  • Electric Field:   E = F/q = k_e * q/r²")
    print("  • Electric Potential: V = k_e * q/r,  E = -∇V")
    print("  • Gauss's Law:      ∮ E·dA = Q_enc/ε₀")
    print("  • Superposition:    E_total = Σ E_i")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_key_concepts()
    
    # Create visualization
    print("Generating electric field visualization...")
    plt = visualize_electric_field_lines()
    plt.savefig('electric_field_dipole.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'electric_field_dipole.png'")
    plt.show()
