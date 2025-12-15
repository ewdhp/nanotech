"""
Superposition Principle - Linear Field Addition

The total electric field from multiple charges is the vector sum of 
individual fields. This linearity is fundamental to electromagnetism.

Key Concept: E_total = E₁ + E₂ + E₃ + ... = Σ E_i

This applies to:
- Electric fields
- Electric potentials
- Forces on charges
"""

import numpy as np
import matplotlib.pyplot as plt

K_E = 8.987551787e9  # Coulomb's constant (N⋅m²/C²)


def field_from_charge(q, obs_point, charge_pos):
    """Calculate electric field vector from single charge."""
    r = np.array(obs_point)
    r0 = np.array(charge_pos)
    displacement = r - r0
    distance = np.linalg.norm(displacement)
    
    if distance == 0:
        raise ValueError("Observation point cannot coincide with charge")
    
    E = K_E * q * displacement / (distance ** 3)
    return E


def superposition_field(charges, positions, obs_point):
    """
    Calculate total electric field using superposition principle.
    
    E_total = Σ E_i
    
    Parameters:
    - charges: list of charges [q1, q2, ...]
    - positions: list of positions [[x1,y1,z1], [x2,y2,z2], ...]
    - obs_point: observation point [x, y, z]
    
    Returns: Total field vector [Ex, Ey, Ez]
    """
    E_total = np.zeros(3)
    
    for q, pos in zip(charges, positions):
        E_i = field_from_charge(q, obs_point, pos)
        E_total += E_i
    
    return E_total


def superposition_potential(charges, positions, obs_point):
    """
    Calculate total electric potential using superposition.
    
    V_total = Σ V_i = Σ (k_e * q_i / r_i)
    """
    V_total = 0
    obs = np.array(obs_point)
    
    for q, pos in zip(charges, positions):
        r = np.linalg.norm(obs - np.array(pos))
        if r > 0:
            V_total += K_E * q / r
    
    return V_total


def visualize_dipole():
    """Visualize electric field from dipole using superposition."""
    x = np.linspace(-3, 3, 25)
    y = np.linspace(-3, 3, 25)
    X, Y = np.meshgrid(x, y)
    
    # Electric dipole configuration
    charges = [1e-9, -1e-9]  # +1 nC and -1 nC
    positions = [[1, 0, 0], [-1, 0, 0]]
    
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    
    for i in range(len(x)):
        for j in range(len(y)):
            point = [X[j, i], Y[j, i], 0]
            
            # Skip points too close to charges
            too_close = False
            for pos in positions:
                if np.linalg.norm(np.array(point) - np.array(pos)) < 0.2:
                    too_close = True
                    break
            
            if too_close:
                Ex[j, i] = 0
                Ey[j, i] = 0
                continue
            
            # Apply superposition principle
            E = superposition_field(charges, positions, point)
            Ex[j, i] = E[0]
            Ey[j, i] = E[1]
    
    plt.figure(figsize=(12, 5))
    
    # Field lines
    plt.subplot(121)
    magnitude = np.sqrt(Ex**2 + Ey**2)
    magnitude[magnitude == 0] = 1
    plt.streamplot(X, Y, Ex/magnitude, Ey/magnitude,
                   color=np.log10(magnitude + 1e-10),
                   cmap='plasma', linewidth=1.5, density=1.8)
    plt.plot(1, 0, 'ro', markersize=15, label='+q')
    plt.plot(-1, 0, 'bo', markersize=15, label='−q')
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('y (m)', fontsize=12)
    plt.title('Electric Field (Superposition)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.colorbar(label='log₁₀(|E|)')
    
    # Potential
    plt.subplot(122)
    V = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            point = [X[j, i], Y[j, i], 0]
            V[j, i] = superposition_potential(charges, positions, point)
    
    contour = plt.contourf(X, Y, V, levels=30, cmap='RdBu_r')
    plt.plot(1, 0, 'ro', markersize=15, label='+q')
    plt.plot(-1, 0, 'bo', markersize=15, label='−q')
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('y (m)', fontsize=12)
    plt.title('Electric Potential (Superposition)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.axis('equal')
    plt.colorbar(contour, label='V (Volts)')
    
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    print("SUPERPOSITION PRINCIPLE - Linear Field Addition")
    print("=" * 60)
    
    # Example 1: Three charges
    charges = [1e-9, -1e-9, 2e-9]  # nC
    positions = [[1, 0, 0], [-1, 0, 0], [0, 1, 0]]
    obs_point = [0, 0, 0]
    
    E_total = superposition_field(charges, positions, obs_point)
    E_mag = np.linalg.norm(E_total)
    
    print(f"\nExample 1: Field from three charges")
    print(f"Charges: {[f'{q*1e9:.1f} nC' for q in charges]}")
    print(f"Positions: {positions}")
    print(f"Observation point: {obs_point}")
    print(f"Total field: E = [{E_total[0]:.2f}, {E_total[1]:.2f}, {E_total[2]:.2f}] V/m")
    print(f"Magnitude: |E| = {E_mag:.2f} V/m")
    
    # Example 2: Potential from same charges
    V_total = superposition_potential(charges, positions, obs_point)
    print(f"\nExample 2: Potential at same point")
    print(f"Total potential: V = {V_total:.2f} V")
    
    # Example 3: Compare individual contributions
    print(f"\nExample 3: Individual field contributions")
    for i, (q, pos) in enumerate(zip(charges, positions), 1):
        E_i = field_from_charge(q, obs_point, pos)
        print(f"  Charge {i}: E_{i} = [{E_i[0]:.2f}, {E_i[1]:.2f}, {E_i[2]:.2f}] V/m")
    print(f"  Sum: E_total = [{E_total[0]:.2f}, {E_total[1]:.2f}, {E_total[2]:.2f}] V/m")
    
    plt = visualize_dipole()
    plt.savefig('superposition_principle.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'superposition_principle.png'")
    plt.show()
