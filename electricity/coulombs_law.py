"""
Coulomb's Law - Force Between Charges

Mathematical foundation of electrostatic force between point charges.

Key Formula: F = k_e * (q₁ * q₂) / r²

where:
- F: Force in Newtons (N)
- k_e = 8.99 × 10⁹ N⋅m²/C² (Coulomb's constant)
- q₁, q₂: Charges in Coulombs (C)
- r: Distance between charges in meters (m)
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical Constants
K_E = 8.987551787e9  # Coulomb's constant (N⋅m²/C²)


def coulombs_force(q1, q2, r):
    """
    Calculate electrostatic force between two point charges.
    
    F = k_e * (q₁ * q₂) / r²
    
    Returns:
        Positive value = repulsive force
        Negative value = attractive force
    """
    if r == 0:
        raise ValueError("Distance cannot be zero")
    return K_E * (q1 * q2) / (r ** 2)


def force_vector(q1, q2, pos1, pos2):
    """
    Calculate force vector on q1 due to q2.
    
    F = k_e * q₁ * q₂ * (r₁ - r₂) / |r₁ - r₂|³
    """
    r1 = np.array(pos1)
    r2 = np.array(pos2)
    displacement = r1 - r2
    distance = np.linalg.norm(displacement)
    
    if distance == 0:
        raise ValueError("Charges cannot occupy same position")
    
    force_magnitude = K_E * q1 * q2 / (distance ** 2)
    force_direction = displacement / distance
    return force_magnitude * force_direction


def visualize_force_vs_distance():
    """Visualize how force varies with distance."""
    q1, q2 = 1e-6, 1e-6  # Both 1 μC
    distances = np.linspace(0.01, 1, 100)
    forces = [coulombs_force(q1, q2, r) for r in distances]
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, forces, linewidth=2)
    plt.xlabel('Distance (m)', fontsize=12)
    plt.ylabel('Force (N)', fontsize=12)
    plt.title('Coulomb\'s Law: F ∝ 1/r²', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    return plt


if __name__ == "__main__":
    print("COULOMB'S LAW - Force Between Charges")
    print("=" * 60)
    
    # Example 1: Two positive charges (repulsive)
    q1, q2, r = 1e-6, 2e-6, 0.1
    F = coulombs_force(q1, q2, r)
    print(f"\nExample 1: Two positive charges")
    print(f"q₁ = {q1*1e6:.1f} μC, q₂ = {q2*1e6:.1f} μC, r = {r} m")
    print(f"Force: F = {F:.4f} N (repulsive)")
    
    # Example 2: Opposite charges (attractive)
    q1, q2, r = 1e-6, -2e-6, 0.1
    F = coulombs_force(q1, q2, r)
    print(f"\nExample 2: Opposite charges")
    print(f"q₁ = {q1*1e6:.1f} μC, q₂ = {q2*1e6:.1f} μC, r = {r} m")
    print(f"Force: F = {F:.4f} N (attractive)")
    
    # Visualization
    plt = visualize_force_vs_distance()
    plt.savefig('coulombs_law.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'coulombs_law.png'")
    plt.show()
