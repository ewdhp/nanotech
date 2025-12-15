"""
Electric Field - Vector Field Representation

The electric field describes the force per unit charge at any point in space.

Key Formula: E = F/q₀ = k_e * q / r²

where:
- E: Electric field in N/C or V/m
- q: Source charge in Coulombs (C)
- r: Distance from source in meters (m)

Vector Form: E(r) = k_e * q * (r - r₀) / |r - r₀|³
"""

import numpy as np
import matplotlib.pyplot as plt

K_E = 8.987551787e9  # Coulomb's constant (N⋅m²/C²)


def electric_field_magnitude(q, r):
    """
    Calculate electric field magnitude from point charge.
    
    E = k_e * q / r²
    """
    if r == 0:
        raise ValueError("Distance cannot be zero")
    return K_E * abs(q) / (r ** 2)


def electric_field_vector(q, position, source_position):
    """
    Calculate electric field vector at position due to charge q.
    
    E(r) = k_e * q * (r - r₀) / |r - r₀|³
    
    Returns: [Ex, Ey, Ez] in V/m
    """
    r = np.array(position)
    r0 = np.array(source_position)
    displacement = r - r0
    distance = np.linalg.norm(displacement)
    
    if distance == 0:
        raise ValueError("Position cannot coincide with charge")
    
    E_vector = K_E * q * displacement / (distance ** 3)
    return E_vector


def visualize_field_lines():
    """Visualize electric field lines around a single charge."""
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    q = 1e-9  # 1 nC
    source = [0, 0, 0]
    
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    
    for i in range(len(x)):
        for j in range(len(y)):
            point = [X[j, i], Y[j, i], 0]
            if np.linalg.norm(np.array(point) - np.array(source)) < 0.2:
                continue
            E = electric_field_vector(q, point, source)
            Ex[j, i] = E[0]
            Ey[j, i] = E[1]
    
    plt.figure(figsize=(10, 8))
    magnitude = np.sqrt(Ex**2 + Ey**2)
    magnitude[magnitude == 0] = 1
    
    plt.streamplot(X, Y, Ex/magnitude, Ey/magnitude, 
                   color=np.log10(magnitude), cmap='viridis', 
                   linewidth=1.5, density=2)
    plt.plot(0, 0, 'ro', markersize=15, label='Positive charge')
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('y (m)', fontsize=12)
    plt.title('Electric Field Lines', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.colorbar(label='log₁₀(|E|)')
    return plt


if __name__ == "__main__":
    print("ELECTRIC FIELD - Vector Field Representation")
    print("=" * 60)
    
    # Example: Field from point charge
    q = 1e-6  # 1 μC
    r = 0.1   # 0.1 m
    E = electric_field_magnitude(q, r)
    
    print(f"\nPoint charge: q = {q*1e6:.1f} μC")
    print(f"Distance: r = {r} m")
    print(f"Electric field: E = {E:.2f} N/C")
    print(f"Direction: Radially {'outward' if q > 0 else 'inward'}")
    
    # Vector example
    E_vec = electric_field_vector(q, [0.1, 0, 0], [0, 0, 0])
    print(f"\nVector field at (0.1, 0, 0):")
    print(f"E = [{E_vec[0]:.2e}, {E_vec[1]:.2e}, {E_vec[2]:.2e}] V/m")
    
    plt = visualize_field_lines()
    plt.savefig('electric_field.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'electric_field.png'")
    plt.show()
