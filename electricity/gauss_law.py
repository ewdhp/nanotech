"""
Gauss's Law - Flux and Charge Relationships

One of Maxwell's equations relating electric flux to enclosed charge.

Integral Form: ∮ E·dA = Q_enclosed / ε₀

Differential Form: ∇·E = ρ / ε₀

where:
- E: Electric field (V/m)
- dA: Surface area element (m²)
- Q_enclosed: Total charge inside surface (C)
- ε₀ = 8.854 × 10⁻¹² F/m (permittivity of free space)
- ρ: Charge density (C/m³)
"""

import numpy as np
import matplotlib.pyplot as plt

EPSILON_0 = 8.854187817e-12  # Permittivity of free space (F/m)
K_E = 8.987551787e9          # Coulomb's constant (N⋅m²/C²)


def electric_flux_to_charge(flux):
    """
    Calculate enclosed charge from electric flux.
    
    Q = ε₀ * Φ
    where Φ = ∮ E·dA
    """
    return EPSILON_0 * flux


def charge_to_electric_flux(Q):
    """
    Calculate electric flux from enclosed charge.
    
    Φ = Q / ε₀
    """
    return Q / EPSILON_0


def flux_through_sphere(q, radius):
    """
    Calculate flux through spherical surface around point charge.
    
    For a sphere: Φ = E * 4πr² = (k_e * q / r²) * 4πr² = q / ε₀
    """
    # Electric field at radius r
    E = K_E * q / (radius ** 2)
    # Surface area of sphere
    A = 4 * np.pi * radius ** 2
    # Flux through sphere
    flux = E * A
    return flux


def field_from_charge_density(rho, is_sphere=True, radius=None):
    """
    Calculate electric field from charge distribution using Gauss's law.
    
    For uniform sphere: E = (ρ * r) / (3 * ε₀) inside
                        E = (ρ * R³) / (3 * ε₀ * r²) outside
    """
    if is_sphere and radius:
        # Inside sphere
        E_inside = lambda r: (rho * r) / (3 * EPSILON_0)
        # Outside sphere
        total_charge = rho * (4/3) * np.pi * radius**3
        E_outside = lambda r: K_E * total_charge / (r ** 2)
        return E_inside, E_outside
    return None


def visualize_gauss_law():
    """Visualize Gauss's law for spherical symmetry."""
    Q = 1e-9  # 1 nC
    radii = np.linspace(0.01, 2, 100)
    
    # Calculate flux for different radii (should be constant)
    fluxes = [flux_through_sphere(Q, r) for r in radii]
    expected_flux = Q / EPSILON_0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Flux vs radius
    ax1.plot(radii, fluxes, linewidth=2, label='Calculated flux')
    ax1.axhline(y=expected_flux, color='r', linestyle='--', 
                linewidth=2, label=f'Expected: Q/ε₀')
    ax1.set_xlabel('Radius (m)', fontsize=12)
    ax1.set_ylabel('Electric Flux (V⋅m)', fontsize=12)
    ax1.set_title('Gauss\'s Law: Flux is Independent of Radius', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Electric field vs radius
    E_field = [K_E * Q / (r**2) for r in radii]
    ax2.plot(radii, E_field, linewidth=2, color='green')
    ax2.set_xlabel('Radius (m)', fontsize=12)
    ax2.set_ylabel('Electric Field (N/C)', fontsize=12)
    ax2.set_title('Electric Field: E ∝ 1/r²', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    print("GAUSS'S LAW - Flux and Charge Relationships")
    print("=" * 60)
    
    # Example 1: Flux from charge
    Q = 1e-9  # 1 nC
    flux = charge_to_electric_flux(Q)
    
    print(f"\nExample 1: Flux from point charge")
    print(f"Enclosed charge: Q = {Q*1e9:.1f} nC")
    print(f"Electric flux: Φ = {flux:.4e} V⋅m")
    print(f"Formula: Φ = Q/ε₀")
    
    # Example 2: Charge from flux
    flux_measured = 1e2  # V⋅m
    Q_calc = electric_flux_to_charge(flux_measured)
    
    print(f"\nExample 2: Charge from measured flux")
    print(f"Measured flux: Φ = {flux_measured:.2e} V⋅m")
    print(f"Enclosed charge: Q = {Q_calc*1e9:.4f} nC")
    
    # Example 3: Verify for spherical surface
    Q = 1e-9  # 1 nC
    radii = [0.1, 0.5, 1.0]
    
    print(f"\nExample 3: Flux through spheres of different radii")
    print(f"Charge: Q = {Q*1e9:.1f} nC")
    for r in radii:
        flux_sphere = flux_through_sphere(Q, r)
        print(f"  Radius {r} m: Φ = {flux_sphere:.4e} V⋅m")
    print(f"  → All equal to Q/ε₀ = {Q/EPSILON_0:.4e} V⋅m")
    
    plt = visualize_gauss_law()
    plt.savefig('gauss_law.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'gauss_law.png'")
    plt.show()
