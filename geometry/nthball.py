import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import math

def nball_volume(n, R=1.0):
    """
    Compute the volume of an n-dimensional ball of radius R.
    
    Parameters:
    n (int): Dimension (n ≥ 0)
    R (float): Radius (default: 1.0)
    
    Returns:
    float: Volume V_n(R)
    """
    if n < 0:
        raise ValueError("Dimension n must be non-negative")
    
    # Formula: V_n(R) = (π^(n/2) / Γ(n/2 + 1)) * R^n
    numerator = math.pi ** (n / 2)
    denominator = gamma(n / 2 + 1)
    return (numerator / denominator) * (R ** n)

def nball_surface_area(n, R=1.0):
    """
    Compute the surface area of an n-dimensional sphere of radius R.
    
    Formula: S_n(R) = (2π^(n/2) / Γ(n/2)) * R^(n-1)
    """
    if n <= 0:
        raise ValueError("Dimension n must be positive")
    
    numerator = 2 * math.pi ** (n / 2)
    denominator = gamma(n / 2)
    return (numerator / denominator) * (R ** (n - 1))

def test_known_cases():
    """Test against known volumes for dimensions 0 through 5."""
    # Known volumes for unit ball (R=1)
    known_volumes = {
        0: 1.0,           # 0D: point
        1: 2.0,           # 1D: interval [-1, 1] length 2
        2: math.pi,       # 2D: disk area π
        3: 4/3 * math.pi, # 3D: sphere volume
        4: 0.5 * math.pi**2,  # 4D: π²/2
        5: 8/15 * math.pi**2  # 5D: (8π²)/15
    }
    
    print("Testing known volumes (R=1):")
    print("-" * 40)
    for n, expected in known_volumes.items():
        computed = nball_volume(n, R=1.0)
        error = abs(computed - expected)
        print(f"V_{n} = {computed:.8f} (expected {expected:.8f}) | error: {error:.2e}")
    print()

def volume_vs_dimension(max_n=20, R=1.0):
    """
    Plot how volume changes with dimension for unit ball.
    
    Shows the surprising fact that volume peaks around n=5 and then decays to 0.
    """
    dimensions = np.arange(0, max_n + 1)
    volumes = [nball_volume(n, R) for n in dimensions]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, volumes, 'b-o', linewidth=2, markersize=6)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Mark maximum
    max_idx = np.argmax(volumes)
    plt.plot(dimensions[max_idx], volumes[max_idx], 'ro', markersize=10, 
             label=f'Max at n={dimensions[max_idx]}: {volumes[max_idx]:.4f}')
    
    plt.xlabel('Dimension n', fontsize=12)
    plt.ylabel(f'Volume V_n({R})', fontsize=12)
    plt.title(f'Volume of {R}-radius Ball vs Dimension', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return dimensions, volumes

def radius_effect(dimension=3, max_radius=3):
    """Show how volume scales with radius for a fixed dimension."""
    radii = np.linspace(0, max_radius, 100)
    volumes = nball_volume(dimension, radii)
    
    plt.figure(figsize=(10, 6))
    plt.plot(radii, volumes, 'r-', linewidth=3)
    plt.xlabel(f'Radius R', fontsize=12)
    plt.ylabel(f'Volume V_{dimension}(R)', fontsize=12)
    plt.title(f'Volume vs Radius in {dimension} Dimensions', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add power law reference
    power_law = radii ** dimension
    plt.plot(radii, power_law / power_law[-1] * volumes[-1], 
             'b--', alpha=0.7, label=f'R^{dimension} scaling')
    plt.legend()
    plt.show()

def surface_to_volume_ratio(max_n=30, R=1.0):
    """
    Compute S_n(R) / V_n(R) = n/R.
    
    This shows an interesting relationship: for fixed R,
    the surface area to volume ratio grows linearly with dimension.
    """
    dimensions = np.arange(1, max_n + 1)
    ratios = []
    
    for n in dimensions:
        V = nball_volume(n, R)
        S = nball_surface_area(n, R)
        ratios.append(S / V)
    
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, ratios, 'g-s', linewidth=2, markersize=6, label='S/V')
    plt.plot(dimensions, dimensions/R, 'r--', linewidth=1, label='n/R (theoretical)')
    plt.xlabel('Dimension n', fontsize=12)
    plt.ylabel('Surface Area / Volume', fontsize=12)
    plt.title(f'Surface-to-Volume Ratio for R={R} Ball', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def monte_carlo_estimate(n, R=1.0, num_samples=100000):
    """
    Estimate volume using Monte Carlo for verification.
    Works for moderate dimensions (n <= ~10).
    """
    # Generate random points in [-R, R]^n
    points = np.random.uniform(-R, R, size=(num_samples, n))
    # Compute squared distances from origin
    distances_sq = np.sum(points**2, axis=1)
    # Count points inside the ball
    inside = np.sum(distances_sq <= R**2)
    # Volume of bounding cube
    cube_volume = (2*R)**n
    # Estimated volume
    estimated = (inside / num_samples) * cube_volume
    exact = nball_volume(n, R)
    
    return exact, estimated

# ================= MAIN EXAMPLE =================
if __name__ == "__main__":
    print("=" * 60)
    print("N-DIMENSIONAL BALL VOLUME CALCULATIONS")
    print("=" * 60)
    
    # 1. Test known cases
    test_known_cases()
    
    # 2. Show volume peaks then decays
    print("Volume for dimensions 0 through 10 (R=1):")
    print("-" * 40)
    for n in range(11):
        V = nball_volume(n, R=1.0)
        print(f"V_{n} = {V:.8f}")
    
    # 3. Visualizations
    print("\nGenerating visualizations...")
    
    # Volume vs dimension
    dims, vols = volume_vs_dimension(max_n=20, R=1.0)
    
    # Radius effect in 3D (for intuition)
    radius_effect(dimension=3, max_radius=3)
    
    # Surface-to-volume ratio
    surface_to_volume_ratio(max_n=15, R=1.0)
    
    # 4. Monte Carlo verification for n=3
    print("\nMonte Carlo verification for n=3, R=1:")
    exact, estimated = monte_carlo_estimate(3, R=1.0, num_samples=50000)
    print(f"Exact volume: {exact:.6f}")
    print(f"Monte Carlo estimate: {estimated:.6f}")
    print(f"Relative error: {abs(exact - estimated)/exact:.2%}")
    
    # 5. Higher dimension example
    print("\nHigh-dimensional example:")
    print(f"Volume of 10D unit ball: {nball_volume(10):.8f}")
    print(f"Volume of 20D unit ball: {nball_volume(20):.8e}")
    
    print("\n" + "=" * 60)
    print("Note: Volume → 0 as n → ∞ for fixed radius R=1.")
    print("=" * 60)