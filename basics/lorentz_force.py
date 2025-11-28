"""
Lorentz Force Theory and Linear Algebra
========================================

The Lorentz force is the force experienced by a charged particle moving through
electromagnetic fields. This script explores the theory and its mathematical
representation using linear algebra.

Theory:
-------
The Lorentz force F on a particle with charge q is given by:
    F = q(E + v × B)

where:
    E = electric field vector (N/C or V/m)
    B = magnetic field vector (Tesla or Wb/m²)
    v = velocity vector of the particle (m/s)
    × = cross product operator

Components:
-----------
1. Electric Force: F_E = qE (parallel to E field)
2. Magnetic Force: F_B = q(v × B) (perpendicular to both v and B)

Linear Algebra Perspective:
---------------------------
The cross product v × B can be represented using:
1. Determinant form with unit vectors
2. Skew-symmetric matrix (antisymmetric matrix)
3. Component-wise calculation

Cross Product as Skew-Symmetric Matrix:
----------------------------------------
For vectors v = [v_x, v_y, v_z] and B = [B_x, B_y, B_z],
the cross product v × B can be computed as:

    v × B = [v]_× · B

where [v]_× is the skew-symmetric matrix:

    [v]_× = | 0    -v_z   v_y |
            | v_z   0    -v_x |
            |-v_y   v_x   0   |

This gives: v × B = | v_y*B_z - v_z*B_y |
                    | v_z*B_x - v_x*B_z |
                    | v_x*B_y - v_y*B_x |
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform


class Arrow3D(FancyArrowPatch):
    """Custom 3D arrow for better visualization"""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        return np.min(zs)


def skew_symmetric_matrix(v):
    """
    Convert a 3D vector to its skew-symmetric matrix representation.
    
    This matrix represents the cross product operation:
    [v]_× · B = v × B
    
    Parameters:
    -----------
    v : array-like, shape (3,)
        Input vector [v_x, v_y, v_z]
    
    Returns:
    --------
    numpy.ndarray, shape (3, 3)
        Skew-symmetric matrix representation
    """
    v = np.array(v)
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])


def cross_product_manual(a, b):
    """
    Calculate cross product manually using component formula.
    
    a × b = (a_y*b_z - a_z*b_y)i + (a_z*b_x - a_x*b_z)j + (a_x*b_y - a_y*b_x)k
    
    Parameters:
    -----------
    a, b : array-like, shape (3,)
        Input vectors
    
    Returns:
    --------
    numpy.ndarray, shape (3,)
        Cross product result
    """
    a, b = np.array(a), np.array(b)
    return np.array([
        a[1]*b[2] - a[2]*b[1],  # i component
        a[2]*b[0] - a[0]*b[2],  # j component
        a[0]*b[1] - a[1]*b[0]   # k component
    ])


def cross_product_matrix(a, b):
    """
    Calculate cross product using skew-symmetric matrix multiplication.
    
    a × b = [a]_× · b
    
    Parameters:
    -----------
    a, b : array-like, shape (3,)
        Input vectors
    
    Returns:
    --------
    numpy.ndarray, shape (3,)
        Cross product result
    """
    return skew_symmetric_matrix(a) @ b


def lorentz_force(q, E, v, B, method='numpy'):
    """
    Calculate the Lorentz force on a charged particle.
    
    F = q(E + v × B)
    
    Parameters:
    -----------
    q : float
        Charge of particle (Coulombs)
    E : array-like, shape (3,)
        Electric field vector (V/m)
    v : array-like, shape (3,)
        Velocity vector (m/s)
    B : array-like, shape (3,)
        Magnetic field vector (Tesla)
    method : str
        Method for cross product: 'numpy', 'manual', or 'matrix'
    
    Returns:
    --------
    tuple
        (Total force, Electric force, Magnetic force) in Newtons
    """
    E = np.array(E)
    v = np.array(v)
    B = np.array(B)
    
    # Electric force component
    F_electric = q * E
    
    # Magnetic force component using specified method
    if method == 'numpy':
        v_cross_B = np.cross(v, B)
    elif method == 'manual':
        v_cross_B = cross_product_manual(v, B)
    elif method == 'matrix':
        v_cross_B = cross_product_matrix(v, B)
    else:
        raise ValueError("Method must be 'numpy', 'manual', or 'matrix'")
    
    F_magnetic = q * v_cross_B
    
    # Total Lorentz force
    F_total = F_electric + F_magnetic
    
    return F_total, F_electric, F_magnetic


def demonstrate_cross_product_properties():
    """Demonstrate key properties of the cross product."""
    print("=" * 70)
    print("CROSS PRODUCT PROPERTIES")
    print("=" * 70)
    
    # Example vectors
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    # Property 1: Anticommutativity (a × b = -(b × a))
    print("\n1. Anticommutativity: a × b = -(b × a)")
    print(f"   a = {a}")
    print(f"   b = {b}")
    cross_ab = np.cross(a, b)
    cross_ba = np.cross(b, a)
    print(f"   a × b = {cross_ab}")
    print(f"   b × a = {cross_ba}")
    print(f"   -(b × a) = {-cross_ba}")
    print(f"   Equal? {np.allclose(cross_ab, -cross_ba)}")
    
    # Property 2: Perpendicularity (result is perpendicular to both inputs)
    print("\n2. Perpendicularity: (a × b) ⊥ a and (a × b) ⊥ b")
    print(f"   (a × b) · a = {np.dot(cross_ab, a):.10f} (should be ≈ 0)")
    print(f"   (a × b) · b = {np.dot(cross_ab, b):.10f} (should be ≈ 0)")
    
    # Property 3: Magnitude relation
    print("\n3. Magnitude: |a × b| = |a||b|sin(θ)")
    magnitude_cross = np.linalg.norm(cross_ab)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    cos_theta = np.dot(a, b) / (magnitude_a * magnitude_b)
    sin_theta = np.sqrt(1 - cos_theta**2)
    print(f"   |a × b| = {magnitude_cross:.6f}")
    print(f"   |a||b|sin(θ) = {magnitude_a * magnitude_b * sin_theta:.6f}")
    
    # Property 4: Distributivity
    c = np.array([7, 8, 9])
    print("\n4. Distributivity: a × (b + c) = (a × b) + (a × c)")
    print(f"   c = {c}")
    left_side = np.cross(a, b + c)
    right_side = np.cross(a, b) + np.cross(a, c)
    print(f"   a × (b + c) = {left_side}")
    print(f"   (a × b) + (a × c) = {right_side}")
    print(f"   Equal? {np.allclose(left_side, right_side)}")


def demonstrate_matrix_cross_product():
    """Demonstrate cross product using skew-symmetric matrix."""
    print("\n" + "=" * 70)
    print("CROSS PRODUCT AS MATRIX MULTIPLICATION")
    print("=" * 70)
    
    v = np.array([1, 2, 3])
    B = np.array([4, 5, 6])
    
    print(f"\nVectors:")
    print(f"v = {v}")
    print(f"B = {B}")
    
    # Show skew-symmetric matrix
    v_skew = skew_symmetric_matrix(v)
    print(f"\nSkew-symmetric matrix [v]_×:")
    print(v_skew)
    
    # Compare methods
    print("\nCross product v × B calculated three ways:")
    result_numpy = np.cross(v, B)
    result_manual = cross_product_manual(v, B)
    result_matrix = cross_product_matrix(v, B)
    
    print(f"1. NumPy (np.cross):        {result_numpy}")
    print(f"2. Manual formula:          {result_manual}")
    print(f"3. Matrix multiplication:   {result_matrix}")
    print(f"\nAll methods equal? {np.allclose(result_numpy, result_manual) and np.allclose(result_numpy, result_matrix)}")


def example_lorentz_force_calculation():
    """Calculate Lorentz force for various scenarios."""
    print("\n" + "=" * 70)
    print("LORENTZ FORCE EXAMPLES")
    print("=" * 70)
    
    # Example 1: Electron in uniform fields
    print("\n1. Electron in uniform E and B fields")
    print("-" * 70)
    q = -1.602e-19  # Coulombs (electron charge)
    E = np.array([1000, 0, 0])  # V/m
    v = np.array([1e6, 0, 0])  # m/s
    B = np.array([0, 0, 0.1])  # Tesla
    
    print(f"Charge: q = {q:.3e} C (electron)")
    print(f"Electric field: E = {E} V/m")
    print(f"Velocity: v = {v} m/s")
    print(f"Magnetic field: B = {B} T")
    
    F_total, F_E, F_B = lorentz_force(q, E, v, B)
    
    print(f"\nForce components:")
    print(f"Electric force:  F_E = {F_E}")
    print(f"Magnetic force:  F_B = {F_B}")
    print(f"Total force:     F   = {F_total}")
    print(f"\nMagnitudes:")
    print(f"|F_E| = {np.linalg.norm(F_E):.3e} N")
    print(f"|F_B| = {np.linalg.norm(F_B):.3e} N")
    print(f"|F|   = {np.linalg.norm(F_total):.3e} N")
    
    # Example 2: Proton with velocity perpendicular to B
    print("\n2. Proton moving perpendicular to magnetic field")
    print("-" * 70)
    q = 1.602e-19  # Coulombs (proton charge)
    E = np.array([0, 0, 0])  # No electric field
    v = np.array([1e5, 0, 0])  # m/s (perpendicular to B)
    B = np.array([0, 0, 0.5])  # Tesla
    
    print(f"Charge: q = {q:.3e} C (proton)")
    print(f"Electric field: E = {E} V/m")
    print(f"Velocity: v = {v} m/s")
    print(f"Magnetic field: B = {B} T")
    
    F_total, F_E, F_B = lorentz_force(q, E, v, B)
    
    print(f"\nMagnetic force: F_B = {F_B}")
    print(f"|F_B| = {np.linalg.norm(F_B):.3e} N")
    
    # Calculate radius of circular motion
    m_proton = 1.673e-27  # kg
    v_mag = np.linalg.norm(v)
    B_mag = np.linalg.norm(B)
    r = (m_proton * v_mag) / (q * B_mag)
    print(f"\nCircular motion radius: r = mv/(qB) = {r:.6f} m")
    
    # Example 3: Particle with velocity parallel to B
    print("\n3. Particle moving parallel to magnetic field")
    print("-" * 70)
    q = 1.602e-19  # Coulombs
    E = np.array([0, 0, 0])
    v = np.array([0, 0, 1e6])  # m/s (parallel to B)
    B = np.array([0, 0, 0.5])  # Tesla
    
    print(f"Charge: q = {q:.3e} C")
    print(f"Velocity: v = {v} m/s")
    print(f"Magnetic field: B = {B} T")
    print("Note: v is parallel to B")
    
    F_total, F_E, F_B = lorentz_force(q, E, v, B)
    
    print(f"\nMagnetic force: F_B = {F_B}")
    print(f"Since v || B, the cross product v × B = 0")
    print(f"Therefore, no magnetic force acts on the particle")


def visualize_lorentz_force():
    """Visualize the Lorentz force components in 3D."""
    print("\n" + "=" * 70)
    print("GENERATING 3D VISUALIZATION")
    print("=" * 70)
    
    # Setup with stronger fields for better visualization
    q = 1.602e-19  # Positive charge
    E = np.array([1000, 0, 0])  # V/m
    v = np.array([0, 1e6, 0])  # m/s
    B = np.array([0, 0, 0.5])  # Tesla
    
    F_total, F_E, F_B = lorentz_force(q, E, v, B)
    
    # Scale all vectors to similar magnitudes for visualization
    # Use logarithmic scaling for forces to better visualize
    scale = 1.5
    E_viz = E / np.linalg.norm(E) * scale
    v_viz = v / np.linalg.norm(v) * scale
    B_viz = B / np.linalg.norm(B) * scale
    
    # Scale forces uniformly
    F_E_viz = F_E / np.linalg.norm(F_E) * scale if np.linalg.norm(F_E) > 0 else np.array([0, 0, 0])
    F_B_viz = F_B / np.linalg.norm(F_B) * scale if np.linalg.norm(F_B) > 0 else np.array([0, 0, 0])
    F_viz = F_total / np.linalg.norm(F_total) * scale if np.linalg.norm(F_total) > 0 else np.array([0, 0, 0])
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    origin = np.array([0, 0, 0])
    
    # Plot field vectors
    ax.quiver(0, 0, 0, E_viz[0], E_viz[1], E_viz[2], 
              color='blue', arrow_length_ratio=0.15, linewidth=3, label='E field (V/m)', alpha=0.9)
    ax.quiver(0, 0, 0, v_viz[0], v_viz[1], v_viz[2], 
              color='green', arrow_length_ratio=0.15, linewidth=3, label='velocity v (m/s)', alpha=0.9)
    ax.quiver(0, 0, 0, B_viz[0], B_viz[1], B_viz[2], 
              color='purple', arrow_length_ratio=0.15, linewidth=3, label='B field (T)', alpha=0.9)
    
    # Plot force vectors
    ax.quiver(0, 0, 0, F_E_viz[0], F_E_viz[1], F_E_viz[2], 
              color='cyan', arrow_length_ratio=0.15, linewidth=3.5, label='F_E = qE', alpha=0.9)
    ax.quiver(0, 0, 0, F_B_viz[0], F_B_viz[1], F_B_viz[2], 
              color='orange', arrow_length_ratio=0.15, linewidth=3.5, label='F_B = q(v×B)', alpha=0.9)
    ax.quiver(0, 0, 0, F_viz[0], F_viz[1], F_viz[2], 
              color='red', arrow_length_ratio=0.15, linewidth=4, label='F_total', alpha=1.0)
    
    # Plot particle
    ax.scatter([0], [0], [0], color='black', s=150, label='Particle (q>0)', zorder=10)
    
    # Labels and formatting
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title('Lorentz Force Components\nF = q(E + v × B)\n(vectors normalized for visualization)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set equal aspect ratio
    max_range = scale * 1.3
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Add actual values as text
    info_text = f"Actual values:\n"
    info_text += f"E = {E} V/m\n"
    info_text += f"v = {v} m/s\n"
    info_text += f"B = {B} T\n"
    info_text += f"q = {q:.3e} C\n"
    info_text += f"|F_E| = {np.linalg.norm(F_E):.3e} N\n"
    info_text += f"|F_B| = {np.linalg.norm(F_B):.3e} N\n"
    info_text += f"|F| = {np.linalg.norm(F_total):.3e} N"
    
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
              fontsize=9, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lorentz_force_components.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to: lorentz_force_components.png")
    
    return fig


def visualize_charged_particle_trajectory():
    """Simulate and visualize the trajectory of a charged particle in E and B fields."""
    print("\n" + "=" * 70)
    print("SIMULATING PARTICLE TRAJECTORY")
    print("=" * 70)
    
    # Particle properties
    q = 1.602e-19  # Coulombs (proton)
    m = 1.673e-27  # kg (proton mass)
    
    # Initial conditions
    r0 = np.array([0.0, 0.0, 0.0])  # Initial position (m)
    v0 = np.array([1e5, 1e5, 0.0])  # Initial velocity (m/s)
    
    # Fields
    E = np.array([0.0, 0.0, 0.0])  # Electric field (V/m)
    B = np.array([0.0, 0.0, 0.1])  # Magnetic field (Tesla)
    
    print(f"Particle: proton (q = {q:.3e} C, m = {m:.3e} kg)")
    print(f"Initial position: r0 = {r0} m")
    print(f"Initial velocity: v0 = {v0} m/s")
    print(f"Electric field: E = {E} V/m")
    print(f"Magnetic field: B = {B} T")
    
    # Time parameters
    dt = 1e-9  # Time step (seconds)
    t_max = 1e-6  # Total time (seconds)
    steps = int(t_max / dt)
    
    # Arrays to store trajectory
    positions = np.zeros((steps, 3))
    velocities = np.zeros((steps, 3))
    
    positions[0] = r0
    velocities[0] = v0
    
    # Velocity Verlet integration
    for i in range(steps - 1):
        v = velocities[i]
        r = positions[i]
        
        # Calculate Lorentz force
        F, _, _ = lorentz_force(q, E, v, B)
        a = F / m  # Acceleration
        
        # Update position and velocity
        positions[i + 1] = r + v * dt + 0.5 * a * dt**2
        
        # Calculate force at new position (for better accuracy)
        v_half = v + 0.5 * a * dt
        F_new, _, _ = lorentz_force(q, E, v_half, B)
        a_new = F_new / m
        
        velocities[i + 1] = v + 0.5 * (a + a_new) * dt
    
    # Plot trajectory
    fig = plt.figure(figsize=(14, 6))
    
    # 3D trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1.5)
    ax1.scatter([r0[0]], [r0[1]], [r0[2]], color='green', s=100, label='Start', zorder=5)
    ax1.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 
                color='red', s=100, label='End', zorder=5)
    
    # Plot B field direction
    ax1.quiver(0, 0, 0, 0, 0, 0.001, color='purple', arrow_length_ratio=0.3, 
               linewidth=2, label='B field')
    
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_zlabel('Z (m)', fontsize=11)
    ax1.set_title('3D Trajectory of Charged Particle\nin Magnetic Field', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2D projection (X-Y plane)
    ax2 = fig.add_subplot(122)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5)
    ax2.scatter([r0[0]], [r0[1]], color='green', s=100, label='Start', zorder=5)
    ax2.scatter([positions[-1, 0]], [positions[-1, 1]], 
                color='red', s=100, label='End', zorder=5)
    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_title('X-Y Projection\n(Circular motion due to Lorentz force)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('particle_trajectory.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to: particle_trajectory.png")
    
    # Calculate theoretical radius
    v_perp = np.sqrt(v0[0]**2 + v0[1]**2)  # Velocity perpendicular to B
    B_mag = np.linalg.norm(B)
    r_theoretical = (m * v_perp) / (abs(q) * B_mag)
    
    print(f"\nTheoretical radius of circular motion: r = {r_theoretical:.6f} m")
    print(f"Observed approximate radius: r ≈ {np.max(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)):.6f} m")
    
    return fig


def main():
    """Main function to demonstrate Lorentz force theory."""
    print("\n" + "=" * 70)
    print("LORENTZ FORCE AND LINEAR ALGEBRA")
    print("=" * 70)
    print("\nThis script demonstrates the Lorentz force F = q(E + v × B)")
    print("and its relationship with linear algebra operations.")
    
    # Demonstrate cross product properties
    demonstrate_cross_product_properties()
    
    # Demonstrate matrix representation
    demonstrate_matrix_cross_product()
    
    # Calculate example forces
    example_lorentz_force_calculation()
    
    # Visualizations
    try:
        print("\nGenerating visualizations...")
        fig1 = visualize_lorentz_force()
        fig2 = visualize_charged_particle_trajectory()
        
        print("\n" + "=" * 70)
        print("Visualizations saved and ready to display!")
        print("Plot files saved as backup:")
        print("  - lorentz_force_components.png")
        print("  - particle_trajectory.png")
        print("\nDisplaying interactive plots...")
        print("Close the plot windows to exit.")
        print("=" * 70)
        plt.show()
        
    except Exception as e:
        print(f"\nVisualization error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlots saved to PNG files. View them with an image viewer.")


if __name__ == "__main__":
    main()
