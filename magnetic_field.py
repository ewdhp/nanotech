"""
Magnetic Field Theory and Applications
=======================================

This script covers the most important concepts of magnetic fields including:
- Magnetic field definition and properties
- Biot-Savart law for current elements
- Ampere's law and applications
- Magnetic force on moving charges
- Magnetic force on current-carrying wires
- Magnetic dipoles and moments
- Solenoids and toroids
- Magnetic flux and Faraday's law

Theory:
-------
Magnetic Field B from a current element (Biot-Savart Law):
    dB = (μ₀/4π) × (I dl × r̂) / r²

Where:
    μ₀ = 4π×10⁻⁷ T·m/A (permeability of free space)
    I = current (A)
    dl = current element vector (m)
    r = distance from element (m)
    r̂ = unit vector from element to field point

Key Concepts:
------------
1. Magnetic Field Strength: B (Tesla or Wb/m²)
2. Magnetic Force: F = q(v × B) or F = I(L × B)
3. Magnetic Flux: Φ_B = ∫B·dA (Weber)
4. Ampere's Law: ∮B·dl = μ₀I_enclosed
5. Magnetic Dipole Moment: μ = IA (A·m²)
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Physical constants
MU_0 = 4 * np.pi * 1e-7  # T·m/A - Permeability of free space
K_M = MU_0 / (4 * np.pi)  # Magnetic constant


def biot_savart_straight_wire(I, wire_start, wire_end, r_field):
    """
    Calculate magnetic field from a straight current-carrying wire segment.
    
    Using Biot-Savart law integrated along the wire.
    
    Parameters:
    -----------
    I : float
        Current (Amperes)
    wire_start : array-like, shape (3,)
        Starting point of wire (m)
    wire_end : array-like, shape (3,)
        Ending point of wire (m)
    r_field : array-like, shape (N, 3) or (3,)
        Position(s) where field is calculated (m)
    
    Returns:
    --------
    B : ndarray
        Magnetic field vector(s) (Tesla)
    """
    wire_start = np.array(wire_start)
    wire_end = np.array(wire_end)
    r_field = np.array(r_field)
    
    if r_field.ndim == 1:
        r_field = r_field.reshape(1, -1)
        single_point = True
    else:
        single_point = False
    
    # Wire vector
    L = wire_end - wire_start
    L_mag = np.linalg.norm(L)
    L_hat = L / L_mag
    
    B = np.zeros_like(r_field)
    
    # Numerical integration using segments
    n_segments = 100
    for i in range(n_segments):
        # Position along wire
        t = i / n_segments
        r_wire = wire_start + t * L
        dl = L / n_segments
        
        # Vector from wire element to field point
        r_vec = r_field - r_wire
        r_mag = np.linalg.norm(r_vec, axis=1, keepdims=True)
        
        # Avoid singularity
        r_mag[r_mag < 1e-10] = 1e-10
        
        # Biot-Savart law: dB = (μ₀/4π) × I(dl × r̂)/r²
        dl_cross_r = np.cross(dl, r_vec)
        dB = K_M * I * dl_cross_r / (r_mag ** 3)
        B += dB
    
    if single_point:
        return B[0]
    return B


def magnetic_field_infinite_wire(I, r):
    """
    Magnetic field from an infinite straight wire (Ampere's law).
    
    B = (μ₀I)/(2πr) in tangential direction
    
    Parameters:
    -----------
    I : float
        Current (Amperes)
    r : float
        Perpendicular distance from wire (m)
    
    Returns:
    --------
    B : float
        Magnetic field magnitude (Tesla)
    """
    return (MU_0 * I) / (2 * np.pi * r)


def magnetic_field_circular_loop(I, R, r_field):
    """
    Magnetic field at the center of a circular current loop.
    
    At center: B = (μ₀I)/(2R) along axis
    
    Parameters:
    -----------
    I : float
        Current (Amperes)
    R : float
        Radius of loop (m)
    r_field : array-like, shape (3,)
        Position where field is calculated (m)
    
    Returns:
    --------
    B : ndarray
        Magnetic field vector (Tesla)
    """
    # Simplified: field at center
    r_field = np.array(r_field)
    
    # For a loop in xy-plane centered at origin
    # Field at center points in z-direction
    B_mag = (MU_0 * I) / (2 * R)
    
    # Distance from center
    r_mag = np.linalg.norm(r_field[:2])
    
    if r_mag < 0.01 * R:  # Near center
        B = np.array([0, 0, B_mag])
    else:
        # Off-axis calculation (simplified)
        z = r_field[2]
        B_z = (MU_0 * I * R**2) / (2 * (R**2 + z**2)**(3/2))
        B = np.array([0, 0, B_z])
    
    return B


def magnetic_field_solenoid(n, I, r_field, length, radius):
    """
    Magnetic field inside an ideal solenoid.
    
    Inside: B = μ₀nI (uniform, along axis)
    Outside: B ≈ 0
    
    Parameters:
    -----------
    n : float
        Number of turns per unit length (turns/m)
    I : float
        Current (Amperes)
    r_field : array-like, shape (3,)
        Position where field is calculated (m)
    length : float
        Length of solenoid (m)
    radius : float
        Radius of solenoid (m)
    
    Returns:
    --------
    B : ndarray
        Magnetic field vector (Tesla)
    """
    r_field = np.array(r_field)
    
    # Check if inside solenoid
    r_perp = np.sqrt(r_field[0]**2 + r_field[1]**2)
    z = r_field[2]
    
    if r_perp < radius and abs(z) < length/2:
        # Inside: uniform field along z-axis
        B_mag = MU_0 * n * I
        B = np.array([0, 0, B_mag])
    else:
        # Outside: approximately zero for ideal solenoid
        B = np.array([0, 0, 0])
    
    return B


def magnetic_force_on_charge(q, v, B):
    """
    Calculate magnetic force on a moving charge.
    
    F = q(v × B)
    
    Parameters:
    -----------
    q : float
        Charge (Coulombs)
    v : array-like, shape (3,)
        Velocity vector (m/s)
    B : array-like, shape (3,)
        Magnetic field vector (Tesla)
    
    Returns:
    --------
    F : ndarray
        Force vector (Newtons)
    """
    v = np.array(v)
    B = np.array(B)
    return q * np.cross(v, B)


def magnetic_force_on_wire(I, L, B):
    """
    Calculate magnetic force on a current-carrying wire.
    
    F = I(L × B)
    
    Parameters:
    -----------
    I : float
        Current (Amperes)
    L : array-like, shape (3,)
        Wire length vector (m)
    B : array-like, shape (3,)
        Magnetic field vector (Tesla)
    
    Returns:
    --------
    F : ndarray
        Force vector (Newtons)
    """
    L = np.array(L)
    B = np.array(B)
    return I * np.cross(L, B)


def demonstrate_biot_savart():
    """Demonstrate Biot-Savart law calculations."""
    print("=" * 70)
    print("BIOT-SAVART LAW")
    print("=" * 70)
    
    # Example 1: Straight wire segment
    print("\n1. Magnetic field from a straight wire segment")
    print("-" * 70)
    I = 10  # 10 A
    wire_start = np.array([-0.5, 0, 0])
    wire_end = np.array([0.5, 0, 0])
    r_field = np.array([0, 0.1, 0])  # 10 cm from center
    
    B = biot_savart_straight_wire(I, wire_start, wire_end, r_field)
    
    print(f"Current: I = {I} A")
    print(f"Wire: from {wire_start} to {wire_end} m")
    print(f"Field point: r = {r_field} m")
    print(f"Magnetic field: B = {B} T")
    print(f"Magnitude: |B| = {np.linalg.norm(B)*1e6:.3f} μT")
    
    # Example 2: Infinite wire (Ampere's law)
    print("\n2. Infinite straight wire (Ampere's law)")
    print("-" * 70)
    I = 10  # 10 A
    r = 0.05  # 5 cm
    
    B_inf = magnetic_field_infinite_wire(I, r)
    
    print(f"Current: I = {I} A")
    print(f"Distance: r = {r*100:.1f} cm")
    print(f"Magnetic field: B = μ₀I/(2πr) = {B_inf*1e6:.3f} μT")
    
    # Example 3: Circular loop
    print("\n3. Circular current loop")
    print("-" * 70)
    I = 5  # 5 A
    R = 0.1  # 10 cm radius
    r_center = np.array([0, 0, 0])
    
    B_loop = magnetic_field_circular_loop(I, R, r_center)
    
    print(f"Current: I = {I} A")
    print(f"Loop radius: R = {R*100:.1f} cm")
    print(f"Field at center: B = μ₀I/(2R) = {B_loop} T")
    print(f"Magnitude: |B| = {np.linalg.norm(B_loop)*1e6:.3f} μT")


def demonstrate_ampere_law():
    """Demonstrate Ampere's law applications."""
    print("\n" + "=" * 70)
    print("AMPERE'S LAW")
    print("=" * 70)
    print("\n∮ B·dl = μ₀I_enclosed")
    
    # Example 1: Solenoid
    print("\n1. Ideal solenoid")
    print("-" * 70)
    N = 1000  # turns
    length = 0.5  # 50 cm
    n = N / length  # turns per meter
    I = 2  # 2 A
    radius = 0.02  # 2 cm
    
    r_inside = np.array([0, 0, 0])
    B_solenoid = magnetic_field_solenoid(n, I, r_inside, length, radius)
    
    print(f"Number of turns: N = {N}")
    print(f"Length: L = {length*100:.1f} cm")
    print(f"Turn density: n = N/L = {n:.1f} turns/m")
    print(f"Current: I = {I} A")
    print(f"Radius: R = {radius*100:.1f} cm")
    print(f"Magnetic field inside: B = μ₀nI = {B_solenoid} T")
    print(f"Magnitude: |B| = {np.linalg.norm(B_solenoid)*1e3:.3f} mT")
    
    # Example 2: Toroid
    print("\n2. Toroidal coil")
    print("-" * 70)
    N = 500
    r_avg = 0.1  # 10 cm average radius
    I = 1  # 1 A
    
    B_toroid = (MU_0 * N * I) / (2 * np.pi * r_avg)
    
    print(f"Number of turns: N = {N}")
    print(f"Average radius: r = {r_avg*100:.1f} cm")
    print(f"Current: I = {I} A")
    print(f"Magnetic field: B = μ₀NI/(2πr) = {B_toroid*1e3:.3f} mT")


def demonstrate_magnetic_forces():
    """Demonstrate magnetic forces."""
    print("\n" + "=" * 70)
    print("MAGNETIC FORCES")
    print("=" * 70)
    
    # Example 1: Force on moving charge
    print("\n1. Force on a moving charged particle")
    print("-" * 70)
    q = 1.602e-19  # electron charge
    v = np.array([1e6, 0, 0])  # 1 Mm/s
    B = np.array([0, 0, 0.1])  # 0.1 T
    
    F = magnetic_force_on_charge(q, v, B)
    
    print(f"Charge: q = {q:.3e} C")
    print(f"Velocity: v = {v} m/s")
    print(f"Magnetic field: B = {B} T")
    print(f"Force: F = q(v × B) = {F} N")
    print(f"Magnitude: |F| = {np.linalg.norm(F):.3e} N")
    
    # Cyclotron radius
    m = 9.109e-31  # electron mass
    v_mag = np.linalg.norm(v)
    B_mag = np.linalg.norm(B)
    r_cyclotron = (m * v_mag) / (abs(q) * B_mag)
    print(f"\nCyclotron radius: r = mv/(qB) = {r_cyclotron*1e3:.3f} mm")
    
    # Example 2: Force on current-carrying wire
    print("\n2. Force on a current-carrying wire")
    print("-" * 70)
    I = 10  # 10 A
    L = np.array([0, 0.5, 0])  # 50 cm wire along y-axis
    B = np.array([0, 0, 0.5])  # 0.5 T along z-axis
    
    F_wire = magnetic_force_on_wire(I, L, B)
    
    print(f"Current: I = {I} A")
    print(f"Wire length vector: L = {L} m")
    print(f"Magnetic field: B = {B} T")
    print(f"Force: F = I(L × B) = {F_wire} N")
    print(f"Magnitude: |F| = {np.linalg.norm(F_wire):.3f} N")
    print(f"Force per unit length: F/L = {np.linalg.norm(F_wire)/np.linalg.norm(L):.3f} N/m")
    
    # Example 3: Force between parallel wires
    print("\n3. Force between two parallel wires")
    print("-" * 70)
    I1 = 5  # 5 A
    I2 = 3  # 3 A
    d = 0.1  # 10 cm separation
    L = 1.0  # 1 m length
    
    # Magnetic field from wire 1 at wire 2
    B1_at_2 = magnetic_field_infinite_wire(I1, d)
    
    # Force on wire 2
    F_per_length = I2 * B1_at_2
    F_total = F_per_length * L
    
    print(f"Wire 1 current: I₁ = {I1} A")
    print(f"Wire 2 current: I₂ = {I2} A")
    print(f"Separation: d = {d*100:.1f} cm")
    print(f"Wire length: L = {L} m")
    print(f"B-field from wire 1 at wire 2: B = {B1_at_2*1e6:.3f} μT")
    print(f"Force per unit length: F/L = μ₀I₁I₂/(2πd) = {F_per_length:.3e} N/m")
    print(f"Total force: F = {F_total:.3e} N")
    print(f"Direction: {'Attractive (same direction)' if I1*I2 > 0 else 'Repulsive (opposite)'}")


def visualize_magnetic_field_wire():
    """Visualize magnetic field around a straight wire."""
    print("\n" + "=" * 70)
    print("GENERATING MAGNETIC FIELD VISUALIZATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Field around a straight wire (cross-section)
    ax = axes[0, 0]
    I = 10  # 10 A
    
    # Create circular field pattern
    theta = np.linspace(0, 2*np.pi, 20)
    radii = np.array([0.02, 0.04, 0.06, 0.08, 0.1])
    
    for r in radii:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Magnetic field magnitude
        B = magnetic_field_infinite_wire(I, r)
        
        # Field is tangential (circular)
        Bx = -B * np.sin(theta)
        By = B * np.cos(theta)
        
        ax.quiver(x, y, Bx, By, scale=0.01, width=0.003, color='blue', alpha=0.6)
        ax.plot(x, y, 'k--', alpha=0.3, linewidth=0.5)
    
    # Wire at center
    ax.plot(0, 0, 'ro', markersize=15, label=f'Wire (I={I}A, out of page)')
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_title('Magnetic Field Around Straight Wire\n(Cross-sectional view)', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([-0.12, 0.12])
    ax.set_ylim([-0.12, 0.12])
    
    # 2. Solenoid field
    ax = axes[0, 1]
    
    # Draw solenoid coils
    n_coils = 15
    length = 0.6
    radius = 0.1
    
    for i in range(n_coils):
        z = -length/2 + i * length/(n_coils-1)
        circle = plt.Circle((z, 0), radius, fill=False, color='orange', linewidth=2)
        ax.add_patch(circle)
    
    # Field lines inside
    y_inside = np.linspace(-radius*0.8, radius*0.8, 5)
    z_inside = np.linspace(-length/2 + 0.05, length/2 - 0.05, 10)
    for y in y_inside:
        ax.arrow(-length/2 + 0.05, y, length - 0.1, 0, 
                head_width=0.02, head_length=0.03, fc='blue', ec='blue', linewidth=2)
    
    ax.set_xlabel('z (m)', fontsize=11)
    ax.set_ylabel('r (m)', fontsize=11)
    ax.set_title('Magnetic Field in Solenoid\n(Uniform inside, zero outside)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim([-0.15, 0.15])
    
    # 3. Circular loop field
    ax = axes[1, 0]
    
    # Draw loop
    theta_loop = np.linspace(0, 2*np.pi, 100)
    R = 0.1
    x_loop = R * np.cos(theta_loop)
    y_loop = R * np.sin(theta_loop)
    ax.plot(x_loop, y_loop, 'r-', linewidth=3, label='Current loop')
    
    # Field at center
    ax.arrow(0, 0, 0, 0.08, head_width=0.015, head_length=0.015, 
            fc='blue', ec='blue', linewidth=3, label='B-field')
    
    # Field lines (simplified representation)
    z_vals = np.linspace(-0.15, 0.15, 7)
    for z in z_vals:
        if abs(z) > 0.02:
            r_val = 0.05
            ax.plot([r_val], [z], 'b.', markersize=8)
            ax.plot([-r_val], [z], 'b.', markersize=8)
    
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('z (m)', fontsize=11)
    ax.set_title('Magnetic Field from Circular Loop\n(Side view)', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    
    # 4. Parallel wires
    ax = axes[1, 1]
    
    # Two parallel wires
    y1, y2 = -0.05, 0.05
    I1, I2 = 5, 5  # Same direction
    
    # Draw wires
    ax.plot(0, y1, 'ro', markersize=20, label=f'Wire 1 (I={I1}A, ⊙)')
    ax.plot(0, y2, 'ro', markersize=20, label=f'Wire 2 (I={I2}A, ⊙)')
    
    # Field from wire 1
    theta = np.linspace(0, 2*np.pi, 16)
    for r in [0.02, 0.04]:
        x = r * np.cos(theta)
        y = y1 + r * np.sin(theta)
        B = magnetic_field_infinite_wire(I1, r)
        Bx = -B * np.sin(theta)
        By = B * np.cos(theta)
        ax.quiver(x, y, Bx, By, scale=0.005, width=0.002, color='blue', alpha=0.4)
    
    # Field from wire 2
    for r in [0.02, 0.04]:
        x = r * np.cos(theta)
        y = y2 + r * np.sin(theta)
        B = magnetic_field_infinite_wire(I2, r)
        Bx = -B * np.sin(theta)
        By = B * np.cos(theta)
        ax.quiver(x, y, Bx, By, scale=0.005, width=0.002, color='red', alpha=0.4)
    
    # Force arrows
    ax.annotate('', xy=(0, y2-0.01), xytext=(0, y1+0.01),
                arrowprops=dict(arrowstyle='<->', color='green', lw=3))
    ax.text(0.01, 0, 'Attractive\nForce', fontsize=10, color='green', fontweight='bold')
    
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_title('Parallel Current-Carrying Wires\n(Same direction currents attract)', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    
    plt.tight_layout()
    plt.savefig('magnetic_field_patterns.png', dpi=150, bbox_inches='tight')
    print("\nSaved: magnetic_field_patterns.png")
    
    return fig


def visualize_lorentz_force_motion():
    """Visualize particle motion under magnetic force."""
    print("\n" + "=" * 70)
    print("GENERATING CHARGED PARTICLE MOTION VISUALIZATION")
    print("=" * 70)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Particle properties
    q = 1.602e-19  # proton
    m = 1.673e-27  # proton mass
    B = np.array([0, 0, 0.1])  # 0.1 T
    
    # Scenario 1: Perpendicular motion (circular)
    ax1 = fig.add_subplot(131)
    v0_perp = np.array([1e5, 0, 0])
    
    # Cyclotron radius and frequency
    v_mag = np.linalg.norm(v0_perp)
    B_mag = np.linalg.norm(B)
    r = (m * v_mag) / (q * B_mag)
    omega = (q * B_mag) / m
    T = 2 * np.pi / omega
    
    t = np.linspace(0, T, 100)
    x = r * (1 - np.cos(omega * t))
    y = r * np.sin(omega * t)
    
    ax1.plot(x*1000, y*1000, 'b-', linewidth=2, label='Particle path')
    ax1.plot(x[0]*1000, y[0]*1000, 'go', markersize=10, label='Start')
    ax1.quiver([0], [0], [0], [B_mag*100], scale=1, color='purple', 
               width=0.01, label='B field (into page)')
    ax1.set_xlabel('x (mm)', fontsize=10)
    ax1.set_ylabel('y (mm)', fontsize=10)
    ax1.set_title(f'Circular Motion (v ⊥ B)\nr = {r*1000:.2f} mm', 
                  fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Scenario 2: Parallel motion (no deflection)
    ax2 = fig.add_subplot(132)
    v0_parallel = np.array([0, 0, 1e5])
    
    t = np.linspace(0, 1e-6, 100)
    z = v0_parallel[2] * t
    
    ax2.plot(t*1e9, z, 'b-', linewidth=2, label='Particle path')
    ax2.axhline(y=0, color='purple', linestyle='--', label='B field direction')
    ax2.set_xlabel('Time (ns)', fontsize=10)
    ax2.set_ylabel('z position (m)', fontsize=10)
    ax2.set_title('Straight Motion (v ∥ B)\nNo magnetic force', 
                  fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Scenario 3: Helical motion (both components)
    ax3 = fig.add_subplot(133, projection='3d')
    v0_helix = np.array([5e4, 0, 1e5])
    
    v_perp = np.sqrt(v0_helix[0]**2 + v0_helix[1]**2)
    r_helix = (m * v_perp) / (q * B_mag)
    omega_helix = (q * B_mag) / m
    v_parallel = v0_helix[2]
    pitch = 2 * np.pi * v_parallel / omega_helix
    
    t = np.linspace(0, 3*T, 200)
    x = r_helix * (1 - np.cos(omega_helix * t))
    y = r_helix * np.sin(omega_helix * t)
    z = v_parallel * t
    
    ax3.plot(x*1000, y*1000, z, 'b-', linewidth=2, label='Helical path')
    ax3.plot([x[0]*1000], [y[0]*1000], [z[0]], 'go', markersize=10, label='Start')
    ax3.set_xlabel('x (mm)', fontsize=9)
    ax3.set_ylabel('y (mm)', fontsize=9)
    ax3.set_zlabel('z (m)', fontsize=9)
    ax3.set_title(f'Helical Motion (v has ⊥ and ∥ components)\nPitch = {pitch:.3e} m', 
                  fontsize=11, fontweight='bold')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('magnetic_force_motion.png', dpi=150, bbox_inches='tight')
    print("\nSaved: magnetic_force_motion.png")
    
    return fig


def main():
    """Main function demonstrating magnetic field concepts."""
    print("\n" + "=" * 70)
    print("MAGNETIC FIELD THEORY AND APPLICATIONS")
    print("=" * 70)
    print(f"\nPermeability of free space: μ₀ = {MU_0:.3e} T·m/A")
    print(f"Magnetic constant: μ₀/(4π) = {K_M:.3e} T·m/A")
    
    # Demonstrate concepts
    demonstrate_biot_savart()
    demonstrate_ampere_law()
    demonstrate_magnetic_forces()
    
    # Visualizations
    try:
        print("\nGenerating visualizations...")
        fig1 = visualize_magnetic_field_wire()
        fig2 = visualize_lorentz_force_motion()
        
        print("\n" + "=" * 70)
        print("All visualizations complete!")
        print("Saved files:")
        print("  - magnetic_field_patterns.png")
        print("  - magnetic_force_motion.png")
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
