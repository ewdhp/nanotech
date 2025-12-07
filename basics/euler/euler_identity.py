"""
Euler's Identity: e^(iπ) + 1 = 0

"The most beautiful equation in mathematics" - Richard Feynman

This script demonstrates Euler's identity and its deep connections between
five fundamental mathematical constants:
- e (Euler's number, base of natural logarithm)
- i (imaginary unit, √(-1))
- π (pi, ratio of circle circumference to diameter)
- 1 (multiplicative identity)
- 0 (additive identity)

The identity is a special case of Euler's formula: e^(ix) = cos(x) + i·sin(x)
When x = π: e^(iπ) = cos(π) + i·sin(π) = -1 + 0i = -1
Therefore: e^(iπ) + 1 = 0

Applications:
- Quantum mechanics (wave functions)
- Signal processing (Fourier transforms)
- Electrical engineering (AC circuits)
- Complex analysis
- Number theory

Author: Leonhard Euler (1707-1783)
Published: 1748 in "Introductio in analysin infinitorum"
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


class Arrow3D(FancyArrowPatch):
    """Custom 3D arrow for matplotlib"""
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


def verify_euler_identity():
    """
    Numerically verify Euler's identity using multiple methods
    """
    print("=" * 80)
    print("EULER'S IDENTITY: e^(iπ) + 1 = 0")
    print("=" * 80)
    
    print("\n1. DIRECT COMPUTATION")
    print("-" * 80)
    
    # Method 1: Using numpy's complex exponential
    result1 = np.exp(1j * np.pi) + 1
    print(f"   np.exp(i·π) + 1 = {result1}")
    print(f"   Real part: {result1.real:.15e}")
    print(f"   Imaginary part: {result1.imag:.15e}")
    print(f"   Magnitude: {abs(result1):.15e}")
    
    # Method 2: Using Euler's formula explicitly
    result2 = complex(np.cos(np.pi), np.sin(np.pi)) + 1
    print(f"\n   cos(π) + i·sin(π) + 1 = {result2}")
    print(f"   Real part: {result2.real:.15e}")
    print(f"   Imaginary part: {result2.imag:.15e}")
    
    # Method 3: Using series expansion
    print("\n2. SERIES EXPANSION VERIFICATION")
    print("-" * 80)
    print("   e^(iπ) = ∑(iπ)^n/n! from n=0 to ∞")
    
    def exp_series(z, n_terms=50):
        """Compute e^z using Taylor series"""
        result = 0
        for n in range(n_terms):
            result += z**n / np.math.factorial(n)
        return result
    
    z = 1j * np.pi
    result3 = exp_series(z, n_terms=100) + 1
    print(f"\n   Using 100 terms of Taylor series:")
    print(f"   Result: {result3}")
    print(f"   Magnitude: {abs(result3):.15e}")
    
    # Show convergence
    print("\n   Convergence of series expansion:")
    for n in [5, 10, 20, 50, 100]:
        approx = exp_series(z, n) + 1
        print(f"     {n:3d} terms: |e^(iπ) + 1| = {abs(approx):.15e}")


def demonstrate_euler_formula():
    """
    Show how Euler's identity is a special case of Euler's formula
    """
    print("\n" + "=" * 80)
    print("EULER'S FORMULA: e^(ix) = cos(x) + i·sin(x)")
    print("=" * 80)
    
    print("\nSpecial values:")
    special_angles = [
        (0, "0"),
        (np.pi/6, "π/6"),
        (np.pi/4, "π/4"),
        (np.pi/3, "π/3"),
        (np.pi/2, "π/2"),
        (np.pi, "π"),
        (3*np.pi/2, "3π/2"),
        (2*np.pi, "2π")
    ]
    
    for angle, label in special_angles:
        exp_val = np.exp(1j * angle)
        euler_val = complex(np.cos(angle), np.sin(angle))
        print(f"   x = {label:6s}: e^(ix) = {exp_val:20s}, "
              f"cos(x) + i·sin(x) = {euler_val:20s}")
    
    print(f"\n   At x = π:")
    print(f"     e^(iπ) = cos(π) + i·sin(π)")
    print(f"            = -1 + 0i")
    print(f"            = -1")
    print(f"   Therefore: e^(iπ) + 1 = 0  ✓")


def show_five_constants():
    """
    Explain the significance of the five fundamental constants
    """
    print("\n" + "=" * 80)
    print("THE FIVE FUNDAMENTAL CONSTANTS")
    print("=" * 80)
    
    print("\n1. e ≈ 2.71828... (Euler's number)")
    print("   • Base of natural logarithm")
    print("   • lim(1 + 1/n)^n as n → ∞")
    print("   • ∑1/n! from n=0 to ∞")
    print("   • Appears in growth/decay, compound interest, calculus")
    e_approx = sum(1/np.math.factorial(n) for n in range(50))
    print(f"   • Value: {np.e:.15f}")
    print(f"   • Series: {e_approx:.15f}")
    
    print("\n2. i = √(-1) (imaginary unit)")
    print("   • i² = -1")
    print("   • Extends real numbers to complex plane")
    print("   • Essential for quantum mechanics, signal processing")
    print("   • Enables solving x² + 1 = 0")
    
    print("\n3. π ≈ 3.14159... (pi)")
    print("   • Ratio of circle circumference to diameter")
    print("   • ∑(-1)^n/(2n+1) · 4 (Leibniz formula)")
    print("   • Appears in geometry, trigonometry, waves")
    print(f"   • Value: {np.pi:.15f}")
    
    print("\n4. 1 (multiplicative identity)")
    print("   • a × 1 = a for all a")
    print("   • Foundation of arithmetic")
    
    print("\n5. 0 (additive identity)")
    print("   • a + 0 = a for all a")
    print("   • Origin point, null element")
    
    print("\nEuler's identity unites all five in one elegant equation:")
    print("   e^(iπ) + 1 = 0")
    print("\nRearranged: e^(iπ) = -1")
    print("   An irrational number raised to an imaginary power")
    print("   equals a real number!")


def create_visualizations():
    """
    Create comprehensive visualizations of Euler's identity
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Unit circle and e^(iθ)
    ax1 = plt.subplot(2, 3, 1)
    theta = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(theta)
    y = np.sin(theta)
    
    ax1.plot(x, y, 'b-', linewidth=2, label='Unit circle: |z| = 1')
    ax1.plot([0, np.cos(np.pi)], [0, np.sin(np.pi)], 'r-', linewidth=3, 
             label='e^(iπ) = -1')
    ax1.plot([np.cos(np.pi), 0], [np.sin(np.pi), 0], 'g--', linewidth=2,
             label='e^(iπ) + 1 = 0')
    
    # Mark special points
    special_points = [
        (0, np.exp(1j*0), '1 = e^(i·0)'),
        (np.pi/2, np.exp(1j*np.pi/2), 'i = e^(iπ/2)'),
        (np.pi, np.exp(1j*np.pi), '-1 = e^(iπ)'),
        (3*np.pi/2, np.exp(1j*3*np.pi/2), '-i = e^(i3π/2)'),
    ]
    
    for angle, point, label in special_points:
        ax1.plot(point.real, point.imag, 'ro', markersize=10)
        offset_x = 0.15 * np.cos(angle)
        offset_y = 0.15 * np.sin(angle)
        ax1.text(point.real + offset_x, point.imag + offset_y, label,
                fontsize=9, ha='center')
    
    ax1.plot(0, 0, 'ko', markersize=8)
    ax1.text(0.1, -0.15, '0', fontsize=11, fontweight='bold')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.set_xlabel('Real axis', fontsize=11)
    ax1.set_ylabel('Imaginary axis', fontsize=11)
    ax1.set_title("Euler's Identity on Complex Plane", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    
    # 2. Euler's formula visualization
    ax2 = plt.subplot(2, 3, 2)
    theta_range = np.linspace(0, 2*np.pi, 100)
    real_parts = np.cos(theta_range)
    imag_parts = np.sin(theta_range)
    
    ax2.plot(theta_range/np.pi, real_parts, 'b-', linewidth=2, label='cos(x) = Re(e^(ix))')
    ax2.plot(theta_range/np.pi, imag_parts, 'r-', linewidth=2, label='sin(x) = Im(e^(ix))')
    ax2.axvline(x=1, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax2.plot(1, -1, 'go', markersize=12, label='x=π: e^(iπ)=-1')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.set_xlabel('x/π', fontsize=11)
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_title('e^(ix) = cos(x) + i·sin(x)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 2])
    
    # 3. 3D spiral representation
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    t = np.linspace(0, 4*np.pi, 500)
    z_vals = np.exp(1j * t)
    ax3.plot(z_vals.real, z_vals.imag, t, 'b-', linewidth=2, alpha=0.8)
    
    # Mark special points
    for mult in [0, 0.5, 1, 1.5, 2]:
        t_special = mult * np.pi
        z_special = np.exp(1j * t_special)
        ax3.plot([z_special.real], [z_special.imag], [t_special], 
                'ro', markersize=8)
    
    ax3.set_xlabel('Re(e^(it))', fontsize=10)
    ax3.set_ylabel('Im(e^(it))', fontsize=10)
    ax3.set_zlabel('t', fontsize=10)
    ax3.set_title('3D Spiral: e^(it) as t varies', fontsize=12, fontweight='bold')
    
    # 4. Series convergence
    ax4 = plt.subplot(2, 3, 4)
    n_max = 30
    n_terms = range(1, n_max + 1)
    
    def series_approx(z, n):
        return sum(z**k / np.math.factorial(k) for k in range(n))
    
    z = 1j * np.pi
    errors = [abs(series_approx(z, n) + 1) for n in n_terms]
    
    ax4.semilogy(n_terms, errors, 'b-o', linewidth=2, markersize=5)
    ax4.set_xlabel('Number of terms', fontsize=11)
    ax4.set_ylabel('|e^(iπ) + 1|', fontsize=11)
    ax4.set_title('Series Convergence to Zero', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.axhline(y=1e-15, color='r', linestyle='--', alpha=0.5, 
                label='Machine precision')
    ax4.legend(fontsize=9)
    
    # 5. Polar form
    ax5 = plt.subplot(2, 3, 5, projection='polar')
    
    # Draw angle for π
    theta_arc = np.linspace(0, np.pi, 100)
    r_arc = np.ones_like(theta_arc)
    ax5.plot(theta_arc, r_arc, 'r-', linewidth=3)
    ax5.plot([np.pi], [1], 'ro', markersize=15, label='e^(iπ) at θ=π, r=1')
    
    # Mark angle
    ax5.set_theta_zero_location('E')
    ax5.set_theta_direction(1)
    ax5.set_title('Polar Form: e^(iθ), θ=π', fontsize=13, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', fontsize=9)
    
    # 6. Historical context and applications
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    info_text = """
EULER'S IDENTITY
═══════════════════════════════════

    e^(iπ) + 1 = 0

Historical Context:
──────────────────
• Discovered: Leonhard Euler (1748)
• Published: "Introductio in analysin 
  infinitorum"
• Called "the most beautiful equation"
  by Richard Feynman

Significance:
─────────────
• Unites 5 fundamental constants
• Bridge between analysis & geometry
• Foundation of complex analysis
• Key to Fourier transforms

Applications:
─────────────
• Quantum Mechanics
  ψ(x,t) = Ae^(i(kx-ωt))
  
• Signal Processing
  Fourier transform basis
  
• Electrical Engineering
  AC circuit analysis
  
• Differential Equations
  Solutions to ODEs/PDEs

Generalizations:
────────────────
• e^(ix) = cos(x) + i·sin(x)
• |e^(ix)| = 1 for all real x
• e^(2πi) = 1 (full rotation)
═══════════════════════════════════
    """
    
    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle("Euler's Identity: The Most Beautiful Equation in Mathematics", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def demonstrate_applications():
    """
    Show practical applications of Euler's identity
    """
    print("\n" + "=" * 80)
    print("APPLICATIONS OF EULER'S IDENTITY")
    print("=" * 80)
    
    print("\n1. QUANTUM MECHANICS - Wave Function")
    print("-" * 80)
    print("   ψ(x,t) = A·e^(i(kx - ωt))")
    print("   where k is wave number, ω is angular frequency")
    print("\n   Using Euler's formula:")
    print("   ψ(x,t) = A[cos(kx - ωt) + i·sin(kx - ωt)]")
    
    print("\n2. FOURIER TRANSFORM")
    print("-" * 80)
    print("   F(ω) = ∫f(t)·e^(-iωt) dt")
    print("   Decomposes signals into frequency components")
    
    print("\n3. SOLVING DIFFERENTIAL EQUATIONS")
    print("-" * 80)
    print("   For y'' + y = 0:")
    print("   Try y = e^(rx), get r² + 1 = 0, so r = ±i")
    print("   General solution: y = c₁e^(ix) + c₂e^(-ix)")
    print("                      = A·cos(x) + B·sin(x)")
    
    print("\n4. ELECTRICAL ENGINEERING - AC Circuits")
    print("-" * 80)
    print("   Voltage: V(t) = V₀·e^(iωt)")
    print("   Impedance calculations using complex numbers")
    
    print("\n5. ROTATIONS IN 2D")
    print("-" * 80)
    print("   Rotation by angle θ:")
    print("   z' = z·e^(iθ)")
    print("   Combines scaling and rotation elegantly")


def main():
    """
    Main demonstration of Euler's identity
    """
    print("\n" + "=" * 80)
    print(" EULER'S IDENTITY: e^(iπ) + 1 = 0")
    print(" The Most Beautiful Equation in Mathematics")
    print("=" * 80)
    
    # Numerical verification
    verify_euler_identity()
    
    # Show Euler's formula
    demonstrate_euler_formula()
    
    # Explain the five constants
    show_five_constants()
    
    # Show applications
    demonstrate_applications()
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    
    # Create visualizations
    fig = create_visualizations()
    
    print("\n✓ Visualizations created successfully!")
    print("\nKey Insights:")
    print("  • Euler's identity unites algebra, geometry, and analysis")
    print("  • It's a special case of e^(ix) = cos(x) + i·sin(x)")
    print("  • Connects the five most important constants in mathematics")
    print("  • Foundation for quantum mechanics, signal processing, and more")
    print("  • Demonstrates the deep unity of mathematical structures")
    
    plt.show()


if __name__ == "__main__":
    main()
