"""
Euler's Formula: e^(ix) = cos(x) + i·sin(x)

The foundation of complex analysis and one of Euler's greatest discoveries.

This formula establishes a profound connection between:
- Exponential functions (e^x)
- Trigonometric functions (cos, sin)
- Complex numbers (i = √-1)

Key Properties:
1. |e^(ix)| = 1 for all real x (unit circle)
2. e^(ix) rotates counterclockwise by angle x
3. e^(2πi) = 1 (full rotation returns to start)
4. Derivative: d/dx[e^(ix)] = i·e^(ix)

Special Cases:
- x = 0:   e^0 = 1
- x = π/2: e^(iπ/2) = i
- x = π:   e^(iπ) = -1 (Euler's identity)
- x = 2π:  e^(i2π) = 1

Applications:
- Quantum mechanics (wave functions)
- Signal processing (Fourier analysis)
- Oscillations and waves
- AC circuit analysis
- Differential equations
- Computer graphics (rotations)

Proof:
Using Taylor series:
  e^(ix) = 1 + ix + (ix)²/2! + (ix)³/3! + (ix)⁴/4! + ...
         = 1 + ix - x²/2! - ix³/3! + x⁴/4! + ix⁵/5! - ...
         = (1 - x²/2! + x⁴/4! - ...) + i(x - x³/3! + x⁵/5! - ...)
         = cos(x) + i·sin(x)  ✓

Author: Leonhard Euler (1707-1783)
Published: 1748
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Circle, Wedge
from matplotlib import cm

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)


def taylor_series_comparison(x, n_terms=20):
    """
    Compare e^(ix) computed directly vs. Taylor series expansion
    
    Returns both the direct computation and series approximation
    """
    # Direct computation
    direct = np.exp(1j * x)
    
    # Taylor series: e^(ix) = ∑(ix)^n/n!
    series = 0
    for n in range(n_terms):
        series += (1j * x)**n / np.math.factorial(n)
    
    return direct, series


def demonstrate_euler_formula_proof():
    """
    Prove Euler's formula using Taylor series
    """
    print("=" * 80)
    print("EULER'S FORMULA: e^(ix) = cos(x) + i·sin(x)")
    print("=" * 80)
    
    print("\n1. TAYLOR SERIES PROOF")
    print("-" * 80)
    print("\nRecall the Taylor series:")
    print("   e^z = 1 + z + z²/2! + z³/3! + z⁴/4! + z⁵/5! + ...")
    print("   cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...")
    print("   sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...")
    
    print("\nSubstitute z = ix into e^z:")
    print("   e^(ix) = 1 + (ix) + (ix)²/2! + (ix)³/3! + (ix)⁴/4! + (ix)⁵/5! + ...")
    
    print("\nNote that i² = -1, i³ = -i, i⁴ = 1, i⁵ = i, ...")
    print("\nExpanding:")
    print("   e^(ix) = 1 + ix - x²/2! - ix³/3! + x⁴/4! + ix⁵/5! - x⁶/6! - ...")
    
    print("\nGroup real and imaginary parts:")
    print("   Real: 1 - x²/2! + x⁴/4! - x⁶/6! + ... = cos(x)")
    print("   Imag: x - x³/3! + x⁵/5! - x⁷/7! + ... = sin(x)")
    
    print("\nTherefore: e^(ix) = cos(x) + i·sin(x)  ✓")
    
    # Numerical verification
    print("\n2. NUMERICAL VERIFICATION")
    print("-" * 80)
    
    test_values = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi, 2*np.pi]
    labels = ['0', 'π/6', 'π/4', 'π/3', 'π/2', 'π', '2π']
    
    print(f"\n{'x':>6s}  {'e^(ix) (direct)':>25s}  {'cos(x) + i·sin(x)':>25s}  {'Difference':>12s}")
    print("-" * 80)
    
    for x, label in zip(test_values, labels):
        exp_val = np.exp(1j * x)
        euler_val = complex(np.cos(x), np.sin(x))
        diff = abs(exp_val - euler_val)
        
        print(f"{label:>6s}  {exp_val:>25s}  {euler_val:>25s}  {diff:12.2e}")


def demonstrate_properties():
    """
    Demonstrate key properties of Euler's formula
    """
    print("\n" + "=" * 80)
    print("KEY PROPERTIES OF EULER'S FORMULA")
    print("=" * 80)
    
    print("\n1. MAGNITUDE IS ALWAYS 1")
    print("-" * 80)
    print("   |e^(ix)| = √[cos²(x) + sin²(x)] = 1")
    print("\n   This means e^(ix) always lies on the unit circle!")
    
    # Verify for random values
    print("\n   Verification for random x values:")
    np.random.seed(42)
    for _ in range(5):
        x = np.random.uniform(-10, 10)
        mag = abs(np.exp(1j * x))
        print(f"     x = {x:8.4f}: |e^(ix)| = {mag:.15f}")
    
    print("\n2. ROTATION PROPERTY")
    print("-" * 80)
    print("   Multiplying by e^(ix) rotates by angle x:")
    print("   z' = z · e^(ix)")
    
    z = 1 + 1j
    for angle, label in [(np.pi/4, 'π/4'), (np.pi/2, 'π/2'), (np.pi, 'π')]:
        z_rotated = z * np.exp(1j * angle)
        print(f"\n   z = {z}, rotate by {label}:")
        print(f"   z' = {z_rotated}")
        print(f"   |z'| = {abs(z_rotated):.6f} (same as |z| = {abs(z):.6f})")
    
    print("\n3. PERIODICITY")
    print("-" * 80)
    print("   e^(i(x + 2π)) = e^(ix) · e^(i2π) = e^(ix) · 1 = e^(ix)")
    print("   Period = 2π")
    
    x = np.pi/3
    for k in range(5):
        val = np.exp(1j * (x + k * 2 * np.pi))
        print(f"   e^(i({x:.4f} + {k}·2π)) = {val}")
    
    print("\n4. DIFFERENTIATION")
    print("-" * 80)
    print("   d/dx[e^(ix)] = i · e^(ix)")
    
    # Numerical derivative check
    def numerical_derivative(f, x, h=1e-8):
        return (f(x + h) - f(x - h)) / (2 * h)
    
    x = 1.0
    analytical = 1j * np.exp(1j * x)
    numerical = numerical_derivative(lambda t: np.exp(1j * t), x)
    
    print(f"\n   At x = {x}:")
    print(f"   Analytical: i·e^(ix) = {analytical}")
    print(f"   Numerical:  d/dx[e^(ix)] = {numerical}")
    print(f"   Difference: {abs(analytical - numerical):.2e}")
    
    print("\n5. ADDITION FORMULA")
    print("-" * 80)
    print("   e^(i(x+y)) = e^(ix) · e^(iy)")
    print("   This leads to trig addition formulas!")
    
    x, y = np.pi/3, np.pi/4
    lhs = np.exp(1j * (x + y))
    rhs = np.exp(1j * x) * np.exp(1j * y)
    
    print(f"\n   x = π/3, y = π/4:")
    print(f"   e^(i(x+y)) = {lhs}")
    print(f"   e^(ix)·e^(iy) = {rhs}")
    print(f"   Difference: {abs(lhs - rhs):.2e}")
    
    # Derive cosine addition formula
    print("\n   Deriving cos(x+y) formula:")
    print("   e^(i(x+y)) = cos(x+y) + i·sin(x+y)")
    print("   e^(ix)·e^(iy) = [cos(x) + i·sin(x)][cos(y) + i·sin(y)]")
    print("                 = [cos(x)cos(y) - sin(x)sin(y)]")
    print("                   + i[sin(x)cos(y) + cos(x)sin(y)]")
    print("\n   Comparing real parts:")
    print("   cos(x+y) = cos(x)cos(y) - sin(x)sin(y)  ✓")


def demonstrate_applications():
    """
    Show practical applications
    """
    print("\n" + "=" * 80)
    print("APPLICATIONS OF EULER'S FORMULA")
    print("=" * 80)
    
    print("\n1. QUANTUM MECHANICS - Plane Wave")
    print("-" * 80)
    print("   ψ(x,t) = A·e^(i(kx - ωt))")
    print("   where k = 2π/λ (wave number), ω = 2πf (angular frequency)")
    
    A, k, omega = 1, 2*np.pi, np.pi
    t = 0
    x_vals = np.linspace(0, 2, 100)
    psi = A * np.exp(1j * (k * x_vals - omega * t))
    
    print(f"\n   At t=0, A={A}, k={k:.2f}, ω={omega:.2f}:")
    print(f"   Max |ψ| = {np.max(np.abs(psi)):.2f}")
    print(f"   ψ is complex: Re(ψ) and Im(ψ) are 90° out of phase")
    
    print("\n2. FOURIER TRANSFORM")
    print("-" * 80)
    print("   F(ω) = ∫_{-∞}^{∞} f(t)·e^(-iωt) dt")
    print("   Basis functions: e^(-iωt) = cos(ωt) - i·sin(ωt)")
    
    print("\n3. SOLVING DIFFERENTIAL EQUATIONS")
    print("-" * 80)
    print("   Example: y'' + 4y = 0")
    print("   Try y = e^(rx): r² + 4 = 0 → r = ±2i")
    print("   Solution: y = c₁e^(2ix) + c₂e^(-2ix)")
    print("           = A·cos(2x) + B·sin(2x)")
    
    print("\n4. SIGNAL PROCESSING - Phasor Representation")
    print("-" * 80)
    print("   AC voltage: V(t) = V₀·cos(ωt + φ)")
    print("   Phasor: Ṽ = V₀·e^(iφ)")
    print("   Time domain: V(t) = Re[Ṽ·e^(iωt)]")


def create_visualizations():
    """
    Create comprehensive visualizations
    """
    fig = plt.figure(figsize=(18, 13))
    
    # 1. Unit circle and rotating vector
    ax1 = plt.subplot(3, 4, 1)
    theta = np.linspace(0, 2*np.pi, 1000)
    
    ax1.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2, alpha=0.5, label='Unit circle')
    
    # Show several angles
    angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, np.pi, 4*np.pi/3, 3*np.pi/2]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(angles)))
    
    for angle, color in zip(angles, colors):
        z = np.exp(1j * angle)
        ax1.arrow(0, 0, z.real*0.95, z.imag*0.95, head_width=0.05, 
                 head_length=0.05, fc=color, ec=color, alpha=0.7)
        ax1.plot(z.real, z.imag, 'o', color=color, markersize=6)
    
    ax1.set_xlabel('Real: cos(x)', fontsize=10)
    ax1.set_ylabel('Imaginary: sin(x)', fontsize=10)
    ax1.set_title('e^(ix) on Unit Circle', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    # 2. Real and Imaginary parts
    ax2 = plt.subplot(3, 4, 2)
    x = np.linspace(0, 4*np.pi, 1000)
    
    ax2.plot(x, np.cos(x), 'b-', linewidth=2, label='Re(e^(ix)) = cos(x)')
    ax2.plot(x, np.sin(x), 'r-', linewidth=2, label='Im(e^(ix)) = sin(x)')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.set_xlabel('x', fontsize=10)
    ax2.set_ylabel('Value', fontsize=10)
    ax2.set_title('Components of e^(ix)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 4*np.pi])
    
    # 3. Magnitude (always 1)
    ax3 = plt.subplot(3, 4, 3)
    x_mag = np.linspace(-2*np.pi, 2*np.pi, 1000)
    magnitude = np.abs(np.exp(1j * x_mag))
    
    ax3.plot(x_mag, magnitude, 'g-', linewidth=3)
    ax3.axhline(y=1, color='r', linestyle='--', linewidth=2, alpha=0.5, label='|e^(ix)| = 1')
    ax3.set_xlabel('x', fontsize=10)
    ax3.set_ylabel('|e^(ix)|', fontsize=10)
    ax3.set_title('Magnitude Always Equals 1', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.9, 1.1])
    
    # 4. Phase angle
    ax4 = plt.subplot(3, 4, 4)
    phase = np.angle(np.exp(1j * x_mag))
    
    ax4.plot(x_mag, phase, 'purple', linewidth=2)
    ax4.set_xlabel('x', fontsize=10)
    ax4.set_ylabel('arg(e^(ix))', fontsize=10)
    ax4.set_title('Phase Angle = x (mod 2π)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linewidth=0.5)
    
    # 5. 3D helix
    ax5 = plt.subplot(3, 4, 5, projection='3d')
    t = np.linspace(0, 4*np.pi, 500)
    z_3d = np.exp(1j * t)
    
    ax5.plot(z_3d.real, z_3d.imag, t, 'b-', linewidth=2)
    ax5.set_xlabel('Re(e^(it))', fontsize=9)
    ax5.set_ylabel('Im(e^(it))', fontsize=9)
    ax5.set_zlabel('t', fontsize=9)
    ax5.set_title('3D Helix Representation', fontsize=11, fontweight='bold')
    
    # 6. Taylor series convergence
    ax6 = plt.subplot(3, 4, 6)
    x_test = np.pi/4
    n_terms_range = range(1, 21)
    errors = []
    
    for n in n_terms_range:
        direct, series = taylor_series_comparison(x_test, n)
        errors.append(abs(direct - series))
    
    ax6.semilogy(list(n_terms_range), errors, 'ro-', linewidth=2, markersize=5)
    ax6.set_xlabel('Number of terms', fontsize=10)
    ax6.set_ylabel('|Error|', fontsize=10)
    ax6.set_title(f'Taylor Series Convergence (x=π/4)', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')
    
    # 7. Rotation demonstration
    ax7 = plt.subplot(3, 4, 7)
    z0 = 1 + 0.5j
    angles_rot = np.linspace(0, 2*np.pi, 9)
    
    ax7.plot(np.cos(theta), np.sin(theta), 'gray', linewidth=1, alpha=0.3)
    ax7.plot([0, z0.real], [0, z0.imag], 'b-', linewidth=3, label='Original z')
    ax7.plot(z0.real, z0.imag, 'bo', markersize=10)
    
    for i, angle in enumerate(angles_rot[1:]):
        z_rot = z0 * np.exp(1j * angle)
        alpha = 0.3 + 0.5 * (i / len(angles_rot))
        ax7.plot([0, z_rot.real], [0, z_rot.imag], 'r-', linewidth=2, alpha=alpha)
        ax7.plot(z_rot.real, z_rot.imag, 'ro', markersize=6, alpha=alpha)
    
    ax7.set_xlabel('Real', fontsize=10)
    ax7.set_ylabel('Imaginary', fontsize=10)
    ax7.set_title('Rotation: z·e^(ix)', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.set_aspect('equal')
    ax7.axhline(y=0, color='k', linewidth=0.5)
    ax7.axvline(x=0, color='k', linewidth=0.5)
    
    # 8. Complex plane trajectory
    ax8 = plt.subplot(3, 4, 8)
    t_traj = np.linspace(0, 2*np.pi, 1000)
    z_traj = np.exp(1j * t_traj)
    
    # Color by angle
    colors_traj = t_traj / (2*np.pi)
    scatter = ax8.scatter(z_traj.real, z_traj.imag, c=colors_traj, 
                         cmap='hsv', s=10, alpha=0.8)
    
    ax8.set_xlabel('Real', fontsize=10)
    ax8.set_ylabel('Imaginary', fontsize=10)
    ax8.set_title('Colored by Angle', fontsize=11, fontweight='bold')
    ax8.set_aspect('equal')
    ax8.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax8, label='Angle / 2π')
    
    # 9. Quantum wave function
    ax9 = plt.subplot(3, 4, 9)
    x_qm = np.linspace(0, 4, 1000)
    k, omega, t_qm = 3*np.pi, 0, 0
    psi = np.exp(1j * k * x_qm)
    
    ax9.plot(x_qm, psi.real, 'b-', linewidth=2, label='Re(ψ)')
    ax9.plot(x_qm, psi.imag, 'r-', linewidth=2, label='Im(ψ)')
    ax9.plot(x_qm, np.abs(psi), 'g--', linewidth=2, label='|ψ|')
    ax9.axhline(y=0, color='k', linewidth=0.5)
    ax9.set_xlabel('Position x', fontsize=10)
    ax9.set_ylabel('Wave Function', fontsize=10)
    ax9.set_title('Quantum Plane Wave: ψ = e^(ikx)', fontsize=11, fontweight='bold')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    # 10. Derivative visualization
    ax10 = plt.subplot(3, 4, 10)
    x_deriv = np.linspace(0, 2*np.pi, 100)
    f = np.exp(1j * x_deriv)
    df = 1j * f  # Derivative
    
    # Plot as vectors in complex plane
    for i in range(0, len(x_deriv), 10):
        # Function value
        ax10.arrow(0, 0, f[i].real*0.9, f[i].imag*0.9, 
                  head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.5)
        # Derivative (perpendicular)
        ax10.arrow(f[i].real, f[i].imag, df[i].real*0.15, df[i].imag*0.15,
                  head_width=0.04, head_length=0.04, fc='red', ec='red', alpha=0.7)
    
    ax10.plot(np.cos(theta), np.sin(theta), 'gray', linewidth=1, alpha=0.3)
    ax10.set_xlabel('Real', fontsize=10)
    ax10.set_ylabel('Imaginary', fontsize=10)
    ax10.set_title('d/dx[e^(ix)] = i·e^(ix)', fontsize=11, fontweight='bold')
    ax10.set_aspect('equal')
    ax10.grid(True, alpha=0.3)
    
    # 11. Periodicity
    ax11 = plt.subplot(3, 4, 11)
    x_period = np.linspace(-4*np.pi, 4*np.pi, 1000)
    
    ax11.plot(x_period, np.exp(1j * x_period).real, 'b-', linewidth=2, label='Re(e^(ix))')
    ax11.axvline(x=-2*np.pi, color='r', linestyle='--', alpha=0.5)
    ax11.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax11.axvline(x=2*np.pi, color='r', linestyle='--', alpha=0.5, label='Period = 2π')
    ax11.axhline(y=0, color='k', linewidth=0.5)
    ax11.set_xlabel('x', fontsize=10)
    ax11.set_ylabel('Re(e^(ix))', fontsize=10)
    ax11.set_title('Periodicity: e^(i(x+2π)) = e^(ix)', fontsize=11, fontweight='bold')
    ax11.legend(fontsize=9)
    ax11.grid(True, alpha=0.3)
    
    # 12. Summary
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    summary = """
EULER'S FORMULA
═══════════════════════════

  e^(ix) = cos(x) + i·sin(x)

Key Properties:
───────────────
• |e^(ix)| = 1 always
• Arg(e^(ix)) = x
• Period = 2π
• d/dx[e^(ix)] = i·e^(ix)

Special Values:
───────────────
• e^(i·0) = 1
• e^(iπ/2) = i
• e^(iπ) = -1
• e^(i3π/2) = -i
• e^(i2π) = 1

Applications:
─────────────
• Quantum mechanics
• Fourier analysis
• Differential equations
• Signal processing
• AC circuits
• Rotations (2D/3D)

Connections:
────────────
• Links exp, trig, complex
• Unifies analysis & geometry
• Foundation of complex theory
═══════════════════════════
    """
    
    ax12.text(0.05, 0.95, summary, transform=ax12.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle("Euler's Formula: e^(ix) = cos(x) + i·sin(x)", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def main():
    """
    Main demonstration
    """
    print("\n" + "=" * 80)
    print(" EULER'S FORMULA: e^(ix) = cos(x) + i·sin(x)")
    print(" Foundation of Complex Analysis")
    print("=" * 80)
    
    # Proof and verification
    demonstrate_euler_formula_proof()
    
    # Properties
    demonstrate_properties()
    
    # Applications
    demonstrate_applications()
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    
    # Create visualizations
    fig = create_visualizations()
    
    print("\n✓ Visualizations created successfully!")
    print("\nKey Insights:")
    print("  • Euler's formula unites exponentials, trig, and complex numbers")
    print("  • e^(ix) traces the unit circle as x varies")
    print("  • Provides elegant solutions to differential equations")
    print("  • Foundation for Fourier analysis and quantum mechanics")
    print("  • Makes complex arithmetic geometrically intuitive")
    
    plt.show()


if __name__ == "__main__":
    main()
