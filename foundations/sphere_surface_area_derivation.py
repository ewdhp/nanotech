import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
RADIUS = 2.0
NUM_RINGS = 20  # Number of rings for visualization

print("=" * 70)
print("SPHERE SURFACE AREA: Calculus Derivation")
print("=" * 70)
print("\nWe slice a sphere into thin horizontal rings and sum their areas.")
print("\nFor a sphere of radius r:")
print("  - Each ring is at angle θ from the top (0 ≤ θ ≤ π)")
print("  - Ring radius at angle θ: r sin(θ)")
print("  - Ring circumference: 2πr sin(θ)")
print("  - Ring thickness (arc length): r dθ")
print("  - Ring surface area: dA = (2πr sin(θ)) × (r dθ)")
print("\nIntegrating from θ=0 to θ=π:")
print("  A = ∫₀^π 2πr² sin(θ) dθ")
print("  A = 2πr² ∫₀^π sin(θ) dθ")
print("  A = 2πr² [-cos(θ)]₀^π")
print("  A = 2πr² [-cos(π) - (-cos(0))]")
print("  A = 2πr² [-(-1) - (-1)]")
print("  A = 2πr² [1 + 1]")
print("  A = 2πr² × 2")
print("  A = 4πr²")
print("=" * 70)

# Numerical integration to verify
print(f"\nNumerical Verification for r = {RADIUS}:")
print("-" * 70)

# Analytical result
analytical_area = 4 * np.pi * RADIUS**2
print(f"Analytical formula: A = 4π({RADIUS})² = {analytical_area:.6f}")

# Numerical integration using different numbers of slices
for n_slices in [10, 100, 1000, 10000]:
    theta = np.linspace(0, np.pi, n_slices)
    dtheta = np.pi / (n_slices - 1)
    
    # Ring area at each theta: 2πr² sin(θ)
    ring_areas = 2 * np.pi * RADIUS**2 * np.sin(theta)
    
    # Sum using trapezoidal rule
    numerical_area = np.trapz(ring_areas, theta)
    error = abs(numerical_area - analytical_area)
    error_percent = (error / analytical_area) * 100
    
    print(f"  n={n_slices:5d} slices: A = {numerical_area:.6f}  "
          f"(error: {error:.6f}, {error_percent:.4f}%)")

print("=" * 70)

# Create visualization
fig = plt.figure(figsize=(18, 6))

# Plot 1: 3D Sphere with Rings
ax1 = fig.add_subplot(131, projection='3d')

# Draw the sphere surface
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = RADIUS * np.outer(np.cos(u), np.sin(v))
y_sphere = RADIUS * np.outer(np.sin(u), np.sin(v))
z_sphere = RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))

ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='lightblue')

# Draw horizontal rings
theta_rings = np.linspace(0, np.pi, NUM_RINGS)
colors = plt.cm.rainbow(np.linspace(0, 1, NUM_RINGS))

for i, theta in enumerate(theta_rings):
    z_ring = RADIUS * np.cos(theta)
    ring_radius = RADIUS * np.sin(theta)
    
    phi = np.linspace(0, 2*np.pi, 100)
    x_ring = ring_radius * np.cos(phi)
    y_ring = ring_radius * np.sin(phi)
    z_ring_array = np.full_like(phi, z_ring)
    
    ax1.plot(x_ring, y_ring, z_ring_array, color=colors[i], linewidth=2, alpha=0.8)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title(f'Sphere Sliced into {NUM_RINGS} Rings\n(r = {RADIUS})', 
              fontsize=12, fontweight='bold')
ax1.set_box_aspect([1,1,1])

# Plot 2: Ring dimensions as function of θ
ax2 = fig.add_subplot(132)

theta_continuous = np.linspace(0, np.pi, 200)
ring_radius_continuous = RADIUS * np.sin(theta_continuous)
ring_circumference = 2 * np.pi * ring_radius_continuous
ring_arc_length = RADIUS * np.gradient(theta_continuous)

ax2_twin = ax2.twinx()

# Plot ring radius
line1 = ax2.plot(theta_continuous, ring_radius_continuous, 'b-', 
                 linewidth=2.5, label='Ring radius: r sin(θ)')
ax2.fill_between(theta_continuous, 0, ring_radius_continuous, alpha=0.2, color='blue')

# Plot ring circumference on twin axis
line2 = ax2_twin.plot(theta_continuous, ring_circumference, 'r-', 
                      linewidth=2.5, label='Ring circumference: 2πr sin(θ)')

# Mark key angles
key_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
key_labels = ['0', 'π/4', 'π/2', '3π/4', 'π']

for angle, label in zip(key_angles, key_labels):
    ax2.axvline(angle, color='gray', linestyle='--', alpha=0.3)
    if angle in [0, np.pi/2, np.pi]:
        r_val = RADIUS * np.sin(angle)
        ax2.plot(angle, r_val, 'bo', markersize=8)

ax2.set_xlabel('Angle θ (radians)', fontsize=11)
ax2.set_ylabel('Ring Radius (units)', fontsize=11, color='b')
ax2_twin.set_ylabel('Ring Circumference (units)', fontsize=11, color='r')
ax2.set_title('Ring Dimensions vs Angle θ', fontsize=12, fontweight='bold')
ax2.set_xticks(key_angles)
ax2.set_xticklabels(key_labels)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='y', labelcolor='b')
ax2_twin.tick_params(axis='y', labelcolor='r')

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper right', fontsize=10)

# Plot 3: Ring surface area contribution
ax3 = fig.add_subplot(133)

# Ring area as function of theta: dA = 2πr² sin(θ) dθ
ring_area_density = 2 * np.pi * RADIUS**2 * np.sin(theta_continuous)

ax3.plot(theta_continuous, ring_area_density, 'g-', linewidth=2.5, 
         label='dA/dθ = 2πr² sin(θ)')
ax3.fill_between(theta_continuous, 0, ring_area_density, alpha=0.3, color='green')

# Shade the area under curve (this represents the integral)
ax3.text(np.pi/2, max(ring_area_density)*0.5, 
         f'Total Area =\n∫₀^π 2πr² sin(θ) dθ\n= 4πr²\n= {analytical_area:.2f}',
         fontsize=11, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.7))

# Mark maximum at θ = π/2
max_idx = np.argmax(ring_area_density)
max_theta = theta_continuous[max_idx]
max_value = ring_area_density[max_idx]
ax3.plot(max_theta, max_value, 'ro', markersize=10, 
         label=f'Maximum at θ=π/2')
ax3.annotate(f'Max = 2πr²\n= {max_value:.2f}',
            xy=(max_theta, max_value), xytext=(max_theta + 0.5, max_value),
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax3.set_xlabel('Angle θ (radians)', fontsize=11)
ax3.set_ylabel('Ring Area Density dA/dθ', fontsize=11)
ax3.set_title('Surface Area Contribution by Ring', fontsize=12, fontweight='bold')
ax3.set_xticks(key_angles)
ax3.set_xticklabels(key_labels)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.set_xlim(0, np.pi)
ax3.set_ylim(0, max_value * 1.1)

plt.tight_layout()
output_file = '/home/ewd/nanotech/sphere_surface_area_derivation.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')

print(f"\nVisualization saved to: {output_file}")
print("Displaying plot window...")
print("=" * 70)

plt.show()
