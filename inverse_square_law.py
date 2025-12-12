import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
TOTAL_FLUX = 1000  # lumens (or coulombs for electric field)
NUM_FLUX_LINES = 24  # number of flux lines to draw

def calculate_intensity(r, total_flux=TOTAL_FLUX):
    """Calculate flux density at distance r using inverse square law"""
    if r == 0:
        return np.inf
    surface_area = 4 * np.pi * r**2
    intensity = total_flux / surface_area
    return intensity

# Calculate and display intensities at different distances
print("=" * 60)
print("INVERSE SQUARE LAW: Flux Density Calculations")
print("=" * 60)
print(f"Total Flux (Φ): {TOTAL_FLUX} lumens")
print(f"Formula: I = Φ / (4πr²)")
print("=" * 60)

distances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
intensities = []

for r in distances:
    intensity = calculate_intensity(r)
    intensities.append(intensity)
    surface_area = 4 * np.pi * r**2
    print(f"\nAt r = {r} m:")
    print(f"  Surface Area = 4π({r})² = {surface_area:.2f} m²")
    print(f"  Intensity = {TOTAL_FLUX}/{surface_area:.2f} = {intensity:.2f} lumens/m²")

# Verify inverse square relationship
print("\n" + "=" * 60)
print("VERIFICATION: Inverse Square Relationship")
print("=" * 60)
r1, r2 = 1.0, 2.0
i1 = calculate_intensity(r1)
i2 = calculate_intensity(r2)
ratio_intensity = i1 / i2
ratio_distance = (r2 / r1)**2
print(f"When distance doubles from {r1}m to {r2}m:")
print(f"  Intensity ratio: I₁/I₂ = {ratio_intensity:.2f}")
print(f"  Distance ratio squared: (r₂/r₁)² = {ratio_distance:.2f}")
print(f"  Match: {np.isclose(ratio_intensity, ratio_distance)}")

# Create visualization
fig = plt.figure(figsize=(16, 6))

# Plot 1: 2D Flux Lines (Top View)
ax1 = fig.add_subplot(131)
angles = np.linspace(0, 2*np.pi, NUM_FLUX_LINES, endpoint=False)

for angle in angles:
    r_line = np.linspace(0.1, 3, 100)
    x = r_line * np.cos(angle)
    y = r_line * np.sin(angle)
    ax1.plot(x, y, 'b-', alpha=0.6, linewidth=0.8)

# Draw circles at different radii
radii = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for r in radii:
    circle = plt.Circle((0, 0), r, fill=False, color='red', 
                        linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.add_patch(circle)
    # Add labels
    ax1.text(r, 0.1, f'r={r}', fontsize=9, ha='center')

# Source point
ax1.plot(0, 0, 'ro', markersize=12, label='Source (S)')
ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-3.5, 3.5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Distance (m)', fontsize=11)
ax1.set_ylabel('Distance (m)', fontsize=11)
ax1.set_title('2D View: Flux Lines from Point Source', fontsize=12, fontweight='bold')
ax1.legend()

# Plot 2: 3D Flux Lines
ax2 = fig.add_subplot(132, projection='3d')

# Create flux lines in 3D using spherical coordinates
n_lines_theta = 6
n_lines_phi = 8
theta_vals = np.linspace(0, np.pi, n_lines_theta)
phi_vals = np.linspace(0, 2*np.pi, n_lines_phi, endpoint=False)

for theta in theta_vals:
    for phi in phi_vals:
        r_line = np.linspace(0.1, 3, 50)
        x = r_line * np.sin(theta) * np.cos(phi)
        y = r_line * np.sin(theta) * np.sin(phi)
        z = r_line * np.cos(theta)
        ax2.plot(x, y, z, 'b-', alpha=0.5, linewidth=0.8)

# Draw spheres at different radii
for r in [1.0, 2.0, 3.0]:
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = r * np.outer(np.cos(u), np.sin(v))
    y_sphere = r * np.outer(np.sin(u), np.sin(v))
    z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_wireframe(x_sphere, y_sphere, z_sphere, 
                       color='red', alpha=0.3, linewidth=0.5)

ax2.scatter([0], [0], [0], color='red', s=100, label='Source (S)')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_zlabel('Z (m)')
ax2.set_title('3D View: Flux Lines in Space', fontsize=12, fontweight='bold')
ax2.legend()
ax2.set_box_aspect([1,1,1])

# Plot 3: Intensity vs Distance
ax3 = fig.add_subplot(133)
r_continuous = np.linspace(0.1, 3, 300)
i_continuous = [calculate_intensity(r) for r in r_continuous]

ax3.plot(r_continuous, i_continuous, 'b-', linewidth=2.5, label='I = Φ/(4πr²)')
ax3.plot(distances, intensities, 'ro', markersize=10, 
         label='Calculated Points', zorder=5)

# Add annotations for key points
for r, i in zip(distances[::2], intensities[::2]):
    ax3.annotate(f'({r}m, {i:.1f})', 
                xy=(r, i), xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax3.set_xlabel('Distance from Source, r (m)', fontsize=11)
ax3.set_ylabel('Intensity (lumens/m²)', fontsize=11)
ax3.set_title('Inverse Square Law: Intensity vs Distance', 
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.set_xlim(0, 3.2)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("Visualization complete!")
print("=" * 60)