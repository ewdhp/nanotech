"""
CENTROID THEORY AND APPLICATIONS
=================================

MATHEMATICAL THEORY:
-------------------

1. DISCRETE POINTS (Point Mass System):
   For n points P_i = (x_i, y_i, z_i) with masses m_i:
   
   C = (C_x, C_y, C_z) where:
   C_x = Σ(m_i * x_i) / Σ(m_i)
   C_y = Σ(m_i * y_i) / Σ(m_i)
   C_z = Σ(m_i * z_i) / Σ(m_i)
   
   For uniform mass (m_i = 1):
   C = (1/n) * Σ P_i

2. CONTINUOUS REGIONS (Area/Volume):
   For a region R with density ρ:
   
   2D (Area):
   C_x = (1/A) ∬_R x * ρ(x,y) dA
   C_y = (1/A) ∬_R y * ρ(x,y) dA
   where A = ∬_R ρ(x,y) dA
   
   3D (Volume):
   C_x = (1/V) ∭_R x * ρ(x,y,z) dV
   C_y = (1/V) ∭_R y * ρ(x,y,z) dV
   C_z = (1/V) ∭_R z * ρ(x,y,z) dV
   where V = ∭_R ρ(x,y,z) dV

3. POLYGON CENTROID (Shoelace Formula):
   For a polygon with vertices (x_i, y_i):
   
   Area: A = (1/2) |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
   
   C_x = (1/(6A)) * Σ[(x_i + x_{i+1})(x_i * y_{i+1} - x_{i+1} * y_i)]
   C_y = (1/(6A)) * Σ[(y_i + y_{i+1})(x_i * y_{i+1} - x_{i+1} * y_i)]

4. COMPOSITE SHAPES:
   For composite shape with regions R_i having centroids C_i and areas/volumes A_i:
   
   C = Σ(A_i * C_i) / Σ(A_i)
   
   For subtraction (holes), use negative areas.

5. MOMENTS OF AREA:
   First moment of area about x-axis: M_x = ∬_R y dA
   First moment of area about y-axis: M_y = ∬_R x dA
   
   Then: C_x = M_y / A,  C_y = M_x / A

PHYSICAL SIGNIFICANCE:
---------------------
- Center of mass (with uniform density)
- Balance point (where object balances on a pin)
- Geometric center for symmetric objects
- Neutral axis in beam bending
- Center of pressure in fluid mechanics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CentroidCalculator:
    """Advanced centroid calculations for various geometric shapes."""
    
    @staticmethod
    def weighted_points(points, weights=None):
        """
        Calculate centroid of weighted points.
        
        Theory: C = Σ(w_i * P_i) / Σ(w_i)
        
        Args:
            points: Array of shape (n, d) where n is number of points, d is dimension
            weights: Array of shape (n,) with weights for each point
        
        Returns:
            Centroid coordinates
        """
        points = np.array(points)
        if weights is None:
            weights = np.ones(len(points))
        else:
            weights = np.array(weights)
        
        return np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)
    
    @staticmethod
    def polygon_2d(vertices):
        """
        Calculate centroid of a 2D polygon using the shoelace formula.
        
        Theory: Uses signed area method
        A = (1/2) Σ(x_i * y_{i+1} - x_{i+1} * y_i)
        C_x = (1/6A) Σ(x_i + x_{i+1})(x_i * y_{i+1} - x_{i+1} * y_i)
        C_y = (1/6A) Σ(y_i + y_{i+1})(x_i * y_{i+1} - x_{i+1} * y_i)
        
        Args:
            vertices: Array of shape (n, 2) with polygon vertices (counter-clockwise)
        
        Returns:
            (centroid, area)
        """
        vertices = np.array(vertices)
        n = len(vertices)
        
        # Close the polygon
        x = np.append(vertices[:, 0], vertices[0, 0])
        y = np.append(vertices[:, 1], vertices[0, 1])
        
        # Signed area using shoelace formula
        signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        
        # Centroid coordinates
        cx = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * signed_area)
        cy = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * signed_area)
        
        return np.array([cx, cy]), abs(signed_area)
    
    @staticmethod
    def triangle_3d(p1, p2, p3):
        """
        Calculate centroid of a 3D triangle.
        
        Theory: C = (P1 + P2 + P3) / 3
        
        Args:
            p1, p2, p3: 3D coordinates of triangle vertices
        
        Returns:
            (centroid, area)
        """
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        
        # Centroid is average of vertices
        centroid = (p1 + p2 + p3) / 3
        
        # Area using cross product
        v1 = p2 - p1
        v2 = p3 - p1
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        
        return centroid, area
    
    @staticmethod
    def tetrahedron(vertices):
        """
        Calculate centroid of a tetrahedron.
        
        Theory: C = (P1 + P2 + P3 + P4) / 4
        Volume: V = |det([P2-P1, P3-P1, P4-P1])| / 6
        
        Args:
            vertices: Array of shape (4, 3) with tetrahedron vertices
        
        Returns:
            (centroid, volume)
        """
        vertices = np.array(vertices)
        centroid = np.mean(vertices, axis=0)
        
        # Volume using determinant
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        v3 = vertices[3] - vertices[0]
        volume = abs(np.dot(v1, np.cross(v2, v3))) / 6
        
        return centroid, volume
    
    @staticmethod
    def composite_shapes(centroids, areas, subtract_mask=None):
        """
        Calculate centroid of composite shape.
        
        Theory: C = Σ(±A_i * C_i) / Σ(±A_i)
        Use negative area for holes/subtracted regions.
        
        Args:
            centroids: List of centroid coordinates for each region
            areas: List of areas/volumes for each region
            subtract_mask: Boolean array indicating which regions to subtract
        
        Returns:
            Composite centroid
        """
        centroids = np.array(centroids)
        areas = np.array(areas)
        
        if subtract_mask is not None:
            areas = areas * np.where(subtract_mask, -1, 1)
        
        return np.sum(centroids * areas[:, np.newaxis], axis=0) / np.sum(areas)


# ============================================================================
# EXAMPLE 1: Weighted Point Cloud
# ============================================================================
print("=" * 70)
print("EXAMPLE 1: Weighted Point Cloud (Molecular Structure)")
print("=" * 70)

# Simulate a molecule with different atomic masses
atoms = np.array([
    [0, 0, 0],      # Carbon (mass 12)
    [1, 0, 0],      # Hydrogen (mass 1)
    [0, 1, 0],      # Hydrogen (mass 1)
    [0, 0, 1],      # Hydrogen (mass 1)
    [-1, 0, 0],     # Oxygen (mass 16)
])
masses = np.array([12, 1, 1, 1, 16])

centroid_weighted = CentroidCalculator.weighted_points(atoms, masses)
centroid_uniform = CentroidCalculator.weighted_points(atoms)

print(f"Uniform centroid (geometric center): {centroid_uniform}")
print(f"Weighted centroid (center of mass):  {centroid_weighted}")
print(f"Difference due to mass distribution:  {np.linalg.norm(centroid_weighted - centroid_uniform):.4f}")

fig = plt.figure(figsize=(12, 5))

# Plot weighted
ax1 = fig.add_subplot(121, projection='3d')
colors = ['black', 'blue', 'blue', 'blue', 'red']
sizes = masses * 20
ax1.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], c=colors, s=sizes, alpha=0.6, label="Atoms")
ax1.scatter(*centroid_weighted, color='green', s=200, marker='*', 
            label="Center of Mass", edgecolors='black', linewidth=2)
ax1.set_xlabel("X (Å)")
ax1.set_ylabel("Y (Å)")
ax1.set_zlabel("Z (Å)")
ax1.set_title("Weighted Centroid (Center of Mass)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot uniform
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], c=colors, s=100, alpha=0.6, label="Atoms")
ax2.scatter(*centroid_uniform, color='orange', s=200, marker='*', 
            label="Geometric Center", edgecolors='black', linewidth=2)
ax2.set_xlabel("X (Å)")
ax2.set_ylabel("Y (Å)")
ax2.set_zlabel("Z (Å)")
ax2.set_title("Uniform Centroid (Geometric Center)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/centroid_weighted_points.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/centroid_weighted_points.png\n")


# ============================================================================
# EXAMPLE 2: Irregular Polygon (2D)
# ============================================================================
print("=" * 70)
print("EXAMPLE 2: Irregular Polygon (L-Shape)")
print("=" * 70)

# L-shaped polygon
l_shape = np.array([
    [0, 0],
    [3, 0],
    [3, 1],
    [1, 1],
    [1, 3],
    [0, 3]
])

centroid_poly, area_poly = CentroidCalculator.polygon_2d(l_shape)

print(f"Polygon area: {area_poly:.4f} square units")
print(f"Centroid: ({centroid_poly[0]:.4f}, {centroid_poly[1]:.4f})")
print("Note: Centroid is NOT at the simple average of vertices for non-uniform shapes")
print(f"Simple average: ({l_shape[:, 0].mean():.4f}, {l_shape[:, 1].mean():.4f})")

fig, ax = plt.subplots(figsize=(8, 8))

polygon = MplPolygon(l_shape, fill=True, alpha=0.3, color='skyblue', edgecolor='blue', linewidth=2)
ax.add_patch(polygon)

# Plot vertices
ax.scatter(l_shape[:, 0], l_shape[:, 1], color='blue', s=100, zorder=5, label="Vertices")

# Plot centroid
ax.scatter(*centroid_poly, color='red', s=300, marker='X', zorder=10, 
           label="True Centroid", edgecolors='black', linewidth=2)

# Plot simple average for comparison
simple_avg = l_shape.mean(axis=0)
ax.scatter(*simple_avg, color='orange', s=200, marker='o', zorder=10, 
           label="Simple Average", edgecolors='black', linewidth=2, alpha=0.7)

# Draw lines from centroid to show balance
for vertex in l_shape:
    ax.plot([centroid_poly[0], vertex[0]], [centroid_poly[1], vertex[1]], 
            'r--', alpha=0.2, linewidth=0.5)

ax.set_xlabel("X", fontsize=12)
ax.set_ylabel("Y", fontsize=12)
ax.set_title("L-Shape Polygon Centroid (Shoelace Formula)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)

plt.tight_layout()
plt.savefig('/tmp/centroid_polygon.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/centroid_polygon.png\n")


# ============================================================================
# EXAMPLE 3: Composite Shape (Rectangle with Hole)
# ============================================================================
print("=" * 70)
print("EXAMPLE 3: Composite Shape (Plate with Circular Hole)")
print("=" * 70)

# Outer rectangle
outer_rect_vertices = np.array([[0, 0], [10, 0], [10, 6], [0, 6]])
outer_centroid, outer_area = CentroidCalculator.polygon_2d(outer_rect_vertices)

# Inner circle (approximated as polygon)
circle_center = np.array([7, 3])
circle_radius = 1.5
n_circle_points = 50
theta = np.linspace(0, 2*np.pi, n_circle_points)
circle_vertices = circle_center + circle_radius * np.column_stack([np.cos(theta), np.sin(theta)])
circle_centroid, circle_area = CentroidCalculator.polygon_2d(circle_vertices)

# Composite centroid (subtracting the hole)
composite_centroid = CentroidCalculator.composite_shapes(
    centroids=[outer_centroid, circle_centroid],
    areas=[outer_area, circle_area],
    subtract_mask=[False, True]
)

final_area = outer_area - circle_area

print(f"Outer rectangle: Area = {outer_area:.4f}, Centroid = ({outer_centroid[0]:.4f}, {outer_centroid[1]:.4f})")
print(f"Inner circle: Area = {circle_area:.4f}, Centroid = ({circle_centroid[0]:.4f}, {circle_centroid[1]:.4f})")
print(f"Composite: Area = {final_area:.4f}, Centroid = ({composite_centroid[0]:.4f}, {composite_centroid[1]:.4f})")
print(f"Centroid shift from hole: {np.linalg.norm(composite_centroid - outer_centroid):.4f} units")

fig, ax = plt.subplots(figsize=(10, 6))

# Draw outer rectangle
outer_poly = MplPolygon(outer_rect_vertices, fill=True, alpha=0.4, 
                        color='lightblue', edgecolor='blue', linewidth=2)
ax.add_patch(outer_poly)

# Draw inner circle (hole)
circle_poly = MplPolygon(circle_vertices, fill=True, alpha=0.7, 
                         color='white', edgecolor='red', linewidth=2, linestyle='--')
ax.add_patch(circle_poly)

# Plot centroids
ax.scatter(*outer_centroid, color='blue', s=200, marker='o', 
           label="Rectangle Centroid", edgecolors='black', linewidth=2, zorder=10)
ax.scatter(*circle_centroid, color='red', s=200, marker='o', 
           label="Circle Centroid (removed)", edgecolors='black', linewidth=2, zorder=10)
ax.scatter(*composite_centroid, color='green', s=300, marker='X', 
           label="Composite Centroid", edgecolors='black', linewidth=2, zorder=10)

# Draw arrow showing shift
ax.annotate('', xy=composite_centroid, xytext=outer_centroid,
            arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
ax.text((outer_centroid[0] + composite_centroid[0])/2 - 0.5, 
        (outer_centroid[1] + composite_centroid[1])/2 + 0.3,
        'Shift', fontsize=10, color='purple')

ax.set_xlabel("X (mm)", fontsize=12)
ax.set_ylabel("Y (mm)", fontsize=12)
ax.set_title("Composite Shape: Plate with Hole", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 7)

plt.tight_layout()
plt.savefig('/tmp/centroid_composite.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/centroid_composite.png\n")


# ============================================================================
# EXAMPLE 4: 3D Tetrahedron
# ============================================================================
print("=" * 70)
print("EXAMPLE 4: 3D Tetrahedron")
print("=" * 70)

# Tetrahedron vertices
tetra_vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0.5, np.sqrt(3)/2, 0],
    [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
])

centroid_tetra, volume_tetra = CentroidCalculator.tetrahedron(tetra_vertices)

print(f"Tetrahedron volume: {volume_tetra:.6f} cubic units")
print(f"Centroid: ({centroid_tetra[0]:.6f}, {centroid_tetra[1]:.6f}, {centroid_tetra[2]:.6f})")
print("Theory: Centroid is at 1/4 of the height from base")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw tetrahedron edges
edges = [
    [0, 1], [0, 2], [0, 3],
    [1, 2], [1, 3], [2, 3]
]

for edge in edges:
    points = tetra_vertices[edge]
    ax.plot3D(*points.T, 'b-', linewidth=2)

# Draw faces
faces = [
    [tetra_vertices[0], tetra_vertices[1], tetra_vertices[2]],
    [tetra_vertices[0], tetra_vertices[1], tetra_vertices[3]],
    [tetra_vertices[0], tetra_vertices[2], tetra_vertices[3]],
    [tetra_vertices[1], tetra_vertices[2], tetra_vertices[3]]
]

poly3d = Poly3DCollection(faces, alpha=0.2, facecolor='cyan', edgecolor='blue', linewidth=2)
ax.add_collection3d(poly3d)

# Plot vertices
ax.scatter(tetra_vertices[:, 0], tetra_vertices[:, 1], tetra_vertices[:, 2], 
           color='blue', s=100, label="Vertices", edgecolors='black', linewidth=1)

# Plot centroid
ax.scatter(*centroid_tetra, color='red', s=300, marker='*', 
           label="Centroid", edgecolors='black', linewidth=2, zorder=10)

# Draw lines from centroid to vertices
for vertex in tetra_vertices:
    ax.plot3D([centroid_tetra[0], vertex[0]], 
              [centroid_tetra[1], vertex[1]], 
              [centroid_tetra[2], vertex[2]], 
              'r--', alpha=0.3, linewidth=1)

ax.set_xlabel("X", fontsize=12)
ax.set_ylabel("Y", fontsize=12)
ax.set_zlabel("Z", fontsize=12)
ax.set_title("Tetrahedron Centroid", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/centroid_tetrahedron.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/centroid_tetrahedron.png\n")


# ============================================================================
# EXAMPLE 5: Moment of Inertia and Parallel Axis Theorem
# ============================================================================
print("=" * 70)
print("EXAMPLE 5: Practical Application - Parallel Axis Theorem")
print("=" * 70)
print("""
The centroid is crucial for calculating moments of inertia:

Parallel Axis Theorem: I = I_c + M*d²

where:
- I = moment of inertia about any axis
- I_c = moment of inertia about axis through centroid
- M = total mass
- d = distance from centroid to the axis

This shows why finding the centroid is important in structural mechanics!
""")

# Create a simple beam cross-section (I-beam)
# Top flange
top_flange = np.array([[0, 8], [6, 8], [6, 10], [0, 10]])
top_c, top_a = CentroidCalculator.polygon_2d(top_flange)

# Web
web = np.array([[2.5, 2], [3.5, 2], [3.5, 8], [2.5, 8]])
web_c, web_a = CentroidCalculator.polygon_2d(web)

# Bottom flange
bottom_flange = np.array([[0, 0], [6, 0], [6, 2], [0, 2]])
bottom_c, bottom_a = CentroidCalculator.polygon_2d(bottom_flange)

# Overall centroid
beam_centroid = CentroidCalculator.composite_shapes(
    centroids=[top_c, web_c, bottom_c],
    areas=[top_a, web_a, bottom_a]
)

print(f"Top flange: Area = {top_a:.2f}, Centroid = ({top_c[0]:.2f}, {top_c[1]:.2f})")
print(f"Web:        Area = {web_a:.2f}, Centroid = ({web_c[0]:.2f}, {web_c[1]:.2f})")
print(f"Bottom flange: Area = {bottom_a:.2f}, Centroid = ({bottom_c[0]:.2f}, {bottom_c[1]:.2f})")
print(f"I-Beam:     Area = {top_a + web_a + bottom_a:.2f}, Centroid = ({beam_centroid[0]:.2f}, {beam_centroid[1]:.2f})")
print("\nNote: The neutral axis passes through this centroid!")

fig, ax = plt.subplots(figsize=(8, 10))

# Draw I-beam components
for vertices, color, label in [
    (top_flange, 'lightcoral', 'Top Flange'),
    (web, 'lightblue', 'Web'),
    (bottom_flange, 'lightgreen', 'Bottom Flange')
]:
    poly = MplPolygon(vertices, fill=True, alpha=0.5, 
                      edgecolor='black', linewidth=2, label=label)
    ax.add_patch(poly)

# Draw neutral axis (horizontal line through centroid)
ax.axhline(y=beam_centroid[1], color='red', linestyle='--', linewidth=2, 
           label='Neutral Axis', alpha=0.7)

# Plot component centroids
ax.scatter(*top_c, color='red', s=100, marker='o', alpha=0.5, edgecolors='black')
ax.scatter(*web_c, color='blue', s=100, marker='o', alpha=0.5, edgecolors='black')
ax.scatter(*bottom_c, color='green', s=100, marker='o', alpha=0.5, edgecolors='black')

# Plot overall centroid
ax.scatter(*beam_centroid, color='darkred', s=300, marker='X', 
           label='Beam Centroid', edgecolors='black', linewidth=2, zorder=10)

ax.set_xlabel("Width (cm)", fontsize=12)
ax.set_ylabel("Height (cm)", fontsize=12)
ax.set_title("I-Beam Cross-Section with Neutral Axis", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 11)

plt.tight_layout()
plt.savefig('/tmp/centroid_i_beam.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/centroid_i_beam.png\n")

print("=" * 70)
print("ALL EXAMPLES COMPLETED")
print("=" * 70)
print("\nKey Takeaways:")
print("1. Weighted centroids account for mass distribution")
print("2. Polygon centroids require the shoelace formula (not simple averaging)")
print("3. Composite shapes use area-weighted averaging")
print("4. 3D centroids extend these principles to volumes")
print("5. Centroids are essential for structural analysis (neutral axis, etc.)")
