"""
Euler's Characteristic χ (chi)

A topological invariant that describes the shape/structure of a polyhedron or surface.

Euler's Polyhedron Formula (1750):
V - E + F = χ

Where:
- V = number of vertices
- E = number of edges
- F = number of faces
- χ = Euler characteristic

For Convex Polyhedra:
χ = 2 (Euler's formula: V - E + F = 2)

For Surfaces:
χ = 2 - 2g where g is the genus (number of holes)

Examples:
• Sphere: χ = 2 (g = 0)
• Torus: χ = 0 (g = 1)
• Double torus: χ = -2 (g = 2)
• Mobius strip: χ = 0
• Klein bottle: χ = 0

Platonic Solids:
• Tetrahedron: V=4, E=6, F=4 → χ=2
• Cube: V=8, E=12, F=6 → χ=2
• Octahedron: V=6, E=12, F=8 → χ=2
• Dodecahedron: V=20, E=30, F=12 → χ=2
• Icosahedron: V=12, E=30, F=20 → χ=2

Applications:
• Topology (classification of surfaces)
• Graph theory (planar graphs)
• Chemistry (molecular structures)
• Computer graphics (mesh validation)
• Differential geometry (Gauss-Bonnet theorem)
• Network analysis

Generalizations:
• Euler-Poincaré characteristic (simplicial complexes)
• Orbifold Euler characteristic
• Equivariant Euler characteristic

Author: Leonhard Euler (1707-1783)
Published: 1750, in a letter to Goldbach
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)


class Polyhedron:
    """Class to represent and analyze polyhedra"""
    
    def __init__(self, name, vertices, faces):
        """
        Initialize polyhedron
        
        Parameters:
        -----------
        name : str
            Name of the polyhedron
        vertices : array-like
            List of vertex coordinates
        faces : list of lists
            List of faces, each face is a list of vertex indices
        """
        self.name = name
        self.vertices = np.array(vertices)
        self.faces = faces
        
    def count_vertices(self):
        """Count number of vertices"""
        return len(self.vertices)
    
    def count_edges(self):
        """Count number of edges"""
        edges = set()
        for face in self.faces:
            n = len(face)
            for i in range(n):
                edge = tuple(sorted([face[i], face[(i+1) % n]]))
                edges.add(edge)
        return len(edges)
    
    def count_faces(self):
        """Count number of faces"""
        return len(self.faces)
    
    def euler_characteristic(self):
        """Compute Euler characteristic χ = V - E + F"""
        V = self.count_vertices()
        E = self.count_edges()
        F = self.count_faces()
        return V - E + F, V, E, F
    
    def get_stats(self):
        """Get complete statistics"""
        chi, V, E, F = self.euler_characteristic()
        return {
            'name': self.name,
            'vertices': V,
            'edges': E,
            'faces': F,
            'chi': chi
        }


def create_tetrahedron():
    """Create a regular tetrahedron"""
    vertices = [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]
    faces = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]
    return Polyhedron("Tetrahedron", vertices, faces)


def create_cube():
    """Create a cube"""
    vertices = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # top
    ]
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5]   # right
    ]
    return Polyhedron("Cube", vertices, faces)


def create_octahedron():
    """Create a regular octahedron"""
    vertices = [
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ]
    faces = [
        [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
        [1, 2, 4], [1, 4, 3], [1, 3, 5], [1, 5, 2]
    ]
    return Polyhedron("Octahedron", vertices, faces)


def create_dodecahedron():
    """Create a regular dodecahedron"""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    vertices = [
        # Cube vertices
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        # Rectangle vertices
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
    ]
    
    faces = [
        [0, 16, 17, 2, 12], [0, 12, 14, 4, 8], [0, 8, 9, 1, 16],
        [1, 9, 5, 15, 13], [1, 13, 3, 17, 16], [2, 17, 3, 11, 10],
        [2, 10, 6, 14, 12], [3, 13, 15, 7, 11], [4, 14, 6, 19, 18],
        [4, 18, 5, 9, 8], [5, 18, 19, 7, 15], [6, 10, 11, 7, 19]
    ]
    
    return Polyhedron("Dodecahedron", vertices, faces)


def create_icosahedron():
    """Create a regular icosahedron"""
    phi = (1 + np.sqrt(5)) / 2
    
    vertices = [
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    
    faces = [
        [0, 2, 8], [0, 8, 4], [0, 4, 6], [0, 6, 10], [0, 10, 2],
        [3, 1, 9], [3, 9, 5], [3, 5, 7], [3, 7, 11], [3, 11, 1],
        [2, 5, 8], [8, 9, 4], [4, 1, 6], [6, 11, 10], [10, 7, 2],
        [5, 2, 7], [8, 5, 9], [9, 1, 4], [1, 11, 6], [7, 10, 11]
    ]
    
    return Polyhedron("Icosahedron", vertices, faces)


def demonstrate_euler_formula():
    """
    Demonstrate Euler's formula for various polyhedra
    """
    print("=" * 80)
    print("EULER'S POLYHEDRON FORMULA: V - E + F = χ")
    print("=" * 80)
    
    print("\n1. PLATONIC SOLIDS (Regular Convex Polyhedra)")
    print("-" * 80)
    print(f"{'Polyhedron':15s}  {'V':>4s}  {'E':>4s}  {'F':>4s}  {'V-E+F':>6s}  {'χ':>3s}")
    print("-" * 80)
    
    platonic_solids = [
        create_tetrahedron(),
        create_cube(),
        create_octahedron(),
        create_dodecahedron(),
        create_icosahedron()
    ]
    
    for solid in platonic_solids:
        stats = solid.get_stats()
        print(f"{stats['name']:15s}  {stats['vertices']:4d}  {stats['edges']:4d}  "
              f"{stats['faces']:4d}  {stats['chi']:6d}  {stats['chi']:3d}")
    
    print("\n   Result: All convex polyhedra have χ = 2")
    
    print("\n2. OTHER POLYHEDRA")
    print("-" * 80)
    
    # Pyramid (square base)
    pyramid = Polyhedron("Pyramid", 
                        [[0,0,1], [1,1,0], [-1,1,0], [-1,-1,0], [1,-1,0]],
                        [[0,1,2], [0,2,3], [0,3,4], [0,4,1], [1,2,3,4]])
    
    # Prism (triangular)
    prism = Polyhedron("Triangular Prism",
                      [[0,1,0], [1,0,0], [-1,0,0], [0,1,1], [1,0,1], [-1,0,1]],
                      [[0,1,2], [3,4,5], [0,1,4,3], [1,2,5,4], [2,0,3,5]])
    
    for poly in [pyramid, prism]:
        stats = poly.get_stats()
        print(f"{stats['name']:15s}  {stats['vertices']:4d}  {stats['edges']:4d}  "
              f"{stats['faces']:4d}  {stats['chi']:6d}  {stats['chi']:3d}")


def demonstrate_topology():
    """
    Demonstrate topological interpretation
    """
    print("\n" + "=" * 80)
    print("TOPOLOGICAL INTERPRETATION: χ = 2 - 2g")
    print("=" * 80)
    
    print("\nFor closed surfaces:")
    print("   χ = 2 - 2g")
    print("   where g = genus (number of holes)")
    
    print("\n" + "-" * 80)
    print(f"{'Surface':20s}  {'Genus (g)':>10s}  {'χ = 2-2g':>10s}")
    print("-" * 80)
    
    surfaces = [
        ("Sphere", 0),
        ("Torus (1 hole)", 1),
        ("Double torus (2 holes)", 2),
        ("Triple torus (3 holes)", 3),
        ("Coffee mug", 1),
        ("Pretzel", 2),
    ]
    
    for name, genus in surfaces:
        chi = 2 - 2 * genus
        print(f"{name:20s}  {genus:10d}  {chi:10d}")
    
    print("\n3. NON-ORIENTABLE SURFACES")
    print("-" * 80)
    print(f"{'Surface':20s}  {'χ':>5s}  {'Note':>30s}")
    print("-" * 80)
    print(f"{'Möbius strip':20s}  {0:5d}  {'Non-orientable, 1 boundary'}")
    print(f"{'Klein bottle':20s}  {0:5d}  {'Non-orientable, no boundary'}")
    print(f"{'Real projective plane':20s}  {1:5d}  {'Non-orientable'}")


def demonstrate_graph_theory():
    """
    Demonstrate applications in graph theory
    """
    print("\n" + "=" * 80)
    print("GRAPH THEORY APPLICATION")
    print("=" * 80)
    
    print("\nFor planar graphs (graphs that can be drawn on a plane without")
    print("edge crossings), Euler's formula applies:")
    print("   V - E + F = 2")
    print("   where F includes the unbounded outer region")
    
    print("\n" + "-" * 80)
    print("Example: Complete graph K₄")
    print("-" * 80)
    print("   Vertices (V): 4")
    print("   Edges (E): 6 (each vertex connected to all others)")
    print("   Faces (F): 4 (3 triangular regions + 1 outer region)")
    print("   V - E + F = 4 - 6 + 4 = 2 ✓")
    
    print("\n" + "-" * 80)
    print("Non-planar graphs:")
    print("-" * 80)
    print("   K₅ (complete graph on 5 vertices) is non-planar")
    print("   K₃,₃ (complete bipartite graph) is non-planar")
    print("   These cannot be embedded in a plane without crossings")


def demonstrate_applications():
    """
    Show practical applications
    """
    print("\n" + "=" * 80)
    print("APPLICATIONS OF EULER CHARACTERISTIC")
    print("=" * 80)
    
    print("\n1. CHEMISTRY - Molecular Structure")
    print("-" * 80)
    print("   Fullerenes (carbon molecules):")
    print("   - C₆₀ (Buckminsterfullerene): 60 vertices, 90 edges, 32 faces")
    print(f"   - χ = 60 - 90 + 32 = {60 - 90 + 32}")
    print("   - Confirms sphere-like topology")
    
    print("\n2. COMPUTER GRAPHICS - Mesh Validation")
    print("-" * 80)
    print("   3D meshes should satisfy V - E + F = 2 for closed surfaces")
    print("   Used to detect holes, non-manifold edges, etc.")
    
    print("\n3. DIFFERENTIAL GEOMETRY - Gauss-Bonnet Theorem")
    print("-" * 80)
    print("   ∫∫ K dA = 2πχ")
    print("   where K is Gaussian curvature")
    print("   Connects local geometry (curvature) to global topology (χ)")
    
    print("\n4. NETWORK ANALYSIS")
    print("-" * 80)
    print("   Analyzing connectivity of transportation networks")
    print("   Circuit board design and layout")
    
    print("\n5. DATA ANALYSIS - Topological Data Analysis")
    print("-" * 80)
    print("   Persistent homology uses Euler characteristic")
    print("   Feature detection in high-dimensional data")


def create_visualizations():
    """
    Create comprehensive visualizations
    """
    fig = plt.figure(figsize=(18, 13))
    
    platonic = [
        create_tetrahedron(),
        create_cube(),
        create_octahedron(),
        create_dodecahedron(),
        create_icosahedron()
    ]
    
    # Plot all 5 Platonic solids
    for idx, solid in enumerate(platonic):
        ax = fig.add_subplot(3, 5, idx + 1, projection='3d')
        
        # Create face collection
        faces_collection = []
        for face in solid.faces:
            face_vertices = [solid.vertices[i] for i in face]
            faces_collection.append(face_vertices)
        
        # Plot faces
        poly_collection = Poly3DCollection(faces_collection, alpha=0.7, 
                                          facecolors='cyan', edgecolors='black', 
                                          linewidths=1.5)
        ax.add_collection3d(poly_collection)
        
        # Plot vertices
        ax.scatter(solid.vertices[:, 0], solid.vertices[:, 1], solid.vertices[:, 2],
                  c='red', s=50, alpha=0.8)
        
        # Set limits
        max_range = np.max(np.abs(solid.vertices))
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        stats = solid.get_stats()
        ax.set_title(f"{stats['name']}\nV={stats['vertices']}, E={stats['edges']}, "
                    f"F={stats['faces']}\nχ={stats['chi']}", fontsize=9)
        ax.set_axis_off()
    
    # 6. Euler characteristic vs genus
    ax6 = fig.add_subplot(3, 5, 6)
    genus_vals = np.arange(0, 6)
    chi_vals = 2 - 2 * genus_vals
    
    ax6.plot(genus_vals, chi_vals, 'bo-', linewidth=2, markersize=10)
    ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax6.axhline(y=2, color='g', linestyle='--', alpha=0.5, label='Sphere (g=0)')
    ax6.set_xlabel('Genus (g)', fontsize=11)
    ax6.set_ylabel('Euler Characteristic (χ)', fontsize=11)
    ax6.set_title('χ = 2 - 2g for Surfaces', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=9)
    
    # 7. Comparison table visualization
    ax7 = fig.add_subplot(3, 5, 7)
    ax7.axis('off')
    
    table_data = []
    for solid in platonic:
        stats = solid.get_stats()
        table_data.append([stats['name'], stats['vertices'], stats['edges'], 
                          stats['faces'], stats['chi']])
    
    table = ax7.table(cellText=table_data,
                     colLabels=['Polyhedron', 'V', 'E', 'F', 'χ'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color χ column
    for i in range(1, len(table_data) + 1):
        table[(i, 4)].set_facecolor('#90ee90')
    
    ax7.set_title('Platonic Solids Statistics', fontsize=12, fontweight='bold')
    
    # 8. Graph theory example
    ax8 = fig.add_subplot(3, 5, 8)
    
    # Draw K4 graph
    angle = np.linspace(0, 2*np.pi, 5)[:-1]
    x_nodes = np.cos(angle)
    y_nodes = np.sin(angle)
    
    # Draw all edges
    for i in range(4):
        for j in range(i+1, 4):
            ax8.plot([x_nodes[i], x_nodes[j]], [y_nodes[i], y_nodes[j]], 
                    'b-', linewidth=1.5, alpha=0.6)
    
    # Draw nodes
    ax8.plot(x_nodes, y_nodes, 'ro', markersize=15)
    
    # Label nodes
    for i in range(4):
        offset = 1.2
        ax8.text(x_nodes[i]*offset, y_nodes[i]*offset, f'{i+1}', 
                ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax8.set_xlim([-1.5, 1.5])
    ax8.set_ylim([-1.5, 1.5])
    ax8.set_aspect('equal')
    ax8.axis('off')
    ax8.set_title('K₄ Graph\nV=4, E=6, F=4, χ=2', fontsize=11, fontweight='bold')
    
    # 9. Topology examples
    ax9 = fig.add_subplot(3, 5, 9)
    ax9.axis('off')
    
    topology_text = """
SURFACE TOPOLOGY
═══════════════════

Surface          g    χ
────────────────────────
Sphere           0    2
Torus            1    0
Double Torus     2   -2
Triple Torus     3   -4

Formula: χ = 2 - 2g

Non-orientable:
────────────────────────
Möbius strip          0
Klein bottle          0
Projective plane      1
    """
    
    ax9.text(0.05, 0.95, topology_text, transform=ax9.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 10. Summary
    ax10 = fig.add_subplot(3, 5, 10)
    ax10.axis('off')
    
    summary = """
EULER CHARACTERISTIC
════════════════════════

Polyhedron Formula:
  V - E + F = χ

Convex Polyhedra:
  χ = 2

Surfaces:
  χ = 2 - 2g
  (g = genus)

Applications:
────────────────────────
• Topology
• Graph theory
• Chemistry
• Computer graphics
• Differential geometry

Key Insight:
────────────────────────
χ is a topological
invariant - it doesn't
change under continuous
deformations!

A coffee mug and a
torus have the same
χ = 0 (both g = 1)
    """
    
    ax10.text(0.05, 0.95, summary, transform=ax10.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 11-15: Additional visualizations showing edges and structure
    for idx in range(5):
        ax = fig.add_subplot(3, 5, 11 + idx)
        solid = platonic[idx]
        
        # Simple 2D projection of edges
        vertices_2d = solid.vertices[:, :2]  # Project to XY plane
        
        # Draw edges
        for face in solid.faces:
            n = len(face)
            for i in range(n):
                v1 = vertices_2d[face[i]]
                v2 = vertices_2d[face[(i+1) % n]]
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'b-', alpha=0.3, linewidth=1)
        
        # Draw vertices
        ax.scatter(vertices_2d[:, 0], vertices_2d[:, 1], c='red', s=30, zorder=5)
        
        ax.set_aspect('equal')
        ax.axis('off')
        stats = solid.get_stats()
        ax.set_title(f"{stats['name']} (2D projection)", fontsize=9)
    
    plt.suptitle("Euler's Characteristic: V - E + F = χ", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def main():
    """
    Main demonstration
    """
    print("\n" + "=" * 80)
    print(" EULER'S CHARACTERISTIC χ")
    print(" Euler's Polyhedron Formula: V - E + F = 2")
    print("=" * 80)
    
    # Demonstrate formula
    demonstrate_euler_formula()
    
    # Topology
    demonstrate_topology()
    
    # Graph theory
    demonstrate_graph_theory()
    
    # Applications
    demonstrate_applications()
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    
    # Create visualizations
    fig = create_visualizations()
    
    print("\n✓ Visualizations created successfully!")
    print("\nKey Insights:")
    print("  • Euler characteristic is a topological invariant")
    print("  • All convex polyhedra have χ = 2")
    print("  • χ = 2 - 2g for closed orientable surfaces (g = genus)")
    print("  • Fundamental in topology, graph theory, and geometry")
    print("  • Shows deep connection between combinatorics and topology")
    
    plt.show()


if __name__ == "__main__":
    main()
