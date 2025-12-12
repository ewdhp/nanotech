import numpy as np
import plotly.graph_objects as go

def fibonacci_sphere(num_points: int, radius: float = 1.0):
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    points = []
    for i in range(num_points):
        y = 1 - 2 * (i + 0.5) / num_points
        r = np.sqrt(1 - y * y)
        theta = golden_angle * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append(radius * np.array([x, y, z]))
    return np.array(points)

# Parameters
N = 500
R = 1.0
coords = fibonacci_sphere(N, R)

# Plotly 3D scatter plot
fig = go.Figure(
    data=[go.Scatter3d(
        x=coords[:,0], y=coords[:,1], z=coords[:,2],
        mode='markers',
        marker=dict(size=3, color=coords[:,2], colorscale='Viridis', opacity=0.8)
    )]
)
fig.update_layout(
    title=f'Fibonacci Lattice on a Sphere (N={N})',
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)
fig.show()