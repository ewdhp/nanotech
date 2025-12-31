import numpy as np
import matplotlib.pyplot as plt

# Define 3D points
points = np.array([
    [1, 2, 3],
    [4, 1, 2],
    [2, 5, 4],
    [3, 3, 1]
])

# Calculate centroid
centroid = points.mean(axis=0)

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], label="Points")
ax.scatter(centroid[0], centroid[1], centroid[2],
           color='red', s=100, label="Centroid")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Centroid of 3D Points")
ax.legend()

plt.show()
