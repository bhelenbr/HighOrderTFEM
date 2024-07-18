import matplotlib.pyplot as plt
from matplotlib.collections import TriMesh, PolyCollection
from matplotlib.tri.triangulation import Triangulation
from matplotlib.colors import ListedColormap
import seaborn as sns
import random
from mesh import Mesh
sns.set_theme()

mesh = Mesh("../demoMeshes/Results2/square3_b0.grd", True)

point_adjacency = [[] for _ in range(mesh.n_point)]

colors = [-1 for _ in range(mesh.n_tri)]
for i, tri in enumerate(mesh.tris):
    for p in tri:
        point_adjacency[p].append(i)

visit_order = list(range(mesh.n_tri))
random.shuffle(visit_order)
unique_colors = 0
for tri_ind in visit_order:
    c = -1
    unique = False
    while(unique is False):
        c += 1
        unique = True
        for p in mesh.tris[tri_ind]:
            for adj_t_ind in point_adjacency[p]:
                if c == colors[adj_t_ind]:
                    unique = False
                    break
    colors[tri_ind] = c
    unique_colors = max(c + 1, unique_colors)

# Colored
cmap = sns.color_palette("muted", n_colors= unique_colors)
print(unique_colors)
plt_colors = [cmap[i] for i in colors]
plt_mesh = PolyCollection([tuple(mesh.points[i] for i in mesh.tris[j]) for j in range(mesh.n_tri)], facecolors = plt_colors, edgecolors = "black")

plt.figure()
plt.gca().add_collection(plt_mesh)
plt.gca().set_aspect("equal")
plt.draw()
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.title("Vertex-Adjacent Colored Mesh")
plt.savefig("mesh_colored.png")
plt.close()

# Uncolored
plt_mesh = PolyCollection([tuple(mesh.points[i] for i in mesh.tris[j]) for j in range(mesh.n_tri)], facecolors = (0.0, 0.0, 0.0, 0.0), edgecolors =( "black"))

plt.figure()
plt.gca().add_collection(plt_mesh)
plt.gca().set_aspect("equal")
plt.draw()
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.title("Mesh")
plt.savefig("mesh_uncolored.png")
plt.close()
