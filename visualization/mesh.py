import re
import random
import math

class Point():
    def __init__(self, coords):
        self.xy = coords

def cross2D(v1, v2, relative_to = (0, 0)):
    v1 = tuple(v1[i] -relative_to[i] for i in range(2))
    v2 = tuple(v2[i] -relative_to[i] for i in range(2))

    return (v1[0] * v2[1]) - (v1[1] * v2[0])

class Mesh():
    def __init__(self, filename, perturb = False):
        with open(filename, "r") as grd_file:
            grd_str = grd_file.read()
        
        lines = grd_str.split("\n")
        lineno = 0

        def popline():
            nonlocal lineno
            lineno+=1
            return lines[lineno-1]
        
        

        match = re.match("npnt: (\d+) nseg: (\d+) ntri: (\d+)", popline())
        self.n_point, self.n_edge, self.n_tri = (int(x) for x in match.groups())

        perturb_amount = 1/4 * (2.0 / (math.sqrt(self.n_point)-1))
        def draw_perturb():
            nonlocal perturb_amount
            return perturb_amount * (2 * random.random() - 1)
        
        self.points = []
        for i in range(self.n_point):
            match = re.match(f"{i}: (\S+) (\S+)", popline())
            point = tuple(float(x) for x in match.groups())
            if perturb and (abs(point[0]) < 0.95) and (abs(point[1]) < 0.95):
                point = (point[0] +draw_perturb(), point[1] + draw_perturb())
            self.points.append(point)

        self.edges = []
        for i in range(self.n_edge):
            match = re.match(f"{i}: (\d+) (\d+)", popline())
            self.edges.append(tuple(int(x) for x in match.groups()))

        self.tris = []
        for i in range(self.n_tri):
            match = re.match(f"{i}: (\d+) (\d+) (\d+)", popline())
            tri_points = tuple(int(x) for x in match.groups())
            # sort tri_points anticlockwise for plt tri mesh reasons. Luckily
            # for a triangle swapping any two points swaps the ordering,
            # and we can check based on the cross product of the two vectors:
            if(cross2D(self.points[tri_points[1]], self.points[tri_points[2]], self.points[tri_points[0]]) < 0):
                tri_points = (tri_points[0], tri_points[2], tri_points[1])
            
            self.tris.append(tri_points)

if(__name__ == "__main__"):
    mesh = Mesh("../demoMeshes/Results2/square3_b0.grd")
