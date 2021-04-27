import numpy as np
from numpy.linalg import norm
from numpy import array
from numba import jit


class pointLocator(object):

    def __init__(
        self,
        mesh_triangles=None,
        mesh_vertices=None
    ):
        self.mesh_vertices = mesh_vertices
        self.mesh_triangles = mesh_triangles
        return

    # projects point onto a line
    # INPUTS
    # p is the 3d point being projected (numpy array with shape = (3))
    # a, b are the 3d line endpoints (numpy arrays with shape = (3))
    # OUTPUTS 
    # [0] t - the parametric coordinate between [0,1] of the projection (float)
    # [1] - the projected 3d point (numpy array with shape (3))

    def ProjectPointOntoLine(self, p, a, b):
        u = b - a
        u2 = u.dot(u)
        
        if u2 == 0: 
            return a.copy()

        x = p - a
        t = x.dot(u) / u2

        if t < 0:
            return 0, a.copy()
        elif t > 1.0:
            return 1, b.copy()
        else:
            return t, a + t * u

    # projects point onto a single triangle
    # INPUTS
    # p is the 3d point being projected (numpy array with shape = (3))
    # p1, p2, p3 are the 3d triangle vertices (numpy arrays with shape = (3))
    # OUTPUTS 
    # [0] xi - the barycentric coordinates on the triangle (numpy array with shape (3))
    # [1] proj - the projected 3d point (numpy array with shape (3))

    def ProjectPointOntoTriangle(self, p, triangle):
        p1, p2, p3 = self.get_pts(triangle)
        u = p2 - p1 # - 
        v = p3 - p1
        w = p3 - p2 # - 
        n = np.cross(u,v)
        n2 = n.dot(n)
        x = p - p1
        y = p - p2

        #
        # barycentric coordinates of the triangle
        gamma = np.cross(u,x).dot(n) / n2
        beta = np.cross(x,v).dot(n) / n2
        alpha = np.cross(w,y).dot(n) / n2

        # print(f"{alpha=}")
        # print(f"{beta=}")
        # print(f"{gamma=}")
        # print("===========")

        #
        # This is if the point already lies on the triangle
        if 0 <= gamma and gamma <= 1 and 0 <= beta and beta <= 1 and 0 <= alpha and alpha <= 1:
            xi = np.array([alpha, beta, gamma])
            return xi, alpha * p1 + beta * p2 + gamma * p3
        
        if gamma <= 0:
            t, proj = self.ProjectPointOntoLine(p, p1, p2)
            xi = np.array([1-t, t, 0])
        if beta <= 0:
            t, proj = self.ProjectPointOntoLine(p, p1, p3)
            xi = np.array([1-t, 0, t])
        if alpha <= 0:
            t, proj = self.ProjectPointOntoLine(p, p2, p3)
            xi = np.array([0, 1-t, t])

        return xi, proj


    # Projects a point onto a triangle mesh
    # INPUTS
    # verts - array of n 3d mesh vertices (numpy array with shape = (n, 3))
    # tris - array containing the vertex indices for m triangles (numpy array with shape = (m, 3))
    # pt - the point being projected onto the mesh
    # OUTPUTS
    # [0] closest_tri - the index of the triangle containing the projection (int)
    # [1] proj - the 3d point projection (numpy array with shape = (3))
    # [2] min_dist - the distance between pt and proj (float)

    def ProjectPointOntoTriMesh(self, verts, tris, pt):
        closest_tri = 0
        tmp_index = tris[0,0]
        min_dist = norm(
            pt - verts[tmp_index,:]
        )
        proj = (
            array([1,0,0]), 
            verts[tmp_index,:]
        )
        m = len(tris)
        for i in range(m):
            tri = tris[i]
            triproj = self.ProjectPointOntoTriangle(pt, tri)
            dist = norm(triproj[1] - pt)
            print(dist)
            if dist < min_dist:
                min_dist = dist
                closest_tri = i
                proj = triproj
        exit()
        return closest_tri, proj, min_dist

    def get_pts(self, triangle):
        return (
            self.mesh_vertices[triangle[0]],
            self.mesh_vertices[triangle[1]],
            self.mesh_vertices[triangle[2]]
        )