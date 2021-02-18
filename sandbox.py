import numpy as np
import cupy as cp
import time
from utils import load_mesh, load_query_points

#
# Load data
tris, verts = load_mesh('test_mesh')
all_pts = load_query_points('test')
vertices = verts[tris[:,:]]

#
# Fix common dimensions
num_verts = vertices.shape[0]
chunk_size = 100

#
# Compute static information about 
# the triangles
edges = cp.roll(vertices,1,axis=2) - vertices             # (p3-p1, p1-p2, p2-p3)
normals = cp.cross(edges[:,0], edges[:,1])
norms = cp.linalg.norm(normals, axis=1)
normssq = cp.square(norms)

#
# Match the dimensions for subtraction etc
vertices = cp.tile(vertices, (1, chunk_size, 1))
edges = cp.tile(edges, (1, chunk_size, 1))
normals = cp.tile(cp.expand_dims(normals,axis=1), (1, chunk_size*3, 1))
norms = cp.tile(cp.expand_dims(norms,axis=1), (1, chunk_size*3))
normssq = cp.tile(cp.expand_dims(normssq,axis=1), (1, chunk_size*3))

#
# Zeros and ones for condition testing
z = cp.zeros((num_verts, chunk_size, 3))
o = cp.ones((num_verts, chunk_size, 3))

begin = time.time()
#
# Load and extend the batch
num_chunks = all_pts[0] // chunk_size
pts = all_pts[:chunk_size,:]
pts = cp.tile(pts,(num_verts, 3, 1))

#
# Compute the differences between
# vertices on each triangle and the
# points of interest
diff_vectors = vertices - pts

# barycentric_ext = cp.empty(vertices_ext.shape)
# gamma
barycentric = cp.cross(edges, diff_vectors)
barycentric = cp.divide(
        cp.sum(
            cp.multiply(
                barycentric, 
                normals
            ),
        axis=2
    ),
    normssq
)
barycentric = cp.reshape(
    barycentric,
    (
        num_verts,
        chunk_size,
        3
    )
)

#
# Test conditions
less_than_one = cp.less_equal(
    barycentric,
    o
)
more_than_zero = cp.greater_equal(
    barycentric,
    z
)

#
#     if 0 <= gamma and gamma <= 1 
#    and 0 <= beta and beta <= 1 
#    and 0 <= alpha and alpha <= 1:
cond1 = cp.logical_and(
    less_than_one,
    more_than_zero
)

#
#     if gamma <= 0:
cond2 = cp.logical_not(
    less_than_one[:,:,0]
)

#
#     if beta <= 0:
cond3 = cp.logical_not(
    less_than_one[:,:,1]
)

#
#     if alpha <= 0:
cond4 = cp.logical_not(
    less_than_one[:,:,2]
)

total = time.time() - begin
newline = "\n"
print(f"Computed all {num_verts} alpha, beta, gamma values{newline}for each of the {chunk_size} query points")
print(f"Seconds Elapsed: {total}")

# def ProjectPointOntoLine(p, a, b):
#     u = b - a
#     u2 = cp.dot(u, u)
    
#     if u2 == 0: 
#         return a.copy()

#     x = p - a
#     t = cp.dot(x, u) / u2

#     if t < 0:
#         return 0, a.copy()
#     elif t > 1.0:
#         return 1, b.copy()
#     else:
#         return t, a + t * u


#     #
#     # barycentric coordinates of the triangle
#     gamma = cp.dot(cp.cross(u,x), n) / n2
#     beta = cp.dot(cp.cross(x,v), n) / n2
#     alpha = cp.dot(cp.cross(w,y), n) / n2

#     #
#     # This is if the point already lies on the triangle
#         xi[0] = alpha
#         xi[1] = beta
#         xi[2] = gamma
#         return xi, alpha * p1 + beta * p2 + gamma * p3
    

#         t, proj = ProjectPointOntoLine(p, p1, p2)
#         xi[0] = 1-t
#         xi[1] = t
#         xi[2] = 0

#         t, proj = ProjectPointOntoLine(p, p1, p3)
#         xi[0] = 1-t
#         xi[1] = 0
#         xi[2] = t
#         t, proj = ProjectPointOntoLine(p, p2, p3)
#         xi[0] = 0
#         xi[1] = 1-t
#         xi[2] = t

#     return xi, proj


# def ProjectPointOntoTriMesh(verts, tris, pt):
#     closest_tri = 0
#     tmp_index = tris[0,0]
#     min_dist = cp.linalg.norm(
#         pt - verts[tmp_index,:]
#     )
#     proj = (
#         cp.asarray([1,0,0]), 
#         verts[tmp_index,:]
#     )
#     m = len(tris)
#     for i in range(m):
#         tri = tris[i]
#         triproj = ProjectPointOntoTriangle(pt, tri, verts)
#         dist = cp.linalg.norm(triproj[1] - pt)
#         if dist < min_dist:
#             min_dist = dist
#             closest_tri = i
#             proj = triproj
#     return closest_tri, proj, min_dist

# def get_pts(vertices, triangle):
#     return (
#         vertices[triangle[0]],
#         vertices[triangle[1]],
#         vertices[triangle[2]]
#     )

# for i in range(pts.shape[0]):
#     ProjectPointOntoTriMesh(verts, tris, pts[i,:])
#     print(f"We're moving... {i}")