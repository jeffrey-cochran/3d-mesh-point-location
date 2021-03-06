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
#
# ====================
# EDGE ORDER
# [:, 0, :] = v
# [:, 1, :] = -u
# [:, 2, :] = -w
# ====================
edges = cp.roll(vertices,1,axis=1) - vertices             # (p3-p1, p1-p2, p2-p3)

#
# Correct edge signs and ordering
edges[:,1,:] = edges[:,1,:] * -1
edges[:,2,:] = edges[:,2,:] * -1
edges = cp.roll(edges, 2, axis=1)

#
# Compute normals and lengths
normals = cp.cross(edges[:,0], edges[:,1])
norms = cp.linalg.norm(normals, axis=1)
normssq = cp.square(norms)
#
# Match the dimensions for subtraction etc
#                                   REPEATED
# [triangle_index, vert_index, querypoint_index, coordinates]
vertices = cp.tile(
    cp.expand_dims(vertices, axis=2), 
    (1, 1, chunk_size, 1), 
)
#                                   REPEATED
# [triangle_index, edge_index, querypoint_index, coordinates]
# ===================
# [:, 0, :, :] = u = p2 - p1
# [:, 1, :, :] = v = p3 - p1
# [:, 2, :, :] = w = p3 - p2
edges = cp.tile(
    cp.expand_dims(edges, axis=2),
    (1, 1, chunk_size, 1)
)
edge_norms = cp.linalg.norm(edges,axis=3)
edge_normssq = cp.square(edge_norms)
#                   REPEATED        REPEATED
# [triangle_index, edge_index, querypoint_index, coordinates]
normals = cp.tile(
    cp.expand_dims(normals,axis=(1,2)), 
    (1, 3, chunk_size, 1)
)
#                   REPEATED        REPEATED
# [triangle_index, edge_index, querypoint_index]
norms = cp.tile(
    cp.expand_dims(norms,axis=(1,2)), 
    (1, 3, chunk_size)
)
#                   REPEATED        REPEATED
# [triangle_index, edge_index, querypoint_index]
normssq = cp.tile(
    cp.expand_dims(normssq,axis=(1,2)), 
    (1, 3, chunk_size)
)
#
# Zeros and ones for condition testing
z = cp.zeros((num_verts, 3, chunk_size))
o = cp.ones((num_verts, 3, chunk_size))

begin = time.time()

#
# Load and extend the batch
num_chunks = all_pts.shape[0] // chunk_size
for i in range(num_chunks):
    #
    # Get subset of the query points
    pts = all_pts[:chunk_size,:]

    #
    # Match the dimensions to those assumed above.
    #    REPEATED       REPEATED        
    # [triangle_index, vert_index, querypoint_index, coordinates]
    pts = cp.tile(
        cp.expand_dims(pts, axis=(0,1)),
        (num_verts, 3, 1, 1)
    )

    #
    # Compute the differences between
    # vertices on each triangle and the
    # points of interest
    #                   
    # [triangle_index, vert_index, querypoint_index, coordinates]
    # ===================
    # [:,0,:,:] = p - p1
    # [:,1,:,:] = p - p2
    # [:,2,:,:] = p - p3
    diff_vectors = pts - vertices


    #
    # Compute alpha, beta, gamma
    barycentric = cp.empty(
        diff_vectors.shape
    )

    #
    # gamma = u x (p - p1)
    barycentric[:,0,:,:] = cp.cross(
        edges[:,0,:,:], 
        diff_vectors[:,0,:,:]
    )
    # beta = (p - p1) x v
    barycentric[:,1,:,:] = cp.cross(
        diff_vectors[:,0,:,:],
        edges[:,1,:,:] 
    )
    # alpha = w x (p - p2)
    barycentric[:,2,:,:] = cp.cross(
        edges[:,2,:,:],
        diff_vectors[:,1,:,:]
    )
    barycentric = cp.divide(
            cp.sum(
                cp.multiply(
                    barycentric, 
                    normals
                ),
            axis=3
        ),
        normssq
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
        more_than_zero[:,0,:]
    )
    cond2 = cp.tile(
        cp.expand_dims(
            cond2,
            axis=1
        ),
        (1,3,1)
    )

    #
    #     if beta <= 0:
    cond3 = cp.logical_not(
        more_than_zero[:,1,:]
    )
    cond3 = cp.tile(
        cp.expand_dims(
            cond3,
            axis=1
        ),
        (1,3,1)
    )

    #
    #     if alpha <= 0:
    cond4 = cp.logical_not(
        more_than_zero[:,1,:]
    )
    cond4 = cp.tile(
        cp.expand_dims(
            cond4,
            axis=1
        ),
        (1,3,1)
    )

    #
    # Get the projections for each case
    xi = cp.empty(barycentric.shape)
    barycentric_ext = cp.tile(
        cp.expand_dims(
            barycentric,
            axis=3
        ),
        (1,1,1,3)
    )
    proj = cp.sum(
        cp.multiply(
            barycentric_ext,
            vertices
        ),
        axis=1
    )
    #
    #     if 0 <= gamma and gamma <= 1 
    #    and 0 <= beta and beta <= 1 
    #    and 0 <= alpha and alpha <= 1:
    xi[cond1] = barycentric[cond1]

    #
    # if gamma <= 0:
    #  x = p - p1
    #  u = p2 - p1
    #  a = p1
    #  b = p2
    t2 = cp.divide(
        #
        # u.dot(x)
        cp.sum(
                cp.multiply(
                    edges[:,0,:,:], 
                    diff_vectors[:,0,:,:]
                ),
            axis=2
        ),
        edge_normssq[:,0]
    )
    xi2 = cp.zeros((t2.shape[0], 3, t2.shape[1]))
    xi2[:,0,:] = -t2 + 1
    xi2[:,1,:] = t2
    #
    t2 = cp.tile(
        cp.expand_dims(
            t2,
            axis=2
        ),
        (1,1,3)
    )
    lz = cp.less(
        t2,
        cp.zeros(t2.shape)
    )
    go = cp.greater(
        t2,
        cp.ones(t2.shape)
    )
    proj2 = vertices[:,0,:,:] + cp.multiply(t2, edges[:,0,:,:])
    proj2[lz] = vertices[:,0,:,:][lz]
    proj2[go] = vertices[:,1,:,:][go]
    #
    xi[cond2] = xi2[cond2]
    proj[cp.swapaxes(cond2,1,2)] = proj2[cp.swapaxes(cond2,1,2)]

    #
    # if beta <= 0:
    #  x = p - p1
    #  v = p3 - p1
    #  a = p1
    #  b = p3
    t3 = cp.divide(
        #
        # v.dot(x)
        cp.sum(
                cp.multiply(
                    edges[:,1,:,:], 
                    diff_vectors[:,0,:,:]
                ),
            axis=2
        ),
        edge_normssq[:,1]
    )
    xi3 = cp.zeros((t3.shape[0], 3, t3.shape[1]))
    xi3[:,0,:] = -t3 + 1
    xi3[:,2,:] = t3
    #
    t3 = cp.tile(
        cp.expand_dims(
            t3,
            axis=2
        ),
        (1,1,3)
    )
    lz = cp.less(
        t3,
        cp.zeros(t3.shape)
    )
    go = cp.greater(
        t3,
        cp.ones(t3.shape)
    )
    proj3 = vertices[:,0,:,:] + cp.multiply(t3, edges[:,1,:,:])
    proj3[lz] = vertices[:,0,:,:][lz]
    proj3[go] = vertices[:,2,:,:][go]
    #
    xi[cond3] = xi3[cond3]
    proj[cp.swapaxes(cond3,1,2)] = proj3[cp.swapaxes(cond3,1,2)]

    #
    #     if alpha <= 0:
    #  y = p - p2
    #  w = p3 - p2
    #  a = p2
    #  b = p3
    t4 = cp.divide(
        #
        # w.dot(y)
        cp.sum(
                cp.multiply(
                    edges[:,2,:,:], 
                    diff_vectors[:,1,:,:]
                ),
            axis=2
        ),
        edge_normssq[:,2]
    )
    xi4 = cp.zeros((t4.shape[0], 3, t4.shape[1]))
    xi4[:,1,:] = -t4 + 1
    xi4[:,2,:] = t4
    #
    t4 = cp.tile(
        cp.expand_dims(
            t4,
            axis=2
        ),
        (1,1,3)
    )
    lz = cp.less(
        t4,
        cp.zeros(t4.shape)
    )
    go = cp.greater(
        t4,
        cp.ones(t4.shape)
    )
    proj4 = vertices[:,1,:,:] + cp.multiply(t4, edges[:,2,:,:])
    proj4[lz] = vertices[:,1,:,:][lz]
    proj4[go] = vertices[:,2,:,:][go]
    #
    xi[cond4] = xi4[cond4]
    proj[cp.swapaxes(cond4,1,2)] = proj4[cp.swapaxes(cond4,1,2)]

    distances = cp.linalg.norm(
        pts[:,0,:,:] - proj,
        axis=2
    )
    closest_triangles = cp.argmin(
        distances,
        axis=0
    )
    min_distances = cp.min(
        distances,
        axis=0
    )
    projections = proj[closest_triangles,np.arange(chunk_size),:]

total = time.time() - begin
newline = "\n"
print(f"Computed all {num_verts} alpha, beta, gamma values{newline}for each of the {chunk_size * num_chunks} query points")
print(f"Seconds Elapsed: {total}")

