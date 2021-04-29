import numpy as np
import cupy as cp
import time
from utils import load_mesh, load_query_points

def get_expanded_tensors(
    vertices: cp.ndarray=None,
    edges: cp.ndarray=None,
    normals: cp.ndarray=None,
    norms: cp.ndarray=None,
    normssq: cp.ndarray=None,
    chunk_size: int=None,
    num_verts: int=None
) -> dict:
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
    #
    return {
        "vertices": vertices,
        "edges": edges,
        "edge_norms": edge_norms,
        "edge_normssq": edge_normssq,
        "normals": normals,
        "norms": norms,
        "normssq": normssq,
        "zero_tensor": z,
        "one_tensor": o,
        "chunk_size": chunk_size,
        "num_verts": num_verts
    }

def evaluate_chunks(
    results: [cp.ndarray, cp.ndarray, cp.ndarray], # closest triangle, distance, projection
    all_pts: cp.ndarray=None,
    vertices: cp.ndarray=None,
    edges: cp.ndarray=None,
    edge_norms: cp.ndarray=None,
    edge_normssq: cp.ndarray=None,
    normals: cp.ndarray=None,
    norms: cp.ndarray=None,
    normssq: cp.ndarray=None,
    zero_tensor: cp.ndarray=None,
    one_tensor: cp.ndarray=None,
    tris: cp.ndarray=None,
    vertex_normals: cp.ndarray=None,
    bounding_box: dict=None,
    chunk_size: int=None,
    num_verts: int=None
) -> None:

    #
    # Expand vertex normals if non empty
    if vertex_normals is not None:
        vertex_normals = vertex_normals[tris]
        vertex_normals = cp.tile(
            cp.expand_dims(vertex_normals, axis=2),
            (1, 1, chunk_size, 1)
        )

    # begin = time.time()
    #
    # Load and extend the batch
    num_chunks = all_pts.shape[0] // chunk_size
    for i in range(num_chunks):
        #
        # Get subset of the query points
        start_index = i * chunk_size
        end_index = (i+1) * chunk_size
        pts = all_pts[start_index:end_index, :]

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
        barycentric[:,2,:,:] = cp.cross(
            edges[:,0,:,:], 
            diff_vectors[:,0,:,:]
        )
        # beta = (p - p1) x v
        barycentric[:,1,:,:] = cp.cross(
            diff_vectors[:,0,:,:],
            edges[:,1,:,:] 
        )
        # alpha = w x (p - p2)
        barycentric[:,0,:,:] = cp.cross(
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
            one_tensor
        )
        more_than_zero = cp.greater_equal(
            barycentric,
            zero_tensor
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
            more_than_zero[:,2,:]
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
            more_than_zero[:,0,:]
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

        vec_to_point = pts[:,0,:,:] - proj
        distances = cp.linalg.norm(
            vec_to_point,
            axis=2
        )



        # n = "\n"
        # print(f"{pts[:,0,:,:]=}")
        # print(f"{proj=}")
        # print(f"{pts[:,0,:,:] - proj=}")
        # print(f"{distances=}")

        min_distances = cp.min(
            distances,
            axis=0
        )

        closest_triangles = cp.argmin(
            distances,
            axis=0
        )
 
        projections = proj[closest_triangles,np.arange(chunk_size),:]

        #
        # Distinguish close triangles
        is_close = cp.isclose(distances, min_distances)
        
        #
        # Determine sign
        signed_normal = normals[:, 0, :, :]
        if vertex_normals is not None:
            signed_normal = cp.sum(
                vertex_normals.transpose() * xi.transpose(),
                axis=2
            ).transpose()
        
        is_negative = cp.less_equal(
                cp.sum(
                    cp.multiply(
                        vec_to_point, 
                        signed_normal
                    ),
                axis=2
            ),
            0.
        )

        #
        # Combine
        is_close_and_negative = cp.logical_and(
            is_close,
            is_negative
        )

        #
        # Determine if inside
        is_inside = cp.all(
            cp.logical_or(
                is_close_and_negative,
                cp.logical_not(is_close)
            ),
            axis=0
        )

        #
        # Overwrite the signs of points
        # that are outside of the box
        if bounding_box is not None:
            #
            # Extract
            rotation_matrix = cp.asarray(bounding_box['rotation_matrix'])
            translation_vector = cp.asarray(bounding_box['translation_vector'])
            size = cp.asarray(bounding_box['size'])
            #
            # Transform
            transformed_pts = cp.dot(
                all_pts[start_index:end_index, :] - translation_vector, 
                rotation_matrix
            )

            #
            # Determine if outside bbox
            inside_bbox = cp.all(
                cp.logical_and(
                    cp.less_equal(0., transformed_pts),
                    cp.less_equal(transformed_pts, size)
                ),
                axis=1
            )

            #
            # Treat points outside bbox as
            # being outside of lumen
            print(f"{inside_bbox=}")
            is_inside = cp.logical_and(is_inside, inside_bbox)
            
        #
        # Apply sign to indicate whether the distance is 
        # inside or outside the mesh.
        min_distances[is_inside] = -1 * min_distances[is_inside]

        #
        # Emplace results
        # [triangle_index, vert_index, querypoint_index, coordinates]
        results[0][start_index:end_index] = closest_triangles
        results[1][start_index:end_index] = min_distances
        results[2][start_index:end_index,:] = projections

def _locate_points(
    all_pts: cp.ndarray=None,
    vertices: cp.ndarray=None,
    edges: cp.ndarray=None,
    normals: cp.ndarray=None,
    norms: cp.ndarray=None,
    normssq: cp.ndarray=None,
    chunk_size: int=None,
    num_verts: int=None,
    tris: cp.ndarray=None,
    vertex_normals: cp.ndarray=None,
    bounding_box: dict=None
):
    #
    # Instatiate results
    num_pts = all_pts.shape[0]
    num_chunks = num_pts // chunk_size
    fake_num_pts = chunk_size * (num_chunks + 1)
    extension_size = fake_num_pts - num_pts
    all_pts = cp.concatenate(
        (
            all_pts, 
            cp.zeros((
                extension_size,
                3
            ))
        ),
        axis=0
    )
    results = [
        cp.zeros((fake_num_pts,)), 
        cp.zeros((fake_num_pts,)), 
        cp.zeros((fake_num_pts,3))
    ]
    #
    # Iterate over as many full-sized
    # chunks as is possible
    d = get_expanded_tensors(
        vertices=vertices,
        edges=edges,
        normals=normals,
        norms=norms,
        normssq=normssq,
        chunk_size=chunk_size,
        num_verts=num_verts
    )
    evaluate_chunks(
        results, # closest triangle, distance, projection
        all_pts=all_pts,
        tris=tris,
        vertex_normals=vertex_normals,
        bounding_box=bounding_box,
        **d
    )
    #
    # Remove extraneous points

    # Closest triangle
    results[0] = results[0][:num_pts] 

    # Distance
    results[1] = results[1][:num_pts]

    # Projection
    results[2] = results[2][:num_pts, :]

    #
    return results

def locate_points(
    mesh_prefix: str=None,
    pts_prefix: str=None,
    chunk_size: int=None,
    bounding_box: dict=None
) -> [cp.ndarray, cp.ndarray, cp.ndarray]:

    #
    # Load data
    tris, verts, vert_normals = load_mesh(mesh_prefix)
    all_pts = load_query_points(pts_prefix)
    vertices = verts[tris[:,:]]
    #
    # Fix common dimensions
    num_verts = vertices.shape[0]
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
    tmp = edges[:,1,:].copy()
    edges[:,1,:] = edges[:,2,:] 
    edges[:,2,:] = tmp
    #
    # Compute normals and lengths
    normals = cp.cross(edges[:,0], edges[:,1])
    norms = cp.linalg.norm(normals, axis=1)
    normssq = cp.square(norms)
    #
    return _locate_points(
        all_pts=all_pts,
        vertices=vertices,
        edges=edges,
        normals=normals,
        norms=norms,
        normssq=normssq,
        chunk_size=chunk_size,
        num_verts=num_verts,
        tris=tris,
        vertex_normals=vert_normals,
        bounding_box=bounding_box
    )