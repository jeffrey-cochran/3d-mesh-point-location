from pointLocator import pointLocator
from utils import load_mesh_np, load_query_points_np

tris, verts =  load_mesh_np("test_tetra")
all_pts = load_query_points_np("test_tetra_distance")

pl = pointLocator(
    mesh_triangles=tris,
    mesh_vertices=verts
)

for pt in all_pts:
    ret = pl.ProjectPointOntoTriMesh(
        verts, 
        tris, 
        pt
    )

    print(f"{ret=}")
