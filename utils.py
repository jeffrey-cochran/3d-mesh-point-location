from os.path import join
from constants import mesh_dir, query_dir
from numpy import loadtxt
from cupy import asarray

def load_cuda_array_from_txt(
    file_name,
    **kwargs
):
    return asarray(
        loadtxt(
            file_name,
            **kwargs
        )
    )

def load_query_points(query_name):
    return load_cuda_array_from_txt(
        join(query_dir, query_name+"_points.csv"),
        delimiter=','
    )

def load_mesh(mesh_name):
    return (
        load_cuda_array_from_txt(
            join(mesh_dir, mesh_name+"_triangles.csv"),
            dtype=int, 
            delimiter=','
        ),
        load_cuda_array_from_txt(
            join(mesh_dir, mesh_name+"_verts.csv"),
            delimiter=','
        ),
    )
