import trimesh
import pickle

import numpy as np

from sklearn.neighbors import KDTree

# for benchmarking
import time
from functools import wraps

MESH_PATH = "./data/datasets/scannetpp_mesh_norm_-1_1_0.95/00777c41d4/normalized.obj"
NUM_POINTS = 100000000
POINT_PATH = "./data/datasets/scannetpp_mesh_norm_-1_1_0.95/00777c41d4/points_{}.npz".format(NUM_POINTS)
KDTREE_PATH = "./data/datasets/scannetpp_mesh_norm_-1_1_0.95/00777c41d4/kdtree_{}.pkl".format(NUM_POINTS)

def timeit(func):
    """
    Decorator to measure the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Use perf_counter for more precise timing
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute.")
        return result
    return wrapper

@timeit
def load_mesh(mesh_path):
    return trimesh.load(mesh_path)

@timeit
def sample_surface_points(mesh, num_points):
    return mesh.sample(num_points, return_index=False)

@timeit
def save_points(points_path, points):
    np.savez_compressed(points_path, data=points)

@timeit
def load_points(points_path):
    return np.load(points_path)["data"]

@timeit
def create_kdtree(points):
    return KDTree(points)

@timeit
def save_kdtree(save_path, kdtree):
    with open(save_path, "wb") as f:
        pickle.dump(kdtree, f)

@timeit 
def load_kdtree(load_path):
    with open(load_path, "rb") as f:
        return pickle.load(f)

@timeit 
def query_distance(kdtree, query, k=1):
    return kdtree.query(query, k)

# mesh = load_mesh(MESH_PATH)
# mesh_points = sample_surface_points(mesh, NUM_POINTS)
# print(mesh_points.shape)
# save_points(POINT_PATH, mesh_points)

mesh_points_reload = load_points(POINT_PATH)
print(mesh_points_reload.shape)

kdtree = create_kdtree(mesh_points_reload)
save_kdtree(KDTREE_PATH, kdtree)

kdtree_reload = load_kdtree(KDTREE_PATH)

query = np.random.rand(200000, 3)
distance_A, _ = query_distance(kdtree, query)
distance_B, _ = query_distance(kdtree_reload, query)
print(np.array_equal(distance_A, distance_B))
