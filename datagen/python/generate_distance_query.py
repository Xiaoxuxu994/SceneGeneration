import os
import shutil
import argparse
import trimesh

import numpy as np
import open3d as o3d
import multiprocessing as mp

from pathlib import Path
from skimage import measure

MESH_FILE = "normalized.obj"
QUERY_FILE = "queries_{}" # number of query points
QUERY_FILE_SUFFIX = ".npz"

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)

parser.add_argument("--num_queries", type=int, default=10000000)
parser.add_argument("--chunk_size", type=int, default=2000000)
parser.add_argument("--boxmin", type=float, default=-1)
parser.add_argument("--boxmax", type=float, default=1)

parser.add_argument("--fast", action="store_true", help="use the faster but less accurate distance implementation.")

parser.add_argument("--force", action="store_true", help="generate octree even if target exists.")
parser.add_argument("--num_workers", type=int, default=1)
args = parser.parse_args()

assert args.num_queries >= args.chunk_size, "number of queries must be greater than the chunk size, \
    got {} and {}".format(args.num_queries, args.chunk_size)

in_dir = Path(args.in_dir)
out_dir = Path(args.out_dir)

out_dir.mkdir(exist_ok=True, parents=True)

def create_query_coords(num_queries):
    normalized_queries = np.random.rand(num_queries, 3)
    queries = normalized_queries * (args.boxmax - args.boxmin) + args.boxmin
    num_chunks = int(np.ceil(args.num_queries / args.chunk_size))
    query_chunks = np.array_split(queries, num_chunks)
    print("splitting {} query points to {} queries in {} chunks".format(
        num_queries, query_chunks[0].shape[0], num_chunks))
    
    return queries, query_chunks

def compute_angle(A, B):
    dot_product = (A*B).sum(1)
    magnitude_A = np.linalg.norm(A, axis=1)
    magnitude_B = np.linalg.norm(B, axis=1)
    cosine_angle = dot_product / (magnitude_A * magnitude_B + 1e-16)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def compute_grad(query_coords, mesh):
    print("computing gradients for {} query points".format(query_coords.shape[0]))

    if args.fast:
        pass
    else:
        mesh_o3d = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        )
        mesh_o3d.compute_triangle_normals()
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d))
        
        query_point = o3d.core.Tensor.from_numpy(query_coords.astype(np.float32))
        query_grads = scene.compute_distance(query_point)
    
    return query_grads

def worker_job(input_dir):
    mesh_path = input_dir / MESH_FILE
    
    if not mesh_path.exists():
        print("{} doesn't exist.".format(str(mesh_path)))
    
    output_dir = out_dir / input_dir.name
    output_dir.mkdir(exist_ok=True, parents=True)
    output_name = QUERY_FILE.format(args.num_queries)
    # if args.grad: output_name += "_grad"
    output_path = output_dir / output_name 
    output_path = output_path.with_suffix(QUERY_FILE_SUFFIX)
    
    if output_path.exists() and not args.force:
        print("found existing query file {}, skipping...".format(str(output_path)))
    else:
        print("processing {}...".format(str(input_dir)))
        
        print("loading mesh {}".format(mesh_path))
        mesh = trimesh.load_mesh(mesh_path, validate=False, process=False) # obj file
        query_coords, chunk_query_coords = create_query_coords(args.num_queries)  
        
        list_queries, list_distances = [], []
        for cur_query_coords in chunk_query_coords:
            distances = compute_grad(cur_query_coords, mesh)
            
            distances = distances.numpy().tolist()
            list_distances += distances
        
        array_queries = query_coords.astype(np.float16)
        array_distances = np.array(list_distances).astype(np.float16)
        assert array_queries.shape[0] == array_distances.shape[0]
        
        np.savez_compressed(output_path, queries=array_queries, distances=array_distances)

if __name__ == "__main__":
    in_dir_list = list(sorted(in_dir.iterdir()))
    assert len(in_dir_list) > 0, "empty input data list"
    
    if args.num_workers > 1:
        pool = mp.Pool(processes=args.num_workers)
        pool.map(worker_job, in_dir_list)
    else:
        for in_dir in in_dir_list:
            # for debugging
            # in_dir = Path("data/nvme2/scannetpp_volume_norm_-1_1_0.95/sdf/836363121f")
            worker_job(in_dir)
            # exit()
