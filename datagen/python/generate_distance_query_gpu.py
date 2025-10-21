import os
import sys
import argparse

import torch
import pytorch3d

import numpy as np

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Pointclouds

from pathlib import Path

sys.path.append(".")
from lib.pytorch3d.point_mesh_distance import point_mesh_face_distance

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
parser.add_argument("--devices", type=str, default='0', help="split the devices with comma, e.g. 0,1,2,3")

args = parser.parse_args()

assert args.num_queries >= args.chunk_size, "number of queries must be greater than the chunk size, \
    got {} and {}".format(args.num_queries, args.chunk_size)

in_dir = Path(args.in_dir)
out_dir = Path(args.out_dir)

out_dir.mkdir(exist_ok=True, parents=True)

def map_jobs_to_devices(job_list):
    if len(args.devices) == 1:
        device_list = [args.devices]
    else:
        device_list = args.devices.split(',')
        
    device_list = [torch.device("cuda:{}".format(e)) for e in device_list]
    
    mapping_list = []
    for idx, job in enumerate(job_list):
        cur_idx = idx % len(device_list)
        mapping_list.append((job, device_list[cur_idx]))
        
    return mapping_list

def create_query_coords(num_queries, device):
    normalized_queries = np.random.rand(num_queries, 3)
    queries = normalized_queries * (args.boxmax - args.boxmin) + args.boxmin
    num_chunks = int(np.ceil(args.num_queries / args.chunk_size))
    query_chunks = np.array_split(queries, num_chunks)
    print("splitting {} query points to {} queries in {} chunks".format(
        num_queries, query_chunks[0].shape[0], num_chunks))
    
    query_chunks = [Pointclouds([torch.from_numpy(q).float().to(device)]) for q in query_chunks]
    
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
    return point_mesh_face_distance(mesh, query_coords)

def worker_job(job):
    input_dir, device = job
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
        print("found existing query file, skipping...")
    else:
        if mesh_path.exists():
            print("loading mesh {}".format(mesh_path))
            mesh = load_objs_as_meshes([mesh_path], device=device, load_textures=False) # load obj file
            query_coords, chunk_query_coords = create_query_coords(args.num_queries, device)  
            
            try:
                list_queries, list_distances = [], []
                for cur_query_coords in chunk_query_coords:
                    distances = compute_grad(cur_query_coords, mesh)
                    print(distances.shape)
                    exit()
                    
                    distances = distances.numpy().tolist()
                    list_distances += distances
                
                array_queries = query_coords.astype(np.float16)
                array_distances = np.array(list_distances).astype(np.float16)
                assert array_queries.shape[0] == array_distances.shape[0]
                
                np.savez_compressed(output_path, queries=array_queries, distances=array_distances)
            except IndexError:
                print("invalid query or mesh data, skipping...")
        else:
            print("mesh file doesn't exist, skipping...")

if __name__ == "__main__":
    in_dir_list = list(sorted(in_dir.iterdir()))
    assert len(in_dir_list) > 0, "empty input data list"
    
    job_list = map_jobs_to_devices(in_dir_list)
    
    # if args.num_workers > 1:
    #     pool = mp.Pool(processes=args.num_workers)
    #     pool.map(worker_job, in_dir_list)
    # else:
    #     for in_dir in in_dir_list:
    #         # for debugging
    #         # in_dir = Path("data/nvme2/scannetpp_volume_norm_-1_1_0.95/sdf/836363121f")
    #         worker_job(in_dir)
    #         # exit()
    
    for job in job_list:
        worker_job(job)