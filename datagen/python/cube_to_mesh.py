import os
import sys
import argparse

import numpy as np
import open3d as o3d
import multiprocessing as mp
from skimage import measure

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
# parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--voxel_res", type=int, default=128)
parser.add_argument("--mode", type=str, choices=["sdf", "udf", "occ"])
parser.add_argument("--boxmin", type=float, default=-1)
parser.add_argument("--boxmax", type=float, default=1)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--force", action="store_true", help="generate mesh even if target exists.")
args = parser.parse_args()

in_dir = Path(args.in_dir) / args.mode
out_dir = Path(args.in_dir) / "marching_cubes" / args.mode

voxel_size = (args.boxmax - args.boxmin) / args.voxel_res

alpha = args.alpha
volume_filename = f'vs_{args.voxel_res}.npy'

def worker_job(src_path):
    src_path_name = src_path.name
    src_dir_name = src_path.parent.name
    tar_dir = out_dir / src_dir_name
    tar_dir.mkdir(exist_ok=True, parents=True)
    tar_path = (tar_dir / src_path_name).with_suffix(".obj")
    if not tar_path.exists() or args.force:
        print("processing {}".format(str(src_path)))
        try:
            grid_to_mesh(str(src_path), str(tar_path))
            print("process complete. saved to {}".format(str(tar_path)))
        except ValueError:
            print("value error. skip {}".format(str(src_path)))

def grid_to_mesh(grid_path: str, mesh_path: str):
    grid = np.load(grid_path)
    
    if args.mode == "occ":
        print(grid.max(), grid.min(), np.sum(grid))
        exit()
        
    verts, faces, _, _ = measure.marching_cubes(grid, 
        level=voxel_size*alpha, 
        spacing=(voxel_size, voxel_size, voxel_size)
    )
    
    # move the min bound back to [-1, -1, -1]
    # shifts = (verts.max(axis=0) + verts.min(axis=0)) / 2
    # verts -= shifts
    verts -= 1
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color([0.75, 0.75, 0.75])
    o3d.io.write_triangle_mesh(mesh_path, mesh)

def main():
    src_npy_list = list(map(lambda x: x / volume_filename, sorted(list(in_dir.iterdir()))))
    assert len(src_npy_list) > 0, "empty input data list"
    
    if args.num_workers > 1:
        pool = mp.Pool(processes=args.num_workers)
        pool.map(worker_job, src_npy_list)
    else:
        for src_path in src_npy_list:
            # for debugging
            worker_job(src_path)
            exit()
        
main()