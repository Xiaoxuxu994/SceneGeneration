import os
import sys
import argparse

import numpy as np
import open3d as o3d
import multiprocessing as mp

from skimage import measure
from pathlib import Path

import sys
sys.path.append(os.path.dirname(__file__))
from lib.utils.hash import decode_coords

OCTREE_FILE = "vs{}_to_vs{}_th{}.npy"

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--src_size", type=int, required=True)
parser.add_argument("--tar_size", type=int, required=True)
parser.add_argument("--thres", type=int, required=True)
parser.add_argument("--mc_alpha", type=float, default=1.5)
parser.add_argument("--boxmin", type=float, default=-1)
parser.add_argument("--boxmax", type=float, default=1)
parser.add_argument("--alpha", type=float, default=3)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--force", action="store_true", help="generate octree even if target exists.")
args = parser.parse_args()

in_dir = Path(args.in_dir)
out_dir = Path(args.out_dir)

thres = ((args.boxmax - args.boxmin) / args.src_size) * args.thres
voxel_size = (args.boxmax - args.boxmin) / args.tar_size

def worker_job(input_dir):
    input_dir_name = input_dir.name
    output_dir = out_dir / input_dir_name
    output_dir.mkdir(exist_ok=True, parents=True)
    octree_name = OCTREE_FILE.format(args.src_size, args.tar_size, args.thres)
    input_path = input_dir / octree_name
    output_path = (output_dir / octree_name).with_suffix(".obj")
    
    if not output_path.exists() or args.force:
        print("processing {}".format(str(input_path)))
        octree = np.load(input_path)
        volume = np.full(shape=(args.tar_size, args.tar_size, args.tar_size), fill_value=thres)
        for row in range(octree.shape[0]):
            _, coords, value = octree[row]
            i, j, k = decode_coords(int(coords))
            volume[i, j, k] = value
            
        # create a mesh
        verts, faces, _, _ = measure.marching_cubes(volume, 
            level=voxel_size*args.mc_alpha, 
            spacing=(voxel_size, voxel_size, voxel_size)
        )
        verts -= abs(args.boxmin)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.io.write_triangle_mesh(str(output_path), mesh)

def main():
    oct_dir_list = list(sorted(list(in_dir.iterdir())))
    assert len(oct_dir_list) > 0, "empty input data list"
    
    if args.num_workers > 1:
        pool = mp.Pool(processes=args.num_workers)
        pool.map(worker_job, oct_dir_list)
    else:
        for oct_dir in oct_dir_list:
            # for debugging
            worker_job(oct_dir)
            exit()
        
main()