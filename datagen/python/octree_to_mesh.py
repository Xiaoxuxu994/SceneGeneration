import os
import sys
import argparse

import numpy as np
import open3d as o3d
import multiprocessing as mp

from skimage import measure
from pathlib import Path

import sys

try:
    sys.path.append("lib/MeshUDF/custom_mc")
    from _marching_cubes_lewiner import udf_mc_lewiner
    print("Lewiner Marching Cubes imported.")
except ImportError:
    print("MeshUDF not installed. Please refer to https://github.com/cvlab-epfl/MeshUDF")

OCTREE_FILE = "vs{}_to_vs{}_th{}"
OCTREE_SUFFIX = ".npz"

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--src_size", type=int, required=True)
parser.add_argument("--tar_size", type=int, required=True)
parser.add_argument("--thres", type=int, required=True)
parser.add_argument("--mc_alpha", type=float, default=1.5)
parser.add_argument("--engine", type=str, choices=["mc", "poisson", "meshudf"])
parser.add_argument("--boxmin", type=float, default=-1)
parser.add_argument("--boxmax", type=float, default=1)
parser.add_argument("--alpha", type=float, default=3)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--grad", action="store_true", help="if gradients are provided with octree data.")
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
    if args.grad: octree_name += "_grad"
    input_path = (input_dir / octree_name).with_suffix(OCTREE_SUFFIX)
    output_name = octree_name+'_'+args.engine
    output_path = (output_dir / output_name).with_suffix(".ply")
    
    if not output_path.exists() or args.force:
        
        if args.engine == "mc":
            
            octree = np.load(input_path)
            
            print("creating volume from {}".format(str(input_path)))
            volume = np.full(shape=(args.tar_size, args.tar_size, args.tar_size), fill_value=thres)
            X, Y, Z = octree["tar_coords"].T
            volume[X, Y, Z] = octree["tar_values"]
                
            # run marching cubes
            print("running marching cubes...")
            verts, faces, _, _ = measure.marching_cubes(volume, 
                level=voxel_size*args.mc_alpha, 
                spacing=(voxel_size, voxel_size, voxel_size)
            )
            
            # normalize
            verts -= abs(args.boxmin)
            
            # create mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.io.write_triangle_mesh(str(output_path), mesh)
        
        elif args.engine == "poisson":
            
            octree = np.load(input_path)
            
            # load
            anchor_coords = octree["tar_coords"].astype(float)
            distances = octree["tar_values"]
            gradients = octree["tar_grads"]
            
            # normalize
            anchor_coords -= args.tar_size // 2
            anchor_coords /= args.tar_size // 2
            
            # apply gradients
            anchor_coords += gradients * distances[:, None]
            
            # create point clouds
            verts = o3d.utility.Vector3dVector(anchor_coords)
            pcd = o3d.geometry.PointCloud(verts)
            o3d.io.write_point_cloud(str(output_path), pcd)
            
            # # run poisson reconstruction
            # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, n_threads=1)
            # # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([0.005]))
            # vertices_to_remove = densities < np.quantile(densities, 0.05)
            # mesh.remove_vertices_by_mask(vertices_to_remove)
            # mesh.paint_uniform_color([0.8, 0.8, 0.8])

            # o3d.io.write_triangle_mesh(str(output_path), mesh)

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