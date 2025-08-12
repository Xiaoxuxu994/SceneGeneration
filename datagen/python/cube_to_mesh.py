import os
import sys
import argparse
import trimesh

import numpy as np
import open3d as o3d
import multiprocessing as mp
from skimage import measure

from pathlib import Path

try:
    sys.path.append("lib/MeshUDF/custom_mc")
    from _marching_cubes_lewiner import udf_mc_lewiner
    print("Lewiner Marching Cubes imported.")
except ImportError:
    print("MeshUDF not installed. Please refer to https://github.com/cvlab-epfl/MeshUDF")

try:
    sys.path.append("lib/DuaMesh-UDF")
    from DualMeshUDF import extract_mesh
    print("DualMeshUDF imported.")
except ImportError:
    print("DualMeshUDF not installed. Please refer to https://github.com/cong-yi/DualMesh-UDF")
    
try:
    sys.path.append("lib/DCUDF/dcudf")
    from mesh_cut import mesh_cut
    print("DCUDF imported.")
except ImportError:
    print("DCUDF not installed. Please refer to https://github.com/jjjkkyz/DCUDF")

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
# parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--voxel_res", type=int, default=128)
parser.add_argument("--mode", type=str, choices=["sdf", "udf", "occ"])
parser.add_argument("--engine", type=str, choices=["mc", "meshudf", "dualmeshudf", "dcudf"])
parser.add_argument("--boxmin", type=float, default=-1)
parser.add_argument("--boxmax", type=float, default=1)
parser.add_argument("--level", type=float, default=0.005)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--force", action="store_true", help="generate mesh even if target exists.")
args = parser.parse_args()

in_dir = Path(args.in_dir) / args.mode
out_dir = Path(args.in_dir) / "marching_cubes" / args.mode

voxel_size = (args.boxmax - args.boxmin) / args.voxel_res

level = args.level
volume_filename = f'vs_{args.voxel_res}.npy'

def worker_job(src_path):
    src_path_name = src_path.name
    src_dir_name = src_path.parent.name
    tar_dir = out_dir / src_dir_name
    tar_dir.mkdir(exist_ok=True, parents=True)
    tar_path = (tar_dir / "{}_{}".format(str(src_path.stem), args.engine)).with_suffix(".obj")
    if not tar_path.exists() or args.force:
        print("processing {}".format(str(src_path)))
        try:
            grid_to_mesh(str(src_path), str(tar_path))
            print("process complete. saved to {}".format(str(tar_path)))
        except ValueError:
            print("value error. skip {}".format(str(src_path)))

def grid_to_mesh(grid_path: str, mesh_path: str):
    grid = np.load(grid_path)
    print(grid.min(), grid.max())
    
    if args.mode == "occ":
        print(grid.max(), grid.min(), np.sum(grid))
        exit()
    
    if args.engine == "meshudf":
        gradients = np.stack(np.gradient(grid), axis=-1)
        verts, faces, _, _ = udf_mc_lewiner(grid,
            -gradients,
            spacing=(voxel_size, voxel_size, voxel_size)
        )
        # move the min bound back to [-1, -1, -1]
        verts -= 1
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color([0.75, 0.75, 0.75])
    elif args.engine == "dualmeshudf": # NOTE NOT READY!!!
        verts, faces, _, _ = extract_mesh(grid, int(np.log2(args.voxel_res)), udf_threshold=level)
        # move the min bound back to [-1, -1, -1]
        verts -= 1
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color([0.75, 0.75, 0.75])
    elif args.engine == "dcudf":
        verts, faces, _, _ = measure.marching_cubes(grid, 
            level=level, 
            spacing=(voxel_size, voxel_size, voxel_size)
        )
        
        # move the min bound back to [-1, -1, -1]
        verts -= 1
        raw_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh_1, mesh_2 = mesh_cut(raw_mesh)
        if len(mesh_1.vertices)>len(mesh_2.vertices):
            raw_mesh = mesh_1
        else:
            raw_mesh = mesh_2
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(raw_mesh.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(raw_mesh.faces)
        mesh.paint_uniform_color([0.75, 0.75, 0.75])
    else:
        verts, faces, _, _ = measure.marching_cubes(grid, 
            level=level, 
            spacing=(voxel_size, voxel_size, voxel_size)
        )
    
        # move the min bound back to [-1, -1, -1]
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