import os
import sys
import torch
import torch_npu
import argparse
import trimesh
import mcubes

import numpy as np
import open3d as o3d
import multiprocessing as mp

from skimage import measure
from sklearn.preprocessing import normalize
from pathlib import Path

parser = argparse.ArgumentParser(
    description="convert octree to neural vector field (NVF) \
                specifically, each voxel in an octree encodes the truncated unsigned distance \
                to the mesh surface, while omitting the exact direction pointing to the closest \
                point on the mesh. \
                NVF extends the TUDF volume with the unit vector pointing to the mesh surface."
)
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--grad_dir", type=str, default="")
parser.add_argument("--mesh_dir", type=str, default="")
# parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--voxel_res", type=int, default=128)
parser.add_argument("--mode", type=str, choices=["sdf", "udf", "occ"])
parser.add_argument("--engine", type=str, 
    choices=["mc", "mcubes", "meshudf", "dualmeshudf", "dcudf", "capudf", "odc"])
parser.add_argument("--boxmin", type=float, default=-1)
parser.add_argument("--boxmax", type=float, default=1)
parser.add_argument("--level", type=float, default=0.005)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--precise_grad", action="store_true", help="compute precise gradients to mesh.")
parser.add_argument("--force", action="store_true", help="generate mesh even if target exists.")
args = parser.parse_args()

if len(args.grad_dir) > 0: assert len(args.mesh_dir) > 0

print("generate mesh data using engine: {}".format(args.engine))

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
        # try:
        #     grid_to_mesh(str(src_path), str(tar_path))
        #     print("process complete. saved to {}".format(str(tar_path)))
        # except ValueError:
        #     print("value error. skip {}".format(str(src_path)))    
        grid_to_mesh(str(src_path), str(tar_path))

def compute_grad(grid, mesh_path: str, grad_path: str, thres=0.01):
    # truncated gradients are represented by original grid gradients
    grad = -np.stack(np.gradient(grid), axis=-1)
    grad_shape = grad.shape
    grad = normalize(grad.reshape((-1, 3)), axis=1).reshape(grad_shape)
    
    # compute gradients based on truncated UDFs
    query_coords = np.where(grid < thres)
    query_coords = np.stack(query_coords, axis=-1)
    query_points = query_coords.astype(float)
    query_points /= args.voxel_res
    query_points *= args.boxmax - args.boxmin
    query_points += args.boxmin

    print("loading mesh {}".format(mesh_path))
    mesh = trimesh.load_mesh(mesh_path)

    print("computing gradients for {} query points with threshold {}".format(query_points.shape[0], thres))
    
    # # slow implementation
    # closest_points, distance, triangle_id = trimesh.proximity.closest_point(mesh, query_points)
    # query_grads = (closest_points - query_points) / np.stack([distance]*3, axis=1)

    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d))
    return_dict = scene.compute_closest_points(o3d.core.Tensor.from_numpy(query_points.astype(np.float32)))
    closest_points = return_dict["points"].numpy()
    
    distance = np.linalg.norm(closest_points - query_points, axis=1) + 1e-16
    query_grads = (closest_points - query_points) / np.stack([distance]*3, axis=1)

    query_i, query_j, query_k = query_coords.T
    grad[query_i, query_j, query_k] = query_grads
    
    np.save(grad_path, grad)
    
    return grad

def to_func(xyz, grid):
    x, y, z = (xyz).T
    return grid[x.int(), y.int(), z.int()]

def grid_to_mesh(grid_path: str, mesh_path: str):
    grid = np.load(grid_path)
    
    if args.engine in ["meshudf", "dualmeshudf", "capudf"]:
        if len(args.grad_dir) > 0 and args.precise_grad:
            scene_name = Path(grid_path).parent.stem
            grad_dir = Path(args.grad_dir) / "grad" / scene_name
            grad_dir.mkdir(exist_ok=True, parents=True)
            file_name = Path(grid_path).stem
            grad_path = grad_dir / "{}.npy".format(str(file_name))
            original_mesh_path = Path(args.mesh_dir) / scene_name / "normalized.obj"
            if not grad_path.exists():
                print("computing precise gradients to the mesh surface.")
                grad = compute_grad(grid, str(original_mesh_path), str(grad_path), level)
            else:
                print("loading pre-computed gradients from {}".format(str(grad_path)))
                grad = np.load(grad_path)
        else:
            grad = -np.stack(np.gradient(grid), axis=-1)
            grad_shape = grad.shape
            grad = normalize(grad.reshape((-1, 3)), axis=1).reshape(grad_shape)
    
    if args.mode == "occ":
        print(grid.max(), grid.min(), np.sum(grid))
        exit()
    
    if args.engine == "meshudf":
        verts, faces, _, _ = udf_mc_lewiner(grid,
            grad,
            spacing=(voxel_size, voxel_size, voxel_size)
        )
        # move the min bound back to [-1, -1, -1]
        verts -= 1
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        
        o3d.io.write_triangle_mesh(mesh_path, mesh)
    elif args.engine == "capudf":
        verts, faces = surface_extraction(grid, grad, args.voxel_res, level)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        
        o3d.io.write_triangle_mesh(mesh_path, mesh)
    elif args.engine == "dualmeshudf": # NOTE NOT READY!!!
        verts, faces = extract_mesh(
            grid,
            grad,
            int(np.log2(args.voxel_res)), 
            udf_threshold=level,
            boxmin=args.boxmin,
            boxmax=args.boxmax
        )
        # move the min bound back to [-1, -1, -1]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        
        o3d.io.write_triangle_mesh(mesh_path, mesh)
    elif args.engine == "dcudf": # cut the double-layer mesh
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
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        
        o3d.io.write_triangle_mesh(mesh_path, mesh)
    elif args.engine == "mcubes": # NOTE this implementation of marching cubes produces more clumsy results
        verts, faces = mcubes.marching_cubes(grid, level)
        
        # normalize vertices
        verts -= args.voxel_res // 2
        verts /= args.voxel_res // 2
        
        mcubes.export_obj(verts, faces, mesh_path)
    elif args.engine == "odc":
        odc = occupancy_dual_contouring(
            # torch.device("cpu")
            torch.device("npu:0")
        )
        verts, faces = odc.extract_mesh(
            to_func,
            grid,
            min_coord=[0]*3,
            max_coord=[args.voxel_res-1]*3,
            num_grid=args.voxel_res,
            isolevel=level
        )
        verts, faces = verts.cpu().numpy(), faces.cpu().numpy()
        # normalize vertices
        verts -= args.voxel_res // 2
        verts /= args.voxel_res // 2
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
    
        o3d.io.write_triangle_mesh(mesh_path, mesh)
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
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
    
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