import os
import shutil
import argparse
import trimesh

import numpy as np
import open3d as o3d
import multiprocessing as mp

from pathlib import Path
from skimage import measure
from copy import deepcopy

# import sys
# sys.path.append(os.path.dirname(__file__))
# from lib.utils.hash import encode_coords, decode_coords, encode_coords_np, decode_coords_np

VOLUME_FILE = "vs_{}.npy"
OCTREE_FILE = "vs{}_to_vs{}_th{}"
OCTREE_SUFFIX = ".npz"

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--mesh_dir", type=str, default="")
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--out_dir_sec", type=str, default="", help="secondary output directory.")
parser.add_argument("--mc_dir", type=str, default="", help="marching cubes output directory.")
parser.add_argument("--mc_alpha", type=float, default=1)
parser.add_argument("--src_size", type=int, required=True)
parser.add_argument("--tar_size", type=int, required=True)
parser.add_argument("--boxmin", type=float, default=-1)
parser.add_argument("--boxmax", type=float, default=1)
parser.add_argument("--thres", type=int, required=True, help="number of voxel in the finest resolution.")
parser.add_argument("--check", action="store_true", help="check octree.")
parser.add_argument("--stats", action="store_true", help="check compression stats.")
parser.add_argument("--grad", action="store_true", help="generate octree with gradients, \
                    i.e. unit vectors to the mesh surface.")
parser.add_argument("--sign", action="store_true", help="generate octree with gradient signs, \
                    i.e. whether the gradients are pointing at the same direction as the triangle normals.")
parser.add_argument("--force", action="store_true", help="generate octree even if target exists.")
parser.add_argument("--num_workers", type=int, default=1)
args = parser.parse_args()

in_dir = Path(args.in_dir)
out_dir = Path(args.out_dir)
mesh_dir = Path(args.mesh_dir)

out_dir.mkdir(exist_ok=True, parents=True)

assert args.tar_size > args.src_size, \
    "{} must be greater than {}".format(args.tar_size, args.src_size)
if args.grad: assert len(args.mesh_dir) > 0, \
    "mesh_dir must be given if you want to compute the gradients."

def compute_angle(A, B):
    dot_product = (A*B).sum(1)
    magnitude_A = np.linalg.norm(A, axis=1)
    magnitude_B = np.linalg.norm(B, axis=1)
    cosine_angle = dot_product / (magnitude_A * magnitude_B + 1e-16)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def compute_grad(query_coords, # indices in a volume
        mesh_path: str):
    # compute gradients based on truncated UDFs
    query_points = query_coords.astype(float)
    query_points /= args.tar_size
    query_points *= args.boxmax - args.boxmin
    query_points += args.boxmin

    print("loading mesh {}".format(mesh_path))
    # mesh = trimesh.load_mesh(mesh_path) # obj file
    mesh = trimesh.load_mesh(mesh_path, validate=False, process=False) # obj file

    print("computing gradients for {} query points".format(query_points.shape[0]))
    
    # # slow implementation
    # closest_points, distance, triangle_id = trimesh.proximity.closest_point(mesh, query_points)
    # query_grads = (closest_points - query_points) / np.stack([distance]*3, axis=1)

    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    )
    mesh_o3d.compute_triangle_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d))
    return_dict = scene.compute_closest_points(o3d.core.Tensor.from_numpy(query_points.astype(np.float32)))
    closest_points = return_dict["points"].numpy()
    closest_triangle_ids = [e.item() for e in return_dict["primitive_ids"]]
    
    distance = np.linalg.norm(closest_points - query_points, axis=1) + 1e-16
    query_grads = (closest_points - query_points) / np.stack([distance]*3, axis=1)
    
    # signs
    triangle_normals = np.array(mesh_o3d.triangle_normals)
    closest_triangle_normals = triangle_normals[closest_triangle_ids]
    grad_normal_angles = compute_angle(query_grads, closest_triangle_normals)
    grad_normal_signs = (grad_normal_angles <= 90).astype(int) # 0 ~ 1
    grad_normal_signs = grad_normal_signs * 2 - 1 # -1 ~ 1
    
    return query_grads, grad_normal_signs

def worker_job(input_dir):
    src_path = input_dir / VOLUME_FILE.format(args.src_size)
    tar_path = input_dir / VOLUME_FILE.format(args.tar_size)
    
    src_flag = True
    if not src_path.exists():
        print("{} doesn't exist, skipping...".format(str(src_path)))
        src_flag = False
    
    tar_flag = True
    if not tar_path.exists():
        print("{} doesn't exist, skipping...".format(str(tar_path)))
        tar_flag = False
    
    if src_flag and tar_flag:
    
        output_dir = out_dir / input_dir.name
        output_dir.mkdir(exist_ok=True, parents=True)
        output_name = OCTREE_FILE.format(args.src_size, args.tar_size, args.thres)
        # if args.grad: output_name += "_grad"
        output_path = output_dir / output_name 
        output_path = output_path.with_suffix(OCTREE_SUFFIX)
        
        if output_path.exists() and not args.force:
            print("found existing octree file {}, skipping...".format(str(output_path)))
        else:
            print("processing {}...".format(str(input_dir)))
            
            try:
                src_vol = np.load(src_path, allow_pickle=True)
                tar_vol = np.load(tar_path, allow_pickle=True)
                
                '''
                    Follow the fine-to-coarse-to-fine strategy, i.e.
                    1. go to the finest resolution and select the valid voxel with a threshold;
                    2. merge the fine voxels to form the coarse voxels;
                    3. divide the coarse voxels again to fine voxels.
                    This strategy makes sure that all valid voxels are covered.
                '''
                
                # fine-to-coarse
                thres = (args.boxmax - args.boxmin) / args.tar_size * args.thres
                tar_ids_1 = np.stack(np.where(tar_vol <= thres), axis=1) # Nf x 3
                
                rate = args.tar_size // args.src_size
                src_ids_expanded = tar_ids_1 // rate
                src_ids_masks = tar_ids_1 % rate == 0 # Nf x 3
                src_ids_masks = np.sum(src_ids_masks, axis=1) == 3 # all true row should sum up to 3
                src_ids = src_ids_expanded[src_ids_masks]
                
                # coarse-to-fine
                offsets = np.meshgrid(np.arange(rate), np.arange(rate), np.arange(rate), indexing="ij")
                offsets = np.stack([a.reshape(-1) for a in offsets], axis=1)
                num_offsets = offsets.shape[0]
                
                origins = src_ids.copy()
                num_origins = origins.shape[0]
                
                offsets = np.repeat(offsets[None, :, :], num_origins, axis=0)
                origins = np.repeat(origins[:, None ,:], num_offsets, axis=1)
                coords = origins * rate + offsets
                
                src_i, src_j, src_k = origins.reshape((-1, 3)).T
                tar_i, tar_j, tar_k = coords.reshape((-1, 3)).T
                
                src_coords = np.stack([src_i, src_j, src_k], axis=1).astype(int)
                tar_coords = np.stack([tar_i, tar_j, tar_k], axis=1).astype(int)
                tar_values = tar_vol[tar_i, tar_j, tar_k].astype(float)
                
                entry_data = {
                    "src_coords": src_coords,
                    "tar_coords": tar_coords,
                    "tar_values": tar_values
                }
                    
                print("save to {}".format(str(output_path)))
                # np.savez_compressed(output_path, src_coords=src_coords, tar_coords=tar_coords, tar_values=tar_values)
                
                np.savez_compressed(output_path, **entry_data)
                
                if args.check:
                    X, Y, Z = tar_coords.T
                    original_values = tar_vol[X, Y, Z]
                    compressed_values = tar_values
                    if not np.array_equal(original_values, compressed_values):
                        raise ValueError("compressed volume are not consistent with the original volume.")
                    
            except ValueError as e:
                print("captured error: {}, skipping...".format(e))
                print(src_path)
                print(tar_path)
        
        if args.grad:
            output_name += "_grad"
            grad_path = output_dir / output_name 
            grad_path = grad_path.with_suffix(OCTREE_SUFFIX)
            
            output_name += "_sign"
            grad_sign_path = output_dir / output_name 
            grad_sign_path = grad_sign_path.with_suffix(OCTREE_SUFFIX)
                
            if grad_path.exists() and grad_sign_path.exists() and not args.force:
                print("found existing gradients file {}, skipping...".format(str(grad_path)))
            else:    
                print("computing gradients using octree {}...".format(str(output_path)))
                
                if output_path.exists():
                    print("loading existing octree from {}".format(str(output_path)))
                    entry_data = np.load(output_path)
                    tar_coords = entry_data["tar_coords"]
                
                scene_name = input_dir.name
                mesh_path = mesh_dir / scene_name / "normalized.obj"
                tar_grads, tar_grad_signs = compute_grad(tar_coords, mesh_path)
                
                if not grad_path.exists():
                    np.savez_compressed(grad_path, tar_grads=tar_grads)      
            
                # print("computing gradients using octree {}...".format(str(output_path)))
                # scene_name = input_dir.name
                # mesh_path = mesh_dir / scene_name / "normalized.obj"
                # tar_grads, tar_grad_signs = compute_grad(tar_coords, mesh_path)
                
                # print("saving gradients to {}".format(str(grad_path)))
                # np.savez_compressed(grad_path, tar_grads=tar_grads)
            
                if args.sign: # NOTE only enable this when gradients are enabled
                    print("saving gradient signs to {}".format(str(grad_sign_path)))
                    np.savez_compressed(grad_sign_path, tar_grad_signs=tar_grad_signs)
        
        # copy to a separate secondary output directory
        if len(args.out_dir_sec) > 0:
            out_dir_sec = Path(args.out_dir_sec)
            output_dir_sec = out_dir_sec / input_dir.name
            output_dir_sec.mkdir(exist_ok=True, parents=True)
            output_path_sec = output_dir_sec / OCTREE_FILE.format(args.src_size, args.tar_size, args.thres)
            if not output_path_sec.exists():
                print("copy to secondary output directory {}".format(args.out_dir_sec))
                shutil.copy(output_path, output_path_sec)

        # convert a octree to a volume and run marching cubes
        if len(args.mc_dir) > 0:
            mc_dir = Path(args.mc_dir)
            mc_dir = mc_dir / input_dir.name
            mc_dir.mkdir(exist_ok=True, parents=True)
            mc_path = (mc_dir / output_path.name).with_suffix(".obj")
            
            # recover volumes
            octree = np.load(output_path)
            voxel_size = (args.boxmax - args.boxmin) / args.tar_size
            volume = np.full(shape=(args.tar_size, args.tar_size, args.tar_size), fill_value=octree["tar_values"].max())
            
            X, Y, Z = octree["tar_coords"].T
            volume[X, Y, Z] = octree["tar_values"]
            
            level = voxel_size*args.mc_alpha
            
            print("exporting mesh data to {}".format(str(mc_path)))
            verts, faces, _, _ = measure.marching_cubes(volume, 
                level=level, 
                spacing=(voxel_size, voxel_size, voxel_size)
            )
            verts -= abs(args.boxmin)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.io.write_triangle_mesh(str(mc_path), mesh)
            
            output_path = str(mc_dir / output_path.stem) + "_reference.obj"
            print("exporting mesh data to {}".format(str(output_path)))
            verts, faces, _, _ = measure.marching_cubes(tar_vol, 
                level=level, 
                spacing=(voxel_size, voxel_size, voxel_size)
            )
            verts -= abs(args.boxmin)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.io.write_triangle_mesh(str(output_path), mesh)

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
            
    if args.stats:
        sizes = []
        for in_dir in in_dir_list:
            tar_path = in_dir / VOLUME_FILE.format(args.tar_size)
            output_dir = out_dir / in_dir.name
            output_path = output_dir / OCTREE_FILE.format(args.src_size, args.tar_size, args.thres)
            output_path = output_path.with_suffix(OCTREE_SUFFIX)
            print(output_path)
            
            try:
                original_size = os.path.getsize(tar_path)
                compressed_size = os.path.getsize(output_path)
                sizes.append([original_size, compressed_size])
            except:
                pass
            
        sizes = np.array(sizes) / (1024*1024) # convert to MB
        rates = (sizes[:, 0] - sizes[:, 1]) / sizes[:, 0] * 100
        
        print("\nstats:")
        print("found {} valid data".format(len(sizes)))
        print("original size: min -> {:.2f} MB, max -> {:.2f} MB, avg -> {:.2f} MB".format(sizes[:, 0].min(), sizes[:, 0].max(), sizes[:, 0].mean()))
        print("compress size: min -> {:.2f} MB, max -> {:.2f} MB, avg -> {:.2f} MB".format(sizes[:, 1].min(), sizes[:, 1].max(), sizes[:, 1].mean()))
        print("compress rate: min -> {:.2f} %, max -> {:.2f} %, avg -> {:.2f} %".format(rates.min(), rates.max(), rates.mean()))
        
    print("done!!!")
