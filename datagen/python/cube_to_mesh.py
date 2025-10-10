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

try:
    sys.path.append("lib/MeshUDF/custom_mc")
    from _marching_cubes_lewiner import udf_mc_lewiner
    print("Lewiner Marching Cubes imported.")
except ImportError:
    print("MeshUDF not installed. Please refer to https://github.com/cvlab-epfl/MeshUDF")

try:
    sys.path.append("lib/DualMeshUDF")
    from extract_mesh import extract_mesh
    print("DualMeshUDF imported.")
except ImportError:
    print("DualMeshUDF not installed. Please refer to https://github.com/cong-yi/DualMesh-UDF")
    
try:
    sys.path.append("lib/DCUDF/dcudf")
    from mesh_cut import mesh_cut
    print("DCUDF imported.")
except ImportError:
    print("DCUDF not installed. Please refer to https://github.com/jjjkkyz/DCUDF")
    
try:
    sys.path.append("lib/CAPUDF/tools")
    from surface_extraction import surface_extraction
    print("CAP-UDF imported.")
except ImportError:
    print("CAP-UDF not installed. Please refer to https://github.com/junshengzhou/CAP-UDFF")
    
try:
    sys.path.append("lib/ODC")
    from occupancy_dual_contouring import occupancy_dual_contouring
    print("ODC imported.")
except ImportError:
    print("ODC not installed. Please refer to https://github.com/KAIST-Visual-AI-Group/ODC")

OCTREE_FILE = "vs{}_to_vs{}_th{}"
OCTREE_SUFFIX = ".npz"

POISSON_EXE_CMD = (lambda in_path, out_path, depth: f"./lib/AdaptiveSolvers_17.00/Bin/Linux/PoissonRecon --in {in_path} --out {out_path} --depth {depth} --density --threads 1 --verbose")
SSD_EXE_CMD = (lambda in_path, out_path, depth: f"./lib/AdaptiveSolvers_17.00/Bin/Linux/SSDRecon --in {in_path} --out {out_path} --depth {depth} --density --threads 1 --verbose")
EXE_CMD = {"poisson_exe": POISSON_EXE_CMD, "ssd_exe": SSD_EXE_CMD}
TRIM_CMD = (lambda in_path, out_path, value: f"./lib/AdaptiveSolvers_17.00/Bin/Linux/SurfaceTrimmer --in {in_path} --out {out_path} --trim {value} --verbose")

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--grad_dir", type=str, default="")
parser.add_argument("--sign_dir", type=str, default="")
parser.add_argument("--mesh_dir", type=str, default="")
# parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--voxel_res", type=int, default=128)

# parser.add_argument("--type", type=str, choices=["cube", "octree"], help="specify the input type, \
#     cube -> a complete volume, octree -> an octree from a truncated volume.")
parser.add_argument("--mode", type=str, choices=["sdf", "udf", "occ", "octree"])
parser.add_argument("--oct_src_size", type=int, default=256, help="only necessary for an octree input")
parser.add_argument("--oct_tar_size", type=int, default=512, help="only necessary for an octree input")
parser.add_argument("--oct_thres", type=int, default=3, help="only necessary for an octree input")

# NOTE poisson_exe and ssd_exe have to be compiled beforehand, please refer to 
# https://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version17.00/
# it is recommended to use v17.00
# download the source code and put it under ./lib
# unzip the file, enter the folder, and run `make`
parser.add_argument("--engine", type=str, 
    choices=["mc", "mcubes", "meshudf", "dualmeshudf", "dcudf", "capudf", "odc", "pcd", "poisson",
             "poisson_exe", "ssd_exe" 
             ])

parser.add_argument("--suffix", type=str,
    choices=["obj", "ply", "xyz"], default="ply")
parser.add_argument("--boxmin", type=float, default=-1)
parser.add_argument("--boxmax", type=float, default=1)
parser.add_argument("--level", type=float, default=0.005)
parser.add_argument("--depth", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--precise_grad", action="store_true", help="compute precise gradients to mesh.")
parser.add_argument("--normal", action="store_true", help="save normals using gradient vectors and signs.")
parser.add_argument("--trim", action="store_true", help="trim low density submesh.")
parser.add_argument("--force", action="store_true", help="generate mesh even if target exists.")
args = parser.parse_args()

print("generate mesh data using engine: {}".format(args.engine))

in_dir = Path(args.in_dir) / args.mode
out_dir = Path(args.in_dir) / "marching_cubes" / args.mode

if len(args.grad_dir) > 0 and args.mode != "octree": 
    assert len(args.mesh_dir) > 0
    
voxel_size = (args.boxmax - args.boxmin) / args.voxel_res

level = args.level

if args.mode == "octree":
    volume_filename = OCTREE_FILE.format(args.oct_src_size, args.oct_tar_size, args.oct_thres)
    volume_filename += OCTREE_SUFFIX
else:
    volume_filename = f'vs_{args.voxel_res}.npy'

def worker_job(src_path):
    src_path_name = src_path.name
    src_dir_name = src_path.parent.name
    tar_dir = out_dir / src_dir_name
    tar_dir.mkdir(exist_ok=True, parents=True)
    tar_name = "{}_{}_{}".format(str(src_path.stem), args.mode, args.engine)
    if args.engine == "poisson": 
        tar_name += "_{}".format(args.depth)
        if args.trim: tar_name += "_trim"
    tar_path = (tar_dir / tar_name).with_suffix(".{}".format(args.suffix))
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

    if args.mode == "octree":
        try:
            print("loading octree from {}".format(grid_path))
            octree = np.load(grid_path)
            
            octree_coords = octree["tar_coords"]
            octree_values = octree["tar_values"]
            
            oct_thres = (args.boxmax - args.boxmin) / args.oct_tar_size * args.oct_thres
            grid = np.full(shape=(args.oct_tar_size, args.oct_tar_size, args.oct_tar_size), fill_value=oct_thres)
            
            X, Y, Z = octree_coords.T
            grid[X, Y, Z] = octree_values
        except KeyError:
            print("please make sure the input octree has tar_coords and tar_values.")
            exit()
        
    else:
        print("loading volume from {}".format(grid_path))
        grid = np.load(grid_path)        
        
    if len(args.grad_dir) > 0 and args.precise_grad:
        
        if args.mode == "octree":
        
            scene_name = Path(grid_path).parent.stem
            grad_dir = Path(args.grad_dir) / args.mode / scene_name
            grad_name = OCTREE_FILE.format(args.oct_src_size, args.oct_tar_size, args.oct_thres)+"_grad"
            grad_path = (grad_dir / grad_name).with_suffix(OCTREE_SUFFIX)
            
            print("loading precise octree gradients from {}".format(str(grad_path)))
            
            octree_grads = np.load(grad_path)["tar_grads"]
            
            if args.engine in ["meshudf", "capudf"]:
                print("creating a dummy gradient volume for octree data...")
            
                grad = -np.stack(np.gradient(grid), axis=-1)
                grad_shape = grad.shape
                grad = normalize(grad.reshape((-1, 3)), axis=1).reshape(grad_shape)
                
                grad[X, Y, Z] = octree_grads
            
        else:
            scene_name = Path(grid_path).parent.stem
            grad_dir = Path(args.grad_dir) / "grad" / scene_name
            grad_dir.mkdir(exist_ok=True, parents=True)
            file_name = Path(grid_path).stem
            grad_path = grad_dir / "{}.npy".format(str(file_name))
            
            if not grad_path.exists():
                print("computing precise gradients to the mesh surface.")
                original_mesh_path = Path(args.mesh_dir) / scene_name / "normalized.obj"
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
        
    print("creating mesh file using engine {}".format(args.engine))
    
    if args.engine == "pcd":
        
        if args.mode == "octree" and args.precise_grad:
            octree_coords = octree_coords.astype(float)        
            octree_coords -= args.oct_tar_size // 2
            octree_coords /= args.oct_tar_size // 2
            
            verts = octree_coords + octree_grads * octree_values[:, None]
            
            # create point clouds
            verts = o3d.utility.Vector3dVector(verts)
            pcd = o3d.geometry.PointCloud(verts)
            
            if args.normal:
                scene_name = Path(grid_path).parent.stem
                sign_dir = Path(args.sign_dir) / args.mode / scene_name
                sign_name = OCTREE_FILE.format(args.oct_src_size, args.oct_tar_size, args.oct_thres)+"_grad"+"_sign"
                sign_path = (sign_dir / sign_name).with_suffix(OCTREE_SUFFIX)
                signs = np.load(sign_path)["tar_grad_signs"]
                normals = octree_grads * signs[:, None]
                normals = o3d.utility.Vector3dVector(normals)
                pcd.normals = o3d.utility.Vector3dVector(normals)
            
            o3d.io.write_point_cloud(str(mesh_path), pcd)
            
        else:
            print("please set the mode as 'octree' and use flag '--precise_grad'")
            
    elif args.engine == "poisson":
        
        if args.mode == "octree" and args.precise_grad and len(args.sign_dir) > 0 and args.normal:
            
            octree_coords = octree_coords.astype(float)        
            octree_coords -= args.oct_tar_size // 2
            octree_coords /= args.oct_tar_size // 2
            
            verts = octree_coords + octree_grads * octree_values[:, None]
            
            # create point clouds
            verts = o3d.utility.Vector3dVector(verts)
            pcd = o3d.geometry.PointCloud(verts)
            
            # load point normals
            scene_name = Path(grid_path).parent.stem
            sign_dir = Path(args.sign_dir) / args.mode / scene_name
            sign_name = OCTREE_FILE.format(args.oct_src_size, args.oct_tar_size, args.oct_thres)+"_grad"+"_sign"
            sign_path = (sign_dir / sign_name).with_suffix(OCTREE_SUFFIX)
            signs = np.load(sign_path)["tar_grad_signs"]
            normals = octree_grads * signs[:, None]
            normals = o3d.utility.Vector3dVector(normals)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
            # create a mesh
            print("running Poisson Reconstruction with depth {}".format(args.depth))
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, 
                depth=args.depth, n_threads=1)
            
            mesh.paint_uniform_color([0.8, 0.8, 0.8])
            
            # trim low density submesh
            if args.trim:
                vertices_to_remove = densities < args.depth-1
                # vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)
                
                # mesh_path = Path(mesh_path)
                # mesh_stem = str(mesh_path.stem)+"_trim"
                # mesh_path = mesh_path.with_stem(mesh_stem)
            
            # save mesh
            o3d.io.write_triangle_mesh(str(mesh_path), mesh)
            
        else:
            print("please set the mode as 'octree', set the path to normal signs, and use flag '--precise_grad' and '--normal'")
            
    elif args.engine == "poisson_exe" or args.engine == "ssd_exe":
        
        if args.mode == "octree" and args.precise_grad and len(args.sign_dir) > 0 and args.normal:
            
            octree_coords = octree_coords.astype(float)        
            octree_coords -= args.oct_tar_size // 2
            octree_coords /= args.oct_tar_size // 2
            
            verts = octree_coords + octree_grads * octree_values[:, None]
            
            # create point clouds
            verts = o3d.utility.Vector3dVector(verts)
            pcd = o3d.geometry.PointCloud(verts)
            
            # load point normals
            scene_name = Path(grid_path).parent.stem
            sign_dir = Path(args.sign_dir) / args.mode / scene_name
            sign_name = OCTREE_FILE.format(args.oct_src_size, args.oct_tar_size, args.oct_thres)+"_grad"+"_sign"
            sign_path = (sign_dir / sign_name).with_suffix(OCTREE_SUFFIX)
            signs = np.load(sign_path)["tar_grad_signs"]
            normals = octree_grads * signs[:, None]
            normals = o3d.utility.Vector3dVector(normals)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
            mesh_path = Path(mesh_path)
            mesh_stem = mesh_path.stem
            
            # create a point cloud
            pcd_stem = str(mesh_stem)+"_pcd"
            pcd_path = mesh_path.with_stem(pcd_stem)
            if not pcd_path.exists():
                print("exporting point cloud file with normal vectors")
                o3d.io.write_point_cloud(str(pcd_path), pcd)
            
            if not mesh_path.exists():
                print("running Poisson Reconstruction with depth {}".format(args.depth))
                os.system(EXE_CMD[args.engine](str(pcd_path), str(mesh_path), args.depth))
            
            # trim low density submesh
            if args.trim:    
                trim_stem = str(mesh_path.stem)+"_trim"
                trim_path = mesh_path.with_stem(trim_stem)
                os.system(TRIM_CMD(str(mesh_path), str(trim_path), args.depth-1))
            
        else:
            print("please set the mode as 'octree', set the path to normal signs, and use flag '--precise_grad' and '--normal'")
            
    elif args.engine == "meshudf":
        verts, faces, _, _ = udf_mc_lewiner(
            grid.astype(np.float32),
            grad.astype(np.float32),
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
        verts, faces = surface_extraction(
            grid, 
            grad, 
            args.voxel_res, 
            level
        )
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
        verts, faces, _, _ = measure.marching_cubes(
            grid, 
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
            # exit()
        
main()