# read list of shapenet meshes
# find them on disk
# scale to unity
# convert to df highres and lowres
from pathlib import Path
import trimesh
from argparse import ArgumentParser
import subprocess
import numpy as np
import marching_cubes as mc
import math
from tqdm import tqdm
import os
import torch

highres_dim = 64
lowres_dim = 9
padding_lowres = 0
padding_highres = math.ceil(highres_dim / lowres_dim) * padding_lowres
highres_voxel_size = 1 / (highres_dim - 2 * padding_highres)
lowres_voxel_size = 1 / (lowres_dim - 2 * padding_lowres)


shapenet_dir = Path("/cluster_HDD/pegasus/yawar/ShapeNetCore.v2/")

print(f"HighresVoxelRes: {highres_voxel_size}")
print(f"LowresVoxelRes: {lowres_voxel_size}")

sdf_gen_highres_cmd = lambda in_filepath, out_filepath: f"bin/sdf_gen_shapenet {in_filepath} {out_filepath} {highres_voxel_size} {padding_highres}"
sdf_gen_lowres_cmd = lambda in_filepath, out_filepath: f"bin/sdf_gen_shapenet {in_filepath} {out_filepath} {lowres_voxel_size} {padding_lowres}"


def visualize_highres(df_path):
    df = np.load(str(df_path)+".npz")["arr"]
    vertices, triangles = mc.marching_cubes(df, highres_voxel_size * 0.75)
    mc.export_obj(vertices, triangles, str(df_path) + "_vis.obj")


def visualize_lowres_vox(df_path):
    to_point_list = lambda s: np.concatenate([c[:, np.newaxis] for c in np.where(s == True)], axis=1)
    df = np.load(str(df_path)+".npz")["arr"]
    point_list = to_point_list(df <= 0.75 * lowres_voxel_size)
    if point_list.shape[0] > 0:
        base_mesh = trimesh.voxel.ops.multibox(centers=point_list, pitch=1)
        base_mesh.export(str(df_path) + "_vis.obj")

def visualize_lowres(df_path):
    df = np.load(str(df_path)+".npz")["arr"]
    vertices, triangles = mc.marching_cubes(df, lowres_voxel_size * 0.5)
    mc.export_obj(vertices, triangles, str(df_path) + "_vis.obj")


def export_distance_field(mesh_dir, output_path_lowres, output_path_highres, visualize=False):
    output_path_lowres.parents[0].mkdir(exist_ok=True, parents=True)
    output_path_highres.parents[0].mkdir(exist_ok=True, parents=True)
    failure_lr = subprocess.call(sdf_gen_lowres_cmd(str(mesh_dir) + ".obj", str(output_path_lowres)), shell=True)
    os.remove(str(output_path_lowres) + "_if.npy")
    failure_hr = subprocess.call(sdf_gen_highres_cmd(str(mesh_dir) + ".obj", str(output_path_highres)), shell=True)
    os.remove(str(output_path_highres) + "_if.npy")
    ratio = highres_dim / lowres_dim
    df_lowres = np.load(str(output_path_lowres)+".npy")
    df_highres = np.load(str(output_path_highres)+".npy")
    #assert df_lowres.shape == (lowres_dim, lowres_dim, lowres_dim)
    #assert df_highres.shape == (highres_dim, highres_dim, highres_dim)
    df_lowres_padded = df_lowres
    df_highres_padded = df_highres
    # new_shape_lowres = [lowres_dim] * 3
    # new_shape_highres = [highres_dim] * 3
    # df_highres_padded = np.ones(new_shape_highres) * df_highres.max()
    # df_lowres_padded = np.ones(new_shape_lowres) * df_lowres.max()

    # lower_idx = [(new_shape_highres[idx] - df_highres.shape[idx]) // 2 for idx in range(3)]
    # lower_idx = [0 for idx in range(3)]
    # upper_idx = [lower_idx[idx] + df_highres.shape[idx] for idx in range(3)]
    # df_highres_padded[lower_idx[0]:upper_idx[0], lower_idx[1]:upper_idx[1], lower_idx[2]:upper_idx[2]] = df_highres
    
    # df_lowres_padded = (torch.nn.functional.interpolate(torch.from_numpy(df_highres_padded * lowres_voxel_size / highres_voxel_size).float().unsqueeze(0).unsqueeze(0), size=new_shape_lowres, mode='trilinear', align_corners=True)).squeeze(0).squeeze(0).numpy()
    # lower_idx = [round((new_shape_highres[idx] - df_highres.shape[idx]) // 2 / ratio) for idx in range(3)]
    # lower_idx = [0 for idx in range(3)]
    # upper_idx = [lower_idx[idx] + df_lowres.shape[idx] for idx in range(3)]
    # df_lowres_padded[lower_idx[0]:upper_idx[0], lower_idx[1]:upper_idx[1], lower_idx[2]:upper_idx[2]] = df_lowres
    
    df_lowres_padded[np.abs(df_lowres_padded) > 3 * lowres_voxel_size] = 3 * lowres_voxel_size
    df_highres_padded[np.abs(df_highres_padded) > 3 * highres_voxel_size] = 3 * highres_voxel_size

    np.savez_compressed(str(output_path_lowres), arr=df_lowres_padded)
    np.savez_compressed(str(output_path_highres), arr=df_highres_padded)
    os.remove(str(output_path_lowres) + ".npy")
    os.remove(str(output_path_highres) + ".npy")
    if visualize:
        visualize_highres(output_path_highres)
        visualize_lowres(output_path_lowres)


def get_valid_objects(limit=None):
    splitsdir = Path("../Repatch3D/data/splits/ShapeNetV2/official")
    all_items = []
    for split in ["train.txt", "val.txt"]:
        all_items.extend((splitsdir/split).read_text().splitlines())
    return sorted(all_items[:limit])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--mesh_dir", type=str, default='mesh_dir')
    parser.add_argument("--df_lowres_dir", type=str, default='df_lowres')
    parser.add_argument("--df_highres_dir", type=str, default='df_highres')
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--proc', default=0, type=int)

    args = parser.parse_args()
    shapes = get_valid_objects(limit=None)
    shapes = [x for i, x in enumerate(shapes) if i % args.num_proc == args.proc]
    for p in [args.df_highres_dir, args.df_lowres_dir, args.mesh_dir]:
        Path(p).mkdir(exist_ok=True, parents=True)
    
    test_list = ["02828884__da39c2c025a9bd469200298427982555", "02933112__5172c96ea99b9f5fde533fa000314311", "03636649__1d2c2f3f398fe0ede6597d391ab6fcc1"]

    for shape in tqdm(shapes):
        shape = '/'.join(shape.split("__"))
        print("Processing: ", shape)
        shape_id = '__'.join(shape.split('/'))
        if shape_id not in test_list:
            continue
        shape_path = shapenet_dir / shape / "models" / "model_normalized.obj"
        mesh = trimesh.load(shape_path, force='mesh')
        if type(mesh) != trimesh.Trimesh:
            mesh = mesh.dump().sum()
        bbox = mesh.bounding_box.bounds
        loc = (bbox[0] + bbox[1])/2
        mesh.apply_translation(-loc)
        scale = (bbox[1] - bbox[0]).max()
        mesh.apply_scale(1 / scale)
        mesh.export(Path(args.mesh_dir) / (shape_id + '.obj'))
        export_distance_field(Path(args.mesh_dir) / shape_id, Path(args.df_lowres_dir) / f"{shape_id}", Path(args.df_highres_dir) / f"{shape_id}", visualize=True)
