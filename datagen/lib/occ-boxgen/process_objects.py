from argparse import ArgumentParser
from pathlib import Path
import subprocess
import numpy as np
import marching_cubes as mc
import trimesh
import os
import math
from tqdm import tqdm 


highres_dim = 64
lowres_dim = 8
padding_highres = math.ceil(highres_dim / lowres_dim)
padding_lowres = 1
highres_voxel_size = 2.6 / (highres_dim - 2 * padding_highres)
lowres_voxel_size = 2.6 / (lowres_dim - 2 * padding_lowres)

print(f"HighresVoxelRes: {highres_voxel_size}")
print(f"LowresVoxelRes: {lowres_voxel_size}")

sdf_gen_highres_cmd = lambda in_filepath, out_filepath: f"bin/sdf_gen {in_filepath} {out_filepath} {highres_voxel_size} {padding_highres}"
sdf_gen_lowres_cmd = lambda in_filepath, out_filepath: f"bin/sdf_gen {in_filepath} {out_filepath} {lowres_voxel_size} {padding_lowres}"


def get_valid_objects(mesh_dir, limit=None):
    list_of_scenes = Path(mesh_dir).iterdir()
    list_of_objects = []
    for scene in list_of_scenes:
        for room in scene.iterdir():
            objects = [Path(str(x).split(".")[0]) for x in room.iterdir() if x.name.endswith(".obj") and x.name != "mesh.obj"]
            list_of_objects.extend(objects)
    return list_of_objects[:limit]



def export_distance_field(object_path, output_path_lowres, output_path_highres, output_path_if, visualize=False):
    output_path_lowres.parents[0].mkdir(exist_ok=True, parents=True)
    output_path_highres.parents[0].mkdir(exist_ok=True, parents=True)
    failure_lr = subprocess.call(sdf_gen_lowres_cmd(str(object_path)+".obj", str(output_path_lowres)), shell=True)
    os.remove(str(output_path_lowres) + "_if.npy")
    failure_hr = subprocess.call(sdf_gen_highres_cmd(str(object_path)+".obj", str(output_path_highres)), shell=True)
    os.rename(str(output_path_highres) + "_if.npy", str(output_path_if) + ".npy")
    if visualize:
        visualize_highres(output_path_highres)
        visualize_lowres(output_path_lowres)


def visualize_highres_colored(df_path):
    df = np.load(str(df_path)+".npy")
    idf = np.load(str(df_path)+"_if.npy")
    colors = np.zeros([idf.shape[0], idf.shape[1], idf.shape[2], 3])
    colors[:, :, :, 0] = (idf[:, :, :] % 256) / 255
    colors[:, :, :, 1] = ((idf[:, :, :] // 256) % 256) / 255
    colors[:, :, :, 2] = ((idf[:, :, :] // 256 // 256) % 256) / 255
    vertices, triangles = mc.marching_cubes_color(df, colors, highres_voxel_size * 0.75)
    mc.export_obj(vertices, triangles, str(df_path) + "_vis.obj")

def visualize_highres(df_path):
    df = np.load(str(df_path)+".npy")
    vertices, triangles = mc.marching_cubes(df, highres_voxel_size * 0.75)
    mc.export_obj(vertices, triangles, str(df_path) + "_vis.obj")


def visualize_lowres(df_path):
    to_point_list = lambda s: np.concatenate([c[:, np.newaxis] for c in np.where(s == True)], axis=1)
    df = np.load(str(df_path)+".npy")
    point_list = to_point_list(df <= 0.5 * lowres_voxel_size)
    if point_list.shape[0] > 0:
        base_mesh = trimesh.voxel.ops.multibox(centers=point_list, pitch=1)
        base_mesh.export(str(df_path) + "_vis.obj")


def chunk_scene(df_path_lowres, df_path_highres, df_path_if, output_chunk_dir_lowres, output_chunk_dir_highres, output_chunk_dir_if, visualize=False):
    df_highres = np.load(str(df_path_highres)+".npy")
    if_highres = np.load(str(df_path_if)+".npy")
    df_lowres = np.load(str(df_path_lowres)+".npy")
    ratio = highres_dim / lowres_dim
    new_shape_highres = (np.ceil(np.array(df_highres.shape) / highres_dim) * highres_dim).astype(np.int32)
    new_shape_lowres = (np.array(new_shape_highres) / ratio).astype(np.int32)

    if not (new_shape_highres[0] <= highres_dim and new_shape_highres[1] <= highres_dim and new_shape_highres[2] <= highres_dim):
        return

    bound_hr_x = round((highres_dim - df_highres.shape[0]) / 2)
    bound_hr_y = round((highres_dim - df_highres.shape[1]) / 2)
    bound_hr_z = round((highres_dim - df_highres.shape[2]) / 2)
    bound_lr_x = round((highres_dim - df_highres.shape[0]) / 2 / ratio)
    bound_lr_y = round((highres_dim - df_highres.shape[1]) / 2 / ratio)
    bound_lr_z = round((highres_dim - df_highres.shape[2]) / 2 / ratio)
    
    # print(f"{df_highres.shape} -> {bound_hr_x}:{bound_hr_x + df_highres.shape[0]},{bound_hr_y}:{bound_hr_y + df_highres.shape[1]},{bound_hr_z}:{bound_hr_z + df_highres.shape[2]}")
    # print(f"{df_lowres.shape} -> {bound_lr_x}:{bound_lr_x + df_lowres.shape[0]},{bound_lr_y}:{bound_lr_y + df_lowres.shape[1]},{bound_lr_z}:{bound_lr_z + df_lowres.shape[2]}")

    df_highres_padded = np.ones(new_shape_highres) * df_highres.max()
    if_highres_padded = np.zeros(new_shape_highres)
    df_lowres_padded = np.ones(new_shape_lowres) * df_lowres.max()
    df_highres_padded[bound_hr_x:bound_hr_x + df_highres.shape[0],bound_hr_y:bound_hr_y + df_highres.shape[1], bound_hr_z:bound_hr_z + df_highres.shape[2]] = df_highres
    if_highres_padded[bound_hr_x:bound_hr_x + if_highres.shape[0],bound_hr_y:bound_hr_y + if_highres.shape[1], bound_hr_z:bound_hr_z + if_highres.shape[2]] = if_highres
    df_lowres_padded[bound_lr_x:bound_lr_x + df_lowres.shape[0],bound_lr_y:bound_lr_y + df_lowres.shape[1], bound_lr_z:bound_lr_z + df_lowres.shape[2]] = df_lowres
    stride_highres = int(new_shape_highres[1])
    stride_lowres = int(new_shape_lowres[1])
    for i in range(new_shape_highres[0] // stride_highres):
        for k in range(new_shape_highres[2] // stride_highres):
            xs_lr = i * stride_lowres
            xe_lr = (i + 1) * stride_lowres
            zs_lr = k * stride_lowres
            ze_lr = (k + 1) * stride_lowres
            xs_hr = i * stride_highres
            xe_hr = (i + 1) * stride_highres
            zs_hr = k * stride_highres
            ze_hr = (k + 1) * stride_highres
            filename = f"{df_path_highres.name}__{stride_highres:02d}__{xs_hr:03d}_{0:03d}_{zs_hr:03d}"
            np.save(output_chunk_dir_lowres / filename, df_lowres_padded[xs_lr:xe_lr, :, zs_lr:ze_lr])
            np.save(output_chunk_dir_highres / filename, df_highres_padded[xs_hr:xe_hr, :, zs_hr:ze_hr])
            np.save(output_chunk_dir_if / filename, if_highres_padded[xs_hr:xe_hr, :, zs_hr:ze_hr])
            if visualize:
                visualize_highres(output_chunk_dir_highres / filename)
                visualize_lowres(output_chunk_dir_lowres / filename)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--mesh_dir", type=str, default='outputs')
    parser.add_argument("--df_lowres_dir", type=str, default='df_lowres')
    parser.add_argument("--df_highres_dir", type=str, default='df_highres')
    parser.add_argument("--df_if_dir", type=str, default='df_if')
    parser.add_argument("--chunk_lowres_dir", type=str, default='chunk_lowres')
    parser.add_argument("--chunk_highres_dir", type=str, default='chunk_highres')
    parser.add_argument("--chunk_if_dir", type=str, default='chunk_if')
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--proc', default=0, type=int)

    args = parser.parse_args()
    valid_objects = get_valid_objects(args.mesh_dir)
    valid_objects = [x for i, x in enumerate(valid_objects) if i % args.num_proc == args.proc]
    
    for p in [args.df_highres_dir, args.df_lowres_dir, args.chunk_highres_dir, args.chunk_lowres_dir, args.df_if_dir, args.chunk_if_dir]:
        Path(p).mkdir(exist_ok=True, parents=True)
    
    for _object in tqdm(valid_objects):
        object_id = f"{_object.parents[1].name}__{_object.parents[0].name}__{_object.name}"
        print("Processing: ", _object)
        export_distance_field(_object, Path(args.df_lowres_dir) / f"{object_id}", Path(args.df_highres_dir) / f"{object_id}", Path(args.df_if_dir) / f"{object_id}", visualize=False)
        chunk_scene(Path(args.df_lowres_dir) / f"{object_id}", Path(args.df_highres_dir) / f"{object_id}", Path(args.df_if_dir) / f"{object_id}", Path(args.chunk_lowres_dir), Path(args.chunk_highres_dir), Path(args.chunk_if_dir), visualize=False)
