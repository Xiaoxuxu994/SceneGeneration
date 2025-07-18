from pathlib import Path
import trimesh
from argparse import ArgumentParser
import subprocess
import numpy as np
import marching_cubes as mc
import math
from tqdm import tqdm
import os
import shutil


quadriflow_path = "/rhome/ysiddiqui/quadriflow/quadriflow"

dims = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 128, 192, 224, 256, 272]

paddings = [2, ]
paddings = paddings + [int(math.ceil(x / dims[0]) * paddings[0]) for x in dims[1:]]
voxel_resolutions = [(1 / (dim - 2 * pad)) for dim, pad in zip(dims, paddings)]


def iso(dim):
    max_iso = 1
    min_iso = 1
    m = (max_iso - min_iso) / (dims[-1] - dims[0])
    c = max_iso - m * dims[-1]
    return m * dim + c


print('Dims:', dims)
print('Pads:', paddings)
print('Voxs:', voxel_resolutions)

sdf_gen_cmd = lambda inpath, outpath, voxres, pad: f"bin/sdf_gen_shapenet {inpath} {outpath} {voxres} {pad}"


def visualize_distance_field(df_path, vox_res, dim):
    df = np.load(df_path)
    vertices, triangles = mc.marching_cubes(df, vox_res * iso(dim))
    mc.export_obj(vertices, triangles, df_path.parent / (df_path.stem + ".obj"))
    mesh = trimesh.load(df_path.parent / (df_path.stem + ".obj"), process=False)
    mesh.apply_scale(dims[-1] / dim)
    mesh.export(df_path.parent / (df_path.stem + ".obj"))
    os.remove(df_path)
    os.remove(df_path.parent / "material.mtl")


def export_distance_field(mesh_path, visualize=False):
    mesh = trimesh.load(mesh_path, process=False)
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    scale = (mesh.bounds[1] - mesh.bounds[0]).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale)
    mesh.export(mesh_path)
    for dim, res, pad in zip(dims, voxel_resolutions, paddings):
        failure_lr = subprocess.call(sdf_gen_cmd(str(mesh_path), str(mesh_path.parent / f"{dim:03d}"), res, pad), shell=True)
        os.remove(str(mesh_path.parent / f"{dim:03d}") + "_if.npy")
        if visualize:
            visualize_distance_field(mesh_path.parent / f"{dim:03d}.npy", res, dim)


def center_and_normalize(mesh_dir):
    mesh = trimesh.load(mesh_dir / f"{dims[-1]:03d}.obj", process=False)
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    scale = (mesh.bounds[1] - mesh.bounds[0]).max()
    all_meshes = [x for x in mesh_dir.iterdir() if x.name.endswith('.obj') and x.name != 'model_normalized.obj' and not x.name.startswith("quad_")]
    for m in all_meshes:
        mesh = trimesh.load(m, process=False)
        mesh.apply_translation(-center)
        mesh.apply_scale(1 / scale)
        mesh.export(m)
    os.remove(mesh_dir / "material.mtl")


def quadriflow_executor(*args):
    tmpdir = Path(f'/tmp/{args[0][0]}')
    tmpdir.mkdir(exist_ok=True)
    cur_dir = os.getcwd()
    shutil.copy2(quadriflow_path, tmpdir / "quadriflow")
    try:
        os.chdir(str(tmpdir))
        failure_lr = subprocess.call(f'{"./quadriflow"} {args[0][1]}', shell=True)
        if failure_lr != 0:
            print("Retrying...")
            new_args = " ".join(args[0][1].split(" ")[1:])
            failure_lr = subprocess.call(f'{"./quadriflow"} {new_args}', shell=True)
            if failure_lr != 0:
                raise Exception(f"error code {failure_lr} for {new_args}")
    except Exception as err:
        print("ERROR:", err)
        Path(args[0][0] + '.txt').write_text(str(err))
    os.chdir(cur_dir)
    shutil.rmtree(tmpdir)


def quadriflow_wrapper(input_path):
    fcount = 24576
    res = dims[-3]
    output_filename = f'quadsdf_{fcount:05d}_{res:03d}'
    quad_dirname = f"{input_path.name}_{output_filename}"
    args = ["-i", f"{str(input_path / f'{res:03d}_sdf.obj')}", '-o', f"{str(input_path / output_filename)}.obj", "-f", f"{fcount}"]
    quadriflow_executor([quad_dirname, ' '.join(args)])


def single_df_export(mesh_path):
    dim = dims[-3]
    res = voxel_resolutions[-3]
    pad = paddings[-3]
    failure_lr = subprocess.call(sdf_gen_cmd(str(mesh_path), str(mesh_path.parent / f"{dim:03d}"), res, pad), shell=True)
    os.remove(str(mesh_path.parent / f"{dim:03d}") + "_if.npy")
    df = np.load(str(mesh_path.parent / f"{dim:03d}") + ".npy")
    os.remove(str(mesh_path.parent / f"{dim:03d}") + ".npy")
    df[df > 5 * res] = 5 * res
    df[df < -5 * res] = -5 * res
    vertices, triangles = mc.marching_cubes(df, res * iso(dim))
    mc.export_obj(vertices, triangles, mesh_path.parent / f"{dim:03d}_sdf.obj")
    mesh = trimesh.load(mesh_path.parent / f"{dim:03d}_sdf.obj", process=False)
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    scale = (mesh.bounds[1] - mesh.bounds[0]).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale)
    mesh.export(mesh_path.parent / f"{dim:03d}_sdf.obj")
    quadriflow_wrapper(mesh_path.parent)
    

    # np.save(str(mesh_path.parent / f"{dim:03d}") + ".npy", df)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str)
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    args = parser.parse_args()
    files = sorted([x for x in Path(args.input_folder).iterdir()])
    files = [x for i, x in enumerate(files) if i % args.num_proc == args.proc]
    #files = [Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs-manifold-highres/shape02349_rank01_pair15882"),
    #         Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs-manifold-highres/shape02317_rank01_pair35802")]
    
    for f in tqdm(files):
        # export_distance_field(f / "model_normalized.obj", True)
        # center_and_normalize(f)
        single_df_export(f / "model_normalized.obj")
