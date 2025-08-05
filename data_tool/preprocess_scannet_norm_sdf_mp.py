#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
from joblib import Parallel,delayed
import os 
import logging
from joblib import parallel_backend

def setup_logger(log_path):
    logger = logging.getLogger("ScenePreprocess")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def get_processed_ids(output_root, split_name):
    folder = os.path.join(output_root, split_name)
    if not os.path.exists(folder):
        return set()
    # return {f.split('.')[0] for f in os.listdir(folder) if f.endswith('.pth')}
    return {f.split('.')[0] for f in os.listdir(folder) if f.endswith('.npz')}

def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    nf = np.cross(v01, v02)
    area = np.linalg.norm(nf, axis=1, keepdims=True) + 1e-8
    nf /= area
    return nf, area * 0.5

def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    weighted_nf = nf * area
    nv = np.zeros_like(vertex)
    np.add.at(nv, face[:, 0], weighted_nf)
    np.add.at(nv, face[:, 1], weighted_nf)
    np.add.at(nv, face[:, 2], weighted_nf)
    norm = np.linalg.norm(nv, axis=1, keepdims=True) + 1e-8
    return nv / norm

def normalize_vertices(mesh,boxmin=-1,boxmax=1,boxscale=0.95):
    import open3d as o3d
    # normalize vertices
    verts = np.array(mesh.vertices)
    shifts = (verts.max(axis=0) + verts.min(axis=0)) / 2
    verts -= shifts
    scale = (boxmax - boxmin) / (verts.max(axis=0) - verts.min(axis=0)).max()
    scale *= boxscale
    verts *= scale
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    return mesh

def safe_process(scene_id, dataset_root, sdf_label_root, seg_root, output_root, split, voxel_res, parse_normals, logger):
    try:
        process_one_scene(scene_id, dataset_root, sdf_label_root, seg_root ,output_root, split, voxel_res, parse_normals,logger)
        logger.info(f"{scene_id} processed successfully.")
    except Exception as e:
        logger.error(f"Failed processing {scene_id}: {e}")

def process_one_scene(sid, dataset_root, sdf_label_root, seg_root, output_root, split, voxel_res, parse_normals,logger):

    import open3d as o3d
    import numpy as np 
    import torch 

    ## mesh
    MESH_FILE_PFIX = "mesh_aligned_0.05"
    mesh_path = os.path.join(dataset_root, sid, f"scans/{MESH_FILE_PFIX}.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    mesh_norm = normalize_vertices(mesh,boxmin=-1,boxmax=1,boxscale=0.95)
    o3d.io.write_triangle_mesh(os.path.join(output_root, 'mesh_norm', f'{sid}.ply'), mesh_norm)

    coords = np.asarray(mesh_norm.vertices, dtype=np.float32)
    faces = np.asarray(mesh_norm.triangles)

    save_dict = {
        'scene_id': sid,
        'coord': coords,
    }

    ## seg
    seg_path = os.path.join(seg_root, split, f"{sid}.pth")
    seg_data = torch.load(seg_path)
    save_dict['segment'] = seg_data['segment']

    for res in voxel_res:
        sdf_label_path = os.path.join(sdf_label_root, f'sdf_{res}', sid, f"vs_{res}.npy")
        sdf_label = np.load(sdf_label_path).astype(np.float32).reshape(-1)

        min_val = float(sdf_label.min())
        max_val = float(sdf_label.max())
        mean_val = float(sdf_label.mean())
        perc_95 = float(np.percentile(sdf_label, 95))

        tqdm.write(
            f"[{sid} | vs_{res}] UDF stats: "
            f"min={min_val:.6f}, max={max_val:.6f}, "
            f"mean={mean_val:.6f}, 95%={perc_95:.6f}"
        )

        logger.info(
            f"[{sid} | vs_{res}] UDF stats: "
            f"min={min_val:.6f}, max={max_val:.6f}, "
            f"mean={mean_val:.6f}, 95%={perc_95:.6f}"
        )

        # 异常检测
        if not np.all(np.isfinite(sdf_label)):
            logger.warning(f"[{sid} | vs_{res}] contains NaN or Inf values.")
        elif np.all(sdf_label == 0):
            logger.warning(f"[{sid} | vs_{res}] UDF values are all zero.")
        elif max_val < 1e-4:
            logger.warning(f"[{sid} | vs_{res}] UDF max value is too small ({max_val:.6f}), possibly empty.")

        voxel_key = f"{res}"
        save_dict[f'pc_query_sdf_{voxel_key}'] = sdf_label

    if mesh.has_vertex_colors():
        save_dict['color'] = np.asarray(mesh.vertex_colors, dtype=np.float32)
    if parse_normals:
        save_dict['normal'] = vertex_normal(coords, faces)

    out_file = os.path.join(output_root, split, f"{sid}.pth")
    torch.save(save_dict, out_file)
    # out_file = os.path.join(output_root, split, f"{sid}.npz")

    # # 将所有值转换为 numpy array（如 segment 可能是 tensor）
    # for k, v in save_dict.items():
    #     if isinstance(v, torch.Tensor):
    #         save_dict[k] = v.cpu().numpy()
    #     elif isinstance(v, list):
    #         save_dict[k] = np.array(v)

    # np.savez_compressed(out_file, **save_dict)


if __name__ == "__main__":
    from multiprocessing import set_start_method
    set_start_method("spawn",force=True)

    dataset_root = '/data07/z00533370/dataset/scannetpp/data/'
    sdf_label_root = './scannetpp_volume_norm_-1_1_0.95/'
    seg_root = '/data07/z00533370/dataset/scannetpp_trainval_v3/'
    output_root = './scannetpp_norm_v3_03/'
    voxel_res = [16,32,64,128,256,512]
    parse_normals = True
    split_name = 'val'

    with open(f"/data07/z00533370/dataset/scannetpp/splits/nvs_sem_{split_name}.txt") as f:
        scene_ids = f.read().splitlines()

    os.makedirs(os.path.join(output_root, split_name), exist_ok=True)
    os.makedirs(os.path.join(output_root, "mesh_norm"), exist_ok=True)
    
    logger = setup_logger(os.path.join(output_root, f'{split_name}_preprocess.log'))

    done_ids = get_processed_ids(output_root, split_name)
    pending_ids = [sid for sid in scene_ids if sid not in done_ids]
    logger.info(f"{len(pending_ids)} scenes to process out of {len(scene_ids)} total.")

    try:
        with parallel_backend('loky'):
            Parallel(n_jobs=16)(
                delayed(safe_process)(scene_id, dataset_root, sdf_label_root, seg_root, output_root, split_name, voxel_res, parse_normals, logger)
                for scene_id in tqdm(pending_ids)
            )
    except KeyboardInterrupt:
        print("\n Interrupted by user. Exiting safely...")
        logger.warning("KeyboardInterrupt received. Preprocessing aborted.")