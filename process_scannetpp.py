import os
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import argparse


# ---------- HELPER FUNCTIONS ----------
def parse_colmap_images_txt(path):
    """
    Parse COLMAP images.txt file to extract extrinsics.
    Returns dict: {image_name: (qvec, tvec)}
    """
    extrinsics = {}
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        image_name = parts[9]
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        extrinsics[image_name] = ((qw, qx, qy, qz), (tx, ty, tz))
        i += 2  # skip the following POINTS2D line
    return extrinsics


def qvec_tvec_to_w2c(qvec, tvec):
    """
    Convert COLMAP quaternion+translation to 4x4 world-to-camera matrix.
    """
    rot = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()  # COLMAP uses [qw, qx, qy, qz]
    t = np.array(tvec).reshape(3, 1)
    w2c = np.eye(4)
    w2c[:3, :3] = rot
    w2c[:3, 3] = t.flatten()
    return w2c.tolist()


def load_intrinsics(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [data["fl_x"], data["fl_y"], data["cx"], data["cy"]]


# ---------- MAIN ----------
def make_scene_metadata(data_root, output_metadata_dir, scene_id):
    scene_dir = os.path.join(data_root, "data", scene_id)
    colmap_path = os.path.join(scene_dir, colmap_rel_path)
    nerf_path = os.path.join(scene_dir, nerfstudio_rel_path)
    images_dir = os.path.join(scene_dir, images_rel_dir)
    metadata_out = os.path.join(output_metadata_dir, f"{scene_id}.json")

    extrinsics = parse_colmap_images_txt(colmap_path)
    intrinsics = load_intrinsics(nerf_path)

    frames = []
    for img_name, (qvec, tvec) in extrinsics.items():
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Missing image {img_path}")
            continue
        w2c = qvec_tvec_to_w2c(qvec, tvec)
        frames.append({
            "image_path": os.path.abspath(img_path),
            "fxfycxcy": intrinsics,
            "w2c": w2c
        })

    meta = {
        "scene_name": scene_id,
        "frames": frames
    }

    with open(metadata_out, "w") as f:
        json.dump(meta, f, indent=4)
    return metadata_out


def process_split(data_root, output_metadata_dir, split_name):
    in_split = os.path.join(f"data/scannetpp/splits/hq_nvs_{split_name}.txt")
    out_split = os.path.join(data_root, "splits", f"hq_lvsm_{split_name}.txt")

    with open(in_split, "r") as f:
        scene_ids = [x.strip() for x in f.readlines() if x.strip()]

    abs_paths = []
    for scene_id in tqdm(scene_ids):
        meta_path = make_scene_metadata(data_root, output_metadata_dir, scene_id)
        abs_paths.append(os.path.abspath(meta_path))

    with open(out_split, "w") as f:
        f.write("\n".join(abs_paths))
    print(f"✅ Wrote {out_split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='Process ScanNet++v2')
    parser.add_argument('--base_path', type=str, required=True)
    args = parser.parse_args()
    
    colmap_rel_path = "dslr/colmap/images.txt"
    nerfstudio_rel_path = "dslr/nerfstudio/transforms_undistorted.json"
    images_rel_dir = "dslr/resized_undistorted_images"
    output_metadata_dir = os.path.join(args.base_path, "LVSM_metadata")
    os.makedirs(output_metadata_dir, exist_ok=True)
    
    process_split(args.base_path, output_metadata_dir, "train")
    process_split(args.base_path, output_metadata_dir, "val")
