import torch
import torch_npu

import numpy as np
import open3d as o3d

import imageio
from PIL import Image
from pathlib import Path

from _bootstrap import BENCH_ROOT

# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("npu:0")
DTYPE = torch.float  

NUM_FRAMES = 10
INTERVAL = 40
ITERATOR = range(0, INTERVAL * NUM_FRAMES, INTERVAL)

# NOTE this is hard-coded and kept identical to pi3
# please check https://github.com/yyfz/Pi3/blob/evaluation/datasets/sevenscenes.py#L130
FX, FY, CX, CY = [525, 525, 320, 240]

# paths
CKPT = str(BENCH_ROOT / "checkpoints" / "VGGT-1B")
SEQ_ROOT = BENCH_ROOT / "samples" / "7scenes" / "data" / "chess" / "seq-01"
OUTPUT_ROOT = BENCH_ROOT / "outputs" / "7scenes" / "sample_seq"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# dataset-specific files
IMAGES = [str(SEQ_ROOT / f"frame-{s:06d}.color.png") for s in ITERATOR]
POSES  = [str(SEQ_ROOT / f"frame-{s:06d}.pose.txt")  for s in ITERATOR]
DEPTHS = [str(SEQ_ROOT / f"frame-{s:06d}.depth.proj.png") for s in ITERATOR]

# GT build params
GT_OUTPUT_WIDTH = 518
DEPTH_FAR = 10.0
DEPTH_NEAR = 1e-3
INVALID_DEPTH_RAW = 60000.0  # raw depth invalid threshold
DEPTH_SCALE = 1000.0         # mm -> m

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3
from training.data.dataset_util import depth_to_world_coords_points
from training.data.cropping import resize_image_depth_and_intrinsic
from training.train_utils.normalization import normalize_camera_extrinsics_and_points_batch

# -----------------------------
# Small helper function
# -----------------------------
def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x

def save_npy(root: Path, name: str, x) -> None:
    np.save(root / name, to_numpy(x))

def load_model(ckpt=CKPT):
    model = VGGT.from_pretrained(ckpt)
    model.eval()
    return model

def load_rgb(image_path_list=IMAGES):
    image_pils = [Image.open(p) for p in image_path_list]
    images = load_and_preprocess_images(image_path_list)  # torch tensor on CPU
    return image_pils, images

def load_depth(depth_path_list=DEPTHS, invalid_depth=INVALID_DEPTH_RAW, scale=DEPTH_SCALE):
    depths = np.stack([imageio.v2.imread(p).astype(np.float32) for p in depth_path_list])  # (S,H,W)
    depths[depths >= invalid_depth] = 0.0
    depths = depths / scale  # mm -> m
    return torch.from_numpy(depths)

def filter_depth(depthmaps, far=DEPTH_FAR, near=DEPTH_NEAR):
    # depthmaps: np.ndarray
    depthmaps[depthmaps > far] = 0.0
    depthmaps[depthmaps < near] = 0.0
    depth_masks = depthmaps != 0.0
    return depthmaps, depth_masks

def load_extrinsic(pose_path_list=POSES): # load the world-to-camera matrices
    # txt -> c2w(4x4) -> w2c(4x4)
    def parse_pose(pose_path): # get the camera-to-world matrices
        with open(pose_path) as f:
            lines = []
            for line in f.readlines():
                line = [float(l.lstrip().rstrip()) for l in line.lstrip().rstrip().split("\t")]
                lines.append(line)
        c2w = np.array(lines)
        return c2w
    
    c2w_arr = []
    for pose_path in pose_path_list:
        c2w = parse_pose(pose_path)
        c2w_arr.append(c2w)
    c2w_arr = torch.from_numpy(np.stack(c2w_arr)) # S, 4, 4
    w2c_arr = closed_form_inverse_se3(c2w_arr) # NOTE this is NOT lossless inversion
    return w2c_arr

def load_intrinsic(fx=FX, fy=FY, cx=CX, cy=CY):
    mat = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    mat = torch.from_numpy(mat).reshape(1, 3, 3).repeat(NUM_FRAMES, 1, 1)
    return mat

def build_gt_geometry_data(image_pils, depths, raw_extrinsics, raw_intrinsics,output_width=GT_OUTPUT_WIDTH):
    S = depths.shape[0]
    
    new_depthmaps, new_intrinsics, extrinsics_3x4, cam_points, world_points = [], [], [], [], []
    for i in range(S):
        image_pil = image_pils[i]
        depth_map = depths[i].numpy()
        extrinsic = raw_extrinsics[i, :3].numpy() # 3, 4
        intrinsic = raw_intrinsics[i].numpy() # 3, 3
        
        _, depth_map, intrinsic = resize_image_depth_and_intrinsic(
            image=image_pil,
            depth_map=depth_map,
            intrinsic=intrinsic,
            output_width=output_width
        )
        world_coords_points, cam_coords_points, _ = depth_to_world_coords_points(depth_map, extrinsic, intrinsic)
        
        new_depthmaps.append(depth_map)
        new_intrinsics.append(intrinsic)
        extrinsics_3x4.append(extrinsic)
        cam_points.append(cam_coords_points)
        world_points.append(world_coords_points)
        
    new_depthmaps = np.stack(new_depthmaps)
    new_intrinsics = np.stack(new_intrinsics)
    extrinsics_3x4 = np.stack(extrinsics_3x4)
    cam_points = np.stack(cam_points)
    world_points = np.stack(world_points)
    
    new_depthmaps, depth_masks = filter_depth(new_depthmaps)
        
    new_extrinsics, new_cam_points, new_world_points, new_depthmaps = normalize_camera_extrinsics_and_points_batch(
        extrinsics=torch.from_numpy(extrinsics_3x4).unsqueeze(0),
        cam_points=torch.from_numpy(cam_points).unsqueeze(0),
        world_points=torch.from_numpy(world_points).unsqueeze(0),
        depths=torch.from_numpy(new_depthmaps).unsqueeze(0),
        point_masks=torch.from_numpy(depth_masks).unsqueeze(0),
    )
    
    return new_intrinsics, new_extrinsics, new_cam_points, new_world_points, new_depthmaps, depth_masks        
        
def run_vggt_infer(model, images_cpu):
    # model inference interface
    images = images_cpu.unsqueeze(0).to(DEVICE)  # (B=1,...)

    # autocast model infer
    with torch.autocast(device_type=DEVICE.type, dtype=DTYPE):
        tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
            pose_enc, images.shape[-2:], is_se3=True
        )  # extrinsics B, S, 4, 4 - SE(3) # intrinsics B, S, 3, 3

        # Predict Depth Maps
        pred_depth_maps, _ = model.depth_head(tokens_list, images, ps_idx)
        # Predict Point Maps
        pred_point_maps, _ = model.point_head(tokens_list, images, ps_idx)

        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        pred_point_maps_by_unprojection = unproject_depth_map_to_point_map(
            pred_depth_maps.squeeze(0),
            pred_extrinsics.squeeze(0),
            pred_intrinsics.squeeze(0),
        )

    return pred_extrinsics, pred_intrinsics, pred_depth_maps, pred_point_maps, pred_point_maps_by_unprojection

    
def main():
    # load inputs
    image_pils, images = load_rgb()
    depths = load_depth()
    gt_intrinsics = load_intrinsic()
    gt_w2c = load_extrinsic()

    # bulid GT normalized
    gt_intrinsics_rs, gt_extrinsics, gt_cam_points, gt_point_maps, gt_depth_maps, gt_depth_masks = \
        build_gt_geometry_data(image_pils, depths, gt_w2c, gt_intrinsics)
    
    model = load_model().to(DEVICE)

    with torch.inference_mode():
        pred_w2c, pred_K, pred_depth_maps, pred_point_maps, pred_points_unproj = run_vggt_infer(model, images)

        # pose evaluation expects c2w
        # NOTE to-be-evaluated poses must be world-from-camera matrices
        pred_c2w = closed_form_inverse_se3(pred_w2c.squeeze(0))
        gt_c2w = closed_form_inverse_se3(gt_extrinsics.squeeze(0))

        # Save predictions
        save_npy(OUTPUT_ROOT, "pred_extrinsics.npy", pred_c2w)
        save_npy(OUTPUT_ROOT, "pred_depth_maps.npy", pred_depth_maps.squeeze(0))
        save_npy(OUTPUT_ROOT, "pred_point_maps.npy", pred_point_maps.squeeze(0))
        save_npy(OUTPUT_ROOT, "pred_point_maps_by_unprojection.npy", pred_points_unproj) 

        # Save GTs
        save_npy(OUTPUT_ROOT, "gt_extrinsics.npy", gt_c2w)
        save_npy(OUTPUT_ROOT, "gt_depth_maps.npy", gt_depth_maps.squeeze(0))
        save_npy(OUTPUT_ROOT, "gt_point_maps.npy", gt_point_maps.squeeze(0))
        save_npy(OUTPUT_ROOT, "gt_depth_masks.npy", gt_depth_masks)

    print("[OK] All results saved to:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()