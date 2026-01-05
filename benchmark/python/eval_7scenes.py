"""
    Reconstruction evaluation per sequence.
    
    For pose evaluation, please provide .npy files with following names:
    - pred_extrinsics.npy
    - gt_extrinsics.npy

    For pointmap evaluation, please provide .npy files with following names:
    - pred_point_map.npy / pred_point_map_by_unprojection.npy
    - gt_point_map.npy
    - gt_depth_masks.npy

"""

import argparse
import numpy as np

import torch
import torch_npu

from pathlib import Path

from lib.evaluation import eval_angle, eval_trajectory, eval_depth_map, eval_point_map

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--pred_extrinsics", type=str, default="pred_extrinsics.npy")
parser.add_argument("--pred_depth_maps", type=str, default="pred_depth_maps.npy")
parser.add_argument("--pred_point_maps", type=str, default="pred_point_maps.npy")
parser.add_argument("--gt_extrinsics", type=str, default="gt_extrinsics.npy")
parser.add_argument("--gt_depth_maps", type=str, default="gt_depth_maps.npy")
parser.add_argument("--gt_point_maps", type=str, default="gt_point_maps.npy")
parser.add_argument("--gt_depth_masks", type=str, default="gt_depth_masks.npy")
parser.add_argument("--pose_threshold", type=int, default=30)
parser.add_argument("--pointmap_threshold", type=float, default=0.1)
args = parser.parse_args()

input_dir = Path(args.input_dir)

pred_extrinsics = np.load(input_dir / args.pred_extrinsics)
pred_extrinsics = torch.from_numpy(pred_extrinsics)
gt_extrinsics = np.load(input_dir / args.gt_extrinsics)
gt_extrinsics = torch.from_numpy(gt_extrinsics)

pred_point_maps = np.load(input_dir / args.pred_point_maps)
pred_depth_maps = np.load(input_dir / args.pred_depth_maps)

gt_point_maps = np.load(input_dir / args.gt_point_maps)
gt_depth_maps = np.load(input_dir / args.gt_depth_maps)
gt_depth_masks = np.load(input_dir / args.gt_depth_masks)

# relative angle
rra_acc, rta_acc, auc_at_threshold = eval_angle(
    pred_extrinsics, 
    gt_extrinsics, 
    args.pose_threshold
)
print(rra_acc, rta_acc, auc_at_threshold)

# pose trajectory
ate, rpe_trans, rpe_rot = eval_trajectory(pred_extrinsics, gt_extrinsics)
print(ate, rpe_trans, rpe_rot)
exit()

# point map
acc, acc_med, nc1, nc1_med, comp, comp_med, nc2, nc2_med = eval_pointmap(
    pred_point_map, 
    gt_point_map, 
    gt_depth_masks, 
    args.pointmap_threshold
)
print(acc, acc_med, nc1, nc1_med, comp, comp_med, nc2, nc2_med)

# depth map
results = eval_depth_map(
    pred_depth_maps,
    gt_depth_maps,
    gt_depth_masks,
    align_with_lstsq=False,
    align_with_lad=False,
    align_with_lad2=False,
    align_with_scale=False,
    device="npu:1"
)
print(results)
