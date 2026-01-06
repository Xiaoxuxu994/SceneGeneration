import os
import json
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
import imageio
from PIL import Image
from tqdm import tqdm
from datetime import datetime

from _bootstrap import BENCH_ROOT  # noqa

# -----------------------------------------------
# VGGT imports 
# -----------------------------------------------
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

from training.data.dataset_util import depth_to_world_coords_points
from training.data.cropping import resize_image_depth_and_intrinsic
from training.train_utils.normalization import normalize_camera_extrinsics_and_points_batch

# -----------------------------------------------
# Pi3 imports 
# -----------------------------------------------
from pi3.models.pi3 import Pi3

# -----------------------------------------------
# Reliev3R imports 
# -----------------------------------------------
from utils.config_loader import LoadConfig
from model import Reliev3R


# -----------------------------
# Config
# -----------------------------
# NOTE this is hard-coded and kept identical to pi3
# please check https://github.com/yyfz/Pi3/blob/evaluation/datasets/sevenscenes.py#L130
FX, FY, CX, CY = [525, 525, 320, 240]
GT_OUTPUT_WIDTH = 518
DEPTH_FAR = 10.0
DEPTH_NEAR = 0.1
DEPTH_SCALE = 1000.0 # mm -> m

GT_FILES = [
    "gt_intrinsics.npy", 
    "gt_extrinsics.npy", 
    "gt_depth_maps.npy", 
    "gt_point_maps.npy", 
    "gt_depth_masks.npy", 
    "gt_images.npy"
]

# -----------------------------
# Small helper function
# -----------------------------
def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x

def save_npy(root: Path, name: str, x) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(str(root / name), to_numpy(x))

def strip_ddp_prefix(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    ks = list(state_dict.keys())
    if all(k.startswith("module.") for k in ks):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

def load_rgb(image_path_list):
    image_pils = [Image.open(p) for p in image_path_list]
    images = load_and_preprocess_images(image_path_list)  # torch tensor on CPU
    return image_pils, images

def load_depth(depth_path_list, invalid_depth, scale=DEPTH_SCALE):
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

def load_extrinsic(pose_path_list):
    # pose.txt: c2w(4x4) -> w2c(4x4)
    # c2w = np.stack([np.loadtxt(p).astype(np.float32) for p in pose_path_list])
    c2w = np.stack([np.loadtxt(p) for p in pose_path_list])
    c2w = torch.from_numpy(c2w)
    w2c = closed_form_inverse_se3(c2w)
    return w2c

def load_intrinsic(num_frames, fx=FX, fy=FY, cx=CX, cy=CY):
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return torch.from_numpy(K).reshape(1, 3, 3).repeat(num_frames, 1, 1)

def build_gt_geometry_data(image_pils, depths, raw_extrinsics, raw_intrinsics, output_width=GT_OUTPUT_WIDTH):
    S = depths.shape[0]
    new_depthmaps, extrinsics_3x4, cam_points, world_points = [], [], [], []
    new_intrinsics = []
    new_images = []

    for i in range(S):
        depth_map = depths[i].numpy()
        extrinsic = raw_extrinsics[i, :3].numpy()   # (3,4)
        intrinsic = raw_intrinsics[i].numpy()       # (3,3)

        img_rs, depth_map, intrinsic = resize_image_depth_and_intrinsic(
            image=image_pils[i],
            depth_map=depth_map,
            intrinsic=intrinsic,
            output_width=output_width
        )

        # save resized rgb (same H,W as depth_map)
        if img_rs.mode != "RGB":
            img_rs = img_rs.convert("RGB")
        new_images.append(np.asarray(img_rs, dtype=np.uint8))  # (H,W,3)

        world_pts, cam_pts, _ = depth_to_world_coords_points(depth_map, extrinsic, intrinsic)

        new_depthmaps.append(depth_map)
        new_intrinsics.append(intrinsic)
        extrinsics_3x4.append(extrinsic)
        cam_points.append(cam_pts)
        world_points.append(world_pts)

    new_depthmaps = np.stack(new_depthmaps)
    new_intrinsics = np.stack(new_intrinsics)
    extrinsics_3x4 = np.stack(extrinsics_3x4)
    cam_points = np.stack(cam_points)
    world_points = np.stack(world_points)
    new_images = np.stack(new_images, axis=0)

    new_depthmaps, depth_masks = filter_depth(new_depthmaps)

    new_extrinsics, new_cam_points, new_world_points, new_depthmaps = normalize_camera_extrinsics_and_points_batch(
        extrinsics=torch.from_numpy(extrinsics_3x4).unsqueeze(0),
        cam_points=torch.from_numpy(cam_points).unsqueeze(0),
        world_points=torch.from_numpy(world_points).unsqueeze(0),
        depths=torch.from_numpy(new_depthmaps).unsqueeze(0),
        point_masks=torch.from_numpy(depth_masks).unsqueeze(0),
    )

    return new_intrinsics, new_extrinsics, new_cam_points, new_world_points, new_depthmaps, depth_masks, new_images

# ---------------------- running vggt infer ------------------------------------------------
def load_model_vggt(ckpt):
    model = VGGT.from_pretrained(ckpt)
    model.eval()
    return model

def run_vggt_infer(model, images_cpu, device: torch.device, autocast_dtype: torch.dtype):
    images = images_cpu.unsqueeze(0).to(device)  # (B=1,...)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        tokens_list, ps_idx = model.aggregator(images)

        pose_enc = model.camera_head(tokens_list)[-1]
        pred_w2c, pred_K = pose_encoding_to_extri_intri(
            pose_enc, images.shape[-2:], is_se3=True
        )

        pred_depth, _ = model.depth_head(tokens_list, images, ps_idx)
        pred_point, _ = model.point_head(tokens_list, images, ps_idx)

        pred_unproj = unproject_depth_map_to_point_map(
            pred_depth.squeeze(0),
            pred_w2c.squeeze(0),
            pred_K.squeeze(0),
        )

    return pred_w2c, pred_K, pred_depth, pred_point, pred_unproj

# ---------------------- running p3 infer ------------------------------------------------
def load_model_pi3(ckpt):
    model = Pi3.from_pretrained(ckpt)
    model.eval()
    return model

def run_pi3_infer(model, images_cpu, device: torch.device, autocast_dtype: torch.dtype):
    images = images_cpu.unsqueeze(0).to(device)  # (B=1,...)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):        
        results = model(images)
        pred_depth = results["local_points"][..., -1].cpu().squeeze(0).numpy()
        pred_points = results["points"].cpu()
        pred_c2w = results["camera_poses"].cpu()

        # align to the first frame
        pred_w2c = closed_form_inverse_se3(pred_c2w.squeeze(0)).unsqueeze(0) # (1,V,4,4)
        pred_w2c_aligned, _, pred_points_aligned, _ = normalize_camera_extrinsics_and_points_batch(
            extrinsics=pred_w2c[:,:,:3],
            world_points=pred_points,
            scale_by_points=False,
        )
        
    return pred_w2c_aligned.squeeze(0).numpy(), pred_depth, pred_points_aligned.squeeze(0).numpy()

# ---------------------- running reliev3r infer ------------------------------------------
def load_model_reliev3r(config_path: str, ckpt_path: str, device: torch.device, strict: bool = True):
    cfg = LoadConfig(config_path)
    cfg.eval = True
    model = Reliev3R(cfg.model).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = strip_ddp_prefix(state)
    model.load_state_dict(state, strict=strict)
    print(f"[OK] Reliev3R loaded ckpt: {ckpt_path} (strict={strict})")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  [Reliev3R] #params = {num_params/1e6:.2f} M ({num_params:,})")
    return model

def run_reliev3r_infer(
    model,
    images_cpu: torch.Tensor,          # (V,3,H,W) from load_and_preprocess_images, on CPU
    gt_intrinsics_rs: np.ndarray,       # (V,3,3) from build_gt_geometry_data first return
    gt_extrinsics, # only ofr debugging
    device: torch.device,
    autocast_dtype: torch.dtype,
    image_paths=None,
    scene_name="7scenes",
):
    """
    Returns:
      pred_c2w: torch.Tensor (V,4,4)
      pred_depth_maps: torch.Tensor (V,H,W)    # meters
      pred_unproj: torch.Tensor (V,H,W,3)      # world points
    """
    if not torch.is_tensor(images_cpu) or images_cpu.ndim != 4:
        raise ValueError(f"images_cpu must be (V,3,H,W), got {type(images_cpu)} {getattr(images_cpu,'shape',None)}")

    V = images_cpu.shape[0]

    # -----------------------
    # build minimal batch
    # -----------------------
    if image_paths is not None:
        names = [osp.basename(p) for p in image_paths]
    else:
        names = ["frame"] * V

    data = {
        "scene_names":       [scene_name],
        "input_images":      images_cpu.to(device=device, dtype=torch.float32).unsqueeze(0),  # (1,V,3,H,W)
        "input_depths":      None,
        "sample_indices":    [list(range(V))],
        "input_viewmats":    None,
        "input_Ks":          None,
        "input_image_names": [names],
        "data_idxs":         [0],
        "matching_infos":    None,
    }

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        pred = model.forward(data)

    pred_depth = pred["scaled_depths"].permute(0, 1, 3, 4, 2).contiguous()
    pred_w2c = pred["camera_extrs"]

    ##
    print('========normalize extrinsic===========')
    pred_w2c, _, _, _ = normalize_camera_extrinsics_and_points_batch(
        extrinsics=pred_w2c[:,:,:3].cpu(),
        scale_by_points = False,
    )
    pred_w2c = pred_w2c.to(device.type)
    # -----------------------
    # GT intrinsics (V,3,3) -> torch
    # -----------------------
    pred_K = torch.from_numpy(gt_intrinsics_rs.astype(np.float32)).to(device=device, dtype=torch.float32)  # (V,3,3)

    # -----------------------
    # unprojection (use the SAME function as VGGT)
    # -----------------------
    pred_unproj = unproject_depth_map_to_point_map(
        pred_depth.squeeze(0),
        pred_w2c.squeeze(0),
        # gt_extrinsics.squeeze(0),
        pred_K,
    )  # (V,H,W,3) typically

    return pred_w2c, pred_depth, pred_unproj


# -----------------------------
# split/stride 7scene dataset
# -----------------------------
def list_sequences(sevenscenes_dir: str, split: str):
    split_txt = f"{split.capitalize()}Split.txt"
    scenes = sorted([d for d in os.listdir(sevenscenes_dir) if osp.isdir(osp.join(sevenscenes_dir, d))])

    seqs = []
    for scene in scenes:
        txt = osp.join(sevenscenes_dir, scene, split_txt)
        if not osp.exists(txt):
            continue
        with open(txt, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        for raw in lines:
            if raw.startswith("sequence"):
                sid = int(raw[len("sequence"):])
                seqs.append(f"{scene}/seq-{sid:02d}")
            else:
                seqs.append(f"{scene}/{raw}")
    return seqs


def count_seq_len(sevenscenes_dir: str, seq: str):
    seq_path = osp.join(sevenscenes_dir, seq)

    list_seqs_func = lambda suffix: sorted(
        [d for d in os.listdir(seq_path) if d.endswith(suffix)]
    )
    seq_rgbs = list_seqs_func(".color.png")
    seq_depths = list_seqs_func(".depth.proj.png")
    seq_poses = list_seqs_func(".pose.txt")

    if not (len(seq_rgbs) == len(seq_depths) == len(seq_poses)):
        raise ValueError(
            f"Sequence {seq} has mismatched number of RGB, depth, and pose files: "
            f"{len(seq_rgbs)}, {len(seq_depths)}, {len(seq_poses)}"
        )

    get_number_func = lambda x: int(x.split("-")[1].split(".")[0])

    if not (get_number_func(seq_rgbs[0]) == get_number_func(seq_depths[0]) == get_number_func(seq_poses[0]) == 0):
        raise ValueError(
            f"First frame of sequence {seq} has number of RGB, depth, and pose != 0: "
            f"{seq_rgbs[0]}, {seq_depths[0]}, {seq_poses[0]}"
        )

    if not (get_number_func(seq_rgbs[-1]) == get_number_func(seq_depths[-1]) == get_number_func(seq_poses[-1]) == len(seq_rgbs) - 1):
        raise ValueError(
            f"Last frame of sequence {seq} has number of RGB, depth, and pose != N-1: "
            f"{seq_rgbs[-1]}, {seq_depths[-1]}, {seq_poses[-1]}, N={len(seq_rgbs)}"
        )

    return len(seq_rgbs)



def make_paths(sevenscenes_dir: str, seq: str, frame_ids):
    seq_dir = osp.join(sevenscenes_dir, seq)
    images = [osp.join(seq_dir, f"frame-{i:06d}.color.png") for i in frame_ids]
    poses  = [osp.join(seq_dir, f"frame-{i:06d}.pose.txt")  for i in frame_ids]
    depths = [osp.join(seq_dir, f"frame-{i:06d}.depth.proj.png") for i in frame_ids]
    return images, poses, depths

def check_gt_data_dump(gt_seq_dir: Path):
    flag = False # do not dump new GT data
    
    for file in GT_FILES:
        if not (gt_seq_dir / file).exists():
            flag = True # dump new GT data
    
    return flag

def dump_one_seq(model, model_type, 
        image_paths, pose_paths, depth_paths,
        # paths_dict, # {frame_id: {"image_path": ..., "pose_path": ..., "depth_path": ...}}
        pred_seq_dir: Path,
        gt_seq_dir: Path,
        invalid_depth_raw: float,
        device: torch.device, autocast_dtype: torch.dtype):
    
    # image_paths = [d["image_path"] for _, d in paths_dict.items()]
    # depth_paths = [d["depth_path"] for _, d in paths_dict.items()]
    # pose_paths = [d["pose_path"] for _, d in paths_dict.items()]
    
    # load inputs
    image_pils, images = load_rgb(image_paths)
    
    dump_gt_flag = check_gt_data_dump(gt_seq_dir)
    
    if dump_gt_flag:
        # geometry GT info
        gt_intrinsics = load_intrinsic(num_frames=len(image_paths))
        gt_w2c = load_extrinsic(pose_paths)

        depths = load_depth(depth_paths, invalid_depth=invalid_depth_raw)
        gt_intrinsics_rs, gt_extrinsics, _, gt_point_maps, gt_depth_maps, gt_depth_masks, gt_images = \
            build_gt_geometry_data(image_pils, depths, gt_w2c, gt_intrinsics)

        gt_c2w = closed_form_inverse_se3(gt_extrinsics.squeeze(0))
        
        save_npy(gt_seq_dir, "gt_intrinsics.npy", gt_intrinsics_rs)
        save_npy(gt_seq_dir, "gt_extrinsics.npy", gt_c2w)
        save_npy(gt_seq_dir, "gt_depth_maps.npy", gt_depth_maps.squeeze(0))
        save_npy(gt_seq_dir, "gt_point_maps.npy", gt_point_maps.squeeze(0))
        save_npy(gt_seq_dir, "gt_depth_masks.npy", gt_depth_masks)
        save_npy(gt_seq_dir, "gt_images.npy", gt_images) 
    else: # load dumped GT data for inference
        gt_intrinsics_rs = np.load(gt_seq_dir / "gt_intrinsics.npy")
        gt_extrinsics = np.load(gt_seq_dir / "gt_extrinsics.npy")

    # ---- infer
    with torch.inference_mode():
        if model_type == "vggt":
            pred_w2c, _, pred_depth_maps, pred_point_maps, pred_unproj = run_vggt_infer(
                model, images, device=device, autocast_dtype=autocast_dtype
            )
            pred_c2w = closed_form_inverse_se3(pred_w2c.squeeze(0))

            save_npy(pred_seq_dir, "pred_extrinsics.npy", pred_c2w)
            save_npy(pred_seq_dir, "pred_depth_maps.npy", pred_depth_maps.squeeze(0).squeeze(-1))
            save_npy(pred_seq_dir, "pred_point_maps.npy", pred_point_maps.squeeze(0))
            save_npy(pred_seq_dir, "pred_point_maps_by_unprojection.npy", pred_unproj)

        elif model_type == "reliev3r":
            pred_w2c, pred_depth_maps, pred_unproj = run_reliev3r_infer(
                model=model,
                images_cpu=images,                  # (V,3,H,W) from load_and_preprocess_images
                gt_intrinsics_rs=gt_intrinsics_rs,  # GT内参
                gt_extrinsics=gt_extrinsics,        # only for debugging
                device=device,
                autocast_dtype=autocast_dtype,
                image_paths=image_paths,
                scene_name="7scenes",
            )
            pred_c2w = closed_form_inverse_se3(pred_w2c.squeeze(0))

            save_npy(pred_seq_dir, "pred_extrinsics.npy", pred_c2w)         # (V,4,4)
            save_npy(pred_seq_dir, "pred_depth_maps.npy", pred_depth_maps.squeeze(0).squeeze(-1))  # (V,H,W)
            save_npy(pred_seq_dir, "pred_point_maps_by_unprojection.npy", pred_unproj)

        elif model_type == "pi3":
            pred_w2c, pred_depth_maps, pred_point_maps = run_pi3_infer(
                model, images, device=device, autocast_dtype=autocast_dtype
            )
            pred_c2w = closed_form_inverse_se3(pred_w2c)

            save_npy(pred_seq_dir, "pred_extrinsics.npy", pred_c2w)         # (V,4,4)
            save_npy(pred_seq_dir, "pred_depth_maps.npy", pred_depth_maps)  # (V,H,W)
            save_npy(pred_seq_dir, "pred_point_maps.npy", pred_point_maps)  # (V,H,W,3)
        else:
            raise ValueError(f"Unknown model_type={model_type}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    # dataset
    p.add_argument("--sevenscenes_dir", type=str, default=str(BENCH_ROOT / "data" / "7scenes" / "data"))
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--view", type=str, default="dense", choices=["dense", "sparse"])
    p.add_argument("--stride", type=int, default=125)  # 0 -> paper default
    p.add_argument("--seq", type=str, default="")    # chess/seq-01

    # model
    p.add_argument("--model", type=str, default="reliev3r", choices=["vggt", "pi3", "reliev3r"])
    # vggt ckpt
    p.add_argument("--ckpt", type=str, default=str(BENCH_ROOT / "checkpoints" / "VGGT-1B"))
    # reliev3r config/ckpt
    # p.add_argument("--reliev3r_config", type=str, default="/efs-mi-east4-2/exp/ssff3d/251213-450M-pi3Base-s32v8-MoGe2/config.json")
    # p.add_argument("--reliev3r_ckpt", type=str, default="/efs-mi-east4-2/exp/ssff3d/251213-450M-pi3Base-s32v8-MoGe2/ckpt/60000.ckpt")

    p.add_argument("--reliev3r_config", type=str, default="/mnt/nvme1/share/chenyouyu/codespace/Reliev3R-251228/exp/Reliev3R_251231_stg1_v8c32s1_p50/config.json")
    p.add_argument("--reliev3r_ckpt", type=str, default="/mnt/nvme1/share/chenyouyu/codespace/Reliev3R-251228/exp/Reliev3R_251231_stg1_v8c32s1_p50/ckpt/50000.ckpt")

    p.add_argument("--device", type=str, default="npu:5")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--invalid_depth_raw", type=float, default=60000.0)
    # NOTE invalid_depth_raw in *.depth.proj.png in not necessarily 65535

    p.add_argument("--out_root", type=str, default=str(BENCH_ROOT / "outputs" / "7scenes" / "v6"))
    args = p.parse_args()

    device = torch.device(args.device)
    autocast_dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    stride = args.stride or (40 if args.view == "dense" else 200)

    sequences = [args.seq] if args.seq else list_sequences(args.sevenscenes_dir, args.split)

    # stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # setting_dir = Path(args.out_root) / f"{args.split}_{args.view}_stride{stride}_{stamp}"
    setting_dir = Path(args.out_root) / f"{args.split}_{args.view}_stride{stride}"
    setting_dir.mkdir(parents=True, exist_ok=True)

    # load model by branch
    if args.model == "vggt":
        model = load_model_vggt(args.ckpt).to(device)
    elif args.model == "pi3":
        model = load_model_pi3(args.ckpt).to(device)
    else:
        if not args.reliev3r_ckpt:
            raise ValueError("--reliev3r_ckpt is required when --model reliev3r")
        model = load_model_reliev3r(args.reliev3r_config, args.reliev3r_ckpt, device=device, strict=True)

    print(f"[INFO] model={args.model}, num_seqs={len(sequences)}, stride={stride}, device={device}, out={setting_dir}")


    pbar = tqdm(sequences, desc="7Scenes", unit="seq")
    for seq in pbar:
        pred_seq_dir = setting_dir / args.model / seq   #
        gt_seq_dir   = setting_dir / "gt" / seq
        
        pred_seq_dir.mkdir(parents=True, exist_ok=True)
        
        gt_data_paths = gt_seq_dir / "data_path.json"
        
        if gt_data_paths.exists():
            with open(gt_data_paths) as f:
                gt_paths_dict = json.load(f)
            
            # parse paths
            frame_ids, image_paths, pose_paths, depth_paths = [], [], [], []
            for f_id, d in gt_paths_dict.items():
                frame_ids.append(f_id)
                image_paths.append(d["image_path"])
                pose_paths.append(d["pose_path"])
                depth_paths.append(d["depth_path"])
        else:
            gt_seq_dir.mkdir(parents=True, exist_ok=True)
            
            seq_len = count_seq_len(args.sevenscenes_dir, seq)
            # get test frame IDs
            frame_ids = list(range(0, seq_len, stride))
            if not frame_ids:
                pbar.set_postfix_str(f"{seq}: empty")
                continue

            image_paths, pose_paths, depth_paths = make_paths(args.sevenscenes_dir, seq, frame_ids)

            gt_paths_dict = {f_id: {"image_path": i_path, "pose_path": p_path, "depth_path": d_path} \
                for f_id, i_path, p_path, d_path in zip(frame_ids, image_paths, pose_paths, depth_paths)}
            
            with open(gt_data_paths, "w") as f:
                json.dump(gt_paths_dict, f, indent=4)

        dump_one_seq(
            model, args.model, 
            image_paths, pose_paths, depth_paths,
            # gt_paths_dict,
            pred_seq_dir=pred_seq_dir,
            gt_seq_dir=gt_seq_dir,
            invalid_depth_raw=args.invalid_depth_raw,
            device=device,
            autocast_dtype=autocast_dtype,
        )

        pbar.set_postfix_str(f"{seq} K={len(frame_ids)}")
        tqdm.write(f"[OK] {seq}  K={len(frame_ids)}")


if __name__ == "__main__":
    main()