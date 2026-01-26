import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import torch_npu  
from tqdm import tqdm

from _bootstrap import BENCH_ROOT 

# -----------------------------------------------
# VGGT imports 
# -----------------------------------------------
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

# -----------------------------------------------
# Pi3 imports 
# -----------------------------------------------
from models.pi3_official.pi3.models.pi3 import Pi3

# -----------------------------------------------
# Reliev3R imports 
# -----------------------------------------------
from models.reliev3r.utils.config_loader import LoadConfig
from models.reliev3r.model import Reliev3R

# -----------------------------------------------
# dataset related
# -----------------------------------------------
from datasets import get_dataset

# -----------------------------------------------
# utils
# -----------------------------------------------
from lib.utils import save_npy, build_gt_geometry_data, fovs_to_K_norm,K_denormalize,rdepth_to_zdepth
from lib.utils import strip_ddp_prefix
from lib.normalization import normalize_camera_extrinsics_and_points_batch

# -----------------------------
# Config
# -----------------------------
# dataset
DATA_ROOT_DEFAULTS = {
    "7scenes": BENCH_ROOT / "data" / "7scenes" / "data",
    "dtu":     BENCH_ROOT / "data" / "DTU" / "spann3r_version" / "dtu_test_mvsnet_release",
    "nrgbd":   BENCH_ROOT / "data" / "NRGBD" / "data",
    "scannet": BENCH_ROOT / "data" / "scannet" / "posed_images",
    "scannetpp": BENCH_ROOT / "data" / "scannetppv2_val50",
    "replica": BENCH_ROOT / "data" / "Replica" ,
    "dl3dv":   "/mnt/nvme1/share/chenyouyu/dataset/DL3DV/data/1K/",
}
MODEL_DEFAULTS = {
    "vggt": {
        "ckpt": BENCH_ROOT / "checkpoints" / "VGGT-1B",
        "config": None,
    },
    "pi3": {
        "ckpt": BENCH_ROOT / "checkpoints" / "Pi3",
        "config": None,
    },
    "reliev3r": {
        "config": Path("/efs-mi-east4-2/exp/ssff3d/251213-450M-pi3Base-s32v8-MoGe2/config.json"),
        "ckpt":   Path("/efs-mi-east4-2/exp/ssff3d/251213-450M-pi3Base-s32v8-MoGe2/ckpt/60000.ckpt"),
    },
}

GT_OUTPUT_WIDTH = 518
DEPTH_FAR = 10.0
DEPTH_NEAR = 0.1

GT_FILES = [
    "gt_intrinsics.npy", 
    "gt_extrinsics.npy", 
    "gt_depth_maps.npy", 
    "gt_point_maps.npy", 
    "gt_depth_masks.npy", 
    "gt_images.npy"
]

# ---------------------- running vggt infer ------------------------------------------------
# def load_model_vggt(ckpt):
#     model = VGGT.from_pretrained(ckpt)
#     model.eval()
#     num_params = sum(p.numel() for p in model.parameters())
#     print(f"[VGGT] #params = {num_params/1e6:.2f} M")
#     return model


def load_model_vggt(ckpt: str):
    ckpt_p = Path(ckpt)
    if ("/" in ckpt and not ckpt_p.exists()) or ckpt_p.is_dir():
        model = VGGT.from_pretrained(ckpt) 
    else:
        model = VGGT()  
        obj = torch.load(str(ckpt_p), map_location="cpu")
        state_dict = obj.get("state_dict", obj) if isinstance(obj, dict) else obj

        fixed = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            k = k.replace("model.", "")
            fixed[k] = v

        missing, unexpected = model.load_state_dict(fixed, strict=False)
        print("missing:", missing)
        print("unexpected:", unexpected)

    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[VGGT] #params = {num_params/1e6:.2f} M")
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
    print(f"[Reliev3R] #params = {num_params/1e6:.2f} M")
    return model

def run_reliev3r_infer(
    model,
    images_cpu: torch.Tensor,          # (V,3,H,W) from load_and_preprocess_images, on CPU
    gt_intrinsics_rs: np.ndarray,       # (V,3,3) from build_gt_geometry_data first return
    gt_extrinsics: np.ndarray,
    device: torch.device,
    autocast_dtype: torch.dtype,
    image_paths=None,
    scene_name="7scenes",
    unproj_with_gt_extrinsics=False,
    unproj_with_gt_intrinsics=False,
):
    """
    Returns:
      pred_c2w: torch.Tensor (V,4,4)
      pred_depth_maps: torch.Tensor (V,H,W)    # meters
      pred_unproj: torch.Tensor (V,H,W,3)      # world points
    """
    if not torch.is_tensor(images_cpu) or images_cpu.ndim != 4:
        raise ValueError(f"images_cpu must be (V,3,H,W), got {type(images_cpu)} {getattr(images_cpu,'shape',None)}")

    V,C,H,W = images_cpu.shape

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

    ray_depth = pred["scaled_depths"].permute(0, 1, 3, 4, 2).contiguous()
    # depth_np = pred_depth.squeeze(0).squeeze(-1).detach().cpu().numpy()  # (V,H,W) np
    # depth_np, _ = filter_depth(depth_np, far=DEPTH_FAR, near=DEPTH_NEAR)  #
    # pred_depth = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)[None, ..., None]

    pred_w2c = pred["camera_extrs"]
    
    # GT intrinsics (V,3,3)
    gt_K = torch.from_numpy(gt_intrinsics_rs.astype(np.float32)).to(device=device, dtype=torch.float32)  # (V,3,3)

    ## predict intrinsic (1,8,2)
    fovs_pred = pred["camera_fovs"]      # (B,V,2)
    K_norm = fovs_to_K_norm(fovs_pred).squeeze(0)          # (B,V,3,3)
    pred_K  = K_denormalize(K_norm, W=W, H=H)

    ## raydepth to zdepth
    ray_depth = ray_depth.squeeze(0).permute(0,3,1,2)
    pred_depth = rdepth_to_zdepth(ray_depth,K_norm)
    pred_depth = pred_depth.unsqueeze(0).permute(0,1,3,4,2)

    # unprojection 
    # if unproj_with_gt:
    #     pred_unproj = unproject_depth_map_to_point_map(
    #         pred_depth.squeeze(0),
    #         gt_extrinsics,
    #         pred_K,
    #     ) 
    # else:
    #     pred_unproj = unproject_depth_map_to_point_map(
    #         pred_depth.squeeze(0),
    #         pred_w2c.squeeze(0),
    #         pred_K,
    #     )  
        
    pred_unproj = unproject_depth_map_to_point_map(
        pred_depth.squeeze(0),
        gt_extrinsics if unproj_with_gt_extrinsics else pred_w2c.squeeze(0),
        gt_K if unproj_with_gt_intrinsics else pred_K,
    )        

    # print(f"[Reliev3R] normalize_camera_extrinsics & unproj pointcloud")
    pred_w2c_aligned, _, pred_unproj_aligned, _ = normalize_camera_extrinsics_and_points_batch(
        extrinsics=torch.from_numpy(gt_extrinsics).unsqueeze(0)[:,:,:3].to(dtype=torch.float32) if unproj_with_gt_extrinsics else pred_w2c[:,:,:3].cpu(),
        world_points=torch.from_numpy(pred_unproj).to(dtype=torch.float32),
        scale_by_points = False,
    )
    return pred_w2c_aligned, pred_depth, pred_unproj_aligned.squeeze(0).numpy()

def dump_one_seq(model, 
                 model_type,
                 batch,
                 pred_seq_dir: Path,
                 gt_seq_dir: Path,
                 device: torch.device, 
                 autocast_dtype: torch.dtype,
                 unproj_with_gt_extrinsics: bool,
                 unproj_with_gt_intrinsics: bool,
                 ):
    
    def check_gt_data_dump(gt_seq_dir: Path, gt_files: list):
        flag = False # do not dump new GT data
        
        for file in gt_files:
            if not (gt_seq_dir / file).exists():
                flag = True # dump new GT data
        
        return flag
    
    # check if the GTs are already there
    dump_gt_flag = check_gt_data_dump(gt_seq_dir, GT_FILES)
    
    if dump_gt_flag:
        # --- build GT (resized + normalized) ---
        gt_intrinsics_rs, gt_extrinsics, _, gt_point_maps, gt_depth_maps, gt_depth_masks, gt_images = build_gt_geometry_data(
            image_pils=batch.image_pils, 
            depths=batch.depths, 
            raw_w2c=batch.raw_w2c,
            raw_K=batch.raw_K,
            output_width=GT_OUTPUT_WIDTH,
            depth_far=DEPTH_FAR,
            depth_near=DEPTH_NEAR,
            )
        
        gt_extrinsics = gt_extrinsics.squeeze(0)
        gt_c2w = closed_form_inverse_se3(gt_extrinsics)
        
        save_npy(gt_seq_dir, "gt_intrinsics.npy", gt_intrinsics_rs)
        save_npy(gt_seq_dir, "gt_extrinsics.npy", gt_c2w)
        save_npy(gt_seq_dir, "gt_depth_maps.npy", gt_depth_maps.squeeze(0))
        save_npy(gt_seq_dir, "gt_point_maps.npy", gt_point_maps.squeeze(0))
        save_npy(gt_seq_dir, "gt_depth_masks.npy", gt_depth_masks)
        save_npy(gt_seq_dir, "gt_images.npy", gt_images) 
    else: # load dumped GT data for inference
        gt_intrinsics_rs = np.load(gt_seq_dir / "gt_intrinsics.npy")
        gt_c2w = np.load(gt_seq_dir / "gt_extrinsics.npy")
        # ---- fix: convert loaded c2w back to w2c (extrinsics) ----
        gt_c2w = torch.from_numpy(gt_c2w)
        gt_extrinsics = closed_form_inverse_se3(gt_c2w).cpu().numpy() # for debug only

    # ---- infer
    with torch.inference_mode():
        if model_type == "vggt":
            pred_w2c, _, pred_depth_maps, pred_point_maps, pred_unproj = run_vggt_infer(
                model, batch.images_cpu, device=device, autocast_dtype=autocast_dtype
            )
            pred_c2w = closed_form_inverse_se3(pred_w2c.squeeze(0))

            save_npy(pred_seq_dir, "pred_extrinsics.npy", pred_c2w)
            save_npy(pred_seq_dir, "pred_depth_maps.npy", pred_depth_maps.squeeze(0).squeeze(-1))
            save_npy(pred_seq_dir, "pred_point_maps.npy", pred_point_maps.squeeze(0))
            save_npy(pred_seq_dir, "pred_point_maps_by_unprojection.npy", pred_unproj)

        elif model_type == "reliev3r":
            pred_w2c, pred_depth_maps, pred_unproj = run_reliev3r_infer(
                model=model,
                images_cpu=batch.images_cpu,                  # (V,3,H,W) from load_and_preprocess_images
                gt_intrinsics_rs=gt_intrinsics_rs,  # GT内参
                gt_extrinsics=gt_extrinsics,
                device=device,
                autocast_dtype=autocast_dtype,
                image_paths=batch.image_paths,
                scene_name=batch.seq,
                unproj_with_gt_extrinsics=unproj_with_gt_extrinsics,
                unproj_with_gt_intrinsics=unproj_with_gt_intrinsics,
            )
            pred_c2w = closed_form_inverse_se3(pred_w2c.squeeze(0))

            save_npy(pred_seq_dir, "pred_extrinsics.npy", pred_c2w)         # (V,4,4)
            save_npy(pred_seq_dir, "pred_depth_maps.npy", pred_depth_maps.squeeze(0).squeeze(-1))  # (V,H,W)
            save_npy(pred_seq_dir, "pred_point_maps_by_unprojection.npy", pred_unproj)

        elif model_type == "pi3":
            pred_w2c, pred_depth_maps, pred_point_maps = run_pi3_infer(
                model, batch.images_cpu, device=device, autocast_dtype=autocast_dtype
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
    p.add_argument("--dataset", type=str, default="7scenes", choices=list(DATA_ROOT_DEFAULTS.keys()))
    p.add_argument("--data_root", type=str, default=None,
               help="Dataset root path. If not set, inferred from --dataset.")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--seq", type=str, default="")      # chess/seq-01
    p.add_argument("--num_seq", type=int, default=10)   # 0 -> use all squences

    # view sample
    p.add_argument("--mode", type=str, default="stride", choices=["stride", "random"])
    p.add_argument("--stride", type=int, default=40)
    p.add_argument("--max_seq_len", type=int, default=8)  # stride: 截断数量; random: 采样数量(0=全量)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reuse_frame_ids", type=int, default=1)
    
    p.add_argument("--use_gt_pose", type=int, default=0) # project with GT extrinsics
    p.add_argument("--use_gt_k", type=int, default=0) # project with GT intrinsics
    
    # # model
    # p.add_argument("--model", type=str, default="vggt", choices=list(MODEL_DEFAULTS.keys()))
    # p.add_argument("--ckpt", type=str, default='/mnt/nvme1/share/chenyouyu/codespace/vggt_sz/epoch008-loss_valid0.2872-metric_valid0.2346.ckpt/epoch008-loss_valid0.2872-metric_valid0.2346.ckpt/pytorch_model.bin', help="(vggt/pi3/reliev3r) checkpoint path/dir. If not set, inferred from --model.")
    # p.add_argument("--config", type=str, default=None, help="(reliev3r) config.json. If not set, inferred from --model.")

    # p.add_argument("--model", type=str, default="pi3", choices=list(MODEL_DEFAULTS.keys()))
    # p.add_argument("--ckpt", type=str, default=None, help="(vggt/pi3/reliev3r) checkpoint path/dir. If not set, inferred from --model.")
    # p.add_argument("--config", type=str, default=None, help="(reliev3r) config.json. If not set, inferred from --model.")

    # # model debug
    p.add_argument("--model", type=str, default="reliev3r", choices=list(MODEL_DEFAULTS.keys()))
    p.add_argument("--ckpt", type=str, default='/mnt/nvme1/share/chenyouyu/codespace/Reliev3R-260108/exp/Reliev3R_260108_v8b2c32s1_n-1_rdepth_Ipred/ckpt/20000.ckpt', help="(vggt/pi3/reliev3r) checkpoint path/dir. If not set, inferred from --model.")
    p.add_argument("--config", type=str, default='/mnt/nvme1/share/chenyouyu/codespace/Reliev3R-260108/exp/Reliev3R_260108_v8b2c32s1_n-1_rdepth_Ipred/config.json', help="(reliev3r) config.json. If not set, inferred from --model.")

    # device
    p.add_argument("--device", type=str, default="npu:5")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])

    # output
    p.add_argument("--out_root", type=str, default=str(BENCH_ROOT / "outputs"))
    p.add_argument("--prefix", type=str, default="")
    args = p.parse_args()
    
    # ------------------------------------------------------------------------------------------------------
    # ---- auto match data_root after parsing ----
    if args.data_root is None:
        args.data_root = str(DATA_ROOT_DEFAULTS[args.dataset])
    else:
        args.data_root = str(Path(args.data_root))
    if not Path(args.data_root).exists():
        raise FileNotFoundError(f"data_root does not exist: {args.data_root}")
    # ---- auto match model ckpt/config after parsing ----
    mdef = MODEL_DEFAULTS[args.model]
    if args.ckpt is None:
        if mdef.get("ckpt") is None:
            raise ValueError(f"--ckpt is required for model={args.model} (no default configured).")
        args.ckpt = str(mdef["ckpt"])
    else:
        args.ckpt = str(Path(args.ckpt))

    if args.config is None:
        args.config = str(mdef["config"])
    else:
        args.config = str(Path(args.config))
    # ------------------------------------------------------------------------------------------------------

    device = torch.device(args.device)
    autocast_dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    # dump_gt = (args.dump_gt != 0)

    # unprojection flags
    unproj_with_gt_extrinsics = True if args.use_gt_pose != 0 else False
    unproj_with_gt_intrinsics = True if args.use_gt_k != 0 else False

    # dataset
    ds = get_dataset(args.dataset, args.data_root)
    stride = args.stride or ds.default_stride(args.view)
    sequences = [args.seq] if args.seq else ds.list_sequences(args.split)
    if args.num_seq > 0: sequences = sequences[:args.num_seq]

    # outdir
    setting_dir = Path(args.out_root) / args.dataset / (
        f"{args.split}_" +
        (f"stride{stride}_max{args.max_seq_len if args.max_seq_len>0 else 'all'}"
        if args.mode == "stride"
        else f"rand{args.max_seq_len if args.max_seq_len>0 else 'all'}_seed{args.seed}")
    )
    setting_dir.mkdir(parents=True, exist_ok=True)

    # load model by branch
    if args.model == "vggt":
        model = load_model_vggt(args.ckpt).to(device)
    elif args.model == "pi3":
        model = load_model_pi3(args.ckpt).to(device)
    elif args.model == 'reliev3r':
        if not args.ckpt:
            raise ValueError("--reliev3r_ckpt is required when --model reliev3r")
        model = load_model_reliev3r(args.config, args.ckpt, device=device, strict=True)
    else:
        raise ValueError(f"Unknown model_type={args.model}")

    print(
        "[INFO] "
        f"dataset={args.dataset} | split={args.split} | "
        f"model={args.model} | num_seqs={len(sequences)} | "
        f"mode={args.mode} | stride={stride} | max_seq_len={args.max_seq_len} | seed={args.seed} | "
        f"reuse_frame_ids={args.reuse_frame_ids} | device={device}\n"
        f"[INFO] use_gt_pose={args.use_gt_pose} use_gt_K={args.use_gt_k}\n"
        f"[INFO] data_root={args.data_root}\n"
        f"[INFO] out={setting_dir}"
    )

    pbar = tqdm(sequences, desc=args.dataset, unit="seq")
    for seq in pbar:
        name = args.model if not args.prefix else f"{args.model}_{args.prefix}"
        if unproj_with_gt_extrinsics: name += "_gt_pose"
        if unproj_with_gt_intrinsics: name += "_gt_K"
        pred_seq_dir = setting_dir / name / seq
        gt_seq_dir   = setting_dir / "gt" / seq
        pred_seq_dir.mkdir(parents=True, exist_ok=True)
        gt_seq_dir.mkdir(parents=True, exist_ok=True)
        
        seq_len = ds.count_seq_len(seq)
        frame_ids = ds.get_frame_ids(
            seq=seq,
            seq_len=seq_len,
            out_dir=setting_dir,
            mode=args.mode,                      # "stride" or "random"
            stride=stride,                       # stride
            max_seq_len=args.max_seq_len,        # stride: 截断; random: 采样数量
            seed=args.seed,
            reuse=(args.reuse_frame_ids != 0),
        )

        batch = ds.load_sequence(seq, frame_ids)

        dump_one_seq(
            model = model, 
            model_type = args.model,
            batch = batch,
            pred_seq_dir=pred_seq_dir,
            gt_seq_dir=gt_seq_dir,
            device=device,
            autocast_dtype=autocast_dtype,
            unproj_with_gt_extrinsics=unproj_with_gt_extrinsics,
            unproj_with_gt_intrinsics=unproj_with_gt_intrinsics,
        )

        pbar.set_postfix_str(f"{seq} K={len(frame_ids)}")
        tqdm.write(f"[OK] {seq}  K={len(frame_ids)}")


if __name__ == "__main__":
    main()
