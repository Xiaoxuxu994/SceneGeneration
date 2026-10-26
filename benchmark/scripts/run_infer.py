import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import torch_npu  
import torchvision.transforms as TF
from tqdm import tqdm
from omegaconf import OmegaConf
from safetensors.torch import load_file
import subprocess
try:  # only needed by the commented-out ZeRO conversion below; checkpoints load without it
    import deepspeed  # noqa: F401
except ImportError:
    deepspeed = None

from _bootstrap import BENCH_ROOT 

# -----------------------------------------------
# VGGT imports 
# -----------------------------------------------
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
# from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

# -----------------------------------------------
# VGGT-Ω imports 
# -----------------------------------------------
from models.vggt_omega.models.vggt_omega import VGGTOmega
from models.vggt_omega.utils.pose_enc import encoding_to_camera

# import sys
# try:
#     import numpy._core
# except ImportError:
#     import numpy.core
#     sys.modules['numpy._core'] = numpy.core

# -----------------------------------------------
# Pi3 imports 
# -----------------------------------------------
# from models.pi3_official.pi3.models.pi3 import Pi3 ## official
from models.pi3.models.pi3 import Pi3 ## ours
from models.pi3loop.models.pi3_training import Pi3 as Pi3Train  ## SceneGeneration-trained Pi3 (pi3loop)
from models.pi3loop.models.pi3_loop import Pi3Loop             ## SceneGeneration loop decoder (pi3loop)
from transformers.modeling_utils import load_sharded_checkpoint

# -----------------------------------------------
# Reliev3R imports 
# -----------------------------------------------
from models.reliev3r.utils.config_loader import LoadConfig
from models.reliev3r.model.reliev3r import Reliev3R, Reliev3R_Pi3
# from models.reliev3r.submodules.pi3.models.pi3 import Pi3 as Reliev3R_Pi3

# -----------------------------------------------
# DA3 imports 
# -----------------------------------------------
from models.depth_anything_3.api import DepthAnything3

# -----------------------------------------------
# dataset related
# -----------------------------------------------
from datasets import get_dataset, get_dataset_config

# -----------------------------------------------
# utils
# -----------------------------------------------
from lib.utils import save_npy, build_gt_geometry_data, fovs_to_K_norm,K_denormalize,rdepth_to_zdepth
from lib.utils import strip_ddp_prefix
from lib.normalization import normalize_camera_extrinsics_and_points_batch
from lib.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3, unproject_local_point_map_to_world_point_map

# -----------------------------
# Config
# -----------------------------
# dataset
DATA_ROOT_DEFAULTS = {
    "7scenes": BENCH_ROOT / "data" / "7scenes" / "data",
    "dtu":     BENCH_ROOT / "data" / "DTU" / "spann3r_version" / "dtu_test_mvsnet_release",
    "eth3d":   BENCH_ROOT / "data" / "ETH3D" / "multi_view_training_dslr",
    "nrgbd":   BENCH_ROOT / "data" / "NRGBD" / "data",
    "scannet": BENCH_ROOT / "data" / "scannet" / "posed_images",
    "scannetpp": BENCH_ROOT / "data" / "scannetppv2_val50",
    "scannetpp_r3": BENCH_ROOT / "data" / "scannetpp_r3",   # reliev3r layout; override with --data_root
    # "dl3dv":   "/mnt/nvme1/share/chenyouyu/dataset/DL3DV/data/1K/",
    "dl3dv":   "/efs-m3ga/z00818731/dl3dv-new/reliev3r_sample_list_c32.npz",
    "ase":     BENCH_ROOT / "data" / "ASE" /"reliev3r_sample_list_c32_s1.npz.update.npz",
    "re10k":   "/efs-m3ga/h00548084/relive3r_dataset/re10k_12.19/reliev3r_sample_list_c32.npz",
    "sekai":   "/efs-m3ga/c50041401/datasets/sekai/sekai_clips_all/sub_000009/",
    "kitti":   BENCH_ROOT / "data" / "KITTI",
    "sintel": BENCH_ROOT / "data" / "MPI-Sintel",
    "bonn": BENCH_ROOT / "data" / "bonn",
    "tum": BENCH_ROOT / "data" / "tum",
    "spatialvid":   "/efs-m3ga/c50041401/datasets/SpatialVID/reliev3r_clips_all/reliev3r_sample_list_c32.npz",
    "hiroom": BENCH_ROOT / "data" / "hiroom",
}
MODEL_DEFAULTS = {
    "vggt": {
        "ckpt": BENCH_ROOT / "ckpts" / "VGGT-1B",
        "config": None,
    },
    "vggt_omega": {
        "ckpt": Path("/efs-mi-east4-2/w60114355/checkpoints/modelscope.cn/models/sanqilu376/VGGT-Omega/resolve/master/vggt_omega_1b_512.pt"),
        "config": None,
    },
    "pi3": {
        "ckpt": BENCH_ROOT / "ckpts" / "Pi3",
        "config": None,
    },
    "reliev3r": {
        "config": Path("/efs-mi-east4-2/exp/ssff3d/251213-450M-pi3Base-s32v8-MoGe2/config.json"),
        "ckpt":   Path("/efs-mi-east4-2/exp/ssff3d/251213-450M-pi3Base-s32v8-MoGe2/ckpt/60000.ckpt"),
    },
    "da3": {
        "ckpt": BENCH_ROOT / "ckpts" / "depthanything3",
        "config": None,
    },
    "pi3loop": {
        "ckpt": None,   # --ckpt required
        "config": None,
    },
}

# GT_OUTPUT_WIDTH = 518
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
        if (ckpt_p / "model.safetensors").exists():
            model = VGGT.from_pretrained(ckpt)
        else:
            model = VGGT()
            actual_ckpt_p = ckpt_p / "mp_rank_00_model_states.pt"
            if actual_ckpt_p.exists():
                checkpoint = torch.load(actual_ckpt_p, map_location="cpu")
                if 'module' in checkpoint:
                    state_dict = checkpoint['module']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
            else:
                raise NotImplementedError
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
        # print("missing:", missing)
        # print("unexpected:", unexpected)

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


# ---------------------- running vggt-Omega infer ------------------------------------------------
def load_model_vggt_omega(ckpt, enable_alignment=False):
    model = VGGTOmega(enable_alignment=enable_alignment)
    
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_file():
        candidate = ckpt_path / "model.pt"
        if candidate.exists():
            ckpt_path = candidate
        else:
            raise ValueError(f"Cannot find valid checkpoint in {ckpt}")
            
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    
    if 'module' in checkpoint:
        state_dict = checkpoint['module']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[VGGT-Omega] #params = {num_params/1e6:.2f} M")
    
    return model

def run_vggt_omega_infer(model, images_cpu, device: torch.device, autocast_dtype: torch.dtype):
    images = images_cpu.unsqueeze(0).to(device)  # (B=1,...)

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):        
            predictions = model(images)

            pred_depth = predictions["depth"]

            pred_w2c, pred_K = encoding_to_camera(
                predictions["pose_enc"],
                images.shape[-2:],
            )

            pred_unproj = unproject_depth_map_to_point_map(
                pred_depth.squeeze(0),
                pred_w2c.squeeze(0),
                pred_K.squeeze(0),
            )

    return pred_w2c, pred_K, pred_depth, pred_unproj

# ---------------------- running p3 infer ------------------------------------------------
# def load_model_pi3(ckpt):
#     model = Pi3.from_pretrained(ckpt)
#     model.eval()
#     return model

def load_model_pi3(
    ckpt,
    decoder_size='large',
    model_size='dinov3_huge+_3Bv2'
):
    # ckpt_p = Path(ckpt)
    # weight_path = ckpt_p / "pytorch_model.bin" if ckpt_p.is_dir() else ckpt_p
    
    model = Pi3(
        decoder_size=decoder_size,
        model_size=model_size
    )
    
    if ckpt.is_dir():
        ckpt_path = ckpt / "mp_rank_00_model_states.pt"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            if 'module' in checkpoint:
                state_dict = checkpoint['module']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
        else:
            print('================ Pi3 Model ================================')
            print(model)
            load_result = load_sharded_checkpoint(model, ckpt, strict=False)
            print(load_result)
            print('===========================================================')
    else:
        if "safetensors" in str(ckpt):
            state_dict = load_file(ckpt)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("checkpoint is not supported.")

    # state_dict = torch.load(str(weight_path), map_location="cpu")
    # if isinstance(state_dict, dict) and "state_dict" in state_dict:
    #     state_dict = state_dict["state_dict"]

    # fixed = {}
    # for k, v in state_dict.items():
    #     if k.startswith("module."):
    #         k = k[len("module."):]
    #     if k.startswith("model."):
    #         k = k[len("model."):]
    #     fixed[k] = v

    # msg = model.load_state_dict(fixed, strict=False)
    # print("[Pi3] load_state_dict:", msg)

    model.eval()
    return model


def run_pi3_infer(model, images_cpu, depth_masks, device: torch.device, autocast_dtype: torch.dtype):
    images = images_cpu.to(device).unsqueeze(0)  # (B=1,...)
    # depth_masks = torch.from_numpy(depth_masks).to(device=device, dtype=torch.float32)

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
            # point_masks=depth_masks.unsqueeze(0)
        )
        
    return pred_w2c_aligned.squeeze(0).numpy(), pred_depth, pred_points_aligned.squeeze(0).numpy()


# ---------------------- running SceneGeneration-trained Pi3 / loop models (pi3loop) ------------------------
# models/pi3loop = pi3/ of SceneGeneration branch xiaoxu/Pi3_training_loop_transformer (pi3_training.Pi3,
# pi3_loop.Pi3Loop); the inference / output path is the same as for "pi3" (run_pi3_infer).
def load_model_pi3loop(ckpt, model_size, loop_layers=0, loop_k=0):
    """loop_layers == 0 -> pi3_training.Pi3 with the given spec (e.g. dinov3_small_w384; dinov2_large_0.95B loads
    the official Pi3 weights, confidence head ignored); loop_layers > 0 -> Pi3Loop with that many (frame, global)
    block pairs applied loop_k times. ckpt: accelerate / DeepSpeed checkpoint dir (checkpoint_N, best_model or its
    pytorch_model/), mp_rank_00_model_states.pt, .safetensors or a plain state dict; the encoder comes from it."""
    orig = Pi3Train._build_backbone_cfg

    def no_download():
        cfg = orig()
        for enc in cfg.values():
            for spec in enc.values():
                b = spec["builder"]
                spec["builder"] = (lambda b: (lambda pretrained=True, model_dir=None: b(pretrained=False)))(b)
        return cfg

    Pi3Train._build_backbone_cfg = staticmethod(no_download)
    try:
        kw = dict(pos_type="rope100", decoder_size="large", model_size=model_size, load_vggt=False,
                  freeze_encoder=False, use_global_points=False, train_conf=False,
                  num_dec_blk_not_to_checkpoint=0, ckpt=None)
        if loop_layers > 0:
            assert loop_k > 0, "--loop_k is required for a loop model (--loop_layers > 0)"
            model = Pi3Loop(**kw, loop_layers=loop_layers, loop_k_min=loop_k, loop_k_max=loop_k, loop_k_infer=loop_k)
        else:
            model = Pi3Train(**kw)
    finally:
        Pi3Train._build_backbone_cfg = staticmethod(orig)

    p = Path(ckpt)
    res = None
    if p.is_dir() and any(p.glob("*.index.json")):
        # sharded checkpoint (e.g. .../ckpts/model_ckpt78), as loaded by pi3_training.Pi3(ckpt=<dir>) in training
        res = load_sharded_checkpoint(model, str(p), strict=False)
    elif p.is_dir():
        cands = [p / "pytorch_model" / "mp_rank_00_model_states.pt", p / "mp_rank_00_model_states.pt",
                 p / "model.safetensors", p / "pytorch_model.bin"]
        p = next((c for c in cands if c.exists()), None)
        if p is None:
            raise FileNotFoundError(f"no weights under {ckpt} (tried *.index.json, {[str(c) for c in cands]})")
    if res is not None:
        pass
    elif p.suffix == ".safetensors":
        state_dict = load_file(str(p))
    else:
        try:
            obj = torch.load(str(p), map_location="cpu", weights_only=False)
        except ModuleNotFoundError as e:  # objects pickled by the training framework (e.g. deepspeed)
            raise ModuleNotFoundError(f"loading {p} needs the Python package of the training framework: {e}. "
                                      f"Install it (e.g. pip install deepspeed) and re-run.") from e
        state_dict = obj.get("module", obj.get("model", obj)) if isinstance(obj, dict) else obj
    if res is None:
        state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
        res = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in res.missing_keys if k != "pos_special"]          # constant buffer, absent in official Pi3
    unexpected = [k for k in res.unexpected_keys if not k.startswith(("conf_decoder.", "conf_head."))]
    if missing or unexpected:
        raise RuntimeError(f"[pi3loop] {p} does not match {model_size} (loop_layers={loop_layers}): "
                           f"missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)})")
    print(f"[pi3loop] loaded {p}: {model_size}, loop_layers={loop_layers}, loop_k={loop_k}, "
          f"{sum(x.numel() for x in model.parameters()) / 1e6:.2f}M params")
    model = model.float().eval()
    return model

# ---------------------- running da3 infer ------------------------------------------------
def load_model_da3(ckpt):
    model = DepthAnything3.from_pretrained(ckpt)
    model.eval()
    return model

def run_da3_infer(model, images_cpu, depth_masks, device: torch.device, autocast_dtype: torch.dtype):
    images = images_cpu.to(device).unsqueeze(0)  # (B=1,...)
    images = TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(images)
    # depth_masks = torch.from_numpy(depth_masks).to(device=device, dtype=torch.float32)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):        
        results = model(images)
        pred_depth = results["depth"].cpu() # (B, V, H, W)
        pred_w2c = results["extrinsics"].cpu() # (B, V, 3, 4)
        pred_K = results["intrinsics"].cpu() # (B, V, 3, 3)
        
        pred_points = unproject_depth_map_to_point_map(
            pred_depth.squeeze(0).unsqueeze(-1),
            pred_w2c.squeeze(0),
            pred_K.squeeze(0)
        ) # (V, H, W, 3)

        # align to the first frame
        pred_w2c_aligned, _, pred_points_aligned, _ = normalize_camera_extrinsics_and_points_batch(
            extrinsics=pred_w2c.double(),
            world_points=torch.from_numpy(pred_points).unsqueeze(0),
            scale_by_points=False,
            # point_masks=depth_masks.unsqueeze(0)
        )
        
    return pred_w2c_aligned.squeeze(0).numpy(), pred_depth.squeeze(0).numpy(), pred_points_aligned.squeeze(0).numpy()

# ---------------------- running reliev3r infer ------------------------------------------
def load_model_reliev3r(config_path: str, ckpt_path: str, device: torch.device, strict: bool = True):
    cfg = LoadConfig(config_path)
    cfg.eval = True
    
    if cfg.get("train_pi3", False):
        os.environ['pred_z_depth'] = "True"
        model = Reliev3R_Pi3(cfg.model).to(device).eval()
    else:
        model = Reliev3R(cfg.model).to(device).eval()

    if "safetensors" in str(ckpt_path):
        state = load_file(ckpt_path)
        model.load_state_dict(state, strict=strict)
    elif Path(ckpt_path).is_dir():
        ckpt_dir = Path(ckpt_path)
        ckpt_path = ckpt_dir / "mp_rank_00_model_states.pt"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            if 'module' in checkpoint:
                state_dict = checkpoint['module']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            # model.load_state_dict(state_dict, strict=False)
            model.load_state_dict(state_dict)
        else:
            # print('================ Pi3 Model ================================')
            # print(model)
            ckpt_path = ckpt_dir
            load_result = load_sharded_checkpoint(model, ckpt_path, strict=False)
            print(load_result)
            print('===========================================================')
            
        # ckpt_path = Path(ckpt_path)
        # ckpt_dir, ckpt_tag = ckpt_path.parent, ckpt_path.name
        # actual_ckpt_path = ckpt_dir / f"{ckpt_tag}.pt"
        # # state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(ckpt_dir, ckpt_tag)
        # exitcode = subprocess.call(["python", str(ckpt_dir/"zero_to_fp32.py"), 
        #     str(ckpt_dir), str(ckpt_dir), 
        #     "--tag", ckpt_tag])
        # if exitcode == 0 and actual_ckpt_path.exists():
        #     state_dict = load_file(actual_ckpt_path)
        #     model.load_state_dict(state_dict, strict=False)
        # else:
        #     raise ImportError
    else:
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
    depth_masks: np.ndarray,          # (V,H,W)
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
    print("[INFO] V: {}, H: {}, W: {}".format(V, H, W))

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
        "input_depths":      torch.ones(V, 1, H, W).to(device=device, dtype=torch.float32).unsqueeze(0),  # (1,V,1,H,W)
        "sample_indices":    [list(range(V))],
        "input_viewmats":    None,
        "input_Ks":          None,
        "input_image_names": [names],
        "data_idxs":         [0],
        "matching_infos":    None,
    }

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        pred = model.forward(data)
    
    pred_local_points = pred["local_points"].permute(0, 1, 3, 4, 2) # B,V,H,W,3
    pred_w2c = pred["camera_extrs"] # B,V,4,4
    pred_depth = pred_local_points[..., -1:] # B,V,H,W,1
    
    pred_unproj = unproject_local_point_map_to_world_point_map(
        pred_local_points.squeeze(0),
        pred_w2c.squeeze(0)
    )
        
    # print(f"[Reliev3R] normalize_camera_extrinsics & unproj pointcloud")
    pred_w2c_aligned, _, pred_unproj_aligned, _ = normalize_camera_extrinsics_and_points_batch(
        extrinsics=torch.from_numpy(gt_extrinsics).unsqueeze(0)[:,:,:3].to(dtype=torch.float32) if unproj_with_gt_extrinsics else pred_w2c[:,:,:3].cpu(),
        world_points=torch.from_numpy(pred_unproj).to(dtype=torch.float32),
        scale_by_points=False,
        # point_masks=torch.from_numpy(depth_masks).unsqueeze(0)
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
        depth_min,
        depth_max
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
        assert batch.images_cpu.shape[-1] == batch.depths.shape[-1] \
            and batch.images_cpu.shape[-2] == batch.depths.shape[-2], \
                "input shape mismatch, got {}x{} image and {}x{} depth".format(
                    batch.images_cpu.shape[-2], batch.images_cpu.shape[-1],
                    batch.depths.shape[-2], batch.depths.shape[-1]
                )
                
        V,C,H,W = batch.images_cpu.shape
        has_gt_depth = batch.meta.get("has_gt_depth", True)
        
        gt_intrinsics_rs, gt_extrinsics, _, gt_point_maps, gt_depth_maps, gt_depth_masks, gt_images = build_gt_geometry_data(
            image_pils=batch.image_pils, 
            depths=batch.depths, 
            raw_w2c=batch.raw_w2c,
            raw_K=batch.raw_K,
            output_width=W, # GT_OUTPUT_WIDTH
            depth_far=depth_max,
            depth_near=depth_min,
            has_gt_depth=has_gt_depth
        )
        
        gt_extrinsics = gt_extrinsics.squeeze(0)
        gt_c2w = closed_form_inverse_se3(gt_extrinsics)
        
        save_npy(gt_seq_dir, "gt_intrinsics.npy", gt_intrinsics_rs)
        save_npy(gt_seq_dir, "gt_extrinsics.npy", gt_c2w)
        save_npy(gt_seq_dir, "gt_depth_maps.npy", gt_depth_maps.squeeze(0))
        save_npy(gt_seq_dir, "gt_point_maps.npy", gt_point_maps.squeeze(0))
        save_npy(gt_seq_dir, "gt_depth_masks.npy", gt_depth_masks)
        save_npy(gt_seq_dir, "gt_images.npy", gt_images) 
        print(f"[INFO] GT dumped to {str(gt_seq_dir)}")
    else: # load dumped GT data for inference
        gt_intrinsics_rs = np.load(gt_seq_dir / "gt_intrinsics.npy")
        gt_c2w = np.load(gt_seq_dir / "gt_extrinsics.npy")
        # ---- fix: convert loaded c2w back to w2c (extrinsics) ----
        gt_c2w = torch.from_numpy(gt_c2w)
        gt_extrinsics = closed_form_inverse_se3(gt_c2w).cpu().numpy() # for debug only
        gt_depth_masks = np.load(gt_seq_dir / "gt_depth_masks.npy")
        print(f"[INFO] GT loaded from {str(gt_seq_dir)}")
        
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
        
        elif model_type == "vggt_omega":
            pred_w2c, _, pred_depth_maps, pred_unproj = run_vggt_omega_infer(
                model, batch.images_cpu, device=device, autocast_dtype=autocast_dtype
            )
            pred_c2w = closed_form_inverse_se3(pred_w2c.squeeze(0))

            save_npy(pred_seq_dir, "pred_extrinsics.npy", pred_c2w)
            save_npy(pred_seq_dir, "pred_depth_maps.npy", pred_depth_maps.squeeze(0).squeeze(-1))
            save_npy(pred_seq_dir, "pred_point_maps_by_unprojection.npy", pred_unproj)

        elif model_type == "reliev3r":
            pred_w2c, pred_depth_maps, pred_unproj = run_reliev3r_infer(
                model=model,
                images_cpu=batch.images_cpu,                  # (V,3,H,W) from load_and_preprocess_images
                depth_masks=gt_depth_masks,
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

        elif model_type in ("pi3", "pi3loop"):
            pred_w2c, pred_depth_maps, pred_point_maps = run_pi3_infer(
                model, batch.images_cpu, gt_depth_masks, device=device, autocast_dtype=autocast_dtype
            )
            pred_c2w = closed_form_inverse_se3(pred_w2c)

            save_npy(pred_seq_dir, "pred_extrinsics.npy", pred_c2w)         # (V,4,4)
            save_npy(pred_seq_dir, "pred_depth_maps.npy", pred_depth_maps)  # (V,H,W)
            save_npy(pred_seq_dir, "pred_point_maps.npy", pred_point_maps)  # (V,H,W,3)
        elif model_type == "da3":
            pred_w2c, pred_depth_maps, pred_point_maps = run_da3_infer(
                model, batch.images_cpu, gt_depth_masks, device=device, autocast_dtype=autocast_dtype
            )
            pred_c2w = closed_form_inverse_se3(pred_w2c)

            save_npy(pred_seq_dir, "pred_extrinsics.npy", pred_c2w)         # (V,4,4)
            save_npy(pred_seq_dir, "pred_depth_maps.npy", pred_depth_maps)  # (V,H,W)
            save_npy(pred_seq_dir, "pred_point_maps.npy", pred_point_maps)  # (V,H,W,3)
        else:
            raise ValueError(f"Unknown model_type={model_type}")

        print(f"[INFO] predictions dumped to {str(pred_seq_dir)}")

def main():
    import argparse
    p = argparse.ArgumentParser()

    # dataset
    p.add_argument("--dataset", type=str, default="7scenes", choices=list(DATA_ROOT_DEFAULTS.keys()))
    p.add_argument("--data_root", type=str, default=None,
               help="Dataset root path. If not set, inferred from --dataset.")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--seq", type=str, default="")      # chess/seq-01
    p.add_argument("--num_seq", type=int, default=0)   # 0 -> use all squences

    # dataset configs
    p.add_argument("--rgb_origin_W", type=int, default=518)   # 0 -> use all squences
    p.add_argument("--rgb_origin_H", type=int, default=518)   # 0 -> use all squences
    p.add_argument("--crop_left", type=int, default=0)   # 0 -> use all squences
    p.add_argument("--crop_right", type=int, default=0)   # 0 -> use all squences
    p.add_argument("--crop_top", type=int, default=0)   # 0 -> use all squences
    p.add_argument("--crop_bottom", type=int, default=0)   # 0 -> use all squences
    p.add_argument("--rgb_resize_W", type=int, default=0)   # 0 -> use all squences
    p.add_argument("--rgb_resize_H", type=int, default=0)   # 0 -> use all squences
    p.add_argument("--depth_scale", type=float, default=1000.0)   # 0 -> use all squences
    p.add_argument("--depth_min", type=float, default=0.1)   # 0 -> use all squences
    p.add_argument("--depth_max", type=float, default=10.0)   # 0 -> use all squences
    p.add_argument("--fx", type=float, default=1.0)   # 0 -> use all squences
    p.add_argument("--fy", type=float, default=1.0)   # 0 -> use all squences
    p.add_argument("--cx", type=float, default=1.0)   # 0 -> use all squences
    p.add_argument("--cy", type=float, default=1.0)   # 0 -> use all squences

    # view sample
    p.add_argument("--mode", type=str, default="stride", choices=["stride", "random"])
    p.add_argument("--stride", type=int, default=40)
    p.add_argument("--seq_len", type=int, default=8)  # number of views, actual length of the sequence, -1 means the whole sequence
    p.add_argument("--max_seq_len", type=int, default=-1)  # stride: 截断数量; random: 采样数量(0=全量)
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
    p.add_argument("--model_size", type=str, default="dinov2_large")
    p.add_argument("--loop_layers", type=int, default=0)  # pi3loop: (frame, global) pairs in the loop body, 0 = no loop
    p.add_argument("--loop_k", type=int, default=0)       # pi3loop: inference loop steps K
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
    ds_config = get_dataset_config(vars(args))
    ds = get_dataset(args.dataset, ds_config)
    stride = args.stride # args.stride or ds.default_stride(args.view)
    sequences = [args.seq] if args.seq else ds.list_sequences(args.split)
    if args.num_seq > 0: sequences = sequences[:args.num_seq]

    # outdir
    setting_dir = Path(args.out_root) / args.dataset / (
        f"{args.split}_" +
        (f"stride{stride}_max{args.max_seq_len if args.max_seq_len>0 else 'all'}_len{args.seq_len}_numseq{args.num_seq}_{args.rgb_resize_W}x{args.rgb_resize_H}"
        if args.mode == "stride"
        else f"rand_max{args.max_seq_len if args.max_seq_len>0 else 'all'}_len{args.seq_len}_numseq{args.num_seq}_{args.rgb_resize_W}x{args.rgb_resize_H}")
    )
    setting_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[INFO] "
        f"dataset={args.dataset} | split={args.split} | "
        f"model={args.model} | num_seqs={len(sequences)} | "
        f"mode={args.mode} | stride={stride} | max_seq_len={args.max_seq_len if args.max_seq_len>0 else 'all'} | "
        f"seq_len={args.seq_len} | num_seq={args.num_seq} | seed={args.seed} | "
        f"reuse_frame_ids={args.reuse_frame_ids} | device={device}\n"
        f"[INFO] use_gt_pose={args.use_gt_pose} use_gt_K={args.use_gt_k}\n"
        f"[INFO] data_root={args.data_root}\n"
        f"[INFO] out={setting_dir}"
    )

    # load model by branch
    if args.model == "vggt":
        model = load_model_vggt(args.ckpt).to(device)
    elif args.model == "vggt_omega":
        model = load_model_vggt_omega(args.ckpt).to(device)
    elif args.model == "pi3":
        model = load_model_pi3(Path(args.ckpt), model_size=args.model_size).to(device)
    elif args.model == "pi3loop":
        model = load_model_pi3loop(args.ckpt, args.model_size, args.loop_layers, args.loop_k).to(device)
    elif args.model == "da3":
        model = load_model_da3(args.ckpt).to(device)
    elif args.model == 'reliev3r':
        if not args.ckpt:
            raise ValueError("--reliev3r_ckpt is required when --model reliev3r")
        model = load_model_reliev3r(args.config, args.ckpt, device=device, strict=False)
    else:
        raise ValueError(f"Unknown model_type={args.model}")
    # model = None

    pbar = tqdm(sequences, desc=args.dataset, unit="seq")
    for seq in pbar:
        name = args.model if not args.prefix else f"{args.model}_{args.prefix}"
        if unproj_with_gt_extrinsics: name += "_gt_pose"
        if unproj_with_gt_intrinsics: name += "_gt_K"
        pred_seq_dir = setting_dir / name / seq
        gt_seq_dir   = setting_dir / "gt" / seq
        pred_seq_dir.mkdir(parents=True, exist_ok=True)
        gt_seq_dir.mkdir(parents=True, exist_ok=True)
        
        seq_len = ds.count_seq_len(seq) if args.seq_len < 0 else args.seq_len
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
        if not np.any(frame_ids): 
            print("[INFO] Sequence:", seq, "frame IDs:", frame_ids)
            continue
        
        print("[INFO] Sequence:", seq, "frame IDs:", frame_ids)
        
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
            depth_min=args.depth_min,
            depth_max=args.depth_max
        )

        pbar.set_postfix_str(f"{seq} K={len(frame_ids)}")
        tqdm.write(f"[OK] {seq}  K={len(frame_ids)}")


if __name__ == "__main__":
    main()
