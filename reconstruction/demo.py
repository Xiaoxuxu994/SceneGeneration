#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch

from utils.config_loader import LoadConfig
from model import Reliev3R

# =========================
# Hard-coded paths / device
# =========================
CONFIG_PATH = "configs/reliev3r_full/main.yaml"
CKPT_PATH   = "/mnt/nvme1/share/chenyouyu/codespace/Reliev3R-251228/exp/Reliev3R_251231_stg1_v8c32s1_p50/ckpt/50000.ckpt"                  # <- "" means random init
DEVICE      = "npu:5"             # "cuda:0" or "cpu"

# =========================
# Dummy input shape
# =========================
B = 1
V = 10
H = 518
W = 518


def strip_ddp_prefix(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    ks = list(state_dict.keys())
    if all(k.startswith("module.") for k in ks):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def build_dummy_batch(device: str):
    # input_images: (B,V,3,H,W)
    input_images = torch.rand((B, V, 3, H, W), dtype=torch.float32, device=device)

    # input_Ks: (B,V,3,3) in pixel units
    Ks = torch.eye(3, dtype=torch.float32, device=device).view(1, 1, 3, 3).repeat(B, V, 1, 1)
    Ks[..., 0, 0] = 525.0
    Ks[..., 1, 1] = 525.0
    Ks[..., 0, 2] = W / 2.0
    Ks[..., 1, 2] = H / 2.0

    # input_viewmats: (B,V,4,4) world-to-camera
    viewmats = torch.eye(4, dtype=torch.float32, device=device).view(1, 1, 4, 4).repeat(B, V, 1, 1)
    viewmats[..., :3, 3] = torch.tensor([0.1, 0.0, 0.0], device=device).view(1, 1, 3).repeat(B, V, 1)

    data = {
        "scene_names":       ["demo"] * B,
        "input_images":      input_images,
        "input_depths":      None,
        "sample_indices":    [list(range(V)) for _ in range(B)],
        "input_viewmats":    viewmats,
        "input_Ks":          Ks,
        "input_image_names": [["dummy.png"] * V for _ in range(B)],
        "data_idxs":         list(range(B)),
        "matching_infos":    None,
    }
    return data


def main():
    # 1) load config + build model
    cfg = LoadConfig(CONFIG_PATH)
    cfg.eval = True
    model = Reliev3R(cfg.model).to(DEVICE).eval()

    # 2) load checkpoint if provided
    if CKPT_PATH and os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        state = strip_ddp_prefix(state)
        model.load_state_dict(state, strict=True)
        print("[OK] model loaded from ckpt:", CKPT_PATH)
    else:
        print("[OK] CKPT_PATH empty or not found -> using random initialized model")

    print("  #params =", sum(p.numel() for p in model.parameters()))

    # 3) dummy forward
    data = build_dummy_batch(DEVICE)
    with torch.inference_mode():
        pred = model._forward(data) if hasattr(model, "_forward") else model(data)

    # 4) print outputs
    if isinstance(pred, dict):
        print("[OK] forward returned dict, keys:", list(pred.keys()))
        for k in ["depths", "scaled_depths", "depths_conf", "camera_extrs"]:
            if k in pred and torch.is_tensor(pred[k]):
                t = pred[k]
                try:
                    print(f"  - {k}: shape={tuple(t.shape)} min/max=({t.min().item():.4g},{t.max().item():.4g})")
                except Exception:
                    print(f"  - {k}: shape={tuple(t.shape)}")
    else:
        print("[OK] forward returned:", type(pred))


if __name__ == "__main__":
    main()
