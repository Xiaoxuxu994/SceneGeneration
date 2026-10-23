"""Run a model on a benchmark built by build_scannetpp_val8.py and write <bench>/<name>/<clip>/pred.npz.

  # official Pi3 (pi3_training.Pi3 with the dinov2_large_0.95B spec; the confidence head is not used)
  python tools/eval_bench/infer.py --bench B --name pi3_official --model-size dinov2_large_0.95B \
      --weights /path/to/yyfz233/Pi3/model.safetensors
  # a training run: model config name + checkpoint dir (accelerate/DeepSpeed checkpoint_N or best_model) or file
  python tools/eval_bench/infer.py --bench B --name exp1022_00 --model pi3_dinov3s_w384 \
      --weights /efs-m3ga/z00533370/exp/pi3sup/outputs/exp1022_00_.../ckpts/checkpoint_14
  # loop model at a different inference K
  python tools/eval_bench/infer.py ... --model pi3_dinov3s_loop_w384_L1_k14to18 --k 12 --name exp1022_01_K12

Several processes (one per NPU / GPU) can share one --name: clips are claimed atomically.
pred.npz: c2w (N,4,4 cam->world as predicted), K (N,3,3 at prediction resolution, principal point at the centre,
focal fitted to local_points), depth (N,h,w = local_points z), sec.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

try:  # Ascend
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    DEV_TYPE = "npu"
except ImportError:
    DEV_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def no_download_backbones():
    """The encoder weights come from the checkpoint; do not load / download pretrained backbones."""
    import pi3.models.pi3_training as P
    orig = P.Pi3._build_backbone_cfg

    def cfg_no_dl():
        cfg = orig()
        for enc in cfg.values():
            for spec in enc.values():
                b = spec["builder"]
                spec["builder"] = (lambda b: (lambda pretrained=True, model_dir=None: b(pretrained=False)))(b)
        return cfg
    P.Pi3._build_backbone_cfg = staticmethod(cfg_no_dl)


def build_model(args):
    no_download_backbones()
    if args.model:
        import hydra
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.2"):
            cfg = compose(config_name="default", overrides=[
                f"model={args.model}", "hydra/job_logging=default", "hydra/hydra_logging=default"])
        cfg.model.ckpt = None
        return hydra.utils.instantiate(cfg.model)
    import pi3.models.pi3_training as P
    return P.Pi3(pos_type="rope100", decoder_size="large", model_size=args.model_size, load_vggt=False,
                 freeze_encoder=False, use_global_points=False, train_conf=False, num_dec_blk_not_to_checkpoint=0,
                 ckpt=None)


def load_weights(model, path):
    p = Path(path)
    if p.is_dir():
        cands = [p / "pytorch_model" / "mp_rank_00_model_states.pt", p / "mp_rank_00_model_states.pt",
                 p / "model.safetensors", p / "pytorch_model.bin"]
        p = next((c for c in cands if c.exists()), None)
        if p is None:
            raise FileNotFoundError(f"no weights in {path} (tried {[str(c) for c in cands]})")
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(p))
    else:
        obj = torch.load(str(p), map_location="cpu", weights_only=False)
        sd = obj["module"] if isinstance(obj, dict) and "module" in obj else obj
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if k != "pos_special"]  # constant zero buffer, absent in official Pi3
    unexpected = [k for k in res.unexpected_keys if not k.startswith(("conf_decoder.", "conf_head."))]
    if missing or unexpected:
        raise RuntimeError(f"weights {p} do not match the model: missing {missing[:5]} ({len(missing)}), "
                           f"unexpected {unexpected[:5]} ({len(unexpected)})")
    print(f"[infer] loaded {p} ({len(sd)} tensors; ignored {len(res.unexpected_keys)} confidence-head keys)", flush=True)
    return str(p)


def target_size(w, h, patch, limit):
    """Official Pi3 preprocessing (pi3.utils.basic.load_images_as_tensor), for any patch size."""
    s = (limit / (w * h)) ** 0.5
    wt, ht = w * s, h * s
    k, m = round(wt / patch), round(ht / patch)
    while (k * patch) * (m * patch) > limit:
        if k / m > wt / ht:
            k -= 1
        else:
            m -= 1
    return max(1, k) * patch, max(1, m) * patch


def load_frames(d, n, patch, resize, limit):
    imgs = [cv2.cvtColor(cv2.imread(str(d / f"{i:06d}.png")), cv2.COLOR_BGR2RGB) for i in range(n)]
    h, w = imgs[0].shape[:2]
    if resize == "auto":
        resize = "native" if (h % patch == 0 and w % patch == 0) else "official"
    if resize == "official":
        tw, th = target_size(w, h, patch, limit)
        from PIL import Image
        imgs = [np.asarray(Image.fromarray(im).resize((tw, th), Image.Resampling.LANCZOS)) for im in imgs]
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
    return x, resize


def fit_K(lp):
    """Pinhole K per frame from local points (N,h,w,3): principal point at the image centre, fx / fy by
    least squares on u - cx = fx * X/Z (pixel centres at integer coordinates)."""
    n, h, w, _ = lp.shape
    z = lp[..., 2]
    ok = torch.isfinite(lp).all(-1) & (z > 1e-6)
    zs = torch.where(ok, z, torch.ones_like(z))
    xn, yn = lp[..., 0] / zs, lp[..., 1] / zs
    u = (torch.arange(w, dtype=lp.dtype) - (w - 1) / 2)[None, None, :].expand(n, h, w)
    v = (torch.arange(h, dtype=lp.dtype) - (h - 1) / 2)[None, :, None].expand(n, h, w)
    okf = ok.to(lp.dtype)
    fx = (u * xn * okf).sum((1, 2)) / (xn * xn * okf).sum((1, 2)).clamp_min(1e-12)
    fy = (v * yn * okf).sum((1, 2)) / (yn * yn * okf).sum((1, 2)).clamp_min(1e-12)
    K = torch.zeros(n, 3, 3, dtype=lp.dtype)
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = fx, fy, (w - 1) / 2, (h - 1) / 2, 1
    return K


def claim(d):
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--name", required=True, help="output sub-directory (method name in the results)")
    ap.add_argument("--weights", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="configs/model/<name>.yaml of a training run")
    g.add_argument("--model-size", help="pi3_training spec, e.g. dinov2_large_0.95B for the official Pi3")
    ap.add_argument("--k", type=int, default=None, help="inference K of a loop model (default: loop_k_infer)")
    ap.add_argument("--resize", default="auto", choices=["auto", "native", "official"],
                    help="native: frames as stored; official: Pi3 rule (<= --pixel-limit, multiples of the patch)")
    ap.add_argument("--pixel-limit", type=int, default=255000)
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (debug)")
    args = ap.parse_args()

    B, OV = Path(args.bench), Path(args.bench) / args.name
    dev = torch.device(f"{DEV_TYPE}:0" if DEV_TYPE != "cpu" else "cpu")
    t0 = time.time()
    model = build_model(args)
    wpath = load_weights(model, args.weights)
    if args.k is not None:
        assert hasattr(model, "loop_k_infer"), "--k is only for loop models"
        if not model.loop_k_min <= args.k <= model.loop_k_max:
            print(f"[infer] WARNING: K={args.k} outside the trained range [{model.loop_k_min}, {model.loop_k_max}]")
        model.loop_k_infer = args.k
    model = model.to(dev).eval()
    load_sec = time.time() - t0
    patch = int(getattr(model, "patch_size", 14))
    info = {"name": args.name, "weights": wpath, "model": args.model or args.model_size, "k": getattr(model, "loop_k_infer", None),
            "loop_layers": getattr(model, "loop_layers", None), "patch": patch, "resize": args.resize,
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3), "device": DEV_TYPE}
    OV.mkdir(parents=True, exist_ok=True)
    (OV / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)

    clips = json.loads((B / "clips.json").read_text())[: args.limit]
    amp = dict(device_type=DEV_TYPE, dtype=torch.bfloat16, enabled=DEV_TYPE != "cpu")
    done = 0
    for c in clips:
        name, n = c["clip"], c["n_frames"]
        if (OV / name / "pred.npz").exists() or not claim(OV / name):
            continue
        try:
            x, mode = load_frames(B / "frames" / name, n, patch, args.resize, args.pixel_limit)
            x = x.to(dev)
            if DEV_TYPE != "cpu":
                torch.cuda.synchronize()
            t1 = time.time()
            with torch.no_grad(), torch.autocast(**amp):
                res = model(x[None])
            if DEV_TYPE != "cpu":
                torch.cuda.synchronize()
            sec = time.time() - t1
            lp = res["local_points"][0].float().cpu()
            np.savez(OV / name / "pred.npz", c2w=res["camera_poses"][0].float().cpu().numpy().astype(np.float64),
                     K=fit_K(lp).numpy().astype(np.float64), depth=lp[..., 2].numpy().astype(np.float32), sec=sec)
            rec = {"clip": name, "ok": True, "sec": round(sec, 3), "hw": list(lp.shape[1:3]), "resize": mode}
        except Exception as e:  # noqa: BLE001
            rec = {"clip": name, "ok": False, "error": repr(e)[:300]}
        with open(OV / "status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        done += 1
        if done % 50 == 0 or not rec["ok"]:
            print(json.dumps(rec), flush=True)
    print(f"[infer] {args.name}: {done} clips in this process, load {load_sec:.1f}s", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
