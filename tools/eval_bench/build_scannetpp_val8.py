"""Export the fixed ScanNet++ held-out test set as a benchmark: exactly the clips of the training-time validation.

The test dataset is built from the same Hydra config as training (data=scannetpp_only: last 10% of the sorted
scenes, 8 views, 512x336, max_distance 8), resized to the same number of samples and seeded the same way
(set_epoch(0, base_seed)), so clip i here is clip i of every validation during training.

  python tools/eval_bench/build_scannetpp_val8.py --out /efs-m3ga/z00533370/eval_bench/scannetpp_val8 [--n 2000]

Layout (read by infer.py / evaluate.py):
  <out>/clips.json                    [{clip, index, scene, frames, n_frames, hw}]
  <out>/frames/<clip>/000000.png ...  RGB exactly as fed to the model during validation (after crop / resize)
  <out>/depth/<clip>/000000.png ...   GT depth, uint16 millimetres, 0 = invalid
  <out>/gt/<clip>.npz                 c2w (N,4,4) OpenCV cam->world, K (N,3,3) at frame resolution
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def compose_cfg(overrides):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.2"):
        return compose(config_name="default", overrides=overrides + [
            "hydra/job_logging=default", "hydra/hydra_logging=default"])


class Export(torch.utils.data.Dataset):
    def __init__(self, ds, out, idxs, frame_num):
        self.ds, self.out, self.idxs, self.frame_num = ds, out, idxs, frame_num

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, j):
        i = int(self.idxs[j])
        views = self.ds[(i, 0, self.frame_num)]
        scene = str(views[0]["label"])
        clip = f"{i:04d}_{scene}"
        fdir, ddir = self.out / "frames" / clip, self.out / "depth" / clip
        fdir.mkdir(parents=True, exist_ok=True)
        ddir.mkdir(parents=True, exist_ok=True)
        c2w, K = [], []
        for k, v in enumerate(views):
            img = (v["img"].permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(fdir / f"{k:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            d = np.nan_to_num(np.asarray(v["depthmap"], np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            cv2.imwrite(str(ddir / f"{k:06d}.png"), np.clip(np.round(d * 1000.0), 0, 65535).astype(np.uint16))
            c2w.append(np.asarray(v["camera_pose"], np.float64))
            K.append(np.asarray(v["camera_intrinsics"], np.float64))
        np.savez(self.out / "gt" / f"{clip}.npz", c2w=np.stack(c2w), K=np.stack(K))
        return {"clip": clip, "index": i, "scene": scene, "frames": [str(v["instance"]) for v in views],
                "n_frames": len(views), "hw": list(img.shape[:2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None, help="number of clips (default: test_dataset weight, 2000)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--data", default="scannetpp_only")
    ap.add_argument("--train", default="train_scannetpp_dinov3")
    ap.add_argument("--data-root", default=None, help="override test_dataset.Scannetpp.data_root")
    args = ap.parse_args()

    import hydra
    from omegaconf import open_dict
    cfg = compose_cfg([f"data={args.data}", f"train={args.train}", "model=pi3_dinov3s_w384"])
    dcfg = cfg.test_dataset
    if args.data_root:
        with open_dict(dcfg):
            dcfg.Scannetpp.data_root = args.data_root
    ds = hydra.utils.instantiate(dcfg.Scannetpp)
    ds.convert_attributes()
    size = int(dcfg.weights.Scannetpp)
    R = size @ ds                              # ResizedDataset, as in datasets.create_dataloader
    R.set_epoch(0, base_seed=cfg.train.base_seed)  # as in Pi3Trainer.before_epoch for the test set
    n = min(args.n or size, size)
    frame_num = int(dcfg.Scannetpp.get("frame_num", 8))
    print(f"test scenes {len(ds)}, clips {n}/{size}, {frame_num} views, base_seed {cfg.train.base_seed}", flush=True)

    out = Path(args.out)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    loader = torch.utils.data.DataLoader(Export(R, out, np.arange(n), frame_num), batch_size=None,
                                         num_workers=args.workers, collate_fn=lambda x: x)
    recs = []
    for k, r in enumerate(loader):
        recs.append(r)
        if (k + 1) % 100 == 0 or k + 1 == n:
            print(f"[{k + 1}/{n}] {r['clip']}", flush=True)
    recs.sort(key=lambda r: r["index"])
    (out / "clips.json").write_text(json.dumps(recs, indent=1))
    meta = {"source": "ScanNet++ held-out scenes (data=%s, split_ratio=%s)" % (args.data, dcfg.Scannetpp.split_ratio),
            "n_clips": len(recs), "n_scenes": len({r["scene"] for r in recs}), "frame_num": frame_num,
            "base_seed": int(cfg.train.base_seed), "data_root": str(dcfg.Scannetpp.data_root),
            "depth": "uint16 mm, 0 = invalid", "pose": "c2w OpenCV"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
