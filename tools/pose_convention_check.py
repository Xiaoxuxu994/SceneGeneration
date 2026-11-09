#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位姿约定 / 深度尺度诊断：同一组帧，分别假设 loader 给出的 camera_pose 是 c2w（原样）或 w2c（求逆），
并把深度乘几个尺度，统计多视角重投影的一致率与违例率。正确的组合应当一致率最高、违例率最低。

    python tools/pose_convention_check.py --config configs/data/spatialvid_check.yaml --name SpatialVID --n 4

一致 cons：投到另一帧后与那里的深度相对差 < tau；违例 viol：跑到可见表面前面（rel < -tau）。
都按"至少落在一个其他帧有效像素上"的点计（与 tools/gt_vis_dump.py 的 per_pixel_consistency 同一套投影）。
"""
import argparse
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import gt_audit as G  # noqa: E402
from gt_vis_dump import per_pixel_consistency  # noqa: E402


def score(depths, valids, Ks, c2ws, tau):
    n_cons, n_viol = per_pixel_consistency(depths, valids, Ks, c2ws, tau)
    seen = sum(int(((c + v) > 0).sum()) for c, v in zip(n_cons, n_viol))
    cons = sum(int((c > 0).sum()) for c in n_cons)
    viol = sum(int(((v > 0) & (v >= c)).sum()) for c, v in zip(n_cons, n_viol))
    n_valid = sum(int(v.sum()) for v in valids)
    return cons / max(seen, 1), viol / max(seen, 1), seen / max(n_valid, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--res", type=int, nargs=2, default=[512, 336])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from omegaconf import OmegaConf
    cfg_all = OmegaConf.load(args.config)
    ds, z_far = G.build_dataset(cfg_all, args.name, tuple(args.res), args.frames, "train", args.seed)
    n = min(args.n, len(ds))
    total = {}
    for idx in range(n):
        views = G.fetch_views(ds, idx, args.res, args.seed * 100003 + idx)
        depths, valids, Ks, c2ws, _ = G.unpack(views, z_far)
        print(f"\n== idx={idx} {views[0].get('label', '')}  frames {[v.get('instance') for v in views]}", flush=True)
        for conv in ("as_loaded", "inverted"):
            poses = c2ws if conv == "as_loaded" else [np.linalg.inv(T) for T in c2ws]
            for s in args.scales:
                cons, viol, cover = score([d * s for d in depths], valids, Ks, poses, args.tau)
                total.setdefault((conv, s), []).append((cons, viol))
                print(f"   pose={conv:9s} depth x{s:<4g} cons={cons:6.1%} viol={viol:6.1%} overlap={cover:6.1%}", flush=True)
    print("\n== mean over samples", flush=True)
    best = max(total, key=lambda k: np.mean([c for c, _ in total[k]]) - np.mean([v for _, v in total[k]]))
    for (conv, s), vals in total.items():
        c, v = np.mean([x for x, _ in vals]), np.mean([y for _, y in vals])
        print(f"   pose={conv:9s} depth x{s:<4g} cons={c:6.1%} viol={v:6.1%}{'   <- best' if (conv, s) == best else ''}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
