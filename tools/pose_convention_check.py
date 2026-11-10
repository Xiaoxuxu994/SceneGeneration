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


def pair_ratios(depths, valids, Ks, c2ws):
    """每对帧 (i, j)：i 帧的点投到 j 帧后，投影深度 / j 帧深度 的中位数。偏离 1 说明两帧深度尺度不一致。"""
    F = len(depths)
    H, W = depths[0].shape
    R = np.full((F, F), np.nan)
    rays = [G.pixel_rays(H, W, K) for K in Ks]
    w2cs = [np.linalg.inv(T) for T in c2ws]
    with np.errstate(all="ignore"):
        for i in range(F):
            vv, uu = np.nonzero(valids[i])
            X = rays[i][vv, uu] * depths[i][vv, uu][:, None]
            Xw = X @ c2ws[i][:3, :3].T + c2ws[i][:3, 3]
            for j in range(F):
                if i == j:
                    continue
                Xj = Xw @ w2cs[j][:3, :3].T + w2cs[j][:3, 3]
                z = Xj[:, 2]
                front = z > 1e-6
                u = np.round(Ks[j][0, 0] * Xj[:, 0] / np.where(front, z, 1) + Ks[j][0, 2]).astype(np.int64)
                v = np.round(Ks[j][1, 1] * Xj[:, 1] / np.where(front, z, 1) + Ks[j][1, 2]).astype(np.int64)
                ok = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                ok[ok] = valids[j][v[ok], u[ok]]
                if ok.sum() > 100:
                    R[i, j] = np.median(z[ok] / depths[j][v[ok], u[ok]])
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--res", type=int, nargs=2, default=[512, 336])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.03, 0.1], help="as_loaded / x1 下再按这些阈值算一致率")
    ap.add_argument("--tscales", type=float, nargs="*", default=[],
                    help="as_loaded 下把相机平移（相对第一帧）乘这些系数，搜索每个 clip 最合适的 平移/深度 尺度比")
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
        for tau in args.taus:
            cons, viol, _ = score(depths, valids, Ks, c2ws, tau)
            print(f"   as_loaded x1 tau={tau:<5g} cons={cons:6.1%} viol={viol:6.1%}", flush=True)
        if args.tscales:
            best_t = None
            for ts in args.tscales:
                poses = []
                for T in c2ws:
                    T2 = T.copy()
                    T2[:3, 3] = c2ws[0][:3, 3] + ts * (T[:3, 3] - c2ws[0][:3, 3])
                    poses.append(T2)
                cons, viol, _ = score(depths, valids, Ks, poses, args.tau)
                if best_t is None or cons - viol > best_t[1] - best_t[2]:
                    best_t = (ts, cons, viol)
                print(f"   translation x{ts:<5g} cons={cons:6.1%} viol={viol:6.1%}", flush=True)
            print(f"   best translation scale x{best_t[0]:g}: cons={best_t[1]:.1%} viol={best_t[2]:.1%}", flush=True)
        R = pair_ratios(depths, valids, Ks, c2ws)
        adj = [R[i, i + 1] for i in range(len(depths) - 1)]
        to0 = [R[i, 0] for i in range(1, len(depths))]
        print("   median(proj depth / target depth)  adjacent frames: " + " ".join(f"{x:.3f}" for x in adj), flush=True)
        print("                                       frame k -> 0:    " + " ".join(f"{x:.3f}" for x in to0), flush=True)
    print("\n== mean over samples", flush=True)
    best = max(total, key=lambda k: np.mean([c for c, _ in total[k]]) - np.mean([v for _, v in total[k]]))
    for (conv, s), vals in total.items():
        c, v = np.mean([x for x, _ in vals]), np.mean([y for _, y in vals])
        print(f"   pose={conv:9s} depth x{s:<4g} cons={c:6.1%} viol={v:6.1%}{'   <- best' if (conv, s) == best else ''}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
