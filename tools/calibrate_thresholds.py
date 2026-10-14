#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
伪标签质量门槛校准（需要真值）。

输入：与 quality_eval.py pseudo 相同的目录格式，但每帧 npz 必须带 gt_depth（LiDAR / 扫描 / 渲染真值）。
做法：
  1. 像素级参数网格：tau × k_verify（默认 {0.02,0.03,0.05} × {1,2,3}）。每组参数对每条序列算一次 M1–M6 和 G1
     （保留集合 = 已验证像素）。
  2. 序列级门槛网格：FSV 上限 × VPR 下限。被剔除的序列不贡献保留像素。
  3. 目标：在 pooled AbsRel_kept ≤ --target-absrel 的前提下，让 GTCov（保留且有真值 / 全部有真值）最大。
  4. --holdout-frac > 0 时，按序列名哈希切出验证集：只在校准集上选参数，再在验证集上报告一次（避免看着答案调参）。
输出 --out 指定的 json，可直接传给 quality_eval.py --thresholds：
  {"tau":…, "k_verify":…, "thresholds": {"fsv": [...], "vpr": [...], …}, "calibration": {...报告...}}

用法：
  python tools/calibrate_thresholds.py --root /path/to/pseudo_with_gt --frames 8 \
      --target-absrel 0.03 --holdout-frac 0.3 --out calib.json
"""
import argparse
import hashlib
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from utils import quality_metrics as Q  # noqa: E402
import quality_eval as QE  # noqa: E402


def split_of(name, frac):
    h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "holdout" if h < frac else "calib"


def pooled(rows, fsv_max, vpr_min):
    """在序列级门槛下，汇总保留像素的精度与真值覆盖率。"""
    kept_gt, all_gt, err_sum, n_acc = 0, 0, 0.0, 0
    for r in rows:
        g = r.get("gt")
        if not g:
            continue
        all_gt += g["n_gt"]
        m = r["metrics"]
        if not (np.isfinite(m.get("fsv", np.nan)) and m["fsv"] <= fsv_max and m.get("vpr", 0) >= vpr_min):
            continue
        if not np.isfinite(g["abs_rel_kept"]):
            continue
        kept_gt += g["n_kept_gt"]
        err_sum += g["abs_rel_kept"] * g["n_kept_gt"]
        n_acc += 1
    return dict(abs_rel_kept=err_sum / kept_gt if kept_gt else float("nan"),
                gt_cov=kept_gt / all_gt if all_gt else float("nan"), n_accepted=n_acc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--z-far", type=float, default=0.0)
    ap.add_argument("--taus", type=float, nargs="*", default=[0.02, 0.03, 0.05])
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--fsv-max", type=float, nargs="*", default=[0.05, 0.10, 0.15, 0.20, 1.0])
    ap.add_argument("--vpr-min", type=float, nargs="*", default=[0.0, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--target-absrel", type=float, default=0.03)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="calib.json")
    args = ap.parse_args()

    seqs = QE.find_sequences(args.root)
    frames_by_seq = {}
    for d in seqs:
        fr = QE.load_frames(d)
        if all("gt_depth" in f for f in fr):
            frames_by_seq[os.path.relpath(d, args.root)] = fr
    if not frames_by_seq:
        sys.exit(f"[ERROR] {args.root} 下没有带 gt_depth 的序列，无法校准")
    split = {n: split_of(n, args.holdout_frac) for n in frames_by_seq}
    print(f"{len(frames_by_seq)} sequences with GT: calib={sum(v == 'calib' for v in split.values())} "
          f"holdout={sum(v == 'holdout' for v in split.values())}", flush=True)

    trials = []
    for tau, k in itertools.product(args.taus, args.ks):
        cfg = Q.QualityConfig(tau=tau, k_verify=k, stride=args.stride)
        rows = {}
        for n, fr in frames_by_seq.items():
            rows[n] = QE.eval_sequence(fr, cfg, F=args.frames, frame_step=args.frame_step, z_far=args.z_far,
                                       with_scale_search=False)
        for fmax, vmin in itertools.product(args.fsv_max, args.vpr_min):
            c = pooled([rows[n] for n in rows if split[n] == "calib"], fmax, vmin)
            h = pooled([rows[n] for n in rows if split[n] == "holdout"], fmax, vmin)
            trials.append(dict(tau=tau, k_verify=k, fsv_max=fmax, vpr_min=vmin, calib=c, holdout=h))
        print(f"  tau={tau} k={k} done", flush=True)

    feasible = [t for t in trials if np.isfinite(t["calib"]["abs_rel_kept"]) and t["calib"]["abs_rel_kept"] <= args.target_absrel]
    if not feasible:
        best = min(trials, key=lambda t: t["calib"]["abs_rel_kept"] if np.isfinite(t["calib"]["abs_rel_kept"]) else 1e9)
        print(f"[WARN] 没有组合满足 AbsRel_kept ≤ {args.target_absrel}；取误差最小的一组", flush=True)
    else:
        best = max(feasible, key=lambda t: (t["calib"]["gt_cov"], -t["calib"]["abs_rel_kept"]))

    th = {k: list(v) for k, v in Q.DEFAULT_THRESHOLDS.items()}
    th["fsv"] = ["lower", min(0.01, best["fsv_max"]), min(0.05, best["fsv_max"]), best["fsv_max"]]
    th["vpr"] = ["higher", max(0.60, best["vpr_min"]), best["vpr_min"], best["vpr_min"]]
    out = dict(tau=best["tau"], k_verify=best["k_verify"], thresholds=th,
               calibration=dict(target_absrel=args.target_absrel, chosen=best, n_trials=len(trials),
                                top5=sorted(feasible, key=lambda t: -t["calib"]["gt_cov"])[:5]))
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=QE._json_default)
    c, h = best["calib"], best["holdout"]
    print(f"\nchosen: tau={best['tau']} k={best['k_verify']} fsv_max={best['fsv_max']} vpr_min={best['vpr_min']}")
    print(f"  calib  : AbsRel_kept={c['abs_rel_kept']:.4f} GTCov={c['gt_cov']:.3f} accepted={c['n_accepted']}")
    print(f"  holdout: AbsRel_kept={h['abs_rel_kept']:.4f} GTCov={h['gt_cov']:.3f} accepted={h['n_accepted']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
