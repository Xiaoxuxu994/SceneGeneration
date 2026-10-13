#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GT 体检：用多视角重投影把"肉眼看着对"变成可比较的数字。只读，不改任何 loader。

对每个数据集抽 N 个样本（每个样本 F 帧），绕过 __getitem__ 的静默重试，直接调 _get_views，
然后复刻 base_dataset 的 valid_mask 逻辑（depth>0 & depth<z_far & finite），输出：

  [几何一致性]  i 帧的点投到 j 帧，与 j 帧深度比较，rel = (z_proj - d_j) / d_j
      cons   : |rel| < tau 的比例（越高越好；遮挡会拉低，所以只做横向对比）
      viol   : rel < -tau 的比例 = 点跑到了 j 帧可见表面的"前面"，遮挡解释不了，是真错误
      cons_center / cons_corner : 按目标像素到主点的距离分桶，仅供参考（会受场景内容影响，判定以 ray2z 为准）
  [假设检验]    同一批样本再算两遍，看哪种假设让一致性明显变好：
      ray2z  : 把深度当作沿射线距离转换成 z 之后的 cons。明显高于原始 cons => 数据其实是 ray depth
      best_s : 只缩放深度（位姿不动）时 cons 最高的尺度。远离 1 => 深度和位姿量纲不一致
  [掩码收益]    被 loader 自己的 depth_mask / 边缘 / 98 分位截断去掉的像素，和留下像素的 viol 对比
      edge_viol vs flat_viol : 深度不连续处的飞点有多差
      mask_out_viol          : loader 返回了 depth_mask 却没进 valid_mask 时（如 DL3DV），被它排除的像素有多差
      pct_removed / pct_removed_cons : --raw-compare 时，98 分位截断去掉了多少有效像素、它们本来是否一致
  [量纲]        depth p5/p50/p95、valid_ratio、相机基线/中位深度、R 的正交误差
  [失败率]      _get_views 抛异常的比例和前几条错误信息（训练时这些会被静默重试吞掉）

用法（在 SceneGeneration 根目录）：
    python tools/gt_audit.py --config configs/data/datamix_all_v10-1.yaml \
        --datasets Hypersim MatrixCity Scannetpp DL3DV CO3DV2 OmniObject3DNew \
        --n 30 --raw-compare --out gt_audit_out
不传 --datasets 时跑 config 里 weights 列出的全部数据集。
"""
import argparse
import importlib
import json
import os
import sys
import time
import traceback

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# geometry helpers (numpy only)
# ---------------------------------------------------------------------------
def pixel_rays(H, W, K):
    """K^-1 [u, v, 1]，z 分量为 1 的射线方向，(H, W, 3)。与 geometry.depthmap_to_camera_coordinates 同一像素约定。"""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def ray_to_z(depth, K):
    """把沿射线的距离转成 z-depth：z = r / ||K^-1 [u v 1]||。"""
    rays = pixel_rays(*depth.shape, K)
    return depth / np.linalg.norm(rays, axis=-1)


def radial_norm(H, W, K):
    """像素到主点的距离，按半对角线归一化到 ~[0, 1]。"""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    r = np.sqrt((u - K[0, 2]) ** 2 + (v - K[1, 2]) ** 2)
    return r / (0.5 * np.sqrt(W ** 2 + H ** 2))


def depth_edge_np(depth, valid, rtol=0.03):
    """3x3 邻域内 (max-min)/depth > rtol 视为深度不连续，与 pi3.utils.geometry.depth_edge 同口径。"""
    d = np.where(valid, depth, np.nan)
    p = np.pad(d, 1, mode="edge")
    H, W = depth.shape
    stack = np.stack([p[i:i + H, j:j + W] for i in range(3) for j in range(3)], axis=0)
    with np.errstate(invalid="ignore", all="ignore"):
        mx = np.nanmax(stack, axis=0)
        mn = np.nanmin(stack, axis=0)
        edge = (mx - mn) / np.maximum(depth, 1e-6) > rtol
    return edge & valid


def make_valid(depth, z_far):
    valid = (depth > 0) & np.isfinite(depth)
    if z_far and z_far > 0:
        valid &= depth < z_far
    return valid


def reproject_stats(depths, valids, Ks, c2ws, tau, stride=2, src_masks=None, max_pairs=None):
    """
    对所有有序帧对 (i, j)，把 i 帧（下采样 stride）的有效点投到 j 帧。
    src_masks: 可选 dict{name: [HxW bool per view]}，对这些子集单独统计 viol。
    返回计数字典（可跨样本累加）。
    """
    F = len(depths)
    H, W = depths[0].shape
    rad = [radial_norm(H, W, K) for K in Ks]
    rays = [pixel_rays(H, W, K) for K in Ks]
    w2cs = [np.linalg.inv(T) for T in c2ws]

    acc = {"n": 0, "cons": 0, "viol": 0,
           "n_center": 0, "cons_center": 0, "n_corner": 0, "cons_corner": 0}
    if src_masks:
        for k in src_masks:
            acc[f"n_{k}"] = 0
            acc[f"viol_{k}"] = 0
            acc[f"cons_{k}"] = 0

    pairs = [(i, j) for i in range(F) for j in range(F) if i != j]
    if max_pairs is not None and len(pairs) > max_pairs:
        sel = np.linspace(0, len(pairs) - 1, max_pairs).round().astype(int)
        pairs = [pairs[k] for k in sel]

    sl = (slice(None, None, stride), slice(None, None, stride))
    with np.errstate(all="ignore"):
        for i, j in pairs:
            _accumulate_pair(i, j, depths, valids, Ks, c2ws, w2cs, rays, rad, tau, sl, src_masks, acc, H, W)
    return acc


def _accumulate_pair(i, j, depths, valids, Ks, c2ws, w2cs, rays, rad, tau, sl, src_masks, acc, H, W):
    m_i = valids[i][sl]
    if not m_i.any():
        return
    X_cam = rays[i][sl][m_i] * depths[i][sl][m_i][:, None]
    X_w = X_cam @ c2ws[i][:3, :3].T + c2ws[i][:3, 3]
    X_j = X_w @ w2cs[j][:3, :3].T + w2cs[j][:3, 3]
    z = X_j[:, 2]
    front = z > 1e-6
    u = Ks[j][0, 0] * X_j[:, 0] / np.where(front, z, 1) + Ks[j][0, 2]
    v = Ks[j][1, 1] * X_j[:, 1] / np.where(front, z, 1) + Ks[j][1, 2]
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not inb.any():
        return
    ui_c, vi_c = ui[inb], vi[inb]
    ok = valids[j][vi_c, ui_c]
    if not ok.any():
        return
    d_j = depths[j][vi_c, ui_c][ok]
    rel = (z[inb][ok] - d_j) / d_j
    cons = np.abs(rel) < tau
    viol = rel < -tau

    acc["n"] += rel.size
    acc["cons"] += int(cons.sum())
    acc["viol"] += int(viol.sum())

    r = rad[j][vi_c, ui_c][ok]
    c_m, k_m = r < 0.3, r > 0.7
    acc["n_center"] += int(c_m.sum())
    acc["cons_center"] += int(cons[c_m].sum())
    acc["n_corner"] += int(k_m.sum())
    acc["cons_corner"] += int(cons[k_m].sum())

    if src_masks:
        idx_inb = np.flatnonzero(inb)[ok]
        for k, masks in src_masks.items():
            sub = masks[i][sl][m_i][idx_inb]
            acc[f"n_{k}"] += int(sub.sum())
            acc[f"viol_{k}"] += int(viol[sub].sum())
            acc[f"cons_{k}"] += int(cons[sub].sum())


def merge(acc_total, acc):
    for k, v in acc.items():
        acc_total[k] = acc_total.get(k, 0) + v


def ratio(a, b):
    return float(a) / b if b else float("nan")


# ---------------------------------------------------------------------------
# dataset construction
# ---------------------------------------------------------------------------
def load_cfg(path):
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


def build_dataset(cfg_all, name, resolution, frames, mode, seed):
    from omegaconf import OmegaConf
    import torchvision.transforms as tvf

    cfg = OmegaConf.to_container(cfg_all.train_dataset[name], resolve=True)
    target = cfg.pop("_target_")
    cfg.pop("transform", None)
    mod_name, cls_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    cfg.update(dict(resolution=[list(resolution)], transform=tvf.ToTensor(),
                    aug_crop=False, aug_focal=False, shuffle=False, mode=mode,
                    frame_num=frames, seed=seed))
    ds = cls(**cfg)
    ds.frame_num = frames
    return ds, cfg.get("z_far", 0) or getattr(ds, "z_far", 0)


_ORIG_THRESHOLD = None


def set_percentile_threshold(enabled):
    """--raw-compare：把各 loader 模块里的 threshold_depth_map 换成只保留 max_depth 截断的版本。"""
    global _ORIG_THRESHOLD
    import datasets.sample_utils.common as C
    if _ORIG_THRESHOLD is None:
        _ORIG_THRESHOLD = C.threshold_depth_map

    def no_percentile(depth_map, max_percentile=-1, min_percentile=-1, max_depth=-1):
        return _ORIG_THRESHOLD(depth_map, max_percentile=-1, min_percentile=-1, max_depth=max_depth)

    fn = _ORIG_THRESHOLD if enabled else no_percentile
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("datasets") and mod is not None:
            cur = getattr(mod, "threshold_depth_map", None)
            if cur is _ORIG_THRESHOLD or getattr(cur, "__name__", "") == "no_percentile":
                setattr(mod, "threshold_depth_map", fn)


def fetch_views(ds, idx, resolution, seed):
    rng = np.random.default_rng(seed)
    return ds._get_views(idx, list(resolution), rng)


def unpack(views, z_far):
    depths, valids, Ks, c2ws, dmasks = [], [], [], [], []
    for v in views:
        d = np.nan_to_num(np.asarray(v["depthmap"], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        depths.append(d)
        valids.append(make_valid(d, z_far))
        Ks.append(np.asarray(v["camera_intrinsics"], dtype=np.float64))
        c2ws.append(np.asarray(v["camera_pose"], dtype=np.float64))
        dm = v.get("depth_mask", None)
        dmasks.append(None if dm is None else np.asarray(dm).astype(bool))
    return depths, valids, Ks, c2ws, dmasks


# ---------------------------------------------------------------------------
# per-dataset audit
# ---------------------------------------------------------------------------
SCALE_GRID = np.concatenate([10.0 ** np.arange(-3, 3.01, 0.25), [1.0]])


def best_scale(depths, valids, Ks, c2ws, tau, stride):
    """只缩放深度、位姿不动，取一致点数最多的尺度。用点数而不是比例：尺度极小时点全部投出画面，比例无意义。"""
    best_s, best_c = 1.0, -1
    for s in SCALE_GRID:
        b = reproject_stats([d * s for d in depths], valids, Ks, c2ws, tau, stride, None, 12)
        if b["cons"] > best_c or (b["cons"] == best_c and np.isclose(s, 1.0)):
            best_s, best_c = float(s), b["cons"]
    return best_s


def audit_dataset(cfg_all, name, args):
    t0 = time.time()
    resolution = tuple(args.res)
    ds, z_far = build_dataset(cfg_all, name, resolution, args.frames, args.mode, args.seed)
    n_total = len(ds)
    idxs = np.unique(np.linspace(0, n_total - 1, min(args.n, n_total)).round().astype(int))

    acc_raw, acc_ray, acc_masks = {}, {}, {}
    scale_votes, frame_stats, failures = [], [], []
    pct_removed, pct_total, pct_cons = 0, 0, {}
    worst = []

    for k, idx in enumerate(idxs):
        seed = args.seed * 100003 + int(idx)
        try:
            set_percentile_threshold(True)
            views = fetch_views(ds, int(idx), resolution, seed)
        except Exception as e:
            failures.append(f"idx={idx}: {type(e).__name__}: {str(e)[:160]}")
            continue
        try:
            depths, valids, Ks, c2ws, dmasks = unpack(views, z_far)
            if not all(np.isfinite(T).all() for T in c2ws):
                failures.append(f"idx={idx}: non-finite camera_pose")
                continue

            # --- 帧级统计
            for d, m, T in zip(depths, valids, c2ws):
                R = T[:3, :3]
                frame_stats.append(dict(
                    valid_ratio=float(m.mean()),
                    p5=float(np.percentile(d[m], 5)) if m.any() else np.nan,
                    p50=float(np.percentile(d[m], 50)) if m.any() else np.nan,
                    p95=float(np.percentile(d[m], 95)) if m.any() else np.nan,
                    orth_err=float(np.abs(R @ R.T - np.eye(3)).max()),
                    det=float(np.linalg.det(R)),
                ))
            centers = np.stack([T[:3, 3] for T in c2ws])
            base = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
            med_depth = np.nanmedian([fs["p50"] for fs in frame_stats[-len(depths):]])
            frame_stats[-1]["baseline_over_depth"] = float(np.median(base[np.triu_indices(len(depths), 1)]) / max(med_depth, 1e-9))

            # --- 子集掩码：边缘 / 非边缘 / loader 自带 depth_mask 排除的像素
            src_masks = {
                "edge": [depth_edge_np(d, m) for d, m in zip(depths, valids)],
            }
            src_masks["flat"] = [m & ~e for m, e in zip(valids, src_masks["edge"])]
            if all(dm is not None for dm in dmasks):
                src_masks["mask_out"] = [m & ~dm for m, dm in zip(valids, dmasks)]
                src_masks["mask_in"] = [m & dm for m, dm in zip(valids, dmasks)]

            a = reproject_stats(depths, valids, Ks, c2ws, args.tau, args.stride, src_masks, args.max_pairs)
            merge(acc_raw, a)
            worst.append((ratio(a["viol"], a["n"]), int(idx), views[0].get("label", ""), [v.get("instance", "") for v in views]))

            # --- 假设 1：深度其实是 ray distance
            depths_z = [ray_to_z(d, K) for d, K in zip(depths, Ks)]
            valids_z = [make_valid(d, z_far) for d in depths_z]
            merge(acc_ray, reproject_stats(depths_z, valids_z, Ks, c2ws, args.tau, args.stride, None, args.max_pairs))

            # --- 假设 2：深度与位姿差一个全局尺度
            scale_votes.append(best_scale(depths, valids, Ks, c2ws, args.tau, args.stride * 2))

            # --- 98 分位截断去掉了什么（同一 seed => 同一组帧）
            if args.raw_compare:
                set_percentile_threshold(False)
                views_np = fetch_views(ds, int(idx), resolution, seed)
                set_percentile_threshold(True)
                d_np, v_np, K_np, T_np, _ = unpack(views_np, z_far)
                same = all(np.allclose(T1, T2) for T1, T2 in zip(c2ws, T_np))
                if same:
                    removed = [vn & ~vo for vn, vo in zip(v_np, valids)]
                    pct_removed += int(sum(r.sum() for r in removed))
                    pct_total += int(sum(vn.sum() for vn in v_np))
                    b = reproject_stats(d_np, v_np, K_np, T_np, args.tau, args.stride, {"removed": removed}, args.max_pairs)
                    merge(pct_cons, {k2: b[k2] for k2 in ("n_removed", "cons_removed", "viol_removed")})
        except Exception as e:
            failures.append(f"idx={idx} (metrics): {type(e).__name__}: {str(e)[:160]}")
            if args.verbose:
                traceback.print_exc()
        if args.verbose and (k + 1) % 5 == 0:
            print(f"  [{name}] {k + 1}/{len(idxs)}", flush=True)

    fs = {key: float(np.nanmedian([f[key] for f in frame_stats if key in f])) if frame_stats else float("nan")
          for key in ("valid_ratio", "p5", "p50", "p95", "orth_err", "det", "baseline_over_depth")}
    out = dict(
        dataset=name, n_total=n_total, n_sampled=len(idxs), n_fail=len(failures),
        z_far=z_far, tau=args.tau,
        cons=ratio(acc_raw.get("cons", 0), acc_raw.get("n", 0)),
        viol=ratio(acc_raw.get("viol", 0), acc_raw.get("n", 0)),
        cons_center=ratio(acc_raw.get("cons_center", 0), acc_raw.get("n_center", 0)),
        cons_corner=ratio(acc_raw.get("cons_corner", 0), acc_raw.get("n_corner", 0)),
        cons_if_ray2z=ratio(acc_ray.get("cons", 0), acc_ray.get("n", 0)),
        best_scale_median=float(np.median(scale_votes)) if scale_votes else float("nan"),
        best_scale_frac_eq1=float(np.mean(np.isclose(scale_votes, 1.0))) if scale_votes else float("nan"),
        edge_frac=ratio(acc_raw.get("n_edge", 0), acc_raw.get("n", 0)),
        edge_viol=ratio(acc_raw.get("viol_edge", 0), acc_raw.get("n_edge", 0)),
        flat_viol=ratio(acc_raw.get("viol_flat", 0), acc_raw.get("n_flat", 0)),
        mask_out_frac=ratio(acc_raw.get("n_mask_out", 0), acc_raw.get("n", 0)),
        mask_out_viol=ratio(acc_raw.get("viol_mask_out", 0), acc_raw.get("n_mask_out", 0)),
        mask_in_viol=ratio(acc_raw.get("viol_mask_in", 0), acc_raw.get("n_mask_in", 0)),
        pct_removed=ratio(pct_removed, pct_total) if args.raw_compare else float("nan"),
        pct_removed_cons=ratio(pct_cons.get("cons_removed", 0), pct_cons.get("n_removed", 0)) if args.raw_compare else float("nan"),
        pct_removed_viol=ratio(pct_cons.get("viol_removed", 0), pct_cons.get("n_removed", 0)) if args.raw_compare else float("nan"),
        **{f"frame_{k}": v for k, v in fs.items()},
        failures=failures[:5],
        worst_samples=sorted(worst, key=lambda x: -x[0] if np.isfinite(x[0]) else 0)[:5],
        seconds=round(time.time() - t0, 1),
    )
    return out


def flags(r):
    """把数字翻译成需要人看的结论。阈值是经验值，只用于排优先级。"""
    msg = []
    if r["n_sampled"] and r["n_fail"] / r["n_sampled"] > 0.05:
        msg.append(f"加载失败率 {r['n_fail']}/{r['n_sampled']}")
    if np.isfinite(r["cons_if_ray2z"]) and r["cons_if_ray2z"] - r["cons"] > 0.05:
        msg.append("ray→z 后一致性明显变好：深度很可能是射线距离")
    if np.isfinite(r["best_scale_median"]) and not (0.75 <= r["best_scale_median"] <= 1.34):
        msg.append(f"深度与位姿尺度不一致，最佳缩放约 {r['best_scale_median']:.3g}")
    if np.isfinite(r["viol"]) and r["viol"] > 0.10:
        msg.append(f"自由空间违例 {r['viol']:.0%}：位姿或深度有系统误差")
    if np.isfinite(r["edge_viol"]) and np.isfinite(r["flat_viol"]) and r["edge_viol"] > 3 * max(r["flat_viol"], 0.005) and r["edge_viol"] - r["flat_viol"] > 0.08:
        msg.append(f"边缘飞点：edge viol {r['edge_viol']:.0%} vs flat {r['flat_viol']:.0%}")
    if np.isfinite(r["mask_out_viol"]) and np.isfinite(r["mask_in_viol"]) and r["mask_out_viol"] > 2 * max(r["mask_in_viol"], 0.005):
        msg.append(f"loader 自带 depth_mask 排除的像素更差（{r['mask_out_viol']:.0%} vs {r['mask_in_viol']:.0%}），应接入 valid_mask")
    if np.isfinite(r["pct_removed"]) and r["pct_removed"] > 0.01 and np.isfinite(r["pct_removed_cons"]) and r["pct_removed_cons"] >= 0.8 * r["cons"]:
        msg.append(f"98 分位截掉 {r['pct_removed']:.1%} 有效像素，且它们和保留像素一样一致：截断在删好数据")
    if np.isfinite(r["frame_orth_err"]) and r["frame_orth_err"] > 1e-3:
        msg.append("旋转矩阵非正交")
    if np.isfinite(r["frame_det"]) and r["frame_det"] < 0:
        msg.append("旋转矩阵行列式为负（镜像坐标系）")
    if np.isfinite(r["frame_valid_ratio"]) and r["frame_valid_ratio"] < 0.1:
        msg.append(f"有效像素中位数仅 {r['frame_valid_ratio']:.0%}")
    return msg


def print_table(rows):
    cols = [("dataset", "{:<16}"), ("cons", "{:>6.1%}"), ("viol", "{:>6.1%}"),
            ("cons_center", "{:>6.1%}"), ("cons_corner", "{:>6.1%}"), ("cons_if_ray2z", "{:>6.1%}"),
            ("best_scale_median", "{:>7.3g}"), ("edge_viol", "{:>6.1%}"), ("flat_viol", "{:>6.1%}"),
            ("pct_removed", "{:>6.1%}"), ("pct_removed_cons", "{:>6.1%}"),
            ("frame_p50", "{:>7.2f}"), ("frame_valid_ratio", "{:>6.1%}"), ("n_fail", "{:>4d}")]
    head = ["dataset", "cons", "viol", "ctr", "corner", "ray2z", "scale", "edgeV", "flatV",
            "pctRm", "pctRmC", "d_p50", "valid", "fail"]
    print("  ".join(f"{h:>7}" if i else f"{h:<16}" for i, h in enumerate(head)))
    for r in rows:
        cells = []
        for (k, fmt), h in zip(cols, head):
            v = r.get(k)
            try:
                cells.append(fmt.format(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else f"{'-':>7}")
            except (ValueError, TypeError):
                cells.append(f"{str(v):>7}")
        print("  ".join(cells))
    print()
    for r in rows:
        for m in flags(r):
            print(f"[{r['dataset']}] {m}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/datamix_all_v10-1.yaml")
    ap.add_argument("--datasets", nargs="*", default=None, help="config 里 train_dataset 下的名字，默认 weights 全部")
    ap.add_argument("--n", type=int, default=30, help="每个数据集抽多少个样本")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--res", type=int, nargs=2, default=[512, 336], metavar=("W", "H"))
    ap.add_argument("--mode", default="train")
    ap.add_argument("--tau", type=float, default=0.03, help="相对深度误差阈值；真实数据可放到 0.05")
    ap.add_argument("--stride", type=int, default=2, help="源帧像素下采样步长")
    ap.add_argument("--max-pairs", type=int, default=56, help="每个样本最多比较多少个有序帧对")
    ap.add_argument("--raw-compare", action="store_true", help="额外跑一遍关闭 98 分位截断的版本做对比")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gt_audit_out")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg_all = load_cfg(args.config)
    names = args.datasets or list(cfg_all.train_dataset.weights.keys())
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for name in names:
        if name not in cfg_all.train_dataset:
            print(f"[skip] {name} 不在 config 的 train_dataset 里")
            continue
        print(f"==> {name}", flush=True)
        try:
            r = audit_dataset(cfg_all, name, args)
        except Exception as e:
            print(f"[{name}] 构建失败: {type(e).__name__}: {e}")
            if args.verbose:
                traceback.print_exc()
            continue
        rows.append(r)
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(r, f, ensure_ascii=False, indent=2, default=str)
        print(f"    cons={r['cons']:.1%} viol={r['viol']:.1%} ray2z={r['cons_if_ray2z']:.1%} "
              f"scale={r['best_scale_median']:.3g} fail={r['n_fail']}/{r['n_sampled']} ({r['seconds']}s)", flush=True)

    print()
    print_table(rows)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
