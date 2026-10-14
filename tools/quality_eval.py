#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标签质量评估（M1–M7，有真值时加 G1）。指标实现在 utils/quality_metrics.py，定义见报告"质量指标定义"一页。

两种输入：

1) 训练数据集（GT 体检）：复用 config 里的 loader，每个数据集均匀取 N 个样本，每个样本 F 帧。
     python tools/quality_eval.py dataset --config configs/data/datamix_all_v9-5.yaml \
         --datasets Hypersim DL3DV --n 30 --out quality_out
   额外的 M7 规则评估：loader 返回 depth_mask 时自动评估 "depth_mask"；--raw-compare 时评估 "percentile98"。

2) 伪标签目录（无真值也可以跑）：
     <root>/<序列名>/*.npz        每帧一个 npz（按文件名排序）
       必需：depth [H,W]（z-depth）、K [3,3]、c2w [4,4]（相机到世界）
       可选：valid [H,W] bool、sky [H,W] bool、dyn [H,W] bool、conf [H,W] float、gt_depth [H,W]（有真值时）
     python tools/quality_eval.py pseudo --root /path/to/pseudo --frames 8 --out quality_out \
         [--conf-thr 0.5] [--save-masks] [--thresholds calib.json]
   每条序列按 F 帧一个窗口切分（--window-stride 默认 = F，不重叠），窗口内算 M1–M6，按计数合并成序列级指标；
   有 gt_depth 时以"已验证像素"为保留集合算 G1。

输出（--out 目录）：
  dataset 模式：<数据集>.json（汇总 + 分档统计）、<数据集>_samples.jsonl（逐样本）
  pseudo  模式：sequences.jsonl（逐序列：指标、分档、原因、G1）、summary.json；--save-masks 时 masks/<序列>/<帧>.npz
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils import quality_metrics as Q  # noqa: E402


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_thresholds(path):
    """读取 calibrate_thresholds.py 的输出：{"tau":…, "k_verify":…, "thresholds":{…}}。返回 (cfg_overrides, thresholds)。"""
    if not path:
        return {}, None
    with open(path) as f:
        d = json.load(f)
    th = {k: tuple(tuple(x) if isinstance(x, list) else x for x in v) for k, v in d.get("thresholds", {}).items()}
    return {k: d[k] for k in ("tau", "k_verify", "min_tri_deg") if k in d}, ({**Q.DEFAULT_THRESHOLDS, **th} if th else None)


def summarize_grades(rows):
    c = Counter(r["grade"] for r in rows)
    n = max(len(rows), 1)
    return {g: {"count": c.get(g, 0), "frac": c.get(g, 0) / n} for g in ("A", "B", "C", "reject")}


def compact(m):
    return {k: v for k, v in m.items() if not isinstance(v, list)}


# ---------------------------------------------------------------------------
# dataset mode
# ---------------------------------------------------------------------------
def run_dataset_mode(args, cfg, thresholds):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import gt_audit as GA  # 复用 loader 构建、取样（同一 seed 公式）与分位截断开关

    cfg_all = GA.load_cfg(args.config)
    names = args.datasets or list(cfg_all.train_dataset.weights.keys())
    os.makedirs(args.out, exist_ok=True)
    table = []
    for name in names:
        t0 = time.time()
        row = dict(dataset=name, status="OK")
        if name not in cfg_all.train_dataset:
            row.update(status="NOT_IN_CONFIG")
            table.append(row)
            continue
        root = cfg_all.train_dataset[name].get("data_root", None)
        if root is None or not os.path.exists(str(root)):
            print(f"[ERROR][{name}] PATH_NOT_FOUND: data_root 不存在: {root}", flush=True)
            row.update(status="PATH_NOT_FOUND")
            table.append(row)
            continue
        try:
            ds, z_far = GA.build_dataset(cfg_all, name, tuple(args.res), args.frames, args.mode, args.seed)
        except Exception as e:
            print(f"[ERROR][{name}] BUILD_ERROR: {type(e).__name__}: {e}", flush=True)
            row.update(status="BUILD_ERROR", error=str(e)[:300])
            table.append(row)
            continue
        idxs = np.unique(np.linspace(0, len(ds) - 1, min(args.n, len(ds))).round().astype(int))
        results, samples, ffl, fails = [], [], {}, []
        print(f"==> {name}: {len(idxs)} samples", flush=True)
        for idx in idxs:
            seed = args.seed * 100003 + int(idx)
            try:
                GA.set_percentile_threshold(True)
                views = GA.fetch_views(ds, int(idx), tuple(args.res), seed)
                depths, valids, Ks, c2ws, dmasks = GA.unpack(views, z_far)
                s = Q.prepare_sample(np.stack(depths), np.stack(Ks), np.stack(c2ws), valids=np.stack(valids))
                r = Q.compute_metrics(s, cfg, with_scale_search=not args.no_scale_search, with_ray2z=True)
                g, reasons, _ = Q.grade(r["metrics"], thresholds)
                results.append(r)
                rules = {}
                if all(dm is not None for dm in dmasks):
                    rules["depth_mask"] = Q.filter_fidelity(s, ~np.stack(dmasks), cfg, counts=r["raw"])
                if args.raw_compare:
                    GA.set_percentile_threshold(False)
                    try:
                        vnp = GA.fetch_views(ds, int(idx), tuple(args.res), seed)
                    finally:
                        GA.set_percentile_threshold(True)
                    d2, v2, K2, T2, _ = GA.unpack(vnp, z_far)
                    if all(np.allclose(a, b) for a, b in zip(c2ws, T2)):
                        s_np = Q.prepare_sample(np.stack(d2), np.stack(K2), np.stack(T2), valids=np.stack(v2))
                        rules["percentile98"] = Q.filter_fidelity(s_np, s_np.valids & ~s.valids, cfg)
                for k, v in rules.items():
                    ffl.setdefault(k, []).append(v)
                r.pop("raw", None)          # 逐像素计数图很大，合并只需要 counts
                r.pop("masks", None)
                samples.append(dict(idx=int(idx), label=str(views[0].get("label", "")), grade=g, reasons=reasons,
                                    metrics=compact(r["metrics"]), ffl=rules))
            except Exception as e:
                fails.append(f"idx={idx}: {type(e).__name__}: {str(e)[:200]}")
                if args.verbose:
                    traceback.print_exc()
        agg = Q.aggregate_counts(results, cfg.bfr_floor) if results else {}
        ffl_agg = {}
        for k, lst in ffl.items():  # 计数加权：按被删像素数加权平均
            w = np.array([x["n_removed_observed"] for x in lst], float)
            def wavg(key):
                vals = np.array([x[key] for x in lst], float)
                ok = np.isfinite(vals) & (w > 0)
                return float((vals[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")
            ffl_agg[k] = {"rr": float(np.nanmean([x["rr"] for x in lst])), "fsv_removed": wavg("fsv_removed"),
                          "fsv_kept": wavg("fsv_kept"), "mvc_removed": wavg("mvc_removed"), "mvc_kept": wavg("mvc_kept")}
            fk, mk = ffl_agg[k]["fsv_kept"], ffl_agg[k]["mvc_kept"]
            ffl_agg[k]["lift"] = ffl_agg[k]["fsv_removed"] / fk if fk and fk > 0 else float("nan")
            ffl_agg[k]["good_loss"] = ffl_agg[k]["mvc_removed"] / mk if mk else float("nan")
        g, reasons, _ = Q.grade(agg, thresholds) if agg else ("na", [], {})
        row.update(n_samples=len(results), n_fail=len(fails), dataset_grade=g, reasons=reasons,
                   sample_grades=summarize_grades(samples), metrics=agg, ffl=ffl_agg, failures=fails[:5],
                   seconds=round(time.time() - t0, 1))
        if fails and not results:
            row["status"] = "ALL_SAMPLES_FAILED"
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(row, f, ensure_ascii=False, indent=2, default=_json_default)
        with open(os.path.join(args.out, f"{name}_samples.jsonl"), "w") as f:
            for srow in samples:
                f.write(json.dumps(srow, ensure_ascii=False, default=_json_default) + "\n")
        table.append(row)
        print(f"    fsv={agg.get('fsv', float('nan')):.2%} vpr={agg.get('vpr', float('nan')):.2%} "
              f"tri={agg.get('tri_angle_median', float('nan')):.2f}° drift={agg.get('drift', float('nan')):.3f} "
              f"bfr={agg.get('bfr', float('nan')):.1f} grade={g} ({row['seconds']}s)", flush=True)
    print_table(table, key="dataset")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(table, f, ensure_ascii=False, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# pseudo-label mode
# ---------------------------------------------------------------------------
def find_sequences(root):
    seqs = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.npz")):
            seqs.append(d)
    if not seqs and glob.glob(os.path.join(root, "*.npz")):
        seqs = [root]
    return seqs


def load_frames(seq_dir):
    frames = []
    for p in sorted(glob.glob(os.path.join(seq_dir, "*.npz"))):
        with np.load(p, allow_pickle=False) as z:
            f = {k: z[k] for k in z.files}
        for k in ("depth", "K", "c2w"):
            if k not in f:
                raise KeyError(f"{p} 缺少字段 {k}")
        f["name"] = os.path.splitext(os.path.basename(p))[0]
        frames.append(f)
    return frames


def eval_sequence(frames, cfg, thresholds=None, F=8, window_stride=None, frame_step=1, z_far=0.0,
                  conf_thr=None, with_scale_search=True, mask_out_dir=None):
    """评估一条伪标签序列。返回逐序列结果 dict（可被 calibrate_thresholds 复用）。"""
    frames = frames[::max(frame_step, 1)]
    stride = window_stride or F
    starts = list(range(0, max(len(frames) - F, 0) + 1, stride)) or [0]
    results, ffl, gt_L, gt_G, gt_K = [], [], [], [], []
    for st in starts:
        win = frames[st:st + F]
        if len(win) < 2:
            continue
        extra = []
        for key, inv in (("valid", False), ("sky", True)):
            if all(key in f for f in win):
                m = np.stack([np.asarray(f[key], bool) for f in win])
                extra.append(~m if inv else m)
        dyn = np.stack([np.asarray(f["dyn"], bool) for f in win]) if all("dyn" in f for f in win) else None
        depth = np.stack([f["depth"] for f in win])
        s = Q.prepare_sample(depth, np.stack([f["K"] for f in win]), np.stack([f["c2w"] for f in win]),
                             z_far=z_far, dyn=dyn, extra_masks=extra)
        r = Q.compute_metrics(s, cfg, with_scale_search=with_scale_search)
        results.append(r)
        if conf_thr is not None and all("conf" in f for f in win):
            conf = np.stack([np.asarray(f["conf"], float) for f in win])
            ffl.append(Q.filter_fidelity(s, conf < conf_thr, cfg, counts=r["raw"]))
        if all("gt_depth" in f for f in win):
            gt_L.append(s.depths)
            gt_G.append(np.stack([np.asarray(f["gt_depth"], float) for f in win]))
            gt_K.append(r["masks"]["verified"])
        r.pop("raw", None)
        if mask_out_dir:
            os.makedirs(mask_out_dir, exist_ok=True)
            for k, f in enumerate(win):
                np.savez_compressed(os.path.join(mask_out_dir, f"{f['name']}.npz"),
                                    verified=r["masks"]["verified"][k], viol=r["masks"]["viol"][k],
                                    boundary=r["masks"]["boundary"][k])
        r.pop("masks", None)
    if not results:
        return dict(n_frames=len(frames), grade="reject", reasons=["fewer than 2 frames"], metrics={})
    agg = Q.aggregate_counts(results, cfg.bfr_floor)
    g, reasons, bands = Q.grade(agg, thresholds)
    out = dict(n_frames=len(frames), n_windows=len(results), grade=g, reasons=reasons, bands=bands, metrics=agg)
    if gt_L:
        out["gt"] = Q.gt_metrics(np.concatenate(gt_L), np.concatenate(gt_G), np.concatenate(gt_K))
    if ffl:
        out["ffl_conf"] = {k: float(np.nanmean([x[k] for x in ffl])) for k in ("rr", "lift", "good_loss")}
    return out


def run_pseudo_mode(args, cfg, thresholds):
    seqs = find_sequences(args.root)
    if not seqs:
        print(f"[ERROR] {args.root} 下没有找到 npz 序列", flush=True)
        return
    os.makedirs(args.out, exist_ok=True)
    rows = []
    with open(os.path.join(args.out, "sequences.jsonl"), "w") as fo:
        for d in seqs:
            name = os.path.relpath(d, args.root) if d != args.root else os.path.basename(d.rstrip("/"))
            t0 = time.time()
            try:
                frames = load_frames(d)
                r = eval_sequence(frames, cfg, thresholds, F=args.frames, window_stride=args.window_stride,
                                  frame_step=args.frame_step, z_far=args.z_far, conf_thr=args.conf_thr,
                                  with_scale_search=not args.no_scale_search,
                                  mask_out_dir=os.path.join(args.out, "masks", name) if args.save_masks else None)
            except Exception as e:
                r = dict(grade="error", reasons=[f"{type(e).__name__}: {str(e)[:200]}"], metrics={})
                if args.verbose:
                    traceback.print_exc()
            r["sequence"] = name
            r["seconds"] = round(time.time() - t0, 1)
            fo.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
            rows.append(r)
            m = r.get("metrics", {})
            gt = r.get("gt", {})
            print(f"  {name}: grade={r['grade']} fsv={m.get('fsv', float('nan')):.2%} vpr={m.get('vpr', float('nan')):.2%} "
                  f"tri={m.get('tri_angle_median', float('nan')):.2f}° drift={m.get('drift', float('nan')):.3f}"
                  + (f" absrel_kept={gt.get('abs_rel_kept', float('nan')):.3f} sel={gt.get('sel', float('nan')):.2f}" if gt else "")
                  + (f"  [{'; '.join(r['reasons'])}]" if r.get("reasons") else ""), flush=True)
    ok = [r for r in rows if r.get("metrics")]
    summary = dict(root=args.root, n_sequences=len(rows), grades=summarize_grades(rows),
                   metrics_median={k: float(np.nanmedian([r["metrics"].get(k, np.nan) for r in ok]))
                                   for k in ("fsv", "mvc", "vpr", "vcov", "drift", "tri_angle_median", "bfr", "valid_ratio")} if ok else {})
    gts = [r["gt"] for r in rows if "gt" in r]
    if gts:
        nk = np.array([g["n_kept_gt"] for g in gts], float)
        def pooled(key):
            v = np.array([g[key] for g in gts], float)
            m = np.isfinite(v) & (nk > 0)
            return float((v[m] * nk[m]).sum() / nk[m].sum()) if m.any() else float("nan")
        summary["gt"] = dict(abs_rel_kept=pooled("abs_rel_kept"), delta1_kept=pooled("delta1_kept"),
                             gt_cov=float(sum(g["n_kept_gt"] for g in gts) / max(sum(g["n_gt"] for g in gts), 1)))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


# ---------------------------------------------------------------------------
def _fmt(v, f):
    try:
        return f.format(v) if v is not None and np.isfinite(v) else "-"
    except (TypeError, ValueError):
        return str(v)


def print_table(rows, key="dataset"):
    cols = [("fsv", "{:.2%}"), ("mvc", "{:.1%}"), ("vpr", "{:.1%}"), ("best_scale", "{:.3g}"), ("drift", "{:.3f}"),
            ("tri_angle_median", "{:.2f}"), ("baseline_over_depth", "{:.3f}"), ("bfr", "{:.1f}"), ("valid_ratio", "{:.1%}")]
    head = [key, "status", "grade"] + [c for c, _ in cols] + ["A/B/C/rej"]
    print("\n" + "\t".join(head))
    for r in rows:
        m = r.get("metrics", {}) or {}
        sg = r.get("sample_grades", {})
        dist = "/".join(str(sg.get(g, {}).get("count", 0)) for g in ("A", "B", "C", "reject")) if sg else "-"
        print("\t".join([str(r.get(key, "")), str(r.get("status", "")), str(r.get("dataset_grade", r.get("grade", "")))]
                        + [_fmt(m.get(c), f) for c, f in cols] + [dist]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="quality_out")
    common.add_argument("--frames", type=int, default=8)
    common.add_argument("--tau", type=float, default=None)
    common.add_argument("--k-verify", type=int, default=None)
    common.add_argument("--stride", type=int, default=1, help="源像素下采样步长（加速；掩码仍为全分辨率）")
    common.add_argument("--max-pairs", type=int, default=None)
    common.add_argument("--no-scale-search", action="store_true", help="跳过 M4 s* 搜索（最耗时的一步）")
    common.add_argument("--thresholds", default=None, help="calibrate_thresholds.py 输出的 json")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--verbose", action="store_true")

    d = sub.add_parser("dataset", parents=[common], help="训练数据集（GT 体检）")
    d.add_argument("--config", default="configs/data/datamix_all_v9-5.yaml")
    d.add_argument("--datasets", nargs="*", default=None)
    d.add_argument("--n", type=int, default=30)
    d.add_argument("--res", type=int, nargs=2, default=[512, 336], metavar=("W", "H"))
    d.add_argument("--mode", default="train")
    d.add_argument("--raw-compare", action="store_true", help="额外评估 98 分位截断这条规则（M7）")

    p = sub.add_parser("pseudo", parents=[common], help="伪标签目录（每帧一个 npz）")
    p.add_argument("--root", required=True)
    p.add_argument("--window-stride", type=int, default=None, help="窗口起点间隔，默认 = frames（不重叠）")
    p.add_argument("--frame-step", type=int, default=1, help="先按此步长抽帧，再切窗口（用来增大视差）")
    p.add_argument("--z-far", type=float, default=0.0)
    p.add_argument("--conf-thr", type=float, default=None, help="评估“conf < 阈值则删除”这条规则（M7）")
    p.add_argument("--save-masks", action="store_true", help="保存每帧的 verified / viol / boundary 掩码")
    args = ap.parse_args()

    overrides, thresholds = load_thresholds(args.thresholds)
    if args.tau is not None:
        overrides["tau"] = args.tau
    if args.k_verify is not None:
        overrides["k_verify"] = args.k_verify
    cfg = Q.QualityConfig(stride=args.stride, max_pairs=args.max_pairs, seed=args.seed, **overrides)
    print(f"[config] tau={cfg.tau} k_verify={cfg.k_verify} min_tri_deg={cfg.min_tri_deg} stride={cfg.stride} "
          f"thresholds={'custom' if thresholds else 'default'}", flush=True)
    if args.cmd == "dataset":
        run_dataset_mode(args, cfg, thresholds)
    else:
        run_pseudo_mode(args, cfg, thresholds)


if __name__ == "__main__":
    main()
