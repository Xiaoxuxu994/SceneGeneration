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

3) 合并并行结果：python tools/quality_eval.py summarize --out quality_out [--datasets ...]
   一键脚本见 tools/run_quality_eval.sh。

输出（--out 目录）：
  dataset 模式：<数据集>.json（汇总 + 分档统计）、<数据集>_samples.jsonl（逐样本）、
               summary.json、summary_compact.tsv（违例率 / 可用率 / 视差 + 尺度 / 边界检查）、summary_full.tsv（全部指标）
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


GRADES = ("A", "B", "C", "U", "reject")   # U = 无法验证（视差 / 有效像素 / 可验证像素不足），不等于标签错


def summarize_grades(rows):
    c = Counter(r["grade"] for r in rows)
    n = max(len(rows), 1)
    return {g: {"count": c.get(g, 0), "frac": c.get(g, 0) / n} for g in GRADES}


def compact(m):
    return {k: v for k, v in m.items() if not isinstance(v, list)}


# ---------------------------------------------------------------------------
# dataset mode
# ---------------------------------------------------------------------------
def _dump_row(out, row):
    with open(os.path.join(out, f"{row['dataset']}.json"), "w") as f:
        json.dump(row, f, ensure_ascii=False, indent=2, default=_json_default)


def dataset_switches(ds):
    """BaseDataset 上的数据过滤开关（xiaoxu/Pi3_training_v2，见 datasets/FILTER_SWITCHES.md）；旧版 loader 上全部视为关闭。"""
    return dict(pct_valid_only=bool(getattr(ds, "pct_valid_only", False)),
                pct_disable=bool(getattr(ds, "pct_disable", False)),
                use_dataset_masks=bool(getattr(ds, "use_dataset_masks", False)),
                min_baseline_ratio=float(getattr(ds, "min_baseline_ratio", 0.0) or 0.0),
                baseline_retries=int(getattr(ds, "baseline_retries", 3)),
                seq_blacklist=bool(getattr(ds, "seq_blacklist", None)))


def _set_depth_filter(sw):
    try:
        from datasets.sample_utils.common import set_depth_filter
    except ImportError:
        return
    set_depth_filter(sw["pct_valid_only"], sw["pct_disable"])


def _merge_extra_valid(views, valids, sw):
    if not sw["use_dataset_masks"]:
        return valids
    out = []
    for v, m in zip(views, valids):
        ev = v.get("extra_valid", None)
        out.append(m & np.asarray(ev).astype(bool) if ev is not None and np.shape(ev) == m.shape else m)
    return out


def _baseline_over_depth(c2ws, depths, valids):
    c = np.stack([T[:3, 3] for T in c2ws])
    if len(c) < 2:
        return float("inf")
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)[np.triu_indices(len(c), 1)]
    meds = [np.median(x[m]) for x, m in zip(depths, valids) if m.any()]
    return float(np.median(d) / np.median(meds)) if meds and np.median(meds) > 0 else float("inf")


def fetch_with_switches(GA, ds, idx, res, seed, z_far, sw):
    """与训练时 BaseDataset.__getitem__ 一致地取样本：分位开关、extra_valid 合并、视差不足时换种子重选帧。
    返回 (views, depths, valids, Ks, c2ws, dmasks, 实际使用的 seed)。"""
    tries = 1 + (sw["baseline_retries"] if sw["min_baseline_ratio"] > 0 else 0)
    for t in range(tries):
        sd = seed + 7777 * t
        _set_depth_filter(sw)
        views = GA.fetch_views(ds, idx, res, sd)
        depths, valids, Ks, c2ws, dmasks = GA.unpack(views, z_far)
        valids = _merge_extra_valid(views, valids, sw)
        if sw["min_baseline_ratio"] <= 0 or _baseline_over_depth(c2ws, depths, valids) >= sw["min_baseline_ratio"]:
            break
    return views, depths, valids, Ks, c2ws, dmasks, sd


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
            _dump_row(args.out, row)
            table.append(row)
            continue
        root = cfg_all.train_dataset[name].get("data_root", None)
        if root is None or not GA.root_exists(cfg_all.train_dataset[name]):
            print(f"[ERROR][{name}] PATH_NOT_FOUND: data_root 不存在: {root}", flush=True)
            row.update(status="PATH_NOT_FOUND")
            _dump_row(args.out, row)
            table.append(row)
            continue
        try:
            ds, z_far = GA.build_dataset(cfg_all, name, tuple(args.res), args.frames, args.mode, args.seed)
        except Exception as e:
            print(f"[ERROR][{name}] BUILD_ERROR: {type(e).__name__}: {e}", flush=True)
            row.update(status="BUILD_ERROR", error=str(e)[:300])
            _dump_row(args.out, row)
            table.append(row)
            continue
        sw = dataset_switches(ds)
        if sw["seq_blacklist"] and hasattr(ds, "_apply_blacklist"):
            ds._apply_blacklist()
        row["switches"] = sw
        idxs = np.unique(np.linspace(0, len(ds) - 1, min(args.n, len(ds))).round().astype(int))
        results, samples, ffl, fails = [], [], {}, []
        print(f"==> {name}: {len(idxs)} samples", flush=True)
        for idx in idxs:
            seed = args.seed * 100003 + int(idx)
            try:
                GA.set_percentile_threshold(True)
                views, depths, valids, Ks, c2ws, dmasks, seed = fetch_with_switches(
                    GA, ds, int(idx), tuple(args.res), seed, z_far, sw)
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
                        _set_depth_filter(sw)
                        vnp = GA.fetch_views(ds, int(idx), tuple(args.res), seed)
                    finally:
                        GA.set_percentile_threshold(True)
                    d2, v2, K2, T2, _ = GA.unpack(vnp, z_far)
                    v2 = _merge_extra_valid(vnp, v2, sw)
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
        _dump_row(args.out, row)
        with open(os.path.join(args.out, f"{name}_samples.jsonl"), "w") as f:
            for srow in samples:
                f.write(json.dumps(srow, ensure_ascii=False, default=_json_default) + "\n")
        table.append(row)
        print(f"    fsv={agg.get('fsv', float('nan')):.2%} vpr={agg.get('vpr', float('nan')):.2%} "
              f"tri={agg.get('tri_angle_median', float('nan')):.2f}° drift={agg.get('drift', float('nan')):.3f} "
              f"bfr={agg.get('bfr', float('nan')):.1f} grade={g} ({row['seconds']}s)", flush=True)
    if not args.no_summary:
        write_dataset_summary(table, args.out, args.full)


def write_dataset_summary(table, out, full=False):
    """终端打印精简表（--full 打印完整表）；完整表总是另存为 summary_full.tsv，可直接贴 Excel。"""
    print_table(table, key="dataset", full=full)
    with open(os.path.join(out, "summary_full.tsv"), "w") as f:
        f.write("\n".join(table_lines(table, "dataset", full=True)) + "\n")
    with open(os.path.join(out, "summary_compact.tsv"), "w") as f:
        f.write("\n".join(table_lines(table, "dataset", full=False)) + "\n")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(table, f, ensure_ascii=False, indent=2, default=_json_default)


def regrade_dataset_dir(out, names=None, thresholds=None, full=False):
    """不重跑数据：用 <数据集>_samples.jsonl 里保存的逐样本指标和 <数据集>.json 里的汇总指标，按当前分档规则重新分档。
    原档位保存在 grade_prev / dataset_grade_prev。"""
    if not names:
        names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(out, "*.json"))
                       if os.path.basename(p) != "summary.json")
    for name in names:
        p = os.path.join(out, f"{name}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            row = json.load(f)
        if row.get("status") != "OK":
            continue
        sp = os.path.join(out, f"{name}_samples.jsonl")
        samples = []
        if os.path.exists(sp):
            with open(sp) as f:
                samples = [json.loads(l) for l in f if l.strip()]
            for r in samples:
                r.setdefault("grade_prev", r["grade"])
                r["grade"], r["reasons"], _ = Q.grade(r["metrics"], thresholds)
            with open(sp, "w") as f:
                for r in samples:
                    f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
        row.setdefault("dataset_grade_prev", row.get("dataset_grade"))
        row["dataset_grade"], row["reasons"], _ = Q.grade(row["metrics"], thresholds)
        row["sample_grades"] = summarize_grades(samples)
        _dump_row(out, row)
    summarize_dataset_dir(out, names, full)


def summarize_dataset_dir(out, names=None, full=False):
    """并行跑完（每个数据集一个进程）后，把 <out>/<数据集>.json 合并成一张表。"""
    if not names:
        names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(out, "*.json"))
                       if os.path.basename(p) != "summary.json")
    table = []
    for name in names:
        p = os.path.join(out, f"{name}.json")
        if os.path.exists(p):
            with open(p) as f:
                table.append(json.load(f))
        else:
            table.append(dict(dataset=name, status="NO_RESULT"))
    write_dataset_summary(table, out, full)


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


FULL_COLS = [("fsv", "{:.2%}"), ("mvc", "{:.1%}"), ("vpr", "{:.1%}"), ("best_scale", "{:.3g}"), ("drift", "{:.3f}"),
             ("tri_angle_median", "{:.2f}"), ("baseline_over_depth", "{:.3f}"), ("bfr", "{:.1f}"), ("valid_ratio", "{:.1%}")]


def _grade_dist(r):
    sg = r.get("sample_grades", {})
    return "/".join(str(sg.get(g, {}).get("count", 0)) for g in GRADES) if sg else "-"


def scale_check(m, thresholds=None):
    """尺度检查（M4 合成一项）：s* 在 usable 区间内且 drift 不超过 usable 上限 → OK。"""
    th = dict(Q.DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    bs, dr = m.get("best_scale"), m.get("drift")
    if bs is None and (dr is None or not np.isfinite(dr)):
        return "-"
    bad = []
    if bs is not None and np.isfinite(bs):
        lo, hi = th["best_scale"][2]
        fsv = m.get("fsv")
        if not lo <= bs <= hi:
            if fsv is not None and np.isfinite(fsv) and fsv <= tuple(th["fsv"])[2]:
                return f"不可观测(s*={bs:.3g})"   # s=1 已自洽，s* 偏离只说明对尺度不敏感，见 Q.grade
            bad.append(f"s*={bs:.3g}")
    if dr is not None and np.isfinite(dr) and dr > th["drift"][2]:
        bad.append(f"drift={dr:.2f}")
    return "OK" if not bad else "FAIL(" + ",".join(bad) + ")"


def boundary_check(m):
    """边界检查（M6）：边界违例率是内部的几倍；> BFR_HANDLE 时需要边界处理。"""
    b = m.get("bfr")
    if b is None or not np.isfinite(b):
        return "-"
    return f"{b:.1f}x" + (" 需处理" if b > Q.BFR_HANDLE else "")


def table_lines(rows, key="dataset", full=False):
    if full:
        head = [key, "status", "grade"] + [c for c, _ in FULL_COLS] + ["A/B/C/U/rej"]
        lines = ["\t".join(head)]
        for r in rows:
            m = r.get("metrics", {}) or {}
            lines.append("\t".join([str(r.get(key, "")), str(r.get("status", "")),
                                    str(r.get("dataset_grade", r.get("grade", "")))]
                                   + [_fmt(m.get(c), f) for c, f in FULL_COLS] + [_grade_dist(r)]))
        return lines
    # 精简视图：3 个核心指标（违例率 / 可用率 / 视差）+ 2 项检查（尺度 / 边界）+ 档位
    head = [key, "status", "档位", "违例率", "可用率", "视差°", "尺度", "边界", "A/B/C/U/rej", "原因"]
    lines = ["\t".join(head)]
    for r in rows:
        m = r.get("metrics", {}) or {}
        ok = bool(m)
        lines.append("\t".join([str(r.get(key, "")), str(r.get("status", "")),
                                str(r.get("dataset_grade", r.get("grade", ""))),
                                _fmt(m.get("fsv"), "{:.2%}"), _fmt(m.get("vpr"), "{:.1%}"),
                                _fmt(m.get("tri_angle_median"), "{:.1f}"),
                                scale_check(m) if ok else "-", boundary_check(m) if ok else "-",
                                _grade_dist(r), "; ".join(r.get("reasons", []) or [])]))
    return lines


def print_table(rows, key="dataset", full=False):
    print("\n" + "\n".join(table_lines(rows, key, full)))


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
    common.add_argument("--full", action="store_true", help="终端打印完整指标表（默认只打印 3 个核心指标 + 2 项检查）")

    d = sub.add_parser("dataset", parents=[common], help="训练数据集（GT 体检）")
    d.add_argument("--config", default="configs/data/datamix_all_v9-5.yaml")
    d.add_argument("--datasets", nargs="*", default=None)
    d.add_argument("--n", type=int, default=30)
    d.add_argument("--res", type=int, nargs=2, default=[512, 336], metavar=("W", "H"))
    d.add_argument("--mode", default="train")
    d.add_argument("--raw-compare", action="store_true", help="额外评估 98 分位截断这条规则（M7）")
    d.add_argument("--no-summary", action="store_true", help="不写 summary.json（并行时每个进程只跑一个数据集，最后用 summarize 合并）")

    sm = sub.add_parser("summarize", help="合并 dataset 模式的 <out>/<数据集>.json 为 summary.json")
    sm.add_argument("--out", default="quality_out")
    sm.add_argument("--datasets", nargs="*", default=None, help="按此顺序列出；缺结果的标 NO_RESULT")
    sm.add_argument("--full", action="store_true", help="打印完整指标表")

    rg = sub.add_parser("regrade", help="不重跑数据，按当前分档规则重新给 dataset 模式的结果分档")
    rg.add_argument("--out", default="quality_out")
    rg.add_argument("--datasets", nargs="*", default=None)
    rg.add_argument("--thresholds", default=None)
    rg.add_argument("--full", action="store_true")

    p = sub.add_parser("pseudo", parents=[common], help="伪标签目录（每帧一个 npz）")
    p.add_argument("--root", required=True)
    p.add_argument("--window-stride", type=int, default=None, help="窗口起点间隔，默认 = frames（不重叠）")
    p.add_argument("--frame-step", type=int, default=1, help="先按此步长抽帧，再切窗口（用来增大视差）")
    p.add_argument("--z-far", type=float, default=0.0)
    p.add_argument("--conf-thr", type=float, default=None, help="评估“conf < 阈值则删除”这条规则（M7）")
    p.add_argument("--save-masks", action="store_true", help="保存每帧的 verified / viol / boundary 掩码")
    args = ap.parse_args()
    if args.cmd == "summarize":
        summarize_dataset_dir(args.out, args.datasets, args.full)
        return
    if args.cmd == "regrade":
        regrade_dataset_dir(args.out, args.datasets, load_thresholds(args.thresholds)[1], args.full)
        return

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
