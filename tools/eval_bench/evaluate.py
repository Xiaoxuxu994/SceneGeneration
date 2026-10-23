"""Score every method under a benchmark directory (built by build_scannetpp_val8.py, predictions by infer.py).

  python tools/eval_bench/evaluate.py BENCH [METHOD ...] [--workers 16] [--tag _v1]
  -> BENCH/eval/per_clip{tag}.csv, summary{tag}.json, summary{tag}.md

Metrics per clip (8 views):
  relative pose over all view pairs, as in training validation (utils/eval_utils.py): RRA@30, RTA@30 and
    AUC@5 / @10 / @30 (angle of the relative rotation; angle between relative translation directions, sign-free)
  trajectory, as in the eval3d benchmark (code/bench/eval3d_bench.py, vipe_eval/eval_vipe.py): ATE after Sim3 / SE3
    alignment of camera centres (cm), Sim3 scale, RTE / RRE between consecutive views, FOV error
  depth (GT > 0, < MAX_DEPTH): AbsRel / delta1 / LogRMSE with per-sequence median scale (seq), per-frame median
    scale (frame) and no scaling (metric)
  point cloud (vipe_eval/geo_vipe.py): GT and predicted points on the same pixels fused over all views, prediction
    mapped to GT by a Sim3 fitted on the point pairs; Acc / Comp (median, m), Chamfer, F-score@5cm
Summary: mean over clips (median for ATE, RTE, point-cloud distances, as they are heavy-tailed).
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MAX_DEPTH = 80.0
PC_STRIDE = 4


# ------------------------------------------------------------- eval3d / vipe_eval (same definitions)
def umeyama(src, dst, with_scale):
    """Find s, R, t minimising ||dst - (s R src + t)||. src, dst: (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def pose_metrics(T_pred, T_gt):
    p, g = T_pred[:, :3, 3], T_gt[:, :3, 3]
    out = {}
    s, R, t = umeyama(p, g, True)
    out["ate_sim3_cm"] = 100 * np.sqrt(((s * p @ R.T + t - g) ** 2).sum(1).mean())
    out["sim3_scale"] = s
    _, R1, t1 = umeyama(p, g, False)
    out["ate_se3_cm"] = 100 * np.sqrt(((p @ R1.T + t1 - g) ** 2).sum(1).mean())
    rte, rre = [], []
    for k in range(len(p) - 1):
        dp = np.linalg.inv(T_pred[k]) @ T_pred[k + 1]
        dg = np.linalg.inv(T_gt[k]) @ T_gt[k + 1]
        rte.append(np.linalg.norm(s * dp[:3, 3] - dg[:3, 3]))
        rre.append(rot_angle_deg(dp[:3, :3].T @ dg[:3, :3]))
    out["rte_cm"] = 100 * float(np.mean(rte))
    out["rre_deg"] = float(np.mean(rre))
    out["traj_len_m"] = float(np.linalg.norm(np.diff(g, axis=0), axis=1).sum())
    return out


def depth_errors(pred, gt):
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "absrel": float(np.mean(np.abs(pred - gt) / gt)),
        "delta1": float(np.mean(ratio < 1.25)),
        "logrmse": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
    }


def unproject(depth, K, T, stride=1, mask=None):
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride]
    m = (z > 0) & np.isfinite(z)
    if mask is not None:
        m &= mask[::stride, ::stride]
    X = np.stack([(u[m] - K[0, 2]) / K[0, 0] * z[m], (v[m] - K[1, 2]) / K[1, 1] * z[m], z[m]], 1)
    return X @ T[:3, :3].T + T[:3, 3], m


def fused_clouds(T_gt, K_gt, GD, T_p, K_p, D_p, stride=PC_STRIDE):
    """Fused GT / predicted clouds over the same pixels, prediction mapped into the GT frame (Sim3 on point pairs)."""
    G, P = [], []
    for k in range(len(T_gt)):
        g, p = GD[k], D_p[k]
        common = (g > 0) & np.isfinite(p) & (p > 0)
        Xg, _ = unproject(g, K_gt[k], T_gt[k], stride, mask=common)
        Xp, _ = unproject(p, K_p[k], T_p[k], stride, mask=common)
        G.append(Xg)
        P.append(Xp)
    G, P = np.concatenate(G), np.concatenate(P)
    s, R, t = umeyama(P, G, True)
    return G, s * P @ R.T + t, s


def cloud_metrics(G, P):
    dp = cKDTree(G).query(P, workers=1)[0]
    dg = cKDTree(P).query(G, workers=1)[0]
    a, c = (dp < 0.05).mean(), (dg < 0.05).mean()
    return {"pc_acc_m": float(np.median(dp)), "pc_comp_m": float(np.median(dg)),
            "pc_chamfer_m": float((dp.mean() + dg.mean()) / 2), "pc_acc_5cm": float(a), "pc_comp_5cm": float(c),
            "pc_fscore_5cm": float(2 * a * c / max(a + c, 1e-9))}


def to_grid(depth, K, H, W):
    """Resize predicted depth (nearest) to the GT grid and rescale K (pixel centres at integer coordinates)."""
    n, h, w = depth.shape
    sx, sy = W / w, H / h
    D = np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) for d in depth])
    K = K.copy()
    K[:, 0, :] *= sx
    K[:, 1, :] *= sy
    K[:, 0, 2] += 0.5 * sx - 0.5
    K[:, 1, 2] += 0.5 * sy - 0.5
    return D, K


# ------------------------------------------------------------- training-validation pose metrics (numpy port)
def relative_pose_errors(T_pred, T_gt):
    """utils/eval_utils.se3_to_relative_pose_error: all pairs i < j, rel = T_j @ inv(T_i) (c2w inputs);
    rotation error = angle of R_gt^T R_pred; translation error = sign-free angle between translation directions.
    As in Pi3Trainer.calculate_metrics, both trajectories are first expressed in their first camera's frame
    (rel is a world-frame quantity, so the arbitrary world frame of the prediction must be removed)."""
    T_pred = np.linalg.inv(T_pred[0]) @ T_pred
    T_gt = np.linalg.inv(T_gt[0]) @ T_gt
    n = len(T_gt)
    i1, i2 = np.triu_indices(n, 1)
    rel_g = T_gt[i2] @ np.linalg.inv(T_gt[i1])
    rel_p = T_pred[i2] @ np.linalg.inv(T_pred[i1])
    Rd = np.transpose(rel_g[:, :3, :3], (0, 2, 1)) @ rel_p[:, :3, :3]
    r_err = np.degrees(np.arccos(np.clip((np.trace(Rd, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    tg, tp = rel_g[:, :3, 3], rel_p[:, :3, 3]
    tg = tg / (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-15)
    tp = tp / (np.linalg.norm(tp, axis=1, keepdims=True) + 1e-15)
    t_err = np.degrees(np.arccos(np.clip(np.abs((tg * tp).sum(1)), 0, 1)))
    return r_err, t_err


def auc(r_err, t_err, thr):
    """utils/eval_utils.calculate_auc: histogram of max(r, t) with thr + 1 bins on [0, thr], mean of the CDF."""
    e = np.maximum(r_err, t_err)
    hist, _ = np.histogram(e, bins=thr + 1, range=(0, thr))
    return float(np.cumsum(hist / len(e)).mean())


# ------------------------------------------------------------- per clip
def eval_clip(job):
    B, method, c = job
    clip, n = c["clip"], c["n_frames"]
    r = {"method": method, "clip": clip, "scene": c.get("scene", ""), "n_frames": n}
    gt = np.load(B / "gt" / f"{clip}.npz")
    T_gt, K_gt = gt["c2w"], gt["K"]
    GD = np.stack([cv2.imread(str(B / "depth" / clip / f"{k:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                   for k in range(n)])
    H, W = GD.shape[1:]
    try:
        z = np.load(B / method / clip / "pred.npz")
        T_p, D, K_p = z["c2w"].astype(np.float64), z["depth"].astype(np.float32), z["K"].astype(np.float64)
        r["sec"] = float(z["sec"]) if "sec" in z.files else np.nan
        assert len(T_p) == n, f"{len(T_p)} poses vs {n} frames"
    except Exception as e:  # noqa: BLE001
        r["error"] = str(e)[:200]
        return r
    D, K_p = to_grid(D, K_p, H, W)

    r_err, t_err = relative_pose_errors(T_p, T_gt)
    r["rra30"] = float((r_err <= 30).mean())
    r["rta30"] = float((t_err <= 30).mean())
    for thr in (5, 10, 30):
        r[f"auc{thr}"] = auc(r_err, t_err, thr)
    r["rel_rot_err_deg"] = float(np.mean(r_err))
    r["rel_trans_err_deg"] = float(np.mean(t_err))
    r.update(pose_metrics(T_p, T_gt))
    fov = lambda fx: np.degrees(2 * np.arctan(W / 2 / fx))  # noqa: E731
    r["fov_err_deg"] = float(np.abs(fov(K_p[:, 0, 0]) - fov(K_gt[:, 0, 0])).mean())

    P, G = [], []
    for k in range(n):
        g, p = GD[k], D[k]
        m = (g > 0) & (g < MAX_DEPTH) & np.isfinite(p) & (p > 0)
        if m.sum() < 50:
            P.append(None); G.append(None)
            continue
        P.append(p[m]); G.append(g[m])
    valid = [i for i in range(n) if P[i] is not None]
    if valid:
        acc = {f"{t}_{k}": [] for t in ("metric", "seq", "frame") for k in ("absrel", "delta1", "logrmse")}
        s_seq = np.median(np.concatenate([G[i] for i in valid])) / np.median(np.concatenate([P[i] for i in valid]))
        for i in valid:
            for t, sc in (("metric", 1.0), ("seq", s_seq), ("frame", np.median(G[i]) / np.median(P[i]))):
                for kk, v in depth_errors(P[i] * sc, G[i]).items():
                    acc[f"{t}_{kk}"].append(v)
        r.update({f"depth_{k}": float(np.mean(v)) for k, v in acc.items()})
        r["depth_seq_scale"] = float(s_seq)
    try:
        Gc, Pc, s_pair = fused_clouds(T_gt, K_gt, np.where(GD < MAX_DEPTH, GD, 0), T_p, K_p, D)
        r.update(cloud_metrics(Gc, Pc))
        r["pc_pair_scale"] = float(s_pair)
    except Exception as e:  # noqa: BLE001
        r["geo_error"] = str(e)[:200]
    return r


MEAN_KEYS = ["auc5", "auc10", "auc30", "rra30", "rta30", "rel_rot_err_deg", "rel_trans_err_deg", "rre_deg",
             "fov_err_deg", "depth_seq_absrel", "depth_seq_delta1", "depth_frame_absrel", "depth_frame_delta1",
             "depth_seq_logrmse", "depth_metric_absrel", "pc_fscore_5cm", "pc_acc_5cm", "pc_comp_5cm", "sec"]
MEDIAN_KEYS = ["ate_sim3_cm", "rte_cm", "pc_acc_m", "pc_comp_m", "pc_chamfer_m", "sim3_scale"]
TABLE = [("AUC@5", "auc5", "mean", 100), ("AUC@10", "auc10", "mean", 100), ("AUC@30", "auc30", "mean", 100),
         ("RRA@30", "rra30", "mean", 100), ("RTA@30", "rta30", "mean", 100), ("ATE Sim3 cm", "ate_sim3_cm", "median", 1),
         ("RRE deg", "rre_deg", "mean", 1), ("FOV err deg", "fov_err_deg", "mean", 1),
         ("AbsRel (seq)", "depth_seq_absrel", "mean", 1), ("δ1 (seq)", "depth_seq_delta1", "mean", 100),
         ("F@5cm", "pc_fscore_5cm", "mean", 100), ("Acc m", "pc_acc_m", "median", 1), ("Comp m", "pc_comp_m", "median", 1),
         ("sec/clip", "sec", "mean", 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("methods", nargs="*")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    B = Path(args.bench)
    clips = json.loads((B / "clips.json").read_text())
    methods = args.methods or sorted(p.name for p in B.iterdir()
                                     if p.is_dir() and p.name not in ("frames", "depth", "gt", "eval") and (p / "info.json").exists())
    jobs = [(B, m, c) for m in methods for c in clips]
    print(f"{len(methods)} methods x {len(clips)} clips: {methods}", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(eval_clip, jobs, chunksize=8))
    df = pd.DataFrame(rows)
    out = B / "eval"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"per_clip{args.tag}.csv", index=False, float_format="%.5f")

    ok = df[df["error"].isna()] if "error" in df else df
    summ = {"n_clips": len(clips), "methods": {}, "errors": []}
    if "error" in df:
        summ["errors"] = df.loc[df["error"].notna(), ["method", "clip", "error"]].head(50).to_dict("records")
    for m in methods:
        g = ok[ok["method"] == m]
        info = json.loads((B / m / "info.json").read_text()) if (B / m / "info.json").exists() else {}
        summ["methods"][m] = {
            "n_ok": int(len(g)), "info": info,
            "mean": {k: round(float(g[k].mean()), 5) for k in MEAN_KEYS if k in g and g[k].notna().any()},
            "median": {k: round(float(g[k].median()), 5) for k in MEDIAN_KEYS if k in g and g[k].notna().any()}}
    (out / f"summary{args.tag}.json").write_text(json.dumps(summ, indent=1, default=str))

    lines = [f"# {B.name}: {len(clips)} clips", "",
             "| method | params M | K | n | " + " | ".join(h for h, *_ in TABLE) + " |",
             "|---|---|---|---|" + "---|" * len(TABLE)]
    for m, s in summ["methods"].items():
        vals = []
        for _, k, agg, mul in TABLE:
            v = s[agg].get(k)
            vals.append("–" if v is None else f"{v * mul:.{1 if mul == 100 else 3}f}")
        lines.append(f"| {m} | {s['info'].get('params_M', '')} | {s['info'].get('k') or ''} | {s['n_ok']} | " + " | ".join(vals) + " |")
    lines += ["", "AUC / RRA / RTA / δ1 / F-score in %; ATE, Acc, Comp are medians over clips, the rest means."]
    (out / f"summary{args.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if summ["errors"]:
        print(f"[WARN] {int(df['error'].notna().sum())} failed clip evaluations, see summary{args.tag}.json", file=sys.stderr)


if __name__ == "__main__":
    main()
