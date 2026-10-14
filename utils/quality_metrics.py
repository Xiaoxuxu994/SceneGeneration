# -*- coding: utf-8 -*-
"""
3D 标签质量指标（GT 体检与伪标签质检共用）。只依赖 numpy，不依赖任何模型。

定义见报告"质量指标定义"一页，这里是它的参考实现：
  M1 FSV  自由空间违例率（主指标）      M2 MVC  多视角一致率
  M3 VPR  可验证像素率                  M4 SC   尺度一致性（s* 与 drift）
  M5 TRI  三角化条件                    M6 BFR  边界飞点比（法向判据，分母下限 1%）
  M7 FFL  过滤保真度（filter_fidelity）
  G1      留存精度与选择性（gt_metrics，需要真值）
  G2      教师一致率（teacher_agreement，依赖多个教师）

核心做法：对一个样本（F 帧）的所有有序帧对 (i, j)，把 i 帧的有效像素投到 j 帧，
按 r = (z_proj - D_j) / D_j 分成 不可见 / 一致 / 遮挡 / 违例，并逐像素累计
可见次数 obs(p)、一致次数 c(p)、违例次数 v(p)、三角化角达标的一致次数 c'(p)。
所有比例类指标都由这些逐像素计数求和得到（微平均），所以任意像素子集
（边界、被某条规则删掉的像素……）的 FSV / MVC 都能直接算。

用法：
    from utils.quality_metrics import QualityConfig, prepare_sample, compute_metrics
    s = prepare_sample(depths, Ks, c2ws, valids=None, z_far=80)
    r = compute_metrics(s, QualityConfig())
    r["metrics"]["fsv"], r["masks"]["verified"]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "QualityConfig", "Sample", "prepare_sample", "pixel_rays", "ray_to_z",
    "reproject_counts", "compute_metrics", "aggregate_counts", "metrics_from_counts",
    "subset_stats", "filter_fidelity", "boundary_mask", "legacy_edge_mask",
    "best_scale", "gt_metrics", "teacher_agreement", "grade",
    "DEFAULT_THRESHOLDS",
]


# ---------------------------------------------------------------------------
# config & sample
# ---------------------------------------------------------------------------
@dataclass
class QualityConfig:
    tau: float = 0.03                 # 相对深度误差阈值
    k_verify: int = 2                 # M3：至少在 k 个其他帧中一致
    min_tri_deg: float = 1.0          # M3：只有三角化角 >= 该值的帧对才计入 c'(p)
    max_pairs: Optional[int] = None   # 帧对上限（None = 全部有序对）
    stride: int = 1                   # 源像素下采样步长（只影响统计，不影响掩码分辨率）
    drift_rel: float = 0.3            # M4：估计帧对尺度时使用 |r| < drift_rel 的像素
    boundary_cos: float = math.cos(math.radians(85.0))  # M6：|n·d| < cos(85°) 视为掠射
    boundary_jump: float = 0.10       # M6：3x3 邻域相对深度跳变阈值
    bfr_floor: float = 0.01           # M6：BFR 分母的下限，避免内部违例接近 0 时比值失真
    tri_samples_per_pair: int = 4000  # M5：每个帧对最多采样多少个一致点算角度
    scale_grid: Tuple[float, ...] = tuple(float(10.0 ** e) for e in np.arange(-3, 3.01, 0.25)) + (1.0,)
    scale_stride: int = 2             # M4 s* 搜索时的下采样步长
    scale_max_pairs: int = 12         # M4 s* 搜索时的帧对上限
    seed: int = 0


@dataclass
class Sample:
    depths: np.ndarray                # [F, H, W] z-depth
    Ks: np.ndarray                    # [F, 3, 3]
    c2ws: np.ndarray                  # [F, 4, 4]
    valids: np.ndarray                # [F, H, W] bool
    dyn: Optional[np.ndarray] = None  # [F, H, W] bool，动态区域（不参与几何检查）
    meta: Dict = field(default_factory=dict)

    @property
    def F(self) -> int:
        return int(self.depths.shape[0])

    @property
    def HW(self) -> Tuple[int, int]:
        return int(self.depths.shape[1]), int(self.depths.shape[2])


def prepare_sample(depths, Ks, c2ws, valids=None, z_far: float = 0.0, dyn=None, extra_masks=None, meta=None) -> Sample:
    """统一成 float64 数组，并按 base_dataset 的口径生成有效掩码：depth>0 & finite & depth<z_far [& extra & ~dyn]。"""
    D = np.nan_to_num(np.asarray(depths, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if D.ndim == 2:
        D = D[None]
    F = D.shape[0]
    K = np.asarray(Ks, dtype=np.float64).reshape(F, 3, 3)
    T = np.asarray(c2ws, dtype=np.float64).reshape(F, 4, 4)
    V = (D > 0) & np.isfinite(D)
    if z_far and z_far > 0:
        V &= D < z_far
    if valids is not None:
        V &= np.asarray(valids, dtype=bool).reshape(D.shape)
    if extra_masks is not None:
        for m in extra_masks:
            if m is not None:
                V &= np.asarray(m, dtype=bool).reshape(D.shape)
    dyn_arr = None
    if dyn is not None:
        dyn_arr = np.asarray(dyn, dtype=bool).reshape(D.shape)
        V &= ~dyn_arr
    D = np.where(V, D, 0.0)
    return Sample(depths=D, Ks=K, c2ws=T, valids=V, dyn=dyn_arr, meta=dict(meta or {}))


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def pixel_rays(H: int, W: int, K: np.ndarray) -> np.ndarray:
    """K^-1 [u, v, 1]（z 分量为 1），[H, W, 3]。像素坐标与 pi3.utils.geometry 一致（整数网格，不加 0.5）。"""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    return np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)


def ray_to_z(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """把"沿射线的距离"换算成 z-depth：z = d / ||K^-1 [u v 1]||。"""
    return depth / np.linalg.norm(pixel_rays(*depth.shape, K), axis=-1)


def _pairs(F: int, max_pairs: Optional[int]) -> List[Tuple[int, int]]:
    pairs = [(i, j) for i in range(F) for j in range(F) if i != j]
    if max_pairs is not None and len(pairs) > max_pairs:
        sel = np.linspace(0, len(pairs) - 1, max_pairs).round().astype(int)
        pairs = [pairs[k] for k in sel]
    return pairs


def legacy_edge_mask(depth: np.ndarray, valid: np.ndarray, rtol: float = 0.03) -> np.ndarray:
    """旧的边界定义（与 pi3.utils.geometry.depth_edge 同口径）：3x3 邻域 (max-min)/depth > rtol。会把斜面误判为边界，仅供对照。"""
    d = np.where(valid, depth, np.nan)
    H, W = depth.shape
    p = np.pad(d, 1, mode="edge")
    stack = np.stack([p[a:a + H, b:b + W] for a in range(3) for b in range(3)], axis=0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rng = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
        return (rng / np.maximum(depth, 1e-9) > rtol) & valid


def boundary_mask(depth: np.ndarray, K: np.ndarray, valid: np.ndarray, cfg: QualityConfig = QualityConfig()) -> np.ndarray:
    """
    M6 的边界集合（法向判据）：表面几乎平行于视线 |n·d| < cos(85°)，且 3x3 邻域内相对深度跳变 > 10%。
    只用有效邻居；与无效像素相邻不算跳变（避免把物体与背景的分界全标成边界）。
    """
    H, W = depth.shape
    P = pixel_rays(H, W, K) * depth[..., None]
    Pp = np.pad(P, ((1, 1), (1, 1), (0, 0)), mode="edge")
    Vp = np.pad(valid, 1, mode="constant", constant_values=False)
    dx = Pp[1:-1, 2:] - Pp[1:-1, :-2]
    dy = Pp[2:, 1:-1] - Pp[:-2, 1:-1]
    ok = valid & Vp[1:-1, 2:] & Vp[1:-1, :-2] & Vp[2:, 1:-1] & Vp[:-2, 1:-1]
    n = np.cross(dx, dy)
    with np.errstate(all="ignore"):
        n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        d = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-12)
        grazing = np.abs((n * d).sum(-1)) < cfg.boundary_cos
    Dp = np.pad(np.where(valid, depth, np.nan), 1, mode="edge")
    stack = np.stack([Dp[a:a + H, b:b + W] for a in range(3) for b in range(3)], axis=0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            jump = np.nanmax(np.abs(stack - depth[None]), axis=0) / np.maximum(depth, 1e-9)
    return ok & grazing & (jump > cfg.boundary_jump)


# ---------------------------------------------------------------------------
# reprojection core
# ---------------------------------------------------------------------------
def _world_points(sample: Sample, i: int, scale: float, stride: int, rays_cache: dict):
    H, W = sample.HW
    key = i
    if key not in rays_cache:
        rays_cache[key] = pixel_rays(H, W, sample.Ks[i])
    m = sample.valids[i].copy()
    if stride > 1:
        keep = np.zeros_like(m)
        keep[::stride, ::stride] = True
        m &= keep
    vv, uu = np.nonzero(m)
    X = rays_cache[key][vv, uu] * (sample.depths[i][vv, uu] * scale)[:, None]
    Xw = X @ sample.c2ws[i][:3, :3].T + sample.c2ws[i][:3, 3]
    return vv, uu, Xw


def _project(sample: Sample, j: int, Xw: np.ndarray, w2c: np.ndarray, scale: float):
    H, W = sample.HW
    Xj = Xw @ w2c[:3, :3].T + w2c[:3, 3]
    z = Xj[:, 2]
    front = z > 1e-9
    zz = np.where(front, z, 1.0)
    K = sample.Ks[j]
    u = np.round(K[0, 0] * Xj[:, 0] / zz + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * Xj[:, 1] / zz + K[1, 2]).astype(np.int64)
    ok = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.flatnonzero(ok)
    vis = sample.valids[j][v[idx], u[idx]]
    idx = idx[vis]
    dj = sample.depths[j][v[idx], u[idx]] * scale
    rel = (z[idx] - dj) / dj
    return idx, rel, dj, z[idx]


def reproject_counts(sample: Sample, cfg: QualityConfig = QualityConfig(), scale: float = 1.0,
                     collect_extras: bool = True) -> Dict:
    """
    返回逐像素计数图（每帧 [H, W] int32）：obs、cons、viol、cons_tri；以及（collect_extras）
    每个帧对的尺度 log s_ij、三角化角样本、帧对基线。
    """
    F = sample.F
    H, W = sample.HW
    obs = np.zeros((F, H, W), np.int32)
    cons = np.zeros((F, H, W), np.int32)
    viol = np.zeros((F, H, W), np.int32)
    cons_tri = np.zeros((F, H, W), np.int32)
    w2cs = [np.linalg.inv(T) for T in sample.c2ws]
    centers = sample.c2ws[:, :3, 3]
    rays_cache: dict = {}
    rng = np.random.default_rng(cfg.seed)
    log_s, pair_ij, tri_samples = [], [], []
    min_cos = math.cos(math.radians(cfg.min_tri_deg))

    pts_cache = {}
    with np.errstate(all="ignore"):
        for i, j in _pairs(F, cfg.max_pairs):
            if i not in pts_cache:
                pts_cache[i] = _world_points(sample, i, scale, cfg.stride, rays_cache)
            vv, uu, Xw = pts_cache[i]
            if Xw.shape[0] == 0:
                continue
            idx, rel, dj, zj = _project(sample, j, Xw, w2cs[j], scale)
            if idx.size == 0:
                continue
            pv, pu = vv[idx], uu[idx]
            c = np.abs(rel) < cfg.tau
            vi = rel <= -cfg.tau
            obs[i, pv, pu] += 1
            cons[i, pv, pu] += c
            viol[i, pv, pu] += vi
            # 三角化角：射线 X - C_i 与 X - C_j 的夹角（只对一致点）
            a = Xw[idx] - centers[i]
            b = Xw[idx] - centers[j]
            cosang = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
            tri_ok = c & (cosang <= min_cos)
            cons_tri[i, pv, pu] += tri_ok
            if collect_extras:
                near = np.abs(rel) < cfg.drift_rel
                if near.sum() >= 20:
                    log_s.append(float(np.log(np.median(dj[near] / zj[near]))))
                    pair_ij.append((i, j))
                ci = np.flatnonzero(c)
                if ci.size:
                    if ci.size > cfg.tri_samples_per_pair:
                        ci = rng.choice(ci, cfg.tri_samples_per_pair, replace=False)
                    tri_samples.append(np.degrees(np.arccos(np.clip(cosang[ci], -1, 1))))
    base = np.linalg.norm(centers[:, None] - centers[None], axis=-1)[np.triu_indices(F, 1)] if F > 1 else np.zeros(0)
    return dict(obs=obs, cons=cons, viol=viol, cons_tri=cons_tri,
                log_s=np.asarray(log_s), pair_ij=pair_ij,
                tri=np.concatenate(tri_samples) if tri_samples else np.zeros(0),
                baselines=base)


def _count_consistent(sample: Sample, cfg: QualityConfig, scale: float) -> int:
    sub = QualityConfig(**{**asdict(cfg), "stride": cfg.scale_stride, "max_pairs": cfg.scale_max_pairs})
    return int(reproject_counts(sample, sub, scale=scale, collect_extras=False)["cons"].sum())


def best_scale(sample: Sample, cfg: QualityConfig = QualityConfig()) -> float:
    """M4 s*：位姿不动，深度乘以 s，取一致点数最多的 s；并列时优先 1。"""
    best_s, best_c = 1.0, -1
    for s in cfg.scale_grid:
        c = _count_consistent(sample, cfg, s)
        if c > best_c or (c == best_c and abs(s - 1.0) < 1e-12):
            best_s, best_c = float(s), c
    return best_s


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _ratio(a, b) -> float:
    return float(a) / float(b) if b else float("nan")


def subset_stats(counts: Dict, subset: np.ndarray) -> Dict:
    """任意像素子集（[F, H, W] bool）上的 FSV / MVC。"""
    o = int(counts["obs"][subset].sum())
    return dict(n_obs=o, fsv=_ratio(counts["viol"][subset].sum(), o), mvc=_ratio(counts["cons"][subset].sum(), o),
                n_pix=int(subset.sum()))


def compute_metrics(sample: Sample, cfg: QualityConfig = QualityConfig(), with_scale_search: bool = True,
                    with_ray2z: bool = False, with_boundary: bool = True) -> Dict:
    """
    计算一个样本的 M1–M6（M7 用 filter_fidelity 单独算）。
    返回 {"metrics": {...}, "masks": {...}, "counts": {...可用于 aggregate_counts 的计数...}}。
    """
    F = sample.F
    H, W = sample.HW
    cnt = reproject_counts(sample, cfg)
    V = sample.valids
    viol_mask = V & (cnt["viol"] >= 1) & (cnt["viol"] >= cnt["cons"])
    verified = V & (cnt["cons_tri"] >= cfg.k_verify) & (cnt["viol"] < cnt["cons_tri"])

    # M4 drift 与逐帧尺度
    log_s = cnt["log_s"]
    drift = float(1.4826 * np.median(np.abs(log_s - np.median(log_s)))) if log_s.size else float("nan")
    per_frame = []
    for k in range(F):
        vals = [ls for ls, (i, j) in zip(log_s, cnt["pair_ij"]) if j == k]
        per_frame.append(float(np.exp(np.median(vals))) if vals else float("nan"))

    # M5
    tri = cnt["tri"]
    med_depth = float(np.median(sample.depths[V])) if V.any() else float("nan")
    base = cnt["baselines"]
    b_over_d = float(np.median(base) / med_depth) if base.size and med_depth > 0 else float("nan")

    # M6
    bmask = np.stack([boundary_mask(sample.depths[k], sample.Ks[k], V[k], cfg) for k in range(F)]) if with_boundary \
        else np.zeros_like(V)
    sb, si = subset_stats(cnt, bmask), subset_stats(cnt, V & ~bmask)

    fsv_pf = [_ratio(cnt["viol"][k].sum(), cnt["obs"][k].sum()) for k in range(F)]
    m = dict(
        F=F, H=H, W=W,
        valid_ratio=_ratio(V.sum(), F * H * W),
        fsv=_ratio(cnt["viol"].sum(), cnt["obs"].sum()),
        fsv_per_frame=fsv_pf,
        fsv_max_frame=float(np.nanmax(fsv_pf)) if any(np.isfinite(fsv_pf)) else float("nan"),
        mvc=_ratio(cnt["cons"].sum(), cnt["obs"].sum()),
        vpr=_ratio(verified.sum(), V.sum()),
        vcov=_ratio(verified.sum(), F * H * W),
        drift=drift,
        per_frame_scale=per_frame,
        tri_angle_median=float(np.median(tri)) if tri.size else float("nan"),
        tri_angle_ge2=float(np.mean(tri >= 2.0)) if tri.size else float("nan"),
        baseline_over_depth=b_over_d,
        n_zero_baseline_pairs=int((base < 1e-6 * max(med_depth, 1e-9)).sum()) if base.size else 0,
        n_pairs=int(base.size),
        median_depth=med_depth,
        bfr=float(sb["fsv"] / max(si["fsv"], cfg.bfr_floor)) if np.isfinite(sb["fsv"]) and np.isfinite(si["fsv"]) else float("nan"),
        boundary_frac=_ratio(bmask.sum(), V.sum()),
        fsv_boundary=sb["fsv"], fsv_interior=si["fsv"],
        pose_orth_err=float(max(np.abs(T[:3, :3] @ T[:3, :3].T - np.eye(3)).max() for T in sample.c2ws)),
        pose_det_min=float(min(np.linalg.det(T[:3, :3]) for T in sample.c2ws)),
    )
    if with_scale_search:
        m["best_scale"] = best_scale(sample, cfg)
    if with_ray2z:
        sz = prepare_sample(np.stack([ray_to_z(sample.depths[k], sample.Ks[k]) for k in range(F)]),
                            sample.Ks, sample.c2ws, valids=V)
        cz = reproject_counts(sz, cfg, collect_extras=False)
        m["mvc_ray2z"] = _ratio(cz["cons"].sum(), cz["obs"].sum())
    counts = dict(n_obs=int(cnt["obs"].sum()), n_cons=int(cnt["cons"].sum()), n_viol=int(cnt["viol"].sum()),
                  n_valid=int(V.sum()), n_verified=int(verified.sum()), n_pix=int(F * H * W),
                  b_obs=sb["n_obs"], b_viol=int(cnt["viol"][bmask].sum()), b_pix=int(bmask.sum()),
                  i_obs=si["n_obs"], i_viol=int(cnt["viol"][V & ~bmask].sum()),
                  log_s=log_s.tolist(), tri=tri.tolist()[:2000], baselines=base.tolist(), med_depth=med_depth)
    return dict(metrics=m, masks=dict(viol=viol_mask, verified=verified, boundary=bmask), counts=counts, raw=cnt)


def aggregate_counts(results: Sequence[Dict], bfr_floor: float = 0.01) -> Dict:
    """把多个样本（例如同一序列的多个窗口，或一个数据集的多个样本）按计数微平均合并成序列/数据集级指标。"""
    keys = ["n_obs", "n_cons", "n_viol", "n_valid", "n_verified", "n_pix", "b_obs", "b_viol", "b_pix", "i_obs", "i_viol"]
    tot = {k: sum(r["counts"][k] for r in results) for k in keys}
    log_s = np.asarray([x for r in results for x in r["counts"]["log_s"]])
    tri = np.asarray([x for r in results for x in r["counts"]["tri"]])
    bod = [r["metrics"]["baseline_over_depth"] for r in results if np.isfinite(r["metrics"]["baseline_over_depth"])]
    return metrics_from_counts(tot, log_s, tri, bod, results, bfr_floor)


def metrics_from_counts(tot: Dict, log_s: np.ndarray, tri: np.ndarray, bod: List[float], results: Sequence[Dict],
                        bfr_floor: float = 0.01) -> Dict:
    fsv_b, fsv_i = _ratio(tot["b_viol"], tot["b_obs"]), _ratio(tot["i_viol"], tot["i_obs"])
    out = dict(
        n_samples=len(results),
        valid_ratio=_ratio(tot["n_valid"], tot["n_pix"]),
        fsv=_ratio(tot["n_viol"], tot["n_obs"]),
        fsv_max_frame=float(np.nanmax([r["metrics"]["fsv_max_frame"] for r in results])) if results else float("nan"),
        mvc=_ratio(tot["n_cons"], tot["n_obs"]),
        vpr=_ratio(tot["n_verified"], tot["n_valid"]),
        vcov=_ratio(tot["n_verified"], tot["n_pix"]),
        drift=float(1.4826 * np.median(np.abs(log_s - np.median(log_s)))) if log_s.size else float("nan"),
        tri_angle_median=float(np.median(tri)) if tri.size else float("nan"),
        tri_angle_ge2=float(np.mean(tri >= 2.0)) if tri.size else float("nan"),
        baseline_over_depth=float(np.median(bod)) if bod else float("nan"),
        n_zero_baseline_pairs=int(sum(r["metrics"]["n_zero_baseline_pairs"] for r in results)),
        bfr=float(fsv_b / max(fsv_i, bfr_floor)) if np.isfinite(fsv_b) and np.isfinite(fsv_i) else float("nan"),
        fsv_boundary=fsv_b, fsv_interior=fsv_i,
        boundary_frac=_ratio(tot["b_pix"], tot["n_valid"]),
        pose_det_min=float(min(r["metrics"]["pose_det_min"] for r in results)) if results else float("nan"),
    )
    bs = [r["metrics"]["best_scale"] for r in results if "best_scale" in r["metrics"]]
    if bs:
        out["best_scale"] = float(np.median(bs))
        out["best_scale_frac_eq1"] = float(np.mean(np.isclose(bs, 1.0)))
    rz = [r["metrics"]["mvc_ray2z"] for r in results if "mvc_ray2z" in r["metrics"]]
    if rz:
        out["mvc_ray2z"] = float(np.nanmean(rz))
    return out


def filter_fidelity(sample_before: Sample, removed: np.ndarray, cfg: QualityConfig = QualityConfig(),
                    counts: Optional[Dict] = None) -> Dict:
    """
    M7：评估一条过滤规则。sample_before = 规则生效前的标签；removed = 规则删掉的像素 [F, H, W]。
    RR = |R| / |V|；Lift = FSV(R) / FSV(V\\R)；GoodLoss = MVC(R) / MVC(V\\R)。
    """
    cnt = counts if counts is not None else reproject_counts(sample_before, cfg, collect_extras=False)
    V = sample_before.valids
    R = np.asarray(removed, dtype=bool) & V
    sr, sk = subset_stats(cnt, R), subset_stats(cnt, V & ~R)
    return dict(rr=_ratio(R.sum(), V.sum()),
                lift=_ratio(sr["fsv"], sk["fsv"]) if np.isfinite(sr["fsv"]) and sk["fsv"] and sk["fsv"] > 0 else float("nan"),
                good_loss=_ratio(sr["mvc"], sk["mvc"]) if np.isfinite(sr["mvc"]) and sk["mvc"] else float("nan"),
                fsv_removed=sr["fsv"], fsv_kept=sk["fsv"], mvc_removed=sr["mvc"], mvc_kept=sk["mvc"],
                n_removed=int(R.sum()), n_removed_observed=sr["n_obs"])


# ---------------------------------------------------------------------------
# G1 / G2
# ---------------------------------------------------------------------------
def gt_metrics(label_depths: np.ndarray, gt_depths: np.ndarray, kept: np.ndarray, label_valid: Optional[np.ndarray] = None,
               align: str = "median") -> Dict:
    """
    G1：先在"保留且有真值"的像素上用中位数比值把标签整体对齐到真值（每个调用 = 一条序列一个尺度），再算：
    AbsRel_kept、δ1.25_kept、AbsRel_rej、Sel = AbsRel_rej / AbsRel_kept、GTCov = |保留∩有真值| / |有真值|。
    """
    L = np.asarray(label_depths, dtype=np.float64)
    G = np.asarray(gt_depths, dtype=np.float64)
    has_gt = np.isfinite(G) & (G > 0)
    lv = (L > 0) & np.isfinite(L) if label_valid is None else (np.asarray(label_valid, bool) & (L > 0))
    K = np.asarray(kept, dtype=bool) & lv & has_gt
    Rj = lv & has_gt & ~np.asarray(kept, dtype=bool)
    out = dict(gt_cov=_ratio(K.sum(), has_gt.sum()), n_gt=int(has_gt.sum()), n_kept_gt=int(K.sum()))
    if K.sum() < 10:
        return {**out, "abs_rel_kept": float("nan"), "delta1_kept": float("nan"), "abs_rel_rej": float("nan"), "sel": float("nan"), "scale": float("nan")}
    s = float(np.median(G[K] / L[K])) if align == "median" else 1.0
    def absrel(m):
        return float(np.mean(np.abs(s * L[m] - G[m]) / G[m])) if m.sum() else float("nan")
    ark, arr = absrel(K), absrel(Rj)
    d1 = float(np.mean(np.maximum(s * L[K] / G[K], G[K] / (s * L[K])) < 1.25))
    return {**out, "abs_rel_kept": ark, "delta1_kept": d1, "abs_rel_rej": arr,
            "sel": (arr / ark if ark > 0 else float("inf")) if np.isfinite(arr) else float("nan"), "scale": s}


def teacher_agreement(depth_list: Sequence[np.ndarray], valid: Optional[np.ndarray] = None, rel: float = 0.05,
                      min_agree: int = 2, ref: int = 0) -> Dict:
    """
    G2：多个教师（已对齐到同一尺度）的逐像素一致性。以第 ref 个为当前标签，统计至少 min_agree 个其他教师与之相差 < rel 的像素比例。
    注意：依赖模型；教师会一起犯错，不能替代 M1/M3。
    """
    Ds = [np.asarray(d, dtype=np.float64) for d in depth_list]
    L = Ds[ref]
    v = (L > 0) & np.isfinite(L) if valid is None else (np.asarray(valid, bool) & (L > 0))
    n = np.zeros(L.shape, np.int32)
    for k, D in enumerate(Ds):
        if k == ref:
            continue
        with np.errstate(all="ignore"):
            n += (np.abs(D - L) / np.maximum(L, 1e-9) < rel) & (D > 0)
    agree = v & (n >= min_agree)
    return dict(ta=_ratio(agree.sum(), v.sum()), agree_mask=agree)


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------
# 每项：(方向, 好, 可用, 剔除)。direction "lower" = 越小越好；"higher" = 越大越好；"band" = 落在区间内。
# 落在"可用"与"剔除"之间记为 marginal（C 档）。阈值为初始值，需在带真值子集上校准（tools/calibrate_thresholds.py）。
DEFAULT_THRESHOLDS = {
    "fsv":              ("lower", 0.01, 0.05, 0.15),
    "vpr":              ("higher", 0.60, 0.30, 0.30),
    "drift":            ("lower", 0.03, 0.10, 0.10),
    "tri_angle_median": ("higher", 3.0, 1.5, 1.5),
    "valid_ratio":      ("higher", 0.50, 0.05, 0.05),
    "best_scale":       ("band", (0.9, 1.1), (0.75, 1.34), (0.75, 1.34)),
}
# M6 BFR 不参与序列分档：> BFR_HANDLE 时对该数据集开启边界处理（只去掉边界上的违例像素）。
BFR_GOOD, BFR_HANDLE = 5.0, 10.0


def _band(value, rule) -> str:
    kind, good, usable, reject = rule
    if value is None or not np.isfinite(value):
        return "na"
    if kind == "lower":
        return "good" if value < good else "usable" if value <= usable else "reject" if value > reject else "marginal"
    if kind == "higher":
        return "good" if value >= good else "usable" if value >= usable else "reject" if value < reject else "marginal"
    lo_g, hi_g = good
    lo_u, hi_u = usable
    lo_r, hi_r = reject
    if lo_g <= value <= hi_g:
        return "good"
    if lo_u <= value <= hi_u:
        return "usable"
    return "reject" if (value < lo_r or value > hi_r) else "marginal"


def grade(metrics: Dict, thresholds: Optional[Dict] = None, skip: Sequence[str] = ()) -> Tuple[str, List[str], Dict[str, str]]:
    """
    序列分档：任一项 reject → "reject"；全部 good → "A"；全部不差于 usable → "B"；否则 "C"。
    视差不足（tri reject）时 best_scale 不可观测，自动跳过 best_scale。返回 (档位, 原因列表, 各项档位)；
    原因列表包含所有未达 good 的项，B 档也能看出是哪一项拖了后腿。
    """
    th = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    bands = {}
    for k, rule in th.items():
        if k in skip or k not in metrics:
            continue
        bands[k] = _band(metrics[k], tuple(rule) if not isinstance(rule, tuple) else rule)
    if bands.get("tri_angle_median") in ("reject", "marginal"):
        bands.pop("best_scale", None)
    reasons = [f"{k}={metrics[k]:.4g} ({b})" for k, b in bands.items() if b in ("reject", "marginal", "usable")]
    vals = [b for b in bands.values() if b != "na"]
    if "reject" in vals:
        g = "reject"
    elif vals and all(b == "good" for b in vals):
        g = "A"
    elif all(b in ("good", "usable") for b in vals):
        g = "B"
    else:
        g = "C"
    return g, reasons, bands
