# -*- coding: utf-8 -*-
"""
utils/quality_metrics.py 的单元测试：用解析渲染的房间场景（盒子房间 + 一个物体），
分别注入 射线距离当 z、深度尺度错误、位姿错配、边缘飞点、重复帧、无视差，确认对应指标报警。
只依赖 numpy。运行：python -m pytest tests/test_quality_metrics.py -q   或   python tests/test_quality_metrics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import quality_metrics as Q  # noqa: E402

W, H = 128, 96
FX = 0.5 * W / np.tan(np.deg2rad(30))
K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])
CFG = Q.QualityConfig(scale_grid=tuple(float(10.0 ** e) for e in np.arange(-2, 2.01, 0.25)) + (1.0,))


def look_at(c, target):
    z = target - c
    z /= np.linalg.norm(z)
    x = np.cross(np.array([0, -1.0, 0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = c
    return T


def render(T):
    d = Q.pixel_rays(H, W, K) @ T[:3, :3].T  # 世界坐标下的射线方向（相机 z 分量为 1）
    o = T[:3, 3]
    lo, hi = np.array([-4, -3, -4.0]), np.array([4, 3, 4.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / d, (hi - o) / d
        tmax = np.maximum(t1, t2)
        t_room = np.min(np.where(tmax > 0, tmax, np.inf), axis=-1)
        a, b = (np.array([-0.8, 0.0, 0.5]) - o) / d, (np.array([0.8, 3.0, 2.0]) - o) / d
    tn, tf = np.max(np.minimum(a, b), -1), np.min(np.maximum(a, b), -1)
    hit = (tf >= tn) & (tn > 0)
    return np.where(hit, np.minimum(tn, t_room), t_room)  # 射线 z 分量为 1 ⇒ t 即 z-depth


def scene(n=6, spread=1.2):
    xs = np.linspace(-spread, spread, n)
    cams = [look_at(np.array([x, 0.3 * np.sin(x), -2.5 + 0.2 * x]), np.array([0.3 * x, 0.5, 3.0])) for x in xs]
    return np.stack([render(T) for T in cams]), np.stack([K] * n), np.stack(cams)


D0, KS, CAMS = scene()


def run(depths, cams=CAMS, **kw):
    s = Q.prepare_sample(depths, KS, cams, z_far=100)
    return Q.compute_metrics(s, CFG, **kw)


def test_clean_scene_is_clean():
    m = run(D0, with_ray2z=True)["metrics"]
    assert m["fsv"] < 0.01, m["fsv"]
    assert m["best_scale"] == 1.0
    assert m["mvc_ray2z"] < m["mvc"]          # 本来就是 z-depth：按射线换算只会变差
    assert m["tri_angle_median"] > 3.0
    assert m["vpr"] > 0.5
    assert m["pose_det_min"] > 0.99
    g, reasons, _ = Q.grade(m)
    assert g in ("A", "B"), (g, reasons)


def test_ray_depth_detected():
    norm = np.linalg.norm(Q.pixel_rays(H, W, K), axis=-1)
    m = run(D0 * norm[None], with_ray2z=True)["metrics"]
    assert m["mvc_ray2z"] - m["mvc"] > 0.05, (m["mvc"], m["mvc_ray2z"])


def test_scale_error_detected():
    m = run(D0 * 0.1)["metrics"]
    assert abs(np.log10(m["best_scale"]) - 1.0) < 0.3, m["best_scale"]   # 应找回 ×10


def test_pose_mismatch_detected():
    cams = np.concatenate([CAMS[1:], CAMS[:1]])
    m = run(D0, cams=cams)["metrics"]
    assert m["fsv"] > 0.10, m["fsv"]
    assert Q.grade(m)[0] in ("C", "reject")


def test_flying_pixels_raise_bfr_and_are_flagged():
    D = D0.copy()
    for k in range(D.shape[0]):
        e = Q.legacy_edge_mask(D[k], D[k] > 0, 0.03)
        D[k][e] = 0.5 * D[k][e] + 0.5 * 0.7 * np.roll(D[k], 1, 1)[e]
    r = run(D)
    clean = run(D0)["metrics"]
    assert r["metrics"]["fsv"] > clean["fsv"]
    assert r["masks"]["viol"].sum() > 0
    assert clean["bfr"] <= Q.BFR_GOOD, clean["bfr"]          # 真实深度跳变处的重采样误差不应触发
    assert r["metrics"]["bfr"] > Q.BFR_HANDLE, r["metrics"]["bfr"]


def test_duplicate_frames_and_no_parallax():
    cams = np.stack([CAMS[0]] * 3 + [CAMS[1]] * 3)
    D = np.stack([D0[0]] * 3 + [D0[1]] * 3)
    m = run(D, cams=cams)["metrics"]
    assert m["n_zero_baseline_pairs"] == 6
    # 全部是同一帧的重复：处处"一致"，但三角化角为 0，不能算作多视角验证
    same = run(np.stack([D0[0]] * 6), cams=np.stack([CAMS[0]] * 6))["metrics"]
    assert same["mvc"] > 0.99 and same["vpr"] == 0.0, (same["mvc"], same["vpr"])


def test_static_camera_rotation_only():
    c = np.array([0.0, 0.0, -2.5])
    cams = np.stack([look_at(c, np.array([0.4 * x, 0.5, 3.0])) for x in np.linspace(-1, 1, 6)])
    D = np.stack([render(T) for T in cams])
    m = run(D, cams=cams)["metrics"]
    assert m["baseline_over_depth"] < 1e-6
    assert not np.isfinite(m["tri_angle_median"]) or m["tri_angle_median"] < 0.5
    g, reasons, bands = Q.grade(m)
    assert "best_scale" not in bands          # 视差不足时 s* 不参与判定


def test_filter_fidelity():
    D = D0.copy()
    bad = np.zeros(D.shape, bool)
    bad[:, 10:20, 10:40] = True
    D[bad] *= 0.8                             # 注入一块错误深度
    s = Q.prepare_sample(D, KS, CAMS)
    good_rule = Q.filter_fidelity(s, bad, CFG)
    rng = np.random.default_rng(0)
    rand = (rng.random(D.shape) < bad.mean()) & s.valids
    rand_rule = Q.filter_fidelity(s, rand, CFG)
    assert good_rule["lift"] > 3, good_rule
    assert rand_rule["lift"] < good_rule["lift"]


def test_gt_metrics_selectivity():
    L = D0 * 2.0                              # 尺度不同的标签
    L_bad = L.copy()
    bad = np.zeros(L.shape, bool)
    bad[:, 30:50, 30:60] = True
    L_bad[bad] *= 1.3
    g = Q.gt_metrics(L_bad, D0, kept=~bad)
    assert g["abs_rel_kept"] < 1e-6 and g["sel"] > 10, g
    assert 0 < g["gt_cov"] < 1


def test_teacher_agreement():
    t = Q.teacher_agreement([D0, D0 * 1.01, D0 * 0.99, D0 * 1.5])
    assert t["ta"] > 0.99


def test_aggregate_counts_matches_single():
    r = run(D0)
    a = Q.aggregate_counts([r])
    assert abs(a["fsv"] - r["metrics"]["fsv"]) < 1e-12
    assert abs(a["vpr"] - r["metrics"]["vpr"]) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("ok ", f.__name__)
    print(f"{len(fns)} passed")
