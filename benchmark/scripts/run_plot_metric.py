#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

## output defaulst
OUTPUT_ROOT_DEFAULTS = {
    "7scenes": "7scenes/test_stride40_max8",
    "dtu":     "dtu/test_stride1_max8",
    "nrgbd":   "nrgbd/test_stride100_max8",
    "scannet": "scannet/test_stride40_max8",
}

# =========================================================
# 1) 手动配置：横轴顺序 + legend + 文件匹配
# =========================================================
# 你想要的横轴顺序（全局一致）
X_ORDER = ["vggt", "pi3", "reliev3r_5k", "reliev3r_15k", "reliev3r_20k", "reliev3r_30k"]
X_TICKLABELS = ["vggt", "pi3", "5k", "15k", "20k", "30k"]

# 每个“点/线”对应哪个 xlsx（用 glob，按你的目录结构很好匹配）
MODEL_FILES = {
    "vggt": "metrics_vggt*.xlsx",
    "pi3": "metrics_pi3*.xlsx",
    "reliev3r_5k": "metrics_reliev3r*ckpt5k*.xlsx",
    "reliev3r_15k": "metrics_reliev3r*ckpt15k*.xlsx",
    "reliev3r_20k": "metrics_reliev3r*ckpt20k*.xlsx",
    "reliev3r_30k": "metrics_reliev3r*ckpt30k*.xlsx",
}

# legend 显示成三类：Pi3、Reliev3R、VGGT
LEGEND_NAME = { "vggt": "VGGT", "pi3": "Pi3", "reliev3r": "Reliev3R"}

# 颜色（你明确说“一个颜色”）
COLOR = {"vggt": "#2ca02c", "pi3": "#d62728", "reliev3r": "#1f77b4"}


# =========================================================
# 2) 解析你这种“合并单元格层级 summary”格式
#    输出：Series(index=metric_key, value=float)
# =========================================================
STAT_TOKENS = {"mean", "med", "median", "avg", "average", "mean.", "med."}


def pick_summary_sheet(sheet_names: List[str]) -> str:
    for n in sheet_names:
        if "summary" in str(n).lower():
            return n
    return sheet_names[0]


def _is_number(x) -> bool:
    if x is None:
        return False
    try:
        if isinstance(x, str) and x.strip() == "":
            return False
        float(x)
        return True
    except Exception:
        return False


def detect_value_col(df: pd.DataFrame) -> int:
    for j in range(df.shape[1] - 1, -1, -1):
        if df.iloc[:, j].map(_is_number).any():
            return j
    raise ValueError("Cannot find numeric value column in summary sheet.")


def detect_stat_col(text_df: pd.DataFrame) -> Optional[int]:
    best_j, best_score = None, 0
    for j in range(text_df.shape[1]):
        col = text_df.iloc[:, j].dropna()
        if col.empty:
            continue
        score = sum(str(v).strip().lower() in STAT_TOKENS for v in col.astype(str))
        if score > best_score:
            best_score, best_j = score, j
    return best_j if best_score > 0 else None


def read_scannet_style_summary(xlsx_path: Path) -> pd.Series:
    xls = pd.read_excel(xlsx_path, sheet_name=None, header=None, engine="openpyxl")
    sheet = pick_summary_sheet(list(xls.keys()))
    raw = xls[sheet].copy()

    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if raw.empty:
        raise ValueError(f"Empty summary sheet: {xlsx_path}")

    vcol = detect_value_col(raw)
    text_df = raw.iloc[:, :vcol].copy()
    val = raw.iloc[:, vcol].copy()

    stat_col = detect_stat_col(text_df)

    # 合并单元格：dataset/group/metric ffill；stat 列不要 ffill（避免 pose 被错误继承 Mean/Med）
    for j in range(text_df.shape[1]):
        if stat_col is not None and j == stat_col:
            continue
        text_df.iloc[:, j] = text_df.iloc[:, j].ffill()

    keys, values = [], []
    for i in range(raw.shape[0]):
        if not _is_number(val.iloc[i]):
            continue
        v = float(val.iloc[i])

        parts = []
        for j in range(text_df.shape[1]):
            x = text_df.iat[i, j]
            if x is None or (isinstance(x, float) and pd.isna(x)):
                parts.append("")
            else:
                parts.append(str(x).strip())

        dataset = parts[0] if len(parts) >= 1 else ""
        group = parts[1] if len(parts) >= 2 else ""
        metric = parts[2] if len(parts) >= 3 else ""
        stat = parts[stat_col] if (stat_col is not None and stat_col < len(parts)) else ""

        key = "/".join([k for k in [dataset, group, metric] if k])
        if stat:
            key = f"{key} ({stat})"

        keys.append(key)
        values.append(v)

    s = pd.Series(values, index=keys, dtype=float)
    s = s[~s.index.duplicated(keep="last")]
    return s


# =========================================================
# 3) 画图：Pi3/VGGT点；Reliev3R按 ckpt 连线
# =========================================================
def find_one_file(root: Path, pattern: str) -> Path:
    hits = sorted(root.glob(pattern))
    if len(hits) == 0:
        raise FileNotFoundError(f"Not found: {root}/{pattern}")
    if len(hits) > 1:
        # 允许多匹配时取第一个，但你也可以改成 raise
        print(f"[WARN] multiple matched for {pattern}, take: {hits[0].name}")
    return hits[0]


def plot_manual_axis(
    root_dir: str,
    out_path: str = "summary_manual_axis.png",
    ncols: int = 3,
    figsize_per_subplot: Tuple[float, float] = (6.0, 3.2),
):
    root = Path(root_dir)

    # 读取配置中指定的 xlsx
    series_by_key: Dict[str, pd.Series] = {}
    for k in X_ORDER:
        p = find_one_file(root, MODEL_FILES[k])
        series_by_key[k] = read_scannet_style_summary(p)

    # 对齐成 DataFrame：rows = X_ORDER，cols = all metrics
    df = pd.DataFrame({k: s for k, s in series_by_key.items()}).T
    df = df.reindex(X_ORDER)

    metrics = list(df.columns)
    n_metrics = len(metrics)

    nrows = int(math.ceil(n_metrics / ncols))
    figsize = (figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    x = list(range(len(X_ORDER)))

    # legend handle 收集（只收集一次）
    legend_handles = {}
    legend_labels = {}

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # --- Pi3: 单点 ---
        if "pi3" in df.index:
            y_pi3 = df.loc["pi3", metric]
            ax.scatter([x[X_ORDER.index("pi3")]], [y_pi3], s=35, color=COLOR["pi3"])
            if "pi3" not in legend_handles:
                legend_handles["pi3"] = ax.scatter([], [], s=35, color=COLOR["pi3"])
                legend_labels["pi3"] = LEGEND_NAME["pi3"]

        # --- VGGT: 单点 ---
        if "vggt" in df.index:
            y_vggt = df.loc["vggt", metric]
            ax.scatter([x[X_ORDER.index("vggt")]], [y_vggt], s=35, color=COLOR["vggt"])
            if "vggt" not in legend_handles:
                legend_handles["vggt"] = ax.scatter([], [], s=35, color=COLOR["vggt"])
                legend_labels["vggt"] = LEGEND_NAME["vggt"]

        # --- Reliev3R: 只连 reliev3r_* 这些点 ---
        rel_keys = [k for k in X_ORDER if k.startswith("reliev3r_")]
        rel_x = [x[X_ORDER.index(k)] for k in rel_keys]
        rel_y = [df.loc[k, metric] for k in rel_keys]
        ax.plot(rel_x, rel_y, marker="o", linewidth=1.4, color=COLOR["reliev3r"])
        if "reliev3r" not in legend_handles:
            (h,) = ax.plot([], [], marker="o", linewidth=1.4, color=COLOR["reliev3r"])
            legend_handles["reliev3r"] = h
            legend_labels["reliev3r"] = LEGEND_NAME["reliev3r"]

        ax.set_title(str(metric))
        ax.set_xticks(x)
        ax.set_xticklabels(X_TICKLABELS, rotation=45, ha="right")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # hide unused
    for j in range(n_metrics, len(axes)):
        axes[j].axis("off")

    # 全局 legend（统一）
    fig.legend(
        [legend_handles["pi3"], legend_handles["reliev3r"], legend_handles["vggt"]],
        [legend_labels["pi3"], legend_labels["reliev3r"], legend_labels["vggt"]],
        loc="lower right",
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench_root",
        type=str,
        default="/mnt/nvme3/z00533370/3drecon/benchmark",
        help="benchmark root dir (contains 7scenes/, dtu/, ...)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["7scenes","nrgbd"],#,"nrgbd","dtu","scannet"],
        choices=list(OUTPUT_ROOT_DEFAULTS.keys()),
        help="dataset keys list (e.g. --dataset 7scenes scannet nrgbd)",
    )
    parser.add_argument("--out", type=str, default="outputs/", help="optional output png path")
    parser.add_argument("--ncols", type=int, default=3)
    args = parser.parse_args()

    bench_root = Path(args.bench_root)

    for ds in args.dataset:
        folder = bench_root / OUTPUT_ROOT_DEFAULTS[ds]          # 含 metrics_*.xlsx 的目录
        if args.out is None:
            out_path = folder / f"{ds}_summary_metrics.png"
        else:
            out_path = Path(args.out) / f"{ds}_summary_metrics.png"
        out = plot_manual_axis(str(folder), out_path=str(out_path), ncols=args.ncols)
        print(f"[OK] saved: {out.resolve()}")