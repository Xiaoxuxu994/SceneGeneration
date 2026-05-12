#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import re
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ENTER_RE = re.compile(
    r"""
    \[rank(?P<rank>\d+)\]\s+ENTER\s+
    epoch=(?P<epoch>\d+)\s+
    iter=(?P<iter>\d+)\s+
    global_step=(?P<global_step>\d+)\s+
    dataset=(?P<dataset>[^|]+?)\s*\|\s*
    batch_type=(?P<batch_type>[^|]+?)\s*\|\s*
    batch_len=(?P<batch_len>\d+)
    """,
    re.VERBOSE,
)

TIMING_RE = re.compile(
    r"""
    \[rank(?P<rank>\d+)\]\s+TIMING\s+
    epoch=(?P<epoch>\d+)\s+
    iter=(?P<iter>\d+)\s+
    global_step=(?P<global_step>\d+)\s+
    dataset=(?P<dataset>\S+)\s+
    iter_total=(?P<iter_total>[0-9.]+)s\s+
    data=(?P<data>[0-9.]+)s\s+
    move=(?P<move>[0-9.]+)s\s+
    forward=(?P<forward>[0-9.]+)s\s+
    loss=(?P<loss_time>[0-9.]+)s\s+
    backward=(?P<backward>[0-9.]+)s\s+
    gather_loss=(?P<gather_loss>[0-9.]+)s\s+
    grad_clip=(?P<grad_clip>[0-9.]+)s\s+
    optim=(?P<optim>[0-9.]+)s
    """,
    re.VERBOSE,
)

LOSS_RE = re.compile(
    r"""
    \[rank(?P<rank>\d+)\]\s+
    iter=(?P<iter>\d+)\s+
    loss_done\s+
    loss=(?P<loss>[0-9.]+)
    """,
    re.VERBOSE,
)

IMG_SHAPE_RE = re.compile(r"img_shape=\((?P<img_shape>[^)]+)\)")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_img_bs_from_shape(shape_str):
    if pd.isna(shape_str) or shape_str is None:
        return None
    try:
        vals = [int(x.strip()) for x in str(shape_str).strip("()").split(",")]
        if len(vals) >= 1:
            return vals[0]
    except Exception:
        pass
    return None


def build_node_label(row):
    bl = row.get("batch_len", None)
    ib = row.get("img_bs", None)

    bl_txt = "NA" if pd.isna(bl) else str(int(bl))
    ib_txt = "NA" if pd.isna(ib) else str(int(ib))
    return f"{bl_txt}|{ib_txt}"


def parse_log(log_file: str) -> pd.DataFrame:
    enter_rows: List[Dict] = []
    timing_rows: List[Dict] = []
    loss_rows: List[Dict] = []

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            m = ENTER_RE.search(line)
            if m:
                d = m.groupdict()
                img_shape_match = IMG_SHAPE_RE.search(line)
                img_shape = None
                if img_shape_match:
                    img_shape = "(" + img_shape_match.group("img_shape").strip() + ")"

                enter_rows.append(
                    {
                        "rank": int(d["rank"]),
                        "epoch": int(d["epoch"]),
                        "iter": int(d["iter"]),
                        "enter_global_step": int(d["global_step"]),
                        "dataset_enter": d["dataset"].strip(),
                        "batch_type": d["batch_type"].strip(),
                        "batch_len": int(d["batch_len"]),
                        "img_shape": img_shape,
                    }
                )
                continue

            m = TIMING_RE.search(line)
            if m:
                d = m.groupdict()
                timing_rows.append(
                    {
                        "rank": int(d["rank"]),
                        "epoch": int(d["epoch"]),
                        "iter": int(d["iter"]),
                        "global_step": int(d["global_step"]),
                        "dataset_timing": d["dataset"],
                        "iter_total": float(d["iter_total"]),
                        "data": float(d["data"]),
                        "move": float(d["move"]),
                        "forward": float(d["forward"]),
                        "loss_time": float(d["loss_time"]),
                        "backward": float(d["backward"]),
                        "gather_loss": float(d["gather_loss"]),
                        "grad_clip": float(d["grad_clip"]),
                        "optim": float(d["optim"]),
                    }
                )
                continue

            m = LOSS_RE.search(line)
            if m:
                d = m.groupdict()
                loss_rows.append(
                    {
                        "rank": int(d["rank"]),
                        "iter": int(d["iter"]),
                        "loss": float(d["loss"]),
                    }
                )

    timing_df = pd.DataFrame(timing_rows)
    enter_df = pd.DataFrame(enter_rows)
    loss_df = pd.DataFrame(loss_rows)

    if timing_df.empty:
        raise ValueError("没有解析到 TIMING 行，请检查日志格式。")

    if not enter_df.empty:
        timing_df = timing_df.merge(
            enter_df,
            on=["rank", "epoch", "iter"],
            how="left",
        )

    if not loss_df.empty:
        timing_df = timing_df.merge(
            loss_df,
            on=["rank", "iter"],
            how="left",
        )

    if "dataset_enter" in timing_df.columns and "dataset_timing" in timing_df.columns:
        timing_df["dataset"] = timing_df["dataset_enter"].fillna(timing_df["dataset_timing"])
    elif "dataset_timing" in timing_df.columns:
        timing_df["dataset"] = timing_df["dataset_timing"]
    else:
        timing_df["dataset"] = timing_df["dataset_enter"]

    if "img_shape" not in timing_df.columns:
        timing_df["img_shape"] = None
    if "batch_len" not in timing_df.columns:
        timing_df["batch_len"] = None

    timing_df["img_bs"] = timing_df["img_shape"].apply(parse_img_bs_from_shape)
    timing_df["node_label"] = timing_df.apply(build_node_label, axis=1)

    timing_df = timing_df.sort_values(["rank", "epoch", "iter", "global_step"]).reset_index(drop=True)
    return timing_df


def filter_df_by_epoch(df: pd.DataFrame, epoch_min=None, epoch_max=None) -> pd.DataFrame:
    out = df.copy()
    if epoch_min is not None:
        out = out[out["epoch"] >= epoch_min]
    if epoch_max is not None:
        out = out[out["epoch"] <= epoch_max]
    return out.reset_index(drop=True)


def build_dataset_style_map(datasets: List[str]):
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
    cmap = plt.get_cmap("tab20")
    style_map = {}
    for i, ds in enumerate(sorted(datasets)):
        style_map[ds] = {
            "color": cmap(i % 20),
            "marker": markers[i % len(markers)],
        }
    return style_map


def save_csv(df: pd.DataFrame, out_dir: str):
    out_csv = os.path.join(out_dir, "parsed_timing_with_meta.csv")
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved: {out_csv}")

    metric_cols = [
        c for c in
        ["iter_total", "data", "move", "forward", "loss_time", "backward",
         "gather_loss", "grad_clip", "optim", "loss"]
        if c in df.columns
    ]
    if metric_cols:
        summary = df.groupby("rank")[metric_cols].mean(numeric_only=True)
        summary_path = os.path.join(out_dir, "rank_mean_summary.csv")
        summary.to_csv(summary_path)
        print(f"[OK] saved: {summary_path}")


def annotate_node_labels(ax, sub_df: pd.DataFrame, xaxis: str, metric: str, fontsize: int = 8):
    offsets = [
        (4, 6), (4, -10), (-24, 6), (-24, -10),
        (8, 14), (8, -16), (-32, 14), (-32, -16),
    ]

    for j, (_, row) in enumerate(sub_df.iterrows()):
        y = row.get(metric, None)
        if pd.isna(y):
            continue

        label = row.get("node_label", "")
        if not label:
            continue

        dx, dy = offsets[j % len(offsets)]
        ax.annotate(
            label,
            (row[xaxis], y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=fontsize,
            alpha=0.9,
        )


def make_legends(ax, ranks: List[int], datasets: List[str], style_map: Dict):
    rank_handles = [
        Line2D([0], [0], color="gray", linewidth=1.5, alpha=0.6, label=f"rank{r}")
        for r in ranks
    ]
    ds_handles = [
        Line2D(
            [0], [0],
            color=style_map[ds]["color"],
            marker=style_map[ds]["marker"],
            linestyle="None",
            markersize=7,
            label=ds
        )
        for ds in datasets
    ]

    leg1 = ax.legend(
        handles=rank_handles,
        title="Ranks",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=ds_handles,
        title="Datasets",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.45),
        borderaxespad=0.0,
    )


def plot_metric_all_ranks(
    df: pd.DataFrame,
    metric: str,
    out_dir: str,
    style_map: Dict,
    dpi: int = 160,
    xaxis: str = "global_step",
    show_legend: bool = True,
    annotate_nodes: bool = True,
    node_label_fontsize: int = 8,
):
    if metric not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    ranks = sorted(df["rank"].dropna().unique().tolist())
    datasets = sorted(df["dataset"].dropna().unique().tolist())

    has_any = False

    rank_groups = {rank: sub.sort_values([xaxis, "iter"]) for rank, sub in df.groupby("rank")}
    ds_groups = {ds: sub.sort_values([xaxis, "iter"]) for ds, sub in df.groupby("dataset")}

    for rank in ranks:
        sub = rank_groups.get(rank)
        if sub is None or sub.empty or sub[metric].notna().sum() == 0:
            continue
        ax.plot(
            sub[xaxis],
            sub[metric],
            color="gray",
            linewidth=1.2,
            alpha=0.55,
        )
        has_any = True

    for ds in datasets:
        sub = ds_groups.get(ds)
        if sub is None or sub.empty or sub[metric].notna().sum() == 0:
            continue
        ax.scatter(
            sub[xaxis],
            sub[metric],
            s=40,
            color=style_map[ds]["color"],
            marker=style_map[ds]["marker"],
            alpha=0.92,
        )

    if not has_any:
        plt.close(fig)
        return

    if annotate_nodes:
        for rank in ranks:
            sub = rank_groups.get(rank)
            if sub is None or sub.empty or sub[metric].notna().sum() == 0:
                continue
            annotate_node_labels(ax, sub, xaxis, metric, fontsize=node_label_fontsize)

    ax.set_xlabel(xaxis)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {xaxis} | gray line = rank trend, color/marker = dataset")
    ax.grid(True, alpha=0.3)

    if show_legend:
        make_legends(ax, ranks, datasets, style_map)

    plt.tight_layout(rect=[0, 0, 0.80 if show_legend else 1.0, 1])
    out_path = os.path.join(out_dir, f"{metric}_by_rank_dataset_style.png")
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[OK] saved: {out_path}")


def plot_rank_subplots(
    df: pd.DataFrame,
    rank: int,
    out_dir: str,
    style_map: Dict,
    dpi: int = 160,
    xaxis: str = "global_step",
    show_legend: bool = True,
    annotate_nodes: bool = True,
    node_label_fontsize: int = 8,
):
    metrics = [
        "iter_total", "data", "move", "forward", "loss_time",
        "backward", "gather_loss", "grad_clip", "optim", "loss",
    ]
    sub = df[df["rank"] == rank].sort_values([xaxis, "iter"])
    available_metrics = [m for m in metrics if m in sub.columns and sub[m].notna().sum() > 0]
    if not available_metrics:
        return

    datasets = sorted(sub["dataset"].dropna().unique().tolist())
    ds_groups = {ds: s.sort_values([xaxis, "iter"]) for ds, s in sub.groupby("dataset")}

    n = len(available_metrics)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.8 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, metric in enumerate(available_metrics):
        ax = axes[i]

        ax.plot(
            sub[xaxis],
            sub[metric],
            color="gray",
            linewidth=1.2,
            alpha=0.6,
        )

        for ds in datasets:
            sub_ds = ds_groups.get(ds)
            if sub_ds is None or sub_ds.empty or sub_ds[metric].notna().sum() == 0:
                continue
            ax.scatter(
                sub_ds[xaxis],
                sub_ds[metric],
                s=38,
                color=style_map[ds]["color"],
                marker=style_map[ds]["marker"],
                alpha=0.92,
            )

        if annotate_nodes:
            annotate_node_labels(ax, sub, xaxis, metric, fontsize=node_label_fontsize)

        ax.set_title(f"rank{rank} - {metric}")
        ax.set_xlabel(xaxis)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    for j in range(len(available_metrics), len(axes)):
        fig.delaxes(axes[j])

    if show_legend:
        ds_handles = [
            Line2D(
                [0], [0],
                color=style_map[ds]["color"],
                marker=style_map[ds]["marker"],
                linestyle="None",
                markersize=7,
                label=ds
            )
            for ds in datasets
        ]
        fig.legend(
            handles=ds_handles,
            title="Datasets",
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
        )

    plt.tight_layout(rect=[0, 0, 0.92 if show_legend else 1.0, 1])
    out_path = os.path.join(out_dir, f"rank{rank}_all_metrics_dataset_style.png")
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[OK] saved: {out_path}")


def plot_metric_epoch_trend(
    df: pd.DataFrame,
    metric: str,
    out_dir: str,
    style_map: Dict,
    dpi: int = 160,
    show_legend: bool = True,
    annotate_nodes: bool = True,
    node_label_fontsize: int = 8,
):
    if metric not in df.columns:
        return

    epochs = sorted(df["epoch"].dropna().unique().tolist())
    if not epochs:
        return

    ranks = sorted(df["rank"].dropna().unique().tolist())
    datasets = sorted(df["dataset"].dropna().unique().tolist())

    n = len(epochs)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    epoch_groups = {epoch: sub.sort_values(["iter", "rank"]) for epoch, sub in df.groupby("epoch")}

    for i, epoch in enumerate(epochs):
        ax = axes[i]
        sub_epoch = epoch_groups.get(epoch)
        if sub_epoch is None or sub_epoch.empty:
            continue

        for rank in ranks:
            sub_rank = sub_epoch[sub_epoch["rank"] == rank].sort_values("iter")
            if sub_rank.empty or sub_rank[metric].notna().sum() == 0:
                continue
            ax.plot(
                sub_rank["iter"],
                sub_rank[metric],
                color="gray",
                linewidth=1.0,
                alpha=0.45,
            )

        for ds in datasets:
            sub_ds = sub_epoch[sub_epoch["dataset"] == ds].sort_values("iter")
            if sub_ds.empty or sub_ds[metric].notna().sum() == 0:
                continue
            ax.scatter(
                sub_ds["iter"],
                sub_ds[metric],
                s=34,
                color=style_map[ds]["color"],
                marker=style_map[ds]["marker"],
                alpha=0.9,
            )

        mean_df = (
            sub_epoch.groupby("iter")[metric]
            .agg(mean="mean", std="std")
            .reset_index()
        )

        if not mean_df.empty:
            ax.plot(
                mean_df["iter"],
                mean_df["mean"],
                color="black",
                linewidth=2.0,
                label="mean_over_ranks",
            )
            std = mean_df["std"].fillna(0.0)
            ax.fill_between(
                mean_df["iter"],
                mean_df["mean"] - std,
                mean_df["mean"] + std,
                color="black",
                alpha=0.10,
            )

        if annotate_nodes:
            for rank in ranks:
                sub_rank = sub_epoch[sub_epoch["rank"] == rank].sort_values("iter")
                if sub_rank.empty or sub_rank[metric].notna().sum() == 0:
                    continue
                annotate_node_labels(ax, sub_rank, "iter", metric, fontsize=node_label_fontsize)

        ax.set_title(f"epoch {epoch} - {metric}")
        ax.set_xlabel("iter (within epoch)")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    for j in range(len(epochs), len(axes)):
        fig.delaxes(axes[j])

    if show_legend:
        legend_handles = [Line2D([0], [0], color="gray", linewidth=1.2, alpha=0.5, label="each rank")]
        legend_handles.append(Line2D([0], [0], color="black", linewidth=2.0, label="mean_over_ranks"))
        legend_handles.extend([
            Line2D(
                [0], [0],
                color=style_map[ds]["color"],
                marker=style_map[ds]["marker"],
                linestyle="None",
                markersize=7,
                label=ds
            )
            for ds in datasets
        ])

        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
        )

    plt.tight_layout(rect=[0, 0, 0.90 if show_legend else 1.0, 1])
    out_path = os.path.join(out_dir, f"{metric}_epoch_iter_trend.png")
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[OK] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default='output_debug/debug_log/debug_log_2x16.log', help="日志文件路径")
    parser.add_argument("--out_dir", type=str, default="output_debug/debug_log/plots_rank", help="输出目录")
    parser.add_argument(
        "--xaxis",
        type=str,
        default="global_step",
        choices=["global_step", "iter"],
        help="总体图横轴",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="iter_total,data,move,forward,loss_time,backward,gather_loss,grad_clip,optim,loss",
        help="要绘制的指标，逗号分隔",
    )
    parser.add_argument("--epoch_min", type=int, default=2, help="只保留 epoch >= epoch_min 的数据")
    parser.add_argument("--epoch_max", type=int, default=5, help="只保留 epoch <= epoch_max 的数据")
    parser.add_argument("--dpi", type=int, default=160, help="保存图片 dpi，越低越快")
    parser.add_argument("--skip_overall_plots", action="store_true", help="跳过总体图")
    parser.add_argument("--skip_rank_plots", action="store_true", help="跳过每个 rank 的总览图")
    parser.add_argument("--skip_epoch_trend", action="store_true", help="跳过按 epoch 的 iter 趋势图")
    parser.add_argument("--no_legend", action="store_true", help="不画 legend")
    parser.add_argument("--disable_node_labels", action="store_true", help="关闭每个点上的 batch_len|img_bs 标注")
    parser.add_argument("--node_label_fontsize", type=int, default=8, help="节点标注字体大小")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df_raw = parse_log(args.log_file)

    all_ranks = sorted(df_raw["rank"].dropna().unique().tolist())
    all_epochs = sorted(df_raw["epoch"].dropna().unique().tolist())
    all_datasets = sorted(df_raw["dataset"].dropna().unique().tolist())

    print(f"[INFO] all detected ranks: {all_ranks}")
    print(f"[INFO] all detected epochs: {all_epochs}")
    print(f"[INFO] all detected datasets: {all_datasets}")

    df = filter_df_by_epoch(
        df_raw,
        epoch_min=args.epoch_min,
        epoch_max=args.epoch_max,
    )

    if df.empty:
        raise ValueError(
            f"过滤后没有数据。请检查 epoch 范围：epoch_min={args.epoch_min}, epoch_max={args.epoch_max}"
        )

    save_csv(df, args.out_dir)

    ranks = sorted(df["rank"].dropna().unique().tolist())
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    epochs = sorted(df["epoch"].dropna().unique().tolist())

    style_map = build_dataset_style_map(datasets)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    show_legend = not args.no_legend
    annotate_nodes = not args.disable_node_labels

    print(f"[INFO] filtered ranks: {ranks}")
    print(f"[INFO] num_ranks: {len(ranks)}")
    print(f"[INFO] filtered epochs: {epochs}")
    print(f"[INFO] filtered datasets: {datasets}")
    print(f"[INFO] epoch filter: min={args.epoch_min}, max={args.epoch_max}")
    print(f"[INFO] annotate nodes: {annotate_nodes}")

    if not args.skip_overall_plots:
        for metric in metrics:
            plot_metric_all_ranks(
                df=df,
                metric=metric,
                out_dir=args.out_dir,
                style_map=style_map,
                dpi=args.dpi,
                xaxis=args.xaxis,
                show_legend=show_legend,
                annotate_nodes=annotate_nodes,
                node_label_fontsize=args.node_label_fontsize,
            )

    if not args.skip_rank_plots:
        for rank in ranks:
            plot_rank_subplots(
                df=df,
                rank=rank,
                out_dir=args.out_dir,
                style_map=style_map,
                dpi=args.dpi,
                xaxis=args.xaxis,
                show_legend=show_legend,
                annotate_nodes=annotate_nodes,
                node_label_fontsize=args.node_label_fontsize,
            )

    if not args.skip_epoch_trend:
        for metric in metrics:
            plot_metric_epoch_trend(
                df=df,
                metric=metric,
                out_dir=args.out_dir,
                style_map=style_map,
                dpi=args.dpi,
                show_legend=show_legend,
                annotate_nodes=annotate_nodes,
                node_label_fontsize=args.node_label_fontsize,
            )

    print("\nDone.")
    print(f"共解析到 {len(df)} 条 TIMING 记录")


if __name__ == "__main__":
    main()