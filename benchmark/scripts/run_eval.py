#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch_npu  # noqa: F401
from tqdm import tqdm

# 如果你项目里靠 _bootstrap 设置 pythonpath，建议保留
import _bootstrap  # noqa: F401

from lib.evaluation import eval_angle, eval_trajectory, eval_depth_map, eval_point_map

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.styles import Border, Side

def build_summary_rows(pose_thr: int, mean_dict: Dict[str, Any]) -> List[Tuple[str, str, str, str, Any]]:
    """返回 summary 的 rows: (group, metric, stat, kind, value)"""
    def get_first(keys: List[str]) -> Any:
        for k in keys:
            if k in mean_dict and mean_dict[k] is not None:
                return mean_dict[k]
        return None

    depth_absrel = get_first(["depth_Abs Rel"])
    depth_delta  = get_first(["depth_δ < 1.25"])

    rows: List[Tuple[str, str, str, str, Any]] = []

    # pose (6)
    rows += [
        ("pose", f"RRA@{pose_thr}↑", "", "pct", get_first(["pose_rra_acc"])),
        ("pose", f"RTA@{pose_thr}↑", "", "pct", get_first(["pose_rta_acc"])),
        ("pose", f"AUC@{pose_thr}↑", "", "pct", get_first(["pose_auc"])),
        ("pose", "ATE↓", "", "num", get_first(["traj_ate"])),
        ("pose", "RPE trans↓", "", "num", get_first(["traj_rpe_trans"])),
        ("pose", "RPE rot↓", "", "num", get_first(["traj_rpe_rot"])),
    ]

    # pointmap (6)
    rows += [
        ("pointmap", "Acc.↓", "Mean", "num", get_first(["pm_acc_mean"])),
        ("pointmap", "Acc.↓", "Med.", "num", get_first(["pm_acc_med"])),
        ("pointmap", "Comp.↓", "Mean", "num", get_first(["pm_comp_mean"])),
        ("pointmap", "Comp.↓", "Med.", "num", get_first(["pm_comp_med"])),
        ("pointmap", "N.C.↑", "Mean", "num", get_first(["pm_nc1_mean", "pm_nc2_mean"])),
        ("pointmap", "N.C.↑", "Med.", "num", get_first(["pm_nc1_med", "pm_nc2_med"])),
    ]

    # depth (2)
    rows += [
        ("depth", "Abs Rel.↓", "", "num", depth_absrel),
        ("depth", "δ < 1.25↑", "", "num", depth_delta),
    ]
    return rows


def print_summary_to_terminal(dataset_name: str, pose_thr: int, mean_dict: Dict[str, Any]) -> None:
    rows = build_summary_rows(pose_thr=pose_thr, mean_dict=mean_dict)

    print("\n" + "=" * 90)
    print(f"[SUMMARY] dataset={dataset_name}  pose_thr={pose_thr}")
    print("-" * 90)
    print(f"{'group':<10} | {'metric':<18} | {'stat':<4} | {'value':>12}")
    print("-" * 90)

    for grp, metric, stat, kind, v in rows:
        if kind == "pct":
            pv = safe_percent_value(v)
            s = f"{pv:.2f}" if pv is not None else "NA"
        else:
            try:
                fv = float(v)
                s = f"{fv:.3f}"
            except Exception:
                s = "NA"
        stat_show = stat if stat else "-"
        print(f"{grp:<10} | {metric:<18} | {stat_show:<4} | {s:>12}")

    print("=" * 90 + "\n")


def extract_mean_dict_from_ws(ws) -> Dict[str, Any]:
    """
    从 write_sheet() 写出的 per_seq/per_scene 表里，读取最后一行（MEAN行）的数值：
      - 第1行是header
      - ws.max_row 是MEAN行
    返回：{col_name: mean_value}
    """
    header = [ws.cell(row=1, column=j).value for j in range(1, ws.max_column + 1)]
    mean_row = ws.max_row
    mean_dict: Dict[str, Any] = {}
    for j, k in enumerate(header, start=1):
        if k is None:
            continue
        mean_dict[str(k)] = ws.cell(row=mean_row, column=j).value
    return mean_dict


def safe_percent_value(v: Any) -> Optional[float]:
    """把 0~1 或 0~100 的数都转成 Excel 百分比用的 0~1。"""
    if v is None:
        return None
    try:
        v = float(v)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v * 100.0

def write_summary_sheet(ws, dataset_name: str, pose_thr: int, mean_dict: Dict[str, Any]) -> None:
    ws.title = "summary"

    def get_first(keys: List[str]) -> Any:
        for k in keys:
            if k in mean_dict and mean_dict[k] is not None:
                return mean_dict[k]
        return None

    depth_absrel = get_first(["depth_Abs Rel"])
    depth_delta  = get_first(["depth_δ < 1.25"])

    # 按截图结构：A dataset | B group | C metric | D stat | E value
    rows = rows = build_summary_rows(pose_thr=pose_thr, mean_dict=mean_dict)

    # 写表
    for i, (grp, metric, stat, kind, v) in enumerate(rows, start=1):
        ws.cell(i, 2, grp)
        ws.cell(i, 3, metric)
        ws.cell(i, 4, stat)
        cell = ws.cell(i, 5, None)

        if kind == "pct":
            pv = safe_percent_value(v)
            cell.value = pv if pv is not None else None
            cell.number_format = "0.00"
        else:
            # 常见指标保留三位；你想两位也行
            try:
                cell.value = float(v) if v is not None else None
            except Exception:
                cell.value = None
            cell.number_format = "0.000"

    # 合并单元格（和截图一致的层级感）
    ws.cell(1, 1, dataset_name)
    ws.merge_cells("A1:A14")
    ws.merge_cells("B1:B6")
    ws.merge_cells("B7:B12")
    ws.merge_cells("B13:B14")
    ws.merge_cells("C7:C8")
    ws.merge_cells("C9:C10")
    ws.merge_cells("C11:C12")

    # 样式
    bold = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin = Side(style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for r in range(1, 15):
        for c in range(1, 6):
            cell = ws.cell(r, c)
            cell.alignment = center
            cell.border = border

    ws["A1"].font = bold
    for addr in ["B1", "B7", "B13"]:
        ws[addr].font = bold
    for r in range(1, 15):
        ws.cell(r, 3).font = bold

    # 列宽
    widths = [12, 10, 18, 8, 12]
    for j, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(j)].width = w

# -------------------------
# IO / utils
# -------------------------
def np_load(p: Path) -> np.ndarray:
    return np.load(str(p))


def torch_f32(x: np.ndarray) -> torch.Tensor:
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return torch.from_numpy(x)


def pick_pred_pointmap(pred_seq: Path, use_predicted_pointmap: bool) -> Optional[Path]:
    # p = pred_seq / "pred_point_maps_by_unprojection.npy"
    # if p.exists():
    #     return p
    # p = pred_seq / "pred_point_maps.npy"
    # if p.exists():
    #     return p
    
    p_predicted = pred_seq / "pred_point_maps.npy"
    p_unprojected = pred_seq / "pred_point_maps_by_unprojection.npy"
    
    if use_predicted_pointmap and p_predicted.exists():
        return p_predicted
    elif not use_predicted_pointmap and p_unprojected.exists():
        return p_unprojected
    else:
        return None


def is_number(v: Any) -> bool:
    try:
        fv = float(v)
        return np.isfinite(fv)
    except Exception:
        return False


def safe_mean(values: List[Any]) -> Optional[float]:
    xs = [float(v) for v in values if is_number(v)]
    if not xs:
        return None
    return float(np.mean(xs))


def list_rel_seqs(gt_root: Path) -> List[Path]:
    # rels: List[Path] = []
    # for scene_dir in sorted(gt_root.iterdir()):
    #     if not scene_dir.is_dir():
    #         continue
    #     for seq_dir in sorted(scene_dir.iterdir()):
    #         if seq_dir.is_dir():
    #             rels.append(seq_dir.relative_to(gt_root))
                
    def _walk(d: Path):
        entries = list(d.iterdir())
        if any(not e.is_dir() for e in entries):
            yield d
            return
        for sub in sorted([e for e in entries if e.is_dir()]):
            yield from _walk(sub)
            
    rels = [p.relative_to(gt_root) for p in _walk(gt_root)]

    return rels


def flatten_depth_dict(d: Dict[str, Any], prefix: str = "depth_") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        # depth_evaluation 通常返回标量
        try:
            out[prefix + str(k)] = float(v)
        except Exception:
            out[prefix + str(k)] = v
    return out

# -------------------------
# Excel writer
# -------------------------
def write_sheet(ws, rows: List[Dict[str, Any]], title: str) -> None:
    ws.title = title

    # collect columns (stable)
    keyset = set()
    for r in rows:
        keyset.update(r.keys())

    front = ["scene", "seq", "status", "num_frames"]
    cols = [k for k in front if k in keyset]
    rest = sorted([k for k in keyset if k not in cols])
    cols += rest

    # header style
    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(color="FFFFFF", bold=True)
    header_align = Alignment(horizontal="center", vertical="center")

    ws.append(cols)
    for j, k in enumerate(cols, start=1):
        c = ws.cell(row=1, column=j, value=k)
        c.fill = header_fill
        c.font = header_font
        c.alignment = header_align

    # data
    for r in rows:
        ws.append([r.get(k, None) for k in cols])

    # MEAN row (python mean, ignore non-numeric)
    mean_row = ws.max_row + 1
    ws.cell(row=mean_row, column=1, value="MEAN").font = Font(bold=True)

    for j, k in enumerate(cols, start=1):
        if k in ("scene", "seq", "status"):
            continue
        vals = []
        for i in range(2, mean_row):  # only data rows
            vals.append(ws.cell(row=i, column=j).value)
        m = safe_mean(vals)
        if m is not None:
            ws.cell(row=mean_row, column=j, value=m)

    # widths
    for j, k in enumerate(cols, start=1):
        ws.column_dimensions[get_column_letter(j)].width = max(12, min(42, len(k) + 2))

    ws.freeze_panes = "A2"


def save_excel(per_seq_rows: List[Dict[str, Any]],
               per_scene_rows: List[Dict[str, Any]],
               out_xlsx: Path,
               dataset_name: str,
               pose_thr: int=30) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws1 = wb.active
    write_sheet(ws1, per_seq_rows, "per_seq")

    ws2 = wb.create_sheet("per_scene")
    write_sheet(ws2, per_scene_rows, "per_scene")

    # --- summary: 直接用 per_seq 最后一行 MEAN ---
    mean_dict = extract_mean_dict_from_ws(ws1)
    ws3 = wb.create_sheet("summary")
    write_summary_sheet(ws3, dataset_name=dataset_name, pose_thr=pose_thr, mean_dict=mean_dict)
    print_summary_to_terminal(dataset_name=dataset_name, pose_thr=pose_thr, mean_dict=mean_dict)

    wb.save(str(out_xlsx))


# -------------------------
# main eval loop
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default='/mnt/nvme3/z00533370/3drecon/result/0107_ckpt15000/7scenes/test_stride10_max8/',
                    help="包含 gt/ 和 vggt/ 的根目录，例如 outputs/.../test_dense_stride40")
    ap.add_argument("--dataset_name", type=str, default="7scenes")
    ap.add_argument("--gt_subdir", type=str, default="gt")
    ap.add_argument("--pred_subdir", type=str, default="reliev3r")

    ap.add_argument("--pose_threshold", type=int, default=30)
    ap.add_argument("--pointmap_threshold", type=float, default=0.1)
    
    ap.add_argument("--predicted_pointmap", type=int, default=0)
    
    ap.add_argument("--depth_align", type=str, default="scale_shift", choices=["scale", "scale_shift"])
    ap.add_argument("--device", type=str, default="npu:0",
                    help="eval_depth_map 内部对齐用的 device，例如 npu:1")

    ap.add_argument("--out_xlsx", type=str, default="",
                    help="输出 excel 路径；为空则写到 root 下 metrics_<pred>.xlsx")
    args = ap.parse_args()

    root = Path(args.root)
    gt_root = root / args.gt_subdir
    pred_root = root / args.pred_subdir

    if not gt_root.exists():
        raise FileNotFoundError(f"gt_root not found: {gt_root}")
    if not pred_root.exists():
        raise FileNotFoundError(f"pred_root not found: {pred_root}")

    out_xlsx = Path(args.out_xlsx) if args.out_xlsx else (root / f"metrics_{args.pred_subdir}.xlsx")

    rels = list_rel_seqs(gt_root)
    print(f"[INFO] num_seqs={len(rels)}")
    print(f"[INFO] gt_root={gt_root}")
    print(f"[INFO] pred_root={pred_root}")
    print(f"[INFO] device={args.device}")
    print(f"[INFO] out={out_xlsx}")

    per_seq_rows: List[Dict[str, Any]] = []
    # for per-scene aggregation
    scene_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for rel in tqdm(rels, desc="eval", unit="seq"):
        gt_seq = gt_root / rel
        pred_seq = pred_root / rel

        scene = rel.parts[0] if len(rel.parts) > 0 else ""
        row: Dict[str, Any] = {
            "scene": scene,
            "seq": str(rel),
            "status": "ok",
        }

        # try:
        #     # ---------- load pose ----------
        #     gt_extr_p = gt_seq / "gt_extrinsics.npy"
        #     pred_extr_p = pred_seq / "pred_extrinsics.npy"
        #     if not gt_extr_p.exists() or not pred_extr_p.exists():
        #         row["status"] = "missing_extrinsics"
        #         per_seq_rows.append(row)
        #         continue

        #     pred_extr = torch_f32(np_load(pred_extr_p))
        #     gt_extr = torch_f32(np_load(gt_extr_p))

        #     # ---------- pose metrics ----------
        #     rra_acc, rta_acc, auc = eval_angle(pred_extr, gt_extr, threshold=args.pose_threshold)
        #     ate, rpe_trans, rpe_rot = eval_trajectory(pred_extr, gt_extr)

        #     row["pose_rra_acc"] = float(rra_acc)
        #     row["pose_rta_acc"] = float(rta_acc)
        #     row["pose_auc"] = float(auc)
        #     row["traj_ate"] = float(ate)
        #     row["traj_rpe_trans"] = float(rpe_trans)
        #     row["traj_rpe_rot"] = float(rpe_rot)

        #     # ---------- load depth/mask ----------
        #     gt_depth_p = gt_seq / "gt_depth_maps.npy"
        #     pred_depth_p = pred_seq / "pred_depth_maps.npy"
        #     gt_mask_p = gt_seq / "gt_depth_masks.npy"

        #     if pred_depth_p.exists():
        #         dd = np_load(pred_depth_p)
        #         row["num_frames"] = int(dd.shape[0]) if dd.ndim == 3 else 1

        #     # ---------- depth metrics ----------
        #     if pred_depth_p.exists() and gt_depth_p.exists() and gt_mask_p.exists():
        #         pred_d = np_load(pred_depth_p).astype(np.float32)
        #         gt_d = np_load(gt_depth_p).astype(np.float32)
        #         mask = np_load(gt_mask_p).astype(bool)

        #         # refer to https://github.com/yyfz/Pi3/blob/evaluation/videodepth/eval.py#L54
        #         if args.depth_align == "scale":
        #             depth_res = eval_depth_map(
        #                 pred_d, gt_d, mask,
        #                 align_with_scale=True,
        #                 device=args.device,
        #             )
        #         elif args.depth_align == "scale_shift":
        #             depth_res = eval_depth_map(
        #                 pred_d, gt_d, mask,
        #                 align_with_lstsq=True,
        #                 # align_with_lad=True,
        #                 # align_with_lad2=True,
        #                 device=args.device,
        #             )
        #         else:
        #             raise NotImplementedError
                
        #         # print(depth_res)
        #         row.update(flatten_depth_dict(depth_res, prefix="depth_"))
        #     else:
        #         # 不覆盖已有 status，但记录一下
        #         # if row["status"] == "ok":
        #         #     row["status"] = "missing_depth_or_mask"
        #         row["status"] = "missing_depth_or_mask"

        #     # ---------- point map metrics ----------
        #     gt_pm_p = gt_seq / "gt_point_maps.npy"
        #     pred_pm_p = pick_pred_pointmap(pred_seq, args.predicted_pointmap != 0)

        #     if gt_pm_p.exists() and gt_mask_p.exists() and pred_pm_p is not None:
        #         pred_pm = np_load(pred_pm_p).astype(np.float32)
        #         gt_pm = np_load(gt_pm_p).astype(np.float32)
        #         mask = np_load(gt_mask_p).astype(bool)

        #         acc, acc_med, nc1, nc1_med, comp, comp_med, nc2, nc2_med = eval_point_map(
        #             pred_pm, gt_pm, mask, threshold=args.pointmap_threshold
        #         )
        #         row["pm_acc_mean"] = float(acc)
        #         row["pm_acc_med"] = float(acc_med)
        #         row["pm_nc1_mean"] = float(nc1)
        #         row["pm_nc1_med"] = float(nc1_med)
        #         row["pm_comp_mean"] = float(comp)
        #         row["pm_comp_med"] = float(comp_med)
        #         row["pm_nc2_mean"] = float(nc2)
        #         row["pm_nc2_med"] = float(nc2_med)
        #     else:
        #         row["status"] = "missing_pointmap_or_mask"

        # except Exception as e:
        #     row["status"] = f"error:{type(e).__name__}"
        #     row["error_msg"] = str(e)

        # per_seq_rows.append(row)
        # scene_bucket[scene].append(row)
        
        # ---------- load pose ----------
        gt_extr_p = gt_seq / "gt_extrinsics.npy"
        pred_extr_p = pred_seq / "pred_extrinsics.npy"
        if not gt_extr_p.exists() or not pred_extr_p.exists():
            row["status"] = "missing_extrinsics"
            per_seq_rows.append(row)
            continue

        pred_extr = torch_f32(np_load(pred_extr_p))
        gt_extr = torch_f32(np_load(gt_extr_p))

        # ---------- pose metrics ----------
        rra_acc, rta_acc, auc = eval_angle(pred_extr, gt_extr, threshold=args.pose_threshold)
        ate, rpe_trans, rpe_rot = eval_trajectory(pred_extr, gt_extr)

        row["pose_rra_acc"] = float(rra_acc)
        row["pose_rta_acc"] = float(rta_acc)
        row["pose_auc"] = float(auc)
        row["traj_ate"] = float(ate)
        row["traj_rpe_trans"] = float(rpe_trans)
        row["traj_rpe_rot"] = float(rpe_rot)

        # ---------- load depth/mask ----------
        gt_depth_p = gt_seq / "gt_depth_maps.npy"
        pred_depth_p = pred_seq / "pred_depth_maps.npy"
        gt_mask_p = gt_seq / "gt_depth_masks.npy"

        if pred_depth_p.exists():
            dd = np_load(pred_depth_p)
            row["num_frames"] = int(dd.shape[0]) if dd.ndim == 3 else 1

        # ---------- depth metrics ----------
        if pred_depth_p.exists() and gt_depth_p.exists() and gt_mask_p.exists():
            pred_d = np_load(pred_depth_p).astype(np.float32)
            gt_d = np_load(gt_depth_p).astype(np.float32)
            mask = np_load(gt_mask_p).astype(bool)

            # refer to https://github.com/yyfz/Pi3/blob/evaluation/videodepth/eval.py#L54
            if args.depth_align == "scale":
                depth_res = eval_depth_map(
                    pred_d, gt_d, mask,
                    align_with_scale=True,
                    device=args.device,
                )
            elif args.depth_align == "scale_shift":
                depth_res = eval_depth_map(
                    pred_d, gt_d, mask,
                    align_with_lstsq=True,
                    # align_with_lad=True,
                    # align_with_lad2=True,
                    device=args.device,
                )
            else:
                raise NotImplementedError
            
            # print(depth_res)
            row.update(flatten_depth_dict(depth_res, prefix="depth_"))
        else:
            # 不覆盖已有 status，但记录一下
            # if row["status"] == "ok":
            #     row["status"] = "missing_depth_or_mask"
            row["status"] = "missing_depth_or_mask"

        # ---------- point map metrics ----------
        gt_pm_p = gt_seq / "gt_point_maps.npy"
        pred_pm_p = pick_pred_pointmap(pred_seq, args.predicted_pointmap != 0)

        if gt_pm_p.exists() and gt_mask_p.exists() and pred_pm_p is not None:
            pred_pm = np_load(pred_pm_p).astype(np.float32)
            gt_pm = np_load(gt_pm_p).astype(np.float32)
            mask = np_load(gt_mask_p).astype(bool)

            acc, acc_med, nc1, nc1_med, comp, comp_med, nc2, nc2_med = eval_point_map(
                pred_pm, gt_pm, mask, threshold=args.pointmap_threshold
            )
            row["pm_acc_mean"] = float(acc)
            row["pm_acc_med"] = float(acc_med)
            row["pm_nc1_mean"] = float(nc1)
            row["pm_nc1_med"] = float(nc1_med)
            row["pm_comp_mean"] = float(comp)
            row["pm_comp_med"] = float(comp_med)
            row["pm_nc2_mean"] = float(nc2)
            row["pm_nc2_med"] = float(nc2_med)
        
        exit()

    # -------- per-scene aggregation --------
    # 对每个 scene，把 status=ok 的 seq 做均值
    numeric_keys = set()
    for r in per_seq_rows:
        for k, v in r.items():
            if k in ("scene", "seq", "status", "error_msg"):
                continue
            if is_number(v):
                numeric_keys.add(k)

    per_scene_rows: List[Dict[str, Any]] = []
    for scene, rs in sorted(scene_bucket.items(), key=lambda x: x[0]):
        ok_rs = [r for r in rs if str(r.get("status", "")) == "ok"]
        out: Dict[str, Any] = {"scene": scene, "seq": "ALL", "status": "ok", "num_seqs": len(ok_rs)}
        if not ok_rs:
            out["status"] = "no_ok_seq"
            per_scene_rows.append(out)
            continue

        for k in sorted(numeric_keys):
            out[k] = safe_mean([r.get(k, None) for r in ok_rs])
        per_scene_rows.append(out)

    save_excel(per_seq_rows, per_scene_rows, out_xlsx, dataset_name=args.dataset_name)
    print("[OK] wrote excel:", out_xlsx)


if __name__ == "__main__":
    main()
