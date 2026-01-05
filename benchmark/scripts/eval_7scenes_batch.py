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
    rels: List[Path] = []
    for scene_dir in sorted(gt_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for seq_dir in sorted(scene_dir.iterdir()):
            if seq_dir.is_dir():
                rels.append(seq_dir.relative_to(gt_root))
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
               out_xlsx: Path) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws1 = wb.active
    write_sheet(ws1, per_seq_rows, "per_seq")

    ws2 = wb.create_sheet("per_scene")
    write_sheet(ws2, per_scene_rows, "per_scene")

    wb.save(str(out_xlsx))


# -------------------------
# main eval loop
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default='./outputs/7scenes/test_dense_stride40/',
                    help="包含 gt/ 和 vggt/ 的根目录，例如 outputs/.../test_dense_stride40")
    ap.add_argument("--gt_subdir", type=str, default="gt")
    ap.add_argument("--pred_subdir", type=str, default="vggt")

    ap.add_argument("--pose_threshold", type=int, default=30)
    ap.add_argument("--pointmap_threshold", type=float, default=0.1)
    
    ap.add_argument("--predicted_pointmap", type=int, default=1)
    
    ap.add_argument("--depth_align", type=str, default="scale_shift", choices=["scale", "scale_shift"])
    ap.add_argument("--depth_device", type=str, default="npu:0",
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
    print(f"[INFO] depth_device={args.depth_device}")
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

        try:
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
                        device=args.depth_device,
                    )
                elif args.depth_align == "scale_shift":
                    depth_res = eval_depth_map(
                        pred_d, gt_d, mask,
                        align_with_lstsq=True,
                        # align_with_lad=True,
                        # align_with_lad2=True,
                        device=args.depth_device,
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
            else:
                row["status"] = "missing_pointmap_or_mask"

        except Exception as e:
            row["status"] = f"error:{type(e).__name__}"
            row["error_msg"] = str(e)

        per_seq_rows.append(row)
        scene_bucket[scene].append(row)

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

    save_excel(per_seq_rows, per_scene_rows, out_xlsx)
    print("[OK] wrote excel:", out_xlsx)


if __name__ == "__main__":
    main()
