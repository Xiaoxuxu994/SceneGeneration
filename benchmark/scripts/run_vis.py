#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional, Iterator, Tuple

import numpy as np
import torch
import open3d as o3d
import imageio.v2 as imageio

# 关键：先 import _bootstrap（如果它负责把 vggt 加入 sys.path）
import _bootstrap  # noqa: F401
from vggt.utils.camera import export_cam_as_ply
# from models.reliev3r.utils.log_utils import export_pcds_and_cam_as_ply

def iter_seq_dirs(root: Path) -> Iterator[Tuple[Path, Path]]:
    def _walk(d: Path) -> Iterator[Path]:
        entries = list(d.iterdir())
        if any(not e.is_dir() for e in entries):
            yield d
            return
        for sub in sorted([e for e in entries if e.is_dir()]):
            yield from _walk(sub)

    for seq_dir in _walk(root):
        yield seq_dir, seq_dir.relative_to(root)

def to_4x4_batch(T: np.ndarray) -> np.ndarray:
    """
    支持：
      (S,4,4)
      (S,3,4) -> pad成 (S,4,4)
      (4,4)   -> (1,4,4)
      (3,4)   -> (1,4,4)
    """
    if T.ndim == 2:
        T = T[None, ...]
    if T.shape[-2:] == (4, 4):
        return T
    if T.shape[-2:] == (3, 4):
        S = T.shape[0]
        out = np.zeros((S, 4, 4), dtype=T.dtype)
        out[:, :3, :] = T
        out[:, 3, 3] = 1.0
        return out
    raise ValueError("Unexpected extrinsics shape: {}".format(T.shape))


def export_pointcloud(
    point_maps: np.ndarray,
    out_ply: Path,
    mask: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
) -> None:
    """
    point_maps: (S,H,W,3)
    mask: (S,H,W) optional
    colors: (S,H,W,3) uint8/float optional (0~255 or 0~1)
    """
    
    # if point_maps.ndim != 4 or point_maps.shape[-1] != 3:
    #     raise ValueError("point_maps must be (S,H,W,3), got {}".format(point_maps.shape))

    pts = point_maps.reshape(-1, 3)

    cols = None
    if colors is not None:
        if colors.shape[:3] != point_maps.shape[:3]:
            raise ValueError(f"colors shape mismatch: colors {colors.shape} vs pts {point_maps.shape}")
        if colors.ndim == 3:  # (S,H,W) 灰度
            colors = np.stack([colors, colors, colors], axis=-1)
        if colors.shape[-1] == 4:  # RGBA -> RGB
            colors = colors[..., :3]
        if colors.shape[-1] != 3:
            raise ValueError(f"colors must be (...,3), got {colors.shape}")
        cols = colors.reshape(-1, 3)

    if mask is not None:
        m = mask.reshape(-1).astype(bool)
        pts = pts[m]
        if cols is not None:
            cols = cols[m]

    good = np.isfinite(pts).all(axis=1)
    pts = pts[good]
    if cols is not None:
        cols = cols[good]

    pcd = o3d.geometry.PointCloud()
    if pts.shape[0] > 0:
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
        if cols is not None:
            c = cols.astype(np.float32)
            if c.max() > 1.5:  # uint8 0~255
                c = c / 255.0
            c = np.clip(c, 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float32))

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_ply), pcd)



def export_cameras(c2w: np.ndarray, out_ply: Path) -> None:
    """export_cam_as_ply expects torch tensor (B,S,4,4)."""
    c2w = to_4x4_batch(c2w).astype(np.float32)  # (S,4,4)
    cams = torch.from_numpy(c2w)[None, ...]     # (1,S,4,4)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    export_cam_as_ply(str(out_ply), cams)


def depth_to_vis(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    import matplotlib.cm as cm
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0.0
    d = np.clip(d, vmin, vmax)
    norm = (d - vmin) / max(1e-6, (vmax - vmin))
    rgb = (cm.turbo(norm)[..., :3] * 255.0).astype(np.uint8)  # <-- 这里
    return rgb


def export_depth(depth_maps: np.ndarray, out_dir: Path, vmin: float, vmax: float) -> None:
    """depth_maps: (S,H,W) meters"""
    if depth_maps.ndim != 3:
        raise ValueError("depth_maps must be (S,H,W), got {}".format(depth_maps.shape))

    out_dir.mkdir(parents=True, exist_ok=True)
    S = depth_maps.shape[0]
    for i in range(S):
        d = depth_maps[i].astype(np.float32)
        vis = depth_to_vis(d, vmin=vmin, vmax=vmax)
        imageio.imwrite(str(out_dir / "depth_vis_{:03d}.png".format(i)), vis)

def load_any(seq_dir: Path, names) -> Optional[np.ndarray]:
    for n in names:
        p = seq_dir / n
        if p.exists():
            return np.load(p)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=str,
        default='outputs/scannetpp/test_stride10_max8/gt',
        help="dump 根目录，例如 .../test_dense_stride40/gt 或 .../test_dense_stride40/vggt",
    )
    ap.add_argument("--depth_vmin", type=float, default=0.1)
    ap.add_argument("--depth_vmax", type=float, default=10.0)
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        raise FileNotFoundError("dir not found: {}".format(root))

    is_gt = (root.name.lower() == "gt")

    # 在 dir 同目录下新建 <basename>_vis（按你要求取最后一个 basename）
    vis_root = root.parent / (root.name + "_vis")
    vis_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] src:", root)
    print("[INFO] dst:", vis_root)
    print("[INFO] is_gt:", is_gt)

    ok, skip = 0, 0
    for seq_dir, rel in iter_seq_dirs(root):
        out_dir = vis_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            if is_gt:
                extr = seq_dir / "gt_extrinsics.npy"
                ptsf = seq_dir / "gt_point_maps.npy"
                depf = seq_dir / "gt_depth_maps.npy"
                maskf = seq_dir / "gt_depth_masks.npy"
                aligned_ptsf = seq_dir / "gt_point_maps_aligned.npy"

                if not (extr.exists() and ptsf.exists() and depf.exists()):
                    skip += 1
                    continue

                c2w = np.load(extr)
                pts = np.load(ptsf)
                dep = np.load(depf).astype(np.float32)
                mask = np.load(maskf).astype(bool) if maskf.exists() else None

                ## add image
                imgf = seq_dir / "gt_images.npy"
                gt_images = None
                if imgf.exists():
                    gt_images = np.load(imgf)  # 期望 (S,H,W,3) 或类似
                    # 统一成 uint8 RGB
                    if gt_images.ndim == 3:  # (S,H,W) 灰度
                        gt_images = np.stack([gt_images, gt_images, gt_images], axis=-1)
                    if gt_images.shape[-1] == 4:
                        gt_images = gt_images[..., :3]
                    if gt_images.dtype != np.uint8:
                        x = gt_images.astype(np.float32)
                        if x.max() <= 1.5:  # 0~1
                            x = x * 255.0
                        gt_images = np.clip(x, 0, 255).astype(np.uint8)

                    # 如果帧数不一致，最简单处理：截断到最小帧数
                    S = min(int(pts.shape[0]), int(gt_images.shape[0]))
                    if gt_images.shape[0] != pts.shape[0]:
                        gt_images = gt_images[:S]
                        pts = pts[:S]
                        dep = dep[:S]
                        if mask is not None:
                            mask = mask[:S]

                    # --- GT 情况下保存每一帧 png ---
                    img_out_dir = out_dir / "images"
                    img_out_dir.mkdir(parents=True, exist_ok=True)
                    for i in range(gt_images.shape[0]):
                        imageio.imwrite(str(img_out_dir / f"image_{i:03d}.png"), gt_images[i])

            else:
                extr = seq_dir / "pred_extrinsics.npy"
                depf = seq_dir / "pred_depth_maps.npy"
                aligned_ptsf = seq_dir / "pred_point_maps_aligned.npy"
                if not (extr.exists() and depf.exists()):
                    skip += 1
                    continue

                c2w = np.load(extr)
                dep = np.load(depf).astype(np.float32)
                pts = load_any(seq_dir, ["pred_point_maps_by_unprojection.npy", "pred_point_maps.npy"])
                # pts = load_any(seq_dir, ["pred_point_maps.npy"])
                if pts is None:
                    skip += 1
                    continue
                mask = None

                ## add image
                gt_root = root.parent / "gt"
                imgf = gt_root / rel/ "gt_images.npy"
                gt_images = None
                if imgf.exists():
                    gt_images = np.load(imgf)  # 期望 (S,H,W,3) 或类似
                    # 统一成 uint8 RGB
                    if gt_images.ndim == 3:  # (S,H,W) 灰度
                        gt_images = np.stack([gt_images, gt_images, gt_images], axis=-1)
                    if gt_images.shape[-1] == 4:
                        gt_images = gt_images[..., :3]
                    if gt_images.dtype != np.uint8:
                        x = gt_images.astype(np.float32)
                        if x.max() <= 1.5:  # 0~1
                            x = x * 255.0
                        gt_images = np.clip(x, 0, 255).astype(np.uint8)

                    # 如果帧数不一致，最简单处理：截断到最小帧数
                    S = min(int(pts.shape[0]), int(gt_images.shape[0]))
                    if gt_images.shape[0] != pts.shape[0]:
                        gt_images = gt_images[:S]
                        pts = pts[:S]
                        dep = dep[:S]
                        if mask is not None:
                            mask = mask[:S]

            # export_pointcloud(pts, out_dir / "vis_pointcloud.ply", mask=mask)
            export_pointcloud(pts, out_dir / "vis_pointcloud.ply", mask=mask, colors=gt_images)
            export_cameras(c2w, out_dir / "vis_cameras.ply")            
            export_depth(dep, out_dir / "vis_depth", vmin=args.depth_vmin, vmax=args.depth_vmax)

            try:
                aligned_pts = np.load(aligned_ptsf)
                export_pointcloud(aligned_pts, out_dir / "vis_pointcloud_aligned.ply")
            except Exception as e:
                # print(e)
                pass

            ok += 1
            print("[OK] {}".format(rel))

        except Exception as e:
            skip += 1
            print("[SKIP] {}  err={}".format(rel, e))

    print("[DONE] ok={}, skip={}".format(ok, skip))


if __name__ == "__main__":
    main()
