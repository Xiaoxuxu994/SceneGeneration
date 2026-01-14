# benchmark/datasets/dl3dv.py
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from PIL import Image, ImageFile

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch

ImageFile.LOAD_TRUNCATED_IMAGES = True


# -------------------------
# helpers
# -------------------------
def _extract_int(s: str) -> Optional[int]:
    stem = Path(s.replace("\\", "/")).stem
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def _sorted_by_int_then_name(paths: List[Path]) -> List[Path]:
    def key(p: Path):
        n = _extract_int(p.name)
        return (n if n is not None else 10**18, p.name)

    return sorted(paths, key=key)


def _resolve_sibling_by_stem(src_img: Path, dst_dir: Path, exts: List[str]) -> Path:
    """
    用 src_img 的 stem 去 dst_dir 里找同名文件（不同后缀），如:
      rgb/frame_00123.png -> moge_depth/frame_00123.exr
      rgb/frame_00123.png -> moge_depth_mask/frame_00123.png
    """
    stem = src_img.stem
    for ext in exts:
        p = dst_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # 退化：完全同名（包含后缀）
    p2 = dst_dir / src_img.name
    if p2.exists():
        return p2
    raise FileNotFoundError(f"[DL3DV] cannot resolve sibling for {src_img.name} under {dst_dir}")


def _read_exr_depth(path: Path) -> np.ndarray:
    """
    读取 EXR 深度:
      - 返回 float32 (H,W)
      - 如果读出来是 (H,W,C) 则取第0通道
      - nan/inf -> 0
    """
    try:
        import imageio.v2 as iio  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"[DL3DV] imageio not available to read EXR. Please install imageio. err={e}"
        )

    try:
        d = iio.imread(str(path))
    except Exception as e:
        raise RuntimeError(
            f"[DL3DV] failed to read EXR depth: {path}\n"
            f"Try: pip install imageio imageio-ffmpeg (or install OpenEXR plugin)\n"
            f"err={type(e).__name__}: {e}"
        )

    d = np.array(d)
    if d.ndim == 3:
        d = d[..., 0]
    d = d.astype(np.float32)

    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return d


def _load_pose_pack_npy(p: Path) -> Dict[str, np.ndarray]:
    """
    npy 内容是 object 包装的 dict:
      array({'extr_mat': (N,4,4), 'intr_mat': (N,3,3)}, dtype=object)
    """
    obj = np.load(str(p), allow_pickle=True)

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        # shape=() / (1,) / ...
        try:
            obj = obj.item()
        except Exception:
            obj = obj.reshape(-1)[0].item()

    if not isinstance(obj, dict):
        raise ValueError(f"[DL3DV] pose npy is not a dict-like object: {p} type={type(obj)}")

    if "extr_mat" not in obj:
        raise KeyError(f"[DL3DV] pose npy missing key 'extr_mat': {p}")
    if "intr_mat" not in obj:
        raise KeyError(f"[DL3DV] pose npy missing key 'intr_mat': {p}")

    extr = np.asarray(obj["extr_mat"], dtype=np.float32)
    intr = np.asarray(obj["intr_mat"], dtype=np.float32)

    if extr.ndim != 3 or extr.shape[1:] != (4, 4):
        raise ValueError(f"[DL3DV] extr_mat shape invalid: {extr.shape} in {p}")
    if intr.ndim != 3 or intr.shape[1:] != (3, 3):
        raise ValueError(f"[DL3DV] intr_mat shape invalid: {intr.shape} in {p}")

    return {"extr_mat": extr, "intr_mat": intr}


def _pick_pose_pack_npy(scene_dir: Path, min_len: int) -> Path:
    """
    在 scene_dir 下挑一个包含 extr_mat/intr_mat 的 npy，且 N>=min_len
    多个候选时选 N 最大的那个
    """
    best_p: Optional[Path] = None
    best_n = -1

    for p in sorted(scene_dir.glob("*.npy")):
        if not p.is_file():
            continue
        try:
            pack = _load_pose_pack_npy(p)
        except Exception:
            continue
        n = int(pack["extr_mat"].shape[0])
        if n >= min_len and n > best_n:
            best_n = n
            best_p = p

    if best_p is None:
        raise FileNotFoundError(f"[DL3DV] cannot find pose-pack npy with N>={min_len} under {scene_dir}")
    return best_p


# -------------------------
# dataset
# -------------------------
class DL3DVDataset(BaseDataset):
    """
    Scene structure (your new format):
      root/<scene_id>/
        rgb/                 frame_00123.png
        moge_depth/          frame_00123.exr
        moge_depth_mask/     frame_00123.png
        meta_info.json
        *.npy                object dict: {'extr_mat': (N,4,4), 'intr_mat': (N,3,3)}

    Notes:
      - intr_mat assumed normalized (0~1). We'll convert to pixel K using real W/H.
      - extr_mat default assumed w2c (common convention for "extrinsics").
        If your extr_mat is c2w, construct dataset with extr_is_w2c=False.
    """

    def __init__(
        self,
        root: str | Path,
        image_dir: str = "rgb",
        depth_dir: str = "moge_depth",
        mask_dir: str = "moge_depth_mask",
        meta_name: str = "meta_info.json",
        extr_is_w2c: bool = True,
    ):
        super().__init__(root)
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.meta_name = meta_name
        self.extr_is_w2c = extr_is_w2c
        self._cache: Dict[str, Dict[str, Any]] = {}

    def list_sequences(self, split: str = "all") -> List[str]:
        seqs: List[str] = []
        for d in sorted(os.listdir(self.root)):
            seq_dir = Path(self.root) / d
            if not seq_dir.is_dir():
                continue
            if not (seq_dir / self.meta_name).exists():
                continue
            if not (seq_dir / self.image_dir).is_dir():
                continue
            if not (seq_dir / self.depth_dir).is_dir():
                continue
            if not (seq_dir / self.mask_dir).is_dir():
                continue
            seqs.append(d)
        return seqs

    def count_seq_len(self, seq: str) -> int:
        return int(self._get_meta(seq)["num_frames"])

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        meta = self._get_meta(seq)

        img_paths_all: List[str] = meta["image_paths"]
        depth_paths_all: List[str] = meta["depth_paths"]
        pose_npy_path: str = meta["pose_npy_path"]

        images = [img_paths_all[i] for i in frame_ids]
        poses = [f"{pose_npy_path}#{i}" for i in frame_ids]    # pose.npy#idx
        depths = [depth_paths_all[i] for i in frame_ids]       # exr path
        return images, poses, depths

    # -------- loaders --------
    def _load_rgb(self, image_paths: List[str]):
        image_pils = [Image.open(p).convert("RGB") for p in image_paths]
        images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
        return image_pils, images_cpu

    def _load_depth(self, depth_paths: List[str], mask_paths: List[str], image_pils: List[Image.Image]) -> torch.Tensor:
        """
        depth: exr
        mask : png -> bool
        输出 depths: (V,H,W) float32；mask 外 depth 置 0
        """
        W, H = image_pils[0].size
        outs: List[np.ndarray] = []

        for dp, mp in zip(depth_paths, mask_paths):
            d = _read_exr_depth(Path(dp))  # (h,w)
            m = np.array(Image.open(mp).convert("L")) > 0

            # 对齐尺寸（如不一致则 nearest resize）
            if d.shape != (H, W):
                d_img = Image.fromarray(d.astype(np.float32))
                d_img = d_img.resize((W, H), resample=Image.NEAREST)
                d = np.array(d_img).astype(np.float32)

            if m.shape != (H, W):
                m_img = Image.fromarray((m.astype(np.uint8) * 255))
                m_img = m_img.resize((W, H), resample=Image.NEAREST)
                m = (np.array(m_img) > 0)

            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            d[~m] = 0.0
            outs.append(d.astype(np.float32))

        depths = np.stack(outs, axis=0).astype(np.float32)
        return torch.from_numpy(depths)

    def _load_w2c(self, seq: str, pose_paths: List[str]) -> torch.Tensor:
        if len(pose_paths) == 0:
            return torch.zeros((0, 4, 4), dtype=torch.float64)

        meta = self._get_meta(seq)
        extr_all: np.ndarray = meta["extr_all"]  # (N,4,4)
        idxs = [int(s.split("#", 1)[1]) for s in pose_paths]

        extr = torch.from_numpy(extr_all[idxs].astype(np.float64))  # (V,4,4)

        if self.extr_is_w2c:
            w2c = extr
        else:
            w2c = closed_form_inverse_se3(extr)

        # safety: non-finite -> identity
        finite = torch.isfinite(w2c).view(w2c.shape[0], -1).all(dim=1)
        if not bool(finite.all()):
            bad = (~finite).nonzero(as_tuple=False).squeeze(-1).tolist()
            print(f"[DL3DV][WARN] non-finite w2c frames={bad} -> replace with identity")
            eye = torch.eye(4, dtype=w2c.dtype, device=w2c.device)
            for i in bad:
                w2c[i] = eye
        return w2c

    def _load_K(self, seq: str, frame_ids: List[int], real_w: int, real_h: int) -> torch.Tensor:
        """
        intr_mat 是归一化 K（0~1），转成像素 K:
          K[0,:] *= W
          K[1,:] *= H
        """
        meta = self._get_meta(seq)
        intr_all_norm: np.ndarray = meta["intr_all_norm"]  # (N,3,3)

        K = intr_all_norm[frame_ids].copy().astype(np.float32)  # (V,3,3)
        K[:, 0, :] *= float(real_w)
        K[:, 1, :] *= float(real_h)

        # safety
        K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return torch.from_numpy(K)

    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_paths, pose_paths, depth_paths = self.make_paths(seq, frame_ids)

        image_pils, images_cpu = self._load_rgb(image_paths)

        meta = self._get_meta(seq)
        mask_paths_all: List[str] = meta["mask_paths"]
        mask_paths = [mask_paths_all[i] for i in frame_ids]

        depths = self._load_depth(depth_paths, mask_paths, image_pils)

        raw_w2c = self._load_w2c(seq, pose_paths)

        W, H = image_pils[0].size
        raw_K = self._load_K(seq, frame_ids=frame_ids, real_w=W, real_h=H)

        return SequenceBatch(
            seq=seq,
            frame_ids=frame_ids,
            image_paths=image_paths,
            pose_paths=pose_paths,
            depth_paths=depth_paths,
            image_pils=image_pils,
            images_cpu=images_cpu,
            depths=depths,
            raw_w2c=raw_w2c,
            raw_K=raw_K,
            meta={
                "dataset": "dl3dv",
                "has_gt_depth": True,
            },
        )

    def _get_meta(self, seq: str) -> Dict[str, Any]:
        if seq in self._cache:
            return self._cache[seq]

        seq_dir = Path(self.root) / seq
        meta_path = seq_dir / self.meta_name
        if not meta_path.exists():
            raise FileNotFoundError(f"[DL3DV] meta not found: {meta_path}")

        # meta_info.json 目前不强依赖（你说参数从 npy 读）
        # 这里读一下只是为了存在性校验/未来扩展
        try:
            with open(meta_path, "r") as f:
                _ = json.load(f)
        except Exception:
            # 不让它成为硬失败
            pass

        rgb_dir = seq_dir / self.image_dir
        depth_dir = seq_dir / self.depth_dir
        mask_dir = seq_dir / self.mask_dir

        # list rgb images
        imgs: List[Path] = []
        for ext in [".png", ".jpg", ".jpeg"]:
            imgs += list(rgb_dir.glob(f"*{ext}"))
        if len(imgs) == 0:
            raise FileNotFoundError(f"[DL3DV] no rgb images in {rgb_dir}")

        imgs = _sorted_by_int_then_name(imgs)
        image_paths = [str(p) for p in imgs]

        # real image size
        im0 = Image.open(image_paths[0])
        W, H = im0.size

        # depth(exr) + mask(png) aligned by stem
        depth_paths: List[str] = []
        mask_paths: List[str] = []
        for ip in imgs:
            dp = _resolve_sibling_by_stem(ip, depth_dir, exts=[".exr"])
            mp = _resolve_sibling_by_stem(ip, mask_dir, exts=[".png"])
            depth_paths.append(str(dp))
            mask_paths.append(str(mp))

        # pose/intr from pose-pack npy (object dict)
        pose_npy = _pick_pose_pack_npy(seq_dir, min_len=len(image_paths))
        pack = _load_pose_pack_npy(pose_npy)
        extr_all = pack["extr_mat"]                 # (N,4,4)
        intr_all_norm = pack["intr_mat"]            # (N,3,3) normalized

        if extr_all.shape[0] < len(image_paths) or intr_all_norm.shape[0] < len(image_paths):
            raise ValueError(
                f"[DL3DV] pose/intr length < #images: extr={extr_all.shape[0]} intr={intr_all_norm.shape[0]} imgs={len(image_paths)}"
            )

        meta = {
            "meta_path": str(meta_path),
            "num_frames": len(image_paths),
            "image_paths": image_paths,
            "depth_paths": depth_paths,
            "mask_paths": mask_paths,
            "pose_npy_path": str(pose_npy),
            "extr_all": extr_all,
            "intr_all_norm": intr_all_norm,
            "w": W,
            "h": H,
        }
        self._cache[seq] = meta
        return meta
