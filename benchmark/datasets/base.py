# benchmark/datasets/base.py
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from lib.matching import *
from lib.load_fn import load_and_preprocess_images
from lib.geometry import closed_form_inverse_se3

@dataclass
class SequenceBatch:
    seq: str
    frame_ids: List[int]
    image_paths: List[str]
    pose_paths: List[str]
    depth_paths: List[str]

    image_pils: List[Image.Image]         # len=V, 原始PIL
    images_cpu: torch.Tensor              # (V,3,H,W) CPU tensor, 预处理后
    depths: torch.Tensor                  # (V,H,W) meters, CPU tensor

    raw_w2c: torch.Tensor                 # (V,4,4) or (V,3,4) 统一为 w2c
    raw_K: torch.Tensor                   # (V,3,3) 每帧内参

    meta: dict                            # 数据集特定字段（可选）

class BaseDataset(ABC):
    def __init__(self, root: str | Path, num_seq: int):
        self.root = Path(root)
        self.num_seq = num_seq

    @abstractmethod
    def list_sequences(self, split: str) -> List[str]:
        ...

    @abstractmethod
    def count_seq_len(self, seq: str) -> int:
        ...

    @abstractmethod
    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        ...

    @abstractmethod
    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        ...

    # def get_frame_ids(
    #     self,
    #     seq: str,
    #     seq_len: int,
    #     out_dir: str | Path,
    #     *,
    #     mode: str = "stride",          # "stride" | "random"
    #     stride: int = 1,
    #     max_seq_len: int = 0,          # stride mode: take first max_seq_len; random mode: sample n=max_seq_len (or all if 0)
    #     seed: int = 0,
    #     reuse: bool = True,
    #     fname: str = "frame_ids.npy"
    # ) -> List[int]:
    #     out_dir = Path(out_dir)
    #     frame_id_dir = out_dir / "frame_ids" / seq
    #     frame_id_dir.mkdir(parents=True, exist_ok=True)
    #     fp = frame_id_dir / fname

    #     if reuse and fp.exists():
    #         ids = np.load(fp).astype(np.int64).tolist()
    #         print(f"[INFO] Load frame id from {fp}\n")
    #         return sorted({i for i in ids if 0 <= i < seq_len})

    #     mode = str(mode).lower()
    #     stride = max(1, int(stride))
    #     max_seq_len = int(max_seq_len)

    #     if seq_len <= 0:
    #         ids: List[int] = []

    #     elif mode == "stride":
    #         ids = list(range(0, seq_len, stride))
    #         if max_seq_len > 0:
    #             ids = ids[:max_seq_len]

    #     elif mode == "random":
    #         n = seq_len if max_seq_len <= 0 else min(max_seq_len, seq_len)
    #         n = max(1, n)
    #         np.random.seed(int(seed))
    #         ids = np.random.choice(seq_len, size=n, replace=False).astype(np.int64).tolist()
    #         ids.sort()
    #     else:
    #         raise ValueError(f"Unknown mode={mode}, expected stride|random")

    #     np.save(fp, np.array(ids, dtype=np.int32))
    #     return ids
    
    def _normalize_K(self, pix_K, H, W):
        norm_K = pix_K.clone()
        norm_K[..., 0, 0] = pix_K[..., 0, 0] / W
        norm_K[..., 1, 1] = pix_K[..., 1, 1] / H
        norm_K[..., 0, 2] = pix_K[..., 0, 2] / W
        norm_K[..., 1, 2] = pix_K[..., 1, 2] / H
        norm_K[..., 2, 2] = 1.0
        return norm_K
    
    def _denormalize_K(self, norm_K, W, H):
        pix_K = norm_K.clone()
        pix_K[..., 0, 0] = norm_K[..., 0, 0] * W
        pix_K[..., 1, 1] = norm_K[..., 1, 1] * H
        pix_K[..., 0, 2] = norm_K[..., 0, 2] * W
        pix_K[..., 1, 2] = norm_K[..., 1, 2] * H
        pix_K[..., 2, 2] = 1.0
        return pix_K
    
    def view_sampler_iterative(self, visibs: torch.Tensor, max_seq_len: int, seq_len: int, thre: float):
        """
        Sample views of a sample from a clip in an iterative approach.
        Given a sampled view set S from a clip C, each time add a view x in C-S, which has a minimal maximum covisibility with views in S that \
            above a given threshold. In this way, the given threshold represents sparcity of the sampled views.

        Args:
            visibs (torch.Tensor): Covisibility of pair-wise tracking pixels. Tensor of shape (V(V-1)/2, gs*gs)
            clip_viwe_num (int): Number of views in the clip.
        """
        visib_mat = deflat_visibility_mat(visibs, max_seq_len) # v v, upper-triangle matrix.
        visib_mat[visib_mat < thre] = 0
        
        view_ids = []
        sampled_mask = torch.zeros(max_seq_len, dtype=torch.bool)
        # first_view_id = legal_roots[np.random.randint(low=0, high=len(legal_roots))]
        first_view_id = np.random.randint(low=0, high=max_seq_len)
        view_ids.append(first_view_id)
        sampled_mask[first_view_id] = True
        sampled_view_cnt = 1
        while sampled_view_cnt < seq_len:
            _visib_mat = visib_mat.clone()
            _visib_mat[sampled_mask] = 0
            _visib_mat[:, ~sampled_mask] = 0
            _visib_vec = _visib_mat.max(dim=-1).values # v
            _visib_vec[_visib_vec == 0] = 2
            new_view_id = _visib_vec.min(dim=0).indices
            if _visib_vec[new_view_id] == 2:
                return None
            else:
                view_ids.append(new_view_id.item())
                sampled_mask[new_view_id] = True
                sampled_view_cnt += 1
                
        return np.array(view_ids)
    
    def get_frame_ids(
        self,
        seq: str,
        seq_len: int,                  # actual length of the sampled views from the original clip
        out_dir: str | Path,
        *,
        mode: str = "stride",          # "stride" | "random"
        stride: int = 1,
        max_seq_len: int = -1,         # maximum length of the original clip, default to use all
        seed: int = 0,
        reuse: bool = True,
        fname: str = "frame_ids.npy",
        max_retry: int = 10
    ) -> List[int]:
        out_dir = Path(out_dir)
        frame_id_dir = out_dir / "frame_ids" / seq
        frame_id_dir.mkdir(parents=True, exist_ok=True)
        fp = frame_id_dir / fname

        if reuse and fp.exists():
            ids = np.load(fp).astype(np.int64).tolist()
            print(f"[INFO] Load frame id from {fp}")
        else:
            mode = str(mode).lower()

            assert seq_len > 0, "length of the sampled sequence must be greater than 0"

            if mode == "stride":
                stride = max(1, int(stride))
                actual_seq_len = self.count_seq_len(seq)
                max_seq_len = actual_seq_len if max_seq_len == -1 else min(actual_seq_len, max_seq_len)
            
                ids = np.arange(0, max_seq_len, stride)
                ids = ids[:seq_len]

            elif mode == "random":
                # np.random.seed(int(seed))
                
                actual_seq_len = self.count_seq_len(seq)
                max_seq_len = actual_seq_len if max_seq_len == -1 else min(actual_seq_len, max_seq_len)
                all_ids = [int(i) for i in range(max_seq_len)]
                
                batch = self.load_sequence(seq, all_ids)
                        
                depths          = batch.depths           # V,H,W
                w2c             = batch.raw_w2c.float()  # V,4,4
                raw_K           = batch.raw_K            # V,3,3
                grid_size       = 100
                match_eps       = 0.01
                samples_to_keep = 10000
                min_thres       = 0.08
                
                # if the intrinsics is not normalized
                if raw_K[0, 0, 0] - 0.5 > 0.05 or raw_K[0, 1, 1] - 0.5 > 0.05:
                    V,H,W = depths.shape
                    norm_K = self._normalize_K(raw_K, H, W)
                
                ## Dataset extrinsics are defined as world-to-camera. Transform to camera-to-world when passed.
                _, _, visibs = exhaustive_geo2match(
                    depths, closed_form_inverse_se3(w2c), norm_K, eps=match_eps, samples_to_keep=samples_to_keep, grid_size=grid_size)
                visibs = visibs.to(dtype=torch.float32).mean(dim=-1)
                
                for try_id in range(max_retry):
                    ids = self.view_sampler_iterative(visibs, max_seq_len, seq_len, min_thres)
                    
                    if ids is None:
                        ids = np.zeros_like(ids)
                        print(f"[INFO] not a valid sequence - starting sampling trial {try_id+1}")
                        continue
                            
            else:
                raise ValueError(f"Unknown mode={mode}, expected stride|random")
            
            ids.sort()
            ids = ids.tolist()
            np.save(fp, np.array(ids, dtype=np.int32))
        
        return ids
