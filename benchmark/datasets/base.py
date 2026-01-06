# benchmark/datasets/base.py
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from pathlib import Path

import torch
import numpy as np
from PIL import Image

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
    def __init__(self, root: str | Path):
        self.root = Path(root)

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

    # 可选
    def get_or_make_frame_ids(
        self,
        *,
        seq: str,
        seq_len: int,
        out_dir: str | Path,
        stride: int,
        random: bool = False,
        num_frames: int = 0,
        seed: int = 0,
        reuse: bool = True,
        fname: str = "frame_ids.npy",
    ) -> List[int]:
        """
        If saved frame_ids exists -> load; else generate and save.

        - save path: out_dir/frame_ids/<seq>/<fname>
        - random=False: ids = range(0, seq_len, stride)
        - random=True : ids = np.random.choice(seq_len, n, replace=False)
          where n = num_frames if >0 else len(range(0,seq_len,stride))
        """
        out_dir = Path(out_dir)
        frame_id_dir = out_dir / "frame_ids" / seq
        frame_id_dir.mkdir(parents=True, exist_ok=True)
        fp = frame_id_dir / fname

        if reuse and fp.exists():
            ids = np.load(fp).astype(np.int64).tolist()
            # sanitize
            ids = sorted({i for i in ids if 0 <= i < seq_len})
            return ids

        # generate
        if not random:
            ids = list(range(0, seq_len, max(1, int(stride))))
        else:
            default_n = len(range(0, seq_len, max(1, int(stride))))
            n = int(num_frames) if int(num_frames) > 0 else default_n
            n = max(1, min(n, seq_len))

            np.random.seed(int(seed))
            ids = np.random.choice(seq_len, size=n, replace=False).astype(np.int64).tolist()
            ids.sort()

        np.save(fp, np.array(ids, dtype=np.int32))
        return ids
