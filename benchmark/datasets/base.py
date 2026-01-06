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

    def get_frame_ids(
        self,
        seq: str,
        seq_len: int,
        out_dir: str | Path,
        *,
        mode: str = "stride",          # "stride" | "random"
        stride: int = 1,
        max_seq_len: int = 0,          # stride mode: take first max_seq_len; random mode: sample n=max_seq_len (or all if 0)
        seed: int = 0,
        reuse: bool = True,
        fname: str = "frame_ids.npy",
    ) -> List[int]:
        out_dir = Path(out_dir)
        frame_id_dir = out_dir / "frame_ids" / seq
        frame_id_dir.mkdir(parents=True, exist_ok=True)
        fp = frame_id_dir / fname

        if reuse and fp.exists():
            ids = np.load(fp).astype(np.int64).tolist()
            return sorted({i for i in ids if 0 <= i < seq_len})

        mode = str(mode).lower()
        stride = max(1, int(stride))
        max_seq_len = int(max_seq_len)

        if seq_len <= 0:
            ids: List[int] = []

        elif mode == "stride":
            ids = list(range(0, seq_len, stride))
            if max_seq_len > 0:
                ids = ids[:max_seq_len]

        elif mode == "random":
            # n = max_seq_len (if >0), else sample all frames
            n = seq_len if max_seq_len <= 0 else min(max_seq_len, seq_len)
            n = max(1, n)
            np.random.seed(int(seed))
            ids = np.random.choice(seq_len, size=n, replace=False).astype(np.int64).tolist()
            ids.sort()

        else:
            raise ValueError(f"Unknown mode={mode}, expected stride|random")

        np.save(fp, np.array(ids, dtype=np.int32))
        return ids
