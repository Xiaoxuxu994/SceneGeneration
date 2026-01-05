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

    # 可选：不同数据集默认 stride/invalid depth 等
    def default_stride(self, view: str) -> int:
        return 40 if view == "dense" else 200
