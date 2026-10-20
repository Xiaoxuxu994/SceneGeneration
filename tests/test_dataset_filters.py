# -*- coding: utf-8 -*-
"""
数据过滤开关的单元测试（xiaoxu/Pi3_training_v2）：分位截断开关、extra_valid 合并、视差重选帧、序列黑名单、
ScanNet++ 的 searchsorted 修复。不需要真实数据；没有 torch 的环境会用桩模块代替 torch / torchvision。
运行：python tests/test_dataset_filters.py   或   python -m pytest tests/test_dataset_filters.py -q
"""
import os
import sys
import types
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
except ImportError:  # 本机没有 torch：只为能 import BaseDataset，换成桩模块
    for name in ['torch', 'torch.nn', 'torch.nn.functional', 'torch.utils', 'torch.utils.data',
                 'torch.utils.data.dataloader', 'torch.distributed', 'torchvision', 'torchvision.transforms',
                 'torchvision.transforms.v2', 'torchvision.transforms.v2.functional']:
        sys.modules[name] = mock.MagicMock(name=name)

try:
    import hydra  # noqa: F401
except ImportError:
    sys.modules['hydra'] = mock.MagicMock(name='hydra')

import PIL.Image  # noqa: E402
from datasets.sample_utils import common as C  # noqa: E402
from datasets.base.base_dataset import BaseDataset, baseline_over_depth  # noqa: E402


# ---------------------------------------------------------------------------
def test_threshold_default_counts_zeros():
    d = np.zeros((100, 100))
    d[:10] = np.linspace(1, 100, 1000).reshape(10, 100)      # 10% 有效像素
    C.set_depth_filter(False, False)
    old = C.threshold_depth_map(d, max_percentile=98)
    # 旧行为：分位数在全图（含 90% 的 0）上算，98 分位落在有效值里的 80% 处 → 删掉 20% 的有效像素
    assert 0.15 < 1 - (old > 0).sum() / (d > 0).sum() < 0.25


def test_threshold_valid_only_and_disabled():
    d = np.zeros((100, 100))
    d[:10] = np.linspace(1, 100, 1000).reshape(10, 100)
    try:
        C.set_depth_filter(True, False)
        new = C.threshold_depth_map(d, max_percentile=98)
        assert abs(1 - (new > 0).sum() / (d > 0).sum() - 0.02) < 0.005      # 只删有效像素的 2%
        C.set_depth_filter(False, True)
        off = C.threshold_depth_map(d, max_percentile=98, max_depth=50)
        assert (off > 0).sum() == ((d > 0) & (d <= 50)).sum()             # 只剩 max_depth 截断
        C.set_depth_filter(True, False)
        assert C.threshold_depth_map(np.zeros((4, 4)), max_percentile=98).sum() == 0   # 全无效不报错
    finally:
        C.set_depth_filter(False, False)


# ---------------------------------------------------------------------------
H = W = 32
K = np.array([[30, 0, 16], [0, 30, 16], [0, 0, 1.0]], np.float32)


def pose(x):
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = x
    return T


class ToyDataset(BaseDataset):
    """每条序列 8 帧、深度恒为 2 m；序列 'still' 的相机不动，其余序列每帧平移 0.2 m。"""

    def __init__(self, **kw):
        super().__init__(resolution=[[W, H]], transform=lambda im: im, **kw)
        self.dataset_label = 'Toy'
        self.sequences = ['moving', 'still', 'bad/seq_3']
        self.scene_to_path = {'moving': 'root/moving', 'still': 'root/still', 'bad/seq_3': 'root/bad/seq_3'}
        self.calls = []

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, idx, resolution, rng):
        seq = self.sequences[idx]
        self.calls.append(seq)
        step = 0.0 if seq == 'still' else 0.2
        views = []
        for k in range(self.frame_num):
            ev = np.ones((H, W), bool)
            ev[:, :8] = False
            views.append(dict(img=PIL.Image.new('RGB', (W, H)), depthmap=np.full((H, W), 2.0, np.float32),
                              camera_intrinsics=K, camera_pose=pose(step * k), dataset='Toy', label=seq,
                              instance=str(k), extra_valid=ev))
        return views


def test_extra_valid_only_when_enabled():
    off = ToyDataset(frame_num=4, shuffle=False)[0]
    assert all(v['valid_mask'].all() for v in off) and 'extra_valid' not in off[0]
    on = ToyDataset(frame_num=4, shuffle=False, use_dataset_masks=True)[0]
    assert all(v['valid_mask'][:, :8].sum() == 0 and v['valid_mask'][:, 8:].all() for v in on)
    assert all((v['depthmap'][:, :8] == 0).all() for v in on)


def test_low_parallax_resamples_then_accepts():
    ds = ToyDataset(frame_num=4, shuffle=False, min_baseline_ratio=0.05, baseline_retries=3)
    views = ds[1]                                   # 'still'：基线为 0，重选 3 次后照常返回
    assert len(views) == 4 and ds.calls == ['still'] * 4 and ds._n_low_parallax == 3
    ds.calls.clear()
    ds[0]                                           # 'moving'：基线/深度 = 0.3/2 = 0.15 ≥ 0.05，一次通过
    assert ds.calls == ['moving']
    still = [dict(camera_pose=pose(0.0), depthmap=np.full((2, 2), 2.0), valid_mask=np.ones((2, 2), bool)) for _ in range(4)]
    moving = [dict(camera_pose=pose(0.2 * k), depthmap=np.full((2, 2), 2.0), valid_mask=np.ones((2, 2), bool)) for k in range(4)]
    assert baseline_over_depth(still) == 0.0 and abs(baseline_over_depth(moving) - 0.15) < 1e-6   # 两两间距中位数 0.3 / 深度 2


def test_blacklist_by_name_or_path():
    ds = ToyDataset(frame_num=2, seq_blacklist=['seq_3', 'still'])
    ds._apply_blacklist()
    assert list(ds.sequences) == ['moving']


def test_scannetpp_sorter_fix():
    full = np.array(['c.jpg', 'a.jpg', 'b.jpg'])                 # 无序
    sel = np.array(['b.jpg', 'c.jpg'])
    sorter = np.argsort(full)
    old = np.searchsorted(full, sel, sorter=sorter)
    new = sorter[np.searchsorted(full, sel, sorter=sorter)]
    assert list(full[new]) == list(sel) and list(full[old]) != list(sel)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for f in fns:
        f()
        print('ok ', f.__name__)
    print(f'{len(fns)} passed')
