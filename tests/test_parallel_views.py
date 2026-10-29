# -*- coding: utf-8 -*-
"""
样本内多帧并发读取（DATA_VIEW_THREADS）的单元测试：顺序不变、确实并发、每帧 rng 可复现、异常照常进入重试。
不需要真实数据。运行：python tests/test_parallel_views.py
"""
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_dataset_filters as TDF  # noqa: E402  （torch / hydra 桩）
from datasets.base import base_dataset as bd  # noqa: E402


class SlowToy(TDF.ToyDataset):
    """和各 loader 一样：逐帧 views.append(self._load_one_view(..., rng=rng))；每帧模拟 0.2s 的 OBS 读取。"""
    delay = 0.2
    fail_frame = None

    def _load_one_view(self, seq, k, rng):
        time.sleep(self.delay)
        if self.fail_frame == k:
            raise IOError(f"obs read failed frame {k}")
        v = dict(img=TDF.PIL.Image.new('RGB', (TDF.W, TDF.H)), depthmap=np.full((TDF.H, TDF.W), 2.0, np.float32),
                 camera_intrinsics=TDF.K, camera_pose=TDF.pose(0.2 * k), dataset='Toy', label=seq,
                 instance=str(k))
        v['_draw'] = float(rng.random())
        v['_thread'] = threading.current_thread().name
        return v

    def _get_views(self, idx, resolution, rng):
        seq = self.sequences[idx]
        views = []
        for k in range(self.frame_num):
            views.append(self._load_one_view(seq, k, rng=rng))
        return views


def _run(threads, seed=0, **kw):
    bd.DATA_VIEW_THREADS = threads
    ds = SlowToy(frame_num=8, seed=seed, **kw)
    ds._rng = np.random.default_rng(seed)
    t = time.time()
    views = ds._get_views_maybe_parallel(0, (TDF.W, TDF.H), ds._rng)
    return views, time.time() - t


def test_serial_unchanged():
    views, dt = _run(1)
    assert [v['instance'] for v in views] == [str(k) for k in range(8)]
    assert dt >= 8 * SlowToy.delay * 0.9
    assert all(not v['_thread'].startswith('view') for v in views)


def test_parallel_order_speed_and_reproducible():
    v1, dt = _run(8)
    assert [v['instance'] for v in v1] == [str(k) for k in range(8)]       # 顺序与串行一致
    assert dt < 3 * SlowToy.delay                                          # 8 帧并发，约 1 帧的时间
    assert any(v['_thread'].startswith('view') for v in v1)
    v2, _ = _run(8)
    assert [v['_draw'] for v in v1] == [v['_draw'] for v in v2]            # 同一 seed 每帧随机数相同
    assert len(set(v['_draw'] for v in v1)) == 8                           # 各帧 rng 互不相同
    assert '_load_one_view' not in vars(SlowToy(frame_num=8))              # 实例上的临时替换已撤销


def test_parallel_exception_goes_to_retry():
    bd.DATA_VIEW_THREADS = 8
    SlowToy.fail_frame = 3
    try:
        ds = SlowToy(frame_num=8)
        try:
            ds._get_views_maybe_parallel(0, (TDF.W, TDF.H), np.random.default_rng(0))
            raise AssertionError("should raise")
        except IOError as e:
            assert "frame 3" in str(e)
        assert '_load_one_view' not in vars(ds)
        # 走完整 __getitem__：异常被重试逻辑接住，最终仍报错（每次都失败），不会卡死
        try:
            ds[(0, 0, 8)]
        except IOError:
            pass
        assert ds._n_fail >= 1
    finally:
        SlowToy.fail_frame = None
        bd.DATA_VIEW_THREADS = 1


class HandleToy(SlowToy):
    """模拟 GTA-sfm：with 打开文件，把句柄传给 _load_one_view；with 结束后句柄失效。"""

    class _H:
        def __init__(self):
            self.open = True

        def close(self):
            self.open = False

    def _load_one_view(self, seq, k, rng, f=None):
        assert f is None or f.open, "handle used after close"
        return super()._load_one_view(seq, k, rng)

    def _get_views(self, idx, resolution, rng):
        seq = self.sequences[idx]
        views = []
        f = self._H()
        try:
            for k in range(self.frame_num):
                views.append(self._load_one_view(seq, k, rng=rng, f=f))
        finally:
            f.close()
        return views


def test_parallel_with_open_handle_runs_inline():
    bd.DATA_VIEW_THREADS = 8
    try:
        ds = HandleToy(frame_num=8)
        views = ds._get_views_maybe_parallel(0, (TDF.W, TDF.H), np.random.default_rng(0))
        assert [v['instance'] for v in views] == [str(k) for k in range(8)]
    finally:
        bd.DATA_VIEW_THREADS = 1


def test_getitem_parallel_end_to_end():
    bd.DATA_VIEW_THREADS = 8
    try:
        ds = SlowToy(frame_num=8)
        t = time.time()
        views = ds[(0, 0, 8)]
        assert len(views) == 8 and all('pts3d' in v and v['valid_mask'].any() for v in views)
        assert time.time() - t < 4 * SlowToy.delay
    finally:
        bd.DATA_VIEW_THREADS = 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("ok ", f.__name__)
    print(f"{len(fns)} passed")
