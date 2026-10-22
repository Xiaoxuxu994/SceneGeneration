# -*- coding: utf-8 -*-
"""
datasets/base/fileio.py 的单元测试：用内存里的假 moxing 模拟 obs://，验证路径映射和各种格式的读取；
另外跑一遍 datasets/base/debug.py 的单独调试入口。不需要真实数据与 moxing。
运行：python tests/test_fileio.py
"""
import io
import json
import os
import pickle
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_dataset_filters as TDF  # noqa: E402  （顺带完成 torch / hydra 的桩模块）

from datasets.base import fileio as fio  # noqa: E402

# ---------------------------------------------------------------------------
# 假 moxing：obs://bucket/... 映射到内存字典
# ---------------------------------------------------------------------------
STORE = {}
READS = []


class _File(io.BytesIO):
    def __init__(self, path, mode):
        super().__init__()
        self.path = path

    def close(self):
        STORE[self.path] = self.getvalue()
        super().close()


def _install_fake_moxing():
    f = types.SimpleNamespace()
    f.read = lambda p, binary=True: (READS.append(p), STORE[p])[1]
    f.write = lambda p, data, binary=True: STORE.__setitem__(p, data)
    f.exists = lambda p: p in STORE or any(k.startswith(p.rstrip("/") + "/") for k in STORE)
    f.is_directory = lambda p: any(k.startswith(p.rstrip("/") + "/") for k in STORE)
    f.list_directory = lambda p: sorted({k[len(p.rstrip("/")) + 1:].split("/")[0] for k in STORE if k.startswith(p.rstrip("/") + "/")})
    f.glob = lambda pat: sorted(k for k in STORE if __import__("fnmatch").fnmatch(k, pat))
    f.File = _File
    f.set_auth = lambda **kw: None
    fio._moxing_mod = types.SimpleNamespace(file=f)
    fio._moxing_checked = True


def _encode(ext, arr):
    import cv2
    ok, buf = cv2.imencode(ext, arr)
    assert ok
    return buf.tobytes()


def setup_store():
    import h5py
    STORE.clear()
    b = "obs://bucket/data/scene1"
    rgb = (np.arange(4 * 6 * 3) % 255).astype(np.uint8).reshape(4, 6, 3)
    STORE[f"{b}/rgb/000.png"] = _encode(".png", rgb)
    depth16 = (np.arange(24).reshape(4, 6) * 100).astype(np.uint16)
    STORE[f"{b}/depth/000.png"] = _encode(".png", depth16)
    STORE[f"{b}/meta.json"] = json.dumps({"fx": 1.5}).encode()
    bio = io.BytesIO(); np.save(bio, np.eye(4)); STORE[f"{b}/pose.npy"] = bio.getvalue()
    bio = io.BytesIO()
    with h5py.File(bio, "w") as h:
        h["dataset"] = np.ones((2, 3))
    STORE[f"{b}/depth.hdf5"] = bio.getvalue()
    STORE[f"{b}/traj.txt"] = b"1 2 3\n4 5 6\n"
    STORE["obs://bucket/cache/X_cache.pkl"] = pickle.dumps({"sequences": ["a", "b"]})
    return rgb, depth16


# ---------------------------------------------------------------------------
def test_local_paths_unchanged():
    fio.set_path_remap(None)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.txt")
        with fio.open(p, "w") as f:
            f.write("hi")
        assert fio.exists(p) and fio.isfile(p) and fio.isdir(d) and fio.listdir(d) == ["a.txt"]
        assert fio.glob(os.path.join(d, "*.txt")) == [p]
        assert fio.abspath("rel") == os.path.abspath("rel") and fio.abspath("obs://b/x") == "obs://b/x"


def test_remap_and_obs_reads():
    from datasets.base.utils import read_any_file
    _install_fake_moxing()
    rgb, depth16 = setup_store()
    fio.set_path_remap({"/efs/data": "obs://bucket/data", "/efs/data/scene1/rgb": "obs://bucket/data/scene1/rgb"})
    assert fio.resolve("/efs/data/scene1/meta.json") == "obs://bucket/data/scene1/meta.json"
    assert fio.resolve("/efs/database/x") == "/efs/database/x"           # 只按完整目录前缀替换
    try:
        r = read_any_file("/efs/data/scene1/rgb/000.png")
        assert r.shape == (4, 6, 3) and (r == rgb[..., ::-1]).all()    # png 经 PIL 解码为 RGB
        d = read_any_file("/efs/data/scene1/depth/000.png")
        assert d.dtype == np.uint16 and (d == depth16).all()
        assert json.load(fio.open("/efs/data/scene1/meta.json"))["fx"] == 1.5
        assert np.allclose(fio.np_load("/efs/data/scene1/pose.npy"), np.eye(4))
        assert read_any_file("/efs/data/scene1/depth.hdf5").shape == (2, 3)
        assert fio.loadtxt("/efs/data/scene1/traj.txt").shape == (2, 3)
        assert np.asarray(fio.pil_open("/efs/data/scene1/rgb/000.png")).shape == (4, 6, 3)
        assert fio.exists("/efs/data/scene1") and fio.isdir("/efs/data/scene1")
        assert fio.listdir("/efs/data/scene1") == ["depth", "depth.hdf5", "meta.json", "pose.npy", "rgb", "traj.txt"]
        assert fio.glob("/efs/data/scene1/rgb/*.png") == ["obs://bucket/data/scene1/rgb/000.png"]
        assert fio.load_pickle("obs://bucket/cache/X_cache.pkl")["sequences"] == ["a", "b"]
        fio.save_pickle({"k": 1}, "obs://bucket/cache/new.pkl")
        assert pickle.loads(STORE["obs://bucket/cache/new.pkl"]) == {"k": 1}
    finally:
        fio.set_path_remap(None)


def test_base_dataset_cache_and_remap():
    _install_fake_moxing()
    setup_store()
    ds = TDF.ToyDataset(frame_num=2, cache_dir="obs://bucket/cache", path_remap={"/efs/data": "obs://bucket/data"})
    assert ds.cache_dir == "obs://bucket/cache" and fio.resolve("/efs/data/x") == "obs://bucket/data/x"
    assert ds._load_cache("obs://bucket/cache/X_cache.pkl") and list(ds.sequences) == ["a", "b"]
    fio.set_path_remap(None)
    ds2 = TDF.ToyDataset(frame_num=2)
    assert ds2.cache_dir == (os.environ.get("DATASET_CACHE_DIR") or "/efs-mi-east4-2/z00533370/proj_3dgs/3drecon/data/dataset_cache")


def test_remap_survives_convert_attributes():
    # 训练和 debug 都会先 convert_attributes（dict -> pd.Series）再在 getitem 里 set_path_remap
    _install_fake_moxing()
    setup_store()
    ds = TDF.ToyDataset(frame_num=2, path_remap={"/efs/data": "obs://bucket/data"})
    ds.convert_attributes()
    assert isinstance(ds.path_remap, dict)
    fio.set_path_remap(None)
    fio.set_path_remap(ds.path_remap)
    assert fio.resolve("/efs/data/x") == "obs://bucket/data/x"
    import pandas as pd
    fio.set_path_remap(pd.Series({"/efs/data": "obs://bucket/data"}))
    assert fio.resolve("/efs/data/x") == "obs://bucket/data/x"
    fio.set_path_remap(pd.Series(dtype=object))
    assert fio.resolve("/efs/data/x") == "/efs/data/x"


def test_debug_main_runs():
    from datasets.base.debug import debug_main
    rc = debug_main(TDF.ToyDataset, ["--config", "", "--n", "2", "--res", "32", "32", "--frames", "3"])
    assert rc == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("ok ", f.__name__)
    print(f"{len(fns)} passed")
