# -*- coding: utf-8 -*-
"""
统一的文件读写层：同时支持本地路径和华为 OBS（obs://），各 loader 通过 `from datasets.base import fileio as fio` 使用。

- 本地路径：直接调用 os / glob / cv2 / PIL / numpy / h5py，行为与改造前完全一致。
- obs:// 路径：惰性导入 moxing，按"整读字节 → 内存解码"的方式读取（比流式 open 快），每 N 次读取手动 gc
  一次（moxing 早期版本有内存泄漏，做法同生产脚本 obs_read/load_storage.py）。凭证沿用 moxing 的环境变量。
- 路径前缀映射（path remap）：数据集缓存 pkl 里存的往往是建缓存时的本地绝对路径（如 /efs-m3ga/...）。
  配置 `path_remap: {"/efs-m3ga/xxx": "obs://bucket/xxx"}` 后，所有读取会先按最长前缀替换，再判断走本地还是 OBS，
  不需要重建缓存。BaseDataset 在构造和每次取样本前调用 set_path_remap()，不同数据集的映射互不影响。
  目录层级对不上、前缀替换解决不了时，用正则规则：键写成 "re:<正则>"，值是替换串（\\1 引用分组），
  先于前缀规则按配置顺序尝试，第一条匹配的生效。例如 Taskonomy 的 rgb/<场景>/<场景>/x → rgb/taskonomy/<场景>/x。
"""
import builtins
import glob as _glob
import io
import os
import re
import tempfile

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")   # 必须在 import cv2 之前；cv2.imdecode 读 EXR 需要
import cv2  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# moxing（惰性导入）
# ---------------------------------------------------------------------------
_moxing_mod = None
_moxing_checked = False
_OBS_GC_INTERVAL = 25
_obs_read_count = 0


def _ensure_moxing():
    global _moxing_mod, _moxing_checked
    if not _moxing_checked:
        _moxing_checked = True
        try:
            os.environ.setdefault("MOX_SILENT_MODE", "1")
            import moxing as mox
            mox.file.set_auth(path_style=True, is_secure=False, retry=10, retry_wait=1)
            try:   # 压掉 moxing 自身日志，避免多 worker 高频读取时刷屏
                import logging
                from moxing.framework.util import runtime
                runtime.reset_logger(level=logging.CRITICAL)
            except Exception:
                pass
            _moxing_mod = mox
        except ImportError:
            _moxing_mod = None
    return _moxing_mod


def _mox():
    mox = _ensure_moxing()
    if mox is None:
        raise RuntimeError("读取 obs:// 路径需要 moxing（ModelArts 镜像自带；本地可 pip install moxing）")
    return mox


def _obs_read_tick():
    global _obs_read_count
    _obs_read_count += 1
    if _obs_read_count >= _OBS_GC_INTERVAL:
        import gc
        gc.collect()
        _obs_read_count = 0


# ---------------------------------------------------------------------------
# 路径映射
# ---------------------------------------------------------------------------
_REMAP = []      # [(src_prefix, dst_prefix)]，按 src 长度降序
_REMAP_RE = []   # [(compiled_regex, repl)]，按配置顺序


def set_path_remap(remap=None):
    """remap: dict {src_prefix: dst_prefix}、[(src, dst), ...] 或 None。键以 "re:" 开头的是正则规则。"""
    global _REMAP, _REMAP_RE
    if remap is None or len(remap) == 0:     # 不用 `not remap`：pd.Series / ndarray 的真值有歧义
        _REMAP, _REMAP_RE = [], []
        return
    items = list(remap.items()) if hasattr(remap, "items") else [tuple(x) for x in remap]
    _REMAP_RE = [(re.compile(str(a)[3:]), str(b)) for a, b in items if str(a).startswith("re:")]
    _REMAP = sorted(((str(a), str(b)) for a, b in items if not str(a).startswith("re:")), key=lambda x: -len(x[0]))


def resolve(path):
    if not isinstance(path, str):
        return path
    for pat, repl in _REMAP_RE:
        new, n = pat.subn(repl, path, count=1)
        if n:
            return new
    for src, dst in _REMAP:
        if path == src or path.startswith(src.rstrip("/") + "/"):
            return dst.rstrip("/") + path[len(src.rstrip("/")):]
    return path


def is_obs(path):
    return isinstance(path, str) and path.startswith("obs://")


def abspath(path):
    """os.path.abspath 会把 obs://b/a 破坏成 <cwd>/obs:/b/a；obs 路径原样返回。缓存文件名用的是映射前的路径。"""
    return path if is_obs(path) else os.path.abspath(path)


# ---------------------------------------------------------------------------
# 目录与存在性
# ---------------------------------------------------------------------------
def exists(path):
    p = resolve(path)
    return _mox().file.exists(p) if is_obs(p) else os.path.exists(p)


def isdir(path):
    p = resolve(path)
    return _mox().file.is_directory(p) if is_obs(p) else os.path.isdir(p)


def isfile(path):
    p = resolve(path)
    if is_obs(p):
        mox = _mox()
        return mox.file.exists(p) and not mox.file.is_directory(p)
    return os.path.isfile(p)


def listdir(path):
    p = resolve(path)
    if is_obs(p):
        return [x.rstrip("/").split("/")[-1] for x in _mox().file.list_directory(p)]
    return os.listdir(p)


def glob(pattern, recursive=False):
    """返回的是映射后的真实路径；obs 下用 mox.file.glob（只支持 * 通配）。"""
    p = resolve(pattern)
    if is_obs(p):
        return list(_mox().file.glob(p))
    return _glob.glob(p, recursive=recursive)


def makedirs(path, exist_ok=True):
    p = resolve(path)
    if is_obs(p):
        return   # OBS 无需建目录
    os.makedirs(p, exist_ok=exist_ok)


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------
def read_bytes(path):
    p = resolve(path)
    if not is_obs(p):
        with builtins.open(p, "rb") as f:
            return f.read()
    mox = _mox()
    last = None
    for _ in range(3):
        try:
            data = mox.file.read(p, binary=True)
            _obs_read_tick()
            return data
        except Exception as e:
            last = e
    raise RuntimeError(f"obs 读取失败（重试 3 次）: {p}") from last


def open(path, mode="r", *args, **kwargs):  # noqa: A001  与内置 open 同名，loader 中以 fio.open 调用
    p = resolve(path)
    if not is_obs(p):
        return builtins.open(p, mode, *args, **kwargs)
    if any(c in mode for c in "wa+"):
        return _mox().file.File(p, mode)
    data = read_bytes(p)
    if "b" in mode:
        return io.BytesIO(data)
    return io.StringIO(data.decode(kwargs.get("encoding") or "utf-8"))


def pil_open(fp, *args, **kwargs):
    """Image.open 的替代；返回已 load 的 PIL.Image（obs 读进内存后解码）。也接受文件对象。"""
    from PIL import Image
    if isinstance(fp, str):
        p = resolve(fp)
        if is_obs(p):
            im = Image.open(io.BytesIO(read_bytes(p)), *args, **kwargs)
            im.load()
            return im
        return Image.open(p, *args, **kwargs)
    return Image.open(fp, *args, **kwargs)


def cv2_imread(path, flags=cv2.IMREAD_COLOR):
    p = resolve(path)
    if not is_obs(p):
        return cv2.imread(p, flags)
    return cv2.imdecode(np.frombuffer(read_bytes(p), np.uint8), flags)


def np_load(file, *args, **kwargs):
    if isinstance(file, str):
        p = resolve(file)
        if is_obs(p):
            return np.load(io.BytesIO(read_bytes(p)), *args, **kwargs)
        return np.load(p, *args, **kwargs)
    return np.load(file, *args, **kwargs)


def h5_open(path, mode="r", *args, **kwargs):
    import h5py
    if isinstance(path, str):
        p = resolve(path)
        if is_obs(p):
            return h5py.File(io.BytesIO(read_bytes(p)), mode, *args, **kwargs)
        return h5py.File(p, mode, *args, **kwargs)
    return h5py.File(path, mode, *args, **kwargs)


def imageio_imread(path, *args, **kwargs):
    import imageio.v2 as imageio
    p = resolve(path) if isinstance(path, str) else path
    if not is_obs(p):
        return imageio.imread(p, *args, **kwargs)
    ext = os.path.splitext(p)[-1]
    with tempfile.NamedTemporaryFile(suffix=ext) as tmp:   # 部分格式（EXR）需要真实扩展名
        tmp.write(read_bytes(p))
        tmp.flush()
        return imageio.imread(tmp.name, *args, **kwargs)


def load_pickle(path):
    import pickle
    return pickle.loads(read_bytes(path))


def save_pickle(obj, path):
    import pickle
    p = resolve(path)
    if is_obs(p):
        _mox().file.write(p, pickle.dumps(obj), binary=True)
        return
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with builtins.open(p, "wb") as f:
        pickle.dump(obj, f)


def read_csv(path, *args, **kwargs):
    import pandas as pd
    if isinstance(path, str):
        p = resolve(path)
        if is_obs(p):
            return pd.read_csv(io.BytesIO(read_bytes(p)), *args, **kwargs)
        return pd.read_csv(p, *args, **kwargs)
    return pd.read_csv(path, *args, **kwargs)


def loadtxt(fname, *args, **kwargs):
    if isinstance(fname, str):
        p = resolve(fname)
        if is_obs(p):
            return np.loadtxt(io.StringIO(read_bytes(p).decode("utf-8")), *args, **kwargs)
        return np.loadtxt(p, *args, **kwargs)
    return np.loadtxt(fname, *args, **kwargs)


def fromfile(f, dtype=float, *args, **kwargs):
    """np.fromfile 需要真实文件句柄；obs 读出来的是 BytesIO，改用 frombuffer 读剩余字节。"""
    if isinstance(f, str):
        return np.frombuffer(read_bytes(f), dtype=dtype)
    if isinstance(f, (io.BytesIO, io.StringIO)):
        return np.frombuffer(f.read(), dtype=dtype)
    return np.fromfile(f, dtype, *args, **kwargs)
