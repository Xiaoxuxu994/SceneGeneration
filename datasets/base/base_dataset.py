from datasets.base.easy_dataset import EasyDataset
from pi3.utils.geometry import depthmap_to_absolute_camera_coordinates
import numpy as np
import os
import PIL
import pi3.utils.cropping as cropping
import torchvision.transforms as tvf
from omegaconf import OmegaConf
from .transforms import *
import pandas as pd
from .utils import *
import pickle
from datasets.sample_utils.common import set_depth_filter
from datasets.base import fileio as fio

# 旧版各 loader 写死的缓存目录；没有传 cache_dir、也没有设环境变量 DATASET_CACHE_DIR 时沿用它
LEGACY_CACHE_DIR = "/efs-mi-east4-2/z00533370/proj_3dgs/3drecon/data/dataset_cache"


class _LowParallax(Exception):
    """所选帧的基线/深度低于 min_baseline_ratio：对同一条序列重新选帧。"""


# ---------------------------------------------------------------------------
# 样本内多帧并发读取
# ---------------------------------------------------------------------------
# 读 OBS 的耗时主要是每次请求的延迟（约 0.3s/次），各 loader 的 _get_views 都是逐帧串行调用 _load_one_view，
# 8 帧的延迟叠加起来，一个样本 5–15s（tools/profile_data.py 实测 OBS 占 70–90%）。
# 环境变量 DATA_VIEW_THREADS=N（>1）时，_get_views 里对 _load_one_view 的调用改为提交到 N 个线程并发执行，
# 返回值先用占位对象代替，_get_views 结束后再按原顺序取回结果。所有 loader 都是
#   for ...: views.append(self._load_one_view(..., rng=rng)); return views
# 这种写法，不需要改 loader。每帧的 rng 由主线程按顺序从样本 rng 派生（结果可复现，线程间不共享 Generator）。
DATA_VIEW_THREADS = int(os.environ.get("DATA_VIEW_THREADS", "1") or 1)


class _PendingView:
    __slots__ = ("future",)

    def __init__(self, future):
        self.future = future


def baseline_over_depth(views):
    """相机中心两两距离的中位数 ÷ 各帧有效深度中位数的中位数（与 tools/quality_eval 的定义一致）。"""
    centers = np.stack([np.asarray(v['camera_pose'])[:3, 3] for v in views])
    if len(centers) < 2:
        return float('inf')
    d = np.linalg.norm(centers[:, None] - centers[None], axis=-1)[np.triu_indices(len(centers), 1)]
    meds = [np.median(v['depthmap'][v['valid_mask']]) for v in views if v['valid_mask'].any()]
    if not meds or np.median(meds) <= 0:
        return float('inf')
    return float(np.median(d) / np.median(meds))


class BaseDataset(EasyDataset):
    def __init__(
        self,
        seed=2024,
        resolution=None,            # (width, height) or list of (width, height) or list of int
        aug_crop=False,             # False or int, slightly scale the image a bit larger than the target resolution
        aug_focal=False,            # False or float in [0, 1]
        z_far=0,
        frame_num=2,
        transform=tvf.ToTensor(),
        mode='train',
        # max_refetch=3,
        max_distance=8,
        random_sample_thres=0.0,
        stride_sample_thres=0.0,
        stride_candidates=None,
        force_global_sampling=False,
        shuffle=True,
        use_sparse_depth=False,
        # ---- 数据过滤开关（默认全部关闭，行为与旧版一致）----
        pct_valid_only=False,       # 98 分位只在有效像素上算（修复把无效像素 0 算进分位数）
        pct_disable=False,          # 关掉分位截断，只保留 max_depth / z_far（物体类、合成数据）
        use_dataset_masks=False,    # 把 loader 返回的 extra_valid（自带 mask / confidence）并入 valid_mask
        min_baseline_ratio=0.0,     # > 0 时要求所选帧的 基线/深度 ≥ 该值，不满足则对同一序列重新选帧
        baseline_retries=3,         # 重新选帧的次数上限，用完仍不满足则照常使用
        seq_blacklist=None,         # 序列黑名单：txt 路径（每行一个序列名）或列表
        # ---- 数据位置（本地或 obs://，见 datasets/base/fileio.py）----
        cache_dir=None,             # 索引缓存 pkl 的目录；默认环境变量 DATASET_CACHE_DIR，再默认旧的写死目录
        cache_file=None,            # 直接指定缓存 pkl 的完整路径（优先于 cache_dir）
        path_remap=None,            # {本地前缀: obs 前缀}，缓存里存的是本地绝对路径时用它改读 OBS
    ):
        super().__init__()
        self.cache_dir = cache_dir or os.environ.get("DATASET_CACHE_DIR") or LEGACY_CACHE_DIR
        self.cache_file = cache_file
        if OmegaConf.is_config(path_remap):
            path_remap = OmegaConf.to_container(path_remap, resolve=True)
        self.path_remap = dict(path_remap) if path_remap else None
        fio.set_path_remap(self.path_remap)
        self.pct_valid_only = pct_valid_only
        self.pct_disable = pct_disable
        self.use_dataset_masks = use_dataset_masks
        self.min_baseline_ratio = float(min_baseline_ratio or 0.0)
        self.baseline_retries = int(baseline_retries)
        self.seq_blacklist = seq_blacklist
        self._n_load = 0
        self._n_fail = 0
        self._n_low_parallax = 0
        self.frame_num = frame_num
        self.max_distance = max_distance

        self.transform = transform

        self.shuffle = shuffle

        self.use_sparse_depth = use_sparse_depth

        self._rng = np.random.default_rng(seed)
        self._set_resolutions(resolution)

        self.aug_crop = aug_crop
        self.aug_focal = aug_focal

        self.z_far = z_far

        self.dataset_label = 'BaseDataset'

        self.mode = mode
        # self.max_refetch = max_refetch

        self.random_sample_thres = random_sample_thres
        self.stride_sample_thres = stride_sample_thres
        self.stride_candidates = stride_candidates
        self.force_global_sampling = force_global_sampling

    _KEEP_PY_ATTRS = ("path_remap", "seq_blacklist")

    def convert_attributes(self):
        """
        Avoid memory leak caused by python list or python dict
        https://github.com/pytorch/pytorch/issues/13246
        """
        self._apply_blacklist()

        def _is_equivalent(original, converted):
            """
            Check if the converted data structure is equivalent to the original.
            """
            try:
                return original == converted
            except Exception:
                return False

        for attr_name in dir(self):
            if attr_name.startswith("__") or callable(getattr(self, attr_name)):
                continue
            if attr_name in self._KEEP_PY_ATTRS:     # 配置项保持原样（path_remap 转成 pd.Series 后 `if not remap` 会报错）
                continue
            
            attr_value = getattr(self, attr_name)
            
            if isinstance(attr_value, list):
                try:
                    converted_value = np.array(attr_value)
                    
                    if _is_equivalent(attr_value, converted_value.tolist()):
                        setattr(self, attr_name, converted_value)
                    else:
                        print(f"[{self.dataset_label}] <{attr_name}> conversion may not be equivalent, skipping.", flush=True)
                except ValueError as e:
                    print(f"[{self.dataset_label}] Error converting <{attr_name}>: {e}", flush=True)
            
            elif isinstance(attr_value, dict):
                try:
                    converted_value = pd.Series(attr_value)
                    if _is_equivalent(attr_value, converted_value.to_dict()):
                        setattr(self, attr_name, converted_value)
                    else:
                        print(f"[{self.dataset_label}] <{attr_name}> conversion may not be equivalent, skipping.", flush=True)
                except ValueError as e:
                    print(f"[{self.dataset_label}] Error converting <{attr_name}>: {e}", flush=True)

    def _apply_blacklist(self):
        """按 seq_blacklist 去掉序列。名字与序列 key、scene_to_path 的值或它们的末级目录名任一相同即剔除
        （quality_eval 输出的 label 一般就是其中之一）。在 datasets/__init__.py 实例化后的 convert_attributes 里调用。"""
        bl = self.seq_blacklist
        if not bl or not hasattr(self, 'sequences'):
            return
        if isinstance(bl, str):
            with fio.open(bl) as f:
                names = {l.strip() for l in f if l.strip() and not l.startswith('#')}
        else:
            names = {str(x) for x in bl}
        s2p = getattr(self, 'scene_to_path', {})

        def keys(seq):
            ks = {str(seq), os.path.basename(str(seq))}
            try:
                p = str(s2p[seq])
                ks |= {p, os.path.basename(p.rstrip('/'))}
            except Exception:
                pass
            return ks

        before = len(self.sequences)
        self.sequences = [s for s in list(self.sequences) if not (keys(s) & names)]
        print(f"[{self.dataset_label}] seq_blacklist: {before} -> {len(self.sequences)} sequences", flush=True)

    def _view_pool(self):
        """每个进程（dataloader worker）各自一个线程池；fork 之后线程不会被继承，按 pid 重建。"""
        pool = getattr(self, "_view_pool_obj", None)
        if pool is None or getattr(self, "_view_pool_pid", None) != os.getpid():
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(max_workers=DATA_VIEW_THREADS, thread_name_prefix="view")
            self._view_pool_obj, self._view_pool_pid = pool, os.getpid()
        return pool

    def _get_views_maybe_parallel(self, idx, resolution, rng):
        if DATA_VIEW_THREADS <= 1 or not hasattr(self, "_load_one_view"):
            return self._get_views(idx, resolution, rng)
        pool = self._view_pool()
        serial = self._load_one_view            # 绑定方法

        def child(r):
            return np.random.default_rng(int(r.integers(0, 2 ** 63 - 1))) if isinstance(r, np.random.Generator) else r

        def deferred(*args, **kwargs):
            args = tuple(child(a) for a in args)
            kwargs = {k: child(v) for k, v in kwargs.items()}
            return _PendingView(pool.submit(serial, *args, **kwargs))

        self._load_one_view = deferred           # 实例属性遮住类方法，只在这次 _get_views 内生效
        try:
            views = self._get_views(idx, resolution, rng)
        finally:
            del self._load_one_view
        return [v.future.result() if isinstance(v, _PendingView) else v for v in views]

    def _sample_frame_ids(self, frame_ids, rng):
        """
        Mixed frame sampler:
          1. global random sampling: long-baseline augmentation
          2. stride-like sampling: mimic dataset-specific eval stride
          3. local window sampling: stable high-overlap samples

        Returned frame ids are sorted by temporal order.
        """
        mode = str(getattr(self, "mode", "train")).lower()
        is_train = mode == "train"
        n = len(frame_ids)
        assert n > 0

        f = self.frame_num
        assert f > 0

        # ------------------------------------------------------------
        # Case 0: sequence too short
        # ------------------------------------------------------------
        if n <= f:
            if n == f:
                return [frame_ids[i] for i in range(n)]

            if is_train:
                pos = rng.choice(n, size=f, replace=True)
                pos = sorted([int(i) for i in pos])
            else:
                repeat = int(np.ceil(f / n))
                pos = (list(range(n)) * repeat)[:f]
                pos = sorted([int(i) for i in pos])

            return [frame_ids[i] for i in pos]

        # ------------------------------------------------------------
        # Probabilities
        # ------------------------------------------------------------
        global_prob = float(getattr(self, "random_sample_thres", 0.0))
        stride_prob = float(getattr(self, "stride_sample_thres", 0.0))
        force_global = bool(getattr(self, "force_global_sampling", False))

        stride_candidates = getattr(self, "stride_candidates", None)
        if stride_candidates is None:
            stride_candidates = []

        # OmegaConf ListConfig -> normal list
        try:
            stride_candidates = list(stride_candidates)
        except TypeError:
            stride_candidates = []

        r = rng.random()

        # ------------------------------------------------------------
        # Branch 1: global sampling
        #   - train: controlled by global_prob
        #   - val/test: if force_global_sampling=True, use deterministic global sampling
        # ------------------------------------------------------------
        if (is_train and r < global_prob) or force_global:
            if is_train:
                pos = rng.choice(n, size=f, replace=False)
                pos = sorted([int(i) for i in pos])
            else:
                pos = np.linspace(0, n - 1, f)
                pos = np.round(pos).astype(np.int64)
                pos = sorted([int(i) for i in pos])

            return [frame_ids[i] for i in pos]

        # ------------------------------------------------------------
        # Branch 2: dataset-specific stride-like sampling
        # ------------------------------------------------------------
        if is_train and r < global_prob + stride_prob and len(stride_candidates) > 0:
            valid_strides = [
                int(s) for s in stride_candidates
                if n >= int(s) * (f - 1) + 1
            ]

            if len(valid_strides) > 0:
                stride = int(rng.choice(valid_strides))

                max_start = n - 1 - stride * (f - 1)
                start = int(rng.integers(0, max_start + 1))

                # base positions: start, start+stride, ..., start+stride*(f-1)
                pos = start + np.arange(f) * stride

                # jitter proportional to stride, but capped
                # stride=25 -> about 2
                # stride=100 -> about 5
                jitter_radius = max(2, min(5, int(round(stride * 0.05))))
                jitter = rng.integers(-jitter_radius, jitter_radius + 1, size=f)

                pos = np.clip(pos + jitter, 0, n - 1)
                pos = sorted(set([int(i) for i in pos]))

                # jitter may create duplicates; fill missing frames locally
                if len(pos) < f:
                    need = f - len(pos)

                    span_start = max(0, start - stride)
                    span_end = min(n, start + stride * (f - 1) + stride + 1)

                    candidates = [
                        i for i in range(span_start, span_end)
                        if i not in pos
                    ]

                    if len(candidates) >= need:
                        extra = rng.choice(candidates, size=need, replace=False)
                    else:
                        all_candidates = [
                            i for i in range(n)
                            if i not in pos
                        ]

                        extra = rng.choice(
                            all_candidates,
                            size=need,
                            replace=(len(all_candidates) < need),
                        )

                    pos = sorted(pos + [int(i) for i in extra])

                # safety: if somehow more than f frames, uniformly keep f
                if len(pos) > f:
                    keep = np.linspace(0, len(pos) - 1, f).round().astype(int)
                    pos = [pos[int(i)] for i in keep]

                return [frame_ids[i] for i in pos]

            # If sequence is too short for stride-like sampling,
            # fall through to local window sampling.

        # ------------------------------------------------------------
        # Branch 3: local window sampling
        # ------------------------------------------------------------
        ref = int(rng.integers(0, n))

        max_dist = int(self.max_distance / 8 * f)

        start = max(0, ref - max_dist)
        end = min(n, ref + max_dist + 1)

        pool = np.arange(start, end)
        pool = pool[pool != ref]
        
        extra = rng.choice(
            pool,
            size=f - 1,
            replace=(len(pool) < f - 1),
        )

        pos = [ref] + [int(i) for i in extra]
        pos = sorted(pos)

        return [frame_ids[int(i)] for i in pos]
    
    def _set_resolutions(self, resolutions):
        assert resolutions is not None, 'undefined resolution'
        if OmegaConf.is_config(resolutions):
            resolutions = OmegaConf.to_object(resolutions)

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            # assert width >= height
            # self._resolutions.append((width, height))
            self._resolutions.append([width, height])

        self.num_resoluions = len(self._resolutions)

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None, normal=None, far_mask=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal, far_mask=far_mask)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        # NOTE: Here we don't care about portrait image.
        # assert resolution[0] >= resolution[1]
        # if H > 1.1*W:
        #     # image is portrait mode
        #     resolution = resolution[::-1]
        # elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
        #     # image is square, so we chose (portrait, landscape) randomly
        #     if rng.integers(2):
        #         resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5) # beta distribution, bi-modal
            image, depthmap, intrinsics, normal, far_mask = cropping.center_crop_image_depthmap(image, depthmap, intrinsics, crop_scale, normal=normal, far_mask=far_mask)

        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, intrinsics, normal, far_mask = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, normal=normal, far_mask=far_mask) # slightly scale the image a bit larger than the target resolution
        # image, depthmap, intrinsics, normal, far_mask = cropping.rescale_image_depthmap(image, None, intrinsics, target_resolution, normal=normal, far_mask=far_mask) # slightly scale the image a bit larger than the target resolution

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal, far_mask=far_mask)
        # image, depthmap, intrinsics2, normal, far_mask = cropping.crop_image_depthmap(image, None, intrinsics, crop_bbox, normal=normal, far_mask=far_mask)

        other = [x for x in [normal, far_mask] if x is not None]
        return image, depthmap, intrinsics2, *other
    
    def __getitem__(self, idx):
        mode = str(getattr(self, "mode", "train")).lower()
        is_train = mode == "train"
        max_retry = 10 if is_train else 100

        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            if len(idx) == 3:
                idx, ar_idx, frame_num = idx
                self.frame_num = frame_num
            else:
                idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
        
        idx = int(idx)
        origin_idx = idx
        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        
        error = None
        views = None
        low_parallax_tries = 0
        for retry in range(max_retry):              # default: 3
            try:
                if hasattr(self, "this_views_info"):
                    delattr(self, "this_views_info")
                set_depth_filter(self.pct_valid_only, self.pct_disable)
                fio.set_path_remap(getattr(self, "path_remap", None))
                self._n_load += 1
                views = self._get_views_maybe_parallel(idx, resolution, self._rng)

                # assert len(views) == self.frame_num
                if self.shuffle and is_train:
                    self._rng.shuffle(views)

                # check data-types
                for v, view in enumerate(views):
                    assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
                    view['idx'] = (idx, ar_idx, v)
                    # view['idx'] = (idx, v)

                    # encode the image
                    width, height = view['img'].size
                    view['true_shape'] = np.int32((height, width))

                    assert 'camera_intrinsics' in view
                    if 'camera_pose' not in view:
                        view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
                    else:
                        assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
                    assert 'pts3d' not in view
                    assert 'valid_mask' not in view
                    assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
                    # loader 可返回 extra_valid（自带 mask / confidence，已与深度同尺寸），只在 use_dataset_masks 时生效
                    extra_valid = view.pop('extra_valid', None)
                    view['z_far'] = self.z_far
                    pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

                    view['pts3d'] = pts3d
                    view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
                    if self.use_dataset_masks and extra_valid is not None:
                        view['valid_mask'] &= np.asarray(extra_valid).astype(bool)

                    view['depthmap'][~view['valid_mask']] = 0.0

                    assert view['valid_mask'].sum() > 0, f"[{self.dataset_label}] empty valid_mask for view {view_name(view)}"

                    if 'normal' not in view:
                        view['normal'] = None

                if self.min_baseline_ratio > 0 and low_parallax_tries < self.baseline_retries:
                    if baseline_over_depth(views) < self.min_baseline_ratio:
                        low_parallax_tries += 1
                        raise _LowParallax()

                    # # check all datatypes
                    # for key, val in view.items():
                    #     res, err_msg = is_good_type(key, val)
                    #     assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
                    # K = view['camera_intrinsics']

                for view in views:
                    view['img'] = self.transform(view['img'])

                # # last thing done!
                # for view in views:
                #     # transpose to make sure all views are the same size
                #     # transpose_to_landscape(view)  # NOTE: Here we don't care about portrait image.
                #     # this allows to check whether the RNG is is the same state each time
                #     view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')

                # overlap = self.check_overlap(views)

                # if overlap is False:
                #     raise ValueError("Views are not overlapped!")

            except _LowParallax:
                # 视差不足：同一条序列重新选帧，不换 idx、不算加载失败
                views = None
                self._n_low_parallax += 1
                continue
            except Exception as e:
                views = None
                self._n_fail += 1
                if self._n_fail % 200 == 0:
                    print(f"[{self.dataset_label}] load failures so far: {self._n_fail}/{self._n_load} "
                          f"(this worker; failed samples are replaced by random ones)", flush=True)
                if hasattr(self, 'this_views_info'):
                    print(
                        f"Failed to load data from {self.dataset_label}-{idx} ({self.this_views_info}) for error {e}.", flush=True
                    )
                else:
                    print(
                        f"Failed to load data from {self.dataset_label}-{idx} for error {e}.", flush=True
                    )
                
                if is_train:
                    idx = int(self._rng.integers(0, len(self)))
                else:
                    # val 确定性 fallback，避免某个 rank 随机 retry 太久
                    idx = int((origin_idx + retry + 1) % len(self))
                error = e
            
            if views is not None:
                error = None
                break

        if views is None:
            raise error if error is not None else RuntimeError(f"[{self.dataset_label}] no valid sample for idx {origin_idx}")
        
        return views
    
    def _load_cache(self, cache_file):
        try:
            data = fio.load_pickle(cache_file)

            if not isinstance(data, dict):
                print("Error: cache file does not contain a dictionary.")
                return False

            for key, value in data.items():
                setattr(self, key, value)
            return True
        except Exception as e:
            print(f"An error occurred while loading the cache: {e}")
            return False

    def _save_cache(self, keys, save_path):
        save_dict = {key: getattr(self, key) for key in keys}
        fio.save_pickle(save_dict, save_path)

        print(f"Saved cache to {save_path}.", flush=True)

