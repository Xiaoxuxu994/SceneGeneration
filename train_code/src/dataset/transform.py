import numpy as np
import torch
from collections.abc import Sequence, Mapping
import random

from src.registry import Registry

TRANSFORMS = Registry("transforms")

class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict

@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data

@TRANSFORMS.register_module()
class DecodeQuerySample(object):
    def __init__(
        self,
        sample_rate=0.1,
        max_number = 100000,
        vs_list=[128,],
    ):
        '''
            currently only support single voxel size input
        '''
        self.sample_rate = sample_rate
        self.vs_list = vs_list # for voxel size prefix
        self.max_number = max_number
    def sample_points(self, points, labels, segment=None, mask=None):
        """
            Helper function to sample points and labels for both classes
        """

        # Split points and labels by their classes (0 and 1)
        points_0 = points[labels == 0]
        points_1 = points[labels == 1]

        # Handle segment data if it exists
        segment_0, segment_1 = None, None
        mask_0, mask_1 = None, None

        if segment is not None:
            segment_0 = segment[labels == 0]
            segment_1 = segment[labels == 1]

        if mask is not None:
            mask_0 = mask[labels == 0]
            mask_1 = mask[labels == 1]

        num_points_0, num_points_1 = len(points_0), len(points_1)

        # Deal with class 0
        indices_0 = torch.randperm(num_points_0)[:min(self.max_number, int(num_points_0 * self.sample_rate))] # use torch
        points_0_down = points_0[indices_0]
        labels_0_down = np.zeros(len(indices_0), dtype=np.float32)

        # Deal with class 1
        indices_1 = torch.randperm(num_points_1)[:min(self.max_number, int(num_points_1 * self.sample_rate))] # use torch
        points_1_down = points_1[indices_1]
        labels_1_down = np.ones(len(indices_1), dtype=np.float32)

        if segment is not None:
            segment_0_down = segment_0[indices_0]
            segment_1_down = segment_1[indices_1]

        else:
            segment_0_down = segment_1_down = None

        if mask is not None:
            mask_0_down = mask_0[indices_0]
            mask_1_down = mask_1[indices_1]
        else:
            mask_0_down = mask_1_down = None

        return (
            points_0_down, points_1_down,
            labels_0_down, labels_1_down,
            segment_0_down, segment_1_down,
            mask_0_down, mask_1_down
        )

    def shuffle_and_combine(self, points_0_down, points_1_down, labels_0_down, labels_1_down, segment_0_down=None, segment_1_down=None, mask_0=None, mask_1=None):
        """
        Helper function to combine and shuffle data for training or testing
        """

        # Combine and shuffle the sampled points, labels, and segments
        points_combined = np.concatenate((points_0_down, points_1_down), axis=0)
        labels_combined = np.concatenate((labels_0_down, labels_1_down), axis=0)

        if segment_0_down is not None and segment_1_down is not None:
            seg_combined = np.concatenate((segment_0_down, segment_1_down), axis=0)
        else:
            seg_combined = None

        if mask_0 is not None and mask_1 is not None:
            mask_combined = np.concatenate((mask_0, mask_1), axis=0)
        else:
            mask_combined = None

        # Shuffle the combined data
        indices = torch.randperm(len(points_combined))
        points_combined = points_combined[indices]
        labels_combined = labels_combined[indices]

        if seg_combined is not None:
            seg_combined = seg_combined[indices]
        if mask_combined is not None:
            mask_combined = mask_combined[indices]

        return points_combined, labels_combined,seg_combined,mask_combined
        
    def __call__(self, data_dict):

        # Extract the point cloud, labels, and optionally segments
        for vs in self.vs_list:
            points = data_dict[f'pc_query_{vs}']
            labels = data_dict[f'pc_query_label_{vs}']
            mask = data_dict.get(f'mask_gt_{vs}', None)
            segment = data_dict.get(f'pc_query_segment_{vs}', None)
            # If sample_rate is 1.0, return data_dict as is
            if self.sample_rate == 1.0:
                data_dict[f'pc_query_{vs}'] = points
                data_dict[f'pc_query_label_{vs}'] = labels
                if segment is not None:
                    data_dict[f'pc_query_segment_{vs}'] = segment
                if mask is not None:
                    data_dict[f'mask_gt_{vs}'] = mask
                return data_dict

            out = self.sample_points(points, labels, segment, mask)

            # Combine and shuffle
            points_comb, labels_comb, seg_comb, mask_comb = self.shuffle_and_combine(*out)

            if seg_comb is not None:
                data_dict[f'pc_query_segment_{vs}'] = seg_comb
            if mask_comb is not None:
                data_dict[f'mask_gt_{vs}'] = mask_comb

            # Update the data_dict with the sampled and shuffled points and labels
            data_dict[f'pc_query_{vs}'] = points_comb
            data_dict[f'pc_query_label_{vs}'] = labels_comb
                   
        return data_dict

@TRANSFORMS.register_module()
class DecodeQuerySampleSDF(object):
    def __init__(
        self,
        sample_rate=0.1,
        max_number = 100000,
        vs_list=[128,],
        near_thresh=0.05,
        near_ratio=0.8,
    ):
        '''
            currently only support single voxel size input
        '''
        self.sample_rate = sample_rate
        self.vs_list = vs_list # for voxel size prefix
        self.max_number = max_number
        self.near_thresh = near_thresh
        self.near_ratio = near_ratio

    def sample_points(self, points, labels, segment=None, mask=None):
        """
            Helper function to sample points and labels for both classes
        """

        # Split points and labels with threshold
        near_mask = labels <= self.near_thresh
        far_mask = labels > self.near_thresh

        points_near = points[near_mask]
        points_far = points[far_mask]
        labels_near = labels[near_mask]
        labels_far = labels[far_mask]

        # Handle segment data if it exists
        segment_near, segment_far = None, None
        mask_near, mask_far = None, None

        if segment is not None:
            segment_near = segment[near_mask]
            segment_far = segment[far_mask]

        if mask is not None:
            mask_near = mask[near_mask]
            mask_far = mask[far_mask]

        total_target = min(self.max_number, int(len(points) * self.sample_rate))
        near_target = int(total_target * self.near_ratio)
        far_target = total_target - near_target

        near_idx = torch.randperm(len(points_near))[:min(near_target, len(points_near))]
        far_idx = torch.randperm(len(points_far))[:min(far_target, len(points_far))]

        sampled_points = np.concatenate([points_near[near_idx], points_far[far_idx]], axis=0)
        sampled_label = np.concatenate([labels_near[near_idx], labels_far[far_idx]], axis=0)

        if segment is not None:
            sampled_segment = np.concatenate([segment_near[near_idx], segment_far[far_idx]], axis=0)
        else:
            sampled_segment = None

        if mask is not None:
            sampled_mask = np.concatenate([mask_near[near_idx], mask_far[far_idx]], axis=0)
        else:
            sampled_mask = None

        perm = torch.randperm(len(sampled_points))
        sampled_points = sampled_points[perm]
        sampled_label = sampled_label[perm]
        if sampled_segment is not None:
            sampled_segment = sampled_segment[perm]
        if sampled_mask is not None:
            sampled_mask = sampled_mask[perm]

        return sampled_points, sampled_label, sampled_segment, sampled_mask
    
        
    def __call__(self, data_dict):

        # Extract the point cloud, labels, and optionally segments
        for vs in self.vs_list:
            points = data_dict[f'pc_query_{vs}']
            labels = data_dict[f'pc_query_sdf_{vs}']
            mask = data_dict.get(f'mask_gt_{vs}', None)
            segment = data_dict.get(f'pc_query_segment_{vs}', None)
            # If sample_rate is 1.0, return data_dict as is
            if self.sample_rate == 1.0:
                data_dict[f'pc_query_{vs}'] = points
                data_dict[f'pc_query_sdf_{vs}'] = labels
                if segment is not None:
                    data_dict[f'pc_query_segment_{vs}'] = segment
                if mask is not None:
                    data_dict[f'mask_gt_{vs}'] = mask
                return data_dict

            sampled_points, sampled_label, sampled_segment, sampled_mask = self.sample_points(points, labels, segment, mask)

            if sampled_segment is not None:
                data_dict[f'pc_query_segment_{vs}'] = sampled_segment
            if sampled_mask is not None:
                data_dict[f'mask_gt_{vs}'] = sampled_mask

            # Update the data_dict with the sampled and shuffled points and labels
            data_dict[f'pc_query_{vs}'] = sampled_points
            data_dict[f'pc_query_sdf_{vs}'] = sampled_label
                   
        return data_dict

@TRANSFORMS.register_module()
class DecodeQueryRandSampleSDF(object):
    def __init__(
        self,
        sample_rate=0.1,
        max_number=100000,
        vs_list=[128,],
    ):
        """
        纯随机采样版本：对每个 vs 的点集，随机抽样
        min(max_number, sample_rate * N) 个样本。
        """
        self.sample_rate = float(sample_rate)
        self.vs_list = vs_list
        self.max_number = int(max_number)
        
    @staticmethod
    def _rand_indices(N, k, backend="np", device=None):
        """返回长度为 k 的随机不重复下标"""
        if N == 0 or k == 0:
            return None
        if backend == "torch":
            idx = torch.randperm(N, device=device)[:k]
            return idx
        else:
            # numpy
            return np.random.choice(N, size=k, replace=False)

    def _uniform_random_sample(self, points, labels, segment=None, mask=None):
        """
        统一随机采样（保持输入/输出类型一致；segment/mask 同步采样）
        """
        # 检查后端与形状
        is_torch = isinstance(points, torch.Tensor)
        N = points.shape[0]
        if N == 0:
            return points, labels, segment, mask

        # 目标采样数
        k = min(self.max_number, int(np.ceil(N * self.sample_rate)))
        k = max(1, k) if N > 0 else 0
        if self.sample_rate >= 1.0 or k >= N:
            # 等价于不采样（直接返回）
            return points, labels, segment, mask

        # 生成随机下标
        if is_torch:
            device = points.device
            idx = self._rand_indices(N, k, backend="torch", device=device)
            # 索引
            pts_s = points.index_select(0, idx)
            lbl_s = labels.index_select(0, idx)
            seg_s = segment.index_select(0, idx) if segment is not None else None
            msk_s = mask.index_select(0, idx) if mask is not None else None
        else:
            idx = self._rand_indices(N, k, backend="np")
            pts_s = points[idx]
            lbl_s = labels[idx]
            seg_s = segment[idx] if segment is not None else None
            msk_s = mask[idx] if mask is not None else None

        return pts_s, lbl_s, seg_s, msk_s

    def __call__(self, data_dict):
        """
        期望 data_dict 中包含：
          - pc_query_{vs}        : (N, D)
          - pc_query_sdf_{vs}    : (N,) 或 (N,1)
          - 可选 pc_query_segment_{vs} / mask_gt_{vs}
        """
        for vs in self.vs_list:
            key_pts = f'pc_query_{vs}'
            key_lbl = f'pc_query_sdf_{vs}'
            key_seg = f'pc_query_segment_{vs}'
            key_msk = f'mask_gt_{vs}'

            if key_pts not in data_dict or key_lbl not in data_dict:
                continue  # 缺关键字段则跳过该 vs

            points  = data_dict[key_pts]
            labels  = data_dict[key_lbl]
            segment = data_dict.get(key_seg, None)
            mask    = data_dict.get(key_msk, None)

            # 统一 labels 维度为 (N,)
            if isinstance(labels, np.ndarray):
                if labels.ndim == 1:
                    labels = labels[:, None]
            elif isinstance(labels, torch.Tensor):
                if labels.dim() == 1:
                    labels = labels.unsqueeze(-1)

            # sample_rate==1.0 直接返回原数据
            if self.sample_rate >= 1.0:
                data_dict[key_pts] = points
                data_dict[key_lbl] = labels
                if segment is not None:
                    data_dict[key_seg] = segment
                if mask is not None:
                    data_dict[key_msk] = mask
                continue

            # 随机采样
            pts_s, lbl_s, seg_s, msk_s = self._uniform_random_sample(points, labels, segment, mask)

            # 写回
            data_dict[key_pts] = pts_s
            data_dict[key_lbl] = lbl_s
            if seg_s is not None:
                data_dict[key_seg] = seg_s
            if msk_s is not None:
                data_dict[key_msk] = msk_s

        return data_dict
                            
@TRANSFORMS.register_module()
class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "normal", "segment"),
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                ##### !!!!TODO change type outside!!!!
                # data_dict["grid_coord"] = grid_coord[idx_unique]
                data_dict["grid_coord"] = grid_coord[idx_unique].astype(np.float32)
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                if self.return_inverse:
                    data_dict["inverse"] = np.zeros_like(inverse)
                    data_dict["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

@TRANSFORMS.register_module()
class GridSampleV2(object):
    def __init__(
        self,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "normal", "segment"),
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict, grid_size):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                if self.return_inverse:
                    data_dict["inverse"] = np.zeros_like(inverse)
                    data_dict["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr
        
@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class ShufflePoint(object):
    def __call__(self, data_dict,vs_list=[128]):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"][shuffle_index]
        if "grid_coord" in data_dict.keys():
            data_dict["grid_coord"] = data_dict["grid_coord"][shuffle_index]
        if "displacement" in data_dict.keys():
            data_dict["displacement"] = data_dict["displacement"][shuffle_index]
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"][shuffle_index]
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"][shuffle_index]
        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"][shuffle_index]
        
        ## for decoder shuffle seperately
        ## TODO to check the multi level input
        for vs in vs_list:
            shuffle_index = np.arange(data_dict[f"pc_query_{vs}"].shape[0])
            np.random.shuffle(shuffle_index)
            data_dict[f"pc_query_{vs}"] = data_dict[f"pc_query_{vs}"][shuffle_index]
            data_dict[f"pc_query_grid_{vs}"] = data_dict[f"pc_query_grid_{vs}"][shuffle_index]

            # for data
            if vs == vs_list[-1]:
                data_dict[f"pc_query_label_{vs}"] = data_dict[f"pc_query_label_{vs}"][shuffle_index]
            else:
                data_dict[f"pc_query_label_fine_prop_{vs}"] = data_dict[f"pc_query_label_fine_prop_{vs}"][shuffle_index]

        return data_dict

@TRANSFORMS.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx]
        return data_dict

@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))

        if "pc_query" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["pc_query"].min(axis=0)
                x_max, y_max, z_max = data_dict["pc_query"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["pc_query"] -= center
            data_dict["pc_query"] = np.dot(data_dict["pc_query"], np.transpose(rot_t))
            data_dict["pc_query"] += center
        return data_dict

@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False,vs_list=[128]):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic
        self.vs_list = vs_list

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            ).astype(data_dict["coord"].dtype)
            data_dict["coord"] *= scale
            for vs in self.vs_list:
                data_dict[f'pc_query_{vs}'] *= scale
        return data_dict

@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5, vs_list=[128]):
        self.p = p
        self.vs_list = vs_list

    def apply_flip_to_mask(self, mask, grid, vs, flip_x=False, flip_y=False):
        active_grid = grid[mask]

        if flip_x:
            grid[:, 0] = vs - 1 - grid[:, 0]
            active_grid[:, 0] = vs - 1 - active_grid[:, 0]
        if flip_y:
            grid[:, 1] = vs - 1 - grid[:, 1]
            active_grid[:, 1] = vs - 1 - active_grid[:, 1]

        def hash_grid(coords):
            return coords[:, 0] * vs * vs + coords[:, 1] * vs + coords[:, 2]

        full_hash = hash_grid(grid)
        flipped_hash = hash_grid(active_grid)

        new_mask = np.isin(full_hash, flipped_hash)

        return new_mask


    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
            for vs in self.vs_list:
                data_dict[f"pc_query_{vs}"][:, 0] = -data_dict[f"pc_query_{vs}"][:, 0]
                ## for debug
                mask = data_dict[f"mask_gt_{vs}"]

                grid = data_dict[f"pc_query_grid_{vs}"]
                data_dict[f"mask_gt_{vs}"] = self.apply_flip_to_mask(mask, grid, vs, flip_x=True, flip_y=False)
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
            for vs in self.vs_list:
                data_dict[f"pc_query_{vs}"][:, 1] = -data_dict[f"pc_query_{vs}"][:, 1]
                ## for debug
                mask = data_dict[f"mask_gt_{vs}"]
                grid = data_dict[f"pc_query_grid_{vs}"]
                data_dict[f"mask_gt_{vs}"] = self.apply_flip_to_mask(mask, grid, vs, flip_x=False, flip_y=True)
        return data_dict
                

@TRANSFORMS.register_module()
class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if self.mode == "all":
            # TODO: Optimize
            if "index" not in data_dict.keys():
                data_dict["index"] = np.arange(data_dict["coord"].shape[0])
            data_part_list = []
            # coord_list, color_list, dist2_list, idx_list, offset_list = [], [], [], [], []
            if data_dict["coord"].shape[0] > point_max:
                coord_p, idx_uni = np.random.rand(
                    data_dict["coord"].shape[0]
                ) * 1e-3, np.array([])
                while idx_uni.size != data_dict["index"].shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist2 = np.sum(
                        np.power(data_dict["coord"] - data_dict["coord"][init_idx], 2),
                        1,
                    )
                    idx_crop = np.argsort(dist2)[:point_max]

                    data_crop_dict = dict()
                    if "coord" in data_dict.keys():
                        data_crop_dict["coord"] = data_dict["coord"][idx_crop]
                    if "grid_coord" in data_dict.keys():
                        data_crop_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
                    if "normal" in data_dict.keys():
                        data_crop_dict["normal"] = data_dict["normal"][idx_crop]
                    if "color" in data_dict.keys():
                        data_crop_dict["color"] = data_dict["color"][idx_crop]
                    if "displacement" in data_dict.keys():
                        data_crop_dict["displacement"] = data_dict["displacement"][
                            idx_crop
                        ]
                    if "strength" in data_dict.keys():
                        data_crop_dict["strength"] = data_dict["strength"][idx_crop]
                    data_crop_dict["weight"] = dist2[idx_crop]
                    data_crop_dict["index"] = data_dict["index"][idx_crop]
                    data_part_list.append(data_crop_dict)

                    delta = np.square(
                        1 - data_crop_dict["weight"] / np.max(data_crop_dict["weight"])
                    )
                    coord_p[idx_crop] += delta
                    idx_uni = np.unique(
                        np.concatenate((idx_uni, data_crop_dict["index"]))
                    )
            else:
                data_crop_dict = data_dict.copy()
                data_crop_dict["weight"] = np.zeros(data_dict["coord"].shape[0])
                data_crop_dict["index"] = data_dict["index"]
                data_part_list.append(data_crop_dict)
            return data_part_list
        # mode is "random" or "center"
        elif data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    np.random.randint(data_dict["coord"].shape[0])
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "origin_coord" in data_dict.keys():
                data_dict["origin_coord"] = data_dict["origin_coord"][idx_crop]
            if "grid_coord" in data_dict.keys():
                data_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx_crop]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx_crop]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx_crop]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx_crop]
            if "displacement" in data_dict.keys():
                data_dict["displacement"] = data_dict["displacement"][idx_crop]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx_crop]
        return data_dict


class UniformDownSample(object):
    def __init__(self, 
                 point_max=80000, 
                 sample_rate=None, 
                 random=False):
        """
        Args:
            point_max: 最多保留的点数（当 sample_rate 为 None 时生效）
            sample_rate: 下采样比例（优先级高于 point_max）
            random: 是否随机选择采样点（True 为随机下采样，False 为等间隔）
        """
        self.point_max = point_max
        self.sample_rate = sample_rate
        self.random = random

    def __call__(self, data_dict):
        assert "coord" in data_dict
        N = data_dict["coord"].shape[0]

        # 计算采样点数
        num_sample = (
            int(self.sample_rate * N)
            if self.sample_rate is not None
            else min(self.point_max, N)
        )

        if N <= num_sample:
            return data_dict  # 不需要采样

        if self.random:
            idx_sample = np.random.choice(N, num_sample, replace=False)
        else:
            idx_sample = np.linspace(0, N - 1, num=num_sample, dtype=int)

        data_dict["coord"] = data_dict["coord"][idx_sample]
        return data_dict