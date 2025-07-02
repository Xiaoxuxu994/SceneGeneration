import torch
import torch_npu
import argparse
import os
from tqdm import tqdm
import numpy as np

def farthest_point_sampling_npu(points, device, num_samples):
    sampled_indices = [torch.randint(len(points), (1,)).item()]
    distances = torch.full((len(points),), float('inf')).to(device)
    for _ in range(num_samples - 1):
        last_sampled = points[sampled_indices[-1]]
        dist = torch.norm(points - last_sampled, dim=1)
        distances = torch.minimum(distances, dist)
        sampled_indices.append(torch.argmax(distances).item())

    return sampled_indices, points[sampled_indices]

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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--npuid", default='0', type=int)  # 单卡指定卡号
    parser.add_argument("--path", default='./', type=str)  # 原始点云数据集
    parser.add_argument("--outpath", default='./', type=str) # 采样输出路径
    parser.add_argument("--startid", default='0', type=int) # 原始点云数据集的采样开始下标, 0<=startid<N 
    parser.add_argument("--endid", default='0', type=int) # 原始点云数据集的采样结束下标, 0<endid<=N
    parser.add_argument("--seed", default='42', type=int) # 随机化种子
    parser.add_argument("--grid_size", default='0.05', type=float) # 点云网格分辨率
    parser.add_argument("--epoch_num", default='50', type=int) # epoch数
    parser.add_argument("--hash_type", default='fnv', type=str)
    parser.add_argument("--fps_num", default=2048, type=int)
    args = parser.parse_args()
    hash = fnv_hash_vec if args.hash_type == "fnv" else ravel_hash_vec
    # mkdir all epochs dir
    epoch_name = [str(i).zfill(6) for i in range(args.epoch_num)]
    for _, fn in enumerate(epoch_name):
        os.makedirs(os.path.join(args.outpath, fn), exist_ok=True)
    device = 'npu:' + str(args.npuid)
    file = sorted(os.listdir(args.path))
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # deal with
    for i, fn in tqdm(enumerate(file), total=len(file), desc='processing fps'):
        if i >= args.startid and i < args.endid:
            data = torch.load(os.path.join(args.path, fn))
            scaled_coord = data['coord'] / np.array(args.grid_size)
            grid_coord = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord.min(0)
            grid_coord -= min_coord
            scaled_coord -= min_coord
            min_coord = min_coord * np.array(args.grid_size)
            key = hash(grid_coord)
            idx_sort = np.argsort(key)
            key_sort = key[idx_sort]
            _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
            for j in range(args.epoch_num):
                # 给定 grid_size 采样         
                idx_select = (
                    np.cumsum(np.insert(count, 0, 0)[0:-1])
                    + np.random.randint(0, count.max(), count.size) % count
                )
                idx_unique = idx_sort[idx_select]
                pcd = torch.from_numpy(data['coord'][idx_unique]).to(device)
                # fps 采样
                fps_select, fps_pcd = farthest_point_sampling_npu(pcd, device, args.fps_num)
                fps_res = {"idx_select": idx_select, "fps_select": fps_select}
                torch.save(fps_res, os.path.join(args.outpath, epoch_name[j], fn))
                torch_npu.npu.empty_cache()