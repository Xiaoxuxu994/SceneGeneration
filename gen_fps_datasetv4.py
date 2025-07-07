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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--npuid", default='0', type=int) # 单卡指定卡号
    parser.add_argument("--path", default='./', type=str) # 原始点云数据集,假设共N个
    parser.add_argument("--outpath", default='./', type=str) # 采样输出路径
    parser.add_argument("--startid", default='0', type=int) # 原始点云数据集的采样开始下标, 0<=startid<N
    parser.add_argument("--endid", default='0', type=int) # 原始点云数据集的采样结束下标, 0<endid<=N
    parser.add_argument("--seed", default='42', type=int) # 随机化种子
    parser.add_argument("--grid_size", default='0.05', type=float) # 点云网格分辨率
    parser.add_argument("--fps_num", default=2048, type=int)
    args = parser.parse_args()
    device = 'npu:' + str(args.npuid)
    file = sorted(os.listdir(args.path))
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # deal with
    for i, fn in tqdm(enumerate(file), total=len(file), desc='processing fps'):
        if i >= args.startid and i < args.endid:
            data = torch.load(os.path.join(args.path, fn))
            pcd = torch.from_numpy(data['coord']).to(device)
            scaled_coord = data["coord"] / np.array(args.grid_size)
            grid_coord = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord.min(0)
            grid_coord -= min_coord
            # fps 采样
            index, fps = farthest_point_sampling_npu(pcd, device, args.fps_num)
            grid = grid_coord[index]
            sample = {'fps_coord': fps, 'grid_coord':grid}
            torch.save(sample, os.path.join(args.outpath, fn))
            torch_npu.npu.empty_cache()