import torch
import torch_npu
import argparse
import os
from tqdm import tqdm
import numpy as np
from util import sample_farthest_points

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
    parser.add_argument("--npuid", default='0', type=int)
    parser.add_argument("--startid", default='0', type=int)
    parser.add_argument("--endid", default='0', type=int)
    parser.add_argument("--path", default='./', type=str)
    parser.add_argument("--outpath", default='./', type=str)
    parser.add_argument("--seed", default='42', type=int)
    parser.add_argument("--grid_size", default='0.05', type=float)
    parser.add_argument("--fps_num", default=2048, type=int)
    args = parser.parse_args()
    device = 'npu:' + str(args.npuid)
    file = sorted(os.listdir(args.path))
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    use_cpu = True
    if use_cpu:
        print('Using fps_cpu kernel...')
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
            N, D = pcd.shape
            if N <= 120000:
                index = sample_farthest_points(pcd.unsqueeze(0).cpu(), lengths=torch.tensor([N], dtype=torch.long), K=args.fps_num, random_start_point=True).view(-1)
                fps = pcd[index]
                
            else:
                # idx = self.fps_fallback(pos, batch, ratio=ratio, random_start=True)
                index, fps = farthest_point_sampling_npu(pcd, device, args.fps_num)
            grid = grid_coord[index]
            sample = {'fps_coord': fps, 'grid_coord':grid}
            torch.save(sample, os.path.join(args.outpath, fn))
            torch_npu.npu.empty_cache()
