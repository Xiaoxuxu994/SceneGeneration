import torch
import torch_npu
import math
import time
import os
from typing import Optional, Union, List, Tuple
import fps_cpu

def sample_farthest_points(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List, torch.Tensor] = 50,
    random_start_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative farthest point sampling algorithm [1] to subsample a set of
    K points from a given pointcloud. At each iteration, a point is selected
    which has the largest nearest neighbor distance to any of the
    already selected points.

    Farthest point sampling provides more uniform coverage of the input
    point cloud compared to uniform random sampling.

    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.

    Args:
        points: (N, P, D) array containing the batch of pointclouds
        lengths: (N,) number of points in each pointcloud (to support heterogeneous
            batches of pointclouds)
        K: samples required in each sampled point cloud (this is typically << P). If
            K is an int then the same number of samples are selected for each
            pointcloud in the batch. If K is a tensor is should be length (N,)
            giving the number of samples to select for each element in the batch
        random_start_point: bool, if True, a random point is selected as the starting
            point for iterative sampling.

    Returns:
        selected_points: (N, K, D), array of selected values from points. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            0.0 for batch elements where k_i < max(K).
        selected_indices: (N, K) array of selected indices. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            -1 for batch elements where k_i < max(K).
    """
    N, P, D = points.shape
    device = points.device
    # import pdb; pdb.set_trace()
    # Validate inputs
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)
    else:
        if lengths.shape != (N,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("A value in lengths was too large.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # Check dtypes are correct and convert if necessary
    if not (points.dtype == torch.float32):
        points = points.to(torch.float32)
    if not (lengths.dtype == torch.int64):
        lengths = lengths.to(torch.int64)
    if not (K.dtype == torch.int64):
        K = K.to(torch.int64)

    # Generate the starting indices for sampling
    start_idxs = torch.zeros_like(lengths)
    if random_start_point:
        for n in range(N):
            # pyre-fixme[6]: For 1st param expected `int` but got `Tensor`.
            start_idxs[n] = torch.randint(high=lengths[n], size=(1,)).item()

    with torch.no_grad():
        # pyre-fixme[16]: `pytorch3d_._C` has no attribute `sample_farthest_points`.
        idx = fps_cpu.farthest_point_sampling_cpu(points, lengths, K, start_idxs)
    # sampled_points = masked_gather(points, idx)ffff

    return idx


def fps_fallback(x, batch=None, ratio=0.25, random_start=True):
    """
    PyTorch fallback version of torch_cluster.fps (Furthest Point Sampling)

    Args:
        x (Tensor): (N, D) point positions
        batch (LongTensor): (N,) batch indices or None
        ratio (float): sampling ratio ∈ (0, 1]
        random_start (bool): whether to randomly initialize the first point

    Returns:
        Tensor: (M,) indices of sampled points (subset of input x)
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    batch_size = batch.max().item() + 1
    output_idx = []

    for b in range(batch_size):
        mask = batch == b
        pts = x[mask]
        N = pts.size(0)
        M = max(1, math.ceil(N * ratio))  # 至少采样1个点,follow torch_cluster, 向上取整

        centroids = torch.zeros(M, dtype=torch.long, device=x.device)
        distance = torch.full((N,), 1e10, device=x.device)
        if random_start:
            farthest = torch.randint(0, N, (1,), device=x.device).item()
        else:
            farthest = 0

        for i in range(M):   
            centroids[i] = farthest
            centroid = pts[farthest].unsqueeze(0)
            dist = torch.sum((pts - centroid) ** 2, dim=1)
            mask_dist = dist < distance
            distance[mask_dist] = dist[mask_dist]
            farthest = torch.argmax(distance).item()

        batch_indices = torch.where(mask)[0][centroids]  # 恢复原始索引
        output_idx.append(batch_indices)

    return torch.cat(output_idx, dim=0)

if __name__ == '__main__':
    n_iter = 10
    n_pts = 50000
    num_latent = 2048
    pos = torch.rand(n_pts,3,dtype=torch.float32)
    batch = torch.zeros([n_pts],dtype=torch.long)
    ratio = 1.0 * num_latent / n_pts
    device = torch.device('npu:4')
    os.environ["OMP_NUM_THREADS"] = "16"  # 使用4个线程
    torch.set_num_threads(16)  # 设置PyTorch的CPU线程数


    ## torch fps fall back 调用
    print('======== torch fps_fallback =============')
    idx_fps_fallback = fps_fallback(pos.to(device), batch=batch.to(device), ratio=ratio,random_start=True)


    # ============== fallback FPS NPU ==================
    print("======== fps_fallback NPU (avg of 10 runs) =============")
    fallback_time = 0.0
    fallback_time2 = 0.0
    profiling = True
    if profiling:
        # import torch_npu
        experimental_config = torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)
        prof = torch_npu.profiler.profile(
                            activities=[torch_npu.profiler.ProfilerActivity.NPU,
                                        torch_npu.profiler.ProfilerActivity.CPU],
                            with_stack=True,
                            record_shapes=False,
                            profile_memory=True,
                            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=0,
                                                                    skip_first=0),
                            experimental_config=experimental_config,
                            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./profiling/fps/test_cpu'))
    if profiling:
        prof.start()
    for i in range(n_iter):
        # torch_npu.npu.synchronize()
        # start = time.time()
        idx = fps_fallback(pos.to(device), batch=batch.to(device), ratio=ratio, random_start=False)
        # torch_npu.npu.synchronize()
        # end = time.time()
        # fallback_time += end - start
        # torch_npu.npu.synchronize()
        # start2 = time.time()
        # idx2 = FarthestPointSampling.apply(pos, batch[0], num_latent)
        idx2 = sample_farthest_points(pos.unsqueeze(0), lengths=torch.tensor([n_pts], dtype=torch.long), K=num_latent, random_start_point=False).squeeze().to(device)
        # print(idx, idx2)
        # torch_npu.npu.synchronize()
        # end2 = time.time()
        # fallback_time2 += end2 - start
        if profiling:
            prof.step()
            if i > 5:
                print('Finished Profiling')
                prof.stop()
                exit(-1)
        # assert torch.equal(idx, idx2), 'Two methods must be the same.'

    

    # print(f"[fps_fallback] average time over {n_iter} runs: {fallback_time / n_iter:.4f}s")
    # print(f"[fps_fallback2] average time over {n_iter} runs: {fallback_time2 / n_iter:.4f}s")
