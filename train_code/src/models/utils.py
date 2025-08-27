import torch
from typing import Optional, Union, List, Tuple
import math
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
        if P<=30000:
            idx = fps_cpu.farthest_point_sampling_cpu_small(points, lengths, K, start_idxs)
        else:
            idx = fps_cpu.farthest_point_sampling_cpu_large(points, lengths, K, start_idxs)
    # sampled_points = masked_gather(points, idx)

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
        farthest = torch.randint(0, N, (1,), device=x.device).item() if random_start else 0

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


def hash_coords(coords, resolution):
    return coords[:, 0] * (resolution+1) * (resolution+1) + coords[:, 1] * (resolution+1) + coords[:, 2]

def isin(elements, test_elements):
    # 需要先对 test_elements 排序
    test_elements_sorted, _ = torch.sort(test_elements.unique())
    idx = torch.searchsorted(test_elements_sorted, elements)
    idx[idx == len(test_elements_sorted)] = 0
    return test_elements_sorted[idx] == elements

def get_fine_voxel_mask_from_coarse_labels(
    coarse_label, scale_factor, resolution_fine, coarse_grid, fine_grid
):
    device = coarse_label.device
    active_coarse = coarse_grid[coarse_label.bool()]
    if active_coarse.shape[0] == 0:
        return torch.zeros((fine_grid.shape[0],), dtype=torch.bool, device=device)

    offsets = torch.tensor([[i, j, k] for i in range(scale_factor)
                                       for j in range(scale_factor)
                                       for k in range(scale_factor)], dtype=torch.long, device=device)

    base = active_coarse * scale_factor
    fine_candidates = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 3)

    fine_hash_all = hash_coords(fine_grid, resolution_fine)
    fine_candidate_hash = hash_coords(fine_candidates, resolution_fine)
    return isin(fine_hash_all, fine_candidate_hash)

def logit2label(logit, threshold=0.5):
    logit = logit.squeeze(-1).squeeze(0)           
    prob = torch.sigmoid(logit)         
    return (prob >= threshold)

