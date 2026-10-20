import torch
import numpy as np

def calc_intr_inv(intr: torch.Tensor):
    assert intr.shape[-2] == 3 and intr.shape[-1] == 3
    ret = torch.zeros_like(intr)
    ret[..., 0, 0] = 1 / intr[..., 0, 0]
    ret[..., 1, 1] = 1 / intr[..., 1, 1]
    ret[..., 0, 2] = -intr[..., 0, 2] / intr[..., 0, 0]
    ret[..., 1, 2] = -intr[..., 1, 2] / intr[..., 1, 1]
    ret[..., 2, 2] = ret[..., 2, 2] + 1.0
    return ret

def rdepth_to_zdepth(r_depths: torch.Tensor, intrs: torch.Tensor, intr_is_normalized: bool = False):
    """
    Transform ray depth into z depth.
    Args:
        r_depths (torch.Tensor): Tensor of shape (N, 1, H, W).
        intrs (torch.Tensor): Tensor of shape (N, 3, 3), normalized.
    """
    N, _, H, W = r_depths.shape
    device, dtype = r_depths.device, r_depths.dtype
    intrs = intrs.to(device=device, dtype=dtype)

    if not intr_is_normalized:
        intrs = intrs.clone()
        intrs[:, 0, 0] /= W
        intrs[:, 0, 2] /= W
        intrs[:, 1, 1] /= H
        intrs[:, 1, 2] /= H

    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixel_vec = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)[None]  # 1,H,W,3

    pixel_vec = (calc_intr_inv(intrs)[:, None, None] @ pixel_vec[..., None])[..., 0]
    pixel_vec = pixel_vec / (pixel_vec.norm(dim=-1, keepdim=True) + 1e-12)

    return (pixel_vec[..., 2:3] * r_depths.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# 分位截断的全局开关：BaseDataset 在每次取样本前按该数据集的配置设置（set_depth_filter），
# 各 loader 无需改动。默认值与旧行为完全一致。
PERCENTILE_VALID_ONLY = False   # True：分位数只在 depth > 0 的有效像素上计算（旧行为把无效像素的 0 也算进去）
PERCENTILE_DISABLED = False     # True：跳过分位截断，只保留 max_depth 截断


def set_depth_filter(valid_only=False, disabled=False):
    global PERCENTILE_VALID_ONLY, PERCENTILE_DISABLED
    PERCENTILE_VALID_ONLY = bool(valid_only)
    PERCENTILE_DISABLED = bool(disabled)


def threshold_depth_map(
    depth_map: np.ndarray,
    max_percentile: float = -1,
    min_percentile: float = -1,
    max_depth: float = -1,
) -> np.ndarray:
    """Thresholds a depth map using percentile-based limits and optional
    maximum depth clamping.

    Steps:
      1. If `max_depth > 0`, clamp all values above `max_depth` to zero.
      2. Compute `max_percentile` and `min_percentile` thresholds using nanpercentile.
      3. Zero out values above/below these thresholds, if thresholds are > 0.

    Args:
        depth_map (np.ndarray):
            Input depth map (H, W).
        max_percentile (float):
            Upper percentile (0-100). Values above this will be set to zero.
        min_percentile (float):
            Lower percentile (0-100). Values below this will be set to zero.
        max_depth (float):
            Absolute maximum depth. If > 0, any depth above this is set to zero.
            If <= 0, no maximum-depth clamp is applied.

    Returns:
        np.ndarray:
            Depth map (H, W) after thresholding. Some or all values may be zero.
            Returns None if depth_map is None.
    """
    if depth_map is None:
        return None

    depth_map = depth_map.astype(float, copy=True)

    # Optional clamp by max_depth
    if max_depth > 0:
        depth_map[depth_map > max_depth] = 0.0

    if PERCENTILE_DISABLED:
        max_percentile = min_percentile = -1

    # Percentile-based thresholds
    src = depth_map
    if PERCENTILE_VALID_ONLY and (max_percentile > 0 or min_percentile > 0):
        src = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
        if src.size == 0:
            return depth_map
    depth_max_thres = np.nanpercentile(src, max_percentile) if max_percentile > 0 else None
    depth_min_thres = np.nanpercentile(src, min_percentile) if min_percentile > 0 else None

    # Apply the thresholds if they are > 0
    if depth_max_thres is not None and depth_max_thres > 0:
        depth_map[depth_map > depth_max_thres] = 0.0
    if depth_min_thres is not None and depth_min_thres > 0:
        depth_map[depth_map < depth_min_thres] = 0.0

    return depth_map