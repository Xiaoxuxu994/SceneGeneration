import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

# %%
""" loss """
def l1_loss(pred, gt, mask=None):
    """
    pred, gt: torch.Tensor of any shape
    mask: torch.Tensor (same shape as pred or broadcastable), or None
    """
    diff = torch.abs(pred - gt)

    if mask is not None:
        mask = mask.to(diff.dtype)  # 支持 bool, int, float
        masked_diff = diff * mask
        denom = mask.sum()
        if denom == 0:
            return torch.tensor(0.0, device=pred.device)
        return masked_diff.sum() / denom
    else:
        return diff.mean()


def gaussian_kernel(window_size: int, sigma: float, device=None):
    """Create a 2D Gaussian kernel normalized to sum=1."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel_2d = g[:, None] * g[None, :]
    return kernel_2d / kernel_2d.sum()

def masked_ssim_batch(img1, img2, mask, window_size=11, sigma=1.5):
    """
    Vectorized masked SSIM.
    img1, img2: [B, C, H, W], float in [0,1]
    mask:       [B, H, W] or [B, 1, H, W], bool or float
    return:     [B], masked SSIM score per image
    """
    assert img1.shape == img2.shape, "Images must have same shape"
    B, C, H, W = img1.shape
    device = img1.device

    # Ensure mask shape
    mask = mask.float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)  # [B, 1, H, W]

    # Gaussian kernel (expanded per channel)
    window = gaussian_kernel(window_size, sigma, device)
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, W, W]
    window = window.expand(C, 1, window_size, window_size)  # [C, 1, W, W]

    # Local mask sum for normalization
    mask_sum = F.conv2d(mask, window[:1], padding=window_size // 2, groups=1)

    # === Local means ===
    mu1 = F.conv2d(img1 * mask, window, padding=window_size // 2, groups=C) / (mask_sum + 1e-6)
    mu2 = F.conv2d(img2 * mask, window, padding=window_size // 2, groups=C) / (mask_sum + 1e-6)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # === Local variances & covariance ===
    sigma1_sq = F.conv2d(img1**2 * mask, window, padding=window_size // 2, groups=C) / (mask_sum + 1e-6) - mu1_sq
    sigma2_sq = F.conv2d(img2**2 * mask, window, padding=window_size // 2, groups=C) / (mask_sum + 1e-6) - mu2_sq
    sigma12   = F.conv2d(img1 * img2 * mask, window, padding=window_size // 2, groups=C) / (mask_sum + 1e-6) - mu1_mu2

    # === SSIM constants ===
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # === SSIM map ===
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # === Mean over channels ===
    ssim_map = ssim_map.mean(dim=1, keepdim=True)  # [B, 1, H, W]

    # === Masked spatial average ===
    masked_ssim = (ssim_map * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-6)
    return masked_ssim.squeeze(1)  # [B]


def dispersion_loss_voxel_scaled(xyz, voxel_size=0.01, margin_rel=0.5, beta=10, eps=1e-5, activation="softplus"):
    # xyz: [N, M, 3]
    N, M, _ = xyz.shape
    pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
    if not pairs:
        return torch.tensor(0.0, device=xyz.device)

    # Use clamp before sqrt to avoid NaNs
    dists = [((xyz[:, i] - xyz[:, j])**2).sum(-1).clamp(min=eps).sqrt() for i, j in pairs]  # [N, num_pairs]
    pairwise = torch.stack(dists, dim=1)
    d_min = pairwise.min(dim=1)[0]  # [N]

    d_norm = d_min / voxel_size  # relative to voxel size

    if margin_rel is not None:
        if activation == 'softplus':
            loss = F.softplus(margin_rel - d_norm, beta=beta)
        else:
            loss = F.relu(margin_rel - d_norm)
    else:
        safe = torch.clamp(d_norm, min=eps)
        loss = 1.0 / safe

    return loss.mean()

def feature_diversity_loss(
    feat_div: torch.Tensor,  # [N_voxel, G, D]
    mode: str = "pairwise_l2", 
    eps: float = 1e-6,
    margin: float | None = None
) -> torch.Tensor:
    """
    Compute feature diversity loss for voxel internal Gaussians.

    Args:
        feat_div: [N_voxel, G, D] (color+opacity)
        mode: "neg_var" | "inv_var" | "pairwise_l2"
        eps: small constant
        margin: only for pairwise_l2

    Returns:
        loss: scalar tensor
    """
    N, G, D = feat_div.shape

    if mode == "neg_var":
        var_per_dim = feat_div.var(dim=1, unbiased=False)  # [N, D]
        v = var_per_dim.mean(dim=-1)                       # [N]
        return -v.mean()

    elif mode == "inv_var":
        var_per_dim = feat_div.var(dim=1, unbiased=False)  # [N, D]
        v = var_per_dim.mean(dim=-1)
        return (1.0 / (v + eps)).mean()

    elif mode == "pairwise_l2":
        # [N, G, 1, D] - [N, 1, G, D] -> [N, G, G, D]
        diff = feat_div[:, :, None, :] - feat_div[:, None, :, :]
        dists = diff.norm(dim=-1)  # [N, G, G]

        # 取上三角，不包含对角线
        mask = torch.triu(torch.ones(G, G, device=feat_div.device), diagonal=1).bool()
        pairwise_dists = dists[:, mask]  # [N, G*(G-1)/2]

        if margin is not None:
            # 只有距离小于 margin 才惩罚
            loss = F.relu(margin - pairwise_dists)
            return loss.mean()
        else:
            return -pairwise_dists.mean()  # 最大化两两距离

    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_psnr(rends, gts, eps=1e-8):
    """
    Args:
        rends: [N,3,H,W] tensor, values in [0,1]
        gts: [N,3,H,W] tensor, values in [0,1]
    Returns:
        psnr: float, mean PSNR over batch
    """
    mse = torch.mean((rends - gts) ** 2, dim=[1,2,3])  # 每张图的 MSE
    psnr = -10.0 * torch.log10(mse + eps)              # 防止 log0
    return psnr.mean().item()

def compute_SSIM(rends, gts):
    """
    Args:
        rends: [N,3,H,W]
        gts: [N,3,H,W]
    Returns:
        mean SSIM over batch
    """
    # torchmetrics SSIM expects (N,C,H,W) in [0,1]
    return ssim(rends, gts).item()    

import lpips
def compute_LPIPS(lpips_model, rends, gts, device):
    """
    Args:
        rends: [N,3,H,W], values in [0,1]
        gts: [N,3,H,W], values in [0,1]
    Returns:
        mean LPIPS over batch
    """
    rends_norm = 2*rends - 1  # LPIPS expects [-1,1]
    gts_norm = 2*gts - 1
    lpips_vals = []
    for r, g in zip(rends_norm, gts_norm):
        val = lpips_model(r.unsqueeze(0).to(device), g.unsqueeze(0).to(device))
        lpips_vals.append(val.item())
    return float(np.mean(lpips_vals))

def compute_metrics(rends, gts):
    psnr = compute_psnr(rends, gts)
    lpips = compute_LPIPS(rends, gts)
    ssim = compute_SSIM(rends, gts)
    return psnr, lpips, ssim
