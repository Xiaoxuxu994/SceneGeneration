import torch
import numpy as np

def concat_data_by_view(data):
    images = torch.cat([data["input_images"], data["output_images"]], dim=1)
    plucker_maps = torch.cat([data["input_cams_plucker"], data["output_cams_plucker"]], dim=1)
    cameras = []
    for input_cams, output_cams in zip(data["input_cams"], data["output_cams"]):
        cameras.append(input_cams + output_cams)
    return images, cameras, plucker_maps

@torch.jit.script 
def calc_intr_inv(intr: torch.Tensor):
    assert intr.shape[-2] == 3 and intr.shape[-1] == 3
    ret = torch.zeros_like(intr)
    ret[..., 0, 0] = 1 / intr[..., 0, 0]
    ret[..., 1, 1] = 1 / intr[..., 1, 1]
    ret[..., 0, 2] = -intr[..., 0, 2] / intr[..., 0, 0]
    ret[..., 1, 2] = -intr[..., 1, 2] / intr[..., 1, 1]
    ret[..., 2, 2] = ret[..., 2, 2] + 1.0
    return ret

@torch.jit.script 
def calc_extr_inv(extr: torch.Tensor):
    assert extr.shape[-2] == 4 and extr.shape[-1] == 4
    rot   = extr[..., :3, :3]
    trans = extr[..., :3, 3:]
    ret = torch.zeros_like(extr)
    ret[..., :3, :3] = rot.transpose(-2, -1)
    ret[..., :3, 3:] = torch.matmul(rot.transpose(-2, -1), -trans)
    ret[..., 3, 3] = ret[..., 3, 3] + 1.0
    return ret

# NOTE: original version
class WeightedMAD(torch.nn.Module):
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def weighted_median(self, x: torch.Tensor, weights: torch.Tensor, q: float=0.5):
        """Calculate the weighted median.
        
        Args:
            x (torch.Tensor): Tensor of shape [..., D].
            weights (torch.Tensor) Tensor of shape [..., D].
            q (float): Target pencertage of median.
        """
        
        orig_shape = x.shape
        flat_x = x.reshape(-1, orig_shape[-1])
        flat_weights = weights.reshape(-1, orig_shape[-1])
        
        sorted_x, sorted_idx = flat_x.sort(dim=-1)
        sorted_weights = torch.gather(flat_weights, -1, sorted_idx)
        
        cum_weights = sorted_weights.cumsum(dim=-1)
        total_weights = cum_weights[..., -1].unsqueeze(-1)
        
        target = q * total_weights
        med_idx = (cum_weights <= target).sum(dim=-1, keepdim=True).long()
        
        lower_bound = torch.gather(sorted_x, -1, med_idx.clamp(max=orig_shape[-1] - 1))
        upper_bound = torch.gather(sorted_x, -1, (med_idx + 1).clamp(max=orig_shape[-1] - 1))
        
        median = (lower_bound + upper_bound) / 2
        return median.reshape(*orig_shape[:-1], 1)
    
    def forward(self, x: torch.Tensor, weights: torch.Tensor=None) -> torch.Tensor:
        """A apply weighted MAD to x.

        Args:
            x (torch.Tensor): Tensor of shape [..., D].
            weights (torch.Tensor, optional): Tensor of shape [..., D]. Defaults to None.

        Returns:
            torch.Tensor: MAD normalized x.
        """
        if weights is None:
            weights = torch.ones_like(x)
        assert x.shape == weights.shape
        
        t_w = self.weighted_median(x, weights) # [..., 1]
        s_w = (torch.abs(x - t_w) * weights).sum(dim=-1, keepdim=True) / (weights.sum(dim=-1, keepdim=True) + self.eps) # [..., 1]
        
        normalized_x = (x - t_w) / (s_w + self.eps) # [..., D]
        
        return normalized_x


from typing import Optional

class WeightedMADFast(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    @torch.jit.export
    def weighted_median(self, x: torch.Tensor, w: torch.Tensor, q: float = 0.5) -> torch.Tensor:
        # x,w: (..., D)
        D = x.size(-1)
        x2 = x.reshape(-1, D)
        w2 = w.reshape(-1, D)

        sx, idx = torch.sort(x2, dim=-1)
        sw = torch.gather(w2, -1, idx)

        cdf = torch.cumsum(sw, dim=-1)                  # (N,D)
        target = (cdf[:, -1] * q).unsqueeze(-1)         # (N,1)

        # 找到第一个 cdf >= target 的位置
        # (cdf < target).sum 相比 (<=) 少一个边界问题，通常更稳定
        midx = (cdf < target).sum(dim=-1, keepdim=True).clamp_(0, D - 1)

        # 取该位置及下一个位置做平均（和你原逻辑一致）
        lo = torch.gather(sx, -1, midx)
        hi = torch.gather(sx, -1, (midx + 1).clamp(max=D - 1))
        med = 0.5 * (lo + hi)
        return med.reshape(*x.shape[:-1], 1)

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        if w is None:
            w = torch.ones_like(x)
        tw = self.weighted_median(x, w)  # (...,1)
        sw = (torch.abs(x - tw) * w).sum(dim=-1, keepdim=True) / (w.sum(dim=-1, keepdim=True) + self.eps)
        return (x - tw) / (sw + self.eps)

class WeightedMADHistogram(torch.nn.Module):
    def __init__(self, eps: float = 1e-8, nbins: int = 64, clip: float = 3.0):
        super().__init__()
        self.eps = eps
        self.nbins = nbins
        self.clip = clip

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        if w is None:
            w = torch.ones_like(x)

        # x,w: (..., D)
        D = x.size(-1)
        x2 = x.reshape(-1, D)
        w2 = w.reshape(-1, D)

        # 用加权均值和加权方差做一个稳健范围（避免全局min/max太受异常值影响）
        wsum = w2.sum(dim=-1, keepdim=True) + self.eps
        mu = (x2 * w2).sum(dim=-1, keepdim=True) / wsum
        var = (w2 * (x2 - mu) ** 2).sum(dim=-1, keepdim=True) / wsum
        std = torch.sqrt(var + self.eps)

        lo = mu - self.clip * std
        hi = mu + self.clip * std
        # 防止 hi==lo
        scale = (hi - lo).clamp_min_(1e-6)

        # bin index: [0, nbins-1]
        t = ((x2 - lo) / scale * (self.nbins - 1)).clamp(0, self.nbins - 1)
        idx = t.to(torch.long)

        # weighted histogram: (N, nbins)
        N = x2.size(0)
        hist = x2.new_zeros((N, self.nbins))
        hist.scatter_add_(1, idx, w2)

        cdf = torch.cumsum(hist, dim=-1)
        target = (cdf[:, -1] * 0.5).unsqueeze(-1)

        b = (cdf < target).sum(dim=-1).clamp(0, self.nbins - 1)  # (N,)
        # bin center -> median
        med = lo.squeeze(-1) + (b.to(x2.dtype) + 0.5) / self.nbins * scale.squeeze(-1)
        med = med.view(*x.shape[:-1], 1)

        # MAD scale（你原逻辑）
        sw = (torch.abs(x - med) * w).sum(dim=-1, keepdim=True) / (w.sum(dim=-1, keepdim=True) + self.eps)
        return (x - med) / (sw + self.eps)