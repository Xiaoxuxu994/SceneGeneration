import torch
import numpy as np

def concat_data_by_view(data):
    images = torch.cat([data["input_images"], data["output_images"]], dim=1)
    plucker_maps = torch.cat([data["input_cams_plucker"], data["output_cams_plucker"]], dim=1)
    cameras = []
    for input_cams, output_cams in zip(data["input_cams"], data["output_cams"]):
        cameras.append(input_cams + output_cams)
    return images, cameras, plucker_maps

def calc_intr_inv(intr: torch.Tensor):
    assert intr.shape[-2] == 3 and intr.shape[-1] == 3
    ret = torch.zeros_like(intr)
    ret[..., 0, 0] = 1 / intr[..., 0, 0]
    ret[..., 1, 1] = 1 / intr[..., 1, 1]
    ret[..., 0, 2] = -intr[..., 0, 2] / intr[..., 0, 0]
    ret[..., 1, 2] = -intr[..., 1, 2] / intr[..., 1, 1]
    ret[..., 2, 2] = ret[..., 2, 2] + 1.0
    return ret

def calc_extr_inv(extr: torch.Tensor):
    assert extr.shape[-2] == 4 and extr.shape[-1] == 4
    rot   = extr[..., :3, :3]
    trans = extr[..., :3, 3:]
    ret = torch.zeros_like(extr)
    ret[..., :3, :3] = rot.transpose(-2, -1)
    ret[..., :3, 3:] = torch.matmul(rot.transpose(-2, -1), -trans)
    ret[..., 3, 3] = ret[..., 3, 3] + 1.0
    return ret

def norm_camera_by_pcd(cameras: torch.Tensor, point_clouds: torch.Tensor):
    """Normalize camera translation by point cloud coordinate.

    Args:
        cameras (torch.Tensor): Cameras to be normalized. Tensor of shape (B, V, 4, 4).
        point_clouds (torch.Tensor): Point cloud to draw scaling factor from. Tensor of shape (B, V, H, W, 3)

    Returns:
        torch.Tensor: Cameras with normalized translation.
    """
    assert cameras.ndim == 4 and point_clouds.ndim == 5
    B = cameras.shape[0]
    point_clouds = point_clouds.reshape(B, -1, 3) # b vhw 3
    norm_factor = point_clouds.norm(dim=-1).mean(dim=-1)[..., None, None].detach() # b 1 1
    cameras_normed = cameras.clone()
    cameras_normed[..., :3, -1] = cameras[..., :3, -1] / norm_factor # b v 3
    return cameras_normed, norm_factor

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


def normalize_trans(extr: torch.Tensor, detach_norm_factor: bool=False, return_norm_factor=False):
    """Normalize tranlation of extrinsics by their camera coordinate.

    Args:
        extr (torch.Tensor): Extrinsics. Tensor of shape (..., v, 4, 4)

    Returns:
        torch.Tensor: Normalized extrinsics. Tensor of shape (..., v, 4, 4)
    """
    # t: Tensor of shape (..., v, 4, 4)
    t = extr[..., :3, 3] # ... v 3
    t_magnitude = t.norm(dim=-1).mean(dim=-1)[..., None, None] # ... 1 1
    if detach_norm_factor:
        t_magnitude = t_magnitude.detach()
    t = t / (t_magnitude + 1e-12)
    extr = extr.clone()
    extr[..., :3, 3] = t
    if not return_norm_factor:
        return extr
    return extr, t_magnitude[..., 0, 0] # (..., v, 4, 4), (..., )