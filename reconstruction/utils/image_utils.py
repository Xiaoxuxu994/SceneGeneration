import os
import torch
import matplotlib.pyplot as plt

from einops import rearrange
from torchvision.utils import draw_keypoints


def gray2hot(x: torch.Tensor) -> torch.Tensor:
    """Transform gray scale images into hot images. 

    Args:
        x (torch.Tensor): Gray scale images, of shape (B, V, C, H, W) or (N, C, H, W)

    Returns:
        torch.Tensor: Transformed images.
    """
    
    if x.ndim == 5:
        B, V, C, H, W = x.shape
        BV = B * V
        _x = x.reshape(-1, C, H, W)
    else:
        _x = x
        BV = x.shape[0]
    
    # _x = (_x - _x.min()) / (_x.max() - _x.min())
    lower = _x.reshape(BV, -1).min(dim=-1).values[..., None, None, None]
    upper = _x.reshape(BV, -1).max(dim=-1).values[..., None, None, None]
    _x = (_x - lower) / (upper - lower)
    _x = _x.squeeze(dim=1).cpu()
    
    cmapper = plt.get_cmap("magma_r")
    hot = torch.tensor(cmapper(_x)[..., :3])
    hot = hot.permute(0, 3, 1, 2)
    
    if x.ndim == 5:
        hot = hot.reshape(B, V, 3, H, W)
    
    return hot


def gray2rb(x: torch.Tensor) -> torch.Tensor:
    """Transform gray scale positive/negative images into colored images. 

    Args:
        x (torch.Tensor): Gray scale positive/negative images, of shape (B, V, 1, H, W) or (N, 1, H, W)

    Returns:
        torch.Tensor: Transformed images.
    """
    positive_mask = x > 0
    negative_mask = x < 0
    y = torch.cat([torch.zeros_like(x) for _ in range(3)], dim=-3)
    y[..., 0:1, :, :] = x * positive_mask
    y[..., 2:3, :, :] = x * negative_mask * -1
    return y


def normalize_gray(x: torch.Tensor) -> torch.Tensor:
    b, v, c, h, w = x.shape
    _x = x.reshape(b, v, -1)
    _x = _x / (_x.max(dim=-1, keepdim=True).values)
    return torch.cat([_x.reshape(*x.shape) for _ in range(3)], dim=2)


def visualize_prediction(pred, data, matching_info=None):
    """Visualize the prediction of model.

    Args:
        output_dir (_type_): _description_
        pred (_type_): _description_
        data (_type_): _description_
    """
    
    visualized_predictions = []
    
    images         = torch.cat([data["input_images"]], dim=1).detach().cpu()
    tracked_images = images.clone()
    if matching_info is not None:
        tracks = matching_info["matches"]      # b v k 2, b v k 2
        visibi = matching_info["visibilities"] # b v k 1, b v k 1
        B, V = tracks.shape[:2]
        for b in range(B):
            visibility_mask = torch.all(visibi[b], dim=0) # k 1
            for v in range(V):
                tracked_images[b, v] = draw_keypoints(tracked_images[b, v], tracks[b, v][visibility_mask].unsqueeze(0), colors="red")
                
    depths_mono    = torch.cat([data["input_depths"]], dim=1).detach().cpu()
    depths_mono    = torch.log10(depths_mono + 1.0)
    depths_aligned = torch.cat([pred["depths"]], dim=1).detach().cpu()
    depths_aligned = torch.log10(depths_aligned + 1.0)
    conf_maps      = torch.cat([pred["depths_conf"]], dim=1).detach().cpu().reshape(*depths_mono.shape)
    depths_mono    = gray2hot(depths_mono)
    depths_aligned = gray2hot(depths_aligned)
    conf_maps      = normalize_gray(conf_maps)
    images         = rearrange(images, "b v c h w -> c (b v h) w")
    tracked_images = rearrange(tracked_images, "b v c h w -> c (b v h) w")
    depths_mono    = rearrange(depths_mono, "b v c h w -> c (b v h) w")
    depths_aligned = rearrange(depths_aligned, "b v c h w -> c (b v h) w")
    conf_maps      = rearrange(conf_maps, "b v c h w -> c (b v h) w")
    visualized_predictions.extend([images, tracked_images, depths_mono, depths_aligned, conf_maps])

    visualize_image = torch.stack(visualized_predictions, dim=0)
    return visualize_image