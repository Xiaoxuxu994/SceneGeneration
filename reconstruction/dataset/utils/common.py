import numpy as np

import torch
from utils import PLATFORM, DEVICE_NAME
from utils.common import calc_extr_inv, calc_intr_inv
if PLATFORM == "ASCEND":
    import torch_npu
    import torch_npu.npu as torch_device_lib
else:
    import torch.cuda as torch_device_lib


def get_tracking_idx_by_pair(i, j, total=32):
    if i > j:
        i, j = j, i
    return ((total - 1) + (total - i)) * i // 2 + (j - i - 1)


def customized_collate_fn(batch_inputs):
    input_images = []
    input_depths = []
    input_depth_masks = []
    input_tracks0 = []
    input_tracks1 = []
    input_visibs0 = []
    input_visibs1 = []
    input_extrs = []
    input_intrs = []
    input_points = []
    for batch_input in batch_inputs:
        if batch_input is None:
            return None
        input_images.append(batch_input["input_images"])
        input_depths.append(batch_input["input_depths"])
        input_depth_masks.append(batch_input["input_depth_masks"])
        input_tracks0.append(batch_input["input_tracks0"])
        input_tracks1.append(batch_input["input_tracks1"])
        input_visibs0.append(batch_input["input_visibs0"])
        input_visibs1.append(batch_input["input_visibs1"])
        input_extrs.append(batch_input["input_extrs"])
        input_intrs.append(batch_input["input_intrs"])
        input_points.append(batch_input["input_points"])
    return dict(
        input_images=torch.stack(input_images), 
        input_depths=torch.stack(input_depths), 
        input_depth_masks=torch.stack(input_depth_masks) if input_depth_masks[0] is not None else None, 
        input_tracks0=torch.stack(input_tracks0), 
        input_tracks1=torch.stack(input_tracks1), 
        input_visibs0=torch.stack(input_visibs0) if input_visibs0[0] is not None else None, 
        input_visibs1=torch.stack(input_visibs1) if input_visibs1[0] is not None else None, 
        input_viewmats=torch.stack(input_extrs), 
        input_Ks=torch.stack(input_intrs), 
        input_points=torch.stack(input_points) if input_points[0] is not None else None, 
    )


def customized_collate_fn_placeholder(batch_inputs):
    """A placeholder collate fn for dataloader. """
    return batch_inputs[0]


def worker_init_fn(worker_id, device_id):
    torch_device_lib.set_device(device_id)


def pair_geo2match(depth0, depth1, extr0, extr1, intr0, intr1, eps=1e-2, samples_to_keep=10000, device="cpu"):
    """
    Calculate matching of two frames given their depth maps and all camera parameters. 
    Args:
        depth0 (numpy.ndarray): Depth map of frame 0, shape (b, h, w).
        depth1 (numpy.ndarray): Depth map of frame 1, shape (b, h, w).
        intr0 (numpy.ndarray): Intrinsic of frame 0, shape (b, 3, 3), supposed to be normalized.
        intr1 (numpy.ndarray): Intrinsic of frame 1, shape (b, 3, 3), supposed to be normalized.
        extr0 (numpy.ndarray): Extrinsic of frame 0, shape (b, 4, 4), supposed to be camera-to-world transformation.
        extr1 (numpy.ndarray): Extrinsic of frame 1, shape (b, 4, 4), supposed to be camera-to-world transformation.
        eps (float): Error threshold for reprojection to judge two pixels as matched. 
        samples_to_keep (int): Number of matched pixel pairs to be saved. 
        device (str): Device to perform calculation on, optional in ['cpu', 'cuda', 'npu'].
    """
    B, H, W = depth0.shape
    grid_x, grid_y = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H) # (H, W), (H, W)
    img_coord = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)[None] # 1, H, W, 3
    img_coord = torch.from_numpy(img_coord).to(device=device, dtype=torch.float32)
    img_coord = torch.cat([img_coord for _ in range(B)])
    extr0 = extr0[:, None, None]
    extr1 = extr1[:, None, None]
    intr0 = intr0[:, None, None]
    intr1 = intr1[:, None, None]
    
    img_coord0 = (img_coord * depth0[..., None]).unsqueeze(-1) # b, H, W, 3, 1
    cam_coord0 = calc_intr_inv(intr0) @ img_coord0 # b, H, W, 3, 1
    cam_coord0 = torch.concatenate([cam_coord0, torch.ones_like(cam_coord0[..., :1, :])], dim=-2) # b, H, W, 4, 1
    world_coord0 = extr0 @ cam_coord0 # b, H, W, 4, 1
    cam_coord0_1 = calc_extr_inv(extr1) @ world_coord0 # b, H, W, 4, 1; point map 0 in camera 1 coordinate
    cam_coord0_1 = cam_coord0_1[..., :3, :] # b, H, W, 3, 1
    
    img_coord0_1 = intr1 @ cam_coord0_1 # b, H, W, 3, 1
    depth0_1 = img_coord0_1[..., -1, 0] # b, H, W
    img_coord0_1 = img_coord0_1[..., :2, 0] / img_coord0_1[..., 2:, 0] # b, H, W, 2
    img_coord0_1[img_coord0_1.isnan()] = -1.0
    img_coord0_1[img_coord0_1.isinf()] = -1.0
    
    is_reproj_valid = torch.all(torch.stack([
        img_coord0_1[..., 0] > 0, 
        img_coord0_1[..., 0] < 1, 
        img_coord0_1[..., 1] > 0, 
        img_coord0_1[..., 1] < 1, 
    ], dim=-1), dim=-1) # b, H, W
    
    reproj_depth_samp01 = torch.nn.functional.grid_sample(
        depth1[:, None],                             # b, 1(c), h, w
        (img_coord0_1 * 2 - 1),                      # b, h, w, 2
        mode="nearest", padding_mode="border") # b, 1(c), h, w
    reproj_depth_samp01 = reproj_depth_samp01.reshape(*depth0_1.shape) # b, h, w

    matched_mask = torch.all(torch.stack([
            (reproj_depth_samp01 - depth0_1).abs() < eps, 
            is_reproj_valid, 
    ], dim=-1), dim=-1) # b, H, W
    
    tracks0 = img_coord[..., :2].reshape(B, -1, 2) # B, K, 2
    tracks1 = img_coord0_1.reshape(B, -1, 2) # B, K, 2
    matched_mask = matched_mask.reshape(B, -1) # B, K
    factor = torch.tensor([W, H], device=device)[None, None]
    
    filtered_tracks0 = []
    filtered_tracks1 = []
    filtered_matched_masks = []
    for b in range(B):
        matched_pixel_ids = matched_mask[b].nonzero()[:, 0]
        if len(matched_pixel_ids) >= samples_to_keep:
            pixel_ids_to_keep = np.random.choice(np.arange(len(matched_pixel_ids)), samples_to_keep, replace=False)
            pixel_ids_to_keep = torch.from_numpy(pixel_ids_to_keep).to(device=device, dtype=torch.int64)
            pixel_ids_to_keep = matched_pixel_ids[pixel_ids_to_keep]
        else:
            pixel_ids_to_keep = torch.cat([matched_pixel_ids, 
                                           torch.zeros(samples_to_keep - len(matched_pixel_ids), dtype=torch.int64, device=device)])
        filtered_track0 = tracks0[b][pixel_ids_to_keep]
        filtered_track1 = tracks1[b][pixel_ids_to_keep]
        filtered_matched_mask = matched_mask[b][pixel_ids_to_keep]
        filtered_matched_mask[-(samples_to_keep - len(matched_pixel_ids)):] = False
        filtered_tracks0.append(filtered_track0)
        filtered_tracks1.append(filtered_track1)
        filtered_matched_masks.append(filtered_matched_mask)
    
    filtered_tracks0 = torch.stack(filtered_tracks0)
    filtered_tracks1 = torch.stack(filtered_tracks1)
    filtered_matched_masks = torch.stack(filtered_matched_masks)
    # return tracks0 * factor, tracks1 * factor, matched_mask
    return filtered_tracks0 * factor, filtered_tracks1 * factor, filtered_matched_masks


def exhaustive_geo2match(depths: torch.Tensor, extrs: torch.Tensor, intrs: torch.Tensor, eps=0.01, samples_to_keep=10000):
    """
    Args:
        depths (torch.Tensor): Depth maps of shape (N, 1, H, W)
        extrs (torch.Tensor): Camera extrinsics of shape (N, 4, 4), camera-to-world.
        intrs (torch.Tensor): Camera intrinsics of shape (N, 3, 3)
    """
    N = depths.shape[0]
    pairs_x, pairs_y = [], []
    for i in range(N):
        for j in range(i + 1, N):
            pairs_x.append(i)
            pairs_y.append(j)
    pairs_x = torch.tensor(pairs_x, dtype=torch.int64)
    pairs_y = torch.tensor(pairs_y, dtype=torch.int64)
    depths0, depths1 = depths[pairs_x], depths[pairs_y]
    extrs0, extrs1 = extrs[pairs_x], extrs[pairs_y]
    intrs0, intrs1 = intrs[pairs_x], intrs[pairs_y]
    
    depths0, depths1 = depths0.squeeze(dim=1), depths1.squeeze(dim=1)
    
    tracks0, tracks1, visibs = pair_geo2match(depths0, depths1, extrs0, extrs1, intrs0, intrs1, eps=eps, samples_to_keep=samples_to_keep)
    
    return tracks0, tracks1, visibs


def rdepth_to_zdepth(r_depths: torch.Tensor, intrs: torch.Tensor):
    """
    Transform ray depth into z depth.
    Args:
        r_depths (torch.Tensor): Tensor of shape (N, 1, H, W).
        intrs (torch.Tensor): Tensor of shape (N, 3, 3), normalized.
    """
    H, W = r_depths.shape[-2:]
    image_grid = torch.cartesian_prod((torch.arange(H) + 0.5) / H, (torch.arange(W) + 0.5) / W).reshape(H, W, 2).flip(dims=[-1])
    
    pixel_vec = torch.cat([image_grid, torch.ones_like(image_grid[..., :1])], dim=-1)[None] # 1 H W 3
    pixel_vec = (calc_intr_inv(intrs)[:, None, None] @ pixel_vec[..., None])[..., 0]
    pixel_vec = pixel_vec / (pixel_vec.norm(dim=-1, keepdim=True) + 1e-12) # N H W 3
    cam_coord = pixel_vec * r_depths.permute(0, 2, 3, 1)
    z_depth = cam_coord[..., -1:].permute(0, 3, 1, 2)
        
    return z_depth
