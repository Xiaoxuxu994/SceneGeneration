import numpy as np

import torch

def get_tracking_idx_by_pair(i, j, total=32):
    if i > j:
        i, j = j, i
    return ((total - 1) + (total - i)) * i // 2 + (j - i - 1)

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

def pair_geo2match(depth0, depth1, extr0, extr1, intr0, intr1, eps=1e-2, samples_to_keep=10000, device="cpu", grid_size=25):
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
    # timer = MyTimer()
    # timer.record_start("geo2match_0_Total")
    # timer.record_start("geo2match_1_ReProj_matmul")
    
    B, H, W = depth0.shape
    extr0 = extr0[:, None, None]
    extr1 = extr1[:, None, None]
    intr0 = intr0[:, None, None]
    intr1 = intr1[:, None, None]

    if grid_size is None:
        grid_x, grid_y = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H) # (H, W), (H, W)
    else:
        grid_x, grid_y = np.meshgrid((np.linspace(0, W-1, grid_size) + 0.5) / W, (np.linspace(0, H-1, grid_size) + 0.5) / H) # (gs, gs), (gs, gs)
    img_coord = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)[None] # 1, H, W, 3
    img_coord = torch.from_numpy(img_coord).to(device=device, dtype=torch.float32)
    img_coord = torch.cat([img_coord for _ in range(B)]) # B, gs, gs, 3
    if grid_size is None:
        depth0_grid = depth0.clone()
    else:
        depth0_grid = torch.nn.functional.grid_sample(
            depth0[:, None], 
            (img_coord[..., :2] * 2 - 1), 
            mode="nearest", padding_mode="zeros")[:, 0] # B gs gs
    
    img_coord0 = (img_coord * depth0_grid[..., None]).unsqueeze(-1) # b, gs, gs, 3, 1
    cam_coord0 = calc_intr_inv(intr0) @ img_coord0 # b, gs, gs, 3, 1
    cam_coord0 = torch.concatenate([cam_coord0, torch.ones_like(cam_coord0[..., :1, :])], dim=-2) # b, gs, gs, 4, 1
    world_coord0 = extr0 @ cam_coord0 # b, gs, gs, 4, 1
    cam_coord0_1 = calc_extr_inv(extr1) @ world_coord0 # b, gs, gs, 4, 1; point map 0 in camera 1 coordinate
    cam_coord0_1 = cam_coord0_1[..., :3, :] # b, gs, gs, 3, 1
    
    img_coord0_1 = intr1 @ cam_coord0_1 # b, gs, gs, 3, 1
    depth0_1 = img_coord0_1[..., -1, 0] # b, gs, gs
    img_coord0_1 = img_coord0_1[..., :2, 0] / img_coord0_1[..., 2:, 0] # b, gs, gs, 2
    # timer.record_end("geo2match_1_ReProj_matmul")
    # timer.record_start("geo2match_2_ReProj_mask")
    img_coord0_1[img_coord0_1.isnan()] = -1.0
    img_coord0_1[img_coord0_1.isinf()] = -1.0
    # timer.record_end("geo2match_2_ReProj_mask")
    
    # timer.record_start("geo2match_3_ReProj_gridsamp")
    
    is_reproj_valid = torch.all(torch.stack([
        img_coord0_1[..., 0] > 0, 
        img_coord0_1[..., 0] < 1, 
        img_coord0_1[..., 1] > 0, 
        img_coord0_1[..., 1] < 1, 
        depth0_1 > 0, 
    ], dim=-1), dim=-1) # b, gs, gs
    
    reproj_depth_samp01 = torch.nn.functional.grid_sample(
        depth1[:, None],                             # b, 1(c), h, w
        (img_coord0_1 * 2 - 1),                      # b, gs, gs, 2
        mode="nearest", padding_mode="zeros") # b, 1(c), gs, gs
    reproj_depth_samp01 = reproj_depth_samp01.reshape(*depth0_1.shape) # b, gs, gs
    
    # timer.record_end("geo2match_3_ReProj_gridsamp")
    # timer.record_start("geo2match_4_BatchWiseSamp")
    
    matched_mask = torch.all(torch.stack([
        (reproj_depth_samp01 - depth0_1).abs() < eps, 
        is_reproj_valid, 
    ], dim=-1), dim=-1) # b, gs, gs
    
    tracks0 = img_coord[..., :2].reshape(B, -1, 2) # B, gsxgs, 2
    tracks1 = img_coord0_1.reshape(B, -1, 2) # B, gsxgs, 2
    matched_mask = matched_mask.reshape(B, -1) # B, gsxgs
    factor = torch.tensor([W, H], device=device)[None, None]
    
    if grid_size is None:
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
    else:
        filtered_tracks0 = tracks0
        filtered_tracks1 = tracks1
        filtered_matched_masks = matched_mask
    
    # timer.record_end("geo2match_4_BatchWiseSamp")
    # timer.record_end("geo2match_0_Total")
    # timer.report()
    
    return filtered_tracks0 * factor, filtered_tracks1 * factor, filtered_matched_masks


def deflat_visibility_mat(visibs: torch.Tensor, clip_view_num: int):
    """Deflat a flattened visibility mat into a matrix of shape (clip_view_num, clip_view_num)

    Args:
        visibs (torch.Tensor): Tensor of shape (clip_view_num(clip_view_num-1)/2, )
        clip_view_num (int): View number corresponds to the visibility.
    """
    assert len(visibs) == clip_view_num * (clip_view_num - 1) // 2, "Visibs shape: {}, clip view num: {}".format(visibs.shape, clip_view_num)
    visib_mat = torch.zeros((clip_view_num, clip_view_num), dtype=torch.float32)
    for v in range(clip_view_num):
        visib_mat[v, v + 1:] = visibs[get_tracking_idx_by_pair(i=v, j=v+1, total=clip_view_num): 
                                      get_tracking_idx_by_pair(i=v, j=v+1, total=clip_view_num) + clip_view_num - (v + 1)]
        visib_mat[v + 1:, v] = visib_mat[v, v + 1:]
    return visib_mat


def exhaustive_geo2match(depths: torch.Tensor, extrs: torch.Tensor, intrs: torch.Tensor, eps=0.01, samples_to_keep=10000, grid_size=None):
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
    
    tracks0, tracks1, visibs = pair_geo2match(depths0, depths1, extrs0, extrs1, intrs0, intrs1, eps=eps, 
                                              samples_to_keep=samples_to_keep, grid_size=grid_size)
    
    return tracks0, tracks1, visibs