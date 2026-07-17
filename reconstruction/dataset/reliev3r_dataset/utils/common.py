import numpy as np

import torch
from utils import PLATFORM, DEVICE_NAME
from utils.common import calc_extr_inv, calc_intr_inv, depth2pointmap, get_norm_map
if PLATFORM == "ASCEND":
    import torch_npu
    import torch_npu.npu as torch_device_lib
else:
    import torch.cuda as torch_device_lib


import time
from collections import OrderedDict
class MyTimer:
    def __init__(self):
        self.infos = OrderedDict()
    
    def add_item(self, desc):
        if desc not in self.infos:
            self.infos.update({desc: {"total": 0.0, "start": None}})
    
    def record_start(self, desc):
        if desc not in self.infos:
            self.add_item(desc)
        update_info = self.infos[desc]
        update_info["start"] = time.time()
        self.infos.update({desc: update_info})
    
    def record_end(self, desc):
        update_info = self.infos[desc]
        update_info["total"] += time.time() - update_info["start"]
        update_info["start"] = None
        self.infos.update({desc: update_info})
    
    def report(self):
        return
        for k, v in self.infos.items():
            print("[ TIMER ] {}: {:.4f} s".format(k, v["total"]))
    
    def clear(self):
        self.infos.clear()


def get_tracking_idx_by_pair(i, j, total=32, bidirect=False):
    if bidirect:
        return i * (total - 1) + (j if j < i else j - 1)
    if i > j:
        i, j = j, i
    return ((total - 1) + (total - i)) * i // 2 + (j - i - 1)


def deflat_visibility_mat(visibs: torch.Tensor, clip_view_num: int, bidirect=False):
    """Deflat a flattened visibility mat into a matrix of shape (clip_view_num, clip_view_num)

    Args:
        visibs (torch.Tensor): Tensor of shape (clip_view_num(clip_view_num-1)/2, )
        clip_view_num (int): View number corresponds to the visibility.
    """
    if not bidirect:
        assert len(visibs) == clip_view_num * (clip_view_num - 1) // 2, "Visibs shape: {}, clip view num: {}".format(visibs.shape, clip_view_num)
    else:
        assert len(visibs) == clip_view_num * (clip_view_num - 1), "Visibs shape: {}, clip view num: {}".format(visibs.shape, clip_view_num)
    visib_mat = torch.zeros((clip_view_num, clip_view_num), dtype=torch.float32)
    for v in range(clip_view_num):
        idx_v_vp1 = get_tracking_idx_by_pair(i=v, j=v+1, total=clip_view_num, bidirect=bidirect)
        visib_mat[v, v + 1:] = visibs[idx_v_vp1: idx_v_vp1 + clip_view_num - (v + 1)]
        visib_mat[v + 1:, v] = visib_mat[v, v + 1:]
    return visib_mat


def customized_collate_fn(batch_inputs):
    input_images = []
    input_depths = []
    input_depth_masks = []
    input_tracks0 = []
    input_tracks1 = []
    input_visibs0 = []
    input_visibs1 = []
    input_tracks_view_id0 = []
    input_tracks_view_id1 = []
    input_extrs = []
    input_intrs = []
    input_points = []
    input_points_local = []
    dataset_names = []
    data_type = None
    view_ids = []
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
        input_tracks_view_id0.append(batch_input["input_tracks_view_id0"])
        input_tracks_view_id1.append(batch_input["input_tracks_view_id1"])
        input_extrs.append(batch_input["input_extrs"])
        input_intrs.append(batch_input["input_intrs"])
        input_points.append(batch_input["input_points"])
        input_points_local.append(batch_input["input_points_local"])
        dataset_names.append(batch_input["dataset_name"])
        data_type = batch_input["data_type"]
        view_ids.append(batch_input["view_ids"])
    return dict(
        input_images=torch.stack(input_images), 
        input_depths=torch.stack(input_depths), 
        input_depth_masks=torch.stack(input_depth_masks) if input_depth_masks[0] is not None else None, 
        input_tracks0=torch.stack(input_tracks0), 
        input_tracks1=torch.stack(input_tracks1), 
        input_visibs0=torch.stack(input_visibs0) if input_visibs0[0] is not None else None, 
        input_visibs1=torch.stack(input_visibs1) if input_visibs1[0] is not None else None, 
        input_tracks_view_id0=torch.stack(input_tracks_view_id0), 
        input_tracks_view_id1=torch.stack(input_tracks_view_id1), 
        input_viewmats=torch.stack(input_extrs), 
        input_Ks=torch.stack(input_intrs), 
        input_points=torch.stack(input_points) if input_points[0] is not None else None, 
        input_points_local=torch.stack(input_points_local) if input_points_local[0] is not None else None, 
        dataset_names=dataset_names, 
        data_type=data_type, 
        view_ids=torch.stack(view_ids)
    )


def customized_collate_fn_placeholder(batch_inputs):
    """A placeholder collate fn for dataloader. """
    return batch_inputs[0]


def worker_init_fn(worker_id, device_id):
    torch_device_lib.set_device(device_id)


def pair_geo2match(depth0, depth1, depth_mask0, depth_mask1, extr0, extr1, intr0, intr1, 
                   eps=1e-2, samples_to_keep=10000, device="cpu", grid_size=25):
    """
    Calculate matching of two frames given their depth maps and all camera parameters. 
    Args:
        depth0 (numpy.ndarray): Depth map of frame 0, shape (b, h, w).
        depth1 (numpy.ndarray): Depth map of frame 1, shape (b, h, w).
        depth_mask0 (numpy.ndarray): Depth mask of frame 0, shape (b, h, w).
        depth_mask1 (numpy.ndarray): Depth mask of frame 1, shape (b, h, w).
        intr0 (numpy.ndarray): Intrinsic of frame 0, shape (b, 3, 3), supposed to be normalized.
        intr1 (numpy.ndarray): Intrinsic of frame 1, shape (b, 3, 3), supposed to be normalized.
        extr0 (numpy.ndarray): Extrinsic of frame 0, shape (b, 4, 4), supposed to be camera-to-world transformation.
        extr1 (numpy.ndarray): Extrinsic of frame 1, shape (b, 4, 4), supposed to be camera-to-world transformation.
        eps (float): Error threshold for reprojection to judge two pixels as matched. 
        samples_to_keep (int): Number of matched pixel pairs to be saved. 
        device (str): Device to perform calculation on, optional in ['cpu', 'cuda', 'npu'].
    """
    timer = MyTimer()
    timer.record_start("geo2match_0_Total")
    timer.record_start("geo2match_1_ReProj_matmul")
    
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
        depth_mask0_grid = depth_mask0.clone()
    else:
        depth0_grid = torch.nn.functional.grid_sample(
            depth0[:, None], 
            (img_coord[..., :2] * 2 - 1), 
            mode="nearest", padding_mode="zeros", align_corners=False)[:, 0] # B gs gs
        depth_mask0_grid = torch.nn.functional.grid_sample(
            depth_mask0[:, None], 
            (img_coord[..., :2] * 2 - 1), 
            mode="nearest", padding_mode="zeros", align_corners=False)[:, 0] # B gs gs
    
    img_coord0 = (img_coord * depth0_grid[..., None]).unsqueeze(-1) # b, gs, gs, 3, 1
    cam_coord0 = calc_intr_inv(intr0) @ img_coord0 # b, gs, gs, 3, 1
    cam_coord0 = torch.concatenate([cam_coord0, torch.ones_like(cam_coord0[..., :1, :])], dim=-2) # b, gs, gs, 4, 1
    world_coord0 = extr0 @ cam_coord0 # b, gs, gs, 4, 1
    cam_coord0_1 = calc_extr_inv(extr1) @ world_coord0 # b, gs, gs, 4, 1; point map 0 in camera 1 coordinate
    cam_coord0_1 = cam_coord0_1[..., :3, :] # b, gs, gs, 3, 1
    
    img_coord0_1 = intr1 @ cam_coord0_1 # b, gs, gs, 3, 1
    depth0_1 = img_coord0_1[..., -1, 0] # b, gs, gs
    img_coord0_1 = img_coord0_1[..., :2, 0] / img_coord0_1[..., 2:, 0] # b, gs, gs, 2
    timer.record_end("geo2match_1_ReProj_matmul")
    timer.record_start("geo2match_2_ReProj_mask")
    img_coord0_1[img_coord0_1.isnan()] = -1.0
    img_coord0_1[img_coord0_1.isinf()] = -1.0
    timer.record_end("geo2match_2_ReProj_mask")
    
    timer.record_start("geo2match_3_ReProj_gridsamp")
    
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
        mode="nearest", padding_mode="zeros", align_corners=False) # b, 1(c), gs, gs
    reproj_depth_samp01 = reproj_depth_samp01.reshape(*depth0_1.shape) # b, gs, gs
    reproj_depth_samp01_mask = torch.nn.functional.grid_sample(
        depth_mask1[:, None],                        # b, 1(c), h, w
        (img_coord0_1 * 2 - 1),                      # b, gs, gs, 2
        mode="nearest", padding_mode="zeros", align_corners=False) # b, 1(c), gs, gs
    reproj_depth_samp01_mask = reproj_depth_samp01_mask.reshape(*depth0_1.shape) # b, gs, gs
    
    timer.record_end("geo2match_3_ReProj_gridsamp")
    timer.record_start("geo2match_4_BatchWiseSamp")
    
    matched_mask = torch.all(torch.stack([
        (reproj_depth_samp01 - depth0_1).abs() < eps, 
        is_reproj_valid, 
    ], dim=-1), dim=-1) # b, gs, gs
    matched_mask = matched_mask & (depth_mask0_grid > 1e-2) & (reproj_depth_samp01_mask > 1e-2)
    
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
    
    timer.record_end("geo2match_4_BatchWiseSamp")
    timer.record_end("geo2match_0_Total")
    timer.report()
    
    return filtered_tracks0 * factor, filtered_tracks1 * factor, filtered_matched_masks


def exhaustive_geo2match(depths: torch.Tensor, depth_masks: torch.Tensor, extrs: torch.Tensor, intrs: torch.Tensor, 
                         eps=0.01, samples_to_keep=10000, grid_size=None):
    """
    Args:
        depths (torch.Tensor): Depth maps of shape (N, 1, H, W)
        depth_masks (torch.Tensor): Masks for valid depth values of shape (N, 1, H, W)
        extrs (torch.Tensor): Camera extrinsics of shape (N, 4, 4), camera-to-world.
        intrs (torch.Tensor): Camera intrinsics of shape (N, 3, 3)
    """
    N = depths.shape[0]
    pairs_x, pairs_y = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                pairs_x.append(i)
                pairs_y.append(j)
    pairs_x = torch.tensor(pairs_x, dtype=torch.int64)
    pairs_y = torch.tensor(pairs_y, dtype=torch.int64)
    depths0, depths1 = depths[pairs_x], depths[pairs_y]
    depth_masks0, depth_masks1 = depth_masks[pairs_x], depth_masks[pairs_y]
    extrs0, extrs1 = extrs[pairs_x], extrs[pairs_y]
    intrs0, intrs1 = intrs[pairs_x], intrs[pairs_y]
    
    depths0, depths1 = depths0.squeeze(dim=1), depths1.squeeze(dim=1)
    depth_masks0, depth_masks1 = depth_masks0.squeeze(dim=1), depth_masks1.squeeze(dim=1)
    
    tracks0, tracks1, visibs = pair_geo2match(depths0, depths1, depth_masks0, depth_masks1, extrs0, extrs1, intrs0, intrs1, 
                                              eps=eps, samples_to_keep=samples_to_keep, grid_size=grid_size)
    
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


def zdepth_to_rdepth(z_depths: torch.Tensor, intrs: torch.Tensor):
    """
    Transform z depth into ray depth.
    Args:
        z_depths (torch.Tensor): Tensor of shape (N, 1, H, W).
        intrs (torch.Tensor): Tensor of shape (N, 3, 3), normalized.
    """
    H, W = z_depths.shape[-2:]
    image_grid = torch.cartesian_prod((torch.arange(H) + 0.5) / H, (torch.arange(W) + 0.5) / W).reshape(H, W, 2).flip(dims=[-1])
    
    pixel_vec = torch.cat([image_grid, torch.ones_like(image_grid[..., :1])], dim=-1)[None] # 1 H W 3
    pixel_vec = (calc_intr_inv(intrs)[:, None, None] @ pixel_vec[..., None])[..., 0] # N H W 3
    pixel_vec = pixel_vec * z_depths.permute(0, 2, 3, 1) # N H W 3
    r_depths = pixel_vec.norm(dim=-1, keepdim=True).permute(0, 3, 1, 2) # N 1 H W
        
    return r_depths


def aug_by_crop(images: torch.Tensor, intrs: torch.Tensor, min_crop_rate=0.0, max_crop_rate=0.2, depths=None, depth_masks=None):
    """Augment images by cropping, along with modifying the ground-truth intrinsics.

    Args:
        images (torch.Tensor): Tensor of shape (V, C, H, W)
        intrs (torch.Tensor): Tensor of shape (V, 3, 3)
        min_crop_rate (float, optional): Minimal rate to crop images. Defaults to 0.0.
        max_crop_rate (float, optional): Maximal rate to crop images. Defaults to 0.2.
    """
    assert min_crop_rate < max_crop_rate and min_crop_rate >= 0.0 and max_crop_rate < 1.0
    h_crop_rate = np.random.rand() * (max_crop_rate - min_crop_rate) + min_crop_rate
    w_crop_rate = np.random.rand() * (max_crop_rate - min_crop_rate) + min_crop_rate
    h, w = images.shape[-2:]
    new_h, new_w = int(h * (1 - h_crop_rate)), int(w * (1 - w_crop_rate))
    left_up_row = (h - new_h) // 2
    left_up_col = (w - new_w) // 2
    right_down_row = h - (h - new_h) // 2
    right_down_col = w - (w - new_w) // 2
    
    images = images[..., left_up_row: right_down_row, left_up_col: right_down_col]
    new_h, new_w = images.shape[-2:]
    intrs[..., 0, 0] *= w / new_w
    intrs[..., 1, 1] *= h / new_h
    if depths is not None:
        depths = depths[..., left_up_row: right_down_row, left_up_col: right_down_col]
    if depth_masks is not None:
        depth_masks = depth_masks[..., left_up_row: right_down_row, left_up_col: right_down_col]
    
    return images, intrs, depths, depth_masks
    
    