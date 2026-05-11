import torch
import numpy as np
import open3d as o3d
from torchvision.utils import draw_keypoints

from utils.image_utils import gray2hot
from utils.common import calc_extr_inv, calc_intr_inv
from .common import get_tracking_idx_by_pair

def single_depth_intr_to_pcd(output_path: str, image: torch.Tensor, depth: torch.Tensor, intr: torch.Tensor, device="cpu"):
    """
    Args:
        output_path: Output file path.
        image: Tensor of shape (H, W, 3)
        depth: Tensor of shape (H, W)
        intr: Tensor of shape (3, 3)
    """
    H, W = depth.shape
    assert image.shape[0] == H and image.shape[1] == W and image.shape[2] == 3
    grid_x, grid_y = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H) # (H, W), (H, W)
    img_coord = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1) # H, W, 3
    img_coord = torch.from_numpy(img_coord).to(device=device, dtype=torch.float32)
    
    img_coord = (img_coord * depth[..., None]).unsqueeze(-1) # H, W, 3, 1
    cam_coord = calc_intr_inv(intr) @ img_coord # H, W, 3, 1
    
    points = cam_coord.reshape(H * W, 3).cpu().numpy()
    colors = image.reshape(H * W, 3).cpu().numpy()
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pointcloud)


def depth_maps_to_pcd(output_path: str, images: torch.Tensor, depths: torch.Tensor, 
                      extrs: torch.Tensor, intrs: torch.Tensor, device="cpu"):
    """
    Args:
        output_path: Output file path.
        images: Tensor of shape (N, H, W, 3)
        depths: Tensor of shape (N, H, W)
        extrs: Tensor of shape (N, 4, 4), camera to world
        intrs: Tensor of shape (N, 3, 3)
    """
    B, H, W = depths.shape
    assert (images.shape[0] == B and images.shape[1] == H and images.shape[2] == W and 
            images.shape[3] == 3 and images.ndim == 4)
    grid_x, grid_y = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H) # (H, W), (H, W)
    img_coord = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)[None] # 1, H, W, 3
    img_coord = torch.from_numpy(img_coord).to(device=device, dtype=torch.float32)
    img_coord = torch.cat([img_coord for _ in range(B)]) # B H W 3
    extrs = extrs[:, None, None] # B 1 1 4 4
    intrs = intrs[:, None, None] # B 1 1 3 3
    
    img_coord = (img_coord * depths[..., None]).unsqueeze(-1) # B, H, W, 3, 1
    cam_coord = calc_intr_inv(intrs) @ img_coord # B, H, W, 3, 1
    cam_coord = torch.cat([cam_coord, torch.ones_like(cam_coord[..., :1, :])], dim=-2) # B, H, W, 4, 1
    world_coord = extrs @ cam_coord # B, H, W, 4, 1
    
    points = world_coord[..., :3, 0].reshape(-1, 3).cpu().numpy()
    colors = images.reshape(-1, 3).cpu().numpy()
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pointcloud)


def vis_batch_data(data, keypoint_radius=5):
    images = data["input_images"][0] # v c h w
    depths = data["input_depths"][0] # v c h w
    depths = gray2hot(depths)
    depth_masks = data["input_depth_masks"][0] if data["input_depth_masks"] is not None else None # v c h w
    depth_masks = gray2hot(depth_masks) if depth_masks is not None else torch.zeros_like(depths)
    tracks0 = data["input_tracks0"][0] # v k 2
    tracks1 = data["input_tracks1"][0] # v k 2
    visibs0 = data["input_visibs0"][0] if data["input_visibs0"] is not None else None # v k
    visibs1 = data["input_visibs1"][0] if data["input_visibs1"] is not None else None # v k
    
    def comb_samps(image0, image1, depth0, depth1, depth_mask0, depth_mask1, track0, track1, visib0, visib1):
        if visibs0 is not None:
            track0 = track0[visib0]
            track1 = track1[visib1]
        track_img0 = draw_keypoints(image0, track0[None], colors="red", radius=keypoint_radius)
        track_img1 = draw_keypoints(image1, track1[None], colors="red", radius=keypoint_radius)
        return torch.cat([image0, image1, track_img0, track_img1, depth0, depth1, depth_mask0, depth_mask1], dim=-1)
    
    V = images.shape[0]
    all_lines = []
    for v in range(1, V):
        track_idx = get_tracking_idx_by_pair(0, v)
        all_lines.append(comb_samps(
            images[0], images[v], 
            depths[0], depths[v], 
            depth_masks[0], depth_masks[v], 
            tracks0[track_idx], tracks1[track_idx], 
            visibs0[track_idx] if visibs0 is not None else None, visibs1[track_idx] if visibs1 is not None else None, 
        ))
    sample_vis_img = torch.cat(all_lines, dim=-2)
    return sample_vis_img