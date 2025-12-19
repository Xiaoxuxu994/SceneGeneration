import os
import torch
import numpy as np
import open3d as o3d

from einops import rearrange

from .platform import DEVICE_NAME
from .common import calc_extr_inv, calc_intr_inv


def save_pcds_as_ply(path_prefix: str, point_maps: torch.Tensor, images: torch.Tensor):
    """Save inferenced point maps into .ply format point cloud for visualization.  

    Args:
        path_prefix (str): _description_
        point_maps (torch.Tensor): Predicted point map, tensor of shape (B, V, H, W, 3)
        images (torch.Tensor): Images to register color to point map, tensor of shape (B, V, 3, H, W)
    """
    
    assert point_maps.ndim == 5, "This function is designed to process batched data."
    
    batch_size = len(point_maps)
    for b in range(batch_size):
        pcd = o3d.geometry.PointCloud()
        points = np.array(point_maps[b].detach().reshape(-1, 3).cpu())
        colors = np.array(images[b].detach().permute(0, 2, 3, 1).reshape(-1, 3).cpu())
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(path_prefix + "_batch{}.ply".format(b), pcd)


def create_camera_frustum(position, rotation, scale=0.5, fov=60, aspect_ratio=1.0, color=None, point_per_edge=100):
    """_summary_

    Args:
        position (_type_): camera-to-world
        rotation (_type_): camera-to-world
        scale (float, optional): _description_. Defaults to 0.5.
        fov (int, optional): _description_. Defaults to 60.
        aspect_ratio (float, optional): _description_. Defaults to 1.0.
        color (_type_, optional): _description_. Defaults to None.
        point_per_edge (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    fov_rad = np.deg2rad(fov)
    half_height = np.tan(fov_rad / 2) * scale
    half_width = half_height * aspect_ratio
    z = scale
    
    points_cam = np.array([
        [0, 0, 0], 
        [-half_width, -half_height, z], 
        [half_width, -half_height, z], 
        [half_width, half_height, z], 
        [-half_width, half_height, z], 
    ])
    
    # R_inv = np.linalg.inv(rotation)
    points_cam_world = (rotation[None] @ points_cam[..., None])[..., 0] + position[None]
    
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4], 
        [1, 2], [2, 3], [3, 4], [4, 1], 
    ])
    
    # line_set = o3d.geometry.LineSet()
    # line_set.points = o3d.utility.Vector3dVector(points_cam_world)
    # line_set.lines = o3d.utility.Vector2iVector(lines)
    # if color is not None:
    #     line_set.colors = o3d.utility.Vector3dVector(np.array([color for _ in range(len(lines))]))
    # return line_set

    point_set = o3d.geometry.PointCloud()
    points = []
    for line in lines:
        for p in np.linspace(0, 1, num=point_per_edge):
            point = points_cam_world[line[0]] * p + points_cam_world[line[1]] * (1 - p)
            points.append(point)
    point_set.points = o3d.utility.Vector3dVector(np.stack(points, axis=0))
    point_set.colors = o3d.utility.Vector3dVector(np.stack([color for _ in range(len(points))], axis=0))
    return point_set


def export_pcds_and_cam_as_ply(path_prefix: str, point_maps: torch.Tensor, images: torch.Tensor, 
                               extrs: torch.Tensor, intrs: torch.Tensor, confs: torch.Tensor=None, conf_thre=0.5):
    """Export batched point map and cameras to .ply file for visualization.

    Args:
        path_prefix (str): Name information, typically optimized steps of the model.
        point_maps (torch.Tensor): Tensor of shape (B, V, H, W, 3).
        images (torch.Tensor): Tensor of shape (B, V, 3, H, W).
        extrs (torch.Tensor): Tensor of shape (B, V, 4, 4).
        intrs (torch.Tensor): Tensor of shape (B, V, 3, 3).
        confs (torch.Tensor): Tensor of shape (B, V, H, W, 1)
    """
    assert point_maps.ndim == 5, "This function is designed to process batched data."
    cam_color_ends = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    extrs = torch.linalg.inv(extrs)
    
    batch_size = len(point_maps)
    
    for b in range(batch_size):
        
        ### Create point cloud object.
        pcd = o3d.geometry.PointCloud()
        points = np.array(point_maps[b].detach().reshape(-1, 3).cpu())
        colors = np.array(images[b].detach().permute(0, 2, 3, 1).reshape(-1, 3).cpu())
        if confs is not None:
            conf_mask = (confs[b].detach().reshape(-1) > conf_thre).cpu()
            points = points[conf_mask]
            colors = colors[conf_mask]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # scene_objects.append(pcd)
        o3d.io.write_point_cloud(path_prefix + "_batch{}.ply".format(b), pcd)
        
        ### Create camera object.
        # scene_cameras = []
        n_views = len(point_maps[b])
        # merged_cameras = o3d.geometry.LineSet()
        merged_cameras = o3d.geometry.PointCloud()
        for v in range(n_views):
            extr = extrs[b, v]
            intr = intrs[b, v]
            cam_frustum = create_camera_frustum(extr[:3, 3].detach().cpu().numpy(), extr[:3, :3].detach().cpu().numpy(), 
                                                color=cam_color_ends[0] * (n_views - v) / n_views + cam_color_ends[1] * v / n_views, 
                                                scale=0.05)
            # cam_frustum.paint_uniform_color()
            # scene_cameras.append(cam_frustum)
            merged_cameras += cam_frustum
            # o3d.io.write_line_set(path_prefix + "_cam{}".format(v) + "_batch{}.ply".format(b), cam_frustum)
        
        # o3d.io.write_line_set(path_prefix + "_cam" + "_batch{}.ply".format(b), merged_cameras)
        o3d.io.write_point_cloud(path_prefix + "_cam" + "_batch{}.ply".format(b), merged_cameras)


def depth2pointmap(depths: torch.Tensor, extrs: torch.Tensor, intrs: torch.Tensor, device=DEVICE_NAME):
    """Back-project depth maps into 3D point maps with given cameras. 

    Args:
        depths (torch.Tensor): Depth maps, tensor of shape (B, 1, H, W).
        extrs (torch.Tensor): Extrinsics (world-to-camera), tensor of shape (B, 4, 4).
        intrs (torch.Tensor): Intrinsics (not normalized, assuming c corresponds to the image center), tensor of shape (B, 3, 3).
        device (str): The default device to perform operations on.
    
    Returns:
        torch.Tensor: The back-projected point maps, tensor of shape (B, H, W, 3).
    """
    B, C, H, W = depths.shape
    assert depths.ndim == 4 and C == 1
    
    intr_normalizer = torch.tensor([W, H, 1], dtype=torch.float32, device=device).reshape(1, 3, 1)
    intrs = intrs / intr_normalizer
    
    grids = torch.cartesian_prod((torch.arange(W, device=device) + 0.5) / W, (torch.arange(H, device=device) + 0.5) / H)
    grids = grids.reshape(1, W, H, 2).permute(0, 2, 1, 3).expand(B, -1, -1, -1) # B H W 2
    orth_depths = torch.cat([grids, torch.ones_like(grids[..., :1])], dim=-1) * rearrange(depths, "n c h w -> n h w c") # b h w 3

    # intrs_inv = torch.linalg.inv(intrs) # b, 3, 3
    intrs_inv = calc_intr_inv(intrs)
    points_local = torch.matmul(intrs_inv[..., None, None, :, :], orth_depths[..., None]) # b h w 3 1
    
    # extrs_inv = torch.linalg.inv(extrs) # b, 4, 4
    extrs_inv = calc_extr_inv(extrs)
    
    orth_points_local = torch.cat([points_local, torch.ones_like(points_local[..., :1, :])], dim=-2) # b h w 4 1
    points_world = torch.matmul(extrs_inv[..., None, None, :, :], orth_points_local) # b h w 4 1
    points_world = points_world[..., :3, 0] / points_world[..., 3:, 0] # b h w 3
    
    return points_world


def localpoint2global(points: torch.Tensor, extrs: torch.Tensor):
    """Align local point map under the global coordinate.

    Args:
        points (torch.Tensor): Local point map, tensor of shape (..., h, w, 3).
        extrs (torch.Tensor): World-to-camera transform, tensor of shape (..., 4, 4)
    """
    points_homo = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1).unsqueeze(-1) # ..., h, w, 4, 1
    points_world_homo = calc_extr_inv(extrs)[..., None, None, :, :] @ points_homo
    
    return points_world_homo[..., :3, 0]


def pointmap2pixel(points: torch.Tensor, extrs: torch.Tensor, intrs: torch.Tensor, device=DEVICE_NAME):
    """Project point maps into image coordinate with given cameras. 

    Args:
        points (torch.Tensor): Point maps, tensor of shape (B, N, 3).
        extrs (torch.Tensor): Extrinsics (world-to-camera), tensor of shape (B, 4, 4).
        intrs (torch.Tensor): Intrinsics (not normalized, assuming c corresponds to the image center), tensor of shape (B, 3, 3).
        device (str): The default device to perform operations on.
    
    Returns:
        torch.Tensor: The back-projected point maps, tensor of shape (B, N, 2).
    """
    points_world_orth = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    points_local_orth = torch.matmul(extrs.unsqueeze(-3), points_world_orth.unsqueeze(-1)) # B N 4 1
    points_pixel_orth = torch.matmul(intrs.unsqueeze(-3), points_local_orth[..., :3, :] / points_local_orth[..., 3:, :])[..., 0] # B N 3
    points_pixel      = points_pixel_orth[..., :2] / points_pixel_orth[..., 2:]
    return points_pixel

class Depth2PointMap(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, depths: torch.Tensor, extrs: torch.Tensor, intrs: torch.Tensor):
        ctx.save_for_backward(depths, extrs, intrs)
        return depth2pointmap(depths, extrs, intrs)
    
    @staticmethod
    def backward(ctx, d_l_points: torch.Tensor):
        
        with torch.enable_grad():
            depths, extrs, intrs = ctx.saved_tensors
            depths = depths.detach().requires_grad_()
            extrs = extrs.detach().requires_grad_()
            intrs = intrs.detach().requires_grad_()
            points_world = depth2pointmap(depths, extrs, intrs)
            # d_points_depths, d_points_extrs, d_points_intrs = torch.autograd.grad(points_world, [depths, extrs, intrs]) # b 1 h w
            grads = torch.autograd.grad(points_world, [depths, extrs, intrs], grad_outputs=d_l_points)
            d_l_depths = grads[0] # b 1 h w
            d_l_extrs = grads[1]
            d_l_intrs = grads[2]
        
        # print("[ DEBUG ] d_l_points.shape: {}, d_l_points norm sum: {}".format(
        #     d_l_points.shape, d_l_points.norm(dim=-1, keepdim=True).sum().item()))
        # print("[ DEBUG ] d_points_depths.shape: {}, d_points_depths norm median: {}, max: {}".format(
        #     d_l_depths.shape, d_l_depths.norm(dim=-3, keepdim=True).median().item(), d_l_depths.norm(dim=-3, keepdim=True).max().item()))
        
        d_l_depths_normed = d_l_depths / (d_l_depths.norm(dim=-3, keepdim=True)) * \
            d_l_points.norm(dim=-1).reshape(*d_l_depths.shape)
        d_l_depths_normed[d_l_depths_normed.isnan()] = 0.0
        
        return d_l_depths_normed, d_l_extrs, d_l_intrs
            
