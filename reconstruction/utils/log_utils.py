import torch
import numpy as np
import open3d as o3d

from .common import calc_extr_inv, trans_geo_to_first_frame_and_norm


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
    
    points_cam_world = (rotation[None] @ points_cam[..., None])[..., 0] + position[None]
    
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4], 
        [1, 2], [2, 3], [3, 4], [4, 1], 
    ])
    
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
                               extrs: torch.Tensor, confs: torch.Tensor=None, conf_thre=0.5):
    """Export batched point map and cameras to .ply file for visualization.

    Args:
        path_prefix (str): Name information, typically optimized steps of the model.
        point_maps (torch.Tensor): Tensor of shape (B, V, H, W, 3).
        images (torch.Tensor): Tensor of shape (B, V, 3, H, W).
        extrs (torch.Tensor): Tensor of shape (B, V, 4, 4).
        confs (torch.Tensor): Tensor of shape (B, V, H, W, 1)
    """
    assert point_maps.ndim == 5, "This function is designed to process batched data."
    cam_color_ends = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    point_maps, extrs, _ = trans_geo_to_first_frame_and_norm(point_maps, extrs)
    
    extrs = calc_extr_inv(extrs)
    
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
        o3d.io.write_point_cloud(path_prefix + "_batch{}.ply".format(b), pcd)
        
        ### Create camera object.
        n_views = len(point_maps[b])
        merged_cameras = o3d.geometry.PointCloud()
        for v in range(n_views):
            extr = extrs[b, v]
            cam_frustum = create_camera_frustum(extr[:3, 3].detach().cpu().numpy(), extr[:3, :3].detach().cpu().numpy(), 
                                                color=cam_color_ends[0] * (n_views - v) / n_views + cam_color_ends[1] * v / n_views, 
                                                scale=0.05)
            merged_cameras += cam_frustum
        o3d.io.write_point_cloud(path_prefix + "_cam" + "_batch{}.ply".format(b), merged_cameras)
