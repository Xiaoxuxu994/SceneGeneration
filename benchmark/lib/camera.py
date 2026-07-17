import os
import torch
import numpy as np
import open3d as o3d

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


def export_cam_as_ply(path: str, extrs: torch.Tensor):
    """Export batched point map and cameras to .ply file for visualization.

    Args:
        path (str): Name information, typically optimized steps of the model.
        extrs (torch.Tensor): World-from-camera matrices, Tensor of shape (B, V, 4, 4).
    """
    
    cam_color_ends = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    
    batch_size, n_views, _, _ = extrs.shape
    
    if extrs.shape[2] == 3:
        new_extrs = torch.eye(4, device=extrs.device).reshape(1, 1, 4, 4).repeat(batch_size, n_views, 1, 1)
        new_extrs[:, :, :3, :] = extrs
        extrs = new_extrs
    
    for b in range(batch_size):
        ### Create camera object.
        # scene_cameras = []

        # merged_cameras = o3d.geometry.LineSet()
        merged_cameras = o3d.geometry.PointCloud()
        for v in range(n_views):
            extr = extrs[b, v]
            cam_frustum = create_camera_frustum(extr[:3, 3].detach().cpu().numpy(), extr[:3, :3].detach().cpu().numpy(), 
                                                color=cam_color_ends[0] * (n_views - v) / n_views + cam_color_ends[1] * v / n_views, 
                                                scale=0.05)
            # cam_frustum.paint_uniform_color()
            # scene_cameras.append(cam_frustum)
            merged_cameras += cam_frustum
            # o3d.io.write_line_set(path_prefix + "_cam{}".format(v) + "_batch{}.ply".format(b), cam_frustum)
        
        # o3d.io.write_line_set(path_prefix + "_cam" + "_batch{}.ply".format(b), merged_cameras)
        o3d.io.write_point_cloud(path, merged_cameras)