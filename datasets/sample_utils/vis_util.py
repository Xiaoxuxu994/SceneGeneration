import os
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
import open3d as o3d

def _img_to_numpy(img):
    if torch.is_tensor(img):
        img = img.detach().cpu()
        if img.ndim == 3:
            img = img.permute(1, 2, 0).numpy()
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255)
        return img.astype(np.uint8)

    if isinstance(img, Image.Image):
        return np.array(img)

    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255)
            img = img.astype(np.uint8)
        return img

    raise TypeError(f"Unsupported image type: {type(img)}")


# def _depth_to_color(depth, mask=None):
#     depth = depth.astype(np.float32)
#     if mask is not None:
#         depth = np.where(mask, depth, 0.0)

#     valid = depth > 0
#     vis = np.zeros_like(depth, dtype=np.float32)
#     if valid.any():
#         d = np.log1p(depth[valid])
#         d = (d - d.min()) / (d.max() - d.min() + 1e-8)
#         vis[valid] = d

#     vis = torch.from_numpy(vis)[None, ...].repeat(3, 1, 1)  # 3,H,W
#     return vis

def _depth_to_color(depth, mask=None):
    depth = depth.astype(np.float32)

    if mask is not None:
        depth = np.where(mask, depth, 0.0)

    depth = np.log10(depth + 1.0)

    depth_t = torch.from_numpy(depth)[None, None, ...]   # 1,1,H,W
    lower = depth_t.reshape(1, -1).min(dim=-1).values[..., None, None, None]
    upper = depth_t.reshape(1, -1).max(dim=-1).values[..., None, None, None]
    depth_t = (depth_t - lower) / (upper - lower + 1e-8)

    depth_t = depth_t.squeeze(0).repeat(3, 1, 1)  # 3,H,W
    return depth_t

def save_ply(points, colors, ply_path):
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

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

def visualize_gt(views, out_dir, prefix="sample"):
    """
    views: list[dict] from BaseDataset.__getitem__()
    saves:
      - {out_dir}/{prefix}.png
      - {out_dir}/{prefix}.ply   # 所有 views 合并后的点云
    """
    os.makedirs(out_dir, exist_ok=True)
    cam_color_ends = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    n_views = len(views)

    vis_rows = []
    all_points = []
    all_colors = []
    camera_geoms = []

    for i, view in enumerate(views):
        img_np = _img_to_numpy(view["img"])            # H,W,3
        depth = view["depthmap"].astype(np.float32)    # H,W
        mask = view["valid_mask"].astype(bool)         # H,W
        pts3d = view["pts3d"].astype(np.float32)       # H,W,3
        extr = np.asarray(view["camera_pose"], dtype=np.float32)
        K = np.asarray(view["camera_intrinsics"], dtype=np.float32)

        # 收集所有 view 的点云
        all_points.append(pts3d[mask])   # N,3
        all_colors.append(img_np[mask])  # N,3

        # 可视化图
        img_t = TF.to_tensor(img_np)
        depth_t = _depth_to_color(depth, mask=mask)
        mask_t = torch.from_numpy(mask.astype(np.float32))[None].repeat(3, 1, 1)

        vis_list = [img_t, depth_t, mask_t]

        # 如果有 normal，就一起可视化
        if view['normal'] is not None:
            normal = np.asarray(view["normal"], dtype=np.float32)  # H,W,3 or 3,H,W

            if normal.ndim == 3 and normal.shape[0] == 3 and normal.shape[-1] != 3:
                normal = np.transpose(normal, (1, 2, 0))  # -> H,W,3

            if normal.ndim == 2:
                normal = normal[..., None]
            if normal.shape[-1] > 3:
                normal = normal[..., :3]

            # 重新归一化，防止 resize 后长度变化
            norm = np.linalg.norm(normal, axis=-1, keepdims=True)
            normal = normal / np.clip(norm, 1e-8, None)

            # [-1,1] -> [0,1]
            normal_vis = (np.clip(normal, -1.0, 1.0) + 1.0) * 0.5

            # 如果有 mask，无效区域置黑
            normal_vis[~mask] = 0.0

            normal_t = torch.from_numpy(normal_vis).permute(2, 0, 1).float()
            vis_list.append(normal_t)

        row = torch.cat(vis_list, dim=-1)  # 3,H,(3 or 4)*W
        vis_rows.append(row)

        # 可视化camera
        cam_color = (
            cam_color_ends[0] * (n_views - i) / n_views
            + cam_color_ends[1] * i / n_views
        )

        cam_frustum = create_camera_frustum(
            extr[:3, 3],
            extr[:3, :3],
            color=cam_color,
            scale=0.05,
        )
        camera_geoms.append(cam_frustum)

    # 保存总可视化图
    vis = torch.cat(vis_rows, dim=1)  # 3,V*H,(3 or 4)*W
    png_path = os.path.join(out_dir, f"{prefix}.png")
    torchvision.utils.save_image(vis, png_path)
    print(f"[INFO] visualization saved to: {png_path}")

    # 保存合并后的总点云
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    ply_path = os.path.join(out_dir, f"{prefix}.ply")
    save_ply(all_points, all_colors, ply_path)

    print(f"[INFO] merged ply saved to: {ply_path}")

    # 保存相机
    if len(camera_geoms) > 0:
        merged_cameras = o3d.geometry.PointCloud()
        for cam_frustum in camera_geoms:
            merged_cameras += cam_frustum

        cam_ply_path = os.path.join(out_dir, f"{prefix}_cam.ply")
        o3d.io.write_point_cloud(cam_ply_path, merged_cameras)
        print(f"[INFO] camera ply saved to: {cam_ply_path}")

        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
        scene_pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64) / 255.0)
        scene_pcd += merged_cameras

        scene_ply_path = os.path.join(out_dir, f"{prefix}_scene.ply")
        o3d.io.write_point_cloud(scene_ply_path, scene_pcd)
        print(f"[INFO] scene+camera ply saved to: {scene_ply_path}")

