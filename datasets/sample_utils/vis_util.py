import os
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
import open3d as o3d
import matplotlib

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

# def _depth_to_color(depth, mask=None):
#     depth = depth.astype(np.float32)

#     if mask is not None:
#         depth = np.where(mask, depth, 0.0)

#     depth = np.log10(depth + 1.0)

#     depth_t = torch.from_numpy(depth)[None, None, ...]   # 1,1,H,W
#     lower = depth_t.reshape(1, -1).min(dim=-1).values[..., None, None, None]
#     upper = depth_t.reshape(1, -1).max(dim=-1).values[..., None, None, None]
#     depth_t = (depth_t - lower) / (upper - lower + 1e-8)

#     depth_t = depth_t.squeeze(0).repeat(3, 1, 1)  # 3,H,W
#     return depth_t

def _get_cmap(name):
    # 兼容新老 matplotlib（3.7+ 用 colormaps，老版用 cm.get_cmap）
    try:
        return matplotlib.colormaps[name]
    except Exception:
        import matplotlib.cm as cm
        return cm.get_cmap(name)


def _depth_to_color(depth, mask=None, cmap_name="Spectral", invert=False, use_log=True):
    """
    depth: (H,W) numpy
    mask:  (H,W) bool，可选；只在有效区归一化，无效区置黑
    cmap_name: "turbo"(推荐,彩虹) / "Spectral" / "viridis" / "magma" / "jet"
    invert: 翻转近远配色。默认近处=暖色(红/黄)、远处=冷色(蓝)。
    use_log: log 压缩深度动态范围（室内外尺度差大时更均匀好看）
    返回: torch.FloatTensor (3,H,W)，值域 [0,1]，可直接进 torchvision.save_image
    """
    depth = np.asarray(depth, dtype=np.float32)
    H, W = depth.shape

    if mask is None:
        mask = depth > 0
    mask = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0)

    norm = np.zeros((H, W), dtype=np.float32)
    if mask.any():
        d = depth[mask]
        if use_log:
            d = np.log10(d + 1e-6)
        # 分位数裁剪，抗离群点（比 min()/max() 稳，远处个别噪点不会毁掉配色）
        lo = np.percentile(d, 2)
        hi = np.percentile(d, 98)
        dn = np.clip((d - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        # turbo: 低值=蓝、高值=红。深度小=近，默认翻转让"近处=暖色"，更符合直觉。
        if not invert:
            dn = 1.0 - dn
        norm[mask] = dn

    cmap = _get_cmap(cmap_name)
    colored = np.asarray(cmap(norm))[..., :3].astype(np.float32)   # (H,W,3) in [0,1]
    colored[~mask] = 0.0                                            # 无效区黑

    return torch.from_numpy(colored).permute(2, 0, 1).float()      # (3,H,W)

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



# ============================================================================
# 归一化后点云可视化 —— 追加到 datasets/sample_utils/vis_util.py 末尾
# 严格复刻 pi3/models/loss.py 的 Pi3Loss.prepare_gt，并对比多种归一化口径。
#
# norm_mode:
#   "camera"      = 现状：原点=第一帧相机，norm_factor=mean(到相机距离)
#                   → 物体被相机距离主导而压扁
#   "centered"    = 减中心 + 除norm_factor(到中心mean)：物体撑开且被移到原点
#                   → 但 global point loss 在第一帧系比较，移动原点有平移偏差风险
#   "scale_only"  = 【选择A，推荐】算 norm_factor 时用"到中心距离"(物体尺度正确)，
#                   但点只除 norm_factor、不减 center → 物体撑到正确大小，
#                   位置仍在第一帧相机前方，global 比较基准不变，混训自洽
#   "centered_p90"= 减中心 + 除p90(到中心) 基准参照
#
# 复用 vis_util.py 已定义的 _img_to_numpy / save_ply / create_camera_frustum。
# ============================================================================
import os
import numpy as np


def _se3_inverse_np(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _homog(pts):
    ones = np.ones((*pts.shape[:-1], 1), dtype=pts.dtype)
    return np.concatenate([pts, ones], axis=-1)


def _normalize_like_prepare_gt(all_pts_world, poses_c2w, norm_mode="camera"):
    """
    复刻 prepare_gt（含选择A）：
      1) 转到第一帧相机系
      2) 算 norm_factor（参考点/统计量见各模式）
      3) 归一化点 + 相机平移
    返回 norm_pts_per_view, cam_centers_norm, poses_cam0, info
    """
    poses = np.asarray(poses_c2w, dtype=np.float64).copy()
    w2c0 = _se3_inverse_np(poses[0])

    # 1) 所有点 -> 第一帧相机系
    pts_cam0 = []
    for p in all_pts_world:
        if p.shape[0] == 0:
            pts_cam0.append(p.astype(np.float64)); continue
        ph = _homog(p.astype(np.float64))
        pts_cam0.append((w2c0 @ ph.T).T[:, :3])
    cat = np.concatenate([p for p in pts_cam0 if p.shape[0] > 0], axis=0)

    # 2) norm_factor + 是否平移
    pts_center = np.median(cat, axis=0)        # 点云中心（到中心距离都用它）
    if norm_mode == "camera":
        ref = np.zeros(3)                       # 参考点=相机(原点)
        nf = np.linalg.norm(cat - ref, axis=-1).mean()
        sub_center = np.zeros(3)                # 不平移
    elif norm_mode == "centered":
        ref = pts_center
        nf = np.linalg.norm(cat - ref, axis=-1).mean()
        sub_center = pts_center                 # 平移：减中心
    elif norm_mode == "scale_only":            # 选择A
        ref = pts_center
        nf = np.linalg.norm(cat - ref, axis=-1).mean()  # norm_factor 用到中心距离(物体尺度正确)
        sub_center = np.zeros(3)                # 但点不减中心(原点不变)
    elif norm_mode == "centered_p90":
        ref = pts_center
        nf = np.percentile(np.linalg.norm(cat - ref, axis=-1), 90)
        sub_center = pts_center
    else:
        raise ValueError(norm_mode)
    nf = float(nf) + 1e-8

    # 诊断指标
    dc = np.linalg.norm(cat - pts_center, axis=-1)
    extent_p90 = float(np.percentile(dc, 90))
    compress_ratio = extent_p90 / nf

    # 3) 归一化点 + 相机
    norm_pts_per_view = []
    for pc in pts_cam0:
        if pc.shape[0] == 0:
            norm_pts_per_view.append(pc); continue
        norm_pts_per_view.append(((pc - sub_center) / nf).astype(np.float32))

    poses_cam0 = np.einsum('ij,njk->nik', w2c0, poses)
    cam_centers = poses_cam0[:, :3, 3]
    cam_centers_norm = ((cam_centers - sub_center) / nf).astype(np.float32)

    info = dict(norm_factor=nf, extent_p90=extent_p90, compress_ratio=compress_ratio,
                norm_mode=norm_mode, sub_center=sub_center, pts_center=pts_center)
    return norm_pts_per_view, cam_centers_norm, poses_cam0, info


def visualize_gt_normalized(views, out_dir, prefix="sample", norm_mode="camera"):
    """输出按某种归一化处理后的点云+相机 PLY。
    输出: {prefix}_norm_{mode}.ply 与 {prefix}_norm_{mode}_scene.ply"""
    os.makedirs(out_dir, exist_ok=True)
    cam_color_ends = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    n_views = len(views)

    all_pts_world, all_colors, poses_c2w = [], [], []
    for view in views:
        mask = view["valid_mask"].astype(bool)
        pts3d = view["pts3d"].astype(np.float32)
        img_np = _img_to_numpy(view["img"])
        all_pts_world.append(pts3d[mask])
        all_colors.append(img_np[mask])
        poses_c2w.append(np.asarray(view["camera_pose"], dtype=np.float64))
    poses_c2w = np.stack(poses_c2w, axis=0)

    norm_pts_per_view, cam_centers_norm, poses_cam0, info = _normalize_like_prepare_gt(
        all_pts_world, poses_c2w, norm_mode=norm_mode)

    print(f"[norm:{norm_mode}] norm_factor={info['norm_factor']:.4f}  "
          f"extent_p90={info['extent_p90']:.4f}  compress_ratio={info['compress_ratio']:.4f}")

    norm_pts = np.concatenate([p for p in norm_pts_per_view if p.shape[0] > 0], axis=0)
    colors = np.concatenate(all_colors, axis=0)

    ply_path = os.path.join(out_dir, f"{prefix}_norm_{norm_mode}.ply")
    save_ply(norm_pts, colors, ply_path)
    print(f"[INFO] normalized ply saved to: {ply_path}")

    try:
        import open3d as o3d
        merged = o3d.geometry.PointCloud()
        for i in range(n_views):
            R = poses_cam0[i][:3, :3]
            pos = cam_centers_norm[i]
            cam_color = (cam_color_ends[0] * (n_views - i) / n_views
                         + cam_color_ends[1] * i / n_views)
            merged += create_camera_frustum(pos, R, color=cam_color, scale=0.05)
        scene = o3d.geometry.PointCloud()
        scene.points = o3d.utility.Vector3dVector(norm_pts.astype(np.float64))
        scene.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        scene += merged
        scene_path = os.path.join(out_dir, f"{prefix}_norm_{norm_mode}_scene.ply")
        o3d.io.write_point_cloud(scene_path, scene)
        print(f"[INFO] normalized scene+camera ply saved to: {scene_path}")
    except Exception as e:
        print(f"[WARN] camera ply skipped: {e}")

    return info


def visualize_gt_norm_compare(views, out_dir, prefix="sample"):
    """一次输出 camera / centered / scale_only / centered_p90 四种，便于对比。
    重点对比：
      camera     → 物体压扁(compress_ratio<<1)、在原点前方
      scale_only → 物体撑到正确大小、仍在原点前方（选择A：尺度对、原点不变）
      centered   → 物体撑开、被移到原点（位置也变了）
    """
    modes = ["camera", "scale_only", "centered", "centered_p90"]
    infos = {}
    for m in modes:
        infos[m] = visualize_gt_normalized(views, out_dir, prefix, norm_mode=m)
    print("\n[compare] mode -> norm_factor / compress_ratio")
    for m in modes:
        print(f"  {m:14s} nf={infos[m]['norm_factor']:8.4f}  compress_ratio={infos[m]['compress_ratio']:.4f}")
    print("\n说明：scale_only 与 centered 的 norm_factor/compress_ratio 相同（尺度一样），")
    print("      差别只在 scale_only 不减中心 → 物体仍在第一帧相机前方，global 比较基准不变。")
    return infos
