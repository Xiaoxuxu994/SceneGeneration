import os
import imageio
import numpy as np
# import utils3d
import open3d as o3d
from PIL import Image
import cv2
from tqdm import tqdm

import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.utils import save_image
# from torchmetrics.image import StructuralSimilarityIndexMeasure
# from torchvision import transforms

from .render_utils_npu import render_gtpose

# %%
""" FOR DEMO """

def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs, center=(0.0, 0.0, 0.0)):
    center = torch.tensor(center, dtype=torch.float32).cuda()

    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)

    extrinsics = []
    intrinsics = []

    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()

        # 相机位置在以 center 为球心的球面上
        offset = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r

        orig = center + offset  # 新的相机位置

        extr = utils3d.torch.extrinsics_look_at(orig, center, torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)

        extrinsics.append(extr)
        intrinsics.append(intr)

    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]

    return extrinsics, intrinsics

from moviepy import VideoFileClip
def gen_demo(GS, center, render_option='gsplat', device="cuda:0", 
                  r=0.5, fov=100,
                  new_w=480, new_h=268, 
                  interp_folder="demo"):
    yaws = torch.linspace(0, 2 * torch.pi, 300)
    pitches = 0.1 * torch.cos(yaws)
    interp_exts_lst, interp_ints_lst = yaw_pitch_r_fov_to_extrinsics_intrinsics(
                                            yaws.tolist(), pitches.tolist(), r, fov, center=center
                                        )
    interp_exts_lst = torch.stack(interp_exts_lst).squeeze()
    interp_ints_lst = torch.stack(interp_ints_lst)
    if render_option == 'gsplat':
        interp_ints_lst[:,0,:] *= new_w
        interp_ints_lst[:,1,:] *= new_h
    
    rend_images_lst, rend_depth_lst = [], []
    for iii in tqdm(range(0,len(interp_exts_lst)), desc="rendering interpolated views"):
        
        results = render_utils.render_gtpose(
                GS,
                resolution=(new_h, new_w),
                extrinsics=interp_exts_lst[iii,:,:],
                intrinsics=interp_ints_lst[iii,:,:],
                render_option='gsplat',
                render_mode='RGB+D',
                sh_degree=0,
                is_infer=True,
                bg_color=(0, 0, 0),
                verbose=False
            )
        
        # results = render_gsplat(GS,
        #                         extrinsics=interp_exts_lst[iii,:,:].float().to(device),
        #                         intrinsics=interp_ints_lst[iii,:,:].float().to(device),
        #                         render_mode="RGB+D",
        #                         resolution=(new_h, new_w )) #  416, 700 
        
        video = results['color']
        depths = results['depth'] 
        rend_images_lst.append(video[0])
        rend_depth_lst.append(depths[0])
        # to_pil(video[0]).save(os.path.join(interp_folder, f'view_{iii}.png'))
        # to_pil(depths[0]).save(os.path.join(interp_depth_folder, f'depth_{iii}.png'))
    depth_min = min(np.min(frame) for frame in rend_depth_lst)
    depth_max = max(np.max(frame) for frame in rend_depth_lst)
    depth_video = []
    for frame in rend_depth_lst:
        frame_2d = frame[:, :, 0] # 去掉最后的 1 维，变成 (H, W)
        frame_norm = 255 * (frame_2d - depth_min) / (depth_max - depth_min + 1e-8) # 归一化到 0~255
        frame_uint8 = frame_norm.astype(np.uint8)
        frame_rgb = np.stack([frame_uint8]*3, axis=-1) # # 转成 (H, W, 3) 
        depth_video.append(frame_rgb)
    imageio.mimsave(os.path.join(interp_folder, f'demo_rgb.mp4'), rend_images_lst, fps=30)
    imageio.mimsave(os.path.join(interp_folder, f'demo_depth.mp4'), depth_video, fps=30)

    clip = VideoFileClip(os.path.join(interp_folder, f'demo_depth.mp4'))
    # clip = clip.resize(width=480)  # 调整宽度，保持比例
    clip.write_gif(os.path.join(interp_folder, f'demo_depth.gif'), fps=30)
    clip = VideoFileClip(os.path.join(interp_folder, f'demo_rgb.mp4'))
    # clip = clip.resize(width=480)  # 调整宽度，保持比例
    clip.write_gif(os.path.join(interp_folder, f'demo_rgb.gif'), fps=30)

# for demo
# from gsplat.rendering import rasterization
# def render_gsplat(gausssian, 
#                   extrinsics: torch.Tensor,
#                   intrinsics: torch.Tensor,
#                   resolution: tuple,
#                   render_mode: str = "RGB+D",
#                   sh_degree: int = 0):

#     means3D = gausssian.get_xyz
#     opacity = gausssian.get_opacity
#     scales = gausssian.get_scaling
#     rotations = gausssian.get_rotation
#     shs = gausssian.get_features
#     shs = shs.view(means3D.shape[0], -1, 3)
    
#     if len(extrinsics.shape) == 4 and len(intrinsics.shape) == 4:
#         extrinsics = extrinsics.squeeze(0)
#         intrinsics = intrinsics.squeeze(0)
    
#     renders_images, _, _ = rasterization(
#             means = means3D,  # [N, 3]
#             quats = rotations,  # [N, 4]
#             scales = scales,  # [N, 3]
#             opacities = opacity.squeeze(),  # [N]
#             sh_degree=sh_degree,
#             colors = shs,  # [N, S, 3]
#             viewmats = extrinsics.unsqueeze(0),  # [C, 4, 4] # 。squeeze(0)
#             Ks = intrinsics.unsqueeze(0),  # [C, 3, 3] # .squeeze(0)
#             width = resolution[1],
#             height = resolution[0],
#             render_mode=render_mode,
#             packed=False,
#             # distributed=world_size > 1,
#         )
#     if renders_images.shape[-1] == 4:
#         colors, depths = renders_images[..., 0:3], renders_images[..., 3:4]
#     else:
#         colors, depths = renders_images, None

#     if depths is None:
#         ret = edict({
#             'color': colors
#             })
#     else:
#         ret = edict({
#                 'color': colors,
#                 'depth': depths,
#             })
    
#     if 'color' not in ret: ret['color'] = []
#     if 'depth' not in ret: ret['depth'] = []
#     ret['color'] = np.clip(ret['color'].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
#     if 'percent_depth' in ret:
#         ret['depth'] = ret['percent_depth'].detach().cpu().numpy()
#     elif 'depth' in ret:
#         ret['depth'] = ret['depth'].detach().cpu().numpy()
#     else:
#         ret['depth'] = None
#     return ret


# 可视化
def depth_to_vis(depths):
    """
    把单通道深度转成可视化的灰度图
    """
    depth_np = depths.detach().cpu().numpy()
    max_val = depth_np.max()
    depth_norm = depth_np  / max_val
    depth_norm[depth_np <= 1e-6] = 0  # 把无效区设为黑色
    depth_vis = (depth_norm * 255).astype(np.uint8)
    return cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

def vis_uv_in_depth(depths, u, v, mask, save_path):
    """
    在深度图上画出可见点
    """
    depth_img = depth_to_vis(depths)

    # 只保留可见点
    u_int = u[mask].detach().cpu().long().numpy()
    v_int = v[mask].detach().cpu().long().numpy()

    for x, y in zip(u_int, v_int):
        if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
            cv2.circle(depth_img, (x, y), 2, (0, 0, 255), -1)

    cv2.imwrite(save_path, depth_img)


def vis_visibility_mask(points, mask, save_path):
    """
    保存可见点到 ply
    """
    pts = points[mask].detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(save_path, pcd)
