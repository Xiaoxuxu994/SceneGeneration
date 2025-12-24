import os
import sys
import math
import time
import numpy as np
import open3d as o3d
import shutil, pathlib
from pathlib import Path
import random
import gc
import cv2
from sklearn.decomposition import PCA
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

DEBUG = False
if DEBUG:
    out_dir = "outputs_TRELLISdev/test_results/test_differentiable_projection"
    os.makedirs(out_dir, exist_ok=True)

def clean_mem():
    gc.collect()
    torch_npu.npu.empty_cache()

def seed_everything(seed=42):
    random.seed(seed)                    
    np.random.seed(seed)            
    torch.manual_seed(seed)         
    torch_npu.npu.manual_seed(seed)     
    # torch_npu.npu.manual_seed_all(seed)      
    # torch.backends.cudnn.deterministic = True  
    # torch.backends.cudnn.benchmark = False     

def log_memory_usage(stage=""):
    """
    Logs the memory usage on the GPU at a given stage.
    
    Args:
        stage (str): A string to indicate the stage of computation.
    """
    allocated = torch_npu.npu.memory_allocated()
    cached = torch_npu.npu.memory_cached()
    
    print(f"[{stage}] Memory allocated: {allocated / 1024 ** 2:.2f} MB")
    print(f"[{stage}] Memory cached: {cached / 1024 ** 2:.2f} MB")

import torch.distributed as dist
def rank_barrier(msg="sync"):
    torch_npu.npu.synchronize()
    if dist.is_initialized():
        dist.barrier()
    print(f"[Rank {dist.get_rank()}] {msg} @ {time.time()}")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')
    

def nonzero_mean_feature(features):
    nonzero_mask = (features.abs().sum(dim=-1) > 0)  # [V, N]，True 表示该视角下该点可用
    mask = nonzero_mask.unsqueeze(-1)  # [V, N, 1]
    
    features_sum = (features * mask).sum(dim=0)  # [N, C]
    valid_counts = mask.sum(dim=0)               # [N, 1]
    
    print("valid_counts: ", valid_counts)

    mean_feature = features_sum / torch.clamp(valid_counts, min=1.0)  # [N, C] 
    mean_feature[valid_counts.squeeze(-1) == 0] = 0  # 或用 torch.nan , valid_counts == 0 设置为 NaN 或保持为 0
    
    return mean_feature

def first_nonzero_feature(x):
    """
    x: Tensor of shape [N, M, C]
    Returns: Tensor of shape [M, C], each row is the first non-zero vector in N for each M.
    """
    N, M, C = x.shape

    # 判断哪些向量是全零：结果是 [N, M]
    is_nonzero = (x != 0).any(dim=2)  # True 表示这个 [C] 特征不是全 0

    # 找每个 M 上，第一个非全零的索引（如果全是 0，则默认是 0）
    first_nonzero_indices = is_nonzero.float().cumsum(dim=0).eq(1).float().argmax(dim=0)  # shape [M]

    # 构建索引，从 [N, M, C] 中挑出每个 M 对应的第一个非零特征
    batch_idx = first_nonzero_indices  # shape [M]
    point_idx = torch.arange(M, device=x.device)

    output = x[batch_idx, point_idx]  # shape [M, C]
    return output 

def nearest_divisible_by_14(width, height):
    # 找到小于等于宽度且最接近的能被14整除的值
    new_width = (width // 14) * 14
    # 找到小于等于高度且最接近的能被14整除的值
    new_height = (height // 14) * 14
    return new_width, new_height

def print_memory_usage(tag=""):
    allocated = torch_npu.npu.memory_allocated() / 1024**2  # MB
    reserved = torch_npu.npu.memory_reserved() / 1024**2  # MB
    print(f"[{tag}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

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


def vis_projected_color_on_depth(depth, uv_tensor, features_1024, savepath='projected_feature_on_depth.png'):
    """
    可视化高维特征（1024D），投影到深度图上。特征用 PCA 降维为 RGB。
    Args:
        depth: [H, W] tensor or numpy array
        uv_tensor: [N, 2] in normalized [-1,1]
        features_1024: [N, 1024] tensor
    """
    if isinstance(depth, torch.Tensor):
        depth_np = depth.cpu().numpy()
    else:
        depth_np = depth

    H, W = depth_np.shape
    N = features_1024.shape[0]

    # 1. 降维 [N,1024] -> [N,3]
    pca = PCA(n_components=3)
    features_np = features_1024.detach().cpu().numpy()
    colors_3d = pca.fit_transform(features_np)  # [N,3]
    colors_3d -= colors_3d.min()
    colors_3d /= (colors_3d.max() + 1e-6)  # normalize to [0,1]

    # 2. 像素坐标还原
    u_norm = uv_tensor[:, 0].cpu()
    v_norm = uv_tensor[:, 1].cpu()
    u = ((u_norm + 1) / 2) * (W - 1)
    v = ((v_norm + 1) / 2) * (H - 1)

    # 3. 可视化
    plt.figure(figsize=(8,6))
    plt.imshow(depth_np, cmap='gray')
    plt.scatter(u, v, c=colors_3d, s=10)
    plt.title('Projected High-Dim Feature (PCA RGB) on Depth')
    plt.axis('off')
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

def vis_visibility_mask(points, mask, save_path):
    """
    保存可见点到 ply
    """
    pts = points[mask].detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(save_path, pcd)

def feature_to_color(feat):
    """将高维特征 [N, C] 映射为 RGB [N, 3]"""
    feat = feat.detach().cpu()
    feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-6)
    feat = feat[:, :3] if feat.shape[1] >= 3 else torch.nn.functional.pad(feat, (0, 3 - feat.shape[1]))
    feat = feat - feat.min()
    feat = feat / (feat.max() + 1e-6)
    return feat.numpy()

def to_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def get_model_summary(model):
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += '\n'
    model_summary += f'Number of parameters: {num_params}, Model size: {num_params/1e6}M\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params}, Model size: {num_params/1e6}M\n'
    return model_summary

# norm之前的filter
def _filter_dataset(metadata, num_voxels_threshold=50_000):
    # metadata = metadata[metadata[f'feature_{opt.feat_model}'] == True]
    selected_scene_types = ['apartment', 'office', 'bedroom / hotel', 'bathroom', 
                        'conference room', 'kitchen', 'classroom', 'living room / lounge', 
                        'copy / mail room', 'bedroom', 'laundry room', 'bedroom / hotel']
    metadata = metadata[metadata[f'scene_types'].isin(selected_scene_types)]
    metadata = metadata[metadata['num_voxels'] < num_voxels_threshold]
    return metadata

# norm之后的filter
def _filter_dataset_new(metadata):
    # metadata = metadata[metadata[f'feature_{opt.feat_model}'] == True]
    selected_scene_types = ['apartment', 'office', 'bedroom / hotel', 'bathroom', 
                        'conference room', 'kitchen', 'classroom', 'living room / lounge', 
                        'copy / mail room', 'bedroom', 'laundry room', 'bedroom / hotel']
    metadata = metadata[metadata[f'scene_types'].isin(selected_scene_types)]
    return metadata

import re
def get_latest_ckpt(weights_dir):
    if os.path.exists(os.path.join(weights_dir, 'last.ckpt')):
        print(f"[INFO] Latest weight file: {os.path.join(weights_dir, 'last.ckpt')}")
        return 'last.ckpt'
    
    weights_files = [f for f in os.listdir(weights_dir) if f.endswith('.ckpt')]
    # print("weight_files: ", weights_files)

    epoch_pattern = re.compile(r'epochepoch=(\d+)\.ckpt')

    max_epoch = -1
    latest_weight_file = None

    for weight_file in weights_files:
        match = epoch_pattern.search(weight_file)
        if match:
            epoch = int(match.group(1))  # 提取 epoch 数字并转换为整数
            if epoch > max_epoch:
                max_epoch = epoch
                latest_weight_file = weight_file

    if latest_weight_file:
        print(f"[INFO] Latest weight file: {latest_weight_file} with epoch {max_epoch}")
    else:
        print("[INFO] No valid weight files found.")
    
    return latest_weight_file

def deepspeed_load(model, load_dir=''):
    load_path, client_state = model.load_checkpoint(load_dir=load_dir)
    if load_path:
        epoch, step, idx = client_state["epoch"], client_state["step"], client_state["start_idx"]
    else:
        print('[WARNING] not find checkpoint, start training from beginning')
        epoch = step = idx = 0
    return epoch, step, idx

def batch_project_and_aggregate_chunkwise(points, features, depths, intrinsics, w2cs, 
                                          T_norm=None, depth_tol=0.05,
                                          fill_empty=True, knn_k=8, knn_eps=1e-6,
                                          chunk_size_V=4, chunk_size_N=50000,
                                          opstype='fp16'):
    """
    分块计算版本, 避免显存OOM
    
    Args:
        points: [N, 3] 3D点世界坐标
        features: [V, C, H, W] 多视角特征图
        depths: [V, 1, H, W] 多视角深度图（单位与相机坐标系一致）
        intrinsics: [V, 3, 3] 相机内参
        w2cs: [V, 4, 4] 世界->相机 外参
        T_norm: 归一化矩阵
        depth_tol: 深度threshold
        fill_empty: 是否对无特征点做KNN插值
        knn_k: KNN邻居数
        knn_eps: 防除零小数
        chunk_size_V: 每次处理的视角块大小
        chunk_size_N: 每次处理的点数块大小
    
    Returns:
        aggregated_feats: [N, C] 聚合后的点特征
        visibility_mask: [N] 是否至少在一个视角可见
    """
    device = points.device
    V, C, H, W = features.shape
    N = points.shape[0]
    
    if opstype=='fp16':
        dtype=torch.float16
    elif opstype=='bf16':
        dtype = torch.bfloat16 
    else:
        dtype = torch.float
    with torch.autocast('npu', dtype):

        aggregated_feats = torch.zeros((N, C), device=device)
        weight_sum = torch.zeros((N, 1), device=device)

        # 齐次坐标提前计算，避免重复
        homog_points = torch.cat([points, torch.ones((N, 1), device=device)], dim=-1)  # [N,4]

        visibility_mask_all = torch.zeros((N,), dtype=torch.bool, device=device)
        
        # 使用autocast来启用低精度计算
        # with torch_npu.npu.amp.autocast(dtype=dtype):  # 或者 dtype=torch.float16，选择你需要的精度
            # 分块循环处理视角
        for v_start in range(0, V, chunk_size_V):
            v_end = min(v_start + chunk_size_V, V)

            features_chunk = features[v_start:v_end]    # [V_c,C,H,W]
            depths_chunk = depths[v_start:v_end]        # [V_c,H,W]
            intr_chunk = intrinsics[v_start:v_end]      # [V_c,3,3]
            w2c_chunk = w2cs[v_start:v_end]             # [V_c,4,4]

            V_c = v_end - v_start

            # 分块循环处理点
            for n_start in range(0, N, chunk_size_N):
                n_end = min(n_start + chunk_size_N, N)

                points_chunk = points[n_start:n_end]          # [N_c,3]
                homog_points_chunk = homog_points[n_start:n_end]  # [N_c,4]
                N_c = n_end - n_start

                # 世界坐标 -> 相机坐标 [V_c, N_c, 4]
                cam_points = torch.einsum('vij,nj->vni', w2c_chunk, homog_points_chunk)
                cam_points_xyz = cam_points[..., :3]  # [V_c,N_c,3]

                # 判断是否在相机前方
                z_cam = cam_points_xyz[..., 2]  # [V_c,N_c]
                if T_norm is not None:
                    z_cam = z_cam * T_norm[2, 2]
                in_front = z_cam > 1e-6

                # 归一化投影坐标
                z_safe = z_cam.clamp(min=1e-6)
                x_norm = cam_points_xyz[..., 0] / z_safe
                y_norm = cam_points_xyz[..., 1] / z_safe

                # 相机内参
                fx = intr_chunk[:, 0, 0].view(V_c, 1)  # [V_c,1]
                fy = intr_chunk[:, 1, 1].view(V_c, 1)
                cx = intr_chunk[:, 0, 2].view(V_c, 1)
                cy = intr_chunk[:, 1, 2].view(V_c, 1)

                # 投影像素坐标
                u = fx * x_norm + cx  # [V_c, N_c]
                v = fy * y_norm + cy  # [V_c, N_c]

                # 图像内范围判断
                valid_u = (u >= 0) & (u <= (W - 1))
                valid_v = (v >= 0) & (v <= (H - 1))
                in_img = valid_u & valid_v  # [V_c, N_c]

                # 归一化到[-1,1]，grid_sample要求
                u_norm = 2.0 * (u / (W - 1)) - 1.0  # [V_c, N_c]
                v_norm = 2.0 * (v / (H - 1)) - 1.0  # [V_c, N_c]
                grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(2).to(depths.device)  # [V_c, N_c, 1, 2]
                grid = torch.clamp(grid, -1 + 1e-3, 1 - 1e-3)
                
                # 采样深度 [V_c, 1, H, W] 和grid采样 [V_c, N_c, 1, 2]
                sampled_depths = F.grid_sample(
                    depths_chunk.float(),  # [V_c,1,H,W]
                    grid.float(),                      # [V_c,N_c,1,2]
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1).squeeze(-1).bfloat16()  # [V_c, N_c]
                
                # 深度有效性判定
                depth_positive = sampled_depths > 0
                depth_close = torch.abs(sampled_depths - z_cam) < depth_tol
                depth_valid = depth_positive & depth_close  # [V_c, N_c]

                # 综合有效性mask
                valid_mask = in_front & in_img & depth_valid  # [V_c, N_c]
                
                any_visible = valid_mask.any(dim=0)  # [N_c]
                visibility_mask_all[n_start:n_end] |= any_visible

                # 从feature map采样特征
                sampled_feats = F.grid_sample(
                    features_chunk.float(),          # [V_c, C, H, W]
                    grid.float(),                   # [V_c, N_c, 1, 2]
                    mode='bilinear',
                    align_corners=True
                ).squeeze(-1).bfloat16()              # [V_c, C, N_c]
                sampled_feats = sampled_feats.permute(2, 0, 1)  # [N_c, V_c, C]
                
                if DEBUG:
                    for vi in range(V_c):
                        n_start_str = str(n_start).zfill(4)
                        v_start_str = str(v_start).zfill(4)
                        vis_uv_in_depth(depths_chunk[vi].squeeze(), grid[vi].squeeze(), 
                                        os.path.join(out_dir, f'debug_{n_start_str}_{v_start_str}_view{vi}_uv_on_depth.png'))
                        vis_visibility_mask(points_chunk, valid_mask[vi], 
                                            os.path.join(out_dir, f'debug_{n_start_str}_{v_start_str}_view{vi}_visible_points.ply'))
                        vis_projected_color_on_depth(depths_chunk[vi].squeeze(), grid[vi].squeeze(), 
                                                        sampled_feats[:, vi, :], 
                                                        os.path.join(out_dir, f'debug_{n_start_str}_{v_start_str}_view{vi}_colored_proj.png'))
            
                
                # 加权平均（根据valid_mask）
                weights = valid_mask.float().permute(1, 0)  # [N_c, V_c]
                total_weights = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [N_c,1]
                weighted_feats = sampled_feats * weights.unsqueeze(-1)  # [N_c, V_c, C]
                feat_chunk = weighted_feats.sum(dim=1)  # [N_c, C]

                # 累加结果
                aggregated_feats[n_start:n_end] = aggregated_feats[n_start:n_end] + feat_chunk
                weight_sum[n_start:n_end] = weight_sum[n_start:n_end] + total_weights

        # 归一化
        aggregated_feats = aggregated_feats / weight_sum.clamp(min=1e-6)
        # visibility_mask = (weight_sum.squeeze(-1) > 0)
        visibility_mask = visibility_mask_all

        if DEBUG:
            color_before = feature_to_color(aggregated_feats)
            pcd_before = to_pcd(points, color_before)
            o3d.io.write_point_cloud(os.path.join(out_dir, f'debug_features_before_fill.ply'), pcd_before)

        # 对无特征点做KNN插值补全
        if fill_empty:
            missing_mask = ~visibility_mask  # 无特征点mask
            if missing_mask.any():
                src_points = points[visibility_mask]  # [M,3]
                src_feats = aggregated_feats[visibility_mask]  # [M,C]
                tgt_points = points[missing_mask]  # [K,3]

                k = min(knn_k, src_points.shape[0])
                dists = torch.cdist(tgt_points, src_points, p=2)  # [K, M]
                knn_dists, knn_idx = torch.topk(dists, k, largest=False, dim=1)

                knn_weights = 1.0 / (knn_dists + knn_eps)
                knn_weights = knn_weights / knn_weights.sum(dim=1, keepdim=True)  # 归一化

                interpolated_feats = torch.sum(src_feats[knn_idx] * knn_weights.unsqueeze(-1), dim=1)  # [K,C]

                aggregated_feats[missing_mask] = interpolated_feats

        torch_npu.npu.empty_cache()

    return aggregated_feats, visibility_mask


# def batch_project_and_aggregate_chunkwise(points, features, depths, intrinsics, w2cs, 
#                                           T_norm=None, depth_tol=0.05,
#                                           fill_empty=True, knn_k=8, knn_eps=1e-6,
#                                           chunk_size_V=4, chunk_size_N=50000):
#     """
#     分块计算版本, 避免显存OOM
    
#     Args:
#         points: [N, 3] 3D点世界坐标
#         features: [V, C, H, W] 多视角特征图
#         depths: [V, 1, H, W] 多视角深度图（单位与相机坐标系一致）
#         intrinsics: [V, 3, 3] 相机内参
#         w2cs: [V, 4, 4] 世界->相机 外参
#         T_norm: 归一化矩阵
#         depth_tol: 深度threshold
#         fill_empty: 是否对无特征点做KNN插值
#         knn_k: KNN邻居数
#         knn_eps: 防除零小数
#         chunk_size_V: 每次处理的视角块大小
#         chunk_size_N: 每次处理的点数块大小
    
#     Returns:
#         aggregated_feats: [N, C] 聚合后的点特征
#         visibility_mask: [N] 是否至少在一个视角可见
#     """
#     device = points.device
#     V, C, H, W = features.shape
#     N = points.shape[0]

#     aggregated_feats = torch.zeros((N, C), device=device)
#     weight_sum = torch.zeros((N, 1), device=device)

#     # 齐次坐标提前计算，避免重复
#     homog_points = torch.cat([points, torch.ones((N, 1), device=device)], dim=-1)  # [N,4]

#     visibility_mask_all = torch.zeros((N,), dtype=torch.bool, device=device)
    
#     # 分块循环处理视角
#     for v_start in range(0, V, chunk_size_V):
#         v_end = min(v_start + chunk_size_V, V)

#         features_chunk = features[v_start:v_end]    # [V_c,C,H,W]
#         depths_chunk = depths[v_start:v_end]        # [V_c,H,W]
#         intr_chunk = intrinsics[v_start:v_end]      # [V_c,3,3]
#         w2c_chunk = w2cs[v_start:v_end]             # [V_c,4,4]

#         V_c = v_end - v_start

#         # 分块循环处理点
#         for n_start in range(0, N, chunk_size_N):
#             n_end = min(n_start + chunk_size_N, N)

#             points_chunk = points[n_start:n_end]          # [N_c,3]
#             homog_points_chunk = homog_points[n_start:n_end]  # [N_c,4]
#             N_c = n_end - n_start

#             # 世界坐标 -> 相机坐标 [V_c, N_c, 4]
#             cam_points = torch.einsum('vij,nj->vni', w2c_chunk, homog_points_chunk)
#             cam_points_xyz = cam_points[..., :3]  # [V_c,N_c,3]

#             # 判断是否在相机前方
#             z_cam = cam_points_xyz[..., 2]  # [V_c,N_c]
#             if T_norm is not None:
#                 z_cam = z_cam * T_norm[2, 2]
#             in_front = z_cam > 1e-6

#             # 归一化投影坐标
#             z_safe = z_cam.clamp(min=1e-6)
#             x_norm = cam_points_xyz[..., 0] / z_safe
#             y_norm = cam_points_xyz[..., 1] / z_safe

#             # 相机内参
#             fx = intr_chunk[:, 0, 0].view(V_c, 1)  # [V_c,1]
#             fy = intr_chunk[:, 1, 1].view(V_c, 1)
#             cx = intr_chunk[:, 0, 2].view(V_c, 1)
#             cy = intr_chunk[:, 1, 2].view(V_c, 1)

#             # 投影像素坐标
#             u = fx * x_norm + cx  # [V_c, N_c]
#             v = fy * y_norm + cy  # [V_c, N_c]

#             # 图像内范围判断
#             valid_u = (u >= 0) & (u <= (W - 1))
#             valid_v = (v >= 0) & (v <= (H - 1))
#             in_img = valid_u & valid_v  # [V_c, N_c]

#             # 归一化到[-1,1]，grid_sample要求
#             u_norm = 2.0 * (u / (W - 1)) - 1.0  # [V_c, N_c]
#             v_norm = 2.0 * (v / (H - 1)) - 1.0  # [V_c, N_c]
#             grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(2).to(depths.device)  # [V_c, N_c, 1, 2]
#             grid = torch.clamp(grid, -1 + 1e-3, 1 - 1e-3)
            
#             # 采样深度 [V_c, 1, H, W] 和grid采样 [V_c, N_c, 1, 2]
#             sampled_depths = F.grid_sample(
#                 depths_chunk,  # [V_c,1,H,W]
#                 grid,                      # [V_c,N_c,1,2]
#                 mode='bilinear',
#                 align_corners=True
#             ).squeeze(1).squeeze(-1)  # [V_c, N_c]
            
#             # 深度有效性判定
#             depth_positive = sampled_depths > 0
#             depth_close = torch.abs(sampled_depths - z_cam) < depth_tol
#             depth_valid = depth_positive & depth_close  # [V_c, N_c]

#             # 综合有效性mask
#             valid_mask = in_front & in_img & depth_valid  # [V_c, N_c]
            
#             any_visible = valid_mask.any(dim=0)  # [N_c]
#             visibility_mask_all[n_start:n_end] |= any_visible

#             # 从feature map采样特征
#             sampled_feats = F.grid_sample(
#                 features_chunk,          # [V_c, C, H, W]
#                 grid,                   # [V_c, N_c, 1, 2]
#                 mode='bilinear',
#                 align_corners=True
#             ).squeeze(-1)              # [V_c, C, N_c]
#             sampled_feats = sampled_feats.permute(2, 0, 1)  # [N_c, V_c, C]
            
#             if DEBUG:
#                 for vi in range(V_c):
#                     n_start_str = str(n_start).zfill(4)
#                     v_start_str = str(v_start).zfill(4)
#                     vis_uv_in_depth(depths_chunk[vi].squeeze(), grid[vi].squeeze(), 
#                                     os.path.join(out_dir, f'debug_{n_start_str}_{v_start_str}_view{vi}_uv_on_depth.png'))
#                     vis_visibility_mask(points_chunk, valid_mask[vi], 
#                                         os.path.join(out_dir, f'debug_{n_start_str}_{v_start_str}_view{vi}_visible_points.ply'))
#                     vis_projected_color_on_depth(depths_chunk[vi].squeeze(), grid[vi].squeeze(), 
#                                                     sampled_feats[:, vi, :], 
#                                                     os.path.join(out_dir, f'debug_{n_start_str}_{v_start_str}_view{vi}_colored_proj.png'))
            
#             # 加权平均（根据valid_mask）
#             weights = valid_mask.float().permute(1, 0)  # [N_c, V_c]
#             total_weights = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [N_c,1]
#             weighted_feats = sampled_feats * weights.unsqueeze(-1)  # [N_c, V_c, C]
#             feat_chunk = weighted_feats.sum(dim=1)  # [N_c, C]

#             # 累加结果
#             aggregated_feats[n_start:n_end] = aggregated_feats[n_start:n_end] + feat_chunk
#             weight_sum[n_start:n_end] = weight_sum[n_start:n_end] + total_weights

#     # 归一化
#     aggregated_feats = aggregated_feats / weight_sum.clamp(min=1e-6)
#     # visibility_mask = (weight_sum.squeeze(-1) > 0)
#     visibility_mask = visibility_mask_all
    
#     if DEBUG:
#         color_before = feature_to_color(aggregated_feats)
#         pcd_before = to_pcd(points, color_before)
#         o3d.io.write_point_cloud(os.path.join(out_dir, f'debug_features_before_fill.ply'), pcd_before)

#     # 对无特征点做KNN插值补全
#     if fill_empty:
#         missing_mask = ~visibility_mask  # 无特征点mask
#         if missing_mask.any():
#             src_points = points[visibility_mask]  # [M,3]
#             src_feats = aggregated_feats[visibility_mask]  # [M,C]
#             tgt_points = points[missing_mask]  # [K,3]

#             k = min(knn_k, src_points.shape[0])
#             dists = torch.cdist(tgt_points, src_points, p=2)  # [K, M]
#             knn_dists, knn_idx = torch.topk(dists, k, largest=False, dim=1)

#             knn_weights = 1.0 / (knn_dists + knn_eps)
#             knn_weights = knn_weights / knn_weights.sum(dim=1, keepdim=True)  # 归一化

#             interpolated_feats = torch.sum(src_feats[knn_idx] * knn_weights.unsqueeze(-1), dim=1)  # [K,C]

#             aggregated_feats[missing_mask] = interpolated_feats
                    
#     if DEBUG:
#         color_after = feature_to_color(aggregated_feats)
#         pcd_after = to_pcd(points, color_after)
#         o3d.io.write_point_cloud( os.path.join(out_dir, f'debug_features_after_fill.ply'), pcd_after)
    
#     torch_npu.npu.empty_cache()

#     return aggregated_feats, visibility_mask
