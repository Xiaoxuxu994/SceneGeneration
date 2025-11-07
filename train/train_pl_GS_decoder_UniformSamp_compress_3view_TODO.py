"""
20250919 把之前的 perview train 修改为 adjacent 3 view
"""
# %%
import os
os.environ['ATTN_BACKEND'] = 'flash-attn' # flash-attn, xformers
# os.environ['NCCL_BLOCKING_WAIT'] = '1'
# os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
# os.environ['NCCL_DEBUG'] = 'INFO'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import copy
import json
import math
import time
import argparse
import torch
# torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.preferred_linalg_library('magma') # magma
from torchvision.utils import save_image
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from loguru import logger
import utils3d
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import lpips
import gc
import json
from easydict import EasyDict as edict
import shutil, pathlib
from pathlib import Path
import random
import glob
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from dataset_toolkits_scannet.extract_feature_scannet import get_data
# from dataset_toolkits.extract_feature import get_data
# from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import trellis.models as models
import trellis.modules.sparse as sp
# from trellis.models.sparse_structure_vae import SparseStructureVAE
# from trellis.models.sparse_latent_vae import SparseLatentVAEscannet, SparseLatentVAESepMLPscannet
# from trellis.utils import render_utils, postprocessing_utils
from trellis.utils import render_utils_scannet as render_utils

import torch.multiprocessing as mp
# mp.set_start_method('spawn')

DEBUG = False
out_dir = './'

def seed_everything(seed=42):
    random.seed(seed)                    
    np.random.seed(seed)            
    torch.manual_seed(seed)         
    torch.cuda.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed)      
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     

seed_everything(520)

to_pil = transforms.ToPILImage()

torch.set_grad_enabled(True)

import torch.distributed as dist

def rank_barrier(msg="sync"):
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    print(f"[Rank {dist.get_rank()}] {msg} @ {time.time()}")

# transform = transforms.Compose([
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
from torch.utils.data._utils.collate import default_collate

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # 可以在 training_step 做判断
    return default_collate(batch)


from torch.optim.lr_scheduler import LambdaLR
def cosine_scheduler_with_warmup(optimizer, base_lr, warmup_steps, total_steps, min_lr=0.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return (min_lr + (base_lr - min_lr) * cosine_decay) / base_lr
    return LambdaLR(optimizer, lr_lambda)


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

def l1_loss(pred, gt, mask=None):
    """
    pred, gt: torch.Tensor of any shape
    mask: torch.Tensor (same shape as pred or broadcastable), or None
    """
    diff = torch.abs(pred - gt)

    if mask is not None:
        mask = mask.to(diff.dtype)  # 支持 bool, int, float
        masked_diff = diff * mask
        denom = mask.sum()
        if denom == 0:
            return torch.tensor(0.0, device=pred.device)
        return masked_diff.sum() / denom
    else:
        return diff.mean()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # 归一化到 [-1, 1]
])

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
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2  # MB
    print(f"[{tag}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

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

# ========================================================================================

def gaussian_kernel(window_size: int, sigma: float, device=None):
    """Create a 2D Gaussian kernel normalized to sum=1."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel_2d = g[:, None] * g[None, :]
    return kernel_2d / kernel_2d.sum()

def masked_ssim_batch(img1, img2, mask, window_size=11, sigma=1.5):
    """
    Vectorized masked SSIM.
    img1, img2: [B, C, H, W], float in [0,1]
    mask:       [B, H, W] or [B, 1, H, W], bool or float
    return:     [B], masked SSIM score per image
    """
    assert img1.shape == img2.shape, "Images must have same shape"
    B, C, H, W = img1.shape
    device = img1.device

    # Ensure mask shape
    mask = mask.float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)  # [B, 1, H, W]

    # Gaussian kernel (expanded per channel)
    window = gaussian_kernel(window_size, sigma, device)
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, W, W]
    window = window.expand(C, 1, window_size, window_size)  # [C, 1, W, W]

    # Local mask sum for normalization
    mask_sum = F.conv2d(mask, window[:1], padding=window_size // 2, groups=1)

    # === Local means ===
    mu1 = F.conv2d(img1 * mask, window, padding=window_size // 2, groups=C) / (mask_sum + 1e-6)
    mu2 = F.conv2d(img2 * mask, window, padding=window_size // 2, groups=C) / (mask_sum + 1e-6)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # === Local variances & covariance ===
    sigma1_sq = F.conv2d(img1**2 * mask, window, padding=window_size // 2, groups=C) / (mask_sum + 1e-6) - mu1_sq
    sigma2_sq = F.conv2d(img2**2 * mask, window, padding=window_size // 2, groups=C) / (mask_sum + 1e-6) - mu2_sq
    sigma12   = F.conv2d(img1 * img2 * mask, window, padding=window_size // 2, groups=C) / (mask_sum + 1e-6) - mu1_mu2

    # === SSIM constants ===
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # === SSIM map ===
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # === Mean over channels ===
    ssim_map = ssim_map.mean(dim=1, keepdim=True)  # [B, 1, H, W]

    # === Masked spatial average ===
    masked_ssim = (ssim_map * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-6)
    return masked_ssim.squeeze(1)  # [B]
# ========================================================================================


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

def dispersion_loss_voxel_scaled(xyz, voxel_size=0.01, margin_rel=0.5, beta=10, eps=1e-5, activation="softplus"):
    # xyz: [N, M, 3]
    N, M, _ = xyz.shape
    pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
    if not pairs:
        return torch.tensor(0.0, device=xyz.device)

    # Use clamp before sqrt to avoid NaNs
    dists = [((xyz[:, i] - xyz[:, j])**2).sum(-1).clamp(min=eps).sqrt() for i, j in pairs]  # [N, num_pairs]
    pairwise = torch.stack(dists, dim=1)
    d_min = pairwise.min(dim=1)[0]  # [N]

    d_norm = d_min / voxel_size  # relative to voxel size

    if margin_rel is not None:
        if activation == 'softplus':
            loss = F.softplus(margin_rel - d_norm, beta=beta)
        else:
            loss = F.relu(margin_rel - d_norm)
    else:
        safe = torch.clamp(d_norm, min=eps)
        loss = 1.0 / safe

    return loss.mean()

def feature_diversity_loss(
    feat_div: torch.Tensor,  # [N_voxel, G, D]
    mode: str = "pairwise_l2", 
    eps: float = 1e-6,
    margin: float | None = None
) -> torch.Tensor:
    """
    Compute feature diversity loss for voxel internal Gaussians.

    Args:
        feat_div: [N_voxel, G, D] (color+opacity)
        mode: "neg_var" | "inv_var" | "pairwise_l2"
        eps: small constant
        margin: only for pairwise_l2

    Returns:
        loss: scalar tensor
    """
    N, G, D = feat_div.shape

    if mode == "neg_var":
        var_per_dim = feat_div.var(dim=1, unbiased=False)  # [N, D]
        v = var_per_dim.mean(dim=-1)                       # [N]
        return -v.mean()

    elif mode == "inv_var":
        var_per_dim = feat_div.var(dim=1, unbiased=False)  # [N, D]
        v = var_per_dim.mean(dim=-1)
        return (1.0 / (v + eps)).mean()

    elif mode == "pairwise_l2":
        # [N, G, 1, D] - [N, 1, G, D] -> [N, G, G, D]
        diff = feat_div[:, :, None, :] - feat_div[:, None, :, :]
        dists = diff.norm(dim=-1)  # [N, G, G]

        # 取上三角，不包含对角线
        mask = torch.triu(torch.ones(G, G, device=feat_div.device), diagonal=1).bool()
        pairwise_dists = dists[:, mask]  # [N, G*(G-1)/2]

        if margin is not None:
            # 只有距离小于 margin 才惩罚
            loss = F.relu(margin - pairwise_dists)
            return loss.mean()
        else:
            return -pairwise_dists.mean()  # 最大化两两距离

    else:
        raise ValueError(f"Unknown mode: {mode}")


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
        results = render_gsplat(GS,
                                extrinsics=interp_exts_lst[iii,:,:].float().to(device),
                                intrinsics=interp_ints_lst[iii,:,:].float().to(device),
                                render_mode="RGB+D",
                                resolution=(new_h, new_w )) #  416, 700 
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
from gsplat.rendering import rasterization
def render_gsplat(gausssian, 
                  extrinsics: torch.Tensor,
                  intrinsics: torch.Tensor,
                  resolution: tuple,
                  render_mode: str = "RGB+D",
                  sh_degree: int = 0):

    means3D = gausssian.get_xyz
    opacity = gausssian.get_opacity
    scales = gausssian.get_scaling
    rotations = gausssian.get_rotation
    shs = gausssian.get_features
    shs = shs.view(means3D.shape[0], -1, 3)
    
    if len(extrinsics.shape) == 4 and len(intrinsics.shape) == 4:
        extrinsics = extrinsics.squeeze(0)
        intrinsics = intrinsics.squeeze(0)
    
    renders_images, _, _ = rasterization(
            means = means3D,  # [N, 3]
            quats = rotations,  # [N, 4]
            scales = scales,  # [N, 3]
            opacities = opacity.squeeze(),  # [N]
            sh_degree=sh_degree,
            colors = shs,  # [N, S, 3]
            viewmats = extrinsics.unsqueeze(0),  # [C, 4, 4] # 。squeeze(0)
            Ks = intrinsics.unsqueeze(0),  # [C, 3, 3] # .squeeze(0)
            width = resolution[1],
            height = resolution[0],
            render_mode=render_mode,
            packed=False,
            # distributed=world_size > 1,
        )
    if renders_images.shape[-1] == 4:
        colors, depths = renders_images[..., 0:3], renders_images[..., 3:4]
    else:
        colors, depths = renders_images, None

    if depths is None:
        ret = edict({
            'color': colors
            })
    else:
        ret = edict({
                'color': colors,
                'depth': depths,
            })
    
    if 'color' not in ret: ret['color'] = []
    if 'depth' not in ret: ret['depth'] = []
    ret['color'] = np.clip(ret['color'].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
    if 'percent_depth' in ret:
        ret['depth'] = ret['percent_depth'].detach().cpu().numpy()
    elif 'depth' in ret:
        ret['depth'] = ret['depth'].detach().cpu().numpy()
    else:
        ret['depth'] = None
    return ret

# 计算位置的欧式距离
def compute_position_distance(t1, t2):
    return torch.norm(t1 - t2)

# 计算旋转矩阵之间的 Frobenius 范数
def compute_orientation_distance(R1, R2):
    return torch.norm(R1 - R2)

# 计算视锥重叠度，内参（焦距）对视锥的影响
def compute_fov_overlap(R1, t1, R2, t2, points_3d, intrinsic1, intrinsic2):
    """
    计算两个相机的视锥重叠度，考虑内参的影响（假设内参已归一化）。
    """
    # 归一化后的内参通常表示相机的视场角
    f_x1, f_y1 = intrinsic1[0, 0], intrinsic1[1, 1]
    f_x2, f_y2 = intrinsic2[0, 0], intrinsic2[1, 1]
    
    # 将3D点投影到相机的视平面
    projected_points_1 = torch.matmul(R1, points_3d.T) + t1.unsqueeze(1)  # 投影到相机1
    projected_points_2 = torch.matmul(R2, points_3d.T) + t2.unsqueeze(1)  # 投影到相机2

    # 判断点是否在视场内（根据内参决定视场大小）
    visibility_1 = torch.abs(projected_points_1[0, :]) < f_x1  # 假设在视场内
    visibility_2 = torch.abs(projected_points_2[0, :]) < f_x2

    # 计算视场重叠度
    overlap = torch.sum(visibility_1 & visibility_2).float() / torch.sum(visibility_1 | visibility_2).float()  # Jaccard overlap
    return overlap

# 综合计算相机接近度，考虑内参影响
def compute_total_distance(extrinsics1, extrinsics2, points_3d, intrinsic1, intrinsic2, alpha=1.0, beta=1.0):
    R1, t1 = extrinsics1[:3, :3], extrinsics1[:3, 3]
    R2, t2 = extrinsics2[:3, :3], extrinsics2[:3, 3]
    
    position_dist = compute_position_distance(t1, t2)
    orientation_dist = compute_orientation_distance(R1, R2)
    fov_overlap = compute_fov_overlap(R1, t1, R2, t2, points_3d, intrinsic1, intrinsic2)
    
    # 综合加权计算
    return alpha * position_dist + beta * orientation_dist - (1.0 - beta) * fov_overlap

# 找到与目标视角最接近的两个视角
def find_closest_views(extrinsics, intrinsics, target_idx, points_3d, alpha=1.0, beta=1.0):
    """
    给定一个目标视角，找到与其最接近的两个视角（排除目标视角本身）。
    """
    distances = []
    
    # 获取目标视角的外参和内参
    extrinsics_target = extrinsics[target_idx]
    intrinsic_target = intrinsics[target_idx]
    
    found_indices = [target_idx]
    for i in range(extrinsics.shape[0]):
        if i != target_idx:  # 忽略目标视角本身
            dist = compute_total_distance(extrinsics[i], extrinsics_target, points_3d, intrinsics[i], intrinsic_target, alpha, beta)
            distances.append(dist)
        else:
            distances.append(0)
    # print("distances: ", distances)
    
    # 按照接近度排序（从小到大）
    distances_tensor = torch.tensor(distances)
    sorted_indices = torch.argsort(distances_tensor)
    # print("sorted_indices: ", sorted_indices)
    
    closest_indices = sorted_indices[:3]  # 选取距离最小的两个
 
    # 返回最接近的两个视角的索引
    return closest_indices

# 定义 transform，将图片转换为 tensor，并归一化到 [0,1]
transform_novel = transforms.Compose([
    transforms.ToTensor(),  # 会把 [0,255] 转为 [0,1] 的 tensor
])

# %%
# ---------------------------
# 1. 定义 Dataset
# ---------------------------
class SparseLatentVaeDataset(Dataset):
    def __init__(self, metadata, opt, stage='train'):
        """
        metadata: 包含所有样本信息的 DataFrame，每个样本至少包含 'sha256'
        opt: 参数选项，包含 voxel_size 等参数
        """
        self.metadata = metadata.reset_index(drop=True)
        self.opt = opt
        self.vs_size = 2/128
        self.feats_folder_name = self.opt.feats_folder_name if hasattr(self.opt, 'feats_folder_name') else 'features_0p05_correct'
        self.feat_model_name =  self.opt.feat_model_name if hasattr(self.opt, 'feat_model_name') else 'dinov2_vitl14_reg'
        self.voxels_folder_name = self.opt.voxels_folder_name if hasattr(self.opt, 'voxels_folder_name') else 'voxels_0p05'
        self.depth_folder_name = self.opt.depth_folder_name if hasattr(self.opt, 'depth_folder_name') else 'depths_pytorch3d'
        assert self.opt.feat_type in ['nonzero_mean_agg_feats', 'max_agg_feats', 'fifo_agg_feats'], "opt.feat_type should be in nonzero_mean_agg_feats, max_agg_feats, fifo_agg_feats"
        self.feat_type = self.opt.feat_type if hasattr(self.opt, 'feat_type') else 'max_agg_feats'
        self.stage = stage
        if self.stage == 'train':
            self.num_rend_views = 3
        elif self.stage in ['val', 'test']:
            self.num_rend_views = 32 
        elif self.stage == 'test_trainset_novel':
            self.num_rend_views = 32 
            self.test_frames_folder_name = 'test_frames_rgb_depth_v14_fix32'
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

    def __len__(self):
        return len(self.metadata)

    def is_valid_depth(self, depth, valid_ratio_thresh=0.6):
        # depth: [H, W] tensor or numpy array
        if isinstance(depth, torch.Tensor):
            valid_mask = (depth > 0) & torch.isfinite(depth)
            return valid_mask.sum().item() / depth.numel() >= valid_ratio_thresh
        else:
            valid_mask = (depth > 0) & np.isfinite(depth)
            return valid_mask.sum() / depth.size >= valid_ratio_thresh
    
    @staticmethod
    def get_extrinsics_for_samples(view, T_norm=None):
        c2w = torch.tensor(view['transform_matrix'])
        c2w[2, :] *= -1
        c2w = c2w[np.array([1,0,2,3]), :]
        c2w[:3, 1:3] *= -1
        w2c = torch.inverse(c2w)
        w2c_Tnorm = w2c @ torch.inverse(T_norm) if T_norm is not None else w2c
        return w2c_Tnorm
    
    @staticmethod
    def select_visible_points(points, depths, intrinsics, w2c, T_norm=None, depth_tol=0.05):
        """
        筛选在单个视角下可见的3D点。

        Args:
            points: [N,3] world坐标
            depths: [H,W] 单视角深度图
            intrinsics: [3,3] 相机内参
            w2c: [4,4] 世界到相机外参
            T_norm: [4,4] 3D旋转缩放矩阵
            depth_tol: float, 可见性深度容差
        
        Returns:
            visible_points: [M,3] 可见的3D点
            visibility_mask: [N] bool mask，指示每个点是否可见
        """
        # print(f"Select visible points: points {points.shape}, depths {depths.shape}, intrinsics {intrinsics.shape}, w2c {w2c.shape}, T_norm {T_norm.shape if T_norm is not None else None}, depth_tol {depth_tol}")
        device = points.device
        N = points.shape[0]
        H, W = depths.shape

        # 1. 齐次坐标
        homog_points = torch.cat([points, torch.ones((N,1), device=device)], dim=-1)  # [N,4]

        # 2. 世界 -> 相机坐标
        cam_points = (w2c @ homog_points.T).T  # [N,4]
        cam_xyz = cam_points[:, :3]
        
        # 3. 前方点
        z_cam = cam_xyz[:, 2]
        if T_norm is not None:
            z_cam *= T_norm[2,2]
        in_front = z_cam > 1e-6  
        # print("depth projection 0: ", z_cam[in_front].min(), z_cam[in_front].max())       
        # print("depth depthmap: ", depths.min(), depths.max())
        # print(f"in_front: {in_front.sum().item()}/{N} ({in_front.sum().item()/N*100:.2f}%)")

        # 4. 投影到像素坐标
        fx, fy = intrinsics[0,0], intrinsics[1,1]
        cx, cy = intrinsics[0,2], intrinsics[1,2]
        x_norm = cam_xyz[:,0] / z_cam.clamp(min=1e-6)
        y_norm = cam_xyz[:,1] / z_cam.clamp(min=1e-6)
        u = fx * x_norm + cx
        v = fy * y_norm + cy

        # 5. 图像范围
        in_img = (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1)
        # print(f"in_img: {in_img.sum().item()}/{N} ({in_img.sum().item()/N*100:.2f}%)")

        # 6. 对应深度
        u_norm = 2.0 * (u / (W - 1)) - 1.0  # [V,N]
        v_norm = 2.0 * (v / (H - 1)) - 1.0  # [V,N]
        grid = torch.stack([u_norm, v_norm], dim=-1).view(1, -1, 1, 2).to(depths.device)  # [V,N,1,2]        
        sampled_depths = F.grid_sample(
            depths.unsqueeze(0).unsqueeze(0).contiguous(),  # [V,1,H,W]
            grid.contiguous(),  # [V,N,1,2]
            mode='bilinear',
            align_corners=True
        ).squeeze()  # [V,N]
        # sampled_depths = depths[v0, u0]
        # 7. 深度有效性
        depth_positive = sampled_depths > 0
        # print(f"depth_positive: {depth_positive.sum().item()}/{N} ({depth_positive.sum().item()/N*100:.2f}%)")
        # if T_norm is not None:
        #     z_cam *= T_norm[2,2]
        depth_close = torch.abs(sampled_depths - z_cam) < depth_tol
        # print(f"depth_close: {depth_close.sum().item()}/{N} ({depth_close.sum().item()/N*100:.2f}%)")
        depth_valid = depth_positive & depth_close  # [V,N]
        # print(f"depth_valid: {depth_valid.sum().item()}/{N} ({depth_valid.sum().item()/N*100:.2f}%)")

        # 8. 综合可见性
        valid_mask = in_front & in_img & depth_valid
        # print(f"Visible points: {valid_mask.sum().item()}/{N} ({valid_mask.sum().item()/N*100:.2f}%)")
        
        if DEBUG:
            vis_uv_in_depth(depths, u, v, valid_mask, os.path.join(out_dir, "debug_uv_on_depth.png"))
            vis_visibility_mask(points, valid_mask, os.path.join(out_dir, "debug_visible_points.ply"))

        return valid_mask

    @staticmethod
    def select_visible_points1(points, depths, intrinsics, w2c, T_norm=None, depth_tol=0.05):
        """
        筛选在单个视角下可见的3D点。

        Args:
            points: [N,3] world坐标
            depths: [H,W] 单视角深度图
            intrinsics: [3,3] 相机内参
            w2c: [4,4] 世界到相机外参
            T_norm: [3,3] 3D旋转缩放矩阵
            depth_tol: float, 可见性深度容差
        
        Returns:
            visible_points: [M,3] 可见的3D点
            visibility_mask: [N] bool mask，指示每个点是否可见
        """
        device = points.device
        N = points.shape[0]
        H, W = depths.shape

        # 1. 齐次坐标
        homog_points = torch.cat([points, torch.ones((N,1), device=device)], dim=-1)  # [N,4]

        # 2. 世界 -> 相机坐标
        cam_points = (w2c @ homog_points.T).T  # [N,4]
        cam_xyz = cam_points[:, :3]
        
        # 3. 前方点
        z_cam = cam_xyz[:, 2]
        # if T_norm is not None:
        #     z_cam *= T_norm[2,2]
        in_front = z_cam > 1e-6    
        # print("depth projection 0: ", z_cam[in_front].min(), z_cam[in_front].max())       
        # print("depth depthmap: ", depths.min(), depths.max())
        # print(f"in_front: {in_front.sum().item()}/{N} ({in_front.sum().item()/N*100:.2f}%)")
        
        # 4. 投影到像素坐标
        fx, fy = intrinsics[0,0], intrinsics[1,1]
        cx, cy = intrinsics[0,2], intrinsics[1,2]
        x_norm = cam_xyz[:,0] / z_cam.clamp(min=1e-6)
        y_norm = cam_xyz[:,1] / z_cam.clamp(min=1e-6)
        u = fx * x_norm + cx
        v = fy * y_norm + cy

        # 5. 图像范围
        in_img = (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1)
        # print(f"in_img: {in_img.sum().item()}/{N} ({in_img.sum().item()/N*100:.2f}%)")

        # 6. 对应深度
        u_norm = 2.0 * (u / (W - 1)) - 1.0  # [V,N]
        v_norm = 2.0 * (v / (H - 1)) - 1.0  # [V,N]
        grid = torch.stack([u_norm, v_norm], dim=-1).view(1, -1, 1, 2).to(depths.device)  # [V,N,1,2]        
        sampled_depths = F.grid_sample(
            depths.unsqueeze(0).unsqueeze(0).contiguous(),  # [V,1,H,W]
            grid.contiguous(),  # [V,N,1,2]
            mode='bilinear',
            align_corners=True
        ).squeeze()  # [V,N]
        # sampled_depths = depths[v0, u0]
        # 7. 深度有效性
        depth_positive = sampled_depths > 0
        # print(f"depth_positive: {depth_positive.sum().item()}/{N} ({depth_positive.sum().item()/N*100:.2f}%)")        
        if T_norm is not None:
            z_cam *= T_norm[2,2]
        depth_close = torch.abs(sampled_depths - z_cam) < depth_tol
        # print(f"depth_close: {depth_close.sum().item()}/{N} ({depth_close.sum().item()/N*100:.2f}%)")        
        depth_valid = depth_positive & depth_close  # [V,N]
        # print(f"depth_valid: {depth_valid.sum().item()}/{N} ({depth_valid.sum().item()/N*100:.2f}%)")                

        # 8. 综合可见性
        valid_mask = in_front & in_img & depth_valid
        # print(f"Visible points: {valid_mask.sum().item()}/{N} ({valid_mask.sum().item()/N*100:.2f}%)")
        
        if DEBUG:
            vis_uv_in_depth(depths, u, v, valid_mask, os.path.join(out_dir, "debug_uv_on_depth.png"))
            vis_visibility_mask(points, valid_mask, os.path.join(out_dir, "debug_visible_points.ply"))

        return valid_mask

    def __getitem__(self, idx):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        
        new_w, new_h = self.opt.img_resolution
        sha256 = self.metadata.loc[idx, 'sha256']
        # 加载dino特征
        try:
            all_feats = torch.load(os.path.join(self.opt.dataset_dir, self.feats_folder_name, self.feat_model_name, f'{sha256}.pt'), weights_only=False, map_location="cpu")
            SCALE = all_feats.get("scale", 1)
            feats = all_feats[self.feat_type].cpu()
            voxel_indices = all_feats['voxel_indices'].cpu()
            origin_raw = all_feats.get("voxle_grid_origin", None)
            if origin_raw is not None:
                grid_origin = origin_raw.float().cpu()
            else:
                grid_origin = None
            voxel_points = all_feats['voxel_points'].float().cpu()
            vs_size = all_feats.get("vs_size", 0.05)
            T_norm = all_feats.get("T_norm", None)
            if T_norm is not None:
                T_norm = T_norm.float().cpu()
                T_norm_inv = torch.linalg.inv(T_norm) # torch.inverse(T_norm)
            
            with open(os.path.join(self.opt.dataset_dir, 'renders', sha256, 'transforms.json'), 'r') as f:
                cam_infos = json.load(f)
            fx, fy, cx, cy, w, h = cam_infos['fl_x'], cam_infos['fl_y'], cam_infos['cx'], cam_infos['cy'], cam_infos['w'], cam_infos['h']
            intrinsics = torch.tensor([[fx/w, 0.0, cx/w],[0.0, fy/h, cy/h],[0.0, 0.0, 1.0]]) # TODO: check一下   
            
            batch_data = {"sha256": sha256, 
                          "scale": SCALE,
                          "vs_size": vs_size,
                          }
            if T_norm is not None:
                batch_data.update({
                    "T_norm": T_norm,
                    "T_norm_inv": T_norm_inv})

            # if self.stage in ['train', 'val', 'test']:
            # all_intrinsics_lst = all_feats['normed_intrinsics_norm128_v14_fix32'].cpu()
            all_extrinsics_lst = all_feats['extrinsics_Tnorm_norm128_v14_fix32'].cpu()
            all_ori_extrinsics_lst = all_feats['ori_extrinsics_norm128_v14_fix32'].cpu()
            all_depths_lst = all_feats['depthmap_1080x1620_norm128_v14_fix32'].cpu() # [N, H, W]
            # all_depths_lst = all_depths_lst.unsqueeze(1)
            # all_depths_lst = F.interpolate(all_depths_lst, size=(new_h, new_w), mode='bilinear', align_corners=False)
            # all_depths_lst = all_depths_lst.squeeze(1) # [N, H, W]
            all_intrinsics_lst = intrinsics.unsqueeze(0).repeat(all_extrinsics_lst.shape[0], 1, 1).cpu()
            
            ori_img_path = os.path.join(self.opt.dataset_dir, self.feats_folder_name, self.feat_model_name, f'{sha256}_img_feats_oriImgs.npy')
            all_images_lst = torch.from_numpy(np.load(ori_img_path)).float().cpu() / 255. # [N, 3, H, W]
            if all_images_lst.shape[-2:] != (new_h, new_w):
                all_images_lst = F.interpolate(all_images_lst, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            assert all_images_lst.shape[0] == all_extrinsics_lst.shape[0] == all_intrinsics_lst.shape[0] == all_depths_lst.shape[0], \
                f"Shape mismatch: images {all_images_lst.shape}, extrinsics {all_extrinsics_lst.shape}, intrinsics {all_intrinsics_lst.shape}, depths {all_depths_lst.shape}"
                        
            if self.num_rend_views == 3:
                
                # 少选最近视角
                target_idx = random.choice(range(all_images_lst.shape[0]))
                c2w_all = torch.inverse(all_extrinsics_lst)
                closest_indices = find_closest_views(c2w_all, all_intrinsics_lst, target_idx, voxel_points, alpha=1.0, beta=1.0)
                
                all_images_lst = all_images_lst[closest_indices]
                all_extrinsics_lst = all_extrinsics_lst[closest_indices]
                all_intrinsics_lst = all_intrinsics_lst[closest_indices]
                all_depths_lst = all_depths_lst[closest_indices]
                all_ori_extrinsics_lst = all_ori_extrinsics_lst[closest_indices]
                
                # print(voxel_points.shape)
                visibility_mask = torch.zeros(voxel_points.shape[0], dtype=torch.bool)

                for idx in range(len(closest_indices)):
                    # proj_intrinsics_lst = all_intrinsics_lst.clone()
                    # proj_intrinsics_lst[0,:] = all_intrinsics_lst[0,:] * all_depths_lst.shape[-1] # *w
                    # proj_intrinsics_lst[1,:] = all_intrinsics_lst[1,:] * all_depths_lst.shape[-2] # *h
                    # visibility_mask = self.select_visible_points1(voxel_points, all_depths_lst, proj_intrinsics_lst, all_extrinsics_lst,
                    #                                              T_norm=T_norm, 
                    #                                              depth_tol=2*self.vs_size)
                    intrinsic = all_intrinsics_lst[idx]
                    depth_lst = all_depths_lst[idx]
                    extrinsics = all_extrinsics_lst [idx]
                    
                    proj_intrinsics_lst = intrinsic.clone()
                    proj_intrinsics_lst[0,0] /= SCALE 
                    proj_intrinsics_lst[1,1] /= SCALE 
                    proj_intrinsics_lst[0,:] *= depth_lst.shape[-1] # *w
                    proj_intrinsics_lst[1,:] *= depth_lst.shape[-2] # *h
                    
                    mask = self.select_visible_points(voxel_points, depth_lst, proj_intrinsics_lst, extrinsics,
                                                                T_norm=T_norm, 
                                                                depth_tol=2*self.vs_size)
                    
                    # print(f"[Info] For {sha256}, visibility_mask: {visibility_mask.shape}, voxel_points: {voxel_points.shape}, num visible points: {visibility_mask.sum().item()}/{len(visibility_mask)}")
                    visibility_mask = visibility_mask | mask  # 标记所有符合条件的位置为 True

                voxel_points = voxel_points[visibility_mask] 
                                                
                if feats is not None:
                    feats = feats[visibility_mask] 
                    batch_data.update({"feats": feats})
                if voxel_indices is not None:
                    voxel_indices = voxel_indices[visibility_mask] 
                    batch_data.update({"voxel_indices": voxel_indices})
            else:
                if feats is not None:
                    batch_data.update({"feats": feats})
                if voxel_indices is not None:
                    batch_data.update({"voxel_indices": voxel_indices})
            
            if grid_origin is not None:
                batch_data.update({"grid_origin": grid_origin})
            else:
                batch_data.update({"voxel_points": voxel_points})
            
            batch_data.update( {
                                "rand_img_lst":all_images_lst, #.unsqueeze(0) if self.num_rend_views==1 else all_images_lst, 
                                "rand_ori_extrinsics_lst": all_ori_extrinsics_lst, #.unsqueeze(0) if self.num_rend_views==1 else all_ori_extrinsics_lst,
                                "rand_extrinsics_lst": all_extrinsics_lst, #.unsqueeze(0) if self.num_rend_views==1 else all_extrinsics_lst, 
                                "rand_intrinsics_lst": all_intrinsics_lst, #.unsqueeze(0) if self.num_rend_views==1 else all_intrinsics_lst, 
                                "rand_depths_lst": all_depths_lst #.unsqueeze(0) if self.num_rend_views==1 else all_depths_lst,
                        })
            
            # print(f"[INFO] For {sha256}, selected {all_images_lst.shape[0]} views. voxel_points: {voxel_points.shape}, all_images_lst: {all_images_lst.shape}, all_intrinsics_lst: {all_intrinsics_lst.shape}, all_extrinsics_lst: {all_extrinsics_lst.shape}, all_depths_lst: {all_depths_lst.shape}")
            
            # test 测试novel view
            if self.stage in ['test_trainset_novel']:
                
                train_frames = cam_infos['frames'] # [::4]
                test_frames = cam_infos['test_frames'] # [::4]
                print(f"train frames: {len(train_frames)}, test_frames: {len(test_frames)}")

                test_frames_infos = np.load(os.path.join(self.opt.dataset_dir, self.opt.test_frames_folder_name, f'{sha256}_test_frames.npz'))
                novel_img_lst = torch.from_numpy(test_frames_infos['normed_intrinsics'])
                novel_ext_lst = torch.from_numpy(test_frames_infos['extrinsics_Tnorm'])
                novel_img_lst = torch.from_numpy(test_frames_infos['images'])
                novel_depth_lst = torch.from_numpy(test_frames_infos['depthmap'])
                
                # existing_names = set(np.load(os.path.join(self.opt.dataset_dir, self.feats_folder_name, self.feat_model_name, f'{sha256}_img_feats_filenames.npy')).tolist())
                # other_images = [f for f in train_frames if f['file_path'] not in existing_names]
            
                # if len(other_images) >= self.num_rend_views:
                #     sampled_views = random.sample(other_images, self.num_rend_views)
                # else:
                #     sampled_views = other_images  # 如果不足32张，就全取
                # novel_img_lst = []
                # novel_ext_lst = []
                # for view in sampled_views:
                #     img_path = os.path.join(self.opt.dataset_dir, 'renders', sha256, view['file_path'])
                #     img = Image.open(img_path).convert("RGB")  # 确保是3通道
                #     img_tensor = transform_novel(img)  # [C, H, W], float32, 0~1
                #     novel_img_lst.append(img_tensor)
                #     w2c_Tnorm = self.get_extrinsics_for_samples(view, T_norm=T_norm)
                #     novel_ext_lst.append(w2c_Tnorm)
                # novel_img_lst = torch.stack(novel_img_lst, dim=0)
                # novel_ext_lst = torch.stack(novel_ext_lst, dim=0)
                if novel_img_lst.shape[-2:] != (new_h, new_w):
                    novel_img_lst = F.interpolate(novel_img_lst, size=(new_h, new_w), mode='bilinear', align_corners=False)
                if novel_depth_lst.shape[-2:] != (new_h, new_w):
                    novel_depth_lst = F.interpolate(novel_depth_lst, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    
                novel_int_lst = intrinsics.unsqueeze(0).repeat(novel_img_lst.shape[0], 1, 1).cpu()
                # print(f"[INFO] For {sha256}, sampled {len(sampled_views)} novel views.")
                # print(f"novel_img_lst: {novel_img_lst.shape}, novel_int_lst: {novel_int_lst.shape}, novel_ext_lst: {novel_ext_lst.shape}")

                batch_data.update({
                                    "novel_img_lst": novel_img_lst, 
                                    "novel_depths_lst": novel_depth_lst,
                                    # "novel_ori_extrinsics_lst": novel_ori_ext_lst,
                                    "novel_extrinsics_lst": novel_ext_lst, 
                                    "novel_intrinsics_lst": novel_int_lst, 
                            })
                
            return batch_data
                
        except Exception as e:
            raise ValueError(f"Error loading renders images or extrinsics or intrinsics for {sha256}: {e}")

# ---------------------------
# 2. 封装模型到 LightningModule
# ---------------------------
class AppearenceVAEModel(pl.LightningModule):
    
    def on_train_start(self):
        self.model.to(self.device)
    
    def __init__(self, opt):
        super().__init__()
        # self.use_lpips = use_lpips
        self.opt = opt
        self.sh_degree = opt.models.args.representation_config.sh_degree if hasattr(opt.models.args.representation_config, 'sh_degree') else 0
        self.rgb_dim = 3 * ((self.sh_degree + 1) **2)
        self.voxel_align_ratio = opt.models.voxel_align_ratio if hasattr(opt.models, 'voxel_align_ratio') else 5
        self.if_fix_xyz = opt.models.if_fix_xyz if hasattr(opt.models, 'if_fix_xyz') else False
        self.num_GS = opt.models.args.representation_config['num_gaussians']
        self.warmup_epochs = self.opt.trainer.lr_scheduler.args.warmup_steps
        self.vs_size = 2/128
        
        self.model = getattr(models, opt.models.name)(**opt.models.args)
        
        model_summary = get_model_summary(self.model)
        # print(model_summary)
        with open(os.path.join(opt.output_dir, f'model_summary.txt'), 'w') as fp:
            print(model_summary, file=fp)
        
        # if self.use_lpips:
        #     self.lpips_loss = lpips.LPIPS(net='vgg').eval() # .to(self.device)
        #     self.lpips_loss.eval()
        #     for p in self.lpips_loss.parameters():
        #         p.requires_grad = False
            
        self.ssim_metric = StructuralSimilarityIndexMeasure().eval() #.to(outputs[0].device)
        
        self.if_norm_depth = True
        
        # 指定在哪些 epoch 做 inference
        self.infer_epochs = [59, 119, 199] + list(range(199, self.opt.trainer.max_epochs, 200))
                
    # def forward(self, feats):
    #     return self.model(feats)

    def configure_optimizers(self):

        # Initialize optimizer 
        if opt.models.name in ['SLatQuerySepMLPGaussianDecoderOnly', 
                               'SLatQuerySepMLPC2FGaussianDecoderOnly',
                               'SLatQuerySepMLPC2FFGaussianDecoderOnly']:
            mlp_params = [
                {'params': self.model.out_layer_xyz.parameters(), 
                 'lr': self.opt.trainer.out_mlp_lr.lr_xyz, "name": "mlp_xyz"},
                {'params': self.model.out_layer_fdc.parameters(), 
                 'lr': self.opt.trainer.out_mlp_lr.lr_fdc, "name": "mlp_fdc"},
                {'params': self.model.out_layer_sca.parameters(), 
                 'lr': self.opt.trainer.out_mlp_lr.lr_sca, "name": "mlp_sca"},
                {'params': self.model.out_layer_rot.parameters(), 
                 'lr': self.opt.trainer.out_mlp_lr.lr_rot, "name": "mlp_rot"},
                {'params': self.model.out_layer_opa.parameters(), 
                 'lr': self.opt.trainer.out_mlp_lr.lr_opa, "name": "mlp_opa"}
            ]
            
            layer_name = ['out_layer_xyz', 'out_layer_fdc', 'out_layer_sca', 'out_layer_rot', 'out_layer_opa']
            
            if self.opt.models.name in ['SLatQuerySepMLPC2FGaussianDecoderOnly',
                                        'SLatQuerySepMLPC2FFGaussianDecoderOnly']:       
                mlp_params.extend(
                        [
                            {'params': self.model.out_layer2_xyz.parameters(), 
                            'lr': self.opt.trainer.out_mlp_lr.lr_xyz, "name": "mlp_xyz"},
                            {'params': self.model.out_layer2_fdc.parameters(), 
                            'lr': self.opt.trainer.out_mlp_lr.lr_fdc, "name": "mlp_fdc"},
                            {'params': self.model.out_layer2_sca.parameters(), 
                            'lr': self.opt.trainer.out_mlp_lr.lr_sca, "name": "mlp_sca"},
                            {'params': self.model.out_layer2_rot.parameters(), 
                            'lr': self.opt.trainer.out_mlp_lr.lr_rot, "name": "mlp_rot"},
                            {'params': self.model.out_layer2_opa.parameters(), 
                            'lr': self.opt.trainer.out_mlp_lr.lr_opa, "name": "mlp_opa"}
                        ]
                        )
                layer_name.extend(['out_layer2_xyz', 'out_layer2_fdc', 'out_layer2_sca', 'out_layer2_rot', 'out_layer2_opa'])  
                
                if self.opt.models.name in ['SLatQuerySepMLPC2FFGaussianDecoderOnly']:       
                    mlp_params.extend(
                            [
                                {'params': self.model.out_layer3_xyz.parameters(), 
                                'lr': self.opt.trainer.out_mlp_lr.lr_xyz, "name": "mlp_xyz"},
                                {'params': self.model.out_layer3_fdc.parameters(), 
                                'lr': self.opt.trainer.out_mlp_lr.lr_fdc, "name": "mlp_fdc"},
                                {'params': self.model.out_layer3_sca.parameters(), 
                                'lr': self.opt.trainer.out_mlp_lr.lr_sca, "name": "mlp_sca"},
                                {'params': self.model.out_layer3_rot.parameters(), 
                                'lr': self.opt.trainer.out_mlp_lr.lr_rot, "name": "mlp_rot"},
                                {'params': self.model.out_layer3_opa.parameters(), 
                                'lr': self.opt.trainer.out_mlp_lr.lr_opa, "name": "mlp_opa"}
                            ]
                            )
                    layer_name.extend(['out_layer3_xyz', 'out_layer3_fdc', 'out_layer3_sca', 'out_layer3_rot', 'out_layer3_opa'])              
        
            
            other_params = [param for name, param in self.model.named_parameters() 
                                if not any(mlp_layer in name for mlp_layer in layer_name)]
            other_param_group = [{'params': other_params, 'lr': opt.trainer.lr, "name": "others"}]

            model_params = mlp_params + other_param_group
        else:
            model_params = self.model.parameters() # [param for _, param in model.named_parameters()]

        if hasattr(torch.optim, self.opt.trainer.optimizer['name']):
            optimizer = getattr(torch.optim,  self.opt.trainer.optimizer['name'])(model_params, **self.opt.trainer.optimizer['args'])

        # Initalize learning rate scheduler 
        if self.opt.trainer.lr_scheduler is not None:
            logger.info(f"[INFO] lr_scheduler: {self.opt.trainer.lr_scheduler['name']}, {self.opt.trainer.lr_scheduler['args']}")
            if hasattr(torch.optim.lr_scheduler, self.opt.trainer.lr_scheduler['name']):
                scheduler = getattr(torch.optim.lr_scheduler, self.opt.trainer.lr_scheduler['name'])(optimizer, **self.opt.trainer.lr_scheduler['args'])
            else:
                scheduler = cosine_scheduler_with_warmup(optimizer, **self.opt.trainer.lr_scheduler['args'])

        return [optimizer], [scheduler]

    # =====================================================
    # 统一 forward（推理逻辑）
    # =====================================================
    def forward(self, batch, query=None):
        sha256 = batch.get("sha256")
        SCALE = batch.get("scale").item()
        voxel_indices = batch.get("voxel_indices", None) # .to("cpu")
        voxel_points = batch.get("voxel_points", None)
        grid_origin = batch.get("grid_origin", None)
        rand_img_lst = batch.get("rand_img_lst") # .to("cpu")
        rand_extrinsics_lst = batch.get("rand_extrinsics_lst")
        rand_ori_extrinsics_lst = batch.get("rand_ori_extrinsics_lst")
        rand_intrinsics_lst = batch.get("rand_intrinsics_lst")
        rand_depths_lst = batch.get("rand_depths_lst")
        T_norm = batch.get("T_norm", None).to("cpu")  # 显式加默认值
        T_norm_inv = batch.get("T_norm_inv", None).to("cpu")  # 显式加默认值
        if voxel_points is not None: 
            grid_origin = None # 如果使用了voxel_points, 就不需要grid_origin来反算xyz了
            voxel_points = voxel_points[0]
        if "norm" in self.opt.feats_folder_name:
            assert T_norm is not None, "T_norm should not be None if using norm128 features"
            assert T_norm_inv is not None, "T_norm_inv should not be None if using norm128 features"
            assert voxel_points is not None, "voxel_points should not be None if using norm128 features"
            assert grid_origin is None, "grid_origin should be None if using norm128 features"
            T_norm = T_norm[0]
        else:
            assert T_norm is None, "T_norm should be None if using norm128 features"
            assert T_norm_inv is None, "T_norm_inv should be None if not using norm128 features"    
        # rand_depths_lst[rand_depths_lst < 0] = 0  # 将无效深度值 <0 替换为 0
                        
        # bs = feats.shape[0] # 这里我希望bs=1, 因为trellis那个sparsetensor之后的处理没有考虑多batch的操作
        assert len(rand_img_lst) == len(rand_extrinsics_lst) == len(rand_intrinsics_lst) == 1

        sha256 = sha256[0]
        rand_img_lst = rand_img_lst[0] #.to("cpu")
        rand_intrinsics_lst = rand_intrinsics_lst[0].contiguous()
        rand_extrinsics_lst = rand_extrinsics_lst[0].contiguous()
        rand_ori_extrinsics_lst = rand_ori_extrinsics_lst[0]
        rand_depths_lst = rand_depths_lst[0].contiguous().unsqueeze(1) # shape: [N, 1, H, W]
        
        query = None

        model_name = self.opt.models.name

        if model_name in ['SLatQuerySepMLPC2FGaussianDecoderOnly',
                          'SLatQuerySepMLPC2FFGaussianDecoderOnly']:
            feats = batch.get("feats")
            feats = sp.SparseTensor(
                feats=feats.squeeze(0).float(),
                coords=torch.cat([
                    torch.zeros(feats.squeeze(0).shape[0], 1).int().to(self.device),
                    voxel_indices.squeeze(0).int().to(self.device),
                ], dim=1),
            )
            outputs, _ = self.model(
                feats, query=query,
                grid_origin=None,
                voxel_points=voxel_points,
                voxel_align_ratio=self.voxel_align_ratio,
                if_fix_xyz=self.if_fix_xyz
            )
            # del feats

        elif model_name in ['SLatQueryRGBCNNCamDepthInSepMLPGaussianDecoderOnly']:
            voxel_indices = voxel_indices[0]
            outputs = self.model(
                feats=None,
                rgbOris=rand_img_lst,
                depths=rand_depths_lst,
                intrinsic=rand_intrinsics_lst.clone(),
                w2cs=rand_extrinsics_lst,
                ori_w2cs=rand_ori_extrinsics_lst,
                scale=SCALE,
                T_norm=T_norm,
                grid_origin=None,
                voxel_points=voxel_points,
                voxel_indices=voxel_indices,
                # vs_size=self.vs_size, 
                voxel_align_ratio=self.voxel_align_ratio,
                if_fix_xyz=self.if_fix_xyz
            )

        elif model_name in ['SLatQueryCamDepthInSepMLPGaussianDecoderOnly']:
            feats = batch.get("feats")
            feats = sp.SparseTensor(
                feats=feats.squeeze(0).float(),
                coords=torch.cat([
                    torch.zeros(feats.squeeze(0).shape[0], 1).int().to(self.device),
                    voxel_indices.squeeze(0).int().to(self.device),
                ], dim=1),
            )
            outputs = self.model(
                feats, query=query,
                depths=rand_depths_lst,
                intrinsic=rand_intrinsics_lst.clone(),
                w2cs=rand_extrinsics_lst,
                ori_w2cs=rand_ori_extrinsics_lst,
                scale=SCALE,
                T_norm=T_norm,
                grid_origin=None,
                voxel_points=voxel_points,
                voxel_align_ratio=self.voxel_align_ratio,
                if_fix_xyz=self.if_fix_xyz
            )
            # del feats

        else:
            feats = batch.get("feats")
            feats = sp.SparseTensor(
                feats=feats.squeeze(0).float(),
                coords=torch.cat([
                    torch.zeros(feats.squeeze(0).shape[0], 1).int().to(self.device),
                    voxel_indices.squeeze(0).int().to(self.device),
                ], dim=1),
            )
            outputs = self.model(
                feats, query=query,
                grid_origin=None,
                voxel_points=voxel_points,
                voxel_align_ratio=self.voxel_align_ratio,
                if_fix_xyz=self.if_fix_xyz
            )
            # del feats
        
        # del grid_origin, voxel_points, voxel_indices, rand_ori_extrinsics_lst 
        gc.collect()
        torch.cuda.empty_cache()

        return {"sha256": sha256, 
                "outputs": outputs, 
                "rand_img_lst": rand_img_lst, 
                "rand_depths_lst": rand_depths_lst,
                "rand_extrinsics_lst": rand_extrinsics_lst,
                "rand_intrinsics_lst": rand_intrinsics_lst}
    
    def compute_rend_loss_per_frame(self, sha256, rend_rgb, gt_rgb, rend_depth=None, gt_depth=None, sv_mask=None):
        """
        计算单视角的渲染 loss 和 psnr
        """
        device = self.device
        SCALE = self.opt.scale if hasattr(self.opt, 'scale') else 1.0
        new_w, new_h = self.opt.img_resolution
        
        # ==========================================================================================================================
        # ----------------- loss: img reconstruction ---------------------------------------------------------
        if torch.isnan(rend_rgb).any() or torch.isnan(gt_rgb).any():
            raise ValueError(f"NaN detected in inputs! sha256 is {sha256}")

        if self.opt.use_depth_mask and sv_mask is not None:
            diff = torch.abs(rend_rgb - gt_rgb)
            loss_img_rec_l1 = self.opt.trainer.loss_weight_img_rec_l1 * diff.masked_select(sv_mask).mean()
        else:
            loss_img_rec_l1 = self.opt.trainer.loss_weight_img_rec_l1 * l1_loss(rend_rgb, gt_rgb)
        
        if self.opt.use_depth_mask and sv_mask is not None:
            ssim_score = masked_ssim_batch(rend_rgb.unsqueeze(0), gt_rgb.unsqueeze(0), sv_mask.unsqueeze(0))                
            loss_img_rec_ssim = self.opt.trainer.loss_weight_img_rec_ssim * (1 - ssim_score[0])
        else:
            self.ssim_metric.update(rend_rgb.unsqueeze(0), gt_rgb.unsqueeze(0))
            loss_img_rec_ssim = self.opt.trainer.loss_weight_img_rec_ssim * (1- self.ssim_metric.compute()) 
            self.ssim_metric.reset()
        
        # loss_img_rec_lpips = self.opt.trainer.loss_weight_img_rec_lpips * self.lpips_loss(rend_rgb, gt_rgb).mean()

        if gt_depth is not None and rend_depth is not None and sv_mask is not None:
            loss_depth_rec_l1 = self.opt.trainer.loss_weight_depth_l1 * l1_loss(rend_depth, gt_depth, sv_mask)
        else:
            loss_depth_rec_l1 = None
                        
        # ==========================================================================================================================
        if self.opt.use_depth_mask and sv_mask is not None:
            diff = ((rend_rgb - gt_rgb) ** 2).masked_select(sv_mask)
            if diff.numel() == 0 or torch.any(torch.isnan(diff)):
                mse = torch.tensor(1e-6, device=rend_rgb.device)  # 防止空mask导致nan
            else:
                mse = diff.mean().clamp(min=1e-6)  # 避免除0或太小
        else:
            mse = F.mse_loss(rend_rgb, gt_rgb).clamp(min=1e-6)
        psnr = 10 * torch.log10(1.0 / mse)
        
        return loss_img_rec_l1, loss_img_rec_ssim, None, loss_depth_rec_l1, psnr

    def compute_loss(self, outputs, rand_img_lst, rand_depths_lst,
                     rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                     batch_idx, sha256, mode='train'): # mode='val'
        """
        计算完整训练 loss，包括:
        - Gaussian opacity & scale 正则
        - GS dispersion & diversity loss
        - 渲染后的 RGB + Depth 重建 loss (L1 + SSIM)
        - PSNR 评价指标
        """
        device = self.device
        new_w, new_h = self.opt.img_resolution
        
        # start_time = time.time()
        # ------------------------- 预处理 -------------------------
        rand_depths_lst = F.interpolate(rand_depths_lst, size=(new_h, new_w),
                                        mode='bilinear', align_corners=False).squeeze(1).to("cpu")
        rand_img_lst = rand_img_lst.to("cpu")  # [N, 3, H, W]
        depth_max = rand_depths_lst.max().item()

        if self.if_norm_depth:
            rand_depths_lst = rand_depths_lst / depth_max

        if self.opt.use_depth_mask:
            mask = rand_depths_lst <= 1e-6
            rand_img_lst = rand_img_lst.masked_fill(mask.unsqueeze(1), 0)

        if self.opt.trainer.render_option == 'gsplat':
            rand_intrinsics_lst[:, 0, :] *= new_w
            rand_intrinsics_lst[:, 1, :] *= new_h
        # end_time = time.time()
        # print("耗时 compute_loss - 预处理: {:.2f}秒".format(end_time - start_time))

        # start_time = time.time()
        # ------------------------- Gaussian Loss -------------------------
        loss_opa, loss_scale = 0, 0
        loss_gs_dispersion = torch.tensor(0.0, device=device)
        loss_gs_diversity = torch.tensor(0.0, device=device)

        for output in outputs:
            # opacity & scale
            loss_opa += self.opt.trainer.loss_weight_opacity * ((1 - output.get_opacity) ** 2).mean()
            loss_scale += self.opt.trainer.loss_weight_scale * torch.prod(output.get_scaling, dim=1).mean()

            # print(output.get_scaling.mean(dim=0))

            # GS dispersion
            if hasattr(self.opt.trainer, "loss_weight_GSdispersion") and self.opt.trainer.loss_weight_GSdispersion > 0:
                weight_disp = 0 if self.trainer.current_epoch < self.warmup_epochs else self.opt.trainer.loss_weight_GSdispersion
                loss_gs_dispersion += weight_disp * dispersion_loss_voxel_scaled(
                    output.get_xyz.reshape(-1, self.num_GS, 3),
                    margin_rel=0.5, voxel_size=2/128, beta=20, activation='softplus'
                )

            # GS diversity
            if hasattr(self.opt.trainer, "loss_weight_GSdiversity") and self.opt.trainer.loss_weight_GSdiversity > 0:
                if self.trainer.current_epoch > self.warmup_epochs:
                    feat_div = torch.cat([
                        output.get_features.reshape(-1, self.num_GS, self.rgb_dim),
                        output.get_opacity.reshape(-1, self.num_GS, 1)
                    ], dim=-1)
                    current_margin = 0.02 + 0.02 * min(self.trainer.current_epoch / 200, 1.0)
                    loss_gs_diversity += self.opt.trainer.loss_weight_GSdiversity * feature_diversity_loss(
                        feat_div, mode="pairwise_l2", margin=current_margin
                    )
        # end_time = time.time()
        # print("耗时 compute_loss - Gaussian Loss: {:.2f}秒".format(end_time - start_time))

        # ------------------------- 渲染 loss -------------------------
        loss_rec, psnr, loss_depth = 0, 0, 0
        all_train_videos, all_train_depths, sv_masks = [], [], []

        num_valid_view = len(rand_extrinsics_lst) if len(rand_extrinsics_lst.shape) == 3 else 1
        
        if mode in ['val', 'test']:
            # start_time = time.time()
            results = render_utils.render_gtpose(outputs[0], 
                                            resolution=(new_h, new_w), 
                                            extrinsics=rand_extrinsics_lst, #.to(outputs[0].device), 
                                            intrinsics=rand_intrinsics_lst, #.to(outputs[0].device), 
                                            render_option=self.opt.trainer.render_option,
                                            AbsGrad=self.opt.trainer.get("AbsGrad", False),
                                            is_infer=True,
                                            render_mode='RGB+D',
                                            sh_degree=self.sh_degree,
                                            verbose=False)
            all_train_videos = torch.from_numpy(results['color']) # [V, H, W, 3]
            all_train_depths = torch.from_numpy(results['depth']).squeeze(-1) / SCALE # [V, H, W]
            if self.if_norm_depth:
                all_train_depths = all_train_depths/depth_max
            
            if self.opt.trainer.render_option == 'gsplat':
                all_train_videos = all_train_videos.permute(0, 3, 1, 2).contiguous() / 255.
            if self.opt.trainer.render_option == 'diff-gaussian-rasterization':
                all_train_videos = torch.cat([frame.unsqueeze(0) for frame in all_train_videos], dim=0)    
            # end_time = time.time()
            # print("耗时 compute_loss - Rendering: {:.2f}秒".format(end_time - start_time))
            
            # start_time = time.time()
            for i in range(len(all_train_videos)):
                gt_rgb = rand_img_lst[i].to(self.device)   
                rend_rgb = all_train_videos[i].to(self.device)   
                gt_depth = rand_depths_lst[i].to(self.device)    # [1, H, W]   
                rend_depth = all_train_depths[i].unsqueeze(0).to(self.device)    # [1, H, W] 
                sv_mask = (gt_depth > 1e-6) # [1, H, W] 
                if self.opt.use_depth_mask:
                    gt_rgb = gt_rgb.masked_fill(~sv_mask, 0)
                    
                rend_rgb = rend_rgb.clamp(0, 1)
                gt_rgb = gt_rgb.clamp(0, 1)
                loss_img_rec_l1, loss_img_rec_ssim, _, loss_depth_rec_l1, psnr_frame = self.compute_rend_loss_per_frame(sha256, rend_rgb, gt_rgb, rend_depth, gt_depth, sv_mask)
                loss_rec += loss_img_rec_l1 + loss_img_rec_ssim # + loss_img_rec_lpips           
                loss_depth += loss_depth_rec_l1
                psnr += psnr_frame
            
            all_train_depths = all_train_depths.unsqueeze(1) # [V, 1, H, W]
            
            # end_time = time.time()
            # print("耗时 compute_loss - rend_loss_frames: {:.2f}秒".format(end_time - start_time))

        elif mode == 'train':
            # 节省显存
            rand_extrinsics_lst = rand_extrinsics_lst.to("cpu")
            rand_intrinsics_lst = rand_intrinsics_lst.to("cpu")
            
            for i in range(num_valid_view):

                rand_ext = rand_extrinsics_lst[i].unsqueeze(0).to(device)
                rand_int = rand_intrinsics_lst[i].unsqueeze(0).to(device)

                results = render_utils.render_gtpose(
                    outputs[0],
                    resolution=(new_h, new_w),
                    extrinsics=rand_ext,
                    intrinsics=rand_int,
                    render_option=self.opt.trainer.render_option,
                    AbsGrad=self.opt.trainer.get("AbsGrad", False),
                    render_mode="RGB+D",
                    sh_degree=self.sh_degree,
                    is_infer=False,
                    verbose=False
                )

                video = results['color']
                depths = results['depth'].squeeze(-1) / SCALE
                if self.if_norm_depth:
                    depths = depths / depth_max

                if self.opt.trainer.render_option == 'gsplat':
                    video = video.permute(0, 3, 1, 2)
                elif self.opt.trainer.render_option == 'diff-gaussian-rasterization':
                    video = torch.cat([frame.unsqueeze(0) for frame in video], dim=0)
                    depths = torch.cat([frame.unsqueeze(0) for frame in depths], dim=0)

                # 保存用于可视化
                if self.trainer.current_epoch % self.opt.trainer.save_img_every_epoch == 0 and batch_idx % 10 == 0:
                    all_train_videos.append(video.detach().cpu())
                    all_train_depths.append(depths.detach().cpu())
                
                gt_rgb = rand_img_lst[i].to(device)
                rend_rgb = video[0].to(device)
                gt_depth = rand_depths_lst[i].to(device).unsqueeze(0)
                rend_depth = depths[0].unsqueeze(0).to(device)

                sv_mask = gt_depth > 1e-6
                if self.opt.use_depth_mask:
                    gt_rgb = gt_rgb.masked_fill(~sv_mask, 0)
                    sv_masks.append(sv_mask.float().repeat(3, 1, 1).unsqueeze(0))

                rend_rgb = rend_rgb.clamp(0, 1)
                gt_rgb = gt_rgb.clamp(0, 1)
                loss_img_rec_l1, loss_img_rec_ssim, _, loss_depth_rec_l1, psnr_frame = self.compute_rend_loss_per_frame(sha256, rend_rgb, gt_rgb, rend_depth, gt_depth, sv_mask)
                loss_rec += loss_img_rec_l1 + loss_img_rec_ssim # + loss_img_rec_lpips              
                loss_depth += loss_depth_rec_l1
                psnr += psnr_frame
                
            if len(all_train_videos)>0:
                all_train_videos = torch.cat(all_train_videos, dim=0)
            
            # del video, rand_ext, rand_int, depths

        # ------------------------- 平均 -------------------------
        # print(f"[DEBUG] num_valid_view: {num_valid_view}")
        avg_psnr = psnr / num_valid_view
        avg_loss_rec = loss_rec / num_valid_view
        avg_loss_depth = loss_depth / num_valid_view

        # ------------------------- 总 loss -------------------------
        loss_total = avg_loss_rec + loss_opa + loss_scale + avg_loss_depth + loss_gs_dispersion + loss_gs_diversity

        # start_time = time.time()
        # ------------------------- 日志 -------------------------
        if mode == 'train':
            tag = "Training" 
        elif mode == 'val':
            tag = "Validation"
        elif mode == 'test':
            tag = "Test"
        log_dict = {
            f"{tag}/{mode}_loss_total": loss_total.item(),
            f"{tag}/{mode}_loss_rec_avg": avg_loss_rec.item(),
            f"{tag}/{mode}_loss_opa": loss_opa.item(),
            f"{tag}/{mode}_loss_scale": loss_scale.item(),
            f"{tag}/{mode}_loss_depth_avg": avg_loss_depth.item()
            # f"{tag}/{mode}_avg_psnr_supervised_views": avg_psnr.item()
        }
        if mode == 'train':
            log_dict[f"{tag}/{mode}_avg_psnr_supervised_views"] = avg_psnr.item()
        elif mode == 'val':
            log_dict[f"{tag}/{mode}_avg_psnr"] = avg_psnr.item()
            
        if hasattr(self.opt.trainer, "loss_weight_GSdispersion") and self.opt.trainer.loss_weight_GSdispersion > 0:
            log_dict[f"{tag}/{mode}_loss_gs_dispersion"] = loss_gs_dispersion.item()
        if hasattr(self.opt.trainer, "loss_weight_GSdiversity") and self.opt.trainer.loss_weight_GSdiversity > 0:
            log_dict[f"{tag}/{mode}_loss_gs_diversity"] = loss_gs_diversity.item()

        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False if mode == 'train' else True)
        logger.info(f"[{tag}] epoch: {self.trainer.current_epoch}, sha256: {sha256}, {mode}_log_dict = " + json.dumps(log_dict, indent=2))
        # print(f"[{tag}] epoch: {self.trainer.current_epoch}, sha256: {sha256}, {mode}_log_dict = " + json.dumps(log_dict, indent=2))
        # end_time = time.time()
        # print("耗时 compute_loss - logging: {:.2f}秒".format(end_time - start_time))
        
        # start_time = time.time()
        # ------------------------- 可视化 -------------------------
        train_save_trigger = (mode == 'train' and self.trainer.current_epoch % (2*self.opt.trainer.save_img_every_epoch) == 0 and len(all_train_videos) > 0) 
        if train_save_trigger or (mode == 'val') or (mode == 'test' and self.trainer.current_epoch in self.infer_epochs):

            save_image(all_train_videos, os.path.join(self.logger.log_dir, tag.lower(),
                        f"{sha256}_{mode}_rgb_image.png")) # epoch_{self.trainer.current_epoch}
            
            save_image(rand_img_lst, os.path.join(self.logger.log_dir, tag.lower(),
                    f"{sha256}_{mode}_GT_rgb_image.png")) # epoch_{self.trainer.current_epoch}     
            
            save_image(all_train_depths, os.path.join(self.logger.log_dir, tag.lower(),
                        f"{sha256}_{mode}_depth.png")) # epoch_{self.trainer.current_epoch}
            
            save_image(rand_depths_lst.unsqueeze(1), os.path.join(self.logger.log_dir, tag.lower(),
                    f"{sha256}_{mode}_GT_depth.png")) # epoch_{self.trainer.current_epoch}
            
            outputs[0].save_ply(os.path.join(self.logger.log_dir, tag.lower(),
                    f'{sha256}_{mode}_pred_GS.ply')) # epoch_{self.trainer.current_epoch}
        # end_time = time.time()
        # print("耗时 compute_loss - saving: {:.2f}秒".format(end_time - start_time))

        # del all_train_videos, all_train_depths
        # del rand_img_lst, gt_rgb, rend_rgb, rand_extrinsics_lst, rand_intrinsics_lst
        # del gt_depth, rend_depth, sv_mask, rand_depths_lst
        # del avg_loss_rec, loss_rec, loss_img_rec_l1, loss_img_rec_ssim, loss_depth_rec_l1, loss_depth, avg_loss_depth
        # del loss_scale, loss_opa, avg_psnr, psnr, psnr_frame, loss_gs_dispersion, loss_gs_diversity  # , loss_img_rec_lpips #, loss_kl 
        # del outputs

        return loss_total
    
    def compute_psnr_for_test(self, outputs, rand_img_lst, rand_depths_lst,
                     rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                     batch_idx, sha256, mode='test', ifsave=False): # mode='val'
        """
        计算完整训练 loss，包括:
        - Gaussian opacity & scale 正则
        - GS dispersion & diversity loss
        - 渲染后的 RGB + Depth 重建 loss (L1 + SSIM)
        - PSNR 评价指标
        """
        device = self.device
        new_w, new_h = self.opt.img_resolution
        # ------------------------- 预处理 -------------------------
        rand_depths_lst = F.interpolate(rand_depths_lst, size=(new_h, new_w),
                                        mode='bilinear', align_corners=False).squeeze(1).to("cpu")
        rand_img_lst = rand_img_lst.to("cpu")  # [N, 3, H, W]
        if rand_depths_lst is not None:
            depth_max = rand_depths_lst.max().item()
            if self.if_norm_depth:
                rand_depths_lst = rand_depths_lst / depth_max
            if self.opt.use_depth_mask:
                mask = rand_depths_lst <= 1e-6
                rand_img_lst = rand_img_lst.masked_fill(mask.unsqueeze(1), 0)

        if self.opt.trainer.render_option == 'gsplat':
            rand_intrinsics_lst[:, 0, :] *= new_w
            rand_intrinsics_lst[:, 1, :] *= new_h
        
        # ------------------------- 渲染 loss -------------------------
        all_train_videos, all_train_depths, sv_masks = [], [], []
        
        results = render_utils.render_gtpose(outputs[0], 
                                        resolution=(new_h, new_w), 
                                        extrinsics=rand_extrinsics_lst, #.to(outputs[0].device), 
                                        intrinsics=rand_intrinsics_lst, #.to(outputs[0].device), 
                                        render_option=self.opt.trainer.render_option,
                                        AbsGrad=self.opt.trainer.get("AbsGrad", False),
                                        is_infer=True,
                                        render_mode='RGB+D',
                                        sh_degree=self.sh_degree,
                                        verbose=False)
        all_train_videos = torch.from_numpy(results['color']) # [V, H, W, 3]
        all_train_depths = torch.from_numpy(results['depth']).squeeze(-1) / SCALE # [V, H, W]
        
        if rand_depths_lst is not None and self.if_norm_depth:
            all_train_depths = all_train_depths/depth_max
        
        if self.opt.trainer.render_option == 'gsplat':
            all_train_videos = all_train_videos.permute(0, 3, 1, 2).contiguous() / 255.
        if self.opt.trainer.render_option == 'diff-gaussian-rasterization':
            all_train_videos = torch.cat([frame.unsqueeze(0) for frame in all_train_videos], dim=0)    
        
        psnr = 0
        for i in range(len(all_train_videos)):
            gt_rgb = rand_img_lst[i].to(self.device)   
            rend_rgb = all_train_videos[i].to(self.device)   
            rend_depth = all_train_depths[i].unsqueeze(0).to(self.device)    # [1, H, W] 
            
            if rand_depths_lst is not None:
                gt_depth = rand_depths_lst[i].to(self.device)    # [1, H, W] 
                sv_mask = (gt_depth > 1e-6) # [1, H, W]   
                if self.opt.use_depth_mask:
                    gt_rgb = gt_rgb.masked_fill(~sv_mask, 0)
            else:
                gt_depth, sv_mask = None, None

            rend_rgb = rend_rgb.clamp(0, 1)
            gt_rgb = gt_rgb.clamp(0, 1)
            _, _, _, _, psnr_frame = self.compute_rend_loss_per_frame(sha256, rend_rgb, gt_rgb, rend_depth, gt_depth, sv_mask)
            psnr += psnr_frame
        
        all_train_depths = all_train_depths.unsqueeze(1) # [V, 1, H, W]

        # ------------------------- 平均 -------------------------
        avg_psnr = psnr / len(all_train_videos)

        # ------------------------- 日志 -------------------------
        tag = "Test"
        log_dict = {f"{tag}/{mode}_avg_psnr":  avg_psnr.item()}
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False if mode == 'train' else True)
        logger.info(f"[{tag}] epoch: {self.trainer.current_epoch}, sha256: {sha256}, {mode}_log_dict = " + json.dumps(log_dict, indent=2))
        
        if ifsave:
            # ------------------------- 可视化 -------------------------
            save_image(all_train_videos, os.path.join(self.logger.log_dir, tag.lower(),
                        f"{sha256}_{mode}_rgb_image.png")) # epoch_{self.trainer.current_epoch}
            
            save_image(rand_img_lst, os.path.join(self.logger.log_dir, tag.lower(),
                    f"{sha256}_{mode}_GT_rgb_image.png")) # epoch_{self.trainer.current_epoch}     
            
            save_image(all_train_depths, os.path.join(self.logger.log_dir, tag.lower(),
                        f"{sha256}_{mode}_depth.png")) # epoch_{self.trainer.current_epoch}
            
            save_image(rand_depths_lst.unsqueeze(1), os.path.join(self.logger.log_dir, tag.lower(),
                    f"{sha256}_{mode}_GT_depth.png")) # epoch_{self.trainer.current_epoch}
            
            # outputs[0].save_ply(os.path.join(self.logger.log_dir, tag.lower(),
            #         f'{sha256}_{mode}_pred_GS.ply')) # epoch_{self.trainer.current_epoch}

        # del all_train_videos, all_train_depths
        # del rand_img_lst, gt_rgb, rend_rgb, rand_extrinsics_lst, rand_intrinsics_lst
        # del gt_depth, rend_depth, sv_mask, rand_depths_lst
        # del psnr, psnr_frame
        # del outputs

        return avg_psnr

    
    def training_step(self, batch, batch_idx):
        # torch.autograd.set_detect_anomaly(True)
        self.model.train()
        SCALE = batch.get("scale").item()
        
        torch.cuda.empty_cache()
        
        out = self.forward(batch)
        
        sha256 = out.get("sha256")
        outputs = out.get("outputs")    
        rand_depths_lst = out.get("rand_depths_lst") # shape: [N, 1, H, W]
        rand_img_lst = out.get("rand_img_lst") # shape: [N, 3, H, W]
        rand_extrinsics_lst = out.get("rand_extrinsics_lst") # shape: [N, 4, 4]
        rand_intrinsics_lst = out.get("rand_intrinsics_lst") # shape: [N, 3, 3]
        # print(f"[RANK {self.global_rank}] scene: {sha256}, rand_depths_lst: {rand_depths_lst.shape} ")
        loss_total = self.compute_loss(outputs, rand_img_lst, rand_depths_lst,
                                       rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                                       batch_idx, sha256, mode='train')
        
        return loss_total

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        SCALE = batch.get("scale").item()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            # start_time = time.time()
            out = self.forward(batch)
            # end_time = time.time()
            # print("耗时 forward: {:.2f}秒".format(end_time - start_time))
            
            sha256 = out.get("sha256")
            outputs = out.get("outputs")    
            rand_depths_lst = out.get("rand_depths_lst") # shape: [N, 1, H, W]
            rand_img_lst = out.get("rand_img_lst") # shape: [N, 3, H, W]
            rand_extrinsics_lst = out.get("rand_extrinsics_lst") # shape: [N, 4, 4]
            rand_intrinsics_lst = out.get("rand_intrinsics_lst") # shape: [N, 3, 3]
            # print(f"[RANK {self.global_rank}] scene: {sha256}, rand_depths_lst: {rand_depths_lst.shape} ")
            assert len(rand_extrinsics_lst) == len(rand_intrinsics_lst) == len(rand_depths_lst) == len(rand_img_lst)
            
            # start_time = time.time()
            loss_total = self.compute_loss(outputs, rand_img_lst, rand_depths_lst,
                                        rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                                        batch_idx, sha256, mode='val')
            # end_time = time.time()
            # print("耗时 compute_loss: {:.2f}秒".format(end_time - start_time))
        
    def on_test_start(self):
        # 每次 test 前清空存储
        self.test_outputs = {}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.eval()
        SCALE = batch.get("scale").item()
        
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            
            out = self.forward(batch)
            
            sha256 = out.get("sha256")
            outputs = out.get("outputs")    
            rand_depths_lst = out.get("rand_depths_lst") # shape: [N, 1, H, W]
            rand_img_lst = out.get("rand_img_lst") # shape: [N, 3, H, W]
            rand_extrinsics_lst = out.get("rand_extrinsics_lst") # shape: [N, 4, 4]
            rand_intrinsics_lst = out.get("rand_intrinsics_lst") # shape: [N, 3, 3]
            # print(f"[RANK {self.global_rank}] scene: {sha256}, rand_depths_lst: {rand_depths_lst.shape} ")

            if hasattr(batch, "novel_extrinsics_lst") and hasattr(batch, "novel_img_lst") and hasattr(batch, "novel_intrinsics_lst"):
                rand_extrinsics_lst = batch.get("novel_extrinsics_lst")
                rand_img_lst = batch.get("novel_img_lst")
                rand_intrinsics_lst = batch.get("novel_intrinsics_lst")

            avg_psnr = self.compute_psnr_for_test(outputs, rand_img_lst, rand_depths_lst,
                                                    rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                                                    batch_idx, sha256, mode=f'test_{dataloader_idx}', ifsave=True)
            
            # 收集结果
            if dataloader_idx not in self.test_outputs:
                self.test_outputs[dataloader_idx] = []
            
            self.test_outputs[dataloader_idx].append({"scene": sha256, "avg_psnr": avg_psnr.detach()})
            # print(f"dataloader_idx: {dataloader_idx}, scene: {sha256}, test avg_psnr: {avg_psnr.detach()}")
            
            if self.opt.demo:
                interp_folder = os.path.join(self.logger.log_dir, f'demo_dataloader{dataloader_idx}', f'{sha256}_interp_views')
                os.makedirs(interp_folder, exist_ok=True)
                voxel_points = batch.get("voxel_points")[0]
                center = voxel_points.mean(dim=0, keepdim=True)
                gen_demo(outputs[0], center, self.opt.trainer.render_option, 
                        device=self.device, new_w=480, new_h=268, 
                        interp_folder=interp_folder)

            return {"sha256": sha256, "avg_psnr": avg_psnr.detach()}
    
    def on_test_epoch_end(self):
        for idx, results in self.test_outputs.items():
            avg_psnr = torch.stack([x["avg_psnr"] for x in results]).mean()
            num_scenes = len(results)
            print(f"[Prediction dataload_idx {idx}] num_scenes={num_scenes:.4f}, avg_psnr={avg_psnr:.4f}")
        self.log(f"test/avg_psnr_data{idx}", avg_psnr, prog_bar=True, sync_dist=True)
        
        self.test_outputs.clear() # 清空，避免多个 epoch 累加
            
# %%
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    # ======================================================
    # 1. args & cfg
    # ======================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/Slat_vae_full/exp_04_27_pl.json',
                        help='Path to the config file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the output')
    parser.add_argument('--select_scene', type=str, default=None, # 'e0d1ed2159'
                        help='the scenes to calculate')     
    parser.add_argument('--use_depth_mask', action='store_true',
                        help='whether to use depth mask')    
    parser.add_argument('--waitn', type=int, default=9,
                        help='2GPUs=18, 3GPUs=12, 4GPUs=9')
    parser.add_argument('--test', action='store_true', help='whether to test only')
    parser.add_argument('--demo', action='store_true', help='whether to generate demo video only')
    args = parser.parse_args()
    
    config = json.load(open(args.config_path, 'r'))
    opt = edict()
    opt.update(args.__dict__)
    opt.update(config)
    
    if opt.select_scene is not None and opt.output_dir is None:
        opt.output_dir = './outputs_TRELLISdev/outputs_Slat_vae_single/' + f'{opt.select_scene}_' + opt.config_path.split('/')[-1].split('.')[0]+'_accumulateGrad_3viewTrain' # outputs_Slat_vae_full
    elif opt.select_scene is None and opt.output_dir is None:
        # opt.output_dir = './outputs_TRELLISdev/outputs_Slat_vae_full/' + args.config_path.split('/')[-1].split('.')[0]+'_accumulateGrad_resume' # outputs_Slat_vae_full
        opt.output_dir = './outputs_TRELLISdev/outputs_Slat_vae_full/' + opt.config_path.split('/')[-1].split('.')[0] + f'_DpMask{opt.use_depth_mask}_' + 'accumulateGrad_3viewTrain' # outputs_Slat_vae_full
    else:
        opt.output_dir = opt.output_dir + f'{opt.select_scene}_' + opt.config_path.split('/')[-1].split('.')[0]+'_accumulateGrad_perviewTrain' # outputs_Slat_vae_full
    os.makedirs(opt.output_dir, mode=0o777, exist_ok=True)
    
    exp_dir = opt.output_dir
    TBlogger = TensorBoardLogger(save_dir=exp_dir, name="logs", version=0)
    
    logger.remove(handler_id=None)
    logger.add(os.path.join(exp_dir, "loss.log"), level="INFO")
    logger.info(f"opt: {opt}")
    
    # save_config(opt, save_path=os.path.join(opt.output_dir, args.config_path.split('/')[-1]))
    logger.info('\n\nConfig:')
    logger.info('=' * 80)
    json_str = json.dumps(opt.__dict__, indent=4)  # 将对象转为格式化的 JSON 字符串
    with open(exp_dir+'/opt_config.json', 'w') as f: # 将 JSON 字符串保存到文件
        f.write(json_str)
    logger.info(json_str)
    
    try:
        saveRuntimeCode(os.path.join(exp_dir, 'backup'))
    except:
        logger.info(f'save code failed~')
    
    if opt.img_resolution is None:
        new_w, new_h = 1428, 798
    else:
        new_w, new_h = opt.img_resolution
    voxel_size = opt.models.args.representation_config.voxel_size
    
    # 加载 metadata
    if os.path.exists(os.path.join(opt.dataset_dir, f'metadata_norm128_v14.csv')):
        metadata = pd.read_csv(os.path.join(opt.dataset_dir, f'metadata_norm128_v14.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if opt.select_scene is not None:
        metadata = metadata[metadata['sha256'] == opt.select_scene]
        print(f"[INFO] selected scene is {opt.select_scene}, num_metadata: {len(metadata)}")
        
    less_coverage_csv = pd.read_csv(os.path.join(opt.dataset_dir, opt.feats_folder_name, opt.feat_model_name, f'less_coverage_scenes_all.csv'))
    # 根据 unconverage_ratio 进行筛选
    threshold = 0.87
    removed_sha256 = less_coverage_csv[less_coverage_csv['unconverage_ratio'] > threshold]['sha256'].tolist()

    excluded_sha256 = ["1204e08f17", "30966f4c6e", "cc5237fd77"] # 验证，这两个场景很差
    excluded_sha256 += ["edbf65c385", "684d4da1fe", "909a9ea5fc"] # OOM voxel太多
    excluded_sha256 += ["45d2e33be1", "9471b8d485", "116456116b", "40b56bf310","2748de13fb"]
    excluded_sha256 += ['2b5ef64cad', '0e75f3c4d9', '4c5c60fa76', '4d451d9c36', '5ea3e738c3', '5fb5d2dbf2', '7cd2ac43b4', '13c3e046d7', '036bce3393', '41b00feddb', '82ff39b7ef', '99fa5c25e1', '623c973c56', '724c40236c', '00777c41d4', '816e996553', '2489b7f4fe', '4117a4f02c', '020312de8d', '13285009a4', 'a4e227f506', 'a23f391ba9', 'a66af39cef', 'aad55d46ad', 'b37c033757', 'b24697b3a1', 'c4c04e6d6c', 'c5f701a8c7', 'c545851c4f', 'c601466b77', 'd415cc449b', 'd918af9c5f', 'e3b3b0d0c7', 'e3c39f4aea', 'e4007ff6b5', 'f248c2bcdc'] # 工厂
    excluded_sha256 += ['0f3474b837', '4f144c7489', '5d152fab1b', '7efa3f6b1b', '9f21bdec45','108ec0b806', '893fb90e89', '3583c4c70f', '20871b98f3', '635852d56e', '4422722c49', '5654092cc2', 'a6e1a3bdda', 'abb6639ceb', 'abf29d2474', 'de3c77cecd', 'f5401524e5', 'fe1733741f'] # storage room  'ada5304e41' 'c07c8a276d' 'd3b7d054c4
    excluded_sha256 += ['1b378f1114', '8e22c48c20', '19cfd590f4', '20ff72df6e', '40b56bf310', '45ba9ea301', '307e3262f1', '1204e08f17', '546292a9db', 'ab046f8faf', 'c4d4cb61f6', 'e3ecd49e2b', 'e909f8213d', 'eeeb9836b8', 'f2e6c43543'] # 不够32
    excluded_sha256 += ['5dd2b849be', 'be0ed6b33c', '546292a9db', '87f6d7d564', 'ab046f8faf']
    train_metadata = metadata[:856]
    train_metadata = train_metadata[
        train_metadata["sha256"].apply(
            lambda sha256: os.path.exists(
                os.path.join(opt.dataset_dir, opt.feats_folder_name, opt.feat_model_name, f'{sha256}.pt')
            )
        )
    ]
    train_metadata = train_metadata[~train_metadata["sha256"].isin(removed_sha256)]
    train_metadata = train_metadata[~train_metadata["sha256"].isin(excluded_sha256)]
    train_metadata = train_metadata[train_metadata['num_voxels_norm128_v14_fix32'] <= 80_000]
    # train_metadata = _filter_dataset(train_metadata, num_voxels_threshold=60_000) # 70_000  metadata[:856]  metadata[:230]
    train_metadata = _filter_dataset_new(train_metadata) # 70_000  metadata[:856]  metadata[:230]
    val_metadata = metadata[856:]
    val_metadata = val_metadata[~val_metadata["sha256"].isin(removed_sha256)]
    val_metadata = val_metadata[~val_metadata["sha256"].isin(excluded_sha256)]
    val_metadata = _filter_dataset(metadata[856:], num_voxels_threshold=60_000)   # 70_000  metadata[856:]  metadata[230:]
                        
    logger.info('\n\n'+'=='*60)
    logger.info(f"[INFO] metadata num: {len(metadata)}, filtered metadata num: {len(train_metadata) + len(val_metadata)}, train_metadata num: {len(train_metadata)}, val_metadata num: {len(val_metadata)}")
    logger.info('=='*60 + '\n\n')
    
    if opt.select_scene is None:
        train_dataset = SparseLatentVaeDataset(train_metadata, opt, stage='train')
        val_dataset = SparseLatentVaeDataset(val_metadata, opt, stage='val')
        train_loader = DataLoader(train_dataset, batch_size=opt.trainer.batch_size, 
                                  shuffle=True, num_workers=32, pin_memory=False, 
                                  persistent_workers=True,  collate_fn=collate_skip_none)
        val_loader = DataLoader(val_dataset, batch_size=opt.trainer.batch_size, 
                                shuffle=False, num_workers=32, pin_memory=False, 
                                persistent_workers=True )
        if opt.test:
            test_metadata = val_metadata
            test_dataset = SparseLatentVaeDataset(test_metadata, opt, stage='test')
            test_loader = DataLoader(test_dataset, batch_size=opt.trainer.batch_size, 
                                shuffle=False, num_workers=16, pin_memory=True, 
                                persistent_workers=True )
            
            test_train_novel_metadata = train_metadata
            test_train_novel_dataset = SparseLatentVaeDataset(test_train_novel_metadata, opt, stage='test_trainset_novel')
            test_train_novel_loader = DataLoader(test_train_novel_dataset, batch_size=opt.trainer.batch_size, 
                                shuffle=False, num_workers=16, pin_memory=True, 
                                persistent_workers=True )

    else:
        train_dataset = SparseLatentVaeDataset(metadata, opt, stage='train')
        train_loader = DataLoader(train_dataset, batch_size=opt.trainer.batch_size, 
                                  shuffle=True, num_workers=2, pin_memory=True, 
                                  persistent_workers=True,  collate_fn=collate_skip_none)
        val_loader = None
    
      
    # 创建两个 ModelCheckpoint 回调：
    # 1. 每隔 100 个 epoch 保存一次 checkpoint（保存所有，不仅仅最佳）
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir,
        filename="epoch{epoch:02d}",
        save_top_k=-1,            # 保存所有符合条件的 checkpoint
        every_n_epochs=opt.trainer.save_img_every_epoch,   
        verbose=True
    )

    # 2. 根据验证指标（这里选择 val_iou，指标越大越好）保存最佳模型
    best_checkpoint_callback = ModelCheckpoint(
        monitor="Validation/val_avg_psnr",
        dirpath=exp_dir,
        filename="best-{epoch:02d}-{val_avg_psnr:.2f}", 
        save_top_k=1,
        save_last=True,
        mode="max",            
        verbose=True
    )
    
    # 检查是否有 checkpoint 文件，如果有则从 checkpoint 恢复训练
    if not opt.test and os.path.exists(os.path.join(exp_dir, "last.ckpt")):
        # ckpt_list = glob.glob(os.path.join(exp_dir, "best-epoch=*.ckpt"))
        # if ckpt_list:
        #     resume_from_checkpoint = max(ckpt_list, key=os.path.getmtime) # 如果有多个，可以取最新修改时间的        
        ckpt_list = glob.glob(os.path.join(exp_dir, "last*.ckpt"))
        if ckpt_list:
            resume_from_checkpoint = max(ckpt_list, key=os.path.getmtime) # 如果有多个，可以取最新修改时间的        
        else:
            resume_from_checkpoint = os.path.join(exp_dir, "last.ckpt")
        logger.info(f"INFO] Resuming from checkpoint: {resume_from_checkpoint if resume_from_checkpoint else 'None'}")      
    elif opt.test:
        ckpt_list = glob.glob(os.path.join(exp_dir, "best-epoch=*.ckpt"))
        if ckpt_list:
            resume_from_checkpoint = max(ckpt_list, key=os.path.getmtime) # 如果有多个，可以取最新修改时间的
        # resume_from_checkpoint = os.path.join(exp_dir, "last.ckpt")
        logger.info(f"INFO] Resuming from checkpoint: {resume_from_checkpoint if resume_from_checkpoint else 'None'}")
    elif opt.ckpt and os.path.exists(opt.ckpt):
        resume_from_checkpoint = opt.ckpt    
        print('[INFO] resume_from_checkpoint' )
        # =========================修改ckpt Tmax===================================
        # ckpt = torch.load(resume_from_checkpoint)
        # ckpt['lr_schedulers'][0]['T_max'] = 20
        # torch.save(ckpt, resume_from_checkpoint)
        # print('[INFO] Edit LR_scheduler in CKPT!!!!')
        # =========================修改ckpt Tmax===================================
        logger.info(f"[INFO] Resuming from checkpoint: {resume_from_checkpoint if resume_from_checkpoint else 'None'}")  
    else:
        resume_from_checkpoint = None
        logger.info('INFO] No checkpoint is resumed. Training from start!')
        
    # 创建 Trainer，并开启多 GPU 训练（例如使用所有 GPU，分布式训练 backend 为 ddp）
    from pytorch_lightning.strategies import DDPStrategy
    callbacks = [best_checkpoint_callback] if opt.select_scene is None else [checkpoint_callback]
    trainer = Trainer(
        max_epochs=int(opt.trainer.max_epochs),
        # precision="bf16-mixed", # 混合精度计算
        accelerator='gpu',  # 使用 GPU 训练
        devices=-1,         # 使用所有可见 GPU
        strategy="ddp",
        # strategy=DDPStrategy(find_unused_parameters=True),
        detect_anomaly=True,
        callbacks=callbacks, #, grad_norm_callback], #ema_callback],
        logger=TBlogger,
        log_every_n_steps=opt.trainer.save_img_every_epoch,
        check_val_every_n_epoch=opt.trainer.save_img_every_epoch,
        num_sanity_val_steps=0,
        accumulate_grad_batches=opt.waitn,
        gradient_clip_val=opt.trainer.gradient_clip_val,
        # log_grad_norm='global',        # 关键设置
        limit_val_batches=1.0 if opt.select_scene is None else 0,  # 如果是验证集，则不限制验证批次
    )
    os.makedirs(os.path.join(trainer.logger.log_dir, 'training'), exist_ok=True)
    os.makedirs(os.path.join(trainer.logger.log_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(trainer.logger.log_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(trainer.logger.log_dir, 'latents_train'), exist_ok=True)
    os.makedirs(os.path.join(trainer.logger.log_dir, 'latents_val'), exist_ok=True)
    
    # print_memory_usage("*** 训练:  训练开始前")  # 训练前显存占用
    if resume_from_checkpoint is None:
        print('[INFO] loading from scrach!')
        model = AppearenceVAEModel(opt)
    else:
        print(f"[INFO] load model from ckpt: {resume_from_checkpoint}")
        model = AppearenceVAEModel.load_from_checkpoint(resume_from_checkpoint, opt=opt)
        
    if opt.test:
        trainer.test(model, dataloaders=[test_loader, test_train_novel_loader], ckpt_path=resume_from_checkpoint)  # 或者 trainer.predict()
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from_checkpoint)

    final_ckpt_path = os.path.join(exp_dir, "final_checkpoint.ckpt")
    trainer.save_checkpoint(final_ckpt_path)
    logger.info(f"Final checkpoint saved at: {final_ckpt_path}")

    logger.info(f"Best checkpoint saved at: {best_checkpoint_callback.best_model_path}")
