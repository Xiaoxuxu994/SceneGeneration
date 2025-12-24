"""
2025.04.21 : render的时候是一张一张render的，因为多张图一起render显存占用很大
2025.04.22 : 修改了数据集，增加了dataset_filter
2025.04.29 : 修改config输入实验设置，和构建模型等

2025.06.30 : 增加了depth监督
2025.07.01 : 增加fixed select images and frames

2025.10.21 : 修改流程框架

2025.10.23 : 相较于v2版本, 删除了gs voxel feature作为输入, 同时，delta直接加在gs激活后的属性上，而不是gs激活前的属性上
"""

# %%
import os
# os.environ['ATTN_BACKEND'] = 'flash-attn' # flash-attn, xformers
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
import copy
import json
import math
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from loguru import logger
# import utils3d
# import open3d as o3d
# from PIL import Image
import lpips
import gc
import json
import glob
import matplotlib.pyplot as plt
import csv


import torch
torch.backends.cuda.preferred_linalg_library('magma') # magma
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as pretrainModels
from torchvision.utils import save_image

import src.modules.sparse as sp
from src.representations.gaussian import Gaussian
from src.utils.demo_utils import gen_demo
from src.utils.train_utils import batch_project_and_aggregate_chunkwise, log_memory_usage

import torch.multiprocessing as mp
# mp.set_start_method('spawn')

# seed_everything(520)

to_pil = transforms.ToPILImage()

torch.set_grad_enabled(True)

from torch.utils.data._utils.collate import default_collate
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # 可以在 training_step 做判断
    return default_collate(batch)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # 归一化到 [-1, 1]
])

# 定义 transform，将图片转换为 tensor，并归一化到 [0,1]
transform_novel = transforms.Compose([
    transforms.ToTensor(),  # 会把 [0,255] 转为 [0,1] 的 tensor
])

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

resnet18 = pretrainModels.resnet18(pretrained=True)
class FeatureExtractor(nn.Module):
    def __init__(self, resnet):
        super(FeatureExtractor, self).__init__()
        self.resnet = resnet
        self.features = nn.ModuleList([self.resnet.conv1, 
                                       self.resnet.bn1, 
                                       self.resnet.relu, 
                                       self.resnet.maxpool,
                                       self.resnet.layer1, 
                                       self.resnet.layer2])

    def forward(self, x): 
        # Feature extraction for different stages
        feature_maps = [] # x:  Size: (B, 3, H, W)
        
        # Stage 1: Before maxpool (1/2 resolution)
        x = self.features[0](x) 
        x = self.features[1](x) 
        x = self.features[2](x)
        feature_maps.append(x)  # This is at 1/2 resolution (B, 64, H/2, W/2)
        # Stage 2: After maxpool (1/4 resolution)
        x = self.features[3](x)
        feature_maps.append(x)  # This is at 1/4 resolution (B, 64, H/4, W/4)
        # Stage 3: After layer1 (1/8 resolution)
        x = self.features[4](x)
        feature_maps.append(x)  # This is at 1/8 resolution (B, 128, H/8, W/8)
        return feature_maps


def save_rgb_sequence(tensor, output_dir, prefix="rgb"):
    """
    保存 [N, 3, H, W] 的 RGB 图片序列
    """
    os.makedirs(output_dir, exist_ok=True)
    N = tensor.shape[0]
    for i in range(N):
        save_image(tensor[i], os.path.join(output_dir, f"{prefix}_{i:03d}.png"))
        
def save_depth_sequence(depth_tensor, output_dir, prefix="depth", cmap="plasma"):
    """
    保存 [N, 1, H, W] 深度图序列，自动加上 colormap
    """
    os.makedirs(output_dir, exist_ok=True)
    depth_tensor = depth_tensor.detach().cpu()
    N = depth_tensor.shape[0]

    for i in range(N):
        depth = depth_tensor[i, 0].numpy()

        # 归一化到 [0,1] 区间以便上色
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth[:] = 0  # 避免全0或NaN

        # 使用 colormap
        colored_depth = plt.get_cmap(cmap)(depth)[:, :, :3]  # RGBA -> RGB

        # 保存为 PNG
        plt.imsave(os.path.join(output_dir, f"{prefix}_{i:03d}.png"), colored_depth)



# %%
# ---------------------------
# 2. 封装模型到 LightningModule
# ---------------------------
from train.baseModel import BaseModel
class RefinerModel(BaseModel):
    def __init__(self, opt, if_norm_depth=True):
        super(RefinerModel, self).__init__(opt)
        self.if_norm_depth = if_norm_depth
        self.scale_factor = 1e3
        
        # 指定在哪些 epoch 做 inference
        self.infer_epochs = [59, 119, 199] + list(range(199, self.opt.trainer.max_epochs, 200))
        
        self.feature_extractor = FeatureExtractor(resnet18)
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        
        # 用于保存每个场景的输出
        self.num_refiners = 3
        self.refiner_gamma = 0.9

    def forward(self, batch, batch_idx, mode='train', test_dataloader_idx=0):
        """_summary_

        Args:
            batch (_type_): _description_
            mode (str, optional): forward mode. Defaults to 'train'.
            test_dataloader_idx (int, optional): only activated when using mode='test. Defaults to 0.

        Returns:
            _type_: _description_
        """
        
        sha256 = batch.get("sha256")[0]
        SCALE = batch.get("scale").item()
        T_norm = batch.get("T_norm")[0]
        attr_GS = batch.get("feat_GS", None)[0]
        feat_voxel = batch.get("feat_voxel")[0]
        rand_img_lst = batch.get("rand_img_lst")[0]
        rand_extrinsics_lst = batch.get("rand_extrinsics_lst")[0]
        rand_intrinsics_lst = batch.get("rand_intrinsics_lst")[0]
        rand_depths_lst = batch.get("rand_depths_lst")[0]
        
        intrinsics_err = batch.get("intrinsics_err")[0] # NOTE: refiner 独有
        depths_error = batch.get("depths_error")[0] # NOTE: refiner 独有
        depths_error = depths_error
        
        render_error_GS = batch.get("rend_err_GS")
        
        # log_memory_usage("After input data extraction")
        
        # ------------------------- 预处理 -------------------------
        new_w, new_h = self.opt.img_resolution
        if rand_depths_lst.shape[-2:] != (new_h, new_w): 
            rand_depths_lst = F.interpolate(rand_depths_lst, size=(new_h, new_w),
                                            mode='bilinear', align_corners=False).squeeze(1).to("cpu")
        depth_max = rand_depths_lst.max().item()
        
        # print(f"[RANK {self.global_rank}], scene: {sha256}, attr_GS: {attr_GS.shape}")
        
        if self.opt.trainer.render_option == 'gsplat':
            rand_intrinsics_lst[:, 0, :] *= new_w
            rand_intrinsics_lst[:, 1, :] *= new_h
        
        # 归一化depth
        if self.if_norm_depth:
            rand_depths_lst = rand_depths_lst / depth_max
        
        # maskr住无效深度的gt 图像区域
        if self.opt.use_depth_mask:
            invalid_mask = rand_depths_lst <= 1e-6
            # rand_img_lst = rand_img_lst.masked_fill(invalid_mask.unsqueeze(1), 0)
        
        # log_memory_usage("After preprocessing")

        if isinstance(render_error_GS, list) and isinstance(render_error_GS[0], str):
            filepath = render_error_GS[0]
            # print("filepath: ", filepath)

            with torch.no_grad():
                refiner_idx = 0
                # 首次计算render error作为网络输入
                all_train_videos, all_train_depths = self.rendering(attr_GS, rand_extrinsics_lst, rand_intrinsics_lst,
                                                                    SCALE=SCALE, 
                                                                    # invalid_mask=invalid_mask, 
                                                                    depth_max=depth_max,
                                                                    # batch_idx=batch_idx, 
                                                                    sha256=sha256, 
                                                                    render_mode='RGB+D', 
                                                                    mode=mode)
                # log_memory_usage(f"After refiner_idx {refiner_idx} rendering")
                render_error = self.compute_render_err(normalize(all_train_videos), normalize(rand_img_lst), sha256, 
                                                    mode=mode, ifsave=True, refiner_iter=refiner_idx)
                render_error_GS, _ = batch_project_and_aggregate_chunkwise(points=attr_GS[..., :3], 
                                                                            features=render_error, 
                                                                            depths=depths_error, 
                                                                            intrinsics=intrinsics_err, 
                                                                            w2cs=rand_extrinsics_lst,
                                                                            T_norm=T_norm,
                                                                            depth_tol=4/128,
                                                                            fill_empty=False, 
                                                                            chunk_size_V=3, 
                                                                            chunk_size_N=20000,
                                                                            opstype='bf16')
                # log_memory_usage(f"After refiner_idx {refiner_idx} render error computation")
                
                loss_rgb_0, avg_psnr_0 = self.compute_log_loss_rgb_only_for_refiner(attr_GS, 
                                                                                    rand_img_lst, 
                                                                                    rand_depths_lst, 
                                                                                    all_train_videos, 
                                                                                    all_train_depths,
                                                                                    sha256, 
                                                                                    # batch_idx, 
                                                                                    refiner_idx=refiner_idx, 
                                                                                    mode=mode) 
                # res = dict(np.load(filepath))
                # res.update({"rend_err_GS": render_error_GS.detach().cpu().numpy()})
                # np.savez_compressed(filepath, **res)
                # print(f'Updated render error GS to {filepath}')

                
            del loss_rgb_0, avg_psnr_0, render_error, all_train_videos, all_train_depths
            self.clean_mem()   
            # log_memory_usage(f"After refiner_idx {refiner_idx} compute loss and log")    
        else:
            render_error_GS = render_error_GS[0]        
        # ------------------------- forward -------------------------
        
        loss_all = 0
        # for idx in range(self.num_refiners):
        
        # ------------------------- refiner 1 -------------------------  
        refiner_idx = 1  
        feat_GS = feat_voxel.unsqueeze(1).repeat(1,4,1).reshape(attr_GS.shape[0], -1)

        gs_cls_1 = self.model(attr_GS, feat_GS, render_error_GS,
                              batch_size_cdist=4096, batch_size_attn=4096)
        
        # log_memory_usage("After model forward pass")
        
        all_train_videos_1, all_train_depths_1 = self.rendering(gs_cls_1, 
                                                                rand_extrinsics_lst, 
                                                                rand_intrinsics_lst, 
                                                                SCALE=SCALE, 
                                                                # invalid_mask=invalid_mask, 
                                                                depth_max=depth_max,
                                                                sha256=sha256, 
                                                                render_mode='RGB+D', 
                                                                mode=mode)
        
        # log_memory_usage(f"After refiner_idx {refiner_idx} second render error computation")
        
        # with torch.no_grad():
        # 更新 render_err
        render_error_1 = self.compute_render_err(normalize(all_train_videos_1), 
                                                normalize(rand_img_lst), 
                                                sha256, 
                                                mode=mode, 
                                                ifsave=True, 
                                                refiner_iter=refiner_idx)
        render_error_GS_1, _ = batch_project_and_aggregate_chunkwise(points=gs_cls_1[..., :3], 
                                                                    features=render_error_1, 
                                                                    depths=depths_error, 
                                                                    intrinsics=intrinsics_err, 
                                                                    w2cs=rand_extrinsics_lst,
                                                                    T_norm=T_norm,
                                                                    depth_tol=4/128,
                                                                    fill_empty=False, 
                                                                    chunk_size_V=3, 
                                                                    chunk_size_N=20000,
                                                                    opstype='bf16')
        
        # log_memory_usage(f"After refiner_idx {refiner_idx} second render error computation")

        
        if mode == 'test':
            _, log_dict = self.evaluate_log(gs_cls_1, 
                                            rand_img_lst, 
                                            rand_depths_lst, 
                                            all_train_videos_1, 
                                            all_train_depths_1,
                                            sha256, 
                                            refiner_idx=refiner_idx, 
                                            mode=mode,
                                            test_dataloader_idx=test_dataloader_idx)
            self.clean_mem() 
            return gs_cls_1, log_dict
        else:
            # 计算并log每个refiner的loss
            loss_rgb_1, avg_psnr_1 = self.compute_log_loss_rgb_only_for_refiner(gs_cls_1, 
                                                                                rand_img_lst, 
                                                                                rand_depths_lst, 
                                                                                all_train_videos_1, 
                                                                                all_train_depths_1,
                                                                                sha256, 
                                                                                batch_idx, 
                                                                                refiner_idx=refiner_idx, 
                                                                                mode=mode)
            # loss_rgb = self.compute_loss_rgb(all_train_videos, rand_img_lst)
            # avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
            weight = self.refiner_gamma ** (self.num_refiners - 1 - (refiner_idx-1))
            loss_all += weight * loss_rgb_1
        
        # log_memory_usage(f"After refiner_idx {refiner_idx} second refiner loss computation")
        
        del render_error_1, render_error_GS_1, rand_img_lst, rand_depths_lst, rand_extrinsics_lst, rand_intrinsics_lst, invalid_mask, depths_error, intrinsics_err
        self.clean_mem() 
        
        # log_memory_usage("End")    
        
        # # ------------------------- refiner 2 -------------------------  
        # refiner_idx = 2
        
        # gs_cls_2, feat_GS_2, _, _ = self.model(gs_cls_1, feat_GS_1, render_error_GS_1)
        
        # # feat_GS_2 = feat_GS_1 + delta_z_2
        
        # all_train_videos_2, all_train_depths_2 = self.rendering(gs_cls_2, rand_extrinsics_lst, rand_intrinsics_lst, 
        #                                                         SCALE=SCALE, 
        #                                                         invalid_mask=invalid_mask, 
        #                                                         depth_max=depth_max,
        #                                                         batch_idx=batch_idx, 
        #                                                         sha256=sha256, 
        #                                                         render_mode='RGB+D', 
        #                                                         mode=mode)

        # # 更新 render_err
        # render_error_2 = self.compute_render_err(normalize(all_train_videos_2), 
        #                                          normalize(rand_img_lst), 
        #                                          sha256, 
        #                                          mode=mode, 
        #                                          ifsave=True, 
        #                                          refiner_iter=refiner_idx)
        # render_error_GS_2, vis_mask_2 = batch_project_and_aggregate_chunkwise(points=gs_cls_2.get_xyz, 
        #                                                           features=render_error_2, 
        #                                                           depths=depths_error, 
        #                                                           intrinsics=intrinsics_err, 
        #                                                           w2cs=rand_extrinsics_lst,
        #                                                           T_norm=T_norm,
        #                                                           depth_tol=4/128)
        
        # # 计算并log每个refiner的loss
        # loss_rgb_2, avg_psnr_2 = self.compute_log_loss_rgb_only_for_refiner(gs_cls_2, 
        #                                                                     rand_img_lst, 
        #                                                                     rand_depths_lst, 
        #                                                                     all_train_videos_2, 
        #                                                                     all_train_depths_2,
        #                                                                     sha256, 
        #                                                                     batch_idx, 
        #                                                                     refiner_idx=refiner_idx, 
        #                                                                     mode=mode)
        # # loss_rgb = self.compute_loss_rgb(all_train_videos, rand_img_lst)
        # # avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
        # weight = self.refiner_gamma ** (self.num_refiners - 1 - (refiner_idx-1))
        # loss_all += weight * loss_rgb_2
        
        # # ------------------------- refiner 3 -------------------------  
        # refiner_idx = 3
        
        # gs_cls_3, feat_GS_3, _, _ = self.model(gs_cls_2, feat_GS_2, render_error_GS_2)
        
        # # feat_GS_3 = feat_GS_2 + delta_z_3
        
        # all_train_videos_3, all_train_depths_3 = self.rendering(gs_cls_3, rand_extrinsics_lst, rand_intrinsics_lst, 
        #                                                         SCALE=SCALE, 
        #                                                         invalid_mask=invalid_mask, 
        #                                                         depth_max=depth_max,
        #                                                         batch_idx=batch_idx, 
        #                                                         sha256=sha256, 
        #                                                         render_mode='RGB+D', 
        #                                                         mode=mode)

        # # 更新 render_err
        # render_error_3 = self.compute_render_err(normalize(all_train_videos_3), 
        #                                          normalize(rand_img_lst), 
        #                                          sha256, 
        #                                          mode=mode, 
        #                                          ifsave=True, 
        #                                          refiner_iter=refiner_idx)
        # render_error_GS_3, vis_mask_3 = batch_project_and_aggregate_chunkwise(points=gs_cls_3.get_xyz, 
        #                                                           features=render_error_3, 
        #                                                           depths=depths_error, 
        #                                                           intrinsics=intrinsics_err, 
        #                                                           w2cs=rand_extrinsics_lst,
        #                                                           T_norm=T_norm,
        #                                                           depth_tol=4/128)
        
        # # 计算并log每个refiner的loss
        # loss_rgb_3, avg_psnr_3 = self.compute_log_loss_rgb_only_for_refiner(gs_cls_3, 
        #                                                                     rand_img_lst, 
        #                                                                     rand_depths_lst, 
        #                                                                     all_train_videos_3, 
        #                                                                     all_train_depths_3,
        #                                                                     sha256, 
        #                                                                     batch_idx, 
        #                                                                     refiner_idx=refiner_idx, 
        #                                                                     mode=mode)
        # # loss_rgb = self.compute_loss_rgb(all_train_videos, rand_img_lst)
        # # avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
        # weight = self.refiner_gamma ** (self.num_refiners - 1 - (refiner_idx-1))
        # loss_all += weight * loss_rgb_3

        return loss_all, log_dict
    
    def extract_and_concat_features(self, input_image):
        # Forward pass through the feature extractor
        feature_maps = self.feature_extractor(input_image)

        # Extract the feature maps at 1/2, 1/4, and 1/8 resolutions
        feature_map_1_2 = feature_maps[0]  # Stage 2
        feature_map_1_4 = feature_maps[1]  # Stage 3
        feature_map_1_8 = feature_maps[2]  # Stage 4

        # Resize to 1/4 resolution (same resolution as feature_map_1_4)
        feature_map_1_2 = F.interpolate(feature_map_1_2, 
                                        size=(feature_map_1_4.shape[2], feature_map_1_4.shape[3]), 
                                        mode='bilinear', 
                                        align_corners=False) # [V, 64, H/4, W/4]
        feature_map_1_8 = F.interpolate(feature_map_1_8, 
                                        size=(feature_map_1_4.shape[2], feature_map_1_4.shape[3]), 
                                        mode='bilinear', 
                                        align_corners=False) # [V, 64, H/4, W/4]

        # Concatenate the features along the channel dimension
        concatenated_features = torch.cat([feature_map_1_2, feature_map_1_4, feature_map_1_8], dim=1) # [V, 64*3, H/4, W/4] ]
        
        return concatenated_features
    
    def compute_log_loss_rgb_only_for_refiner(self, outputs, 
                                              rand_img_lst, rand_depths_lst, 
                                              all_train_videos, all_train_depths,
                                              sha256, refiner_idx=0, mode=f'train'):
        """
        outputs: Gaussian Class
        """
        # ------------------------- 可视化 -------------------------
        with torch.no_grad():
            train_save_trigger = (mode == 'train') and (self.trainer.current_epoch % self.opt.trainer.save_img_every_epoch == 0) 
            if train_save_trigger or (mode == 'val') or ('test' in mode):
                
                post_fix = f'epoch{self.trainer.current_epoch}_refiner{refiner_idx}' if self.trainer.current_epoch == 0 else f'refiner{refiner_idx}'

                save_image(all_train_videos, os.path.join(self.opt.output_dir, mode,
                            f"{sha256}_{mode}_{post_fix}_rgb_image.png")) 
                
                save_image(rand_img_lst, os.path.join(self.opt.output_dir, mode,
                        f"{sha256}_{mode}_{post_fix}_GT_rgb_image.png"))
                
                save_image(all_train_depths.unsqueeze(1) if len(all_train_depths.shape)==3 else all_train_depths, 
                           os.path.join(self.opt.output_dir, mode,
                            f"{sha256}_{mode}_{post_fix}_depth.png"))
                
                save_image(rand_depths_lst.unsqueeze(1) if len(rand_depths_lst.shape)==3 else rand_depths_lst, 
                           os.path.join(self.opt.output_dir, mode,
                        f"{sha256}_{mode}_{post_fix}_GT_depth.png"))
                
                if isinstance(outputs, torch.Tensor):
                    dict_GS = {'feat_GS': outputs.clone().detach().cpu().numpy()}
                    GS_savepath = os.path.join(self.opt.output_dir, mode, 
                                               f'{sha256}_{mode}_{post_fix}_refined_GS.npz') 
                    np.savez(GS_savepath, **dict_GS)
                    
                    gaussians = Gaussian(
                        sh_degree=self.sh_degree,
                        mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
                        scaling_bias = self.rep_config['scaling_bias'],
                        opacity_bias = self.rep_config['opacity_bias'],
                        scaling_activation = self.rep_config['scaling_activation']
                    )
                    
                    gaussians.from_xyz(outputs[:,:3])
                    gaussians.from_opacity(outputs[:, 3:4])
                    gaussians.from_scaling(outputs[:, 4:7])
                    gaussians.from_rotation(outputs[:, 7:11])
                    gaussians.from_features(outputs[:, 11:])
                    gaussians.save_ply(os.path.join(self.opt.output_dir, mode,
                                    f'{sha256}_{mode}_{post_fix}_refiner{refiner_idx}_GS.ply')) # epoch_{self.trainer.current_epoch}
                    
                else:
                    gausssian = outputs
                    feat_GS = torch.cat([
                                            gausssian.get_xyz,
                                            gausssian.get_opacity,
                                            gausssian.get_scaling,
                                            gausssian.get_rotation,
                                            gausssian.get_features.reshape(-1,self.rgb_dim),
                                        ], dim=-1).clone().detach().cpu().numpy()
                    GS_savepath = os.path.join(self.opt.output_dir, mode, f'{sha256}_{mode}_{post_fix}_refiner{refiner_idx}_GS.npz') 
                    np.savez(GS_savepath, feat_GS=feat_GS)
                    outputs.save_ply(os.path.join(self.opt.output_dir, mode,
                                     f'{sha256}_{mode}_{post_fix}_refiner{refiner_idx}_GS.ply')) # epoch_{self.trainer.current_epoch}
        
        # --------------------------------------------------------------------------------
        avg_loss_rec = self.compute_loss_rgb(all_train_videos, rand_img_lst)
        avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
        loss_total = avg_loss_rec

        # ------------------------- 日志 -------------------------
        log_dict = {
            # "lr": self.trainer.optimizers[0].param_groups[0]['lr'],
            f"{mode}/refiner{refiner_idx}_loss_total": loss_total.item(),
            f"{mode}/refiner{refiner_idx}_loss_rec_avg": avg_loss_rec.item(),
            f"{mode}/refiner{refiner_idx}_avg_psnr": avg_psnr.item()
        }

        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False if mode == 'train' else True)
        # logger.info(f"[{tag}] epoch: {self.trainer.current_epoch}, sha256: {sha256}, {mode}_log_dict = " + json.dumps(log_dict, indent=2))
        
        return loss_total, avg_psnr
    
    
    def compute_render_err(self, all_train_videos, rand_img_lst, sha256, 
                           mode='train', ifsave=True, refiner_iter=0):
        
        rend_feat = self.extract_and_concat_features(all_train_videos)
        gt_feat = self.extract_and_concat_features(rand_img_lst)
        render_error = rend_feat - gt_feat
        
        if ifsave and self.trainer.current_epoch % self.opt.trainer.save_img_every_epoch ==0:
            post_fix = f'epoch{self.trainer.current_epoch}_refiner{refiner_iter}' if self.trainer.current_epoch == 0 else f'refiner{refiner_iter}'
            rend_feat_avg = rend_feat.mean(dim=1).unsqueeze(1)
            save_image(rend_feat_avg, os.path.join(self.opt.output_dir, mode,
                        f"{sha256}_{mode}_epoch{self.trainer.current_epoch}_refiner{refiner_iter}_rend_feat_image.png")) # epoch_{self.trainer.current_epoch}
            
            gt_feat_avg = gt_feat.mean(dim=1).unsqueeze(1)
            save_image(gt_feat_avg, os.path.join(self.opt.output_dir, mode,
                    f"{sha256}_{mode}_epoch{self.trainer.current_epoch}_refiner{refiner_iter}_gt_feat_image.png")) #     

            render_error_avg = render_error.mean(dim=1).unsqueeze(1)
            save_image(render_error_avg, os.path.join(self.opt.output_dir, mode,
                        f"{sha256}_{mode}_epoch{self.trainer.current_epoch}_refiner{refiner_iter}_render_error.png")) # epoch_{self.trainer.current_epoch}

        return render_error

    def evaluate_log(self, outputs, 
                    rand_img_lst, rand_depths_lst, 
                    all_train_videos, all_train_depths,
                    sha256, batch_idx=None, refiner_idx=0, mode=f'train',
                    test_dataloader_idx=0):
        """
        outputs: Gaussian Class
        """
        # ------------------------- 可视化 -------------------------
        post_fix = f'epoch{self.trainer.current_epoch}_refiner{refiner_idx}' if self.trainer.current_epoch == 0 else f'refiner{refiner_idx}'

        # CROP to 448 x 448
        all_train_videos = all_train_videos[..., 112:560]
        all_train_depths = all_train_depths[..., 112:560]
        rand_img_lst = rand_img_lst[..., 112:560]
        rand_depths_lst = rand_depths_lst[..., 112:560]  
        rand_depths_lst[rand_depths_lst <= 1e-6] = 1e-6
        all_train_depths[all_train_depths <= 1e-6] = 1e-6
        if all_train_depths.ndim == 3: all_train_depths.unsqueeze_(1)
        if rand_depths_lst.ndim == 3: rand_depths_lst.unsqueeze_(1)

        output_dir = os.path.join(self.opt.output_dir, mode, str(test_dataloader_idx), sha256)
        os.makedirs(output_dir, exist_ok=True)

        save_image(all_train_videos, os.path.join(output_dir, f"{sha256}_{mode}_{post_fix}_rgb_image.png")) 
        
        save_image(rand_img_lst, os.path.join(output_dir, f"{sha256}_{mode}_{post_fix}_GT_rgb_image.png"))
        
        save_image(all_train_depths, 
                    os.path.join(output_dir, f"{sha256}_{mode}_{post_fix}_depth.png"))
        
        save_image(rand_depths_lst, 
                    os.path.join(output_dir, f"{sha256}_{mode}_{post_fix}_GT_depth.png"))
        
        save_rgb_sequence(rand_img_lst, output_dir, prefix="rgb_gt")
        save_rgb_sequence(all_train_videos, output_dir, prefix="rgb_rend")
        save_depth_sequence(rand_depths_lst, output_dir, prefix="depth_gt", cmap="plasma")
        save_depth_sequence(all_train_depths, output_dir, prefix="depth_rend", cmap="plasma")
        
        
        if isinstance(outputs, torch.Tensor):
            GS_savepath = os.path.join(output_dir, 
                                        f'{sha256}_{mode}_{post_fix}_refiner{refiner_idx}_GS.npz') 
            np.savez(GS_savepath, feat_GS=outputs.clone().detach().cpu().numpy())
            numGS = outputs.shape[0]
            gaussians = Gaussian(
                sh_degree=self.sh_degree,
                mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
                scaling_bias = self.rep_config['scaling_bias'],
                opacity_bias = self.rep_config['opacity_bias'],
                scaling_activation = self.rep_config['scaling_activation']
            )
            
            gaussians.from_xyz(outputs[:,:3])
            gaussians.from_opacity(outputs[:, 3:4])
            gaussians.from_scaling(outputs[:, 4:7])
            gaussians.from_rotation(outputs[:, 7:11])
            gaussians.from_features(outputs[:, 11:])
            gaussians.save_ply(os.path.join(output_dir,
                            f'{sha256}_{mode}_{post_fix}_refiner{refiner_idx}_GS.ply')) # epoch_{self.trainer.current_epoch}
            
        else:
            gausssian = outputs
            feat_GS = torch.cat([
                                    gausssian.get_xyz,
                                    gausssian.get_opacity,
                                    gausssian.get_scaling,
                                    gausssian.get_rotation,
                                    gausssian.get_features.reshape(-1,self.rgb_dim),
                                ], dim=-1).clone().detach().cpu().numpy()
            GS_savepath = os.path.join(output_dir, f'{sha256}_{mode}_{post_fix}_refiner{refiner_idx}_GS.npz') 
            np.savez(GS_savepath, feat_GS=feat_GS)
            outputs.save_ply(os.path.join(output_dir, f'{sha256}_{mode}_{post_fix}_refined_GS.ply')) # epoch_{self.trainer.current_epoch}
            numGS = gausssian.get_xyz.shape[0]
        # ------------------------- 计算指标 ------------------------- 
        # maskr住无效深度的gt图像和depth区域
        if self.opt.use_depth_mask:
            invalid_mask = rand_depths_lst <= 1e-6
            rand_img_lst = rand_img_lst.masked_fill(invalid_mask.unsqueeze(1), 0)
            all_train_videos = all_train_videos.masked_fill(invalid_mask.unsqueeze(1), 0)
            rand_depths_lst = rand_depths_lst.masked_fill(invalid_mask, 1e-6)  
            all_train_depths = all_train_depths.masked_fill(invalid_mask, 1e-6)  
        else:
            invalid_mask = None  
        
        if invalid_mask is not None:
            valid_mask = ~invalid_mask
        else:
            valid_mask = (all_train_depths>1e-6) & (rand_depths_lst>1e-6)  
        
        img_rec_l1, img_rec_ssim, img_rec_lpips, avg_psnr = self.evaluate_rgb(all_train_videos.bfloat16(), rand_img_lst.bfloat16())
        absRel = self.abs_relative_difference(all_train_depths, rand_depths_lst, valid_mask=valid_mask)
        delta1 = self.delta1_acc(all_train_depths, rand_depths_lst, valid_mask=valid_mask)
        
        # ------------------------- 日志 -------------------------
        log_dict = {
            # "lr": self.trainer.optimizers[0].param_groups[0]['lr'],
            f"{mode}/Refiner{refiner_idx}/absRel": absRel.item(),
            f"{mode}/Refiner{refiner_idx}/delta1": delta1.item(),
            f"{mode}/Refiner{refiner_idx}/img_rec_l1": img_rec_l1.item(),
            f"{mode}/Refiner{refiner_idx}/img_rec_ssim": img_rec_ssim.item(),
            f"{mode}/Refiner{refiner_idx}/img_rec_lpips": img_rec_lpips.item(),
            f"{mode}/Refiner{refiner_idx}/avg_psnr": avg_psnr.item(),
            f"{mode}/Refiner{refiner_idx}/num_view": len(rand_img_lst),
            f"{mode}/Refiner{refiner_idx}/num_GS": numGS
        }

        return None, log_dict
    
    def training_step(self, batch, batch_idx):
        # torch.autograd.set_detect_anomaly(True)
        self.model.train()
        
        loss_all = self.forward(batch, batch_idx, mode='train')

        return loss_all
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        
        with torch.no_grad():
            loss_all = self.forward(batch, batch_idx, mode='val')
                
    def on_test_start(self):
        # 每次 test 前清空存储
        self.test_outputs = {}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0, mode='test'):
        self.model.eval()
        with torch.no_grad():
            sha256 = batch.get("sha256")
            rend_GS, log_dict = self.forward(batch, batch_idx, mode=mode, test_dataloader_idx=dataloader_idx)
            log_dict['sha256'] = sha256
            
            # 收集结果
            if dataloader_idx not in self.test_outputs:
                self.test_outputs[dataloader_idx] = []
            self.test_outputs[dataloader_idx].append(log_dict)
            print(f"dataloader_idx: {dataloader_idx}", log_dict)

            if self.opt.demo:
                interp_folder = os.path.join(self.opt.output_dir, f'demo_dataloader{dataloader_idx}', f'{sha256}_interp_views')
                os.makedirs(interp_folder, exist_ok=True)
                center = rend_GS[..., :3].mean(dim=0, keepdim=True)
                gen_demo(rend_GS, center, self.opt.trainer.render_option, 
                        device=self.device, new_w=480, new_h=268, 
                        interp_folder=interp_folder)
    
    def on_test_epoch_end(self):
        print(self.test_outputs)
        for dataloader_idx, dataloader_res in self.test_outputs.items():
            num_scene = len(dataloader_res)
            print("num_scene: ", num_scene)
            fieldnames = list(dataloader_res[0].keys())
            fieldnames = ['sha256'] + [key for key in fieldnames if key != 'sha256']
            # 保存为CSV文件
            csv_file_path = os.path.join(self.opt.output_dir, 'test', f'test_results_dataloader{dataloader_idx}.csv')
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row in dataloader_res:
                    row = {key: row[key] for key in fieldnames} 
                    writer.writerow(row)
            print(f"save evaluation metrics to {os.path.join(self.opt.output_dir, 'test', f'test_results_dataloader{dataloader_idx}.csv')}")
        
            df = pd.read_csv(csv_file_path)
            numeric_cols = [col for col in df.columns if col not in ['scene', 'method', 'sha256']]
            mean_row = df[numeric_cols].mean().to_dict() # 按列计算平均值
            mean_row['scene'] = 'average'
            mean_row['method'] = ''
            mean_row['sha256'] = ''
            df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
            df.to_csv(csv_file_path, index=False)
            print("✅ 已在 CSV 最后一行添加平均值")

        self.test_outputs.clear() # 清空，避免多个 epoch 累加
