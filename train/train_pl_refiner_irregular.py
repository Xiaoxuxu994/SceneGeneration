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
os.environ['ATTN_BACKEND'] = 'flash-attn' # flash-attn, xformers
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as pretrainModels

# import trellis.models as models
# import trellis.modules.sparse as sp
# from trellis.representations.gaussian import Gaussian
# from trellis.utils import render_utils_scannet as render_utils
# from trellis.utils.demo_utils import gen_demo
# from trellis.utils.train_utils import batch_project_and_aggregate_chunkwise, _filter_dataset_new, _filter_dataset, get_model_summary, get_latest_ckpt, seed_everything, saveRuntimeCode
# from trellis.utils.loss_utils import masked_ssim_batch, l1_loss, dispersion_loss_voxel_scaled, feature_diversity_loss
# from trellis.utils import LRscheduler_utils as LRschedulers # WarmupCosineAnnealingLR, WarmupConstLR, cosine_scheduler_with_warmup,

import src.modules.sparse as sp
from src.representations.gaussian import Gaussian
from src.utils.demo_utils import gen_demo
from src.utils.train_utils import batch_project_and_aggregate_chunkwise, _filter_dataset_new, _filter_dataset, get_latest_ckpt, seed_everything, saveRuntimeCode, log_memory_usage

import torch.multiprocessing as mp
# mp.set_start_method('spawn')

seed_everything(520)

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

# %%
# ---------------------------
# 1. 定义 Dataset
# ---------------------------
from train.datasets import RefinerDataset as dataset
# ---------------------------
# 2. 封装模型到 LightningModule
# ---------------------------
from train.baseModel import BaseModel
class RefinerModel(BaseModel):
    def __init__(self, opt, if_norm_depth=True):
        super(RefinerModel).__init__(opt)
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

    # =====================================================
    # 统一 forward（推理逻辑）
    # =====================================================
    def forward(self, batch, batch_idx, mode='train'):
        
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
        depths_error = depths_error.unsqueeze(1)
        
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
            rand_img_lst = rand_img_lst.masked_fill(invalid_mask.unsqueeze(1), 0)
        
        # log_memory_usage("After preprocessing")

        with torch.no_grad():
            refiner_idx = 0
            # 首次计算render error作为网络输入
            all_train_videos, all_train_depths = self.rendering(attr_GS, rand_extrinsics_lst, rand_intrinsics_lst,
                                                                SCALE=SCALE, 
                                                                invalid_mask=invalid_mask, 
                                                                depth_max=depth_max,
                                                                batch_idx=batch_idx, 
                                                                sha256=sha256, 
                                                                render_mode='RGB+D', 
                                                                mode=mode)

            # log_memory_usage(f"After refiner_idx {refiner_idx} rendering")            

            render_error = self.compute_render_err(normalize(all_train_videos), normalize(rand_img_lst), sha256, 
                                                mode=mode, ifsave=True, refiner_iter=0)
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
                                                                                batch_idx, 
                                                                                refiner_idx=refiner_idx, 
                                                                                mode=mode) 
            
            
            
            del loss_rgb_0, avg_psnr_0, render_error, all_train_videos, all_train_depths
            self.clean_mem()   
            # log_memory_usage(f"After refiner_idx {refiner_idx} compute loss and log")            
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
                                                                invalid_mask=invalid_mask, 
                                                                depth_max=depth_max,
                                                                batch_idx=batch_idx, 
                                                                sha256=sha256, 
                                                                render_mode='RGB+D', 
                                                                mode=mode)
        
        # log_memory_usage(f"After refiner_idx {refiner_idx} second render error computation")
        
        with torch.no_grad():
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

        return loss_all
    
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
    
    def compute_log_loss(self, outputs, 
                        rand_img_lst, rand_depths_lst, 
                        all_train_videos, all_train_depths,
                        sha256, batch_idx, refiner_idx=0, mode=f'train'):
        """
        outputs: Gaussian Class
        """
        
        avg_loss_rec = self.compute_loss_rgb(all_train_videos, rand_img_lst)
        avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
        # ------------------------- 总 loss -------------------------
        loss_total = avg_loss_rec

        # ------------------------- 日志 -------------------------
        log_dict = {
            "lr": self.trainer.optimizers[0].param_groups[0]['lr'],
            f"{mode}/refiner{refiner_idx}_loss_total": loss_total.item(),
            f"{mode}/refiner{refiner_idx}_loss_rec_avg": avg_loss_rec.item(),
            f"{mode}/refiner{refiner_idx}_avg_psnr": avg_psnr.item()
        }

        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False if mode == 'train' else True)
        # logger.info(f"[{tag}] epoch: {self.trainer.current_epoch}, sha256: {sha256}, {mode}_log_dict = " + json.dumps(log_dict, indent=2))
        
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
                                    f'{sha256}_{mode}_{post_fix}_refined_GS.ply')) # epoch_{self.trainer.current_epoch}
                    
                else:
                    gausssian = outputs
                    gs_xyz = gausssian.get_xyz
                    gs_opa = gausssian.get_opacity
                    gs_sca = gausssian.get_scaling
                    gs_rot = gausssian.get_rotation
                    gs_fdc = gausssian.get_features.reshape(-1,3)
                    feat_GS = torch.cat([
                                            gs_xyz,
                                            gs_opa,
                                            gs_sca,
                                            gs_rot,
                                            gs_fdc,
                                        ], dim=-1).clone().detach().cpu().numpy()
                    dict_GS = {'feat_GS': feat_GS}
                    GS_savepath = os.path.join(self.opt.output_dir, mode, f'{sha256}_{mode}_{post_fix}_refined_GS.npz') 
                    np.savez(GS_savepath, **dict_GS)
                    outputs.save_ply(os.path.join(self.opt.output_dir, mode,
                                    f'{sha256}_{mode}_{post_fix}_refined_GS.ply')) # epoch_{self.trainer.current_epoch}
        
        return loss_total, avg_psnr
    
    def compute_render_err(self, all_train_videos, rand_img_lst, sha256, 
                           mode='train', ifsave=True, refiner_iter=0):
        
        rend_feat = self.extract_and_concat_features(all_train_videos)
        gt_feat = self.extract_and_concat_features(rand_img_lst)
        render_error = rend_feat - gt_feat
        
        if ifsave and self.trainer.current_epoch % self.opt.trainer.save_img_every_epoch ==0:

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
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.eval()
        with torch.no_grad():
            loss_all = self.forward(batch, batch_idx, mode='val')
            
            # 收集结果
            if dataloader_idx not in self.test_outputs:
                self.test_outputs[dataloader_idx] = []
            
            self.test_outputs[dataloader_idx].append({"scene": sha256, "avg_psnr": avg_psnr.detach()})
            print(f"dataloader_idx: {dataloader_idx}, scene: {sha256}, test avg_psnr: {avg_psnr.detach()}")
            
            if self.opt.demo:
                interp_folder = os.path.join(self.opt.output_dir, f'demo_dataloader{dataloader_idx}', f'{sha256}_interp_views')
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
        opt.output_dir = './outputs/Refiner_single/' + f'{opt.select_scene}_' + opt.config_path.split('/')[-1].split('.')[0] # outputs_Slat_vae_full
    elif opt.select_scene is None and opt.output_dir is None:
        opt.output_dir = './outputs/Refiner_full/' + opt.config_path.split('/')[-1].split('.')[0] + f'_DpMask{opt.use_depth_mask}' # outputs_Slat_vae_full
    else:
        opt.output_dir = opt.output_dir + f'{opt.select_scene}_' + opt.config_path.split('/')[-1].split('.')[0]+'_accumulateGrad' # outputs_Slat_vae_full
    os.makedirs(opt.output_dir, mode=0o777, exist_ok=True)
    
    exp_dir = opt.output_dir
    logger_dir = opt.output_dir.replace('outputs', 'tensorboadblogs')
    TBlogger = TensorBoardLogger(save_dir=logger_dir, name="logs", version=0)
    
    logger.remove(handler_id=None)
    logger.add(os.path.join(opt.output_dir, "loss.log"), level="INFO")
    logger.info(f"opt: {opt}")
    
    # save_config(opt, save_path=os.path.join(opt.output_dir, args.config_path.split('/')[-1]))
    logger.info('\n\nConfig:')
    logger.info('=' * 80)
    json_str = json.dumps(opt.__dict__, indent=4)  # 将对象转为格式化的 JSON 字符串
    with open(opt.output_dir+'/opt_config.json', 'w') as f: # 将 JSON 字符串保存到文件
        f.write(json_str)
    logger.info(json_str)
    
    try:
        saveRuntimeCode(os.path.join(opt.output_dir, 'backup'))
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
    val_excluded_sha256 = ['f3685d06a9'] # 这个场景之前的anchor点数量超过10k。
    val_metadata = val_metadata[~val_metadata["sha256"].isin(removed_sha256)]
    val_metadata = val_metadata[~val_metadata["sha256"].isin(excluded_sha256)]
    val_metadata = _filter_dataset(metadata[856:], num_voxels_threshold=60_000)   # 70_000  metadata[856:]  metadata[230:]
    val_metadata = val_metadata[~val_metadata["sha256"].isin(val_excluded_sha256)]
                  
    logger.info('\n\n'+'=='*60)
    logger.info(f"[INFO] metadata num: {len(metadata)}, filtered metadata num: {len(train_metadata) + len(val_metadata)}, train_metadata num: {len(train_metadata)}, val_metadata num: {len(val_metadata)}")
    logger.info('=='*60 + '\n\n')
    
    if opt.select_scene is None:
        train_dataset = datasets(train_metadata, opt, stage='train')
        val_dataset = datasets(val_metadata, opt, stage='val')
        train_loader = DataLoader(train_dataset, batch_size=opt.trainer.batch_size, 
                                  shuffle=True, num_workers=6, pin_memory=True, 
                                  persistent_workers=True,  collate_fn=collate_skip_none)
        val_loader = DataLoader(val_dataset, batch_size=opt.trainer.batch_size, 
                                shuffle=False, num_workers=4, pin_memory=True, 
                                persistent_workers=True )
        if opt.test:
            test_metadata = val_metadata
            test_dataset = datasets(test_metadata, opt, stage='test')
            test_loader = DataLoader(test_dataset, batch_size=opt.trainer.batch_size, 
                                shuffle=False, num_workers=16, pin_memory=True, 
                                persistent_workers=True )
            
            test_train_novel_metadata = train_metadata
            test_train_novel_dataset = datasets(test_train_novel_metadata, opt, stage='test_trainset_novel')
            test_train_novel_loader = DataLoader(test_train_novel_dataset, batch_size=opt.trainer.batch_size, 
                                shuffle=False, num_workers=16, pin_memory=True, 
                                persistent_workers=True )

    else:
        train_dataset = datasets(metadata, opt, stage='train')
        train_loader = DataLoader(train_dataset, batch_size=opt.trainer.batch_size, 
                                  shuffle=True, num_workers=4, pin_memory=True, 
                                  persistent_workers=True,  collate_fn=collate_skip_none)
        test_train_novel_dataset = datasets(metadata, opt, stage='test_trainset_novel')        
        val_loader = DataLoader(test_train_novel_dataset, batch_size=opt.trainer.batch_size, 
                                shuffle=False, num_workers=2, pin_memory=True, 
                                persistent_workers=True )
    
      
    # 创建两个 ModelCheckpoint 回调：
    # 1. 每隔 100 个 epoch 保存一次 checkpoint（保存所有，不仅仅最佳）
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.output_dir,
        filename="epoch{epoch:02d}",
        save_top_k=-1, 
        every_n_epochs=opt.trainer.save_img_every_epoch,   
        verbose=True
    )

    # 2. 根据验证指标（这里选择 val_iou，指标越大越好）保存最佳模型
    best_checkpoint_callback = ModelCheckpoint(
        monitor="val/refiner1_avg_psnr",
        dirpath=opt.output_dir,
        filename="best-{epoch:02d}-{val_refiner_psnr:.2f}", 
        save_top_k=1,
        save_last=True,
        mode="max",            
        verbose=True
    )
    
    # 检查是否有 checkpoint 文件，如果有则从 checkpoint 恢复训练
    if not opt.test and os.path.exists(os.path.join(exp_dir, "last.ckpt")):
        resume_from_checkpoint = os.path.join(exp_dir, "last.ckpt")

        # ckpt_list = glob.glob(os.path.join(exp_dir, "best-epoch=*.ckpt"))
        # if ckpt_list:
        #     resume_from_checkpoint = max(ckpt_list, key=os.path.getmtime) # 如果有多个，可以取最新修改时间的
        
        # ckpt = torch.load(resume_from_checkpoint, map_location='cpu')
        # if 'lr_schedulers' in ckpt and len(ckpt['lr_schedulers']) > 0:
        #     if ckpt['lr_schedulers'][0]['max_lr']>5e-6:
        #         ckpt['lr_schedulers'][0]['max_lr'] = 5e-6
        #         ckpt['lr_schedulers'][0]['base_lr'] = 1e-7
        #         # Save the modified checkpoint
        #         resume_from_checkpoint_new = resume_from_checkpoint.replace('.ckpt', '_modified.ckpt')
        #         torch.save(ckpt, resume_from_checkpoint_new)
        #         # logger.info(f"[INFO] Resumed from checkpoint: {resume_from_checkpoint_new}")
        #     else:
        #         logger.warning("Warning: Checkpoint file is already modified.")
        #         resume_from_checkpoint_new = resume_from_checkpoint
        # resume_from_checkpoint = resume_from_checkpoint_new
        logger.info(f"INFO] Resuming from checkpoint: {resume_from_checkpoint if resume_from_checkpoint else 'None'}") 

    elif not opt.test and os.path.exists(os.path.join(exp_dir, "final_checkpoint.ckpt")):
        resume_from_checkpoint = os.path.join(exp_dir, "final_checkpoint.ckpt")
        ckpt = torch.load(resume_from_checkpoint, map_location='cpu')
        if 'lr_schedulers' in ckpt and len(ckpt['lr_schedulers']) > 0:
            if ckpt['lr_schedulers'][0]['max_epochs']!=10000 and ckpt['lr_schedulers'][0]['T_max']!=10000:
                ckpt['lr_schedulers'][0]['max_epochs'] = 10000
                ckpt['lr_schedulers'][0]['T_max'] = 10000
                resume_from_checkpoint_new = resume_from_checkpoint.replace('.ckpt', '_modified.ckpt')
                torch.save(ckpt, resume_from_checkpoint_new)
            else:
                logger.warning("Warning: Checkpoint file is already modified.")
                resume_from_checkpoint_new = resume_from_checkpoint
        resume_from_checkpoint = resume_from_checkpoint_new
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
    
    # from pytorch_lightning.profilers.advanced import AdvancedProfiler
    # profiler = AdvancedProfiler(dirpath=f"{opt.output_dir}", filename="10epochs_advanced_profile_report.txt")
    callbacks = [best_checkpoint_callback] if opt.select_scene is None else [checkpoint_callback]

    from pytorch_lightning.strategies import DDPStrategy
    trainer = Trainer(
        max_epochs=int(opt.trainer.max_epochs),
        # precision="bf16-mixed", # 混合精度计算
        accelerator='gpu',  # 使用 GPU 训练
        devices=-1,         # 使用所有可见 GPU
        # strategy="ddp",
        strategy=DDPStrategy(find_unused_parameters=True),
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
        # profiler=profiler
    )
    os.makedirs(os.path.join(trainer.logger.log_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(trainer.logger.log_dir, 'val'), exist_ok=True)
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
        trainer.test(model, dataloaders=[test_loader], ckpt_path=resume_from_checkpoint)  # , test_train_novel_loader
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from_checkpoint)

    final_ckpt_path = os.path.join(exp_dir, "final_checkpoint.ckpt")
    trainer.save_checkpoint(final_ckpt_path)
    logger.info(f"Final checkpoint saved at: {final_ckpt_path}")

    logger.info(f"Best checkpoint saved at: {best_checkpoint_callback.best_model_path}")
    