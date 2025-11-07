# %%
import os
os.environ['ATTN_BACKEND'] = 'flash_attn_npu' # flash-attn, xformers
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
# from torchvision.utils import save_image
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from easydict import EasyDict as edict
from loguru import logger
import json
from pathlib import Path
import glob
import csv

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, InitProcessGroupKwargs

import torch
import torch.nn as nn
import torch_npu
# from torch_npu.contrib import transfer_to_npu
torch._C._jit_set_profiling_mode(False)

from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
# from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms
import torchvision.models as pretrainModels

import src.modules.sparse as sp
from src.representations.gaussian import Gaussian
from src.utils.demo_utils import gen_demo
from src.utils.train_utils import deepspeed_load, batch_project_and_aggregate_chunkwise, _filter_dataset_new, _filter_dataset, get_latest_ckpt, seed_everything, saveRuntimeCode, log_memory_usage

# import torch.multiprocessing as mp
# mp.set_start_method('spawn')

seed_everything(520)

to_pil = transforms.ToPILImage()

torch.set_grad_enabled(True)

# %%
# ---------------------------
# 2. 封装模型到 LightningModule
# ---------------------------
from train.baseModel import BaseModel

from train_pl_npu_refiner_irregularv2 import RefinerModel

class RefinerModel(BaseModel):
    def __init__(self, opt, if_norm_depth=True):
        super(RefinerModel, self).__init__(opt)
        self.if_norm_depth = if_norm_depth
        self.scale_factor = 1e3
        
        # 指定在哪些 epoch 做 inference
        self.infer_epochs = [59, 119, 199] + list(range(199, self.opt.trainer.max_epochs, 200))
        
        self.feature_extractor = FeatureExtractor(resnet18)
        self.feature_extractor.eval().requires_grad_(False)
        
        # 用于保存每个场景的输出
        self.num_refiners = 3
        self.refiner_gamma = 0.9


    def forward(self, batch, mode='train'):
        
        sha256 = batch.get("sha256")
        SCALE = batch.get("scale").item()
        T_norm = batch.get("T_norm")
        attr_GS = batch.get("feat_GS", None)
        feat_voxel = batch.get("feat_voxel")
        rand_img_lst = batch.get("rand_img_lst")
        rand_extrinsics_lst = batch.get("rand_extrinsics_lst")
        rand_intrinsics_lst = batch.get("rand_intrinsics_lst")
        # print(rand_extrinsics_lst.shape, rand_intrinsics_lst.shape)
        rand_depths_lst = batch.get("rand_depths_lst")
        
        intrinsics_err = batch.get("intrinsics_err") # NOTE: refiner 独有
        depths_error = batch.get("depths_error") # NOTE: refiner 独有
        depths_error = depths_error.unsqueeze(1)
        
        # log_memory_usage("After input data extraction")
        
        # ------------------------- 预处理 -------------------------
        new_w, new_h = self.opt.img_resolution
        if rand_depths_lst.shape[-2:] != (new_h, new_w): 
            rand_depths_lst = F.interpolate(rand_depths_lst, size=(new_h, new_w),
                                            mode='bilinear', align_corners=False).squeeze(1).to("cpu")
        depth_max = rand_depths_lst.max().item()
        
        # print(f"[RANK {self.global_rank}], scene: {sha256}, attr_GS: {attr_GS.shape}")
        
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
                                                                sha256=sha256, 
                                                                render_mode='RGB+D', 
                                                                mode=mode)
            # breakpoint()
            # all_train_videos = all_train_videos.to(next(self.parameters()).dtype)

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
            
            loss_rgb_0, log_dict_0 = self.compute_log_loss_rgb_only_for_refiner(attr_GS, 
                                                                                rand_img_lst, 
                                                                                rand_depths_lst, 
                                                                                all_train_videos, 
                                                                                all_train_depths,
                                                                                sha256, 
                                                                                refiner_idx=refiner_idx, 
                                                                                mode=mode) 
            
            
            
            del loss_rgb_0, log_dict_0, render_error, all_train_videos, all_train_depths
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
        # breakpoint()
        all_train_videos_1, all_train_depths_1 = self.rendering(gs_cls_1, 
                                                                rand_extrinsics_lst, 
                                                                rand_intrinsics_lst, 
                                                                SCALE=SCALE, 
                                                                invalid_mask=invalid_mask, 
                                                                depth_max=depth_max,
                                                                sha256=sha256, 
                                                                render_mode='RGB+D', 
                                                                mode=mode)
        # all_train_videos_1 = all_train_videos_1.to(next(self.parameters()).dtype)
        
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
        
        # 计算并log每个refiner的loss
        loss_rgb_1, log_dict_1 = self.compute_log_loss_rgb_only_for_refiner(gs_cls_1, 
                                                                            rand_img_lst, 
                                                                            rand_depths_lst, 
                                                                            all_train_videos_1, 
                                                                            all_train_depths_1,
                                                                            sha256, 
                                                                            refiner_idx=refiner_idx, 
                                                                            mode=mode)
        # loss_rgb = self.compute_loss_rgb(all_train_videos, rand_img_lst)
        # avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
        weight = self.refiner_gamma ** (self.num_refiners - 1 - (refiner_idx-1))
        loss_all += weight * loss_rgb_1
        
        # log_memory_usage(f"After refiner_idx {refiner_idx} second refiner loss computation")
        
        # del render_error_1, render_error_GS_1, rand_img_lst, rand_depths_lst, rand_extrinsics_lst, rand_intrinsics_lst, invalid_mask, depths_error, intrinsics_err
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
        #                                                                     refiner_idx=refiner_idx, 
        #                                                                     mode=mode)
        # # loss_rgb = self.compute_loss_rgb(all_train_videos, rand_img_lst)
        # # avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
        # weight = self.refiner_gamma ** (self.num_refiners - 1 - (refiner_idx-1))
        # loss_all += weight * loss_rgb_3
        if mode == 'test':
            return loss_all, log_dict_1, render_error_GS_1
        return loss_all, log_dict_1

    def test(self, batch, mode='test'):
        
        sha256 = batch.get("sha256")
        SCALE = batch.get("scale").item()
        T_norm = batch.get("T_norm")
        attr_GS = batch.get("feat_GS", None)
        feat_voxel = batch.get("feat_voxel")
        rand_img_lst = batch.get("rand_img_lst")
        rand_extrinsics_lst = batch.get("rand_extrinsics_lst")
        rand_intrinsics_lst = batch.get("rand_intrinsics_lst")
        # print(rand_extrinsics_lst.shape, rand_intrinsics_lst.shape)
        rand_depths_lst = batch.get("rand_depths_lst")
        
        intrinsics_err = batch.get("intrinsics_err") # NOTE: refiner 独有
        depths_error = batch.get("depths_error") # NOTE: refiner 独有
        depths_error = depths_error.unsqueeze(1)
        
        # log_memory_usage("After input data extraction")
        
        # ------------------------- 预处理 -------------------------
        new_w, new_h = self.opt.img_resolution
        if rand_depths_lst.shape[-2:] != (new_h, new_w): 
            rand_depths_lst = F.interpolate(rand_depths_lst, size=(new_h, new_w),
                                            mode='bilinear', align_corners=False).squeeze(1).to("cpu")
        depth_max = rand_depths_lst.max().item()
        
        # print(f"[RANK {self.global_rank}], scene: {sha256}, attr_GS: {attr_GS.shape}")
        
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
                                                                sha256=sha256, 
                                                                render_mode='RGB+D', 
                                                                mode=mode)
            # breakpoint()
            # all_train_videos = all_train_videos.to(next(self.parameters()).dtype)

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
            
            loss_rgb_0, log_dict_0 = self.compute_log_loss_rgb_only_for_refiner(attr_GS, 
                                                                                rand_img_lst, 
                                                                                rand_depths_lst, 
                                                                                all_train_videos, 
                                                                                all_train_depths,
                                                                                sha256, 
                                                                                refiner_idx=refiner_idx, 
                                                                                mode=mode) 
            del loss_rgb_0, log_dict_0, render_error, all_train_videos, all_train_depths
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
        # breakpoint()
        all_train_videos_1, all_train_depths_1 = self.rendering(gs_cls_1, 
                                                                rand_extrinsics_lst, 
                                                                rand_intrinsics_lst, 
                                                                SCALE=SCALE, 
                                                                invalid_mask=invalid_mask, 
                                                                depth_max=depth_max,
                                                                sha256=sha256, 
                                                                render_mode='RGB+D', 
                                                                mode=mode)
        # all_train_videos_1 = all_train_videos_1.to(next(self.parameters()).dtype)
        
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
                                            mode=mode)
            self.clean_mem() 
            return gs_cls_1, log_dict
        else:
            # 计算并log每个refiner的loss
            loss_rgb_1, log_dict_1 = self.compute_log_loss_rgb_only_for_refiner(gs_cls_1, 
                                                                                rand_img_lst, 
                                                                                rand_depths_lst, 
                                                                                all_train_videos_1, 
                                                                                all_train_depths_1,
                                                                                sha256, 
                                                                                refiner_idx=refiner_idx, 
                                                                                mode=mode)
            # loss_rgb = self.compute_loss_rgb(all_train_videos, rand_img_lst)
            # avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
            weight = self.refiner_gamma ** (self.num_refiners - 1 - (refiner_idx-1))
            loss_all += weight * loss_rgb_1
        
            # log_memory_usage(f"After refiner_idx {refiner_idx} second refiner loss computation")
            
            # del render_error_1, render_error_GS_1, rand_img_lst, rand_depths_lst, rand_extrinsics_lst, rand_intrinsics_lst, invalid_mask, depths_error, intrinsics_err
            self.clean_mem() 
            return loss_all, log_dict_1
    
    def extract_and_concat_features(self, input_image):
        # Forward pass through the feature extractor
        input_image = input_image.to(next(self.parameters()).dtype)
        # breakpoint()
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
    
    def evaluate_log(self, outputs, 
                     rand_img_lst, rand_depths_lst, 
                     all_train_videos, all_train_depths,
                     sha256, refiner_idx=0, mode=f'test'):
        """ 
        outputs: Gaussian Class
        """
        img_rec_l1, img_rec_ssim, img_rec_lpips, avg_psnr = self.evaluate_rgb(all_train_videos, rand_img_lst)
        # eval_res = (outputs, img_rec_l1, img_rec_ssim, img_rec_lpips, avg_psnr)
        
        # ------------------------- 日志 -------------------------
        log_dict = {
            # "lr": self.optimizers[0].param_groups[0]['lr'],
            f"{mode}/refiner{refiner_idx}_img_rec_l1": img_rec_l1.item(),
            f"{mode}/refiner{refiner_idx}_img_rec_ssim": img_rec_ssim.item(),
            f"{mode}/refiner{refiner_idx}_img_rec_lpips": img_rec_lpips.item(),
            f"{mode}/refiner{refiner_idx}_avg_psnr": avg_psnr.item()
        }

        # self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False if mode == 'train' else True)
        logger.info(f"[{mode}] epoch: {self.current_epoch}, sha256: {sha256}, {mode}_log_dict = " + json.dumps(log_dict, indent=2))
        # self.log_dict(log_dict)
                
        # ------------------------- 可视化 -------------------------
        with torch.no_grad():
            train_save_trigger = (mode == 'train') and (self.current_epoch % self.opt.trainer.save_img_every_epoch == 0) 
            if train_save_trigger or (mode == 'val') or ('test' in mode):
                
                post_fix = f'epoch{self.current_epoch}_refiner{refiner_idx}' if self.current_epoch == 0 else f'refiner{refiner_idx}'

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
                    
                    # gaussians = Gaussian(
                    #     sh_degree=self.sh_degree,
                    #     mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
                    #     scaling_bias = self.rep_config['scaling_bias'],
                    #     opacity_bias = self.rep_config['opacity_bias'],
                    #     scaling_activation = self.rep_config['scaling_activation']
                    # )
                    
                    # gaussians.from_xyz(outputs[:,:3])
                    # gaussians.from_opacity(outputs[:, 3:4])
                    # gaussians.from_scaling(outputs[:, 4:7])
                    # gaussians.from_rotation(outputs[:, 7:11])
                    # gaussians.from_features(outputs[:, 11:])
                    # gaussians.save_ply(os.path.join(self.opt.output_dir, mode,
                    #                 f'{sha256}_{mode}_{post_fix}_refined_GS.ply')) # epoch_{self.current_epoch}
                    
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
                    # outputs.save_ply(os.path.join(self.opt.output_dir, mode,
                    #                 f'{sha256}_{mode}_{post_fix}_refined_GS.ply')) # epoch_{self.current_epoch}
        
        return None, log_dict
    
    
    def compute_log_loss_rgb_only_for_refiner(self, outputs, 
                                              rand_img_lst, rand_depths_lst, 
                                              all_train_videos, all_train_depths,
                                              sha256, refiner_idx=0, mode=f'train'):
        """
        outputs: Gaussian Class
        """
        
        avg_loss_rec = self.compute_loss_rgb(all_train_videos, rand_img_lst)
        avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
        # ------------------------- 总 loss -------------------------
        loss_total = avg_loss_rec

        # ------------------------- 日志 -------------------------
        log_dict = {
            # "lr": self.optimizers[0].param_groups[0]['lr'],
            f"{mode}/refiner{refiner_idx}_loss_total": loss_total.item(),
            f"{mode}/refiner{refiner_idx}_loss_rec_avg": avg_loss_rec.item(),
            f"{mode}/refiner{refiner_idx}_avg_psnr": avg_psnr.item()
        }

        # self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False if mode == 'train' else True)
        logger.info(f"[{mode}] epoch: {self.current_epoch}, sha256: {sha256}, {mode}_log_dict = " + json.dumps(log_dict, indent=2))
        # self.log_dict(log_dict)
                
        # ------------------------- 可视化 -------------------------
        with torch.no_grad():
            train_save_trigger = (mode == 'train') and (self.current_epoch % self.opt.trainer.save_img_every_epoch == 0) 
            if train_save_trigger or (mode == 'val') or ('test' in mode):
                
                post_fix = f'epoch{self.current_epoch}_refiner{refiner_idx}' if self.current_epoch == 0 else f'refiner{refiner_idx}'

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
                    
                    # gaussians = Gaussian(
                    #     sh_degree=self.sh_degree,
                    #     mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
                    #     scaling_bias = self.rep_config['scaling_bias'],
                    #     opacity_bias = self.rep_config['opacity_bias'],
                    #     scaling_activation = self.rep_config['scaling_activation']
                    # )
                    
                    # gaussians.from_xyz(outputs[:,:3])
                    # gaussians.from_opacity(outputs[:, 3:4])
                    # gaussians.from_scaling(outputs[:, 4:7])
                    # gaussians.from_rotation(outputs[:, 7:11])
                    # gaussians.from_features(outputs[:, 11:])
                    # gaussians.save_ply(os.path.join(self.opt.output_dir, mode,
                    #                 f'{sha256}_{mode}_{post_fix}_refined_GS.ply')) # epoch_{self.current_epoch}
                    
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
                    # outputs.save_ply(os.path.join(self.opt.output_dir, mode,
                    #                 f'{sha256}_{mode}_{post_fix}_refined_GS.ply')) # epoch_{self.current_epoch}
        
        return loss_total, log_dict
    
    def compute_render_err(self, all_train_videos, rand_img_lst, sha256, 
                           mode='train', ifsave=True, refiner_iter=0):
        
        rend_feat = self.extract_and_concat_features(all_train_videos)
        gt_feat = self.extract_and_concat_features(rand_img_lst)
        render_error = rend_feat - gt_feat
        
        if ifsave and self.current_epoch % self.opt.trainer.save_img_every_epoch ==0:
            post_fix = f'epoch{self.current_epoch}_refiner{refiner_iter}' if self.current_epoch == 0 else f'refiner{refiner_iter}'
            rend_feat_avg = rend_feat.mean(dim=1).unsqueeze(1)
            save_image(rend_feat_avg, os.path.join(self.opt.output_dir, mode,
                        f"{sha256}_{mode}_{post_fix}_rend_feat_image.png")) # epoch_{self.current_epoch}
            
            gt_feat_avg = gt_feat.mean(dim=1).unsqueeze(1)
            save_image(gt_feat_avg, os.path.join(self.opt.output_dir, mode,
                    f"{sha256}_{mode}_{post_fix}_gt_feat_image.png")) #     

            render_error_avg = render_error.mean(dim=1).unsqueeze(1)
            save_image(render_error_avg, os.path.join(self.opt.output_dir, mode,
                        f"{sha256}_{mode}_{post_fix}_render_error.png")) # epoch_{self.current_epoch}

        return render_error
    
    # def training_step(self, batch):
    #     # torch.autograd.set_detect_anomaly(True)
    #     self.model.train()
        
    #     loss_all, log_dict = self.forward(batch, mode='train')

    #     return loss_all
    # def validation_step(self, batch):
    #     self.model.eval()
        
    #     with torch.no_grad():
    #         loss_all, log_dict = self.forward(batch, mode='val')
                
    def on_test_start(self):
        # 每次 test 前清空存储
        self.test_outputs = {}

    def test_step(self, batch, dataloader_idx=0, mode='test'):
        refiner_idx = 1
        self.model.eval()
        with torch.no_grad():
            sha256 = batch.get("sha256")
            
            rend_GS, log_dict = self.forward(batch, mode=mode)
            
            # log_dict = {
            #     # "lr": self.optimizers[0].param_groups[0]['lr'],
            #     f"{mode}/refiner{refiner_idx}_img_rec_l1": img_rec_l1.item(),
            #     f"{mode}/refiner{refiner_idx}_img_rec_ssim": img_rec_ssim.item(),
            #     f"{mode}/refiner{refiner_idx}_img_rec_lpips": img_rec_lpips.item(),
            #     f"{mode}/refiner{refiner_idx}_avg_psnr": avg_psnr.item()
            # }
            log_dict['sha256'] = sha256
            
            # rend_GS, img_rec_l1, img_rec_ssim, img_rec_lpips, avg_psnr = eval_res
            
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

        for dataloader_idx, dataloader_res in self.test_outputs.items():

            fieldnames = dataloader_res.keys()
            # 保存为CSV文件
            with open(os.path.join(self.opt.output_dir, f'test_results_dataloader{dataloader_idx}.csv'), mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()  # 写入列名（字段）
                writer.writerows(dataloader_res)  # 写入字典数据
        
        self.test_outputs.clear() # 清空，避免多个 epoch 累加


def main():
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
    opt.logger_dir = opt.output_dir.replace('outputs', 'tensorboadblogs')    
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
    
    
    # 
    # accelerate 替换 pl: 手动调用 Module 中的函数
    # accelerator = Accelerator(log_with='tensorboard', project_dir=exp_dir)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(
        backend="hccl"
    )
    accelerator = Accelerator(
        # project_config=project_config,
        log_with='tensorboard',
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        project_dir=exp_dir
        # kwargs_handlers=[ddp_kwargs],
    )
    
    device = torch.device(f'npu:{accelerator.local_process_index}')
    if accelerator.local_main_process_first() == 0:
        print("====== cfg: ", opt)
    else:
        print("====== exp_dir: ", exp_dir)    
        
        
    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    from train.datasets import RefinerDataset as dataset
    from train.datasets import prepare_dataloader
    
    if opt.select_scene is None:
        train_dataset = dataset(train_metadata, opt, stage='train')
        val_dataset = dataset(val_metadata, opt, stage='val')
        
        train_loader = prepare_dataloader(train_dataset, 
                                          batch_size=opt.trainer.batch_size, 
                                          num_workers=6,
                                          shuffle=True,
                                          drop_last=True,
                                          pin_memory=True,
                                          process_group=get_data_parallel_group(),
                                          world_size=accelerator.num_processes,
                                          rank=accelerator.process_index)
        val_loader = prepare_dataloader(val_dataset, 
                                        batch_size=opt.trainer.batch_size, 
                                        num_workers=6,
                                        shuffle=False,
                                        drop_last=False,
                                        pin_memory=True,
                                        process_group=get_data_parallel_group(),
                                        world_size=accelerator.num_processes,
                                        rank=accelerator.process_index)
        # train_loader = DataLoader(train_dataset, batch_size=opt.trainer.batch_size, 
        #                           shuffle=True, num_workers=6, pin_memory=True, 
        #                           persistent_workers=True,  collate_fn=collate_skip_none)
        # val_loader = DataLoader(val_dataset, batch_size=opt.trainer.batch_size, 
        #                         shuffle=False, num_workers=4, pin_memory=True, 
        #                         persistent_workers=True )
        if opt.test:
            test_metadata = val_metadata
            test_dataset = dataset(test_metadata, opt, stage='test')
            test_loader = prepare_dataloader(test_dataset, 
                                        batch_size=opt.trainer.batch_size, 
                                        num_workers=6,
                                        shuffle=False,
                                        drop_last=False,
                                        pin_memory=True,
                                        process_group=get_data_parallel_group(),
                                        world_size=accelerator.num_processes,
                                        rank=accelerator.process_index)
            # test_loader = DataLoader(test_dataset, batch_size=opt.trainer.batch_size, 
            #                     shuffle=False, num_workers=16, pin_memory=True, 
            #                     persistent_workers=True )
            
            # test_train_novel_metadata = train_metadata
            # test_train_novel_dataset = dataset(test_train_novel_metadata, opt, stage='test_trainset_novel')
            # test_train_novel_loader = DataLoader(test_train_novel_dataset, batch_size=opt.trainer.batch_size, 
            #                     shuffle=False, num_workers=16, pin_memory=True, 
            #                     persistent_workers=True )

    else:
        train_dataset = dataset(metadata, opt, stage='train')
        val_dataset = dataset(metadata, opt, stage='val')
        train_loader = prepare_dataloader(train_dataset, 
                                          batch_size=opt.trainer.batch_size, 
                                          num_workers=4,
                                          shuffle=True,
                                          drop_last=True,
                                          pin_memory=True,
                                          process_group=get_data_parallel_group(),
                                          world_size=accelerator.num_processes,
                                          rank=accelerator.process_index)
        val_loader = prepare_dataloader(val_dataset, 
                                        batch_size=opt.trainer.batch_size, 
                                        num_workers=4,
                                        shuffle=False,
                                        drop_last=False,
                                        pin_memory=True,
                                        process_group=get_data_parallel_group(),
                                        world_size=accelerator.num_processes,
                                        rank=accelerator.process_index)
        
        # train_loader = DataLoader(train_dataset, batch_size=opt.trainer.batch_size, 
        #                           shuffle=True, num_workers=4, pin_memory=True, 
        #                           persistent_workers=True,  collate_fn=collate_skip_none)
        # val_loader = DataLoader(val_dataset, batch_size=opt.trainer.batch_size, 
        #                           shuffle=True, num_workers=4, pin_memory=True, 
        #                           persistent_workers=True,  collate_fn=collate_skip_none)
    
    os.makedirs(os.path.join(opt.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'test'), exist_ok=True)
    # os.makedirs(os.path.join(opt.output_dir, 'latents_train'), exist_ok=True)
    # os.makedirs(os.path.join(opt.output_dir, 'latents_val'), exist_ok=True)
    
    
    
    model = RefinerModel(opt)
    if opt.models.args.use_fp16:
        dtype = torch.float16
    elif opt.models.args.use_bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
        
    # model = model.to(device, dtype)
    
    optimizer, scheduler = model.configure_optimizers()
    optimizer, scheduler = optimizer[0], scheduler[0]
    # model: RefinerModel = accelerator.prepare_model(model)
    # optimizer = accelerator.prepare_optimizer(optimizer)
    # scheduler = accelerator.prepare_scheduler(scheduler)
    # accelerator.register_for_checkpointing(model, optimizer, scheduler)
    rank0 = accelerator.is_main_process
    
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel): model = model.module

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    # torch.set_default_dtype(torch.float)
    num_steps_per_epoch = len(train_loader)
    num_steps_per_epoch_val = len(val_loader)
    logger.info("Boost model for distributed training")
    # print(next(model.feature_extractor.parameters()).dtype)
    # breakpoint()
    
    # =======================================================
    # 6. training loop
    # =======================================================
    
    start_epoch = start_step = log_step = sampler_start_idx = start_step_eval = 0
    
    # 6.1. resume training
    ckpt_list = glob.glob(os.path.join(exp_dir, "epoch*.ckpt"))    
    if not opt.test and len(ckpt_list) > 0:
        resume_from_checkpoint = os.path.join(exp_dir, "last.ckpt")
    elif opt.ckpt and os.path.exists(opt.ckpt):
        last_epoch, last_step, sampler_start_idx = deepspeed_load(model, opt.ckpt)
        logger.info(f"[INFO] Loaded checkpoint {opt.ckpt} at epoch {last_epoch} step {last_step}")
        last_global_step = last_epoch * num_steps_per_epoch + last_step
        scheduler.load_state_dict(torch.load(os.path.join(opt.ckpt, f'global_step{last_global_step + 1}/lr_scheduler.pt'))) 
        start_step = last_step + 1 if last_step < num_steps_per_epoch - 1 else 0
        start_epoch = last_epoch if last_step < num_steps_per_epoch - 1 else last_epoch + 1
        # start_epoch, start_step, sampler_start_idx = load(None, model, optimizer, lr_scheduler, cfg.resume) 
    logger.info(f"[INFO] Training for {opt.trainer.max_epochs} epochs with {num_steps_per_epoch} steps per epoch")
    logger.info(f"[INFO] Evaluating every {opt.trainer.ckpt_every} interval with {num_steps_per_epoch_val} steps per epoch")
    
    
    profiling = False

    if profiling:
        # time_fwd, time_bwd = [], []
        import torch_npu
        experimental_config = torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)
        prof = torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.NPU,
                                    torch_npu.profiler.ProfilerActivity.CPU],
                        with_stack=True,
                        record_shapes=False,
                        profile_memory=True,
                        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=0,
                                                                skip_first=0),
                        experimental_config=experimental_config,
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./profiling/train_sdf_256/'))
    
    # =======================================================
    # 9. Training Loop
    # =======================================================
    all_step = opt.trainer.max_epochs * num_steps_per_epoch
    # if opt.test:
    #     test_loader = accelerator.prepare_data_loader(test_loader)
    #     model.on_test_start()
    #     for batch_idx, test_batch in enumerate(test_loader):
    #         out = model.test_step(test_batch, batch_idx)
    #         for k, v in out.items(): model.test_outputs[k].append(v)
    #     model.on_test_epoch_end()
    # else:
        # if rank0: print('PREPARING DATALOADER')
        # train_loader = accelerator.prepare_data_loader(train_loader)
        # val_loader = accelerator.prepare_data_loader(val_loader)
    # breakpoint()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel): model.module.on_train_start()
    else: model.on_train_start()
    for epoch in trange(int(opt.trainer.max_epochs), disable=not rank0):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel): model.module.current_epoch = epoch
        else: model.current_rpoch = epoch
        train_loader_iter = iter(train_loader)
        logger.info(f"Beginning epoch {epoch}...")
        log_step = 0
        # tbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, disable=not rank0)
        with tqdm(range(len(train_loader)), desc=f"Epoch {epoch}",
            disable=not accelerator.local_process_index == 0, 
            total=len(train_loader)) as pbar:
            for step in pbar:
                input_dict = next(train_loader_iter)
                print(input_dict)
                input_dict = {k: v[0] if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
                model.train()
                
                loss, log_dict = model(input_dict, mode='train')

                optimizer.zero_grad()
                # breakpoint()
                accelerator.backward(loss)
                optimizer.step()
                
                log_dict = reduce_metrics(log_dict, device)    
                log_step += 1
                global_step = epoch * num_steps_per_epoch + step
                
                pbar.set_postfix(log_dict)
                
                if dist.get_rank() == 0:
                    accelerator.log(log_dict)
                                
                ## Log to tensorboard
                # if (epoch % opt.trainer.log_every == 0 or epoch + 1 == opt.trainer.max_epochs):
                #     if accelerator.is_main_process: # 只在global_rank = 0上写入log和writer
                #         for k, v in log_dict.items():
                #             writer.add_scalar(k, v, global_step)
                        
                #     running_metrics = {k: 0.0 for k in running_metrics}
                #     log_step = 0

                del input_dict, loss
            
                # Save checkpoint
                if rank0 and not (epoch % opt.trainer.save_img_every_epoch):
                    accelerator.save_state(os.path.join(exp_dir, f'epoch{epoch:04}.ckpt'))
                    logger.info(f"Saved checkpoint at epoch {epoch} step {step} global_step {global_step} to {exp_dir}")
                
                # Evaluation
                if opt.trainer.eval_every > 0 and (epoch % opt.trainer.eval_every == 0):
                    logger.info(f"Evaluating on epoch {epoch} global_step {global_step}......")
                    model.eval()
                    with torch.no_grad():
                        val_loader_iter = iter(val_loader)
                        with tqdm(range(start_step_eval, num_steps_per_epoch_val),
                                  desc=f"Validation Epoch {epoch}",
                                  disable=not accelerator.local_process_index==0) as pbar_val:
                            for _ in pbar_val:
                                input_dict = next(val_loader_iter)
                                input_dict = {k: v[0].to(device) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
                                loss_val, log_val_dict = model(input_dict, mode='val')
                                
                                pbar_val.set_postfix(log_val_dict)
                                
                    # all readuce mean         
                    running_eval_metrics = reduce_metrics(running_eval_metrics, device)    
                    if rank0: accelerator.log(log_val_dict)
            scheduler.step()
            
                                
    if rank0: accelerator.save_state(os.path.join(exp_dir, "final_checkpoint.ckpt"))


if __name__ == '__main__':
    main()