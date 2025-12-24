"""
2025.04.21 : render的时候是一张一张render的，因为多张图一起render显存占用很大
2025.04.22 : 修改了数据集，增加了dataset_filter
2025.04.29 : 修改config输入实验设置，和构建模型等

2025.06.30 : 增加了depth监督
2025.07.01 : 增加fixed select images and frames
2025.10.27 : 修改了模型forward的统一接口，整理了模型框架
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
import json
import time
import argparse

import numpy as np
import pandas as pd
# from loguru import logger
# import open3d as o3d
# from PIL import Image
import lpips
import gc
import json
import glob
import csv
import matplotlib.pyplot as plt

import torch
import torch_npu
# torch.autograd.set_detect_anomaly(True)
# torch.backends.cuda.preferred_linalg_library('magma') # magma
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms as tvf

import src.modules.sparse as sp
from src.representations.gaussian import Gaussian
from src.utils.demo_utils import gen_demo
from src.utils.train_utils import batch_project_and_aggregate_chunkwise, _filter_dataset_new, _filter_dataset, get_latest_ckpt, seed_everything, saveRuntimeCode
from src.models.GSdecoder.decoder_RGBcnnCamDepthV2 import DecoderOnlyRGBCNNCamDepthInV2
import torch.multiprocessing as mp
# mp.set_start_method('spawn')

# seed_everything(520)
seed_everything(42)

torch.set_grad_enabled(True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # 归一化到 [-1, 1]
])

# 定义 transform，将图片转换为 tensor，并归一化到 [0,1]
transform_novel = transforms.Compose([
    transforms.ToTensor(),  # 会把 [0,255] 转为 [0,1] 的 tensor
])

# SAVE_ROOT = '/efs-mi-east4-2/datasets/scannetpp_v2_opencv_pinhole/GSandVoxel_feats_normV14/GSandVoxel_feats_11_11_32views_val21.6'
SAVE_ROOT = None


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

from train.baseModel import BaseModel
class GSDecoderModel(BaseModel):
    def __init__(self, opt, if_norm_depth=True, load_pretrained=None):
        super(GSDecoderModel, self).__init__(opt)
        self.vs_size = 2/128
        self.if_norm_depth = if_norm_depth
        # 指定在哪些 epoch 做 inference
        self.infer_epochs = [59, 119, 199] + list(range(199, self.opt.trainer.max_epochs, 200))
        self.if_fix_xyz = False
        self.model: DecoderOnlyRGBCNNCamDepthInV2

        if load_pretrained:
            ckpt = torch.load(load_pretrained, map_location='cpu')
            featEnc_weights = {k.replace('model.featEnc.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('model.featEnc')}
            self.model.featEnc.load_state_dict(featEnc_weights)

    # =====================================================
    # 统一 forward（推理逻辑）
    # =====================================================
    def forward(self, batch, query=None):
        SCALE = batch.get("scale").item()
        voxel_points = batch.get("voxel_points")[0]
        rand_img_lst = batch.get("rand_img_lst")[0] # .to("cpu")
        rand_extrinsics_lst = batch.get("rand_extrinsics_lst")[0]
        rand_ori_extrinsics_lst = batch.get("rand_ori_extrinsics_lst")[0]
        rand_intrinsics_lst = batch.get("rand_intrinsics_lst")[0]
        rand_depths_lst = batch.get("rand_depths_lst")[0] # shape: [N, 1, H, W]
        T_norm = batch.get("T_norm")[0].to("cpu")  # 显式加默认值
        
        grid_origin = None # 如果使用了voxel_points, 就不需要grid_origin来反算xyz了
        # bs = feats.shape[0] # 这里我希望bs=1, 因为trellis那个sparsetensor之后的处理没有考虑多batch的操作
        # assert len(rand_img_lst) == len(rand_extrinsics_lst) == len(rand_intrinsics_lst) == 1
        
        # print(f"voxel_points: {voxel_points.shape}, rand_img_lst: {rand_img_lst.shape}, rand_extrinsics_lst: {rand_extrinsics_lst.shape}",
        #       f"rand_intrinsics_lst: {rand_intrinsics_lst.shape}, rand_depths_lst: {rand_depths_lst.shape}, T_norm: {T_norm.shape}",
        #       f"SCALE: {SCALE}")
        
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
                if_fix_xyz=False
            )
            # del feats

        elif model_name in ['SLatQueryRGBCNNCamDepthInSepMLPGaussianDecoderOnly']:
            # voxel_indices = voxel_indices[0]
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
                voxel_indices=None,
                voxel_align_ratio=self.voxel_align_ratio,
                if_fix_xyz=False
            )
            
        elif self.opt.models.name in ['SLatQueryRGBCNNCamDepthInIrregSampSepMLPGaussianDecoderOnly',
                                      'SLatQueryRGBCNNCamDepthInIrregSampCASepMLPGaussianDecoderOnly']:
            outputs = self.model(rgbOris=rand_img_lst,
                                 depths=rand_depths_lst, 
                                 intrinsic=rand_intrinsics_lst.clone(),
                                 ori_w2cs=rand_ori_extrinsics_lst,
                                 w2cs=rand_extrinsics_lst, 
                                 scale=SCALE, 
                                 T_norm=T_norm, 
                                 voxel_points=voxel_points, 
                                 voxel_align_ratio=self.voxel_align_ratio)

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
                if_fix_xyz=False
            )

        else:
            raise NotImplementedError(f"Model {model_name} not implemented in src.models")
        
        self.clean_mem()

        return {
                # "sha256": sha256, 
                "outputs": outputs, 
                # "rand_img_lst": rand_img_lst, 
                # "rand_depths_lst": rand_depths_lst,
                # "rand_extrinsics_lst": rand_extrinsics_lst,
                # "rand_intrinsics_lst": rand_intrinsics_lst
                }
    
    def compute_log_loss(self, outputs, 
                         rand_img_lst, rand_depths_lst, 
                         rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                         sha256, mode='train'):
        
        depth_max = rand_depths_lst.max().item()
        # new_w, new_h = self.opt.img_resolution
        new_w, new_h = self.opt.trainer.rend_sv_res
        
        rand_img_lst = F.interpolate(rand_img_lst, size=(new_h, new_w), mode='area')
        rand_depths_lst = F.interpolate(rand_depths_lst.unsqueeze(1), size=(new_h, new_w), mode='nearest').squeeze(1)
        
        # ------------------------- 预处理 -------------------------
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

        assert isinstance(outputs, list) and len(outputs) == 1
        
        # ------------------------- 渲染 -------------------------
        all_train_videos, all_train_depths = self.rendering(outputs[0], rand_extrinsics_lst, rand_intrinsics_lst,
                                                            SCALE=SCALE, 
                                                            # invalid_mask=invalid_mask, 
                                                            depth_max=depth_max,
                                                            # batch_idx=batch_idx, 
                                                            sha256=sha256, 
                                                            render_mode='RGB+D', 
                                                            img_res=(new_w, new_h),
                                                            mode=mode)
        
        # ------------------------- 可视化 -------------------------
        train_save_trigger = (mode == 'train') and (self.trainer.current_epoch % self.opt.trainer.save_img_every_epoch == 0)
        if train_save_trigger or (mode == 'val') or ('test' in mode):
            if isinstance(outputs, list) and len(outputs) == 1:
                gausssian = outputs[0]
            elif isinstance(outputs, Gaussian):
                gausssian = outputs
            else:
                raise NotImplementedError(f"Unsupported output type for visualization: {type(outputs)}")
            
            if SAVE_ROOT is None:  # validation or test
                post_fix = 'epoch0_' if self.trainer.current_epoch == 0 else '_'
                save_image(all_train_videos, os.path.join(self.opt.output_dir, mode,
                            f"{sha256}_{mode}_{post_fix}rgb_image.png")) # epoch_{self.trainer.current_epoch}
                save_image(rand_img_lst, os.path.join(self.opt.output_dir, mode,
                        f"{sha256}_{mode}_{post_fix}GT_rgb_image.png")) # epoch_{self.trainer.current_epoch}
                save_image(all_train_depths.unsqueeze(1), os.path.join(self.opt.output_dir, mode,
                            f"{sha256}_{mode}_{post_fix}depth.png")) # epoch_{self.trainer.current_epoch}
                save_image(rand_depths_lst.unsqueeze(1), os.path.join(self.opt.output_dir, mode,
                        f"{sha256}_{mode}_{post_fix}GT_depth.png")) # epoch_{self.trainer.current_epoch}
                gausssian.save_ply(os.path.join(self.opt.output_dir, mode,
                        f'{sha256}_{mode}_{post_fix}_decoded_GS.ply')) # epoch_{self.trainer.current_epoch}
            else:  # save features for refiner training
                feat_GS = torch.cat([
                    gausssian.get_xyz,
                    gausssian.get_opacity,
                    gausssian.get_scaling,
                    gausssian.get_rotation,
                    gausssian.get_features.reshape(-1, self.rgb_dim),
                ], dim=-1).clone().detach().cpu().numpy()

                GS_savepath = os.path.join(f'{SAVE_ROOT}/{sha256[0]}_decOnly_train_dict_GS.npz')
                print('saved feat GS in', GS_savepath)
                np.savez(GS_savepath, feat_GS=feat_GS)

        # ------------------------- 计算指标 ------------------------- 
        # maskr住无效深度的gt图像和depth区域
        if self.opt.use_depth_mask:
            rand_img_lst = rand_img_lst.masked_fill(invalid_mask.unsqueeze(1), 0)
            all_train_videos = all_train_videos.masked_fill(invalid_mask.unsqueeze(1), 0)
            rand_depths_lst = rand_depths_lst.masked_fill(invalid_mask, 1e-6)  
            all_train_depths = all_train_depths.masked_fill(invalid_mask, 1e-6)  

        avg_loss_rec = self.compute_loss_rgb(all_train_videos, rand_img_lst)
        avg_loss_depth = self.compute_loss_depth(all_train_depths, rand_depths_lst)
        loss_opa, loss_scale, loss_gs_dispersion, loss_gs_diversity = self.compute_loss_gs(outputs)
        
        avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
        
        # ------------------------- 总 loss -------------------------
        loss_total = avg_loss_rec + loss_opa + loss_scale + avg_loss_depth + loss_gs_dispersion + loss_gs_diversity

        # ------------------------- 日志 -------------------------
        log_dict = {
            "lr": self.trainer.optimizers[0].param_groups[0]['lr'],
            f"{mode}/loss_total": loss_total.item(),
            f"{mode}/loss_rec_avg": avg_loss_rec.item(),
            f"{mode}/loss_opa": loss_opa.item(),
            f"{mode}/loss_scale": loss_scale.item(),
            f"{mode}/loss_depth_avg": avg_loss_depth.item(),
            f"{mode}/avg_psnr": avg_psnr.item()
        }
            
        if hasattr(self.opt.trainer, "loss_weight_GSdispersion") and self.opt.trainer.loss_weight_GSdispersion > 0:
            log_dict[f"{mode}/loss_gs_dispersion"] = loss_gs_dispersion.item()
        if hasattr(self.opt.trainer, "loss_weight_GSdiversity") and self.opt.trainer.loss_weight_GSdiversity > 0:
            log_dict[f"{mode}/loss_gs_diversity"] = loss_gs_diversity.item()

        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False if mode == 'train' else True)
        # logger.info(f"[{tag}] epoch: {self.trainer.current_epoch}, sha256: {sha256}, {mode}_log_dict = " + json.dumps(log_dict, indent=2))

        return loss_total, avg_psnr
    
    @staticmethod
    def process_image_tensor_batch_zxx(img_tensor):
        if img_tensor.min() < 0 or img_tensor.max() > 1:
            raise ValueError("Input tensor should be in the [0, 1] range")

        batch_size, _, height, width = img_tensor.shape

        if width > height:
            new_height = 448
            new_width = int(width * (new_height / height))
        else:
            new_width = 448
            new_height = int(height * (new_width / width))
        
        transform_resize = tvf.Resize((new_height, new_width))
        img_tensor_resized = torch.stack([transform_resize(img) for img in img_tensor])

        left = (new_width - 448) // 2
        top = (new_height - 448) // 2
        img_tensor_cropped = img_tensor_resized[:, :, top:top+448, left:left+448]

        return img_tensor_cropped

    @staticmethod
    def process_depth_tensor_batch_zxx(depth_tensor):
        batch_size, _, height, width = depth_tensor.shape

        if width > height:
            new_height = 448
            new_width = int(width * (new_height / height))
        else:
            new_width = 448
            new_height = int(height * (new_width / width))

        depth_tensor_resized = F.interpolate(
            depth_tensor, 
            size=(new_height, new_width),  
            mode='nearest',
            # align_corners=False  
        )
        left = (new_width - 448) // 2
        top = (new_height - 448) // 2
        depth_tensor_cropped = depth_tensor_resized[:, :, top:top+448, left:left+448]

        return depth_tensor_cropped
    
    @staticmethod
    def process_intriscis(K, original_width, original_height):

        if original_width > original_height:
            new_height = 448
            new_width = int(original_width * (new_height / original_height))
        else:
            new_width = 448
            new_height = int(original_height * (new_width / original_width))
        print(new_width, new_height, original_width, original_height)

        scale_x = new_width / original_width
        scale_y = new_height / original_height
        
        K_new = torch.tensor(K).clone()
        
        # 更新 fx, fy
        K_new[:, 0, 0] *= scale_x  # 更新 fx
        K_new[:, 1, 1] *= scale_y  # 更新 fy
        
        # Center crop
        left = (new_width - 448) // 2
        top = (new_height - 448) // 2
        right = left + 448
        bottom = top + 448
        
        # 计算裁剪后的中心坐标
        crop_center_x = (left + right) / 2
        crop_center_y = (top + bottom) / 2
        
        # 更新 cx, cy
        K_new[:, 0, 2] = (K_new[:, 0, 2] - (original_width / 2)) * scale_x + crop_center_x  # 更新 cx
        K_new[:, 1, 2] = (K_new[:, 1, 2] - (original_height / 2)) * scale_y + crop_center_y  # 更新 cy

        return K_new
    
    def evaluate_log(self, outputs, rand_img_lst, rand_depths_lst, 
                    rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                    sha256, batch_idx, isCenterCrop=True, mode='test', test_dataloader_idx=0):
        # print(f"all_train_depths: {all_train_depths.shape, all_train_depths.min(), all_train_depths.max()}")
        # print(f"rand_depths_lst: {rand_depths_lst.shape, rand_depths_lst.min(), rand_depths_lst.max()}, ")   

        print(f"rand_img_lst: {rand_img_lst.shape}, rand_extrinsics_lst: {rand_extrinsics_lst.shape}",
              f"rand_intrinsics_lst: {rand_intrinsics_lst.shape}, rand_depths_lst: {rand_depths_lst.shape}, SCALE: {SCALE}")

        depth_max = rand_depths_lst.max().item()
        # new_w, new_h = self.opt.img_resolution
        new_w, new_h = self.opt.trainer.rend_sv_res
        if self.opt.trainer.render_option == 'gsplat':
            rand_intrinsics_lst[:, 0, :] *= new_w
            rand_intrinsics_lst[:, 1, :] *= new_h
        
        rand_img_lst = F.interpolate(rand_img_lst, size=(new_h, new_w), mode='area')
        rand_depths_lst = F.interpolate(rand_depths_lst, size=(new_h, new_w), mode='nearest')   
        
        # 归一化depth 
        if self.if_norm_depth:
            rand_depths_lst = rand_depths_lst / depth_max

        # maskr住无效深度的gt 图像区域
        if self.opt.use_depth_mask:
            invalid_mask = rand_depths_lst <= 1e-6
        else: invalid_mask = None
        
        if isCenterCrop:
            # Center Crop
            rand_img_lst = self.process_image_tensor_batch_zxx(rand_img_lst)
            rand_depths_lst = self.process_depth_tensor_batch_zxx(rand_depths_lst)
            rand_intrinsics_lst = self.process_intriscis(rand_intrinsics_lst, new_w, new_h)
            # rand_intrinsics_lst = rand_intrinsics.unsqueeze(0).repeat(len(rand_img_lst), 1, 1)
            img_res=(448, 448)
        else:
            img_res=(new_w, new_h)
        
        render_time_start = time.time()
        all_train_videos, all_train_depths = self.rendering(outputs, rand_extrinsics_lst, rand_intrinsics_lst,
                                                            SCALE=SCALE, 
                                                            # invalid_mask=invalid_mask, 
                                                            depth_max=depth_max,
                                                            # batch_idx=batch_idx, 
                                                            sha256=sha256, 
                                                            render_mode='RGB+D',
                                                            img_res=img_res,
                                                            mode=mode)
        render_time = time.time() - render_time_start
        
        print(f"all_train_depths: {all_train_depths.shape, all_train_depths.min(), all_train_depths.max()}")
        print(f"rand_depths_lst: {rand_depths_lst.shape, rand_depths_lst.min(), rand_depths_lst.max()}, ")   
        
        # 归一化depth
        if self.if_norm_depth:
            all_train_depths = all_train_depths / all_train_depths.max().item()
        
        if not isCenterCrop:
            all_train_videos = all_train_videos[..., 112:560]
            all_train_depths = all_train_depths[..., 112:560]
            rand_img_lst = rand_img_lst[..., 112:560]
            rand_depths_lst = rand_depths_lst[..., 112:560]
            rand_depths_lst[rand_depths_lst <= 1e-6] = 1e-6
            all_train_depths[all_train_depths <= 1e-6] = 1e-6
        
                
        print(f"all_train_depths: {all_train_depths.shape, all_train_depths.min(), all_train_depths.max()}")
        print(f"rand_depths_lst: {rand_depths_lst.shape, rand_depths_lst.min(), rand_depths_lst.max()}, ")   
        
        if all_train_depths.ndim == 3: all_train_depths.unsqueeze_(1)
        print('rendering: ', all_train_depths.shape)
        # ------------------------- 可视化 -------------------------            
        # post_fix = 'epoch0_' if self.trainer.current_epoch == 0 else '_'
        post_fix = ''

        output_dir = os.path.join(self.opt.output_dir, mode, str(test_dataloader_idx), sha256)
        os.makedirs(output_dir, exist_ok=True)
        
        save_image(all_train_videos, os.path.join(output_dir, f"{sha256}_{mode}_{post_fix}rgb_image.png")) 
        
        save_image(rand_img_lst, os.path.join(output_dir, f"{sha256}_{mode}_{post_fix}GT_rgb_image.png")) 
        
        save_image(all_train_depths, os.path.join(output_dir, f"{sha256}_{mode}_{post_fix}depth.png")) 
        
        save_image(rand_depths_lst, os.path.join(output_dir, f"{sha256}_{mode}_{post_fix}GT_depth.png")) 
        
        save_rgb_sequence(rand_img_lst, output_dir, prefix="rgb_gt")
        save_rgb_sequence(all_train_videos, output_dir, prefix="rgb_rend")
        save_depth_sequence(rand_depths_lst, output_dir, prefix="depth_gt", cmap="plasma")
        save_depth_sequence(all_train_depths, output_dir, prefix="depth_rend", cmap="plasma")
        
        if isinstance(outputs, list) and len(outputs) == 1:
            gausssian = outputs[0]
        elif isinstance(outputs, Gaussian):
            gausssian = outputs
        else:
            raise NotImplementedError(f"Unsupported output type for visualization: {type(outputs)}")
        
        gausssian.save_ply(os.path.join(output_dir, f'{sha256}_{mode}_{post_fix}_decoded_GS.ply'))
        
        if SAVE_ROOT is not None:  # validation or test
            gs_xyz = gausssian.get_xyz
            gs_opa = gausssian.get_opacity
            gs_sca = gausssian.get_scaling
            gs_rot = gausssian.get_rotation
            gs_fdc = gausssian.get_features.reshape(-1, self.rgb_dim) # [N, 3*(sh+1)**2]
            feat_GS = torch.cat([
                                    gausssian.get_xyz,
                                    gausssian.get_opacity,
                                    gausssian.get_scaling,
                                    gausssian.get_rotation,
                                    gausssian.get_features.reshape(-1, self.rgb_dim),
                                ], dim=-1).clone().detach().cpu().numpy()
            GS_savepath = os.path.join(f'{SAVE_ROOT}/{sha256[0]}_decOnly_train_dict_GS.npz')
            print('saved feat GS in', GS_savepath)
            np.savez(GS_savepath, feat_GS=feat_GS)
        
        # ------------------------- 计算指标 ------------------------- 
        # maskr住无效深度的gt图像和depth区域
        if self.opt.use_depth_mask:
            rand_img_lst = rand_img_lst.masked_fill(invalid_mask.unsqueeze(1), 0)
            all_train_videos = all_train_videos.masked_fill(invalid_mask.unsqueeze(1), 0)
            rand_depths_lst = rand_depths_lst.masked_fill(invalid_mask, 1e-6)  
            all_train_depths = all_train_depths.masked_fill(invalid_mask, 1e-6)  
        
        if invalid_mask is not None:
            valid_mask = ~invalid_mask
        else:
            valid_mask = (all_train_depths>1e-6) & (rand_depths_lst>1e-6)
        img_rec_l1, img_rec_ssim, img_rec_lpips, avg_psnr = self.evaluate_rgb(all_train_videos, rand_img_lst)
        absRel = self.abs_relative_difference(all_train_depths, rand_depths_lst, valid_mask=valid_mask) # 
        delta1 = self.delta1_acc(all_train_depths, rand_depths_lst, valid_mask=~invalid_mask if invalid_mask is not None else None)
        # ------------------------- 日志 -------------------------
        log_dict = {
            # "lr": self.trainer.optimizers[0].param_groups[0]['lr'],
            f"{mode}/absRel": absRel.item(),
            f"{mode}/delta1": delta1.item(),
            f"{mode}/img_rec_l1": img_rec_l1.item(),
            f"{mode}/img_rec_ssim": img_rec_ssim.item(),
            f"{mode}/img_rec_lpips": img_rec_lpips.item(),
            f"{mode}/avg_psnr": avg_psnr.item(),
            f"{mode}/num_view": len(rand_img_lst),
            f"{mode}/num_GS": gausssian.get_xyz.shape[0],
            f"{mode}/render_time": render_time,
        }
        return None, log_dict
    
    def training_step(self, batch, batch_idx):
        # torch.autograd.set_detect_anomaly(True)
        self.model.train()
        sha256 = batch.get("sha256")[0]
        SCALE = batch.get("scale").item()
        
        self.clean_mem()
        
        out = self.forward(batch)
        outputs = out.get("outputs")    
        
        rand_depths_lst = batch.get("rand_depths_lst")[0] # shape: [N, 1, H, W]
        rand_img_lst = batch.get("rand_img_lst")[0] # shape: [N, 3, H, W]
        rand_extrinsics_lst = batch.get("rand_extrinsics_lst")[0] # shape: [N, 4, 4]
        rand_intrinsics_lst = batch.get("rand_intrinsics_lst")[0] # shape: [N, 3, 3]
        
        loss_total, _ = self.compute_log_loss(outputs,  
                                            rand_img_lst, rand_depths_lst, 
                                            rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                                            sha256, batch_idx, mode=f'train')
        
        return loss_total

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        sha256 = out.get("sha256")[0]
        SCALE = batch.get("scale").item()          
        
        self.clean_mem()
        
        with torch.no_grad():
            
            out = self.forward(batch)
            
            sha256 = out.get("sha256")
            outputs = out.get("outputs")   
            
            rand_depths_lst = batch.get("rand_depths_lst")[0] # shape: [N, 1, H, W]
            rand_img_lst = batch.get("rand_img_lst")[0] # shape: [N, 3, H, W]
            rand_extrinsics_lst = batch.get("rand_extrinsics_lst")[0] # shape: [N, 4, 4]
            rand_intrinsics_lst = batch.get("rand_intrinsics_lst")[0] # shape: [N, 3, 3] 
            
            assert len(rand_extrinsics_lst) == len(rand_intrinsics_lst) == len(rand_depths_lst) == len(rand_img_lst)
            loss_total, _ = self.compute_log_loss(outputs,  
                                                    rand_img_lst, rand_depths_lst, 
                                                    rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                                                    sha256, batch_idx, mode=f'val')
        
    def on_test_start(self):
        # 每次 test 前清空存储
        self.test_outputs = {}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.eval()
        sha256 = batch.get("sha256")[0]
        SCALE = batch.get("scale").item()
        
        self.clean_mem()
        
        with torch.no_grad():
            
            start_forward_time = time.time()
            out = self.forward(batch)
            forward_time = time.time() - start_forward_time

            # info = [sha256, end_forward_time-start_forward_time]            
            outputs = out.get("outputs")    
            
            # rand_depths_lst = batch.get("rand_depths_lst")[0] # shape: [N, 1, H, W]
            # rand_img_lst = batch.get("rand_img_lst")[0] # shape: [N, 3, H, W]
            # rand_extrinsics_lst = batch.get("rand_extrinsics_lst")[0] # shape: [N, 4, 4]
            # rand_intrinsics_lst = batch.get("rand_intrinsics_lst")[0] # shape: [N, 3, 3] 
            
            rand_depths_lst = batch.get("nv_depths_lst")[0] # shape: [N, 1, H, W]
            rand_img_lst = batch.get("nv_img_lst")[0] # shape: [N, 3, H, W]
            rand_extrinsics_lst = batch.get("nv_extrinsics_lst")[0] # shape: [N, 4, 4]
            rand_intrinsics_lst = batch.get("nv_intrinsics_lst")[0] # shape: [N, 3, 3]             

            _, log_dict = self.evaluate_log(outputs[0], rand_img_lst, rand_depths_lst,
                                            rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                                            sha256, batch_idx, isCenterCrop=False,
                                            mode=f'test', test_dataloader_idx=dataloader_idx)
            # info.append(avg_psnr.detach().item())
            log_dict['sha256'] = sha256
            log_dict['forward_time'] = forward_time

            # 收集结果
            if dataloader_idx not in self.test_outputs:
                self.test_outputs[dataloader_idx] = []
            self.test_outputs[dataloader_idx].append(log_dict)
            print(f"dataloader_idx: {dataloader_idx}, sha256: {sha256}",  log_dict)
            
            if self.opt.demo:
                interp_folder = os.path.join(self.logger.log_dir, f'demo_dataloader{dataloader_idx}', f'{sha256}_interp_views')
                os.makedirs(interp_folder, exist_ok=True)
                voxel_points = batch.get("voxel_points")[0]
                center = outputs[0].get_xyz.mean(dim=0, keepdim=True)
                # center = voxel_points.mean(dim=0, keepdim=True)                
                center += torch.tensor([[-0.1,0.1,0.]]).to(center.device)
                gen_demo(outputs[0], center, self.opt.trainer.render_option, 
                        device=self.device, new_w=480, new_h=268, 
                        interp_folder=interp_folder)

            # return {"sha256": sha256, "avg_psnr": avg_psnr.detach(), "info": info}
    
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


            
# # %%
# if __name__ == '__main__':
#     import multiprocessing
#     multiprocessing.set_start_method('spawn', force=True)
#     # ======================================================
#     # 1. args & cfg
#     # ======================================================
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config_path', type=str, default='configs/Slat_vae_full/exp_04_27_pl.json',
#                         help='Path to the config file')
#     parser.add_argument('--output_dir', type=str, default=None,
#                         help='Directory to save the output')
#     parser.add_argument('--select_scene', type=str, default=None, # 'e0d1ed2159'
#                         help='the scenes to calculate')     
#     parser.add_argument('--use_depth_mask', action='store_true',
#                         help='whether to use depth mask')    
#     parser.add_argument('--waitn', type=int, default=9,
#                         help='2GPUs=18, 3GPUs=12, 4GPUs=9')
#     parser.add_argument('--test', action='store_true', help='whether to test only')
#     parser.add_argument('--demo', action='store_true', help='whether to generate demo video only')
#     args = parser.parse_args()
    
#     config = json.load(open(args.config_path, 'r'))
#     opt = edict()
#     opt.update(args.__dict__)
#     opt.update(config)
    
#     if opt.select_scene is not None and opt.output_dir is None:
#         opt.output_dir = './outputs/GSDecoder_single/' + f'{opt.select_scene}_' + opt.config_path.split('/')[-1].split('.')[0] # outputs_Slat_vae_full
#     elif opt.select_scene is None and opt.output_dir is None:
#         opt.output_dir = './outputs/GSDecoder_full/' + opt.config_path.split('/')[-1].split('.')[0] + f'_DpMask{opt.use_depth_mask}' # outputs_Slat_vae_full
#     else:
#         opt.output_dir = opt.output_dir + f'{opt.select_scene}_' + opt.config_path.split('/')[-1].split('.')[0] # outputs_Slat_vae_full
#     os.makedirs(opt.output_dir, mode=0o777, exist_ok=True)
    
#     exp_dir = opt.output_dir
#     logger_dir = opt.output_dir.replace('outputs', 'tensorboadblogs')
#     TBlogger = TensorBoardLogger(save_dir=logger_dir, name="logs", version=0)
    
#     logger.remove(handler_id=None)
#     logger.add(os.path.join(opt.output_dir, "loss.log"), level="INFO")
#     logger.info(f"opt: {opt}")
    
#     # save_config(opt, save_path=os.path.join(opt.output_dir, args.config_path.split('/')[-1]))
#     logger.info('\n\nConfig:')
#     logger.info('=' * 80)
#     json_str = json.dumps(opt.__dict__, indent=4)  # 将对象转为格式化的 JSON 字符串
#     with open(opt.output_dir+'/opt_config.json', 'w') as f: # 将 JSON 字符串保存到文件
#         f.write(json_str)
#     logger.info(json_str)
    
#     os.makedirs(os.path.join(opt.output_dir, 'train'), exist_ok=True)
#     os.makedirs(os.path.join(opt.output_dir, 'val'), exist_ok=True)
#     os.makedirs(os.path.join(opt.output_dir, 'test'), exist_ok=True)
#     # os.makedirs(os.path.join(opt.output_dir, 'GSandVoxel_feats_10_18_best'), exist_ok=True)
    
#     try:
#         saveRuntimeCode(os.path.join(opt.output_dir, 'backup'))
#     except:
#         logger.info(f'save code failed~')
    
#     if opt.img_resolution is None:
#         new_w, new_h = 1428, 798
#     else:
#         new_w, new_h = opt.img_resolution
#     voxel_size = opt.models.args.representation_config.voxel_size
    
#     # 加载 metadata
#     if os.path.exists(os.path.join(opt.dataset_dir, f'metadata_norm128_v14.csv')):
#         metadata = pd.read_csv(os.path.join(opt.dataset_dir, f'metadata_norm128_v14.csv'))
#     else:
#         raise ValueError('metadata.csv not found')
#     if opt.select_scene is not None:
#         metadata = metadata[metadata['sha256'] == opt.select_scene]
#         print(f"[INFO] selected scene is {opt.select_scene}, num_metadata: {len(metadata)}")
        
#     less_coverage_csv = pd.read_csv(os.path.join(opt.dataset_dir, opt.feats_folder_name, opt.feat_model_name, f'less_coverage_scenes_all.csv'))
#     # 根据 unconverage_ratio 进行筛选
#     threshold = 0.87
#     removed_sha256 = less_coverage_csv[less_coverage_csv['unconverage_ratio'] > threshold]['sha256'].tolist()

#     excluded_sha256 = ["1204e08f17", "30966f4c6e", "cc5237fd77"] # 验证，这两个场景很差
#     excluded_sha256 += ["edbf65c385", "684d4da1fe", "909a9ea5fc"] # OOM voxel太多
#     excluded_sha256 += ["45d2e33be1", "9471b8d485", "116456116b", "40b56bf310","2748de13fb"]
#     excluded_sha256 += ['2b5ef64cad', '0e75f3c4d9', '4c5c60fa76', '4d451d9c36', '5ea3e738c3', '5fb5d2dbf2', '7cd2ac43b4', '13c3e046d7', '036bce3393', '41b00feddb', '82ff39b7ef', '99fa5c25e1', '623c973c56', '724c40236c', '00777c41d4', '816e996553', '2489b7f4fe', '4117a4f02c', '020312de8d', '13285009a4', 'a4e227f506', 'a23f391ba9', 'a66af39cef', 'aad55d46ad', 'b37c033757', 'b24697b3a1', 'c4c04e6d6c', 'c5f701a8c7', 'c545851c4f', 'c601466b77', 'd415cc449b', 'd918af9c5f', 'e3b3b0d0c7', 'e3c39f4aea', 'e4007ff6b5', 'f248c2bcdc'] # 工厂
#     excluded_sha256 += ['0f3474b837', '4f144c7489', '5d152fab1b', '7efa3f6b1b', '9f21bdec45','108ec0b806', '893fb90e89', '3583c4c70f', '20871b98f3', '635852d56e', '4422722c49', '5654092cc2', 'a6e1a3bdda', 'abb6639ceb', 'abf29d2474', 'de3c77cecd', 'f5401524e5', 'fe1733741f'] # storage room  'ada5304e41' 'c07c8a276d' 'd3b7d054c4
#     excluded_sha256 += ['1b378f1114', '8e22c48c20', '19cfd590f4', '20ff72df6e', '40b56bf310', '45ba9ea301', '307e3262f1', '1204e08f17', '546292a9db', 'ab046f8faf', 'c4d4cb61f6', 'e3ecd49e2b', 'e909f8213d', 'eeeb9836b8', 'f2e6c43543'] # 不够32
#     excluded_sha256 += ['5dd2b849be', 'be0ed6b33c', '546292a9db', '87f6d7d564', 'ab046f8faf']
#     train_metadata = metadata[:856]
#     train_metadata = train_metadata[
#         train_metadata["sha256"].apply(
#             lambda sha256: os.path.exists(
#                 os.path.join(opt.dataset_dir, opt.feats_folder_name, opt.feat_model_name, f'{sha256}.pt')
#             )
#         )
#     ]
#     train_metadata = train_metadata[~train_metadata["sha256"].isin(removed_sha256)]
#     train_metadata = train_metadata[~train_metadata["sha256"].isin(excluded_sha256)]
#     train_metadata = train_metadata[train_metadata['num_voxels_norm128_v14_fix32'] <= 80_000]
#     # train_metadata = _filter_dataset(train_metadata, num_voxels_threshold=60_000) # 70_000  metadata[:856]  metadata[:230]
#     train_metadata = _filter_dataset_new(train_metadata) # 70_000  metadata[:856]  metadata[:230]
    
#     train_metadata.to_csv("train_metadata.csv", index=False)
    
#     val_metadata = metadata[856:]
#     val_metadata = val_metadata[~val_metadata["sha256"].isin(removed_sha256)]
#     val_metadata = val_metadata[~val_metadata["sha256"].isin(excluded_sha256)]
#     val_metadata = _filter_dataset(metadata[856:], num_voxels_threshold=60_000)   # 70_000  metadata[856:]  metadata[230:]
    
#     val_metadata.to_csv("val_metadata.csv", index=False)
    
#     logger.info('\n\n'+'=='*60)
#     logger.info(f"[INFO] metadata num: {len(metadata)}, filtered metadata num: {len(train_metadata) + len(val_metadata)}, train_metadata num: {len(train_metadata)}, val_metadata num: {len(val_metadata)}")
#     logger.info('=='*60 + '\n\n')
    
#     if opt.select_scene is None:
#         train_dataset = DataClass(train_metadata, opt, stage='train')
#         val_dataset = DataClass(val_metadata, opt, stage='val')
#         train_loader = DataLoader(train_dataset, batch_size=opt.trainer.batch_size, 
#                                   shuffle=True, num_workers=6, pin_memory=True, 
#                                   persistent_workers=True,  collate_fn=collate_skip_none)
#         val_loader = DataLoader(val_dataset, batch_size=opt.trainer.batch_size, 
#                                 shuffle=False, num_workers=4, pin_memory=True, 
#                                 persistent_workers=True )
#         if opt.test:
#             test_loaders = []
#             metadatas = [val_metadata, ] # val_metadata,  train_metadata, val_metadata, val_metadata
#             stages = ['test_val_nv', ] # 'test_val_sv',  'test_train_nv', 'test_val_nv_testframe', 'test_val_sv'
#             for idx, (test_metadata, stage) in enumerate(zip(metadatas, stages)):
#                 test_dataset = DataClass(test_metadata, opt, stage=stage)
#                 test_loader = DataLoader(test_dataset, batch_size=opt.trainer.batch_size, 
#                                         shuffle=False, num_workers=4, pin_memory=True, 
#                                         persistent_workers=True )
#                 test_loaders.append(test_loader)
#                 logger.info(f"Test Dataset [{idx}] contains {len(test_loader.dataset)}")
#                 os.makedirs(os.path.join(opt.output_dir, 'test', str(idx)), exist_ok=True)

#     else:
#         train_dataset = DataClass(metadata, opt, stage='train')
#         train_loader = DataLoader(train_dataset, batch_size=opt.trainer.batch_size, 
#                                   shuffle=True, num_workers=4, pin_memory=True, 
#                                   persistent_workers=True,  collate_fn=collate_skip_none)
#         val_dataset = DataClass(metadata, opt, stage='val')
#         val_loader = DataLoader(val_dataset, batch_size=opt.trainer.batch_size, 
#                                   shuffle=True, num_workers=4, pin_memory=True, 
#                                   persistent_workers=True,  collate_fn=collate_skip_none)
#         if opt.test:
#             test_datasets = []
#             # train scenes with novel views
#             test_metadata_0 = metadata
#             test_dataset_0 = DataClass(metadata, opt, stage='test_train_novel')
#             test_datasets.append(test_dataset_0)
#             # val scenes with novel views
#             test_metadata_1 = val_metadata
#             test_dataset_1 = DataClass(metadata, opt, stage='test_val_novel')
#             test_datasets.append(test_dataset_1)
            
#             test_loaders = []
#             metadatas = [val_metadata, train_metadata, val_metadata]
#             stages = ['test_val_sv', 'test_train_nv', 'test_val_nv']
#             for idx, (test_metadata, stage) in enumerate(zip(metadatas, stages)):
#                 test_dataset = DataClass(test_metadata, opt, stage=stage)
#                 test_loader = DataLoader(test_dataset, batch_size=opt.trainer.batch_size, 
#                                         shuffle=False, num_workers=4, pin_memory=True, 
#                                         persistent_workers=True )
#                 test_loaders.append(test_loader)
#                 logger.info(f"Test Dataset [{idx}] contains {len(test_loader.dataset)}")
#                 os.makedirs(os.path.join(opt.output_dir, 'test', str(idx)), exist_ok=True)
            
#     # 创建两个 ModelCheckpoint 回调：
#     # 1. 每隔 100 个 epoch 保存一次 checkpoint（保存所有，不仅仅最佳）
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=opt.output_dir,
#         filename="epoch{epoch:02d}",
#         save_top_k=-1, 
#         every_n_epochs=opt.trainer.save_img_every_epoch,   
#         verbose=True
#     )

#     # 2. 根据验证指标（这里选择 val_iou，指标越大越好）保存最佳模型
#     best_checkpoint_callback = ModelCheckpoint(
#         monitor="val/avg_psnr",
#         dirpath=opt.output_dir,
#         filename="best-{epoch:02d}-{val_psnr:.2f}", 
#         save_top_k=1,
#         save_last=True,
#         mode="max",            
#         verbose=True
#     )
    
#     # 检查是否有 checkpoint 文件，如果有则从 checkpoint 恢复训练
#     ckpt_list = glob.glob(os.path.join(opt.output_dir, "best-epoch=*.ckpt"))
#     if not opt.test and os.path.exists(os.path.join(opt.output_dir, "last.ckpt")):
#         resume_from_checkpoint = os.path.join(opt.output_dir, "last.ckpt")
#         logger.info(f"INFO] Resuming from checkpoint: {resume_from_checkpoint if resume_from_checkpoint else 'None'}")    
#     if not opt.test and ckpt_list:
#         resume_from_checkpoint = max(ckpt_list, key=os.path.getmtime)
#         logger.info(f"INFO] Resuming from checkpoint: {resume_from_checkpoint if resume_from_checkpoint else 'None'}")      
#     elif opt.test and ckpt_list:
#         if ckpt_list:
#             resume_from_checkpoint = max(ckpt_list, key=os.path.getmtime) # 如果有多个，可以取最新修改时间的
#         else:
#             raise ValueError('[ERROR] No checkpoint found for testing!')
#         # resume_from_checkpoint = os.path.join(opt.output_dir, "last.ckpt")
#         logger.info(f"INFO] Resuming from checkpoint: {resume_from_checkpoint if resume_from_checkpoint else 'None'}")
#     elif opt.ckpt and os.path.exists(opt.ckpt):
#         resume_from_checkpoint = opt.ckpt    
#         print('[INFO] resume_from_checkpoint' )
#         # =========================修改ckpt Tmax===================================
#         # ckpt = torch.load(resume_from_checkpoint)
#         # ckpt['lr_schedulers'][0]['T_max'] = 20
#         # torch.save(ckpt, resume_from_checkpoint)
#         # print('[INFO] Edit LR_scheduler in CKPT!!!!')
#         # =========================修改ckpt Tmax===================================
#         logger.info(f"[INFO] Resuming from checkpoint: {resume_from_checkpoint if resume_from_checkpoint else 'None'}")  
#     else:
#         resume_from_checkpoint = None
#         logger.info('INFO] No checkpoint is resumed. Training from start!')
        
#     # 创建 Trainer，并开启多 GPU 训练（例如使用所有 GPU，分布式训练 backend 为 ddp）
#     from pytorch_lightning.strategies import DDPStrategy
#     callbacks = [best_checkpoint_callback] if opt.select_scene is None else [checkpoint_callback]
#     trainer = Trainer(
#         max_epochs=int(opt.trainer.max_epochs),
#         # precision="bf16-mixed", # 混合精度计算
#         accelerator='gpu',  # 使用 GPU 训练
#         devices=-1,         # 使用所有可见 GPU
#         strategy="ddp",
#         # strategy=DDPStrategy(find_unused_parameters=True),
#         detect_anomaly=True,
#         callbacks=callbacks, #, grad_norm_callback], #ema_callback],
#         logger=TBlogger,
#         log_every_n_steps=opt.trainer.save_img_every_epoch,
#         check_val_every_n_epoch=opt.trainer.save_img_every_epoch,
#         num_sanity_val_steps=0,
#         accumulate_grad_batches=opt.waitn,
#         gradient_clip_val=opt.trainer.gradient_clip_val,
#         # log_grad_norm='global',        # 关键设置
#         limit_val_batches=1.0 if opt.select_scene is None else 0,  # 如果是验证集，则不限制验证批次
#     )

#     # os.makedirs(os.path.join(opt.output_dir, 'latents_train'), exist_ok=True)
#     # os.makedirs(os.path.join(opt.output_dir, 'latents_val'), exist_ok=True)
    
#     # print_memory_usage("*** 训练:  训练开始前")  # 训练前显存占用
#     if resume_from_checkpoint is None:
#         print('[INFO] loading from scrach!')
#         model = GSDecoderModel(opt)
#     else:
#         print(f"[INFO] load model from ckpt: {resume_from_checkpoint}")
#         # model = GSDecoderModel.load_from_checkpoint(resume_from_checkpoint, opt=opt)
#         model = GSDecoderModel(opt)
#         state_dict = torch.load(resume_from_checkpoint, map_location='cpu')[0]
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if 'lpips' not in k:
#                 new_state_dict[k.replace('module.model.', '')] = v
            
#         model.model.load_state_dict(new_state_dict)
        
#     if opt.test:
#         trainer.test(model, dataloaders=test_loaders)  # 或者 trainer.predict() , test_train_novel_loader
#     else:
#         trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from_checkpoint)

#         final_ckpt_path = os.path.join(opt.output_dir, "final_checkpoint.ckpt")
#         trainer.save_checkpoint(final_ckpt_path)
#         logger.info(f"Final checkpoint saved at: {final_ckpt_path}")
#         logger.info(f"Best checkpoint saved at: {best_checkpoint_callback.best_model_path}")
