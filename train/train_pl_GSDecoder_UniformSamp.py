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
# torch.backends.cuda.preferred_linalg_library('magma') # magma
from torchvision.utils import save_image
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from loguru import logger
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
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


import src.modules.sparse as sp
from src.representations.gaussian import Gaussian
from src.utils.demo_utils import gen_demo
from src.utils.train_utils import batch_project_and_aggregate_chunkwise, _filter_dataset_new, _filter_dataset, get_latest_ckpt, seed_everything, saveRuntimeCode

import torch.multiprocessing as mp
# mp.set_start_method('spawn')

seed_everything(520)

to_pil = transforms.ToPILImage()

torch.set_grad_enabled(True)

# transform = transforms.Compose([
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
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


# %%
# ---------------------------
# 1. 定义 Dataset
# ---------------------------
from train.datasets import GSDecoderDataset as dataset
# ---------------------------
# 2. 封装模型到 LightningModule
# ---------------------------
from train.baseModel import BaseModel
class GSDecoderModel(BaseModel):
    def __init__(self, opt, if_norm_depth=True):
        super(GSDecoderModel).__init__(opt)
        self.opt = opt
        self.sh_degree = opt.models.args.representation_config.sh_degree if hasattr(opt.models.args.representation_config, 'sh_degree') else 0
        self.rgb_dim = 3 * ((self.sh_degree + 1) **2)
        self.voxel_align_ratio = opt.models.voxel_align_ratio if hasattr(opt.models, 'voxel_align_ratio') else 5
        # self.if_fix_xyz = opt.models.if_fix_xyz if hasattr(opt.models, 'if_fix_xyz') else False
        self.num_GS = opt.models.args.representation_config['num_gaussians']
        self.warmup_epochs = self.opt.trainer.lr_scheduler.args.warmup_steps

        self.vs_size = 2/128
        
        self.if_norm_depth = if_norm_depth
        # 指定在哪些 epoch 做 inference
        self.infer_epochs = [59, 119, 199] + list(range(199, self.opt.trainer.max_epochs, 200))

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
        rand_depths_lst = batch.get("rand_depths_lst")[0].unsqueeze(1)  # shape: [N, 1, H, W]
        T_norm = batch.get("T_norm")[0].to("cpu")  # 显式加默认值
        
        grid_origin = None # 如果使用了voxel_points, 就不需要grid_origin来反算xyz了
        # bs = feats.shape[0] # 这里我希望bs=1, 因为trellis那个sparsetensor之后的处理没有考虑多batch的操作
        assert len(rand_img_lst) == len(rand_extrinsics_lst) == len(rand_intrinsics_lst) == 1
        
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

        return {"sha256": sha256, 
                "outputs": outputs, 
                # "rand_img_lst": rand_img_lst, 
                # "rand_depths_lst": rand_depths_lst,
                # "rand_extrinsics_lst": rand_extrinsics_lst,
                # "rand_intrinsics_lst": rand_intrinsics_lst
                }
    
    def compute_log_loss(self, outputs, 
                         rand_img_lst, rand_depths_lst, 
                         rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                         sha256, batch_idx, mode=f'train'):
        
        if 'test' in mode:
            log = []
        
        depth_max = rand_depths_lst.max().item()
        new_w, new_h = self.opt.img_resolution
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
        
        assert isinstance(outputs, list) and len(outputs) == 1
        
        all_train_videos, all_train_depths = self.rendering(outputs[0], rand_extrinsics_lst, rand_intrinsics_lst,
                                                            SCALE=SCALE, 
                                                            invalid_mask=invalid_mask, 
                                                            depth_max=depth_max,
                                                            batch_idx=batch_idx, 
                                                            sha256=sha256, 
                                                            render_mode='RGB+D', 
                                                            mode=mode)
        
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
            f"{mode}/loss_depth_avg": avg_loss_depth.item()
            f"{mode}/avg_psnr": avg_psnr.item()
        }
            
        if hasattr(self.opt.trainer, "loss_weight_GSdispersion") and self.opt.trainer.loss_weight_GSdispersion > 0:
            log_dict[f"{mode}/loss_gs_dispersion"] = loss_gs_dispersion.item()
        if hasattr(self.opt.trainer, "loss_weight_GSdiversity") and self.opt.trainer.loss_weight_GSdiversity > 0:
            log_dict[f"{mode}/loss_gs_diversity"] = loss_gs_diversity.item()

        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False if mode == 'train' else True)
        # logger.info(f"[{tag}] epoch: {self.trainer.current_epoch}, sha256: {sha256}, {mode}_log_dict = " + json.dumps(log_dict, indent=2))
        
        # ------------------------- 可视化 -------------------------
        train_save_trigger = (mode == 'train') and (self.trainer.current_epoch % self.opt.trainer.save_img_every_epoch == 0)
        if train_save_trigger or (mode == 'val') or ('test' in mode):
            
            post_fix = 'epoch0_' if self.trainer.current_epoch == 0 else '_'

            save_image(all_train_videos, os.path.join(self.opt.output_dir, mode,
                        f"{sha256}_{mode}_{post_fix}rgb_image.png")) # epoch_{self.trainer.current_epoch}
            
            save_image(rand_img_lst, os.path.join(self.opt.output_dir, mode,
                    f"{sha256}_{mode}_{post_fix}GT_rgb_image.png")) # epoch_{self.trainer.current_epoch}     
            
            save_image(all_train_depths, os.path.join(self.opt.output_dir, mode,
                        f"{sha256}_{mode}_{post_fix}depth.png")) # epoch_{self.trainer.current_epoch}
            
            save_image(rand_depths_lst.unsqueeze(1), os.path.join(self.opt.output_dir, mode,
                    f"{sha256}_{mode}_{post_fix}GT_depth.png")) # epoch_{self.trainer.current_epoch}
            
            if isinstance(outputs, list) and len(outputs) == 1:
                gausssian = outputs[0]
            elif isinstance(outputs, Gaussian):
                gausssian = outputs
            else:
                raise NotImplementedError(f"Unsupported output type for visualization: {type(outputs)}")
            
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
            GS_savepath = os.path.join(self.opt.output_dir, tag.lower(), f'{sha256}_{mode}_{post_fix}_decoded_GS.npz') 
            np.savez(GS_savepath, **dict_GS)

            gausssian.save_ply(os.path.join(self.opt.output_dir, tag.lower(),
                    f'{sha256}_{mode}_{post_fix}_decoded_GS.ply')) # epoch_{self.trainer.current_epoch}

        return loss_total, avg_psnr
    
    def training_step(self, batch, batch_idx):
        # torch.autograd.set_detect_anomaly(True)
        self.model.train()
        sha256 = out.get("sha256")[0]
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
        sha256 = out.get("sha256")[0]
        SCALE = batch.get("scale").item()
        
        self.clean_mem()
        
        with torch.no_grad():
            
            start_forward_time = time.time()
            out = self.forward(batch)
            end_forward_time = time.time()

            
            info = [sha256, end_forward_time-start_forward_time]            
            outputs = out.get("outputs")    
            
            rand_depths_lst = batch.get("rand_depths_lst")[0] # shape: [N, 1, H, W]
            rand_img_lst = batch.get("rand_img_lst")[0] # shape: [N, 3, H, W]
            rand_extrinsics_lst = batch.get("rand_extrinsics_lst")[0] # shape: [N, 4, 4]
            rand_intrinsics_lst = batch.get("rand_intrinsics_lst")[0] # shape: [N, 3, 3] 
            
            if hasattr(batch, "novel_extrinsics_lst") and hasattr(batch, "novel_img_lst") and hasattr(batch, "novel_intrinsics_lst"):
                rand_extrinsics_lst = batch.get("novel_extrinsics_lst")
                rand_img_lst = batch.get("novel_img_lst")
                rand_intrinsics_lst = batch.get("novel_intrinsics_lst")

            avg_psnr = self.compute_log_loss(outputs, rand_img_lst, rand_depths_lst,
                                                    rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                                                    batch_idx, sha256, mode=f'test_{dataloader_idx}')
            info.append(avg_psnr.detach().item())
            
            # 收集结果
            if dataloader_idx not in self.test_outputs:
                self.test_outputs[dataloader_idx] = []
            
            self.test_outputs[dataloader_idx].append({"scene": sha256, "avg_psnr": avg_psnr.detach().item()})
            print(f"dataloader_idx: {dataloader_idx}, scene: {sha256}, test avg_psnr: {avg_psnr.detach().item()}, info: {info}")
            
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

            return {"sha256": sha256, "avg_psnr": avg_psnr.detach(), "info": info}
    
    def on_test_epoch_end(self):
        summary = []
        for idx, results in self.test_outputs.items():
            for x in results:
                print(results)
                print("x", x)
                summary.append(x['info'])
            avg_psnr = torch.stack([x["avg_psnr"] for x in results]).mean()
            num_scenes = len(results)
            print(f"[Prediction dataload_idx {idx}] num_scenes={num_scenes:.4f}, avg_psnr={avg_psnr:.4f}")
        with open(os.path.join(self.logger.log_dir, 'test_summary.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(summary)  # 将整个二维 list 存储为多行
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
        opt.output_dir = './outputs/GSDecoder_single/' + f'{opt.select_scene}_' + opt.config_path.split('/')[-1].split('.')[0] # outputs_Slat_vae_full
    elif opt.select_scene is None and opt.output_dir is None:
        opt.output_dir = './outputs/GSDecoder_full/' + opt.config_path.split('/')[-1].split('.')[0] + f'_DpMask{opt.use_depth_mask}' # outputs_Slat_vae_full
    else:
        opt.output_dir = opt.output_dir + f'{opt.select_scene}_' + opt.config_path.split('/')[-1].split('.')[0] # outputs_Slat_vae_full
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
    val_metadata = val_metadata[~val_metadata["sha256"].isin(removed_sha256)]
    val_metadata = val_metadata[~val_metadata["sha256"].isin(excluded_sha256)]
    val_metadata = _filter_dataset(metadata[856:], num_voxels_threshold=60_000)   # 70_000  metadata[856:]  metadata[230:]
                        
    logger.info('\n\n'+'=='*60)
    logger.info(f"[INFO] metadata num: {len(metadata)}, filtered metadata num: {len(train_metadata) + len(val_metadata)}, train_metadata num: {len(train_metadata)}, val_metadata num: {len(val_metadata)}")
    logger.info('=='*60 + '\n\n')
    
    if opt.select_scene is None:
        train_dataset = datasets(train_metadata, opt, stage='train')
        val_dataset = datasets(val_metadata, opt, stage='val')
        train_loader = DataLoader(train_dataset, batch_size=opt.trainer.batch_size, 
                                  shuffle=True, num_workers=32, pin_memory=True, 
                                  persistent_workers=True,  collate_fn=collate_skip_none)
        val_loader = DataLoader(val_dataset, batch_size=opt.trainer.batch_size, 
                                shuffle=False, num_workers=16, pin_memory=True, 
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
                                  shuffle=True, num_workers=2, pin_memory=True, 
                                  persistent_workers=True,  collate_fn=collate_skip_none)
        val_loader = None
    
      
    # 创建两个 ModelCheckpoint 回调：
    # 1. 每隔 100 个 epoch 保存一次 checkpoint（保存所有，不仅仅最佳）
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.output_dir,
        filename="epoch{epoch:02d}",
        save_top_k=-1,            # 保存所有符合条件的 checkpoint
        every_n_epochs=opt.trainer.save_img_every_epoch,   
        verbose=True
    )

    # 2. 根据验证指标（这里选择 val_iou，指标越大越好）保存最佳模型
    best_checkpoint_callback = ModelCheckpoint(
        monitor="val/avg_psnr",
        dirpath=opt.output_dir,
        filename="best-{epoch:02d}-{val_psnr:.2f}", 
        save_top_k=1,
        save_last=True,
        mode="max",            
        verbose=True
    )
    
    # 检查是否有 checkpoint 文件，如果有则从 checkpoint 恢复训练
    if not opt.test and os.path.exists(os.path.join(opt.output_dir, "last.ckpt")):
        resume_from_checkpoint = os.path.join(opt.output_dir, "last.ckpt")
        logger.info(f"INFO] Resuming from checkpoint: {resume_from_checkpoint if resume_from_checkpoint else 'None'}")      
    elif opt.test:
        ckpt_list = glob.glob(os.path.join(opt.output_dir, "best-epoch=*.ckpt"))
        if ckpt_list:
            resume_from_checkpoint = max(ckpt_list, key=os.path.getmtime) # 如果有多个，可以取最新修改时间的
        # resume_from_checkpoint = os.path.join(opt.output_dir, "last.ckpt")
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
        trainer.test(model, dataloaders=[test_loader], ckpt_path=resume_from_checkpoint)  # 或者 trainer.predict() , test_train_novel_loader
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from_checkpoint)

    final_ckpt_path = os.path.join(opt.output_dir, "final_checkpoint.ckpt")
    trainer.save_checkpoint(final_ckpt_path)
    logger.info(f"Final checkpoint saved at: {final_ckpt_path}")
    logger.info(f"Best checkpoint saved at: {best_checkpoint_callback.best_model_path}")
