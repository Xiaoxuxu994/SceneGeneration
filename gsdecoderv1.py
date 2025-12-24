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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch_npu
from torchvision.utils import save_image
import csv
import torch.nn.functional as F
from matplotlib import pyplot as plt


# import src.modules.sparse as sp
from src.representations.gaussian import Gaussian
from src.utils.demo_utils import gen_demo
from src.models.GSdecoder.decoder_RGBcnnCamDepthV2 import DecoderOnlyRGBCNNCamDepthInV2

# SAVE_ROOT = '/efs-mi-east4-2/datasets/scannetpp_v2_opencv_pinhole/GSandVoxel_feats_normV14/GSandVoxel_feats_11_13_32views_final'
# SAVE_ROOT = None


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




@torch.inference_mode()
def depth_to_world(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor):
    '''
    Project depth maps to world coordinates.
    Args:
        depth: (V, H, W) depth maps
        intrinsics: (V, 3, 3) normalized camera intrinsics
        extrinsics: (V, 4, 4) world to camera extrinsics, rotations may contain scaling
    Returns:
        points: (N, 3) world coordinates of valid depth pixels
    '''
    V, H, W = depth.shape
    x = torch.arange(W, device=depth.device, dtype=depth.dtype) / W
    y = torch.arange(H, device=depth.device, dtype=depth.dtype) / H
    rays = depth.new_ones(V, H, W, 4)
    rays[..., 0] = (x - intrinsics[:, :1, 2:]) / intrinsics[:, :1, :1]
    rays[..., 1] = (y.view(H, 1) - intrinsics[:, 1:2, 2:]) / intrinsics[:, 1:2, 1:2]
    rays[..., :3] *= depth[..., None]
    # points = (rays.flatten(1, 2) - extrinsics[:, None, :3, 3]) @ extrinsics[:, :3, :3]  # only applicable to orthonormal rotation matrix with RR^T=I
    points = torch.einsum('vmn,vhwn->vhwm', extrinsics.float().inverse().to(rays.dtype)[:, :3], rays)
    return points[depth > 1e-4]


@torch.inference_mode()
def voxel_downsample(points: torch.Tensor, voxel_size: float):
    '''
    Voxel based point downsampling.
    Args:
        points: (N, 3) input points
        voxel_size: float
    Returns:
        points: (M, 3) downsampled points
    '''
    indices = (points / voxel_size).round().int().unique(dim=0)
    return indices.to(points.dtype) * voxel_size


@torch.inference_mode()
def tsdf_fusion(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor, voxel_size: float,
                voxel_range: tuple[float, float] = (-.25, 2.25), trunc: float = None, thresh: float = None):
    '''
    TSDF fusion based point sampling.
    Args:
        depth: (V, H, W) depth maps
        intrinsics: (V, 3, 3) normalized (0 < uv < 1) camera intrinsics
        extrinsics: (V, 4, 4) world to camera extrinsics, rotations may contain scaling
        voxel_size: float
        voxel_range: (min, max) tuple of voxel grid along each axis
        trunc: float, truncation distance
        thresh: float, threshold for selecting points near surface
    Returns:
        points: (M, 3) sampled points near surface
    '''
    if trunc is None: trunc = 8 * voxel_size
    if thresh is None: thresh = voxel_size
    V, H, W = depth.shape
    grid = torch.arange(voxel_range[0], voxel_range[1], voxel_size, dtype=depth.dtype, device=depth.device)
    grid = torch.stack(torch.meshgrid(grid, grid, grid, indexing='ij')).flatten(1)
    coords = intrinsics @ (extrinsics[:, :3, :3] @ grid + extrinsics[:, :3, 3:])
    coords[:, :2] /= coords[:, 2:]
    x, y, z = (W * coords[:, 0].clamp(0, 1)).floor().int(), (H * coords[:, 1].clamp(0, 1)).floor().int(), coords[:, 2]
    dist = depth[torch.arange(V, device=depth.device)[:, None].expand_as(z), y, x] - z
    dist[(coords[:, :2] < 0).any(1) | (coords[:, :2] > 1).any(1) | (z < trunc) | (dist.abs() > trunc)] = torch.nan
    return grid[:, dist.nanmean(0).abs() < thresh].T


from train.baseModel import BaseModel
class GSDecoderModel(BaseModel):
    def __init__(self, opt, if_norm_depth=True, load_pretrained=None):
        super(GSDecoderModel, self).__init__(opt)
        self.vs_size = .01
        self.if_norm_depth = if_norm_depth
        # 指定在哪些 epoch 做 inference
        self.infer_epochs = [59, 119, 199] + list(range(199, self.opt.trainer.max_epochs, 200))
        self.SAVE_ROOT = opt.voxelGS_feats_dir if hasattr(opt, 'voxelGS_feats_dir') else None
        self.if_fix_xyz = False
        self.sampler = opt.get("anchor_sampler", "tsdf")        
        self.model: DecoderOnlyRGBCNNCamDepthInV2

        if load_pretrained:
            ckpt = torch.load(load_pretrained, map_location='cpu')
            featEnc_weights = {k.replace('model.featEnc.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('model.featEnc')}
            self.model.featEnc.load_state_dict(featEnc_weights)
    
    def compute_log_loss(self, outputs, 
                         rand_img_lst, rand_depths_lst, invalid_mask,
                         rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                         sha256, mode='train'):
        
        depth_max = rand_depths_lst.max()
        # new_w, new_h = self.opt.img_resolution
        
        new_w, new_h = self.opt.trainer.rend_sv_res
        
        rand_img_lst = F.interpolate(rand_img_lst, size=(new_h, new_w), mode='area')
        rand_depths_lst = F.interpolate(rand_depths_lst.unsqueeze(1), size=(new_h, new_w), mode='nearest').squeeze(1)
        
        # print(f"rendering: {rand_depths_lst.shape, rand_img_lst.shape}")
        
        # ------------------------- 预处理 -------------------------
        rand_intrinsics_lst[:, 0, :] *= new_w
        rand_intrinsics_lst[:, 1, :] *= new_h

        # 归一化depth
        if self.if_norm_depth:
            rand_depths_lst = rand_depths_lst / depth_max
        
        # maskr住无效深度的gt 图像区域
        # if self.opt.use_depth_mask:
        #     invalid_mask = rand_depths_lst <= 1e-6
            # rand_img_lst = rand_img_lst.masked_fill(invalid_mask.unsqueeze(1), 0)
        
        # print(f"rendering invalid_mask: {invalid_mask.shape, rand_img_lst.shape}")
        
        assert isinstance(outputs, list) and len(outputs) == 1
        
        # ------------------------- 渲染 -------------------------
        all_train_videos, all_train_depths = self.rendering(outputs[0], rand_extrinsics_lst, rand_intrinsics_lst,
                                                            SCALE=SCALE, 
                                                            # invalid_mask=invalid_mask, 
                                                            # depth_max=depth_max,
                                                            # batch_idx=batch_idx, 
                                                            sha256=sha256, 
                                                            render_mode='RGB+D', 
                                                            img_res=(new_w, new_h),
                                                            mode=mode)
        
        all_train_depths = all_train_depths / SCALE # [1, H, W]
        if self.if_norm_depth:
            all_train_depths = all_train_depths / depth_max
        
        # print(f'scale: {SCALE}',
        #       f'all_train_depths: {all_train_depths.shape, all_train_depths.min(), all_train_depths.max()}', 
        #       f'rand_depths_lst: {rand_depths_lst.shape, rand_depths_lst.min(), rand_depths_lst.max()}')
        
        # ------------------------- 可视化 -------------------------
        train_save_trigger = (mode == 'train') and (self.current_epoch % self.opt.trainer.save_img_every_epoch == 0)
        scene_id = sha256[0] if isinstance(sha256, list) else str(sha256.item())
        if train_save_trigger or (mode == 'val') or ('test' in mode):
            if self.SAVE_ROOT is None:  # validation or test
                post_fix = 'epoch0_' if self.current_epoch == 0 else '_'
                save_image(all_train_videos, os.path.join(self.opt.output_dir, mode,
                            f"{scene_id}_{mode}_{post_fix}rgb_image.png")) # epoch_{self.trainer.current_epoch}
                save_image(rand_img_lst, os.path.join(self.opt.output_dir, mode,
                        f"{scene_id}_{mode}_{post_fix}GT_rgb_image.png")) # epoch_{self.trainer.current_epoch}     
                save_image(all_train_depths.unsqueeze(1), os.path.join(self.opt.output_dir, mode,
                            f"{scene_id}_{mode}_{post_fix}depth.png")) # epoch_{self.trainer.current_epoch}
                save_image(rand_depths_lst.unsqueeze(1), os.path.join(self.opt.output_dir, mode,
                        f"{scene_id}_{mode}_{post_fix}GT_depth.png")) # epoch_{self.trainer.current_epoch}
            else:  # save features for refiner training
                if isinstance(outputs, list) and len(outputs) == 1: gausssian = outputs[0]
                elif isinstance(outputs, Gaussian): gausssian = outputs
                else: raise NotImplementedError(f"Unsupported output type for visualization: {type(outputs)}")
                feat_GS = torch.cat([
                    gausssian.get_xyz,
                    gausssian.get_opacity,
                    gausssian.get_scaling,
                    gausssian.get_rotation,
                    gausssian.get_features.reshape(-1, self.rgb_dim),
                ], -1).cpu().numpy()

                GS_savepath = os.path.join(f'{self.SAVE_ROOT}/{scene_id}_decOnly_train_dict_GS.npz')             
                print('saved feat GS in', GS_savepath)
                np.savez(GS_savepath, feat_GS=feat_GS)
            # dict_GS = {'feat_GS': feat_GS}
            # GS_savepath = os.path.join(self.opt.output_dir, tag.lower(), f'{sha256}_{mode}_{post_fix}_decoded_GS.npz') 
            # np.savez(GS_savepath, **dict_GS)

            # gausssian.save_ply(os.path.join(self.opt.output_dir, tag.lower(),
                    # f'{sha256}_{mode}_{post_fix}_decoded_GS.ply')) # epoch_{self.trainer.current_epoch}
        
        # ------------------------- 计算指标 -------------------------         
        # maskr住无效深度的gt图像和depth区域
        if self.opt.use_depth_mask:
            rand_img_lst = rand_img_lst.masked_fill(invalid_mask.unsqueeze(1), 0)
            all_train_videos = all_train_videos.masked_fill(invalid_mask.unsqueeze(1), 0)
            rand_depths_lst = rand_depths_lst.masked_fill(invalid_mask, 1e-6)  
            all_train_depths = all_train_depths.masked_fill(invalid_mask, 1e-6)  
        # print(f'all_train_videos, rand_img_lst: {all_train_videos.shape, rand_img_lst.shape}')
        avg_loss_rec = self.compute_loss_rgb(all_train_videos, rand_img_lst)
        avg_loss_depth = self.compute_loss_depth(all_train_depths, rand_depths_lst)
        loss_opa, loss_scale, loss_gs_dispersion, loss_gs_diversity = self.compute_loss_gs(outputs)
        
        with torch.inference_mode(): avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
        
        # ------------------------- 总 loss -------------------------
        loss_total = avg_loss_rec + loss_opa + loss_scale + avg_loss_depth + loss_gs_dispersion + loss_gs_diversity

        # ------------------------- 日志 -------------------------
        log_dict = {
            # "lr": self.trainer.optimizers[0].param_groups[0]['lr'],
            f"{mode}/loss_total": loss_total,
            f"{mode}/loss_rec_avg": avg_loss_rec,
            f"{mode}/loss_opa": loss_opa,
            f"{mode}/loss_scale": loss_scale,
            f"{mode}/loss_depth_avg": avg_loss_depth,
            f"{mode}/avg_psnr": avg_psnr
        }
            
        if hasattr(self.opt.trainer, "loss_weight_GSdispersion") and self.opt.trainer.loss_weight_GSdispersion > 0:
            log_dict[f"{mode}/loss_gs_dispersion"] = loss_gs_dispersion
        if hasattr(self.opt.trainer, "loss_weight_GSdiversity") and self.opt.trainer.loss_weight_GSdiversity > 0:
            log_dict[f"{mode}/loss_gs_diversity"] = loss_gs_diversity

        # self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False if mode == 'train' else True)
        # logger.info(f"[{tag}] epoch: {self.trainer.current_epoch}, sha256: {sha256}, {mode}_log_dict = " + json.dumps(log_dict, indent=2))

        return loss_total, log_dict

    def evaluate_log(self, outputs, rand_img_lst, rand_depths_lst, 
                    rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                    sha256, batch_idx, mode='test', test_dataloader_idx=0):
        
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
        else: invalid_mask = None
        
        assert isinstance(outputs, list) and len(outputs) == 1
        
        all_train_videos, all_train_depths = self.rendering(outputs[0], rand_extrinsics_lst, rand_intrinsics_lst,
                                                            SCALE=SCALE, 
                                                            # invalid_mask=invalid_mask, 
                                                            depth_max=depth_max,
                                                            batch_idx=batch_idx, 
                                                            sha256=sha256, 
                                                            render_mode='RGB+D', 
                                                            mode=mode)
        
        img_rec_l1, img_rec_ssim, img_rec_lpips, avg_psnr = self.evaluate_rgb(all_train_videos, rand_img_lst)

        # ------------------------- 日志 -------------------------
        log_dict = {
            # "lr": self.trainer.optimizers[0].param_groups[0]['lr'],
            f"{mode}/img_rec_l1": img_rec_l1,
            f"{mode}/img_rec_ssim": img_rec_ssim,
            f"{mode}/img_rec_lpips": img_rec_lpips,
            f"{mode}/avg_psnr": avg_psnr,
        }

        # ------------------------- 可视化 -------------------------            
        # post_fix = 'epoch0_' if self.trainer.current_epoch == 0 else '_'
        post_fix = ''
        scene_id = sha256[0] if isinstance(sha256, list) else str(sha256.item())
        save_image(all_train_videos, os.path.join(self.opt.output_dir, mode, test_dataloader_idx,
                    f"{scene_id}_{mode}_{post_fix}rgb_image.png")) # epoch_{self.trainer.current_epoch}
        
        save_image(rand_img_lst, os.path.join(self.opt.output_dir, mode, test_dataloader_idx,
                f"{scene_id}_{mode}_{post_fix}GT_rgb_image.png")) # epoch_{self.trainer.current_epoch}     
        
        save_image(all_train_depths, os.path.join(self.opt.output_dir, mode, test_dataloader_idx,
                    f"{scene_id}_{mode}_{post_fix}depth.png")) # epoch_{self.trainer.current_epoch}
        
        save_image(rand_depths_lst.unsqueeze(1), os.path.join(self.opt.output_dir, mode, test_dataloader_idx,
                f"{scene_id}_{mode}_{post_fix}GT_depth.png")) # epoch_{self.trainer.current_epoch}
        
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
        gs_fdc = gausssian.get_features.reshape(-1, self.rgb_dim) # [N, 3*(sh+1)**2]
        feat_GS = torch.cat([
                                gs_xyz,
                                gs_opa,
                                gs_sca,
                                gs_rot,
                                gs_fdc,
                            ], dim=-1).clone().detach().cpu().numpy()
        # dict_GS = {'feat_GS': feat_GS}
        # GS_savepath = os.path.join(self.opt.output_dir, mode, test_dataloader_idx, f'{sha256}_{mode}_{post_fix}_decoded_GS.npz') 
        # np.savez(GS_savepath, **dict_GS)

        # gausssian.save_ply(os.path.join(self.opt.output_dir, mode, test_dataloader_idx,
        #         f'{sha256}_{mode}_{post_fix}_decoded_GS.ply')) # epoch_{self.trainer.current_epoch}

        return None, log_dict
    
    def forward(self, batch, mode='train', test_dataloader_idx=0):
        sha256 = batch.get("sha256")
        scene_id = sha256[0] if isinstance(sha256, list) else str(sha256.item())
                
        SCALE = batch.get("scale").item()
        rand_img_lst = batch.get("rand_img_lst")
        rand_extrinsics_lst = batch.get("rand_extrinsics_lst")
        rand_ori_extrinsics_lst = batch.get("rand_ori_extrinsics_lst")
        rand_intrinsics_lst = batch.get("rand_intrinsics_lst")
        rand_depths_lst = batch.get("rand_depths_lst").unsqueeze(1)  # shape: [N, 1, H, W]
        T_norm = batch.get("T_norm")

        if "voxel_points" in batch: voxel_points = batch.get("voxel_points")
        elif self.sampler == 'tsdf': voxel_points = tsdf_fusion(batch.get("raw_depths_lst"), rand_intrinsics_lst, rand_extrinsics_lst, self.vs_size)
        else:
            voxel_points = depth_to_world(batch.get("raw_depths_lst"), rand_intrinsics_lst, rand_extrinsics_lst)
            voxel_points = voxel_downsample(voxel_points, self.vs_size)

        grid_origin = None # 如果使用了voxel_points, 就不需要grid_origin来反算xyz了
        # bs = feats.shape[0] # 这里我希望bs=1, 因为trellis那个sparsetensor之后的处理没有考虑多batch的操作
        # breakpoint()
        # assert len(rand_img_lst) == len(rand_extrinsics_lst) == len(rand_intrinsics_lst) == 1

        model_name = self.opt.models.name            
        if self.opt.models.name in ['DecoderOnlyRGBCNNCamDepthInV1',
                                    'DecoderOnlyRGBCNNCamDepthInV0',
                                    'DecoderOnlyRGBCNNCamDepthInV2',
                                    'DecoderOnlyRGBCNNCamDepthInV3',
                                    'DecoderOnlyRGBCNNCamDepthInV4',
                                    'DecoderOnlyRGBCNNCamDepthInV2saveMem',
                                    'DecoderOnlyInputViTV2',
                                    'DecoderOnlyRGBCNNCamDepthInV4unify']:
            outputs = self.model.forward(
                rgbOris=rand_img_lst,
                depths=rand_depths_lst,
                intrinsic=rand_intrinsics_lst.clone(),
                ori_w2cs=rand_ori_extrinsics_lst,
                w2cs=rand_extrinsics_lst,
                scale=SCALE,
                T_norm=T_norm,
                voxel_points=voxel_points,
                voxel_align_ratio=self.voxel_align_ratio,
                feat_savepath=None if self.SAVE_ROOT is None else f'{self.SAVE_ROOT}/{scene_id}_decOnly_dict_voxel.npz'
            )

        else:
            raise NotImplementedError(f"Model {model_name} not implemented in src.models")
        self.clean_mem()
        
        rand_depths_lst = batch.get("rand_depths_lst") # shape: [N, 1, H, W]
        rand_img_lst = batch.get("rand_img_lst") # shape: [N, 3, H, W]
        rand_extrinsics_lst = batch.get("rand_extrinsics_lst") # shape: [N, 4, 4]
        rand_intrinsics_lst = batch.get("rand_intrinsics_lst") # shape: [N, 3, 3]
        invalid_mask = batch.get("invalid_mask")
        # print(rand_depths_lst.shape, rand_img_lst.shape, rand_extrinsics_lst.shape, rand_intrinsics_lst.shape)
        # assert len(rand_extrinsics_lst) == len(rand_intrinsics_lst) == len(rand_depths_lst) == len(rand_img_lst)
        loss_total, log_dict = self.compute_log_loss(outputs, rand_img_lst, rand_depths_lst, invalid_mask,
                                                     rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                                                     sha256, mode=mode)

        return loss_total, log_dict
        
    def on_test_start(self):
        # 每次 test 前清空存储
        self.test_outputs = {}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # self.model.eval()
        sha256 = batch.get("sha256")
        SCALE = batch.get("scale")
        
        self.clean_mem()
        
        with torch.no_grad():
            
            out = self.forward(batch)         
            outputs = out.get("outputs")    
            
            rand_depths_lst = batch.get("rand_depths_lst") # shape: [N, 1, H, W]
            rand_img_lst = batch.get("rand_img_lst") # shape: [N, 3, H, W]
            rand_extrinsics_lst = batch.get("rand_extrinsics_lst") # shape: [N, 4, 4]
            rand_intrinsics_lst = batch.get("rand_intrinsics_lst") # shape: [N, 3, 3] 

            _, log_dict = self.evaluate_log(outputs[0], rand_img_lst, rand_depths_lst,
                                            rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                                            sha256, batch_idx, 
                                            mode=f'test', test_dataloader_idx=dataloader_idx)
            log_dict['sha256'] = sha256

            # 收集结果
            if dataloader_idx not in self.test_outputs:
                self.test_outputs[dataloader_idx] = []
            self.test_outputs[dataloader_idx].append(log_dict)
            print(f"dataloader_idx: {dataloader_idx}, sha256: {sha256}",  log_dict)

            if self.opt.demo:
                interp_folder = os.path.join(self.opt.output_dir, f'demo_dataloader{dataloader_idx}', f'{sha256}_interp_views')
                os.makedirs(interp_folder, exist_ok=True)
                voxel_points = batch.get("voxel_points")[0]
                center = outputs[0].get_xyz.mean(dim=0, keepdim=True)
                # center = voxel_points.mean(dim=0, keepdim=True)                
                center += torch.tensor([[-0.1,0.1,0.]]).to(center.device)
                gen_demo(outputs[0], center, self.opt.trainer.render_option, 
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
            with open(os.path.join(self.opt.output_dir, 'test', f'test_results_dataloader{dataloader_idx}.csv'), mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row in dataloader_res:
                    row = {key: row[key] for key in fieldnames} 
                    writer.writerow(row)
        
        self.test_outputs.clear() # 清空，避免多个 epoch 累加
