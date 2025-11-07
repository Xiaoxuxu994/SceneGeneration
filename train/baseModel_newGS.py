"""
pytorch lightning 基类模型
"""
import os
import numpy as np
import gc
from torchmetrics.functional import structural_similarity_index_measure as ssim
import lpips

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import src.models as models
from src.representations.gaussian import Gaussian
# from src.utils import render_utils_scannet as render_utils
from src.utils import render_utils_npu as render_utils
from src.utils import LRscheduler_utils as LRschedulers 
from src.utils.train_utils import get_model_summary
from src.utils.loss_utils import masked_ssim_batch, l1_loss, dispersion_loss_voxel_scaled, feature_diversity_loss

# from meta_gauss_render.ascend_gauss_render import AscendGaussRender
from train.rasterizer import Rasterizer

class BaseModel(nn.Module):
    def __init__(self, opt, if_norm_depth=True):
        super(BaseModel, self).__init__()
        self.opt = opt  # 配置选项

        self.rep_config = opt.models.args.representation_config
        self.sh_degree = opt.models.args.representation_config.sh_degree if hasattr(opt.models.args.representation_config, 'sh_degree') else 0
        self.rgb_dim = 3 * ((self.sh_degree + 1) **2)
        self.num_GS = opt.models.args.representation_config['num_gaussians']
        self.voxel_align_ratio = opt.models.voxel_align_ratio if hasattr(opt.models, 'voxel_align_ratio') else 5
        self.warmup_epochs = self.opt.trainer.lr_scheduler.args.warmup_steps
        self.model = getattr(models, opt.models.name)(**opt.models.args)
        model_summary = get_model_summary(self.model)
        with open(os.path.join(opt.output_dir, f'model_summary.txt'), 'w') as fp:
            print(model_summary, file=fp)
        self.lpips_loss = lpips.LPIPS(net='vgg').eval()
        self.lpips_loss.eval()
        for p in self.lpips_loss.parameters():
            p.requires_grad = False
        # self.ssim_metric = StructuralSimilarityIndexMeasure().eval() #.to(outputs[0].device)
        self.if_norm_depth = if_norm_depth
        
        # self.gauss_render = AscendGaussRender(width=opt.img_resolution[0], 
        #                                       height=opt.img_resolution[1], 
        #                                       active_sh_degree=self.sh_degree, 
        #                                       isect_mode='flashgs', 
        #                                       cpu_radix_sort=True)
        self.rasterizer = Rasterizer()

        # self.logger = SummaryWriter(opt.logger_dir)
        self.current_epoch = 0
    
    def on_train_start(self):
        self.device = next(self.parameters()).device
    
    def forward(self, batch, mode='train'):
        pass

    def training_step(self, batch):
        pass

    def validation_step(self, batch):
        pass

    def configure_optimizers(self):
        model_params = self.model.parameters() # [param for _, param in model.named_parameters()]

        if hasattr(torch.optim, self.opt.trainer.optimizer['name']):
            optimizer = getattr(torch.optim,  self.opt.trainer.optimizer['name'])(model_params, **self.opt.trainer.optimizer['args'])

        # Initalize learning rate scheduler 
        if self.opt.trainer.lr_scheduler is not None:
            if hasattr(torch.optim.lr_scheduler, self.opt.trainer.lr_scheduler['name']):
                scheduler = getattr(torch.optim.lr_scheduler, self.opt.trainer.lr_scheduler['name'])(optimizer, **self.opt.trainer.lr_scheduler['args'])
            elif hasattr(LRschedulers, self.opt.trainer.lr_scheduler['name']):
                scheduler = getattr(LRschedulers, self.opt.trainer.lr_scheduler['name'])(optimizer, **self.opt.trainer.lr_scheduler['args'])
            else:
                raise ValueError(f"Unsupported lr_scheduler: {self.opt.trainer.lr_scheduler['name']}")
        else:
            scheduler = None
            
        return [optimizer], [scheduler]

    @staticmethod
    def clean_mem():
        gc.collect()
        torch.npu.empty_cache()
    
    def rendering(self, outputs, rand_extrinsics_lst, rand_intrinsics_lst, SCALE: float,
                  sha256, invalid_mask=None, depth_max=0, render_mode='RGB+D', mode='train'): # mode='val'
        """
        outputs: Gaussian Class or Torch Tensor [N, 14]
        渲染 RGB + D
        invalid_mask: [V, H, W]
        
        return: rgb:  [V,3,H,W], depth: [V,H,W]
        """
        
        # device = self.device
        new_w, new_h = self.opt.img_resolution
        # ------------------------- 预处理 -------------------------


        # ------------------------- 渲染  -------------------------
        all_train_videos, all_train_depths = [], []
        num_valid_view = len(rand_extrinsics_lst)
        # is_infer = False if mode=='train' else True

        if isinstance(outputs, Gaussian):
            splats = dict(
                means=outputs.get_xyz,  # [N, 3]
                quats=outputs.get_rotation,  # [N, 4]
                scales=outputs.get_scaling,  # [N, 3]
                opacities=outputs.get_opacity[..., 0],  # [N,]
            )
            colors = outputs.get_features # [N,K,3] # .reshape(-1, (self.sh_degree + 1) **2, 3)  # [N, 3]
            print("GS class colors.shape: ", colors.shape)
            colors = colors.reshape(-1, self.rgb_dim//3, 3)
        elif isinstance(outputs, torch.Tensor):
            splats = dict(
                means=outputs[:, :3],
                quats=outputs[:, 7:11],  # [N, 4]
                scales=outputs[:, 4:7],  # [N, 3]
                opacities=outputs[:, 3],  # [N,]
            )
            colors = outputs[:, None, 11:].reshape(-1, self.rgb_dim//3, 3)
            print("GS tensor colors.shape: ", colors.shape)

        # all_train_videos, all_train_depths, _ = self.rasterizer.ascend_rasterize_splats(
        #     camtoworlds=rand_extrinsics_lst,  # [1, 4, 4]
        #     Ks=rand_intrinsics_lst,  # [1, 3, 3]
        #     width=new_w,
        #     height=new_h,
        #     splats=splats,
        #     colors=colors,
        #     sh_degree=0
        # )

        # 节省显存
        for i in range(num_valid_view):

            rand_ext = rand_extrinsics_lst[i].unsqueeze(0)
            rand_int = rand_intrinsics_lst[i].unsqueeze(0)

            # results = render_utils.render_gtpose(
            #     self.gauss_render,
            #     outputs,
            #     resolution=(new_h, new_w),
            #     extrinsics=rand_ext,
            #     intrinsics=rand_int,
            #     render_option=self.opt.trainer.render_option,
            #     render_mode=render_mode,
            #     sh_degree=self.sh_degree,
            #     is_infer=is_infer,
            #     bg_color=(0, 0, 0),
            #     verbose=False
            # )

            video, depth, _ = self.rasterizer.ascend_rasterize_splats(
                camtoworlds=rand_ext,  # [1, 4, 4]
                Ks=rand_int,  # [1, 3, 3]
                width=new_w,
                height=new_h,
                splats=splats,
                colors=colors,
                sh_degree=0
            )

            all_train_videos.append(video.movedim(3, 1))
            all_train_depths.append(depth[..., 0])

        all_train_videos = torch.cat(all_train_videos, dim=0).clamp(0, 1) # [V, 3, H, W]
        all_train_depths = torch.cat(all_train_depths, dim=0) / SCALE # [V, H, W]

        if self.if_norm_depth: all_train_depths = all_train_depths / depth_max
        all_train_videos = all_train_videos.masked_fill(invalid_mask[:, None], 0)
        all_train_depths = all_train_depths.masked_fill(invalid_mask, 0)

        return all_train_videos, all_train_depths

    def compute_loss_rgb(self, all_train_videos, rand_img_lst):
        # ------------------------- 渲染 loss -------------------------
        rand_img_lst = rand_img_lst.clamp(0.0, 1.0)
        all_train_videos = all_train_videos.clamp(0.0, 1.0)
        loss_img_rec_l1 = self.opt.trainer.loss_weight_img_rec_l1 * F.l1_loss(rand_img_lst, all_train_videos, reduction='mean')
        loss_img_rec_ssim = self.opt.trainer.loss_weight_img_rec_ssim * (1- ssim(all_train_videos, rand_img_lst))

        loss_img_rec_lpips = 0
        if self.opt.trainer.loss_weight_img_rec_lpips > 0:
            for i in range(len(all_train_videos)): 
                loss_img_rec_lpips += self.lpips_loss(all_train_videos[i], rand_img_lst[i])
            loss_img_rec_lpips = self.opt.trainer.loss_weight_img_rec_lpips * loss_img_rec_lpips / len(all_train_videos)        

        loss_img_rec = loss_img_rec_l1 + loss_img_rec_ssim + loss_img_rec_lpips 

        return loss_img_rec

    def evaluate_rgb(self, all_train_videos, rand_img_lst):
        # ------------------------- 渲染 loss -------------------------
        rand_img_lst = rand_img_lst.clamp(0.0, 1.0)
        all_train_videos = all_train_videos.clamp(0.0, 1.0)
        img_rec_l1 = F.l1_loss(rand_img_lst, all_train_videos, reduction='mean')
        img_rec_ssim = ssim(all_train_videos, rand_img_lst)
        
        for i in range(len(all_train_videos)): 
            img_rec_lpips += self.lpips_loss(all_train_videos[i], rand_img_lst[i])
        img_rec_lpips = img_rec_lpips / len(all_train_videos)        
        
        avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
        
        return img_rec_l1, img_rec_ssim, img_rec_lpips, avg_psnr
    
    def compute_psnr(self, all_train_videos, rand_img_lst, eps=1e-8):
        # ------------------------- 计算PSNR -------------------------
        mse = torch.mean((all_train_videos - rand_img_lst) ** 2, dim=[1,2,3]) 
        psnr = -10.0 * torch.log10(mse + eps)              # 防止 log0
        return psnr.mean()
    
    def compute_loss_depth(self, all_train_depths, rand_depths_lst):
        loss_depth_rec_l1 = self.opt.trainer.loss_weight_depth_l1 * F.l1_loss(all_train_depths, rand_depths_lst, reduction='mean')  
        return loss_depth_rec_l1
    
    def compute_loss_gs(self, outputs):
        # ------------------------- Gaussian Loss -------------------------
        loss_opa, loss_scale = 0, 0
        loss_gs_dispersion = 0
        loss_gs_diversity = 0

        for output in outputs:
            # opacity & scale
            loss_opa += self.opt.trainer.loss_weight_opacity * ((1 - output.get_opacity) ** 2).mean()
            loss_scale += self.opt.trainer.loss_weight_scale * torch.prod(output.get_scaling, dim=1).mean()

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
                    
        return loss_opa, loss_scale, loss_gs_dispersion, loss_gs_diversity
    
    def on_test_start(self):
        pass
        # # 每次 test 前清空存储
        # self.test_outputs = {}
        
    def on_test_epoch_end(self):
        pass
        # for idx, results in self.test_outputs.items():
        #     avg_psnr = torch.stack([x["avg_psnr"] for x in results]).mean()
        #     num_scenes = len(results)
        #     print(f"[Prediction dataload_idx {idx}] num_scenes={num_scenes:.4f}, avg_psnr={avg_psnr:.4f}")
        # self.log(f"test/avg_psnr_data{idx}", avg_psnr, prog_bar=True, sync_dist=True)
        
        # self.test_outputs.clear() # 清空，避免多个 epoch 累加
    
    # def log_dict(self, scalar_dict):
    #     for k, v in scalar_dict.items():
    #         self.logger.add_scalar(k, v)
    
    # def compute_log_loss(self, outputs, 
    #                      rand_img_lst, rand_depths_lst, 
    #                      all_train_videos, all_train_depths,
    #                      sha256, batch_idx, mode=f'train'):
        
    #     avg_loss_rec = self.compute_loss_rgb(all_train_videos, rand_img_lst)
    #     avg_loss_depth = self.compute_loss_depth(all_train_depths, rand_depths_lst)
    #     loss_opa, loss_scale, loss_gs_dispersion, loss_gs_diversity = self.compute_loss_gs(outputs)
        
    #     avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
        
    #     # ------------------------- 总 loss -------------------------
    #     loss_total = avg_loss_rec + loss_opa + loss_scale + avg_loss_depth + loss_gs_dispersion + loss_gs_diversity

    #     # ------------------------- 日志 -------------------------
    #     if mode == 'train':
    #         tag = "Training" 
    #     elif mode == 'val':
    #         tag = "Validation"
    #     elif 'test' in mode:
    #         tag = "Test"
    #     log_dict = {
    #         "lr": self.trainer.optimizers[0].param_groups[0]['lr'],
    #         f"{tag}/{mode}_loss_total": loss_total.item(),
    #         f"{tag}/{mode}_loss_rec_avg": avg_loss_rec.item(),
    #         f"{tag}/{mode}_loss_opa": loss_opa.item(),
    #         f"{tag}/{mode}_loss_scale": loss_scale.item(),
    #         f"{tag}/{mode}_loss_depth_avg": avg_loss_depth.item()
    #         # f"{tag}/{mode}_avg_psnr_supervised_views": avg_psnr.item()
    #     }
    #     if mode == 'train':
    #         log_dict[f"{tag}/{mode}_avg_psnr_supervised_views"] = avg_psnr.item()
    #     elif mode == 'val':
    #         log_dict[f"{tag}/{mode}_avg_psnr"] = avg_psnr.item()
    #     elif 'test' in mode:
    #         log_dict = {f"{tag}/{mode}_avg_psnr": avg_psnr.item()}
    #         loss_total = avg_psnr
    #     else:
    #         raise NotImplementedError(f"Undefined {mode}")
            
    #     if hasattr(self.opt.trainer, "loss_weight_GSdispersion") and self.opt.trainer.loss_weight_GSdispersion > 0:
    #         log_dict[f"{tag}/{mode}_loss_gs_dispersion"] = loss_gs_dispersion.item()
    #     if hasattr(self.opt.trainer, "loss_weight_GSdiversity") and self.opt.trainer.loss_weight_GSdiversity > 0:
    #         log_dict[f"{tag}/{mode}_loss_gs_diversity"] = loss_gs_diversity.item()

    #     self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False if mode == 'train' else True)
    #     # logger.info(f"[{tag}] epoch: {self.trainer.current_epoch}, sha256: {sha256}, {mode}_log_dict = " + json.dumps(log_dict, indent=2))
        
    #     # ------------------------- 可视化 -------------------------
    #     train_save_trigger = (mode == 'train') and (self.trainer.current_epoch % self.opt.trainer.save_img_every_epoch == 0)
    #     if train_save_trigger or (mode == 'val') or ('test' in mode):
            
    #         post_fix = 'epoch0_' if self.trainer.current_epoch == 0 else '_'

    #         save_image(all_train_videos, os.path.join(self.opt.output_dir, tag.lower(),
    #                     f"{sha256}_{mode}_{post_fix}rgb_image.png")) # epoch_{self.trainer.current_epoch}
            
    #         save_image(rand_img_lst, os.path.join(self.opt.output_dir, tag.lower(),
    #                 f"{sha256}_{mode}_{post_fix}GT_rgb_image.png")) # epoch_{self.trainer.current_epoch}     
            
    #         save_image(all_train_depths, os.path.join(self.opt.output_dir, tag.lower(),
    #                     f"{sha256}_{mode}_{post_fix}depth.png")) # epoch_{self.trainer.current_epoch}
            
    #         save_image(rand_depths_lst.unsqueeze(1), os.path.join(self.opt.output_dir, tag.lower(),
    #                 f"{sha256}_{mode}_{post_fix}GT_depth.png")) # epoch_{self.trainer.current_epoch}
            
    #         if isinstance(outputs, list) and len(outputs) == 1:
    #             gausssian = outputs[0]
    #         elif isinstance(outputs, Gaussian):
    #             gausssian = outputs
    #         else:
    #             raise NotImplementedError(f"Unsupported output type for visualization: {type(outputs)}")
            
    #         gs_xyz = gausssian.get_xyz
    #         gs_opa = gausssian.get_opacity
    #         gs_sca = gausssian.get_scaling
    #         gs_rot = gausssian.get_rotation
    #         gs_fdc = gausssian.get_features.reshape(-1,3)
    #         feat_GS = torch.cat([
    #                                 gs_xyz,
    #                                 gs_opa,
    #                                 gs_sca,
    #                                 gs_rot,
    #                                 gs_fdc,
    #                             ], dim=-1).clone().detach().cpu().numpy()
    #         dict_GS = {'feat_GS': feat_GS}
    #         GS_savepath = os.path.join(self.opt.output_dir, tag.lower(), f'{sha256}_{mode}_{post_fix}_densifier_GS.npz') 
    #         np.savez(GS_savepath, **dict_GS)

    #         gausssian.save_ply(os.path.join(self.opt.output_dir, tag.lower(),
    #                 f'{sha256}_{mode}_{post_fix}_densified_GS.ply')) # epoch_{self.trainer.current_epoch}
        
    #     return loss_total
    
    # def compute_log_loss_rgb_only_for_refiner(self, outputs, 
    #                                           rand_img_lst, rand_depths_lst, 
    #                                           all_train_videos, all_train_depths,
    #                                           sha256, batch_idx, refiner_idx=0, mode=f'train'):
    #     """
    #     outputs: Gaussian Class
    #     """
        
    #     avg_loss_rec = self.compute_loss_rgb(all_train_videos, rand_img_lst)
    #     avg_psnr = self.compute_psnr(all_train_videos, rand_img_lst)
    #     # ------------------------- 总 loss -------------------------
    #     loss_total = avg_loss_rec 

    #     # ------------------------- 日志 -------------------------
    #     if mode == 'train':
    #         tag = "Training" 
    #     elif mode == 'val':
    #         tag = "Validation"
    #     elif 'test' in mode:
    #         tag = "Test"
    #     log_dict = {
    #         "lr": self.trainer.optimizers[0].param_groups[0]['lr'],
    #         f"{tag}/{mode}_refiner{refiner_idx}_loss_total": loss_total.item(),
    #         f"{tag}/{mode}_refiner{refiner_idx}_loss_rec_avg": avg_loss_rec.item(),
    #         f"{tag}/{mode}_refiner{refiner_idx}_avg_psnr": avg_psnr.item()
    #     }

    #     self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False if mode == 'train' else True)
    #     # logger.info(f"[{tag}] epoch: {self.trainer.current_epoch}, sha256: {sha256}, {mode}_log_dict = " + json.dumps(log_dict, indent=2))
        
    #     # ------------------------- 可视化 -------------------------
    #     with torch.no_grad():
    #         train_save_trigger = (mode == 'train') and (self.trainer.current_epoch % self.opt.trainer.save_img_every_epoch == 0) 
    #         if train_save_trigger or (mode == 'val') or ('test' in mode):
                
    #             post_fix = f'epoch{self.trainer.current_epoch}_refiner{refiner_idx}' if self.trainer.current_epoch == 0 else f'refiner{refiner_idx}'

    #             save_image(all_train_videos, os.path.join(self.opt.output_dir, tag.lower(),
    #                         f"{sha256}_{mode}_{post_fix}_rgb_image.png")) 
                
    #             save_image(rand_img_lst, os.path.join(self.opt.output_dir, tag.lower(),
    #                     f"{sha256}_{mode}_{post_fix}_GT_rgb_image.png"))
                
    #             save_image(all_train_depths.unsqueeze(1) if len(all_train_depths.shape)==3 else all_train_depths, 
    #                        os.path.join(self.opt.output_dir, tag.lower(),
    #                         f"{sha256}_{mode}_{post_fix}_depth.png"))
                
    #             save_image(rand_depths_lst.unsqueeze(1) if len(rand_depths_lst.shape)==3 else rand_depths_lst, 
    #                        os.path.join(self.opt.output_dir, tag.lower(),
    #                     f"{sha256}_{mode}_{post_fix}_GT_depth.png"))
                
    #             if isinstance(outputs, torch.Tensor):
    #                 dict_GS = {'feat_GS': outputs.clone().detach().cpu().numpy()}
    #                 GS_savepath = os.path.join(self.opt.output_dir, tag.lower(), f'{sha256}_{mode}_{post_fix}_refined_GS.npz') 
    #                 np.savez(GS_savepath, **dict_GS)
                    
    #                 gaussians = Gaussian(
    #                     sh_degree=self.sh_degree,
    #                     mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
    #                     scaling_bias = self.rep_config['scaling_bias'],
    #                     opacity_bias = self.rep_config['opacity_bias'],
    #                     scaling_activation = self.rep_config['scaling_activation']
    #                 )
                    
    #                 gaussians.from_xyz(outputs[:,:3])
    #                 gaussians.from_opacity(outputs[:, 3:4])
    #                 gaussians.from_scaling(outputs[:, 4:7])
    #                 gaussians.from_rotation(outputs[:, 7:11])
    #                 gaussians.from_features(outputs[:, 11:])
    #                 gaussians.save_ply(os.path.join(self.opt.output_dir, tag.lower(),
    #                                 f'{sha256}_{mode}_{post_fix}_refined_GS.ply')) # epoch_{self.trainer.current_epoch}
                    
    #             else:
    #                 gausssian = outputs
    #                 gs_xyz = gausssian.get_xyz
    #                 gs_opa = gausssian.get_opacity
    #                 gs_sca = gausssian.get_scaling
    #                 gs_rot = gausssian.get_rotation
    #                 gs_fdc = gausssian.get_features.reshape(-1,3)
    #                 feat_GS = torch.cat([
    #                                         gs_xyz,
    #                                         gs_opa,
    #                                         gs_sca,
    #                                         gs_rot,
    #                                         gs_fdc,
    #                                     ], dim=-1).clone().detach().cpu().numpy()
    #                 dict_GS = {'feat_GS': feat_GS}
    #                 GS_savepath = os.path.join(self.opt.output_dir, tag.lower(), f'{sha256}_{mode}_{post_fix}_refined_GS.npz') 
    #                 np.savez(GS_savepath, **dict_GS)
    #                 outputs.save_ply(os.path.join(self.opt.output_dir, tag.lower(),
    #                                 f'{sha256}_{mode}_{post_fix}_refined_GS.ply')) # epoch_{self.trainer.current_epoch}
        
    #     return loss_total, avg_psnr