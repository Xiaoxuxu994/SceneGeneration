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
import torch
import torch_npu
from torchvision.utils import save_image
import csv

from src.representations.gaussian import Gaussian
from src.utils.demo_utils import gen_demo
from src.models.GSdecoder.decoder_RGBcnnCamDepthV1 import DecoderOnlyRGBCNNCamDepthInV1


from train.baseModel import BaseModel
class GSDecoderModel(BaseModel):
    def __init__(self, opt, if_norm_depth=True):
        super(GSDecoderModel, self).__init__(opt)
        self.vs_size = 2/128
        self.if_norm_depth = if_norm_depth
        # 指定在哪些 epoch 做 inference
        self.infer_epochs = [59, 119, 199] + list(range(199, self.opt.trainer.max_epochs, 200))
        self.if_fix_xyz = False
        self.model: DecoderOnlyRGBCNNCamDepthInV1

    def compute_log_loss(self, outputs, 
                         rand_img_lst, rand_depths_lst, 
                         rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                         sha256, mode='train'):
        
        depth_max = rand_depths_lst.max().item()
        new_w, new_h = self.opt.img_resolution
        
        # ------------------------- 预处理 -------------------------
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
        
        # ------------------------- 渲染 -------------------------
        all_train_videos, all_train_depths = self.rendering(outputs[0], rand_extrinsics_lst, rand_intrinsics_lst,
                                                            SCALE=SCALE, 
                                                            invalid_mask=invalid_mask, 
                                                            depth_max=depth_max,
                                                            # batch_idx=batch_idx, 
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
        
        # ------------------------- 可视化 -------------------------
        train_save_trigger = (mode == 'train') and (self.current_epoch % self.opt.trainer.save_img_every_epoch == 0)
        if train_save_trigger or (mode == 'val') or ('test' in mode):
            
            post_fix = 'epoch0_' if self.current_epoch == 0 else '_'

            save_image(all_train_videos, os.path.join(self.opt.output_dir, mode,
                        f"{sha256}_{mode}_{post_fix}rgb_image.png")) # epoch_{self.trainer.current_epoch}
            
            save_image(rand_img_lst, os.path.join(self.opt.output_dir, mode,
                    f"{sha256}_{mode}_{post_fix}GT_rgb_image.png")) # epoch_{self.trainer.current_epoch}     
            
            save_image(all_train_depths.unsqueeze(1), os.path.join(self.opt.output_dir, mode,
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
            gs_fdc = gausssian.get_features.reshape(-1, self.rgb_dim) # [N, 3*(sh+1)**2]
            feat_GS = torch.cat([
                                    gs_xyz,
                                    gs_opa,
                                    gs_sca,
                                    gs_rot,
                                    gs_fdc,
                                ], dim=-1).clone().detach().cpu().numpy()
            # dict_GS = {'feat_GS': feat_GS}
            # GS_savepath = os.path.join(self.opt.output_dir, tag.lower(), f'{sha256}_{mode}_{post_fix}_decoded_GS.npz') 
            # np.savez(GS_savepath, **dict_GS)

            # gausssian.save_ply(os.path.join(self.opt.output_dir, tag.lower(),
                    # f'{sha256}_{mode}_{post_fix}_decoded_GS.ply')) # epoch_{self.trainer.current_epoch}

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
        
        assert isinstance(outputs, list) and len(outputs) == 1
        
        all_train_videos, all_train_depths = self.rendering(outputs[0], rand_extrinsics_lst, rand_intrinsics_lst,
                                                            SCALE=SCALE, 
                                                            invalid_mask=invalid_mask, 
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

        save_image(all_train_videos, os.path.join(self.opt.output_dir, mode, test_dataloader_idx,
                    f"{sha256}_{mode}_{post_fix}rgb_image.png")) # epoch_{self.trainer.current_epoch}
        
        save_image(rand_img_lst, os.path.join(self.opt.output_dir, mode, test_dataloader_idx,
                f"{sha256}_{mode}_{post_fix}GT_rgb_image.png")) # epoch_{self.trainer.current_epoch}     
        
        save_image(all_train_depths, os.path.join(self.opt.output_dir, mode, test_dataloader_idx,
                    f"{sha256}_{mode}_{post_fix}depth.png")) # epoch_{self.trainer.current_epoch}
        
        save_image(rand_depths_lst.unsqueeze(1), os.path.join(self.opt.output_dir, mode, test_dataloader_idx,
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
        gs_fdc = gausssian.get_features.reshape(-1, self.rgb_dim) # [N, 3*(sh+1)**2]
        feat_GS = torch.cat([
                                gs_xyz,
                                gs_opa,
                                gs_sca,
                                gs_rot,
                                gs_fdc,
                            ], dim=-1).clone().detach().cpu().numpy()
        dict_GS = {'feat_GS': feat_GS}
        # GS_savepath = os.path.join(self.opt.output_dir, mode, test_dataloader_idx, f'{sha256}_{mode}_{post_fix}_decoded_GS.npz') 
        # np.savez(GS_savepath, **dict_GS)

        # gausssian.save_ply(os.path.join(self.opt.output_dir, mode, test_dataloader_idx,
        #         f'{sha256}_{mode}_{post_fix}_decoded_GS.ply')) # epoch_{self.trainer.current_epoch}

        return None, log_dict
    
    def forward(self, batch, mode='train', test_dataloader_idx=0):
        sha256 = batch.get("sha256")
        SCALE = batch.get("scale").item()
        voxel_points = batch.get("voxel_points")
        rand_img_lst = batch.get("rand_img_lst")
        rand_extrinsics_lst = batch.get("rand_extrinsics_lst")
        rand_ori_extrinsics_lst = batch.get("rand_ori_extrinsics_lst")
        rand_intrinsics_lst = batch.get("rand_intrinsics_lst")
        rand_depths_lst = batch.get("rand_depths_lst").unsqueeze(1)  # shape: [N, 1, H, W]
        T_norm = batch.get("T_norm")
        
        grid_origin = None # 如果使用了voxel_points, 就不需要grid_origin来反算xyz了
        # bs = feats.shape[0] # 这里我希望bs=1, 因为trellis那个sparsetensor之后的处理没有考虑多batch的操作
        # breakpoint()
        # assert len(rand_img_lst) == len(rand_extrinsics_lst) == len(rand_intrinsics_lst) == 1
        
        model_name = self.opt.models.name            
        
        # if model_name in ['SLatQuerySepMLPC2FGaussianDecoderOnly',
        #                   'SLatQuerySepMLPC2FFGaussianDecoderOnly']:
        #     feats = batch.get("feats")
        #     feats = sp.SparseTensor(
        #         feats=feats.squeeze(0).float(),
        #         coords=torch.cat([
        #             torch.zeros(feats.squeeze(0).shape[0], 1).int().to(self.device),
        #             voxel_indices.squeeze(0).int().to(self.device),
        #         ], dim=1),
        #     )
        #     outputs, _ = self.model(
        #         feats, query=query,
        #         grid_origin=None,
        #         voxel_points=voxel_points,
        #         voxel_align_ratio=self.voxel_align_ratio,
        #         if_fix_xyz=False
        #     )
        # #     # del feats

        # elif model_name in ['SLatQueryRGBCNNCamDepthInSepMLPGaussianDecoderOnly']:
        #     # voxel_indices = voxel_indices[0]
        #     outputs = self.model(
        #         feats=None,
        #         rgbOris=rand_img_lst,
        #         depths=rand_depths_lst,
        #         intrinsic=rand_intrinsics_lst.clone(),
        #         w2cs=rand_extrinsics_lst,
        #         ori_w2cs=rand_ori_extrinsics_lst,
        #         scale=SCALE,
        #         T_norm=T_norm,
        #         grid_origin=None,
        #         voxel_points=voxel_points,
        #         voxel_indices=None,
        #         voxel_align_ratio=self.voxel_align_ratio,
        #         if_fix_xyz=False
        #     )
            
        if self.opt.models.name in ['DecoderOnlyRGBCNNCamDepthInV1',
                                    'DecoderOnlyRGBCNNCamDepthInV0']:
            outputs = self.model.forward(
                rgbOris=rand_img_lst,
                depths=rand_depths_lst,
                intrinsic=rand_intrinsics_lst.clone(),
                ori_w2cs=rand_ori_extrinsics_lst,
                w2cs=rand_extrinsics_lst,
                scale=SCALE,
                T_norm=T_norm,
                grid_origin=None,
                voxel_points=voxel_points,
                voxel_align_ratio=self.voxel_align_ratio,
                if_fix_xyz=self.if_fix_xyz
            )

        # elif model_name in ['SLatQueryCamDepthInSepMLPGaussianDecoderOnly']:
        #     feats = batch.get("feats")
        #     feats = sp.SparseTensor(
        #         feats=feats.squeeze(0).float(),
        #         coords=torch.cat([
        #             torch.zeros(feats.squeeze(0).shape[0], 1).int().to(self.device),
        #             voxel_indices.squeeze(0).int().to(self.device),
        #         ], dim=1),
        #     )
        #     outputs = self.model(
        #         feats, query=query,
        #         depths=rand_depths_lst,
        #         intrinsic=rand_intrinsics_lst.clone(),
        #         w2cs=rand_extrinsics_lst,
        #         ori_w2cs=rand_ori_extrinsics_lst,
        #         scale=SCALE,
        #         T_norm=T_norm,
        #         grid_origin=None,
        #         voxel_points=voxel_points,
        #         voxel_align_ratio=self.voxel_align_ratio,
        #         if_fix_xyz=False
        #     )

        else:
            raise NotImplementedError(f"Model {model_name} not implemented in src.models")
        
        self.clean_mem()
        
        rand_depths_lst = batch.get("rand_depths_lst") # shape: [N, 1, H, W]
        rand_img_lst = batch.get("rand_img_lst") # shape: [N, 3, H, W]
        rand_extrinsics_lst = batch.get("rand_extrinsics_lst") # shape: [N, 4, 4]
        rand_intrinsics_lst = batch.get("rand_intrinsics_lst") # shape: [N, 3, 3]

        # print(rand_depths_lst.shape, rand_img_lst.shape, rand_extrinsics_lst.shape, rand_intrinsics_lst.shape)
        # assert len(rand_extrinsics_lst) == len(rand_intrinsics_lst) == len(rand_depths_lst) == len(rand_img_lst)
        
        loss_total, log_dict = self.compute_log_loss(outputs, rand_img_lst, rand_depths_lst,
                                                     rand_extrinsics_lst, rand_intrinsics_lst, SCALE,
                                                     sha256, mode=f'train')

        return loss_total, log_dict
        
    def on_test_start(self):
        # 每次 test 前清空存储
        self.test_outputs = {}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.eval()
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

            # 收集结果
            if dataloader_idx not in self.test_outputs:
                self.test_outputs[dataloader_idx] = []
            self.test_outputs[dataloader_idx].append(log_dict)
            print(f"dataloader_idx: {dataloader_idx}", log_dict)

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
        for dataloader_idx, dataloader_res in self.test_outputs.items():

            fieldnames = dataloader_res.keys()
            # 保存为CSV文件
            with open(os.path.join(self.opt.output_dir, 'test', f'test_results_dataloader{dataloader_idx}.csv'), mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()  # 写入列名（字段）
                writer.writerows(dataloader_res)  # 写入字典数据
        
        self.test_outputs.clear() # 清空，避免多个 epoch 累加
