"""
模型Datasets, 包括 GSDecoder， Densifier， Refiner
"""
from easydict import EasyDict as edict
import os
from pandas import DataFrame
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

idx32 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
idx48 = [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25, 27, 28, 29, 31, 32, 34, 35, 36, 38, 39, 40, 42, 43, 44, 46, 47, 48, 50, 51, 52, 54, 55, 56, 58, 59, 60, 62, 63]
idx64 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

def get_even_indices(total: int, n: int):
    """
    在 [0, total) 里按顺序均匀采样 n 个索引。
    - total: 原始视角数，比如 64
    - n: 想要的个数，比如 48 / 32
    返回: 一个 Python list，比如 [0, 1, 3, ...]
    """
    if n >= total:
        return list(range(total))
    idx = np.linspace(0, total - 1, n)
    idx = np.round(idx).astype(int)
    # 去重并保持顺序
    _, uniq = np.unique(idx, return_index=True)
    idx = idx[np.sort(uniq)]
    return idx.tolist()

class BaseDataset(Dataset):
    def __init__(self, metadata: DataFrame, opt: edict, stage: str = 'train', shard_data: bool = False):
        """
        metadata: 包含所有样本信息的 DataFrame，每个样本至少包含 'sha256'
        opt: 参数选项，包含 voxel_size 等参数
        """
        self.metadata = metadata.reset_index(drop=True)
        self.opt = opt
        if stage == 'test':
            self.NVviewInfo_dir = opt.NVviewInfo_dir
            self.NVdepth_dir = opt.NVdepth_dir if hasattr(opt, 'NVdepth_dir') else None
        self.NUM_SV_VIEW = opt.NUM_SV_VIEW if hasattr(opt, 'NUM_SV_VIEW') else 32
        self.NUM_NV_VIEW = opt.NUM_NV_VIEW if hasattr(opt, 'NUM_NV_VIEW') else 4

        self.viewInfo_dir = opt.viewInfo_dir
        self.depth_dir = opt.depth_dir if hasattr(opt, 'depth_dir') else None
        self.mask_dir = opt.mask_dir if hasattr(opt, 'mask_dir') else None
        self.select = idx32 if opt.view_select == 32 else idx48 if opt.view_select == 48 else idx64
        self.select = self.select if 'fix64' in self.viewInfo_dir else ...
        self.num_rend_views = opt.trainer.num_rend_views if stage == 'train' else opt.trainer.num_rend_views_val if stage == 'val' else 32
        self.stage = stage
        self.shard_data = shard_data and self.stage == 'train'
        if self.shard_data: self.viewInfo_cache, self.depth_cache, self.mask_cache = {}, {}, {}
        self.use_ori_metric = True

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

    def __getitem__(self, idx: int):
        new_w, new_h = self.opt.img_resolution
        sha256 = str(self.metadata.loc[idx, 'sha256'])

        try:
            if self.shard_data and idx in self.viewInfo_cache: viewInfo = self.viewInfo_cache[idx]
            else:
                viewInfo = np.load(os.path.join(self.viewInfo_dir, f'{sha256}_viewInfo.npz'))
                if self.shard_data: self.viewInfo_cache[idx] = viewInfo
            T_norm = torch.tensor(viewInfo['T_norm'], dtype=torch.float)
            SCALE = viewInfo['scale'].item()
            
            for key in 'ori_extrinsics_norm128_v14_fix32', 'ori_extrinsics_norm128_v15_fix32':
                if key in viewInfo: all_ori_extrinsics_lst = torch.tensor(viewInfo[key][self.select], dtype=torch.float)
            for key in 'normed_intrinsics_norm128_v14_fix32', 'normed_intrinsics_norm128_v15_fix32':
                if key in viewInfo: all_intrinsics_lst = torch.tensor(viewInfo[key][self.select], dtype=torch.float)
            if not self.use_ori_metric:
                for key in 'extrinsics_Tnorm_norm128_v14_fix32', 'extrinsics_Tnorm_norm128_v15_fix32':
                    if key in viewInfo: all_extrinsics_lst = torch.tensor(viewInfo[key][self.select], dtype=torch.float)           

            batch_data = dict(
                sha256=sha256,
                scale=SCALE,
                T_norm=T_norm
            )

            for key in 'ori_image', 'ori_img_tensor':
                if key in viewInfo: all_images_lst = torch.tensor(viewInfo[key][self.select], dtype=torch.float)
            if self.depth_dir is None:
                for key in 'depthmap_1080x1620_norm128_v14_fix32', 'depthmap_1168x1752_norm128_v15_fix32':
                    if key in viewInfo: all_depths_lst =  torch.tensor(viewInfo[key][self.select], dtype=torch.float)
                if all_depths_lst.ndim == 3: all_depths_lst.unsqueeze_(1)
            elif self.shard_data and idx in self.depth_cache: all_depths_lst = self.depth_cache[idx]
            else:
                all_depths_lst = torch.tensor(np.load(os.path.join(self.depth_dir, f'{sha256}.npz'))['data'], dtype=torch.float) # / SCALE
                if all_depths_lst.ndim == 3: all_depths_lst.unsqueeze_(1)
                if self.shard_data: self.depth_cache[idx] = all_depths_lst

            if all_images_lst.shape[-2:] != (new_h, new_w):
                all_images_lst = F.interpolate(all_images_lst, size=(new_h, new_w), mode='area')
            if all_depths_lst.shape[-2:] != (new_h, new_w):
                all_depths_lst = F.interpolate(all_depths_lst, size=(new_h, new_w), mode='nearest').squeeze(1)
            if all_depths_lst.ndim == 4: all_depths_lst.squeeze_(1)

            if self.mask_dir is None: all_masks_lst = all_depths_lst <= 1e-6
            elif self.shard_data and idx in self.mask_cache: all_masks_lst = self.mask_cache[idx]
            else:
                all_masks_lst = torch.tensor(~np.load(os.path.join(self.mask_dir, f'{sha256}_large_cc.npz'))['large_cc_masks'][self.select], dtype=torch.bool)
                if all_masks_lst.shape[-2:] != (new_h, new_w):
                    all_masks_lst = F.interpolate(all_masks_lst.float(), size=(new_h, new_w), mode='area').squeeze(1).round_().bool()
                if self.shard_data: self.mask_cache[idx] = all_masks_lst

            assert all_images_lst.shape[0] == all_ori_extrinsics_lst.shape[0] == all_intrinsics_lst.shape[0] == all_depths_lst.shape[0] == all_masks_lst.shape[0], \
                f"Shape mismatch: images {all_images_lst.shape}, extrinsics {all_ori_extrinsics_lst.shape}, intrinsics {all_intrinsics_lst.shape}, depths {all_depths_lst.shape}, masks {all_masks_lst.shape}"

            if self.stage == 'train': # Train阶段采样 num_rend_views 个有效视角， val或者test不进行shuffle
                rand_indices = torch.randperm(all_images_lst.shape[0])[:self.num_rend_views]
            else: rand_indices = torch.linspace(0, all_images_lst.shape[0] - 1, self.num_rend_views).round_().int().unique_consecutive()
            all_images_lst = all_images_lst[rand_indices]
            all_intrinsics_lst = all_intrinsics_lst[rand_indices]
            all_depths_lst = all_depths_lst[rand_indices]
            all_ori_extrinsics_lst = all_ori_extrinsics_lst[rand_indices]
            all_masks_lst = all_masks_lst[rand_indices]
            if self.use_ori_metric:
                all_extrinsics_lst = all_ori_extrinsics_lst
            else:
                all_extrinsics_lst = all_extrinsics_lst[rand_indices]

            batch_data.update(
                rand_img_lst=all_images_lst, 
                rand_ori_extrinsics_lst=all_ori_extrinsics_lst,
                rand_extrinsics_lst=all_extrinsics_lst, 
                rand_intrinsics_lst=all_intrinsics_lst, 
                rand_depths_lst=all_depths_lst,
                invalid_mask=all_masks_lst,
            )
            if self.stage in ['test']:
                viewInfo = np.load(os.path.join(self.NVviewInfo_dir, f'{sha256}_viewInfo.npz'))
                T_norm = torch.tensor(viewInfo['T_norm'], dtype=torch.float)
                SCALE = viewInfo['scale'].item()

                for key in 'extrinsics_Tnorm_norm128_v14_fix32', 'extrinsics_Tnorm_norm128_v15_fix32':
                    if key in viewInfo: all_extrinsics_lst = torch.tensor(viewInfo[key], dtype=torch.float)
                for key in 'ori_extrinsics_norm128_v14_fix32', 'ori_extrinsics_norm128_v15_fix32':
                    if key in viewInfo: all_ori_extrinsics_lst = torch.tensor(viewInfo[key], dtype=torch.float)
                for key in 'normed_intrinsics_norm128_v14_fix32', 'normed_intrinsics_norm128_v15_fix32':
                    if key in viewInfo: all_intrinsics_lst = torch.tensor(viewInfo[key], dtype=torch.float)
                for key in 'ori_image', 'ori_img_tensor':
                    if key in viewInfo: all_images_lst = torch.tensor(viewInfo[key], dtype=torch.float)
                if self.depth_dir is None:
                    for key in 'depthmap_1080x1620_norm128_v14_fix32', 'depthmap_1168x1752_norm128_v15_fix32':
                        if key in viewInfo: all_depths_lst =  torch.tensor(viewInfo[key], dtype=torch.float)
                else: all_depths_lst = torch.tensor(np.load(os.path.join(self.NVdepth_dir, f'{sha256}.npz'))['data'], dtype=torch.float) # / SCALE
                if all_depths_lst.ndim == 3: all_depths_lst.unsqueeze_(1)

                sv_w, sv_h = self.opt.trainer.rend_sv_res
                
                if all_images_lst.shape[-2:] != (sv_h, sv_w):
                    all_images_lst = F.interpolate(all_images_lst, size=(sv_h, sv_w), mode='area')
                if all_depths_lst.shape[-2:] != (sv_h, sv_w):
                    all_depths_lst = F.interpolate(all_depths_lst, size=(sv_h, sv_w), mode='nearest')

                assert all_images_lst.shape[0] == all_extrinsics_lst.shape[0] == all_intrinsics_lst.shape[0], \
                    f"Shape mismatch: images {all_images_lst.shape}, extrinsics {all_extrinsics_lst.shape}, intrinsics {all_intrinsics_lst.shape}, depths {all_depths_lst.shape}"

                # 从8个视角中选择需要的视角数量
                selct_indices = get_even_indices(8, self.NUM_NV_VIEW)
                all_images_lst = all_images_lst[selct_indices]
                all_extrinsics_lst = all_extrinsics_lst[selct_indices]
                all_intrinsics_lst = all_intrinsics_lst[selct_indices]
                all_ori_extrinsics_lst = all_ori_extrinsics_lst[selct_indices]
                # depth不用变，已经选好了
                if self.use_ori_metric:
                    all_extrinsics_lst = all_ori_extrinsics_lst
                else:
                    all_extrinsics_lst = all_extrinsics_lst[rand_indices]
                
                batch_data.update(
                    nv_img_lst=all_images_lst, 
                    nv_ori_extrinsics_lst=all_ori_extrinsics_lst,
                    nv_extrinsics_lst=all_extrinsics_lst, 
                    nv_intrinsics_lst=all_intrinsics_lst, 
                    nv_depths_lst=all_depths_lst,
                )

            return batch_data
        except: raise Exception(f'BaseDataset error while loading {sha256}')

    @staticmethod
    def get_extrinsics_for_samples(view, T_norm=None):
        c2w = torch.tensor(view['transform_matrix'])
        c2w[2, :] *= -1
        c2w = c2w[np.array([1,0,2,3]), :]
        c2w[:3, 1:3] *= -1
        w2c = torch.inverse(c2w)
        w2c_Tnorm = w2c @ torch.inverse(T_norm) if T_norm is not None else w2c
        return w2c_Tnorm


class GSDecoderDataset(BaseDataset):
    def __init__(self, metadata: DataFrame, opt: edict, stage: str = 'train', shard_data: bool = False):
        super().__init__(metadata, opt, stage, shard_data)
        self.pts_dir = self.opt.pts_dir
        if self.shard_data: self.pts_cache = {}

    def __getitem__(self, idx: int):
        try:
            batch_data = super().__getitem__(idx)
            if self.shard_data and idx in self.pts_cache: voxel_points = self.pts_cache[idx]
            else:
                pts_path = os.path.join(self.pts_dir, f'{batch_data["sha256"]}_pts.npy')
                if not os.path.isfile(pts_path): pts_path = os.path.join(self.pts_dir, f'{batch_data["sha256"]}.npy')
                voxel_points = torch.tensor(np.load(pts_path), dtype=torch.float)
                if self.shard_data: self.pts_cache[idx] = voxel_points
            batch_data.update(voxel_points=voxel_points)
            return batch_data
        except: raise Exception(f'GSDecoderDataset error while loading {batch_data["sha256"]}')


class RefinerDataset(BaseDataset):
    def __init__(self, metadata: DataFrame, opt: edict, stage: str = 'train', shard_data: bool = False):
        super().__init__(metadata, opt, stage, shard_data)
        self.voxelGS_feats_dir = self.opt.voxelGS_feats_dir
        if self.shard_data: self.feat_voxel_cache, self.feat_GS_cache, self.rend_err_GS_cache = {}, {}, {}

    def __getitem__(self, idx: int):
        try: 
            batch_data = super().__getitem__(idx)

            if self.shard_data and idx in self.feat_voxel_cache: feat_voxel = self.feat_voxel_cache[idx]
            else:
                voxel_feats = np.load(os.path.join(self.voxelGS_feats_dir, f'{batch_data["sha256"]}_decOnly_dict_voxel.npz'))
                feat_voxel = torch.tensor(voxel_feats['feat_voxel'], dtype=torch.bfloat16)
                if self.shard_data: self.feat_voxel_cache[idx] = feat_voxel

            if self.shard_data and idx in self.feat_GS_cache: feat_GS = self.feat_GS_cache[idx]
            else:
                GS_feats_path = os.path.join(self.voxelGS_feats_dir, f'{batch_data["sha256"]}_decOnly_train_dict_GS.npz')
                GS_feats = np.load(GS_feats_path)
                feat_GS = torch.tensor(GS_feats['feat_GS'], dtype=torch.bfloat16)
                if self.shard_data: self.feat_GS_cache[idx] = feat_GS

            if self.shard_data and idx in self.rend_err_GS_cache: rend_err_GS = self.rend_err_GS_cache[idx]
            else:
                rend_err_path = os.path.join(self.voxelGS_feats_dir, f'{batch_data["sha256"]}_decOnly_render_error_GS.npz')
                if os.path.isfile(rend_err_path):
                    rend_err_GS = np.load(rend_err_path)
                    if 'rend_err_GS' in rend_err_GS:
                        rend_err_GS = torch.tensor(rend_err_GS['rend_err_GS'], dtype=torch.bfloat16)
                        if self.shard_data: self.rend_err_GS_cache[idx] = rend_err_GS
                    else: rend_err_GS = rend_err_path
                else: rend_err_GS = rend_err_path

            new_w, new_h = self.opt.img_resolution
            h_err, w_err = new_h // 4, new_w // 4
            intrinsics_err = batch_data['rand_intrinsics_lst'].clone()
            intrinsics_err[:,0,0] /= batch_data['scale'] # *w
            intrinsics_err[:,1,1] /= batch_data['scale'] # *h
            intrinsics_err[:,0,:] *= w_err
            intrinsics_err[:,1,:] *= h_err
            depths_error = F.interpolate(batch_data['rand_depths_lst'].unsqueeze(1), size=(h_err, w_err), mode='nearest').squeeze(1)

            batch_data.update(
                intrinsics_err=intrinsics_err,
                depths_error=depths_error,
                feat_voxel=feat_voxel,
                feat_GS=feat_GS,
                rend_err_GS=rend_err_GS
            )

            return batch_data
        except: raise Exception(f'RefinerDataset error while loading {batch_data["sha256"]}')


# class DensifierDataset(Dataset):
#     def __init__(self, metadata, opt, stage='train'):
#         """
#         metadata: 包含所有样本信息的 DataFrame，每个样本至少包含 'sha256'
#         opt: 参数选项，包含 voxel_size 等参数
#         """
#         self.metadata = metadata.reset_index(drop=True)
#         self.opt = opt
        
#         self.feats_folder_name = self.opt.feats_folder_name if hasattr(self.opt, 'feats_folder_name') else 'features_0p05_correct'
#         self.feat_model_name =  self.opt.feat_model_name if hasattr(self.opt, 'feat_model_name') else 'dinov2_vitl14_reg'
#         # self.voxels_folder_name = self.opt.voxels_folder_name if hasattr(self.opt, 'voxels_folder_name') else 'voxels_0p05'
#         self.pts_folder_name = self.opt.pts_folder_name if hasattr(self.opt, 'pts_folder_name') else '1p0globalFPS_align_unipts'
        
#         voxelGS_feats_folder_name = self.opt.voxelGS_feats_folder_name if hasattr(self.opt, 'voxelGS_feats_folder_name') else 'GSandVoxel_feats'
#         self.voxelGS_feats_dir = os.path.join(self.opt.dataset_dir, self.feats_folder_name, self.feat_model_name, self.pts_folder_name, voxelGS_feats_folder_name)

#         self.stage = stage
#         if self.stage == 'train':
#             self.num_rend_views = 32 if not hasattr(self.opt.trainer, 'num_rend_views') else self.opt.trainer.num_rend_views
#         elif self.stage in ['val', 'test']:
#             self.num_rend_views = 32 
#         elif self.stage == 'test_trainset_novel':
#             self.num_rend_views = 32 
#             self.test_frames_folder_name = 'test_frames_rgb_depth_v14_fix32'
#         else:
#             raise ValueError(f"Unknown stage: {self.stage}")

#     def __len__(self):
#         return len(self.metadata)

#     def is_valid_depth(self, depth, valid_ratio_thresh=0.6):
#         # depth: [H, W] tensor or numpy array
#         if isinstance(depth, torch.Tensor):
#             valid_mask = (depth > 0) & torch.isfinite(depth)
#             return valid_mask.sum().item() / depth.numel() >= valid_ratio_thresh
#         else:
#             valid_mask = (depth > 0) & np.isfinite(depth)
#             return valid_mask.sum() / depth.size >= valid_ratio_thresh
    
#     @staticmethod
#     def get_extrinsics_for_samples(view, T_norm=None):
#         c2w = torch.tensor(view['transform_matrix'])
#         c2w[2, :] *= -1
#         c2w = c2w[np.array([1,0,2,3]), :]
#         c2w[:3, 1:3] *= -1
#         w2c = torch.inverse(c2w)
#         w2c_Tnorm = w2c @ torch.inverse(T_norm) if T_norm is not None else w2c
#         return w2c_Tnorm

#     def __getitem__(self, idx):
#         # device = torch.device(f"cuda:{torch.cuda.current_device()}")
        
#         new_w, new_h = self.opt.img_resolution
#         sha256 = self.metadata.loc[idx, 'sha256']
#         # 加载dino特征
#         try:
            
#             voxel_feats = np.load(os.path.join(self.voxelGS_feats_dir, f'{sha256}_decOnly_dict_voxel.npz'))
#             GS_feats = np.load(os.path.join(self.voxelGS_feats_dir, f'{sha256}_decOnly_train_dict_GS.npz'))
#             feat_voxel = torch.tensor(voxel_feats['feat_voxel'])
#             feat_GS = torch.tensor(GS_feats['feat_GS'])
#             # grad_GS = torch.tensor(GS_feats['grad_GS'])         
            
#             viewInfo = np.load(os.path.join(self.opt.dataset_dir, 'fix32_viewInfos', f'{sha256}_viewInfo.npz'))

#             all_extrinsics_lst = torch.tensor(viewInfo['extrinsics_Tnorm_norm128_v14_fix32'])
#             all_depths_lst = torch.tensor(viewInfo['depthmap_1080x1620_norm128_v14_fix32']) # [N, H, W]
#             all_intrinsics_lst = torch.tensor(viewInfo['normed_intrinsics_norm128_v14_fix32'])
#             T_norm = torch.tensor(viewInfo['T_norm'])
#             SCALE = viewInfo['scale'].item()

#             batch_data = {"sha256": sha256, 
#                           "scale": SCALE,
#                           "T_norm": T_norm,
#                           "feat_GS": feat_GS,
#                           "feat_voxel": feat_voxel
#                           }
            
#             if os.path.exists(os.path.join(self.voxelGS_feats_dir, f'{sha256}_decOnly_render_error_GS.npz')):
#                 grad_GS = np.load(os.path.join(self.voxelGS_feats_dir, f'{sha256}_decOnly_render_error_GS.npz'))['grad_GS']       
#                 batch_data.update({"grad_GS": grad_GS})

#             ori_img_path = os.path.join(self.opt.dataset_dir, self.feats_folder_name, self.feat_model_name, f'{sha256}_img_feats_oriImgs.npy')
#             all_images_lst = torch.tensor(np.load(ori_img_path)).float().cpu() / 255. # [N, 3, H, W]
#             if all_images_lst.shape[-2:] != (new_h, new_w):
#                 all_images_lst = F.interpolate(all_images_lst, size=(new_h, new_w), mode='bilinear', align_corners=False)
#             if all_depths_lst.shape[-2:] != (new_h, new_w):
#                 all_depths_lst = F.interpolate(all_depths_lst.unsqueeze(1), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(1)
            
#             assert all_images_lst.shape[0] == all_extrinsics_lst.shape[0] == all_intrinsics_lst.shape[0] == all_depths_lst.shape[0], \
#                 f"Shape mismatch: images {all_images_lst.shape}, extrinsics {all_extrinsics_lst.shape}, intrinsics {all_intrinsics_lst.shape}, depths {all_depths_lst.shape}"

#             # if self.num_rend_views < all_images_lst.shape[0]:
#             if self.stage == 'train': # 采样 num_rend_views 个有效视角
#                 rand_indices = torch.randperm(all_images_lst.shape[0])[:self.num_rend_views]
#                 all_images_lst = all_images_lst[rand_indices]
#                 all_extrinsics_lst = all_extrinsics_lst[rand_indices]
#                 all_intrinsics_lst = all_intrinsics_lst[rand_indices]
#                 all_depths_lst = all_depths_lst[rand_indices]
            
#             batch_data.update( {
#                                 "rand_img_lst":all_images_lst, 
#                                 "rand_extrinsics_lst": all_extrinsics_lst, 
#                                 "rand_intrinsics_lst": all_intrinsics_lst, 
#                                 "rand_depths_lst": all_depths_lst,
#                         })
            
#             # test 测试novel view
#             if self.stage in ['test_trainset_novel']:  # TODO: 需要修改
                
#                 train_frames = cam_infos['frames'] # [::4]
#                 test_frames = cam_infos['test_frames'] # [::4]
#                 print(f"train frames: {len(train_frames)}, test_frames: {len(test_frames)}")

#                 test_frames_infos = np.load(os.path.join(self.opt.dataset_dir, self.opt.test_frames_folder_name, f'{sha256}_test_frames.npz'))
#                 novel_img_lst = torch.tensor(test_frames_infos['normed_intrinsics'])
#                 novel_ext_lst = torch.tensor(test_frames_infos['extrinsics_Tnorm'])
#                 novel_img_lst = torch.tensor(test_frames_infos['images'])
#                 novel_depth_lst = torch.tensor(test_frames_infos['depthmap'])
                
#                 if novel_img_lst.shape[-2:] != (new_h, new_w):
#                     novel_img_lst = F.interpolate(novel_img_lst, size=(new_h, new_w), mode='bilinear', align_corners=False)
#                 if novel_depth_lst.shape[-2:] != (new_h, new_w):
#                     novel_depth_lst = F.interpolate(novel_depth_lst, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    
#                 novel_int_lst = intrinsics.unsqueeze(0).repeat(novel_img_lst.shape[0], 1, 1).cpu()
#                 # print(f"[INFO] For {sha256}, sampled {len(sampled_views)} novel views.")
#                 # print(f"novel_img_lst: {novel_img_lst.shape}, novel_int_lst: {novel_int_lst.shape}, novel_ext_lst: {novel_ext_lst.shape}")

#                 batch_data.update({
#                                     "novel_img_lst": novel_img_lst, 
#                                     "novel_depths_lst": novel_depth_lst,
#                                     "novel_extrinsics_lst": novel_ext_lst, 
#                                     "novel_intrinsics_lst": novel_int_lst, 
#                             })

#             return batch_data
                
#         except Exception as e:
#             raise ValueError(f"Error loading renders images or extrinsics or intrinsics for {sha256}: {e}")

# class VAEDataset(Dataset):
#     def __init__(self, metadata, opt, stage='train'):
#         """
#         metadata: 包含所有样本信息的 DataFrame，每个样本至少包含 'sha256'
#         opt: 参数选项，包含 voxel_size 等参数
#         """
#         self.metadata = metadata.reset_index(drop=True)
#         self.opt = opt
#         self.feats_folder_name = self.opt.feats_folder_name if hasattr(self.opt, 'feats_folder_name') else 'features_0p05_correct'
#         self.feat_model_name =  self.opt.feat_model_name if hasattr(self.opt, 'feat_model_name') else 'dinov2_vitl14_reg'
#         # self.voxels_folder_name = self.opt.voxels_folder_name if hasattr(self.opt, 'voxels_folder_name') else 'voxels_0p05'
#         self.pts_folder_name = self.opt.pts_folder_name if hasattr(self.opt, 'pts_folder_name') else 'irregular_pts_1p0globalFPS'

#         self.stage = stage
        
#         self.agg_feats_minval = -8.5
#         self.agg_feats_maxval = 8.5

#     def __len__(self):
#         return len(self.metadata)

#     def is_valid_depth(self, depth, valid_ratio_thresh=0.6):
#         # depth: [H, W] tensor or numpy array
#         if isinstance(depth, torch.Tensor):
#             valid_mask = (depth > 0) & torch.isfinite(depth)
#             return valid_mask.sum().item() / depth.numel() >= valid_ratio_thresh
#         else:
#             valid_mask = (depth > 0) & np.isfinite(depth)
#             return valid_mask.sum() / depth.size >= valid_ratio_thresh
    
#     @staticmethod
#     def get_extrinsics_for_samples(view, T_norm=None):
#         c2w = torch.tensor(view['transform_matrix'])
#         c2w[2, :] *= -1
#         c2w = c2w[np.array([1,0,2,3]), :]
#         c2w[:3, 1:3] *= -1
#         w2c = torch.inverse(c2w)
#         w2c_Tnorm = w2c @ torch.inverse(T_norm) if T_norm is not None else w2c
#         return w2c_Tnorm

#     def __getitem__(self, idx):
#         # device = torch.device(f"cuda:{torch.cuda.current_device()}")
        
#         new_w, new_h = self.opt.img_resolution
#         sha256 = self.metadata.loc[idx, 'sha256']
#         # 加载dino特征
#         try:
#             pts_feats_path = os.path.join(self.opt.dataset_dir, self.feats_folder_name, self.feat_model_name, self.pts_folder_name, f'{sha256}_rgbcnn_aggfeats.npz')
#             pts_feats = np.load(pts_feats_path)
#             voxel_points = torch.tensor(pts_feats['positions']).float().cpu()
#             agg_feats = torch.tensor(pts_feats['agg_feats']).float().cpu()
            
#             agg_feats_normalized =  (agg_feats - self.agg_feats_minval) / (self.agg_feats_maxval - self.agg_feats_minval)
            
#             batch_data = {"sha256": sha256,
#                           "voxel_points": voxel_points, 
#                           "agg_feats": agg_feats_normalized,
#                           "agg_feats_minval": self.agg_feats_minval,
#                           "agg_feats_maxval": self.agg_feats_maxval
#                           }
            
            
#             if self.stage != 'train':
#                 # 只在 val/test 时加载
#                 # 读取 renders 的图像和相机参数                
#                 viewInfo_path = os.path.join(self.opt.dataset_dir, 'fix32_viewInfos', f'{sha256}_viewInfo.npz')
#                 viewInfos = np.load(viewInfo_path)
                
#                 all_extrinsics_lst = torch.tensor(viewInfos['extrinsics_Tnorm_norm128_v14_fix32']).float().cpu()
#                 all_intrinsics_lst = torch.tensor(viewInfos['normed_intrinsics_norm128_v14_fix32']).float().cpu()
#                 SCALE = torch.tensor(viewInfos['scale']).float().cpu()
#                 T_norm = torch.tensor(viewInfos['T_norm']).float().cpu()
#                 all_depths_lst = torch.tensor(viewInfos['depthmap_1080x1620_norm128_v14_fix32']).float().cpu()
#                 all_depths_lst = all_depths_lst.unsqueeze(1)
#                 all_depths_lst = F.interpolate(all_depths_lst, size=(new_h, new_w), mode='bilinear', align_corners=False)
#                 all_depths_lst = all_depths_lst.squeeze(1) # [N, H, W]
                
#                 rgbs_path = os.path.join(self.opt.dataset_dir, self.feats_folder_name, self.feat_model_name, 'rgb_feats', f'{sha256}_img_feats.pt')                
#                 rgbfeats = torch.load(rgbs_path, map_location='cpu')
#                 all_imgs_lst = rgbfeats['ori_rgbImg_1168x1752'].float().cpu()
#                 all_imgs_lst = F.interpolate(all_imgs_lst, size=(new_h, new_w), mode='bilinear', align_corners=False)                
                
#                 batch_data.update( {
#                                     "rand_img_lst": all_imgs_lst, 
#                                     "rand_extrinsics_lst": all_extrinsics_lst, 
#                                     "rand_intrinsics_lst": all_intrinsics_lst, 
#                                     "rand_depths_lst": all_depths_lst,
#                                     "scale": SCALE,
#                                     "T_norm": T_norm,
#                             })
                    
#             return batch_data
                
#         except Exception as e:
#             raise ValueError(f"Error loading renders images or extrinsics or intrinsics for {sha256}: {e}")
