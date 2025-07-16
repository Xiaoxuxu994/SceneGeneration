import torch
import torch_npu
import torch.nn.functional as F
from torch import nn
# from torch_cluster import fps
import math
# from torch_cluster import fps
from src.registry import MODELS, build_module
from src.models.transformer.transformer_base import PointEmbed,PreNorm,Attention,FeedForward
from util import sample_farthest_points
import torch.distributed as dist

from ...registry import MODELS
def log_memory_usage(info=''):
    print(f"当前显存占用: {torch_npu.npu.memory_allocated()/1024**2:.2f}MB / {torch_npu.npu.max_memory_allocated()/1024**2:.2f}MB, {info}, {dist.get_rank()}")

def isin(elements, test_elements):
    # 需要先对 test_elements 排序
    test_elements_sorted, _ = torch.sort(test_elements.unique())
    idx = torch.searchsorted(test_elements_sorted, elements)
    idx[idx == len(test_elements_sorted)] = 0
    return test_elements_sorted[idx] == elements

@MODELS.register_module()
class VecSetAEV4CTF(nn.Module):
    def __init__(self, 
                input_pcd_num = 50000,
                num_latent = 512,
                latent_dim = 512,
                use_cpu_fps=False,
                use_skip = False,
                encoder_dict = dict(
                    type = "PointTransformerV3_2",
                    in_channels = 3, # ['coord]
                    enable_flash=True,
                    attn_dtype='fp16',
                    enable_ca_flash=False,
                    drop_path=0.3,
                    cls_mode=False, #True for enc only; False for enc-dec backbone
                ),
                query_encoder_dict = dict(
                    type = "PointTransformerV3_2",
                    in_channels = 3, # ['coord]
                    enable_flash=True,
                    attn_dtype='fp16',
                    enable_ca_flash=False,
                    drop_path=0.3,
                    cls_mode=False, #True for enc only; False for enc-dec backbone
                ),
                decoder_dict1 = dict(
                    type = "VecSetDecoder", #['VecSetDecoder','VecSetKLDecoder']
                    depth = 6,
                    dim = 512,
                    queries_dim = 512,
                    enable_flash=True,
                    attn_dtype='fp16',
                    enable_ca_flash=False,
                    output_dim = 1,
                    # deterministic = True, # ['Auto' or 'KL']
                ),
                decoder_dict2 = dict(
                    type = "VecSetDecoder", #['VecSetDecoder','VecSetKLDecoder']
                    depth = 6,
                    dim = 512,
                    queries_dim = 512,
                    enable_flash=True,
                    attn_dtype='fp16',
                    enable_ca_flash=False,
                    output_dim = 1,
                    # deterministic = True, # ['Auto' or 'KL']
                ),
                decoder_dict3 = dict(
                    type = "VecSetDecoder", #['VecSetDecoder','VecSetKLDecoder']
                    depth = 6,
                    dim = 512,
                    queries_dim = 512,
                    enable_flash=True,
                    attn_dtype='fp16',
                    enable_ca_flash=False,
                    output_dim = 1,
                    # deterministic = True, # ['Auto' or 'KL']
                ),
                decoder_dict4 = dict(
                    type = "VecSetDecoder", #['VecSetDecoder','VecSetKLDecoder']
                    depth = 6,
                    dim = 512,
                    queries_dim = 512,
                    enable_flash=True,
                    attn_dtype='fp16',
                    enable_ca_flash=False,
                    output_dim = 1,
                    # deterministic = True, # ['Auto' or 'KL']
                ),
    ):
        
        super().__init__()
        
        self.input_pcd_num = input_pcd_num
        self.num_latent = num_latent
        self.latent_dim = latent_dim

        self.encoder_dict = encoder_dict
        self.query_encoder_dict = query_encoder_dict
        self.decoder_dict1 = decoder_dict1
        self.decoder_dict2 = decoder_dict2
        self.decoder_dict3 = decoder_dict3
        self.decoder_dict4 = decoder_dict4
        self.use_skip = use_skip
        self.use_cpu_fps = use_cpu_fps
        if use_cpu_fps:
            print('Using fps_cpu kernel...')

        ## ptv3 encoder
        if self.encoder_dict.type == 'PointTransformerV3' or 'PointTransformerV3_2':
            self.encoder = build_module(self.encoder_dict, MODELS)     

        ## query encoder
        if self.query_encoder_dict.type == 'PointTransformerV3' or 'PointTransformerV3_2':
            self.query_encoder = build_module(self.query_encoder_dict, MODELS)   

        ## encoder cross attention
        if not self.encoder_dict.enable_ca_flash:
            self.cross_attend_blocks = nn.ModuleList([
                PreNorm(self.latent_dim, Attention(self.latent_dim, self.latent_dim, heads = 1, dim_head = self.latent_dim), context_dim = self.latent_dim),
                PreNorm(self.latent_dim, FeedForward(self.latent_dim))
            ])
        else:
            self.cross_attend_blocks = nn.ModuleList([
                PreNorm(self.latent_dim, Attention(self.latent_dim, self.latent_dim, heads = 32, dim_head = self.latent_dim // 32), context_dim = self.latent_dim),
                PreNorm(self.latent_dim, FeedForward(self.latent_dim))
            ])            

        ## decoder
        self.decoder1 = build_module(self.decoder_dict1, MODELS)
        self.decoder2 = build_module(self.decoder_dict2, MODELS)
        self.decoder3 = build_module(self.decoder_dict3, MODELS)
        self.decoder4 = build_module(self.decoder_dict3, MODELS)

    # def pc_fps(self,pc):
    #     # input point cloud: B x N x 3
    #     # use fps to get the initial learnable query points
    #     pc = pc.unsqueeze(0) # b=1
    #     B, N, D = pc.shape
    #     # assert N == self.input_pcd_num    
    #     self.input_pcd_num = N    ### !!!!!!!!!!!!!!!!!! 
    #     # fps
    #     flattened = pc.view(B * N, D)
    #     batch = torch.arange(B).to(pc.device)
    #     batch = torch.repeat_interleave(batch, N)

    #     pos = flattened
    #     ratio = 1.0 * self.num_latent / self.input_pcd_num
    #     # idx = fps(pos, batch, ratio=ratio,random_start = self.encoder.training)
    #     idx = fps(pos, batch, ratio=ratio)

    #     sampled_pc = pos[idx]
    #     sampled_pc = sampled_pc.view(B, -1, 3)
    #     return idx, sampled_pc

    def pc_fps(self,pc):
        # input point cloud: B x N x 3
        # use fps to get the initial learnable query points
        pc = pc.unsqueeze(0) # b=1
        B, N, D = pc.shape
        # assert N == self.input_pcd_num    
        self.input_pcd_num = N    ### !!!!!!!!!!!!!!!!!! 
        # fps
        flattened = pc.view(B * N, D)
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened
        ratio = 1.0 * self.num_latent / self.input_pcd_num
        # idx = fps(pos, batch, ratio=ratio,random_start = self.encoder.training)
        # idx = fps(pos, batch, ratio=ratio)
        # 120000为经验数值。最终版本应不添加此限制。
        if self.use_cpu_fps and N <= 120000:
            idx = sample_farthest_points(pc.cpu(), lengths=torch.tensor([N], dtype=torch.long), K=self.num_latent, random_start_point=True).view(-1).to(pc.device)
        else:
            idx = self.fps_fallback(pos, batch, ratio=ratio, random_start=True)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        return idx, sampled_pc
    
    def fps_fallback(self, x, batch=None, ratio=0.25, random_start=True):
        """
        PyTorch fallback version of torch_cluster.fps (Furthest Point Sampling)

        Args:
            x (Tensor): (N, D) point positions
            batch (LongTensor): (N,) batch indices or None
            ratio (float): sampling ratio ∈ (0, 1]
            random_start (bool): whether to randomly initialize the first point

        Returns:
            Tensor: (M,) indices of sampled points (subset of input x)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        batch_size = batch.max().item() + 1
        output_idx = []

        for b in range(batch_size):
            mask = batch == b
            pts = x[mask]
            N = pts.size(0)
            M = max(1, math.ceil(N * ratio))  # 至少采样1个点,follow torch_cluster, 向上取整

            centroids = torch.zeros(M, dtype=torch.long, device=x.device)
            distance = torch.full((N,), 1e10, device=x.device)
            if random_start:
                farthest = torch.randint(0, N, (1,), device=x.device).item()
            else:
                farthest = 0

            for i in range(M):   
                centroids[i] = farthest
                centroid = pts[farthest].unsqueeze(0)
                dist = torch.sum((pts - centroid) ** 2, dim=1)
                mask_dist = dist < distance
                distance[mask_dist] = dist[mask_dist]
                farthest = torch.argmax(distance).item()

            batch_indices = torch.where(mask)[0][centroids]  # 恢复原始索引
            output_idx.append(batch_indices)

        return torch.cat(output_idx, dim=0)


    def hash_coords(self,coords, resolution):
        return coords[:, 0] * resolution * resolution + coords[:, 1] * resolution + coords[:, 2]

    def get_fine_voxel_mask_from_coarse_labels(
        self, coarse_label, scale_factor, resolution_fine, coarse_grid, fine_grid
    ):
        device = coarse_label.device
        active_coarse = coarse_grid[coarse_label.bool()]
        if active_coarse.shape[0] == 0:
            return torch.zeros((fine_grid.shape[0],), dtype=torch.bool, device=device)

        offsets = torch.tensor([[i, j, k] for i in range(scale_factor) for j in range(scale_factor) for k in range(scale_factor)],
                            dtype=torch.long, device=device)

        base = active_coarse * scale_factor
        fine_candidates = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 3)

        fine_hash_all = self.hash_coords(fine_grid, resolution_fine)
        fine_candidate_hash = self.hash_coords(fine_candidates, resolution_fine)
        return isin(fine_hash_all, fine_candidate_hash)

    def logit2label(self, logit, threshold=0.5):
        """
        logit: Tensor of shape (1, N) or (1, N, 1)
        return: Float tensor of shape (N,), value 0.0 or 1.0
        """
        logit = logit.squeeze(-1).squeeze(0)           
        prob = torch.sigmoid(logit)         
        return (prob >= threshold)#.float() 
    
    def encode(self, input_dict):
        
        ## -- pcd encoder
        if self.encoder_dict.type == 'PointTransformerV3' or 'PointTransformerV3_2':
            # pcd ptv3 encoder
            fea_enc = self.encoder(input_dict)
            pc_embeddings = fea_enc['feat'].unsqueeze(0)
        else:
            ## pcd vecset encoder
            pc_embeddings = self.point_embed(input_dict['coord'].unsqueeze(0))

        ## -- query encoder
        if self.query_encoder_dict.type == 'None':
            ## directly use the pcd encoder feature
            latent =  pc_embeddings
        else:
            if input_dict['data_type'] == 2: # 限定了 PointTransformerV3_2 + cls_mode = True
                #if self.query_encoder_dict.type == 'VecSet':
                #   sampled_pc_embeddings =  self.point_embed(sampled_pc) # [1,num_latent,512]
                # if self.query_encoder_dict.type == 'PointTransformerV3' or 'PointTransformerV3_2':
                #     if self.encoder_dict.cls_mode == True:
      
                # ptvs enc only
                sample_pc_dict = input_dict.copy()
                sample_pc_dict["coord"] = input_dict['fps_coord']
                sample_pc_dict["grid_coord"] = input_dict['fps_grid_coord']
                sample_pc_dict["feat"] = input_dict['fps_feat']
                sample_pc_dict['offset'] = torch.tensor([self.num_latent], device=pc_embeddings.device)
                sample_fea_enc = self.query_encoder(sample_pc_dict)
                sampled_pc_embeddings = sample_fea_enc['feat'].unsqueeze(0)    
            else:
                if input_dict['data_type'] == 0:
                    idx, sampled_pc = self.pc_fps(input_dict['coord'])

                if input_dict['data_type'] == 1:
                    sampled_pc = input_dict['coord_fps']
                    idx = input_dict['fps_index']

                if input_dict['data_type'] == 3:
                    idx = input_dict['fps_index']
                    sampled_pc = input_dict['coord'][idx]

                if self.query_encoder_dict.type == 'VecSet':
                    sampled_pc_embeddings =  self.point_embed(sampled_pc) # [1,num_latent,512]

                if self.query_encoder_dict.type == 'PointTransformerV3' or 'PointTransformerV3_2':
                    if self.encoder_dict.cls_mode == True:
                        # ptvs enc only
                        sample_pc_dict = input_dict.copy()
                        if "coord" in input_dict.keys():
                            sample_pc_dict["coord"] = input_dict["coord"][idx]
                        if "grid_coord" in input_dict.keys():
                            sample_pc_dict["grid_coord"] = input_dict["grid_coord"][idx]
                        if "segment" in input_dict.keys():
                            sample_pc_dict["segment"] = input_dict["segment"][idx]
                        if "ptv3_embedding" in input_dict.keys():
                            sample_pc_dict["ptv3_embedding"] = input_dict["ptv3_embedding"][idx]
                        if "feat" in input_dict.keys():
                            sample_pc_dict["feat"] = input_dict["feat"][idx]
                        sample_pc_dict['offset'] = torch.tensor([self.num_latent]).to(pc_embeddings.device) # updata pcd offset

                        sample_fea_enc = self.query_encoder(sample_pc_dict)
                        sampled_pc_embeddings = sample_fea_enc['feat'].unsqueeze(0)
                    else:
                        # ptv3 full backbone
                        sampled_pc_embeddings = pc_embeddings[:,idx,:]
                        # sampled_pc_embeddings = pc_embeddings.half()[idx].unsqueeze(0)
                
            ## cross attn block
            cross_attn, cross_ff = self.cross_attend_blocks
            latent= cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None, enable_flash = self.encoder_dict.enable_ca_flash) + sampled_pc_embeddings
            latent = cross_ff(latent) + latent

        return latent

    # def _get_level_inputs(self, input_dict, scale, mask=None):
    #     query_key = f'pc_query_{scale}'
    #     label_key = f'pc_query_label_fine_prop_{scale}'
    #     grid_key = f'pc_query_grid_{scale}'

    #     query = input_dict[query_key]
    #     label = input_dict[label_key]
    #     grid = input_dict[grid_key]

    #     if mask is not None:
    #         query = query[mask]
    #         label = label[mask]
    #         grid = grid[mask]

    #     return query, label, grid


    # def coarse_to_fine_decode(self, latent, input_dict, scale_list, 
    #                           decoder_list, scale_factor):
    #     logit_list, targets_list = [], []

    #     prev_mask = None
    #     prev_grid = None

    #     for i, s in enumerate(scale_list):
    #         decoder = decoder_list[i]

    #         # 获取当前层数据
    #         query_pts, target, grid = self._get_level_inputs(input_dict, s, prev_mask)

    #         # Decoder forward
    #         logit, latent = decoder(latent, query_pts.unsqueeze(0))  # logit: (1, N)

    #         logit_list.append(logit)
    #         targets_list.append(target)

    #         # coarse prediction + GT union（仅用于训练）
    #         if self.training:
    #             pred_label = self.logit2label(logit, 0.5)
    #             label_c = target.bool() | pred_label
    #         else:
    #             label_c = self.logit2label(logit, 0.5)

    #         # 不是最后一层 → 生成 mask for finer resolution
    #         if i < len(scale_list) - 1:
    #             fine_s = scale_list[i + 1]
    #             fine_grid = input_dict[f'pc_query_grid_{fine_s}']
    #             mask = self.get_fine_voxel_mask_from_coarse_labels(
    #                 coarse_label=label_c,
    #                 scale_factor=scale_factor,
    #                 resolution_fine=fine_s,
    #                 coarse_grid=grid,
    #                 fine_grid=fine_grid
    #             )
    #             prev_mask = mask
    #             prev_grid = fine_grid

    #     return logit_list, targets_list


    # def forward(self, input_dict):
    #     latent = self.encode(input_dict)

    #     scale_list = [16, 32, 64, 128]
    #     decoder_list = [self.decoder1, self.decoder2, self.decoder3, self.decoder3]
    #     scale_factor = 2

    #     return self.coarse_to_fine_decode(
    #         latent, input_dict, scale_list, decoder_list, scale_factor
    #     )


    def forward(self, input_dict,return_mask=False):
        # encoder
        # log_memory_usage('Before encode')
        latent = self.encode(input_dict)
        # log_memory_usage('After encode')
        ### Coarse to fine decoder
        logit_list,targets_list,mask_list= [],[],[None,]
        # ------------------------------------------------------------------------
        ## -- vs16 --
        query_points_16 = input_dict['pc_query_16'].unsqueeze(0)
        # print('query_points_16: ',query_points_16.shape)
        target_16 = input_dict['pc_query_label_fine_prop_16']
        grid_16 = input_dict['pc_query_grid_16']
        # log_memory_usage('Before decode1')
        logit_16,latent_16 = self.decoder1(latent,query_points_16)
        # log_memory_usage('After decode1')
        label_16 = self.logit2label(logit_16,0.5)
        if self.training:
            label_c = target_16.bool() | label_16
        else:
            label_c = label_16

        #- mask for next level
        mask = self.get_fine_voxel_mask_from_coarse_labels(
            coarse_label=label_c,
            scale_factor=2,
            resolution_fine=32,
            coarse_grid=grid_16,
            fine_grid=input_dict['pc_query_grid_32']
        )
        # log_memory_usage('After mask generation 1')
        #- current pred-gt pair
        logit_list.append(logit_16.squeeze(-1).squeeze(0))
        targets_list.append(target_16)
        mask_list.append(mask)
        # ------------------------------------------------------------------------
        ## -- vs32 --
        query_points_32 = input_dict['pc_query_32'][mask].unsqueeze(0)
        # print('query_points_32: ',query_points_32.shape)
        target_32 = input_dict['pc_query_label_fine_prop_32'][mask]
        grid_32 = input_dict['pc_query_grid_32'][mask]

        # 控制 skip connection 的输入
        latent_input = latent + latent_16 if self.use_skip else latent_16
        # log_memory_usage('Before decode2')
        logit_32,latent_32 = self.decoder2(latent_input,query_points_32)
        # log_memory_usage('After decode2')
        label_32 = self.logit2label(logit_32,0.5)
        if self.training:
            label_c = target_32.bool() | label_32
        else:
            label_c = label_32

        #- mask for next level
        mask = self.get_fine_voxel_mask_from_coarse_labels(
            coarse_label=label_c,
            scale_factor=2,
            resolution_fine=64,
            coarse_grid=grid_32,
            fine_grid=input_dict['pc_query_grid_64']
        )
        # log_memory_usage('After mask generation 2')
        #- current pred-gt pair
        logit_list.append(logit_32.squeeze(-1).squeeze(0))
        targets_list.append(target_32)
        mask_list.append(mask)
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        ## -- vs64 --
        query_points_64 = input_dict['pc_query_64'][mask].unsqueeze(0)
        # print('query_points_64: ',query_points_64.shape)
        target_64 = input_dict['pc_query_label_fine_prop_64'][mask]
        grid_64 = input_dict['pc_query_grid_64'][mask]

        latent_input = latent_16 + latent_32 if self.use_skip else latent_32
        # log_memory_usage('Before decode3')
        logit_64,latent_64 = self.decoder3(latent_input,query_points_64)
        # log_memory_usage('After decode3')
        label_64 = self.logit2label(logit_64,0.5)
        
        if self.training:
            label_c = target_64.bool() | label_64
        else:
            label_c = label_64

        #- mask for next level
        # if dist.get_rank() == 0:
        #     import pdb; pdb.set_trace()
        # else:
        #     import time; time.sleep(10000)
        mask = self.get_fine_voxel_mask_from_coarse_labels(
            coarse_label=label_c,
            scale_factor=2,
            resolution_fine=128,
            coarse_grid=grid_64,
            fine_grid=input_dict['pc_query_grid_128']
        )
        #- current pred-gt pair
        # log_memory_usage('After mask generation 3')
        logit_list.append(logit_64.squeeze(-1).squeeze(0))
        targets_list.append(target_64)
        mask_list.append(mask)
        # ------------------------------------------------------------------------
        # ------------------------------------------------------------------------
        ## -- vs128 --
        query_points_128 = input_dict['pc_query_128'][mask].unsqueeze(0)
        # print('query_points_128: ',query_points_128.shape)
        target_128 = input_dict['pc_query_label_128'][mask]
        latent_input = latent_32 + latent_64 if self.use_skip else latent_64
        # print(query_points_128.shape, mask.shape, '********', torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(), torch.cuda.max_memory_cached())
        # log_memory_usage('Before decode4')
        logit_128,latent_128 = self.decoder4(latent_input,query_points_128)
        # log_memory_usage('After decode4')
        #- current pred-gt pair
        logit_list.append(logit_128.squeeze(-1).squeeze(0))
        targets_list.append(target_128)
        mask_list.append(mask)
        # ------------------------------------------------------------------------       
        if return_mask:
            return logit_list,targets_list,mask_list
        else:
            return logit_list,targets_list
    
    def forward_gt(self, input_dict, return_mask=False):
        # encoder
        latent = self.encode(input_dict)

        ### Coarse to fine decoder
        logit_list,targets_list,mask_list= [],[],[None,]
        # ------------------------------------------------------------------------
        ## -- vs16 --
        query_points_16 = input_dict['pc_query_16'].unsqueeze(0)
        # print('query_points_16: ',query_points_16.shape)
        target_16 = input_dict['pc_query_label_fine_prop_16']
        grid_16 = input_dict['pc_query_grid_16']

        logit_16,latent_16 = self.decoder1(latent,query_points_16)
        label_16 = self.logit2label(logit_16,0.5)
        # if self.training:
        #     label_c = target_16.bool() #| label_16
        # else:
        #     label_c = label_16
        label_c = target_16.bool()

        #- mask for next level
        mask = self.get_fine_voxel_mask_from_coarse_labels(
            coarse_label=label_c,
            scale_factor=2,
            resolution_fine=32,
            coarse_grid=grid_16,
            fine_grid=input_dict['pc_query_grid_32']
        )
        #- current pred-gt pair
        logit_list.append(logit_16.squeeze(-1).squeeze(0))
        targets_list.append(target_16)
        mask_list.append(mask)
        # ------------------------------------------------------------------------
        ## -- vs32 --
        query_points_32 = input_dict['pc_query_32'][mask].unsqueeze(0)
        # print('query_points_32: ',query_points_32.shape)
        target_32 = input_dict['pc_query_label_fine_prop_32'][mask]
        grid_32 = input_dict['pc_query_grid_32'][mask]

        # 控制 skip connection 的输入
        latent_input = latent + latent_16 if self.use_skip else latent_16
        
        logit_32,latent_32 = self.decoder2(latent_input,query_points_32)
        label_32 = self.logit2label(logit_32,0.5)
        # if self.training:
        #     label_c = target_32.bool() #| label_32
        # else:
        #     label_c = label_32
        label_c = target_32.bool()
        # print('level32: ',label_c.sum())
        #- mask for next level
        mask = self.get_fine_voxel_mask_from_coarse_labels(
            coarse_label=label_c,
            scale_factor=2,
            resolution_fine=64,
            coarse_grid=grid_32,
            fine_grid=input_dict['pc_query_grid_64']
        )
        #- current pred-gt pair
        logit_list.append(logit_32.squeeze(-1).squeeze(0))
        targets_list.append(target_32)
        mask_list.append(mask)
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        ## -- vs64 --
        query_points_64 = input_dict['pc_query_64'][mask].unsqueeze(0)
        # print('query_points_64: ',query_points_64.shape)
        target_64 = input_dict['pc_query_label_fine_prop_64'][mask]
        grid_64 = input_dict['pc_query_grid_64'][mask]

        latent_input = latent_16 + latent_32 if self.use_skip else latent_32
        logit_64,latent_64 = self.decoder3(latent_input,query_points_64)
        label_64 = self.logit2label(logit_64,0.5)
        
        # if self.training:
        #     label_c = target_64.bool() #| label_64
        # else:
        #     label_c = label_64
        label_c = target_64.bool()
        # print('level64: ',label_c.sum())
        #- mask for next level
        mask = self.get_fine_voxel_mask_from_coarse_labels(
            coarse_label=label_c,
            scale_factor=2,
            resolution_fine=128,
            coarse_grid=grid_64,
            fine_grid=input_dict['pc_query_grid_128']
        )
        #- current pred-gt pair
        logit_list.append(logit_64.squeeze(-1).squeeze(0))
        targets_list.append(target_64)
        mask_list.append(mask)
        # ------------------------------------------------------------------------
        # ------------------------------------------------------------------------
        ## -- vs128 --
        # print()
        query_points_128 = input_dict['pc_query_128'][mask].unsqueeze(0)
        # print('query_points_128: ',query_points_128.shape)
        target_128 = input_dict['pc_query_label_128'][mask]
        
        latent_input = latent_32 + latent_64 if self.use_skip else latent_64

        logit_128,latent_128 = self.decoder4(latent_input,query_points_128)
        #- current pred-gt pair
        logit_list.append(logit_128.squeeze(-1).squeeze(0))
        targets_list.append(target_128)
        # ------------------------------------------------------------------------       
        if return_mask:
            return logit_list,targets_list,mask_list
        else:
            return logit_list,targets_list
    
