import torch
import torch.nn.functional as F
from torch import nn
# from torch_cluster import fps
import math
# from torch_cluster import fps
from src.registry import MODELS, build_module
from src.models.transformer.transformer_base import PointEmbed,PreNorm,Attention,FeedForward
from util import sample_farthest_points

from ...registry import MODELS
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

        self.mode = 'coarse_to_fine'

        ## ptv3 encoder
        if self.encoder_dict.type in ['PointTransformerV3', 'PointTransformerV3_2']:
            self.encoder = build_module(self.encoder_dict, MODELS) 
        elif self.encoder_dict.type == "PE":
            self.encoder = PointEmbed(dim=self.latent_dim)
        else:
            raise ValueError(f"Unknown mode: {self.encoder_dict.type}")


        ## query encoder
        if self.query_encoder_dict.type in ['PointTransformerV3', 'PointTransformerV3_2']:
            self.query_encoder = build_module(self.query_encoder_dict, MODELS)   
        elif self.query_encoder_dict.type == "PE":
            self.query_encoder = PointEmbed(dim=self.latent_dim)
        else:
            raise ValueError(f"Unknown mode: {self.query_encoder_dict.type}")
        
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
        self.decoder1 = build_module(self.decoder_dict1, MODELS) if self.decoder_dict1 else None
        self.decoder2 = build_module(self.decoder_dict2, MODELS) if self.decoder_dict2 else None
        self.decoder3 = build_module(self.decoder_dict3, MODELS) if self.decoder_dict3 else None
        self.decoder4 = build_module(self.decoder_dict4, MODELS) if self.decoder_dict4 else None

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

    def set_mode(self, mode):
        assert mode in ['gt', 'coarse_to_fine']
        self.mode = mode

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
        return torch.isin(fine_hash_all, fine_candidate_hash)

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
        if self.encoder_dict.type in ['PointTransformerV3', 'PointTransformerV3_2']:
            # pcd ptv3 encoder
            fea_enc = self.encoder(input_dict)
            pc_embeddings = fea_enc['feat'].unsqueeze(0)
        else:
            ## pcd vecset encoder
            pc_embeddings = self.encoder(input_dict['coord'].unsqueeze(0))

        ## -- query encoder
        if self.query_encoder_dict.type == 'None':
            ## directly use the pcd encoder feature
            latent =  pc_embeddings
        else:
            idx, sampled_pc = self.pc_fps(input_dict['coord'])

            if self.query_encoder_dict.type == 'PE':
               sampled_pc_embeddings =  self.query_encoder(sampled_pc) # [1,num_latent,512]

            if self.query_encoder_dict.type in ['PointTransformerV3', 'PointTransformerV3_2']:
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

    def _forward_by_mode(self, input_dict, gt: bool, return_mask=False):
        latent = self.encode(input_dict)

        logit_list, targets_list, mask_list = [], [], [None]
        prev_latents = []
        latent_input = latent

        decoder_info = [
            ("decoder1", 16, 32, 'pc_query_label_fine_prop_16'),
            ("decoder2", 32, 64, 'pc_query_label_fine_prop_32'),
            ("decoder3", 64, 128, 'pc_query_label_fine_prop_64'),
            ("decoder4", 128, None, 'pc_query_label_128'),
        ]

        for i, (decoder_name, res, next_res, label_key) in enumerate(decoder_info):
            decoder = getattr(self, decoder_name)
            if decoder is None:
                continue

            query_key = f'pc_query_{res}'
            grid_key = f'pc_query_grid_{res}'
            if mask_list[-1] is not None:
                query_pts = input_dict[query_key][mask_list[-1]].unsqueeze(0)
                target = input_dict[label_key][mask_list[-1]]
                grid = input_dict[grid_key][mask_list[-1]]
            else:
                query_pts = input_dict[query_key].unsqueeze(0)
                target = input_dict[label_key]
                grid = input_dict[grid_key]

            # Skip connection: only add previous block
            if self.use_skip:
                if len(prev_latents) == 1:
                    # 第二层：latent + latent_16
                    latent_input = prev_latents[0] + latent  # latent_16 + encoder latent
                elif len(prev_latents) >= 2:
                    # 第三及以上层：latent_n + latent_{n-1}
                    latent_input = prev_latents[-1] + prev_latents[-2]
                else:
                    # 第1层（不跳跃）
                    latent_input = latent
            else:
                latent_input = prev_latents[-1] if prev_latents else latent


            logit, latent_out = decoder(latent_input, query_pts)

            # coarse-to-fine label propagation logic
            if not gt and next_res is not None:
                label_pred = self.logit2label(logit, 0.5)
                label_c = target.bool() | label_pred if self.training else label_pred
            else:
                label_c = target.bool()

            logit_list.append(logit.squeeze(-1).squeeze(0))
            targets_list.append(target)
            prev_latents.append(latent_out)

            if next_res is not None:
                fine_grid = input_dict[f'pc_query_grid_{next_res}']
                mask = self.get_fine_voxel_mask_from_coarse_labels(
                    coarse_label=label_c,
                    scale_factor=2,
                    resolution_fine=next_res,
                    coarse_grid=grid,
                    fine_grid=fine_grid
                )
                mask_list.append(mask)

        return (logit_list, targets_list, mask_list) if return_mask else (logit_list, targets_list)

    def forward_ctf(self, input_dict, return_mask=False):
        return self._forward_by_mode(input_dict, gt=False, return_mask=return_mask)

    def forward_gt(self, input_dict, return_mask=False):
        return self._forward_by_mode(input_dict, gt=True, return_mask=return_mask)

    def forward(self, input_dict,return_mask=False):
        if self.mode == 'gt':
            return self.forward_gt(input_dict,return_mask=return_mask)
        elif self.mode == 'coarse_to_fine':
            return self.forward_ctf(input_dict,return_mask=return_mask)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")