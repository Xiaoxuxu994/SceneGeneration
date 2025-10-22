import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from src.registry import MODELS, build_module
from src.models.transformer.transformer_base import PointEmbed,PreNorm,Attention,FeedForward
from src.models.utils import *
from ...registry import MODELS

class DiagonalGaussianDistribution(nn.Module):
    def __init__(self, mean, logvar, 
                 deterministic=False,
                logvar_min = -30.0,
                logvar_max = 20.0,):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, logvar_min, logvar_max)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
        
@MODELS.register_module()
class VecSetAEV4CTFKL(nn.Module):
    def __init__(self, 
                num_latent = 512,
                latent_dim = 512,
                kl_dim = 64,
                chunk_size = 100000,
                logvar_min = -30.0,
                logvar_max = 20.0,  
                use_cpu_fps=False,
                use_skip = False,
                encoder_dict = dict(
                    type = "PE",
                ),
                query_encoder_dict = dict(
                    type = "PE",
                ),
                transformer_dict = dict(
                    type = 'TransformerEncoderBlock',
                    depth = 6,
                    dim = 512,
                    heads = 8,
                    dim_head = 64,
                    enable_flash = False,
                    attn_dtype = 'fp16',
                    drop_path_rate = 0.1,
                ),
                decoder_dicts = [
                    dict(
                        type="VecSetDecoder",
                        depth=6,
                        dim=512,
                        queries_dim=512,
                        enable_flash=True,
                        attn_dtype='fp16',
                        enable_ca_flash=False,
                        output_dim=1,
                    )
                    for _ in range(4)
                ],                    
                
    ):
        
        super().__init__()
        
        self.num_latent = num_latent
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.chunk_size = chunk_size
        self.use_skip = use_skip
        self.use_cpu_fps = use_cpu_fps
        if use_cpu_fps:
            print('Using fps_cpu kernel...')

        ## training mode
        self.mode = 'coarse_to_fine'
        self.encoder_dict_type = encoder_dict.type
        self.query_encoder_dict_type = query_encoder_dict.type

        ## encoder
        if encoder_dict.type in ["PE"]:
            self.encoder = PointEmbed(dim=latent_dim)
        else:
            raise ValueError(f"Unknown mode: {encoder_dict.type}")

        ## query encoder
        if query_encoder_dict.type in ["PE"]:
            self.query_encoder = PointEmbed(dim=latent_dim)
        else:
            raise ValueError(f"Unknown mode: {query_encoder_dict.type}")

        ## encoder CA - not enable FA
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, latent_dim, heads = 1, dim_head = latent_dim), context_dim = latent_dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])     

        ## encoder SA
        self.self_attend_blocks = build_module(transformer_dict, MODELS) if transformer_dict else None


        ## KL componet
        self.mean_fc = nn.Linear(latent_dim, kl_dim)
        self.logvar_fc = nn.Linear(latent_dim, kl_dim)
        self.kl_proj = nn.Linear(kl_dim, latent_dim)       

        ## decoder
        self.decoders = nn.ModuleList([
            build_module(dec_dict,MODELS) for dec_dict in decoder_dicts
        ])

        self.decoder_info = [
            (0, 16, 32, 'pc_query_label_fine_prop_16'),
            (1, 32, 64, 'pc_query_label_fine_prop_32'),
            (2, 64, 128, 'pc_query_label_fine_prop_64'),
            (3, 128, None, 'pc_query_label_128')
        ]

    def set_mode(self, mode):
        assert mode in ['gt', 'coarse_to_fine']
        self.mode = mode

    def pc_fps(self,pc):
        # input point cloud: B x N x 3
        # use fps to get the initial learnable query points
        pc = pc.unsqueeze(0) # b=1
        B, N, D = pc.shape

        # fps
        flattened = pc.view(B * N, D)
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened
        ratio = 1.0 * self.num_latent / N

        # 120000为经验数值。最终版本应不添加此限制。
        if self.use_cpu_fps and N <= 120000:
            idx = sample_farthest_points(pc.cpu(), lengths=torch.tensor([N], dtype=torch.long), K=self.num_latent, random_start_point=True).view(-1).to(pc.device)
        else:
            idx = fps_fallback(pos, batch, ratio=ratio, random_start=True)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        return idx, sampled_pc
    

    
    def encode(self, input_dict):

        ## pcd encoder
        if self.encoder_dict_type in ['PE']:
            pc_embeddings = self.encoder(input_dict['coord'].unsqueeze(0))

        ## fps query encoder
        _, sampled_pc = self.pc_fps(input_dict['coord'])
        if self.query_encoder_dict_type in ['PE']:
            sampled_pc_embeddings =  self.query_encoder(sampled_pc) # [1,num_latent,512]
      
        ## cross attn block
        cross_attn, cross_ff = self.cross_attend_blocks
        latent= cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None, enable_flash = False) + sampled_pc_embeddings
        latent = cross_ff(latent) + latent

        ## self attn block
        if self.self_attend_blocks is not None:
            latent = self.self_attend_blocks(latent)

        ## KL
        mean = self.mean_fc(latent)
        logvar = self.logvar_fc(latent)
        
        posterior = DiagonalGaussianDistribution(mean, logvar,logvar_min=self.logvar_min,logvar_max=self.logvar_max)
        latent = posterior.sample()
        kl = posterior.kl()

        return kl, latent

    def _forward_by_mode(self, input_dict, gt: bool, return_mask=False):
        ## simple pe encoder + cross attention
        kl, latent = self.encode(input_dict)
        latent = self.kl_proj(latent)

        logit_list, targets_list, mask_list = [], [], [None]
        prev_latents = []
        latent_input = latent

        for i, (idx, res, next_res, label_key) in enumerate(self.decoder_info):
            decoder = self.decoders[idx]
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
                    latent_input = prev_latents[0] + latent  # latent_16 + encoder latent
                elif len(prev_latents) >= 2:
                    latent_input = prev_latents[-1] + prev_latents[-2]
                else:
                    latent_input = latent
            else:
                latent_input = prev_latents[-1] if prev_latents else latent


            logit, latent_out = decoder(latent_input, query_pts, chunk_size = self.chunk_size)

            # coarse-to-fine label propagation logic
            if not gt and next_res is not None:
                label_pred = logit2label(logit, 0.5)
                label_c = target.bool() | label_pred if self.training else label_pred
            else:
                label_c = target.bool()

            logit_list.append(logit.squeeze(-1).squeeze(0))
            targets_list.append(target)
            prev_latents.append(latent_out)

            if next_res is not None:
                fine_grid = input_dict[f'pc_query_grid_{next_res}']
                mask = get_fine_voxel_mask_from_coarse_labels(
                    coarse_label=label_c,
                    scale_factor=2,
                    resolution_fine=next_res,
                    coarse_grid=grid,
                    fine_grid=fine_grid
                )
                mask_list.append(mask)

        return (kl, logit_list, targets_list, mask_list) if return_mask else (kl, logit_list, targets_list)

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
        

@MODELS.register_module()
class VecSetAEV5CTFKL(nn.Module):
    def __init__(self, 
                num_latent = 512,
                latent_dim = 512,
                kl_dim = 64,
                chunk_size = 100000,
                logvar_min = -30.0,
                logvar_max = 20.0,  
                use_cpu_fps=False,
                use_kernel_fps=False,
                use_skip = False,
                encoder_dict = dict(
                    type = "PE",
                ),
                query_encoder_dict = dict(
                    type = "PE",
                ),
                transformer_dict = dict(
                    type = 'TransformerEncoderBlock',
                    depth = 6,
                    dim = 512,
                    heads = 8,
                    dim_head = 64,
                    enable_flash = False,
                    attn_dtype = 'fp16',
                    drop_path_rate = 0.1,
                ),
                decoder_dicts = [
                    dict(
                        type="VecSetDecoder",
                        depth=6,
                        dim=512,
                        queries_dim=512,
                        enable_flash=True,
                        attn_dtype='fp16',
                        enable_ca_flash=False,
                        output_dim=1,
                    )
                    for _ in range(4)
                ],  
                decoder_info = [
                (0, 32, 64, 'pc_query_sdf_32', 3.0*2.0/64),
                (1, 64, 128, 'pc_query_sdf_64',3.0*2.0/128),
                (2, 128, None, 'pc_query_label_128',0.0)
        ]                  
                
    ):
        
        super().__init__()
        
        self.num_latent = num_latent
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.chunk_size = chunk_size
        self.use_skip = use_skip
        self.use_cpu_fps = use_cpu_fps
        self.use_kernel_fps = use_kernel_fps
        if use_cpu_fps:
            print('Using fps_cpu kernel...')
        if use_kernel_fps:
            print('Using custom fps kernel...')

        ## training mode
        self.mode = 'coarse_to_fine'
        self.encoder_dict_type = encoder_dict.type
        self.query_encoder_dict_type = query_encoder_dict.type

        ## encoder
        if encoder_dict.type in ["PE"]:
            self.encoder = PointEmbed(dim=latent_dim)
        else:
            raise ValueError(f"Unknown mode: {encoder_dict.type}")

        ## query encoder
        if query_encoder_dict.type in ["PE"]:
            self.query_encoder = PointEmbed(dim=latent_dim)
        else:
            raise ValueError(f"Unknown mode: {query_encoder_dict.type}")

        ## encoder CA - not enable FA
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, latent_dim, heads = 1, dim_head = latent_dim), context_dim = latent_dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])     

        ## encoder SA
        self.self_attend_blocks = build_module(transformer_dict, MODELS) if transformer_dict else None


        ## KL componet
        self.mean_fc = nn.Linear(latent_dim, kl_dim)
        self.logvar_fc = nn.Linear(latent_dim, kl_dim)
        self.kl_proj = nn.Linear(kl_dim, latent_dim)       

        ## decoder
        self.decoders = nn.ModuleList([
            build_module(dec_dict,MODELS) for dec_dict in decoder_dicts
        ])

        self.decoder_info = decoder_info

    def set_mode(self, mode):
        assert mode in ['gt', 'coarse_to_fine']
        self.mode = mode

    def pc_fps(self,pc):
        # input point cloud: B x N x 3
        # use fps to get the initial learnable query points
        pc = pc.unsqueeze(0) # b=1
        B, N, D = pc.shape

        # fps
        flattened = pc.view(B * N, D)
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened
        ratio = 1.0 * self.num_latent / N

        if self.use_kernel_fps:
            x = pc[0,:,0].contiguous()
            y = pc[0,:,1].contiguous()
            z = pc[0,:,2].contiguous()
            seed_idx = np.random.randint(0, N-1)
            torch.npu.synchronize()
            idx,_ = cust_fps(x, y, z, N, self.num_latent, seed_idx)
            torch.npu.synchronize() 
        # 120000为经验数值。最终版本应不添加此限制。
        elif (self.use_cpu_fps) and (N <= 120000):
            idx = sample_farthest_points(pc.cpu(), lengths=torch.tensor([N], dtype=torch.long), K=self.num_latent, random_start_point=True).view(-1).to(pc.device)
        else:
            idx = fps_fallback(pos, batch, ratio=ratio, random_start=True)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        return idx, sampled_pc
    

    
    def encode(self, input_dict):

        ## pcd encoder
        if self.encoder_dict_type in ['PE']:
            pc_embeddings = self.encoder(input_dict['coord'].unsqueeze(0))

        ## fps query encoder
        _, sampled_pc = self.pc_fps(input_dict['coord'])
        if self.query_encoder_dict_type in ['PE']:
            sampled_pc_embeddings =  self.query_encoder(sampled_pc) # [1,num_latent,512]
      
        ## cross attn block
        cross_attn, cross_ff = self.cross_attend_blocks
        latent= cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None, enable_flash = False) + sampled_pc_embeddings
        latent = cross_ff(latent) + latent

        ## self attn block
        if self.self_attend_blocks is not None:
            latent = self.self_attend_blocks(latent)

        ## KL
        mean = self.mean_fc(latent)
        logvar = self.logvar_fc(latent)
        
        posterior = DiagonalGaussianDistribution(mean, logvar,logvar_min=self.logvar_min,logvar_max=self.logvar_max)
        latent = posterior.sample()
        kl = posterior.kl()

        return kl, latent

    def _forward_by_mode(self, input_dict, gt: bool, return_mask=False):
        ## simple pe encoder + cross attention
        kl, latent = self.encode(input_dict)
        latent = self.kl_proj(latent)

        logit_list, targets_list, mask_list = [], [], [None]

        # target = 0.0 #!!!!!!!!!!!!!!!!!!!!!!!!
        prev_latents = []
        latent_input = latent

        for i, (idx, res, next_res, label_key, thres) in enumerate(self.decoder_info):
            decoder = self.decoders[idx]
            if decoder is None:
                continue

            query_key = f'pc_query_{res}'
            grid_key = f'pc_query_grid_{res}'
            if mask_list[-1] is not None:
                query_pts = input_dict[query_key][mask_list[-1]].unsqueeze(0)
                target = input_dict[label_key][mask_list[-1]] #!!!!!!!!!!!!!!!!!!!
                grid = input_dict[grid_key][mask_list[-1]]
            else:
                query_pts = input_dict[query_key].unsqueeze(0)
                target = input_dict[label_key] #!@!!!!!!!!!!!!!!!!!!!!!
                grid = input_dict[grid_key]

            # Skip connection: only add previous block
            if self.use_skip:
                if len(prev_latents) == 1:
                    latent_input = prev_latents[0] + latent  # latent_16 + encoder latent
                elif len(prev_latents) >= 2:
                    latent_input = prev_latents[-1] + prev_latents[-2]
                else:
                    latent_input = latent
            else:
                latent_input = prev_latents[-1] if prev_latents else latent


            logit, latent_out = decoder(latent_input, query_pts, chunk_size = self.chunk_size)

            # coarse-to-fine label propagation logic
            if not gt and next_res is not None:
                label_pred = logit <= thres
                label_pred = label_pred.squeeze(0).squeeze(-1)
                label_c = (target <= thres) | label_pred if self.training else label_pred
            else:
                label_c = target <= thres

            logit_list.append(logit.squeeze(-1).squeeze(0))
            targets_list.append(target) #!!!!!!!!!!!!!!!!!!!!!!!
            prev_latents.append(latent_out)

            if next_res is not None:
                fine_grid = input_dict[f'pc_query_grid_{next_res}']
                # ## org one
                # mask = get_fine_voxel_mask_from_coarse_labels(
                #     coarse_label=label_c,
                #     scale_factor=2,
                #     resolution_fine=next_res,
                #     coarse_grid=grid,
                #     fine_grid=fine_grid
                # )
                ## fast version v1
                mask = get_fine_voxel_mask_from_coarse_labels_fast(
                    coarse_label = label_c,                      
                    scale_factor = 2,                   
                    resolution_coarse = res,              
                    coarse_grid = grid,                        
                    fine_grid = fine_grid, 
                )

                mask_list.append(mask)

        return (kl, logit_list, targets_list, mask_list) if return_mask else (kl, logit_list, targets_list)

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
        

@MODELS.register_module()
class VecSetAEV4SINGLEKL(nn.Module):
    def __init__(self, 
                num_latent = 512,
                latent_dim = 512,
                kl_dim = 64,
                chunk_size = 200000,
                use_cpu_fps=False,
                use_kernel_fps=False,
                use_skip = False,
                logvar_min = -30.0,
                logvar_max = 20.0,                
                encoder_dict = dict(
                    type = "VecSet",
                ),
                query_encoder_dict = dict(
                    type = "VecSet",
                ),
                transformer_dict = dict(
                    type = 'TransformerEncoderBlock',
                    depth = 8,
                    dim = 512,
                    heads = 8,
                    dim_head = 64,
                    enable_flash = False,
                    attn_dtype = 'fp16',
                    drop_path_rate = 0.1,
                ),
                decoder_dict4 = dict(
                    type = "VecSetDecoder",
                    depth = 6,
                    dim = 512,
                    queries_dim = 512,
                    enable_flash=True,
                    attn_dtype='fp16',
                    enable_ca_flash=False,
                    output_dim = 1,
                ),
    ):
        
        super().__init__()
        
        self.num_latent = num_latent
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.chunk_size = chunk_size
        self.use_skip = use_skip
        self.use_cpu_fps = use_cpu_fps
        self.use_kernel_fps = use_kernel_fps
        if use_cpu_fps:
            print('Using fps_cpu kernel...')
        if use_kernel_fps:
            print('Using custom fps kernel...')


        ## training mode
        self.mode = 'coarse_to_fine'
        self.encoder_dict_type = encoder_dict.type
        self.query_encoder_dict_type = query_encoder_dict.type

        ## encoder
        if encoder_dict.type in ["PE"]:
            self.encoder = PointEmbed(dim=latent_dim)
        else:
            raise ValueError(f"Unknown mode: {encoder_dict.type}")

        ## query encoder
        if query_encoder_dict.type in ["PE"]:
            self.query_encoder = PointEmbed(dim=latent_dim)
        else:
            raise ValueError(f"Unknown mode: {query_encoder_dict.type}")

        ## encoder CA - not enable FA
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, latent_dim, heads = 1, dim_head = latent_dim), context_dim = latent_dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])     

        ## encoder SA
        self.self_attend_blocks = build_module(transformer_dict, MODELS) if transformer_dict else None


        ## KL componet
        self.mean_fc = nn.Linear(latent_dim, kl_dim)
        self.logvar_fc = nn.Linear(latent_dim, kl_dim)
        self.kl_proj = nn.Linear(kl_dim, latent_dim)       

        # decoder
        self.decoder4 = build_module(decoder_dict4, MODELS) if decoder_dict4 else None

    def pc_fps(self,pc):
        # input point cloud: B x N x 3
        # use fps to get the initial learnable query points
        pc = pc.unsqueeze(0) # b=1
        B, N, D = pc.shape

        # fps
        flattened = pc.view(B * N, D)
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened
        ratio = 1.0 * self.num_latent / N

        if self.use_kernel_fps:
            x = pc[0,:,0].contiguous()
            y = pc[0,:,1].contiguous()
            z = pc[0,:,2].contiguous()
            seed_idx = np.random.randint(0, N-1)
            torch.npu.synchronize()
            idx,_ = cust_fps(x, y, z, N, self.num_latent, seed_idx)
            torch.npu.synchronize() 
        # 120000为经验数值。最终版本应不添加此限制。
        elif (self.use_cpu_fps) and (N <= 120000):
            idx = sample_farthest_points(pc.cpu(), lengths=torch.tensor([N], dtype=torch.long), K=self.num_latent, random_start_point=True).view(-1).to(pc.device)
        else:
            idx = fps_fallback(pos, batch, ratio=ratio, random_start=True)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        return idx, sampled_pc
    
    def encode(self, input_dict):

        ## pcd encoder
        if self.encoder_dict_type in ['PE']:
            pc_embeddings = self.encoder(input_dict['coord'].unsqueeze(0))

        ## fps query encoder
        _, sampled_pc = self.pc_fps(input_dict['coord'])
        if self.query_encoder_dict_type in ['PE']:
            sampled_pc_embeddings =  self.query_encoder(sampled_pc) # [1,num_latent,512]
      
        ## cross attn block
        cross_attn, cross_ff = self.cross_attend_blocks
        latent= cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None, enable_flash = False) + sampled_pc_embeddings
        latent = cross_ff(latent) + latent

        ## self attn block
        if self.self_attend_blocks is not None:
            latent = self.self_attend_blocks(latent)

        ## KL
        mean = self.mean_fc(latent)
        logvar = self.logvar_fc(latent)
        
        posterior = DiagonalGaussianDistribution(mean, logvar,logvar_min=self.logvar_min,logvar_max=self.logvar_max)
        latent = posterior.sample()
        kl = posterior.kl()

        return kl, latent

    def decode(self, latent, query_pts,chunk_size=None):
        latent = self.kl_proj(latent)
        logit,latent = self.decoder4(latent, query_pts,chunk_size)
        return logit,latent
    
    def forward(self, input_dict,vs):
        ## simple pe encoder + cross attention
        kl, latent = self.encode(input_dict)

        # decoder vs128 train with
        logit_list, targets_list = [], []
        query_pts = input_dict[f'pc_query_{vs}'].unsqueeze(0)
        target = input_dict[f'pc_query_sdf_{vs}']

        logit,_ = self.decode(latent, query_pts,chunk_size=self.chunk_size)
        
        # activation by channel count
        C = logit.shape[-1]
        if C == 5:
            logit = torch.cat([logit[..., :1], torch.tanh(logit[..., 1:-1]), logit[..., -1:]], dim=-1)  # [1, N, 4]
            logit_list.append(logit.squeeze(0))                                      # [N, 4]
        elif C == 4:
            logit = torch.cat([logit[..., :1], torch.tanh(logit[..., 1:])], dim=-1)  # [1, N, 4]
            logit_list.append(logit.squeeze(0))                                      # [N, 4]
        elif C == 1:
            logit_list.append(logit.squeeze(-1).squeeze(0))                          # [N]
        else:
            logit_list.append(logit.squeeze(0))                                      # [N, C]

        targets_list.append(target)

        return kl, logit_list, targets_list

        # logit,_ = self.decode(latent, query_pts,chunk_size=self.chunk_size)
        # logit_list.append(logit.squeeze(-1).squeeze(0))
        # targets_list.append(target)
        # return kl, logit_list, targets_list
    
    def forward_test(self, input_dict,vs):
        ## simple pe encoder + cross attention
        kl, latent = self.encode(input_dict)

        # decoder vs128 train with
        logit_list = []
        # # ## sdf label
        query_pts = input_dict[f'pc_query_{vs}'].unsqueeze(0)
        logit,_ = self.decode(latent, query_pts,chunk_size=self.chunk_size)
        logit_list.append(logit.squeeze(-1).squeeze(0))

        return logit_list

    def forward_ctf_infer(self, input_dict, vs, latent=None, return_latent=False):
        ## simple pe encoder + cross attention
        if latent is None:
            kl, latent = self.encode(input_dict)

        # decoder vs128 train with
        query_pts = input_dict[f'pc_query_{vs}'].unsqueeze(0)

        if f'pc_query_sdf_{vs}' in input_dict.keys():
            target = input_dict[f'pc_query_sdf_{vs}']
        else:
            target = []

        logit,_ = self.decode(latent, query_pts,chunk_size=self.chunk_size)

        if return_latent:
            return logit.squeeze(-1).squeeze(0), target, latent
        else:
            return logit.squeeze(-1).squeeze(0), target
    
@MODELS.register_module()
class VecSetAEV5SINGLEKL(nn.Module):
    def __init__(self, 
                use_normal = False,
                num_latent = 512,
                latent_dim = 512,
                kl_dim = 64,
                chunk_size = 200000,
                use_cpu_fps=False,
                use_skip = False,
                logvar_min = -30.0,
                logvar_max = 20.0,                
                encoder_dict = dict(
                    type = "VecSet",
                ),
                query_encoder_dict = dict(
                    type = "VecSet",
                ),
                transformer_dict = dict(
                    type = 'TransformerEncoderBlock',
                    depth = 8,
                    dim = 512,
                    heads = 8,
                    dim_head = 64,
                    enable_flash = False,
                    attn_dtype = 'fp16',
                    drop_path_rate = 0.1,
                ),
                decoder_dict4 = dict(
                    type = "VecSetDecoder",
                    depth = 6,
                    dim = 512,
                    queries_dim = 512,
                    enable_flash=True,
                    attn_dtype='fp16',
                    enable_ca_flash=False,
                    output_dim = 1,
                ),
    ):
        
        super().__init__()
        
        self.num_latent = num_latent
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.chunk_size = chunk_size
        self.use_skip = use_skip
        self.use_cpu_fps = use_cpu_fps
        self.use_normal = use_normal
        if use_cpu_fps:
            print('Using fps_cpu kernel...')

        ## training mode
        self.mode = 'coarse_to_fine'
        self.encoder_dict_type = encoder_dict.type
        self.query_encoder_dict_type = query_encoder_dict.type

        ## encoder
        if encoder_dict.type in ["PE"]:
            self.encoder = PointEmbed(dim=latent_dim,use_normal=self.use_normal)
        else:
            raise ValueError(f"Unknown mode: {encoder_dict.type}")

        ## query encoder
        if query_encoder_dict.type in ["PE"]:
            self.query_encoder = PointEmbed(dim=latent_dim,use_normal=self.use_normal)
        else:
            raise ValueError(f"Unknown mode: {query_encoder_dict.type}")

        ## encoder CA - not enable FA
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, latent_dim, heads = 1, dim_head = latent_dim), context_dim = latent_dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])     

        ## encoder SA
        self.self_attend_blocks = build_module(transformer_dict, MODELS) if transformer_dict else None


        ## KL componet
        self.mean_fc = nn.Linear(latent_dim, kl_dim)
        self.logvar_fc = nn.Linear(latent_dim, kl_dim)
        self.kl_proj = nn.Linear(kl_dim, latent_dim)       

        # decoder
        self.decoder4 = build_module(decoder_dict4, MODELS) if decoder_dict4 else None

    def pc_fps(self,pc):
        # input point cloud: B x N x 3
        # use fps to get the initial learnable query points
        pc = pc.unsqueeze(0) # b=1
        B, N, D = pc.shape

        # fps
        flattened = pc.view(B * N, D)
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened
        ratio = 1.0 * self.num_latent / N

        # 120000为经验数值。最终版本应不添加此限制。
        if self.use_cpu_fps and N <= 120000:
            idx = sample_farthest_points(pc.cpu(), lengths=torch.tensor([N], dtype=torch.long), K=self.num_latent, random_start_point=True).view(-1).to(pc.device)
        else:
            idx = fps_fallback(pos, batch, ratio=ratio, random_start=True)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        return idx, sampled_pc
    
    def encode(self, input_dict):

        coord = input_dict['coord']                       
        normal = input_dict.get('normal', None)    

        if self.use_normal and (normal is not None):
            feat_all = torch.cat([coord, normal], dim=-1)    
        else:
            feat_all = coord                                
        pc_embeddings = self.encoder(feat_all.unsqueeze(0)) 

        idx, sampled_pc = self.pc_fps(coord)             

        if self.use_normal and (normal is not None):
            sampled_normal = normal[idx]                    
            sampled_feat = torch.cat([sampled_pc.squeeze(0), 
                                    sampled_normal], dim=-1)  
            sampled_feat = sampled_feat.unsqueeze(0)        
        else:
            sampled_feat = sampled_pc                     
        sampled_pc_embeddings = self.query_encoder(sampled_feat)  
      
        ## cross attn block
        cross_attn, cross_ff = self.cross_attend_blocks
        latent= cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None, enable_flash = False) + sampled_pc_embeddings
        latent = cross_ff(latent) + latent

        ## self attn block
        if self.self_attend_blocks is not None:
            latent = self.self_attend_blocks(latent)

        ## KL
        mean = self.mean_fc(latent)
        logvar = self.logvar_fc(latent)
        
        posterior = DiagonalGaussianDistribution(mean, logvar,logvar_min=self.logvar_min,logvar_max=self.logvar_max)
        latent = posterior.sample()
        kl = posterior.kl()

        return kl, latent

    def decode(self, latent, query_pts,chunk_size=None):
        latent = self.kl_proj(latent)
        logit,latent = self.decoder4(latent, query_pts,chunk_size)
        return logit,latent
    
    def forward(self, input_dict,vs):
        ## simple pe encoder + cross attention
        kl, latent = self.encode(input_dict)

        # decoder vs128 train with
        logit_list, targets_list = [], []
        query_pts = input_dict[f'pc_query_{vs}'].unsqueeze(0)
        target = input_dict[f'pc_query_sdf_{vs}']

        logit,_ = self.decode(latent, query_pts,chunk_size=self.chunk_size)
        
        # activation by channel count
        C = logit.shape[-1]
        if C == 5:
            logit = torch.cat([logit[..., :1], torch.tanh(logit[..., 1:-1]), logit[..., -1:]], dim=-1)  # [1, N, 4]
            logit_list.append(logit.squeeze(0))                                      # [N, 4]
        elif C == 4:
            logit = torch.cat([logit[..., :1], torch.tanh(logit[..., 1:])], dim=-1)  # [1, N, 4]
            logit_list.append(logit.squeeze(0))                                      # [N, 4]
        elif C == 1:
            logit_list.append(logit.squeeze(-1).squeeze(0))                          # [N]
        else:
            logit_list.append(logit.squeeze(0))                                      # [N, C]

        targets_list.append(target)

        return kl, logit_list, targets_list

        # logit,_ = self.decode(latent, query_pts,chunk_size=self.chunk_size)
        # logit_list.append(logit.squeeze(-1).squeeze(0))
        # targets_list.append(target)
        # return kl, logit_list, targets_list
    
    def forward_test(self, input_dict,vs):
        ## simple pe encoder + cross attention
        kl, latent = self.encode(input_dict)

        # decoder vs128 train with
        logit_list = []
        # # ## sdf label
        query_pts = input_dict[f'pc_query_{vs}'].unsqueeze(0)
        logit,_ = self.decode(latent, query_pts,chunk_size=self.chunk_size)
        logit_list.append(logit.squeeze(-1).squeeze(0))

        return logit_list
    

@MODELS.register_module()
class VecSetAEV6SINGLEKL(nn.Module):
    def __init__(self, 
                num_latent = 512,
                latent_dim = 512,
                kl_dim = 64,
                chunk_size = 200000,
                use_cpu_fps=False,
                use_kernel_fps=False,
                use_skip = False,
                logvar_min = -30.0,
                logvar_max = 20.0,                
                encoder_dict = dict(
                    type = "VecSet",
                ),
                query_encoder_dict = dict(
                    type = "VecSet",
                ),
                transformer_dict = dict(
                    type = 'TransformerEncoderBlock',
                    depth = 8,
                    dim = 512,
                    heads = 8,
                    dim_head = 64,
                    enable_flash = False,
                    attn_dtype = 'fp16',
                    drop_path_rate = 0.1,
                ),
                decoder_dict4 = dict(
                    type = "VecSetDecoder",
                    depth = 6,
                    dim = 512,
                    queries_dim = 512,
                    enable_flash=True,
                    attn_dtype='fp16',
                    enable_ca_flash=False,
                    output_dim = 1,
                ),
    ):
        
        super().__init__()
        
        self.num_latent = num_latent
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.chunk_size = chunk_size
        self.use_skip = use_skip
        self.use_cpu_fps = use_cpu_fps
        self.use_kernel_fps = use_kernel_fps
        if use_cpu_fps:
            print('Using fps_cpu kernel...')
        if use_kernel_fps:
            print('Using custom fps kernel...')


        ## training mode
        self.mode = 'coarse_to_fine'
        self.encoder_dict_type = encoder_dict.type
        self.query_encoder_dict_type = query_encoder_dict.type

        ## encoder
        if encoder_dict.type in ["PE"]:
            self.encoder = PointEmbed(dim=latent_dim)
        else:
            raise ValueError(f"Unknown mode: {encoder_dict.type}")

        ## query encoder
        if query_encoder_dict.type in ["PE"]:
            self.query_encoder = PointEmbed(dim=latent_dim)
        else:
            raise ValueError(f"Unknown mode: {query_encoder_dict.type}")

        ## encoder CA - not enable FA
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, latent_dim, heads = 1, dim_head = latent_dim), context_dim = latent_dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])     

        ## encoder SA
        self.self_attend_blocks = build_module(transformer_dict, MODELS) if transformer_dict else None


        ## KL componet
        self.mean_fc = nn.Linear(latent_dim, kl_dim)
        self.logvar_fc = nn.Linear(latent_dim, kl_dim)
        self.kl_proj = nn.Linear(kl_dim, latent_dim)       

        # decoder
        self.decoder4 = build_module(decoder_dict4, MODELS) if decoder_dict4 else None

    def pc_fps(self,pc):
        # input point cloud: B x N x 3
        # use fps to get the initial learnable query points
        pc = pc.unsqueeze(0) # b=1
        B, N, D = pc.shape

        # fps
        flattened = pc.view(B * N, D)
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened
        ratio = 1.0 * self.num_latent / N

        if self.use_kernel_fps:
            x = pc[0,:,0].contiguous()
            y = pc[0,:,1].contiguous()
            z = pc[0,:,2].contiguous()
            seed_idx = np.random.randint(0, N-1)
            torch.npu.synchronize()
            idx,_ = cust_fps(x, y, z, N, self.num_latent, seed_idx)
            torch.npu.synchronize() 
        # 120000为经验数值。最终版本应不添加此限制。
        elif (self.use_cpu_fps) and (N <= 120000):
            idx = sample_farthest_points(pc.cpu(), lengths=torch.tensor([N], dtype=torch.long), K=self.num_latent, random_start_point=True).view(-1).to(pc.device)
        else:
            idx = fps_fallback(pos, batch, ratio=ratio, random_start=True)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        return idx, sampled_pc
    
    def encode(self, input_dict):

        ## pcd encoder
        if self.encoder_dict_type in ['PE']:
            pc_embeddings = self.encoder(input_dict['coord'].unsqueeze(0))

        ## fps query encoder
        _, sampled_pc = self.pc_fps(input_dict['coord'])
        if self.query_encoder_dict_type in ['PE']:
            sampled_pc_embeddings =  self.query_encoder(sampled_pc) # [1,num_latent,512]
      
        ## cross attn block
        cross_attn, cross_ff = self.cross_attend_blocks
        latent= cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None, enable_flash = False) + sampled_pc_embeddings
        latent = cross_ff(latent) + latent

        ## self attn block
        if self.self_attend_blocks is not None:
            latent = self.self_attend_blocks(latent)

        ## KL
        mean = self.mean_fc(latent)
        logvar = self.logvar_fc(latent)
        
        posterior = DiagonalGaussianDistribution(mean, logvar,logvar_min=self.logvar_min,logvar_max=self.logvar_max)
        latent = posterior.sample()
        kl = posterior.kl()

        return kl, latent

    def decode(self, latent, query_pts,chunk_size=None):
        latent = self.kl_proj(latent)
        logit,latent = self.decoder4(latent, query_pts,chunk_size)
        return logit,latent
    
    def forward(self, input_dict, vs_list):

        logit_list, targets_list = [], []

        for vs in vs_list:
            ## simple pe encoder + cross attention
            kl, latent = self.encode(input_dict)
            
            query_pts = input_dict[f'pc_query_{vs}'].unsqueeze(0)
            target = input_dict[f'pc_query_sdf_{vs}']

            logit,_ = self.decode(latent, query_pts,chunk_size=self.chunk_size)
            
            C = logit.shape[-1]
            if C == 1:
                logit_list.append(logit.squeeze(-1).squeeze(0))                   
            else:
                logit_list.append(logit.squeeze(0))                          

            targets_list.append(target)

        return kl, logit_list, targets_list
    
    def forward_ctf_infer(self, input_dict, vs, latent=None, return_latent=False):
        ## simple pe encoder + cross attention
        if latent is None:
            kl, latent = self.encode(input_dict)

        # decoder vs128 train with
        query_pts = input_dict[f'pc_query_{vs}'].unsqueeze(0)
        target = input_dict[f'pc_query_sdf_{vs}']

        logit,_ = self.decode(latent, query_pts,chunk_size=self.chunk_size)

        if return_latent:
            return logit.squeeze(-1).squeeze(0), target, latent
        else:
            return logit.squeeze(-1).squeeze(0), target