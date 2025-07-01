import torch
import torch.nn.functional as F
from torch import nn
# from torch_cluster import fps
import math
from src.registry import MODELS, build_module
from src.models.transformer.transformer_base import PointEmbed,PreNorm,Attention_SBMS,FeedForward
import numpy as np
from ...registry import MODELS

from util import sample_farthest_points

def fps(points, device, num_samples, random_start=True):
    with torch.no_grad():
        sampled_indices = torch.zeros((num_samples), dtype=torch.long, device=device)
        start = torch.randint(0, len(points), (1,), dtype=torch.long, device=device) if random_start else torch.tensor([0], dtype=torch.long, device=device)
        distances = torch.full((len(points),), 1e10, device=device)
        for i in range(num_samples):
            sampled_indices[i] = start 
            last_sampled = points[start]
            dist_tmp = torch.sum((points - last_sampled) ** 2, dim=1)
            distances = torch.minimum(distances, dist_tmp)
            start = torch.argmax(distances).to(torch.long)
    return sampled_indices

class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean, device=self.mean.device)
    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape, device=self.mean.device)
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
class VecSetAEV4_SBMS(nn.Module):
    def __init__(self, 
                input_pcd_num = 50000,
                num_latent = 512,
                latent_dim = 512,
                kl_latent_dim = 64,
                enable_kl = False,
                enable_multilevel=False,
                feafuse_channels=[64,128,256,512],
                enable_seg_head=False,
                seg_dim = 512,
                seg_class = 100,
                enable_ca_flash = False,
                encoder_dict = dict(
                    type = "PointTransformerV3_2",
                    in_channels = 3, # ['coord]
                    enable_flash=True,
                    attn_dtype='fp16',
                    drop_path=0.3,
                    cls_mode=False, #True for enc only; False for enc-dec backbone
                ),
                query_encoder_dict = dict(
                    type = "PointTransformerV3_2",
                    in_channels = 3, # ['coord]
                    enable_flash=True,
                    attn_dtype='fp16',
                    drop_path=0.3,
                    cls_mode=False, #True for enc only; False for enc-dec backbone
                ),
                decoder_dict = dict(
                    type = "VecSetDecoder_SBMS", #['VecSetDecoder','VecSetKLDecoder']
                    depth = 24,
                    dim = 512,
                    queries_dim = 512,
                    enable_flash=True,
                    attn_dtype='fp16',
                    enable_ca_flash=False,
                    output_dim = 1,
                    # deterministic = True, # ['Auto' or 'KL']
                ),
                use_cpu_fps=False
    ):
        
        super().__init__()
        
        self.input_pcd_num = input_pcd_num
        self.num_latent = num_latent
        self.latent_dim = latent_dim
        self.kl_latent_dim = kl_latent_dim
        self.enable_kl = enable_kl
        self.enable_multilevel = enable_multilevel
        self.feafuse_channels = feafuse_channels
        self.enable_ca_flash = enable_ca_flash
        self.encoder_dict = encoder_dict
        self.query_encoder_dict = query_encoder_dict
        self.decoder_dict = decoder_dict

        self.enable_seg_head = enable_seg_head
        self.seg_dim = seg_dim
        self.seg_class = seg_class

        ## ptv3 encoder
        if self.encoder_dict.type == 'PointTransformerV3' or 'PointTransformerV3_2':
            self.encoder = build_module(self.encoder_dict, MODELS)     
            if self.enable_multilevel:
                self.feafuse = nn.Linear(sum(self.feafuse_channels),self.latent_dim)

        ## query encoder
        if self.query_encoder_dict.type == 'PointTransformerV3' or 'PointTransformerV3_2':
            self.query_encoder = build_module(self.query_encoder_dict, MODELS)   

        ## encoder cross attention
        if not self.enable_ca_flash:
            self.cross_attend_blocks = nn.ModuleList([
                PreNorm(self.latent_dim, Attention_SBMS(self.latent_dim, self.latent_dim, heads = 1, dim_head = self.latent_dim), context_dim = self.latent_dim),
                PreNorm(self.latent_dim, FeedForward(self.latent_dim))
            ])

            if self.enable_multilevel:
                self.cross_attend_blocks_list = nn.ModuleList()
                for feafuse_channel in self.feafuse_channels[::-1]:
                    self.cross_attend_blocks = nn.ModuleList([
                        PreNorm(feafuse_channel, Attention_SBMS(feafuse_channel, feafuse_channel, heads = 1, dim_head = feafuse_channel), context_dim = feafuse_channel),
                        PreNorm(feafuse_channel, FeedForward(feafuse_channel))
                    ])
                    self.cross_attend_blocks_list.append(self.cross_attend_blocks)

        else:
            self.cross_attend_blocks = nn.ModuleList([
                PreNorm(self.latent_dim, Attention_SBMS(self.latent_dim, self.latent_dim, heads = 32, dim_head = self.latent_dim // 32), context_dim = self.latent_dim),
                PreNorm(self.latent_dim, FeedForward(self.latent_dim))
            ])            

        ## decoder
        self.decoder = build_module(self.decoder_dict, MODELS)

        ## seg head 
        if self.enable_seg_head:
            self.seg_head = nn.Linear(self.seg_dim, self.seg_class)
        
        if self.enable_kl:
            ## kl latent
            self.mean_fc = nn.Linear(latent_dim, kl_latent_dim)
            self.logvar_fc = nn.Linear(latent_dim, kl_latent_dim)
            ## kl inv proj
            self.proj = nn.Linear(kl_latent_dim, latent_dim)

        self.use_cpu_fps = use_cpu_fps
        if use_cpu_fps:
            print('Using fps_cpu kernel...')

    def pc_fps_sbms(self, pc, offset:torch.Tensor):
        B = offset.shape[0]
        idx = torch.zeros((self.num_latent * B), dtype=torch.long, device=pc.device)
        offset_ext = torch.cat([torch.tensor([0], device=pc.device, dtype=torch.long), offset], dim=0) # [0, s1, s1+s2, ..., s1+s2+...+sb]
        for i in range(B):
            N = offset_ext[i+1] - offset_ext[i]
            if self.use_cpu_fps and N <= 120000:
                idx[(i*self.num_latent):((i+1)*self.num_latent)] = sample_farthest_points(pc[offset_ext[i]:offset_ext[i+1]].unsqueeze(0).cpu(), lengths=torch.tensor([N], dtype=torch.long), K=self.num_latent, random_start_point=True).view(-1).to(pc.device)
            else:
                idx[(i*self.num_latent):((i+1)*self.num_latent)] = fps(pc[offset_ext[i]:offset_ext[i+1]], pc.device, self.num_latent, random_start=True) + offset_ext[i]            
        sampled_pc = pc[idx].view(-1, self.num_latent * B, 3)
        return idx, sampled_pc

    def pc_fps(self, pc):
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
            # idx = self.fps_fallback(pos, batch, ratio=ratio, random_start=True)
            idx = fps(pos, pos.device, self.num_latent, random_start=True)
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

    def encode(self, input_dict):
        ## -- pcd encoder
        if self.encoder_dict.type == 'PointTransformerV3' or 'PointTransformerV3_2':
            # pcd ptv3 encoder
            fea_enc = self.encoder(input_dict) #[S,D] or [sum(S), D]
            pc_embeddings = fea_enc['feat'].unsqueeze(0) # [1,S,D] or [1,sum(S),D]
            # if not self.encoder_dict.cls_mode and self.pc_latent_embedding:
            #     # pcd ptv3 full backbone, enc-dec, add one linear to lift feature to latent dim
            #     pc_embeddings = self.pc_latent_embedding(pc_embeddings)
        else:
            ## pcd vecset encoder
            pc_embeddings = self.point_embed(input_dict['coord'].unsqueeze(0))

        ## ---- seg head
        if self.enable_seg_head:
            seg_logits = self.seg_head(pc_embeddings)

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
                sample_pc_dict['offset'] = torch.tensor([self.num_latent*(i+1) for i in range(input_dict['offset'].shape[0])], dtype=torch.long, device=pc_embeddings.device)
                sample_fea_enc = self.query_encoder(sample_pc_dict)
                sampled_pc_embeddings = sample_fea_enc['feat'].unsqueeze(0)    
            else:
                if input_dict['data_type'] == 0:
                    idx, sampled_pc = self.pc_fps_sbms(input_dict['coord'], input_dict['offset'])

                if input_dict['data_type'] == 1:
                    sampled_pc = input_dict['coord_fps']
                    idx = input_dict['fps_index']
                    for i in range(input_dict['offset'].shape[0] - 1): # 多序列dataloader载入的idx只拼接是没有offset偏移的
                        idx[((i+1)*self.num_latent):((i+2)*self.num_latent)] += input_dict['offset'][i]

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
                        sample_pc_dict['offset'] = torch.tensor([self.num_latent*(i+1) for i in range(input_dict['offset'].shape[0])], dtype=torch.long, device=pc_embeddings.device)
                        sample_fea_enc = self.query_encoder(sample_pc_dict)
                        sampled_pc_embeddings = sample_fea_enc['feat'].unsqueeze(0)
                    else:
                        # ptv3 full backbone
                        sampled_pc_embeddings = pc_embeddings[:,idx,:]
                        # sampled_pc_embeddings = pc_embeddings.half()[idx].unsqueeze(0)
            ## cross attn block
            cross_attn, cross_ff = self.cross_attend_blocks
            latent = cross_attn(sampled_pc_embeddings, sample_pc_dict['offset'], context = pc_embeddings, context_offset=input_dict['offset'], mask = None, enable_flash = self.enable_ca_flash) + sampled_pc_embeddings
            latent = cross_ff(latent) + latent
        
        if self.enable_kl:
            mean = self.mean_fc(latent)
            logvar = self.logvar_fc(latent)
            posterior = DiagonalGaussianDistribution(mean, logvar, deterministic=False)
            latent = posterior.sample()
            kl = posterior.kl()
        
        if not self.enable_seg_head:
            seg_logits = None
        if not self.enable_kl:
            kl = None
        return latent, seg_logits, kl
    
    def encode_multilevel(self, input_dict):
        pc_embeddings_list = []
        sampled_pc_embeddings_list = []
        ## -- pcd encoder
        if self.encoder_dict.type == 'PointTransformerV3':
            # pcd ptv3 encoder
            fea_enc = self.encoder(input_dict)
            pc_embeddings = fea_enc['feat'].unsqueeze(0)

            pc_embeddings_list.append(pc_embeddings)
            pc_embeddings = fea_enc['pooling_parent']
            for _ in range(len(self.feafuse_channels)-1):
                pc_embeddings_list.append(pc_embeddings['feat'].unsqueeze(0))
                pc_embeddings = pc_embeddings['pooling_parent']
            
            idx, _ = self.pc_fps(input_dict['coord'])

            if self.query_encoder_dict.type == 'PointTransformerV3':
                sample_pc_dict = {}
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        sample_pc_dict[key] = input_dict[key].clone()
                if "coord" in input_dict.keys():
                    sample_pc_dict["coord"] = sample_pc_dict["coord"][idx]
                if "grid_coord" in input_dict.keys():
                    sample_pc_dict["grid_coord"] = sample_pc_dict["grid_coord"][idx]
                if "segment" in input_dict.keys():
                    sample_pc_dict["segment"] = sample_pc_dict["segment"][idx]
                if "feat" in input_dict.keys():
                    sample_pc_dict["feat"] = sample_pc_dict["feat"][idx]
                sample_pc_dict['offset'] = torch.tensor([self.num_latent], device=pc_embeddings.feat.device) # updata pcd offset

                sample_fea_enc = self.query_encoder(sample_pc_dict)
                
                sampled_pc_embeddings = sample_fea_enc['feat'].unsqueeze(0)
                sampled_pc_embeddings_list.append(sampled_pc_embeddings)
                sampled_pc_embeddings = sample_fea_enc['pooling_parent']
                for _ in range(len(self.feafuse_channels)-1):
                    sampled_pc_embeddings_list.append(sampled_pc_embeddings['feat'].unsqueeze(0))
                    sampled_pc_embeddings = sampled_pc_embeddings['pooling_parent']
            
            latent_list = []
            for i in range(len(self.feafuse_channels)):
                self.cross_attend_blocks = self.cross_attend_blocks_list[i]
                sampled_pc_embeddings = sampled_pc_embeddings_list[i]
                pc_embeddings = pc_embeddings_list[i]
                ## cross attn block
                cross_attn, cross_ff = self.cross_attend_blocks
                latent= cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None, enable_flash = self.enable_ca_flash) + sampled_pc_embeddings
                latent = cross_ff(latent) + latent
                latent_list.append(latent)

            latent = self.feafuse(torch.cat(latent_list,dim=-1))
        return latent


    def decode(self, latent, *args):
        if self.enable_kl:
            latent = self.proj(latent)
        output = self.decoder(latent, *args)
        return output

    def forward(self, input_dict):
        if self.enable_multilevel:
            latent = self.encode_multilevel(input_dict)
            seg_logits, kl = None, None
        else:
            latent, seg_logits, kl = self.encode(input_dict)
        query_points = input_dict['pc_query'].unsqueeze(0) # [1,sum(s),d]
        latent_offset = torch.tensor([self.num_latent*(i+1) for i in range(input_dict['pc_query_offset'].shape[0])], dtype=torch.long, device=input_dict['pc_query_offset'].device)
        output = self.decode(latent, latent_offset, query_points, input_dict['pc_query_offset'])
        output = output.squeeze(-1)
        return {'logits': output,'seg_logits': seg_logits, 'kl': kl}