import torch
from torch import nn
# from torch_cluster import fps
from src.registry import MODELS, build_module
from src.models.transformer.transformer_base import PreNorm,Attention,FeedForward

@MODELS.register_module()
class VecSetAEV4(nn.Module):
    def __init__(self, 
                input_pcd_num = 50000,
                num_latent = 512,
                latent_dim = 512,
                enable_multilevel=False,
                feafuse_channels=[64,128,256,512],
                enable_seg_head=False,
                seg_dim = 512,
                seg_class = 100,
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
                decoder_dict = dict(
                    type = "VecSetDecoder", #['VecSetDecoder','VecSetKLDecoder']
                    depth = 24,
                    dim = 512,
                    queries_dim = 512,
                    enable_flash=True,
                    attn_dtype='fp16',
                    enable_ca_flash=False,
                    output_dim = 1,
                    # deterministic = True, # ['Auto' or 'KL']
                )
    ):
        
        super().__init__()
        
        self.input_pcd_num = input_pcd_num
        self.num_latent = num_latent
        self.latent_dim = latent_dim
        self.enable_multilevel = enable_multilevel
        self.feafuse_channels = feafuse_channels


        self.encoder_dict = encoder_dict
        self.query_encoder_dict = query_encoder_dict
        self.decoder_dict = decoder_dict

        self.enable_seg_head = enable_seg_head
        self.seg_dim = seg_dim
        self.seg_class = seg_class

        ## ptv3 encoder
        self.encoder = build_module(self.encoder_dict, MODELS)     

        ## query encoder
        self.query_encoder = build_module(self.query_encoder_dict, MODELS)   

        ## encoder cross attention
        if not self.encoder_dict["enable_ca_flash"]:
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
        self.decoder = build_module(self.decoder_dict, MODELS)

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
            M = max(1, int(N * ratio))  # 至少采样1个点

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

    def pc_fps(self,pc):
        # input point cloud: B x N x 3
        # use fps to get the initial learnable query points
        pc = pc.unsqueeze(0) # b=1
        B, N, D = pc.shape
        self.input_pcd_num = N    ### !!!!!!!!!!!!!!!!!! 
        # fps
        flattened = pc.view(B * N, D)
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened
        ratio = 1.0 * self.num_latent / self.input_pcd_num
        idx = self.fps_fallback(pos, batch, ratio=ratio) ## fps callback torch实现

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        return idx, sampled_pc

    def encode(self, input_dict):
        
        ## -- pcd encoder
        # pcd ptv3 encoder
        fea_enc = self.encoder(input_dict)
        pc_embeddings = fea_enc['feat'].unsqueeze(0)

        ## -- query encoder
        idx, sampled_pc = self.pc_fps(input_dict['coord'])
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
                
        ## cross attn block
        cross_attn, cross_ff = self.cross_attend_blocks
        latent= cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None, enable_flash = self.encoder_dict['enable_ca_flash']) + sampled_pc_embeddings
        latent = cross_ff(latent) + latent

        return latent,sampled_pc

    def decode(self, latent, query_points):
        output = self.decoder(latent, query_points)
        return output

    def forward_encode_latent(self, input_dict):
        # encoder
        latent,sampled_pc = self.encode(input_dict)
        return latent,sampled_pc
        
    def forward(self, input_dict):
        # encoder
        latent,_ = self.encode(input_dict)
    
        # decoder
        query_points = input_dict['pc_query'].unsqueeze(0)
        output = self.decode(latent, query_points)
        output = output.squeeze(-1)
        return {'logits': output}