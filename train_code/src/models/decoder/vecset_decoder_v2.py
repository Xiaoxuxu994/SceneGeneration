
from torch import nn
from src.models.transformer.transformer_base import PointEmbed,PreNorm,Attention,FeedForward,cache_fn,exists
from timm.models.layers import DropPath

from ...registry import MODELS
@MODELS.register_module()
class VecSetDecoderV2(nn.Module):
    def __init__(
        self,
        depth = 24,
        dim = 512,
        queries_dim = 512,
        output_dim = 1,
        heads = 8,
        dim_head = 64,
        enable_flash = False,
        attn_dtype = 'fp16',
        enable_ca_flash = False,
        weight_tie_layers = False,
        decoder_ff = False,
        drop_path_rate = 0.1,
        seg_head = False,
        seg_class = 100,
        seg_dim = 512,
        seg_head_drop_rate=0.0,
    ):
        super().__init__()

        self.depth = depth
        self.enable_flash = enable_flash
        self.attn_dtype = attn_dtype
        self.enable_ca_flash = enable_ca_flash
        self.drop_path_rate = drop_path_rate
        self.point_embed = PointEmbed(dim=dim)
        self.seg_head = seg_head
        self.seg_class = seg_class 
        self.seg_dim = seg_dim
        self.seg_head_drop_rate = seg_head_drop_rate

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=self.drop_path_rate))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=self.drop_path_rate))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        if not self.enable_ca_flash:
            self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        else:
            self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 32, dim_head = dim // 32), context_dim = dim)

        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

        if self.seg_head:
            self.to_outputs_seg = nn.Linear(self.seg_dim,self.seg_class)
            self.seg_head_drop_path = DropPath(self.seg_head_drop_rate) if self.seg_head_drop_rate > 0. else nn.Identity()



    def forward(self, x, queries):
        
        for self_attn, self_ff in self.layers:
            x = self_attn(x, enable_flash = self.enable_flash,attn_dtype=self.attn_dtype) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x, enable_flash = self.enable_ca_flash)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents),x


    # def forward(self, x, input_dict):
    #     queries_0p1, queries_grid_0p1 = input_dict['']
    #     queries_0p05, queries_grid_0p05 = input_dict['']

    #     ## first level latent for query 0.1
    #     for self_attn, self_ff in self.layers[:12]:
    #         x = self_attn(x, enable_flash = self.enable_flash,attn_dtype=self.attn_dtype) + x
    #         x = self_ff(x) + x

    #     queries_embeddings = self.point_embed(queries_0p1)
    #     latents = self.decoder_cross_attn0p1(queries_embeddings, context = x, enable_flash = self.enable_ca_flash)
    #     out_0p1 = self.to_outputs0p1(latents)
        
    #     ## second level latent for query 0.05
    #     # based on previous level to select 0.05 voxel for this level - check the memory or use for loop
    #     queries_grid_0p05_to_0p1 = queries_grid_0p05 // self.scale
    #     mask = (queries_grid_0p05_to_0p1[:, None, :] == queries_grid_0p1[None, :, :]).all(dim=2)
    #     matching_indices = mask.any(dim=1)
    #     queries_0p05 = queries_0p05[matching_indices] 
        

    #     for self_attn, self_ff in self.layers[12:]:
    #         x = self_attn(x, enable_flash = self.enable_flash,attn_dtype=self.attn_dtype) + x
    #         x = self_ff(x) + x
    #     queries_embeddings = self.point_embed(queries_0p05)
    #     latents = self.decoder_cross_attn0p05(queries_embeddings, context = x, enable_flash = self.enable_ca_flash)
    #     out_0p05 = self.to_outputs0p05(latents)
        
    #     # update input_dict
        
    #     return out_0p1, out_0p05