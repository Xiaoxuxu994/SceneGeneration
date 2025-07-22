
from torch import nn
from src.models.transformer.transformer_base import PointEmbed,PreNorm,Attention,FeedForward,cache_fn,exists
from timm.models.layers import DropPath
import torch

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



    def forward(self, x, queries, chunk_size: int = None):
        for self_attn, self_ff in self.layers:
            x = self_attn(x, enable_flash=self.enable_flash, attn_dtype=self.attn_dtype) + x
            x = self_ff(x) + x

        queries_embeddings = self.point_embed(queries)  # [B, N, C]
        B, N, C = queries_embeddings.shape

        if chunk_size is not None and N > chunk_size:
            # chunked cross-attention
            latents_chunks = []
            for i in range(0, N, chunk_size):
                qe_chunk = queries_embeddings[:, i:i+chunk_size, :]  # [B, chunk_size, C]
                latents_chunk = self.decoder_cross_attn(qe_chunk, context=x, enable_flash=self.enable_ca_flash)
                if exists(self.decoder_ff):
                    latents_chunk = latents_chunk + self.decoder_ff(latents_chunk)
                latents_chunks.append(latents_chunk)
            latents = torch.cat(latents_chunks, dim=1)
        else:
            # full cross-attention
            latents = self.decoder_cross_attn(queries_embeddings, context=x, enable_flash=self.enable_ca_flash)
        
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        return self.to_outputs(latents), x