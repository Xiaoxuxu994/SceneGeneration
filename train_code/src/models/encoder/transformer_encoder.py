
from torch import nn
from src.models.transformer.transformer_base import PreNorm,Attention,FeedForward,cache_fn

from ...registry import MODELS
@MODELS.register_module()
class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        depth = 24,
        dim = 512,
        heads = 8,
        dim_head = 64,
        enable_flash = False,
        attn_dtype = 'fp16',
        drop_path_rate = 0.1,
    ):
        super().__init__()

        self.depth = depth
        self.enable_flash = enable_flash
        self.attn_dtype = attn_dtype

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=drop_path_rate))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=drop_path_rate))

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(),
                get_latent_ff()
            ]))


    def forward(self, x):
        
        for self_attn, self_ff in self.layers:
            x = self_attn(x, enable_flash = self.enable_flash,attn_dtype=self.attn_dtype) + x
            x = self_ff(x) + x
       
        return x