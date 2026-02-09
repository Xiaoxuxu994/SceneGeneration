import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange

from .mlp import MLP
from .utils import init_weights
from submodules.ProPE import PropeDotProductAttention

from utils.embedder import AtomicAttnPEHelper

# try:
#     import xformers.ops as xops
#     XFORMER_ENABLED = True
# except:
#     print("[ INFO ] xformer is not found or supported on this platform, using vanilla attention instead.")
#     XFORMER_ENABLED = False

# print("[ INFO ] xformer is not found or supported on this platform, using vanilla attention instead.")
XFORMER_ENABLED = False


class RoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, D, 2).float().to(device) / D)
            )
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        assert (
            tokens.size(3) % 2 == 0
        ), "number of dimensions should be a multiple of two"
        D = tokens.size(3) // 2
        assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2
        cos, sin = self.get_cos_sin(
            D, int(positions.max()) + 1, tokens.device, tokens.dtype
        )
        # split features into two along the feature dimension, and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        tokens = torch.cat((y, x), dim=-1)
        return tokens


def get_positions(token, ph, pw, device):
    """Create positional embedding for token list.

    Args:
        token (torch.Tensor): The token to generate positional embedding for, tensor of shape (b, l, d)
        ph (_type_): Height of patchified image.
        pw (_type_): Width of patchified image.
        device: torch device.

    Returns:
        pe (torch.Tensor): Generated patch positionals, tensor of shape (b, l, 2)
    """
    x = torch.arange(pw, device=device)
    y = torch.arange(ph, device=device)
    positions = torch.cartesian_prod(y, x)
    positions = positions.view(1, ph * pw, 2).expand(token.shape[0], -1, -1).clone()
    return positions


class RMSNorm(nn.Module):
    """
    RMS normalization layer. 
    Reference: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
    """
    
    def __init__(self, dim: int, eps: float=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.type_as(x)
    

def multihead_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                        pe_helper: AtomicAttnPEHelper=None) -> torch.Tensor:
    '''
    Multihead memory efficient attention block, following `"Self-Attention Does Not Need O(n^2) Memory" <http://arxiv.org/abs/2112.05682>`_.
    Args:
        q (torch.Tensor): Query tokens of shape (n, l, n_head, d_head).
        k (torch.Tensor): Key tokens of shape (n, l, n_head, d_head).
        v (torch.Tensor): Value tokens of shape (n, l, n_head, d_head).
        attn_bias (torch.Tensor): Bias applied to attention matrix, defaults to None.
        p (float): Dropout probability to attention matrix, defaults to None.
    Returns:
        out (torch.Tensor): Result of QKV attention process.
    '''
    q = rearrange(q, "n l nh dh -> n nh l dh")
    k = rearrange(k, "n l nh dh -> n nh l dh")
    v = rearrange(v, "n l nh dh -> n nh l dh")
    
    assert pe_helper is not None
    if pe_helper is not None:
        q = pe_helper.pe_q(q)
        k = pe_helper.pe_k(k)
        v = pe_helper.pe_v(v)
    
    """Official attention implementation.
    """
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    
    if pe_helper is not None:
        out = pe_helper.pe_o(out)
    
    return rearrange(out, "n nh l dh -> n l nh dh")


class QK_Norm_Attention(nn.Module):
    """
    Attention with optional Q-K normalization.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self, 
        dim, 
        head_dim, 
        qkv_bias=False, 
        fc_bias=False, 
        attn_dropout=0.0, 
        fc_dropout=0.0, 
        use_qk_norm=True, 
    ):
        """
        Args:
            dim (int): Input dimension.
            head_dim (int): Dimension of each attention head. 
            qkv_bias (bool): Whether to use bias in QKV projection. Defaults to False. 
            fc_bias (bool): Whether to use bias in output projection. Defaults to False. 
            attn_dropout (float): Dropout probability for attention weights. Defaults to 0.
            fc_dropout (float): Dropout probability for output projection. Defaults to 0. 
            use_qk_norm (bool): Whether to use Q-K normalization. Defaults to True. 
        """
        super().__init__()
        assert dim % head_dim == 0, f"Token dimension {dim} should be devisible by head dimension {head_dim}"
        
        self.dim = dim
        self.head_dim = head_dim
        self.num_head = dim // head_dim
        self.attn_dropout = attn_dropout
        self.use_qk_norm = use_qk_norm
        
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias) # .apply(init_weights)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias) # .apply(init_weights)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias) # .apply(init_weights)
        self.fc   = nn.Linear(dim, dim, bias=fc_bias)  # .apply(init_weights)
        self.attn_fc_dropout = nn.Dropout(fc_dropout)

        # if XFORMER_ENABLED:
        #     self.attention_block = lambda q, k, v, attn_bias, p: \
        #         xops.memory_efficient_attention( \
        #             q, k, v, \
        #             attn_bias=attn_bias, \
        #             p=p, \
        #             op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp), 
        #         )
        # else:
        self.attention_block = multihead_attention
        
        # Optional Q-K normalization
        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor=None, pe_helper=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).
            y (torch.Tensor): Input tensor of shape (batch, seq_len, dim).
            attn_func (function): The atomatic attention function operating on multi-head attention mechanism.
            embedder (function): Positional embedding function for inputs, activated only if ``attn_func`` is not passed.
            pos (torch.Tensor): Helper information for ``embedder``.
        
        Returns:
            Output tensor of shape (batch, seq_len, dim).
        """
        q = self.to_q(x)
        k = self.to_k(y) if y is not None else self.to_k(x)
        v = self.to_v(y) if y is not None else self.to_v(x)
        
        q, k, v = (rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim) for t in (q, k, v))
        
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # x = self.attention_block(q, k, v, attn_bias, p=self.attn_dropout if self.training else 0.0, 
        #                          embedder=embedder, pos=pos)
        """If attn_func is not passed directly, use SDPA by default with embedder given. 
        """
        x = self.attention_block(q, k, v, pe_helper=pe_helper)
        
        x = rearrange(x, "b l nh dh -> b l (nh dh)")
        x = self.attn_fc_dropout(self.fc(x))
        
        return x


class SelfAttn_Transformer_Block(nn.Module):
    
    def __init__(
        self, 
        dim, 
        head_dim, 
        ln_bias=False, 
        attn_qk_bias=False, 
        attn_fc_bias=False, 
        attn_qk_dropout=0.0, 
        attn_fc_dropout=0.0, 
        mlp_ratio=4, 
        mlp_bias=False, 
        mlp_dropout=0.0, 
        use_qk_norm=True, 
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, bias=ln_bias)
        self.attn  = QK_Norm_Attention(
            dim=dim, 
            head_dim=head_dim, 
            qkv_bias=attn_qk_bias, 
            fc_bias=attn_fc_bias, 
            attn_dropout=attn_qk_dropout, 
            fc_dropout=attn_fc_dropout, 
            use_qk_norm=use_qk_norm, 
        )
        self.norm2 = nn.LayerNorm(dim, bias=ln_bias)
        self.mlp   = MLP(
            d_in=dim, 
            d_out=dim, 
            mlp_ratio=mlp_ratio, 
            bias=mlp_bias, 
            dropout=mlp_dropout, 
        )
    
    def forward(self, x, pe_helper=None):
        x = x + self.attn(self.norm1(x), pe_helper=pe_helper)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttn_Transformer_Block(nn.Module):
    
    def __init__(
        self, 
        dim, 
        head_dim, 
        ln_bias=False, 
        attn_qk_bias=False, 
        attn_fc_bias=False, 
        attn_qk_dropout=0.0, 
        attn_fc_dropout=0.0, 
        mlp_ratio=4, 
        mlp_bias=False, 
        mlp_dropout=0.0, 
        use_qk_norm=True, 
    ):
        super().__init__()
        self.norm1_x = nn.LayerNorm(dim, bias=ln_bias)
        self.norm1_y = nn.LayerNorm(dim, bias=ln_bias)
        self.attn  = QK_Norm_Attention(
            dim=dim, 
            head_dim=head_dim, 
            qkv_bias=attn_qk_bias, 
            fc_bias=attn_fc_bias, 
            attn_dropout=attn_qk_dropout, 
            fc_dropout=attn_fc_dropout, 
            use_qk_norm=use_qk_norm, 
        )
        self.norm2 = nn.LayerNorm(dim, bias=ln_bias)
        self.mlp   = MLP(
            d_in=dim, 
            d_out=dim, 
            mlp_ratio=mlp_ratio, 
            bias=mlp_bias, 
            dropout=mlp_dropout, 
        )
    
    def forward(self, x, y, pe_helper=None):
        x, y = self.norm1_x(x), self.norm1_y(y)
        x = x + self.attn(x, y, pe_helper=pe_helper)
        x = x + self.mlp(self.norm2(x))
        return x


class GlobalxLocalSelfAttnBlock(nn.Module):
    """
    The basic self attention block, following the design of VGGT to perform global and local attention.
    """
    
    def __init__(
        self, 
        dim,
        head_dim, 
        ln_bias=False, 
        attn_qk_bias=False, 
        attn_fc_bias=False, 
        attn_qk_dropout=0.0, 
        attn_fc_dropout=0.0, 
        use_qk_norm=True, 
    ):
        super().__init__()
        self.attention_global = SelfAttn_Transformer_Block(dim, head_dim, ln_bias, attn_qk_bias, 
                                                           attn_fc_bias, attn_qk_dropout, attn_fc_dropout, 
                                                           use_qk_norm=use_qk_norm)
        self.attention_local  = SelfAttn_Transformer_Block(dim, head_dim, ln_bias, attn_qk_bias, 
                                                           attn_fc_bias, attn_qk_dropout, attn_fc_dropout, 
                                                           use_qk_norm=use_qk_norm)
    
    def forward(self, x: torch.Tensor, N: int, 
                pe_helper_global=None, pe_helper_local=None) -> torch.Tensor:
        """Forward of self attention block.

        Args:
            x (torch.Tensor): Input token sequence, tensor of shape (B, L, D).
            N (int): Number of local groups.
            embedder: Class instance of positional embedder.
            pos (torch.Tensor): Position information of input tokens, tensor of shape (B, L, 2).

        Returns:
            torch.Tensor: Output tokens of self attention block.
        """
        b, l, d = x.shape
        assert l % N == 0, f"Length of input tokens {l} must be divisible with number of local groups {N}!"
        
        global_x = self.attention_global(x, pe_helper_global)
        x        = rearrange(global_x, "b (n g) d -> (b n) g d", n=N)
        x        = self.attention_local(x, pe_helper_local)
        local_x  = rearrange(x, "(b n) g d -> b (n g) d", n=N)
        
        return [global_x, local_x]
