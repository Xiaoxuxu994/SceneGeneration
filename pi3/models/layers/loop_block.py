"""Building blocks for the looped (weight-shared) Pi3 decoder, following DejaView (DVLT, NeurIPS 2026).

One frame-attention block and one global-attention block are applied K times with shared weights.
Each application is conditioned on the time interval (t_k, t_{k+1}) through three channel-wise scales:
    x <- x + s_attn * LS1(Attn(LN(x)))
    x <- x + s_mlp  * LS2(MLP(LN(x)))
    x <- s_out * x                       (state gate)
with s = 1 + MLP(gamma(t_k, t_{k+1})). The MLP is zero-initialized, so at init every gate is exactly 1
and the block behaves like a plain BlockRope.
"""

import math

import torch
import torch.nn as nn

from .block import BlockRope


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """t: (...,) in [0, 1] -> (..., dim) sin/cos features (t is scaled by 1000 like diffusion timesteps)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = (t.float() * 1000.0)[..., None] * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class IntervalEmbedding(nn.Module):
    """gamma(t_k, t_{k+1}) followed by a zero-initialized MLP producing (num_scales, dim) gates (as 1 + delta)."""

    def __init__(self, dim: int, num_scales: int, freq_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.freq_dim = freq_dim
        self.num_scales = num_scales
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_scales * dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([sinusoidal_embedding(t0, self.freq_dim), sinusoidal_embedding(t1, self.freq_dim)], dim=-1)
        emb = emb.to(self.mlp[0].weight.dtype)
        return 1.0 + self.mlp(emb).view(self.num_scales, self.dim)


class GatedBlockRope(BlockRope):
    """BlockRope with optional per-call channel gates. Parameter names are identical to BlockRope,
    so weights of a regular Pi3 decoder block can be loaded directly."""

    def forward_gated(self, x, s_attn, s_mlp, s_out, xpos=None, xpos_max=None):
        x = x + s_attn * self.ls1(self.attn(self.norm1(x), xpos=xpos, xpos_max=xpos_max))
        x = x + s_mlp * self.ls2(self.mlp(self.norm2(x)))
        return s_out * x
