"""Building blocks for the looped (weight-shared) Pi3 decoder, following DejaView (DVLT, NeurIPS 2026).

A loop body of L (frame-attention, global-attention) block pairs is applied K times with shared weights.
Each application is conditioned on the time interval (t_k, t_{k+1}) through three channel-wise scales:
    x <- x + s_attn * LS1(Attn(LN(x)))
    x <- x + s_mlp  * LS2(MLP(LN(x)))
    x <- s_out * x                       (state gate)
with s = 1 + MLP(gamma(t_k, t_{k+1})). As in the official DVLT code (IntervalDepthScaling), every block owns
a small gate MLP (2 x 64 sinusoidal features -> 64 -> 3 x dim) whose last layer is zero-initialized, so at init
every gate is exactly 1 and the block behaves like a plain BlockRope.
"""

import math

import torch
import torch.nn as nn

from .block import BlockRope


class IntervalGate(nn.Module):
    """DVLT IntervalDepthScaling: sinusoidal features of (t_now, t_next) -> MLP -> (3, dim) gates centered at 1."""

    def __init__(self, dim: int, hidden_dim: int = 64, num_gates: int = 3):
        super().__init__()
        half = hidden_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.num_gates = num_gates
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_gates * dim),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def _sinusoidal(self, t: float) -> torch.Tensor:
        args = t * self.freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, t_now: float, t_next: float) -> torch.Tensor:
        emb = torch.cat([self._sinusoidal(t_now), self._sinusoidal(t_next)], dim=-1)
        emb = emb.to(self.proj[0].weight.dtype)
        return 1.0 + self.proj(emb).view(self.num_gates, self.dim)


class GatedBlockRope(BlockRope):
    """BlockRope with an optional own interval gate. Block parameter names are identical to BlockRope
    (the gate lives under `.gate`), so weights of a regular Pi3 decoder block can be loaded directly."""

    def __init__(self, *args, gate_hidden_dim: int = 64, use_gate: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.norm1.normalized_shape[0]
        self.gate = IntervalGate(dim, hidden_dim=gate_hidden_dim) if use_gate else None

    def gates(self, t_now: float, t_next: float, ref: torch.Tensor) -> torch.Tensor:
        """(3, dim) = (s_attn, s_mlp, s_out); all ones when the gate is disabled."""
        if self.gate is None:
            return ref.new_ones(3, ref.shape[-1])
        return self.gate(t_now, t_next).to(ref.dtype)

    def forward_gated(self, x, s_attn, s_mlp, s_out, xpos=None, xpos_max=None):
        x = x + s_attn * self.ls1(self.attn(self.norm1(x), xpos=xpos, xpos_max=xpos_max))
        x = x + s_mlp * self.ls2(self.mlp(self.norm2(x)))
        return s_out * x
