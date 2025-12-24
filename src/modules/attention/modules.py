from math import sqrt
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)


# class RotaryPositionEmbedder(nn.Module):
#     def __init__(self, hidden_size: int, in_channels: int = 3):
#         super().__init__()
#         assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
#         self.hidden_size = hidden_size
#         self.in_channels = in_channels
#         self.freq_dim = hidden_size // in_channels // 2
#         self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
#         self.freqs = 1.0 / (10000 ** self.freqs)
        
#     def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
#         self.freqs = self.freqs.to(indices.device)
#         phases = torch.outer(indices, self.freqs)
#         phases = torch.polar(torch.ones_like(phases), phases)
#         return phases
        
#     def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
#         print(f"embeding: {x.shape, phases.shape}")
#         x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
#         print(f"x_complex: {x_complex.shape, phases.shape}")        
#         x_rotated = x_complex * phases
#         x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
#         return x_embed
        
#     def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             q (sp.SparseTensor): [..., N, D] tensor of queries
#             k (sp.SparseTensor): [..., N, D] tensor of keys
#             indices (torch.Tensor): [..., N, C] tensor of spatial positions
#         """
#         if indices is None:
#             indices = torch.arange(q.shape[-2], device=q.device)
#             if len(q.shape) > 2:
#                 indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))
#         print(f"indices: {indices.shape}")
#         phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
#         print(f"phases: {phases.shape}")
#         if phases.shape[1] < self.hidden_size // 2:
#             phases = torch.cat([phases, torch.polar(
#                 torch.ones(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device),
#                 torch.zeros(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device)
#             )], dim=-1)
#         print(f"updated phases: {phases.shape}")            
#         q_embed = self._rotary_embedding(q, phases)
#         k_embed = self._rotary_embedding(k, phases)
#         return q_embed, k_embed
    
class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        # Ensure frequencies are on the same device as indices
        self.freqs = self.freqs.to(indices.device)
        # Compute the outer product between indices and frequency values
        phases = torch.outer(indices, self.freqs)  # Shape: [N, freq_dim]
        # Convert phases to complex numbers (real=1, imaginary=phases)
        phases = torch.polar(torch.ones_like(phases), phases)  # Shape: [N, freq_dim], complex numbers
        return phases
        
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # Shape: [..., N, F]
        
        if phases.shape[-1] < x_complex.shape[-1]:
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], x_complex.shape[-1] - phases.shape[-1], device=phases.device),
                torch.zeros(*phases.shape[:-1], x_complex.shape[-1] - phases.shape[-1], device=phases.device)
            )], dim=-1)
        elif phases.shape[-1] > x_complex.shape[-1]:
            # Interpolate phases to match x_complex's shape
            phases = phases.float()
            target_size = x_complex.shape[-1]
            phases = F.interpolate(phases, size=target_size, mode='linear', align_corners=False).bfloat16()
        x_rotated = x_complex * phases.unsqueeze(-2)
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (sp.SparseTensor): [..., N, D] tensor of queries
            k (sp.SparseTensor): [..., N, D] tensor of keys
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))
        
        # Calculate the phases for the rotary embedding
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)  # Shape: [..., freq_dim]
        
        # Apply rotary embedding to queries and keys
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        
        return q_embed, k_embed



class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        backend: str = 'flash_attn_npu',
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.scale = 1 / sqrt(self.head_dim)
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
            
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)
        self.backend = backend
        self.max_cmp_seq_kvlen = 14000
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None, 
                kv_index: torch.Tensor = None,
                return_scores: bool = False, backend: str = 'flash_attn_npu') -> torch.Tensor:
        """
        NOTE: assume return_scores = True only when doing cross-attention
        """
        B, L, C = x.shape
        attn_kwargs = dict(head_num=self.num_heads, scale=self.scale, BACKEND=self.backend, kv_index=kv_index)
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.use_rope:
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                # qkv = torch.stack([q, k, v], dim=2)
            if self.attn_mode == "full":
                q, k, v = qkv.unbind(dim=2)
                if self.qk_rms_norm:
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    # print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
                    h = scaled_dot_product_attention(q, k, v, return_scores=return_scores, **attn_kwargs)
                else:
                    # print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
                    h = scaled_dot_product_attention(q, k, v, return_scores=return_scores, **attn_kwargs)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                if return_scores:
                    h, attn_weights = scaled_dot_product_attention(q, k, v, return_scores=return_scores, **attn_kwargs)
                    return h, attn_weights
                else:
                    h = scaled_dot_product_attention(q, k, v, return_scores=return_scores, **attn_kwargs)
            else:
                if return_scores:
                    h, attn_weights = scaled_dot_product_attention(q, k, v, return_scores=return_scores, **attn_kwargs)
                    return h.reshape(B, L, -1), attn_weights.mean(dim=1)
                else:
                    k, v = kv.unbind(dim=2)
                    h = scaled_dot_product_attention(q, k, v, return_scores=return_scores, **attn_kwargs)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h
