from typing import *
import numpy as np
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32

# %%
class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """
    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed

class PTV3PosEmbedding(nn.Module):
    def __init__(self, hidden_dim=48, dim=128, act_layer=None, isLog=False, max_freq=1000.0):
        super().__init__()

        assert hidden_dim % 6 == 0, "hidden_dim 必须是 6 的倍数（因为用了 sin+cos 和 3D）"

        self.embedding_dim = hidden_dim

        # 构造三维方向频率基底（3 x [hidden_dim // 6]）
        if isLog:
            freq = torch.logspace(0.0, np.log10(max_freq), hidden_dim // 6) * np.pi  # [num_freqs]
        else:
            freq = torch.pow(2, torch.arange(hidden_dim // 6)).float() * np.pi
            
        basis = torch.stack([
            torch.cat([freq, torch.zeros_like(freq), torch.zeros_like(freq)]),
            torch.cat([torch.zeros_like(freq), freq, torch.zeros_like(freq)]),
            torch.cat([torch.zeros_like(freq), torch.zeros_like(freq), freq]),
        ])
        self.register_buffer('basis', basis)  # shape [3, hidden_dim // 2]

        # 前馈网络，输入是 embedding + 原始坐标
        layers = [nn.Linear(hidden_dim + 3, dim)]
        if act_layer is not None:
            layers.append(act_layer())
        self.stem = nn.Sequential(*layers)

    def embed(self, x, basis):
        """
        Args:
            x: [B, N, 3]
            basis: [3, F]
        Returns:
            embedded: [B, N, 2F]
        """
        proj = torch.einsum('bnd,de->bne', x, basis)  # [B, N, F]
        return torch.cat([proj.sin(), proj.cos()], dim=-1)  # [B, N, 2F]

    def forward(self, coords: torch.Tensor, feat: torch.Tensor = None):
        """
        Args:
            coords: [N, 3] or [B, N, 3] point coordinates
            feat: optional [N, C] features, if provided will be ignored
        Returns:
            [N, dim] embedded features
        """
        if len(coords.shape) == 2:
            x = coords.unsqueeze(0)  # [1, N, 3]
        else: 
            x = coords
        emb = self.embed(x, self.basis)  # [1, N, hidden_dim]
        inp = torch.cat([emb, x], dim=-1)  # [1, N, hidden_dim + 3]
        out = self.stem(inp)  # [1, N, dim]
        return out.squeeze()  # [N, dim]

"""
voxel_xyz → [FourierEmbedding or MLP] → PosEncoding
voxel_index → learnable Embedding Table → GridEncoding
"""

class FourierPositionalEncoding(nn.Module):
    def __init__(self, input_dim=3, num_frequencies=10, scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.scale = scale

        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies) * scale
        self.register_buffer('freq_bands', freq_bands)

        self.out_dim = input_dim * num_frequencies * 2

    def forward(self, x):  # x: [N, 3]
        x = x.unsqueeze(-1) * self.freq_bands  # [N, 3, F]
        x = x.view(x.shape[0], -1)  # flatten freq dim
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [N, 2 * 3 * F]

class DualPathPositionalEncoding(nn.Module):
    def __init__(
        self,
        max_voxel_dim: int = 128,  # 假设 x/y/z 的最大索引不超过这个值
        index_embed_dim: int = 32,  # 每个 axis 的 embedding dim
        coord_embed_dim: int = 256,
        coord_type: str = 'fourier',  # or 'mlp'
        mlp_hidden_dim: int = 128,
        num_freqs: int = 10,
    ):
        super().__init__()

        # 三个 embedding table 分别对应 x/y/z 索引
        self.index_embedding_x = nn.Embedding(max_voxel_dim, index_embed_dim)
        self.index_embedding_y = nn.Embedding(max_voxel_dim, index_embed_dim)
        self.index_embedding_z = nn.Embedding(max_voxel_dim, index_embed_dim)
        self.index_embed_dim = index_embed_dim * 3  # 合并后维度

        # coord encoder
        if coord_type == 'fourier':
            self.coord_encoder = FourierPositionalEncoding(input_dim=3, num_frequencies=num_freqs)
            self.coord_proj = nn.Linear(self.coord_encoder.out_dim, coord_embed_dim)
        elif coord_type == 'mlp':
            self.coord_encoder = nn.Sequential(
                nn.Linear(3, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, coord_embed_dim)
            )
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")

        self.total_dim = self.index_embed_dim + coord_embed_dim

    def forward(self, voxel_xyz: torch.Tensor, voxel_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel_xyz: [N, 3] float coordinates
            voxel_index: [N, 3] long indices along x, y, z
        Returns:
            [N, total_dim] combined positional encoding
        """
        ix, iy, iz = voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]  # [N]

        ex = self.index_embedding_x(ix)
        ey = self.index_embedding_y(iy)
        ez = self.index_embedding_z(iz)

        idx_feat = torch.cat([ex, ey, ez], dim=-1)  # [N, index_embed_dim * 3]

        if isinstance(self.coord_encoder, FourierPositionalEncoding):
            coord_feat = self.coord_encoder(voxel_xyz)  # [N, F]
            coord_feat = self.coord_proj(coord_feat)   # [N, coord_embed_dim]
        else:
            coord_feat = self.coord_encoder(voxel_xyz)  # [N, coord_embed_dim]

        return torch.cat([idx_feat, coord_feat], dim=-1)  # [N, total_dim]

# %%
class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN).
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[int] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )

    def _forward(self, x: torch.Tensor, return_scores: bool = False) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, return_scores=return_scores)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class TransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN).
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )

    def _forward(self, x: torch.Tensor, context: torch.Tensor, return_scores: bool = False):
        h = self.norm1(x)
        h = self.self_attn(h)
        x = x + h
        h = self.norm2(x)
        if return_scores:
            h, weights = self.cross_attn(h, context, return_scores=return_scores)
            return h, weights
        else:
            h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor, return_scores: bool = False):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, context, return_scores, use_reentrant=False)
        else:
            return self._forward(x, context, return_scores)
        