import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim_query: int,
        dim_context: int,
        embed_dim: int,
        num_heads: int = 16,
        mlp: bool = True,
        pre_norm: bool = True,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        """
        Single cross-attention block supporting different query/context dims.

        Args:
            dim_query: Dimension of query tokens (Eq).
            dim_context: Dimension of context tokens (Ek / Ev).
            embed_dim: Attention projection dimension (internal).
            num_heads: Number of attention heads.
            mlp: Whether to include MLP after attention.
            pre_norm: Pre-Norm (LN before attention) vs Post-Norm.
            dropout: Dropout for MLP.
            batch_first: Whether input shape is [B, L, D] or [L, B, D].
        """
        super().__init__()
        self.pre_norm = pre_norm
        self.mlp_on = mlp

        # MultiheadAttention handles different input dims via kdim/vdim
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=dim_context,
            vdim=dim_context,
            batch_first=batch_first,
        )

        self.ln1 = nn.LayerNorm(embed_dim)

        if mlp:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            self.ln2 = nn.LayerNorm(embed_dim)

        # Optional projection back to query dim if different from embed_dim
        self.out_proj = nn.Linear(embed_dim, dim_query) if embed_dim != dim_query else nn.Identity()

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor):
        """
        Args:
            query_tokens: [B, Lq, Eq] if batch_first else [Lq, B, Eq]
            context_tokens: [B, Lc, Ec] if batch_first else [Lc, B, Ec]
        Returns:
            Updated query_tokens: [B, Lq, Eq] (same as input)
        """
        q_input = query_tokens

        if self.pre_norm:
            # Pre-Norm: normalize query before attention
            q_norm = self.ln1(q_input)
            attn_out, _ = self.attn(q_norm, context_tokens, context_tokens, need_weights=False)
            q_input = q_input + attn_out

            if self.mlp_on:
                q_input = q_input + self.mlp(self.ln2(q_input))

        else:  # Post-Norm
            attn_out, _ = self.attn(q_input, context_tokens, context_tokens, need_weights=False)
            q_input = self.ln1(q_input + attn_out)

            if self.mlp_on:
                q_input = self.ln2(q_input + self.mlp(q_input))

        return self.out_proj(q_input)


class CrossAttentionStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim_query: int,
        dim_context: int,
        embed_dim: int,
        num_heads: int = 16,
        mlp: bool = True,
        pre_norm: bool = True,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                dim_query=dim_query,
                dim_context=dim_context,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp=mlp,
                pre_norm=pre_norm,
                dropout=dropout,
                batch_first=batch_first,
            )
            for _ in range(num_layers)
        ])

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor):
        for layer in self.layers:
            query_tokens = layer(query_tokens, context_tokens)
        return query_tokens
