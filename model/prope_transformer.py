from typing import Callable, Literal, Optional, Type

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, LayerNorm, Linear, Dropout, Sequential, Parameter
from torch.nn.init import constant_, xavier_uniform_
from einops import rearrange

from prope.torch import PropeDotProductAttention

def _get_clones(module, N):
    return nn.ModuleList([module for _ in range(N)])


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)
    
    
class MultiheadAttention(torch.nn.Module):
    """
    Same as torch.nn.MultiheadAttention for bidirectional attention, but supports:
    - RMSNorm
    - customized attention function
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.bias = bias
        self.qk_norm = qk_norm

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))

        if self.bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sdpa_fn: Callable = F.scaled_dot_product_attention,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, T, D)
            key: (B, T, D)
            value: (B, T, D)
            sdpa_fn: (q, k, v, **kwargs) -> Tensor
        Returns:
            output: (B, T, D)
        """
        q_proj_weight, k_proj_weight, v_proj_weight = self.in_proj_weight.chunk(
            3, dim=0
        )
        if self.in_proj_bias is not None:
            q_proj_bias, k_proj_bias, v_proj_bias = self.in_proj_bias.chunk(3, dim=0)
        else:
            q_proj_bias = k_proj_bias = v_proj_bias = None

        q = F.linear(query, q_proj_weight, q_proj_bias)
        k = F.linear(key, k_proj_weight, k_proj_bias)
        v = F.linear(value, v_proj_weight, v_proj_bias)
        q, k, v = (
            rearrange(x, "b t (h c) -> b t h c", h=self.num_heads) for x in [q, k, v]
        )
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k, v = (rearrange(x, "b t h c -> b h t c") for x in [q, k, v])
        o = sdpa_fn(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        o = rearrange(o, "b h t c -> b t (h c)", h=self.num_heads)
        attn_output = F.linear(o, self.out_proj.weight, self.out_proj.bias)

        return attn_output
    
    
class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        bias: bool = True,
        qk_norm: bool = False,
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
        elementwise_affine: bool = True,
        activation: nn.Module = nn.ReLU(),
        norm_type: str = "standard",
        modulation_activation: nn.Module = None,
    ) -> None:
        super().__init__()

        if norm_type == "adaLN-Zero":
            assert modulation_activation is not None, \
                "modulation_activation must be provided for adaLN-Zero"
            assert norm_first, "only norm_first=True is supported for adaLN-Zero"

        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            qk_norm=qk_norm,
        )

        # Feedforward
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(
            d_model,
            eps=layer_norm_eps,
            bias=bias,
            elementwise_affine=elementwise_affine,
        )
        self.norm2 = LayerNorm(
            d_model,
            eps=layer_norm_eps,
            bias=bias,
            elementwise_affine=elementwise_affine,
        )
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = activation
        self.norm_type = norm_type

        if norm_type == "adaLN-Zero":
            self.modulation_mlp = Sequential(
                modulation_activation,
                Linear(d_model, 6 * d_model, bias=bias),
            )

    def forward(
        self,
        src: torch.Tensor,
        sdpa_fn: Callable = F.scaled_dot_product_attention,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert (cond is None and self.norm_type == "layer_norm") or (
            cond is not None and self.norm_type == "adaLN-Zero"
        ), "cond must be None for layer_norm, and not None for adaLN-Zero"

        if self.norm_type == "adaLN-Zero":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation_mlp(cond).chunk(6, dim=1)
            )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            if self.norm_type == "adaLN-Zero":
                y = modulate(self.norm1(x), shift_msa, scale_msa)
                x = x + gate_msa.unsqueeze(1) * self._sa_block(y, sdpa_fn=sdpa_fn)
                x = x + gate_mlp.unsqueeze(1) * self._ff_block(
                    modulate(self.norm2(x), shift_mlp, scale_mlp)
                )
            else:
                x = x + self._sa_block(self.norm1(x), sdpa_fn=sdpa_fn)
                x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, sdpa_fn=sdpa_fn))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: torch.Tensor, sdpa_fn: Callable = F.scaled_dot_product_attention) -> torch.Tensor:
        x = self.self_attn(x, x, x, sdpa_fn=sdpa_fn)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: bool = False,
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,
        elementwise_affine: bool = True,
        activation: nn.Module = nn.ReLU(),
        norm_type: str = "layer_norm",
        modulation_activation: nn.Module = None,
        input_norm: bool = True,
        output_norm: bool = True,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()
        
        # build one encoder layer and clone
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bias=bias,
            qk_norm=qk_norm,
            norm_first=norm_first,
            layer_norm_eps=layer_norm_eps,
            elementwise_affine=elementwise_affine,
            activation=activation,
            norm_type=norm_type,
            modulation_activation=modulation_activation,
        )
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

        self.in_norm = (
            LayerNorm(
                d_model,
                eps=layer_norm_eps,
                elementwise_affine=elementwise_affine,
                bias=bias,
            )
            if input_norm
            else None
        )

        self.out_norm = (
            LayerNorm(
                d_model,
                eps=layer_norm_eps,
                elementwise_affine=elementwise_affine,
                bias=bias,
            )
            if output_norm
            else None
        )

        if norm_type == "adaLN-Zero":
            assert modulation_activation is not None, \
                "modulation_activation must be provided for adaLN-Zero"
            assert norm_first, "only norm_first=True is supported for adaLN-Zero"

            self.final_modulation_mlp = Sequential(
                modulation_activation,
                Linear(d_model, 2 * d_model, bias=bias),
            )
            self.final_norm = LayerNorm(
                d_model,
                eps=layer_norm_eps,
                elementwise_affine=elementwise_affine,
                bias=bias,
            )

        self.checkpointing = checkpointing

    def forward(
        self,
        src: torch.Tensor,
        sdpa_fn: Callable = F.scaled_dot_product_attention,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src

        if self.in_norm is not None:
            output = self.in_norm(output)

        for mod in self.layers:
            if self.checkpointing:
                output = torch.utils.checkpoint.checkpoint(
                    mod,
                    output,
                    sdpa_fn=sdpa_fn,
                    cond=cond,
                    use_reentrant=False,
                )
            else:
                output = mod(output, sdpa_fn=sdpa_fn, cond=cond)

        if self.out_norm is not None:
            output = self.out_norm(output)

        if self.layers[0].norm_type == "adaLN-Zero":
            shift, scale = self.final_modulation_mlp(cond).chunk(2, dim=-1)
            output = modulate(self.final_norm(output), shift, scale)

        return output