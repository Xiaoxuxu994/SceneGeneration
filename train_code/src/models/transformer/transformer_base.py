from functools import wraps
import numpy as np
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
import torch.distributed as dist

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def linear_manual_nd(x, linear_layer: nn.Linear):
    # x: [B, N, C], weight: [D_out, C]
    weight = linear_layer.weight
    bias = linear_layer.bias
    out = torch.matmul(x, weight.T.contiguous())
    if bias is not None:
        out += bias
    return out

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, *args, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

# class Attention(nn.Module):
#     def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
#         super().__init__()
#         inner_dim = dim_head * heads
#         context_dim = default(context_dim, query_dim)
#         self.scale = dim_head ** -0.5
#         self.heads = heads

#         self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
#         self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
#         self.to_out = nn.Linear(inner_dim, query_dim)
#         self.drop_path_rate = drop_path_rate

#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

#     def forward(self, x, context = None, mask = None, enable_flash = False, attn_dtype='fp16'):
#         h = self.heads

#         # q = self.to_q(x)
#         q = linear_manual_nd(x, self.to_q)
#         context = default(context, x)
#         # k, v = self.to_kv(context).chunk(2, dim = -1)
#         k, v = linear_manual_nd(context, self.to_kv).chunk(2, dim=-1)

#         if not enable_flash:
#             q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
#             sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
#             if exists(mask):
#                 mask = rearrange(mask, 'b ... -> b (...)')
#                 max_neg_value = -torch.finfo(sim.dtype).max
#                 mask = repeat(mask, 'b j -> (b h) () j', h = h)
#                 sim.masked_fill_(~mask, max_neg_value)

#             # attention, what we cannot get enough of
#             attn = sim.softmax(dim = -1)
#             out = torch.einsum('b i j, b j d -> b i d', attn, v)
#             out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
#         else:
#             if attn_dtype == 'fp16':
#                 out = torch_npu.npu_fusion_attention(
#                     q.half(),
#                     k.half(),
#                     v.half(),
#                     num_heads = h,
#                     scale_value = self.scale,
#                     input_layout = "BSH",
#                 )[0]
#             if attn_dtype == 'bf16':
#                 out = torch_npu.npu_fusion_attention(
#                     q.bfloat16(),
#                     k.bfloat16(),
#                     v.bfloat16(),
#                     head_num=h,
#                     scale=self.scale,
#                     input_layout="BSH"
#                 )[0]
#             out = out.to(q.dtype)
#         return self.drop_path(linear_manual_nd(out, self.to_out))


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.drop_path_rate = drop_path_rate

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, x_offset, context=None, context_offset=None, mask=None, enable_flash=False, attn_dtype='fp16'):
        # x[1,s1+s2+..+sb, d]
        # x_offset[s1, s1+s2,...,s1+s2+...+sb] : torch.Tensor
        # context[1, r1+r2+..rb, d]
        # context_offset[r1,r1+r2,...,r1+r2+...+rb] : torch:Tensor
        h = self.heads  
        context = default(context, x)
        b = x_offset.shape[0]
        x_offset_ext = torch.cat([torch.tensor([0],dtype=x_offset.dtype, device=x_offset.device), x_offset], dim=0)
        context_offset = default(context_offset, x_offset)
        context_offset_ext = torch.cat([torch.tensor([0],dtype=context_offset.dtype, device=context_offset.device), context_offset], dim=0)
        # qkv attention projection
        # q = self.to_q(x)
        q = linear_manual_nd(x, self.to_q)
        # k, v = self.to_kv(context).chunk(2, dim = -1)
        k, v = linear_manual_nd(context, self.to_kv).chunk(2, dim=-1)

        if not enable_flash:
            out_list = []
            for i in range(b):
                qi, ki, vi = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h),
                              (q[:, x_offset_ext[i]:x_offset_ext[i+1]], 
                               k[:, context_offset_ext[i]:context_offset_ext[i+1]],  
                               v[:, context_offset_ext[i]:context_offset_ext[i+1]]))
                sim = torch.einsum('b i d, b j d -> b i j', qi, ki) * self.scale
                # sim = (qi * self.scale) @ ki.transpose(-2, -1)
                if exists(mask):
                    mask = rearrange(mask, 'b ... -> b (...)')
                    max_neg_value = - torch.finfo(sim.dtype).max
                    mask = repeat(mask, 'b j -> (b h) () j', h = h)
                    sim.masked_fill_(~mask, max_neg_value)
                # attention, what we cannot get enough of
                attn = sim.softmax(dim = -1)
                out = torch.einsum('b i j, b j d -> b i d', attn, vi)
                # out = attn @ vi
                out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
                out_list.append(out)
            out = torch.concat(out_list, dim = 1)
        else:
            q = q.squeeze(0).view(x_offset_ext[-1], h, -1)
            k = k.squeeze(0).view(context_offset_ext[-1], h, -1)
            v = v.squeeze(0).view(context_offset_ext[-1], h, -1)
            if attn_dtype == 'fp16':
                out = torch_npu.npu_fusion_attention(
                    q.half(), k.half(), v.half(),
                    head_num=h,
                    input_layout="TND",# BSH,SBH,BSND,TND
                    actual_seq_qlen=x_offset_ext[1:].cpu().tolist(),
                    actual_seq_kvlen=context_offset_ext[1:].cpu().tolist(),
                    scale=self.scale,
                )
            if attn_dtype == 'bf16':
                out = torch_npu.npu_fusion_attention(
                    q.bfloat16(), k.bfloat16(), v.bfloat16(),
                    head_num=h,
                    input_layout="TND", # BSH,SBH,BSND,TND
                    actual_seq_qlen=x_offset_ext[1:].cpu().tolist(),
                    actual_seq_kvlen=context_offset_ext[1:].cpu().tolist(),
                    scale=self.scale,
                )
            out = out[0].view(1, x_offset[-1], -1)
            out = out.to(q.dtype)
        return self.drop_path(linear_manual_nd(out, self.to_out))

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed