from functools import wraps
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
import torch.distributed as dist

try:
    import flash_attn
except ImportError:
    flash_attn = None

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

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

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

    def forward(self, x, context = None, mask = None, enable_flash = False, attn_dtype='fp16'):
        h = self.heads
        
        # q = self.to_q(x)
        q = linear_manual_nd(x, self.to_q)
        context = default(context, x)
        # k, v = self.to_kv(context).chunk(2, dim = -1)
        k, v = linear_manual_nd(context, self.to_kv).chunk(2, dim=-1)

        if not enable_flash:
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h = h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim = -1)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        
        else:
            import torch_npu
            if attn_dtype == 'fp16':
                out = torch_npu.npu_prompt_flash_attention(
                    q.half(),
                    k.half(),
                    v.half(),
                    num_heads = h,
                    scale_value = self.scale,
                    input_layout = "BSH",
                )
            if attn_dtype == 'bf16':
                # out = torch_npu.npu_prompt_flash_attention(
                #     q.bfloat16(),
                #     k.bfloat16(),
                #     v.bfloat16(),
                #     num_heads = h,
                #     scale_value = self.scale,
                #     input_layout = "BSH",
                # )
                
                out = torch_npu.npu_fusion_attention(
                    q.bfloat16(),
                    k.bfloat16(),
                    v.bfloat16(),
                    head_num=h,
                    scale=self.scale,
                    input_layout="BSH"
                )[0]
                # print(out.requires_grad)
            # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = h), (q, k, v))
            # ## enable flash_attn
            # if attn_dtype == 'fp16':
            #     out = flash_attn.flash_attn_func(q.half(), k.half(), v.half(), dropout_p= self.drop_path_rate, softmax_scale=self.scale, causal=False,
            #             window_size=(-1, -1), alibi_slopes=None, deterministic=False)  
            # if attn_dtype == 'bf16':
            #     out = flash_attn.flash_attn_func(q.bfloat16(), k.bfloat16(), v.bfloat16(), dropout_p= self.drop_path_rate, softmax_scale=self.scale, causal=False,
            #             window_size=(-1, -1), alibi_slopes=None, deterministic=False)  
            # out = rearrange(out, 'b n h d -> b n (h d)', h = h)
            out = out.to(q.dtype)
        return self.drop_path(linear_manual_nd(out, self.to_out))


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128, use_normal=False):
        super().__init__()

        assert hidden_dim % 6 == 0
        self.embedding_dim = hidden_dim
        self.use_normal = use_normal
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

        in_dim = self.embedding_dim + 3 + (3 if use_normal else 0)
        self.mlp = nn.Linear(in_dim, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        '''
        input:
            - B x N x 3: xyz
            - B x N x 6: xyz+normal
        '''
        xyz = input[...,:3]
        xyz_pe = self.embed(xyz, self.basis)
        feats = [xyz_pe, xyz]
        if self.use_normal and input.size(-1) >=6 :
            normal = input[...,3:6]
            feats.append(normal)
        feat = torch.cat(feats, dim=-1)
        return self.mlp(feat)


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean