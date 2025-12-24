"""
Point Transformer - V3 Mode1
Pointcept detached version

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
from functools import partial
from addict import Dict
import math
import torch
import torch_npu
import torch.nn as nn
# import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath
from collections import OrderedDict
import numpy as np
from typing import Iterable, Tuple
import torch.nn.functional as F

from .serialization import encode
from .point_embed import PointEmbed

def check_mem(msg=''):
    # 查看当前NPU显存已分配的大小（以字节为单位）
    allocated_memory = torch.npu.memory_allocated()
    # 查看当前NPU显存的已保留部分（以字节为单位）
    reserved_memory = torch.npu.memory_reserved()
    print(f"{msg} Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB, Reserved memory: {reserved_memory / (1024 ** 3):.2f} GB")

def optimize_memory():
    torch_npu.clean_mem() # 清理 NPU 内存
    gc.collect() # 手动触发 Python 垃圾回收  
    torch_npu.empty_cache() # 清空 PyTorch 缓存

@torch.inference_mode()
def offset2bincount(offset: torch.Tensor) -> torch.Tensor:
    """
    offset: (B,) long, inclusive cumsum of counts
    returns: (B,) long, per-batch counts
    """
    # avoid torch.diff + prepend tiny cuda tensor
    out = offset.new_empty(offset.shape)
    out[0] = offset[0]
    out[1:] = offset[1:] - offset[:-1]
    return out

@torch.inference_mode()
def offset2batch(offset: torch.Tensor) -> torch.Tensor:
    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(batch.bincount(), dim=0).long()

def _to_cpu_maybe_pin(x: torch.Tensor, pin: bool) -> torch.Tensor:
    x = x.to("cpu")
    if pin and x.dtype in (torch.int32, torch.int64, torch.float16, torch.float32, torch.float64):
        # pin_memory 仅对 CPU tensor 有意义；dtype 限制只是为了稳妥
        try:
            x = x.pin_memory()
        except RuntimeError:
            pass
    return x

class Point(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    @torch.inference_mode()
    def ensure_serialized(self, device: torch.device, non_blocking: bool = True):
        """
        如果你把 serialized_* 存在 CPU，用之前调用它把相关张量搬回 GPU。
        """
        for k in ("serialized_code", "serialized_order", "serialized_inverse"):
            if k in self and isinstance(self[k], torch.Tensor) and self[k].device != device:
                self[k] = self[k].to(device, non_blocking=non_blocking)

    @torch.inference_mode()
    def serialization(
        self,
        order="z",
        depth=None,
        shuffle_orders=False,
        store_code=True,
        offload: Iterable[str] = ("code", "order", "inverse"),  # 选择哪些存 CPU
        pin_memory: bool = True,                               # CPU 侧 pin，便于非阻塞搬回 GPU
    ):
        assert "batch" in self.keys()

        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            coord_min = torch.amin(self.coord, dim=0)
            grid_coord = torch.div(self.coord - coord_min, self.grid_size, rounding_mode="trunc")
            self["grid_coord"] = grid_coord.int()

        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16

        if isinstance(order, str):
            order_list = list(order) if len(order) > 1 else [order]
        else:
            order_list = list(order)

        K = len(order_list)
        N = self.grid_coord.shape[0]
        device = self.grid_coord.device
        arangeN = torch.arange(N, device=device, dtype=torch.long)

        orders, inverses = [], []
        codes = [] if store_code else None

        # 逐个算，降低峰值显存
        for order_ in order_list:
            c = encode(self.grid_coord, self.batch, depth, order=order_)  # (N,)
            o = torch.argsort(c)  # (N,)
            inv = torch.empty_like(o)
            inv.scatter_(0, o, arangeN)

            orders.append(o)
            inverses.append(inv)
            if store_code:
                codes.append(c)

            del c, o, inv

        order_t = torch.stack(orders, dim=0)       # (K,N)
        inverse_t = torch.stack(inverses, dim=0)   # (K,N)
        self["serialized_order"] = order_t
        self["serialized_inverse"] = inverse_t

        if store_code:
            self["serialized_code"] = torch.stack(codes, dim=0)  # (K,N)
        else:
            self.pop("serialized_code", None)

        if shuffle_orders:
            perm = torch.randperm(K, device=device)
            if "serialized_code" in self:
                self["serialized_code"] = self["serialized_code"][perm]
            self["serialized_order"] = self["serialized_order"][perm]
            self["serialized_inverse"] = self["serialized_inverse"][perm]

        # ---- CPU offload：把大索引张量常驻从 GPU 挪走 ----
        off = set(offload or [])
        if "code" in off and "serialized_code" in self:
            self["serialized_code"] = _to_cpu_maybe_pin(self["serialized_code"], pin_memory)
        if "order" in off and "serialized_order" in self:
            self["serialized_order"] = _to_cpu_maybe_pin(self["serialized_order"], pin_memory)
        if "inverse" in off and "serialized_inverse" in self:
            self["serialized_inverse"] = _to_cpu_maybe_pin(self["serialized_inverse"], pin_memory)
    
    @torch.inference_mode()
    def sparsify(self, pad=96):
        """
        prepare spconv.SparseConvTensor
        rely on ["grid_coord" or "coord"+"grid_size", "batch", "feat"]
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            coord_min = torch.amin(self.coord, dim=0)
            grid_coord = torch.div(self.coord - coord_min, self.grid_size, rounding_mode="trunc")
            self["grid_coord"] = grid_coord.int()
        # (你的 spconv 逻辑保持原样即可)
        # if "sparse_shape" in self.keys():
        #     sparse_shape = self.sparse_shape
        # else:
        #     sparse_shape = torch.add(
        #         torch.max(self.grid_coord, dim=0).values, pad
        #     ).tolist()
        # sparse_conv_feat = spconv.SparseConvTensor(
        #     features=self.feat,
        #     indices=torch.cat(
        #         [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
        #     ).contiguous(),
        #     spatial_shape=sparse_shape,
        #     batch_size=self.batch[-1].tolist() + 1,
        # )
        # self["sparse_shape"] = sparse_shape
        # self["sparse_conv_feat"] = sparse_conv_feat


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for _ in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for _, module in self._modules.items():
            if isinstance(module, PointModule):
                input = module(input)
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(input.feat)
                else:
                    input = module(input)
        return input
    

class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = tuple(conditions)
        self.cond2idx = {c: i for i, c in enumerate(self.conditions)}
        self.decouple = decouple
        self.adaptive = adaptive

        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in self.conditions])
        else:
            self.norm = norm_layer(num_features)

        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(context_channels, 2 * num_features, bias=True),
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        cond = point.condition if isinstance(point.condition, str) else point.condition[0]
        norm = self.norm[self.cond2idx[cond]] if self.decouple else self.norm

        feat = norm(point.feat)

        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)

            # 推理/无梯度：尽量 in-place，减少临时
            if (not self.training) or (not feat.requires_grad):
                feat.addcmul_(feat, scale)  # feat = feat + feat*scale  (等价 feat*(1+scale))
                feat.add_(shift)
            else:
                feat = feat + feat * scale
                feat = feat + shift

        point.feat = feat
        return point


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)
        self.register_buffer(
            "axis_stride",
            torch.tensor([0, 1, 2], dtype=torch.long) * self.rpe_num,
            persistent=False,
        )

    def forward(self, coord):
        """
        coord: (n_patch, K, K, 3)
        return: (n_patch, H, K, K)
        """
        c = coord.clamp(-self.pos_bnd, self.pos_bnd).to(torch.long) + self.pos_bnd  # (...,3)
        n, K1, K2, _ = c.shape

        flat = c.reshape(-1, 3)  # (n*K*K, 3)
        x = flat[:, 0] + self.axis_stride[0]
        y = flat[:, 1] + self.axis_stride[1]
        z = flat[:, 2] + self.axis_stride[2]

        out = F.embedding(x, self.rpe_table)
        out = out + F.embedding(y, self.rpe_table)
        out = out + F.embedding(z, self.rpe_table)

        out = out.view(n, K1, K2, self.num_heads).permute(0, 3, 1, 2).contiguous()
        return out

    
class SerializedAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        attn_dtype="fp16",          # "fp16" or "bf16"
        upcast_attention=False,     # flash 时必须 False
        upcast_softmax=False,       # flash 时必须 False

        # ---- memory knobs ----
        chunk_patches=64,           # 越小越省显存（更慢）
        pin_cpu_cache=False,
        out_dtype="auto",           # "auto" / "fp32" / "bf16" / "fp16"
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index

        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        self.attn_dtype = attn_dtype
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.chunk_patches = int(chunk_patches)
        self.pin_cpu_cache = pin_cpu_cache
        self.out_dtype = out_dtype

        if enable_flash:
            assert enable_rpe is False, "enable_flash=True 时请关闭 RPE"
            assert upcast_attention is False, "enable_flash=True 时请关闭 upcast_attention"
            assert upcast_softmax is False, "enable_flash=True 时请关闭 upcast_softmax"
            self.patch_size = patch_size
            self.attn_drop_p = float(attn_drop)
        else:
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # 你如果后续要 RPE，可以按你已有版本接回；这里保持一致
        self.rpe = None

    def _maybe_pin(self, t_cpu: torch.Tensor):
        if self.pin_cpu_cache:
            try:
                return t_cpu.pin_memory()
            except RuntimeError:
                return t_cpu
        return t_cpu

    @torch.no_grad()
    def get_padding_and_inverse_cpu(self, point):
        """
        直接在 CPU 生成 pad/unpad/cu_seqlens，避免 NPU 上大整数张量峰值。
        key 带 patch_size(K)，因为 non-flash 下 K 可能变化。
        """
        K = self.patch_size
        pad_key = f"pad_k{K}"
        unpad_key = f"unpad_k{K}"
        cu_key = f"cu_seqlens_k{K}"

        if pad_key in point and unpad_key in point and cu_key in point:
            return point[pad_key], point[unpad_key], point[cu_key]

        offset_cpu = point.offset.detach().to("cpu")
        bincount = offset2bincount(offset_cpu)

        bincount_pad = (torch.div(bincount + K - 1, K, rounding_mode="trunc") * K)
        mask_pad = bincount > K
        bincount_pad = (~mask_pad) * bincount + mask_pad * bincount_pad

        _offset = F.pad(offset_cpu, (1, 0))
        _offset_pad = F.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))

        pad = torch.arange(_offset_pad[-1].item(), device="cpu", dtype=torch.long)
        unpad = torch.arange(_offset[-1].item(), device="cpu", dtype=torch.long)

        cu_seqlens = []
        for i in range(len(offset_cpu)):
            unpad[_offset[i]:_offset[i + 1]] += _offset_pad[i] - _offset[i]
            if bincount[i] != bincount_pad[i]:
                pad[
                    _offset_pad[i + 1] - K + (bincount[i] % K): _offset_pad[i + 1]
                ] = pad[
                    _offset_pad[i + 1] - 2 * K + (bincount[i] % K): _offset_pad[i + 1] - K
                ]
            pad[_offset_pad[i]:_offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
            cu_seqlens.append(
                torch.arange(
                    _offset_pad[i].item(),
                    _offset_pad[i + 1].item(),
                    step=K,
                    dtype=torch.int32,
                    device="cpu",
                )
            )

        cu_seqlens = F.pad(torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1].item())

        point[pad_key] = self._maybe_pin(pad)
        point[unpad_key] = self._maybe_pin(unpad)
        point[cu_key] = self._maybe_pin(cu_seqlens)
        return point[pad_key], point[unpad_key], point[cu_key]

    def _resolve_out_dtype(self, feat: torch.Tensor):
        if self.out_dtype == "auto":
            return feat.dtype
        if self.out_dtype == "fp32":
            return torch.float32
        if self.out_dtype == "bf16":
            return torch.bfloat16
        if self.out_dtype == "fp16":
            return torch.float16
        return feat.dtype

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(offset2bincount(point.offset).min().tolist(), self.patch_size_max)

        H = self.num_heads
        K = self.patch_size
        C = self.channels
        device = point.feat.device
        N = point.feat.shape[0]

        # ---- pad/unpad：CPU 缓存 ----
        pad_cpu, unpad_cpu, _ = self.get_padding_and_inverse_cpu(point)

        # ---- serialized_*：统一 CPU 做索引 ----
        so = point.serialized_order[self.order_index]
        si = point.serialized_inverse[self.order_index]
        so_cpu = so.detach().to("cpu") if so.device.type != "cpu" else so
        si_cpu = si.detach().to("cpu") if si.device.type != "cpu" else si

        # order_cpu: (N',)   inverse_cpu: (N,)
        order_cpu = so_cpu[pad_cpu]
        inverse_cpu = unpad_cpu[si_cpu]

        # ---- 为“直接写回 out_final”准备：按 inverse_cpu 排序，后面每个 chunk 用 searchsorted 取对应 orig_idx ----
        # inv_sorted_cpu: padded位置(0..N'-1) 排序；orig_sorted_cpu: 对应原始点 idx
        inv_sorted_cpu, orig_sorted_cpu = torch.sort(inverse_cpu)  # 都在 CPU
        inv_sorted_cpu = inv_sorted_cpu.contiguous()
        orig_sorted_cpu = orig_sorted_cpu.contiguous()

        # ---- out_final：直接就是 N×C（原始顺序） ----
        out_dtype = self._resolve_out_dtype(point.feat)
        out_final = point.feat.new_empty((N, C), dtype=out_dtype)

        check_mem(msg='[SerializedAttention] start attention')

        # patch 数（基于 padded 后的 order_cpu）
        n_patch = order_cpu.numel() // K
        cp = self.chunk_patches if self.chunk_patches > 0 else n_patch

        # 小函数：把当前 chunk 的 out_blk(长度 b*K) 写回 out_final(长度 N)
        def _write_back(out_blk: torch.Tensor, start_pos: int, end_pos: int):
            # 找出所有 inverse_cpu 落在 [start_pos, end_pos) 的原始点索引
            l = torch.searchsorted(inv_sorted_cpu, start_pos)
            r = torch.searchsorted(inv_sorted_cpu, end_pos)
            if l == r:
                return  # 这个 chunk 可能全是 padding duplicate，对最终 N 没贡献

            orig_idx_cpu = orig_sorted_cpu[l:r]                    # (m,) 原始点 idx
            pos_cpu = (inv_sorted_cpu[l:r] - start_pos).to(torch.long)  # (m,) chunk 内位置

            orig_idx = orig_idx_cpu.to(device, non_blocking=True).long()
            pos = pos_cpu.to(device, non_blocking=True).long()

            out_valid = out_blk.index_select(0, pos)

            # ✅ 修复你遇到的 dtype mismatch：写回前对齐 dtype
            if out_valid.dtype != out_final.dtype:
                out_valid = out_valid.to(out_final.dtype)
            if not out_valid.is_contiguous():
                out_valid = out_valid.contiguous()

            # NPU 上 index_copy_ 要求 dtype 一致（你之前的报错就是这个）
            out_final.index_copy_(0, orig_idx, out_valid)

        # ========== flash 路径：仍然 chunk（避免一次 qkv 太大）==========
        if self.enable_flash:
            # 注意：这里完全不建立 feat_out，也不做最后 index_select
            for ps in range(0, n_patch, cp):
                pe = min(ps + cp, n_patch)
                start_pos = ps * K
                end_pos = pe * K

                ord_blk_cpu = order_cpu[start_pos:end_pos]
                ord_blk = ord_blk_cpu.to(device, non_blocking=True)

                feat_blk = point.feat.index_select(0, ord_blk)  # (b*K, C)
                qkv = self.qkv(feat_blk)                        # (b*K, 3C)

                # npu_fusion_attention
                if self.attn_dtype == "fp16":
                    qkv2 = qkv.half().reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv2.unbind(0)
                    out = torch_npu.npu_fusion_attention(
                        q, k, v, head_num=H, scale=self.scale, input_layout="BSH"
                    )[0].reshape(-1, C).to(qkv.dtype)
                else:
                    qkv2 = qkv.bfloat16().reshape(-1, K, 3, C)
                    q, k, v = qkv2.unbind(2)
                    out = torch_npu.npu_fusion_attention(
                        q, k, v, head_num=H, scale=self.scale, input_layout="BSH"
                    )[0].reshape(-1, C).to(qkv.dtype)

                _write_back(out, start_pos, end_pos)

                del ord_blk, feat_blk, qkv, qkv2, q, k, v, out

        # ========== non-flash：chunk 内做 qkv/attn，并直接写回 ==========
        else:
            for ps in range(0, n_patch, cp):
                pe = min(ps + cp, n_patch)
                start_pos = ps * K
                end_pos = pe * K

                ord_blk_cpu = order_cpu[start_pos:end_pos]
                ord_blk = ord_blk_cpu.to(device, non_blocking=True)

                feat_blk = point.feat.index_select(0, ord_blk)  # (b*K, C)
                qkv_blk = self.qkv(feat_blk)                    # (b*K, 3C)

                b = pe - ps
                qkv_blk = qkv_blk.view(b, K, 3, H, C // H)
                q, k, v = qkv_blk.permute(2, 0, 3, 1, 4).unbind(dim=0)

                if self.upcast_attention:
                    q = q.float()
                    k = k.float()

                attn = (q * self.scale) @ k.transpose(-2, -1)  # (b, H, K, K)

                if self.upcast_softmax:
                    attn = attn.float()

                attn = self.softmax(attn)
                attn = self.attn_drop(attn).to(qkv_blk.dtype)

                out = (attn @ v).transpose(1, 2).reshape(-1, C)  # (b*K, C)

                _write_back(out, start_pos, end_pos)

                del ord_blk, feat_blk, qkv_blk, q, k, v, attn, out

        check_mem(msg='[SerializedAttention] end attention')

        # ---- proj：这里开始是 N×C，不再需要 index_select/inverse ----
        # 若 out_final 是 bf16/fp16，但 proj.weight 是 fp32，线性会自己处理/或你可显式转
        feat = self.proj(out_final.to(self.proj.weight.dtype))
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # check_mem(msg='[MLP] ptv3 start MLP')
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # check_mem(msg='[MLP] ptv3 end MLP')
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        attn_dtype='fp16',
        upcast_attention=False, # True, NOTE
        upcast_softmax=False, # True, NOTE
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            attn_dtype=attn_dtype,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        # point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

class BlockWithoutCPE(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        attn_dtype='fp16',
        upcast_attention=True,
        upcast_softmax=True,
        # ---- new knobs ----
        use_checkpoint=True,     # 训练省显存最有效（会变慢）
        inplace_residual=False,    # residual add 用 add_ 省临时张量
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.use_checkpoint = use_checkpoint
        self.inplace_residual = inplace_residual

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            attn_dtype=attn_dtype,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def _attn_path(self, point: Point):
        # 拆开写，减少中间对象引用
        if self.pre_norm:
            point = self.norm1(point)
        point = self.attn(point)
        point = self.drop_path(point)
        if not self.pre_norm:
            point = self.norm1(point)
        return point

    def _mlp_path(self, point: Point):
        if self.pre_norm:
            point = self.norm2(point)
        point = self.mlp(point)
        point = self.drop_path(point)
        if not self.pre_norm:
            point = self.norm2(point)
        return point

    def forward(self, point: Point):
        # check_mem('[BlockWithoutCPE] start forward')

        # -------- attention + residual --------
        shortcut = point.feat  # 只是引用，不额外占显存

        if self.use_checkpoint and self.training:
            # checkpoint 需要函数输入输出是 Tensor；这里用 feat 走 checkpoint
            # 注意：checkpoint 只能对 Tensor 生效，我们用闭包操作 point.feat
            def _fn(feat):
                point_local = point
                point_local.feat = feat
                out = self._attn_path(point_local).feat
                return out
            attn_out = torch.utils.checkpoint.checkpoint(_fn, point.feat)
            point.feat = attn_out
        else:
            point = self._attn_path(point)

        if self.inplace_residual and (not self.training or not point.feat.requires_grad):
            # 推理/无梯度：inplace 最省
            point.feat.add_(shortcut)
        else:
            # 训练更稳妥（避免 autograd 版本冲突）
            point.feat = point.feat + shortcut

        # -------- mlp + residual --------
        shortcut = point.feat

        if self.use_checkpoint and self.training:
            def _fn2(feat):
                point_local = point
                point_local.feat = feat
                out = self._mlp_path(point_local).feat
                return out
            mlp_out = torch.utils.checkpoint.checkpoint(_fn2, point.feat)
            point.feat = mlp_out
        else:
            point = self._mlp_path(point)

        if self.inplace_residual and (not self.training or not point.feat.requires_grad):
            point.feat.add_(shortcut)
        else:
            point.feat = point.feat + shortcut

        # check_mem('[BlockWithoutCPE] end forward')
        return point

def _slim_parent_for_unpool(parent: "Point", feat_dtype=torch.bfloat16):
    """
    关键：encoder 阶段把 skip 的 parent.feat 从 NPU 挪走（大头），
    同时保留 decoder 继续跑 attention 必要的 serialized_* / offset / batch。
    """
    parent = drop_runtime_cache_inplace(parent)

    keep = [
        # 必要的几何/批信息
        "coord", "grid_coord", "batch", "offset",
        # decoder attention 必需
        "serialized_order", "serialized_inverse", "serialized_depth",
        # 条件/上下文
        "condition", "context",
    ]
    slim = {}
    for k in keep:
        if k in parent:
            slim[k] = parent[k]

    # ⭐️ 大头：feat 直接 offload 到 CPU，并压到 fp16/bf16
    feat = parent.feat
    if isinstance(feat, torch.Tensor):
        # CPU 上用 fp16 足够（也可以换 bfloat16）
        slim["feat"] = feat.detach().to("cpu", dtype=feat_dtype, non_blocking=False)
    else:
        slim["feat"] = feat

    # parent 不再需要 serialized_code（decoder 不用它），可以丢掉减少 CPU/RAM
    # （如果你后面确实不会再用 parent.serialized_code）
    # slim.pop("serialized_code", None)

    return Point(slim)
 
class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        else:
            self.norm = None
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0

        assert {"serialized_code", "serialized_order", "serialized_inverse", "serialized_depth"}.issubset(point.keys()), \
            "Run point.serialization() before SerializedPooling"

        device = point.feat.device

        # ✅ 全程 CPU 做聚类（避免 NPU 上 K×N long 峰值）
        code_cpu = point.serialized_code
        if code_cpu.device.type != "cpu":
            code_cpu = code_cpu.detach().to("cpu")

        code_cpu = code_cpu >> (pooling_depth * 3)      # CPU 位移
        code0_cpu = code_cpu[0]                         # CPU

        code0_u, cluster_cpu, counts_cpu = torch.unique(
            code0_cpu, sorted=True, return_inverse=True, return_counts=True
        )
        indices_cpu = torch.argsort(cluster_cpu)
        idx_ptr_cpu = torch.cat([counts_cpu.new_zeros(1), torch.cumsum(counts_cpu, dim=0)])
        head_indices_cpu = indices_cpu[idx_ptr_cpu[:-1]]

        # 下面真正做 feature/coord pooling 时再把 indices/idx_ptr/head_indices 搬到 NPU
        indices = indices_cpu.to(device, non_blocking=True)
        idx_ptr = idx_ptr_cpu.to(device, non_blocking=True)
        head_indices = head_indices_cpu.to(device, non_blocking=True)

        # downsample code/order/inverse 可以继续留 CPU（省显存），需要时再搬
        code_down_cpu = code_cpu[:, head_indices_cpu]   # CPU (K,M)
        order_cpu = torch.argsort(code_down_cpu)        # CPU (K,M)
        src_cpu = torch.arange(code_down_cpu.shape[1], dtype=order_cpu.dtype).expand(order_cpu.shape[0], -1)
        inverse_cpu = torch.zeros_like(order_cpu).scatter_(1, order_cpu, src_cpu)

        # 如果后面 attention 要用 serialized_order/inverse，建议也保留 CPU（与你整体策略一致）
        code_down = code_down_cpu
        order = order_cpu
        inverse = inverse_cpu
 
        if self.shuffle_orders:
            perm = torch.randperm(code_down.shape[0], device=order.device)
            code_down = code_down[perm]
            order = order[perm]
            inverse = inverse[perm]

        # ---- 聚合特征/坐标（这块无法避免，但尽量少临时）----
        feat_proj = self.proj(point.feat)                  # (N, out)
        feat_sorted = feat_proj.index_select(0, indices)   # (N, out) 只是重排
        feat_pooled = torch_scatter.segment_csr(feat_sorted, idx_ptr, reduce=self.reduce)

        coord_sorted = point.coord.index_select(0, indices)
        coord_pooled = torch_scatter.segment_csr(coord_sorted, idx_ptr, reduce="mean")

        # grid/batch 只取 head
        grid_pooled = point.grid_coord.index_select(0, head_indices) >> pooling_depth
        batch_pooled = point.batch.index_select(0, head_indices)

        # ---- 组装新 point ----
        point_dict = Dict(
            feat=feat_pooled,
            coord=coord_pooled,
            grid_coord=grid_pooled,
            serialized_code=code_down,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=batch_pooled,
        )
        if "condition" in point:
            point_dict["condition"] = point.condition
        if "context" in point:
            point_dict["context"] = point.context

        # ---- traceable：把“大对象”转 CPU/瘦身，避免显存被钉死 ----
        if self.traceable:
            # cluster 是 N 大索引：放 CPU，且可用 int32 省一半
            point_dict["pooling_inverse"] = _to_cpu_idx(cluster, dtype=torch.int32, pin=False)
            # parent 瘦身，避免把 serialized_* / pad/unpad / rel_pos / sparse_conv_feat 等挂住
            point_dict["pooling_parent"] = _slim_parent_for_unpool(point, feat_dtype=torch.bfloat16)

        new_point = Point(point_dict)

        if self.norm is not None:
            new_point = self.norm(new_point)
        if self.act is not None:
            new_point = self.act(new_point)

        new_point.sparsify()
        return new_point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()

        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")

        # inverse 我们在 pooling 里放 CPU int32 了，这里用前搬回 device 并转 long
        device = point.feat.device if isinstance(point, Point) else parent.feat.device
        if isinstance(inverse, torch.Tensor):
            if inverse.device.type == "cpu":
                inverse = inverse.to(device, non_blocking=True)
            if inverse.dtype != torch.long:
                inverse = inverse.long()

        point = self.proj(point)
        # ---- 把 parent.feat 从 CPU 拉回 NPU（只在 unpool 这一刻）----
        if isinstance(parent.feat, torch.Tensor) and parent.feat.device.type == "cpu":
            parent.feat = parent.feat.to(device, dtype=point.feat.dtype, non_blocking=True)        
        parent = self.proj_skip(parent)

        # parent.feat = parent.feat + point.feat[inverse]  会产生额外临时
        gathered = point.feat.index_select(0, inverse)  # (N_parent, C)

        if (not self.training) or (not parent.feat.requires_grad):
            parent.feat.add_(gathered)   # 推理/无梯度：in-place 最省
        else:
            parent.feat = parent.feat + gathered

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent

    

class Embedding(PointModule):
    def __init__(self, hidden_dim=192, dim=128, out_dim=512, act_layer=None, chunk_points=2048):
        super().__init__()
        assert hidden_dim % 6 == 0
        self.embedding_dim = hidden_dim
        self.dim = dim
        self.out_dim = out_dim
        self.chunk_points = int(chunk_points)

        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e, torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer("basis", e, persistent=False)

        # 直接用 Linear 保存权重
        self.linear = nn.Linear(self.embedding_dim + dim, out_dim, bias=True)
        self.act = act_layer() if act_layer is not None else None

    def _embed_chunk(self, pos: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        proj = pos @ basis                      # (n, E)
        return torch.cat([proj.sin(), proj.cos()], dim=1)  # (n, 2E)=hidden_dim

    def forward(self, point: Point):
        pos = point.coord
        feat = point.feat
        device = feat.device

        basis = self.basis.to(device=device, dtype=pos.dtype)

        # 你要求的 bf16
        pos_proj = pos.to(torch.bfloat16) if (pos.is_cuda or (hasattr(pos, "is_npu") and pos.is_npu)) else pos
        basis_proj = basis.to(torch.bfloat16) if (pos.is_cuda or (hasattr(pos, "is_npu") and pos.is_npu)) else basis

        W = self.linear.weight   # (out_dim, hidden_dim+dim)
        b = self.linear.bias
        W_emb = W[:, :self.embedding_dim]
        W_feat = W[:, self.embedding_dim:]

        N = feat.shape[0]
        cp = self.chunk_points

        out = feat.new_empty((N, self.out_dim))
        if cp and N > cp:
            for s in range(0, N, cp):
                e = min(s + cp, N)
                emb = self._embed_chunk(pos_proj[s:e], basis_proj)      # bf16
                # ⭐️ 不 cat：拆成两次 linear 相加
                y = F.linear(emb.to(feat.dtype), W_emb, b) + F.linear(feat[s:e], W_feat, None)
                if self.act is not None:
                    y = self.act(y)
                out[s:e].copy_(y)
                del emb, y
        else:
            emb = self._embed_chunk(pos_proj, basis_proj)
            y = F.linear(emb.to(feat.dtype), W_emb, b) + F.linear(feat, W_feat, None)
            if self.act is not None:
                y = self.act(y)
            out.copy_(y)
            del emb, y

        point.feat = out
        return point
    

def drop_runtime_cache_inplace(p: Point):
    for k in list(p.keys()):
        if k.startswith(("pad_k", "unpad_k", "cu_seqlens_k", "rel_pos_")):
            p.pop(k, None)
    return p

class PointTransformerV3_2(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        use_checkpoint=True,
        shuffle_orders=True,
        enable_cpe_block=False,
        enable_rpe=False,
        enable_flash=True,
        attn_dtype='fp16',
        enable_ca_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.enable_cpe_block = enable_cpe_block
        self.use_checkpoint = use_checkpoint
        assert self.enable_cpe_block is False, "npu cannot use CPE BLOCK, pls set enable_cpe_block as False"
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1
        # assert self.num_stages == len(dec_depths) + 1
        # assert self.num_stages == len(dec_channels) + 1
        # assert self.num_stages == len(dec_num_head) + 1
        # assert self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(dim=in_channels, out_dim=enc_channels[0], act_layer=None) # first channel number of enc_channel

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=None,#ln_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            # if self.enable_cpe_block:
            #     for i in range(enc_depths[s]):
            #         enc.add(
            #             Block(
            #                 channels=enc_channels[s],
            #                 num_heads=enc_num_head[s],
            #                 patch_size=enc_patch_size[s],
            #                 mlp_ratio=mlp_ratio,
            #                 qkv_bias=qkv_bias,
            #                 qk_scale=qk_scale,
            #                 attn_drop=attn_drop,
            #                 proj_drop=proj_drop,
            #                 drop_path=enc_drop_path_[i],
            #                 norm_layer=ln_layer,
            #                 act_layer=act_layer,
            #                 pre_norm=pre_norm,
            #                 order_index=i % len(self.order),
            #                 cpe_indice_key=f"stage{s}",
            #                 enable_rpe=enable_rpe,
            #                 enable_flash=enable_flash,
            #                 attn_dtype=attn_dtype,
            #                 upcast_attention=upcast_attention,
            #                 upcast_softmax=upcast_softmax,
            #             ),
            #             name=f"block{i}",
            #         )
            # else:
            for i in range(enc_depths[s]):
                enc.add(
                    BlockWithoutCPE(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        attn_dtype=attn_dtype,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        use_checkpoint=self.use_checkpoint, 
                        inplace_residual=False
                    ),
                    name=f"block{i}",
                )                
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=None,#ln_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                
                if self.enable_cpe_block:
                    for i in range(dec_depths[s]):
                        dec.add(
                            Block(
                                channels=dec_channels[s],
                                num_heads=dec_num_head[s],
                                patch_size=dec_patch_size[s],
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                attn_drop=attn_drop,
                                proj_drop=proj_drop,
                                drop_path=enc_drop_path_[i],
                                norm_layer=ln_layer,
                                act_layer=act_layer,
                                pre_norm=pre_norm,
                                order_index=i % len(self.order),
                                cpe_indice_key=f"stage{s}",
                                enable_rpe=enable_rpe,
                                enable_flash=enable_flash,
                                attn_dtype=attn_dtype,
                                upcast_attention=upcast_attention,
                                upcast_softmax=upcast_softmax,
                            ),
                            name=f"block{i}",
                        )
                else:
                    for i in range(dec_depths[s]):
                        dec.add(
                            BlockWithoutCPE(
                                channels=dec_channels[s],
                                num_heads=dec_num_head[s],
                                patch_size=dec_patch_size[s],
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                attn_drop=attn_drop,
                                proj_drop=proj_drop,
                                drop_path=enc_drop_path_[i],
                                norm_layer=ln_layer,
                                act_layer=act_layer,
                                pre_norm=pre_norm,
                                order_index=i % len(self.order),
                                cpe_indice_key=f"stage{s}",
                                enable_rpe=enable_rpe,
                                enable_flash=enable_flash,
                                attn_dtype=attn_dtype,
                                upcast_attention=upcast_attention,
                                upcast_softmax=upcast_softmax,
                                use_checkpoint=self.use_checkpoint, 
                                inplace_residual=False
                            ),
                            name=f"block{i}",
                        )  
                if len(dec) != 0:
                    self.dec.add(module=dec, name=f"dec{s}")

    def _forward(self, data_dict):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        # check_mem(msg='[PointTransformerV3_2] start ptv3 transformer')
        point = Point(data_dict)
        # check_mem(msg='[PointTransformerV3_2] build Point')
        point.serialization(
            order=self.order,
            shuffle_orders=self.shuffle_orders,
            store_code=True,                 # ✅ 最狠：不存 serialized_code（如果后面不需要）
            offload=("code", "order", "inverse"),    # code/order/inverse 全放 CPU
            pin_memory=False                  # NPU 通常没必要 pin
        )
        # check_mem(msg='[PointTransformerV3_2] After serialization')
        point.sparsify()
        # check_mem(msg='[PointTransformerV3_2] After sparsify')
        # 如果后面不用 spconv，立刻释放常驻
        point.pop("sparse_conv_feat", None)
        point.pop("sparse_shape", None)        

        point = self.embedding(point) # dict_keys(['coord', 'feat', 'offset', 'grid_size', 'batch', 'grid_coord', 'serialized_depth', 'serialized_code', 'serialized_order', 'serialized_inverse', 'sparse_shape', 'sparse_conv_feat'])
        # check_mem(msg='[PointTransformerV3_2] After embedding')
        point = self.enc(point)
        # check_mem(msg='[PointTransformerV3_2] After encoding')
        if not self.cls_mode:
            point = self.dec(point)
            # check_mem(msg='[PointTransformerV3_2] After decoding')
            
        # 末尾再清一次运行时缓存（可选）
        drop_runtime_cache(point)
        return point
    
    def forward(self, data_dict):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, data_dict, use_reentrant=False)
        else:
            return self._forward(data_dict)

def drop_runtime_cache(point: Point):
    # attention padding cache
    for k in list(point.keys()):
        if k.startswith("pad_k") or k.startswith("unpad_k") or k.startswith("cu_seqlens_k"):
            point.pop(k, None)
        if k.startswith("rel_pos_"):
            point.pop(k, None)
    return point


# from ...registry import MODELS
# @MODELS.register_module()
# class PointTransformerV3_2(PointModule):
#     def __init__(
#         self,
#         in_channels=6,
#         order=("z", "z-trans", "hilbert", "hilbert-trans"),
#         stride=(2, 2, 2, 2),
#         enc_depths=(2, 2, 2, 6, 2),
#         enc_channels=(32, 64, 128, 256, 512),
#         enc_num_head=(2, 4, 8, 16, 32),
#         enc_patch_size=(1024, 1024, 1024, 1024, 1024),
#         dec_depths=(2, 2, 2, 2),
#         dec_channels=(64, 64, 128, 256),
#         dec_num_head=(4, 4, 8, 16),
#         dec_patch_size=(1024, 1024, 1024, 1024),
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         drop_path=0.3,
#         pre_norm=True,
#         shuffle_orders=True,
#         enable_cpe_block=True,
#         enable_rpe=False,
#         enable_flash=True,
#         attn_dtype='fp16',
#         enable_ca_flash=False,
#         upcast_attention=False,
#         upcast_softmax=False,
#         cls_mode=False,
#         pdnorm_bn=False,
#         pdnorm_ln=False,
#         pdnorm_decouple=True,
#         pdnorm_adaptive=False,
#         pdnorm_affine=True,
#         pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
#     ):
#         super().__init__()
#         self.num_stages = len(enc_depths)
#         self.order = [order] if isinstance(order, str) else order
#         self.cls_mode = cls_mode
#         self.shuffle_orders = shuffle_orders
#         self.enable_cpe_block = enable_cpe_block

#         assert self.num_stages == len(stride) + 1
#         assert self.num_stages == len(enc_depths)
#         assert self.num_stages == len(enc_channels)
#         assert self.num_stages == len(enc_num_head)
#         assert self.num_stages == len(enc_patch_size)
#         assert self.cls_mode or self.num_stages == len(dec_depths) + 1
#         assert self.cls_mode or self.num_stages == len(dec_channels) + 1
#         assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
#         assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1
#         # assert self.num_stages == len(dec_depths) + 1
#         # assert self.num_stages == len(dec_channels) + 1
#         # assert self.num_stages == len(dec_num_head) + 1
#         # assert self.num_stages == len(dec_patch_size) + 1

#         # norm layers
#         if pdnorm_bn:
#             bn_layer = partial(
#                 PDNorm,
#                 norm_layer=partial(
#                     nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
#                 ),
#                 conditions=pdnorm_conditions,
#                 decouple=pdnorm_decouple,
#                 adaptive=pdnorm_adaptive,
#             )
#         else:
#             bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
#         if pdnorm_ln:
#             ln_layer = partial(
#                 PDNorm,
#                 norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
#                 conditions=pdnorm_conditions,
#                 decouple=pdnorm_decouple,
#                 adaptive=pdnorm_adaptive,
#             )
#         else:
#             ln_layer = nn.LayerNorm
#         # activation layers
#         act_layer = nn.GELU

#         self.embedding = Embedding(dim=in_channels, out_dim=enc_channels[0], act_layer=None) # first channel number of enc_channel

#         # encoder
#         enc_drop_path = [
#             x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
#         ]
#         self.enc = PointSequential()
#         for s in range(self.num_stages):
#             enc_drop_path_ = enc_drop_path[
#                 sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
#             ]
#             enc = PointSequential()
#             if s > 0:
#                 enc.add(
#                     SerializedPooling(
#                         in_channels=enc_channels[s - 1],
#                         out_channels=enc_channels[s],
#                         stride=stride[s - 1],
#                         norm_layer=None,#ln_layer,
#                         act_layer=act_layer,
#                     ),
#                     name="down",
#                 )
#             if self.enable_cpe_block:
#                 for i in range(enc_depths[s]):
#                     enc.add(
#                         Block(
#                             channels=enc_channels[s],
#                             num_heads=enc_num_head[s],
#                             patch_size=enc_patch_size[s],
#                             mlp_ratio=mlp_ratio,
#                             qkv_bias=qkv_bias,
#                             qk_scale=qk_scale,
#                             attn_drop=attn_drop,
#                             proj_drop=proj_drop,
#                             drop_path=enc_drop_path_[i],
#                             norm_layer=ln_layer,
#                             act_layer=act_layer,
#                             pre_norm=pre_norm,
#                             order_index=i % len(self.order),
#                             cpe_indice_key=f"stage{s}",
#                             enable_rpe=enable_rpe,
#                             enable_flash=enable_flash,
#                             attn_dtype=attn_dtype,
#                             upcast_attention=upcast_attention,
#                             upcast_softmax=upcast_softmax,
#                         ),
#                         name=f"block{i}",
#                     )
#             else:
#                 for i in range(enc_depths[s]):
#                     enc.add(
#                         BlockWithoutCPE(
#                             channels=enc_channels[s],
#                             num_heads=enc_num_head[s],
#                             patch_size=enc_patch_size[s],
#                             mlp_ratio=mlp_ratio,
#                             qkv_bias=qkv_bias,
#                             qk_scale=qk_scale,
#                             attn_drop=attn_drop,
#                             proj_drop=proj_drop,
#                             drop_path=enc_drop_path_[i],
#                             norm_layer=ln_layer,
#                             act_layer=act_layer,
#                             pre_norm=pre_norm,
#                             order_index=i % len(self.order),
#                             cpe_indice_key=f"stage{s}",
#                             enable_rpe=enable_rpe,
#                             enable_flash=enable_flash,
#                             attn_dtype=attn_dtype,
#                             upcast_attention=upcast_attention,
#                             upcast_softmax=upcast_softmax,
#                         ),
#                         name=f"block{i}",
#                     )                
#             if len(enc) != 0:
#                 self.enc.add(module=enc, name=f"enc{s}")

#         # decoder
#         if not self.cls_mode:
#             dec_drop_path = [
#                 x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
#             ]
#             self.dec = PointSequential()
#             dec_channels = list(dec_channels) + [enc_channels[-1]]
#             for s in reversed(range(self.num_stages - 1)):
#                 dec_drop_path_ = dec_drop_path[
#                     sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
#                 ]
#                 dec_drop_path_.reverse()
#                 dec = PointSequential()
#                 dec.add(
#                     SerializedUnpooling(
#                         in_channels=dec_channels[s + 1],
#                         skip_channels=enc_channels[s],
#                         out_channels=dec_channels[s],
#                         norm_layer=None,#ln_layer,
#                         act_layer=act_layer,
#                     ),
#                     name="up",
#                 )
                
#                 if self.enable_cpe_block:
#                     for i in range(dec_depths[s]):
#                         dec.add(
#                             Block(
#                                 channels=dec_channels[s],
#                                 num_heads=dec_num_head[s],
#                                 patch_size=dec_patch_size[s],
#                                 mlp_ratio=mlp_ratio,
#                                 qkv_bias=qkv_bias,
#                                 qk_scale=qk_scale,
#                                 attn_drop=attn_drop,
#                                 proj_drop=proj_drop,
#                                 drop_path=enc_drop_path_[i],
#                                 norm_layer=ln_layer,
#                                 act_layer=act_layer,
#                                 pre_norm=pre_norm,
#                                 order_index=i % len(self.order),
#                                 cpe_indice_key=f"stage{s}",
#                                 enable_rpe=enable_rpe,
#                                 enable_flash=enable_flash,
#                                 attn_dtype=attn_dtype,
#                                 upcast_attention=upcast_attention,
#                                 upcast_softmax=upcast_softmax,
#                             ),
#                             name=f"block{i}",
#                         )
#                 else:
#                     for i in range(dec_depths[s]):
#                         dec.add(
#                             BlockWithoutCPE(
#                                 channels=dec_channels[s],
#                                 num_heads=dec_num_head[s],
#                                 patch_size=dec_patch_size[s],
#                                 mlp_ratio=mlp_ratio,
#                                 qkv_bias=qkv_bias,
#                                 qk_scale=qk_scale,
#                                 attn_drop=attn_drop,
#                                 proj_drop=proj_drop,
#                                 drop_path=enc_drop_path_[i],
#                                 norm_layer=ln_layer,
#                                 act_layer=act_layer,
#                                 pre_norm=pre_norm,
#                                 order_index=i % len(self.order),
#                                 cpe_indice_key=f"stage{s}",
#                                 enable_rpe=enable_rpe,
#                                 enable_flash=enable_flash,
#                                 attn_dtype=attn_dtype,
#                                 upcast_attention=upcast_attention,
#                                 upcast_softmax=upcast_softmax,
#                             ),
#                             name=f"block{i}",
#                         )  
#                 if len(dec) != 0:
#                     self.dec.add(module=dec, name=f"dec{s}")

#     def forward(self, data_dict):
#         """
#         A data_dict is a dictionary containing properties of a batched point cloud.
#         It should contain the following properties for PTv3:
#         1. "feat": feature of point cloud
#         2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
#         3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
#         """
#         point = Point(data_dict)
#         point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
#         point.sparsify()

#         point = self.embedding(point) # dict_keys(['coord', 'feat', 'offset', 'grid_size', 'batch', 'grid_coord', 'serialized_depth', 'serialized_code', 'serialized_order', 'serialized_inverse', 'sparse_shape', 'sparse_conv_feat'])
#         point = self.enc(point)
#         if not self.cls_mode:
#             point = self.dec(point)
#         return point
