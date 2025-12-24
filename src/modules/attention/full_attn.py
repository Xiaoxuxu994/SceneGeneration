from typing import *
import torch
import torch_npu
import math
from torch.nn.functional import scaled_dot_product_attention as sdpa

__all__ = [
    'scaled_dot_product_attention',
]


def _naive_sdpa(q, k, v, return_scores=False):
    """
    Naive implementation of scaled dot product attention.
    """
    q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
    k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
    v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
    scale_factor = 1 / math.sqrt(q.size(-1))
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    out = attn_weight @ v
    out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
    if return_scores:
        return out, attn_weight
    return out


@overload
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 * , return_scores: bool = True,  **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply scaled dot product attention, output out and attn_weight.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] tensor containing Vs.
        return_scores (bool): whether to output attn_weight                

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

def scaled_dot_product_attention(*args, return_scores: bool = False, **kwargs):
    
    BACKEND = kwargs.get('BACKEND', 'flash_attn_npu') # default=flash_attn_npu
    
    q = args[0] if len(args) > 0 else kwargs['q']
    k = args[1] if len(args) > 1 else kwargs['k']
    v = args[2] if len(args) > 2 else kwargs['v']
    assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
    assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
    assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
    assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
    device = q.device    
    
    # 如果要 scores，则强制 naive
    if BACKEND == 'flash_attn_npu':
        # if num_all_args == 1:
        #     out = flash_attn.flash_attn_qkvpacked_func(qkv)
        # elif num_all_args == 2:
        #     out = flash_attn.flash_attn_kvpacked_func(q, kv)
        # elif num_all_args == 3:
        q = q.reshape(q.shape[0], q.shape[1], -1)
        k = k.reshape(k.shape[0], k.shape[1], -1)
        v = v.reshape(v.shape[0], v.shape[1], -1)
        out = torch_npu.npu_fusion_attention(q, k, v, head_num=kwargs.get('head_num'), scale=kwargs.get('scale'), input_layout='BSH')[0]
        return out

    elif BACKEND == 'nsa_npu': 
        seqlen = q.shape[1]
        out = torch_npu.npu_nsa_compress_attention(q.squeeze(0), k.squeeze(0), v.squeeze(0), 
                                                   scale_value=kwargs.get('scale'), head_num=kwargs.get('head_num'), compress_block_size=kwargs.get('compress_block_size', 32), 
                                                   compress_stride=kwargs.get('compress_stride', 16), select_block_size=kwargs.get('select_block_size', 64), select_block_count=kwargs.get('select_block_count', 16), 
                                                   actual_seq_qlen=kwargs.get('compress_stride', [seqlen]), actual_cmp_seq_kvlen=kwargs.get('actual_cmp_seq_kvlen', [14000]), actual_sel_seq_kvlen=kwargs.get('actual_cmp_seq_kvlen', [14000]),
                                                   topk_mask=None, atten_mask=None)[0]
        return out

    elif BACKEND == 'nsa_npu_sample': 
        seqlen = q.shape[1]
        indices = kwargs.get('kv_index')
        k, v = k[:,indices], v[:,indices]
        kvlen = k.shape[1]
        out = torch_npu.npu_nsa_compress_attention(q.squeeze(0), k.squeeze(0), v.squeeze(0), 
                                                   scale_value=kwargs.get('scale'), head_num=kwargs.get('head_num'), compress_block_size=kwargs.get('compress_block_size', 32), 
                                                   compress_stride=kwargs.get('compress_stride', 16), select_block_size=kwargs.get('select_block_size', 64), select_block_count=kwargs.get('select_block_count', 16), 
                                                   actual_seq_qlen=kwargs.get('compress_stride', [seqlen]), actual_cmp_seq_kvlen=kwargs.get('actual_cmp_seq_kvlen', [kvlen]), actual_sel_seq_kvlen=kwargs.get('actual_cmp_seq_kvlen', [kvlen]),
                                                   topk_mask=None, atten_mask=None)[0]
        return out

        
    elif BACKEND == 'sdpa':
        q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
        k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
        v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
        out = sdpa(q, k, v)         # [N, H, L, C]
        out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
        return out

    elif BACKEND == 'naive':
        out = _naive_sdpa(q, k, v)
        return out

    else:
        raise ValueError(f"Unknown attention module: {BACKEND}")
    
    