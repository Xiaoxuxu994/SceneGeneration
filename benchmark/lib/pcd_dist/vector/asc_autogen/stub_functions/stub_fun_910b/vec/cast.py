from ....utils import Tensor, GMTensor, Var, Instruction, Position, DT, RoundModeInst
from .... import context
from typing import Union, Optional


def cast(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var], round_mode: RoundModeInst):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    
    g_vec.append(Instruction('CAST', dst=dst, src=src, mode=round_mode, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))
