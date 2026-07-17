from ....utils import Tensor, GMTensor, Var, Instruction, Position, DT
from .... import context
from typing import Union, Optional


def dup(dst: Tensor, value: Union[float, int, Var], repeat: Union[int, Var], dst_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    supported_dtype = [DT.float, DT.half, DT.int16, DT.uint16, DT.int, DT.int32, DT.uint32, DT.bfloat16]
    assert dst.dtype in supported_dtype, f'Not supported type {dst.dtype}. Supported: {supported_dtype}'

    g_vec.append(Instruction('DUP', dst=dst, src=value, repeat=repeat, dst_blk_stride=dst_blk_stride, dst_rep_stride=dst_rep_stride))


def brcb(dst: Tensor, src: Tensor, dst_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], repeat: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('BRCB', dst=dst, src=src, dst_blk_stride=dst_blk_stride, dst_rep_stride=dst_rep_stride, repeat=repeat))

