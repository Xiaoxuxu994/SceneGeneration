from ....utils import Tensor, GMTensor, Var, Instruction, Position, DT
from .... import context
from typing import Union, Optional


def exp(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('EXP', dst=dst, src=src, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def ln(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('LN', dst=dst, src=src, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def abs(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('ABS', dst=dst, src=src, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def rec(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('REC', dst=dst, src=src, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def sqrt(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('SQRT', dst=dst, src=src, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def rsqrt(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('RSQRT', dst=dst, src=src, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def vnot(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.int16, DT.uint16]

    g_vec.append(Instruction('NOT', dst=dst, src=src, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def relu(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('RELU', dst=dst, src=src, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))

