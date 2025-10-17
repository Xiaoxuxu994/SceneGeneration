from ....utils import Tensor, GMTensor, Var, Instruction, Position, DT
from .... import context
from typing import Union, Optional


def adds(dst: Tensor, src: Tensor, val: Union[int, float, Var], repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert isinstance(val, (int, float, Var))
    assert dst.dtype in [DT.float, DT.half, DT.int16, DT.int32]

    g_vec.append(Instruction('ADDS', dst=dst, src=src, val=val, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def muls(dst: Tensor, src: Tensor, val: Union[int, float, Var], repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert isinstance(val, (int, float, Var))
    assert dst.dtype in [DT.float, DT.half, DT.int16, DT.int32]

    g_vec.append(Instruction('MULS', dst=dst, src=src, val=val, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def vmaxs(dst: Tensor, src: Tensor, val: Union[int, float, Var], repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert isinstance(val, (int, float, Var))
    assert dst.dtype in [DT.float, DT.half, DT.int16, DT.int32]

    g_vec.append(Instruction('MAXS', dst=dst, src=src, val=val, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def vmins(dst: Tensor, src: Tensor, val: Union[int, float, Var], repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert isinstance(val, (int, float, Var))
    assert dst.dtype in [DT.float, DT.half, DT.int16, DT.int32]

    g_vec.append(Instruction('MINS', dst=dst, src=src, val=val, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def lrelu(dst: Tensor, src: Tensor, val: Union[int, float, Var], repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert isinstance(val, (int, float, Var))
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('LRELU', dst=dst, src=src, val=val, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))


def axpy(dst: Tensor, src: Tensor, val: Union[int, float, Var], repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert isinstance(val, (int, float, Var))
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('AXPY', dst=dst, src=src, val=val, repeat=repeat, dst_blk_stride=dst_blk_stride, src_blk_stride=src_blk_stride, dst_rep_stride=dst_rep_stride, src_rep_stride=src_rep_stride))
