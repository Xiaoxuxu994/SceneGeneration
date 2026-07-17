from ....utils import Tensor, GMTensor, Var, Instruction, Position, DT
from .... import context
from typing import Union, Optional


def add(dst: Tensor, src1: Tensor, src2: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src1_blk_stride: Union[int, Var], src2_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src1_rep_stride: Union[int, Var], src2_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src1, Tensor)
    assert isinstance(src2, Tensor)
    assert src1.pos==Position.UB, 'src1 must be Tensor on UB'
    assert src2.pos==Position.UB, 'src2 must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'

    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    assert dst.dtype in [DT.float, DT.half, DT.int16, DT.int32]

    g_vec.append(Instruction('ADD', dst=dst, src1=src1, src2=src2, repeat=repeat, dst_blk_stride=dst_blk_stride, src1_blk_stride=src1_blk_stride, src2_blk_stride=src2_blk_stride, dst_rep_stride=dst_rep_stride, src1_rep_stride=src1_rep_stride, src2_rep_stride=src2_rep_stride))


def sub(dst: Tensor, src1: Tensor, src2: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src1_blk_stride: Union[int, Var], src2_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src1_rep_stride: Union[int, Var], src2_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src1, Tensor)
    assert isinstance(src2, Tensor)
    assert src1.pos==Position.UB, 'src1 must be Tensor on UB'
    assert src2.pos==Position.UB, 'src2 must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'

    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    assert dst.dtype in [DT.float, DT.half, DT.int16, DT.int32]

    g_vec.append(Instruction('SUB', dst=dst, src1=src1, src2=src2, repeat=repeat, dst_blk_stride=dst_blk_stride, src1_blk_stride=src1_blk_stride, src2_blk_stride=src2_blk_stride, dst_rep_stride=dst_rep_stride, src1_rep_stride=src1_rep_stride, src2_rep_stride=src2_rep_stride))


def mul(dst: Tensor, src1: Tensor, src2: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src1_blk_stride: Union[int, Var], src2_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src1_rep_stride: Union[int, Var], src2_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src1, Tensor)
    assert isinstance(src2, Tensor)
    assert src1.pos==Position.UB, 'src1 must be Tensor on UB'
    assert src2.pos==Position.UB, 'src2 must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'

    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    assert dst.dtype in [DT.float, DT.half, DT.int16, DT.int32]

    g_vec.append(Instruction('MUL', dst=dst, src1=src1, src2=src2, repeat=repeat, dst_blk_stride=dst_blk_stride, src1_blk_stride=src1_blk_stride, src2_blk_stride=src2_blk_stride, dst_rep_stride=dst_rep_stride, src1_rep_stride=src1_rep_stride, src2_rep_stride=src2_rep_stride))


def div(dst: Tensor, src1: Tensor, src2: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src1_blk_stride: Union[int, Var], src2_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src1_rep_stride: Union[int, Var], src2_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src1, Tensor)
    assert isinstance(src2, Tensor)
    assert src1.pos==Position.UB, 'src1 must be Tensor on UB'
    assert src2.pos==Position.UB, 'src2 must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'

    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    assert dst.dtype in [DT.float, DT.half]

    g_vec.append(Instruction('DIV', dst=dst, src1=src1, src2=src2, repeat=repeat, dst_blk_stride=dst_blk_stride, src1_blk_stride=src1_blk_stride, src2_blk_stride=src2_blk_stride, dst_rep_stride=dst_rep_stride, src1_rep_stride=src1_rep_stride, src2_rep_stride=src2_rep_stride))


def vmax(dst: Tensor, src1: Tensor, src2: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src1_blk_stride: Union[int, Var], src2_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src1_rep_stride: Union[int, Var], src2_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src1, Tensor)
    assert isinstance(src2, Tensor)
    assert src1.pos==Position.UB, 'src1 must be Tensor on UB'
    assert src2.pos==Position.UB, 'src2 must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'

    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    assert dst.dtype in [DT.float, DT.half, DT.int16, DT.int32]

    g_vec.append(Instruction('MAX', dst=dst, src1=src1, src2=src2, repeat=repeat, dst_blk_stride=dst_blk_stride, src1_blk_stride=src1_blk_stride, src2_blk_stride=src2_blk_stride, dst_rep_stride=dst_rep_stride, src1_rep_stride=src1_rep_stride, src2_rep_stride=src2_rep_stride))


def vmin(dst: Tensor, src1: Tensor, src2: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src1_blk_stride: Union[int, Var], src2_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src1_rep_stride: Union[int, Var], src2_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src1, Tensor)
    assert isinstance(src2, Tensor)
    assert src1.pos==Position.UB, 'src1 must be Tensor on UB'
    assert src2.pos==Position.UB, 'src2 must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'

    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    assert dst.dtype in [DT.float, DT.half, DT.int16, DT.int32]

    g_vec.append(Instruction('MIN', dst=dst, src1=src1, src2=src2, repeat=repeat, dst_blk_stride=dst_blk_stride, src1_blk_stride=src1_blk_stride, src2_blk_stride=src2_blk_stride, dst_rep_stride=dst_rep_stride, src1_rep_stride=src1_rep_stride, src2_rep_stride=src2_rep_stride))


def vand(dst: Tensor, src1: Tensor, src2: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src1_blk_stride: Union[int, Var], src2_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src1_rep_stride: Union[int, Var], src2_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src1, Tensor)
    assert isinstance(src2, Tensor)
    assert src1.pos==Position.UB, 'src1 must be Tensor on UB'
    assert src2.pos==Position.UB, 'src2 must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'

    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    assert dst.dtype in [DT.int16, DT.uint16]

    g_vec.append(Instruction('AND', dst=dst, src1=src1, src2=src2, repeat=repeat, dst_blk_stride=dst_blk_stride, src1_blk_stride=src1_blk_stride, src2_blk_stride=src2_blk_stride, dst_rep_stride=dst_rep_stride, src1_rep_stride=src1_rep_stride, src2_rep_stride=src2_rep_stride))


def vor(dst: Tensor, src1: Tensor, src2: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src1_blk_stride: Union[int, Var], src2_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src1_rep_stride: Union[int, Var], src2_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src1, Tensor)
    assert isinstance(src2, Tensor)
    assert src1.pos==Position.UB, 'src1 must be Tensor on UB'
    assert src2.pos==Position.UB, 'src2 must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'

    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    assert dst.dtype in [DT.int16, DT.uint16]

    g_vec.append(Instruction('OR', dst=dst, src1=src1, src2=src2, repeat=repeat, dst_blk_stride=dst_blk_stride, src1_blk_stride=src1_blk_stride, src2_blk_stride=src2_blk_stride, dst_rep_stride=dst_rep_stride, src1_rep_stride=src1_rep_stride, src2_rep_stride=src2_rep_stride))
