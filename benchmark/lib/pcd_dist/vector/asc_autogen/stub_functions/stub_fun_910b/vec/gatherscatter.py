from ....utils import Tensor, GMTensor, Var, Instruction, Position, DT
from .... import context
from typing import Union, Optional


def gather(dst: Tensor, src: Tensor, offset: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert isinstance(offset, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert offset.dtype==DT.uint32

    g_vec.append(Instruction('GATHER', dst=dst, src=src, offset=offset, repeat=repeat, dst_rep_stride=dst_rep_stride))


def scatter(dst: Tensor, src: Tensor, offset: Tensor, repeat: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert isinstance(offset, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert dst.dtype==src.dtype
    assert offset.dtype==DT.uint32

    g_vec.append(Instruction('SCATTER', dst=dst, src=src, offset=offset, repeat=repeat, src_rep_stride=src_rep_stride))
