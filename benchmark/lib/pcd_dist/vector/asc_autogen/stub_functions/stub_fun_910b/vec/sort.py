from ....utils import Tensor, GMTensor, Var, Instruction, Position, DATATYPE
from .... import context
from typing import Union, Optional


def sort32(dst: Tensor, src: Tensor, idx: Tensor, repeat: Union[Var, int]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert isinstance(idx, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert src.dtype == dst.dtype
    assert dst.dtype in [DATATYPE.float, DATATYPE.half]
    assert idx.dtype==DATATYPE.uint32

    g_vec.append(Instruction('SORT32', dst=dst, src=src, idx=idx, repeat=repeat))

