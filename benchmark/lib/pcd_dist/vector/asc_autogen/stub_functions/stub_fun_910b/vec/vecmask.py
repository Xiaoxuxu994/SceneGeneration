from ....utils import Tensor, GMTensor, Var, Instruction, Position, DT
from .... import context
from typing import Union, Optional


def set_mask(maskHigh: Union[int, Var], maskLow: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None 
    if isinstance(maskHigh, Var):
        assert maskHigh.dtype==DT.uint64
    if isinstance(maskLow, Var):
        assert maskLow.dtype==DT.uint64

    g_vec.append(Instruction('SETMASK', low=maskLow, high=maskHigh))
    

def reset_mask():
    g_vec = context.active_vec
    assert g_vec is not None 

    g_vec.append(Instruction('RESETMASK'))
