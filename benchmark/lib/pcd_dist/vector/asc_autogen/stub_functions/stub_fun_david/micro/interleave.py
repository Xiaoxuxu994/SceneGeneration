from .... import context
from ....utils import Reg, Tensor, Var, MaskReg, Instruction, Position, sizeof
from typing import Union 


# interleaves 
def deinterleave(dst0: Reg, dst1: Reg, src0: Reg, src1: Reg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst0, Reg)
    assert isinstance(dst1, Reg)
    assert isinstance(src0, Reg)
    assert isinstance(src1, Reg)
    assert dst0.dtype == dst1.dtype
    assert src0.dtype == src1.dtype
    assert dst0.dtype == src0.dtype

    g_micro.append(Instruction('DINTERLEAVE', dst0=dst0, dst1=dst1, src0=src0, src1=src1))


def interleave(dst0: Reg, dst1: Reg, src0: Reg, src1: Reg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst0, Reg)
    assert isinstance(dst1, Reg)
    assert isinstance(src0, Reg)
    assert isinstance(src1, Reg)
    assert dst0.dtype == dst1.dtype
    assert src0.dtype == src1.dtype
    assert dst0.dtype == src0.dtype

    g_micro.append(Instruction('INTERLEAVE', dst0=dst0, dst1=dst1, src0=src0, src1=src1))
# end: interleaves 