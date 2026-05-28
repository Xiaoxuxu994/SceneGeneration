from .... import context
from ....utils import Reg, Tensor, Var, MaskReg, Instruction, Position, sizeof, CompareMode
from typing import Union 


# compare and select 
def compare(dst: MaskReg, src1: Reg, src2: Union[Var, int, float, Reg], mask: MaskReg, mode: CompareMode):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, MaskReg)
    assert isinstance(src1, Reg)
    assert isinstance(mask, MaskReg)
    dtype = dst.dtype

    if isinstance(src2, Reg):
        assert src1.dtype==src2.dtype
        g_micro.append(Instruction('COMPARE', dst=dst, src1=src1, src2=src2, dtype=dtype, mode=mode, mask=mask))
    else:
        if isinstance(src2, Var):
            v = str(src2)
        elif dst.dtype=='float':
            v = str(float(src2)) + 'f'
        elif dst.dtype=='half':
            v = '(half)' + str(float(src2)) + 'f'
        elif dst.dtype=='bfloat16_t':
            v = '(bfloat16_t)' + str(float(src2)) + 'f'
        else:
            v = '%d'%src2
        g_micro.append(Instruction('COMPARES', dst=dst, src1=src1, src2=src2, dtype=v, mode=mode, mask=mask))


def select(dst: Reg, src1: Reg, src2: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src1, Reg)
    assert isinstance(src2, Reg)
    assert isinstance(mask, MaskReg)
    assert src1.dtype==src2.dtype

    g_micro.append(Instruction('SELECT', dst=dst, src1=src1, src2=src2, mask=mask))
# end: compare and select 

