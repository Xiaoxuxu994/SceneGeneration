from .... import context
from ....utils import Reg, Tensor, Var, MaskReg, Instruction, Position, sizeof
from typing import Union 


# data filling
def dup(dst: Reg, src: Union[Reg, float, int, Var], mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, (Reg, Var, int, float))
    assert isinstance(mask, MaskReg)

    if isinstance(src, (Reg, Var)):
        v = str(src)
    elif dst.dtype.ctype=='float':
        v = str(float(src)) + 'f'
    elif dst.dtype.ctype=='half':
        v = '(half)' + str(float(src)) + 'f'
    elif dst.dtype.ctype=='bfloat16_t':
        v = '(bfloat16_t)' + str(float(src)) + 'f'
    else:
        v = '%d'%src

    g_micro.append(Instruction('VDUP', dst=dst, src=v, mask=mask))
# end: data filling 

    
