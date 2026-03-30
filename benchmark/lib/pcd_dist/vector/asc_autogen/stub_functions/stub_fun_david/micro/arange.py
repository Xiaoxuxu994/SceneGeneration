from .... import context
from ....utils import Reg, Tensor, Var, MaskReg, Instruction, Position, sizeof
from typing import Union 


# Arange 
def arange(dst: Reg, start: Union[Var, int, float], increase: bool=True):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    dtype = dst.dtype
    mode = 'MicroAPI::IndexOrder::INCREASE_ORDER' if increase else 'MicroAPI::IndexOrder::DECREASE_ORDER'
    
    if isinstance(start, Var):
        v = str(start)
    elif dst.dtype.ctype=='float':
        v = str(float(start)) + 'f'
    elif dst.dtype.ctype=='half':
        v = '(half)' + str(float(start)) + 'f'
    elif dst.dtype.ctype=='bfloat16_t':
        v = '(bfloat16_t)' + str(float(start)) + 'f'
    else:
        v = '%d'%start

    g_micro.append(Instruction('ARANGE', dst=dst, v=v, dtype=dtype, mode=mode))
# end: arange

