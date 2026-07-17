from .... import context
from ....utils import Reg, Tensor, Var, MaskReg, Instruction, Position, sizeof
from typing import Union 


# src + scalar
def vmaxs(dst: Reg, src: Reg, value: Union[Var, int, float], mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    if isinstance(value, Var):
        v = str(value)
    elif src.dtype.ctype=='float':
        v = str(float(value)) + 'f'
    elif src.dtype.ctype=='half':
        v = '(half)' + str(float(value)) + 'f'
    elif src.dtype.ctype=='bfloat16_t':
        v = '(bfloat16_t)' + str(float(value)) + 'f'
    else:
        v = '%d'%value
    g_micro.append(Instruction('VMAXS', dst=dst, src=src, v=v, mask=mask))

def vmins(dst: Reg, src: Reg, value: Union[Var, int, float], mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    if isinstance(value, Var):
        v = str(value)
    elif src.dtype.ctype=='float':
        v = str(float(value)) + 'f'
    elif src.dtype.ctype=='half':
        v = '(half)' + str(float(value)) + 'f'
    elif src.dtype.ctype=='bfloat16_t':
        v = '(bfloat16_t)' + str(float(value)) + 'f'
    else:
        v = '%d'%value
    g_micro.append(Instruction('VMINS', dst=dst, src=src, v=v, mask=mask))

def adds(dst: Reg, src: Reg, value: Union[Var, int, float], mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    if isinstance(value, Var):
        v = str(value)
    elif src.dtype.ctype=='float':
        v = str(float(value)) + 'f'
    elif src.dtype.ctype=='half':
        v = '(half)' + str(float(value)) + 'f'
    elif src.dtype.ctype=='bfloat16_t':
        v = '(bfloat16_t)' + str(float(value)) + 'f'
    else:
        v = '%d'%value
    g_micro.append(Instruction('VADDS', dst=dst, src=src, v=v, mask=mask))

def muls(dst: Reg, src: Reg, value: Union[Var, int, float], mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    if isinstance(value, Var):
        v = str(value)
    elif src.dtype.ctype=='float':
        v = str(float(value)) + 'f'
    elif src.dtype.ctype=='half':
        v = '(half)' + str(float(value)) + 'f'
    elif src.dtype.ctype=='bfloat16_t':
        v = '(bfloat16_t)' + str(float(value)) + 'f'
    else:
        v = '%d'%value
    g_micro.append(Instruction('VMULS', dst=dst, src=src, v=v, mask=mask))

def lrelu(dst: Reg, src: Reg, value: Union[Var, int, float], mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    if isinstance(value, Var):
        v = str(value)
    elif src.dtype.ctype=='float':
        v = str(float(value)) + 'f'
    elif src.dtype.ctype=='half':
        v = '(half)' + str(float(value)) + 'f'
    elif src.dtype.ctype=='bfloat16_t':
        v = '(bfloat16_t)' + str(float(value)) + 'f'
    else:
        v = '%d'%value
    g_micro.append(Instruction('VLRELU', dst=dst, src=src, v=v, mask=mask))


def shiftls(dst: Reg, src: Reg, value: Union[Var, int, float], mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    if isinstance(value, Var):
        v = str(value)
    elif src.dtype.ctype=='float':
        v = str(float(value)) + 'f'
    elif src.dtype.ctype=='half':
        v = '(half)' + str(float(value)) + 'f'
    elif src.dtype.ctype=='bfloat16_t':
        v = '(bfloat16_t)' + str(float(value)) + 'f'
    else:
        v = '%d'%value
    g_micro.append(Instruction('SHIFTLS', dst=dst, src=src, v=v, mask=mask))


def shiftrs(dst: Reg, src: Reg, value: Union[Var, int, float], mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    if isinstance(value, Var):
        v = str(value)
    elif src.dtype.ctype=='float':
        v = str(float(value)) + 'f'
    elif src.dtype.ctype=='half':
        v = '(half)' + str(float(value)) + 'f'
    elif src.dtype.ctype=='bfloat16_t':
        v = '(bfloat16_t)' + str(float(value)) + 'f'
    else:
        v = '%d'%value
    g_micro.append(Instruction('SHIFTRS', dst=dst, src=src, v=v, mask=mask))

def axpy(dst: Reg, src: Reg, value: Union[Var, int, float], mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    if isinstance(value, Var):
        v = str(value)
    elif src.dtype.ctype=='float':
        v = str(float(value)) + 'f'
    elif src.dtype.ctype=='half':
        v = '(half)' + str(float(value)) + 'f'
    elif src.dtype.ctype=='bfloat16_t':
        v = '(bfloat16_t)' + str(float(value)) + 'f'
    else:
        v = '%d'%value
    g_micro.append(Instruction('VAXPY', dst=dst, src=src, v=v, mask=mask))

# end: src + scalar
