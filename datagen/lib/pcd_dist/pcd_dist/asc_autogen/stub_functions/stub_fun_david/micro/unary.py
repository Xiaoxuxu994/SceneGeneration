from .... import context
from ....utils import Reg, Tensor, Var, MaskReg, Instruction, Position, sizeof
from typing import Union 


# single src 
def exp(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert dst.dtype==src.dtype
    assert dst.dtype==mask.dtype

    g_micro.append(Instruction('VEXP', dst=dst, src=src, mask=mask))

def abs(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert dst.dtype==src.dtype
    assert dst.dtype==mask.dtype

    g_micro.append(Instruction('VABS', dst=dst, src=src, mask=mask))

def relu(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert dst.dtype==src.dtype
    assert dst.dtype==mask.dtype

    g_micro.append(Instruction('VRELU', dst=dst, src=src, mask=mask))

def sqrt(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert dst.dtype==src.dtype
    assert dst.dtype==mask.dtype

    g_micro.append(Instruction('VSQRT', dst=dst, src=src, mask=mask))

def ln(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert dst.dtype==src.dtype
    assert dst.dtype==mask.dtype

    g_micro.append(Instruction('VLN', dst=dst, src=src, mask=mask))

def log(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert dst.dtype==src.dtype
    assert dst.dtype==mask.dtype

    g_micro.append(Instruction('VLOG', dst=dst, src=src, mask=mask))

def log2(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert dst.dtype==src.dtype
    assert dst.dtype==mask.dtype

    g_micro.append(Instruction('VLOG2', dst=dst, src=src, mask=mask))

def log10(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert dst.dtype==src.dtype
    assert dst.dtype==mask.dtype

    g_micro.append(Instruction('VLOG10', dst=dst, src=src, mask=mask))

def neg(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert dst.dtype==src.dtype
    assert dst.dtype==mask.dtype

    g_micro.append(Instruction('VNEG', dst=dst, src=src, mask=mask))

def vnot(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert dst.dtype==src.dtype
    assert dst.dtype==mask.dtype

    g_micro.append(Instruction('VNOT', dst=dst, src=src, mask=mask))

def vcopy(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert dst.dtype==src.dtype
    assert dst.dtype==mask.dtype

    g_micro.append(Instruction('VCOPY', dst=dst, src=src, mask=mask))
# end: single src 
