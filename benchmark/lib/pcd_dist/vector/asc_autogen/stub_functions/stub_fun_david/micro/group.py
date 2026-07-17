from .... import context
from ....utils import Reg, Tensor, Var, MaskReg, Instruction, Position, sizeof
from typing import Union 


# group functions 
def cmax(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    g_micro.append(Instruction('VCMAX', dst=dst, src=src, mask=mask))
    

def cgmax(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    g_micro.append(Instruction('VCGMAX', dst=dst, src=src, mask=mask))


def cmin(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    g_micro.append(Instruction('VCMIN', dst=dst, src=src, mask=mask))
    

def cgmin(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    g_micro.append(Instruction('VCGMIN', dst=dst, src=src, mask=mask))
    

def cadd(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    g_micro.append(Instruction('VCADD', dst=dst, src=src, mask=mask))
    

def cgadd(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    g_micro.append(Instruction('VCGADD', dst=dst, src=src, mask=mask))


def cpadd(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype == src.dtype

    g_micro.append(Instruction('VCPADD', dst=dst, src=src, mask=mask))
# end: group functions 