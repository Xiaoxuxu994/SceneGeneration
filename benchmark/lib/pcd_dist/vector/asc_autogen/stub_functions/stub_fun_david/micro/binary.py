from .... import context
from ....utils import Reg, Tensor, Var, MaskReg, Instruction, Position, sizeof
from typing import Union 


# dual src
def vmax(dst: Reg, src1: Reg, src2: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src1, Reg)
    assert isinstance(src2, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    
    g_micro.append(Instruction('VMAX', dst=dst, src1=src1, src2=src2, mask=mask))

def vmin(dst: Reg, src1: Reg, src2: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src1, Reg)
    assert isinstance(src2, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    
    g_micro.append(Instruction('VMIN', dst=dst, src1=src1, src2=src2, mask=mask))

def add(dst: Reg, src1: Reg, src2: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src1, Reg)
    assert isinstance(src2, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    
    g_micro.append(Instruction('VADD', dst=dst, src1=src1, src2=src2, mask=mask))

def sub(dst: Reg, src1: Reg, src2: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src1, Reg)
    assert isinstance(src2, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    
    g_micro.append(Instruction('VSUB', dst=dst, src1=src1, src2=src2, mask=mask))


def mul(dst: Reg, src1: Reg, src2: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src1, Reg)
    assert isinstance(src2, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    
    g_micro.append(Instruction('VMUL', dst=dst, src1=src1, src2=src2, mask=mask))

def div(dst: Reg, src1: Reg, src2: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src1, Reg)
    assert isinstance(src2, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    
    g_micro.append(Instruction('VDIV', dst=dst, src1=src1, src2=src2, mask=mask))

def vand(dst: Reg, src1: Reg, src2: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src1, Reg)
    assert isinstance(src2, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    
    g_micro.append(Instruction('VAND', dst=dst, src1=src1, src2=src2, mask=mask))

def vor(dst: Reg, src1: Reg, src2: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src1, Reg)
    assert isinstance(src2, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    
    g_micro.append(Instruction('VOR', dst=dst, src1=src1, src2=src2, mask=mask))

def vxor(dst: Reg, src1: Reg, src2: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src1, Reg)
    assert isinstance(src2, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    
    g_micro.append(Instruction('VXOR', dst=dst, src1=src1, src2=src2, mask=mask))

def prelu(dst: Reg, src1: Reg, src2: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src1, Reg)
    assert isinstance(src2, Reg)
    assert isinstance(dst, Reg)
    assert isinstance(mask, MaskReg)
    assert dst.dtype==src1.dtype
    assert dst.dtype==src2.dtype
    
    g_micro.append(Instruction('VXOR', dst=dst, src1=src1, src2=src2, mask=mask))
## end: dual src 
