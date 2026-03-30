from .... import context
from ....utils import Reg, Tensor, Var, MaskReg, Instruction, Position, sizeof, DATATYPE
from typing import Union 


# Mask operations 
def mask_not(dst: MaskReg, src: MaskReg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, MaskReg)
    assert isinstance(src, MaskReg)
    assert isinstance(mask, MaskReg)

    g_micro.append(Instruction('MASKNOT', dst=dst, src=src, mask=mask)) 
    

def mask_and(dst: MaskReg, src1: MaskReg, src2: MaskReg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, MaskReg)
    assert isinstance(src1, MaskReg)
    assert isinstance(src2, MaskReg)
    assert isinstance(mask, MaskReg)

    g_micro.append(Instruction('MASKAND', dst=dst, src1=src1, src2=src2, mask=mask)) 
    

def mask_or(dst: MaskReg, src1: MaskReg, src2: MaskReg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, MaskReg)
    assert isinstance(src1, MaskReg)
    assert isinstance(src2, MaskReg)
    assert isinstance(mask, MaskReg)

    g_micro.append(Instruction('MASKOR', dst=dst, src1=src1, src2=src2, mask=mask)) 
    

def mask_xor(dst: MaskReg, src1: MaskReg, src2: MaskReg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, MaskReg)
    assert isinstance(src1, MaskReg)
    assert isinstance(src2, MaskReg)
    assert isinstance(mask, MaskReg)

    g_micro.append(Instruction('MASKXOR', dst=dst, src1=src1, src2=src2, mask=mask)) 
    

def mask_mov(dst: MaskReg, src: MaskReg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, MaskReg)
    assert isinstance(src, MaskReg)
    assert isinstance(mask, MaskReg)

    g_micro.append(Instruction('MASKMOV', dst=dst, src=src, mask=mask)) 
    

def mask_interleave(dst0: MaskReg, dst1: MaskReg, src0: MaskReg, src1: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst0, MaskReg)
    assert isinstance(dst1, MaskReg)
    assert isinstance(src0, MaskReg)
    assert isinstance(src1, MaskReg)

    g_micro.append(Instruction('MASKINTERL', dst0=dst0, dst1=dst1, src0=src0, src1=src1)) 
    

def mask_deinterleave(dst0: MaskReg, dst1: MaskReg, src0: MaskReg, src1: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst0, MaskReg)
    assert isinstance(dst1, MaskReg)
    assert isinstance(src0, MaskReg)
    assert isinstance(src1, MaskReg)

    g_micro.append(Instruction('MASKDEINTERL', dst0=dst0, dst1=dst1, src0=src0, src1=src1)) 
    

def mask_sel(dst: MaskReg, src1: MaskReg, src2: MaskReg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, MaskReg)
    assert isinstance(src1, MaskReg)
    assert isinstance(src2, MaskReg)
    assert isinstance(mask, MaskReg)

    g_micro.append(Instruction('MASKSEL', dst=dst, src1=src1, src2=src2, mask=mask)) 
    

def mask_pack(dst: MaskReg, src: MaskReg, low_part: bool=True):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, MaskReg)
    assert isinstance(src, MaskReg)

    if low_part:
        mode = 'MicroAPI::HighLowPart::LOWEST'
    else:
        mode = 'MicroAPI::HighLowPart::HIGHEST'

    g_micro.append(Instruction('MASKPACK', dst=dst, src=src, mode=mode)) 


def mask_unpack(dst: MaskReg, src: MaskReg, low_part: bool=True):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, MaskReg)
    assert isinstance(src, MaskReg)

    if low_part:
        mode = 'MicroAPI::HighLowPart::LOWEST'
    else:
        mode = 'MicroAPI::HighLowPart::HIGHEST'

    g_micro.append(Instruction('MASKUNPACK', dst=dst, src=src, mode=mode)) 


def move_mask_spr(dst: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, MaskReg)

    g_micro.append(Instruction('MOVEMASKSPR', dst=dst)) 


def update_mask(dst: MaskReg, cnt: Var):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, MaskReg)
    assert isinstance(cnt, Var), 'Count can only be Var'
    assert cnt._module==g_micro, 'counter var should be created in the same forward function'
    assert cnt.dtype==DATATYPE.uint32, 'update_mask\'s counter only accepts uint32'

    g_micro.append(Instruction('UPDATEMASK', dst=dst, cnt=cnt))
# end: mask ops 
