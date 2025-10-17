from .... import context
from ....utils import Reg, Tensor, Var, MaskReg, Instruction, Position, sizeof, LOADDIST, LoadDistInst, DT, STOREDIST, StoreDistInst
from typing import Union 


# micro register related data move
def ub_to_reg(dst: Reg, src: Tensor, blk_stride: Union[int, Var], mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src, Tensor)
    assert src.pos == Position.UB, f'SRC POS: {src.pos}'
    assert isinstance(dst, Reg)
    assert dst.dtype == src.dtype
    
    g_micro.append(Instruction('UB2REG', dst=dst, src=src, blk_stride=blk_stride, mask=mask))


def reg_to_ub(dst: Tensor, src: Reg, blk_stride: Union[int, Var], mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Tensor)
    assert dst.pos == Position.UB, f'DST POS: {dst.pos}'
    assert isinstance(src, Reg)
    assert dst.dtype == src.dtype
    
    g_micro.append(Instruction('REG2UB', dst=dst, src=src, blk_stride=blk_stride, mask=mask))
    

def ub_to_reg_continuous(dst: Reg, src: Tensor, loaddist: LoadDistInst):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(src, Tensor)
    assert src.pos == Position.UB, f'SRC POS: {src.pos}'
    assert isinstance(dst, Reg)
    assert isinstance(loaddist, LoadDistInst)
    assert dst.dtype == src.dtype

    if loaddist==LOADDIST.DOWNSAMPLE:
        if sizeof(dst.dtype)==1:
            mode = 'DIST_DS_B8'
        elif sizeof(dst.dtype)==2:
            mode = 'DIST_DS_B16'
        else:
            raise ValueError('Not support 32bit types')
    elif loaddist==LOADDIST.UPSAMPLE:
        if sizeof(dst.dtype)==1:
            mode = 'DIST_US_B8'
        elif sizeof(dst.dtype)==2:
            mode = 'DIST_US_B16'
        else:
            raise ValueError('Not support 32bit types')
    elif loaddist==LOADDIST.SINGLE_VALUE:
        if sizeof(dst.dtype)==1:
            mode = 'DIST_BRC_B8'
        elif sizeof(dst.dtype)==2:
            mode = 'DIST_BRC_B16'
        elif sizeof(dst.dtype)==4:
            mode = 'DIST_BRC_B32'
        else:
            raise ValueError(f'{dst.dtype} is not supported')
    elif loaddist==LOADDIST.BRCB:
        if sizeof(dst.dtype)==2:
            mode = 'DIST_E2B_B16'
        elif sizeof(dst.dtype)==4:
            mode = 'DIST_E2B_B32'
        else:
            raise ValueError(f'{dst.dtype} is not supported')
    elif loaddist==LOADDIST.UNPACK:
        if sizeof(dst.dtype)==1:
            mode = 'DIST_UNPACK_B8'
        elif sizeof(dst.dtype)==2:
            mode = 'DIST_UNPACK_B16'
        elif sizeof(dst.dtype)==4:
            mode = 'DIST_UNPACK_B32'
    elif loaddist==LOADDIST.UNPACK4:
        if sizeof(dst.dtype)==1:
            mode = 'DIST_UNPACK4_B8'
        else:
            raise ValueError('UNPACK4 only supports 8-bit datatypes')
        
    g_micro.append(Instruction('UB2REGCONT', dst=dst, src=src, mode=mode))


def reg_to_ub_continuous(dst: Tensor, src: Reg, mask: MaskReg, storedist: StoreDistInst):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Tensor)
    assert dst.pos == Position.UB, f'SRC POS: {dst.pos}'
    assert isinstance(src, Reg)
    assert isinstance(mask, MaskReg)
    assert isinstance(storedist, StoreDistInst)
    assert dst.dtype == src.dtype

    if storedist==STOREDIST.DOWNSAMPLE:
        if sizeof(dst.dtype)==1:
            mode = 'DIST_PACK_B16'
        elif sizeof(dst.dtype)==2:
            mode = 'DIST_PACK_B32'
        elif sizeof(dst.dtype)==4:
            mode = 'DIST_PACK_B64'
        else:
            raise ValueError('Not support 64bit types')
    elif storedist==STOREDIST.SINGLE_VALUE:
        if sizeof(dst.dtype)==1:
            mode = 'DIST_FIRST_ELEMENT_B8'
        elif sizeof(dst.dtype)==2:
            mode = 'DIST_FIRST_ELEMENT_B16'
        elif sizeof(dst.dtype)==4:
            mode = 'DIST_FIRST_ELEMENT_B32'
        else:
            raise ValueError(f'{dst.dtype} is not supported')
    elif storedist==STOREDIST.NORMAL:
        if sizeof(dst.dtype)==1:
            mode = 'DIST_NORM_B8'
        elif sizeof(dst.dtype)==2:
            mode = 'DIST_NORM_B16'
        elif sizeof(dst.dtype)==4:
            mode = 'DIST_NORM_B32'
        else:
            raise ValueError(f'{dst.dtype} is not supported')
        
    g_micro.append(Instruction('REG2UBCONT', dst=dst, src=src, mask=mask, mode=mode))

# end: micro register related data move 

# Gather and scatter 
def ub_to_reg_gather(dst: Reg, src: Tensor, index: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Tensor)
    assert isinstance(index, Reg)
    assert isinstance(mask, MaskReg)
    assert src.dtype==dst.dtype

    if sizeof(src.dtype)==2:
        assert index.dtype==DT.uint16
    elif sizeof(src.dtype)==4:
        assert index.dtype==DT.uint32

    g_micro.append(Instruction('DATACOPYGATHER', dst=dst, src=src, index=index, mask=mask))


def reg_to_ub_scatter(dst: Tensor, src: Reg, index: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Reg)
    assert isinstance(index, Reg)
    assert isinstance(mask, MaskReg)

    assert src.dtype==dst.dtype
    
    if sizeof(src.dtype)==2:
        assert index.dtype==DT.uint16
    elif sizeof(src.dtype)==4:
        assert index.dtype==DT.uint32

    g_micro.append(Instruction('DATACOPYSCATTER', dst=dst, src=src, index=index, mask=mask))


def gather(dst: Reg, src: Reg, index: Reg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert isinstance(index, Reg)

    assert src.dtype==dst.dtype
    
    if sizeof(src.dtype)==2:
        assert index.dtype==DT.uint16
    elif sizeof(src.dtype)==4:
        assert index.dtype==DT.uint32

    g_micro.append(Instruction('GATHER', dst=dst, src=src, index=index))
    

def gather_mask(dst: Reg, src: Reg, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None 
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert isinstance(mask, MaskReg)

    g_micro.append(Instruction('GATHERMASK', dst=dst, src=src, mask=mask))
# end: gather and scatter
