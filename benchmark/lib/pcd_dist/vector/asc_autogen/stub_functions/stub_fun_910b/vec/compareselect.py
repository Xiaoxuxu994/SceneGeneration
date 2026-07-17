from ....utils import Tensor, GMTensor, Var, Instruction, Position, DT, CompareMode, SelectMode, SELMODE
from .... import context
from typing import Union, Optional


def compare(dst: Tensor, src1: Tensor, src2: Tensor, repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src1_blk_stride: Union[int, Var], src2_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src1_rep_stride: Union[int, Var], src2_rep_stride: Union[int, Var], mode: CompareMode):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src1, Tensor)
    assert isinstance(src2, Tensor)
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert src1.pos==Position.UB, 'src1 must be Tensor on UB'
    assert src2.pos==Position.UB, 'src2 must be Tensor on UB'
    assert src1.dtype==src2.dtype

    assert dst.dtype in [DT.int8, DT.uint8]
    assert src1.dtype in [DT.float, DT.half, DT.int16, DT.int32]
    
    g_vec.append(Instruction('COMPARE', dst=dst, src1=src1, src2=src2, mode=mode, repeat=repeat, dst_blk_stride=dst_blk_stride, src1_blk_stride=src1_blk_stride, src2_blk_stride=src2_blk_stride, dst_rep_stride=dst_rep_stride, src1_rep_stride=src1_rep_stride, src2_rep_stride=src2_rep_stride))


def compare_scalar(dst: Tensor, src1: Tensor, src2: Union[int, float, Var], repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src1_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src1_rep_stride: Union[int, Var], mode: CompareMode):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src1, Tensor)
    assert src1.pos==Position.UB, 'src1 must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'

    assert dst.dtype in [DT.int8, DT.uint8]
    assert src1.dtype in [DT.float, DT.half, DT.int16, DT.int32]
    
    g_vec.append(Instruction('COMPARES', dst=dst, src1=src1, src2=src2, mode=mode, repeat=repeat, dst_blk_stride=dst_blk_stride, src1_blk_stride=src1_blk_stride, dst_rep_stride=dst_rep_stride, src1_rep_stride=src1_rep_stride))



def compare_to_reg(src1: Tensor, src2: Union[Tensor, int, float, Var], dst_blk_stride: Union[int, Var], src1_blk_stride: Union[int, Var], src2_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src1_rep_stride: Union[int, Var], src2_rep_stride: Union[int, Var], mode: CompareMode):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(src1, Tensor)
    assert isinstance(src2, Tensor)
    assert src1.pos==Position.UB, 'src1 must be Tensor on UB'
    assert src2.pos==Position.UB, 'src2 must be Tensor on UB'

    assert src1.dtype==src1.dtype
    assert src1.dtype in [DT.float, DT.half, DT.int16, DT.int32]
    g_vec.append(Instruction('COMPARETOREG', src1=src1, src2=src2, mode=mode, dst_blk_stride=dst_blk_stride, src1_blk_stride=src1_blk_stride, src2_blk_stride=src2_blk_stride, dst_rep_stride=dst_rep_stride, src1_rep_stride=src1_rep_stride, src2_rep_stride=src2_rep_stride))
    

def set_cmpmask(src: Tensor):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src1 must be Tensor on UB'

    g_vec.append(Instruction('SETCMPMASK', src=src))


def select(dst: Tensor, selmask: Union[Tensor, Var], src1: Tensor, src2: Union[Tensor, Var, int, float], repeat: Union[int, Var], dst_blk_stride: Union[int, Var], src1_blk_stride: Union[int, Var], src2_blk_stride: Union[int, Var], dst_rep_stride: Union[int, Var], src1_rep_stride: Union[int, Var], src2_rep_stride: Union[int, Var], mode: SelectMode):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src1, Tensor)
    assert src1.pos==Position.UB, 'src1 must be Tensor on UB'
    
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'

    assert dst.dtype==src1.dtype
    assert selmask.dtype in [DT.uint8, DT.uint16, DT.uint32, DT.uint64]
    assert dst.dtype in [DT.float, DT.half, DT.int16, DT.int32]

    if mode is SELMODE.TENSOR_SCALAR:
        assert isinstance(src2, (float, int, Var))
    else:
        assert isinstance(src2, Tensor)
        assert src2.pos==Position.UB, 'src2 must be Tensor on UB'
        assert dst.dtype==src2.dtype

    g_vec.append(Instruction('SELECT', dst=dst, mode=mode, selmask=selmask, src1=src1, src2=src2, repeat=repeat, dst_blk_stride=dst_blk_stride, src1_blk_stride=src1_blk_stride, src2_blk_stride=src2_blk_stride, dst_rep_stride=dst_rep_stride, src1_rep_stride=src1_rep_stride, src2_rep_stride=src2_rep_stride))
