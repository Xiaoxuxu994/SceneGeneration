from ....utils import Tensor, GMTensor, Var, Instruction, Position, DT
from .... import context
from typing import Union, Optional
from rich.text import Text
from rich.style import Style
from rich.console import Console
from functools import partial
print = partial(Console(width=150).print, justify='left')


_vc_warning = True
_vcg_warning = True 
_vcp_warning = True 

def warn_vc():
    global _vc_warning
    if _vc_warning:
        print(Text('Usage reminder', style=Style(color='grey82', underline=True)).append(Text(': vcg opearators has destination stride in unit of 2bytes for fp16 and 4bytes for fp32', style=Style(color='grey50', underline=False))))
        _vc_warning = False

def warn_vcg():
    global _vcg_warning
    if _vcg_warning:
        print(Text('Usage reminder', style=Style(color='grey82', underline=True)).append(Text(': vcg opearators has destination stride in unit of 16bytes for fp16 and 32bytes for fp32', style=Style(color='grey50', underline=False))))
        _vcg_warning = False

def warn_vcp():
    global _vcp_warning
    if _vcp_warning:
        print(Text('Usage reminder', style=Style(color='grey82', underline=True)).append(Text(': vcp opearators has destination stride in unit of 128bytes', style=Style(color='grey50', underline=False))))
        _vcp_warning = False


# group functions 
def cmax(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var], mode: str = 'ORDER_ONLY_VALUE'):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    # assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert src.dtype in [DT.float, DT.half], '[CMAX] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCMAX', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride, mode=mode))
    warn_vc()
    

def cgmax(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.dtype in [DT.float, DT.half], '[CGMAX] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCGMAX', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride))
    warn_vcg()


def cmin(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.dtype in [DT.float, DT.half], '[CMIN] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCMIN', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride))
    warn_vc()
    

def cgmin(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.dtype in [DT.float, DT.half], '[CGMIN] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCGMIN', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride))
    warn_vcg()
    

def cadd(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.dtype in [DT.float, DT.half], '[CADD] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCADD', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride))
    warn_vc()
    

def cgadd(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.dtype in [DT.float, DT.half], '[CGADD] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCGADD', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride))
    warn_vcg()


def cpadd(dst: Tensor, src: Tensor, repeat: Union[int, Var], dst_rep_stride: Union[int, Var], src_blk_stride: Union[int, Var], src_rep_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'OP must be called within forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert dst.dtype==src.dtype, 'dst and src dtype should be the same'
    assert dst.dtype in [DT.float, DT.half], '[CGADD] Only half or float is supported'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('VCPADD', dst=dst, src=src, repeat=repeat, dst_rep_stride=dst_rep_stride, src_blk_stride=src_blk_stride, src_rep_stride=src_rep_stride))
    warn_vcp()
