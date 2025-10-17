from ....utils import Tensor, GMTensor, Var, Instruction, Position, DATATYPE
from .... import context
from typing import Union, Optional


# Data move
def gm_to_ub(dst: Tensor, src: GMTensor, n_burst: Union[int, Var], burst_len: Union[int, Var], src_stride: Union[int, Var], dst_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call gm_to_ub within vector forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, GMTensor), 'src must be GMTensor on GM'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('GM2UB', dst=dst, src=src, n_burst=n_burst, burst_len=burst_len, src_stride=src_stride, dst_stride=dst_stride))


def ub_to_gm(dst: GMTensor, src: Tensor, n_burst: Union[int, Var], burst_len: Union[int, Var], src_stride: Union[int, Var], dst_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call ub_to_gm within vector forward function'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert isinstance(dst, GMTensor), 'dst must be GMTensor on GM'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('UB2GM', dst=dst, src=src, n_burst=n_burst, burst_len=burst_len, src_stride=src_stride, dst_stride=dst_stride))

## --------------------------------------------------------------------------
# Data move Pad
def gm_to_ub_pad(dst: Tensor, src: GMTensor, n_burst: Union[int, Var], burst_len: Union[int, Var], src_stride: Union[int, Var], dst_stride: Union[int, Var],
                 isPad: Union[int, bool], leftPadding: Union[int, Var], rightPadding: Union[int,Var], paddingValue: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call gm_to_ub within vector forward function'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert isinstance(src, GMTensor), 'src must be GMTensor on GM'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('GM2UBPad', dst=dst, src=src, n_burst=n_burst, burst_len=burst_len, src_stride=src_stride, dst_stride=dst_stride,
                             isPad = isPad, leftPadding=leftPadding, rightPadding=rightPadding, paddingValue=paddingValue))


def ub_to_gm_pad(dst: GMTensor, src: Tensor, n_burst: Union[int, Var], burst_len: Union[int, Var], src_stride: Union[int, Var], dst_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call ub_to_gm within vector forward function'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert isinstance(dst, GMTensor), 'dst must be GMTensor on GM'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('UB2GMPad', dst=dst, src=src, n_burst=n_burst, burst_len=burst_len, src_stride=src_stride, dst_stride=dst_stride))

## --------------------------------------------------------------------------

def ub_to_ub(dst: Tensor, src: Tensor, n_burst: Union[int, Var], burst_len: Union[int, Var], src_stride: Union[int, Var], dst_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call ub_to_ub within vector forward function'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('UB2UB', dst=dst, src=src, n_burst=n_burst, burst_len=burst_len, src_stride=src_stride, dst_stride=dst_stride))


def ub_to_ub_nd2nz(dst: Tensor, src: Tensor, m_dst: Union[int, Var], n_dst: Union[int, Var], m_src: Union[int, Var], n_src: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call ub_to_ub within vector forward function'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('UB2UB_ND2NZ', dst=dst, src=src, m_dst=m_dst, n_dst=n_dst, m_src=m_src, n_src=n_src))


def ub_to_ub_nd2nz_compact(dst: Tensor, src: Tensor, m: Union[int, Var], n: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call ub_to_ub within vector forward function'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('UB2UB_ND2NZ_CPCT', dst=dst, src=src, m=m, n=n))
