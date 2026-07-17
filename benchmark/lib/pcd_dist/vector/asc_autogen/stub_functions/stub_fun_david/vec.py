from ...utils import Tensor, GMTensor, Var, Instruction, Position, DATATYPE
from ... import context
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


def ub_to_ub(dst: Tensor, src: Tensor, n_burst: Union[int, Var], burst_len: Union[int, Var], src_stride: Union[int, Var], dst_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call ub_to_ub within vector forward function'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert isinstance(dst, Tensor), 'dst must be Tensor on UB'
    assert dst.pos==Position.UB, 'Tensor position must be on UB'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('UB2UB', dst=dst, src=src, n_burst=n_burst, burst_len=burst_len, src_stride=src_stride, dst_stride=dst_stride))


def ub_to_l1(dst: Tensor, src: Tensor, n_burst: Union[int, Var], burst_len: Union[int, Var], src_stride: Union[int, Var], dst_stride: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call ub_to_l1 within vector forward function'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert isinstance(dst, Tensor), 'dst must be Tensor on L1'
    assert dst.pos==Position.L1, 'Tensor position must be on L1'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('UB2L1', dst=dst, src=src, n_burst=n_burst, burst_len=burst_len, src_stride=src_stride, dst_stride=dst_stride))


def ub_to_l1_nz(dst: Tensor, src: Tensor, m_dst: Union[int, Var], n_dst: Union[int, Var], m_src: Union[int, Var], n_src: Union[int, Var]):
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call ub_to_l1_nz within vector forward function'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert isinstance(dst, Tensor), 'dst must be Tensor on L1'
    assert dst.pos==Position.L1, 'Tensor position must be on L1'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('UB2L1_NZ', dst=dst, src=src, m_dst=m_dst, n_dst=n_dst, m_src=m_src, n_src=n_src))

def ub_to_l1_nd2nz(dst: Tensor, src: Tensor, m_dst: Union[int, Var], n_dst: Union[int, Var], m_src: Union[int, Var], n_src: Union[int, Var]):
    print('WARNING: MTE2/MTE3 instruction queue size is 16. If n_src/C0_size is larger than this number, the whole pipeline will become sequential (stuck on Scalar).')
    print('!!!! Make sure you can handle this. Otherwise use ub_to_ub_nd2nz and ub_to_l1_nz !!!!')
    g_vec = context.active_vec
    assert g_vec is not None, 'You must call ub_to_l1_nz within vector forward function'
    assert isinstance(src, Tensor), 'src must be Tensor on UB'
    assert isinstance(dst, Tensor), 'dst must be Tensor on L1'
    assert dst.pos==Position.L1, 'Tensor position must be on L1'
    assert src.pos==Position.UB, 'Tensor position must be on UB'

    g_vec.append(Instruction('UB2L1_ND2NZ', dst=dst, src=src, m_dst=m_dst, n_dst=n_dst, m_src=m_src, n_src=n_src))


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


def sort32(dst: Tensor, src: Tensor, idx: Tensor, repeat: Union[Var, int]):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert isinstance(idx, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert src.dtype == dst.dtype
    assert dst.dtype in [DATATYPE.float, DATATYPE.half]
    assert idx.dtype==DATATYPE.uint32

    g_vec.append(Instruction('SORT32', dst=dst, src=src, idx=idx, repeat=repeat))


def mergesort4(dst: Tensor, src: Tensor):
    g_vec = context.active_vec
    assert g_vec is not None 
    assert isinstance(dst, Tensor)
    assert isinstance(src, Tensor)
    assert src.pos==Position.UB, 'src must be Tensor on UB'
    assert dst.pos==Position.UB, 'dst must be Tensor on UB'
    assert src.dtype == dst.dtype
    assert dst.dtype in [DATATYPE.float, DATATYPE.half]

    g_vec.append(Instruction('MERGESORT4', dst=dst, src=src))
