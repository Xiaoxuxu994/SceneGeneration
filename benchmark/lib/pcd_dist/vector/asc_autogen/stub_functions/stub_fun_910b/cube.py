from ...utils import Tensor, GMTensor, Var, Instruction, Position, DualModeInst
from ... import context
from typing import Union, Optional


def gm_to_l1_nd2nz(dst: Tensor, src: GMTensor, m: Union[int, Var], n: Union[int ,Var], N: Union[int, Var], M: Optional[Union[int, Var]]=None):
    g_cube = context.active_cube
    assert g_cube is not None, 'You must call gm_to_l1_nd2nz within cube forward function'
    assert isinstance(src, GMTensor), 'src must be GMTensor'
    assert isinstance(dst, Tensor), 'dst must be Tensor'
    assert dst.pos==Position.L1, f'Tensor position must be on L1. Got {dst.pos}'

    if M is None:
        M = m 

    g_cube.append(Instruction('L1ND2NZ', dst=dst, src=src, m=m, n=n, N=N, M=M))


def l1_to_l0_nz2zn(dst: Tensor, src: Tensor, mdst: Union[int, Var], ndst: Union[int ,Var], msrc: Union[int ,Var], nsrc: Union[int, Var]):
    g_cube = context.active_cube
    assert g_cube is not None, 'You must call l1_to_l0_nz2zn within cube forward function'
    assert isinstance(src, Tensor), 'src must be Tensor'
    assert isinstance(dst, Tensor), 'dst must be Tensor'
    assert dst.pos in [Position.L0A, Position.L0B], f'DST Tensor position must be on L0A/L0B. Got {dst.pos}'
    assert src.pos==Position.L1, f'SRC Tensor position must be on L1. Got {src.pos}'

    g_cube.append(Instruction('L0NZ2ZN', dst=dst, src=src, mdst=mdst, ndst=ndst, msrc=msrc, nsrc=nsrc))


def l1_to_l0_nz2zz(dst: Tensor, src: Tensor, mdst: Union[int, Var], ndst: Union[int ,Var], msrc: Union[int ,Var], nsrc: Union[int, Var]):
    g_cube = context.active_cube
    assert g_cube is not None, 'You must call l1_to_l0_nz2zn within cube forward function'
    assert isinstance(src, Tensor), 'src must be Tensor'
    assert isinstance(dst, Tensor), 'dst must be Tensor'
    assert dst.pos in [Position.L0A, Position.L0B], f'DST Tensor position must be on L0A/L0B. Got {dst.pos}'
    assert src.pos==Position.L1, f'SRC Tensor position must be on L1. Got {src.pos}'

    g_cube.append(Instruction('L0NZ2ZZ', dst=dst, src=src, mdst=mdst, ndst=ndst, msrc=msrc, nsrc=nsrc))


def l1_to_l0_nz2nn(dst: Tensor, src: Tensor, mdst: Union[int, Var], ndst: Union[int ,Var], msrc: Union[int ,Var], nsrc: Union[int, Var]):
    g_cube = context.active_cube
    assert g_cube is not None, 'You must call l1_to_l0_nz2zn within cube forward function'
    assert isinstance(src, Tensor), 'src must be Tensor'
    assert isinstance(dst, Tensor), 'dst must be Tensor'
    assert dst.pos in [Position.L0A, Position.L0B], f'DST Tensor position must be on L0A/L0B. Got {dst.pos}'
    assert src.pos==Position.L1, f'SRC Tensor position must be on L1. Got {src.pos}'

    g_cube.append(Instruction('L0NZ2NN', dst=dst, src=src, mdst=mdst, ndst=ndst, msrc=msrc, nsrc=nsrc))


def l1_to_l0_nz2nz(dst: Tensor, src: Tensor, mdst: Union[int, Var], ndst: Union[int ,Var], msrc: Union[int ,Var], nsrc: Union[int, Var]):
    g_cube = context.active_cube
    assert g_cube is not None, 'You must call l1_to_l0_nz2nz within cube forward function'
    assert isinstance(src, Tensor), 'src must be Tensor'
    assert isinstance(dst, Tensor), 'dst must be Tensor'
    assert dst.pos in [Position.L0A, Position.L0B], f'DST Tensor position must be on L0A/L0B. Got {dst.pos}'
    assert src.pos==Position.L1, f'SRC Tensor position must be on L1. Got {src.pos}'

    g_cube.append(Instruction('L0NZ2NZ', dst=dst, src=src, mdst=mdst, ndst=ndst, msrc=msrc, nsrc=nsrc))


def l1_to_l0(dst: Tensor, src: Tensor, m: Union[int, Var], n: Union[int, Var]):
    g_cube = context.active_cube
    assert g_cube is not None, 'You must call l1_to_l0 within cube forward function'
    assert isinstance(src, Tensor), 'src must be Tensor'
    assert isinstance(dst, Tensor), 'dst must be Tensor'
    assert dst.pos in [Position.L0A, Position.L0B], f'DST Tensor position must be on L0A/L0B. Got {dst.pos}'
    assert src.pos==Position.L1, f'SRC Tensor position must be on L1. Got {src.pos}'

    g_cube.append(Instruction('LOADL0', dst=dst, src=src, m=m, n=n))


def l0c_to_gm_nz2nd(dst: GMTensor, src: Tensor, m: Union[int, Var], n: Union[int, Var], N_dst: Union[int, Var], m_src: Union[int, Var]):
    g_cube = context.active_cube
    assert g_cube is not None, 'You must call l1_to_l0 within cube forward function'
    assert isinstance(src, Tensor), 'src must be Tensor'
    assert isinstance(dst, GMTensor), 'dst must be GMTensor'
    assert src.pos==Position.L0C, f'SRC Tensor position must be on L0C. Got {src.pos}'

    g_cube.append(Instruction('L0C2GM_NZ2ND', dst=dst, src=src, m=m, n=n, N_dst=N_dst, m_src=m_src))


# mad 
def mad(dst: Tensor, srca: Tensor, srcb: Tensor, m: Union[int, Var], k: Union[int, Var], n: Union[int, Var], init_val: Union[bool, Var], uflag: Union[int, Var]=0):
    g_cube = context.active_cube
    assert g_cube is not None, 'You must call mad within cube forward function'
    assert isinstance(srca, Tensor), 'srca must be Tensor'
    assert isinstance(srcb, Tensor), 'srcb must be Tensor'
    assert isinstance(dst, Tensor), 'dst must be Tensor'
    assert srca.pos==Position.L0A, f'SRCA Tensor position must be on L0A. Got {srca.pos}'
    assert srcb.pos==Position.L0B, f'SRCB Tensor position must be on L0B. Got {srcb.pos}'
    assert dst.pos==Position.L0C, f'DST Tensor position must be on L0C. Got {dst.pos}'

    if isinstance(init_val, Var):
        init_val_str = str(init_val)
    elif isinstance(init_val, bool):
        init_val_str = 'true' if init_val else 'false'
    else:
        raise Exception('init_val must be bool or Var')

    g_cube.append(Instruction('MAD', dst=dst, srca=srca, srcb=srcb, m=m, k=k, n=n, init=init_val_str))

