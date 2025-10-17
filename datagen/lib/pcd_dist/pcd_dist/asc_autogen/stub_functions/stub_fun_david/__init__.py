from ... import context
context.set_device_type('david')

from ...flowcontrol import Loop, If, Else, Elif

from .flags import setflag, waitflag
from .barrier import barrier, bar_all, bar_fix, bar_m, bar_mte1, bar_mte2, bar_mte3, bar_v
from .consts import ComputeOffset, Max, Min, Align128, Align16, Align256, Align32, Align64, scalar_sqrt, CeilDiv, GetCubeIdx, GetCubeNum, GetSubBlockIdx, GetVecIdx, GetVecNum
from .cube import gm_to_l1_nd2nz, l1_to_l0, l1_to_l0_nz2nz, l1_to_l0_nz2zn, l0c_to_gm_nz2nd, l0c_to_ub_nz2nd, mad
from .crosscore import cube_ready, wait_cube, vec_ready, wait_vec, allvec_ready, allvec_wait, allcube_ready, allcube_wait
from .vec import gm_to_ub, gm_to_ub_pad, ub_to_gm, ub_to_gm_pad, ub_to_l1, ub_to_l1_nd2nz, ub_to_l1_nz, ub_to_ub, ub_to_ub_nd2nz, ub_to_ub_nd2nz_compact, sort32, mergesort4
from .micro.datamove import ub_to_reg, reg_to_ub, ub_to_reg_continuous, ub_to_reg_gather, reg_to_ub_scatter, gather, gather_mask, reg_to_ub_continuous
from .micro.unary import exp, abs, relu, sqrt, ln, log, log10, log2, neg, vnot, vcopy
from .micro.binary import vmax, vmin, add, sub, mul, div, vand, vor, vxor, prelu
from .micro.unaryscalar import vmaxs, vmins, muls, adds, lrelu, shiftls, shiftrs, axpy
from .micro.group import cmax, cmin, cgadd, cgmax, cgmin, cpadd, cadd
from .micro.dup import dup
from .micro.interleave import interleave, deinterleave
from .micro.cast import cast
from .micro.arange import arange 
from .micro.compare import compare, select
from .micro.mask import mask_and, mask_deinterleave, mask_interleave, mask_mov, mask_not, mask_or, mask_pack, mask_sel, mask_unpack, mask_xor, move_mask_spr, update_mask

from ...utils import LOADDIST, ROUNDMODE, CMPMODE, CastConfig, PIPE, DATATYPE, DT, Position, DUALMODE, STOREDIST
from ...utils import Var, Tensor, GMTensor, DBuff, TBuff, QBuff, PBuff, SEvent, DEvent, Reg, MaskReg, RegLayout, MaskType

from ...cube import CubeModule
from ...vec import VecModule
from ...micro import MicroModule
from ...kernel import KernelBase
