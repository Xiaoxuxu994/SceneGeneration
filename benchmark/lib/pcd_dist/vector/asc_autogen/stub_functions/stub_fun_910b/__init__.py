from ... import context
context.set_device_type('910b3')

from ...flowcontrol import Loop, If, Else, Elif

from .flags import setflag, waitflag
from .barrier import barrier, bar_all, bar_fix, bar_m, bar_mte1, bar_mte2, bar_mte3, bar_v
from .consts import ComputeOffset, Max, Min, Align128, Align16, Align256, Align32, Align64, scalar_sqrt, CeilDiv, GetCubeIdx, GetCubeNum, GetSubBlockIdx, GetVecIdx, GetVecNum
from .cube import gm_to_l1_nd2nz, l1_to_l0, l1_to_l0_nz2nz, l1_to_l0_nz2zn, l1_to_l0_nz2nn, l1_to_l0_nz2zz, l0c_to_gm_nz2nd, mad
from .crosscore import cube_ready, wait_cube, vec_ready, wait_vec, allvec_ready, allvec_wait, allcube_ready, allcube_wait

from .vec.vecmask import set_mask, reset_mask
from .vec.datamove import gm_to_ub, ub_to_gm, gm_to_ub_pad, ub_to_gm_pad, ub_to_ub, ub_to_ub_nd2nz, ub_to_ub_nd2nz_compact
from .vec.sort import sort32
from .vec.unary import exp, ln, abs, rec, sqrt, rsqrt, vnot, relu
from .vec.binary import add, sub, mul, div, vmax, vmin, vand, vor
from .vec.unaryscalar import adds, muls, vmaxs, vmins, lrelu, axpy
from .vec.compareselect import compare, compare_scalar, compare_to_reg, set_cmpmask, select
from .vec.cast import cast 
from .vec.group import cadd, cgadd, cpadd, cmax, cgmax, cmin, cgmin
from .vec.dupbrcb import dup, brcb
from .vec.gatherscatter import gather, scatter

from ...utils import ROUNDMODE, CMPMODE, PIPE, DATATYPE, DT, Position, SELMODE
from ...utils import Var, Tensor, GMTensor, DBuff, TBuff, QBuff, PBuff, SEvent, DEvent

from ...cube import CubeModule
from ...vec import VecModule
from ...kernel import KernelBase
