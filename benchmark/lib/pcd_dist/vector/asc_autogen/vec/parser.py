from ..utils import CodeHelper, Var, Instruction, Tensor
from collections.abc import Callable

_select_cnt = 0

def set_flag(i: Instruction, h: CodeHelper):
    h(f'set_flag(PIPE_{i.src}, PIPE_{i.dst}, (event_t){i.idd});')

def wait_flag(i: Instruction, h: CodeHelper):
    h(f'wait_flag(PIPE_{i.src}, PIPE_{i.dst}, (event_t){i.idd});')

def bar(i: Instruction, h: CodeHelper):
    h(f'PipeBarrier<PIPE_{i.pipe}>();')

def assign_var(i: Instruction, h: CodeHelper):
    h(f'{i.v.name} = {i.src};')

def create_var(i: Instruction, h: CodeHelper):
    v: Var = i.v
    h(f'{v.dtype} {v.name} = {i.value};')

def get_val(i: Instruction, h: CodeHelper):
    h(f'{i.dst} = ({i.dst.dtype}) {i.src}.GetValue(0);')

def set_val(i: Instruction, h: CodeHelper):
    h(f'{i.src}.SetValue(0, ({i.dst.dtype})({i.dst}));')

def reinterpret(i: Instruction, h: CodeHelper):
    h(f'LocalTensor<{i.dst.dtype}> {i.dst} = {i.src}.ReinterpretCast<{i.dst.dtype}>();')

# flow control 
def start_loop(i: Instruction, h: CodeHelper):
    h(f'for (int {i.name}={i.start}; {i.name}<{i.end}; {i.name}+={i.step}){{')
    h.ir()

def end_loop(i: Instruction, h: CodeHelper):
    h.il()
    h('}')

def start_if(i: Instruction, h: CodeHelper):
    h(f'if ({i.cond}){{')
    h.ir()

def start_elif(i: Instruction, h: CodeHelper):
    h(f'else if({i.cond}){{')
    h.ir()

def start_else(i: Instruction, h: CodeHelper):
    h('else {')
    h.ir()

def end_if(i: Instruction, h: CodeHelper):
    h.il()
    h('}')

# event 
def event_set(i: Instruction, h: CodeHelper):
    h(f'{i.name}.set();')

def event_wait(i: Instruction, h: CodeHelper):
    h(f'{i.name}.wait();')

def event_setall(i: Instruction, h: CodeHelper):
    h(f'{i.name}.setall();')

def event_release(i: Instruction, h: CodeHelper):
    h(f'{i.name}.release();')

# cross-core
def vec_ready(i: Instruction, h: CodeHelper):
    h(f'VEC_READY({i.flag}, PIPE_{i.pipe});')

def wait_cube(i: Instruction, h: CodeHelper):
    h(f'WAIT_CUBE({i.flag});')

def allvec_ready(i: Instruction, h: CodeHelper):
    h(f'ALLVEC_READY({i.flag}, PIPE_{i.pipe});')

def allvec_wait(i: Instruction, h: CodeHelper):
    h(f'ALLVEC_WAIT({i.flag});')

# mte2 
def gm_to_ub(i: Instruction, h: CodeHelper):
    h(f'GM2UB({i.dst}, {i.src}, {i.n_burst}, {i.burst_len}, {i.src_stride}, {i.dst_stride});')

def gm_to_ub_pad(i: Instruction, h: CodeHelper):
    h(f'GM2UBPad({i.dst}, {i.src}, {i.n_burst}, {i.burst_len}, {i.src_stride}, {i.dst_stride},{i.isPad},{i.leftPadding},{i.rightPadding},{i.paddingValue});')
    # h(f'GM2UB({i.dst}, {i.src}, {i.n_burst}, {i.burst_len}, {i.src_stride}, {i.dst_stride});')

def ub_to_ub_nd2nz(i: Instruction, h: CodeHelper):
    h(f'UB2UB_ND2NZ({i.dst}, {i.src}, {i.m_dst}, {i.n_dst}, {i.m_src}, {i.n_src});')

def ub_to_ub_nd2nz_compact(i: Instruction, h: CodeHelper):
    h(f'UB2UB_ND2NZ_COMPACT({i.dst}, {i.src}, {i.m}, {i.n});')

# V 
def ub_to_ub(i: Instruction, h: CodeHelper):
    h(f'UB2UB({i.dst}, {i.src}, {i.n_burst}, {i.burst_len}, {i.src_stride}, {i.dst_stride});') 

# mte3
def ub_to_gm(i: Instruction, h: CodeHelper):
    h(f'UB2GM({i.dst}, {i.src}, {i.n_burst}, {i.burst_len}, {i.src_stride}, {i.dst_stride});')

def ub_to_gm_pad(i: Instruction, h: CodeHelper):
    h(f'UB2GMPad({i.dst}, {i.src}, {i.n_burst}, {i.burst_len}, {i.src_stride}, {i.dst_stride});')

def ub_to_l1(i: Instruction, h: CodeHelper):
    h(f'UB2L1({i.dst}, {i.src}, {i.n_burst}, {i.burst_len}, {i.src_stride}, {i.dst_stride});')

def ub_to_l1_nz(i: Instruction, h: CodeHelper):
    h(f'UB2L1_NZ({i.dst}, {i.src}, {i.m_dst}, {i.n_dst}, {i.m_src}, {i.n_src});')

def ub_to_l1_nd2nz(i: Instruction, h: CodeHelper):
    h(f'UB2L1_ND2NZ({i.dst}, {i.src}, {i.m_dst}, {i.n_dst}, {i.m_src}, {i.n_src});')

# sort 
def sort32(i: Instruction, h: CodeHelper):
    h(f'Sort32({i.dst}, {i.src}, {i.idx}, {i.repeat});')

def mergesort4(i: Instruction, h: CodeHelper):
    h(f'MERGESORT4({i.dst}, {i.src});')

# call micro 
def call_micro(i: Instruction, h: CodeHelper):
    tmp = []
    for a in i.args:
        if isinstance(a, Tensor):
            tmp.append(f'(__ubuf__ {a.dtype}*) {a.name}.GetPhyAddr()')
        else:
            tmp.append(str(a))
    args_str = ", ".join(tmp)
    h(f'{i.name}({args_str});')

# vector masks 
def set_mask(i: Instruction, h: CodeHelper):
    h(f'SetVectorMask<half, MaskMode::NORMAL>({i.high}, {i.low});')

def reset_mask(i: Instruction, h: CodeHelper):
    h('ResetMask();')


## 910B computations 
# unary 
def exp(i: Instruction, h: CodeHelper):
    h(f'Exp<{i.dst.dtype}, false>({i.dst}, {i.src}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

def ln(i: Instruction, h: CodeHelper):
    h(f'Ln<{i.dst.dtype}, false>({i.dst}, {i.src}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

def abs(i: Instruction, h: CodeHelper):
    h(f'Abs<{i.dst.dtype}, false>({i.dst}, {i.src}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

def rec(i: Instruction, h: CodeHelper):
    h(f'Reciprocal<{i.dst.dtype}, false>({i.dst}, {i.src}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

def sqrt(i: Instruction, h: CodeHelper):
    h(f'Sqrt<{i.dst.dtype}, false>({i.dst}, {i.src}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

def rsqrt(i: Instruction, h: CodeHelper):
    h(f'Rsqrt<{i.dst.dtype}, false>({i.dst}, {i.src}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

def vnot(i: Instruction, h: CodeHelper):
    h(f'Not<{i.dst.dtype}, false>({i.dst}, {i.src}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

def relu(i: Instruction, h: CodeHelper):
    h(f'Relu<{i.dst.dtype}, false>({i.dst}, {i.src}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

# binary 
def add(i: Instruction, h: CodeHelper):
    h(f'Add<{i.dst.dtype}, false>({i.dst}, {i.src1}, {i.src2}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride}}});')

def sub(i: Instruction, h: CodeHelper):
    h(f'Sub<{i.dst.dtype}, false>({i.dst}, {i.src1}, {i.src2}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride}}});')

def mul(i: Instruction, h: CodeHelper):
    h(f'Mul<{i.dst.dtype}, false>({i.dst}, {i.src1}, {i.src2}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride}}});')

def div(i: Instruction, h: CodeHelper):
    h(f'Div<{i.dst.dtype}, false>({i.dst}, {i.src1}, {i.src2}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride}}});')

def vmax(i: Instruction, h: CodeHelper):
    h(f'Max<{i.dst.dtype}, false>({i.dst}, {i.src1}, {i.src2}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride}}});')

def vmin(i: Instruction, h: CodeHelper):
    h(f'Min<{i.dst.dtype}, false>({i.dst}, {i.src1}, {i.src2}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride}}});')

def vand(i: Instruction, h: CodeHelper):
    h(f'And<{i.dst.dtype}, false>({i.dst}, {i.src1}, {i.src2}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride}}});')

def vor(i: Instruction, h: CodeHelper):
    h(f'Or<{i.dst.dtype}, false>({i.dst}, {i.src1}, {i.src2}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride}}});')

# unaryscalar
def adds(i: Instruction, h: CodeHelper):
    h(f'Adds<{i.dst.dtype}, false>({i.dst}, {i.src}, ({i.dst.dtype}){i.val}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

def muls(i: Instruction, h: CodeHelper):
    h(f'Muls<{i.dst.dtype}, false>({i.dst}, {i.src}, ({i.dst.dtype}){i.val}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

def vmaxs(i: Instruction, h: CodeHelper):
    h(f'Maxs<{i.dst.dtype}, false>({i.dst}, {i.src}, ({i.dst.dtype}){i.val}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

def vmins(i: Instruction, h: CodeHelper):
    h(f'Mins<{i.dst.dtype}, false>({i.dst}, {i.src}, ({i.dst.dtype}){i.val}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

def lrelu(i: Instruction, h: CodeHelper):
    h(f'LeakyRelu<{i.dst.dtype}, false>({i.dst}, {i.src}, ({i.dst.dtype}){i.val}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

def axpy(i: Instruction, h: CodeHelper):
    h(f'Axpy<{i.dst.dtype}, {i.src.dtype}, false>({i.dst}, {i.src}, ({i.dst.dtype}){i.val}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

# compare 
def compare(i: Instruction, h: CodeHelper):
    h(f'Compare<{i.src1.dtype}, {i.dst.dtype}, false>({i.dst}, {i.src1}, {i.src2}, CMPMODE::{i.mode}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride}}});')

def compares(i: Instruction, h: CodeHelper):
    h(f'CompareScalar<{i.src1.dtype}, {i.dst.dtype}, false>({i.dst}, {i.src1}, ({i.src1.dtype}){i.src2}, CMPMODE::{i.mode}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src1_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}}});')

def compare_to_reg(i: Instruction, h: CodeHelper):
    h(f'Compare<{i.src1.dtype}, false>({i.src1}, {i.src2}, CMPMODE::{i.mode}, MASK_PLACEHOLDER, {{{i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride}}});')

def select(i: Instruction, h: CodeHelper):
    global _select_cnt
    h(f'uint64_t maskfull_{_select_cnt}[2] = {{(uint64_t)-1, (uint64_t)-1}};')
    
    if isinstance(i.src2, Tensor):
        h(f'Select<{i.dst.dtype}, {i.selmask.dtype}, false>({i.dst}, {i.selmask}, {i.src1}, {i.src2}, SELMODE::VSEL_{i.mode}, maskfull_{_select_cnt}, {i.repeat}, {{{i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride}}});')
    else:
        h(f'Select<{i.dst.dtype}, {i.selmask.dtype}, false>({i.dst}, {i.selmask}, {i.src1}, ({i.src1.dtype}){i.src2}, SELMODE::VSEL_{i.mode}, maskfull_{_select_cnt}, {i.repeat}, {{{i.dst_blk_stride}, {i.src1_blk_stride}, {i.src2_blk_stride}, {i.dst_rep_stride}, {i.src1_rep_stride}, {i.src2_rep_stride}}});')
        
    _select_cnt += 1 

def set_cmpmask(i: Instruction, h: CodeHelper):
    h(f'SetCmpMask({i.src});')
    
# cast 
def cast(i: Instruction, h: CodeHelper):
    h(f'Cast<{i.dst.dtype}, {i.src.dtype}, false>({i.dst}, {i.src}, RoundMode::{i.mode}, MASK_PLACEHOLDER, {i.repeat}, {{{i.dst_blk_stride}, {i.src_blk_stride}, {i.dst_rep_stride}, {i.src_rep_stride}}});')

# group functions 
def vcadd(i: Instruction, h: CodeHelper):
    h(f'WholeReduceSum<{i.dst.dtype}, false>({i.dst}, {i.src}, MASK_PLACEHOLDER, {i.repeat}, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride});')

def vcgadd(i: Instruction, h: CodeHelper):
    h(f'BlockReduceSum<{i.dst.dtype}, false>({i.dst}, {i.src}, {i.repeat}, MASK_PLACEHOLDER, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride});')

def vcpadd(i: Instruction, h: CodeHelper):
    h(f'PairReduceSum<{i.dst.dtype}, false>({i.dst}, {i.src}, {i.repeat}, MASK_PLACEHOLDER, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride});')

def vcmax(i: Instruction, h: CodeHelper):
    h(f'WholeReduceMax<{i.dst.dtype}, false>({i.dst}, {i.src}, MASK_PLACEHOLDER, {i.repeat}, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride}, ReduceOrder::{i.mode});')

def vcgmax(i: Instruction, h: CodeHelper):
    h(f'BlockReduceMax<{i.dst.dtype}, false>({i.dst}, {i.src}, {i.repeat}, MASK_PLACEHOLDER, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride});')

def vcmin(i: Instruction, h: CodeHelper):
    h(f'WholeReduceMin<{i.dst.dtype}, false>({i.dst}, {i.src}, MASK_PLACEHOLDER, {i.repeat}, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride}, ReduceOrder::ORDER_ONLY_VALUE);')

def vcgmin(i: Instruction, h: CodeHelper):
    h(f'BlockReduceMin<{i.dst.dtype}, false>({i.dst}, {i.src}, {i.repeat}, MASK_PLACEHOLDER, {i.dst_rep_stride}, {i.src_blk_stride}, {i.src_rep_stride});')

# dup brcb 
def dup(i: Instruction, h: CodeHelper):
    h(f'Duplicate<{i.dst.dtype}, false>({i.dst}, ({i.dst.dtype}){i.src}, MASK_PLACEHOLDER, {i.repeat}, {i.dst_blk_stride}, {i.dst_rep_stride});')

def brcb(i: Instruction, h: CodeHelper):
    h(f'Brcb({i.dst}, {i.src}, {i.repeat}, {{{i.dst_blk_stride}, {i.dst_rep_stride}}});')

# gather and scatter 
def gather(i: Instruction, h: CodeHelper):
    h(f'Gather({i.dst}, {i.src}, {i.offset}, 0, VECTORFULLMASK, {i.repeat}, {i.dst_rep_stride});')

def scatter(i: Instruction, h: CodeHelper):
    h(f'Scatter({i.dst}, {i.src}, {i.offset}, 0, VECTORFULLMASK, {i.repeat}, {i.src_rep_stride});')


INST_MAPPING: dict[str, Callable[[Instruction, 'CodeHelper'], None]] = {
    'BAR'                   : bar,
    'SETFLAG'               : set_flag,
    'WAITFLAG'              : wait_flag,
    'ASSIGNVAR'             : assign_var,
    'CREATEVAR'             : create_var,
    'GETVAL'                : get_val,
    'SETVAL'                : set_val,
    'REINTERPRET'           : reinterpret,
    # flowcontrol
    'STARTLOOP'             : start_loop,
    'ENDLOOP'               : end_loop,
    'STARTIF'               : start_if,
    'STARTELIF'             : start_elif,
    'STARTELSE'             : start_else,
    'ENDIF'                 : end_if,
    # events 
    'EVENTSET'              : event_set,
    'EVENTWAIT'             : event_wait,
    'EVENTSETALL'           : event_setall,
    'EVENTRELEASE'          : event_release,
    # cross-core
    'VECREADY'              : vec_ready,
    'WAITCUBE'              : wait_cube,
    'ALLVECREADY'           : allvec_ready,
    'ALLVECWAIT'            : allvec_wait,
    # mte2
    'GM2UB'                 : gm_to_ub,
    'GM2UBPad'              : gm_to_ub_pad,
    # V
    'UB2UB'                 : ub_to_ub,
    'UB2UB_ND2NZ'           : ub_to_ub_nd2nz,
    'UB2UB_ND2NZ_CPCT'      : ub_to_ub_nd2nz_compact,
    # mte3
    'UB2GM'                 : ub_to_gm,
    'UB2GMPad'              : ub_to_gm_pad,
    'UB2L1'                 : ub_to_l1,
    'UB2L1_NZ'              : ub_to_l1_nz,
    'UB2L1_ND2NZ'           : ub_to_l1_nd2nz,
    # sort 
    'SORT32'                : sort32,
    'MERGESORT4'            : mergesort4,
    # call micro 
    'CALLMICRO'             : call_micro, 
    # # 910B - Vector computations 
    # vector masks 
    'SETMASK'               : set_mask,
    'RESETMASK'             : reset_mask,
    # unary 
    'EXP'                   : exp,
    'LN'                    : ln,
    'ABS'                   : abs,
    'REC'                   : rec,
    'SQRT'                  : sqrt,
    'RSQRT'                 : rsqrt,
    'VNOT'                  : vnot,
    'RELU'                  : relu,
    'NOT'                   : vnot,
    # binary 
    'ADD'                   : add,
    'SUB'                   : sub,
    'MUL'                   : mul,
    'DIV'                   : div,
    'MAX'                   : vmax,
    'MIN'                   : vmin,
    'AND'                   : vand,
    'OR'                    : vor,
    # unaryscalar
    'ADDS'                  : adds,
    'MULS'                  : muls,
    'MAXS'                  : vmaxs,
    'MINS'                  : vmins,
    'LRELU'                 : lrelu,
    'AXPY'                  : axpy,
    # compare 
    'COMPARE'               : compare,
    'COMPARES'              : compares,
    'COMPARETOREG'          : compare_to_reg,
    'SETCMPMASK'            : set_cmpmask,
    'SELECT'                : select,
    # cast 
    'CAST'                  : cast,
    # group
    'VCADD'                 : vcadd,
    'VCGADD'                : vcgadd,
    'VCPADD'                : vcpadd,
    'VCMAX'                 : vcmax,
    'VCGMAX'                : vcgmax,
    'VCMIN'                 : vcmin,
    'VCGMIN'                : vcgmin,
    # dup brcb 
    'DUP'                   : dup,
    'BRCB'                  : brcb,
    # gather scatter 
    'GATHER'                : gather,
    'SCATTER'               : scatter,
}


def parse_all(inst_list: list[Instruction], h: 'CodeHelper'):
    for i in inst_list:
        INST_MAPPING[i._inst](i, h)
