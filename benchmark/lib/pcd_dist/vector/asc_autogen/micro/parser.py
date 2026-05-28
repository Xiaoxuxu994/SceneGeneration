from ..utils import CodeHelper, Var, Instruction, Tensor
from collections.abc import Callable


def assign_var(i: Instruction, h: CodeHelper):
    h(f'{i.v.name} = {i.src};')

def create_var(i: Instruction, h: CodeHelper):
    v: Var = i.v
    h(f'{v.dtype} {v.name} = {i.value};')

def reinterpret(i: Instruction, h: CodeHelper):
    h(f'__ubuf__ {i.dst.dtype}* {i.dst} = (__ubuf__ {i.dst.dtype}*) {i.src};')

# flow control 
def start_loop(i: Instruction, h: CodeHelper):
    if i.step==1:
        h(f'for (uint16_t {i.name}=(uint16_t){i.start}; {i.name}<(uint16_t){i.end}; ++{i.name}){{')
    else:
        raise SyntaxError('Loop in MicroModule should only have step=1')
        h(f'for (uint16_t {i.name}=(uint16_t){i.start}; {i.name}<(uint16_t){i.end}; {i.name}+=(uint16_t){i.step}){{')
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

# data move 
def ub_to_reg(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::DataCopy<{i.dst.dtype}, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>({i.dst}, {i.src}, {i.blk_stride}, {i.mask});')


def reg_to_ub(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::DataCopy<{i.dst.dtype}, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>({i.dst}, {i.src}, {i.blk_stride}, {i.mask});')


def ub_to_reg_cont(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::DataCopy<{i.dst.dtype}, MicroAPI::LoadDist::{i.mode}>({i.dst}, {i.src});')

def reg_to_ub_cont(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::DataCopy<{i.dst.dtype}, MicroAPI::StoreDist::{i.mode}>({i.dst}, {i.src}, {i.mask});')

# unary
def exp(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Exp({i.dst}, {i.src}, {i.mask});')

def abs(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Abs({i.dst}, {i.src}, {i.mask});')

def relu(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Relu({i.dst}, {i.src}, {i.mask});')

def sqrt(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Sqrt({i.dst}, {i.src}, {i.mask});')

def ln(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Ln({i.dst}, {i.src}, {i.mask});')

def log(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Log({i.dst}, {i.src}, {i.mask});')

def log2(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Log2({i.dst}, {i.src}, {i.mask});')

def log10(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Log10({i.dst}, {i.src}, {i.mask});')

def neg(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Neg({i.dst}, {i.src}, {i.mask});')

def vnot(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Not({i.dst}, {i.src}, {i.mask});')

def vcopy(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Copy({i.dst}, {i.src}, {i.mask});')

# binary 
def vmax(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Max({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def vmin(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Min({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def vsub(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Sub({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def vadd(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Add({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def vmul(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Mul({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def vdiv(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Div({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def vand(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::And({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def vor(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Or({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def vxor(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Xor({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def prelu(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Prelu({i.dst}, {i.src1}, {i.src2}, {i.mask});')

# src + val
def vmaxs(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Maxs({i.dst}, {i.src}, {i.v}, {i.mask});')

def vmins(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Mins({i.dst}, {i.src}, {i.v}, {i.mask});')

def vadds(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Adds({i.dst}, {i.src}, {i.v}, {i.mask});')

def vmuls(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Muls({i.dst}, {i.src}, {i.v}, {i.mask});')

def vlrelu(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::LeakyRelu({i.dst}, {i.src}, {i.v}, {i.mask});')

def vshiftrs(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::ShiftRights({i.dst}, {i.src}, (int16_t){i.v}, {i.mask});')

def vshiftls(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::ShiftLefts({i.dst}, {i.src}, (int16_t){i.v}, {i.mask});')

def vaxpy(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Axpy({i.dst}, {i.src}, {i.v}, {i.mask});')

# group functions 
def vcadd(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::ReduceSum({i.dst}, {i.src}, {i.mask});')

def vcmax(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::ReduceMax({i.dst}, {i.src}, {i.mask});')

def vcmin(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::ReduceMin({i.dst}, {i.src}, {i.mask});')

def vcgadd(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::ReduceSumWithDataBlock({i.dst}, {i.src}, {i.mask});')

def vcgmax(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::ReduceMaxWithDataBlock({i.dst}, {i.src}, {i.mask});')

def vcgmin(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::ReduceMinWithDataBlock({i.dst}, {i.src}, {i.mask});')

def vcpadd(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::PairReduceSum({i.dst}, {i.src}, {i.mask});')

# dup
def vdup(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Duplicate({i.dst}, {i.src}, {i.mask});')

# interleave 
def dinterleave(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::DeInterleave({i.dst0}, {i.dst1}, {i.src0}, {i.src1});')

def interleave(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Interleave({i.dst0}, {i.dst1}, {i.src0}, {i.src1});')

# cast 
def cast(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Cast<{i.ddst}, {i.dsrc}, {i.config}>({i.dst}, {i.src}, {i.mask});')

# arange 
def arange(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Arange<{i.dtype}, {i.mode}>({i.dst}, {i.v});')

# datacopy gather/scatter
def datacopygather(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::DataCopyGather({i.dst}, {i.src}, {i.index}, {i.mask});')

def datacopyscatter(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::DataCopyScatter({i.dst}, {i.src}, {i.index}, {i.mask});')

def gather(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Gather({i.dst}, {i.src}, {i.index});')

def gather_mask(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::GatherMask({i.dst}, {i.src}, {i.mask});')

# compare and select 
def compare(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Compare<{i.dtype}, CMPMODE::{i.mode}>({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def compares(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::CompareScalar<{i.dtype}, CMPMODE::{i.mode}>({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def select(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::Select({i.dst}, {i.src1}, {i.src2}, {i.mask});')


# mask functions 
def maskand(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::MaskAnd({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def maskor(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::MaskOr({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def maskxor(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::MaskXor({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def masknot(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::MaskNot({i.dst}, {i.src}, {i.mask});')

def maskmov(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::MaskMov({i.dst}, {i.src}, {i.mask});')

def maskinterleave(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::MaskInterleave({i.dst0}, {i.dst1}, {i.src0}, {i.src1});')

def maskdeinterleave(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::MaskDeInterleave({i.dst0}, {i.dst1}, {i.src0}, {i.src1});')

def masksel(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::MaskSel({i.dst}, {i.src1}, {i.src2}, {i.mask});')

def movemaskspr(i: Instruction, h: CodeHelper):
    h(f'{i.dst} = MicroAPI::MoveMask<{i.dst.dtype}>();')

def update_mask(i: Instruction, h: CodeHelper):
    h(f'{i.dst} = MicroAPI::UpdateMask<{i.dst.dtype}>({i.cnt});')

def maskpack(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::MaskPack<{i.mode}>({i.dst}, {i.src});')

def maskunpack(i: Instruction, h: CodeHelper):
    h(f'MicroAPI::MaskUnPack<{i.mode}>({i.dst}, {i.src});')



INST_MAPPING: dict[str, Callable[[Instruction, 'CodeHelper'], None]] = {
    'ASSIGNVAR'           : assign_var,
    'CREATEVAR'           : create_var,
    'REINTERPRET'         : reinterpret,
    # flowcontrol
    'STARTLOOP'           : start_loop,
    'ENDLOOP'             : end_loop,
    'STARTIF'             : start_if,
    'STARTELIF'           : start_elif,
    'STARTELSE'           : start_else,
    'ENDIF'               : end_if,
    # data move 
    'UB2REG'              : ub_to_reg,
    'REG2UB'              : reg_to_ub,
    'UB2REGCONT'          : ub_to_reg_cont,
    'REG2UBCONT'          : reg_to_ub_cont,
    # unary 
    'VEXP'                : exp,
    'VABS'                : abs,
    'VRELU'               : relu,
    'VSQRT'               : sqrt,
    'VLN'                 : ln,
    'VLOG'                : log,
    'VLOG2'               : log2,
    'VLOG10'              : log10,
    'VNEG'                : neg,
    'VNOT'                : vnot,
    'VCOPY'               : vcopy,
    # binary
    'VMAX'                : vmax,
    'VMIN'                : vmin,
    'VADD'                : vadd,
    'VSUB'                : vsub,
    'VMUL'                : vmul,
    'VDIV'                : vdiv,
    'VAND'                : vand,
    'VOR'                 : vor,
    'VXOR'                : vxor,
    'VPRELU'              : prelu,
    # src + val
    'VMAXS'               : vmaxs,
    'VMINS'               : vmins,
    'VADDS'               : vadds,
    'VMULS'               : vmuls,
    'VLRELU'              : vlrelu,
    'SHIFTRS'             : vshiftrs,
    'SHIFTLS'             : vshiftls,
    'VAXPY'               : vaxpy,
    # reduce functions 
    'VCADD'               : vcadd,
    'VCMAX'               : vcmax,
    'VCMIN'               : vcmin,
    'VCGADD'              : vcgadd,
    'VCGMAX'              : vcgmax,
    'VCGMIN'              : vcgmin,
    'VCPADD'              : vcpadd,
    # duplicate 
    'VDUP'                : vdup,
    # interleaves 
    'DINTERLEAVE'         : dinterleave,
    'INTERLEAVE'          : interleave,
    # Data type cast 
    'CAST'                : cast,
    # arange
    'ARANGE'              : arange,
    # gather and scatter 
    'DATACOPYGATHER'      : datacopygather,
    'DATACOPYSCATTER'     : datacopyscatter,
    'GATHER'              : gather,
    'GATHERMASK'          : gather_mask,
    # compare and select 
    'COMPARE'             : compare,
    'COMPARES'            : compares,
    'SELECT'              : select,
    # mask ops 
    'MASKMOV'             : maskmov,
    'MASKAND'             : maskand,
    'MASKOR'              : maskor,
    'MASKXOR'             : maskxor,
    'MASKNOT'             : masknot,
    'MASKINTERL'          : maskinterleave,
    'MASKDEINTERL'        : maskdeinterleave,
    'MASKSEL'             : masksel,
    'MOVEMASKSPR'         : movemaskspr,
    'UPDATEMASK'          : update_mask,
    'MASKPACK'            : maskpack,
    'MASKUNPACK'          : maskunpack,
}


def parse_all(inst_list: list[Instruction], h: 'CodeHelper'):
    for i in inst_list:
        INST_MAPPING[i._inst](i, h)
