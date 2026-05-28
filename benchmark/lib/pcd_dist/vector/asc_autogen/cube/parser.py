from ..utils import CodeHelper, Var, Instruction
from collections.abc import Callable


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

# mte
def gm_to_l1_nd2nz(i: Instruction, h: CodeHelper):
    h(f'L1ND2NZ({i.dst}, {i.src}, {i.m}, {i.n}, {i.N}, {i.M});')

def l0_nz2nz(i: Instruction, h: CodeHelper):
    h(f'L0NZ2NZ({i.dst}, {i.src}, {i.mdst}, {i.ndst}, {i.msrc}, {i.nsrc});')

def l0_nz2zn(i: Instruction, h: CodeHelper):
    h(f'L0NZ2ZN({i.dst}, {i.src}, {i.mdst}, {i.ndst}, {i.msrc}, {i.nsrc});')

def l0_nz2zz(i: Instruction, h: CodeHelper):
    h(f'L0NZ2ZZ({i.dst}, {i.src}, {i.mdst}, {i.ndst}, {i.msrc}, {i.nsrc});')

def l0_nz2nn(i: Instruction, h: CodeHelper):
    h(f'L0NZ2NN({i.dst}, {i.src}, {i.mdst}, {i.ndst}, {i.msrc}, {i.nsrc});')

def load_l0(i: Instruction, h: CodeHelper):
    h(f'LOADL0({i.dst}, {i.src}, {i.m}, {i.n});')

# mad 
def mad(i: Instruction, h: CodeHelper):
    h(f'MMAD({i.dst}, {i.srca}, {i.srcb}, {i.m}, {i.k}, {i.n}, {i.init}, 0);')

# fix 
def l0c_to_ub_nz2nd(i: Instruction, h: CodeHelper):
    subid = 0 if not i.subid else i.subid
    h(f'L0C2UB_NZ2ND({i.dst}, {i.src}, {i.m}, {i.n}, {i.N_dst}, {i.m_src}, {i.dualmode}, {subid});')

def l0c_to_gm_nz2nd(i: Instruction, h: CodeHelper):
    h(f'L0C2GM_NZ2ND({i.dst}, {i.src}, {i.m}, {i.n}, {i.N_dst}, {i.m_src}, {0});')

# cross-core
def cube_ready(i: Instruction, h: CodeHelper):
    h(f'CUBE_READY({i.flag}, PIPE_{i.pipe});')

def wait_vec(i: Instruction, h: CodeHelper):
    h(f'WAIT_VEC({i.flag});')

def allcube_ready(i: Instruction, h: CodeHelper):
    h(f'ALLCUBE_READY({i.flag}, PIPE_{i.pipe});')

def allcube_wait(i: Instruction, h: CodeHelper):
    h(f'ALLCUBE_WAIT({i.flag});')


INST_MAPPING: dict[str, Callable[[Instruction, 'CodeHelper'], None]] = {
    'BAR'                 : bar,
    'SETFLAG'             : set_flag,
    'WAITFLAG'            : wait_flag,
    'ASSIGNVAR'           : assign_var,
    'CREATEVAR'           : create_var,
    'GETVAL'              : get_val,
    'REINTERPRET'         : reinterpret,
    # flowcontrol
    'STARTLOOP'           : start_loop,
    'ENDLOOP'             : end_loop,
    'STARTIF'             : start_if,
    'STARTELIF'           : start_elif,
    'STARTELSE'           : start_else,
    'ENDIF'               : end_if,
    # events 
    'EVENTSET'            : event_set,
    'EVENTWAIT'           : event_wait,
    'EVENTSETALL'         : event_setall,
    'EVENTRELEASE'        : event_release,
    # mte2 
    'L1ND2NZ'             : gm_to_l1_nd2nz,
    # mte1
    'L0NZ2NZ'             : l0_nz2nz,
    'L0NZ2ZN'             : l0_nz2zn,
    'L0NZ2ZZ'             : l0_nz2zz,
    'L0NZ2NN'             : l0_nz2nn,
    'LOADL0'              : load_l0,
    # fix
    'L0C2GM_NZ2ND'        : l0c_to_gm_nz2nd,
    'L0C2UB_NZ2ND'        : l0c_to_ub_nz2nd,
    # mad 
    'MAD'                 : mad,
    # cross-core
    'CUBEREADY'           : cube_ready,
    'WAITVEC'             : wait_vec,
    'ALLCUBEREADY'        : allcube_ready,
    'ALLCUBEWAIT'         : allcube_wait,
}


def parse_all(inst_list: list[Instruction], h: 'CodeHelper'):
    for i in inst_list:
        INST_MAPPING[i._inst](i, h)
