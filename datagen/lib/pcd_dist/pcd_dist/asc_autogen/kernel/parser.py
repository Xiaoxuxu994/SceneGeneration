from collections.abc import Callable
from ..utils import Instruction, Var, CodeHelper
    


def run(i: Instruction, h: 'CodeHelper'):
    h(f'{i.name}.Compute();')


def assign_var(i: Instruction, h: CodeHelper):
    h(f'{i.v.name} = {i.src};')

def create_var(i: Instruction, h: CodeHelper):
    v: Var = i.v
    h(f'{v.dtype} {v.name} = {i.value};')


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


INST_MAPPING: dict[str, Callable[[Instruction, 'CodeHelper'], None]] = {
    'RUN'                 : run,
    'ASSIGNVAR'           : assign_var,
    'CREATEVAR'           : create_var,
    # flowcontrol
    'STARTLOOP'           : start_loop,
    'ENDLOOP'             : end_loop,
    'STARTIF'             : start_if,
    'STARTELIF'           : start_elif,
    'STARTELSE'           : start_else,
    'ENDIF'               : end_if,
}

def parse_all(inst_list: list[Instruction], h: 'CodeHelper'):
    for i in inst_list:
        INST_MAPPING[i._inst](i, h)

