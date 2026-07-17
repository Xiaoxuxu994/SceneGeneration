from typing import Any, Union, TypeVar
from ..micro import MicroModule
from ..utils import Var, GMTensor, Tensor, DBuff, TBuff, QBuff, PBuff, Instruction, DT, Position, SEvent, DEvent, CodeHelper, DTYPE, PIPE, Pos, sizeof
from .. import context
from . import parser

from rich.style import Style
from rich.text import Text 
from rich.console import Console
from functools import partial
print = partial(Console(width=150).print, justify='center')


BUFFER_T = TypeVar('BUFFER_T', bound=Union[DBuff, TBuff, QBuff, PBuff, Tensor])


class VecModule():
    _curr_phase: str
    _input_vars: list[Var]
    _input_tensors: list[GMTensor]
    _init_inst_list: list[Instruction]
    _compute_inst_list: list[Instruction]
    _buffer_list: list[Union[Tensor, DBuff, TBuff, QBuff, PBuff]]
    _event_list: list[Union[SEvent, DEvent]]
    _micro_list: list[MicroModule]
    _name: str
    _tmp_idx: int 

    def __init__(self, *args: Var):
        self._name = ''
        self._curr_phase = 'init'
        self._input_vars = []
        self._input_tensors = []
        self._init_inst_list = []
        self._compute_inst_list = []
        self._event_list = []
        self._buffer_list = []
        self._micro_list = []
        self._tmp_idx = 0
        
        for a in args:
            assert isinstance(a, Var), 'All input args should be Var'
            self._input_vars.append(a)
        
        context.active_cube = self
        self.in_empty = DEvent(PIPE.V, PIPE.MTE2)
        self.in_ready = DEvent(PIPE.MTE2, PIPE.V)


        self.out_empty = DEvent(PIPE.MTE3, PIPE.V)
        self.out_ready = DEvent(PIPE.V, PIPE.MTE3)
        self.initialize(*args)
        context.active_cube = None 

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            super().__setattr__(name, value)
            return 
        assert self._curr_phase=='init', 'Can only assign class variables in initialize'
        if isinstance(value, float):
            new_var = Var(f'shape.{name}', DT.float, value)
            self.append(Instruction('ASSIGNVAR', v=new_var, src=value))
        elif isinstance(value, int):
            new_var = Var(f'shape.{name}', DT.int, value)
            self.append(Instruction('ASSIGNVAR', v=new_var, src=value))
        elif isinstance(value, Var):
            new_var = value.copy()
            new_var.name = f'shape.{name}'
            new_var.varstr = f'shape.{name}'
            self.append(Instruction('ASSIGNVAR', v=new_var, src=value))
        elif isinstance(value, (Tensor, DBuff, TBuff, QBuff, PBuff)):
            new_var = value
            new_var.name = name 
            if new_var.pos is None:
                new_var.pos = Position.UB
            self._buffer_list.append(new_var)
        elif isinstance(value, (SEvent, DEvent)):
            new_var = value
            new_var.set_name(name)
            new_var.set_module(self)
            self._event_list.append(new_var)
        elif isinstance(value, MicroModule):
            self._micro_list.append(value)
            new_var = value
        else:
            raise TypeError(f'Not supported type {type(value)}')
        super().__setattr__(name, new_var)

    def set_name(self, name: str):
        self._name = name 
    
    def initialize(self, *args: Var):
        self.init(*args)
    
    def init(self, *args: Var):
        ...

    def __call__(self, *args: GMTensor):
        assert context.active_kernel is not None 
        context.active_kernel.run_module(self)
        self._curr_phase = 'computing'
        for a in args:
            assert isinstance(a, GMTensor), 'Forward inputs must be GMTensors'
            self._input_tensors.append(a)

        context.active_vec = self
        self.in_empty.setall()
        self.out_empty.setall()

        self.forward(*args)

        self.in_empty.release()
        self.out_empty.release()
        context.active_vec = None 
        self._curr_phase = 'finished'

    def forward(self, *args: GMTensor):
        ... 

    def append(self, inst: Instruction):
        if self._curr_phase=='init':
            self._init_inst_list.append(inst)
        elif self._curr_phase=='computing':
            self._compute_inst_list.append(inst)
        else:
            raise Exception(f'Can only call in initialize or forward. Not {self._curr_phase}')
        
    def create_l1(self, t: type[BUFFER_T], dtype: DTYPE, length: Union[int, Var]) -> BUFFER_T:
        new_obj = t(dtype, length)
        new_obj.pos = Position.L1
        return new_obj

    def create_var(self, name: str, dtype: DTYPE, value: Union[Var, int, float]):
        assert self._curr_phase=='computing', 'Can only create vars in forward function'
        new_var = Var(name, dtype)
        new_var.assign_module(self)
        self.append(Instruction('CREATEVAR', v=new_var, value=value))
        return new_var
    
    def wait_in_ready(self):
        self.in_ready.wait()

    def set_in_ready(self):
        self.in_ready.set()

    def wait_in_empty(self):
        self.in_empty.wait()

    def set_in_empty(self):
        self.in_empty.set()

    def wait_out_ready(self):
        self.out_ready.wait()
    
    def set_out_ready(self):
        self.out_ready.set()

    def wait_out_empty(self):
        self.out_empty.wait()

    def set_out_empty(self):
        self.out_empty.set()


    # codegen related 
    def gen_header(self, h: CodeHelper):
        # some includes and defines 
        h('#pragma once')
        h('#include "tensorutils.h"')
        for m in self._micro_list:
            h(f'#include "{m.__class__.__name__}.h"')
        h()
        h()

    def gen_tiling(self, h: CodeHelper):
        h(f'struct {self._name}ShapeInfo{{')
        h.ir()
        var_list = []
        for i in self._init_inst_list:
            if i._inst=='ASSIGNVAR':
                v: Var = i.v 
                if v.name in var_list:
                    continue 
                if 'shape.' in v.name:
                    var_list.append(v.name)
                    h(f'{v.dtype} {v.name.replace("shape.", "")};')
        h.il()
        h('};')
        h()
        h()

        arg_list = [f'{v.dtype} {v.name}' for v in self._input_vars]
        arg_list.append(f'{self._name}ShapeInfo &shape')
        arg_str = ', '.join(arg_list)
        h(f'__aicore__ inline void tilingShape{self._name}({arg_str}){{')
        h.ir()
        parser.parse_all(self._init_inst_list, h)
        h.il()
        h('}')
        h()
        h()

    def gen_init(self, h: CodeHelper, name: str):
        arg_list = ''
        for i in self._input_tensors:
            arg_list += f'GM_ADDR {i.name}_, '
        arg_list += f'{name}ShapeInfo shape_'
        h(f'__aicore__ inline void Init({arg_list}){{')
        h.ir()
        h('shape = shape_;')
        h('// Reset tpipe. Start resource distribution')
        h('TPipe* pipe_ptr = GetTPipePtr();')
        h('pipe_ptr->Reset();')
        h()
        h('// Global Tensors')
        for i in self._input_tensors:
            # h(f'{i.name} = Tensor<{i.dtype}, PGM>({i.name}_);')
            h(f'{i.name}.SetGlobalBuffer((__gm__ {i.dtype}*) {i.name}_);')
        h()
        h('// Local buffers')
        for i in self._buffer_list:
            if isinstance(i, (DBuff, TBuff, QBuff, PBuff)):
                h(f'{i.name}.Init({i.length});')
            else:
                # h(f'{i.name} = Tensor<{i.dtype}, PUB>(0, {i.length}, offset);')
                if i.pos==Position.UB:
                    h(f'AllocateLocalTensor<TPosition::VECCALC>({i.name}, {i.length});')
                elif i.pos==Position.L1:
                    h(f'AllocateLocalTensor<TPosition::A1>({i.name}, {i.length});')
                else:
                    raise ValueError(f'Unsupported position: {i.pos}')
        
        h('// Initialize events')
        for e in self._event_list:
            h(f'{e.name}.Init();')
        h.il()
        h('}')
        h()

    def gen_compute(self, h: CodeHelper):
        h('__aicore__ inline void Compute(){')
        h.ir()
        parser.parse_all(self._compute_inst_list, h)
        h.il()
        h('}')
        h()
    
    def gen_members(self, h: CodeHelper, name: str):
        h(f'{name}ShapeInfo shape;')
        h('// Global Tensors')
        for i in self._input_tensors:
            h(f'GlobalTensor<{i.dtype}> {i.name};')
        h('// Local buffers')
        for i in self._buffer_list:
            if i.pos==Position.UB:
                if isinstance(i, DBuff):
                    h(f'DBuff<{i.dtype}, TPosition::VECCALC> {i.name};')
                elif isinstance(i, TBuff):
                    h(f'TBuff<{i.dtype}, TPosition::VECCALC> {i.name};')
                elif isinstance(i, QBuff):
                    h(f'QBuff<{i.dtype}, TPosition::VECCALC> {i.name};')
                elif isinstance(i, PBuff):
                    h(f'PBuff<{i.dtype}, TPosition::VECCALC> {i.name};')
                else:
                    h(f'LocalTensor<{i.dtype}> {i.name};')
        for i in self._buffer_list:
            if i.pos==Position.L1:
                if isinstance(i, DBuff):
                    h(f'DBuff<{i.dtype}, TPosition::A1> {i.name};')
                elif isinstance(i, TBuff):
                    h(f'TBuff<{i.dtype}, TPosition::A1> {i.name};')
                elif isinstance(i, QBuff):
                    h(f'QBuff<{i.dtype}, TPosition::A1> {i.name};')
                elif isinstance(i, PBuff):
                    h(f'PBuff<{i.dtype}, TPosition::A1> {i.name};')
                else:
                    h(f'LocalTensor<{i.dtype}> {i.name};')
        h()
        h('// Events')
        for e in self._event_list:
            if isinstance(e, SEvent):
                h(f'SEvent<PIPE_{e.src}, PIPE_{e.dst}> {e.name};' )
            if isinstance(e, DEvent):
                h(f'DEvent<PIPE_{e.src}, PIPE_{e.dst}> {e.name};' )

    def check_usage_pos(self, pos: Pos):
        print(f'┌───────────────── {str(pos).center(5)} Usage ─────────────────┐')
        total = 0
        for i in self._buffer_list:
            multiplier = 1
            if isinstance(i, DBuff):
                multiplier = 2 
            if isinstance(i, TBuff):
                multiplier = 3
            if isinstance(i, QBuff):
                multiplier = 4
            if isinstance(i, PBuff):
                multiplier = 5
            if i.pos==pos:
                if isinstance(i.length, Var):
                    if i.length.value is None:
                        print(f'Cannot determine size of buffer {i.name}')
                    else:
                        s = i.length.value * sizeof(i.dtype) * multiplier
                        total += s
                        print(f'[{i.__class__.__name__}] {i.name} : {s/1024} KB')
                elif isinstance(i.length, int):
                    s = i.length * sizeof(i.dtype) * multiplier
                    total += s
                    print(f'[{i.__class__.__name__}] {i.name} : {s/1024} KB')
        print('-------------------------------------------')
        print(f'Total {pos} usage: {total/1024} / {context.get_max_size(pos)/1024} KB')
        if total>context.get_max_size(pos):
            print(Text(f'WARNING: Exceed max capacity!!', style=Style(color='deep_pink4', bold=True, reverse=True)))
        print('└───────────────────────────────────────────────┘')

    def check_starting_pos(self, pos: Pos):
        print(f'┌──────────── {str(pos).center(5)} Starting Position ────────────┐')
        total = 0
        for i in self._buffer_list:
            multiplier = 1
            if isinstance(i, DBuff):
                multiplier = 2 
            if isinstance(i, TBuff):
                multiplier = 3
            if isinstance(i, QBuff):
                multiplier = 4
            if isinstance(i, PBuff):
                multiplier = 5
            if i.pos==pos:
                if isinstance(i.length, Var):
                    if i.length.value is None:
                        print(f'Cannot determine size of buffer {i.name}')
                    else:
                        s = i.length.value * sizeof(i.dtype) * multiplier
                        total += s
                        print(f'[{i.__class__.__name__}] {i.name} : {total:,}')
                elif isinstance(i.length, int):
                    s = i.length * sizeof(i.dtype) * multiplier
                    total += s
                    print(f'[{i.__class__.__name__}] {i.name} : {total:,}')
        print('└─────────────────────────────────────────────────┘')

    def gen_code(self, filename: str):
        print()
        outstr = Text()
        outstr.append('VEC: Generating [')
        outstr.append(Text(self._name, style=Style(color='red', bold=True, underline=True)))
        outstr.append(']')
        Console(width=150).rule(outstr)
        if self._curr_phase!='finished':
            print(Text('Warning', style='yellow').append(Text(': [', style='white')).append(Text(f'{self._name}', style='red')).append(Text('] is not forwarded. Skip generation.', style='white')))
            return 
        
        h = CodeHelper()
        self.gen_header(h)
        self.gen_tiling(h)
        
        h(f'class {self._name}{{')
        h('public:')
        h.ir()
        h(f'__aicore__ inline {self._name}(){{}}')
        self.gen_init(h, self._name)
        self.gen_compute(h)
        h.il()
        h('private:')
        h.ir()
        self.gen_members(h, self._name)
        h.il()
        h('};')

        for p in [Position.UB, Position.L1]:
            self.check_usage_pos(p)

        fout = open(filename, 'w')
        fout.write(h.result)
        fout.close()

        for m in self._micro_list:
            m.gen_code('codebase/' + m.__class__.__name__ + '.h')
