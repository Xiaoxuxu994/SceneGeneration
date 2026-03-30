from typing import Any, Union, TypeVar
from ..utils import Var, Tensor, Instruction, DT, CodeHelper, DTYPE, sizeof, Reg, CastConfig, MaskReg, Position
from .. import context
from . import parser

from rich.style import Style
from rich.text import Text 
from rich.console import Console
from functools import partial
print = partial(Console(width=150).print, justify='center')


class MicroModule():
    _curr_phase: str
    _reg_list: list[Union[Reg, MaskReg]]
    _input_list: list[Union[Tensor, Var]]
    _inst_list: list[Instruction]
    _cast_list: list[CastConfig]
    _tmp_idx: int 

    def __init__(self):
        self._curr_phase = 'init'
        self._input_list = []
        self._reg_list = []
        self._inst_list = []
        self._cast_list = []
        self._tmp_idx = 0

        context.active_micro = self
        self.initialize()
        context.active_micro = None 

    def initialize(self):
        self.init()
    
    def init(self):
        ...

    def create_reg_list(self, dtype: DTYPE, n: int):
        res = []
        for i in range(n):
            res.append(Reg(dtype))
        return res 

    def create_var(self, name: str, dtype: DTYPE, value: Union[Var, int, float]):
        new_var = Var(name, dtype)
        new_var.assign_module(self)
        self.append(Instruction('CREATEVAR', v=new_var, value=value))
        return new_var
    
    def append(self, inst: Instruction):
        self._inst_list.append(inst)
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            super().__setattr__(name, value)
            return 
        assert self._curr_phase=='init', 'Can only assign class variables in initialize'
        if isinstance(value, Reg):
            value.name = name
            self._reg_list.append(value)
        elif isinstance(value, MaskReg):
            value.name = name
            self._reg_list.append(value)
        elif isinstance(value, CastConfig):
            value.name = self.__class__.__name__ + f'_{name}'
            self._cast_list.append(value)
        elif isinstance(value, list):
            for idx,i in enumerate(value):
                assert isinstance(i, Reg)
                i.name = name+'_'+str(idx)
                self._reg_list.append(i)
        elif isinstance(value, (int, float)):
            ...
        else:
            raise TypeError('Not supported data type')
        super().__setattr__(name, value)
        return 
    
    def __call__(self, *args: Union[Var, Tensor, int, float]):
        g_vec = context.active_vec
        assert g_vec is not None 
        assert g_vec._curr_phase=='computing'
        # add call-micro to vec
        for a in args:
            assert isinstance(a, (Tensor, Var, float, int)), 'Micro forward only support these types'
        g_vec.append(Instruction('CALLMICRO', name=self.__class__.__name__, args=args))

        # if forwarded, just return 
        if self._curr_phase=='finished':
            return 
        
        # process new args. Avoid naming conflict 
        idx = 0
        new_args: list[Union[Var, Tensor]] = []
        for a in args:
            if isinstance(a, Tensor):
                assert a.pos==Position.UB, 'Tensor must on UB'
                new_tensor = Tensor(a.dtype, a.length)
                new_tensor.name = f'tsr{idx}'
                new_tensor.pos = Position.UB
                new_args.append(new_tensor)
                idx += 1 
            elif isinstance(a, Var):
                new_var = a.copy()
                new_var.name = f'v{idx}'
                new_var.varstr = f'v{idx}'
                new_args.append(new_var)
                idx += 1 
            elif isinstance(a, int):
                new_var = Var(f'v{idx}', DT.int)
                new_args.append(new_var)
                idx += 1 
            elif isinstance(a, float):
                new_var = Var(f'v{idx}', DT.float)
                new_args.append(new_var)
                idx += 1 
            else:
                raise TypeError(f'Micro forward only support [Tensor, Var, int, float]')
        self._input_list = new_args

        # do forward once 
        self._curr_phase = 'computing'
        context.active_micro = self
        self.forward(*new_args)
        context.active_micro = None 
        self._curr_phase = 'finished'

    def forward(self, *args: Union[Var, Tensor]):
        ...


    # codegen related 
    def gen_input_args_str(self):
        if len(self._input_list)==0:
            return ''
        
        tmp = []
        for i in self._input_list:
            if isinstance(i, Tensor):
                tmp.append(f'__ubuf__ {i.dtype}* {i.name}')
            elif isinstance(i, Var):
                tmp.append(f'const {i.dtype} &{i.name}')
        return ', '.join(tmp)

    def gen_code(self, filename: str):
        print()
        print('>> Micro Module: Trying to generate', filename, '<<')
        h = CodeHelper()
        h('#pragma once')
        h('#include "tensorutils.h"')
        h()
        # generate cast configs 
        for c in self._cast_list:
            sat = 'SAT' if c.saturate else 'NO_SAT'
            h(f'static constexpr MicroAPI::CastTrait {c.name} = {{ MicroAPI::RegLayout::{c.reg_layout}, MicroAPI::SatMode::{sat}, MicroAPI::MaskMergeMode::ZEROING, RoundMode::{c.round_mode} }};')
        h()

        h(f'__aicore__ inline void {self.__class__.__name__}({self.gen_input_args_str()}){{')
        h.ir()
        h('__VEC_SCOPE__')
        h('{')
        h.ir()

        for i in self._reg_list:
            if isinstance(i, Reg):
                h(f'MicroAPI::RegTensor<{i.dtype}> {i.name};')
            elif isinstance(i, MaskReg):
                h(f'MicroAPI::MaskReg {i.name} = MicroAPI::CreateMask<{i.dtype}, MicroAPI::MaskPattern::{i.init_mode}>();')
            else:
                raise TypeError(f'Unknown type {type(i)}')
        
        h()
        parser.parse_all(self._inst_list, h)


        h.il()
        h('}')
        h.il()
        h('}')
        h()

        # print(h.result)
        fout = open(filename, 'w')
        fout.write(h.result)
        fout.close()

