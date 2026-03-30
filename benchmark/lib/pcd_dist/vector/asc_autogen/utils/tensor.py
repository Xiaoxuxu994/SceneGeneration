from .datatype import DTYPE
from .position import Pos
from .utilfuncs import sizeof
from .instruction import Instruction
from .. import context
from typing import Optional, Union, TYPE_CHECKING
from .var import Var
    

class GMTensor():
    name: str 
    dtype: DTYPE 
    length: Optional[Union[int, Var]]
    is_output: bool

    def __init__(self, name: str, dtype: DTYPE, length: Optional[Union[int, Var]]=None, is_output: bool=False):
        self.name = name 
        self.dtype = dtype 
        self.length = length
        self.is_output = is_output

    def __getitem__(self, idx: Union[int, Var]):
        new_tensor = GMTensor(f'{self.name}[{str(idx)}]', self.dtype)
        return new_tensor
    
    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'[GMTensor] name={self.name} dtype={self.dtype}'

    def load_bin(self, file: str):
        ...


class Tensor():
    dtype: DTYPE 
    length: Union[int, Var]
    name: str 
    pos: Optional[Pos] 

    def __init__(self, dtype: DTYPE, length: Union[int, Var]):
        self.dtype = dtype 
        self.length = length
        self.name = ''
        self.pos = None 
    
    def reinterpret(self, dtype: DTYPE):
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_micro = context.active_micro
        curr_mod = g_micro or g_vec or g_cube
        assert curr_mod is not None
        new_tsr = Tensor(dtype, self.length*sizeof(self.dtype)//sizeof(dtype))
        new_tsr.pos = self.pos
        idx = curr_mod._tmp_idx
        curr_mod._tmp_idx += 1 
        new_tsr.name = 'tmptsr_%d'%idx
        curr_mod.append(Instruction('REINTERPRET', dst=new_tsr, src=self))
        return new_tsr

    def __getitem__(self, idx: Union[int, Var]):
        new_tensor = Tensor(self.dtype, self.length)
        # new_tensor.name = self.name + f'[{str(idx)}]'
        if context.active_micro is not None:
            new_tensor.name = self.name + f' + {str(idx)}'
        else:
            new_tensor.name = self.name + f'[{str(idx)}]'
        new_tensor.pos = self.pos
        return new_tensor

    def __str__(self) -> str:
        return self.name 
    
    def __repr__(self) -> str:
        return f'[Tensor] name={self.name} dtype={self.dtype} length={str(self.length)} pos={self.pos}'


class DBuff():
    dtype: DTYPE 
    length: Union[int, Var]
    name: str 
    pos: Optional[Pos]

    def __init__(self, dtype: DTYPE, length: Union[int, Var]):
        self.dtype = dtype 
        self.length = length
        self.name = ''
        self.pos = None

    def get(self, i: Union[int, Var]):
        new_tensor = Tensor(self.dtype, self.length)
        new_tensor.name = self.name + f'.get({str(i)})'
        new_tensor.pos = self.pos
        return new_tensor
    
    def __getitem__(self, idx: Union[int, Var]):
        return self.get(idx)

    def __str__(self) -> str:
        return self.name 
    
    def __repr__(self) -> str:
        return f'[DBuff] name={self.name} dtype={self.dtype} length={str(self.length)}'


class TBuff():
    dtype: DTYPE 
    length: Union[int, Var]
    name: str 
    pos: Optional[Pos]

    def __init__(self, dtype: DTYPE, length: Union[int, Var]):
        self.dtype = dtype 
        self.length = length
        self.name = ''
        self.pos = None 
        
    def get(self, i: Union[int, Var]):
        new_tensor = Tensor(self.dtype, self.length)
        new_tensor.name = self.name + f'.get({str(i)})'
        new_tensor.pos = self.pos
        return new_tensor
    
    def __getitem__(self, idx: Union[int, Var]):
        return self.get(idx)

    def __str__(self) -> str:
        return self.name 
    
    def __repr__(self) -> str:
        return f'[TBuff] name={self.name} dtype={self.dtype} length={str(self.length)}'


class QBuff():
    dtype: DTYPE 
    length: Union[int, Var]
    name: str 
    pos: Optional[Pos]

    def __init__(self, dtype: DTYPE, length: Union[int, Var]):
        self.dtype = dtype 
        self.length = length
        self.name = ''
        self.pos = None 
        
    def get(self, i: Union[int, Var]):
        new_tensor = Tensor(self.dtype, self.length)
        new_tensor.name = self.name + f'.get({str(i)})'
        new_tensor.pos = self.pos
        return new_tensor
    
    def __getitem__(self, idx: Union[int, Var]):
        return self.get(idx)

    def __str__(self) -> str:
        return self.name 
    
    def __repr__(self) -> str:
        return f'[QBuff] name={self.name} dtype={self.dtype} length={str(self.length)}'
    

class PBuff():
    dtype: DTYPE 
    length: Union[int, Var]
    name: str 
    pos: Optional[Pos]

    def __init__(self, dtype: DTYPE, length: Union[int, Var]):
        self.dtype = dtype 
        self.length = length
        self.name = ''
        self.pos = None 
        
    def get(self, i: Union[int, Var]):
        new_tensor = Tensor(self.dtype, self.length)
        new_tensor.name = self.name + f'.get({str(i)})'
        new_tensor.pos = self.pos
        return new_tensor
    
    def __getitem__(self, idx: Union[int, Var]):
        return self.get(idx)

    def __str__(self) -> str:
        return self.name 
    
    def __repr__(self) -> str:
        return f'[PBuff] name={self.name} dtype={self.dtype} length={str(self.length)}'
    
