from typing import Union, TYPE_CHECKING
from .pipe import PipeInst
from .instruction import Instruction

if TYPE_CHECKING:
    from ..cube import CubeModule
    from ..vec import VecModule


class SEvent():
    def __init__(self, src: PipeInst, dst: PipeInst):
        self.src = src 
        self.dst = dst 
        
    def set_module(self, module: Union['CubeModule', 'VecModule']):
        self.module = module 

    def set_name(self, name: str):
        self.name = name 

    def set_id(self, idd: int):
        self.idd = idd 

    def set(self):
        self.module.append(Instruction('EVENTSET', name=self.name))
    
    def wait(self):
        self.module.append(Instruction('EVENTWAIT', name=self.name))
    
    def setall(self):
        self.module.append(Instruction('EVENTSETALL', name=self.name))
    
    def release(self):
        self.module.append(Instruction('EVENTRELEASE', name=self.name))

    def __str__(self):
        return self.name 

    def __repr__(self):
        return f'[SEvent]  name:{self.name}  src:{self.src}  dst:{self.dst}  id:{self.idd}'


class DEvent():
    def __init__(self, src: PipeInst, dst: PipeInst):
        self.src = src 
        self.dst = dst 
    
    def set_module(self, module: Union['CubeModule', 'VecModule']):
        self.module = module 

    def set_name(self, name: str):
        self.name = name 

    def set_id(self, idd: int):
        self.idd = idd 

    def set(self):
        self.module.append(Instruction('EVENTSET', name=self.name))
    
    def wait(self):
        self.module.append(Instruction('EVENTWAIT', name=self.name))
    
    def setall(self):
        self.module.append(Instruction('EVENTSETALL', name=self.name))
    
    def release(self):
        self.module.append(Instruction('EVENTRELEASE', name=self.name))

    def __str__(self):
        return self.name 

    def __repr__(self):
        return f'[DEvent]  name:{self.name}  src:{self.src}  dst:{self.dst}  id:{self.idd}'
