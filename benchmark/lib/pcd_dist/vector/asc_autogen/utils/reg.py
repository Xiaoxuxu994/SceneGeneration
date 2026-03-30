from .datatype import DTYPE


class MaskTypeInst:
    def __init__(self, mode: str):
        self.mode = mode 
    
    def __str__(self):
        return self.mode

    def __eq__(self, other: 'MaskTypeInst'):
        return self.mode==other.mode


class MaskType:
    ALL = MaskTypeInst('ALL')
    LOWEST1 = MaskTypeInst('VL1')
    LOWEST2 = MaskTypeInst('VL2')
    LOWEST3 = MaskTypeInst('VL3')
    LOWEST4 = MaskTypeInst('VL4')
    LOWEST8 = MaskTypeInst('VL8')
    LOWEST16 = MaskTypeInst('VL16')
    LOWEST32 = MaskTypeInst('VL32')
    LOWEST128 = MaskTypeInst('VL128')
    MULTI3 = MaskTypeInst('M3')
    MULTI4 = MaskTypeInst('M4')
    LOWHALF = MaskTypeInst('H')
    LOWQUAT = MaskTypeInst('Q')
    NONE = MaskTypeInst('ALLF')


class Reg():
    dtype: DTYPE
    name: str
    
    def __init__(self, dtype: DTYPE):
        self.dtype = dtype 
        self.name = ''

    def __str__(self):
        return self.name 
    
class MaskReg():
    dtype: DTYPE
    name: str
    init_mode: MaskTypeInst 

    def __init__(self, dtype: DTYPE, init_mode: MaskTypeInst=MaskType.ALL):
        self.dtype = dtype 
        self.init_mode = init_mode
        self.name = ''

    def __str__(self):
        return self.name 



