from typing import TYPE_CHECKING, Optional, Literal


if TYPE_CHECKING:
    from .utils import Pos
    from .kernel import KernelBase
    from .vec import VecModule
    from .cube import CubeModule
    from .micro import MicroModule

active_vec: Optional['VecModule'] = None
active_cube: Optional['CubeModule'] = None
active_kernel: Optional['KernelBase'] = None
active_micro: Optional['MicroModule'] = None

num_cores: int = 20
L1_size = 512*1024
L0A_size = 64*1024
L0B_size = 64*1024
L0C_size = 128*1024
UB_size = 192*1024
device_type = '910b3'

def get_max_size(pos: 'Pos'):
    from .utils import Position
    if pos==Position.UB:
        return UB_size
    if pos==Position.L1:
        return L1_size
    if pos==Position.L0A:
        return L0A_size
    if pos==Position.L0B:
        return L0B_size
    if pos==Position.L0C:
        return L0C_size
    return 0

def set_core_num(n: int):
    global num_cores
    num_cores = n

def set_device_type(device_type_i: Literal['david', '910b4', '910b3', '910b2', '910b1']):
    global num_cores, L1_size, L0A_size, L0B_size, L0C_size, UB_size, device_type
    if device_type_i=='david':
        num_cores = 32
        L1_size = 512*1024
        L0A_size = 64*1024
        L0B_size = 64*1024
        L0C_size = 256*1024
        UB_size = 256*1024
        device_type = device_type_i
    elif device_type_i=='910b1' or device_type_i=='910b2':
        num_cores = 24
        L1_size = 512*1024
        L0A_size = 64*1024
        L0B_size = 64*1024
        L0C_size = 128*1024
        UB_size = 192*1024
        device_type = device_type_i
    elif device_type_i=='910b4' or device_type_i=='910b3':
        num_cores = 20
        L1_size = 512*1024
        L0A_size = 64*1024
        L0B_size = 64*1024
        L0C_size = 128*1024
        UB_size = 192*1024
        device_type = device_type_i
    else:
        raise NotImplementedError
    
    