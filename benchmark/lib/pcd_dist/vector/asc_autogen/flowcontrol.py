from .utils import Var, Instruction, DATATYPE
from typing import Union 
from . import context 


class Loop():
    def __init__(self, name: str, start: Union[int, 'Var'], end: Union[int, 'Var'], step: Union[int, 'Var']=1) -> None:
        self.name = name 
        self.start = start
        self.end = end 
        self.step = step 
    
    def __enter__(self):
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_micro = context.active_micro
        g_kernel = context.active_kernel
        # micro should be considered first 
        if g_micro is not None:
            g_micro.append(Instruction('STARTLOOP', name=self.name, start=self.start, end=self.end, step=self.step))
            loop_const = Var(self.name, DATATYPE.int)
            loop_const.assign_module(g_micro)
            return loop_const
        elif g_cube is not None:
            g_cube.append(Instruction('STARTLOOP', name=self.name, start=self.start, end=self.end, step=self.step))
            loop_const = Var(self.name, DATATYPE.int)
            loop_const.assign_module(g_cube)
            return loop_const
        elif g_vec is not None:
            g_vec.append(Instruction('STARTLOOP', name=self.name, start=self.start, end=self.end, step=self.step))
            loop_const = Var(self.name, DATATYPE.int)
            loop_const.assign_module(g_vec)
            return loop_const
        elif g_kernel is not None:
            g_kernel.append(Instruction('STARTLOOP', name=self.name, start=self.start, end=self.end, step=self.step))
            loop_const = Var(self.name, DATATYPE.int)
            loop_const.assign_module(g_kernel)
            return loop_const
        else:
            raise Exception('Loop must be called in VecModule or CubeModule or MicroModule')

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            # raise exec_type(exec_val)
            return False
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_micro = context.active_micro
        g_kernel = context.active_kernel
        if g_micro is not None:
            g_micro.append(Instruction('ENDLOOP'))
            return True 
        elif g_cube is not None:
            g_cube.append(Instruction('ENDLOOP'))
            return True
        elif g_vec is not None:
            g_vec.append(Instruction('ENDLOOP'))
            return True
        elif g_kernel is not None:
            g_kernel.append(Instruction('ENDLOOP'))
            return True 
        else:
            raise Exception('Loop must be called in VecModule or CubeModule or MicroModule')


class If():
    def __init__(self, cond: Union[int, 'Var']) -> None:
        self.cond = cond 
    
    def __enter__(self):
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_micro = context.active_micro
        g_kernel = context.active_kernel
        if g_micro is not None:
            g_micro.append(Instruction('STARTIF', cond=self.cond))
        elif g_cube is not None:
            g_cube.append(Instruction('STARTIF', cond=self.cond))
        elif g_vec is not None:
            g_vec.append(Instruction('STARTIF', cond=self.cond))
        elif g_kernel is not None:
            g_kernel.append(Instruction('STARTIF', cond=self.cond))
        else:
            raise Exception('If must be called in VecModule or CubeModule')

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            # raise exec_type(exec_val)
            return False
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_micro = context.active_micro
        g_kernel = context.active_kernel
        if g_micro is not None:
            g_micro.append(Instruction('ENDIF'))
            return True
        elif g_cube is not None:
            g_cube.append(Instruction('ENDIF'))
            return True
        elif g_vec is not None:
            g_vec.append(Instruction('ENDIF'))
            return True
        elif g_kernel is not None:
            g_kernel.append(Instruction('ENDIF'))
            return True 
        else:
            raise Exception('If must be called in VecModule or CubeModule')


class Elif():
    def __init__(self, cond: Union[int, 'Var']) -> None:
        self.cond = cond 
    
    def __enter__(self):
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_micro = context.active_micro
        g_kernel = context.active_kernel
        if g_micro is not None:
            g_micro.append(Instruction('STARTELIF', cond=self.cond))
        elif g_cube is not None:
            g_cube.append(Instruction('STARTELIF', cond=self.cond))
        elif g_vec is not None:
            g_vec.append(Instruction('STARTELIF', cond=self.cond))
        elif g_kernel is not None:
            g_kernel.append(Instruction('STARTELIF', cond=self.cond))
        else:
            raise Exception('Elif must be called in VecModule or CubeModule')

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            # raise exec_type(exec_val)
            return False
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_micro = context.active_micro
        g_kernel = context.active_kernel
        if g_micro is not None:
            g_micro.append(Instruction('ENDIF'))
            return True
        elif g_cube is not None:
            g_cube.append(Instruction('ENDIF'))
            return True
        elif g_vec is not None:
            g_vec.append(Instruction('ENDIF'))
            return True
        elif g_kernel is not None:
            g_kernel.append(Instruction('ENDIF'))
        else:
            raise Exception('Elif must be called in VecModule or CubeModule')


class Else():
    def __enter__(self):
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_micro = context.active_micro
        g_kernel = context.active_kernel
        if g_micro is not None:
            g_micro.append(Instruction('STARTELSE'))
        elif g_cube is not None:
            g_cube.append(Instruction('STARTELSE'))
        elif g_vec is not None:
            g_vec.append(Instruction('STARTELSE'))
        elif g_kernel is not None:
            g_kernel.append(Instruction('STARTELSE'))
        else:
            raise Exception('Elif must be called in VecModule or CubeModule')

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            # raise exec_type(exec_val)
            return False
        g_vec = context.active_vec
        g_cube = context.active_cube
        g_micro = context.active_micro
        g_kernel = context.active_kernel
        if g_micro is not None:
            g_micro.append(Instruction('ENDIF'))
            return True
        elif g_cube is not None:
            g_cube.append(Instruction('ENDIF'))
            return True
        elif g_vec is not None:
            g_vec.append(Instruction('ENDIF'))
            return True
        elif g_kernel is not None:
            g_kernel.append(Instruction('ENDIF'))
        else:
            raise Exception('Elif must be called in VecModule or CubeModule')
