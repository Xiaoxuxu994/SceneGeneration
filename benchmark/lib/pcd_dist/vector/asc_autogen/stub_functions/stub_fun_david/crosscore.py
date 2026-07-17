from ... import context
from ...utils import Instruction, PipeInst, PIPE


def cube_ready(flag: int=0, pipe: PipeInst=PIPE.FIX):
    assert pipe in [PIPE.MTE2, PIPE.MTE1, PIPE.FIX, PIPE.M]
    if context.active_cube is not None:
        context.active_cube.append(Instruction('CUBEREADY', flag=flag, pipe=pipe))
    else:
        raise NotImplementedError
    
def wait_vec(flag: int=0):
    if context.active_cube is not None:
        context.active_cube.append(Instruction('WAITVEC', flag=flag))
    else:
        raise NotImplementedError

def vec_ready(flag: int=0, pipe: PipeInst=PIPE.MTE3):
    assert pipe in [PIPE.MTE2, PIPE.MTE3]
    if context.active_vec is not None:
        context.active_vec.append(Instruction('VECREADY', flag=flag, pipe=pipe))
    else:
        raise NotImplementedError

def wait_cube(flag: int=0):
    if context.active_vec is not None:
        context.active_vec.append(Instruction('WAITCUBE', flag=flag))
    else:
        raise NotImplementedError

def allcube_ready(flag: int=0, pipe: PipeInst=PIPE.FIX):
    assert pipe in [PIPE.MTE2, PIPE.MTE1, PIPE.FIX, PIPE.M]
    if context.active_cube is not None:
        context.active_cube.append(Instruction('ALLCUBEREADY', flag=flag, pipe=pipe))
    else:
        raise NotImplementedError
    
def allcube_wait(flag: int=0):
    if context.active_cube is not None:
        context.active_cube.append(Instruction('ALLCUBEWAIT', flag=flag))
    else:
        raise NotImplementedError

def allvec_wait(flag: int=0):
    if context.active_vec is not None:
        context.active_vec.append(Instruction('ALLVECWAIT', flag=flag))
    else:
        raise NotImplementedError

def allvec_ready(flag: int=0, pipe: PipeInst=PIPE.MTE3):
    assert pipe in [PIPE.MTE2, PIPE.MTE3]
    if context.active_vec is not None:
        context.active_vec.append(Instruction('ALLVECREADY', flag=flag, pipe=pipe))
    else:
        raise NotImplementedError





