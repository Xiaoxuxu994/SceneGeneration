from ...utils import Var, DT, DATATYPE
from typing import Union 
import math 
from ... import context


def ComputeOffset(idx: list[Union[Var, int]], shape: list[Union[Var, int]]) -> Var:
    assert len(idx)==len(shape)
    offset = None 
    strides = []
    for i in range(1, len(shape)):
        tmp = None 
        for j in range(i, len(shape)):
            if tmp is None:
                tmp = shape[j]
            else:
                tmp = tmp * shape[j]
        strides.append(tmp)
    strides.append(1)

    for i,s in zip(idx, strides):
        if isinstance(i, int) and i==0:
            continue 
        if offset is None:
            offset = i*s 
        else:
            offset = offset + i*s 

    assert offset is not None 
    return offset 

# some functions 
def Max(a: Union[int, float, Var], b: Union[int, float, Var]):
    if isinstance(a, Var) and isinstance(b, Var):
        assert a.dtype==b.dtype, 'Data type must be the same'
    if isinstance(a, Var):
        dtype = a.dtype
    elif isinstance(b, Var):
        dtype = b.dtype 
    else:
        dtype = DT.int
    new_const = Var('tmp_max', dtype)
    new_const.varstr = f'(({str(a)})>({str(b)})) ? ({str(a)}) : ({str(b)})'
    return new_const

def Min(a: Union[int, float, Var], b: Union[int, float, Var]):
    if isinstance(a, Var) and isinstance(b, Var):
        assert a.dtype==b.dtype, 'Data type must be the same'
    if isinstance(a, Var):
        dtype = a.dtype
    elif isinstance(b, Var):
        dtype = b.dtype 
    else:
        dtype = DT.int
    new_const = Var('tmp_max', dtype)
    new_const.varstr = f'(({str(a)})<({str(b)})) ? ({str(a)}) : ({str(b)})'
    return new_const


def scalar_sqrt(v: Union[Var, float]):
    if isinstance(v, Var):
        assert v.dtype==DT.float
    new_const = Var('sqrt_tmp', DATATYPE.float)
    new_const.varstr = f'sqrt({str(v)})'
    if isinstance(v, Var) and v.value is not None:
        new_const.value = math.sqrt(v.value)
    elif isinstance(v, float):
        new_const.value = math.sqrt(v)
    return new_const

def Align16(a: Union[Var, int]):
    if isinstance(a, Var):
        assert a.dtype==DT.int, 'Data type must be int'
    new_const = Var('align_tmp', DATATYPE.int)
    new_const.varstr = f'Align16B({str(a)})'
    aval = None
    if isinstance(a, Var) and a.value is not None:
        aval = a.value
    else:
        aval = a 
    if aval is not None:
        new_const.value = (aval + 15) // 16 * 16  # type: ignore
    return new_const

def Align32(a: Union[Var, int]):
    if isinstance(a, Var):
        assert a.dtype==DT.int, 'Data type must be int'
    new_const = Var('align_tmp', DATATYPE.int)
    new_const.varstr = f'Align32B({str(a)})'
    aval = None
    if isinstance(a, Var) and a.value is not None:
        aval = a.value
    else:
        aval = a 
    if aval is not None:
        new_const.value = (aval + 31) // 32 * 32  # type: ignore
    return new_const

def Align64(a: Union[Var, int]):
    if isinstance(a, Var):
        assert a.dtype==DT.int, 'Data type must be int'
    new_const = Var('align_tmp', DATATYPE.int)
    new_const.varstr = f'Align64B({str(a)})'
    aval = None
    if isinstance(a, Var) and a.value is not None:
        aval = a.value
    else:
        aval = a 
    if aval is not None:
        new_const.value = (aval + 63) // 64 * 64  # type: ignore
    return new_const

def Align128(a: Union[Var, int]):
    if isinstance(a, Var):
        assert a.dtype==DT.int, 'Data type must be int'
    new_const = Var('align_tmp', DATATYPE.int)
    new_const.varstr = f'Align128B({str(a)})'
    aval = None
    if isinstance(a, Var) and a.value is not None:
        aval = a.value
    else:
        aval = a 
    if aval is not None:
        new_const.value = (aval + 127) // 128 * 128  # type: ignore
    return new_const

def Align256(a: Union[Var, int]):
    if isinstance(a, Var):
        assert a.dtype==DT.int, 'Data type must be int'
    new_const = Var('align_tmp', DATATYPE.int)
    new_const.varstr = f'Align256B({str(a)})'
    aval = None
    if isinstance(a, Var) and a.value is not None:
        aval = a.value
    else:
        aval = a 
    if aval is not None:
        new_const.value = (aval + 255) // 256 * 256  # type: ignore
    return new_const

def CeilDiv(a: Union[Var, int], b: Union[Var, int]):
    if isinstance(a, Var):
        assert a.dtype==DT.int, 'Data type must be int'
    if isinstance(b, Var):
        assert b.dtype==DT.int, 'Data type must be int'
    new_const = Var('ceildiv_tmp', DATATYPE.int)
    new_const.varstr = f'CeilDiv({str(a)},{str(b)})'
    aval = None
    bval = None 
    if isinstance(a, Var) and a.value is not None:
        aval = a.value
    else:
        aval = a 
    if isinstance(b, Var) and b.value is not None:
        bval = b.value
    else:
        bval = b
    if aval is not None and bval is not None:
        new_const.value = (aval + bval - 1) // bval  # type: ignore
    return new_const

def GetCubeNum():
    new_const = Var('BlkNum', DATATYPE.int)
    new_const.varstr = 'GetBlockNum()'
    new_const.value = context.num_cores
    return new_const

def GetVecNum():
    new_const = Var('BlkNum', DATATYPE.int)
    new_const.varstr = 'GetBlockNum()'
    new_const.value = context.num_cores
    return new_const*2

def GetCubeIdx():
    new_const = Var('BlkDim', DATATYPE.int)
    new_const.varstr = 'get_block_idx()'
    new_const.value = 0
    return new_const

def GetVecIdx():
    new_const = Var('BlkDim', DATATYPE.int)
    new_const.varstr = 'GetBlockIdx()'
    new_const.value = 0
    return new_const

def GetSubBlockIdx():
    new_const = Var('BlkDim', DATATYPE.int)
    new_const.varstr = 'get_subblockid()'
    new_const.value = 0
    return new_const