from .datatype import DTYPE

def sizeof(dtype: DTYPE):
    assert isinstance(dtype, DTYPE)
    if dtype.ctype in ['half', 'bfloat16_t', 'uint16_t', 'int16_t']:
        return 2
    elif dtype.ctype in ['float', 'int', 'uint32_t', 'int32_t']:
        return 4 
    elif dtype.ctype in ['int8_t', 'uint8_t']:
        return 1 
    elif dtype.ctype in ['int64_t', 'uint64_t']:
        return 8
    elif dtype.ctype in ['void']:
        return 1
    else:
        raise NotImplementedError(f'Unknown dtype {dtype.ctype}')

