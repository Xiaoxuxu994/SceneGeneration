OPDEF_TENSOR_DT_MAPPING: dict[str, str] = {
    'half'        : 'ge::DT_FLOAT16',
    'float'       : 'ge::DT_FLOAT',
    'bfloat16_t'  : 'ge::DT_BF16',
    'int32_t'     : 'ge::DT_INT32',
    'int'         : 'ge::DT_INT32',
    'int64_t'     : 'ge::DT_INT64',
    'uint32_t'    : 'ge::DT_UINT32',
    'uint64_t'    : 'ge::DT_UINT64',
    'void'        : 'ge::DT_INT32'
}

OPDEF_ATTR_DT_MAPPING: dict[str, str] = {
    'int32_t'  : 'int32_t',
    'uint32_t' : 'uint32_t',
    'int'      : 'int32_t',
    'int64_t'  : 'int64_t',
    'uint64_t' : 'uint64_t',
    'float'    : 'float',
}

ASCENDEBUG_TENSOR_DT_MAPPING: dict[str, str] = {
    'bfloat16_t'     : 'bfloat16',
    'half'           : 'float16',
    'float'          : 'float32',
    'int64_t'        : 'int64',
    'uint64_t'       : 'uint64',
    'int32_t'        : 'int32',
    'uint32_t'       : 'uint32',
    'void'           : 'int32',
}

ASCENDEBUG_ATTR_DT_MAPPING: dict[str, str] = {
    'float'          : 'float',
    'int'            : 'int',
    'int32_t'        : 'int',
}



class DTYPE:
    def __init__(self, ctype: str):
        self.ctype = ctype 

    def __eq__(self, other: 'DTYPE'):
        if self.ctype==other.ctype:
            return True 
        return False
    
    def __str__(self):
        return self.ctype

    @property
    def opdef_attr(self):
        return OPDEF_ATTR_DT_MAPPING[self.ctype]
    
    @property
    def opdef_tsr(self):
        return OPDEF_TENSOR_DT_MAPPING[self.ctype]

    @property
    def ascendebug_tsr(self):
        return ASCENDEBUG_TENSOR_DT_MAPPING[self.ctype]

    @property
    def ascendebug_attr(self):
        return ASCENDEBUG_ATTR_DT_MAPPING[self.ctype]


class DATATYPE:
    half = DTYPE('half')
    fp16 = DTYPE('half')
    float = DTYPE('float')
    fp32 = DTYPE('float')
    bfloat16 = DTYPE('bfloat16_t')
    bf16 = DTYPE('bfloat16_t')
    int = DTYPE('int')
    uint = DTYPE('uint32_t')
    int32 = DTYPE('int32_t')
    uint32 = DTYPE('uint32_t')
    int16 = DTYPE('int16_t')
    uint16 = DTYPE('uint16_t')
    int8 = DTYPE('int8_t')
    uint8 = DTYPE('uint8_t')
    int64 = DTYPE('int64_t')
    uint64 = DTYPE('uint64_t')
    int4 = DTYPE('void')

