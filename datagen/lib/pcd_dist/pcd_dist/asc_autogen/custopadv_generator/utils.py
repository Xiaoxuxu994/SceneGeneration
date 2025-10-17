from .gen_tiling_def import generate_tiling_def
from .gen_tiling_h import generate_tiling_h
from .gen_tiling_cpp import generate_tiling_cpp
from .gen_tiling_cc import generate_tiling_cc
from .gen_cmakelist import generate_cmakelist


class OPDEF_TENSOR_DT:
    fp16 = 'ge::DT_FLOAT16'
    bf16 = 'ge::DT_BF16'
    fp32 = 'ge::DT_FLOAT'
    int32 = 'ge::DT_INT32'
    int64 = 'ge::DT_INT64'
    uint32 = 'ge::DT_UINT32'
    uint64 = 'ge::DT_UINT64'

class OPDEF_ATTR_DT:
    int = 'int32_t'
    uint = 'uint32_t'
    int64 = 'int64_t'
    uint64 = 'uint64_t'
    float = 'float'


class OPInput():
    def __init__(self, name: str, shape: list[int], dtypes: list[str], is_required: bool=True):
        self.shape = shape 
        self.dtypes = dtypes
        self.name = name 
        self.is_required = is_required


class OPOutput():
    def __init__(self, name: str, shape: list[int], dtypes: list[str], is_required: bool=True):
        self.shape = shape 
        self.dtypes = dtypes
        self.name = name 
        self.is_required = is_required


class OpAttr():
    def __init__(self, name: str, dtype: str):
        ...


def snake_to_upper_camel(snake_str: str) -> str:
    words = [word for word in snake_str.split('_') if word]
    return ''.join(word.capitalize() for word in words)


class TilingBase():
    fields: list[list[str]]
    
    def __init__(self):
        self.fields = []
    
    def add_field(self, name: str, dtype: str):
        self.fields.append([name, dtype])


class OpdefBase():
    def __init__(self, name: str, inputs: list[OPInput], outputs: list[OPOutput], tiling: TilingBase):
        self.name = name
        self.camelname = snake_to_upper_camel(name)
        self.inputs = inputs 
        self.outputs = outputs 
        self.tiling = tiling
    
    def gen_op_project(self, root_path: str):
        generate_tiling_def(self, root_path)
        generate_tiling_h(self, root_path)
        generate_tiling_cpp(self, root_path)
        generate_tiling_cc(self, root_path)
        generate_cmakelist(self, root_path)

