import os 
import os.path as osp 
from .helper import CodeHelper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils import OpdefBase


def gen_inputs(opdef: 'OpdefBase', h: CodeHelper):
    for i in opdef.inputs:
        h(f'this->Input("{i.name}")')
        h.ir()
        h(f'.ParamType({"REQUIRED" if i.is_required else "OPTIONAL"})')
        dtypestr = ', '.join(i.dtypes)
        h(f'.DataType({{{dtypestr}}})')
        formatstr = ', '.join(['ge::FORMAT_ND']*len(i.dtypes))
        h(f'.Format({{{formatstr}}})')
        h(f'.UnknownShapeFormat({{{formatstr}}})')
        h(f'.AutoContiguous();')
        h.il()
        h()


def gen_outputs(opdef: 'OpdefBase', h: CodeHelper):
    for i in opdef.outputs:
        h(f'this->Output("{i.name}")')
        h.ir()
        h(f'.ParamType({"REQUIRED" if i.is_required else "OPTIONAL"})')
        dtypestr = ', '.join(i.dtypes)
        h(f'.DataType({{{dtypestr}}})')
        formatstr = ', '.join(['ge::FORMAT_ND']*len(i.dtypes))
        h(f'.Format({{{formatstr}}})')
        h(f'.UnknownShapeFormat({{{formatstr}}})')
        h(f'.InitValue(0);')
        h.il()
        h()


def gen_aicore_config(h: CodeHelper):
    h('OpAICoreConfig aicore_config;')
    h('aicore_config.DynamicCompileStaticFlag(true)')
    h.ir()
    h('.DynamicFormatFlag(true)')
    h('.DynamicRankSupportFlag(true)')
    h('.DynamicShapeSupportFlag(true)')
    h('.NeedCheckSupportFlag(false)')
    h('.PrecisionReduceFlag(true)')
    h('.ExtendCfgInfo("prebuildPattern.value", "Opaque")')
    h('.ExtendCfgInfo("coreType.value", "AiCore")')
    h('.ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");')
    h.il()
    h()
    h('this->AICore().AddConfig("ascend910_95", aicore_config);')


def gen_tiling_class(opdef: 'OpdefBase', h: CodeHelper):
    h('namespace ops {')
    h(f'class {opdef.camelname} : public OpDef {{')
    h('public:')
    h.ir()
    h(f'explicit {opdef.camelname}(const char* name) : OpDef(name)')
    h('{')
    h.ir()

    gen_inputs(opdef, h)
    gen_outputs(opdef, h)
    gen_aicore_config(h)

    h.il()
    h('}')
    h.il()
    h('};')
    h()
    h(f'OP_ADD({opdef.camelname});')
    h('}')


def generate_tiling_def(opdef: 'OpdefBase', root_path: str):
    os.makedirs(osp.join(root_path, opdef.name, 'ophost'), exist_ok=True)
    file_out = open(osp.join(root_path, opdef.name, 'ophost', f'{opdef.name}_def.cpp'), 'w')

    helper = CodeHelper()
    helper('#include "register/op_def_registry.h"')
    helper()

    gen_tiling_class(opdef, helper)

    file_out.write(helper.code)
    file_out.close()

    