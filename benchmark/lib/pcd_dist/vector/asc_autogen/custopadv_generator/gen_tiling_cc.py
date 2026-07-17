import os 
import os.path as osp 
from .helper import CodeHelper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils import OpdefBase


def gen_infer_shape(opdef: 'OpdefBase', h: CodeHelper):
    h(f'static ge::graphStatus InferShape4{opdef.camelname}(gert::InferShapeContext* context) {{')
    h.ir()
    h('const gert::Shape* inputShape = context->GetInputShape(0);')
    h('gert::Shape* outputShape = context->GetOutputShape(0);')
    h('*outputShape = *inputShape;')
    h('return ge::GRAPH_SUCCESS;')
    h.il()
    h('}')
    h()


def gen_infer_data_type(opdef: 'OpdefBase', h: CodeHelper):
    h(f'static ge::graphStatus InferDataType4{opdef.camelname}(gert::InferDataTypeContext* context) {{')
    h.ir()
    h('auto inputDtype = context->GetInputDataType(0);')
    h('context->SetOutputDataType(0, inputDtype);')
    h('return ge::GRAPH_SUCCESS;')
    h.il()
    h('}')
    h()


def generate_tiling_cc(opdef: 'OpdefBase', root_path: str):
    os.makedirs(osp.join(root_path, opdef.name, 'ophost'), exist_ok=True)
    file_out = open(osp.join(root_path, opdef.name, 'ophost', f'{opdef.name}.cc'), 'w')

    h = CodeHelper()
    h('#include <graph/utils/type_utils.h>')
    h('#include <register/op_impl_registry.h>')
    h('#include "log/ops_log.h"')
    h()
    h('using namespace ge;')
    h()
    h('namespace ops {')
    h()
    gen_infer_shape(opdef, h)
    gen_infer_data_type(opdef, h)
    h(f'IMPL_OP_INFERSHAPE({opdef.camelname})')
    h.ir()
    h(f'.InferShape(InferShape4{opdef.camelname})')
    h(f'.InferDataType(InferDataType4{opdef.camelname});')
    h.il()
    h()
    h('}')

    file_out.write(h.code)
    file_out.close()

