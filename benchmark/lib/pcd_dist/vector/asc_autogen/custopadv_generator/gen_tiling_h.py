import os 
import os.path as osp 
from .helper import CodeHelper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils import OpdefBase



def gen_tiling_struct(opdef: 'OpdefBase', h: CodeHelper):
    tiling = opdef.tiling
    h(f'BEGIN_TILING_DATA_DEF({opdef.camelname}TilingData)')
    h.ir()
    for name, dtype in tiling.fields:
        h(f'TILING_DATA_FIELD_DEF({dtype}, {name})')
    h.il()
    h('END_TILING_DATA_DEF;')
    h()
    h(f'REGISTER_TILING_DATA_CLASS({opdef.camelname}, {opdef.camelname}TilingData)')
    h()


def gen_compile_info(opdef: 'OpdefBase', h: CodeHelper):
    h(f'struct {opdef.camelname}CompileInfo{{')
    h.ir()
    h('uint32_t aicNum;')
    h('uint32_t aivNum;')
    h('uint64_t ubSize;')
    h('uint64_t l1Size;')
    h('uint64_t l2Size;')
    h('uint64_t l0CSize;')
    h('uint64_t l0ASize;')
    h('uint64_t l0BSize;')
    h('platform_ascendc::SocVersion socVersion;')
    h.il()
    h('};')
    h()


def gen_class_def(opdef: 'OpdefBase', h: CodeHelper):
    h(f'class {opdef.camelname}Tiling: public TilingBaseClass {{')
    h('public:')
    h.ir()
    h(f'explicit {opdef.camelname}Tiling(gert::TilingContext *context): TilingBaseClass(context){{')
    h('    Reset();')
    h('}')
    h(f'~{opdef.camelname}Tiling() override = default;')
    h('void Reset(gert::TilingContext *context) override {')
    h.ir()
    h('TilingBaseClass::Reset(context);')
    h('Reset();')
    h.il()
    h('}')
    h.il()

    h()
    h('protected:')
    h.ir()
    h('bool IsCapable() override { return true; }')
    h('ge::graphStatus GetPlatformInfo() override { return ge::GRAPH_SUCCESS; }')
    h('ge::graphStatus GetShapeAttrsInfo() override { return ge::GRAPH_SUCCESS; }')
    h('void Reset() { tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData()); }')
    h('uint64_t GetTilingKey() const override { return 10000000; }')
    h('ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }')
    h()
    h('ge::graphStatus DoOpTiling() override;')
    h('ge::graphStatus GetWorkspaceSize() override;')
    h('ge::graphStatus PostTiling() override;')
    h()
    h.il()
    h('private:')
    h.ir()
    h('const gert::Shape GetShapeOfInput(const size_t index) { return context_->GetInputShape(index)->GetStorageShape(); }')
    
    # h('bool PreparePlatformInfo();')
    h(f'{opdef.camelname}TilingData tilingData_;')
    h.il()

    h('};')
    h()


def generate_tiling_h(opdef: 'OpdefBase', root_path: str):
    os.makedirs(osp.join(root_path, opdef.name, 'ophost'), exist_ok=True)
    file_out = open(osp.join(root_path, opdef.name, 'ophost', f'{opdef.name}_tiling.h'), 'w')
    
    h = CodeHelper()
    h(f'#ifndef {opdef.name.upper()}_TILING_H')
    h(f'#define {opdef.name.upper()}_TILING_H')
    h()
    h('#include <cstddef>')
    h('#include <cstdint>')
    h('#include <sys/types.h>')
    h('#include <utility>')
    h()

    h('#include <exe_graph/runtime/tiling_context.h>')
    h('#include <graph/utils/type_utils.h>')
    h('#include <register/tilingdata_base.h>')
    h('#include <register/op_def_registry.h>')
    h('#include <tiling/tiling_api.h>')
    h('#include "tiling/tiling_base.h"')
    h()
    
    h('namespace optiling {')
    h()
    # h.ir()

    gen_tiling_struct(opdef, h)
    gen_compile_info(opdef, h)
    gen_class_def(opdef, h)

    # h.il()
    h('}')
    h('#endif')

    file_out.write(h.code)
    file_out.close()
