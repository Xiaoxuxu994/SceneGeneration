import os 
import os.path as osp 
from .helper import CodeHelper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils import OpdefBase


def gen_op_tiling_func(opdef: 'OpdefBase', h: CodeHelper):
    h(f'ge::graphStatus {opdef.camelname}Tiling::DoOpTiling(){{')
    h.ir()
    h('// Should modify this function to set your tiling information')
    h('auto platformInfoPtr = context_->GetPlatformInfo();')
    h('auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);')
    h('auto aicNum_ = ascendcPlatform.GetCoreNumAic();')
    h('auto aivNum_ = ascendcPlatform.GetCoreNumAiv();')
    h()
    h('// // Hint on how to get shape info')
    h('// const auto& outGradShape = GetShapeOfInput(INDEXZERO);')
    h('// auto nBlocks = static_cast<uint32_t>(outGradShape.GetDim(INDEXZERO));')
    h('// auto nHeads = static_cast<uint32_t>(outGradShape.GetDim(INDEXONE));')
    h('// auto dimHead = static_cast<uint32_t>(outGradShape.GetDim(INDEXTWO));')
    h()
    h('// // Hint on how to get optional shape info')
    h('// const auto& actSeqLenShape = context_->GetOptionalInputShape(INDEXTHREE);')
    h('// if (actSeqLenShape != nullptr) {')
    h('//     batchSize = static_cast<uint32_t>(actSeqLenShape->GetStorageShape().GetDim(INDEXZERO));')
    h('// }')
    h()
    h('// // Hint on how to get attribute info')
    h('// auto attrs = context_->GetAttrs();')
    h('// OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);')
    h('// const auto* blockSizePtr = attrs->GetAttrPointer<uint32_t>(INDEXZERO);')
    h('// OPS_LOG_E_IF_NULL(context_, blockSizePtr, return ge::GRAPH_FAILED);')
    h('// uint32_t blockSize = *blockSizePtr;')
    h()
    h('// // Hint on how to set tiling info')
    h('// tilingData_.set_batchSize(batchSize);')
    h()
    h('// // Shortcut to set tiling info from attr')
    h('// SET_ATTR_TO_TILING(M, int32_t, 0);')
    h()
    if len(opdef.tiling.fields)>0:
        h('// Auto generated tiling assignment')
        h('auto attrs = context_->GetAttrs();')
        for i in range(len(opdef.tiling.fields)):
            h(f'SET_ATTR_TO_TILING({opdef.tiling.fields[i][0]}, {opdef.tiling.fields[i][1]}, {i});')
        h()
    h('return ge::GRAPH_SUCCESS;')
    h.il()
    h('}')
    h()


def gen_workspace_size(opdef: 'OpdefBase', h: CodeHelper):
    h(f'ge::graphStatus {opdef.camelname}Tiling::GetWorkspaceSize() {{')
    h.ir()
    h('size_t *workspaces = context_->GetWorkspaceSizes(1);')
    h('size_t sysWorkspaceSize = static_cast<uint64_t>(16) * 1024 * 1024;')
    h('size_t usrWorkspaceSize = 256 * 1024 * 1024;  // Modify this to fit your workspace size. Default to 256MB')
    h('workspaces[0] = sysWorkspaceSize + usrWorkspaceSize;')
    h('return ge::GRAPH_SUCCESS;')
    h.il()
    h('}')
    h()


def gen_post_tiling(opdef: 'OpdefBase', h: CodeHelper):
    h(f'ge::graphStatus {opdef.camelname}Tiling::PostTiling(){{')
    h.ir()
    h('auto platformInfoPtr = context_->GetPlatformInfo();')
    h('auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);')
    h('uint32_t aicNum_ = ascendcPlatform.GetCoreNumAic();')
    h('context_->SetBlockDim(aicNum_);')
    h('context_->SetTilingKey(GetTilingKey());')
    h('tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());')
    h('context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());')
    h('return ge::GRAPH_SUCCESS;')
    h.il()
    h('}')
    h()


def gen_tiling_prepare(opdef: 'OpdefBase', h: CodeHelper):
    h(f'ge::graphStatus TilingPrepareFor{opdef.camelname}(gert::TilingParseContext* context){{')
    h.ir()
    h('auto platformInfoPtr = context->GetPlatformInfo();')
    h('OPS_LOG_E_IF_NULL(context, platformInfoPtr, return ge::GRAPH_FAILED);')
    h(f'auto compileInfoPtr = context->GetCompiledInfo<{opdef.camelname}CompileInfo>();')
    h('OPS_LOG_E_IF_NULL(context, compileInfoPtr, return ge::GRAPH_FAILED);')
    h('auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);')
    h('compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();')
    h('compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();')
    h('compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();')
    h('ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);')
    h('ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);')
    h('ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);')
    h('ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);')
    h('ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);')
    h('ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);')
    h('return ge::GRAPH_SUCCESS;')
    h.il()
    h('}')
    h()


def generate_tiling_cpp(opdef: 'OpdefBase', root_path: str):
    os.makedirs(osp.join(root_path, opdef.name, 'ophost'), exist_ok=True)
    file_out = open(osp.join(root_path, opdef.name, 'ophost', f'{opdef.name}_tiling.cpp'), 'w')

    h = CodeHelper()

    h(f'#include "{opdef.name}_tiling.h"')
    h('#include <queue>')
    h('#include <cmath>')
    h('#include <cfloat>')
    h('#include <register/op_impl_registry.h>')
    h('#include "tiling/tiling_templates_registry.h"')
    h()
    h('#define SET_ATTR_TO_TILING(name, dtype, idx) const auto* name##_ptr = attrs->GetAttrPointer<int32_t>(idx); \\')
    h('                                   dtype name = *name##_ptr;  tilingData_.set_##name(name);')
    h()

    h('namespace optiling {')
    h()
    # h.ir()
    gen_op_tiling_func(opdef, h)
    gen_workspace_size(opdef, h)
    gen_post_tiling(opdef, h)
    h(f'REGISTER_TILING_TEMPLATE("{opdef.camelname}", {opdef.camelname}Tiling, 0);')
    h()
    h(f'ASCENDC_EXTERN_C ge::graphStatus Tiling{opdef.camelname}(gert::TilingContext* context) {{')
    h.ir()
    h(f'{opdef.camelname}Tiling tiling(context);')
    h('return tiling.DoTiling();')
    h.il()
    h('}')
    h()
    gen_tiling_prepare(opdef, h)

    h(f'IMPL_OP_OPTILING({opdef.camelname})')
    h.ir()
    h(f'.Tiling(Tiling{opdef.camelname})')
    h(f'.TilingParse<{opdef.camelname}CompileInfo>(TilingPrepareFor{opdef.camelname});')
    h.il()

    # h.il()
    h()
    h('}')

    file_out.write(h.code)
    file_out.close()
