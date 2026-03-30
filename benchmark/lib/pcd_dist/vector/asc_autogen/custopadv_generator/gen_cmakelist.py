import os 
import os.path as osp 
from .helper import CodeHelper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils import OpdefBase


def generate_cmakelist(opdef: 'OpdefBase', root_path: str):
    os.makedirs(osp.join(root_path, opdef.name, 'ophost'), exist_ok=True)
    file_out = open(osp.join(root_path, opdef.name, 'ophost', 'CMakeLists.txt'), 'w')

    h = CodeHelper()
    h('add_ops_compile_options(')
    h.ir()
    h(f'OP_NAME {opdef.camelname}')
    h('OPTIONS --cce-auto-sync')
    h.ir()
    h('-Wno-deprecated-declarations')
    h('-Werror')
    h.il()
    h.il()
    h(')')
    h()

    h('target_sources(op_host_aclnn PRIVATE')
    h.ir()
    h(f'{opdef.name}_def.cpp')
    h.il()
    h(')')
    h()

    h('target_sources(optiling PRIVATE')
    h.ir()
    h(f'{opdef.name}_tiling.cpp')
    h.il()
    h(')')
    h()

    h('if (NOT BUILD_OPEN_PROJECT)')
    h.ir()
    h('target_sources(opmaster_ct PRIVATE')
    h.ir()
    h(f'{opdef.name}_tiling.cpp')
    h.il()
    h(')')
    h.il()
    h('endif ()')
    h()

    h('target_sources(opsproto PRIVATE')
    h(f'    {opdef.name}.cc')
    h(')')
    h()

    h('target_include_directories(optiling PRIVATE')
    h('    ${CMAKE_CURRENT_SOURCE_DIR}')
    h(')')

    file_out.write(h.code)
    file_out.close()
