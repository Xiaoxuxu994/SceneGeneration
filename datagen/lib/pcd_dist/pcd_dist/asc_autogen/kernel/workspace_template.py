BUILD_SH_TEMPLATE = '''
#!/bin/bash

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"

SHORT=r:,v:,i:,b:,p:,
LONG=run-mode:,soc-version:,install-path:,build-type:,install-prefix:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"


while :; do
    case "$1" in
    -r | --run-mode)
        RUN_MODE="$2"
        shift 2
        ;;
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -b | --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    -p | --install-prefix)
        INSTALL_PREFIX="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done

RUN_MODE_LIST="cpu sim npu"
if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    echo "ERROR: RUN_MODE error, This sample only support specify cpu, sim or npu!"
    exit -1
fi

VERSION_LIST="Ascend910A Ascend910B Ascend310B1 Ascend310B2 Ascend310B3 Ascend310B4 Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "ERROR: SOC_VERSION should be in [$VERSION_LIST]"
    exit -1
fi

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
if [ "${RUN_MODE}" = "npu" ]; then
    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
elif [ "${RUN_MODE}" = "sim" ]; then
    # in case of running op in simulator, use stub .so instead
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/runtime/lib64/stub:$LD_LIBRARY_PATH
    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
    if [ ! $CAMODEL_LOG_PATH ]; then
        export CAMODEL_LOG_PATH=$(pwd)/sim_log
    fi
    if [ -d "$CAMODEL_LOG_PATH" ]; then
        rm -rf $CAMODEL_LOG_PATH
    fi
    mkdir -p $CAMODEL_LOG_PATH
elif [ "${RUN_MODE}" = "cpu" ]; then
    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

set -e
rm -rf build out
mkdir -p build
cmake -B build \\
    -DRUN_MODE=${RUN_MODE} \\
    -DSOC_VERSION=${SOC_VERSION} \\
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \\
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \\
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
cmake --build build -j
cmake --install build

rm -f ascendc_kernels_bbit
cp ./out/bin/ascendc_kernels_bbit ./
mkdir -p input output

'''


RUN_SH_TEMPALTE = '''
#!/bin/bash

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"

SHORT=r:,v:,i:,b:,p:,
LONG=run-mode:,soc-version:,install-path:,build-type:,install-prefix:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"


while :; do
    case "$1" in
    -r | --run-mode)
        RUN_MODE="$2"
        shift 2
        ;;
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -b | --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    -p | --install-prefix)
        INSTALL_PREFIX="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done

RUN_MODE_LIST="cpu sim npu"
if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    echo "ERROR: RUN_MODE error, This sample only support specify cpu, sim or npu!"
    exit -1
fi

VERSION_LIST="Ascend910A Ascend910B Ascend310B1 Ascend310B2 Ascend310B3 Ascend310B4 Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "ERROR: SOC_VERSION should be in [$VERSION_LIST]"
    exit -1
fi

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
if [ "${RUN_MODE}" = "npu" ]; then
    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
elif [ "${RUN_MODE}" = "sim" ]; then
    # in case of running op in simulator, use stub .so instead
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/runtime/lib64/stub:$LD_LIBRARY_PATH
    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
    if [ ! $CAMODEL_LOG_PATH ]; then
        export CAMODEL_LOG_PATH=$(pwd)/sim_log
    fi
    if [ -d "$CAMODEL_LOG_PATH" ]; then
        rm -rf $CAMODEL_LOG_PATH
    fi
    mkdir -p $CAMODEL_LOG_PATH
elif [ "${RUN_MODE}" = "cpu" ]; then
    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

export LD_LIBRARY_PATH=$(pwd)/out/lib:$(pwd)/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH
./ascendc_kernels_bbit

'''

CLEAR_SH_TEMPLATE = '''
#!/bin/bash

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR
rm -rf build 
rm -rf input 
rm -rf out 
rm -rf output 
rm ascendc_kernels_bbit

'''

NPU_LIB_CMAKE_TEMPLATE = '''
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist ,please check whether the cann package is installed")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# set(CUSTOM_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/includes")

ascendc_library(ascendc_kernels_${RUN_MODE} SHARED
    ${KERNEL_FILES}
)

ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE 
  $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:INSIMULATION>>
  -DASCENDC_DUMP
)
'''

B_SH_TEMPLATE = 'bash ./workspace/build.sh -r npu -v Ascend910B3'
R_SH_TEMPALTE = 'bash ./workspace/run.sh -r npu -v Ascend910B3'
BB_SH_TEMPLATE = 'bash ./workspace/build.sh -r sim -v Ascend910B3'
RR_SH_TEMPALTE = 'bash ./workspace/run.sh -r sim -v Ascend910B3'


VSCODE_CPP_PROPERTIES_TEMPLATE = '''
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/home/ma-user/work/reference_headers",
                "/home/ma-user/work/reference_headers/**",
                "/home/ma-user/work/reference_headers/impl",
                "/home/ma-user/work/reference_headers/lib",
                "/home/ma-user/work/reference_headers/utils",
                "/home/ma-user/work/reference_headers/impl_2",
                "/home/ma-user/work/reference_headers/impl_2/impl"
            ],
            "defines": [
                "JUSTFORHINTER"
            ],
            "compilerPath": "/usr/bin/gcc",
            "intelliSenseMode": "linux-gcc-arm64",
            "cppStandard": "c++17",
            "cStandard": "c17"
        }
    ],
    "version": 4
}
'''

VSCODE_SETTINGS_TEMPLATE = '''
{
    "files.associations": {
        "*.cpp": "cpp",
        "*.cce": "cpp",
        "type_traits": "cpp",
        "string": "cpp",
        "array": "cpp",
        "atomic": "cpp",
        "*.tcc": "cpp",
        "cctype": "cpp",
        "chrono": "cpp",
        "clocale": "cpp",
        "cmath": "cpp",
        "csignal": "cpp",
        "cstdarg": "cpp",
        "cstddef": "cpp",
        "cstdint": "cpp",
        "cstdio": "cpp",
        "cstdlib": "cpp",
        "cstring": "cpp",
        "ctime": "cpp",
        "cwchar": "cpp",
        "cwctype": "cpp",
        "deque": "cpp",
        "unordered_map": "cpp",
        "unordered_set": "cpp",
        "vector": "cpp",
        "exception": "cpp",
        "algorithm": "cpp",
        "functional": "cpp",
        "iterator": "cpp",
        "map": "cpp",
        "memory": "cpp",
        "memory_resource": "cpp",
        "numeric": "cpp",
        "optional": "cpp",
        "random": "cpp",
        "ratio": "cpp",
        "set": "cpp",
        "string_view": "cpp",
        "system_error": "cpp",
        "tuple": "cpp",
        "utility": "cpp",
        "fstream": "cpp",
        "initializer_list": "cpp",
        "iomanip": "cpp",
        "iosfwd": "cpp",
        "iostream": "cpp",
        "istream": "cpp",
        "limits": "cpp",
        "new": "cpp",
        "ostream": "cpp",
        "sstream": "cpp",
        "stdexcept": "cpp",
        "streambuf": "cpp",
        "thread": "cpp",
        "typeinfo": "cpp"
    }
}
'''