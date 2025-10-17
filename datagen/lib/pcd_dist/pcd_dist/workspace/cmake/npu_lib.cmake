
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
