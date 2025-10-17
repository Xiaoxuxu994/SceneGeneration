#ifndef __KERNEL_CUSTOM_DIST__KERNEL_FUN_H__
#define __KERNEL_CUSTOM_DIST__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define kernel_cust_dist kernel_cust_dist_origin
#include "/home/yyx/code/pcd_dist/pcd_dist/codebase/kernel_custom_dist.cpp"

#undef kernel_cust_dist
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_kernel_cust_dist_kernel(
GM_ADDR ffts_addr, __attribute__((cce_global)) uint8_t* xmtx, __attribute__((cce_global)) uint8_t* ymtx, __attribute__((cce_global)) uint8_t* zmtx, __attribute__((cce_global)) uint8_t* x0mtx, __attribute__((cce_global)) uint8_t* y0mtx, __attribute__((cce_global)) uint8_t* z0mtx, __attribute__((cce_global)) uint8_t* outmtx, __attribute__((cce_global)) uint8_t* workspace, int M, int K, GM_ADDR overflow_status) {
    icache_preload(1);
    if (ffts_addr != nullptr) {
        set_ffts_base_addr((uint64_t)ffts_addr);
    }
#ifdef ASCENDC_TIME_STAMP_ON
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_FFTS_ADDR));
#endif
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = M;
#else
    workspace_param = K;
#endif
    if (workspace_param == nullptr) {
        return;
    }
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(REGIST_MATMUL_OBJ) || defined(__MIX_CORE_MACRO__)
    if constexpr (g_coreType == AscendC::AIC) {
        matmul::clearWorkspace(workspace_param);
#ifdef ASCENDC_TIME_STAMP_ON
        AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_CLEAR_WK_SPAC));
#endif
    }
#endif
#if defined(HAVE_TILING)
    M = workspace_usr;
#else
    K = workspace_usr;
#endif
#endif
    kernel_cust_dist_origin(xmtx, ymtx, zmtx, x0mtx, y0mtx, z0mtx, outmtx, workspace, M, K);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
}

#endif
#include "inner_interface/inner_kernel_operator_intf.h"
