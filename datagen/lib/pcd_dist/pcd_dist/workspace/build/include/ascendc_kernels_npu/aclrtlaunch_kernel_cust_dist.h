#ifndef HEADER_ACLRTLAUNCH_KERNEL_CUST_DIST_H
#define HEADER_ACLRTLAUNCH_KERNEL_CUST_DIST_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_kernel_cust_dist(uint32_t blockDim, aclrtStream stream, void* xmtx, void* ymtx, void* zmtx, void* x0mtx, void* y0mtx, void* z0mtx, void* outmtx, void* workspace, int M, int K);
#endif
