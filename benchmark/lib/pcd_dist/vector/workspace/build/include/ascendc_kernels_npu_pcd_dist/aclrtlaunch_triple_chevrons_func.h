
#ifndef HEADER_ACLRTLAUNCH_KERNEL_PCD_DIST_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_KERNEL_PCD_DIST_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_kernel_pcd_dist(uint32_t blockDim, void* stream, void* xmtx, void* ymtx, void* zmtx, void* x0mtx, void* y0mtx, void* z0mtx, void* outmtx, void* workspace, int M, int K);

inline uint32_t kernel_pcd_dist(uint32_t blockDim, void* hold, void* stream, void* xmtx, void* ymtx, void* zmtx, void* x0mtx, void* y0mtx, void* z0mtx, void* outmtx, void* workspace, int M, int K)
{
    (void)hold;
    return aclrtlaunch_kernel_pcd_dist(blockDim, stream, xmtx, ymtx, zmtx, x0mtx, y0mtx, z0mtx, outmtx, workspace, M, K);
}

#endif
