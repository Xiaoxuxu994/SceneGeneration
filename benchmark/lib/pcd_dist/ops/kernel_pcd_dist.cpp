#include "torch/extension.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" uint32_t aclrtlaunch_kernel_pcd_dist(uint32_t blockDim, aclrtStream stream,
    void* xpt, void* ypt, void* zpt, void* x0pt, void* y0pt, void* z0pt, void* out, void* workspace, int M, int K);

void kernel_pcd_dist(at::Tensor queries, at::Tensor targets, at::Tensor distance, at::Tensor workspace){
    int devidx = queries.device().index();
    // aclrtSetDevice(devidx);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    aclrtlaunch_kernel_pcd_dist(40, aclstream, \
        (void*)(queries.select(1, 0).contiguous().storage().data()), \
        (void*)(queries.select(1, 1).contiguous().storage().data()), \
        (void*)(queries.select(1, 2).contiguous().storage().data()), \
        (void*)(targets.select(1, 0).contiguous().storage().data()), \
        (void*)(targets.select(1, 1).contiguous().storage().data()), \
        (void*)(targets.select(1, 2).contiguous().storage().data()), \
        (void*)(distance.storage().data()), \
        (void*)(workspace.storage().data()), \
        queries.size(0), targets.size(0));
    // aclrtResetDevice(devidx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel_pcd_dist", &kernel_pcd_dist, "kernel_pcd_dist");
}
