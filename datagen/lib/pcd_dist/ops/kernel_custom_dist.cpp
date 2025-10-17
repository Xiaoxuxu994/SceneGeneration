#include "torch/extension.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" uint32_t aclrtlaunch_kernel_cust_dist(uint32_t blockDim, aclrtStream stream,
    void* xpt, void* ypt, void* zpt, void* x0pt, void* y0pt, void* z0pt, void* out, void* workspace, int M, int K);

void kernel_custom_dist(at::Tensor x_query, at::Tensor y_query, at::Tensor z_query, at::Tensor x_target, at::Tensor y_target, at::Tensor z_target, at::Tensor distance, at::Tensor workspace){
    int devidx = x_query.device().index();
    int M = x_query.size(0);
    int K = x_target.size(0);
    // aclrtSetDevice(devidx);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    aclrtlaunch_kernel_cust_dist(20, aclstream, \
        (void*)(x_query.storage().data()), (void*)(y_query.storage().data()), (void*)(z_query.storage().data()), \
        (void*)(x_target.storage().data()), (void*)(y_target.storage().data()), (void*)(z_target.storage().data()), \
        (void*)(distance.storage().data()), (void*)(workspace.storage().data()), M, K);
    // aclrtResetDevice(devidx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel_custom_dist", &kernel_custom_dist, "kernel_custom_dist");
}
