#include "testmacros.h"
#include "aclrtlaunch_kernel_cust_dist.h"


int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 20;
    
    // constants
    int M = 40960;
    int K = 1048576;
    
    INIT_ACL();
    
    // Define Tensors
    DEFINE_VARIABLE(xmtx, M*sizeof(float));
    DEFINE_VARIABLE(ymtx, M*sizeof(float));
    DEFINE_VARIABLE(zmtx, M*sizeof(float));
    DEFINE_VARIABLE(x0mtx, K*sizeof(float));
    DEFINE_VARIABLE(y0mtx, K*sizeof(float));
    DEFINE_VARIABLE(z0mtx, K*sizeof(float));
    DEFINE_VARIABLE(outmtx, M*sizeof(float));
    
    DEFINE_VARIABLE(workspace, 1024);
    
    // Copy input tensors
    COPY_VARIABLE(xmtx);
    COPY_VARIABLE(ymtx);
    COPY_VARIABLE(zmtx);
    COPY_VARIABLE(x0mtx);
    COPY_VARIABLE(y0mtx);
    COPY_VARIABLE(z0mtx);
    
    // Run
    RUN_KERNEL(kernel_cust_dist, xmtx, ymtx, zmtx, x0mtx, y0mtx, z0mtx, outmtx, workspace, M, K);
    
#ifndef NOSPEEDTEST
    SPEED_TEST(kernel_cust_dist, xmtx, ymtx, zmtx, x0mtx, y0mtx, z0mtx, outmtx, workspace, M, K);
#endif
    
    // Write Tensors
    SYNC_STREAM();
    WRITE_VARIABLE(outmtx);
    WRITE_VARIABLE(workspace);
    
    // End
    END_ACL();
    
    return 0;
}
// Auto-generated code. Readability is not guaranteed
