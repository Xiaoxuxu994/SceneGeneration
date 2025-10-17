#include "tensorutils.h"
#include "CustVec.h"


class CubeHandler{
public:
    __aicore__ inline CubeHandler(){}
    __aicore__ inline void Init(GM_ADDR xmtx, GM_ADDR ymtx, GM_ADDR zmtx, GM_ADDR x0mtx, GM_ADDR y0mtx, GM_ADDR z0mtx, GM_ADDR outmtx, GM_ADDR workspace, int M, int K){
        // workspace allocation
        
        // kernel tiling
        
        // kernel initialization
    }
    
    __aicore__ inline void Compute(){
    }

private:
    TPipe pipe;
};


class VecHandler{
public:
    __aicore__ inline VecHandler(){}
    __aicore__ inline void Init(GM_ADDR xmtx, GM_ADDR ymtx, GM_ADDR zmtx, GM_ADDR x0mtx, GM_ADDR y0mtx, GM_ADDR z0mtx, GM_ADDR outmtx, GM_ADDR workspace, int M, int K){
        // workspace allocation
        
        // kernel tiling
        tilingShapeCustVec(M, K, vec0shape);
        
        // kernel initialization
        vec0.Init(xmtx, ymtx, zmtx, x0mtx, y0mtx, z0mtx, outmtx, vec0shape);
    }
    
    __aicore__ inline void Compute(){
        vec0.Compute();
    }

private:
    TPipe pipe;
    CustVec vec0;
    CustVecShapeInfo vec0shape;
};


extern "C" __global__ __aicore__ void kernel_cust_dist(GM_ADDR xmtx, GM_ADDR ymtx, GM_ADDR zmtx, GM_ADDR x0mtx, GM_ADDR y0mtx, GM_ADDR z0mtx, GM_ADDR outmtx, GM_ADDR workspace, int M, int K){
    if ASCEND_IS_AIC{
        CubeHandler cube;
        cube.Init(xmtx, ymtx, zmtx, x0mtx, y0mtx, z0mtx, outmtx, workspace, M, K);
        cube.Compute();
        mad((__cc__ float*)0, (__ca__ float*)0, (__cb__ float*)0, 32, 32, 32, 0x0, false, false, true);  // dummy instruction to make compiler happy
    }
    if ASCEND_IS_AIV{
        VecHandler vec;
        vec.Init(xmtx, ymtx, zmtx, x0mtx, y0mtx, z0mtx, outmtx, workspace, M, K);
        vec.Compute();
        copy_ubuf_to_ubuf((__ubuf__ float*)0, (__ubuf__ float*)0, 0, 1, 1, 0, 0);  // dummy instruction to make compiler happy
    }
}
