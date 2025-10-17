#pragma once
#include "tensorutils.h"


struct CustVecShapeInfo{
    int M;
    int K;
    int MPERCORE_UPPER;
    int M1;
    int M2;
};


__aicore__ inline void tilingShapeCustVec(int M, int K, CustVecShapeInfo &shape){
    shape.M = M;
    shape.K = K;
    shape.MPERCORE_UPPER = (CeilDiv(CeilDiv(shape.M,1024),(GetBlockNum() * 2)) * 1024);
    shape.M1 = (shape.MPERCORE_UPPER * GetBlockIdx());
    shape.M2 = (((shape.M1 + shape.MPERCORE_UPPER))<(shape.M)) ? ((shape.M1 + shape.MPERCORE_UPPER)) : (shape.M);
}


class CustVec{
public:
    __aicore__ inline CustVec(){}
    __aicore__ inline void Init(GM_ADDR xmtx_, GM_ADDR ymtx_, GM_ADDR zmtx_, GM_ADDR x0mtx_, GM_ADDR y0mtx_, GM_ADDR z0mtx_, GM_ADDR outmtx_, CustVecShapeInfo shape_){
        shape = shape_;
        // Reset tpipe. Start resource distribution
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->Reset();
        
        // Global Tensors
        xmtx.SetGlobalBuffer((__gm__ float*) xmtx_);
        ymtx.SetGlobalBuffer((__gm__ float*) ymtx_);
        zmtx.SetGlobalBuffer((__gm__ float*) zmtx_);
        x0mtx.SetGlobalBuffer((__gm__ float*) x0mtx_);
        y0mtx.SetGlobalBuffer((__gm__ float*) y0mtx_);
        z0mtx.SetGlobalBuffer((__gm__ float*) z0mtx_);
        outmtx.SetGlobalBuffer((__gm__ float*) outmtx_);
        
        // Local buffers
        AllocateLocalTensor<TPosition::VECCALC>(xdata, 1024);
        AllocateLocalTensor<TPosition::VECCALC>(ydata, 1024);
        AllocateLocalTensor<TPosition::VECCALC>(zdata, 1024);
        xbuf.Init(1024);
        ybuf.Init(1024);
        zbuf.Init(1024);
        prevdistbuf.Init(1024);
        AllocateLocalTensor<TPosition::VECCALC>(distbuf, 1024);
        AllocateLocalTensor<TPosition::VECCALC>(xcurr, 64);
        AllocateLocalTensor<TPosition::VECCALC>(ycurr, 64);
        AllocateLocalTensor<TPosition::VECCALC>(zcurr, 64);
        distoutbuf.Init(1024);
        // Initialize events
        in_empty.Init();
        in_ready.Init();
        out_empty.Init();
        out_ready.Init();
    }
    
    __aicore__ inline void Compute(){
        in_empty.setall();
        out_empty.setall();
        int cnt = 0;
        float xanchor = 0;
        float yanchor = 0;
        float zanchor = 0;
        for (int m=shape.M1; m<shape.M2; m+=1024){
            int curr_idx = 0;
            for (int k=0; k<shape.K; k+=1){
                in_empty.wait();
                if ((k == 0)){
                    if (((m + 1024) <= shape.M2)){
                        GM2UB(xdata[cnt], xmtx[m], 1, 128, 0, 0);
                        GM2UB(ydata[cnt], ymtx[m], 1, 128, 0, 0);
                        GM2UB(zdata[cnt], zmtx[m], 1, 128, 0, 0);
                    }
                    else {
                        GM2UBPad(xdata[cnt], xmtx[m], 1, ((shape.M2 - m) * 4), 0, 0,0,0,0,0);
                        GM2UBPad(ydata[cnt], ymtx[m], 1, ((shape.M2 - m) * 4), 0, 0,0,0,0,0);
                        GM2UBPad(zdata[cnt], zmtx[m], 1, ((shape.M2 - m) * 4), 0, 0,0,0,0,0);
                    }
                }
                if (((shape.K - (((int)curr_idx / (int)8) * 8)) >= 8)){
                    GM2UB(xcurr, x0mtx[(((int)curr_idx / (int)8) * 8)], 1, 1, 0, 0);
                    GM2UB(ycurr, y0mtx[(((int)curr_idx / (int)8) * 8)], 1, 1, 0, 0);
                    GM2UB(zcurr, z0mtx[(((int)curr_idx / (int)8) * 8)], 1, 1, 0, 0);
                }
                else {
                    GM2UBPad(xcurr, x0mtx[(((int)curr_idx / (int)8) * 8)], 1, ((shape.K - (((int)curr_idx / (int)8) * 8)) * 4), 0, 0,0,0,0,0);
                    GM2UBPad(ycurr, y0mtx[(((int)curr_idx / (int)8) * 8)], 1, ((shape.K - (((int)curr_idx / (int)8) * 8)) * 4), 0, 0,0,0,0,0);
                    GM2UBPad(zcurr, z0mtx[(((int)curr_idx / (int)8) * 8)], 1, ((shape.K - (((int)curr_idx / (int)8) * 8)) * 4), 0, 0,0,0,0,0);
                }
                in_ready.set();
                in_ready.wait();
                xanchor = (float) xcurr[(curr_idx % 8)].GetValue(0);
                yanchor = (float) ycurr[(curr_idx % 8)].GetValue(0);
                zanchor = (float) zcurr[(curr_idx % 8)].GetValue(0);
                xanchor = ((float)(xanchor * (float)-1));
                yanchor = ((float)(yanchor * (float)-1));
                zanchor = ((float)(zanchor * (float)-1));
                PipeBarrier<PIPE_ALL>();
                Adds<float, false>(xbuf.get(cnt), xdata[cnt], (float)xanchor, MASK_PLACEHOLDER, 16, {1, 1, 8, 8});
                Adds<float, false>(ybuf.get(cnt), ydata[cnt], (float)yanchor, MASK_PLACEHOLDER, 16, {1, 1, 8, 8});
                Adds<float, false>(zbuf.get(cnt), zdata[cnt], (float)zanchor, MASK_PLACEHOLDER, 16, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                Mul<float, false>(xbuf.get(cnt), xbuf.get(cnt), xbuf.get(cnt), MASK_PLACEHOLDER, 16, {1, 1, 1, 8, 8, 8});
                Mul<float, false>(ybuf.get(cnt), ybuf.get(cnt), ybuf.get(cnt), MASK_PLACEHOLDER, 16, {1, 1, 1, 8, 8, 8});
                Mul<float, false>(zbuf.get(cnt), zbuf.get(cnt), zbuf.get(cnt), MASK_PLACEHOLDER, 16, {1, 1, 1, 8, 8, 8});
                PipeBarrier<PIPE_V>();
                Add<float, false>(distbuf, xbuf.get(cnt), ybuf.get(cnt), MASK_PLACEHOLDER, 16, {1, 1, 1, 8, 8, 8});
                PipeBarrier<PIPE_V>();
                Add<float, false>(distoutbuf.get(cnt), distbuf, zbuf.get(cnt), MASK_PLACEHOLDER, 16, {1, 1, 1, 8, 8, 8});
                PipeBarrier<PIPE_V>();
                in_empty.set();
                if ((k != 0)){
                    Min<float, false>(distoutbuf.get(cnt), distoutbuf.get(cnt), prevdistbuf.get(cnt), MASK_PLACEHOLDER, 16, {1, 1, 1, 8, 8, 8});
                    PipeBarrier<PIPE_ALL>();
                }
                curr_idx = (curr_idx + 1);
                UB2UB(prevdistbuf.get(cnt), distoutbuf.get(cnt), 1, 128, 0, 0);
            }
            out_empty.wait();
            Sqrt<float, false>(distoutbuf.get(cnt), distoutbuf.get(cnt), MASK_PLACEHOLDER, 16, {1, 1, 8, 8});
            PipeBarrier<PIPE_ALL>();
            out_ready.set();
            out_ready.wait();
            if (((m + 1024) <= shape.M2)){
                UB2GM(outmtx[m], distoutbuf.get(cnt), 1, 128, 0, 0);
            }
            else {
                UB2GMPad(outmtx[m], distoutbuf.get(cnt), 1, ((shape.M2 - m) * 4), 0, 0);
            }
            out_empty.set();
            PipeBarrier<PIPE_ALL>();
            cnt = (cnt + 1);
            PipeBarrier<PIPE_ALL>();
        }
        in_empty.release();
        out_empty.release();
    }
    
private:
    CustVecShapeInfo shape;
    // Global Tensors
    GlobalTensor<float> xmtx;
    GlobalTensor<float> ymtx;
    GlobalTensor<float> zmtx;
    GlobalTensor<float> x0mtx;
    GlobalTensor<float> y0mtx;
    GlobalTensor<float> z0mtx;
    GlobalTensor<float> outmtx;
    // Local buffers
    LocalTensor<float> xdata;
    LocalTensor<float> ydata;
    LocalTensor<float> zdata;
    DBuff<float, TPosition::VECCALC> xbuf;
    DBuff<float, TPosition::VECCALC> ybuf;
    DBuff<float, TPosition::VECCALC> zbuf;
    DBuff<float, TPosition::VECCALC> prevdistbuf;
    LocalTensor<float> distbuf;
    LocalTensor<float> xcurr;
    LocalTensor<float> ycurr;
    LocalTensor<float> zcurr;
    DBuff<float, TPosition::VECCALC> distoutbuf;
    
    // Events
    DEvent<PIPE_V, PIPE_MTE2> in_empty;
    DEvent<PIPE_MTE2, PIPE_V> in_ready;
    DEvent<PIPE_MTE3, PIPE_V> out_empty;
    DEvent<PIPE_V, PIPE_MTE3> out_ready;
};
