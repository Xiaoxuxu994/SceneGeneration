TENSORUTILS_TEMPLATE = '''
#pragma once
#include "kernel_operator.h"



#ifdef JUSTFORHINTER
// just for type hinting.
#undef __aicore__
#define __aicore__
#include "stub_fun.h"
#endif 


using namespace AscendC;

#define PIPE_FIX (pipe_t)10

#define ALLCUBE_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x0, append_to_pipe>(iiii)
#define ALLCUBE_WAIT(iiii) CrossCoreWaitFlag(iiii)
#define ALLVEC_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x0, append_to_pipe>(iiii)
#define ALLVEC_WAIT(iiii) CrossCoreWaitFlag(iiii)
#define CUBE_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x2, append_to_pipe>(iiii)
#define WAIT_CUBE(iiii) CrossCoreWaitFlag(iiii)
#define VEC_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x2, append_to_pipe>(iiii)
#define WAIT_VEC(iiii) CrossCoreWaitFlag(iiii)

constexpr uint64_t VECTORFULLMASK[2] = {(uint64_t)-1, (uint64_t)-1};
#ifdef __DAV_C310__
constexpr FixpipeConfig CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};
#endif 

__aicore__ constexpr HardEvent GetHardEventByPipe(pipe_t src, pipe_t dst){
    if (src==PIPE_MTE2){
        if (dst==PIPE_MTE1){
            return HardEvent::MTE2_MTE1;
        }else if(dst==PIPE_V){
            return HardEvent::MTE2_V;
        }
    }else if(src==PIPE_MTE1){
        if (dst==PIPE_MTE2){
            return HardEvent::MTE1_MTE2;
        }else if(dst==PIPE_M){
            return HardEvent::MTE1_M;
        }else if(dst==PIPE_FIX){
            return HardEvent::MTE1_FIX;
        }
    }else if(src==PIPE_M){
        if (dst==PIPE_MTE1){
            return HardEvent::M_MTE1;
        }else if (dst==PIPE_FIX){
            return HardEvent::M_FIX;
        }
    }else if(src==PIPE_FIX){
        if (dst==PIPE_M){
            return HardEvent::FIX_M;
        }else if (dst==PIPE_MTE1){
            return HardEvent::FIX_MTE1;
        }
    }else if(src==PIPE_V){
        if (dst==PIPE_MTE2){
            return HardEvent::V_MTE2;
        }else if(dst==PIPE_MTE3){
            return HardEvent::V_MTE3;
        }
    }else if(src==PIPE_MTE3){
        if (dst==PIPE_V){
            return HardEvent::MTE3_V;
        }
    }
}

__aicore__ inline void OccupyMMTE1Events(){
    if ASCEND_IS_AIC{
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->AllocEventID<HardEvent::M_MTE1>();
        pipe_ptr->AllocEventID<HardEvent::M_MTE1>();
        pipe_ptr->AllocEventID<HardEvent::M_MTE1>();
    }
}


__aicore__ constexpr int Align16B(int x){
    return (x + 15) / 16 * 16;
}

__aicore__ constexpr int Align32B(int x){
    return (x + 31) / 32 * 32;
}

__aicore__ constexpr int Align64B(int x){
    return (x + 63) / 64 * 64;
}

__aicore__ constexpr int Align128B(int x){
    return (x + 127) / 128 * 128;
}

__aicore__ constexpr int Align256B(int x){
    return (x + 255) / 256 * 256;
}

__aicore__ constexpr int Align512B(int x){
    return (x + 511) / 512 * 512;
}

__aicore__ inline int CeilDiv(int a, int b){
    return (a + b - 1) / b;
}

template <typename T, typename T1, typename T2>
__aicore__ inline T1 shiftAddr(T1 base, uint64_t size, T2 &offset){
    auto res = base + offset;
    offset += size*sizeof(T);
    return res;
}


/* ------------- Tensor ------------- */ 
template <TPosition pos, typename T>
__aicore__ inline void AllocateLocalTensor(LocalTensor<T> &tsr, int len){
    TBuf<pos> tbuf;
    TPipe* ptr = GetTPipePtr();
    ptr->InitBuffer(tbuf, len * sizeof(T));
    tsr = tbuf.template Get<T>();
}

/* ------------- Tensor ------------- */ 


/* ------------- Double Buffer ------------- */ 

template <typename T, TPosition pos>
class DBuff{
public:
    __aicore__ inline DBuff(){}
    __aicore__ inline void Init(int len){
        TPipe* ptr = GetTPipePtr();
        ptr->InitBuffer(buf1, len * sizeof(T));
        ptr->InitBuffer(buf2, len * sizeof(T));
        tsr1 = buf1.template Get<T>();
        tsr2 = buf2.template Get<T>();
    }
    
    __aicore__ inline LocalTensor<T> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    TBuf<pos> buf1, buf2;
    LocalTensor<T> tsr1, tsr2;
};
/* ------------- Double Buffer ------------- */ 


/* ------------- Triple Buffer ------------- */ 
template <typename T, TPosition pos>
class TBuff{
public:
    __aicore__ inline TBuff(){}
    __aicore__ inline void Init(int len){
        TPipe* ptr = GetTPipePtr();
        ptr->InitBuffer(buf1, len * sizeof(T));
        ptr->InitBuffer(buf2, len * sizeof(T));
        ptr->InitBuffer(buf3, len * sizeof(T));
        tsr1 = buf1.template Get<T>();
        tsr2 = buf2.template Get<T>();
        tsr3 = buf3.template Get<T>();
    }
    
    __aicore__ inline LocalTensor<T> get(int i){
        if (i%3==0){
            return tsr1;
        }else if (i%3==1){
            return tsr2;
        }else{
            return tsr3;
        }
    }
private:
    TBuf<pos> buf1, buf2, buf3;
    LocalTensor<T> tsr1, tsr2, tsr3;
};
/* ------------- Triple Buffer ------------- */ 


/* ------------- Quadro Buffer ------------- */ 
template <typename T, TPosition pos>
class QBuff{
public:
    __aicore__ inline QBuff(){}
    __aicore__ inline void Init(int len){
        TPipe* ptr = GetTPipePtr();
        ptr->InitBuffer(buf1, len * sizeof(T));
        ptr->InitBuffer(buf2, len * sizeof(T));
        ptr->InitBuffer(buf3, len * sizeof(T));
        ptr->InitBuffer(buf4, len * sizeof(T));
        tsr1 = buf1.template Get<T>();
        tsr2 = buf2.template Get<T>();
        tsr3 = buf3.template Get<T>();
        tsr4 = buf4.template Get<T>();
    }
    
    __aicore__ inline LocalTensor<T> get(int i){
        if (i%4==0){
            return tsr1;
        }else if (i%4==1){
            return tsr2;
        }else if (i%4==2){
            return tsr3;
        }else{
            return tsr4;
        }
    }
private:
    TBuf<pos> buf1, buf2, buf3, buf4;
    LocalTensor<T> tsr1, tsr2, tsr3, tsr4;
};
/* ------------- Quadro Buffer ------------- */ 


/* ------------- Penta Buffer ------------- */ 
template <typename T, TPosition pos>
class PBuff{
public:
    __aicore__ inline PBuff(){}
    __aicore__ inline void Init(int len){
        TPipe* ptr = GetTPipePtr();
        ptr->InitBuffer(buf1, len * sizeof(T));
        ptr->InitBuffer(buf2, len * sizeof(T));
        ptr->InitBuffer(buf3, len * sizeof(T));
        ptr->InitBuffer(buf4, len * sizeof(T));
        ptr->InitBuffer(buf5, len * sizeof(T));
        tsr1 = buf1.template Get<T>();
        tsr2 = buf2.template Get<T>();
        tsr3 = buf3.template Get<T>();
        tsr4 = buf4.template Get<T>();
        tsr5 = buf5.template Get<T>();
    }
    
    __aicore__ inline LocalTensor<T> get(int i){
        if (i%5==0){
            return tsr1;
        }else if (i%5==1){
            return tsr2;
        }else if (i%5==2){
            return tsr3;
        }else if (i%5==3){
            return tsr4;
        }else{
            return tsr5;
        }
    }
private:
    TBuf<pos> buf1, buf2, buf3, buf4, buf5;
    LocalTensor<T> tsr1, tsr2, tsr3, tsr4, tsr5;
};
/* ------------- Penta Buffer ------------- */ 


/* ------------- Events ------------- */ 

template <pipe_t p1, pipe_t p2>
class SEvent{
public:
    __aicore__ inline SEvent(){}
    __aicore__ inline void Init(){
        TPipe* pipe_ptr = GetTPipePtr();
        id1 = (event_t)pipe_ptr->AllocEventID<GetHardEventByPipe(p1, p2)>();
    }
    __aicore__ inline void wait(){
        wait_flag(p1, p2, id1);
    }
    __aicore__ inline void set(){
        set_flag(p1, p2, id1);
    }
    __aicore__ inline void setall(){
        set();
    }
    __aicore__ inline void release(){
        wait();
    }

private:
    event_t id1;
};



template <pipe_t p1, pipe_t p2>
class DEvent{
public:
    __aicore__ inline DEvent(){}
    __aicore__ inline void Init(){
        TPipe* pipe_ptr = GetTPipePtr();
        id1 = (event_t)pipe_ptr->AllocEventID<GetHardEventByPipe(p1, p2)>();
        id2 = (event_t)pipe_ptr->AllocEventID<GetHardEventByPipe(p1, p2)>();
    }
    __aicore__ inline void wait(){
        if (wait_cnt%2==0){
            wait_flag(p1, p2, id1);
        }else{
            wait_flag(p1, p2, id2);
        }
        wait_cnt ++;
    }
    __aicore__ inline void set(){
        if (set_cnt%2==0){
            set_flag(p1, p2, id1);
        }else{
            set_flag(p1, p2, id2);
        }
        set_cnt ++;
    }
    __aicore__ inline void setall(){
        set();
        set();
    }
    __aicore__ inline void release(){
        for (int i=wait_cnt; i<set_cnt; ++i){
            wait();
        }
    }

private:
    event_t id1, id2;
    int wait_cnt = 0;
    int set_cnt = 0;
};



/* ------------- Funcs -------------- */
template<typename T>
__aicore__ inline void L1ND2NZ(LocalTensor<T> dst, GlobalTensor<T> src, int h, int w, int W, int Hdst){
    Nd2NzParams param;
    param.ndNum = 1;
    param.nValue = h;
    param.dValue = w;
    param.srcNdMatrixStride = 0;
    param.srcDValue = W;
    param.dstNzC0Stride = (Hdst + 15) / 16 * 16;
    param.dstNzNStride = 1;
    param.dstNzMatrixStride = 0;
    DataCopy(dst, src, param);
}

template<typename T>
__aicore__ inline void GM2L1(LocalTensor<T> dst, GlobalTensor<T> src, int nBurst, int burstLen, int srcStride, int dstStride){
    DataCopyParams param;
    param.blockCount = nBurst;
    param.blockLen = burstLen;
    param.srcStride = srcStride;
    param.dstStride = dstStride;
    DataCopy(dst, src, param);
}

template <typename T>
__aicore__ inline void L0NZ2ZZ(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
#ifdef __DAV_C220_CUBE__
    LoadData2DParams param;
    param.repeatTimes = (ndst+32/sizeof(T)-1)/(32/sizeof(T));
    param.srcStride = (msrc+15)/16;

    for (int i=0; i<(mdst+15)/16; ++i){
        // load_cbuf_to_ca(dst[16*i*((ndst+15)/16*16)].ptr(), src[i*16*16].ptr(), 0, (ndst+15)/16, (msrc+15)/16, 0, 0, false, (addr_cal_mode_t)0);
        LoadData(dst[16*i*((ndst+15)/16*16)], src[i*16*16], param);
    }
#elif __DAV_C310__

#endif
}


template<typename T> 
__aicore__ inline void L0NZ2ZN(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
#ifdef __DAV_C220_CUBE__
    LoadData2DParams param;
    param.repeatTimes = (ndst+32/sizeof(T)-1)/(32/sizeof(T));
    param.srcStride = (msrc+15)/16;
    param.ifTranspose = true;

    for (int i=0; i<(mdst+15)/16; ++i){
        // load_cbuf_to_ca(dst[16*i*((ndst+15)/16*16)].ptr(), src[i*16*16].ptr(), 0, (ndst+15)/16, (msrc+15)/16, 0, 0, true, (addr_cal_mode_t)0);
        LoadData(dst[16*i*((ndst+15)/16*16)], src[i*16*16], param);
    }
#elif __DAV_C310__
    // LoadData2DParamsV2 param;
    // param.mStep = 1;
    // param.kStep = (ndst+32/sizeof(T)-1)/(32/sizeof(T));
    // param.srcStride = (msrc+15)/16;
    // param.dstStride = 1;
    // param.ifTranspose = true;
    // for (int i=0; i<(mdst+15)/16; ++i){
    //     // load_cbuf_to_ca(dst[16*i*((ndst+15)/16*16)].ptr(), src[i*16*16].ptr(), 0, (ndst+15)/16, (msrc+15)/16, 0, 0, false, (addr_cal_mode_t)0);
    //     LoadData(dst[16*i*((ndst+15)/16*16)], src[i*16*16], param);
    // }
    LoadData2DParamsV2 param;
    param.mStep = (mdst+15)/16;
    param.kStep = (ndst+15)/16;
    param.srcStride = (msrc+15)/16;
    param.dstStride = (ndst+15)/16;
    param.ifTranspose = true;
    LoadData(dst, src, param);
#endif 
}


template<typename T> 
__aicore__ inline void L0NZ2NZ(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
#ifdef __DAV_C220_CUBE__
    LoadData2DParams param;
    param.repeatTimes = (mdst+15)/16;
    param.srcStride = 1;

    for (int i=0; i<(ndst+15)/16; ++i){
        // load_cbuf_to_ca(dst[16*i*((mdst+15)/16*16)].ptr(), src[16*i*((msrc+15)/16*16)].ptr(), 0, (mdst+15)/16, 1, 0, 0, false, (addr_cal_mode_t)0);
        LoadData(dst[16*i*((mdst+15)/16*16)], src[16*i*((msrc+15)/16*16)], param);
    }
#elif __DAV_C310__
    LoadData2DParamsV2 param;
    param.mStep = (mdst+15)/16;
    param.kStep = (ndst+32/sizeof(T)-1)/(32/sizeof(T));
    param.srcStride = (msrc+15)/16;
    param.dstStride = (mdst+15)/16;
    LoadData(dst, src, param);
#endif 
}


template<typename T> 
__aicore__ inline void L0NZ2NN(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
#ifdef __DAV_C220_CUBE__
    LoadData2DParams param;
    param.repeatTimes = (mdst+15)/16;
    param.srcStride = 1;
    param.ifTranspose = true;

    for (int i=0; i<(ndst+15)/16; ++i){
        // load_cbuf_to_ca(dst[16*i*((mdst+15)/16*16)].ptr(), src[16*i*((msrc+15)/16*16)].ptr(), 0, (mdst+15)/16, 1, 0, 0, false, (addr_cal_mode_t)0);
        LoadData(dst[16*i*((mdst+15)/16*16)], src[16*i*((msrc+15)/16*16)], param);
    }
#elif __DAV_C310__

#endif 
}


template<typename T> 
__aicore__ inline void LOADL0(LocalTensor<T> dst, LocalTensor<T> src, int m, int n){
#ifdef __DAV_C220_CUBE__
    LoadData2DParams param;
    param.repeatTimes = m*n*sizeof(T)/32/16;
    param.srcStride = 1;
    LoadData(dst, src, param);
#elif __DAV_C310__
    LoadData2DParamsV2 param;
    param.mStep = 1;
    param.kStep = m*n*sizeof(T)/32/16;
    param.srcStride = 1;
    param.dstStride = 1;
    LoadData(dst, src, param);
#endif 
}


template<typename T>
__aicore__ inline void GM2UB(LocalTensor<T> dst, GlobalTensor<T> src, int nBurst, int burstLen, int srcStride, int dstStride){
    DataCopyParams param;
    param.blockCount = nBurst;
    param.blockLen = burstLen;
    param.srcStride = srcStride;
    param.dstStride = dstStride;
    DataCopy(dst, src, param);
}

template<typename T>
__aicore__ inline void UB2GM(GlobalTensor<T> dst, LocalTensor<T> src, int nBurst, int burstLen, int srcStride, int dstStride){
    DataCopyParams param;
    param.blockCount = nBurst;
    param.blockLen = burstLen;
    param.srcStride = srcStride;
    param.dstStride = dstStride;
    DataCopy(dst, src, param);
}

template<typename T>
__aicore__ inline void UB2UB(LocalTensor<T> dst, LocalTensor<T> src, int nBurst, int burstLen, int srcStride, int dstStride){
    DataCopyParams param;
    param.blockCount = nBurst;
    param.blockLen = burstLen;
    param.srcStride = srcStride;
    param.dstStride = dstStride;
    DataCopy(dst, src, param);
}

template<typename T>
__aicore__ inline void UB2L1(LocalTensor<T> dst, LocalTensor<T> src, int nBurst, int burstLen, int srcStride, int dstStride){
    DataCopyParams param;
    param.blockCount = nBurst;
    param.blockLen = burstLen;
    param.srcStride = srcStride;
    param.dstStride = dstStride;
    DataCopy(dst, src, param);
}

template<typename T>
__aicore__ inline void UB2L1_NZ(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
    const int C0 = 32 / sizeof(T);
    DataCopyParams param;
    param.blockCount = (nsrc + C0 - 1) / C0;
    param.blockLen = msrc;
    param.srcStride = 0;
    param.dstStride = (mdst + 15) / 16 * 16 - msrc;
    DataCopy(dst, src, param);
}

template<typename T>
__aicore__ inline void UB2L1_ND2NZ(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
    // default msrc <= mdst
    const int C0 = 32 / sizeof(T);
    DataCopyParams param;
    param.blockCount = msrc;
    param.blockLen = 1;
    param.srcStride = (nsrc + C0 - 1) / C0 - 1;
    param.dstStride = 0;
    for (int i=0; i<(nsrc + C0 - 1) / C0; ++i){
        DataCopy(dst[i*C0*((mdst + 15) / 16 * 16)], src[i*C0], param);
    }
}

template<typename T>
__aicore__ inline void UB2UB_ND2NZ(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
    const int C0 = 32 / sizeof(T);
    DataCopyParams param;
    param.blockCount = (nsrc + C0 - 1) / C0;
    param.blockLen = 1;
    param.srcStride = 0;
    param.dstStride = (mdst + 15) / 16 * 16 - 1;
    for (int i=0; i<msrc; ++i){
        DataCopy(dst[i*C0], src[i*nsrc], param);
    }
}

template<typename T>
__aicore__ inline void UB2UB_ND2NZ_COMPACT(LocalTensor<T> dst, LocalTensor<T> src, int m, int n){
    const int C0 = 32 / sizeof(T);
    DataCopyParams param;
    param.blockCount = (n + C0 - 1) / C0;
    param.blockLen = 1;
    param.srcStride = 0;
    param.dstStride = m - 1;
    for (int i=0; i<m; ++i){
        DataCopy(dst[i*C0], src[i*n], param);
    }
}


template <typename T, typename T2>
__aicore__ inline void L0C2GM_NZ2ND(GlobalTensor<T> dst, LocalTensor<T2> src, int m, int n, int N, int nz_M, uint8_t uflag){
    QuantMode_t q;
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        // copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, uflag, NoQuant, 0, false, true);
        q = NoQuant; 
    }else if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        // copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, uflag, F322F16, 0, false, true);
        q = F322F16;
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        // copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, uflag, F322BF16, 0, false, true);
        q = F322BF16;
    }else{
        // copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, uflag, NoQuant, 0, false, true);
        q = NoQuant;
    }
#ifdef __DAV_C220_CUBE__
    FixpipeParamsV220 fixpipeParams;
    fixpipeParams.nSize = n;
    fixpipeParams.mSize = m;
    fixpipeParams.srcStride = (nz_M+15)/16*16;
    fixpipeParams.dstStride = N;
    fixpipeParams.ndNum = 1;
    fixpipeParams.srcNdStride = 1;
    fixpipeParams.dstNdStride = 1;
    fixpipeParams.quantPre = q;
    Fixpipe(dst, src, fixpipeParams);
#elif __DAV_C310__
    FixpipeParamsC310 fixpipeParams;
    fixpipeParams.nSize = n;
    fixpipeParams.mSize = m;
    fixpipeParams.srcStride = (nz_M+15)/16*16;
    fixpipeParams.dstStride = N;
    fixpipeParams.quantPre = q;
    Fixpipe(dst, src, fixpipeParams);
#endif
}


template <typename T, typename T2>
__aicore__ inline void L0C2UB_NZ2ND(LocalTensor<T> dst, LocalTensor<T2> src, int m, int n, int N, int nz_M, int dualMode, bool subBlkId){
    QuantMode_t q;
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        // copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, uflag, NoQuant, 0, false, true);
        q = NoQuant; 
    }else if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        // copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, uflag, F322F16, 0, false, true);
        q = F322F16;
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        // copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, uflag, F322BF16, 0, false, true);
        q = F322BF16;
    }else{
        // copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, uflag, NoQuant, 0, false, true);
        q = NoQuant;
    }
#ifdef __DAV_C310__
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = n;
    fixpipeParams.mSize = m;
    fixpipeParams.srcStride = (nz_M+15)/16*16;
    fixpipeParams.dstStride = N;
    fixpipeParams.quantPre = q;
    fixpipeParams.dualDstCtl = dualMode;
    fixpipeParams.subBlockId = subBlkId;
    Fixpipe<T, T2, CFG_ROW_MAJOR_UB>(dst, src, fixpipeParams);
#endif
}


template <typename T1, typename T2, typename T3>
__aicore__ inline void MMAD(LocalTensor<T1> dst, LocalTensor<T2> src0, LocalTensor<T3> src1, uint16_t m, uint16_t k, uint16_t n, bool cmatrixInitVal, uint8_t unitFlag){
    MmadParams param;
    param.m = m;
    param.n = n;
    param.k = k;
    param.cmatrixInitVal = cmatrixInitVal;
    // param.unitFlag = unitFlag;
    Mmad(dst, src0, src1, param);
} 


template <typename T>
__aicore__ inline void MERGESORT4(LocalTensor<T> dst, LocalTensor<T> src){
    MrgSort4Info params;
    params.elementLengths[0] = 32;
    params.elementLengths[1] = 32;
    params.elementLengths[2] = 32;
    params.elementLengths[3] = 32;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b1111;
    params.repeatTimes = 1;

    MrgSortSrcList<T> srcList;
    srcList.src1 = src[0];
    srcList.src2 = src[32 * 1 * 2 * 4 / sizeof(T)];
    srcList.src3 = src[32 * 2 * 2 * 4 / sizeof(T)];
    srcList.src4 = src[32 * 3 * 2 * 4 / sizeof(T)];

    MrgSort<T>(dst, srcList, params);
} 

'''