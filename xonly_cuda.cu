#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <stdexcept>
#include <exception>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "secp_core.cuh"

static const unsigned FULL_WARP = 0xffffffffu;

struct HData { PointJ point; FE x; };
struct LaneState {
    FE prevX;       // affine x((k-S)G)
    FE curX;        // affine x(kG)
    U96 curK;       // scalar corresponding to curX
    uint32_t hasPrev;
    uint32_t hasCur;
};
struct Best {
    uint32_t valid;
    uint32_t distHi;
    uint64_t distLo;
    U96 scalar;
    FE x;
};
struct Prefix80 { uint32_t hi; uint64_t lo; };

__device__ __forceinline__ FE shfl_up_fe(const FE &a, int delta) {
    FE r;
    #pragma unroll
    for (int i=0;i<8;i++) r.v[i]=__shfl_up_sync(FULL_WARP,a.v[i],delta,32);
    return r;
}
__device__ __forceinline__ FE shfl_down_fe(const FE &a, int delta) {
    FE r;
    #pragma unroll
    for (int i=0;i<8;i++) r.v[i]=__shfl_down_sync(FULL_WARP,a.v[i],delta,32);
    return r;
}
__device__ __forceinline__ FE shfl_fe(const FE &a, int src) {
    FE r;
    #pragma unroll
    for (int i=0;i<8;i++) r.v[i]=__shfl_sync(FULL_WARP,a.v[i],src,32);
    return r;
}

// Montgomery batch inversion over one complete warp: one field inversion
// produces 32 individual inverses. Inputs must be nonzero.
__device__ __forceinline__ FE warp_batch_inverse(const FE &value) {
    const int lane = threadIdx.x & 31;
    FE prefix=value;
    #pragma unroll
    for (int delta=1; delta<32; delta<<=1) {
        FE other=shfl_up_fe(prefix,delta);
        if (lane>=delta) prefix=fe_mul(prefix,other);
    }
    FE total=shfl_fe(prefix,31);
    FE invTotal=fe_one();
    if (lane==0) invTotal=fe_inv(total);
    invTotal=shfl_fe(invTotal,0);

    FE suffix=value;
    #pragma unroll
    for (int delta=1; delta<32; delta<<=1) {
        FE other=shfl_down_fe(suffix,delta);
        if (lane+delta<32) suffix=fe_mul(suffix,other);
    }
    FE before=(lane==0)?fe_one():shfl_up_fe(prefix,1);
    FE after=(lane==31)?fe_one():shfl_down_fe(suffix,1);
    return fe_mul(invTotal,fe_mul(before,after));
}

__device__ __forceinline__ Prefix80 get_prefix80(const FE &x) {
    Prefix80 r;
    r.hi=x.v[7];
    r.lo=((uint64_t)x.v[6]<<16) | (uint64_t)(x.v[5]>>16);
    return r;
}
__device__ __forceinline__ void prefix_distance(Prefix80 p, Prefix80 t,
                                                 uint32_t *dh, uint64_t *dl) {
    bool ge=(p.hi>t.hi) || (p.hi==t.hi && p.lo>=t.lo);
    if (ge) {
        uint64_t borrow=(p.lo<t.lo);
        *dl=(p.lo-t.lo)&0x0000FFFFFFFFFFFFULL;
        *dh=p.hi-t.hi-(uint32_t)borrow;
    } else {
        uint64_t borrow=(t.lo<p.lo);
        *dl=(t.lo-p.lo)&0x0000FFFFFFFFFFFFULL;
        *dh=t.hi-p.hi-(uint32_t)borrow;
    }
}
__device__ __forceinline__ bool scalar_less(U96 a,U96 b) { return u96_lt(a,b); }
__device__ __forceinline__ void consider(Best &best, const FE &x, U96 scalar, Prefix80 target) {
    Prefix80 p=get_prefix80(x); uint32_t dh; uint64_t dl;
    prefix_distance(p,target,&dh,&dl);
    bool better=!best.valid || dh<best.distHi ||
        (dh==best.distHi && (dl<best.distLo || (dl==best.distLo && scalar_less(scalar,best.scalar))));
    if (better) { best.valid=1;best.distHi=dh;best.distLo=dl;best.scalar=scalar;best.x=x; }
}

__global__ void make_stride_point(uint64_t stride, HData *hdata) {
    if (blockIdx.x==0 && threadIdx.x==0) {
        U96 s={stride,0};
        PointJ H=point_mul_g(s);
        FE z2=fe_sqr(H.Z);
        FE hx=fe_mul(H.X,fe_inv(z2));
        hdata[0].point=H;
        hdata[0].x=hx;
    }
}

__global__ void initialize_lanes(LaneState *states, Best *bests, const HData *hdata,
                                 U96 L, U96 U, uint64_t stride, uint32_t threads) {
    uint32_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=threads) return;
    U96 k0=u96_add64(L,tid);
    U96 k1=u96_add64(k0,stride);
    bool has0=u96_le(k0,U), has1=u96_le(k1,U);
    FE xPrevProj=fe_zero(), zPrev=fe_one();
    FE xCurProj=fe_zero(), zCur=fe_one();
    if (has0) {
        PointJ cur=point_mul_g(k1);
        PointJ negH=hdata[0].point;
        negH.Y=fe_neg(negH.Y);
        PointJ prev=point_add(cur,negH);
        xPrevProj=prev.X; zPrev=fe_sqr(prev.Z);
        xCurProj=cur.X;  zCur=fe_sqr(cur.Z);
    }
    // Normalize initial X coordinates with warp-batched inversions.
    FE invPrev=warp_batch_inverse(zPrev);
    FE prevX=fe_mul(xPrevProj,invPrev);
    FE invCur=warp_batch_inverse(zCur);
    FE curX=fe_mul(xCurProj,invCur);
    LaneState st;
    st.prevX=prevX; st.curX=curX; st.curK=k1;
    st.hasPrev=has0?1u:0u; st.hasCur=has1?1u:0u;
    states[tid]=st;
    Best b={}; bests[tid]=b;
}

__global__ void search_chunk(LaneState *states, Best *bests, const HData *hdata,
                             U96 L, U96 U, Prefix80 target, uint64_t stride,
                             uint32_t threads, uint32_t rounds, uint32_t firstChunk) {
    uint32_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=threads) return;
    LaneState st=states[tid];
    Best best=bests[tid];
    FE a=hdata[0].x;
    FE fourteen=fe_zero(); fourteen.v[0]=14;

    if (firstChunk) {
        U96 k0=u96_sub64(st.curK,stride);
        if (st.hasPrev) consider(best,st.prevX,k0,target);
        if (st.hasCur) consider(best,st.curX,st.curK,target);
    }

    for (uint32_t step=0; step<rounds; step++) {
        U96 nextK=u96_add64(st.curK,stride);
        bool active=st.hasCur && u96_le(nextK,U);
        if (!__any_sync(FULL_WARP,active)) break;

        FE invD=fe_one();
        FE d=fe_one();
        if (active) d=fe_sub(st.curX,a);
        invD=warp_batch_inverse(d);
        if (active) {
            FE invD2=fe_sqr(invD);
            FE n=fe_add(fe_mul(fe_mul(a,st.curX),fe_add(st.curX,a)),fourteen);
            FE nextX=fe_sub(fe_double(fe_mul(n,invD2)),st.prevX);
            consider(best,nextX,nextK,target);
            st.prevX=st.curX;
            st.curX=nextX;
            st.curK=nextK;
        }
    }
    states[tid]=st;
    bests[tid]=best;
}

static unsigned __int128 parse_u128(const std::string &s, bool hexAllowed=true) {
    if (s.empty()) throw std::runtime_error("empty integer");
    size_t i=0; unsigned base=10;
    if (hexAllowed && s.size()>2 && s[0]=='0' && (s[1]=='x'||s[1]=='X')) {base=16;i=2;}
    unsigned __int128 v=0;
    const unsigned __int128 umax=~(unsigned __int128)0;
    if (i==s.size()) throw std::runtime_error("missing digits: "+s);
    for (;i<s.size();i++) {
        unsigned d;
        char c=s[i];
        if(c>='0'&&c<='9') d=c-'0';
        else if(base==16&&c>='a'&&c<='f') d=c-'a'+10;
        else if(base==16&&c>='A'&&c<='F') d=c-'A'+10;
        else throw std::runtime_error("invalid integer: "+s);
        if(d>=base) throw std::runtime_error("invalid integer: "+s);
        if(v>(umax-d)/base) throw std::runtime_error("integer exceeds 128-bit range: "+s);
        v=v*base+d;
    }
    return v;
}
static std::string u128_dec(unsigned __int128 x) {
    if(!x) return "0"; std::string s;
    while(x){s.push_back(char('0'+x%10));x/=10;} std::reverse(s.begin(),s.end()); return s;
}
static std::string u128_hex(unsigned __int128 x) {
    if(!x) return "0"; const char *d="0123456789abcdef"; std::string s;
    while(x){s.push_back(d[(unsigned)(x&15)]);x>>=4;} std::reverse(s.begin(),s.end()); return s;
}
static U96 to_u96(unsigned __int128 x) { U96 r={(uint64_t)x,(uint32_t)(x>>64)}; return r; }
static unsigned __int128 from_u96(U96 x) { return ((unsigned __int128)x.hi<<64)|x.lo; }
static void cuda_check(cudaError_t e,const char *where) {
    if(e!=cudaSuccess){std::cerr<<where<<": "<<cudaGetErrorString(e)<<"\n";std::exit(2);}
}
static bool host_best_less(const Best &a,const Best &b) {
    if(!a.valid)return false; if(!b.valid)return true;
    if(a.distHi!=b.distHi)return a.distHi<b.distHi;
    if(a.distLo!=b.distLo)return a.distLo<b.distLo;
    return from_u96(a.scalar)<from_u96(b.scalar);
}
static Prefix80 host_get_prefix80(const FE &x) { Prefix80 r; r.hi=x.v[7]; r.lo=((uint64_t)x.v[6]<<16)|(x.v[5]>>16); return r; }
static std::string fe_hex(const FE &x) {
    char buf[65]; std::snprintf(buf,sizeof(buf),"%08x%08x%08x%08x%08x%08x%08x%08x",
        x.v[7],x.v[6],x.v[5],x.v[4],x.v[3],x.v[2],x.v[1],x.v[0]); return std::string(buf);
}
static void usage(const char *p) {
    std::cerr<<"Usage: "<<p<<" [--L N] [--U N] [--target 20_HEX] [--threads N] [--chunk N] [--max-rounds N] [--selftest]\n"
             <<"Scans secp256k1 scalars in [L,U], comparing the top 80 X bits numerically.\n";
}

int main(int argc,char **argv) {
    unsigned __int128 L=1, U=((unsigned __int128)1<<80)-1;
    unsigned __int128 targetValue=0;
    std::string targetText="03000000000000000000";
    uint32_t threads=65536, chunk=256;
    bool selftest=false, threadsExplicit=false, quiet=false, maxRoundsSet=false;
    unsigned __int128 maxRounds=0;
    try {
        for(int i=1;i<argc;i++){
            std::string a=argv[i];
            auto value=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};
            if(a=="--L")L=parse_u128(value());
            else if(a=="--U")U=parse_u128(value());
            else if(a=="--target")targetText=value();
            else if(a=="--threads"){
                unsigned __int128 v=parse_u128(value());
                if(v>1048576) throw std::runtime_error("--threads must be <= 1048576");
                threads=(uint32_t)v; threadsExplicit=true;
            }
            else if(a=="--chunk"){
                unsigned __int128 v=parse_u128(value());
                if(v>0xffffffffu) throw std::runtime_error("--chunk is too large");
                chunk=(uint32_t)v;
            }
            else if(a=="--max-rounds"){maxRounds=parse_u128(value());maxRoundsSet=true;}
            else if(a=="--selftest")selftest=true;
            else if(a=="--quiet")quiet=true;
            else if(a=="--help"){usage(argv[0]);return 0;}
            else throw std::runtime_error("unknown option: "+a);
        }
        targetValue=parse_u128("0x"+targetText);
    } catch(const std::exception &e){std::cerr<<"Argument error: "<<e.what()<<"\n";usage(argv[0]);return 2;}

    if(selftest){
        L=1; U=10000; targetText="03000000000000000000";
        targetValue=parse_u128("0x"+targetText);
        if(!threadsExplicit)threads=256;
        maxRoundsSet=false;
    }
    const unsigned __int128 limit=((unsigned __int128)1<<80)-1;
    if(L<1 || U<L || U>limit || targetValue>limit || targetText.size()!=20 || threads<128 || (threads%128)!=0 || chunk==0){
        std::cerr<<"Invalid parameters: require 1<=L<=U<2^80, a 20-hex-digit target, threads>=128 multiple of 128, chunk>=1.\n";return 2;
    }

    Prefix80 target={(uint32_t)(targetValue>>48),(uint64_t)(targetValue&((((unsigned __int128)1)<<48)-1))};
    int device=0; cuda_check(cudaGetDevice(&device),"cudaGetDevice");
    cudaDeviceProp prop; cuda_check(cudaGetDeviceProperties(&prop,device),"cudaGetDeviceProperties");
    if(!quiet) std::cout<<"GPU: "<<prop.name<<" | SMs="<<prop.multiProcessorCount<<" | threads="<<threads<<" | stride="<<threads<<"\n";

    HData *d_h=nullptr; LaneState *d_states=nullptr; Best *d_bests=nullptr;
    cuda_check(cudaMalloc((void**)&d_h,sizeof(HData)),"cudaMalloc H");
    cuda_check(cudaMalloc((void**)&d_states,(size_t)threads*sizeof(LaneState)),"cudaMalloc states");
    cuda_check(cudaMalloc((void**)&d_bests,(size_t)threads*sizeof(Best)),"cudaMalloc bests");
    make_stride_point<<<1,1>>>(threads,d_h);
    cuda_check(cudaGetLastError(),"make_stride_point launch");
    cuda_check(cudaDeviceSynchronize(),"make_stride_point sync");
    U96 Lu=to_u96(L), Uu=to_u96(U);
    initialize_lanes<<<threads/128,128>>>(d_states,d_bests,d_h,Lu,Uu,threads,threads);
    cuda_check(cudaGetLastError(),"initialize_lanes launch");
    cuda_check(cudaDeviceSynchronize(),"initialize_lanes sync");

    unsigned __int128 total=U-L+1;
    unsigned __int128 maxIndex=(total-1)/threads;
    unsigned __int128 roundsTotal=(maxIndex>1)?maxIndex-1:0;
    unsigned __int128 roundsToRun=roundsTotal;
    if(maxRoundsSet && maxRounds<roundsToRun) roundsToRun=maxRounds;
    unsigned __int128 remaining=roundsToRun, roundsDone=0;
    bool first=true, foundZero=false;
    std::vector<Best> hostBest(threads);
    Best globalBest={};
    const auto searchStart=std::chrono::steady_clock::now();
    do {
        uint32_t now=(remaining>chunk)?chunk:(uint32_t)remaining;
        search_chunk<<<threads/128,128>>>(d_states,d_bests,d_h,Lu,Uu,target,threads,threads,now,first?1u:0u);
        cuda_check(cudaGetLastError(),"search_chunk launch");
        cuda_check(cudaDeviceSynchronize(),"search_chunk sync");
        cuda_check(cudaMemcpy(hostBest.data(),d_bests,(size_t)threads*sizeof(Best),cudaMemcpyDeviceToHost),"copy best results");
        globalBest={};
        for(uint32_t i=0;i<threads;i++) if(host_best_less(hostBest[i],globalBest)) globalBest=hostBest[i];
        if(globalBest.valid && globalBest.distHi==0 && globalBest.distLo==0){foundZero=true;}
        if(!quiet){
            unsigned __int128 doneCount=(unsigned __int128)threads*(2+roundsDone+now);
            if(doneCount>total)doneCount=total;
            std::cout<<"scanned="<<u128_dec(doneCount)<<"/"<<u128_dec(total);
            if(globalBest.valid) std::cout<<" best_distance="<<u128_dec(((unsigned __int128)globalBest.distHi<<48)|globalBest.distLo)<<" scalar="<<u128_dec(from_u96(globalBest.scalar));
            const auto nowTime=std::chrono::steady_clock::now();
            double elapsed=std::chrono::duration<double>(nowTime-searchStart).count();
            double rate=elapsed>0.0?(double)doneCount/elapsed/1.0e6:0.0;
            std::cout<<" rate="<<std::fixed<<std::setprecision(2)<<rate<<" Mpoints/s elapsed="<<std::setprecision(2)<<elapsed<<" s\n";
        }
        first=false;
        roundsDone+=now;
        if(remaining>=now) remaining-=now; else remaining=0;
        if(foundZero || remaining==0) break;
    } while(true);

    if(globalBest.valid){
        // Final host-side full-point verification, independent of the X-only recurrence.
        PointJ verifyPoint=point_mul_g(globalBest.scalar);
        if(point_isinf(verifyPoint)){
            std::cerr<<"verification failed: scalar mapped to infinity\n";
            cudaFree(d_bests); cudaFree(d_states); cudaFree(d_h); return 3;
        }
        FE verifyX=fe_mul(verifyPoint.X,fe_inv(fe_sqr(verifyPoint.Z)));
        if(!fe_equal(verifyX,globalBest.x)){
            std::cerr<<"verification failed: full scalar multiplication disagrees with search X\n";
            cudaFree(d_bests); cudaFree(d_states); cudaFree(d_h); return 3;
        }
        unsigned __int128 dist=((unsigned __int128)globalBest.distHi<<48)|globalBest.distLo;
        std::cout<<"best_scalar="<<u128_dec(from_u96(globalBest.scalar))<<" (0x"<<u128_hex(from_u96(globalBest.scalar))<<")\n";
        std::cout<<"x="<<fe_hex(globalBest.x)<<"\n";
        unsigned __int128 prefixValue=((unsigned __int128)host_get_prefix80(globalBest.x).hi<<48)|host_get_prefix80(globalBest.x).lo;
        std::string prefixHex=u128_hex(prefixValue);
        std::cout<<"prefix="<<std::string(prefixHex.size()<20?20-prefixHex.size():0,'0')<<prefixHex<<"\n";
        std::cout<<"distance="<<u128_dec(dist)<<" (0x"<<u128_hex(dist)<<")\n";
        std::cout<<"verification=PASS (host full-point multiplication)\n";
    } else std::cout<<"no candidates\n";

    unsigned __int128 scanned=(unsigned __int128)threads*(2+roundsDone);
    if(scanned>total)scanned=total;
    if(foundZero) std::cout<<"coverage=EARLY STOP; distance zero proves global optimality\n";
    else if(scanned==total && roundsDone==roundsTotal) std::cout<<"coverage=COMPLETE; global minimum proven\n";
    else std::cout<<"coverage=PARTIAL; best among scanned candidates only\n";

    if(selftest){
        const char *expectedX="02fd360f1680b7b65f49f64d5a648e8c97e4a756fe88e10791fce9b6295cfc85";
        const unsigned __int128 expectedDist=parse_u128("0x2c9f0e97f4849a0b7");
        if(!globalBest.valid || from_u96(globalBest.scalar)!=9325 || fe_hex(globalBest.x)!=expectedX ||
           (((unsigned __int128)globalBest.distHi<<48)|globalBest.distLo)!=expectedDist || scanned!=total){
            std::cerr<<"SELFTEST FAIL\n";return 1;
        }
        std::cout<<"SELFTEST PASS: GPU scan matches the independently checked reference vector\n";
    }
    cudaFree(d_bests);cudaFree(d_states);cudaFree(d_h);
    return 0;
}
