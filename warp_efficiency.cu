// WARP EFFICIENCY ANALYZER — Instrumented BVH4 traversal kernels
//
// KEY DESIGN DECISIONS:
// 1. Use __activemask() instead of __ballot_sync(0xFFFFFFFF,...) for metrics
//    collection. On Volta (sm_70) with Independent Thread Scheduling, threads
//    that broke out of the inner while loop won't participate in full-warp
//    syncs, causing deadlock. __activemask() is safe in divergent contexts.
// 2. Thread-local accumulators flushed once per ray → eliminates per-node
//    atomic contention that caused the first version to hang.
// 3. For child-hit ballots, use the active mask from __activemask().

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <stdint.h>

#define SAH_BINS 16
#define LEAF_SIZE 4
#define SHORT_STACK 12
#define SMEM_BVH4_NODES 128
#define CONST_BVH4 1023

// Metrics buffer layout (unsigned long long each)
enum MetIdx {
    M_AB_1_7=0, M_AB_8_15, M_AB_16_23, M_AB_24_31, M_AB_32,     // 0-4
    M_STK0, M_STK1, M_STK2, M_STK3, M_STK4, M_STK5, M_STK6,    // 5-11
    M_STK7, M_STK8, M_STK9, M_STK10, M_STK11, M_STK12,          // 12-17
    M_CH0, M_CH1, M_CH2, M_CH3, M_CH4,                           // 18-22 child-hit histogram
    M_CB_SUM0, M_CB_SUM1, M_CB_SUM2, M_CB_SUM3,                  // 23-26 child ballot sums
    M_CB_CNT0, M_CB_CNT1, M_CB_CNT2, M_CB_CNT3,                  // 27-30 child ballot counts
    M_NODES, M_WSTEPS, M_SUMACT, M_SUMDEPTH, M_MAXDEPTH, M_NRAYS,
    M_SIZE
};

__constant__ int4 c_bvh4[CONST_BVH4 * 4];
__constant__ int c_bvh4N;
__device__ unsigned int g_rayCounter;
__device__ unsigned int g_rayCounterV11;

struct Tri{float3 v0,v1,v2;};
struct AABB{float3 bmin,bmax;};
struct Hit{float t;int tri;float u,v;};
struct BN{AABB b;int l,r,ts,tc;};
struct BVH4H{float minX[4],minY[4],minZ[4],maxX[4],maxY[4],maxZ[4];int child[4];int nChildren;};

// ═══ CPU BVH BUILD ═══
AABB unite(const AABB&a,const AABB&b){return{{fminf(a.bmin.x,b.bmin.x),fminf(a.bmin.y,b.bmin.y),fminf(a.bmin.z,b.bmin.z)},{fmaxf(a.bmax.x,b.bmax.x),fmaxf(a.bmax.y,b.bmax.y),fmaxf(a.bmax.z,b.bmax.z)}};}
AABB triBox(const Tri&t){return{{fminf(fminf(t.v0.x,t.v1.x),t.v2.x),fminf(fminf(t.v0.y,t.v1.y),t.v2.y),fminf(fminf(t.v0.z,t.v1.z),t.v2.z)},{fmaxf(fmaxf(t.v0.x,t.v1.x),t.v2.x),fmaxf(fmaxf(t.v0.y,t.v1.y),t.v2.y),fmaxf(fmaxf(t.v0.z,t.v1.z),t.v2.z)}};}
float3 triCen(const Tri&t){return{(t.v0.x+t.v1.x+t.v2.x)/3,(t.v0.y+t.v1.y+t.v2.y)/3,(t.v0.z+t.v1.z+t.v2.z)/3};}
float saArea(const AABB&b){float dx=b.bmax.x-b.bmin.x,dy=b.bmax.y-b.bmin.y,dz=b.bmax.z-b.bmin.z;return 2.0f*(dx*dy+dy*dz+dz*dx);}

int buildSAH(BN*n,Tri*t,int*idx,int&nc,int s,int c,int d){
    int ni=nc++;BN&nd=n[ni];AABB bounds=triBox(t[idx[s]]);
    for(int i=1;i<c;i++)bounds=unite(bounds,triBox(t[idx[s+i]]));nd.b=bounds;
    if(c<=LEAF_SIZE||d>28){nd.l=-1;nd.r=-1;nd.ts=s;nd.tc=c;return ni;}
    AABB cb;cb.bmin=cb.bmax=triCen(t[idx[s]]);
    for(int i=1;i<c;i++){float3 cc=triCen(t[idx[s+i]]);
        cb.bmin={fminf(cb.bmin.x,cc.x),fminf(cb.bmin.y,cc.y),fminf(cb.bmin.z,cc.z)};
        cb.bmax={fmaxf(cb.bmax.x,cc.x),fmaxf(cb.bmax.y,cc.y),fmaxf(cb.bmax.z,cc.z)};}
    float bc=FLT_MAX;int ba=-1,bb=-1;float ps=saArea(bounds);
    for(int ax=0;ax<3;ax++){
        float amn=ax==0?cb.bmin.x:ax==1?cb.bmin.y:cb.bmin.z,amx=ax==0?cb.bmax.x:ax==1?cb.bmax.y:cb.bmax.z;
        if(amx-amn<1e-7f)continue;
        struct B{AABB b;int c;};B bins[SAH_BINS];
        for(int i=0;i<SAH_BINS;i++){bins[i].b.bmin={FLT_MAX,FLT_MAX,FLT_MAX};bins[i].b.bmax={-FLT_MAX,-FLT_MAX,-FLT_MAX};bins[i].c=0;}
        float sc=SAH_BINS/(amx-amn);
        for(int i=0;i<c;i++){float3 cc=triCen(t[idx[s+i]]);float cv=ax==0?cc.x:ax==1?cc.y:cc.z;
            int b2=fminf(fmaxf((int)((cv-amn)*sc),0),SAH_BINS-1);bins[b2].b=unite(bins[b2].b,triBox(t[idx[s+i]]));bins[b2].c++;}
        AABB lB[SAH_BINS];int lC[SAH_BINS];lB[0]=bins[0].b;lC[0]=bins[0].c;
        for(int i=1;i<SAH_BINS;i++){lB[i]=unite(lB[i-1],bins[i].b);lC[i]=lC[i-1]+bins[i].c;}
        AABB rB[SAH_BINS];int rC[SAH_BINS];rB[SAH_BINS-1]=bins[SAH_BINS-1].b;rC[SAH_BINS-1]=bins[SAH_BINS-1].c;
        for(int i=SAH_BINS-2;i>=0;i--){rB[i]=unite(rB[i+1],bins[i].b);rC[i]=rC[i+1]+bins[i].c;}
        for(int i=0;i<SAH_BINS-1;i++){if(lC[i]==0||rC[i+1]==0)continue;
            float cost=1.0f+(lC[i]*saArea(lB[i])+rC[i+1]*saArea(rB[i+1]))/ps;
            if(cost<bc){bc=cost;ba=ax;bb=i;}}
    }
    if(ba==-1||bc>(float)c){nd.l=-1;nd.r=-1;nd.ts=s;nd.tc=c;return ni;}
    float amn=ba==0?cb.bmin.x:ba==1?cb.bmin.y:cb.bmin.z,amx=ba==0?cb.bmax.x:ba==1?cb.bmax.y:cb.bmax.z;float sc=SAH_BINS/(amx-amn);
    int i=s,j=s+c-1;
    while(i<=j){float3 cc=triCen(t[idx[i]]);float cv=ba==0?cc.x:ba==1?cc.y:cc.z;
        int b2=fminf(fmaxf((int)((cv-amn)*sc),0),SAH_BINS-1);if(b2<=bb)i++;else{int tmp=idx[i];idx[i]=idx[j];idx[j]=tmp;j--;}}
    int lc=i-s;if(lc==0)lc=1;if(lc==c)lc=c-1;nd.ts=-1;nd.tc=0;
    nd.l=buildSAH(n,t,idx,nc,s,lc,d+1);nd.r=buildSAH(n,t,idx,nc,s+lc,c-lc,d+1);return ni;
}
void treeReorder(BN*src,int nc,BN*dst,int*remap){
    int*stk=(int*)malloc(nc*4);int sp=0,out=0;stk[sp++]=0;
    while(sp>0){int i=stk[--sp];remap[i]=out;dst[out++]=src[i];
        if(src[i].l>=0){stk[sp++]=src[i].r;stk[sp++]=src[i].l;}}
    for(int i=0;i<nc;i++){if(dst[i].l>=0){dst[i].l=remap[dst[i].l];dst[i].r=remap[dst[i].r];}}
    free(stk);}
static int collapseRec(BN*bn,int bIdx,BVH4H*out,int*nOut){
    if(bn[bIdx].l==-1){int ni=(*nOut)++;BVH4H&nd=out[ni];nd.nChildren=1;
        nd.minX[0]=bn[bIdx].b.bmin.x;nd.minY[0]=bn[bIdx].b.bmin.y;nd.minZ[0]=bn[bIdx].b.bmin.z;
        nd.maxX[0]=bn[bIdx].b.bmax.x;nd.maxY[0]=bn[bIdx].b.bmax.y;nd.maxZ[0]=bn[bIdx].b.bmax.z;
        nd.child[0]=-((bn[bIdx].ts<<3)|(bn[bIdx].tc-1))-2;for(int i=1;i<4;i++)nd.child[i]=-1;return ni;}
    int ol[4],oc=0;ol[oc++]=bn[bIdx].l;ol[oc++]=bn[bIdx].r;
    while(oc<4){int bI=-1;float bS=-1;for(int i=0;i<oc;i++){if(bn[ol[i]].l!=-1){float s=saArea(bn[ol[i]].b);if(s>bS){bS=s;bI=i;}}}
        if(bI==-1)break;int bi=ol[bI];ol[bI]=bn[bi].l;ol[oc++]=bn[bi].r;}
    int ni=(*nOut)++;BVH4H&nd=out[ni];nd.nChildren=oc;
    for(int i=0;i<oc;i++){int bi=ol[i];
        nd.minX[i]=bn[bi].b.bmin.x;nd.minY[i]=bn[bi].b.bmin.y;nd.minZ[i]=bn[bi].b.bmin.z;
        nd.maxX[i]=bn[bi].b.bmax.x;nd.maxY[i]=bn[bi].b.bmax.y;nd.maxZ[i]=bn[bi].b.bmax.z;
        if(bn[bi].l==-1)nd.child[i]=-((bn[bi].ts<<3)|(bn[bi].tc-1))-2;
        else nd.child[i]=collapseRec(bn,bi,out,nOut);}
    for(int i=oc;i<4;i++){nd.child[i]=-1;nd.minX[i]=nd.maxX[i]=nd.minY[i]=nd.maxY[i]=nd.minZ[i]=nd.maxZ[i]=0;}
    return ni;}
void reorderBVH4(BVH4H*src,int n4,BVH4H*dst,int*remap){
    int*stk=(int*)malloc(n4*4);int sp=0,out=0;stk[sp++]=0;
    while(sp>0){int i=stk[--sp];remap[i]=out;dst[out++]=src[i];
        for(int c=src[i].nChildren-1;c>=0;c--)if(src[i].child[c]>=0)stk[sp++]=src[i].child[c];}
    for(int i=0;i<n4;i++)for(int c=0;c<4;c++)if(dst[i].child[c]>=0)dst[i].child[c]=remap[dst[i].child[c]];
    free(stk);}
void packBVH4GPU(BVH4H*nodes,int n4,int4*gpuData){
    float eps=5e-4f;
    for(int i=0;i<n4;i++){BVH4H&nd=nodes[i];
        unsigned short hMnX[4],hMnY[4],hMnZ[4],hMxX[4],hMxY[4],hMxZ[4];
        for(int c=0;c<4;c++){if(nd.child[c]==-1){hMnX[c]=hMnY[c]=hMnZ[c]=0x7C00;hMxX[c]=hMxY[c]=hMxZ[c]=0xFC00;}
            else{__half h;h=__float2half(nd.minX[c]-eps);memcpy(&hMnX[c],&h,2);h=__float2half(nd.minY[c]-eps);memcpy(&hMnY[c],&h,2);
                h=__float2half(nd.minZ[c]-eps);memcpy(&hMnZ[c],&h,2);h=__float2half(nd.maxX[c]+eps);memcpy(&hMxX[c],&h,2);
                h=__float2half(nd.maxY[c]+eps);memcpy(&hMxY[c],&h,2);h=__float2half(nd.maxZ[c]+eps);memcpy(&hMxZ[c],&h,2);}}
        gpuData[i*4+0]=make_int4((hMnX[1]<<16)|hMnX[0],(hMnX[3]<<16)|hMnX[2],(hMxX[1]<<16)|hMxX[0],(hMxX[3]<<16)|hMxX[2]);
        gpuData[i*4+1]=make_int4((hMnY[1]<<16)|hMnY[0],(hMnY[3]<<16)|hMnY[2],(hMxY[1]<<16)|hMxY[0],(hMxY[3]<<16)|hMxY[2]);
        gpuData[i*4+2]=make_int4((hMnZ[1]<<16)|hMnZ[0],(hMnZ[3]<<16)|hMnZ[2],(hMxZ[1]<<16)|hMxZ[0],(hMxZ[3]<<16)|hMxZ[2]);
        gpuData[i*4+3]=make_int4(nd.child[0],nd.child[1],nd.child[2],nd.child[3]);}}

// ═══ DEVICE HELPERS ═══
__device__ __forceinline__ void d4h(int lo,int hi,float&v0,float&v1,float&v2,float&v3){
    v0=__half2float(__ushort_as_half((unsigned short)(lo)));v1=__half2float(__ushort_as_half((unsigned short)(lo>>16)));
    v2=__half2float(__ushort_as_half((unsigned short)(hi)));v3=__half2float(__ushort_as_half((unsigned short)(hi>>16)));}
__device__ __forceinline__ void cswap(float&a,float&b,int&ia,int&ib){
    bool s=a>b;float tf=s?a:b;a=s?b:a;b=tf;int ti=s?ia:ib;ia=s?ib:ia;ib=ti;}
__device__ __forceinline__ float warpMin(float v){
    v=fminf(v,__shfl_xor_sync(0xFFFFFFFF,v,16));
    v=fminf(v,__shfl_xor_sync(0xFFFFFFFF,v,8));
    v=fminf(v,__shfl_xor_sync(0xFFFFFFFF,v,4));
    v=fminf(v,__shfl_xor_sync(0xFFFFFFFF,v,2));
    v=fminf(v,__shfl_xor_sync(0xFFFFFFFF,v,1));
    return v;}

// Per-ray metrics struct to minimize register pressure
struct RayMetrics {
    int nodeVisits;   // total internal node visits
    int maxDepth;     // max stack depth reached
    // The 5 active-thread bins + 5 child-hit bins + 4 child ballot sums/counts
    // are too many registers. We use a compact encoding:
    // Accumulate just: sum of active counts, count of full-32 steps,
    // count of low (<16) steps, and child hit sums.
    int warpSteps;    // lane0: total warp steps
    int sumActive;    // lane0: sum of active thread counts
    int full32;       // lane0: count of steps where all 32 active
    int low1_15;      // lane0: count of steps where <16 active
    int childHits[5]; // lane0: histogram of 0..4 children hit
};

// ═══ INSTRUMENTED V10 KERNEL ═══
extern __shared__ int4 s_bvh4[];

__global__ void __launch_bounds__(256,3) traceV10_instr(
    const int4*__restrict__ d_bvh4,int n4,int smN,
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    const float*__restrict__ rix,const float*__restrict__ riy,const float*__restrict__ riz,
    Hit*__restrict__ hits,int numRays,
    unsigned long long*__restrict__ met)
{
    int smN4=min(n4,smN);
    for(int i=threadIdx.x;i<smN4*4;i+=blockDim.x) s_bvh4[i]=d_bvh4[i];
    __syncthreads();
    int cnst=c_bvh4N, lane=threadIdx.x&31;

    while(true){
        int bs; if(lane==0) bs=atomicAdd(&g_rayCounter,32);
        bs=__shfl_sync(0xFFFFFFFF,bs,0);
        if(bs>=numRays) break;
        int ri=bs+lane;
        if(ri>=numRays) continue;

        float ox=rox[ri],oy=roy[ri],oz=roz[ri];
        float dx2=rdx[ri],dy2=rdy[ri],dz2=rdz[ri];
        float ix=rix[ri],iy=riy[ri],iz=riz[ri];
        float hitT=1e30f; int hitTri=-1; float hitU=0,hitV=0;
        int stk[SHORT_STACK]; int sp=0, ni=0;

        // Thread-local metrics
        int myMaxDepth=0, myNodes=0;
        int myWarpSteps=0, mySumActive=0, myFull32=0, myLow=0;
        int myCH[5]={0,0,0,0,0};

        while(true){
            if(ni==-1){if(sp>0)ni=stk[--sp];else break;continue;}
            if(ni<=-2){int val=-(ni+2),tc=(val&7)+1,ts=val>>3;
                for(int i=0;i<tc;i++){int ti=ts+i;
                    float v0x=tv0x[ti],v0y=tv0y[ti],v0z=tv0z[ti];
                    float e1x=tv1x[ti]-v0x,e1y=tv1y[ti]-v0y,e1z=tv1z[ti]-v0z;
                    float e2x=tv2x[ti]-v0x,e2y=tv2y[ti]-v0y,e2z=tv2z[ti]-v0z;
                    float hx=dy2*e2z-dz2*e2y,hy=dz2*e2x-dx2*e2z,hz=dx2*e2y-dy2*e2x;
                    float a2=e1x*hx+e1y*hy+e1z*hz;if(fabsf(a2)<1e-8f)continue;
                    float f=__frcp_rn(a2);float sx=ox-v0x,sy=oy-v0y,sz=oz-v0z;
                    float u=f*(sx*hx+sy*hy+sz*hz);if(u<0.f||u>1.f)continue;
                    float qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x;
                    float v=f*(dx2*qx+dy2*qy+dz2*qz);if(v<0.f||u+v>1.f)continue;
                    float tt=f*(e2x*qx+e2y*qy+e2z*qz);
                    if(tt>0.001f&&tt<hitT){hitT=tt;hitTri=ti;hitU=u;hitV=v;}}
                if(sp>0)ni=stk[--sp];else break;continue;}

            // ── INTERNAL NODE: metrics ──
            // Use __activemask() — safe in divergent code on Volta
            unsigned amask = __activemask();
            int ac = __popc(amask);
            myNodes++;
            if(sp > myMaxDepth) myMaxDepth = sp;

            // Lane 0 of each warp tracks warp-level active count
            if(lane==0){
                myWarpSteps++;
                mySumActive += ac;
                if(ac==32) myFull32++;
                if(ac<16)  myLow++;
            }

            // Load BVH4 node
            int4 bx,by,bz,ch;
            if(ni<smN4){bx=s_bvh4[ni*4];by=s_bvh4[ni*4+1];bz=s_bvh4[ni*4+2];ch=s_bvh4[ni*4+3];}
            else if(ni<cnst){bx=c_bvh4[ni*4];by=c_bvh4[ni*4+1];bz=c_bvh4[ni*4+2];ch=c_bvh4[ni*4+3];}
            else{bx=__ldg(&d_bvh4[ni*4]);by=__ldg(&d_bvh4[ni*4+1]);bz=__ldg(&d_bvh4[ni*4+2]);ch=__ldg(&d_bvh4[ni*4+3]);}

            float mn0x,mn1x,mn2x,mn3x,mx0x,mx1x,mx2x,mx3x;
            float mn0y,mn1y,mn2y,mn3y,mx0y,mx1y,mx2y,mx3y;
            float mn0z,mn1z,mn2z,mn3z,mx0z,mx1z,mx2z,mx3z;
            d4h(bx.x,bx.y,mn0x,mn1x,mn2x,mn3x);d4h(bx.z,bx.w,mx0x,mx1x,mx2x,mx3x);
            d4h(by.x,by.y,mn0y,mn1y,mn2y,mn3y);d4h(by.z,by.w,mx0y,mx1y,mx2y,mx3y);
            d4h(bz.x,bz.y,mn0z,mn1z,mn2z,mn3z);d4h(bz.z,bz.w,mx0z,mx1z,mx2z,mx3z);

            float t0n,t0x,t1n,t1x,t2n,t2x,t3n,t3x;
            {float a=(mn0x-ox)*ix,b=(mx0x-ox)*ix;t0n=fminf(a,b);t0x=fmaxf(a,b);a=(mn0y-oy)*iy;b=(mx0y-oy)*iy;t0n=fmaxf(t0n,fminf(a,b));t0x=fminf(t0x,fmaxf(a,b));a=(mn0z-oz)*iz;b=(mx0z-oz)*iz;t0n=fmaxf(t0n,fminf(a,b));t0x=fminf(t0x,fmaxf(a,b));}
            {float a=(mn1x-ox)*ix,b=(mx1x-ox)*ix;t1n=fminf(a,b);t1x=fmaxf(a,b);a=(mn1y-oy)*iy;b=(mx1y-oy)*iy;t1n=fmaxf(t1n,fminf(a,b));t1x=fminf(t1x,fmaxf(a,b));a=(mn1z-oz)*iz;b=(mx1z-oz)*iz;t1n=fmaxf(t1n,fminf(a,b));t1x=fminf(t1x,fmaxf(a,b));}
            {float a=(mn2x-ox)*ix,b=(mx2x-ox)*ix;t2n=fminf(a,b);t2x=fmaxf(a,b);a=(mn2y-oy)*iy;b=(mx2y-oy)*iy;t2n=fmaxf(t2n,fminf(a,b));t2x=fminf(t2x,fmaxf(a,b));a=(mn2z-oz)*iz;b=(mx2z-oz)*iz;t2n=fmaxf(t2n,fminf(a,b));t2x=fminf(t2x,fmaxf(a,b));}
            {float a=(mn3x-ox)*ix,b=(mx3x-ox)*ix;t3n=fminf(a,b);t3x=fmaxf(a,b);a=(mn3y-oy)*iy;b=(mx3y-oy)*iy;t3n=fmaxf(t3n,fminf(a,b));t3x=fminf(t3x,fmaxf(a,b));a=(mn3z-oz)*iz;b=(mx3z-oz)*iz;t3n=fmaxf(t3n,fminf(a,b));t3x=fminf(t3x,fmaxf(a,b));}

            int c0=ch.x,c1=ch.y,c2=ch.z,c3=ch.w;
            bool h0=(c0!=-1)&&(t0x>=fmaxf(t0n,0.f))&&(t0n<hitT);
            bool h1=(c1!=-1)&&(t1x>=fmaxf(t1n,0.f))&&(t1n<hitT);
            bool h2=(c2!=-1)&&(t2x>=fmaxf(t2n,0.f))&&(t2n<hitT);
            bool h3=(c3!=-1)&&(t3x>=fmaxf(t3n,0.f))&&(t3n<hitT);
            int nh=h0+h1+h2+h3;

            // Child ballot using active mask
            unsigned b0=__ballot_sync(amask, h0);
            unsigned b1=__ballot_sync(amask, h1);
            unsigned b2=__ballot_sync(amask, h2);
            unsigned b3=__ballot_sync(amask, h3);

            if(lane==0){
                int wch=(__popc(b0)>0)+(__popc(b1)>0)+(__popc(b2)>0)+(__popc(b3)>0);
                myCH[wch]++;
            }

            if(nh==0){if(sp>0)ni=stk[--sp];else break;continue;}
            float d4[4]={h0?t0n:1e30f,h1?t1n:1e30f,h2?t2n:1e30f,h3?t3n:1e30f};
            int ci[4]={c0,c1,c2,c3};
            cswap(d4[0],d4[1],ci[0],ci[1]);cswap(d4[2],d4[3],ci[2],ci[3]);
            cswap(d4[0],d4[2],ci[0],ci[2]);cswap(d4[1],d4[3],ci[1],ci[3]);cswap(d4[1],d4[2],ci[1],ci[2]);
            for(int i=nh-1;i>=1&&sp<SHORT_STACK;i--)stk[sp++]=ci[i];
            ni=ci[0]; continue;
        }

        // ── FLUSH per ray ──
        hits[ri].t=hitT; hits[ri].tri=hitTri; hits[ri].u=hitU; hits[ri].v=hitV;
        int db = myMaxDepth < SHORT_STACK ? myMaxDepth : SHORT_STACK;
        atomicAdd(&met[M_STK0 + db], 1ULL);
        atomicAdd(&met[M_SUMDEPTH], (unsigned long long)myMaxDepth);
        atomicMax((unsigned long long*)&met[M_MAXDEPTH], (unsigned long long)myMaxDepth);
        atomicAdd(&met[M_NRAYS], 1ULL);
        atomicAdd(&met[M_NODES], (unsigned long long)myNodes);
        if(lane==0){
            atomicAdd(&met[M_WSTEPS], (unsigned long long)myWarpSteps);
            atomicAdd(&met[M_SUMACT], (unsigned long long)mySumActive);
            atomicAdd(&met[M_AB_32], (unsigned long long)myFull32);
            int mid = myWarpSteps - myFull32 - myLow;
            atomicAdd(&met[M_AB_16_23], (unsigned long long)(mid > 0 ? mid : 0));
            atomicAdd(&met[M_AB_1_7], (unsigned long long)myLow);
            for(int k=0;k<5;k++) if(myCH[k]) atomicAdd(&met[M_CH0+k], (unsigned long long)myCH[k]);
        }
    }
}

// ═══ INSTRUMENTED V11 KERNEL ═══
__global__ void __launch_bounds__(256,3) traceV11_instr(
    const int4*__restrict__ d_bvh4,int n4,int smN,
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    const float*__restrict__ rix,const float*__restrict__ riy,const float*__restrict__ riz,
    Hit*__restrict__ hits,int numRays,
    unsigned long long*__restrict__ met)
{
    int smN4=min(n4,smN);
    for(int i=threadIdx.x;i<smN4*4;i+=blockDim.x) s_bvh4[i]=d_bvh4[i];
    __syncthreads();
    int cnst=c_bvh4N, lane=threadIdx.x&31;

    while(true){
        int bs; if(lane==0) bs=atomicAdd(&g_rayCounterV11,32);
        bs=__shfl_sync(0xFFFFFFFF,bs,0);
        if(bs>=numRays) break;
        int ri=bs+lane;
        if(ri>=numRays) continue;

        float ox=rox[ri],oy=roy[ri],oz=roz[ri];
        float dx2=rdx[ri],dy2=rdy[ri],dz2=rdz[ri];
        float ix=rix[ri],iy=riy[ri],iz=riz[ri];
        float hitT=1e30f; int hitTri=-1; float hitU=0,hitV=0;
        int stk[SHORT_STACK]; int sp=0, ni=0;

        int myMaxDepth=0, myNodes=0;
        int myWarpSteps=0, mySumActive=0, myFull32=0, myLow=0;
        int myCH[5]={0,0,0,0,0};

        while(true){
            if(ni==-1){if(sp>0)ni=stk[--sp];else break;continue;}
            if(ni<=-2){int val=-(ni+2),tc=(val&7)+1,ts=val>>3;
                for(int i=0;i<tc;i++){int ti=ts+i;
                    float v0x=tv0x[ti],v0y=tv0y[ti],v0z=tv0z[ti];
                    float e1x=tv1x[ti]-v0x,e1y=tv1y[ti]-v0y,e1z=tv1z[ti]-v0z;
                    float e2x=tv2x[ti]-v0x,e2y=tv2y[ti]-v0y,e2z=tv2z[ti]-v0z;
                    float hx=dy2*e2z-dz2*e2y,hy=dz2*e2x-dx2*e2z,hz=dx2*e2y-dy2*e2x;
                    float a2=e1x*hx+e1y*hy+e1z*hz;if(fabsf(a2)<1e-8f)continue;
                    float f=__frcp_rn(a2);float sx=ox-v0x,sy=oy-v0y,sz=oz-v0z;
                    float u=f*(sx*hx+sy*hy+sz*hz);if(u<0.f||u>1.f)continue;
                    float qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x;
                    float v=f*(dx2*qx+dy2*qy+dz2*qz);if(v<0.f||u+v>1.f)continue;
                    float tt=f*(e2x*qx+e2y*qy+e2z*qz);
                    if(tt>0.001f&&tt<hitT){hitT=tt;hitTri=ti;hitU=u;hitV=v;}}
                if(sp>0)ni=stk[--sp];else break;continue;}

            unsigned amask = __activemask();
            int ac = __popc(amask);
            myNodes++;
            if(sp > myMaxDepth) myMaxDepth = sp;

            if(lane==0){
                myWarpSteps++;
                mySumActive += ac;
                if(ac==32) myFull32++;
                if(ac<16)  myLow++;
            }

            int4 bx,by,bz,ch;
            if(ni<smN4){bx=s_bvh4[ni*4];by=s_bvh4[ni*4+1];bz=s_bvh4[ni*4+2];ch=s_bvh4[ni*4+3];}
            else if(ni<cnst){bx=c_bvh4[ni*4];by=c_bvh4[ni*4+1];bz=c_bvh4[ni*4+2];ch=c_bvh4[ni*4+3];}
            else{bx=__ldg(&d_bvh4[ni*4]);by=__ldg(&d_bvh4[ni*4+1]);bz=__ldg(&d_bvh4[ni*4+2]);ch=__ldg(&d_bvh4[ni*4+3]);}

            float mn0x,mn1x,mn2x,mn3x,mx0x,mx1x,mx2x,mx3x;
            float mn0y,mn1y,mn2y,mn3y,mx0y,mx1y,mx2y,mx3y;
            float mn0z,mn1z,mn2z,mn3z,mx0z,mx1z,mx2z,mx3z;
            d4h(bx.x,bx.y,mn0x,mn1x,mn2x,mn3x);d4h(bx.z,bx.w,mx0x,mx1x,mx2x,mx3x);
            d4h(by.x,by.y,mn0y,mn1y,mn2y,mn3y);d4h(by.z,by.w,mx0y,mx1y,mx2y,mx3y);
            d4h(bz.x,bz.y,mn0z,mn1z,mn2z,mn3z);d4h(bz.z,bz.w,mx0z,mx1z,mx2z,mx3z);

            float t0n,t0x2,t1n,t1x2,t2n,t2x2,t3n,t3x2;
            {float a=(mn0x-ox)*ix,b=(mx0x-ox)*ix;t0n=fminf(a,b);t0x2=fmaxf(a,b);a=(mn0y-oy)*iy;b=(mx0y-oy)*iy;t0n=fmaxf(t0n,fminf(a,b));t0x2=fminf(t0x2,fmaxf(a,b));a=(mn0z-oz)*iz;b=(mx0z-oz)*iz;t0n=fmaxf(t0n,fminf(a,b));t0x2=fminf(t0x2,fmaxf(a,b));}
            {float a=(mn1x-ox)*ix,b=(mx1x-ox)*ix;t1n=fminf(a,b);t1x2=fmaxf(a,b);a=(mn1y-oy)*iy;b=(mx1y-oy)*iy;t1n=fmaxf(t1n,fminf(a,b));t1x2=fminf(t1x2,fmaxf(a,b));a=(mn1z-oz)*iz;b=(mx1z-oz)*iz;t1n=fmaxf(t1n,fminf(a,b));t1x2=fminf(t1x2,fmaxf(a,b));}
            {float a=(mn2x-ox)*ix,b=(mx2x-ox)*ix;t2n=fminf(a,b);t2x2=fmaxf(a,b);a=(mn2y-oy)*iy;b=(mx2y-oy)*iy;t2n=fmaxf(t2n,fminf(a,b));t2x2=fminf(t2x2,fmaxf(a,b));a=(mn2z-oz)*iz;b=(mx2z-oz)*iz;t2n=fmaxf(t2n,fminf(a,b));t2x2=fminf(t2x2,fmaxf(a,b));}
            {float a=(mn3x-ox)*ix,b=(mx3x-ox)*ix;t3n=fminf(a,b);t3x2=fmaxf(a,b);a=(mn3y-oy)*iy;b=(mx3y-oy)*iy;t3n=fmaxf(t3n,fminf(a,b));t3x2=fminf(t3x2,fmaxf(a,b));a=(mn3z-oz)*iz;b=(mx3z-oz)*iz;t3n=fmaxf(t3n,fminf(a,b));t3x2=fminf(t3x2,fmaxf(a,b));}

            int c0=ch.x,c1=ch.y,c2=ch.z,c3=ch.w;
            bool h0=(c0!=-1)&&(t0x2>=fmaxf(t0n,0.f))&&(t0n<hitT);
            bool h1=(c1!=-1)&&(t1x2>=fmaxf(t1n,0.f))&&(t1n<hitT);
            bool h2=(c2!=-1)&&(t2x2>=fmaxf(t2n,0.f))&&(t2n<hitT);
            bool h3=(c3!=-1)&&(t3x2>=fmaxf(t3n,0.f))&&(t3n<hitT);

            // Use amask for ballot (safe in divergent code)
            unsigned bb0=__ballot_sync(amask,h0);
            unsigned bb1=__ballot_sync(amask,h1);
            unsigned bb2=__ballot_sync(amask,h2);
            unsigned bb3=__ballot_sync(amask,h3);

            if(lane==0){
                int wch=(__popc(bb0)>0)+(__popc(bb1)>0)+(__popc(bb2)>0)+(__popc(bb3)>0);
                myCH[wch]++;
            }

            // V11 warp-coherent ordering (uses amask-based ballots)
            unsigned m0=bb0,m1=bb1,m2=bb2,m3=bb3;
            int warpHits=(m0?1:0)+(m1?1:0)+(m2?1:0)+(m3?1:0);
            if(warpHits==0){if(sp>0)ni=stk[--sp];else break;continue;}

            // warpMin needs full-warp sync; use amask version
            float myT0 = h0?t0n:1e30f, myT1 = h1?t1n:1e30f;
            float myT2 = h2?t2n:1e30f, myT3 = h3?t3n:1e30f;
            // Warp-min using active mask shuffles
            float w0=1e30f,w1=1e30f,w2=1e30f,w3=1e30f;
            if(m0){w0=myT0;for(int d=16;d>=1;d>>=1)w0=fminf(w0,__shfl_xor_sync(amask,w0,d));}
            if(m1){w1=myT1;for(int d=16;d>=1;d>>=1)w1=fminf(w1,__shfl_xor_sync(amask,w1,d));}
            if(m2){w2=myT2;for(int d=16;d>=1;d>>=1)w2=fminf(w2,__shfl_xor_sync(amask,w2,d));}
            if(m3){w3=myT3;for(int d=16;d>=1;d>>=1)w3=fminf(w3,__shfl_xor_sync(amask,w3,d));}

            float wd[4]={w0,w1,w2,w3}; int wci[4]={c0,c1,c2,c3};
            cswap(wd[0],wd[1],wci[0],wci[1]);cswap(wd[2],wd[3],wci[2],wci[3]);
            cswap(wd[0],wd[2],wci[0],wci[2]);cswap(wd[1],wd[3],wci[1],wci[3]);cswap(wd[1],wd[2],wci[1],wci[2]);
            for(int i=warpHits-1;i>=1&&sp<SHORT_STACK;i--)stk[sp++]=wci[i];
            ni=wci[0]; continue;
        }

        hits[ri].t=hitT; hits[ri].tri=hitTri; hits[ri].u=hitU; hits[ri].v=hitV;
        int db = myMaxDepth < SHORT_STACK ? myMaxDepth : SHORT_STACK;
        atomicAdd(&met[M_STK0 + db], 1ULL);
        atomicAdd(&met[M_SUMDEPTH], (unsigned long long)myMaxDepth);
        atomicMax((unsigned long long*)&met[M_MAXDEPTH], (unsigned long long)myMaxDepth);
        atomicAdd(&met[M_NRAYS], 1ULL);
        atomicAdd(&met[M_NODES], (unsigned long long)myNodes);
        if(lane==0){
            atomicAdd(&met[M_WSTEPS], (unsigned long long)myWarpSteps);
            atomicAdd(&met[M_SUMACT], (unsigned long long)mySumActive);
            atomicAdd(&met[M_AB_32], (unsigned long long)myFull32);
            int mid = myWarpSteps - myFull32 - myLow;
            atomicAdd(&met[M_AB_16_23], (unsigned long long)(mid > 0 ? mid : 0));
            atomicAdd(&met[M_AB_1_7], (unsigned long long)myLow);
            for(int k=0;k<5;k++) if(myCH[k]) atomicAdd(&met[M_CH0+k], (unsigned long long)myCH[k]);
        }
    }
}

// ═══ SCENE + SORTING ═══
void addQuad(Tri*t,int&ti,float3 a,float3 b,float3 c,float3 d){t[ti++]={a,b,c};t[ti++]={a,c,d};}
void addBox(Tri*t,int&ti,float3 mn,float3 mx){
    float3 a={mn.x,mn.y,mn.z},b={mx.x,mn.y,mn.z},cv={mx.x,mx.y,mn.z},d={mn.x,mx.y,mn.z};
    float3 e={mn.x,mn.y,mx.z},f={mx.x,mn.y,mx.z},g={mx.x,mx.y,mx.z},h={mn.x,mx.y,mx.z};
    addQuad(t,ti,a,b,cv,d);addQuad(t,ti,e,f,g,h);addQuad(t,ti,a,b,f,e);addQuad(t,ti,d,cv,g,h);addQuad(t,ti,a,d,h,e);addQuad(t,ti,b,cv,g,f);}
void addSubQuad(Tri*t,int&ti,float3 o,float3 ux,float3 uy,int nx,int ny){
    for(int i=0;i<nx;i++)for(int j=0;j<ny;j++){float u0=(float)i/nx,u1=(float)(i+1)/nx,v0=(float)j/ny,v1=(float)(j+1)/ny;
        float3 a={o.x+ux.x*u0+uy.x*v0,o.y+ux.y*u0+uy.y*v0,o.z+ux.z*u0+uy.z*v0};
        float3 b={o.x+ux.x*u1+uy.x*v0,o.y+ux.y*u1+uy.y*v0,o.z+ux.z*u1+uy.z*v0};
        float3 cv={o.x+ux.x*u1+uy.x*v1,o.y+ux.y*u1+uy.y*v1,o.z+ux.z*u1+uy.z*v1};
        float3 d2={o.x+ux.x*u0+uy.x*v1,o.y+ux.y*u0+uy.y*v1,o.z+ux.z*u0+uy.z*v1};
        t[ti++]={a,b,cv};t[ti++]={a,cv,d2};}}
int genConference(Tri*t,int maxTris){
    int ti=0;float W=10,H=5,D=7.5f;int subdiv=(int)sqrtf((float)maxTris/60);
    if(subdiv<2)subdiv=2;if(subdiv>200)subdiv=200;
    addSubQuad(t,ti,{-W,0,-D},{2*W,0,0},{0,0,2*D},subdiv,subdiv);
    addSubQuad(t,ti,{-W,H,-D},{2*W,0,0},{0,0,2*D},subdiv,subdiv);
    addSubQuad(t,ti,{-W,0,-D},{2*W,0,0},{0,H,0},subdiv,subdiv/2);
    addSubQuad(t,ti,{-W,0,D},{2*W,0,0},{0,H,0},subdiv,subdiv/2);
    addSubQuad(t,ti,{-W,0,-D},{0,0,2*D},{0,H,0},subdiv,subdiv/2);
    addSubQuad(t,ti,{W,0,-D},{0,0,2*D},{0,H,0},subdiv,subdiv/2);
    srand(42);int numT=maxTris>50000?20:8;
    for(int i=0;i<numT&&ti+12<maxTris;i++){float tx=((float)rand()/RAND_MAX)*16-8,tz=((float)rand()/RAND_MAX)*12-6;
        addBox(t,ti,{tx-1.f,.7f,tz-.5f},{tx+1.f,.8f,tz+.5f});addBox(t,ti,{tx-.9f,0,tz-.05f},{tx-.8f,.7f,tz+.05f});addBox(t,ti,{tx+.8f,0,tz-.05f},{tx+.9f,.7f,tz+.05f});}
    int numC=maxTris>50000?40:16;
    for(int i=0;i<numC&&ti+12<maxTris;i++){float cx=((float)rand()/RAND_MAX)*18-9,cz=((float)rand()/RAND_MAX)*14-7;
        addBox(t,ti,{cx-.25f,.4f,cz-.25f},{cx+.25f,.45f,cz+.25f});addBox(t,ti,{cx-.25f,.45f,cz-.25f},{cx+.25f,.9f,cz-.2f});}
    while(ti+2<maxTris){float cx=((float)rand()/RAND_MAX)*16-8,cy=.8f+((float)rand()/RAND_MAX)*.3f;float cz=((float)rand()/RAND_MAX)*12-6,s=.05f+((float)rand()/RAND_MAX)*.1f;
        t[ti].v0={cx-s,cy,cz-s};t[ti].v1={cx+s,cy,cz+s};t[ti].v2={cx,cy+s*2,cz};ti++;
        t[ti].v0={cx-s,cy,cz+s};t[ti].v1={cx+s,cy,cz-s};t[ti].v2={cx,cy+s*2,cz};ti++;}
    return ti;}

struct RayAoS{float3 o,d,id;};
int octant(float3 d){return (d.x<0?4:0)|(d.y<0?2:0)|(d.z<0?1:0);}
static inline uint32_t expand3(uint32_t v){v&=0x3FF;v=(v|(v<<16))&0x30000FF;v=(v|(v<<8))&0x300F00F;v=(v|(v<<4))&0x30C30C3;v=(v|(v<<2))&0x9249249;return v;}
uint32_t morton3D(float x,float y,float z,float3 mn,float3 mx){
    float nx=(x-mn.x)/(mx.x-mn.x+1e-7f),ny=(y-mn.y)/(mx.y-mn.y+1e-7f),nz=(z-mn.z)/(mx.z-mn.z+1e-7f);
    uint32_t ix2=fminf(fmaxf(nx*1023.f,0.f),1023.f),iy2=fminf(fmaxf(ny*1023.f,0.f),1023.f),iz2=fminf(fmaxf(nz*1023.f,0.f),1023.f);
    return expand3(ix2)|(expand3(iy2)<<1)|(expand3(iz2)<<2);}
void sortMortonOctant(RayAoS*r,int n,float3 smn,float3 smx){
    struct SK{uint32_t key;int idx;};SK*keys=(SK*)malloc(n*sizeof(SK));
    for(int i=0;i<n;i++){uint32_t m=morton3D(r[i].o.x,r[i].o.y,r[i].o.z,smn,smx);uint32_t o=(uint32_t)octant(r[i].d);keys[i]={(o<<27)|(m>>3),i};}
    std::sort(keys,keys+n,[](const SK&a,const SK&b){return a.key<b.key;});
    RayAoS*tmp=(RayAoS*)malloc(n*sizeof(RayAoS));for(int i=0;i<n;i++)tmp[i]=r[keys[i].idx];
    memcpy(r,tmp,n*sizeof(RayAoS));free(tmp);free(keys);}

void genPrimary(RayAoS*r,int n){int w=(int)sqrtf((float)n);
    for(int i=0;i<n;i++){int px=i%w,py=i/w;float u=(2.f*px/w-1.f)*1.2f,v=(2.f*py/(n/w)-1.f)*.6f;
        r[i].o={0,2.5f,12};float3 d={u,-v,-1.5f};float l=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);
        d.x/=l;d.y/=l;d.z/=l;r[i].d=d;r[i].id={1.f/d.x,1.f/d.y,1.f/d.z};}}
void genDiffuse(RayAoS*r,int n,Tri*tris,int nt){srand(1337);
    for(int i=0;i<n;i++){int ti=rand()%nt;Tri&tr=tris[ti];
        float u=((float)rand()/RAND_MAX),v=((float)rand()/RAND_MAX);if(u+v>1){u=1-u;v=1-v;}
        float3 o={tr.v0.x+u*(tr.v1.x-tr.v0.x)+v*(tr.v2.x-tr.v0.x),tr.v0.y+u*(tr.v1.y-tr.v0.y)+v*(tr.v2.y-tr.v0.y),tr.v0.z+u*(tr.v1.z-tr.v0.z)+v*(tr.v2.z-tr.v0.z)};
        float th=acosf(sqrtf((float)rand()/RAND_MAX)),ph=2.f*3.14159f*((float)rand()/RAND_MAX);
        float3 d={sinf(th)*cosf(ph),fabsf(cosf(th))+0.01f,sinf(th)*sinf(ph)};
        float l=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);d.x/=l;d.y/=l;d.z/=l;
        o.x+=d.x*0.002f;o.y+=d.y*0.002f;o.z+=d.z*0.002f;
        r[i].o=o;r[i].d=d;r[i].id={1.f/d.x,1.f/d.y,1.f/d.z};}}

// ═══ HARNESS ═══
typedef void(*IKFn)(const int4*,int,int,
    const float*,const float*,const float*,const float*,const float*,const float*,
    const float*,const float*,const float*,const float*,const float*,const float*,
    const float*,const float*,const float*,const float*,const float*,const float*,
    Hit*,int,unsigned long long*);

struct MR { unsigned long long m[M_SIZE]; int hitPct; };

MR collect(IKFn kern, bool v11ctr, RayAoS*rays, int nR, int4*d_bvh4, int n4, float*d_v[9], cudaDeviceProp&prop){
    float*h_ray[9]; for(int j=0;j<9;j++) h_ray[j]=(float*)malloc(nR*4);
    for(int i=0;i<nR;i++){
        h_ray[0][i]=rays[i].o.x;h_ray[1][i]=rays[i].o.y;h_ray[2][i]=rays[i].o.z;
        h_ray[3][i]=rays[i].d.x;h_ray[4][i]=rays[i].d.y;h_ray[5][i]=rays[i].d.z;
        h_ray[6][i]=rays[i].id.x;h_ray[7][i]=rays[i].id.y;h_ray[8][i]=rays[i].id.z;}
    float*d_ray[9]; Hit*d_hits; unsigned long long*d_met;
    for(int j=0;j<9;j++){cudaMalloc(&d_ray[j],nR*4);cudaMemcpy(d_ray[j],h_ray[j],nR*4,cudaMemcpyHostToDevice);}
    cudaMalloc(&d_hits,nR*sizeof(Hit));
    cudaMalloc(&d_met,M_SIZE*sizeof(unsigned long long));
    cudaMemset(d_met,0,M_SIZE*sizeof(unsigned long long));
    int nb=prop.multiProcessorCount*4, smB=SMEM_BVH4_NODES*4*sizeof(int4);
    unsigned int z=0;
    if(v11ctr) cudaMemcpyToSymbol(g_rayCounterV11,&z,4);
    else cudaMemcpyToSymbol(g_rayCounter,&z,4);
    kern<<<nb,256,smB>>>(d_bvh4,n4,SMEM_BVH4_NODES,
        d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
        d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],
        d_hits,nR,d_met);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("  CUDA ERROR: %s\n", cudaGetErrorString(err));
    }
    MR res;
    cudaMemcpy(res.m, d_met, M_SIZE*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    Hit*hh=(Hit*)malloc(nR*sizeof(Hit));
    cudaMemcpy(hh,d_hits,nR*sizeof(Hit),cudaMemcpyDeviceToHost);
    int hc=0; for(int i=0;i<nR;i++) if(hh[i].tri>=0) hc++;
    res.hitPct=100*hc/nR;
    for(int j=0;j<9;j++){cudaFree(d_ray[j]);free(h_ray[j]);}
    cudaFree(d_hits);cudaFree(d_met);free(hh);
    return res;
}

void printReport(const char* label, const MR& r){
    unsigned long long tws = r.m[M_WSTEPS];
    unsigned long long sa  = r.m[M_SUMACT];
    double avg = tws > 0 ? (double)sa/tws : 0;

    // Active thread bins: we stored full32, low(<16), and mid=total-full32-low
    unsigned long long f32 = r.m[M_AB_32];
    unsigned long long low = r.m[M_AB_1_7];  // actually <16
    unsigned long long mid = r.m[M_AB_16_23]; // 16..31 (non-full)
    unsigned long long total = f32 + low + mid;
    double pF32 = total>0 ? 100.0*f32/total : 0;
    double pMid = total>0 ? 100.0*mid/total : 0;
    double pLow = total>0 ? 100.0*low/total : 0;

    unsigned long long nR = r.m[M_NRAYS];
    double avgD = nR > 0 ? (double)r.m[M_SUMDEPTH]/nR : 0;
    unsigned long long maxD = r.m[M_MAXDEPTH];

    unsigned long long tn = 0;
    for(int i=0;i<5;i++) tn += r.m[M_CH0+i];
    double chPct[5];
    for(int i=0;i<5;i++) chPct[i] = tn > 0 ? 100.0*r.m[M_CH0+i]/tn : 0;

    printf("  %s (hit%%=%d):\n", label, r.hitPct);
    printf("    Active threads/warp step:  32/32: %5.1f%%  16-31: %5.1f%%  1-15: %5.1f%%\n",
        pF32, pMid, pLow);
    printf("    Avg active threads: %.1f/32 = %.1f%%\n", avg, avg/32.0*100.0);
    printf("    Total warp-steps: %llu  (%llu rays)\n", tws, nR);
    printf("    Avg nodes/ray: %.1f\n", nR>0?(double)r.m[M_NODES]/nR:0.0);
    printf("    Max stack depth: avg %.1f, max %llu\n", avgD, maxD);

    printf("    Stack depth histogram:");
    for(int i=0;i<=SHORT_STACK;i++){
        unsigned long long v=r.m[M_STK0+i];
        if(v>0) printf(" [%d]=%llu",i,v);
    }
    printf("\n");

    printf("    Child hits/node (warp): 0: %5.1f%%  1: %5.1f%%  2: %5.1f%%  3: %5.1f%%  4: %5.1f%%\n",
        chPct[0], chPct[1], chPct[2], chPct[3], chPct[4]);
    printf("\n");
}

int main(){
    printf("==========================================================================\n");
    printf("  WARP EFFICIENCY ANALYZER — Instrumented BVH4 Traversal\n");
    printf("  Uses: __activemask(), __ballot_sync(amask,...), thread-local accum\n");
    printf("==========================================================================\n\n");

    cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
    printf("  GPU: %s  SMs: %d  Arch: sm_%d%d\n\n",
        prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    float3 sMn={-10,0,-7.5f},sMx={10,5,7.5f};
    int maxTris=100000, numRays=1048576;

    Tri*h_tris=(Tri*)malloc(maxTris*sizeof(Tri));int nt=genConference(h_tris,maxTris);
    BN*h_nodes=(BN*)calloc(nt*2,sizeof(BN));int*tidx=(int*)malloc(nt*4);
    for(int i=0;i<nt;i++)tidx[i]=i;int nc=0;buildSAH(h_nodes,h_tris,tidx,nc,0,nt,0);
    BN*h_ord=(BN*)malloc(nc*sizeof(BN));int*remap=(int*)malloc(nc*4);treeReorder(h_nodes,nc,h_ord,remap);
    Tri*h_to=(Tri*)malloc(nt*sizeof(Tri));for(int i=0;i<nt;i++)h_to[i]=h_tris[tidx[i]];
    BVH4H*h_b4=(BVH4H*)calloc(nc,sizeof(BVH4H));int n4=0;collapseRec(h_ord,0,h_b4,&n4);
    BVH4H*h_b4o=(BVH4H*)malloc(n4*sizeof(BVH4H));int*r4=(int*)malloc(n4*4);reorderBVH4(h_b4,n4,h_b4o,r4);
    int4*h_gpu=(int4*)malloc(n4*4*sizeof(int4));packBVH4GPU(h_b4o,n4,h_gpu);
    int cn4=n4<CONST_BVH4?n4:CONST_BVH4;
    cudaMemcpyToSymbol(c_bvh4,h_gpu,cn4*4*sizeof(int4));cudaMemcpyToSymbol(c_bvh4N,&cn4,4);
    int4*d_bvh4;cudaMalloc(&d_bvh4,n4*4*sizeof(int4));cudaMemcpy(d_bvh4,h_gpu,n4*4*sizeof(int4),cudaMemcpyHostToDevice);
    float*h_v[9];for(int j=0;j<9;j++)h_v[j]=(float*)malloc(nt*4);
    for(int i=0;i<nt;i++){h_v[0][i]=h_to[i].v0.x;h_v[1][i]=h_to[i].v0.y;h_v[2][i]=h_to[i].v0.z;
        h_v[3][i]=h_to[i].v1.x;h_v[4][i]=h_to[i].v1.y;h_v[5][i]=h_to[i].v1.z;
        h_v[6][i]=h_to[i].v2.x;h_v[7][i]=h_to[i].v2.y;h_v[8][i]=h_to[i].v2.z;}
    float*d_v[9];for(int j=0;j<9;j++){cudaMalloc(&d_v[j],nt*4);cudaMemcpy(d_v[j],h_v[j],nt*4,cudaMemcpyHostToDevice);}

    printf("  Scene: %dK tris | %d BVH4 nodes | %.2f MB BVH | %dK rays\n\n",
        nt/1000, n4, n4*64/(1024.f*1024.f), numRays/1024);

    RayAoS*rays=(RayAoS*)malloc(numRays*sizeof(RayAoS));
    unsigned int z=0;

    printf("══════════════════════════════════════════════════════════════\n");
    printf("=== WARP EFFICIENCY REPORT ===\n");
    printf("══════════════════════════════════════════════════════════════\n\n");

    // Collect all 6 configs
    struct Config { const char* label; bool v11; bool morton; bool primary; };
    Config cfgs[6] = {
        {"PRIMARY RAYS V10 (1M, coherent)", false, false, true},
        {"PRIMARY RAYS V11 (1M, coherent)", true,  false, true},
        {"DIFFUSE UNSORTED V10 (1M)",       false, false, false},
        {"DIFFUSE UNSORTED V11 (1M)",       true,  false, false},
        {"DIFFUSE MORTON-SORTED V10 (1M)",  false, true,  false},
        {"DIFFUSE MORTON-SORTED V11 (1M)",  true,  true,  false},
    };
    MR results[6];

    for(int c=0;c<6;c++){
        printf("  Running: %s ...\n", cfgs[c].label);
        fflush(stdout);
        if(cfgs[c].primary) genPrimary(rays,numRays);
        else genDiffuse(rays,numRays,h_to,nt);
        if(cfgs[c].morton) sortMortonOctant(rays,numRays,sMn,sMx);
        if(cfgs[c].v11){
            cudaMemcpyToSymbol(g_rayCounterV11,&z,4);
            results[c] = collect(traceV11_instr, true, rays, numRays, d_bvh4, n4, d_v, prop);
        } else {
            cudaMemcpyToSymbol(g_rayCounter,&z,4);
            results[c] = collect(traceV10_instr, false, rays, numRays, d_bvh4, n4, d_v, prop);
        }
    }
    printf("\n");

    for(int c=0;c<6;c++) printReport(cfgs[c].label, results[c]);

    // ═══ SUMMARY TABLE ═══
    printf("══════════════════════════════════════════════════════════════\n");
    printf("=== SUMMARY COMPARISON ===\n");
    printf("══════════════════════════════════════════════════════════════\n\n");

    printf("  %-35s  %10s  %8s  %8s\n", "Config", "Avg Active", "Warp Eff", "%%32/32");
    printf("  %-35s  %10s  %8s  %8s\n", "-----------------------------------", "----------", "--------", "--------");
    for(int c=0;c<6;c++){
        unsigned long long tws=results[c].m[M_WSTEPS], sa=results[c].m[M_SUMACT];
        double avg = tws>0 ? (double)sa/tws : 0;
        unsigned long long f32=results[c].m[M_AB_32], lo=results[c].m[M_AB_1_7], mi=results[c].m[M_AB_16_23];
        unsigned long long tot=f32+lo+mi;
        double p32 = tot>0 ? 100.0*f32/tot : 0;
        printf("  %-35s  %6.1f/32   %5.1f%%    %5.1f%%\n",
            cfgs[c].label, avg, avg/32.0*100.0, p32);
    }
    printf("\n");

    printf("  V11 vs V10 warp efficiency delta:\n");
    for(int i=0;i<6;i+=2){
        double a0 = results[i].m[M_WSTEPS]>0 ? (double)results[i].m[M_SUMACT]/results[i].m[M_WSTEPS] : 0;
        double a1 = results[i+1].m[M_WSTEPS]>0 ? (double)results[i+1].m[M_SUMACT]/results[i+1].m[M_WSTEPS] : 0;
        printf("    %-25s  %.1f → %.1f  (%+.1f%%)\n",
            i==0?"Primary":i==2?"Diffuse unsorted":"Diffuse morton",
            a0, a1, a0>0?(a1/a0-1)*100:0);
    }
    printf("\n");

    for(int j=0;j<9;j++){cudaFree(d_v[j]);free(h_v[j]);}
    cudaFree(d_bvh4);free(h_tris);free(h_to);free(h_nodes);free(h_ord);free(remap);free(tidx);
    free(h_b4);free(h_b4o);free(r4);free(h_gpu);free(rays);
    return 0;
}
