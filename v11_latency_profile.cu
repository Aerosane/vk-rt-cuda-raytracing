// V11 Per-Stage Latency Profiler
// Measures exact time breakdown of each stage in the BVH4 traversal pipeline.
// Each stage isolated into a micro-kernel running 1M iterations per thread,
// timed with cudaEvents. Uses real scene data from v11's conference generator.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <stdint.h>

// ═══ Config ═══
#define NUM_RAYS       (4*1024*1024)
#define LOOP_ITERS     4096
#define LEAF_SIZE      4
#define SAH_BINS       16
#define SHORT_STACK    12
#define SMEM_BVH4_NODES 128
#define CONST_BVH4     1023
#define BLK            256

// ═══ Data structures (from v11) ═══
struct Tri  { float3 v0, v1, v2; };
struct AABB { float3 bmin, bmax; };
struct Hit  { float t; int tri; float u, v; };
struct BN   { AABB b; int l, r, ts, tc; };
struct BVH4H{ float minX[4],minY[4],minZ[4],maxX[4],maxY[4],maxZ[4]; int child[4]; int nChildren; };
struct RayAoS { float3 o, d, id; };

__constant__ int4 c_bvh4[CONST_BVH4 * 4];
__constant__ int  c_bvh4N;

// ═══ Scene generation (copied from cuda_rt_v11.cu) ═══
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

void addQuad(Tri*t,int&ti,float3 a,float3 b,float3 c,float3 d){t[ti++]={a,b,c};t[ti++]={a,c,d};}
void addBox(Tri*t,int&ti,float3 mn,float3 mx){
    float3 a={mn.x,mn.y,mn.z},b={mx.x,mn.y,mn.z},cv={mx.x,mx.y,mn.z},d={mn.x,mx.y,mn.z};
    float3 e={mn.x,mn.y,mx.z},f={mx.x,mn.y,mx.z},g={mx.x,mx.y,mx.z},h={mn.x,mx.y,mx.z};
    addQuad(t,ti,a,b,cv,d);addQuad(t,ti,e,f,g,h);addQuad(t,ti,a,b,f,e);addQuad(t,ti,d,cv,g,h);addQuad(t,ti,a,d,h,e);addQuad(t,ti,b,cv,g,f);}
void addSubQuad(Tri*t,int&ti,float3 o,float3 ux,float3 uy,int nx,int ny){
    for(int i=0;i<nx;i++)for(int j=0;j<ny;j++){float u0=(float)i/nx,u1=(float)(i+1)/nx,v0=(float)j/ny,v1=(float)(j+1)/ny;
        float3 a={o.x+ux.x*u0+uy.x*v0,o.y+ux.y*u0+uy.y*v0,o.z+ux.z*u0+uy.z*v0};
        float3 b={o.x+ux.x*u1+uy.x*v0,o.y+ux.y*u1+uy.y*v0,o.z+ux.z*u1+uy.z*v0};
        float3 cv2={o.x+ux.x*u1+uy.x*v1,o.y+ux.y*u1+uy.y*v1,o.z+ux.z*u1+uy.z*v1};
        float3 d2={o.x+ux.x*u0+uy.x*v1,o.y+ux.y*u0+uy.y*v1,o.z+ux.z*u0+uy.z*v1};
        t[ti++]={a,b,cv2};t[ti++]={a,cv2,d2};}}
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
void genDiffuse(RayAoS*r,int n,Tri*tris,int nt){srand(1337);
    for(int i=0;i<n;i++){int ti=rand()%nt;Tri&tr=tris[ti];
        float u=((float)rand()/RAND_MAX),v=((float)rand()/RAND_MAX);if(u+v>1){u=1-u;v=1-v;}
        float3 o={tr.v0.x+u*(tr.v1.x-tr.v0.x)+v*(tr.v2.x-tr.v0.x),tr.v0.y+u*(tr.v1.y-tr.v0.y)+v*(tr.v2.y-tr.v0.y),tr.v0.z+u*(tr.v1.z-tr.v0.z)+v*(tr.v2.z-tr.v0.z)};
        float th=acosf(sqrtf((float)rand()/RAND_MAX)),ph=2.f*3.14159f*((float)rand()/RAND_MAX);
        float3 d={sinf(th)*cosf(ph),fabsf(cosf(th))+0.01f,sinf(th)*sinf(ph)};
        float l=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);d.x/=l;d.y/=l;d.z/=l;
        o.x+=d.x*0.002f;o.y+=d.y*0.002f;o.z+=d.z*0.002f;
        r[i].o=o;r[i].d=d;r[i].id={1.f/d.x,1.f/d.y,1.f/d.z};}}

// ═══════════════════════════════════════════════════════════════════════════
// STAGE 1: BVH Node Fetch — 4 x int4 loads per node (64 bytes)
// Sequential: BFS order (cache-friendly). Random: hash-based (cache-hostile).
// ═══════════════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(BLK,4) bench_fetch_sequential(
    const int4* __restrict__ d_bvh4, int n4, int iters, float* __restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int it = 0; it < iters; it++) {
        int ni = (tid + it * 37) % n4;  // sequential-ish: stride through BFS
        int4 bx = __ldg(&d_bvh4[ni*4]);
        int4 by = __ldg(&d_bvh4[ni*4+1]);
        int4 bz = __ldg(&d_bvh4[ni*4+2]);
        int4 ch = __ldg(&d_bvh4[ni*4+3]);
        sum += __int_as_float(bx.x ^ by.y ^ bz.z ^ ch.w);
    }
    out[tid] = sum;
}

__global__ void __launch_bounds__(BLK,4) bench_fetch_random(
    const int4* __restrict__ d_bvh4, int n4, int iters, float* __restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    unsigned h = tid * 2654435761u;
    for (int it = 0; it < iters; it++) {
        h ^= h << 13; h ^= h >> 17; h ^= h << 5;  // xorshift
        int ni = h % n4;
        int4 bx = __ldg(&d_bvh4[ni*4]);
        int4 by = __ldg(&d_bvh4[ni*4+1]);
        int4 bz = __ldg(&d_bvh4[ni*4+2]);
        int4 ch = __ldg(&d_bvh4[ni*4+3]);
        sum += __int_as_float(bx.x ^ by.y ^ bz.z ^ ch.w);
    }
    out[tid] = sum;
}

// Shared memory fetch path (like v11 for top nodes)
extern __shared__ int4 s_bvh4_prof[];
__global__ void __launch_bounds__(BLK,4) bench_fetch_smem(
    const int4* __restrict__ d_bvh4, int n4, int smN, int iters, float* __restrict__ out)
{
    int smN4 = min(n4, smN);
    for (int i = threadIdx.x; i < smN4*4; i += blockDim.x) s_bvh4_prof[i] = d_bvh4[i];
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int it = 0; it < iters; it++) {
        int ni = (tid + it * 37) % smN4;
        int4 bx = s_bvh4_prof[ni*4];
        int4 by = s_bvh4_prof[ni*4+1];
        int4 bz = s_bvh4_prof[ni*4+2];
        int4 ch = s_bvh4_prof[ni*4+3];
        sum += __int_as_float(bx.x ^ by.y ^ bz.z ^ ch.w);
    }
    out[tid] = sum;
}

// Constant memory fetch path
__global__ void __launch_bounds__(BLK,4) bench_fetch_const(
    int cnst, int iters, float* __restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int it = 0; it < iters; it++) {
        int ni = (tid + it * 37) % cnst;
        int4 bx = c_bvh4[ni*4];
        int4 by = c_bvh4[ni*4+1];
        int4 bz = c_bvh4[ni*4+2];
        int4 ch = c_bvh4[ni*4+3];
        sum += __int_as_float(bx.x ^ by.y ^ bz.z ^ ch.w);
    }
    out[tid] = sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// STAGE 2: FP16 Decode — __ushort_as_half → float for 24 values per node
// ═══════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void d4h(int lo, int hi, float&v0, float&v1, float&v2, float&v3) {
    v0 = __half2float(__ushort_as_half((unsigned short)(lo)));
    v1 = __half2float(__ushort_as_half((unsigned short)(lo>>16)));
    v2 = __half2float(__ushort_as_half((unsigned short)(hi)));
    v3 = __half2float(__ushort_as_half((unsigned short)(hi>>16)));
}

__global__ void __launch_bounds__(BLK,4) bench_fp16_decode(
    const int4* __restrict__ d_bvh4, int n4, int iters, float* __restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int it = 0; it < iters; it++) {
        int ni = (tid + it) % n4;
        int4 bx = __ldg(&d_bvh4[ni*4]);
        int4 by = __ldg(&d_bvh4[ni*4+1]);
        int4 bz = __ldg(&d_bvh4[ni*4+2]);
        float mn0x,mn1x,mn2x,mn3x,mx0x,mx1x,mx2x,mx3x;
        float mn0y,mn1y,mn2y,mn3y,mx0y,mx1y,mx2y,mx3y;
        float mn0z,mn1z,mn2z,mn3z,mx0z,mx1z,mx2z,mx3z;
        d4h(bx.x,bx.y,mn0x,mn1x,mn2x,mn3x); d4h(bx.z,bx.w,mx0x,mx1x,mx2x,mx3x);
        d4h(by.x,by.y,mn0y,mn1y,mn2y,mn3y); d4h(by.z,by.w,mx0y,mx1y,mx2y,mx3y);
        d4h(bz.x,bz.y,mn0z,mn1z,mn2z,mn3z); d4h(bz.z,bz.w,mx0z,mx1z,mx2z,mx3z);
        sum += mn0x + mn1y + mn2z + mx3x + mx0y + mx1z;
    }
    out[tid] = sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// STAGE 3: AABB Slab Test — fminf/fmaxf intersection for 4 children
// ═══════════════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(BLK,4) bench_aabb_slab(
    const int4* __restrict__ d_bvh4, int n4, int iters,
    const float* __restrict__ rox, const float* __restrict__ roy, const float* __restrict__ roz,
    const float* __restrict__ rix, const float* __restrict__ riy, const float* __restrict__ riz,
    int numRays, float* __restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ri = tid % numRays;
    float ox = rox[ri], oy = roy[ri], oz = roz[ri];
    float ix = rix[ri], iy = riy[ri], iz = riz[ri];
    float hitT = 1e30f;
    int hits = 0;

    for (int it = 0; it < iters; it++) {
        int ni = (tid + it * 7) % n4;
        int4 bx = __ldg(&d_bvh4[ni*4]);
        int4 by = __ldg(&d_bvh4[ni*4+1]);
        int4 bz = __ldg(&d_bvh4[ni*4+2]);
        float mn0x,mn1x,mn2x,mn3x,mx0x,mx1x,mx2x,mx3x;
        float mn0y,mn1y,mn2y,mn3y,mx0y,mx1y,mx2y,mx3y;
        float mn0z,mn1z,mn2z,mn3z,mx0z,mx1z,mx2z,mx3z;
        d4h(bx.x,bx.y,mn0x,mn1x,mn2x,mn3x); d4h(bx.z,bx.w,mx0x,mx1x,mx2x,mx3x);
        d4h(by.x,by.y,mn0y,mn1y,mn2y,mn3y); d4h(by.z,by.w,mx0y,mx1y,mx2y,mx3y);
        d4h(bz.x,bz.y,mn0z,mn1z,mn2z,mn3z); d4h(bz.z,bz.w,mx0z,mx1z,mx2z,mx3z);

        // 4-child slab test (exact v11 math)
        float t0n,t0x2,t1n,t1x2,t2n,t2x2,t3n,t3x2;
        {float a=(mn0x-ox)*ix,b=(mx0x-ox)*ix;t0n=fminf(a,b);t0x2=fmaxf(a,b);a=(mn0y-oy)*iy;b=(mx0y-oy)*iy;t0n=fmaxf(t0n,fminf(a,b));t0x2=fminf(t0x2,fmaxf(a,b));a=(mn0z-oz)*iz;b=(mx0z-oz)*iz;t0n=fmaxf(t0n,fminf(a,b));t0x2=fminf(t0x2,fmaxf(a,b));}
        {float a=(mn1x-ox)*ix,b=(mx1x-ox)*ix;t1n=fminf(a,b);t1x2=fmaxf(a,b);a=(mn1y-oy)*iy;b=(mx1y-oy)*iy;t1n=fmaxf(t1n,fminf(a,b));t1x2=fminf(t1x2,fmaxf(a,b));a=(mn1z-oz)*iz;b=(mx1z-oz)*iz;t1n=fmaxf(t1n,fminf(a,b));t1x2=fminf(t1x2,fmaxf(a,b));}
        {float a=(mn2x-ox)*ix,b=(mx2x-ox)*ix;t2n=fminf(a,b);t2x2=fmaxf(a,b);a=(mn2y-oy)*iy;b=(mx2y-oy)*iy;t2n=fmaxf(t2n,fminf(a,b));t2x2=fminf(t2x2,fmaxf(a,b));a=(mn2z-oz)*iz;b=(mx2z-oz)*iz;t2n=fmaxf(t2n,fminf(a,b));t2x2=fminf(t2x2,fmaxf(a,b));}
        {float a=(mn3x-ox)*ix,b=(mx3x-ox)*ix;t3n=fminf(a,b);t3x2=fmaxf(a,b);a=(mn3y-oy)*iy;b=(mx3y-oy)*iy;t3n=fmaxf(t3n,fminf(a,b));t3x2=fminf(t3x2,fmaxf(a,b));a=(mn3z-oz)*iz;b=(mx3z-oz)*iz;t3n=fmaxf(t3n,fminf(a,b));t3x2=fminf(t3x2,fmaxf(a,b));}

        bool h0 = (t0x2 >= fmaxf(t0n,0.f)) && (t0n < hitT);
        bool h1 = (t1x2 >= fmaxf(t1n,0.f)) && (t1n < hitT);
        bool h2 = (t2x2 >= fmaxf(t2n,0.f)) && (t2n < hitT);
        bool h3 = (t3x2 >= fmaxf(t3n,0.f)) && (t3n < hitT);
        hits += h0 + h1 + h2 + h3;
    }
    out[tid] = (float)hits;
}

// ═══════════════════════════════════════════════════════════════════════════
// STAGE 4: Child Sort — Branchless sorting network (5 cswap)
// ═══════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void cswap(float&a,float&b,int&ia,int&ib){
    bool s=a>b;float tf=s?a:b;a=s?b:a;b=tf;int ti=s?ia:ib;ia=s?ib:ia;ib=ti;}

__global__ void __launch_bounds__(BLK,4) bench_child_sort(
    int iters, float* __restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned h = tid * 2654435761u;
    float sum = 0;

    for (int it = 0; it < iters; it++) {
        h ^= h << 13; h ^= h >> 17; h ^= h << 5;
        float d0 = __uint_as_float((h & 0x007FFFFF) | 0x3F800000);
        h ^= h << 13; h ^= h >> 17; h ^= h << 5;
        float d1 = __uint_as_float((h & 0x007FFFFF) | 0x3F800000);
        h ^= h << 13; h ^= h >> 17; h ^= h << 5;
        float d2 = __uint_as_float((h & 0x007FFFFF) | 0x3F800000);
        h ^= h << 13; h ^= h >> 17; h ^= h << 5;
        float d3 = __uint_as_float((h & 0x007FFFFF) | 0x3F800000);

        int c0 = 0, c1 = 1, c2 = 2, c3 = 3;
        // 5-comparator sorting network for 4 elements
        cswap(d0,d1,c0,c1); cswap(d2,d3,c2,c3);
        cswap(d0,d2,c0,c2); cswap(d1,d3,c1,c3); cswap(d1,d2,c1,c2);
        sum += d0 + (float)c0;
    }
    out[tid] = sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// STAGE 5: Warp Shuffle — __shfl_xor_sync warp-min (5 shuffles per value)
// v11 does 4 children × 5 shuffles = 20 shuffles per node
// ═══════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ float warpMin(float v) {
    v = fminf(v, __shfl_xor_sync(0xFFFFFFFF, v, 16));
    v = fminf(v, __shfl_xor_sync(0xFFFFFFFF, v, 8));
    v = fminf(v, __shfl_xor_sync(0xFFFFFFFF, v, 4));
    v = fminf(v, __shfl_xor_sync(0xFFFFFFFF, v, 2));
    v = fminf(v, __shfl_xor_sync(0xFFFFFFFF, v, 1));
    return v;
}

__global__ void __launch_bounds__(BLK,4) bench_warp_shuffle(
    int iters, float* __restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (float)tid;

    for (int it = 0; it < iters; it++) {
        // 4 warpMin calls (one per child) = 20 shuffles
        float w0 = warpMin(v + (float)it);
        float w1 = warpMin(v - (float)it);
        float w2 = warpMin(v * 0.5f + (float)it);
        float w3 = warpMin(v * 1.5f - (float)it);
        v = w0 + w1 + w2 + w3;
    }
    out[tid] = v;
}

// ═══════════════════════════════════════════════════════════════════════════
// STAGE 5b: __ballot_sync + __popc overhead
// ═══════════════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(BLK,4) bench_ballot_popc(
    int iters, float* __restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned h = tid * 2654435761u;
    int sum = 0;

    for (int it = 0; it < iters; it++) {
        h ^= h << 13; h ^= h >> 17; h ^= h << 5;
        bool h0 = (h & 1);
        bool h1 = (h & 2);
        bool h2 = (h & 4);
        bool h3 = (h & 8);
        // 4 ballots + 4 popcs (exact v11 pattern)
        unsigned m0 = __ballot_sync(0xFFFFFFFF, h0);
        unsigned m1 = __ballot_sync(0xFFFFFFFF, h1);
        unsigned m2 = __ballot_sync(0xFFFFFFFF, h2);
        unsigned m3 = __ballot_sync(0xFFFFFFFF, h3);
        int warpHits = (m0?1:0) + (m1?1:0) + (m2?1:0) + (m3?1:0);
        sum += warpHits + __popc(m0) + __popc(m1);
    }
    out[tid] = (float)sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// STAGE 6: Triangle Intersection — Möller-Trumbore
// ═══════════════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(BLK,4) bench_moller_trumbore(
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    int numTris, int iters,
    const float* __restrict__ rox, const float* __restrict__ roy, const float* __restrict__ roz,
    const float* __restrict__ rdx, const float* __restrict__ rdy, const float* __restrict__ rdz,
    int numRays, float* __restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ri = tid % numRays;
    float ox = rox[ri], oy = roy[ri], oz = roz[ri];
    float dx = rdx[ri], dy = rdy[ri], dz = rdz[ri];
    float hitT = 1e30f;

    for (int it = 0; it < iters; it++) {
        int ti = (tid + it * 13) % numTris;
        float v0x=tv0x[ti],v0y=tv0y[ti],v0z=tv0z[ti];
        float e1x=tv1x[ti]-v0x,e1y=tv1y[ti]-v0y,e1z=tv1z[ti]-v0z;
        float e2x=tv2x[ti]-v0x,e2y=tv2y[ti]-v0y,e2z=tv2z[ti]-v0z;
        float hx=dy*e2z-dz*e2y,hy=dz*e2x-dx*e2z,hz=dx*e2y-dy*e2x;
        float a2=e1x*hx+e1y*hy+e1z*hz;
        if(fabsf(a2)<1e-8f) continue;
        float f=__frcp_rn(a2);float sx=ox-v0x,sy=oy-v0y,sz=oz-v0z;
        float u=f*(sx*hx+sy*hy+sz*hz);
        if(u<0.f||u>1.f) continue;
        float qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x;
        float v=f*(dx*qx+dy*qy+dz*qz);
        if(v<0.f||u+v>1.f) continue;
        float tt=f*(e2x*qx+e2y*qy+e2z*qz);
        if(tt>0.001f&&tt<hitT) hitT=tt;
    }
    out[tid] = hitT;
}

// ═══════════════════════════════════════════════════════════════════════════
// STAGE 7: Stack Push/Pop — register-based short stack operations
// ═══════════════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(BLK,4) bench_stack_ops(
    int iters, float* __restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stk[SHORT_STACK];
    int sp = 0;
    int sum = 0;
    unsigned h = tid * 2654435761u;

    for (int it = 0; it < iters; it++) {
        h ^= h << 13; h ^= h >> 17; h ^= h << 5;
        int pushCount = (h & 3); // 0-3 children to push

        // Push (like v11 pushing sorted children)
        for (int i = 0; i < pushCount && sp < SHORT_STACK; i++)
            stk[sp++] = (int)(h >> (i*8)) & 0xFFFF;

        // Pop (like v11 popping next node)
        if (sp > 0) {
            int val = stk[--sp];
            sum += val;
        }
    }
    out[tid] = (float)sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// STAGE 8: Full V11 pipeline (for comparison)
// ═══════════════════════════════════════════════════════════════════════════
__device__ unsigned int g_rayCounter_prof;

__global__ void __launch_bounds__(256,4) traceV11_prof(
    const int4*__restrict__ d_bvh4,int n4,int smN,
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    const float*__restrict__ rix,const float*__restrict__ riy,const float*__restrict__ riz,
    Hit*__restrict__ hits,int numRays,unsigned long long*__restrict__ stats)
{
    extern __shared__ int4 s_bvh4_v11[];
    int smN4=min(n4,smN);for(int i=threadIdx.x;i<smN4*4;i+=blockDim.x)s_bvh4_v11[i]=d_bvh4[i];__syncthreads();
    int cnst=c_bvh4N,lane=threadIdx.x&31;unsigned long long ln=0,lt=0;
    while(true){int bs;if(lane==0)bs=atomicAdd(&g_rayCounter_prof,32);bs=__shfl_sync(0xFFFFFFFF,bs,0);
        if(bs>=numRays)break;int ri=bs+lane;if(ri>=numRays)continue;
        float ox=rox[ri],oy=roy[ri],oz=roz[ri],dx=rdx[ri],dy=rdy[ri],dz=rdz[ri];
        float ix=rix[ri],iy=riy[ri],iz=riz[ri];float hitT=1e30f;int hitTri=-1;float hitU=0,hitV=0;
        int stk[SHORT_STACK];int sp=0,ni=0;
        while(true){
            if(ni==-1){if(sp>0)ni=stk[--sp];else break;continue;}
            if(ni<=-2){int val=-(ni+2),tc=(val&7)+1,ts=val>>3;
                for(int i=0;i<tc;i++){int ti=ts+i;lt++;
                    float v0x=tv0x[ti],v0y=tv0y[ti],v0z=tv0z[ti];
                    float e1x=tv1x[ti]-v0x,e1y=tv1y[ti]-v0y,e1z=tv1z[ti]-v0z;
                    float e2x=tv2x[ti]-v0x,e2y=tv2y[ti]-v0y,e2z=tv2z[ti]-v0z;
                    float hx=dy*e2z-dz*e2y,hy=dz*e2x-dx*e2z,hz=dx*e2y-dy*e2x;
                    float a2=e1x*hx+e1y*hy+e1z*hz;if(fabsf(a2)<1e-8f)continue;
                    float f=__frcp_rn(a2);float sx=ox-v0x,sy=oy-v0y,sz=oz-v0z;
                    float u=f*(sx*hx+sy*hy+sz*hz);if(u<0.f||u>1.f)continue;
                    float qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x;
                    float v=f*(dx*qx+dy*qy+dz*qz);if(v<0.f||u+v>1.f)continue;
                    float tt=f*(e2x*qx+e2y*qy+e2z*qz);
                    if(tt>0.001f&&tt<hitT){hitT=tt;hitTri=ti;hitU=u;hitV=v;}}
                if(sp>0)ni=stk[--sp];else break;continue;}
            int4 bx,by,bz,ch;
            if(ni<smN4){bx=s_bvh4_v11[ni*4];by=s_bvh4_v11[ni*4+1];bz=s_bvh4_v11[ni*4+2];ch=s_bvh4_v11[ni*4+3];}
            else if(ni<cnst){bx=c_bvh4[ni*4];by=c_bvh4[ni*4+1];bz=c_bvh4[ni*4+2];ch=c_bvh4[ni*4+3];}
            else{bx=__ldg(&d_bvh4[ni*4]);by=__ldg(&d_bvh4[ni*4+1]);bz=__ldg(&d_bvh4[ni*4+2]);ch=__ldg(&d_bvh4[ni*4+3]);}
            ln++;float mn0x,mn1x,mn2x,mn3x,mx0x,mx1x,mx2x,mx3x;
            float mn0y,mn1y,mn2y,mn3y,mx0y,mx1y,mx2y,mx3y;float mn0z,mn1z,mn2z,mn3z,mx0z,mx1z,mx2z,mx3z;
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
            unsigned m0=__ballot_sync(0xFFFFFFFF,h0),m1=__ballot_sync(0xFFFFFFFF,h1);
            unsigned m2=__ballot_sync(0xFFFFFFFF,h2),m3=__ballot_sync(0xFFFFFFFF,h3);
            int warpHits=(m0?1:0)+(m1?1:0)+(m2?1:0)+(m3?1:0);
            if(warpHits==0){if(sp>0)ni=stk[--sp];else break;continue;}
            float w0=m0?warpMin(h0?t0n:1e30f):1e30f;
            float w1=m1?warpMin(h1?t1n:1e30f):1e30f;
            float w2=m2?warpMin(h2?t2n:1e30f):1e30f;
            float w3=m3?warpMin(h3?t3n:1e30f):1e30f;
            float wd[4]={w0,w1,w2,w3};int wci[4]={c0,c1,c2,c3};
            cswap(wd[0],wd[1],wci[0],wci[1]);cswap(wd[2],wd[3],wci[2],wci[3]);
            cswap(wd[0],wd[2],wci[0],wci[2]);cswap(wd[1],wd[3],wci[1],wci[3]);cswap(wd[1],wd[2],wci[1],wci[2]);
            for(int i=warpHits-1;i>=1&&sp<SHORT_STACK;i--)stk[sp++]=wci[i];
            ni=wci[0];continue;}
        hits[ri].t=hitT;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;}
    atomicAdd(&stats[0],ln);atomicAdd(&stats[1],lt);
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK HARNESS
// ═══════════════════════════════════════════════════════════════════════════
struct StageResult {
    const char* name;
    double totalOps;       // total operations across all threads
    float  ms;             // kernel time (avg over runs)
    double opsPerSec;      // throughput
    double nsPerOp;        // latency
    double gbps;           // bandwidth (0 if not applicable)
};

int main() {
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  V11 Per-Stage Latency Profiler — BVH4 FP16 Traversal Pipeline             ║\n");
    printf("║  %d iterations/thread — cudaEvent timing                                   ║\n", LOOP_ITERS);
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int nSMs = prop.multiProcessorCount;
    int nb = nSMs * 4;
    int nThreads = nb * BLK;
    int warmRuns = 2, benchRuns = 5;

    printf("  GPU: %s | %d SMs | %dKB L2 | %.0f GB/s HBM\n",
        prop.name, nSMs, prop.l2CacheSize/1024,
        2.0 * prop.memoryClockRate * (prop.memoryBusWidth/8) / 1e6);
    printf("  Grid: %d blocks × %d threads = %d threads\n", nb, BLK, nThreads);
    printf("  Loop iterations per thread: %d\n\n", LOOP_ITERS);

    // ═══ Build scene (100K tris) ═══
    int maxTris = 100000;
    Tri* h_tris = (Tri*)malloc(maxTris * sizeof(Tri));
    int nt = genConference(h_tris, maxTris);

    BN* h_nodes = (BN*)calloc(nt*2, sizeof(BN));
    int* tidx = (int*)malloc(nt*4);
    for (int i = 0; i < nt; i++) tidx[i] = i;
    int nc = 0;
    buildSAH(h_nodes, h_tris, tidx, nc, 0, nt, 0);

    BN* h_ord = (BN*)malloc(nc * sizeof(BN));
    int* remap = (int*)malloc(nc * 4);
    treeReorder(h_nodes, nc, h_ord, remap);

    Tri* h_to = (Tri*)malloc(nt * sizeof(Tri));
    for (int i = 0; i < nt; i++) h_to[i] = h_tris[tidx[i]];

    BVH4H* h_b4 = (BVH4H*)calloc(nc, sizeof(BVH4H));
    int n4 = 0;
    collapseRec(h_ord, 0, h_b4, &n4);

    BVH4H* h_b4o = (BVH4H*)malloc(n4 * sizeof(BVH4H));
    int* r4 = (int*)malloc(n4 * 4);
    reorderBVH4(h_b4, n4, h_b4o, r4);

    int4* h_gpu = (int4*)malloc(n4 * 4 * sizeof(int4));
    packBVH4GPU(h_b4o, n4, h_gpu);

    int cn4 = n4 < CONST_BVH4 ? n4 : CONST_BVH4;
    cudaMemcpyToSymbol(c_bvh4, h_gpu, cn4 * 4 * sizeof(int4));
    cudaMemcpyToSymbol(c_bvh4N, &cn4, 4);

    printf("  Scene: %dK tris | %d BVH4 nodes | %.2f MB | %d nodes in const\n\n",
        nt/1000, n4, n4*64.0f/(1024*1024), cn4);

    // Upload BVH to GPU
    int4* d_bvh4;
    cudaMalloc(&d_bvh4, n4 * 4 * sizeof(int4));
    cudaMemcpy(d_bvh4, h_gpu, n4 * 4 * sizeof(int4), cudaMemcpyHostToDevice);

    // Upload triangles SOA
    float* h_v[9];
    for (int j = 0; j < 9; j++) h_v[j] = (float*)malloc(nt * 4);
    for (int i = 0; i < nt; i++) {
        h_v[0][i]=h_to[i].v0.x; h_v[1][i]=h_to[i].v0.y; h_v[2][i]=h_to[i].v0.z;
        h_v[3][i]=h_to[i].v1.x; h_v[4][i]=h_to[i].v1.y; h_v[5][i]=h_to[i].v1.z;
        h_v[6][i]=h_to[i].v2.x; h_v[7][i]=h_to[i].v2.y; h_v[8][i]=h_to[i].v2.z;
    }
    float* d_v[9];
    for (int j = 0; j < 9; j++) {
        cudaMalloc(&d_v[j], nt * 4);
        cudaMemcpy(d_v[j], h_v[j], nt * 4, cudaMemcpyHostToDevice);
    }

    // Generate 4M diffuse rays
    int numRays = NUM_RAYS;
    RayAoS* h_rays = (RayAoS*)malloc(numRays * sizeof(RayAoS));
    genDiffuse(h_rays, numRays, h_to, nt);

    float* h_rox = (float*)malloc(numRays*4);
    float* h_roy = (float*)malloc(numRays*4);
    float* h_roz = (float*)malloc(numRays*4);
    float* h_rdx = (float*)malloc(numRays*4);
    float* h_rdy = (float*)malloc(numRays*4);
    float* h_rdz = (float*)malloc(numRays*4);
    float* h_rix = (float*)malloc(numRays*4);
    float* h_riy = (float*)malloc(numRays*4);
    float* h_riz = (float*)malloc(numRays*4);
    for (int i = 0; i < numRays; i++) {
        h_rox[i]=h_rays[i].o.x; h_roy[i]=h_rays[i].o.y; h_roz[i]=h_rays[i].o.z;
        h_rdx[i]=h_rays[i].d.x; h_rdy[i]=h_rays[i].d.y; h_rdz[i]=h_rays[i].d.z;
        h_rix[i]=h_rays[i].id.x; h_riy[i]=h_rays[i].id.y; h_riz[i]=h_rays[i].id.z;
    }
    float *d_rox,*d_roy,*d_roz,*d_rdx,*d_rdy,*d_rdz,*d_rix,*d_riy,*d_riz;
    cudaMalloc(&d_rox,numRays*4); cudaMemcpy(d_rox,h_rox,numRays*4,cudaMemcpyHostToDevice);
    cudaMalloc(&d_roy,numRays*4); cudaMemcpy(d_roy,h_roy,numRays*4,cudaMemcpyHostToDevice);
    cudaMalloc(&d_roz,numRays*4); cudaMemcpy(d_roz,h_roz,numRays*4,cudaMemcpyHostToDevice);
    cudaMalloc(&d_rdx,numRays*4); cudaMemcpy(d_rdx,h_rdx,numRays*4,cudaMemcpyHostToDevice);
    cudaMalloc(&d_rdy,numRays*4); cudaMemcpy(d_rdy,h_rdy,numRays*4,cudaMemcpyHostToDevice);
    cudaMalloc(&d_rdz,numRays*4); cudaMemcpy(d_rdz,h_rdz,numRays*4,cudaMemcpyHostToDevice);
    cudaMalloc(&d_rix,numRays*4); cudaMemcpy(d_rix,h_rix,numRays*4,cudaMemcpyHostToDevice);
    cudaMalloc(&d_riy,numRays*4); cudaMemcpy(d_riy,h_riy,numRays*4,cudaMemcpyHostToDevice);
    cudaMalloc(&d_riz,numRays*4); cudaMemcpy(d_riz,h_riz,numRays*4,cudaMemcpyHostToDevice);

    // Output buffer
    float* d_out;
    cudaMalloc(&d_out, nThreads * sizeof(float));

    // Hits + stats for full kernel
    Hit* d_hits;
    cudaMalloc(&d_hits, numRays * sizeof(Hit));
    unsigned long long* d_st;
    cudaMalloc(&d_st, 16);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    int smB = SMEM_BVH4_NODES * 4 * (int)sizeof(int4);

    StageResult results[12];
    int nResults = 0;

    // Helper: warm up, then time. Uses lambda to avoid macro comma issues.
    auto benchStage = [&](const char* label, double totalOps_, int bytesPerOp_, auto launchFn) {
        for(int _w=0;_w<warmRuns;_w++) launchFn();
        cudaDeviceSynchronize();
        cudaError_t e=cudaGetLastError();
        if(e!=cudaSuccess){printf("  %s: CUDA ERROR: %s\n",label,cudaGetErrorString(e));return;}
        float _total=0;
        for(int _r=0;_r<benchRuns;_r++){
            cudaEventRecord(t0); launchFn(); cudaEventRecord(t1); cudaEventSynchronize(t1);
            float _ms; cudaEventElapsedTime(&_ms,t0,t1); _total+=_ms; }
        float _avg=_total/benchRuns;
        double _opsPerSec=totalOps_/(_avg/1000.0);
        double _nsPerOp=(_avg*1e6)/totalOps_;
        double _gbps=(bytesPerOp_>0) ? (totalOps_*bytesPerOp_/(_avg/1000.0))/1e9 : 0;
        results[nResults++]={label,totalOps_,_avg,_opsPerSec,_nsPerOp,_gbps};
    };

    printf("  Running micro-benchmarks...\n\n");

    // Each micro-kernel: nThreads threads × LOOP_ITERS iterations = total operations
    double totalNodeOps = (double)nThreads * LOOP_ITERS;

    // Stage 1a: BVH fetch sequential (64B per node fetch via __ldg)
    benchStage("1a Fetch Sequential", totalNodeOps, 64, [&](){
        bench_fetch_sequential<<<nb,BLK>>>(d_bvh4,n4,LOOP_ITERS,d_out); });

    // Stage 1b: BVH fetch random
    benchStage("1b Fetch Random", totalNodeOps, 64, [&](){
        bench_fetch_random<<<nb,BLK>>>(d_bvh4,n4,LOOP_ITERS,d_out); });

    // Stage 1c: BVH fetch from shared memory
    benchStage("1c Fetch SMEM", totalNodeOps, 64, [&](){
        bench_fetch_smem<<<nb,BLK,smB>>>(d_bvh4,n4,SMEM_BVH4_NODES,LOOP_ITERS,d_out); });

    // Stage 1d: BVH fetch from constant memory
    benchStage("1d Fetch Const", totalNodeOps, 64, [&](){
        bench_fetch_const<<<nb,BLK>>>(cn4,LOOP_ITERS,d_out); });

    // Stage 2: FP16 decode (24 half→float per node, includes load)
    benchStage("2  FP16 Decode", totalNodeOps, 48, [&](){
        bench_fp16_decode<<<nb,BLK>>>(d_bvh4,n4,LOOP_ITERS,d_out); });

    // Stage 3: AABB slab test (4 children, includes load+decode)
    benchStage("3  AABB Slab×4", totalNodeOps, 64, [&](){
        bench_aabb_slab<<<nb,BLK>>>(d_bvh4,n4,LOOP_ITERS,d_rox,d_roy,d_roz,d_rix,d_riy,d_riz,numRays,d_out); });

    // Stage 4: Child sort (5 cswaps, pure ALU)
    benchStage("4  Child Sort", totalNodeOps, 0, [&](){
        bench_child_sort<<<nb,BLK>>>(LOOP_ITERS,d_out); });

    // Stage 5a: Warp shuffle (20 shfl_xor per node = 4 warpMin)
    benchStage("5a WarpShuffle×20", totalNodeOps, 0, [&](){
        bench_warp_shuffle<<<nb,BLK>>>(LOOP_ITERS,d_out); });

    // Stage 5b: ballot_sync + popc (4 ballots + operations per node)
    benchStage("5b Ballot+Popc", totalNodeOps, 0, [&](){
        bench_ballot_popc<<<nb,BLK>>>(LOOP_ITERS,d_out); });

    // Stage 6: Möller-Trumbore (1 tri per iteration, includes load)
    benchStage("6  Moller-Trumb", totalNodeOps, 36, [&](){
        bench_moller_trumbore<<<nb,BLK>>>(d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],nt,LOOP_ITERS,d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,numRays,d_out); });

    // Stage 7: Stack push/pop (pure register ops)
    benchStage("7  Stack Ops", totalNodeOps, 0, [&](){
        bench_stack_ops<<<nb,BLK>>>(LOOP_ITERS,d_out); });

    // Stage 8: Full V11 traversal for comparison
    {
        unsigned int z = 0;
        cudaMemcpyToSymbol(g_rayCounter_prof, &z, 4);
        cudaMemset(d_st, 0, 16);
        traceV11_prof<<<nb,256,smB>>>(d_bvh4,n4,SMEM_BVH4_NODES,
            d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
            d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,d_rix,d_riy,d_riz,
            d_hits,numRays,d_st);
        cudaDeviceSynchronize();
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("  Full V11: CUDA ERROR: %s\n", cudaGetErrorString(e));
        } else {
            // Warmup done, now time
            float fullTotal = 0;
            unsigned long long lastStats[2];
            for (int r = 0; r < benchRuns; r++) {
                cudaMemcpyToSymbol(g_rayCounter_prof, &z, 4);
                cudaMemset(d_st, 0, 16);
                cudaEventRecord(t0);
                traceV11_prof<<<nb,256,smB>>>(d_bvh4,n4,SMEM_BVH4_NODES,
                    d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
                    d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,d_rix,d_riy,d_riz,
                    d_hits,numRays,d_st);
                cudaEventRecord(t1); cudaEventSynchronize(t1);
                float ms; cudaEventElapsedTime(&ms, t0, t1); fullTotal += ms;
                cudaMemcpy(lastStats, d_st, 16, cudaMemcpyDeviceToHost);
            }
            float avgMs = fullTotal / benchRuns;
            double mr = (double)numRays / (avgMs / 1000.0) / 1e6;
            double ndPerRay = (double)lastStats[0] / numRays;
            double triPerRay = (double)lastStats[1] / numRays;
            results[nResults++] = {"8  Full V11", (double)numRays, avgMs, mr * 1e6, 0, 0};

            printf("  Full V11 stats: %.1f nd/ray, %.1f tri/ray\n", ndPerRay, triPerRay);
            printf("  Full V11: %.2f ms → %.1f MRays/s\n\n", avgMs, mr);
        }
    }

    // ═══ RESULTS TABLE — Use ps (picoseconds) and cycles for readability ═══
    // V100 boost clock ~1380 MHz = 0.7246 ns/cycle
    double clockGHz = 1.38;
    double nsPerCycle = 1.0 / clockGHz;

    printf("  ╔═════════════════════╦══════════╦═══════════╦══════════╦═══════╦══════════╗\n");
    printf("  ║ Stage               ║ Time(ms) ║  Gops/s   ║  ps/op   ║ cyc/op║  GB/s    ║\n");
    printf("  ╠═════════════════════╬══════════╬═══════════╬══════════╬═══════╬══════════╣\n");

    float maxMs = 0;
    const char* bottleneckName = "";
    // Skip the Full V11 entry (last) for bottleneck among micro-stages
    int nMicroStages = nResults > 0 ? nResults - 1 : 0;
    for (int i = 0; i < nResults; i++) {
        StageResult& r = results[i];
        double ps = r.nsPerOp * 1000.0;
        double cyc = r.nsPerOp * clockGHz;
        if (r.gbps > 0) {
            printf("  ║ %-19s ║ %7.2f  ║ %8.2f  ║ %7.1f  ║ %5.2f ║ %7.1f  ║\n",
                r.name, r.ms, r.opsPerSec / 1e9, ps, cyc, r.gbps);
        } else {
            printf("  ║ %-19s ║ %7.2f  ║ %8.2f  ║ %7.1f  ║ %5.2f ║    —    ║\n",
                r.name, r.ms, r.opsPerSec / 1e9, ps, cyc);
        }
        if (i < nMicroStages && r.ms > maxMs) {
            maxMs = r.ms;
            bottleneckName = r.name;
        }
    }
    printf("  ╚═════════════════════╩══════════╩═══════════╩══════════╩═══════╩══════════╝\n\n");

    // ═══ ANALYSIS ═══
    printf("  ═══ ANALYSIS: Per-Operation Cost & Bottleneck Identification ═══\n\n");

    double ps_fetch_seq  = results[0].nsPerOp * 1000;
    double ps_fetch_rand = results[1].nsPerOp * 1000;
    double ps_fetch_smem = results[2].nsPerOp * 1000;
    double ps_fetch_cnst = results[3].nsPerOp * 1000;
    double ps_fp16       = results[4].nsPerOp * 1000;
    double ps_aabb       = results[5].nsPerOp * 1000;
    double ps_sort       = results[6].nsPerOp * 1000;
    double ps_shuffle    = results[7].nsPerOp * 1000;
    double ps_ballot     = results[8].nsPerOp * 1000;
    double ps_moller     = results[9].nsPerOp * 1000;
    double ps_stack      = results[10].nsPerOp * 1000;

    printf("  Per-operation throughput cost (ps = picoseconds, amortized across %d threads):\n\n", nThreads);
    printf("    BVH fetch (sequential __ldg): %7.1f ps  (%5.2f cyc)  %.1f GB/s\n", ps_fetch_seq, results[0].nsPerOp*clockGHz, results[0].gbps);
    printf("    BVH fetch (random __ldg):     %7.1f ps  (%5.2f cyc)  %.1f GB/s\n", ps_fetch_rand, results[1].nsPerOp*clockGHz, results[1].gbps);
    printf("    BVH fetch (shared mem):       %7.1f ps  (%5.2f cyc)  %.1f GB/s\n", ps_fetch_smem, results[2].nsPerOp*clockGHz, results[2].gbps);
    printf("    BVH fetch (constant mem):     %7.1f ps  (%5.2f cyc)  %.1f GB/s\n", ps_fetch_cnst, results[3].nsPerOp*clockGHz, results[3].gbps);
    printf("    FP16→FP32 decode (24 vals):   %7.1f ps  (%5.2f cyc)\n", ps_fp16, results[4].nsPerOp*clockGHz);
    printf("    AABB slab test (4 children):  %7.1f ps  (%5.2f cyc)\n", ps_aabb, results[5].nsPerOp*clockGHz);
    printf("    Child sort (5 cswap):         %7.1f ps  (%5.2f cyc)\n", ps_sort, results[6].nsPerOp*clockGHz);
    printf("    Warp shuffle (20 shfl_xor):   %7.1f ps  (%5.2f cyc)\n", ps_shuffle, results[7].nsPerOp*clockGHz);
    printf("    Ballot+popc (4 ballots):      %7.1f ps  (%5.2f cyc)\n", ps_ballot, results[8].nsPerOp*clockGHz);
    printf("    Möller-Trumbore (1 tri):      %7.1f ps  (%5.2f cyc)\n", ps_moller, results[9].nsPerOp*clockGHz);
    printf("    Stack push/pop:               %7.1f ps  (%5.2f cyc)\n\n", ps_stack, results[10].nsPerOp*clockGHz);

    // ═══ Proportional analysis: each stage's kernel time as fraction of full traversal ═══
    // Sum of all micro-stage times (excluding const, which v11 barely uses for deep nodes)
    // Real traversal per node: fetch + decode + slab + sort + shuffle + ballot + stack
    // Use kernel times directly as proportional weight
    float ms_nodeStages = results[0].ms + results[4].ms + results[5].ms +
                          results[6].ms + results[7].ms + results[8].ms + results[10].ms;
    float ms_triStage   = results[9].ms;
    float ms_allStages  = ms_nodeStages + ms_triStage;

    printf("  ═══ TIME BUDGET (kernel time proportions) ═══\n\n");
    printf("    Stage                    Time(ms)    %%total    %%node-only\n");
    printf("    ─────────────────────────────────────────────────────────\n");
    printf("    Fetch (seq __ldg)       %8.2f    %5.1f%%     %5.1f%%\n",
        results[0].ms, 100*results[0].ms/ms_allStages, 100*results[0].ms/ms_nodeStages);
    printf("    FP16 Decode             %8.2f    %5.1f%%     %5.1f%%\n",
        results[4].ms, 100*results[4].ms/ms_allStages, 100*results[4].ms/ms_nodeStages);
    printf("    AABB Slab ×4            %8.2f    %5.1f%%     %5.1f%%\n",
        results[5].ms, 100*results[5].ms/ms_allStages, 100*results[5].ms/ms_nodeStages);
    printf("    Child Sort              %8.2f    %5.1f%%     %5.1f%%\n",
        results[6].ms, 100*results[6].ms/ms_allStages, 100*results[6].ms/ms_nodeStages);
    printf("    Warp Shuffle ×20        %8.2f    %5.1f%%     %5.1f%%\n",
        results[7].ms, 100*results[7].ms/ms_allStages, 100*results[7].ms/ms_nodeStages);
    printf("    Ballot+Popc             %8.2f    %5.1f%%     %5.1f%%\n",
        results[8].ms, 100*results[8].ms/ms_allStages, 100*results[8].ms/ms_nodeStages);
    printf("    Stack Ops               %8.2f    %5.1f%%     %5.1f%%\n",
        results[10].ms, 100*results[10].ms/ms_allStages, 100*results[10].ms/ms_nodeStages);
    printf("    ─────────────────────────────────────────────────────────\n");
    printf("    Node subtotal           %8.2f    %5.1f%%\n", ms_nodeStages, 100*ms_nodeStages/ms_allStages);
    printf("    Möller-Trumbore         %8.2f    %5.1f%%\n", ms_triStage, 100*ms_triStage/ms_allStages);
    printf("    ─────────────────────────────────────────────────────────\n");
    printf("    Total (all stages)      %8.2f    100.0%%\n\n", ms_allStages);

    // V11 warp overhead
    float ms_v11_overhead = results[7].ms + results[8].ms;  // shuffle + ballot
    float ms_v10_node = ms_nodeStages - ms_v11_overhead;
    printf("  ═══ V11 WARP-COHERENT OVERHEAD ═══\n\n");
    printf("    V10 node pipeline (no warp ops):  %7.2f ms\n", ms_v10_node);
    printf("    V11 warp overhead (shfl+ballot):  %7.2f ms  (+%.1f%%)\n",
        ms_v11_overhead, 100*ms_v11_overhead/ms_v10_node);
    printf("      ├─ 20× __shfl_xor_sync:        %7.2f ms\n", results[7].ms);
    printf("      └─ 4× __ballot_sync + __popc:   %7.2f ms\n\n", results[8].ms);

    printf("    Verdict: Warp-coherent ordering adds %.1f%% compute overhead.\n",
        100*ms_v11_overhead/ms_v10_node);
    printf("    This buys ZERO stack divergence → net win if divergence cost > %.1f%%.\n\n",
        100*ms_v11_overhead/ms_v10_node);

    // Memory hierarchy
    double fetch_rand_vs_seq = ps_fetch_rand / ps_fetch_seq;
    printf("  ═══ MEMORY HIERARCHY ═══\n\n");
    printf("    Const mem:     %6.1f ps/node  (%5.1f GB/s) — broadcast, top %d nodes\n", ps_fetch_cnst, results[3].gbps, cn4);
    printf("    Shared mem:    %6.1f ps/node  (%5.1f GB/s) — top %d nodes\n", ps_fetch_smem, results[2].gbps, SMEM_BVH4_NODES);
    printf("    __ldg seq:     %6.1f ps/node  (%5.1f GB/s) — L2-friendly BFS order\n", ps_fetch_seq, results[0].gbps);
    printf("    __ldg random:  %6.1f ps/node  (%5.1f GB/s) — cache-hostile\n", ps_fetch_rand, results[1].gbps);
    printf("    Random/seq ratio: %.1fx slower\n\n", fetch_rand_vs_seq);

    // Bottleneck ranking
    // Bottleneck ranking — core pipeline stages only (indices: 0=seq fetch, 4-10)
    printf("  ═══ BOTTLENECK RANKING (core pipeline stages by kernel time) ═══\n\n");

    // Core pipeline: seq fetch(0), decode(4), slab(5), sort(6), shfl(7), ballot(8), moller(9), stack(10)
    int coreIdx[] = {0, 4, 5, 6, 7, 8, 9, 10};
    const char* coreLabel[] = {"Fetch (__ldg)", "FP16 Decode", "AABB Slab ×4", "Child Sort",
                               "Warp Shuffle", "Ballot+Popc", "Möller-Trumb", "Stack Ops"};
    int nCore = 8;
    int coreOrder[8];
    for (int i = 0; i < nCore; i++) coreOrder[i] = i;
    for (int i = 0; i < nCore - 1; i++)
        for (int j = i+1; j < nCore; j++)
            if (results[coreIdx[coreOrder[i]]].ms < results[coreIdx[coreOrder[j]]].ms)
                { int tmp=coreOrder[i]; coreOrder[i]=coreOrder[j]; coreOrder[j]=tmp; }

    for (int i = 0; i < nCore; i++) {
        int oi = coreOrder[i];
        printf("    %2d. %-20s %7.2f ms  (%5.1f%%)\n",
            i+1, coreLabel[oi], results[coreIdx[oi]].ms, 100*results[coreIdx[oi]].ms/ms_allStages);
    }

    printf("\n  Alternative fetch paths (not in core pipeline total):\n");
    printf("    Fetch Random:    %7.2f ms  (%.1fx seq)\n", results[1].ms, results[1].ms/results[0].ms);
    printf("    Fetch SMEM:      %7.2f ms  (%.2fx seq)\n", results[2].ms, results[2].ms/results[0].ms);
    printf("    Fetch Const:     %7.2f ms  (%.1fx seq — divergent addr serialization)\n",
        results[3].ms, results[3].ms/results[0].ms);

    printf("\n  ═══ OPTIMIZATION PRIORITIES ═══\n\n");
    printf("    #1  %s — %.1f%% of pipeline time\n",
        coreLabel[coreOrder[0]], 100*results[coreIdx[coreOrder[0]]].ms/ms_allStages);
    printf("    #2  %s — %.1f%%\n",
        coreLabel[coreOrder[1]], 100*results[coreIdx[coreOrder[1]]].ms/ms_allStages);
    printf("    #3  %s — %.1f%%\n",
        coreLabel[coreOrder[2]], 100*results[coreIdx[coreOrder[2]]].ms/ms_allStages);
    printf("\n");

    // Cleanup
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_bvh4); cudaFree(d_out); cudaFree(d_hits); cudaFree(d_st);
    for (int j = 0; j < 9; j++) { cudaFree(d_v[j]); free(h_v[j]); }
    cudaFree(d_rox); cudaFree(d_roy); cudaFree(d_roz);
    cudaFree(d_rdx); cudaFree(d_rdy); cudaFree(d_rdz);
    cudaFree(d_rix); cudaFree(d_riy); cudaFree(d_riz);
    free(h_tris); free(h_to); free(h_nodes); free(h_ord);
    free(remap); free(tidx); free(h_b4); free(h_b4o);
    free(r4); free(h_gpu); free(h_rays);
    free(h_rox); free(h_roy); free(h_roz);
    free(h_rdx); free(h_rdy); free(h_rdz);
    free(h_rix); free(h_riy); free(h_riz);

    return 0;
}
