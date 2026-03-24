// Multi-Bounce Ray Tracer with Warp Stream Compaction
//
// Pipeline:
//   Bounce 0: trace primary rays → hits
//   Compact:  ballot+popc removes misses → dense secondary ray buffer
//   Bounce 1: trace compacted diffuse secondary rays → hits
//
// Reports: MR/s per bounce, compaction efficiency,
//          warp occupancy before/after compaction
//
// BVH4 build, scene gen, trace kernel copied from cuda_rt_v11.cu (v10 baseline)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <stdint.h>

#include "warp_compact.cuh"

#define SAH_BINS 16
#define LEAF_SIZE 4
#define SHORT_STACK 12
#define SMEM_BVH4_NODES 128
#define CONST_BVH4 1023
#define BLOCK_SIZE 256
#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

// ═══ GPU GLOBALS ═══
__constant__ int4 c_bvh4[CONST_BVH4 * 4];
__constant__ int  c_bvh4N;
__device__ unsigned int g_rayCounter;

// ═══ STRUCTURES ═══
struct Tri{float3 v0,v1,v2;};
struct AABB{float3 bmin,bmax;};
struct Hit{float t;int tri;float u,v;};
struct BN{AABB b;int l,r,ts,tc;};
struct BVH4H{float minX[4],minY[4],minZ[4],maxX[4],maxY[4],maxZ[4];int child[4];int nChildren;};
struct RayAoS{float3 o,d,id;};

// ═══ BVH BUILD (from cuda_rt_v11.cu) ═══
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
    free(stk);
}

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
    return ni;
}

void reorderBVH4(BVH4H*src,int n4,BVH4H*dst,int*remap){
    int*stk=(int*)malloc(n4*4);int sp=0,out=0;stk[sp++]=0;
    while(sp>0){int i=stk[--sp];remap[i]=out;dst[out++]=src[i];
        for(int c=src[i].nChildren-1;c>=0;c--)if(src[i].child[c]>=0)stk[sp++]=src[i].child[c];}
    for(int i=0;i<n4;i++)for(int c=0;c<4;c++)if(dst[i].child[c]>=0)dst[i].child[c]=remap[dst[i].child[c]];
    free(stk);
}

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
        gpuData[i*4+3]=make_int4(nd.child[0],nd.child[1],nd.child[2],nd.child[3]);
    }
}

// ═══ SCENE GENERATION (from cuda_rt_v11.cu — 100K tri conference room) ═══
void addQuad(Tri*t,int&ti,float3 a,float3 b,float3 c,float3 d){t[ti++]={a,b,c};t[ti++]={a,c,d};}
void addBox(Tri*t,int&ti,float3 mn,float3 mx){
    float3 a={mn.x,mn.y,mn.z},b={mx.x,mn.y,mn.z},cv={mx.x,mx.y,mn.z},d={mn.x,mx.y,mn.z};
    float3 e={mn.x,mn.y,mx.z},f={mx.x,mn.y,mx.z},g={mx.x,mx.y,mx.z},h={mn.x,mx.y,mx.z};
    addQuad(t,ti,a,b,cv,d);addQuad(t,ti,e,f,g,h);addQuad(t,ti,a,b,f,e);addQuad(t,ti,d,cv,g,h);addQuad(t,ti,a,d,h,e);addQuad(t,ti,b,cv,g,f);
}
void addSubQuad(Tri*t,int&ti,float3 o,float3 ux,float3 uy,int nx,int ny){
    for(int i=0;i<nx;i++)for(int j=0;j<ny;j++){float u0=(float)i/nx,u1=(float)(i+1)/nx,v0=(float)j/ny,v1=(float)(j+1)/ny;
        float3 a={o.x+ux.x*u0+uy.x*v0,o.y+ux.y*u0+uy.y*v0,o.z+ux.z*u0+uy.z*v0};
        float3 b={o.x+ux.x*u1+uy.x*v0,o.y+ux.y*u1+uy.y*v0,o.z+ux.z*u1+uy.z*v0};
        float3 cv={o.x+ux.x*u1+uy.x*v1,o.y+ux.y*u1+uy.y*v1,o.z+ux.z*u1+uy.z*v1};
        float3 d2={o.x+ux.x*u0+uy.x*v1,o.y+ux.y*u0+uy.y*v1,o.z+ux.z*u0+uy.z*v1};
        t[ti++]={a,b,cv};t[ti++]={a,cv,d2};}
}
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
    return ti;
}

// ═══ RAY SORTING (Morton + Octant, from cuda_rt_v11.cu) ═══
int octant(float3 d){return (d.x<0?4:0)|(d.y<0?2:0)|(d.z<0?1:0);}
static inline uint32_t expand3(uint32_t v){v&=0x3FF;v=(v|(v<<16))&0x30000FF;v=(v|(v<<8))&0x300F00F;v=(v|(v<<4))&0x30C30C3;v=(v|(v<<2))&0x9249249;return v;}
uint32_t morton3D(float x,float y,float z,float3 mn,float3 mx){
    float nx=(x-mn.x)/(mx.x-mn.x+1e-7f),ny=(y-mn.y)/(mx.y-mn.y+1e-7f),nz=(z-mn.z)/(mx.z-mn.z+1e-7f);
    uint32_t ix=fminf(fmaxf(nx*1023.f,0.f),1023.f),iy=fminf(fmaxf(ny*1023.f,0.f),1023.f),iz=fminf(fmaxf(nz*1023.f,0.f),1023.f);
    return expand3(ix)|(expand3(iy)<<1)|(expand3(iz)<<2);
}
void sortMortonOctant(RayAoS*r,int n,float3 smn,float3 smx){
    struct SK{uint32_t key;int idx;};SK*keys=(SK*)malloc(n*sizeof(SK));
    for(int i=0;i<n;i++){uint32_t m=morton3D(r[i].o.x,r[i].o.y,r[i].o.z,smn,smx);uint32_t o=(uint32_t)octant(r[i].d);keys[i]={(o<<27)|(m>>3),i};}
    std::sort(keys,keys+n,[](const SK&a,const SK&b){return a.key<b.key;});
    RayAoS*tmp=(RayAoS*)malloc(n*sizeof(RayAoS));for(int i=0;i<n;i++)tmp[i]=r[keys[i].idx];
    memcpy(r,tmp,n*sizeof(RayAoS));free(tmp);free(keys);
}

// Camera INSIDE the room so secondary diffuse rays scatter into geometry
void genPrimary(RayAoS*r,int n){int w=(int)sqrtf((float)n);
    for(int i=0;i<n;i++){int px=i%w,py=i/w;float u=(2.f*px/w-1.f)*1.2f,v=(2.f*py/(n/w)-1.f)*.9f;
        r[i].o={0,2.5f,0};float3 d={u,v,-1.0f};float l=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);
        d.x/=l;d.y/=l;d.z/=l;r[i].d=d;r[i].id={1.f/d.x,1.f/d.y,1.f/d.z};}
}

// ═══ DEVICE HELPERS (from cuda_rt_v11.cu) ═══
__device__ __forceinline__ void d4h(int lo,int hi,float&v0,float&v1,float&v2,float&v3){
    v0=__half2float(__ushort_as_half((unsigned short)(lo)));v1=__half2float(__ushort_as_half((unsigned short)(lo>>16)));
    v2=__half2float(__ushort_as_half((unsigned short)(hi)));v3=__half2float(__ushort_as_half((unsigned short)(hi>>16)));
}
__device__ __forceinline__ void cswap(float&a,float&b,int&ia,int&ib){
    bool s=a>b;float tf=s?a:b;a=s?b:a;b=tf;int ti=s?ia:ib;ia=s?ib:ia;ib=ti;
}

// ═══ TRACE KERNEL (v10 baseline from cuda_rt_v11.cu) ═══
extern __shared__ int4 s_bvh4[];
__global__ void __launch_bounds__(256,4) traceKernel(
    const int4*__restrict__ d_bvh4,int n4,int smN,
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    const float*__restrict__ rix,const float*__restrict__ riy,const float*__restrict__ riz,
    Hit*__restrict__ hits,int numRays,unsigned long long*__restrict__ stats)
{
    int smN4=min(n4,smN);for(int i=threadIdx.x;i<smN4*4;i+=blockDim.x)s_bvh4[i]=d_bvh4[i];__syncthreads();
    int cnst=c_bvh4N,lane=threadIdx.x&31;unsigned long long ln=0,lt=0;
    while(true){int bs;if(lane==0)bs=atomicAdd(&g_rayCounter,32);bs=__shfl_sync(0xFFFFFFFF,bs,0);
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
            if(ni<smN4){bx=s_bvh4[ni*4];by=s_bvh4[ni*4+1];bz=s_bvh4[ni*4+2];ch=s_bvh4[ni*4+3];}
            else if(ni<cnst){bx=c_bvh4[ni*4];by=c_bvh4[ni*4+1];bz=c_bvh4[ni*4+2];ch=c_bvh4[ni*4+3];}
            else{bx=__ldg(&d_bvh4[ni*4]);by=__ldg(&d_bvh4[ni*4+1]);bz=__ldg(&d_bvh4[ni*4+2]);ch=__ldg(&d_bvh4[ni*4+3]);}
            ln++;float mn0x,mn1x,mn2x,mn3x,mx0x,mx1x,mx2x,mx3x;
            float mn0y,mn1y,mn2y,mn3y,mx0y,mx1y,mx2y,mx3y;float mn0z,mn1z,mn2z,mn3z,mx0z,mx1z,mx2z,mx3z;
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
            int nh=h0+h1+h2+h3;if(nh==0){if(sp>0)ni=stk[--sp];else break;continue;}
            float d4[4]={h0?t0n:1e30f,h1?t1n:1e30f,h2?t2n:1e30f,h3?t3n:1e30f};
            int ci[4]={c0,c1,c2,c3};
            cswap(d4[0],d4[1],ci[0],ci[1]);cswap(d4[2],d4[3],ci[2],ci[3]);
            cswap(d4[0],d4[2],ci[0],ci[2]);cswap(d4[1],d4[3],ci[1],ci[3]);cswap(d4[1],d4[2],ci[1],ci[2]);
            for(int i=nh-1;i>=1&&sp<SHORT_STACK;i--)stk[sp++]=ci[i];ni=ci[0];continue;}
        hits[ri].t=hitT;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;}
    atomicAdd(&stats[0],ln);atomicAdd(&stats[1],lt);
}

// ═══ PCG HASH (device-side RNG for secondary ray generation) ═══
__device__ __forceinline__ unsigned int pcg_hash(unsigned int v){
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
__device__ __forceinline__ float pcg_float(unsigned int v){
    return pcg_hash(v) * 2.3283064365386963e-10f;
}

// ═══ COMBINED SECONDARY RAY GENERATION + WARP COMPACTION ═══
// For each bounce-0 hit: generate cosine-weighted diffuse secondary ray.
// Misses are eliminated in-place via blockCompact (from warp_compact.cuh).
// Output is a dense, gap-free buffer of secondary rays.
__global__ void genAndCompact(
    // Bounce-0 ray origins/directions (for computing hit points)
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    // Bounce-0 hits
    const Hit*__restrict__ hits,
    // Triangle vertex SoA (for normals)
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    int numRays,
    // Output: compacted secondary rays
    float*__restrict__ out_ox,float*__restrict__ out_oy,float*__restrict__ out_oz,
    float*__restrict__ out_dx,float*__restrict__ out_dy,float*__restrict__ out_dz,
    float*__restrict__ out_ix,float*__restrict__ out_iy,float*__restrict__ out_iz,
    int*__restrict__ d_totalOut,unsigned int seed)
{
    __shared__ int smem[BLOCK_SIZE/32 + 1];
    __shared__ int blockOff;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int alive = (tid < numRays && hits[tid].tri >= 0) ? 1 : 0;

    int localIdx;
    int blockTotal = blockCompact(alive, localIdx, smem);

    if(threadIdx.x == 0) blockOff = atomicAdd(d_totalOut, blockTotal);
    __syncthreads();

    if(!alive) return;

    int oi = blockOff + localIdx;
    float ht = hits[tid].t;

    // Hit point = ray origin + t * direction
    float px = rox[tid] + ht * rdx[tid];
    float py = roy[tid] + ht * rdy[tid];
    float pz = roz[tid] + ht * rdz[tid];

    // Triangle normal (cross product of edges)
    int tri = hits[tid].tri;
    float e1x = tv1x[tri]-tv0x[tri], e1y = tv1y[tri]-tv0y[tri], e1z = tv1z[tri]-tv0z[tri];
    float e2x = tv2x[tri]-tv0x[tri], e2y = tv2y[tri]-tv0y[tri], e2z = tv2z[tri]-tv0z[tri];
    float nx = e1y*e2z - e1z*e2y, ny = e1z*e2x - e1x*e2z, nz = e1x*e2y - e1y*e2x;
    float invLen = rsqrtf(fmaxf(nx*nx + ny*ny + nz*nz, 1e-12f));
    nx *= invLen; ny *= invLen; nz *= invLen;

    // Flip normal to face incoming ray
    if(nx*rdx[tid]+ny*rdy[tid]+nz*rdz[tid] > 0){ nx=-nx; ny=-ny; nz=-nz; }

    // Build orthonormal tangent frame
    float tx, ty, tz;
    if(fabsf(nx) > fabsf(ny)){
        float il = rsqrtf(nx*nx + nz*nz);
        tx = -nz*il; ty = 0; tz = nx*il;
    } else {
        float il = rsqrtf(ny*ny + nz*nz);
        tx = 0; ty = nz*il; tz = -ny*il;
    }
    float bx = ny*tz - nz*ty, by = nz*tx - nx*tz, bz = nx*ty - ny*tx;

    // Cosine-weighted hemisphere sample
    float r1 = pcg_float(seed + tid*2u);
    float r2 = pcg_float(seed + tid*2u + 1u);
    float cosTheta = sqrtf(1.0f - r1);
    float sinTheta = sqrtf(r1);
    float phi = 6.2831853f * r2;
    float lx = sinTheta * cosf(phi), ly = sinTheta * sinf(phi), lz = cosTheta;

    // Rotate local hemisphere sample to world space
    float dx2 = lx*tx + ly*bx + lz*nx;
    float dy2 = lx*ty + ly*by + lz*ny;
    float dz2 = lx*tz + ly*bz + lz*nz;

    // Offset origin along normal to avoid self-intersection
    px += nx*0.001f; py += ny*0.001f; pz += nz*0.001f;

    // Write compacted secondary ray
    out_ox[oi] = px;  out_oy[oi] = py;  out_oz[oi] = pz;
    out_dx[oi] = dx2; out_dy[oi] = dy2; out_dz[oi] = dz2;
    out_ix[oi] = 1.0f/dx2; out_iy[oi] = 1.0f/dy2; out_iz[oi] = 1.0f/dz2;
}

// ═══ NON-COMPACTED SECONDARY RAY GENERATION (baseline for comparison) ═══
// Generates secondary rays at position [tid] — dead rays become quick-miss dummies.
// Same hemisphere sampling as genAndCompact, just no compaction.
__global__ void genSecondaryFlat(
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    const Hit*__restrict__ hits,
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    int numRays,
    float*__restrict__ oox,float*__restrict__ ooy,float*__restrict__ ooz,
    float*__restrict__ odx,float*__restrict__ ody,float*__restrict__ odz,
    float*__restrict__ oix,float*__restrict__ oiy,float*__restrict__ oiz,
    unsigned int seed)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=numRays)return;
    if(hits[tid].tri<0){
        // Dead ray: far above scene, pointing up → instant BVH root miss
        oox[tid]=0;ooy[tid]=1000;ooz[tid]=0;
        odx[tid]=0;ody[tid]=1;odz[tid]=0;
        oix[tid]=1e30f;oiy[tid]=1;oiz[tid]=1e30f;return;}
    float ht=hits[tid].t;
    float px=rox[tid]+ht*rdx[tid],py=roy[tid]+ht*rdy[tid],pz=roz[tid]+ht*rdz[tid];
    int tri=hits[tid].tri;
    float e1x=tv1x[tri]-tv0x[tri],e1y=tv1y[tri]-tv0y[tri],e1z=tv1z[tri]-tv0z[tri];
    float e2x=tv2x[tri]-tv0x[tri],e2y=tv2y[tri]-tv0y[tri],e2z=tv2z[tri]-tv0z[tri];
    float nx=e1y*e2z-e1z*e2y,ny=e1z*e2x-e1x*e2z,nz=e1x*e2y-e1y*e2x;
    float il=rsqrtf(fmaxf(nx*nx+ny*ny+nz*nz,1e-12f));nx*=il;ny*=il;nz*=il;
    if(nx*rdx[tid]+ny*rdy[tid]+nz*rdz[tid]>0){nx=-nx;ny=-ny;nz=-nz;}
    float tx,ty,tz;
    if(fabsf(nx)>fabsf(ny)){float i2=rsqrtf(nx*nx+nz*nz);tx=-nz*i2;ty=0;tz=nx*i2;}
    else{float i2=rsqrtf(ny*ny+nz*nz);tx=0;ty=nz*i2;tz=-ny*i2;}
    float bx=ny*tz-nz*ty,by=nz*tx-nx*tz,bz=nx*ty-ny*tx;
    float r1=pcg_float(seed+tid*2u),r2=pcg_float(seed+tid*2u+1u);
    float cosT=sqrtf(1.0f-r1),sinT=sqrtf(r1),phi=6.2831853f*r2;
    float lx=sinT*cosf(phi),ly=sinT*sinf(phi),lz=cosT;
    float dx2=lx*tx+ly*bx+lz*nx,dy2=lx*ty+ly*by+lz*ny,dz2=lx*tz+ly*bz+lz*nz;
    px+=nx*0.001f;py+=ny*0.001f;pz+=nz*0.001f;
    oox[tid]=px;ooy[tid]=py;ooz[tid]=pz;
    odx[tid]=dx2;ody[tid]=dy2;odz[tid]=dz2;
    oix[tid]=1.0f/dx2;oiy[tid]=1.0f/dy2;oiz[tid]=1.0f/dz2;
}

// ═══ SYNTHETIC KILL: randomly terminate rays to simulate higher miss rates ═══
__global__ void syntheticKill(Hit*hits,int numRays,float killRate,unsigned int seed){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=numRays||hits[tid].tri<0)return;
    if(pcg_float(seed+tid+77777u)<killRate) hits[tid].tri=-1;
}

// ═══ HOST HELPERS ═══

// Upload RayAoS to 9 SoA device arrays, returns device pointers
void uploadRays(RayAoS*rays,int n,float*d_ray[9]){
    float*h[9];for(int j=0;j<9;j++)h[j]=(float*)malloc(n*4);
    for(int i=0;i<n;i++){h[0][i]=rays[i].o.x;h[1][i]=rays[i].o.y;h[2][i]=rays[i].o.z;
        h[3][i]=rays[i].d.x;h[4][i]=rays[i].d.y;h[5][i]=rays[i].d.z;
        h[6][i]=rays[i].id.x;h[7][i]=rays[i].id.y;h[8][i]=rays[i].id.z;}
    for(int j=0;j<9;j++){CHECK(cudaMalloc(&d_ray[j],n*4));CHECK(cudaMemcpy(d_ray[j],h[j],n*4,cudaMemcpyHostToDevice));free(h[j]);}
}

struct BenchResult{double mrs;double nodesPerRay,trisPerRay,bwPerRay;int hitPct;};

BenchResult benchTrace(float*d_ray[9],int numRays,int4*d_bvh4,int n4,float*d_v[9],
                       Hit*d_hits,cudaDeviceProp&prop,int warmups=1,int runs=10)
{
    int nb=prop.multiProcessorCount*4;
    int smB=SMEM_BVH4_NODES*4*(int)sizeof(int4);
    unsigned long long*d_st; CHECK(cudaMalloc(&d_st,16));
    unsigned int z=0;

    // Warmup
    for(int w=0;w<warmups;w++){
        CHECK(cudaMemcpyToSymbol(g_rayCounter,&z,4));
        traceKernel<<<nb,256,smB>>>(d_bvh4,n4,SMEM_BVH4_NODES,
            d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
            d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],
            d_hits,numRays,d_st);
        CHECK(cudaDeviceSynchronize());
    }

    // Timed runs
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    float totalMs=0;
    for(int r=0;r<runs;r++){
        CHECK(cudaMemcpyToSymbol(g_rayCounter,&z,4));
        CHECK(cudaMemset(d_st,0,16));
        cudaEventRecord(t0);
        traceKernel<<<nb,256,smB>>>(d_bvh4,n4,SMEM_BVH4_NODES,
            d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
            d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],
            d_hits,numRays,d_st);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1); totalMs+=ms;
    }

    unsigned long long st[2]; cudaMemcpy(st,d_st,16,cudaMemcpyDeviceToHost);
    Hit*hh=(Hit*)malloc(numRays*sizeof(Hit));
    cudaMemcpy(hh,d_hits,numRays*sizeof(Hit),cudaMemcpyDeviceToHost);
    int hc=0; for(int i=0;i<numRays;i++) if(hh[i].tri>=0) hc++;

    BenchResult br;
    br.mrs=(double)numRays/((totalMs/runs)/1000.0)/1e6;
    br.nodesPerRay=(double)st[0]/numRays;
    br.trisPerRay=(double)st[1]/numRays;
    br.bwPerRay=br.nodesPerRay*64+br.trisPerRay*36;
    br.hitPct=100*hc/numRays;

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_st); free(hh);
    return br;
}

// Compute warp occupancy stats from hit array
void warpOccupancy(const Hit*h,int n,int&activeRays,int&totalWarps,int&activeWarps,double&avgLaneUtil){
    activeRays=0; totalWarps=(n+31)/32; activeWarps=0;
    long long totalActiveLanes=0;
    for(int w=0;w<totalWarps;w++){
        int cnt=0;
        int end = (w+1)*32 < n ? (w+1)*32 : n;
        for(int i=w*32;i<end;i++) if(h[i].tri>=0) cnt++;
        if(cnt>0){ activeWarps++; totalActiveLanes+=cnt; }
        activeRays+=cnt;
    }
    avgLaneUtil = activeWarps>0 ? (double)totalActiveLanes/(activeWarps*32.0) : 0;
}

// ═══ MAIN ═══
int main(){
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Multi-Bounce Ray Tracer with Warp Stream Compaction\n");
    printf("  Pipeline: trace → ballot+popc compact → trace\n");
    printf("  Uses warp_compact.cuh: blockCompact(ballot, popc, shfl)\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    printf("  GPU: %s (%d SMs, %.0f MHz, %.1f GB)\n\n",
        prop.name,prop.multiProcessorCount,prop.clockRate/1000.f,prop.totalGlobalMem/(1024.f*1024*1024));

    float3 sMn={-10,0,-7.5f},sMx={10,5,7.5f};
    int maxTris=100000, numRays=4194304;

    // ─── Build scene ───
    Tri*h_tris=(Tri*)malloc(maxTris*sizeof(Tri));
    int nt=genConference(h_tris,maxTris);

    // ─── Build BVH2 → BVH4 → GPU format ───
    BN*h_nodes=(BN*)calloc(nt*2,sizeof(BN)); int*tidx=(int*)malloc(nt*4);
    for(int i=0;i<nt;i++) tidx[i]=i;
    int nc=0; buildSAH(h_nodes,h_tris,tidx,nc,0,nt,0);
    BN*h_ord=(BN*)malloc(nc*sizeof(BN)); int*remap=(int*)malloc(nc*4);
    treeReorder(h_nodes,nc,h_ord,remap);
    Tri*h_to=(Tri*)malloc(nt*sizeof(Tri)); for(int i=0;i<nt;i++) h_to[i]=h_tris[tidx[i]];
    BVH4H*h_b4=(BVH4H*)calloc(nc,sizeof(BVH4H)); int n4=0; collapseRec(h_ord,0,h_b4,&n4);
    BVH4H*h_b4o=(BVH4H*)malloc(n4*sizeof(BVH4H)); int*r4=(int*)malloc(n4*4);
    reorderBVH4(h_b4,n4,h_b4o,r4);
    int4*h_gpu=(int4*)malloc(n4*4*sizeof(int4)); packBVH4GPU(h_b4o,n4,h_gpu);

    printf("  Scene: %dK tris | BVH4: %d nodes (%.2f MB)\n\n",nt/1000,n4,n4*64/(1024.f*1024.f));

    // ─── Upload BVH to constant + global memory ───
    int cn4=n4<CONST_BVH4?n4:CONST_BVH4;
    CHECK(cudaMemcpyToSymbol(c_bvh4,h_gpu,cn4*4*sizeof(int4)));
    CHECK(cudaMemcpyToSymbol(c_bvh4N,&cn4,4));
    int4*d_bvh4; CHECK(cudaMalloc(&d_bvh4,n4*4*sizeof(int4)));
    CHECK(cudaMemcpy(d_bvh4,h_gpu,n4*4*sizeof(int4),cudaMemcpyHostToDevice));

    // ─── Upload triangle SoA ───
    float*h_v[9]; for(int j=0;j<9;j++) h_v[j]=(float*)malloc(nt*4);
    for(int i=0;i<nt;i++){h_v[0][i]=h_to[i].v0.x;h_v[1][i]=h_to[i].v0.y;h_v[2][i]=h_to[i].v0.z;
        h_v[3][i]=h_to[i].v1.x;h_v[4][i]=h_to[i].v1.y;h_v[5][i]=h_to[i].v1.z;
        h_v[6][i]=h_to[i].v2.x;h_v[7][i]=h_to[i].v2.y;h_v[8][i]=h_to[i].v2.z;}
    float*d_v[9]; for(int j=0;j<9;j++){CHECK(cudaMalloc(&d_v[j],nt*4));CHECK(cudaMemcpy(d_v[j],h_v[j],nt*4,cudaMemcpyHostToDevice));}

    // ─── Generate primary rays (Morton + octant sorted) ───
    RayAoS*rays=(RayAoS*)malloc(numRays*sizeof(RayAoS));
    genPrimary(rays,numRays);
    sortMortonOctant(rays,numRays,sMn,sMx);

    float*d_ray[9]; uploadRays(rays,numRays,d_ray);

    // ╔═══════════════════════════════════════════════╗
    // ║  BOUNCE 0: PRIMARY RAYS                       ║
    // ╚═══════════════════════════════════════════════╝
    Hit*d_hits0; CHECK(cudaMalloc(&d_hits0,numRays*sizeof(Hit)));
    BenchResult b0 = benchTrace(d_ray,numRays,d_bvh4,n4,d_v,d_hits0,prop);

    printf("  ┌─────────────────────────────────────────────────┐\n");
    printf("  │  BOUNCE 0 — Primary Rays (Morton sorted)        │\n");
    printf("  ├─────────────────────────────────────────────────┤\n");
    printf("  │  Rays:     %'12d                          │\n",numRays);
    printf("  │  MR/s:     %12.1f                          │\n",b0.mrs);
    printf("  │  Hit rate: %11d%%                          │\n",b0.hitPct);
    printf("  │  Nodes/ray:%12.1f  Tris/ray: %.1f          │\n",b0.nodesPerRay,b0.trisPerRay);
    printf("  │  BW/ray:   %12.0f B                        │\n",b0.bwPerRay);
    printf("  └─────────────────────────────────────────────────┘\n\n");

    // ╔═══════════════════════════════════════════════╗
    // ║  COMPACTION: ballot + popc + prefix sum        ║
    // ╚═══════════════════════════════════════════════╝
    // Download bounce-0 hits for warp occupancy analysis
    Hit*h_hits0=(Hit*)malloc(numRays*sizeof(Hit));
    CHECK(cudaMemcpy(h_hits0,d_hits0,numRays*sizeof(Hit),cudaMemcpyDeviceToHost));

    int activeRays,totalWarps,activeWarps; double avgLaneUtil;
    warpOccupancy(h_hits0,numRays,activeRays,totalWarps,activeWarps,avgLaneUtil);

    // Allocate compacted output buffers
    float*d_comp[9]; for(int j=0;j<9;j++) CHECK(cudaMalloc(&d_comp[j],numRays*4));
    int*d_totalOut; CHECK(cudaMalloc(&d_totalOut,4)); CHECK(cudaMemset(d_totalOut,0,4));

    // Time the compaction kernel
    cudaEvent_t ct0,ct1; cudaEventCreate(&ct0); cudaEventCreate(&ct1);
    int compBlocks=(numRays+BLOCK_SIZE-1)/BLOCK_SIZE;

    cudaEventRecord(ct0);
    genAndCompact<<<compBlocks,BLOCK_SIZE>>>(
        d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],
        d_hits0,
        d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
        numRays,
        d_comp[0],d_comp[1],d_comp[2],d_comp[3],d_comp[4],d_comp[5],
        d_comp[6],d_comp[7],d_comp[8],
        d_totalOut,12345u);
    cudaEventRecord(ct1); cudaEventSynchronize(ct1);
    float compMs; cudaEventElapsedTime(&compMs,ct0,ct1);

    int numCompacted; CHECK(cudaMemcpy(&numCompacted,d_totalOut,4,cudaMemcpyDeviceToHost));

    // Post-compaction warp occupancy
    int postWarps=(numCompacted+31)/32;
    double postLaneUtil=(double)numCompacted/(postWarps*32.0);

    printf("  ┌─────────────────────────────────────────────────┐\n");
    printf("  │  COMPACTION — __ballot_sync + __popc + shfl     │\n");
    printf("  ├─────────────────────────────────────────────────┤\n");
    printf("  │  Input rays:     %10d                     │\n",numRays);
    printf("  │  Surviving rays: %10d (%5.1f%%)             │\n",numCompacted,100.0*numCompacted/numRays);
    printf("  │  Compaction time:%10.3f ms                  │\n",compMs);
    printf("  │                                                 │\n");
    printf("  │  BEFORE compaction (bounce-0 hits as warp data):│\n");
    printf("  │    Total warps:  %10d                     │\n",totalWarps);
    printf("  │    Active warps: %10d (%5.1f%%)             │\n",activeWarps,100.0*activeWarps/totalWarps);
    printf("  │    Avg lane util:%10.1f%% (in active warps)  │\n",avgLaneUtil*100);
    printf("  │                                                 │\n");
    printf("  │  AFTER compaction (dense warp packing):         │\n");
    printf("  │    Total warps:  %10d                     │\n",postWarps);
    printf("  │    Active warps: %10d (100.0%%)             │\n",postWarps);
    printf("  │    Avg lane util:%10.1f%% (in active warps)  │\n",postLaneUtil*100);
    printf("  │                                                 │\n");
    printf("  │  Occupancy gain: %10.1fx                    │\n",postLaneUtil/avgLaneUtil);
    printf("  └─────────────────────────────────────────────────┘\n\n");

    cudaEventDestroy(ct0); cudaEventDestroy(ct1);

    // ╔═══════════════════════════════════════════════╗
    // ║  BOUNCE 1: DIFFUSE SECONDARY RAYS (compacted) ║
    // ╚═══════════════════════════════════════════════╝
    Hit*d_hits1; CHECK(cudaMalloc(&d_hits1,numCompacted*sizeof(Hit)));
    BenchResult b1 = benchTrace(d_comp,numCompacted,d_bvh4,n4,d_v,d_hits1,prop);

    printf("  ┌─────────────────────────────────────────────────┐\n");
    printf("  │  BOUNCE 1 — Diffuse Secondary (compacted)       │\n");
    printf("  ├─────────────────────────────────────────────────┤\n");
    printf("  │  Rays:     %'12d                          │\n",numCompacted);
    printf("  │  MR/s:     %12.1f                          │\n",b1.mrs);
    printf("  │  Hit rate: %11d%%                          │\n",b1.hitPct);
    printf("  │  Nodes/ray:%12.1f  Tris/ray: %.1f          │\n",b1.nodesPerRay,b1.trisPerRay);
    printf("  │  BW/ray:   %12.0f B                        │\n",b1.bwPerRay);
    printf("  └─────────────────────────────────────────────────┘\n\n");

    // ╔═══════════════════════════════════════════════╗
    // ║  SUMMARY                                      ║
    // ╚═══════════════════════════════════════════════╝
    double l2peak=(3100.0e9/(19.1*64))/1e6;
    printf("  ┌─────────────────────────────────────────────────┐\n");
    printf("  │  SUMMARY                                        │\n");
    printf("  ├─────────────────────────────────────────────────┤\n");
    printf("  │  Bounce 0 (primary):  %8.1f MR/s             │\n",b0.mrs);
    printf("  │  Bounce 1 (diffuse):  %8.1f MR/s             │\n",b1.mrs);
    printf("  │  Bounce 1 / Bounce 0: %8.1f%%                 │\n",100.0*b1.mrs/b0.mrs);
    printf("  │  Compaction overhead:  %7.3f ms (%4.1f%% of B1)│\n",compMs,100.0*compMs/((double)numCompacted/b1.mrs/1e6*1000));
    printf("  │  L2 peak (diffuse):   %8.0f MR/s (theory)    │\n",l2peak);
    printf("  │  Bounce 1 / L2 peak:  %8.1f%%                 │\n",100.0*b1.mrs/l2peak);
    printf("  │                                                 │\n");
    printf("  │  Rays surviving:      %5.1f%% → warp util:    │\n",100.0*numCompacted/numRays);
    printf("  │    Before compact:    %5.1f%% lane utilization │\n",avgLaneUtil*100);
    printf("  │    After  compact:    %5.1f%% lane utilization │\n",postLaneUtil*100);
    printf("  └─────────────────────────────────────────────────┘\n");

    // ╔═══════════════════════════════════════════════════════════╗
    // ║  COMPACTION BENEFIT: SURVIVAL RATE SWEEP                  ║
    // ║  Synthetic kill rates simulate open scenes / deep bounces ║
    // ╚═══════════════════════════════════════════════════════════╝
    printf("\n  ┌──────────────────────────────────────────────────────────────┐\n");
    printf("  │  COMPACTION BENEFIT vs SURVIVAL RATE                        │\n");
    printf("  │  (synthetic kill applied to bounce-0 hits)                  │\n");
    printf("  ├───────────┬──────────┬──────────┬──────────┬───────────────┤\n");
    printf("  │ Survive %%│ No-Comp  │ Compact  │ Eff-NC   │ Eff-Comp  Spd │\n");
    printf("  │           │  MR/s    │  MR/s    │  MR/s    │  MR/s     Up  │\n");
    printf("  ├───────────┼──────────┼──────────┼──────────┼───────────────┤\n");

    float killRates[] = {0.0f, 0.25f, 0.50f, 0.75f, 0.90f};
    float*d_flat[9]; for(int j=0;j<9;j++) CHECK(cudaMalloc(&d_flat[j],numRays*4));
    Hit*d_hitsTemp; CHECK(cudaMalloc(&d_hitsTemp,numRays*sizeof(Hit)));
    Hit*d_hitsNC; CHECK(cudaMalloc(&d_hitsNC,numRays*sizeof(Hit)));

    for(int ki=0;ki<5;ki++){
        float kr = killRates[ki];
        // Copy original hits and apply synthetic kill
        CHECK(cudaMemcpy(d_hitsTemp,d_hits0,numRays*sizeof(Hit),cudaMemcpyDeviceToDevice));
        if(kr>0) syntheticKill<<<compBlocks,BLOCK_SIZE>>>(d_hitsTemp,numRays,kr,99999u+ki);
        CHECK(cudaDeviceSynchronize());

        // Path A: No compaction — generate all N rays (dead→dummy), trace N
        genSecondaryFlat<<<compBlocks,BLOCK_SIZE>>>(
            d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],
            d_hitsTemp,
            d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
            numRays,
            d_flat[0],d_flat[1],d_flat[2],d_flat[3],d_flat[4],d_flat[5],
            d_flat[6],d_flat[7],d_flat[8],12345u);
        CHECK(cudaDeviceSynchronize());
        BenchResult bNC = benchTrace(d_flat,numRays,d_bvh4,n4,d_v,d_hitsNC,prop,1,5);

        // Path B: Compact then trace M
        CHECK(cudaMemset(d_totalOut,0,4));
        genAndCompact<<<compBlocks,BLOCK_SIZE>>>(
            d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],
            d_hitsTemp,
            d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
            numRays,
            d_comp[0],d_comp[1],d_comp[2],d_comp[3],d_comp[4],d_comp[5],
            d_comp[6],d_comp[7],d_comp[8],
            d_totalOut,12345u);
        CHECK(cudaDeviceSynchronize());
        int nComp; CHECK(cudaMemcpy(&nComp,d_totalOut,4,cudaMemcpyDeviceToHost));
        Hit*d_hC; CHECK(cudaMalloc(&d_hC,nComp*sizeof(Hit)));
        BenchResult bC = benchTrace(d_comp,nComp,d_bvh4,n4,d_v,d_hC,prop,1,5);
        cudaFree(d_hC);

        // Effective MR/s = useful rays per second of total pipeline time
        // NC: tracing N rays takes (N / bNC.mrs) seconds, only nComp are useful
        double effNC = (double)nComp * bNC.mrs / numRays;
        // Compact: tracing nComp rays takes (nComp / bC.mrs) seconds
        double effC = bC.mrs; // all rays are useful

        double survPct = 100.0*nComp/numRays;
        double speedup = effC / effNC;
        printf("  │  %5.1f%%  │ %7.1f  │ %7.1f  │ %7.1f  │ %7.1f %4.1fx │\n",
            survPct, bNC.mrs, bC.mrs, effNC, effC, speedup);
    }
    printf("  └───────────┴──────────┴──────────┴──────────┴───────────────┘\n");
    printf("  No-Comp  = trace ALL N rays (dead→dummy quick-miss)\n");
    printf("  Compact  = compact to M live rays, trace M only\n");
    printf("  Eff-*    = effective MR/s = live rays / total trace time\n");
    printf("  Spd Up   = Eff-Compact / Eff-NoCompact\n");

    // ─── Cleanup ───
    for(int j=0;j<9;j++){cudaFree(d_ray[j]);cudaFree(d_comp[j]);cudaFree(d_flat[j]);cudaFree(d_v[j]);free(h_v[j]);}
    cudaFree(d_bvh4);cudaFree(d_hits0);cudaFree(d_hits1);cudaFree(d_totalOut);
    cudaFree(d_hitsTemp);cudaFree(d_hitsNC);
    free(h_tris);free(h_to);free(h_nodes);free(h_ord);free(remap);free(tidx);
    free(h_b4);free(h_b4o);free(r4);free(h_gpu);free(rays);free(h_hits0);

    return 0;
}
