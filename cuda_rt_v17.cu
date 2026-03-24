// ═══════════════════════════════════════════════════════════════════════════
// V17 — BVH8 QUANTIZED (uint8) + AUTO-THRESHOLD
//
// Two code paths:
//   PATH A: BVH4 FP16 (64B/node) — for scenes where BVH fits in L2
//   PATH B: BVH8 uint8-quantized (96B/node) — for large scenes
//
// BVH8 format: parent AABB (6×FP16=12B) + 8×6 uint8 offsets (48B) + 
//              8 × int32 children (32B) + 4B pad = 96B = 6 × int4
//
// Decode: child_min_a = pmin_a + pext_a * (qmin/255.0f)
// 8-wide test done as 2×4 batches to reuse BVH4 slab logic
//
// Threshold: if BVH4_size > 70% L2 → use BVH8 quantized
// ═══════════════════════════════════════════════════════════════════════════

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
#define STACK4 8
#define STACK8 6   // BVH8 is shallower

__device__ unsigned int g_rayCounterA;
__device__ unsigned int g_rayCounterB;

struct Tri{float3 v0,v1,v2;};
struct AABB{float3 bmin,bmax;};
struct Hit{float t;int tri;float u,v;};
struct BN{AABB b;int l,r,ts,tc;};

// Host-side BVH8 node
struct BVH8H {
    float minX[8],minY[8],minZ[8],maxX[8],maxY[8],maxZ[8];
    int child[8];
    int nChildren;
    AABB parentAABB; // for quantization
};

// ═══ FP16 decode (same as v12) ═══
__device__ __forceinline__ void d4h(int lo,int hi,float&v0,float&v1,float&v2,float&v3){
    v0=__half2float(__ushort_as_half((unsigned short)(lo)));
    v1=__half2float(__ushort_as_half((unsigned short)(lo>>16)));
    v2=__half2float(__ushort_as_half((unsigned short)(hi)));
    v3=__half2float(__ushort_as_half((unsigned short)(hi>>16)));
}

// ═══ uint8 extract helpers ═══
__device__ __forceinline__ void extract4u8(int packed, float s, float o,
    float &v0, float &v1, float &v2, float &v3) {
    // packed = [b3<<24 | b2<<16 | b1<<8 | b0]
    // s = extent/255.0f (pre-scaled), dequant: val = origin + s * byte
    v0 = o + s * (float)(packed & 0xFF);
    v1 = o + s * (float)((packed>>8) & 0xFF);
    v2 = o + s * (float)((packed>>16) & 0xFF);
    v3 = o + s * (float)((packed>>24) & 0xFF);
}

__device__ __forceinline__ void cswap(float&a,float&b,int&ia,int&ib){
    bool s=a>b;float tf=s?a:b;a=s?b:a;b=tf;int ti=s?ia:ib;ia=s?ib:ia;ib=ti;
}

// ═══ BVH BUILD ═══
AABB unite(const AABB&a,const AABB&b){return{{fminf(a.bmin.x,b.bmin.x),fminf(a.bmin.y,b.bmin.y),fminf(a.bmin.z,b.bmin.z)},{fmaxf(a.bmax.x,b.bmax.x),fmaxf(a.bmax.y,b.bmax.y),fmaxf(a.bmax.z,b.bmax.z)}};}
AABB triBox(const Tri&t){return{{fminf(fminf(t.v0.x,t.v1.x),t.v2.x),fminf(fminf(t.v0.y,t.v1.y),t.v2.y),fminf(fminf(t.v0.z,t.v1.z),t.v2.z)},{fmaxf(fmaxf(t.v0.x,t.v1.x),t.v2.x),fmaxf(fmaxf(t.v0.y,t.v1.y),t.v2.y),fmaxf(fmaxf(t.v0.z,t.v1.z),t.v2.z)}};}
float3 triCen(const Tri&t){return{(t.v0.x+t.v1.x+t.v2.x)/3,(t.v0.y+t.v1.y+t.v2.y)/3,(t.v0.z+t.v1.z+t.v2.z)/3};}
float saArea(const AABB&b){float dx=b.bmax.x-b.bmin.x,dy=b.bmax.y-b.bmin.y,dz=b.bmax.z-b.bmin.z;return 2.0f*(dx*dy+dy*dz+dz*dx);}
int buildSAH(BN*n,Tri*t,int*idx,int&nc,int s,int c,int d){int ni=nc++;BN&nd=n[ni];AABB bounds=triBox(t[idx[s]]);for(int i=1;i<c;i++)bounds=unite(bounds,triBox(t[idx[s+i]]));nd.b=bounds;if(c<=LEAF_SIZE||d>28){nd.l=-1;nd.r=-1;nd.ts=s;nd.tc=c;return ni;}AABB cb;cb.bmin=cb.bmax=triCen(t[idx[s]]);for(int i=1;i<c;i++){float3 cc=triCen(t[idx[s+i]]);cb.bmin={fminf(cb.bmin.x,cc.x),fminf(cb.bmin.y,cc.y),fminf(cb.bmin.z,cc.z)};cb.bmax={fmaxf(cb.bmax.x,cc.x),fmaxf(cb.bmax.y,cc.y),fmaxf(cb.bmax.z,cc.z)};}float bc=FLT_MAX;int ba=-1,bb=-1;float ps=saArea(bounds);for(int ax=0;ax<3;ax++){float amn=ax==0?cb.bmin.x:ax==1?cb.bmin.y:cb.bmin.z,amx=ax==0?cb.bmax.x:ax==1?cb.bmax.y:cb.bmax.z;if(amx-amn<1e-7f)continue;struct B{AABB b;int c;};B bins[SAH_BINS];for(int i=0;i<SAH_BINS;i++){bins[i].b.bmin={FLT_MAX,FLT_MAX,FLT_MAX};bins[i].b.bmax={-FLT_MAX,-FLT_MAX,-FLT_MAX};bins[i].c=0;}float sc=SAH_BINS/(amx-amn);for(int i=0;i<c;i++){float3 cc=triCen(t[idx[s+i]]);float cv=ax==0?cc.x:ax==1?cc.y:cc.z;int b2=fminf(fmaxf((int)((cv-amn)*sc),0),SAH_BINS-1);bins[b2].b=unite(bins[b2].b,triBox(t[idx[s+i]]));bins[b2].c++;}AABB lB[SAH_BINS];int lC[SAH_BINS];lB[0]=bins[0].b;lC[0]=bins[0].c;for(int i=1;i<SAH_BINS;i++){lB[i]=unite(lB[i-1],bins[i].b);lC[i]=lC[i-1]+bins[i].c;}AABB rB[SAH_BINS];int rC[SAH_BINS];rB[SAH_BINS-1]=bins[SAH_BINS-1].b;rC[SAH_BINS-1]=bins[SAH_BINS-1].c;for(int i=SAH_BINS-2;i>=0;i--){rB[i]=unite(rB[i+1],bins[i].b);rC[i]=rC[i+1]+bins[i].c;}for(int i=0;i<SAH_BINS-1;i++){if(lC[i]==0||rC[i+1]==0)continue;float cost=1.0f+(lC[i]*saArea(lB[i])+rC[i+1]*saArea(rB[i+1]))/ps;if(cost<bc){bc=cost;ba=ax;bb=i;}}}if(ba==-1||bc>(float)c){nd.l=-1;nd.r=-1;nd.ts=s;nd.tc=c;return ni;}float amn=ba==0?cb.bmin.x:ba==1?cb.bmin.y:cb.bmin.z,amx=ba==0?cb.bmax.x:ba==1?cb.bmax.y:cb.bmax.z;float sc=SAH_BINS/(amx-amn);int i=s,j=s+c-1;while(i<=j){float3 cc=triCen(t[idx[i]]);float cv=ba==0?cc.x:ba==1?cc.y:cc.z;int b2=fminf(fmaxf((int)((cv-amn)*sc),0),SAH_BINS-1);if(b2<=bb)i++;else{int tmp=idx[i];idx[i]=idx[j];idx[j]=tmp;j--;}}int lc=i-s;if(lc==0)lc=1;if(lc==c)lc=c-1;nd.ts=-1;nd.tc=0;nd.l=buildSAH(n,t,idx,nc,s,lc,d+1);nd.r=buildSAH(n,t,idx,nc,s+lc,c-lc,d+1);return ni;}
void treeReorder(BN*src,int nc,BN*dst,int*remap){int*stk=(int*)malloc(nc*4);int sp=0,out=0;stk[sp++]=0;while(sp>0){int i=stk[--sp];remap[i]=out;dst[out++]=src[i];if(src[i].l>=0){stk[sp++]=src[i].r;stk[sp++]=src[i].l;}}for(int i=0;i<nc;i++){if(dst[i].l>=0){dst[i].l=remap[dst[i].l];dst[i].r=remap[dst[i].r];}}free(stk);}

// ═══ BVH4 COLLAPSE + PACK (proven, same as v12) ═══
struct BVH4H{float minX[4],minY[4],minZ[4],maxX[4],maxY[4],maxZ[4];int child[4];int nChildren;};
static int collapse4(BN*bn,int bIdx,BVH4H*out,int*nOut){
    if(bn[bIdx].l==-1){int ni=(*nOut)++;BVH4H&nd=out[ni];nd.nChildren=1;nd.minX[0]=bn[bIdx].b.bmin.x;nd.minY[0]=bn[bIdx].b.bmin.y;nd.minZ[0]=bn[bIdx].b.bmin.z;nd.maxX[0]=bn[bIdx].b.bmax.x;nd.maxY[0]=bn[bIdx].b.bmax.y;nd.maxZ[0]=bn[bIdx].b.bmax.z;nd.child[0]=-((bn[bIdx].ts<<3)|(bn[bIdx].tc-1))-2;for(int i=1;i<4;i++)nd.child[i]=-1;return ni;}
    int ol[4],oc=0;ol[oc++]=bn[bIdx].l;ol[oc++]=bn[bIdx].r;while(oc<4){int bI=-1;float bS=-1;for(int i=0;i<oc;i++){if(bn[ol[i]].l!=-1){float s=saArea(bn[ol[i]].b);if(s>bS){bS=s;bI=i;}}}if(bI==-1)break;int bi=ol[bI];ol[bI]=bn[bi].l;ol[oc++]=bn[bi].r;}
    int ni=(*nOut)++;BVH4H&nd=out[ni];nd.nChildren=oc;for(int i=0;i<oc;i++){int bi=ol[i];nd.minX[i]=bn[bi].b.bmin.x;nd.minY[i]=bn[bi].b.bmin.y;nd.minZ[i]=bn[bi].b.bmin.z;nd.maxX[i]=bn[bi].b.bmax.x;nd.maxY[i]=bn[bi].b.bmax.y;nd.maxZ[i]=bn[bi].b.bmax.z;if(bn[bi].l==-1)nd.child[i]=-((bn[bi].ts<<3)|(bn[bi].tc-1))-2;else nd.child[i]=collapse4(bn,bi,out,nOut);}for(int i=oc;i<4;i++){nd.child[i]=-1;nd.minX[i]=nd.maxX[i]=nd.minY[i]=nd.maxY[i]=nd.minZ[i]=nd.maxZ[i]=0;}return ni;}
void reorderBVH4(BVH4H*src,int n4,BVH4H*dst,int*remap){int*stk=(int*)malloc(n4*4);int sp=0,out=0;stk[sp++]=0;while(sp>0){int i=stk[--sp];remap[i]=out;dst[out++]=src[i];for(int c=src[i].nChildren-1;c>=0;c--)if(src[i].child[c]>=0)stk[sp++]=src[i].child[c];}for(int i=0;i<n4;i++)for(int c=0;c<4;c++)if(dst[i].child[c]>=0)dst[i].child[c]=remap[dst[i].child[c]];free(stk);}
void packBVH4GPU(BVH4H*nodes,int n4,int4*gpuData){float eps=5e-4f;for(int i=0;i<n4;i++){BVH4H&nd=nodes[i];unsigned short hMnX[4],hMnY[4],hMnZ[4],hMxX[4],hMxY[4],hMxZ[4];for(int c=0;c<4;c++){if(nd.child[c]==-1){hMnX[c]=hMnY[c]=hMnZ[c]=0x7C00;hMxX[c]=hMxY[c]=hMxZ[c]=0xFC00;}else{__half h;h=__float2half(nd.minX[c]-eps);memcpy(&hMnX[c],&h,2);h=__float2half(nd.minY[c]-eps);memcpy(&hMnY[c],&h,2);h=__float2half(nd.minZ[c]-eps);memcpy(&hMnZ[c],&h,2);h=__float2half(nd.maxX[c]+eps);memcpy(&hMxX[c],&h,2);h=__float2half(nd.maxY[c]+eps);memcpy(&hMxY[c],&h,2);h=__float2half(nd.maxZ[c]+eps);memcpy(&hMxZ[c],&h,2);}}gpuData[i*4+0]=make_int4((hMnX[1]<<16)|hMnX[0],(hMnX[3]<<16)|hMnX[2],(hMxX[1]<<16)|hMxX[0],(hMxX[3]<<16)|hMxX[2]);gpuData[i*4+1]=make_int4((hMnY[1]<<16)|hMnY[0],(hMnY[3]<<16)|hMnY[2],(hMxY[1]<<16)|hMxY[0],(hMxY[3]<<16)|hMxY[2]);gpuData[i*4+2]=make_int4((hMnZ[1]<<16)|hMnZ[0],(hMnZ[3]<<16)|hMnZ[2],(hMxZ[1]<<16)|hMxZ[0],(hMxZ[3]<<16)|hMxZ[2]);gpuData[i*4+3]=make_int4(nd.child[0],nd.child[1],nd.child[2],nd.child[3]);}}

// ═══ BVH8 COLLAPSE — expand largest SA child until 8 or all leaves ═══
static int collapse8(BN*bn, int bIdx, BVH8H*out, int*nOut) {
    if(bn[bIdx].l==-1) {
        int ni=(*nOut)++;
        BVH8H&nd=out[ni]; nd.nChildren=1;
        nd.parentAABB = bn[bIdx].b;
        nd.minX[0]=bn[bIdx].b.bmin.x; nd.minY[0]=bn[bIdx].b.bmin.y; nd.minZ[0]=bn[bIdx].b.bmin.z;
        nd.maxX[0]=bn[bIdx].b.bmax.x; nd.maxY[0]=bn[bIdx].b.bmax.y; nd.maxZ[0]=bn[bIdx].b.bmax.z;
        nd.child[0] = -((bn[bIdx].ts<<3)|(bn[bIdx].tc-1))-2;
        for(int i=1;i<8;i++) nd.child[i]=-1;
        return ni;
    }
    int ol[8],oc=0;
    ol[oc++]=bn[bIdx].l; ol[oc++]=bn[bIdx].r;
    while(oc<8) {
        int bI=-1; float bS=-1;
        for(int i=0;i<oc;i++) {
            if(bn[ol[i]].l!=-1) {
                float s=saArea(bn[ol[i]].b);
                if(s>bS){bS=s;bI=i;}
            }
        }
        if(bI==-1) break;
        int bi=ol[bI];
        ol[bI]=bn[bi].l;
        ol[oc++]=bn[bi].r;
    }
    int ni=(*nOut)++;
    BVH8H&nd=out[ni]; nd.nChildren=oc;
    nd.parentAABB = bn[bIdx].b;
    for(int i=0;i<oc;i++) {
        int bi=ol[i];
        nd.minX[i]=bn[bi].b.bmin.x; nd.minY[i]=bn[bi].b.bmin.y; nd.minZ[i]=bn[bi].b.bmin.z;
        nd.maxX[i]=bn[bi].b.bmax.x; nd.maxY[i]=bn[bi].b.bmax.y; nd.maxZ[i]=bn[bi].b.bmax.z;
        if(bn[bi].l==-1) nd.child[i] = -((bn[bi].ts<<3)|(bn[bi].tc-1))-2;
        else nd.child[i] = collapse8(bn, bi, out, nOut);
    }
    for(int i=oc;i<8;i++) nd.child[i]=-1;
    return ni;
}

void reorderBVH8(BVH8H*src,int n8,BVH8H*dst,int*remap) {
    int*stk=(int*)malloc(n8*4);int sp=0,out=0;
    stk[sp++]=0;
    while(sp>0) {
        int i=stk[--sp]; remap[i]=out; dst[out++]=src[i];
        for(int c=src[i].nChildren-1;c>=0;c--)
            if(src[i].child[c]>=0) stk[sp++]=src[i].child[c];
    }
    for(int i=0;i<n8;i++)
        for(int c=0;c<8;c++)
            if(dst[i].child[c]>=0) dst[i].child[c]=remap[dst[i].child[c]];
    free(stk);
}

// ═══ QUANTIZED BVH8 PACKING — 96 bytes = 6 × int4 ═══
// Layout:
//   int4[0]: parentAABB as 6×FP16 (pmin xyz, pext xyz) + qminX[0..3]
//   int4[1]: qminX[4..7], qmaxX[0..3], qmaxX[4..7], qminY[0..3]
//   int4[2]: qminY[4..7], qmaxY[0..3], qmaxY[4..7], qminZ[0..3]
//   int4[3]: qminZ[4..7], qmaxZ[0..3], qmaxZ[4..7], padding
//   int4[4]: child[0..3]
//   int4[5]: child[4..7]
static inline unsigned char quantize_u8(float val, float origin, float extent) {
    if(extent < 1e-7f) return 0;
    float t = (val - origin) / extent;
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    return (unsigned char)(t * 255.0f + 0.5f);
}

void packBVH8GPU(BVH8H*nodes, int n8, int4*gpuData) {
    float eps = 5e-4f;
    for(int i=0;i<n8;i++) {
        BVH8H&nd = nodes[i];
        AABB pbb = nd.parentAABB;
        // Conservative parent expansion
        float pminx = pbb.bmin.x - eps, pminy = pbb.bmin.y - eps, pminz = pbb.bmin.z - eps;
        float pextx = (pbb.bmax.x+eps) - pminx;
        float pexty = (pbb.bmax.y+eps) - pminy;
        float pextz = (pbb.bmax.z+eps) - pminz;

        // Quantize children
        unsigned char qmnX[8],qmxX[8],qmnY[8],qmxY[8],qmnZ[8],qmxZ[8];
        for(int c=0;c<8;c++) {
            if(nd.child[c]==-1) {
                // Empty child: set min>max so slab always misses
                qmnX[c]=255; qmxX[c]=0;
                qmnY[c]=255; qmxY[c]=0;
                qmnZ[c]=255; qmxZ[c]=0;
            } else {
                // Conservative: floor min, ceil max
                qmnX[c] = quantize_u8(nd.minX[c]-eps, pminx, pextx);
                qmxX[c] = quantize_u8(nd.maxX[c]+eps, pminx, pextx);
                qmnY[c] = quantize_u8(nd.minY[c]-eps, pminy, pexty);
                qmxY[c] = quantize_u8(nd.maxY[c]+eps, pminy, pexty);
                qmnZ[c] = quantize_u8(nd.minZ[c]-eps, pminz, pextz);
                qmxZ[c] = quantize_u8(nd.maxZ[c]+eps, pminz, pextz);
                // Ensure min <= max after quantization
                if(qmnX[c] > 0) qmnX[c]--;
                if(qmxX[c] < 255) qmxX[c]++;
                if(qmnY[c] > 0) qmnY[c]--;
                if(qmxY[c] < 255) qmxY[c]++;
                if(qmnZ[c] > 0) qmnZ[c]--;
                if(qmxZ[c] < 255) qmxZ[c]++;
            }
        }

        // Pack parent AABB as FP16
        unsigned short hpminx, hpminy, hpminz, hpextx, hpexty, hpextz;
        __half h;
        h=__float2half(pminx); memcpy(&hpminx,&h,2);
        h=__float2half(pminy); memcpy(&hpminy,&h,2);
        h=__float2half(pminz); memcpy(&hpminz,&h,2);
        h=__float2half(pextx); memcpy(&hpextx,&h,2);
        h=__float2half(pexty); memcpy(&hpexty,&h,2);
        h=__float2half(pextz); memcpy(&hpextz,&h,2);

        #define P4(a,b,c,d) ((int)((a)|((b)<<8)|((c)<<16)|((d)<<24)))
        int base = i*6;
        gpuData[base+0] = make_int4(
            (hpminy<<16)|hpminx,
            (hpextx<<16)|hpminz,
            (hpextz<<16)|hpexty,
            P4(qmnX[0],qmnX[1],qmnX[2],qmnX[3]));
        gpuData[base+1] = make_int4(
            P4(qmnX[4],qmnX[5],qmnX[6],qmnX[7]),
            P4(qmxX[0],qmxX[1],qmxX[2],qmxX[3]),
            P4(qmxX[4],qmxX[5],qmxX[6],qmxX[7]),
            P4(qmnY[0],qmnY[1],qmnY[2],qmnY[3]));
        gpuData[base+2] = make_int4(
            P4(qmnY[4],qmnY[5],qmnY[6],qmnY[7]),
            P4(qmxY[0],qmxY[1],qmxY[2],qmxY[3]),
            P4(qmxY[4],qmxY[5],qmxY[6],qmxY[7]),
            P4(qmnZ[0],qmnZ[1],qmnZ[2],qmnZ[3]));
        gpuData[base+3] = make_int4(
            P4(qmnZ[4],qmnZ[5],qmnZ[6],qmnZ[7]),
            P4(qmxZ[0],qmxZ[1],qmxZ[2],qmxZ[3]),
            P4(qmxZ[4],qmxZ[5],qmxZ[6],qmxZ[7]),
            0);
        gpuData[base+4] = make_int4(nd.child[0],nd.child[1],nd.child[2],nd.child[3]);
        gpuData[base+5] = make_int4(nd.child[4],nd.child[5],nd.child[6],nd.child[7]);
        #undef P4
    }
}

// ═══ KERNEL A: BVH4 FP16 (proven baseline) ═══
__global__ void __launch_bounds__(256,5) traceBVH4(
    const int4*__restrict__ d_bvh4,int n4,int _u,
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    const float*__restrict__ rix,const float*__restrict__ riy,const float*__restrict__ riz,
    Hit*__restrict__ hits,int numRays,unsigned long long*__restrict__ stats)
{
    int lane=threadIdx.x&31;
    while(true){
        int bs;if(lane==0)bs=atomicAdd(&g_rayCounterA,32);
        bs=__shfl_sync(0xFFFFFFFF,bs,0);
        if(bs>=numRays)break;int ri=bs+lane;if(ri>=numRays)continue;
        float ox=rox[ri],oy=roy[ri],oz=roz[ri],dx=rdx[ri],dy=rdy[ri],dz=rdz[ri],ix=rix[ri],iy=riy[ri],iz=riz[ri];
        float hitT=1e30f;int hitTri=-1;float hitU=0,hitV=0;int stk[STACK4];int sp=0,ni=0;
        while(true){
            if(ni==-1){if(sp>0)ni=stk[--sp];else break;continue;}
            if(ni<=-2){int val=-(ni+2),tc=(val&7)+1,ts=val>>3;
                for(int i=0;i<tc;i++){int ti=ts+i;
                    float v0x=tv0x[ti],v0y=tv0y[ti],v0z=tv0z[ti];float e1x=tv1x[ti]-v0x,e1y=tv1y[ti]-v0y,e1z=tv1z[ti]-v0z;float e2x=tv2x[ti]-v0x,e2y=tv2y[ti]-v0y,e2z=tv2z[ti]-v0z;float hx=dy*e2z-dz*e2y,hy=dz*e2x-dx*e2z,hz=dx*e2y-dy*e2x;float a2=e1x*hx+e1y*hy+e1z*hz;if(fabsf(a2)<1e-8f)continue;float f=__frcp_rn(a2);float sx=ox-v0x,sy=oy-v0y,sz=oz-v0z;float u=f*(sx*hx+sy*hy+sz*hz);if(u<0.f||u>1.f)continue;float qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x;float v=f*(dx*qx+dy*qy+dz*qz);if(v<0.f||u+v>1.f)continue;float tt=f*(e2x*qx+e2y*qy+e2z*qz);if(tt>0.001f&&tt<hitT){hitT=tt;hitTri=ti;hitU=u;hitV=v;}}
                if(sp>0)ni=stk[--sp];else break;continue;}
            int4 bx=__ldg(&d_bvh4[ni*4]),by=__ldg(&d_bvh4[ni*4+1]),bz=__ldg(&d_bvh4[ni*4+2]),ch=__ldg(&d_bvh4[ni*4+3]);
            float mn0x,mn1x,mn2x,mn3x,mx0x,mx1x,mx2x,mx3x,mn0y,mn1y,mn2y,mn3y,mx0y,mx1y,mx2y,mx3y,mn0z,mn1z,mn2z,mn3z,mx0z,mx1z,mx2z,mx3z;
            d4h(bx.x,bx.y,mn0x,mn1x,mn2x,mn3x);d4h(bx.z,bx.w,mx0x,mx1x,mx2x,mx3x);
            d4h(by.x,by.y,mn0y,mn1y,mn2y,mn3y);d4h(by.z,by.w,mx0y,mx1y,mx2y,mx3y);
            d4h(bz.x,bz.y,mn0z,mn1z,mn2z,mn3z);d4h(bz.z,bz.w,mx0z,mx1z,mx2z,mx3z);
            float t0n,t0x2,t1n,t1x2,t2n,t2x2,t3n,t3x2;
            {float a=(mn0x-ox)*ix,b=(mx0x-ox)*ix;t0n=fminf(a,b);t0x2=fmaxf(a,b);a=(mn0y-oy)*iy;b=(mx0y-oy)*iy;t0n=fmaxf(t0n,fminf(a,b));t0x2=fminf(t0x2,fmaxf(a,b));a=(mn0z-oz)*iz;b=(mx0z-oz)*iz;t0n=fmaxf(t0n,fminf(a,b));t0x2=fminf(t0x2,fmaxf(a,b));}
            {float a=(mn1x-ox)*ix,b=(mx1x-ox)*ix;t1n=fminf(a,b);t1x2=fmaxf(a,b);a=(mn1y-oy)*iy;b=(mx1y-oy)*iy;t1n=fmaxf(t1n,fminf(a,b));t1x2=fminf(t1x2,fmaxf(a,b));a=(mn1z-oz)*iz;b=(mx1z-oz)*iz;t1n=fmaxf(t1n,fminf(a,b));t1x2=fminf(t1x2,fmaxf(a,b));}
            {float a=(mn2x-ox)*ix,b=(mx2x-ox)*ix;t2n=fminf(a,b);t2x2=fmaxf(a,b);a=(mn2y-oy)*iy;b=(mx2y-oy)*iy;t2n=fmaxf(t2n,fminf(a,b));t2x2=fminf(t2x2,fmaxf(a,b));a=(mn2z-oz)*iz;b=(mx2z-oz)*iz;t2n=fmaxf(t2n,fminf(a,b));t2x2=fminf(t2x2,fmaxf(a,b));}
            {float a=(mn3x-ox)*ix,b=(mx3x-ox)*ix;t3n=fminf(a,b);t3x2=fmaxf(a,b);a=(mn3y-oy)*iy;b=(mx3y-oy)*iy;t3n=fmaxf(t3n,fminf(a,b));t3x2=fminf(t3x2,fmaxf(a,b));a=(mn3z-oz)*iz;b=(mx3z-oz)*iz;t3n=fmaxf(t3n,fminf(a,b));t3x2=fminf(t3x2,fmaxf(a,b));}
            int c0=ch.x,c1=ch.y,c2=ch.z,c3=ch.w;
            bool h0=(c0!=-1)&&(t0x2>=fmaxf(t0n,0.f))&&(t0n<hitT);bool h1=(c1!=-1)&&(t1x2>=fmaxf(t1n,0.f))&&(t1n<hitT);
            bool h2=(c2!=-1)&&(t2x2>=fmaxf(t2n,0.f))&&(t2n<hitT);bool h3=(c3!=-1)&&(t3x2>=fmaxf(t3n,0.f))&&(t3n<hitT);
            int nh=h0+h1+h2+h3;if(nh==0){if(sp>0)ni=stk[--sp];else break;continue;}
            float dd[4]={h0?t0n:1e30f,h1?t1n:1e30f,h2?t2n:1e30f,h3?t3n:1e30f};int ci[4]={c0,c1,c2,c3};
            cswap(dd[0],dd[1],ci[0],ci[1]);cswap(dd[2],dd[3],ci[2],ci[3]);cswap(dd[0],dd[2],ci[0],ci[2]);cswap(dd[1],dd[3],ci[1],ci[3]);cswap(dd[1],dd[2],ci[1],ci[2]);
            for(int i=nh-1;i>=1&&sp<STACK4;i--)stk[sp++]=ci[i];ni=ci[0];}
        hits[ri].t=hitT;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;}
}

// ═══ KERNEL B: BVH8 QUANTIZED — 2×4 batch slab test ═══
__global__ void __launch_bounds__(256,4) traceBVH8Q(
    const int4*__restrict__ d_bvh8,int n8,int _u,
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    const float*__restrict__ rix,const float*__restrict__ riy,const float*__restrict__ riz,
    Hit*__restrict__ hits,int numRays,unsigned long long*__restrict__ stats)
{
    int lane=threadIdx.x&31;
    const float inv255 = 1.0f / 255.0f;
    while(true){
        int bs;if(lane==0)bs=atomicAdd(&g_rayCounterB,32);
        bs=__shfl_sync(0xFFFFFFFF,bs,0);
        if(bs>=numRays)break;int ri=bs+lane;if(ri>=numRays)continue;
        float ox=rox[ri],oy=roy[ri],oz=roz[ri],dx=rdx[ri],dy=rdy[ri],dz=rdz[ri],ix=rix[ri],iy=riy[ri],iz=riz[ri];
        float hitT=1e30f;int hitTri=-1;float hitU=0,hitV=0;
        int stk[STACK8];int sp=0,ni=0;
        while(true){
            if(ni==-1){if(sp>0)ni=stk[--sp];else break;continue;}
            if(ni<=-2){int val=-(ni+2),tc=(val&7)+1,ts=val>>3;
                for(int i=0;i<tc;i++){int ti=ts+i;
                    float v0x=tv0x[ti],v0y=tv0y[ti],v0z=tv0z[ti];float e1x=tv1x[ti]-v0x,e1y=tv1y[ti]-v0y,e1z=tv1z[ti]-v0z;float e2x=tv2x[ti]-v0x,e2y=tv2y[ti]-v0y,e2z=tv2z[ti]-v0z;float hx=dy*e2z-dz*e2y,hy=dz*e2x-dx*e2z,hz=dx*e2y-dy*e2x;float a2=e1x*hx+e1y*hy+e1z*hz;if(fabsf(a2)<1e-8f)continue;float f=__frcp_rn(a2);float sx=ox-v0x,sy=oy-v0y,sz=oz-v0z;float u=f*(sx*hx+sy*hy+sz*hz);if(u<0.f||u>1.f)continue;float qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x;float v=f*(dx*qx+dy*qy+dz*qz);if(v<0.f||u+v>1.f)continue;float tt=f*(e2x*qx+e2y*qy+e2z*qz);if(tt>0.001f&&tt<hitT){hitT=tt;hitTri=ti;hitU=u;hitV=v;}}
                if(sp>0)ni=stk[--sp];else break;continue;}

            // Load 6 int4s (96 bytes)
            int base = ni*6;
            int4 hdr  = __ldg(&d_bvh8[base+0]);
            int4 bndX = __ldg(&d_bvh8[base+1]);
            int4 bndY = __ldg(&d_bvh8[base+2]);
            int4 bndZ = __ldg(&d_bvh8[base+3]);
            int4 ch03 = __ldg(&d_bvh8[base+4]);
            int4 ch47 = __ldg(&d_bvh8[base+5]);

            // Decode parent AABB
            float pmnx = __half2float(__ushort_as_half((unsigned short)(hdr.x)));
            float pmny = __half2float(__ushort_as_half((unsigned short)(hdr.x>>16)));
            float pmnz = __half2float(__ushort_as_half((unsigned short)(hdr.y)));
            float pexX = __half2float(__ushort_as_half((unsigned short)(hdr.y>>16)));
            float pexY = __half2float(__ushort_as_half((unsigned short)(hdr.z)));
            float pexZ = __half2float(__ushort_as_half((unsigned short)(hdr.z>>16)));
            float sX = pexX*inv255, sY = pexY*inv255, sZ = pexZ*inv255;

            // ── BATCH 1: children 0-3 ──
            float mn0x,mn1x,mn2x,mn3x,mx0x,mx1x,mx2x,mx3x;
            float mn0y,mn1y,mn2y,mn3y,mx0y,mx1y,mx2y,mx3y;
            float mn0z,mn1z,mn2z,mn3z,mx0z,mx1z,mx2z,mx3z;
            extract4u8(hdr.w,  sX, pmnx, mn0x,mn1x,mn2x,mn3x);
            extract4u8(bndX.y, sX, pmnx, mx0x,mx1x,mx2x,mx3x);
            extract4u8(bndX.w, sY, pmny, mn0y,mn1y,mn2y,mn3y);
            extract4u8(bndY.y, sY, pmny, mx0y,mx1y,mx2y,mx3y);
            extract4u8(bndY.w, sZ, pmnz, mn0z,mn1z,mn2z,mn3z);
            extract4u8(bndZ.y, sZ, pmnz, mx0z,mx1z,mx2z,mx3z);

            float t0n,t0x2,t1n,t1x2,t2n,t2x2,t3n,t3x2;
            {float a=(mn0x-ox)*ix,b=(mx0x-ox)*ix;t0n=fminf(a,b);t0x2=fmaxf(a,b);a=(mn0y-oy)*iy;b=(mx0y-oy)*iy;t0n=fmaxf(t0n,fminf(a,b));t0x2=fminf(t0x2,fmaxf(a,b));a=(mn0z-oz)*iz;b=(mx0z-oz)*iz;t0n=fmaxf(t0n,fminf(a,b));t0x2=fminf(t0x2,fmaxf(a,b));}
            {float a=(mn1x-ox)*ix,b=(mx1x-ox)*ix;t1n=fminf(a,b);t1x2=fmaxf(a,b);a=(mn1y-oy)*iy;b=(mx1y-oy)*iy;t1n=fmaxf(t1n,fminf(a,b));t1x2=fminf(t1x2,fmaxf(a,b));a=(mn1z-oz)*iz;b=(mx1z-oz)*iz;t1n=fmaxf(t1n,fminf(a,b));t1x2=fminf(t1x2,fmaxf(a,b));}
            {float a=(mn2x-ox)*ix,b=(mx2x-ox)*ix;t2n=fminf(a,b);t2x2=fmaxf(a,b);a=(mn2y-oy)*iy;b=(mx2y-oy)*iy;t2n=fmaxf(t2n,fminf(a,b));t2x2=fminf(t2x2,fmaxf(a,b));a=(mn2z-oz)*iz;b=(mx2z-oz)*iz;t2n=fmaxf(t2n,fminf(a,b));t2x2=fminf(t2x2,fmaxf(a,b));}
            {float a=(mn3x-ox)*ix,b=(mx3x-ox)*ix;t3n=fminf(a,b);t3x2=fmaxf(a,b);a=(mn3y-oy)*iy;b=(mx3y-oy)*iy;t3n=fmaxf(t3n,fminf(a,b));t3x2=fminf(t3x2,fmaxf(a,b));a=(mn3z-oz)*iz;b=(mx3z-oz)*iz;t3n=fmaxf(t3n,fminf(a,b));t3x2=fminf(t3x2,fmaxf(a,b));}

            int c0=ch03.x,c1=ch03.y,c2=ch03.z,c3=ch03.w;
            // Collect hits from batch 1
            float hd[8]; int hc[8]; int nh=0;
            if(c0!=-1 && t0x2>=fmaxf(t0n,0.f) && t0n<hitT){hd[nh]=t0n;hc[nh]=c0;nh++;}
            if(c1!=-1 && t1x2>=fmaxf(t1n,0.f) && t1n<hitT){hd[nh]=t1n;hc[nh]=c1;nh++;}
            if(c2!=-1 && t2x2>=fmaxf(t2n,0.f) && t2n<hitT){hd[nh]=t2n;hc[nh]=c2;nh++;}
            if(c3!=-1 && t3x2>=fmaxf(t3n,0.f) && t3n<hitT){hd[nh]=t3n;hc[nh]=c3;nh++;}

            // ── BATCH 2: children 4-7 ──
            extract4u8(bndX.x, sX, pmnx, mn0x,mn1x,mn2x,mn3x);
            extract4u8(bndX.z, sX, pmnx, mx0x,mx1x,mx2x,mx3x);
            extract4u8(bndY.x, sY, pmny, mn0y,mn1y,mn2y,mn3y);
            extract4u8(bndY.z, sY, pmny, mx0y,mx1y,mx2y,mx3y);
            extract4u8(bndZ.x, sZ, pmnz, mn0z,mn1z,mn2z,mn3z);
            extract4u8(bndZ.z, sZ, pmnz, mx0z,mx1z,mx2z,mx3z);

            {float a=(mn0x-ox)*ix,b=(mx0x-ox)*ix;t0n=fminf(a,b);t0x2=fmaxf(a,b);a=(mn0y-oy)*iy;b=(mx0y-oy)*iy;t0n=fmaxf(t0n,fminf(a,b));t0x2=fminf(t0x2,fmaxf(a,b));a=(mn0z-oz)*iz;b=(mx0z-oz)*iz;t0n=fmaxf(t0n,fminf(a,b));t0x2=fminf(t0x2,fmaxf(a,b));}
            {float a=(mn1x-ox)*ix,b=(mx1x-ox)*ix;t1n=fminf(a,b);t1x2=fmaxf(a,b);a=(mn1y-oy)*iy;b=(mx1y-oy)*iy;t1n=fmaxf(t1n,fminf(a,b));t1x2=fminf(t1x2,fmaxf(a,b));a=(mn1z-oz)*iz;b=(mx1z-oz)*iz;t1n=fmaxf(t1n,fminf(a,b));t1x2=fminf(t1x2,fmaxf(a,b));}
            {float a=(mn2x-ox)*ix,b=(mx2x-ox)*ix;t2n=fminf(a,b);t2x2=fmaxf(a,b);a=(mn2y-oy)*iy;b=(mx2y-oy)*iy;t2n=fmaxf(t2n,fminf(a,b));t2x2=fminf(t2x2,fmaxf(a,b));a=(mn2z-oz)*iz;b=(mx2z-oz)*iz;t2n=fmaxf(t2n,fminf(a,b));t2x2=fminf(t2x2,fmaxf(a,b));}
            {float a=(mn3x-ox)*ix,b=(mx3x-ox)*ix;t3n=fminf(a,b);t3x2=fmaxf(a,b);a=(mn3y-oy)*iy;b=(mx3y-oy)*iy;t3n=fmaxf(t3n,fminf(a,b));t3x2=fminf(t3x2,fmaxf(a,b));a=(mn3z-oz)*iz;b=(mx3z-oz)*iz;t3n=fmaxf(t3n,fminf(a,b));t3x2=fminf(t3x2,fmaxf(a,b));}

            int c4=ch47.x,c5=ch47.y,c6=ch47.z,c7=ch47.w;
            if(c4!=-1 && t0x2>=fmaxf(t0n,0.f) && t0n<hitT){hd[nh]=t0n;hc[nh]=c4;nh++;}
            if(c5!=-1 && t1x2>=fmaxf(t1n,0.f) && t1n<hitT){hd[nh]=t1n;hc[nh]=c5;nh++;}
            if(c6!=-1 && t2x2>=fmaxf(t2n,0.f) && t2n<hitT){hd[nh]=t2n;hc[nh]=c6;nh++;}
            if(c7!=-1 && t3x2>=fmaxf(t3n,0.f) && t3n<hitT){hd[nh]=t3n;hc[nh]=c7;nh++;}

            if(nh==0){if(sp>0)ni=stk[--sp];else break;continue;}

            // Insertion sort (typically 1-3 hits, very cheap)
            for(int a=1;a<nh;a++){
                float kd=hd[a]; int kc=hc[a]; int b=a-1;
                while(b>=0 && hd[b]>kd){hd[b+1]=hd[b];hc[b+1]=hc[b];b--;}
                hd[b+1]=kd; hc[b+1]=kc;
            }
            // Push farthest first, traverse nearest
            for(int i=nh-1;i>=1&&sp<STACK8;i--) stk[sp++]=hc[i];
            ni=hc[0];
        }
        hits[ri].t=hitT;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;}
}

// ═══ SCENE ═══
void addQuad(Tri*t,int&ti,float3 a,float3 b,float3 c,float3 d){t[ti++]={a,b,c};t[ti++]={a,c,d};}
void addBox(Tri*t,int&ti,float3 mn,float3 mx){float3 a={mn.x,mn.y,mn.z},b={mx.x,mn.y,mn.z},cv={mx.x,mx.y,mn.z},d={mn.x,mx.y,mn.z};float3 e={mn.x,mn.y,mx.z},f={mx.x,mn.y,mx.z},g={mx.x,mx.y,mx.z},h={mn.x,mx.y,mx.z};addQuad(t,ti,a,b,cv,d);addQuad(t,ti,e,f,g,h);addQuad(t,ti,a,b,f,e);addQuad(t,ti,d,cv,g,h);addQuad(t,ti,a,d,h,e);addQuad(t,ti,b,cv,g,f);}
void addSubQuad(Tri*t,int&ti,float3 o,float3 ux,float3 uy,int nx,int ny){for(int i=0;i<nx;i++)for(int j=0;j<ny;j++){float u0=(float)i/nx,u1=(float)(i+1)/nx,v0=(float)j/ny,v1=(float)(j+1)/ny;float3 a={o.x+ux.x*u0+uy.x*v0,o.y+ux.y*u0+uy.y*v0,o.z+ux.z*u0+uy.z*v0};float3 b={o.x+ux.x*u1+uy.x*v0,o.y+ux.y*u1+uy.y*v0,o.z+ux.z*u1+uy.z*v0};float3 cv={o.x+ux.x*u1+uy.x*v1,o.y+ux.y*u1+uy.y*v1,o.z+ux.z*u1+uy.z*v1};float3 d2={o.x+ux.x*u0+uy.x*v1,o.y+ux.y*u0+uy.y*v1,o.z+ux.z*u0+uy.z*v1};t[ti++]={a,b,cv};t[ti++]={a,cv,d2};}}
int genConference(Tri*t,int maxTris){int ti=0;float W=10,H=5,D=7.5f;int subdiv=(int)sqrtf((float)maxTris/60);if(subdiv<2)subdiv=2;if(subdiv>200)subdiv=200;addSubQuad(t,ti,{-W,0,-D},{2*W,0,0},{0,0,2*D},subdiv,subdiv);addSubQuad(t,ti,{-W,H,-D},{2*W,0,0},{0,0,2*D},subdiv,subdiv);addSubQuad(t,ti,{-W,0,-D},{2*W,0,0},{0,H,0},subdiv,subdiv/2);addSubQuad(t,ti,{-W,0,D},{2*W,0,0},{0,H,0},subdiv,subdiv/2);addSubQuad(t,ti,{-W,0,-D},{0,0,2*D},{0,H,0},subdiv,subdiv/2);addSubQuad(t,ti,{W,0,-D},{0,0,2*D},{0,H,0},subdiv,subdiv/2);srand(42);int numT=maxTris>50000?20:8;for(int i=0;i<numT&&ti+12<maxTris;i++){float tx=((float)rand()/RAND_MAX)*16-8,tz=((float)rand()/RAND_MAX)*12-6;addBox(t,ti,{tx-1.f,.7f,tz-.5f},{tx+1.f,.8f,tz+.5f});addBox(t,ti,{tx-.9f,0,tz-.05f},{tx-.8f,.7f,tz+.05f});addBox(t,ti,{tx+.8f,0,tz-.05f},{tx+.9f,.7f,tz+.05f});}int numC=maxTris>50000?40:16;for(int i=0;i<numC&&ti+12<maxTris;i++){float cx=((float)rand()/RAND_MAX)*18-9,cz=((float)rand()/RAND_MAX)*14-7;addBox(t,ti,{cx-.25f,.4f,cz-.25f},{cx+.25f,.45f,cz+.25f});addBox(t,ti,{cx-.25f,.45f,cz-.25f},{cx+.25f,.9f,cz-.2f});}while(ti+2<maxTris){float cx=((float)rand()/RAND_MAX)*16-8,cy=.8f+((float)rand()/RAND_MAX)*.3f;float cz=((float)rand()/RAND_MAX)*12-6,s=.05f+((float)rand()/RAND_MAX)*.1f;t[ti].v0={cx-s,cy,cz-s};t[ti].v1={cx+s,cy,cz+s};t[ti].v2={cx,cy+s*2,cz};ti++;t[ti].v0={cx-s,cy,cz+s};t[ti].v1={cx+s,cy,cz-s};t[ti].v2={cx,cy+s*2,cz};ti++;}return ti;}
struct RayAoS{float3 o,d,id;};
int octant(float3 d){return (d.x<0?4:0)|(d.y<0?2:0)|(d.z<0?1:0);}
static inline uint32_t expand3(uint32_t v){v&=0x3FF;v=(v|(v<<16))&0x30000FF;v=(v|(v<<8))&0x300F00F;v=(v|(v<<4))&0x30C30C3;v=(v|(v<<2))&0x9249249;return v;}
uint32_t morton3D(float x,float y,float z,float3 mn,float3 mx){float nx=(x-mn.x)/(mx.x-mn.x+1e-7f),ny=(y-mn.y)/(mx.y-mn.y+1e-7f),nz=(z-mn.z)/(mx.z-mn.z+1e-7f);uint32_t ix2=fminf(fmaxf(nx*1023.f,0.f),1023.f),iy2=fminf(fmaxf(ny*1023.f,0.f),1023.f),iz2=fminf(fmaxf(nz*1023.f,0.f),1023.f);return expand3(ix2)|(expand3(iy2)<<1)|(expand3(iz2)<<2);}
void sortMortonOctant(RayAoS*r,int n,float3 smn,float3 smx){struct SK{uint32_t key;int idx;};SK*keys=(SK*)malloc(n*sizeof(SK));for(int i=0;i<n;i++){uint32_t m=morton3D(r[i].o.x,r[i].o.y,r[i].o.z,smn,smx);uint32_t o=(uint32_t)octant(r[i].d);keys[i]={(o<<27)|(m>>3),i};}std::sort(keys,keys+n,[](const SK&a,const SK&b){return a.key<b.key;});RayAoS*tmp=(RayAoS*)malloc(n*sizeof(RayAoS));for(int i=0;i<n;i++)tmp[i]=r[keys[i].idx];memcpy(r,tmp,n*sizeof(RayAoS));free(tmp);free(keys);}
void genPrimary(RayAoS*r,int n){int w=(int)sqrtf((float)n);for(int i=0;i<n;i++){int px=i%w,py=i/w;float u=(2.f*px/w-1.f)*1.2f,v=(2.f*py/(n/w)-1.f)*.6f;r[i].o={0,2.5f,12};float3 d={u,-v,-1.5f};float l=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);d.x/=l;d.y/=l;d.z/=l;r[i].d=d;r[i].id={1.f/d.x,1.f/d.y,1.f/d.z};}}
void genDiffuse(RayAoS*r,int n,Tri*tris,int nt){srand(1337);for(int i=0;i<n;i++){int ti=rand()%nt;Tri&tr=tris[ti];float u=((float)rand()/RAND_MAX),v=((float)rand()/RAND_MAX);if(u+v>1){u=1-u;v=1-v;}float3 o={tr.v0.x+u*(tr.v1.x-tr.v0.x)+v*(tr.v2.x-tr.v0.x),tr.v0.y+u*(tr.v1.y-tr.v0.y)+v*(tr.v2.y-tr.v0.y),tr.v0.z+u*(tr.v1.z-tr.v0.z)+v*(tr.v2.z-tr.v0.z)};float th=acosf(sqrtf((float)rand()/RAND_MAX)),ph=2.f*3.14159f*((float)rand()/RAND_MAX);float3 d={sinf(th)*cosf(ph),fabsf(cosf(th))+0.01f,sinf(th)*sinf(ph)};float l=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);d.x/=l;d.y/=l;d.z/=l;o.x+=d.x*0.002f;o.y+=d.y*0.002f;o.z+=d.z*0.002f;r[i].o=o;r[i].d=d;r[i].id={1.f/d.x,1.f/d.y,1.f/d.z};}}

typedef void(*KernelFn)(const int4*,int,int,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,Hit*,int,unsigned long long*);
struct BR{double mr;int hitPct;};
BR benchK(KernelFn kern,bool useB,RayAoS*rays,int numRays,int4*d_bvh,int nb_nodes,float*d_v[9],cudaDeviceProp&prop){
    float*h_ray[9];for(int j=0;j<9;j++)h_ray[j]=(float*)malloc(numRays*4);
    for(int i=0;i<numRays;i++){h_ray[0][i]=rays[i].o.x;h_ray[1][i]=rays[i].o.y;h_ray[2][i]=rays[i].o.z;h_ray[3][i]=rays[i].d.x;h_ray[4][i]=rays[i].d.y;h_ray[5][i]=rays[i].d.z;h_ray[6][i]=rays[i].id.x;h_ray[7][i]=rays[i].id.y;h_ray[8][i]=rays[i].id.z;}
    float*d_ray[9];Hit*d_hits;unsigned long long*d_st;
    for(int j=0;j<9;j++){cudaMalloc(&d_ray[j],numRays*4);cudaMemcpy(d_ray[j],h_ray[j],numRays*4,cudaMemcpyHostToDevice);}
    cudaMalloc(&d_hits,numRays*sizeof(Hit));cudaMalloc(&d_st,16);
    int nblk=prop.multiProcessorCount*4;unsigned int z=0;
    cudaMemcpyToSymbol(useB?g_rayCounterB:g_rayCounterA,&z,4);
    kern<<<nblk,256>>>(d_bvh,nb_nodes,0,d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],d_hits,numRays,d_st);
    cudaDeviceSynchronize();
    cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);float total=0;
    for(int r=0;r<20;r++){
        cudaMemcpyToSymbol(useB?g_rayCounterB:g_rayCounterA,&z,4);cudaMemset(d_st,0,16);
        cudaEventRecord(t0);kern<<<nblk,256>>>(d_bvh,nb_nodes,0,d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],d_hits,numRays,d_st);
        cudaEventRecord(t1);cudaEventSynchronize(t1);float ms;cudaEventElapsedTime(&ms,t0,t1);total+=ms;}
    Hit*hh=(Hit*)malloc(numRays*sizeof(Hit));cudaMemcpy(hh,d_hits,numRays*sizeof(Hit),cudaMemcpyDeviceToHost);
    int hc=0;for(int i=0;i<numRays;i++)if(hh[i].tri>=0)hc++;
    BR res;res.mr=(double)numRays/((total/20)/1000.0)/1e6;res.hitPct=100*hc/numRays;
    cudaEventDestroy(t0);cudaEventDestroy(t1);for(int j=0;j<9;j++){cudaFree(d_ray[j]);free(h_ray[j]);}cudaFree(d_hits);cudaFree(d_st);free(hh);return res;}

int main(){
    printf("══════════════════════════════════════════════════════════════════════════\n");
    printf("  V17 — BVH8 Quantized (uint8) + Auto-Threshold\n");
    printf("  BVH4 FP16 (64B) for small scenes | BVH8 uint8 (96B) for large\n");
    printf("  Threshold: BVH4 size > 70%% L2 → switch to BVH8\n");
    printf("══════════════════════════════════════════════════════════════════════════\n\n");
    cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
    int l2 = prop.l2CacheSize;
    printf("  GPU: %s | L2: %d KB\n\n", prop.name, l2/1024);
    float3 sMn={-10,0,-7.5f},sMx={10,5,7.5f};

    for(int sceneK : {100, 300, 500, 1000}) {
        int maxTris=sceneK*1000, numRays=4194304;
        Tri*h_tris=(Tri*)malloc(maxTris*sizeof(Tri));
        int nt=genConference(h_tris,maxTris);

        // Build binary BVH
        BN*h_nodes=(BN*)calloc(nt*2,sizeof(BN));
        int*tidx=(int*)malloc(nt*4);for(int i=0;i<nt;i++)tidx[i]=i;
        int nc=0; buildSAH(h_nodes,h_tris,tidx,nc,0,nt,0);
        BN*h_ord=(BN*)malloc(nc*sizeof(BN));int*remap=(int*)malloc(nc*4);
        treeReorder(h_nodes,nc,h_ord,remap);
        Tri*h_to=(Tri*)malloc(nt*sizeof(Tri));for(int i=0;i<nt;i++)h_to[i]=h_tris[tidx[i]];

        // Build BVH4
        BVH4H*h_b4=(BVH4H*)calloc(nc,sizeof(BVH4H));int n4=0;
        collapse4(h_ord,0,h_b4,&n4);
        BVH4H*h_b4o=(BVH4H*)malloc(n4*sizeof(BVH4H));int*r4=(int*)malloc(n4*4);
        reorderBVH4(h_b4,n4,h_b4o,r4);
        int4*h_gpu4=(int4*)malloc(n4*4*sizeof(int4));packBVH4GPU(h_b4o,n4,h_gpu4);
        int bvh4_bytes = n4*64;

        // Build BVH8
        BVH8H*h_b8=(BVH8H*)calloc(nc,sizeof(BVH8H));int n8=0;
        collapse8(h_ord,0,h_b8,&n8);
        BVH8H*h_b8o=(BVH8H*)malloc(n8*sizeof(BVH8H));int*r8=(int*)malloc(n8*4);
        reorderBVH8(h_b8,n8,h_b8o,r8);
        int4*h_gpu8=(int4*)malloc(n8*6*sizeof(int4));packBVH8GPU(h_b8o,n8,h_gpu8);
        int bvh8_bytes = n8*96;

        float l2pct4 = 100.0f*bvh4_bytes/l2;
        float l2pct8 = 100.0f*bvh8_bytes/l2;
        const char* autoChoice = (bvh4_bytes > l2*0.7) ? "→ BVH8" : "→ BVH4";

        printf("  === %dK tris ===\n", nt/1000);
        printf("  BVH4: %d nodes × 64B = %.2f MB (%.0f%% L2)\n", n4, bvh4_bytes/1048576.0, l2pct4);
        printf("  BVH8: %d nodes × 96B = %.2f MB (%.0f%% L2)\n", n8, bvh8_bytes/1048576.0, l2pct8);
        printf("  Auto-threshold: %s\n", autoChoice);

        // Upload
        int4*d_bvh4; cudaMalloc(&d_bvh4,n4*4*sizeof(int4)); cudaMemcpy(d_bvh4,h_gpu4,n4*4*sizeof(int4),cudaMemcpyHostToDevice);
        int4*d_bvh8; cudaMalloc(&d_bvh8,n8*6*sizeof(int4)); cudaMemcpy(d_bvh8,h_gpu8,n8*6*sizeof(int4),cudaMemcpyHostToDevice);
        float*h_v[9],*d_v[9];for(int j=0;j<9;j++){h_v[j]=(float*)malloc(nt*4);cudaMalloc(&d_v[j],nt*4);}
        for(int i=0;i<nt;i++){h_v[0][i]=h_to[i].v0.x;h_v[1][i]=h_to[i].v0.y;h_v[2][i]=h_to[i].v0.z;h_v[3][i]=h_to[i].v1.x;h_v[4][i]=h_to[i].v1.y;h_v[5][i]=h_to[i].v1.z;h_v[6][i]=h_to[i].v2.x;h_v[7][i]=h_to[i].v2.y;h_v[8][i]=h_to[i].v2.z;}
        for(int j=0;j<9;j++)cudaMemcpy(d_v[j],h_v[j],nt*4,cudaMemcpyHostToDevice);

        RayAoS*rays=(RayAoS*)malloc(numRays*sizeof(RayAoS));
        for(int test=0;test<2;test++){
            const char*name=test==0?"PRIMARY":"DIFFUSE MORTON";
            if(test==0)genPrimary(rays,numRays);
            else{genDiffuse(rays,numRays,h_to,nt);sortMortonOctant(rays,numRays,sMn,sMx);}

            BR r4r=benchK(traceBVH4,false,rays,numRays,d_bvh4,n4,d_v,prop);
            BR r8r=benchK(traceBVH8Q,true,rays,numRays,d_bvh8,n8,d_v,prop);
            double delta=(r8r.mr-r4r.mr)/r4r.mr*100;
            printf("  %-20s BVH4: %6.0f  BVH8Q: %6.0f  %+.1f%%  (hit: %d%%/%d%%)\n",
                name, r4r.mr, r8r.mr, delta, r4r.hitPct, r8r.hitPct);
        }
        printf("\n");
        free(rays);free(h_tris);free(h_nodes);free(tidx);free(h_ord);free(remap);free(h_to);
        free(h_b4);free(h_b4o);free(r4);free(h_gpu4);free(h_b8);free(h_b8o);free(r8);free(h_gpu8);
        for(int j=0;j<9;j++){free(h_v[j]);cudaFree(d_v[j]);}cudaFree(d_bvh4);cudaFree(d_bvh8);
    }
    return 0;
}
