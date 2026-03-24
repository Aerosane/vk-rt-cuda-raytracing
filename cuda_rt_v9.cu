// V9: BVH4 (4-wide) + FP16 compressed bounds + sorted traversal
// Key innovations over v8 binary FP16:
// 1. 4-wide BVH nodes: 4 child AABBs per node → ~20-25 node visits vs ~45
// 2. SAH-guided collapse: binary BVH collapsed to optimal 4-way
// 3. 64 bytes per BVH4 node, loaded as 4 × 128-bit __ldg()
// 4. Sorted child traversal (nearest-first) for optimal pruning
// 5. FP16 bounds with epsilon expansion for containment safety
// Target: 20-25 nodes/ray × 64B = 1280-1600 B/ray, fewer serial deps → 4000+ MR/s

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <algorithm>

#define SAH_BINS 16
#define LEAF_SIZE 4
#define MAX_STACK 32

// BVH4 GPU node: 64 bytes = 4 × int4
// Layout: boundsX[8](FP16) | boundsY[8](FP16) | boundsZ[8](FP16) | child[4](int)
// boundsX = {minX0,minX1,minX2,minX3, maxX0,maxX1,maxX2,maxX3}
// child: >=0 internal, ==-1 empty, <=-2 leaf (encoded)
#define CONST_BVH4 1023
__constant__ int4 c_bvh4[CONST_BVH4 * 4]; // 65472 bytes
__constant__ int c_bvh4N;                   // 4 bytes  → 65476 < 65536

struct Tri{float3 v0,v1,v2;};
struct AABB{float3 bmin,bmax;};
struct Hit{float t;int tri;float u,v;};
struct BN{AABB b;int l,r,ts,tc;}; // binary BVH node

// BVH4 host node (for construction)
struct BVH4H{
    float minX[4],minY[4],minZ[4],maxX[4],maxY[4],maxZ[4];
    int child[4]; // >=0: internal BVH4 idx, -1: empty, <=-2: leaf
    int nChildren; // 2-4 valid children
};

AABB unite(const AABB&a,const AABB&b){return{{fminf(a.bmin.x,b.bmin.x),fminf(a.bmin.y,b.bmin.y),fminf(a.bmin.z,b.bmin.z)},{fmaxf(a.bmax.x,b.bmax.x),fmaxf(a.bmax.y,b.bmax.y),fmaxf(a.bmax.z,b.bmax.z)}};}
AABB triBox(const Tri&t){return{{fminf(fminf(t.v0.x,t.v1.x),t.v2.x),fminf(fminf(t.v0.y,t.v1.y),t.v2.y),fminf(fminf(t.v0.z,t.v1.z),t.v2.z)},{fmaxf(fmaxf(t.v0.x,t.v1.x),t.v2.x),fmaxf(fmaxf(t.v0.y,t.v1.y),t.v2.y),fmaxf(fmaxf(t.v0.z,t.v1.z),t.v2.z)}};}
float3 triCen(const Tri&t){return{(t.v0.x+t.v1.x+t.v2.x)/3,(t.v0.y+t.v1.y+t.v2.y)/3,(t.v0.z+t.v1.z+t.v2.z)/3};}
float saArea(const AABB&b){float dx=b.bmax.x-b.bmin.x,dy=b.bmax.y-b.bmin.y,dz=b.bmax.z-b.bmin.z;return 2.0f*(dx*dy+dy*dz+dz*dx);}

// ═══ BINARY BVH (SAH) ═══
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

// DFS reorder for cache-friendly layout
void treeReorder(BN*src,int nc,BN*dst,int*remap){
    int*stack=(int*)malloc(nc*4);int sp=0,out=0;stack[sp++]=0;
    while(sp>0){int i=stack[--sp];remap[i]=out;dst[out++]=src[i];
        if(src[i].l>=0){stack[sp++]=src[i].r;stack[sp++]=src[i].l;}}
    for(int i=0;i<nc;i++){if(dst[i].l>=0){dst[i].l=remap[dst[i].l];dst[i].r=remap[dst[i].r];}}
    free(stack);
}

// ═══ BINARY → BVH4 COLLAPSE ═══
// SAH-guided: repeatedly expand the largest-SA internal child until 4 children
static int collapseRec(BN*bn, int bIdx, BVH4H*out, int*nOut) {
    if(bn[bIdx].l == -1) {
        // Binary leaf → BVH4 leaf (single-child node)
        int ni = (*nOut)++;
        BVH4H&nd = out[ni];
        nd.nChildren = 1;
        nd.minX[0]=bn[bIdx].b.bmin.x; nd.minY[0]=bn[bIdx].b.bmin.y; nd.minZ[0]=bn[bIdx].b.bmin.z;
        nd.maxX[0]=bn[bIdx].b.bmax.x; nd.maxY[0]=bn[bIdx].b.bmax.y; nd.maxZ[0]=bn[bIdx].b.bmax.z;
        int ts=bn[bIdx].ts, tc=bn[bIdx].tc;
        nd.child[0] = -((ts << 3) | (tc-1)) - 2; // leaf encoding
        for(int i=1;i<4;i++) nd.child[i] = -1; // invalid
        return ni;
    }

    // Internal node: gather up to 4 children
    int openList[4]; // binary node indices
    int openCount = 0;
    openList[openCount++] = bn[bIdx].l;
    openList[openCount++] = bn[bIdx].r;

    while(openCount < 4) {
        // Find largest-SA internal node in openList
        int bestI = -1; float bestSA = -1;
        for(int i = 0; i < openCount; i++) {
            int bi = openList[i];
            if(bn[bi].l != -1) { // internal
                float s = saArea(bn[bi].b);
                if(s > bestSA) { bestSA = s; bestI = i; }
            }
        }
        if(bestI == -1) break; // all leaves
        int bi = openList[bestI];
        openList[bestI] = bn[bi].l; // replace with left
        openList[openCount++] = bn[bi].r; // append right
    }

    // Create BVH4 node
    int ni = (*nOut)++;
    BVH4H&nd = out[ni];
    nd.nChildren = openCount;

    for(int i = 0; i < openCount; i++) {
        int bi = openList[i];
        nd.minX[i]=bn[bi].b.bmin.x; nd.minY[i]=bn[bi].b.bmin.y; nd.minZ[i]=bn[bi].b.bmin.z;
        nd.maxX[i]=bn[bi].b.bmax.x; nd.maxY[i]=bn[bi].b.bmax.y; nd.maxZ[i]=bn[bi].b.bmax.z;
        if(bn[bi].l == -1) {
            // Leaf child
            int ts=bn[bi].ts, tc=bn[bi].tc;
            nd.child[i] = -((ts << 3) | (tc-1)) - 2;
        } else {
            // Internal: recurse
            nd.child[i] = collapseRec(bn, bi, out, nOut);
        }
    }
    for(int i = openCount; i < 4; i++) {
        nd.child[i] = -1; // mark empty
        nd.minX[i]=nd.maxX[i]=nd.minY[i]=nd.maxY[i]=nd.minZ[i]=nd.maxZ[i]=0;
    }
    return ni;
}

// DFS reorder BVH4 for cache locality
void reorderBVH4(BVH4H*src, int n4, BVH4H*dst, int*remap) {
    int*stack=(int*)malloc(n4*4); int sp=0,out=0;
    stack[sp++] = 0;
    while(sp > 0) {
        int i = stack[--sp]; remap[i] = out; dst[out++] = src[i];
        // Push children in reverse order (rightmost first) so leftmost is visited first
        for(int c = src[i].nChildren-1; c >= 0; c--) {
            if(src[i].child[c] >= 0) stack[sp++] = src[i].child[c];
        }
    }
    // Fix child pointers
    for(int i = 0; i < n4; i++) {
        for(int c = 0; c < 4; c++) {
            if(dst[i].child[c] >= 0) dst[i].child[c] = remap[dst[i].child[c]];
        }
    }
    free(stack);
}

// ═══ PACK BVH4 → GPU FORMAT (FP16 bounds + int children) ═══
// Each node = 4 × int4 (64 bytes):
// int4[0]: boundsX = {minX01, minX23, maxX01, maxX23} as packed FP16 pairs
// int4[1]: boundsY same layout
// int4[2]: boundsZ same layout
// int4[3]: {child[0], child[1], child[2], child[3]}
void packBVH4GPU(BVH4H*nodes, int n4, int4*gpuData) {
    float eps = 5e-4f; // epsilon for FP16 containment
    for(int i = 0; i < n4; i++) {
        BVH4H&nd = nodes[i];
        unsigned short hMinX[4],hMinY[4],hMinZ[4],hMaxX[4],hMaxY[4],hMaxZ[4];
        for(int c = 0; c < 4; c++) {
            if(nd.child[c] == -1) {
                // Invalid: set degenerate AABB that never hits
                hMinX[c]=hMinY[c]=hMinZ[c]=0x7C00; // +inf
                hMaxX[c]=hMaxY[c]=hMaxZ[c]=0xFC00; // -inf
            } else {
                __half h;
                h=__float2half(nd.minX[c]-eps); memcpy(&hMinX[c],&h,2);
                h=__float2half(nd.minY[c]-eps); memcpy(&hMinY[c],&h,2);
                h=__float2half(nd.minZ[c]-eps); memcpy(&hMinZ[c],&h,2);
                h=__float2half(nd.maxX[c]+eps); memcpy(&hMaxX[c],&h,2);
                h=__float2half(nd.maxY[c]+eps); memcpy(&hMaxY[c],&h,2);
                h=__float2half(nd.maxZ[c]+eps); memcpy(&hMaxZ[c],&h,2);
            }
        }
        // Pack into int4's: each int4 = 4 ints = 16 bytes
        // boundsX: {(minX1<<16|minX0), (minX3<<16|minX2), (maxX1<<16|maxX0), (maxX3<<16|maxX2)}
        gpuData[i*4+0] = make_int4(
            (hMinX[1]<<16)|hMinX[0], (hMinX[3]<<16)|hMinX[2],
            (hMaxX[1]<<16)|hMaxX[0], (hMaxX[3]<<16)|hMaxX[2]);
        gpuData[i*4+1] = make_int4(
            (hMinY[1]<<16)|hMinY[0], (hMinY[3]<<16)|hMinY[2],
            (hMaxY[1]<<16)|hMaxY[0], (hMaxY[3]<<16)|hMaxY[2]);
        gpuData[i*4+2] = make_int4(
            (hMinZ[1]<<16)|hMinZ[0], (hMinZ[3]<<16)|hMinZ[2],
            (hMaxZ[1]<<16)|hMaxZ[0], (hMaxZ[3]<<16)|hMaxZ[2]);
        gpuData[i*4+3] = make_int4(nd.child[0],nd.child[1],nd.child[2],nd.child[3]);
    }
}

// ═══ DEVICE: decode 4 FP16 values from packed int pair ═══
__device__ __forceinline__ void decode4h(int lo, int hi,
    float&v0, float&v1, float&v2, float&v3) {
    v0 = __half2float(__ushort_as_half((unsigned short)(lo)));
    v1 = __half2float(__ushort_as_half((unsigned short)(lo >> 16)));
    v2 = __half2float(__ushort_as_half((unsigned short)(hi)));
    v3 = __half2float(__ushort_as_half((unsigned short)(hi >> 16)));
}

// ═══ BVH4 TRAVERSAL KERNEL ═══
__global__ void __launch_bounds__(256,3) traceV9(
    const int4*__restrict__ d_bvh4, int n4,
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    const float*__restrict__ rix,const float*__restrict__ riy,const float*__restrict__ riz,
    Hit*__restrict__ hits, int numRays, unsigned long long*__restrict__ stats)
{
    int gid=blockIdx.x*blockDim.x+threadIdx.x, stride=gridDim.x*blockDim.x;
    int cnst = c_bvh4N;
    unsigned long long lnodes=0, ltris=0;

    for(int ri=gid; ri<numRays; ri+=stride) {
        float ox=rox[ri],oy=roy[ri],oz=roz[ri];
        float dx=rdx[ri],dy=rdy[ri],dz=rdz[ri];
        float ix=rix[ri],iy=riy[ri],iz=riz[ri];
        float hitT=1e30f; int hitTri=-1; float hitU=0,hitV=0;

        int stack[MAX_STACK]; int sp=0;
        int nodeIdx = 0; // root

        while(true) {
            if(nodeIdx == -1) { // empty
                if(sp > 0) nodeIdx = stack[--sp]; else break;
                continue;
            }

            if(nodeIdx <= -2) {
                // ═══ LEAF: intersect triangles ═══
                int val = -(nodeIdx + 2);
                int tc = (val & 7) + 1;
                int ts = val >> 3;
                for(int i=0; i<tc; i++) {
                    int ti=ts+i; ltris++;
                    float v0x=tv0x[ti],v0y=tv0y[ti],v0z=tv0z[ti];
                    float e1x=tv1x[ti]-v0x,e1y=tv1y[ti]-v0y,e1z=tv1z[ti]-v0z;
                    float e2x=tv2x[ti]-v0x,e2y=tv2y[ti]-v0y,e2z=tv2z[ti]-v0z;
                    float hx=dy*e2z-dz*e2y,hy=dz*e2x-dx*e2z,hz=dx*e2y-dy*e2x;
                    float a2=e1x*hx+e1y*hy+e1z*hz;
                    if(fabsf(a2)<1e-8f) continue;
                    float f=__frcp_rn(a2);
                    float sx=ox-v0x,sy=oy-v0y,sz=oz-v0z;
                    float u=f*(sx*hx+sy*hy+sz*hz); if(u<0.f||u>1.f) continue;
                    float qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x;
                    float v=f*(dx*qx+dy*qy+dz*qz); if(v<0.f||u+v>1.f) continue;
                    float tt=f*(e2x*qx+e2y*qy+e2z*qz);
                    if(tt>0.001f && tt<hitT) { hitT=tt; hitTri=ti; hitU=u; hitV=v; }
                }
                if(sp > 0) nodeIdx = stack[--sp]; else break;
                continue;
            }

            // ═══ INTERNAL BVH4 NODE: test 4 children ═══
            int4 bx, by, bz, ch;
            if(nodeIdx < cnst) {
                bx = c_bvh4[nodeIdx*4+0];
                by = c_bvh4[nodeIdx*4+1];
                bz = c_bvh4[nodeIdx*4+2];
                ch = c_bvh4[nodeIdx*4+3];
            } else {
                bx = __ldg(&d_bvh4[nodeIdx*4+0]);
                by = __ldg(&d_bvh4[nodeIdx*4+1]);
                bz = __ldg(&d_bvh4[nodeIdx*4+2]);
                ch = __ldg(&d_bvh4[nodeIdx*4+3]);
            }
            lnodes++;

            // Decode FP16 bounds for all 4 children
            float cMinX[4],cMaxX[4],cMinY[4],cMaxY[4],cMinZ[4],cMaxZ[4];
            decode4h(bx.x, bx.y, cMinX[0],cMinX[1],cMinX[2],cMinX[3]);
            decode4h(bx.z, bx.w, cMaxX[0],cMaxX[1],cMaxX[2],cMaxX[3]);
            decode4h(by.x, by.y, cMinY[0],cMinY[1],cMinY[2],cMinY[3]);
            decode4h(by.z, by.w, cMaxY[0],cMaxY[1],cMaxY[2],cMaxY[3]);
            decode4h(bz.x, bz.y, cMinZ[0],cMinZ[1],cMinZ[2],cMinZ[3]);
            decode4h(bz.z, bz.w, cMaxZ[0],cMaxZ[1],cMaxZ[2],cMaxZ[3]);

            int cChild[4] = {ch.x, ch.y, ch.z, ch.w};

            // Slab test all 4 children
            float tMinArr[4]; int hitMask = 0;
            #pragma unroll
            for(int c=0; c<4; c++) {
                if(cChild[c] == -1) continue; // empty slot
                float t1x=(cMinX[c]-ox)*ix, t2x=(cMaxX[c]-ox)*ix;
                float tmn=fminf(t1x,t2x), tmx=fmaxf(t1x,t2x);
                float t1y=(cMinY[c]-oy)*iy, t2y=(cMaxY[c]-oy)*iy;
                tmn=fmaxf(tmn,fminf(t1y,t2y)); tmx=fminf(tmx,fmaxf(t1y,t2y));
                float t1z=(cMinZ[c]-oz)*iz, t2z=(cMaxZ[c]-oz)*iz;
                tmn=fmaxf(tmn,fminf(t1z,t2z)); tmx=fminf(tmx,fmaxf(t1z,t2z));
                if(tmx >= fmaxf(tmn, 0.0f) && tmn < hitT) {
                    tMinArr[c] = tmn;
                    hitMask |= (1 << c);
                }
            }

            int numHits = __popc(hitMask);
            if(numHits == 0) {
                if(sp > 0) nodeIdx = stack[--sp]; else break;
                continue;
            }

            // Compact hits into sorted order (nearest first)
            int hIdx[4]; float hDist[4]; int h=0;
            for(int c=0; c<4; c++) {
                if(hitMask & (1<<c)) { hIdx[h]=cChild[c]; hDist[h]=tMinArr[c]; h++; }
            }

            // Insertion sort (2-5 compares for 1-4 elements)
            for(int i=1; i<numHits; i++) {
                float key=hDist[i]; int ki=hIdx[i]; int j=i-1;
                while(j>=0 && hDist[j]>key) { hDist[j+1]=hDist[j]; hIdx[j+1]=hIdx[j]; j--; }
                hDist[j+1]=key; hIdx[j+1]=ki;
            }

            // Push far children (reverse order), traverse nearest
            for(int i=numHits-1; i>=1; i--) stack[sp++] = hIdx[i];
            nodeIdx = hIdx[0];
            continue;
        }

        hits[ri].t=hitT; hits[ri].tri=hitTri; hits[ri].u=hitU; hits[ri].v=hitV;
    }
    atomicAdd(&stats[0], lnodes);
    atomicAdd(&stats[1], ltris);
}

// ═══ SCENE GENERATORS ═══
void addQuad(Tri*t,int&ti,float3 a,float3 b,float3 c,float3 d){
    t[ti++]={a,b,c};t[ti++]={a,c,d};}
void addBox(Tri*t,int&ti,float3 mn,float3 mx){
    float3 a={mn.x,mn.y,mn.z},b={mx.x,mn.y,mn.z},c_v={mx.x,mx.y,mn.z},d={mn.x,mx.y,mn.z};
    float3 e={mn.x,mn.y,mx.z},f={mx.x,mn.y,mx.z},g={mx.x,mx.y,mx.z},h={mn.x,mx.y,mx.z};
    addQuad(t,ti,a,b,c_v,d);addQuad(t,ti,e,f,g,h);addQuad(t,ti,a,b,f,e);
    addQuad(t,ti,d,c_v,g,h);addQuad(t,ti,a,d,h,e);addQuad(t,ti,b,c_v,g,f);}
void addSubQuad(Tri*t,int&ti,float3 o,float3 ux,float3 uy,int nx,int ny){
    for(int i=0;i<nx;i++)for(int j=0;j<ny;j++){
        float u0=(float)i/nx,u1=(float)(i+1)/nx,v0=(float)j/ny,v1=(float)(j+1)/ny;
        float3 a={o.x+ux.x*u0+uy.x*v0,o.y+ux.y*u0+uy.y*v0,o.z+ux.z*u0+uy.z*v0};
        float3 b={o.x+ux.x*u1+uy.x*v0,o.y+ux.y*u1+uy.y*v0,o.z+ux.z*u1+uy.z*v0};
        float3 c_v={o.x+ux.x*u1+uy.x*v1,o.y+ux.y*u1+uy.y*v1,o.z+ux.z*u1+uy.z*v1};
        float3 d={o.x+ux.x*u0+uy.x*v1,o.y+ux.y*u0+uy.y*v1,o.z+ux.z*u0+uy.z*v1};
        t[ti++]={a,b,c_v};t[ti++]={a,c_v,d};}}
int genConference(Tri*t,int maxTris){
    int ti=0;float W=10,H=5,D=7.5f;int subdiv=(int)sqrtf((float)maxTris/60);
    if(subdiv<2)subdiv=2;if(subdiv>200)subdiv=200;
    addSubQuad(t,ti,{-W,0,-D},{2*W,0,0},{0,0,2*D},subdiv,subdiv);
    addSubQuad(t,ti,{-W,H,-D},{2*W,0,0},{0,0,2*D},subdiv,subdiv);
    addSubQuad(t,ti,{-W,0,-D},{2*W,0,0},{0,H,0},subdiv,subdiv/2);
    addSubQuad(t,ti,{-W,0,D},{2*W,0,0},{0,H,0},subdiv,subdiv/2);
    addSubQuad(t,ti,{-W,0,-D},{0,0,2*D},{0,H,0},subdiv,subdiv/2);
    addSubQuad(t,ti,{W,0,-D},{0,0,2*D},{0,H,0},subdiv,subdiv/2);
    srand(42);
    int numTables=maxTris>50000?20:8;
    for(int i=0;i<numTables&&ti+12<maxTris;i++){
        float tx=((float)rand()/RAND_MAX)*16-8,tz=((float)rand()/RAND_MAX)*12-6;
        addBox(t,ti,{tx-1.0f,0.7f,tz-0.5f},{tx+1.0f,0.8f,tz+0.5f});
        addBox(t,ti,{tx-0.9f,0,tz-0.05f},{tx-0.8f,0.7f,tz+0.05f});
        addBox(t,ti,{tx+0.8f,0,tz-0.05f},{tx+0.9f,0.7f,tz+0.05f});}
    int numChairs=maxTris>50000?40:16;
    for(int i=0;i<numChairs&&ti+12<maxTris;i++){
        float cx=((float)rand()/RAND_MAX)*18-9,cz=((float)rand()/RAND_MAX)*14-7;
        addBox(t,ti,{cx-0.25f,0.4f,cz-0.25f},{cx+0.25f,0.45f,cz+0.25f});
        addBox(t,ti,{cx-0.25f,0.45f,cz-0.25f},{cx+0.25f,0.9f,cz-0.2f});}
    while(ti+2<maxTris){
        float cx=((float)rand()/RAND_MAX)*16-8,cy=0.8f+((float)rand()/RAND_MAX)*0.3f;
        float cz=((float)rand()/RAND_MAX)*12-6,s=0.05f+((float)rand()/RAND_MAX)*0.1f;
        t[ti].v0={cx-s,cy,cz-s};t[ti].v1={cx+s,cy,cz+s};t[ti].v2={cx,cy+s*2,cz};ti++;
        t[ti].v0={cx-s,cy,cz+s};t[ti].v1={cx+s,cy,cz-s};t[ti].v2={cx,cy+s*2,cz};ti++;}
    return ti;
}

struct RayAoS{float3 o,d,id;int origIdx;};
void genPrimaryCoherent(RayAoS*r,int n){
    int w=(int)sqrtf((float)n);
    for(int i=0;i<n;i++){
        int px=i%w,py=i/w;
        float u=(2.0f*px/w-1.0f)*1.2f,v=(2.0f*py/(n/w)-1.0f)*0.6f;
        r[i].o={0,2.5f,12};
        float3 d={u,-v,-1.5f};float l=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);
        d.x/=l;d.y/=l;d.z/=l;r[i].d=d;r[i].id={1.0f/d.x,1.0f/d.y,1.0f/d.z};
        r[i].origIdx=i;
    }
    int tw=4;
    std::sort(r,r+n,[w,tw](const RayAoS&a,const RayAoS&b){
        int ax=a.origIdx%w,ay=a.origIdx/w,bx=b.origIdx%w,by=b.origIdx/w;
        int atx=ax/tw,aty=ay/tw,btx=bx/tw,bty=by/tw;
        if(aty!=bty)return aty<bty;if(atx!=btx)return atx<btx;
        unsigned am=0,bm=0;int alx=ax%tw,aly=ay%tw,blx=bx%tw,bly=by%tw;
        for(int i=0;i<3;i++){am|=((alx>>i)&1)<<(2*i)|((aly>>i)&1)<<(2*i+1);
            bm|=((blx>>i)&1)<<(2*i)|((bly>>i)&1)<<(2*i+1);}
        return am<bm;
    });
}

// ═══ BENCHMARK ═══
void runBench(const char*label, Tri*h_tris, int nt, RayAoS*h_rays, int numRays,
              cudaDeviceProp&prop)
{
    // Build binary BVH
    BN*h_nodes=(BN*)calloc(nt*2,sizeof(BN)); int*tidx=(int*)malloc(nt*4);
    for(int i=0;i<nt;i++) tidx[i]=i; int nc=0;
    buildSAH(h_nodes,h_tris,tidx,nc,0,nt,0);
    BN*h_ord=(BN*)malloc(nc*sizeof(BN)); int*remap=(int*)malloc(nc*4);
    treeReorder(h_nodes,nc,h_ord,remap);

    // Reorder triangles to match BVH leaf order
    Tri*h_to=(Tri*)malloc(nt*sizeof(Tri));
    for(int i=0;i<nt;i++) h_to[i]=h_tris[tidx[i]];

    // Collapse binary → BVH4
    BVH4H*h_bvh4=(BVH4H*)calloc(nc, sizeof(BVH4H)); // worst case: nc BVH4 nodes
    int n4=0;
    collapseRec(h_ord, 0, h_bvh4, &n4);

    // DFS reorder BVH4
    BVH4H*h_bvh4_ord=(BVH4H*)malloc(n4*sizeof(BVH4H));
    int*remap4=(int*)malloc(n4*4);
    reorderBVH4(h_bvh4, n4, h_bvh4_ord, remap4);

    // Pack to GPU format
    int4*h_gpu=(int4*)malloc(n4*4*sizeof(int4));
    packBVH4GPU(h_bvh4_ord, n4, h_gpu);

    // Upload to constant + global memory
    int cn4 = n4 < CONST_BVH4 ? n4 : CONST_BVH4;
    cudaMemcpyToSymbol(c_bvh4, h_gpu, cn4*4*sizeof(int4));
    cudaMemcpyToSymbol(c_bvh4N, &cn4, 4);

    int4*d_bvh4;
    cudaMalloc(&d_bvh4, n4*4*sizeof(int4));
    cudaMemcpy(d_bvh4, h_gpu, n4*4*sizeof(int4), cudaMemcpyHostToDevice);

    // Triangle SoA
    float*h_v[9]; for(int j=0;j<9;j++) h_v[j]=(float*)malloc(nt*4);
    for(int i=0;i<nt;i++){
        h_v[0][i]=h_to[i].v0.x;h_v[1][i]=h_to[i].v0.y;h_v[2][i]=h_to[i].v0.z;
        h_v[3][i]=h_to[i].v1.x;h_v[4][i]=h_to[i].v1.y;h_v[5][i]=h_to[i].v1.z;
        h_v[6][i]=h_to[i].v2.x;h_v[7][i]=h_to[i].v2.y;h_v[8][i]=h_to[i].v2.z;}
    float*d_v[9];
    for(int j=0;j<9;j++){cudaMalloc(&d_v[j],nt*4);cudaMemcpy(d_v[j],h_v[j],nt*4,cudaMemcpyHostToDevice);}

    // Ray SoA
    float*h_ray[9]; for(int j=0;j<9;j++) h_ray[j]=(float*)malloc(numRays*4);
    for(int i=0;i<numRays;i++){
        h_ray[0][i]=h_rays[i].o.x;h_ray[1][i]=h_rays[i].o.y;h_ray[2][i]=h_rays[i].o.z;
        h_ray[3][i]=h_rays[i].d.x;h_ray[4][i]=h_rays[i].d.y;h_ray[5][i]=h_rays[i].d.z;
        h_ray[6][i]=h_rays[i].id.x;h_ray[7][i]=h_rays[i].id.y;h_ray[8][i]=h_rays[i].id.z;}
    float*d_ray[9]; Hit*d_hits; unsigned long long*d_st;
    for(int j=0;j<9;j++){cudaMalloc(&d_ray[j],numRays*4);cudaMemcpy(d_ray[j],h_ray[j],numRays*4,cudaMemcpyHostToDevice);}
    cudaMalloc(&d_hits,numRays*sizeof(Hit)); cudaMalloc(&d_st,16);

    int nb=prop.multiProcessorCount*6; // 3 blocks/SM × occupancy headroom
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    // Warmup
    cudaMemset(d_st,0,16);
    traceV9<<<nb,256>>>(d_bvh4,n4,
        d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
        d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],
        d_hits,numRays,d_st);
    cudaDeviceSynchronize();

    // Benchmark: 10 runs
    float totalMs=0;
    for(int r=0;r<10;r++){
        cudaMemset(d_st,0,16); cudaEventRecord(t0);
        traceV9<<<nb,256>>>(d_bvh4,n4,
            d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
            d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],
            d_hits,numRays,d_st);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1); totalMs+=ms;
    }
    float avgMs=totalMs/10; double mr=(double)numRays/(avgMs/1000.0)/1e6;
    unsigned long long st[2]; cudaMemcpy(st,d_st,16,cudaMemcpyDeviceToHost);

    // Verify hits
    Hit*hh=(Hit*)malloc(numRays*sizeof(Hit));
    cudaMemcpy(hh,d_hits,numRays*sizeof(Hit),cudaMemcpyDeviceToHost);
    int hitCount=0; for(int i=0;i<numRays;i++) if(hh[i].tri>=0) hitCount++;

    float bvhMB=n4*64/(1024.0f*1024.0f);
    float binMB=nc*32/(1024.0f*1024.0f);
    double nodesPerRay=(double)st[0]/numRays;
    double trisPerRay=(double)st[1]/numRays;
    double bytesPerRay=nodesPerRay*64 + trisPerRay*36; // 64B/BVH4 node + 36B/tri

    printf("  │%5dK│%-7s│%7.1f│%5d│%5.1f│ %5.1f│%5.1f%%│%6.0f│%5.2f→%-5.2f│\n",
        nt/1000, label, mr, n4, nodesPerRay, trisPerRay,
        100.0*hitCount/numRays, bytesPerRay, binMB, bvhMB);

    // Cleanup
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_bvh4);
    for(int j=0;j<9;j++){cudaFree(d_v[j]);cudaFree(d_ray[j]);free(h_v[j]);free(h_ray[j]);}
    cudaFree(d_hits); cudaFree(d_st);
    free(h_nodes);free(h_ord);free(remap);free(tidx);free(h_to);
    free(h_bvh4);free(h_bvh4_ord);free(remap4);free(h_gpu);free(hh);
}

int main(){
    printf("╔═════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  V100 RT Engine v9 — BVH4 (4-wide) + FP16 Compressed + Sorted Traversal       ║\n");
    printf("║  Node: 64B (4×child AABB as FP16 + 4×int child) loaded as 4×128-bit            ║\n");
    printf("║  Target: ~20-25 node visits/ray (vs ~45 binary) = fewer serial memory deps      ║\n");
    printf("║  Known v7: 1944/1358 MR/s   Known v8: 3360/3142/2336/1790 MR/s                 ║\n");
    printf("╚═════════════════════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    printf("  GPU: %s | %d SMs | L2: %dKB | BW: ~898 GB/s\n\n",
        prop.name, prop.multiProcessorCount, prop.l2CacheSize/1024);

    int numRays = 4194304;
    int triTargets[] = {50000, 100000, 200000, 500000};

    printf("  ┌─────┬───────┬───────┬─────┬─────┬─────┬─────┬──────┬──────────┐\n");
    printf("  │Tris │Scene  │ MR/s  │ N4  │Nd/R │Tri/R│ Hit%%│ B/ray│BVH MB    │\n");
    printf("  ├─────┼───────┼───────┼─────┼─────┼─────┼─────┼──────┼──────────┤\n");

    for(int s=0; s<4; s++){
        int maxTris=triTargets[s];
        Tri*h_tris=(Tri*)malloc(maxTris*sizeof(Tri));
        int nt=genConference(h_tris,maxTris);
        RayAoS*h_rays=(RayAoS*)malloc(numRays*sizeof(RayAoS));
        genPrimaryCoherent(h_rays,numRays);
        runBench("Primary",h_tris,nt,h_rays,numRays,prop);
        free(h_tris); free(h_rays);
    }

    printf("  └─────┴───────┴───────┴─────┴─────┴─────┴─────┴──────┴──────────┘\n\n");
    printf("  Comparison: v7(binary FP32)=1944 MR/s | v8(binary FP16)=3142 MR/s | v9(BVH4 FP16)=?\n");
    printf("  Target: 20-25 nodes/ray × 64B = 1280-1600 B/ray\n");

    return 0;
}
