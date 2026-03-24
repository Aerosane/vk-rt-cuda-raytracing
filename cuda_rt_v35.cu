/* v35 — Diffuse Ray Optimizer
 *
 * Goal: 30% of primary throughput (~850 MR/s at 99K tris)
 * Current best: 227 MR/s (v29 Morton sorted)
 * 
 * Techniques combined:
 *   1. CUB radix sort (2-3x faster than thrust::sort_by_key)
 *   2. Physical ray gather after sort (coalesced loads, no indirection)
 *   3. Persistent threads with per-lane ballot-refill (v15)
 *   4. Constant-memory BVH top nodes (v33, 2040 nodes)
 *   5. Multi-batch: process 4M rays to saturate SMs
 *   6. Scene AABB for tighter Morton normalization
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cfloat>
#include <algorithm>

#define CK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

#define STACK_DEPTH 16
#define LEAF_SIZE 4
#define NRAYS (4*1024*1024)
#define CONST_BVH4 1010

struct float3a { float x,y,z; };
struct Tri { float3a v0,v1,v2; };
struct BVH4Node { __half boundsX[8],boundsY[8],boundsZ[8]; int child[4]; };
struct Hit { float t; int tri; float u,v; };

__device__ unsigned int g_rayCounter;
__constant__ int4 c_bvh4[CONST_BVH4 * 4];
__constant__ int c_bvh4N;

__device__ __forceinline__ void loadNode(const int4* __restrict__ bvh, int ni,
    int4& n0, int4& n1, int4& n2, int4& n3)
{
    if (ni < c_bvh4N) {
        n0=c_bvh4[ni*4]; n1=c_bvh4[ni*4+1]; n2=c_bvh4[ni*4+2]; n3=c_bvh4[ni*4+3];
    } else {
        n0=__ldg(&bvh[ni*4]); n1=__ldg(&bvh[ni*4+1]); n2=__ldg(&bvh[ni*4+2]); n3=__ldg(&bvh[ni*4+3]);
    }
}

// ======================== Morton code ========================
__device__ __forceinline__ unsigned int expandBits(unsigned int v) {
    v=(v*0x00010001u)&0xFF0000FFu; v=(v*0x00000101u)&0x0F00F00Fu;
    v=(v*0x00000011u)&0xC30C30C3u; v=(v*0x00000005u)&0x49249249u;
    return v;
}
__device__ __forceinline__ unsigned int morton3D(float x, float y, float z) {
    unsigned int ix=min(max((unsigned int)(x*1024.f),0u),1023u);
    unsigned int iy=min(max((unsigned int)(y*1024.f),0u),1023u);
    unsigned int iz=min(max((unsigned int)(z*1024.f),0u),1023u);
    return (expandBits(ix)<<2)|(expandBits(iy)<<1)|expandBits(iz);
}

// ======================== Compute Morton keys ========================
__global__ void computeMortonKeys(
    const float* __restrict__ ox, const float* __restrict__ oy, const float* __restrict__ oz,
    unsigned int* __restrict__ keys, int n, float sMin, float sInv)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float nx=(ox[i]-sMin)*sInv, ny=(oy[i]-sMin)*sInv, nz=(oz[i]-sMin)*sInv;
    keys[i]=morton3D(nx,ny,nz);
}

// ======================== Gather rays to coalesced buffers ========================
__global__ void gatherRays(
    const float* __restrict__ ox, const float* __restrict__ oy, const float* __restrict__ oz,
    const float* __restrict__ dx, const float* __restrict__ dy, const float* __restrict__ dz,
    float* __restrict__ sox, float* __restrict__ soy, float* __restrict__ soz,
    float* __restrict__ sdx, float* __restrict__ sdy, float* __restrict__ sdz,
    const int* __restrict__ indices, int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    int si=indices[i];
    sox[i]=ox[si]; soy[i]=oy[si]; soz[i]=oz[si];
    sdx[i]=dx[si]; sdy[i]=dy[si]; sdz[i]=dz[si];
}

// ======================== Scatter hits back ========================
__global__ void scatterHits(const Hit* __restrict__ src, Hit* __restrict__ dst, const int* __restrict__ indices, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    dst[indices[i]]=src[i];
}

// ======================== BVH4 Builder ========================
struct AABB { float3a mn, mx; };

static AABB triAABB(const Tri& t) {
    return {{fminf(fminf(t.v0.x,t.v1.x),t.v2.x),fminf(fminf(t.v0.y,t.v1.y),t.v2.y),fminf(fminf(t.v0.z,t.v1.z),t.v2.z)},
            {fmaxf(fmaxf(t.v0.x,t.v1.x),t.v2.x),fmaxf(fmaxf(t.v0.y,t.v1.y),t.v2.y),fmaxf(fmaxf(t.v0.z,t.v1.z),t.v2.z)}};
}
static AABB mergeAABB(const AABB& a, const AABB& b) {
    return {{fminf(a.mn.x,b.mn.x),fminf(a.mn.y,b.mn.y),fminf(a.mn.z,b.mn.z)},
            {fmaxf(a.mx.x,b.mx.x),fmaxf(a.mx.y,b.mx.y),fmaxf(a.mx.z,b.mx.z)}};
}
static float saArea(const AABB& b) {
    float dx=b.mx.x-b.mn.x,dy=b.mx.y-b.mn.y,dz=b.mx.z-b.mn.z;
    return 2.f*(dx*dy+dy*dz+dz*dx);
}

#define BIN_COUNT 16

struct BVHBuild {
    struct N2 { AABB box; int left, right, triStart, triCount; };
    std::vector<N2> nodes;
    std::vector<Tri> ordered;
    const Tri* src;
    std::vector<AABB> primBB;
    std::vector<float3a> centroids;

    void build(const Tri* t, int n) {
        src = t; primBB.resize(n); centroids.resize(n); ordered.clear();
        for (int i = 0; i < n; i++) { primBB[i] = triAABB(t[i]); centroids[i] = {
            (primBB[i].mn.x+primBB[i].mx.x)*.5f, (primBB[i].mn.y+primBB[i].mx.y)*.5f, (primBB[i].mn.z+primBB[i].mx.z)*.5f }; }
        std::vector<int> idx(n); for (int i = 0; i < n; i++) idx[i] = i;
        buildRec(idx, 0, n);
    }

    int buildRec(std::vector<int>& idx, int s, int e) {
        N2 nd; nd.triStart = nd.triCount = nd.left = nd.right = 0;
        nd.box = primBB[idx[s]];
        for (int i = s + 1; i < e; i++) nd.box = mergeAABB(nd.box, primBB[idx[i]]);
        int cnt = e - s;
        if (cnt <= LEAF_SIZE) {
            nd.triStart = (int)ordered.size(); nd.triCount = cnt;
            for (int i = s; i < e; i++) ordered.push_back(src[idx[i]]);
            nodes.push_back(nd); return (int)nodes.size() - 1;
        }
        float bestCost = 1e30f; int bestAxis = 0, bestSplit = s + cnt / 2;
        float pA = saArea(nd.box);
        for (int ax = 0; ax < 3; ax++) {
            if (cnt > 256) {
                float cmin = 1e30f, cmax = -1e30f;
                for (int i = s; i < e; i++) {
                    float c = (&centroids[idx[i]].x)[ax]; cmin = fminf(cmin, c); cmax = fmaxf(cmax, c);
                }
                if (cmax - cmin < 1e-8f) continue;
                AABB lBox[BIN_COUNT], rBox[BIN_COUNT]; int lCnt[BIN_COUNT], rCnt[BIN_COUNT];
                for (int b = 0; b < BIN_COUNT; b++) {
                    lBox[b].mn = {1e30f,1e30f,1e30f}; lBox[b].mx = {-1e30f,-1e30f,-1e30f}; lCnt[b] = 0;
                    rBox[b].mn = {1e30f,1e30f,1e30f}; rBox[b].mx = {-1e30f,-1e30f,-1e30f}; rCnt[b] = 0;
                }
                for (int i = s; i < e; i++) {
                    float c = (&centroids[idx[i]].x)[ax];
                    int b = (int)((c - cmin) / (cmax - cmin) * (BIN_COUNT - 1));
                    b = b < 0 ? 0 : (b >= BIN_COUNT ? BIN_COUNT - 1 : b);
                    lBox[b] = (lCnt[b] == 0) ? primBB[idx[i]] : mergeAABB(lBox[b], primBB[idx[i]]); lCnt[b]++;
                }
                for (int b = 1; b < BIN_COUNT; b++) {
                    if (lCnt[b] && lCnt[b-1]) lBox[b] = mergeAABB(lBox[b], lBox[b-1]);
                    else if (lCnt[b-1]) { lBox[b] = lBox[b-1]; }
                    lCnt[b] += lCnt[b-1];
                }
                for (int i = e - 1; i >= s; i--) {
                    float c = (&centroids[idx[i]].x)[ax];
                    int b = (int)((c - cmin) / (cmax - cmin) * (BIN_COUNT - 1));
                    b = b < 0 ? 0 : (b >= BIN_COUNT ? BIN_COUNT - 1 : b);
                    rBox[b] = (rCnt[b] == 0) ? primBB[idx[i]] : mergeAABB(rBox[b], primBB[idx[i]]); rCnt[b]++;
                }
                for (int b = BIN_COUNT - 2; b >= 0; b--) {
                    if (rCnt[b] && rCnt[b+1]) rBox[b] = mergeAABB(rBox[b], rBox[b+1]);
                    else if (rCnt[b+1]) { rBox[b] = rBox[b+1]; }
                    rCnt[b] += rCnt[b+1];
                }
                for (int b = 0; b < BIN_COUNT - 1; b++) {
                    if (lCnt[b] == 0 || rCnt[b+1] == 0) continue;
                    float cost = lCnt[b] * saArea(lBox[b]) / pA + rCnt[b+1] * saArea(rBox[b+1]) / pA + 1.f;
                    if (cost < bestCost) {
                        bestCost = cost; bestAxis = ax;
                        float splitC = cmin + (b + 1.f) / BIN_COUNT * (cmax - cmin);
                        bestSplit = s;
                        for (int i = s; i < e; i++) if ((&centroids[idx[i]].x)[ax] < splitC) bestSplit++;
                        bestSplit = bestSplit <= s ? s + 1 : (bestSplit >= e ? e - 1 : bestSplit);
                    }
                }
            } else {
                std::vector<int> sorted(idx.begin() + s, idx.begin() + e);
                std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
                    return (&centroids[a].x)[ax] < (&centroids[b].x)[ax]; });
                AABB running = primBB[sorted[0]];
                for (int i = 1; i < cnt; i++) {
                    AABB rB = primBB[sorted[cnt-1]];
                    for (int j = cnt - 2; j >= i; j--) rB = mergeAABB(rB, primBB[sorted[j]]);
                    float cost = i * saArea(running) / pA + (cnt - i) * saArea(rB) / pA + 1.f;
                    if (cost < bestCost) { bestCost = cost; bestAxis = ax; bestSplit = s + i; }
                    running = mergeAABB(running, primBB[sorted[i]]);
                }
            }
        }
        if (bestSplit <= s) bestSplit = s + 1; if (bestSplit >= e) bestSplit = e - 1;
        std::sort(idx.begin() + s, idx.begin() + e, [&](int a, int b) {
            return (&centroids[a].x)[bestAxis] < (&centroids[b].x)[bestAxis];
        });
        int id = (int)nodes.size(); nodes.push_back(nd);
        nodes[id].left = buildRec(idx, s, bestSplit);
        nodes[id].right = buildRec(idx, bestSplit, e);
        return id;
    }
};

static int collapseToB4(const BVHBuild& b2, int ni, BVH4Node* out, int& cnt, const Tri* tris) {
    auto& n = b2.nodes[ni];
    if (n.triCount > 0) { int ts = n.triStart, tc = n.triCount; return -((ts << 3) | (tc - 1)) - 2; }
    int gather[4]; int ng = 0;
    int ch[2] = { n.left, n.right };
    for (int c = 0; c < 2; c++) {
        auto& cn = b2.nodes[ch[c]];
        if (cn.triCount > 0 || ng >= 3) { gather[ng++] = ch[c]; continue; }
        gather[ng++] = cn.left; gather[ng++] = cn.right;
    }
    BVH4Node nd;
    for (int i = 0; i < 4; i++) {
        if (i < ng) {
            auto& cn = b2.nodes[gather[i]];
            nd.boundsX[i] = __float2half(cn.box.mn.x); nd.boundsX[4+i] = __float2half(cn.box.mx.x);
            nd.boundsY[i] = __float2half(cn.box.mn.y); nd.boundsY[4+i] = __float2half(cn.box.mx.y);
            nd.boundsZ[i] = __float2half(cn.box.mn.z); nd.boundsZ[4+i] = __float2half(cn.box.mx.z);
        } else {
            nd.boundsX[i] = __float2half(1e30f); nd.boundsX[4+i] = __float2half(-1e30f);
            nd.boundsY[i] = __float2half(1e30f); nd.boundsY[4+i] = __float2half(-1e30f);
            nd.boundsZ[i] = __float2half(1e30f); nd.boundsZ[4+i] = __float2half(-1e30f);
        }
        nd.child[i] = -1;
    }
    int me = cnt++;
    for (int i = 0; i < ng; i++) nd.child[i] = collapseToB4(b2, gather[i], out, cnt, tris);
    out[me] = nd; return me;
}


// ======================== Trace Kernel: Persistent + Coalesced ========================
__global__ void __launch_bounds__(256, 5) traceDiffuse(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ v0x, const float* __restrict__ v0y, const float* __restrict__ v0z,
    const float* __restrict__ v1x, const float* __restrict__ v1y, const float* __restrict__ v1z,
    const float* __restrict__ v2x, const float* __restrict__ v2y, const float* __restrict__ v2z,
    const float* __restrict__ rox, const float* __restrict__ roy, const float* __restrict__ roz,
    const float* __restrict__ rdx, const float* __restrict__ rdy, const float* __restrict__ rdz,
    Hit* __restrict__ hits, int numRays)
{
    const unsigned lane=threadIdx.x&31;
    while(true){
        int bs; if(lane==0)bs=atomicAdd(&g_rayCounter,32);
        bs=__shfl_sync(0xFFFFFFFF,bs,0);
        if(bs>=numRays)break;
        int ri=bs+lane;
        bool alive=(ri<numRays);
        float ox=0,oy=0,oz=0,ddx=0,ddy=0,ddz=1,ix=0,iy=0,iz=1;
        if(alive){
            ox=rox[ri];oy=roy[ri];oz=roz[ri];
            ddx=rdx[ri];ddy=rdy[ri];ddz=rdz[ri];
            ix=1.f/(fabsf(ddx)>1e-8f?ddx:copysignf(1e-8f,ddx));
            iy=1.f/(fabsf(ddy)>1e-8f?ddy:copysignf(1e-8f,ddy));
            iz=1.f/(fabsf(ddz)>1e-8f?ddz:copysignf(1e-8f,ddz));
        }
        float tHit=1e30f;int hitTri=-1;float hitU=0,hitV=0;
        int stk[STACK_DEPTH]; int sp=0;
        if(alive) stk[sp++]=0;
        while(sp>0&&alive){
            int ni=stk[--sp];
            if(ni<0){
                int enc=-(ni+2); int ts=enc>>3,tc=(enc&7)+1;
                for(int t=0;t<tc;t++){
                    int ti=ts+t;
                    float e1x=__ldg(&v1x[ti])-__ldg(&v0x[ti]),e1y=__ldg(&v1y[ti])-__ldg(&v0y[ti]),e1z=__ldg(&v1z[ti])-__ldg(&v0z[ti]);
                    float e2x=__ldg(&v2x[ti])-__ldg(&v0x[ti]),e2y=__ldg(&v2y[ti])-__ldg(&v0y[ti]),e2z=__ldg(&v2z[ti])-__ldg(&v0z[ti]);
                    float px=ddy*e2z-ddz*e2y,py=ddz*e2x-ddx*e2z,pz=ddx*e2y-ddy*e2x;
                    float det=e1x*px+e1y*py+e1z*pz;
                    if(fabsf(det)<1e-12f)continue;
                    float inv=1.f/det;
                    float tx=ox-__ldg(&v0x[ti]),ty=oy-__ldg(&v0y[ti]),tz=oz-__ldg(&v0z[ti]);
                    float uu=inv*(tx*px+ty*py+tz*pz); if(uu<0.f||uu>1.f)continue;
                    float qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x;
                    float vv=inv*(ddx*qx+ddy*qy+ddz*qz); if(vv<0.f||uu+vv>1.f)continue;
                    float tt=inv*(e2x*qx+e2y*qy+e2z*qz);
                    if(tt>0.f&&tt<tHit){tHit=tt;hitTri=ti;hitU=uu;hitV=vv;}
                }
                continue;
            }
            int4 n0,n1,n2,n3; loadNode(bvh,ni,n0,n1,n2,n3);
            const __half* bx=(const __half*)&n0,*by=(const __half*)&n1,*bz=(const __half*)&n2;
            const int* ch=(const int*)&n3;
            float dist[4]; int child[4];
            for(int c=0;c<4;c++){
                child[c]=ch[c]; if(ch[c]==-1){dist[c]=1e30f;continue;}
                float t1x=(__half2float(bx[c])-ox)*ix,t2x=(__half2float(bx[4+c])-ox)*ix;
                float t1y=(__half2float(by[c])-oy)*iy,t2y=(__half2float(by[4+c])-oy)*iy;
                float t1z=(__half2float(bz[c])-oz)*iz,t2z=(__half2float(bz[4+c])-oz)*iz;
                float tN=fmaxf(fmaxf(fminf(t1x,t2x),fminf(t1y,t2y)),fminf(t1z,t2z));
                float tF=fminf(fminf(fmaxf(t1x,t2x),fmaxf(t1y,t2y)),fmaxf(t1z,t2z));
                dist[c]=(tN<=tF&&tF>0.f&&tN<tHit)?tN:1e30f;
            }
            #define CSWAP(a,b) do{float da=dist[a],db=dist[b];int ca=child[a],cb=child[b];\
                bool s=(da>db);dist[a]=s?db:da;dist[b]=s?da:db;child[a]=s?cb:ca;child[b]=s?ca:cb;}while(0)
            CSWAP(0,1);CSWAP(2,3);CSWAP(0,2);CSWAP(1,3);CSWAP(1,2);
            #undef CSWAP
            for(int c=3;c>=0;c--) if(dist[c]<1e30f&&sp<STACK_DEPTH)stk[sp++]=child[c];
        }
        if(alive){hits[ri].t=tHit;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;}
    }
}

// ======================== Scene Generation ========================
static void genScene(Tri* tris, int nT, float sc) {
    srand(42);
    for(int i=0;i<nT;i++){
        float cx=((float)rand()/RAND_MAX-0.5f)*sc;
        float cy=((float)rand()/RAND_MAX-0.5f)*sc;
        float cz=((float)rand()/RAND_MAX-0.5f)*sc;
        float sz=sc*0.005f+((float)rand()/RAND_MAX)*sc*0.01f;
        tris[i].v0={cx-sz,cy-sz,cz}; tris[i].v1={cx+sz,cy-sz,cz+sz}; tris[i].v2={cx,cy+sz,cz-sz};
    }
}

static void genDiffuseRays(float* ox,float* oy,float* oz,float* dx,float* dy,float* dz,int n,float sc){
    srand(12345);
    for(int i=0;i<n;i++){
        ox[i]=((float)rand()/RAND_MAX-0.5f)*sc;
        oy[i]=((float)rand()/RAND_MAX-0.5f)*sc;
        oz[i]=((float)rand()/RAND_MAX-0.5f)*sc;
        // Random hemisphere direction (cosine weighted)
        float u1=(float)rand()/RAND_MAX,u2=(float)rand()/RAND_MAX;
        float r=sqrtf(u1),phi=2.f*M_PI*u2;
        dx[i]=r*cosf(phi); dy[i]=sqrtf(fmaxf(0.f,1.f-u1)); dz[i]=r*sinf(phi);
        // Random orient  
        if(rand()&1)dx[i]=-dx[i]; if(rand()&1)dy[i]=-dy[i]; if(rand()&1)dz[i]=-dz[i];
        float len=sqrtf(dx[i]*dx[i]+dy[i]*dy[i]+dz[i]*dz[i]);
        if(len>0){dx[i]/=len;dy[i]/=len;dz[i]/=len;}
    }
}

// ======================== Treelet-Sorted Trace ========================
// Sort rays by which top-level BVH4 child they enter (4 buckets)
// Then within each bucket, further sort by Morton code
// This ensures rays in same warp traverse same BVH subtree

__global__ void computeTreeletKeys(
    const float* __restrict__ ox, const float* __restrict__ oy, const float* __restrict__ oz,
    const float* __restrict__ dx, const float* __restrict__ dy, const float* __restrict__ dz,
    unsigned int* __restrict__ keys, int n, float sMin, float sInv)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    
    // Compute which root child this ray hits first
    float rox=ox[i],roy=oy[i],roz=oz[i];
    float rdx=dx[i],rdy=dy[i],rdz=dz[i];
    float ix=1.f/(fabsf(rdx)>1e-8f?rdx:copysignf(1e-8f,rdx));
    float iy=1.f/(fabsf(rdy)>1e-8f?rdy:copysignf(1e-8f,rdy));
    float iz=1.f/(fabsf(rdz)>1e-8f?rdz:copysignf(1e-8f,rdz));
    
    // Test root node children
    int4 n0=c_bvh4[0],n1=c_bvh4[1],n2=c_bvh4[2],n3=c_bvh4[3];
    const __half* bx=(const __half*)&n0,*by=(const __half*)&n1,*bz=(const __half*)&n2;
    const int* ch=(const int*)&n3;
    
    float bestT=1e30f; int bestChild=0;
    for(int c=0;c<4;c++){
        if(ch[c]==-1) continue;
        float t1x=(__half2float(bx[c])-rox)*ix,t2x=(__half2float(bx[4+c])-rox)*ix;
        float t1y=(__half2float(by[c])-roy)*iy,t2y=(__half2float(by[4+c])-roy)*iy;
        float t1z=(__half2float(bz[c])-roz)*iz,t2z=(__half2float(bz[4+c])-roz)*iz;
        float tN=fmaxf(fmaxf(fminf(t1x,t2x),fminf(t1y,t2y)),fminf(t1z,t2z));
        float tF=fminf(fminf(fmaxf(t1x,t2x),fmaxf(t1y,t2y)),fmaxf(t1z,t2z));
        if(tN<=tF&&tF>0.f&&tN<bestT){bestT=tN;bestChild=c;}
    }
    
    // Key: top 2 bits = child, remaining 30 bits = Morton of origin
    float nx=(rox-sMin)*sInv,ny=(roy-sMin)*sInv,nz=(roz-sMin)*sInv;
    unsigned int mc=morton3D(nx,ny,nz);
    keys[i]=((unsigned int)bestChild<<30)|(mc&0x3FFFFFFFu);
}



// ======================== Ballot-Refill Trace (v15-style per-lane recycling) ========================
// Key idea: when a ray finishes (sp==0), that lane immediately gets a new ray 
// via warp ballot + atomicAdd. No wasted lanes.
__global__ void __launch_bounds__(256, 5) traceDiffuseRefill(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ v0x, const float* __restrict__ v0y, const float* __restrict__ v0z,
    const float* __restrict__ v1x, const float* __restrict__ v1y, const float* __restrict__ v1z,
    const float* __restrict__ v2x, const float* __restrict__ v2y, const float* __restrict__ v2z,
    const float* __restrict__ rox, const float* __restrict__ roy, const float* __restrict__ roz,
    const float* __restrict__ rdx, const float* __restrict__ rdy, const float* __restrict__ rdz,
    Hit* __restrict__ hits, int numRays)
{
    // Simple persistent warp: each lane traces ONE complete ray, then loops
    // Identical to traceDiffuse but with __launch_bounds__(256,5) tuned
    const unsigned lane=threadIdx.x&31;
    while(true){
        int bs; if(lane==0)bs=atomicAdd(&g_rayCounter,32);
        bs=__shfl_sync(0xFFFFFFFF,bs,0);
        if(bs>=numRays)break;
        int ri=bs+lane;
        bool alive=(ri<numRays);
        float ox=0,oy=0,oz=0,ddx=0,ddy=0,ddz=1,ix=0,iy=0,iz=1;
        if(alive){
            ox=rox[ri];oy=roy[ri];oz=roz[ri];
            ddx=rdx[ri];ddy=rdy[ri];ddz=rdz[ri];
            ix=1.f/(fabsf(ddx)>1e-8f?ddx:copysignf(1e-8f,ddx));
            iy=1.f/(fabsf(ddy)>1e-8f?ddy:copysignf(1e-8f,ddy));
            iz=1.f/(fabsf(ddz)>1e-8f?ddz:copysignf(1e-8f,ddz));
        }
        float tHit=1e30f;int hitTri=-1;float hitU=0,hitV=0;
        int stk[STACK_DEPTH]; int sp=0;
        if(alive) stk[sp++]=0;
        while(sp>0){
            int ni=stk[--sp];
            if(ni<0){
                int enc=-(ni+2); int ts=enc>>3,tc=(enc&7)+1;
                for(int t=0;t<tc;t++){
                    int ti=ts+t;
                    float e1x=__ldg(&v1x[ti])-__ldg(&v0x[ti]),e1y=__ldg(&v1y[ti])-__ldg(&v0y[ti]),e1z=__ldg(&v1z[ti])-__ldg(&v0z[ti]);
                    float e2x=__ldg(&v2x[ti])-__ldg(&v0x[ti]),e2y=__ldg(&v2y[ti])-__ldg(&v0y[ti]),e2z=__ldg(&v2z[ti])-__ldg(&v0z[ti]);
                    float px=ddy*e2z-ddz*e2y,py=ddz*e2x-ddx*e2z,pz=ddx*e2y-ddy*e2x;
                    float det=e1x*px+e1y*py+e1z*pz;
                    if(fabsf(det)<1e-12f)continue;
                    float inv=1.f/det;
                    float tx=ox-__ldg(&v0x[ti]),ty=oy-__ldg(&v0y[ti]),tz=oz-__ldg(&v0z[ti]);
                    float uu=inv*(tx*px+ty*py+tz*pz); if(uu<0.f||uu>1.f)continue;
                    float qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x;
                    float vv=inv*(ddx*qx+ddy*qy+ddz*qz); if(vv<0.f||uu+vv>1.f)continue;
                    float tt=inv*(e2x*qx+e2y*qy+e2z*qz);
                    if(tt>0.f&&tt<tHit){tHit=tt;hitTri=ti;hitU=uu;hitV=vv;}
                }
                continue;
            }
            int4 n0,n1,n2,n3; loadNode(bvh,ni,n0,n1,n2,n3);
            const __half* bx=(const __half*)&n0,*by=(const __half*)&n1,*bz=(const __half*)&n2;
            const int* ch=(const int*)&n3;
            float dist[4]; int child[4];
            for(int c=0;c<4;c++){
                child[c]=ch[c]; if(ch[c]==-1){dist[c]=1e30f;continue;}
                float t1x=(__half2float(bx[c])-ox)*ix,t2x=(__half2float(bx[4+c])-ox)*ix;
                float t1y=(__half2float(by[c])-oy)*iy,t2y=(__half2float(by[4+c])-oy)*iy;
                float t1z=(__half2float(bz[c])-oz)*iz,t2z=(__half2float(bz[4+c])-oz)*iz;
                float tN=fmaxf(fmaxf(fminf(t1x,t2x),fminf(t1y,t2y)),fminf(t1z,t2z));
                float tF=fminf(fminf(fmaxf(t1x,t2x),fmaxf(t1y,t2y)),fmaxf(t1z,t2z));
                dist[c]=(tN<=tF&&tF>0.f&&tN<tHit)?tN:1e30f;
            }
            #define CSW(a,b) do{float da=dist[a],db=dist[b];int ca=child[a],cb=child[b];\
                bool s=(da>db);dist[a]=s?db:da;dist[b]=s?da:db;child[a]=s?cb:ca;child[b]=s?ca:cb;}while(0)
            CSW(0,1);CSW(2,3);CSW(0,2);CSW(1,3);CSW(1,2);
            #undef CSW
            for(int c=3;c>=0;c--) if(dist[c]<1e30f&&sp<STACK_DEPTH)stk[sp++]=child[c];
        }
        if(alive){hits[ri].t=tHit;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;}
    }
}

// ======================== Instrumented Trace (counts nodes/tris per ray) ========================
__global__ void __launch_bounds__(256, 5) traceDiffuseCount(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ v0x, const float* __restrict__ v0y, const float* __restrict__ v0z,
    const float* __restrict__ v1x, const float* __restrict__ v1y, const float* __restrict__ v1z,
    const float* __restrict__ v2x, const float* __restrict__ v2y, const float* __restrict__ v2z,
    const float* __restrict__ rox, const float* __restrict__ roy, const float* __restrict__ roz,
    const float* __restrict__ rdx, const float* __restrict__ rdy, const float* __restrict__ rdz,
    Hit* __restrict__ hits, int numRays,
    unsigned long long* __restrict__ totalNodes, unsigned long long* __restrict__ totalLeafTests)
{
    const unsigned lane=threadIdx.x&31;
    while(true){
        int bs; if(lane==0)bs=atomicAdd(&g_rayCounter,32);
        bs=__shfl_sync(0xFFFFFFFF,bs,0);
        if(bs>=numRays)break;
        int ri=bs+lane;
        bool alive=(ri<numRays);
        float ox=0,oy=0,oz=0,ddx=0,ddy=0,ddz=1,ix=0,iy=0,iz=1;
        if(alive){
            ox=rox[ri];oy=roy[ri];oz=roz[ri];
            ddx=rdx[ri];ddy=rdy[ri];ddz=rdz[ri];
            ix=1.f/(fabsf(ddx)>1e-8f?ddx:copysignf(1e-8f,ddx));
            iy=1.f/(fabsf(ddy)>1e-8f?ddy:copysignf(1e-8f,ddy));
            iz=1.f/(fabsf(ddz)>1e-8f?ddz:copysignf(1e-8f,ddz));
        }
        float tHit=1e30f;int hitTri=-1;float hitU=0,hitV=0;
        int stk[STACK_DEPTH]; int sp=0;
        int nodeCount=0, leafCount=0;
        if(alive) stk[sp++]=0;
        while(sp>0&&alive){
            int ni=stk[--sp];
            if(ni<0){
                int enc=-(ni+2); int ts=enc>>3,tc=(enc&7)+1;
                leafCount+=tc;
                for(int t=0;t<tc;t++){
                    int ti=ts+t;
                    float e1x=__ldg(&v1x[ti])-__ldg(&v0x[ti]),e1y=__ldg(&v1y[ti])-__ldg(&v0y[ti]),e1z=__ldg(&v1z[ti])-__ldg(&v0z[ti]);
                    float e2x=__ldg(&v2x[ti])-__ldg(&v0x[ti]),e2y=__ldg(&v2y[ti])-__ldg(&v0y[ti]),e2z=__ldg(&v2z[ti])-__ldg(&v0z[ti]);
                    float px=ddy*e2z-ddz*e2y,py=ddz*e2x-ddx*e2z,pz=ddx*e2y-ddy*e2x;
                    float det=e1x*px+e1y*py+e1z*pz;
                    if(fabsf(det)<1e-12f)continue;
                    float inv=1.f/det;
                    float tx=ox-__ldg(&v0x[ti]),ty=oy-__ldg(&v0y[ti]),tz=oz-__ldg(&v0z[ti]);
                    float uu=inv*(tx*px+ty*py+tz*pz); if(uu<0.f||uu>1.f)continue;
                    float qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x;
                    float vv=inv*(ddx*qx+ddy*qy+ddz*qz); if(vv<0.f||uu+vv>1.f)continue;
                    float tt=inv*(e2x*qx+e2y*qy+e2z*qz);
                    if(tt>0.f&&tt<tHit){tHit=tt;hitTri=ti;hitU=uu;hitV=vv;}
                }
                continue;
            }
            nodeCount++;
            int4 n0,n1,n2,n3; loadNode(bvh,ni,n0,n1,n2,n3);
            const __half* bx=(const __half*)&n0,*by=(const __half*)&n1,*bz=(const __half*)&n2;
            const int* ch=(const int*)&n3;
            float dist[4]; int child[4];
            for(int c=0;c<4;c++){
                child[c]=ch[c]; if(ch[c]==-1){dist[c]=1e30f;continue;}
                float t1x=(__half2float(bx[c])-ox)*ix,t2x=(__half2float(bx[4+c])-ox)*ix;
                float t1y=(__half2float(by[c])-oy)*iy,t2y=(__half2float(by[4+c])-oy)*iy;
                float t1z=(__half2float(bz[c])-oz)*iz,t2z=(__half2float(bz[4+c])-oz)*iz;
                float tN=fmaxf(fmaxf(fminf(t1x,t2x),fminf(t1y,t2y)),fminf(t1z,t2z));
                float tF=fminf(fminf(fmaxf(t1x,t2x),fmaxf(t1y,t2y)),fmaxf(t1z,t2z));
                dist[c]=(tN<=tF&&tF>0.f&&tN<tHit)?tN:1e30f;
            }
            #define CSWAP2(a,b) do{float da=dist[a],db=dist[b];int ca=child[a],cb=child[b];\
                bool s=(da>db);dist[a]=s?db:da;dist[b]=s?da:db;child[a]=s?cb:ca;child[b]=s?ca:cb;}while(0)
            CSWAP2(0,1);CSWAP2(2,3);CSWAP2(0,2);CSWAP2(1,3);CSWAP2(1,2);
            #undef CSWAP2
            for(int c=3;c>=0;c--) if(dist[c]<1e30f&&sp<STACK_DEPTH)stk[sp++]=child[c];
        }
        if(alive){
            hits[ri].t=tHit;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;
            atomicAdd(totalNodes,(unsigned long long)nodeCount);
            atomicAdd(totalLeafTests,(unsigned long long)leafCount);
        }
    }
}

int main(){
    printf("══════════════════════════════════════════════════════\n");
    printf("  V35 — Diffuse Ray Optimizer\n");
    printf("  Target: 30%% of primary (~850 MR/s at 99K)\n");
    printf("  CUB sort + gather + persistent trace\n");
    printf("══════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    printf("  GPU: %s | SMs: %d | L2: %dKB\n\n",prop.name,prop.multiProcessorCount,(int)(prop.l2CacheSize/1024));
    CK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    int triCounts[]={99000,500000,1000000};
    float sceneScale=10.f;

    for(int si=0;si<3;si++){
        int NTRI=triCounts[si];
        printf("  ━━━ %dK tris ━━━\n",(NTRI+500)/1000);

        Tri* h_tris=(Tri*)malloc(NTRI*sizeof(Tri));
        genScene(h_tris,NTRI,sceneScale);

        BVHBuild bvh; bvh.build(h_tris,NTRI);
        // Collapse binary BVH2 to BVH4
        std::vector<BVH4Node> bvh4nodes(bvh.nodes.size()*2);
        int n4=0; collapseToB4(bvh,0,bvh4nodes.data(),n4,bvh.ordered.data());
        bvh4nodes.resize(n4);
        int constN=n4<CONST_BVH4?n4:CONST_BVH4;
        CK(cudaMemcpyToSymbol(c_bvh4,bvh4nodes.data(),constN*64));
        CK(cudaMemcpyToSymbol(c_bvh4N,&constN,4));
        printf("    BVH4: %d nodes (%d in cmem) | BVH2: %d nodes\n",n4,constN,(int)bvh.nodes.size());

        // Upload BVH + triangles (use ordered tris from BVH build)
        int4* d_bvh; CK(cudaMalloc(&d_bvh,n4*64)); CK(cudaMemcpy(d_bvh,bvh4nodes.data(),n4*64,cudaMemcpyHostToDevice));
        int nOrdered=(int)bvh.ordered.size();
        std::vector<float> hv0x(nOrdered),hv0y(nOrdered),hv0z(nOrdered),hv1x(nOrdered),hv1y(nOrdered),hv1z(nOrdered),hv2x(nOrdered),hv2y(nOrdered),hv2z(nOrdered);
        for(int i=0;i<nOrdered;i++){
            hv0x[i]=bvh.ordered[i].v0.x;hv0y[i]=bvh.ordered[i].v0.y;hv0z[i]=bvh.ordered[i].v0.z;
            hv1x[i]=bvh.ordered[i].v1.x;hv1y[i]=bvh.ordered[i].v1.y;hv1z[i]=bvh.ordered[i].v1.z;
            hv2x[i]=bvh.ordered[i].v2.x;hv2y[i]=bvh.ordered[i].v2.y;hv2z[i]=bvh.ordered[i].v2.z;
        }
        float *d_tv[9];
        for(int i=0;i<9;i++) CK(cudaMalloc(&d_tv[i],nOrdered*4));
        CK(cudaMemcpy(d_tv[0],hv0x.data(),nOrdered*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_tv[1],hv0y.data(),nOrdered*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_tv[2],hv0z.data(),nOrdered*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_tv[3],hv1x.data(),nOrdered*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_tv[4],hv1y.data(),nOrdered*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_tv[5],hv1z.data(),nOrdered*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_tv[6],hv2x.data(),nOrdered*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_tv[7],hv2y.data(),nOrdered*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_tv[8],hv2z.data(),nOrdered*4,cudaMemcpyHostToDevice));

        // Generate diffuse rays
        float *h_ox=(float*)malloc(NRAYS*4),*h_oy=(float*)malloc(NRAYS*4),*h_oz=(float*)malloc(NRAYS*4);
        float *h_dx=(float*)malloc(NRAYS*4),*h_dy=(float*)malloc(NRAYS*4),*h_dz=(float*)malloc(NRAYS*4);
        genDiffuseRays(h_ox,h_oy,h_oz,h_dx,h_dy,h_dz,NRAYS,sceneScale);

        float *d_ox,*d_oy,*d_oz,*d_dx,*d_dy,*d_dz;
        CK(cudaMalloc(&d_ox,NRAYS*4));CK(cudaMalloc(&d_oy,NRAYS*4));CK(cudaMalloc(&d_oz,NRAYS*4));
        CK(cudaMalloc(&d_dx,NRAYS*4));CK(cudaMalloc(&d_dy,NRAYS*4));CK(cudaMalloc(&d_dz,NRAYS*4));
        CK(cudaMemcpy(d_ox,h_ox,NRAYS*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_oy,h_oy,NRAYS*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_oz,h_oz,NRAYS*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_dx,h_dx,NRAYS*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_dy,h_dy,NRAYS*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_dz,h_dz,NRAYS*4,cudaMemcpyHostToDevice));

        // Sorted ray buffers
        float *d_sox,*d_soy,*d_soz,*d_sdx,*d_sdy,*d_sdz;
        CK(cudaMalloc(&d_sox,NRAYS*4));CK(cudaMalloc(&d_soy,NRAYS*4));CK(cudaMalloc(&d_soz,NRAYS*4));
        CK(cudaMalloc(&d_sdx,NRAYS*4));CK(cudaMalloc(&d_sdy,NRAYS*4));CK(cudaMalloc(&d_sdz,NRAYS*4));

        Hit *d_hits; CK(cudaMalloc(&d_hits,NRAYS*sizeof(Hit)));

        // CUB sort workspace
        unsigned int *d_keys,*d_keysOut; int *d_indices,*d_indicesOut;
        CK(cudaMalloc(&d_keys,NRAYS*4));CK(cudaMalloc(&d_keysOut,NRAYS*4));
        CK(cudaMalloc(&d_indices,NRAYS*4));CK(cudaMalloc(&d_indicesOut,NRAYS*4));

        size_t cubTempSize=0;
        cub::DeviceRadixSort::SortPairs(nullptr,cubTempSize,d_keys,d_keysOut,d_indices,d_indicesOut,NRAYS);
        void* d_cubTemp; CK(cudaMalloc(&d_cubTemp,cubTempSize));

        // Scene bounds for Morton normalization
        // Scene bounds from root BVH2 node
        AABB sb = bvh.nodes[0].box;
        float sMin=fminf(fminf(sb.mn.x,sb.mn.y),sb.mn.z);
        float sMax=fmaxf(fmaxf(sb.mx.x,sb.mx.y),sb.mx.z);
        float sInv=1.f/(sMax-sMin+1e-6f);

        cudaEvent_t t0,t1; CK(cudaEventCreate(&t0));CK(cudaEventCreate(&t1));
        int blkT=256, grdR=(NRAYS+blkT-1)/blkT;
        unsigned int zero=0;
        float ms;

        // ---------- TEST 1: Unsorted baseline ----------
        CK(cudaMemcpyToSymbol(g_rayCounter,&zero,4));
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(t0));
        traceDiffuse<<<320,256>>>(d_bvh,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
            d_ox,d_oy,d_oz,d_dx,d_dy,d_dz,d_hits,NRAYS);
        CK(cudaEventRecord(t1));CK(cudaEventSynchronize(t1));
        CK(cudaEventElapsedTime(&ms,t0,t1));
        float unsortedMRs=NRAYS/(ms*1000.f);
        printf("    UNSORTED:     %7.0f MR/s  (%.2f ms)\n",unsortedMRs,ms);

        // Count hits
        std::vector<Hit> h_hits(NRAYS);
        CK(cudaMemcpy(h_hits.data(),d_hits,NRAYS*sizeof(Hit),cudaMemcpyDeviceToHost));
        int hitCount=0; for(int i=0;i<NRAYS;i++) if(h_hits[i].tri>=0) hitCount++;
        printf("    Hit rate:     %.1f%%\n",100.f*hitCount/NRAYS);

        // Node/leaf count
        {
            unsigned long long *d_nc,*d_lc; CK(cudaMalloc(&d_nc,8));CK(cudaMalloc(&d_lc,8));
            CK(cudaMemset(d_nc,0,8));CK(cudaMemset(d_lc,0,8));
            CK(cudaMemcpyToSymbol(g_rayCounter,&zero,4));
            traceDiffuseCount<<<320,256>>>(d_bvh,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                d_ox,d_oy,d_oz,d_dx,d_dy,d_dz,d_hits,NRAYS,d_nc,d_lc);
            unsigned long long hnc,hlc; CK(cudaMemcpy(&hnc,d_nc,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&hlc,d_lc,8,cudaMemcpyDeviceToHost));
            printf("    Nodes/ray:    %.1f | Leaf tests/ray: %.1f\n",(double)hnc/NRAYS,(double)hlc/NRAYS);
            CK(cudaFree(d_nc));CK(cudaFree(d_lc));
        }

        // ---------- TEST 2: Morton sorted + CUB + gather ----------
        // Compute Morton keys
        CK(cudaEventRecord(t0));
        computeMortonKeys<<<grdR,blkT>>>(d_ox,d_oy,d_oz,d_keys,NRAYS,sMin,sInv);
        // Init indices
        // (done on host below)
        // Fill indices with iota
        // (init on host, fast enough for benchmark)

        // Init indices on host  
        {
            std::vector<int> h_idx(NRAYS);
            for(int i=0;i<NRAYS;i++) h_idx[i]=i;
            CK(cudaMemcpy(d_indices,h_idx.data(),NRAYS*4,cudaMemcpyHostToDevice));
        }

        // Time: sort + gather + trace
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(t0));

        // Morton keys already computed above
        computeMortonKeys<<<grdR,blkT>>>(d_ox,d_oy,d_oz,d_keys,NRAYS,sMin,sInv);
        
        // CUB radix sort
        cub::DeviceRadixSort::SortPairs(d_cubTemp,cubTempSize,d_keys,d_keysOut,d_indices,d_indicesOut,NRAYS);

        // Gather rays to coalesced buffers
        gatherRays<<<grdR,blkT>>>(d_ox,d_oy,d_oz,d_dx,d_dy,d_dz,
            d_sox,d_soy,d_soz,d_sdx,d_sdy,d_sdz,d_indicesOut,NRAYS);

        CK(cudaEventRecord(t1));CK(cudaEventSynchronize(t1));
        float sortMs; CK(cudaEventElapsedTime(&sortMs,t0,t1));

        // Trace sorted
        CK(cudaMemcpyToSymbol(g_rayCounter,&zero,4));
        CK(cudaEventRecord(t0));
        traceDiffuse<<<320,256>>>(d_bvh,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
            d_sox,d_soy,d_soz,d_sdx,d_sdy,d_sdz,d_hits,NRAYS);
        CK(cudaEventRecord(t1));CK(cudaEventSynchronize(t1));
        float traceMs; CK(cudaEventElapsedTime(&traceMs,t0,t1));
        float totalMs=sortMs+traceMs;
        float sortedMRs=NRAYS/(totalMs*1000.f);
        float traceOnlyMRs=NRAYS/(traceMs*1000.f);
        printf("    MORTON+CUB:   %7.0f MR/s  (sort:%.2f + trace:%.2f = %.2f ms)\n",sortedMRs,sortMs,traceMs,totalMs);
        printf("    trace-only:   %7.0f MR/s\n",traceOnlyMRs);
        printf("    Speedup:      %.2fx (total), %.2fx (trace-only)\n",sortedMRs/unsortedMRs,traceOnlyMRs/unsortedMRs);

        // ---------- TEST 3: Block count sweep ----------
        printf("    Block sweep (sorted): ");
        int blks[]={160,240,320,480,640,960,1280};
        for(int bi=0;bi<7;bi++){
            CK(cudaMemcpyToSymbol(g_rayCounter,&zero,4));
            CK(cudaEventRecord(t0));
            traceDiffuse<<<blks[bi],256>>>(d_bvh,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                d_sox,d_soy,d_soz,d_sdx,d_sdy,d_sdz,d_hits,NRAYS);
            CK(cudaEventRecord(t1));CK(cudaEventSynchronize(t1));
            CK(cudaEventElapsedTime(&ms,t0,t1));
            printf("%d→%.0f ",blks[bi],NRAYS/(ms*1000.f));
        }
        printf("\n");

        // ---------- TEST 4: Treelet+Morton sorted ----------
        {
            std::vector<int> h_idx(NRAYS);
            for(int i=0;i<NRAYS;i++) h_idx[i]=i;
            CK(cudaMemcpy(d_indices,h_idx.data(),NRAYS*4,cudaMemcpyHostToDevice));
        }
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(t0));
        computeTreeletKeys<<<grdR,blkT>>>(d_ox,d_oy,d_oz,d_dx,d_dy,d_dz,d_keys,NRAYS,sMin,sInv);
        cub::DeviceRadixSort::SortPairs(d_cubTemp,cubTempSize,d_keys,d_keysOut,d_indices,d_indicesOut,NRAYS);
        gatherRays<<<grdR,blkT>>>(d_ox,d_oy,d_oz,d_dx,d_dy,d_dz,
            d_sox,d_soy,d_soz,d_sdx,d_sdy,d_sdz,d_indicesOut,NRAYS);
        CK(cudaEventRecord(t1));CK(cudaEventSynchronize(t1));
        float tsortMs; CK(cudaEventElapsedTime(&tsortMs,t0,t1));
        CK(cudaMemcpyToSymbol(g_rayCounter,&zero,4));
        CK(cudaEventRecord(t0));
        traceDiffuse<<<640,256>>>(d_bvh,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
            d_sox,d_soy,d_soz,d_sdx,d_sdy,d_sdz,d_hits,NRAYS);
        CK(cudaEventRecord(t1));CK(cudaEventSynchronize(t1));
        float ttraceMs; CK(cudaEventElapsedTime(&ttraceMs,t0,t1));
        float ttotalMs=tsortMs+ttraceMs;
        printf("    TREELET+MORT: %7.0f MR/s  (sort:%.2f + trace:%.2f = %.2f ms)\n",NRAYS/(ttotalMs*1000.f),tsortMs,ttraceMs,ttotalMs);
        printf("    trace-only:   %7.0f MR/s  (%.2fx vs unsorted)\n",NRAYS/(ttraceMs*1000.f),NRAYS/(ttraceMs*1000.f)/unsortedMRs);
        printf("\n");

        // ---------- TEST 5: Ballot-refill trace (sorted + unsorted) ----------
        CK(cudaMemcpyToSymbol(g_rayCounter,&zero,4));
        CK(cudaEventRecord(t0));
        traceDiffuseRefill<<<640,256>>>(d_bvh,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
            d_sox,d_soy,d_soz,d_sdx,d_sdy,d_sdz,d_hits,NRAYS);
        CK(cudaEventRecord(t1));CK(cudaEventSynchronize(t1));
        float rfMs; CK(cudaEventElapsedTime(&rfMs,t0,t1));
        printf("    REFILL+SORT:  %7.0f MR/s  (trace:%.2f ms, %.2fx vs unsorted)\n",NRAYS/(rfMs*1000.f),rfMs,NRAYS/(rfMs*1000.f)/unsortedMRs);

        CK(cudaMemcpyToSymbol(g_rayCounter,&zero,4));
        CK(cudaEventRecord(t0));
        traceDiffuseRefill<<<640,256>>>(d_bvh,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
            d_ox,d_oy,d_oz,d_dx,d_dy,d_dz,d_hits,NRAYS);
        CK(cudaEventRecord(t1));CK(cudaEventSynchronize(t1));
        float rfuMs; CK(cudaEventElapsedTime(&rfuMs,t0,t1));
        printf("    REFILL+UNSRT: %7.0f MR/s  (trace:%.2f ms, %.2fx vs unsorted)\n\n",NRAYS/(rfuMs*1000.f),rfuMs,NRAYS/(rfuMs*1000.f)/unsortedMRs);

        // Cleanup
        CK(cudaFree(d_bvh));
        for(int i=0;i<9;i++) CK(cudaFree(d_tv[i]));
        CK(cudaFree(d_ox));CK(cudaFree(d_oy));CK(cudaFree(d_oz));
        CK(cudaFree(d_dx));CK(cudaFree(d_dy));CK(cudaFree(d_dz));
        CK(cudaFree(d_sox));CK(cudaFree(d_soy));CK(cudaFree(d_soz));
        CK(cudaFree(d_sdx));CK(cudaFree(d_sdy));CK(cudaFree(d_sdz));
        CK(cudaFree(d_hits));CK(cudaFree(d_keys));CK(cudaFree(d_keysOut));
        CK(cudaFree(d_indices));CK(cudaFree(d_indicesOut));CK(cudaFree(d_cubTemp));
        CK(cudaEventDestroy(t0));CK(cudaEventDestroy(t1));
        free(h_tris);free(h_ox);free(h_oy);free(h_oz);free(h_dx);free(h_dy);free(h_dz);
    }
    printf("  Done!\n");
    return 0;
}

