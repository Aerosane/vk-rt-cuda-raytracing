/* cuda_bvh_backend.cu — CUDA BVH4+CWBVH builder/tracer backend for the Vulkan RT layer
 *
 * Extracted from v37 hybrid engine. Builds BVH4 (for primary) and CWBVH (for diffuse)
 * from raw triangle data. Provides trace functions callable from the Vulkan layer.
 */

#include "cuda_bvh_backend.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cfloat>
#include <algorithm>

#define CK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"[CudaBVH] CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));}}while(0)

// ======================== Data Structures ========================
struct float3a { float x,y,z; };
struct Tri { float3a v0,v1,v2; };
struct AABB { float3a mn,mx; };
struct BVH4Node {
    __half boundsX[8], boundsY[8], boundsZ[8];
    int child[4];
};
struct Hit { float t; int tri; float u, v; };

// ======================== BVH4 Device Code ========================
#define STACK_DEPTH 16
#define CONST_BVH4 1023

__device__ int g_bvhRayCounter;
__device__ int g_bvhRayCounter2;
__constant__ int4 c_bvh4_bk[CONST_BVH4 * 4];
__constant__ int c_bvh4N_bk;

__device__ __forceinline__ void loadBVH4Node_bk(const int4* __restrict__ bvh, int ni,
    int4& n0, int4& n1, int4& n2, int4& n3)
{
    if (ni < c_bvh4N_bk) {
        n0 = c_bvh4_bk[ni*4]; n1 = c_bvh4_bk[ni*4+1]; n2 = c_bvh4_bk[ni*4+2]; n3 = c_bvh4_bk[ni*4+3];
    } else {
        n0 = __ldg(&bvh[ni*4]); n1 = __ldg(&bvh[ni*4+1]); n2 = __ldg(&bvh[ni*4+2]); n3 = __ldg(&bvh[ni*4+3]);
    }
}

// BVH4 broadphase root test
__global__ void __launch_bounds__(256, 8) bk_broadphaseRoot(
    int* __restrict__ d_survivors, int* __restrict__ d_numSurvivors,
    int numRays, int side, float camOx, float camOy, float camOz)
{
    const unsigned lane = threadIdx.x & 31;
    while (true) {
        int bs; if (lane == 0) bs = atomicAdd(&g_bvhRayCounter, 32);
        bs = __shfl_sync(0xFFFFFFFF, bs, 0);
        if (bs >= numRays) break;
        int ri = bs + lane;
        bool anyHit = false;
        if (ri < numRays) {
            int px = ri % side, py = ri / side;
            float u = (px + 0.5f) / side * 2.f - 1.f;
            float v = (py + 0.5f) / side * 2.f - 1.f;
            float rlen = rsqrtf(u*u + v*v + 1.f);
            float ox=camOx, oy=camOy, oz=camOz;
            float dx=u*rlen, dy=v*rlen, dz=rlen;
            float ix = 1.f/(fabsf(dx)>1e-8f?dx:copysignf(1e-8f,dx));
            float iy = 1.f/(fabsf(dy)>1e-8f?dy:copysignf(1e-8f,dy));
            float iz = 1.f/dz;
            int4 n0=c_bvh4_bk[0],n1=c_bvh4_bk[1],n2=c_bvh4_bk[2],n3=c_bvh4_bk[3];
            const __half* bx=(const __half*)&n0,*by=(const __half*)&n1,*bz=(const __half*)&n2;
            const int* ch=(const int*)&n3;
            for(int c=0;c<4;c++){
                if(ch[c]==-1)continue;
                float t1x=(__half2float(bx[c])-ox)*ix,t2x=(__half2float(bx[4+c])-ox)*ix;
                float t1y=(__half2float(by[c])-oy)*iy,t2y=(__half2float(by[4+c])-oy)*iy;
                float t1z=(__half2float(bz[c])-oz)*iz,t2z=(__half2float(bz[4+c])-oz)*iz;
                float tN=fmaxf(fmaxf(fminf(t1x,t2x),fminf(t1y,t2y)),fminf(t1z,t2z));
                float tF=fminf(fminf(fmaxf(t1x,t2x),fmaxf(t1y,t2y)),fmaxf(t1z,t2z));
                if(tN<=tF&&tF>0.f){anyHit=true;break;}
            }
        }
        unsigned mask=__ballot_sync(0xFFFFFFFF,anyHit);
        int warpHits=__popc(mask);
        int warpBase;
        if(lane==0&&warpHits>0) warpBase=atomicAdd(d_numSurvivors,warpHits);
        warpBase=__shfl_sync(0xFFFFFFFF,warpBase,0);
        if(anyHit){
            int myIdx=__popc(mask&((1u<<lane)-1));
            d_survivors[warpBase+myIdx]=ri;
        }
    }
}

// BVH4 dense primary traversal
__global__ void __launch_bounds__(256, 5) bk_tracePrimaryDense(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const int* __restrict__ d_survivors, const int* __restrict__ d_numSurvivors,
    float* __restrict__ hitT, int side, float camOx, float camOy, float camOz)
{
    __shared__ int s_numS;
    if(threadIdx.x==0) s_numS=*d_numSurvivors;
    __syncthreads();
    int numS=s_numS;
    const unsigned lane=threadIdx.x&31;
    while(true){
        int bs; if(lane==0) bs=atomicAdd(&g_bvhRayCounter2,32);
        bs=__shfl_sync(0xFFFFFFFF,bs,0);
        if(bs>=numS)break;
        int localIdx=bs+lane;
        int ri=(localIdx<numS)?d_survivors[localIdx]:-1;
        bool alive=(ri>=0);
        float ox=0,oy=0,oz=0,dx=0,dy=0,dz=1,ix=0,iy=0,iz=1;
        if(alive){
            int px=ri%side,py=ri/side;
            float u=(px+0.5f)/side*2.f-1.f;
            float v=(py+0.5f)/side*2.f-1.f;
            float rlen=rsqrtf(u*u+v*v+1.f);
            ox=camOx;oy=camOy;oz=camOz;
            dx=u*rlen;dy=v*rlen;dz=rlen;
            ix=1.f/(fabsf(dx)>1e-8f?dx:copysignf(1e-8f,dx));
            iy=1.f/(fabsf(dy)>1e-8f?dy:copysignf(1e-8f,dy));
            iz=1.f/dz;
        }
        float tHit=1e30f;
        int stk[STACK_DEPTH];int sp=0;
        if(alive)stk[sp++]=0;
        while(sp>0&&alive){
            int ni=stk[--sp];
            if(ni<0){
                int enc=-(ni+2);int ts=enc>>3,tc=(enc&7)+1;
                for(int t=0;t<tc;t++){
                    int ti=ts+t;
                    float e1x=__ldg(&tv1x[ti])-__ldg(&tv0x[ti]),e1y=__ldg(&tv1y[ti])-__ldg(&tv0y[ti]),e1z=__ldg(&tv1z[ti])-__ldg(&tv0z[ti]);
                    float e2x=__ldg(&tv2x[ti])-__ldg(&tv0x[ti]),e2y=__ldg(&tv2y[ti])-__ldg(&tv0y[ti]),e2z=__ldg(&tv2z[ti])-__ldg(&tv0z[ti]);
                    float ppx=dy*e2z-dz*e2y,ppy=dz*e2x-dx*e2z,ppz=dx*e2y-dy*e2x;
                    float det=e1x*ppx+e1y*ppy+e1z*ppz;
                    if(fabsf(det)<1e-12f)continue;
                    float inv=1.f/det;
                    float tx=ox-__ldg(&tv0x[ti]),ty=oy-__ldg(&tv0y[ti]),tz=oz-__ldg(&tv0z[ti]);
                    float uu=inv*(tx*ppx+ty*ppy+tz*ppz);if(uu<0.f||uu>1.f)continue;
                    float qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x;
                    float vv=inv*(dx*qx+dy*qy+dz*qz);if(vv<0.f||uu+vv>1.f)continue;
                    float tt=inv*(e2x*qx+e2y*qy+e2z*qz);
                    if(tt>0.f&&tt<tHit)tHit=tt;
                }
                continue;
            }
            int4 n0,n1,n2,n3;
            loadBVH4Node_bk(bvh,ni,n0,n1,n2,n3);
            const __half* bx=(const __half*)&n0,*by=(const __half*)&n1,*bz=(const __half*)&n2;
            const int* ch=(const int*)&n3;
            float dist[4];int child[4];
            for(int c=0;c<4;c++){
                child[c]=ch[c];
                if(ch[c]==-1){dist[c]=1e30f;continue;}
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
            for(int c=3;c>=0;c--)if(dist[c]<1e30f&&sp<STACK_DEPTH)stk[sp++]=child[c];
        }
        if(alive) hitT[ri]=tHit;
    }
}

// ======================== CPU BVH Builders ========================
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

struct BVH2Node { AABB box; int left,right,triStart,triCount; };

struct BVH2Builder {
    std::vector<BVH2Node> nodes;
    std::vector<Tri> ordered;
    std::vector<AABB> primBB;
    std::vector<float3a> centroids;
    const Tri* src;
    void build(const Tri* t, int n) {
        src=t;primBB.resize(n);centroids.resize(n);ordered.clear();
        nodes.reserve(n*2);
        for(int i=0;i<n;i++){primBB[i]=triAABB(t[i]);centroids[i]={
            (primBB[i].mn.x+primBB[i].mx.x)*.5f,(primBB[i].mn.y+primBB[i].mx.y)*.5f,(primBB[i].mn.z+primBB[i].mx.z)*.5f};}
        std::vector<int> idx(n);for(int i=0;i<n;i++)idx[i]=i;
        buildRec(idx,0,n);
    }
    int buildRec(std::vector<int>&idx,int s,int e){
        BVH2Node nd;nd.triStart=nd.triCount=nd.left=nd.right=0;
        nd.box=primBB[idx[s]];for(int i=s+1;i<e;i++)nd.box=mergeAABB(nd.box,primBB[idx[i]]);
        int cnt=e-s;
        if(cnt<=3){nd.triStart=(int)ordered.size();nd.triCount=cnt;
            for(int i=s;i<e;i++)ordered.push_back(src[idx[i]]);
            nodes.push_back(nd);return(int)nodes.size()-1;}
        float bestCost=1e30f;int bestAxis=0,bestSplit=s+cnt/2;float pA=saArea(nd.box);
        for(int ax=0;ax<3;ax++){
            float cmin=1e30f,cmax=-1e30f;
            for(int i=s;i<e;i++){float c=(&centroids[idx[i]].x)[ax];cmin=fminf(cmin,c);cmax=fmaxf(cmax,c);}
            if(cmax-cmin<1e-8f)continue;
            const int NB=16;AABB lBox[NB],rBox[NB];int lCnt[NB],rCnt[NB];
            for(int b=0;b<NB;b++){lBox[b].mn={1e30f,1e30f,1e30f};lBox[b].mx={-1e30f,-1e30f,-1e30f};lCnt[b]=0;
                rBox[b].mn={1e30f,1e30f,1e30f};rBox[b].mx={-1e30f,-1e30f,-1e30f};rCnt[b]=0;}
            for(int i=s;i<e;i++){float c=(&centroids[idx[i]].x)[ax];
                int b=(int)((c-cmin)/(cmax-cmin)*(NB-1));b=b<0?0:(b>=NB?NB-1:b);
                lBox[b]=(lCnt[b]==0)?primBB[idx[i]]:mergeAABB(lBox[b],primBB[idx[i]]);lCnt[b]++;}
            for(int b=1;b<NB;b++){if(lCnt[b]&&lCnt[b-1])lBox[b]=mergeAABB(lBox[b],lBox[b-1]);else if(lCnt[b-1])lBox[b]=lBox[b-1];lCnt[b]+=lCnt[b-1];}
            for(int i=e-1;i>=s;i--){float c=(&centroids[idx[i]].x)[ax];
                int b=(int)((c-cmin)/(cmax-cmin)*(NB-1));b=b<0?0:(b>=NB?NB-1:b);
                rBox[b]=(rCnt[b]==0)?primBB[idx[i]]:mergeAABB(rBox[b],primBB[idx[i]]);rCnt[b]++;}
            for(int b=NB-2;b>=0;b--){if(rCnt[b]&&rCnt[b+1])rBox[b]=mergeAABB(rBox[b],rBox[b+1]);else if(rCnt[b+1])rBox[b]=rBox[b+1];rCnt[b]+=rCnt[b+1];}
            for(int b=0;b<NB-1;b++){if(lCnt[b]==0||rCnt[b+1]==0)continue;
                float cost=lCnt[b]*saArea(lBox[b])/pA+rCnt[b+1]*saArea(rBox[b+1])/pA+1.f;
                if(cost<bestCost){bestCost=cost;bestAxis=ax;
                    float splitC=cmin+(b+1.f)/NB*(cmax-cmin);bestSplit=s;
                    for(int i=s;i<e;i++)if((&centroids[idx[i]].x)[ax]<splitC)bestSplit++;
                    bestSplit=bestSplit<=s?s+1:(bestSplit>=e?e-1:bestSplit);}}}
        if(bestSplit<=s)bestSplit=s+1;if(bestSplit>=e)bestSplit=e-1;
        std::sort(idx.begin()+s,idx.begin()+e,[&](int a,int b){return(&centroids[a].x)[bestAxis]<(&centroids[b].x)[bestAxis];});
        int id=(int)nodes.size();nodes.push_back(nd);
        int lc=buildRec(idx,s,bestSplit);int rc=buildRec(idx,bestSplit,e);
        nodes[id].left=lc;nodes[id].right=rc;return id;
    }
};

static int collapseToB4(const BVH2Builder& b2, int ni, BVH4Node* out, int& cnt) {
    auto& n = b2.nodes[ni];
    if(n.triCount>0){int ts=n.triStart,tc=n.triCount;return-((ts<<3)|(tc-1))-2;}
    int gather[4];int ng=0;
    int ch[2]={n.left,n.right};
    for(int c=0;c<2;c++){
        auto& cn=b2.nodes[ch[c]];
        if(cn.triCount>0||ng>=3){gather[ng++]=ch[c];continue;}
        gather[ng++]=cn.left;gather[ng++]=cn.right;
    }
    BVH4Node nd;
    for(int i=0;i<4;i++){
        if(i<ng){
            auto& cn=b2.nodes[gather[i]];
            nd.boundsX[i]=__float2half(cn.box.mn.x);nd.boundsX[4+i]=__float2half(cn.box.mx.x);
            nd.boundsY[i]=__float2half(cn.box.mn.y);nd.boundsY[4+i]=__float2half(cn.box.mx.y);
            nd.boundsZ[i]=__float2half(cn.box.mn.z);nd.boundsZ[4+i]=__float2half(cn.box.mx.z);
        } else {
            nd.boundsX[i]=__float2half(1e30f);nd.boundsX[4+i]=__float2half(-1e30f);
            nd.boundsY[i]=__float2half(1e30f);nd.boundsY[4+i]=__float2half(-1e30f);
            nd.boundsZ[i]=__float2half(1e30f);nd.boundsZ[4+i]=__float2half(-1e30f);
        }
        nd.child[i]=-1;
    }
    int me=cnt++;
    for(int i=0;i<ng;i++)nd.child[i]=collapseToB4(b2,gather[i],out,cnt);
    out[me]=nd;return me;
}

// ======================== Internal Handle ========================
struct CudaBVHHandle {
    // BVH4
    int4*  d_bvh4;
    int    numB4Nodes;
    float* d_tv0x; float* d_tv0y; float* d_tv0z;
    float* d_tv1x; float* d_tv1y; float* d_tv1z;
    float* d_tv2x; float* d_tv2y; float* d_tv2z;
    int    numOrderedTris;

    // Host-side copies for Vulkan compute upload
    void*  h_bvh4Copy;    // BVH4 nodes (numB4Nodes * 4 * sizeof(int4))
    float* h_triCopy[9];  // SoA triangle arrays (numOrderedTris * sizeof(float) each)
    uint32_t* h_bvh2Stackless;  // DFS-ordered stackless BVH2 (8 uint32s per node)
    int numBVH2Nodes;
    float* h_triPacked;         // Packed tris for BVH2: 3 vec4s per tri (12 floats)
    int numTriPacked;           // Number of vec4s (= numOrderedTris * 3)

    // Scene bounds (for auto-camera)
    float bboxMin[3], bboxMax[3], bboxCenter[3], bboxExtent;

    // Work buffers
    int*   d_survivors;
    int*   d_numSurvivors;
    float* d_hitT;
    int    maxRays;
};

// ======================== Stackless BVH2 Builder ========================
// DFS-ordered BVH2 with skip pointers for stackless GLSL traversal.
// Each node = 32 bytes (2 × uvec4):
//   word0: bminx, bminy, bminz, bmaxx  (as float bits)
//   word1: bmaxy, bmaxz, leaf_enc, skip
// Inner nodes: leaf_enc=0, skip=DFS index of next node after subtree
// Leaf nodes:  leaf_enc=-(triStart*8+(count-1))-2, skip=DFS index after leaf

struct DFSNode {
    float bmin[3], bmax[3];
    int leaf_enc;  // 0=inner, <0=leaf encoding
    int skip;      // DFS index to jump to on miss/after leaf
};

static void buildDFSBVH2(const BVH2Builder& b2, int root,
                          std::vector<DFSNode>& out)
{
    // DFS traversal to assign indices
    std::vector<int> dfs_order;
    std::vector<int> skip_targets;
    
    struct WorkItem { int bvh2_idx; int skip_dfs; };
    std::vector<WorkItem> work_stack;
    work_stack.push_back({root, -1});
    
    while (!work_stack.empty()) {
        WorkItem item = work_stack.back();
        work_stack.pop_back();
        
        dfs_order.push_back(item.bvh2_idx);
        skip_targets.push_back(item.skip_dfs);
        
        auto& n = b2.nodes[item.bvh2_idx];
        if (n.triCount == 0) {
            // Inner: push right first (so left is processed first)
            work_stack.push_back({n.right, item.skip_dfs});
            work_stack.push_back({n.left, -2});  // placeholder, fixed in pass 2
        }
    }
    
    // Pass 2: Fix left children's skip pointers using subtree sizes
    int numDFS = (int)dfs_order.size();
    std::vector<int> subtree_size(numDFS, 1);
    
    for (int i = numDFS - 1; i >= 0; i--) {
        auto& n = b2.nodes[dfs_order[i]];
        if (n.triCount == 0) {
            int left_dfs = i + 1;
            subtree_size[i] = 1 + subtree_size[left_dfs];
            int right_dfs = i + 1 + subtree_size[left_dfs];
            if (right_dfs < numDFS)
                subtree_size[i] += subtree_size[right_dfs];
            if (skip_targets[left_dfs] == -2)
                skip_targets[left_dfs] = right_dfs;
        }
    }
    
    // Build output
    out.resize(numDFS);
    for (int i = 0; i < numDFS; i++) {
        auto& n = b2.nodes[dfs_order[i]];
        out[i].bmin[0] = n.box.mn.x; out[i].bmin[1] = n.box.mn.y; out[i].bmin[2] = n.box.mn.z;
        out[i].bmax[0] = n.box.mx.x; out[i].bmax[1] = n.box.mx.y; out[i].bmax[2] = n.box.mx.z;
        out[i].skip = skip_targets[i];
        if (n.triCount > 0) {
            int ts = n.triStart, tc = n.triCount;
            out[i].leaf_enc = -((ts << 3) | (tc - 1)) - 2;
        } else {
            out[i].leaf_enc = 0;
        }
    }
    
    fprintf(stderr, "[CudaBVH] Stackless BVH2: %d DFS nodes (%.1f KB)\n",
            numDFS, numDFS * 32 / 1024.f);
}

// ======================== API Implementation ========================
extern "C" {

CudaBVH_t cudaBVH_build(const CudaTri* tris, int numTris) {
    fprintf(stderr, "[CudaBVH] Building BVH4 from %d triangles...\n", numTris);

    // Convert to internal format and compute scene bounds
    std::vector<Tri> itris(numTris);
    float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = 0; i < numTris; i++) {
        itris[i].v0 = {tris[i].v0[0], tris[i].v0[1], tris[i].v0[2]};
        itris[i].v1 = {tris[i].v1[0], tris[i].v1[1], tris[i].v1[2]};
        itris[i].v2 = {tris[i].v2[0], tris[i].v2[1], tris[i].v2[2]};
        for (int a = 0; a < 3; a++) {
            float v0a = tris[i].v0[a], v1a = tris[i].v1[a], v2a = tris[i].v2[a];
            bmin[a] = std::min({bmin[a], v0a, v1a, v2a});
            bmax[a] = std::max({bmax[a], v0a, v1a, v2a});
        }
    }

    // Build BVH2
    BVH2Builder bvh2;
    bvh2.build(itris.data(), numTris);
    int numOrd = (int)bvh2.ordered.size();
    fprintf(stderr, "[CudaBVH] BVH2: %d nodes, %d ordered tris\n",
            (int)bvh2.nodes.size(), numOrd);

    // Collapse to BVH4
    int maxB4 = (int)bvh2.nodes.size();
    BVH4Node* h_bvh4 = (BVH4Node*)calloc(maxB4, sizeof(BVH4Node));
    int numB4 = 0;
    collapseToB4(bvh2, 0, h_bvh4, numB4);
    fprintf(stderr, "[CudaBVH] BVH4: %d nodes (%.1f KB)\n", numB4, numB4*64/1024.f);

    // Allocate handle
    CudaBVHHandle* h = new CudaBVHHandle();
    h->numB4Nodes = numB4;
    h->numOrderedTris = numOrd;

    // Upload BVH4
    CK(cudaMalloc(&h->d_bvh4, numB4 * 4 * sizeof(int4)));
    CK(cudaMemcpy(h->d_bvh4, h_bvh4, numB4 * 4 * sizeof(int4), cudaMemcpyHostToDevice));

    // Constant memory
    int constN = (numB4 < CONST_BVH4) ? numB4 : CONST_BVH4;
    CK(cudaMemcpyToSymbol(c_bvh4_bk, h_bvh4, constN * sizeof(BVH4Node)));
    CK(cudaMemcpyToSymbol(c_bvh4N_bk, &constN, sizeof(int)));
    // Keep host copy for Vulkan compute upload
    h->h_bvh4Copy = h_bvh4;  // DON'T free — needed for compute shader

    // Upload SoA triangles
    float *h0x=(float*)malloc(numOrd*4),*h0y=(float*)malloc(numOrd*4),*h0z=(float*)malloc(numOrd*4);
    float *h1x=(float*)malloc(numOrd*4),*h1y=(float*)malloc(numOrd*4),*h1z=(float*)malloc(numOrd*4);
    float *h2x=(float*)malloc(numOrd*4),*h2y=(float*)malloc(numOrd*4),*h2z=(float*)malloc(numOrd*4);
    for(int i=0;i<numOrd;i++){
        h0x[i]=bvh2.ordered[i].v0.x;h0y[i]=bvh2.ordered[i].v0.y;h0z[i]=bvh2.ordered[i].v0.z;
        h1x[i]=bvh2.ordered[i].v1.x;h1y[i]=bvh2.ordered[i].v1.y;h1z[i]=bvh2.ordered[i].v1.z;
        h2x[i]=bvh2.ordered[i].v2.x;h2y[i]=bvh2.ordered[i].v2.y;h2z[i]=bvh2.ordered[i].v2.z;
    }
    CK(cudaMalloc(&h->d_tv0x,numOrd*4));CK(cudaMalloc(&h->d_tv0y,numOrd*4));CK(cudaMalloc(&h->d_tv0z,numOrd*4));
    CK(cudaMalloc(&h->d_tv1x,numOrd*4));CK(cudaMalloc(&h->d_tv1y,numOrd*4));CK(cudaMalloc(&h->d_tv1z,numOrd*4));
    CK(cudaMalloc(&h->d_tv2x,numOrd*4));CK(cudaMalloc(&h->d_tv2y,numOrd*4));CK(cudaMalloc(&h->d_tv2z,numOrd*4));
    CK(cudaMemcpy(h->d_tv0x,h0x,numOrd*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(h->d_tv0y,h0y,numOrd*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(h->d_tv0z,h0z,numOrd*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(h->d_tv1x,h1x,numOrd*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(h->d_tv1y,h1y,numOrd*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(h->d_tv1z,h1z,numOrd*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(h->d_tv2x,h2x,numOrd*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(h->d_tv2y,h2y,numOrd*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(h->d_tv2z,h2z,numOrd*4,cudaMemcpyHostToDevice));
    // Keep host copies for Vulkan compute upload
    h->h_triCopy[0]=h0x; h->h_triCopy[1]=h0y; h->h_triCopy[2]=h0z;
    h->h_triCopy[3]=h1x; h->h_triCopy[4]=h1y; h->h_triCopy[5]=h1z;
    h->h_triCopy[6]=h2x; h->h_triCopy[7]=h2y; h->h_triCopy[8]=h2z;

    // Pre-allocate work buffers for up to 4M rays
    h->maxRays = 4*1024*1024;
    CK(cudaMalloc(&h->d_survivors, h->maxRays * sizeof(int)));
    CK(cudaMalloc(&h->d_numSurvivors, sizeof(int)));
    CK(cudaMalloc(&h->d_hitT, h->maxRays * sizeof(float)));

    // Store scene bounds for auto-camera
    for (int a = 0; a < 3; a++) {
        h->bboxMin[a] = bmin[a];
        h->bboxMax[a] = bmax[a];
        h->bboxCenter[a] = (bmin[a] + bmax[a]) * 0.5f;
    }
    float ex = bmax[0]-bmin[0], ey = bmax[1]-bmin[1], ez = bmax[2]-bmin[2];
    h->bboxExtent = sqrtf(ex*ex + ey*ey + ez*ez);
    fprintf(stderr, "[CudaBVH] Scene bounds: (%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f) extent=%.1f\n",
            bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2], h->bboxExtent);

    // Build stackless BVH2 for GLSL compute shader
    std::vector<DFSNode> dfsNodes;
    buildDFSBVH2(bvh2, 0, dfsNodes);
    h->numBVH2Nodes = (int)dfsNodes.size();
    h->h_bvh2Stackless = (uint32_t*)malloc(h->numBVH2Nodes * 8 * sizeof(uint32_t));
    for (int i = 0; i < h->numBVH2Nodes; i++) {
        uint32_t* dst = h->h_bvh2Stackless + i * 8;
        memcpy(dst + 0, &dfsNodes[i].bmin[0], 4);
        memcpy(dst + 1, &dfsNodes[i].bmin[1], 4);
        memcpy(dst + 2, &dfsNodes[i].bmin[2], 4);
        memcpy(dst + 3, &dfsNodes[i].bmax[0], 4);
        memcpy(dst + 4, &dfsNodes[i].bmax[1], 4);
        memcpy(dst + 5, &dfsNodes[i].bmax[2], 4);
        memcpy(dst + 6, &dfsNodes[i].leaf_enc, 4);
        memcpy(dst + 7, &dfsNodes[i].skip, 4);
    }

    // Pack triangles as vec4 triples for BVH2 SPIR-V traversal
    h->numTriPacked = numOrd * 3;
    h->h_triPacked = (float*)malloc(h->numTriPacked * 4 * sizeof(float));
    for (int i = 0; i < numOrd; i++) {
        float* p = h->h_triPacked + i * 12;
        // p0 = {v0.x, v0.y, v0.z, v1.x}
        p[0] = h->h_triCopy[0][i]; p[1] = h->h_triCopy[1][i]; p[2] = h->h_triCopy[2][i];
        p[3] = h->h_triCopy[3][i];
        // p1 = {v1.y, v1.z, v2.x, v2.y}
        p[4] = h->h_triCopy[4][i]; p[5] = h->h_triCopy[5][i]; p[6] = h->h_triCopy[6][i];
        p[7] = h->h_triCopy[7][i];
        // p2 = {v2.z, 0, 0, 0}
        p[8] = h->h_triCopy[8][i]; p[9] = 0; p[10] = 0; p[11] = 0;
    }
    fprintf(stderr, "[CudaBVH] Packed %d tris → %d vec4s (%.1f KB)\n",
            numOrd, h->numTriPacked, h->numTriPacked * 16 / 1024.f);

    fprintf(stderr, "[CudaBVH] Build complete: %d BVH4 nodes, %d BVH2-DFS nodes, %d tris\n",
            numB4, h->numBVH2Nodes, numOrd);
    return (CudaBVH_t)h;
}

void cudaBVH_destroy(CudaBVH_t bvh) {
    if (!bvh) return;
    CudaBVHHandle* h = (CudaBVHHandle*)bvh;
    CK(cudaFree(h->d_bvh4));
    CK(cudaFree(h->d_tv0x));CK(cudaFree(h->d_tv0y));CK(cudaFree(h->d_tv0z));
    CK(cudaFree(h->d_tv1x));CK(cudaFree(h->d_tv1y));CK(cudaFree(h->d_tv1z));
    CK(cudaFree(h->d_tv2x));CK(cudaFree(h->d_tv2y));CK(cudaFree(h->d_tv2z));
    CK(cudaFree(h->d_survivors));CK(cudaFree(h->d_numSurvivors));CK(cudaFree(h->d_hitT));
    free(h->h_bvh2Stackless);
    free(h->h_triPacked);
    for (int i = 0; i < 9; i++) free(h->h_triCopy[i]);
    delete h;
    fprintf(stderr, "[CudaBVH] Destroyed\n");
}

float cudaBVH_tracePrimary(CudaBVH_t bvh, int side, float camOx, float camOy, float camOz, float* outHitT) {
    CudaBVHHandle* h = (CudaBVHHandle*)bvh;
    int numRays = side * side;
    if (numRays > h->maxRays) { fprintf(stderr, "[CudaBVH] Too many rays\n"); return 0; }

    int zero = 0;
    CK(cudaMemcpyToSymbol(g_bvhRayCounter, &zero, 4));
    CK(cudaMemcpyToSymbol(g_bvhRayCounter2, &zero, 4));
    CK(cudaMemcpy(h->d_numSurvivors, &zero, 4, cudaMemcpyHostToDevice));

    // Initialize hitT to miss
    CK(cudaMemset(h->d_hitT, 0x7f, numRays * sizeof(float))); // ~1e38

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));

    bk_broadphaseRoot<<<640,256>>>(h->d_survivors, h->d_numSurvivors, numRays, side, camOx, camOy, camOz);
    bk_tracePrimaryDense<<<320,256>>>(h->d_bvh4, h->numB4Nodes*4,
        h->d_tv0x, h->d_tv0y, h->d_tv0z,
        h->d_tv1x, h->d_tv1y, h->d_tv1z,
        h->d_tv2x, h->d_tv2y, h->d_tv2z,
        h->d_survivors, h->d_numSurvivors,
        h->d_hitT, side, camOx, camOy, camOz);

    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    CK(cudaEventDestroy(t0));
    CK(cudaEventDestroy(t1));

    if (outHitT) {
        CK(cudaMemcpy(outHitT, h->d_hitT, numRays * sizeof(float), cudaMemcpyDeviceToHost));
    }

    float mrs = (float)numRays / (ms * 1000.f);
    fprintf(stderr, "[CudaBVH] Primary trace: %dx%d = %d rays in %.2fms → %.0f MR/s\n",
            side, side, numRays, ms, mrs);
    return mrs;
}

// ======================== Depth-to-RGBA shading kernel ========================
__global__ void bk_shadeDepth(const float* __restrict__ hitT, uint32_t* __restrict__ rgba,
                               int width, int height, int side, float nearZ, float farZ)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    
    // Map output pixel to the square trace grid
    // The trace grid is side×side with camera looking at center
    // We want to extract the center width×height portion
    int outX = idx % width, outY = idx / width;
    int offsetX = (side - width) / 2;
    int offsetY = (side - height) / 2;
    int traceIdx = (outY + offsetY) * side + (outX + offsetX);
    
    float t = hitT[traceIdx];
    uint32_t color;
    if (t >= 1e20f) {
        // Miss: dark blue sky gradient
        float v = (float)outY / height;
        uint8_t r = (uint8_t)(20 + 10 * (1.f - v));
        uint8_t g = (uint8_t)(30 + 20 * (1.f - v));
        uint8_t b = (uint8_t)(60 + 100 * v);
        color = (255u << 24) | (b << 16) | (g << 8) | r;
    } else {
        // Hit: depth-based shading (white near, dark far)
        float d = fminf(fmaxf((t - nearZ) / (farZ - nearZ), 0.f), 1.f);
        float shade = 1.f - d * 0.85f;
        uint8_t c = (uint8_t)(shade * 245.f + 10.f);
        color = (255u << 24) | (c << 16) | (c << 8) | c;
    }
    rgba[idx] = color;
}

// Combined trace+shade kernel with arbitrary camera look-at direction
// outFmt: 0=RGBA8, 1=BGRA8, 2=RGBA16F
__global__ void bk_traceShadeRGBA(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    void* __restrict__ outBuf,
    int width, int height,
    float camOx, float camOy, float camOz,
    float fwdX, float fwdY, float fwdZ,
    float rightX, float rightY, float rightZ,
    float upX, float upY, float upZ,
    float fov, float nearZ, float farZ,
    int outFmt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int px = idx % width, py = idx / width;
    float u = ((float)px + 0.5f) / width * 2.f - 1.f;
    float v = ((float)py + 0.5f) / height * 2.f - 1.f;
    float aspect = (float)width / height;
    u *= aspect * fov;
    v *= fov;

    // Ray direction = forward + u*right + v*up
    float dx = fwdX + u * rightX + v * upX;
    float dy = fwdY + u * rightY + v * upY;
    float dz = fwdZ + u * rightZ + v * upZ;
    float rlen = rsqrtf(dx*dx + dy*dy + dz*dz);
    dx *= rlen; dy *= rlen; dz *= rlen;

    float ox = camOx, oy = camOy, oz = camOz;
    float ix = 1.f/(fabsf(dx)>1e-8f?dx:copysignf(1e-8f,dx));
    float iy = 1.f/(fabsf(dy)>1e-8f?dy:copysignf(1e-8f,dy));
    float iz = 1.f/(fabsf(dz)>1e-8f?dz:copysignf(1e-8f,dz));

    float tHit = 1e30f;
    int hitTri = -1;
    int stk[STACK_DEPTH]; int sp = 0;
    stk[sp++] = 0;

    while (sp > 0) {
        int ni = stk[--sp];
        if (ni < 0) {
            int enc = -(ni+2); int ts = enc>>3, tc = (enc&7)+1;
            for (int t = 0; t < tc; t++) {
                int ti = ts+t;
                float e1x=__ldg(&tv1x[ti])-__ldg(&tv0x[ti]),e1y=__ldg(&tv1y[ti])-__ldg(&tv0y[ti]),e1z=__ldg(&tv1z[ti])-__ldg(&tv0z[ti]);
                float e2x=__ldg(&tv2x[ti])-__ldg(&tv0x[ti]),e2y=__ldg(&tv2y[ti])-__ldg(&tv0y[ti]),e2z=__ldg(&tv2z[ti])-__ldg(&tv0z[ti]);
                float ppx=dy*e2z-dz*e2y,ppy=dz*e2x-dx*e2z,ppz=dx*e2y-dy*e2x;
                float det=e1x*ppx+e1y*ppy+e1z*ppz;
                if(fabsf(det)<1e-12f) continue;
                float inv=1.f/det;
                float tx=ox-__ldg(&tv0x[ti]),ty=oy-__ldg(&tv0y[ti]),tz=oz-__ldg(&tv0z[ti]);
                float uu=inv*(tx*ppx+ty*ppy+tz*ppz); if(uu<0.f||uu>1.f) continue;
                float qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x;
                float vv=inv*(dx*qx+dy*qy+dz*qz); if(vv<0.f||uu+vv>1.f) continue;
                float tt=inv*(e2x*qx+e2y*qy+e2z*qz);
                if(tt>0.f && tt<tHit) { tHit=tt; hitTri=ti; }
            }
            continue;
        }
        int4 n0,n1,n2,n3;
        loadBVH4Node_bk(bvh,ni,n0,n1,n2,n3);
        const __half* bx=(const __half*)&n0,*by=(const __half*)&n1,*bz=(const __half*)&n2;
        const int* ch=(const int*)&n3;
        float dist[4]; int child[4];
        for(int c=0;c<4;c++){
            child[c]=ch[c];
            if(ch[c]==-1){dist[c]=1e30f;continue;}
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
        for(int c=3;c>=0;c--) if(dist[c]<1e30f&&sp<STACK_DEPTH) stk[sp++]=child[c];
    }

    // Shade with face normal lighting
    float rf, gf, bf;
    if (tHit >= 1e20f) {
        // Sky gradient
        float vv = (float)py / height;
        rf = (20.f + 10.f*(1.f-vv)) / 255.f;
        gf = (30.f + 20.f*(1.f-vv)) / 255.f;
        bf = (60.f + 100.f*vv) / 255.f;
    } else {
        // Compute face normal from hit triangle
        float e1x=__ldg(&tv1x[hitTri])-__ldg(&tv0x[hitTri]);
        float e1y=__ldg(&tv1y[hitTri])-__ldg(&tv0y[hitTri]);
        float e1z=__ldg(&tv1z[hitTri])-__ldg(&tv0z[hitTri]);
        float e2x=__ldg(&tv2x[hitTri])-__ldg(&tv0x[hitTri]);
        float e2y=__ldg(&tv2y[hitTri])-__ldg(&tv0y[hitTri]);
        float e2z=__ldg(&tv2z[hitTri])-__ldg(&tv0z[hitTri]);
        float nx=e1y*e2z-e1z*e2y, ny=e1z*e2x-e1x*e2z, nz=e1x*e2y-e1y*e2x;
        float nlen = rsqrtf(nx*nx+ny*ny+nz*nz+1e-20f);
        nx*=nlen; ny*=nlen; nz*=nlen;

        if (nx*dx+ny*dy+nz*dz > 0.f) { nx=-nx; ny=-ny; nz=-nz; }

        float lx=0.5f, ly=0.7f, lz=0.5f;
        float ll=rsqrtf(lx*lx+ly*ly+lz*lz); lx*=ll; ly*=ll; lz*=ll;
        float ndotl = fmaxf(nx*lx+ny*ly+nz*lz, 0.f);

        float ambient = 0.15f;
        float diffuse = 0.65f * ndotl;
        float ndotv = fabsf(nx*(-dx)+ny*(-dy)+nz*(-dz));
        float rim = 0.2f * powf(1.f - ndotv, 3.f);
        float shade = fminf(ambient + diffuse + rim, 1.f);

        rf = shade * 0.92f + 0.08f;
        gf = shade * 0.88f + 0.07f;
        bf = shade * 0.82f + 0.06f;
    }

    // Output in requested format
    if (outFmt == 3) {
        // RGBA32F (R32G32B32A32_SFLOAT) — 16 bytes per pixel
        float* out32 = (float*)outBuf;
        out32[idx*4+0] = rf;
        out32[idx*4+1] = gf;
        out32[idx*4+2] = bf;
        out32[idx*4+3] = 1.0f;
    } else if (outFmt == 2) {
        // RGBA16F (R16G16B16A16_SFLOAT) — 8 bytes per pixel
        __half* out16 = (__half*)outBuf;
        out16[idx*4+0] = __float2half(rf);
        out16[idx*4+1] = __float2half(gf);
        out16[idx*4+2] = __float2half(bf);
        out16[idx*4+3] = __float2half(1.0f);
    } else {
        // RGBA8 or BGRA8 — 4 bytes per pixel
        uint8_t r = (uint8_t)(rf * 255.f);
        uint8_t g = (uint8_t)(gf * 255.f);
        uint8_t b = (uint8_t)(bf * 255.f);
        uint32_t color = (outFmt == 1) ?
            (255u<<24)|(r<<16)|(g<<8)|b :   // BGRA
            (255u<<24)|(b<<16)|(g<<8)|r;    // RGBA
        ((uint32_t*)outBuf)[idx] = color;
    }
}

// Trace and output RGBA buffer with auto look-at camera
int cudaBVH_traceToRGBA(CudaBVH_t bvh, int width, int height,
                        float camOx, float camOy, float camOz,
                        uint32_t* outRGBA_host)
{
    CudaBVHHandle* h = (CudaBVHHandle*)bvh;

    // Compute look-at direction from camera to scene center
    float fwdX = h->bboxCenter[0] - camOx;
    float fwdY = h->bboxCenter[1] - camOy;
    float fwdZ = h->bboxCenter[2] - camOz;
    float flen = sqrtf(fwdX*fwdX + fwdY*fwdY + fwdZ*fwdZ);
    if (flen < 1e-6f) flen = 1.f;
    fwdX /= flen; fwdY /= flen; fwdZ /= flen;

    // Right = forward × (0,1,0)
    float rightX = fwdZ, rightY = 0, rightZ = -fwdX;
    float rlen = sqrtf(rightX*rightX + rightZ*rightZ);
    if (rlen > 1e-6f) { rightX /= rlen; rightZ /= rlen; }
    else { rightX = 1; rightZ = 0; }

    // Up = right × forward
    float upX = rightY*fwdZ - rightZ*fwdY;
    float upY = rightZ*fwdX - rightX*fwdZ;
    float upZ = rightX*fwdY - rightY*fwdX;

    float fov = 0.6f; // ~60° FOV
    float camDist = flen;
    float nearZ = fmaxf(camDist - h->bboxExtent * 0.7f, 0.1f);
    float farZ = camDist + h->bboxExtent * 0.7f;

    uint32_t* d_rgba;
    CK(cudaMalloc(&d_rgba, width * height * 4));
    int blocks = (width * height + 255) / 256;
    bk_traceShadeRGBA<<<blocks, 256>>>(
        h->d_bvh4, h->numB4Nodes*4,
        h->d_tv0x, h->d_tv0y, h->d_tv0z,
        h->d_tv1x, h->d_tv1y, h->d_tv1z,
        h->d_tv2x, h->d_tv2y, h->d_tv2z,
        d_rgba, width, height,
        camOx, camOy, camOz,
        fwdX, fwdY, fwdZ,
        rightX, rightY, rightZ,
        upX, upY, upZ,
        fov, nearZ, farZ,
        0);  // RGBA for PPM output

    CK(cudaMemcpy(outRGBA_host, d_rgba, width * height * 4, cudaMemcpyDeviceToHost));
    CK(cudaFree(d_rgba));
    return 0;
}

// Trace to an externally-provided GPU buffer (for CUDA-Vulkan interop zero-copy)
int cudaBVH_traceToGPUPtr(CudaBVH_t bvh, int width, int height,
                          float camOx, float camOy, float camOz,
                          void* d_outRGBA, int outFmt)
{
    CudaBVHHandle* h = (CudaBVHHandle*)bvh;

    // Compute look-at direction from camera to scene center
    float fwdX = h->bboxCenter[0] - camOx;
    float fwdY = h->bboxCenter[1] - camOy;
    float fwdZ = h->bboxCenter[2] - camOz;
    float flen = sqrtf(fwdX*fwdX + fwdY*fwdY + fwdZ*fwdZ);
    if (flen < 1e-6f) flen = 1.f;
    fwdX /= flen; fwdY /= flen; fwdZ /= flen;

    float rightX = fwdZ, rightY = 0, rightZ = -fwdX;
    float rlen = sqrtf(rightX*rightX + rightZ*rightZ);
    if (rlen > 1e-6f) { rightX /= rlen; rightZ /= rlen; }
    else { rightX = 1; rightZ = 0; }

    float upX = rightY*fwdZ - rightZ*fwdY;
    float upY = rightZ*fwdX - rightX*fwdZ;
    float upZ = rightX*fwdY - rightY*fwdX;

    float fov = 0.6f;
    float camDist = flen;
    float nearZ = fmaxf(camDist - h->bboxExtent * 0.7f, 0.1f);
    float farZ = camDist + h->bboxExtent * 0.7f;

    int blocks = (width * height + 255) / 256;
    bk_traceShadeRGBA<<<blocks, 256>>>(
        h->d_bvh4, h->numB4Nodes*4,
        h->d_tv0x, h->d_tv0y, h->d_tv0z,
        h->d_tv1x, h->d_tv1y, h->d_tv1z,
        h->d_tv2x, h->d_tv2y, h->d_tv2z,
        (uint32_t*)d_outRGBA, width, height,
        camOx, camOy, camOz,
        fwdX, fwdY, fwdZ,
        rightX, rightY, rightZ,
        upX, upY, upZ,
        fov, nearZ, farZ,
        outFmt);  // format passed from layer

    // No explicit sync — caller uses cudaMemcpy which implicitly syncs the default stream
    return 0;
}

// ═══════════════════════════════════════════════════════════════
// Async background tracer: CUDA runs on a separate thread to avoid
// Vulkan-CUDA serialization. Main thread copies from cached buffer.
// ═══════════════════════════════════════════════════════════════
#include <pthread.h>


struct AsyncTracer {
    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t requestCond;
    pthread_cond_t readyCond;
    
    // Double-buffered host output (HOST_VISIBLE path)
    void* hostBuf[2];
    uint64_t bufSize;
    int readIdx;       // buffer ready for reading by main thread
    int writeIdx;      // buffer being written by CUDA thread
    
    // Request parameters
    CudaBVH_t bvh;
    int width, height;
    float camOx, camOy, camOz;
    int outFmt;
    void* gpuTarget;   // non-null = DEVICE_LOCAL path (write directly to GPU)
    
    volatile int hasRequest;
    volatile int hasResult;
    volatile int running;
    bool initialized;
};

static AsyncTracer g_async = {};

static void* asyncTraceThread(void* arg) {
    AsyncTracer* at = (AsyncTracer*)arg;
    void* d_buf = nullptr;
    uint64_t d_bufSize = 0;
    
    while (at->running) {
        // Wait for a trace request
        pthread_mutex_lock(&at->lock);
        while (!at->hasRequest && at->running) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 10000000; // 10ms timeout
            if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
            pthread_cond_timedwait(&at->requestCond, &at->lock, &ts);
        }
        if (!at->running) { pthread_mutex_unlock(&at->lock); break; }
        
        CudaBVH_t bvh = at->bvh;
        int w = at->width, h = at->height;
        float cx = at->camOx, cy = at->camOy, cz = at->camOz;
        int fmt = at->outFmt;
        uint64_t sz = at->bufSize;
        int wIdx = at->writeIdx;
        void* gpuTgt = at->gpuTarget;
        at->hasRequest= 0;
        pthread_mutex_unlock(&at->lock);
        
        // Compute camera basis
        CudaBVHHandle* bh = (CudaBVHHandle*)bvh;
        float fwdX = bh->bboxCenter[0] - cx;
        float fwdY = bh->bboxCenter[1] - cy;
        float fwdZ = bh->bboxCenter[2] - cz;
        float flen = sqrtf(fwdX*fwdX + fwdY*fwdY + fwdZ*fwdZ);
        if (flen < 1e-6f) flen = 1.f;
        fwdX /= flen; fwdY /= flen; fwdZ /= flen;
        float rightX = fwdZ, rightY = 0, rightZ = -fwdX;
        float rlen = sqrtf(rightX*rightX + rightZ*rightZ);
        if (rlen > 1e-6f) { rightX /= rlen; rightZ /= rlen; }
        else { rightX = 1; rightZ = 0; }
        float upX = rightY*fwdZ - rightZ*fwdY;
        float upY = rightZ*fwdX - rightX*fwdZ;
        float upZ = rightX*fwdY - rightY*fwdX;
        float fov = 0.6f;
        float camDist = flen;
        float nearZ = fmaxf(camDist - bh->bboxExtent * 0.7f, 0.1f);
        float farZ = camDist + bh->bboxExtent * 0.7f;
        
        int blocks = (w * h + 255) / 256;

        if (gpuTgt) {
            // DEVICE_LOCAL path: write directly to imported GPU buffer
            bk_traceShadeRGBA<<<blocks, 256>>>(
                bh->d_bvh4, bh->numB4Nodes*4,
                bh->d_tv0x, bh->d_tv0y, bh->d_tv0z,
                bh->d_tv1x, bh->d_tv1y, bh->d_tv1z,
                bh->d_tv2x, bh->d_tv2y, bh->d_tv2z,
                (uint32_t*)gpuTgt, w, h,
                cx, cy, cz,
                fwdX, fwdY, fwdZ,
                rightX, rightY, rightZ,
                upX, upY, upZ,
                fov, nearZ, farZ,
                fmt);
            cudaDeviceSynchronize();  // blocks ~66ms due to Vulkan serialization
        } else {
            // HOST_VISIBLE path: trace to internal buffer, then D→H copy
            if (!d_buf || d_bufSize < sz) {
                if (d_buf) cudaFree(d_buf);
                cudaMalloc(&d_buf, sz);
                d_bufSize = sz;
            }
            bk_traceShadeRGBA<<<blocks, 256>>>(
                bh->d_bvh4, bh->numB4Nodes*4,
                bh->d_tv0x, bh->d_tv0y, bh->d_tv0z,
                bh->d_tv1x, bh->d_tv1y, bh->d_tv1z,
                bh->d_tv2x, bh->d_tv2y, bh->d_tv2z,
                (uint32_t*)d_buf, w, h,
                cx, cy, cz,
                fwdX, fwdY, fwdZ,
                rightX, rightY, rightZ,
                upX, upY, upZ,
                fov, nearZ, farZ,
                fmt);
            cudaMemcpy(at->hostBuf[wIdx], d_buf, sz, cudaMemcpyDeviceToHost);
        }
        
        // Signal result ready
        pthread_mutex_lock(&at->lock);
        at->readIdx = wIdx;
        at->writeIdx = 1 - wIdx;
        at->hasResult= 1;
        pthread_cond_signal(&at->readyCond);
        pthread_mutex_unlock(&at->lock);
    }
    
    if (d_buf) cudaFree(d_buf);
    return nullptr;
}

static void initAsyncTracer(uint64_t bufSize) {
    if (g_async.initialized) return;
    pthread_mutex_init(&g_async.lock, nullptr);
    pthread_cond_init(&g_async.requestCond, nullptr);
    pthread_cond_init(&g_async.readyCond, nullptr);
    g_async.hostBuf[0] = malloc(bufSize);
    g_async.hostBuf[1] = malloc(bufSize);
    g_async.bufSize = bufSize;
    g_async.readIdx = 0;
    g_async.writeIdx = 1;
    g_async.gpuTarget = nullptr;
    g_async.hasRequest= 0;
    g_async.hasResult= 0;
    g_async.running= 1;
    g_async.initialized = true;
    memset(g_async.hostBuf[0], 0, bufSize);
    memset(g_async.hostBuf[1], 0, bufSize);
    pthread_create(&g_async.thread, nullptr, asyncTraceThread, &g_async);
    fprintf(stderr, "[CudaBVH] Async tracer thread started (2×%.1fMB buffers)\n", bufSize/1048576.0);
}

// Main entry point: fires off async trace, copies from last-ready buffer
int cudaBVH_traceToHostPtr(CudaBVH_t bvh, int width, int height,
                           float camOx, float camOy, float camOz,
                           void* h_outRGBA, int outFmt, uint64_t bufSize)
{
    static int s_callCount = 0;
    s_callCount++;
    
    // Initialize async tracer on first call
    if (!g_async.initialized) {
        initAsyncTracer(bufSize);
    }
    
    // Submit trace request (non-blocking)
    pthread_mutex_lock(&g_async.lock);
    g_async.bvh = bvh;
    g_async.width = width;
    g_async.height = height;
    g_async.camOx = camOx;
    g_async.camOy = camOy;
    g_async.camOz = camOz;
    g_async.outFmt = outFmt;
    g_async.hasRequest= 1;
    pthread_cond_signal(&g_async.requestCond);
    pthread_mutex_unlock(&g_async.lock);
    
    // On first frame, wait for the first result (blocking)
    if (s_callCount == 1) {
        pthread_mutex_lock(&g_async.lock);
        while (!g_async.hasResult) {
            pthread_cond_wait(&g_async.readyCond, &g_async.lock);
        }
        pthread_mutex_unlock(&g_async.lock);
        fprintf(stderr, "[CudaBVH] First frame ready\n");
    }
    
    // Copy from ready buffer to staging (host→host, ~1ms for 14MB)
    if (g_async.hasResult) {
        memcpy(h_outRGBA, g_async.hostBuf[g_async.readIdx], bufSize);
    }
    
    if (s_callCount <= 5 || (s_callCount % 200) == 0) {
        fprintf(stderr, "[CudaBVH] Frame %d: async copy from buf[%d] %.1fMB\n",
                s_callCount, g_async.readIdx, bufSize/1048576.0);
    }
    
    return 0;
}

// GPU-direct async trace: fire-and-forget to DEVICE_LOCAL buffer
// The background thread writes directly to the GPU pointer (no D→H copy).
// Main thread records CmdCopyBufferToImage — GPU→GPU copy is fast.
// CUDA-Vulkan serialization ensures no read/write race.
void cudaBVH_traceToGPUAsync(CudaBVH_t bvh, int width, int height,
                             float camOx, float camOy, float camOz,
                             void* d_outRGBA, int outFmt)
{
    static int s_callCount = 0;
    s_callCount++;
    
    uint64_t bpp = (outFmt == 3) ? 16 : (outFmt == 2) ? 8 : 4;
    uint64_t bufSize = (uint64_t)width * height * bpp;
    
    // Initialize async tracer on first call (no host buffers needed for GPU path)
    if (!g_async.initialized) {
        initAsyncTracer(bufSize);
    }
    
    // Submit GPU trace request (non-blocking)
    pthread_mutex_lock(&g_async.lock);
    g_async.bvh = bvh;
    g_async.width = width;
    g_async.height = height;
    g_async.camOx = camOx;
    g_async.camOy = camOy;
    g_async.camOz = camOz;
    g_async.outFmt = outFmt;
    g_async.gpuTarget = d_outRGBA;
    g_async.hasRequest= 1;
    pthread_cond_signal(&g_async.requestCond);
    pthread_mutex_unlock(&g_async.lock);
    
    // On first frame, wait for initial result
    if (s_callCount == 1) {
        pthread_mutex_lock(&g_async.lock);
        while (!g_async.hasResult) {
            pthread_cond_wait(&g_async.readyCond, &g_async.lock);
        }
        pthread_mutex_unlock(&g_async.lock);
        fprintf(stderr, "[CudaBVH] First GPU-direct frame ready\n");
    }
    
    if (s_callCount <= 5 || (s_callCount % 200) == 0) {
        fprintf(stderr, "[CudaBVH] GPU-direct frame %d: target=%p %.1fMB\n",
                s_callCount, d_outRGBA, bufSize/1048576.0);
    }
}
void* cudaBVH_importBufferFd(int fd, uint64_t size)
{
    cudaExternalMemory_t extMem;
    cudaExternalMemoryHandleDesc desc = {};
    desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size = size;
    desc.flags = 0;

    cudaError_t err = cudaImportExternalMemory(&extMem, &desc);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CudaBVH] Failed to import fd %d as external memory: %s\n",
                fd, cudaGetErrorString(err));
        return nullptr;
    }

    void* devPtr = nullptr;
    cudaExternalMemoryBufferDesc bufDesc = {};
    bufDesc.offset = 0;
    bufDesc.size = size;
    bufDesc.flags = 0;

    err = cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufDesc);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CudaBVH] Failed to map external memory: %s\n",
                cudaGetErrorString(err));
        return nullptr;
    }

    fprintf(stderr, "[CudaBVH] Imported fd %d → CUDA ptr %p (%lu bytes)\n",
            fd, devPtr, (unsigned long)size);
    return devPtr;
}

void cudaBVH_getBounds(CudaBVH_t bvh, float* cx, float* cy, float* cz, float* extent) {
    CudaBVHHandle* h = (CudaBVHHandle*)bvh;
    if (cx) *cx = h->bboxCenter[0];
    if (cy) *cy = h->bboxCenter[1];
    if (cz) *cz = h->bboxCenter[2];
    if (extent) *extent = h->bboxExtent;
}

float cudaBVH_traceDiffuse(CudaBVH_t bvh, int numRays,
                           const float* rayOx, const float* rayOy, const float* rayOz,
                           const float* rayDx, const float* rayDy, const float* rayDz,
                           float* outHitT) {
    // TODO: implement CWBVH diffuse trace
    fprintf(stderr, "[CudaBVH] Diffuse trace not yet implemented\n");
    return 0;
}

int cudaBVH_getNumTris(CudaBVH_t bvh) {
    return bvh ? ((CudaBVHHandle*)bvh)->numOrderedTris : 0;
}
int cudaBVH_getNumBVH4Nodes(CudaBVH_t bvh) {
    return bvh ? ((CudaBVHHandle*)bvh)->numB4Nodes : 0;
}
int cudaBVH_getNumCWBVHNodes(CudaBVH_t bvh) {
    return 0; // TODO
}

int cudaBVH_readGPUMemoryFd(int fd, uint64_t allocationSize,
                             uint64_t offset, uint64_t size, void* dst) {
    cudaExternalMemoryHandleDesc extMemDesc = {};
    extMemDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    extMemDesc.handle.fd = fd;
    extMemDesc.size = allocationSize;

    cudaExternalMemory_t extMem;
    cudaError_t err = cudaImportExternalMemory(&extMem, &extMemDesc);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CudaBVH] cudaImportExternalMemory failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    // Map a buffer from the external memory
    cudaExternalMemoryBufferDesc bufDesc = {};
    bufDesc.offset = offset;
    bufDesc.size = size;

    void* devPtr = nullptr;
    err = cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufDesc);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CudaBVH] cudaExternalMemoryGetMappedBuffer failed: %s\n",
                cudaGetErrorString(err));
        cudaDestroyExternalMemory(extMem);
        return -1;
    }

    // Copy GPU → host
    err = cudaMemcpy(dst, devPtr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CudaBVH] cudaMemcpy D2H failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(devPtr);
        cudaDestroyExternalMemory(extMem);
        return -1;
    }

    cudaFree(devPtr);
    cudaDestroyExternalMemory(extMem);
    return 0;
}

} // extern "C"

// Non-extern-C accessors for Vulkan compute upload
extern "C" {

const void* cudaBVH_getNodeData(CudaBVH_t bvh) {
    if (!bvh) return nullptr;
    return ((CudaBVHHandle*)bvh)->h_bvh4Copy;
}

void cudaBVH_getTriData(CudaBVH_t bvh, const float* outPtrs[9]) {
    if (!bvh) { for (int i=0;i<9;i++) outPtrs[i]=nullptr; return; }
    CudaBVHHandle* h = (CudaBVHHandle*)bvh;
    for (int i = 0; i < 9; i++) outPtrs[i] = h->h_triCopy[i];
}

// Get stackless BVH2 data for GLSL compute shader
// Returns number of nodes, fills outData with 8 uint32s per node
int cudaBVH_getStacklessBVH2(CudaBVH_t bvh, uint32_t** outData) {
    if (!bvh) { *outData = nullptr; return 0; }
    CudaBVHHandle* h = (CudaBVHHandle*)bvh;
    if (!h->h_bvh2Stackless) { *outData = nullptr; return 0; }
    *outData = h->h_bvh2Stackless;
    return h->numBVH2Nodes;
}

int cudaBVH_getPackedTris(CudaBVH_t bvh, float** outData) {
    if (!bvh) { *outData = nullptr; return 0; }
    CudaBVHHandle* h = (CudaBVHHandle*)bvh;
    if (!h->h_triPacked) { *outData = nullptr; return 0; }
    *outData = h->h_triPacked;
    return h->numTriPacked;
}

// Build BVH2 from AABBs (for TLAS over instances).
// Creates "degenerate" triangles whose bounding box = the input AABB.
// The BVH leaf encoding maps each "triangle" index back to the AABB/instance index.
CudaBVH_t cudaBVH_buildFromAABBs(const float* aabbs, int numAABBs) {
    if (!aabbs || numAABBs <= 0) return nullptr;
    fprintf(stderr, "[CudaBVH] Building TLAS BVH from %d AABBs...\n", numAABBs);

    // For each AABB, create 1 degenerate triangle whose bbox matches the AABB.
    // We place the 3 vertices at strategic AABB corners so the tri's bbox = the AABB.
    // Specifically: v0 = (minX,minY,minZ), v1 = (maxX,maxY,minZ), v2 = (minX,minY,maxZ)
    // This gives: bbox min = (minX,minY,minZ), bbox max = (maxX,maxY,maxZ)
    // Wait, that misses maxZ in v0/v1. Let's use:
    // v0 = (minX,minY,minZ), v1 = (maxX,maxY,maxZ), v2 = (maxX,minY,minZ)
    // bbox min = min(minX,maxX,maxX, ...) = (minX,minY,minZ) ✓
    // bbox max = max(minX,maxX,maxX, ...) = (maxX,maxY,maxZ) ✓
    std::vector<CudaTri> tris(numAABBs);
    for (int i = 0; i < numAABBs; i++) {
        float mnx = aabbs[i*6+0], mny = aabbs[i*6+1], mnz = aabbs[i*6+2];
        float mxx = aabbs[i*6+3], mxy = aabbs[i*6+4], mxz = aabbs[i*6+5];
        tris[i].v0[0] = mnx; tris[i].v0[1] = mny; tris[i].v0[2] = mnz;
        tris[i].v1[0] = mxx; tris[i].v1[1] = mxy; tris[i].v1[2] = mxz;
        tris[i].v2[0] = mxx; tris[i].v2[1] = mny; tris[i].v2[2] = mnz;
    }

    // Build using the standard BVH builder — each "triangle" = one instance.
    // The BVH leaf encoding (triStart, triCount) will give us the instance index.
    CudaBVH_t bvh = cudaBVH_build(tris.data(), numAABBs);
    if (bvh) {
        fprintf(stderr, "[CudaBVH] TLAS BVH built: %d BVH2 nodes over %d instances\n",
                ((CudaBVHHandle*)bvh)->numBVH2Nodes, numAABBs);
    }
    return bvh;
}

void cudaBVH_resetDevice(void) {
    cudaDeviceSynchronize();
    cudaDeviceReset();
}

} // extern "C"
