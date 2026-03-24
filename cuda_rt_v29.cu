/* v29 — Morton-Sorted Diffuse + Two-Phase Primary
 *
 * PRIMARY (from v28, proven 6263 MR/s):
 *   Phase 1: broadphaseRoot — test root BVH4 (constant mem), warp ballot compact
 *   Phase 2: tracePrimaryDense — full BVH4 traversal on survivors, CSWAP sort
 *
 * DIFFUSE (NEW — sorted megakernel):
 *   Problem: diffuse rays have random origins+directions → each thread in a warp
 *   accesses completely different BVH nodes → 32 L2 cache misses per node fetch.
 *
 *   Solution: Sort all diffuse rays by a combined key:
 *     Key = (direction_octant << 30) | morton3D(origin)
 *   This groups rays by both spatial locality AND direction similarity.
 *   Adjacent threads in a warp then traverse similar BVH paths →
 *   shared L2 cache lines (up to 32× fewer fetches).
 *
 *   Three diffuse variants benchmarked:
 *   1. MEGA (unsorted) — baseline reference
 *   2. MORTON — sorted by Morton code of origin
 *   3. OCTANT+MORTON — sorted by direction octant, then Morton within octant
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cfloat>

#define STACK_DEPTH 16
#define NRAYS (4*1024*1024)
#define BIN_COUNT 16
#define CONST_BVH4 1023

struct float3a { float x, y, z; };
struct AABB { float3a mn, mx; };
struct Tri { float3a v0, v1, v2; };
struct BVH4Node {
    __half boundsX[8], boundsY[8], boundsZ[8];
    int child[4];
};
struct Hit { float t; int tri; float u, v; };

__device__ unsigned int g_rayCounter;
__device__ unsigned int g_rayCounter2;

__constant__ int4 c_bvh4[CONST_BVH4 * 4];
__constant__ int c_bvh4N;

__device__ __forceinline__ void loadBVH4Node(const int4* __restrict__ bvh, int ni,
    int4& n0, int4& n1, int4& n2, int4& n3)
{
    if (ni < c_bvh4N) {
        n0 = c_bvh4[ni*4]; n1 = c_bvh4[ni*4+1]; n2 = c_bvh4[ni*4+2]; n3 = c_bvh4[ni*4+3];
    } else {
        n0 = __ldg(&bvh[ni*4]); n1 = __ldg(&bvh[ni*4+1]); n2 = __ldg(&bvh[ni*4+2]); n3 = __ldg(&bvh[ni*4+3]);
    }
}

// ======================== Morton code for 3D spatial hashing ========================
__device__ __forceinline__ unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__ unsigned int morton3D(float x, float y, float z) {
    // Clamp to [0,1023]
    unsigned int ix = min(max((unsigned int)(x * 1024.f), 0u), 1023u);
    unsigned int iy = min(max((unsigned int)(y * 1024.f), 0u), 1023u);
    unsigned int iz = min(max((unsigned int)(z * 1024.f), 0u), 1023u);
    return (expandBits(ix) << 2) | (expandBits(iy) << 1) | expandBits(iz);
}

// Compute sort key for a diffuse ray: octant(3 bits) << 30 | morton(30 bits)
__global__ void computeSortKeys(
    const float* __restrict__ rox, const float* __restrict__ roy, const float* __restrict__ roz,
    const float* __restrict__ rdx, const float* __restrict__ rdy, const float* __restrict__ rdz,
    unsigned int* __restrict__ keys, int* __restrict__ indices,
    int numRays, float sceneMin, float sceneInvSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRays) return;

    // Normalize origin to [0,1]
    float nx = (rox[i] - sceneMin) * sceneInvSize;
    float ny = (roy[i] - sceneMin) * sceneInvSize;
    float nz = (roz[i] - sceneMin) * sceneInvSize;

    // Direction octant: 3 bits from sign of dx, dy, dz
    unsigned int octant = ((rdx[i] < 0.f) ? 4u : 0u) |
                          ((rdy[i] < 0.f) ? 2u : 0u) |
                          ((rdz[i] < 0.f) ? 1u : 0u);

    unsigned int mc = morton3D(nx, ny, nz);
    keys[i] = (octant << 29) | (mc & 0x1FFFFFFFu);  // 3 bits octant + 29 bits morton
    indices[i] = i;
}

// Morton-only keys (no octant)
__global__ void computeMortonKeys(
    const float* __restrict__ rox, const float* __restrict__ roy, const float* __restrict__ roz,
    unsigned int* __restrict__ keys, int* __restrict__ indices,
    int numRays, float sceneMin, float sceneInvSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRays) return;
    float nx = (rox[i] - sceneMin) * sceneInvSize;
    float ny = (roy[i] - sceneMin) * sceneInvSize;
    float nz = (roz[i] - sceneMin) * sceneInvSize;
    keys[i] = morton3D(nx, ny, nz);
    indices[i] = i;
}

// Gather kernel: physically reorder ray data into sorted order for coalesced access
__global__ void gatherRays(
    const float* __restrict__ src_ox, const float* __restrict__ src_oy, const float* __restrict__ src_oz,
    const float* __restrict__ src_dx, const float* __restrict__ src_dy, const float* __restrict__ src_dz,
    float* __restrict__ dst_ox, float* __restrict__ dst_oy, float* __restrict__ dst_oz,
    float* __restrict__ dst_dx, float* __restrict__ dst_dy, float* __restrict__ dst_dz,
    const int* __restrict__ indices, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ri = indices[i];
    dst_ox[i] = src_ox[ri]; dst_oy[i] = src_oy[ri]; dst_oz[i] = src_oz[ri];
    dst_dx[i] = src_dx[ri]; dst_dy[i] = src_dy[ri]; dst_dz[i] = src_dz[ri];
}

// Scatter-back kernel: write hits from sorted order back to original ray order
__global__ void scatterHits(
    const Hit* __restrict__ sorted_hits, Hit* __restrict__ orig_hits,
    const int* __restrict__ indices, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    orig_hits[indices[i]] = sorted_hits[i];
}

// ======================== Binned SAH BVH Builder ========================
static AABB triAABB(const Tri& t) {
    AABB b;
    b.mn.x = fminf(fminf(t.v0.x, t.v1.x), t.v2.x);
    b.mn.y = fminf(fminf(t.v0.y, t.v1.y), t.v2.y);
    b.mn.z = fminf(fminf(t.v0.z, t.v1.z), t.v2.z);
    b.mx.x = fmaxf(fmaxf(t.v0.x, t.v1.x), t.v2.x);
    b.mx.y = fmaxf(fmaxf(t.v0.y, t.v1.y), t.v2.y);
    b.mx.z = fmaxf(fmaxf(t.v0.z, t.v1.z), t.v2.z);
    return b;
}
static float saArea(const AABB& b) {
    float dx = b.mx.x - b.mn.x, dy = b.mx.y - b.mn.y, dz = b.mx.z - b.mn.z;
    return 2.f * (dx * dy + dy * dz + dx * dz);
}
static AABB mergeAABB(const AABB& a, const AABB& b) {
    AABB r;
    r.mn.x = fminf(a.mn.x, b.mn.x); r.mn.y = fminf(a.mn.y, b.mn.y); r.mn.z = fminf(a.mn.z, b.mn.z);
    r.mx.x = fmaxf(a.mx.x, b.mx.x); r.mx.y = fmaxf(a.mx.y, b.mx.y); r.mx.z = fmaxf(a.mx.z, b.mx.z);
    return r;
}

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
        if (cnt <= 4) {
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

// ======================== Phase 1: Broadphase Root (Primary) ========================
__global__ void __launch_bounds__(256, 8) broadphaseRoot(
    int* __restrict__ d_survivors,
    int* __restrict__ d_numSurvivors,
    int numRays, int side,
    float camOx, float camOy, float camOz)
{
    const unsigned lane = threadIdx.x & 31;
    while (true) {
        int bs; if (lane == 0) bs = atomicAdd(&g_rayCounter, 32);
        bs = __shfl_sync(0xFFFFFFFF, bs, 0);
        if (bs >= numRays) break;
        int ri = bs + lane;
        bool anyHit = false;
        if (ri < numRays) {
            int px = ri % side, py = ri / side;
            float u = (px + 0.5f) / side * 2.f - 1.f;
            float v = (py + 0.5f) / side * 2.f - 1.f;
            float rlen = rsqrtf(u * u + v * v + 1.f);
            float ox = camOx, oy = camOy, oz = camOz;
            float dx = u * rlen, dy = v * rlen, dz = rlen;
            float ix = 1.f / (fabsf(dx) > 1e-8f ? dx : copysignf(1e-8f, dx));
            float iy = 1.f / (fabsf(dy) > 1e-8f ? dy : copysignf(1e-8f, dy));
            float iz = 1.f / dz;
            int4 n0 = c_bvh4[0], n1 = c_bvh4[1], n2 = c_bvh4[2], n3 = c_bvh4[3];
            const __half* bx = (const __half*)&n0, *by = (const __half*)&n1, *bz = (const __half*)&n2;
            const int* ch = (const int*)&n3;
            for (int c = 0; c < 4; c++) {
                if (ch[c] == -1) continue;
                float t1x = (__half2float(bx[c]) - ox) * ix, t2x = (__half2float(bx[4+c]) - ox) * ix;
                float t1y = (__half2float(by[c]) - oy) * iy, t2y = (__half2float(by[4+c]) - oy) * iy;
                float t1z = (__half2float(bz[c]) - oz) * iz, t2z = (__half2float(bz[4+c]) - oz) * iz;
                float tN = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
                float tF = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));
                if (tN <= tF && tF > 0.f) { anyHit = true; break; }
            }
        }
        unsigned mask = __ballot_sync(0xFFFFFFFF, anyHit);
        int warpHits = __popc(mask);
        int warpBase;
        if (lane == 0 && warpHits > 0) warpBase = atomicAdd(d_numSurvivors, warpHits);
        warpBase = __shfl_sync(0xFFFFFFFF, warpBase, 0);
        if (anyHit) {
            int myIdx = __popc(mask & ((1u << lane) - 1));
            d_survivors[warpBase + myIdx] = ri;
        }
    }
}

// ======================== Phase 2: Dense Primary Traversal (CSWAP) ========================
__global__ void __launch_bounds__(256, 5) tracePrimaryDense(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const int* __restrict__ d_survivors, const int* __restrict__ d_numSurvivors,
    Hit* __restrict__ hits, int side,
    float camOx, float camOy, float camOz)
{
    __shared__ int s_numS;
    if (threadIdx.x == 0) s_numS = *d_numSurvivors;
    __syncthreads();
    int numS = s_numS;
    const unsigned lane = threadIdx.x & 31;
    while (true) {
        int bs; if (lane == 0) bs = atomicAdd(&g_rayCounter2, 32);
        bs = __shfl_sync(0xFFFFFFFF, bs, 0);
        if (bs >= numS) break;
        int localIdx = bs + lane;
        int ri = (localIdx < numS) ? d_survivors[localIdx] : -1;
        bool alive = (ri >= 0);
        float ox=0,oy=0,oz=0,dx=0,dy=0,dz=1,ix=0,iy=0,iz=1;
        if (alive) {
            int px = ri % side, py = ri / side;
            float u = (px + 0.5f) / side * 2.f - 1.f;
            float v = (py + 0.5f) / side * 2.f - 1.f;
            float rlen = rsqrtf(u*u + v*v + 1.f);
            ox = camOx; oy = camOy; oz = camOz;
            dx = u*rlen; dy = v*rlen; dz = rlen;
            ix = 1.f/(fabsf(dx)>1e-8f?dx:copysignf(1e-8f,dx));
            iy = 1.f/(fabsf(dy)>1e-8f?dy:copysignf(1e-8f,dy));
            iz = 1.f/dz;
        }
        float tHit = 1e30f; int hitTri = -1; float hitU=0, hitV=0;
        int stk[STACK_DEPTH]; int sp = 0;
        if (alive) stk[sp++] = 0;
        while (sp > 0 && alive) {
            int ni = stk[--sp];
            if (ni < 0) {
                int enc = -(ni+2); int ts = enc>>3, tc = (enc&7)+1;
                for (int t=0; t<tc; t++) {
                    int ti = ts+t;
                    float e1x=__ldg(&tv1x[ti])-__ldg(&tv0x[ti]),e1y=__ldg(&tv1y[ti])-__ldg(&tv0y[ti]),e1z=__ldg(&tv1z[ti])-__ldg(&tv0z[ti]);
                    float e2x=__ldg(&tv2x[ti])-__ldg(&tv0x[ti]),e2y=__ldg(&tv2y[ti])-__ldg(&tv0y[ti]),e2z=__ldg(&tv2z[ti])-__ldg(&tv0z[ti]);
                    float ppx=dy*e2z-dz*e2y,ppy=dz*e2x-dx*e2z,ppz=dx*e2y-dy*e2x;
                    float det=e1x*ppx+e1y*ppy+e1z*ppz;
                    if (fabsf(det)<1e-12f) continue;
                    float inv=1.f/det;
                    float tx=ox-__ldg(&tv0x[ti]),ty=oy-__ldg(&tv0y[ti]),tz=oz-__ldg(&tv0z[ti]);
                    float uu=inv*(tx*ppx+ty*ppy+tz*ppz); if(uu<0.f||uu>1.f) continue;
                    float qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x;
                    float vv=inv*(dx*qx+dy*qy+dz*qz); if(vv<0.f||uu+vv>1.f) continue;
                    float tt=inv*(e2x*qx+e2y*qy+e2z*qz);
                    if (tt>0.f&&tt<tHit){tHit=tt;hitTri=ti;hitU=uu;hitV=vv;}
                }
                continue;
            }
            int4 n0,n1,n2,n3;
            loadBVH4Node(bvh,ni,n0,n1,n2,n3);
            const __half* bx=(const __half*)&n0,*by=(const __half*)&n1,*bz=(const __half*)&n2;
            const int* ch=(const int*)&n3;
            float dist[4]; int child[4];
            for (int c=0;c<4;c++) {
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
        if(alive){hits[ri].t=tHit;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;}
    }
}

// ======================== Diffuse Megakernel (sorted or unsorted) ========================
// Takes an indirection array: thread i traces ray indices[i].
// If indices==NULL, thread i traces ray i directly.
__global__ void __launch_bounds__(256, 5) traceDiffuseSorted(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const float* __restrict__ rox, const float* __restrict__ roy, const float* __restrict__ roz,
    const float* __restrict__ rdx, const float* __restrict__ rdy, const float* __restrict__ rdz,
    Hit* __restrict__ hits, int numRays,
    const int* __restrict__ indices)  // NULL = direct, otherwise sorted indirection
{
    const unsigned lane = threadIdx.x & 31;
    while (true) {
        int bs; if(lane==0) bs=atomicAdd(&g_rayCounter,32);
        bs=__shfl_sync(0xFFFFFFFF,bs,0);
        if(bs>=numRays) break;
        int si = bs+lane;
        int ri = (si < numRays) ? (indices ? indices[si] : si) : -1;
        bool alive = (ri >= 0);
        float ox=0,oy=0,oz=0,dx=0,dy=0,dz=1,ix=0,iy=0,iz=1;
        if(alive){
            ox=rox[ri];oy=roy[ri];oz=roz[ri];
            dx=rdx[ri];dy=rdy[ri];dz=rdz[ri];
            ix=1.f/(fabsf(dx)>1e-8f?dx:copysignf(1e-8f,dx));
            iy=1.f/(fabsf(dy)>1e-8f?dy:copysignf(1e-8f,dy));
            iz=1.f/(fabsf(dz)>1e-8f?dz:copysignf(1e-8f,dz));
        }
        float tHit=1e30f;int hitTri=-1;float hitU=0,hitV=0;
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
                    if(tt>0.f&&tt<tHit){tHit=tt;hitTri=ti;hitU=uu;hitV=vv;}
                }
                continue;
            }
            int4 n0,n1,n2,n3;
            loadBVH4Node(bvh,ni,n0,n1,n2,n3);
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
        if(alive){hits[ri].t=tHit;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;}
    }
}

// ======================== Init hits ========================
__global__ void initHitsMiss(Hit* hits, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { hits[i].t = 1e30f; hits[i].tri = -1; hits[i].u = 0; hits[i].v = 0; }
}

// ======================== Scene Generation ========================
static void genScene(Tri* tris, int nT, float sc) {
    srand(42);
    for (int i = 0; i < nT; i++) {
        float cx = ((float)rand()/RAND_MAX - 0.5f) * sc;
        float cy = ((float)rand()/RAND_MAX - 0.5f) * sc;
        float cz = ((float)rand()/RAND_MAX - 0.5f) * sc;
        float sz = sc * 0.005f + ((float)rand()/RAND_MAX) * sc * 0.01f;
        tris[i].v0 = {cx-sz, cy-sz, cz}; tris[i].v1 = {cx+sz, cy-sz, cz+sz}; tris[i].v2 = {cx, cy+sz, cz-sz};
    }
}

static void genDiffuseRays(float* ox, float* oy, float* oz,
    float* dx, float* dy, float* dz, int n, float sc) {
    srand(12345);
    for (int i = 0; i < n; i++) {
        ox[i] = ((float)rand()/RAND_MAX - 0.5f) * sc;
        oy[i] = ((float)rand()/RAND_MAX - 0.5f) * sc;
        oz[i] = ((float)rand()/RAND_MAX - 0.5f) * sc;
        float th = ((float)rand()/RAND_MAX) * 6.2831853f;
        float cp = ((float)rand()/RAND_MAX) * 2.f - 1.f;
        float sp = sqrtf(1.f - cp*cp);
        dx[i] = sp * cosf(th); dy[i] = sp * sinf(th); dz[i] = cp;
    }
}

// ======================== Main ========================
int main() {
    printf("══════════════════════════════════════════════════════════════════════════\n");
    printf("  V29 — Morton-Sorted Diffuse + Two-Phase Primary\n");
    printf("  Primary: broadphase+dense CSWAP | Diffuse: Morton/Octant sorted mega\n");
    printf("══════════════════════════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("  GPU: %s | SMs: %d | L2: %dKB\n\n", prop.name, prop.multiProcessorCount,
        (int)(prop.l2CacheSize / 1024));
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    int triCounts[] = {99000, 500000, 1000000};
    int nScenes = 3;

    for (int si = 0; si < nScenes; si++) {
        int NTRI = triCounts[si];
        printf("  ━━━ %dK tris ━━━\n", (NTRI + 500) / 1000);

        Tri* h_tris = (Tri*)malloc(NTRI * sizeof(Tri));
        genScene(h_tris, NTRI, 10.f);

        printf("  Building BVH...");
        BVHBuild b2; b2.build(h_tris, NTRI);
        int maxN4 = (int)b2.nodes.size() * 2;
        BVH4Node* h_b4 = (BVH4Node*)calloc(maxN4, sizeof(BVH4Node));
        int n4 = 0; collapseToB4(b2, 0, h_b4, n4, h_tris);
        printf(" %d nodes (%.1f KB)\n", n4, n4 * 64.f / 1024.f);

        int4* d_bvh4; cudaMalloc(&d_bvh4, n4 * sizeof(BVH4Node));
        cudaMemcpy(d_bvh4, h_b4, n4 * sizeof(BVH4Node), cudaMemcpyHostToDevice);
        int cN = n4 > CONST_BVH4 ? CONST_BVH4 : n4;
        cudaMemcpyToSymbol(c_bvh4, h_b4, cN * sizeof(BVH4Node));
        cudaMemcpyToSymbol(c_bvh4N, &cN, sizeof(int));

        Tri* ord = b2.ordered.data(); int nOT = (int)b2.ordered.size();
        float *h_tv[9], *d_tv[9];
        for (int j=0;j<9;j++){h_tv[j]=(float*)malloc(nOT*4);cudaMalloc(&d_tv[j],nOT*4);}
        for (int i=0;i<nOT;i++){
            h_tv[0][i]=ord[i].v0.x;h_tv[1][i]=ord[i].v0.y;h_tv[2][i]=ord[i].v0.z;
            h_tv[3][i]=ord[i].v1.x;h_tv[4][i]=ord[i].v1.y;h_tv[5][i]=ord[i].v1.z;
            h_tv[6][i]=ord[i].v2.x;h_tv[7][i]=ord[i].v2.y;h_tv[8][i]=ord[i].v2.z;
        }
        for(int j=0;j<9;j++)cudaMemcpy(d_tv[j],h_tv[j],nOT*4,cudaMemcpyHostToDevice);

        Hit* d_hits; cudaMalloc(&d_hits, NRAYS * sizeof(Hit));
        Hit* h_hits = (Hit*)malloc(NRAYS * sizeof(Hit));
        int* d_survivors; cudaMalloc(&d_survivors, NRAYS * sizeof(int));
        int* d_numSurvivors; cudaMalloc(&d_numSurvivors, sizeof(int));
        int side = (int)sqrtf((float)NRAYS);

        // ──── PRIMARY: TWO-PHASE ────
        {
            unsigned int zero = 0;
            initHitsMiss<<<(NRAYS+255)/256,256>>>(d_hits,NRAYS);
            cudaMemset(d_numSurvivors,0,4);
            cudaMemcpyToSymbol(g_rayCounter,&zero,4);
            broadphaseRoot<<<640,256>>>(d_survivors,d_numSurvivors,NRAYS,side,0.f,0.f,-20.f);
            cudaDeviceSynchronize();
            cudaMemcpyToSymbol(g_rayCounter2,&zero,4);
            tracePrimaryDense<<<400,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                d_survivors,d_numSurvivors,d_hits,side,0.f,0.f,-20.f);
            cudaDeviceSynchronize();
            int h_ns;cudaMemcpy(&h_ns,d_numSurvivors,4,cudaMemcpyDeviceToHost);

            cudaEvent_t t0,t1,t2;cudaEventCreate(&t0);cudaEventCreate(&t1);cudaEventCreate(&t2);
            float bestTotal=1e30f,bestP1=0,bestP2=0;
            for(int r=0;r<3;r++){
                initHitsMiss<<<(NRAYS+255)/256,256>>>(d_hits,NRAYS);
                cudaMemset(d_numSurvivors,0,4);
                cudaMemcpyToSymbol(g_rayCounter,&zero,4);
                cudaMemcpyToSymbol(g_rayCounter2,&zero,4);
                cudaEventRecord(t0);
                broadphaseRoot<<<640,256>>>(d_survivors,d_numSurvivors,NRAYS,side,0.f,0.f,-20.f);
                cudaEventRecord(t1);
                tracePrimaryDense<<<400,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                    d_survivors,d_numSurvivors,d_hits,side,0.f,0.f,-20.f);
                cudaEventRecord(t2);cudaEventSynchronize(t2);
                float msP1,msP2;
                cudaEventElapsedTime(&msP1,t0,t1);cudaEventElapsedTime(&msP2,t1,t2);
                float msT=msP1+msP2;
                if(msT<bestTotal){bestTotal=msT;bestP1=msP1;bestP2=msP2;}
            }
            cudaMemcpy(h_hits,d_hits,NRAYS*sizeof(Hit),cudaMemcpyDeviceToHost);
            int hc=0;for(int i=0;i<NRAYS;i++)if(h_hits[i].tri>=0)hc++;
            printf("  PRIMARY 2-PHASE:   %7.0f MR/s  hit:%5.1f%%  surv:%dk  P1:%.2f P2:%.2fms\n",
                (float)NRAYS/bestTotal/1000.f,100.f*hc/NRAYS,(h_ns+500)/1000,bestP1,bestP2);
            cudaEventDestroy(t0);cudaEventDestroy(t1);cudaEventDestroy(t2);
        }

        // ──── DIFFUSE: Generate rays ────
        float *h_rox=(float*)malloc(NRAYS*4),*h_roy=(float*)malloc(NRAYS*4),*h_roz=(float*)malloc(NRAYS*4);
        float *h_rdx=(float*)malloc(NRAYS*4),*h_rdy=(float*)malloc(NRAYS*4),*h_rdz=(float*)malloc(NRAYS*4);
        genDiffuseRays(h_rox,h_roy,h_roz,h_rdx,h_rdy,h_rdz,NRAYS,10.f);
        float *d_rox,*d_roy,*d_roz,*d_rdx,*d_rdy,*d_rdz;
        cudaMalloc(&d_rox,NRAYS*4);cudaMalloc(&d_roy,NRAYS*4);cudaMalloc(&d_roz,NRAYS*4);
        cudaMalloc(&d_rdx,NRAYS*4);cudaMalloc(&d_rdy,NRAYS*4);cudaMalloc(&d_rdz,NRAYS*4);
        cudaMemcpy(d_rox,h_rox,NRAYS*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_roy,h_roy,NRAYS*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_roz,h_roz,NRAYS*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_rdx,h_rdx,NRAYS*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_rdy,h_rdy,NRAYS*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_rdz,h_rdz,NRAYS*4,cudaMemcpyHostToDevice);

        // Sort key buffers
        unsigned int* d_keys; cudaMalloc(&d_keys, NRAYS*sizeof(unsigned int));
        int* d_indices; cudaMalloc(&d_indices, NRAYS*sizeof(int));
        int* d_indices2; cudaMalloc(&d_indices2, NRAYS*sizeof(int)); // for octant sort
        unsigned int* d_keys2; cudaMalloc(&d_keys2, NRAYS*sizeof(unsigned int));

        float sceneMin = -5.f, sceneMax = 5.f;
        float sceneInvSize = 1.f / (sceneMax - sceneMin);

        // ──── DIFFUSE: Unsorted Megakernel (baseline) ────
        {
            unsigned int zero=0;
            initHitsMiss<<<(NRAYS+255)/256,256>>>(d_hits,NRAYS);
            cudaMemcpyToSymbol(g_rayCounter,&zero,4);
            traceDiffuseSorted<<<320,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,d_hits,NRAYS,nullptr);
            cudaDeviceSynchronize();

            cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
            float best=1e30f;
            for(int r=0;r<3;r++){
                initHitsMiss<<<(NRAYS+255)/256,256>>>(d_hits,NRAYS);
                cudaMemcpyToSymbol(g_rayCounter,&zero,4);
                cudaEventRecord(t0);
                traceDiffuseSorted<<<320,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                    d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,d_hits,NRAYS,nullptr);
                cudaEventRecord(t1);cudaEventSynchronize(t1);
                float ms;cudaEventElapsedTime(&ms,t0,t1);if(ms<best)best=ms;
            }
            cudaMemcpy(h_hits,d_hits,NRAYS*sizeof(Hit),cudaMemcpyDeviceToHost);
            int hc=0;for(int i=0;i<NRAYS;i++)if(h_hits[i].tri>=0)hc++;
            printf("  DIFF UNSORTED:     %7.0f MR/s  hit:%5.1f%%  (%.2fms)\n",
                (float)NRAYS/best/1000.f,100.f*hc/NRAYS,best);
            cudaEventDestroy(t0);cudaEventDestroy(t1);
        }

        // ──── DIFFUSE: Morton-sorted ────
        {
            unsigned int zero=0;
            // Compute Morton keys
            computeMortonKeys<<<(NRAYS+255)/256,256>>>(d_rox,d_roy,d_roz,d_keys,d_indices,NRAYS,sceneMin,sceneInvSize);
            // Sort by key
            thrust::sort_by_key(thrust::device_ptr<unsigned int>(d_keys),
                thrust::device_ptr<unsigned int>(d_keys+NRAYS),
                thrust::device_ptr<int>(d_indices));
            cudaDeviceSynchronize();

            // Time sort separately
            cudaEvent_t ts0,ts1;cudaEventCreate(&ts0);cudaEventCreate(&ts1);
            cudaEventRecord(ts0);
            computeMortonKeys<<<(NRAYS+255)/256,256>>>(d_rox,d_roy,d_roz,d_keys,d_indices,NRAYS,sceneMin,sceneInvSize);
            thrust::sort_by_key(thrust::device_ptr<unsigned int>(d_keys),
                thrust::device_ptr<unsigned int>(d_keys+NRAYS),
                thrust::device_ptr<int>(d_indices));
            cudaEventRecord(ts1);cudaEventSynchronize(ts1);
            float sortMs;cudaEventElapsedTime(&sortMs,ts0,ts1);

            // Trace with sorted indices
            cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
            float best=1e30f;
            for(int r=0;r<3;r++){
                initHitsMiss<<<(NRAYS+255)/256,256>>>(d_hits,NRAYS);
                cudaMemcpyToSymbol(g_rayCounter,&zero,4);
                cudaEventRecord(t0);
                traceDiffuseSorted<<<320,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                    d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,d_hits,NRAYS,d_indices);
                cudaEventRecord(t1);cudaEventSynchronize(t1);
                float ms;cudaEventElapsedTime(&ms,t0,t1);if(ms<best)best=ms;
            }
            cudaMemcpy(h_hits,d_hits,NRAYS*sizeof(Hit),cudaMemcpyDeviceToHost);
            int hc=0;for(int i=0;i<NRAYS;i++)if(h_hits[i].tri>=0)hc++;
            printf("  DIFF MORTON:       %7.0f MR/s  hit:%5.1f%%  sort:%.2fms trace:%.2fms total:%.2fms → %0.f MR/s\n",
                (float)NRAYS/best/1000.f,100.f*hc/NRAYS,sortMs,best,sortMs+best,
                (float)NRAYS/(sortMs+best)/1000.f);
            cudaEventDestroy(t0);cudaEventDestroy(t1);
            cudaEventDestroy(ts0);cudaEventDestroy(ts1);

            // Block count sweep for Morton-indirect
            printf("  MORT blk sweep:   ");
            int blkCounts[] = {160, 320, 400, 640, 960};
            for (int bi = 0; bi < 5; bi++) {
                int blk = blkCounts[bi];
                float best2=1e30f;
                for(int r=0;r<3;r++){
                    initHitsMiss<<<(NRAYS+255)/256,256>>>(d_hits,NRAYS);
                    cudaMemcpyToSymbol(g_rayCounter,&zero,4);
                    cudaEvent_t ta,tb;cudaEventCreate(&ta);cudaEventCreate(&tb);
                    cudaEventRecord(ta);
                    traceDiffuseSorted<<<blk,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                        d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,d_hits,NRAYS,d_indices);
                    cudaEventRecord(tb);cudaEventSynchronize(tb);
                    float ms;cudaEventElapsedTime(&ms,ta,tb);if(ms<best2)best2=ms;
                    cudaEventDestroy(ta);cudaEventDestroy(tb);
                }
                printf(" %d→%.0f", blk, (float)NRAYS/best2/1000.f);
            }
            printf(" MR/s\n");
        }

        // ──── DIFFUSE: Morton-sorted + physically gathered (coalesced access) ────
        {
            unsigned int zero=0;
            // Allocate sorted ray buffers
            float *d_sox,*d_soy,*d_soz,*d_sdx,*d_sdy,*d_sdz;
            cudaMalloc(&d_sox,NRAYS*4);cudaMalloc(&d_soy,NRAYS*4);cudaMalloc(&d_soz,NRAYS*4);
            cudaMalloc(&d_sdx,NRAYS*4);cudaMalloc(&d_sdy,NRAYS*4);cudaMalloc(&d_sdz,NRAYS*4);
            Hit* d_shits; cudaMalloc(&d_shits, NRAYS*sizeof(Hit));

            // Compute keys, sort, gather
            computeMortonKeys<<<(NRAYS+255)/256,256>>>(d_rox,d_roy,d_roz,d_keys,d_indices,NRAYS,sceneMin,sceneInvSize);
            thrust::sort_by_key(thrust::device_ptr<unsigned int>(d_keys),
                thrust::device_ptr<unsigned int>(d_keys+NRAYS),
                thrust::device_ptr<int>(d_indices));
            gatherRays<<<(NRAYS+255)/256,256>>>(d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,
                d_sox,d_soy,d_soz,d_sdx,d_sdy,d_sdz,d_indices,NRAYS);
            cudaDeviceSynchronize();

            // Time: sort + gather + trace + scatter
            cudaEvent_t ts0,ts1;cudaEventCreate(&ts0);cudaEventCreate(&ts1);
            cudaEventRecord(ts0);
            computeMortonKeys<<<(NRAYS+255)/256,256>>>(d_rox,d_roy,d_roz,d_keys,d_indices,NRAYS,sceneMin,sceneInvSize);
            thrust::sort_by_key(thrust::device_ptr<unsigned int>(d_keys),
                thrust::device_ptr<unsigned int>(d_keys+NRAYS),
                thrust::device_ptr<int>(d_indices));
            gatherRays<<<(NRAYS+255)/256,256>>>(d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,
                d_sox,d_soy,d_soz,d_sdx,d_sdy,d_sdz,d_indices,NRAYS);
            cudaEventRecord(ts1);cudaEventSynchronize(ts1);
            float prepMs;cudaEventElapsedTime(&prepMs,ts0,ts1);

            // Trace physically sorted rays (no indirection - nullptr)
            cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
            float best=1e30f;
            for(int r=0;r<3;r++){
                initHitsMiss<<<(NRAYS+255)/256,256>>>(d_shits,NRAYS);
                cudaMemcpyToSymbol(g_rayCounter,&zero,4);
                cudaEventRecord(t0);
                traceDiffuseSorted<<<320,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                    d_sox,d_soy,d_soz,d_sdx,d_sdy,d_sdz,d_shits,NRAYS,nullptr);
                cudaEventRecord(t1);cudaEventSynchronize(t1);
                float ms;cudaEventElapsedTime(&ms,t0,t1);if(ms<best)best=ms;
            }

            // Scatter hits back to original order and verify
            scatterHits<<<(NRAYS+255)/256,256>>>(d_shits,d_hits,d_indices,NRAYS);
            cudaMemcpy(h_hits,d_hits,NRAYS*sizeof(Hit),cudaMemcpyDeviceToHost);
            int hc=0;for(int i=0;i<NRAYS;i++)if(h_hits[i].tri>=0)hc++;
            printf("  DIFF MORT+GATHER:  %7.0f MR/s  hit:%5.1f%%  prep:%.2fms trace:%.2fms total:%.2fms → %0.f MR/s\n",
                (float)NRAYS/best/1000.f,100.f*hc/NRAYS,prepMs,best,prepMs+best,
                (float)NRAYS/(prepMs+best)/1000.f);

            cudaFree(d_sox);cudaFree(d_soy);cudaFree(d_soz);
            cudaFree(d_sdx);cudaFree(d_sdy);cudaFree(d_sdz);
            cudaFree(d_shits);
            cudaEventDestroy(t0);cudaEventDestroy(t1);
            cudaEventDestroy(ts0);cudaEventDestroy(ts1);
        }

        // ──── DIFFUSE: Octant+Morton sorted ────
        {
            unsigned int zero=0;
            // Compute octant+morton keys
            computeSortKeys<<<(NRAYS+255)/256,256>>>(d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,
                d_keys2,d_indices2,NRAYS,sceneMin,sceneInvSize);
            thrust::sort_by_key(thrust::device_ptr<unsigned int>(d_keys2),
                thrust::device_ptr<unsigned int>(d_keys2+NRAYS),
                thrust::device_ptr<int>(d_indices2));
            cudaDeviceSynchronize();

            // Time sort
            cudaEvent_t ts0,ts1;cudaEventCreate(&ts0);cudaEventCreate(&ts1);
            cudaEventRecord(ts0);
            computeSortKeys<<<(NRAYS+255)/256,256>>>(d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,
                d_keys2,d_indices2,NRAYS,sceneMin,sceneInvSize);
            thrust::sort_by_key(thrust::device_ptr<unsigned int>(d_keys2),
                thrust::device_ptr<unsigned int>(d_keys2+NRAYS),
                thrust::device_ptr<int>(d_indices2));
            cudaEventRecord(ts1);cudaEventSynchronize(ts1);
            float sortMs;cudaEventElapsedTime(&sortMs,ts0,ts1);

            cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
            float best=1e30f;
            for(int r=0;r<3;r++){
                initHitsMiss<<<(NRAYS+255)/256,256>>>(d_hits,NRAYS);
                cudaMemcpyToSymbol(g_rayCounter,&zero,4);
                cudaEventRecord(t0);
                traceDiffuseSorted<<<320,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                    d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,d_hits,NRAYS,d_indices2);
                cudaEventRecord(t1);cudaEventSynchronize(t1);
                float ms;cudaEventElapsedTime(&ms,t0,t1);if(ms<best)best=ms;
            }
            cudaMemcpy(h_hits,d_hits,NRAYS*sizeof(Hit),cudaMemcpyDeviceToHost);
            int hc=0;for(int i=0;i<NRAYS;i++)if(h_hits[i].tri>=0)hc++;
            printf("  DIFF OCT+MORT:     %7.0f MR/s  hit:%5.1f%%  sort:%.2fms trace:%.2fms total:%.2fms → %0.f MR/s\n",
                (float)NRAYS/best/1000.f,100.f*hc/NRAYS,sortMs,best,sortMs+best,
                (float)NRAYS/(sortMs+best)/1000.f);
            cudaEventDestroy(t0);cudaEventDestroy(t1);
            cudaEventDestroy(ts0);cudaEventDestroy(ts1);
        }

        // Cleanup
        cudaFree(d_keys);cudaFree(d_indices);cudaFree(d_keys2);cudaFree(d_indices2);
        cudaFree(d_rox);cudaFree(d_roy);cudaFree(d_roz);
        cudaFree(d_rdx);cudaFree(d_rdy);cudaFree(d_rdz);
        free(h_rox);free(h_roy);free(h_roz);
        free(h_rdx);free(h_rdy);free(h_rdz);
        cudaFree(d_survivors);cudaFree(d_numSurvivors);
        for(int j=0;j<9;j++){cudaFree(d_tv[j]);free(h_tv[j]);}
        cudaFree(d_bvh4);cudaFree(d_hits);free(h_hits);free(h_b4);free(h_tris);
        printf("\n");
    }
    return 0;
}
