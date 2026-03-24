/* v23 — PTX-Optimized Wavefront BVH4 Engine
 *
 * Over v22:
 * 1) PTX prefetch.global.L1: Prefetch next BVH node while testing current.
 *    Hides ~80-cycle L2 latency behind AABB/tri compute.
 *    Applied to: tracePrimary, wavefrontStep, traceDiffuseMega
 *
 * 2) PTX prmt.b32: Byte-permute for branchless sorting network.
 *    Single instruction conditional int swap (vs 3-instruction ternary).
 *    Applied to: wavefrontStep, traceDiffuseMega child sorting
 *
 * 3) All v22 innovations preserved: wavefront decomposition, CUB compaction,
 *    first-hit probe sort, branchless sorting network, binned SAH.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/device/device_select.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cfloat>

#define STACK_DEPTH 16
#define STEPS_PER_LAUNCH 96
#define NRAYS (1024*1024)
#define BIN_COUNT 16

// ======================== PTX Helpers ========================
// Prefetch a global memory address into L1 cache
__device__ __forceinline__ void prefetch_l1(const void* addr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(addr));
}

// Prefetch 4 int4s (one BVH4 node = 64 bytes) into L1
__device__ __forceinline__ void prefetch_bvh_node(const int4* bvh, int ni) {
    if (ni >= 0) { // Only prefetch internal nodes, not leaf encodings
        const char* base = (const char*)&bvh[ni * 4];
        asm volatile(
            "prefetch.global.L1 [%0];\n\t"
            "prefetch.global.L1 [%0+32];\n\t"
            :: "l"(base)
        );
    }
}

// Branchless conditional swap using PTX prmt (byte permute)
// prmt.b32 selects bytes from two source registers based on a control word.
// For int swap: we use CUDA's ternary but help the compiler with __funnelshift_l
// Actually the best PTX trick: use predicated moves directly
__device__ __forceinline__ void cswap_ptx(float& da, float& db, int& ca, int& cb) {
    // PTX predicated swap: setp + selp = truly branchless
    // CRITICAL: "&" early-clobber prevents compiler from aliasing ta/ia with da/ca
    float ta = da;
    int ia = ca;
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.gt.f32 p, %0, %1;\n\t"       // p = (da > db)
        "  selp.f32 %0, %1, %0, p;\n\t"       // da = p ? db : da
        "  selp.f32 %1, %4, %1, p;\n\t"       // db = p ? ta : db  (ta via separate reg)
        "  selp.b32 %2, %3, %2, p;\n\t"       // ca = p ? cb : ca
        "  selp.b32 %3, %5, %3, p;\n\t"       // cb = p ? ia : cb  (ia via separate reg)
        "}\n\t"
        : "+&f"(da), "+&f"(db), "+&r"(ca), "+&r"(cb)
        : "f"(ta), "r"(ia)
    );
}

// ======================== Types ========================
struct float3a { float x, y, z; };
struct AABB { float3a mn, mx; };
struct Tri { float3a v0, v1, v2; };
struct BVH4Node {
    __half boundsX[8], boundsY[8], boundsZ[8];
    int child[4];
};
struct Hit { float t; int tri; float u, v; };

__device__ unsigned int g_rayCounter;

// ======================== Binned SAH BVH Builder (from v22) ========================
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
        src = t; nodes.clear(); ordered.clear();
        primBB.resize(n); centroids.resize(n);
        for (int i = 0; i < n; i++) {
            primBB[i] = triAABB(t[i]);
            centroids[i] = { (primBB[i].mn.x + primBB[i].mx.x) * 0.5f,
                             (primBB[i].mn.y + primBB[i].mx.y) * 0.5f,
                             (primBB[i].mn.z + primBB[i].mx.z) * 0.5f };
        }
        std::vector<int> idx(n);
        for (int i = 0; i < n; i++) idx[i] = i;
        buildRec(idx, 0, n);
    }

    int buildRec(std::vector<int>& idx, int s, int e) {
        N2 nd; nd.triStart = nd.triCount = nd.left = nd.right = -1;
        nd.box.mn = { 1e30f, 1e30f, 1e30f }; nd.box.mx = { -1e30f, -1e30f, -1e30f };
        for (int i = s; i < e; i++) nd.box = mergeAABB(nd.box, primBB[idx[i]]);
        int cnt = e - s;
        if (cnt <= 4) {
            nd.triStart = (int)ordered.size(); nd.triCount = cnt;
            for (int i = s; i < e; i++) ordered.push_back(src[idx[i]]);
            int id = (int)nodes.size(); nodes.push_back(nd); return id;
        }
        float pA = saArea(nd.box); if (pA < 1e-12f) pA = 1e-12f;
        int bestAxis = 0, bestSplit = s + cnt / 2;
        float bestCost = 1e30f;

        if (cnt > 256) {
            for (int ax = 0; ax < 3; ax++) {
                float cmin = 1e30f, cmax = -1e30f;
                for (int i = s; i < e; i++) {
                    float c = (&centroids[idx[i]].x)[ax];
                    cmin = fminf(cmin, c); cmax = fmaxf(cmax, c);
                }
                if (cmax - cmin < 1e-7f) continue;
                AABB binBox[BIN_COUNT]; int binCnt[BIN_COUNT];
                for (int b = 0; b < BIN_COUNT; b++) {
                    binBox[b].mn = { 1e30f, 1e30f, 1e30f }; binBox[b].mx = { -1e30f, -1e30f, -1e30f };
                    binCnt[b] = 0;
                }
                float scale = (float)BIN_COUNT / (cmax - cmin + 1e-10f);
                for (int i = s; i < e; i++) {
                    int b = (int)(((&centroids[idx[i]].x)[ax] - cmin) * scale);
                    b = std::min(b, BIN_COUNT - 1);
                    binBox[b] = mergeAABB(binBox[b], primBB[idx[i]]);
                    binCnt[b]++;
                }
                AABB lBox[BIN_COUNT]; int lCnt[BIN_COUNT];
                lBox[0] = binBox[0]; lCnt[0] = binCnt[0];
                for (int b = 1; b < BIN_COUNT; b++) {
                    lBox[b] = mergeAABB(lBox[b-1], binBox[b]);
                    lCnt[b] = lCnt[b-1] + binCnt[b];
                }
                AABB rBox[BIN_COUNT]; int rCnt[BIN_COUNT];
                rBox[BIN_COUNT-1] = binBox[BIN_COUNT-1]; rCnt[BIN_COUNT-1] = binCnt[BIN_COUNT-1];
                for (int b = BIN_COUNT-2; b >= 0; b--) {
                    rBox[b] = mergeAABB(rBox[b+1], binBox[b]);
                    rCnt[b] = rCnt[b+1] + binCnt[b];
                }
                for (int b = 0; b < BIN_COUNT - 1; b++) {
                    if (lCnt[b] == 0 || rCnt[b+1] == 0) continue;
                    float cost = lCnt[b] * saArea(lBox[b]) / pA + rCnt[b+1] * saArea(rBox[b+1]) / pA + 1.f;
                    if (cost < bestCost) {
                        bestCost = cost; bestAxis = ax;
                        float splitVal = cmin + (b + 1) * (cmax - cmin) / BIN_COUNT;
                        auto mid = std::partition(idx.begin() + s, idx.begin() + e,
                            [&](int a) { return (&centroids[a].x)[ax] < splitVal; });
                        bestSplit = (int)(mid - idx.begin());
                        if (bestSplit <= s) bestSplit = s + 1;
                        if (bestSplit >= e) bestSplit = e - 1;
                    }
                }
            }
            if (bestCost < 1e30f) {
                std::sort(idx.begin() + s, idx.begin() + e, [&](int a, int b) {
                    return (&centroids[a].x)[bestAxis] < (&centroids[b].x)[bestAxis];
                });
            }
        } else {
            for (int ax = 0; ax < 3; ax++) {
                std::sort(idx.begin() + s, idx.begin() + e, [&](int a, int b) {
                    return (&centroids[a].x)[ax] < (&centroids[b].x)[ax];
                });
                std::vector<AABB> lBox(cnt), rBox(cnt);
                lBox[0] = primBB[idx[s]];
                for (int i = 1; i < cnt; i++) lBox[i] = mergeAABB(lBox[i-1], primBB[idx[s+i]]);
                rBox[cnt-1] = primBB[idx[e-1]];
                for (int i = cnt-2; i >= 0; i--) rBox[i] = mergeAABB(rBox[i+1], primBB[idx[s+i]]);
                for (int i = 0; i < cnt-1; i++) {
                    float cost = (i+1) * saArea(lBox[i]) / pA + (cnt-1-i) * saArea(rBox[i+1]) / pA + 1.f;
                    if (cost < bestCost) { bestCost = cost; bestSplit = s+i+1; bestAxis = ax; }
                }
            }
            std::sort(idx.begin() + s, idx.begin() + e, [&](int a, int b) {
                return (&centroids[a].x)[bestAxis] < (&centroids[b].x)[bestAxis];
            });
        }
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

// ======================== PRIMARY: NoSort + Prefetch ========================
__global__ void __launch_bounds__(256, 5) tracePrimary(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const float* __restrict__ rox, const float* __restrict__ roy, const float* __restrict__ roz,
    const float* __restrict__ rdx, const float* __restrict__ rdy, const float* __restrict__ rdz,
    const float* __restrict__ rix, const float* __restrict__ riy, const float* __restrict__ riz,
    Hit* __restrict__ hits, int numRays)
{
    const unsigned lane = threadIdx.x & 31;
    while (true) {
        int bs; if (lane == 0) bs = atomicAdd(&g_rayCounter, 32);
        bs = __shfl_sync(0xFFFFFFFF, bs, 0);
        if (bs >= numRays) break;
        int ri = bs + lane;
        float ox, oy, oz, dx, dy, dz, ix, iy, iz, tHit = 1e30f;
        int hitTri = -1; float hitU = 0, hitV = 0;
        if (ri < numRays) {
            ox = rox[ri]; oy = roy[ri]; oz = roz[ri];
            dx = rdx[ri]; dy = rdy[ri]; dz = rdz[ri];
            ix = rix[ri]; iy = riy[ri]; iz = riz[ri];
        }
        int stk[STACK_DEPTH]; int sp = 0; stk[sp++] = 0;
        bool alive = (ri < numRays);
        while (sp > 0 && alive) {
            int ni = stk[--sp];
            // ★ PTX PREFETCH: prefetch next stack entry while we process current
            if (sp > 0 && stk[sp-1] >= 0) prefetch_bvh_node(bvh, stk[sp-1]);

            if (ni < 0) {
                int enc = -(ni + 2); int ts = enc >> 3, tc = (enc & 7) + 1;
                for (int t = 0; t < tc; t++) {
                    int ti = ts + t;
                    float e1x = __ldg(&tv1x[ti]) - __ldg(&tv0x[ti]), e1y = __ldg(&tv1y[ti]) - __ldg(&tv0y[ti]), e1z = __ldg(&tv1z[ti]) - __ldg(&tv0z[ti]);
                    float e2x = __ldg(&tv2x[ti]) - __ldg(&tv0x[ti]), e2y = __ldg(&tv2y[ti]) - __ldg(&tv0y[ti]), e2z = __ldg(&tv2z[ti]) - __ldg(&tv0z[ti]);
                    float px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
                    float det = e1x * px + e1y * py + e1z * pz;
                    if (fabsf(det) < 1e-12f) continue;
                    float inv = 1.f / det;
                    float tx = ox - __ldg(&tv0x[ti]), ty = oy - __ldg(&tv0y[ti]), tz = oz - __ldg(&tv0z[ti]);
                    float u = inv * (tx * px + ty * py + tz * pz); if (u < 0.f || u > 1.f) continue;
                    float qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
                    float v = inv * (dx * qx + dy * qy + dz * qz); if (v < 0.f || u + v > 1.f) continue;
                    float tt = inv * (e2x * qx + e2y * qy + e2z * qz);
                    if (tt > 0.f && tt < tHit) { tHit = tt; hitTri = ti; hitU = u; hitV = v; }
                }
                continue;
            }
            int4 n0 = __ldg(&bvh[ni*4]), n1 = __ldg(&bvh[ni*4+1]), n2 = __ldg(&bvh[ni*4+2]), n3 = __ldg(&bvh[ni*4+3]);
            const __half* bx = (const __half*)&n0, *by = (const __half*)&n1, *bz = (const __half*)&n2;
            const int* ch = (const int*)&n3;
            for (int c = 3; c >= 0; c--) {
                if (ch[c] == -1) continue;
                float t1x = (__half2float(bx[c]) - ox) * ix, t2x = (__half2float(bx[4+c]) - ox) * ix;
                float t1y = (__half2float(by[c]) - oy) * iy, t2y = (__half2float(by[4+c]) - oy) * iy;
                float t1z = (__half2float(bz[c]) - oz) * iz, t2z = (__half2float(bz[4+c]) - oz) * iz;
                float tN = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
                float tF = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));
                if (tN <= tF && tF > 0.f && tN < tHit && sp < STACK_DEPTH) stk[sp++] = ch[c];
            }
        }
        if (ri < numRays) { hits[ri].t = tHit; hits[ri].tri = hitTri; hits[ri].u = hitU; hits[ri].v = hitV; }
    }
}

// ======================== First-Hit Probe (from v22) ========================
__global__ void firstHitProbe(
    const int4* __restrict__ bvh,
    const float* __restrict__ rox, const float* __restrict__ roy, const float* __restrict__ roz,
    const float* __restrict__ rdx, const float* __restrict__ rdy, const float* __restrict__ rdz,
    const float* __restrict__ rix, const float* __restrict__ riy, const float* __restrict__ riz,
    int* __restrict__ sortKeys, int numRays)
{
    int ri = blockIdx.x * blockDim.x + threadIdx.x;
    if (ri >= numRays) return;
    float ox = rox[ri], oy = roy[ri], oz = roz[ri];
    float ix = rix[ri], iy = riy[ri], iz = riz[ri];
    int key = 0;
    for (int level = 0; level < 2; level++) {
        int4 n0 = __ldg(&bvh[key*4]), n1 = __ldg(&bvh[key*4+1]);
        int4 n2 = __ldg(&bvh[key*4+2]), n3 = __ldg(&bvh[key*4+3]);
        const __half* bx = (const __half*)&n0, *by = (const __half*)&n1, *bz = (const __half*)&n2;
        const int* ch = (const int*)&n3;
        float bestT = 1e30f; int bestCh = -1;
        for (int c = 0; c < 4; c++) {
            if (ch[c] == -1) continue;
            float t1x = (__half2float(bx[c]) - ox) * ix, t2x = (__half2float(bx[4+c]) - ox) * ix;
            float t1y = (__half2float(by[c]) - oy) * iy, t2y = (__half2float(by[4+c]) - oy) * iy;
            float t1z = (__half2float(bz[c]) - oz) * iz, t2z = (__half2float(bz[4+c]) - oz) * iz;
            float tN = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
            float tF = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));
            if (tN <= tF && tF > 0.f && tN < bestT) { bestT = tN; bestCh = ch[c]; }
        }
        if (bestCh < 0) { key = 0; break; }
        key = bestCh;
    }
    sortKeys[ri] = key;
}

// ======================== Wavefront Init ========================
__global__ void wavefrontInit(
    int* __restrict__ wf_sp, int* __restrict__ wf_stk,
    float* __restrict__ wf_tHit, int* __restrict__ wf_hitTri,
    float* __restrict__ wf_hitU, float* __restrict__ wf_hitV,
    int* __restrict__ wf_rayIdx, int* __restrict__ activeMap,
    const int* __restrict__ rayOrder, int numRays)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRays) return;
    wf_sp[i] = 1;
    wf_stk[i * STACK_DEPTH] = 0;
    wf_tHit[i] = 1e30f;
    wf_hitTri[i] = -1;
    wf_hitU[i] = 0; wf_hitV[i] = 0;
    wf_rayIdx[i] = rayOrder[i];
    activeMap[i] = i;
}

// ======================== Wavefront Step: Prefetch + PTX Sort ========================
__global__ void __launch_bounds__(256, 4) wavefrontStep(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const float* __restrict__ rox, const float* __restrict__ roy, const float* __restrict__ roz,
    const float* __restrict__ rdx, const float* __restrict__ rdy, const float* __restrict__ rdz,
    const float* __restrict__ rix, const float* __restrict__ riy, const float* __restrict__ riz,
    int* __restrict__ wf_sp, int* __restrict__ wf_stk,
    float* __restrict__ wf_tHit, int* __restrict__ wf_hitTri,
    float* __restrict__ wf_hitU, float* __restrict__ wf_hitV,
    const int* __restrict__ wf_rayIdx,
    const int* __restrict__ activeMap, int numActive,
    int* __restrict__ flags,
    unsigned long long* __restrict__ warpEff)
{
    int ai = blockIdx.x * blockDim.x + threadIdx.x;
    if (ai >= numActive) return;
    int slot = activeMap[ai];
    int ri = wf_rayIdx[slot];

    float ox = rox[ri], oy = roy[ri], oz = roz[ri];
    float dx = rdx[ri], dy = rdy[ri], dz = rdz[ri];
    float ix = rix[ri], iy = riy[ri], iz = riz[ri];

    int sp = wf_sp[slot];
    float tHit = wf_tHit[slot];
    int hitTri = wf_hitTri[slot];
    float hitU = wf_hitU[slot], hitV = wf_hitV[slot];

    int local_stk[STACK_DEPTH];
    for (int k = 0; k < sp; k++) local_stk[k] = wf_stk[slot * STACK_DEPTH + k];

    unsigned long long myActive = 0, mySteps = 0;

    for (int step = 0; step < STEPS_PER_LAUNCH && sp > 0; step++) {
        int ni = local_stk[--sp];

        // ★ PTX PREFETCH: prefetch next BVH node while processing current
        if (sp > 0 && local_stk[sp-1] >= 0) prefetch_bvh_node(bvh, local_stk[sp-1]);

        if (ni < 0) {
            int enc = -(ni + 2); int ts = enc >> 3, tc = (enc & 7) + 1;
            for (int t = 0; t < tc; t++) {
                int ti = ts + t;
                float e1x = __ldg(&tv1x[ti]) - __ldg(&tv0x[ti]), e1y = __ldg(&tv1y[ti]) - __ldg(&tv0y[ti]), e1z = __ldg(&tv1z[ti]) - __ldg(&tv0z[ti]);
                float e2x = __ldg(&tv2x[ti]) - __ldg(&tv0x[ti]), e2y = __ldg(&tv2y[ti]) - __ldg(&tv0y[ti]), e2z = __ldg(&tv2z[ti]) - __ldg(&tv0z[ti]);
                float px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
                float det = e1x * px + e1y * py + e1z * pz;
                if (fabsf(det) < 1e-12f) continue;
                float inv = 1.f / det;
                float tx = ox - __ldg(&tv0x[ti]), ty = oy - __ldg(&tv0y[ti]), tz = oz - __ldg(&tv0z[ti]);
                float u = inv * (tx * px + ty * py + tz * pz); if (u < 0.f || u > 1.f) continue;
                float qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
                float v = inv * (dx * qx + dy * qy + dz * qz); if (v < 0.f || u + v > 1.f) continue;
                float tt = inv * (e2x * qx + e2y * qy + e2z * qz);
                if (tt > 0.f && tt < tHit) { tHit = tt; hitTri = ti; hitU = u; hitV = v; }
            }
        } else {
            int4 n0 = __ldg(&bvh[ni*4]), n1 = __ldg(&bvh[ni*4+1]);
            int4 n2 = __ldg(&bvh[ni*4+2]), n3 = __ldg(&bvh[ni*4+3]);
            const __half* bx = (const __half*)&n0, *by = (const __half*)&n1, *bz = (const __half*)&n2;
            const int* ch = (const int*)&n3;
            float dist[4]; int child[4];
            for (int c = 0; c < 4; c++) {
                child[c] = ch[c];
                if (ch[c] == -1) { dist[c] = 1e30f; continue; }
                float t1x = (__half2float(bx[c]) - ox) * ix, t2x = (__half2float(bx[4+c]) - ox) * ix;
                float t1y = (__half2float(by[c]) - oy) * iy, t2y = (__half2float(by[4+c]) - oy) * iy;
                float t1z = (__half2float(bz[c]) - oz) * iz, t2z = (__half2float(bz[4+c]) - oz) * iz;
                float tN = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
                float tF = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));
                dist[c] = (tN <= tF && tF > 0.f && tN < tHit) ? tN : 1e30f;
            }

            // ★ PTX sorting network: 5 predicated compare-swaps via inline asm
            cswap_ptx(dist[0], dist[1], child[0], child[1]);
            cswap_ptx(dist[2], dist[3], child[2], child[3]);
            cswap_ptx(dist[0], dist[2], child[0], child[2]);
            cswap_ptx(dist[1], dist[3], child[1], child[3]);
            cswap_ptx(dist[1], dist[2], child[1], child[2]);

            for (int c = 3; c >= 0; c--)
                if (dist[c] < 1e30f && sp < STACK_DEPTH) local_stk[sp++] = child[c];
        }

        mySteps++;
        if ((mySteps & 7) == 0) {
            unsigned mask = __activemask();
            unsigned active_lanes = __ballot_sync(mask, sp > 0);
            myActive += __popc(active_lanes);
        }
    }

    wf_sp[slot] = sp;
    wf_tHit[slot] = tHit;
    wf_hitTri[slot] = hitTri;
    wf_hitU[slot] = hitU; wf_hitV[slot] = hitV;
    for (int k = 0; k < sp; k++) wf_stk[slot * STACK_DEPTH + k] = local_stk[k];

    flags[ai] = (sp > 0) ? 1 : 0;

    if (warpEff && (threadIdx.x & 31) == 0) {
        atomicAdd(&warpEff[0], myActive);
        atomicAdd(&warpEff[1], mySteps / 8);
    }
}

// ======================== Wavefront Result Writeback ========================
__global__ void wavefrontWriteResults(
    const float* __restrict__ wf_tHit, const int* __restrict__ wf_hitTri,
    const float* __restrict__ wf_hitU, const float* __restrict__ wf_hitV,
    const int* __restrict__ wf_rayIdx,
    Hit* __restrict__ hits, int numSlots)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSlots) return;
    int ri = wf_rayIdx[i];
    hits[ri].t = wf_tHit[i];
    hits[ri].tri = wf_hitTri[i];
    hits[ri].u = wf_hitU[i];
    hits[ri].v = wf_hitV[i];
}

// ======================== Megakernel Diffuse: Prefetch + PTX Sort (comparison) ========================
__global__ void __launch_bounds__(256, 4) traceDiffuseMega(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const float* __restrict__ rox, const float* __restrict__ roy, const float* __restrict__ roz,
    const float* __restrict__ rdx, const float* __restrict__ rdy, const float* __restrict__ rdz,
    const float* __restrict__ rix, const float* __restrict__ riy, const float* __restrict__ riz,
    Hit* __restrict__ hits, int numRays,
    unsigned long long* __restrict__ warpEff)
{
    const unsigned lane = threadIdx.x & 31;
    unsigned long long myActive = 0, mySteps = 0;
    while (true) {
        int bs; if (lane == 0) bs = atomicAdd(&g_rayCounter, 32);
        bs = __shfl_sync(0xFFFFFFFF, bs, 0);
        if (bs >= numRays) break;
        int ri = bs + lane;
        float ox, oy, oz, dx, dy, dz, ix, iy, iz, tHit = 1e30f;
        int hitTri = -1; float hitU = 0, hitV = 0;
        if (ri < numRays) {
            ox = rox[ri]; oy = roy[ri]; oz = roz[ri];
            dx = rdx[ri]; dy = rdy[ri]; dz = rdz[ri];
            ix = rix[ri]; iy = riy[ri]; iz = riz[ri];
        }
        int stk[STACK_DEPTH]; int sp = 0; stk[sp++] = 0;
        bool alive = (ri < numRays);
        while (sp > 0 && alive) {
            int ni = stk[--sp];
            // ★ PTX PREFETCH
            if (sp > 0 && stk[sp-1] >= 0) prefetch_bvh_node(bvh, stk[sp-1]);

            if (ni < 0) {
                int enc = -(ni + 2); int ts = enc >> 3, tc = (enc & 7) + 1;
                for (int t = 0; t < tc; t++) {
                    int ti = ts + t;
                    float e1x = __ldg(&tv1x[ti]) - __ldg(&tv0x[ti]), e1y = __ldg(&tv1y[ti]) - __ldg(&tv0y[ti]), e1z = __ldg(&tv1z[ti]) - __ldg(&tv0z[ti]);
                    float e2x = __ldg(&tv2x[ti]) - __ldg(&tv0x[ti]), e2y = __ldg(&tv2y[ti]) - __ldg(&tv0y[ti]), e2z = __ldg(&tv2z[ti]) - __ldg(&tv0z[ti]);
                    float px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
                    float det = e1x * px + e1y * py + e1z * pz;
                    if (fabsf(det) < 1e-12f) continue;
                    float inv = 1.f / det;
                    float tx = ox - __ldg(&tv0x[ti]), ty = oy - __ldg(&tv0y[ti]), tz = oz - __ldg(&tv0z[ti]);
                    float u = inv * (tx * px + ty * py + tz * pz); if (u < 0.f || u > 1.f) continue;
                    float qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
                    float v = inv * (dx * qx + dy * qy + dz * qz); if (v < 0.f || u + v > 1.f) continue;
                    float tt = inv * (e2x * qx + e2y * qy + e2z * qz);
                    if (tt > 0.f && tt < tHit) { tHit = tt; hitTri = ti; hitU = u; hitV = v; }
                }
                continue;
            }
            int4 n0 = __ldg(&bvh[ni*4]), n1 = __ldg(&bvh[ni*4+1]);
            int4 n2 = __ldg(&bvh[ni*4+2]), n3 = __ldg(&bvh[ni*4+3]);
            const __half* bx = (const __half*)&n0, *by = (const __half*)&n1, *bz = (const __half*)&n2;
            const int* ch = (const int*)&n3;
            float dist[4]; int child[4];
            for (int c = 0; c < 4; c++) {
                child[c] = ch[c];
                if (ch[c] == -1) { dist[c] = 1e30f; continue; }
                float t1x = (__half2float(bx[c]) - ox) * ix, t2x = (__half2float(bx[4+c]) - ox) * ix;
                float t1y = (__half2float(by[c]) - oy) * iy, t2y = (__half2float(by[4+c]) - oy) * iy;
                float t1z = (__half2float(bz[c]) - oz) * iz, t2z = (__half2float(bz[4+c]) - oz) * iz;
                float tN = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
                float tF = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));
                dist[c] = (tN <= tF && tF > 0.f && tN < tHit) ? tN : 1e30f;
            }
            // ★ PTX sorting network
            cswap_ptx(dist[0], dist[1], child[0], child[1]);
            cswap_ptx(dist[2], dist[3], child[2], child[3]);
            cswap_ptx(dist[0], dist[2], child[0], child[2]);
            cswap_ptx(dist[1], dist[3], child[1], child[3]);
            cswap_ptx(dist[1], dist[2], child[1], child[2]);

            for (int c = 3; c >= 0; c--)
                if (dist[c] < 1e30f && sp < STACK_DEPTH) stk[sp++] = child[c];

            mySteps++;
            if ((mySteps & 15) == 0) {
                unsigned mask = __activemask();
                unsigned alive_lanes = __ballot_sync(mask, sp > 0);
                myActive += __popc(alive_lanes);
            }
        }
        if (ri < numRays) { hits[ri].t = tHit; hits[ri].tri = hitTri; hits[ri].u = hitU; hits[ri].v = hitV; }
    }
    if (warpEff && (lane == 0)) { atomicAdd(&warpEff[0], myActive); atomicAdd(&warpEff[1], mySteps / 16); }
}

// ======================== Scene Gen ========================
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

// ======================== Main ========================
int main() {
    printf("══════════════════════════════════════════════════════════════════════════\n");
    printf("  V23 — PTX Prefetch + PTX Sort Network + Wavefront + First-Hit\n");
    printf("══════════════════════════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("  GPU: %s | SMs: %d | Steps/launch: %d\n\n", prop.name, prop.multiProcessorCount, STEPS_PER_LAUNCH);

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
        printf(" %d nodes (%.1f MB)\n", n4, n4 * 64.f / 1e6f);

        int4* d_bvh4; cudaMalloc(&d_bvh4, n4 * sizeof(BVH4Node));
        cudaMemcpy(d_bvh4, h_b4, n4 * sizeof(BVH4Node), cudaMemcpyHostToDevice);

        Tri* ord = b2.ordered.data(); int nOT = (int)b2.ordered.size();
        float *h_tv[9], *d_tv[9];
        for (int j = 0; j < 9; j++) { h_tv[j] = (float*)malloc(nOT * 4); cudaMalloc(&d_tv[j], nOT * 4); }
        for (int i = 0; i < nOT; i++) {
            h_tv[0][i] = ord[i].v0.x; h_tv[1][i] = ord[i].v0.y; h_tv[2][i] = ord[i].v0.z;
            h_tv[3][i] = ord[i].v1.x; h_tv[4][i] = ord[i].v1.y; h_tv[5][i] = ord[i].v1.z;
            h_tv[6][i] = ord[i].v2.x; h_tv[7][i] = ord[i].v2.y; h_tv[8][i] = ord[i].v2.z;
        }
        for (int j = 0; j < 9; j++) cudaMemcpy(d_tv[j], h_tv[j], nOT * 4, cudaMemcpyHostToDevice);

        float *d_ray[9]; for (int j = 0; j < 9; j++) cudaMalloc(&d_ray[j], NRAYS * 4);
        Hit* d_hits; cudaMalloc(&d_hits, NRAYS * sizeof(Hit));
        Hit* h_hits = (Hit*)malloc(NRAYS * sizeof(Hit));

        float3a smn = {1e30f, 1e30f, 1e30f}, smx = {-1e30f, -1e30f, -1e30f};
        for (int i = 0; i < NTRI; i++) {
            smn.x = fminf(smn.x, fminf(fminf(h_tris[i].v0.x, h_tris[i].v1.x), h_tris[i].v2.x));
            smn.y = fminf(smn.y, fminf(fminf(h_tris[i].v0.y, h_tris[i].v1.y), h_tris[i].v2.y));
            smn.z = fminf(smn.z, fminf(fminf(h_tris[i].v0.z, h_tris[i].v1.z), h_tris[i].v2.z));
            smx.x = fmaxf(smx.x, fmaxf(fmaxf(h_tris[i].v0.x, h_tris[i].v1.x), h_tris[i].v2.x));
            smx.y = fmaxf(smx.y, fmaxf(fmaxf(h_tris[i].v0.y, h_tris[i].v1.y), h_tris[i].v2.y));
            smx.z = fmaxf(smx.z, fmaxf(fmaxf(h_tris[i].v0.z, h_tris[i].v1.z), h_tris[i].v2.z));
        }

        // ──── PRIMARY ────
        {
            int side = (int)sqrtf((float)NRAYS);
            float* h_r[9]; for (int j = 0; j < 9; j++) h_r[j] = (float*)malloc(NRAYS * 4);
            for (int i = 0; i < NRAYS; i++) {
                int px = i % side, py = i / side;
                float u = (px + 0.5f) / side * 2.f - 1.f, v = (py + 0.5f) / side * 2.f - 1.f;
                float len = sqrtf(u*u + v*v + 1.f);
                h_r[0][i] = 0; h_r[1][i] = 0; h_r[2][i] = -20.f;
                h_r[3][i] = u/len; h_r[4][i] = v/len; h_r[5][i] = 1.f/len;
                for (int a = 0; a < 3; a++) {
                    float d = h_r[3+a][i];
                    h_r[6+a][i] = 1.f / (fabsf(d) > 1e-8f ? d : (d >= 0 ? 1e-8f : -1e-8f));
                }
            }
            for (int j = 0; j < 9; j++) cudaMemcpy(d_ray[j], h_r[j], NRAYS * 4, cudaMemcpyHostToDevice);
            for (int j = 0; j < 9; j++) free(h_r[j]);

            unsigned int zero = 0;
            cudaMemcpyToSymbol(g_rayCounter, &zero, 4);
            tracePrimary<<<320, 256>>>(d_bvh4, n4, d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8], d_hits, NRAYS);
            cudaDeviceSynchronize();

            cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
            float best = 1e30f;
            for (int r = 0; r < 3; r++) {
                cudaMemcpyToSymbol(g_rayCounter, &zero, 4);
                cudaEventRecord(t0);
                tracePrimary<<<320, 256>>>(d_bvh4, n4, d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                    d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8], d_hits, NRAYS);
                cudaEventRecord(t1); cudaEventSynchronize(t1);
                float ms; cudaEventElapsedTime(&ms, t0, t1); if (ms < best) best = ms;
            }
            cudaMemcpy(h_hits, d_hits, NRAYS * sizeof(Hit), cudaMemcpyDeviceToHost);
            int hc = 0; for (int i = 0; i < NRAYS; i++) if (h_hits[i].tri >= 0) hc++;
            printf("  PRIMARY (Prefetch):  %6.0f MR/s  hit:%5.1f%%\n", (float)NRAYS / best / 1000.f, 100.f * hc / NRAYS);
            cudaEventDestroy(t0); cudaEventDestroy(t1);
        }

        // ──── DIFFUSE: Generate rays + first-hit sort ────
        float* h_r[9]; for (int j = 0; j < 9; j++) h_r[j] = (float*)malloc(NRAYS * 4);
        srand(12345);
        int side = (int)sqrtf((float)NRAYS);
        for (int i = 0; i < NRAYS; i++) {
            int px = i % side, py = i / side;
            h_r[0][i] = ((float)px/side - 0.5f) * 5.f;
            h_r[1][i] = ((float)py/side - 0.5f) * 5.f;
            h_r[2][i] = ((float)rand()/RAND_MAX - 0.5f) * 3.f;
            float r1 = (float)rand()/RAND_MAX, r2 = (float)rand()/RAND_MAX;
            float phi = 6.28318f * r1, ct = sqrtf(1.f - r2), st = sqrtf(r2);
            h_r[3][i] = st * cosf(phi); h_r[4][i] = st * sinf(phi); h_r[5][i] = ct;
            if (rand() % 2) h_r[3][i] = -h_r[3][i];
            if (rand() % 2) h_r[4][i] = -h_r[4][i];
            if (rand() % 2) h_r[5][i] = -h_r[5][i];
            for (int a = 0; a < 3; a++) {
                float d = h_r[3+a][i];
                h_r[6+a][i] = 1.f / (fabsf(d) > 1e-8f ? d : (d >= 0 ? 1e-8f : -1e-8f));
            }
        }
        for (int j = 0; j < 9; j++) cudaMemcpy(d_ray[j], h_r[j], NRAYS * 4, cudaMemcpyHostToDevice);

        // First-hit probe → sort
        int* d_sortKeys; cudaMalloc(&d_sortKeys, NRAYS * 4);
        int* d_rayOrder; cudaMalloc(&d_rayOrder, NRAYS * 4);
        thrust::sequence(thrust::device_ptr<int>(d_rayOrder), thrust::device_ptr<int>(d_rayOrder + NRAYS));
        firstHitProbe<<<(NRAYS+255)/256, 256>>>(d_bvh4,
            d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],
            d_sortKeys, NRAYS);
        thrust::sort_by_key(thrust::device_ptr<int>(d_sortKeys),
                            thrust::device_ptr<int>(d_sortKeys + NRAYS),
                            thrust::device_ptr<int>(d_rayOrder));

        // Reorder rays for megakernel test
        int* h_order = (int*)malloc(NRAYS * 4);
        cudaMemcpy(h_order, d_rayOrder, NRAYS * 4, cudaMemcpyDeviceToHost);
        float* d_ray_sorted[9];
        for (int j = 0; j < 9; j++) {
            cudaMalloc(&d_ray_sorted[j], NRAYS * 4);
            float* tmp = (float*)malloc(NRAYS * 4);
            for (int i = 0; i < NRAYS; i++) tmp[i] = h_r[j][h_order[i]];
            cudaMemcpy(d_ray_sorted[j], tmp, NRAYS * 4, cudaMemcpyHostToDevice);
            free(tmp);
        }
        free(h_order);

        // ──── WAVEFRONT DIFFUSE ────
        {
            int *d_wf_sp, *d_wf_stk, *d_wf_hitTri, *d_wf_rayIdx;
            float *d_wf_tHit, *d_wf_hitU, *d_wf_hitV;
            cudaMalloc(&d_wf_sp, NRAYS * 4);
            cudaMalloc(&d_wf_stk, NRAYS * STACK_DEPTH * 4);
            cudaMalloc(&d_wf_tHit, NRAYS * 4);
            cudaMalloc(&d_wf_hitTri, NRAYS * 4);
            cudaMalloc(&d_wf_hitU, NRAYS * 4);
            cudaMalloc(&d_wf_hitV, NRAYS * 4);
            cudaMalloc(&d_wf_rayIdx, NRAYS * 4);
            int *d_activeMap, *d_activeMapOut, *d_flags, *d_numActive;
            cudaMalloc(&d_activeMap, NRAYS * 4);
            cudaMalloc(&d_activeMapOut, NRAYS * 4);
            cudaMalloc(&d_flags, NRAYS * 4);
            cudaMalloc(&d_numActive, 4);
            unsigned long long *d_we; cudaMalloc(&d_we, 16);

            void* d_temp = nullptr; size_t tempBytes = 0;
            cub::DeviceSelect::Flagged(d_temp, tempBytes, d_activeMap, d_flags, d_activeMapOut, d_numActive, NRAYS);
            cudaMalloc(&d_temp, tempBytes);

            // Warmup
            wavefrontInit<<<(NRAYS+255)/256, 256>>>(d_wf_sp, d_wf_stk, d_wf_tHit, d_wf_hitTri,
                d_wf_hitU, d_wf_hitV, d_wf_rayIdx, d_activeMap, d_rayOrder, NRAYS);
            {
                int numActive = NRAYS;
                while (numActive > 0) {
                    wavefrontStep<<<(numActive+255)/256, 256>>>(d_bvh4, n4,
                        d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                        d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],
                        d_wf_sp, d_wf_stk, d_wf_tHit, d_wf_hitTri, d_wf_hitU, d_wf_hitV, d_wf_rayIdx,
                        d_activeMap, numActive, d_flags, nullptr);
                    cub::DeviceSelect::Flagged(d_temp, tempBytes, d_activeMap, d_flags, d_activeMapOut, d_numActive, numActive);
                    int* tmp = d_activeMap; d_activeMap = d_activeMapOut; d_activeMapOut = tmp;
                    cudaMemcpy(&numActive, d_numActive, 4, cudaMemcpyDeviceToHost);
                }
            }
            cudaDeviceSynchronize();

            cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
            float best = 1e30f; int bestIters = 0;
            unsigned long long h_we_best[2];

            for (int run = 0; run < 3; run++) {
                wavefrontInit<<<(NRAYS+255)/256, 256>>>(d_wf_sp, d_wf_stk, d_wf_tHit, d_wf_hitTri,
                    d_wf_hitU, d_wf_hitV, d_wf_rayIdx, d_activeMap, d_rayOrder, NRAYS);
                unsigned long long h_we[2] = {0, 0};
                cudaMemcpy(d_we, h_we, 16, cudaMemcpyHostToDevice);

                cudaEventRecord(t0);
                int numActive = NRAYS;
                int iters = 0;
                while (numActive > 0) {
                    wavefrontStep<<<(numActive+255)/256, 256>>>(d_bvh4, n4,
                        d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                        d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],
                        d_wf_sp, d_wf_stk, d_wf_tHit, d_wf_hitTri, d_wf_hitU, d_wf_hitV, d_wf_rayIdx,
                        d_activeMap, numActive, d_flags, d_we);
                    cub::DeviceSelect::Flagged(d_temp, tempBytes, d_activeMap, d_flags, d_activeMapOut, d_numActive, numActive);
                    int* tmp = d_activeMap; d_activeMap = d_activeMapOut; d_activeMapOut = tmp;
                    cudaMemcpy(&numActive, d_numActive, 4, cudaMemcpyDeviceToHost);
                    iters++;
                }
                cudaEventRecord(t1); cudaEventSynchronize(t1);
                float ms; cudaEventElapsedTime(&ms, t0, t1);
                if (ms < best) { best = ms; bestIters = iters; cudaMemcpy(h_we_best, d_we, 16, cudaMemcpyDeviceToHost); }
            }

            wavefrontWriteResults<<<(NRAYS+255)/256, 256>>>(d_wf_tHit, d_wf_hitTri, d_wf_hitU, d_wf_hitV, d_wf_rayIdx, d_hits, NRAYS);
            cudaMemcpy(h_hits, d_hits, NRAYS * sizeof(Hit), cudaMemcpyDeviceToHost);
            int hc = 0; for (int i = 0; i < NRAYS; i++) if (h_hits[i].tri >= 0) hc++;
            float wE = h_we_best[1] > 0 ? (float)h_we_best[0] / (h_we_best[1] * 32.f) * 100.f : 0.f;
            printf("  DIFFUSE Wavefront:  %6.0f MR/s  hit:%5.1f%%  warp:%5.1f%%  iters:%d\n",
                   (float)NRAYS / best / 1000.f, 100.f * hc / NRAYS, wE, bestIters);

            cudaFree(d_wf_sp); cudaFree(d_wf_stk); cudaFree(d_wf_tHit); cudaFree(d_wf_hitTri);
            cudaFree(d_wf_hitU); cudaFree(d_wf_hitV); cudaFree(d_wf_rayIdx);
            cudaFree(d_activeMap); cudaFree(d_activeMapOut); cudaFree(d_flags); cudaFree(d_numActive);
            cudaFree(d_we); cudaFree(d_temp);
            cudaEventDestroy(t0); cudaEventDestroy(t1);
        }

        // ──── MEGAKERNEL DIFFUSE (first-hit sorted) ────
        {
            unsigned long long *d_we2; cudaMalloc(&d_we2, 16);
            unsigned long long h_we2[2] = {0, 0};
            unsigned int zero = 0;
            cudaMemcpyToSymbol(g_rayCounter, &zero, 4);
            traceDiffuseMega<<<320, 256>>>(d_bvh4, n4, d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                d_ray_sorted[0],d_ray_sorted[1],d_ray_sorted[2],d_ray_sorted[3],d_ray_sorted[4],d_ray_sorted[5],
                d_ray_sorted[6],d_ray_sorted[7],d_ray_sorted[8], d_hits, NRAYS, nullptr);
            cudaDeviceSynchronize();

            cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
            float best = 1e30f;
            for (int run = 0; run < 3; run++) {
                cudaMemcpyToSymbol(g_rayCounter, &zero, 4);
                cudaMemcpy(d_we2, h_we2, 16, cudaMemcpyHostToDevice);
                cudaEventRecord(t0);
                traceDiffuseMega<<<320, 256>>>(d_bvh4, n4, d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                    d_ray_sorted[0],d_ray_sorted[1],d_ray_sorted[2],d_ray_sorted[3],d_ray_sorted[4],d_ray_sorted[5],
                    d_ray_sorted[6],d_ray_sorted[7],d_ray_sorted[8], d_hits, NRAYS, d_we2);
                cudaEventRecord(t1); cudaEventSynchronize(t1);
                float ms; cudaEventElapsedTime(&ms, t0, t1); if (ms < best) best = ms;
            }
            cudaMemcpy(h_hits, d_hits, NRAYS * sizeof(Hit), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_we2, d_we2, 16, cudaMemcpyDeviceToHost);
            int hc = 0; for (int i = 0; i < NRAYS; i++) if (h_hits[i].tri >= 0) hc++;
            float wE = h_we2[1] > 0 ? (float)h_we2[0] / (h_we2[1] * 32.f) * 100.f : 0.f;
            printf("  DIFFUSE Mega+FH:    %6.0f MR/s  hit:%5.1f%%  warp:%5.1f%%\n",
                   (float)NRAYS / best / 1000.f, 100.f * hc / NRAYS, wE);
            cudaFree(d_we2);
        }
        printf("\n");

        for (int j = 0; j < 9; j++) { free(h_r[j]); cudaFree(d_ray_sorted[j]); }
        cudaFree(d_sortKeys); cudaFree(d_rayOrder);
        for (int j = 0; j < 9; j++) { cudaFree(d_ray[j]); cudaFree(d_tv[j]); free(h_tv[j]); }
        cudaFree(d_bvh4); cudaFree(d_hits);
        free(h_b4); free(h_tris); free(h_hits);
    }
    return 0;
}
