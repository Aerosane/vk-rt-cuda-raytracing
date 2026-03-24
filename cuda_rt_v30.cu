/* v30 — Path Tracing Quality Pipeline
 *
 * Built on v29's proven BVH4 FP16 traversal engine (6263 MR/s primary).
 * Implements look.md Tier 1-3:
 *   Tier 1: Cosine-weighted importance sampling, NEE, Russian roulette, firefly clamp
 *   Tier 2: Multi-frame temporal accumulation
 *   Tier 3: Edge-aware bilateral denoiser (depth + normal guided)
 *
 * Scene: Cornell box (colored walls, area light, two boxes)
 * Output: PPM images (raw, denoised, G-buffer)
 *
 * Build: nvcc -O3 -arch=sm_70 --use_fast_math -o v30 cuda_rt_v30.cu
 * Run:   ./v30 [spp] [width]   (defaults: 64 spp, 512 width)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cfloat>

#define STACK_DEPTH 16
#define BIN_COUNT 16
#define CONST_BVH4 1019
#define MAX_BOUNCES 5
#define FIREFLY_CLAMP 10.0f
#define RR_START_BOUNCE 2
#define M_PIf 3.14159265358979323846f

// ======================== Data structures ========================
struct float3a { float x, y, z; };
struct AABB { float3a mn, mx; };
struct Tri { float3a v0, v1, v2; };
struct BVH4Node {
    __half boundsX[8], boundsY[8], boundsZ[8];
    int child[4];
};
struct Hit { float t; int tri; float u, v; };

struct Material {
    float ar, ag, ab;    // albedo RGB
    float er, eg, eb;    // emission RGB
    float roughness;
};

// G-buffer pixel
struct GBuf {
    float depth;
    float nx, ny, nz;   // normal
    float ar, ag, ab;    // albedo
};

// ======================== Device globals ========================
__constant__ int4 c_bvh4[CONST_BVH4 * 4];
__constant__ int c_bvh4N;
__constant__ Material c_materials[8];

// Area light definition (constant memory)
__constant__ float c_lightCorner[3];  // corner position
__constant__ float c_lightEdge1[3];   // edge vector 1
__constant__ float c_lightEdge2[3];   // edge vector 2
__constant__ float c_lightNormal[3];  // light normal
__constant__ float c_lightArea;       // light area
__constant__ float c_lightEmission[3]; // emission color

// ======================== BVH4 node loader ========================
__device__ __forceinline__ void loadBVH4Node(const int4* __restrict__ bvh, int ni,
    int4& n0, int4& n1, int4& n2, int4& n3)
{
    if (ni < c_bvh4N) {
        n0 = c_bvh4[ni*4]; n1 = c_bvh4[ni*4+1]; n2 = c_bvh4[ni*4+2]; n3 = c_bvh4[ni*4+3];
    } else {
        n0 = __ldg(&bvh[ni*4]); n1 = __ldg(&bvh[ni*4+1]); n2 = __ldg(&bvh[ni*4+2]); n3 = __ldg(&bvh[ni*4+3]);
    }
}

// ======================== RNG (PCG variant) ========================
__device__ __forceinline__ uint32_t pcg_hash(uint32_t input) {
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct RNG {
    uint32_t state;
    __device__ __forceinline__ void init(int pixelIdx, int sampleIdx) {
        state = pcg_hash(pixelIdx * 1337u + sampleIdx * 7919u + 0xDEADBEEFu);
    }
    __device__ __forceinline__ float next() {
        state = pcg_hash(state);
        return (state & 0x00FFFFFFu) * (1.0f / 16777216.0f);
    }
};

// ======================== Vector math ========================
struct f3 { float x, y, z; };

__device__ __forceinline__ f3 make_f3(float x, float y, float z) { return {x, y, z}; }
__device__ __forceinline__ f3 operator+(f3 a, f3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__device__ __forceinline__ f3 operator-(f3 a, f3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__device__ __forceinline__ f3 operator*(f3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
__device__ __forceinline__ f3 operator*(f3 a, f3 b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }
__device__ __forceinline__ float dot(f3 a, f3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ __forceinline__ f3 cross(f3 a, f3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
__device__ __forceinline__ f3 normalize(f3 v) {
    float il = rsqrtf(fmaxf(dot(v, v), 1e-20f));
    return v * il;
}
__device__ __forceinline__ float length(f3 v) { return sqrtf(dot(v, v)); }

// Build orthonormal basis from normal
__device__ __forceinline__ void buildONB(f3 n, f3& t, f3& b) {
    f3 up = (fabsf(n.y) < 0.999f) ? make_f3(0, 1, 0) : make_f3(1, 0, 0);
    t = normalize(cross(up, n));
    b = cross(n, t);
}

// Cosine-weighted hemisphere sampling (importance sampling)
__device__ __forceinline__ f3 sampleCosineHemisphere(f3 normal, RNG& rng) {
    float r1 = rng.next();
    float r2 = rng.next();
    float phi = 2.0f * M_PIf * r1;
    float cosTheta = sqrtf(r2);
    float sinTheta = sqrtf(1.0f - r2);

    f3 t, b;
    buildONB(normal, t, b);

    return normalize(t * (sinTheta * cosf(phi)) + b * (sinTheta * sinf(phi)) + normal * cosTheta);
}

// Sample random point on area light
__device__ __forceinline__ f3 sampleLight(RNG& rng) {
    float u = rng.next();
    float v = rng.next();
    return make_f3(
        c_lightCorner[0] + u * c_lightEdge1[0] + v * c_lightEdge2[0],
        c_lightCorner[1] + u * c_lightEdge1[1] + v * c_lightEdge2[1],
        c_lightCorner[2] + u * c_lightEdge1[2] + v * c_lightEdge2[2]
    );
}

// ======================== BVH Builder (from v29) ========================
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
    std::vector<uint8_t> orderedMat; // material IDs in BVH order
    const Tri* src;
    const uint8_t* srcMat;
    std::vector<AABB> primBB;
    std::vector<float3a> centroids;

    void build(const Tri* t, const uint8_t* mat, int n) {
        src = t; srcMat = mat;
        primBB.resize(n); centroids.resize(n);
        ordered.clear(); orderedMat.clear();
        for (int i = 0; i < n; i++) {
            primBB[i] = triAABB(t[i]);
            centroids[i] = {
                (primBB[i].mn.x+primBB[i].mx.x)*.5f,
                (primBB[i].mn.y+primBB[i].mx.y)*.5f,
                (primBB[i].mn.z+primBB[i].mx.z)*.5f };
        }
        std::vector<int> idx(n);
        for (int i = 0; i < n; i++) idx[i] = i;
        buildRec(idx, 0, n);
    }

    int buildRec(std::vector<int>& idx, int s, int e) {
        N2 nd; nd.triStart = nd.triCount = nd.left = nd.right = 0;
        nd.box = primBB[idx[s]];
        for (int i = s + 1; i < e; i++) nd.box = mergeAABB(nd.box, primBB[idx[i]]);
        int cnt = e - s;
        if (cnt <= 4) {
            nd.triStart = (int)ordered.size(); nd.triCount = cnt;
            for (int i = s; i < e; i++) {
                ordered.push_back(src[idx[i]]);
                orderedMat.push_back(srcMat[idx[i]]);
            }
            nodes.push_back(nd);
            return (int)nodes.size() - 1;
        }
        float bestCost = 1e30f; int bestAxis = 0, bestSplit = s + cnt / 2;
        float pA = saArea(nd.box);
        for (int ax = 0; ax < 3; ax++) {
            // Small count: exact sweep
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
        if (bestSplit <= s) bestSplit = s + 1;
        if (bestSplit >= e) bestSplit = e - 1;
        std::sort(idx.begin() + s, idx.begin() + e, [&](int a, int b) {
            return (&centroids[a].x)[bestAxis] < (&centroids[b].x)[bestAxis];
        });
        int id = (int)nodes.size(); nodes.push_back(nd);
        nodes[id].left = buildRec(idx, s, bestSplit);
        nodes[id].right = buildRec(idx, bestSplit, e);
        return id;
    }
};

static int collapseToB4(const BVHBuild& b2, int ni, BVH4Node* out, int& cnt, const Tri*) {
    auto& n = b2.nodes[ni];
    if (n.triCount > 0) return -((n.triStart << 3) | (n.triCount - 1)) - 2;
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
    for (int i = 0; i < ng; i++) nd.child[i] = collapseToB4(b2, gather[i], out, cnt, nullptr);
    out[me] = nd;
    return me;
}

// ======================== Cornell Box Scene ========================
static void genCornellBox(std::vector<Tri>& tris, std::vector<uint8_t>& matIds) {
    // Materials: 0=white, 1=red, 2=green, 3=light
    auto addQuad = [&](float3a a, float3a b, float3a c, float3a d, uint8_t mat) {
        tris.push_back({a, b, c}); matIds.push_back(mat);
        tris.push_back({a, c, d}); matIds.push_back(mat);
    };

    float S = 5.5f; // room size

    // Floor (y=0, white)
    addQuad({0,0,0}, {S,0,0}, {S,0,S}, {0,0,S}, 0);
    // Ceiling (y=S, white)
    addQuad({0,S,0}, {0,S,S}, {S,S,S}, {S,S,0}, 0);
    // Back wall (z=S, white)
    addQuad({0,0,S}, {S,0,S}, {S,S,S}, {0,S,S}, 0);
    // Left wall (x=0, red)
    addQuad({0,0,0}, {0,0,S}, {0,S,S}, {0,S,0}, 1);
    // Right wall (x=S, green)
    addQuad({S,0,0}, {S,S,0}, {S,S,S}, {S,0,S}, 2);

    // Area light (slightly below ceiling)
    float L0 = 2.13f, L1 = 3.43f, LY = S - 0.02f, LZ0 = 2.27f, LZ1 = 3.32f;
    addQuad({L0,LY,LZ0}, {L1,LY,LZ0}, {L1,LY,LZ1}, {L0,LY,LZ1}, 3);

    // Short box (white, h=1.65)
    float bx0=1.3f, bx1=2.95f, by1=1.65f, bz0=0.65f, bz1=2.30f;
    addQuad({bx0,by1,bz0},{bx1,by1,bz0},{bx1,by1,bz1},{bx0,by1,bz1}, 0); // top
    addQuad({bx0,0,bz0},{bx0,0,bz1},{bx0,by1,bz1},{bx0,by1,bz0}, 0);     // left
    addQuad({bx1,0,bz0},{bx1,by1,bz0},{bx1,by1,bz1},{bx1,0,bz1}, 0);     // right
    addQuad({bx0,0,bz0},{bx1,0,bz0},{bx1,by1,bz0},{bx0,by1,bz0}, 0);     // front
    addQuad({bx0,0,bz1},{bx0,by1,bz1},{bx1,by1,bz1},{bx1,0,bz1}, 0);     // back

    // Tall box (white, h=3.30)
    float tx0=2.85f, tx1=4.48f, ty1=3.30f, tz0=3.00f, tz1=4.60f;
    addQuad({tx0,ty1,tz0},{tx1,ty1,tz0},{tx1,ty1,tz1},{tx0,ty1,tz1}, 0); // top
    addQuad({tx0,0,tz0},{tx0,0,tz1},{tx0,ty1,tz1},{tx0,ty1,tz0}, 0);     // left
    addQuad({tx1,0,tz0},{tx1,ty1,tz0},{tx1,ty1,tz1},{tx1,0,tz1}, 0);     // right
    addQuad({tx0,0,tz0},{tx1,0,tz0},{tx1,ty1,tz0},{tx0,ty1,tz0}, 0);     // front
    addQuad({tx0,0,tz1},{tx0,ty1,tz1},{tx1,ty1,tz1},{tx1,0,tz1}, 0);     // back
}

// ======================== Closest-hit BVH4 traversal (device function) ========================
__device__ Hit traceClosest(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    float ox, float oy, float oz, float dx, float dy, float dz, float tmax)
{
    float ix = 1.f/(fabsf(dx)>1e-8f ? dx : copysignf(1e-8f,dx));
    float iy = 1.f/(fabsf(dy)>1e-8f ? dy : copysignf(1e-8f,dy));
    float iz = 1.f/(fabsf(dz)>1e-8f ? dz : copysignf(1e-8f,dz));

    float tHit = tmax; int hitTri = -1; float hitU = 0, hitV = 0;
    int stk[STACK_DEPTH]; int sp = 0;
    stk[sp++] = 0;

    while (sp > 0) {
        int ni = stk[--sp];
        if (ni < 0) {
            int enc = -(ni+2); int ts = enc>>3, tc = (enc&7)+1;
            for (int t = 0; t < tc; t++) {
                int ti = ts + t;
                float e1x=__ldg(&tv1x[ti])-__ldg(&tv0x[ti]), e1y=__ldg(&tv1y[ti])-__ldg(&tv0y[ti]), e1z=__ldg(&tv1z[ti])-__ldg(&tv0z[ti]);
                float e2x=__ldg(&tv2x[ti])-__ldg(&tv0x[ti]), e2y=__ldg(&tv2y[ti])-__ldg(&tv0y[ti]), e2z=__ldg(&tv2z[ti])-__ldg(&tv0z[ti]);
                float ppx=dy*e2z-dz*e2y, ppy=dz*e2x-dx*e2z, ppz=dx*e2y-dy*e2x;
                float det = e1x*ppx + e1y*ppy + e1z*ppz;
                if (fabsf(det) < 1e-12f) continue;
                float inv = 1.f / det;
                float tx2=ox-__ldg(&tv0x[ti]), ty2=oy-__ldg(&tv0y[ti]), tz2=oz-__ldg(&tv0z[ti]);
                float uu = inv*(tx2*ppx + ty2*ppy + tz2*ppz);
                if (uu < 0.f || uu > 1.f) continue;
                float qx=ty2*e1z-tz2*e1y, qy=tz2*e1x-tx2*e1z, qz=tx2*e1y-ty2*e1x;
                float vv = inv*(dx*qx + dy*qy + dz*qz);
                if (vv < 0.f || uu+vv > 1.f) continue;
                float tt = inv*(e2x*qx + e2y*qy + e2z*qz);
                if (tt > 1e-4f && tt < tHit) { tHit = tt; hitTri = ti; hitU = uu; hitV = vv; }
            }
            continue;
        }
        int4 n0, n1, n2, n3;
        loadBVH4Node(bvh, ni, n0, n1, n2, n3);
        const __half* bx = (const __half*)&n0, *by = (const __half*)&n1, *bz = (const __half*)&n2;
        const int* ch = (const int*)&n3;
        float dist[4]; int child[4];
        for (int c = 0; c < 4; c++) {
            child[c] = ch[c];
            if (ch[c] == -1) { dist[c] = 1e30f; continue; }
            float t1x=(__half2float(bx[c])-ox)*ix, t2x=(__half2float(bx[4+c])-ox)*ix;
            float t1y=(__half2float(by[c])-oy)*iy, t2y=(__half2float(by[4+c])-oy)*iy;
            float t1z=(__half2float(bz[c])-oz)*iz, t2z=(__half2float(bz[4+c])-oz)*iz;
            float tN = fmaxf(fmaxf(fminf(t1x,t2x), fminf(t1y,t2y)), fminf(t1z,t2z));
            float tF = fminf(fminf(fmaxf(t1x,t2x), fmaxf(t1y,t2y)), fmaxf(t1z,t2z));
            dist[c] = (tN <= tF && tF > 0.f && tN < tHit) ? tN : 1e30f;
        }
        #define CSWAP(a,b) do { float da=dist[a],db=dist[b]; int ca=child[a],cb=child[b]; \
            bool s=(da>db); dist[a]=s?db:da; dist[b]=s?da:db; child[a]=s?cb:ca; child[b]=s?ca:cb; } while(0)
        CSWAP(0,1); CSWAP(2,3); CSWAP(0,2); CSWAP(1,3); CSWAP(1,2);
        #undef CSWAP
        for (int c = 3; c >= 0; c--)
            if (dist[c] < 1e30f && sp < STACK_DEPTH) stk[sp++] = child[c];
    }
    return {tHit, hitTri, hitU, hitV};
}

// ======================== Shadow ray (any-hit, early exit) ========================
__device__ bool traceShadow(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const uint8_t* __restrict__ matIds,
    float ox, float oy, float oz, float dx, float dy, float dz, float tmax)
{
    float ix = 1.f/(fabsf(dx)>1e-8f ? dx : copysignf(1e-8f,dx));
    float iy = 1.f/(fabsf(dy)>1e-8f ? dy : copysignf(1e-8f,dy));
    float iz = 1.f/(fabsf(dz)>1e-8f ? dz : copysignf(1e-8f,dz));

    int stk[STACK_DEPTH]; int sp = 0;
    stk[sp++] = 0;

    while (sp > 0) {
        int ni = stk[--sp];
        if (ni < 0) {
            int enc = -(ni+2); int ts = enc>>3, tc = (enc&7)+1;
            for (int t = 0; t < tc; t++) {
                int ti = ts + t;
                // Skip light triangles (material 3) — they shouldn't block light
                if (matIds[ti] == 3) continue;
                float e1x=__ldg(&tv1x[ti])-__ldg(&tv0x[ti]), e1y=__ldg(&tv1y[ti])-__ldg(&tv0y[ti]), e1z=__ldg(&tv1z[ti])-__ldg(&tv0z[ti]);
                float e2x=__ldg(&tv2x[ti])-__ldg(&tv0x[ti]), e2y=__ldg(&tv2y[ti])-__ldg(&tv0y[ti]), e2z=__ldg(&tv2z[ti])-__ldg(&tv0z[ti]);
                float ppx=dy*e2z-dz*e2y, ppy=dz*e2x-dx*e2z, ppz=dx*e2y-dy*e2x;
                float det = e1x*ppx + e1y*ppy + e1z*ppz;
                if (fabsf(det) < 1e-12f) continue;
                float inv = 1.f / det;
                float tx2=ox-__ldg(&tv0x[ti]), ty2=oy-__ldg(&tv0y[ti]), tz2=oz-__ldg(&tv0z[ti]);
                float uu = inv*(tx2*ppx + ty2*ppy + tz2*ppz);
                if (uu < 0.f || uu > 1.f) continue;
                float qx=ty2*e1z-tz2*e1y, qy=tz2*e1x-tx2*e1z, qz=tx2*e1y-ty2*e1x;
                float vv = inv*(dx*qx + dy*qy + dz*qz);
                if (vv < 0.f || uu+vv > 1.f) continue;
                float tt = inv*(e2x*qx + e2y*qy + e2z*qz);
                if (tt > 1e-4f && tt < tmax) return true; // occluded!
            }
            continue;
        }
        int4 n0, n1, n2, n3;
        loadBVH4Node(bvh, ni, n0, n1, n2, n3);
        const __half* bx = (const __half*)&n0, *by = (const __half*)&n1, *bz = (const __half*)&n2;
        const int* ch = (const int*)&n3;
        for (int c = 0; c < 4; c++) {
            if (ch[c] == -1) continue;
            float t1x=(__half2float(bx[c])-ox)*ix, t2x=(__half2float(bx[4+c])-ox)*ix;
            float t1y=(__half2float(by[c])-oy)*iy, t2y=(__half2float(by[4+c])-oy)*iy;
            float t1z=(__half2float(bz[c])-oz)*iz, t2z=(__half2float(bz[4+c])-oz)*iz;
            float tN = fmaxf(fmaxf(fminf(t1x,t2x), fminf(t1y,t2y)), fminf(t1z,t2z));
            float tF = fminf(fminf(fmaxf(t1x,t2x), fmaxf(t1y,t2y)), fmaxf(t1z,t2z));
            if (tN <= tF && tF > 0.f && tN < tmax && sp < STACK_DEPTH) stk[sp++] = ch[c];
        }
    }
    return false; // not occluded
}

// ======================== Compute triangle normal ========================
__device__ __forceinline__ f3 getTriNormal(
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    int ti)
{
    f3 e1 = {__ldg(&tv1x[ti])-__ldg(&tv0x[ti]), __ldg(&tv1y[ti])-__ldg(&tv0y[ti]), __ldg(&tv1z[ti])-__ldg(&tv0z[ti])};
    f3 e2 = {__ldg(&tv2x[ti])-__ldg(&tv0x[ti]), __ldg(&tv2y[ti])-__ldg(&tv0y[ti]), __ldg(&tv2z[ti])-__ldg(&tv0z[ti])};
    return normalize(cross(e1, e2));
}

// ======================== Path Tracing Megakernel ========================
__global__ void pathTraceKernel(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const uint8_t* __restrict__ matIds,
    float* __restrict__ accumR, float* __restrict__ accumG, float* __restrict__ accumB,
    GBuf* __restrict__ gbuf,
    int width, int height, int sampleIdx,
    float camPx, float camPy, float camPz,
    float camFx, float camFy, float camFz,  // forward
    float camRx, float camRy, float camRz,  // right
    float camUx, float camUy, float camUz,  // up
    float fovTan)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;
    int pidx = py * width + px;

    RNG rng;
    rng.init(pidx, sampleIdx);

    // Camera ray with sub-pixel jitter
    float u = (2.0f * (px + rng.next()) / width - 1.0f) * fovTan;
    float v = (2.0f * (py + rng.next()) / height - 1.0f) * fovTan * ((float)height / width);

    f3 dir = normalize(make_f3(
        camFx + u * camRx + v * camUx,
        camFy + u * camRy + v * camUy,
        camFz + u * camRz + v * camUz
    ));
    f3 orig = make_f3(camPx, camPy, camPz);

    f3 radiance = {0, 0, 0};
    f3 throughput = {1, 1, 1};

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        Hit hit = traceClosest(bvh, n4,
            tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z,
            orig.x, orig.y, orig.z, dir.x, dir.y, dir.z, 1e30f);

        if (hit.tri < 0) break; // miss — background is black

        // Hit point and normal
        f3 hitPos = orig + dir * hit.t;
        f3 normal = getTriNormal(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, hit.tri);

        // Ensure normal faces the ray
        if (dot(normal, dir) > 0.f) normal = normal * (-1.f);

        // Get material
        uint8_t matId = matIds[hit.tri];
        Material mat = c_materials[matId];

        // Store G-buffer on first bounce
        if (bounce == 0 && sampleIdx == 0) {
            gbuf[pidx].depth = hit.t;
            gbuf[pidx].nx = normal.x; gbuf[pidx].ny = normal.y; gbuf[pidx].nz = normal.z;
            gbuf[pidx].ar = mat.ar; gbuf[pidx].ag = mat.ag; gbuf[pidx].ab = mat.ab;
        }

        // Emission (hit a light directly)
        f3 emission = make_f3(mat.er, mat.eg, mat.eb);
        radiance = radiance + throughput * emission;

        // If we hit the light, stop bouncing
        if (mat.er > 0 || mat.eg > 0 || mat.eb > 0) break;

        f3 albedo = make_f3(mat.ar, mat.ag, mat.ab);

        // ---- Next Event Estimation (direct light sampling) ----
        {
            f3 lightPoint = sampleLight(rng);
            f3 toLight = lightPoint - hitPos;
            float lightDist = length(toLight);
            f3 lightDir = toLight * (1.0f / lightDist);

            float NdotL = dot(normal, lightDir);
            f3 lNorm = make_f3(c_lightNormal[0], c_lightNormal[1], c_lightNormal[2]);
            float LNdotL = -dot(lNorm, lightDir); // light faces down

            if (NdotL > 0.f && LNdotL > 0.f) {
                // Shadow ray
                bool occluded = traceShadow(bvh, n4,
                    tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z,
                    matIds,
                    hitPos.x + normal.x * 1e-3f,
                    hitPos.y + normal.y * 1e-3f,
                    hitPos.z + normal.z * 1e-3f,
                    lightDir.x, lightDir.y, lightDir.z,
                    lightDist - 2e-3f);

                if (!occluded) {
                    // Light PDF: 1/area, convert to solid angle: dA * cos(θ_light) / dist²
                    float solidAnglePdf = (lightDist * lightDist) / (LNdotL * c_lightArea);
                    f3 le = make_f3(c_lightEmission[0], c_lightEmission[1], c_lightEmission[2]);
                    // BRDF for Lambertian: albedo / π
                    // Contribution: Le * BRDF * cos(θ) / pdf
                    f3 directLight = le * albedo * (NdotL / (M_PIf * solidAnglePdf));
                    radiance = radiance + throughput * directLight;
                }
            }
        }

        // ---- Russian roulette ----
        if (bounce >= RR_START_BOUNCE) {
            float survivalProb = fmaxf(fmaxf(albedo.x, albedo.y), albedo.z);
            survivalProb = fmaxf(survivalProb, 0.05f); // minimum 5% survival
            if (rng.next() >= survivalProb) break;
            throughput = throughput * (1.0f / survivalProb);
        }

        // ---- Sample next bounce direction (cosine-weighted) ----
        f3 newDir = sampleCosineHemisphere(normal, rng);

        // BRDF * cos / pdf for cosine-weighted sampling of Lambertian:
        // BRDF = albedo/π, pdf = cos(θ)/π → BRDF * cos / pdf = albedo
        throughput = throughput * albedo;

        // Offset origin slightly along normal to avoid self-intersection
        orig = hitPos + normal * 1e-3f;
        dir = newDir;
    }

    // Firefly clamp
    float maxC = fmaxf(fmaxf(radiance.x, radiance.y), radiance.z);
    if (maxC > FIREFLY_CLAMP) {
        float scale = FIREFLY_CLAMP / maxC;
        radiance = radiance * scale;
    }

    // Accumulate
    accumR[pidx] += radiance.x;
    accumG[pidx] += radiance.y;
    accumB[pidx] += radiance.z;
}

// ======================== Bilateral Denoiser ========================
__global__ void bilateralDenoise(
    const float* __restrict__ inR, const float* __restrict__ inG, const float* __restrict__ inB,
    float* __restrict__ outR, float* __restrict__ outG, float* __restrict__ outB,
    const GBuf* __restrict__ gbuf,
    int width, int height, int spp,
    float sigmaSpace, float sigmaDepth, float sigmaNormal)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;
    int pidx = py * width + px;

    float invSpp = 1.0f / spp;
    float cd = gbuf[pidx].depth;
    float cnx = gbuf[pidx].nx, cny = gbuf[pidx].ny, cnz = gbuf[pidx].nz;

    float sumR = 0, sumG = 0, sumB = 0, sumW = 0;
    float invSigSp2 = -0.5f / (sigmaSpace * sigmaSpace);
    float invSigDp2 = -0.5f / (sigmaDepth * sigmaDepth);
    float invSigNm2 = -0.5f / (sigmaNormal * sigmaNormal);

    const int RADIUS = 5;
    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {
            int nx = px + dx, ny = py + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            int nidx = ny * width + nx;

            float spatialDist2 = (float)(dx*dx + dy*dy);
            float nd = gbuf[nidx].depth;
            float depthDiff = cd - nd;
            float nnx = gbuf[nidx].nx, nny = gbuf[nidx].ny, nnz = gbuf[nidx].nz;
            float normalDot = cnx*nnx + cny*nny + cnz*nnz;
            float normalDiff = 1.0f - fmaxf(normalDot, 0.f);

            float w = expf(spatialDist2 * invSigSp2 +
                          depthDiff * depthDiff * invSigDp2 +
                          normalDiff * normalDiff * invSigNm2);

            float nr = inR[nidx] * invSpp, ng = inG[nidx] * invSpp, nb = inB[nidx] * invSpp;
            sumR += nr * w; sumG += ng * w; sumB += nb * w; sumW += w;
        }
    }

    float inv = (sumW > 0) ? 1.0f / sumW : 1.0f;
    outR[pidx] = sumR * inv;
    outG[pidx] = sumG * inv;
    outB[pidx] = sumB * inv;
}

// ======================== sRGB tonemapping ========================
__global__ void tonemapSRGB(
    const float* __restrict__ inR, const float* __restrict__ inG, const float* __restrict__ inB,
    uint8_t* __restrict__ outRGB,
    int width, int height, float exposure, bool isDivBySpp, int spp)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;
    int pidx = py * width + px;

    float scale = exposure * (isDivBySpp ? (1.0f / spp) : 1.0f);
    float r = inR[pidx] * scale;
    float g = inG[pidx] * scale;
    float b = inB[pidx] * scale;

    // Reinhard tonemapping
    r = r / (1.0f + r);
    g = g / (1.0f + g);
    b = b / (1.0f + b);

    // sRGB gamma
    r = powf(fmaxf(r, 0.f), 1.0f / 2.2f);
    g = powf(fmaxf(g, 0.f), 1.0f / 2.2f);
    b = powf(fmaxf(b, 0.f), 1.0f / 2.2f);

    int idx = (py * width + px) * 3;
    outRGB[idx+0] = (uint8_t)fminf(r * 255.f + 0.5f, 255.f);
    outRGB[idx+1] = (uint8_t)fminf(g * 255.f + 0.5f, 255.f);
    outRGB[idx+2] = (uint8_t)fminf(b * 255.f + 0.5f, 255.f);
}

// ======================== PPM Writer ========================
static void writePPM(const char* filename, const uint8_t* rgb, int w, int h) {
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("  ERROR: Cannot open %s\n", filename); return; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    // Flip vertically (image convention: y=0 is bottom)
    for (int y = h-1; y >= 0; y--)
        fwrite(rgb + y * w * 3, 1, w * 3, f);
    fclose(f);
    printf("  Saved: %s (%dx%d)\n", filename, w, h);
}

// ======================== G-buffer visualizer ========================
__global__ void visualizeGBuf(
    const GBuf* __restrict__ gbuf, uint8_t* __restrict__ outRGB,
    int width, int height, int mode) // 0=depth, 1=normals, 2=albedo
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;
    int pidx = py * width + px;
    int idx = pidx * 3;

    if (mode == 0) { // Depth (log scale)
        float d = gbuf[pidx].depth;
        float v = (d < 1e20f) ? (1.0f - logf(1.f + d) / logf(1.f + 20.f)) : 0.f;
        uint8_t c = (uint8_t)(fminf(fmaxf(v, 0.f), 1.f) * 255.f);
        outRGB[idx] = outRGB[idx+1] = outRGB[idx+2] = c;
    } else if (mode == 1) { // Normals → RGB
        outRGB[idx+0] = (uint8_t)((gbuf[pidx].nx * 0.5f + 0.5f) * 255.f);
        outRGB[idx+1] = (uint8_t)((gbuf[pidx].ny * 0.5f + 0.5f) * 255.f);
        outRGB[idx+2] = (uint8_t)((gbuf[pidx].nz * 0.5f + 0.5f) * 255.f);
    } else { // Albedo
        outRGB[idx+0] = (uint8_t)(powf(gbuf[pidx].ar, 1.f/2.2f) * 255.f);
        outRGB[idx+1] = (uint8_t)(powf(gbuf[pidx].ag, 1.f/2.2f) * 255.f);
        outRGB[idx+2] = (uint8_t)(powf(gbuf[pidx].ab, 1.f/2.2f) * 255.f);
    }
}

// ======================== Main ========================
int main(int argc, char** argv) {
    int SPP = (argc > 1) ? atoi(argv[1]) : 64;
    int WIDTH = (argc > 2) ? atoi(argv[2]) : 512;
    int HEIGHT = WIDTH;

    printf("══════════════════════════════════════════════════════════════════════════\n");
    printf("  V30 — Path Tracing Quality Pipeline\n");
    printf("  Resolution: %dx%d | SPP: %d | Max bounces: %d\n", WIDTH, HEIGHT, SPP, MAX_BOUNCES);
    printf("  Features: NEE, cosine IS, Russian roulette, bilateral denoise\n");
    printf("══════════════════════════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  GPU: %s | SMs: %d | L2: %dKB\n\n", prop.name, prop.multiProcessorCount,
        (int)(prop.l2CacheSize / 1024));
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    // ---- Materials ----
    Material h_materials[4] = {
        {0.73f, 0.73f, 0.73f,  0,0,0,  1.0f},  // 0: white
        {0.65f, 0.05f, 0.05f,  0,0,0,  1.0f},  // 1: red
        {0.12f, 0.45f, 0.15f,  0,0,0,  1.0f},  // 2: green
        {0.0f,  0.0f,  0.0f,   17.f,12.f,4.f, 1.0f},  // 3: light (warm white emission)
    };
    cudaMemcpyToSymbol(c_materials, h_materials, sizeof(h_materials));

    // ---- Area light definition ----
    float L0 = 2.13f, L1 = 3.43f, LY = 5.48f, LZ0 = 2.27f, LZ1 = 3.32f;
    float h_corner[3] = {L0, LY, LZ0};
    float h_edge1[3] = {L1-L0, 0, 0};
    float h_edge2[3] = {0, 0, LZ1-LZ0};
    float h_lnorm[3] = {0, -1, 0}; // pointing down
    float h_larea = (L1-L0) * (LZ1-LZ0);
    float h_lemit[3] = {17.f, 12.f, 4.f};
    cudaMemcpyToSymbol(c_lightCorner, h_corner, 12);
    cudaMemcpyToSymbol(c_lightEdge1, h_edge1, 12);
    cudaMemcpyToSymbol(c_lightEdge2, h_edge2, 12);
    cudaMemcpyToSymbol(c_lightNormal, h_lnorm, 12);
    cudaMemcpyToSymbol(c_lightArea, &h_larea, 4);
    cudaMemcpyToSymbol(c_lightEmission, h_lemit, 12);

    // ---- Generate scene ----
    printf("  Generating Cornell box...\n");
    std::vector<Tri> tris;
    std::vector<uint8_t> matIdVec;
    genCornellBox(tris, matIdVec);
    int nTri = (int)tris.size();
    printf("  Scene: %d triangles\n", nTri);

    // ---- Build BVH ----
    printf("  Building BVH4...");
    BVHBuild b2;
    b2.build(tris.data(), matIdVec.data(), nTri);
    int maxN4 = (int)b2.nodes.size() * 2;
    BVH4Node* h_b4 = (BVH4Node*)calloc(maxN4, sizeof(BVH4Node));
    int n4 = 0;
    collapseToB4(b2, 0, h_b4, n4, nullptr);
    printf(" %d BVH4 nodes\n", n4);

    // Upload BVH
    int4* d_bvh4;
    cudaMalloc(&d_bvh4, n4 * sizeof(BVH4Node));
    cudaMemcpy(d_bvh4, h_b4, n4 * sizeof(BVH4Node), cudaMemcpyHostToDevice);
    int cN = n4 > CONST_BVH4 ? CONST_BVH4 : n4;
    cudaMemcpyToSymbol(c_bvh4, h_b4, cN * sizeof(BVH4Node));
    cudaMemcpyToSymbol(c_bvh4N, &cN, sizeof(int));

    // Upload triangles (SoA)
    Tri* ord = b2.ordered.data();
    int nOT = (int)b2.ordered.size();
    float *h_tv[9], *d_tv[9];
    for (int j = 0; j < 9; j++) {
        h_tv[j] = (float*)malloc(nOT * 4);
        cudaMalloc(&d_tv[j], nOT * 4);
    }
    for (int i = 0; i < nOT; i++) {
        h_tv[0][i]=ord[i].v0.x; h_tv[1][i]=ord[i].v0.y; h_tv[2][i]=ord[i].v0.z;
        h_tv[3][i]=ord[i].v1.x; h_tv[4][i]=ord[i].v1.y; h_tv[5][i]=ord[i].v1.z;
        h_tv[6][i]=ord[i].v2.x; h_tv[7][i]=ord[i].v2.y; h_tv[8][i]=ord[i].v2.z;
    }
    for (int j = 0; j < 9; j++)
        cudaMemcpy(d_tv[j], h_tv[j], nOT * 4, cudaMemcpyHostToDevice);

    // Upload material IDs (ordered)
    uint8_t* d_matIds;
    cudaMalloc(&d_matIds, nOT);
    cudaMemcpy(d_matIds, b2.orderedMat.data(), nOT, cudaMemcpyHostToDevice);

    // ---- Allocate framebuffer ----
    int nPixels = WIDTH * HEIGHT;
    float *d_accumR, *d_accumG, *d_accumB;
    cudaMalloc(&d_accumR, nPixels * 4); cudaMalloc(&d_accumG, nPixels * 4); cudaMalloc(&d_accumB, nPixels * 4);
    cudaMemset(d_accumR, 0, nPixels * 4); cudaMemset(d_accumG, 0, nPixels * 4); cudaMemset(d_accumB, 0, nPixels * 4);

    GBuf* d_gbuf;
    cudaMalloc(&d_gbuf, nPixels * sizeof(GBuf));
    cudaMemset(d_gbuf, 0, nPixels * sizeof(GBuf));

    // ---- Camera setup ----
    // Camera looking into the Cornell box from the front
    float S = 5.5f;
    float camPos[3] = {S * 0.5f, S * 0.5f, -S * 1.3f};
    float camTarget[3] = {S * 0.5f, S * 0.5f, S * 0.5f};
    // Compute camera basis
    float fwd[3] = {camTarget[0]-camPos[0], camTarget[1]-camPos[1], camTarget[2]-camPos[2]};
    float fwdLen = sqrtf(fwd[0]*fwd[0] + fwd[1]*fwd[1] + fwd[2]*fwd[2]);
    fwd[0]/=fwdLen; fwd[1]/=fwdLen; fwd[2]/=fwdLen;
    float up[3] = {0, 1, 0};
    float right[3] = {fwd[1]*up[2]-fwd[2]*up[1], fwd[2]*up[0]-fwd[0]*up[2], fwd[0]*up[1]-fwd[1]*up[0]};
    float rLen = sqrtf(right[0]*right[0]+right[1]*right[1]+right[2]*right[2]);
    right[0]/=rLen; right[1]/=rLen; right[2]/=rLen;
    float camUp[3] = {right[1]*fwd[2]-right[2]*fwd[1], right[2]*fwd[0]-right[0]*fwd[2], right[0]*fwd[1]-right[1]*fwd[0]};

    float fovDeg = 39.3f;
    float fovTan = tanf(fovDeg * 0.5f * M_PIf / 180.0f);

    // ---- Path trace ----
    printf("\n  Path tracing %d spp...\n", SPP);
    dim3 block(16, 16);
    dim3 grid((WIDTH+15)/16, (HEIGHT+15)/16);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int s = 0; s < SPP; s++) {
        pathTraceKernel<<<grid, block>>>(
            d_bvh4, n4,
            d_tv[0], d_tv[1], d_tv[2], d_tv[3], d_tv[4], d_tv[5], d_tv[6], d_tv[7], d_tv[8],
            d_matIds,
            d_accumR, d_accumG, d_accumB, d_gbuf,
            WIDTH, HEIGHT, s,
            camPos[0], camPos[1], camPos[2],
            fwd[0], fwd[1], fwd[2],
            right[0], right[1], right[2],
            camUp[0], camUp[1], camUp[2],
            fovTan);
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float traceMs;
    cudaEventElapsedTime(&traceMs, t0, t1);

    // Count total rays (approximate: nPixels * SPP * avgBounces)
    printf("  Trace time: %.1f ms (%.1f ms/sample)\n", traceMs, traceMs / SPP);
    printf("  Throughput: %.0f Ksamples/s\n", (float)nPixels * SPP / traceMs);

    // ---- Save raw render ----
    uint8_t* d_rgb;
    cudaMalloc(&d_rgb, nPixels * 3);
    uint8_t* h_rgb = (uint8_t*)malloc(nPixels * 3);

    tonemapSRGB<<<grid, block>>>(d_accumR, d_accumG, d_accumB, d_rgb, WIDTH, HEIGHT, 1.0f, true, SPP);
    cudaMemcpy(h_rgb, d_rgb, nPixels * 3, cudaMemcpyDeviceToHost);
    char fname[256];
    snprintf(fname, sizeof(fname), "/workspaces/codespace/VK_RT/render_raw_%dspp.ppm", SPP);
    writePPM(fname, h_rgb, WIDTH, HEIGHT);

    // ---- Denoise ----
    printf("\n  Denoising (bilateral, 11x11, depth+normal guided)...\n");
    float *d_denR, *d_denG, *d_denB;
    cudaMalloc(&d_denR, nPixels*4); cudaMalloc(&d_denG, nPixels*4); cudaMalloc(&d_denB, nPixels*4);

    cudaEvent_t td0, td1;
    cudaEventCreate(&td0); cudaEventCreate(&td1);
    cudaEventRecord(td0);

    bilateralDenoise<<<grid, block>>>(
        d_accumR, d_accumG, d_accumB,
        d_denR, d_denG, d_denB,
        d_gbuf, WIDTH, HEIGHT, SPP,
        3.0f,   // sigma_spatial
        0.1f,   // sigma_depth
        0.1f);  // sigma_normal

    cudaEventRecord(td1); cudaEventSynchronize(td1);
    float denoiseMs;
    cudaEventElapsedTime(&denoiseMs, td0, td1);
    printf("  Denoise time: %.2f ms\n", denoiseMs);

    // Save denoised
    tonemapSRGB<<<grid, block>>>(d_denR, d_denG, d_denB, d_rgb, WIDTH, HEIGHT, 1.0f, false, 1);
    cudaMemcpy(h_rgb, d_rgb, nPixels * 3, cudaMemcpyDeviceToHost);
    snprintf(fname, sizeof(fname), "/workspaces/codespace/VK_RT/render_denoised_%dspp.ppm", SPP);
    writePPM(fname, h_rgb, WIDTH, HEIGHT);

    // ---- G-buffer visualization ----
    printf("\n  Saving G-buffer...\n");
    const char* gbufNames[] = {"depth", "normals", "albedo"};
    for (int m = 0; m < 3; m++) {
        visualizeGBuf<<<grid, block>>>(d_gbuf, d_rgb, WIDTH, HEIGHT, m);
        cudaMemcpy(h_rgb, d_rgb, nPixels * 3, cudaMemcpyDeviceToHost);
        snprintf(fname, sizeof(fname), "/workspaces/codespace/VK_RT/gbuf_%s.ppm", gbufNames[m]);
        writePPM(fname, h_rgb, WIDTH, HEIGHT);
    }

    // ---- Progressive quality report ----
    printf("\n  ─── Quality Pipeline Status ───\n");
    printf("  ✅ Cosine-weighted importance sampling\n");
    printf("  ✅ Next Event Estimation (area light)\n");
    printf("  ✅ Russian roulette (bounce ≥ %d)\n", RR_START_BOUNCE);
    printf("  ✅ Firefly clamp (max %.0f)\n", FIREFLY_CLAMP);
    printf("  ✅ Multi-frame accumulation (%d spp)\n", SPP);
    printf("  ✅ Edge-aware bilateral denoiser\n");
    printf("  ✅ G-buffer output (depth, normals, albedo)\n");
    printf("  ✅ sRGB tonemapping (Reinhard)\n");

    printf("\n  ─── Timings ───\n");
    printf("  Path trace: %.1f ms (%d spp × %d pixels)\n", traceMs, SPP, nPixels);
    printf("  Denoise:    %.2f ms\n", denoiseMs);
    printf("  Total:      %.1f ms\n", traceMs + denoiseMs);

    // Cleanup
    free(h_b4); free(h_rgb);
    for (int j = 0; j < 9; j++) { free(h_tv[j]); cudaFree(d_tv[j]); }
    cudaFree(d_bvh4); cudaFree(d_matIds);
    cudaFree(d_accumR); cudaFree(d_accumG); cudaFree(d_accumB);
    cudaFree(d_denR); cudaFree(d_denG); cudaFree(d_denB);
    cudaFree(d_gbuf); cudaFree(d_rgb);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaEventDestroy(td0); cudaEventDestroy(td1);

    printf("\n  Done! Check VK_RT/ for .ppm images.\n");
    return 0;
}
