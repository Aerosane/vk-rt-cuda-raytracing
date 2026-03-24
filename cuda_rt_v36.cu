/* v36 — CWBVH (Compressed Wide BVH) Ray Tracer
 *
 * NVIDIA Research paper: "Efficient Incoherent Ray Traversal on GPUs
 * Through Compressed Wide BVHs" (Ylitie, Karras, Laine — HPG 2017)
 *
 * Key innovations over our BVH4 (v29):
 *   - 8-wide BVH nodes (fewer levels, less stack)
 *   - uint8 quantized bounds (80B/node, cache friendly)
 *   - Octant-based child ordering (direction-aware traversal)
 *   - Compressed stack (uint2 bitmask entries)
 *   - Woop triangle pre-transform (fast intersection)
 *   - Dynamic ray fetch (idle threads grab new work)
 *   - Triangle postponing (defer leaf tests when warp diverges)
 *   - Two-phase primary: broadphase root cull → dense CWBVH trace
 *
 * Target: match OptiX 8.1 diffuse (551 MR/s) + beat primary (6 GR/s)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cfloat>
#include <algorithm>

#define CK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

// ======================== Data Structures ========================
struct float3a { float x,y,z; };
struct Tri { float3a v0,v1,v2; };
struct AABB { float3a mn,mx; };

// Ray/Hit in float4 format for coalesced access
struct Ray { float4 origin_tmin; float4 dir_tmax; };
struct Hit { float4 t_triId_u_v; };

// CWBVH node = 5 × float4 = 80 bytes
// n0: p.xyz, {e.x, e.y, e.z, imask}
// n1: childBaseIdx, triBaseIdx, meta[0:3], meta[4:7]
// n2: qlo.x[0:3]|qlo.x[4:7], qhi.x[0:3]|qhi.x[4:7]  (packed as uint8 in float4)
// n3: qlo.y[0:3]|qhi.y[0:3], qlo.y[4:7]|qhi.y[4:7]   (re-swizzled for direction)
// n4: qlo.z[0:3]|qhi.z[0:3], qlo.z[4:7]|qhi.z[4:7]

// Woop triangle = 3 × float4 = 48 bytes (pre-transformed)

// ======================== Device Helpers ========================
__device__ unsigned __bfind_dev(unsigned i) {
    unsigned b; asm volatile("bfind.u32 %0, %1;" : "=r"(b) : "r"(i)); return b;
}
__device__ __inline__ unsigned int sign_extend_s8x4(unsigned int i) {
    unsigned int v; asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(v) : "r"(i)); return v;
}
__device__ __inline__ unsigned int extract_byte(unsigned int i, unsigned int n) {
    return (i >> (n * 8)) & 0xFF;
}
// Video instruction min/max (3-input, fused)
__device__ __inline__ int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }

__device__ int g_rayCounter;

// ======================== CWBVH Traversal Kernel ========================
#define CWBVH_STACK_SIZE 32
#define SM_STACK_SIZE 8
#define DYNAMIC_FETCH 1
#define TRIANGLE_POSTPONING 1

__device__ const float4* d_cwbvhNodes;
__device__ const float4* d_woopTris;
__device__ const int*    d_triIndices;
__device__ int g_nodeVisits;
__device__ int g_triTests;
__device__ int g_hitCount;

#define STACK_POP(X) { --stackPtr; if(stackPtr<SM_STACK_SIZE) X=smStack[threadIdx.x][threadIdx.y][stackPtr]; else X=localStack[stackPtr-SM_STACK_SIZE]; }
#define STACK_PUSH(X) { if(stackPtr<SM_STACK_SIZE) smStack[threadIdx.x][threadIdx.y][stackPtr]=X; else localStack[stackPtr-SM_STACK_SIZE]=X; stackPtr++; }

__global__ void __launch_bounds__(64, 16) traceCWBVH(
    Ray* __restrict__ rayBuffer,
    Hit* __restrict__ hitBuffer,
    int rayCount,
    int* __restrict__ finishedRayCount)
{
    const float ooeps = exp2f(-80.0f);

    uint2 localStack[CWBVH_STACK_SIZE];
    __shared__ uint2 smStack[32][2][SM_STACK_SIZE];
    __shared__ int nextRayArray[2];

    const float4* nodes = d_cwbvhNodes;
    const float4* tris  = d_woopTris;

    int rayidx = rayCount;  // init to invalid so first termination doesn't write
    float3 orig, dir;
    float tmin, tmax;
    float idirx, idiry, idirz;
    unsigned int octinv;
    uint2 nodeGroup = make_uint2(0,0);
    uint2 triangleGroup = make_uint2(0,0);
    int stackPtr = 0;
    int hitAddr = -1;
    float2 triangleuv;

    do {
        volatile int& rayBase = nextRayArray[threadIdx.y];
        bool terminated = (stackPtr == 0 && nodeGroup.y <= 0x00FFFFFF && triangleGroup.y == 0);
        const unsigned int maskTerminated = __ballot_sync(__activemask(), terminated);
        const int numTerminated = __popc(maskTerminated);
        const int idxTerminated = __popc(maskTerminated & ((1u << threadIdx.x) - 1));

        if (terminated) {
            if (hitAddr != -1 || rayidx < rayCount) {
                // Store previous hit
                if (rayidx < rayCount) {
                    hitBuffer[rayidx].t_triId_u_v = make_float4(tmax, __int_as_float(hitAddr), triangleuv.x, triangleuv.y);
                }
            }
            if (idxTerminated == 0)
                rayBase = atomicAdd(finishedRayCount, numTerminated);

            rayidx = rayBase + idxTerminated;
            if (rayidx >= rayCount) break;

            float4 o4 = rayBuffer[rayidx].origin_tmin;
            float4 d4 = rayBuffer[rayidx].dir_tmax;
            orig = make_float3(o4.x, o4.y, o4.z);
            dir  = make_float3(d4.x, d4.y, d4.z);
            tmin = o4.w;
            tmax = d4.w;

            idirx = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
            idiry = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
            idirz = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));

            octinv = 7 - (((dir.x < 0) ? 4 : 0) | ((dir.y < 0) ? 2 : 0) | ((dir.z < 0) ? 1 : 0));

            nodeGroup    = make_uint2(0, 0x80000000);
            triangleGroup = make_uint2(0,0);
            stackPtr = 0;
            hitAddr = -1;
        }

#if DYNAMIC_FETCH
        int lostIters = 0;
#endif

        do {
            if (nodeGroup.y > 0x00FFFFFF) {
                // Process internal node
                const unsigned int hits = nodeGroup.y;
                const unsigned int imask = nodeGroup.y;
                const unsigned int child_bit_index = __bfind_dev(hits);
                const unsigned int child_node_base_index = nodeGroup.x;

                nodeGroup.y &= ~(1 << child_bit_index);
                if (nodeGroup.y > 0x00FFFFFF) { STACK_PUSH(nodeGroup); }

                const unsigned int slot_index = (child_bit_index - 24) ^ octinv;
                const unsigned int octinv4 = octinv * 0x01010101u;
                const unsigned int relative_index = __popc(imask & ~(0xFFFFFFFF << slot_index));
                const unsigned int child_node_index = child_node_base_index + relative_index;

                float4 n0 = __ldg(nodes + child_node_index * 5 + 0);
                float4 n1 = __ldg(nodes + child_node_index * 5 + 1);
                float4 n2 = __ldg(nodes + child_node_index * 5 + 2);
                float4 n3 = __ldg(nodes + child_node_index * 5 + 3);
                float4 n4 = __ldg(nodes + child_node_index * 5 + 4);

                float3 p = make_float3(n0.x, n0.y, n0.z);
                int ex = *((char*)&n0.w + 0);
                int ey = *((char*)&n0.w + 1);
                int ez = *((char*)&n0.w + 2);

                nodeGroup.x = __float_as_uint(n1.x);
                triangleGroup.x = __float_as_uint(n1.y);
                triangleGroup.y = 0;
                unsigned int hitmask = 0;

                const float adjusted_idirx = __uint_as_float((ex + 127) << 23) * idirx;
                const float adjusted_idiry = __uint_as_float((ey + 127) << 23) * idiry;
                const float adjusted_idirz = __uint_as_float((ez + 127) << 23) * idirz;
                const float origx = -(orig.x - p.x) * idirx;
                const float origy = -(orig.y - p.y) * idiry;
                const float origz = -(orig.z - p.z) * idirz;

                // Test first 4 children
                {
                    const unsigned int meta4 = __float_as_uint(n1.z);
                    const unsigned int is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
                    const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
                    const unsigned int bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
                    const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

                    unsigned int swzLox = (idirx < 0) ? __float_as_uint(n3.z) : __float_as_uint(n2.x);
                    unsigned int swzHix = (idirx < 0) ? __float_as_uint(n2.x) : __float_as_uint(n3.z);
                    unsigned int swzLoy = (idiry < 0) ? __float_as_uint(n4.x) : __float_as_uint(n2.z);
                    unsigned int swzHiy = (idiry < 0) ? __float_as_uint(n2.z) : __float_as_uint(n4.x);
                    unsigned int swzLoz = (idirz < 0) ? __float_as_uint(n4.z) : __float_as_uint(n3.x);
                    unsigned int swzHiz = (idirz < 0) ? __float_as_uint(n3.x) : __float_as_uint(n4.z);

                    for (int c = 0; c < 4; c++) {
                        float tmnx = ((swzLox >> (c*8)) & 0xFF) * adjusted_idirx + origx;
                        float tmny = ((swzLoy >> (c*8)) & 0xFF) * adjusted_idiry + origy;
                        float tmnz = ((swzLoz >> (c*8)) & 0xFF) * adjusted_idirz + origz;
                        float tmxx = ((swzHix >> (c*8)) & 0xFF) * adjusted_idirx + origx;
                        float tmxy = ((swzHiy >> (c*8)) & 0xFF) * adjusted_idiry + origy;
                        float tmxz = ((swzHiz >> (c*8)) & 0xFF) * adjusted_idirz + origz;
                        float cmin = fmaxf(fmax_fmax(tmnx, tmny, tmnz), tmin);
                        float cmax = fminf(fmin_fmin(tmxx, tmxy, tmxz), tmax);
                        if (cmin <= cmax) {
                            hitmask |= extract_byte(child_bits4, c) << extract_byte(bit_index4, c);
                        }
                    }
                }
                // Test second 4 children
                {
                    const unsigned int meta4 = __float_as_uint(n1.w);
                    const unsigned int is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
                    const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
                    const unsigned int bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
                    const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

                    unsigned int swzLox = (idirx < 0) ? __float_as_uint(n3.w) : __float_as_uint(n2.y);
                    unsigned int swzHix = (idirx < 0) ? __float_as_uint(n2.y) : __float_as_uint(n3.w);
                    unsigned int swzLoy = (idiry < 0) ? __float_as_uint(n4.y) : __float_as_uint(n2.w);
                    unsigned int swzHiy = (idiry < 0) ? __float_as_uint(n2.w) : __float_as_uint(n4.y);
                    unsigned int swzLoz = (idirz < 0) ? __float_as_uint(n4.w) : __float_as_uint(n3.y);
                    unsigned int swzHiz = (idirz < 0) ? __float_as_uint(n3.y) : __float_as_uint(n4.w);

                    for (int c = 0; c < 4; c++) {
                        float tmnx = ((swzLox >> (c*8)) & 0xFF) * adjusted_idirx + origx;
                        float tmny = ((swzLoy >> (c*8)) & 0xFF) * adjusted_idiry + origy;
                        float tmnz = ((swzLoz >> (c*8)) & 0xFF) * adjusted_idirz + origz;
                        float tmxx = ((swzHix >> (c*8)) & 0xFF) * adjusted_idirx + origx;
                        float tmxy = ((swzHiy >> (c*8)) & 0xFF) * adjusted_idiry + origy;
                        float tmxz = ((swzHiz >> (c*8)) & 0xFF) * adjusted_idirz + origz;
                        float cmin = fmaxf(fmax_fmax(tmnx, tmny, tmnz), tmin);
                        float cmax = fminf(fmin_fmin(tmxx, tmxy, tmxz), tmax);
                        if (cmin <= cmax) {
                            hitmask |= extract_byte(child_bits4, c) << extract_byte(bit_index4, c);
                        }
                    }
                }

                nodeGroup.y = (hitmask & 0xFF000000) | (*((unsigned char*)&n0.w + 3));
                triangleGroup.y = hitmask & 0x00FFFFFF;
            } else {
                triangleGroup = nodeGroup;
                nodeGroup = make_uint2(0,0);
            }

#if TRIANGLE_POSTPONING
            const int totalThreads = __popc(__activemask());
#endif
            // Process triangles
            while (triangleGroup.y != 0) {
#if TRIANGLE_POSTPONING
                const float Rt = 0.2f;
                const int threshold = (int)(totalThreads * Rt);
                if (__popc(__activemask()) < threshold) {
                    STACK_PUSH(triangleGroup);
                    break;
                }
#endif
                int triIndex = __bfind_dev(triangleGroup.y);
                int triAddr = triangleGroup.x * 3 + triIndex * 3;

                float4 v00 = __ldg(tris + triAddr + 0);
                float4 v11 = __ldg(tris + triAddr + 1);
                float4 v22 = __ldg(tris + triAddr + 2);

                // Woop intersection
                float Oz = v00.w - orig.x*v00.x - orig.y*v00.y - orig.z*v00.z;
                float invDz = 1.0f / (dir.x*v00.x + dir.y*v00.y + dir.z*v00.z);
                float t = Oz * invDz;

                float Ox = v11.w + orig.x*v11.x + orig.y*v11.y + orig.z*v11.z;
                float Dx = dir.x*v11.x + dir.y*v11.y + dir.z*v11.z;
                float u = Ox + t * Dx;

                float Oy = v22.w + orig.x*v22.x + orig.y*v22.y + orig.z*v22.z;
                float Dy = dir.x*v22.x + dir.y*v22.y + dir.z*v22.z;
                float v = Oy + t * Dy;

                if (t > tmin && t < tmax && u >= 0.0f && u <= 1.0f && v >= 0.0f && u + v <= 1.0f) {
                    triangleuv = make_float2(u, v);
                    tmax = t;
                    hitAddr = triAddr;
                }
                triangleGroup.y &= ~(1 << triIndex);
            }

            if (nodeGroup.y <= 0x00FFFFFF) {
                if (stackPtr > 0) { STACK_POP(nodeGroup); }
                else break;
            }

#if DYNAMIC_FETCH
            const int Nd = 4, Nw = 16;
            lostIters += __popc(~__activemask()) - Nd;
            if (lostIters >= Nw) break;
#endif
        } while (true);
    } while (true);
}

// ======================== Two-Phase Broadphase for Primary ========================
// Tests ray against root CWBVH node's 8 children, compacts survivors
__global__ void __launch_bounds__(256, 8) broadphaseRootCWBVH(
    int* __restrict__ d_survivors,
    int* __restrict__ d_numSurvivors,
    int numRays, int side,
    float camOx, float camOy, float camOz)
{
    const float ooeps = exp2f(-80.0f);
    const float4* nodes = d_cwbvhNodes;
    const unsigned lane = threadIdx.x & 31;

    // Load root node (node index 0)
    float4 n0 = __ldg(nodes + 0);
    float4 n1 = __ldg(nodes + 1);
    float4 n2 = __ldg(nodes + 2);
    float4 n3 = __ldg(nodes + 3);
    float4 n4 = __ldg(nodes + 4);

    float3 p = make_float3(n0.x, n0.y, n0.z);
    int ex = *((char*)&n0.w + 0);
    int ey = *((char*)&n0.w + 1);
    int ez = *((char*)&n0.w + 2);

    while (true) {
        int bs;
        if (lane == 0) bs = atomicAdd(&g_rayCounter, 32);
        bs = __shfl_sync(0xFFFFFFFF, bs, 0);
        if (bs >= numRays) break;
        int ri = bs + lane;
        bool anyHit = false;

        if (ri < numRays) {
            int px = ri % side, py = ri / side;
            float u = (px + 0.5f) / side * 2.f - 1.f;
            float v = (py + 0.5f) / side * 2.f - 1.f;
            float rlen = rsqrtf(u*u + v*v + 1.f);
            float dx = u*rlen, dy = v*rlen, dz = rlen;
            float idirx = 1.0f / (fabsf(dx) > ooeps ? dx : copysignf(ooeps, dx));
            float idiry = 1.0f / (fabsf(dy) > ooeps ? dy : copysignf(ooeps, dy));
            float idirz = 1.0f / (fabsf(dz) > ooeps ? dz : copysignf(ooeps, dz));

            float adj_idirx = __uint_as_float((ex + 127) << 23) * idirx;
            float adj_idiry = __uint_as_float((ey + 127) << 23) * idiry;
            float adj_idirz = __uint_as_float((ez + 127) << 23) * idirz;
            float ox = -(camOx - p.x) * idirx;
            float oy = -(camOy - p.y) * idiry;
            float oz = -(camOz - p.z) * idirz;

            // Test all 8 children of root
            // First 4
            unsigned int swzLox = (idirx < 0) ? __float_as_uint(n3.z) : __float_as_uint(n2.x);
            unsigned int swzHix = (idirx < 0) ? __float_as_uint(n2.x) : __float_as_uint(n3.z);
            unsigned int swzLoy = (idiry < 0) ? __float_as_uint(n4.x) : __float_as_uint(n2.z);
            unsigned int swzHiy = (idiry < 0) ? __float_as_uint(n2.z) : __float_as_uint(n4.x);
            unsigned int swzLoz = (idirz < 0) ? __float_as_uint(n4.z) : __float_as_uint(n3.x);
            unsigned int swzHiz = (idirz < 0) ? __float_as_uint(n3.x) : __float_as_uint(n4.z);
            for (int c = 0; c < 4 && !anyHit; c++) {
                float tmnx = ((swzLox >> (c*8)) & 0xFF) * adj_idirx + ox;
                float tmny = ((swzLoy >> (c*8)) & 0xFF) * adj_idiry + oy;
                float tmnz = ((swzLoz >> (c*8)) & 0xFF) * adj_idirz + oz;
                float tmxx = ((swzHix >> (c*8)) & 0xFF) * adj_idirx + ox;
                float tmxy = ((swzHiy >> (c*8)) & 0xFF) * adj_idiry + oy;
                float tmxz = ((swzHiz >> (c*8)) & 0xFF) * adj_idirz + oz;
                float cmin = fmaxf(fmaxf(fmaxf(tmnx,tmny),tmnz), 0.001f);
                float cmax = fminf(fminf(fminf(tmxx,tmxy),tmxz), 1e30f);
                if (cmin <= cmax) anyHit = true;
            }
            // Second 4
            if (!anyHit) {
                swzLox = (idirx < 0) ? __float_as_uint(n3.w) : __float_as_uint(n2.y);
                swzHix = (idirx < 0) ? __float_as_uint(n2.y) : __float_as_uint(n3.w);
                swzLoy = (idiry < 0) ? __float_as_uint(n4.y) : __float_as_uint(n2.w);
                swzHiy = (idiry < 0) ? __float_as_uint(n2.w) : __float_as_uint(n4.y);
                swzLoz = (idirz < 0) ? __float_as_uint(n4.w) : __float_as_uint(n3.y);
                swzHiz = (idirz < 0) ? __float_as_uint(n3.y) : __float_as_uint(n4.w);
                for (int c = 0; c < 4 && !anyHit; c++) {
                    float tmnx = ((swzLox >> (c*8)) & 0xFF) * adj_idirx + ox;
                    float tmny = ((swzLoy >> (c*8)) & 0xFF) * adj_idiry + oy;
                    float tmnz = ((swzLoz >> (c*8)) & 0xFF) * adj_idirz + oz;
                    float tmxx = ((swzHix >> (c*8)) & 0xFF) * adj_idirx + ox;
                    float tmxy = ((swzHiy >> (c*8)) & 0xFF) * adj_idiry + oy;
                    float tmxz = ((swzHiz >> (c*8)) & 0xFF) * adj_idirz + oz;
                    float cmin = fmaxf(fmaxf(fmaxf(tmnx,tmny),tmnz), 0.001f);
                    float cmax = fminf(fminf(fminf(tmxx,tmxy),tmxz), 1e30f);
                    if (cmin <= cmax) anyHit = true;
                }
            }
        }
        unsigned int mask = __ballot_sync(0xFFFFFFFF, anyHit);
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

// ======================== Generate rays kernel ========================
__global__ void generatePrimaryRays(Ray* rays, int* survivors, int numSurvivors, int side,
                                     float camOx, float camOy, float camOz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSurvivors) return;
    int ri = survivors[i];
    int px = ri % side, py = ri / side;
    float u = (px + 0.5f) / side * 2.f - 1.f;
    float v = (py + 0.5f) / side * 2.f - 1.f;
    float rlen = rsqrtf(u*u + v*v + 1.f);
    rays[i].origin_tmin = make_float4(camOx, camOy, camOz, 0.001f);
    rays[i].dir_tmax    = make_float4(u*rlen, v*rlen, rlen, 1e30f);
}

__global__ void generateDiffuseRays(Ray* rays, int numRays) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRays) return;
    unsigned int seed = i * 1973 + 9277;
    auto rng = [&]() -> float {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        return float(seed & 0xFFFFFF) / float(0xFFFFFF);
    };
    float ox = rng()*10.f-5.f, oy = rng()*3.f+0.1f, oz = rng()*10.f-5.f;
    float theta = acosf(rng()), phi = 6.2831853f * rng();
    float dx = sinf(theta)*cosf(phi), dy = fabsf(cosf(theta)), dz = sinf(theta)*sinf(phi);
    rays[i].origin_tmin = make_float4(ox, oy, oz, 0.001f);
    rays[i].dir_tmax    = make_float4(dx, dy, dz, 1e30f);
}

// ======================== CPU: BVH2 Builder (SAH) ========================
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
        src=t; primBB.resize(n); centroids.resize(n); ordered.clear();
        nodes.reserve(n*2);  // prevent reallocation during buildRec
        for(int i=0;i<n;i++){primBB[i]=triAABB(t[i]); centroids[i]={
            (primBB[i].mn.x+primBB[i].mx.x)*.5f,(primBB[i].mn.y+primBB[i].mx.y)*.5f,(primBB[i].mn.z+primBB[i].mx.z)*.5f};}
        std::vector<int> idx(n); for(int i=0;i<n;i++) idx[i]=i;
        buildRec(idx,0,n);
    }
    int buildRec(std::vector<int>&idx,int s,int e){
        BVH2Node nd; nd.triStart=nd.triCount=nd.left=nd.right=0;
        nd.box=primBB[idx[s]]; for(int i=s+1;i<e;i++) nd.box=mergeAABB(nd.box,primBB[idx[i]]);
        int cnt=e-s;
        if(cnt<=3){nd.triStart=(int)ordered.size();nd.triCount=cnt;
            for(int i=s;i<e;i++) ordered.push_back(src[idx[i]]);
            nodes.push_back(nd); return (int)nodes.size()-1;}
        float bestCost=1e30f;int bestAxis=0,bestSplit=s+cnt/2;float pA=saArea(nd.box);
        for(int ax=0;ax<3;ax++){
            float cmin=1e30f,cmax=-1e30f;
            for(int i=s;i<e;i++){float c=(&centroids[idx[i]].x)[ax];cmin=fminf(cmin,c);cmax=fmaxf(cmax,c);}
            if(cmax-cmin<1e-8f) continue;
            const int NB=16; AABB lBox[NB],rBox[NB]; int lCnt[NB],rCnt[NB];
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
        int leftChild=buildRec(idx,s,bestSplit);int rightChild=buildRec(idx,bestSplit,e);
        nodes[id].left=leftChild;nodes[id].right=rightChild;return id;
    }
};

// ======================== CPU: BVH2 → BVH8 Collapse ========================
struct BVH8Node {
    AABB childBounds[8];
    int children[8];      // -1=empty, >=0 internal BVH8 index, <-1 leaf
    int childCount;
    int leafTriStart, leafTriCount; // for leaf-only nodes
};

// Collapse BVH2 → BVH8 by opening children greedily
struct BVH8Builder {
    std::vector<BVH8Node> nodes;
    const BVH2Builder& bvh2;

    BVH8Builder(const BVH2Builder& b):bvh2(b){ nodes.reserve(200000); }

    int collapse(int b2idx) {
        auto& n = bvh2.nodes[b2idx];
        if (n.triCount > 0) {
            // Leaf
            BVH8Node nd; nd.childCount = 0; nd.leafTriStart = n.triStart; nd.leafTriCount = n.triCount;
            for(int i=0;i<8;i++){nd.children[i]=-1;}
            nodes.push_back(nd); return (int)nodes.size()-1;
        }

        // Gather up to 8 children by opening internal nodes
        struct Candidate { int b2idx; float cost; AABB box; };
        std::vector<Candidate> children;
        children.push_back({n.left, saArea(bvh2.nodes[n.left].box), bvh2.nodes[n.left].box});
        children.push_back({n.right, saArea(bvh2.nodes[n.right].box), bvh2.nodes[n.right].box});

        while (children.size() < 8) {
            // Find best internal child to open (highest cost = largest area)
            int bestIdx = -1; float bestCost = -1;
            for (int i = 0; i < (int)children.size(); i++) {
                if (bvh2.nodes[children[i].b2idx].triCount == 0 && children[i].cost > bestCost) {
                    bestCost = children[i].cost; bestIdx = i;
                }
            }
            if (bestIdx < 0) break; // All remaining are leaves

            // Open this child
            int openIdx = children[bestIdx].b2idx;
            auto& openNode = bvh2.nodes[openIdx];
            children[bestIdx] = {openNode.left, saArea(bvh2.nodes[openNode.left].box), bvh2.nodes[openNode.left].box};
            children.push_back({openNode.right, saArea(bvh2.nodes[openNode.right].box), bvh2.nodes[openNode.right].box});
        }

        BVH8Node nd; nd.childCount = (int)children.size(); nd.leafTriStart = 0; nd.leafTriCount = 0;
        for(int i=0;i<8;i++) nd.children[i] = -1;

        int id = (int)nodes.size(); nodes.push_back(nd);

        // Octant-based child ordering (greedy assignment)
        {
            AABB parentBox = n.box;
            float3a parentCtr = {(parentBox.mn.x+parentBox.mx.x)*.5f,(parentBox.mn.y+parentBox.mx.y)*.5f,(parentBox.mn.z+parentBox.mx.z)*.5f};
            float costs[8][8];
            int assignment[8]; bool slotUsed[8];
            for(int i=0;i<8;i++){assignment[i]=-1;slotUsed[i]=false;}
            for(int s=0;s<8;s++){
                float dsx=(s&4)?-1.f:1.f, dsy=(s&2)?-1.f:1.f, dsz=(s&1)?-1.f:1.f;
                for(int c=0;c<(int)children.size();c++){
                    float3a ctr={(children[c].box.mn.x+children[c].box.mx.x)*.5f,
                                 (children[c].box.mn.y+children[c].box.mx.y)*.5f,
                                 (children[c].box.mn.z+children[c].box.mx.z)*.5f};
                    costs[s][c]=(ctr.x-parentCtr.x)*dsx+(ctr.y-parentCtr.y)*dsy+(ctr.z-parentCtr.z)*dsz;
                }
                for(int c=(int)children.size();c<8;c++) costs[s][c]=1e30f;
            }
            // Greedy matching
            for(int iter=0;iter<(int)children.size();iter++){
                float minC=1e30f;int ms=-1,mc=-1;
                for(int s=0;s<8;s++)for(int c=0;c<(int)children.size();c++){
                    if(!slotUsed[s]&&assignment[c]==-1&&costs[s][c]<minC){minC=costs[s][c];ms=s;mc=c;}}
                if(ms>=0){slotUsed[ms]=true;assignment[mc]=ms;}
            }
            // Place children in assigned slots
            for(int c=0;c<(int)children.size();c++){
                int slot = assignment[c]; if(slot<0) slot=c;
                nodes[id].childBounds[slot] = children[c].box;
                nodes[id].children[slot] = collapse(children[c].b2idx);
            }
        }
        return id;
    }
};

// ======================== CPU: BVH8 → CWBVH Compression ========================
struct CWBVHData {
    std::vector<float4> nodeData;    // 5 float4 per node
    std::vector<float4> woopTris;    // 3 float4 per triangle
    std::vector<int>    triIndices;  // original triangle index

    void woopify(float3a v0, float3a v1, float3a v2, float4& o0, float4& o1, float4& o2) {
        // Woop pre-transform: convert triangle to affine space
        float e1x=v0.x-v2.x,e1y=v0.y-v2.y,e1z=v0.z-v2.z;
        float e2x=v1.x-v2.x,e2y=v1.y-v2.y,e2z=v1.z-v2.z;
        float nx=e1y*e2z-e1z*e2y,ny=e1z*e2x-e1x*e2z,nz=e1x*e2y-e1y*e2x;
        float det=e1x*(e2y*nz-e2z*ny)-e1y*(e2x*nz-e2z*nx)+e1z*(e2x*ny-e2y*nx);
        if(fabsf(det)<1e-20f) det=1e-20f;
        float inv=1.f/det;
        // Row 2 of inverse (for t computation)
        float m20=inv*(e2y*nz-e2z*ny), m21=inv*(e2z*nx-e2x*nz), m22=inv*(e2x*ny-e2y*nx);
        float m23=-(m20*v2.x+m21*v2.y+m22*v2.z);
        // Row 0 of inverse (for u computation)
        float m00=inv*(e2y*nz-nz*e2y); // This needs proper matrix inverse
        // Use simpler formulation:
        // M = [e1 e2 n]^-1, then:
        // t = row2 . (orig - v2), u = row0 . (orig - v2), v = row1 . (orig - v2)

        // Actually, let's use the standard Woop formulation from the reference
        // Construct matrix [v0-v2, v1-v2, cross(v0-v2, v1-v2), v2; 0 0 0 1]
        // and invert it
        float cx=e1y*e2z-e1z*e2y, cy=e1z*e2x-e1x*e2z, cz=e1x*e2y-e1y*e2x;
        // 3x3 matrix = [e1, e2, c] column-wise, invert via cofactors
        float d = e1x*(e2y*cz-e2z*cy) - e1y*(e2x*cz-e2z*cx) + e1z*(e2x*cy-e2y*cx);
        if(fabsf(d)<1e-30f) d=1e-30f;
        float id=1.f/d;
        // Inverse rows:
        float r0x=id*(e2y*cz-e2z*cy), r0y=id*(e1z*cy-e1y*cz), r0z=id*(e1y*e2z-e1z*e2y);
        float r1x=id*(e2z*cx-e2x*cz), r1y=id*(e1x*cz-e1z*cx), r1z=id*(e1z*e2x-e1x*e2z);
        float r2x=id*(e2x*cy-e2y*cx), r2y=id*(e1y*cx-e1x*cy), r2z=id*(e1x*e2y-e1y*e2x);

        o0 = make_float4(r2x,r2y,r2z, +(r2x*v2.x+r2y*v2.y+r2z*v2.z));  // t (NOTE: +, not -, per reference)
        o1 = make_float4(r0x,r0y,r0z, -(r0x*v2.x+r0y*v2.y+r0z*v2.z));    // u
        o2 = make_float4(r1x,r1y,r1z, -(r1x*v2.x+r1y*v2.y+r1z*v2.z));    // v
    }

    void convert(const BVH8Builder& bvh8, const Tri* orderedTris) {
        nodeData.clear(); woopTris.clear(); triIndices.clear();
        // Preallocate root
        for(int i=0;i<5;i++) nodeData.push_back(make_float4(0.f,0.f,0.f,0.f));
        convertNode(bvh8, 0, 0, orderedTris);
    }

    void convertNode(const BVH8Builder& bvh8, int nodeIdx, int outNodeAddr, const Tri* orderedTris) {
        auto& node = bvh8.nodes[nodeIdx];
        if (node.childCount == 0) return; // leaf only node - handled by parent

        // Compute node AABB
        float3a nodeLo={1e30f,1e30f,1e30f}, nodeHi={-1e30f,-1e30f,-1e30f};
        for(int c=0;c<8;c++){
            if(node.children[c]==-1) continue;
            nodeLo.x=fminf(nodeLo.x,node.childBounds[c].mn.x);
            nodeLo.y=fminf(nodeLo.y,node.childBounds[c].mn.y);
            nodeLo.z=fminf(nodeLo.z,node.childBounds[c].mn.z);
            nodeHi.x=fmaxf(nodeHi.x,node.childBounds[c].mx.x);
            nodeHi.y=fmaxf(nodeHi.y,node.childBounds[c].mx.y);
            nodeHi.z=fmaxf(nodeHi.z,node.childBounds[c].mx.z);
        }

        // Quantization exponents
        signed char ex = (signed char)ceilf(log2f(fmaxf(nodeHi.x-nodeLo.x,1e-20f)/255.f));
        signed char ey = (signed char)ceilf(log2f(fmaxf(nodeHi.y-nodeLo.y,1e-20f)/255.f));
        signed char ez = (signed char)ceilf(log2f(fmaxf(nodeHi.z-nodeLo.z,1e-20f)/255.f));

        unsigned char imask = 0;
        int internalChildCount = 0, leafTriTotal = 0;
        int childBaseIndex = 0, triangleBaseIndex = 0;

        // First pass: count children and allocate space
        std::vector<int> internalChildNodeAddrs;
        for(int c=0;c<8;c++){
            if(node.children[c]==-1) continue;
            auto& child = bvh8.nodes[node.children[c]];
            if(child.childCount > 0) {
                // Internal child
                if(internalChildCount == 0) childBaseIndex = (int)nodeData.size()/5;
                for(int j=0;j<5;j++) nodeData.push_back(make_float4(0.f,0.f,0.f,0.f));
                internalChildNodeAddrs.push_back((int)nodeData.size()-5);
                imask |= (1 << c);
                internalChildCount++;
            } else {
                // Leaf child
                if(leafTriTotal == 0) triangleBaseIndex = (int)woopTris.size()/3;
                for(int t=0;t<child.leafTriCount;t++){
                    float4 w0,w1,w2;
                    auto& tri = orderedTris[child.leafTriStart+t];
                    woopify(tri.v0,tri.v1,tri.v2,w0,w1,w2);
                    woopTris.push_back(w0);woopTris.push_back(w1);woopTris.push_back(w2);
                    triIndices.push_back(child.leafTriStart+t);
                }
                leafTriTotal += child.leafTriCount;
            }
        }

        // Encode quantized bounds and meta
        unsigned char qBounds[48] = {}; // 6 components × 8 children
        unsigned char metaField[8] = {};
        int internalIdx = 0, leafOffset = 0;

        for(int c=0;c<8;c++){
            if(node.children[c]==-1){metaField[c]=0;continue;}
            auto& child = bvh8.nodes[node.children[c]];
            // Quantize bounds
            float scx=powf(2.f,ex), scy=powf(2.f,ey), scz=powf(2.f,ez);
            int qlox=(int)floorf((node.childBounds[c].mn.x-nodeLo.x)/scx);
            int qloy=(int)floorf((node.childBounds[c].mn.y-nodeLo.y)/scy);
            int qloz=(int)floorf((node.childBounds[c].mn.z-nodeLo.z)/scz);
            int qhix=(int)ceilf((node.childBounds[c].mx.x-nodeLo.x)/scx);
            int qhiy=(int)ceilf((node.childBounds[c].mx.y-nodeLo.y)/scy);
            int qhiz=(int)ceilf((node.childBounds[c].mx.z-nodeLo.z)/scz);
            qlox=std::max(0,std::min(255,qlox)); qhix=std::max(0,std::min(255,qhix));
            qloy=std::max(0,std::min(255,qloy)); qhiy=std::max(0,std::min(255,qhiy));
            qloz=std::max(0,std::min(255,qloz)); qhiz=std::max(0,std::min(255,qhiz));

            qBounds[c+0]  = qlox; qBounds[c+8]  = qloy; qBounds[c+16] = qloz;
            qBounds[c+24] = qhix; qBounds[c+32] = qhiy; qBounds[c+40] = qhiz;

            if(child.childCount > 0){
                metaField[c] = (1<<5) | ((24+c)&0x1F);
                internalIdx++;
            } else {
                int unary = child.leafTriCount==1?0b001:child.leafTriCount==2?0b011:0b111;
                metaField[c] = (unary<<5) | (leafOffset&0x1F);
                leafOffset += child.leafTriCount;
            }
        }

        // Pack into 5 × float4
        unsigned char exyzImask[4]={(unsigned char)ex,(unsigned char)ey,(unsigned char)ez,imask};
        nodeData[outNodeAddr/1+0] = make_float4(nodeLo.x,nodeLo.y,nodeLo.z, *(float*)exyzImask);

        unsigned int meta03 = metaField[0]|(metaField[1]<<8)|(metaField[2]<<16)|(metaField[3]<<24);
        unsigned int meta47 = metaField[4]|(metaField[5]<<8)|(metaField[6]<<16)|(metaField[7]<<24);
        nodeData[outNodeAddr/1+1] = make_float4(*(float*)&childBaseIndex, *(float*)&triangleBaseIndex,
                                                  *(float*)&meta03, *(float*)&meta47);

        // Pack quantized bounds: n2,n3,n4
        // n2.x = qlox[0:3], n2.y = qlox[4:7], n2.z = qloy[0:3], n2.w = qloy[4:7]
        unsigned int qlox03=qBounds[0]|(qBounds[1]<<8)|(qBounds[2]<<16)|(qBounds[3]<<24);
        unsigned int qlox47=qBounds[4]|(qBounds[5]<<8)|(qBounds[6]<<16)|(qBounds[7]<<24);
        unsigned int qloy03=qBounds[8]|(qBounds[9]<<8)|(qBounds[10]<<16)|(qBounds[11]<<24);
        unsigned int qloy47=qBounds[12]|(qBounds[13]<<8)|(qBounds[14]<<16)|(qBounds[15]<<24);
        nodeData[outNodeAddr/1+2] = make_float4(*(float*)&qlox03,*(float*)&qlox47,*(float*)&qloy03,*(float*)&qloy47);

        // n3.x = qloz[0:3], n3.y = qloz[4:7], n3.z = qhix[0:3], n3.w = qhix[4:7]
        unsigned int qloz03=qBounds[16]|(qBounds[17]<<8)|(qBounds[18]<<16)|(qBounds[19]<<24);
        unsigned int qloz47=qBounds[20]|(qBounds[21]<<8)|(qBounds[22]<<16)|(qBounds[23]<<24);
        unsigned int qhix03=qBounds[24]|(qBounds[25]<<8)|(qBounds[26]<<16)|(qBounds[27]<<24);
        unsigned int qhix47=qBounds[28]|(qBounds[29]<<8)|(qBounds[30]<<16)|(qBounds[31]<<24);
        nodeData[outNodeAddr/1+3] = make_float4(*(float*)&qloz03,*(float*)&qloz47,*(float*)&qhix03,*(float*)&qhix47);

        // n4.x = qhiy[0:3], n4.y = qhiy[4:7], n4.z = qhiz[0:3], n4.w = qhiz[4:7]
        unsigned int qhiy03=qBounds[32]|(qBounds[33]<<8)|(qBounds[34]<<16)|(qBounds[35]<<24);
        unsigned int qhiy47=qBounds[36]|(qBounds[37]<<8)|(qBounds[38]<<16)|(qBounds[39]<<24);
        unsigned int qhiz03=qBounds[40]|(qBounds[41]<<8)|(qBounds[42]<<16)|(qBounds[43]<<24);
        unsigned int qhiz47=qBounds[44]|(qBounds[45]<<8)|(qBounds[46]<<16)|(qBounds[47]<<24);
        nodeData[outNodeAddr/1+4] = make_float4(*(float*)&qhiy03,*(float*)&qhiy47,*(float*)&qhiz03,*(float*)&qhiz47);

        // Recurse into internal children
        internalIdx = 0;
        for(int c=0;c<8;c++){
            if(node.children[c]==-1) continue;
            if(bvh8.nodes[node.children[c]].childCount > 0){
                convertNode(bvh8, node.children[c], internalChildNodeAddrs[internalIdx], orderedTris);
                internalIdx++;
            }
        }
    }
};

// ======================== Scene Generator ========================
static int genScene(Tri* tris, int n, float S) {
    int ng = (int)cbrtf((float)n / 12.f); if(ng<1) ng=1;
    float sp = S / ng; int idx = 0;
    tris[idx++] = {{-S,0,-S},{S,0,-S},{S,0,S}};
    tris[idx++] = {{-S,0,-S},{S,0,S},{-S,0,S}};
    for(int ix=0;ix<ng;ix++)for(int iy=0;iy<ng;iy++)for(int iz=0;iz<ng;iz++){
        float cx=-S/2+(ix+.5f)*sp, cy=.5f+(iy+.5f)*sp, cz=-S/2+(iz+.5f)*sp, r=sp*.3f;
        float x0=cx-r,x1=cx+r,y0=cy-r,y1=cy+r,z0=cz-r,z1=cz+r;
        if(idx+12>n)goto done;
        tris[idx++]={{x0,y0,z1},{x1,y0,z1},{x1,y1,z1}};tris[idx++]={{x0,y0,z1},{x1,y1,z1},{x0,y1,z1}};
        tris[idx++]={{x1,y0,z0},{x0,y0,z0},{x0,y1,z0}};tris[idx++]={{x1,y0,z0},{x0,y1,z0},{x1,y1,z0}};
        tris[idx++]={{x0,y1,z0},{x0,y1,z1},{x1,y1,z1}};tris[idx++]={{x0,y1,z0},{x1,y1,z1},{x1,y1,z0}};
        tris[idx++]={{x0,y0,z1},{x0,y0,z0},{x1,y0,z0}};tris[idx++]={{x0,y0,z1},{x1,y0,z0},{x1,y0,z1}};
        tris[idx++]={{x0,y0,z0},{x0,y0,z1},{x0,y1,z1}};tris[idx++]={{x0,y0,z0},{x0,y1,z1},{x0,y1,z0}};
        tris[idx++]={{x1,y0,z1},{x1,y0,z0},{x1,y1,z0}};tris[idx++]={{x1,y0,z1},{x1,y1,z0},{x1,y1,z1}};
    }
    done: return idx;
}

// ======================== Main ========================
int main() {
    printf("══════════════════════════════════════════════════════\n");
    printf("  V36 — CWBVH (Compressed Wide BVH) Ray Tracer\n");
    printf("  8-wide nodes, uint8 quantized, Woop triangles\n");
    printf("  Dynamic fetch + triangle postponing\n");
    printf("  Two-phase primary (broadphase root → dense trace)\n");
    printf("══════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    printf("  GPU: %s | SMs: %d | L2: %dKB\n\n",prop.name,prop.multiProcessorCount,(int)(prop.l2CacheSize/1024));

    int NRAYS = 2048*2048; // ~4M
    int side = 2048;

    cudaEvent_t t0,t1; CK(cudaEventCreate(&t0));CK(cudaEventCreate(&t1));

    int triCounts[] = {100000, 500000, 1000000, 2000000};

    for(int si = 0; si < 4; si++) {
        int NTRI = triCounts[si];
        printf("  ━━━ %dK tris ━━━\n", (NTRI+500)/1000);

        Tri* h_tris = (Tri*)malloc(NTRI * sizeof(Tri));
        NTRI = genScene(h_tris, NTRI, 10.f);

        // Build BVH2
        BVH2Builder bvh2; bvh2.build(h_tris, NTRI);
        printf("    BVH2: %d nodes, %d ordered tris\n", (int)bvh2.nodes.size(), (int)bvh2.ordered.size());

        // Collapse to BVH8
        BVH8Builder bvh8(bvh2);
        bvh8.collapse(0);
        printf("    BVH8: %d nodes\n", (int)bvh8.nodes.size());

        // Compress to CWBVH
        CWBVHData cwbvh;
        cwbvh.convert(bvh8, bvh2.ordered.data());
        int numCWBVHNodes = (int)cwbvh.nodeData.size() / 5;
        int numWoopTris = (int)cwbvh.woopTris.size() / 3;
        printf("    CWBVH: %d nodes (%.1f KB), %d Woop tris (%.1f KB)\n",
               numCWBVHNodes, cwbvh.nodeData.size()*16/1024.f,
               numWoopTris, cwbvh.woopTris.size()*16/1024.f);

        // Upload to GPU
        float4 *d_nodes, *d_tris;
        int* d_triIdx;
        CK(cudaMalloc(&d_nodes, cwbvh.nodeData.size()*sizeof(float4)));
        CK(cudaMalloc(&d_tris, cwbvh.woopTris.size()*sizeof(float4)));
        CK(cudaMalloc(&d_triIdx, cwbvh.triIndices.size()*sizeof(int)));
        CK(cudaMemcpy(d_nodes, cwbvh.nodeData.data(), cwbvh.nodeData.size()*sizeof(float4), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_tris, cwbvh.woopTris.data(), cwbvh.woopTris.size()*sizeof(float4), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_triIdx, cwbvh.triIndices.data(), cwbvh.triIndices.size()*sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpyToSymbol(d_cwbvhNodes, &d_nodes, sizeof(float4*)));
        CK(cudaMemcpyToSymbol(d_woopTris, &d_tris, sizeof(float4*)));
        CK(cudaMemcpyToSymbol(d_triIndices, &d_triIdx, sizeof(int*)));

        // Allocate ray/hit buffers
        Ray *d_rays; Hit *d_hits;
        CK(cudaMalloc(&d_rays, NRAYS*sizeof(Ray)));
        CK(cudaMalloc(&d_hits, NRAYS*sizeof(Hit)));

        int *d_survivors, *d_numSurvivors, *d_finishedCount;
        CK(cudaMalloc(&d_survivors, NRAYS*sizeof(int)));
        CK(cudaMalloc(&d_numSurvivors, sizeof(int)));
        CK(cudaMalloc(&d_finishedCount, sizeof(int)));

        float camOx=0,camOy=3,camOz=-15;

        // ─── PRIMARY: Two-Phase ───
        // Warmup
        int zero=0;
        CK(cudaMemcpyToSymbol(g_rayCounter,&zero,4));
        CK(cudaMemcpy(d_numSurvivors,&zero,4,cudaMemcpyHostToDevice));
        broadphaseRootCWBVH<<<640,256>>>(d_survivors,d_numSurvivors,NRAYS,side,camOx,camOy,camOz);
        CK(cudaDeviceSynchronize()); // check broadphase
        int h_numSurv; CK(cudaMemcpy(&h_numSurv,d_numSurvivors,4,cudaMemcpyDeviceToHost));
        printf("    warmup: surv=%d (%.1f%%)\n", h_numSurv, 100.f*h_numSurv/NRAYS);
        generatePrimaryRays<<<(h_numSurv+255)/256,256>>>(d_rays,d_survivors,h_numSurv,side,camOx,camOy,camOz);
        CK(cudaDeviceSynchronize()); // check raygen
        CK(cudaMemcpy(d_finishedCount,&zero,4,cudaMemcpyHostToDevice));
        if(h_numSurv > 0) {
            traceCWBVH<<<dim3(32,32),dim3(32,2)>>>(d_rays,d_hits,h_numSurv,d_finishedCount);
            CK(cudaDeviceSynchronize()); // check trace
        }

        // Benchmark PRIMARY
        int iters=20;
        CK(cudaEventRecord(t0));
        for(int i=0;i<iters;i++){
            CK(cudaMemcpyToSymbol(g_rayCounter,&zero,4));
            CK(cudaMemcpy(d_numSurvivors,&zero,4,cudaMemcpyHostToDevice));
            broadphaseRootCWBVH<<<640,256>>>(d_survivors,d_numSurvivors,NRAYS,side,camOx,camOy,camOz);
            CK(cudaMemcpy(&h_numSurv,d_numSurvivors,4,cudaMemcpyDeviceToHost));
            generatePrimaryRays<<<(h_numSurv+255)/256,256>>>(d_rays,d_survivors,h_numSurv,side,camOx,camOy,camOz);
            CK(cudaMemcpy(d_finishedCount,&zero,4,cudaMemcpyHostToDevice));
            traceCWBVH<<<dim3(32,32),dim3(32,2)>>>(d_rays,d_hits,h_numSurv,d_finishedCount);
        }
        CK(cudaEventRecord(t1));CK(cudaEventSynchronize(t1));
        float priMs; CK(cudaEventElapsedTime(&priMs,t0,t1));
        float priMRs = (float)NRAYS*iters/(priMs*1000.f);
        printf("    PRIMARY (2-phase): %.2fms → %6.0f MR/s  surv:%dk (%.1f%%)\n",
               priMs/iters, priMRs, h_numSurv/1000, 100.f*h_numSurv/NRAYS);

        // ─── DIFFUSE ───
        generateDiffuseRays<<<(NRAYS+255)/256,256>>>(d_rays,NRAYS);
        CK(cudaMemcpy(d_finishedCount,&zero,4,cudaMemcpyHostToDevice));
        traceCWBVH<<<dim3(32,32),dim3(32,2)>>>(d_rays,d_hits,NRAYS,d_finishedCount);
        CK(cudaDeviceSynchronize());

        CK(cudaEventRecord(t0));
        for(int i=0;i<iters;i++){
            CK(cudaMemcpy(d_finishedCount,&zero,4,cudaMemcpyHostToDevice));
            traceCWBVH<<<dim3(32,32),dim3(32,2)>>>(d_rays,d_hits,NRAYS,d_finishedCount);
        }
        CK(cudaEventRecord(t1));CK(cudaEventSynchronize(t1));
        float difMs; CK(cudaEventElapsedTime(&difMs,t0,t1));
        float difMRs = (float)NRAYS*iters/(difMs*1000.f);
        printf("    DIFFUSE: %.2fms → %6.0f MR/s\n", difMs/iters, difMRs);
        printf("    Ratio diff/pri: %.1f%%\n", 100.f*difMRs/priMRs);

        // Reference numbers
        printf("    ── vs OptiX 8.1: pri=%d diff=%d MR/s\n",
               si==0?3642:si==1?3038:si==2?2640:0,
               si==0?551:si==1?336:si==2?292:0);
        printf("    ── vs BVH4 v29:  pri=%d diff=%d MR/s\n\n",
               si==0?6113:si==1?5076:si==2?3992:0,
               si==0?268:si==1?171:si==2?130:0);

        CK(cudaFree(d_nodes));CK(cudaFree(d_tris));CK(cudaFree(d_triIdx));
        CK(cudaFree(d_rays));CK(cudaFree(d_hits));
        CK(cudaFree(d_survivors));CK(cudaFree(d_numSurvivors));CK(cudaFree(d_finishedCount));
        free(h_tris);
    }

    printf("Done.\n");
    return 0;
}
