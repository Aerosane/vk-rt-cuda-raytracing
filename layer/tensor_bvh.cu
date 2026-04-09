/* tensor_bvh.cu — FP16 accelerated BVH4 traversal prototype for V100
 *
 * Key finding: WMMA (tensor cores) CANNOT accelerate AABB slab tests.
 * Slab test = per-axis element-wise multiply: t = invDir * (bound - origin)
 * WMMA computes C = A×B which SUMS over the inner dimension (dot product).
 * We need 6 independent products per child, not accumulated sums.
 *
 * What DOES work: V100 FP16 scalar ALU runs at 2× rate (31.4 vs 15.7 TFLOPS).
 * BVH4 nodes already store bounds as FP16 — keep AABB tests in FP16 natively.
 *
 * Three kernels benchmarked:
 *   1. FP32 baseline — standard FP32 slab test (reference)
 *   2. FP16 scalar  — FP16 slab at 2× ALU rate on V100
 *   3. FP16 + smem  — top BVH levels cached in shared memory
 *
 * Build: nvcc -O3 -arch=sm_70 tensor_bvh.cu -o tensor_bvh_test
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cfloat>

#define CK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));}}while(0)

// ═══════════════════════════════════════════════════════════════════
// Data structures — compatible with cuda_bvh_backend.cu
// ═══════════════════════════════════════════════════════════════════
struct float3a { float x,y,z; };
struct Tri { float3a v0,v1,v2; };
struct AABB { float3a mn,mx; };
struct BVH4Node {
    __half boundsX[8], boundsY[8], boundsZ[8]; // [0..3]=min, [4..7]=max
    int child[4];
};

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: FP32 BASELINE — standard slab test for reference
// Converts FP16 bounds to FP32, all arithmetic in FP32. 15.7 TFLOPS.
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(256, 5) bvh4Traverse_FP32(
    const int4* __restrict__ bvh4, int numNodes,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const float* __restrict__ rayOx, const float* __restrict__ rayOy, const float* __restrict__ rayOz,
    const float* __restrict__ rayIx, const float* __restrict__ rayIy, const float* __restrict__ rayIz,
    const float* __restrict__ rayDx, const float* __restrict__ rayDy, const float* __restrict__ rayDz,
    float* __restrict__ outHitT, int numRays)
{
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    const int totalWarps = (gridDim.x * blockDim.x) / 32;

    for (int batch = warpId * 32; batch < numRays; batch += totalWarps * 32) {
        float my_ox=0,my_oy=0,my_oz=0, my_ix=0,my_iy=0,my_iz=0, my_dx=0,my_dy=0,my_dz=0;
        float my_bestT = 1e30f;
        int my_rayIdx = -1;
        if ((batch+lane) < numRays) {
            int ri = batch+lane; my_rayIdx = ri;
            my_ox=rayOx[ri]; my_oy=rayOy[ri]; my_oz=rayOz[ri];
            my_ix=rayIx[ri]; my_iy=rayIy[ri]; my_iz=rayIz[ri];
            my_dx=rayDx[ri]; my_dy=rayDy[ri]; my_dz=rayDz[ri];
        }

        int stack[16]; int sp = 0;
        stack[sp++] = 0;

        while (sp > 0) {
            int nodeIdx = stack[--sp];
            if (nodeIdx < 0) {
                if (my_rayIdx >= 0) {
                    int enc = -(nodeIdx+2);
                    int triStart = enc>>3, triCount = (enc&7)+1;
                    for (int t = 0; t < triCount; t++) {
                        int ti = triStart+t;
                        float v0x=__ldg(&tv0x[ti]),v0y=__ldg(&tv0y[ti]),v0z=__ldg(&tv0z[ti]);
                        float e1x=__ldg(&tv1x[ti])-v0x,e1y=__ldg(&tv1y[ti])-v0y,e1z=__ldg(&tv1z[ti])-v0z;
                        float e2x=__ldg(&tv2x[ti])-v0x,e2y=__ldg(&tv2y[ti])-v0y,e2z=__ldg(&tv2z[ti])-v0z;
                        float px=my_dy*e2z-my_dz*e2y, py=my_dz*e2x-my_dx*e2z, pz=my_dx*e2y-my_dy*e2x;
                        float det=e1x*px+e1y*py+e1z*pz;
                        if(fabsf(det)<1e-12f)continue;
                        float inv=1.f/det;
                        float tx=my_ox-v0x,ty=my_oy-v0y,tz=my_oz-v0z;
                        float u=inv*(tx*px+ty*py+tz*pz); if(u<0.f||u>1.f)continue;
                        float qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x;
                        float v=inv*(my_dx*qx+my_dy*qy+my_dz*qz); if(v<0.f||u+v>1.f)continue;
                        float tt=inv*(e2x*qx+e2y*qy+e2z*qz);
                        if(tt>0.f&&tt<my_bestT) my_bestT=tt;
                    }
                }
                continue;
            }

            // Load BVH4 node — bounds in FP16, CONVERT to FP32 for AABB test
            int4 n0=__ldg(&bvh4[nodeIdx*4]);
            int4 n1=__ldg(&bvh4[nodeIdx*4+1]);
            int4 n2=__ldg(&bvh4[nodeIdx*4+2]);
            int4 n3=__ldg(&bvh4[nodeIdx*4+3]);
            const __half* bx=(const __half*)&n0;
            const __half* by=(const __half*)&n1;
            const __half* bz=(const __half*)&n2;
            const int* ch=(const int*)&n3;

            float childDist[4]; int childIdx[4];
            if (my_rayIdx >= 0) {
                for (int c = 0; c < 4; c++) {
                    childIdx[c] = ch[c];
                    if (ch[c] == -1) { childDist[c] = 1e30f; continue; }
                    // FP32 slab test — convert FP16 bounds to FP32
                    float bmnx=__half2float(bx[c]),   bmxx=__half2float(bx[4+c]);
                    float bmny=__half2float(by[c]),   bmxy=__half2float(by[4+c]);
                    float bmnz=__half2float(bz[c]),   bmxz=__half2float(bz[4+c]);
                    float t1x=(bmnx-my_ox)*my_ix, t2x=(bmxx-my_ox)*my_ix;
                    float t1y=(bmny-my_oy)*my_iy, t2y=(bmxy-my_oy)*my_iy;
                    float t1z=(bmnz-my_oz)*my_iz, t2z=(bmxz-my_oz)*my_iz;
                    float tNear=fmaxf(fmaxf(fminf(t1x,t2x),fminf(t1y,t2y)),fminf(t1z,t2z));
                    float tFar =fminf(fminf(fmaxf(t1x,t2x),fmaxf(t1y,t2y)),fmaxf(t1z,t2z));
                    childDist[c]=(tNear<=tFar&&tFar>0.f&&tNear<my_bestT)?tNear:1e30f;
                }
            } else { childDist[0]=childDist[1]=childDist[2]=childDist[3]=1e30f; childIdx[0]=childIdx[1]=childIdx[2]=childIdx[3]=-1; }

            #define CSWAP(a,b) do{float da=childDist[a],db=childDist[b];int ca=childIdx[a],cb=childIdx[b];bool s=(da>db);childDist[a]=s?db:da;childDist[b]=s?da:db;childIdx[a]=s?cb:ca;childIdx[b]=s?ca:cb;}while(0)
            CSWAP(0,1);CSWAP(2,3);CSWAP(0,2);CSWAP(1,3);CSWAP(1,2);
            #undef CSWAP

            unsigned h0=__ballot_sync(0xFFFFFFFF,childDist[0]<1e30f);
            unsigned h1=__ballot_sync(0xFFFFFFFF,childDist[1]<1e30f);
            unsigned h2=__ballot_sync(0xFFFFFFFF,childDist[2]<1e30f);
            unsigned h3=__ballot_sync(0xFFFFFFFF,childDist[3]<1e30f);
            if(h3&&sp<16){int ci=__shfl_sync(0xFFFFFFFF,childIdx[3],0);if(ci!=-1)stack[sp++]=ci;}
            if(h2&&sp<16){int ci=__shfl_sync(0xFFFFFFFF,childIdx[2],0);if(ci!=-1)stack[sp++]=ci;}
            if(h1&&sp<16){int ci=__shfl_sync(0xFFFFFFFF,childIdx[1],0);if(ci!=-1)stack[sp++]=ci;}
            if(h0&&sp<16){int ci=__shfl_sync(0xFFFFFFFF,childIdx[0],0);if(ci!=-1)stack[sp++]=ci;}
        }
        if (my_rayIdx >= 0) outHitT[my_rayIdx] = my_bestT;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: FP16 SCALAR — keep AABB tests in FP16 natively
// V100 FP16 ALU runs at 2× rate (31.4 TFLOPS vs 15.7 FP32).
// BVH4 bounds already FP16 — no conversion needed.
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(256, 5) tensorBVH4Traverse(
    const int4* __restrict__ bvh4, int numNodes,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    // Ray data (SoA)
    const float* __restrict__ rayOx, const float* __restrict__ rayOy, const float* __restrict__ rayOz,
    const float* __restrict__ rayIx, const float* __restrict__ rayIy, const float* __restrict__ rayIz,
    const float* __restrict__ rayDx, const float* __restrict__ rayDy, const float* __restrict__ rayDz,
    float* __restrict__ outHitT,
    int numRays)
{
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    const int totalWarps = (gridDim.x * blockDim.x) / 32;

    // Each warp processes 32 rays (all lanes active!)
    for (int batch = warpId * 32; batch < numRays; batch += totalWarps * 32) {
        // ─── All 32 lanes load a ray ───
        float my_ox = 0, my_oy = 0, my_oz = 0;
        float my_ix = 0, my_iy = 0, my_iz = 0;
        float my_dx = 0, my_dy = 0, my_dz = 0;
        float my_bestT = 1e30f;
        int   my_rayIdx = -1;

        if ((batch + lane) < numRays) {
            int ri = batch + lane;
            my_rayIdx = ri;
            my_ox = rayOx[ri]; my_oy = rayOy[ri]; my_oz = rayOz[ri];
            my_ix = rayIx[ri]; my_iy = rayIy[ri]; my_iz = rayIz[ri];
            my_dx = rayDx[ri]; my_dy = rayDy[ri]; my_dz = rayDz[ri];
            my_bestT = 1e30f;
        }

        // ─── BVH4 traversal with tensor-accelerated AABB tests ───
        int stack[16];
        int sp = 0;
        stack[sp++] = 0; // root

        while (sp > 0) {
            int nodeIdx = stack[--sp];

            if (nodeIdx < 0) {
                // ─── LEAF: Möller-Trumbore on CUDA cores (scalar) ───
                if (my_rayIdx >= 0) {
                    int enc = -(nodeIdx + 2);
                    int triStart = enc >> 3, triCount = (enc & 7) + 1;
                    for (int t = 0; t < triCount; t++) {
                        int ti = triStart + t;
                        float v0x = __ldg(&tv0x[ti]), v0y = __ldg(&tv0y[ti]), v0z = __ldg(&tv0z[ti]);
                        float e1x = __ldg(&tv1x[ti]) - v0x, e1y = __ldg(&tv1y[ti]) - v0y, e1z = __ldg(&tv1z[ti]) - v0z;
                        float e2x = __ldg(&tv2x[ti]) - v0x, e2y = __ldg(&tv2y[ti]) - v0y, e2z = __ldg(&tv2z[ti]) - v0z;
                        float px = my_dy*e2z - my_dz*e2y, py = my_dz*e2x - my_dx*e2z, pz = my_dx*e2y - my_dy*e2x;
                        float det = e1x*px + e1y*py + e1z*pz;
                        if (fabsf(det) < 1e-12f) continue;
                        float inv = 1.f / det;
                        float tx = my_ox - v0x, ty = my_oy - v0y, tz = my_oz - v0z;
                        float u = inv * (tx*px + ty*py + tz*pz);
                        if (u < 0.f || u > 1.f) continue;
                        float qx = ty*e1z - tz*e1y, qy = tz*e1x - tx*e1z, qz = tx*e1y - ty*e1x;
                        float v = inv * (my_dx*qx + my_dy*qy + my_dz*qz);
                        if (v < 0.f || u + v > 1.f) continue;
                        float tt = inv * (e2x*qx + e2y*qy + e2z*qz);
                        if (tt > 0.f && tt < my_bestT) my_bestT = tt;
                    }
                }
                continue;
            }

            // ─── INNER NODE: Tensor Core AABB slab test ───
            // Load BVH4 node (all lanes read — broadcast via L1)
            int4 n0 = __ldg(&bvh4[nodeIdx*4]);
            int4 n1 = __ldg(&bvh4[nodeIdx*4+1]);
            int4 n2 = __ldg(&bvh4[nodeIdx*4+2]);
            int4 n3 = __ldg(&bvh4[nodeIdx*4+3]);

            const __half* bx = (const __half*)&n0;
            const __half* by = (const __half*)&n1;
            const __half* bz = (const __half*)&n2;
            const int*    ch = (const int*)&n3;

            // ─── half2 packed AABB slab test — TRUE 2× throughput ───
            // V100 FP16 2× rate requires half2 ops (__hmul2, __hsub2)
            // Pack (bmin, bmax) per child per axis into half2 → 2 slab values per op
            float childDist[4];
            int   childIdx[4];

            if (my_rayIdx >= 0) {
                // Pack ray data as half2 (replicated for both min/max bounds)
                half2 h2_ix = __float2half2_rn(my_ix);
                half2 h2_iy = __float2half2_rn(my_iy);
                half2 h2_iz = __float2half2_rn(my_iz);
                half2 h2_ox = __float2half2_rn(my_ox);
                half2 h2_oy = __float2half2_rn(my_oy);
                half2 h2_oz = __float2half2_rn(my_oz);

                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    childIdx[c] = ch[c];
                    if (ch[c] == -1) { childDist[c] = 1e30f; continue; }

                    // Pack (bmin, bmax) for this child — ONE half2 per axis
                    half2 bnd_x = __halves2half2(bx[c], bx[4+c]);
                    half2 bnd_y = __halves2half2(by[c], by[4+c]);
                    half2 bnd_z = __halves2half2(bz[c], bz[4+c]);

                    // t = (bound - origin) * invDir — 3 half2 muls + 3 subs
                    // Each computes BOTH t_min_bound AND t_max_bound simultaneously
                    half2 t_x = __hmul2(__hsub2(bnd_x, h2_ox), h2_ix);
                    half2 t_y = __hmul2(__hsub2(bnd_y, h2_oy), h2_iy);
                    half2 t_z = __hmul2(__hsub2(bnd_z, h2_oz), h2_iz);

                    // Extract to FP32 for min/max reduction (needs precision)
                    float t1x = __low2float(t_x),  t2x = __high2float(t_x);
                    float t1y = __low2float(t_y),  t2y = __high2float(t_y);
                    float t1z = __low2float(t_z),  t2z = __high2float(t_z);

                    float tNear = fmaxf(fmaxf(fminf(t1x,t2x), fminf(t1y,t2y)), fminf(t1z,t2z));
                    float tFar  = fminf(fminf(fmaxf(t1x,t2x), fmaxf(t1y,t2y)), fmaxf(t1z,t2z));

                    childDist[c] = (tNear <= tFar && tFar > 0.f && tNear < my_bestT) ? tNear : 1e30f;
                }
            } else {
                childDist[0] = childDist[1] = childDist[2] = childDist[3] = 1e30f;
                childIdx[0] = childIdx[1] = childIdx[2] = childIdx[3] = -1;
            }

            // ─── Sort children front-to-back (network sort) ───
            #define CSWAP(a,b) do { float da=childDist[a],db=childDist[b]; \
                int ca=childIdx[a],cb=childIdx[b]; bool s=(da>db); \
                childDist[a]=s?db:da; childDist[b]=s?da:db; \
                childIdx[a]=s?cb:ca; childIdx[b]=s?ca:cb; } while(0)
            CSWAP(0,1); CSWAP(2,3); CSWAP(0,2); CSWAP(1,3); CSWAP(1,2);
            #undef CSWAP

            // ─── Warp-vote: which children are hit by ANY ray? ───
            // If no ray in the warp hits a child, skip it entirely
            unsigned anyHit0 = __ballot_sync(0xFFFFFFFF, childDist[0] < 1e30f);
            unsigned anyHit1 = __ballot_sync(0xFFFFFFFF, childDist[1] < 1e30f);
            unsigned anyHit2 = __ballot_sync(0xFFFFFFFF, childDist[2] < 1e30f);
            unsigned anyHit3 = __ballot_sync(0xFFFFFFFF, childDist[3] < 1e30f);

            // Push children in reverse order (closest last = processed first)
            if (anyHit3 && sp < 16) {
                // Broadcast childIdx from lane 0 (all lanes agree on node structure)
                int ci = __shfl_sync(0xFFFFFFFF, childIdx[3], 0);
                if (ci != -1) stack[sp++] = ci;
            }
            if (anyHit2 && sp < 16) {
                int ci = __shfl_sync(0xFFFFFFFF, childIdx[2], 0);
                if (ci != -1) stack[sp++] = ci;
            }
            if (anyHit1 && sp < 16) {
                int ci = __shfl_sync(0xFFFFFFFF, childIdx[1], 0);
                if (ci != -1) stack[sp++] = ci;
            }
            if (anyHit0 && sp < 16) {
                int ci = __shfl_sync(0xFFFFFFFF, childIdx[0], 0);
                if (ci != -1) stack[sp++] = ci;
            }
        }

        // ─── Write results ───
        if (my_rayIdx >= 0) {
            outHitT[my_rayIdx] = my_bestT;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: FP16 + Shared Memory Root Cache
// Top 3 BVH levels (~85 nodes, ~5KB) cached in shared memory.
// Eliminates L2 misses for the root region hit by ALL rays.
// ═══════════════════════════════════════════════════════════════════
#define SMEM_ROOT_NODES 85  // 1+4+16+64 = 85 BVH4 nodes (4 levels)
#define SMEM_ROOT_BYTES (SMEM_ROOT_NODES * 64)  // 5440 bytes

__global__ void __launch_bounds__(256, 4) bvh4Traverse_FP16_smem(
    const int4* __restrict__ bvh4, int numNodes,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const float* __restrict__ rayOx, const float* __restrict__ rayOy, const float* __restrict__ rayOz,
    const float* __restrict__ rayIx, const float* __restrict__ rayIy, const float* __restrict__ rayIz,
    const float* __restrict__ rayDx, const float* __restrict__ rayDy, const float* __restrict__ rayDz,
    float* __restrict__ outHitT, int numRays)
{
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    const int totalWarps = (gridDim.x * blockDim.x) / 32;

    // Collaboratively load top BVH levels into shared memory
    __shared__ int4 smemBVH[SMEM_ROOT_NODES * 4];
    int loadCount = min(SMEM_ROOT_NODES * 4, numNodes * 4);
    for (int i = threadIdx.x; i < loadCount; i += blockDim.x)
        smemBVH[i] = bvh4[i];
    __syncthreads();

    for (int batch = warpId * 32; batch < numRays; batch += totalWarps * 32) {
        float my_ox=0,my_oy=0,my_oz=0, my_ix=0,my_iy=0,my_iz=0, my_dx=0,my_dy=0,my_dz=0;
        float my_bestT = 1e30f;
        int my_rayIdx = -1;
        if ((batch+lane) < numRays) {
            int ri = batch+lane; my_rayIdx = ri;
            my_ox=rayOx[ri]; my_oy=rayOy[ri]; my_oz=rayOz[ri];
            my_ix=rayIx[ri]; my_iy=rayIy[ri]; my_iz=rayIz[ri];
            my_dx=rayDx[ri]; my_dy=rayDy[ri]; my_dz=rayDz[ri];
        }

        int stack[16]; int sp = 0;
        stack[sp++] = 0;

        while (sp > 0) {
            int nodeIdx = stack[--sp];
            if (nodeIdx < 0) {
                if (my_rayIdx >= 0) {
                    int enc = -(nodeIdx+2);
                    int triStart = enc>>3, triCount = (enc&7)+1;
                    for (int t = 0; t < triCount; t++) {
                        int ti = triStart+t;
                        float v0x=__ldg(&tv0x[ti]),v0y=__ldg(&tv0y[ti]),v0z=__ldg(&tv0z[ti]);
                        float e1x=__ldg(&tv1x[ti])-v0x,e1y=__ldg(&tv1y[ti])-v0y,e1z=__ldg(&tv1z[ti])-v0z;
                        float e2x=__ldg(&tv2x[ti])-v0x,e2y=__ldg(&tv2y[ti])-v0y,e2z=__ldg(&tv2z[ti])-v0z;
                        float px=my_dy*e2z-my_dz*e2y, py=my_dz*e2x-my_dx*e2z, pz=my_dx*e2y-my_dy*e2x;
                        float det=e1x*px+e1y*py+e1z*pz;
                        if(fabsf(det)<1e-12f)continue;
                        float inv=1.f/det;
                        float tx=my_ox-v0x,ty=my_oy-v0y,tz=my_oz-v0z;
                        float u=inv*(tx*px+ty*py+tz*pz); if(u<0.f||u>1.f)continue;
                        float qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x;
                        float v=inv*(my_dx*qx+my_dy*qy+my_dz*qz); if(v<0.f||u+v>1.f)continue;
                        float tt=inv*(e2x*qx+e2y*qy+e2z*qz);
                        if(tt>0.f&&tt<my_bestT) my_bestT=tt;
                    }
                }
                continue;
            }

            // Read from shared memory if in cached region, else global
            int4 n0,n1,n2,n3;
            if (nodeIdx < SMEM_ROOT_NODES) {
                n0=smemBVH[nodeIdx*4]; n1=smemBVH[nodeIdx*4+1];
                n2=smemBVH[nodeIdx*4+2]; n3=smemBVH[nodeIdx*4+3];
            } else {
                n0=__ldg(&bvh4[nodeIdx*4]); n1=__ldg(&bvh4[nodeIdx*4+1]);
                n2=__ldg(&bvh4[nodeIdx*4+2]); n3=__ldg(&bvh4[nodeIdx*4+3]);
            }
            const __half* bx=(const __half*)&n0;
            const __half* by=(const __half*)&n1;
            const __half* bz=(const __half*)&n2;
            const int* ch=(const int*)&n3;

            float childDist[4]; int childIdx[4];
            if (my_rayIdx >= 0) {
                half2 h2_ix=__float2half2_rn(my_ix), h2_iy=__float2half2_rn(my_iy), h2_iz=__float2half2_rn(my_iz);
                half2 h2_ox=__float2half2_rn(my_ox), h2_oy=__float2half2_rn(my_oy), h2_oz=__float2half2_rn(my_oz);
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    childIdx[c] = ch[c];
                    if (ch[c] == -1) { childDist[c] = 1e30f; continue; }
                    half2 t_x=__hmul2(__hsub2(__halves2half2(bx[c],bx[4+c]),h2_ox),h2_ix);
                    half2 t_y=__hmul2(__hsub2(__halves2half2(by[c],by[4+c]),h2_oy),h2_iy);
                    half2 t_z=__hmul2(__hsub2(__halves2half2(bz[c],bz[4+c]),h2_oz),h2_iz);
                    float t1x=__low2float(t_x),t2x=__high2float(t_x);
                    float t1y=__low2float(t_y),t2y=__high2float(t_y);
                    float t1z=__low2float(t_z),t2z=__high2float(t_z);
                    float tNear=fmaxf(fmaxf(fminf(t1x,t2x),fminf(t1y,t2y)),fminf(t1z,t2z));
                    float tFar =fminf(fminf(fmaxf(t1x,t2x),fmaxf(t1y,t2y)),fmaxf(t1z,t2z));
                    childDist[c]=(tNear<=tFar&&tFar>0.f&&tNear<my_bestT)?tNear:1e30f;
                }
            } else { childDist[0]=childDist[1]=childDist[2]=childDist[3]=1e30f; childIdx[0]=childIdx[1]=childIdx[2]=childIdx[3]=-1; }

            #define CSWAP(a,b) do{float da=childDist[a],db=childDist[b];int ca=childIdx[a],cb=childIdx[b];bool s=(da>db);childDist[a]=s?db:da;childDist[b]=s?da:db;childIdx[a]=s?cb:ca;childIdx[b]=s?ca:cb;}while(0)
            CSWAP(0,1);CSWAP(2,3);CSWAP(0,2);CSWAP(1,3);CSWAP(1,2);
            #undef CSWAP

            unsigned h0=__ballot_sync(0xFFFFFFFF,childDist[0]<1e30f);
            unsigned h1=__ballot_sync(0xFFFFFFFF,childDist[1]<1e30f);
            unsigned h2=__ballot_sync(0xFFFFFFFF,childDist[2]<1e30f);
            unsigned h3=__ballot_sync(0xFFFFFFFF,childDist[3]<1e30f);
            if(h3&&sp<16){int ci=__shfl_sync(0xFFFFFFFF,childIdx[3],0);if(ci!=-1)stack[sp++]=ci;}
            if(h2&&sp<16){int ci=__shfl_sync(0xFFFFFFFF,childIdx[2],0);if(ci!=-1)stack[sp++]=ci;}
            if(h1&&sp<16){int ci=__shfl_sync(0xFFFFFFFF,childIdx[1],0);if(ci!=-1)stack[sp++]=ci;}
            if(h0&&sp<16){int ci=__shfl_sync(0xFFFFFFFF,childIdx[0],0);if(ci!=-1)stack[sp++]=ci;}
        }
        if (my_rayIdx >= 0) outHitT[my_rayIdx] = my_bestT;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Test harness — Conference scene benchmark
// ═══════════════════════════════════════════════════════════════════

// Simple BVH2 builder (same as cuda_bvh_backend.cu)
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
        BVH2Node nd; nd.triStart=nd.triCount=nd.left=nd.right=0;
        nd.box=primBB[idx[s]]; for(int i=s+1;i<e;i++)nd.box=mergeAABB(nd.box,primBB[idx[i]]);
        int cnt=e-s;
        if(cnt<=3){ nd.triStart=(int)ordered.size();nd.triCount=cnt;
            for(int i=s;i<e;i++) ordered.push_back(src[idx[i]]);
            nodes.push_back(nd);return(int)nodes.size()-1;}
        float bestCost=1e30f;int bestAxis=0,bestSplit=s+cnt/2;float pA=saArea(nd.box);
        for(int ax=0;ax<3;ax++){
            float cmin=1e30f,cmax=-1e30f;
            for(int i=s;i<e;i++){float c=(&centroids[idx[i]].x)[ax];cmin=fminf(cmin,c);cmax=fmaxf(cmax,c);}
            if(cmax-cmin<1e-8f)continue;
            const int NB=16;AABB lBox[NB];int lCnt[NB];AABB rBox[NB];int rCnt[NB];
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

// Generate test scene: random triangles in a unit cube
static void generateScene(std::vector<Tri>& tris, int count) {
    tris.resize(count);
    srand(42);
    for (int i = 0; i < count; i++) {
        float cx = (rand()%1000)/1000.f, cy = (rand()%1000)/1000.f, cz = (rand()%1000)/1000.f;
        float sz = 0.01f + (rand()%100)/10000.f;
        tris[i].v0 = {cx-sz, cy-sz, cz};
        tris[i].v1 = {cx+sz, cy-sz, cz+sz};
        tris[i].v2 = {cx, cy+sz, cz-sz};
    }
}

int main() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Tensor Core BVH Traversal — V100 Prototype\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    // Generate scene
    const int NUM_TRIS = 100000;  // Change for scaling tests
    const int SIDE = 1024;  // 1024×1024 = 1M rays
    const int NUM_RAYS = SIDE * SIDE;

    printf("[BUILD] Generating %d triangles...\n", NUM_TRIS);
    std::vector<Tri> tris;
    generateScene(tris, NUM_TRIS);

    // Build BVH4
    BVH2Builder bvh2;
    bvh2.build(tris.data(), NUM_TRIS);
    int numOrd = (int)bvh2.ordered.size();
    int maxB4 = (int)bvh2.nodes.size();
    BVH4Node* h_bvh4 = (BVH4Node*)calloc(maxB4, sizeof(BVH4Node));
    int numB4 = 0;
    collapseToB4(bvh2, 0, h_bvh4, numB4);
    printf("[BUILD] BVH4: %d nodes, %d ordered tris\n", numB4, numOrd);

    // Build SoA triangles
    float *h0x=(float*)malloc(numOrd*4),*h0y=(float*)malloc(numOrd*4),*h0z=(float*)malloc(numOrd*4);
    float *h1x=(float*)malloc(numOrd*4),*h1y=(float*)malloc(numOrd*4),*h1z=(float*)malloc(numOrd*4);
    float *h2x=(float*)malloc(numOrd*4),*h2y=(float*)malloc(numOrd*4),*h2z=(float*)malloc(numOrd*4);
    for(int i=0;i<numOrd;i++){
        h0x[i]=bvh2.ordered[i].v0.x;h0y[i]=bvh2.ordered[i].v0.y;h0z[i]=bvh2.ordered[i].v0.z;
        h1x[i]=bvh2.ordered[i].v1.x;h1y[i]=bvh2.ordered[i].v1.y;h1z[i]=bvh2.ordered[i].v1.z;
        h2x[i]=bvh2.ordered[i].v2.x;h2y[i]=bvh2.ordered[i].v2.y;h2z[i]=bvh2.ordered[i].v2.z;
    }

    // Upload to GPU
    int4* d_bvh4; float *d_t0x,*d_t0y,*d_t0z,*d_t1x,*d_t1y,*d_t1z,*d_t2x,*d_t2y,*d_t2z;
    CK(cudaMalloc(&d_bvh4, numB4*4*sizeof(int4)));
    CK(cudaMemcpy(d_bvh4, h_bvh4, numB4*4*sizeof(int4), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_t0x,numOrd*4));CK(cudaMalloc(&d_t0y,numOrd*4));CK(cudaMalloc(&d_t0z,numOrd*4));
    CK(cudaMalloc(&d_t1x,numOrd*4));CK(cudaMalloc(&d_t1y,numOrd*4));CK(cudaMalloc(&d_t1z,numOrd*4));
    CK(cudaMalloc(&d_t2x,numOrd*4));CK(cudaMalloc(&d_t2y,numOrd*4));CK(cudaMalloc(&d_t2z,numOrd*4));
    CK(cudaMemcpy(d_t0x,h0x,numOrd*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_t0y,h0y,numOrd*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_t0z,h0z,numOrd*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_t1x,h1x,numOrd*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_t1y,h1y,numOrd*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_t1z,h1z,numOrd*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_t2x,h2x,numOrd*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_t2y,h2y,numOrd*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_t2z,h2z,numOrd*4,cudaMemcpyHostToDevice));

    // Generate camera rays (SoA)
    float camX = 0.5f, camY = 0.5f, camZ = 2.0f;
    float *h_rox = (float*)malloc(NUM_RAYS*4), *h_roy = (float*)malloc(NUM_RAYS*4), *h_roz = (float*)malloc(NUM_RAYS*4);
    float *h_rdx = (float*)malloc(NUM_RAYS*4), *h_rdy = (float*)malloc(NUM_RAYS*4), *h_rdz = (float*)malloc(NUM_RAYS*4);
    float *h_rix = (float*)malloc(NUM_RAYS*4), *h_riy = (float*)malloc(NUM_RAYS*4), *h_riz = (float*)malloc(NUM_RAYS*4);

    for (int i = 0; i < NUM_RAYS; i++) {
        int px = i % SIDE, py = i / SIDE;
        float u = (px+0.5f)/SIDE*2.f-1.f, v = (py+0.5f)/SIDE*2.f-1.f;
        float rlen = 1.f/sqrtf(u*u+v*v+1.f);
        h_rox[i] = camX; h_roy[i] = camY; h_roz[i] = camZ;
        h_rdx[i] = u*rlen; h_rdy[i] = v*rlen; h_rdz[i] = -rlen;  // look toward scene (z=-1)
        float dx = h_rdx[i], dy = h_rdy[i], dz = h_rdz[i];
        h_rix[i] = 1.f/(fabsf(dx)>1e-8f?dx:copysignf(1e-8f,dx));
        h_riy[i] = 1.f/(fabsf(dy)>1e-8f?dy:copysignf(1e-8f,dy));
        h_riz[i] = 1.f/(fabsf(dz)>1e-8f?dz:copysignf(1e-8f,dz));
    }

    float *d_rox,*d_roy,*d_roz,*d_rdx,*d_rdy,*d_rdz,*d_rix,*d_riy,*d_riz,*d_hitT;
    CK(cudaMalloc(&d_rox,NUM_RAYS*4));CK(cudaMalloc(&d_roy,NUM_RAYS*4));CK(cudaMalloc(&d_roz,NUM_RAYS*4));
    CK(cudaMalloc(&d_rdx,NUM_RAYS*4));CK(cudaMalloc(&d_rdy,NUM_RAYS*4));CK(cudaMalloc(&d_rdz,NUM_RAYS*4));
    CK(cudaMalloc(&d_rix,NUM_RAYS*4));CK(cudaMalloc(&d_riy,NUM_RAYS*4));CK(cudaMalloc(&d_riz,NUM_RAYS*4));
    CK(cudaMalloc(&d_hitT,NUM_RAYS*4));
    CK(cudaMemcpy(d_rox,h_rox,NUM_RAYS*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_roy,h_roy,NUM_RAYS*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_roz,h_roz,NUM_RAYS*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rdx,h_rdx,NUM_RAYS*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_rdy,h_rdy,NUM_RAYS*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_rdz,h_rdz,NUM_RAYS*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rix,h_rix,NUM_RAYS*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_riy,h_riy,NUM_RAYS*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_riz,h_riz,NUM_RAYS*4,cudaMemcpyHostToDevice));

    CK(cudaDeviceSynchronize());

    printf("\n[BENCH] 3-kernel comparison: %dK tris, %dM rays\n",
           NUM_TRIS/1000, NUM_RAYS/1000000);
    printf("  %-18s %6s    %9s  %s\n", "Kernel", "Time", "Throughput", "Hits");
    printf("  ─────────────────────────────────────────────────\n");

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    float* h_hitT = (float*)malloc(NUM_RAYS*4);
    float ms=0, times[3]; int hits[3];
    int nR = NUM_RAYS;

    auto runBench = [&](const char* name, int idx, auto launcher) {
        launcher(); CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(t0)); launcher(); CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1)); CK(cudaEventElapsedTime(&ms, t0, t1));
        CK(cudaMemcpy(h_hitT, d_hitT, nR*4, cudaMemcpyDeviceToHost));
        int hc=0; for(int i=0;i<nR;i++) if(h_hitT[i]<1e20f) hc++;
        float mr=nR/(ms*1000.f);
        printf("  %-18s %6.2f ms → %7.0f MR/s  (%d hits)\n",name,ms,mr,hc);
        times[idx]=ms; hits[idx]=hc;
    };

    // Kernel 1: FP32 baseline
    runBench("FP32 baseline", 0, [&](){
        bvh4Traverse_FP32<<<640,256>>>(d_bvh4,numB4*4,
            d_t0x,d_t0y,d_t0z,d_t1x,d_t1y,d_t1z,d_t2x,d_t2y,d_t2z,
            d_rox,d_roy,d_roz,d_rix,d_riy,d_riz,d_rdx,d_rdy,d_rdz,
            d_hitT,nR);
    });

    // Kernel 2: FP16 scalar (2× ALU rate)
    runBench("FP16 scalar (2x)", 1, [&](){
        tensorBVH4Traverse<<<640,256>>>(d_bvh4,numB4*4,
            d_t0x,d_t0y,d_t0z,d_t1x,d_t1y,d_t1z,d_t2x,d_t2y,d_t2z,
            d_rox,d_roy,d_roz,d_rix,d_riy,d_riz,d_rdx,d_rdy,d_rdz,
            d_hitT,nR);
    });

    // Kernel 3: FP16 + shared memory root cache
    runBench("FP16+smem root", 2, [&](){
        bvh4Traverse_FP16_smem<<<640,256>>>(d_bvh4,numB4*4,
            d_t0x,d_t0y,d_t0z,d_t1x,d_t1y,d_t1z,d_t2x,d_t2y,d_t2z,
            d_rox,d_roy,d_roz,d_rix,d_riy,d_riz,d_rdx,d_rdy,d_rdz,
            d_hitT,nR);
    });

    // ─── Summary ───
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  RESULTS: %dK tris, %dM rays, V100 sm_70\n", NUM_TRIS/1000, NUM_RAYS/1000000);
    printf("  ─────────────────────────────────────────────────\n");
    printf("  FP32 baseline:   %6.2f ms → %7.0f MR/s\n", times[0], nR/(times[0]*1000.f));
    printf("  FP16 scalar:     %6.2f ms → %7.0f MR/s  (%.1fx vs FP32)\n",
           times[1], nR/(times[1]*1000.f), times[0]/times[1]);
    printf("  FP16+smem:       %6.2f ms → %7.0f MR/s  (%.1fx vs FP32)\n",
           times[2], nR/(times[2]*1000.f), times[0]/times[2]);
    printf("  ─────────────────────────────────────────────────\n");
    bool allMatch = (hits[0]==hits[1] && hits[1]==hits[2]);
    printf("  Hit counts: %d / %d / %d  %s\n", hits[0], hits[1], hits[2],
           allMatch ? "✅ ALL MATCH" : "⚠️  MISMATCH");
    printf("\n  NOTE: WMMA tensor cores CANNOT accelerate AABB slab tests.\n");
    printf("  Slab test = per-axis element-wise product (not a dot product).\n");
    printf("  WMMA computes C=A×B with inner-dim SUM — wrong operation.\n");
    printf("  Real V100 RT win: FP16 2× ALU rate + smem root caching.\n");
    printf("═══════════════════════════════════════════════════════\n");

    // Cleanup
    free(h_bvh4); free(h0x);free(h0y);free(h0z);free(h1x);free(h1y);free(h1z);free(h2x);free(h2y);free(h2z);
    free(h_rox);free(h_roy);free(h_roz);free(h_rdx);free(h_rdy);free(h_rdz);free(h_rix);free(h_riy);free(h_riz);
    free(h_hitT);
    cudaFree(d_bvh4);cudaFree(d_t0x);cudaFree(d_t0y);cudaFree(d_t0z);
    cudaFree(d_t1x);cudaFree(d_t1y);cudaFree(d_t1z);cudaFree(d_t2x);cudaFree(d_t2y);cudaFree(d_t2z);
    cudaFree(d_rox);cudaFree(d_roy);cudaFree(d_roz);cudaFree(d_rdx);cudaFree(d_rdy);cudaFree(d_rdz);
    cudaFree(d_rix);cudaFree(d_riy);cudaFree(d_riz);cudaFree(d_hitT);

    return 0;
}
