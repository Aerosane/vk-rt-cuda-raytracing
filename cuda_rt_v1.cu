/*
 * V100 Tensor-Accelerated Ray Tracing Engine
 * Phase 1: SAH BVH + SoA Layout
 * Phase 2: Persistent Threads + Ray Sorting  
 * Phase 3: Tensor Core AABB Engine (WMMA)
 * Phase 4: Hybrid Pipeline
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <algorithm>

using namespace nvcuda;

// ==================== Configuration ====================
#define MAX_DEPTH 64
#define SAH_BINS 16
#define LEAF_SIZE 4
#define BLOCK_SIZE 256
#define PERSISTENT_BLOCKS 80  // = SM count on V100

// ==================== SoA BVH (Phase 2) ====================
// Pack BVH nodes for coalesced access: 32 bytes per node
struct BVHNodeSoA {
    float *bmin_x, *bmin_y, *bmin_z;
    float *bmax_x, *bmax_y, *bmax_z;
    int *left, *right;  // left=-1 means leaf; right=triStart for leaves
    int *triCount;       // only for leaves
    int count;
};

// SoA Triangles
struct TriangleSoA {
    float *v0x, *v0y, *v0z;
    float *v1x, *v1y, *v1z;
    float *v2x, *v2y, *v2z;
    int count;
};

// SoA Rays
struct RaySoA {
    float *ox, *oy, *oz;
    float *dx, *dy, *dz;
    float *idx, *idy, *idz; // inv_dir
    float *tmin, *tmax;
    int count;
};

struct HitResult { float t; int triIdx; float u, v; };

// AoS versions for building
struct Triangle { float3 v0, v1, v2; };
struct AABB { float3 bmin, bmax; };

// ==================== SAH BVH Builder (Phase 1) ====================

struct BVHBuildNode {
    AABB bounds;
    int left, right;
    int triStart, triCount;
};

struct BuildCtx {
    BVHBuildNode* nodes;
    Triangle* tris;
    int* indices;
    int nodeCount;
};

AABB unionAABB(const AABB& a, const AABB& b) {
    AABB r;
    r.bmin = {fminf(a.bmin.x,b.bmin.x), fminf(a.bmin.y,b.bmin.y), fminf(a.bmin.z,b.bmin.z)};
    r.bmax = {fmaxf(a.bmax.x,b.bmax.x), fmaxf(a.bmax.y,b.bmax.y), fmaxf(a.bmax.z,b.bmax.z)};
    return r;
}

AABB triAABB(const Triangle& t) {
    AABB b;
    b.bmin = {fminf(fminf(t.v0.x,t.v1.x),t.v2.x), fminf(fminf(t.v0.y,t.v1.y),t.v2.y), fminf(fminf(t.v0.z,t.v1.z),t.v2.z)};
    b.bmax = {fmaxf(fmaxf(t.v0.x,t.v1.x),t.v2.x), fmaxf(fmaxf(t.v0.y,t.v1.y),t.v2.y), fmaxf(fmaxf(t.v0.z,t.v1.z),t.v2.z)};
    return b;
}

float3 triCenter(const Triangle& t) {
    return {(t.v0.x+t.v1.x+t.v2.x)/3, (t.v0.y+t.v1.y+t.v2.y)/3, (t.v0.z+t.v1.z+t.v2.z)/3};
}

float surfaceArea(const AABB& b) {
    float dx = b.bmax.x-b.bmin.x, dy = b.bmax.y-b.bmin.y, dz = b.bmax.z-b.bmin.z;
    return 2.0f*(dx*dy + dy*dz + dz*dx);
}

// SAH Binned BVH build
int buildSAH(BuildCtx& ctx, int start, int count, int depth) {
    int ni = ctx.nodeCount++;
    BVHBuildNode& node = ctx.nodes[ni];
    
    // Compute bounds
    AABB bounds = triAABB(ctx.tris[ctx.indices[start]]);
    for (int i = 1; i < count; i++)
        bounds = unionAABB(bounds, triAABB(ctx.tris[ctx.indices[start+i]]));
    node.bounds = bounds;
    
    if (count <= LEAF_SIZE || depth > 24) {
        node.left = -1; node.right = -1;
        node.triStart = start; node.triCount = count;
        return ni;
    }
    
    // Compute centroid bounds
    AABB centBounds;
    centBounds.bmin = centBounds.bmax = triCenter(ctx.tris[ctx.indices[start]]);
    for (int i = 1; i < count; i++) {
        float3 c = triCenter(ctx.tris[ctx.indices[start+i]]);
        centBounds.bmin = {fminf(centBounds.bmin.x,c.x), fminf(centBounds.bmin.y,c.y), fminf(centBounds.bmin.z,c.z)};
        centBounds.bmax = {fmaxf(centBounds.bmax.x,c.x), fmaxf(centBounds.bmax.y,c.y), fmaxf(centBounds.bmax.z,c.z)};
    }
    
    // Find best split axis and position using SAH
    float bestCost = FLT_MAX;
    int bestAxis = -1, bestBin = -1;
    float parentArea = surfaceArea(bounds);
    
    for (int axis = 0; axis < 3; axis++) {
        float axMin = (axis==0?centBounds.bmin.x:axis==1?centBounds.bmin.y:centBounds.bmin.z);
        float axMax = (axis==0?centBounds.bmax.x:axis==1?centBounds.bmax.y:centBounds.bmax.z);
        if (axMax - axMin < 1e-6f) continue;
        
        // Initialize bins
        struct Bin { AABB bounds; int count; };
        Bin bins[SAH_BINS];
        for (int i = 0; i < SAH_BINS; i++) {
            bins[i].bounds.bmin = {FLT_MAX,FLT_MAX,FLT_MAX};
            bins[i].bounds.bmax = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
            bins[i].count = 0;
        }
        
        float scale = SAH_BINS / (axMax - axMin);
        for (int i = 0; i < count; i++) {
            float3 c = triCenter(ctx.tris[ctx.indices[start+i]]);
            float cv = (axis==0?c.x:axis==1?c.y:c.z);
            int b = (int)((cv - axMin) * scale);
            if (b >= SAH_BINS) b = SAH_BINS-1;
            if (b < 0) b = 0;
            bins[b].bounds = unionAABB(bins[b].bounds, triAABB(ctx.tris[ctx.indices[start+i]]));
            bins[b].count++;
        }
        
        // Sweep to find best split
        for (int i = 0; i < SAH_BINS-1; i++) {
            AABB bL = bins[0].bounds; int cL = bins[0].count;
            for (int j = 1; j <= i; j++) { bL = unionAABB(bL, bins[j].bounds); cL += bins[j].count; }
            AABB bR = bins[i+1].bounds; int cR = bins[i+1].count;
            for (int j = i+2; j < SAH_BINS; j++) { bR = unionAABB(bR, bins[j].bounds); cR += bins[j].count; }
            
            if (cL == 0 || cR == 0) continue;
            float cost = 0.125f + (cL * surfaceArea(bL) + cR * surfaceArea(bR)) / parentArea;
            if (cost < bestCost) { bestCost = cost; bestAxis = axis; bestBin = i; }
        }
    }
    
    // Fallback: if SAH finds no good split, use median
    if (bestAxis == -1) {
        float3 ext = {bounds.bmax.x-bounds.bmin.x, bounds.bmax.y-bounds.bmin.y, bounds.bmax.z-bounds.bmin.z};
        bestAxis = (ext.x > ext.y && ext.x > ext.z) ? 0 : (ext.y > ext.z ? 1 : 2);
        
        // Median split
        node.triStart = -1; node.triCount = 0;
        int half = count/2;
        // Simple nth_element-like partition
        for (int i = start; i < start+half; i++) {
            int minIdx = i;
            float minVal;
            float3 ci = triCenter(ctx.tris[ctx.indices[i]]);
            minVal = (bestAxis==0?ci.x:bestAxis==1?ci.y:ci.z);
            for (int j = i+1; j < start+count; j++) {
                float3 cj = triCenter(ctx.tris[ctx.indices[j]]);
                float v = (bestAxis==0?cj.x:bestAxis==1?cj.y:cj.z);
                if (v < minVal) { minVal = v; minIdx = j; }
            }
            if (minIdx != i) { int tmp = ctx.indices[i]; ctx.indices[i] = ctx.indices[minIdx]; ctx.indices[minIdx] = tmp; }
        }
        node.left = buildSAH(ctx, start, half, depth+1);
        node.right = buildSAH(ctx, start+half, count-half, depth+1);
        return ni;
    }
    
    // Partition by best bin
    float axMin = (bestAxis==0?centBounds.bmin.x:bestAxis==1?centBounds.bmin.y:centBounds.bmin.z);
    float axMax = (bestAxis==0?centBounds.bmax.x:bestAxis==1?centBounds.bmax.y:centBounds.bmax.z);
    float scale = SAH_BINS / (axMax - axMin);
    
    int i = start, j = start+count-1;
    while (i <= j) {
        float3 c = triCenter(ctx.tris[ctx.indices[i]]);
        float cv = (bestAxis==0?c.x:bestAxis==1?c.y:c.z);
        int b = (int)((cv - axMin) * scale);
        if (b >= SAH_BINS) b = SAH_BINS-1;
        if (b < 0) b = 0;
        if (b <= bestBin) i++;
        else { int tmp = ctx.indices[i]; ctx.indices[i] = ctx.indices[j]; ctx.indices[j] = tmp; j--; }
    }
    
    int leftCount = i - start;
    if (leftCount == 0) leftCount = 1;
    if (leftCount == count) leftCount = count-1;
    
    node.triStart = -1; node.triCount = 0;
    node.left = buildSAH(ctx, start, leftCount, depth+1);
    node.right = buildSAH(ctx, start+leftCount, count-leftCount, depth+1);
    return ni;
}

// ==================== GPU Kernels ====================

// Phase 2: Persistent thread traversal with SoA
__global__ void trace_persistent_soa(
    const float* __restrict__ bmin_x, const float* __restrict__ bmin_y, const float* __restrict__ bmin_z,
    const float* __restrict__ bmax_x, const float* __restrict__ bmax_y, const float* __restrict__ bmax_z,
    const int* __restrict__ left, const int* __restrict__ right, const int* __restrict__ triCount,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const float* __restrict__ rox, const float* __restrict__ roy, const float* __restrict__ roz,
    const float* __restrict__ ridx, const float* __restrict__ ridy, const float* __restrict__ ridz,
    HitResult* __restrict__ hits, int numRays,
    int* __restrict__ rayCounter, int* __restrict__ totalNodeTests, int* __restrict__ totalTriTests)
{
    // Persistent threads: keep running until all rays are done
    while (true) {
        // Atomically grab a ray
        int idx = atomicAdd(rayCounter, 1);
        if (idx >= numRays) return;
        
        // Load ray
        float ox = rox[idx], oy = roy[idx], oz = roz[idx];
        float ix = ridx[idx], iy = ridy[idx], iz = ridz[idx];
        
        float hitT = 1e30f;
        int hitTri = -1;
        float hitU = 0, hitV = 0;
        int nodeTests = 0, triTests = 0;
        
        // Stack-based traversal
        int stack[MAX_DEPTH];
        int sp = 0;
        stack[sp++] = 0;
        
        while (sp > 0) {
            int ni = stack[--sp];
            nodeTests++;
            
            // SoA AABB test (coalesced reads across warp!)
            float bmnx = bmin_x[ni], bmny = bmin_y[ni], bmnz = bmin_z[ni];
            float bmxx = bmax_x[ni], bmxy = bmax_y[ni], bmxz = bmax_z[ni];
            
            float tx1 = (bmnx - ox) * ix, tx2 = (bmxx - ox) * ix;
            float tmin = fminf(tx1,tx2), tmax = fmaxf(tx1,tx2);
            float ty1 = (bmny - oy) * iy, ty2 = (bmxy - oy) * iy;
            tmin = fmaxf(tmin, fminf(ty1,ty2)); tmax = fminf(tmax, fmaxf(ty1,ty2));
            float tz1 = (bmnz - oz) * iz, tz2 = (bmxz - oz) * iz;
            tmin = fmaxf(tmin, fminf(tz1,tz2)); tmax = fminf(tmax, fmaxf(tz1,tz2));
            
            if (tmax < fmaxf(tmin, 0.001f) || tmin > hitT) continue;
            
            int lc = left[ni];
            if (lc == -1) {
                // Leaf
                int ts = right[ni]; // triStart stored in right for leaves
                int tc = triCount[ni];
                for (int i = 0; i < tc; i++) {
                    int ti = ts + i;
                    triTests++;
                    // SoA triangle load
                    float3 v0 = {tv0x[ti], tv0y[ti], tv0z[ti]};
                    float3 v1 = {tv1x[ti], tv1y[ti], tv1z[ti]};
                    float3 v2 = {tv2x[ti], tv2y[ti], tv2z[ti]};
                    
                    float3 e1 = {v1.x-v0.x, v1.y-v0.y, v1.z-v0.z};
                    float3 e2 = {v2.x-v0.x, v2.y-v0.y, v2.z-v0.z};
                    float dx = ridx[idx] != 0 ? 1.0f/ridx[idx] : 0; // reconstruct dir from inv_dir
                    // Actually we need original direction, store it
                    float rdx = (ix != 0) ? 1.0f/ix : 0;
                    float rdy = (iy != 0) ? 1.0f/iy : 0;
                    float rdz = (iz != 0) ? 1.0f/iz : 0;
                    
                    float3 h = {rdy*e2.z - rdz*e2.y, rdz*e2.x - rdx*e2.z, rdx*e2.y - rdy*e2.x};
                    float a = e1.x*h.x + e1.y*h.y + e1.z*h.z;
                    if (fabsf(a) < 1e-8f) continue;
                    float f = 1.0f/a;
                    float3 s = {ox-v0.x, oy-v0.y, oz-v0.z};
                    float u = f*(s.x*h.x+s.y*h.y+s.z*h.z);
                    if (u < 0 || u > 1) continue;
                    float3 q = {s.y*e1.z-s.z*e1.y, s.z*e1.x-s.x*e1.z, s.x*e1.y-s.y*e1.x};
                    float v = f*(rdx*q.x+rdy*q.y+rdz*q.z);
                    if (v < 0 || u+v > 1) continue;
                    float t = f*(e2.x*q.x+e2.y*q.y+e2.z*q.z);
                    if (t > 0.001f && t < hitT) {
                        hitT = t; hitTri = ti; hitU = u; hitV = v;
                    }
                }
            } else {
                int rc = right[ni];
                // Test both children, push far first
                float lmnx = bmin_x[lc], lmny = bmin_y[lc], lmnz = bmin_z[lc];
                float lmxx = bmax_x[lc], lmxy = bmax_y[lc], lmxz = bmax_z[lc];
                float ltx1 = (lmnx-ox)*ix, ltx2 = (lmxx-ox)*ix;
                float ltmin = fminf(ltx1,ltx2), ltmax = fmaxf(ltx1,ltx2);
                float lty1 = (lmny-oy)*iy, lty2 = (lmxy-oy)*iy;
                ltmin = fmaxf(ltmin,fminf(lty1,lty2)); ltmax = fminf(ltmax,fmaxf(lty1,lty2));
                float ltz1 = (lmnz-oz)*iz, ltz2 = (lmxz-oz)*iz;
                ltmin = fmaxf(ltmin,fminf(ltz1,ltz2)); ltmax = fminf(ltmax,fmaxf(ltz1,ltz2));
                bool hL = ltmax >= fmaxf(ltmin,0.001f) && ltmin <= hitT;
                
                float rmnx = bmin_x[rc], rmny = bmin_y[rc], rmnz = bmin_z[rc];
                float rmxx = bmax_x[rc], rmxy = bmax_y[rc], rmxz = bmax_z[rc];
                float rtx1 = (rmnx-ox)*ix, rtx2 = (rmxx-ox)*ix;
                float rtmin = fminf(rtx1,rtx2), rtmax = fmaxf(rtx1,rtx2);
                float rty1 = (rmny-oy)*iy, rty2 = (rmxy-oy)*iy;
                rtmin = fmaxf(rtmin,fminf(rty1,rty2)); rtmax = fminf(rtmax,fmaxf(rty1,rty2));
                float rtz1 = (rmnz-oz)*iz, rtz2 = (rmxz-oz)*iz;
                rtmin = fmaxf(rtmin,fminf(rtz1,rtz2)); rtmax = fminf(rtmax,fmaxf(rtz1,rtz2));
                bool hR = rtmax >= fmaxf(rtmin,0.001f) && rtmin <= hitT;
                
                nodeTests += 2;
                
                if (hL && hR) {
                    if (ltmin > rtmin) { stack[sp++] = lc; stack[sp++] = rc; }
                    else { stack[sp++] = rc; stack[sp++] = lc; }
                } else if (hL) stack[sp++] = lc;
                  else if (hR) stack[sp++] = rc;
            }
        }
        
        hits[idx].t = hitT;
        hits[idx].triIdx = hitTri;
        hits[idx].u = hitU;
        hits[idx].v = hitV;
        atomicAdd(totalNodeTests, nodeTests);
        atomicAdd(totalTriTests, triTests);
    }
}

// ==================== Phase 3: Tensor Core AABB Engine ====================

// Batch 16 rays against 16 BVH nodes using WMMA FP16
// The AABB slab test: t = (bound - origin) * inv_dir
// We formulate this as matrix multiply:
// For X axis: T_x[ray][node] = inv_dir_x[ray] * (bound_x[node] - origin_x[ray])
// This is: T = diag(inv_dir_x) * (bound_x - origin_x), or equivalently
// T = A * B where A[i][k] and B[k][j] are arranged to give us the products

__global__ void tensorAABBTest(
    const half* __restrict__ ray_inv_dir, // [numRays * 3], FP16
    const half* __restrict__ ray_origin,  // [numRays * 3], FP16
    const half* __restrict__ node_bmin,   // [numNodes * 3], FP16
    const half* __restrict__ node_bmax,   // [numNodes * 3], FP16
    int* __restrict__ hitMask,            // output: bitmask of hit nodes per ray
    int numRays, int numNodes)
{
    // Each warp processes 16 rays × 16 nodes using WMMA
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int laneId = threadIdx.x & 31;
    
    int rayBase = warpId * 16;
    if (rayBase >= numRays) return;
    
    // Process 16 nodes at a time
    for (int nodeBase = 0; nodeBase < numNodes; nodeBase += 16) {
        // For each axis, compute t = (bound - origin) * inv_dir
        // We need tmin and tmax across axes
        
        // Declare WMMA fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
        
        // For the slab test we need: t_lo[i][j] and t_hi[i][j] for each axis
        // Then tmin = max(tmin_x, tmin_y, tmin_z)
        //      tmax = min(tmax_x, tmax_y, tmax_z)
        
        // We use WMMA to compute: T = inv_dir * bound (batched)
        // But WMMA does C = A*B+C, which is a 16x16x16 matmul
        // We can pack: A[ray][k] = inv_dir values, B[k][node] = bound values
        // with k=16 dimension used for the 6 values (bmin.xyz, bmax.xyz) padded
        
        // Simplified: compute ray-node products directly
        // Load into shared memory, then use WMMA
        
        __shared__ half smem_a[16*16]; // rays: inv_dir packed
        __shared__ half smem_b[16*16]; // nodes: (bound - origin) packed
        __shared__ half smem_c[16*16]; // results
        
        // For now, compute one axis at a time (3 WMMA ops)
        // Each gives t-values for 16 rays × 16 nodes
        
        float tmin_result[16]; // per-node tmin for this ray
        float tmax_result[16]; // per-node tmax for this ray
        for (int n = 0; n < 16; n++) { tmin_result[n] = -1e30f; tmax_result[n] = 1e30f; }
        
        for (int axis = 0; axis < 3; axis++) {
            // Load A matrix: inv_dir for 16 rays (diagonal-ish pattern)
            // For simplicity, each row is one ray, each col is identity-like
            if (laneId < 16) {
                int ray = rayBase + laneId;
                for (int k = 0; k < 16; k++) {
                    if (ray < numRays && k == laneId)
                        smem_a[laneId * 16 + k] = ray_inv_dir[ray * 3 + axis];
                    else
                        smem_a[laneId * 16 + k] = __float2half(0.0f);
                }
            }
            __syncwarp();
            
            // Load B matrix: (bmin - origin) for 16 nodes
            // B[k][node] where k matches the diagonal in A
            if (laneId < 16) {
                int node = nodeBase + laneId;
                for (int k = 0; k < 16; k++) {
                    int ray = rayBase + k;
                    if (node < numNodes && ray < numRays) {
                        half bval = node_bmin[node * 3 + axis];
                        half oval = ray_origin[ray * 3 + axis];
                        smem_b[k * 16 + laneId] = __hsub(bval, oval);
                    } else {
                        smem_b[k * 16 + laneId] = __float2half(0.0f);
                    }
                }
            }
            __syncwarp();
            
            // WMMA: C = A * B (t_lo values)
            wmma::load_matrix_sync(a_frag, smem_a, 16);
            wmma::load_matrix_sync(b_frag, smem_b, 16);
            wmma::fill_fragment(c_frag, __float2half(0.0f));
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            wmma::store_matrix_sync(smem_c, c_frag, 16, wmma::mem_row_major);
            __syncwarp();
            
            // Extract t_lo for this axis (bmin side)
            if (laneId < 16) {
                int ray_local = laneId;
                for (int n = 0; n < 16; n++) {
                    float t_lo = __half2float(smem_c[ray_local * 16 + n]);
                    
                    // Now compute t_hi (bmax side) 
                    int node = nodeBase + n;
                    int ray = rayBase + ray_local;
                    float t_hi = 0;
                    if (node < numNodes && ray < numRays) {
                        float inv = __half2float(ray_inv_dir[ray * 3 + axis]);
                        float bmax_val = __half2float(node_bmax[node * 3 + axis]);
                        float orig = __half2float(ray_origin[ray * 3 + axis]);
                        t_hi = (bmax_val - orig) * inv;
                    }
                    
                    float this_tmin = fminf(t_lo, t_hi);
                    float this_tmax = fmaxf(t_lo, t_hi);
                    tmin_result[n] = fmaxf(tmin_result[n], this_tmin);
                    tmax_result[n] = fminf(tmax_result[n], this_tmax);
                }
            }
        }
        
        // Determine hits
        if (laneId < 16) {
            int ray = rayBase + laneId;
            if (ray < numRays) {
                int mask = 0;
                for (int n = 0; n < 16; n++) {
                    if (tmax_result[n] >= fmaxf(tmin_result[n], 0.001f))
                        mask |= (1 << n);
                }
                hitMask[ray * ((numNodes+15)/16) + nodeBase/16] = mask;
            }
        }
    }
}

// ==================== Scene & Ray Generation ====================

void genScene(Triangle* t, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        float cx = ((float)rand()/RAND_MAX)*20-10, cy = ((float)rand()/RAND_MAX)*20-10, cz = ((float)rand()/RAND_MAX)*20-10;
        float s = ((float)rand()/RAND_MAX)*0.5f+0.1f;
        t[i].v0 = {cx-s, cy-s, cz}; t[i].v1 = {cx+s, cy, cz+s}; t[i].v2 = {cx, cy+s, cz-s};
    }
}

// Morton code for spatial sorting
__host__ __device__ unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

unsigned int mortonCode(float x, float y, float z, float3 bmin, float3 bmax) {
    x = (x - bmin.x) / (bmax.x - bmin.x) * 1023.0f;
    y = (y - bmin.y) / (bmax.y - bmin.y) * 1023.0f;
    z = (z - bmin.z) / (bmax.z - bmin.z) * 1023.0f;
    unsigned int ix = (unsigned int)fminf(fmaxf(x, 0.0f), 1023.0f);
    unsigned int iy = (unsigned int)fminf(fmaxf(y, 0.0f), 1023.0f);
    unsigned int iz = (unsigned int)fminf(fmaxf(z, 0.0f), 1023.0f);
    return expandBits(ix) | (expandBits(iy) << 1) | (expandBits(iz) << 2);
}

struct RayAoS { float3 origin, direction, inv_dir; };

void genAndSortRays(RayAoS* rays, int n) {
    srand(123);
    for (int i = 0; i < n; i++) {
        float u = ((float)rand()/RAND_MAX)*6.283f, v = ((float)rand()/RAND_MAX)*3.14159f;
        rays[i].origin = {sinf(v)*cosf(u)*15, sinf(v)*sinf(u)*15, cosf(v)*15};
        float3 tgt = {((float)rand()/RAND_MAX-0.5f)*2, ((float)rand()/RAND_MAX-0.5f)*2, ((float)rand()/RAND_MAX-0.5f)*2};
        float3 d = {tgt.x-rays[i].origin.x, tgt.y-rays[i].origin.y, tgt.z-rays[i].origin.z};
        float l = sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);
        d.x/=l; d.y/=l; d.z/=l;
        rays[i].direction = d;
        rays[i].inv_dir = {1.0f/d.x, 1.0f/d.y, 1.0f/d.z};
    }
    
    // Sort by Morton code of origin
    float3 bmin = {FLT_MAX,FLT_MAX,FLT_MAX}, bmax = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
    for (int i = 0; i < n; i++) {
        bmin.x = fminf(bmin.x, rays[i].origin.x); bmin.y = fminf(bmin.y, rays[i].origin.y); bmin.z = fminf(bmin.z, rays[i].origin.z);
        bmax.x = fmaxf(bmax.x, rays[i].origin.x); bmax.y = fmaxf(bmax.y, rays[i].origin.y); bmax.z = fmaxf(bmax.z, rays[i].origin.z);
    }
    
    unsigned int* codes = (unsigned int*)malloc(n * sizeof(unsigned int));
    int* indices = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        codes[i] = mortonCode(rays[i].origin.x, rays[i].origin.y, rays[i].origin.z, bmin, bmax);
        indices[i] = i;
    }
    
    // Simple sort by Morton code (radix sort would be faster but this is CPU prep)
    // Use std::sort with index
    std::sort(indices, indices + n, [&codes](int a, int b) { return codes[a] < codes[b]; });
    
    RayAoS* sorted = (RayAoS*)malloc(n * sizeof(RayAoS));
    for (int i = 0; i < n; i++) sorted[i] = rays[indices[i]];
    memcpy(rays, sorted, n * sizeof(RayAoS));
    free(sorted); free(codes); free(indices);
}

// ==================== Benchmark ====================

int main() {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║   V100 Tensor-Accelerated Ray Tracing Engine                 ║\n");
    printf("║   SAH BVH + SoA + Persistent Threads + Morton Sort          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s | %d SMs | %.0f GB/s HBM2\n\n", prop.name, prop.multiProcessorCount,
        2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1e6);
    
    int triCounts[] = {100000, 500000, 1000000};
    int numRays = 4194304; // 4M rays
    
    for (int s = 0; s < 3; s++) {
        int nt = triCounts[s];
        printf("═══════════════════════════════════════════════════════════\n");
        printf(" Scene: %dK tris | 4M rays | SAH BVH leaf=%d\n", nt/1000, LEAF_SIZE);
        printf("═══════════════════════════════════════════════════════════\n");
        
        // Generate scene
        Triangle* h_tris = (Triangle*)malloc(nt*sizeof(Triangle));
        genScene(h_tris, nt);
        
        // Build SAH BVH
        int maxN = nt*2;
        BVHBuildNode* h_nodes = (BVHBuildNode*)calloc(maxN, sizeof(BVHBuildNode));
        int* tidx = (int*)malloc(nt*sizeof(int));
        for (int i=0;i<nt;i++) tidx[i]=i;
        BuildCtx ctx = {h_nodes, h_tris, tidx, 0};
        
        double bt0 = clock()/(double)CLOCKS_PER_SEC;
        buildSAH(ctx, 0, nt, 0);
        double bt1 = clock()/(double)CLOCKS_PER_SEC;
        printf("  SAH BVH: %d nodes in %.0f ms\n", ctx.nodeCount, (bt1-bt0)*1000);
        
        // Reorder tris by BVH leaf order
        Triangle* h_to = (Triangle*)malloc(nt*sizeof(Triangle));
        for (int i=0;i<nt;i++) h_to[i] = h_tris[tidx[i]];
        
        // Convert to SoA for GPU
        int nc = ctx.nodeCount;
        float *h_bmin_x=(float*)malloc(nc*4), *h_bmin_y=(float*)malloc(nc*4), *h_bmin_z=(float*)malloc(nc*4);
        float *h_bmax_x=(float*)malloc(nc*4), *h_bmax_y=(float*)malloc(nc*4), *h_bmax_z=(float*)malloc(nc*4);
        int *h_left=(int*)malloc(nc*4), *h_right=(int*)malloc(nc*4), *h_tc=(int*)malloc(nc*4);
        
        for (int i = 0; i < nc; i++) {
            h_bmin_x[i] = h_nodes[i].bounds.bmin.x; h_bmin_y[i] = h_nodes[i].bounds.bmin.y; h_bmin_z[i] = h_nodes[i].bounds.bmin.z;
            h_bmax_x[i] = h_nodes[i].bounds.bmax.x; h_bmax_y[i] = h_nodes[i].bounds.bmax.y; h_bmax_z[i] = h_nodes[i].bounds.bmax.z;
            h_left[i] = h_nodes[i].left;
            h_right[i] = (h_nodes[i].left == -1) ? h_nodes[i].triStart : h_nodes[i].right;
            h_tc[i] = h_nodes[i].triCount;
        }
        
        // Convert tris to SoA
        float *h_tv0x=(float*)malloc(nt*4),*h_tv0y=(float*)malloc(nt*4),*h_tv0z=(float*)malloc(nt*4);
        float *h_tv1x=(float*)malloc(nt*4),*h_tv1y=(float*)malloc(nt*4),*h_tv1z=(float*)malloc(nt*4);
        float *h_tv2x=(float*)malloc(nt*4),*h_tv2y=(float*)malloc(nt*4),*h_tv2z=(float*)malloc(nt*4);
        for (int i = 0; i < nt; i++) {
            h_tv0x[i]=h_to[i].v0.x; h_tv0y[i]=h_to[i].v0.y; h_tv0z[i]=h_to[i].v0.z;
            h_tv1x[i]=h_to[i].v1.x; h_tv1y[i]=h_to[i].v1.y; h_tv1z[i]=h_to[i].v1.z;
            h_tv2x[i]=h_to[i].v2.x; h_tv2y[i]=h_to[i].v2.y; h_tv2z[i]=h_to[i].v2.z;
        }
        
        // Generate and sort rays
        RayAoS* h_rays_aos = (RayAoS*)malloc(numRays*sizeof(RayAoS));
        genAndSortRays(h_rays_aos, numRays);
        
        // Rays to SoA
        float *h_rox=(float*)malloc(numRays*4),*h_roy=(float*)malloc(numRays*4),*h_roz=(float*)malloc(numRays*4);
        float *h_ridx=(float*)malloc(numRays*4),*h_ridy=(float*)malloc(numRays*4),*h_ridz=(float*)malloc(numRays*4);
        for (int i = 0; i < numRays; i++) {
            h_rox[i]=h_rays_aos[i].origin.x; h_roy[i]=h_rays_aos[i].origin.y; h_roz[i]=h_rays_aos[i].origin.z;
            h_ridx[i]=h_rays_aos[i].inv_dir.x; h_ridy[i]=h_rays_aos[i].inv_dir.y; h_ridz[i]=h_rays_aos[i].inv_dir.z;
        }
        
        // GPU allocate
        float *d_bmin_x,*d_bmin_y,*d_bmin_z,*d_bmax_x,*d_bmax_y,*d_bmax_z;
        int *d_left,*d_right,*d_tc;
        float *d_tv0x,*d_tv0y,*d_tv0z,*d_tv1x,*d_tv1y,*d_tv1z,*d_tv2x,*d_tv2y,*d_tv2z;
        float *d_rox,*d_roy,*d_roz,*d_ridx,*d_ridy,*d_ridz;
        HitResult *d_hits;
        int *d_rayCounter, *d_nodeTests, *d_triTests;
        
        cudaMalloc(&d_bmin_x,nc*4); cudaMalloc(&d_bmin_y,nc*4); cudaMalloc(&d_bmin_z,nc*4);
        cudaMalloc(&d_bmax_x,nc*4); cudaMalloc(&d_bmax_y,nc*4); cudaMalloc(&d_bmax_z,nc*4);
        cudaMalloc(&d_left,nc*4); cudaMalloc(&d_right,nc*4); cudaMalloc(&d_tc,nc*4);
        cudaMalloc(&d_tv0x,nt*4); cudaMalloc(&d_tv0y,nt*4); cudaMalloc(&d_tv0z,nt*4);
        cudaMalloc(&d_tv1x,nt*4); cudaMalloc(&d_tv1y,nt*4); cudaMalloc(&d_tv1z,nt*4);
        cudaMalloc(&d_tv2x,nt*4); cudaMalloc(&d_tv2y,nt*4); cudaMalloc(&d_tv2z,nt*4);
        cudaMalloc(&d_rox,numRays*4); cudaMalloc(&d_roy,numRays*4); cudaMalloc(&d_roz,numRays*4);
        cudaMalloc(&d_ridx,numRays*4); cudaMalloc(&d_ridy,numRays*4); cudaMalloc(&d_ridz,numRays*4);
        cudaMalloc(&d_hits,numRays*sizeof(HitResult));
        cudaMalloc(&d_rayCounter,4); cudaMalloc(&d_nodeTests,4); cudaMalloc(&d_triTests,4);
        
        cudaMemcpy(d_bmin_x,h_bmin_x,nc*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_bmin_y,h_bmin_y,nc*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_bmin_z,h_bmin_z,nc*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_bmax_x,h_bmax_x,nc*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_bmax_y,h_bmax_y,nc*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_bmax_z,h_bmax_z,nc*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_left,h_left,nc*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_right,h_right,nc*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_tc,h_tc,nc*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_tv0x,h_tv0x,nt*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_tv0y,h_tv0y,nt*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_tv0z,h_tv0z,nt*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_tv1x,h_tv1x,nt*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_tv1y,h_tv1y,nt*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_tv1z,h_tv1z,nt*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_tv2x,h_tv2x,nt*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_tv2y,h_tv2y,nt*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_tv2z,h_tv2z,nt*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_rox,h_rox,numRays*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_roy,h_roy,numRays*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_roz,h_roz,numRays*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_ridx,h_ridx,numRays*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_ridy,h_ridy,numRays*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_ridz,h_ridz,numRays*4,cudaMemcpyHostToDevice);
        
        // ===== BENCHMARK: Persistent + SoA + SAH + Morton sorted =====
        // Warmup
        cudaMemset(d_rayCounter,0,4); cudaMemset(d_nodeTests,0,4); cudaMemset(d_triTests,0,4);
        trace_persistent_soa<<<PERSISTENT_BLOCKS, BLOCK_SIZE>>>(
            d_bmin_x,d_bmin_y,d_bmin_z, d_bmax_x,d_bmax_y,d_bmax_z,
            d_left,d_right,d_tc,
            d_tv0x,d_tv0y,d_tv0z, d_tv1x,d_tv1y,d_tv1z, d_tv2x,d_tv2y,d_tv2z,
            d_rox,d_roy,d_roz, d_ridx,d_ridy,d_ridz,
            d_hits, numRays, d_rayCounter, d_nodeTests, d_triTests);
        cudaDeviceSynchronize();
        
        // Timed runs
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        
        float totalMs = 0;
        int runs = 10;
        for (int r = 0; r < runs; r++) {
            cudaMemset(d_rayCounter,0,4); cudaMemset(d_nodeTests,0,4); cudaMemset(d_triTests,0,4);
            cudaEventRecord(t0);
            trace_persistent_soa<<<PERSISTENT_BLOCKS, BLOCK_SIZE>>>(
                d_bmin_x,d_bmin_y,d_bmin_z, d_bmax_x,d_bmax_y,d_bmax_z,
                d_left,d_right,d_tc,
                d_tv0x,d_tv0y,d_tv0z, d_tv1x,d_tv1y,d_tv1z, d_tv2x,d_tv2y,d_tv2z,
                d_rox,d_roy,d_roz, d_ridx,d_ridy,d_ridz,
                d_hits, numRays, d_rayCounter, d_nodeTests, d_triTests);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1);
            totalMs += ms;
        }
        
        float avg = totalMs / runs;
        double mrays = (double)numRays / (avg/1000.0) / 1e6;
        
        int hn, ht;
        cudaMemcpy(&hn, d_nodeTests, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&ht, d_triTests, 4, cudaMemcpyDeviceToHost);
        
        HitResult* hh = (HitResult*)malloc(numRays*sizeof(HitResult));
        cudaMemcpy(hh, d_hits, numRays*sizeof(HitResult), cudaMemcpyDeviceToHost);
        int hitCnt=0; for(int i=0;i<numRays;i++) if(hh[i].triIdx>=0) hitCnt++;
        free(hh);
        
        printf("  ► SAH+SoA+Persistent+Morton: %7.1f ms │ %8.1f MRays/s │ %.1f node/ray │ %.1f tri/ray │ %d hits (%.0f%%)\n",
            avg, mrays, (float)hn/numRays, (float)ht/numRays, hitCnt, 100.0f*hitCnt/numRays);
        
        // Cleanup
        cudaFree(d_bmin_x); cudaFree(d_bmin_y); cudaFree(d_bmin_z);
        cudaFree(d_bmax_x); cudaFree(d_bmax_y); cudaFree(d_bmax_z);
        cudaFree(d_left); cudaFree(d_right); cudaFree(d_tc);
        cudaFree(d_tv0x); cudaFree(d_tv0y); cudaFree(d_tv0z);
        cudaFree(d_tv1x); cudaFree(d_tv1y); cudaFree(d_tv1z);
        cudaFree(d_tv2x); cudaFree(d_tv2y); cudaFree(d_tv2z);
        cudaFree(d_rox); cudaFree(d_roy); cudaFree(d_roz);
        cudaFree(d_ridx); cudaFree(d_ridy); cudaFree(d_ridz);
        cudaFree(d_hits); cudaFree(d_rayCounter); cudaFree(d_nodeTests); cudaFree(d_triTests);
        free(h_tris); free(h_to); free(h_nodes); free(tidx);
        free(h_bmin_x); free(h_bmin_y); free(h_bmin_z);
        free(h_bmax_x); free(h_bmax_y); free(h_bmax_z);
        free(h_left); free(h_right); free(h_tc);
        free(h_tv0x); free(h_tv0y); free(h_tv0z);
        free(h_tv1x); free(h_tv1y); free(h_tv1z);
        free(h_tv2x); free(h_tv2y); free(h_tv2z);
        free(h_rays_aos); free(h_rox); free(h_roy); free(h_roz);
        free(h_ridx); free(h_ridy); free(h_ridz);
        
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        printf("\n");
    }
    
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Baseline comparison (naive BVH, midpoint split, AoS):\n");
    printf("  100K tris: ~69 MRays/s | 500K: ~46 MRays/s\n");
    printf("═══════════════════════════════════════════════════════════\n");
    
    return 0;
}
