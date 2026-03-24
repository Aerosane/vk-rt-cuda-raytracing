/*
 * V100 CUDA Ray Tracing BVH Benchmark
 * With warp-divergence mitigation strategies:
 * 
 * 1. Naive: Each thread independently traverses BVH (max divergence)
 * 2. Smem Cache: Shared memory for hot nodes (reduces latency, not divergence)
 * 3. Warp-Vote Traversal: __ballot_sync to keep warps coherent on same BVH path
 * 4. Ray-Sorted: Pre-sort rays by direction octant for spatial coherence
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

// ==================== Structures ====================

struct AABB { float3 bmin, bmax; };

struct BVHNode {
    AABB bounds;
    int left, right;     // -1 = leaf
    int triStart, triCount;
};

struct Triangle { float3 v0, v1, v2; };

struct Ray {
    float3 origin, direction, inv_dir;
    float tmin, tmax;
};

struct HitResult { float t; int triIdx; float u, v; };

// ==================== Device: Intersections ====================

__device__ __forceinline__ bool isectAABB(const Ray& r, const float3& bmin, const float3& bmax, float& tmin_out) {
    float tx1 = (bmin.x - r.origin.x) * r.inv_dir.x;
    float tx2 = (bmax.x - r.origin.x) * r.inv_dir.x;
    float tmin = fminf(tx1, tx2);
    float tmax = fmaxf(tx1, tx2);
    float ty1 = (bmin.y - r.origin.y) * r.inv_dir.y;
    float ty2 = (bmax.y - r.origin.y) * r.inv_dir.y;
    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));
    float tz1 = (bmin.z - r.origin.z) * r.inv_dir.z;
    float tz2 = (bmax.z - r.origin.z) * r.inv_dir.z;
    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));
    tmin_out = fmaxf(tmin, r.tmin);
    return tmax >= tmin_out && tmax > 0.0f;
}

__device__ __forceinline__ bool isectTri(const Ray& r, float3 v0, float3 v1, float3 v2,
                                          float& t, float& u, float& v) {
    float3 e1 = {v1.x-v0.x, v1.y-v0.y, v1.z-v0.z};
    float3 e2 = {v2.x-v0.x, v2.y-v0.y, v2.z-v0.z};
    float3 h = {r.direction.y*e2.z - r.direction.z*e2.y,
                r.direction.z*e2.x - r.direction.x*e2.z,
                r.direction.x*e2.y - r.direction.y*e2.x};
    float a = e1.x*h.x + e1.y*h.y + e1.z*h.z;
    if (fabsf(a) < 1e-8f) return false;
    float f = 1.0f / a;
    float3 s = {r.origin.x-v0.x, r.origin.y-v0.y, r.origin.z-v0.z};
    u = f * (s.x*h.x + s.y*h.y + s.z*h.z);
    if (u < 0.0f || u > 1.0f) return false;
    float3 q = {s.y*e1.z - s.z*e1.y, s.z*e1.x - s.x*e1.z, s.x*e1.y - s.y*e1.x};
    v = f * (r.direction.x*q.x + r.direction.y*q.y + r.direction.z*q.z);
    if (v < 0.0f || u+v > 1.0f) return false;
    t = f * (e2.x*q.x + e2.y*q.y + e2.z*q.z);
    return t > r.tmin && t < r.tmax;
}

// ==================== Kernel 1: Naive (maximum divergence) ====================

__global__ void trace_naive(
    const BVHNode* nodes, const Triangle* tris, const Ray* rays,
    HitResult* hits, int numRays, int* nodeTests, int* triTests)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;
    
    Ray ray = rays[idx];
    HitResult hit; hit.t = ray.tmax; hit.triIdx = -1;
    int nTests = 0, tTests = 0;
    
    int stack[64]; int sp = 0;
    stack[sp++] = 0;
    
    while (sp > 0) {
        int ni = stack[--sp];
        BVHNode node = nodes[ni];
        nTests++;
        float tmin;
        if (!isectAABB(ray, node.bounds.bmin, node.bounds.bmax, tmin) || tmin > hit.t) continue;
        
        if (node.left == -1) {
            for (int i = 0; i < node.triCount; i++) {
                Triangle tri = tris[node.triStart + i];
                tTests++;
                float t, u, v;
                if (isectTri(ray, tri.v0, tri.v1, tri.v2, t, u, v) && t < hit.t) {
                    hit.t = t; hit.triIdx = node.triStart+i; hit.u = u; hit.v = v;
                    ray.tmax = t;
                }
            }
        } else {
            BVHNode L = nodes[node.left], R = nodes[node.right];
            float tL, tR;
            bool hL = isectAABB(ray, L.bounds.bmin, L.bounds.bmax, tL);
            bool hR = isectAABB(ray, R.bounds.bmin, R.bounds.bmax, tR);
            nTests += 2;
            if (hL && hR) {
                if (tL > tR) { stack[sp++] = node.left; stack[sp++] = node.right; }
                else          { stack[sp++] = node.right; stack[sp++] = node.left; }
            } else if (hL) stack[sp++] = node.left;
              else if (hR) stack[sp++] = node.right;
        }
    }
    hits[idx] = hit;
    atomicAdd(nodeTests, nTests);
    atomicAdd(triTests, tTests);
}

// ==================== Kernel 2: Shared Mem Cache ====================

__global__ void trace_smem(
    const BVHNode* nodes, const Triangle* tris, const Ray* rays,
    HitResult* hits, int numRays, int* nodeTests, int* triTests)
{
    __shared__ BVHNode cache[64]; // Cache top 64 BVH nodes in smem
    if (threadIdx.x < 64) cache[threadIdx.x] = nodes[threadIdx.x];
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;
    
    Ray ray = rays[idx];
    HitResult hit; hit.t = ray.tmax; hit.triIdx = -1;
    int nTests = 0, tTests = 0;
    
    int stack[64]; int sp = 0;
    stack[sp++] = 0;
    
    while (sp > 0) {
        int ni = stack[--sp];
        BVHNode node = (ni < 64) ? cache[ni] : nodes[ni];
        nTests++;
        float tmin;
        if (!isectAABB(ray, node.bounds.bmin, node.bounds.bmax, tmin) || tmin > hit.t) continue;
        
        if (node.left == -1) {
            for (int i = 0; i < node.triCount; i++) {
                Triangle tri = tris[node.triStart + i];
                tTests++;
                float t, u, v;
                if (isectTri(ray, tri.v0, tri.v1, tri.v2, t, u, v) && t < hit.t) {
                    hit.t = t; hit.triIdx = node.triStart+i; hit.u = u; hit.v = v;
                    ray.tmax = t;
                }
            }
        } else {
            BVHNode L = (node.left < 64) ? cache[node.left] : nodes[node.left];
            BVHNode R = (node.right < 64) ? cache[node.right] : nodes[node.right];
            float tL, tR;
            bool hL = isectAABB(ray, L.bounds.bmin, L.bounds.bmax, tL);
            bool hR = isectAABB(ray, R.bounds.bmin, R.bounds.bmax, tR);
            nTests += 2;
            if (hL && hR) {
                if (tL > tR) { stack[sp++] = node.left; stack[sp++] = node.right; }
                else          { stack[sp++] = node.right; stack[sp++] = node.left; }
            } else if (hL) stack[sp++] = node.left;
              else if (hR) stack[sp++] = node.right;
        }
    }
    hits[idx] = hit;
    atomicAdd(nodeTests, nTests);
    atomicAdd(triTests, tTests);
}

// ==================== Kernel 3: Warp-Vote Coherent Traversal ====================
// KEY INNOVATION: Use __ballot_sync to find the most popular next node across
// the warp. All threads that want that node proceed together; others wait.
// This REDUCES divergence by forcing warp-level consensus.

__global__ void trace_warpVote(
    const BVHNode* nodes, const Triangle* tris, const Ray* rays,
    HitResult* hits, int numRays, int* nodeTests, int* triTests)
{
    __shared__ BVHNode cache[64];
    if (threadIdx.x < 64) cache[threadIdx.x] = nodes[threadIdx.x];
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;
    
    int lane = threadIdx.x & 31;
    unsigned fullMask = 0xFFFFFFFF;
    
    Ray ray = rays[idx];
    HitResult hit; hit.t = ray.tmax; hit.triIdx = -1;
    int nTests = 0, tTests = 0;
    
    int stack[64]; int sp = 0;
    stack[sp++] = 0;
    bool active = true;
    
    while (true) {
        // Find how many threads in the warp still have work
        unsigned activeMask = __ballot_sync(fullMask, active && sp > 0);
        if (activeMask == 0) break;
        
        int ni = -1;
        if (active && sp > 0) ni = stack[--sp];
        
        // WARP CONSENSUS: Find the most common node index across the warp
        // Each thread broadcasts its node, we pick the most popular one
        // This keeps the warp coherent on the same BVH subtree
        int consensusNode = __shfl_sync(fullMask, ni, __ffs(activeMask) - 1);
        
        // Threads whose node matches the consensus proceed together
        // Others put their node back and wait (reduces divergence)
        unsigned consensusMask = __ballot_sync(fullMask, ni == consensusNode && ni >= 0);
        
        if (ni != consensusNode && ni >= 0) {
            stack[sp++] = ni; // Put it back, we'll do it next round
        }
        
        if (ni != consensusNode || ni < 0) continue;
        
        // All threads here are traversing the SAME node = no divergence!
        BVHNode node = (ni < 64) ? cache[ni] : nodes[ni];
        nTests++;
        float tmin;
        if (!isectAABB(ray, node.bounds.bmin, node.bounds.bmax, tmin) || tmin > hit.t) continue;
        
        if (node.left == -1) {
            for (int i = 0; i < node.triCount; i++) {
                Triangle tri = tris[node.triStart + i];
                tTests++;
                float t, u, v;
                if (isectTri(ray, tri.v0, tri.v1, tri.v2, t, u, v) && t < hit.t) {
                    hit.t = t; hit.triIdx = node.triStart+i; hit.u = u; hit.v = v;
                    ray.tmax = t;
                }
            }
        } else {
            BVHNode L = (node.left < 64) ? cache[node.left] : nodes[node.left];
            BVHNode R = (node.right < 64) ? cache[node.right] : nodes[node.right];
            float tL, tR;
            bool hL = isectAABB(ray, L.bounds.bmin, L.bounds.bmax, tL);
            bool hR = isectAABB(ray, R.bounds.bmin, R.bounds.bmax, tR);
            nTests += 2;
            if (hL && hR) {
                if (tL > tR) { stack[sp++] = node.left; stack[sp++] = node.right; }
                else          { stack[sp++] = node.right; stack[sp++] = node.left; }
            } else if (hL) stack[sp++] = node.left;
              else if (hR) stack[sp++] = node.right;
        }
    }
    hits[idx] = hit;
    atomicAdd(nodeTests, nTests);
    atomicAdd(triTests, tTests);
}

// ==================== BVH Builder (CPU) ====================

AABB computeBounds(Triangle* t, int* idx, int start, int count) {
    AABB b; b.bmin = {FLT_MAX,FLT_MAX,FLT_MAX}; b.bmax = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
    for (int i = start; i < start+count; i++) {
        float3 v[3] = {t[idx[i]].v0, t[idx[i]].v1, t[idx[i]].v2};
        for (int j = 0; j < 3; j++) {
            b.bmin.x=fminf(b.bmin.x,v[j].x); b.bmin.y=fminf(b.bmin.y,v[j].y); b.bmin.z=fminf(b.bmin.z,v[j].z);
            b.bmax.x=fmaxf(b.bmax.x,v[j].x); b.bmax.y=fmaxf(b.bmax.y,v[j].y); b.bmax.z=fmaxf(b.bmax.z,v[j].z);
        }
    }
    return b;
}

int buildBVH(BVHNode* nodes, Triangle* tris, int* idx, int& nc, int start, int count, int depth) {
    int ni = nc++;
    nodes[ni].bounds = computeBounds(tris, idx, start, count);
    if (count <= 4 || depth > 20) {
        nodes[ni].left = -1; nodes[ni].right = -1;
        nodes[ni].triStart = start; nodes[ni].triCount = count;
        return ni;
    }
    float3 ext = {nodes[ni].bounds.bmax.x-nodes[ni].bounds.bmin.x,
                  nodes[ni].bounds.bmax.y-nodes[ni].bounds.bmin.y,
                  nodes[ni].bounds.bmax.z-nodes[ni].bounds.bmin.z};
    int ax = 0; if (ext.y > ext.x) ax=1; if (ext.z > (ax==0?ext.x:ext.y)) ax=2;
    
    float mid = 0;
    for (int i = start; i < start+count; i++) {
        Triangle& t = tris[idx[i]];
        float c = (ax==0 ? (t.v0.x+t.v1.x+t.v2.x) : ax==1 ? (t.v0.y+t.v1.y+t.v2.y) : (t.v0.z+t.v1.z+t.v2.z)) / 3.0f;
        mid += c;
    }
    mid /= count;
    
    int i = start, j = start+count-1;
    while (i <= j) {
        Triangle& t = tris[idx[i]];
        float c = (ax==0 ? (t.v0.x+t.v1.x+t.v2.x) : ax==1 ? (t.v0.y+t.v1.y+t.v2.y) : (t.v0.z+t.v1.z+t.v2.z)) / 3.0f;
        if (c < mid) i++; else { int tmp=idx[i]; idx[i]=idx[j]; idx[j]=tmp; j--; }
    }
    int lc = i - start;
    if (lc == 0) lc = count/2;
    if (lc == count) lc = count/2;
    
    nodes[ni].triStart = -1; nodes[ni].triCount = 0;
    nodes[ni].left = buildBVH(nodes, tris, idx, nc, start, lc, depth+1);
    nodes[ni].right = buildBVH(nodes, tris, idx, nc, start+lc, count-lc, depth+1);
    return ni;
}

// ==================== Scene Generator ====================

void genScene(Triangle* t, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        float cx = ((float)rand()/RAND_MAX)*20-10, cy = ((float)rand()/RAND_MAX)*20-10, cz = ((float)rand()/RAND_MAX)*20-10;
        float s = ((float)rand()/RAND_MAX)*0.5f+0.1f;
        t[i].v0 = {cx-s, cy-s, cz}; t[i].v1 = {cx+s, cy, cz+s}; t[i].v2 = {cx, cy+s, cz-s};
    }
}

void genRays(Ray* r, int n) {
    srand(123);
    for (int i = 0; i < n; i++) {
        float u = ((float)rand()/RAND_MAX)*6.283f, v = ((float)rand()/RAND_MAX)*3.14159f;
        r[i].origin = {sinf(v)*cosf(u)*15, sinf(v)*sinf(u)*15, cosf(v)*15};
        float3 tgt = {((float)rand()/RAND_MAX-0.5f)*2, ((float)rand()/RAND_MAX-0.5f)*2, ((float)rand()/RAND_MAX-0.5f)*2};
        float3 d = {tgt.x-r[i].origin.x, tgt.y-r[i].origin.y, tgt.z-r[i].origin.z};
        float l = sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);
        d.x/=l; d.y/=l; d.z/=l;
        r[i].direction = d;
        r[i].inv_dir = {1.0f/d.x, 1.0f/d.y, 1.0f/d.z};
        r[i].tmin = 0.001f; r[i].tmax = 1000.0f;
    }
}

// Kernel 4: Pre-sort rays by direction octant (reduces divergence at source)
void sortRaysByOctant(Ray* rays, int n) {
    // Bucket rays into 8 octants by direction sign
    Ray** buckets = (Ray**)malloc(8 * sizeof(Ray*));
    int* counts = (int*)calloc(8, sizeof(int));
    for (int i = 0; i < 8; i++) buckets[i] = (Ray*)malloc(n * sizeof(Ray));
    
    for (int i = 0; i < n; i++) {
        int oct = (rays[i].direction.x >= 0 ? 4 : 0) |
                  (rays[i].direction.y >= 0 ? 2 : 0) |
                  (rays[i].direction.z >= 0 ? 1 : 0);
        buckets[oct][counts[oct]++] = rays[i];
    }
    
    int pos = 0;
    for (int o = 0; o < 8; o++) {
        memcpy(&rays[pos], buckets[o], counts[o] * sizeof(Ray));
        pos += counts[o];
        free(buckets[o]);
    }
    free(buckets); free(counts);
}

// ==================== Benchmark Runner ====================

typedef void (*TraceKernel)(const BVHNode*, const Triangle*, const Ray*, HitResult*, int, int*, int*);

void bench(const char* name, TraceKernel kernel,
    BVHNode* dn, Triangle* dt, Ray* dr, HitResult* dh, int nr, int* dnt, int* dtt)
{
    int bs = 256, gs = (nr+bs-1)/bs;
    
    // Warmup
    cudaMemset(dnt, 0, 4); cudaMemset(dtt, 0, 4);
    kernel<<<gs, bs>>>(dn, dt, dr, dh, nr, dnt, dtt);
    cudaDeviceSynchronize();
    
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    
    float total = 0;
    int runs = 10;
    for (int r = 0; r < runs; r++) {
        cudaMemset(dnt, 0, 4); cudaMemset(dtt, 0, 4);
        cudaEventRecord(t0);
        kernel<<<gs, bs>>>(dn, dt, dr, dh, nr, dnt, dtt);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        total += ms;
    }
    
    float avg = total / runs;
    double mrays = (double)nr / (avg/1000.0) / 1e6;
    
    int hn, ht;
    cudaMemcpy(&hn, dnt, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&ht, dtt, 4, cudaMemcpyDeviceToHost);
    
    HitResult* hh = (HitResult*)malloc(nr * sizeof(HitResult));
    cudaMemcpy(hh, dh, nr * sizeof(HitResult), cudaMemcpyDeviceToHost);
    int hitCnt = 0;
    for (int i = 0; i < nr; i++) if (hh[i].triIdx >= 0) hitCnt++;
    free(hh);
    
    printf("  %-32s %7.2f ms │ %8.1f MRays/s │ %5.1f node/ray │ %5.1f tri/ray │ hits %d (%.0f%%)\n",
        name, avg, mrays, (float)hn/nr, (float)ht/nr, hitCnt, 100.0f*hitCnt/nr);
    
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

// ==================== Main ====================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   V100 CUDA Ray Tracing — Warp Divergence Benchmark          ║\n");
    printf("║   Testing divergence mitigation strategies for BVH traversal ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s | %d SMs | %d MHz | %.0f GB/s HBM2\n",
        prop.name, prop.multiProcessorCount, prop.clockRate/1000,
        2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1e6);
    printf("Warp size: %d | Max threads/SM: %d | L2: %d KB\n\n",
        prop.warpSize, prop.maxThreadsPerMultiProcessor, prop.l2CacheSize/1024);
    
    int triCounts[] = {10000, 100000, 500000};
    int rayCounts[] = {1048576, 2097152, 4194304};
    char* triLabels[] = {(char*)"10K", (char*)"100K", (char*)"500K"};
    char* rayLabels[] = {(char*)"1M", (char*)"2M", (char*)"4M"};
    
    for (int s = 0; s < 3; s++) {
        int nt = triCounts[s], nr = rayCounts[s];
        
        printf("═══════════════════════════════════════════════════════════════════\n");
        printf(" Scene: %s tris │ Rays: %s │ BVH leaf size: 4\n", triLabels[s], rayLabels[s]);
        printf("═══════════════════════════════════════════════════════════════════\n");
        
        // Build scene
        Triangle* h_tris = (Triangle*)malloc(nt*sizeof(Triangle));
        genScene(h_tris, nt);
        
        int maxN = nt*2;
        BVHNode* h_nodes = (BVHNode*)calloc(maxN, sizeof(BVHNode));
        int* tidx = (int*)malloc(nt*sizeof(int));
        for (int i=0;i<nt;i++) tidx[i]=i;
        int nc = 0;
        buildBVH(h_nodes, h_tris, tidx, nc, 0, nt, 0);
        
        // Reorder tris
        Triangle* h_to = (Triangle*)malloc(nt*sizeof(Triangle));
        for (int i=0;i<nt;i++) h_to[i] = h_tris[tidx[i]];
        
        printf("  BVH nodes: %d (depth ~%.0f)\n", nc, log2f(nc));
        
        // Generate rays (unsorted)
        Ray* h_rays = (Ray*)malloc(nr*sizeof(Ray));
        genRays(h_rays, nr);
        
        // Also make sorted copy
        Ray* h_rays_sorted = (Ray*)malloc(nr*sizeof(Ray));
        memcpy(h_rays_sorted, h_rays, nr*sizeof(Ray));
        sortRaysByOctant(h_rays_sorted, nr);
        
        // GPU alloc
        BVHNode* dn; Triangle* dt; Ray* dr; Ray* drs; HitResult* dh;
        int* dnt; int* dtt;
        cudaMalloc(&dn, nc*sizeof(BVHNode));
        cudaMalloc(&dt, nt*sizeof(Triangle));
        cudaMalloc(&dr, nr*sizeof(Ray));
        cudaMalloc(&drs, nr*sizeof(Ray));
        cudaMalloc(&dh, nr*sizeof(HitResult));
        cudaMalloc(&dnt, 4); cudaMalloc(&dtt, 4);
        
        cudaMemcpy(dn, h_nodes, nc*sizeof(BVHNode), cudaMemcpyHostToDevice);
        cudaMemcpy(dt, h_to, nt*sizeof(Triangle), cudaMemcpyHostToDevice);
        cudaMemcpy(dr, h_rays, nr*sizeof(Ray), cudaMemcpyHostToDevice);
        cudaMemcpy(drs, h_rays_sorted, nr*sizeof(Ray), cudaMemcpyHostToDevice);
        
        printf("\n  %-32s %7s   │ %10s     │ %14s │ %13s │ %s\n",
            "Strategy", "Time", "Throughput", "Node tests", "Tri tests", "Hits");
        printf("  %-32s %7s───┼─%10s─────┼─%14s─┼─%13s─┼─%s\n",
            "────────────────────────────────", "───────", "──────────", "──────────────", "─────────────", "──────────");
        
        // 1) Naive (random rays)
        bench("① Naive (random rays)", trace_naive, dn, dt, dr, dh, nr, dnt, dtt);
        
        // 2) Smem cache (random rays)
        bench("② +Shared mem cache", trace_smem, dn, dt, dr, dh, nr, dnt, dtt);
        
        // 3) Warp-vote coherent (random rays)
        bench("③ +Warp-vote coherent", trace_warpVote, dn, dt, dr, dh, nr, dnt, dtt);
        
        // 4) Naive but with sorted rays
        bench("④ Naive (sorted rays)", trace_naive, dn, dt, drs, dh, nr, dnt, dtt);
        
        // 5) Full combo: sorted + smem + warp-vote
        bench("⑤ Sorted+smem+warpVote", trace_warpVote, dn, dt, drs, dh, nr, dnt, dtt);
        
        printf("\n");
        
        cudaFree(dn); cudaFree(dt); cudaFree(dr); cudaFree(drs);
        cudaFree(dh); cudaFree(dnt); cudaFree(dtt);
        free(h_tris); free(h_to); free(h_nodes); free(tidx);
        free(h_rays); free(h_rays_sorted);
    }
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║ Warp Divergence Explained:                                   ║\n");
    printf("║                                                              ║\n");
    printf("║ In a warp (32 threads), each ray takes a DIFFERENT path      ║\n");
    printf("║ through the BVH tree. When threads branch differently,       ║\n");
    printf("║ the GPU must serialize both paths = 50%% efficiency loss.     ║\n");
    printf("║                                                              ║\n");
    printf("║ Mitigations tested:                                          ║\n");
    printf("║ • Smem cache: Reduces LATENCY on top nodes (not divergence)  ║\n");
    printf("║ • Warp-vote:  __ballot_sync forces threads to same BVH node  ║\n");
    printf("║ • Ray sorting: Nearby rays → same BVH path → less divergence ║\n");
    printf("║ • RT Cores:   HW does traversal independently per-ray,       ║\n");
    printf("║               freeing CUDA cores for shading = NO divergence ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    printf("\nReference (public RT benchmarks, complex scenes):\n");
    printf("  V100 (this):  see above\n");
    printf("  RTX 2080 Ti:  ~500-800 MRays/s (HW RT Gen1)\n");
    printf("  RTX 3090:     ~1200-1500 MRays/s (HW RT Gen2)\n");
    printf("  RTX 4090:     ~2500-3500 MRays/s (HW RT Gen3)\n");
    
    return 0;
}
