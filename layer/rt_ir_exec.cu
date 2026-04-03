// rt_ir_exec.cu — CUDA executor for IR programs
// Runs IRProgram against BVH2 data on GPU.
// Each thread = one pixel/ray, executing the IR node stream.
//
#include "rt_ir.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>

using namespace rt_ir;


// ═══════════════════════════════════════════════════════════════
// GPU-side slot storage (per-thread register file)
// ═══════════════════════════════════════════════════════════════

struct IRSlotBank {
    Ray      rays[16];
    Hit      hits[16];
    Payload  payloads[4];
    Material materials[8];
    IRFloat3   vec3s[32];
    float    scalars[16];
    int      bools[8];

    // Counters for each type
    int nRay, nHit, nPayload, nMat, nVec3, nScalar, nBool;
};

// ═══════════════════════════════════════════════════════════════
// BVH2 traversal interface (linked from cuda_bvh_backend.cu)
// ═══════════════════════════════════════════════════════════════

// External: BVH2 device pointers (set by VkLayer_CudaRT)
extern "C" {
    extern void* g_ir_bvhNodes;     // BVH2 node array
    extern void* g_ir_bvhTris;      // packed triangle array
    extern void* g_ir_tlasNodes;    // TLAS node array
    extern void* g_ir_instances;    // instance array
    extern uint32_t g_ir_nodeCount;
    extern uint32_t g_ir_triCount;
}

// Simple BVH2 ray-AABB test (inlined for the executor)
__device__ static bool rayAABB(IRFloat3 ro, IRFloat3 rd, IRFloat3 invRd,
                                IRFloat3 bmin, IRFloat3 bmax, float tmax) {
    float t1 = (bmin.x - ro.x) * invRd.x;
    float t2 = (bmax.x - ro.x) * invRd.x;
    float t3 = (bmin.y - ro.y) * invRd.y;
    float t4 = (bmax.y - ro.y) * invRd.y;
    float t5 = (bmin.z - ro.z) * invRd.z;
    float t6 = (bmax.z - ro.z) * invRd.z;
    float tmin_box = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
    float tmax_box = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));
    return tmax_box >= fmaxf(tmin_box, 0.0f) && tmin_box < tmax;
}

// Möller-Trumbore triangle intersection
__device__ static bool rayTriangle(IRFloat3 ro, IRFloat3 rd,
                                    IRFloat3 v0, IRFloat3 v1, IRFloat3 v2,
                                    float& t, IRFloat3& normal) {
    IRFloat3 e1 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
    IRFloat3 e2 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
    IRFloat3 h = {rd.y * e2.z - rd.z * e2.y,
                rd.z * e2.x - rd.x * e2.z,
                rd.x * e2.y - rd.y * e2.x};
    float a = e1.x * h.x + e1.y * h.y + e1.z * h.z;
    if (fabsf(a) < 1e-7f) return false;
    float f = 1.0f / a;
    IRFloat3 s = {ro.x - v0.x, ro.y - v0.y, ro.z - v0.z};
    float u = f * (s.x * h.x + s.y * h.y + s.z * h.z);
    if (u < 0.0f || u > 1.0f) return false;
    IRFloat3 q = {s.y * e1.z - s.z * e1.y,
                s.z * e1.x - s.x * e1.z,
                s.x * e1.y - s.y * e1.x};
    float v = f * (rd.x * q.x + rd.y * q.y + rd.z * q.z);
    if (v < 0.0f || u + v > 1.0f) return false;
    t = f * (e2.x * q.x + e2.y * q.y + e2.z * q.z);
    if (t < 1e-4f) return false;
    // Face normal
    normal.x = e1.y * e2.z - e1.z * e2.y;
    normal.y = e1.z * e2.x - e1.x * e2.z;
    normal.z = e1.x * e2.y - e1.y * e2.x;
    float len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (len > 1e-7f) { normal.x /= len; normal.y /= len; normal.z /= len; }
    return true;
}

// ═══════════════════════════════════════════════════════════════
// Per-op execution (device-side switch)
// ═══════════════════════════════════════════════════════════════

__device__ static void execNode(const Node& node, IRSlotBank& bank,
                                 const float* consts,
                                 uint32_t pixelX, uint32_t pixelY, uint32_t seed) {
    switch (node.op) {

    case OP_MAKE_RAY: {
        Ray& r = bank.rays[bank.nRay++ & 15];
        // in0 = origin IRFloat3 slot, in1 = dir IRFloat3 slot
        r.origin = bank.vec3s[node.in0 & 31];
        r.dir    = bank.vec3s[node.in1 & 31];
        r.tmin   = 0.001f;
        r.tmax   = 1e30f;
        break;
    }

    case OP_MAKE_SHADOW_RAY: {
        Ray& r = bank.rays[bank.nRay++ & 15];
        Hit& h = bank.hits[node.in0 & 15];
        IRFloat3& lightPos = bank.vec3s[node.in1 & 31];
        // Shadow ray from hit point toward light
        IRFloat3 hitP = {h.normal.x * 0.001f, h.normal.y * 0.001f, h.normal.z * 0.001f};
        r.origin = {hitP.x, hitP.y, hitP.z}; // offset along normal
        r.dir = {lightPos.x - hitP.x, lightPos.y - hitP.y, lightPos.z - hitP.z};
        float dist = sqrtf(r.dir.x * r.dir.x + r.dir.y * r.dir.y + r.dir.z * r.dir.z);
        if (dist > 1e-7f) { r.dir.x /= dist; r.dir.y /= dist; r.dir.z /= dist; }
        r.tmin = 0.001f;
        r.tmax = dist - 0.001f;
        break;
    }

    case OP_TRACE_CLOSEST: {
        // Simplified single-level BVH2 traversal for now
        // Full TLAS+BLAS traversal wired via external BVH pointers
        Hit& hit = bank.hits[bank.nHit++ & 15];
        Ray& ray = bank.rays[node.in0 & 15];
        hit.hit = 0;
        hit.t = ray.tmax;
        hit.materialId = -1;
        hit.primitiveId = -1;
        hit.instanceId = -1;
        // BVH traversal would go here — currently returns miss
        // Will be wired to the existing cuda_bvh_backend traversal
        break;
    }

    case OP_TRACE_ANY: {
        // Shadow test — same as closest but early exit
        bank.bools[bank.nBool++ & 7] = 0;  // 0 = not occluded (placeholder)
        break;
    }

    case OP_SHADE_DIFFUSE: {
        Hit& h = bank.hits[node.in0 & 15];
        IRFloat3& result = bank.vec3s[node.out0 & 31];
        if (h.hit) {
            // Lambert diffuse: albedo * max(dot(N, L), 0) / π
            float NdotL = fmaxf(h.normal.y, 0.0f); // simple top-down light
            result = {NdotL * 0.8f, NdotL * 0.8f, NdotL * 0.8f};
        } else {
            // Sky color for miss
            float t = 0.5f * (h.normal.y + 1.0f);
            result = {0.5f + 0.5f * t, 0.7f + 0.3f * t, 1.0f};
        }
        break;
    }

    case OP_SHADE_SPECULAR: {
        Hit& h = bank.hits[node.in0 & 15];
        IRFloat3& result = bank.vec3s[node.out0 & 31];
        // GGX placeholder
        result = {0.04f, 0.04f, 0.04f};
        break;
    }

    case OP_SHADE_EMISSIVE: {
        IRFloat3& result = bank.vec3s[node.out0 & 31];
        result = {0.0f, 0.0f, 0.0f};  // no emission by default
        break;
    }

    case OP_SAMPLE_LIGHT: {
        IRFloat3& light = bank.vec3s[bank.nVec3++ & 31];
        // Simple directional light
        light = {0.577f, 0.577f, 0.577f};
        break;
    }

    case OP_SAMPLE_ENVIRONMENT: {
        Ray& r = bank.rays[node.in0 & 15];
        IRFloat3& env = bank.vec3s[node.out0 & 31];
        // Gradient sky
        float t = 0.5f * (r.dir.y + 1.0f);
        env.x = (1.0f - t) * 1.0f + t * 0.5f;
        env.y = (1.0f - t) * 1.0f + t * 0.7f;
        env.z = 1.0f;
        break;
    }

    case OP_ACCUMULATE: {
        Payload& p = bank.payloads[node.in0 & 3];
        IRFloat3& rad = bank.vec3s[node.in1 & 31];
        p.radiance.x += p.throughput.x * rad.x;
        p.radiance.y += p.throughput.y * rad.y;
        p.radiance.z += p.throughput.z * rad.z;
        break;
    }

    case OP_RUSSIAN_ROULETTE: {
        Payload& p = bank.payloads[node.in0 & 3];
        float lum = 0.299f * p.throughput.x + 0.587f * p.throughput.y + 0.114f * p.throughput.z;
        // Simple hash RNG
        uint32_t h = seed ^ (pixelX * 1973 + pixelY * 9277 + p.depth * 26699);
        h = (h ^ 61) ^ (h >> 16); h *= 9; h ^= h >> 4; h *= 0x27d4eb2d; h ^= h >> 15;
        float rng = (float)(h & 0xFFFF) / 65535.0f;
        if (rng > fmaxf(lum, 0.05f)) p.depth = 999; // terminate
        else {
            float scale = 1.0f / fmaxf(lum, 0.05f);
            p.throughput.x *= scale;
            p.throughput.y *= scale;
            p.throughput.z *= scale;
        }
        break;
    }

    case OP_REFLECT: {
        IRFloat3& d = bank.vec3s[node.in0 & 31];
        IRFloat3& n = bank.vec3s[node.in1 & 31];
        IRFloat3& r = bank.vec3s[node.out0 & 31];
        float dot = d.x * n.x + d.y * n.y + d.z * n.z;
        r.x = d.x - 2.0f * dot * n.x;
        r.y = d.y - 2.0f * dot * n.y;
        r.z = d.z - 2.0f * dot * n.z;
        break;
    }

    case OP_REFRACT: {
        IRFloat3& d = bank.vec3s[node.in0 & 31];
        IRFloat3& n = bank.vec3s[node.in1 & 31];
        IRFloat3& r = bank.vec3s[node.out0 & 31];
        float eta = 1.0f / 1.5f;  // glass IOR default
        float dot = d.x * n.x + d.y * n.y + d.z * n.z;
        float k = 1.0f - eta * eta * (1.0f - dot * dot);
        if (k < 0.0f) {
            // Total internal reflection
            r.x = d.x - 2.0f * dot * n.x;
            r.y = d.y - 2.0f * dot * n.y;
            r.z = d.z - 2.0f * dot * n.z;
        } else {
            float sq = sqrtf(k);
            r.x = eta * d.x - (eta * dot + sq) * n.x;
            r.y = eta * d.y - (eta * dot + sq) * n.y;
            r.z = eta * d.z - (eta * dot + sq) * n.z;
        }
        break;
    }

    case OP_BRANCH: {
        // Handled by the main loop (conditional jump)
        break;
    }

    case OP_TERMINATE:
    case OP_DENOISE:
    case OP_ACCUMULATE_FRAME:
    default:
        break;
    }
}

// ═══════════════════════════════════════════════════════════════
// Main execution kernel — one thread per pixel
// ═══════════════════════════════════════════════════════════════

__global__ void irExecKernel(const Program* prog, float4* output,
                              uint32_t width, uint32_t height, uint32_t frameIdx) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Per-thread slot bank (register file)
    IRSlotBank bank;
    memset(&bank, 0, sizeof(bank));

    // Initialize camera ray slots (slot 0 = origin, slot 1 = direction)
    float u = ((float)x + 0.5f) / width  * 2.0f - 1.0f;
    float v = ((float)y + 0.5f) / height * 2.0f - 1.0f;
    float aspect = (float)width / (float)height;
    bank.vec3s[0] = {0.0f, 1.0f, 3.0f};  // camera pos (default)
    bank.vec3s[1] = {u * aspect, -v, -1.0f};  // ray dir (pinhole)
    bank.nVec3 = 2;

    // Initialize payload
    bank.payloads[0] = {{0, 0, 0}, {1, 1, 1}, 0, 0};
    bank.nPayload = 1;

    uint32_t seed = frameIdx * 65537 + x * 1973 + y * 9277;

    // Execute IR nodes sequentially
    for (uint32_t i = 0; i < prog->nodeCount; i++) {
        const Node& node = prog->nodes[i];

        // Branch handling
        if (node.op == OP_BRANCH) {
            // Check condition in bool/hit slot
            bool cond = false;
            SlotType st = prog->slots[node.in0].type;
            if (st == SLOT_HIT)  cond = bank.hits[node.in0 & 15].hit != 0;
            if (st == SLOT_BOOL) cond = bank.bools[node.in0 & 7] != 0;
            if (cond && node.branchTarget < prog->nodeCount) {
                i = node.branchTarget - 1;  // -1 because loop increments
                continue;
            }
        }

        if (node.op == OP_TERMINATE) break;

        // Check depth limit
        if (bank.payloads[0].depth >= (int)prog->maxDepth) break;

        execNode(node, bank, prog->consts, x, y, seed);
        seed = seed * 1664525 + 1013904223;  // LCG advance
    }

    // Write output
    uint32_t idx = y * width + x;
    Payload& p = bank.payloads[0];
    output[idx] = make_float4(p.radiance.x, p.radiance.y, p.radiance.z, 1.0f);
}

// ═══════════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════════

static struct {
    Program*  d_program;    // device-side program copy
    float4*   d_output;     // device output framebuffer
    cudaStream_t stream;
    uint32_t  width, height;
    uint32_t  frameIdx;
    bool      ready;
} g_irExec = {};

extern "C" {

int ir_exec_init(uint32_t width, uint32_t height) {
    if (g_irExec.ready) return 1;

    cudaStreamCreate(&g_irExec.stream);
    cudaMalloc(&g_irExec.d_program, sizeof(Program));
    cudaMalloc(&g_irExec.d_output, width * height * sizeof(float4));
    g_irExec.width = width;
    g_irExec.height = height;
    g_irExec.frameIdx = 0;
    g_irExec.ready = true;

    fprintf(stderr, "[IR:Exec] Initialized %ux%u\n", width, height);
    return 1;
}

// Upload program to GPU and execute
float ir_exec_run(const Program* hostProgram, float4* hostOutput) {
    if (!g_irExec.ready || !hostProgram || !hostProgram->valid()) return -1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Upload program
    cudaMemcpyAsync(g_irExec.d_program, hostProgram, sizeof(Program),
                     cudaMemcpyHostToDevice, g_irExec.stream);

    cudaEventRecord(start, g_irExec.stream);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((g_irExec.width + 15) / 16, (g_irExec.height + 15) / 16);
    irExecKernel<<<grid, block, 0, g_irExec.stream>>>(
        g_irExec.d_program, g_irExec.d_output,
        g_irExec.width, g_irExec.height, g_irExec.frameIdx++);

    cudaEventRecord(stop, g_irExec.stream);

    // Download output if requested
    if (hostOutput) {
        cudaMemcpyAsync(hostOutput, g_irExec.d_output,
                         g_irExec.width * g_irExec.height * sizeof(float4),
                         cudaMemcpyDeviceToHost, g_irExec.stream);
    }

    cudaStreamSynchronize(g_irExec.stream);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// Get device output pointer (for Vulkan interop — no download needed)
float4* ir_exec_output_ptr() {
    return g_irExec.d_output;
}

void ir_exec_shutdown() {
    if (!g_irExec.ready) return;
    cudaFree(g_irExec.d_program);
    cudaFree(g_irExec.d_output);
    cudaStreamDestroy(g_irExec.stream);
    g_irExec.ready = false;
}

} // extern "C"
