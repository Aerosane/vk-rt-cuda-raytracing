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
    IRFloat3 vec3s[32];
    float    scalars[16];
    int      bools[8];
    int nRay, nHit, nPayload, nMat, nVec3, nScalar, nBool;
};

// ═══════════════════════════════════════════════════════════════
// BVH2 scene data — passed to kernel
// ═══════════════════════════════════════════════════════════════

struct BVH2SceneGPU {
    const uint32_t* blasNodes;    // stackless BVH2 nodes (8 uint32 each)
    const float*    blasTris;     // packed tris (12 floats each)
    const uint32_t* tlasNodes;    // TLAS BVH2 nodes (8 uint32 each)
    const float*    instances;    // InstanceGPU as floats (36 per inst = 9 vec4)
    int numBlasNodes;
    int numTlasNodes;
    int numInstances;
};

// ═══════════════════════════════════════════════════════════════
// Stackless BVH2 TLAS+BLAS traversal — matches SPIR-V rewriter
// ═══════════════════════════════════════════════════════════════

__device__ static bool slab_test(IRFloat3 ro, IRFloat3 invRd,
                                  const float* node, float bestT) {
    float t1x = (node[0] - ro.x) * invRd.x;
    float t2x = (node[3] - ro.x) * invRd.x;
    float t1y = (node[1] - ro.y) * invRd.y;
    float t2y = (node[4] - ro.y) * invRd.y;
    float t1z = (node[2] - ro.z) * invRd.z;
    float t2z = (node[5] - ro.z) * invRd.z;
    float tmin = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
    float tmax = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));
    return tmax >= fmaxf(tmin, 0.0f) && tmin < bestT;
}

__device__ static bool moller_trumbore(IRFloat3 ro, IRFloat3 rd,
                                        const float* tri,
                                        float& t, float& u, float& v) {
    IRFloat3 v0 = {tri[0], tri[1], tri[2]};
    IRFloat3 v1 = {tri[3], tri[4], tri[5]};
    IRFloat3 v2 = {tri[6], tri[7], tri[8]};
    IRFloat3 e1 = {v1.x-v0.x, v1.y-v0.y, v1.z-v0.z};
    IRFloat3 e2 = {v2.x-v0.x, v2.y-v0.y, v2.z-v0.z};
    IRFloat3 h = {rd.y*e2.z - rd.z*e2.y, rd.z*e2.x - rd.x*e2.z, rd.x*e2.y - rd.y*e2.x};
    float a = e1.x*h.x + e1.y*h.y + e1.z*h.z;
    if (fabsf(a) < 1e-7f) return false;
    float f = 1.0f / a;
    IRFloat3 s = {ro.x-v0.x, ro.y-v0.y, ro.z-v0.z};
    u = f * (s.x*h.x + s.y*h.y + s.z*h.z);
    if (u < 0.0f || u > 1.0f) return false;
    IRFloat3 q = {s.y*e1.z - s.z*e1.y, s.z*e1.x - s.x*e1.z, s.x*e1.y - s.y*e1.x};
    v = f * (rd.x*q.x + rd.y*q.y + rd.z*q.z);
    if (v < 0.0f || u + v > 1.0f) return false;
    t = f * (e2.x*q.x + e2.y*q.y + e2.z*q.z);
    return t > 1e-5f;
}

__device__ static void transformRay(const float* inst,
                                     IRFloat3 wO, IRFloat3 wD,
                                     IRFloat3& lO, IRFloat3& lD) {
    const float* inv = inst + 12; // invTransform at offset 12
    lO.x = inv[0]*wO.x + inv[1]*wO.y + inv[2]*wO.z  + inv[3];
    lO.y = inv[4]*wO.x + inv[5]*wO.y + inv[6]*wO.z  + inv[7];
    lO.z = inv[8]*wO.x + inv[9]*wO.y + inv[10]*wO.z + inv[11];
    lD.x = inv[0]*wD.x + inv[1]*wD.y + inv[2]*wD.z;
    lD.y = inv[4]*wD.x + inv[5]*wD.y + inv[6]*wD.z;
    lD.z = inv[8]*wD.x + inv[9]*wD.y + inv[10]*wD.z;
}

__device__ static void traceRayBVH2(const BVH2SceneGPU& scene,
                                     IRFloat3 origin, IRFloat3 dir,
                                     float tmin, float tmax,
                                     Hit& hit, bool anyHit) {
    hit.hit = 0; hit.t = tmax;
    hit.primitiveId = -1; hit.instanceId = -1; hit.materialId = -1;
    hit.normal = {0, 0, 0};

    if (!scene.tlasNodes || scene.numTlasNodes <= 0) return;

    IRFloat3 invDir = {1.0f / (fabsf(dir.x) > 1e-8f ? dir.x : copysignf(1e-8f, dir.x)),
                       1.0f / (fabsf(dir.y) > 1e-8f ? dir.y : copysignf(1e-8f, dir.y)),
                       1.0f / (fabsf(dir.z) > 1e-8f ? dir.z : copysignf(1e-8f, dir.z))};
    float bestT = tmax;

    // TLAS traversal
    int tlasNi = 0;
    for (int tlasIter = 0; tlasNi >= 0 && tlasNi < scene.numTlasNodes && tlasIter < scene.numTlasNodes * 2; tlasIter++) {
        const uint32_t* tn = scene.tlasNodes + tlasNi * 8;
        int leaf_enc, skip_val;
        memcpy(&leaf_enc, &tn[6], 4);
        memcpy(&skip_val, &tn[7], 4);

        if (!slab_test(origin, invDir, (const float*)tn, bestT)) {
            tlasNi = skip_val;
            continue;
        }

        if (leaf_enc == 0) { tlasNi++; continue; } // internal

        // TLAS leaf → instance
        int enc = -(leaf_enc + 2);
        int instIdx = enc >> 3;
        if (instIdx < 0 || instIdx >= scene.numInstances) { tlasNi = skip_val; continue; }

        const float* inst = scene.instances + instIdx * 36;
        uint32_t blasNodeOff, blasTriOff;
        memcpy(&blasNodeOff, &inst[26], 4);
        memcpy(&blasTriOff,  &inst[31], 4);

        IRFloat3 lO, lD;
        transformRay(inst, origin, dir, lO, lD);
        IRFloat3 lInvD = {1.0f / (fabsf(lD.x) > 1e-8f ? lD.x : copysignf(1e-8f, lD.x)),
                          1.0f / (fabsf(lD.y) > 1e-8f ? lD.y : copysignf(1e-8f, lD.y)),
                          1.0f / (fabsf(lD.z) > 1e-8f ? lD.z : copysignf(1e-8f, lD.z))};

        // BLAS traversal
        int blasNi = 0;
        for (int blasIter = 0; blasNi >= 0 && blasIter < 4096; blasIter++) {
            int absNi = (int)blasNodeOff + blasNi;
            if (absNi < 0 || absNi >= scene.numBlasNodes) break;
            const uint32_t* bn = scene.blasNodes + absNi * 8;
            int bleaf, bskip;
            memcpy(&bleaf, &bn[6], 4);
            memcpy(&bskip, &bn[7], 4);

            if (!slab_test(lO, lInvD, (const float*)bn, bestT)) {
                blasNi = bskip; continue;
            }
            if (bleaf == 0) { blasNi++; continue; }

            // Leaf → triangles
            int benc = -(bleaf + 2);
            int triStart = benc >> 3;
            int triCount = (benc & 7) + 1;

            for (int ti = 0; ti < triCount; ti++) {
                int triIdx = (int)blasTriOff + triStart + ti;
                const float* tri = scene.blasTris + triIdx * 12;
                float ht, hu, hv;
                if (moller_trumbore(lO, lD, tri, ht, hu, hv) && ht < bestT && ht > tmin) {
                    bestT = ht;
                    hit.hit = 1; hit.t = ht;
                    hit.instanceId = instIdx;
                    int origPrim; memcpy(&origPrim, &tri[9], 4);
                    hit.primitiveId = origPrim;
                    hit.materialId = instIdx;
                    // Face normal in local space
                    IRFloat3 e1 = {tri[3]-tri[0], tri[4]-tri[1], tri[5]-tri[2]};
                    IRFloat3 e2 = {tri[6]-tri[0], tri[7]-tri[1], tri[8]-tri[2]};
                    IRFloat3 n = {e1.y*e2.z-e1.z*e2.y, e1.z*e2.x-e1.x*e2.z, e1.x*e2.y-e1.y*e2.x};
                    float nl = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
                    if (nl > 1e-7f) { n.x/=nl; n.y/=nl; n.z/=nl; }
                    // World normal via transpose of upper-left 3×3 of transform
                    hit.normal.x = inst[0]*n.x + inst[4]*n.y + inst[8]*n.z;
                    hit.normal.y = inst[1]*n.x + inst[5]*n.y + inst[9]*n.z;
                    hit.normal.z = inst[2]*n.x + inst[6]*n.y + inst[10]*n.z;
                    float wnl = sqrtf(hit.normal.x*hit.normal.x + hit.normal.y*hit.normal.y + hit.normal.z*hit.normal.z);
                    if (wnl > 1e-7f) { hit.normal.x/=wnl; hit.normal.y/=wnl; hit.normal.z/=wnl; }
                    if (anyHit) return;
                }
            }
            blasNi = bskip;
        }
        tlasNi = skip_val;
    }
}

// ═══════════════════════════════════════════════════════════════
// Per-op execution
// ═══════════════════════════════════════════════════════════════

__device__ static void execNode(const Node& node, IRSlotBank& bank,
                                 const BVH2SceneGPU& scene,
                                 const float* consts,
                                 uint32_t px, uint32_t py, uint32_t seed) {
    switch (node.op) {

    case OP_MAKE_RAY: {
        Ray& r = bank.rays[node.out0 & 15];
        r.origin = bank.vec3s[node.in0 & 31];
        r.dir    = bank.vec3s[node.in1 & 31];
        r.tmin = 0.001f; r.tmax = 1e30f;
        break;
    }

    case OP_MAKE_SHADOW_RAY: {
        Ray& r = bank.rays[node.out0 & 15];
        Hit& h = bank.hits[node.in0 & 15];
        IRFloat3& lp = bank.vec3s[node.in1 & 31];
        // Offset origin along normal to avoid self-intersection
        r.origin = {h.normal.x * 0.002f, h.normal.y * 0.002f, h.normal.z * 0.002f};
        // If we have a valid hit point, use hit position
        if (h.hit) {
            Ray& srcRay = bank.rays[0]; // use the primary ray
            r.origin.x = srcRay.origin.x + srcRay.dir.x * h.t + h.normal.x * 0.002f;
            r.origin.y = srcRay.origin.y + srcRay.dir.y * h.t + h.normal.y * 0.002f;
            r.origin.z = srcRay.origin.z + srcRay.dir.z * h.t + h.normal.z * 0.002f;
        }
        r.dir = {lp.x - r.origin.x, lp.y - r.origin.y, lp.z - r.origin.z};
        float dist = sqrtf(r.dir.x*r.dir.x + r.dir.y*r.dir.y + r.dir.z*r.dir.z);
        if (dist > 1e-7f) { r.dir.x/=dist; r.dir.y/=dist; r.dir.z/=dist; }
        r.tmin = 0.001f; r.tmax = dist - 0.001f;
        break;
    }

    case OP_TRACE_CLOSEST: {
        Hit& hit = bank.hits[node.out0 & 15];
        Ray& ray = bank.rays[node.in0 & 15];
        traceRayBVH2(scene, ray.origin, ray.dir, ray.tmin, ray.tmax, hit, false);
        break;
    }

    case OP_TRACE_ANY: {
        Ray& ray = bank.rays[node.in0 & 15];
        Hit tmpHit;
        traceRayBVH2(scene, ray.origin, ray.dir, ray.tmin, ray.tmax, tmpHit, true);
        bank.bools[node.out0 & 7] = tmpHit.hit;
        break;
    }

    case OP_SHADE_DIFFUSE: {
        Hit& h = bank.hits[node.in0 & 15];
        IRFloat3& result = bank.vec3s[node.out0 & 31];
        if (h.hit) {
            // Lambert: N·L with simple directional light
            float NdotL = fmaxf(h.normal.x * 0.577f + h.normal.y * 0.577f + h.normal.z * 0.577f, 0.0f);
            // Color by instance for variety
            float r = 0.7f + 0.3f * sinf((float)h.instanceId * 1.1f);
            float g = 0.7f + 0.3f * sinf((float)h.instanceId * 2.3f);
            float b = 0.7f + 0.3f * sinf((float)h.instanceId * 3.7f);
            result = {r * NdotL, g * NdotL, b * NdotL};
        } else {
            // Sky gradient
            float t = 0.5f * (bank.rays[0].dir.y + 1.0f);
            result = {(1.0f-t)*1.0f + t*0.5f, (1.0f-t)*1.0f + t*0.7f, 1.0f};
        }
        break;
    }

    case OP_SHADE_SPECULAR: {
        IRFloat3& result = bank.vec3s[node.out0 & 31];
        result = {0.04f, 0.04f, 0.04f};
        break;
    }

    case OP_SHADE_EMISSIVE: {
        IRFloat3& result = bank.vec3s[node.out0 & 31];
        result = {0.0f, 0.0f, 0.0f};
        break;
    }

    case OP_SAMPLE_LIGHT: {
        IRFloat3& light = bank.vec3s[node.out0 & 31];
        light = {0.577f, 0.577f, 0.577f};
        break;
    }

    case OP_SAMPLE_ENVIRONMENT: {
        Ray& r = bank.rays[node.in0 & 15];
        IRFloat3& env = bank.vec3s[node.out0 & 31];
        float t = 0.5f * (r.dir.y + 1.0f);
        env = {(1.0f-t)*1.0f + t*0.5f, (1.0f-t)*1.0f + t*0.7f, 1.0f};
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
        float lum = 0.299f*p.throughput.x + 0.587f*p.throughput.y + 0.114f*p.throughput.z;
        uint32_t h = seed ^ (px*1973 + py*9277 + p.depth*26699);
        h = (h^61)^(h>>16); h*=9; h^=h>>4; h*=0x27d4eb2d; h^=h>>15;
        float rng = (float)(h & 0xFFFF) / 65535.0f;
        if (rng > fmaxf(lum, 0.05f)) p.depth = 999;
        else {
            float s = 1.0f / fmaxf(lum, 0.05f);
            p.throughput.x *= s; p.throughput.y *= s; p.throughput.z *= s;
        }
        break;
    }

    case OP_REFLECT: {
        IRFloat3& d = bank.vec3s[node.in0 & 31];
        IRFloat3& n = bank.vec3s[node.in1 & 31];
        IRFloat3& r = bank.vec3s[node.out0 & 31];
        float dot = d.x*n.x + d.y*n.y + d.z*n.z;
        r = {d.x - 2.0f*dot*n.x, d.y - 2.0f*dot*n.y, d.z - 2.0f*dot*n.z};
        break;
    }

    case OP_REFRACT: {
        IRFloat3& d = bank.vec3s[node.in0 & 31];
        IRFloat3& n = bank.vec3s[node.in1 & 31];
        IRFloat3& r = bank.vec3s[node.out0 & 31];
        float eta = 1.0f / 1.5f;
        float dot = d.x*n.x + d.y*n.y + d.z*n.z;
        float k = 1.0f - eta*eta*(1.0f - dot*dot);
        if (k < 0.0f) {
            r = {d.x - 2.0f*dot*n.x, d.y - 2.0f*dot*n.y, d.z - 2.0f*dot*n.z};
        } else {
            float sq = sqrtf(k);
            r = {eta*d.x-(eta*dot+sq)*n.x, eta*d.y-(eta*dot+sq)*n.y, eta*d.z-(eta*dot+sq)*n.z};
        }
        break;
    }

    case OP_BRANCH:
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

__global__ void irExecKernel(const Program* prog, const BVH2SceneGPU* scene,
                              float4* output, uint32_t width, uint32_t height,
                              uint32_t frameIdx) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    IRSlotBank bank;
    memset(&bank, 0, sizeof(bank));

    // Camera ray: slot 0 = origin, slot 1 = direction
    float u = ((float)x + 0.5f) / width  * 2.0f - 1.0f;
    float v = ((float)y + 0.5f) / height * 2.0f - 1.0f;
    float aspect = (float)width / (float)height;
    bank.vec3s[0] = {0.0f, 1.0f, 3.0f};         // camera pos
    bank.vec3s[1] = {u * aspect, -v, -1.0f};     // pinhole dir
    // Normalize direction
    IRFloat3& d = bank.vec3s[1];
    float dlen = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
    if (dlen > 0) { d.x/=dlen; d.y/=dlen; d.z/=dlen; }
    // Init ALL payloads with unit throughput (builder may place payload at any slot)
    for (int pi = 0; pi < 4; pi++)
        bank.payloads[pi] = {{0,0,0}, {1,1,1}, 0, 0};

    uint32_t seed = frameIdx * 65537 + x * 1973 + y * 9277;

    for (uint32_t i = 0; i < prog->nodeCount; i++) {
        const Node& node = prog->nodes[i];
        if (node.op == OP_BRANCH) {
            bool cond = false;
            SlotType st = prog->slots[node.in0].type;
            if (st == SLOT_HIT)  cond = bank.hits[node.in0 & 15].hit != 0;
            if (st == SLOT_BOOL) cond = bank.bools[node.in0 & 7] != 0;
            if (cond && node.branchTarget < prog->nodeCount) {
                i = node.branchTarget - 1;
                continue;
            }
        }
        if (node.op == OP_TERMINATE) break;
        if (bank.payloads[0].depth >= (int)prog->maxDepth) break;
        execNode(node, bank, *scene, prog->consts, x, y, seed);
        seed = seed * 1664525 + 1013904223;
    }

    uint32_t idx = y * width + x;
    // Find the payload with accumulated radiance (check all slots)
    IRFloat3 rad = {0,0,0};
    for (int pi = 0; pi < 4; pi++) {
        Payload& p = bank.payloads[pi];
        if (p.radiance.x != 0 || p.radiance.y != 0 || p.radiance.z != 0) {
            rad = p.radiance;
            break;
        }
    }
    output[idx] = make_float4(rad.x, rad.y, rad.z, 1.0f);
}

// ═══════════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════════

static struct {
    Program*       d_program;
    BVH2SceneGPU*  d_scene;
    float4*        d_output;
    cudaStream_t   stream;
    uint32_t       width, height;
    uint32_t       frameIdx;
    bool           ready;
} g_irExec = {};

// Host-side scene data (updated by layer)
static BVH2SceneGPU g_hostScene = {};

extern "C" {

int ir_exec_init(uint32_t width, uint32_t height) {
    if (g_irExec.ready) return 1;
    cudaStreamCreate(&g_irExec.stream);
    cudaMalloc(&g_irExec.d_program, sizeof(Program));
    cudaMalloc(&g_irExec.d_scene, sizeof(BVH2SceneGPU));
    cudaMalloc(&g_irExec.d_output, width * height * sizeof(float4));
    g_irExec.width = width;
    g_irExec.height = height;
    g_irExec.frameIdx = 0;
    g_irExec.ready = true;
    fprintf(stderr, "[IR:Exec] Initialized %ux%u\n", width, height);
    return 1;
}

// Set BVH2 scene data pointers (called by layer when BVH2 is ready)
void ir_exec_set_scene(const void* blasNodes, int numBlasNodes,
                        const void* blasTris,
                        const void* tlasNodes, int numTlasNodes,
                        const void* instances, int numInstances) {
    g_hostScene.blasNodes = (const uint32_t*)blasNodes;
    g_hostScene.blasTris  = (const float*)blasTris;
    g_hostScene.tlasNodes = (const uint32_t*)tlasNodes;
    g_hostScene.instances = (const float*)instances;
    g_hostScene.numBlasNodes = numBlasNodes;
    g_hostScene.numTlasNodes = numTlasNodes;
    g_hostScene.numInstances = numInstances;
    fprintf(stderr, "[IR:Exec] Scene: %d BLAS nodes, %d TLAS nodes, %d instances\n",
            numBlasNodes, numTlasNodes, numInstances);
}

float ir_exec_run(const void* hostProgram, float4* hostOutput) {
    if (!g_irExec.ready || !hostProgram) return -1.0f;
    const Program* prog = (const Program*)hostProgram;
    if (!prog->valid()) return -1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpyAsync(g_irExec.d_program, prog, sizeof(Program),
                     cudaMemcpyHostToDevice, g_irExec.stream);
    cudaMemcpyAsync(g_irExec.d_scene, &g_hostScene, sizeof(BVH2SceneGPU),
                     cudaMemcpyHostToDevice, g_irExec.stream);

    cudaEventRecord(start, g_irExec.stream);

    dim3 block(16, 16);
    dim3 grid((g_irExec.width + 15) / 16, (g_irExec.height + 15) / 16);
    irExecKernel<<<grid, block, 0, g_irExec.stream>>>(
        g_irExec.d_program, g_irExec.d_scene, g_irExec.d_output,
        g_irExec.width, g_irExec.height, g_irExec.frameIdx++);

    cudaEventRecord(stop, g_irExec.stream);

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

float4* ir_exec_output_ptr() { return g_irExec.d_output; }

void ir_exec_shutdown() {
    if (!g_irExec.ready) return;
    cudaFree(g_irExec.d_program);
    cudaFree(g_irExec.d_scene);
    cudaFree(g_irExec.d_output);
    cudaStreamDestroy(g_irExec.stream);
    g_irExec.ready = false;
}

} // extern "C"
