// rt_ir_test.cu — Standalone test for the RT IR pipeline
// Builds an IR program, executes it on GPU, validates output
//
#include "rt_ir.h"
#include "rt_ir_builder.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>

extern "C" {
    int    ir_exec_init(uint32_t width, uint32_t height);
    void   ir_exec_set_scene(const void* blasNodes, int numBlasNodes,
                              const void* blasTris,
                              const void* tlasNodes, int numTlasNodes,
                              const void* instances, int numInstances);
    float  ir_exec_run(const void* hostProgram, float4* hostOutput);
    void   ir_exec_shutdown();
}

// Build a minimal test scene: ground plane + cube
// BVH2 stackless node: 8 uint32 = [bmin.xyz, bmax.xyz, leaf_enc, skip]
// Packed tri: 12 floats = [v0.xyz, v1.xyz, v2.xyz, primIdx, 0, 0]
// InstanceGPU: 36 floats = 9 vec4 (transform, invTransform, bounds, metadata)

static void buildTestScene(uint32_t** blasNodes, int* nBlasNodes,
                           float** blasTris, int* nBlasTris,
                           uint32_t** tlasNodes, int* nTlasNodes,
                           float** instances, int* nInstances) {
    // 2 triangles forming a ground plane at y=0
    static float tris[] = {
        // tri 0: (-5,0,-5) (5,0,-5) (5,0,5)
        -5,0,-5,  5,0,-5,  5,0,5,  0, 0,0,0,0,
        // tri 1: (-5,0,-5) (5,0,5) (-5,0,5)
        -5,0,-5,  5,0,5,  -5,0,5,  0, 0,0,0,0,
    };
    // Set primIdx (float-bitcast int)
    int idx0 = 0, idx1 = 1;
    memcpy(&tris[9], &idx0, 4);
    memcpy(&tris[9+12], &idx1, 4);
    *blasTris = tris;
    *nBlasTris = 2;

    // Single BLAS leaf node encompassing both tris
    // leaf_enc = -(triStart*8 + triCount + 2) = -(0*8 + 2 + 2) = -4
    // For triStart=0, triCount=2: enc = triStart*8 + (triCount-1) = 0*8+1 = 1
    // leaf_enc = -(1 + 2) = -3
    static uint32_t bNodes[8];
    float bmin[] = {-5, -0.01f, -5};
    float bmax[] = {5, 0.01f, 5};
    memcpy(&bNodes[0], bmin, 12);
    memcpy(&bNodes[3], bmax, 12);
    int leaf_enc = -((0 * 8 + 1) + 2);  // triStart=0, triCount=2 → (enc&7)+1=2
    int skip = -1;
    memcpy(&bNodes[6], &leaf_enc, 4);
    memcpy(&bNodes[7], &skip, 4);
    *blasNodes = bNodes;
    *nBlasNodes = 1;

    // Single TLAS node (leaf → instance 0)
    // For instIdx=0: enc = instIdx*8 = 0, leaf_enc = -(0+2) = -2
    static uint32_t tNodes[8];
    memcpy(&tNodes[0], bmin, 12);
    memcpy(&tNodes[3], bmax, 12);
    int tleaf = -((0 * 8) + 2);
    int tskip = -1;
    memcpy(&tNodes[6], &tleaf, 4);
    memcpy(&tNodes[7], &tskip, 4);
    *tlasNodes = tNodes;
    *nTlasNodes = 1;

    // Instance: identity transform, BLAS offset 0
    // 36 floats = 9 vec4
    static float inst[36];
    memset(inst, 0, sizeof(inst));
    // Transform (3×4 identity): rows 0-2
    inst[0] = 1; inst[5] = 1; inst[10] = 1;  // diagonal
    // InvTransform (same): rows 3-5
    inst[12] = 1; inst[17] = 1; inst[22] = 1;
    // BLAS bounds: vec4[6] = (bmin, blasNodeOff=0)
    inst[24] = -5; inst[25] = -0.01f; inst[26] = -5;
    uint32_t zero = 0;
    memcpy(&inst[26], &zero, 4);  // blasNodeOff = 0 (overwrite bmin.z — actually vec4[6].w)
    // Wait: layout is transform[12], invTransform[12], then blas bounds
    // vec4[6] = floats [24,25,26,27] = (blasMinX, blasMinY, blasMinZ, blasNodeOff)
    inst[24] = -5; inst[25] = -0.01f; inst[26] = -5;
    memcpy(&inst[27], &zero, 4);  // blasNodeOff = 0
    // vec4[7] = floats [28,29,30,31] = (blasMaxX, blasMaxY, blasMaxZ, blasTriOff)
    inst[28] = 5; inst[29] = 0.01f; inst[30] = 5;
    memcpy(&inst[31], &zero, 4);  // blasTriOff = 0
    // vec4[8] = floats [32,33,34,35] = (customIdx, sbtOffset, mask, flags)
    uint32_t mask = 0xFF;
    memcpy(&inst[34], &mask, 4);  // instanceMask = 0xFF
    *instances = inst;
    *nInstances = 1;
}

int main() {
    printf("═══ RT IR Pipeline Test (BVH2 Traversal) ═══\n\n");

    // 1. Build test scene
    uint32_t *blasN, *tlasN;
    float *blasT, *instF;
    int nBN, nBT, nTN, nI;
    buildTestScene(&blasN, &nBN, &blasT, &nBT, &tlasN, &nTN, &instF, &nI);
    printf("Scene: %d BLAS nodes, %d tris, %d TLAS nodes, %d instances\n",
           nBN, nBT, nTN, nI);

    // Upload scene to GPU
    uint32_t *d_blasN, *d_tlasN;
    float *d_blasT, *d_inst;
    cudaMalloc(&d_blasN, nBN * 8 * sizeof(uint32_t));
    cudaMalloc(&d_blasT, nBT * 12 * sizeof(float));
    cudaMalloc(&d_tlasN, nTN * 8 * sizeof(uint32_t));
    cudaMalloc(&d_inst, nI * 36 * sizeof(float));
    cudaMemcpy(d_blasN, blasN, nBN * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blasT, blasT, nBT * 12 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tlasN, tlasN, nTN * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inst, instF, nI * 36 * sizeof(float), cudaMemcpyHostToDevice);

    // 2. Build path tracer program
    rt_ir::Builder builder;
    builder.buildPathTracer(2);
    builder.setResolution(512, 512);
    const rt_ir::Program& prog = builder.program();
    printf("Program: %u nodes, %u slots, maxDepth=%u\n",
           prog.nodeCount, prog.slotCount, prog.maxDepth);
    builder.dump(stdout);

    // 3. Init executor with scene
    const uint32_t W = 512, H = 512;
    ir_exec_init(W, H);
    ir_exec_set_scene(d_blasN, nBN, d_blasT, d_tlasN, nTN, d_inst, nI);

    // 4. Execute
    float4* output = new float4[W * H];
    float ms = ir_exec_run(&prog, output);
    printf("\nExecution: %.3f ms (%.1f Mrays/s)\n", ms, (W * H) / (ms * 1000.0f));

    // 5. Validate — count hits
    int hits = 0, misses = 0;
    float maxR = 0, maxG = 0, maxB = 0;
    for (uint32_t i = 0; i < W * H; i++) {
        float r = output[i].x, g = output[i].y, b = output[i].z;
        if (r > 0.001f || g > 0.001f || b > 0.001f) hits++;
        else misses++;
        if (r > maxR) maxR = r;
        if (g > maxG) maxG = g;
        if (b > maxB) maxB = b;
    }
    printf("Hits: %d  Misses: %d  (%.1f%% hit rate)\n",
           hits, misses, 100.0f * hits / (W * H));
    printf("Max RGB: (%.3f, %.3f, %.3f)\n", maxR, maxG, maxB);
    printf("Center pixel: (%.4f, %.4f, %.4f)\n",
           output[H/2 * W + W/2].x, output[H/2 * W + W/2].y, output[H/2 * W + W/2].z);

    // Cleanup
    delete[] output;
    cudaFree(d_blasN); cudaFree(d_blasT); cudaFree(d_tlasN); cudaFree(d_inst);
    ir_exec_shutdown();

    printf("\n═══ Test complete ═══\n");
    return 0;
}
