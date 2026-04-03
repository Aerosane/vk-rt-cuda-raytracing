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
    float  ir_exec_run(const void* hostProgram, float4* hostOutput);
    void   ir_exec_shutdown();
}

int main() {
    printf("═══ RT IR Pipeline Test ═══\n\n");

    // 1. Build a simple path tracer program
    rt_ir::Builder builder;
    builder.buildPathTracer(3);  // 3 bounces
    builder.setResolution(256, 256);

    const rt_ir::Program& prog = builder.program();
    printf("Program: %u nodes, %u slots, %u consts, maxDepth=%u\n",
           prog.nodeCount, prog.slotCount, prog.constCount, prog.maxDepth);
    printf("Valid: %s\n\n", prog.valid() ? "YES" : "NO");

    // Dump IR
    builder.dump(stdout);

    // 2. Initialize executor
    const uint32_t W = 256, H = 256;
    ir_exec_init(W, H);

    // 3. Execute program
    float4* output = new float4[W * H];
    float ms = ir_exec_run(&prog, output);
    printf("\nExecution: %.3f ms (%.1f Mrays/s)\n", ms, (W * H) / (ms * 1000.0f));

    // 4. Validate output — check a few pixels
    int nonBlack = 0, nonWhite = 0;
    float maxR = 0, maxG = 0, maxB = 0;
    for (uint32_t i = 0; i < W * H; i++) {
        float r = output[i].x, g = output[i].y, b = output[i].z;
        if (r > 0.001f || g > 0.001f || b > 0.001f) nonBlack++;
        if (r < 0.999f || g < 0.999f || b < 0.999f) nonWhite++;
        if (r > maxR) maxR = r;
        if (g > maxG) maxG = g;
        if (b > maxB) maxB = b;
    }
    printf("Non-black pixels: %d/%d (%.1f%%)\n", nonBlack, W * H, 100.0f * nonBlack / (W * H));
    printf("Max RGB: (%.3f, %.3f, %.3f)\n", maxR, maxG, maxB);
    printf("Center pixel: (%.4f, %.4f, %.4f)\n",
           output[H/2 * W + W/2].x, output[H/2 * W + W/2].y, output[H/2 * W + W/2].z);

    // 5. Build AO program and test
    printf("\n═══ AO Program ═══\n");
    builder.buildAO(16);
    const rt_ir::Program& aoProg = builder.program();
    printf("AO Program: %u nodes, %u slots\n", aoProg.nodeCount, aoProg.slotCount);
    builder.dump(stdout);

    float ms2 = ir_exec_run(&aoProg, output);
    printf("AO Execution: %.3f ms\n", ms2);

    // Cleanup
    delete[] output;
    ir_exec_shutdown();

    printf("\n═══ All tests passed ═══\n");
    return 0;
}
