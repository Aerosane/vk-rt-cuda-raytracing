#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstring>
#include "../layer/cuda_bvh_backend.h"

// ═══════════════════════════════════════════════════════
// CuRT-OptiX 2026 PROXY SHIM (V4 - ULTRA STABLE)
// ═══════════════════════════════════════════════════════

typedef int OptixResult;
#define OPTIX_SUCCESS 0
typedef void (*OptixFunc)(void);
typedef unsigned long long OptixTraversableHandle;

extern "C" void run_neural_recon_on_stream(cudaStream_t stream, int pixels);

// The real entry point we will call
typedef OptixResult (*PFN_optixQueryFunctionTable)(int, unsigned int, void*, const void**, void*, size_t);
static PFN_optixQueryFunctionTable real_queryTable = nullptr;

static int stub_func(...) { return 0; }

static int mock_launch(void* pipeline, cudaStream_t stream, CUdeviceptr params, size_t sz, const void* sbt, unsigned int w, unsigned int h, unsigned int d) {
    static int lc = 0;
    if (lc % 100 == 0) std::cout << "[CuRT-OptiX] >>> optixLaunch #" << lc << " intercepted (" << w << "x" << h << ")" << std::endl;
    lc++;
    run_neural_recon_on_stream(stream, w * h);
    return OPTIX_SUCCESS;
}

static int mock_accel_build(void* ctx, cudaStream_t stream, const void* opts, const void* in, unsigned int n, CUdeviceptr tmp, size_t tsz, CUdeviceptr out, size_t osz, OptixTraversableHandle* h, const void* em, unsigned int en) {
    if (h) *h = 0xdeadbeef00001000ULL;
    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixQueryFunctionTable(int abiId, unsigned int numOptions, void* options, const void** reserved, void* functionTable, size_t sizeOfTable) {
    if (!real_queryTable) {
        const char* paths[] = {
            "/usr/lib/x86_64-linux-gnu/libnvoptix.so.1",
            "/usr/local/cuda/lib64/libnvoptix.so.1"
        };
        for (const char* p : paths) {
            void* handle = dlopen(p, RTLD_LAZY);
            if (handle) {
                real_queryTable = (PFN_optixQueryFunctionTable)dlsym(handle, "optixQueryFunctionTable");
                if (real_queryTable) break;
            }
        }
    }

    std::cout << "[CuRT-OptiX] optixQueryFunctionTable (ABI=" << abiId << ")" << std::endl;

    OptixResult res = OPTIX_SUCCESS;
    if (real_queryTable) {
        res = real_queryTable(abiId, numOptions, options, reserved, functionTable, sizeOfTable);
    }

    if (functionTable) {
        OptixFunc* t = (OptixFunc*)functionTable;
        int nf = sizeOfTable / sizeof(OptixFunc);
        
        if (!real_queryTable) {
            for(int i=0; i<nf; i++) t[i] = (OptixFunc)stub_func;
        }

        if (nf > 23) t[23] = (OptixFunc)mock_accel_build;
        if (nf > 30) t[30] = (OptixFunc)mock_launch;
        
        std::cout << "[CuRT-OptiX] Successfully patched function table" << std::endl;
    }

    return res;
}
