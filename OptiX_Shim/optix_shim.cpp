#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstring>

typedef int OptixResult;
#define OPTIX_SUCCESS 0
typedef void (*OptixFunc)(void);
typedef OptixResult (*PFN_optixQueryFunctionTable)(int, unsigned int, void*, const void**, void*, size_t);

extern "C" void run_neural_recon_on_stream(cudaStream_t stream, int pixels);

static PFN_optixQueryFunctionTable real_queryTable = nullptr;

// We will overwrite EVERY slot to find the real one
static int my_brute_force_launch(void* p, cudaStream_t s, CUdeviceptr pr, size_t sz, const void* sbt, unsigned int w, unsigned int h, unsigned int d) {
    // We don't know the index yet, but we'll print if we get here
    static bool first = true;
    if (first) {
        std::cout << "[CuRT-HIJACK] >>> CRITICAL HIT! optixLaunch INTERCEPTED." << std::endl;
        first = false;
    }
    
    run_neural_recon_on_stream(s, w * h);
    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixQueryFunctionTable(int abiId, unsigned int numOptions, void* options, const void** reserved, void* functionTable, size_t sizeOfTable) {
    if (!real_queryTable) {
        void* handle = dlopen("/usr/lib/x86_64-linux-gnu/libnvoptix.so.1", RTLD_NOW);
        if (handle) real_queryTable = (PFN_optixQueryFunctionTable)dlsym(handle, "optixQueryFunctionTable");
    }

    OptixResult res = real_queryTable ? real_queryTable(abiId, numOptions, options, reserved, functionTable, sizeOfTable) : -1;

    if (res == OPTIX_SUCCESS && functionTable) {
        OptixFunc* t = (OptixFunc*)functionTable;
        int nf = sizeOfTable / sizeof(OptixFunc);
        std::cout << "[CuRT-HIJACK] Patching " << nf << " slots..." << std::endl;
        
        // Brute force: Overwrite the likely launch range (25-45)
        for (int i = 25; i < 45; i++) {
            if (i < nf) t[i] = (OptixFunc)my_brute_force_launch;
        }
    }
    return res;
}
