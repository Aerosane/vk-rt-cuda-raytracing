#include <iostream>
#include <dlfcn.h>
#include <cuda_runtime.h>

// Mock OptiX 7 structures
struct OptixFunctionTable {
    void* padding[4]; // skip first 4
    int (*optixLaunch)(void*, cudaStream_t, void*, size_t, void*, unsigned int, unsigned int, unsigned int);
    void* more_padding[100];
};

typedef int (*PFN_optixQueryFunctionTable)(int, void*);

int main() {
    std::cout << "=== OptiX App Simulation ===" << std::endl;
    
    // Load our shim
    void* handle = dlopen("./libnvoptix.so.1", RTLD_NOW);
    if (!handle) {
        std::cerr << "Failed to load libnvoptix.so.1: " << dlerror() << std::endl;
        return 1;
    }
    
    PFN_optixQueryFunctionTable queryTable = (PFN_optixQueryFunctionTable)dlsym(handle, "optixQueryFunctionTable");
    if (!queryTable) {
        std::cerr << "Failed to find optixQueryFunctionTable" << std::endl;
        return 1;
    }
    
    OptixFunctionTable table = {};
    if (queryTable(0, &table) != 0) {
        std::cerr << "optixQueryFunctionTable failed" << std::endl;
        return 1;
    }
    
    std::cout << "[App] Calling optixLaunch..." << std::endl;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Call the intercepted function
    table.optixLaunch(nullptr, stream, nullptr, 0, nullptr, 1920, 1080, 1);
    
    cudaStreamSynchronize(stream);
    std::cout << "[App] Done!" << std::endl;
    
    dlclose(handle);
    return 0;
}