#include <iostream>
#include <dlfcn.h>

// Function pointer types based on 'nm' output
typedef int (*PFN_GetAPIVersion)(unsigned int*);
typedef int (*PFN_GetGPUArchitecture)(void*, int*);

int main() {
    std::cout << "========================================================\n";
    std::cout << "  DLSS V100 ARCHITECTURE PROBE\n";
    std::cout << "========================================================\n";

    void* handle = dlopen("/workspaces/codespace/linux-unpacked/resources/binaries/libnvidia-ngx-dlss.so.3.1.1", RTLD_NOW);
    if (!handle) {
        std::cerr << "[FAIL] Could not load DLSS library: " << dlerror() << std::endl;
        return 1;
    }

    PFN_GetAPIVersion getVersion = (PFN_GetAPIVersion)dlsym(handle, "NVSDK_NGX_GetAPIVersion");
    PFN_GetGPUArchitecture getArch = (PFN_GetGPUArchitecture)dlsym(handle, "NVSDK_NGX_GetGPUArchitecture");

    if (!getVersion || !getArch) {
        std::cerr << "[FAIL] Could not find symbols in library" << std::endl;
        return 1;
    }

    unsigned int apiVer = 0;
    getVersion(&apiVer);
    std::cout << "[INFO] DLSS API Version: " << apiVer << std::endl;

    int arch = 0;
    // NVSDK_NGX_GetGPUArchitecture typically takes a pointer to some internal state,
    // but let's see if it works with nullptr for a quick probe.
    getArch(nullptr, &arch);
    
    std::cout << "[INFO] Reported GPU Architecture: 0x" << std::hex << arch << std::dec << std::endl;

    if (arch == 0x70) {
        std::cout << "[INFO] Detected VOLTA (V100). Standard DLSS check will likely fail.\n";
    } else if (arch >= 0x75) {
        std::cout << "[INFO] Detected TURING+ (RTX). DLSS native path is clear.\n";
    }

    std::cout << "========================================================\n";
    dlclose(handle);
    return 0;
}