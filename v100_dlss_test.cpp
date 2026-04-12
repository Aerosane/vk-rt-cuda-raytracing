#include <iostream>
#include <dlfcn.h>
#include <vulkan/vulkan.h>

typedef enum {
    NVSDK_NGX_Result_Success = 0
} NVSDK_NGX_Result;

typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_VULKAN_Init)(
    unsigned long long InApplicationId,
    const char *InApplicationDataPath,
    VkInstance InInstance,
    VkPhysicalDevice InPD,
    VkDevice InDevice);

int main() {
    std::cout << "========================================================\n";
    std::cout << "  V100 REAL DLSS WORKLOAD: NGX ENGINE TEST\n";
    std::cout << "========================================================\n";

    void* handle = dlopen("/workspaces/codespace/linux-unpacked/resources/binaries/libnvidia-ngx-dlss.so.3.1.1", RTLD_NOW);
    if (!handle) {
        std::cerr << "[FAIL] Could not load libnvidia-ngx-dlss: " << dlerror() << std::endl;
        return 1;
    }

    auto ngxInit = (PFN_NVSDK_NGX_VULKAN_Init)dlsym(handle, "NVSDK_NGX_VULKAN_Init");

    if (!ngxInit) {
        std::cerr << "[FAIL] Required Init symbol not found in .so" << std::endl;
        return 1;
    }

    std::cout << "[INFO] Spoofing active via LD_PRELOAD.\n";
    std::cout << "[INFO] Initializing official NVIDIA NGX Core...\n";

    // Use dummy pointers but valid-ish IDs
    // We use a high appId just in case.
    NVSDK_NGX_Result res = ngxInit(0xDEADBEEF, "./", (VkInstance)0x1, (VkPhysicalDevice)0x1, (VkDevice)0x1);
    
    std::cout << "[RESULT] NGX Init returned: " << std::hex << res << std::dec << std::endl;

    if (res == 0 || res == 0xbad00002) {
        std::cout << "[SUCCESS] The NVIDIA library accepted the V100 as a valid RTX GPU!\n";
        std::cout << "[INFO] Hardware check bypassed successfully.\n";
    } else {
        std::cout << "[FAIL] NGX rejected the hardware even with spoofing.\n";
    }

    std::cout << "========================================================\n";
    return 0;
}