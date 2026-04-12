#include <iostream>
#include <dlfcn.h>
#include <vulkan/vulkan.h>

// NGX Init function signature
typedef int (*PFN_NGX_VULKAN_Init)(unsigned long long appId, const char* path, VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device);

int main() {
    std::cout << "========================================================\n";
    std::cout << "  DLSS VULKAN INITIALIZATION TEST (SPOOFED)\n";
    std::cout << "========================================================\n";

    void* handle = dlopen("/workspaces/codespace/linux-unpacked/resources/binaries/libnvidia-ngx-dlss.so.3.1.1", RTLD_NOW);
    if (!handle) {
        std::cerr << "[FAIL] Could not load DLSS library: " << dlerror() << std::endl;
        return 1;
    }

    PFN_NGX_VULKAN_Init ngxInit = (PFN_NGX_VULKAN_Init)dlsym(handle, "NVSDK_NGX_VULKAN_Init");
    if (!ngxInit) {
        std::cerr << "[FAIL] Could not find NVSDK_NGX_VULKAN_Init" << std::endl;
        return 1;
    }

    // Note: In a real test, we need a valid VkInstance/Device.
    // For this 'dry-run', we see if it rejects us before checking Vulkan.
    std::cout << "[INFO] Attempting DLSS Init with NULL Vulkan handles (Pre-check)...\n";
    int res = ngxInit(1337, ".", (VkInstance)0x1, (VkPhysicalDevice)0x1, (VkDevice)0x1);
    
    std::cout << "[RESULT] DLSS Init returned: 0x" << std::hex << res << std::dec << std::endl;

    if (res != 0) {
        std::cout << "[INFO] Initialization failed (as expected with dummy handles).\n";
        std::cout << "[INFO] If it didn't SEGFAULT, the architecture check passed!\n";
    }

    std::cout << "========================================================\n";
    dlclose(handle);
    return 0;
}