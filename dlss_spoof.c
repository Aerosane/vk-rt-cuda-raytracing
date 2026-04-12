#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

int cudaGetDeviceProperties_v2(void* prop, int device) {
    static int (*real_func)(void*, int) = NULL;
    if (!real_func) real_func = (int (*)(void*, int))dlsym(RTLD_NEXT, "cudaGetDeviceProperties_v2");
    
    int res = real_func(prop, device);
    if (res == 0) {
        printf("[CuRT-Spoof] Intercepted cudaGetDeviceProperties_v2\n");
        
        // Brute force search for the architecture fields (major=7, minor=0)
        int* p = (int*)prop;
        // The struct is usually ~1KB, so search about 256 ints
        for (int i = 0; i < 256; i++) {
            if (p[i] == 7 && p[i+1] == 0) {
                printf("  -> Found Arch at offset %d. Spoofing to 7.5\n", i);
                p[i] = 7;
                p[i+1] = 5;
                // No break, in case of multiple matches
            }
        }
        strcpy((char*)prop, "NVIDIA GeForce RTX 2080 Ti (CuRT Spoof)");
    }
    return res;
}
