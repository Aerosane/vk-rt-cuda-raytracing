// Public C interface for cuda_vk_interop.cu
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CudaInteropMem {
    void* ext;
    void* devPtr;
    size_t size;
};

struct CudaInteropSem {
    void* ext;
};

int cuda_interop_import_fd(int fd, size_t size, struct CudaInteropMem* out);
int cuda_interop_destroy_mem(struct CudaInteropMem* m);

int cuda_interop_import_sem_fd(int fd, int isTimeline, struct CudaInteropSem* out);
int cuda_interop_signal_sem(struct CudaInteropSem* s, uint64_t value, int isTimeline);
int cuda_interop_wait_sem(struct CudaInteropSem* s, uint64_t value, int isTimeline);
int cuda_interop_destroy_sem(struct CudaInteropSem* s);

int cuda_interop_fill_pattern(void* devPtr, size_t bytes, uint32_t seed);
int cuda_interop_selftest(uint32_t* outFirstWord);

// Direct operations on imported device memory (replaces Vulkan cmd equivalents).
int cuda_interop_memset_u32(void* devPtr, uint32_t value, size_t bytes);
int cuda_interop_memcpy_d2d(void* dst, const void* src, size_t bytes);
int cuda_interop_memcpy_h2d(void* dst, const void* src, size_t bytes);

#ifdef __cplusplus
}
#endif
