// CUDA-Vulkan interop scaffolding for V100 driver fallback.
//
// Goal: provide a proven-working external-memory + external-semaphore
// pipeline that can carry rendering work from a crashing Vulkan queue
// onto a CUDA stream. The Vulkan layer allocates an exportable buffer,
// exports the fd, and hands it to cuda_interop_import_fd; the returned
// device pointer aliases the same VRAM pages.
//
// This file intentionally has no Vulkan headers — the layer side wraps
// vkGetMemoryFdKHR / vkGetSemaphoreFdKHR and calls us with raw fds.
//
// Build: nvcc -arch=sm_70 --compiler-options=-fPIC

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

extern "C" {

struct CudaInteropMem {
    cudaExternalMemory_t ext;
    void*                devPtr;
    size_t               size;
};

struct CudaInteropSem {
    cudaExternalSemaphore_t ext;
};

// Import an opaque-fd Vulkan allocation into CUDA.
// fd ownership: CUDA takes ownership on success.
// Returns 0 on success, non-zero on failure.
int cuda_interop_import_fd(int fd, size_t size, CudaInteropMem* out) {
    if (!out || fd < 0) return -1;
    cudaExternalMemoryHandleDesc desc = {};
    desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size = size;
    desc.flags = 0;

    cudaError_t er = cudaImportExternalMemory(&out->ext, &desc);
    if (er != cudaSuccess) {
        fprintf(stderr, "[interop] cudaImportExternalMemory failed: %s\n",
                cudaGetErrorString(er));
        close(fd);
        return -2;
    }

    cudaExternalMemoryBufferDesc bufDesc = {};
    bufDesc.offset = 0;
    bufDesc.size   = size;
    bufDesc.flags  = 0;
    er = cudaExternalMemoryGetMappedBuffer(&out->devPtr, out->ext, &bufDesc);
    if (er != cudaSuccess) {
        fprintf(stderr, "[interop] cudaExternalMemoryGetMappedBuffer failed: %s\n",
                cudaGetErrorString(er));
        cudaDestroyExternalMemory(out->ext);
        return -3;
    }
    out->size = size;
    return 0;
}

int cuda_interop_destroy_mem(CudaInteropMem* m) {
    if (!m || !m->ext) return 0;
    cudaFree(m->devPtr);
    cudaDestroyExternalMemory(m->ext);
    memset(m, 0, sizeof(*m));
    return 0;
}

int cuda_interop_import_sem_fd(int fd, int isTimeline, CudaInteropSem* out) {
    if (!out || fd < 0) return -1;
    cudaExternalSemaphoreHandleDesc desc = {};
    desc.type = isTimeline
        ? cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd
        : cudaExternalSemaphoreHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.flags = 0;
    cudaError_t er = cudaImportExternalSemaphore(&out->ext, &desc);
    if (er != cudaSuccess) {
        fprintf(stderr, "[interop] cudaImportExternalSemaphore failed: %s\n",
                cudaGetErrorString(er));
        close(fd);
        return -2;
    }
    return 0;
}

int cuda_interop_signal_sem(CudaInteropSem* s, uint64_t value, int isTimeline) {
    if (!s || !s->ext) return -1;
    cudaExternalSemaphoreSignalParams p = {};
    if (isTimeline) p.params.fence.value = value;
    cudaError_t er = cudaSignalExternalSemaphoresAsync(&s->ext, &p, 1, 0);
    return er == cudaSuccess ? 0 : -2;
}

int cuda_interop_wait_sem(CudaInteropSem* s, uint64_t value, int isTimeline) {
    if (!s || !s->ext) return -1;
    cudaExternalSemaphoreWaitParams p = {};
    if (isTimeline) p.params.fence.value = value;
    cudaError_t er = cudaWaitExternalSemaphoresAsync(&s->ext, &p, 1, 0);
    return er == cudaSuccess ? 0 : -2;
}

int cuda_interop_destroy_sem(CudaInteropSem* s) {
    if (!s || !s->ext) return 0;
    cudaDestroyExternalSemaphore(s->ext);
    memset(s, 0, sizeof(*s));
    return 0;
}

// Simple sanity kernel: fills a device buffer with a known pattern so the
// Vulkan side can read it back and confirm memory is actually shared.
__global__ void k_fill_pattern(uint32_t* buf, size_t nWords, uint32_t seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < nWords) buf[i] = seed ^ (uint32_t)i;
}

int cuda_interop_fill_pattern(void* devPtr, size_t bytes, uint32_t seed) {
    if (!devPtr) return -1;
    size_t nWords = bytes / 4;
    if (nWords == 0) return 0;
    int block = 256;
    int grid  = (int)((nWords + block - 1) / block);
    if (grid > 65535) grid = 65535;
    k_fill_pattern<<<grid, block>>>((uint32_t*)devPtr, nWords, seed);
    cudaError_t er = cudaGetLastError();
    if (er != cudaSuccess) {
        fprintf(stderr, "[interop] fill kernel launch failed: %s\n",
                cudaGetErrorString(er));
        return -2;
    }
    return 0;
}

// Roundtrip self-test: allocate internal CUDA buffer, fill it, return
// expected first-word value. Used by the layer to confirm CUDA works
// before attempting Vulkan interop imports.
int cuda_interop_selftest(uint32_t* outFirstWord) {
    void* p = nullptr;
    if (cudaMalloc(&p, 1024) != cudaSuccess) return -1;
    if (cuda_interop_fill_pattern(p, 1024, 0xCAFEBABE) != 0) {
        cudaFree(p);
        return -2;
    }
    cudaDeviceSynchronize();
    uint32_t host = 0;
    if (cudaMemcpy(&host, p, 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(p);
        return -3;
    }
    cudaFree(p);
    if (outFirstWord) *outFirstWord = host;
    // Expected: seed ^ 0 = 0xCAFEBABE
    return (host == 0xCAFEBABE) ? 0 : -4;
}

} // extern "C"
