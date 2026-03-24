#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cuda_fp16.h>

using namespace nvcuda;

// Helper to launch and measure cycles
template <typename Kernel>
void run_bench(const char* name, Kernel k, half* d_a, half* d_b, void* d_c, long long* d_cycles) {
    long long h_cycles = 0;
    k<<<1, 32>>>(d_a, d_b, d_c, d_cycles);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_cycles, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost);
    printf("  %-30s : %4lld cycles per instruction\n", name, h_cycles);
}

// 1. 16x16x16 | FP16 Accumulate
__global__ void bench_16x16x16_fp16(half* a, half* b, void* c, long long* cycles) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);
    
    // Warmup
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    long long start = clock64();
    #pragma unroll
    for(int i=0; i<100; i++) {
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    long long end = clock64();
    
    if(threadIdx.x == 0) *cycles = (end - start) / 100;
    wmma::store_matrix_sync((half*)c, c_frag, 16, wmma::mem_row_major);
}

// 2. 16x16x16 | FP32 Accumulate
__global__ void bench_16x16x16_fp32(half* a, half* b, void* c, long long* cycles) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);
    
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    long long start = clock64();
    #pragma unroll
    for(int i=0; i<100; i++) {
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    long long end = clock64();
    
    if(threadIdx.x == 0) *cycles = (end - start) / 100;
    wmma::store_matrix_sync((float*)c, c_frag, 16, wmma::mem_row_major);
}

// 3. 32x8x16 | FP32 Accumulate
__global__ void bench_32x8x16_fp32(half* a, half* b, void* c, long long* cycles) {
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 32, 8, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 8);
    
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    long long start = clock64();
    #pragma unroll
    for(int i=0; i<100; i++) {
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    long long end = clock64();
    
    if(threadIdx.x == 0) *cycles = (end - start) / 100;
    wmma::store_matrix_sync((float*)c, c_frag, 8, wmma::mem_row_major);
}

// 4. 8x32x16 | FP32 Accumulate
__global__ void bench_8x32x16_fp32(half* a, half* b, void* c, long long* cycles) {
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 32, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 32);
    
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    long long start = clock64();
    #pragma unroll
    for(int i=0; i<100; i++) {
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    long long end = clock64();
    
    if(threadIdx.x == 0) *cycles = (end - start) / 100;
    wmma::store_matrix_sync((float*)c, c_frag, 32, wmma::mem_row_major);
}

int main() {
    printf("==================================================\n");
    printf("   V100 (Volta sm_70) Tensor Core Deep Dive       \n");
    printf("==================================================\n\n");

    half *d_a, *d_b;
    void *d_c;
    long long *d_cycles;
    
    cudaMalloc(&d_a, 1024 * sizeof(half));
    cudaMalloc(&d_b, 1024 * sizeof(half));
    cudaMalloc(&d_c, 1024 * sizeof(float)); // Max size needed
    cudaMalloc(&d_cycles, sizeof(long long));

    printf("--- 1. INSTRUCTION LATENCY (Unrolled loop, pure math) ---\n");
    run_bench("16x16x16 (FP16 Acc)", bench_16x16x16_fp16, d_a, d_b, d_c, d_cycles);
    run_bench("16x16x16 (FP32 Acc)", bench_16x16x16_fp32, d_a, d_b, d_c, d_cycles);
    run_bench("32x8x16  (FP32 Acc)", bench_32x8x16_fp32, d_a, d_b, d_c, d_cycles);
    run_bench("8x32x16  (FP32 Acc)", bench_8x32x16_fp32, d_a, d_b, d_c, d_cycles);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_cycles);
    return 0;
}
