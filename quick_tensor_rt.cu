#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cuda_fp16.h>

using namespace nvcuda;

#define ITERATIONS 10000

// 1. STANDARD CUDA CORE APPROACH (SIMT)
// Each thread tests 1 ray against 16 boxes. (1 Warp = 32 rays testing 16 boxes).
__global__ void benchStandardCUDA(const half* __restrict__ rays, const half* __restrict__ boxes, half* out) {
    int tid = threadIdx.x;
    half ray_val = rays[tid]; // Simplified scalar for benchmark
    half sum = 0;
    
    for (int it = 0; it < ITERATIONS; it++) {
        for (int b = 0; b < 16; b++) {
            // Simulated intersection math (mul-add)
            sum = __hadd(sum, __hmul(ray_val, boxes[b])); 
        }
    }
    out[tid] = sum;
}

// 2. TENSOR CORE APPROACH (WMMA)
// A warp treats 16 rays as a matrix and 16 boxes as a matrix.
// It tests 16 rays against 16 boxes in a SINGLE instruction.
__global__ void benchTensorCore(const half* __restrict__ rays, const half* __restrict__ boxes, half* out) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> ray_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> box_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);
    wmma::load_matrix_sync(ray_frag, rays, 16);
    wmma::load_matrix_sync(box_frag, boxes, 16);

    for (int i = 0; i < ITERATIONS; i++) {
        // 16x16x16 intersection matrix math in one hardware cycle
        wmma::mma_sync(acc_frag, ray_frag, box_frag, acc_frag);
    }
    
    wmma::store_matrix_sync(out, acc_frag, 16, wmma::mem_row_major);
}

int main() {
    half *d_rays, *d_boxes, *d_out;
    cudaMalloc(&d_rays, 256 * sizeof(half));
    cudaMalloc(&d_boxes, 256 * sizeof(half));
    cudaMalloc(&d_out, 256 * sizeof(half));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("--- Quick Latency Test: CUDA Cores vs Tensor Cores for RT Batching ---\n\n");

    // Warmup
    benchStandardCUDA<<<1, 32>>>(d_rays, d_boxes, d_out);
    cudaDeviceSynchronize();

    // 1. Standard CUDA
    cudaEventRecord(start);
    benchStandardCUDA<<<1, 32>>>(d_rays, d_boxes, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_cuda;
    cudaEventElapsedTime(&ms_cuda, start, stop);
    
    // 2. Tensor Cores
    cudaEventRecord(start);
    benchTensorCore<<<1, 32>>>(d_rays, d_boxes, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_tensor;
    cudaEventElapsedTime(&ms_tensor, start, stop);

    printf("  Standard CUDA Cores (1 Warp): %.4f ms\n", ms_cuda);
    printf("  Tensor Cores (1 WMMA Warp):   %.4f ms\n\n", ms_tensor);
    
    printf("  Speedup: %.2fx faster to use Tensor Cores.\n", ms_cuda / ms_tensor);

    cudaFree(d_rays); cudaFree(d_boxes); cudaFree(d_out);
    return 0;
}
