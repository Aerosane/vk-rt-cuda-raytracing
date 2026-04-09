/* nrc.cu — Neural Radiance Cache implementation
 *
 * WMMA tensor core accelerated hash grid + MLP for V100.
 * See nrc.h for API documentation.
 *
 * Build: compiled as part of libVkLayer_CudaRT.so via build.sh
 */

#include "nrc.h"
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>

using namespace nvcuda;

#define NRC_CK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"[NRC] CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));}}while(0)

// ═══════════════════════════════════════════════════════════════
// Device helpers
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__ uint32_t nrc_hash(int x, int y, int z, int level) {
    const uint32_t primes[] = {1u, 2654435761u, 805459861u, 3674653429u,
                                2097192037u, 1227099533u, 3999999979u, 2860486313u};
    uint32_t h = (uint32_t)x * primes[0] ^ (uint32_t)y * primes[1] ^
                 (uint32_t)z * primes[2] ^ primes[level & 7];
    return h % NRC_HASH_TABLE_SIZE;
}

__device__ __forceinline__ float nrc_level_res(int level) {
    float b = expf(logf((float)NRC_MAX_RES / NRC_BASE_RES) / (NRC_HASH_LEVELS - 1));
    return NRC_BASE_RES * powf(b, (float)level);
}

// ═══════════════════════════════════════════════════════════════
// Hash grid encoder — interleaved xyz input
// ═══════════════════════════════════════════════════════════════

__global__ void nrc_encode_xyz(
    const float* __restrict__ pos,     // [N][3] interleaved
    const half*  __restrict__ hashTable,
    half*        __restrict__ features,
    int numQueries)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= numQueries) return;

    float px = pos[qi*3+0], py = pos[qi*3+1], pz = pos[qi*3+2];

    #pragma unroll
    for (int level = 0; level < NRC_HASH_LEVELS; level++) {
        float res = nrc_level_res(level);
        float fx = px * res, fy = py * res, fz = pz * res;
        int ix = (int)floorf(fx), iy = (int)floorf(fy), iz = (int)floorf(fz);
        float wx = fx - ix, wy = fy - iy, wz = fz - iz;

        half2 acc01 = make_half2(__float2half(0.f), __float2half(0.f));
        half2 acc23 = make_half2(__float2half(0.f), __float2half(0.f));

        #pragma unroll
        for (int corner = 0; corner < 8; corner++) {
            int dx = corner & 1, dy = (corner>>1)&1, dz = (corner>>2)&1;
            float w = (dx ? wx : 1-wx) * (dy ? wy : 1-wy) * (dz ? wz : 1-wz);
            half2 hw = __float2half2_rn(w);
            uint32_t h = nrc_hash(ix+dx, iy+dy, iz+dz, level);
            int base = (level * NRC_HASH_TABLE_SIZE + h) * NRC_FEATURES_PER_LEVEL;
            const half2* p = (const half2*)(hashTable + base);
            acc01 = __hfma2(hw, p[0], acc01);
            acc23 = __hfma2(hw, p[1], acc23);
        }

        int outBase = qi * NRC_TOTAL_FEATURES + level * NRC_FEATURES_PER_LEVEL;
        ((half2*)(features + outBase))[0] = acc01;
        ((half2*)(features + outBase))[1] = acc23;
    }
}

// ═══════════════════════════════════════════════════════════════
// WMMA MLP Forward — 32→64(ReLU)→64(ReLU)→4(sigmoid)
// ═══════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(256, 2) nrc_mlp_fwd(
    const half* __restrict__ input,
    const half* __restrict__ W1, const half* __restrict__ b1,
    const half* __restrict__ W2, const half* __restrict__ b2,
    const half* __restrict__ W3, const half* __restrict__ b3,
    half*       __restrict__ act1,
    half*       __restrict__ act2,
    float*      __restrict__ output,
    int batchSize)
{
    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int numWarps = blockDim.x / 32;

    int batchOffset = blockIdx.x * NRC_BATCH_SIZE;
    if (batchOffset >= batchSize) return;
    int localBatch = min(NRC_BATCH_SIZE, batchSize - batchOffset);

    __shared__ float tileOut[8][16 * 16];
    int tilesM = (localBatch + 15) / 16;

    // Layer 1: input[B×32] × W1[32×64] + bias → ReLU → act1
    {
        int tN_L = NRC_MLP_WIDTH / 16, tK_L = NRC_TOTAL_FEATURES / 16;
        int totalTiles = tilesM * tN_L;
        for (int tile = warpId; tile < totalTiles; tile += numWarps) {
            int tM = tile / tN_L, tN = tile % tN_L;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int tK = 0; tK < tK_L; tK++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> A;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B;
                wmma::load_matrix_sync(A, input + (batchOffset+tM*16)*NRC_TOTAL_FEATURES + tK*16, NRC_TOTAL_FEATURES);
                wmma::load_matrix_sync(B, W1 + tK*16*NRC_MLP_WIDTH + tN*16, NRC_MLP_WIDTH);
                wmma::mma_sync(acc, A, B, acc);
            }
            wmma::store_matrix_sync(tileOut[warpId], acc, 16, wmma::mem_row_major);
            int outR = tM*16, outC = tN*16;
            for (int i = lane; i < 256; i += 32) {
                int r = i/16, c = i%16;
                if (outR+r < localBatch) {
                    float v = tileOut[warpId][i] + __half2float(b1[outC+c]);
                    act1[(batchOffset+outR+r)*NRC_MLP_WIDTH + outC+c] = __float2half(fmaxf(v, 0.0f));
                }
            }
        }
    }
    __syncthreads();

    // Layer 2: act1[B×64] × W2[64×64] + bias → ReLU → act2
    {
        int tN_L = NRC_MLP_WIDTH / 16, tK_L = NRC_MLP_WIDTH / 16;
        int totalTiles = tilesM * tN_L;
        for (int tile = warpId; tile < totalTiles; tile += numWarps) {
            int tM = tile / tN_L, tN = tile % tN_L;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int tK = 0; tK < tK_L; tK++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> A;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B;
                wmma::load_matrix_sync(A, act1 + (batchOffset+tM*16)*NRC_MLP_WIDTH + tK*16, NRC_MLP_WIDTH);
                wmma::load_matrix_sync(B, W2 + tK*16*NRC_MLP_WIDTH + tN*16, NRC_MLP_WIDTH);
                wmma::mma_sync(acc, A, B, acc);
            }
            wmma::store_matrix_sync(tileOut[warpId], acc, 16, wmma::mem_row_major);
            int outR = tM*16, outC = tN*16;
            for (int i = lane; i < 256; i += 32) {
                int r = i/16, c = i%16;
                if (outR+r < localBatch) {
                    float v = tileOut[warpId][i] + __half2float(b2[outC+c]);
                    act2[(batchOffset+outR+r)*NRC_MLP_WIDTH + outC+c] = __float2half(fmaxf(v, 0.0f));
                }
            }
        }
    }
    __syncthreads();

    // Layer 3: act2[B×64] × W3[64×16pad] + bias → sigmoid → output
    {
        int tK_L = NRC_MLP_WIDTH / 16;
        for (int tM = warpId; tM < tilesM; tM += numWarps) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int tK = 0; tK < tK_L; tK++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> A;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B;
                wmma::load_matrix_sync(A, act2 + (batchOffset+tM*16)*NRC_MLP_WIDTH + tK*16, NRC_MLP_WIDTH);
                wmma::load_matrix_sync(B, W3 + tK*16*16, 16);
                wmma::mma_sync(acc, A, B, acc);
            }
            wmma::store_matrix_sync(tileOut[warpId], acc, 16, wmma::mem_row_major);
            int outR = tM * 16;
            for (int i = lane; i < 16*NRC_OUTPUT_DIM; i += 32) {
                int r = i/NRC_OUTPUT_DIM, c = i%NRC_OUTPUT_DIM;
                if (outR+r < localBatch) {
                    float v = tileOut[warpId][r*16+c] + __half2float(b3[c]);
                    v = 1.0f / (1.0f + expf(-v));
                    output[(batchOffset+outR+r)*NRC_OUTPUT_DIM + c] = v;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Training kernels
// ═══════════════════════════════════════════════════════════════

__global__ void nrc_loss_grad(
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    half*        __restrict__ dOut,
    float*       __restrict__ loss,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float sumSq = 0.f;
    for (int c = 0; c < NRC_OUTPUT_DIM; c++) {
        float p = predicted[i*NRC_OUTPUT_DIM + c];
        float t = target[i*NRC_OUTPUT_DIM + c];
        float diff = p - t;
        sumSq += diff * diff;
        float grad = 2.0f * diff * p * (1.0f - p);
        dOut[i*16 + c] = __float2half(grad);
    }
    for (int c = NRC_OUTPUT_DIM; c < 16; c++)
        dOut[i*16 + c] = __float2half(0.f);
    atomicAdd(loss, sumSq / N);
}

__global__ void __launch_bounds__(256, 2) nrc_backward(
    const half* __restrict__ input,
    const half* __restrict__ dOutput,
    const half* __restrict__ Wt,
    half*       __restrict__ dInput,
    int batchSize, int inDim, int outDim, bool applyRelu)
{
    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int numWarps = blockDim.x / 32;

    int batchOffset = blockIdx.x * NRC_BATCH_SIZE;
    if (batchOffset >= batchSize) return;
    int localBatch = min(NRC_BATCH_SIZE, batchSize - batchOffset);

    __shared__ float tileOut[8][16 * 16];
    int tilesM = (localBatch + 15) / 16;
    int tilesN = inDim / 16;
    int tilesK = outDim / 16;
    int totalTiles = tilesM * tilesN;

    for (int tile = warpId; tile < totalTiles; tile += numWarps) {
        int tM = tile / tilesN, tN = tile % tilesN;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int tK = 0; tK < tilesK; tK++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> A;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B;
            wmma::load_matrix_sync(A, dOutput + (batchOffset+tM*16)*outDim + tK*16, outDim);
            wmma::load_matrix_sync(B, Wt + tK*16*inDim + tN*16, inDim);
            wmma::mma_sync(acc, A, B, acc);
        }
        wmma::store_matrix_sync(tileOut[warpId], acc, 16, wmma::mem_row_major);
        int outR = tM*16, outC = tN*16;
        for (int i = lane; i < 256; i += 32) {
            int r = i/16, c = i%16;
            if (outR+r < localBatch) {
                float v = tileOut[warpId][i];
                if (applyRelu) {
                    float act = __half2float(input[(batchOffset+outR+r)*inDim + outC+c]);
                    if (act <= 0.f) v = 0.f;
                }
                dInput[(batchOffset+outR+r)*inDim + outC+c] = __float2half(v);
            }
        }
    }
}

__global__ void __launch_bounds__(256, 2) nrc_weight_grad(
    const half* __restrict__ input,
    const half* __restrict__ dOutput,
    float*      __restrict__ dW_fp32,
    int batchSize, int inDim, int outDim)
{
    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int numWarps = blockDim.x / 32;

    __shared__ float tileOut[8][16*16];
    int tilesM = inDim / 16;
    int tilesN = outDim / 16;
    int tilesK = (batchSize + 15) / 16;
    int totalTiles = tilesM * tilesN;

    for (int tile = warpId; tile < totalTiles; tile += numWarps) {
        int tM = tile / tilesN, tN = tile % tilesN;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int tK = 0; tK < tilesK; tK++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B;
            wmma::load_matrix_sync(A, input + tK*16*inDim + tM*16, inDim);
            wmma::load_matrix_sync(B, dOutput + tK*16*outDim + tN*16, outDim);
            wmma::mma_sync(acc, A, B, acc);
        }
        wmma::store_matrix_sync(tileOut[warpId], acc, 16, wmma::mem_row_major);
        for (int i = lane; i < 256; i += 32) {
            int r = i/16, c = i%16;
            if (tM*16+r < inDim && tN*16+c < outDim)
                atomicAdd(&dW_fp32[(tM*16+r)*outDim + tN*16+c], tileOut[warpId][i]);
        }
    }
}

__global__ void nrc_sgd(half* __restrict__ W, const float* __restrict__ dW,
                         float lr, int N, int batchSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float w = __half2float(W[i]);
    w -= lr * dW[i] / (float)batchSize;
    W[i] = __float2half(w);
}

__global__ void nrc_transpose_k(const half* src, half* dst, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i % cols;
    dst[c * rows + r] = src[r * cols + c];
}

// ═══════════════════════════════════════════════════════════════
// Host-side weight initialization
// ═══════════════════════════════════════════════════════════════

static void init_xavier(half* W, int rows, int cols) {
    float scale = sqrtf(6.0f / (rows + cols));
    for (int i = 0; i < rows * cols; i++) {
        float r = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
        W[i] = __float2half(r);
    }
}

// ═══════════════════════════════════════════════════════════════
// Public API implementation
// ═══════════════════════════════════════════════════════════════

NRCState* nrc_create(int maxQueries, int maxTrainSamples) {
    NRCState* nrc = new NRCState();
    memset(nrc, 0, sizeof(NRCState));
    nrc->maxQueries = maxQueries;
    nrc->maxTrainSamples = maxTrainSamples;

    int hashSize = NRC_HASH_LEVELS * NRC_HASH_TABLE_SIZE * NRC_FEATURES_PER_LEVEL;
    int W1_n = NRC_TOTAL_FEATURES * NRC_MLP_WIDTH;
    int W2_n = NRC_MLP_WIDTH * NRC_MLP_WIDTH;
    int W3_n = NRC_MLP_WIDTH * 16;

    // Host-side init
    half* h_hash = new half[hashSize];
    half* h_W1 = new half[W1_n];
    half* h_b1 = new half[NRC_MLP_WIDTH];
    half* h_W2 = new half[W2_n];
    half* h_b2 = new half[NRC_MLP_WIDTH];
    half* h_W3 = new half[W3_n];
    half* h_b3 = new half[NRC_OUTPUT_DIM];

    srand(42);
    for (int i = 0; i < hashSize; i++)
        h_hash[i] = __float2half(((float)rand()/RAND_MAX - 0.5f) * 0.1f);
    init_xavier(h_W1, NRC_TOTAL_FEATURES, NRC_MLP_WIDTH);
    init_xavier(h_W2, NRC_MLP_WIDTH, NRC_MLP_WIDTH);
    memset(h_W3, 0, W3_n * sizeof(half));
    for (int r = 0; r < NRC_MLP_WIDTH; r++)
        for (int c = 0; c < NRC_OUTPUT_DIM; c++) {
            float s = sqrtf(6.0f / (NRC_MLP_WIDTH + NRC_OUTPUT_DIM));
            h_W3[r*16+c] = __float2half(((float)rand()/RAND_MAX - 0.5f)*2.0f*s);
        }
    for (int i = 0; i < NRC_MLP_WIDTH; i++) h_b1[i] = __float2half(0.01f);
    for (int i = 0; i < NRC_MLP_WIDTH; i++) h_b2[i] = __float2half(0.01f);
    for (int i = 0; i < NRC_OUTPUT_DIM; i++) h_b3[i] = __float2half(0.0f);

    // Allocate + upload weights
    NRC_CK(cudaMalloc(&nrc->d_hash, hashSize * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_W1, W1_n * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_b1, NRC_MLP_WIDTH * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_W2, W2_n * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_b2, NRC_MLP_WIDTH * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_W3, W3_n * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_b3, NRC_OUTPUT_DIM * sizeof(half)));

    NRC_CK(cudaMemcpy(nrc->d_hash, h_hash, hashSize*sizeof(half), cudaMemcpyHostToDevice));
    NRC_CK(cudaMemcpy(nrc->d_W1, h_W1, W1_n*sizeof(half), cudaMemcpyHostToDevice));
    NRC_CK(cudaMemcpy(nrc->d_b1, h_b1, NRC_MLP_WIDTH*sizeof(half), cudaMemcpyHostToDevice));
    NRC_CK(cudaMemcpy(nrc->d_W2, h_W2, W2_n*sizeof(half), cudaMemcpyHostToDevice));
    NRC_CK(cudaMemcpy(nrc->d_b2, h_b2, NRC_MLP_WIDTH*sizeof(half), cudaMemcpyHostToDevice));
    NRC_CK(cudaMemcpy(nrc->d_W3, h_W3, W3_n*sizeof(half), cudaMemcpyHostToDevice));
    NRC_CK(cudaMemcpy(nrc->d_b3, h_b3, NRC_OUTPUT_DIM*sizeof(half), cudaMemcpyHostToDevice));

    // Transposed weights
    NRC_CK(cudaMalloc(&nrc->d_W1t, W1_n * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_W2t, W2_n * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_W3t, W3_n * sizeof(half)));
    nrc_transpose_k<<<(W1_n+255)/256, 256>>>(nrc->d_W1, nrc->d_W1t, NRC_TOTAL_FEATURES, NRC_MLP_WIDTH);
    nrc_transpose_k<<<(W2_n+255)/256, 256>>>(nrc->d_W2, nrc->d_W2t, NRC_MLP_WIDTH, NRC_MLP_WIDTH);
    nrc_transpose_k<<<(W3_n+255)/256, 256>>>(nrc->d_W3, nrc->d_W3t, NRC_MLP_WIDTH, 16);

    // Scratch buffers
    NRC_CK(cudaMalloc(&nrc->d_features, maxQueries * NRC_TOTAL_FEATURES * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_act1, maxQueries * NRC_MLP_WIDTH * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_act2, maxQueries * NRC_MLP_WIDTH * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_output, maxQueries * NRC_OUTPUT_DIM * sizeof(float)));

    // Training buffers
    NRC_CK(cudaMalloc(&nrc->d_dOut, maxTrainSamples * 16 * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_dH2, maxTrainSamples * NRC_MLP_WIDTH * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_dH1, maxTrainSamples * NRC_MLP_WIDTH * sizeof(half)));
    NRC_CK(cudaMalloc(&nrc->d_dW1_fp32, W1_n * sizeof(float)));
    NRC_CK(cudaMalloc(&nrc->d_dW2_fp32, W2_n * sizeof(float)));
    NRC_CK(cudaMalloc(&nrc->d_dW3_fp32, W3_n * sizeof(float)));
    NRC_CK(cudaMalloc(&nrc->d_loss, sizeof(float)));

    NRC_CK(cudaDeviceSynchronize());

    delete[] h_hash; delete[] h_W1; delete[] h_b1;
    delete[] h_W2; delete[] h_b2; delete[] h_W3; delete[] h_b3;

    nrc->initialized = true;
    nrc->frameCount = 0;

    fprintf(stderr, "[NRC] Created: hash=%.1fMB, MLP=%d→%d→%d→%d (%.0f params), "
            "maxQ=%dK, maxTrain=%dK\n",
            hashSize*sizeof(half)/1048576.f,
            NRC_TOTAL_FEATURES, NRC_MLP_WIDTH, NRC_MLP_WIDTH, NRC_OUTPUT_DIM,
            (float)(W1_n + NRC_MLP_WIDTH + W2_n + NRC_MLP_WIDTH + NRC_MLP_WIDTH*NRC_OUTPUT_DIM + NRC_OUTPUT_DIM),
            maxQueries/1024, maxTrainSamples/1024);

    return nrc;
}

void nrc_destroy(NRCState* nrc) {
    if (!nrc) return;
    cudaFree(nrc->d_hash); cudaFree(nrc->d_W1); cudaFree(nrc->d_b1);
    cudaFree(nrc->d_W2); cudaFree(nrc->d_b2); cudaFree(nrc->d_W3); cudaFree(nrc->d_b3);
    cudaFree(nrc->d_W1t); cudaFree(nrc->d_W2t); cudaFree(nrc->d_W3t);
    cudaFree(nrc->d_features); cudaFree(nrc->d_act1); cudaFree(nrc->d_act2); cudaFree(nrc->d_output);
    cudaFree(nrc->d_dOut); cudaFree(nrc->d_dH2); cudaFree(nrc->d_dH1);
    cudaFree(nrc->d_dW1_fp32); cudaFree(nrc->d_dW2_fp32); cudaFree(nrc->d_dW3_fp32);
    cudaFree(nrc->d_loss);
    delete nrc;
}

void nrc_inference(NRCState* nrc,
                   const float* d_positions_xyz,
                   float* d_output_rgba,
                   int numQueries,
                   cudaStream_t stream)
{
    if (!nrc || !nrc->initialized || numQueries <= 0) return;
    int N = min(numQueries, nrc->maxQueries);

    int encGrid = (N + 255) / 256;
    nrc_encode_xyz<<<encGrid, 256, 0, stream>>>(
        d_positions_xyz, nrc->d_hash, nrc->d_features, N);

    int mlpBatches = (N + NRC_BATCH_SIZE - 1) / NRC_BATCH_SIZE;
    nrc_mlp_fwd<<<mlpBatches, 256, 0, stream>>>(
        nrc->d_features, nrc->d_W1, nrc->d_b1, nrc->d_W2, nrc->d_b2,
        nrc->d_W3, nrc->d_b3, nrc->d_act1, nrc->d_act2,
        d_output_rgba, N);
}

float nrc_train_step(NRCState* nrc,
                     const float* d_positions_xyz,
                     const float* d_target_rgba,
                     int numSamples,
                     float learningRate,
                     cudaStream_t stream)
{
    if (!nrc || !nrc->initialized || numSamples <= 0) return -1.f;
    int N = min(numSamples, nrc->maxTrainSamples);
    int batches = (N + NRC_BATCH_SIZE - 1) / NRC_BATCH_SIZE;
    int W1_n = NRC_TOTAL_FEATURES * NRC_MLP_WIDTH;
    int W2_n = NRC_MLP_WIDTH * NRC_MLP_WIDTH;
    int W3_n = NRC_MLP_WIDTH * 16;

    // Forward pass (encodes + MLP, caches activations in act1/act2)
    nrc_inference(nrc, d_positions_xyz, nrc->d_output, N, stream);

    // Compute loss gradient
    NRC_CK(cudaMemsetAsync(nrc->d_loss, 0, sizeof(float), stream));
    nrc_loss_grad<<<(N+255)/256, 256, 0, stream>>>(
        nrc->d_output, d_target_rgba, nrc->d_dOut, nrc->d_loss, N);

    // Zero gradient accumulators
    NRC_CK(cudaMemsetAsync(nrc->d_dW1_fp32, 0, W1_n*sizeof(float), stream));
    NRC_CK(cudaMemsetAsync(nrc->d_dW2_fp32, 0, W2_n*sizeof(float), stream));
    NRC_CK(cudaMemsetAsync(nrc->d_dW3_fp32, 0, W3_n*sizeof(float), stream));

    // Backward L3: dH2 = dOut × W3^T
    nrc_backward<<<batches, 256, 0, stream>>>(
        nrc->d_act2, nrc->d_dOut, nrc->d_W3t, nrc->d_dH2,
        N, NRC_MLP_WIDTH, 16, true);

    // Backward L2: dH1 = dH2 × W2^T
    nrc_backward<<<batches, 256, 0, stream>>>(
        nrc->d_act1, nrc->d_dH2, nrc->d_W2t, nrc->d_dH1,
        N, NRC_MLP_WIDTH, NRC_MLP_WIDTH, true);

    // Weight gradient accumulation
    nrc_weight_grad<<<1, 256, 0, stream>>>(nrc->d_act2, nrc->d_dOut, nrc->d_dW3_fp32, N, NRC_MLP_WIDTH, 16);
    nrc_weight_grad<<<1, 256, 0, stream>>>(nrc->d_act1, nrc->d_dH2, nrc->d_dW2_fp32, N, NRC_MLP_WIDTH, NRC_MLP_WIDTH);
    nrc_weight_grad<<<1, 256, 0, stream>>>(nrc->d_features, nrc->d_dH1, nrc->d_dW1_fp32, N, NRC_TOTAL_FEATURES, NRC_MLP_WIDTH);

    // SGD update
    nrc_sgd<<<(W1_n+255)/256, 256, 0, stream>>>(nrc->d_W1, nrc->d_dW1_fp32, learningRate, W1_n, N);
    nrc_sgd<<<(W2_n+255)/256, 256, 0, stream>>>(nrc->d_W2, nrc->d_dW2_fp32, learningRate, W2_n, N);
    nrc_sgd<<<(W3_n+255)/256, 256, 0, stream>>>(nrc->d_W3, nrc->d_dW3_fp32, learningRate, W3_n, N);

    // Update transposed weights
    nrc_transpose_k<<<(W1_n+255)/256, 256, 0, stream>>>(nrc->d_W1, nrc->d_W1t, NRC_TOTAL_FEATURES, NRC_MLP_WIDTH);
    nrc_transpose_k<<<(W2_n+255)/256, 256, 0, stream>>>(nrc->d_W2, nrc->d_W2t, NRC_MLP_WIDTH, NRC_MLP_WIDTH);
    nrc_transpose_k<<<(W3_n+255)/256, 256, 0, stream>>>(nrc->d_W3, nrc->d_W3t, NRC_MLP_WIDTH, 16);

    nrc->frameCount++;

    // Read loss (sync)
    float h_loss = 0;
    NRC_CK(cudaMemcpyAsync(&h_loss, nrc->d_loss, sizeof(float), cudaMemcpyDeviceToHost, stream));
    NRC_CK(cudaStreamSynchronize(stream));
    return h_loss;
}

void nrc_reset(NRCState* nrc) {
    if (!nrc) return;
    // Re-initialize with fresh random weights
    int hashSize = NRC_HASH_LEVELS * NRC_HASH_TABLE_SIZE * NRC_FEATURES_PER_LEVEL;
    int W1_n = NRC_TOTAL_FEATURES * NRC_MLP_WIDTH;
    int W2_n = NRC_MLP_WIDTH * NRC_MLP_WIDTH;
    int W3_n = NRC_MLP_WIDTH * 16;

    int maxN = hashSize;
    if (W1_n > maxN) maxN = W1_n;
    if (W2_n > maxN) maxN = W2_n;
    if (W3_n > maxN) maxN = W3_n;
    half* h_tmp = new half[maxN];

    srand(42);
    for (int i = 0; i < hashSize; i++)
        h_tmp[i] = __float2half(((float)rand()/RAND_MAX - 0.5f) * 0.1f);
    NRC_CK(cudaMemcpy(nrc->d_hash, h_tmp, hashSize*sizeof(half), cudaMemcpyHostToDevice));

    init_xavier(h_tmp, NRC_TOTAL_FEATURES, NRC_MLP_WIDTH);
    NRC_CK(cudaMemcpy(nrc->d_W1, h_tmp, W1_n*sizeof(half), cudaMemcpyHostToDevice));

    init_xavier(h_tmp, NRC_MLP_WIDTH, NRC_MLP_WIDTH);
    NRC_CK(cudaMemcpy(nrc->d_W2, h_tmp, W2_n*sizeof(half), cudaMemcpyHostToDevice));

    memset(h_tmp, 0, W3_n * sizeof(half));
    float s = sqrtf(6.0f / (NRC_MLP_WIDTH + NRC_OUTPUT_DIM));
    for (int r = 0; r < NRC_MLP_WIDTH; r++)
        for (int c = 0; c < NRC_OUTPUT_DIM; c++)
            h_tmp[r*16+c] = __float2half(((float)rand()/RAND_MAX - 0.5f)*2.0f*s);
    NRC_CK(cudaMemcpy(nrc->d_W3, h_tmp, W3_n*sizeof(half), cudaMemcpyHostToDevice));

    nrc_transpose_k<<<(W1_n+255)/256, 256>>>(nrc->d_W1, nrc->d_W1t, NRC_TOTAL_FEATURES, NRC_MLP_WIDTH);
    nrc_transpose_k<<<(W2_n+255)/256, 256>>>(nrc->d_W2, nrc->d_W2t, NRC_MLP_WIDTH, NRC_MLP_WIDTH);
    nrc_transpose_k<<<(W3_n+255)/256, 256>>>(nrc->d_W3, nrc->d_W3t, NRC_MLP_WIDTH, 16);

    NRC_CK(cudaDeviceSynchronize());
    delete[] h_tmp;
    nrc->frameCount = 0;
    fprintf(stderr, "[NRC] Weights reset\n");
}
