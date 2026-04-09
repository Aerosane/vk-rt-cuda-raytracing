/* nrc.h — Neural Radiance Cache API for VK_RT layer integration
 *
 * Provides NRC inference + training on V100 tensor cores (WMMA).
 * Designed to be called from rt_ir_exec.cu at bounce depth > 1.
 *
 * Usage:
 *   NRCState* nrc = nrc_create();           // once at layer init
 *   nrc_inference(nrc, positions, output);   // per-frame inference
 *   nrc_train_step(nrc, positions, target);  // per-frame training
 *   nrc_destroy(nrc);                        // at layer teardown
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ═══════════════════════════════════════════════════════════════
// Network architecture constants
// ═══════════════════════════════════════════════════════════════

#define NRC_HASH_LEVELS         8
#define NRC_FEATURES_PER_LEVEL  4
#define NRC_HASH_TABLE_SIZE     65536
#define NRC_TOTAL_FEATURES      (NRC_HASH_LEVELS * NRC_FEATURES_PER_LEVEL) // 32
#define NRC_MLP_WIDTH           64
#define NRC_OUTPUT_DIM          4     // RGB + confidence
#define NRC_BASE_RES            16
#define NRC_MAX_RES             512
#define NRC_BATCH_SIZE          256

// ═══════════════════════════════════════════════════════════════
// NRC persistent state — lives for the lifetime of the layer
// ═══════════════════════════════════════════════════════════════

struct NRCState {
    // Network weights (device memory)
    half* d_hash;      // [LEVELS][TABLE_SIZE][FEATURES_PER_LEVEL]
    half* d_W1;        // [32][64]
    half* d_b1;        // [64]
    half* d_W2;        // [64][64]
    half* d_b2;        // [64]
    half* d_W3;        // [64][16] (padded from 64×4)
    half* d_b3;        // [4]

    // Transposed weights for backward pass
    half* d_W1t;       // [64][32]
    half* d_W2t;       // [64][64]
    half* d_W3t;       // [16][64]

    // Scratch buffers (reused each frame)
    half*  d_features;  // [maxQueries][32]
    half*  d_act1;      // [maxQueries][64]
    half*  d_act2;      // [maxQueries][64]
    float* d_output;    // [maxQueries][4]

    // Training scratch
    half*  d_dOut;      // [maxTrain][16]
    half*  d_dH2;       // [maxTrain][64]
    half*  d_dH1;       // [maxTrain][64]
    float* d_dW1_fp32;  // [32][64]
    float* d_dW2_fp32;  // [64][64]
    float* d_dW3_fp32;  // [64][16]
    float* d_loss;      // scalar

    int maxQueries;     // max inference batch (e.g. 1920*1080)
    int maxTrainSamples; // max training batch (e.g. 65536)
    int frameCount;      // for training schedule
    bool initialized;

    // Per-frame sample collection (filled by IR executor, consumed by train)
    float* d_trainPositions;  // [maxTrainSamples][3] — world positions
    float* d_trainTargets;    // [maxTrainSamples][4] — ground truth RGBA
    uint32_t numTrainSamples; // current count (reset each frame after training)
};

// ═══════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════

// Create and initialize NRC with Xavier-init weights
NRCState* nrc_create(int maxQueries = 1920 * 1080, int maxTrainSamples = 65536);

// Destroy NRC and free all GPU memory
void nrc_destroy(NRCState* nrc);

// Inference: position (x,y,z per query) → RGBA output
// Runs hash grid encoding + WMMA MLP forward pass
// positions_xyz: interleaved [x0,y0,z0, x1,y1,z1, ...] in device memory
// output: [N][4] float RGBA, device memory
void nrc_inference(NRCState* nrc,
                   const float* d_positions_xyz, // [N][3] device
                   float* d_output_rgba,          // [N][4] device
                   int numQueries,
                   cudaStream_t stream = 0);

// Training step: given positions + ground truth radiance, do one SGD step
// d_positions_xyz: [N][3] device — world positions of training samples
// d_target_rgba:   [N][4] device — ground truth radiance from traced rays
// Returns MSE loss
float nrc_train_step(NRCState* nrc,
                     const float* d_positions_xyz,
                     const float* d_target_rgba,
                     int numSamples,
                     float learningRate = 0.001f,
                     cudaStream_t stream = 0);

// Training step using internally collected samples (d_trainPositions/d_trainTargets)
float nrc_train_step(NRCState* nrc,
                     int numSamples,
                     float learningRate = 0.001f,
                     cudaStream_t stream = 0);

// Reset weights (e.g. on scene change)
void nrc_reset(NRCState* nrc);
