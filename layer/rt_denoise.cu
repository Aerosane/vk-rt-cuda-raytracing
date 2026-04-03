// VK_RT TensorRT Denoiser — neural denoising for software ray tracing output
// Runs after RT compute dispatch, before the app reads results.
//
// Pipeline: Noisy RT output (VkImage) → fd-export → CUDA → TRT inference → writeback
// Uses the existing vit_ray_reconstruct model (10-channel FP16 input, 3-channel output)
// Channels: RGB(noisy) + depth + normals(xyz) + albedo(rgb) = 10

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <NvInfer.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <vector>

using namespace nvinfer1;

// ═══════════════════════════════════════════
// TRT Logger
// ═══════════════════════════════════════════
class RTDenoiseLogger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[VK_RT:Denoise] %s\n", msg);
    }
};
static RTDenoiseLogger s_denoiseLogger;

// ═══════════════════════════════════════════
// Denoiser state
// ═══════════════════════════════════════════
struct RTDenoiser {
    IRuntime*          runtime  = nullptr;
    ICudaEngine*       engine   = nullptr;
    IExecutionContext*  context  = nullptr;
    void*              d_input  = nullptr;  // FP16 [1,10,H,W]
    void*              d_output = nullptr;  // FP16 [1,3,H,W]
    void*              d_rgba_in  = nullptr;  // RGBA8 scratch (input)
    void*              d_rgba_out = nullptr;  // RGBA8 scratch (output)
    uint32_t           width = 0, height = 0;
    bool               ready = false;
    bool               hasTRT = false;
    cudaStream_t       stream = nullptr;
    uint64_t           framesDenoised = 0;
    float              lastLatencyMs = 0.0f;
};

static RTDenoiser g_rtDenoise = {};

// ═══════════════════════════════════════════
// CUDA Kernels
// ═══════════════════════════════════════════

// Convert RGBA float (R32G32B32A32_SFLOAT storage image data) → FP16 10-ch NCHW
// Q2RTX writes HDR float4 to storage images
__global__ void float4_to_fp16_10ch(
    const float* __restrict__ src,   // float4 [H][W] (RGBA HDR)
    half* __restrict__ dst,          // FP16 [1][10][H][W]
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixIdx = y * width + x;
    int srcIdx = pixIdx * 4;  // RGBA float stride
    int planeSize = width * height;

    // Channels 0-2: RGB (tone-mapped to [0,1] via simple Reinhard)
    float r = src[srcIdx + 0];
    float g = src[srcIdx + 1];
    float b = src[srcIdx + 2];
    // Simple Reinhard tone mapping for HDR→[0,1]
    float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float scale = 1.0f / (1.0f + lum);
    dst[0 * planeSize + pixIdx] = __float2half(r * scale);
    dst[1 * planeSize + pixIdx] = __float2half(g * scale);
    dst[2 * planeSize + pixIdx] = __float2half(b * scale);

    // Channels 3-9: zero for now (depth, normals, albedo from G-buffer)
    for (int c = 3; c < 10; c++)
        dst[c * planeSize + pixIdx] = __float2half(0.0f);
}

// Convert denoised FP16 3-ch NCHW → float4 (write back to storage image)
__global__ void fp16_3ch_to_float4(
    const half* __restrict__ src,    // FP16 [1][3][H][W]
    float* __restrict__ dst,         // float4 [H][W] (RGBA HDR)
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixIdx = y * width + x;
    int planeSize = width * height;
    int dstIdx = pixIdx * 4;

    // Inverse Reinhard: recover HDR from denoised [0,1] values
    float r = __half2float(src[0 * planeSize + pixIdx]);
    float g = __half2float(src[1 * planeSize + pixIdx]);
    float b = __half2float(src[2 * planeSize + pixIdx]);
    // Inverse Reinhard: x / (1 - x), clamped
    float eps = 1e-6f;
    r = fminf(r / fmaxf(1.0f - r, eps), 100.0f);
    g = fminf(g / fmaxf(1.0f - g, eps), 100.0f);
    b = fminf(b / fmaxf(1.0f - b, eps), 100.0f);

    dst[dstIdx + 0] = r;
    dst[dstIdx + 1] = g;
    dst[dstIdx + 2] = b;
    dst[dstIdx + 3] = 1.0f;  // alpha
}

// Fallback: simple spatial denoise for RGBA float4 (no TRT needed)
__global__ void spatial_denoise_float4(
    float* __restrict__ color,  // float4 [H][W], in-place
    int width, int height, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sumR = 0, sumG = 0, sumB = 0;
    int count = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);
            int idx = (ny * width + nx) * 4;
            sumR += color[idx + 0];
            sumG += color[idx + 1];
            sumB += color[idx + 2];
            count++;
        }
    }
    int idx = (y * width + x) * 4;
    color[idx + 0] = sumR / count;
    color[idx + 1] = sumG / count;
    color[idx + 2] = sumB / count;
}

// ═══════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════
extern "C" {

int rt_denoise_init(uint32_t width, uint32_t height, const char* plan_path)
{
    if (g_rtDenoise.ready) return 1;

    g_rtDenoise.width  = width;
    g_rtDenoise.height = height;

    cudaStreamCreate(&g_rtDenoise.stream);

    // Try loading TRT plan
    if (plan_path && plan_path[0]) {
        std::ifstream file(plan_path, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);
            std::vector<char> buf(size);
            if (file.read(buf.data(), size)) {
                g_rtDenoise.runtime = createInferRuntime(s_denoiseLogger);
                if (g_rtDenoise.runtime) {
                    g_rtDenoise.engine = g_rtDenoise.runtime->deserializeCudaEngine(buf.data(), size);
                    if (g_rtDenoise.engine) {
                        g_rtDenoise.context = g_rtDenoise.engine->createExecutionContext();
                        if (g_rtDenoise.context) {
                            // Verify dimensions match the plan's static shape
                            auto inDims = g_rtDenoise.engine->getTensorShape("input");
                            bool dimsMatch = (inDims.nbDims == 4 &&
                                              inDims.d[2] == (int)height &&
                                              inDims.d[3] == (int)width);
                            if (dimsMatch) {
                                size_t inSize  = 1 * 10 * height * width * sizeof(half);
                                size_t outSize = 1 *  3 * height * width * sizeof(half);
                                if (cudaMalloc(&g_rtDenoise.d_input, inSize) == cudaSuccess &&
                                    cudaMalloc(&g_rtDenoise.d_output, outSize) == cudaSuccess) {
                                    g_rtDenoise.context->setInputShape("input",
                                        Dims4{1, 10, (int)height, (int)width});
                                    g_rtDenoise.hasTRT = true;
                                    fprintf(stderr, "[VK_RT:Denoise] TRT engine loaded: %s (%ux%u)\n",
                                            plan_path, width, height);
                                }
                            } else {
                                fprintf(stderr, "[VK_RT:Denoise] TRT plan shape mismatch: "
                                        "need [1,10,%u,%u] but plan has [1,%d,%d,%d]\n",
                                        height, width,
                                        inDims.nbDims >= 2 ? inDims.d[1] : -1,
                                        inDims.nbDims >= 3 ? inDims.d[2] : -1,
                                        inDims.nbDims >= 4 ? inDims.d[3] : -1);
                            }
                        }
                    }
                }
            }
        }
        if (!g_rtDenoise.hasTRT)
            fprintf(stderr, "[VK_RT:Denoise] TRT plan load failed, using spatial fallback\n");
    }

    g_rtDenoise.ready = true;
    fprintf(stderr, "[VK_RT:Denoise] Initialized %ux%u (%s)\n",
            width, height, g_rtDenoise.hasTRT ? "TRT neural" : "spatial fallback");
    return 1;
}

// Denoise a float4 HDR storage image (CUDA device pointer, in-place)
// srcPtr: device pointer to float4 [height][width] (RGBA HDR from RT)
// Returns latency in ms
float rt_denoise_run(void* srcPtr, uint32_t width, uint32_t height)
{
    if (!g_rtDenoise.ready || !srcPtr) return -1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, g_rtDenoise.stream);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    if (g_rtDenoise.hasTRT && g_rtDenoise.context &&
        width == g_rtDenoise.width && height == g_rtDenoise.height) {
        // TRT path: float4 → FP16 10ch → TRT → FP16 3ch → float4
        float4_to_fp16_10ch<<<grid, block, 0, g_rtDenoise.stream>>>(
            (const float*)srcPtr, (half*)g_rtDenoise.d_input, width, height);

        g_rtDenoise.context->setInputTensorAddress("input", g_rtDenoise.d_input);
        g_rtDenoise.context->setOutputTensorAddress("output", g_rtDenoise.d_output);
        g_rtDenoise.context->enqueueV3(g_rtDenoise.stream);

        fp16_3ch_to_float4<<<grid, block, 0, g_rtDenoise.stream>>>(
            (const half*)g_rtDenoise.d_output, (float*)srcPtr, width, height);
    } else {
        // Spatial fallback: 3x3 box blur
        spatial_denoise_float4<<<grid, block, 0, g_rtDenoise.stream>>>(
            (float*)srcPtr, width, height, 1);
    }

    cudaEventRecord(stop, g_rtDenoise.stream);
    cudaStreamSynchronize(g_rtDenoise.stream);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    g_rtDenoise.framesDenoised++;
    g_rtDenoise.lastLatencyMs = ms;
    return ms;
}

int rt_denoise_has_trt() { return g_rtDenoise.hasTRT ? 1 : 0; }
uint64_t rt_denoise_frame_count() { return g_rtDenoise.framesDenoised; }
float rt_denoise_last_latency() { return g_rtDenoise.lastLatencyMs; }

void rt_denoise_destroy()
{
    if (g_rtDenoise.d_input)  cudaFree(g_rtDenoise.d_input);
    if (g_rtDenoise.d_output) cudaFree(g_rtDenoise.d_output);
    if (g_rtDenoise.context)  delete g_rtDenoise.context;
    if (g_rtDenoise.engine)   delete g_rtDenoise.engine;
    if (g_rtDenoise.runtime)  delete g_rtDenoise.runtime;
    if (g_rtDenoise.stream)   cudaStreamDestroy(g_rtDenoise.stream);
    g_rtDenoise = {};
}

} // extern "C"
