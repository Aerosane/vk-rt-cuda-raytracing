// RasterBoost Upscale Engine — TensorRT-based neural upscaling for VkLayer_CudaRT
// Loads a serialized TRT plan and runs inference on CUDA-imported swapchain images.
//
// Input:  Low-res rendered frame (e.g. 540p RGBA8)
// Output: Full-res upscaled frame (e.g. 1080p RGBA8)
// Pipeline: RGBA8→FP16 normalize → TRT inference → FP16→RGBA8 denormalize

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <NvInfer.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <mutex>

using namespace nvinfer1;

// ═══════════════════════════════════════════
// TensorRT Logger (stderr, minimal)
// ═══════════════════════════════════════════
class TRTLogger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[RasterBoost:TRT] %s\n", msg);
    }
};
static TRTLogger s_trtLogger;

// ═══════════════════════════════════════════
// Upscale engine state
// ═══════════════════════════════════════════
struct UpscaleEngine {
    IRuntime*          runtime  = nullptr;
    ICudaEngine*       engine   = nullptr;
    IExecutionContext*  context  = nullptr;
    void*              d_input  = nullptr;  // FP16 input  [1, 3, renderH, renderW]
    void*              d_output = nullptr;  // FP16 output [1, 3, outputH, outputW]
    uint32_t           renderW = 0, renderH = 0;
    uint32_t           outputW = 0, outputH = 0;
    bool               ready   = false;
    cudaStream_t       stream  = nullptr;
};

static UpscaleEngine g_upscale;
static std::mutex    g_upscaleLock;

// ═══════════════════════════════════════════
// CUDA kernels: format conversion
// ═══════════════════════════════════════════

// RGBA8 → FP16 NCHW (10 channels: RGB + 7 zero-padded for TRT plan compatibility)
__global__ void rgba8_to_fp16_nchw(
    const unsigned char* __restrict__ src,  // RGBA8 [H][W][4]
    half* __restrict__ dst,                 // FP16  [1][10][H][W]
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixIdx = y * width + x;
    int srcIdx = pixIdx * 4;  // RGBA8 stride

    float r = src[srcIdx + 0] * (1.0f / 255.0f);
    float g = src[srcIdx + 1] * (1.0f / 255.0f);
    float b = src[srcIdx + 2] * (1.0f / 255.0f);

    int planeSize = width * height;
    dst[0 * planeSize + pixIdx] = __float2half(r);  // R plane
    dst[1 * planeSize + pixIdx] = __float2half(g);  // G plane
    dst[2 * planeSize + pixIdx] = __float2half(b);  // B plane
    // Channels 3-9: zero (depth, normals, motion, albedo — to be filled by G-buffer)
    for (int c = 3; c < 10; c++)
        dst[c * planeSize + pixIdx] = __float2half(0.0f);
}

// FP16 NCHW → RGBA8 (denormalize, clamp, alpha=255)
__global__ void fp16_nchw_to_rgba8(
    const half* __restrict__ src,           // FP16  [1][3][H][W]
    unsigned char* __restrict__ dst,        // RGBA8 [H][W][4]
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixIdx = y * width + x;
    int planeSize = width * height;

    float r = __half2float(src[0 * planeSize + pixIdx]);
    float g = __half2float(src[1 * planeSize + pixIdx]);
    float b = __half2float(src[2 * planeSize + pixIdx]);

    // Clamp to [0, 255]
    int dstIdx = pixIdx * 4;
    dst[dstIdx + 0] = (unsigned char)fminf(fmaxf(r * 255.0f, 0.0f), 255.0f);
    dst[dstIdx + 1] = (unsigned char)fminf(fmaxf(g * 255.0f, 0.0f), 255.0f);
    dst[dstIdx + 2] = (unsigned char)fminf(fmaxf(b * 255.0f, 0.0f), 255.0f);
    dst[dstIdx + 3] = 255;
}

// Simple bilinear upscale fallback (RGBA8 → RGBA8, no TensorRT needed)
__global__ void bilinear_upscale_rgba8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int srcW, int srcH, int dstW, int dstH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    float sx = (float)x * srcW / dstW;
    float sy = (float)y * srcH / dstH;
    int x0 = (int)sx, y0 = (int)sy;
    int x1 = min(x0 + 1, srcW - 1), y1 = min(y0 + 1, srcH - 1);
    float fx = sx - x0, fy = sy - y0;

    for (int c = 0; c < 4; c++) {
        float v00 = src[(y0 * srcW + x0) * 4 + c];
        float v10 = src[(y0 * srcW + x1) * 4 + c];
        float v01 = src[(y1 * srcW + x0) * 4 + c];
        float v11 = src[(y1 * srcW + x1) * 4 + c];
        float val = v00 * (1-fx) * (1-fy) + v10 * fx * (1-fy)
                  + v01 * (1-fx) * fy + v11 * fx * fy;
        dst[(y * dstW + x) * 4 + c] = (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);
    }
}

// ═══════════════════════════════════════════
// Public API (called from VkLayer_CudaRT.cpp)
// ═══════════════════════════════════════════

extern "C" {

// Initialize the TRT upscale engine. Call once at device creation.
// plan_path: path to serialized TensorRT engine (.plan)
// Returns 1 on success, 0 on failure (falls back to bilinear)
int rasterboost_upscale_init(
    uint32_t renderW, uint32_t renderH,
    uint32_t outputW, uint32_t outputH,
    const char* plan_path)
{
    std::lock_guard<std::mutex> lock(g_upscaleLock);
    if (g_upscale.ready) return 1;

    fprintf(stderr, "[RasterBoost] Initializing TRT upscale: %ux%u → %ux%u\n",
            renderW, renderH, outputW, outputH);

    g_upscale.renderW = renderW;
    g_upscale.renderH = renderH;
    g_upscale.outputW = outputW;
    g_upscale.outputH = outputH;

    cudaStreamCreate(&g_upscale.stream);

    // Try loading TRT plan
    if (plan_path && plan_path[0]) {
        std::ifstream file(plan_path, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);
            std::vector<char> buf(size);
            if (file.read(buf.data(), size)) {
                g_upscale.runtime = createInferRuntime(s_trtLogger);
                if (g_upscale.runtime) {
                    g_upscale.engine = g_upscale.runtime->deserializeCudaEngine(buf.data(), size);
                    if (g_upscale.engine) {
                        g_upscale.context = g_upscale.engine->createExecutionContext();
                        if (g_upscale.context) {
                            // Allocate FP16 buffers
                            size_t inSize  = 1 * 10 * renderH * renderW * sizeof(half);
                            size_t outSize = 1 * 3 * outputH * outputW * sizeof(half);
                            if (cudaMalloc(&g_upscale.d_input, inSize) == cudaSuccess &&
                                cudaMalloc(&g_upscale.d_output, outSize) == cudaSuccess) {
                                // Set input shape for dynamic profile
                                g_upscale.context->setInputShape("input",
                                    Dims4{1, 10, (int)renderH, (int)renderW});
                                g_upscale.ready = true;
                                fprintf(stderr, "[RasterBoost] TRT engine loaded: %s (%.1f KB)\n",
                                        plan_path, size / 1024.0f);
                                return 1;
                            }
                        }
                    }
                }
            }
        }
        fprintf(stderr, "[RasterBoost] WARNING: TRT plan load failed, using bilinear fallback\n");
    }

    // Fallback: no TRT, just bilinear upscale
    g_upscale.ready = true;  // "ready" but without TRT (bilinear mode)
    fprintf(stderr, "[RasterBoost] Using bilinear upscale fallback (no TRT plan)\n");
    return 1;
}

// Run upscale on a CUDA device pointer to an RGBA8 image
// srcPtr: device pointer to low-res RGBA8 [renderH][renderW][4]
// dstPtr: device pointer to full-res RGBA8 [outputH][outputW][4]
// Returns 0 on success
int rasterboost_upscale_run(void* srcPtr, void* dstPtr)
{
    if (!g_upscale.ready || !srcPtr || !dstPtr) return -1;

    dim3 block(16, 16);

    if (g_upscale.context && g_upscale.d_input && g_upscale.d_output) {
        // TRT path: RGBA8 → FP16 → TRT → FP16 → RGBA8
        dim3 gridIn((g_upscale.renderW + 15) / 16, (g_upscale.renderH + 15) / 16);
        rgba8_to_fp16_nchw<<<gridIn, block, 0, g_upscale.stream>>>(
            (const unsigned char*)srcPtr, (half*)g_upscale.d_input,
            g_upscale.renderW, g_upscale.renderH);

        // TRT inference
        g_upscale.context->setInputTensorAddress("input", g_upscale.d_input);
        g_upscale.context->setOutputTensorAddress("output", g_upscale.d_output);
        g_upscale.context->enqueueV3(g_upscale.stream);

        dim3 gridOut((g_upscale.outputW + 15) / 16, (g_upscale.outputH + 15) / 16);
        fp16_nchw_to_rgba8<<<gridOut, block, 0, g_upscale.stream>>>(
            (const half*)g_upscale.d_output, (unsigned char*)dstPtr,
            g_upscale.outputW, g_upscale.outputH);
    } else {
        // Bilinear fallback
        dim3 grid((g_upscale.outputW + 15) / 16, (g_upscale.outputH + 15) / 16);
        bilinear_upscale_rgba8<<<grid, block, 0, g_upscale.stream>>>(
            (const unsigned char*)srcPtr, (unsigned char*)dstPtr,
            g_upscale.renderW, g_upscale.renderH,
            g_upscale.outputW, g_upscale.outputH);
    }

    cudaStreamSynchronize(g_upscale.stream);
    return 0;
}

// Cleanup
void rasterboost_upscale_destroy() {
    std::lock_guard<std::mutex> lock(g_upscaleLock);
    if (g_upscale.d_input)  cudaFree(g_upscale.d_input);
    if (g_upscale.d_output) cudaFree(g_upscale.d_output);
    if (g_upscale.context)  delete g_upscale.context;
    if (g_upscale.engine)   delete g_upscale.engine;
    if (g_upscale.runtime)  delete g_upscale.runtime;
    if (g_upscale.stream)   cudaStreamDestroy(g_upscale.stream);
    g_upscale = {};
}

// Query if TRT is being used (vs bilinear fallback)
int rasterboost_upscale_has_trt() {
    return (g_upscale.context != nullptr) ? 1 : 0;
}

} // extern "C"
