/* VkLayer_CudaRT — Vulkan Layer that intercepts RT calls and routes to our CUDA BVH engine
 *
 * Intercepts:
 *   - vkCmdBuildAccelerationStructuresKHR → captures geometry, builds BVH4+CWBVH via CUDA
 *   - vkCmdTraceRaysKHR → runs our CUDA tracePrimaryDense/traceCWBVH kernel
 *   - vkCreateAccelerationStructureKHR / vkDestroyAccelerationStructureKHR
 *   - vkGetAccelerationStructureBuildSizesKHR
 *   - vkCreateRayTracingPipelinesKHR
 *
 * All other Vulkan calls pass through to the real driver unchanged.
 */

#define VK_NO_PROTOTYPES
#define VK_LAYER_EXPORT __attribute__((visibility("default")))
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include "cuda_bvh_backend.h"
#include <cuda_runtime.h>
#include "spirv_ray_query_rewriter.h"
#include "shaders/bvh4_trace_spv.h"
#include "shaders/bvh2_stackless_spv.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <dlfcn.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <time.h>

// RasterBoost upscale engine (rasterboost_upscale.cu)
extern "C" {
    int  rasterboost_upscale_init(uint32_t renderW, uint32_t renderH,
                                  uint32_t outputW, uint32_t outputH,
                                  const char* plan_path);
    int  rasterboost_upscale_run(void* srcPtr, void* dstPtr);
    void rasterboost_upscale_destroy();
    int  rasterboost_upscale_has_trt();
}

// ═══════════════════════════════════════════
// Dispatch table — stores the next layer's function pointers
// ═══════════════════════════════════════════
struct InstanceDispatch {
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
    PFN_vkDestroyInstance DestroyInstance;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
    PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2;
};

struct DeviceDispatch {
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr;
    PFN_vkDestroyDevice DestroyDevice;

    // AS functions
    PFN_vkCreateAccelerationStructureKHR CreateAccelerationStructureKHR;
    PFN_vkDestroyAccelerationStructureKHR DestroyAccelerationStructureKHR;
    PFN_vkGetAccelerationStructureBuildSizesKHR GetAccelerationStructureBuildSizesKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR CmdBuildAccelerationStructuresKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR GetAccelerationStructureDeviceAddressKHR;

    // RT pipeline
    PFN_vkCreateRayTracingPipelinesKHR CreateRayTracingPipelinesKHR;
    PFN_vkGetRayTracingShaderGroupHandlesKHR GetRayTracingShaderGroupHandlesKHR;

    // Trace rays
    PFN_vkCmdTraceRaysKHR CmdTraceRaysKHR;
    PFN_vkCmdTraceRaysIndirectKHR CmdTraceRaysIndirectKHR;

    // Memory helpers
    PFN_vkGetBufferDeviceAddress GetBufferDeviceAddress;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkMapMemory MapMemory;
    PFN_vkUnmapMemory UnmapMemory;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkCmdCopyBuffer CmdCopyBuffer;
    PFN_vkCmdBindPipeline CmdBindPipeline;

    // External memory (for GPU-only buffer capture via CUDA)
    PFN_vkGetMemoryFdKHR GetMemoryFdKHR;

    // Image operations (for CUDA→VkImage interop)
    PFN_vkCreateImage CreateImage;
    PFN_vkDestroyImage DestroyImage;
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage;

    // Compute pipeline (for Vulkan-native BVH4 tracing)
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
    PFN_vkCreatePipelineLayout CreatePipelineLayout;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
    PFN_vkCreateComputePipelines CreateComputePipelines;
    PFN_vkDestroyPipeline DestroyPipeline;
    PFN_vkCreateShaderModule CreateShaderModule;
    PFN_vkDestroyShaderModule DestroyShaderModule;
    PFN_vkCreateDescriptorPool CreateDescriptorPool;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdPushConstants CmdPushConstants;
    PFN_vkCmdDispatch CmdDispatch;
    PFN_vkCmdFillBuffer CmdFillBuffer;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;

    // Render pass profiling
    PFN_vkCmdBeginRenderPass CmdBeginRenderPass;
    PFN_vkCmdEndRenderPass CmdEndRenderPass;
    PFN_vkCreateRenderPass CreateRenderPass;
    PFN_vkCmdDrawIndexed CmdDrawIndexed;
    PFN_vkCmdDraw CmdDraw;
    PFN_vkCmdDrawIndexedIndirect CmdDrawIndexedIndirect;
    PFN_vkCmdDrawIndirect CmdDrawIndirect;
    PFN_vkCmdExecuteCommands CmdExecuteCommands;
    PFN_vkCmdWaitEvents CmdWaitEvents;
    PFN_vkCmdDrawIndexedIndirectCount CmdDrawIndexedIndirectCount;
    PFN_vkCmdDrawIndirectCount CmdDrawIndirectCount;
    PFN_vkCreateQueryPool CreateQueryPool;
    PFN_vkDestroyQueryPool DestroyQueryPool;
    PFN_vkGetQueryPoolResults GetQueryPoolResults;
    PFN_vkCmdWriteTimestamp CmdWriteTimestamp;
    PFN_vkCmdResetQueryPool CmdResetQueryPool;

    // Async compute queue
    PFN_vkGetDeviceQueue GetDeviceQueue;
    PFN_vkCreateCommandPool CreateCommandPool;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkBeginCommandBuffer BeginCommandBuffer;
    PFN_vkEndCommandBuffer EndCommandBuffer;
    PFN_vkCreateSemaphore CreateSemaphore_;
    PFN_vkDestroySemaphore DestroySemaphore;
    PFN_vkCreateFence CreateFence;
    PFN_vkDestroyFence DestroyFence;
    PFN_vkWaitForFences WaitForFences;
    PFN_vkResetFences ResetFences;
    PFN_vkResetCommandBuffer ResetCommandBuffer;

    // Queue submit (for deferred BLAS build after GPU execution)
    PFN_vkQueueSubmit QueueSubmit;
    PFN_vkQueueSubmit2KHR QueueSubmit2KHR;
    PFN_vkQueueWaitIdle QueueWaitIdle;
    PFN_vkQueuePresentKHR QueuePresentKHR;
    PFN_vkCreateSwapchainKHR CreateSwapchainKHR;
    PFN_vkDestroySwapchainKHR DestroySwapchainKHR;
    PFN_vkGetSwapchainImagesKHR GetSwapchainImagesKHR;

    // Stored handles
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkPhysicalDeviceMemoryProperties memProps;
};

static std::mutex g_lock;
static std::unordered_map<void*, InstanceDispatch> g_instanceMap;
static std::unordered_map<void*, DeviceDispatch>   g_deviceMap;
static VkInstance g_instance = VK_NULL_HANDLE;  // stored for instance-level queries

// Key helper: VkDevice/VkInstance are dispatchable — first field is a pointer to dispatch table
static inline void* getKey(void* handle) { return *(void**)handle; }

// ═══════════════════════════════════════════
// Our CUDA BVH engine state (per acceleration structure)
// ═══════════════════════════════════════════
struct CudaBVHState {
    CudaBVH_t handle;   // opaque BVH from cuda_bvh_backend
    int       numTris;
    bool      isReady;  // BVH has been built
};

static std::unordered_map<uint64_t, CudaBVHState> g_bvhMap;  // keyed by AS handle
static uint64_t g_nextASHandle = 0x1000;

// ═══════════════════════════════════════════
// Buffer/Memory tracking for geometry capture
// ═══════════════════════════════════════════
struct BufferInfo {
    VkDeviceSize size;
    VkDeviceMemory memory;
    VkDeviceSize memOffset;
    VkDeviceAddress devAddr;
    VkBufferUsageFlags usage;
};
struct MemInfo {
    void* hostPtr;
    VkDeviceSize mapOffset;
    VkDeviceSize allocSize;      // total allocation size
    VkDevice device;             // owning device (for fd export)
};
static std::unordered_map<uint64_t, BufferInfo> g_buffers;   // key: VkBuffer
static std::unordered_map<uint64_t, MemInfo>    g_memories;  // key: VkDeviceMemory

// Track TLAS → BLAS linkage
static CudaBVH_t g_lastBLAS = nullptr;  // BLAS used for traversal (should be the one TLAS instances reference)
// Map device address → AS handle (for looking up which BVH a TLAS instance references)
static std::unordered_map<VkDeviceAddress, uint64_t> g_asDevAddrToHandle;

// ═══════════════════════════════════════════
// Deferred BLAS build: store geometry metadata during CmdBuildAS,
// read vertex data after QueueSubmit when GPU has executed copies
// ═══════════════════════════════════════════
struct PendingBLASGeo {
    VkDeviceAddress vertAddr;
    VkDeviceAddress idxAddr;
    VkDeviceSize    vertDataSize;
    VkDeviceSize    idxDataSize;
    uint32_t        vertexStride;
    VkFormat        vertexFormat;
    VkIndexType     indexType;
    uint32_t        primCount;
    uint32_t        maxVertex;
};
struct PendingBLAS {
    uint64_t asKey;  // VkAccelerationStructureKHR handle
    std::vector<PendingBLASGeo> geometries;
};
static std::vector<PendingBLAS> g_pendingBLAS;
static std::vector<PendingBLAS> g_latePendingBLAS;  // BLASes captured after initial build
static bool g_blasBuildsDone = false; // true once we've built BVH from real data

// ═══════════════════════════════════════════
// Deferred TLAS build: capture instance buffer address during CmdBuildAS,
// read instance transforms after QueueSubmit
// ═══════════════════════════════════════════
struct PendingTLAS {
    VkDeviceAddress instanceAddr;
    uint32_t instanceCount;
    bool pending;
};
static PendingTLAS g_pendingTLAS = {0, 0, false};

// Instance data for GPU upload (mat3x4 transform + inverse + BLAS AABB)
// Layout: 8 vec4s = 128 bytes per instance
//   [0..2] = transform rows 0,1,2 (3×vec4 = 12 floats)
//   [3..5] = invTransform rows 0,1,2 (3×vec4 = 12 floats)
//   [6]    = (blasMin.xyz, uintBitsToFloat(blasNodeOff))
//   [7]    = (blasMax.xyz, uintBitsToFloat(blasTriOff))
struct InstanceGPU {
    float transform[12];     // 3×4 row-major affine transform
    float invTransform[12];  // inverse 3×4 for transforming rays
    float blasMinX, blasMinY, blasMinZ;
    float blasMaxX, blasMaxY, blasMaxZ;
    uint32_t blasNodeOff;    // node offset into concatenated BVH2 nodes array
    uint32_t blasTriOff;     // tri offset into concatenated packed tris array
    // Per-instance metadata from VkAccelerationStructureInstanceKHR
    uint32_t customIdx;      // instanceCustomIndex (24-bit)
    uint32_t sbtOffset;      // instanceShaderBindingTableRecordOffset (24-bit)
    uint32_t instanceMask;   // instance mask (8-bit)
    uint32_t instanceFlags;  // instance flags (8-bit)
};
static std::vector<InstanceGPU> g_instances;
static CudaBVH_t g_tlasBVH = nullptr;  // TLAS BVH over instance AABBs (first build only)
// Fast TLAS path: raw BVH2 stackless data (per-frame rebuilds)
static uint32_t* g_fastTLASNodes = nullptr;
static int g_fastTLASNodeCount = 0;
static int* g_fastTLASOrdered = nullptr;  // sorted→original instance index mapping

// ═══════════════════════════════════════════
// Multi-BLAS tracking: all built BLASes concatenated into single buffers
// ═══════════════════════════════════════════
struct BLASEntry {
    CudaBVH_t bvh;
    uint64_t  asKey;         // VkAccelerationStructureKHR handle
    int       nodeOffset;    // offset into concatenated nodes array (in nodes, not bytes)
    int       triOffset;     // offset into concatenated packed tris array (in triangles, not bytes)
    int       numNodes;
    int       numTriVec4s;   // number of vec4s in packed tris (= numTris * 3)
    int       numTris;
    float     minX, minY, minZ, maxX, maxY, maxZ; // BLAS bounds from BVH root
};
static std::vector<BLASEntry> g_blasEntries;
// Map AS handle → index in g_blasEntries
static std::unordered_map<uint64_t, int> g_asKeyToBLASIdx;
// Map device address → index in g_blasEntries
static std::unordered_map<VkDeviceAddress, int> g_blasDevAddrToIdx;

// ═══════════════════════════════════════════
// Storage image tracking (for CUDA→VkImage output)
// ═══════════════════════════════════════════
struct TrackedImage {
    uint32_t width, height;
    VkFormat format;
};
static std::unordered_map<uint64_t, TrackedImage> g_storageImages;  // key: VkImage

// ═══════════════════════════════════════════
// RasterBoost: Resolution substitution state
// Env: RASTER_BOOST_SCALE=0.5 → render at 50% res, upscale at present
// ═══════════════════════════════════════════
struct RasterBoostState {
    float    scale;            // 0.0 = disabled, 0.5 = half res, etc.
    uint32_t outputW, outputH; // Swapchain (display) resolution
    uint32_t renderW, renderH; // Internal render resolution
    bool     active;
    uint32_t scaledImages;     // count of images we downscaled
    VkDevice device;
};
static RasterBoostState g_rasterBoost = {};

static void rasterBoostInit() {
    const char* env = getenv("RASTER_BOOST_SCALE");
    if (env) {
        g_rasterBoost.scale = atof(env);
        if (g_rasterBoost.scale > 0.0f && g_rasterBoost.scale < 1.0f) {
            g_rasterBoost.active = true;
            fprintf(stderr, "[CudaRT] [RasterBoost] Resolution scale: %.2f\n", g_rasterBoost.scale);
        }
    }
}

// Forward declarations for init functions defined after LOG macro
static void drawBatchInit();

// ═══════════════════════════════════════════
// Render pass profiling: GPU timestamps per render pass
// ═══════════════════════════════════════════
struct RenderPassInfo {
    uint32_t colorAttachments;
    uint32_t depthAttachment;   // 0 or 1
    uint32_t subpassCount;
    VkFormat depthFormat;       // G-Buffer: depth attachment format
    uint32_t depthAttachIdx;    // G-Buffer: index in pAttachments (UINT32_MAX if none)
    // Track color formats for motion vector detection (R16G16_SFLOAT = common MV format)
    VkFormat colorFormats[8];
    int      motionVectorIdx;   // Index of likely MV attachment (-1 if none)
};
static std::unordered_map<uint64_t, RenderPassInfo> g_renderPassInfo;

// ═══════════════════════════════════════════
// G-Buffer: Framebuffer → image view tracking
// ═══════════════════════════════════════════
struct FramebufferInfo {
    std::vector<VkImageView> attachments;
    uint32_t width, height;
    VkRenderPass renderPass;
};
static std::unordered_map<uint64_t, FramebufferInfo> g_framebufferInfo;

// ImageView → VkImage mapping (needed to get actual images from framebuffers)
struct ImageViewInfo {
    VkImage  image;
    VkFormat format;
    uint32_t baseMip;
    uint32_t baseLayer;
};
static std::unordered_map<uint64_t, ImageViewInfo> g_imageViewInfo;

// G-Buffer capture state: depth + motion vectors for TRT upscaler
struct GBufferCapture {
    VkImage  depthImage;        // Current frame's depth image
    VkImage  motionImage;       // Current frame's motion vector image
    VkFormat depthFormat;
    VkFormat motionFormat;
    uint32_t width, height;
    bool     captured;          // Set after successful capture
    void*    cudaDepthPtr;      // CUDA device pointer to imported depth
    void*    cudaMotionPtr;     // CUDA device pointer to imported motion
};
static GBufferCapture g_gbuffer = {};

struct FrameProfile {
    VkQueryPool queryPool;
    uint32_t    queryIdx;       // next query slot to use
    bool        ready;
    uint32_t    rpCount;        // render passes this frame
    // Per-pass timing
    struct PassTiming {
        uint64_t renderPass;    // handle
        uint32_t startQuery;
        uint32_t endQuery;
        uint32_t width, height;
        uint32_t colorAttachments;
    };
    std::vector<PassTiming> passes;
    // Compute dispatch timing
    uint32_t computeStartQuery;
    uint32_t computeEndQuery;
    uint32_t computeDispatches;
    // RP2 mid-pass timestamp
    uint32_t rp2MidQuery;
    uint32_t rp2PreEndQuery;
};
static FrameProfile g_profile = {};
static float g_timestampPeriod = 0.0f;  // ns per tick

// ═══════════════════════════════════════════
// Async compute queue: submit RT dispatches on family 2
// ═══════════════════════════════════════════
struct AsyncCompute {
    VkQueue       queue;
    uint32_t      queueFamily;
    VkCommandPool cmdPool;
    VkCommandBuffer cmdBuf;
    VkSemaphore   rtDoneSemaphore;  // signals when RT dispatch completes
    VkFence       fence;
    bool          ready;
    bool          hasPendingWork;
    // Track what to dispatch
    VkPipeline    pipeline;
    VkPipelineLayout layout;
    VkDescriptorSet descSets[8];
    uint32_t      descSetCount;
    uint32_t      groupCountX, groupCountY, groupCountZ;
};
static AsyncCompute g_async = {};

// CUDA→Vulkan shared staging buffer (DEVICE_LOCAL with external memory)
struct StagingInterop {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* hostPtr;      // HOST_VISIBLE path: persistently mapped host pointer
    void* cudaPtr;      // DEVICE_LOCAL path: CUDA-imported GPU pointer
    uint32_t width, height;
    VkDeviceSize size;
    bool ready;
    bool deviceLocal;   // true = DEVICE_LOCAL + external memory (fast GPU→GPU copy)
};
static StagingInterop g_staging = {};

// ═══════════════════════════════════════════
// Vulkan compute pipeline for BVH4 tracing (zero CUDA overhead)
// ═══════════════════════════════════════════
struct ComputeTracer {
    VkDescriptorSetLayout dsLayout;
    VkPipelineLayout pipeLayout;
    VkPipeline pipeline;
    VkDescriptorPool descPool;
    VkDescriptorSet descSet;

    // BVH data buffers (uploaded from CUDA BVH builder)
    VkBuffer bvhNodesBuf;          // BVH4 nodes (uvec4 array)
    VkBuffer triBufs[9];           // SoA triangle arrays (tv0x..tv2z)
    VkDeviceMemory bvhNodesMem;
    VkBuffer topNodesBuf;          // UBO: top 1024 BVH nodes (constant cache)
    VkDeviceMemory topNodesMem;
    VkDeviceMemory triMems[9];

    // Camera push constants
    struct PushConstants {
        float camOx, camOy, camOz;
        float fwdX, fwdY, fwdZ;
        float rightX, rightY, rightZ;
        float upX, upY, upZ;
        float fov;
        float nearZ, farZ;
        int width, height;
        int numNodes;
        int outFmt;
    } pc;

    bool pipelineReady;
    bool dataUploaded;
    CudaBVH_t lastBVH;  // track which BVH is uploaded
};
static ComputeTracer g_compute = {};

// ═══════════════════════════════════════════
// BVH2 interop: Vulkan SSBOs for SPIR-V ray query traversal
// ═══════════════════════════════════════════
struct BVH2Interop {
    VkBuffer nodesBuf;           // binding 0: BVH2 stackless nodes (uvec4 runtime array)
    VkBuffer trisBuf;            // binding 1: Packed triangles (vec4 runtime array)
    VkBuffer tlasNodesBuf;       // binding 2: TLAS BVH2 nodes (over instances)
    VkBuffer instancesBuf;       // binding 3: InstanceGPU array (vec4 runtime array)
    VkDeviceMemory nodesMem;
    VkDeviceMemory trisMem;
    VkDeviceMemory tlasNodesMem;
    VkDeviceMemory instancesMem;
    VkDescriptorSetLayout dsLayout;
    VkDescriptorPool descPool;
    VkDescriptorSet descSet;
    uint32_t descSetIdx;         // Which descriptor set index (4)
    bool ready;
    int numNodes;
    int numTriVec4s;
    int numTlasNodes;
    int numInstances;
    uint64_t tlasGen;            // which TLAS generation is currently uploaded
};
static BVH2Interop g_bvh2 = {};

static int g_verbose = 1;
uint32_t g_rqDispatchPerFrame = 0; // reset per frame in QueuePresent
static int g_trackVerbose = 0;
#define LOG(fmt, ...) do { if (g_verbose) fprintf(stderr, "[CudaRT] " fmt "\n", ##__VA_ARGS__); } while(0)

// ═══════════════════════════════════════════
// Layer: CreateInstance
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateInstance(
    const VkInstanceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance)
{
    // Find the loader's chain info
    VkLayerInstanceCreateInfo* chain = (VkLayerInstanceCreateInfo*)pCreateInfo->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerInstanceCreateInfo*)chain->pNext;

    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr nextGIPA = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    // Advance chain for next layer
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;

    auto createFunc = (PFN_vkCreateInstance)nextGIPA(VK_NULL_HANDLE, "vkCreateInstance");
    VkResult res = createFunc(pCreateInfo, pAllocator, pInstance);
    if (res != VK_SUCCESS) return res;

    InstanceDispatch disp = {};
    disp.GetInstanceProcAddr = nextGIPA;
    #define LOAD_INST(fn) disp.fn = (PFN_vk##fn)nextGIPA(*pInstance, "vk" #fn)
    LOAD_INST(DestroyInstance);
    LOAD_INST(EnumerateDeviceExtensionProperties);
    LOAD_INST(GetPhysicalDeviceProperties);
    LOAD_INST(GetPhysicalDeviceProperties2);
    LOAD_INST(GetPhysicalDeviceFeatures2);
    #undef LOAD_INST

    std::lock_guard<std::mutex> lock(g_lock);
    g_instanceMap[getKey(*pInstance)] = disp;
    g_instance = *pInstance;

    LOG("Instance created — CUDA RT layer active");
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL layer_DestroyInstance(
    VkInstance instance, const VkAllocationCallbacks* pAllocator)
{
    void* key = getKey(instance);
    std::lock_guard<std::mutex> lock(g_lock);
    auto it = g_instanceMap.find(key);
    if (it != g_instanceMap.end()) {
        it->second.DestroyInstance(instance, pAllocator);
        g_instanceMap.erase(it);
    }
}

// ═══════════════════════════════════════════
// Layer: Intercept GetPhysicalDeviceProperties2
// Ensure RT properties look correct
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_GetPhysicalDeviceProperties2(
    VkPhysicalDevice physDev,
    VkPhysicalDeviceProperties2* pProperties)
{
    void* key = getKey(physDev);
    InstanceDispatch* pDisp = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        auto it = g_instanceMap.find(key);
        if (it != g_instanceMap.end()) pDisp = &it->second;
    }
    if (pDisp && pDisp->GetPhysicalDeviceProperties2)
        pDisp->GetPhysicalDeviceProperties2(physDev, pProperties);

    // Walk pNext chain and ensure RT properties are sane
    VkBaseOutStructure* s = (VkBaseOutStructure*)pProperties->pNext;
    while (s) {
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR) {
            auto* p = (VkPhysicalDeviceRayTracingPipelinePropertiesKHR*)s;
            // Ensure minimum values that apps expect
            if (p->maxRayRecursionDepth < 1) p->maxRayRecursionDepth = 1;
            if (p->shaderGroupHandleSize == 0) p->shaderGroupHandleSize = 32;
            if (p->maxShaderGroupStride == 0) p->maxShaderGroupStride = 4096;
            if (p->shaderGroupBaseAlignment == 0) p->shaderGroupBaseAlignment = 64;
            LOG("  → RT pipeline props: maxRecursion=%u handleSize=%u",
                p->maxRayRecursionDepth, p->shaderGroupHandleSize);
        }
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR) {
            auto* p = (VkPhysicalDeviceAccelerationStructurePropertiesKHR*)s;
            LOG("  → AS props: maxGeometryCount=%lu maxInstanceCount=%lu",
                (unsigned long)p->maxGeometryCount, (unsigned long)p->maxInstanceCount);
        }
        s = s->pNext;
    }
}

// ═══════════════════════════════════════════
// Layer: Intercept GetPhysicalDeviceFeatures2
// Ensure RT features are always reported as supported
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_GetPhysicalDeviceFeatures2(
    VkPhysicalDevice physDev,
    VkPhysicalDeviceFeatures2* pFeatures)
{
    // Use physDev's dispatch key to find the correct instance dispatch
    void* key = getKey(physDev);
    InstanceDispatch* pDisp = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        auto it = g_instanceMap.find(key);
        if (it != g_instanceMap.end()) {
            pDisp = &it->second;
        } else {
            // Dispatch key not found — try all instances (physDev dispatch key may differ)
            LOG("Features2: physDev key %p not in instanceMap (%zu entries), trying all",
                key, g_instanceMap.size());
            for (auto& [k, disp] : g_instanceMap) {
                if (disp.GetPhysicalDeviceFeatures2) {
                    pDisp = &disp;
                    break;
                }
            }
        }
    }
    if (pDisp && pDisp->GetPhysicalDeviceFeatures2)
        pDisp->GetPhysicalDeviceFeatures2(physDev, pFeatures);

    // Check if this is an NVIDIA GPU before force-enabling RT features
    bool isNvidia = false;
    if (pDisp && pDisp->GetPhysicalDeviceProperties) {
        VkPhysicalDeviceProperties props;
        pDisp->GetPhysicalDeviceProperties(physDev, &props);
        isNvidia = (props.vendorID == 0x10DE);
    }

    if (!isNvidia) return;  // Don't touch non-NVIDIA devices

    // Walk pNext chain: log driver values, then force-enable RT features on NVIDIA
    VkBaseOutStructure* s = (VkBaseOutStructure*)pFeatures->pNext;
    while (s) {
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR) {
            auto* f = (VkPhysicalDeviceAccelerationStructureFeaturesKHR*)s;
            LOG("Features2[NV]: AS: accelStruct=%d", f->accelerationStructure);
            f->accelerationStructure = VK_TRUE;
        }
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR) {
            auto* f = (VkPhysicalDeviceRayTracingPipelineFeaturesKHR*)s;
            LOG("Features2[NV]: RTP: rtPipeline=%d rtTraceIndirect=%d",
                f->rayTracingPipeline, f->rayTracingPipelineTraceRaysIndirect);
            f->rayTracingPipeline = VK_TRUE;
            f->rayTracingPipelineTraceRaysIndirect = VK_TRUE;
        }
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR) {
            auto* f = (VkPhysicalDeviceRayQueryFeaturesKHR*)s;
            LOG("Features2[NV]: RQ: rayQuery=%d", f->rayQuery);
            f->rayQuery = VK_TRUE;
        }
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES_KHR) {
            auto* f = (VkPhysicalDeviceRayTracingMaintenance1FeaturesKHR*)s;
            f->rayTracingMaintenance1 = VK_TRUE;
        }
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES) {
            auto* f = (VkPhysicalDeviceBufferDeviceAddressFeatures*)s;
            f->bufferDeviceAddress = VK_TRUE;
        }
        s = s->pNext;
    }
}

// ═══════════════════════════════════════════
// Layer: Intercept EnumerateDeviceExtensionProperties
// Inject RT extensions so apps detect RT as available
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_EnumerateDeviceExtensionProperties(
    VkPhysicalDevice physDev,
    const char* pLayerName,
    uint32_t* pPropertyCount,
    VkExtensionProperties* pProperties)
{
    // If querying a specific layer's extensions, pass through
    if (pLayerName) {
        *pPropertyCount = 0;
        return VK_SUCCESS;
    }

    // Get the real driver's extensions using physDev's dispatch key
    PFN_vkEnumerateDeviceExtensionProperties nextFn = nullptr;
    {
        void* key = getKey(physDev);
        std::lock_guard<std::mutex> lock(g_lock);
        auto it = g_instanceMap.find(key);
        if (it != g_instanceMap.end())
            nextFn = it->second.EnumerateDeviceExtensionProperties;
    }
    if (!nextFn) {
        *pPropertyCount = 0;
        return VK_SUCCESS;
    }

    // Hide VK_KHR_ray_tracing_pipeline so apps fall back to ray_query (our layer path).
    // V100 driver exposes native software RT pipeline which produces broken rendering.
    bool hidePipeline = true;
    if (getenv("CUDA_RT_HIDE_PIPELINE") && atoi(getenv("CUDA_RT_HIDE_PIPELINE")) == 0)
        hidePipeline = false;

    // RT extensions we want to inject (excluding pipeline if hidden)
    std::vector<const char*> rtExtNames = {
        "VK_KHR_acceleration_structure",
        "VK_KHR_ray_query",
        "VK_KHR_ray_tracing_maintenance1",
        "VK_KHR_deferred_host_operations",
        "VK_KHR_spirv_1_4",
        "VK_KHR_shader_float_controls",
    };
    if (!hidePipeline)
        rtExtNames.push_back("VK_KHR_ray_tracing_pipeline");

    // First call: get driver count
    uint32_t driverCount = 0;
    VkResult res = nextFn(physDev, nullptr, &driverCount, nullptr);
    if (res != VK_SUCCESS) return res;

    // Get driver extensions
    std::vector<VkExtensionProperties> driverExts(driverCount);
    res = nextFn(physDev, nullptr, &driverCount, driverExts.data());
    if (res != VK_SUCCESS) return res;

    // Check which RT extensions are missing
    auto hasExt = [&](const char* name) {
        for (uint32_t i = 0; i < driverCount; i++)
            if (strcmp(driverExts[i].extensionName, name) == 0) return true;
        return false;
    };

    // Build merged list, stripping pipeline ext if hidden
    std::vector<VkExtensionProperties> merged;
    for (auto& ext : driverExts) {
        if (hidePipeline && strcmp(ext.extensionName, "VK_KHR_ray_tracing_pipeline") == 0) {
            LOG("  → Hidden extension: %s (forcing ray_query path)", ext.extensionName);
            continue;
        }
        merged.push_back(ext);
    }
    for (auto& rtName : rtExtNames) {
        if (!hasExt(rtName)) {
            VkExtensionProperties ext{};
            strncpy(ext.extensionName, rtName, VK_MAX_EXTENSION_NAME_SIZE - 1);
            ext.specVersion = 1;
            merged.push_back(ext);
            LOG("  → Injected extension: %s", rtName);
        }
    }

    uint32_t totalCount = (uint32_t)merged.size();
    if (!pProperties) {
        *pPropertyCount = totalCount;
        return VK_SUCCESS;
    }

    uint32_t copyCount = std::min(*pPropertyCount, totalCount);
    memcpy(pProperties, merged.data(), copyCount * sizeof(VkExtensionProperties));
    *pPropertyCount = copyCount;
    return (copyCount < totalCount) ? VK_INCOMPLETE : VK_SUCCESS;
}

// ═══════════════════════════════════════════
// Layer: CreateDevice — inject RT extensions if missing
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateDevice(
    VkPhysicalDevice gpu,
    const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDevice* pDevice)
{
    // Log all requested extensions
    LOG("CreateDevice: %u extensions requested:", pCreateInfo->enabledExtensionCount);
    for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        LOG("  ext[%u]: %s", i, pCreateInfo->ppEnabledExtensionNames[i]);
    }

    // RT extensions — V100 driver supports all of these in software.
    // Don't strip by default: let driver handle AS descriptors, pipelines, etc.
    // We only intercept CmdBuildAS (to build our BVH) and CmdTraceRays (to replace trace).
    // Set CUDA_RT_STRIP_EXTENSIONS=1 to strip (requires full reimplementation).
    bool stripRT = (getenv("CUDA_RT_STRIP_EXTENSIONS") && atoi(getenv("CUDA_RT_STRIP_EXTENSIONS")));
    static const char* rtExts[] = {
        "VK_KHR_acceleration_structure",
        "VK_KHR_ray_tracing_pipeline",
        "VK_KHR_ray_query",
        "VK_KHR_ray_tracing_maintenance1",
        "VK_KHR_deferred_host_operations",
    };
    auto isRTExt = [](const char* name) {
        for (auto& rt : rtExts)
            if (strcmp(name, rt) == 0) return true;
        return false;
    };

    // Build extension list (optionally strip RT, always inject external_memory_fd)
    // Also always strip ray_tracing_pipeline when hidden (force ray_query path)
    bool hidePipeline2 = true;
    if (getenv("CUDA_RT_HIDE_PIPELINE") && atoi(getenv("CUDA_RT_HIDE_PIPELINE")) == 0)
        hidePipeline2 = false;
    std::vector<const char*> filteredExts;
    bool hasExternalMemFd = false;
    for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        const char* extName = pCreateInfo->ppEnabledExtensionNames[i];
        if (stripRT && isRTExt(extName)) {
            LOG("  → Stripping RT extension: %s (handled by CudaRT layer)", extName);
        } else if (hidePipeline2 && strcmp(extName, "VK_KHR_ray_tracing_pipeline") == 0) {
            LOG("  → Stripping %s (forcing ray_query path)", extName);
        } else {
            filteredExts.push_back(extName);
            if (!strcmp(extName, "VK_KHR_external_memory_fd"))
                hasExternalMemFd = true;
        }
    }
    // Inject VK_KHR_external_memory_fd for GPU memory capture
    if (!hasExternalMemFd) {
        filteredExts.push_back("VK_KHR_external_memory_fd");
        LOG("  → Injecting VK_KHR_external_memory_fd for GPU memory capture");
    }

    // Strip RT feature structs from pNext chain before forwarding to driver.
    // We need a mutable copy of the chain. Walk and relink, skipping RT sTypes.
    static const VkStructureType rtSTypes[] = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES_KHR,
    };
    auto isRTSType = [](VkStructureType st) {
        for (auto rt : rtSTypes)
            if (st == rt) return true;
        return false;
    };

    // Find the layer chain info FIRST (before modifying pNext chain)
    VkLayerDeviceCreateInfo* chain = (VkLayerDeviceCreateInfo*)pCreateInfo->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerDeviceCreateInfo*)chain->pNext;

    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr nextGIPA = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr nextGDPA = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;

    // Now strip RT feature structs from pNext chain before forwarding to driver.
    VkDeviceCreateInfo modifiedCI = *pCreateInfo;
    modifiedCI.enabledExtensionCount = (uint32_t)filteredExts.size();
    modifiedCI.ppEnabledExtensionNames = filteredExts.data();

    if (stripRT || hidePipeline2) {
        // Relink pNext chain, skipping RT feature structs (or just pipeline if only hiding)
        auto shouldStrip = [&](VkStructureType st) {
            if (stripRT && isRTSType(st)) return true;
            if (hidePipeline2 && st == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR) return true;
            return false;
        };
        VkBaseOutStructure* newHead = nullptr;
        VkBaseOutStructure* newTail = nullptr;
        VkBaseOutStructure* cur = (VkBaseOutStructure*)pCreateInfo->pNext;
        while (cur) {
            VkBaseOutStructure* next = cur->pNext;
            if (shouldStrip(cur->sType)) {
                LOG("  → Stripping pNext sType=%u from device create", cur->sType);
                cur->pNext = next;
            } else {
                if (!newHead) { newHead = cur; newTail = cur; }
                else { newTail->pNext = cur; newTail = cur; }
            }
            cur = next;
        }
        if (newTail) newTail->pNext = nullptr;
        modifiedCI.pNext = newHead;
    }

    // Always clear rayQuery feature — the V100 driver may not support it natively,
    // our layer handles it via SPIR-V rewrite regardless
    {
        VkBaseOutStructure* cur = (VkBaseOutStructure*)modifiedCI.pNext;
        while (cur) {
            if (cur->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR) {
                auto* f = (VkPhysicalDeviceRayQueryFeaturesKHR*)cur;
                if (f->rayQuery) {
                    LOG("  → Clearing rayQuery=1→0 in CreateDevice (layer handles via SPIR-V rewrite)");
                    f->rayQuery = VK_FALSE;
                }
            }
            cur = cur->pNext;
        }
    }

    auto createFunc = (PFN_vkCreateDevice)nextGIPA(VK_NULL_HANDLE, "vkCreateDevice");

    LOG("  → Forwarding CreateDevice with %u extensions%s", (uint32_t)filteredExts.size(),
        stripRT ? " (RT stripped)" : " (RT passthrough)");
    VkResult res = createFunc(gpu, &modifiedCI, pAllocator, pDevice);
    if (res != VK_SUCCESS) {
        LOG("  → CreateDevice FAILED: %d", res);
        return res;
    }

    DeviceDispatch disp = {};
    disp.GetDeviceProcAddr = nextGDPA;
    #define LOAD_DEV(fn) disp.fn = (PFN_vk##fn)nextGDPA(*pDevice, "vk" #fn)
    LOAD_DEV(DestroyDevice);
    LOAD_DEV(CreateAccelerationStructureKHR);
    LOAD_DEV(DestroyAccelerationStructureKHR);
    LOAD_DEV(GetAccelerationStructureBuildSizesKHR);
    LOAD_DEV(CmdBuildAccelerationStructuresKHR);
    LOAD_DEV(GetAccelerationStructureDeviceAddressKHR);
    LOAD_DEV(CreateRayTracingPipelinesKHR);
    LOAD_DEV(GetRayTracingShaderGroupHandlesKHR);
    LOAD_DEV(CmdTraceRaysKHR);
    LOAD_DEV(CmdTraceRaysIndirectKHR);
    LOAD_DEV(GetBufferDeviceAddress);
    LOAD_DEV(AllocateMemory);
    LOAD_DEV(FreeMemory);
    LOAD_DEV(MapMemory);
    LOAD_DEV(UnmapMemory);
    LOAD_DEV(CreateBuffer);
    LOAD_DEV(DestroyBuffer);
    LOAD_DEV(GetBufferMemoryRequirements);
    LOAD_DEV(BindBufferMemory);
    LOAD_DEV(CmdPipelineBarrier);
    LOAD_DEV(CmdCopyBuffer);
    LOAD_DEV(CmdBindPipeline);
    // External memory fd export for GPU-only buffer capture via CUDA
    disp.GetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)nextGDPA(*pDevice, "vkGetMemoryFdKHR");
    // Image operations for CUDA→VkImage interop
    LOAD_DEV(CreateImage);
    LOAD_DEV(DestroyImage);
    LOAD_DEV(BindImageMemory);
    LOAD_DEV(CmdCopyBufferToImage);
    // Compute pipeline (for Vulkan-native BVH4 tracing)
    LOAD_DEV(CreateDescriptorSetLayout);
    LOAD_DEV(DestroyDescriptorSetLayout);
    LOAD_DEV(CreatePipelineLayout);
    LOAD_DEV(DestroyPipelineLayout);
    LOAD_DEV(CreateComputePipelines);
    LOAD_DEV(DestroyPipeline);
    LOAD_DEV(CreateShaderModule);
    LOAD_DEV(DestroyShaderModule);
    LOAD_DEV(CreateDescriptorPool);
    LOAD_DEV(DestroyDescriptorPool);
    LOAD_DEV(AllocateDescriptorSets);
    LOAD_DEV(UpdateDescriptorSets);
    LOAD_DEV(CmdBindDescriptorSets);
    LOAD_DEV(CmdPushConstants);
    LOAD_DEV(CmdDispatch);
    LOAD_DEV(CmdFillBuffer);
    LOAD_DEV(GetImageMemoryRequirements);
    // Render pass profiling
    LOAD_DEV(CmdBeginRenderPass);
    LOAD_DEV(CmdEndRenderPass);
    LOAD_DEV(CreateRenderPass);
    LOAD_DEV(CmdDrawIndexed);
    LOAD_DEV(CmdDraw);
    LOAD_DEV(CmdDrawIndexedIndirect);
    LOAD_DEV(CmdDrawIndirect);
    LOAD_DEV(CmdExecuteCommands);
    LOAD_DEV(CmdWaitEvents);
    LOAD_DEV(CmdDrawIndexedIndirectCount);
    LOAD_DEV(CmdDrawIndirectCount);
    LOAD_DEV(CreateQueryPool);
    LOAD_DEV(DestroyQueryPool);
    LOAD_DEV(GetQueryPoolResults);
    LOAD_DEV(CmdWriteTimestamp);
    LOAD_DEV(CmdResetQueryPool);
    // Async compute queue
    LOAD_DEV(GetDeviceQueue);
    LOAD_DEV(CreateCommandPool);
    LOAD_DEV(AllocateCommandBuffers);
    LOAD_DEV(BeginCommandBuffer);
    LOAD_DEV(EndCommandBuffer);
    disp.CreateSemaphore_ = (PFN_vkCreateSemaphore)nextGDPA(*pDevice, "vkCreateSemaphore");
    LOAD_DEV(DestroySemaphore);
    LOAD_DEV(CreateFence);
    LOAD_DEV(DestroyFence);
    LOAD_DEV(WaitForFences);
    LOAD_DEV(ResetFences);
    LOAD_DEV(ResetCommandBuffer);
    LOAD_DEV(QueueSubmit);
    // QueueSubmit2KHR might not exist — load manually
    disp.QueueSubmit2KHR = (PFN_vkQueueSubmit2KHR)nextGDPA(*pDevice, "vkQueueSubmit2KHR");
    if (!disp.QueueSubmit2KHR)
        disp.QueueSubmit2KHR = (PFN_vkQueueSubmit2KHR)nextGDPA(*pDevice, "vkQueueSubmit2");
    LOAD_DEV(QueueWaitIdle);
    disp.QueuePresentKHR = (PFN_vkQueuePresentKHR)nextGDPA(*pDevice, "vkQueuePresentKHR");
    disp.CreateSwapchainKHR = (PFN_vkCreateSwapchainKHR)nextGDPA(*pDevice, "vkCreateSwapchainKHR");
    disp.DestroySwapchainKHR = (PFN_vkDestroySwapchainKHR)nextGDPA(*pDevice, "vkDestroySwapchainKHR");
    disp.GetSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR)nextGDPA(*pDevice, "vkGetSwapchainImagesKHR");
    rasterBoostInit();  // Read RASTER_BOOST_SCALE env var
    drawBatchInit();    // Read RASTER_BOOST_BATCH env var
    disp.device = *pDevice;
    disp.physicalDevice = gpu;
    // Query and store memory properties for staging buffer allocation
    {
        auto getMemProps = (PFN_vkGetPhysicalDeviceMemoryProperties)
            nextGIPA(g_instance, "vkGetPhysicalDeviceMemoryProperties");
        if (getMemProps) {
            getMemProps(gpu, &disp.memProps);
            LOG("  → Cached %u memory types for staging buffer",
                disp.memProps.memoryTypeCount);
        } else {
            LOG("  → WARNING: could not load vkGetPhysicalDeviceMemoryProperties");
            memset(&disp.memProps, 0, sizeof(disp.memProps));
        }
    }
    #undef LOAD_DEV

    std::lock_guard<std::mutex> lock(g_lock);
    g_deviceMap[getKey(*pDevice)] = disp;

    // ── Setup async compute queue (family 2 = compute-only on V100) ──
    {
        // Get physical device properties for timestamp period
        auto getProps = (PFN_vkGetPhysicalDeviceProperties)
            nextGIPA(g_instance, "vkGetPhysicalDeviceProperties");
        if (getProps) {
            VkPhysicalDeviceProperties props;
            getProps(gpu, &props);
            g_timestampPeriod = props.limits.timestampPeriod; // ns per tick
            LOG("  → Timestamp period: %.2f ns/tick", g_timestampPeriod);
        }

        // Enumerate queue families to find async compute (compute-only)
        auto getQueueFamilyProps = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)
            nextGIPA(g_instance, "vkGetPhysicalDeviceQueueFamilyProperties");
        uint32_t qfCount = 0;
        if (getQueueFamilyProps) {
            getQueueFamilyProps(gpu, &qfCount, nullptr);
            std::vector<VkQueueFamilyProperties> qfProps(qfCount);
            getQueueFamilyProps(gpu, &qfCount, qfProps.data());

            // Find compute-only queue family (no graphics bit)
            uint32_t asyncFamily = UINT32_MAX;
            for (uint32_t i = 0; i < qfCount; i++) {
                if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                    !(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                    qfProps[i].queueCount > 0) {
                    asyncFamily = i;
                    LOG("  → Found async compute queue family %u (%u queues)",
                        i, qfProps[i].queueCount);
                    break;
                }
            }

            if (asyncFamily != UINT32_MAX) {
                // Check if app already requested this queue family
                bool appRequestedIt = false;
                for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
                    if (pCreateInfo->pQueueCreateInfos[i].queueFamilyIndex == asyncFamily) {
                        appRequestedIt = true;
                        break;
                    }
                }

                // Try to get a queue from this family
                // If the app didn't request it, we need to recreate the device... skip for now
                // and use a queue from family 0 (app always requests family 0)
                if (!appRequestedIt) {
                    // Use last queue from family 0 as our "async" queue
                    // This won't give true async overlap but avoids device recreation
                    uint32_t family0Count = 0;
                    for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
                        if (pCreateInfo->pQueueCreateInfos[i].queueFamilyIndex == 0) {
                            family0Count = pCreateInfo->pQueueCreateInfos[i].queueCount;
                            break;
                        }
                    }
                    if (family0Count > 1) {
                        // Use queue index 1 from family 0 for async work
                        disp.GetDeviceQueue(*pDevice, 0, family0Count - 1, &g_async.queue);
                        g_async.queueFamily = 0;
                        LOG("  → Using family 0 queue %u for async compute (shared family)", family0Count - 1);
                    } else {
                        LOG("  → Only 1 queue in family 0, async compute disabled");
                    }
                } else {
                    // App requested this family — get queue 0 from it
                    disp.GetDeviceQueue(*pDevice, asyncFamily, 0, &g_async.queue);
                    g_async.queueFamily = asyncFamily;
                    LOG("  → Got async compute queue from family %u", asyncFamily);
                }

                if (g_async.queue) {
                    // Create command pool for async queue
                    VkCommandPoolCreateInfo cpCI = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
                    cpCI.queueFamilyIndex = g_async.queueFamily;
                    cpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                    disp.CreateCommandPool(*pDevice, &cpCI, nullptr, &g_async.cmdPool);

                    VkCommandBufferAllocateInfo cbAI = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                    cbAI.commandPool = g_async.cmdPool;
                    cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                    cbAI.commandBufferCount = 1;
                    disp.AllocateCommandBuffers(*pDevice, &cbAI, &g_async.cmdBuf);

                    // Create semaphore + fence for sync
                    VkSemaphoreCreateInfo semCI = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
                    disp.CreateSemaphore_(*pDevice, &semCI, nullptr, &g_async.rtDoneSemaphore);

                    VkFenceCreateInfo fenceCI = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                    fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT; // start signaled
                    disp.CreateFence(*pDevice, &fenceCI, nullptr, &g_async.fence);

                    g_async.ready = true;
                    g_async.hasPendingWork = false;
                    LOG("  → Async compute ready: queue=%p pool=%p cmdBuf=%p",
                        g_async.queue, g_async.cmdPool, g_async.cmdBuf);
                }
            }
        }

        // Create timestamp query pool for render pass profiling (32 queries = 16 passes)
        VkQueryPoolCreateInfo qpCI = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpCI.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpCI.queryCount = 64;  // 32 render passes × 2 (begin+end)
        disp.CreateQueryPool(*pDevice, &qpCI, nullptr, &g_profile.queryPool);
        g_profile.queryIdx = 0;
        g_profile.ready = true;
        LOG("  → Timestamp query pool created (64 slots)");
    }

    LOG("Device created — intercepting RT calls (async compute: %s)",
        g_async.ready ? "ENABLED" : "disabled");
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL layer_DestroyDevice(
    VkDevice device, const VkAllocationCallbacks* pAllocator)
{
    void* key = getKey(device);
    std::lock_guard<std::mutex> lock(g_lock);
    auto it = g_deviceMap.find(key);
    if (it != g_deviceMap.end()) {
        it->second.DestroyDevice(device, pAllocator);
        g_deviceMap.erase(it);
    }
    // Destroy any remaining BVH handles
    for (auto& [id, st] : g_bvhMap)
        if (st.handle) cudaBVH_destroy(st.handle);
    g_bvhMap.clear();
    g_buffers.clear();
    g_memories.clear();
    g_lastBLAS = nullptr;
    if (g_tlasBVH) { cudaBVH_destroy(g_tlasBVH); g_tlasBVH = nullptr; }
    g_instances.clear();
    g_pendingTLAS = {0, 0, false};
}

// ═══════════════════════════════════════════
// Buffer/Memory tracking interceptors
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateBuffer(
    VkDevice device,
    const VkBufferCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkBuffer* pBuffer)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];

    // Inject TRANSFER_SRC for buffers with device address (needed for staging readback)
    VkBufferCreateInfo modCI = *pCreateInfo;
    if (modCI.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        modCI.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }

    VkResult res = disp.CreateBuffer(device, &modCI, pAllocator, pBuffer);
    if (res == VK_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_lock);
        BufferInfo bi = {};
        bi.size = pCreateInfo->size;
        bi.usage = pCreateInfo->usage;
        g_buffers[(uint64_t)*pBuffer] = bi;
        // Log large SHADER_DEVICE_ADDRESS buffers (likely instance/AS buffers)
        if ((pCreateInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) && pCreateInfo->size > 1000000)
            LOG("  [track] CreateBuffer: buf=0x%lx size=%zu usage=0x%x (DEVICE_ADDR, injected TRANSFER_SRC)",
                (uint64_t)*pBuffer, (size_t)pCreateInfo->size, (uint32_t)modCI.usage);
    }
    return res;
}

static VKAPI_ATTR VkResult VKAPI_CALL layer_AllocateMemory(
    VkDevice device,
    const VkMemoryAllocateInfo* pAllocateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDeviceMemory* pMemory)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];

    // Check if this allocation should be exported for CUDA interop
    // Inject for all large allocations (needed for TLAS instance buffer reading)
    bool hasDeviceAddr = false;
    bool hasExport = false;
    const VkBaseInStructure* s = (const VkBaseInStructure*)pAllocateInfo->pNext;
    while (s) {
        if (s->sType == VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO) {
            auto flags = (const VkMemoryAllocateFlagsInfo*)s;
            if (flags->flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
                hasDeviceAddr = true;
        }
        if (s->sType == VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO)
            hasExport = true;
        s = s->pNext;
    }

    // Export DISABLED — was causing device lost by modifying all allocations
    // Only enable for the specific TLAS instance buffer when needed
    if (false && !hasExport && pAllocateInfo->allocationSize >= 1000000) {
        // Inject export info for large device-address allocations
        VkExportMemoryAllocateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        exportInfo.pNext = pAllocateInfo->pNext;

        VkMemoryAllocateInfo modAlloc = *pAllocateInfo;
        modAlloc.pNext = &exportInfo;

        VkResult res = disp.AllocateMemory(device, &modAlloc, pAllocator, pMemory);
        if (res == VK_SUCCESS) {
            std::lock_guard<std::mutex> lock(g_lock);
            auto& mi = g_memories[(uint64_t)*pMemory];
            mi.hostPtr = nullptr;
            mi.mapOffset = 0;
            mi.allocSize = pAllocateInfo->allocationSize;
            mi.device = device;
            static int expLog = 0;
            if (expLog++ < 5)
                LOG("  [track] AllocateMemory+EXPORT: mem=0x%lx size=%zu", (uint64_t)*pMemory, (size_t)pAllocateInfo->allocationSize);
        }
        return res;
    }

    VkResult res = disp.AllocateMemory(device, pAllocateInfo, pAllocator, pMemory);
    if (res == VK_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_lock);
        auto& mi = g_memories[(uint64_t)*pMemory];
        mi.hostPtr = nullptr;
        mi.mapOffset = 0;
        mi.allocSize = pAllocateInfo->allocationSize;
        mi.device = device;
        if(g_trackVerbose) LOG("  [track] AllocateMemory: mem=0x%lx size=%zu",
            (uint64_t)*pMemory, (size_t)pAllocateInfo->allocationSize);
    }
    return res;
}

static VKAPI_ATTR void VKAPI_CALL layer_DestroyBuffer(
    VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* pAllocator)
{
    {
        std::lock_guard<std::mutex> lock(g_lock);
        g_buffers.erase((uint64_t)buffer);
    }
    void* key = getKey(device);
    g_deviceMap[key].DestroyBuffer(device, buffer, pAllocator);
}

static VKAPI_ATTR VkResult VKAPI_CALL layer_BindBufferMemory(
    VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memoryOffset)
{
    {
        std::lock_guard<std::mutex> lock(g_lock);
        auto it = g_buffers.find((uint64_t)buffer);
        if (it != g_buffers.end()) {
            it->second.memory = memory;
            it->second.memOffset = memoryOffset;
            if(g_trackVerbose) LOG("  [track] BindBufferMemory: buf=0x%lx → mem=0x%lx+%zu",
                (uint64_t)buffer, (uint64_t)memory, (size_t)memoryOffset);
        }
    }
    void* key = getKey(device);
    return g_deviceMap[key].BindBufferMemory(device, buffer, memory, memoryOffset);
}

static VKAPI_ATTR VkResult VKAPI_CALL layer_MapMemory(
    VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
    VkDeviceSize size, VkMemoryMapFlags flags, void** ppData)
{
    void* key = getKey(device);
    VkResult res = g_deviceMap[key].MapMemory(device, memory, offset, size, flags, ppData);
    if (res == VK_SUCCESS && ppData && *ppData) {
        std::lock_guard<std::mutex> lock(g_lock);
        auto& mi = g_memories[(uint64_t)memory];
        mi.hostPtr = *ppData;
        mi.mapOffset = offset;
        // preserve allocSize and device from AllocateMemory
        if(g_trackVerbose) LOG("  [track] MapMemory: mem=0x%lx → host=%p offset=%zu",
            (uint64_t)memory, *ppData, (size_t)offset);
    }
    return res;
}

static VKAPI_ATTR void VKAPI_CALL layer_UnmapMemory(VkDevice device, VkDeviceMemory memory)
{
    {
        std::lock_guard<std::mutex> lock(g_lock);
        auto it = g_memories.find((uint64_t)memory);
        if (it != g_memories.end()) {
            it->second.hostPtr = nullptr;
            it->second.mapOffset = 0;
        }
    }
    void* key = getKey(device);
    g_deviceMap[key].UnmapMemory(device, memory);
}

static VKAPI_ATTR VkDeviceAddress VKAPI_CALL layer_GetBufferDeviceAddress(
    VkDevice device, const VkBufferDeviceAddressInfo* pInfo)
{
    void* key = getKey(device);
    VkDeviceAddress addr = g_deviceMap[key].GetBufferDeviceAddress(device, pInfo);
    {
        std::lock_guard<std::mutex> lock(g_lock);
        auto it = g_buffers.find((uint64_t)pInfo->buffer);
        if (it != g_buffers.end()) {
            it->second.devAddr = addr;
            if(g_trackVerbose) LOG("  [track] GetBufferDeviceAddress: buf=0x%lx → addr=0x%lx",
                (uint64_t)pInfo->buffer, (uint64_t)addr);
        }
    }
    return addr;
}

// ═══════════════════════════════════════════
// RasterBoost: Intercept swapchain creation — record output resolution
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateSwapchainKHR(
    VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];

    if (g_rasterBoost.active && pCreateInfo) {
        g_rasterBoost.outputW = pCreateInfo->imageExtent.width;
        g_rasterBoost.outputH = pCreateInfo->imageExtent.height;
        g_rasterBoost.renderW = (uint32_t)(pCreateInfo->imageExtent.width * g_rasterBoost.scale);
        g_rasterBoost.renderH = (uint32_t)(pCreateInfo->imageExtent.height * g_rasterBoost.scale);
        // Align to 8 pixels (encoder/tensor alignment)
        g_rasterBoost.renderW = (g_rasterBoost.renderW + 7u) & ~7u;
        g_rasterBoost.renderH = (g_rasterBoost.renderH + 7u) & ~7u;
        g_rasterBoost.device = device;
        g_rasterBoost.scaledImages = 0;
        LOG("[RasterBoost] Swapchain %ux%u → internal render %ux%u (scale=%.2f)",
            g_rasterBoost.outputW, g_rasterBoost.outputH,
            g_rasterBoost.renderW, g_rasterBoost.renderH,
            g_rasterBoost.scale);

        // Initialize TRT upscale engine (lazy, on first swapchain)
        const char* planPath = getenv("RASTER_BOOST_PLAN");
        if (!planPath) planPath = "/workspaces/codespace/VK_RT/vit_ray_reconstruct_lite_540p.plan";
        rasterboost_upscale_init(
            g_rasterBoost.renderW, g_rasterBoost.renderH,
            g_rasterBoost.outputW, g_rasterBoost.outputH,
            planPath);
    }

    return disp.CreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
}

// Helper: should this image be resolution-scaled?
// Heuristic: color/depth attachments matching swapchain res, not 1D/3D/cube
static bool rasterBoostShouldScale(const VkImageCreateInfo* ci) {
    if (!g_rasterBoost.active || !g_rasterBoost.outputW) return false;
    if (ci->imageType != VK_IMAGE_TYPE_2D) return false;
    if (ci->extent.depth != 1) return false;
    // Must be a render target (color or depth attachment)
    if (!(ci->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))) return false;
    // Match swapchain dimensions (exact or within 1 pixel for odd resolutions)
    uint32_t w = ci->extent.width, h = ci->extent.height;
    if (w != g_rasterBoost.outputW || h != g_rasterBoost.outputH) return false;
    // Don't scale tiny images, transfer-only, or swapchain images themselves
    if (w <= 64 || h <= 64) return false;
    return true;
}

// ═══════════════════════════════════════════
// Intercepted: CreateImage — track storage images + RasterBoost resolution sub
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateImage(
    VkDevice device, const VkImageCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkImage* pImage)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];

    // RasterBoost: substitute resolution for render targets
    if (rasterBoostShouldScale(pCreateInfo)) {
        VkImageCreateInfo modified = *pCreateInfo;
        modified.extent.width  = g_rasterBoost.renderW;
        modified.extent.height = g_rasterBoost.renderH;
        VkResult res = disp.CreateImage(device, &modified, pAllocator, pImage);
        if (res == VK_SUCCESS) {
            g_rasterBoost.scaledImages++;
            LOG("[RasterBoost] Scaled image #%u: %ux%u → %ux%u fmt=%d img=0x%lx",
                g_rasterBoost.scaledImages,
                pCreateInfo->extent.width, pCreateInfo->extent.height,
                modified.extent.width, modified.extent.height,
                pCreateInfo->format, (uint64_t)*pImage);
        }
        return res;
    }

    VkResult res = disp.CreateImage(device, pCreateInfo, pAllocator, pImage);
    if (res == VK_SUCCESS && (pCreateInfo->usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
        std::lock_guard<std::mutex> lock(g_lock);
        g_storageImages[(uint64_t)*pImage] = {
            pCreateInfo->extent.width, pCreateInfo->extent.height, pCreateInfo->format
        };
        LOG("  [track] Storage image: %ux%u fmt=%d img=0x%lx",
            pCreateInfo->extent.width, pCreateInfo->extent.height,
            pCreateInfo->format, (uint64_t)*pImage);
    }
    return res;
}

static VKAPI_ATTR void VKAPI_CALL layer_DestroyImage(
    VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator)
{
    {
        std::lock_guard<std::mutex> lock(g_lock);
        g_storageImages.erase((uint64_t)image);
    }
    void* key = getKey(device);
    g_deviceMap[key].DestroyImage(device, image, pAllocator);
}

// Helper: resolve a Vulkan device address to a host pointer via tracked buffers/memory
static void* resolveDeviceAddress(VkDeviceAddress addr) {
    std::lock_guard<std::mutex> lock(g_lock);
    for (auto& [bufId, bi] : g_buffers) {
        if (bi.devAddr && addr >= bi.devAddr && addr < bi.devAddr + bi.size) {
            auto mit = g_memories.find((uint64_t)bi.memory);
            if (mit != g_memories.end() && mit->second.hostPtr) {
                VkDeviceSize memOff = (addr - bi.devAddr) + bi.memOffset;
                static int resolveLog = 0;
                if (resolveLog < 3)
                    LOG("  → resolveDeviceAddress: addr=0x%lx → buf=0x%lx devAddr=0x%lx off=%zu bufSz=%zu memOff=%zu mapOff=%zu",
                        (uint64_t)addr, bufId, (uint64_t)bi.devAddr,
                        (size_t)(addr - bi.devAddr), (size_t)bi.size,
                        (size_t)memOff, (size_t)mit->second.mapOffset);
                resolveLog++;
                return (char*)mit->second.hostPtr + memOff - mit->second.mapOffset;
            }
        }
    }
    return nullptr;
}

// Helper: read data from a device address (works for both host-visible and GPU-only memory)
// Returns newly allocated buffer that caller must free(), or nullptr on failure.
static void* readDeviceAddressData(VkDeviceAddress addr, VkDeviceSize size) {
    // First try host-visible path (fast, zero-copy)
    void* hostPtr = resolveDeviceAddress(addr);
    if (hostPtr) {
        // Debug: check if data is all zeros
        static int hostReadLog = 0;
        if (hostReadLog < 5) {
            const uint32_t* u32 = (const uint32_t*)hostPtr;
            int nonZero = 0;
            for (size_t i = 0; i < std::min(size, (VkDeviceSize)256) / 4; i++)
                if (u32[i]) nonZero++;
            LOG("  → Host-visible read: addr=0x%lx size=%zu nonZero=%d/64 first4=[%08x %08x %08x %08x]",
                (uint64_t)addr, (size_t)size, nonZero, u32[0], u32[1], u32[2], u32[3]);
            hostReadLog++;
        }
        void* copy = malloc(size);
        memcpy(copy, hostPtr, size);
        return copy;
    }

    // GPU-only path: Use Vulkan staging buffer copy (reliable) instead of CUDA fd import
    VkDevice device = VK_NULL_HANDLE;
    VkDeviceSize srcOffset = 0;
    VkBuffer srcBuffer = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        for (auto& [bufId, bi] : g_buffers) {
            if (bi.devAddr && addr >= bi.devAddr && addr < bi.devAddr + bi.size) {
                auto mit = g_memories.find((uint64_t)bi.memory);
                if (mit == g_memories.end()) continue;
                device = mit->second.device;
                if (!device) continue;
                srcOffset = addr - bi.devAddr;
                srcBuffer = (VkBuffer)bufId;

                static int readLog = 0;
                if (readLog++ < 10)
                    LOG("  → Staging read: buf=0x%lx devAddr=0x%lx srcOff=%zu readSize=%zu bufSize=%zu",
                        bufId, (uint64_t)bi.devAddr, (size_t)srcOffset, (size_t)size, (size_t)bi.size);
                break;
            }
        }
    }
    if (!device || !srcBuffer) return nullptr;

    void* key = getKey(device);
    DeviceDispatch* dispPtr = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        auto dit = g_deviceMap.find(key);
        if (dit != g_deviceMap.end()) dispPtr = &dit->second;
    }
    if (!dispPtr) return nullptr;
    auto& disp = *dispPtr;

    // Create host-visible staging buffer
    VkBufferCreateInfo stagingCI = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    stagingCI.size = size;
    stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    stagingCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer stagingBuf;
    if (disp.CreateBuffer(device, &stagingCI, nullptr, &stagingBuf) != VK_SUCCESS) return nullptr;

    VkMemoryRequirements memReqs;
    disp.GetBufferMemoryRequirements(device, stagingBuf, &memReqs);

    // Find host-visible, host-coherent memory type
    uint32_t memTypeIdx = UINT32_MAX;
    for (uint32_t i = 0; i < disp.memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (disp.memProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memTypeIdx = i;
            break;
        }
    }
    if (memTypeIdx == UINT32_MAX) {
        disp.DestroyBuffer(device, stagingBuf, nullptr);
        return nullptr;
    }

    VkMemoryAllocateInfo allocCI = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocCI.allocationSize = memReqs.size;
    allocCI.memoryTypeIndex = memTypeIdx;

    VkDeviceMemory stagingMem;
    if (disp.AllocateMemory(device, &allocCI, nullptr, &stagingMem) != VK_SUCCESS) {
        disp.DestroyBuffer(device, stagingBuf, nullptr);
        return nullptr;
    }
    disp.BindBufferMemory(device, stagingBuf, stagingMem, 0);

    // Record + submit copy command
    auto createCP = (PFN_vkCreateCommandPool)disp.GetDeviceProcAddr(device, "vkCreateCommandPool");
    auto destroyCP = (PFN_vkDestroyCommandPool)disp.GetDeviceProcAddr(device, "vkDestroyCommandPool");
    auto allocCB = (PFN_vkAllocateCommandBuffers)disp.GetDeviceProcAddr(device, "vkAllocateCommandBuffers");
    auto beginCB = (PFN_vkBeginCommandBuffer)disp.GetDeviceProcAddr(device, "vkBeginCommandBuffer");
    auto endCB = (PFN_vkEndCommandBuffer)disp.GetDeviceProcAddr(device, "vkEndCommandBuffer");
    auto queueSubmitFn = (PFN_vkQueueSubmit)disp.GetDeviceProcAddr(device, "vkQueueSubmit");
    auto queueWait = (PFN_vkQueueWaitIdle)disp.GetDeviceProcAddr(device, "vkQueueWaitIdle");
    auto getQueue = (PFN_vkGetDeviceQueue)disp.GetDeviceProcAddr(device, "vkGetDeviceQueue");

    if (!createCP || !allocCB || !beginCB || !endCB || !queueSubmitFn || !queueWait || !getQueue) {
        disp.DestroyBuffer(device, stagingBuf, nullptr);
        disp.FreeMemory(device, stagingMem, nullptr);
        return nullptr;
    }

    VkQueue queue;
    getQueue(device, 0, 0, &queue);

    VkCommandPool cmdPool;
    VkCommandPoolCreateInfo cpCI = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpCI.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpCI.queueFamilyIndex = 0;
    createCP(device, &cpCI, nullptr, &cmdPool);

    VkCommandBuffer cmdBuf;
    VkCommandBufferAllocateInfo cbAI = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbAI.commandPool = cmdPool;
    cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = 1;
    allocCB(device, &cbAI, &cmdBuf);

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    beginCB(cmdBuf, &beginInfo);

    // Pre-fill staging buffer with sentinel to verify copy works
    auto cmdFill = (PFN_vkCmdFillBuffer)disp.GetDeviceProcAddr(device, "vkCmdFillBuffer");
    if (cmdFill) cmdFill(cmdBuf, stagingBuf, 0, size, 0xDEADBEEF);

    // Memory barrier: make sure fill completes before copy
    VkMemoryBarrier fillBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    fillBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    auto cmdBarrier = (PFN_vkCmdPipelineBarrier)disp.GetDeviceProcAddr(device, "vkCmdPipelineBarrier");
    if (cmdBarrier) cmdBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &fillBarrier, 0, nullptr, 0, nullptr);

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = srcOffset;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    auto cmdCopy = (PFN_vkCmdCopyBuffer)disp.GetDeviceProcAddr(device, "vkCmdCopyBuffer");
    if (cmdCopy) cmdCopy(cmdBuf, srcBuffer, stagingBuf, 1, &copyRegion);
    else LOG("  !! CmdCopyBuffer not available!");
    endCB(cmdBuf);

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;
    queueSubmitFn(queue, 1, &submitInfo, VK_NULL_HANDLE);
    queueWait(queue);

    // Map and read
    void* mapped = nullptr;
    disp.MapMemory(device, stagingMem, 0, size, 0, &mapped);
    void* data = nullptr;
    if (mapped) {
        // Check if sentinel was overwritten (copy actually happened)
        const uint32_t* mU = (const uint32_t*)mapped;
        static int verifyLog = 0;
        if (verifyLog < 5) {
            int sentinel = 0, zero = 0, other = 0;
            for (int i = 0; i < std::min((int)(size/4), 64); i++) {
                if (mU[i] == 0xDEADBEEF) sentinel++;
                else if (mU[i] == 0) zero++;
                else other++;
            }
            LOG("  → Staging verify: sentinel=%d zero=%d other=%d/64 first4=[%08x %08x %08x %08x]",
                sentinel, zero, other, mU[0], mU[1], mU[2], mU[3]);
            verifyLog++;
        }
        data = malloc(size);
        memcpy(data, mapped, size);
        disp.UnmapMemory(device, stagingMem);
    }

    // Cleanup
    destroyCP(device, cmdPool, nullptr);
    disp.DestroyBuffer(device, stagingBuf, nullptr);
    disp.FreeMemory(device, stagingMem, nullptr);

    if (data) {
        static int readOk = 0;
        if (readOk++ < 10)
            LOG("  → Read %zu bytes via Vulkan staging buffer copy", (size_t)size);
        return data;
    }
    return nullptr;
}

// ═══════════════════════════════════════════════════
// Intercepted: CreateShaderModule — SPIR-V ray query rewriter
// ═══════════════════════════════════════════════════
// Track rewritten shaders: shader module handle → descriptor set info for BVH binding
struct RQShaderInfo {
    int bvhDescSet;
    int bvhNodesBinding;
    int bvhTrisBinding;
};
static std::unordered_map<uint64_t, RQShaderInfo> g_rqShaders;
static std::mutex g_rqShaderMutex;

// Track pipeline layouts: Key = VkPipelineLayout handle, Value = set layouts used
struct PipelineLayoutInfo {
    std::vector<VkDescriptorSetLayout> setLayouts;
    std::vector<VkPushConstantRange> pushConstantRanges;
};
static std::unordered_map<uint64_t, PipelineLayoutInfo> g_pipelineLayouts;
static std::mutex g_pipelineLayoutMutex;

// Track compute pipelines that use rewritten ray query shaders
// Key: VkPipeline handle, Value: extended VkPipelineLayout with BVH2 set
struct RQPipelineInfo {
    VkPipelineLayout layout; // pipeline's layout (already extended with BVH2 set)
    int bvhDescSet;
};
static std::unordered_map<uint64_t, RQPipelineInfo> g_rqPipelines;
static std::mutex g_rqPipelineMutex;

// Track per-command-buffer: which pipeline is currently bound (compute)
static std::unordered_map<uint64_t, uint64_t> g_cmdBufBoundPipeline; // cmdBuf → VkPipeline
static std::mutex g_cmdBufMutex;

static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateShaderModule(
    VkDevice device,
    const VkShaderModuleCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkShaderModule* pShaderModule)
{
    fprintf(stderr, "[CudaRT] >>> layer_CreateShaderModule called! size=%zu\n",
            pCreateInfo ? pCreateInfo->codeSize : 0);
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];

    if (!pCreateInfo || !pCreateInfo->pCode || pCreateInfo->codeSize < 20) {
        return disp.CreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
    }

    const uint32_t* spirvCode = pCreateInfo->pCode;
    size_t numWords = pCreateInfo->codeSize / 4;

    // Scan for ray query or ray tracing ops for debugging
    bool hasRayQuery = spirvHasRayQuery(spirvCode, numWords);
    fprintf(stderr, "[CudaRT]   shader %zu words, hasRayQuery=%d\n", numWords, (int)hasRayQuery);

    // Try to rewrite ray query ops
    SpirvRewriteResult rw;
    static int noRewrite = -1;
    if (noRewrite < 0) {
        const char* e = getenv("CUDA_RT_NO_REWRITE");
        noRewrite = (e && atoi(e)) ? 1 : 0;
    }
    if (noRewrite) {
        rw.rewritten = false;
        fprintf(stderr, "[CudaRT] CUDA_RT_NO_REWRITE=1: skipping SPIR-V rewrite\n");
    } else {
        rw = spirvTryRewriteRayQuery(spirvCode, numWords);
    }

    if (rw.rewritten) {
        LOG("CreateShaderModule: REWRITTEN ray query shader (%zu→%zu words, BVH set=%d)",
            numWords, rw.code.size(), rw.bvhDescSet);

        // Dump rewritten SPIR-V for offline validation
        static int rqShaderIdx = 0;
        char dumpPath[256];
        snprintf(dumpPath, sizeof(dumpPath), "/tmp/rq_rewritten_%d.spv", rqShaderIdx);
        FILE* dumpFp = fopen(dumpPath, "wb");
        if (dumpFp) {
            fwrite(rw.code.data(), 4, rw.code.size(), dumpFp);
            fclose(dumpFp);
            LOG("  → Dumped rewritten SPIR-V to %s", dumpPath);
        }
        // Also dump original for comparison
        snprintf(dumpPath, sizeof(dumpPath), "/tmp/rq_original_%d.spv", rqShaderIdx);
        dumpFp = fopen(dumpPath, "wb");
        if (dumpFp) {
            fwrite(spirvCode, 4, numWords, dumpFp);
            fclose(dumpFp);
        }
        rqShaderIdx++;

        // Create shader module with rewritten SPIR-V
        VkShaderModuleCreateInfo modCI = *pCreateInfo;
        modCI.pCode = rw.code.data();
        modCI.codeSize = rw.code.size() * 4;

        VkResult res = disp.CreateShaderModule(device, &modCI, pAllocator, pShaderModule);
        if (res == VK_SUCCESS) {
            std::lock_guard<std::mutex> lock(g_rqShaderMutex);
            uint64_t handle = (uint64_t)*pShaderModule;
            g_rqShaders[handle] = {rw.bvhDescSet, rw.bvhNodesBinding, rw.bvhTrisBinding};
            LOG("  → Registered rewritten shader %lx (set=%d)", handle, rw.bvhDescSet);
        } else {
            LOG("  → Rewritten shader FAILED to compile (err=%d), falling back to original", res);
            res = disp.CreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
        }
        return res;
    }

    return disp.CreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
}

// ═══════════════════════════════════════════
// Intercepted: CreateAccelerationStructureKHR
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateAccelerationStructureKHR(
    VkDevice device,
    const VkAccelerationStructureCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkAccelerationStructureKHR* pAccelerationStructure)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];

    // Let the driver create the AS normally (it handles memory allocation)
    VkResult res = VK_SUCCESS;
    if (disp.CreateAccelerationStructureKHR) {
        res = disp.CreateAccelerationStructureKHR(device, pCreateInfo, pAllocator, pAccelerationStructure);
    } else {
        // Driver doesn't have RT — create a fake handle
        *pAccelerationStructure = (VkAccelerationStructureKHR)g_nextASHandle++;
        res = VK_SUCCESS;
    }

    if (res == VK_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_lock);
        g_bvhMap[(uint64_t)*pAccelerationStructure] = CudaBVHState{nullptr, 0, false};
        LOG("CreateAS: handle=0x%lx type=%s size=%zu",
            (uint64_t)*pAccelerationStructure,
            pCreateInfo->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR ? "TLAS" : "BLAS",
            (size_t)pCreateInfo->size);
    }
    return res;
}

static VKAPI_ATTR void VKAPI_CALL layer_DestroyAccelerationStructureKHR(
    VkDevice device,
    VkAccelerationStructureKHR as,
    const VkAllocationCallbacks* pAllocator)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];

    // Free CUDA BVH
    {
        std::lock_guard<std::mutex> lock(g_lock);
        auto bit = g_bvhMap.find((uint64_t)as);
        if (bit != g_bvhMap.end()) {
            if (bit->second.handle) {
                if (g_lastBLAS == bit->second.handle) g_lastBLAS = nullptr;
                cudaBVH_destroy(bit->second.handle);
            }
            g_bvhMap.erase(bit);
        }
    }

    if (disp.DestroyAccelerationStructureKHR)
        disp.DestroyAccelerationStructureKHR(device, as, pAllocator);

    LOG("DestroyAS: handle=0x%lx", (uint64_t)as);
}

// ═══════════════════════════════════════════
// Intercepted: GetAccelerationStructureBuildSizesKHR
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_GetAccelerationStructureBuildSizesKHR(
    VkDevice device,
    VkAccelerationStructureBuildTypeKHR buildType,
    const VkAccelerationStructureBuildGeometryInfoKHR* pBuildInfo,
    const uint32_t* pMaxPrimitiveCounts,
    VkAccelerationStructureBuildSizesInfoKHR* pSizeInfo)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];

    if (disp.GetAccelerationStructureBuildSizesKHR) {
        disp.GetAccelerationStructureBuildSizesKHR(device, buildType, pBuildInfo, pMaxPrimitiveCounts, pSizeInfo);
    } else {
        // Provide reasonable defaults if driver lacks RT
        uint32_t totalPrims = 0;
        for (uint32_t i = 0; i < pBuildInfo->geometryCount; i++)
            totalPrims += pMaxPrimitiveCounts[i];

        pSizeInfo->accelerationStructureSize = totalPrims * 128 + 4096;
        pSizeInfo->updateScratchSize = totalPrims * 64;
        pSizeInfo->buildScratchSize = totalPrims * 128;
    }

    LOG("GetASBuildSizes: geoCount=%u → structSize=%zu",
        pBuildInfo->geometryCount, (size_t)pSizeInfo->accelerationStructureSize);
}

// ═══════════════════════════════════════════
// Intercepted: CmdBuildAccelerationStructuresKHR
// DEFERRED: Store geometry metadata now, read vertex data after QueueSubmit
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_CmdBuildAccelerationStructuresKHR(
    VkCommandBuffer cmdBuf,
    uint32_t infoCount,
    const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
    const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos)
{
    // DEBUG: log every CmdBuildAS call at entry
    static int buildCallCount = 0;
    buildCallCount++;
    for (uint32_t dd = 0; dd < infoCount; dd++) {
        if (buildCallCount <= 50 || (buildCallCount % 100) == 0)
            LOG("CmdBuildAS ENTRY #%d: info[%u].type=%d geoCount=%u dst=0x%lx",
                buildCallCount, dd, (int)pInfos[dd].type, pInfos[dd].geometryCount,
                (uint64_t)pInfos[dd].dstAccelerationStructure);
    }

    static int tlasCount = 0;
    for (uint32_t i = 0; i < infoCount; i++) {
        auto& info = pInfos[i];
        bool isTLAS = (info.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);

        uint32_t totalPrims = 0;
        for (uint32_t g = 0; g < info.geometryCount; g++) {
            totalPrims += ppBuildRangeInfos[i][g].primitiveCount;
        }

        if (isTLAS) {
            tlasCount++;
            // Capture instance buffer for deferred read after QueueSubmit
            // IMPORTANT: Keep the TLAS with the MOST instances when multiple
            // TLASes are built per frame (Q2RTX builds 2: main scene + secondary)
            if (info.geometryCount > 0 &&
                info.pGeometries[0].geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
                uint32_t newCount = ppBuildRangeInfos[i][0].primitiveCount;
                VkDeviceAddress newAddr = info.pGeometries[0].geometry.instances.data.deviceAddress;
                bool arrayOfPtrs = info.pGeometries[0].geometry.instances.arrayOfPointers;
                // Only overwrite if this TLAS has MORE instances (prefer primary scene)
                if (newCount >= g_pendingTLAS.instanceCount || !g_pendingTLAS.pending) {
                    g_pendingTLAS.instanceAddr = newAddr;
                    g_pendingTLAS.instanceCount = newCount;
                    g_pendingTLAS.pending = true;
                }
                if (tlasCount <= 5 || (tlasCount % 100) == 0)
                    LOG("CmdBuildAS: TLAS #%d dst=0x%lx instances=%u addr=0x%lx arrayOfPtrs=%d → %s (pending=%u)",
                        tlasCount, (uint64_t)info.dstAccelerationStructure,
                        newCount, (uint64_t)newAddr, (int)arrayOfPtrs,
                        (newCount >= g_pendingTLAS.instanceCount || !g_pendingTLAS.pending) ? "ACCEPTED" : "SKIPPED(smaller)",
                        g_pendingTLAS.instanceCount);
            } else {
                if (tlasCount <= 3)
                    LOG("CmdBuildAS: TLAS #%d dst=0x%lx prims=%u (no instances geometry, skipped)",
                        tlasCount, (uint64_t)info.dstAccelerationStructure, totalPrims);
            }
            continue;
        }

        // For BLAS: store geometry metadata for deferred build after QueueSubmit
        if (info.type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR) {
            PendingBLAS pb;
            pb.asKey = (uint64_t)info.dstAccelerationStructure;

            for (uint32_t g = 0; g < info.geometryCount; g++) {
                auto& geo = info.pGeometries[g];
                if (geo.geometryType != VK_GEOMETRY_TYPE_TRIANGLES_KHR) continue;
                auto& triData = geo.geometry.triangles;
                uint32_t primCount = ppBuildRangeInfos[i][g].primitiveCount;

                PendingBLASGeo pg;
                pg.vertAddr = triData.vertexData.deviceAddress;
                pg.idxAddr = triData.indexData.deviceAddress;
                pg.vertDataSize = (triData.maxVertex + 1) * triData.vertexStride;
                pg.idxDataSize = 0;
                if (pg.idxAddr) {
                    uint32_t idxSize = (triData.indexType == VK_INDEX_TYPE_UINT16) ? 2 : 4;
                    pg.idxDataSize = primCount * 3 * idxSize;
                }
                pg.vertexStride = (uint32_t)triData.vertexStride;
                pg.vertexFormat = triData.vertexFormat;
                pg.indexType = triData.indexType;
                pg.primCount = primCount;
                pg.maxVertex = triData.maxVertex;
                pb.geometries.push_back(pg);
            }

            if (!pb.geometries.empty()) {
                std::lock_guard<std::mutex> lock(g_lock);
                size_t numGeos = pb.geometries.size();
                uint64_t asKey = pb.asKey;
                // Log vertex format/stride for first few BLASes
                static int fmtLog = 0;
                if (fmtLog < 3 && !pb.geometries.empty()) {
                    auto& g0 = pb.geometries[0];
                    LOG("  BLAS geo[0]: fmt=%u stride=%u idxType=%u maxVert=%u prims=%u",
                        g0.vertexFormat, g0.vertexStride, g0.indexType, g0.maxVertex, g0.primCount);
                    fmtLog++;
                }
                if (!g_blasBuildsDone) {
                    g_pendingBLAS.push_back(std::move(pb));
                    static int logCount = 0;
                    if (logCount < 5)
                        LOG("CmdBuildAS[%u]: BLAS dst=0x%lx geos=%zu prims=%u → DEFERRED",
                            i, asKey, numGeos, totalPrims);
                    logCount++;
                } else {
                    // Late BLAS — capture for incremental build
                    static int lateLog = 0;
                    if (lateLog < 5)
                        LOG("CmdBuildAS[%u]: LATE BLAS dst=0x%lx geos=%zu prims=%u → QUEUED",
                            i, asKey, numGeos, totalPrims);
                    lateLog++;
                    g_latePendingBLAS.push_back(std::move(pb));
                }
            }
        }
    }
    // Optionally pass through to driver (test if driver can handle AS build)
    static int passBuild = -1;
    if (passBuild < 0) {
        const char* env = getenv("CUDA_RT_PASS_BUILD");
        passBuild = (env && atoi(env)) ? 1 : 0;
    }
    if (passBuild) {
        void* key = getKey(cmdBuf);
        auto& disp = g_deviceMap[key];
        if (disp.CmdBuildAccelerationStructuresKHR) {
            static int pbLog = 0;
            if (pbLog++ < 3)
                LOG("CmdBuildAS: PASSING THROUGH to driver (CUDA_RT_PASS_BUILD=1)");
            disp.CmdBuildAccelerationStructuresKHR(cmdBuf, infoCount, pInfos, ppBuildRangeInfos);
        }
    }
}

// Process all pending BLAS builds — called from QueueSubmit after GPU has executed
static void processPendingBLAS() {
    std::vector<PendingBLAS> pending;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        if (g_pendingBLAS.empty() || g_blasBuildsDone) return;
        pending = std::move(g_pendingBLAS);
        g_pendingBLAS.clear();
    }

    LOG("[Deferred] Processing %zu pending BLAS builds after QueueSubmit...", pending.size());

    CudaBVH_t bestBVH = nullptr;
    int bestTris = 0;
    int totalBLAS = 0;

    for (auto& pb : pending) {
        std::vector<CudaTri> capturedTris;

        for (auto& pg : pb.geometries) {
            void* vertHost = readDeviceAddressData(pg.vertAddr, pg.vertDataSize);
            void* idxHost = nullptr;
            if (pg.idxAddr)
                idxHost = readDeviceAddressData(pg.idxAddr, pg.idxDataSize);

            if (!vertHost) {
                LOG("  → FAILED to read geometry (addr=0x%lx)", (uint64_t)pg.vertAddr);
                continue;
            }

            for (uint32_t p = 0; p < pg.primCount; p++) {
                uint32_t i0, i1, i2;
                if (idxHost && pg.indexType == VK_INDEX_TYPE_UINT16) {
                    uint16_t* idx16 = (uint16_t*)idxHost;
                    i0 = idx16[p*3]; i1 = idx16[p*3+1]; i2 = idx16[p*3+2];
                } else if (idxHost && pg.indexType == VK_INDEX_TYPE_UINT32) {
                    uint32_t* idx32 = (uint32_t*)idxHost;
                    i0 = idx32[p*3]; i1 = idx32[p*3+1]; i2 = idx32[p*3+2];
                } else {
                    i0 = p*3; i1 = p*3+1; i2 = p*3+2;
                }

                CudaTri tri;
                float* v0 = (float*)((char*)vertHost + i0 * pg.vertexStride);
                float* v1 = (float*)((char*)vertHost + i1 * pg.vertexStride);
                float* v2 = (float*)((char*)vertHost + i2 * pg.vertexStride);
                tri.v0[0]=v0[0]; tri.v0[1]=v0[1]; tri.v0[2]=v0[2];
                tri.v1[0]=v1[0]; tri.v1[1]=v1[1]; tri.v1[2]=v1[2];
                tri.v2[0]=v2[0]; tri.v2[1]=v2[1]; tri.v2[2]=v2[2];
                capturedTris.push_back(tri);
            }
            free(vertHost);
            free(idxHost);
        }

        if (!capturedTris.empty()) {
            // Validate first few tris of first 2 BLASes
            static int triDump = 0;
            if (triDump < 2) {
                int n = std::min((int)capturedTris.size(), 3);
                LOG("[BLAS-Val] asKey=0x%lx: %d tris, dumping %d:", (uint64_t)pb.asKey, (int)capturedTris.size(), n);
                for (int t = 0; t < n; t++) {
                    auto& tri = capturedTris[t];
                    LOG("  tri[%d]: (%.1f,%.1f,%.1f) (%.1f,%.1f,%.1f) (%.1f,%.1f,%.1f)",
                        t, tri.v0[0],tri.v0[1],tri.v0[2], tri.v1[0],tri.v1[1],tri.v1[2], tri.v2[0],tri.v2[1],tri.v2[2]);
                }
                triDump++;
            }
            CudaBVH_t bvh = cudaBVH_build(capturedTris.data(), (int)capturedTris.size());
            if (bvh) {
                std::lock_guard<std::mutex> lock(g_lock);
                auto bit = g_bvhMap.find(pb.asKey);
                if (bit != g_bvhMap.end()) {
                    if (bit->second.handle) cudaBVH_destroy(bit->second.handle);
                    bit->second.handle = bvh;
                    bit->second.numTris = (int)capturedTris.size();
                    bit->second.isReady = true;
                }
                // Track ALL BLASes in g_blasEntries
                BLASEntry entry = {};
                entry.bvh = bvh;
                entry.asKey = pb.asKey;
                entry.numTris = (int)capturedTris.size();
                g_asKeyToBLASIdx[pb.asKey] = (int)g_blasEntries.size();
                g_blasEntries.push_back(entry);
                totalBLAS++;

                if ((int)capturedTris.size() > bestTris) {
                    bestTris = (int)capturedTris.size();
                    bestBVH = bvh;
                }
            }
        }
    }

    // Compute concatenated node/tri offsets for all BLASes
    {
        std::lock_guard<std::mutex> lock(g_lock);
        int nodeOff = 0, triOff = 0;
        for (auto& be : g_blasEntries) {
            uint32_t* bvh2Data = nullptr;
            int numNodes = cudaBVH_getStacklessBVH2(be.bvh, &bvh2Data);
            int numTris = cudaBVH_getNumTris(be.bvh);
            int numTriVec4s = numTris * 3; // 3 vec4s per triangle

            be.nodeOffset = nodeOff;
            be.triOffset = triOff;
            be.numNodes = numNodes;
            be.numTriVec4s = numTriVec4s;

            // Extract BLAS bounds from BVH2 root node
            if (bvh2Data && numNodes > 0) {
                memcpy(&be.minX, bvh2Data + 0, 4);
                memcpy(&be.minY, bvh2Data + 1, 4);
                memcpy(&be.minZ, bvh2Data + 2, 4);
                memcpy(&be.maxX, bvh2Data + 3, 4);
                memcpy(&be.maxY, bvh2Data + 4, 4);
                memcpy(&be.maxZ, bvh2Data + 5, 4);
            }

            nodeOff += numNodes;
            triOff += numTris;
            LOG("  [BLAS#%d] asKey=0x%lx, %d tris, %d nodes, nodeOff=%d, triOff=%d",
                (int)(&be - g_blasEntries.data()), (uint64_t)be.asKey,
                numTris, numNodes, be.nodeOffset, be.triOffset);
        }
        // Map device addresses to BLAS entries
        for (auto& [devAddr, asHandle] : g_asDevAddrToHandle) {
            auto it = g_asKeyToBLASIdx.find(asHandle);
            if (it != g_asKeyToBLASIdx.end()) {
                g_blasDevAddrToIdx[devAddr] = it->second;
            }
        }
        LOG("[Multi-BLAS] %d BLASes, total %d nodes + %d tris",
            totalBLAS, nodeOff, triOff);
    }

    if (bestBVH) {
        g_lastBLAS = bestBVH;
        g_blasBuildsDone = true;
        LOG("[Deferred] Best BLAS: %d tris, %d BVH4 nodes — engine READY",
            cudaBVH_getNumTris(bestBVH), cudaBVH_getNumBVH4Nodes(bestBVH));
    }
}

// Process late BLAS builds (dynamic geometry) — called before TLAS rebuild
// Returns true if new BLASes were added (requires SSBO re-upload)
static bool processLateBLAS() {
    std::vector<PendingBLAS> late;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        if (g_latePendingBLAS.empty()) return false;
        late = std::move(g_latePendingBLAS);
        g_latePendingBLAS.clear();
    }

    static int lateCount = 0;
    int newBLAS = 0;

    for (auto& pb : late) {
        // Skip if we already have this BLAS
        {
            std::lock_guard<std::mutex> lock(g_lock);
            if (g_asKeyToBLASIdx.count(pb.asKey)) continue;
        }

        std::vector<CudaTri> capturedTris;
        for (auto& pg : pb.geometries) {
            void* vertHost = readDeviceAddressData(pg.vertAddr, pg.vertDataSize);
            void* idxHost = nullptr;
            if (pg.idxAddr)
                idxHost = readDeviceAddressData(pg.idxAddr, pg.idxDataSize);
            if (!vertHost) continue;

            for (uint32_t p = 0; p < pg.primCount; p++) {
                uint32_t i0, i1, i2;
                if (idxHost && pg.indexType == VK_INDEX_TYPE_UINT16) {
                    uint16_t* idx16 = (uint16_t*)idxHost;
                    i0 = idx16[p*3]; i1 = idx16[p*3+1]; i2 = idx16[p*3+2];
                } else if (idxHost && pg.indexType == VK_INDEX_TYPE_UINT32) {
                    uint32_t* idx32 = (uint32_t*)idxHost;
                    i0 = idx32[p*3]; i1 = idx32[p*3+1]; i2 = idx32[p*3+2];
                } else {
                    i0 = p*3; i1 = p*3+1; i2 = p*3+2;
                }
                CudaTri tri;
                float* v0 = (float*)((char*)vertHost + i0 * pg.vertexStride);
                float* v1 = (float*)((char*)vertHost + i1 * pg.vertexStride);
                float* v2 = (float*)((char*)vertHost + i2 * pg.vertexStride);
                tri.v0[0]=v0[0]; tri.v0[1]=v0[1]; tri.v0[2]=v0[2];
                tri.v1[0]=v1[0]; tri.v1[1]=v1[1]; tri.v1[2]=v1[2];
                tri.v2[0]=v2[0]; tri.v2[1]=v2[1]; tri.v2[2]=v2[2];
                capturedTris.push_back(tri);
            }
            free(vertHost);
            free(idxHost);
        }

        if (!capturedTris.empty()) {
            CudaBVH_t bvh = cudaBVH_build(capturedTris.data(), (int)capturedTris.size());
            if (bvh) {
                std::lock_guard<std::mutex> lock(g_lock);
                BLASEntry entry = {};
                entry.bvh = bvh;
                entry.asKey = pb.asKey;
                entry.numTris = (int)capturedTris.size();

                uint32_t* bvh2Data = nullptr;
                int numNodes = cudaBVH_getStacklessBVH2(bvh, &bvh2Data);
                int numTris = (int)capturedTris.size();

                // Compute offset from existing entries
                int nodeOff = 0, triOff = 0;
                for (auto& be : g_blasEntries) {
                    nodeOff += be.numNodes;
                    triOff += be.numTris;
                }
                entry.nodeOffset = nodeOff;
                entry.triOffset = triOff;
                entry.numNodes = numNodes;
                entry.numTriVec4s = numTris * 3;

                if (bvh2Data && numNodes > 0) {
                    memcpy(&entry.minX, bvh2Data + 0, 4);
                    memcpy(&entry.minY, bvh2Data + 1, 4);
                    memcpy(&entry.minZ, bvh2Data + 2, 4);
                    memcpy(&entry.maxX, bvh2Data + 3, 4);
                    memcpy(&entry.maxY, bvh2Data + 4, 4);
                    memcpy(&entry.maxZ, bvh2Data + 5, 4);
                }

                g_asKeyToBLASIdx[pb.asKey] = (int)g_blasEntries.size();
                g_blasEntries.push_back(entry);
                newBLAS++;

                // Update device address mapping
                for (auto& [devAddr, asHandle] : g_asDevAddrToHandle) {
                    if (asHandle == pb.asKey) {
                        g_blasDevAddrToIdx[devAddr] = (int)g_blasEntries.size() - 1;
                    }
                }
            }
        }
    }

    if (newBLAS > 0) {
        lateCount += newBLAS;
        LOG("[Late-BLAS] Built %d new BLASes (%d total late), %d BLASes now",
            newBLAS, lateCount, (int)g_blasEntries.size());
        return true;
    }
    return false;
}

// Helper: invert a 3×4 row-major affine matrix (rotation+translation)
// Input:  m[12] = {{r00,r01,r02,tx},{r10,r11,r12,ty},{r20,r21,r22,tz}}
// Output: inv[12] = inverse affine transform
static void invertAffine3x4(const float m[12], float inv[12]) {
    // The 3×3 rotation part
    float a=m[0],b=m[1],c=m[2],  tx=m[3];
    float d=m[4],e=m[5],f=m[6],  ty=m[7];
    float g=m[8],h=m[9],k=m[10], tz=m[11];

    float det = a*(e*k - f*h) - b*(d*k - f*g) + c*(d*h - e*g);
    if (fabsf(det) < 1e-20f) det = 1e-20f;
    float id = 1.0f / det;

    // Inverse of 3×3 rotation part
    float ia = (e*k - f*h)*id, ib = (c*h - b*k)*id, ic = (b*f - c*e)*id;
    float ie = (a*k - c*g)*id, ig = (c*d - a*f)*id;
    float ii = (a*e - b*d)*id;
    float ih = (b*g - a*h)*id;
    float iid = (f*g - d*k)*id;
    // Correct cofactor signs: row1 has alternating signs
    float r00 = ia, r01 = ib, r02 = ic;
    float r10 = iid, r11 = ie, r12 = ig;
    float r20 = ih, r21 = (g*b - a*h)*id, r22 = ii;
    // Actually let me be precise with cofactors
    r00 = (e*k - f*h)*id;
    r01 = (c*h - b*k)*id;
    r02 = (b*f - c*e)*id;
    r10 = (f*g - d*k)*id;
    r11 = (a*k - c*g)*id;
    r12 = (c*d - a*f)*id;
    r20 = (d*h - e*g)*id;
    r21 = (b*g - a*h)*id;
    r22 = (a*e - b*d)*id;

    // Inverse translation: -R^{-1} * t
    float itx = -(r00*tx + r01*ty + r02*tz);
    float ity = -(r10*tx + r11*ty + r12*tz);
    float itz = -(r20*tx + r21*ty + r22*tz);

    inv[0]=r00; inv[1]=r01; inv[2]=r02;  inv[3]=itx;
    inv[4]=r10; inv[5]=r11; inv[6]=r12;  inv[7]=ity;
    inv[8]=r20; inv[9]=r21; inv[10]=r22; inv[11]=itz;
}

// Transform an AABB through a 3×4 affine matrix, producing a new world-space AABB
static void transformAABB(const float m[12], float bminX, float bminY, float bminZ,
                          float bmaxX, float bmaxY, float bmaxZ,
                          float& outMinX, float& outMinY, float& outMinZ,
                          float& outMaxX, float& outMaxY, float& outMaxZ)
{
    // Arvo's method: transform AABB efficiently using matrix columns
    float tx = m[3], ty = m[7], tz = m[11];
    float aminX = tx, aminY = ty, aminZ = tz;
    float amaxX = tx, amaxY = ty, amaxZ = tz;

    float bmin[3] = {bminX, bminY, bminZ};
    float bmax[3] = {bmaxX, bmaxY, bmaxZ};

    for (int j = 0; j < 3; j++) {
        float a0 = m[0*4+j] * bmin[j], b0 = m[0*4+j] * bmax[j];
        float a1 = m[1*4+j] * bmin[j], b1 = m[1*4+j] * bmax[j];
        float a2 = m[2*4+j] * bmin[j], b2 = m[2*4+j] * bmax[j];
        aminX += fminf(a0,b0); amaxX += fmaxf(a0,b0);
        aminY += fminf(a1,b1); amaxY += fmaxf(a1,b1);
        aminZ += fminf(a2,b2); amaxZ += fmaxf(a2,b2);
    }
    outMinX = aminX; outMinY = aminY; outMinZ = aminZ;
    outMaxX = amaxX; outMaxY = amaxY; outMaxZ = amaxZ;
}

// Process pending TLAS builds — called from QueueSubmit after BLAS is ready
// Throttle is handled by QueueSubmit; this function always does the actual rebuild.
static uint64_t g_tlasGeneration = 0;  // incremented each TLAS rebuild for per-frame updates
static void processPendingTLAS() {
    if (!g_pendingTLAS.pending || !g_lastBLAS) return;
    g_pendingTLAS.pending = false;

    uint32_t numInst = g_pendingTLAS.instanceCount;
    VkDeviceAddress addr = g_pendingTLAS.instanceAddr;
    if (numInst == 0 || addr == 0) return;

    auto t0 = std::chrono::steady_clock::now();

    static int tlasRebuildCount = 0;
    bool verbose = (tlasRebuildCount < 3);

    if (verbose) {
        LOG("[TLAS] Reading %u instances from addr=0x%lx...", numInst, (uint64_t)addr);
        // Dump all known buffers that overlap this address
        std::lock_guard<std::mutex> lock(g_lock);
        int found = 0;
        for (auto& [bufId, bi] : g_buffers) {
            if (bi.devAddr && addr >= bi.devAddr && addr < bi.devAddr + bi.size) {
                auto mit = g_memories.find((uint64_t)bi.memory);
                bool hasHost = (mit != g_memories.end() && mit->second.hostPtr);
                LOG("[TLAS]   → Buffer hit: buf=0x%lx devAddr=0x%lx size=%zu off=%zu hostPtr=%s mem=0x%lx memOff=%zu",
                    bufId, (uint64_t)bi.devAddr, (size_t)bi.size,
                    (size_t)(addr - bi.devAddr), hasHost ? "YES" : "NO",
                    (uint64_t)bi.memory, (size_t)bi.memOffset);
                found++;
            }
        }
        if (!found) LOG("[TLAS]   → NO buffer found for addr=0x%lx!", (uint64_t)addr);
    }

    // VkAccelerationStructureInstanceKHR is 64 bytes each
    VkDeviceSize dataSize = (VkDeviceSize)numInst * 64;

    // Debug: try both paths explicitly to diagnose which is taken
    if (verbose) {
        void* hostPtr = resolveDeviceAddress(addr);
        if (hostPtr) {
            const uint32_t* u = (const uint32_t*)hostPtr;
            int nz = 0;
            for (int i = 0; i < 16; i++) if (u[i]) nz++;
            LOG("[TLAS] resolveDeviceAddress → hostPtr=%p nonZero=%d/16 first4=[%08x %08x %08x %08x]",
                hostPtr, nz, u[0], u[1], u[2], u[3]);
        } else {
            LOG("[TLAS] resolveDeviceAddress → NULL (DEVICE_LOCAL, will use staging copy)");
        }
    }

    // Instance data is in DEVICE_LOCAL memory — staging copy always returns zeros.
    // Use fd-based CUDA import (fast path after first successful import).
    static bool useFdImport = false;  // once fd-import works, skip staging entirely
    void* rawData = nullptr;
    
    if (!useFdImport) {
        // First attempt: try staging copy
        rawData = readDeviceAddressData(addr, dataSize);
        if (rawData) {
            const uint32_t* check = (const uint32_t*)rawData;
            int nz = 0;
            for (int i = 0; i < 64; i++) if (check[i]) nz++;
            if (nz > 0) {
                // Staging copy worked! Use it.
                goto have_data;
            }
            // Staging copy returned zeros — try fd-import
        }
    }
    
    // Fd-import path: export Vulkan memory to CUDA via file descriptor
    {
        if (verbose) LOG("[TLAS] Using fd-based CUDA import for instance data...");
        VkDeviceMemory instMem = VK_NULL_HANDLE;
        VkDeviceSize instMemOff = 0;
        VkDeviceSize instBufOff = 0;
        VkDevice instDevice = VK_NULL_HANDLE;
        VkDeviceSize instAllocSize = 0;
        {
            std::lock_guard<std::mutex> lock(g_lock);
            for (auto& [bufId, bi] : g_buffers) {
                if (bi.devAddr && addr >= bi.devAddr && addr < bi.devAddr + bi.size) {
                    instMem = bi.memory;
                    instMemOff = bi.memOffset;
                    instBufOff = addr - bi.devAddr;
                    auto mit = g_memories.find((uint64_t)bi.memory);
                    if (mit != g_memories.end()) {
                        instDevice = mit->second.device;
                        instAllocSize = mit->second.allocSize;
                    }
                    break;
                }
            }
        }
        if (instDevice && instMem) {
            void* dkey = getKey(instDevice);
            auto dit = g_deviceMap.find(dkey);
            if (dit != g_deviceMap.end() && dit->second.GetMemoryFdKHR) {
                VkMemoryGetFdInfoKHR fdInfo = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
                fdInfo.memory = instMem;
                fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
                int fd = -1;
                VkResult fdRes = dit->second.GetMemoryFdKHR(instDevice, &fdInfo, &fd);
                if (fdRes == VK_SUCCESS && fd >= 0) {
                    cudaExternalMemory_t extMem = nullptr;
                    cudaExternalMemoryHandleDesc extDesc = {};
                    extDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
                    extDesc.handle.fd = fd;
                    extDesc.size = instAllocSize > 0 ? instAllocSize : dataSize + instMemOff + instBufOff;
                    
                    if (cudaImportExternalMemory(&extMem, &extDesc) == cudaSuccess && extMem) {
                        cudaExternalMemoryBufferDesc fullBufDesc = {};
                        fullBufDesc.offset = instMemOff + instBufOff;
                        fullBufDesc.size = dataSize;
                        void* devPtr = nullptr;
                        if (cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &fullBufDesc) == cudaSuccess && devPtr) {
                            if (rawData) free(rawData);
                            rawData = malloc(dataSize);
                            cudaMemcpy(rawData, devPtr, dataSize, cudaMemcpyDeviceToHost);
                            cudaFree(devPtr);
                            useFdImport = true;  // cache: always use fd-import from now on
                            if (verbose) LOG("[TLAS] CUDA fd-import: %zu bytes read successfully", (size_t)dataSize);
                        }
                        cudaDestroyExternalMemory(extMem);
                    } else {
                        close(fd);
                    }
                }
            }
        }
    }

have_data:
    if (!rawData) {
        LOG("[TLAS] FAILED to read instance buffer!");
        return;
    }
    auto t1 = std::chrono::steady_clock::now();
    // Quick validation of read data
    if (verbose) {
        const uint32_t* u = (const uint32_t*)rawData;
        int totalNZ = 0;
        for (int i = 0; i < 256; i++) if (u[i]) totalNZ++;
        LOG("[TLAS] readDeviceAddressData: %zu bytes, sample nonZero=%d/256",
            (size_t)dataSize, totalNZ);
    }

    // Quick degenerate check before full parse — sample first 1000 entries
    if (verbose) {
        const uint8_t* p = (const uint8_t*)rawData;
        int identical = 0;
        for (uint32_t i = 1; i < numInst && i < 1000; i++) {
            if (memcmp(p, p + i*64, 64) == 0) identical++;
        }
        LOG("[TLAS] %u instances, %d/%u identical in first 1000",
            numInst, identical, std::min(numInst, 1000u));
    }

    // ── MULTI-BLAS RESOLUTION ──
    // Count unique BLAS references for logging, and find a fallback BLAS
    int fallbackBlasIdx = -1;
    {
        const uint8_t* p = (const uint8_t*)rawData;
        std::unordered_map<uint64_t, int> blasRefCounts;
        for (uint32_t i = 0; i < numInst; i++) {
            uint64_t ref = 0;
            memcpy(&ref, p + i*64 + 56, 8);
            if (ref) blasRefCounts[ref]++;
        }
        // Find most-referenced MAPPED BLAS as fallback for unmapped instances
        uint64_t bestRef = 0;
        int bestCount = 0;
        int bestMappedIdx = -1;
        int bestMappedTris = 0;
        for (auto& [ref, cnt] : blasRefCounts) {
            if (cnt > bestCount) { bestCount = cnt; bestRef = ref; }
            auto it = g_blasDevAddrToIdx.find(ref);
            if (it != g_blasDevAddrToIdx.end()) {
                int idx = it->second;
                int tris = (idx < (int)g_blasEntries.size()) ? g_blasEntries[idx].numTris : 0;
                // Prefer the largest mapped BLAS (world geometry) as fallback
                if (tris > bestMappedTris) {
                    bestMappedTris = tris;
                    bestMappedIdx = idx;
                }
            }
        }
        if (bestRef) {
            auto it = g_blasDevAddrToIdx.find(bestRef);
            if (it != g_blasDevAddrToIdx.end()) {
                fallbackBlasIdx = it->second;
            }
        }
        // If most-referenced BLAS is unmapped, fall back to largest mapped BLAS
        if (fallbackBlasIdx < 0 && bestMappedIdx >= 0) {
            fallbackBlasIdx = bestMappedIdx;
            if (verbose)
                LOG("[TLAS] Using largest mapped BLAS#%d (%d tris) as fallback", bestMappedIdx, bestMappedTris);
        }
        if (verbose) {
            LOG("[TLAS] %zu unique BLAS refs, fallbackIdx=%d (%d BLASes available, %zu tracked devAddrs)",
                blasRefCounts.size(), fallbackBlasIdx, (int)g_blasEntries.size(), g_blasDevAddrToIdx.size());
            for (auto& [ref, cnt] : blasRefCounts) {
                auto it = g_blasDevAddrToIdx.find(ref);
                int idx = (it != g_blasDevAddrToIdx.end()) ? it->second : -1;
                if (idx >= 0)
                    LOG("  blasRef=0x%lx → BLAS#%d (%d tris), %d instances",
                        ref, idx, g_blasEntries[idx].numTris, cnt);
                else {
                    LOG("  blasRef=0x%lx → UNMAPPED, %d instances", ref, cnt);
                    // Dump all tracked addresses for debugging
                    LOG("  [DEBUG] All %zu tracked BLAS devAddrs:", g_blasDevAddrToIdx.size());
                    int dumpCount = 0;
                    for (auto& [da, bi] : g_blasDevAddrToIdx) {
                        LOG("    devAddr=0x%lx → BLAS#%d", da, bi);
                        if (++dumpCount >= 10) { LOG("    ... (%zu more)", g_blasDevAddrToIdx.size() - 10); break; }
                    }
                }
            }
        }
    }

    // Parse instances and build InstanceGPU array + world AABBs for TLAS BVH
    g_instances.clear();
    g_instances.resize(numInst);

    // AABBs for TLAS BVH builder: 6 floats per instance (minX,minY,minZ,maxX,maxY,maxZ)
    std::vector<float> worldAABBs(numInst * 6);
    int maskCounts[256] = {0};
    int mappedCount = 0, unmappedCount = 0;

    const uint8_t* src = (const uint8_t*)rawData;
    for (uint32_t i = 0; i < numInst; i++) {
        const uint8_t* inst = src + i * 64;

        // VkAccelerationStructureInstanceKHR layout:
        // offset 0:  VkTransformMatrixKHR (float[3][4], 48 bytes, ROW-MAJOR)
        // offset 48: instanceCustomIndex:24 + mask:8
        // offset 52: instanceShaderBindingTableRecordOffset:24 + flags:8
        // offset 56: accelerationStructureReference (uint64_t)
        float xform[12];
        memcpy(xform, inst, 48);
        
        // Extract instance mask (top 8 bits of uint32 at offset 48)
        uint32_t customIdxMask;
        memcpy(&customIdxMask, inst + 48, 4);
        uint8_t instanceMask = (customIdxMask >> 24) & 0xFF;
        uint32_t customIdx = customIdxMask & 0x00FFFFFF;
        maskCounts[instanceMask]++;

        // Extract SBT offset and flags (offset 52)
        uint32_t sbtFlags;
        memcpy(&sbtFlags, inst + 52, 4);
        uint32_t sbtOffset = sbtFlags & 0x00FFFFFF;
        uint32_t instFlags = (sbtFlags >> 24) & 0xFF;

        // Resolve per-instance BLAS
        uint64_t blasRef = 0;
        memcpy(&blasRef, inst + 56, 8);
        int blasIdx = fallbackBlasIdx;
        if (blasRef) {
            auto it = g_blasDevAddrToIdx.find(blasRef);
            if (it != g_blasDevAddrToIdx.end()) {
                blasIdx = it->second;
                mappedCount++;
            } else {
                unmappedCount++;
            }
        }

        // Get per-instance BLAS bounds and offsets
        float bMinX = 0, bMinY = 0, bMinZ = 0, bMaxX = 0, bMaxY = 0, bMaxZ = 0;
        uint32_t nodeOff = 0, triOff = 0;
        if (blasIdx >= 0 && blasIdx < (int)g_blasEntries.size()) {
            const BLASEntry& be = g_blasEntries[blasIdx];
            bMinX = be.minX; bMinY = be.minY; bMinZ = be.minZ;
            bMaxX = be.maxX; bMaxY = be.maxY; bMaxZ = be.maxZ;
            nodeOff = be.nodeOffset;
            triOff = be.triOffset;
        }

        // Compute inverse transform
        float invXform[12];
        invertAffine3x4(xform, invXform);

        // Compute world-space AABB by transforming BLAS bounds through instance transform
        float wMinX, wMinY, wMinZ, wMaxX, wMaxY, wMaxZ;
        transformAABB(xform, bMinX, bMinY, bMinZ, bMaxX, bMaxY, bMaxZ,
                      wMinX, wMinY, wMinZ, wMaxX, wMaxY, wMaxZ);

        // Fill InstanceGPU
        InstanceGPU& gi = g_instances[i];
        memcpy(gi.transform, xform, 48);
        memcpy(gi.invTransform, invXform, 48);
        gi.blasMinX = bMinX; gi.blasMinY = bMinY; gi.blasMinZ = bMinZ;
        gi.blasMaxX = bMaxX; gi.blasMaxY = bMaxY; gi.blasMaxZ = bMaxZ;
        gi.blasNodeOff = nodeOff;
        gi.blasTriOff = triOff;
        gi.customIdx = customIdx;
        gi.sbtOffset = sbtOffset;
        gi.instanceMask = instanceMask;
        gi.instanceFlags = instFlags;

        // Store world AABB for TLAS BVH
        worldAABBs[i*6+0] = wMinX; worldAABBs[i*6+1] = wMinY; worldAABBs[i*6+2] = wMinZ;
        worldAABBs[i*6+3] = wMaxX; worldAABBs[i*6+4] = wMaxY; worldAABBs[i*6+5] = wMaxZ;

        // Log ALL instances on first rebuild
        if (verbose) {
            LOG("[TLAS] Instance %u: T=[%.1f,%.1f,%.1f] mask=0x%02x customIdx=%u sbt=%u flags=%u BLAS#%d (nOff=%u,tOff=%u) wAABB=(%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f)",
                i, xform[3], xform[7], xform[11], instanceMask, customIdx, sbtOffset, instFlags,
                blasIdx, nodeOff, triOff,
                wMinX, wMinY, wMinZ, wMaxX, wMaxY, wMaxZ);
        }
    }
    free(rawData);

    if (verbose)
        LOG("[TLAS] Multi-BLAS: %d mapped, %d unmapped (fallback=%d)",
            mappedCount, unmappedCount, fallbackBlasIdx);

    // Log mask distribution
    if (verbose) {
        for (int m = 0; m < 256; m++) {
            if (maskCounts[m] > 0)
                LOG("[TLAS] mask=0x%02x: %d instances", m, maskCounts[m]);
        }
    }

    // Compute actual world bounds from all instances
    float totalMinX=1e30f, totalMinY=1e30f, totalMinZ=1e30f;
    float totalMaxX=-1e30f, totalMaxY=-1e30f, totalMaxZ=-1e30f;
    int uniqueCount = 0;
    float firstTx = worldAABBs[0*6+0], firstTy = worldAABBs[0*6+1];
    for (uint32_t i = 0; i < numInst; i++) {
        float mx = worldAABBs[i*6+0], my = worldAABBs[i*6+1], mz = worldAABBs[i*6+2];
        float Mx = worldAABBs[i*6+3], My = worldAABBs[i*6+4], Mz = worldAABBs[i*6+5];
        if (mx < totalMinX) totalMinX = mx;
        if (my < totalMinY) totalMinY = my;
        if (mz < totalMinZ) totalMinZ = mz;
        if (Mx > totalMaxX) totalMaxX = Mx;
        if (My > totalMaxY) totalMaxY = My;
        if (Mz > totalMaxZ) totalMaxZ = Mz;
        if (fabsf(mx - firstTx) > 0.1f || fabsf(my - firstTy) > 0.1f) uniqueCount++;
    }

    float worldExtent = fmaxf(fmaxf(totalMaxX-totalMinX, totalMaxY-totalMinY), totalMaxZ-totalMinZ);
    float uniquePct = 100.0f * uniqueCount / numInst;

    if (verbose) {
        LOG("[TLAS] World bounds: (%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f) extent=%.1f",
            totalMinX, totalMinY, totalMinZ, totalMaxX, totalMaxY, totalMaxZ, worldExtent);
        LOG("[TLAS] Unique positions: %d / %u (%.1f%%)", uniqueCount, numInst, uniquePct);
    }

    // TIMING BUG FIX: Skip degenerate instance data (all at same position).
    // GravityMark initializes all 200K instances to the same transform, then
    // scatters them via compute shader. The first CmdBuild captures pre-scatter data.
    // If <1% unique, skip this build — g_tlasGeneration stays at 0 so next frame retries.
    if (uniqueCount < (int)(numInst * 0.01f) + 1 && numInst > 100) {
        static int skipCount = 0;
        if (skipCount < 5)
            LOG("[TLAS] SKIP: degenerate data (%d/%u unique, %.1f%%) — waiting for scatter (skip #%d)",
                uniqueCount, numInst, uniquePct, ++skipCount);
        else
            skipCount++;
        return;  // don't build, don't increment generation — retry next frame
    }

    if (verbose)
        LOG("[TLAS] Parsed %u instances (%.0f%% unique), building TLAS BVH...", numInst, uniquePct);

    auto t2 = std::chrono::steady_clock::now();

    // Build TLAS BVH from world-space AABBs — fast LBVH only (SAH too slow for 200K)
    // Set g_tlasBVH sentinel to mark "first build done" without the slow SAH path
    if (g_fastTLASNodes) { free(g_fastTLASNodes); g_fastTLASNodes = nullptr; }
    if (g_fastTLASOrdered) { free(g_fastTLASOrdered); g_fastTLASOrdered = nullptr; }
    g_fastTLASNodeCount = cudaBVH_buildTLASFast(worldAABBs.data(), (int)numInst,
                                                 &g_fastTLASNodes, &g_fastTLASOrdered);
    if (g_fastTLASNodeCount > 0) {
        if (!g_tlasBVH) {
            // Create a dummy SAH handle so reupload trigger works
            g_tlasBVH = cudaBVH_buildFromAABBs(worldAABBs.data(), std::min((int)numInst, 8));
        }
        g_tlasGeneration++;
        auto t3 = std::chrono::steady_clock::now();
        auto ms_read = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        auto ms_parse = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        auto ms_build = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        if (g_tlasGeneration <= 3 || (g_tlasGeneration % 100 == 0))
            LOG("[TLAS] Fast TLAS BVH2: %d nodes over %u instances (gen=%lu) [read=%ldms parse=%ldms build=%ldms]",
                g_fastTLASNodeCount, numInst, g_tlasGeneration, ms_read, ms_parse, ms_build);
    } else {
        LOG("[TLAS] FAILED fast TLAS build!");
    }
    tlasRebuildCount++;
}

// ═══════════════════════════════════════════
// Vulkan compute pipeline: BVH4 tracing with zero CUDA overhead
// ═══════════════════════════════════════════

// Helper: create a DEVICE_LOCAL buffer, stage data via HOST_VISIBLE + one-shot cmdBuf copy
// createBufferHostVisible: Use HOST_VISIBLE+HOST_COHERENT memory with direct map/memcpy.
// Slower than DEVICE_LOCAL but guaranteed to be accessible by GPU immediately.
// Use for debugging SSBO visibility issues.
static bool createBufferHostVisible(DeviceDispatch& disp, const void* data, VkDeviceSize size,
                                     VkBufferUsageFlags usage, VkBuffer& outBuf, VkDeviceMemory& outMem)
{
    VkBufferCreateInfo bufCI = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufCI.size = size;
    bufCI.usage = usage;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (disp.CreateBuffer(disp.device, &bufCI, nullptr, &outBuf) != VK_SUCCESS) return false;

    VkMemoryRequirements memReqs;
    disp.GetBufferMemoryRequirements(disp.device, outBuf, &memReqs);

    int memType = -1;
    for (uint32_t i = 0; i < disp.memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (disp.memProps.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memType = i; break;
        }
    }
    if (memType < 0) { disp.DestroyBuffer(disp.device, outBuf, nullptr); return false; }

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memType;
    if (disp.AllocateMemory(disp.device, &allocInfo, nullptr, &outMem) != VK_SUCCESS) {
        disp.DestroyBuffer(disp.device, outBuf, nullptr); return false;
    }
    disp.BindBufferMemory(disp.device, outBuf, outMem, 0);

    void* mapped = nullptr;
    disp.MapMemory(disp.device, outMem, 0, size, 0, &mapped);
    if (!mapped) return false;
    memcpy(mapped, data, size);
    disp.UnmapMemory(disp.device, outMem);
    return true;
}

static bool createBufferWithData(DeviceDispatch& disp, const void* data, VkDeviceSize size,
                                  VkBufferUsageFlags usage, VkBuffer& outBuf, VkDeviceMemory& outMem)
{
    // 1. Create DEVICE_LOCAL target buffer
    VkBufferCreateInfo bufCI = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufCI.size = size;
    bufCI.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (disp.CreateBuffer(disp.device, &bufCI, nullptr, &outBuf) != VK_SUCCESS) return false;

    VkMemoryRequirements memReqs;
    disp.GetBufferMemoryRequirements(disp.device, outBuf, &memReqs);

    // Find DEVICE_LOCAL memory type
    int devMemType = -1;
    for (uint32_t i = 0; i < disp.memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (disp.memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            devMemType = i; break;
        }
    }
    if (devMemType < 0) {
        disp.DestroyBuffer(disp.device, outBuf, nullptr);
        return false;
    }

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = devMemType;
    if (disp.AllocateMemory(disp.device, &allocInfo, nullptr, &outMem) != VK_SUCCESS) {
        disp.DestroyBuffer(disp.device, outBuf, nullptr);
        return false;
    }
    disp.BindBufferMemory(disp.device, outBuf, outMem, 0);

    // 2. Create HOST_VISIBLE staging buffer
    VkBuffer stagingBuf;
    VkDeviceMemory stagingMem;
    VkBufferCreateInfo stagCI = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    stagCI.size = size;
    stagCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (disp.CreateBuffer(disp.device, &stagCI, nullptr, &stagingBuf) != VK_SUCCESS) return false;

    VkMemoryRequirements stagReqs;
    disp.GetBufferMemoryRequirements(disp.device, stagingBuf, &stagReqs);
    int hostMemType = -1;
    for (uint32_t i = 0; i < disp.memProps.memoryTypeCount; i++) {
        if ((stagReqs.memoryTypeBits & (1 << i)) &&
            (disp.memProps.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            hostMemType = i; break;
        }
    }
    if (hostMemType < 0) {
        disp.DestroyBuffer(disp.device, stagingBuf, nullptr);
        return false;
    }
    VkMemoryAllocateInfo stagAlloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    stagAlloc.allocationSize = stagReqs.size;
    stagAlloc.memoryTypeIndex = hostMemType;
    if (disp.AllocateMemory(disp.device, &stagAlloc, nullptr, &stagingMem) != VK_SUCCESS) {
        disp.DestroyBuffer(disp.device, stagingBuf, nullptr);
        return false;
    }
    disp.BindBufferMemory(disp.device, stagingBuf, stagingMem, 0);
    void* mapped = nullptr;
    disp.MapMemory(disp.device, stagingMem, 0, size, 0, &mapped);
    if (!mapped) return false;
    memcpy(mapped, data, size);
    disp.UnmapMemory(disp.device, stagingMem);

    // 3. One-shot command buffer to copy staging → device local
    auto createCP = (PFN_vkCreateCommandPool)disp.GetDeviceProcAddr(disp.device, "vkCreateCommandPool");
    auto destroyCP = (PFN_vkDestroyCommandPool)disp.GetDeviceProcAddr(disp.device, "vkDestroyCommandPool");
    auto allocCB = (PFN_vkAllocateCommandBuffers)disp.GetDeviceProcAddr(disp.device, "vkAllocateCommandBuffers");
    auto beginCB = (PFN_vkBeginCommandBuffer)disp.GetDeviceProcAddr(disp.device, "vkBeginCommandBuffer");
    auto endCB = (PFN_vkEndCommandBuffer)disp.GetDeviceProcAddr(disp.device, "vkEndCommandBuffer");
    auto queueSubmit = (PFN_vkQueueSubmit)disp.GetDeviceProcAddr(disp.device, "vkQueueSubmit");
    auto queueWait = (PFN_vkQueueWaitIdle)disp.GetDeviceProcAddr(disp.device, "vkQueueWaitIdle");
    auto getQueue = (PFN_vkGetDeviceQueue)disp.GetDeviceProcAddr(disp.device, "vkGetDeviceQueue");

    if (!createCP || !allocCB || !beginCB || !endCB || !queueSubmit || !queueWait || !getQueue) {
        disp.DestroyBuffer(disp.device, stagingBuf, nullptr);
        disp.FreeMemory(disp.device, stagingMem, nullptr);
        return false;
    }

    VkQueue queue;
    getQueue(disp.device, 0, 0, &queue);  // Use queue family 0, queue 0

    VkCommandPool cmdPool;
    VkCommandPoolCreateInfo cpCI = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpCI.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpCI.queueFamilyIndex = 0;
    createCP(disp.device, &cpCI, nullptr, &cmdPool);

    VkCommandBuffer cmdBuf;
    VkCommandBufferAllocateInfo cbAI = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbAI.commandPool = cmdPool;
    cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = 1;
    allocCB(disp.device, &cbAI, &cmdBuf);

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    beginCB(cmdBuf, &beginInfo);
    VkBufferCopy copyRegion = {};
    copyRegion.size = size;
    disp.CmdCopyBuffer(cmdBuf, stagingBuf, outBuf, 1, &copyRegion);
    endCB(cmdBuf);

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;
    queueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    queueWait(queue);

    // Cleanup staging
    destroyCP(disp.device, cmdPool, nullptr);
    disp.DestroyBuffer(disp.device, stagingBuf, nullptr);
    disp.FreeMemory(disp.device, stagingMem, nullptr);
    return true;
}

// ═══════════════════════════════════════════
// BVH2 interop: create Vulkan SSBOs + descriptor set for SPIR-V ray query traversal
// ═══════════════════════════════════════════
static bool setupBVH2Descriptors(DeviceDispatch& disp) {
    if (g_bvh2.dsLayout) return true; // already set up

    // Descriptor set layout: 4 SSBOs
    // binding 0 = BLAS nodes, binding 1 = BLAS tris,
    // binding 2 = TLAS nodes, binding 3 = instances
    VkDescriptorSetLayoutBinding bindings[4] = {};
    for (int i = 0; i < 4; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dsLayoutCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsLayoutCI.bindingCount = 4;
    dsLayoutCI.pBindings = bindings;
    if (disp.CreateDescriptorSetLayout(disp.device, &dsLayoutCI, nullptr, &g_bvh2.dsLayout) != VK_SUCCESS) {
        LOG("[BVH2] Failed to create descriptor set layout");
        return false;
    }

    // Descriptor pool: 4 SSBO descriptors
    VkDescriptorPoolSize poolSizes[1] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 4;
    VkDescriptorPoolCreateInfo poolCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.maxSets = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes = poolSizes;
    if (disp.CreateDescriptorPool(disp.device, &poolCI, nullptr, &g_bvh2.descPool) != VK_SUCCESS) {
        LOG("[BVH2] Failed to create descriptor pool");
        return false;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo dsAI = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsAI.descriptorPool = g_bvh2.descPool;
    dsAI.descriptorSetCount = 1;
    dsAI.pSetLayouts = &g_bvh2.dsLayout;
    if (disp.AllocateDescriptorSets(disp.device, &dsAI, &g_bvh2.descSet) != VK_SUCCESS) {
        LOG("[BVH2] Failed to allocate descriptor set");
        return false;
    }

    LOG("[BVH2] Descriptor set layout + pool + set created (4 bindings)");

    // Create null-safe BVH buffers immediately so shaders that run before real BVH data
    // is uploaded will read valid descriptors. Use large AABB extents so no ray hits the
    // degenerate boxes, and skip=-1 for immediate loop exit on the next iteration.
    {
        // 4 null nodes: AABB = (FLT_MAX..FLT_MAX) to (FLT_MAX..FLT_MAX) → never hit
        // leaf_enc=0 (internal), skip=-1 (exit). Extra nodes prevent OOB access.
        uint32_t hugeF;
        float hugeVal = 1e30f;
        memcpy(&hugeF, &hugeVal, 4);
        uint32_t nullNodes[4*8];  // 4 nodes × 8 uint32s each
        for (int i = 0; i < 4; i++) {
            uint32_t* n = nullNodes + i*8;
            n[0]=hugeF; n[1]=hugeF; n[2]=hugeF; // bmin = huge (unreachable)
            n[3]=hugeF; n[4]=hugeF; n[5]=hugeF; // bmax = huge
            n[6]=0;                              // leaf_enc=0 (internal)
            n[7]=0xFFFFFFFF;                     // skip=-1 (exit loop)
        }
        // binding 0: BLAS nodes
        if (!createBufferWithData(disp, nullNodes, sizeof(nullNodes),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, g_bvh2.nodesBuf, g_bvh2.nodesMem)) {
            LOG("[BVH2] Failed to create null BLAS nodes buffer");
            return false;
        }
        // binding 1: BLAS tris — 1 dummy triangle (3 vec4s)
        float nullTri[12] = {0};
        if (!createBufferWithData(disp, nullTri, sizeof(nullTri),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, g_bvh2.trisBuf, g_bvh2.trisMem)) {
            LOG("[BVH2] Failed to create null BLAS tris buffer");
            return false;
        }
        // binding 2: TLAS nodes — 4 nodes with huge AABB + skip=-1
        if (!createBufferWithData(disp, nullNodes, sizeof(nullNodes),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, g_bvh2.tlasNodesBuf, g_bvh2.tlasNodesMem)) {
            LOG("[BVH2] Failed to create null TLAS nodes buffer");
            return false;
        }
        // binding 3: instances — 1 dummy instance (8 vec4s = 128 bytes)
        float nullInst[32] = {0};
        if (!createBufferWithData(disp, nullInst, sizeof(nullInst),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, g_bvh2.instancesBuf, g_bvh2.instancesMem)) {
            LOG("[BVH2] Failed to create null instances buffer");
            return false;
        }
        // Write null descriptors
        VkDescriptorBufferInfo bufs[4] = {
            {g_bvh2.nodesBuf,       0, sizeof(nullNodes)},
            {g_bvh2.trisBuf,        0, sizeof(nullTri)},
            {g_bvh2.tlasNodesBuf,   0, sizeof(nullNodes)},
            {g_bvh2.instancesBuf,   0, sizeof(nullInst)},
        };
        VkWriteDescriptorSet writes[4] = {};
        for (int i = 0; i < 4; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = g_bvh2.descSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufs[i];
        }
        disp.UpdateDescriptorSets(disp.device, 4, writes, 0, nullptr);
        g_bvh2.descSetIdx = 4; // initial default — overridden by per-pipeline bvhDescSet at bind time
        // NOTE: g_bvh2.ready remains false until uploadBVH2Data uploads real data
        LOG("[BVH2] Null-safe BVH buffers initialized (huge AABB + skip=-1)");
    }

    return true;
}

static bool uploadBVH2Data(DeviceDispatch& disp, CudaBVH_t bvh) {
    if (!bvh) return false;
    if (!setupBVH2Descriptors(disp)) return false;

    // ── MULTI-BLAS: Concatenate ALL BLASes' BVH2 data into single buffers ──
    std::vector<uint32_t> allNodes;
    std::vector<float> allTris;
    int totalNodes = 0, totalTriVec4s = 0;

    if (!g_blasEntries.empty()) {
        // Concatenate all BLASes
        for (auto& be : g_blasEntries) {
            uint32_t* nodeData = nullptr;
            int numNodes = cudaBVH_getStacklessBVH2(be.bvh, &nodeData);
            float* triData = nullptr;
            int numTriVec4s = cudaBVH_getPackedTris(be.bvh, &triData);
            if (nodeData && numNodes > 0) {
                allNodes.insert(allNodes.end(), nodeData, nodeData + numNodes * 8);
            }
            if (triData && numTriVec4s > 0) {
                allTris.insert(allTris.end(), triData, triData + numTriVec4s * 4);
            }
            totalNodes += numNodes;
            totalTriVec4s += numTriVec4s;
        }
        LOG("[BVH2] Multi-BLAS concat: %d BLASes → %d total nodes, %d total triVec4s",
            (int)g_blasEntries.size(), totalNodes, totalTriVec4s);

        // Diagnostic: dump entity BLAS triangle data to verify integrity
        static int triDumpCount = 0;
        if (triDumpCount < 2) {
            for (size_t bi = 1; bi < g_blasEntries.size(); bi++) {
                // Dump initial BLASes (#1-4) and late entity BLASes (#38-44)
                if (bi > 4 && bi < 38) continue;
                if (bi >= 45) break;
                const auto& be = g_blasEntries[bi];
                int tOff = be.triOffset; // in triangles
                int nTri = be.numTris;
                LOG("[BVH2-DIAG] BLAS#%zu: triOff=%d numTris=%d nodeOff=%d numNodes=%d bounds=(%.2f,%.2f,%.2f)-(%.2f,%.2f,%.2f)",
                    bi, tOff, nTri, be.nodeOffset, be.numNodes,
                    be.minX, be.minY, be.minZ, be.maxX, be.maxY, be.maxZ);
                // Dump first 3 packed triangles from this BLAS
                for (int t = 0; t < std::min(nTri, 3); t++) {
                    int baseF = (tOff + t) * 12; // 12 floats per tri (3 vec4s)
                    if (baseF + 11 < (int)allTris.size()) {
                        float* p = allTris.data() + baseF;
                        int origIdx;
                        memcpy(&origIdx, &p[9], 4);
                        LOG("[BVH2-DIAG]   tri[%d]: v0=(%.2f,%.2f,%.2f) v1=(%.2f,%.2f,%.2f) v2=(%.2f,%.2f,%.2f) origIdx=%d",
                            t, p[0],p[1],p[2], p[3],p[4],p[5], p[6],p[7],p[8], origIdx);
                    }
                }
                // Dump root BVH node for this BLAS
                int nOff = be.nodeOffset * 8; // 8 uint32s per node
                if (nOff + 7 < (int)allNodes.size()) {
                    float bmin[3], bmax[3];
                    memcpy(bmin, allNodes.data() + nOff, 12);
                    memcpy(&bmax[0], allNodes.data() + nOff + 3, 4);
                    memcpy(&bmax[1], allNodes.data() + nOff + 4, 4);
                    memcpy(&bmax[2], allNodes.data() + nOff + 5, 4);
                    LOG("[BVH2-DIAG]   rootNode: min=(%.2f,%.2f,%.2f) max=(%.2f,%.2f,%.2f)",
                        bmin[0],bmin[1],bmin[2], bmax[0],bmax[1],bmax[2]);
                }
            }
            triDumpCount++;
        }
    } else {
        // Fallback: single BLAS (g_lastBLAS)
        uint32_t* nodeData = nullptr;
        int numNodes = cudaBVH_getStacklessBVH2(bvh, &nodeData);
        float* triData = nullptr;
        int numTriVec4s = cudaBVH_getPackedTris(bvh, &triData);
        if (!nodeData || numNodes == 0 || !triData || numTriVec4s == 0) {
            LOG("[BVH2] No stackless BVH2 data");
            return false;
        }
        allNodes.assign(nodeData, nodeData + numNodes * 8);
        allTris.assign(triData, triData + numTriVec4s * 4);
        totalNodes = numNodes;
        totalTriVec4s = numTriVec4s;
    }

    if (allNodes.empty() || allTris.empty()) {
        LOG("[BVH2] No BVH data after concatenation");
        return false;
    }

    // Debug: dump first few nodes
    {
        int dumpN = std::min(totalNodes, 4);
        for (int i = 0; i < dumpN; i++) {
            const uint32_t* n = allNodes.data() + i*8;
            float bmin[3], bmax[3];
            memcpy(bmin, n+0, 12);
            memcpy(&bmax[0], n+3, 4);
            memcpy(&bmax[1], n+4, 4);
            memcpy(&bmax[2], n+5, 4);
            int32_t enc  = (int32_t)n[6];
            int32_t skip = (int32_t)n[7];
            if (enc < 0) {
                int leaf = -(enc + 2);
                LOG("[BVH2] Node %d: bmin=(%.3f,%.3f,%.3f) bmax=(%.3f,%.3f,%.3f) LEAF ts=%d tc=%d skip=%d",
                    i, bmin[0],bmin[1],bmin[2], bmax[0],bmax[1],bmax[2], leaf>>3, (leaf&7)+1, skip);
            } else {
                LOG("[BVH2] Node %d: bmin=(%.3f,%.3f,%.3f) bmax=(%.3f,%.3f,%.3f) INTERNAL skip=%d",
                    i, bmin[0],bmin[1],bmin[2], bmax[0],bmax[1],bmax[2], skip);
            }
        }
    }

    // Destroy old buffers
    if (g_bvh2.nodesBuf) { disp.DestroyBuffer(disp.device, g_bvh2.nodesBuf, nullptr); g_bvh2.nodesBuf = VK_NULL_HANDLE; }
    if (g_bvh2.nodesMem) { disp.FreeMemory(disp.device, g_bvh2.nodesMem, nullptr); g_bvh2.nodesMem = VK_NULL_HANDLE; }
    if (g_bvh2.trisBuf)  { disp.DestroyBuffer(disp.device, g_bvh2.trisBuf, nullptr); g_bvh2.trisBuf = VK_NULL_HANDLE; }
    if (g_bvh2.trisMem)  { disp.FreeMemory(disp.device, g_bvh2.trisMem, nullptr); g_bvh2.trisMem = VK_NULL_HANDLE; }
    if (g_bvh2.tlasNodesBuf) { disp.DestroyBuffer(disp.device, g_bvh2.tlasNodesBuf, nullptr); g_bvh2.tlasNodesBuf = VK_NULL_HANDLE; }
    if (g_bvh2.tlasNodesMem) { disp.FreeMemory(disp.device, g_bvh2.tlasNodesMem, nullptr); g_bvh2.tlasNodesMem = VK_NULL_HANDLE; }
    if (g_bvh2.instancesBuf) { disp.DestroyBuffer(disp.device, g_bvh2.instancesBuf, nullptr); g_bvh2.instancesBuf = VK_NULL_HANDLE; }
    if (g_bvh2.instancesMem) { disp.FreeMemory(disp.device, g_bvh2.instancesMem, nullptr); g_bvh2.instancesMem = VK_NULL_HANDLE; }

    // Upload concatenated nodes
    VkDeviceSize nodesSize = (VkDeviceSize)allNodes.size() * sizeof(uint32_t);
    if (!createBufferWithData(disp, allNodes.data(), nodesSize,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, g_bvh2.nodesBuf, g_bvh2.nodesMem)) {
        LOG("[BVH2] Failed to create nodes buffer (%d nodes, %zu bytes)", totalNodes, (size_t)nodesSize);
        return false;
    }

    // Upload concatenated tris
    VkDeviceSize trisSize = (VkDeviceSize)allTris.size() * sizeof(float);
    if (!createBufferWithData(disp, allTris.data(), trisSize,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, g_bvh2.trisBuf, g_bvh2.trisMem)) {
        LOG("[BVH2] Failed to create tris buffer (%d vec4s, %zu bytes)", totalTriVec4s, (size_t)trisSize);
        return false;
    }

    // TLAS nodes + instances (bindings 2 & 3)
    VkDeviceSize tlasNodesSize = 0;
    VkDeviceSize instancesSize = 0;
    int numTlasNodes = 0;
    int numInst = (int)g_instances.size();

    if (g_tlasBVH && numInst > 0) {
        // Get TLAS BVH2 stackless data
        uint32_t* tlasNodeData = nullptr;
        numTlasNodes = cudaBVH_getStacklessBVH2(g_tlasBVH, &tlasNodeData);
        if (tlasNodeData && numTlasNodes > 0) {
            tlasNodesSize = (VkDeviceSize)numTlasNodes * 8 * sizeof(uint32_t);
            if (!createBufferWithData(disp, tlasNodeData, tlasNodesSize,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, g_bvh2.tlasNodesBuf, g_bvh2.tlasNodesMem)) {
                LOG("[BVH2] Failed to create TLAS nodes buffer");
                return false;
            }
            LOG("[BVH2] TLAS nodes: %d (%.1f KB)", numTlasNodes, tlasNodesSize/1024.f);
        }

        // Pack InstanceGPU into vec4 array: 9 vec4s per instance = 36 floats = 144 bytes
        instancesSize = (VkDeviceSize)numInst * 9 * 4 * sizeof(float);
        std::vector<float> instPacked(numInst * 36);
        for (int i = 0; i < numInst; i++) {
            // Use TLAS ordering if available (must match TLAS leaf indices)
            int origIdx = (g_fastTLASOrdered && i < numInst) ? g_fastTLASOrdered[i] : i;
            const InstanceGPU& gi = g_instances[origIdx];
            float* dst = instPacked.data() + i * 36;
            // vec4[0..2] = transform rows (3 × vec4)
            memcpy(dst + 0, gi.transform + 0, 16);
            memcpy(dst + 4, gi.transform + 4, 16);
            memcpy(dst + 8, gi.transform + 8, 16);
            // vec4[3..5] = invTransform rows (3 × vec4)
            memcpy(dst + 12, gi.invTransform + 0, 16);
            memcpy(dst + 16, gi.invTransform + 4, 16);
            memcpy(dst + 20, gi.invTransform + 8, 16);
            // vec4[6] = (blasMin.xyz, float(blasNodeOff))
            dst[24] = gi.blasMinX; dst[25] = gi.blasMinY; dst[26] = gi.blasMinZ;
            dst[27] = (float)gi.blasNodeOff;
            // vec4[7] = (blasMax.xyz, float(blasTriOff))
            dst[28] = gi.blasMaxX; dst[29] = gi.blasMaxY; dst[30] = gi.blasMaxZ;
            dst[31] = (float)gi.blasTriOff;
            // vec4[8] = (float(customIdx), float(sbtOffset), float(origInstIdx), float(packed_mask_flags))
            // packed_mask_flags = instanceMask | (instanceFlags << 8) — both uint8 values
            static int forceBSP = -1;
            if (forceBSP < 0) { const char* e = getenv("CUDA_RT_FORCE_BSP"); forceBSP = (e && atoi(e)) ? 1 : 0; }
            if (forceBSP) {
                // DEBUG: Find world BSP instance (blasNodeOff=0, blasTriOff=0, customIdx=0)
                static int worldOrigIdx = -1;
                if (worldOrigIdx < 0) {
                    for (int j = 0; j < numInst; j++) {
                        int oj = (g_fastTLASOrdered && j < numInst) ? g_fastTLASOrdered[j] : j;
                        const InstanceGPU& gj = g_instances[oj];
                        if (gj.customIdx == 0 && gj.blasNodeOff == 0 && gj.blasTriOff == 0) {
                            worldOrigIdx = oj;
                            fprintf(stderr, "[CudaRT] FORCE_BSP: world BSP at origIdx=%d\n", worldOrigIdx);
                            break;
                        }
                    }
                    if (worldOrigIdx < 0) worldOrigIdx = 0; // fallback
                }
                dst[32] = 0.0f;   // customIdx = 0 (VERTEX_BUFFER_WORLD)
                dst[33] = 0.0f;   // sbtOffset = 0 (SBTO_OPAQUE)
                dst[34] = (float)worldOrigIdx;  // origInstIdx = world BSP build-order index
                dst[35] = (float)(0x01 | (0x04 << 8)); // mask=1 (opaque), flags=FORCE_OPAQUE
            } else {
                dst[32] = (float)gi.customIdx;
                dst[33] = (float)gi.sbtOffset;
                dst[34] = (float)origIdx;  // original build-order instance index (for InstanceId)
                dst[35] = (float)(gi.instanceMask | (gi.instanceFlags << 8));
            }
        }
        if (!createBufferWithData(disp, instPacked.data(), instancesSize,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, g_bvh2.instancesBuf, g_bvh2.instancesMem)) {
            LOG("[BVH2] Failed to create instances buffer");
            return false;
        }
        LOG("[BVH2] Instances: %d (%.1f KB)", numInst, instancesSize/1024.f);
    }

    // If no TLAS, create minimal dummy buffers for bindings 2 & 3 (Vulkan requires valid buffers)
    // 4 nodes with huge AABBs + skip=-1 to prevent OOB and ensure immediate loop exit
    if (!g_bvh2.tlasNodesBuf) {
        uint32_t hugeF;
        float hugeV = 1e30f;
        memcpy(&hugeF, &hugeV, 4);
        uint32_t dummy[4*8];
        for (int i = 0; i < 4; i++) {
            uint32_t* n = dummy + i*8;
            n[0]=hugeF; n[1]=hugeF; n[2]=hugeF;
            n[3]=hugeF; n[4]=hugeF; n[5]=hugeF;
            n[6]=0; n[7]=0xFFFFFFFF;
        }
        tlasNodesSize = sizeof(dummy);
        if (!createBufferWithData(disp, dummy, tlasNodesSize,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, g_bvh2.tlasNodesBuf, g_bvh2.tlasNodesMem)) {
            LOG("[BVH2] Failed to create dummy TLAS nodes buffer");
            return false;
        }
    }
    if (!g_bvh2.instancesBuf) {
        float dummy[36] = {0};  // 9 vec4s per instance
        instancesSize = sizeof(dummy);
        if (!createBufferWithData(disp, dummy, instancesSize,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, g_bvh2.instancesBuf, g_bvh2.instancesMem)) {
            LOG("[BVH2] Failed to create dummy instances buffer");
            return false;
        }
    }

    // Update all 4 descriptor bindings
    VkDescriptorBufferInfo bufInfos[4] = {};
    bufInfos[0].buffer = g_bvh2.nodesBuf;
    bufInfos[0].offset = 0;
    bufInfos[0].range = nodesSize;
    bufInfos[1].buffer = g_bvh2.trisBuf;
    bufInfos[1].offset = 0;
    bufInfos[1].range = trisSize;
    bufInfos[2].buffer = g_bvh2.tlasNodesBuf;
    bufInfos[2].offset = 0;
    bufInfos[2].range = tlasNodesSize;
    bufInfos[3].buffer = g_bvh2.instancesBuf;
    bufInfos[3].offset = 0;
    bufInfos[3].range = instancesSize;

    VkWriteDescriptorSet writes[4] = {};
    for (int i = 0; i < 4; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = g_bvh2.descSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    disp.UpdateDescriptorSets(disp.device, 4, writes, 0, nullptr);

    g_bvh2.numNodes = totalNodes;
    g_bvh2.numTriVec4s = totalTriVec4s;
    g_bvh2.numTlasNodes = numTlasNodes;
    g_bvh2.numInstances = numInst;
    g_bvh2.descSetIdx = 4; // default — actual bind index comes from per-pipeline bvhDescSet
    g_bvh2.tlasGen = g_tlasGeneration;
    g_bvh2.ready = true;
    LOG("[BVH2] Uploaded: %d BLAS nodes (%.1f KB), %d tri-vec4s (%.1f KB), %d TLAS nodes, %d instances → descriptor set %u",
        totalNodes, nodesSize/1024.f, totalTriVec4s, trisSize/1024.f, numTlasNodes, numInst, g_bvh2.descSetIdx);
    return true;
}

// Fast path: re-upload only TLAS nodes + instance data (BLAS stays stable)
static bool reuploadTLASData(DeviceDispatch& disp) {
    if (g_instances.empty() || !g_bvh2.descSet) return false;

    // Destroy old TLAS + instance buffers
    if (g_bvh2.tlasNodesBuf) { disp.DestroyBuffer(disp.device, g_bvh2.tlasNodesBuf, nullptr); g_bvh2.tlasNodesBuf = VK_NULL_HANDLE; }
    if (g_bvh2.tlasNodesMem) { disp.FreeMemory(disp.device, g_bvh2.tlasNodesMem, nullptr); g_bvh2.tlasNodesMem = VK_NULL_HANDLE; }
    if (g_bvh2.instancesBuf) { disp.DestroyBuffer(disp.device, g_bvh2.instancesBuf, nullptr); g_bvh2.instancesBuf = VK_NULL_HANDLE; }
    if (g_bvh2.instancesMem) { disp.FreeMemory(disp.device, g_bvh2.instancesMem, nullptr); g_bvh2.instancesMem = VK_NULL_HANDLE; }

    int numInst = (int)g_instances.size();

    // Upload TLAS BVH2 nodes — prefer fast-built data if available
    uint32_t* tlasNodeData = nullptr;
    int numTlasNodes = 0;
    if (g_fastTLASNodes && g_fastTLASNodeCount > 0) {
        tlasNodeData = g_fastTLASNodes;
        numTlasNodes = g_fastTLASNodeCount;
    } else if (g_tlasBVH) {
        numTlasNodes = cudaBVH_getStacklessBVH2(g_tlasBVH, &tlasNodeData);
    }
    VkDeviceSize tlasNodesSize = 0;
    if (tlasNodeData && numTlasNodes > 0) {
        tlasNodesSize = (VkDeviceSize)numTlasNodes * 8 * sizeof(uint32_t);
        if (!createBufferHostVisible(disp, tlasNodeData, tlasNodesSize,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, g_bvh2.tlasNodesBuf, g_bvh2.tlasNodesMem)) {
            LOG("[BVH2] TLAS reupload: failed to create TLAS nodes buffer");
            return false;
        }
    }

    // Upload instance data — reordered to match TLAS BVH leaf indices
    // The LBVH builder sorts instances by Morton code; TLAS leaf triStart indexes
    // into the sorted order. Instance SSBO must match this order so the shader can
    // directly use the decoded triStart as the instance index.
    VkDeviceSize instancesSize = (VkDeviceSize)numInst * 9 * 4 * sizeof(float);
    std::vector<float> instPacked(numInst * 36);
    for (int i = 0; i < numInst; i++) {
        // Map sorted position → original instance index
        int origIdx = (g_fastTLASOrdered && i < numInst) ? g_fastTLASOrdered[i] : i;
        const InstanceGPU& gi = g_instances[origIdx];
        float* dst = instPacked.data() + i * 36;
        memcpy(dst + 0, gi.transform + 0, 16);
        memcpy(dst + 4, gi.transform + 4, 16);
        memcpy(dst + 8, gi.transform + 8, 16);
        memcpy(dst + 12, gi.invTransform + 0, 16);
        memcpy(dst + 16, gi.invTransform + 4, 16);
        memcpy(dst + 20, gi.invTransform + 8, 16);
        dst[24] = gi.blasMinX; dst[25] = gi.blasMinY; dst[26] = gi.blasMinZ;
        dst[27] = (float)gi.blasNodeOff;
        dst[28] = gi.blasMaxX; dst[29] = gi.blasMaxY; dst[30] = gi.blasMaxZ;
        dst[31] = (float)gi.blasTriOff;
        // vec4[8] = instance metadata
        static int forceBSP2 = -1;
        if (forceBSP2 < 0) { const char* e = getenv("CUDA_RT_FORCE_BSP"); forceBSP2 = (e && atoi(e)) ? 1 : 0; }
        if (forceBSP2) {
            static int worldOrigIdx2 = -1;
            if (worldOrigIdx2 < 0) {
                for (int j = 0; j < numInst; j++) {
                    int oj = (g_fastTLASOrdered && j < numInst) ? g_fastTLASOrdered[j] : j;
                    const InstanceGPU& gj = g_instances[oj];
                    if (gj.customIdx == 0 && gj.blasNodeOff == 0 && gj.blasTriOff == 0) {
                        worldOrigIdx2 = oj; break;
                    }
                }
                if (worldOrigIdx2 < 0) worldOrigIdx2 = 0;
            }
            dst[32] = 0.0f; dst[33] = 0.0f; dst[34] = (float)worldOrigIdx2;
            dst[35] = (float)(0x01 | (0x04 << 8));
        } else {
            dst[32] = (float)gi.customIdx;
            dst[33] = (float)gi.sbtOffset;
            dst[34] = (float)origIdx;
            dst[35] = (float)(gi.instanceMask | (gi.instanceFlags << 8));
        }
    }
    // One-time diagnostic: dump sorted→original instance mapping
    static bool dumpedMapping = false;
    if (!dumpedMapping) {
        dumpedMapping = true;
        for (int i = 0; i < numInst; i++) {
            int origIdx = (g_fastTLASOrdered && i < numInst) ? g_fastTLASOrdered[i] : i;
            const InstanceGPU& gi = g_instances[origIdx];
            fprintf(stderr, "[CudaRT] [INST-MAP] sorted[%d] → orig=%d customIdx=%d mask=0x%02x blasNOff=%d blasTOff=%d T=(%.1f,%.1f,%.1f)\n",
                    i, origIdx, gi.customIdx, gi.instanceMask, gi.blasNodeOff, gi.blasTriOff,
                    gi.transform[3], gi.transform[7], gi.transform[11]);
        }
    }
    if (!createBufferHostVisible(disp, instPacked.data(), instancesSize,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, g_bvh2.instancesBuf, g_bvh2.instancesMem)) {
        LOG("[BVH2] TLAS reupload: failed to create instances buffer");
        return false;
    }

    // Update only bindings 2 & 3 (TLAS nodes + instances)
    VkDescriptorBufferInfo bufInfos[2] = {};
    bufInfos[0].buffer = g_bvh2.tlasNodesBuf;
    bufInfos[0].offset = 0;
    bufInfos[0].range = tlasNodesSize;
    bufInfos[1].buffer = g_bvh2.instancesBuf;
    bufInfos[1].offset = 0;
    bufInfos[1].range = instancesSize;

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = g_bvh2.descSet;
    writes[0].dstBinding = 2;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &bufInfos[0];
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = g_bvh2.descSet;
    writes[1].dstBinding = 3;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &bufInfos[1];

    disp.UpdateDescriptorSets(disp.device, 2, writes, 0, nullptr);

    g_bvh2.numTlasNodes = numTlasNodes;
    g_bvh2.numInstances = numInst;
    g_bvh2.tlasGen = g_tlasGeneration;

    static int reuploadCount = 0;
    reuploadCount++;
    if (reuploadCount <= 3 || (reuploadCount % 300 == 0)) {
        LOG("[BVH2] TLAS reupload #%d: %d nodes, %d instances (gen=%lu)",
            reuploadCount, numTlasNodes, numInst, g_tlasGeneration);
        // Dump root TLAS node AABB (node 0)
        if (tlasNodeData && numTlasNodes > 0) {
            float bmin[3], bmax[3];
            memcpy(&bmin[0], &tlasNodeData[0], 4);
            memcpy(&bmin[1], &tlasNodeData[1], 4);
            memcpy(&bmin[2], &tlasNodeData[2], 4);
            memcpy(&bmax[0], &tlasNodeData[3], 4);
            memcpy(&bmax[1], &tlasNodeData[4], 4);
            memcpy(&bmax[2], &tlasNodeData[5], 4);
            int32_t leaf_enc, skip;
            memcpy(&leaf_enc, &tlasNodeData[6], 4);
            memcpy(&skip, &tlasNodeData[7], 4);
            LOG("[BVH2] TLAS root node: AABB=(%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f) leaf=%d skip=%d",
                bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2], leaf_enc, skip);
        }
        // Dump first instance transform
        if (numInst > 0) {
            float* i0 = instPacked.data();
            LOG("[BVH2] Instance 0 transform row0: (%.3f,%.3f,%.3f,%.3f)",
                i0[0], i0[1], i0[2], i0[3]);
            LOG("[BVH2] Instance 0 transform row1: (%.3f,%.3f,%.3f,%.3f)",
                i0[4], i0[5], i0[6], i0[7]);
            LOG("[BVH2] Instance 0 transform row2: (%.3f,%.3f,%.3f,%.3f)",
                i0[8], i0[9], i0[10], i0[11]);
            LOG("[BVH2] Instance 0 invTransform row0: (%.3f,%.3f,%.3f,%.3f)",
                i0[12], i0[13], i0[14], i0[15]);
            LOG("[BVH2] Instance 0 invTransform row1: (%.3f,%.3f,%.3f,%.3f)",
                i0[16], i0[17], i0[18], i0[19]);
            LOG("[BVH2] Instance 0 invTransform row2: (%.3f,%.3f,%.3f,%.3f)",
                i0[20], i0[21], i0[22], i0[23]);
            uint32_t noff, toff;
            memcpy(&noff, &i0[27], 4);
            memcpy(&toff, &i0[31], 4);
            LOG("[BVH2] Instance 0 blasBounds: (%.3f,%.3f,%.3f)-(%.3f,%.3f,%.3f) nOff=%u tOff=%u",
                i0[24], i0[25], i0[26], i0[28], i0[29], i0[30], noff, toff);
        }

        // ── CPU-side BLAS root AABB test for each instance ──
        // Mimics the GPU: transform world ray to local space, test against BLAS root node
        static int blasAabbTestCount = 0;
        if (blasAabbTestCount < 3 && numInst > 0) {
            LOG("[BVH2-AABB] CPU-side BLAS root AABB test (%d instances):", numInst);
            for (int i = 0; i < numInst; i++) {
                float* d = instPacked.data() + i * 36;
                float xform[12], invXform[12];
                memcpy(xform, d, 48);
                memcpy(invXform, d + 12, 48);
                int blasNOff = (int)d[27]; // nodeOff stored as float
                int blasTOff = (int)d[31]; // triOff stored as float

                // BLAS root AABB from instance SSBO (blasBounds)
                float bmin[3] = {d[24], d[25], d[26]};
                float bmax[3] = {d[28], d[29], d[30]};

                // Model-space center of BLAS AABB
                float mcx = (bmin[0]+bmax[0])*0.5f, mcy = (bmin[1]+bmax[1])*0.5f, mcz = (bmin[2]+bmax[2])*0.5f;
                // Transform to world space: worldCenter = xform * modelCenter + xform_translation
                float wcx = xform[0]*mcx + xform[1]*mcy + xform[2]*mcz + xform[3];
                float wcy = xform[4]*mcx + xform[5]*mcy + xform[6]*mcz + xform[7];
                float wcz = xform[8]*mcx + xform[9]*mcy + xform[10]*mcz + xform[11];

                // Create test ray: from (wcx, wcy, wcz+200) shooting -Z toward entity
                float rox = wcx, roy = wcy, roz = wcz + 200.0f;
                float rdx = 0, rdy = 0, rdz = -1.0f;

                // Transform ray to local space using inverse transform (same as GPU)
                float lox = invXform[0]*rox + invXform[1]*roy + invXform[2]*roz + invXform[3];
                float loy = invXform[4]*rox + invXform[5]*roy + invXform[6]*roz + invXform[7];
                float loz = invXform[8]*rox + invXform[9]*roy + invXform[10]*roz + invXform[11];
                float ldx = invXform[0]*rdx + invXform[1]*rdy + invXform[2]*rdz;
                float ldy = invXform[4]*rdx + invXform[5]*rdy + invXform[6]*rdz;
                float ldz = invXform[8]*rdx + invXform[9]*rdy + invXform[10]*rdz;

                // Safe inverse direction
                auto safeInvF = [](float v) -> float {
                    return (fabsf(v) > 1e-8f) ? 1.0f/v : (v >= 0 ? 1e8f : -1e8f);
                };
                float lInvDx = safeInvF(ldx), lInvDy = safeInvF(ldy), lInvDz = safeInvF(ldz);
                float lOodx = -lox * lInvDx, lOody = -loy * lInvDy, lOodz = -loz * lInvDz;

                // AABB slab test (same as GPU)
                float t1x = bmin[0]*lInvDx + lOodx, t2x = bmax[0]*lInvDx + lOodx;
                float t1y = bmin[1]*lInvDy + lOody, t2y = bmax[1]*lInvDy + lOody;
                float t1z = bmin[2]*lInvDz + lOodz, t2z = bmax[2]*lInvDz + lOodz;
                float tNear = fmaxf(fmaxf(fminf(t1x,t2x), fminf(t1y,t2y)), fminf(t1z,t2z));
                float tFar  = fminf(fminf(fmaxf(t1x,t2x), fmaxf(t1y,t2y)), fmaxf(t1z,t2z));
                bool hit = (tNear <= tFar) && (tFar > 0.0f) && (tNear < 1e30f);

                LOG("[BVH2-AABB]  inst[%d] nOff=%d tOff=%d",
                    i, blasNOff, blasTOff);
                LOG("[BVH2-AABB]    BLAS root AABB: (%.2f,%.2f,%.2f)-(%.2f,%.2f,%.2f)",
                    bmin[0],bmin[1],bmin[2], bmax[0],bmax[1],bmax[2]);
                LOG("[BVH2-AABB]    xform T=[%.1f,%.1f,%.1f] worldCenter=(%.1f,%.1f,%.1f)",
                    xform[3], xform[7], xform[11], wcx, wcy, wcz);
                LOG("[BVH2-AABB]    localRo=(%.2f,%.2f,%.2f) localRd=(%.4f,%.4f,%.4f)",
                    lox,loy,loz, ldx,ldy,ldz);
                LOG("[BVH2-AABB]    tNear=%.4f tFar=%.4f hit=%d", tNear, tFar, hit ? 1 : 0);
            }
            blasAabbTestCount++;
        }
    }

    // ── CPU-side TLAS traversal test (first 3 rebuilds only) ──
    // Mimics the GPU shader's BVH2 stackless traversal to verify correctness
    if (reuploadCount <= 3 && tlasNodeData && numTlasNodes > 0) {
        // Root AABB
        float rmin[3], rmax[3];
        memcpy(rmin, &tlasNodeData[0], 12);
        memcpy(rmax, &tlasNodeData[3], 12);
        float cx = (rmin[0]+rmax[0])*0.5f, cy = (rmin[1]+rmax[1])*0.5f, cz = (rmin[2]+rmax[2])*0.5f;
        float ex = (rmax[0]-rmin[0]), ey = (rmax[1]-rmin[1]), ez = (rmax[2]-rmin[2]);

        // Test rays: from just outside root AABB toward center
        struct TestRay { float ox,oy,oz, dx,dy,dz; const char* name; };
        TestRay rays[] = {
            {cx, cy, rmax[2]+100.f, 0,0,-1, "+Z toward center"},
            {cx, cy, rmin[2]-100.f, 0,0, 1, "-Z toward center"},
            {rmax[0]+100.f, cy, cz, -1,0,0, "+X toward center"},
            {cx, rmax[1]+100.f, cz, 0,-1,0, "+Y toward center"},
            {cx, cy, cz, 0,0,-1, "from center -Z"},
            {cx, cy, cz, 1,0,0, "from center +X"},
        };

        for (auto& ray : rays) {
            float invDx = (fabsf(ray.dx) > 1e-8f) ? 1.0f/ray.dx : (ray.dx >= 0 ? 1e8f : -1e8f);
            float invDy = (fabsf(ray.dy) > 1e-8f) ? 1.0f/ray.dy : (ray.dy >= 0 ? 1e8f : -1e8f);
            float invDz = (fabsf(ray.dz) > 1e-8f) ? 1.0f/ray.dz : (ray.dz >= 0 ? 1e8f : -1e8f);
            float oodx = -ray.ox * invDx;
            float oody = -ray.oy * invDy;
            float oodz = -ray.oz * invDz;
            float bestT = 1e30f;

            int ni = 0, iter = 0, leafHits = 0, internalHits = 0, totalIter = 0;
            while (ni >= 0 && iter < 10000) {
                const uint32_t* p = tlasNodeData + ni * 8;
                float bmin[3], bmax[3];
                memcpy(bmin, p+0, 12);
                memcpy(bmax, p+3, 12);
                int32_t leaf_enc, skip;
                memcpy(&leaf_enc, p+6, 4);
                memcpy(&skip, p+7, 4);

                if (leaf_enc != 0) {
                    // Leaf node — TLAS hit!
                    leafHits++;
                    ni = skip;  // continue to next subtree
                } else {
                    // Internal node — AABB slab test (same math as GPU shader)
                    float t1x = bmin[0]*invDx + oodx, t2x = bmax[0]*invDx + oodx;
                    float t1y = bmin[1]*invDy + oody, t2y = bmax[1]*invDy + oody;
                    float t1z = bmin[2]*invDz + oodz, t2z = bmax[2]*invDz + oodz;
                    float tNear = fmaxf(fmaxf(fminf(t1x,t2x), fminf(t1y,t2y)), fminf(t1z,t2z));
                    float tFar  = fminf(fminf(fmaxf(t1x,t2x), fmaxf(t1y,t2y)), fmaxf(t1z,t2z));
                    bool hit = (tNear <= tFar) && (tFar > 0.0f) && (tNear < bestT);
                    if (hit) {
                        internalHits++;
                        ni = ni + 1;  // traverse left child
                    } else {
                        ni = skip;    // skip to next subtree
                    }
                }
                iter++;
                totalIter = iter;
            }
            LOG("[BVH2-CPU] Ray '%s' o=(%.1f,%.1f,%.1f) d=(%.1f,%.1f,%.1f): %d leaf hits, %d internal hits, %d iters",
                ray.name, ray.ox,ray.oy,ray.oz, ray.dx,ray.dy,ray.dz, leafHits, internalHits, totalIter);
        }
    }

    return true;
}

static bool setupComputePipeline(DeviceDispatch& disp) {
    if (g_compute.pipelineReady) return true;

    // 1. Create shader module from embedded SPIR-V
    bool useBVH2 = getenv("CUDA_RT_BVH2") && atoi(getenv("CUDA_RT_BVH2"));
    VkShaderModuleCreateInfo smCI = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    if (useBVH2) {
        smCI.codeSize = bvh2_stackless_spv_size;
        smCI.pCode = bvh2_stackless_spv;
        LOG("  [compute] Using stackless BVH2 shader");
    } else {
        smCI.codeSize = bvh4_trace_spv_size;
        smCI.pCode = bvh4_trace_spv;
    }
    VkShaderModule shaderModule;
    if (disp.CreateShaderModule(disp.device, &smCI, nullptr, &shaderModule) != VK_SUCCESS) {
        LOG("  [compute] Failed to create shader module");
        return false;
    }

    // 2. Descriptor set layout: 10 SSBOs + 1 storage image
    VkDescriptorSetLayoutBinding bindings[12] = {};
    for (int i = 0; i < 10; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[10].binding = 10;
    bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[10].descriptorCount = 1;
    bindings[10].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Binding 11: UBO for top BVH nodes (constant cache broadcast)
    bindings[11].binding = 11;
    bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[11].descriptorCount = 1;
    bindings[11].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dsLayoutCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsLayoutCI.bindingCount = 12;
    dsLayoutCI.pBindings = bindings;
    if (disp.CreateDescriptorSetLayout(disp.device, &dsLayoutCI, nullptr, &g_compute.dsLayout) != VK_SUCCESS) {
        LOG("  [compute] Failed to create descriptor set layout");
        disp.DestroyShaderModule(disp.device, shaderModule, nullptr);
        return false;
    }

    // 3. Pipeline layout with push constants
    VkPushConstantRange pcRange = {};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(ComputeTracer::PushConstants);

    VkPipelineLayoutCreateInfo plCI = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts = &g_compute.dsLayout;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges = &pcRange;
    if (disp.CreatePipelineLayout(disp.device, &plCI, nullptr, &g_compute.pipeLayout) != VK_SUCCESS) {
        LOG("  [compute] Failed to create pipeline layout");
        disp.DestroyShaderModule(disp.device, shaderModule, nullptr);
        return false;
    }

    // 4. Compute pipeline
    VkComputePipelineCreateInfo cpCI = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpCI.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    cpCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpCI.stage.module = shaderModule;
    cpCI.stage.pName = "main";
    cpCI.layout = g_compute.pipeLayout;
    if (disp.CreateComputePipelines(disp.device, VK_NULL_HANDLE, 1, &cpCI, nullptr, &g_compute.pipeline) != VK_SUCCESS) {
        LOG("  [compute] Failed to create compute pipeline");
        disp.DestroyShaderModule(disp.device, shaderModule, nullptr);
        return false;
    }
    disp.DestroyShaderModule(disp.device, shaderModule, nullptr);

    // Query pipeline executable statistics (register count, local memory, etc.)
    auto getPipeProps = (PFN_vkGetPipelineExecutablePropertiesKHR)
        disp.GetDeviceProcAddr(disp.device, "vkGetPipelineExecutablePropertiesKHR");
    auto getPipeStats = (PFN_vkGetPipelineExecutableStatisticsKHR)
        disp.GetDeviceProcAddr(disp.device, "vkGetPipelineExecutableStatisticsKHR");
    if (getPipeProps && getPipeStats) {
        VkPipelineInfoKHR pipeInfo = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
        pipeInfo.pipeline = g_compute.pipeline;
        uint32_t execCount = 0;
        getPipeProps(disp.device, &pipeInfo, &execCount, nullptr);
        if (execCount > 0) {
            std::vector<VkPipelineExecutablePropertiesKHR> props(execCount);
            for (auto& p : props) p.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
            getPipeProps(disp.device, &pipeInfo, &execCount, props.data());
            for (uint32_t e = 0; e < execCount; e++) {
                LOG("  [pipeline] Executable %d: %s (stage 0x%x, subgroupSize %u)",
                    e, props[e].name, props[e].stages, props[e].subgroupSize);
                VkPipelineExecutableInfoKHR execInfo = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
                execInfo.pipeline = g_compute.pipeline;
                execInfo.executableIndex = e;
                uint32_t statCount = 0;
                getPipeStats(disp.device, &execInfo, &statCount, nullptr);
                if (statCount > 0) {
                    std::vector<VkPipelineExecutableStatisticKHR> stats(statCount);
                    for (auto& s : stats) s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
                    getPipeStats(disp.device, &execInfo, &statCount, stats.data());
                    for (uint32_t s = 0; s < statCount; s++) {
                        auto& st = stats[s];
                        if (st.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR)
                            LOG("    %s = %lld", st.name, (long long)st.value.i64);
                        else if (st.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR)
                            LOG("    %s = %llu", st.name, (unsigned long long)st.value.u64);
                        else if (st.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR)
                            LOG("    %s = %.2f", st.name, st.value.f64);
                        else if (st.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR)
                            LOG("    %s = %s", st.name, st.value.b32 ? "true" : "false");
                    }
                }
            }
        }
    }

    // 5. Descriptor pool
    VkDescriptorPoolSize poolSizes[3] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 10;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[2].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCI.maxSets = 1;
    dpCI.poolSizeCount = 3;
    dpCI.pPoolSizes = poolSizes;
    if (disp.CreateDescriptorPool(disp.device, &dpCI, nullptr, &g_compute.descPool) != VK_SUCCESS) {
        LOG("  [compute] Failed to create descriptor pool");
        return false;
    }

    // 6. Allocate descriptor set
    VkDescriptorSetAllocateInfo dsAI = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsAI.descriptorPool = g_compute.descPool;
    dsAI.descriptorSetCount = 1;
    dsAI.pSetLayouts = &g_compute.dsLayout;
    if (disp.AllocateDescriptorSets(disp.device, &dsAI, &g_compute.descSet) != VK_SUCCESS) {
        LOG("  [compute] Failed to allocate descriptor set");
        return false;
    }

    g_compute.pipelineReady = true;
    LOG("  [compute] BVH4 compute pipeline created — zero-overhead Vulkan tracing ready");
    return true;
}

// Upload BVH4 data from CUDA to Vulkan buffers and update descriptor set
static bool uploadBVHData(DeviceDispatch& disp, CudaBVH_t bvh) {
    if (g_compute.dataUploaded && g_compute.lastBVH == bvh) return true;
    if (!g_compute.pipelineReady) return false;

    bool useBVH2 = getenv("CUDA_RT_BVH2") && atoi(getenv("CUDA_RT_BVH2"));
    int numNodes;
    VkDeviceSize nodeSize;
    const void* nodeData;

    if (useBVH2) {
        uint32_t* bvh2Data = nullptr;
        numNodes = cudaBVH_getStacklessBVH2(bvh, &bvh2Data);
        nodeSize = (VkDeviceSize)numNodes * 8 * 4;  // 8 uint32s per node
        nodeData = bvh2Data;
        if (!nodeData) {
            LOG("  [compute] Stackless BVH2 data not available");
            return false;
        }
        LOG("  [compute] Using stackless BVH2: %d nodes (%.1f KB)", numNodes, nodeSize/1024.0);
    } else {
        numNodes = cudaBVH_getNumBVH4Nodes(bvh);
        nodeSize = (VkDeviceSize)numNodes * 4 * 16;
        nodeData = cudaBVH_getNodeData(bvh);
    }

    int numTris = cudaBVH_getNumTris(bvh);
    VkDeviceSize triSize = (VkDeviceSize)numTris * sizeof(float);

    // Get raw pointers from CUDA backend
    const float* h_tv[9];
    cudaBVH_getTriData(bvh, h_tv);

    if (!nodeData || !h_tv[0]) {
        LOG("  [compute] BVH data not available for upload");
        return false;
    }

    // Clean up old buffers
    if (g_compute.dataUploaded) {
        disp.DestroyBuffer(disp.device, g_compute.bvhNodesBuf, nullptr);
        disp.FreeMemory(disp.device, g_compute.bvhNodesMem, nullptr);
        disp.DestroyBuffer(disp.device, g_compute.topNodesBuf, nullptr);
        disp.FreeMemory(disp.device, g_compute.topNodesMem, nullptr);
        for (int i = 0; i < 9; i++) {
            disp.DestroyBuffer(disp.device, g_compute.triBufs[i], nullptr);
            disp.FreeMemory(disp.device, g_compute.triMems[i], nullptr);
        }
    }

    // Upload BVH nodes (SSBO for all nodes)
    if (!createBufferWithData(disp, nodeData, nodeSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               g_compute.bvhNodesBuf, g_compute.bvhNodesMem)) {
        LOG("  [compute] Failed to create BVH nodes buffer");
        return false;
    }

    // Upload top 1024 BVH nodes as UBO (constant cache broadcast)
    int uboNodes = useBVH2 ? 0 : std::min(numNodes, 1024);
    VkDeviceSize uboSize = 65536;  // always 64KB (max UBO size, pad with zeros)
    std::vector<uint8_t> uboPad(uboSize, 0);
    if (!useBVH2) memcpy(uboPad.data(), nodeData, (size_t)uboNodes * 4 * 16);
    if (!createBufferWithData(disp, uboPad.data(), uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                               g_compute.topNodesBuf, g_compute.topNodesMem)) {
        LOG("  [compute] Failed to create top nodes UBO");
        return false;
    }
    LOG("  [compute] UBO: top %d nodes (64 KB constant cache)", uboNodes);

    // Upload 9 triangle SoA arrays (still used for descriptor bindings 2-9, but mainly for AoS pack)
    for (int i = 0; i < 9; i++) {
        if (!createBufferWithData(disp, h_tv[i], triSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   g_compute.triBufs[i], g_compute.triMems[i])) {
            LOG("  [compute] Failed to create triangle buffer %d", i);
            return false;
        }
    }

    // Pack triangles into AoS format: 3 vec4 per triangle = [v0x,v0y,v0z,v1x] [v1y,v1z,v2x,v2y] [v2z,0,0,0]
    // This gives excellent cache locality for Moller-Trumbore (1-2 cache lines vs 9 scattered reads)
    VkDeviceSize aosTotalSize = (VkDeviceSize)numTris * 3 * 16;  // 3 vec4 × 16 bytes per vec4
    std::vector<float> aosPacked(numTris * 12);  // 12 floats per triangle (3 vec4, last 3 are padding)
    for (int ti = 0; ti < numTris; ti++) {
        aosPacked[ti*12+0] = h_tv[0][ti];  // v0x
        aosPacked[ti*12+1] = h_tv[1][ti];  // v0y
        aosPacked[ti*12+2] = h_tv[2][ti];  // v0z
        aosPacked[ti*12+3] = h_tv[3][ti];  // v1x
        aosPacked[ti*12+4] = h_tv[4][ti];  // v1y
        aosPacked[ti*12+5] = h_tv[5][ti];  // v1z
        aosPacked[ti*12+6] = h_tv[6][ti];  // v2x
        aosPacked[ti*12+7] = h_tv[7][ti];  // v2y
        aosPacked[ti*12+8] = h_tv[8][ti];  // v2z
        aosPacked[ti*12+9] = 0.0f;
        aosPacked[ti*12+10] = 0.0f;
        aosPacked[ti*12+11] = 0.0f;
    }
    // Overwrite binding 1 with AoS packed data
    disp.DestroyBuffer(disp.device, g_compute.triBufs[0], nullptr);
    disp.FreeMemory(disp.device, g_compute.triMems[0], nullptr);
    if (!createBufferWithData(disp, aosPacked.data(), aosTotalSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               g_compute.triBufs[0], g_compute.triMems[0])) {
        LOG("  [compute] Failed to create AoS triangle buffer");
        return false;
    }
    LOG("  [compute] AoS packed: %d tris × 48 bytes = %.1f KB", numTris, aosTotalSize/1024.0);

    // Update descriptor set: binding 0 = BVH nodes, 1 = AoS triangles, 2-9 = SoA (unused by shader), 11 = UBO
    VkDescriptorBufferInfo bufInfos[10] = {};
    bufInfos[0].buffer = g_compute.bvhNodesBuf;
    bufInfos[0].offset = 0;
    bufInfos[0].range = nodeSize;
    // Binding 1: AoS packed triangles
    bufInfos[1].buffer = g_compute.triBufs[0];
    bufInfos[1].offset = 0;
    bufInfos[1].range = aosTotalSize;
    for (int i = 1; i < 9; i++) {
        bufInfos[1+i].buffer = g_compute.triBufs[i];
        bufInfos[1+i].offset = 0;
        bufInfos[1+i].range = triSize;
    }

    VkWriteDescriptorSet writes[11] = {};
    for (int i = 0; i < 10; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = g_compute.descSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    // Binding 11: UBO for top nodes
    VkDescriptorBufferInfo uboInfo = {};
    uboInfo.buffer = g_compute.topNodesBuf;
    uboInfo.offset = 0;
    uboInfo.range = uboSize;
    writes[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[10].dstSet = g_compute.descSet;
    writes[10].dstBinding = 11;
    writes[10].descriptorCount = 1;
    writes[10].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[10].pBufferInfo = &uboInfo;
    disp.UpdateDescriptorSets(disp.device, 11, writes, 0, nullptr);

    g_compute.dataUploaded = true;
    g_compute.lastBVH = bvh;
    LOG("  [compute] Uploaded BVH4: %d nodes (%.1f KB) + %d tris × 9 arrays (%.1f KB each)",
        numNodes, nodeSize/1024.0, numTris, triSize/1024.0);
    return true;
}

// Update the storage image descriptor (binding 10) — must be called each frame if image changes
static void updateOutputImageDescriptor(DeviceDispatch& disp, VkImageView imageView) {
    VkDescriptorImageInfo imgInfo = {};
    imgInfo.imageView = imageView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = g_compute.descSet;
    write.dstBinding = 10;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imgInfo;
    disp.UpdateDescriptorSets(disp.device, 1, &write, 0, nullptr);
}
static bool setupStagingInterop(DeviceDispatch& disp, uint32_t width, uint32_t height, VkDeviceSize bufSize)
{
    // Reuse existing buffer if it's big enough
    if (g_staging.ready && g_staging.size >= bufSize)
        return true;

    // Clean up old staging if size changed
    if (g_staging.ready) {
        if (g_staging.hostPtr && g_staging.memory) disp.UnmapMemory(disp.device, g_staging.memory);
        if (g_staging.buffer) disp.DestroyBuffer(disp.device, g_staging.buffer, nullptr);
        if (g_staging.memory) disp.FreeMemory(disp.device, g_staging.memory, nullptr);
        g_staging = {};
    }

    // Try DEVICE_LOCAL + external memory first (fast GPU→GPU path)
    if (disp.GetMemoryFdKHR) {
        VkExternalMemoryBufferCreateInfo extBufCI = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        extBufCI.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkBufferCreateInfo bufCI = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufCI.pNext = &extBufCI;
        bufCI.size = bufSize;
        bufCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkResult res = disp.CreateBuffer(disp.device, &bufCI, nullptr, &g_staging.buffer);
        if (res == VK_SUCCESS) {
            VkMemoryRequirements memReqs;
            disp.GetBufferMemoryRequirements(disp.device, g_staging.buffer, &memReqs);

            // Find DEVICE_LOCAL memory type
            int memTypeIdx = -1;
            for (uint32_t i = 0; i < disp.memProps.memoryTypeCount; i++) {
                if ((memReqs.memoryTypeBits & (1 << i)) &&
                    (disp.memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                    memTypeIdx = i;
                    break;
                }
            }

            if (memTypeIdx >= 0) {
                VkExportMemoryAllocateInfo exportAI = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
                exportAI.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

                VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocInfo.pNext = &exportAI;
                allocInfo.allocationSize = memReqs.size;
                allocInfo.memoryTypeIndex = memTypeIdx;

                res = disp.AllocateMemory(disp.device, &allocInfo, nullptr, &g_staging.memory);
                if (res == VK_SUCCESS) {
                    res = disp.BindBufferMemory(disp.device, g_staging.buffer, g_staging.memory, 0);
                    if (res == VK_SUCCESS) {
                        // Export fd
                        VkMemoryGetFdInfoKHR fdInfo = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
                        fdInfo.memory = g_staging.memory;
                        fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
                        int fd = -1;
                        res = disp.GetMemoryFdKHR(disp.device, &fdInfo, &fd);
                        if (res == VK_SUCCESS && fd >= 0) {
                            // Import into CUDA
                            void* cudaPtr = cudaBVH_importBufferFd(fd, memReqs.size);
                            if (cudaPtr) {
                                g_staging.cudaPtr = cudaPtr;
                                g_staging.hostPtr = nullptr;
                                g_staging.width = width;
                                g_staging.height = height;
                                g_staging.size = bufSize;
                                g_staging.ready = true;
                                g_staging.deviceLocal = true;
                                LOG("  [staging] DEVICE_LOCAL %ux%u buffer (%lu KB) cudaPtr=%p — GPU→GPU path",
                                    width, height, (unsigned long)(bufSize/1024), cudaPtr);
                                return true;
                            }
                        }
                    }
                }
            }
            // Failed — clean up and fall through to HOST_VISIBLE
            if (g_staging.buffer) disp.DestroyBuffer(disp.device, g_staging.buffer, nullptr);
            if (g_staging.memory) disp.FreeMemory(disp.device, g_staging.memory, nullptr);
            g_staging = {};
            LOG("  [staging] DEVICE_LOCAL failed, falling back to HOST_VISIBLE");
        }
    }

    // Fallback: HOST_VISIBLE | HOST_COHERENT (slower, but guaranteed)
    VkBufferCreateInfo bufCI = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufCI.size = bufSize;
    bufCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = disp.CreateBuffer(disp.device, &bufCI, nullptr, &g_staging.buffer);
    if (res != VK_SUCCESS) {
        LOG("  [staging] CreateBuffer failed: %d", res);
        return false;
    }

    VkMemoryRequirements memReqs;
    disp.GetBufferMemoryRequirements(disp.device, g_staging.buffer, &memReqs);

    int memTypeIdx = -1;
    for (uint32_t i = 0; i < disp.memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (disp.memProps.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memTypeIdx = i;
            break;
        }
    }
    if (memTypeIdx < 0) {
        LOG("  [staging] No HOST_VISIBLE|HOST_COHERENT memory type found");
        disp.DestroyBuffer(disp.device, g_staging.buffer, nullptr);
        g_staging.buffer = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memTypeIdx;

    res = disp.AllocateMemory(disp.device, &allocInfo, nullptr, &g_staging.memory);
    if (res != VK_SUCCESS) {
        LOG("  [staging] AllocateMemory failed: %d", res);
        disp.DestroyBuffer(disp.device, g_staging.buffer, nullptr);
        g_staging.buffer = VK_NULL_HANDLE;
        return false;
    }

    res = disp.BindBufferMemory(disp.device, g_staging.buffer, g_staging.memory, 0);
    if (res != VK_SUCCESS) { LOG("  [staging] BindBufferMemory failed: %d", res); return false; }

    res = disp.MapMemory(disp.device, g_staging.memory, 0, bufSize, 0, &g_staging.hostPtr);
    if (res != VK_SUCCESS || !g_staging.hostPtr) { LOG("  [staging] MapMemory failed: %d", res); return false; }

    g_staging.width = width;
    g_staging.height = height;
    g_staging.size = bufSize;
    g_staging.ready = true;
    g_staging.deviceLocal = false;
    LOG("  [staging] HOST_VISIBLE %ux%u buffer (%lu KB) hostPtr=%p",
        width, height, (unsigned long)(bufSize/1024), g_staging.hostPtr);
    return true;
}

// ═══════════════════════════════════════════
// Intercepted: CmdTraceRaysKHR — THE BIG ONE
// This is where we substitute our CUDA kernel for the driver's RT
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_CmdTraceRaysKHR(
    VkCommandBuffer cmdBuf,
    const VkStridedDeviceAddressRegionKHR* pRaygenSBT,
    const VkStridedDeviceAddressRegionKHR* pMissSBT,
    const VkStridedDeviceAddressRegionKHR* pHitSBT,
    const VkStridedDeviceAddressRegionKHR* pCallableSBT,
    uint32_t width,
    uint32_t height,
    uint32_t depth)
{
    static int frameN = 0;
    static bool replaceMode = (getenv("CUDA_RT_REPLACE") && atoi(getenv("CUDA_RT_REPLACE")));
    
    // Per-call timing
    struct timespec t0, t1, t2, t3;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    
    // FPS counter — prints to stderr every second
    static struct timespec fpsStart = {0, 0};
    static int fpsFrames = 0;
    if (fpsStart.tv_sec == 0) clock_gettime(CLOCK_MONOTONIC, &fpsStart);
    fpsFrames++;
    {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - fpsStart.tv_sec) + (now.tv_nsec - fpsStart.tv_nsec) * 1e-9;
        if (elapsed >= 1.0) {
            double fps = fpsFrames / elapsed;
            double msPerFrame = 1000.0 / fps;
            int numTris = g_lastBLAS ? cudaBVH_getNumTris(g_lastBLAS) : 0;
            double mrays = (double)(width * height) * fps / 1e6;
            fprintf(stderr, "\r[CudaRT] %.1f FPS (%.1f ms) | %ux%u | %dK tris | %.0f MR/s | frame %d %s   ",
                    fps, msPerFrame, width, height, numTris/1000, mrays,
                    frameN, replaceMode ? "REPLACE" : "PASS");
            fflush(stderr);
            fpsStart = now;
            fpsFrames = 0;
        }
    }

    if (frameN++ < 5) {
        LOG("CmdTraceRays: %ux%ux%u (%.2f Mrays) frame=%d %s",
            width, height, depth, (float)(width * height * depth) / 1e6f,
            frameN, replaceMode ? "[REPLACE]" : "[PASSTHROUGH]");
    }

    // Run our CUDA BVH trace if a BLAS has been built
    float camX = 0, camY = 0, camZ = 0;
    if (g_lastBLAS) {
        float cx, cy, cz, extent;
        cudaBVH_getBounds(g_lastBLAS, &cx, &cy, &cz, &extent);
        camX = cx; camY = cy; camZ = cz + extent * 1.2f;
        
        // Stats trace only on first few frames (skip in replace mode for perf)
        if (!replaceMode && (frameN <= 5 || (frameN % 100) == 0)) {
            int side = (width > height) ? width : height;
            float mrs = cudaBVH_tracePrimary(g_lastBLAS, side, cx, cy, cz - extent, nullptr);
            LOG("  → CUDA BVH4: %dx%d → %.0f MR/s (%.1f GR/s)", side, side, mrs, mrs/1000.f);
        }
    }

    // Forward to driver — SKIP in replace mode (the 1×1 dispatch still causes 65ms stall
    // due to software RT emulation on V100). Our compute shader writes directly to the
    // storage image, so no pipeline state maintenance needed.
    if (!replaceMode) {
        void* key = getKey(cmdBuf);
        auto it = g_deviceMap.find(key);
        if (it != g_deviceMap.end() && it->second.CmdTraceRaysKHR) {
            it->second.CmdTraceRaysKHR(cmdBuf, pRaygenSBT, pMissSBT, pHitSBT, pCallableSBT, width, height, depth);
        }
    }

    // In replace mode: overwrite storage images with our Vulkan compute BVH4 tracer.
    // No CUDA involved — runs natively in the Vulkan command buffer, zero serialization.
    if (replaceMode && g_lastBLAS) {
        void* key = getKey(cmdBuf);
        auto& disp = g_deviceMap[key];

        // Setup compute pipeline (once)
        bool useCompute = setupComputePipeline(disp) && uploadBVHData(disp, g_lastBLAS);

        // Find target storage images (HDR only)
        struct TargetImg {
            VkImage image; VkFormat format;
            int outFmt; int bpp;
        };
        std::vector<TargetImg> targets;
        {
            std::lock_guard<std::mutex> lock(g_lock);
            for (auto& [imgId, ti] : g_storageImages) {
                if (ti.width == width && ti.height == height) {
                    if (ti.format != VK_FORMAT_R32G32B32A32_SFLOAT &&
                        ti.format != VK_FORMAT_R16G16B16A16_SFLOAT) continue;
                    TargetImg t;
                    t.image = (VkImage)imgId;
                    t.format = ti.format;
                    t.outFmt = 3; t.bpp = 16;
                    if (ti.format == VK_FORMAT_R16G16B16A16_SFLOAT) { t.outFmt = 2; t.bpp = 8; }
                    targets.push_back(t);
                }
            }
        }

        if (targets.empty() && frameN <= 5) {
            LOG("  → No storage images matching %ux%u found!", width, height);
        }

        if (useCompute && !targets.empty()) {
            clock_gettime(CLOCK_MONOTONIC, &t1);

            // Compute camera basis
            float cx, cy, cz, extent;
            cudaBVH_getBounds(g_lastBLAS, &cx, &cy, &cz, &extent);
            float fcx = cx, fcy = cy, fcz = cz + extent * 1.2f;
            float fwdX = cx - fcx, fwdY = cy - fcy, fwdZ = cz - fcz;
            float flen = sqrtf(fwdX*fwdX + fwdY*fwdY + fwdZ*fwdZ);
            if (flen < 1e-6f) flen = 1.f;
            fwdX /= flen; fwdY /= flen; fwdZ /= flen;
            float rX = fwdZ, rY = 0, rZ = -fwdX;
            float rl = sqrtf(rX*rX + rZ*rZ);
            if (rl > 1e-6f) { rX /= rl; rZ /= rl; } else { rX = 1; rZ = 0; }
            float uX = rY*fwdZ - rZ*fwdY, uY = rZ*fwdX - rX*fwdZ, uZ = rX*fwdY - rY*fwdX;

            for (auto& tgt : targets) {
                // Create image view for this target (cached in map)
                static std::unordered_map<uint64_t, VkImageView> g_imageViews;
                VkImageView imgView = VK_NULL_HANDLE;
                auto ivIt = g_imageViews.find((uint64_t)tgt.image);
                if (ivIt != g_imageViews.end()) {
                    imgView = ivIt->second;
                } else {
                    VkImageViewCreateInfo ivCI = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                    ivCI.image = tgt.image;
                    ivCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    ivCI.format = tgt.format;
                    ivCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    auto createIV = (PFN_vkCreateImageView)disp.GetDeviceProcAddr(disp.device, "vkCreateImageView");
                    if (createIV && createIV(disp.device, &ivCI, nullptr, &imgView) == VK_SUCCESS) {
                        g_imageViews[(uint64_t)tgt.image] = imgView;
                    }
                }
                if (!imgView) continue;

                // Update output image descriptor
                updateOutputImageDescriptor(disp, imgView);

                // Image already in GENERAL from app's pre-trace barriers — perfect for storage image
                // Bind compute pipeline and dispatch
                disp.CmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, g_compute.pipeline);
                disp.CmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                    g_compute.pipeLayout, 0, 1, &g_compute.descSet, 0, nullptr);

                // Push constants: camera + resolution
                ComputeTracer::PushConstants pc = {};
                pc.camOx = fcx; pc.camOy = fcy; pc.camOz = fcz;
                pc.fwdX = fwdX; pc.fwdY = fwdY; pc.fwdZ = fwdZ;
                pc.rightX = rX; pc.rightY = rY; pc.rightZ = rZ;
                pc.upX = uX; pc.upY = uY; pc.upZ = uZ;
                pc.fov = 0.6f;
                pc.nearZ = fmaxf(flen - extent*0.7f, 0.1f);
                pc.farZ = flen + extent*0.7f;
                pc.width = (int)width; pc.height = (int)height;
                {
                    bool bvh2Mode = getenv("CUDA_RT_BVH2") && atoi(getenv("CUDA_RT_BVH2"));
                    if (bvh2Mode) {
                        uint32_t* tmp; pc.numNodes = cudaBVH_getStacklessBVH2(g_lastBLAS, &tmp);
                    } else {
                        pc.numNodes = cudaBVH_getNumBVH4Nodes(g_lastBLAS) * 4;
                    }
                }
                pc.outFmt = tgt.outFmt;

                disp.CmdPushConstants(cmdBuf, g_compute.pipeLayout,
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

                // Dispatch: match shader workgroup size
                uint32_t gx, gy;
                if (getenv("CUDA_RT_BVH2") && atoi(getenv("CUDA_RT_BVH2"))) {
                    gx = (width + 7) / 8; gy = (height + 7) / 8;  // 8×8 WG for stackless BVH2
                } else {
                    gx = (width + 3) / 4; gy = (height + 3) / 4;  // 4×4 WG for BVH4
                }
                disp.CmdDispatch(cmdBuf, gx, gy, 1);

                // Memory barrier: compute shader write → shader read
                VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                disp.CmdPipelineBarrier(cmdBuf,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    0, 1, &memBarrier, 0, nullptr, 0, nullptr);

                if (frameN <= 5 || (frameN % 200) == 0) {
                    LOG("  → Compute dispatch: %ux%u groups (%ux%u) image=0x%lx fmt=%d",
                        gx, gy, width, height, (uint64_t)tgt.image, tgt.format);
                }
            }
            clock_gettime(CLOCK_MONOTONIC, &t2);
            if (frameN <= 5 || (frameN % 200) == 0) {
                double ms = (t2.tv_sec-t1.tv_sec)*1e3 + (t2.tv_nsec-t1.tv_nsec)*1e-6;
                LOG("  → Compute trace: %.2fms (record time), targets=%zu", ms, targets.size());
            }
        }

        // Save PPM on frame 3 for debugging
        if (frameN == 3) {
            uint32_t* rgba = (uint32_t*)malloc(width * height * 4);
            if (rgba && cudaBVH_traceToRGBA(g_lastBLAS, width, height, camX, camY, camZ, rgba) == 0) {
                FILE* fp = fopen("/tmp/cuda_rt_frame.ppm", "wb");
                if (fp) {
                    fprintf(fp, "P6\n%d %d\n255\n", width, height);
                    for (int i = 0; i < (int)(width * height); i++) {
                        uint8_t r = rgba[i] & 0xFF;
                        uint8_t g = (rgba[i] >> 8) & 0xFF;
                        uint8_t b = (rgba[i] >> 16) & 0xFF;
                        fwrite(&r, 1, 1, fp); fwrite(&g, 1, 1, fp); fwrite(&b, 1, 1, fp);
                    }
                    fclose(fp);
                    LOG("  → Saved frame to /tmp/cuda_rt_frame.ppm (%dx%d)", width, height);
                }
            }
            free(rgba);
        }
    }
}

// ═══════════════════════════════════════════
// Intercepted: CmdTraceRaysIndirectKHR
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_CmdTraceRaysIndirectKHR(
    VkCommandBuffer cmdBuf,
    const VkStridedDeviceAddressRegionKHR* pRaygenSBT,
    const VkStridedDeviceAddressRegionKHR* pMissSBT,
    const VkStridedDeviceAddressRegionKHR* pHitSBT,
    const VkStridedDeviceAddressRegionKHR* pCallableSBT,
    VkDeviceAddress indirectDeviceAddress)
{
    LOG("CmdTraceRaysIndirect: addr=0x%lx → CUDA engine dispatch", (uint64_t)indirectDeviceAddress);

    void* key = getKey(cmdBuf);
    auto it = g_deviceMap.find(key);
    if (it != g_deviceMap.end() && it->second.CmdTraceRaysIndirectKHR) {
        it->second.CmdTraceRaysIndirectKHR(cmdBuf, pRaygenSBT, pMissSBT, pHitSBT, pCallableSBT, indirectDeviceAddress);
    }
}

// ═══════════════════════════════════════════
// Intercepted: CreateRayTracingPipelinesKHR
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateRayTracingPipelinesKHR(
    VkDevice device,
    VkDeferredOperationKHR deferredOp,
    VkPipelineCache pipelineCache,
    uint32_t createInfoCount,
    const VkRayTracingPipelineCreateInfoKHR* pCreateInfos,
    const VkAllocationCallbacks* pAllocator,
    VkPipeline* pPipelines)
{
    for (uint32_t i = 0; i < createInfoCount; i++) {
        LOG("CreateRTPipeline[%u]: stages=%u groups=%u maxRecursion=%u",
            i, pCreateInfos[i].stageCount, pCreateInfos[i].groupCount,
            pCreateInfos[i].maxPipelineRayRecursionDepth);
    }

    // V100 driver can crash trying to compile RT shaders (no HW RT cores).
    // Don't forward to driver — return dummy handles. CmdTraceRaysKHR is
    // intercepted by our CUDA BVH engine and doesn't use these pipelines.
    for (uint32_t i = 0; i < createInfoCount; i++)
        pPipelines[i] = (VkPipeline)(uint64_t)(0xCDA70000 + i);
    LOG("  → Created %u dummy RT pipeline handles (CUDA layer handles tracing)", createInfoCount);
    return VK_SUCCESS;
}

// ═══════════════════════════════════════════
// Intercepted: GetRayTracingShaderGroupHandlesKHR
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_GetRayTracingShaderGroupHandlesKHR(
    VkDevice device,
    VkPipeline pipeline,
    uint32_t firstGroup,
    uint32_t groupCount,
    size_t dataSize,
    void* pData)
{
    LOG("GetRTShaderGroupHandles: pipeline=%p first=%u count=%u size=%zu",
        (void*)pipeline, firstGroup, groupCount, dataSize);
    // Check if this is one of our dummy RT pipeline handles
    uint64_t pipeVal = (uint64_t)pipeline;
    if ((pipeVal & 0xFFFF0000ULL) == 0xCDA70000ULL) {
        // Dummy pipeline — return unique-ish handles per group
        // Each handle needs to be handleSize bytes (typically 32)
        // Fill with group index pattern so app can distinguish them
        uint8_t* dst = (uint8_t*)pData;
        memset(dst, 0, dataSize);
        size_t handleSize = dataSize / groupCount;
        for (uint32_t g = 0; g < groupCount; g++) {
            uint32_t idx = firstGroup + g;
            memcpy(dst + g * handleSize, &idx, sizeof(idx));
        }
        LOG("  → Returned %u dummy shader group handles (handleSize=%zu)", groupCount, handleSize);
        return VK_SUCCESS;
    }
    // Real pipeline — forward to driver
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];
    if (disp.GetRayTracingShaderGroupHandlesKHR)
        return disp.GetRayTracingShaderGroupHandlesKHR(device, pipeline, firstGroup, groupCount, dataSize, pData);
    memset(pData, 0, dataSize);
    return VK_SUCCESS;
}

// ═══════════════════════════════════════════
// Intercepted: GetAccelerationStructureDeviceAddressKHR
// ═══════════════════════════════════════════
static VKAPI_ATTR VkDeviceAddress VKAPI_CALL layer_GetAccelerationStructureDeviceAddressKHR(
    VkDevice device,
    const VkAccelerationStructureDeviceAddressInfoKHR* pInfo)
{
    // Check if this is one of our fake AS handles (< 0x10000 = fake from g_nextASHandle)
    uint64_t asVal = (uint64_t)pInfo->accelerationStructure;
    bool isFakeAS = (asVal >= 0x1000 && asVal < 0x100000);

    VkDeviceAddress addr = 0;
    if (!isFakeAS) {
        void* key = getKey(device);
        auto& disp = g_deviceMap[key];
        if (disp.GetAccelerationStructureDeviceAddressKHR)
            addr = disp.GetAccelerationStructureDeviceAddressKHR(device, pInfo);
    }
    if (addr == 0)
        addr = 0xDEAD000000000000ULL | asVal;

    static int addrLogCount = 0;
    if (addrLogCount < 40)
        LOG("GetASDeviceAddr: AS=0x%lx → devAddr=0x%lx%s", asVal, (uint64_t)addr,
            isFakeAS ? " (FAKE)" : "");
    addrLogCount++;

    // Track device address → AS handle mapping (for TLAS→BLAS resolution)
    {
        std::lock_guard<std::mutex> lock(g_lock);
        g_asDevAddrToHandle[addr] = asVal;
        // Also update multi-BLAS device address → index mapping
        auto it = g_asKeyToBLASIdx.find(asVal);
        if (it != g_asKeyToBLASIdx.end()) {
            g_blasDevAddrToIdx[addr] = it->second;
            if (addrLogCount <= 40)
                LOG("  → mapped to BLAS#%d", it->second);
        }
    }
    return addr;
}

// ═══════════════════════════════════════════
// Intercepted: CreatePipelineLayout — pass through (we extend in CreateComputePipelines)
// ═══════════════════════════════════════════
// We DON'T extend all layouts — that breaks the app's own descriptor set bindings.
// Instead we create an extended layout only for RQ compute pipelines.

static VKAPI_ATTR VkResult VKAPI_CALL layer_CreatePipelineLayout(
    VkDevice device,
    const VkPipelineLayoutCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkPipelineLayout* pPipelineLayout)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];
    VkResult result = disp.CreatePipelineLayout(device, pCreateInfo, pAllocator, pPipelineLayout);
    if (result == VK_SUCCESS && pPipelineLayout && *pPipelineLayout) {
        // Track the set layouts and push constants used in this pipeline layout
        PipelineLayoutInfo info;
        info.setLayouts.assign(pCreateInfo->pSetLayouts,
                               pCreateInfo->pSetLayouts + pCreateInfo->setLayoutCount);
        if (pCreateInfo->pushConstantRangeCount > 0 && pCreateInfo->pPushConstantRanges) {
            info.pushConstantRanges.assign(pCreateInfo->pPushConstantRanges,
                                            pCreateInfo->pPushConstantRanges + pCreateInfo->pushConstantRangeCount);
        }
        std::lock_guard<std::mutex> lock(g_pipelineLayoutMutex);
        g_pipelineLayouts[(uint64_t)*pPipelineLayout] = std::move(info);
    }
    return result;
}

// ═══════════════════════════════════════════
// Intercepted: CreateComputePipelines — track pipelines using rewritten shaders
// ═══════════════════════════════════════════
static VkDescriptorSetLayout g_emptyDSLayout = VK_NULL_HANDLE;

static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateComputePipelines(
    VkDevice device,
    VkPipelineCache pipelineCache,
    uint32_t createInfoCount,
    const VkComputePipelineCreateInfo* pCreateInfos,
    const VkAllocationCallbacks* pAllocator,
    VkPipeline* pPipelines)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];

    // Check which pipelines use RQ shaders and need extended layouts
    std::vector<VkComputePipelineCreateInfo> modInfos(pCreateInfos, pCreateInfos + createInfoCount);
    std::vector<VkPipelineLayout> extLayouts(createInfoCount, VK_NULL_HANDLE);
    std::vector<int> rqSet(createInfoCount, -1);

    setupBVH2Descriptors(disp);

    for (uint32_t i = 0; i < createInfoCount; i++) {
        uint64_t shaderHandle = (uint64_t)modInfos[i].stage.module;
        std::lock_guard<std::mutex> lock(g_rqShaderMutex);
        auto it = g_rqShaders.find(shaderHandle);
        if (it == g_rqShaders.end()) continue;
        if (!g_bvh2.dsLayout) continue;

        rqSet[i] = it->second.bvhDescSet;
        uint32_t bvhSet = (uint32_t)rqSet[i];

        // Create empty layout for padding (for sets the app doesn't use)
        if (!g_emptyDSLayout) {
            VkDescriptorSetLayoutCreateInfo emptyCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            emptyCI.bindingCount = 0;
            disp.CreateDescriptorSetLayout(device, &emptyCI, nullptr, &g_emptyDSLayout);
        }

        // Get the app's original pipeline layout's set layouts + push constants
        VkPipelineLayout origLayout = modInfos[i].layout;
        std::vector<VkDescriptorSetLayout> appSetLayouts;
        std::vector<VkPushConstantRange> appPushConstants;
        {
            std::lock_guard<std::mutex> lock2(g_pipelineLayoutMutex);
            auto plt = g_pipelineLayouts.find((uint64_t)origLayout);
            if (plt != g_pipelineLayouts.end()) {
                appSetLayouts = plt->second.setLayouts;
                appPushConstants = plt->second.pushConstantRanges;
            }
        }

        // Build extended layout: app's sets [0..N-1] + padding + BVH2 at bvhSet
        std::vector<VkDescriptorSetLayout> sets(bvhSet + 1, g_emptyDSLayout);
        // Copy app's original set layouts to maintain compatibility
        for (uint32_t s = 0; s < appSetLayouts.size() && s < bvhSet; s++) {
            sets[s] = appSetLayouts[s];
        }
        sets[bvhSet] = g_bvh2.dsLayout;

        VkPipelineLayoutCreateInfo plCI = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plCI.setLayoutCount = bvhSet + 1;
        plCI.pSetLayouts = sets.data();
        plCI.pushConstantRangeCount = (uint32_t)appPushConstants.size();
        plCI.pPushConstantRanges = appPushConstants.empty() ? nullptr : appPushConstants.data();

        if (disp.CreatePipelineLayout(device, &plCI, nullptr, &extLayouts[i]) != VK_SUCCESS) {
            LOG("[BVH2] Failed to create extended pipeline layout for RQ pipeline");
            extLayouts[i] = VK_NULL_HANDLE;
        } else {
            LOG("[BVH2] Created COMPATIBLE extended layout with %zu app sets, %zu push consts + BVH2 at set=%u",
                appSetLayouts.size(), appPushConstants.size(), bvhSet);
        }
    }

    // Create pipelines with EXTENDED layout for RQ pipelines so set 4 is accessible.
    // The extended layout includes the app's original sets [0..N-1] for compatibility,
    // plus our BVH2 SSBO set at index 4. Without this, the driver ignores set 4.
    static int noExtLayout = -1;
    if (noExtLayout < 0) {
        const char* e = getenv("CUDA_RT_NO_EXT_LAYOUT");
        noExtLayout = (e && atoi(e)) ? 1 : 0;
    }
    for (uint32_t i = 0; i < createInfoCount; i++) {
        if (extLayouts[i] && !noExtLayout) {
            modInfos[i].layout = extLayouts[i];
        }
    }
    VkResult result = disp.CreateComputePipelines(device, pipelineCache, createInfoCount,
                                                    modInfos.data(), pAllocator, pPipelines);

    if (result == VK_SUCCESS) {
        for (uint32_t i = 0; i < createInfoCount; i++) {
            if (rqSet[i] < 0 || !pPipelines[i]) continue;
            uint64_t pipeHandle = (uint64_t)pPipelines[i];
            std::lock_guard<std::mutex> lock2(g_rqPipelineMutex);
            g_rqPipelines[pipeHandle] = {extLayouts[i], rqSet[i]};
            LOG("CreateComputePipeline: tracked RQ pipeline 0x%lx (extLayout=0x%lx, bvhSet=%d)",
                pipeHandle, (uint64_t)extLayouts[i], rqSet[i]);
        }
    }

    return result;
}

// ═══════════════════════════════════════════
// Intercepted: DestroyPipeline — skip dummy RT pipeline handles
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_DestroyPipeline(
    VkDevice device,
    VkPipeline pipeline,
    const VkAllocationCallbacks* pAllocator)
{
    uint64_t pipeVal = (uint64_t)pipeline;
    if ((pipeVal & 0xFFFF0000ULL) == 0xCDA70000ULL) {
        LOG("DestroyPipeline: dummy RT pipeline 0x%lx (not forwarded)", pipeVal);
        return; // Our fake handle — don't forward to driver
    }
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];
    disp.DestroyPipeline(device, pipeline, pAllocator);
}

// ═══════════════════════════════════════════
// Intercepted: CreateDescriptorPool — strip AS descriptor type in stripped mode
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateDescriptorPool(
    VkDevice device,
    const VkDescriptorPoolCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDescriptorPool* pDescriptorPool)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];
    // Pass through — driver handles AS descriptor pools natively (driver 580+ has RT extensions)
    return disp.CreateDescriptorPool(device, pCreateInfo, pAllocator, pDescriptorPool);
}

// ═══════════════════════════════════════════
// Intercepted: CreateDescriptorSetLayout — strip AS binding type in stripped mode
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateDescriptorSetLayout(
    VkDevice device,
    const VkDescriptorSetLayoutCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDescriptorSetLayout* pSetLayout)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];
    // Pass through — driver handles AS bindings natively (driver 580+ has RT extensions)
    return disp.CreateDescriptorSetLayout(device, pCreateInfo, pAllocator, pSetLayout);
}

// ═══════════════════════════════════════════
// Intercepted: UpdateDescriptorSets — skip AS descriptor writes in stripped mode
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_UpdateDescriptorSets(
    VkDevice device,
    uint32_t descriptorWriteCount,
    const VkWriteDescriptorSet* pDescriptorWrites,
    uint32_t descriptorCopyCount,
    const VkCopyDescriptorSet* pDescriptorCopies)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];
    // Pass through - driver handles AS descriptor writes natively (real AS handles from CreateAS)
    disp.UpdateDescriptorSets(device, descriptorWriteCount, pDescriptorWrites,
                              descriptorCopyCount, pDescriptorCopies);
}

#if 0 // OLD: Filter out AS descriptor writes — our layer handles TLAS via injected SSBOs
    std::vector<VkWriteDescriptorSet> filtered;
    for (uint32_t i = 0; i < descriptorWriteCount; i++) {
        if (pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
            LOG("UpdateDescriptorSets: skipping AS descriptor write (binding=%u)",
                pDescriptorWrites[i].dstBinding);
            continue;
        }
        // For non-AS writes, strip pNext chain of AS-specific info
        VkWriteDescriptorSet w = pDescriptorWrites[i];
        // Check for VkWriteDescriptorSetAccelerationStructureKHR in pNext
        VkBaseOutStructure* prev = nullptr;
        VkBaseOutStructure* cur = (VkBaseOutStructure*)w.pNext;
        while (cur) {
            if (cur->sType == VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR) {
                if (prev) prev->pNext = cur->pNext;
                else w.pNext = cur->pNext;
            } else {
                prev = cur;
            }
            cur = cur->pNext;
        }
        filtered.push_back(w);
    }

    if (!filtered.empty() || descriptorCopyCount > 0) {
        disp.UpdateDescriptorSets(device, (uint32_t)filtered.size(), filtered.data(),
                                  descriptorCopyCount, pDescriptorCopies);
    }
}

#endif

// ═══════════════════════════════════════════
// Intercepted: CreateRenderPass — track attachment configuration
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateRenderPass(
    VkDevice device,
    const VkRenderPassCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkRenderPass* pRenderPass)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];
    VkResult res = disp.CreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass);
    if (res == VK_SUCCESS && pRenderPass) {
        RenderPassInfo info = {};
        info.depthAttachIdx = UINT32_MAX;
        info.motionVectorIdx = -1;
        memset(info.colorFormats, 0, sizeof(info.colorFormats));
        uint32_t colorIdx = 0;
        // Count color vs depth attachments, track formats for G-buffer
        for (uint32_t i = 0; i < pCreateInfo->attachmentCount; i++) {
            VkFormat fmt = pCreateInfo->pAttachments[i].format;
            if (fmt == VK_FORMAT_D16_UNORM || fmt == VK_FORMAT_D32_SFLOAT ||
                fmt == VK_FORMAT_D16_UNORM_S8_UINT || fmt == VK_FORMAT_D24_UNORM_S8_UINT ||
                fmt == VK_FORMAT_D32_SFLOAT_S8_UINT) {
                info.depthAttachment = 1;
                info.depthFormat = fmt;
                info.depthAttachIdx = i;
            } else {
                if (colorIdx < 8) info.colorFormats[colorIdx] = fmt;
                // Detect motion vectors: R16G16_SFLOAT or R16G16_SNORM
                if (fmt == VK_FORMAT_R16G16_SFLOAT || fmt == VK_FORMAT_R16G16_SNORM) {
                    info.motionVectorIdx = (int)i;
                }
                info.colorAttachments++;
                colorIdx++;
            }
        }
        info.subpassCount = pCreateInfo->subpassCount;
        g_renderPassInfo[(uint64_t)*pRenderPass] = info;

        // Log detailed attachment info: format, samples, loadOp, storeOp, layouts
        for (uint32_t i = 0; i < pCreateInfo->attachmentCount; i++) {
            auto& att = pCreateInfo->pAttachments[i];
            LOG("[PROFILE] CreateRenderPass 0x%lx att[%u]: fmt=%u samples=%u load=%u store=%u initL=%u finalL=%u",
                (uint64_t)*pRenderPass, i, att.format, att.samples, att.loadOp, att.storeOp,
                att.initialLayout, att.finalLayout);
        }
        // Log subpass resolve attachments
        for (uint32_t s = 0; s < pCreateInfo->subpassCount; s++) {
            auto& sub = pCreateInfo->pSubpasses[s];
            if (sub.pResolveAttachments) {
                for (uint32_t r = 0; r < sub.colorAttachmentCount; r++) {
                    if (sub.pResolveAttachments[r].attachment != VK_ATTACHMENT_UNUSED)
                        LOG("[PROFILE] CreateRenderPass 0x%lx subpass[%u] resolve[%u]: att=%u layout=%u",
                            (uint64_t)*pRenderPass, s, r, sub.pResolveAttachments[r].attachment,
                            sub.pResolveAttachments[r].layout);
                }
            }
        }
        // Log subpass dependencies
        for (uint32_t d = 0; d < pCreateInfo->dependencyCount; d++) {
            auto& dep = pCreateInfo->pDependencies[d];
            LOG("[PROFILE] CreateRenderPass 0x%lx dep[%u]: src=%u dst=%u srcStage=0x%x dstStage=0x%x srcAcc=0x%x dstAcc=0x%x flags=0x%x",
                (uint64_t)*pRenderPass, d, dep.srcSubpass, dep.dstSubpass,
                dep.srcStageMask, dep.dstStageMask, dep.srcAccessMask, dep.dstAccessMask, dep.dependencyFlags);
        }
        LOG("[PROFILE] CreateRenderPass 0x%lx: %u color + %u depth, %u subpasses",
            (uint64_t)*pRenderPass, info.colorAttachments, info.depthAttachment, info.subpassCount);
    }
    return res;
}

// ═══════════════════════════════════════════
// G-Buffer: Intercept CreateImageView — track VkImageView → VkImage mapping
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateImageView(
    VkDevice device, const VkImageViewCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkImageView* pView)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];
    // No dedicated CreateImageView in dispatch — use GetDeviceProcAddr
    auto fn = (PFN_vkCreateImageView)disp.GetDeviceProcAddr(device, "vkCreateImageView");
    if (!fn) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult res = fn(device, pCreateInfo, pAllocator, pView);
    if (res == VK_SUCCESS && pView && pCreateInfo) {
        std::lock_guard<std::mutex> lock(g_lock);
        g_imageViewInfo[(uint64_t)*pView] = {
            pCreateInfo->image,
            pCreateInfo->format,
            pCreateInfo->subresourceRange.baseMipLevel,
            pCreateInfo->subresourceRange.baseArrayLayer
        };
    }
    return res;
}

static VKAPI_ATTR void VKAPI_CALL layer_DestroyImageView(
    VkDevice device, VkImageView view, const VkAllocationCallbacks* pAllocator)
{
    {
        std::lock_guard<std::mutex> lock(g_lock);
        g_imageViewInfo.erase((uint64_t)view);
    }
    void* key = getKey(device);
    auto fn = (PFN_vkDestroyImageView)g_deviceMap[key].GetDeviceProcAddr(device, "vkDestroyImageView");
    if (fn) fn(device, view, pAllocator);
}

// ═══════════════════════════════════════════
// G-Buffer: Intercept CreateFramebuffer — track attachment image views
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateFramebuffer(
    VkDevice device, const VkFramebufferCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkFramebuffer* pFramebuffer)
{
    void* key = getKey(device);
    auto& disp = g_deviceMap[key];
    auto fn = (PFN_vkCreateFramebuffer)disp.GetDeviceProcAddr(device, "vkCreateFramebuffer");
    if (!fn) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult res = fn(device, pCreateInfo, pAllocator, pFramebuffer);
    if (res == VK_SUCCESS && pFramebuffer && pCreateInfo &&
        !(pCreateInfo->flags & VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT)) {
        std::lock_guard<std::mutex> lock(g_lock);
        FramebufferInfo fb;
        fb.width = pCreateInfo->width;
        fb.height = pCreateInfo->height;
        fb.renderPass = pCreateInfo->renderPass;
        for (uint32_t i = 0; i < pCreateInfo->attachmentCount; i++) {
            fb.attachments.push_back(pCreateInfo->pAttachments[i]);
        }
        g_framebufferInfo[(uint64_t)*pFramebuffer] = std::move(fb);
        LOG("[G-Buffer] Framebuffer 0x%lx: %ux%u, %u attachments",
            (uint64_t)*pFramebuffer, fb.width, fb.height, pCreateInfo->attachmentCount);
    }
    return res;
}

static VKAPI_ATTR void VKAPI_CALL layer_DestroyFramebuffer(
    VkDevice device, VkFramebuffer fb, const VkAllocationCallbacks* pAllocator)
{
    {
        std::lock_guard<std::mutex> lock(g_lock);
        g_framebufferInfo.erase((uint64_t)fb);
    }
    void* key = getKey(device);
    auto fn = (PFN_vkDestroyFramebuffer)g_deviceMap[key].GetDeviceProcAddr(device, "vkDestroyFramebuffer");
    if (fn) fn(device, fb, pAllocator);
}
// Draw call counting per render pass (diagnostic)
static thread_local uint32_t g_currentRP = UINT32_MAX;
static thread_local uint32_t g_drawsInRP[8] = {};
static thread_local uint32_t g_logRPIdx = 0;
static uint64_t g_drawCountFrame = 0;
static uint64_t g_logFrame = 0;

// ═══════════════════════════════════════════
// RasterBoost Phase 3: Draw call batching state
// Accumulates consecutive CmdDrawIndexed calls for potential MDI merge
// ═══════════════════════════════════════════
struct DrawBatchState {
    bool     enabled;
    // VkDrawIndexedIndirectCommand = { indexCount, instanceCount, firstIndex, vertexOffset, firstInstance }
    struct DrawCmd {
        uint32_t indexCount;
        uint32_t instanceCount;
        uint32_t firstIndex;
        int32_t  vertexOffset;
        uint32_t firstInstance;
    };
    std::vector<DrawCmd> pending;
    VkPipeline lastPipeline;      // Pipeline state when draws were recorded
    uint32_t   batchedDraws;      // Total draws batched this frame
    uint32_t   savedDraws;        // Total draw calls saved via batching
    VkBuffer   indirectBuf;       // GPU buffer for indirect draw commands
    VkDeviceMemory indirectMem;
    VkDeviceSize indirectBufSize;
    bool       bufReady;
};
static thread_local DrawBatchState g_drawBatch = {};
static std::atomic<uint64_t> g_totalBatchedDraws{0};
static std::atomic<uint64_t> g_totalSavedDraws{0};

static void drawBatchInit() {
    const char* env = getenv("RASTER_BOOST_BATCH");
    g_drawBatch.enabled = (env && atoi(env) > 0);
    if (g_drawBatch.enabled) {
        fprintf(stderr, "[CudaRT] [RasterBoost] Draw call batching enabled\n");
    }
}

// Flush accumulated draws as a single MDI call (or individual if only 1)
static void drawBatchFlush(VkCommandBuffer cmdBuf, DeviceDispatch& disp) {
    if (g_drawBatch.pending.empty()) return;

    if (g_drawBatch.pending.size() == 1) {
        // Single draw — emit directly, no batching overhead
        auto& d = g_drawBatch.pending[0];
        disp.CmdDrawIndexed(cmdBuf, d.indexCount, d.instanceCount,
                            d.firstIndex, d.vertexOffset, d.firstInstance);
    } else if (g_drawBatch.pending.size() >= 2) {
        // Multiple draws — merge into CmdDrawIndexedIndirect if buffer available
        // For now, emit individually but track savings potential
        for (auto& d : g_drawBatch.pending) {
            disp.CmdDrawIndexed(cmdBuf, d.indexCount, d.instanceCount,
                                d.firstIndex, d.vertexOffset, d.firstInstance);
        }
        // Track statistics
        g_drawBatch.batchedDraws += (uint32_t)g_drawBatch.pending.size();
        // When indirect buffer is allocated, this becomes:
        // Upload pending to indirectBuf, emit single CmdDrawIndexedIndirect
    }
    g_drawBatch.pending.clear();
}

static VKAPI_ATTR void VKAPI_CALL layer_CmdBeginRenderPass(
    VkCommandBuffer cmdBuf,
    const VkRenderPassBeginInfo* pRenderPassBegin,
    VkSubpassContents contents)
{
    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];

    // Write GPU timestamp before render pass
    if (g_profile.ready && g_profile.queryPool && g_profile.queryIdx < 62) {
        uint32_t qi = g_profile.queryIdx;
        disp.CmdResetQueryPool(cmdBuf, g_profile.queryPool, qi, 2);
        disp.CmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               g_profile.queryPool, qi);

        FrameProfile::PassTiming pt = {};
        pt.renderPass = (uint64_t)pRenderPassBegin->renderPass;
        pt.startQuery = qi;
        pt.endQuery = qi + 1;
        pt.width = pRenderPassBegin->renderArea.extent.width;
        pt.height = pRenderPassBegin->renderArea.extent.height;

        auto rpIt = g_renderPassInfo.find((uint64_t)pRenderPassBegin->renderPass);
        if (rpIt != g_renderPassInfo.end())
            pt.colorAttachments = rpIt->second.colorAttachments;

        g_profile.passes.push_back(pt);
        g_profile.queryIdx += 2;
        g_profile.rpCount++;
    }

    // Track current render pass index for draw call counting
    g_currentRP = g_profile.rpCount > 0 ? g_profile.rpCount - 1 : 0;
    if (g_currentRP < 8) g_drawsInRP[g_currentRP] = 0;

    // Log subpass contents type for diagnostic
    if (g_logFrame < 20) {
        const char* cstr = (contents == VK_SUBPASS_CONTENTS_INLINE) ? "INLINE" : "SECONDARY_CMD_BUFS";
        fprintf(stderr, "[CudaRT] Frame %lu RP%u: contents=%s\n",
                (unsigned long)g_logFrame, g_logRPIdx, cstr);
    }
    g_logRPIdx++;

    // G-Buffer: identify depth/MV images from framebuffer for this render pass
    if (g_rasterBoost.active && pRenderPassBegin) {
        std::lock_guard<std::mutex> lock(g_lock);
        auto rpIt = g_renderPassInfo.find((uint64_t)pRenderPassBegin->renderPass);
        auto fbIt = g_framebufferInfo.find((uint64_t)pRenderPassBegin->framebuffer);
        if (rpIt != g_renderPassInfo.end() && fbIt != g_framebufferInfo.end()) {
            auto& rp = rpIt->second;
            auto& fb = fbIt->second;
            // Capture depth image
            if (rp.depthAttachIdx != UINT32_MAX && rp.depthAttachIdx < fb.attachments.size()) {
                auto viewIt = g_imageViewInfo.find((uint64_t)fb.attachments[rp.depthAttachIdx]);
                if (viewIt != g_imageViewInfo.end()) {
                    g_gbuffer.depthImage  = viewIt->second.image;
                    g_gbuffer.depthFormat = rp.depthFormat;
                    g_gbuffer.width  = pRenderPassBegin->renderArea.extent.width;
                    g_gbuffer.height = pRenderPassBegin->renderArea.extent.height;
                }
            }
            // Capture motion vector image
            if (rp.motionVectorIdx >= 0 && (uint32_t)rp.motionVectorIdx < fb.attachments.size()) {
                auto viewIt = g_imageViewInfo.find((uint64_t)fb.attachments[rp.motionVectorIdx]);
                if (viewIt != g_imageViewInfo.end()) {
                    g_gbuffer.motionImage  = viewIt->second.image;
                    g_gbuffer.motionFormat = rp.colorFormats[rp.motionVectorIdx];
                }
            }
        }
    }

    disp.CmdBeginRenderPass(cmdBuf, pRenderPassBegin, contents);

    // Mid-RP timestamp: for RP2, add a BOTTOM_OF_PIPE timestamp right after begin
    // This tells us if the stall is at render pass begin (load/clear) or at end
    if (g_profile.ready && g_profile.queryPool && g_currentRP == 2 &&
        g_profile.queryIdx < 62) {
        uint32_t qi = g_profile.queryIdx;
        disp.CmdResetQueryPool(cmdBuf, g_profile.queryPool, qi, 1);
        disp.CmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                               g_profile.queryPool, qi);
        g_profile.rp2MidQuery = qi;
        g_profile.queryIdx += 1;
    }
}

static VKAPI_ATTR void VKAPI_CALL layer_CmdDrawIndexed(
    VkCommandBuffer cmdBuf, uint32_t indexCount, uint32_t instanceCount,
    uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
{
    if (g_currentRP < 8) g_drawsInRP[g_currentRP]++;
    // Explicit diagnostic: log ALL draws in RP2
    static uint64_t rp2DrawLog = 0;
    if (g_currentRP == 2 && rp2DrawLog < 10) {
        rp2DrawLog++;
        LOG("!!! CmdDrawIndexed in RP2: idx=%u inst=%u", indexCount, instanceCount);
    }
    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];
    // For RP2 (the bottleneck): optionally reduce draw calls
    static int rpSkip = -1;
    if (rpSkip < 0) { const char* e = getenv("CUDA_RT_SKIP_RP"); rpSkip = e ? atoi(e) : -1; }
    if (rpSkip >= 0 && g_currentRP == (uint32_t)rpSkip) return;

    // RasterBoost: accumulate draws for batching when enabled
    if (g_drawBatch.enabled && g_currentRP != UINT32_MAX) {
        g_drawBatch.pending.push_back({indexCount, instanceCount, firstIndex, vertexOffset, firstInstance});
        return;  // Deferred — will flush at pipeline change, RP end, or batch limit
    }

    disp.CmdDrawIndexed(cmdBuf, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

static VKAPI_ATTR void VKAPI_CALL layer_CmdDraw(
    VkCommandBuffer cmdBuf, uint32_t vertexCount, uint32_t instanceCount,
    uint32_t firstVertex, uint32_t firstInstance)
{
    if (g_currentRP < 8) g_drawsInRP[g_currentRP]++;
    static uint64_t rp2Log = 0;
    if (g_currentRP == 2 && rp2Log < 10) {
        rp2Log++;
        LOG("!!! CmdDraw in RP2: vtx=%u inst=%u", vertexCount, instanceCount);
    }
    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];
    static int rpSkip = -1;
    if (rpSkip < 0) { const char* e = getenv("CUDA_RT_SKIP_RP"); rpSkip = e ? atoi(e) : -1; }
    if (rpSkip >= 0 && g_currentRP == (uint32_t)rpSkip) return;
    disp.CmdDraw(cmdBuf, vertexCount, instanceCount, firstVertex, firstInstance);
}

static VKAPI_ATTR void VKAPI_CALL layer_CmdDrawIndexedIndirect(
    VkCommandBuffer cmdBuf, VkBuffer buffer, VkDeviceSize offset,
    uint32_t drawCount, uint32_t stride)
{
    if (g_currentRP < 8) g_drawsInRP[g_currentRP] += drawCount;
    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];
    static int rpSkip = -1;
    if (rpSkip < 0) { const char* e = getenv("CUDA_RT_SKIP_RP"); rpSkip = e ? atoi(e) : -1; }
    if (rpSkip >= 0 && g_currentRP == (uint32_t)rpSkip) return;
    disp.CmdDrawIndexedIndirect(cmdBuf, buffer, offset, drawCount, stride);
}

static VKAPI_ATTR void VKAPI_CALL layer_CmdDrawIndirect(
    VkCommandBuffer cmdBuf, VkBuffer buffer, VkDeviceSize offset,
    uint32_t drawCount, uint32_t stride)
{
    if (g_currentRP < 8) g_drawsInRP[g_currentRP] += drawCount;
    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];
    static int rpSkip = -1;
    if (rpSkip < 0) { const char* e = getenv("CUDA_RT_SKIP_RP"); rpSkip = e ? atoi(e) : -1; }
    if (rpSkip >= 0 && g_currentRP == (uint32_t)rpSkip) return;
    disp.CmdDrawIndirect(cmdBuf, buffer, offset, drawCount, stride);
}

// ═══════════════════════════════════════════
// Intercepted: CmdDrawIndexedIndirectCount — GPU-driven draw count
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_CmdDrawIndexedIndirectCount(
    VkCommandBuffer cmdBuf, VkBuffer buffer, VkDeviceSize offset,
    VkBuffer countBuffer, VkDeviceSize countBufferOffset,
    uint32_t maxDrawCount, uint32_t stride)
{
    if (g_currentRP < 8) g_drawsInRP[g_currentRP] += maxDrawCount;
    static uint64_t rp2Log = 0;
    if (g_currentRP == 2 && rp2Log < 20) {
        rp2Log++;
        LOG("!!! CmdDrawIndexedIndirectCount in RP2: maxDraw=%u", maxDrawCount);
    }
    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];
    static int rpSkip = -1;
    if (rpSkip < 0) { const char* e = getenv("CUDA_RT_SKIP_RP"); rpSkip = e ? atoi(e) : -1; }
    if (rpSkip >= 0 && g_currentRP == (uint32_t)rpSkip) return;
    // Cap maxDrawCount for performance tuning (CUDA_RT_MAX_DRAW env var)
    static int maxDrawCap = -2;
    if (maxDrawCap == -2) { const char* e = getenv("CUDA_RT_MAX_DRAW"); maxDrawCap = e ? atoi(e) : -1; }
    uint32_t capped = maxDrawCount;
    if (maxDrawCap > 0 && capped > (uint32_t)maxDrawCap) capped = (uint32_t)maxDrawCap;
    disp.CmdDrawIndexedIndirectCount(cmdBuf, buffer, offset, countBuffer, countBufferOffset, capped, stride);
}

static VKAPI_ATTR void VKAPI_CALL layer_CmdDrawIndirectCount(
    VkCommandBuffer cmdBuf, VkBuffer buffer, VkDeviceSize offset,
    VkBuffer countBuffer, VkDeviceSize countBufferOffset,
    uint32_t maxDrawCount, uint32_t stride)
{
    if (g_currentRP < 8) g_drawsInRP[g_currentRP] += maxDrawCount;
    static uint64_t rp2Log = 0;
    if (g_currentRP == 2 && rp2Log < 20) {
        rp2Log++;
        LOG("!!! CmdDrawIndirectCount in RP2: maxDraw=%u", maxDrawCount);
    }
    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];
    static int rpSkip = -1;
    if (rpSkip < 0) { const char* e = getenv("CUDA_RT_SKIP_RP"); rpSkip = e ? atoi(e) : -1; }
    if (rpSkip >= 0 && g_currentRP == (uint32_t)rpSkip) return;
    static int maxDrawCap2 = -2;
    if (maxDrawCap2 == -2) { const char* e = getenv("CUDA_RT_MAX_DRAW"); maxDrawCap2 = e ? atoi(e) : -1; }
    uint32_t capped = maxDrawCount;
    if (maxDrawCap2 > 0 && capped > (uint32_t)maxDrawCap2) capped = (uint32_t)maxDrawCap2;
    disp.CmdDrawIndirectCount(cmdBuf, buffer, offset, countBuffer, countBufferOffset, capped, stride);
}

// ═══════════════════════════════════════════
// Intercepted: CmdWaitEvents — check for event waits inside render passes
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_CmdWaitEvents(
    VkCommandBuffer cmdBuf,
    uint32_t eventCount,
    const VkEvent* pEvents,
    VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask,
    uint32_t memoryBarrierCount,
    const VkMemoryBarrier* pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier* pBufferMemoryBarriers,
    uint32_t imageMemoryBarrierCount,
    const VkImageMemoryBarrier* pImageMemoryBarriers)
{
    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];

    LOG("CmdWaitEvents: RP%u events=%u src=0x%x dst=0x%x", g_currentRP, eventCount, srcStageMask, dstStageMask);

    disp.CmdWaitEvents(cmdBuf, eventCount, pEvents, srcStageMask, dstStageMask,
                       memoryBarrierCount, pMemoryBarriers,
                       bufferMemoryBarrierCount, pBufferMemoryBarriers,
                       imageMemoryBarrierCount, pImageMemoryBarriers);
}

// ═══════════════════════════════════════════
// Intercepted: CmdExecuteCommands — count secondary command buffers
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_CmdExecuteCommands(
    VkCommandBuffer cmdBuf, uint32_t commandBufferCount,
    const VkCommandBuffer* pCommandBuffers)
{
    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];

    static uint64_t execFrame = 0;
    execFrame++;
    if (execFrame <= 10 || execFrame % 300 == 0) {
        LOG("CmdExecuteCommands: RP%u, %u secondary bufs", g_currentRP, commandBufferCount);
    }
    if (g_currentRP < 8) g_drawsInRP[g_currentRP] += commandBufferCount; // count as draws

    disp.CmdExecuteCommands(cmdBuf, commandBufferCount, pCommandBuffers);
}

// ═══════════════════════════════════════════
// Intercepted: CmdEndRenderPass — GPU timestamp end
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_CmdEndRenderPass(
    VkCommandBuffer cmdBuf)
{
    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];

    // Pre-end timestamp (inside RP, before EndRenderPass) for RP2 diagnosis
    uint32_t savedRP = g_currentRP; // save before we reset it
    
    // Log draw counts per render pass (first few frames + periodic)
    g_drawCountFrame++;
    if (g_currentRP < 8 && (g_drawCountFrame <= 10 || g_drawCountFrame % 300 == 0)) {
        if (g_drawsInRP[g_currentRP] > 0)
            LOG("[PROFILE] RP%u ended: %u draw calls", g_currentRP, g_drawsInRP[g_currentRP]);
    }
    g_currentRP = UINT32_MAX;

    // Reset per-frame RP counter
    if (g_logRPIdx >= 5) { g_logRPIdx = 0; g_logFrame++; }

    // For RP2: write timestamp INSIDE the render pass, BEFORE ending it
    if (g_profile.ready && g_profile.queryPool && savedRP == 2 && g_profile.queryIdx < 62) {
        uint32_t qi = g_profile.queryIdx;
        disp.CmdResetQueryPool(cmdBuf, g_profile.queryPool, qi, 1);
        disp.CmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                               g_profile.queryPool, qi);
        g_profile.rp2PreEndQuery = qi;
        g_profile.queryIdx += 1;
    }

    // RasterBoost: flush pending draw batch before ending render pass
    if (g_drawBatch.enabled && !g_drawBatch.pending.empty()) {
        drawBatchFlush(cmdBuf, disp);
    }

    disp.CmdEndRenderPass(cmdBuf);

    // Write GPU timestamp after render pass
    if (g_profile.ready && g_profile.queryPool && !g_profile.passes.empty()) {
        auto& lastPass = g_profile.passes.back();
        if (lastPass.endQuery < 64) {
            disp.CmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                   g_profile.queryPool, lastPass.endQuery);
        }
    }

    // RasterBoost: Detect post-FX passes for async compute overlap
    // Heuristic: fullscreen quad pass = 1-3 draws, no depth, matching output res
    if (g_rasterBoost.active && savedRP < 8 && g_drawsInRP[savedRP] <= 3 &&
        g_drawsInRP[savedRP] > 0) {
        // Check if this was a post-FX pass (no depth, low draw count)
        if (!g_profile.passes.empty()) {
            auto& pt = g_profile.passes.back();
            auto rpIt = g_renderPassInfo.find(pt.renderPass);
            if (rpIt != g_renderPassInfo.end() && rpIt->second.depthAttachment == 0 &&
                rpIt->second.colorAttachments <= 2) {
                // This is likely a post-FX pass — tag for potential async reroute
                static uint64_t postfxCount = 0;
                postfxCount++;
                if (postfxCount <= 5 || postfxCount % 500 == 0) {
                    LOG("[RasterBoost:Async] Post-FX candidate: RP%u, %u draws, %u color, no depth",
                        savedRP, g_drawsInRP[savedRP], rpIt->second.colorAttachments);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════
// Intercepted: CmdBindPipeline — track compute pipeline + RT no-op
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_CmdBindPipeline(
    VkCommandBuffer cmdBuf,
    VkPipelineBindPoint bindPoint,
    VkPipeline pipeline)
{
    if (bindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR) [[unlikely]] {
        return; // Don't forward — we handle RT entirely
    }

    // RasterBoost: flush pending draw batch on pipeline change
    if (g_drawBatch.enabled && bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS &&
        !g_drawBatch.pending.empty()) {
        void* flushKey = getKey(cmdBuf);
        auto& flushDisp = g_deviceMap[flushKey];
        drawBatchFlush(cmdBuf, flushDisp);
    }
    // RP2 diagnostic: log graphics pipeline binds
    static uint64_t rp2BindLog = 0;
    if (g_currentRP == 2 && rp2BindLog < 10) {
        rp2BindLog++;
        LOG("!!! CmdBindPipeline in RP2: bind=%d pipe=0x%lx", bindPoint, (uint64_t)pipeline);
    }
    // Lock-free compute pipeline tracking: use atomic store
    // Only compute pipelines need tracking (for BVH desc binding in CmdDispatch)
    if (bindPoint == VK_PIPELINE_BIND_POINT_COMPUTE) {
        // Use thread-local-ish approach: store last bound pipeline per cmdBuf.
        // Since command buffers are single-threaded by spec, no mutex needed
        // for the write. Readers in CmdDispatch use the same cmdBuf → same thread.
        g_cmdBufBoundPipeline[(uint64_t)cmdBuf] = (uint64_t)pipeline; // lock-free: same thread
    }
    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];
    disp.CmdBindPipeline(cmdBuf, bindPoint, pipeline);
}

// ═══════════════════════════════════════════
// Intercepted: CmdBindDescriptorSets — eat RT bind point
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_CmdBindDescriptorSets(
    VkCommandBuffer cmdBuf,
    VkPipelineBindPoint bindPoint,
    VkPipelineLayout layout,
    uint32_t firstSet,
    uint32_t descriptorSetCount,
    const VkDescriptorSet* pDescriptorSets,
    uint32_t dynamicOffsetCount,
    const uint32_t* pDynamicOffsets)
{
    if (bindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR) [[unlikely]] {
        LOG("CmdBindDescriptorSets: RT bindpoint set=%u count=%u (intercepted)", firstSet, descriptorSetCount);
        return; // Don't forward — driver doesn't support RT bind point in stripped mode
    }
    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];
    disp.CmdBindDescriptorSets(cmdBuf, bindPoint, layout, firstSet, descriptorSetCount,
                               pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
}

// ═══════════════════════════════════════════
// Intercepted: CmdPipelineBarrier — remap RT stage flags
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_CmdPipelineBarrier(
    VkCommandBuffer cmdBuf,
    VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask,
    VkDependencyFlags dependencyFlags,
    uint32_t memoryBarrierCount,
    const VkMemoryBarrier* pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier* pBufferMemoryBarriers,
    uint32_t imageMemoryBarrierCount,
    const VkImageMemoryBarrier* pImageMemoryBarriers)
{
    // Replace RT shader stage bits with compute shader stage bits
    const VkPipelineStageFlags RT_STAGE = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    const VkPipelineStageFlags COMPUTE_STAGE = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags src = srcStageMask;
    VkPipelineStageFlags dst = dstStageMask;
    if (src & RT_STAGE) { src = (src & ~RT_STAGE) | COMPUTE_STAGE; }
    if (dst & RT_STAGE) { dst = (dst & ~RT_STAGE) | COMPUTE_STAGE; }

    // Diagnostic: log barriers (especially inside render passes)
    static uint64_t barrierCount = 0;
    barrierCount++;
    if (barrierCount <= 50 || barrierCount % 300 == 0) {
        LOG("CmdPipelineBarrier: RP%u src=0x%x dst=0x%x mem=%u buf=%u img=%u dep=0x%x",
            g_currentRP, src, dst, memoryBarrierCount,
            bufferMemoryBarrierCount, imageMemoryBarrierCount, dependencyFlags);
        for (uint32_t i = 0; i < imageMemoryBarrierCount && i < 3; i++) {
            LOG("  imgBarrier[%u]: srcAccess=0x%x dstAccess=0x%x oldL=%u newL=%u",
                i, pImageMemoryBarriers[i].srcAccessMask, pImageMemoryBarriers[i].dstAccessMask,
                pImageMemoryBarriers[i].oldLayout, pImageMemoryBarriers[i].newLayout);
        }
    }

    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];
    disp.CmdPipelineBarrier(cmdBuf, src, dst, dependencyFlags,
                            memoryBarrierCount, pMemoryBarriers,
                            bufferMemoryBarrierCount, pBufferMemoryBarriers,
                            imageMemoryBarrierCount, pImageMemoryBarriers);
}

// ═══════════════════════════════════════════
// Intercepted: CmdDispatch — bind BVH2 descriptor set for RQ pipelines
// ═══════════════════════════════════════════
// ═══════════════════════════════════════════
// Intercepted: QueueSubmit — process deferred BLAS builds after GPU execution
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_QueueSubmit(
    VkQueue queue,
    uint32_t submitCount,
    const VkSubmitInfo* pSubmits,
    VkFence fence)
{
    void* key = getKey(queue);
    auto& disp = g_deviceMap[key];

    // Fast path: after BLAS builds are done, pass through with zero overhead
    if (g_blasBuildsDone) {
        return disp.QueueSubmit(queue, submitCount, pSubmits, fence);
    }

    // Call real submit first
    VkResult res = disp.QueueSubmit(queue, submitCount, pSubmits, fence);
    if (res != VK_SUCCESS) return res;

    static int qsCount = 0;
    qsCount++;
    if (qsCount <= 10 || (qsCount % 100) == 0)
        LOG("[QS] QueueSubmit #%d (TLAS gen=%lu pending=%d addr=0x%lx cnt=%u)",
            qsCount, g_tlasGeneration, g_pendingTLAS.pending ? 1 : 0,
            (uint64_t)g_pendingTLAS.instanceAddr, g_pendingTLAS.instanceCount);

    // If we have pending BLAS builds, wait for GPU to finish then read vertex data
    bool hasPending = false;
    bool hasTLAS = false;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        hasPending = !g_pendingBLAS.empty() && !g_blasBuildsDone;
        hasTLAS = g_pendingTLAS.pending;
    }

    // Periodic TLAS retry: if TLAS gen is 0 but we have the address, re-read on later submits
    if (!hasTLAS && !hasPending && g_tlasGeneration == 0 &&
        g_pendingTLAS.instanceAddr != 0 && g_pendingTLAS.instanceCount > 0 &&
        qsCount > 3 && (qsCount % 5) == 0 && qsCount < 200) {
        // Force a retry
        g_pendingTLAS.pending = true;
        hasTLAS = true;
        LOG("[TLAS] Periodic retry at QueueSubmit #%d (gen still 0)", qsCount);
    }

    // TLAS: never process from QueueSubmit — it causes blocking stalls
    // Clear pending flag immediately; QueuePresent will handle TLAS once BVH2 is ready
    if (hasTLAS) {
        std::lock_guard<std::mutex> lock(g_lock);
        // Don't clear pending — let QueuePresent detect it
    }

    if (hasPending) {
        // Wait for this submission to complete so GPU buffers are filled
        disp.QueueWaitIdle(queue);
        processPendingBLAS();
    }
    // Clear TLAS pending without stalling — will be processed from QueuePresent
    if (hasTLAS) {
        std::lock_guard<std::mutex> lock(g_lock);
        g_pendingTLAS.pending = false;
    }

    return res;
}

// ═══════════════════════════════════════════
// Intercepted: QueueSubmit2KHR — same deferred logic as QueueSubmit
// ═══════════════════════════════════════════
static VKAPI_ATTR VkResult VKAPI_CALL layer_QueueSubmit2KHR(
    VkQueue queue,
    uint32_t submitCount,
    const VkSubmitInfo2KHR* pSubmits,
    VkFence fence)
{
    void* key = getKey(queue);
    auto& disp = g_deviceMap[key];

    // Call real submit2 first
    VkResult res = disp.QueueSubmit2KHR(queue, submitCount, pSubmits, fence);
    if (res != VK_SUCCESS) return res;

    static int qs2Count = 0;
    qs2Count++;
    if (qs2Count <= 5 || (qs2Count % 200) == 0)
        LOG("[QS2] QueueSubmit2 #%d (TLAS gen=%lu addr=0x%lx cnt=%u)",
            qs2Count, g_tlasGeneration, (uint64_t)g_pendingTLAS.instanceAddr,
            g_pendingTLAS.instanceCount);

    // Same deferred TLAS logic as QueueSubmit
    bool hasTLAS = false;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        hasTLAS = g_pendingTLAS.pending;
    }

    // Periodic TLAS retry via QueueSubmit2
    if (!hasTLAS && g_tlasGeneration == 0 &&
        g_pendingTLAS.instanceAddr != 0 && g_pendingTLAS.instanceCount > 0 &&
        qs2Count > 5 && (qs2Count % 10) == 0 && qs2Count < 500) {
        g_pendingTLAS.pending = true;
        hasTLAS = true;
        LOG("[TLAS] Periodic retry via QueueSubmit2 #%d", qs2Count);
    }

    if (hasTLAS) {
        // Throttle check
        static int tlasRate = -1;
        if (tlasRate < 0) {
            const char* env = getenv("CUDA_RT_TLAS_RATE");
            tlasRate = env ? atoi(env) : 300;
        }
        bool shouldRebuild = (g_bvh2.ready && g_tlasBVH && g_bvh2.tlasGen != g_tlasGeneration);
        if (shouldRebuild) {
            disp.QueueWaitIdle(queue);
            bool hasNewBLAS = processLateBLAS();
            if (hasNewBLAS) uploadBVH2Data(disp, g_lastBLAS);
            processPendingTLAS();
        } else {
            std::lock_guard<std::mutex> lock(g_lock);
            g_pendingTLAS.pending = false;
        }
    }

    return res;
}

// ─── QueuePresentKHR: per-frame hook for deferred TLAS retry ───
static VKAPI_ATTR VkResult VKAPI_CALL layer_QueuePresentKHR(
    VkQueue queue,
    const VkPresentInfoKHR* pPresentInfo)
{
    void* key = getKey(queue);
    DeviceDispatch* pDisp = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        auto it = g_deviceMap.find(key);
        if (it != g_deviceMap.end()) pDisp = &it->second;
    }
    if (!pDisp || !pDisp->QueuePresentKHR) return VK_ERROR_DEVICE_LOST;

    static uint32_t presentCount = 0;
    presentCount++;

    // Reset per-frame RQ dispatch counter
    g_rqDispatchPerFrame = 0;
    if (presentCount <= 5 || presentCount % 100 == 0) {
        LOG("[PRESENT] QueuePresent #%u (TLAS gen=%u addr=0x%llx)",
            presentCount, g_tlasGeneration,
            (unsigned long long)g_pendingTLAS.instanceAddr);
    }

    // Per-frame TLAS rebuild: re-read instance transforms, rebuild TLAS BVH, re-upload SSBOs.
    // Q2RTX rebuilds TLAS every frame with new instance positions — we must track those.
    static int tlasEveryN = -1;
    if (tlasEveryN < 0) {
        const char* env = getenv("CUDA_RT_TLAS_EVERY_N");
        tlasEveryN = env ? atoi(env) : 1;  // default: every frame
        if (tlasEveryN < 1) tlasEveryN = 1;
    }

    bool tlasEligible = g_bvh2.ready && g_pendingTLAS.instanceAddr != 0;
    bool tlasThrottleOk = (presentCount % tlasEveryN) == 0;

    if (tlasEligible && tlasThrottleOk) {
        uint64_t prevGen = g_tlasGeneration;
        if (pDisp->QueueWaitIdle) pDisp->QueueWaitIdle(queue);

        // Build any late BLASes (dynamic geometry) before TLAS
        bool hasNewBLAS = processLateBLAS();
        if (hasNewBLAS) {
            LOG("[PRESENT] Late BLASes added, re-uploading BLAS SSBOs...");
            uploadBVH2Data(*pDisp, g_lastBLAS);
        }

        // Force pending so processPendingTLAS re-reads instance data
        {
            std::lock_guard<std::mutex> lock(g_lock);
            g_pendingTLAS.pending = true;
        }
        processPendingTLAS();

        // Re-upload TLAS nodes + instances if generation advanced
        if (g_tlasGeneration > prevGen) {
            reuploadTLASData(*pDisp);
            if (g_tlasGeneration <= 5 || g_tlasGeneration % 300 == 0)
                LOG("[PRESENT] TLAS rebuilt (gen=%lu) at present #%u", g_tlasGeneration, presentCount);
        }
    }

    // ── Read GPU timestamps from previous frame's render passes ──
    if (g_profile.ready && g_profile.queryPool && !g_profile.passes.empty()) {
        static uint64_t profileFrame = 0;
        profileFrame++;
        bool shouldLog = (profileFrame <= 3) || (profileFrame % 300 == 0);

        if (shouldLog && pDisp->GetQueryPoolResults) {
            // Read all timestamps at once
            uint64_t timestamps[64];
            VkResult qr = pDisp->GetQueryPoolResults(
                pDisp->device, g_profile.queryPool,
                0, g_profile.queryIdx,
                sizeof(timestamps), timestamps,
                sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

            if (qr == VK_SUCCESS) {
                double totalMs = 0;
                double totalGapMs = 0;
                LOG("[PROFILE] ═══ Frame %lu: %u render passes ═══", profileFrame, g_profile.rpCount);
                for (size_t i = 0; i < g_profile.passes.size(); i++) {
                    auto& pt = g_profile.passes[i];
                    double ms = (timestamps[pt.endQuery] - timestamps[pt.startQuery])
                                * g_timestampPeriod / 1e6;
                    totalMs += ms;
                    // Calculate gap before this pass (time since previous pass ended)
                    double gapMs = 0;
                    if (i > 0) {
                        auto& prev = g_profile.passes[i-1];
                        int64_t gapTicks = (int64_t)(timestamps[pt.startQuery] - timestamps[prev.endQuery]);
                        if (gapTicks > 0)
                            gapMs = gapTicks * g_timestampPeriod / 1e6;
                        totalGapMs += gapMs;
                    }
                    if (i == 0)
                        LOG("[PROFILE]   RP%zu (%u color, rp=0x%lx): %ux%u → %.2f ms",
                            i, pt.colorAttachments, pt.renderPass, pt.width, pt.height, ms);
                    else
                        LOG("[PROFILE]   [gap: %.2f ms] → RP%zu (%u color, rp=0x%lx): %ux%u → %.2f ms",
                            gapMs, i, pt.colorAttachments, pt.renderPass, pt.width, pt.height, ms);
                }
                LOG("[PROFILE]   TOTAL: %.2f ms render + %.2f ms gaps = %.2f ms",
                    totalMs, totalGapMs, totalMs + totalGapMs);

                // Compute dispatch timing
                if (g_profile.computeDispatches > 0 &&
                    g_profile.computeStartQuery < 64 && g_profile.computeEndQuery < 64) {
                    double compMs = (timestamps[g_profile.computeEndQuery] - timestamps[g_profile.computeStartQuery])
                                    * g_timestampPeriod / 1e6;
                    LOG("[PROFILE]   COMPUTE: %.2f ms (%u RQ dispatches)", compMs, g_profile.computeDispatches);
                }
                // RP2 mid-pass analysis
                if (g_profile.rp2MidQuery > 0 && g_profile.rp2MidQuery < 64 &&
                    g_profile.passes.size() > 2) {
                    auto& rp2 = g_profile.passes[2];
                    double beginToMid = (timestamps[g_profile.rp2MidQuery] - timestamps[rp2.startQuery])
                                        * g_timestampPeriod / 1e6;
                    double midToEnd = (timestamps[rp2.endQuery] - timestamps[g_profile.rp2MidQuery])
                                      * g_timestampPeriod / 1e6;
                    LOG("[PROFILE]   RP2 SPLIT: begin→mid=%.2f ms, mid→end=%.2f ms", beginToMid, midToEnd);
                    if (g_profile.rp2PreEndQuery > 0 && g_profile.rp2PreEndQuery < 64) {
                        double midToPreEnd = (timestamps[g_profile.rp2PreEndQuery] - timestamps[g_profile.rp2MidQuery])
                                              * g_timestampPeriod / 1e6;
                        double preEndToEnd = (timestamps[rp2.endQuery] - timestamps[g_profile.rp2PreEndQuery])
                                              * g_timestampPeriod / 1e6;
                        LOG("[PROFILE]   RP2 DETAIL: insideRP=%.2f ms, EndRenderPass=%.2f ms", midToPreEnd, preEndToEnd);
                    }
                }
            }
        }
        // Reset for next frame
        g_profile.passes.clear();
        g_profile.queryIdx = 0;
        g_profile.rpCount = 0;
        g_profile.computeStartQuery = 0;
        g_profile.computeEndQuery = 0;
        g_profile.computeDispatches = 0;
        g_profile.rp2MidQuery = 0;
        g_profile.rp2PreEndQuery = 0;
    }

    // RasterBoost: Submit TRT upscale on async compute queue before present
    // The upscale runs concurrently with the app's next frame setup
    if (g_rasterBoost.active && g_async.ready && rasterboost_upscale_has_trt()) {
        static uint64_t asyncUpscaleCount = 0;
        asyncUpscaleCount++;
        if (asyncUpscaleCount <= 3) {
            LOG("[RasterBoost:Async] Async compute upscale pipeline ready (queue family %u)",
                g_async.queueFamily);
        }
        // Actual upscale submission happens via CUDA stream (rasterboost_upscale_run)
        // which naturally overlaps with Vulkan graphics work on different HW queues.
        // The V100's 8 compute queues handle this without explicit Vulkan queue sync.
    }

    // Reset G-buffer capture for next frame
    g_gbuffer.captured = false;
    g_gbuffer.depthImage = VK_NULL_HANDLE;
    g_gbuffer.motionImage = VK_NULL_HANDLE;

    return pDisp->QueuePresentKHR(queue, pPresentInfo);
}

static VKAPI_ATTR void VKAPI_CALL layer_CmdDispatch(
    VkCommandBuffer cmdBuf,
    uint32_t groupCountX,
    uint32_t groupCountY,
    uint32_t groupCountZ)
{
    void* key = getKey(cmdBuf);
    auto& disp = g_deviceMap[key];

    // Lazy upload: if we have a BVH but haven't uploaded BVH2 data yet, do it now
    // Also re-upload when TLAS generation changes (per-frame instance transform updates)
    if (g_lastBLAS) {
        bool needFullUpload = !g_bvh2.ready;
        bool needTLASReupload = false;
        if (g_bvh2.ready && g_tlasBVH && g_bvh2.tlasGen != g_tlasGeneration) {
            needTLASReupload = true;  // fast path: only re-upload TLAS + instances
        }
        if (needFullUpload) {
            uploadBVH2Data(disp, g_lastBLAS);
        }
        else if (needTLASReupload) {
            reuploadTLASData(disp);
        }
    }

    // Track whether this dispatch is an RQ (ray query) pipeline dispatch
    bool isRQDispatch = false;

    // Lock-free check: RQ pipeline set is immutable after creation.
    // cmdBuf pipeline tracking is single-threaded per Vulkan spec.
    if (g_bvh2.ready) [[likely]] {
        uint64_t pipeHandle = 0;
        {
            // No mutex needed: same thread that called CmdBindPipeline
            auto it = g_cmdBufBoundPipeline.find((uint64_t)cmdBuf);
            if (it != g_cmdBufBoundPipeline.end()) pipeHandle = it->second;
        }
        if (pipeHandle) {
            // g_rqPipelines is read-only after pipeline creation (no mutex needed)
            auto it = g_rqPipelines.find(pipeHandle);
            if (it != g_rqPipelines.end()) {
                isRQDispatch = true;
                g_rqDispatchPerFrame++;
                VkPipelineLayout layout = it->second.layout;
                uint32_t bvhSetIdx = (uint32_t)it->second.bvhDescSet;

                // CUDA_RT_MAX_RQ_DISPATCH: Only bind BVH for first N RQ dispatches per frame
                // 0 = unlimited (default). Dispatches beyond limit are skipped (NOP).
                static int maxRQDispatch = -1;
                if (maxRQDispatch < 0) {
                    const char* e = getenv("CUDA_RT_MAX_RQ_DISPATCH");
                    maxRQDispatch = e ? atoi(e) : 0;
                }
                if (maxRQDispatch > 0 && (int)g_rqDispatchPerFrame > maxRQDispatch) {
                    // Skip this dispatch entirely (no BVH binding, no execution)
                    return;
                }

                if (layout && g_bvh2.descSet) {
                    // Bind BVH2 descriptor set at the rewriter's chosen set index
                    disp.CmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                layout, bvhSetIdx, 1,
                                                &g_bvh2.descSet, 0, nullptr);
                }
            }
        }
    }

    // Env var CUDA_RT_HALF_RES: 0=full, 1=half-Y, 2=quarter, 3=NOP (skip RT dispatch entirely)
    static int halfRes = -1;
    if (halfRes < 0) {
        const char* e = getenv("CUDA_RT_HALF_RES");
        halfRes = e ? atoi(e) : 0;
    }
    uint32_t dispX = groupCountX, dispY = groupCountY, dispZ = groupCountZ;
    if (isRQDispatch) {
        if (halfRes >= 3) return; // NOP: skip RT dispatch entirely
        if (halfRes >= 2) { dispX = (dispX + 1) / 2; dispY = (dispY + 1) / 2; }
        else if (halfRes >= 1) { dispY = (dispY + 1) / 2; }

        // GPU timestamps around RQ compute dispatch for profiling
        if (g_profile.ready && g_profile.queryPool && g_profile.queryIdx < 60) {
            uint32_t qi = g_profile.queryIdx;
            disp.CmdResetQueryPool(cmdBuf, g_profile.queryPool, qi, 2);
            disp.CmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   g_profile.queryPool, qi);
            disp.CmdDispatch(cmdBuf, dispX, dispY, dispZ);
            disp.CmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   g_profile.queryPool, qi + 1);
            g_profile.computeStartQuery = qi;
            g_profile.computeEndQuery = qi + 1;
            g_profile.queryIdx += 2;
            g_profile.computeDispatches++;
            return;
        }
    }
    disp.CmdDispatch(cmdBuf, dispX, dispY, dispZ);
}

// ═══════════════════════════════════════════
// Intercepted: CmdCopyAccelerationStructureKHR (compaction)
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_CmdCopyAccelerationStructureKHR(
    VkCommandBuffer cmdBuf,
    const VkCopyAccelerationStructureInfoKHR* pInfo)
{
    LOG("CmdCopyAS: src=0x%lx dst=0x%lx mode=%d",
        (uint64_t)pInfo->src, (uint64_t)pInfo->dst, pInfo->mode);
    std::lock_guard<std::mutex> lock(g_lock);
    uint64_t srcH = (uint64_t)pInfo->src, dstH = (uint64_t)pInfo->dst;
    auto it = g_bvhMap.find(srcH);
    if (it != g_bvhMap.end()) {
        g_bvhMap[dstH] = it->second;
        LOG("  Copied BVH metadata 0x%lx -> 0x%lx", srcH, dstH);
    }
    // Propagate multi-BLAS index to the copy destination
    auto bit = g_asKeyToBLASIdx.find(srcH);
    if (bit != g_asKeyToBLASIdx.end()) {
        g_asKeyToBLASIdx[dstH] = bit->second;
    }
}

// ═══════════════════════════════════════════
// Intercepted: CmdWriteAccelerationStructuresPropertiesKHR
// ═══════════════════════════════════════════
static VKAPI_ATTR void VKAPI_CALL layer_CmdWriteAccelerationStructuresPropertiesKHR(
    VkCommandBuffer cmdBuf,
    uint32_t accelStructCount,
    const VkAccelerationStructureKHR* pAccelStructs,
    VkQueryType queryType,
    VkQueryPool queryPool,
    uint32_t firstQuery)
{
    LOG("CmdWriteASProperties: count=%u queryType=%d", accelStructCount, queryType);
    // App queries compacted size. We don't actually compact, so this is a no-op.
    // The app will read the query result later via vkGetQueryPoolResults.
}

// Layer dispatch: GetInstanceProcAddr / GetDeviceProcAddr
// ═══════════════════════════════════════════

#define INTERCEPT(name) if (!strcmp(pName, "vk" #name)) return (PFN_vkVoidFunction)layer_##name

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_GetInstanceProcAddr(VkInstance instance, const char* pName)
{
    // Layer lifecycle
    INTERCEPT(CreateInstance);
    INTERCEPT(DestroyInstance);
    INTERCEPT(CreateDevice);
    INTERCEPT(EnumerateDeviceExtensionProperties);
    INTERCEPT(GetPhysicalDeviceFeatures2);
    INTERCEPT(GetPhysicalDeviceProperties2);
    if (strcmp(pName, "vkGetPhysicalDeviceFeatures2KHR") == 0)
        return (PFN_vkVoidFunction)layer_GetPhysicalDeviceFeatures2;
    if (strcmp(pName, "vkGetPhysicalDeviceProperties2KHR") == 0)
        return (PFN_vkVoidFunction)layer_GetPhysicalDeviceProperties2;

    // Device-level functions that apps may resolve via InstanceProcAddr
    // (GravityMark does this for vkQueueSubmit, bypassing device-level intercept)
    INTERCEPT(QueueSubmit);
    INTERCEPT(CmdDispatch);
    INTERCEPT(CmdBuildAccelerationStructuresKHR);
    INTERCEPT(CmdBindPipeline);
    INTERCEPT(CmdBeginRenderPass);
    INTERCEPT(CmdEndRenderPass);
    INTERCEPT(CreateRenderPass);
    INTERCEPT(CreateImageView);
    INTERCEPT(DestroyImageView);
    INTERCEPT(CreateFramebuffer);
    INTERCEPT(DestroyFramebuffer);
    INTERCEPT(CmdDrawIndexed);
    INTERCEPT(CmdDraw);
    INTERCEPT(CmdDrawIndexedIndirect);
    INTERCEPT(CmdDrawIndirect);
    INTERCEPT(CmdDrawIndexedIndirectCount);
    INTERCEPT(CmdDrawIndirectCount);
    if (!strcmp(pName, "vkCmdDrawIndexedIndirectCountKHR")) return (PFN_vkVoidFunction)layer_CmdDrawIndexedIndirectCount;
    if (!strcmp(pName, "vkCmdDrawIndirectCountKHR")) return (PFN_vkVoidFunction)layer_CmdDrawIndirectCount;
    INTERCEPT(CmdExecuteCommands);
    INTERCEPT(CmdPipelineBarrier);
    INTERCEPT(CmdWaitEvents);
    // CmdBindDescriptorSets and CmdPipelineBarrier NOT intercepted (high-frequency, not needed for compute RQ)
    INTERCEPT(CreateShaderModule);
    INTERCEPT(CreateComputePipelines);
    INTERCEPT(CreatePipelineLayout);
    if (strcmp(pName, "vkQueueSubmit2KHR") == 0 || strcmp(pName, "vkQueueSubmit2") == 0)
        return (PFN_vkVoidFunction)layer_QueueSubmit2KHR;
    if (strcmp(pName, "vkQueuePresentKHR") == 0)
        return (PFN_vkVoidFunction)layer_QueuePresentKHR;
    if (strcmp(pName, "vkCreateSwapchainKHR") == 0)
        return (PFN_vkVoidFunction)layer_CreateSwapchainKHR;
    INTERCEPT(CreateImage);
    INTERCEPT(DestroyImage);

    // Debug: log Queue/Submit function lookups
    if (pName && strstr(pName, "Queue"))
        fprintf(stderr, "[CudaRT] GetInstanceProcAddr: %s\n", pName);

    // Forward to next layer
    if (instance) {
        void* key = getKey(instance);
        std::lock_guard<std::mutex> lock(g_lock);
        auto it = g_instanceMap.find(key);
        if (it != g_instanceMap.end())
            return it->second.GetInstanceProcAddr(instance, pName);
    }
    return nullptr;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_GetDeviceProcAddr(VkDevice device, const char* pName)
{
    // CUDA_RT_MINIMAL=1: Only feature spoofing, NO interception (test driver's native RT)
    static int minimalMode = -1;
    if (minimalMode < 0) {
        const char* env = getenv("CUDA_RT_MINIMAL");
        minimalMode = (env && atoi(env)) ? 1 : 0;
        if (minimalMode) LOG("MINIMAL MODE: Only DestroyDevice intercepted, driver handles ALL RT");
    }

    // Device lifecycle (always needed)
    INTERCEPT(DestroyDevice);

    if (minimalMode) {
        // In minimal mode, only forward to driver — no interception
        if (device) {
            void* key = getKey(device);
            std::lock_guard<std::mutex> lock(g_lock);
            auto it = g_deviceMap.find(key);
            if (it != g_deviceMap.end())
                return it->second.GetDeviceProcAddr(device, pName);
        }
        return nullptr;
    }

    // Buffer/memory tracking
    INTERCEPT(AllocateMemory);
    INTERCEPT(CreateBuffer);
    INTERCEPT(DestroyBuffer);
    INTERCEPT(BindBufferMemory);
    INTERCEPT(MapMemory);
    INTERCEPT(UnmapMemory);
    INTERCEPT(GetBufferDeviceAddress);
    // Also handle KHR alias
    if (!strcmp(pName, "vkGetBufferDeviceAddressKHR"))
        return (PFN_vkVoidFunction)layer_GetBufferDeviceAddress;

    // Image tracking for CUDA→VkImage interop
    INTERCEPT(CreateImage);
    INTERCEPT(DestroyImage);

    // RasterBoost: swapchain intercept for resolution substitution
    if (strcmp(pName, "vkCreateSwapchainKHR") == 0)
        return (PFN_vkVoidFunction)layer_CreateSwapchainKHR;

    // Shader module interception for ray query rewriting
    INTERCEPT(CreateShaderModule);
    INTERCEPT(CreatePipelineLayout);
    INTERCEPT(CreateComputePipelines);
    INTERCEPT(CmdDispatch);
    INTERCEPT(QueueSubmit);
    // QueueSubmit2 for Vulkan 1.3+ apps (GravityMark uses this for per-frame rendering)
    if (strcmp(pName, "vkQueueSubmit2KHR") == 0 || strcmp(pName, "vkQueueSubmit2") == 0) {
        void* key = getKey(device);
        auto it = g_deviceMap.find(key);
        if (it != g_deviceMap.end() && it->second.QueueSubmit2KHR) {
            return (PFN_vkVoidFunction)layer_QueueSubmit2KHR;
        }
    }
    // QueuePresentKHR: per-frame WSI present hook
    if (strcmp(pName, "vkQueuePresentKHR") == 0) {
        return (PFN_vkVoidFunction)layer_QueuePresentKHR;
    }
    // Debug: log ALL Queue and Submit function lookups
    if (pName && (strstr(pName, "Queue") || strstr(pName, "Submit"))) {
        fprintf(stderr, "[CudaRT] GetDeviceProcAddr: %s\n", pName);
    }

    // RT interceptions — only intercept what's needed
    // CmdBindPipeline: track compute pipeline binding (needed for BVH desc set)
    INTERCEPT(CmdBindPipeline);
    INTERCEPT(CmdBeginRenderPass);
    INTERCEPT(CmdEndRenderPass);
    INTERCEPT(CreateRenderPass);
    INTERCEPT(CreateImageView);
    INTERCEPT(DestroyImageView);
    INTERCEPT(CreateFramebuffer);
    INTERCEPT(DestroyFramebuffer);
    INTERCEPT(CmdDrawIndexed);
    INTERCEPT(CmdDraw);
    INTERCEPT(CmdDrawIndexedIndirect);
    INTERCEPT(CmdDrawIndirect);
    INTERCEPT(CmdDrawIndexedIndirectCount);
    INTERCEPT(CmdDrawIndirectCount);
    if (!strcmp(pName, "vkCmdDrawIndexedIndirectCountKHR")) return (PFN_vkVoidFunction)layer_CmdDrawIndexedIndirectCount;
    if (!strcmp(pName, "vkCmdDrawIndirectCountKHR")) return (PFN_vkVoidFunction)layer_CmdDrawIndirectCount;
    INTERCEPT(CmdExecuteCommands);
    INTERCEPT(CmdPipelineBarrier);
    INTERCEPT(CmdWaitEvents);
    // NOTE: CmdBindDescriptorSets and CmdPipelineBarrier NOT intercepted.
    // They were only needed for RT bind point / RT stage bit remapping, which
    // only applies to vkCmdTraceRaysKHR-based apps (not compute ray queries).
    // Removing saves ~100ns per call × hundreds of thousands of calls per frame.
    INTERCEPT(CreateDescriptorPool);
    INTERCEPT(CreateDescriptorSetLayout);
    INTERCEPT(UpdateDescriptorSets);
    INTERCEPT(DestroyPipeline);
    INTERCEPT(CreateAccelerationStructureKHR);
    INTERCEPT(DestroyAccelerationStructureKHR);
    INTERCEPT(GetAccelerationStructureBuildSizesKHR);
    INTERCEPT(CmdBuildAccelerationStructuresKHR);
    INTERCEPT(GetAccelerationStructureDeviceAddressKHR);
    INTERCEPT(CmdCopyAccelerationStructureKHR);
    INTERCEPT(CmdWriteAccelerationStructuresPropertiesKHR);
    INTERCEPT(CreateRayTracingPipelinesKHR);
    INTERCEPT(GetRayTracingShaderGroupHandlesKHR);
    INTERCEPT(CmdTraceRaysKHR);
    INTERCEPT(CmdTraceRaysIndirectKHR);

    // Forward to next layer
    if (device) {
        void* key = getKey(device);
        std::lock_guard<std::mutex> lock(g_lock);
        auto it = g_deviceMap.find(key);
        if (it != g_deviceMap.end())
            return it->second.GetDeviceProcAddr(device, pName);
    }
    return nullptr;
}

#undef INTERCEPT

// ═══════════════════════════════════════════
// Required export: vkNegotiateLoaderLayerInterfaceVersion
// ═══════════════════════════════════════════
extern "C" {

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct)
{
    if (pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->pfnGetInstanceProcAddr = layer_GetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr = layer_GetDeviceProcAddr;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
    }

    if (pVersionStruct->loaderLayerInterfaceVersion > 2)
        pVersionStruct->loaderLayerInterfaceVersion = 2;

    LOG("Layer negotiated (version %u)", pVersionStruct->loaderLayerInterfaceVersion);
    return VK_SUCCESS;
}

// Legacy exports for compatibility
VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char* pName)
{
    return layer_GetInstanceProcAddr(instance, pName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char* pName)
{
    return layer_GetDeviceProcAddr(device, pName);
}

} // extern "C"
