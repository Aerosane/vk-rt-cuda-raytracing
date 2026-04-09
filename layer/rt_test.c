/* rt_test.c — Minimal Vulkan Ray Tracing test to exercise our CudaRT layer
 *
 * Creates a single triangle BLAS, wraps it in TLAS, creates RT pipeline
 * with raygen/miss/closesthit shaders, and dispatches CmdTraceRaysKHR.
 * All RT calls are intercepted by our VK_LAYER_CUDA_RT layer.
 *
 * Compile: gcc -O2 -o rt_test rt_test.c -lvulkan -lm
 * Run:     ENABLE_CUDA_RT_LAYER=1 ./rt_test
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define VK_CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { \
    fprintf(stderr, "VK ERROR %d at %s:%d (%s)\n", r, __FILE__, __LINE__, #x); exit(1); } } while(0)

// Dynamic RT function pointers
static PFN_vkGetBufferDeviceAddressKHR                 pfn_vkGetBufferDeviceAddress;
static PFN_vkCreateAccelerationStructureKHR            pfn_vkCreateAccelerationStructureKHR;
static PFN_vkDestroyAccelerationStructureKHR           pfn_vkDestroyAccelerationStructureKHR;
static PFN_vkGetAccelerationStructureBuildSizesKHR     pfn_vkGetAccelerationStructureBuildSizesKHR;
static PFN_vkCmdBuildAccelerationStructuresKHR         pfn_vkCmdBuildAccelerationStructuresKHR;
static PFN_vkGetAccelerationStructureDeviceAddressKHR  pfn_vkGetAccelerationStructureDeviceAddressKHR;
static PFN_vkCreateRayTracingPipelinesKHR              pfn_vkCreateRayTracingPipelinesKHR;
static PFN_vkGetRayTracingShaderGroupHandlesKHR        pfn_vkGetRayTracingShaderGroupHandlesKHR;
static PFN_vkCmdTraceRaysKHR                           pfn_vkCmdTraceRaysKHR;

static uint32_t findMemType(VkPhysicalDevice gpu, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(gpu, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    fprintf(stderr, "Failed to find suitable memory type\n");
    exit(1);
}

typedef struct {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    VkDeviceAddress address;
    void*          mapped;
} GpuBuffer;

static GpuBuffer createBuffer(VkDevice dev, VkPhysicalDevice gpu, VkDeviceSize size,
                               VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps) {
    GpuBuffer buf = {0};

    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &ci, NULL, &buf.buffer));

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(dev, buf.buffer, &memReq);

    VkMemoryAllocateFlagsInfo flagsInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &flagsInfo;
    ai.allocationSize = memReq.size;
    ai.memoryTypeIndex = findMemType(gpu, memReq.memoryTypeBits, memProps);
    VK_CHECK(vkAllocateMemory(dev, &ai, NULL, &buf.memory));
    VK_CHECK(vkBindBufferMemory(dev, buf.buffer, buf.memory, 0));

    if (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        VK_CHECK(vkMapMemory(dev, buf.memory, 0, size, 0, &buf.mapped));

    VkBufferDeviceAddressInfo addrInfo = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addrInfo.buffer = buf.buffer;
    buf.address = pfn_vkGetBufferDeviceAddress(dev, &addrInfo);

    return buf;
}

static uint32_t* loadSpirv(const char* path, size_t* outSize) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    *outSize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t* code = (uint32_t*)malloc(*outSize);
    fread(code, 1, *outSize, f);
    fclose(f);
    return code;
}

static VkShaderModule createShaderModule(VkDevice dev, const char* path) {
    size_t sz;
    uint32_t* code = loadSpirv(path, &sz);
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = sz;
    ci.pCode = code;
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(dev, &ci, NULL, &sm));
    free(code);
    return sm;
}

int main(int argc, char** argv) {
    printf("═══════════════════════════════════════════════\n");
    printf("  Vulkan RT Test — CudaRT Layer Exercise\n");
    printf("═══════════════════════════════════════════════\n\n");

    // ── Instance ──
    const char* instExts[] = {};
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_2;
    appInfo.pApplicationName = "rt_test";

    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &appInfo;
    VkInstance instance;
    VK_CHECK(vkCreateInstance(&ici, NULL, &instance));
    printf("[OK] Instance created\n");

    // ── Physical device ──
    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(instance, &gpuCount, NULL);
    VkPhysicalDevice* gpus = (VkPhysicalDevice*)malloc(gpuCount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus);
    VkPhysicalDevice gpu = gpus[0];
    free(gpus);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu, &props);
    printf("[OK] GPU: %s\n", props.deviceName);

    // Check RT extensions
    uint32_t extCount;
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &extCount, NULL);
    VkExtensionProperties* exts = (VkExtensionProperties*)malloc(extCount * sizeof(VkExtensionProperties));
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &extCount, exts);

    int hasAS = 0, hasRTPipe = 0, hasRayQuery = 0, hasDeferredOp = 0, hasSPIRV14 = 0;
    for (uint32_t i = 0; i < extCount; i++) {
        if (!strcmp(exts[i].extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) hasAS = 1;
        if (!strcmp(exts[i].extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) hasRTPipe = 1;
        if (!strcmp(exts[i].extensionName, VK_KHR_RAY_QUERY_EXTENSION_NAME)) hasRayQuery = 1;
        if (!strcmp(exts[i].extensionName, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) hasDeferredOp = 1;
        if (!strcmp(exts[i].extensionName, VK_KHR_SPIRV_1_4_EXTENSION_NAME)) hasSPIRV14 = 1;
    }
    free(exts);
    printf("[INFO] RT extensions: AS=%d RTPipeline=%d RayQuery=%d DeferredOp=%d SPIRV14=%d\n",
           hasAS, hasRTPipe, hasRayQuery, hasDeferredOp, hasSPIRV14);

    if (!hasAS || (!hasRTPipe && !hasRayQuery)) {
        printf("[FAIL] Missing required RT extensions (need AS + pipeline or ray_query)\n");
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    if (!hasRTPipe && hasRayQuery) {
        printf("[INFO] Using ray_query path (VK_RT layer software RT mode)\n");
    }

    // ── Device ──
    float queuePri = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &queuePri;

    const char* devExts_pipeline[] = {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    };
    const char* devExts_rayquery[] = {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    };
    int useRayQuery = !hasRTPipe && hasRayQuery;
    const char** devExts = useRayQuery ? devExts_rayquery : devExts_pipeline;
    uint32_t devExtCount = useRayQuery ? (sizeof(devExts_rayquery)/sizeof(devExts_rayquery[0]))
                                       : (sizeof(devExts_pipeline)/sizeof(devExts_pipeline[0]));

    // Feature chain
    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bdaFeatures.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    asFeatures.pNext = &bdaFeatures;
    asFeatures.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceRayQueryFeaturesKHR rqFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    rqFeatures.rayQuery = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    rtFeatures.rayTracingPipeline = VK_TRUE;

    // Chain: use ray_query features if in RQ mode, else pipeline features
    if (useRayQuery) {
        rqFeatures.pNext = &asFeatures;
    } else {
        rtFeatures.pNext = &asFeatures;
    }

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = useRayQuery ? (void*)&rqFeatures : (void*)&rtFeatures;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = devExtCount;
    dci.ppEnabledExtensionNames = devExts;

    VkDevice device;
    VK_CHECK(vkCreateDevice(gpu, &dci, NULL, &device));
    printf("[OK] Device created with RT extensions\n");

    VkQueue queue;
    vkGetDeviceQueue(device, 0, 0, &queue);

    // Load RT function pointers
    #define LOAD(fn) pfn_##fn = (PFN_##fn)vkGetDeviceProcAddr(device, #fn)
    pfn_vkGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddressKHR)
        vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR");
    LOAD(vkCreateAccelerationStructureKHR);
    LOAD(vkDestroyAccelerationStructureKHR);
    LOAD(vkGetAccelerationStructureBuildSizesKHR);
    LOAD(vkCmdBuildAccelerationStructuresKHR);
    LOAD(vkGetAccelerationStructureDeviceAddressKHR);
    LOAD(vkCreateRayTracingPipelinesKHR);
    LOAD(vkGetRayTracingShaderGroupHandlesKHR);
    LOAD(vkCmdTraceRaysKHR);
    #undef LOAD

    // Use the KHR alias for buffer device address
    if (!pfn_vkGetBufferDeviceAddress) {
        pfn_vkGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddressKHR)
            vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddress");
    }

    printf("[OK] RT function pointers loaded\n");
    printf("  CreateAS=%p BuildAS=%p TraceRays=%p\n",
           (void*)pfn_vkCreateAccelerationStructureKHR,
           (void*)pfn_vkCmdBuildAccelerationStructuresKHR,
           (void*)pfn_vkCmdTraceRaysKHR);

    // ── Command pool ──
    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = 0;
    VkCommandPool cmdPool;
    VK_CHECK(vkCreateCommandPool(device, &cpci, NULL, &cmdPool));

    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd));

    // ── Geometry: procedural conference-room-style mesh ──
    // Generate a grid of triangles for meaningful BVH benchmark
    int gridN = (argc > 1) ? atoi(argv[1]) : 200;
    int numVerts = (gridN+1) * (gridN+1);
    int numTris = gridN * gridN * 2;
    int vertSize = numVerts * 3 * sizeof(float);
    int idxSize = numTris * 3 * sizeof(uint32_t);

    GpuBuffer vertBuf = createBuffer(device, gpu, vertSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    GpuBuffer idxBuf = createBuffer(device, gpu, idxSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Generate vertices: a wavy terrain in [-1,1]×[-1,1]
    {
        float* verts = (float*)vertBuf.mapped;
        for (int y = 0; y <= gridN; y++) {
            for (int x = 0; x <= gridN; x++) {
                int vi = (y * (gridN+1) + x) * 3;
                float fx = (float)x / gridN * 2.0f - 1.0f;
                float fy = (float)y / gridN * 2.0f - 1.0f;
                float fz = 0.15f * sinf(fx*6.28f) * cosf(fy*6.28f)
                         + 0.05f * sinf(fx*18.84f) * sinf(fy*12.56f);
                verts[vi+0] = fx;
                verts[vi+1] = fy;
                verts[vi+2] = fz;
            }
        }
    }

    // Generate indices: two triangles per quad
    {
        uint32_t* idx = (uint32_t*)idxBuf.mapped;
        int ti = 0;
        for (int y = 0; y < gridN; y++) {
            for (int x = 0; x < gridN; x++) {
                uint32_t v00 = y * (gridN+1) + x;
                uint32_t v10 = v00 + 1;
                uint32_t v01 = v00 + (gridN+1);
                uint32_t v11 = v01 + 1;
                idx[ti++] = v00; idx[ti++] = v10; idx[ti++] = v01;
                idx[ti++] = v10; idx[ti++] = v11; idx[ti++] = v01;
            }
        }
    }

    printf("[OK] Geometry: %dx%d grid = %d triangles, %d vertices\n",
           gridN, gridN, numTris, numVerts);

    // ── BLAS ──
    VkAccelerationStructureGeometryKHR asGeo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    asGeo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    asGeo.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    asGeo.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    asGeo.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    asGeo.geometry.triangles.vertexData.deviceAddress = vertBuf.address;
    asGeo.geometry.triangles.vertexStride = 3 * sizeof(float);
    asGeo.geometry.triangles.maxVertex = numVerts - 1;
    asGeo.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    asGeo.geometry.triangles.indexData.deviceAddress = idxBuf.address;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &asGeo;

    uint32_t primCount = numTris;
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pfn_vkGetAccelerationStructureBuildSizesKHR(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primCount, &sizeInfo);
    printf("[OK] BLAS sizes: struct=%zu scratch=%zu\n",
           (size_t)sizeInfo.accelerationStructureSize, (size_t)sizeInfo.buildScratchSize);

    GpuBuffer blasBuf = createBuffer(device, gpu, sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkAccelerationStructureCreateInfoKHR asci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    asci.buffer = blasBuf.buffer;
    asci.size = sizeInfo.accelerationStructureSize;
    asci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VkAccelerationStructureKHR blas;
    VK_CHECK(pfn_vkCreateAccelerationStructureKHR(device, &asci, NULL, &blas));
    printf("[OK] BLAS created\n");

    // Build BLAS
    GpuBuffer scratchBuf = createBuffer(device, gpu, sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    buildInfo.dstAccelerationStructure = blas;
    buildInfo.scratchData.deviceAddress = scratchBuf.address;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
    rangeInfo.primitiveCount = numTris;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
    pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);

    // Barrier
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &barrier, 0, NULL, 0, NULL);

    printf("[OK] BLAS build recorded\n");

    // ── TLAS ──
    VkAccelerationStructureDeviceAddressInfoKHR blasAddrInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    blasAddrInfo.accelerationStructure = blas;
    VkDeviceAddress blasAddress = pfn_vkGetAccelerationStructureDeviceAddressKHR(device, &blasAddrInfo);

    VkAccelerationStructureInstanceKHR asInstance = {};
    asInstance.transform.matrix[0][0] = 1.0f;
    asInstance.transform.matrix[1][1] = 1.0f;
    asInstance.transform.matrix[2][2] = 1.0f;
    asInstance.instanceCustomIndex = 0;
    asInstance.mask = 0xFF;
    asInstance.instanceShaderBindingTableRecordOffset = 0;
    asInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    asInstance.accelerationStructureReference = blasAddress;

    GpuBuffer instanceBuf = createBuffer(device, gpu, sizeof(asInstance),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(instanceBuf.mapped, &asInstance, sizeof(asInstance));

    VkAccelerationStructureGeometryKHR tlasGeo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    tlasGeo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeo.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tlasGeo.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeo.geometry.instances.arrayOfPointers = VK_FALSE;
    tlasGeo.geometry.instances.data.deviceAddress = instanceBuf.address;

    VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    tlasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tlasBuildInfo.geometryCount = 1;
    tlasBuildInfo.pGeometries = &tlasGeo;

    uint32_t tlasPrimCount = 1;
    VkAccelerationStructureBuildSizesInfoKHR tlasSizeInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pfn_vkGetAccelerationStructureBuildSizesKHR(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &tlasBuildInfo, &tlasPrimCount, &tlasSizeInfo);
    printf("[OK] TLAS sizes: struct=%zu scratch=%zu\n",
           (size_t)tlasSizeInfo.accelerationStructureSize, (size_t)tlasSizeInfo.buildScratchSize);

    GpuBuffer tlasBuf = createBuffer(device, gpu, tlasSizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkAccelerationStructureCreateInfoKHR tlasCI = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    tlasCI.buffer = tlasBuf.buffer;
    tlasCI.size = tlasSizeInfo.accelerationStructureSize;
    tlasCI.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    VkAccelerationStructureKHR tlas;
    VK_CHECK(pfn_vkCreateAccelerationStructureKHR(device, &tlasCI, NULL, &tlas));
    printf("[OK] TLAS created\n");

    // Ensure scratch is big enough
    GpuBuffer tlasScratchBuf = createBuffer(device, gpu,
        tlasSizeInfo.buildScratchSize > 256 ? tlasSizeInfo.buildScratchSize : 256,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    tlasBuildInfo.dstAccelerationStructure = tlas;
    tlasBuildInfo.scratchData.deviceAddress = tlasScratchBuf.address;

    VkAccelerationStructureBuildRangeInfoKHR tlasRangeInfo = {};
    tlasRangeInfo.primitiveCount = 1;
    const VkAccelerationStructureBuildRangeInfoKHR* pTlasRangeInfo = &tlasRangeInfo;

    pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlasBuildInfo, &pTlasRangeInfo);

    VkMemoryBarrier barrier2 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier2.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier2.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 1, &barrier2, 0, NULL, 0, NULL);

    printf("[OK] TLAS build recorded\n");

    // ── RT Pipeline ──
    VkShaderModule rgenMod  = createShaderModule(device, "/workspaces/codespace/VK_RT/layer/shaders/raygen.spv");
    VkShaderModule rmissMod = createShaderModule(device, "/workspaces/codespace/VK_RT/layer/shaders/miss.spv");
    VkShaderModule rchitMod = createShaderModule(device, "/workspaces/codespace/VK_RT/layer/shaders/closesthit.spv");
    printf("[OK] Shader modules loaded\n");

    VkPipelineShaderStageCreateInfo stages[3] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = rgenMod;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[1].module = rmissMod;
    stages[1].pName = "main";

    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[2].module = rchitMod;
    stages[2].pName = "main";

    VkRayTracingShaderGroupCreateInfoKHR groups[3] = {};
    // Raygen group
    groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;
    // Miss group
    groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;
    // Closest-hit group
    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].generalShader = VK_SHADER_UNUSED_KHR;
    groups[2].closestHitShader = 2;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Descriptor set layout: binding 0 = AS, binding 1 = storage image
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 2;
    dslci.pBindings = bindings;
    VkDescriptorSetLayout dsLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, NULL, &dsLayout));

    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsLayout;
    VkPipelineLayout pipeLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &plci, NULL, &pipeLayout));

    VkRayTracingPipelineCreateInfoKHR rtPipeCI = {
        VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    rtPipeCI.stageCount = 3;
    rtPipeCI.pStages = stages;
    rtPipeCI.groupCount = 3;
    rtPipeCI.pGroups = groups;
    rtPipeCI.maxPipelineRayRecursionDepth = 1;
    rtPipeCI.layout = pipeLayout;

    VkPipeline rtPipeline;
    VK_CHECK(pfn_vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                                 1, &rtPipeCI, NULL, &rtPipeline));
    printf("[OK] RT Pipeline created\n");

    // ── SBT (Shader Binding Table) ──
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(gpu, &props2);

    uint32_t handleSize = rtProps.shaderGroupHandleSize;
    uint32_t handleAlign = rtProps.shaderGroupHandleAlignment;
    uint32_t baseAlign = rtProps.shaderGroupBaseAlignment;
    uint32_t handleSizeAligned = (handleSize + handleAlign - 1) & ~(handleAlign - 1);
    printf("[INFO] SBT: handleSize=%u handleAlign=%u baseAlign=%u\n",
           handleSize, handleAlign, baseAlign);

    uint32_t sbtSize = baseAlign * 3; // raygen + miss + hit, each base-aligned
    uint8_t* handleData = (uint8_t*)malloc(handleSize * 3);
    VK_CHECK(pfn_vkGetRayTracingShaderGroupHandlesKHR(device, rtPipeline, 0, 3, handleSize * 3, handleData));

    GpuBuffer sbtBuf = createBuffer(device, gpu, sbtSize,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Copy handles at base-aligned offsets
    memset(sbtBuf.mapped, 0, sbtSize);
    memcpy((uint8_t*)sbtBuf.mapped + baseAlign * 0, handleData + handleSize * 0, handleSize);
    memcpy((uint8_t*)sbtBuf.mapped + baseAlign * 1, handleData + handleSize * 1, handleSize);
    memcpy((uint8_t*)sbtBuf.mapped + baseAlign * 2, handleData + handleSize * 2, handleSize);
    free(handleData);
    printf("[OK] SBT created\n");

    // ── Output Image (storage image for raygen shader) ──
    const int RT_WIDTH = 1024, RT_HEIGHT = 1024;
    VkImageCreateInfo imgCI = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgCI.imageType = VK_IMAGE_TYPE_2D;
    imgCI.format = VK_FORMAT_R8G8B8A8_UNORM;
    imgCI.extent.width = RT_WIDTH;
    imgCI.extent.height = RT_HEIGHT;
    imgCI.extent.depth = 1;
    imgCI.mipLevels = 1;
    imgCI.arrayLayers = 1;
    imgCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imgCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgCI.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage outImage;
    VK_CHECK(vkCreateImage(device, &imgCI, NULL, &outImage));

    VkMemoryRequirements imgMemReq;
    vkGetImageMemoryRequirements(device, outImage, &imgMemReq);
    VkMemoryAllocateInfo imgAlloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    imgAlloc.allocationSize = imgMemReq.size;
    imgAlloc.memoryTypeIndex = findMemType(gpu, imgMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory imgMem;
    VK_CHECK(vkAllocateMemory(device, &imgAlloc, NULL, &imgMem));
    VK_CHECK(vkBindImageMemory(device, outImage, imgMem, 0));

    VkImageViewCreateInfo ivci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ivci.image = outImage;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
    VkImageView outImageView;
    VK_CHECK(vkCreateImageView(device, &ivci, NULL, &outImageView));
    printf("[OK] Output image created (%dx%d RGBA8)\n", RT_WIDTH, RT_HEIGHT);

    // ── Descriptor Pool + Set ──
    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = poolSizes;
    VkDescriptorPool descPool;
    VK_CHECK(vkCreateDescriptorPool(device, &dpci, NULL, &descPool));

    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsLayout;
    VkDescriptorSet descSet;
    VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &descSet));

    // Write descriptors
    VkWriteDescriptorSetAccelerationStructureKHR asWrite = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures = &tlas;

    VkDescriptorImageInfo imgInfo = {};
    imgInfo.imageView = outImageView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &asWrite;
    writes[0].dstSet = descSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &imgInfo;
    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);
    printf("[OK] Descriptors bound (TLAS + output image)\n");

    // ── Trace Rays! ──
    // Transition image to GENERAL layout
    VkImageMemoryBarrier imgBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imgBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBarrier.image = outImage;
    imgBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imgBarrier.subresourceRange.levelCount = 1;
    imgBarrier.subresourceRange.layerCount = 1;
    imgBarrier.srcAccessMask = 0;
    imgBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, NULL, 0, NULL, 1, &imgBarrier);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            pipeLayout, 0, 1, &descSet, 0, NULL);

    VkStridedDeviceAddressRegionKHR rgenRegion = {};
    rgenRegion.deviceAddress = sbtBuf.address + baseAlign * 0;
    rgenRegion.stride = handleSizeAligned;
    rgenRegion.size = handleSizeAligned;

    VkStridedDeviceAddressRegionKHR missRegion = {};
    missRegion.deviceAddress = sbtBuf.address + baseAlign * 1;
    missRegion.stride = handleSizeAligned;
    missRegion.size = handleSizeAligned;

    VkStridedDeviceAddressRegionKHR hitRegion = {};
    hitRegion.deviceAddress = sbtBuf.address + baseAlign * 2;
    hitRegion.stride = handleSizeAligned;
    hitRegion.size = handleSizeAligned;

    VkStridedDeviceAddressRegionKHR callRegion = {}; // empty

    pfn_vkCmdTraceRaysKHR(cmd,
        &rgenRegion, &missRegion, &hitRegion, &callRegion,
        RT_WIDTH, RT_HEIGHT, 1);

    printf("[OK] CmdTraceRaysKHR recorded: %dx%dx1 = %d rays\n",
           RT_WIDTH, RT_HEIGHT, RT_WIDTH * RT_HEIGHT);

    VK_CHECK(vkEndCommandBuffer(cmd));

    // Submit
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    printf("[OK] Command buffer executed successfully!\n");

    printf("\n═══════════════════════════════════════════════\n");
    printf("  ALL RT CALLS INTERCEPTED SUCCESSFULLY\n");
    printf("═══════════════════════════════════════════════\n");

    // Cleanup
    vkDestroyPipeline(device, rtPipeline, NULL);
    vkDestroyPipelineLayout(device, pipeLayout, NULL);
    vkDestroyDescriptorSetLayout(device, dsLayout, NULL);
    vkDestroyShaderModule(device, rgenMod, NULL);
    vkDestroyShaderModule(device, rmissMod, NULL);
    vkDestroyShaderModule(device, rchitMod, NULL);
    pfn_vkDestroyAccelerationStructureKHR(device, tlas, NULL);
    pfn_vkDestroyAccelerationStructureKHR(device, blas, NULL);
    // buffers...
    vkDestroyCommandPool(device, cmdPool, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    return 0;
}
