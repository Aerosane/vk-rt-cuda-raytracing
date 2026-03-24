/* cuda_bvh_backend.h — Interface for CUDA BVH build/trace used by the Vulkan layer */

#ifndef CUDA_BVH_BACKEND_H
#define CUDA_BVH_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a built BVH */
typedef struct CudaBVHHandle* CudaBVH_t;

/* Triangle data as passed from Vulkan vertex/index buffers */
typedef struct {
    float v0[3], v1[3], v2[3];
} CudaTri;

/* Build BVH4+CWBVH from triangle list. Returns opaque handle. */
CudaBVH_t cudaBVH_build(const CudaTri* tris, int numTris);

/* Free BVH resources */
void cudaBVH_destroy(CudaBVH_t bvh);

/* Trace primary rays (BVH4).
 * Generates camera rays for side×side image, writes hit distances to outHitT.
 * Returns MRays/sec. */
float cudaBVH_tracePrimary(CudaBVH_t bvh, int side,
                           float camOx, float camOy, float camOz,
                           float* outHitT);

/* Trace diffuse rays (CWBVH + Morton sort).
 * Takes ray origins/directions as flat float arrays [numRays×3].
 * Writes hit distances to outHitT.
 * Returns MRays/sec. */
float cudaBVH_traceDiffuse(CudaBVH_t bvh, int numRays,
                           const float* rayOx, const float* rayOy, const float* rayOz,
                           const float* rayDx, const float* rayDy, const float* rayDz,
                           float* outHitT);

/* Trace primary rays and convert to RGBA pixels.
 * Width×Height output (can be non-square). Writes to outRGBA_host (must be pre-allocated).
 * Returns 0 on success. */
int cudaBVH_traceToRGBA(CudaBVH_t bvh, int width, int height,
                        float camOx, float camOy, float camOz,
                        uint32_t* outRGBA_host);

/* Trace primary rays and write to an existing GPU buffer pointer.
 * d_outRGBA must point to width*height * bytesPerPixel bytes of GPU memory.
 * outFmt: 0=RGBA8, 1=BGRA8, 2=RGBA16F, 3=RGBA32F
 * Camera looks from (camOx,camOy,camOz) toward scene center.
 * Returns 0 on success. */
int cudaBVH_traceToGPUPtr(CudaBVH_t bvh, int width, int height,
                          float camOx, float camOy, float camOz,
                          void* d_outRGBA, int outFmt);

/* Async trace to GPU pointer via background thread.
 * d_outRGBA must point to DEVICE_LOCAL (CUDA-imported) GPU memory.
 * Fire-and-forget: kernel runs on background thread, no sync on caller's thread.
 * Caller must ensure previous trace is complete before reading the buffer. */
void cudaBVH_traceToGPUAsync(CudaBVH_t bvh, int width, int height,
                             float camOx, float camOy, float camOz,
                             void* d_outRGBA, int outFmt);

/* Trace primary rays to an internal device buffer, then copy to a host pointer.
 * h_outRGBA: host pointer (e.g., mapped Vulkan HOST_VISIBLE buffer)
 * outFmt: 0=RGBA8, 1=BGRA8, 2=RGBA16F, 3=RGBA32F
 * bufSize: total bytes to copy (width*height*bpp)
 * Returns 0 on success. */
int cudaBVH_traceToHostPtr(CudaBVH_t bvh, int width, int height,
                           float camOx, float camOy, float camOz,
                           void* h_outRGBA, int outFmt, uint64_t bufSize);

/* Import a Vulkan-exported GPU memory fd and return a CUDA device pointer.
 * The fd is consumed (closed by CUDA driver).
 * Returns the CUDA device pointer, or NULL on failure. */
void* cudaBVH_importBufferFd(int fd, uint64_t size);

/* Get scene bounds for auto-camera positioning */
void cudaBVH_getBounds(CudaBVH_t bvh, float* centerX, float* centerY, float* centerZ, float* extent);

/* Get stats */
int cudaBVH_getNumTris(CudaBVH_t bvh);
int cudaBVH_getNumBVH4Nodes(CudaBVH_t bvh);
int cudaBVH_getNumCWBVHNodes(CudaBVH_t bvh);

/* Get raw BVH4 node data for Vulkan compute shader upload.
 * Returns pointer to host-side BVH4 node array (numNodes × 4 × 16 bytes).
 * Data format: 4× uvec4 per node — x,y,z bounds as fp16, child indices as int32. */
const void* cudaBVH_getNodeData(CudaBVH_t bvh);

/* Get raw triangle SoA data for Vulkan compute shader upload.
 * Fills outPtrs[0..8] with pointers to: tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z.
 * Each is a float array of numTris elements, hosted on CPU. */
void cudaBVH_getTriData(CudaBVH_t bvh, const float* outPtrs[9]);

/* Get stackless BVH2 DFS-ordered node data for GLSL compute shader.
 * Returns number of nodes. Fills outData with pointer to 8 uint32s per node:
 *   [0-2]=bmin(fp32), [3-5]=bmax(fp32), [6]=leaf_enc, [7]=skip
 * Inner nodes: leaf_enc=0, skip=DFS index of next after subtree
 * Leaf nodes:  leaf_enc<0 (encoded), skip=DFS index after leaf */
int cudaBVH_getStacklessBVH2(CudaBVH_t bvh, uint32_t** outData);

/* Get packed triangle data for BVH2 GLSL/SPIR-V traversal.
 * Returns number of vec4s. Fills outData with pointer to float[numVec4s*4].
 * Format: 3 vec4s per triangle:
 *   p0 = {v0.x, v0.y, v0.z, v1.x}
 *   p1 = {v1.y, v1.z, v2.x, v2.y}
 *   p2 = {v2.z, 0, 0, 0} */
int cudaBVH_getPackedTris(CudaBVH_t bvh, float** outData);

/* Destroy CUDA context to break driver-level CUDA-Vulkan serialization.
 * Call AFTER all BVH data has been uploaded to Vulkan SSBOs.
 * After this call, no CUDA functions should be used. */
void cudaBVH_resetDevice(void);

/* Build BVH2 from axis-aligned bounding boxes (for TLAS over instances).
 * aabbs: 6 floats per AABB (minX, minY, minZ, maxX, maxY, maxZ)
 * numAABBs: number of AABBs
 * Returns opaque BVH handle. Use cudaBVH_getStacklessBVH2() to get traversal data.
 * Leaf nodes encode primitive index = AABB index (for instance lookup). */
CudaBVH_t cudaBVH_buildFromAABBs(const float* aabbs, int numAABBs);

/* Fast TLAS-only builder: BVH2 stackless data only (no BVH4, no CUDA, no SoA).
 * Produces packed uint32 array: 8 words per node.
 * Returns node count. Caller must free(*outNodes) when done.
 * ~10× faster than cudaBVH_buildFromAABBs for per-frame TLAS rebuilds. */
int cudaBVH_buildTLASFast(const float* aabbs, int numAABBs, uint32_t** outNodes);

/* Read GPU-only buffer data via CUDA external memory import.
 * fd: file descriptor exported from vkGetMemoryFdKHR
 * allocationSize: total size of the VkDeviceMemory
 * offset: byte offset into the memory
 * size: bytes to read
 * dst: host destination buffer (must be pre-allocated, size bytes)
 * Returns 0 on success, -1 on failure. */
int cudaBVH_readGPUMemoryFd(int fd, uint64_t allocationSize,
                             uint64_t offset, uint64_t size, void* dst);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_BVH_BACKEND_H */
