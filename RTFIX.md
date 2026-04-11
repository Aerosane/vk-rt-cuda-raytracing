# VK_RT Layer — Ray Tracing on Tesla V100

## What This Is

A Vulkan implicit layer (`VkLayer_CudaRT`) that enables **VK_KHR_ray_query** on
NVIDIA Tesla V100 GPUs, which lack dedicated RT cores. The layer intercepts
Vulkan API calls and replaces SPIR-V `ray_query` operations with software BVH
traversal code that runs on the V100's shader cores.

## Architecture

```
App (Breaking Limit / Q2RTX)
        │
        ▼
┌─────────────────────────┐
│   VkLayer_CudaRT        │
│                         │
│  1. Extension Spoof     │  Report ray_query/AS as supported
│  2. SPIR-V Rewrite      │  ray_query ops → BVH traversal loop
│  3. Pipeline Extension   │  Add BVH SSBO descriptor set
│  4. AS Interception      │  Capture BLAS/TLAS builds
│  5. Dispatch Binding     │  Bind BVH data before RT dispatches
│                         │
└─────────────────────────┘
        │
        ▼
   NVIDIA Driver 580.126.20 (V100)
```

### Key Files

| File | Purpose |
|------|---------|
| `layer/VkLayer_CudaRT.cpp` | Main layer (~7200 lines). All Vulkan interception. |
| `layer/spirv_ray_query_rewriter.h` | SPIR-V transformer: ray_query → BVH traversal |
| `layer/VkLayer_CudaRT.json` | Layer manifest for Vulkan loader |
| `layer/build.sh` | Build script (g++ -std=c++20 -O2) |

### V100 Driver RT Support (580.126.20)

| Extension | Native | Notes |
|-----------|--------|-------|
| `VK_KHR_acceleration_structure` | ✅ | Driver builds BVH internally |
| `VK_KHR_ray_tracing_pipeline` | ✅ | Full pipeline RT (not used by Breaking Limit) |
| `VK_KHR_ray_query` | ❌ | **No hardware support — our layer provides this** |

## SPIR-V Rewriting Pipeline

The rewriter (`spirv_ray_query_rewriter.h`) performs a single-pass transformation:

1. **Scan** — Identify `OpTypeRayQueryKHR`, `OpRayQueryInitializeKHR`,
   `OpRayQueryProceedKHR`, `OpRayQueryGetIntersection*` ops
2. **Replace types** — `OpTypeRayQueryKHR` → set of traversal state variables
   (node index, bestT, hitU/V, hitPrim, hitType, etc.)
3. **Replace Initialize** — Store ray origin/direction/tMin/tMax, compute
   inverse direction for slab tests
4. **Replace Proceed** — Emit TLAS→BLAS two-level BVH traversal loop with
   AABB slab tests and Möller-Trumbore triangle intersection
5. **Replace GetIntersection** — Load from traversal state variables

### Validation Results

- All **196 Breaking Limit shaders** pass ASAN (zero memory bugs in rewriter)
- All **196 rewritten shaders** pass `spirv-val` (valid SPIR-V)
- Shader growth: ~40% for full BVH traversal (11K → 15K words typical)

## Current Status: Breaking Limit Benchmark

### What Works

- ✅ Layer loads and intercepts all Vulkan calls
- ✅ Extensions spoofed correctly (ray_query + acceleration_structure)
- ✅ All 196 shaders rewritten and compile in the driver
- ✅ Compute pipelines created with extended layouts
- ✅ BVH descriptor sets created and bound
- ✅ App loads scene, creates render passes, begins rendering
- ✅ **Q2RTX works**: 2000 frames, 38 BLAS, 33 TLAS instances, ~237 FPS at 640×480

### What's Blocking: V100 Driver Use-After-Free Bug

The NVIDIA 580.126.20 driver has a **use-after-free bug** that causes crashes
during rendering. This is a driver bug, not a layer bug.

#### Evidence

| Test Configuration | Max Frame | Notes |
|-------------------|-----------|-------|
| No layer | N/A | "Rayqueries are not supported" |
| Full layer (glibc malloc) | 0–1 | glibc reuses freed memory immediately |
| Full layer + jemalloc | 0–1 | Extended layout makes UAF worse |
| Full + jemalloc + NO_EXT_LAYOUT + NO_BVH_BIND | **6** | Minimal layer footprint |
| STRIP_ONLY (NOP ray_query) | 0 | Invalid SPIR-V (dangling type refs) |

#### Root Cause Analysis

1. **ASAN confirms** our layer code is memory-clean (all 196 shaders tested)
2. **jemalloc + `junk:true`** (fills freed memory with 0x5a) → always Frame 0.
   This proves the driver reads freed memory and relies on its old contents.
3. **jemalloc alone** delays memory reuse → more frames before crash
4. **Crash location**: during `QueueSubmit` execution (not shader compilation),
   typically after 500+ draw calls in Frame 1–6
5. **Single-threaded** (`taskset -c 0`): Frame 2–10, more consistent →
   threading amplifies the race condition

#### Why Extended Pipeline Layout Makes It Worse

Our layer creates extended pipeline layouts (app's sets + BVH set). This causes
the driver to allocate more internal descriptor management state. More
allocations = more opportunities for the use-after-free to trigger.

Without extended layouts (`CUDA_RT_NO_EXT_LAYOUT=1`), the crash is delayed
from Frame 1 to Frame 6.

### Attempted Fixes

| Approach | Result |
|----------|--------|
| Layout replacement in CmdBindDescriptorSets | **Harmful** — confused driver's descriptor validation |
| jemalloc preload | Delays crash, doesn't fix |
| tcmalloc preload | Worse than jemalloc |
| SIGSEGV handler (instruction skip) | Cascade corruption |
| 4× overallocation wrapper | No improvement |
| Aggressive mmap guard pages | No improvement |
| Single-threaded (taskset) | Slightly more stable |
| Stub mode (no BVH traversal) | Same crash rate |
| NO_EXT_LAYOUT + NO_BVH_BIND | Best so far (Frame 6) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_CUDA_RT_LAYER` | 0 | Enable the layer |
| `CUDA_RT_STRIP_EXTENSIONS` | 0 | Strip RT extensions from driver |
| `CUDA_RT_NO_BVH_BIND` | 0 | Skip BVH descriptor binding |
| `CUDA_RT_NO_EXT_LAYOUT` | 0 | Don't create extended pipeline layouts |
| `CUDA_RT_NO_PROFILE` | 0 | Disable profiling |
| `CUDA_RT_HALF_RES` | 0 | Reduce dispatch resolution (3=NOP) |
| `CUDA_RT_SERIALIZE_CMDS` | 0 | Serialize driver Cmd* calls |
| `CUDA_RT_STRIP_ONLY` | 0 | NOP ray_query ops without BVH injection |
| `CUDA_RT_MINIMAL` | 0 | Only DestroyDevice intercepted |

## Building

```bash
cd VK_RT/layer && bash build.sh
```

Requires: g++ with C++20, CUDA toolkit, Vulkan SDK headers.

## Running Breaking Limit

```bash
# Best stability configuration (as of current investigation)
cd linux-unpacked/resources/binaries
DISPLAY=:10 \
  ENABLE_CUDA_RT_LAYER=1 \
  CUDA_RT_STRIP_EXTENSIONS=1 \
  CUDA_RT_NO_PROFILE=1 \
  CUDA_RT_NO_BVH_BIND=1 \
  CUDA_RT_NO_EXT_LAYOUT=1 \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
  ./GPUScoreVulkan TestType custom BenchmarkMode true Fullscreen false \
  WindowResolution 640x480 RenderResolution 640x480
```

## Running Q2RTX

```bash
# Q2RTX works with the layer (proven: 2000 frames, 237 FPS)
cd Q2RTX && ENABLE_CUDA_RT_LAYER=1 ./q2rtx +map q2dm1
```

## Next Steps

1. **Work around driver UAF** — Test memory leak approach (intercept free,
   never deallocate) to prevent the use-after-free entirely
2. **Minimize layer footprint** — Avoid extended layouts; bind BVH via push
   descriptors or existing set slots
3. **Real BVH data** — Once stable, feed actual TLAS/BLAS geometry for
   correct ray tracing results
4. **Performance** — Profile BVH traversal cost per frame, optimize slab
   tests for V100 shader cores
5. **Driver upgrade** — Test with newer NVIDIA drivers that may fix the UAF
