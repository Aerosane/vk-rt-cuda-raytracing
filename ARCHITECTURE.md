# V100 Custom RT Pipeline — Vulkan Layer Architecture

## The Question
Can we make **any** Vulkan RT application (games, Blender Cycles, etc.) automatically
use our 1944 MR/s CUDA engine instead of NVIDIA's generic software RT?

## Answer: YES — via a Vulkan Implicit Layer

### How the NVIDIA Driver Does Software RT on V100 Today

```
App calls vkCmdTraceRaysKHR() or uses rayQueryEXT in shader
    │
    ▼
NVIDIA Driver (closed-source, libGLX_nvidia.so.0)
    │
    ├─ vkBuildAccelerationStructuresKHR → GPU-parallel BVH build
    │   └─ Driver's internal BVH format (opaque, ~90 bytes/tri)
    │
    ├─ SPIR-V with OpRayQuery* → Driver compiles to PTX/SASS
    │   ├─ OpRayQueryInitializeKHR → sets up ray + BVH root
    │   ├─ OpRayQueryProceedKHR   → ONE traversal step (AABB or tri test)
    │   └─ OpRayQueryGetIntersection* → read hit results
    │
    └─ Execution: generic CUDA compute kernel on 5120 FP32 cores
        └─ No constant memory tricks, no texture cache, no octant sort
```

### How Our Layer Would Replace It

```
App calls Vulkan RT functions
    │
    ▼
VK_LAYER_CUSTOM_RT (our implicit layer, loaded automatically)
    │
    ├─ INTERCEPT: vkBuildAccelerationStructuresKHR
    │   ├─ Read triangle data from app's VkBuffer
    │   ├─ Build our SAH BVH (with CUDA kernel for speed)
    │   ├─ Store as float4 packed nodes in a VkBuffer/SSBO
    │   ├─ Top 2040 nodes flagged for constant memory
    │   └─ DFS treelet reorder for cache coherence
    │
    ├─ INTERCEPT: vkCreateComputePipelines / vkCreateRayTracingPipelinesKHR
    │   ├─ Parse SPIR-V bytecode
    │   ├─ Find OpRayQuery* instructions (5 ops to replace):
    │   │   ├─ OpRayQueryInitializeKHR → load ray, set stack[0]=root
    │   │   ├─ OpRayQueryProceedKHR   → OUR traversal loop (inline SPIR-V)
    │   │   ├─ OpRayQueryGetIntersectionTypeKHR → read hitTri
    │   │   ├─ OpRayQueryGetIntersectionTKHR → read hitT
    │   │   └─ OpTypeRayQueryKHR → struct { stack, hitT, hitTri, ... }
    │   ├─ Inject SSBO binding for our BVH buffer
    │   ├─ Replace ray query ops with our BVH traversal in SPIR-V
    │   └─ Pass modified SPIR-V to real driver
    │
    ├─ INTERCEPT: vkCmdTraceRaysKHR
    │   ├─ Bind our BVH SSBO
    │   └─ Dispatch (real driver compiles our SPIR-V traversal)
    │
    └─ PASS-THROUGH: everything else → real NVIDIA driver
```

## The 5 SPIR-V Ray Query Ops to Replace

From actual compiled shader (`spirv-dis`):
```
OpCapability RayQueryKHR                              → REMOVE, add compute caps
%42 = OpTypeRayQueryKHR                               → REPLACE with struct type
OpRayQueryInitializeKHR %rq %tlas ...origin...dir     → REPLACE with ray setup
%60 = OpRayQueryProceedKHR %bool %rq                  → REPLACE with traversal step
%63 = OpRayQueryGetIntersectionTypeKHR %uint %rq %1   → REPLACE with hitTri check
%72 = OpRayQueryGetIntersectionTKHR %float %rq %1     → REPLACE with hitT read
```

## Implementation Plan

### Phase 1: BVH Build Interception (Easiest)
- Hook `vkBuildAccelerationStructuresKHR`
- Read vertex/index buffers from app
- Build our SAH BVH in parallel CUDA buffer
- Store alongside driver's BLAS (dual-format)
- **Complexity**: Low — just add a CUDA kernel call alongside the driver call

### Phase 2: SPIR-V Rewriting Engine (Hardest)
- Parse SPIR-V binary format (well-documented, ~500 opcodes)
- Use SPIRV-Tools (already installed) for parsing/validation
- Replace 5 ray query ops with ~200 SPIR-V instructions for:
  - BVH node fetch from SSBO
  - AABB slab test (6 mul, 6 min/max, 3 cmp)
  - Möller-Trumbore triangle test
  - Stack push/pop (array in function scope)
- Inject new descriptor set binding for BVH SSBO
- **Complexity**: HIGH — this is the core engineering challenge
- **Reference**: Mesa RADV does exactly this in `radv_nir_lower_ray_queries()`

### Phase 3: Layer Registration
- Create JSON manifest: `/etc/vulkan/implicit_layer.d/VK_LAYER_CUSTOM_RT.json`
- Shared library: `libVkLayer_CustomRT.so`
- Auto-loaded for all Vulkan apps
- Environment variable to enable/disable: `CUSTOM_RT_ENABLE=1`

## What We CANNOT Do in a Layer

1. **Constant memory** — SPIR-V has no constant memory concept; closest is UBO
   - Workaround: Use UBO for top BVH nodes (similar broadcast semantics)
2. **Texture cache fetch** — SPIR-V texelFetch from buffer texture
   - Workaround: Use `OpImageFetch` from `textureBuffer` (maps to TEX unit)
3. **INT32 ∥ FP32** — happens automatically on Volta, no special SPIR-V needed
4. **Register stack** — SPIR-V uses function-scope variables; driver decides reg/local
   - Small arrays (32 ints) usually stay in registers on NVIDIA

## Performance Expectations

| Aspect | Driver's Software RT | Our Layer RT |
|--------|---------------------|--------------|
| BVH Build | 7.5ms/100K (GPU) | Same (we hook, not replace) |
| BVH Quality | Driver SAH | Our SAH (similar) |
| Traversal | Generic kernel | Optimized SPIR-V with UBO top nodes |
| Cache layout | Unknown | DFS treelet reorder |
| Ray coherence | None | App-dependent (can't sort in layer) |
| Expected gain | Baseline | **10-30% faster** (UBO + treelet) |

The gain from a layer is modest (~10-30%) because:
- We lose constant memory (UBO is close but not identical)
- We lose explicit texture cache control
- We can't sort rays (that's the app's job)
- The driver's generic kernel is already decent

## The REAL Win: For Apps We Control

For custom renderers (not arbitrary apps), the win is massive:
- Full CUDA control: constant mem, texture cache, register stack
- Ray sorting: direction-octant coherence
- **1944 MR/s vs driver's ~800-1200 MR/s** (estimated 1.6-2.4× gain)
- CUDA-Vulkan interop for zero-copy buffer sharing

## Files Involved

```
VK_RT/
├── layer/
│   ├── vk_layer_custom_rt.c     # Layer entry point
│   ├── spirv_rewriter.c         # SPIR-V ray query → traversal
│   ├── bvh_builder.cu           # CUDA SAH BVH builder
│   ├── manifest.json            # Vulkan layer manifest
│   └── Makefile
├── cuda_rt_v7.cu                # Reference CUDA engine
└── run_comparison.sh            # Benchmarks
```
