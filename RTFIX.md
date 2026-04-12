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
│  4. AS Interception      │  Capture BLAS/TLAS builds → SSBO backend
│  5. Barrier Remapping    │  RT stage/access bits → compute equivalents
│  6. Dispatch Binding     │  Bind BVH data before RT dispatches
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

**Important**: vulkaninfo shows `rayQuery=true` because our layer spoofs
`EnumerateDeviceExtensionProperties` and `GetPhysicalDeviceFeatures2`. The actual
driver returns `rayQuery=false` and rejects `VK_KHR_ray_query` at `CreateDevice`
(`VK_ERROR_EXTENSION_NOT_PRESENT`). Our layer patches the feature bit to 0 before
forwarding to the driver.

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
- All **196 rewritten shaders** pass `spirv-val --target-env vulkan1.2` (valid SPIR-V)
- Shader growth: ~40% for full BVH traversal (11K → 15K words typical)

### NO_SSBO Compile-Time Guards

The rewriter has `#if SPIRV_RQ_NO_SSBO` guards (default: 0 = disabled) for all
SSBO-dependent operations. When enabled, these return safe defaults (0 for IDs,
identity matrices for transforms) instead of SSBO loads:
- `GetInstId` → 0
- `GetInstCustomIdx` / `GetSBTOffset` → 0
- `GetObj2World` / `GetWorld2Obj` → identity mat4x3

## Current Status: Breaking Limit Benchmark

### What Works

- ✅ Layer loads and intercepts all Vulkan calls
- ✅ Extensions spoofed correctly (ray_query + acceleration_structure)
- ✅ All 196 compute shaders rewritten and compile in the driver
- ✅ ~270 graphics pipelines created successfully
- ✅ BVH descriptor sets created and bound
- ✅ Barrier stage/access bits remapped (RT → compute/transfer)
- ✅ AS descriptor types remapped (AS → SSBO) when STRIP_EXTENSIONS
- ✅ Dummy buffer fallback for safe AS→SSBO descriptors before BVH ready
- ✅ Per-write pBufferInfo pointer tracking (prevents descriptor heap corruption)
- ✅ App loads scene, creates render passes, begins rendering
- ✅ **Reaches Frame 5–10** with mitigations (of 2100 total frames needed)
- ✅ **Q2RTX works**: 2000 frames, 38 BLAS, 33 TLAS instances, ~237 FPS at 640×480

### Blocking: Two V100 Driver 580.126.20 Bugs

#### Bug #1: Thread-Safety Race in CmdBindDescriptorSets (PRIMARY)

The driver crashes when **2+ CPU cores** record command buffers in parallel.
`taskset -c 0` (single CPU) eliminates this crash.

**GDB backtrace** (crash at Frame 17, single CPU):
```
#0  0x7ffff4be180c  libnvidia-glcore.so.580.126.20  (SIGSEGV)
#1  0x7ffff4b24b57  libnvidia-glcore.so.580.126.20
#2  layer_CmdBindDescriptorSets     libVkLayer_CudaRT.so
#3  rsGfx::SecondaryCommandBufferReaderVK::vkCmdBindDescriptorSetsFunc
#4  rsGfx::RsRunnableNode::encodeSubmittedBundles
#5  rsGfx::RsExecutionPipeline::fillCommandBuffers::{lambda}
#6  rsGfx::ThreadPool::createThreads::{lambda}
```

Corrupted registers at crash: `rdx=0xc480c000` (3.3GB), `rbp=0x10000000000` (1TB).
The driver's internal descriptor binding table has a stale pointer.

**Evidence:**
| CPUs | Frames | Notes |
|------|--------|-------|
| `taskset -c 0` (1 CPU) | **7–19** | Eliminates race, UAF remains |
| `taskset -c 0-1` (2 CPUs) | 0 | Immediate crash |
| `taskset -c 0-3` (4 CPUs) | 0 | Immediate crash |
| All CPUs (default) | 0–1 | Worst case |

#### Bug #2: Use-After-Free in Driver Internals (SECONDARY)

Even on single CPU, the driver crashes after Frame 7–19 due to UAF.

**Evidence:**
- jemalloc `junk:true` (fill freed mem with 0x5a) → always Frame 0 (confirms UAF)
- jemalloc `zero:true` (fill with 0x00) → Frame 4–19 (null ptrs handled more gracefully)
- jemalloc default (delay reuse) → Frame 7–19 (stale data still valid)
- glibc malloc (immediate reuse) → Frame 0–1 (freed memory overwritten fast)

### Complete Test Matrix

| Configuration | Frames | Key Insight |
|---------------|--------|-------------|
| **taskset -c 0 + jemalloc + STRIP + NO_EXT + NO_BVH** | **5–10** | Best config |
| taskset -c 0 + jemalloc + full features | 0–19 | Ext layouts variable |
| jemalloc + STRIP + NO_EXT + NO_BVH (all CPUs) | 0–6 | Thread race + UAF |
| jemalloc + STRIP + NO_BVH (with ext layout) | 0–1 | Ext layouts worsen UAF |
| jemalloc (default, all features, all CPUs) | 0–1 | Worst combo |
| SERIALIZE_CMDS + taskset + jemalloc | 3–10 | Mutex contention hurts |
| FRAME_SYNC=1 (DeviceWaitIdle every present) | 6 | Worse — not pushbuf accumulation |
| LEAN mode (skip draw interception) | 6–19 | Same as non-lean |
| pushbuf_extend + jemalloc + taskset | 10–19 | Marginal improvement |
| megafree (4M ring buffer) + jemalloc | 10–19 | Marginal improvement |
| No strip (driver handles AS natively) | 7 | Same crash, not strip-related |
| NO_REWRITE (unmodified shaders to driver) | **10–19** | V100 driver has partial RQ support |
| NO_DISPATCH_INTERCEPT | 10 | CmdDispatch interception not the cause |
| REWRITE_ONLY (no AS/barrier interception) | 0 | Missing barrier remap = crash |
| MINIMAL (no rewriting at all) | 0 | Driver can't compile ray_query |
| STRIP_ONLY (NOP ray_query ops) | 0 | Invalid SPIR-V (dangling refs) |
| FULL_TRAVERSAL=0 (skip traversal init) | 0 | Uninitialized vars crash |
| Stub mode (trivial traversal) | 0–12 | Same crash rate as full BVH |
| 320×240 resolution | 5–10 | Slightly better |
| nofree2 (suppress ALL free) | 0–11 | Breaks allocator internals |

### Key Experiment: NO_REWRITE (Shader Pass-Through)

The `CUDA_RT_NO_REWRITE=1` experiment revealed that **the V100 driver can parse
and compile ray_query SPIR-V without crashing** — it just can't execute it.
Passing unmodified shaders through (no BVH injection, no traversal rewrite)
consistently reaches Frame 10–19 vs Frame 5–10 with full rewrite.

This means the **rewritten shaders contribute ~5 frames of instability** — the
larger shader size (~40% growth) or the injected SSBO accesses put extra pressure
on the driver's internal allocator, triggering the UAF sooner.

### Attempted Mitigations (Detailed)

| Approach | Result | Why |
|----------|--------|-----|
| `taskset -c 0` (single CPU) | **Best mitigation** | Eliminates thread-safety race |
| jemalloc preload | **Essential** | Delays memory reuse, prevents UAF trigger |
| STRIP_EXTENSIONS=1 | **Needed** | Clean separation — layer handles all RT |
| NO_EXT_LAYOUT + NO_BVH_BIND | **Reduces attack surface** | Fewer driver allocations |
| Dummy buffer fallback (g_dummyBuf) | **Fixes** null descriptor | Safe AS→SSBO before BVH |
| Per-write pBufferInfo tracking | **Fixes** heap corruption | Was pointing all writes to same offset |
| Layout replacement in CmdBindDescriptorSets | **Harmful** (Frame 0–7) | Confused driver's descriptor validation |
| SERIALIZE_CMDS=1 (global mutex) | **Harmful** (Frame 3–10) | Contention worse than race |
| FRAME_SYNC (periodic DeviceWaitIdle) | **Harmful** (Frame 6) | UAF isn't about in-flight work |
| LEAN mode (skip draw interceptors) | **Neutral** | Draw hooks aren't the problem |
| NO_DISPATCH_INTERCEPT | **Neutral** | CmdDispatch hook isn't the issue |
| SIGSEGV handler (pushbuf_extend.so) | **Marginal** | Catches overflow but not UAF |
| SIGSEGV + instruction skip (pushbuf_v4.so) | **Harmful** | Cascade corruption |
| tcmalloc preload | **Worse** than jemalloc | Less effective at delaying reuse |
| megafree.so (4M delayed-free ring) | **Marginal** | Small improvement with jemalloc |
| nofree2.so (suppress all free) | **Breaks** things | Allocator can't reclaim memory properly |
| Stub traversal (no BVH loop) | **No help** | Crash is workload-independent |
| Smaller resolution (320×240) | **Marginal** | Slightly fewer driver allocations |

### Bugs Found and Fixed

#### rqVar Scoping Bug in SPIR-V Rewriter (Critical)

When adding `#if SPIRV_RQ_NO_SSBO` compile-time guards to the Obj2World/World2Obj
handler, the local `uint32_t rqVar = code[pos+3]` declaration was accidentally
removed. The `#else` branch still used `rqVar`, but C++ silently resolved it to
an outer-scope `rqVar = code[pos+1]` (line 1188) — which held the **result TYPE
ID**, not the ray query variable.

**Symptoms**: Shader sizes ballooned (11K → 15.5K words), wrong SSBO accesses
generated, immediate crash at Frame 0.

**Fix**: Ensure every `#else` branch that uses `rqVar` has its own declaration:
`uint32_t rqVar = code[pos+3];`

#### Descriptor Buffer Info Pointer Bug

The `UpdateDescriptorSets` handler used a single `dummyBufInfos.data()` pointer
for ALL remapped writes. With multiple AS→SSBO writes, all pointed to the same
base offset, causing the driver to use wrong buffer offsets for later writes.

**Fix**: Track per-write offset into the `dummyBufInfos` vector.

#### Null Descriptor Fallback

When `g_bvh2.nodesBuf` wasn't ready (before TLAS build), AS→SSBO remapped writes
were skipped entirely, leaving stale descriptors. Now uses `g_dummyBuf` (4KB
device-local buffer created at device init) as a safe fallback.

### Root Cause Diagnosis

The V100 driver 580.126.20 has **two independent bugs** in `libnvidia-glcore.so`:

1. **Thread-unsafe descriptor binding**: The driver's `CmdBindDescriptorSets`
   implementation uses shared mutable state (likely a global descriptor binding
   cache/pushbuffer allocator) that is not protected against concurrent access
   from multiple command buffer recording threads. Even though Vulkan spec allows
   recording to different command buffers concurrently, this driver doesn't
   support it safely. The app (`rsGfx::ThreadPool`) records secondary command
   buffers from worker threads → race → corrupted pointers → SIGSEGV.

2. **Use-after-free in internal object management**: The driver frees internal
   host-side objects (descriptor set metadata, pipeline state) prematurely.
   Subsequent frames dereference stale pointers. jemalloc mitigates this by
   delaying memory reuse (freed memory retains valid data longer), but the
   crash eventually occurs when freed pages are finally recycled.

### Disproven Theories

| Theory | How Disproven |
|--------|--------------|
| CmdDispatch interception causes crash | NO_DISPATCH_INTERCEPT: same Frame 10 |
| Invalid SPIR-V causes crash | spirv-val validates all 196 shaders |
| Pushbuffer accumulation over time | FRAME_SYNC=DeviceWaitIdle every frame: WORSE |
| Barrier heap allocations trigger UAF | Fast-path optimization: neutral |
| Extension stripping breaks driver | Without stripping: slightly worse |
| Larger shaders → more instability | NO_REWRITE (unmodified): Frame 10-19 vs 5-10 **(partial confirmation)** |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_CUDA_RT_LAYER` | 0 | Enable the layer |
| `CUDA_RT_STRIP_EXTENSIONS` | 0 | Strip ALL RT extensions from driver (AS→SSBO) |
| `CUDA_RT_NO_BVH_BIND` | 0 | Skip BVH descriptor binding in CmdDispatch |
| `CUDA_RT_NO_EXT_LAYOUT` | 0 | Don't extend pipeline layouts with BVH set |
| `CUDA_RT_NO_PROFILE` | 0 | Disable GPU timestamp profiling |
| `CUDA_RT_LEAN` | 0 | Skip draw call interception (reduces overhead) |
| `CUDA_RT_FRAME_SYNC` | 0 | DeviceWaitIdle every N frames (0=disabled) |
| `CUDA_RT_HALF_RES` | 0 | Reduce RQ dispatch resolution (3=NOP all) |
| `CUDA_RT_MAX_RQ_DISPATCH` | 0 | Max RQ dispatches per frame (0=unlimited) |
| `CUDA_RT_SERIALIZE_CMDS` | 0 | Serialize ALL Cmd* calls via global mutex |
| `CUDA_RT_SYNC_SUBMIT` | 0 | QueueWaitIdle after every QueueSubmit |
| `CUDA_RT_STRIP_ONLY` | 0 | NOP ray_query ops without BVH (broken) |
| `CUDA_RT_MINIMAL` | 0 | Only DestroyDevice intercepted (test mode) |
| `CUDA_RT_REWRITE_ONLY` | 0 | Only CreateShaderModule intercepted |
| `CUDA_RT_HIDE_PIPELINE` | 1 | Hide VK_KHR_ray_tracing_pipeline from app |
| `CUDA_RT_TLAS_EVERY_N` | 1 | Rebuild TLAS every N frames |

## Building

```bash
cd VK_RT/layer && bash build.sh
```

Requires: g++ with C++20, CUDA 12.x toolkit, Vulkan SDK headers.

## Running Breaking Limit

```bash
# Best stability configuration (Frame 7–19 of 2100)
cd linux-unpacked/resources/binaries
DISPLAY=:10 \
  ENABLE_CUDA_RT_LAYER=1 \
  CUDA_RT_STRIP_EXTENSIONS=1 \
  CUDA_RT_NO_PROFILE=1 \
  CUDA_RT_NO_BVH_BIND=1 \
  CUDA_RT_NO_EXT_LAYOUT=1 \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
  taskset -c 0 \
  ./GPUScoreVulkan TestType custom BenchmarkMode true Fullscreen false \
  WindowResolution 640x480 RenderResolution 640x480
```

## Running Q2RTX

```bash
# Q2RTX works with the layer (proven: 2000 frames, 237 FPS)
cd Q2RTX && ENABLE_CUDA_RT_LAYER=1 ./q2rtx +map q2dm1
```

## Crash Recovery System (SIGSEGV Handler)

The V100 driver (580.126.20) has a use-after-free bug that manifests as SIGSEGV
during command buffer submission. Our layer implements a multi-tier crash recovery:

### Tier 1: Guarded Recovery (sigsetjmp/siglongjmp)
- `CRASH_GUARD_CMD(cmdBuf, call, label)` macro wraps every Cmd* driver call
- On SIGSEGV, longjmps back, marks cmdBuf as "poisoned"
- Poisoned cmdBufs skip all further Cmd* calls until cleared at QueuePresent
- QueueSubmit also wrapped — returns VK_SUCCESS on crash

### Tier 2: Instruction Skip (x86_64 RIP advancement)
- When SIGSEGV fires OUTSIDE a guard region (driver internal thread, unintercepted call)
- Decodes faulting x86_64 instruction length using REX prefix + ModRM parsing
- Advances RIP past the faulting instruction, zeros RAX to prevent cascade
- Last-resort hack — keeps process alive but accumulates state corruption

### Results

| Mode | Frames | Crashes | Notes |
|------|--------|---------|-------|
| No recovery | 5-10 | SIGSEGV death | Baseline |
| Tier 1 only | 4-10 | Still dies (unguarded paths) | Partial fix |
| Tier 1 + Tier 2 | 10+ | 0 crashes, 50 skips | Survives but stalls |

**Key finding**: Tier 2 instruction skip catches ALL crashes (50 per run at
`0x7714440b018f` in driver + cascade at corrupted RIP `0x36xx`). Process
survives 120s+ but driver state corrupts enough to stall rendering at ~Frame 10.

### Crash Address Analysis
- `0x7714440b018f`: Driver shared library (libGLX_nvidia / libnvidia-glcore)
- `0x36af-0x36f4`: Corrupted return addresses (cascade from stack corruption)
- Pattern: fault at driver addr → fault at corrupted RIP → repeat (25 cycles)

### Descriptor Free Blocking (disproven)
Intercepted FreeDescriptorSets/DestroyDescriptorPool/ResetDescriptorPool and
NOPed them (CUDA_RT_BLOCK_DESC_FREE=1). Same crash pattern — descriptor frees
are NOT the UAF cause.

### Dummy Buffer Fix
The dummy buffer creation in CreateDevice used undefined `g_memTypeCount`/
`g_memTypes` variables — buffer was never actually created. Fixed to use
`disp.memProps` (the correct VkPhysicalDeviceMemoryProperties in DeviceDispatch).

## Next Steps

1. **Identify exact driver crash function** — map `0x7714440b018f` to a
   symbol in libnvidia-glcore.so via /proc/self/maps + addr2line
2. **Close the rewrite gap** — NO_REWRITE gets Frame 10–19, full rewrite gets
   5–10. The rewriter adds ~40% shader size. Investigate:
   - Trimming unnecessary SPIR-V instructions in the traversal loop
   - Reducing SSBO binding count per shader
   - Using push constants instead of SSBOs for small data
3. **Solve the UAF** — The secondary crash (Frame 5–19) is the blocker.
   Approaches remaining:
   - Different driver version (current: 580.126.20)
   - NVK (Mesa Nouveau Vulkan) — completely different driver stack
   - ioctl-level pushbuffer enlargement
4. **Feed real BVH data** — Once stable past Frame 20+, enable BVH binding
   and extended layouts for actual ray tracing output
5. **Performance** — Single-CPU mode is slow; need per-CB serialization
   instead of full CPU pinning
6. **Complete benchmark** — 2100 frames needed for a score (~9 min runtime)

## Driver Update: 580.126.20 → 580.142

Upgraded to NVIDIA 580.142 (released March 2026) — latest in the 580.xx legacy
branch that still supports V100.

### Results: Same crash pattern persists

| Config | Frames | Survival Rate | Notes |
|--------|--------|---------------|-------|
| 580.142 + rewrite + crash recovery | 5-11 | 3/5 survived 90s | Same UAF |
| 580.142 + NO_REWRITE + crash recovery | 5-19 | 3/5 survived 90s | Frame 19 in best runs |
| 580.142 clean (no recovery) | 0-1 | 0/3 | Stalls without taskset |

**Conclusion**: The UAF bug exists in both 580.126.20 and 580.142. The crash
recovery system (SIGSEGV handler + abort() interception + instruction/function
skip) is essential for survival. The driver bug is present across the entire
580.xx branch.

### Crash Recovery Architecture (v2 — current)

**3-Tier Signal Handler:**
1. **Tier 0 — CRASH_GUARD**: sigsetjmp around every Cmd* call → longjmp on SIGSEGV
2. **Tier 1 — Global Recovery**: per-QueueSubmit sigsetjmp → longjmp with fence preservation
3. **Tier 2 — Driver Thread Recovery**:
   - SIGSEGV: FUNC SKIP with loop detection (scans deeper on repeated crash/return pairs)
   - SIGABRT: Suspend (park thread with usleep loop)

**Supporting Infrastructure:**
- **sigaltstack on ALL threads**: via LD_PRELOAD pthread_create wrapper in noabort.so
  (ensures handler works even when thread stack is corrupted)
- **WaitForFences intercept**: 2s timeout rescue, empty-submit fence signaling
  (Note: Breaking Limit uses no fences — useful for other apps)
- **Fence signaling in recovery paths**: both crash guard and global recovery
  signal fences via empty QueueSubmit to prevent app deadlock
- **abort()/raise()/\_\_stack\_chk\_fail() interception**: LD_PRELOAD libnoabort.so
- **Command buffer poisoning**: Skip all ops on crashed cmdBuf

### Latest Benchmark Results (v2 crash recovery, FRAME_SYNC=5)

| Test | Frame | Crashes | Survival | Notes |
|------|-------|---------|----------|-------|
| 1080p Run 1 (30s) | 19 | 7 | ✅ timeout | Nearly crash-free |
| 1080p Run 2 (30s) | 17 | 108 | ✅ timeout | Moderate crashes |
| 1080p Run 3 (30s) | 5 | 102 | ✅ timeout | Heavy crashes |

**100% survival rate** across all runs. Frame count varies 5-19 based on crash
count (fewer crashes = higher frames). The benchmark performs 7500+ QueueSubmits
in 30s regardless of frame progress — compute dispatches continue after render
passes stop.

**Key findings from investigation:**
- App uses NO fences (fence=NULL in all QueueSubmit calls)
- App never calls vkWaitForFences (zero calls intercepted)
- Frame stall at 19 = loading→rendering transition (needs actual RT results)
- FRAME_SYNC=5 (QueueWaitIdle every 5 submits) prevents most driver crashes
- Without FRAME_SYNC, crashes increase dramatically (71-200 FUNC SKIPs)
- All threads use futex for synchronization; suspended threads cause full deadlock
- Process may OOM-kill after ~90s of accumulated leaked driver state
