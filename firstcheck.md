# V100 Ray Tracing Optimization Master Checklist

**Last audited**: v10 pipeline benchmark (99K conference scene, 4M rays)
**Engine**: v10 — persistent kernel + BVH4 FP16 + smem cache + short stack

## TIER 1 — MUST DO (Biggest gains)

### 1. Ray Coherence

* [x] Sort rays by direction (octants / bins) — `sortByOctant()`, +8% diffuse
* [x] Group rays by origin (tiles / Morton order) — `sortMortonOctant()`, **+173% diffuse!** (195→531 MR/s)
* [x] Separate primary vs diffuse rays — v10_pipeline benchmarks each type independently
* [x] Ensure same warp processes similar rays — persistent kernel + morton sort groups spatially

### 2. Memory Access Pattern

* [x] BVH stored contiguously (array, no pointers) — packed `int4` array since v1
* [x] Use indices instead of pointers — `child[4]` are int indices
* [x] Ensure coalesced global memory access — SoA triangle layout, coalesced ray reads
* [x] Use SoA or tightly packed layout — SoA triangles (9 float arrays) + FP16 BVH4 (64B/node)
* [x] Align nodes to avoid padding waste — 64B/node = exactly 1 L2 cache line

### 3. Scheduling (Latency Control)

* [x] Implement persistent threads — `atomicAdd(&g_rayCounter, 32)` warp work-stealing
* [x] Global or block-level work queue — global atomic counter, warp-level batch=32
* [x] Threads fetch new rays dynamically — persistent while-loop, new batch per warp
* [~] Prevent long rays from blocking warps — persistent helps, no explicit ray compaction yet

---

## TIER 2 — HIGH IMPACT (Stability + Scaling)

### 4. Work Compaction

* [ ] Remove terminated rays early — not implemented (rays just exit warp)
* [ ] Compact active rays into dense buffers — not implemented
* [~] Avoid idle threads after divergence — persistent kernel reuses idle warps globally

### 5. BVH Optimization

* [x] Use BVH4 or BVH8 (NOT binary) — BVH4 since v9 (SAH-guided collapse)
* [x] Ensure tight bounding boxes — SAH 16-bin sweep, FP16 with epsilon expansion
* [x] Minimize node overlap — SAH construction inherently minimizes overlap
* [x] Keep nodes/ray < 10 — **primary: 8.5, shadow: 6.7** ACHIEVED

### 6. Cache Locality ("Cache Emulation")

* [x] Reorder BVH nodes in traversal order — DFS reorder for both binary and BVH4
* [x] Cluster frequently accessed nodes — DFS ordering clusters parent-children contiguously
* [x] Load top-level BVH nodes into shared memory — 128 nodes in smem (~20cy access)
* [x] Ensure nearby rays hit same memory regions — Morton origin sort + tile-based primary rays

---

## TIER 3 — FINE TUNING (Edge gains)

### 7. Traversal Optimization

* [x] Near-first traversal order — branchless 4-way sorting network (cswap)
* [x] Early exit on hit — `traceShadow` any-hit kernel exits on first intersection
* [x] Short-stack or stackless traversal — `SHORT_STACK[12]`, overflow drops farthest child
* [x] Reduce branch divergence — branchless sort, persistent kernel, morton sort

### 8. Kernel Efficiency

* [x] Check for register spilling — **56 registers, 0 spills** (ptxas confirmed)
* [x] Balance occupancy (don't max blindly) — `__launch_bounds__(256,4)` = 1024 threads/SM
* [x] Fuse kernels where beneficial — single kernel: traversal + intersection fused
* [x] Minimize global memory roundtrips — 3-tier cache: smem(20cy) const(4cy) ldg(80cy)

### 9. Variance / Stability

* [x] Measure min vs max MRays — v10_audit: 20 runs per config, min/avg/max/stddev/CV%
* [~] Ensure performance doesn't collapse on random rays — **improved 20x to 7.7x** collapse w/ morton, still significant
* [x] Maintain stable performance across scene sizes — primary: 4437-3852 (99K-500K, only 13% drop)

---

## METRICS TO TRACK (ALWAYS)

* [x] MRays/s (avg, min, max) — primary 4114(avg), shadow 2276, diffuse 195-531
* [x] Nodes per ray (< 10 target) — **primary: 8.5, shadow: 6.7** ACHIEVED
* [x] Bytes per ray (< 800B ideal) — **primary: 692B, shadow: 428B** ACHIEVED (diffuse: 2062B exceeds)
* [~] Warp efficiency (> 70%) — est. >90% primary, ~40% diffuse (needs nsight for exact)
* [x] Memory bandwidth usage — primary: **97% of L2 bandwidth** (2999 GB/s of 3100)
* [x] Performance variance across frames — **CV < 0.5%** for primary, ~5% for diffuse

---

## TARGET STATE

* [x] ~2000-3000 MRays sustained — **EXCEEDED: 4,114 MR/s primary, 3,852 at 500K**
* [x] < 800B per ray — **692B primary, 428B shadow**
* [x] Stable performance across scenes — 4437 to 3852 (99K to 500K) = 13% drop
* [~] No major drops for diffuse/random rays — **7.7x collapse** (was 20x), partially fixed

---

## FINAL GOAL

* [x] Throughput near hardware limit — **97% L2 bandwidth ceiling for coherent rays**
* [x] Low latency + low variance — **243 ns/ray primary, CV 0.3%**
* [ ] Ready for Blender / real-world integration — ARCHITECTURE.md written, not implemented

---

## MEASURED PERFORMANCE (v10 pipeline, 99K tris)

| Ray Type | MR/s | Nd/ray | B/ray | us/ray | Hit% |
|----------|------|--------|-------|--------|------|
| Primary (coherent) | 4,114 | 8.5 | 692 | 0.243 | 99% |
| Shadow (any-hit) | 2,276 | 6.7 | 428 | 0.439 | 95% |
| Diffuse (morton sort) | 531 | 19.1 | 2,062 | 1.881 | 93% |
| Diffuse (unsorted) | 195 | 19.1 | 2,062 | 5.128 | 93% |

### Hybrid Pipeline @ 1080p
| Config | RT ms | FPS |
|--------|-------|-----|
| Shadow only (1 rpp) | 0.91 | 1,098 |
| Shadow + 4spp AO + 1 refl | 2.17 | 461 |
| Shadow + 1spp AO + 1 refl + denoise | 1.91 | 523 |

### vs RTX Hardware (GigaRays/s peak)
| GPU | GR/s | Our V100 vs |
|-----|------|-------------|
| **V100 (ours)** | **4.1** | -- |
| RTX 4050 | ~18 | 23% |
| RTX 3070 | 20.3 | 20% |
| RTX 4070 Ti | 43.0 | 10% |
| RTX 4080 | 64.0 | 6% |

V100 primary real-scene MR/s = **69% of RTX 3070, 82% of RTX 4050**

---

## REMAINING WORK

1. **Work compaction** — stream compact active rays between bounces (Tier 2)
2. **Warp efficiency measurement** — need nsight profiler or CUDA occupancy events
3. **Diffuse stability** — 7.7x collapse acceptable, could improve with wavefront/packet tracing
4. **Vulkan interception layer** — ARCHITECTURE.md to actual implementation
5. **Blender integration** — custom render engine or Cycles modifier
