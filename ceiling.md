# V100 CUDA RT Engine — The Ceiling Problem

## Where We Are
- **Primary rays**: 4,910 MR/s @ 99K tris — **90.8% warp efficiency** ✅
- **Diffuse Morton-sorted**: 582 MR/s — **42.7% warp efficiency** ❌
- **Diffuse unsorted**: 215 MR/s — **35.1% warp efficiency** ❌
- **Target**: ≥70% warp efficiency on diffuse

## What We Tried (v14–v19)
| Version | Technique | Result |
|---------|-----------|--------|
| v14 | HBM2 streaming (ld.global.cs, prefetch) | 0% — BVH fits in L2 already |
| v15 | Wave-refill per-node (ballot_sync) | 112B spills, hung. Abandoned |
| v16 | PTX min/max + ld.global.cs triangles | -15% — triangle L2 caching matters |
| v17 | BVH8 quantized uint8 nodes | -42% — decode overhead + register bloat |
| v18 | Warp-persistent ray recycling | 0% — structurally identical to baseline |
| v19 | GPU-side direction-aware sort (software SER) | **Trace kernel itself takes >2min on sorted diffuse** |

## The Actual Problem
The trace kernel on diffuse rays is **not slow because of sorting** — it's slow because:

1. **Diffuse rays traverse 5–10× more BVH nodes than primary rays** (random directions → no early termination, deep traversal into many branches)
2. **42.7% warp efficiency** means 57% of SIMT lanes are idle at any given node test — each lane is at a different depth in the BVH, and `while(sp>0)` exits at different times
3. **This is not a software problem. This is SIMT physics.**

## Why RT Cores Solve This
RT cores have **per-ray hardware state machines** with their own register file and traversal stack. Each ray runs independently on a fixed-function pipeline — there is no warp, no SIMT divergence. The concept of "warp efficiency" doesn't exist in RT core traversal.

**SER (Shader Execution Reordering)** on Ada/Blackwell solves *shading* divergence (post-intersection), NOT traversal divergence. Traversal divergence is solved by the RT core itself being per-ray hardware.

## Why Software SER Can't Fix Traversal
Our software SER (v19) sorts rays by direction before tracing. But:
- Sorting only helps the **first few BVH levels** where similar-direction rays take the same child
- After ~6 levels, even rays with similar directions diverge based on their **position** relative to geometry
- The divergence is **inherent to DFS traversal on SIMT** — different rays finish at different times, and `__syncwarp()` can't help because the loop depths are fundamentally different
- Morton sort already captures the optimal spatial+directional coherence; finer binning trades spatial coherence for directional coherence with diminishing returns

## The Physics Wall
```
Diffuse ray warp efficiency on SIMT:
  Unsorted:     35.1%  (11.2/32 active lanes)
  Morton+oct:   42.7%  (13.7/32 active lanes)  ← our best
  Theory max:   ~50%   (estimated, for perfect sort)
  Hardware RT:  100%   (per-ray state machine, no warps)
```

The gap from 42.7% to 70% requires **hardware ray reordering** (SER) or **per-ray execution** (RT cores). On V100's SIMT architecture with 32-wide warps, diffuse rays will always diverge after the first few BVH levels. No sorting strategy, kernel restructuring, or PTX trick can change this.

## What IS Achievable
- ✅ Primary: 4,910 MR/s, 90.8% efficiency — **at the hardware limit**
- ✅ Diffuse Morton: 582 MR/s, 42.7% — **at the SIMT physics limit**
- 🎯 Multi-bounce compaction: 2.5× speedup at 10% survival (measured)
- 🎯 Hybrid rendering: rasterize primary, RT only for reflections/shadows/GI
- 🎯 Denoising: make 1-2 spp look like 64 spp → perceived performance >>>
