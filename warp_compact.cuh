// Warp-level stream compaction primitives for CUDA ray tracing
//
// Used between bounces to compact active rays into dense warps,
// eliminating idle-thread divergence from rays that missed geometry.
//
// Requirements:
//   - sm_30+ (warp vote/shuffle intrinsics)
//   - blockDim.x must be a multiple of 32 (full warps only)
//   - __ballot_sync(0xFFFFFFFF, ...) — all 32 lanes must participate

#pragma once
#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════════════
// WARP-LEVEL COMPACTION
// ═══════════════════════════════════════════════════════════════════
// Compute this thread's index in a compacted output within one warp.
//
// active:       1 if this thread's element survives, 0 otherwise
// total_active: (output) total surviving elements in this warp
// Returns:      compacted index for this lane [0..total_active)
//               (only meaningful when active==1)
__device__ __forceinline__ int warpCompact(int active, int& total_active) {
    unsigned mask  = __ballot_sync(0xFFFFFFFF, active);
    total_active   = __popc(mask);
    unsigned lower = (1U << (threadIdx.x & 31)) - 1U;
    return __popc(mask & lower);
}

// ═══════════════════════════════════════════════════════════════════
// BLOCK-LEVEL COMPACTION
// ═══════════════════════════════════════════════════════════════════
// Compact active elements across an entire thread-block by combining
// per-warp compaction with a shared-memory prefix sum over warp totals.
//
// active:  1 if this thread's element survives, 0 otherwise
// out_idx: (output) this thread's global-within-block index [0..total)
//          (only meaningful when active==1)
// smem:    shared memory array — needs at least (blockDim.x/32 + 1) ints
// Returns: total number of active elements in the entire block
//
// Complexity: O(log W) shuffles + 1 syncthreads, where W = warps/block
__device__ int blockCompact(int active, int& out_idx, int* smem) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int nWarps  = blockDim.x >> 5;

    // Step 1: Per-warp compaction via ballot + popcount
    unsigned mask  = __ballot_sync(0xFFFFFFFF, active);
    int warp_total = __popc(mask);
    int warp_idx   = __popc(mask & ((1U << lane) - 1U));

    // Step 2: Write per-warp totals to shared memory
    if (lane == 0) smem[warp_id] = warp_total;
    __syncthreads();

    // Step 3: Exclusive prefix sum over warp totals (first warp only)
    //         Kogge-Stone parallel scan via __shfl_up_sync
    if (warp_id == 0) {
        int val = (lane < nWarps) ? smem[lane] : 0;

        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            int n = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (lane >= offset) val += n;
        }
        // val is now the inclusive prefix sum.
        // Store the grand total (inclusive prefix of the last valid warp).
        if (lane == nWarps - 1) smem[nWarps] = val;

        // Convert inclusive → exclusive (shift down by 1, first = 0)
        int exc = __shfl_up_sync(0xFFFFFFFF, val, 1);
        smem[lane] = (lane == 0) ? 0 : exc;
    }
    __syncthreads();

    // Step 4: Final output index = warp's prefix + intra-warp index
    out_idx = smem[warp_id] + warp_idx;
    return smem[nWarps];
}
