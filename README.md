# VK_RT

CUDA ray tracing engine for V100. No RT cores, just brute-force compute — 1944 Mrays/s at 1080p.

Works as a Vulkan implicit layer that intercepts `vkCmdTraceRaysKHR` / `OpRayQuery*` and routes traversal through a custom CUDA BVH kernel instead of NVIDIA's generic software fallback.

## what's here

- `rt_engine_v*.cu` — the traversal kernel, iterated ~40 times from naive to 1944 Mr/s
- `layer_*.cpp` — Vulkan layer dispatch (intercepts AS builds, rewrites SPIR-V, replaces trace calls)
- `OptiX_Shim/` — drop-in for apps that call OptiX API directly
- `SLSS/` — temporal upscaler (motion vectors + jittered sampling, basically budget DLSS)
- `denoiser_*.cu` — U-Net trained in PyTorch, inference in raw CUDA (no framework at runtime)
- `*_bench.cu` / `*_bench.cpp` — various benchmarks

Q2RTX runs at 121-210 FPS through the layer on a V100. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## build

```
make -j$(nproc)
export VK_INSTANCE_LAYERS=VK_LAYER_CUSTOM_RT
export VK_LAYER_PATH=$PWD
```

Needs CUDA 12.x, Vulkan SDK, and a V100 (sm_70). Other Volta/Turing cards probably work but untested.
