#!/bin/bash
# Build the CUDA RT Vulkan Layer and test app
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== Building CUDA BVH backend ==="
/usr/local/cuda/bin/nvcc -c -O3 -arch=sm_70 --compiler-options=-fPIC \
  -Wno-deprecated-gpu-targets \
  cuda_bvh_backend.cu -o cuda_bvh_backend.o

echo "=== Building RasterBoost upscale engine ==="
/usr/local/cuda/bin/nvcc -c -O3 -arch=sm_70 --compiler-options=-fPIC \
  -Wno-deprecated-gpu-targets \
  -I/usr/include/x86_64-linux-gnu \
  rasterboost_upscale.cu -o rasterboost_upscale.o

echo "=== Building RasterBoost post-FX engine ==="
/usr/local/cuda/bin/nvcc -c -O3 -arch=sm_70 --compiler-options=-fPIC \
  -Wno-deprecated-gpu-targets \
  rasterboost_postfx.cu -o rasterboost_postfx.o

echo "=== Building RasterBoost frame generation ==="
/usr/local/cuda/bin/nvcc -c -O3 -arch=sm_70 --compiler-options=-fPIC \
  -Wno-deprecated-gpu-targets \
  rasterboost_framegen.cu -o rasterboost_framegen.o

echo "=== Building VK_RT TensorRT denoiser ==="
/usr/local/cuda/bin/nvcc -c -O3 -arch=sm_70 --compiler-options=-fPIC \
  -Wno-deprecated-gpu-targets \
  -I/usr/include/x86_64-linux-gnu \
  rt_denoise.cu -o rt_denoise.o

echo "=== Building RT IR executor ==="
/usr/local/cuda/bin/nvcc -c -O3 -arch=sm_70 --compiler-options=-fPIC \
  -Wno-deprecated-gpu-targets \
  rt_ir_exec.cu -o rt_ir_exec.o

echo "=== Building Neural Radiance Cache (WMMA) ==="
/usr/local/cuda/bin/nvcc -c -O3 -arch=sm_70 --compiler-options=-fPIC \
  -Wno-deprecated-gpu-targets -diag-suppress=177 \
  nrc.cu -o nrc.o

echo "=== Building CUDA-Vulkan interop scaffolding ==="
/usr/local/cuda/bin/nvcc -c -O3 -arch=sm_70 --compiler-options=-fPIC \
  -Wno-deprecated-gpu-targets \
  cuda_vk_interop.cu -o cuda_vk_interop.o

echo "=== Building VkLayer_CudaRT.so ==="
g++ -shared -fPIC -fvisibility=hidden -O2 -std=c++20 -Wall \
  -I/usr/local/cuda/include \
  VkLayer_CudaRT.cpp cuda_bvh_backend.o rasterboost_upscale.o rasterboost_postfx.o rasterboost_framegen.o rt_denoise.o rt_ir_exec.o nrc.o cuda_vk_interop.o \
  -L/usr/local/cuda/lib64 -lcudart \
  -L/usr/lib/x86_64-linux-gnu -lnvinfer \
  -o libVkLayer_CudaRT.so

echo "=== Installing layer manifest ==="
sudo cp VkLayer_CudaRT.json /usr/share/vulkan/implicit_layer.d/
# Update manifest with absolute path
sudo python3 -c "
import json
m = json.load(open('/usr/share/vulkan/implicit_layer.d/VkLayer_CudaRT.json'))
m['layer']['library_path'] = '$DIR/libVkLayer_CudaRT.so'
json.dump(m, open('/usr/share/vulkan/implicit_layer.d/VkLayer_CudaRT.json','w'), indent=2)
"

echo "=== Building RT test app ==="
gcc -O2 -o /tmp/rt_test rt_test.c -lvulkan -lm

echo ""
echo "=== All builds successful ==="
echo "Run:  ENABLE_CUDA_RT_LAYER=1 /tmp/rt_test"
echo "Disable layer: unset ENABLE_CUDA_RT_LAYER"
