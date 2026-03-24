#!/bin/bash
# Build the CUDA RT Vulkan Layer and test app
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== Building CUDA BVH backend ==="
/usr/local/cuda/bin/nvcc -c -O3 -arch=sm_70 --compiler-options=-fPIC \
  -Wno-deprecated-gpu-targets \
  cuda_bvh_backend.cu -o cuda_bvh_backend.o

echo "=== Building VkLayer_CudaRT.so ==="
g++ -shared -fPIC -fvisibility=hidden -O2 -std=c++17 -Wall \
  -I/usr/local/cuda/include \
  VkLayer_CudaRT.cpp cuda_bvh_backend.o \
  -L/usr/local/cuda/lib64 -lcudart \
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
