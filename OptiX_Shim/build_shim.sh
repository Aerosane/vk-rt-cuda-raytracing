#!/bin/bash
set -e

echo "=== Building CuRT-OptiX Shim for V100 2026 ==="

# Compile CUDA kernel
/usr/local/cuda-12.9/bin/nvcc -c -O3 -arch=sm_70 --compiler-options=-fPIC \
    neural_recon_kernel.cu -o neural_recon_kernel.o

# Compile C++ shim
g++ -c -O3 -fPIC -std=c++17 \
    -I/usr/local/cuda-12.9/include \
    -I/workspaces/codespace/VK_RT/layer \
    optix_shim.cpp -o optix_shim.o

# Link into libnvoptix.so.1
# Note: Linking with cuda_bvh_backend.o from the layer directory
g++ -shared -fPIC \
    optix_shim.o neural_recon_kernel.o /workspaces/codespace/VK_RT/layer/cuda_bvh_backend.o \
    -L/usr/local/cuda-12.9/lib64 -lcudart -ldl \
    -o libnvoptix.so.1

echo "=== Build Successful: libnvoptix.so.1 created ==="
ls -l libnvoptix.so.1
