NVCC = /usr/local/cuda-12.9/bin/nvcc
NVFLAGS = -O3 -arch=sm_70 -Wno-deprecated-gpu-targets --use_fast_math -Xcompiler "-O3 -march=native"
GCC = gcc
GCCFLAGS = -O2 -w
VKLIBS = -ldl -lm -lvulkan

# CUDA RT engines
CUDA_BINS = cuda_rt_v7 cuda_rt_conference cuda_rt_latency

# Vulkan benchmarks
VK_BINS = vk_blas_bench

.PHONY: all clean cuda vulkan

all: cuda vulkan

cuda: $(CUDA_BINS)

vulkan: $(VK_BINS)

cuda_rt_v7: cuda_rt_v7.cu
	$(NVCC) $(NVFLAGS) --extended-lambda -o $@ $<

cuda_rt_conference: cuda_rt_conference.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

cuda_rt_latency: cuda_rt_latency.cu
	$(NVCC) $(NVFLAGS) --extended-lambda -o $@ $<

vk_blas_bench: vk_blas_bench.c
	$(GCC) $(GCCFLAGS) -o $@ $< $(VKLIBS)

clean:
	rm -f $(CUDA_BINS) $(VK_BINS)

bench: all
	@echo ""
	@echo "═══ CUDA RT v7 (best engine) ═══"
	./cuda_rt_v7
	@echo ""
	@echo "═══ CUDA RT Conference (scene quality test) ═══"
	./cuda_rt_conference
	@echo ""
	@echo "═══ Vulkan BLAS Build Timing ═══"
	./vk_blas_bench
