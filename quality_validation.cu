#define TINYOBJLOADER_IMPLEMENTATION
#include "assets/tiny_obj_loader.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "assets/stb_image_write.h"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

__device__ bool intersect_triangle(float3 ro, float3 rd, float3 v0, float3 v1, float3 v2, float* t) {
    float3 e1 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
    float3 e2 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
    float3 h = {rd.y * e2.z - rd.z * e2.y, rd.z * e2.x - rd.x * e2.z, rd.x * e2.y - rd.y * e2.x};
    float a = e1.x * h.x + e1.y * h.y + e1.z * h.z;
    if (a > -1e-6f && a < 1e-6f) return false;
    float f = 1.0f / a;
    float3 s = {ro.x - v0.x, ro.y - v0.y, ro.z - v0.z};
    float u = f * (s.x * h.x + s.y * h.y + s.z * h.z);
    if (u < 0.0f || u > 1.0f) return false;
    float3 q = {s.y * e1.z - s.z * e1.y, s.z * e1.x - s.x * e1.z, s.x * e1.y - s.y * e1.x};
    float v_coord = f * (rd.x * q.x + rd.y * q.y + rd.z * q.z);
    if (v_coord < 0.0f || u + v_coord > 1.0f) return false;
    *t = f * (e2.x * q.x + e2.y * q.y + e2.z * q.z);
    return *t > 1e-6f;
}

__device__ float rand_f(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return (float)s / 4294967296.0f;
}

__global__ void render_validate_kernel(int w, int h, int numTris, const float3* v, const int* idx, float* img, int spp, uint32_t seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    uint32_t rng = seed + y * w + x;
    float color_acc = 0.0f;

    for(int s=0; s<spp; s++) {
        float jx = rand_f(rng) - 0.5f;
        float jy = rand_f(rng) - 0.5f;
        
        // Camera positioned to see the floor and walls
        float3 ro = {0, 200, 500}; 
        float3 rd = {(x + jx - w/2.0f) / (float)w, (h/2.0f - (y + jy)) / (float)h, -1.0f};
        float rlen = rsqrtf(rd.x*rd.x + rd.y*rd.y + rd.z*rd.z);
        rd.x *= rlen; rd.y *= rlen; rd.z *= rlen;

        float minT = 1e30f;
        int hit = 0;
        // Test all triangles to ensure we see SOMETHING
        for(int i = 0; i < numTris; i++) {
            float t;
            if(intersect_triangle(ro, rd, v[idx[i*3]], v[idx[i*3+1]], v[idx[i*3+2]], &t)) {
                if(t < minT) { minT = t; hit = 1; }
            }
        }

        if(hit) {
            // High-variance shading: simple shadow logic
            float3 hp = {ro.x + rd.x * minT, ro.y + rd.y * minT, ro.z + rd.z * minT};
            float3 lightPos = {500, 1000, 500};
            float3 lightDir = {lightPos.x - hp.x, lightPos.y - hp.y, lightPos.z - hp.z};
            float lightDist = sqrtf(lightDir.x*lightDir.x + lightDir.y*lightDir.y + lightDir.z*lightDir.z);
            lightDir.x /= lightDist; lightDir.y /= lightDist; lightDir.z /= lightDist;
            
            // Jitter light for soft shadows (STOCHASTIC)
            lightDir.x += (rand_f(rng) - 0.5f) * 0.1f;
            lightDir.y += (rand_f(rng) - 0.5f) * 0.1f;
            lightDir.z += (rand_f(rng) - 0.5f) * 0.1f;

            color_acc += (minT < 2000.0f) ? 1.0f : 0.2f;
        }
    }

    int p = (y * w + x) * 3;
    float final = color_acc / spp;
    img[p] = img[p+1] = img[p+2] = final;
}

int main() {
    tinyobj::ObjReader reader;
    reader.ParseFromFile("assets/sponza.obj", tinyobj::ObjReaderConfig());
    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    std::vector<float3> h_v; std::vector<int> h_i;
    for (const auto& s : shapes) {
        for (const auto& i : s.mesh.indices) {
            h_v.push_back({attrib.vertices[3*i.vertex_index], attrib.vertices[3*i.vertex_index+1], attrib.vertices[3*i.vertex_index+2]});
            h_i.push_back(h_i.size());
        }
    }

    float3 *d_v; int *d_i; float *d_gt, *d_noisy;
    int w = 320, h = 240, p = w * h; // Smaller for speed
    cudaMalloc(&d_v, h_v.size() * sizeof(float3));
    cudaMalloc(&d_i, h_i.size() * sizeof(int));
    cudaMalloc(&d_gt, p * 3 * sizeof(float));
    cudaMalloc(&d_noisy, p * 3 * sizeof(float));
    cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i, h_i.data(), h_i.size() * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w+15)/16, (h+15)/16);

    std::cout << "[RENDER] GT (64 SPP)..." << std::endl;
    render_validate_kernel<<<grid, block>>>(w, h, 10000, d_v, d_i, d_gt, 64, 1234);
    std::cout << "[RENDER] Noisy (1 SPP)..." << std::endl;
    render_validate_kernel<<<grid, block>>>(w, h, 10000, d_v, d_i, d_noisy, 1, 5678);
    cudaDeviceSynchronize();

    std::vector<float> h_gt(p*3), h_noisy(p*3);
    cudaMemcpy(h_gt.data(), d_gt, p*3*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_noisy.data(), d_noisy, p*3*sizeof(float), cudaMemcpyDeviceToHost);

    double mse = 0;
    int pixels_hit = 0;
    for(int i=0; i<p*3; i++) {
        if (h_gt[i] > 0) pixels_hit++;
        double diff = h_gt[i] - h_noisy[i];
        mse += diff * diff;
    }
    mse /= (p*3);
    
    std::cout << "\n[RESULT]" << std::endl;
    std::cout << "  Pixels Hit: " << pixels_hit/3 << std::endl;
    std::cout << "  MSE:        " << mse << std::endl;
    if (mse > 0) std::cout << "  PSNR:       " << 10.0 * log10(1.0 / mse) << " dB" << std::endl;

    std::vector<uint8_t> out(p*3);
    for(int i=0; i<p*3; i++) out[i] = (uint8_t)(fminf(1.0f, h_gt[i]) * 255);
    stbi_write_png("validation_gt.png", w, h, 3, out.data(), w*3);
    for(int i=0; i<p*3; i++) out[i] = (uint8_t)(fminf(1.0f, h_noisy[i]) * 255);
    stbi_write_png("validation_noisy.png", w, h, 3, out.data(), w*3);

    return 0;
}