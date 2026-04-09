#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "cuda_bvh_backend.h"

void genTerrain(int gridN, std::vector<CudaTri>& tris) {
    int nv = (gridN+1)*(gridN+1);
    std::vector<float> vx(nv), vy(nv), vz(nv);
    for (int y=0; y<=gridN; y++) for (int x=0; x<=gridN; x++) {
        int i = y*(gridN+1)+x;
        float fx = (float)x/gridN*2.f-1.f, fy = (float)y/gridN*2.f-1.f;
        vx[i]=fx; vy[i]=fy; vz[i]= 0.2f*sinf(fx*6.28f)*cosf(fy*6.28f);
    }
    tris.resize(gridN*gridN*2);
    for (int y=0; y<gridN; y++) for (int x=0; x<gridN; x++) {
        int cell = y*gridN+x;
        int i0=y*(gridN+1)+x, i1=i0+1, i2=i0+(gridN+1), i3=i2+1;
        CudaTri& t0 = tris[cell*2];
        t0.v0[0]=vx[i0];t0.v0[1]=vy[i0];t0.v0[2]=vz[i0];
        t0.v1[0]=vx[i1];t0.v1[1]=vy[i1];t0.v1[2]=vz[i1];
        t0.v2[0]=vx[i2];t0.v2[1]=vy[i2];t0.v2[2]=vz[i2];
        CudaTri& t1 = tris[cell*2+1];
        t1.v0[0]=vx[i1];t1.v0[1]=vy[i1];t1.v0[2]=vz[i1];
        t1.v1[0]=vx[i3];t1.v1[1]=vy[i3];t1.v1[2]=vz[i3];
        t1.v2[0]=vx[i2];t1.v2[1]=vy[i2];t1.v2[2]=vz[i2];
    }
}

int main() {
    int grids[] = {100, 200, 316, 447, 632, 894};
    int sides[] = {512, 1024, 2048};
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  BVH4 Traversal Benchmark — Tesla V100-PCIE-16GB          ║\n");
    printf("╠═════════╦════════╦═══════════╦══════════╦═════════╦════════╣\n");
    printf("║  Grid   ║ Tris   ║ Resolution║  MR/s    ║ ns/ray  ║ ms/frm║\n");
    printf("╠═════════╬════════╬═══════════╬══════════╬═════════╬════════╣\n");
    
    for (int gi = 0; gi < 6; gi++) {
        int gridN = grids[gi];
        std::vector<CudaTri> tris;
        genTerrain(gridN, tris);
        int numTris = (int)tris.size();
        
        CudaBVH_t bvh = cudaBVH_build(tris.data(), numTris);
        if (!bvh) { printf("Build failed for grid %d\n", gridN); continue; }
        
        // Warmup
        cudaBVH_tracePrimary(bvh, 512, 0, 0, 3.36f, nullptr);
        
        for (int si = 0; si < 3; si++) {
            int side = sides[si];
            
            // Average 5 runs
            float totalMRs = 0;
            for (int r = 0; r < 5; r++) {
                totalMRs += cudaBVH_tracePrimary(bvh, side, 0, 0, 3.36f, nullptr);
            }
            float mrs = totalMRs / 5.f;
            float nsPerRay = 1e6f / mrs;
            float msPerFrame = (float)(side*side) / (mrs * 1e3f);
            
            printf("║ %3dx%3d ║ %5dK ║ %4dx%4d ║ %7.0f  ║ %6.1f  ║ %5.2f ║\n",
                   gridN, gridN, numTris/1000, side, side, mrs, nsPerRay, msPerFrame);
        }
        cudaBVH_destroy(bvh);
    }
    printf("╚═════════╩════════╩═══════════╩══════════╩═════════╩════════╝\n");
    
    // 1080p RGBA trace+shade
    printf("\n[RGBA trace+shade @ 1080p]\n");
    std::vector<CudaTri> tris;
    genTerrain(400, tris);
    CudaBVH_t bvh = cudaBVH_build(tris.data(), (int)tris.size());
    uint32_t* rgba = (uint32_t*)malloc(1920*1080*4);
    
    cudaBVH_traceToRGBA(bvh, 1920, 1080, 0, 0, 3.36f, rgba);  // warmup
    
    cudaEvent_t t0,t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i=0; i<10; i++) cudaBVH_traceToRGBA(bvh, 1920, 1080, 0, 0, 3.36f, rgba);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    float avgMs = ms/10.f;
    float numRays = 1920.f*1080.f;
    float mrs = numRays/(avgMs*1000.f);
    printf("  1920x1080 @ %dK tris: %.2f ms/frame → %.0f MR/s → %.1f ns/ray → %.0f FPS\n",
           (int)tris.size()/1000, avgMs, mrs, 1e6f/mrs, 1000.f/avgMs);
    
    free(rgba);
    cudaBVH_destroy(bvh);
    return 0;
}
