#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define N_ITEMS (4*1024*1024)
#define BLK 256

// Module 1: Texture fetch throughput
__global__ void bench_texture(cudaTextureObject_t tex, int numFetch, float* __restrict__ out, int n) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        float sum = 0;
        for (int f = 0; f < numFetch; f++) {
            float4 v = tex1Dfetch<float4>(tex, (i*7+f*13) % numFetch);
            sum += v.x + v.y + v.z + v.w;
        }
        out[i] = sum;
    }
}

// Module 2: Constant memory broadcast
__constant__ float4 c_data[1024];
__global__ void bench_const(float* __restrict__ out, int n, int numRead) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        float sum = 0;
        for (int f = 0; f < numRead; f++) {
            float4 v = c_data[f % 1024];
            sum += v.x + v.y + v.z + v.w;
        }
        out[i] = sum;
    }
}

// Module 3: FP32 AABB slab test
__global__ void bench_fp32_aabb(const float* __restrict__ data, float* __restrict__ out, int n, int numTests) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        float ox=data[i*3],oy=data[i*3+1],oz=data[i*3+2];
        float ix=0.5f,iy=-0.3f,iz=0.7f;
        int hits=0;
        for (int t = 0; t < numTests; t++) {
            float bmnx=-1.0f+t*0.01f,bmny=-1.0f,bmnz=-1.0f;
            float bmxx=1.0f+t*0.01f,bmxy=1.0f,bmxz=1.0f;
            float tx1=(bmnx-ox)*ix,tx2=(bmxx-ox)*ix;
            float tmin=fminf(tx1,tx2),tmax=fmaxf(tx1,tx2);
            float ty1=(bmny-oy)*iy,ty2=(bmxy-oy)*iy;
            tmin=fmaxf(tmin,fminf(ty1,ty2));tmax=fminf(tmax,fmaxf(ty1,ty2));
            float tz1=(bmnz-oz)*iz,tz2=(bmxz-oz)*iz;
            tmin=fmaxf(tmin,fminf(tz1,tz2));tmax=fminf(tmax,fmaxf(tz1,tz2));
            if(tmax>=fmaxf(tmin,0.0f)) hits++;
        }
        out[i]=(float)hits;
    }
}

// Module 4: INT32 stack + index ops
__global__ void bench_int32(int* __restrict__ out, int n, int numOps) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        int stack[48]; int sp=0,acc=0;
        stack[sp++]=i;
        for(int t=0;t<numOps;t++){
            if(sp>0){int v=stack[--sp];acc^=v;}
            stack[sp++]=(acc*7+t)&0xFFFF;
            stack[sp++]=((acc>>4)+t*3)&0xFFFF;
            if(sp>40)sp=1;
            acc+=(acc&0xFFF)+((acc>>12)&0xFFF);
        }
        out[i]=acc;
    }
}

// Module 5: SFU (reciprocal, rsqrt)
__global__ void bench_sfu(const float* __restrict__ in, float* __restrict__ out, int n, int numOps) {
    for(int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        float v = in[i]+1.0f;
        for(int t=0;t<numOps;t++){
            v=__frcp_rn(v+0.001f);
            v=__frsqrt_rn(v+0.001f);
            v=fminf(v,100.0f);
        }
        out[i]=v;
    }
}

// Module 6: Shared memory stack
__global__ void bench_smem(float* __restrict__ out, int n, int numOps) {
    __shared__ int smem[BLK][48];
    int tid=threadIdx.x;
    for(int i = blockIdx.x*blockDim.x+tid; i < n; i += gridDim.x*blockDim.x) {
        int sp=0,acc=0;smem[tid][sp++]=i;
        for(int t=0;t<numOps;t++){
            if(sp>0)acc+=smem[tid][--sp];
            smem[tid][sp++]=acc&0xFF;
            smem[tid][sp++]=(acc>>8)&0xFF;
            if(sp>40)sp=1;
        }
        out[i]=(float)acc;
    }
}

// Module 7: Möller-Trumbore triangle intersection
__global__ void bench_moller(const float* __restrict__ triData, float* __restrict__ out, int n, int numTris) {
    for(int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        float ox=0,oy=0,oz=-5,dx=0,dy=0,dz=1,bestT=1e30f;
        for(int t=0;t<numTris;t++){
            float v0x=triData[t*9+0],v0y=triData[t*9+1],v0z=triData[t*9+2];
            float e1x=triData[t*9+3]-v0x,e1y=triData[t*9+4]-v0y,e1z=triData[t*9+5]-v0z;
            float e2x=triData[t*9+6]-v0x,e2y=triData[t*9+7]-v0y,e2z=triData[t*9+8]-v0z;
            float hx=dy*e2z-dz*e2y,hy=dz*e2x-dx*e2z,hz=dx*e2y-dy*e2x;
            float a=e1x*hx+e1y*hy+e1z*hz;
            if(fabsf(a)<1e-8f)continue;
            float f=__frcp_rn(a);
            float sx=ox-v0x,sy=oy-v0y,sz=oz-v0z;
            float u=f*(sx*hx+sy*hy+sz*hz);
            if(u<0||u>1)continue;
            float qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x;
            float v=f*(dx*qx+dy*qy+dz*qz);
            if(v<0||u+v>1)continue;
            float tt=f*(e2x*qx+e2y*qy+e2z*qz);
            if(tt>0.001f&&tt<bestT)bestT=tt;
        }
        out[i]=bestT;
    }
}

// Module 8: Global memory bandwidth (L2-sized)
__global__ void bench_gmem(const float4* __restrict__ data, float* __restrict__ out, int n, int dataSize) {
    for(int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        float sum=0;
        for(int f=0;f<100;f++){
            float4 v=data[(i*7+f*13)%dataSize];
            sum+=v.x+v.y+v.z+v.w;
        }
        out[i]=sum;
    }
}

// ═══ Combined kernel: ALL units at once (simulates real traversal) ═══
__global__ void bench_combined(
    cudaTextureObject_t tex, int texSize,
    const float* __restrict__ triData, int numTris,
    const float* __restrict__ rayData,
    float* __restrict__ out, int n)
{
    __shared__ int smem[BLK][48];
    int tid = threadIdx.x;
    
    for(int i = blockIdx.x*blockDim.x+tid; i < n; i += gridDim.x*blockDim.x) {
        float ox=rayData[i*3],oy=rayData[i*3+1],oz=rayData[i*3+2];
        float ix=0.5f,iy=-0.3f,iz=0.7f;
        float dx=0,dy=0,dz=1;
        float hitT=1e30f;
        int sp=0; smem[tid][sp++]=0;
        int nodeTests=0,triTests=0;
        
        while(sp>0 && nodeTests<50) {
            int ni=smem[tid][--sp];
            nodeTests++;
            
            // TEX: fetch node (through texture cache)
            float4 nlo = tex1Dfetch<float4>(tex, (ni*2) % texSize);
            float4 nhi = tex1Dfetch<float4>(tex, (ni*2+1) % texSize);
            
            // Also try CONST for top levels
            float4 cval = (ni < 1024) ? c_data[ni] : nlo;
            
            // FP32+SFU: AABB slab test
            float bmnx=nlo.x,bmny=nlo.y,bmnz=nlo.z;
            float bmxx=nhi.x,bmxy=nhi.y,bmxz=nhi.z;
            float tx1=(bmnx-ox)*ix,tx2=(bmxx-ox)*ix;
            float tmin=fminf(tx1,tx2),tmax=fmaxf(tx1,tx2);
            float ty1=(bmny-oy)*iy,ty2=(bmxy-oy)*iy;
            tmin=fmaxf(tmin,fminf(ty1,ty2));tmax=fminf(tmax,fmaxf(ty1,ty2));
            float tz1=(bmnz-oz)*iz,tz2=(bmxz-oz)*iz;
            tmin=fmaxf(tmin,fminf(tz1,tz2));tmax=fminf(tmax,fmaxf(tz1,tz2));
            
            if(tmax<fmaxf(tmin,0.0f)) continue;
            
            // INT32: push children (runs parallel to FP on Volta!)
            int left = ni*2+1, right = ni*2+2;
            if(left < 500 && sp < 40) smem[tid][sp++] = left;
            if(right < 500 && sp < 40) smem[tid][sp++] = right;
            
            // Triangle test every 6th node (simulates leaf)
            if(nodeTests % 6 == 0 && triTests < 8) {
                for(int t=0;t<numTris && t<4;t++){
                    triTests++;
                    float v0x=triData[t*9],v0y=triData[t*9+1],v0z=triData[t*9+2];
                    float e1x=triData[t*9+3]-v0x,e1y=triData[t*9+4]-v0y,e1z=triData[t*9+5]-v0z;
                    float e2x=triData[t*9+6]-v0x,e2y=triData[t*9+7]-v0y,e2z=triData[t*9+8]-v0z;
                    float hx=dy*e2z-dz*e2y,hy=dz*e2x-dx*e2z,hz=dx*e2y-dy*e2x;
                    float a=e1x*hx+e1y*hy+e1z*hz;
                    if(fabsf(a)<1e-8f)continue;
                    float f=__frcp_rn(a);
                    float sx=ox-v0x,sy=oy-v0y,sz=oz-v0z;
                    float u=f*(sx*hx+sy*hy+sz*hz);
                    if(u<0||u>1)continue;
                    float qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x;
                    float v=f*(dx*qx+dy*qy+dz*qz);
                    if(v<0||u+v>1)continue;
                    float tt=f*(e2x*qx+e2y*qy+e2z*qz);
                    if(tt>0.001f&&tt<hitT)hitT=tt;
                }
            }
        }
        out[i]=hitT+(float)nodeTests;
    }
}

int main(){
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  V100 Per-Module Latency Profiler — 4M Work Items                   ║\n");
    printf("║  Calibrated to real RT workload: ~50 node/ray, ~8 tri/ray           ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");
    
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    int nb = p.multiProcessorCount * 4; // 320 blocks
    int n = N_ITEMS;
    
    float *d_fout,*d_fin,*d_triData;
    int *d_iout;
    cudaMalloc(&d_fout,n*4); cudaMalloc(&d_fin,n*3*4); cudaMalloc(&d_iout,n*4);
    
    float* h_tmp=(float*)malloc(n*3*4);
    srand(42); for(int i=0;i<n*3;i++) h_tmp[i]=((float)rand()/RAND_MAX)*20-10;
    cudaMemcpy(d_fin,h_tmp,n*3*4,cudaMemcpyHostToDevice);
    
    int nTri=64;
    float* h_tri=(float*)malloc(nTri*9*4);
    for(int i=0;i<nTri;i++){
        float cx=((float)rand()/RAND_MAX)*4-2,cy=((float)rand()/RAND_MAX)*4-2,cz=((float)rand()/RAND_MAX)*4-2;
        h_tri[i*9+0]=cx-0.5f;h_tri[i*9+1]=cy-0.5f;h_tri[i*9+2]=cz;
        h_tri[i*9+3]=cx+0.5f;h_tri[i*9+4]=cy;h_tri[i*9+5]=cz+0.5f;
        h_tri[i*9+6]=cx;h_tri[i*9+7]=cy+0.5f;h_tri[i*9+8]=cz-0.5f;
    }
    cudaMalloc(&d_triData,nTri*9*4);
    cudaMemcpy(d_triData,h_tri,nTri*9*4,cudaMemcpyHostToDevice);
    
    int texSz=100000;
    float4* d_texData; cudaMalloc(&d_texData,texSz*sizeof(float4));
    float4* h_tex=(float4*)malloc(texSz*sizeof(float4));
    for(int i=0;i<texSz;i++) h_tex[i]={1.0f*i,2.0f*i,3.0f*i,4.0f*i};
    cudaMemcpy(d_texData,h_tex,texSz*sizeof(float4),cudaMemcpyHostToDevice);
    
    cudaResourceDesc rd; memset(&rd,0,sizeof(rd));
    rd.resType=cudaResourceTypeLinear;
    rd.res.linear.devPtr=d_texData;
    rd.res.linear.desc=cudaCreateChannelDesc<float4>();
    rd.res.linear.sizeInBytes=texSz*sizeof(float4);
    cudaTextureDesc td; memset(&td,0,sizeof(td));
    td.readMode=cudaReadModeElementType;
    cudaTextureObject_t texObj=0;
    cudaCreateTextureObject(&texObj,&rd,&td,NULL);
    
    cudaMemcpyToSymbol(c_data,h_tex,1024*sizeof(float4));
    
    float4* d_l2; int l2Sz=6*1024*1024/16;
    cudaMalloc(&d_l2,l2Sz*sizeof(float4));
    
    printf("  Hardware: %s | %d SMs | %dKB L2 | %.0f GB/s\n\n",
        p.name,p.multiProcessorCount,p.l2CacheSize/1024,
        2.0*p.memoryClockRate*(p.memoryBusWidth/8)/1e6);
    
    // Run benchmarks
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    
    struct Result { const char* name; float ms; double mrays; const char* hw; };
    Result results[10]; int ri=0;
    
    auto bench = [&](const char* name, auto fn, const char* hw) {
        fn(); cudaDeviceSynchronize();
        cudaError_t err=cudaGetLastError();
        if(err!=cudaSuccess){printf("  %s: ERROR %s\n",name,cudaGetErrorString(err));return;}
        float total=0;
        for(int r=0;r<10;r++){
            cudaEventRecord(t0); fn();
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms,t0,t1); total+=ms;
        }
        float avg=total/10; double mr=(double)n/(avg/1000.0)/1e6;
        results[ri++]={name,avg,mr,hw};
    };
    
    bench("Tex Fetch(50/ray)", [&](){bench_texture<<<nb,BLK>>>(texObj,50,d_fout,n);}, "320 TexUnits+TexL1$");
    bench("Const Mem(10/ray)", [&](){bench_const<<<nb,BLK>>>(d_fout,n,10);}, "64KB ConstMem bcast");
    bench("FP32 AABB(50/ray)", [&](){bench_fp32_aabb<<<nb,BLK>>>(d_fin,d_fout,n,50);}, "5120 FP32+SFU");
    bench("INT32 Stk(50/ray)", [&](){bench_int32<<<nb,BLK>>>(d_iout,n,50);}, "5120 INT32(∥FP32!)");
    bench("SFU rcp  (20/ray)", [&](){bench_sfu<<<nb,BLK>>>(d_fin,d_fout,n,20);}, "1280 SFU units");
    int smNb=p.multiProcessorCount*2;
    bench("SmemStack(50/ray)", [&](){bench_smem<<<smNb,BLK>>>(d_fout,n,50);}, "96KB smem/SM");
    bench("FP32 Möll(8/ray)",  [&](){bench_moller<<<nb,BLK>>>(d_triData,d_fout,n,8);}, "5120 FP32+SFU rcp");
    bench("Gmem/L2 (100/ray)", [&](){bench_gmem<<<nb,BLK>>>((float4*)d_fin,d_fout,n,n*3/4);}, "6MB L2+900GB/s HBM");
    bench("Combined (all)",    [&](){bench_combined<<<smNb,BLK>>>(texObj,texSz,d_triData,nTri,d_fin,d_fout,n);}, "ALL units together");
    
    // Print results table
    printf("  ┌─────────────────────┬──────────┬──────────────┬──────────────────────────┐\n");
    printf("  │ Module              │ Time(ms) │ MRays/s      │ Hardware Used             │\n");
    printf("  ├─────────────────────┼──────────┼──────────────┼──────────────────────────┤\n");
    
    float maxMs=0;
    const char* bottleneck="";
    for(int i=0;i<ri;i++){
        printf("  │ %-19s │ %7.2f  │ %9.1f    │ %-24s │\n",
            results[i].name,results[i].ms,results[i].mrays,results[i].hw);
        if(i<ri-1 && results[i].ms>maxMs){maxMs=results[i].ms;bottleneck=results[i].name;}
    }
    printf("  └─────────────────────┴──────────┴──────────────┴──────────────────────────┘\n\n");
    
    // Analysis
    printf("  ═══════════════════════════════════════════════════════════════\n");
    printf("  ANALYSIS — Volta Dual-Issue Pipeline Model:\n");
    printf("  ═══════════════════════════════════════════════════════════════\n\n");
    
    float texMs=results[0].ms, constMs=results[1].ms, fp32Ms=results[2].ms;
    float int32Ms=results[3].ms, sfuMs=results[4].ms, smemMs=results[5].ms;
    float mollerMs=results[6].ms, gmemMs=results[7].ms;
    float combinedMs = ri>8 ? results[8].ms : 0;
    
    printf("  Concurrent execution on Volta (same cycle):\n");
    printf("    FP32 pipe:  AABB=%.2fms  Möller=%.2fms  → total FP32=%.2fms\n",fp32Ms,mollerMs,fp32Ms+mollerMs);
    printf("    INT32 pipe: Stack=%.2fms (runs ∥ FP32 = FREE if < FP32)\n",int32Ms);
    printf("    Tex pipe:   Fetch=%.2fms (sep units + sep cache = overlaps ALU)\n",texMs);
    printf("    SFU pipe:   rcp  =%.2fms (sep units = overlaps FP32/INT32)\n",sfuMs);
    printf("    Const:      Read =%.2fms (broadcast, overlaps everything)\n",constMs);
    printf("    Smem:       Stack=%.2fms (same cycle as global loads)\n\n",smemMs);
    
    float theoretical = fmaxf(fp32Ms + mollerMs, fmaxf(texMs, smemMs));
    float theoMrays = (double)n/(theoretical/1000.0)/1e6;
    
    printf("  Projected combined (pipeline model):\n");
    printf("    Bottleneck = max(FP32_total, Tex, Smem) = %.2f ms\n", theoretical);
    printf("    Projected MRays/s = %.1f\n", theoMrays);
    printf("    Actual combined   = %.2f ms = %.1f MRays/s\n", combinedMs, ri>8?results[8].mrays:0);
    printf("    Pipeline efficiency = %.0f%%\n\n", combinedMs>0 ? (theoretical/combinedMs)*100 : 0);
    
    printf("  BOTTLENECK: %s (%.2f ms)\n", bottleneck, maxMs);
    printf("  INT32 overhead: %.0f%% hidden (%.2f ms ∥ %.2f ms FP32)\n",
        int32Ms<fp32Ms ? 100.0f : (1.0f-int32Ms/fp32Ms)*100, int32Ms, fp32Ms);
    
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    cudaDestroyTextureObject(texObj);
    cudaFree(d_texData);cudaFree(d_fout);cudaFree(d_fin);cudaFree(d_iout);
    cudaFree(d_triData);cudaFree(d_l2);
    free(h_tmp);free(h_tri);free(h_tex);
    
    return 0;
}
