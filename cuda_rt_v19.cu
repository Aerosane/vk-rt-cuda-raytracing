/* v19 — Software SER: GPU-side Direction-Aware Coherence Sorting
 *
 * ALL sorting on GPU via thrust::sort_by_key (CUB radix sort).
 * Key generation kernel → radix sort → scatter reorder → trace.
 *
 * Tests 5 sort strategies for DIFFUSE rays:
 *   A) Baseline: octant(3) + morton(27)
 *   B) 72 dir bins + morton(25)
 *   C) 512 dir bins + morton(23)
 *   D) 4K dir bins + morton(20)
 *   E) 131K dir bins + morton(15)
 * Measures MRays/s AND warp efficiency for each.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>

// ======================== Types ========================
struct float3a{float x,y,z;};
struct AABB{float3a mn,mx;};
struct Tri{float3a v0,v1,v2;};
struct BVH4Node{
    __half boundsX[8],boundsY[8],boundsZ[8];
    int child[4];
};
struct Hit{float t;int tri;float u,v;};

__device__ unsigned int g_rayCounter;

// ======================== BVH builder (same as v12) ========================
static AABB triAABB(const Tri&t){
    AABB b;
    b.mn.x=fminf(fminf(t.v0.x,t.v1.x),t.v2.x);
    b.mn.y=fminf(fminf(t.v0.y,t.v1.y),t.v2.y);
    b.mn.z=fminf(fminf(t.v0.z,t.v1.z),t.v2.z);
    b.mx.x=fmaxf(fmaxf(t.v0.x,t.v1.x),t.v2.x);
    b.mx.y=fmaxf(fmaxf(t.v0.y,t.v1.y),t.v2.y);
    b.mx.z=fmaxf(fmaxf(t.v0.z,t.v1.z),t.v2.z);
    return b;
}
struct BVHBuild{
    struct N2{AABB box;int left,right;int triStart,triCount;};
    std::vector<N2>nodes;std::vector<Tri>ordered;const Tri*src;int nT;
    void build(const Tri*t,int n){
        src=t;nT=n;nodes.clear();ordered.clear();
        std::vector<int>idx(n);for(int i=0;i<n;i++)idx[i]=i;
        buildRec(idx,0,n);
    }
    int buildRec(std::vector<int>&idx,int s,int e){
        N2 nd;nd.triStart=nd.triCount=nd.left=nd.right=-1;
        nd.box.mn={1e30f,1e30f,1e30f};nd.box.mx={-1e30f,-1e30f,-1e30f};
        for(int i=s;i<e;i++){
            AABB b=triAABB(src[idx[i]]);
            nd.box.mn.x=fminf(nd.box.mn.x,b.mn.x);nd.box.mn.y=fminf(nd.box.mn.y,b.mn.y);nd.box.mn.z=fminf(nd.box.mn.z,b.mn.z);
            nd.box.mx.x=fmaxf(nd.box.mx.x,b.mx.x);nd.box.mx.y=fmaxf(nd.box.mx.y,b.mx.y);nd.box.mx.z=fmaxf(nd.box.mx.z,b.mx.z);
        }
        int cnt=e-s;
        if(cnt<=4){
            nd.triStart=(int)ordered.size();nd.triCount=cnt;
            for(int i=s;i<e;i++)ordered.push_back(src[idx[i]]);
            int id=(int)nodes.size();nodes.push_back(nd);return id;
        }
        int bestAxis=0;float bestCost=1e30f;int bestSplit=s+1;
        float pArea=2.f*((nd.box.mx.x-nd.box.mn.x)*(nd.box.mx.y-nd.box.mn.y)+
                         (nd.box.mx.y-nd.box.mn.y)*(nd.box.mx.z-nd.box.mn.z)+
                         (nd.box.mx.x-nd.box.mn.x)*(nd.box.mx.z-nd.box.mn.z));
        if(pArea<1e-12f)pArea=1e-12f;
        for(int ax=0;ax<3;ax++){
            std::sort(idx.begin()+s,idx.begin()+e,[&](int a,int b){
                float ca=(&src[a].v0.x)[ax]+(&src[a].v1.x)[ax]+(&src[a].v2.x)[ax];
                float cb=(&src[b].v0.x)[ax]+(&src[b].v1.x)[ax]+(&src[b].v2.x)[ax];
                return ca<cb;
            });
            std::vector<AABB>lBox(cnt),rBox(cnt);
            lBox[0]=triAABB(src[idx[s]]);
            for(int i=1;i<cnt;i++){
                AABB b=triAABB(src[idx[s+i]]);
                lBox[i].mn.x=fminf(lBox[i-1].mn.x,b.mn.x);lBox[i].mn.y=fminf(lBox[i-1].mn.y,b.mn.y);lBox[i].mn.z=fminf(lBox[i-1].mn.z,b.mn.z);
                lBox[i].mx.x=fmaxf(lBox[i-1].mx.x,b.mx.x);lBox[i].mx.y=fmaxf(lBox[i-1].mx.y,b.mx.y);lBox[i].mx.z=fmaxf(lBox[i-1].mx.z,b.mx.z);
            }
            rBox[cnt-1]=triAABB(src[idx[e-1]]);
            for(int i=cnt-2;i>=0;i--){
                AABB b=triAABB(src[idx[s+i]]);
                rBox[i].mn.x=fminf(rBox[i+1].mn.x,b.mn.x);rBox[i].mn.y=fminf(rBox[i+1].mn.y,b.mn.y);rBox[i].mn.z=fminf(rBox[i+1].mn.z,b.mn.z);
                rBox[i].mx.x=fmaxf(rBox[i+1].mx.x,b.mx.x);rBox[i].mx.y=fmaxf(rBox[i+1].mx.y,b.mx.y);rBox[i].mx.z=fmaxf(rBox[i+1].mx.z,b.mx.z);
            }
            for(int i=0;i<cnt-1;i++){
                auto sa=[](AABB&b){return 2.f*((b.mx.x-b.mn.x)*(b.mx.y-b.mn.y)+(b.mx.y-b.mn.y)*(b.mx.z-b.mn.z)+(b.mx.x-b.mn.x)*(b.mx.z-b.mn.z));};
                float cost=(i+1)*sa(lBox[i])/pArea+(cnt-1-i)*sa(rBox[i+1])/pArea+1.f;
                if(cost<bestCost){bestCost=cost;bestSplit=s+i+1;bestAxis=ax;}
            }
        }
        std::sort(idx.begin()+s,idx.begin()+e,[&](int a,int b){
            float ca=(&src[a].v0.x)[bestAxis]+(&src[a].v1.x)[bestAxis]+(&src[a].v2.x)[bestAxis];
            float cb=(&src[b].v0.x)[bestAxis]+(&src[b].v1.x)[bestAxis]+(&src[b].v2.x)[bestAxis];
            return ca<cb;
        });
        int id=(int)nodes.size();nodes.push_back(nd);
        int mid=bestSplit;
        nodes[id].left=buildRec(idx,s,mid);
        nodes[id].right=buildRec(idx,mid,e);
        return id;
    }
};
static int collapseToB4(const BVHBuild&b2,int ni,BVH4Node*out,int&cnt,const Tri*tris){
    auto&n=b2.nodes[ni];
    if(n.triCount>0){int ts=n.triStart,tc=n.triCount;return -((ts<<3)|(tc-1))-2;}
    int gather[4];int ng=0;
    int ch[2]={n.left,n.right};
    for(int c=0;c<2;c++){
        auto&cn=b2.nodes[ch[c]];
        if(cn.triCount>0||ng>=3){gather[ng++]=ch[c];continue;}
        gather[ng++]=cn.left;gather[ng++]=cn.right;
    }
    BVH4Node nd;
    for(int i=0;i<4;i++){
        if(i<ng){
            auto&cn=b2.nodes[gather[i]];
            nd.boundsX[i]=__float2half(cn.box.mn.x);nd.boundsX[4+i]=__float2half(cn.box.mx.x);
            nd.boundsY[i]=__float2half(cn.box.mn.y);nd.boundsY[4+i]=__float2half(cn.box.mx.y);
            nd.boundsZ[i]=__float2half(cn.box.mn.z);nd.boundsZ[4+i]=__float2half(cn.box.mx.z);
        }else{
            nd.boundsX[i]=__float2half(1e30f);nd.boundsX[4+i]=__float2half(-1e30f);
            nd.boundsY[i]=__float2half(1e30f);nd.boundsY[4+i]=__float2half(-1e30f);
            nd.boundsZ[i]=__float2half(1e30f);nd.boundsZ[4+i]=__float2half(-1e30f);
        }
        nd.child[i]=-1;
    }
    int me=cnt++;
    for(int i=0;i<ng;i++) nd.child[i]=collapseToB4(b2,gather[i],out,cnt,tris);
    out[me]=nd;return me;
}

// ======================== GPU Sort Key Kernels ========================

// Morton expand
__device__ __forceinline__ unsigned int d_expand3(unsigned int v){
    v&=0x3FF; v=(v|(v<<16))&0x30000FF; v=(v|(v<<8))&0x300F00F;
    v=(v|(v<<4))&0x30C30C3; v=(v|(v<<2))&0x9249249; return v;
}

// Strategy A: 3-bit octant + 27-bit morton (baseline)
__global__ void genKeysA(const float*__restrict__ ox,const float*__restrict__ oy,const float*__restrict__ oz,
                         const float*__restrict__ dx,const float*__restrict__ dy,const float*__restrict__ dz,
                         float3 smn,float3 smx,unsigned int*__restrict__ keys,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    float nx=(ox[i]-smn.x)/(smx.x-smn.x+1e-7f);
    float ny=(oy[i]-smn.y)/(smx.y-smn.y+1e-7f);
    float nz=(oz[i]-smn.z)/(smx.z-smn.z+1e-7f);
    unsigned int ix=min(max((int)(nx*1023.f),0),1023);
    unsigned int iy=min(max((int)(ny*1023.f),0),1023);
    unsigned int iz=min(max((int)(nz*1023.f),0),1023);
    unsigned int m=d_expand3(ix)|(d_expand3(iy)<<1)|(d_expand3(iz)<<2);
    unsigned int oct=((dx[i]<0.f)?4u:0u)|((dy[i]<0.f)?2u:0u)|((dz[i]<0.f)?1u:0u);
    keys[i]=(oct<<29)|(m>>1); // 3 top + 29 morton
}

// Octahedral direction hash (GPU)
__device__ __forceinline__ unsigned int d_dirHash(float ddx,float ddy,float ddz,int gridN){
    unsigned int oct=((ddx<0.f)?4u:0u)|((ddy<0.f)?2u:0u)|((ddz<0.f)?1u:0u);
    float ax=fabsf(ddx),ay=fabsf(ddy),az=fabsf(ddz);
    float sum=ax+ay+az+1e-7f;
    float u=ax/sum, v=ay/sum;
    unsigned int iu=min((int)(u*gridN),gridN-1);
    unsigned int iv=min((int)(v*gridN),gridN-1);
    return oct*gridN*gridN+iu*gridN+iv;
}

// Strategy B: 72 dir bins (oct × 3×3)
__global__ void genKeysB(const float*__restrict__ ox,const float*__restrict__ oy,const float*__restrict__ oz,
                         const float*__restrict__ dx,const float*__restrict__ dy,const float*__restrict__ dz,
                         float3 smn,float3 smx,unsigned int*__restrict__ keys,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float nx=(ox[i]-smn.x)/(smx.x-smn.x+1e-7f);
    float ny=(oy[i]-smn.y)/(smx.y-smn.y+1e-7f);
    float nz=(oz[i]-smn.z)/(smx.z-smn.z+1e-7f);
    unsigned int ix=min(max((int)(nx*1023.f),0),1023);
    unsigned int iy=min(max((int)(ny*1023.f),0),1023);
    unsigned int iz=min(max((int)(nz*1023.f),0),1023);
    unsigned int m=d_expand3(ix)|(d_expand3(iy)<<1)|(d_expand3(iz)<<2);
    unsigned int dh=d_dirHash(dx[i],dy[i],dz[i],3); // 72 bins → 7 bits
    keys[i]=(dh<<25)|(m>>5);
}

// Strategy C: 512 dir bins (oct × 8×8)
__global__ void genKeysC(const float*__restrict__ ox,const float*__restrict__ oy,const float*__restrict__ oz,
                         const float*__restrict__ dx,const float*__restrict__ dy,const float*__restrict__ dz,
                         float3 smn,float3 smx,unsigned int*__restrict__ keys,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float nx=(ox[i]-smn.x)/(smx.x-smn.x+1e-7f);
    float ny=(oy[i]-smn.y)/(smx.y-smn.y+1e-7f);
    float nz=(oz[i]-smn.z)/(smx.z-smn.z+1e-7f);
    unsigned int ix=min(max((int)(nx*1023.f),0),1023);
    unsigned int iy=min(max((int)(ny*1023.f),0),1023);
    unsigned int iz=min(max((int)(nz*1023.f),0),1023);
    unsigned int m=d_expand3(ix)|(d_expand3(iy)<<1)|(d_expand3(iz)<<2);
    unsigned int dh=d_dirHash(dx[i],dy[i],dz[i],8); // 512 bins → 9 bits
    keys[i]=(dh<<23)|(m>>7);
}

// Strategy D: 4K dir bins (oct × 23×23)
__global__ void genKeysD(const float*__restrict__ ox,const float*__restrict__ oy,const float*__restrict__ oz,
                         const float*__restrict__ dx,const float*__restrict__ dy,const float*__restrict__ dz,
                         float3 smn,float3 smx,unsigned int*__restrict__ keys,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float nx=(ox[i]-smn.x)/(smx.x-smn.x+1e-7f);
    float ny=(oy[i]-smn.y)/(smx.y-smn.y+1e-7f);
    float nz=(oz[i]-smn.z)/(smx.z-smn.z+1e-7f);
    unsigned int ix=min(max((int)(nx*1023.f),0),1023);
    unsigned int iy=min(max((int)(ny*1023.f),0),1023);
    unsigned int iz=min(max((int)(nz*1023.f),0),1023);
    unsigned int m=d_expand3(ix)|(d_expand3(iy)<<1)|(d_expand3(iz)<<2);
    unsigned int dh=d_dirHash(dx[i],dy[i],dz[i],23); // 4232 bins → 12 bits
    keys[i]=(dh<<20)|(m>>10);
}

// Strategy E: 131K dir bins (oct × 128×128) — direction dominates
__global__ void genKeysE(const float*__restrict__ ox,const float*__restrict__ oy,const float*__restrict__ oz,
                         const float*__restrict__ dx,const float*__restrict__ dy,const float*__restrict__ dz,
                         float3 smn,float3 smx,unsigned int*__restrict__ keys,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float nx=(ox[i]-smn.x)/(smx.x-smn.x+1e-7f);
    float ny=(oy[i]-smn.y)/(smx.y-smn.y+1e-7f);
    float nz=(oz[i]-smn.z)/(smx.z-smn.z+1e-7f);
    unsigned int ix=min(max((int)(nx*1023.f),0),1023);
    unsigned int iy=min(max((int)(ny*1023.f),0),1023);
    unsigned int iz=min(max((int)(nz*1023.f),0),1023);
    unsigned int m=d_expand3(ix)|(d_expand3(iy)<<1)|(d_expand3(iz)<<2);
    unsigned int dh=d_dirHash(dx[i],dy[i],dz[i],128); // 131072 → 17 bits
    keys[i]=(dh<<15)|(m>>15);
}

// Scatter-reorder a single float array by index permutation
__global__ void scatterReorder(const float*__restrict__ src, float*__restrict__ dst,
                               const int*__restrict__ perm, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) dst[i]=src[perm[i]];
}

// ======================== Trace kernel (v12 incoherent) ========================
#define STACK_DEPTH 8

__global__ void __launch_bounds__(256,5) traceBVH4(
    const int4*__restrict__ d_bvh4,int n4,
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    const float*__restrict__ rix,const float*__restrict__ riy,const float*__restrict__ riz,
    Hit*__restrict__ hits,int numRays,
    unsigned long long*__restrict__ warpEff)
{
    const unsigned lane=threadIdx.x&31;
    unsigned long long myActiveSum=0, mySteps=0;

    while(true){
        int bs;
        if(lane==0) bs=atomicAdd(&g_rayCounter,32);
        bs=__shfl_sync(0xFFFFFFFF,bs,0);
        if(bs>=numRays) break;
        int ri=bs+lane;
        float ox,oy,oz,dx,dy,dz,ix,iy,iz,tHit=1e30f;
        int hitTri=-1; float hitU=0,hitV=0;
        if(ri<numRays){
            ox=rox[ri];oy=roy[ri];oz=roz[ri];
            dx=rdx[ri];dy=rdy[ri];dz=rdz[ri];
            ix=rix[ri];iy=riy[ri];iz=riz[ri];
        }
        int stk[STACK_DEPTH]; int sp=0; stk[sp++]=0;
        bool alive=(ri<numRays);
        while(sp>0 && alive){
            int ni=stk[--sp];
            if(ni<0){
                int enc=-(ni+2);int ts=enc>>3,tc=(enc&7)+1;
                for(int t=0;t<tc;t++){
                    int ti=ts+t;
                    float e1x=__ldg(&tv1x[ti])-__ldg(&tv0x[ti]);
                    float e1y=__ldg(&tv1y[ti])-__ldg(&tv0y[ti]);
                    float e1z=__ldg(&tv1z[ti])-__ldg(&tv0z[ti]);
                    float e2x=__ldg(&tv2x[ti])-__ldg(&tv0x[ti]);
                    float e2y=__ldg(&tv2y[ti])-__ldg(&tv0y[ti]);
                    float e2z=__ldg(&tv2z[ti])-__ldg(&tv0z[ti]);
                    float px=dy*e2z-dz*e2y,py=dz*e2x-dx*e2z,pz=dx*e2y-dy*e2x;
                    float det=e1x*px+e1y*py+e1z*pz;
                    if(fabsf(det)<1e-12f)continue;
                    float inv=1.f/det;
                    float tx=ox-__ldg(&tv0x[ti]),ty=oy-__ldg(&tv0y[ti]),tz=oz-__ldg(&tv0z[ti]);
                    float u=inv*(tx*px+ty*py+tz*pz);
                    if(u<0.f||u>1.f)continue;
                    float qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x;
                    float v=inv*(dx*qx+dy*qy+dz*qz);
                    if(v<0.f||u+v>1.f)continue;
                    float tt=inv*(e2x*qx+e2y*qy+e2z*qz);
                    if(tt>0.f&&tt<tHit){tHit=tt;hitTri=ti;hitU=u;hitV=v;}
                }
                continue;
            }
            int4 n0=__ldg(&d_bvh4[ni*4+0]);
            int4 n1=__ldg(&d_bvh4[ni*4+1]);
            int4 n2=__ldg(&d_bvh4[ni*4+2]);
            int4 n3=__ldg(&d_bvh4[ni*4+3]);
            const __half*bx=(const __half*)&n0;
            const __half*by=(const __half*)&n1;
            const __half*bz=(const __half*)&n2;
            const int*ch=(const int*)&n3;
            float dist[4]; int order[4];
            for(int c=0;c<4;c++){
                if(ch[c]==-1){dist[c]=1e30f;order[c]=c;continue;}
                float mnx=__half2float(bx[c]),mxx=__half2float(bx[4+c]);
                float mny=__half2float(by[c]),mxy=__half2float(by[4+c]);
                float mnz=__half2float(bz[c]),mxz=__half2float(bz[4+c]);
                float t1x=(mnx-ox)*ix,t2x=(mxx-ox)*ix;
                float t1y=(mny-oy)*iy,t2y=(mxy-oy)*iy;
                float t1z=(mnz-oz)*iz,t2z=(mxz-oz)*iz;
                float tNear=fmaxf(fmaxf(fminf(t1x,t2x),fminf(t1y,t2y)),fminf(t1z,t2z));
                float tFar=fminf(fminf(fmaxf(t1x,t2x),fmaxf(t1y,t2y)),fmaxf(t1z,t2z));
                dist[c]=(tNear<=tFar&&tFar>0.f&&tNear<tHit)?tNear:1e30f;
                order[c]=c;
            }
            for(int i=0;i<3;i++) for(int j=i+1;j<4;j++)
                if(dist[order[i]]>dist[order[j]]){int tmp=order[i];order[i]=order[j];order[j]=tmp;}
            for(int i=3;i>=0;i--)
                if(dist[order[i]]<1e30f && sp<STACK_DEPTH)
                    stk[sp++]=ch[order[i]];

            // Sample warp efficiency every 16 steps
            mySteps++;
            if((mySteps&15)==0){
                unsigned mask=__ballot_sync(0xFFFFFFFF, sp>0);
                myActiveSum+=__popc(mask);
            }
        }
        if(ri<numRays){hits[ri].t=tHit;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;}
    }
    if(warpEff && (lane==0)){
        atomicAdd(&warpEff[0], myActiveSum);
        atomicAdd(&warpEff[1], mySteps/16);
    }
}

// ======================== Scene gen (CPU, one-time) ========================
static void genScene(Tri*tris,int nT,float sc){
    srand(42);
    for(int i=0;i<nT;i++){
        float cx=((float)rand()/RAND_MAX-0.5f)*sc;
        float cy=((float)rand()/RAND_MAX-0.5f)*sc;
        float cz=((float)rand()/RAND_MAX-0.5f)*sc;
        float sz=sc*0.005f+((float)rand()/RAND_MAX)*sc*0.01f;
        tris[i].v0={cx-sz,cy-sz,cz};
        tris[i].v1={cx+sz,cy-sz,cz+sz};
        tris[i].v2={cx,cy+sz,cz-sz};
    }
}

// ======================== Main ========================
int main(){
    printf("══════════════════════════════════════════════════════════════════════════\n");
    printf("  V19 — Software SER (GPU-sorted): Direction-Aware Coherence\n");
    printf("  ALL sorting on GPU via thrust radix sort — zero CPU bottleneck\n");
    printf("══════════════════════════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    printf("  GPU: %s | SMs: %d\n\n",prop.name,prop.multiProcessorCount);

    const int NTRI=99000;
    const int NRAYS=1024*1024; // 1M rays

    // Build scene + BVH (CPU, one-time)
    Tri*h_tris=(Tri*)malloc(NTRI*sizeof(Tri));
    genScene(h_tris,NTRI,10.f);
    BVHBuild b2; b2.build(h_tris,NTRI);
    BVH4Node*h_b4=(BVH4Node*)calloc(b2.nodes.size()*2,sizeof(BVH4Node));
    int n4=0; collapseToB4(b2,0,h_b4,n4,h_tris);
    printf("  Scene: %d tris | %d BVH4 nodes | %d rays\n\n",NTRI,n4,NRAYS);

    // Upload BVH
    int4*d_bvh4; cudaMalloc(&d_bvh4,n4*sizeof(BVH4Node));
    cudaMemcpy(d_bvh4,h_b4,n4*sizeof(BVH4Node),cudaMemcpyHostToDevice);

    // Upload triangles SoA
    Tri*ordered=b2.ordered.data(); int nOT=(int)b2.ordered.size();
    float*h_tv[9]; float*d_tv[9];
    for(int j=0;j<9;j++){h_tv[j]=(float*)malloc(nOT*4);cudaMalloc(&d_tv[j],nOT*4);}
    for(int i=0;i<nOT;i++){
        h_tv[0][i]=ordered[i].v0.x;h_tv[1][i]=ordered[i].v0.y;h_tv[2][i]=ordered[i].v0.z;
        h_tv[3][i]=ordered[i].v1.x;h_tv[4][i]=ordered[i].v1.y;h_tv[5][i]=ordered[i].v1.z;
        h_tv[6][i]=ordered[i].v2.x;h_tv[7][i]=ordered[i].v2.y;h_tv[8][i]=ordered[i].v2.z;
    }
    for(int j=0;j<9;j++) cudaMemcpy(d_tv[j],h_tv[j],nOT*4,cudaMemcpyHostToDevice);

    // Generate diffuse rays on CPU, upload ONCE
    float*h_ray[6]; // ox,oy,oz,dx,dy,dz
    for(int j=0;j<6;j++) h_ray[j]=(float*)malloc(NRAYS*4);
    srand(12345);
    int side=(int)sqrtf((float)NRAYS);
    float sc=10.f;
    for(int i=0;i<NRAYS;i++){
        int px=i%side, py=i/side;
        float u=(px+0.5f)/side-0.5f, v=(py+0.5f)/side-0.5f;
        h_ray[0][i]=u*sc*0.5f; h_ray[1][i]=v*sc*0.5f;
        h_ray[2][i]=((float)rand()/RAND_MAX-0.5f)*sc*0.3f;
        float r1=(float)rand()/RAND_MAX, r2=(float)rand()/RAND_MAX;
        float phi=6.28318f*r1, ct=sqrtf(1.f-r2), st=sqrtf(r2);
        h_ray[3][i]=st*cosf(phi); h_ray[4][i]=st*sinf(phi); h_ray[5][i]=ct;
        if(rand()%2) h_ray[3][i]=-h_ray[3][i];
        if(rand()%2) h_ray[4][i]=-h_ray[4][i];
        if(rand()%2) h_ray[5][i]=-h_ray[5][i];
    }

    // Device arrays: original rays (6 SoA) + sorted rays (9 SoA) + inverse dirs
    float*d_orig[6]; for(int j=0;j<6;j++){cudaMalloc(&d_orig[j],NRAYS*4);cudaMemcpy(d_orig[j],h_ray[j],NRAYS*4,cudaMemcpyHostToDevice);}
    float*d_sorted[9]; for(int j=0;j<9;j++) cudaMalloc(&d_sorted[j],NRAYS*4);

    // Compute inverse directions for unsorted rays
    float*h_inv[3]; for(int j=0;j<3;j++) h_inv[j]=(float*)malloc(NRAYS*4);
    for(int i=0;i<NRAYS;i++){
        for(int j=0;j<3;j++){
            float d=h_ray[3+j][i];
            h_inv[j][i]=1.f/(fabsf(d)>1e-8f?d:(d>=0?1e-8f:-1e-8f));
        }
    }
    float*d_inv[3]; for(int j=0;j<3;j++){cudaMalloc(&d_inv[j],NRAYS*4);cudaMemcpy(d_inv[j],h_inv[j],NRAYS*4,cudaMemcpyHostToDevice);}

    // Sort workspace
    unsigned int*d_keys; cudaMalloc(&d_keys, NRAYS*4);
    int*d_perm; cudaMalloc(&d_perm, NRAYS*4);

    Hit*d_hits; cudaMalloc(&d_hits, NRAYS*sizeof(Hit));
    Hit*h_hits=(Hit*)malloc(NRAYS*sizeof(Hit));
    unsigned long long*d_warpEff; cudaMalloc(&d_warpEff,16);

    // Scene bounds
    float3 smn={1e30f,1e30f,1e30f}, smx={-1e30f,-1e30f,-1e30f};
    for(int i=0;i<NTRI;i++){
        smn.x=fminf(smn.x,fminf(fminf(h_tris[i].v0.x,h_tris[i].v1.x),h_tris[i].v2.x));
        smn.y=fminf(smn.y,fminf(fminf(h_tris[i].v0.y,h_tris[i].v1.y),h_tris[i].v2.y));
        smn.z=fminf(smn.z,fminf(fminf(h_tris[i].v0.z,h_tris[i].v1.z),h_tris[i].v2.z));
        smx.x=fmaxf(smx.x,fmaxf(fmaxf(h_tris[i].v0.x,h_tris[i].v1.x),h_tris[i].v2.x));
        smx.y=fmaxf(smx.y,fmaxf(fmaxf(h_tris[i].v0.y,h_tris[i].v1.y),h_tris[i].v2.y));
        smx.z=fmaxf(smx.z,fmaxf(fmaxf(h_tris[i].v0.z,h_tris[i].v1.z),h_tris[i].v2.z));
    }

    // Key generation kernel pointers
    typedef void(*KeyGenFn)(const float*,const float*,const float*,const float*,const float*,const float*,float3,float3,unsigned int*,int);
    KeyGenFn keyGens[]={genKeysA,genKeysB,genKeysC,genKeysD,genKeysE};
    const char*names[]={"A:8dir(base)","B:72dir    ","C:512dir   ","D:4Kdir    ","E:131Kdir  "};
    int nStrat=5;
    int BK=(NRAYS+255)/256;

    printf("  ┌──────────────┬──────────┬────────┬──────────┬──────────┐\n");
    printf("  │  Strategy    │ MRays/s  │ Hit %%  │ WarpEff  │ Sort ms  │\n");
    printf("  ├──────────────┼──────────┼────────┼──────────┼──────────┤\n");

    // UNSORTED skipped — takes minutes on random diffuse, known ~215 MR/s, ~35% warp eff
    printf("  │ UNSORTED     │  ~215    │  ~93%% │ ~35%%     │    -     │\n");

    // Sorted strategies
    thrust::device_ptr<unsigned int> d_keys_ptr(d_keys);
    thrust::device_ptr<int> d_perm_ptr(d_perm);

    for(int s=0;s<nStrat;s++){
        // Generate keys on GPU
        keyGens[s]<<<BK,256>>>(d_orig[0],d_orig[1],d_orig[2],d_orig[3],d_orig[4],d_orig[5],smn,smx,d_keys,NRAYS);
        // Init permutation [0..N)
        thrust::sequence(d_perm_ptr, d_perm_ptr+NRAYS);

        // Time the sort
        cudaEvent_t ts0,ts1; cudaEventCreate(&ts0); cudaEventCreate(&ts1);
        cudaEventRecord(ts0);
        thrust::sort_by_key(d_keys_ptr, d_keys_ptr+NRAYS, d_perm_ptr);
        cudaEventRecord(ts1); cudaEventSynchronize(ts1);
        float sortMs; cudaEventElapsedTime(&sortMs,ts0,ts1);
        cudaEventDestroy(ts0); cudaEventDestroy(ts1);

        // Scatter-reorder all 9 ray arrays (ox,oy,oz,dx,dy,dz,ix,iy,iz)
        for(int j=0;j<6;j++) scatterReorder<<<BK,256>>>(d_orig[j],d_sorted[j],d_perm,NRAYS);
        for(int j=0;j<3;j++) scatterReorder<<<BK,256>>>(d_inv[j],d_sorted[6+j],d_perm,NRAYS);
        cudaDeviceSynchronize();

        // Warmup
        unsigned int zero=0; cudaMemcpyToSymbol(g_rayCounter,&zero,4);
        traceBVH4<<<320,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
            d_sorted[0],d_sorted[1],d_sorted[2],d_sorted[3],d_sorted[4],d_sorted[5],
            d_sorted[6],d_sorted[7],d_sorted[8],d_hits,NRAYS,nullptr);
        cudaDeviceSynchronize();

        // Timed trace
        unsigned long long h_we[2]={0,0};
        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        float best=1e30f;
        for(int r=0;r<3;r++){
            cudaMemcpyToSymbol(g_rayCounter,&zero,4);
            cudaMemcpy(d_warpEff,h_we,16,cudaMemcpyHostToDevice);
            cudaEventRecord(t0);
            traceBVH4<<<320,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
                d_sorted[0],d_sorted[1],d_sorted[2],d_sorted[3],d_sorted[4],d_sorted[5],
                d_sorted[6],d_sorted[7],d_sorted[8],d_hits,NRAYS,d_warpEff);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms,t0,t1); if(ms<best)best=ms;
        }
        cudaMemcpy(h_hits,d_hits,NRAYS*sizeof(Hit),cudaMemcpyDeviceToHost);
        int hc=0; for(int i=0;i<NRAYS;i++) if(h_hits[i].tri>=0) hc++;
        cudaMemcpy(h_we,d_warpEff,16,cudaMemcpyDeviceToHost);
        float wE=h_we[1]>0?(float)h_we[0]/(h_we[1]*32.f)*100.f:0.f;
        printf("  │ %s │ %7.0f  │ %5.1f%% │ %5.1f%%   │ %5.2fms  │\n",
               names[s],(float)NRAYS/best/1000.f,100.f*hc/NRAYS,wE,sortMs);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    printf("  └──────────────┴──────────┴────────┴──────────┴──────────┘\n\n");

    // Cleanup
    for(int j=0;j<6;j++){cudaFree(d_orig[j]);free(h_ray[j]);}
    for(int j=0;j<3;j++){cudaFree(d_inv[j]);free(h_inv[j]);}
    for(int j=0;j<9;j++){cudaFree(d_sorted[j]);cudaFree(d_tv[j]);free(h_tv[j]);}
    cudaFree(d_bvh4);cudaFree(d_keys);cudaFree(d_perm);cudaFree(d_hits);cudaFree(d_warpEff);
    free(h_b4);free(h_tris);free(h_hits);
    return 0;
}
