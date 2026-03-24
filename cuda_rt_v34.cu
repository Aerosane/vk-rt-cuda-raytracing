/*
 * V34 — Wavefront PBR Path Tracer (NVIDIA Research algorithms)
 *
 * Key change: Megakernel (v33) -> Wavefront architecture (Laine 2013)
 *
 * Problem: v33 pathTrace megakernel has 7.24/32 active threads per warp
 *          (22.6% SIMT utilization) due to path tracing control flow divergence.
 *          62% of cycles, no warp is eligible to issue.
 *
 * Solution: Decompose into separate high-utilization kernels:
 *   1. generateRays     — camera rays, 100% utilization
 *   2. traceExtension   — BVH traversal only, ~60-80% util
 *   3. shadeMaterial     — material eval + NEE, sorted by mat type = ~95% util
 *   4. traceShadow       — shadow ray BVH traversal, ~70% util
 *
 * Additional NVIDIA Research tricks:
 *   - Speculative traversal (Aila & Laine 2009)
 *   - Material-sorted shading (software SER)
 *   - Compressed ray state between stages
 *   - Persistent threads with dynamic ray fetch
 *
 * References:
 *   [1] Laine, Karras, Aila - "Megakernels Considered Harmful" HPG 2013
 *   [2] Aila, Laine - "Understanding the Efficiency of Ray Traversal on GPUs" HPG 2009
 *   [3] Ylitie, Karras, Laine - "CWBVH" HPG 2017
 *   [4] NVIDIA SER Whitepaper 2022
 *
 * Build: nvcc -O3 -arch=sm_70 --use_fast_math -Wno-deprecated-gpu-targets
 *        -Xcompiler "-O3 -march=native" -Xptxas -v -o v34 cuda_rt_v34.cu
 * Run:   ./v34 [max_spp] [width] [mode]
 *        mode: 0=wavefront (default), 1=megakernel (v33 baseline)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cfloat>
#include <map>

#define CK(call) do{cudaError_t e=(call);if(e!=cudaSuccess){printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

#define STACK_DEPTH 16
#define CONST_BVH4 1010
#define MAX_BOUNCES 8
#define FIREFLY_CLAMP 20.0f
#define RR_START_BOUNCE 2
#define PI 3.14159265358979323846f
#define INV_PI 0.3183098861837907f
#define EPS 1e-4f
#define ADAPTIVE_MIN_SPP 4
#define ADAPTIVE_THRESH 0.03f

// ===================== Wavefront Ray Queues =====================
// Compact per-ray state stored in global memory between stages
// Minimized to reduce memory traffic (Laine 2013 key insight)
struct RayState {
    float ox, oy, oz;       // origin
    float dx, dy, dz;       // direction
    float tpR, tpG, tpB;    // throughput
    float radR, radG, radB; // accumulated radiance
    int pixelIdx;           // pixel index
    int bounce;             // current bounce depth
    int flags;              // bit0: specularBounce, bit1: terminated
    uint32_t rngState;      // RNG state for continuation
};

// Hit result (written by trace, read by shade)
struct HitResult {
    float t;
    int id;
    float u, v;
};

// Shadow ray (simplified — just need occluded/not)
struct ShadowRay {
    float ox, oy, oz;
    float dx, dy, dz;
    float tmax;
    float contribR, contribG, contribB; // light contribution if unoccluded
    int pixelIdx;
};

// ===================== Geometry / Scene Types (from v33) =====================
struct float3a { float x, y, z; };
struct AABB { float3a mn, mx; };
struct Tri { float3a v0, v1, v2; };
struct BVH4Node {
    __half boundsX[8], boundsY[8], boundsZ[8];
    int child[4];
};
struct Hit { float t; int id; float u, v; }; // id>=0: tri, id<0: sphere

// Material types
enum MatType : uint8_t { MAT_DIFFUSE=0, MAT_METAL=1, MAT_GLASS=2, MAT_EMIT=3, MAT_GLOSSY=4 };

struct Material {
    float r, g, b;       // albedo/color
    float roughness;     // 0=mirror, 1=rough
    float metallic;      // 0=dielectric, 1=metal
    float ior;           // index of refraction (glass)
    float er, eg, eb;    // emission
    MatType type;
    uint8_t checker;     // 1=checker pattern on this material
    uint8_t pad[2];
};

struct Sphere { float cx, cy, cz, radius; int matId; };

struct GBuf {
    float depth;
    float nx, ny, nz;
    float ar, ag, ab;
};

// ===================== Constant Memory =====================
__constant__ int4 c_bvh4[CONST_BVH4 * 4];
__constant__ int c_bvh4N;
__constant__ Material c_materials[16];
__constant__ int c_numMaterials;
__constant__ Sphere c_spheres[8];
__constant__ int c_numSpheres;

// Area light
__constant__ float c_lCorner[3], c_lE1[3], c_lE2[3], c_lNorm[3];
__constant__ float c_lArea, c_lEmit[3];

// ===================== Vector Math (all __host__ __device__ __forceinline__) =====================
struct f3 { float x, y, z; };
__host__ __device__ __forceinline__ f3 F3(float x, float y, float z) { return {x,y,z}; }
__host__ __device__ __forceinline__ f3 operator+(f3 a, f3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__host__ __device__ __forceinline__ f3 operator-(f3 a, f3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__host__ __device__ __forceinline__ f3 operator*(f3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
__host__ __device__ __forceinline__ f3 operator*(float s, f3 a) { return {a.x*s, a.y*s, a.z*s}; }
__host__ __device__ __forceinline__ f3 operator*(f3 a, f3 b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }
__host__ __device__ __forceinline__ f3 operator/(f3 a, float s) { return {a.x/s, a.y/s, a.z/s}; }
__host__ __device__ __forceinline__ float dot(f3 a, f3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__host__ __device__ __forceinline__ f3 cross(f3 a, f3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
__host__ __device__ __forceinline__ float len(f3 v) { return sqrtf(dot(v,v)); }
__host__ __device__ __forceinline__ f3 norm(f3 v) { return v * rsqrtf(fmaxf(dot(v,v), 1e-20f)); }
__device__ __forceinline__ f3 reflect(f3 v, f3 n) { return v - n * (2.f * dot(v,n)); }
__device__ __forceinline__ float maxcomp(f3 v) { return fmaxf(fmaxf(v.x,v.y),v.z); }
__device__ __forceinline__ float luminance(f3 c) { return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z; }

// Procedural checker texture
__device__ __forceinline__ f3 checkerAlbedo(f3 pos, f3 baseColor) {
    int cx = (int)floorf(pos.x * 1.8f);
    int cz = (int)floorf(pos.z * 1.8f);
    return ((cx + cz) & 1) ? baseColor : baseColor * 0.15f;
}

__device__ __forceinline__ void buildONB(f3 n, f3& t, f3& b) {
    f3 up = (fabsf(n.y) < 0.999f) ? F3(0,1,0) : F3(1,0,0);
    t = norm(cross(up, n)); b = cross(n, t);
}

// ===================== RNG (PCG) =====================
struct RNG {
    uint32_t s;
    __device__ __forceinline__ void init(uint32_t px, uint32_t sample) {
        s = px * 747796405u + sample * 2891336453u + 0xBADCAFEu;
        s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; s = (s >> 22u) ^ s;
    }
    __device__ __forceinline__ float f() {
        s = s * 747796405u + 2891336453u;
        uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
        return ((w >> 22u) ^ w) * 2.3283064365386963e-10f;
    }
};

// ===================== GGX Microfacet BRDF =====================

// GGX NDF
__device__ __forceinline__ float D_GGX(float NdH, float a2) {
    float d = NdH * NdH * (a2 - 1.f) + 1.f;
    return a2 / (PI * d * d + 1e-7f);
}

// Smith G1
__device__ __forceinline__ float G1_Smith(float NdV, float a2) {
    return 2.f * NdV / (NdV + sqrtf(a2 + (1.f - a2) * NdV * NdV) + 1e-7f);
}

// Smith G2 (height-correlated)
__device__ __forceinline__ float G2_Smith(float NdL, float NdV, float a2) {
    float g1l = NdV * sqrtf(a2 + (1.f - a2) * NdL * NdL);
    float g1v = NdL * sqrtf(a2 + (1.f - a2) * NdV * NdV);
    return 2.f * NdL * NdV / (g1l + g1v + 1e-7f);
}

// Schlick Fresnel
__device__ __forceinline__ f3 F_Schlick(float HdV, f3 F0) {
    float t = 1.f - HdV; float t5 = t*t*t*t*t;
    return F0 + (F3(1,1,1) - F0) * t5;
}
__device__ __forceinline__ float F_Schlick1(float HdV, float F0) {
    float t = 1.f - HdV; float t5 = t*t*t*t*t;
    return F0 + (1.f - F0) * t5;
}

// Fresnel for dielectric (exact)
__device__ __forceinline__ float fresnelDielectric(float cosI, float eta) {
    float sinT2 = eta * eta * (1.f - cosI * cosI);
    if (sinT2 > 1.f) return 1.f; // TIR
    float cosT = sqrtf(1.f - sinT2);
    float Rs = (cosI - eta * cosT) / (cosI + eta * cosT + 1e-8f);
    float Rp = (eta * cosI - cosT) / (eta * cosI + cosT + 1e-8f);
    return 0.5f * (Rs * Rs + Rp * Rp);
}

// GGX VNDF importance sampling (Heitz 2018)
__device__ __forceinline__ f3 sampleGGX_VNDF(f3 Ve, float alpha, RNG& rng) {
    // Transform view to hemisphere config
    f3 Vh = norm(F3(alpha * Ve.x, alpha * Ve.y, Ve.z));
    // Build ONB around Vh
    float lensq = Vh.x*Vh.x + Vh.y*Vh.y;
    f3 T1 = lensq > 1e-7f ? F3(-Vh.y, Vh.x, 0) * rsqrtf(lensq) : F3(1,0,0);
    f3 T2 = cross(Vh, T1);
    // Sample point on disk
    float r = sqrtf(rng.f());
    float phi = 2.f * PI * rng.f();
    float t1 = r * cosf(phi), t2 = r * sinf(phi);
    float s = 0.5f * (1.f + Vh.z);
    t2 = (1.f - s) * sqrtf(1.f - t1*t1) + s * t2;
    // Compute half-vector
    f3 Nh = T1 * t1 + T2 * t2 + Vh * sqrtf(fmaxf(0.f, 1.f - t1*t1 - t2*t2));
    // Transform back
    return norm(F3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.f, Nh.z)));
}

// PDF of VNDF sampling
__device__ __forceinline__ float pdfGGX_VNDF(float NdH, float HdV, float NdV, float a2) {
    float D = D_GGX(NdH, a2);
    float G1 = G1_Smith(NdV, a2);
    return D * G1 / (4.f * NdV + 1e-7f); // = D * G1 * HdV / (NdV * 4 * HdV)
}

// Cosine hemisphere sample + pdf
__device__ __forceinline__ f3 sampleCosHemi(f3 N, RNG& rng) {
    float r1 = rng.f(), r2 = rng.f();
    float cosT = sqrtf(r2), sinT = sqrtf(1.f - r2);
    float phi = 2.f * PI * r1;
    f3 t, b; buildONB(N, t, b);
    return norm(t*(sinT*cosf(phi)) + b*(sinT*sinf(phi)) + N*cosT);
}

// Light sample point
__device__ __forceinline__ f3 sampleLightPt(RNG& rng) {
    float u = rng.f(), v = rng.f();
    return F3(c_lCorner[0]+u*c_lE1[0]+v*c_lE2[0],
              c_lCorner[1]+u*c_lE1[1]+v*c_lE2[1],
              c_lCorner[2]+u*c_lE1[2]+v*c_lE2[2]);
}

// Evaluate full Cook-Torrance BRDF * NdotL
__device__ __forceinline__ f3 evalBRDF(
    f3 N, f3 V, f3 L, f3 albedo, float roughness, float metallic, f3& F_out)
{
    float NdL = dot(N, L), NdV = fmaxf(dot(N, V), 0.001f);
    if (NdL <= 0.f) return F3(0,0,0);
    f3 H = norm(V + L);
    float NdH = fmaxf(dot(N, H), 0.f);
    float HdV = fmaxf(dot(H, V), 0.f);
    float a2 = fmaxf(roughness * roughness * roughness * roughness, 1e-4f);
    f3 F0 = F3(0.04f,0.04f,0.04f) * (1.f-metallic) + albedo * metallic;
    f3 F = F_Schlick(HdV, F0);
    F_out = F;
    float D = D_GGX(NdH, a2);
    float G = G2_Smith(NdL, NdV, a2);
    f3 spec = F * (D * G / (4.f * NdV * NdL + 1e-7f));
    f3 diff = (F3(1,1,1) - F) * (1.f - metallic) * albedo * INV_PI;
    return (diff + spec) * NdL;
}

// ===================== BVH4 Node Loader =====================
__device__ __forceinline__ void loadNode(const int4* __restrict__ bvh, int ni,
    int4& n0, int4& n1, int4& n2, int4& n3) {
    if (ni < c_bvh4N) {
        n0=c_bvh4[ni*4]; n1=c_bvh4[ni*4+1]; n2=c_bvh4[ni*4+2]; n3=c_bvh4[ni*4+3];
    } else {
        n0=__ldg(&bvh[ni*4]); n1=__ldg(&bvh[ni*4+1]); n2=__ldg(&bvh[ni*4+2]); n3=__ldg(&bvh[ni*4+3]);
    }
}

// ===================== BVH Builder (SAH, from v29) =====================
static AABB triAABB(const Tri& t) {
    AABB b;
    b.mn.x=fminf(fminf(t.v0.x,t.v1.x),t.v2.x); b.mn.y=fminf(fminf(t.v0.y,t.v1.y),t.v2.y); b.mn.z=fminf(fminf(t.v0.z,t.v1.z),t.v2.z);
    b.mx.x=fmaxf(fmaxf(t.v0.x,t.v1.x),t.v2.x); b.mx.y=fmaxf(fmaxf(t.v0.y,t.v1.y),t.v2.y); b.mx.z=fmaxf(fmaxf(t.v0.z,t.v1.z),t.v2.z);
    return b;
}
static float saArea(const AABB& b) {
    float dx=b.mx.x-b.mn.x,dy=b.mx.y-b.mn.y,dz=b.mx.z-b.mn.z;
    return 2.f*(dx*dy+dy*dz+dx*dz);
}
static AABB mergeAABB(const AABB& a, const AABB& b) {
    AABB r;
    r.mn.x=fminf(a.mn.x,b.mn.x); r.mn.y=fminf(a.mn.y,b.mn.y); r.mn.z=fminf(a.mn.z,b.mn.z);
    r.mx.x=fmaxf(a.mx.x,b.mx.x); r.mx.y=fmaxf(a.mx.y,b.mx.y); r.mx.z=fmaxf(a.mx.z,b.mx.z);
    return r;
}

struct BVHBuild {
    struct N2 { AABB box; int left,right,triStart,triCount; };
    std::vector<N2> nodes;
    std::vector<Tri> ordered;
    std::vector<int> orderedMatId;
    const Tri* src; const int* srcMat;
    std::vector<AABB> primBB;
    std::vector<float3a> centroids;

    void build(const Tri* t, const int* mat, int n) {
        src=t; srcMat=mat; primBB.resize(n); centroids.resize(n);
        ordered.clear(); orderedMatId.clear();
        for(int i=0;i<n;i++){primBB[i]=triAABB(t[i]); centroids[i]={
            (primBB[i].mn.x+primBB[i].mx.x)*.5f,(primBB[i].mn.y+primBB[i].mx.y)*.5f,(primBB[i].mn.z+primBB[i].mx.z)*.5f};}
        std::vector<int> idx(n); for(int i=0;i<n;i++) idx[i]=i;
        rec(idx,0,n);
    }
    int rec(std::vector<int>& idx, int s, int e) {
        N2 nd; nd.triStart=nd.triCount=nd.left=nd.right=0;
        nd.box=primBB[idx[s]];
        for(int i=s+1;i<e;i++) nd.box=mergeAABB(nd.box,primBB[idx[i]]);
        int cnt=e-s;
        if(cnt<=4){
            nd.triStart=(int)ordered.size(); nd.triCount=cnt;
            for(int i=s;i<e;i++){ordered.push_back(src[idx[i]]); orderedMatId.push_back(srcMat[idx[i]]);}
            nodes.push_back(nd); return (int)nodes.size()-1;
        }
        float bestCost=1e30f; int bestAxis=0, bestSplit=s+cnt/2;
        float pA=saArea(nd.box);
        for(int ax=0;ax<3;ax++){
            float cmin=1e30f,cmax=-1e30f;
            for(int i=s;i<e;i++){float c=(&centroids[idx[i]].x)[ax]; cmin=fminf(cmin,c); cmax=fmaxf(cmax,c);}
            if(cmax-cmin<1e-8f) continue;
            const int NB=16;
            AABB lB[NB],rB[NB]; int lC[NB],rC[NB];
            for(int b=0;b<NB;b++){lB[b].mn={1e30f,1e30f,1e30f};lB[b].mx={-1e30f,-1e30f,-1e30f};lC[b]=0;
                                   rB[b].mn={1e30f,1e30f,1e30f};rB[b].mx={-1e30f,-1e30f,-1e30f};rC[b]=0;}
            for(int i=s;i<e;i++){float c=(&centroids[idx[i]].x)[ax]; int b=(int)((c-cmin)/(cmax-cmin)*(NB-1));
                b=b<0?0:(b>=NB?NB-1:b); lB[b]=lC[b]?mergeAABB(lB[b],primBB[idx[i]]):primBB[idx[i]]; lC[b]++;}
            for(int b=1;b<NB;b++){if(lC[b]&&lC[b-1])lB[b]=mergeAABB(lB[b],lB[b-1]);else if(lC[b-1])lB[b]=lB[b-1]; lC[b]+=lC[b-1];}
            for(int i=e-1;i>=s;i--){float c=(&centroids[idx[i]].x)[ax]; int b=(int)((c-cmin)/(cmax-cmin)*(NB-1));
                b=b<0?0:(b>=NB?NB-1:b); rB[b]=rC[b]?mergeAABB(rB[b],primBB[idx[i]]):primBB[idx[i]]; rC[b]++;}
            for(int b=NB-2;b>=0;b--){if(rC[b]&&rC[b+1])rB[b]=mergeAABB(rB[b],rB[b+1]);else if(rC[b+1])rB[b]=rB[b+1]; rC[b]+=rC[b+1];}
            for(int b=0;b<NB-1;b++){
                if(!lC[b]||!rC[b+1]) continue;
                float cost=lC[b]*saArea(lB[b])/pA+rC[b+1]*saArea(rB[b+1])/pA+1.f;
                if(cost<bestCost){bestCost=cost; bestAxis=ax;
                    float splitC=cmin+(b+1.f)/NB*(cmax-cmin); bestSplit=s;
                    for(int i=s;i<e;i++) if((&centroids[idx[i]].x)[ax]<splitC) bestSplit++;
                    bestSplit=bestSplit<=s?s+1:(bestSplit>=e?e-1:bestSplit);}
            }
        }
        if(bestSplit<=s) bestSplit=s+1; if(bestSplit>=e) bestSplit=e-1;
        std::sort(idx.begin()+s,idx.begin()+e,[&](int a,int b){return (&centroids[a].x)[bestAxis]<(&centroids[b].x)[bestAxis];});
        int id=(int)nodes.size(); nodes.push_back(nd);
        nodes[id].left=rec(idx,s,bestSplit); nodes[id].right=rec(idx,bestSplit,e);
        return id;
    }
};

static int collapseB4(const BVHBuild& b2, int ni, BVH4Node* out, int& cnt) {
    auto& n=b2.nodes[ni];
    if(n.triCount>0) return -((n.triStart<<3)|(n.triCount-1))-2;
    int gather[4]; int ng=0;
    int ch[2]={n.left,n.right};
    for(int c=0;c<2;c++){auto& cn=b2.nodes[ch[c]]; if(cn.triCount>0||ng>=3){gather[ng++]=ch[c]; continue;} gather[ng++]=cn.left; gather[ng++]=cn.right;}
    BVH4Node nd;
    for(int i=0;i<4;i++){
        if(i<ng){auto& cn=b2.nodes[gather[i]];
            nd.boundsX[i]=__float2half(cn.box.mn.x); nd.boundsX[4+i]=__float2half(cn.box.mx.x);
            nd.boundsY[i]=__float2half(cn.box.mn.y); nd.boundsY[4+i]=__float2half(cn.box.mx.y);
            nd.boundsZ[i]=__float2half(cn.box.mn.z); nd.boundsZ[4+i]=__float2half(cn.box.mx.z);
        } else {
            nd.boundsX[i]=__float2half(1e30f); nd.boundsX[4+i]=__float2half(-1e30f);
            nd.boundsY[i]=__float2half(1e30f); nd.boundsY[4+i]=__float2half(-1e30f);
            nd.boundsZ[i]=__float2half(1e30f); nd.boundsZ[4+i]=__float2half(-1e30f);
        }
        nd.child[i]=-1;
    }
    int me=cnt++;
    for(int i=0;i<ng;i++) nd.child[i]=collapseB4(b2,gather[i],out,cnt);
    out[me]=nd; return me;
}

// ===================== Sphere Intersection =====================
__device__ __forceinline__ float hitSphere(f3 o, f3 d, Sphere s) {
    f3 oc = F3(o.x-s.cx, o.y-s.cy, o.z-s.cz);
    float b = dot(oc,d), c = dot(oc,oc) - s.radius*s.radius;
    float disc = b*b - c;
    if (disc < 0.f) return -1.f;
    float sd = sqrtf(disc);
    float t1 = -b - sd; if (t1 > EPS) return t1;
    float t2 = -b + sd; if (t2 > EPS) return t2;
    return -1.f;
}

// ===================== Full Scene Trace (BVH + Spheres) =====================
__device__ Hit traceScene(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    f3 o, f3 d, float tmax)
{
    float ix=1.f/(fabsf(d.x)>1e-8f?d.x:copysignf(1e-8f,d.x));
    float iy=1.f/(fabsf(d.y)>1e-8f?d.y:copysignf(1e-8f,d.y));
    float iz=1.f/(fabsf(d.z)>1e-8f?d.z:copysignf(1e-8f,d.z));
    float tHit=tmax; int hitId=-1; float hitU=0,hitV=0;

    // BVH4 traversal
    int stk[STACK_DEPTH]; int sp=0; stk[sp++]=0;
    while(sp>0){
        int ni=stk[--sp];
        if(ni<0){
            int enc=-(ni+2); int ts=enc>>3,tc=(enc&7)+1;
            for(int t=0;t<tc;t++){int ti=ts+t;
                float e1x=__ldg(&tv1x[ti])-__ldg(&tv0x[ti]),e1y=__ldg(&tv1y[ti])-__ldg(&tv0y[ti]),e1z=__ldg(&tv1z[ti])-__ldg(&tv0z[ti]);
                float e2x=__ldg(&tv2x[ti])-__ldg(&tv0x[ti]),e2y=__ldg(&tv2y[ti])-__ldg(&tv0y[ti]),e2z=__ldg(&tv2z[ti])-__ldg(&tv0z[ti]);
                float px=d.y*e2z-d.z*e2y,py=d.z*e2x-d.x*e2z,pz=d.x*e2y-d.y*e2x;
                float det=e1x*px+e1y*py+e1z*pz;
                if(fabsf(det)<1e-12f) continue;
                float inv=1.f/det;
                float tx=o.x-__ldg(&tv0x[ti]),ty=o.y-__ldg(&tv0y[ti]),tz=o.z-__ldg(&tv0z[ti]);
                float uu=inv*(tx*px+ty*py+tz*pz); if(uu<0.f||uu>1.f) continue;
                float qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x;
                float vv=inv*(d.x*qx+d.y*qy+d.z*qz); if(vv<0.f||uu+vv>1.f) continue;
                float tt=inv*(e2x*qx+e2y*qy+e2z*qz);
                if(tt>EPS&&tt<tHit){tHit=tt;hitId=ti;hitU=uu;hitV=vv;}
            }
            continue;
        }
        int4 n0,n1,n2,n3; loadNode(bvh,ni,n0,n1,n2,n3);
        const __half* bx=(const __half*)&n0,*by=(const __half*)&n1,*bz=(const __half*)&n2;
        const int* ch=(const int*)&n3;
        float dist[4]; int child[4];
        for(int c=0;c<4;c++){child[c]=ch[c]; if(ch[c]==-1){dist[c]=1e30f; continue;}
            float t1x=(__half2float(bx[c])-o.x)*ix,t2x=(__half2float(bx[4+c])-o.x)*ix;
            float t1y=(__half2float(by[c])-o.y)*iy,t2y=(__half2float(by[4+c])-o.y)*iy;
            float t1z=(__half2float(bz[c])-o.z)*iz,t2z=(__half2float(bz[4+c])-o.z)*iz;
            float tN=fmaxf(fmaxf(fminf(t1x,t2x),fminf(t1y,t2y)),fminf(t1z,t2z));
            float tF=fminf(fminf(fmaxf(t1x,t2x),fmaxf(t1y,t2y)),fmaxf(t1z,t2z));
            dist[c]=(tN<=tF&&tF>0.f&&tN<tHit)?tN:1e30f;}
        #define CSWAP(a,b) do{float da=dist[a],db=dist[b];int ca=child[a],cb=child[b];\
            bool sw=(da>db);dist[a]=sw?db:da;dist[b]=sw?da:db;child[a]=sw?cb:ca;child[b]=sw?ca:cb;}while(0)
        CSWAP(0,1);CSWAP(2,3);CSWAP(0,2);CSWAP(1,3);CSWAP(1,2);
        #undef CSWAP
        for(int c=3;c>=0;c--) if(dist[c]<1e30f&&sp<STACK_DEPTH) stk[sp++]=child[c];
    }

    // Sphere intersection
    for(int i=0;i<c_numSpheres;i++){
        float t=hitSphere(o,d,c_spheres[i]);
        if(t>0.f&&t<tHit){tHit=t; hitId=-(i+2); hitU=0; hitV=0;}
    }
    return {tHit, hitId, hitU, hitV};
}

// Shadow test (any-hit, early exit)
__device__ bool traceShadow(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const int* __restrict__ matIds,
    f3 o, f3 d, float tmax)
{
    float ix=1.f/(fabsf(d.x)>1e-8f?d.x:copysignf(1e-8f,d.x));
    float iy=1.f/(fabsf(d.y)>1e-8f?d.y:copysignf(1e-8f,d.y));
    float iz=1.f/(fabsf(d.z)>1e-8f?d.z:copysignf(1e-8f,d.z));
    int stk[STACK_DEPTH]; int sp=0; stk[sp++]=0;
    while(sp>0){
        int ni=stk[--sp];
        if(ni<0){
            int enc=-(ni+2); int ts=enc>>3,tc=(enc&7)+1;
            for(int t=0;t<tc;t++){int ti=ts+t;
                if(c_materials[matIds[ti]].type==MAT_EMIT) continue; // don't block on light
                if(c_materials[matIds[ti]].type==MAT_GLASS) continue; // glass is transparent for shadow
                float e1x=__ldg(&tv1x[ti])-__ldg(&tv0x[ti]),e1y=__ldg(&tv1y[ti])-__ldg(&tv0y[ti]),e1z=__ldg(&tv1z[ti])-__ldg(&tv0z[ti]);
                float e2x=__ldg(&tv2x[ti])-__ldg(&tv0x[ti]),e2y=__ldg(&tv2y[ti])-__ldg(&tv0y[ti]),e2z=__ldg(&tv2z[ti])-__ldg(&tv0z[ti]);
                float px=d.y*e2z-d.z*e2y,py=d.z*e2x-d.x*e2z,pz=d.x*e2y-d.y*e2x;
                float det=e1x*px+e1y*py+e1z*pz; if(fabsf(det)<1e-12f) continue;
                float inv=1.f/det;
                float tx=o.x-__ldg(&tv0x[ti]),ty=o.y-__ldg(&tv0y[ti]),tz=o.z-__ldg(&tv0z[ti]);
                float uu=inv*(tx*px+ty*py+tz*pz); if(uu<0.f||uu>1.f) continue;
                float qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x;
                float vv=inv*(d.x*qx+d.y*qy+d.z*qz); if(vv<0.f||uu+vv>1.f) continue;
                float tt=inv*(e2x*qx+e2y*qy+e2z*qz);
                if(tt>EPS&&tt<tmax) return true;
            }
            continue;
        }
        int4 n0,n1,n2,n3; loadNode(bvh,ni,n0,n1,n2,n3);
        const __half* bx=(const __half*)&n0,*by=(const __half*)&n1,*bz=(const __half*)&n2;
        const int* ch=(const int*)&n3;
        for(int c=0;c<4;c++){if(ch[c]==-1)continue;
            float t1x=(__half2float(bx[c])-o.x)*ix,t2x=(__half2float(bx[4+c])-o.x)*ix;
            float t1y=(__half2float(by[c])-o.y)*iy,t2y=(__half2float(by[4+c])-o.y)*iy;
            float t1z=(__half2float(bz[c])-o.z)*iz,t2z=(__half2float(bz[4+c])-o.z)*iz;
            float tN=fmaxf(fmaxf(fminf(t1x,t2x),fminf(t1y,t2y)),fminf(t1z,t2z));
            float tF=fminf(fminf(fmaxf(t1x,t2x),fmaxf(t1y,t2y)),fmaxf(t1z,t2z));
            if(tN<=tF&&tF>0.f&&tN<tmax&&sp<STACK_DEPTH) stk[sp++]=ch[c];
        }
    }
    // Shadow from opaque spheres
    for(int i=0;i<c_numSpheres;i++){
        if(c_materials[c_spheres[i].matId].type==MAT_GLASS) continue;
        float t=hitSphere(o,d,c_spheres[i]);
        if(t>EPS&&t<tmax) return true;
    }
    return false;
}

// Get triangle normal
__device__ __forceinline__ f3 triNormal(
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    int ti) {
    f3 e1={__ldg(&tv1x[ti])-__ldg(&tv0x[ti]),__ldg(&tv1y[ti])-__ldg(&tv0y[ti]),__ldg(&tv1z[ti])-__ldg(&tv0z[ti])};
    f3 e2={__ldg(&tv2x[ti])-__ldg(&tv0x[ti]),__ldg(&tv2y[ti])-__ldg(&tv0y[ti]),__ldg(&tv2z[ti])-__ldg(&tv0z[ti])};
    return norm(cross(e1,e2));
}

// Procedural sky environment
__device__ __forceinline__ f3 envColor(f3 d) {
    float t = 0.5f * (d.y + 1.0f);
    f3 sky = F3(0.5f,0.7f,1.0f) * t + F3(1.0f,1.0f,1.0f) * (1.f-t);
    return sky * 0.3f; // dim ambient
}

// ===================== Path Trace Kernel (Adaptive) =====================
__global__ void pathTrace(
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const int* __restrict__ matIds,
    float* __restrict__ aR, float* __restrict__ aG, float* __restrict__ aB,
    float* __restrict__ aR2, float* __restrict__ aG2, float* __restrict__ aB2, // squared accum for variance
    int* __restrict__ sppMap, // per-pixel sample count
    GBuf* __restrict__ gbuf,
    int W, int H, int sampleIdx,
    float camPx, float camPy, float camPz,
    float camFx, float camFy, float camFz,
    float camRx, float camRy, float camRz,
    float camUx, float camUy, float camUz,
    float fovTan, float aperture, float focusDist)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;
    int pidx = py * W + px;

    // Adaptive: skip converged pixels
    if (sampleIdx >= ADAPTIVE_MIN_SPP) {
        int n = sppMap[pidx];
        if (n >= ADAPTIVE_MIN_SPP) {
            float invN = 1.f / n;
            float meanR = aR[pidx] * invN, meanG = aG[pidx] * invN, meanB = aB[pidx] * invN;
            float varR = fmaxf(aR2[pidx] * invN - meanR*meanR, 0.f);
            float varG = fmaxf(aG2[pidx] * invN - meanG*meanG, 0.f);
            float varB = fmaxf(aB2[pidx] * invN - meanB*meanB, 0.f);
            float lum = 0.2126f*meanR + 0.7152f*meanG + 0.0722f*meanB;
            float relVar = (varR + varG + varB) / (3.f * lum * lum + 1e-6f);
            float relStddev = sqrtf(relVar / (n + 1e-6f));
            if (relStddev < ADAPTIVE_THRESH) return; // converged!
        }
    }

    RNG rng; rng.init(pidx, sampleIdx);

    // Sub-pixel jitter
    float u = (2.f * (px + rng.f()) / W - 1.f) * fovTan;
    float v = (2.f * (py + rng.f()) / H - 1.f) * fovTan * ((float)H / W);

    f3 fwd = F3(camFx,camFy,camFz);
    f3 right = F3(camRx,camRy,camRz);
    f3 up = F3(camUx,camUy,camUz);
    f3 dir = norm(fwd + right * u + up * v);
    f3 orig = F3(camPx, camPy, camPz);

    // Thin-lens DOF
    if (aperture > 0.f) {
        f3 focusPt = orig + dir * focusDist;
        float r = aperture * sqrtf(rng.f());
        float theta = 2.f * PI * rng.f();
        f3 offset = right * (r * cosf(theta)) + up * (r * sinf(theta));
        orig = orig + offset;
        dir = norm(focusPt - orig);
    }

    f3 radiance = F3(0,0,0);
    f3 throughput = F3(1,1,1);
    bool specularBounce = true; // first hit always counts emission

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        Hit hit = traceScene(bvh, n4,
            tv0x,tv0y,tv0z, tv1x,tv1y,tv1z, tv2x,tv2y,tv2z,
            orig, dir, 1e30f);

        if (hit.id == -1) {
            // Miss — sample environment
            radiance = radiance + throughput * envColor(dir);
            break;
        }

        // Determine hit geometry
        f3 hitPos = orig + dir * hit.t;
        f3 N; int matId;

        if (hit.id >= 0) {
            // Triangle
            N = triNormal(tv0x,tv0y,tv0z, tv1x,tv1y,tv1z, tv2x,tv2y,tv2z, hit.id);
            matId = matIds[hit.id];
        } else {
            // Sphere: id = -(sphereIdx+2)
            int si = -(hit.id + 2);
            Sphere sp = c_spheres[si];
            N = norm(F3(hitPos.x-sp.cx, hitPos.y-sp.cy, hitPos.z-sp.cz));
            matId = sp.matId;
        }

        Material mat = c_materials[matId];
        f3 albedo = F3(mat.r, mat.g, mat.b);
        // Procedural checker texture
        if (mat.checker) albedo = checkerAlbedo(hitPos, albedo);
        f3 V = norm(F3(0,0,0) - dir); // view direction (toward camera)
        bool frontFace = dot(N, V) > 0.f;
        if (!frontFace) N = F3(0,0,0) - N;

        // G-buffer on first hit
        if (bounce == 0 && sampleIdx == 0) {
            gbuf[pidx].depth = hit.t;
            gbuf[pidx].nx = N.x; gbuf[pidx].ny = N.y; gbuf[pidx].nz = N.z;
            gbuf[pidx].ar = mat.r; gbuf[pidx].ag = mat.g; gbuf[pidx].ab = mat.b;
        }

        // ---- EMISSIVE: collect if we came from specular (MIS: NEE handles diffuse paths) ----
        f3 emission = F3(mat.er, mat.eg, mat.eb);
        if (maxcomp(emission) > 0.f) {
            if (specularBounce) {
                radiance = radiance + throughput * emission;
            } else {
                // MIS weight for BRDF sampling hitting the light
                float lightDist = hit.t;
                f3 lN = F3(c_lNorm[0], c_lNorm[1], c_lNorm[2]);
                float cosLight = fmaxf(-dot(lN, dir), 0.f);
                float lightPdf = (lightDist * lightDist) / (cosLight * c_lArea + 1e-7f);
                float NdL = fmaxf(dot(N, dir), 0.f);
                float brdfPdf = NdL * INV_PI; // cosine pdf (approximation for MIS)
                float w = brdfPdf * brdfPdf / (brdfPdf * brdfPdf + lightPdf * lightPdf + 1e-10f);
                radiance = radiance + throughput * emission * w;
            }
            break;
        }

        // ===================== Material Dispatch =====================
        if (mat.type == MAT_GLASS) {
            // ---- DIELECTRIC (glass) ----
            float eta = frontFace ? (1.f / mat.ior) : mat.ior;
            float cosI = fmaxf(dot(N, V), 0.f);
            float Fr = fresnelDielectric(cosI, eta);
            specularBounce = true;

            if (rng.f() < Fr) {
                // Reflect
                dir = reflect(F3(0,0,0) - V, N);
                orig = hitPos + N * EPS;
            } else {
                // Refract (Snell)
                f3 Nref = frontFace ? N : F3(0,0,0) - N;
                float etaR = frontFace ? (1.f / mat.ior) : mat.ior;
                float cosi = dot(V, Nref);
                float sin2t = etaR * etaR * (1.f - cosi * cosi);
                if (sin2t > 1.f) {
                    // TIR
                    dir = reflect(F3(0,0,0) - V, Nref);
                    orig = hitPos + Nref * EPS;
                } else {
                    float cost = sqrtf(1.f - sin2t);
                    dir = norm((F3(0,0,0) - V) * etaR + Nref * (etaR * cosi - cost));
                    orig = hitPos - N * EPS; // go through surface
                    // Beer's law absorption (only inside medium)
                    if (!frontFace) {
                        f3 absorp = F3(expf(-mat.r * hit.t * 0.5f),
                                       expf(-mat.g * hit.t * 0.5f),
                                       expf(-mat.b * hit.t * 0.5f));
                        throughput = throughput * absorp;
                    }
                }
            }
            continue; // no NEE for glass

        } else if (mat.type == MAT_METAL) {
            // ---- METAL (GGX specular only) ----
            float alpha = fmaxf(mat.roughness * mat.roughness, 0.001f);
            float a2 = alpha * alpha;
            // Transform V to local space around N
            f3 t, b; buildONB(N, t, b);
            f3 Vlocal = F3(dot(V,t), dot(V,b), dot(V,N));
            f3 Hlocal = sampleGGX_VNDF(Vlocal, alpha, rng);
            f3 H = t * Hlocal.x + b * Hlocal.y + N * Hlocal.z;
            f3 L = reflect(F3(0,0,0) - V, H);
            float NdL = dot(N, L);
            if (NdL <= 0.f) break;
            float NdV = fmaxf(dot(N, V), 0.001f);
            float NdH = fmaxf(dot(N, H), 0.f);
            float HdV = fmaxf(dot(H, V), 0.f);
            f3 F = F_Schlick(HdV, albedo); // metals use albedo as F0
            float G = G2_Smith(NdL, NdV, a2);
            float G1v = G1_Smith(NdV, a2);
            // Weight = F * G2 / G1 (for VNDF sampling)
            throughput = throughput * F * (G / G1v);
            specularBounce = (mat.roughness < 0.15f);
            dir = L;
            orig = hitPos + N * EPS;

        } else if (mat.type == MAT_GLOSSY) {
            // ---- GLOSSY DIELECTRIC (lobe-selection MIS) ----
            float alpha = fmaxf(mat.roughness * mat.roughness, 0.001f);
            float a2 = alpha * alpha;
            float NdV = fmaxf(dot(N, V), 0.001f);
            float F0 = 0.04f;
            float Fr = F_Schlick1(NdV, F0);
            float specProb = fmaxf(Fr, 0.25f);

            if (rng.f() < specProb) {
                // Specular lobe (GGX VNDF)
                f3 t, b; buildONB(N, t, b);
                f3 Vlocal = F3(dot(V,t), dot(V,b), dot(V,N));
                f3 Hlocal = sampleGGX_VNDF(Vlocal, alpha, rng);
                f3 H = t * Hlocal.x + b * Hlocal.y + N * Hlocal.z;
                f3 L = reflect(F3(0,0,0) - V, H);
                float NdL = dot(N, L);
                if (NdL <= 0.f) break;
                float NdH = fmaxf(dot(N, H), 0.f);
                float HdV = fmaxf(dot(H, V), 0.f);
                f3 F = F_Schlick(HdV, F3(F0,F0,F0));
                float G = G2_Smith(NdL, NdV, a2);
                float G1v = G1_Smith(NdV, a2);
                throughput = throughput * F * (G / (G1v * specProb));
                specularBounce = (mat.roughness < 0.15f);
                dir = L; orig = hitPos + N * EPS;
            } else {
                // Diffuse lobe (cosine) with NEE
                specularBounce = false;
                {
                    f3 lPt = sampleLightPt(rng);
                    f3 toLight = lPt - hitPos;
                    float lDist = len(toLight);
                    f3 L = toLight / lDist;
                    float NdL = dot(N, L);
                    f3 lN = F3(c_lNorm[0], c_lNorm[1], c_lNorm[2]);
                    float cosLight = -dot(lN, L);
                    if (NdL > 0.f && cosLight > 0.f) {
                        bool occ = traceShadow(bvh, n4,
                            tv0x,tv0y,tv0z, tv1x,tv1y,tv1z, tv2x,tv2y,tv2z, matIds,
                            hitPos + N * EPS, L, lDist - 2.f*EPS);
                        if (!occ) {
                            float lightPdf = (lDist * lDist) / (cosLight * c_lArea);
                            float brdfPdf = NdL * INV_PI;
                            float w = lightPdf * lightPdf / (lightPdf * lightPdf + brdfPdf * brdfPdf + 1e-10f);
                            f3 Le = F3(c_lEmit[0], c_lEmit[1], c_lEmit[2]);
                            f3 diffBrdf = (F3(1,1,1) - F3(Fr,Fr,Fr)) * albedo * INV_PI * NdL;
                            radiance = radiance + throughput * Le * diffBrdf * (w / (lightPdf * (1.f - specProb)));
                        }
                    }
                }
                f3 L = sampleCosHemi(N, rng);
                float NdL = fmaxf(dot(N, L), 0.f);
                if (NdL <= 0.f) break;
                throughput = throughput * (F3(1,1,1) - F3(Fr,Fr,Fr)) * albedo * INV_PI * (PI / (1.f - specProb));
                dir = L; orig = hitPos + N * EPS;
            }

        } else {
            // ---- DIFFUSE / ROUGH DIELECTRIC ----
            specularBounce = false;
            float NdV = fmaxf(dot(N, V), 0.001f);

            // ---- NEE (Next Event Estimation) with MIS ----
            {
                f3 lPt = sampleLightPt(rng);
                f3 toLight = lPt - hitPos;
                float lDist = len(toLight);
                f3 L = toLight / lDist;
                float NdL = dot(N, L);
                f3 lN = F3(c_lNorm[0], c_lNorm[1], c_lNorm[2]);
                float cosLight = -dot(lN, L);

                if (NdL > 0.f && cosLight > 0.f) {
                    bool occ = traceShadow(bvh, n4,
                        tv0x,tv0y,tv0z, tv1x,tv1y,tv1z, tv2x,tv2y,tv2z, matIds,
                        hitPos + N * EPS, L, lDist - 2.f*EPS);
                    if (!occ) {
                        bool sphOcc = false;
                        for (int si2=0; si2<c_numSpheres; si2++) {
                            if (c_materials[c_spheres[si2].matId].type == MAT_GLASS) continue;
                            float ts = hitSphere(hitPos + N * EPS, L, c_spheres[si2]);
                            if (ts > EPS && ts < lDist - 2.f*EPS) { sphOcc = true; break; }
                        }
                        if (!sphOcc) {
                            float lightPdf = (lDist * lDist) / (cosLight * c_lArea);
                            f3 F_dummy;
                            f3 brdfVal = evalBRDF(N, V, L, albedo, mat.roughness, mat.metallic, F_dummy);
                            float brdfPdf = NdL * INV_PI;
                            float w = lightPdf * lightPdf / (lightPdf * lightPdf + brdfPdf * brdfPdf + 1e-10f);
                            f3 Le = F3(c_lEmit[0], c_lEmit[1], c_lEmit[2]);
                            radiance = radiance + throughput * Le * brdfVal * (w / lightPdf);
                        }
                    }
                }
            }

            // ---- Sample next bounce: cosine-weighted diffuse ----
            f3 L = sampleCosHemi(N, rng);
            float NdL = fmaxf(dot(N, L), 0.f);
            if (NdL <= 0.f) break;

            // BRDF * cos / pdf: for cosine sampling, pdf = cos/π
            // Full Cook-Torrance eval includes specular highlight
            f3 F_dummy;
            f3 brdf = evalBRDF(N, V, L, albedo, mat.roughness, mat.metallic, F_dummy);
            // brdf already has NdL baked in from evalBRDF, pdf = NdL/π
            throughput = throughput * brdf * (PI / (NdL + 1e-7f));

            dir = L;
            orig = hitPos + N * EPS;
        }

        // ---- Russian roulette ----
        if (bounce >= RR_START_BOUNCE) {
            float p = fmaxf(maxcomp(throughput), 0.05f);
            if (rng.f() >= p) break;
            throughput = throughput / p;
        }
    }

    // Firefly clamp
    float mc = maxcomp(radiance);
    if (mc > FIREFLY_CLAMP) radiance = radiance * (FIREFLY_CLAMP / mc);

    aR[pidx] += radiance.x;
    aG[pidx] += radiance.y;
    aB[pidx] += radiance.z;
    // Squared accumulation for variance tracking
    aR2[pidx] += radiance.x * radiance.x;
    aG2[pidx] += radiance.y * radiance.y;
    aB2[pidx] += radiance.z * radiance.z;
    sppMap[pidx]++;
}

// ===================== Edge-Aware Bilateral Denoiser (adaptive-aware) =====================
__global__ void denoise(
    const float* __restrict__ inR, const float* __restrict__ inG, const float* __restrict__ inB,
    float* __restrict__ outR, float* __restrict__ outG, float* __restrict__ outB,
    const GBuf* __restrict__ gbuf, const int* __restrict__ sppMap, int W, int H,
    float sigS, float sigD, float sigN)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;
    int pidx = py * W + px;
    float invSpp = 1.f / fmaxf((float)sppMap[pidx], 1.f);
    float cd = gbuf[pidx].depth;
    float cnx = gbuf[pidx].nx, cny = gbuf[pidx].ny, cnz = gbuf[pidx].nz;
    float sumR=0,sumG=0,sumB=0,sumW=0;
    float iSS = -0.5f/(sigS*sigS), iSD = -0.5f/(sigD*sigD), iSN = -0.5f/(sigN*sigN);
    const int R = 4;
    for (int dy=-R; dy<=R; dy++) for (int dx=-R; dx<=R; dx++) {
        int nx = px+dx, ny = py+dy;
        if (nx<0||nx>=W||ny<0||ny>=H) continue;
        int nidx = ny * W + nx;
        float nInvSpp = 1.f / fmaxf((float)sppMap[nidx], 1.f);
        float sd2 = (float)(dx*dx+dy*dy);
        float dd = cd - gbuf[nidx].depth;
        float nd = 1.f - fmaxf(cnx*gbuf[nidx].nx + cny*gbuf[nidx].ny + cnz*gbuf[nidx].nz, 0.f);
        float w = expf(sd2*iSS + dd*dd*iSD + nd*nd*iSN);
        sumR += inR[nidx]*nInvSpp*w; sumG += inG[nidx]*nInvSpp*w; sumB += inB[nidx]*nInvSpp*w; sumW += w;
    }
    float inv = (sumW > 0) ? 1.f/sumW : 1.f;
    outR[pidx] = sumR*inv; outG[pidx] = sumG*inv; outB[pidx] = sumB*inv;
}

// ===================== ACES Filmic Tonemap =====================
__device__ __forceinline__ float ACESFilm(float x) {
    float a=2.51f, b=0.03f, c=2.43f, d=0.59f, e=0.14f;
    return (x*(a*x+b)) / (x*(c*x+d)+e);
}

__global__ void tonemap(
    const float* __restrict__ inR, const float* __restrict__ inG, const float* __restrict__ inB,
    const int* __restrict__ sppMap,
    uint8_t* __restrict__ out, int W, int H, float exposure, bool divSpp)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;
    int pidx = py * W + px;
    float s = exposure * (divSpp ? (1.f / fmaxf((float)sppMap[pidx], 1.f)) : 1.f);
    float r = ACESFilm(inR[pidx]*s);
    float g = ACESFilm(inG[pidx]*s);
    float b = ACESFilm(inB[pidx]*s);
    // sRGB gamma
    r = powf(fmaxf(r,0.f), 1.f/2.2f);
    g = powf(fmaxf(g,0.f), 1.f/2.2f);
    b = powf(fmaxf(b,0.f), 1.f/2.2f);
    int idx = pidx * 3;
    out[idx]   = (uint8_t)fminf(r*255.f+.5f, 255.f);
    out[idx+1] = (uint8_t)fminf(g*255.f+.5f, 255.f);
    out[idx+2] = (uint8_t)fminf(b*255.f+.5f, 255.f);
}

// ===================== SPP Heatmap =====================
__global__ void visSPP(const int* __restrict__ sppMap, uint8_t* __restrict__ out,
    int W, int H, int maxSpp)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;
    int pidx = py * W + px; int idx = pidx * 3;
    float t = (float)sppMap[pidx] / fmaxf((float)maxSpp, 1.f);
    // Cool-to-hot: blue → green → yellow → red
    float r, g, b;
    if (t < 0.33f) { r=0; g=t*3; b=1-t*3; }
    else if (t < 0.66f) { float u=(t-0.33f)*3; r=u; g=1; b=0; }
    else { float u=(t-0.66f)*3; r=1; g=1-u; b=0; }
    out[idx]   = (uint8_t)(r*255.f); out[idx+1] = (uint8_t)(g*255.f); out[idx+2] = (uint8_t)(b*255.f);
}

// ===================== G-buffer vis =====================
__global__ void visGBuf(const GBuf* __restrict__ g, uint8_t* __restrict__ out,
    int W, int H, int mode) {
    int px=blockIdx.x*blockDim.x+threadIdx.x, py=blockIdx.y*blockDim.y+threadIdx.y;
    if(px>=W||py>=H) return;
    int pidx=py*W+px; int idx=pidx*3;
    if(mode==0){float d=g[pidx].depth; float v=(d<1e20f)?(1.f-logf(1.f+d)/logf(1.f+20.f)):0.f;
        uint8_t c=(uint8_t)(fminf(fmaxf(v,0.f),1.f)*255.f); out[idx]=out[idx+1]=out[idx+2]=c;}
    else if(mode==1){out[idx]=(uint8_t)((g[pidx].nx*.5f+.5f)*255.f);
        out[idx+1]=(uint8_t)((g[pidx].ny*.5f+.5f)*255.f); out[idx+2]=(uint8_t)((g[pidx].nz*.5f+.5f)*255.f);}
    else{out[idx]=(uint8_t)(powf(g[pidx].ar,1.f/2.2f)*255.f);
        out[idx+1]=(uint8_t)(powf(g[pidx].ag,1.f/2.2f)*255.f); out[idx+2]=(uint8_t)(powf(g[pidx].ab,1.f/2.2f)*255.f);}
}

// ===================== PPM Writer =====================
static void writePPM(const char* fn, const uint8_t* rgb, int w, int h) {
    FILE* f = fopen(fn, "wb"); if(!f){printf("  ERR: %s\n",fn); return;}
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for(int y=h-1;y>=0;y--) fwrite(rgb+y*w*3, 1, w*3, f);
    fclose(f); printf("  Saved: %s\n", fn);
}

// ===================== Icosphere Tessellator =====================
static void genIcosphere(std::vector<Tri>& tris, std::vector<int>& matIds,
    float cx, float cy, float cz, float radius, int matId, int subdivisions = 3)
{
    // Start with icosahedron vertices
    const float t = (1.0f + sqrtf(5.0f)) / 2.0f;
    std::vector<float3a> verts = {
        {-1, t,0},{1, t,0},{-1,-t,0},{1,-t,0},
        {0,-1, t},{0, 1, t},{0,-1,-t},{0, 1,-t},
        { t,0,-1},{ t,0, 1},{-t,0,-1},{-t,0, 1}
    };
    struct IdxTri { int a,b,c; };
    std::vector<IdxTri> faces = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}
    };
    // Subdivision
    for (int s = 0; s < subdivisions; s++) {
        std::vector<IdxTri> newFaces;
        std::map<uint64_t, int> midCache;
        auto getMid = [&](int i0, int i1) -> int {
            uint64_t key = (uint64_t)std::min(i0,i1) << 32 | std::max(i0,i1);
            auto it = midCache.find(key);
            if (it != midCache.end()) return it->second;
            float3a& v0 = verts[i0]; float3a& v1 = verts[i1];
            verts.push_back({(v0.x+v1.x)*0.5f, (v0.y+v1.y)*0.5f, (v0.z+v1.z)*0.5f});
            int idx = (int)verts.size()-1;
            midCache[key] = idx;
            return idx;
        };
        for (auto& f : faces) {
            int a=getMid(f.a,f.b), b=getMid(f.b,f.c), c=getMid(f.c,f.a);
            newFaces.push_back({f.a,a,c}); newFaces.push_back({f.b,b,a});
            newFaces.push_back({f.c,c,b}); newFaces.push_back({a,b,c});
        }
        faces = newFaces;
    }
    // Normalize to unit sphere, then scale+translate
    for (auto& v : verts) {
        float l = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
        v.x = v.x/l * radius + cx;
        v.y = v.y/l * radius + cy;
        v.z = v.z/l * radius + cz;
    }
    for (auto& f : faces) {
        tris.push_back({verts[f.a], verts[f.b], verts[f.c]});
        matIds.push_back(matId);
    }
}

// ===================== Enhanced Cornell Box Scene =====================
static void genScene(std::vector<Tri>& tris, std::vector<int>& matIds) {
    auto addQuad = [&](float3a a, float3a b, float3a c, float3a d, int mat) {
        tris.push_back({a,b,c}); matIds.push_back(mat);
        tris.push_back({a,c,d}); matIds.push_back(mat);
    };
    float S = 5.5f;
    // Floor (checker material 0)
    addQuad({0,0,0},{S,0,0},{S,0,S},{0,0,S}, 0);
    // Ceiling
    addQuad({0,S,0},{0,S,S},{S,S,S},{S,S,0}, 6);
    // Back wall
    addQuad({0,0,S},{S,0,S},{S,S,S},{0,S,S}, 6);
    // Left (red)
    addQuad({0,0,0},{0,0,S},{0,S,S},{0,S,0}, 1);
    // Right (green)
    addQuad({S,0,0},{S,S,0},{S,S,S},{S,0,S}, 2);
    // Main light (warm)
    float L0=2.13f,L1=3.43f,LY=S-0.01f,LZ0=2.27f,LZ1=3.32f;
    addQuad({L0,LY,LZ0},{L1,LY,LZ0},{L1,LY,LZ1},{L0,LY,LZ1}, 3);
    // Short box (glossy plastic - mat 7)
    float bx0=1.3f,bx1=2.95f,by1=1.65f,bz0=0.65f,bz1=2.30f;
    addQuad({bx0,by1,bz0},{bx1,by1,bz0},{bx1,by1,bz1},{bx0,by1,bz1}, 7);
    addQuad({bx0,0,bz0},{bx0,0,bz1},{bx0,by1,bz1},{bx0,by1,bz0}, 7);
    addQuad({bx1,0,bz0},{bx1,by1,bz0},{bx1,by1,bz1},{bx1,0,bz1}, 7);
    addQuad({bx0,0,bz0},{bx1,0,bz0},{bx1,by1,bz0},{bx0,by1,bz0}, 7);
    addQuad({bx0,0,bz1},{bx0,by1,bz1},{bx1,by1,bz1},{bx1,0,bz1}, 7);
    // Tall box (white diffuse)
    float tx0=2.85f,tx1=4.48f,ty1=3.30f,tz0=3.00f,tz1=4.60f;
    addQuad({tx0,ty1,tz0},{tx1,ty1,tz0},{tx1,ty1,tz1},{tx0,ty1,tz1}, 6);
    addQuad({tx0,0,tz0},{tx0,0,tz1},{tx0,ty1,tz1},{tx0,ty1,tz0}, 6);
    addQuad({tx1,0,tz0},{tx1,ty1,tz0},{tx1,ty1,tz1},{tx1,0,tz1}, 6);
    addQuad({tx0,0,tz0},{tx1,0,tz0},{tx1,ty1,tz0},{tx0,ty1,tz0}, 6);
    addQuad({tx0,0,tz1},{tx0,ty1,tz1},{tx1,ty1,tz1},{tx1,0,tz1}, 6);

    // Tessellated icosphere (rough copper metal) on floor
    genIcosphere(tris, matIds, 1.2f, 0.8f, 3.8f, 0.8f, 8, 3); // ~1280 tris

    // Small tessellated sphere (glossy plastic) on short box
    genIcosphere(tris, matIds, 2.1f, 2.35f, 1.5f, 0.7f, 7, 2); // ~320 tris
}

// ===================== Main =====================
// ===================== Microsecond Timer =====================
struct uTimer {
    cudaEvent_t a, b; const char* name;
    void init(const char* n) { name=n; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
    void start() { CK(cudaEventRecord(a)); }
    void stop()  { CK(cudaEventRecord(b)); }
    float us()   { float ms; CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms,a,b)); return ms*1000.f; }
    float ms()   { return us()/1000.f; }
    void report(){ printf("    %-38s %8.1f μs  (%6.3f ms)\n", name, us(), ms()); }
    void destroy(){ cudaEventDestroy(a); cudaEventDestroy(b); }
};

// ===== V34 WAVEFRONT KERNELS BELOW =====
// (to be appended)

// ===================== V34 WAVEFRONT KERNELS =====================
// Architecture: generateRays -> [traceExtension -> shadeMaterial -> traceShadow] x bounces
// Each kernel has high SIMT utilization because threads do the SAME operation.
// Queues are compacted between stages to skip terminated rays.

// ─── Kernel 1: Generate Camera Rays (100% utilization) ───
__global__ void wf_generateRays(
    RayState* __restrict__ rays,
    int* __restrict__ activeCount,
    GBuf* __restrict__ gbuf,
    int W, int H, int sampleIdx,
    float camPx, float camPy, float camPz,
    float camFx, float camFy, float camFz,
    float camRx, float camRy, float camRz,
    float camUx, float camUy, float camUz,
    float fovTan, float aperture, float focusDist,
    // For adaptive sampling
    const float* __restrict__ aR, const float* __restrict__ aG, const float* __restrict__ aB,
    const float* __restrict__ aR2, const float* __restrict__ aG2, const float* __restrict__ aB2,
    const int* __restrict__ sppMap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;

    int px = idx % W, py = idx / W;

    // Adaptive: skip converged pixels
    if (sampleIdx >= ADAPTIVE_MIN_SPP) {
        int n = sppMap[idx];
        if (n >= ADAPTIVE_MIN_SPP) {
            float invN = 1.f / n;
            float mr = aR[idx]*invN, mg = aG[idx]*invN, mb = aB[idx]*invN;
            float vr = aR2[idx]*invN - mr*mr, vg = aG2[idx]*invN - mg*mg, vb = aB2[idx]*invN - mb*mb;
            float variance = fmaxf(fmaxf(vr,vg),vb);
            float mean = fmaxf(fmaxf(mr,mg),mb) + 1e-6f;
            float relStddev = sqrtf(fmaxf(variance,0.f)) / mean;
            if (relStddev < ADAPTIVE_THRESH) {
                // Mark terminated — won't get a slot
                return;
            }
        }
    }

    RNG rng;
    rng.s = idx * 1099087573u + sampleIdx * 2654435761u + 1;

    float u = (2.f * (px + rng.f()) / W - 1.f) * fovTan;
    float v = (2.f * (py + rng.f()) / H - 1.f) * fovTan * ((float)H / W);

    f3 fwd = F3(camFx,camFy,camFz);
    f3 right = F3(camRx,camRy,camRz);
    f3 up = F3(camUx,camUy,camUz);
    f3 dir = norm(fwd + right * u + up * v);
    f3 orig = F3(camPx, camPy, camPz);

    // Thin-lens DOF
    if (aperture > 0.f) {
        f3 focusPt = orig + dir * focusDist;
        float r = aperture * sqrtf(rng.f());
        float theta = 2.f * PI * rng.f();
        f3 offset = right * (r * cosf(theta)) + up * (r * sinf(theta));
        orig = orig + offset;
        dir = norm(focusPt - orig);
    }

    // Write to queue slot via atomic counter
    int slot = atomicAdd(activeCount, 1);
    RayState& rs = rays[slot];
    rs.ox = orig.x; rs.oy = orig.y; rs.oz = orig.z;
    rs.dx = dir.x;  rs.dy = dir.y;  rs.dz = dir.z;
    rs.tpR = 1.f; rs.tpG = 1.f; rs.tpB = 1.f;
    rs.radR = 0.f; rs.radG = 0.f; rs.radB = 0.f;
    rs.pixelIdx = idx;
    rs.bounce = 0;
    rs.flags = 1; // specularBounce = true
    rs.rngState = rng.s;
}

// ─── Kernel 2: Trace Extension Rays (BVH traversal only) ───
// Every active thread does the SAME thing: traverse BVH + intersect.
// Much higher SIMT util than megakernel because there's no material branching.
__global__ void wf_traceExtension(
    const RayState* __restrict__ rays,
    HitResult* __restrict__ hits,
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const int* __restrict__ d_numRays)
{
    int numRays = *d_numRays;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    const RayState& rs = rays[idx];
    f3 o = F3(rs.ox, rs.oy, rs.oz);
    f3 d = F3(rs.dx, rs.dy, rs.dz);

    Hit hit = traceScene(bvh, n4, tv0x,tv0y,tv0z, tv1x,tv1y,tv1z, tv2x,tv2y,tv2z, o, d, 1e30f);

    hits[idx].t = hit.t;
    hits[idx].id = hit.id;
    hits[idx].u = hit.u;
    hits[idx].v = hit.v;
}

// ─── Kernel 3: Material Shading + NEE (software SER: sorted by material) ───
// This is where the biggest win comes from. In the megakernel, threads in the
// same warp hit different materials (glass/metal/diffuse) causing massive divergence.
// Here, we pre-sort rays by material type so threads in each warp execute the
// SAME material branch -> near-100% SIMT utilization.
//
// We also generate shadow rays here and write them to the shadow queue.

__global__ void wf_shadeMaterial(
    RayState* __restrict__ rays,
    const HitResult* __restrict__ hits,
    const int* __restrict__ sortedIndices, // rays sorted by material type
    ShadowRay* __restrict__ shadowQueue,
    int* __restrict__ shadowCount,
    RayState* __restrict__ nextRays,    // output queue for next bounce
    int* __restrict__ nextCount,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const int* __restrict__ matIds,
    GBuf* __restrict__ gbuf,
    // Accumulators for direct radiance (terminated rays)
    float* __restrict__ aR, float* __restrict__ aG, float* __restrict__ aB,
    float* __restrict__ aR2, float* __restrict__ aG2, float* __restrict__ aB2,
    int* __restrict__ sppMap,
    const int* __restrict__ d_numRays, int sampleIdx)
{
    int numRays = *d_numRays;
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= numRays) return;

    int rayIdx = sortedIndices[tidx]; // sorted order
    RayState rs = rays[rayIdx];
    HitResult hit = hits[rayIdx];

    // Helper: terminate ray and accumulate radiance directly
    #define TERMINATE_RAY(rad) do { \
        f3 _r = (rad); float _mc = maxcomp(_r); \
        if (_mc > FIREFLY_CLAMP) { float _s = FIREFLY_CLAMP / _mc; _r = _r * _s; } \
        atomicAdd(&aR[rs.pixelIdx], _r.x); \
        atomicAdd(&aG[rs.pixelIdx], _r.y); \
        atomicAdd(&aB[rs.pixelIdx], _r.z); \
        atomicAdd(&aR2[rs.pixelIdx], _r.x*_r.x); \
        atomicAdd(&aG2[rs.pixelIdx], _r.y*_r.y); \
        atomicAdd(&aB2[rs.pixelIdx], _r.z*_r.z); \
        atomicAdd(&sppMap[rs.pixelIdx], 1); \
        return; \
    } while(0)

    // Restore RNG
    RNG rng;
    rng.s = rs.rngState;

    f3 throughput = F3(rs.tpR, rs.tpG, rs.tpB);
    f3 radiance = F3(rs.radR, rs.radG, rs.radB);
    f3 orig = F3(rs.ox, rs.oy, rs.oz);
    f3 dir = F3(rs.dx, rs.dy, rs.dz);
    bool specularBounce = (rs.flags & 1) != 0;
    int bounce = rs.bounce;
    int pidx = rs.pixelIdx;

    // Miss -> accumulate env and terminate
    if (hit.id == -1) {
        radiance = radiance + throughput * envColor(dir);
        TERMINATE_RAY(radiance);
    }

    // Hit geometry
    f3 hitPos = orig + dir * hit.t;
    f3 N; int matId;

    if (hit.id >= 0) {
        N = triNormal(tv0x,tv0y,tv0z, tv1x,tv1y,tv1z, tv2x,tv2y,tv2z, hit.id);
        matId = matIds[hit.id];
    } else {
        int si = -(hit.id + 2);
        Sphere sp = c_spheres[si];
        N = norm(F3(hitPos.x-sp.cx, hitPos.y-sp.cy, hitPos.z-sp.cz));
        matId = sp.matId;
    }

    Material mat = c_materials[matId];
    f3 albedo = F3(mat.r, mat.g, mat.b);
    if (mat.checker) albedo = checkerAlbedo(hitPos, albedo);
    f3 V = norm(F3(0,0,0) - dir);
    bool frontFace = dot(N, V) > 0.f;
    if (!frontFace) N = F3(0,0,0) - N;

    // G-buffer on first hit
    if (bounce == 0 && sampleIdx == 0) {
        gbuf[pidx].depth = hit.t;
        gbuf[pidx].nx = N.x; gbuf[pidx].ny = N.y; gbuf[pidx].nz = N.z;
        gbuf[pidx].ar = mat.r; gbuf[pidx].ag = mat.g; gbuf[pidx].ab = mat.b;
    }

    // Emissive
    f3 emission = F3(mat.er, mat.eg, mat.eb);
    if (maxcomp(emission) > 0.f) {
        if (specularBounce) {
            radiance = radiance + throughput * emission;
        } else {
            float lightDist = hit.t;
            f3 lN = F3(c_lNorm[0], c_lNorm[1], c_lNorm[2]);
            float cosLight = fmaxf(-dot(lN, dir), 0.f);
            float lightPdf = (lightDist * lightDist) / (cosLight * c_lArea + 1e-7f);
            float NdL = fmaxf(dot(N, dir), 0.f);
            float brdfPdf = NdL * INV_PI;
            float w = brdfPdf * brdfPdf / (brdfPdf * brdfPdf + lightPdf * lightPdf + 1e-10f);
            radiance = radiance + throughput * emission * w;
        }
        TERMINATE_RAY(radiance);
    }

    // ── Material dispatch (all threads in warp have SAME material due to sort) ──
    f3 newDir; f3 newOrig; bool newSpecular = false;

    // Helper: immediately enqueue a shadow ray
    #define ENQUEUE_SHADOW(hPos, nrm, sDir, sDist, sContrib) do { \
        int _si = atomicAdd(shadowCount, 1); \
        ShadowRay& _sr = shadowQueue[_si]; \
        _sr.ox = (hPos).x + (nrm).x*EPS; _sr.oy = (hPos).y + (nrm).y*EPS; _sr.oz = (hPos).z + (nrm).z*EPS; \
        _sr.dx = (sDir).x; _sr.dy = (sDir).y; _sr.dz = (sDir).z; \
        _sr.tmax = (sDist) - 2.f*EPS; \
        _sr.contribR = (sContrib).x; _sr.contribG = (sContrib).y; _sr.contribB = (sContrib).z; \
        _sr.pixelIdx = pidx; \
    } while(0)

    if (mat.type == MAT_GLASS) {
        float eta = frontFace ? (1.f / mat.ior) : mat.ior;
        float cosI = fmaxf(dot(N, V), 0.f);
        float Fr = fresnelDielectric(cosI, eta);
        newSpecular = true;
        if (rng.f() < Fr) {
            newDir = reflect(F3(0,0,0) - V, N);
            newOrig = hitPos + N * EPS;
        } else {
            f3 Nref = frontFace ? N : F3(0,0,0) - N;
            float etaR = frontFace ? (1.f / mat.ior) : mat.ior;
            float cosi = dot(V, Nref);
            float sin2t = etaR * etaR * (1.f - cosi * cosi);
            if (sin2t > 1.f) {
                newDir = reflect(F3(0,0,0) - V, Nref);
                newOrig = hitPos + Nref * EPS;
            } else {
                float cost = sqrtf(1.f - sin2t);
                newDir = norm((F3(0,0,0) - V) * etaR + Nref * (etaR * cosi - cost));
                newOrig = hitPos - N * EPS;
                if (!frontFace) {
                    f3 absorp = F3(expf(-mat.r*hit.t*0.5f), expf(-mat.g*hit.t*0.5f), expf(-mat.b*hit.t*0.5f));
                    throughput = throughput * absorp;
                }
            }
        }
    } else if (mat.type == MAT_METAL) {
        float alpha = fmaxf(mat.roughness * mat.roughness, 0.001f);
        float a2 = alpha * alpha;
        f3 t, b; buildONB(N, t, b);
        f3 Vlocal = F3(dot(V,t), dot(V,b), dot(V,N));
        f3 Hlocal = sampleGGX_VNDF(Vlocal, alpha, rng);
        f3 H = t * Hlocal.x + b * Hlocal.y + N * Hlocal.z;
        f3 L = reflect(F3(0,0,0) - V, H);
        float NdL = dot(N, L);
        if (NdL <= 0.f) { TERMINATE_RAY(radiance); }
        float NdV = fmaxf(dot(N, V), 0.001f);
        float HdV = fmaxf(dot(H, V), 0.f);
        f3 F = F_Schlick(HdV, albedo);
        float G = G2_Smith(NdL, NdV, a2);
        float G1v = G1_Smith(NdV, a2);
        throughput = throughput * F * (G / G1v);
        newSpecular = (mat.roughness < 0.15f);
        newDir = L;
        newOrig = hitPos + N * EPS;
    } else if (mat.type == MAT_GLOSSY) {
        float alpha = fmaxf(mat.roughness * mat.roughness, 0.001f);
        float a2 = alpha * alpha;
        float NdV = fmaxf(dot(N, V), 0.001f);
        float F0 = 0.04f;
        float Fr = F_Schlick1(NdV, F0);
        float specProb = fmaxf(Fr, 0.25f);
        if (rng.f() < specProb) {
            f3 t, b; buildONB(N, t, b);
            f3 Vlocal = F3(dot(V,t), dot(V,b), dot(V,N));
            f3 Hlocal = sampleGGX_VNDF(Vlocal, alpha, rng);
            f3 H = t * Hlocal.x + b * Hlocal.y + N * Hlocal.z;
            f3 L = reflect(F3(0,0,0) - V, H);
            float NdL = dot(N, L);
            if (NdL <= 0.f) { TERMINATE_RAY(radiance); }
            float HdV = fmaxf(dot(H, V), 0.f);
            f3 Fs = F_Schlick(HdV, F3(F0,F0,F0));
            float G = G2_Smith(NdL, NdV, a2);
            float G1v = G1_Smith(NdV, a2);
            throughput = throughput * Fs * (G / (G1v * specProb));
            newSpecular = (mat.roughness < 0.15f);
            newDir = L; newOrig = hitPos + N * EPS;
        } else {
            newSpecular = false;
            // NEE for diffuse lobe -> generate shadow ray
            f3 lPt = sampleLightPt(rng);
            f3 toLight = lPt - hitPos;
            float lDist = len(toLight);
            f3 L = toLight / lDist;
            float NdL = dot(N, L);
            f3 lN = F3(c_lNorm[0], c_lNorm[1], c_lNorm[2]);
            float cosLight = -dot(lN, L);
            if (NdL > 0.f && cosLight > 0.f) {
                float lightPdf = (lDist * lDist) / (cosLight * c_lArea);
                float brdfPdf = NdL * INV_PI;
                float w = lightPdf * lightPdf / (lightPdf * lightPdf + brdfPdf * brdfPdf + 1e-10f);
                f3 Le = F3(c_lEmit[0], c_lEmit[1], c_lEmit[2]);
                f3 diffBrdf = (F3(1,1,1) - F3(Fr,Fr,Fr)) * albedo * INV_PI * NdL;
                f3 sc = throughput * Le * diffBrdf * (w / (lightPdf * (1.f - specProb)));
                ENQUEUE_SHADOW(hitPos, N, L, lDist, sc);
            }
            // Bounce direction: cosine hemisphere
            f3 Lb = sampleCosHemi(N, rng);
            float NdLb = fmaxf(dot(N, Lb), 0.f);
            if (NdLb <= 0.f) { TERMINATE_RAY(radiance); }
            throughput = throughput * (F3(1,1,1) - F3(Fr,Fr,Fr)) * albedo * INV_PI * (PI / (1.f - specProb));
            newDir = Lb; newOrig = hitPos + N * EPS;
        }
    } else {
        // DIFFUSE / rough dielectric
        newSpecular = false;
        float NdV = fmaxf(dot(N, V), 0.001f);
        // NEE
        f3 lPt = sampleLightPt(rng);
        f3 toLight = lPt - hitPos;
        float lDist = len(toLight);
        f3 L = toLight / lDist;
        float NdL = dot(N, L);
        f3 lN = F3(c_lNorm[0], c_lNorm[1], c_lNorm[2]);
        float cosLight = -dot(lN, L);
        if (NdL > 0.f && cosLight > 0.f) {
            float lightPdf = (lDist * lDist) / (cosLight * c_lArea);
            f3 F_dummy;
            f3 brdfVal = evalBRDF(N, V, L, albedo, mat.roughness, mat.metallic, F_dummy);
            float brdfPdf = NdL * INV_PI;
            float w = lightPdf * lightPdf / (lightPdf * lightPdf + brdfPdf * brdfPdf + 1e-10f);
            f3 Le = F3(c_lEmit[0], c_lEmit[1], c_lEmit[2]);
            f3 sc = throughput * Le * brdfVal * (w / lightPdf);
            ENQUEUE_SHADOW(hitPos, N, L, lDist, sc);
        }
        // Bounce: cosine hemisphere
        f3 Lb = sampleCosHemi(N, rng);
        float NdLb = fmaxf(dot(N, Lb), 0.f);
        if (NdLb <= 0.f) { TERMINATE_RAY(radiance); }
        f3 F_dummy2;
        f3 brdf = evalBRDF(N, V, Lb, albedo, mat.roughness, mat.metallic, F_dummy2);
        throughput = throughput * brdf * (PI / (NdLb + 1e-7f));
        newDir = Lb; newOrig = hitPos + N * EPS;
    }

    // Russian roulette
    if (bounce >= RR_START_BOUNCE) {
        float p = fmaxf(maxcomp(throughput), 0.05f);
        if (rng.f() >= p) {
            TERMINATE_RAY(radiance);
        }
        throughput = throughput / p;
    }

    // Enqueue next bounce ray
    if (bounce + 1 < MAX_BOUNCES) {
        int ni = atomicAdd(nextCount, 1);
        RayState& nr = nextRays[ni];
        nr.ox = newOrig.x; nr.oy = newOrig.y; nr.oz = newOrig.z;
        nr.dx = newDir.x;  nr.dy = newDir.y;  nr.dz = newDir.z;
        nr.tpR = throughput.x; nr.tpG = throughput.y; nr.tpB = throughput.z;
        nr.radR = radiance.x;  nr.radG = radiance.y;  nr.radB = radiance.z;
        nr.pixelIdx = pidx;
        nr.bounce = bounce + 1;
        nr.flags = newSpecular ? 1 : 0;
        nr.rngState = rng.s;
    } else {
        TERMINATE_RAY(radiance);
    }
    #undef TERMINATE_RAY
    #undef ENQUEUE_SHADOW
}

// --- Kernel 4: Trace Shadow Rays (any-hit BVH traversal) ---
__global__ void wf_traceShadow(
    const ShadowRay* __restrict__ shadowQueue,
    float* __restrict__ aR, float* __restrict__ aG, float* __restrict__ aB,
    const int4* __restrict__ bvh, int n4,
    const float* __restrict__ tv0x, const float* __restrict__ tv0y, const float* __restrict__ tv0z,
    const float* __restrict__ tv1x, const float* __restrict__ tv1y, const float* __restrict__ tv1z,
    const float* __restrict__ tv2x, const float* __restrict__ tv2y, const float* __restrict__ tv2z,
    const int* __restrict__ matIds,
    const int* __restrict__ d_numShadow)
{
    int numShadow = *d_numShadow;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numShadow) return;
    const ShadowRay& sr = shadowQueue[idx];
    f3 o = F3(sr.ox, sr.oy, sr.oz);
    f3 d = F3(sr.dx, sr.dy, sr.dz);
    bool occluded = traceShadow(bvh, n4, tv0x,tv0y,tv0z, tv1x,tv1y,tv1z, tv2x,tv2y,tv2z, matIds, o, d, sr.tmax);
    if (!occluded) {
        atomicAdd(&aR[sr.pixelIdx], sr.contribR);
        atomicAdd(&aG[sr.pixelIdx], sr.contribG);
        atomicAdd(&aB[sr.pixelIdx], sr.contribB);
    }
}

// --- Kernel 5: Finalize ---
__global__ void wf_finalize(
    const RayState* __restrict__ rays,
    float* __restrict__ aR, float* __restrict__ aG, float* __restrict__ aB,
    const int* __restrict__ d_numRays)
{
    int numRays = *d_numRays;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;
    const RayState& rs = rays[idx];
    if (rs.flags & 2) {
        atomicAdd(&aR[rs.pixelIdx], rs.radR);
        atomicAdd(&aG[rs.pixelIdx], rs.radG);
        atomicAdd(&aB[rs.pixelIdx], rs.radB);
    }
}

// --- Material Sorting Kernels ---
__global__ void wf_classifyMaterial(
    const RayState* __restrict__ rays, const HitResult* __restrict__ hits,
    const int* __restrict__ matIds, int* __restrict__ materialBins,
    int* __restrict__ rayMatType, const int* __restrict__ d_numRays)
{
    int numRays = *d_numRays;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;
    int hitId = hits[idx].id;
    int matType = 0;
    if (hitId == -1) { matType = -1; }
    else if (hitId >= 0) { matType = c_materials[matIds[hitId]].type; }
    else { int si = -(hitId + 2); matType = c_materials[c_spheres[si].matId].type; }
    rayMatType[idx] = matType;
    int bin = (matType < 0) ? 4 : matType;
    atomicAdd(&materialBins[bin], 1);
}

__global__ void wf_sortByMaterial(
    const int* __restrict__ rayMatType, const int* __restrict__ prefixSums,
    int* __restrict__ sortedIndices, int* __restrict__ binCounters, const int* __restrict__ d_numRays)
{
    int numRays = *d_numRays;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;
    int matType = rayMatType[idx];
    int bin = (matType < 0) ? 4 : matType;
    int offset = prefixSums[bin] + atomicAdd(&binCounters[bin], 1);
    sortedIndices[offset] = idx;
}


// --- GPU-side prefix sum for 5 bins (single thread) ---
__global__ void wf_prefixSum5(const int* __restrict__ bins, int* __restrict__ prefix) {
    prefix[0] = 0;
    prefix[1] = bins[0];
    prefix[2] = bins[0] + bins[1];
    prefix[3] = bins[0] + bins[1] + bins[2];
    prefix[4] = bins[0] + bins[1] + bins[2] + bins[3];
}

// --- Reset counters (single thread) ---
__global__ void wf_resetCounters(int* c1, int* c2, int* bins, int* binCounters) {
    *c1 = 0; *c2 = 0;
    for (int i = 0; i < 5; i++) { bins[i] = 0; binCounters[i] = 0; }
}

// --- Swap ray buffer pointers (single thread, writes to pointer storage) ---
// Not needed — we use double-buffering indices on device
// ===================== Scene Builders =====================
static void addQuad(std::vector<Tri>& tris, std::vector<int>& mats,
    f3 a, f3 b, f3 c, f3 d, int m) {
    auto cvt = [](f3 v) -> float3a { return {v.x, v.y, v.z}; };
    tris.push_back({cvt(a), cvt(b), cvt(c)}); mats.push_back(m);
    tris.push_back({cvt(a), cvt(c), cvt(d)}); mats.push_back(m);
}

static void addIcosphere(std::vector<Tri>& tris, std::vector<int>& mats,
    f3 ctr, float r, int subdiv, int matId) {
    float t = (1.f + sqrtf(5.f)) / 2.f;
    std::vector<f3> v = {{-1,t,0},{1,t,0},{-1,-t,0},{1,-t,0},{0,-1,t},{0,1,t},{0,-1,-t},{0,1,-t},{t,0,-1},{t,0,1},{-t,0,-1},{-t,0,1}};
    for (auto& p : v) { float l = sqrtf(p.x*p.x+p.y*p.y+p.z*p.z); p.x/=l; p.y/=l; p.z/=l; }
    struct Tri3 { int a,b,c; };
    std::vector<Tri3> faces = {{0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},{1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},{3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},{4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}};
    for (int s = 0; s < subdiv; s++) {
        std::vector<Tri3> nf; std::map<long long,int> em;
        auto mid = [&](int a, int b) -> int {
            long long k = a < b ? (long long)a*100000+b : (long long)b*100000+a;
            auto it = em.find(k); if (it != em.end()) return it->second;
            f3 m = {(v[a].x+v[b].x)*.5f,(v[a].y+v[b].y)*.5f,(v[a].z+v[b].z)*.5f};
            float l = sqrtf(m.x*m.x+m.y*m.y+m.z*m.z); m.x/=l; m.y/=l; m.z/=l;
            int idx = (int)v.size(); v.push_back(m); em[k]=idx; return idx;
        };
        for (auto& f : faces) {
            int ab=mid(f.a,f.b), bc=mid(f.b,f.c), ca=mid(f.c,f.a);
            nf.push_back({f.a,ab,ca}); nf.push_back({f.b,bc,ab}); nf.push_back({f.c,ca,bc}); nf.push_back({ab,bc,ca});
        }
        faces = nf;
    }
    for (auto& f : faces) {
        float3a p0={ctr.x+v[f.a].x*r,ctr.y+v[f.a].y*r,ctr.z+v[f.a].z*r};
        float3a p1={ctr.x+v[f.b].x*r,ctr.y+v[f.b].y*r,ctr.z+v[f.b].z*r};
        float3a p2={ctr.x+v[f.c].x*r,ctr.y+v[f.c].y*r,ctr.z+v[f.c].z*r};
        tris.push_back({p0,p1,p2}); mats.push_back(matId);
    }
}

// ===================== MAIN =====================
int main(int argc, char** argv) {
    int maxSpp = (argc > 1) ? atoi(argv[1]) : 4;
    int W = (argc > 2) ? atoi(argv[2]) : 512;
    int H = W;
    int mode = (argc > 3) ? atoi(argv[3]) : 0;
    int nPx = W * H;

    printf("\n======================================================================\n");
    printf("  V34 -- Wavefront PBR Path Tracer (NVIDIA Research algorithms)\n");
    printf("  %dx%d | max %d spp | mode=%d | %s\n", W, H, maxSpp, mode,
           mode == 0 ? "WAVEFRONT (Laine 2013 + software SER)" : "MEGAKERNEL (v33 baseline)");
    printf("======================================================================\n\n");

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU: %s | SMs: %d\n\n", prop.name, prop.multiProcessorCount);

    std::vector<Tri> tris; std::vector<int> mats;
    float S = 3.f;
    addQuad(tris,mats,{-S,0,-S},{S,0,-S},{S,0,S},{-S,0,S},0);
    addQuad(tris,mats,{-S,2*S,-S},{-S,2*S,S},{S,2*S,S},{S,2*S,-S},0);
    addQuad(tris,mats,{-S,0,-S},{-S,2*S,-S},{S,2*S,-S},{S,0,-S},0);
    addQuad(tris,mats,{-S,0,-S},{-S,0,S},{-S,2*S,S},{-S,2*S,-S},1);
    addQuad(tris,mats,{S,0,-S},{S,2*S,-S},{S,2*S,S},{S,0,S},2);
    float lS=0.8f;
    addQuad(tris,mats,{-lS,2*S-0.01f,-lS},{lS,2*S-0.01f,-lS},{lS,2*S-0.01f,lS},{-lS,2*S-0.01f,lS},3);
    float tw=1.4f,td=0.8f,th=0.9f,tl=0.04f;
    addQuad(tris,mats,{-tw/2,th,-td/2},{tw/2,th,-td/2},{tw/2,th,td/2},{-tw/2,th,td/2},4);
    float lx[]={-tw/2+.06f,tw/2-.06f,tw/2-.06f,-tw/2+.06f},lz[]={-td/2+.06f,-td/2+.06f,td/2-.06f,td/2-.06f};
    for(int i=0;i<4;i++){float x0=lx[i]-.03f,x1=lx[i]+.03f,z0=lz[i]-.03f,z1=lz[i]+.03f;
        addQuad(tris,mats,{x0,0,z0},{x1,0,z0},{x1,th-tl,z0},{x0,th-tl,z0},4);
        addQuad(tris,mats,{x0,0,z1},{x0,th-tl,z1},{x1,th-tl,z1},{x1,0,z1},4);
        addQuad(tris,mats,{x0,0,z0},{x0,th-tl,z0},{x0,th-tl,z1},{x0,0,z1},4);
        addQuad(tris,mats,{x1,0,z0},{x1,0,z1},{x1,th-tl,z1},{x1,th-tl,z0},4);}
    addIcosphere(tris,mats,{-0.6f,th+0.4f,0.f},0.35f,3,5);
    addIcosphere(tris,mats,{0.6f,th+0.4f,0.f},0.35f,3,6);

    // Scale scene to ~100K tris for meaningful wavefront testing
    for (int row = 0; row < 10; row++)
        for (int col = 0; col < 10; col++) {
            if (row == 5 && col == 5) continue;  // skip center
            float x = -2.5f + col * 0.5f;
            float z = -2.5f + row * 0.5f;
            addIcosphere(tris, mats, {x, 0.2f, z}, 0.18f, 4, (row+col)%2 ? 5 : 6);
        }

    int nTris=(int)tris.size();
    printf("  Triangles: %d | Spheres: 2\n",nTris);

    Material materials[]={
        {0.8f,0.8f,0.8f, 0.5f,0.f,1.5f, 0,0,0, MAT_DIFFUSE,1},
        {0.65f,0.05f,0.05f, 0.3f,0.f,1.5f, 0,0,0, MAT_DIFFUSE,0},
        {0.12f,0.45f,0.12f, 0.3f,0.f,1.5f, 0,0,0, MAT_DIFFUSE,0},
        {0,0,0, 0,0,0, 12,12,10, MAT_DIFFUSE,0},
        {0.6f,0.45f,0.3f, 0.2f,0.f,1.5f, 0,0,0, MAT_GLOSSY,0},
        {0.95f,0.64f,0.54f, 0.08f,1.f,1.5f, 0,0,0, MAT_METAL,0},
        {0.9f,0.95f,0.9f, 0.f,0.f,1.52f, 0,0,0, MAT_GLASS,0},
    };
    CK(cudaMemcpyToSymbol(c_materials,materials,sizeof(materials)));
    Sphere spheres[]={{-0.6f,th+0.4f,0.f,0.35f,5},{0.6f,th+0.4f,0.f,0.35f,6}};
    int nSph=2;
    CK(cudaMemcpyToSymbol(c_spheres,spheres,nSph*sizeof(Sphere)));
    CK(cudaMemcpyToSymbol(c_numSpheres,&nSph,sizeof(int)));
    float h_corner[]={-lS,2*S-0.01f,-lS},h_e1[]={2*lS,0,0},h_e2[]={0,0,2*lS},h_lnorm[]={0,-1,0};
    float lArea=4.f*lS*lS; float lEmit[]={12,12,10};
    CK(cudaMemcpyToSymbol(c_lCorner,h_corner,12));CK(cudaMemcpyToSymbol(c_lE1,h_e1,12));
    CK(cudaMemcpyToSymbol(c_lE2,h_e2,12));CK(cudaMemcpyToSymbol(c_lNorm,h_lnorm,12));
    CK(cudaMemcpyToSymbol(c_lArea,&lArea,4));CK(cudaMemcpyToSymbol(c_lEmit,lEmit,12));

    BVHBuild b2; b2.build(tris.data(),mats.data(),nTris);
    std::vector<BVH4Node> bvh4(b2.nodes.size()*2);
    int n4=0; collapseB4(b2,0,bvh4.data(),n4);
    printf("  BVH4: %d nodes\n",n4);

    std::vector<float> hv0x(nTris),hv0y(nTris),hv0z(nTris),hv1x(nTris),hv1y(nTris),hv1z(nTris),hv2x(nTris),hv2y(nTris),hv2z(nTris);
    std::vector<int> hMatIds(nTris);
    for(int i=0;i<nTris;i++){
        hv0x[i]=tris[i].v0.x;hv0y[i]=tris[i].v0.y;hv0z[i]=tris[i].v0.z;
        hv1x[i]=tris[i].v1.x;hv1y[i]=tris[i].v1.y;hv1z[i]=tris[i].v1.z;
        hv2x[i]=tris[i].v2.x;hv2y[i]=tris[i].v2.y;hv2z[i]=tris[i].v2.z;
        hMatIds[i]=mats[i];}
    float *d_v0x,*d_v0y,*d_v0z,*d_v1x,*d_v1y,*d_v1z,*d_v2x,*d_v2y,*d_v2z;
    int *d_matIds; int4 *d_bvh4;
    #define AU(d,h,n) CK(cudaMalloc(&d,(n)*sizeof(*(d))));CK(cudaMemcpy(d,(h).data(),(n)*sizeof(*(d)),cudaMemcpyHostToDevice))
    AU(d_v0x,hv0x,nTris);AU(d_v0y,hv0y,nTris);AU(d_v0z,hv0z,nTris);
    AU(d_v1x,hv1x,nTris);AU(d_v1y,hv1y,nTris);AU(d_v1z,hv1z,nTris);
    AU(d_v2x,hv2x,nTris);AU(d_v2y,hv2y,nTris);AU(d_v2z,hv2z,nTris);
    AU(d_matIds,hMatIds,nTris);
    #undef AU
    CK(cudaMalloc(&d_bvh4,n4*sizeof(int4)*4));
    CK(cudaMemcpy(d_bvh4,bvh4.data(),n4*sizeof(int4)*4,cudaMemcpyHostToDevice));

    float *d_aR,*d_aG,*d_aB,*d_aR2,*d_aG2,*d_aB2; int *d_sppMap; GBuf *d_gbuf;
    CK(cudaMalloc(&d_aR,nPx*4));CK(cudaMalloc(&d_aG,nPx*4));CK(cudaMalloc(&d_aB,nPx*4));
    CK(cudaMalloc(&d_aR2,nPx*4));CK(cudaMalloc(&d_aG2,nPx*4));CK(cudaMalloc(&d_aB2,nPx*4));
    CK(cudaMalloc(&d_sppMap,nPx*4));CK(cudaMalloc(&d_gbuf,nPx*sizeof(GBuf)));
    CK(cudaMemset(d_aR,0,nPx*4));CK(cudaMemset(d_aG,0,nPx*4));CK(cudaMemset(d_aB,0,nPx*4));
    CK(cudaMemset(d_aR2,0,nPx*4));CK(cudaMemset(d_aG2,0,nPx*4));CK(cudaMemset(d_aB2,0,nPx*4));
    CK(cudaMemset(d_sppMap,0,nPx*4));
    unsigned char* h_fb; CK(cudaMallocHost(&h_fb,nPx*3));
    unsigned char* d_fb; CK(cudaMalloc(&d_fb,nPx*3));

    f3 camP={0.f,2.8f,8.5f},camT={0.f,2.5f,0.f};
    f3 fwd=norm(camT-camP); f3 worldUp={0,1,0};
    f3 right=norm(cross(fwd,worldUp)); f3 up=cross(right,fwd);
    float fovTan=tanf(40.f*PI/180.f),aperture=0.02f,focusDist=len(camT-camP);

    RayState *d_rays0,*d_rays1; HitResult *d_hits; ShadowRay *d_shadow;
    int *d_activeCount,*d_shadowCount,*d_nextCount;
    int *d_materialBins,*d_rayMatType,*d_sortedIndices,*d_binCounters,*d_prefixSums;
    CK(cudaMalloc(&d_rays0,nPx*sizeof(RayState)));CK(cudaMalloc(&d_rays1,nPx*sizeof(RayState)));
    CK(cudaMalloc(&d_hits,nPx*sizeof(HitResult)));CK(cudaMalloc(&d_shadow,nPx*sizeof(ShadowRay)));
    CK(cudaMalloc(&d_activeCount,4));CK(cudaMalloc(&d_shadowCount,4));CK(cudaMalloc(&d_nextCount,4));
    CK(cudaMalloc(&d_materialBins,20));CK(cudaMalloc(&d_rayMatType,nPx*4));
    CK(cudaMalloc(&d_sortedIndices,nPx*4));CK(cudaMalloc(&d_binCounters,20));CK(cudaMalloc(&d_prefixSums,20));

    printf("\n  Wavefront buffers: %.1f MB\n",
        (2.0*nPx*sizeof(RayState)+nPx*sizeof(HitResult)+nPx*sizeof(ShadowRay)+nPx*12)/(1024.0*1024.0));

    // No host-side counters needed — all counts stay on GPU

    cudaEvent_t tStart,tEnd; CK(cudaEventCreate(&tStart));CK(cudaEventCreate(&tEnd));
    CK(cudaEventRecord(tStart));
    // Stats tracked on GPU only — no CPU round-trips

    for(int spp=0;spp<maxSpp;spp++){
        CK(cudaMemset(d_activeCount,0,4));
        int blk=256,maxGrd=(nPx+blk-1)/blk;
        wf_generateRays<<<maxGrd,blk>>>(d_rays0,d_activeCount,d_gbuf,W,H,spp,
            camP.x,camP.y,camP.z,fwd.x,fwd.y,fwd.z,
            right.x,right.y,right.z,up.x,up.y,up.z,
            fovTan,aperture,focusDist,d_aR,d_aG,d_aB,d_aR2,d_aG2,d_aB2,d_sppMap);

        // Fully GPU-native bounce loop — ZERO CPU round-trips
        // All kernels read ray counts from device pointers, grid = max size, threads self-gate
        int* curCount = d_activeCount;
        int* nxtCount = d_nextCount;
        RayState* curRays=d_rays0; RayState* nxtRays=d_rays1;

        for(int bounce=0;bounce<MAX_BOUNCES;bounce++){
            // Zero counters via async memset (no kernel launch overhead)
            CK(cudaMemsetAsync(d_shadowCount,0,4));
            CK(cudaMemsetAsync(nxtCount,0,4));
            CK(cudaMemsetAsync(d_materialBins,0,20));
            CK(cudaMemsetAsync(d_binCounters,0,20));

            // Trace extension rays (reads curCount from device memory)
            wf_traceExtension<<<maxGrd,blk>>>(curRays,d_hits,d_bvh4,n4,
                d_v0x,d_v0y,d_v0z,d_v1x,d_v1y,d_v1z,d_v2x,d_v2y,d_v2z,curCount);

            // Classify materials
            wf_classifyMaterial<<<maxGrd,blk>>>(curRays,d_hits,d_matIds,d_materialBins,d_rayMatType,curCount);

            // GPU prefix sum (single thread, 5 elements)
            wf_prefixSum5<<<1,1>>>(d_materialBins, d_prefixSums);

            // Sort by material
            wf_sortByMaterial<<<maxGrd,blk>>>(d_rayMatType,d_prefixSums,d_sortedIndices,d_binCounters,curCount);

            // Shade (writes shadow rays + next bounce rays, reads curCount)
            wf_shadeMaterial<<<maxGrd,blk>>>(curRays,d_hits,d_sortedIndices,
                d_shadow,d_shadowCount,nxtRays,nxtCount,
                d_v0x,d_v0y,d_v0z,d_v1x,d_v1y,d_v1z,d_v2x,d_v2y,d_v2z,
                d_matIds,d_gbuf,d_aR,d_aG,d_aB,d_aR2,d_aG2,d_aB2,d_sppMap,curCount,spp);

            // Trace shadow rays (reads d_shadowCount from device memory)
            wf_traceShadow<<<maxGrd,blk>>>(d_shadow,d_aR,d_aG,d_aB,d_bvh4,n4,
                d_v0x,d_v0y,d_v0z,d_v1x,d_v1y,d_v1z,d_v2x,d_v2y,d_v2z,d_matIds,d_shadowCount);

            // Swap buffers
            RayState* tmpR=curRays; curRays=nxtRays; nxtRays=tmpR;
            int* tmpC=curCount; curCount=nxtCount; nxtCount=tmpC;
        }
    }
    CK(cudaDeviceSynchronize());CK(cudaEventRecord(tEnd));CK(cudaEventSynchronize(tEnd));
    float totalMs; CK(cudaEventElapsedTime(&totalMs,tStart,tEnd));

    printf("\n  --- Wavefront Stats ---\n");
    printf("  Trace time:           %.3f ms (%d spp, %d bounces)\n", totalMs, maxSpp, MAX_BOUNCES);
    printf("  Per-sample:           %.3f ms\n", totalMs / maxSpp);
    printf("  GPU-native:           ZERO CPU round-trips in hot loop\n");

    // Per-kernel timing (1 spp, 1 bounce only for profiling)
    {
        CK(cudaMemset(d_activeCount,0,4));
        int blk=256,maxGrd=(nPx+blk-1)/blk;
        uTimer tg,tt,tc,tp,ts,tsh,tsw;
        tg.init("generateRays"); tt.init("traceExtension"); tc.init("classifyMaterial");
        tp.init("prefixSum5"); ts.init("sortByMaterial"); tsh.init("shadeMaterial"); tsw.init("traceShadow");

        tg.start();
        wf_generateRays<<<maxGrd,blk>>>(d_rays0,d_activeCount,d_gbuf,W,H,99,
            camP.x,camP.y,camP.z,fwd.x,fwd.y,fwd.z,
            right.x,right.y,right.z,up.x,up.y,up.z,
            fovTan,aperture,focusDist,d_aR,d_aG,d_aB,d_aR2,d_aG2,d_aB2,d_sppMap);
        tg.stop();

        CK(cudaMemsetAsync(d_shadowCount,0,4));
        CK(cudaMemsetAsync(d_nextCount,0,4));
        CK(cudaMemsetAsync(d_materialBins,0,20));
        CK(cudaMemsetAsync(d_binCounters,0,20));

        tt.start();
        wf_traceExtension<<<maxGrd,blk>>>(d_rays0,d_hits,d_bvh4,n4,
            d_v0x,d_v0y,d_v0z,d_v1x,d_v1y,d_v1z,d_v2x,d_v2y,d_v2z,d_activeCount);
        tt.stop();

        tc.start();
        wf_classifyMaterial<<<maxGrd,blk>>>(d_rays0,d_hits,d_matIds,d_materialBins,d_rayMatType,d_activeCount);
        tc.stop();

        tp.start();
        wf_prefixSum5<<<1,1>>>(d_materialBins, d_prefixSums);
        tp.stop();

        ts.start();
        wf_sortByMaterial<<<maxGrd,blk>>>(d_rayMatType,d_prefixSums,d_sortedIndices,d_binCounters,d_activeCount);
        ts.stop();

        tsh.start();
        wf_shadeMaterial<<<maxGrd,blk>>>(d_rays0,d_hits,d_sortedIndices,
            d_shadow,d_shadowCount,d_rays1,d_nextCount,
            d_v0x,d_v0y,d_v0z,d_v1x,d_v1y,d_v1z,d_v2x,d_v2y,d_v2z,
            d_matIds,d_gbuf,d_aR,d_aG,d_aB,d_aR2,d_aG2,d_aB2,d_sppMap,d_activeCount,99);
        tsh.stop();

        tsw.start();
        wf_traceShadow<<<maxGrd,blk>>>(d_shadow,d_aR,d_aG,d_aB,d_bvh4,n4,
            d_v0x,d_v0y,d_v0z,d_v1x,d_v1y,d_v1z,d_v2x,d_v2y,d_v2z,d_matIds,d_shadowCount);
        tsw.stop();

        printf("\n  --- Per-Kernel Profile (1 bounce, 262K rays) ---\n");
        tg.report(); tt.report(); tc.report(); tp.report(); ts.report(); tsh.report(); tsw.report();
        float total_kern = tg.us()+tt.us()+tc.us()+tp.us()+ts.us()+tsh.us()+tsw.us();
        printf("    %-38s %8.1f μs  (%6.3f ms)\n", "TOTAL kernel time", total_kern, total_kern/1000.f);
        printf("    %-38s %8.1f μs  (%6.3f ms)\n", "Overhead per bounce (total-trace-shade)",
            total_kern - tt.us() - tsh.us() - tsw.us(), (total_kern - tt.us() - tsh.us() - tsw.us())/1000.f);
        tg.destroy(); tt.destroy(); tc.destroy(); tp.destroy(); ts.destroy(); tsh.destroy(); tsw.destroy();
    }

    dim3 tblk(16,16),tgrd((W+15)/16,(H+15)/16);
    tonemap<<<tgrd,tblk>>>(d_aR,d_aG,d_aB,d_sppMap,d_fb,W,H,1.0f,false);
    CK(cudaMemcpy(h_fb,d_fb,nPx*3,cudaMemcpyDeviceToHost));
    writePPM("/workspaces/codespace/VK_RT/v34_wavefront.ppm",h_fb,W,H);
    printf("  Saved: v34_wavefront.ppm\n");

    // Megakernel comparison
    CK(cudaMemset(d_aR,0,nPx*4));CK(cudaMemset(d_aG,0,nPx*4));CK(cudaMemset(d_aB,0,nPx*4));
    CK(cudaMemset(d_aR2,0,nPx*4));CK(cudaMemset(d_aG2,0,nPx*4));CK(cudaMemset(d_aB2,0,nPx*4));
    CK(cudaMemset(d_sppMap,0,nPx*4));
    dim3 block(16,16),grid((W+15)/16,(H+15)/16);
    cudaEvent_t mStart,mEnd; CK(cudaEventCreate(&mStart));CK(cudaEventCreate(&mEnd));
    CK(cudaEventRecord(mStart));
    for(int spp=0;spp<maxSpp;spp++){
        pathTrace<<<grid,block>>>(d_bvh4,n4,d_v0x,d_v0y,d_v0z,d_v1x,d_v1y,d_v1z,d_v2x,d_v2y,d_v2z,
            d_matIds,d_aR,d_aG,d_aB,d_aR2,d_aG2,d_aB2,d_sppMap,d_gbuf,W,H,spp,
            camP.x,camP.y,camP.z,fwd.x,fwd.y,fwd.z,
            right.x,right.y,right.z,up.x,up.y,up.z,fovTan,aperture,focusDist);}
    CK(cudaEventRecord(mEnd));CK(cudaEventSynchronize(mEnd));
    float megaMs; CK(cudaEventElapsedTime(&megaMs,mStart,mEnd));

    printf("\n  --- Megakernel (v33 baseline) ---\n");
    printf("  Trace time: %.3f ms (%d spp)\n",megaMs,maxSpp);
    printf("  Per-sample:  %.3f ms\n",megaMs/maxSpp);
    printf("\n  === COMPARISON ===\n");
    printf("  Wavefront:   %.3f ms\n",totalMs);
    printf("  Megakernel:  %.3f ms\n",megaMs);
    printf("  Speedup:     %.2fx\n",megaMs/totalMs);

    tonemap<<<tgrd,tblk>>>(d_aR,d_aG,d_aB,d_sppMap,d_fb,W,H,1.0f,false);
    CK(cudaMemcpy(h_fb,d_fb,nPx*3,cudaMemcpyDeviceToHost));
    writePPM("/workspaces/codespace/VK_RT/v34_megakernel.ppm",h_fb,W,H);
    printf("  Saved: v34_megakernel.ppm\n\n  Done!\n");

    CK(cudaFreeHost(h_fb));CK(cudaFree(d_fb));CK(cudaFree(d_bvh4));
    CK(cudaFree(d_v0x));CK(cudaFree(d_v0y));CK(cudaFree(d_v0z));
    CK(cudaFree(d_v1x));CK(cudaFree(d_v1y));CK(cudaFree(d_v1z));
    CK(cudaFree(d_v2x));CK(cudaFree(d_v2y));CK(cudaFree(d_v2z));
    CK(cudaFree(d_matIds));CK(cudaFree(d_aR));CK(cudaFree(d_aG));CK(cudaFree(d_aB));
    CK(cudaFree(d_aR2));CK(cudaFree(d_aG2));CK(cudaFree(d_aB2));
    CK(cudaFree(d_sppMap));CK(cudaFree(d_gbuf));
    CK(cudaFree(d_rays0));CK(cudaFree(d_rays1));CK(cudaFree(d_hits));CK(cudaFree(d_shadow));
    CK(cudaFree(d_activeCount));CK(cudaFree(d_shadowCount));CK(cudaFree(d_nextCount));
    CK(cudaFree(d_materialBins));CK(cudaFree(d_rayMatType));
    CK(cudaFree(d_sortedIndices));CK(cudaFree(d_binCounters));CK(cudaFree(d_prefixSums));
    return 0;
}
