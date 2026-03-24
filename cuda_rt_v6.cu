/*
 * V100 RT Engine v6 — Aila-Laine While-While + Speculative
 *
 * Based on: "Understanding the Efficiency of Ray Traversal on GPUs" (HPG 2009)
 * Key insight: separate traversal (outer while) from intersection (inner while)
 * All rays in warp do traversal together; only rays needing intersection pause
 * Speculative: rays that found a leaf still participate in traversal
 *
 * Also: structured scene (Stanford bunny-like) for realistic BVH quality
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <algorithm>

#define SAH_BINS 16
#define LEAF_SIZE 8
#define MAX_STACK 32
#define CONST_BVH 2040

__constant__ float4 c_bvh[CONST_BVH*2];
__constant__ int c_bvhN;

struct Triangle{float3 v0,v1,v2;};
struct AABB{float3 bmin,bmax;};
struct HitResult{float t;int tri;float u,v;};
struct BN{AABB b;int l,r,ts,tc;};

static inline float i2f(int i){float f;memcpy(&f,&i,4);return f;}
__device__ __forceinline__ int d_f2i(float f){int i;memcpy(&i,&f,4);return i;}

AABB unite(const AABB&a,const AABB&b){return{{fminf(a.bmin.x,b.bmin.x),fminf(a.bmin.y,b.bmin.y),fminf(a.bmin.z,b.bmin.z)},{fmaxf(a.bmax.x,b.bmax.x),fmaxf(a.bmax.y,b.bmax.y),fmaxf(a.bmax.z,b.bmax.z)}};}
AABB triBox(const Triangle&t){return{{fminf(fminf(t.v0.x,t.v1.x),t.v2.x),fminf(fminf(t.v0.y,t.v1.y),t.v2.y),fminf(fminf(t.v0.z,t.v1.z),t.v2.z)},{fmaxf(fmaxf(t.v0.x,t.v1.x),t.v2.x),fmaxf(fmaxf(t.v0.y,t.v1.y),t.v2.y),fmaxf(fmaxf(t.v0.z,t.v1.z),t.v2.z)}};}
float3 triCen(const Triangle&t){return{(t.v0.x+t.v1.x+t.v2.x)/3,(t.v0.y+t.v1.y+t.v2.y)/3,(t.v0.z+t.v1.z+t.v2.z)/3};}
float sa(const AABB&b){float dx=b.bmax.x-b.bmin.x,dy=b.bmax.y-b.bmin.y,dz=b.bmax.z-b.bmin.z;return 2.0f*(dx*dy+dy*dz+dz*dx);}

int buildSAH(BN*n,Triangle*t,int*idx,int&nc,int s,int c,int d){
    int ni=nc++;BN&nd=n[ni];AABB bounds=triBox(t[idx[s]]);
    for(int i=1;i<c;i++)bounds=unite(bounds,triBox(t[idx[s+i]]));
    nd.b=bounds;
    if(c<=LEAF_SIZE||d>26){nd.l=-1;nd.r=-1;nd.ts=s;nd.tc=c;return ni;}
    AABB cb;cb.bmin=cb.bmax=triCen(t[idx[s]]);
    for(int i=1;i<c;i++){float3 cc=triCen(t[idx[s+i]]);
        cb.bmin={fminf(cb.bmin.x,cc.x),fminf(cb.bmin.y,cc.y),fminf(cb.bmin.z,cc.z)};
        cb.bmax={fmaxf(cb.bmax.x,cc.x),fmaxf(cb.bmax.y,cc.y),fmaxf(cb.bmax.z,cc.z)};}
    float bc=FLT_MAX;int ba=-1,bb=-1;float ps=sa(bounds);
    for(int ax=0;ax<3;ax++){
        float amn=ax==0?cb.bmin.x:ax==1?cb.bmin.y:cb.bmin.z;
        float amx=ax==0?cb.bmax.x:ax==1?cb.bmax.y:cb.bmax.z;
        if(amx-amn<1e-7f)continue;
        struct B{AABB b;int c;};B bins[SAH_BINS];
        for(int i=0;i<SAH_BINS;i++){bins[i].b.bmin={FLT_MAX,FLT_MAX,FLT_MAX};bins[i].b.bmax={-FLT_MAX,-FLT_MAX,-FLT_MAX};bins[i].c=0;}
        float sc=SAH_BINS/(amx-amn);
        for(int i=0;i<c;i++){float3 cc=triCen(t[idx[s+i]]);float cv=ax==0?cc.x:ax==1?cc.y:cc.z;
            int b2=min(max((int)((cv-amn)*sc),0),SAH_BINS-1);bins[b2].b=unite(bins[b2].b,triBox(t[idx[s+i]]));bins[b2].c++;}
        AABB lB[SAH_BINS];int lC[SAH_BINS];lB[0]=bins[0].b;lC[0]=bins[0].c;
        for(int i=1;i<SAH_BINS;i++){lB[i]=unite(lB[i-1],bins[i].b);lC[i]=lC[i-1]+bins[i].c;}
        AABB rB[SAH_BINS];int rC[SAH_BINS];rB[SAH_BINS-1]=bins[SAH_BINS-1].b;rC[SAH_BINS-1]=bins[SAH_BINS-1].c;
        for(int i=SAH_BINS-2;i>=0;i--){rB[i]=unite(rB[i+1],bins[i].b);rC[i]=rC[i+1]+bins[i].c;}
        for(int i=0;i<SAH_BINS-1;i++){if(lC[i]==0||rC[i+1]==0)continue;
            float cost=1.0f+(lC[i]*sa(lB[i])+rC[i+1]*sa(rB[i+1]))/ps;
            if(cost<bc){bc=cost;ba=ax;bb=i;}}
    }
    if(ba==-1||bc>(float)c){nd.l=-1;nd.r=-1;nd.ts=s;nd.tc=c;return ni;}
    float amn=ba==0?cb.bmin.x:ba==1?cb.bmin.y:cb.bmin.z;
    float amx=ba==0?cb.bmax.x:ba==1?cb.bmax.y:cb.bmax.z;float sc=SAH_BINS/(amx-amn);
    int i=s,j=s+c-1;
    while(i<=j){float3 cc=triCen(t[idx[i]]);float cv=ba==0?cc.x:ba==1?cc.y:cc.z;
        int b2=min(max((int)((cv-amn)*sc),0),SAH_BINS-1);
        if(b2<=bb)i++;else{int tmp=idx[i];idx[i]=idx[j];idx[j]=tmp;j--;}}
    int lc=i-s;if(lc==0)lc=1;if(lc==c)lc=c-1;
    nd.ts=-1;nd.tc=0;
    nd.l=buildSAH(n,t,idx,nc,s,lc,d+1);
    nd.r=buildSAH(n,t,idx,nc,s+lc,c-lc,d+1);
    return ni;
}

// ═══ Aila-Laine While-While Speculative Traversal Kernel ═══
// Outer loop: traverse BVH (all rays in warp participate)
// When a ray reaches a leaf, it remembers it but keeps traversing speculatively
// Inner loop: process remembered leaves when traversal stack is empty
__global__ void traceAilaLaine(
    cudaTextureObject_t tex, int nn,
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    const float*__restrict__ rix,const float*__restrict__ riy,const float*__restrict__ riz,
    HitResult*__restrict__ hits, int numRays,
    unsigned long long*__restrict__ stats)
{
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int stride=gridDim.x*blockDim.x;
    unsigned long long ln=0,lt=0;
    
    for(int ri=gid;ri<numRays;ri+=stride){
        float ox=rox[ri],oy=roy[ri],oz=roz[ri];
        float dx=rdx[ri],dy=rdy[ri],dz=rdz[ri];
        float ix=rix[ri],iy=riy[ri],iz=riz[ri];
        float hitT=1e30f;int hitTri=-1;float hitU=0,hitV=0;
        
        int stack[MAX_STACK];int sp=0;
        int nodeIdx=0; // current traversal node
        // Speculative: leafAddr stores pending leaf to intersect
        int leafAddr=-1; // -1 = no pending leaf
        int leafStart=0, leafCount=0;
        
        // While-while: outer = traversal, inner = intersection
        while(nodeIdx >= 0 || leafAddr >= 0) {
            // === INNER WHILE: process pending leaf ===
            while(leafAddr >= 0) {
                for(int i=0;i<leafCount;i++){
                    int ti=leafStart+i; lt++;
                    float v0x=tv0x[ti],v0y=tv0y[ti],v0z=tv0z[ti];
                    float e1x=tv1x[ti]-v0x,e1y=tv1y[ti]-v0y,e1z=tv1z[ti]-v0z;
                    float e2x=tv2x[ti]-v0x,e2y=tv2y[ti]-v0y,e2z=tv2z[ti]-v0z;
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
                    float t=f*(e2x*qx+e2y*qy+e2z*qz);
                    if(t>0.001f&&t<hitT){hitT=t;hitTri=ti;hitU=u;hitV=v;}
                }
                leafAddr=-1;
                // Pop from stack if available
                if(sp>0) nodeIdx=stack[--sp];
                else break;
            }
            
            if(nodeIdx<0 && leafAddr<0) break;
            if(nodeIdx<0) continue;
            
            // === OUTER WHILE: BVH traversal ===
            float4 nlo,nhi;
            if(nodeIdx<c_bvhN){nlo=c_bvh[nodeIdx*2];nhi=c_bvh[nodeIdx*2+1];}
            else{nlo=tex1Dfetch<float4>(tex,nodeIdx*2);nhi=tex1Dfetch<float4>(tex,nodeIdx*2+1);}
            
            int leftChild=d_f2i(nlo.w);
            int rightData=d_f2i(nhi.w);
            ln++;
            
            if(leftChild<0){
                // Leaf: remember it, don't process yet (speculative)
                leafAddr=nodeIdx;
                leafStart=rightData;
                leafCount=(-leftChild)-1;
                nodeIdx=(sp>0)?stack[--sp]:-1;
            } else {
                int rc=rightData;
                // Test both children
                float4 llo,lhi,rlo,rhi;
                if(leftChild<c_bvhN){llo=c_bvh[leftChild*2];lhi=c_bvh[leftChild*2+1];}
                else{llo=tex1Dfetch<float4>(tex,leftChild*2);lhi=tex1Dfetch<float4>(tex,leftChild*2+1);}
                if(rc<c_bvhN){rlo=c_bvh[rc*2];rhi=c_bvh[rc*2+1];}
                else{rlo=tex1Dfetch<float4>(tex,rc*2);rhi=tex1Dfetch<float4>(tex,rc*2+1);}
                
                float ltx1=(llo.x-ox)*ix,ltx2=(lhi.x-ox)*ix;
                float lt0=fminf(ltx1,ltx2),lt1=fmaxf(ltx1,ltx2);
                float lty1=(llo.y-oy)*iy,lty2=(lhi.y-oy)*iy;
                lt0=fmaxf(lt0,fminf(lty1,lty2));lt1=fminf(lt1,fmaxf(lty1,lty2));
                float ltz1=(llo.z-oz)*iz,ltz2=(lhi.z-oz)*iz;
                lt0=fmaxf(lt0,fminf(ltz1,ltz2));lt1=fminf(lt1,fmaxf(ltz1,ltz2));
                bool hL=lt1>=fmaxf(lt0,0.0f)&&lt0<=hitT;
                
                float rtx1=(rlo.x-ox)*ix,rtx2=(rhi.x-ox)*ix;
                float rt0=fminf(rtx1,rtx2),rt1=fmaxf(rtx1,rtx2);
                float rty1=(rlo.y-oy)*iy,rty2=(rhi.y-oy)*iy;
                rt0=fmaxf(rt0,fminf(rty1,rty2));rt1=fminf(rt1,fmaxf(rty1,rty2));
                float rtz1=(rlo.z-oz)*iz,rtz2=(rhi.z-oz)*iz;
                rt0=fmaxf(rt0,fminf(rtz1,rtz2));rt1=fminf(rt1,fmaxf(rtz1,rtz2));
                bool hR=rt1>=fmaxf(rt0,0.0f)&&rt0<=hitT;
                
                ln+=2;
                
                if(hL&&hR){
                    if(lt0<rt0){
                        stack[sp++]=rc;
                        nodeIdx=leftChild;
                    }else{
                        stack[sp++]=leftChild;
                        nodeIdx=rc;
                    }
                }else if(hL) nodeIdx=leftChild;
                else if(hR) nodeIdx=rc;
                else nodeIdx=(sp>0)?stack[--sp]:-1;
            }
        }
        hits[ri].t=hitT;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;
    }
    atomicAdd(&stats[0],ln);atomicAdd(&stats[1],lt);
}

// ═══ Scene generators ═══
void genRandom(Triangle*t,int n){srand(42);for(int i=0;i<n;i++){
    float cx=((float)rand()/RAND_MAX)*20-10,cy=((float)rand()/RAND_MAX)*20-10,cz=((float)rand()/RAND_MAX)*20-10;
    float s=((float)rand()/RAND_MAX)*0.5f+0.1f;
    t[i].v0={cx-s,cy-s,cz};t[i].v1={cx+s,cy,cz+s};t[i].v2={cx,cy+s,cz-s};}}

// Structured scene: grid of sphereoids (like a room with objects)
void genStructured(Triangle*t,int n){
    srand(42);
    int objectCount=(int)cbrtf((float)n/12); // ~objects per axis
    float spacing=20.0f/objectCount;
    int ti=0;
    for(int ox=0;ox<objectCount&&ti<n;ox++)
    for(int oy=0;oy<objectCount&&ti<n;oy++)
    for(int oz=0;oz<objectCount&&ti<n;oz++){
        float cx=-10+ox*spacing+spacing/2;
        float cy=-10+oy*spacing+spacing/2;
        float cz=-10+oz*spacing+spacing/2;
        float r=spacing*0.3f;
        // 12 tris per object (icosahedron-like)
        for(int f=0;f<12&&ti<n;f++){
            float a1=f*0.524f,a2=(f+1)*0.524f;
            float h=(f<6)?r:-r;
            t[ti].v0={cx,cy+h,cz};
            t[ti].v1={cx+r*cosf(a1),cy+h*0.5f,cz+r*sinf(a1)};
            t[ti].v2={cx+r*cosf(a2),cy+h*0.5f,cz+r*sinf(a2)};
            ti++;
        }
    }
    // Fill remaining with small tris near center
    while(ti<n){
        float cx=((float)rand()/RAND_MAX)*2-1;
        float cy=((float)rand()/RAND_MAX)*2-1;
        float cz=((float)rand()/RAND_MAX)*2-1;
        t[ti].v0={cx-0.1f,cy-0.1f,cz};t[ti].v1={cx+0.1f,cy,cz+0.1f};t[ti].v2={cx,cy+0.1f,cz-0.1f};
        ti++;
    }
}

unsigned int xBits(unsigned int v){v=(v*0x00010001u)&0xFF0000FFu;v=(v*0x00000101u)&0x0F00F00Fu;v=(v*0x00000011u)&0xC30C30C3u;v=(v*0x00000005u)&0x49249249u;return v;}
struct RayAoS{float3 o,d,id;};

// Primary rays from camera (coherent)
void genPrimaryRays(RayAoS*r,int n){
    int w=(int)sqrtf((float)n);int h=n/w;
    for(int i=0;i<n;i++){
        int px=i%w,py=i/w;
        float u=(2.0f*px/w-1.0f)*1.5f;
        float v=(2.0f*py/h-1.0f)*1.0f;
        r[i].o={0,0,25};
        float3 d={u,-v,-2.5f};float l=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);
        d.x/=l;d.y/=l;d.z/=l;
        r[i].d=d;r[i].id={1.0f/d.x,1.0f/d.y,1.0f/d.z};
    }
}

// Diffuse (incoherent) rays
void genDiffuseRays(RayAoS*r,int n){
    srand(123);for(int i=0;i<n;i++){
        float u=((float)rand()/RAND_MAX)*6.283f,v=((float)rand()/RAND_MAX)*3.14159f;
        r[i].o={sinf(v)*cosf(u)*15,sinf(v)*sinf(u)*15,cosf(v)*15};
        float3 tgt={((float)rand()/RAND_MAX-0.5f)*2,((float)rand()/RAND_MAX-0.5f)*2,((float)rand()/RAND_MAX-0.5f)*2};
        float3 d={tgt.x-r[i].o.x,tgt.y-r[i].o.y,tgt.z-r[i].o.z};
        float l=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);d.x/=l;d.y/=l;d.z/=l;
        r[i].d=d;r[i].id={1.0f/d.x,1.0f/d.y,1.0f/d.z};}
    // Morton sort
    float3 mn={FLT_MAX,FLT_MAX,FLT_MAX},mx={-FLT_MAX,-FLT_MAX,-FLT_MAX};
    for(int i=0;i<n;i++){mn.x=fminf(mn.x,r[i].o.x);mn.y=fminf(mn.y,r[i].o.y);mn.z=fminf(mn.z,r[i].o.z);
        mx.x=fmaxf(mx.x,r[i].o.x);mx.y=fmaxf(mx.y,r[i].o.y);mx.z=fmaxf(mx.z,r[i].o.z);}
    unsigned int*mc=(unsigned int*)malloc(n*4);int*ii=(int*)malloc(n*4);
    for(int i=0;i<n;i++){float x=(r[i].o.x-mn.x)/(mx.x-mn.x+1e-6f)*1023,y=(r[i].o.y-mn.y)/(mx.y-mn.y+1e-6f)*1023,z=(r[i].o.z-mn.z)/(mx.z-mn.z+1e-6f)*1023;
        mc[i]=xBits((unsigned)fminf(fmaxf(x,0),1023))|(xBits((unsigned)fminf(fmaxf(y,0),1023))<<1)|(xBits((unsigned)fminf(fmaxf(z,0),1023))<<2);ii[i]=i;}
    std::sort(ii,ii+n,[&mc](int a,int b){return mc[a]<mc[b];});
    RayAoS*s=(RayAoS*)malloc(n*sizeof(RayAoS));for(int i=0;i<n;i++)s[i]=r[ii[i]];
    memcpy(r,s,n*sizeof(RayAoS));free(s);free(mc);free(ii);
}

float runTest(int nt,int numRays,Triangle*h_tris,RayAoS*h_r,cudaDeviceProp&prop,
              const char*label,unsigned long long*outN,unsigned long long*outT,int*outH){
    BN*h_nodes=(BN*)calloc(nt*2,sizeof(BN));int*tidx=(int*)malloc(nt*4);
    for(int i=0;i<nt;i++)tidx[i]=i;int nc=0;
    buildSAH(h_nodes,h_tris,tidx,nc,0,nt,0);
    
    Triangle*h_to=(Triangle*)malloc(nt*sizeof(Triangle));
    for(int i=0;i<nt;i++)h_to[i]=h_tris[tidx[i]];
    
    float4*h_pk=(float4*)malloc(nc*2*sizeof(float4));
    for(int i=0;i<nc;i++){BN&nd=h_nodes[i];int lv,rv;
        if(nd.l==-1){lv=-(nd.tc+1);rv=nd.ts;}else{lv=nd.l;rv=nd.r;}
        h_pk[i*2]={nd.b.bmin.x,nd.b.bmin.y,nd.b.bmin.z,i2f(lv)};
        h_pk[i*2+1]={nd.b.bmax.x,nd.b.bmax.y,nd.b.bmax.z,i2f(rv)};}
    
    int cn=min(nc,CONST_BVH);
    cudaMemcpyToSymbol(c_bvh,h_pk,cn*2*sizeof(float4));
    cudaMemcpyToSymbol(c_bvhN,&cn,4);
    
    float4*d_pk;cudaMalloc(&d_pk,nc*2*sizeof(float4));
    cudaMemcpy(d_pk,h_pk,nc*2*sizeof(float4),cudaMemcpyHostToDevice);
    cudaResourceDesc rd;memset(&rd,0,sizeof(rd));rd.resType=cudaResourceTypeLinear;
    rd.res.linear.devPtr=d_pk;rd.res.linear.desc=cudaCreateChannelDesc<float4>();
    rd.res.linear.sizeInBytes=nc*2*sizeof(float4);
    cudaTextureDesc td;memset(&td,0,sizeof(td));td.readMode=cudaReadModeElementType;
    cudaTextureObject_t tex=0;cudaCreateTextureObject(&tex,&rd,&td,NULL);
    
    float*h_v[9];for(int j=0;j<9;j++)h_v[j]=(float*)malloc(nt*4);
    for(int i=0;i<nt;i++){
        h_v[0][i]=h_to[i].v0.x;h_v[1][i]=h_to[i].v0.y;h_v[2][i]=h_to[i].v0.z;
        h_v[3][i]=h_to[i].v1.x;h_v[4][i]=h_to[i].v1.y;h_v[5][i]=h_to[i].v1.z;
        h_v[6][i]=h_to[i].v2.x;h_v[7][i]=h_to[i].v2.y;h_v[8][i]=h_to[i].v2.z;}
    
    float*h_ray[9];for(int j=0;j<9;j++)h_ray[j]=(float*)malloc(numRays*4);
    for(int i=0;i<numRays;i++){
        h_ray[0][i]=h_r[i].o.x;h_ray[1][i]=h_r[i].o.y;h_ray[2][i]=h_r[i].o.z;
        h_ray[3][i]=h_r[i].d.x;h_ray[4][i]=h_r[i].d.y;h_ray[5][i]=h_r[i].d.z;
        h_ray[6][i]=h_r[i].id.x;h_ray[7][i]=h_r[i].id.y;h_ray[8][i]=h_r[i].id.z;}
    
    float*d_v[9],*d_ray[9];HitResult*d_h;unsigned long long*d_st;
    for(int j=0;j<9;j++){cudaMalloc(&d_v[j],nt*4);cudaMemcpy(d_v[j],h_v[j],nt*4,cudaMemcpyHostToDevice);}
    for(int j=0;j<9;j++){cudaMalloc(&d_ray[j],numRays*4);cudaMemcpy(d_ray[j],h_ray[j],numRays*4,cudaMemcpyHostToDevice);}
    cudaMalloc(&d_h,numRays*sizeof(HitResult));cudaMalloc(&d_st,16);
    
    int nb=prop.multiProcessorCount*8;
    
    cudaMemset(d_st,0,16);
    traceAilaLaine<<<nb,256>>>(tex,nc,d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
        d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],
        d_h,numRays,d_st);
    cudaDeviceSynchronize();
    
    cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
    float totalMs=0;int runs=10;
    for(int r=0;r<runs;r++){
        cudaMemset(d_st,0,16);cudaEventRecord(t0);
        traceAilaLaine<<<nb,256>>>(tex,nc,d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
            d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],
            d_h,numRays,d_st);
        cudaEventRecord(t1);cudaEventSynchronize(t1);
        float ms;cudaEventElapsedTime(&ms,t0,t1);totalMs+=ms;}
    float avg=totalMs/runs;
    
    unsigned long long st[2];cudaMemcpy(st,d_st,16,cudaMemcpyDeviceToHost);
    HitResult*hh=(HitResult*)malloc(numRays*sizeof(HitResult));
    cudaMemcpy(hh,d_h,numRays*sizeof(HitResult),cudaMemcpyDeviceToHost);
    int hc=0;for(int i=0;i<numRays;i++)if(hh[i].tri>=0)hc++;free(hh);
    
    *outN=st[0];*outT=st[1];*outH=hc;
    
    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaDestroyTextureObject(tex);cudaFree(d_pk);
    for(int j=0;j<9;j++){cudaFree(d_v[j]);cudaFree(d_ray[j]);free(h_v[j]);free(h_ray[j]);}
    cudaFree(d_h);cudaFree(d_st);free(h_nodes);free(tidx);free(h_pk);free(h_to);
    return avg;
}

int main(){
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  V100 RT Engine v6 — Aila-Laine While-While + All Hardware          ║\n");
    printf("║  Grid-Stride + TexCache + ConstMem + Speculative Traversal           ║\n");
    printf("║  Random & Structured scenes × Primary & Diffuse rays                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");
    
    cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
    printf("GPU: %s | %d SMs | %.0f GB/s\n\n",prop.name,prop.multiProcessorCount,
        2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1e6);
    
    int numRays=4194304;
    int triCounts[]={100000,500000,1000000};
    
    printf("  ┌──────────┬────────────┬─────────┬──────────┬────────┬────────┬──────────┐\n");
    printf("  │ Scene    │ Ray Type   │ Time ms │ MRays/s  │ n/ray  │ t/ray  │ Hit%%     │\n");
    printf("  ├──────────┼────────────┼─────────┼──────────┼────────┼────────┼──────────┤\n");
    
    for(int s=0;s<3;s++){
        int nt=triCounts[s];
        
        // Random scene + diffuse rays (worst case)
        Triangle*rTris=(Triangle*)malloc(nt*sizeof(Triangle));
        genRandom(rTris,nt);
        RayAoS*dRays=(RayAoS*)malloc(numRays*sizeof(RayAoS));
        genDiffuseRays(dRays,numRays);
        
        unsigned long long on,ot;int oh;
        float ms=runTest(nt,numRays,rTris,dRays,prop,"rand+diff",&on,&ot,&oh);
        double mr=(double)numRays/(ms/1000.0)/1e6;
        printf("  │ Rnd %3dK │ Diffuse    │ %6.1f  │ %7.1f  │ %5.1f  │ %5.1f  │ %5.1f%%   │\n",
            nt/1000,ms,mr,(double)on/numRays,(double)ot/numRays,100.0*oh/numRays);
        
        // Random scene + primary rays (coherent)
        RayAoS*pRays=(RayAoS*)malloc(numRays*sizeof(RayAoS));
        genPrimaryRays(pRays,numRays);
        ms=runTest(nt,numRays,rTris,pRays,prop,"rand+prim",&on,&ot,&oh);
        mr=(double)numRays/(ms/1000.0)/1e6;
        printf("  │ Rnd %3dK │ Primary    │ %6.1f  │ %7.1f  │ %5.1f  │ %5.1f  │ %5.1f%%   │\n",
            nt/1000,ms,mr,(double)on/numRays,(double)ot/numRays,100.0*oh/numRays);
        
        // Structured scene + diffuse
        Triangle*sTris=(Triangle*)malloc(nt*sizeof(Triangle));
        genStructured(sTris,nt);
        ms=runTest(nt,numRays,sTris,dRays,prop,"str+diff",&on,&ot,&oh);
        mr=(double)numRays/(ms/1000.0)/1e6;
        printf("  │ Str %3dK │ Diffuse    │ %6.1f  │ %7.1f  │ %5.1f  │ %5.1f  │ %5.1f%%   │\n",
            nt/1000,ms,mr,(double)on/numRays,(double)ot/numRays,100.0*oh/numRays);
        
        // Structured scene + primary
        ms=runTest(nt,numRays,sTris,pRays,prop,"str+prim",&on,&ot,&oh);
        mr=(double)numRays/(ms/1000.0)/1e6;
        printf("  │ Str %3dK │ Primary    │ %6.1f  │ %7.1f  │ %5.1f  │ %5.1f  │ %5.1f%%   │\n",
            nt/1000,ms,mr,(double)on/numRays,(double)ot/numRays,100.0*oh/numRays);
        
        printf("  ├──────────┼────────────┼─────────┼──────────┼────────┼────────┼──────────┤\n");
        
        free(rTris);free(sTris);free(dRays);free(pRays);
    }
    
    printf("  └──────────┴────────────┴─────────┴──────────┴────────┴────────┴──────────┘\n\n");
    printf("  Previous: v4=153MR/s(100K,diff) | Aila-Laine GTX680=432MR/s(283K,prim)\n");
    printf("  V100 has 4.7× GTX680 bandwidth → theoretical ~2000 MR/s primary\n");
    
    return 0;
}
