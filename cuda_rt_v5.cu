// v5: Same as v4 but with configurable leaf size + speculative traversal restart
// Also: tighter AABB epsilon, removed unnecessary branches

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <algorithm>

#define SAH_BINS 16
#define MAX_STACK 32
#define CONST_BVH_NODES 2040

__constant__ float4 c_bvh[CONST_BVH_NODES * 2];
__constant__ int c_bvhCount;

struct Triangle{float3 v0,v1,v2;};
struct AABB{float3 bmin,bmax;};
struct HitResult{float t;int triIdx;float u,v;};
struct BN{AABB bounds;int left,right,triStart,triCount;};

static inline float i2f(int i){float f;memcpy(&f,&i,4);return f;}
__device__ __forceinline__ int d_f2i(float f){int i;memcpy(&i,&f,4);return i;}

AABB unite(const AABB&a,const AABB&b){return{{fminf(a.bmin.x,b.bmin.x),fminf(a.bmin.y,b.bmin.y),fminf(a.bmin.z,b.bmin.z)},{fmaxf(a.bmax.x,b.bmax.x),fmaxf(a.bmax.y,b.bmax.y),fmaxf(a.bmax.z,b.bmax.z)}};}
AABB triBox(const Triangle&t){return{{fminf(fminf(t.v0.x,t.v1.x),t.v2.x),fminf(fminf(t.v0.y,t.v1.y),t.v2.y),fminf(fminf(t.v0.z,t.v1.z),t.v2.z)},{fmaxf(fmaxf(t.v0.x,t.v1.x),t.v2.x),fmaxf(fmaxf(t.v0.y,t.v1.y),t.v2.y),fmaxf(fmaxf(t.v0.z,t.v1.z),t.v2.z)}};}
float3 triCen(const Triangle&t){return{(t.v0.x+t.v1.x+t.v2.x)/3,(t.v0.y+t.v1.y+t.v2.y)/3,(t.v0.z+t.v1.z+t.v2.z)/3};}
float sa(const AABB&b){float dx=b.bmax.x-b.bmin.x,dy=b.bmax.y-b.bmin.y,dz=b.bmax.z-b.bmin.z;return 2.0f*(dx*dy+dy*dz+dz*dx);}

int buildSAH(BN*nodes,Triangle*tris,int*idx,int&nc,int start,int count,int depth,int leafSz){
    int ni=nc++;BN&nd=nodes[ni];
    AABB bounds=triBox(tris[idx[start]]);
    for(int i=1;i<count;i++)bounds=unite(bounds,triBox(tris[idx[start+i]]));
    nd.bounds=bounds;
    if(count<=leafSz||depth>26){nd.left=-1;nd.right=-1;nd.triStart=start;nd.triCount=count;return ni;}
    AABB cb;cb.bmin=cb.bmax=triCen(tris[idx[start]]);
    for(int i=1;i<count;i++){float3 c=triCen(tris[idx[start+i]]);
        cb.bmin={fminf(cb.bmin.x,c.x),fminf(cb.bmin.y,c.y),fminf(cb.bmin.z,c.z)};
        cb.bmax={fmaxf(cb.bmax.x,c.x),fmaxf(cb.bmax.y,c.y),fmaxf(cb.bmax.z,c.z)};}
    float bestC=FLT_MAX;int bestA=-1,bestB=-1;float psa=sa(bounds);
    // SAH leaf cost: if splitting isn't worth it, make leaf
    float leafCost = (float)count; // cost of testing all tris
    for(int ax=0;ax<3;ax++){
        float amn=ax==0?cb.bmin.x:ax==1?cb.bmin.y:cb.bmin.z;
        float amx=ax==0?cb.bmax.x:ax==1?cb.bmax.y:cb.bmax.z;
        if(amx-amn<1e-7f)continue;
        struct Bin{AABB b;int c;};Bin bins[SAH_BINS];
        for(int i=0;i<SAH_BINS;i++){bins[i].b.bmin={FLT_MAX,FLT_MAX,FLT_MAX};bins[i].b.bmax={-FLT_MAX,-FLT_MAX,-FLT_MAX};bins[i].c=0;}
        float sc=SAH_BINS/(amx-amn);
        for(int i=0;i<count;i++){float3 c=triCen(tris[idx[start+i]]);float cv=ax==0?c.x:ax==1?c.y:c.z;
            int b=min(max((int)((cv-amn)*sc),0),SAH_BINS-1);bins[b].b=unite(bins[b].b,triBox(tris[idx[start+i]]));bins[b].c++;}
        AABB lB[SAH_BINS];int lC[SAH_BINS];lB[0]=bins[0].b;lC[0]=bins[0].c;
        for(int i=1;i<SAH_BINS;i++){lB[i]=unite(lB[i-1],bins[i].b);lC[i]=lC[i-1]+bins[i].c;}
        AABB rB[SAH_BINS];int rC[SAH_BINS];rB[SAH_BINS-1]=bins[SAH_BINS-1].b;rC[SAH_BINS-1]=bins[SAH_BINS-1].c;
        for(int i=SAH_BINS-2;i>=0;i--){rB[i]=unite(rB[i+1],bins[i].b);rC[i]=rC[i+1]+bins[i].c;}
        for(int i=0;i<SAH_BINS-1;i++){if(lC[i]==0||rC[i+1]==0)continue;
            float cost=1.0f+(lC[i]*sa(lB[i])+rC[i+1]*sa(rB[i+1]))/psa; // traversal cost = 1.0
            if(cost<bestC){bestC=cost;bestA=ax;bestB=i;}}
    }
    // SAH termination: if best split cost > leaf cost, make leaf
    if(bestA==-1 || bestC > leafCost){
        nd.left=-1;nd.right=-1;nd.triStart=start;nd.triCount=count;return ni;
    }
    float amn=bestA==0?cb.bmin.x:bestA==1?cb.bmin.y:cb.bmin.z;
    float amx=bestA==0?cb.bmax.x:bestA==1?cb.bmax.y:cb.bmax.z;float sc=SAH_BINS/(amx-amn);
    int i=start,j=start+count-1;
    while(i<=j){float3 c=triCen(tris[idx[i]]);float cv=bestA==0?c.x:bestA==1?c.y:c.z;
        int b=min(max((int)((cv-amn)*sc),0),SAH_BINS-1);
        if(b<=bestB)i++;else{int t=idx[i];idx[i]=idx[j];idx[j]=t;j--;}}
    int lc=i-start;if(lc==0)lc=1;if(lc==count)lc=count-1;
    nd.triStart=-1;nd.triCount=0;
    nd.left=buildSAH(nodes,tris,idx,nc,start,lc,depth+1,leafSz);
    nd.right=buildSAH(nodes,tris,idx,nc,start+lc,count-lc,depth+1,leafSz);
    return ni;
}

__global__ void traceV5(
    cudaTextureObject_t nodesTex,int numNodes,
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    const float*__restrict__ ridx,const float*__restrict__ ridy,const float*__restrict__ ridz,
    HitResult*__restrict__ hits,int numRays,
    unsigned long long*__restrict__ stats)
{
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int stride=gridDim.x*blockDim.x;
    unsigned long long ln=0,lt=0;
    
    for(int ri=gid;ri<numRays;ri+=stride){
        float ox=rox[ri],oy=roy[ri],oz=roz[ri];
        float dx=rdx[ri],dy=rdy[ri],dz=rdz[ri];
        float ix=ridx[ri],iy=ridy[ri],iz=ridz[ri];
        float hitT=1e30f;int hitTri=-1;float hitU=0,hitV=0;
        
        int stack[MAX_STACK];int sp=0;
        stack[sp++]=0;
        
        while(sp>0){
            int ni=stack[--sp];
            float4 nlo,nhi;
            if(ni<c_bvhCount){nlo=c_bvh[ni*2];nhi=c_bvh[ni*2+1];}
            else{nlo=tex1Dfetch<float4>(nodesTex,ni*2);nhi=tex1Dfetch<float4>(nodesTex,ni*2+1);}
            
            float tx1=(nlo.x-ox)*ix,tx2=(nhi.x-ox)*ix;
            float tmin=fminf(tx1,tx2),tmax=fmaxf(tx1,tx2);
            float ty1=(nlo.y-oy)*iy,ty2=(nhi.y-oy)*iy;
            tmin=fmaxf(tmin,fminf(ty1,ty2));tmax=fminf(tmax,fmaxf(ty1,ty2));
            float tz1=(nlo.z-oz)*iz,tz2=(nhi.z-oz)*iz;
            tmin=fmaxf(tmin,fminf(tz1,tz2));tmax=fminf(tmax,fmaxf(tz1,tz2));
            
            ln++;
            if(tmax<fmaxf(tmin,0.0f)||tmin>hitT)continue;
            
            int leftChild=d_f2i(nlo.w);
            int rightData=d_f2i(nhi.w);
            
            if(leftChild<0){
                int triStart=rightData;
                int triCount=(-leftChild)-1;
                for(int i=0;i<triCount;i++){
                    int ti=triStart+i;lt++;
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
            }else{
                int rc=rightData;
                float4 llo,lhi,rlo,rhi;
                if(leftChild<c_bvhCount){llo=c_bvh[leftChild*2];lhi=c_bvh[leftChild*2+1];}
                else{llo=tex1Dfetch<float4>(nodesTex,leftChild*2);lhi=tex1Dfetch<float4>(nodesTex,leftChild*2+1);}
                if(rc<c_bvhCount){rlo=c_bvh[rc*2];rhi=c_bvh[rc*2+1];}
                else{rlo=tex1Dfetch<float4>(nodesTex,rc*2);rhi=tex1Dfetch<float4>(nodesTex,rc*2+1);}
                
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
                if(hL&&hR){if(lt0<rt0){stack[sp++]=rc;stack[sp++]=leftChild;}else{stack[sp++]=leftChild;stack[sp++]=rc;}}
                else if(hL)stack[sp++]=leftChild;
                else if(hR)stack[sp++]=rc;
            }
        }
        hits[ri].t=hitT;hits[ri].triIdx=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;
    }
    atomicAdd(&stats[0],ln);atomicAdd(&stats[1],lt);
}

void genScene(Triangle*t,int n){srand(42);for(int i=0;i<n;i++){
    float cx=((float)rand()/RAND_MAX)*20-10,cy=((float)rand()/RAND_MAX)*20-10,cz=((float)rand()/RAND_MAX)*20-10;
    float s=((float)rand()/RAND_MAX)*0.5f+0.1f;
    t[i].v0={cx-s,cy-s,cz};t[i].v1={cx+s,cy,cz+s};t[i].v2={cx,cy+s,cz-s};}}

unsigned int xBits(unsigned int v){v=(v*0x00010001u)&0xFF0000FFu;v=(v*0x00000101u)&0x0F00F00Fu;v=(v*0x00000011u)&0xC30C30C3u;v=(v*0x00000005u)&0x49249249u;return v;}
struct RayAoS{float3 o,d,id;};
void genAndSort(RayAoS*r,int n){
    srand(123);for(int i=0;i<n;i++){float u=((float)rand()/RAND_MAX)*6.283f,v=((float)rand()/RAND_MAX)*3.14159f;
        r[i].o={sinf(v)*cosf(u)*15,sinf(v)*sinf(u)*15,cosf(v)*15};
        float3 tgt={((float)rand()/RAND_MAX-0.5f)*2,((float)rand()/RAND_MAX-0.5f)*2,((float)rand()/RAND_MAX-0.5f)*2};
        float3 d={tgt.x-r[i].o.x,tgt.y-r[i].o.y,tgt.z-r[i].o.z};
        float l=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);d.x/=l;d.y/=l;d.z/=l;
        r[i].d=d;r[i].id={1.0f/d.x,1.0f/d.y,1.0f/d.z};}
    float3 mn={FLT_MAX,FLT_MAX,FLT_MAX},mx={-FLT_MAX,-FLT_MAX,-FLT_MAX};
    for(int i=0;i<n;i++){mn.x=fminf(mn.x,r[i].o.x);mn.y=fminf(mn.y,r[i].o.y);mn.z=fminf(mn.z,r[i].o.z);
        mx.x=fmaxf(mx.x,r[i].o.x);mx.y=fmaxf(mx.y,r[i].o.y);mx.z=fmaxf(mx.z,r[i].o.z);}
    unsigned int*mc=(unsigned int*)malloc(n*4);int*ii=(int*)malloc(n*4);
    for(int i=0;i<n;i++){float x=(r[i].o.x-mn.x)/(mx.x-mn.x)*1023,y=(r[i].o.y-mn.y)/(mx.y-mn.y)*1023,z=(r[i].o.z-mn.z)/(mx.z-mn.z)*1023;
        mc[i]=xBits((unsigned)fminf(fmaxf(x,0),1023))|(xBits((unsigned)fminf(fmaxf(y,0),1023))<<1)|(xBits((unsigned)fminf(fmaxf(z,0),1023))<<2);ii[i]=i;}
    std::sort(ii,ii+n,[&mc](int a,int b){return mc[a]<mc[b];});
    RayAoS*s=(RayAoS*)malloc(n*sizeof(RayAoS));for(int i=0;i<n;i++)s[i]=r[ii[i]];
    memcpy(r,s,n*sizeof(RayAoS));free(s);free(mc);free(ii);
}

void runBench(int nt,int numRays,int leafSz,cudaDeviceProp&prop){
    Triangle*h_tris=(Triangle*)malloc(nt*sizeof(Triangle));genScene(h_tris,nt);
    BN*h_nodes=(BN*)calloc(nt*2,sizeof(BN));int*tidx=(int*)malloc(nt*4);
    for(int i=0;i<nt;i++)tidx[i]=i;int nc=0;
    buildSAH(h_nodes,h_tris,tidx,nc,0,nt,0,leafSz);
    
    Triangle*h_to=(Triangle*)malloc(nt*sizeof(Triangle));
    for(int i=0;i<nt;i++)h_to[i]=h_tris[tidx[i]];
    
    float4*h_pk=(float4*)malloc(nc*2*sizeof(float4));
    for(int i=0;i<nc;i++){BN&nd=h_nodes[i];int lv,rv;
        if(nd.left==-1){lv=-(nd.triCount+1);rv=nd.triStart;}
        else{lv=nd.left;rv=nd.right;}
        h_pk[i*2]={nd.bounds.bmin.x,nd.bounds.bmin.y,nd.bounds.bmin.z,i2f(lv)};
        h_pk[i*2+1]={nd.bounds.bmax.x,nd.bounds.bmax.y,nd.bounds.bmax.z,i2f(rv)};}
    
    int cn=min(nc,CONST_BVH_NODES);
    cudaMemcpyToSymbol(c_bvh,h_pk,cn*2*sizeof(float4));
    cudaMemcpyToSymbol(c_bvhCount,&cn,4);
    
    float4*d_pk;cudaMalloc(&d_pk,nc*2*sizeof(float4));
    cudaMemcpy(d_pk,h_pk,nc*2*sizeof(float4),cudaMemcpyHostToDevice);
    cudaResourceDesc rd;memset(&rd,0,sizeof(rd));rd.resType=cudaResourceTypeLinear;
    rd.res.linear.devPtr=d_pk;rd.res.linear.desc=cudaCreateChannelDesc<float4>();
    rd.res.linear.sizeInBytes=nc*2*sizeof(float4);
    cudaTextureDesc td;memset(&td,0,sizeof(td));td.readMode=cudaReadModeElementType;
    cudaTextureObject_t tex=0;cudaCreateTextureObject(&tex,&rd,&td,NULL);
    
    float*h_v0x=(float*)malloc(nt*4),*h_v0y=(float*)malloc(nt*4),*h_v0z=(float*)malloc(nt*4);
    float*h_v1x=(float*)malloc(nt*4),*h_v1y=(float*)malloc(nt*4),*h_v1z=(float*)malloc(nt*4);
    float*h_v2x=(float*)malloc(nt*4),*h_v2y=(float*)malloc(nt*4),*h_v2z=(float*)malloc(nt*4);
    for(int i=0;i<nt;i++){h_v0x[i]=h_to[i].v0.x;h_v0y[i]=h_to[i].v0.y;h_v0z[i]=h_to[i].v0.z;
        h_v1x[i]=h_to[i].v1.x;h_v1y[i]=h_to[i].v1.y;h_v1z[i]=h_to[i].v1.z;
        h_v2x[i]=h_to[i].v2.x;h_v2y[i]=h_to[i].v2.y;h_v2z[i]=h_to[i].v2.z;}
    
    RayAoS*h_r=(RayAoS*)malloc(numRays*sizeof(RayAoS));genAndSort(h_r,numRays);
    float*h_rox=(float*)malloc(numRays*4),*h_roy=(float*)malloc(numRays*4),*h_roz=(float*)malloc(numRays*4);
    float*h_rdx=(float*)malloc(numRays*4),*h_rdy=(float*)malloc(numRays*4),*h_rdz=(float*)malloc(numRays*4);
    float*h_idx=(float*)malloc(numRays*4),*h_idy=(float*)malloc(numRays*4),*h_idz=(float*)malloc(numRays*4);
    for(int i=0;i<numRays;i++){h_rox[i]=h_r[i].o.x;h_roy[i]=h_r[i].o.y;h_roz[i]=h_r[i].o.z;
        h_rdx[i]=h_r[i].d.x;h_rdy[i]=h_r[i].d.y;h_rdz[i]=h_r[i].d.z;
        h_idx[i]=h_r[i].id.x;h_idy[i]=h_r[i].id.y;h_idz[i]=h_r[i].id.z;}
    
    float*d_v0x,*d_v0y,*d_v0z,*d_v1x,*d_v1y,*d_v1z,*d_v2x,*d_v2y,*d_v2z;
    float*d_rox,*d_roy,*d_roz,*d_rdx,*d_rdy,*d_rdz,*d_idx,*d_idy,*d_idz;
    HitResult*d_hits;unsigned long long*d_st;
    cudaMalloc(&d_v0x,nt*4);cudaMalloc(&d_v0y,nt*4);cudaMalloc(&d_v0z,nt*4);
    cudaMalloc(&d_v1x,nt*4);cudaMalloc(&d_v1y,nt*4);cudaMalloc(&d_v1z,nt*4);
    cudaMalloc(&d_v2x,nt*4);cudaMalloc(&d_v2y,nt*4);cudaMalloc(&d_v2z,nt*4);
    cudaMalloc(&d_rox,numRays*4);cudaMalloc(&d_roy,numRays*4);cudaMalloc(&d_roz,numRays*4);
    cudaMalloc(&d_rdx,numRays*4);cudaMalloc(&d_rdy,numRays*4);cudaMalloc(&d_rdz,numRays*4);
    cudaMalloc(&d_idx,numRays*4);cudaMalloc(&d_idy,numRays*4);cudaMalloc(&d_idz,numRays*4);
    cudaMalloc(&d_hits,numRays*sizeof(HitResult));cudaMalloc(&d_st,16);
    
    cudaMemcpy(d_v0x,h_v0x,nt*4,cudaMemcpyHostToDevice);cudaMemcpy(d_v0y,h_v0y,nt*4,cudaMemcpyHostToDevice);cudaMemcpy(d_v0z,h_v0z,nt*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1x,h_v1x,nt*4,cudaMemcpyHostToDevice);cudaMemcpy(d_v1y,h_v1y,nt*4,cudaMemcpyHostToDevice);cudaMemcpy(d_v1z,h_v1z,nt*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2x,h_v2x,nt*4,cudaMemcpyHostToDevice);cudaMemcpy(d_v2y,h_v2y,nt*4,cudaMemcpyHostToDevice);cudaMemcpy(d_v2z,h_v2z,nt*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_rox,h_rox,numRays*4,cudaMemcpyHostToDevice);cudaMemcpy(d_roy,h_roy,numRays*4,cudaMemcpyHostToDevice);cudaMemcpy(d_roz,h_roz,numRays*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_rdx,h_rdx,numRays*4,cudaMemcpyHostToDevice);cudaMemcpy(d_rdy,h_rdy,numRays*4,cudaMemcpyHostToDevice);cudaMemcpy(d_rdz,h_rdz,numRays*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx,h_idx,numRays*4,cudaMemcpyHostToDevice);cudaMemcpy(d_idy,h_idy,numRays*4,cudaMemcpyHostToDevice);cudaMemcpy(d_idz,h_idz,numRays*4,cudaMemcpyHostToDevice);
    
    int nb=prop.multiProcessorCount*4;
    
    // Warmup
    cudaMemset(d_st,0,16);
    traceV5<<<nb,256>>>(tex,nc,d_v0x,d_v0y,d_v0z,d_v1x,d_v1y,d_v1z,d_v2x,d_v2y,d_v2z,
        d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,d_idx,d_idy,d_idz,d_hits,numRays,d_st);
    cudaDeviceSynchronize();
    
    cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
    float totalMs=0;int runs=5;
    for(int r=0;r<runs;r++){
        cudaMemset(d_st,0,16);cudaEventRecord(t0);
        traceV5<<<nb,256>>>(tex,nc,d_v0x,d_v0y,d_v0z,d_v1x,d_v1y,d_v1z,d_v2x,d_v2y,d_v2z,
            d_rox,d_roy,d_roz,d_rdx,d_rdy,d_rdz,d_idx,d_idy,d_idz,d_hits,numRays,d_st);
        cudaEventRecord(t1);cudaEventSynchronize(t1);
        float ms;cudaEventElapsedTime(&ms,t0,t1);totalMs+=ms;}
    float avg=totalMs/runs;double mr=(double)numRays/(avg/1000.0)/1e6;
    unsigned long long st[2];cudaMemcpy(st,d_st,16,cudaMemcpyDeviceToHost);
    HitResult*hh=(HitResult*)malloc(numRays*sizeof(HitResult));
    cudaMemcpy(hh,d_hits,numRays*sizeof(HitResult),cudaMemcpyDeviceToHost);
    int hc=0;for(int i=0;i<numRays;i++)if(hh[i].triIdx>=0)hc++;free(hh);
    
    printf("  leaf=%2d │ %5d nodes │ %6.1f ms │ %7.1f MR/s │ %5.1f n/r │ %5.1f t/r │ %d hits\n",
        leafSz,nc,avg,mr,(double)st[0]/numRays,(double)st[1]/numRays,hc);
    
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    cudaDestroyTextureObject(tex);cudaFree(d_pk);
    cudaFree(d_v0x);cudaFree(d_v0y);cudaFree(d_v0z);cudaFree(d_v1x);cudaFree(d_v1y);cudaFree(d_v1z);cudaFree(d_v2x);cudaFree(d_v2y);cudaFree(d_v2z);
    cudaFree(d_rox);cudaFree(d_roy);cudaFree(d_roz);cudaFree(d_rdx);cudaFree(d_rdy);cudaFree(d_rdz);cudaFree(d_idx);cudaFree(d_idy);cudaFree(d_idz);
    cudaFree(d_hits);cudaFree(d_st);
    free(h_tris);free(h_to);free(h_nodes);free(tidx);free(h_pk);
    free(h_v0x);free(h_v0y);free(h_v0z);free(h_v1x);free(h_v1y);free(h_v1z);free(h_v2x);free(h_v2y);free(h_v2z);
    free(h_r);free(h_rox);free(h_roy);free(h_roz);free(h_rdx);free(h_rdy);free(h_rdz);free(h_idx);free(h_idy);free(h_idz);
}

int main(){
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  V100 RT v5 — SAH Leaf Size Sweep + SAH Termination                 ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");
    
    cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
    int numRays=4194304;
    
    int leafSizes[]={1,2,4,8,12,16};
    int triCounts[]={100000,500000,1000000};
    
    for(int s=0;s<3;s++){
        int nt=triCounts[s];
        printf("  ═══ %dK triangles | 4M rays ═══\n",nt/1000);
        printf("  leaf   │ nodes  │   time   │  MRays/s  │  n/ray │  t/ray │ hits\n");
        printf("  ───────┼────────┼──────────┼───────────┼────────┼────────┼─────────\n");
        for(int l=0;l<6;l++) runBench(nt,numRays,leafSizes[l],prop);
        printf("\n");
    }
    return 0;
}
