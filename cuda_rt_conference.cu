// Quick test: Cornell box scene for realistic BVH quality
// Same kernel as v6, just with a much better scene

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

struct Tri{float3 v0,v1,v2;};
struct AABB{float3 bmin,bmax;};
struct Hit{float t;int tri;float u,v;};
struct BN{AABB b;int l,r,ts,tc;};

static inline float i2f(int i){float f;memcpy(&f,&i,4);return f;}
__device__ __forceinline__ int d_f2i(float f){int i;memcpy(&i,&f,4);return i;}

AABB unite(const AABB&a,const AABB&b){return{{fminf(a.bmin.x,b.bmin.x),fminf(a.bmin.y,b.bmin.y),fminf(a.bmin.z,b.bmin.z)},{fmaxf(a.bmax.x,b.bmax.x),fmaxf(a.bmax.y,b.bmax.y),fmaxf(a.bmax.z,b.bmax.z)}};}
AABB triBox(const Tri&t){return{{fminf(fminf(t.v0.x,t.v1.x),t.v2.x),fminf(fminf(t.v0.y,t.v1.y),t.v2.y),fminf(fminf(t.v0.z,t.v1.z),t.v2.z)},{fmaxf(fmaxf(t.v0.x,t.v1.x),t.v2.x),fmaxf(fmaxf(t.v0.y,t.v1.y),t.v2.y),fmaxf(fmaxf(t.v0.z,t.v1.z),t.v2.z)}};}
float3 triCen(const Tri&t){return{(t.v0.x+t.v1.x+t.v2.x)/3,(t.v0.y+t.v1.y+t.v2.y)/3,(t.v0.z+t.v1.z+t.v2.z)/3};}
float sa(const AABB&b){float dx=b.bmax.x-b.bmin.x,dy=b.bmax.y-b.bmin.y,dz=b.bmax.z-b.bmin.z;return 2.0f*(dx*dy+dy*dz+dz*dx);}

int buildSAH(BN*n,Tri*t,int*idx,int&nc,int s,int c,int d){
    int ni=nc++;BN&nd=n[ni];AABB bounds=triBox(t[idx[s]]);
    for(int i=1;i<c;i++)bounds=unite(bounds,triBox(t[idx[s+i]]));nd.b=bounds;
    if(c<=LEAF_SIZE||d>26){nd.l=-1;nd.r=-1;nd.ts=s;nd.tc=c;return ni;}
    AABB cb;cb.bmin=cb.bmax=triCen(t[idx[s]]);
    for(int i=1;i<c;i++){float3 cc=triCen(t[idx[s+i]]);
        cb.bmin={fminf(cb.bmin.x,cc.x),fminf(cb.bmin.y,cc.y),fminf(cb.bmin.z,cc.z)};
        cb.bmax={fmaxf(cb.bmax.x,cc.x),fmaxf(cb.bmax.y,cc.y),fmaxf(cb.bmax.z,cc.z)};}
    float bc=FLT_MAX;int ba=-1,bb=-1;float ps=sa(bounds);
    for(int ax=0;ax<3;ax++){
        float amn=ax==0?cb.bmin.x:ax==1?cb.bmin.y:cb.bmin.z,amx=ax==0?cb.bmax.x:ax==1?cb.bmax.y:cb.bmax.z;
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
    float amn=ba==0?cb.bmin.x:ba==1?cb.bmin.y:cb.bmin.z,amx=ba==0?cb.bmax.x:ba==1?cb.bmax.y:cb.bmax.z;float sc=SAH_BINS/(amx-amn);
    int i=s,j=s+c-1;
    while(i<=j){float3 cc=triCen(t[idx[i]]);float cv=ba==0?cc.x:ba==1?cc.y:cc.z;
        int b2=min(max((int)((cv-amn)*sc),0),SAH_BINS-1);if(b2<=bb)i++;else{int tmp=idx[i];idx[i]=idx[j];idx[j]=tmp;j--;}}
    int lc=i-s;if(lc==0)lc=1;if(lc==c)lc=c-1;nd.ts=-1;nd.tc=0;
    nd.l=buildSAH(n,t,idx,nc,s,lc,d+1);nd.r=buildSAH(n,t,idx,nc,s+lc,c-lc,d+1);return ni;
}

__global__ void traceK(
    cudaTextureObject_t tex,int nn,
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    const float*__restrict__ rix,const float*__restrict__ riy,const float*__restrict__ riz,
    Hit*__restrict__ hits,int numRays,unsigned long long*__restrict__ stats)
{
    int gid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    unsigned long long ln=0,lt=0;
    for(int ri=gid;ri<numRays;ri+=stride){
        float ox=rox[ri],oy=roy[ri],oz=roz[ri],dx=rdx[ri],dy=rdy[ri],dz=rdz[ri];
        float ix=rix[ri],iy=riy[ri],iz=riz[ri];
        float hitT=1e30f;int hitTri=-1;float hitU=0,hitV=0;
        int stack[MAX_STACK],sp=0,ni=0;
        while(ni>=0){
            float4 nlo,nhi;
            if(ni<c_bvhN){nlo=c_bvh[ni*2];nhi=c_bvh[ni*2+1];}
            else{nlo=tex1Dfetch<float4>(tex,ni*2);nhi=tex1Dfetch<float4>(tex,ni*2+1);}
            int lc=d_f2i(nlo.w),rd=d_f2i(nhi.w);ln++;
            if(lc<0){
                int ts=rd,tc=(-lc)-1;
                for(int i=0;i<tc;i++){int ti=ts+i;lt++;
                    float v0x=tv0x[ti],v0y=tv0y[ti],v0z=tv0z[ti];
                    float e1x=tv1x[ti]-v0x,e1y=tv1y[ti]-v0y,e1z=tv1z[ti]-v0z;
                    float e2x=tv2x[ti]-v0x,e2y=tv2y[ti]-v0y,e2z=tv2z[ti]-v0z;
                    float hx=dy*e2z-dz*e2y,hy=dz*e2x-dx*e2z,hz=dx*e2y-dy*e2x;
                    float a=e1x*hx+e1y*hy+e1z*hz;if(fabsf(a)<1e-8f)continue;
                    float f=__frcp_rn(a),sx=ox-v0x,sy=oy-v0y,sz=oz-v0z;
                    float u=f*(sx*hx+sy*hy+sz*hz);if(u<0||u>1)continue;
                    float qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x;
                    float v=f*(dx*qx+dy*qy+dz*qz);if(v<0||u+v>1)continue;
                    float t=f*(e2x*qx+e2y*qy+e2z*qz);
                    if(t>0.001f&&t<hitT){hitT=t;hitTri=ti;hitU=u;hitV=v;}}
                ni=sp>0?stack[--sp]:-1;
            }else{
                int rc=rd;
                float4 llo,lhi,rlo,rhi;
                if(lc<c_bvhN){llo=c_bvh[lc*2];lhi=c_bvh[lc*2+1];}else{llo=tex1Dfetch<float4>(tex,lc*2);lhi=tex1Dfetch<float4>(tex,lc*2+1);}
                if(rc<c_bvhN){rlo=c_bvh[rc*2];rhi=c_bvh[rc*2+1];}else{rlo=tex1Dfetch<float4>(tex,rc*2);rhi=tex1Dfetch<float4>(tex,rc*2+1);}
                float ltx1=(llo.x-ox)*ix,ltx2=(lhi.x-ox)*ix,lt0=fminf(ltx1,ltx2),lt1=fmaxf(ltx1,ltx2);
                float lty1=(llo.y-oy)*iy,lty2=(lhi.y-oy)*iy;lt0=fmaxf(lt0,fminf(lty1,lty2));lt1=fminf(lt1,fmaxf(lty1,lty2));
                float ltz1=(llo.z-oz)*iz,ltz2=(lhi.z-oz)*iz;lt0=fmaxf(lt0,fminf(ltz1,ltz2));lt1=fminf(lt1,fmaxf(ltz1,ltz2));
                bool hL=lt1>=fmaxf(lt0,0.0f)&&lt0<=hitT;
                float rtx1=(rlo.x-ox)*ix,rtx2=(rhi.x-ox)*ix,rt0=fminf(rtx1,rtx2),rt1=fmaxf(rtx1,rtx2);
                float rty1=(rlo.y-oy)*iy,rty2=(rhi.y-oy)*iy;rt0=fmaxf(rt0,fminf(rty1,rty2));rt1=fminf(rt1,fmaxf(rty1,rty2));
                float rtz1=(rlo.z-oz)*iz,rtz2=(rhi.z-oz)*iz;rt0=fmaxf(rt0,fminf(rtz1,rtz2));rt1=fminf(rt1,fmaxf(rtz1,rtz2));
                bool hR=rt1>=fmaxf(rt0,0.0f)&&rt0<=hitT;
                ln+=2;
                if(hL&&hR){if(lt0<rt0){stack[sp++]=rc;ni=lc;}else{stack[sp++]=lc;ni=rc;}}
                else if(hL)ni=lc;else if(hR)ni=rc;else ni=sp>0?stack[--sp]:-1;
            }
        }
        hits[ri].t=hitT;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;
    }
    atomicAdd(&stats[0],ln);atomicAdd(&stats[1],lt);
}

// ═══ SCENE: Conference-room-like architecture ═══
// Walls create clean spatial partitioning for BVH
// Objects are spatially distinct → few overlapping AABBs
void addQuad(Tri*t,int&ti,float3 a,float3 b,float3 c,float3 d){
    t[ti++]={a,b,c}; t[ti++]={a,c,d};
}
void addBox(Tri*t,int&ti,float3 mn,float3 mx){
    float3 a={mn.x,mn.y,mn.z},b={mx.x,mn.y,mn.z},c={mx.x,mx.y,mn.z},d={mn.x,mx.y,mn.z};
    float3 e={mn.x,mn.y,mx.z},f={mx.x,mn.y,mx.z},g={mx.x,mx.y,mx.z},h={mn.x,mx.y,mx.z};
    addQuad(t,ti,a,b,c,d);addQuad(t,ti,e,f,g,h);addQuad(t,ti,a,b,f,e);
    addQuad(t,ti,d,c,g,h);addQuad(t,ti,a,d,h,e);addQuad(t,ti,b,c,g,f);
}
// Subdivided quad for higher tri count
void addSubQuad(Tri*t,int&ti,float3 o,float3 ux,float3 uy,int nx,int ny){
    for(int i=0;i<nx;i++)for(int j=0;j<ny;j++){
        float u0=(float)i/nx,u1=(float)(i+1)/nx,v0=(float)j/ny,v1=(float)(j+1)/ny;
        float3 a={o.x+ux.x*u0+uy.x*v0,o.y+ux.y*u0+uy.y*v0,o.z+ux.z*u0+uy.z*v0};
        float3 b={o.x+ux.x*u1+uy.x*v0,o.y+ux.y*u1+uy.y*v0,o.z+ux.z*u1+uy.z*v0};
        float3 c={o.x+ux.x*u1+uy.x*v1,o.y+ux.y*u1+uy.y*v1,o.z+ux.z*u1+uy.z*v1};
        float3 d={o.x+ux.x*u0+uy.x*v1,o.y+ux.y*u0+uy.y*v1,o.z+ux.z*u0+uy.z*v1};
        t[ti++]={a,b,c};t[ti++]={a,c,d};
    }
}

int genConference(Tri*t,int maxTris){
    int ti=0;
    // Room: 20×10×15 (like a conference room)
    float W=10,H=5,D=7.5f;
    int subdiv=(int)sqrtf((float)maxTris/60); // subdivisions per wall
    if(subdiv<2)subdiv=2;if(subdiv>200)subdiv=200;
    
    // Floor (y=0)
    addSubQuad(t,ti,{-W,0,-D},{2*W,0,0},{0,0,2*D},subdiv,subdiv);
    // Ceiling (y=H)
    addSubQuad(t,ti,{-W,H,-D},{2*W,0,0},{0,0,2*D},subdiv,subdiv);
    // Back wall (z=-D)
    addSubQuad(t,ti,{-W,0,-D},{2*W,0,0},{0,H,0},subdiv,subdiv/2);
    // Front wall (z=D)
    addSubQuad(t,ti,{-W,0,D},{2*W,0,0},{0,H,0},subdiv,subdiv/2);
    // Left wall (x=-W)
    addSubQuad(t,ti,{-W,0,-D},{0,0,2*D},{0,H,0},subdiv,subdiv/2);
    // Right wall (x=W)
    addSubQuad(t,ti,{W,0,-D},{0,0,2*D},{0,H,0},subdiv,subdiv/2);
    
    // Tables (boxes)
    srand(42);
    int numTables = maxTris > 50000 ? 20 : 8;
    for(int i=0;i<numTables&&ti+12<maxTris;i++){
        float tx=((float)rand()/RAND_MAX)*16-8;
        float tz=((float)rand()/RAND_MAX)*12-6;
        addBox(t,ti,{tx-1.0f,0.7f,tz-0.5f},{tx+1.0f,0.8f,tz+0.5f}); // table top
        addBox(t,ti,{tx-0.9f,0,tz-0.05f},{tx-0.8f,0.7f,tz+0.05f}); // leg
        addBox(t,ti,{tx+0.8f,0,tz-0.05f},{tx+0.9f,0.7f,tz+0.05f}); // leg
    }
    // Chairs
    int numChairs = maxTris > 50000 ? 40 : 16;
    for(int i=0;i<numChairs&&ti+12<maxTris;i++){
        float cx=((float)rand()/RAND_MAX)*18-9;
        float cz=((float)rand()/RAND_MAX)*14-7;
        addBox(t,ti,{cx-0.25f,0.4f,cz-0.25f},{cx+0.25f,0.45f,cz+0.25f}); // seat
        addBox(t,ti,{cx-0.25f,0.45f,cz-0.25f},{cx+0.25f,0.9f,cz-0.2f}); // backrest
    }
    
    // Fill remaining with small decorations
    while(ti+2 < maxTris){
        float cx=((float)rand()/RAND_MAX)*16-8;
        float cy=0.8f+((float)rand()/RAND_MAX)*0.3f;
        float cz=((float)rand()/RAND_MAX)*12-6;
        float s=0.05f+((float)rand()/RAND_MAX)*0.1f;
        t[ti].v0={cx-s,cy,cz-s};t[ti].v1={cx+s,cy,cz+s};t[ti].v2={cx,cy+s*2,cz};ti++;
        t[ti].v0={cx-s,cy,cz+s};t[ti].v1={cx+s,cy,cz-s};t[ti].v2={cx,cy+s*2,cz};ti++;
    }
    return ti;
}

unsigned int xBits(unsigned int v){v=(v*0x00010001u)&0xFF0000FFu;v=(v*0x00000101u)&0x0F00F00Fu;v=(v*0x00000011u)&0xC30C30C3u;v=(v*0x00000005u)&0x49249249u;return v;}
struct RayAoS{float3 o,d,id;};

void genPrimary(RayAoS*r,int n){
    int w=(int)sqrtf((float)n);
    for(int i=0;i<n;i++){
        int px=i%w,py=i/w;
        float u=(2.0f*px/w-1.0f)*1.2f,v=(2.0f*py/(n/w)-1.0f)*0.6f;
        r[i].o={0,2.5f,12}; // camera at front of room
        float3 d={u,-v,-1.5f};float l=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);
        d.x/=l;d.y/=l;d.z/=l;r[i].d=d;r[i].id={1.0f/d.x,1.0f/d.y,1.0f/d.z};
    }
}

int main(){
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  V100 RT — Conference Room Scene (realistic BVH quality)     ║\n");
    printf("║  Walls+tables+chairs → tight AABBs, low node/ray count      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    
    cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
    int numRays=4194304;
    int triTargets[]={10000,50000,100000,200000,500000};
    
    printf("  ┌──────────┬─────────┬──────────┬────────┬────────┬──────────┐\n");
    printf("  │ Tris     │ Time ms │ MRays/s  │ n/ray  │ t/ray  │ Hit%%     │\n");
    printf("  ├──────────┼─────────┼──────────┼────────┼────────┼──────────┤\n");
    
    for(int s=0;s<5;s++){
        int maxTris=triTargets[s];
        Tri*h_tris=(Tri*)malloc(maxTris*sizeof(Tri));
        int nt=genConference(h_tris,maxTris);
        
        BN*h_nodes=(BN*)calloc(nt*2,sizeof(BN));int*tidx=(int*)malloc(nt*4);
        for(int i=0;i<nt;i++)tidx[i]=i;int nc=0;
        buildSAH(h_nodes,h_tris,tidx,nc,0,nt,0);
        
        Tri*h_to=(Tri*)malloc(nt*sizeof(Tri));for(int i=0;i<nt;i++)h_to[i]=h_tris[tidx[i]];
        
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
        for(int i=0;i<nt;i++){h_v[0][i]=h_to[i].v0.x;h_v[1][i]=h_to[i].v0.y;h_v[2][i]=h_to[i].v0.z;
            h_v[3][i]=h_to[i].v1.x;h_v[4][i]=h_to[i].v1.y;h_v[5][i]=h_to[i].v1.z;
            h_v[6][i]=h_to[i].v2.x;h_v[7][i]=h_to[i].v2.y;h_v[8][i]=h_to[i].v2.z;}
        
        RayAoS*h_r=(RayAoS*)malloc(numRays*sizeof(RayAoS));genPrimary(h_r,numRays);
        float*h_ray[9];for(int j=0;j<9;j++)h_ray[j]=(float*)malloc(numRays*4);
        for(int i=0;i<numRays;i++){h_ray[0][i]=h_r[i].o.x;h_ray[1][i]=h_r[i].o.y;h_ray[2][i]=h_r[i].o.z;
            h_ray[3][i]=h_r[i].d.x;h_ray[4][i]=h_r[i].d.y;h_ray[5][i]=h_r[i].d.z;
            h_ray[6][i]=h_r[i].id.x;h_ray[7][i]=h_r[i].id.y;h_ray[8][i]=h_r[i].id.z;}
        
        float*d_v[9],*d_ray[9];Hit*d_h;unsigned long long*d_st;
        for(int j=0;j<9;j++){cudaMalloc(&d_v[j],nt*4);cudaMemcpy(d_v[j],h_v[j],nt*4,cudaMemcpyHostToDevice);}
        for(int j=0;j<9;j++){cudaMalloc(&d_ray[j],numRays*4);cudaMemcpy(d_ray[j],h_ray[j],numRays*4,cudaMemcpyHostToDevice);}
        cudaMalloc(&d_h,numRays*sizeof(Hit));cudaMalloc(&d_st,16);
        
        int nb=prop.multiProcessorCount*8;
        cudaMemset(d_st,0,16);
        traceK<<<nb,256>>>(tex,nc,d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
            d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],d_h,numRays,d_st);
        cudaDeviceSynchronize();
        
        cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);float totalMs=0;
        for(int r=0;r<10;r++){cudaMemset(d_st,0,16);cudaEventRecord(t0);
            traceK<<<nb,256>>>(tex,nc,d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
                d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],d_h,numRays,d_st);
            cudaEventRecord(t1);cudaEventSynchronize(t1);float ms;cudaEventElapsedTime(&ms,t0,t1);totalMs+=ms;}
        float avg=totalMs/10;double mr=(double)numRays/(avg/1000.0)/1e6;
        unsigned long long st[2];cudaMemcpy(st,d_st,16,cudaMemcpyDeviceToHost);
        Hit*hh=(Hit*)malloc(numRays*sizeof(Hit));cudaMemcpy(hh,d_h,numRays*sizeof(Hit),cudaMemcpyDeviceToHost);
        int hc=0;for(int i=0;i<numRays;i++)if(hh[i].tri>=0)hc++;free(hh);
        
        printf("  │ %6dK  │ %6.1f  │ %7.1f  │ %5.1f  │ %5.1f  │ %5.1f%%   │\n",
            nt/1000,avg,mr,(double)st[0]/numRays,(double)st[1]/numRays,100.0*hc/numRays);
        
        cudaEventDestroy(t0);cudaEventDestroy(t1);cudaDestroyTextureObject(tex);cudaFree(d_pk);
        for(int j=0;j<9;j++){cudaFree(d_v[j]);cudaFree(d_ray[j]);free(h_v[j]);free(h_ray[j]);}
        cudaFree(d_h);cudaFree(d_st);free(h_tris);free(h_to);free(h_nodes);free(tidx);free(h_pk);free(h_r);
    }
    
    printf("  └──────────┴─────────┴──────────┴────────┴────────┴──────────┘\n\n");
    printf("  Aila-Laine GTX285 Conference(282K): 142 MRays/s primary\n");
    printf("  Aila-Laine GTX680 Conference(282K): 432 MRays/s primary\n");
    printf("  V100 has 4.7× GTX680 bandwidth → target ~2000 MRays/s\n");
    
    return 0;
}
