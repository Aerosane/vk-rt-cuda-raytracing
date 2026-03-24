// V7: Copy-engine overlap + treelet reordering + direction-octant coherence
// Goals: Push past 2000 MRays/s on conference scenes
// Key additions:
// 1. CUDA Streams: double-buffer rays — DMA copy overlaps with compute
// 2. Treelet BVH reorder: DFS order → cache-line sequential parent→child
// 3. Direction octant sorting: rays in same warp traverse same BVH path
// 4. Warp-persistent while-while with speculative traversal

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <algorithm>

#define SAH_BINS 16
#define LEAF_SIZE 4
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
    if(c<=LEAF_SIZE||d>28){nd.l=-1;nd.r=-1;nd.ts=s;nd.tc=c;return ni;}
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

// ═══ TREELET REORDER: DFS-order BVH nodes for cache coherence ═══
// Parent at index i → children at i+1 (left) and stored right
// Maximizes L2/texture cache hit rate
void treeReorder(BN*src,int nc,BN*dst,int*remap){
    int*stack=(int*)malloc(nc*4);int sp=0,out=0;stack[sp++]=0;
    while(sp>0){int i=stack[--sp];remap[i]=out;dst[out++]=src[i];
        if(src[i].l>=0){stack[sp++]=src[i].r;stack[sp++]=src[i].l;}} // left first (DFS)
    // fix child pointers
    for(int i=0;i<nc;i++){if(dst[i].l>=0){dst[i].l=remap[dst[i].l];dst[i].r=remap[dst[i].r];}}
    free(stack);
}

// ═══ MAIN KERNEL: while-while + register stack + all HW ═══
__global__ void __launch_bounds__(256,4) traceK(
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
        float ox=rox[ri],oy=roy[ri],oz=roz[ri];
        float dx=rdx[ri],dy=rdy[ri],dz=rdz[ri];
        float ix=rix[ri],iy=riy[ri],iz=riz[ri];
        float hitT=1e30f;int hitTri=-1;float hitU=0,hitV=0;

        // Register stack (INT32 pipe, concurrent with FP32)
        int stack[MAX_STACK],sp=0;
        int nodeIdx=0;

        // WHILE-WHILE: outer=traversal, inner=leaf intersection
        while(nodeIdx>=0){
            // ── INNER WHILE: consume leaves ──
            while(nodeIdx>=0){
                float4 nlo,nhi;
                if(nodeIdx<c_bvhN){nlo=c_bvh[nodeIdx*2];nhi=c_bvh[nodeIdx*2+1];}
                else{nlo=tex1Dfetch<float4>(tex,nodeIdx*2);nhi=tex1Dfetch<float4>(tex,nodeIdx*2+1);}
                int lc=d_f2i(nlo.w);
                if(lc>=0) break; // internal node → go to outer
                int ts=d_f2i(nhi.w),tc=(-lc)-1;ln++;
                for(int i=0;i<tc;i++){int ti=ts+i;lt++;
                    float v0x=tv0x[ti],v0y=tv0y[ti],v0z=tv0z[ti];
                    float e1x=tv1x[ti]-v0x,e1y=tv1y[ti]-v0y,e1z=tv1z[ti]-v0z;
                    float e2x=tv2x[ti]-v0x,e2y=tv2y[ti]-v0y,e2z=tv2z[ti]-v0z;
                    float hx=dy*e2z-dz*e2y,hy=dz*e2x-dx*e2z,hz=dx*e2y-dy*e2x;
                    float a=e1x*hx+e1y*hy+e1z*hz;if(fabsf(a)<1e-8f)continue;
                    float f=__frcp_rn(a);float sx=ox-v0x,sy=oy-v0y,sz=oz-v0z;
                    float u=f*(sx*hx+sy*hy+sz*hz);if(u<0.0f||u>1.0f)continue;
                    float qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x;
                    float v=f*(dx*qx+dy*qy+dz*qz);if(v<0.0f||u+v>1.0f)continue;
                    float tt=f*(e2x*qx+e2y*qy+e2z*qz);
                    if(tt>0.001f&&tt<hitT){hitT=tt;hitTri=ti;hitU=u;hitV=v;}}
                nodeIdx=sp>0?stack[--sp]:-1;
            }
            if(nodeIdx<0) break;

            // ── OUTER WHILE: traverse internal nodes ──
            while(nodeIdx>=0){
                float4 nlo,nhi;
                if(nodeIdx<c_bvhN){nlo=c_bvh[nodeIdx*2];nhi=c_bvh[nodeIdx*2+1];}
                else{nlo=tex1Dfetch<float4>(tex,nodeIdx*2);nhi=tex1Dfetch<float4>(tex,nodeIdx*2+1);}
                int lc=d_f2i(nlo.w),rc=d_f2i(nhi.w);ln++;
                if(lc<0){nodeIdx=nodeIdx;break;} // leaf → go to inner

                // Fetch both children (speculative — read before testing)
                float4 llo,lhi,rlo,rhi;
                if(lc<c_bvhN){llo=c_bvh[lc*2];lhi=c_bvh[lc*2+1];}
                else{llo=tex1Dfetch<float4>(tex,lc*2);lhi=tex1Dfetch<float4>(tex,lc*2+1);}
                if(rc<c_bvhN){rlo=c_bvh[rc*2];rhi=c_bvh[rc*2+1];}
                else{rlo=tex1Dfetch<float4>(tex,rc*2);rhi=tex1Dfetch<float4>(tex,rc*2+1);}

                // AABB slab tests (FP32 + SFU fminf/fmaxf)
                float lt1x=(llo.x-ox)*ix,lt2x=(lhi.x-ox)*ix;
                float ltmn=fminf(lt1x,lt2x),ltmx=fmaxf(lt1x,lt2x);
                float lt1y=(llo.y-oy)*iy,lt2y=(lhi.y-oy)*iy;
                ltmn=fmaxf(ltmn,fminf(lt1y,lt2y));ltmx=fminf(ltmx,fmaxf(lt1y,lt2y));
                float lt1z=(llo.z-oz)*iz,lt2z=(lhi.z-oz)*iz;
                ltmn=fmaxf(ltmn,fminf(lt1z,lt2z));ltmx=fminf(ltmx,fmaxf(lt1z,lt2z));
                bool hL=ltmx>=fmaxf(ltmn,0.0f)&&ltmn<hitT;

                float rt1x=(rlo.x-ox)*ix,rt2x=(rhi.x-ox)*ix;
                float rtmn=fminf(rt1x,rt2x),rtmx=fmaxf(rt1x,rt2x);
                float rt1y=(rlo.y-oy)*iy,rt2y=(rhi.y-oy)*iy;
                rtmn=fmaxf(rtmn,fminf(rt1y,rt2y));rtmx=fminf(rtmx,fmaxf(rt1y,rt2y));
                float rt1z=(rlo.z-oz)*iz,rt2z=(rhi.z-oz)*iz;
                rtmn=fmaxf(rtmn,fminf(rt1z,rt2z));rtmx=fminf(rtmx,fmaxf(rt1z,rt2z));
                bool hR=rtmx>=fmaxf(rtmn,0.0f)&&rtmn<hitT;

                ln+=2;
                if(hL&&hR){
                    if(ltmn<rtmn){stack[sp++]=rc;nodeIdx=lc;}
                    else{stack[sp++]=lc;nodeIdx=rc;}
                }else if(hL){nodeIdx=lc;}
                else if(hR){nodeIdx=rc;}
                else{nodeIdx=sp>0?stack[--sp]:-1;}
            }
        }
        hits[ri].t=hitT;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;
    }
    atomicAdd(&stats[0],ln);atomicAdd(&stats[1],lt);
}

// ═══ SCENE GENERATORS ═══
void addQuad(Tri*t,int&ti,float3 a,float3 b,float3 c,float3 d){t[ti++]={a,b,c};t[ti++]={a,c,d};}
void addBox(Tri*t,int&ti,float3 mn,float3 mx){
    float3 a={mn.x,mn.y,mn.z},b={mx.x,mn.y,mn.z},c={mx.x,mx.y,mn.z},d={mn.x,mx.y,mn.z};
    float3 e={mn.x,mn.y,mx.z},f={mx.x,mn.y,mx.z},g={mx.x,mx.y,mx.z},h={mn.x,mx.y,mx.z};
    addQuad(t,ti,a,b,c,d);addQuad(t,ti,e,f,g,h);addQuad(t,ti,a,b,f,e);
    addQuad(t,ti,d,c,g,h);addQuad(t,ti,a,d,h,e);addQuad(t,ti,b,c,g,f);
}
void addSubQuad(Tri*t,int&ti,float3 o,float3 ux,float3 uy,int nx,int ny){
    for(int i=0;i<nx;i++)for(int j=0;j<ny;j++){
        float u0=(float)i/nx,u1=(float)(i+1)/nx,v0=(float)j/ny,v1=(float)(j+1)/ny;
        float3 a={o.x+ux.x*u0+uy.x*v0,o.y+ux.y*u0+uy.y*v0,o.z+ux.z*u0+uy.z*v0};
        float3 b={o.x+ux.x*u1+uy.x*v0,o.y+ux.y*u1+uy.y*v0,o.z+ux.z*u1+uy.z*v0};
        float3 c={o.x+ux.x*u1+uy.x*v1,o.y+ux.y*u1+uy.y*v1,o.z+ux.z*u1+uy.z*v1};
        float3 d={o.x+ux.x*u0+uy.x*v1,o.y+ux.y*u0+uy.y*v1,o.z+ux.z*u0+uy.z*v1};
        t[ti++]={a,b,c};t[ti++]={a,c,d};}}

int genConference(Tri*t,int maxTris){
    int ti=0;float W=10,H=5,D=7.5f;
    int subdiv=(int)sqrtf((float)maxTris/60);
    if(subdiv<2)subdiv=2;if(subdiv>200)subdiv=200;
    addSubQuad(t,ti,{-W,0,-D},{2*W,0,0},{0,0,2*D},subdiv,subdiv);
    addSubQuad(t,ti,{-W,H,-D},{2*W,0,0},{0,0,2*D},subdiv,subdiv);
    addSubQuad(t,ti,{-W,0,-D},{2*W,0,0},{0,H,0},subdiv,subdiv/2);
    addSubQuad(t,ti,{-W,0,D},{2*W,0,0},{0,H,0},subdiv,subdiv/2);
    addSubQuad(t,ti,{-W,0,-D},{0,0,2*D},{0,H,0},subdiv,subdiv/2);
    addSubQuad(t,ti,{W,0,-D},{0,0,2*D},{0,H,0},subdiv,subdiv/2);
    srand(42);
    int numTables=maxTris>50000?20:8;
    for(int i=0;i<numTables&&ti+12<maxTris;i++){
        float tx=((float)rand()/RAND_MAX)*16-8,tz=((float)rand()/RAND_MAX)*12-6;
        addBox(t,ti,{tx-1.0f,0.7f,tz-0.5f},{tx+1.0f,0.8f,tz+0.5f});
        addBox(t,ti,{tx-0.9f,0,tz-0.05f},{tx-0.8f,0.7f,tz+0.05f});
        addBox(t,ti,{tx+0.8f,0,tz-0.05f},{tx+0.9f,0.7f,tz+0.05f});}
    int numChairs=maxTris>50000?40:16;
    for(int i=0;i<numChairs&&ti+12<maxTris;i++){
        float cx=((float)rand()/RAND_MAX)*18-9,cz=((float)rand()/RAND_MAX)*14-7;
        addBox(t,ti,{cx-0.25f,0.4f,cz-0.25f},{cx+0.25f,0.45f,cz+0.25f});
        addBox(t,ti,{cx-0.25f,0.45f,cz-0.25f},{cx+0.25f,0.9f,cz-0.2f});}
    while(ti+2<maxTris){
        float cx=((float)rand()/RAND_MAX)*16-8,cy=0.8f+((float)rand()/RAND_MAX)*0.3f;
        float cz=((float)rand()/RAND_MAX)*12-6,s=0.05f+((float)rand()/RAND_MAX)*0.1f;
        t[ti].v0={cx-s,cy,cz-s};t[ti].v1={cx+s,cy,cz+s};t[ti].v2={cx,cy+s*2,cz};ti++;
        t[ti].v0={cx-s,cy,cz+s};t[ti].v1={cx+s,cy,cz-s};t[ti].v2={cx,cy+s*2,cz};ti++;}
    return ti;
}

// ═══ DIRECTION OCTANT SORTING ═══
// Sort rays by direction octant (sign bits of dx,dy,dz) → warp coherence
// All rays in same warp traverse BVH in same direction → same child order
struct RayAoS{float3 o,d,id;int origIdx;};

int dirOctant(float3 d){return (d.x<0?4:0)|(d.y<0?2:0)|(d.z<0?1:0);}

void genPrimaryCoherent(RayAoS*r,int n){
    int w=(int)sqrtf((float)n);
    for(int i=0;i<n;i++){
        int px=i%w,py=i/w;
        float u=(2.0f*px/w-1.0f)*1.2f,v=(2.0f*py/(n/w)-1.0f)*0.6f;
        r[i].o={0,2.5f,12};
        float3 d={u,-v,-1.5f};float l=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);
        d.x/=l;d.y/=l;d.z/=l;r[i].d=d;r[i].id={1.0f/d.x,1.0f/d.y,1.0f/d.z};
        r[i].origIdx=i;
    }
    // Sort by Morton code on screen pixel → spatial coherence within warps
    // Already in scanline order for primary → good enough, but let's sort by
    // 16×16 tiles for even better warp coherence
    int tw=16;
    std::sort(r,r+n,[w,tw](const RayAoS&a,const RayAoS&b){
        int ax=a.origIdx%w,ay=a.origIdx/w,bx=b.origIdx%w,by=b.origIdx/w;
        int atx=ax/tw,aty=ay/tw,btx=bx/tw,bty=by/tw;
        if(aty!=bty)return aty<bty;if(atx!=btx)return atx<btx;
        // Within tile: Z-curve
        return a.origIdx<b.origIdx;
    });
}

void genDiffuseCoherent(RayAoS*r,int n){
    srand(123);
    for(int i=0;i<n;i++){
        r[i].o={((float)rand()/RAND_MAX)*18-9,((float)rand()/RAND_MAX)*4+0.1f,((float)rand()/RAND_MAX)*13-6.5f};
        float u1=(float)rand()/RAND_MAX,u2=(float)rand()/RAND_MAX;
        float phi=2*M_PI*u1,ct=sqrtf(u2),st=sqrtf(1-u2);
        float3 d={st*cosf(phi),ct,st*sinf(phi)};
        r[i].d=d;r[i].id={1.0f/d.x,1.0f/d.y,1.0f/d.z};r[i].origIdx=i;
    }
    // Sort by direction octant then by origin morton → coherent warps
    std::sort(r,r+n,[](const RayAoS&a,const RayAoS&b){
        int oa=dirOctant(a.d),ob=dirOctant(b.d);
        if(oa!=ob) return oa<ob;
        // within same octant, sort by origin proximity
        int ax=(int)((a.o.x+10)*10),ay=(int)(a.o.y*20),az=(int)((a.o.z+7)*10);
        int bx=(int)((b.o.x+10)*10),by=(int)(b.o.y*20),bz=(int)((b.o.z+7)*10);
        unsigned int am=0,bm=0;
        for(int i=0;i<10;i++){am|=((ax>>i)&1)<<(3*i)|((ay>>i)&1)<<(3*i+1)|((az>>i)&1)<<(3*i+2);
            bm|=((bx>>i)&1)<<(3*i)|((by>>i)&1)<<(3*i+1)|((bz>>i)&1)<<(3*i+2);}
        return am<bm;
    });
}

void runBench(const char*label,Tri*h_tris,int nt,RayAoS*h_rays,int numRays,
              cudaDeviceProp&prop,bool isTreelet)
{
    BN*h_nodes=(BN*)calloc(nt*2,sizeof(BN));int*tidx=(int*)malloc(nt*4);
    for(int i=0;i<nt;i++)tidx[i]=i;int nc=0;
    buildSAH(h_nodes,h_tris,tidx,nc,0,nt,0);

    // Treelet reorder (DFS cache-coherent layout)
    BN*h_ordered;int*remap;
    if(isTreelet){
        h_ordered=(BN*)malloc(nc*sizeof(BN));remap=(int*)malloc(nc*4);
        treeReorder(h_nodes,nc,h_ordered,remap);
    }else{h_ordered=h_nodes;remap=NULL;}

    // Build SoA tris in BVH order
    Tri*h_to=(Tri*)malloc(nt*sizeof(Tri));for(int i=0;i<nt;i++)h_to[i]=h_tris[tidx[i]];

    // Pack BVH nodes as float4 pairs
    float4*h_pk=(float4*)malloc(nc*2*sizeof(float4));
    for(int i=0;i<nc;i++){BN&nd=h_ordered[i];int lv,rv;
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

    // ═══ COPY ENGINE: Use pinned memory + 2 streams for overlap ═══
    float*h_ray[9];
    for(int j=0;j<9;j++){cudaMallocHost(&h_ray[j],numRays*4);}
    for(int i=0;i<numRays;i++){h_ray[0][i]=h_rays[i].o.x;h_ray[1][i]=h_rays[i].o.y;h_ray[2][i]=h_rays[i].o.z;
        h_ray[3][i]=h_rays[i].d.x;h_ray[4][i]=h_rays[i].d.y;h_ray[5][i]=h_rays[i].d.z;
        h_ray[6][i]=h_rays[i].id.x;h_ray[7][i]=h_rays[i].id.y;h_ray[8][i]=h_rays[i].id.z;}

    // Double-buffer: 2 halves of rays, 2 compute streams, 1 copy stream
    int half=numRays/2;
    cudaStream_t sComp[2],sCopy;
    cudaStreamCreate(&sComp[0]);cudaStreamCreate(&sComp[1]);cudaStreamCreate(&sCopy);

    float*d_v[9],*d_ray[2][9];Hit*d_h[2];unsigned long long*d_st;
    for(int j=0;j<9;j++){cudaMalloc(&d_v[j],nt*4);cudaMemcpy(d_v[j],h_v[j],nt*4,cudaMemcpyHostToDevice);}
    for(int b=0;b<2;b++){for(int j=0;j<9;j++)cudaMalloc(&d_ray[b][j],half*4);cudaMalloc(&d_h[b],half*sizeof(Hit));}
    cudaMalloc(&d_st,16);

    int nb=prop.multiProcessorCount*4; // per half

    // ═══ BENCHMARK: single stream (baseline) ═══
    float*d_rayFull[9];Hit*d_hFull;
    for(int j=0;j<9;j++){cudaMalloc(&d_rayFull[j],numRays*4);cudaMemcpy(d_rayFull[j],h_ray[j],numRays*4,cudaMemcpyHostToDevice);}
    cudaMalloc(&d_hFull,numRays*sizeof(Hit));

    // warmup
    cudaMemset(d_st,0,16);
    traceK<<<nb*2,256>>>(tex,nc,d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
        d_rayFull[0],d_rayFull[1],d_rayFull[2],d_rayFull[3],d_rayFull[4],d_rayFull[5],
        d_rayFull[6],d_rayFull[7],d_rayFull[8],d_hFull,numRays,d_st);
    cudaDeviceSynchronize();

    cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);float totalMs=0;
    for(int r=0;r<10;r++){
        cudaMemset(d_st,0,16);cudaEventRecord(t0);
        traceK<<<nb*2,256>>>(tex,nc,d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
            d_rayFull[0],d_rayFull[1],d_rayFull[2],d_rayFull[3],d_rayFull[4],d_rayFull[5],
            d_rayFull[6],d_rayFull[7],d_rayFull[8],d_hFull,numRays,d_st);
        cudaEventRecord(t1);cudaEventSynchronize(t1);float ms;cudaEventElapsedTime(&ms,t0,t1);totalMs+=ms;}
    float avgSingle=totalMs/10;double mrSingle=(double)numRays/(avgSingle/1000.0)/1e6;

    // stats
    unsigned long long st[2];cudaMemcpy(st,d_st,16,cudaMemcpyDeviceToHost);
    Hit*hh=(Hit*)malloc(numRays*sizeof(Hit));cudaMemcpy(hh,d_hFull,numRays*sizeof(Hit),cudaMemcpyDeviceToHost);
    int hc=0;for(int i=0;i<numRays;i++)if(hh[i].tri>=0)hc++;free(hh);

    // ═══ BENCHMARK: double-buffer with copy engine overlap ═══
    totalMs=0;
    for(int r=0;r<10;r++){
        cudaMemset(d_st,0,16);cudaEventRecord(t0);

        // Copy first half
        for(int j=0;j<9;j++)cudaMemcpyAsync(d_ray[0][j],h_ray[j],half*4,cudaMemcpyHostToDevice,sCopy);
        cudaStreamSynchronize(sCopy);

        // Launch compute on first half, overlap copy of second half
        traceK<<<nb,256,0,sComp[0]>>>(tex,nc,d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
            d_ray[0][0],d_ray[0][1],d_ray[0][2],d_ray[0][3],d_ray[0][4],d_ray[0][5],
            d_ray[0][6],d_ray[0][7],d_ray[0][8],d_h[0],half,d_st);
        for(int j=0;j<9;j++)cudaMemcpyAsync(d_ray[1][j],h_ray[j]+half,half*4,cudaMemcpyHostToDevice,sCopy);

        // Wait for copy, launch second half, overlap result writeback
        cudaStreamSynchronize(sCopy);
        traceK<<<nb,256,0,sComp[1]>>>(tex,nc,d_v[0],d_v[1],d_v[2],d_v[3],d_v[4],d_v[5],d_v[6],d_v[7],d_v[8],
            d_ray[1][0],d_ray[1][1],d_ray[1][2],d_ray[1][3],d_ray[1][4],d_ray[1][5],
            d_ray[1][6],d_ray[1][7],d_ray[1][8],d_h[1],half,d_st);

        cudaStreamSynchronize(sComp[0]);cudaStreamSynchronize(sComp[1]);
        cudaEventRecord(t1);cudaEventSynchronize(t1);float ms;cudaEventElapsedTime(&ms,t0,t1);totalMs+=ms;
    }
    float avgDouble=totalMs/10;double mrDouble=(double)numRays/(avgDouble/1000.0)/1e6;

    printf("  │ %6dK  │ %-8s│ %6.1f  │ %7.1f  │ %6.1f  │ %7.1f  │ %5.1f  │ %5.1f  │ %5.1f%%   │\n",
        nt/1000,label,avgSingle,mrSingle,avgDouble,mrDouble,(double)st[0]/numRays,(double)st[1]/numRays,100.0*hc/numRays);

    // cleanup
    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaDestroyTextureObject(tex);cudaFree(d_pk);
    for(int j=0;j<9;j++){cudaFree(d_v[j]);cudaFree(d_rayFull[j]);free(h_v[j]);cudaFreeHost(h_ray[j]);}
    cudaFree(d_hFull);cudaFree(d_st);
    for(int b=0;b<2;b++){for(int j=0;j<9;j++)cudaFree(d_ray[b][j]);cudaFree(d_h[b]);}
    cudaStreamDestroy(sComp[0]);cudaStreamDestroy(sComp[1]);cudaStreamDestroy(sCopy);
    free(h_nodes);if(isTreelet){free(h_ordered);free(remap);}free(tidx);free(h_pk);free(h_to);
}

int main(){
    printf("╔═══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  V100 RT Engine v7 — Copy Engines + Treelet Reorder + Octant Coherence       ║\n");
    printf("║  DMA overlap (7 copy engines) + DFS node layout + direction-sorted warps      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
    printf("GPU: %s | %d SMs | %d MB L2 | asyncEngines: %d\n\n",
        prop.name,prop.multiProcessorCount,prop.l2CacheSize/1024/1024,prop.asyncEngineCount);

    int numRays=4194304;
    int triTargets[]={50000,100000,200000,500000};

    printf("  ┌──────────┬─────────┬─────────┬──────────┬─────────┬──────────┬────────┬────────┬──────────┐\n");
    printf("  │ Tris     │ Rays    │ Single  │ MR/s(1)  │ 2-buf   │ MR/s(2)  │ n/ray  │ t/ray  │ Hit%%     │\n");
    printf("  ├──────────┼─────────┼─────────┼──────────┼─────────┼──────────┼────────┼────────┼──────────┤\n");

    for(int s=0;s<4;s++){
        int maxTris=triTargets[s];
        Tri*h_tris=(Tri*)malloc(maxTris*sizeof(Tri));
        int nt=genConference(h_tris,maxTris);

        // Primary rays with tile-sorted coherence
        RayAoS*h_rays=(RayAoS*)malloc(numRays*sizeof(RayAoS));
        genPrimaryCoherent(h_rays,numRays);
        runBench("Primary",h_tris,nt,h_rays,numRays,prop,true);

        // Diffuse rays with octant+morton coherence
        genDiffuseCoherent(h_rays,numRays);
        runBench("Diffuse",h_tris,nt,h_rays,numRays,prop,true);

        free(h_tris);free(h_rays);
    }

    printf("  └──────────┴─────────┴─────────┴──────────┴─────────┴──────────┴────────┴────────┴──────────┘\n\n");
    printf("  v6 baseline: 686 MR/s (Str 100K Primary) | Target: >1300 MR/s\n");

    return 0;
}
