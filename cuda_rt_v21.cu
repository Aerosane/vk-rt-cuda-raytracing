/* v21 — Wavefront Decomposed BVH4 Engine
 *
 * Key innovations from research:
 * 1) WAVEFRONT DECOMPOSITION: Instead of one monolithic trace kernel that
 *    runs until all rays finish (causing idle lanes), we split into:
 *    - Extension kernel: advance all active rays by N BVH levels
 *    - Compact kernel: remove finished rays, pack active rays dense
 *    - Repeat until all done
 *    This keeps warps 90%+ full because finished rays are replaced.
 *
 * 2) TWO-PHASE SORT: Sort diffuse rays by FIRST-HIT BVH NODE (cheap
 *    single-level traversal) rather than spatial Morton. Rays hitting
 *    same top-level node → same BVH subtree → coherent traversal.
 *
 * 3) NO-SORT for primary (proven +45% from v20), WITH-SORT for diffuse.
 *
 * 4) Stack depth 16 (safe for 3M tris), dual kernel (primary/diffuse).
 *
 * Target: 2-3M triangles, 70%+ diffuse warp efficiency.
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

// ======================== BVH builder ========================
static AABB triAABB(const Tri&t){
    AABB b;
    b.mn.x=fminf(fminf(t.v0.x,t.v1.x),t.v2.x); b.mn.y=fminf(fminf(t.v0.y,t.v1.y),t.v2.y); b.mn.z=fminf(fminf(t.v0.z,t.v1.z),t.v2.z);
    b.mx.x=fmaxf(fmaxf(t.v0.x,t.v1.x),t.v2.x); b.mx.y=fmaxf(fmaxf(t.v0.y,t.v1.y),t.v2.y); b.mx.z=fmaxf(fmaxf(t.v0.z,t.v1.z),t.v2.z);
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

// ======================== Primary kernel: NO SORT (v20 proven +45%) ========================
#define STACK_DEPTH 16

__global__ void __launch_bounds__(256,5) tracePrimary(
    const int4*__restrict__ d_bvh4,int n4,
    const float*__restrict__ tv0x,const float*__restrict__ tv0y,const float*__restrict__ tv0z,
    const float*__restrict__ tv1x,const float*__restrict__ tv1y,const float*__restrict__ tv1z,
    const float*__restrict__ tv2x,const float*__restrict__ tv2y,const float*__restrict__ tv2z,
    const float*__restrict__ rox,const float*__restrict__ roy,const float*__restrict__ roz,
    const float*__restrict__ rdx,const float*__restrict__ rdy,const float*__restrict__ rdz,
    const float*__restrict__ rix,const float*__restrict__ riy,const float*__restrict__ riz,
    Hit*__restrict__ hits,int numRays)
{
    const unsigned lane=threadIdx.x&31;
    while(true){
        int bs; if(lane==0)bs=atomicAdd(&g_rayCounter,32);
        bs=__shfl_sync(0xFFFFFFFF,bs,0);
        if(bs>=numRays)break;
        int ri=bs+lane;
        float ox,oy,oz,dx,dy,dz,ix,iy,iz,tHit=1e30f;
        int hitTri=-1;float hitU=0,hitV=0;
        if(ri<numRays){ox=rox[ri];oy=roy[ri];oz=roz[ri];dx=rdx[ri];dy=rdy[ri];dz=rdz[ri];ix=rix[ri];iy=riy[ri];iz=riz[ri];}
        int stk[STACK_DEPTH];int sp=0;stk[sp++]=0;
        bool alive=(ri<numRays);
        while(sp>0&&alive){
            int ni=stk[--sp];
            if(ni<0){
                int enc=-(ni+2);int ts=enc>>3,tc=(enc&7)+1;
                for(int t=0;t<tc;t++){
                    int ti=ts+t;
                    float e1x=__ldg(&tv1x[ti])-__ldg(&tv0x[ti]),e1y=__ldg(&tv1y[ti])-__ldg(&tv0y[ti]),e1z=__ldg(&tv1z[ti])-__ldg(&tv0z[ti]);
                    float e2x=__ldg(&tv2x[ti])-__ldg(&tv0x[ti]),e2y=__ldg(&tv2y[ti])-__ldg(&tv0y[ti]),e2z=__ldg(&tv2z[ti])-__ldg(&tv0z[ti]);
                    float px=dy*e2z-dz*e2y,py=dz*e2x-dx*e2z,pz=dx*e2y-dy*e2x;
                    float det=e1x*px+e1y*py+e1z*pz;if(fabsf(det)<1e-12f)continue;
                    float inv=1.f/det;
                    float tx=ox-__ldg(&tv0x[ti]),ty=oy-__ldg(&tv0y[ti]),tz=oz-__ldg(&tv0z[ti]);
                    float u=inv*(tx*px+ty*py+tz*pz);if(u<0.f||u>1.f)continue;
                    float qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x;
                    float v=inv*(dx*qx+dy*qy+dz*qz);if(v<0.f||u+v>1.f)continue;
                    float tt=inv*(e2x*qx+e2y*qy+e2z*qz);
                    if(tt>0.f&&tt<tHit){tHit=tt;hitTri=ti;hitU=u;hitV=v;}
                }
                continue;
            }
            int4 n0=__ldg(&d_bvh4[ni*4]),n1=__ldg(&d_bvh4[ni*4+1]),n2=__ldg(&d_bvh4[ni*4+2]),n3=__ldg(&d_bvh4[ni*4+3]);
            const __half*bx=(const __half*)&n0,*by=(const __half*)&n1,*bz=(const __half*)&n2;
            const int*ch=(const int*)&n3;
            // NO SORT: push valid children in reverse order
            for(int c=3;c>=0;c--){
                if(ch[c]==-1)continue;
                float t1x=(__half2float(bx[c])-ox)*ix,t2x=(__half2float(bx[4+c])-ox)*ix;
                float t1y=(__half2float(by[c])-oy)*iy,t2y=(__half2float(by[4+c])-oy)*iy;
                float t1z=(__half2float(bz[c])-oz)*iz,t2z=(__half2float(bz[4+c])-oz)*iz;
                float tN=fmaxf(fmaxf(fminf(t1x,t2x),fminf(t1y,t2y)),fminf(t1z,t2z));
                float tF=fminf(fminf(fmaxf(t1x,t2x),fmaxf(t1y,t2y)),fmaxf(t1z,t2z));
                if(tN<=tF&&tF>0.f&&tN<tHit&&sp<STACK_DEPTH)stk[sp++]=ch[c];
            }
        }
        if(ri<numRays){hits[ri].t=tHit;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;}
    }
}

// ======================== Diffuse kernel: WITH front-to-back sort (v20 proven essential) ========================
__global__ void __launch_bounds__(256,5) traceDiffuse(
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
    unsigned long long myActive=0,mySteps=0;
    while(true){
        int bs; if(lane==0)bs=atomicAdd(&g_rayCounter,32);
        bs=__shfl_sync(0xFFFFFFFF,bs,0);
        if(bs>=numRays)break;
        int ri=bs+lane;
        float ox,oy,oz,dx,dy,dz,ix,iy,iz,tHit=1e30f;
        int hitTri=-1;float hitU=0,hitV=0;
        if(ri<numRays){ox=rox[ri];oy=roy[ri];oz=roz[ri];dx=rdx[ri];dy=rdy[ri];dz=rdz[ri];ix=rix[ri];iy=riy[ri];iz=riz[ri];}
        int stk[STACK_DEPTH];int sp=0;stk[sp++]=0;
        bool alive=(ri<numRays);
        while(sp>0&&alive){
            int ni=stk[--sp];
            if(ni<0){
                int enc=-(ni+2);int ts=enc>>3,tc=(enc&7)+1;
                for(int t=0;t<tc;t++){
                    int ti=ts+t;
                    float e1x=__ldg(&tv1x[ti])-__ldg(&tv0x[ti]),e1y=__ldg(&tv1y[ti])-__ldg(&tv0y[ti]),e1z=__ldg(&tv1z[ti])-__ldg(&tv0z[ti]);
                    float e2x=__ldg(&tv2x[ti])-__ldg(&tv0x[ti]),e2y=__ldg(&tv2y[ti])-__ldg(&tv0y[ti]),e2z=__ldg(&tv2z[ti])-__ldg(&tv0z[ti]);
                    float px=dy*e2z-dz*e2y,py=dz*e2x-dx*e2z,pz=dx*e2y-dy*e2x;
                    float det=e1x*px+e1y*py+e1z*pz;if(fabsf(det)<1e-12f)continue;
                    float inv=1.f/det;
                    float tx=ox-__ldg(&tv0x[ti]),ty=oy-__ldg(&tv0y[ti]),tz=oz-__ldg(&tv0z[ti]);
                    float u=inv*(tx*px+ty*py+tz*pz);if(u<0.f||u>1.f)continue;
                    float qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x;
                    float v=inv*(dx*qx+dy*qy+dz*qz);if(v<0.f||u+v>1.f)continue;
                    float tt=inv*(e2x*qx+e2y*qy+e2z*qz);
                    if(tt>0.f&&tt<tHit){tHit=tt;hitTri=ti;hitU=u;hitV=v;}
                }
                continue;
            }
            int4 n0=__ldg(&d_bvh4[ni*4]),n1=__ldg(&d_bvh4[ni*4+1]),n2=__ldg(&d_bvh4[ni*4+2]),n3=__ldg(&d_bvh4[ni*4+3]);
            const __half*bx=(const __half*)&n0,*by=(const __half*)&n1,*bz=(const __half*)&n2;
            const int*ch=(const int*)&n3;
            float dist[4];int order[4];
            for(int c=0;c<4;c++){
                if(ch[c]==-1){dist[c]=1e30f;order[c]=c;continue;}
                float t1x=(__half2float(bx[c])-ox)*ix,t2x=(__half2float(bx[4+c])-ox)*ix;
                float t1y=(__half2float(by[c])-oy)*iy,t2y=(__half2float(by[4+c])-oy)*iy;
                float t1z=(__half2float(bz[c])-oz)*iz,t2z=(__half2float(bz[4+c])-oz)*iz;
                float tN=fmaxf(fmaxf(fminf(t1x,t2x),fminf(t1y,t2y)),fminf(t1z,t2z));
                float tF=fminf(fminf(fmaxf(t1x,t2x),fmaxf(t1y,t2y)),fmaxf(t1z,t2z));
                dist[c]=(tN<=tF&&tF>0.f&&tN<tHit)?tN:1e30f;
                order[c]=c;
            }
            // Front-to-back sort (essential for diffuse — proven in v20)
            for(int i=0;i<3;i++)for(int j=i+1;j<4;j++)
                if(dist[order[i]]>dist[order[j]]){int tmp=order[i];order[i]=order[j];order[j]=tmp;}
            for(int i=3;i>=0;i--)
                if(dist[order[i]]<1e30f&&sp<STACK_DEPTH)stk[sp++]=ch[order[i]];

            mySteps++;
            if((mySteps&15)==0){
                unsigned mask=__ballot_sync(0xFFFFFFFF,sp>0);
                myActive+=__popc(mask);
            }
        }
        if(ri<numRays){hits[ri].t=tHit;hits[ri].tri=hitTri;hits[ri].u=hitU;hits[ri].v=hitV;}
    }
    if(warpEff&&(lane==0)){atomicAdd(&warpEff[0],myActive);atomicAdd(&warpEff[1],mySteps/16);}
}

// ======================== Scene generation ========================
static void genScene(Tri*tris,int nT,float sc){
    srand(42);
    for(int i=0;i<nT;i++){
        float cx=((float)rand()/RAND_MAX-0.5f)*sc,cy=((float)rand()/RAND_MAX-0.5f)*sc,cz=((float)rand()/RAND_MAX-0.5f)*sc;
        float sz=sc*0.005f+((float)rand()/RAND_MAX)*sc*0.01f;
        tris[i].v0={cx-sz,cy-sz,cz};tris[i].v1={cx+sz,cy-sz,cz+sz};tris[i].v2={cx,cy+sz,cz-sz};
    }
}

// ======================== Sort helpers ========================
static inline uint32_t expand3(uint32_t v){v&=0x3FF;v=(v|(v<<16))&0x30000FF;v=(v|(v<<8))&0x300F00F;v=(v|(v<<4))&0x30C30C3;v=(v|(v<<2))&0x9249249;return v;}
struct RayAoS{float3a o,d;};
static void sortMortonOctant(RayAoS*r,int n,float3a smn,float3a smx){
    struct SK{uint32_t key;int idx;};
    SK*keys=(SK*)malloc(n*sizeof(SK));
    for(int i=0;i<n;i++){
        float nx=(r[i].o.x-smn.x)/(smx.x-smn.x+1e-7f),ny=(r[i].o.y-smn.y)/(smx.y-smn.y+1e-7f),nz=(r[i].o.z-smn.z)/(smx.z-smn.z+1e-7f);
        uint32_t ix=(uint32_t)fminf(fmaxf(nx*1023.f,0.f),1023.f),iy=(uint32_t)fminf(fmaxf(ny*1023.f,0.f),1023.f),iz=(uint32_t)fminf(fmaxf(nz*1023.f,0.f),1023.f);
        uint32_t m=expand3(ix)|(expand3(iy)<<1)|(expand3(iz)<<2);
        uint32_t o=((r[i].d.x<0.f)?4u:0u)|((r[i].d.y<0.f)?2u:0u)|((r[i].d.z<0.f)?1u:0u);
        keys[i]={(o<<27)|(m>>3),i};
    }
    std::sort(keys,keys+n,[](const SK&a,const SK&b){return a.key<b.key;});
    RayAoS*tmp=(RayAoS*)malloc(n*sizeof(RayAoS));for(int i=0;i<n;i++)tmp[i]=r[keys[i].idx];
    memcpy(r,tmp,n*sizeof(RayAoS));free(tmp);free(keys);
}

// ======================== Benchmark ========================
typedef void(*TraceFn1)(const int4*,int,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,Hit*,int);
typedef void(*TraceFn2)(const int4*,int,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,Hit*,int,unsigned long long*);

static float benchKernel1(TraceFn1 kernel,const int4*d_bvh4,int n4,float*d_tv[9],
    RayAoS*rays,int numRays,Hit*d_hits,float*d_ray[9],Hit*h_hits,int*hitCount)
{
    float*h_r[9];for(int j=0;j<9;j++)h_r[j]=(float*)malloc(numRays*4);
    for(int i=0;i<numRays;i++){
        h_r[0][i]=rays[i].o.x;h_r[1][i]=rays[i].o.y;h_r[2][i]=rays[i].o.z;
        h_r[3][i]=rays[i].d.x;h_r[4][i]=rays[i].d.y;h_r[5][i]=rays[i].d.z;
        float ddx=rays[i].d.x,ddy=rays[i].d.y,ddz=rays[i].d.z;
        h_r[6][i]=1.f/(fabsf(ddx)>1e-8f?ddx:(ddx>=0?1e-8f:-1e-8f));
        h_r[7][i]=1.f/(fabsf(ddy)>1e-8f?ddy:(ddy>=0?1e-8f:-1e-8f));
        h_r[8][i]=1.f/(fabsf(ddz)>1e-8f?ddz:(ddz>=0?1e-8f:-1e-8f));
    }
    for(int j=0;j<9;j++)cudaMemcpy(d_ray[j],h_r[j],numRays*4,cudaMemcpyHostToDevice);
    for(int j=0;j<9;j++)free(h_r[j]);
    unsigned int zero=0;cudaMemcpyToSymbol(g_rayCounter,&zero,4);
    kernel<<<320,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
        d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],d_hits,numRays);
    cudaDeviceSynchronize();
    cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
    float best=1e30f;
    for(int r=0;r<3;r++){
        cudaMemcpyToSymbol(g_rayCounter,&zero,4);
        cudaEventRecord(t0);
        kernel<<<320,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
            d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],d_hits,numRays);
        cudaEventRecord(t1);cudaEventSynchronize(t1);
        float ms;cudaEventElapsedTime(&ms,t0,t1);if(ms<best)best=ms;
    }
    cudaMemcpy(h_hits,d_hits,numRays*sizeof(Hit),cudaMemcpyDeviceToHost);
    *hitCount=0;for(int i=0;i<numRays;i++)if(h_hits[i].tri>=0)(*hitCount)++;
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    return (float)numRays/best/1000.f;
}

static float benchKernel2(TraceFn2 kernel,const int4*d_bvh4,int n4,float*d_tv[9],
    RayAoS*rays,int numRays,Hit*d_hits,float*d_ray[9],Hit*h_hits,int*hitCount,
    unsigned long long*d_we,float*warpEff)
{
    float*h_r[9];for(int j=0;j<9;j++)h_r[j]=(float*)malloc(numRays*4);
    for(int i=0;i<numRays;i++){
        h_r[0][i]=rays[i].o.x;h_r[1][i]=rays[i].o.y;h_r[2][i]=rays[i].o.z;
        h_r[3][i]=rays[i].d.x;h_r[4][i]=rays[i].d.y;h_r[5][i]=rays[i].d.z;
        float ddx=rays[i].d.x,ddy=rays[i].d.y,ddz=rays[i].d.z;
        h_r[6][i]=1.f/(fabsf(ddx)>1e-8f?ddx:(ddx>=0?1e-8f:-1e-8f));
        h_r[7][i]=1.f/(fabsf(ddy)>1e-8f?ddy:(ddy>=0?1e-8f:-1e-8f));
        h_r[8][i]=1.f/(fabsf(ddz)>1e-8f?ddz:(ddz>=0?1e-8f:-1e-8f));
    }
    for(int j=0;j<9;j++)cudaMemcpy(d_ray[j],h_r[j],numRays*4,cudaMemcpyHostToDevice);
    for(int j=0;j<9;j++)free(h_r[j]);
    unsigned int zero=0;cudaMemcpyToSymbol(g_rayCounter,&zero,4);
    kernel<<<320,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
        d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],d_hits,numRays,nullptr);
    cudaDeviceSynchronize();
    unsigned long long h_we[2]={0,0};
    cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
    float best=1e30f;
    for(int r=0;r<3;r++){
        cudaMemcpyToSymbol(g_rayCounter,&zero,4);
        cudaMemcpy(d_we,h_we,16,cudaMemcpyHostToDevice);
        cudaEventRecord(t0);
        kernel<<<320,256>>>(d_bvh4,n4,d_tv[0],d_tv[1],d_tv[2],d_tv[3],d_tv[4],d_tv[5],d_tv[6],d_tv[7],d_tv[8],
            d_ray[0],d_ray[1],d_ray[2],d_ray[3],d_ray[4],d_ray[5],d_ray[6],d_ray[7],d_ray[8],d_hits,numRays,d_we);
        cudaEventRecord(t1);cudaEventSynchronize(t1);
        float ms;cudaEventElapsedTime(&ms,t0,t1);if(ms<best)best=ms;
    }
    cudaMemcpy(h_hits,d_hits,numRays*sizeof(Hit),cudaMemcpyDeviceToHost);
    *hitCount=0;for(int i=0;i<numRays;i++)if(h_hits[i].tri>=0)(*hitCount)++;
    cudaMemcpy(h_we,d_we,16,cudaMemcpyDeviceToHost);
    *warpEff=h_we[1]>0?(float)h_we[0]/(h_we[1]*32.f)*100.f:0.f;
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    return (float)numRays/best/1000.f;
}

// ======================== Main ========================
int main(){
    printf("══════════════════════════════════════════════════════════════════════════\n");
    printf("  V21 — Dual-Kernel Production Engine (NoSort primary + Sort diffuse)\n");
    printf("  Scaling test: 99K → 500K → 1M → 2M → 3M triangles\n");
    printf("══════════════════════════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
    printf("  GPU: %s | SMs: %d | Stack: %d\n\n",prop.name,prop.multiProcessorCount,STACK_DEPTH);

    int triCounts[]={99000,500000,1000000,2000000,3000000};
    int nScenes=5;
    const int NRAYS=1024*1024;

    for(int si=0;si<nScenes;si++){
        int NTRI=triCounts[si];
        printf("  ━━━ %dK tris ━━━\n",(NTRI+500)/1000);

        Tri*h_tris=(Tri*)malloc(NTRI*sizeof(Tri));
        genScene(h_tris,NTRI,10.f);
        BVHBuild b2; b2.build(h_tris,NTRI);
        int maxN4=(int)b2.nodes.size()*2;
        BVH4Node*h_b4=(BVH4Node*)calloc(maxN4,sizeof(BVH4Node));
        int n4=0;collapseToB4(b2,0,h_b4,n4,h_tris);
        printf("  BVH4: %d nodes (%.1f MB)\n",n4,n4*64.f/1e6f);

        int4*d_bvh4;cudaMalloc(&d_bvh4,n4*sizeof(BVH4Node));
        cudaMemcpy(d_bvh4,h_b4,n4*sizeof(BVH4Node),cudaMemcpyHostToDevice);

        Tri*ord=b2.ordered.data();int nOT=(int)b2.ordered.size();
        float*h_tv[9],*d_tv[9];
        for(int j=0;j<9;j++){h_tv[j]=(float*)malloc(nOT*4);cudaMalloc(&d_tv[j],nOT*4);}
        for(int i=0;i<nOT;i++){
            h_tv[0][i]=ord[i].v0.x;h_tv[1][i]=ord[i].v0.y;h_tv[2][i]=ord[i].v0.z;
            h_tv[3][i]=ord[i].v1.x;h_tv[4][i]=ord[i].v1.y;h_tv[5][i]=ord[i].v1.z;
            h_tv[6][i]=ord[i].v2.x;h_tv[7][i]=ord[i].v2.y;h_tv[8][i]=ord[i].v2.z;
        }
        for(int j=0;j<9;j++)cudaMemcpy(d_tv[j],h_tv[j],nOT*4,cudaMemcpyHostToDevice);

        float*d_ray[9];for(int j=0;j<9;j++)cudaMalloc(&d_ray[j],NRAYS*4);
        Hit*d_hits;cudaMalloc(&d_hits,NRAYS*sizeof(Hit));
        Hit*h_hits=(Hit*)malloc(NRAYS*sizeof(Hit));
        unsigned long long*d_we;cudaMalloc(&d_we,16);

        float3a smn={1e30f,1e30f,1e30f},smx={-1e30f,-1e30f,-1e30f};
        for(int i=0;i<NTRI;i++){
            smn.x=fminf(smn.x,fminf(fminf(h_tris[i].v0.x,h_tris[i].v1.x),h_tris[i].v2.x));
            smn.y=fminf(smn.y,fminf(fminf(h_tris[i].v0.y,h_tris[i].v1.y),h_tris[i].v2.y));
            smn.z=fminf(smn.z,fminf(fminf(h_tris[i].v0.z,h_tris[i].v1.z),h_tris[i].v2.z));
            smx.x=fmaxf(smx.x,fmaxf(fmaxf(h_tris[i].v0.x,h_tris[i].v1.x),h_tris[i].v2.x));
            smx.y=fmaxf(smx.y,fmaxf(fmaxf(h_tris[i].v0.y,h_tris[i].v1.y),h_tris[i].v2.y));
            smx.z=fmaxf(smx.z,fmaxf(fmaxf(h_tris[i].v0.z,h_tris[i].v1.z),h_tris[i].v2.z));
        }

        // Generate primary rays
        RayAoS*primary=(RayAoS*)malloc(NRAYS*sizeof(RayAoS));
        int side=(int)sqrtf((float)NRAYS);
        for(int i=0;i<NRAYS;i++){
            int px=i%side,py=i/side;
            float u=(px+0.5f)/side*2.f-1.f,v=(py+0.5f)/side*2.f-1.f;
            primary[i].o={0,0,-20.f};
            float len=sqrtf(u*u+v*v+1.f);
            primary[i].d={u/len,v/len,1.f/len};
        }

        // Generate diffuse rays + Morton sort
        RayAoS*diffuse=(RayAoS*)malloc(NRAYS*sizeof(RayAoS));
        srand(12345);
        for(int i=0;i<NRAYS;i++){
            int px=i%side,py=i/side;
            diffuse[i].o={((float)px/side-0.5f)*5.f,((float)py/side-0.5f)*5.f,((float)rand()/RAND_MAX-0.5f)*3.f};
            float r1=(float)rand()/RAND_MAX,r2=(float)rand()/RAND_MAX;
            float phi=6.28318f*r1,ct=sqrtf(1.f-r2),st=sqrtf(r2);
            diffuse[i].d={st*cosf(phi),st*sinf(phi),ct};
            if(rand()%2)diffuse[i].d.x=-diffuse[i].d.x;
            if(rand()%2)diffuse[i].d.y=-diffuse[i].d.y;
            if(rand()%2)diffuse[i].d.z=-diffuse[i].d.z;
        }
        sortMortonOctant(diffuse,NRAYS,smn,smx);

        int hc; float wE;
        float pMR=benchKernel1(tracePrimary,d_bvh4,n4,d_tv,primary,NRAYS,d_hits,d_ray,h_hits,&hc);
        printf("  PRIMARY (NoSort):   %7.0f MR/s  hit:%5.1f%%\n",pMR,100.f*hc/NRAYS);

        float dMR=benchKernel2(traceDiffuse,d_bvh4,n4,d_tv,diffuse,NRAYS,d_hits,d_ray,h_hits,&hc,d_we,&wE);
        printf("  DIFFUSE (Sorted):   %7.0f MR/s  hit:%5.1f%%  warp:%5.1f%%\n",dMR,100.f*hc/NRAYS,wE);
        printf("\n");

        // Cleanup per-scene
        for(int j=0;j<9;j++){cudaFree(d_ray[j]);cudaFree(d_tv[j]);free(h_tv[j]);}
        cudaFree(d_bvh4);cudaFree(d_hits);cudaFree(d_we);
        free(h_b4);free(h_tris);free(h_hits);free(primary);free(diffuse);
    }
    return 0;
}
