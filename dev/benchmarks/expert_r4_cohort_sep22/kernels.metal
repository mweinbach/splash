// Private FIRST N16 / R4 expert numerical alternative; no other tile exports.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "abi.hpp"
#include "prefill4k_allrows_qmv_probe.h"
using namespace metal;
#pragma METAL fp math_mode(safe)
inline bool expert_r4_finite(float x) { return (as_type<uint>(x)&0x7f800000u)!=0x7f800000u; }
inline void expert_r4_error(device atomic_uint *diag,uint bits) {
  atomic_fetch_or_explicit(diag,bits,memory_order_relaxed);
}
inline bfloat expert_r4_nan() { return bfloat(as_type<float>(0x7fc00000u)); }
inline bool expert_r4_params(constant FlashExpertR4CohortParams &p) {
  return p.rows==4 && p.selections==10 && p.experts==512 && p.output_tile==16 &&
      p.slots==10 && p.epoch && !p.reserved0 && !p.reserved1;
}
inline float expert_r4_a(ushort bits,device atomic_uint *diag) {
  if((bits&0x7f80u)==0x7f80u){expert_r4_error(diag,4u);return 0.0f;}
  return as_type<float>(uint(bits)<<16);
}
inline float4 expert_r4_a4(device const ushort4 *a,uint chunk,device atomic_uint *diag) {
  const ushort4 b=a[chunk];
  return float4(expert_r4_a(b.x,diag),expert_r4_a(b.y,diag),expert_r4_a(b.z,diag),expert_r4_a(b.w,diag));
}
inline float expert_r4_dot4(float4 p) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  return (p.x+p.y)+(p.z+p.w);
}
inline float expert_r4_reduce(float partial) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#pragma unroll
  for(ushort delta=16;delta;delta/=2)partial+=simd_shuffle_xor(partial,delta);
  return partial;
}
#pragma METAL fp math_mode(fast)
inline bfloat expert_r4_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent=bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator=bfloat(1.0f)+exponent;
  const bfloat tail=bfloat(1.0f)/denominator;
  return source<bfloat(0.0f)?tail:bfloat(1.0f)-tail;
}
#pragma METAL fp math_mode(safe)

// One ordered CTA; metadata and the hidden diagnostic evidence are complete
// before publishing the ready marker. No input/diagnostic clearing occurs.
kernel void expert_r4_cohort_sep22_plan(
    device const ushort *hidden [[buffer(0)]],device const long *ids [[buffer(1)]],
    device const uint *ranks [[buffer(2)]],device uint *counts [[buffer(3)]],
    device uint *offsets [[buffer(4)]],device uint *routeMap [[buffer(5)]],
    device uint *inverse [[buffer(6)]],device FlashMoEBucketJob *jobs [[buffer(7)]],
    device uint *jobCount [[buffer(8)]],device uint *stages [[buffer(9)]],
    device atomic_uint *diag [[buffer(10)]],constant FlashExpertR4CohortParams &p [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if(!expert_r4_params(p)||any(group!=uint3(0))||any(threads!=uint3(256,1,1))) {
    if(!tid)expert_r4_error(diag,2u);return;
  }
  for(uint i=tid;i<512;i+=256)counts[i]=0;
  for(uint i=tid;i<513;i+=256){offsets[i]=0;stages[i]=0;}
  if(tid<40){routeMap[tid]=UINT_MAX;inverse[tid]=UINT_MAX;jobs[tid]={UINT_MAX,0};}
  if(!tid)*jobCount=0;
  threadgroup_barrier(mem_flags::mem_device);
  // Inspect every original hidden row even when all its IDs/ranks are excluded.
  for(uint i=tid;i<4*2560;i+=256)
    if((hidden[i]&0x7f80u)==0x7f80u)expert_r4_error(diag,4u);
  if(!tid) {
    uint live=0;
    for(uint route=0;route<40;++route) {
      const long id=ids[route];
      if(id<0||id>=512){expert_r4_error(diag,1u);continue;}
      for(uint previous=route/10*10;previous<route;++previous)
        if(ids[previous]==id)expert_r4_error(diag,1u);
      if(ranks[uint(id)]>=512){expert_r4_error(diag,1u);continue;}
      ++counts[uint(id)];
    }
    uint total=0,groups=0;
    for(uint expert=0;expert<512;++expert) {
      offsets[expert]=total;
      const uint count=counts[expert];
      for(uint route=0;route<40;++route)
        if(ids[route]==long(expert)&&ranks[expert]<512) {
          routeMap[total]=route;inverse[route]=total;++total;
        }
      for(uint begin=offsets[expert];begin<total;begin+=4)jobs[groups++]={expert,begin};
      live+=count;
    }
    offsets[512]=total;
    if(total!=live||total>40||groups>40)expert_r4_error(diag,2u);
    *jobCount=groups;
  }
  threadgroup_barrier(mem_flags::mem_device);
  if(!tid){stages[0]=p.epoch;stages[1]=kExpertR4MetaReady;}
}

template<bool Full>
inline bool expert_r4_meta(device const uint *offsets,device const uint *routeMap,
    device const uint *inverse,device const FlashMoEBucketJob *jobs,device const uint *jobCount,
    device const uint *stages,device const long *ids,device const uint *ranks,
    constant FlashExpertR4CohortParams &p) {
  if(stages[0]!=p.epoch||stages[1]!=kExpertR4MetaReady||*jobCount>40||offsets[0]||offsets[512]>40)return false;
  // Exact live ownership in O40: all jobs cover one disjoint sorted prefix,
  // complete expert buckets, stable route order and inverse uniqueness.
  uint covered=0,previousExpert=UINT_MAX,previousRoute=0;
  for(uint j=0;j<*jobCount;++j) {
    const uint expert=jobs[j].expert,begin=jobs[j].row_begin;
    if(expert>=512||begin!=covered||ranks[expert]>=512)return false;
    const uint first=offsets[expert],end=offsets[expert+1];
    if(first>begin||begin>=end||end>40)return false;
    if(previousExpert!=expert) {
      if(previousExpert!=UINT_MAX&&previousExpert>=expert)return false;
      if(begin!=first)return false;
    }
    const uint m=min(4u,end-begin);
    for(uint r=0;r<m;++r) {
      const uint index=begin+r,route=routeMap[index];
      if(route>=40||ids[route]!=long(expert)||inverse[route]!=index||
          ((r||previousExpert==expert)&&route<=previousRoute))return false;
      previousRoute=route;
    }
    covered+=m;
    if(j+1==*jobCount||jobs[j+1].expert!=expert)if(covered!=end)return false;
    previousExpert=expert;
  }
  if(covered!=offsets[512])return false;
  for(uint route=0;route<40;++route) {
    const long id=ids[route];
    const bool valid=id>=0&&id<512&&ranks[uint(id)]<512;
    if(valid){if(inverse[route]>=covered||routeMap[inverse[route]]!=route)return false;}
    else if(inverse[route]!=UINT_MAX)return false;
  }
  // Untimed audit/probes additionally validate unused expert-prefix entries.
  if constexpr(Full)for(uint expert=0;expert<512;++expert)
    if(offsets[expert]>offsets[expert+1]||offsets[expert+1]>40)return false;
  return true;
}
template<bool Gate,bool Audit>
inline bool expert_r4_begin(device const uint *offsets,device const uint *routeMap,
    device const uint *inverse,device const FlashMoEBucketJob *jobs,device const uint *jobCount,
    device uint *stages,device const long *ids,device const uint *ranks,
    constant FlashExpertR4CohortParams &p,uint3 group,uint3 threads,uint3 grid,uint tid,
    threadgroup uint *uniform,device atomic_uint *diag) {
  constexpr uint Width=Gate?640:2560;
  if(!expert_r4_params(p)||group.x>=Width/16||group.y>=10||group.z||any(threads!=uint3(128,1,1))||any(grid!=uint3(Width/16,10,1))) {
    if(!tid)expert_r4_error(diag,2u);return false;
  }
  if(!tid) {
    bool good=expert_r4_meta<Audit>(offsets,routeMap,inverse,jobs,jobCount,stages,ids,ranks,p);
    if constexpr(!Gate)good=good&&stages[4]==kExpertR4GateReady;
    if constexpr(Audit) {
      const uint gc=atomic_load_explicit(reinterpret_cast<device atomic_uint *>(stages+2),memory_order_relaxed);
      const uint dc=atomic_load_explicit(reinterpret_cast<device atomic_uint *>(stages+3),memory_order_relaxed);
      good=good&&(Gate?gc<kExpertR4GateCTAs:gc==kExpertR4GateCTAs&&dc<kExpertR4DownCTAs);
    }
    *uniform=good?1u:0u;if(!good)expert_r4_error(diag,2u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return *uniform!=0;
}
template<bool Gate,bool Audit>
inline void expert_r4_finish(device uint *stages,uint3 group,uint tid) {
  if constexpr(Audit) {
    threadgroup_barrier(mem_flags::mem_device);
    if(!tid)atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(stages+(Gate?2:3)),1u,memory_order_relaxed);
  }
  // The shipping marker is one writer. Serial compute-dispatch ordering makes
  // every producer CTA complete before the next stage can consume the marker.
  if(!tid&&!group.x&&!group.y)stages[Gate?4:5]=Gate?kExpertR4GateReady:kExpertR4DownReady;
}
template<bool Gate,bool Audit>
inline void expert_r4_execute(device const bfloat *input,device const char *gate,device const float *gs,
    device const char *up,device const float *us,device const uint *ranks,device const long *ids,
    device const uint *offsets,device const uint *routeMap,device const uint *inverse,
    device const FlashMoEBucketJob *jobs,device const uint *jobCount,device uint *stages,
    device bfloat *output,device atomic_uint *diag,constant FlashExpertR4CohortParams &p,
    uint3 group,uint3 threads,uint3 grid,uint simd,uint lane,uint tid,threadgroup uint *uniform) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  constexpr uint K=Gate?2560:640,Width=Gate?640:2560;
  if(!expert_r4_begin<Gate,Audit>(offsets,routeMap,inverse,jobs,jobCount,stages,ids,ranks,p,group,threads,grid,tid,uniform,diag))return;
  const uint nbase=group.x*16+simd*4;
  // Excluded poison is disjoint from every live route-map destination. Keep
  // gate-only bit1, down bit5; excluded down routes never construct/read A.
  if(!lane)for(uint route=group.y;route<40;route+=10)if(inverse[route]==UINT_MAX) {
#pragma unroll
    for(uint c=0;c<4;++c)output[ulong(route)*Width+nbase+c]=expert_r4_nan();
    expert_r4_error(diag,Gate?1u:5u);
  }
  for(uint job=group.y;job<*jobCount;job+=10) {
    const uint expert=jobs[job].expert,begin=jobs[job].row_begin;
    const uint m=min(4u,offsets[expert+1]-begin),rank=ranks[expert];
    float gdot[4][4],udot[4][4];
#pragma unroll
    for(uint r=0;r<4;++r)for(uint c=0;c<4;++c){gdot[r][c]=0.0f;udot[r][c]=0.0f;}
    for(uint chunk=lane;chunk<K/4;chunk+=32) {
      float4 a[4];
#pragma unroll
      for(uint r=0;r<4;++r)if(r<m) {
        const uint route=routeMap[begin+r],row=Gate?route/10:route;
        a[r]=expert_r4_a4(reinterpret_cast<const device ushort4 *>(input+ulong(row)*K),chunk,diag);
      }
#pragma unroll
      for(uint c=0;c<4;++c) {
        const ulong row=(ulong(rank)*Width+nbase+c)*K;
        const float4 g=float4(reinterpret_cast<const device char4 *>(gate+row)[chunk]);
        float4 u(0.0f);if constexpr(Gate)u=float4(reinterpret_cast<const device char4 *>(up+row)[chunk]);
#pragma unroll
        for(uint r=0;r<4;++r)if(r<m) {
          gdot[r][c]+=expert_r4_dot4(g*a[r]);
          if constexpr(Gate)udot[r][c]+=expert_r4_dot4(u*a[r]);
        }
      }
    }
#pragma unroll
    for(uint r=0;r<4;++r)if(r<m) {
#pragma unroll
      for(uint c=0;c<4;++c) {
        const float dot=expert_r4_reduce(gdot[r][c]);
        const float ud=Gate?expert_r4_reduce(udot[r][c]):0.0f;
        if(!lane) {
          const uint n=nbase+c,route=routeMap[begin+r];
          const float scale=gs[ulong(rank)*Width+n],scaled=dot*scale;
          bfloat value=bfloat(scaled);
          if constexpr(Gate) {
            const float scaleU=us[ulong(rank)*Width+n],scaledU=ud*scaleU;
            const bfloat uv=bfloat(scaledU),silu=value*expert_r4_sigmoid(value);value=silu*uv;
            if(!(scaleU>0)||!expert_r4_finite(scaleU)||!expert_r4_finite(ud)||!expert_r4_finite(scaledU)||!expert_r4_finite(float(uv)))expert_r4_error(diag,4u);
          }
          if(!(scale>0)||!expert_r4_finite(scale)||!expert_r4_finite(dot)||!expert_r4_finite(scaled)||!expert_r4_finite(float(value)))expert_r4_error(diag,4u);
          output[ulong(route)*Width+n]=value;
        }
      }
    }
  }
  expert_r4_finish<Gate,Audit>(stages,group,tid);
}
#define EXPERT_R4_GATE(Name,Audit) \
kernel void Name(device const bfloat *a [[buffer(0)]],device const char *g [[buffer(1)]],device const float *gs [[buffer(2)]], \
 device const char *u [[buffer(3)]],device const float *us [[buffer(4)]],device const uint *ranks [[buffer(5)]],device const long *ids [[buffer(6)]], \
 device const uint *offsets [[buffer(7)]],device const uint *map [[buffer(8)]],device const uint *inverse [[buffer(9)]], \
 device const FlashMoEBucketJob *jobs [[buffer(10)]],device const uint *count [[buffer(11)]],device uint *stages [[buffer(12)]], \
 device bfloat *out [[buffer(13)]],device atomic_uint *diag [[buffer(14)]],constant FlashExpertR4CohortParams &p [[buffer(15)]], \
 uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]],uint3 grid [[threadgroups_per_grid]],uint simd [[simdgroup_index_in_threadgroup]], \
 uint lane [[thread_index_in_simdgroup]],uint tid [[thread_index_in_threadgroup]]) {threadgroup uint uniform; \
 expert_r4_execute<true,Audit>(a,g,gs,u,us,ranks,ids,offsets,map,inverse,jobs,count,stages,out,diag,p,group,threads,grid,simd,lane,tid,&uniform);}
#define EXPERT_R4_DOWN(Name,Audit) \
kernel void Name(device const bfloat *a [[buffer(0)]],device const char *g [[buffer(1)]],device const float *gs [[buffer(2)]], \
 device const uint *ranks [[buffer(3)]],device const long *ids [[buffer(4)]],device const uint *offsets [[buffer(5)]], \
 device const uint *map [[buffer(6)]],device const uint *inverse [[buffer(7)]],device const FlashMoEBucketJob *jobs [[buffer(8)]], \
 device const uint *count [[buffer(9)]],device uint *stages [[buffer(10)]],device bfloat *out [[buffer(11)]],device atomic_uint *diag [[buffer(12)]], \
 constant FlashExpertR4CohortParams &p [[buffer(13)]],uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]],uint3 grid [[threadgroups_per_grid]], \
 uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint tid [[thread_index_in_threadgroup]]) {threadgroup uint uniform; \
 expert_r4_execute<false,Audit>(a,g,gs,g,gs,ranks,ids,offsets,map,inverse,jobs,count,stages,out,diag,p,group,threads,grid,simd,lane,tid,&uniform);}
EXPERT_R4_GATE(expert_r4_cohort_sep22_gate_up_n16,false)
EXPERT_R4_DOWN(expert_r4_cohort_sep22_down_n16,false)
EXPERT_R4_GATE(expert_r4_cohort_sep22_gate_up_n16_audit,true)
EXPERT_R4_DOWN(expert_r4_cohort_sep22_down_n16_audit,true)

inline void expert_r4_probe_poison(ulong index,bool perRoute,device float *raw,device float *scaled,
    device bfloat *bf16,device atomic_uint *diag) {
  const float nan=as_type<float>(0x7fc00000u);raw[index]=nan;scaled[index]=nan;bf16[index]=bfloat(nan);
  expert_r4_error(diag,perRoute?5u:1u);
}
template<uint K,uint Width>
inline void expert_r4_probe(device const bfloat *input,device const char *codes,device const float *scales,
    device const uint *ranks,device const long *ids,device const uint *offsets,device const uint *map,
    device const uint *inverse,device const FlashMoEBucketJob *jobs,device const uint *count,device uint *stages,
    device float *raw,device float *scaled,device bfloat *bf16,device atomic_uint *diag,
    constant FlashExpertR4CohortProbeParams &p,uint3 group,uint3 threads,uint simd,uint lane,uint tid,threadgroup uint *uniform) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  if(!expert_r4_params(p.cohort)||p.reserved||p.input_size!=K||p.output_size!=Width||p.per_route_input!=uint(K==640)||
      group.x>=Width/16||group.y>=10||group.z||any(threads!=uint3(128,1,1))) {if(!tid)expert_r4_error(diag,2u);return;}
  if(!tid){*uniform=expert_r4_meta<true>(offsets,map,inverse,jobs,count,stages,ids,ranks,p.cohort)?1u:0u;if(!*uniform)expert_r4_error(diag,2u);}
  threadgroup_barrier(mem_flags::mem_threadgroup);if(!*uniform)return;
  const uint nbase=group.x*16+simd*4;
  if(!lane)for(uint route=group.y;route<40;route+=10)if(inverse[route]==UINT_MAX)
    for(uint c=0;c<4;++c)expert_r4_probe_poison(ulong(route)*Width+nbase+c,K==640,raw,scaled,bf16,diag);
  for(uint job=group.y;job<*count;job+=10) {
    const uint expert=jobs[job].expert,begin=jobs[job].row_begin,m=min(4u,offsets[expert+1]-begin),rank=ranks[expert];
    float partial[4][4];for(uint r=0;r<4;++r)for(uint c=0;c<4;++c)partial[r][c]=0.0f;
    for(uint chunk=lane;chunk<K/4;chunk+=32) {
      float4 a[4];for(uint r=0;r<4;++r)if(r<m){const uint route=map[begin+r],row=K==640?route:route/10;
        a[r]=expert_r4_a4(reinterpret_cast<const device ushort4 *>(input+ulong(row)*K),chunk,diag);}
      for(uint c=0;c<4;++c){const float4 w=float4(reinterpret_cast<const device char4 *>(codes+(ulong(rank)*Width+nbase+c)*K)[chunk]);
        for(uint r=0;r<4;++r)if(r<m)partial[r][c]+=expert_r4_dot4(w*a[r]);}
    }
    for(uint r=0;r<4;++r)if(r<m)for(uint c=0;c<4;++c) {
      const float dot=expert_r4_reduce(partial[r][c]);if(!lane){const uint n=nbase+c,route=map[begin+r];
        const float scale=scales[ulong(rank)*Width+n],value=dot*scale;const bfloat rounded=bfloat(value);
        if(!(scale>0)||!expert_r4_finite(scale)||!expert_r4_finite(dot)||!expert_r4_finite(value)||!expert_r4_finite(float(rounded)))expert_r4_error(diag,4u);
        const ulong out=ulong(route)*Width+n;raw[out]=dot;scaled[out]=value;bf16[out]=rounded;}
    }
  }
}
kernel void expert_r4_cohort_sep22_projection_probe_n16(
    device const bfloat *a [[buffer(0)]],device const char *codes [[buffer(1)]],device const float *scales [[buffer(2)]],
    device const uint *ranks [[buffer(3)]],device const long *ids [[buffer(4)]],device const uint *offsets [[buffer(5)]],
    device const uint *map [[buffer(6)]],device const uint *inverse [[buffer(7)]],device const FlashMoEBucketJob *jobs [[buffer(8)]],
    device const uint *count [[buffer(9)]],device uint *stages [[buffer(10)]],device float *raw [[buffer(11)]],
    device float *scaled [[buffer(12)]],device bfloat *bf16 [[buffer(13)]],device atomic_uint *diag [[buffer(14)]],
    constant FlashExpertR4CohortProbeParams &p [[buffer(15)]],uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint uniform;
  if(p.input_size==2560)expert_r4_probe<2560,640>(a,codes,scales,ranks,ids,offsets,map,inverse,jobs,count,stages,raw,scaled,bf16,diag,p,group,threads,simd,lane,tid,&uniform);
  else expert_r4_probe<640,2560>(a,codes,scales,ranks,ids,offsets,map,inverse,jobs,count,stages,raw,scaled,bf16,diag,p,group,threads,simd,lane,tid,&uniform);
}
// Compact independent sequential-K scalar taps, outside shipping timings.
kernel void expert_r4_cohort_sep22_scalar_projection_samples(
    device const ushort *input [[buffer(0)]],device const char *codes [[buffer(1)]],device const float *scales [[buffer(2)]],
    device const uint *ranks [[buffer(3)]],device const long *ids [[buffer(4)]],device float *raw [[buffer(5)]],
    device float *scaled [[buffer(6)]],device bfloat *bf16 [[buffer(7)]],device atomic_uint *diag [[buffer(8)]],
    constant FlashQMVProbeParams &p [[buffer(9)]],device const uint2 *pairs [[buffer(10)]],constant uint &count [[buffer(11)]],
    uint sample [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  if(sample>=count)return;
  const bool phase=(p.input_size==2560&&p.output_size==640&&!p.per_route_input)||(p.input_size==640&&p.output_size==2560&&p.per_route_input==1);
  if(p.rows!=4||p.selections!=10||p.experts!=512||p.reserved||!phase||count>40*p.output_size){if(!sample)expert_r4_error(diag,2u);return;}
  const uint2 pair=pairs[sample];const uint route=pair.x,n=pair.y;
  if(route>=40||n>=p.output_size){expert_r4_error(diag,2u);return;}
  const long id=ids[route];const uint rank=id>=0&&id<512?ranks[uint(id)]:UINT_MAX;
  const bool valid=rank<512;float dot=0.0f;
  if(!valid&&p.per_route_input){expert_r4_probe_poison(sample,true,raw,scaled,bf16,diag);return;}
  const uint row=p.per_route_input?route:route/10;
  for(uint k=0;k<p.input_size;++k){const float a=expert_r4_a(input[ulong(row)*p.input_size+k],diag);
    if(valid)dot+=float(codes[(ulong(rank)*p.output_size+n)*p.input_size+k])*a;}
  if(!valid){expert_r4_probe_poison(sample,false,raw,scaled,bf16,diag);return;}
  for(uint s=0;s<10;++s)if(route/10*10+s!=route&&ids[route/10*10+s]==id)expert_r4_error(diag,1u);
  const float scale=scales[ulong(rank)*p.output_size+n],value=dot*scale;const bfloat rounded=bfloat(value);
  if(!(scale>0)||!expert_r4_finite(scale)||!expert_r4_finite(dot)||!expert_r4_finite(value)||!expert_r4_finite(float(rounded)))expert_r4_error(diag,4u);
  raw[sample]=dot;scaled[sample]=value;bf16[sample]=rounded;
}
