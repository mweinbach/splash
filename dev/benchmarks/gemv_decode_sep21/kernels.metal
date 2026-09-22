// Private direct BF16 × symmetric I8 GEMV numerical alternatives.
// No activation quantization, coefficient cache, MPP or bucket representation.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "flash/FlashGatheredI8QMV.hpp"
#include "prefill4k_allrows_qmv_probe.h"
using namespace metal;
#pragma METAL fp math_mode(safe)

inline bool gemv_decode_sep21_finite(float x) {
  return (as_type<uint>(x) & 0x7f800000u) != 0x7f800000u;
}
inline void gemv_decode_sep21_error(device atomic_uint *diag, uint bits) {
  atomic_fetch_or_explicit(diag,bits,memory_order_relaxed);
}
inline bfloat gemv_decode_sep21_nan() { return bfloat(as_type<float>(0x7fc00000u)); }
inline uint gemv_decode_sep21_rank(device const long *ids, device const uint *ranks,
    ulong route, uint tid, device atomic_uint *diag) {
  const long id=ids[route];
  if (id<0 || id>=512) { if (!tid) gemv_decode_sep21_error(diag,1u); return UINT_MAX; }
  const ulong begin=route/10*10;
  for (uint slot=0;slot<10;++slot)
    if (begin+slot!=route && ids[begin+slot]==id && !tid) gemv_decode_sep21_error(diag,1u);
  const uint rank=ranks[uint(id)];
  if (rank>=512) { if (!tid) gemv_decode_sep21_error(diag,1u); return UINT_MAX; }
  return rank;
}
inline float gemv_decode_sep21_activation(ushort bits, device atomic_uint *diag) {
  if ((bits&0x7f80u)==0x7f80u) { gemv_decode_sep21_error(diag,4u); return 0.0f; }
  return as_type<float>(uint(bits)<<16);
}
#pragma METAL fp math_mode(fast)
inline bfloat gemv_decode_sep21_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent=bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator=bfloat(1.0f)+exponent;
  const bfloat tail=bfloat(1.0f)/denominator;
  return source<bfloat(0.0f)?tail:bfloat(1.0f)-tail;
}
#pragma METAL fp math_mode(safe)

template <ushort Lanes>
inline float4 gemv_decode_sep21_load_a(device const ushort4 *source, uint chunk,
    uint lane, device atomic_uint *diag) {
  static_assert(Lanes==16 || Lanes==32);
  float4 x(0.0f);
  // The two SIMD16 output rows reuse the same four BF16 activations: only
  // the lower half loads, then both halves read its four F32 values.
  if (Lanes==32 || lane<16) {
    const ushort4 bits=source[chunk];
    x=float4(gemv_decode_sep21_activation(bits.x,diag),gemv_decode_sep21_activation(bits.y,diag),
        gemv_decode_sep21_activation(bits.z,diag),gemv_decode_sep21_activation(bits.w,diag));
  }
  if constexpr (Lanes==16) {
    x=float4(simd_shuffle(x.x,lane&15),simd_shuffle(x.y,lane&15),
        simd_shuffle(x.z,lane&15),simd_shuffle(x.w,lane&15));
  }
  return x;
}
template <ushort Lanes>
inline float gemv_decode_sep21_reduce(float4 partial, uint lane) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  // Four per-lane chains, then exactly three component additions, followed
  // by an explicit XOR butterfly of depth log2(Lanes). XOR masks below16
  // never mix the independent lower/upper SIMD16 output rows.
  float result=((partial.x+partial.y)+partial.z)+partial.w;
#pragma unroll
  for (ushort delta=Lanes/2;delta;delta/=2) result+=simd_shuffle_xor(result,delta);
  (void)lane;
  return result;
}
template <ushort Lanes, bool Gate>
inline void gemv_decode_sep21_execute(device const bfloat *input,
    device const char *gate,device const float *gs,device const char *up,device const float *us,
    device const uint *ranks,device const long *ids,device bfloat *output,device atomic_uint *diag,
    constant FlashGatheredI8QMVParams &p,uint3 group,uint3 threads,uint simd,uint lane,uint tid) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  constexpr uint K=Gate?2560:640, Width=Gate?640:2560, Outputs=128/Lanes;
  if (!p.rows || p.rows>16 || p.selections!=10 || p.experts!=512 || p.reserved ||
      group.x>=Width/Outputs || group.y>=p.rows || group.z>=10 ||
      threads.x!=128 || threads.y!=1 || threads.z!=1) {
    if (!tid) gemv_decode_sep21_error(diag,2u); return;
  }
  const ulong route=ulong(group.y)*10+group.z;
  const uint local=lane&(Lanes-1),owner=lane/Lanes;
  const uint n=group.x*Outputs+simd*(32/Lanes)+owner;
  const uint rank=gemv_decode_sep21_rank(ids,ranks,route,tid,diag);
  if constexpr (!Gate) if (rank==UINT_MAX) {
    if (!local) { gemv_decode_sep21_error(diag,5u); output[route*Width+n]=gemv_decode_sep21_nan(); }
    return;
  }
  const ulong row=Gate?ulong(group.y):route;
  const device ushort4 *a=reinterpret_cast<const device ushort4 *>(input+row*K);
  float4 gd(0.0f),ud(0.0f);
  // No coefficient pointer is formed until ID/rank validation has succeeded.
  const device char4 *g=nullptr,*u=nullptr;
  if (rank!=UINT_MAX) {
    g=reinterpret_cast<const device char4 *>(gate+(ulong(rank)*Width+n)*K);
    if constexpr (Gate) u=reinterpret_cast<const device char4 *>(up+(ulong(rank)*Width+n)*K);
  }
  for (uint chunk=local;chunk<K/4;chunk+=Lanes) {
    const float4 x=gemv_decode_sep21_load_a<Lanes>(a,chunk,lane,diag);
    if (rank!=UINT_MAX) {
      gd+=float4(g[chunk])*x;
      if constexpr (Gate) ud+=float4(u[chunk])*x;
    }
  }
  // Invalid gates still inspect every hidden BF16 operand before poisoning.
  if (rank==UINT_MAX) { if (!local) output[route*Width+n]=gemv_decode_sep21_nan(); return; }
  const float dot=gemv_decode_sep21_reduce<Lanes>(gd,lane);
  const float upDot=Gate?gemv_decode_sep21_reduce<Lanes>(ud,lane):0.0f;
  if (!local) {
    const float scale=gs[ulong(rank)*Width+n],scaled=dot*scale;
    bfloat value=bfloat(scaled);
    if constexpr (Gate) {
      const float upScale=us[ulong(rank)*Width+n],upScaled=upDot*upScale;
      const bfloat uv=bfloat(upScaled),silu=value*gemv_decode_sep21_sigmoid(value);
      value=silu*uv;
      if (!(upScale>0) || !gemv_decode_sep21_finite(upScale) || !gemv_decode_sep21_finite(upDot) ||
          !gemv_decode_sep21_finite(upScaled) || !gemv_decode_sep21_finite(float(uv))) gemv_decode_sep21_error(diag,4u);
    }
    if (!(scale>0) || !gemv_decode_sep21_finite(scale) || !gemv_decode_sep21_finite(dot) ||
        !gemv_decode_sep21_finite(scaled) || !gemv_decode_sep21_finite(float(value))) gemv_decode_sep21_error(diag,4u);
    output[route*Width+n]=value;
  }
}

#define GEMV_DECODE_GATE(Name,Lanes) \
kernel void Name(device const bfloat *a [[buffer(0)]],device const char *g [[buffer(1)]], \
 device const float *gs [[buffer(2)]],device const char *u [[buffer(3)]],device const float *us [[buffer(4)]], \
 device const uint *ranks [[buffer(5)]],device const long *ids [[buffer(6)]],device bfloat *out [[buffer(7)]], \
 device atomic_uint *diag [[buffer(8)]],constant FlashGatheredI8QMVParams &p [[buffer(9)]], \
 uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]], \
 uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint tid [[thread_index_in_threadgroup]]) { \
 gemv_decode_sep21_execute<Lanes,true>(a,g,gs,u,us,ranks,ids,out,diag,p,group,threads,simd,lane,tid); }
#define GEMV_DECODE_DOWN(Name,Lanes) \
kernel void Name(device const bfloat *a [[buffer(0)]],device const char *g [[buffer(1)]],device const float *gs [[buffer(2)]], \
 device const uint *ranks [[buffer(3)]],device const long *ids [[buffer(4)]],device bfloat *out [[buffer(5)]], \
 device atomic_uint *diag [[buffer(6)]],constant FlashGatheredI8QMVParams &p [[buffer(7)]], \
 uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]], \
 uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint tid [[thread_index_in_threadgroup]]) { \
 gemv_decode_sep21_execute<Lanes,false>(a,g,gs,g,gs,ranks,ids,out,diag,p,group,threads,simd,lane,tid); }
GEMV_DECODE_GATE(gemv_decode_sep21_v4_l32_o4_gate_up,32)
GEMV_DECODE_DOWN(gemv_decode_sep21_v4_l32_o4_down,32)
GEMV_DECODE_GATE(gemv_decode_sep21_v4_l16_o8_gate_up,16)
GEMV_DECODE_DOWN(gemv_decode_sep21_v4_l16_o8_down,16)

inline bool gemv_decode_sep21_probe_geometry(constant FlashQMVProbeParams &p,uint3 group,
    uint3 threads,uint outputs,uint tid,device atomic_uint *diag) {
  const bool phase=(p.input_size==2560&&p.output_size==640&&!p.per_route_input) ||
      (p.input_size==640&&p.output_size==2560&&p.per_route_input==1);
  if (!p.rows || p.rows>16 || p.selections!=10 || p.experts!=512 || p.reserved || p.columns!=4 || !phase ||
      group.x>=p.output_size/outputs || group.y>=p.rows || group.z>=10 ||
      threads.x!=128 || threads.y!=1 || threads.z!=1) {
    if (!tid) gemv_decode_sep21_error(diag,2u); return false;
  }
  return true;
}
inline void gemv_decode_sep21_projection_write(float dot,uint rank,uint n,ulong out,uint width,
    device const float *scales,device float *raw,device float *scaled,device bfloat *bf16,device atomic_uint *diag) {
  const float scale=scales[ulong(rank)*width+n],value=dot*scale;
  const bfloat rounded=bfloat(value);
  if (!(scale>0) || !gemv_decode_sep21_finite(scale) || !gemv_decode_sep21_finite(dot) ||
      !gemv_decode_sep21_finite(value) || !gemv_decode_sep21_finite(float(rounded))) gemv_decode_sep21_error(diag,4u);
  raw[out]=dot; scaled[out]=value; bf16[out]=rounded;
}
inline void gemv_decode_sep21_projection_poison(ulong out,device float *raw,device float *scaled,
    device bfloat *bf16,device atomic_uint *diag) {
  const float nan=as_type<float>(0x7fc00000u);
  raw[out]=nan; scaled[out]=nan; bf16[out]=bfloat(nan); gemv_decode_sep21_error(diag,5u);
}
template <ushort Lanes>
inline void gemv_decode_sep21_probe(device const bfloat *input,device const char *codes,
    device const float *scales,device const uint *ranks,device const long *ids,device float *raw,
    device float *scaled,device bfloat *bf16,device atomic_uint *diag,constant FlashQMVProbeParams &p,
    uint3 group,uint3 threads,uint simd,uint lane,uint tid) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  constexpr uint Outputs=128/Lanes;
  if (!gemv_decode_sep21_probe_geometry(p,group,threads,Outputs,tid,diag)) return;
  const ulong route=ulong(group.y)*10+group.z;
  const uint local=lane&(Lanes-1),n=group.x*Outputs+simd*(32/Lanes)+lane/Lanes;
  const uint rank=gemv_decode_sep21_rank(ids,ranks,route,tid,diag);
  if (p.per_route_input && rank==UINT_MAX) {
    if (!local) gemv_decode_sep21_projection_poison(route*p.output_size+n,raw,scaled,bf16,diag); return;
  }
  const ulong row=p.per_route_input?route:ulong(group.y);
  const device ushort4 *a=reinterpret_cast<const device ushort4 *>(input+row*p.input_size);
  const device char4 *c=nullptr;
  if (rank!=UINT_MAX) c=reinterpret_cast<const device char4 *>(codes+(ulong(rank)*p.output_size+n)*p.input_size);
  float4 partial(0.0f);
  for (uint chunk=local;chunk<p.input_size/4;chunk+=Lanes) {
    const float4 x=gemv_decode_sep21_load_a<Lanes>(a,chunk,lane,diag);
    if (rank!=UINT_MAX) partial+=float4(c[chunk])*x;
  }
  if (rank==UINT_MAX) {
    if (!local) gemv_decode_sep21_projection_poison(route*p.output_size+n,raw,scaled,bf16,diag); return;
  }
  const float dot=gemv_decode_sep21_reduce<Lanes>(partial,lane);
  if (!local) gemv_decode_sep21_projection_write(dot,rank,n,route*p.output_size+n,p.output_size,
      scales,raw,scaled,bf16,diag);
}
#define GEMV_DECODE_PROBE(Name,Lanes) \
kernel void Name(device const bfloat *a [[buffer(0)]],device const char *c [[buffer(1)]],device const float *s [[buffer(2)]], \
 device const uint *ranks [[buffer(3)]],device const long *ids [[buffer(4)]],device float *raw [[buffer(5)]], \
 device float *scaled [[buffer(6)]],device bfloat *bf16 [[buffer(7)]],device atomic_uint *diag [[buffer(8)]], \
 constant FlashQMVProbeParams &p [[buffer(9)]],uint3 group [[threadgroup_position_in_grid]], \
 uint3 threads [[threads_per_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]], \
 uint lane [[thread_index_in_simdgroup]],uint tid [[thread_index_in_threadgroup]]) { \
 gemv_decode_sep21_probe<Lanes>(a,c,s,ranks,ids,raw,scaled,bf16,diag,p,group,threads,simd,lane,tid); }
GEMV_DECODE_PROBE(gemv_decode_sep21_v4_l32_o4_projection_probe,32)
GEMV_DECODE_PROBE(gemv_decode_sep21_v4_l16_o8_projection_probe,16)

// Scalar dot evidence is separate from shipping chains and is never timed.
// Pairs[sample]={flattened canonical route,output column}; taps are compact.
kernel void gemv_decode_sep21_scalar_projection_samples(
    device const bfloat *input [[buffer(0)]],device const char *codes [[buffer(1)]],
    device const float *scales [[buffer(2)]],device const uint *ranks [[buffer(3)]],device const long *ids [[buffer(4)]],
    device float *raw [[buffer(5)]],device float *scaled [[buffer(6)]],device bfloat *bf16 [[buffer(7)]],
    device atomic_uint *diag [[buffer(8)]],constant FlashQMVProbeParams &p [[buffer(9)]],
    device const uint2 *pairs [[buffer(10)]],constant uint &count [[buffer(11)]],
    uint sample [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  if (sample>=count) return;
  const bool phase=(p.input_size==2560&&p.output_size==640&&!p.per_route_input) ||
      (p.input_size==640&&p.output_size==2560&&p.per_route_input==1);
  if (!p.rows || p.rows>16 || p.selections!=10 || p.experts!=512 || p.reserved || !phase ||
      count>p.rows*10*p.output_size) { if (!sample) gemv_decode_sep21_error(diag,2u); return; }
  const uint2 pair=pairs[sample]; const ulong route=pair.x; const uint n=pair.y;
  if (route>=p.rows*10 || n>=p.output_size) {
    gemv_decode_sep21_error(diag,2u); gemv_decode_sep21_projection_poison(sample,raw,scaled,bf16,diag); return;
  }
  const uint rank=gemv_decode_sep21_rank(ids,ranks,route,0,diag);
  if (p.per_route_input&&rank==UINT_MAX) {
    gemv_decode_sep21_projection_poison(sample,raw,scaled,bf16,diag); return;
  }
  const ulong row=p.per_route_input?route:route/10;
  const device ushort *a=reinterpret_cast<const device ushort *>(input+row*p.input_size);
  const device char *c=nullptr;
  if (rank!=UINT_MAX) c=codes+(ulong(rank)*p.output_size+n)*p.input_size;
  float dot=0.0f;
  for (uint k=0;k<p.input_size;++k) {
    const float x=gemv_decode_sep21_activation(a[k],diag);
    if (rank!=UINT_MAX) dot+=float(c[k])*x;
  }
  if (rank==UINT_MAX) { gemv_decode_sep21_projection_poison(sample,raw,scaled,bf16,diag); return; }
  gemv_decode_sep21_projection_write(dot,rank,n,sample,p.output_size,scales,raw,scaled,bf16,diag);
}
