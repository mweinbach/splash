// Private numerical alternative. Fresh dense2K only. Global F32 probabilities.
// QK and PV use whole-K MPP; original prefix/cache and BF16 gating stay separate.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashQSAFast.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;
inline bool twopass_geometry(constant FlashQSAFastParams &p,uint3 threads) {
  return p.common.rows==2048 && p.common.begin==0 && p.common.capacity>=2048 &&
      p.common.capacity<=262144 && p.partitions==4 && p.maximum_partitions==32 &&
      all(threads==uint3(256,1,1));
}
inline void twopass_fail(device atomic_uint *diag,uint reason) {
  atomic_fetch_or_explicit(diag,reason,memory_order_relaxed);
}
kernel void sep21_qsa_twopass_pack_q(device const bfloat *source [[buffer(0)]],
    device bfloat *packed [[buffer(1)]],device const uint *selected [[buffer(2)]],
    device atomic_uint *diag [[buffer(3)]],constant FlashQSAFastParams &p [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) {
  if (!twopass_geometry(p,threads) || group.y || group.z) {
    if (!tid) twopass_fail(diag,1u<<9);return;
  }
  const ulong index=ulong(group.x)*256+tid;
  if (index>=ulong(2)*24576*256) return;
  const uint kv=uint(index/(24576*256)),flat=uint(index/256)%24576,d=uint(index%256);
  const uint row=flat/12,head=kv*12+flat%12;
  if (!kv && !(flat%12) && !d) {
    for (uint block=0;block<(row+1)/4;++block)
      if (selected[ulong(row)*512+block]!=block) twopass_fail(diag,1u<<9);
  }
  packed[index]=source[(ulong(row)*24+head)*256+d];
}
kernel void sep21_qsa_twopass_qk_m128_n64(device bfloat *q [[buffer(0)]],
    device bfloat *keys [[buffer(1)]],device float *scores [[buffer(2)]],
    device atomic_uint *diag [[buffer(3)]],constant FlashQSAFastParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (!twopass_geometry(p,threads) || group.x>=32 || group.y>=192 || group.z>=2) {
    if (!tid) twopass_fail(diag,1u<<9);return;
  }
  const uint row=group.y*128,token=group.x*64,kv=group.z;
  auto a=tensor(q+ulong(kv)*24576*256+ulong(row)*256,
      dextents<int,2>{256,128},array<int,2>{1,256});
  auto b=tensor(keys+(ulong(token)*2+kv)*256,
      dextents<int,2>{256,64},array<int,2>{1,512});
  constexpr auto descriptor=matmul2d_descriptor(128,64,256,false,true,false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor,execution_simdgroups<8>> operation;
  auto dot=operation.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
  operation.run(a,b,dot);
  for (ushort i=0;i<dot.get_capacity();++i) if (dot.is_valid_element(i)) {
    const auto c=dot.get_multidimensional_index(i);
    const uint flat=row+c[1],t=token+c[0],count=flat/12+1;
    const float value=t<count?dot[i]*0.0625f:-INFINITY;
    scores[(ulong(kv)*24576+flat)*2048+t]=value;
    if (t<count && !isfinite(value)) twopass_fail(diag,1u<<8);
  }
}
kernel void sep21_qsa_twopass_softmax_f32(device float *scores [[buffer(0)]],
    device atomic_uint *diag [[buffer(1)]],constant FlashQSAFastParams &p [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  if (!twopass_geometry(p,threads) || group.x>=24576 || group.y>=2 || group.z) {
    if (!tid) twopass_fail(diag,1u<<9);return;
  }
  const uint count=group.x/12+1;const ulong base=(ulong(group.y)*24576+group.x)*2048;
  threadgroup float partial[8],state[2];
  float retained[8],maximum=-INFINITY;
  for (uint i=0;i<8;++i) {
    const uint token=tid+i*256;
    retained[i]=token<count?scores[base+token]:-INFINITY;
    maximum=max(maximum,retained[i]);
    if (token<count && !isfinite(retained[i])) twopass_fail(diag,1u<<8);
  }
  maximum=simd_max(maximum);if (!lane) partial[simd]=maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!tid) {maximum=partial[0];for (uint i=1;i<8;++i) maximum=max(maximum,partial[i]);state[0]=maximum;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maximum=state[0];float sum=0;
  for (uint i=0;i<8;++i) {retained[i]=tid+i*256<count?exp(retained[i]-maximum):0;sum+=retained[i];}
  sum=simd_sum(sum);if (!lane) partial[simd]=sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!tid) {sum=partial[0];for (uint i=1;i<8;++i) sum+=partial[i];state[1]=sum;
    if (!(sum>0) || !isfinite(sum) || !isfinite(maximum)) twopass_fail(diag,1u<<8);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sum=state[1];
  for (uint i=0;i<8;++i) scores[base+tid+i*256]=retained[i]/sum;
}
kernel void sep21_qsa_twopass_pack_v(device const bfloat *values [[buffer(0)]],
    device bfloat *packed [[buffer(1)]],device atomic_uint *diag [[buffer(2)]],
    constant FlashQSAFastParams &p [[buffer(3)]],uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) {
  if (!twopass_geometry(p,threads) || group.y || group.z) {
    if (!tid) twopass_fail(diag,1u<<9);return;
  }
  const ulong index=ulong(group.x)*256+tid;if (index>=ulong(2)*256*2048) return;
  const uint kv=uint(index/(256*2048)),d=uint(index/2048)%256,token=uint(index%2048);
  packed[index]=values[(ulong(token)*2+kv)*256+d];
}
template<bool PackedV>
inline void twopass_pv(device float *probability,device bfloat *values,device float *raw,
    device atomic_uint *diag,constant FlashQSAFastParams &p,uint3 group,uint3 threads,uint tid) {
  if (!twopass_geometry(p,threads) || group.x>=4 || group.y>=192 || group.z>=2) {
    if (!tid) twopass_fail(diag,1u<<9);return;
  }
  const uint row=group.y*128,column=group.x*64,kv=group.z;
  auto a=tensor(probability+(ulong(kv)*24576+row)*2048,
      dextents<int,2>{2048,128},array<int,2>{1,2048});
  device bfloat *vBegin=PackedV?values+ulong(kv)*256*2048+ulong(column)*2048:values+kv*256+column;
  auto b=tensor(vBegin,dextents<int,2>{2048,64},
      PackedV?array<int,2>{1,2048}:array<int,2>{512,1});
  constexpr auto descriptor=matmul2d_descriptor(128,64,2048,false,true,false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor,execution_simdgroups<8>> operation;
  auto dot=operation.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
  operation.run(a,b,dot);
  for (ushort i=0;i<dot.get_capacity();++i) if (dot.is_valid_element(i)) {
    const auto c=dot.get_multidimensional_index(i);const uint flat=row+c[1],d=column+c[0];
    raw[(ulong(kv)*24576+flat)*256+d]=dot[i];
    if (!isfinite(dot[i])) twopass_fail(diag,1u<<8);
  }
}
#define TWOPASS_PV_ENTRY(Name,Packed) \
kernel void Name(device float *p [[buffer(0)]],device bfloat *v [[buffer(1)]], \
    device float *raw [[buffer(2)]],device atomic_uint *diag [[buffer(3)]], \
    constant FlashQSAFastParams &params [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) { \
  twopass_pv<Packed>(p,v,raw,diag,params,group,threads,tid); \
}
TWOPASS_PV_ENTRY(sep21_qsa_twopass_pv_m128_n64,false)
TWOPASS_PV_ENTRY(sep21_qsa_twopass_pv_packed_v_m128_n64,true)
// Exact source gate/reducer scalar precision policy, after the attention helpers.
#pragma METAL fp math_mode(fast)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
inline bfloat twopass_source_gate(bfloat attention,bfloat source) {
  const bfloat exponential=bfloat(precise::exp(abs(float(source))));
  const bfloat denominator=bfloat(1.0f)+exponential;
  const bfloat tail=bfloat(1.0f)/denominator;
  const bfloat sigmoid=source<bfloat(0.0f)?tail:bfloat(1.0f)-tail;
  return attention*sigmoid;
}
kernel void sep21_qsa_twopass_unpack_gate(device const float *raw [[buffer(0)]],
    device const bfloat *projection [[buffer(1)]],device bfloat *output [[buffer(2)]],
    device atomic_uint *diag [[buffer(3)]],constant FlashQSAFastParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (!twopass_geometry(p,threads) || group.y || group.z) {
    if (!tid) twopass_fail(diag,1u<<9);return;
  }
  const ulong index=ulong(group.x)*256+tid;if (index>=ulong(2048)*6144) return;
  const uint row=uint(index/6144),head=uint(index%6144)/256,d=uint(index%256);
  const float value=raw[(ulong(head/12)*24576+row*12+head%12)*256+d];
  const bfloat gate=projection[ulong(row)*12288+head*512+256+d];
  const bfloat result=twopass_source_gate(bfloat(value),gate);output[index]=result;
  if (!isfinite(value) || !isfinite(float(gate)) || !isfinite(float(result))) twopass_fail(diag,1u<<8);
}
// Diagnostic-only original p1/p4 merge, before original BF16/gate boundaries.
kernel void sep21_qsa_twopass_control_raw(device const float *stats [[buffer(0)]],
    device const float *numerators [[buffer(1)]],device float *raw [[buffer(2)]],
    constant FlashQSAFastParams &p [[buffer(3)]],uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (group.x>=2048 || group.y>=24 || tid>=256 || p.maximum_partitions!=4) return;
  const uint row=group.x,head=group.y,partitions=row<128?1u:4u;
  const ulong base=(ulong(row)*24+head)*4;float maximum=-INFINITY;
  for (uint i=0;i<partitions;++i) maximum=max(maximum,stats[(base+i)*2]);
  float sum=0,value=0;
  for (uint i=0;i<partitions;++i) {const float local=stats[(base+i)*2+1];if (!local) continue;
    const float factor=exp(stats[(base+i)*2]-maximum);sum+=local*factor;value+=numerators[(base+i)*256+tid]*factor;}
  raw[(ulong(head/12)*24576+row*12+head%12)*256+tid]=value/sum;
}
