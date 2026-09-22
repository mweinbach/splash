#if __METAL_VERSION__ >=410
#ifndef Q4NAX_AUDIT
#define Q4NAX_AUDIT 0
#endif
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "dev/benchmarks/prefill4k_q4coded/params.h"
#include "metal/kernels/common/flash_moe_direct_a_common.h"
#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
using namespace metal;
using namespace mpp::tensor_ops;

inline bfloat q4nax_coefficient(device const uchar *w,device const uchar *s,device const uchar *b,
    ulong wr,ulong we,ulong pr,ulong pe,uint expert,uint n,uint k,device uint *diag) {
  const uchar packed=w[ulong(expert) *we +ulong(n) *wr +k /2];
  const uint code=(packed >>((k &1) *4)) &15;
  const ulong parameter=ulong(expert) *pe +ulong(n) *pr +ulong(k /64) *2;
  const bfloat scale=*reinterpret_cast<device const bfloat *>(s +parameter),bias=*reinterpret_cast<device const bfloat *>(b +parameter);
  const float reconstructed=flash_mpp_dequantize_f32(code,scale,bias);const bfloat value=bfloat(reconstructed);
  if (!flash_mpp_finite(reconstructed) || !flash_mpp_finite(value)) { flash_mpp_error(diag,4u);return bfloat(0); }
  return value;
}
inline bool q4nax_job(device const uint *offsets,device const FlashMoEBucketJob *jobs,
    device const uint *count,device const uint *ranks,uint index,uint capacity,uint routes,
    uint stored,uint selector,thread uint &expert,thread uint &begin,thread uint &end,device uint *diag) {
  if (selector >1 || !stored || stored >512) { flash_mpp_error(diag,2u);return false; }
  if (!flash_direct_a_q4x8_job(offsets,jobs,count,index,capacity,routes,expert,begin,end,diag)) return false;
  const uint rank=ranks[expert];if (rank !=UINT_MAX &&rank >=stored) { flash_mpp_error(diag,1u);return false; }
  return selector ||rank ==UINT_MAX;
}
template<bool Relaxed> inline void q4nax_gate(device bfloat *input,device uchar *gw,device const uchar *gs,device const uchar *gb,
    device uchar *uw,device const uchar *us,device const uchar *ub,device const uint *offsets,
    device const FlashMoEBucketJob *jobs,device const uint *count,device bfloat *out,device uint *diag,
    device const uint *ranks,device const float *unusedSums,device float *linearGate,device float *linearUp,
    constant Prefill4KQ4CodedGateParams &params,uint3 group,uint3 threads,uint tid) {
  (void)unusedSums;(void)linearGate;(void)linearUp;
  const auto &p=params.source.blocked;const auto &a=p.affine;
  if (params.reserved0 ||params.reserved1 ||params.reserved2 ||params.source.reserved0 ||params.source.reserved1 ||params.source.flags !=1 ||
      p.reserved ||p.tile_rows !=32 || !a.rows ||a.rows >8192 ||a.selections !=10 ||a.input_size !=2560 ||a.output_size !=640 ||
      a.experts !=512 ||a.reserved0 ||a.reserved1 ||a.reserved2 ||p.route_capacity !=a.rows *10 ||p.job_capacity !=(p.route_capacity +31) /32 +511 ||
      group.x >=10 ||group.y >=p.job_capacity ||group.z ||threads.x !=128 ||threads.y !=1 ||threads.z !=1 ||
      !flash_direct_a_q4x8_strides(640,2560,a.gate_weight_row_stride_bytes,a.gate_weight_expert_stride_bytes,a.gate_parameter_row_stride_bytes,a.gate_parameter_expert_stride_bytes) ||
      !flash_direct_a_q4x8_strides(640,2560,a.up_weight_row_stride_bytes,a.up_weight_expert_stride_bytes,a.up_parameter_row_stride_bytes,a.up_parameter_expert_stride_bytes)) {
    if (!tid) flash_mpp_error(diag,2u);return;
  }
  uint expert,begin,end;if (!q4nax_job(offsets,jobs,count,ranks,group.y,p.job_capacity,p.route_capacity,params.source.stored_experts,params.selector,expert,begin,end,diag)) return;
  const uint simd=tid /32,m0=(simd /2) *16,n0=group.x *64 +(simd %2) *32;
  constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,Relaxed,matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor,execution_simdgroup> operation;
  auto left=operation.template get_left_input_cooperative_tensor<bfloat,bfloat,float>();
  auto right=operation.template get_right_input_cooperative_tensor<bfloat,bfloat,float>();
  auto gate=operation.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
  auto up=operation.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
  for (ushort i=0;i <gate.get_capacity();++i) if (gate.is_valid_element(i)) { gate[i]=0;up[i]=0; }
  // Four ascending K16 MMAs per original G64. The F32 accumulator persists
  // across all groups; original coefficients are once-rounded BF16 registers.
  for (uint groupK=0;groupK <40;++groupK)
    for (uint part=0;part <4;++part) {
      const uint k0=groupK *64 +part *16;
      for (ushort i=0;i <left.get_capacity();++i) {
        if (!left.is_valid_element(i)) continue;const auto index=left.get_multidimensional_index(i);
        const uint row=begin +m0 +index[1];left[i]=row <end ? input[ulong(row) *2560 +k0 +index[0]] :bfloat(0);
      }
      for (ushort i=0;i <right.get_capacity();++i) {
        if (!right.is_valid_element(i)) continue;const auto index=right.get_multidimensional_index(i);
        right[i]=q4nax_coefficient(gw,gs,gb,a.gate_weight_row_stride_bytes,a.gate_weight_expert_stride_bytes,
            a.gate_parameter_row_stride_bytes,a.gate_parameter_expert_stride_bytes,expert,n0 +index[1],k0 +index[0],diag);
      }
      operation.run(left,right,gate);
      for (ushort i=0;i <right.get_capacity();++i) {
        if (!right.is_valid_element(i)) continue;const auto index=right.get_multidimensional_index(i);
        right[i]=q4nax_coefficient(uw,us,ub,a.up_weight_row_stride_bytes,a.up_weight_expert_stride_bytes,
            a.up_parameter_row_stride_bytes,a.up_parameter_expert_stride_bytes,expert,n0 +index[1],k0 +index[0],diag);
      }
      operation.run(left,right,up);
    }
  for (ushort i=0;i <gate.get_capacity();++i) {
    if (!gate.is_valid_element(i)) continue;const auto index=gate.get_multidimensional_index(i);const uint row=begin +m0 +index[1],n=n0 +index[0];
    if (row >=end) continue;const bfloat g=bfloat(gate[i]),u=bfloat(up[i]);const bfloat value=(g *flash_direct_a_q4x8_compiled_sigmoid(g)) *u;
    if (!flash_mpp_finite(gate[i]) || !flash_mpp_finite(up[i]) || !flash_mpp_finite(value)) flash_mpp_error(diag,4u);
#if Q4NAX_AUDIT
    linearGate[ulong(row) *640 +n]=gate[i];linearUp[ulong(row) *640 +n]=up[i];
#endif
    out[ulong(row) *640 +n]=value;
  }
}
template<bool Relaxed> inline void q4nax_down(device bfloat *input,device uchar *w,device const uchar *s,device const uchar *b,
    device const uint *offsets,device const FlashMoEBucketJob *jobs,device const uint *count,device const uint *map,
    device bfloat *out,device uint *diag,device const uint *ranks,device const float *unusedSums,device float *linear,
    constant Prefill4KQ4CodedDownParams &params,uint3 group,uint3 threads,uint tid) {
  (void)unusedSums;(void)linear;const auto &p=params.source.blocked;const auto &a=p.affine;
  if (params.reserved0 ||params.reserved1 ||params.reserved2 ||params.source.reserved0 ||params.source.reserved1 ||params.source.flags !=1 ||
      p.reserved ||p.tile_rows !=32 || !a.rows ||a.rows >8192 ||a.selections !=10 ||a.input_size !=640 ||a.output_size !=2560 ||
      a.experts !=512 ||a.reserved0 ||a.reserved1 ||a.reserved2 ||p.route_capacity !=a.rows *10 ||p.job_capacity !=(p.route_capacity +31) /32 +511 ||
      group.x >=40 ||group.y >=p.job_capacity ||group.z ||threads.x !=128 ||threads.y !=1 ||threads.z !=1 ||
      !flash_direct_a_q4x8_strides(2560,640,a.weight_row_stride_bytes,a.weight_expert_stride_bytes,a.parameter_row_stride_bytes,a.parameter_expert_stride_bytes)) {
    if (!tid) flash_mpp_error(diag,2u);return;
  }
  uint expert,begin,end;if (!q4nax_job(offsets,jobs,count,ranks,group.y,p.job_capacity,p.route_capacity,params.source.stored_experts,params.selector,expert,begin,end,diag)) return;
  const uint simd=tid /32,m0=(simd /2) *16,n0=group.x *64 +(simd %2) *32;
  constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,Relaxed,matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor,execution_simdgroup> operation;
  auto left=operation.template get_left_input_cooperative_tensor<bfloat,bfloat,float>();auto right=operation.template get_right_input_cooperative_tensor<bfloat,bfloat,float>();
  auto total=operation.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
  for (ushort i=0;i <total.get_capacity();++i) if (total.is_valid_element(i)) total[i]=0;
  for (uint groupK=0;groupK <10;++groupK)
    for (uint part=0;part <4;++part) {
      const uint k0=groupK *64 +part *16;
      for (ushort i=0;i <left.get_capacity();++i) {
        if (!left.is_valid_element(i)) continue;const auto index=left.get_multidimensional_index(i);const uint row=begin +m0 +index[1];
        left[i]=row <end ? input[ulong(row) *640 +k0 +index[0]] :bfloat(0);
      }
      for (ushort i=0;i <right.get_capacity();++i) {
        if (!right.is_valid_element(i)) continue;const auto index=right.get_multidimensional_index(i);
        right[i]=q4nax_coefficient(w,s,b,a.weight_row_stride_bytes,a.weight_expert_stride_bytes,a.parameter_row_stride_bytes,
            a.parameter_expert_stride_bytes,expert,n0 +index[1],k0 +index[0],diag);
      }
      operation.run(left,right,total);
    }
  for (ushort i=0;i <total.get_capacity();++i) {
    if (!total.is_valid_element(i)) continue;const auto index=total.get_multidimensional_index(i);const uint row=begin +m0 +index[1],n=n0 +index[0];
    if (row >=end) continue;const uint route=map[row];if (route >=p.route_capacity) { flash_mpp_error(diag,1u);continue; }
    const bfloat value=bfloat(total[i]);if (!flash_mpp_finite(total[i]) || !flash_mpp_finite(value)) flash_mpp_error(diag,4u);
#if Q4NAX_AUDIT
    linear[ulong(row) *2560 +n]=total[i];
#endif
    out[ulong(route) *2560 +n]=value;
  }
}
#define Q4NAX_GATE(NAME, RELAXED) \
kernel void NAME(device bfloat *a [[buffer(0)]],device uchar *gw [[buffer(1)]],device const uchar *gs [[buffer(2)]],device const uchar *gb [[buffer(3)]], \
    device uchar *uw [[buffer(4)]],device const uchar *us [[buffer(5)]],device const uchar *ub [[buffer(6)]],device const uint *offsets [[buffer(7)]], \
    device const FlashMoEBucketJob *jobs [[buffer(8)]],device const uint *count [[buffer(9)]],device bfloat *out [[buffer(10)]],device uint *diag [[buffer(11)]], \
    device const uint *ranks [[buffer(12)]],device const float *unused [[buffer(13)]],device float *lg [[buffer(14)]],device float *lu [[buffer(15)]], \
    constant Prefill4KQ4CodedGateParams &p [[buffer(16)]],uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) { \
  q4nax_gate<RELAXED>(a,gw,gs,gb,uw,us,ub,offsets,jobs,count,out,diag,ranks,unused,lg,lu,p,group,threads,tid); }
#define Q4NAX_DOWN(NAME, RELAXED) \
kernel void NAME(device bfloat *a [[buffer(0)]],device uchar *w [[buffer(1)]],device const uchar *s [[buffer(2)]],device const uchar *b [[buffer(3)]],device const uint *offsets [[buffer(4)]], \
    device const FlashMoEBucketJob *jobs [[buffer(5)]],device const uint *count [[buffer(6)]],device const uint *map [[buffer(7)]],device bfloat *out [[buffer(8)]],device uint *diag [[buffer(9)]], \
    device const uint *ranks [[buffer(10)]],device const float *unused [[buffer(11)]],device float *linear [[buffer(12)]],constant Prefill4KQ4CodedDownParams &p [[buffer(13)]], \
    uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) { \
  q4nax_down<RELAXED>(a,w,s,b,offsets,jobs,count,map,out,diag,ranks,unused,linear,p,group,threads,tid); }
Q4NAX_GATE(prefill4k_q4nax_gate_up_strict_m32_n64,false)
Q4NAX_GATE(prefill4k_q4nax_gate_up_relaxed_m32_n64,true)
Q4NAX_DOWN(prefill4k_q4nax_down_scatter_strict_m32_n64,false)
Q4NAX_DOWN(prefill4k_q4nax_down_scatter_relaxed_m32_n64,true)
#undef Q4NAX_GATE
#undef Q4NAX_DOWN
#endif
