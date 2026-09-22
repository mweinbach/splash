#if __METAL_VERSION__ >=410
#ifndef Q4CODED_AUDIT
#define Q4CODED_AUDIT 0
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

kernel void prefill4k_q4coded_input_sums(device const bfloat *input [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],device float *sums [[buffer(2)]],
    device uint *diag [[buffer(3)]],constant Prefill4KQ4CodedSumParams &p [[buffer(4)]],
    uint index [[thread_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
  if (!p.routes || p.routes >81920 || (p.width !=2560 && p.width !=640) || p.groups !=p.width /64 || p.reserved || offsets[512] >p.routes) {
    if (!tid) flash_mpp_error(diag,2u);return;
  }
  if (index >=p.routes *p.groups) return;
  const uint row =index /p.groups,group =index %p.groups;
  float sum =0.0f;
  if (row <offsets[512])
    for (uint k =0; k <64; ++k) sum +=float(input[ulong(row) *p.width +group *64 +k]);
  if (!flash_mpp_finite(sum)) flash_mpp_error(diag,4u);
  sums[index] =sum;
}
inline bool prefill4k_q4coded_job(device const uint *offsets,device const FlashMoEBucketJob *jobs,
    device const uint *count,device const uint *ranks,uint index,uint capacity,uint routes,
    uint stored,uint selector,thread uint &expert,thread uint &begin,thread uint &valid,device uint *diag) {
  if (selector >1 || !stored || stored >512) { flash_mpp_error(diag,2u);return false; }
  uint end;
  if (!flash_direct_a_q4x8_job(offsets,jobs,count,index,capacity,routes,expert,begin,end,diag)) return false;
  const uint rank =ranks[expert];
  if (rank !=UINT_MAX && rank >=stored) { flash_mpp_error(diag,1u);return false; }
  if (!selector && rank !=UINT_MAX) return false;
  valid =min(32u,end -begin);return true;
}
inline void prefill4k_q4coded_gate(device bfloat *input,device uchar *gw,device const uchar *gs,
    device const uchar *gb,device uchar *uw,device const uchar *us,device const uchar *ub,
    device const uint *offsets,device const FlashMoEBucketJob *jobs,device const uint *count,
    device bfloat *output,device uint *diag,device const uint *ranks,device const float *sums,
    device float *linearGate,device float *linearUp,constant Prefill4KQ4CodedGateParams &params,uint3 group,uint3 threads,uint tid) {
  (void)linearGate;(void)linearUp;
  const auto &p =params.source.blocked;const auto &a =p.affine;
  if (params.reserved0 || params.reserved1 || params.reserved2 || params.source.reserved0 || params.source.reserved1 ||
      params.source.flags !=1 || p.reserved || p.tile_rows !=32 || !a.rows || a.rows >8192 || a.selections !=10 ||
      a.input_size !=2560 || a.output_size !=640 || a.experts !=512 || a.reserved0 || a.reserved1 || a.reserved2 ||
      p.route_capacity !=a.rows *10 || p.job_capacity !=(p.route_capacity +31) /32 +511 || group.x >=10 ||
      group.y >=p.job_capacity || group.z || threads.x !=128 || threads.y !=1 || threads.z !=1 ||
      a.gate_weight_row_stride_bytes !=1280 || a.up_weight_row_stride_bytes !=1280 ||
      !flash_direct_a_q4x8_strides(640,2560,a.gate_weight_row_stride_bytes,a.gate_weight_expert_stride_bytes,a.gate_parameter_row_stride_bytes,a.gate_parameter_expert_stride_bytes) ||
      !flash_direct_a_q4x8_strides(640,2560,a.up_weight_row_stride_bytes,a.up_weight_expert_stride_bytes,a.up_parameter_row_stride_bytes,a.up_parameter_expert_stride_bytes)) {
    if (!tid) flash_mpp_error(diag,2u);return;
  }
  uint expert,begin,valid;
  if (!prefill4k_q4coded_job(offsets,jobs,count,ranks,group.y,p.job_capacity,p.route_capacity,params.source.stored_experts,params.selector,expert,begin,valid,diag)) return;
  const uint n0 =group.x *64;
  auto at =tensor(input +ulong(begin) *2560,dextents<int,2>{2560,int(valid)},array<int,2>{1,2560});
  tensor<device uint4b_format,dextents<int,2>,tensor_inline> gt(gw +ulong(expert) *a.gate_weight_expert_stride_bytes +ulong(n0) *1280,
      dextents<int,2>{2560,64},array<int,2>{1,2560});
  tensor<device uint4b_format,dextents<int,2>,tensor_inline> ut(uw +ulong(expert) *a.up_weight_expert_stride_bytes +ulong(n0) *1280,
      dextents<int,2>{2560,64},array<int,2>{1,2560});
  auto a0 =at.slice<64,dynamic_extent>(0,0);auto g0 =gt.slice<64,64>(0,0);auto u0 =ut.slice<64,64>(0,0);
  constexpr auto descriptor =matmul2d_descriptor(32,64,64,false,true,false,matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor,execution_simdgroups<4>> operation;
  auto gate =operation.get_destination_cooperative_tensor<decltype(a0),decltype(g0),float>();
  auto up =operation.get_destination_cooperative_tensor<decltype(a0),decltype(u0),float>();
  auto gdot =operation.get_destination_cooperative_tensor<decltype(a0),decltype(g0),float>();
  auto udot =operation.get_destination_cooperative_tensor<decltype(a0),decltype(u0),float>();
  for (ushort i =0; i <gate.get_capacity(); ++i) if (gate.is_valid_element(i)) { gate[i] =0;up[i] =0; }
  for (uint kgroup =0; kgroup <40; ++kgroup) {
    auto achunk =at.slice<64,dynamic_extent>(kgroup *64,0);
    auto gchunk =gt.slice<64,64>(kgroup *64,0);auto uchunk =ut.slice<64,64>(kgroup *64,0);
    operation.run(achunk,gchunk,gdot);operation.run(achunk,uchunk,udot);
    for (ushort i =0; i <gate.get_capacity(); ++i) {
      if (!gate.is_valid_element(i)) continue;
      const auto index =gate.get_multidimensional_index(i);const uint n =n0 +index[0],row =begin +index[1];
      if (uint(index[1]) >=valid) continue;
      const ulong go =ulong(expert) *a.gate_parameter_expert_stride_bytes +ulong(n) *a.gate_parameter_row_stride_bytes +kgroup *2;
      const ulong uo =ulong(expert) *a.up_parameter_expert_stride_bytes +ulong(n) *a.up_parameter_row_stride_bytes +kgroup *2;
      const float sf =float(*reinterpret_cast<device const bfloat *>(gs +go)),bf =float(*reinterpret_cast<device const bfloat *>(gb +go));
      const float uf =float(*reinterpret_cast<device const bfloat *>(us +uo)),ubf =float(*reinterpret_cast<device const bfloat *>(ub +uo));
      const float xsum =sums[ulong(row) *40 +kgroup];
      gate[i] +=gdot[i] *sf +xsum *bf;up[i] +=udot[i] *uf +xsum *ubf;
    }
  }
  for (ushort i =0; i <gate.get_capacity(); ++i) {
    if (!gate.is_valid_element(i)) continue;
    const auto index =gate.get_multidimensional_index(i);if (uint(index[1]) >=valid) continue;
    const bfloat g =bfloat(gate[i]),u =bfloat(up[i]);
#if Q4CODED_AUDIT
    linearGate[ulong(begin +index[1]) *640 +n0 +index[0]] =gate[i];
    linearUp[ulong(begin +index[1]) *640 +n0 +index[0]] =up[i];
#endif
    const bfloat value =(g *flash_direct_a_q4x8_compiled_sigmoid(g)) *u;
    if (!flash_mpp_finite(gate[i]) || !flash_mpp_finite(up[i]) || !flash_mpp_finite(value)) flash_mpp_error(diag,4u);
    output[ulong(begin +index[1]) *640 +n0 +index[0]] =value;
  }
}
inline void prefill4k_q4coded_down(device bfloat *input,device uchar *weights,device const uchar *scales,
    device const uchar *biases,device const uint *offsets,device const FlashMoEBucketJob *jobs,
    device const uint *count,device const uint *map,device bfloat *output,device uint *diag,
    device const uint *ranks,device const float *sums,constant Prefill4KQ4CodedDownParams &params,
    device float *linearDown,uint3 group,uint3 threads,uint tid) {
  (void)linearDown;
  const auto &p =params.source.blocked;const auto &a =p.affine;
  if (params.reserved0 || params.reserved1 || params.reserved2 || params.source.reserved0 || params.source.reserved1 ||
      params.source.flags !=1 || p.reserved || p.tile_rows !=32 || !a.rows || a.rows >8192 || a.selections !=10 ||
      a.input_size !=640 || a.output_size !=2560 || a.experts !=512 || a.reserved0 || a.reserved1 || a.reserved2 ||
      p.route_capacity !=a.rows *10 || p.job_capacity !=(p.route_capacity +31) /32 +511 || group.x >=40 ||
      group.y >=p.job_capacity || group.z || threads.x !=32 || threads.y !=1 || threads.z !=1 || a.weight_row_stride_bytes !=320 ||
      !flash_direct_a_q4x8_strides(2560,640,a.weight_row_stride_bytes,a.weight_expert_stride_bytes,a.parameter_row_stride_bytes,a.parameter_expert_stride_bytes)) {
    if (!tid) flash_mpp_error(diag,2u);return;
  }
  uint expert,begin,valid;
  if (!prefill4k_q4coded_job(offsets,jobs,count,ranks,group.y,p.job_capacity,p.route_capacity,params.source.stored_experts,params.selector,expert,begin,valid,diag)) return;
  const uint n0 =group.x *64;
  auto at =tensor(input +ulong(begin) *640,dextents<int,2>{640,int(valid)},array<int,2>{1,640});
  auto a0 =at.slice<64,dynamic_extent>(0,0);
  constexpr auto descriptor =matmul2d_descriptor(32,64,64,false,true,false,matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor,execution_simdgroups<1>> operation;
  // Original down rows have320-byte packed stride, illegal for uint4b_format
  // device row alignment. Decode original unsigned nibbles into register-owned
  // U8 cooperative operands; no coefficient reconstruction/staging/barrier.
  auto b0 =operation.get_right_input_cooperative_tensor<bfloat,uchar,float>();
  // Type-only regular U8 view selects the matching destination layout. The
  // view is never read; original packed bytes are decoded into b0 below.
  auto typeOnly =tensor(weights,dextents<int,2>{64,64},array<int,2>{1,64});
  auto total =operation.get_destination_cooperative_tensor<decltype(a0),decltype(typeOnly),float>();
  auto dot =operation.get_destination_cooperative_tensor<decltype(a0),decltype(typeOnly),float>();
  for (ushort i =0; i <total.get_capacity(); ++i) if (total.is_valid_element(i)) total[i] =0;
  for (uint kgroup =0; kgroup <10; ++kgroup) {
    for (ushort i =0; i <b0.get_capacity(); ++i) {
      if (!b0.is_valid_element(i)) continue;
      const auto index =b0.get_multidimensional_index(i);const uint n =n0 +index[1],k =kgroup *64 +index[0];
      const uchar packed =weights[ulong(expert) *a.weight_expert_stride_bytes +ulong(n) *320 +k /2];
      b0[i] =uchar((packed >>((k &1) *4)) &15);
    }
    auto achunk =at.slice<64,dynamic_extent>(kgroup *64,0);operation.run(achunk,b0,dot);
    for (ushort i =0; i <total.get_capacity(); ++i) {
      if (!total.is_valid_element(i)) continue;
      const auto index =total.get_multidimensional_index(i);if (uint(index[1]) >=valid) continue;
      const ulong po =ulong(expert) *a.parameter_expert_stride_bytes +ulong(n0 +index[0]) *a.parameter_row_stride_bytes +kgroup *2;
      const float scale =float(*reinterpret_cast<device const bfloat *>(scales +po)),bias =float(*reinterpret_cast<device const bfloat *>(biases +po));
      total[i] +=dot[i] *scale +sums[ulong(begin +index[1]) *10 +kgroup] *bias;
    }
  }
  for (ushort i =0; i <total.get_capacity(); ++i) {
    if (!total.is_valid_element(i)) continue;
    const auto index =total.get_multidimensional_index(i);if (uint(index[1]) >=valid) continue;
    const uint route =map[begin +index[1]];if (route >=p.route_capacity) { flash_mpp_error(diag,1u);continue; }
    const bfloat value =bfloat(total[i]);if (!flash_mpp_finite(total[i]) || !flash_mpp_finite(value)) flash_mpp_error(diag,4u);
#if Q4CODED_AUDIT
    linearDown[ulong(begin +index[1]) *2560 +n0 +index[0]] =total[i];
#endif
    output[ulong(route) *2560 +n0 +index[0]] =value;
  }
}
kernel void prefill4k_q4coded_gate_up_m32_n64(device bfloat *a [[buffer(0)]],device uchar *gw [[buffer(1)]],
    device const uchar *gs [[buffer(2)]],device const uchar *gb [[buffer(3)]],device uchar *uw [[buffer(4)]],
    device const uchar *us [[buffer(5)]],device const uchar *ub [[buffer(6)]],device const uint *offsets [[buffer(7)]],
    device const FlashMoEBucketJob *jobs [[buffer(8)]],device const uint *count [[buffer(9)]],device bfloat *out [[buffer(10)]],
    device uint *diag [[buffer(11)]],device const uint *ranks [[buffer(12)]],device const float *sums [[buffer(13)]],
    device float *linearGate [[buffer(14)]],device float *linearUp [[buffer(15)]],
    constant Prefill4KQ4CodedGateParams &p [[buffer(16)]],uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) {
  prefill4k_q4coded_gate(a,gw,gs,gb,uw,us,ub,offsets,jobs,count,out,diag,ranks,sums,linearGate,linearUp,p,group,threads,tid);
}
kernel void prefill4k_q4coded_down_scatter_m32_n64(device bfloat *a [[buffer(0)]],device uchar *w [[buffer(1)]],
    device const uchar *s [[buffer(2)]],device const uchar *b [[buffer(3)]],device const uint *offsets [[buffer(4)]],
    device const FlashMoEBucketJob *jobs [[buffer(5)]],device const uint *count [[buffer(6)]],device const uint *map [[buffer(7)]],
    device bfloat *out [[buffer(8)]],device uint *diag [[buffer(9)]],device const uint *ranks [[buffer(10)]],device const float *sums [[buffer(11)]],
    device float *linearDown [[buffer(12)]],constant Prefill4KQ4CodedDownParams &p [[buffer(13)]],uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) {
  prefill4k_q4coded_down(a,w,s,b,offsets,jobs,count,map,out,diag,ranks,sums,p,linearDown,group,threads,tid);
}
#endif
