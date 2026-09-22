// Private VerifyR4-only padding producer. Timed down arithmetic is the entire
// original SG4 legacy helper, included verbatim under renamed private symbols.
#include "abi.hpp"
#define flash_hc_fused_down_q4_g32_s4 hc_pad_original_down_q4_g32_s4
#define flash_hc_fused_up_mix_q4_g32_s4 hc_pad_original_up_mix_q4_g32_s4
#define flash_hc_fused_down_q4_g32_s8 hc_pad_original_down_q4_g32_s8
#define flash_hc_fused_up_mix_q4_g32_s8 hc_pad_original_up_mix_q4_g32_s8
#define flash_hc_fused_down_q4_g64_s4 hc_pad_original_down_q4_g64_s4
#define flash_hc_fused_up_mix_q4_g64_s4 hc_pad_original_up_mix_q4_g64_s4
#define flash_hc_fused_down_q4_g64_s8 hc_pad_original_down_q4_g64_s8
#define flash_hc_fused_up_mix_q4_g64_s8 hc_pad_original_up_mix_q4_g64_s8
#define flash_hc_fused_down_q4_g128_s4 hc_pad_original_down_q4_g128_s4
#define flash_hc_fused_up_mix_q4_g128_s4 hc_pad_original_up_mix_q4_g128_s4
#define flash_hc_fused_down_q4_g128_s8 hc_pad_original_down_q4_g128_s8
#define flash_hc_fused_up_mix_q4_g128_s8 hc_pad_original_up_mix_q4_g128_s8
#define flash_hc_fused_down_q5_g32_s4 hc_pad_original_down_q5_g32_s4
#define flash_hc_fused_up_mix_q5_g32_s4 hc_pad_original_up_mix_q5_g32_s4
#define flash_hc_fused_down_q5_g32_s8 hc_pad_original_down_q5_g32_s8
#define flash_hc_fused_up_mix_q5_g32_s8 hc_pad_original_up_mix_q5_g32_s8
#define flash_hc_fused_down_q5_g64_s4 hc_pad_original_down_q5_g64_s4
#define flash_hc_fused_up_mix_q5_g64_s4 hc_pad_original_up_mix_q5_g64_s4
#define flash_hc_fused_down_q5_g64_s8 hc_pad_original_down_q5_g64_s8
#define flash_hc_fused_up_mix_q5_g64_s8 hc_pad_original_up_mix_q5_g64_s8
#define flash_hc_fused_down_q5_g128_s4 hc_pad_original_down_q5_g128_s4
#define flash_hc_fused_up_mix_q5_g128_s4 hc_pad_original_up_mix_q5_g128_s4
#define flash_hc_fused_down_q5_g128_s8 hc_pad_original_down_q5_g128_s8
#define flash_hc_fused_up_mix_q5_g128_s8 hc_pad_original_up_mix_q5_g128_s8
#define flash_hc_fused_down_q6_g32_s4 hc_pad_original_down_q6_g32_s4
#define flash_hc_fused_up_mix_q6_g32_s4 hc_pad_original_up_mix_q6_g32_s4
#define flash_hc_fused_down_q6_g32_s8 hc_pad_original_down_q6_g32_s8
#define flash_hc_fused_up_mix_q6_g32_s8 hc_pad_original_up_mix_q6_g32_s8
#define flash_hc_fused_down_q6_g64_s4 hc_pad_original_down_q6_g64_s4
#define flash_hc_fused_up_mix_q6_g64_s4 hc_pad_original_up_mix_q6_g64_s4
#define flash_hc_fused_down_q6_g64_s8 hc_pad_original_down_q6_g64_s8
#define flash_hc_fused_up_mix_q6_g64_s8 hc_pad_original_up_mix_q6_g64_s8
#define flash_hc_fused_down_q6_g128_s4 hc_pad_original_down_q6_g128_s4
#define flash_hc_fused_up_mix_q6_g128_s4 hc_pad_original_up_mix_q6_g128_s4
#define flash_hc_fused_down_q6_g128_s8 hc_pad_original_down_q6_g128_s8
#define flash_hc_fused_up_mix_q6_g128_s8 hc_pad_original_up_mix_q6_g128_s8
#define flash_hc_fused_down_q8_g32_s4 hc_pad_original_down_q8_g32_s4
#define flash_hc_fused_up_mix_q8_g32_s4 hc_pad_original_up_mix_q8_g32_s4
#define flash_hc_fused_down_q8_g32_s8 hc_pad_original_down_q8_g32_s8
#define flash_hc_fused_up_mix_q8_g32_s8 hc_pad_original_up_mix_q8_g32_s8
#define flash_hc_fused_down_q8_g64_s4 hc_pad_original_down_q8_g64_s4
#define flash_hc_fused_up_mix_q8_g64_s4 hc_pad_original_up_mix_q8_g64_s4
#define flash_hc_fused_down_q8_g64_s8 hc_pad_original_down_q8_g64_s8
#define flash_hc_fused_up_mix_q8_g64_s8 hc_pad_original_up_mix_q8_g64_s8
#define flash_hc_fused_down_q8_g128_s4 hc_pad_original_down_q8_g128_s4
#define flash_hc_fused_up_mix_q8_g128_s4 hc_pad_original_up_mix_q8_g128_s4
#define flash_hc_fused_down_q8_g128_s8 hc_pad_original_down_q8_g128_s8
#define flash_hc_fused_up_mix_q8_g128_s8 hc_pad_original_up_mix_q8_g128_s8
#define flash_hc_fused_inject_norm hc_pad_original_inject_norm
#define hc_fused_sigmoid_fast hc_pad_original_sigmoid_fast
#define hc_fused_sigmoid_unary hc_pad_original_sigmoid_unary
#define hc_fused_check hc_pad_original_check
#define hc_fused_format hc_pad_original_format
#define hc_fused_workload hc_pad_original_workload
#define hc_fused_code hc_pad_original_code
#define hc_fused_dot hc_pad_original_dot
#define hc_fused_down hc_pad_original_down
#define hc_fused_up_mix hc_pad_original_up_mix
#include "metal/kernels/shared/flash_hc_fused.metal"
#undef flash_hc_fused_down_q4_g32_s4
#undef flash_hc_fused_up_mix_q4_g32_s4
#undef flash_hc_fused_down_q4_g32_s8
#undef flash_hc_fused_up_mix_q4_g32_s8
#undef flash_hc_fused_down_q4_g64_s4
#undef flash_hc_fused_up_mix_q4_g64_s4
#undef flash_hc_fused_down_q4_g64_s8
#undef flash_hc_fused_up_mix_q4_g64_s8
#undef flash_hc_fused_down_q4_g128_s4
#undef flash_hc_fused_up_mix_q4_g128_s4
#undef flash_hc_fused_down_q4_g128_s8
#undef flash_hc_fused_up_mix_q4_g128_s8
#undef flash_hc_fused_down_q5_g32_s4
#undef flash_hc_fused_up_mix_q5_g32_s4
#undef flash_hc_fused_down_q5_g32_s8
#undef flash_hc_fused_up_mix_q5_g32_s8
#undef flash_hc_fused_down_q5_g64_s4
#undef flash_hc_fused_up_mix_q5_g64_s4
#undef flash_hc_fused_down_q5_g64_s8
#undef flash_hc_fused_up_mix_q5_g64_s8
#undef flash_hc_fused_down_q5_g128_s4
#undef flash_hc_fused_up_mix_q5_g128_s4
#undef flash_hc_fused_down_q5_g128_s8
#undef flash_hc_fused_up_mix_q5_g128_s8
#undef flash_hc_fused_down_q6_g32_s4
#undef flash_hc_fused_up_mix_q6_g32_s4
#undef flash_hc_fused_down_q6_g32_s8
#undef flash_hc_fused_up_mix_q6_g32_s8
#undef flash_hc_fused_down_q6_g64_s4
#undef flash_hc_fused_up_mix_q6_g64_s4
#undef flash_hc_fused_down_q6_g64_s8
#undef flash_hc_fused_up_mix_q6_g64_s8
#undef flash_hc_fused_down_q6_g128_s4
#undef flash_hc_fused_up_mix_q6_g128_s4
#undef flash_hc_fused_down_q6_g128_s8
#undef flash_hc_fused_up_mix_q6_g128_s8
#undef flash_hc_fused_down_q8_g32_s4
#undef flash_hc_fused_up_mix_q8_g32_s4
#undef flash_hc_fused_down_q8_g32_s8
#undef flash_hc_fused_up_mix_q8_g32_s8
#undef flash_hc_fused_down_q8_g64_s4
#undef flash_hc_fused_up_mix_q8_g64_s4
#undef flash_hc_fused_down_q8_g64_s8
#undef flash_hc_fused_up_mix_q8_g64_s8
#undef flash_hc_fused_down_q8_g128_s4
#undef flash_hc_fused_up_mix_q8_g128_s4
#undef flash_hc_fused_down_q8_g128_s8
#undef flash_hc_fused_up_mix_q8_g128_s8
#undef flash_hc_fused_inject_norm
#undef hc_fused_sigmoid_fast
#undef hc_fused_sigmoid_unary
#undef hc_fused_check
#undef hc_fused_format
#undef hc_fused_workload
#undef hc_fused_code
#undef hc_fused_dot
#undef hc_fused_down
#undef hc_fused_up_mix

// Untimed common control/candidate probe. This exact private original body has
// only coherent F32/BF16 raw stores added; restoration journal is artifact-side.
template <ushort Bits, ushort Group, ushort SG>
inline void hc_pad_probe_down(
    const device bfloat *normalized,
    const device uchar *dw, const device uchar *ds, const device uchar *db,
    const device uchar *iw, const device uchar *is, const device uchar *ib,
    device bfloat *activated, device bfloat *gates,
    device atomic_uint *diagnostics, device float *raw_f32,
    device bfloat *raw_bf16, FlashHCFusedParams p,
    uint2 grid, uint simd, uint lane) {
  if (!hc_pad_original_workload(p) || p.simdgroups != SG ||
      !hc_pad_original_format(p.down) || p.down.bits != Bits || p.down.group_size != Group ||
      p.down.input_size != 10240 || p.down.output_size != 320 ||
      (p.has_injection && (!hc_pad_original_format(p.injection) ||
                           p.injection.input_size != 10240 || p.injection.output_size != 4))) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint n = grid.x * SG + simd;
  if (grid.y >= p.rows || n >= p.lowrank + p.has_injection * p.streams) return;
  const device bfloat *x = normalized + ulong(grid.y) * p.width * p.streams;
  float partial;
  if (n < p.lowrank) {
    partial = hc_pad_original_dot<Bits, Group>(x, dw, ds, db, p.down, n, lane, p.arithmetic_mode);
  } else {
    const uint index = n - p.lowrank;
    if (p.injection.bits == Bits && p.injection.group_size == Group)
      partial = hc_pad_original_dot<Bits, Group>(x, iw, is, ib, p.injection, index, lane, p.arithmetic_mode);
    else
      partial = hc_pad_original_dot<0, 0>(x, iw, is, ib, p.injection, index, lane, p.arithmetic_mode);
  }
  const float sum = simd_sum(partial);
  if (!lane) {
    // This raw projection boundary remains even though it is register-only.
    const bfloat raw = bfloat(sum);
    if (p.write_raw_up) {
      const ulong raw_index = ulong(grid.y) *
          (p.lowrank + p.has_injection * p.streams) + n;
      raw_f32[raw_index] = sum;
      raw_bf16[raw_index] = raw;
    }
    hc_pad_original_check(sum, diagnostics); hc_pad_original_check(float(raw), diagnostics);
    const bfloat divided = bfloat(float(raw) / float(p.streams));
    if (n < p.lowrank) {
      const bfloat sigmoid = hc_pad_original_sigmoid_fast(divided);
      const bfloat result = bfloat(float(divided) * float(sigmoid));
      activated[ulong(grid.y) * p.lowrank + n] = result;
      hc_pad_original_check(float(result), diagnostics);
    } else {
      const bfloat sigmoid = hc_pad_original_sigmoid_unary(divided);
      const bfloat result = bfloat(2.0f * float(sigmoid));
      gates[ulong(grid.y) * p.streams + n - p.lowrank] = result;
      hc_pad_original_check(float(result), diagnostics);
    }
  }
}

// Duplicate the complete original metadata guard before either the unchanged
// down helper or any padding stores. Invalid metadata must leave all payloads
// untouched even when row0/n<320 would otherwise qualify for tail zeroing.
template <ushort Bits, ushort Group>
inline bool hc_pad_metadata_valid(
    FlashHCDownPadParams cfg, uint3 actual_threads, uint3 total_groups,
    device atomic_uint *diagnostics, uint lane) {
  const FlashHCFusedParams p = cfg.literal;
  if (cfg.padded_rows != 8 || cfg.reserved0 || cfg.reserved1 || cfg.reserved2 ||
      p.rows != 4 || p.arithmetic_mode != 0 ||
      any(actual_threads != uint3(128, 1, 1))) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return false;
  }
  if (!hc_pad_original_workload(p) || p.simdgroups != 4 ||
      !hc_pad_original_format(p.down) || p.down.bits != Bits || p.down.group_size != Group ||
      p.down.input_size != 10240 || p.down.output_size != 320 ||
      (p.has_injection && (!hc_pad_original_format(p.injection) ||
                           p.injection.input_size != 10240 || p.injection.output_size != 4))) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return false;
  }
  const uint groups = (p.lowrank + p.has_injection * p.streams + 3) / 4;
  if (any(total_groups != uint3(groups, 4, 1))) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return false;
  }
  return true;
}

inline void hc_pad_zero_inactive(
    device bfloat *activated, uint3 grid, uint simd, uint lane) {
  const uint n = grid.x * 4 + simd;
  if (!lane && !grid.y && n < 320)
    for (uint row = 4; row < 8; ++row)
      activated[ulong(row) * 320 + n] = bfloat(0.0f);
}

template <ushort Bits, ushort Group>
inline void hc_pad_timed_entry(
    const device bfloat *normalized,
    const device uchar *dw, const device uchar *ds, const device uchar *db,
    const device uchar *iw, const device uchar *is, const device uchar *ib,
    device bfloat *activated, device bfloat *gates,
    device atomic_uint *diagnostics, FlashHCDownPadParams cfg,
    uint3 grid, uint3 actual_threads, uint3 total_groups, uint simd, uint lane) {
  if (!hc_pad_metadata_valid<Bits, Group>(cfg, actual_threads, total_groups, diagnostics, lane)) return;
  const FlashHCFusedParams p = cfg.literal;
  const uint groups = (p.lowrank + p.has_injection * p.streams + 3) / 4;
  if (grid.x >= groups || grid.y >= p.rows || grid.z) return;
  hc_pad_original_down<Bits, Group, 4>(normalized, dw, ds, db, iw, is, ib,
      activated, gates, diagnostics, p, grid.xy, simd, lane);
  hc_pad_zero_inactive(activated, grid, simd, lane);
}

template <ushort Bits, ushort Group, bool Pad>
inline void hc_pad_probe_entry(
    const device bfloat *normalized,
    const device uchar *dw, const device uchar *ds, const device uchar *db,
    const device uchar *iw, const device uchar *is, const device uchar *ib,
    device bfloat *activated, device bfloat *gates,
    device atomic_uint *diagnostics, device float *raw_f32,
    device bfloat *raw_bf16, FlashHCDownPadParams cfg,
    uint3 grid, uint3 actual_threads, uint3 total_groups, uint simd, uint lane) {
  if (!hc_pad_metadata_valid<Bits, Group>(cfg, actual_threads, total_groups, diagnostics, lane)) return;
  const FlashHCFusedParams p = cfg.literal;
  const uint groups = (p.lowrank + p.has_injection * p.streams + 3) / 4;
  if (grid.x >= groups || grid.y >= p.rows || grid.z) return;
  hc_pad_probe_down<Bits, Group, 4>(normalized, dw, ds, db, iw, is, ib,
      activated, gates, diagnostics, raw_f32, raw_bf16, p, grid.xy, simd, lane);
  if (Pad) hc_pad_zero_inactive(activated, grid, simd, lane);
}

#define HC_PAD_TIMED(BITS, GROUP) \
kernel void flash_hc_down_pad_r4_q##BITS##_g##GROUP##_s4( \
    const device bfloat *normalized [[buffer(0)]], \
    const device uchar *dw [[buffer(1)]], const device uchar *ds [[buffer(2)]], \
    const device uchar *db [[buffer(3)]], const device uchar *iw [[buffer(4)]], \
    const device uchar *is [[buffer(5)]], const device uchar *ib [[buffer(6)]], \
    device bfloat *activated [[buffer(7)]], device bfloat *gates [[buffer(8)]], \
    device atomic_uint *diagnostics [[buffer(9)]], \
    constant FlashHCDownPadParams &cfg [[buffer(10)]], \
    uint3 grid [[threadgroup_position_in_grid]], \
    uint3 actual_threads [[threads_per_threadgroup]], \
    uint3 total_groups [[threadgroups_per_grid]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  hc_pad_timed_entry<BITS, GROUP>(normalized, dw, ds, db, iw, is, ib, activated, \
      gates, diagnostics, cfg, grid, actual_threads, total_groups, simd, lane); \
}
#define HC_PAD_PROBE(BITS, GROUP, PREFIX, PAD) \
kernel void PREFIX##_r4_q##BITS##_g##GROUP##_s4( \
    const device bfloat *normalized [[buffer(0)]], \
    const device uchar *dw [[buffer(1)]], const device uchar *ds [[buffer(2)]], \
    const device uchar *db [[buffer(3)]], const device uchar *iw [[buffer(4)]], \
    const device uchar *is [[buffer(5)]], const device uchar *ib [[buffer(6)]], \
    device bfloat *activated [[buffer(7)]], device bfloat *gates [[buffer(8)]], \
    device atomic_uint *diagnostics [[buffer(9)]], \
    device float *raw_f32 [[buffer(10)]], device bfloat *raw_bf16 [[buffer(11)]], \
    constant FlashHCDownPadParams &cfg [[buffer(12)]], \
    uint3 grid [[threadgroup_position_in_grid]], \
    uint3 actual_threads [[threads_per_threadgroup]], \
    uint3 total_groups [[threadgroups_per_grid]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  hc_pad_probe_entry<BITS, GROUP, PAD>(normalized, dw, ds, db, iw, is, ib, activated, \
      gates, diagnostics, raw_f32, raw_bf16, cfg, grid, actual_threads, total_groups, simd, lane); \
}
#define HC_PAD_FORMAT(BITS, GROUP) \
HC_PAD_TIMED(BITS, GROUP) \
HC_PAD_PROBE(BITS, GROUP, flash_hc_down_pad_probe, true) \
HC_PAD_PROBE(BITS, GROUP, flash_hc_down_control_probe, false)
#define HC_PAD_BITS(BITS) \
HC_PAD_FORMAT(BITS, 32) \
HC_PAD_FORMAT(BITS, 64) \
HC_PAD_FORMAT(BITS, 128)
HC_PAD_BITS(4)
HC_PAD_BITS(5)
HC_PAD_BITS(6)
HC_PAD_BITS(8)
#undef HC_PAD_BITS
#undef HC_PAD_FORMAT
#undef HC_PAD_PROBE
#undef HC_PAD_TIMED

// Diagnostic-only HC-up probe. Original timed HC-up remains linked unchanged.
#define hf_error hc_pad_up_original_error
#define hf_sigmoid hc_pad_up_original_sigmoid
#define hf_up hc_pad_up_original_up
#define flash_hc_up_f32_mpp_m8_n32_s4 hc_pad_up_original_m8_n32_s4
#define flash_hc_up_f32_mpp_m8_n32_s4_debug hc_pad_up_original_m8_n32_s4_debug
#include "metal/kernels/shared/flash_hc_up_f32_mpp.metal"
#undef hf_error
#undef hf_sigmoid
#undef hf_up
#undef flash_hc_up_f32_mpp_m8_n32_s4
#undef flash_hc_up_f32_mpp_m8_n32_s4_debug

// Exact hf_up body: only raw F32 dot store/name/pointer plumbing is added.
template<ushort M,ushort N,ushort S,bool Debug>
inline void hc_pad_up_probe(device bfloat *padded,device float *weights,device const bfloat *normalized,
    device bfloat *mixed,device bfloat *rawDebug,device float *rawF32,device atomic_uint *diagnostics,
    constant FlashFloatDenseSmallRowsParams &p,uint3 group,uint3 threads,uint tid){
  if(!p.rows||p.rows>16||p.input_size!=320||p.output_size!=2560||p.output_begin||p.output_count!=2560||
      p.tile_rows!=M||p.tile_outputs!=N||p.padded_rows!=(p.rows+M-1)/M*M||p.padded_rows>16||
      group.x>=2560/N||group.y>=p.padded_rows/M||group.z||threads.x!=S*32||threads.y!=1||threads.z!=1){
    if(!tid)hc_pad_up_original_error(diagnostics,2);return;}
  const uint rowBase=group.y*M,colBase=group.x*N;
  auto a=tensor(padded+ulong(rowBase)*320,dextents<int,2>{320,M},array<int,2>{1,320});
  auto firstB=tensor(weights+ulong(colBase)*320,dextents<int,2>{320,N},array<int,2>{1,320});
  constexpr auto descriptor=matmul2d_descriptor(M,N,static_cast<int>(dynamic_extent),false,true,false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor,execution_simdgroups<S>> operation;
  auto total=operation.template get_destination_cooperative_tensor<decltype(a),decltype(firstB),float>();
#pragma unroll
  for(ushort i=0;i<total.get_capacity();++i)if(total.is_valid_element(i))total[i]=0;
  for(uint stream=0;stream<4;++stream){
    auto b=tensor(weights+ulong(stream*2560+colBase)*320,dextents<int,2>{320,N},array<int,2>{1,320});
    auto dot=operation.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    operation.run(a,b,dot);
#pragma unroll
    for(ushort i=0;i<total.get_capacity();++i){if(!total.is_valid_element(i))continue;
      const auto index=total.get_multidimensional_index(i);const uint row=rowBase+index[1],col=colBase+index[0];
      if(row>=p.rows)continue;rawF32[(ulong(row)*4+stream)*2560+col]=dot[i];const bfloat raw=bfloat(dot[i]);
      if(!isfinite(dot[i])||!isfinite(float(raw)))hc_pad_up_original_error(diagnostics,4);
      if(Debug)rawDebug[(ulong(row)*4+stream)*2560+col]=raw;
      const bfloat product=bfloat(float(hc_pad_up_original_sigmoid(raw))*float(normalized[(ulong(row)*4+stream)*2560+col]));
      total[i]=float(bfloat(float(product)+total[i]));
    }
  }
#pragma unroll
  for(ushort i=0;i<total.get_capacity();++i){if(!total.is_valid_element(i))continue;
    const auto index=total.get_multidimensional_index(i);const uint row=rowBase+index[1];if(row>=p.rows)continue;
    const bfloat value=bfloat(total[i]/4.0f);mixed[ulong(row)*2560+colBase+index[0]]=value;
    if(!isfinite(float(value)))hc_pad_up_original_error(diagnostics,4);
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void flash_hc_pad_up_probe_f32_m8_n32_s4(
    device bfloat *padded [[buffer(0)]], device float *weights [[buffer(1)]],
    device const bfloat *normalized [[buffer(2)]], device bfloat *mixed [[buffer(3)]],
    device bfloat *raw_bf16 [[buffer(4)]], device atomic_uint *diagnostics [[buffer(5)]],
    device float *raw_f32 [[buffer(6)]],
    constant FlashFloatDenseSmallRowsParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 actual_threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (p.rows != 4 || p.padded_rows != 8) {
    if (!tid) hc_pad_up_original_error(diagnostics, 2u);
    return;
  }
  hc_pad_up_probe<8, 32, 4, true>(padded, weights, normalized, mixed, raw_bf16,
      raw_f32, diagnostics, p, group, actual_threads, tid);
}
