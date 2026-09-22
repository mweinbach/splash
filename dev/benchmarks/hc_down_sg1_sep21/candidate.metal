// Private SG1 HC-down dispatch experiment. Compile against a frozen, byte-identical
// copy of runtime/metal/kernels/shared/flash_hc_fused.metal. Its helpers and
// complete down body are included verbatim, with every exported entry renamed
// so this AIR can coexist with the original baseline AIR.
#include "abi.hpp"

#define flash_hc_fused_down_q4_g32_s4 hc_sg1_original_down_q4_g32_s4
#define flash_hc_fused_up_mix_q4_g32_s4 hc_sg1_original_up_mix_q4_g32_s4
#define flash_hc_fused_down_q4_g32_s8 hc_sg1_original_down_q4_g32_s8
#define flash_hc_fused_up_mix_q4_g32_s8 hc_sg1_original_up_mix_q4_g32_s8
#define flash_hc_fused_down_q4_g64_s4 hc_sg1_original_down_q4_g64_s4
#define flash_hc_fused_up_mix_q4_g64_s4 hc_sg1_original_up_mix_q4_g64_s4
#define flash_hc_fused_down_q4_g64_s8 hc_sg1_original_down_q4_g64_s8
#define flash_hc_fused_up_mix_q4_g64_s8 hc_sg1_original_up_mix_q4_g64_s8
#define flash_hc_fused_down_q4_g128_s4 hc_sg1_original_down_q4_g128_s4
#define flash_hc_fused_up_mix_q4_g128_s4 hc_sg1_original_up_mix_q4_g128_s4
#define flash_hc_fused_down_q4_g128_s8 hc_sg1_original_down_q4_g128_s8
#define flash_hc_fused_up_mix_q4_g128_s8 hc_sg1_original_up_mix_q4_g128_s8
#define flash_hc_fused_down_q5_g32_s4 hc_sg1_original_down_q5_g32_s4
#define flash_hc_fused_up_mix_q5_g32_s4 hc_sg1_original_up_mix_q5_g32_s4
#define flash_hc_fused_down_q5_g32_s8 hc_sg1_original_down_q5_g32_s8
#define flash_hc_fused_up_mix_q5_g32_s8 hc_sg1_original_up_mix_q5_g32_s8
#define flash_hc_fused_down_q5_g64_s4 hc_sg1_original_down_q5_g64_s4
#define flash_hc_fused_up_mix_q5_g64_s4 hc_sg1_original_up_mix_q5_g64_s4
#define flash_hc_fused_down_q5_g64_s8 hc_sg1_original_down_q5_g64_s8
#define flash_hc_fused_up_mix_q5_g64_s8 hc_sg1_original_up_mix_q5_g64_s8
#define flash_hc_fused_down_q5_g128_s4 hc_sg1_original_down_q5_g128_s4
#define flash_hc_fused_up_mix_q5_g128_s4 hc_sg1_original_up_mix_q5_g128_s4
#define flash_hc_fused_down_q5_g128_s8 hc_sg1_original_down_q5_g128_s8
#define flash_hc_fused_up_mix_q5_g128_s8 hc_sg1_original_up_mix_q5_g128_s8
#define flash_hc_fused_down_q6_g32_s4 hc_sg1_original_down_q6_g32_s4
#define flash_hc_fused_up_mix_q6_g32_s4 hc_sg1_original_up_mix_q6_g32_s4
#define flash_hc_fused_down_q6_g32_s8 hc_sg1_original_down_q6_g32_s8
#define flash_hc_fused_up_mix_q6_g32_s8 hc_sg1_original_up_mix_q6_g32_s8
#define flash_hc_fused_down_q6_g64_s4 hc_sg1_original_down_q6_g64_s4
#define flash_hc_fused_up_mix_q6_g64_s4 hc_sg1_original_up_mix_q6_g64_s4
#define flash_hc_fused_down_q6_g64_s8 hc_sg1_original_down_q6_g64_s8
#define flash_hc_fused_up_mix_q6_g64_s8 hc_sg1_original_up_mix_q6_g64_s8
#define flash_hc_fused_down_q6_g128_s4 hc_sg1_original_down_q6_g128_s4
#define flash_hc_fused_up_mix_q6_g128_s4 hc_sg1_original_up_mix_q6_g128_s4
#define flash_hc_fused_down_q6_g128_s8 hc_sg1_original_down_q6_g128_s8
#define flash_hc_fused_up_mix_q6_g128_s8 hc_sg1_original_up_mix_q6_g128_s8
#define flash_hc_fused_down_q8_g32_s4 hc_sg1_original_down_q8_g32_s4
#define flash_hc_fused_up_mix_q8_g32_s4 hc_sg1_original_up_mix_q8_g32_s4
#define flash_hc_fused_down_q8_g32_s8 hc_sg1_original_down_q8_g32_s8
#define flash_hc_fused_up_mix_q8_g32_s8 hc_sg1_original_up_mix_q8_g32_s8
#define flash_hc_fused_down_q8_g64_s4 hc_sg1_original_down_q8_g64_s4
#define flash_hc_fused_up_mix_q8_g64_s4 hc_sg1_original_up_mix_q8_g64_s4
#define flash_hc_fused_down_q8_g64_s8 hc_sg1_original_down_q8_g64_s8
#define flash_hc_fused_up_mix_q8_g64_s8 hc_sg1_original_up_mix_q8_g64_s8
#define flash_hc_fused_down_q8_g128_s4 hc_sg1_original_down_q8_g128_s4
#define flash_hc_fused_up_mix_q8_g128_s4 hc_sg1_original_up_mix_q8_g128_s4
#define flash_hc_fused_down_q8_g128_s8 hc_sg1_original_down_q8_g128_s8
#define flash_hc_fused_up_mix_q8_g128_s8 hc_sg1_original_up_mix_q8_g128_s8
#define flash_hc_fused_inject_norm hc_sg1_original_inject_norm
#define hc_fused_sigmoid_fast hc_sg1_original_sigmoid_fast
#define hc_fused_sigmoid_unary hc_sg1_original_sigmoid_unary
#define hc_fused_check hc_sg1_original_check
#define hc_fused_format hc_sg1_original_format
#define hc_fused_workload hc_sg1_original_workload
#define hc_fused_code hc_sg1_original_code
#define hc_fused_dot hc_sg1_original_dot
#define hc_fused_down hc_sg1_original_down
#define hc_fused_up_mix hc_sg1_original_up_mix
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

// The original down implementation requires p.simdgroups == 4 or 8. Only its
// output-index arithmetic depends on the synthetic grid/simd arguments.
// SG1 maps n directly to the original SG4 index n/4*4+n%4, and changes the local
// configuration to 4 after validating the actual SG1 launch. No arithmetic
// helper, coefficient reconstruction, reduction, or scalar stage is copied.
template <ushort Bits, ushort Group>
inline void hc_sg1_timed_down(
    const device bfloat *normalized,
    const device uchar *dw, const device uchar *ds, const device uchar *db,
    const device uchar *iw, const device uchar *is, const device uchar *ib,
    device bfloat *activated, device bfloat *gates,
    device atomic_uint *diagnostics, FlashHCFusedParams p,
    uint2 grid, uint3 actual_threads, uint lane) {
  if (p.simdgroups != 1 || any(actual_threads != uint3(32, 1, 1))) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  FlashHCFusedParams original = p;
  original.simdgroups = 4;
  hc_sg1_original_down<Bits, Group, 4>(
      normalized, dw, ds, db, iw, is, ib, activated, gates, diagnostics,
      original, uint2(grid.x / 4, grid.y), grid.x % 4, lane);
}

// Untimed diagnostic. p.write_raw_up is the coherent tap control: 0 leaves
// both tap buffers untouched (they may be dummies); 1 writes both boundaries.
// We deliberately recompute the original dot through the full original down
// body after writing the taps, preserving the entire original scalar stage
// without deriving activation/gate outputs from a separately implemented post.
template <ushort Bits, ushort Group, ushort SG>
inline void hc_sg1_probe_down(
    const device bfloat *normalized,
    const device uchar *dw, const device uchar *ds, const device uchar *db,
    const device uchar *iw, const device uchar *is, const device uchar *ib,
    device bfloat *activated, device bfloat *gates,
    device atomic_uint *diagnostics, device float *raw_f32,
    device bfloat *raw_bf16, FlashHCFusedParams p,
    uint2 grid, uint3 actual_threads, uint simd, uint lane) {
  if ((SG != 1 && SG != 4) || p.simdgroups != SG ||
      any(actual_threads != uint3(SG * 32, 1, 1))) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  FlashHCFusedParams original = p;
  original.simdgroups = 4;
  // These are the exact predicates from the original down helper, after the
  // explicit launch check above. Keep the dynamic injection format fallback.
  if (!hc_sg1_original_workload(original) ||
      !hc_sg1_original_format(p.down) ||
      p.down.bits != Bits || p.down.group_size != Group ||
      p.down.input_size != 10240 || p.down.output_size != 320 ||
      (p.has_injection && (!hc_sg1_original_format(p.injection) ||
                           p.injection.input_size != 10240 ||
                           p.injection.output_size != 4))) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint n = grid.x * SG + simd;
  const uint outputs = p.lowrank + p.has_injection * p.streams;
  if (grid.y >= p.rows || n >= outputs) return;
  if (p.write_raw_up) {
    const device bfloat *x = normalized + ulong(grid.y) * p.width * p.streams;
    float partial;
    if (n < p.lowrank) {
      partial = hc_sg1_original_dot<Bits, Group>(
          x, dw, ds, db, p.down, n, lane, p.arithmetic_mode);
    } else {
      const uint index = n - p.lowrank;
      if (p.injection.bits == Bits && p.injection.group_size == Group)
        partial = hc_sg1_original_dot<Bits, Group>(
            x, iw, is, ib, p.injection, index, lane, p.arithmetic_mode);
      else
        partial = hc_sg1_original_dot<0, 0>(
            x, iw, is, ib, p.injection, index, lane, p.arithmetic_mode);
    }
    const float sum = simd_sum(partial);
    if (!lane) {
      const ulong raw_index = ulong(grid.y) * outputs + n;
      raw_f32[raw_index] = sum;
      raw_bf16[raw_index] = bfloat(sum);
    }
  }
  hc_sg1_original_down<Bits, Group, 4>(
      normalized, dw, ds, db, iw, is, ib, activated, gates, diagnostics,
      original, uint2(n / 4, grid.y), n % 4, lane);
}

#define HC_SG1_TIMED(BITS, GROUP) \
kernel void flash_hc_down_sg1_q##BITS##_g##GROUP( \
    const device bfloat *normalized [[buffer(0)]], \
    const device uchar *dw [[buffer(1)]], const device uchar *ds [[buffer(2)]], \
    const device uchar *db [[buffer(3)]], const device uchar *iw [[buffer(4)]], \
    const device uchar *is [[buffer(5)]], const device uchar *ib [[buffer(6)]], \
    device bfloat *activated [[buffer(7)]], device bfloat *gates [[buffer(8)]], \
    device atomic_uint *diagnostics [[buffer(9)]], \
    constant FlashHCFusedParams &p [[buffer(10)]], \
    uint3 grid [[threadgroup_position_in_grid]], \
    uint3 actual_threads [[threads_per_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  if (grid.z) return; \
  hc_sg1_timed_down<BITS, GROUP>(normalized, dw, ds, db, iw, is, ib, activated, \
                               gates, diagnostics, p, grid.xy, actual_threads, lane); \
}

#define HC_SG1_PROBE(BITS, GROUP, SG) \
kernel void flash_hc_down_probe_q##BITS##_g##GROUP##_s##SG( \
    const device bfloat *normalized [[buffer(0)]], \
    const device uchar *dw [[buffer(1)]], const device uchar *ds [[buffer(2)]], \
    const device uchar *db [[buffer(3)]], const device uchar *iw [[buffer(4)]], \
    const device uchar *is [[buffer(5)]], const device uchar *ib [[buffer(6)]], \
    device bfloat *activated [[buffer(7)]], device bfloat *gates [[buffer(8)]], \
    device atomic_uint *diagnostics [[buffer(9)]], \
    device float *raw_f32 [[buffer(10)]], device bfloat *raw_bf16 [[buffer(11)]], \
    constant FlashHCFusedParams &p [[buffer(12)]], \
    uint3 grid [[threadgroup_position_in_grid]], \
    uint3 actual_threads [[threads_per_threadgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  if (grid.z) return; \
  hc_sg1_probe_down<BITS, GROUP, SG>(normalized, dw, ds, db, iw, is, ib, activated, \
      gates, diagnostics, raw_f32, raw_bf16, p, grid.xy, actual_threads, simd, lane); \
}

#define HC_SG1_GROUP(BITS, GROUP) \
HC_SG1_TIMED(BITS, GROUP) \
HC_SG1_PROBE(BITS, GROUP, 1) \
HC_SG1_PROBE(BITS, GROUP, 4)
#define HC_SG1_BITS(BITS) \
HC_SG1_GROUP(BITS, 32) \
HC_SG1_GROUP(BITS, 64) \
HC_SG1_GROUP(BITS, 128)
HC_SG1_BITS(4)
HC_SG1_BITS(5)
HC_SG1_BITS(6)
HC_SG1_BITS(8)
#undef HC_SG1_BITS
#undef HC_SG1_GROUP
#undef HC_SG1_PROBE
#undef HC_SG1_TIMED

// Independent untimed post witness. The operation sequence below is copied
// from the original scalar down stage after its raw BF16 projection boundary.
// The unavailable pre-rounding F32 sum is checked by the down/probe itself.
kernel void flash_hc_down_sg1_epilog_from_raw(
    const device bfloat *raw_bf16 [[buffer(0)]],
    device bfloat *activated [[buffer(1)]], device bfloat *gates [[buffer(2)]],
    device atomic_uint *diagnostics [[buffer(3)]],
    constant FlashHCFusedParams &p [[buffer(4)]],
    uint3 position [[thread_position_in_grid]],
    uint3 actual_threads [[threads_per_threadgroup]]) {
  if (position.y || position.z) return;
  const uint gid = position.x;
  FlashHCFusedParams original = p;
  original.simdgroups = 4;
  if (any(actual_threads != uint3(256, 1, 1)) ||
      (p.simdgroups != 1 && p.simdgroups != 4) ||
      !hc_sg1_original_workload(original)) {
    if (!gid) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint outputs = p.lowrank + p.has_injection * p.streams;
  if (gid >= p.rows * outputs) return;
  const uint row = gid / outputs;
  const uint n = gid % outputs;
  const bfloat raw = raw_bf16[gid];
  hc_sg1_original_check(float(raw), diagnostics);
  const bfloat divided = bfloat(float(raw) / float(p.streams));
  if (n < p.lowrank) {
    const bfloat sigmoid = hc_sg1_original_sigmoid_fast(divided);
    const bfloat result = bfloat(float(divided) * float(sigmoid));
    activated[ulong(row) * p.lowrank + n] = result;
    hc_sg1_original_check(float(result), diagnostics);
  } else {
    const bfloat sigmoid = hc_sg1_original_sigmoid_unary(divided);
    const bfloat result = bfloat(2.0f * float(sigmoid));
    gates[ulong(row) * p.streams + n - p.lowrank] = result;
    hc_sg1_original_check(float(result), diagnostics);
  }
}
