#include "FlashHCDownF32CandidateABI.h"
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline bfloat hc_fused_sigmoid_fast(bfloat x) {
  const bfloat e = bfloat(metal::exp(metal::abs(float(x))));
  const bfloat d = bfloat(1.0f) + e;
  const bfloat t = bfloat(1.0f) / d;
  return x < bfloat(0.0f) ? t : bfloat(1.0f) - t;
}
inline bfloat hc_fused_sigmoid_unary(bfloat x) {
  const bfloat e = bfloat(metal::precise::exp(metal::abs(float(x))));
  const bfloat d = bfloat(1.0f) + e;
  const bfloat t = bfloat(1.0f) / d;
  return x < bfloat(0.0f) ? t : bfloat(1.0f) - t;
}
inline void hc_fused_check(float x, device atomic_uint *diagnostics) {
  if ((as_type<uint>(x) & 0x7f800000u) == 0x7f800000u)
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
}
inline bool hc_fused_format(FlashHCFusedMatrix m) {
  return m.input_size && m.output_size &&
      (m.bits == 4 || m.bits == 5 || m.bits == 6 || m.bits == 8) &&
      (m.group_size == 32 || m.group_size == 64 || m.group_size == 128) &&
      m.input_size % m.group_size == 0 &&
      m.weight_row_stride_bytes >= (ulong(m.input_size) * m.bits + 7) / 8 &&
      m.parameter_row_stride_bytes >= ulong(m.input_size / m.group_size) * 2 &&
      m.parameter_row_stride_bytes % 2 == 0;
}
inline bool hc_fused_workload(FlashHCFusedParams p) {
  return p.rows && p.rows <= 32 && p.width == 2560 && p.streams == 4 &&
      p.lowrank == 320 && p.arithmetic_mode <= 1 && p.has_injection <= 1 &&
      p.write_raw_up <= 1 && (p.simdgroups == 4 || p.simdgroups == 8);
}

template <ushort Bits>
inline uint hc_fused_code(const device uchar *w, uint k, uint dynamic_bits) {
  const uint bits = Bits ? Bits : dynamic_bits;
  const ulong bit = ulong(k) * bits;
  const uint shift = uint(bit & 7);
  const ulong byte = bit >> 3;
  uint code = uint(w[byte]);
  if (shift + bits > 8) code |= uint(w[byte + 1]) << 8;
  return (code >> shift) & ((1u << bits) - 1);
}

// The exact mode keeps k=lane+32*j and reconstructs every F32 coefficient
// before multiplying. Hoisting parameters never distributes the affine bias.
template <ushort Bits, ushort Group>
inline float hc_fused_dot(
    const device bfloat *x, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    FlashHCFusedMatrix m, uint n, uint lane, uint mode) {
  const uint group_size = Group ? Group : m.group_size;
  const device uchar *w = weights + ulong(n) * m.weight_row_stride_bytes;
  const device bfloat *s = reinterpret_cast<const device bfloat *>(
      scales + ulong(n) * m.parameter_row_stride_bytes);
  const device bfloat *b = reinterpret_cast<const device bfloat *>(
      biases + ulong(n) * m.parameter_row_stride_bytes);
  float sum = 0.0f;
  for (uint g = 0; g < m.input_size / group_size; ++g) {
    const float sf = float(s[g]);
    const float bias = float(b[g]);
    float qdot = 0.0f, input_sum = 0.0f;
    for (uint block = 0; block < group_size / 32; ++block) {
      const uint k = g * group_size + block * 32 + lane;
      const float value = float(x[k]);
      const float code = float(hc_fused_code<Bits>(w, k, m.bits));
      if (mode == 0) {
        const float coefficient = code * sf + bias;
        sum += value * coefficient;
      } else {
        qdot += value * code;
        input_sum += value;
      }
    }
    if (mode == 1) {
      const float scaled = sf * qdot;
      const float corrected = bias * input_sum;
      sum += scaled + corrected;
    }
  }
  return sum;
}

inline bfloat hc_down_candidate_activate(float sum) {
  const bfloat raw = bfloat(sum);
  const bfloat divided = bfloat(float(raw) / 4.0f);
  const bfloat sigmoid = hc_fused_sigmoid_fast(divided);
  return bfloat(float(divided) * float(sigmoid));
}
inline bfloat hc_down_candidate_inject(float sum) {
  const bfloat raw = bfloat(sum);
  const bfloat divided = bfloat(float(raw) / 4.0f);
  return bfloat(2.0f * float(hc_fused_sigmoid_unary(divided)));
}

// Literal, untimed production witness: every output has one SIMD and row.
kernel void flash_hc_down_f32_literal_witness(
    device const bfloat *x [[buffer(0)]], device const uchar *dw [[buffer(1)]],
    device const uchar *ds [[buffer(2)]], device const uchar *db [[buffer(3)]],
    device const uchar *iw [[buffer(4)]], device const uchar *is [[buffer(5)]],
    device const uchar *ib [[buffer(6)]], device bfloat *activated [[buffer(7)]],
    device bfloat *gates [[buffer(8)]], device atomic_uint *diagnostic [[buffer(9)]],
    device bfloat *rawBF [[buffer(10)]], device float *rawF [[buffer(11)]],
    constant HCDownF32CandidateParams &params [[buffer(12)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const FlashHCFusedParams p = params.literal;
  if (!hc_fused_workload(p) || p.arithmetic_mode || p.simdgroups != 4 ||
      p.down.input_size != 10240 || p.down.output_size != 320 ||
      !hc_fused_format(p.down) || (p.has_injection && !hc_fused_format(p.injection))) {
    if (!lane) atomic_fetch_or_explicit(diagnostic, 2u, memory_order_relaxed); return;
  }
  const uint n = group.x * 4 + simd;
  if (group.y >= p.rows || n >= 320 + p.has_injection * 4) return;
  const bool inject = n >= 320;
  const float partial = inject
      ? hc_fused_dot<0, 0>(x + ulong(group.y) * 10240, iw, is, ib,
          p.injection, n - 320, lane, 0)
      : hc_fused_dot<0, 0>(x + ulong(group.y) * 10240, dw, ds, db,
          p.down, n, lane, 0);
  const float sum = simd_sum(partial);
  if (!lane) {
    rawBF[ulong(group.y) * 324 + n] = bfloat(sum);
    rawF[ulong(group.y) * 324 + n] = sum;
    hc_fused_check(sum, diagnostic); hc_fused_check(float(bfloat(sum)), diagnostic);
    const bfloat value = inject ? hc_down_candidate_inject(sum) : hc_down_candidate_activate(sum);
    hc_fused_check(float(value), diagnostic);
    if (inject) gates[ulong(group.y) * 4 + n - 320] = value;
    else activated[ulong(group.y) * 320 + n] = value;
  }
}

template <ushort Rows>
inline void hc_down_f32_literal_reuse(device const bfloat *x,
    device const float *cached, device const uchar *iw, device const uchar *is,
    device const uchar *ib, device bfloat *activated, device bfloat *gates,
    device atomic_uint *diagnostic, device bfloat *rawBF, device float *rawF,
    constant HCDownF32CandidateParams &params, uint3 group, uint simd, uint lane) {
  const FlashHCFusedParams p = params.literal;
  if (!hc_fused_workload(p) || p.rows != Rows || p.arithmetic_mode || p.simdgroups != 4 ||
      p.down.input_size != 10240 || p.down.output_size != 320 || params.write_debug > 1 ||
      (p.has_injection && (!hc_fused_format(p.injection) ||
          p.injection.input_size != 10240 || p.injection.output_size != 4)) || group.y || group.z) {
    if (!lane) atomic_fetch_or_explicit(diagnostic, 2u, memory_order_relaxed); return;
  }
  const uint n = group.x * 4 + simd;
  if (n >= 320 + p.has_injection * 4) return;
  float sums[Rows];
#pragma unroll
  for (ushort row = 0; row < Rows; ++row) sums[row] = 0.0f;
  if (n < 320) {
    // Exactly the original k=lane+32*j traversal and per-row accumulation.
    // Original F32 operand bits are never rounded to BF16.
    for (uint j = 0; j < 320; ++j) {
      const uint k = lane + 32 * j;
      const float coefficient = cached[ulong(n) * 10240 + k];
#pragma unroll
      for (ushort row = 0; row < Rows; ++row)
        sums[row] += float(x[ulong(row) * 10240 + k]) * coefficient;
    }
  } else {
    const uint index = n - 320;
    device const uchar *w = iw + ulong(index) * p.injection.weight_row_stride_bytes;
    device const bfloat *s = reinterpret_cast<device const bfloat *>(is + ulong(index) * p.injection.parameter_row_stride_bytes);
    device const bfloat *b = reinterpret_cast<device const bfloat *>(ib + ulong(index) * p.injection.parameter_row_stride_bytes);
    for (uint g = 0; g < 10240 / p.injection.group_size; ++g) {
      const float sf = float(s[g]), bias = float(b[g]);
      for (uint block = 0; block < p.injection.group_size / 32; ++block) {
        const uint k = g * p.injection.group_size + block * 32 + lane;
        const float coefficient = float(hc_fused_code<0>(w, k, p.injection.bits)) * sf + bias;
#pragma unroll
        for (ushort row = 0; row < Rows; ++row)
          sums[row] += float(x[ulong(row) * 10240 + k]) * coefficient;
      }
    }
  }
#pragma unroll
  for (ushort row = 0; row < Rows; ++row) {
    const float sum = simd_sum(sums[row]);
    if (!lane) {
      if (params.write_debug) { rawBF[ulong(row) * 324 + n] = bfloat(sum); rawF[ulong(row) * 324 + n] = sum; }
      hc_fused_check(sum, diagnostic); hc_fused_check(float(bfloat(sum)), diagnostic);
      const bfloat value = n < 320 ? hc_down_candidate_activate(sum) : hc_down_candidate_inject(sum);
      hc_fused_check(float(value), diagnostic);
      if (n < 320) activated[ulong(row) * 320 + n] = value;
      else gates[ulong(row) * 4 + n - 320] = value;
    }
  }
}
#define HC_DOWN_REUSE_ENTRY(Name, Rows)                                     \
kernel void Name(device const bfloat *x [[buffer(0)]],                     \
    device const float *cached [[buffer(1)]], device const uchar *iw [[buffer(2)]], \
    device const uchar *is [[buffer(3)]], device const uchar *ib [[buffer(4)]], \
    device bfloat *activated [[buffer(5)]], device bfloat *gates [[buffer(6)]], \
    device atomic_uint *diagnostic [[buffer(7)]], device bfloat *rawBF [[buffer(8)]], \
    device float *rawF [[buffer(9)]],                                      \
    constant HCDownF32CandidateParams &p [[buffer(10)]],                   \
    uint3 group [[threadgroup_position_in_grid]],                         \
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  hc_down_f32_literal_reuse<Rows>(x, cached, iw, is, ib, activated, gates, diagnostic, rawBF, rawF, p, group, simd, lane); \
}
HC_DOWN_REUSE_ENTRY(flash_hc_down_f32_literal_reuse_r4, 4)
HC_DOWN_REUSE_ENTRY(flash_hc_down_f32_literal_reuse_r8, 8)
HC_DOWN_REUSE_ENTRY(flash_hc_down_f32_literal_reuse_r16, 16)
#undef HC_DOWN_REUSE_ENTRY

kernel void flash_hc_down_f32_literal_injection(
    device const bfloat *x [[buffer(0)]], device const uchar *w [[buffer(1)]],
    device const uchar *s [[buffer(2)]], device const uchar *b [[buffer(3)]],
    device bfloat *gates [[buffer(4)]], device atomic_uint *diagnostic [[buffer(5)]],
    device bfloat *rawBF [[buffer(6)]], device float *rawF [[buffer(7)]],
    constant HCDownF32CandidateParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const FlashHCFusedParams p = params.literal;
  if (!p.has_injection || !hc_fused_workload(p) || !hc_fused_format(p.injection) ||
      p.injection.input_size != 10240 || p.injection.output_size != 4 || group.x || group.y >= p.rows) return;
  const float sum = simd_sum(hc_fused_dot<0, 0>(x + ulong(group.y) * 10240,
      w, s, b, p.injection, simd, lane, 0));
  if (!lane) {
    if (params.write_debug) { rawBF[ulong(group.y) * 324 + 320 + simd] = bfloat(sum); rawF[ulong(group.y) * 324 + 320 + simd] = sum; }
    hc_fused_check(sum, diagnostic); hc_fused_check(float(bfloat(sum)), diagnostic);
    const bfloat value = hc_down_candidate_inject(sum);
    hc_fused_check(float(value), diagnostic); gates[ulong(group.y) * 4 + simd] = value;
  }
}

// Canonical post witness applied to the candidate's own rounded raw dots.
kernel void flash_hc_down_f32_epilog_from_raw(
    device const bfloat *raw [[buffer(0)]], device bfloat *activated [[buffer(1)]],
    device bfloat *gates [[buffer(2)]], device atomic_uint *diagnostic [[buffer(3)]],
    constant HCDownF32CandidateParams &params [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  const uint row = gid / 324, n = gid % 324;
  if (row >= params.literal.rows || n >= 320 + params.literal.has_injection * 4) return;
  const float value = float(raw[gid]);
  const bfloat result = n < 320 ? hc_down_candidate_activate(value) : hc_down_candidate_inject(value);
  hc_fused_check(float(result), diagnostic);
  if (n < 320) activated[ulong(row) * 320 + n] = result;
  else gates[ulong(row) * 4 + n - 320] = result;
}

#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#pragma METAL fp math_mode(safe)
using namespace mpp::tensor_ops;

template <ushort N>
inline void hc_down_f32_mpp(device bfloat *padded, device float *weights,
    device float *partials, device atomic_uint *diagnostic,
    constant HCDownF32CandidateParams &p, uint3 group) {
  if ((p.literal.rows != 4 && p.literal.rows != 8 && p.literal.rows != 16) ||
      p.padded_rows != (p.literal.rows + 7) / 8 * 8 || p.tile_outputs != N ||
      (p.partitions != 1 && p.partitions != 4 && p.partitions != 8) ||
      group.x >= 320 / N || group.y >= p.padded_rows / 8 || group.z >= p.partitions) return;
  const uint start = group.z * (10240 / p.partitions), row = group.y * 8, column = group.x * N;
  const int k = int(10240 / p.partitions);
  auto a = tensor(padded + ulong(row) * 10240 + start,
      dextents<int, 2>{k, 8}, array<int, 2>{1, 10240});
  auto b = tensor(weights + ulong(column) * 10240 + start,
      dextents<int, 2>{k, N}, array<int, 2>{1, 10240});
  constexpr auto descriptor = matmul2d_descriptor(8, N, static_cast<int>(dynamic_extent),
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  operation.run(a, b, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    hc_fused_check(dot[i], diagnostic);
    partials[((ulong(row + index[1]) * p.partitions + group.z) * 320) + column + index[0]] = dot[i];
  }
}
#define HC_DOWN_MPP_ENTRY(Name, N)                                         \
kernel void Name(device bfloat *padded [[buffer(0)]], device float *weights [[buffer(1)]], \
    device float *partial [[buffer(2)]], device atomic_uint *diag [[buffer(3)]], \
    constant HCDownF32CandidateParams &p [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]]) { \
  hc_down_f32_mpp<N>(padded, weights, partial, diag, p, group);             \
}
HC_DOWN_MPP_ENTRY(flash_hc_down_f32_mpp_m8_n32, 32)
HC_DOWN_MPP_ENTRY(flash_hc_down_f32_mpp_m8_n64, 64)
#undef HC_DOWN_MPP_ENTRY

kernel void flash_hc_down_f32_fold_activate(device const float *partials [[buffer(0)]],
    device bfloat *activated [[buffer(1)]], device atomic_uint *diagnostic [[buffer(2)]],
    device bfloat *rawBF [[buffer(3)]], device float *rawF [[buffer(4)]],
    constant HCDownF32CandidateParams &p [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= p.literal.rows * 320) return;
  const uint row = gid / 320, n = gid % 320;
  float sum = 0.0f;
  for (uint part = 0; part < p.partitions; ++part) sum += partials[(ulong(row) * p.partitions + part) * 320 + n];
  if (p.write_debug) { rawBF[ulong(row) * 324 + n] = bfloat(sum); rawF[ulong(row) * 324 + n] = sum; }
  hc_fused_check(sum, diagnostic); hc_fused_check(float(bfloat(sum)), diagnostic);
  const bfloat value = hc_down_candidate_activate(sum);
  hc_fused_check(float(value), diagnostic); activated[gid] = value;
}
#endif
