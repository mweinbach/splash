#include "FlashHCDownPackedABI.h"
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
kernel void flash_hc_down_packed_literal_witness(
    device const bfloat *x [[buffer(0)]], device const uchar *dw [[buffer(1)]],
    device const uchar *ds [[buffer(2)]], device const uchar *db [[buffer(3)]],
    device const uchar *iw [[buffer(4)]], device const uchar *is [[buffer(5)]],
    device const uchar *ib [[buffer(6)]], device bfloat *activated [[buffer(7)]],
    device bfloat *gates [[buffer(8)]], device atomic_uint *diagnostic [[buffer(9)]],
    device bfloat *rawBF [[buffer(10)]], device float *rawF [[buffer(11)]],
    constant HCDownPackedParams &params [[buffer(12)]],
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

inline void hc_down_packed_store(float sum, uint row, uint n,
    device bfloat *activated, device bfloat *gates, device atomic_uint *diagnostic,
    device bfloat *rawBF, device float *rawF, constant HCDownPackedParams &params) {
  if (params.write_debug) {
    rawBF[ulong(row) * 324 + n] = bfloat(sum);
    rawF[ulong(row) * 324 + n] = sum;
  }
  hc_fused_check(sum, diagnostic); hc_fused_check(float(bfloat(sum)), diagnostic);
  const bfloat value = n >= 320 ? hc_down_candidate_inject(sum) : hc_down_candidate_activate(sum);
  hc_fused_check(float(value), diagnostic);
  if (n >= 320) gates[ulong(row) * 4 + n - 320] = value;
  else activated[ulong(row) * 320 + n] = value;
}

// Each SIMD owns one output and only two/four chronological row accumulators.
// All coefficient bits and per-row k=lane+32*j operations match the raw route.
// Grid stays large: R16/reuse2 has648 groups instead of81 cached-F32 groups.
template <ushort Bits, ushort Group, ushort Reuse>
inline void hc_down_packed_reuse(
    device const bfloat *x, device const uchar *dw, device const uchar *ds,
    device const uchar *db, device const uchar *iw, device const uchar *is,
    device const uchar *ib, device bfloat *activated, device bfloat *gates,
    device atomic_uint *diagnostic, device bfloat *rawBF, device float *rawF,
    constant HCDownPackedParams &params, uint2 grid, uint simd, uint lane) {
  const FlashHCFusedParams p = params.literal;
  if (!hc_fused_workload(p) || p.arithmetic_mode || p.simdgroups != 4 ||
      p.down.input_size != 10240 || p.down.output_size != 320 ||
      p.down.bits != Bits || p.down.group_size != Group ||
      !hc_fused_format(p.down) || params.tile_outputs != Reuse || params.write_debug > 1 ||
      (p.has_injection && (!hc_fused_format(p.injection) ||
          p.injection.input_size != 10240 || p.injection.output_size != 4))) {
    if (!lane) atomic_fetch_or_explicit(diagnostic, 2u, memory_order_relaxed); return;
  }
  const uint n = grid.x * 4 + simd;
  const uint rowBase = grid.y * Reuse;
  if (rowBase >= p.rows || n >= 320 + p.has_injection * 4) return;
  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  if (n < 320) {
    const device uchar *w = dw + ulong(n) * p.down.weight_row_stride_bytes;
    const device bfloat *s = reinterpret_cast<const device bfloat *>(ds + ulong(n) * p.down.parameter_row_stride_bytes);
    const device bfloat *b = reinterpret_cast<const device bfloat *>(db + ulong(n) * p.down.parameter_row_stride_bytes);
    for (uint g = 0; g < 10240 / Group; ++g) {
      const float sf = float(s[g]); const float bias = float(b[g]);
      #pragma clang loop unroll(full)
      for (uint block = 0; block < Group / 32; ++block) {
        const uint k = g * Group + block * 32 + lane;
        const float code = float(hc_fused_code<Bits>(w, k, Bits));
        const float coefficient = code * sf + bias;
        const float value0 = float(x[ulong(rowBase) * 10240 + k]);
        sum0 += value0 * coefficient;
        if (rowBase + 1 < p.rows) {
          const float value1 = float(x[ulong(rowBase + 1) * 10240 + k]);
          sum1 += value1 * coefficient;
        }
        if constexpr (Reuse == 4) {
          if (rowBase + 2 < p.rows) {
            const float value2 = float(x[ulong(rowBase + 2) * 10240 + k]);
            sum2 += value2 * coefficient;
          }
          if (rowBase + 3 < p.rows) {
            const float value3 = float(x[ulong(rowBase + 3) * 10240 + k]);
            sum3 += value3 * coefficient;
          }
        }
      }
    }
  } else {
    // Mixed original injection descriptor intentionally stays independent.
    const uint index = n - 320;
    const FlashHCFusedMatrix m = p.injection;
    const device uchar *w = iw + ulong(index) * m.weight_row_stride_bytes;
    const device bfloat *s = reinterpret_cast<const device bfloat *>(is + ulong(index) * m.parameter_row_stride_bytes);
    const device bfloat *b = reinterpret_cast<const device bfloat *>(ib + ulong(index) * m.parameter_row_stride_bytes);
    for (uint g = 0; g < 10240 / m.group_size; ++g) {
      const float sf = float(s[g]); const float bias = float(b[g]);
      for (uint block = 0; block < m.group_size / 32; ++block) {
        const uint k = g * m.group_size + block * 32 + lane;
        const float code = float(hc_fused_code<0>(w, k, m.bits));
        const float coefficient = code * sf + bias;
        const float value0 = float(x[ulong(rowBase) * 10240 + k]);
        sum0 += value0 * coefficient;
        if (rowBase + 1 < p.rows) {
          const float value1 = float(x[ulong(rowBase + 1) * 10240 + k]);
          sum1 += value1 * coefficient;
        }
        if constexpr (Reuse == 4) {
          if (rowBase + 2 < p.rows) {
            const float value2 = float(x[ulong(rowBase + 2) * 10240 + k]);
            sum2 += value2 * coefficient;
          }
          if (rowBase + 3 < p.rows) {
            const float value3 = float(x[ulong(rowBase + 3) * 10240 + k]);
            sum3 += value3 * coefficient;
          }
        }
      }
    }
  }
  const float dots0 = simd_sum(sum0);
  const float dots1 = simd_sum(sum1);
  float dots2 = 0.0f, dots3 = 0.0f;
  if constexpr (Reuse == 4) { dots2 = simd_sum(sum2); dots3 = simd_sum(sum3); }
  if (!lane) {
    hc_down_packed_store(dots0, rowBase, n, activated, gates, diagnostic, rawBF, rawF, params);
    if (rowBase + 1 < p.rows) hc_down_packed_store(dots1, rowBase + 1, n, activated, gates, diagnostic, rawBF, rawF, params);
    if constexpr (Reuse == 4) {
      if (rowBase + 2 < p.rows) hc_down_packed_store(dots2, rowBase + 2, n, activated, gates, diagnostic, rawBF, rawF, params);
      if (rowBase + 3 < p.rows) hc_down_packed_store(dots3, rowBase + 3, n, activated, gates, diagnostic, rawBF, rawF, params);
    }
  }
}
#define HC_DOWN_PACKED_ENTRY(BITS, GROUP, REUSE) \
kernel void flash_hc_down_packed_q##BITS##_g##GROUP##_reuse##REUSE( \
    device const bfloat *x [[buffer(0)]], device const uchar *dw [[buffer(1)]], \
    device const uchar *ds [[buffer(2)]], device const uchar *db [[buffer(3)]], \
    device const uchar *iw [[buffer(4)]], device const uchar *is [[buffer(5)]], \
    device const uchar *ib [[buffer(6)]], device bfloat *activated [[buffer(7)]], \
    device bfloat *gates [[buffer(8)]], device atomic_uint *diagnostic [[buffer(9)]], \
    device bfloat *rawBF [[buffer(10)]], device float *rawF [[buffer(11)]], \
    constant HCDownPackedParams &params [[buffer(12)]], \
    uint2 grid [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  hc_down_packed_reuse<BITS, GROUP, REUSE>(x, dw, ds, db, iw, is, ib, activated, gates, \
      diagnostic, rawBF, rawF, params, grid, simd, lane); \
}
#define HC_DOWN_PACKED_GROUPS(BITS, REUSE) \
HC_DOWN_PACKED_ENTRY(BITS, 32, REUSE) \
HC_DOWN_PACKED_ENTRY(BITS, 64, REUSE) \
HC_DOWN_PACKED_ENTRY(BITS, 128, REUSE)
#define HC_DOWN_PACKED_BITS(REUSE) \
HC_DOWN_PACKED_GROUPS(4, REUSE) \
HC_DOWN_PACKED_GROUPS(5, REUSE) \
HC_DOWN_PACKED_GROUPS(6, REUSE) \
HC_DOWN_PACKED_GROUPS(8, REUSE)
HC_DOWN_PACKED_BITS(2)
HC_DOWN_PACKED_BITS(4)
#undef HC_DOWN_PACKED_BITS
#undef HC_DOWN_PACKED_GROUPS
#undef HC_DOWN_PACKED_ENTRY

kernel void flash_hc_down_packed_epilog_from_raw(
    device const bfloat *raw [[buffer(0)]], device bfloat *activated [[buffer(1)]],
    device bfloat *gates [[buffer(2)]], device atomic_uint *diagnostic [[buffer(3)]],
    constant HCDownPackedParams &params [[buffer(4)]], uint tid [[thread_position_in_grid]]) {
  const uint row = tid / 324, n = tid % 324;
  if (row >= params.literal.rows || n >= 320 + params.literal.has_injection * 4) return;
  const bfloat value = n >= 320 ? hc_down_candidate_inject(float(raw[tid])) : hc_down_candidate_activate(float(raw[tid]));
  hc_fused_check(float(value), diagnostic);
  if (n >= 320) gates[ulong(row) * 4 + n - 320] = value;
  else activated[ulong(row) * 320 + n] = value;
}
