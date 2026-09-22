// PRIVATE standalone ordinary R1 HC term production / chronological replay.
// No production entry is replaced. The original immutable AIR is the control.
// Original source SHA256:
// c32cff6a4b4aab8efc8b0cfcc8212c191684a4b5d9d451f77b5c4a19d17d9a16
// Source journal: LITERAL-BEGIN..END is the original prefix through down with
// only its header include and hc_fused_ -> hc_r1_original_ symbol renaming.
// PROBE-DOT is that exact dot plus argument plumbing and the JOURNAL-DOT store.
// PROBE-DOWN is that exact down plus argument plumbing and JOURNAL-DOWN stores.
// Removing those explicit journal additions restores the literal helper bytes.
// Shipping replay changes only the original partial producer; its simd_sum and
// complete scalar post body are literal. A Tap template adds only boundary stores.
// Compiled SSA/FP attrs and device bit gates remain REQUIRED before timing.
// Full command count/grid is also host-proved; builtins reject malformed dispatch
// geometry but cannot prove that a valid dispatch command was actually submitted.
// Term plane layout is F32[n][j][lane], (n*320+j)*32+lane; no partial sums.
// LITERAL-BEGIN
#include "abi.hpp"
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline bfloat hc_r1_original_sigmoid_fast(bfloat x) {
  const bfloat e = bfloat(metal::exp(metal::abs(float(x))));
  const bfloat d = bfloat(1.0f) + e;
  const bfloat t = bfloat(1.0f) / d;
  return x < bfloat(0.0f) ? t : bfloat(1.0f) - t;
}
inline bfloat hc_r1_original_sigmoid_unary(bfloat x) {
  const bfloat e = bfloat(metal::precise::exp(metal::abs(float(x))));
  const bfloat d = bfloat(1.0f) + e;
  const bfloat t = bfloat(1.0f) / d;
  return x < bfloat(0.0f) ? t : bfloat(1.0f) - t;
}
inline void hc_r1_original_check(float x, device atomic_uint *diagnostics) {
  if ((as_type<uint>(x) & 0x7f800000u) == 0x7f800000u)
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
}
inline bool hc_r1_original_format(FlashHCFusedMatrix m) {
  return m.input_size && m.output_size &&
      (m.bits == 4 || m.bits == 5 || m.bits == 6 || m.bits == 8) &&
      (m.group_size == 32 || m.group_size == 64 || m.group_size == 128) &&
      m.input_size % m.group_size == 0 &&
      m.weight_row_stride_bytes >= (ulong(m.input_size) * m.bits + 7) / 8 &&
      m.parameter_row_stride_bytes >= ulong(m.input_size / m.group_size) * 2 &&
      m.parameter_row_stride_bytes % 2 == 0;
}
inline bool hc_r1_original_workload(FlashHCFusedParams p) {
  return p.rows && p.rows <= 32 && p.width == 2560 && p.streams == 4 &&
      p.lowrank == 320 && p.arithmetic_mode <= 1 && p.has_injection <= 1 &&
      p.write_raw_up <= 1 && (p.simdgroups == 4 || p.simdgroups == 8);
}

template <ushort Bits>
inline uint hc_r1_original_code(const device uchar *w, uint k, uint dynamic_bits) {
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
inline float hc_r1_original_dot(
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
      const float code = float(hc_r1_original_code<Bits>(w, k, m.bits));
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

template <ushort Bits, ushort Group, ushort SG>
inline void hc_r1_original_down(
    const device bfloat *normalized,
    const device uchar *dw, const device uchar *ds, const device uchar *db,
    const device uchar *iw, const device uchar *is, const device uchar *ib,
    device bfloat *activated, device bfloat *gates,
    device atomic_uint *diagnostics, FlashHCFusedParams p,
    uint2 grid, uint simd, uint lane) {
  if (!hc_r1_original_workload(p) || p.simdgroups != SG ||
      !hc_r1_original_format(p.down) || p.down.bits != Bits || p.down.group_size != Group ||
      p.down.input_size != 10240 || p.down.output_size != 320 ||
      (p.has_injection && (!hc_r1_original_format(p.injection) ||
                           p.injection.input_size != 10240 || p.injection.output_size != 4))) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint n = grid.x * SG + simd;
  if (grid.y >= p.rows || n >= p.lowrank + p.has_injection * p.streams) return;
  const device bfloat *x = normalized + ulong(grid.y) * p.width * p.streams;
  float partial;
  if (n < p.lowrank) {
    partial = hc_r1_original_dot<Bits, Group>(x, dw, ds, db, p.down, n, lane, p.arithmetic_mode);
  } else {
    const uint index = n - p.lowrank;
    if (p.injection.bits == Bits && p.injection.group_size == Group)
      partial = hc_r1_original_dot<Bits, Group>(x, iw, is, ib, p.injection, index, lane, p.arithmetic_mode);
    else
      partial = hc_r1_original_dot<0, 0>(x, iw, is, ib, p.injection, index, lane, p.arithmetic_mode);
  }
  const float sum = simd_sum(partial);
  if (!lane) {
    // This raw projection boundary remains even though it is register-only.
    const bfloat raw = bfloat(sum);
    hc_r1_original_check(sum, diagnostics); hc_r1_original_check(float(raw), diagnostics);
    const bfloat divided = bfloat(float(raw) / float(p.streams));
    if (n < p.lowrank) {
      const bfloat sigmoid = hc_r1_original_sigmoid_fast(divided);
      const bfloat result = bfloat(float(divided) * float(sigmoid));
      activated[ulong(grid.y) * p.lowrank + n] = result;
      hc_r1_original_check(float(result), diagnostics);
    } else {
      const bfloat sigmoid = hc_r1_original_sigmoid_unary(divided);
      const bfloat result = bfloat(2.0f * float(sigmoid));
      gates[ulong(grid.y) * p.streams + n - p.lowrank] = result;
      hc_r1_original_check(float(result), diagnostics);
    }
  }
}

// LITERAL-END

// Canonical HC metadata checks are duplicated before ANY term/output/tap write.
// This experiment adds fixed R1/mode0/SG4 and zero-reserved metadata admission.
template <ushort Bits, ushort Group>
inline bool hc_r1_metadata(HCChronologicalTermsParams p) {
  const FlashHCFusedParams q = p.literal;
  return hc_r1_original_workload(q) && q.rows == 1 && q.arithmetic_mode == 0 &&
      q.simdgroups == 4 && !q.reserved0 && !q.reserved1 && !q.reserved2 &&
      !q.reserved3 && !q.reserved4 &&
      hc_r1_original_format(q.down) && q.down.bits == Bits &&
      q.down.group_size == Group && q.down.input_size == 10240 &&
      q.down.output_size == 320 &&
      (!q.has_injection || (hc_r1_original_format(q.injection) &&
                           q.injection.input_size == 10240 &&
                           q.injection.output_size == 4)) &&
      p.term_outputs == 320 + q.has_injection * 4 && p.term_j == 320 &&
      p.slice_terms == 8 && !p.reserved0;
}

template <ushort Bits, ushort Group, ushort Slices>
inline bool hc_r1_dispatch(HCChronologicalTermsParams p, uint3 grid,
    uint3 total_groups, uint3 threads, uint simd, uint lane,
    device atomic_uint *diagnostics) {
  const bool metadata = hc_r1_metadata<Bits, Group>(p);
  const uint expected_x = metadata ? p.term_outputs / 4 : 0;
  const bool valid = metadata && threads.x == 128 && threads.y == 1 &&
      threads.z == 1 && total_groups.x == expected_x &&
      total_groups.y == Slices && total_groups.z == 1 &&
      grid.x < expected_x && grid.y < Slices && grid.z == 0 &&
      simd < 4 && lane < 32;
  if (!valid && !lane)
    atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
  return valid;
}

// The original nested g/block traversal is restricted to exactly one K256
// slice. Group32/64/128 all divide256. There is NO accumulation in this stage.
// All code extraction, casts, coefficient/product operations are literal mode0.
template <ushort Bits, ushort Group>
inline void hc_r1_produce_slice(
    const device bfloat *x, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    FlashHCFusedMatrix m, uint n, uint lane, uint slice,
    device float *terms, uint term_n) {
  const uint group_size = Group ? Group : m.group_size;
  const device uchar *w = weights + ulong(n) * m.weight_row_stride_bytes;
  const device bfloat *s = reinterpret_cast<const device bfloat *>(
      scales + ulong(n) * m.parameter_row_stride_bytes);
  const device bfloat *b = reinterpret_cast<const device bfloat *>(
      biases + ulong(n) * m.parameter_row_stride_bytes);
  const uint first_group = slice * 256 / group_size;
  const uint end_group = (slice + 1) * 256 / group_size;
  for (uint g = first_group; g < end_group; ++g) {
    const float sf = float(s[g]);
    const float bias = float(b[g]);
    for (uint block = 0; block < group_size / 32; ++block) {
      const uint k = g * group_size + block * 32 + lane;
      const float value = float(x[k]);
      const float code = float(hc_r1_original_code<Bits>(w, k, m.bits));
      const float coefficient = code * sf + bias;
      terms[ulong(term_n) * 10240 + k] = value * coefficient;
    }
  }
}

// The original 320 lane additions are replayed in the identical ascending order.
// Each stored operand is an F32 original term; no group/slice partial sum exists.
inline float hc_r1_chronological_lane_sum(
    const device float *terms, uint n, uint lane) {
  float sum = 0.0f;
  for (uint j = 0; j < 320; ++j)
    sum += terms[ulong(n) * 10240 + j * 32 + lane];
  return sum;
}

// REPLAY-DOWN-BEGIN: original validation/reduction/post with partial replacement.
template <ushort Bits, ushort Group, ushort SG, bool Tap>
inline void hc_r1_chronological_down(
    const device bfloat *normalized,
    const device uchar *dw, const device uchar *ds, const device uchar *db,
    const device uchar *iw, const device uchar *is, const device uchar *ib,
    device bfloat *activated, device bfloat *gates,
    device atomic_uint *diagnostics, FlashHCFusedParams p,
    uint2 grid, uint simd, uint lane, const device float *terms,
    device float *raw_f32, device bfloat *raw_bf16) {
  if (!hc_r1_original_workload(p) || p.simdgroups != SG ||
      !hc_r1_original_format(p.down) || p.down.bits != Bits || p.down.group_size != Group ||
      p.down.input_size != 10240 || p.down.output_size != 320 ||
      (p.has_injection && (!hc_r1_original_format(p.injection) ||
                           p.injection.input_size != 10240 || p.injection.output_size != 4))) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint n = grid.x * SG + simd;
  if (grid.y >= p.rows || n >= p.lowrank + p.has_injection * p.streams) return;
  const device bfloat *x = normalized + ulong(grid.y) * p.width * p.streams;
  // UNUSED-ONLY: base bindings retained; replay reads the produced terms.
  (void)x; (void)dw; (void)ds; (void)db; (void)iw; (void)is; (void)ib;
  const float partial = hc_r1_chronological_lane_sum(terms, n, lane);
  const float sum = simd_sum(partial);
  if (!lane) {
    // This raw projection boundary remains even though it is register-only.
    const bfloat raw = bfloat(sum);
    if (Tap) { raw_f32[n] = sum; raw_bf16[n] = raw; }
    hc_r1_original_check(sum, diagnostics); hc_r1_original_check(float(raw), diagnostics);
    const bfloat divided = bfloat(float(raw) / float(p.streams));
    if (n < p.lowrank) {
      const bfloat sigmoid = hc_r1_original_sigmoid_fast(divided);
      const bfloat result = bfloat(float(divided) * float(sigmoid));
      activated[ulong(grid.y) * p.lowrank + n] = result;
      hc_r1_original_check(float(result), diagnostics);
    } else {
      const bfloat sigmoid = hc_r1_original_sigmoid_unary(divided);
      const bfloat result = bfloat(2.0f * float(sigmoid));
      gates[ulong(grid.y) * p.streams + n - p.lowrank] = result;
      hc_r1_original_check(float(result), diagnostics);
    }
  }
}


// REPLAY-DOWN-END

// PROBE-DOT-BEGIN
template <ushort Bits, ushort Group>
inline float hc_r1_original_probe_dot(
    const device bfloat *x, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    FlashHCFusedMatrix m, uint n, uint lane, uint mode,
    device float *direct_terms, uint term_n) {
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
      const float code = float(hc_r1_original_code<Bits>(w, k, m.bits));
      if (mode == 0) {
        const float coefficient = code * sf + bias;
        // JOURNAL-DOT: one extra original product store; recurrence untouched.
        direct_terms[ulong(term_n) * 10240 + k] = value * coefficient;
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

// PROBE-DOT-END

// PROBE-DOWN-BEGIN
template <ushort Bits, ushort Group, ushort SG>
inline void hc_r1_original_probe_down(
    const device bfloat *normalized,
    const device uchar *dw, const device uchar *ds, const device uchar *db,
    const device uchar *iw, const device uchar *is, const device uchar *ib,
    device bfloat *activated, device bfloat *gates,
    device atomic_uint *diagnostics, FlashHCFusedParams p,
    uint2 grid, uint simd, uint lane,
    device float *raw_f32, device bfloat *raw_bf16, device float *direct_terms) {
  if (!hc_r1_original_workload(p) || p.simdgroups != SG ||
      !hc_r1_original_format(p.down) || p.down.bits != Bits || p.down.group_size != Group ||
      p.down.input_size != 10240 || p.down.output_size != 320 ||
      (p.has_injection && (!hc_r1_original_format(p.injection) ||
                           p.injection.input_size != 10240 || p.injection.output_size != 4))) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint n = grid.x * SG + simd;
  if (grid.y >= p.rows || n >= p.lowrank + p.has_injection * p.streams) return;
  const device bfloat *x = normalized + ulong(grid.y) * p.width * p.streams;
  float partial;
  if (n < p.lowrank) {
    partial = hc_r1_original_probe_dot<Bits, Group>(x, dw, ds, db, p.down, n, lane, p.arithmetic_mode, direct_terms, n);
  } else {
    const uint index = n - p.lowrank;
    if (p.injection.bits == Bits && p.injection.group_size == Group)
      partial = hc_r1_original_probe_dot<Bits, Group>(x, iw, is, ib, p.injection, index, lane, p.arithmetic_mode, direct_terms, n);
    else
      partial = hc_r1_original_probe_dot<0, 0>(x, iw, is, ib, p.injection, index, lane, p.arithmetic_mode, direct_terms, n);
  }
  const float sum = simd_sum(partial);
  if (!lane) {
    // This raw projection boundary remains even though it is register-only.
    const bfloat raw = bfloat(sum);
    // JOURNAL-DOWN: only raw boundary stores are added.
    raw_f32[n] = sum;
    raw_bf16[n] = raw;
    hc_r1_original_check(sum, diagnostics); hc_r1_original_check(float(raw), diagnostics);
    const bfloat divided = bfloat(float(raw) / float(p.streams));
    if (n < p.lowrank) {
      const bfloat sigmoid = hc_r1_original_sigmoid_fast(divided);
      const bfloat result = bfloat(float(divided) * float(sigmoid));
      activated[ulong(grid.y) * p.lowrank + n] = result;
      hc_r1_original_check(float(result), diagnostics);
    } else {
      const bfloat sigmoid = hc_r1_original_sigmoid_unary(divided);
      const bfloat result = bfloat(2.0f * float(sigmoid));
      gates[ulong(grid.y) * p.streams + n - p.lowrank] = result;
      hc_r1_original_check(float(result), diagnostics);
    }
  }
}

// PROBE-DOWN-END

#define HC_R1_BASE_BUFFERS \
    const device bfloat *normalized [[buffer(0)]], \
    const device uchar *dw [[buffer(1)]], const device uchar *ds [[buffer(2)]], \
    const device uchar *db [[buffer(3)]], const device uchar *iw [[buffer(4)]], \
    const device uchar *is [[buffer(5)]], const device uchar *ib [[buffer(6)]], \
    device bfloat *activated [[buffer(7)]], device bfloat *gates [[buffer(8)]], \
    device atomic_uint *diagnostics [[buffer(9)]], \
    device float *terms [[buffer(10)]], \
    constant HCChronologicalTermsParams &p [[buffer(11)]]
#define HC_R1_GEOMETRY \
    uint3 grid [[threadgroup_position_in_grid]], \
    uint3 total_groups [[threadgroups_per_grid]], \
    uint3 threads [[threads_per_threadgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]

#define HC_R1_ENTRIES(BITS, GROUP) \
kernel void flash_hc_r1_terms_q##BITS##_g##GROUP##_s4( \
    HC_R1_BASE_BUFFERS, HC_R1_GEOMETRY) { \
  if (!hc_r1_dispatch<BITS, GROUP, 40>(p, grid, total_groups, threads, simd, lane, diagnostics)) return; \
  (void)activated; (void)gates; /* UNUSED-ONLY: producer preserves output bindings. */ \
  const uint n = grid.x * 4 + simd; \
  if (n < p.literal.lowrank) { \
    hc_r1_produce_slice<BITS, GROUP>(normalized, dw, ds, db, p.literal.down, n, lane, grid.y, terms, n); \
  } else { \
    const uint index = n - p.literal.lowrank; \
    if (p.literal.injection.bits == BITS && p.literal.injection.group_size == GROUP) \
      hc_r1_produce_slice<BITS, GROUP>(normalized, iw, is, ib, p.literal.injection, index, lane, grid.y, terms, n); \
    else \
      hc_r1_produce_slice<0, 0>(normalized, iw, is, ib, p.literal.injection, index, lane, grid.y, terms, n); \
  } \
} \
kernel void flash_hc_r1_chronological_sum_q##BITS##_g##GROUP##_s4( \
    HC_R1_BASE_BUFFERS, HC_R1_GEOMETRY) { \
  if (!hc_r1_dispatch<BITS, GROUP, 1>(p, grid, total_groups, threads, simd, lane, diagnostics)) return; \
  hc_r1_chronological_down<BITS, GROUP, 4, false>(normalized, dw, ds, db, iw, is, ib, \
      activated, gates, diagnostics, p.literal, uint2(grid.x, 0), simd, lane, terms, nullptr, nullptr); \
} \
kernel void flash_hc_r1_chronological_sum_probe_q##BITS##_g##GROUP##_s4( \
    HC_R1_BASE_BUFFERS, device float *raw_f32 [[buffer(12)]], \
    device bfloat *raw_bf16 [[buffer(13)]], HC_R1_GEOMETRY) { \
  if (!hc_r1_dispatch<BITS, GROUP, 1>(p, grid, total_groups, threads, simd, lane, diagnostics)) return; \
  hc_r1_chronological_down<BITS, GROUP, 4, true>(normalized, dw, ds, db, iw, is, ib, \
      activated, gates, diagnostics, p.literal, uint2(grid.x, 0), simd, lane, terms, raw_f32, raw_bf16); \
} \
kernel void flash_hc_r1_original_probe_q##BITS##_g##GROUP##_s4( \
    HC_R1_BASE_BUFFERS, device float *raw_f32 [[buffer(12)]], \
    device bfloat *raw_bf16 [[buffer(13)]], device float *direct_terms [[buffer(14)]], HC_R1_GEOMETRY) { \
  if (!hc_r1_dispatch<BITS, GROUP, 1>(p, grid, total_groups, threads, simd, lane, diagnostics)) return; \
  (void)terms; /* UNUSED-ONLY: original direct probe never reads shipping scratch. */ \
  hc_r1_original_probe_down<BITS, GROUP, 4>(normalized, dw, ds, db, iw, is, ib, \
      activated, gates, diagnostics, p.literal, uint2(grid.x, 0), simd, lane, raw_f32, raw_bf16, direct_terms); \
}

#define HC_R1_GROUPS(BITS) \
HC_R1_ENTRIES(BITS, 32) \
HC_R1_ENTRIES(BITS, 64) \
HC_R1_ENTRIES(BITS, 128)
HC_R1_GROUPS(4)
HC_R1_GROUPS(5)
HC_R1_GROUPS(6)
HC_R1_GROUPS(8)
#undef HC_R1_GROUPS
#undef HC_R1_ENTRIES
#undef HC_R1_GEOMETRY
#undef HC_R1_BASE_BUFFERS
