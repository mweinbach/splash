// Private one-layer numerical activation/association alternative. This is
// native device signed-I8 x centered packed signed-I4 -> exact I32 G64 dots,
// followed by manual F32 affine reconstruction and ascending G64 accumulation.
// The original checkpoint, BF16 scale/bias storage and decode Q4 path are not
// changed. Metal 4.1's primary SDK matmul header explicitly lists this input
// and destination type combination; no register_input format is used here.
#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "dev/benchmarks/prefill_w4a8_sep21/abi.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

inline void prefill_w4a8_sep21_error(device uint *diag, uint bits) {
  atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(diag), bits,
                           memory_order_relaxed);
}
inline bool prefill_w4a8_sep21_finite(float value) {
  return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}
inline bool prefill_w4a8_sep21_finite(bfloat value) {
  return (as_type<ushort>(value) & 0x7f80u) != 0x7f80u;
}

// This helper has a unique symbol and the same explicitly compiled BF16
// boundaries as the baseline DirectA SwiGLU. It is independent of every
// previously linked production/control helper and of the I32/F32 epilogue.
#pragma METAL fp math_mode(fast)
inline bfloat prefill_w4a8_sep21_native_bf16_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

template <uint K>
inline void prefill_w4a8_sep21_quantize_row(
    device const bfloat *input, device const uint *offsets,
    device int8_t *codes, device float *row_scales, device int32_t *group_sums,
    device uint *diag, constant PrefillW4A8QuantParams &p,
    uint3 group, uint3 threads, uint tid, threadgroup float *maxima) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const constant FlashMoEBucketParams &b = p.bucket;
  if (!b.rows || b.rows > kFlashMoEBucketMaximumRows || !b.selections ||
      b.selections > kFlashMoEBucketMaximumSelections || b.width != K ||
      b.experts != 512 || b.routes != b.rows * b.selections ||
      b.tile_rows || b.job_capacity || b.reserved ||
      p.physical_rows != b.routes + kPrefillW4A8GuardRows ||
      p.input_row_stride != K || p.code_row_stride != K ||
      p.sum_row_stride != K / 64 || p.reserved0 || p.reserved1 ||
      p.reserved2 || p.reserved3 || threads.x != kPrefillW4A8QuantThreads ||
      threads.y != 1 || threads.z != 1 || group.x >= p.physical_rows ||
      group.y || group.z) {
    if (!tid) prefill_w4a8_sep21_error(diag, 2u); return;
  }
  const uint active = offsets[512];
  if (active > b.routes || offsets[0]) {
    if (!tid) prefill_w4a8_sep21_error(diag, 2u); return;
  }
  const uint row = group.x, lane = tid & 31u, simd = tid >> 5;
  const ulong code_base = ulong(row) * p.code_row_stride;
  const ulong sum_base = ulong(row) * p.sum_row_stride;
  if (row >= active) {
    for (uint k = tid; k < K; k += kPrefillW4A8QuantThreads)
      codes[code_base + k] = int8_t(0);
    for (uint g = tid; g < K / 64; g += kPrefillW4A8QuantThreads)
      group_sums[sum_base + g] = int32_t(0);
    if (!tid) row_scales[row] = 1.0f;
    return;
  }
  const ulong source_base = ulong(row) * p.input_row_stride;
  float maximum = 0.0f;
  bool nonfinite = false;
  for (uint k = tid; k < K; k += kPrefillW4A8QuantThreads) {
    const bfloat source = input[source_base + k];
    const bool finite = prefill_w4a8_sep21_finite(source);
    nonfinite |= !finite;
    if (finite) maximum = max(maximum, abs(float(source)));
  }
  maximum = simd_max(maximum);
  if (!lane) maxima[simd] = maximum;
  // The only CTA barrier. Each SIMD independently reduces the eight row-max
  // partials, so no device readback or second barrier is needed for its scale.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maximum = simd_max(lane < 8 ? maxima[lane] : 0.0f);
  const float scale = maximum > 0.0f
      ? max(maximum / 127.0f, 0x1p-126f) : 1.0f;
  if (!tid) row_scales[row] = scale;
  if (nonfinite) prefill_w4a8_sep21_error(diag, 4u);
  for (uint g = simd; g < K / 64; g += 8) {
    const uint first = g * 64 + lane, second = first + 32;
    const bfloat x0 = input[source_base + first], x1 = input[source_base + second];
    const float v0 = prefill_w4a8_sep21_finite(x0) ? float(x0) : 0.0f;
    const float v1 = prefill_w4a8_sep21_finite(x1) ? float(x1) : 0.0f;
    const int32_t q0 = int32_t(clamp(rint(v0 / scale), -127.0f, 127.0f));
    const int32_t q1 = int32_t(clamp(rint(v1 / scale), -127.0f, 127.0f));
    codes[code_base + first] = int8_t(q0);
    codes[code_base + second] = int8_t(q1);
    const int32_t sum = simd_sum(q0 + q1);
    if (!lane) group_sums[sum_base + g] = sum;
  }
}

#define PREFILL_W4A8_QUANT_ENTRY(NAME, K) \
kernel void NAME(device const bfloat *input [[buffer(0)]], \
    device const uint *offsets [[buffer(1)]], device int8_t *codes [[buffer(2)]], \
    device float *row_scales [[buffer(3)]], device int32_t *group_sums [[buffer(4)]], \
    device uint *diag [[buffer(5)]], constant PrefillW4A8QuantParams &p [[buffer(6)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  threadgroup float maxima[8]; \
  prefill_w4a8_sep21_quantize_row<K>(input, offsets, codes, row_scales, group_sums, \
      diag, p, group, threads, tid, maxima); \
}
PREFILL_W4A8_QUANT_ENTRY(prefill_w4a8_sep21_quant_packed, 2560)
PREFILL_W4A8_QUANT_ENTRY(prefill_w4a8_sep21_quant_down, 640)
#undef PREFILL_W4A8_QUANT_ENTRY

inline bool prefill_w4a8_sep21_original_strides(uint n, uint k,
    ulong weight_row, ulong weight_expert, ulong parameter_row, ulong parameter_expert) {
  constexpr ulong top = ~ulong(0);
  if (weight_row < k / 2 || weight_row % 4 || weight_expert % 4 ||
      parameter_row < k / 32 || parameter_row % 2 || parameter_expert % 2 ||
      weight_row > (top - k / 2) / (n - 1) ||
      parameter_row > (top - k / 32) / (n - 1)) return false;
  const ulong weight_minimum = ulong(n - 1) * weight_row + k / 2;
  const ulong parameter_minimum = ulong(n - 1) * parameter_row + k / 32;
  return weight_expert >= weight_minimum && parameter_expert >= parameter_minimum &&
      weight_expert <= (top - weight_minimum) / 511 &&
      parameter_expert <= (top - parameter_minimum) / 511;
}

inline bool prefill_w4a8_sep21_job(device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    uint job_index, uint capacity, uint routes, thread uint &expert,
    thread uint &row_begin, thread uint &row_end, device uint *diag) {
  const uint active = offsets[512], count = job_count[0];
  if (active > routes || offsets[0] || count > capacity) {
    prefill_w4a8_sep21_error(diag, 2u); return false;
  }
  if (job_index >= count) return false;
  const FlashMoEBucketJob job = jobs[job_index];
  if (job.expert >= 512) { prefill_w4a8_sep21_error(diag, 1u); return false; }
  const uint first = offsets[job.expert], last = offsets[job.expert + 1];
  if (first > last || last > active || job.row_begin < first || job.row_begin >= last ||
      (job.row_begin - first) % kPrefillW4A8TileRows) {
    prefill_w4a8_sep21_error(diag, 2u); return false;
  }
  expert = job.expert; row_begin = job.row_begin; row_end = last;
  return true;
}

// Unsupported result layouts must make every participating thread reject
// together before cooperative execution. This does not rely on undocumented
// per-thread capacity uniformity. SG1 needs no CTA barrier; SG2 uses one flag
// per SIMD and one pre-operation barrier, outside the G64 loop.
template <ushort SG>
inline bool prefill_w4a8_sep21_capacity_failure(bool local_failure,
    uint tid, threadgroup uint *failures) {
  const bool simd_failure = simd_any(local_failure);
  if constexpr (SG == 1) return simd_failure;
  else {
    if (!(tid & 31u)) failures[tid >> 5] = uint(simd_failure);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool failure = false;
#pragma unroll
    for (ushort i = 0; i < SG; ++i) failure |= failures[i] != 0;
    return failure;
  }
}

// Every stage is intentionally F32 with contraction/reassociation disabled.
// The centered signed nibble is original unsigned q-8, hence b+8*s is the
// affine bias correction. Integer dot and group sum are exact before casting.
inline float prefill_w4a8_sep21_group_term(int32_t dot, int32_t sum,
    bfloat source_scale, bfloat source_bias, device uint *diag) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const float scale = float(source_scale), bias = float(source_bias);
  const float scale_dot = scale * float(dot);
  const float centered_bias = bias + 8.0f * scale;
  const float bias_dot = centered_bias * float(sum);
  const float group_term = scale_dot + bias_dot;
  if (!prefill_w4a8_sep21_finite(scale) || !prefill_w4a8_sep21_finite(bias) ||
      !prefill_w4a8_sep21_finite(group_term)) prefill_w4a8_sep21_error(diag, 4u);
  return group_term;
}

template <ushort N, ushort SG, bool Audit>
inline void prefill_w4a8_sep21_gate_tile(
    device const int8_t *input, device const float *row_scales,
    device const int32_t *group_sums, device const uchar *gate_w,
    device const uchar *gate_s, device const uchar *gate_b,
    device const uchar *up_w, device const uchar *up_s, device const uchar *up_b,
    device const uint *offsets, device const FlashMoEBucketJob *jobs,
    device const uint *job_count, device bfloat *output, device uint *diag,
    constant PrefillW4A8GateParams &params, device float *raw_gate, device float *raw_up,
    uint3 group, uint3 threads, uint tid, threadgroup uint *capacity_failures) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const constant FlashMoEBlockedGateParams &blocked = params.source;
  const constant FlashMoEFusedParams &p = blocked.affine;
  constexpr ushort M = kPrefillW4A8TileRows, BK = 64;
  if (!p.rows || p.rows > kFlashMoEBucketMaximumRows || !p.selections ||
      p.selections > kFlashMoEBucketMaximumSelections || p.input_size != 2560 ||
      p.output_size != 640 || p.experts != 512 || p.reserved0 || p.reserved1 ||
      p.reserved2 || blocked.reserved || params.reserved ||
      blocked.route_capacity != p.rows * p.selections || blocked.tile_rows != M ||
      blocked.job_capacity != (blocked.route_capacity + M - 1) / M + 511 ||
      params.physical_rows != blocked.route_capacity + kPrefillW4A8GuardRows ||
      params.activation_row_stride != 2560 || params.sum_row_stride != 40 ||
      params.gate_centered_row_stride_bytes != 1280 ||
      params.gate_centered_expert_stride_bytes != 819200 ||
      params.up_centered_row_stride_bytes != 1280 ||
      params.up_centered_expert_stride_bytes != 819200 ||
      threads.x != uint(SG) * 32 || threads.y != 1 || threads.z != 1 ||
      group.x >= 640 / N || group.y >= blocked.job_capacity || group.z ||
      !prefill_w4a8_sep21_original_strides(640, 2560, p.gate_weight_row_stride_bytes,
          p.gate_weight_expert_stride_bytes, p.gate_parameter_row_stride_bytes,
          p.gate_parameter_expert_stride_bytes) ||
      !prefill_w4a8_sep21_original_strides(640, 2560, p.up_weight_row_stride_bytes,
          p.up_weight_expert_stride_bytes, p.up_parameter_row_stride_bytes,
          p.up_parameter_expert_stride_bytes)) {
    if (!tid) prefill_w4a8_sep21_error(diag, 2u); return;
  }
  uint expert = 0, begin = 0, end = 0;
  if (!prefill_w4a8_sep21_job(offsets, jobs, job_count, group.y,
      blocked.job_capacity, blocked.route_capacity, expert, begin, end, diag)) return;
  const uint column_begin = group.x * N;
  auto a = tensor(const_cast<device int8_t *>(input + ulong(begin) * params.activation_row_stride),
      dextents<int, 2>{2560, M}, array<int, 2>{1, int(params.activation_row_stride)});
  auto a0 = a.template slice<BK, M>(0, 0);
  const ulong gate_base = ulong(expert) * params.gate_centered_expert_stride_bytes +
      ulong(column_begin) * params.gate_centered_row_stride_bytes;
  const ulong up_base = ulong(expert) * params.up_centered_expert_stride_bytes +
      ulong(column_begin) * params.up_centered_row_stride_bytes;
  tensor<device int4b_format, dextents<int, 2>, tensor_inline> g0(
      const_cast<device uchar *>(gate_w + gate_base), dextents<int, 2>{BK, N},
      array<int, 2>{1, int(params.gate_centered_row_stride_bytes * 2)});
  tensor<device int4b_format, dextents<int, 2>, tensor_inline> u0(
      const_cast<device uchar *>(up_w + up_base), dextents<int, 2>{BK, N},
      array<int, 2>{1, int(params.up_centered_row_stride_bytes * 2)});
  constexpr auto descriptor = matmul2d_descriptor(M, N, BK, false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto gate_dot = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(g0), int32_t>();
  auto up_dot = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(u0), int32_t>();
  // Manual F32 arrays use only native I32 result indices/validity. A uniform
  // collective check rejects any compiler layout whose capacity exceeds the
  // descriptor's per-thread logical allocation; no padded layout is assumed.
  static_assert(sizeof(float) == sizeof(int32_t), "F32/I32 storage width");
  constexpr uint AccumulatorCapacity = uint(M) * N / (uint(SG) * 32);
  float gate_acc[AccumulatorCapacity], up_acc[AccumulatorCapacity];
  if (prefill_w4a8_sep21_capacity_failure<SG>(gate_dot.get_capacity() > AccumulatorCapacity ||
      up_dot.get_capacity() != gate_dot.get_capacity(), tid, capacity_failures)) {
    if (!tid) prefill_w4a8_sep21_error(diag, 2u); return;
  }
#pragma unroll
  for (ushort i = 0; i < gate_dot.get_capacity(); ++i)
    if (gate_dot.is_valid_element(i)) { gate_acc[i] = 0.0f; up_acc[i] = 0.0f; }
  for (uint chunk = 0; chunk < 40; ++chunk) {
    auto achunk = a.template slice<BK, M>(chunk * BK, 0);
    tensor<device int4b_format, dextents<int, 2>, tensor_inline> g(
        const_cast<device uchar *>(gate_w + gate_base + chunk * 32),
        dextents<int, 2>{BK, N}, array<int, 2>{1, int(params.gate_centered_row_stride_bytes * 2)});
    tensor<device int4b_format, dextents<int, 2>, tensor_inline> u(
        const_cast<device uchar *>(up_w + up_base + chunk * 32),
        dextents<int, 2>{BK, N}, array<int, 2>{1, int(params.up_centered_row_stride_bytes * 2)});
    operation.run(achunk, g, gate_dot);
    operation.run(achunk, u, up_dot);
#pragma unroll
    for (ushort i = 0; i < gate_dot.get_capacity(); ++i) {
      if (!gate_dot.is_valid_element(i)) continue;
      const auto index = gate_dot.get_multidimensional_index(i);
      const uint row = begin + index[1], n = column_begin + index[0];
      if (row >= end) continue;
      const int32_t sum = group_sums[ulong(row) * params.sum_row_stride + chunk];
      const ulong gate_parameter = ulong(expert) * p.gate_parameter_expert_stride_bytes +
          ulong(n) * p.gate_parameter_row_stride_bytes + chunk * 2;
      const ulong up_parameter = ulong(expert) * p.up_parameter_expert_stride_bytes +
          ulong(n) * p.up_parameter_row_stride_bytes + chunk * 2;
      const bfloat gs = *reinterpret_cast<device const bfloat *>(gate_s + gate_parameter);
      const bfloat gb = *reinterpret_cast<device const bfloat *>(gate_b + gate_parameter);
      const bfloat us = *reinterpret_cast<device const bfloat *>(up_s + up_parameter);
      const bfloat ub = *reinterpret_cast<device const bfloat *>(up_b + up_parameter);
      gate_acc[i] = gate_acc[i] + prefill_w4a8_sep21_group_term(gate_dot[i], sum, gs, gb, diag);
      up_acc[i] = up_acc[i] + prefill_w4a8_sep21_group_term(up_dot[i], sum, us, ub, diag);
    }
  }
#pragma unroll
  for (ushort i = 0; i < gate_dot.get_capacity(); ++i) {
    if (!gate_dot.is_valid_element(i)) continue;
    const auto index = gate_dot.get_multidimensional_index(i);
    const uint row = begin + index[1], n = column_begin + index[0];
    if (row >= end) continue;
    const float activation_scale = row_scales[row];
    const float gate_linear = gate_acc[i] * activation_scale;
    const float up_linear = up_acc[i] * activation_scale;
    const bfloat gate = bfloat(gate_linear), up = bfloat(up_linear);
    const bfloat sigmoid = prefill_w4a8_sep21_native_bf16_sigmoid(gate);
    const bfloat silu = gate * sigmoid;
    const bfloat value = silu * up;
    if (!(activation_scale > 0.0f) || !prefill_w4a8_sep21_finite(activation_scale) ||
        !prefill_w4a8_sep21_finite(gate_linear) || !prefill_w4a8_sep21_finite(up_linear) ||
        !prefill_w4a8_sep21_finite(gate) || !prefill_w4a8_sep21_finite(up) ||
        !prefill_w4a8_sep21_finite(value)) prefill_w4a8_sep21_error(diag, 4u);
    const ulong at = ulong(row) * 640 + n;
    if constexpr (Audit) { raw_gate[at] = gate_linear; raw_up[at] = up_linear; }
    output[at] = value;
  }
}

template <ushort N, ushort SG, bool Audit>
inline void prefill_w4a8_sep21_down_tile(
    device const int8_t *input, device const float *row_scales,
    device const int32_t *group_sums, device const uchar *weights,
    device const uchar *scales, device const uchar *biases,
    device const uint *offsets, device const FlashMoEBucketJob *jobs,
    device const uint *job_count, device const uint *route_map,
    device bfloat *output, device uint *diag,
    constant PrefillW4A8DownParams &params, device float *raw_down,
    uint3 group, uint3 threads, uint tid, threadgroup uint *capacity_failures) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const constant FlashMoEBlockedDownParams &blocked = params.source;
  const constant FlashMoEDownFusedParams &p = blocked.affine;
  constexpr ushort M = kPrefillW4A8TileRows, BK = 64;
  if (!p.rows || p.rows > kFlashMoEBucketMaximumRows || !p.selections ||
      p.selections > kFlashMoEBucketMaximumSelections || p.input_size != 640 ||
      p.output_size != 2560 || p.experts != 512 || p.reserved0 || p.reserved1 ||
      p.reserved2 || blocked.reserved || params.reserved ||
      blocked.route_capacity != p.rows * p.selections || blocked.tile_rows != M ||
      blocked.job_capacity != (blocked.route_capacity + M - 1) / M + 511 ||
      params.physical_rows != blocked.route_capacity + kPrefillW4A8GuardRows ||
      params.activation_row_stride != 640 || params.sum_row_stride != 10 ||
      params.centered_row_stride_bytes != 384 || params.centered_expert_stride_bytes != 983040 ||
      threads.x != uint(SG) * 32 || threads.y != 1 || threads.z != 1 ||
      group.x >= 2560 / N || group.y >= blocked.job_capacity || group.z ||
      !prefill_w4a8_sep21_original_strides(2560, 640, p.weight_row_stride_bytes,
          p.weight_expert_stride_bytes, p.parameter_row_stride_bytes,
          p.parameter_expert_stride_bytes)) {
    if (!tid) prefill_w4a8_sep21_error(diag, 2u); return;
  }
  uint expert = 0, begin = 0, end = 0;
  if (!prefill_w4a8_sep21_job(offsets, jobs, job_count, group.y,
      blocked.job_capacity, blocked.route_capacity, expert, begin, end, diag)) return;
  const uint column_begin = group.x * N;
  auto a = tensor(const_cast<device int8_t *>(input + ulong(begin) * params.activation_row_stride),
      dextents<int, 2>{640, M}, array<int, 2>{1, int(params.activation_row_stride)});
  auto a0 = a.template slice<BK, M>(0, 0);
  const ulong weight_base = ulong(expert) * params.centered_expert_stride_bytes +
      ulong(column_begin) * params.centered_row_stride_bytes;
  tensor<device int4b_format, dextents<int, 2>, tensor_inline> b0(
      const_cast<device uchar *>(weights + weight_base), dextents<int, 2>{BK, N},
      array<int, 2>{1, int(params.centered_row_stride_bytes * 2)});
  constexpr auto descriptor = matmul2d_descriptor(M, N, BK, false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
  constexpr uint AccumulatorCapacity = uint(M) * N / (uint(SG) * 32);
  float acc[AccumulatorCapacity];
  if (prefill_w4a8_sep21_capacity_failure<SG>(dot.get_capacity() > AccumulatorCapacity,
      tid, capacity_failures)) {
    if (!tid) prefill_w4a8_sep21_error(diag, 2u); return;
  }
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i)
    if (dot.is_valid_element(i)) acc[i] = 0.0f;
  for (uint chunk = 0; chunk < 10; ++chunk) {
    auto achunk = a.template slice<BK, M>(chunk * BK, 0);
    tensor<device int4b_format, dextents<int, 2>, tensor_inline> b(
        const_cast<device uchar *>(weights + weight_base + chunk * 32),
        dextents<int, 2>{BK, N}, array<int, 2>{1, int(params.centered_row_stride_bytes * 2)});
    operation.run(achunk, b, dot);
#pragma unroll
    for (ushort i = 0; i < dot.get_capacity(); ++i) {
      if (!dot.is_valid_element(i)) continue;
      const auto index = dot.get_multidimensional_index(i);
      const uint row = begin + index[1], n = column_begin + index[0];
      if (row >= end) continue;
      const int32_t sum = group_sums[ulong(row) * params.sum_row_stride + chunk];
      const ulong parameter = ulong(expert) * p.parameter_expert_stride_bytes +
          ulong(n) * p.parameter_row_stride_bytes + chunk * 2;
      const bfloat source_scale = *reinterpret_cast<device const bfloat *>(scales + parameter);
      const bfloat source_bias = *reinterpret_cast<device const bfloat *>(biases + parameter);
      acc[i] = acc[i] + prefill_w4a8_sep21_group_term(dot[i], sum, source_scale, source_bias, diag);
    }
  }
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    const uint row = begin + index[1], n = column_begin + index[0];
    if (row >= end) continue;
    const uint route = route_map[row];
    if (route >= blocked.route_capacity) { prefill_w4a8_sep21_error(diag, 1u); continue; }
    const float activation_scale = row_scales[row];
    const float linear = acc[i] * activation_scale;
    const bfloat value = bfloat(linear);
    if (!(activation_scale > 0.0f) || !prefill_w4a8_sep21_finite(activation_scale) ||
        !prefill_w4a8_sep21_finite(linear) || !prefill_w4a8_sep21_finite(value))
      prefill_w4a8_sep21_error(diag, 4u);
    const ulong at = ulong(route) * 2560 + n;
    if constexpr (Audit) raw_down[at] = linear;
    output[at] = value;
  }
}

// This compact certificate executes the very same native M32/N/K64 integer
// matmul used by the core variants, rather than a scalar reimplementation of
// the dot. One selected cooperative cell is written per sample and G64.
template <ushort N, ushort SG>
inline void prefill_w4a8_sep21_probe_tile(
    device const int8_t *input, device const int32_t *group_sums,
    device const uchar *weights, device const uchar *scales, device const uchar *biases,
    device const uint *offsets, device const FlashMoEBucketJob *jobs,
    device const uint *job_count, device const PrefillW4A8ProbeSample *samples,
    device int32_t *raw_dot, device int32_t *raw_sum, device float *corrected_bias,
    device uint *diag, constant PrefillW4A8ProbeParams &p,
    uint3 group, uint3 threads, uint tid) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  constexpr ushort M = kPrefillW4A8TileRows, BK = 64;
  const constant FlashMoEBucketParams &b = p.bucket;
  const bool down = p.plane == 2;
  const uint k = down ? 640u : 2560u, n = down ? 2560u : 640u;
  const ulong centered_row = down ? 384ul : 1280ul;
  const ulong centered_expert = down ? 983040ul : 819200ul;
  if (!b.rows || b.rows > kFlashMoEBucketMaximumRows || !b.selections ||
      b.selections > kFlashMoEBucketMaximumSelections || b.width != k ||
      b.experts != 512 || b.routes != b.rows * b.selections || b.tile_rows != M ||
      b.job_capacity != (b.routes + M - 1) / M + 511 || b.reserved ||
      p.physical_rows != b.routes + kPrefillW4A8GuardRows ||
      p.input_row_stride != k || p.sum_row_stride != k / BK ||
      !p.sample_count || p.sample_count > 4096 || p.plane > 2 ||
      p.tile_outputs != N || p.reserved0 || p.reserved1 ||
      p.centered_row_stride_bytes != centered_row ||
      p.centered_expert_stride_bytes != centered_expert ||
      !prefill_w4a8_sep21_original_strides(n, k, k / 2, ulong(n) * k / 2,
          p.parameter_row_stride_bytes, p.parameter_expert_stride_bytes) ||
      threads.x != uint(SG) * 32 || threads.y != 1 || threads.z != 1 ||
      group.x >= p.sample_count || group.y >= k / BK || group.z) {
    if (!tid) prefill_w4a8_sep21_error(diag, 2u); return;
  }
  const PrefillW4A8ProbeSample sample = samples[group.x];
  if (sample.job_index >= b.job_capacity || sample.job_index >= job_count[0] ||
      sample.row_within_job >= M || sample.column_within_tile >= N ||
      sample.column_tile >= n / N) {
    if (!tid) prefill_w4a8_sep21_error(diag, 2u); return;
  }
  uint expert = 0, begin = 0, end = 0;
  if (!prefill_w4a8_sep21_job(offsets, jobs, job_count, sample.job_index,
      b.job_capacity, b.routes, expert, begin, end, diag)) return;
  const uint valid_rows = min(uint(M), end - begin);
  if (sample.row_within_job >= valid_rows) {
    if (!tid) prefill_w4a8_sep21_error(diag, 2u); return;
  }
  const uint column_begin = sample.column_tile * N, chunk = group.y;
  auto a = tensor(const_cast<device int8_t *>(input + ulong(begin) * p.input_row_stride + chunk * BK),
      dextents<int, 2>{BK, int(valid_rows)}, array<int, 2>{1, int(p.input_row_stride)});
  const ulong weight_at = ulong(expert) * p.centered_expert_stride_bytes +
      ulong(column_begin) * p.centered_row_stride_bytes + chunk * (BK / 2);
  tensor<device int4b_format, dextents<int, 2>, tensor_inline> weights_view(
      const_cast<device uchar *>(weights + weight_at), dextents<int, 2>{BK, N},
      array<int, 2>{1, int(p.centered_row_stride_bytes * 2)});
  constexpr auto descriptor = matmul2d_descriptor(M, N, BK, false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<decltype(a), decltype(weights_view), int32_t>();
  operation.run(a, weights_view, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    if (uint(index[1]) != sample.row_within_job ||
        uint(index[0]) != sample.column_within_tile) continue;
    const uint row = begin + sample.row_within_job;
    const uint column = column_begin + sample.column_within_tile;
    const ulong parameter = ulong(expert) * p.parameter_expert_stride_bytes +
        ulong(column) * p.parameter_row_stride_bytes + chunk * 2;
    const bfloat source_scale = *reinterpret_cast<device const bfloat *>(scales + parameter);
    const bfloat source_bias = *reinterpret_cast<device const bfloat *>(biases + parameter);
    const float bias = float(source_bias) + 8.0f * float(source_scale);
    if (!prefill_w4a8_sep21_finite(source_scale) ||
        !prefill_w4a8_sep21_finite(source_bias) || !prefill_w4a8_sep21_finite(bias))
      prefill_w4a8_sep21_error(diag, 4u);
    const ulong at = ulong(group.x) * (k / BK) + chunk;
    raw_dot[at] = dot[i];
    raw_sum[at] = group_sums[ulong(row) * p.sum_row_stride + chunk];
    corrected_bias[at] = bias;
  }
}

#define PREFILL_W4A8_POSITION \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]
#define PREFILL_W4A8_GATE_COMMON \
    device const int8_t *a [[buffer(0)]], device const float *row_scales [[buffer(1)]], \
    device const int32_t *sums [[buffer(2)]], device const uchar *gw [[buffer(3)]], \
    device const uchar *gs [[buffer(4)]], device const uchar *gb [[buffer(5)]], \
    device const uchar *uw [[buffer(6)]], device const uchar *us [[buffer(7)]], \
    device const uchar *ub [[buffer(8)]], device const uint *offsets [[buffer(9)]], \
    device const FlashMoEBucketJob *jobs [[buffer(10)]], device const uint *count [[buffer(11)]], \
    device bfloat *output [[buffer(12)]], device uint *diag [[buffer(13)]], \
    constant PrefillW4A8GateParams &p [[buffer(14)]]
#define PREFILL_W4A8_DOWN_COMMON \
    device const int8_t *a [[buffer(0)]], device const float *row_scales [[buffer(1)]], \
    device const int32_t *sums [[buffer(2)]], device const uchar *w [[buffer(3)]], \
    device const uchar *s [[buffer(4)]], device const uchar *b [[buffer(5)]], \
    device const uint *offsets [[buffer(6)]], device const FlashMoEBucketJob *jobs [[buffer(7)]], \
    device const uint *count [[buffer(8)]], device const uint *map [[buffer(9)]], \
    device bfloat *output [[buffer(10)]], device uint *diag [[buffer(11)]], \
    constant PrefillW4A8DownParams &p [[buffer(12)]]
#define PREFILL_W4A8_ENTRIES(N, SG) \
kernel void prefill_w4a8_sep21_gate_m32_n##N##_sg##SG(PREFILL_W4A8_GATE_COMMON, PREFILL_W4A8_POSITION) { \
  threadgroup uint capacity_failures[SG]; \
  prefill_w4a8_sep21_gate_tile<N, SG, false>(a, row_scales, sums, gw, gs, gb, uw, us, ub, \
      offsets, jobs, count, output, diag, p, nullptr, nullptr, group, threads, tid, capacity_failures); \
} \
kernel void prefill_w4a8_sep21_gate_m32_n##N##_sg##SG##_audit(PREFILL_W4A8_GATE_COMMON, \
    device float *raw_gate [[buffer(15)]], device float *raw_up [[buffer(16)]], PREFILL_W4A8_POSITION) { \
  threadgroup uint capacity_failures[SG]; \
  prefill_w4a8_sep21_gate_tile<N, SG, true>(a, row_scales, sums, gw, gs, gb, uw, us, ub, \
      offsets, jobs, count, output, diag, p, raw_gate, raw_up, group, threads, tid, capacity_failures); \
} \
kernel void prefill_w4a8_sep21_down_m32_n##N##_sg##SG(PREFILL_W4A8_DOWN_COMMON, PREFILL_W4A8_POSITION) { \
  threadgroup uint capacity_failures[SG]; \
  prefill_w4a8_sep21_down_tile<N, SG, false>(a, row_scales, sums, w, s, b, offsets, jobs, count, \
      map, output, diag, p, nullptr, group, threads, tid, capacity_failures); \
} \
kernel void prefill_w4a8_sep21_down_m32_n##N##_sg##SG##_audit(PREFILL_W4A8_DOWN_COMMON, \
    device float *raw_down [[buffer(13)]], PREFILL_W4A8_POSITION) { \
  threadgroup uint capacity_failures[SG]; \
  prefill_w4a8_sep21_down_tile<N, SG, true>(a, row_scales, sums, w, s, b, offsets, jobs, count, \
      map, output, diag, p, raw_down, group, threads, tid, capacity_failures); \
} \
kernel void prefill_w4a8_sep21_probe_m32_n##N##_sg##SG( \
    device const int8_t *a [[buffer(0)]], device const int32_t *sums [[buffer(1)]], \
    device const uchar *w [[buffer(2)]], device const uchar *s [[buffer(3)]], \
    device const uchar *b [[buffer(4)]], device const uint *offsets [[buffer(5)]], \
    device const FlashMoEBucketJob *jobs [[buffer(6)]], device const uint *count [[buffer(7)]], \
    device const PrefillW4A8ProbeSample *samples [[buffer(8)]], \
    device int32_t *raw_dot [[buffer(9)]], device int32_t *raw_sum [[buffer(10)]], \
    device float *bias [[buffer(11)]], device uint *diag [[buffer(12)]], \
    constant PrefillW4A8ProbeParams &p [[buffer(13)]], PREFILL_W4A8_POSITION) { \
  prefill_w4a8_sep21_probe_tile<N, SG>(a, sums, w, s, b, offsets, jobs, count, samples, \
      raw_dot, raw_sum, bias, diag, p, group, threads, tid); \
}
PREFILL_W4A8_ENTRIES(32, 1)
PREFILL_W4A8_ENTRIES(32, 2)
PREFILL_W4A8_ENTRIES(64, 1)
PREFILL_W4A8_ENTRIES(64, 2)
#undef PREFILL_W4A8_POSITION
#undef PREFILL_W4A8_GATE_COMMON
#undef PREFILL_W4A8_DOWN_COMMON
#undef PREFILL_W4A8_ENTRIES
#endif
