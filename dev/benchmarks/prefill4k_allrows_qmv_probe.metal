// Private untimed numeric taps for paired gathered-QMV and saved-I8 MPP dots.
// No activation is fused here: raw F32, late-scaled F32 and BF16 are separate.
#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "prefill4k_allrows_qmv_probe.h"
#include "metal/abi/FlashMoEBuckets.h"

#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

inline bool flash_qmv_probe_finite(float value) {
  return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}
inline void flash_qmv_probe_error(device uint *diag, uint bits) {
  atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(diag),
                          bits, memory_order_relaxed);
}
inline float flash_qmv_probe_nan() { return as_type<float>(0x7fc00000u); }
inline bool flash_qmv_probe_phase(constant FlashQMVProbeParams &p) {
  return p.rows && p.rows <= 16 && p.selections == 10 && p.experts == 512 &&
      !p.reserved &&
      ((p.input_size == 2560 && p.output_size == 640 && !p.per_route_input) ||
       (p.input_size == 640 && p.output_size == 2560 && p.per_route_input == 1));
}
inline float flash_qmv_probe_activation(bfloat value, device uint *diag) {
  const float result = float(value);
  if (!flash_qmv_probe_finite(result)) {
    flash_qmv_probe_error(diag, 4u);
    return 0.0f;
  }
  return result;
}
inline void flash_qmv_probe_write_nan(device float *dot, device float *scaled,
    device bfloat *projection, ulong index) {
  const float value = flash_qmv_probe_nan();
  dot[index] = value;
  scaled[index] = value;
  projection[index] = bfloat(value);
}
inline void flash_qmv_probe_write(float raw, float scale, device float *dot,
    device float *scaled, device bfloat *projection, device uint *diag, ulong index) {
  const float result = raw * scale;
  const bfloat rounded = bfloat(result);
  if (!(scale > 0.0f) || !flash_qmv_probe_finite(scale) ||
      !flash_qmv_probe_finite(raw) || !flash_qmv_probe_finite(result) ||
      !flash_qmv_probe_finite(float(rounded))) flash_qmv_probe_error(diag, 4u);
  dot[index] = raw;
  scaled[index] = result;
  projection[index] = rounded;
}

template <ushort Columns>
inline void flash_qmv_probe_gathered(device const bfloat *input,
    device const int8_t *codes, device const float *scales,
    device const uint *ranks, device const long *ids, device float *dot,
    device float *scaled, device bfloat *projection, device uint *diag,
    constant FlashQMVProbeParams &p, uint3 group, uint3 threads,
    uint simd, uint lane) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  static_assert(Columns == 1 || Columns == 2);
  if (!flash_qmv_probe_phase(p) || p.columns != Columns ||
      threads.x != 128 || threads.y != 1 || threads.z != 1 ||
      group.x >= p.output_size / (4 * Columns) || group.y >= p.rows ||
      group.z >= p.selections) {
    if (!lane) flash_qmv_probe_error(diag, 2u);
    return;
  }
  const ulong route = ulong(group.y) * p.selections + group.z;
  const uint nbase = group.x * (4 * Columns) + simd * Columns;
  const long expert = ids[route];
  uint rank = UINT_MAX;
  if (expert >= 0 && expert < long(p.experts)) {
    const ulong begin = ulong(group.y) * p.selections;
    for (uint slot = 0; slot < p.selections; ++slot)
      if (begin + slot != route && ids[begin + slot] == expert && !lane)
        flash_qmv_probe_error(diag, 1u);
    rank = ranks[uint(expert)];
    if (rank >= p.experts) rank = UINT_MAX;
  }
  const ulong coefficient = rank == UINT_MAX ? 0 :
      (ulong(rank) * p.output_size + nbase) * p.input_size;
  const ulong input_row = p.per_route_input ? route : ulong(group.y);
  float local[Columns];
#pragma unroll
  for (ushort c = 0; c < Columns; ++c) local[c] = 0.0f;
  for (uint k = lane; k < p.input_size; k += 32) {
    const float x = flash_qmv_probe_activation(input[input_row * p.input_size + k], diag);
    if (rank != UINT_MAX) {
#pragma unroll
      for (ushort c = 0; c < Columns; ++c)
        local[c] += float(codes[coefficient + ulong(c) * p.input_size + k]) * x;
    }
  }
  // Hidden rows are inspected even if all original IDs are invalid.
  if (rank == UINT_MAX) {
    if (!lane) {
      flash_qmv_probe_error(diag, 5u);
#pragma unroll
      for (ushort c = 0; c < Columns; ++c)
        flash_qmv_probe_write_nan(dot, scaled, projection,
                                 route * p.output_size + nbase + c);
    }
    return;
  }
#pragma unroll
  for (ushort c = 0; c < Columns; ++c) {
    const float raw = simd_sum(local[c]);
    if (!lane) {
      const uint n = nbase + c;
      const float scale = scales[ulong(rank) * p.output_size + n];
      flash_qmv_probe_write(raw, scale, dot, scaled, projection, diag,
                           route * p.output_size + n);
    }
  }
}

#define FLASH_QMV_PROBE(NAME, COLUMNS) \
kernel void NAME(device const bfloat *input [[buffer(0)]], \
    device const int8_t *codes [[buffer(1)]], device const float *scales [[buffer(2)]], \
    device const uint *ranks [[buffer(3)]], device const long *ids [[buffer(4)]], \
    device float *dot [[buffer(5)]], device float *scaled [[buffer(6)]], \
    device bfloat *projection [[buffer(7)]], device uint *diag [[buffer(8)]], \
    constant FlashQMVProbeParams &p [[buffer(9)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  flash_qmv_probe_gathered<COLUMNS>(input, codes, scales, ranks, ids, dot, scaled, \
                                  projection, diag, p, group, threads, simd, lane); \
}
FLASH_QMV_PROBE(flash_qmv_probe_c1, 1)
FLASH_QMV_PROBE(flash_qmv_probe_c2, 2)
#undef FLASH_QMV_PROBE

kernel void flash_qmv_probe_mpp_m16(
    device bfloat *input [[buffer(0)]], device int8_t *codes [[buffer(1)]],
    device const float *scales [[buffer(2)]], device const uint *ranks [[buffer(3)]],
    device const uint *offsets [[buffer(4)]],
    device const FlashMoEBucketJob *jobs [[buffer(5)]],
    device const uint *job_count [[buffer(6)]], device const uint *route_map [[buffer(7)]],
    device float *dot [[buffer(8)]], device float *scaled [[buffer(9)]],
    device bfloat *projection [[buffer(10)]], device uint *diag [[buffer(11)]],
    constant FlashQMVProbeParams &p [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint routes = p.rows * p.selections;
  if (!flash_qmv_probe_phase(p) || p.columns ||
      threads.x != 128 || threads.y != 1 || threads.z != 1 ||
      group.x >= p.output_size / 64 || group.y >= routes || group.z) {
    if (!tid) flash_qmv_probe_error(diag, 2u);
    return;
  }
  const uint active = job_count[0], total = offsets[512];
  if (active > routes || total > routes || offsets[0] ||
      (!active && total) || (active && !total)) {
    if (!tid) flash_qmv_probe_error(diag, 2u);
    return;
  }
  constexpr ushort M = 16, N = 64, SG = 4;
  const uint column = group.x * N;
  // Every canonical route owns a separate exclusion check. Missing routes can
  // be poisoned without racing any valid packed-row scatter in another group.
  bool included = false;
  for (uint row = 0; row < total; ++row) {
    const uint route = route_map[row];
    if (route >= routes) {
      if (!tid) flash_qmv_probe_error(diag, 2u);
    } else if (route == group.y) included = true;
  }
  if (!included && tid < N) {
    if (!tid) flash_qmv_probe_error(diag, 5u);
    flash_qmv_probe_write_nan(dot, scaled, projection,
                             ulong(group.y) * p.output_size + column + tid);
  }
  if (group.y >= active) return;
  const FlashMoEBucketJob job = jobs[group.y];
  if (job.expert >= 512) {
    if (!tid) flash_qmv_probe_error(diag, 5u);
    return;
  }
  const uint first = offsets[job.expert], end = offsets[job.expert + 1];
  if (first > end || end > total || job.row_begin < first || job.row_begin >= end) {
    if (!tid) flash_qmv_probe_error(diag, 2u);
    return;
  }
  const uint begin = job.row_begin, valid_rows = min(uint(M), end - begin);
  const uint rank = ranks[job.expert];
  if (rank >= 512) {
    if (!tid) flash_qmv_probe_error(diag, 5u);
    for (uint i = tid; i < valid_rows * N; i += 128) {
      const uint route = route_map[begin + i / N];
      if (route >= routes) flash_qmv_probe_error(diag, 2u);
      else flash_qmv_probe_write_nan(dot, scaled, projection,
                                    ulong(route) * p.output_size + column + i % N);
    }
    return;
  }
  // Identical whole-K dynamic descriptor and dynamic tail-row bounds to the
  // shipping saved-I8 M16 producer. The caller provides its sanitized operand.
  auto a = tensor(input + ulong(begin) * p.input_size,
      dextents<int, 2>{int(p.input_size), int(valid_rows)},
      array<int, 2>{1, int(p.input_size)});
  auto b = tensor(codes + (ulong(rank) * p.output_size + column) * p.input_size,
      dextents<int, 2>{int(p.input_size), N}, array<int, 2>{1, int(p.input_size)});
  constexpr auto descriptor = matmul2d_descriptor(M, N, static_cast<int>(dynamic_extent),
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto raw = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  operation.run(a, b, raw);
#pragma unroll
  for (ushort i = 0; i < raw.get_capacity(); ++i) {
    if (!raw.is_valid_element(i)) continue;
    const auto index = raw.get_multidimensional_index(i);
    if (uint(index[1]) >= valid_rows) continue;
    const uint route = route_map[begin + index[1]], n = column + index[0];
    if (route >= routes) { flash_qmv_probe_error(diag, 2u); continue; }
    flash_qmv_probe_write(raw[i], scales[ulong(rank) * p.output_size + n],
                         dot, scaled, projection, diag, ulong(route) * p.output_size + n);
  }
}

kernel void flash_qmv_probe_unpack_activation(
    device const ushort *packed [[buffer(0)]], device const uint *inverse [[buffer(1)]],
    device ushort *canonical [[buffer(2)]], device uint *diag [[buffer(3)]],
    constant FlashQMVProbeParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint routes = p.rows * p.selections;
  if (!flash_qmv_probe_phase(p) || p.input_size != 2560 || p.output_size != 640 ||
      p.per_route_input || p.columns || threads.x != 256 || threads.y != 1 ||
      threads.z != 1 || group.x >= 3 || group.y >= routes || group.z) {
    if (!tid) flash_qmv_probe_error(diag, 2u);
    return;
  }
  const uint column = group.x * 256 + tid;
  if (column >= 640) return;
  const uint row = inverse[group.y];
  const ulong destination = ulong(group.y) * 640 + column;
  if (row >= routes) {
    if (!tid) flash_qmv_probe_error(diag, 5u);
    canonical[destination] = as_type<ushort>(bfloat(flash_qmv_probe_nan()));
  } else canonical[destination] = packed[ulong(row) * 640 + column];
}
#endif
