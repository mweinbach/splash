// Private prefill experiment: lossless signed-I8-to-BF16 B in registers,
// device BF16 A, F32 accumulation, then original scale/SwiGLU/output boundaries.
// One SIMD group owns one M32N64 job via sequential N32 tiles. Since only B
// is cooperative, Metal 4.1 permits a real K64/K128/K256 MPP reduction operation;
// register/register's K32 restriction does not apply. No payload is changed.
// K256 down uses a final K128 device-A slice and zero-padded cooperative B;
// the fixed K256 descriptor bounds A by its remaining logical K extent.
#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashInt8ExpertStore.h"
#include "metal/abi/FlashMoEBuckets.h"
#include "metal/kernels/common/flash_affine_mpp_common.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

// Preserve the existing compiled helper and every BF16 SwiGLU boundary.
#pragma METAL fp math_mode(fast)
inline bfloat prefill_moe_sep21_right_only_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

template <ushort M>
inline bool prefill_moe_sep21_right_only_job(
    constant FlashInt8ExpertStoreParams &p, device const uint *ranks,
    device const uint *offsets, device const FlashMoEBucketJob *jobs,
    device const uint *job_count, device uint *diag, uint3 group,
    uint3 threads, uint tid, thread uint &rank, thread uint &begin,
    thread uint &valid_rows) {
  if (!p.rows || p.rows > kFlashMoEBucketMaximumRows || !p.selections ||
      p.selections > kFlashMoEBucketMaximumSelections ||
      p.route_capacity != p.rows * p.selections ||
      p.job_capacity != (p.route_capacity + M - 1) / M + 511 ||
      p.tile_rows != M || !p.stored_experts || p.stored_experts > 512 ||
      p.scale_group_size || p.reserved || group.y >= p.job_capacity || group.z ||
      threads.x != 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) flash_mpp_error(diag, 2u); return false;
  }
  const uint active = job_count[0];
  if (active > p.job_capacity || offsets[512] > p.route_capacity) {
    if (!tid) flash_mpp_error(diag, 2u); return false;
  }
  if (group.y >= active) return false;
  const auto job = jobs[group.y];
  if (job.expert >= 512) {
    if (!tid) flash_mpp_error(diag, 1u); return false;
  }
  const uint first = offsets[job.expert], end = offsets[job.expert + 1];
  if (first > end || end > p.route_capacity || job.row_begin < first ||
      job.row_begin >= end) {
    if (!tid) flash_mpp_error(diag, 2u); return false;
  }
  rank = ranks[job.expert];
  if (rank == UINT_MAX) return false;
  if (rank >= p.stored_experts) {
    if (!tid) flash_mpp_error(diag, 1u); return false;
  }
  begin = job.row_begin;
  valid_rows = min(uint(M), end - begin);
  return true;
}

template <ushort BK, ushort PartM = 32>
inline void prefill_moe_sep21_right_only_gate(
    device const bfloat *input, device const int8_t *gate,
    device const float *gate_scale, device const int8_t *up,
    device const float *up_scale, device const uint *ranks,
    device const uint *offsets, device const FlashMoEBucketJob *jobs,
    device const uint *job_count, device bfloat *output, device uint *diag,
    constant FlashInt8ExpertStoreParams &p, uint3 group, uint3 threads,
    uint tid) {
  if (group.x >= 10) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  constexpr ushort M = 32, N = 64;
  if (!prefill_moe_sep21_right_only_job<M>(p, ranks, offsets, jobs, job_count,
      diag, group, threads, tid, rank, begin, valid_rows)) return;
  const uint column = group.x * N;
  constexpr auto descriptor = matmul2d_descriptor(PartM, 32, BK, false, true,
      false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> operation;
  using Operand = tensor<device bfloat, dextents<int, 2>>;
#pragma unroll
  for (ushort row_tile = 0; row_tile < M / PartM; ++row_tile) {
    const uint row_base = row_tile * PartM;
    if (row_base >= valid_rows) continue;
#pragma unroll
  for (ushort col_tile = 0; col_tile < 2; ++col_tile) {
  auto b = operation.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto gd = operation.template get_destination_cooperative_tensor<Operand, Operand, float>();
  auto ud = operation.template get_destination_cooperative_tensor<Operand, Operand, float>();
#pragma unroll
  for (ushort i = 0; i < gd.get_capacity(); ++i)
    if (gd.is_valid_element(i)) {
      gd[i] = 0.0f; ud[i] = 0.0f;
    }
  for (uint chunk = 0; chunk < (2560 + BK - 1) / BK; ++chunk) {
    const uint korigin = chunk * BK;
    auto a = tensor(const_cast<device bfloat *>(input +
        ulong(begin + row_base) * 2560 + korigin),
        dextents<int, 2>{int(min(uint(BK), 2560u - korigin)), int(min(uint(PartM), valid_rows - row_base))},
        array<int, 2>{1, 2560});
#pragma unroll
    for (ushort i = 0; i < b.get_capacity(); ++i) {
      if (!b.is_valid_element(i)) continue;
      const auto index = b.get_multidimensional_index(i);
      const uint n = column + col_tile * 32 + index[1], k = korigin + index[0];
      b[i] = bfloat(gate[(ulong(rank) * 640 + n) * 2560 + k]);
    }
    operation.run(a, b, gd);
#pragma unroll
    for (ushort i = 0; i < b.get_capacity(); ++i) {
      if (!b.is_valid_element(i)) continue;
      const auto index = b.get_multidimensional_index(i);
      const uint n = column + col_tile * 32 + index[1], k = korigin + index[0];
      b[i] = bfloat(up[(ulong(rank) * 640 + n) * 2560 + k]);
    }
    operation.run(a, b, ud);
  }
#pragma unroll
  for (ushort i = 0; i < gd.get_capacity(); ++i) {
    if (!gd.is_valid_element(i)) continue;
    const auto index = gd.get_multidimensional_index(i);
    const uint m = row_base + index[1];
    if (m >= valid_rows) continue;
    const uint n = column + col_tile * 32 + index[0];
    const float gs = gate_scale[ulong(rank) * 640 + n];
    const float us = up_scale[ulong(rank) * 640 + n];
    const float gf = gd[i] * gs;
    const float uf = ud[i] * us;
    const bfloat gv = bfloat(gf), uv = bfloat(uf);
    const bfloat silu = gv * prefill_moe_sep21_right_only_sigmoid(gv);
    const bfloat value = silu * uv;
    if (!(gs > 0.0f) || !(us > 0.0f) || !flash_mpp_finite(gs) ||
        !flash_mpp_finite(us) || !flash_mpp_finite(gf) || !flash_mpp_finite(uf) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(begin + m) * 640 + n] = value;
  }
  }
  }
}

template <ushort BK, ushort PartM = 32>
inline void prefill_moe_sep21_right_only_down(
    device const bfloat *input, device const int8_t *weights,
    device const float *scales, device const uint *ranks,
    device const uint *offsets, device const FlashMoEBucketJob *jobs,
    device const uint *job_count, device const uint *route_map,
    device bfloat *output, device uint *diag,
    constant FlashInt8ExpertStoreParams &p, uint3 group, uint3 threads,
    uint tid) {
  if (group.x >= 40) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  constexpr ushort M = 32, N = 64;
  if (!prefill_moe_sep21_right_only_job<M>(p, ranks, offsets, jobs, job_count,
      diag, group, threads, tid, rank, begin, valid_rows)) return;
  const uint column = group.x * N;
  constexpr auto descriptor = matmul2d_descriptor(PartM, 32, BK, false, true,
      false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> operation;
  using Operand = tensor<device bfloat, dextents<int, 2>>;
#pragma unroll
  for (ushort row_tile = 0; row_tile < M / PartM; ++row_tile) {
    const uint row_base = row_tile * PartM;
    if (row_base >= valid_rows) continue;
#pragma unroll
  for (ushort col_tile = 0; col_tile < 2; ++col_tile) {
  auto b = operation.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto dot = operation.template get_destination_cooperative_tensor<Operand, Operand, float>();
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i)
    if (dot.is_valid_element(i)) dot[i] = 0.0f;
  for (uint chunk = 0; chunk < (640 + BK - 1) / BK; ++chunk) {
    const uint korigin = chunk * BK;
    auto a = tensor(const_cast<device bfloat *>(input +
        ulong(begin + row_base) * 640 + korigin),
        dextents<int, 2>{int(min(uint(BK), 640u - korigin)), int(min(uint(PartM), valid_rows - row_base))},
        array<int, 2>{1, 640});
#pragma unroll
    for (ushort i = 0; i < b.get_capacity(); ++i) {
      if (!b.is_valid_element(i)) continue;
      const auto index = b.get_multidimensional_index(i);
      const uint n = column + col_tile * 32 + index[1], k = korigin + index[0];
      b[i] = k < 640 ? bfloat(weights[(ulong(rank) * 2560 + n) * 640 + k]) : bfloat(0.0f);
    }
    operation.run(a, b, dot);
  }
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    const uint m = row_base + index[1];
    if (m >= valid_rows) continue;
    const uint route = route_map[begin + m], n = column + col_tile * 32 + index[0];
    if (route >= p.route_capacity) { flash_mpp_error(diag, 1u); continue; }
    const float scale = scales[ulong(rank) * 2560 + n];
    const float result = dot[i] * scale;
    const bfloat value = bfloat(result);
    if (!(scale > 0.0f) || !flash_mpp_finite(scale) || !flash_mpp_finite(result) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(route) * 2560 + n] = value;
  }
  }
  }
}

#define RIGHT_ONLY_GATE(NAME, BK, PARTM) \
kernel void NAME(device const bfloat *a [[buffer(0)]], device const int8_t *g [[buffer(1)]], \
    device const float *gs [[buffer(2)]], device const int8_t *u [[buffer(3)]], \
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]], \
    device const uint *offsets [[buffer(6)]], device const FlashMoEBucketJob *jobs [[buffer(7)]], \
    device const uint *count [[buffer(8)]], device bfloat *out [[buffer(9)]], \
    device uint *diag [[buffer(10)]], constant FlashInt8ExpertStoreParams &p [[buffer(11)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  prefill_moe_sep21_right_only_gate<BK, PARTM>(a, g, gs, u, us, ranks, offsets, jobs, count, out, diag, p, group, threads, tid); \
}
#define RIGHT_ONLY_DOWN(NAME, BK, PARTM) \
kernel void NAME(device const bfloat *a [[buffer(0)]], device const int8_t *w [[buffer(1)]], \
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]], \
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]], \
    device const uint *count [[buffer(6)]], device const uint *map [[buffer(7)]], \
    device bfloat *out [[buffer(8)]], device uint *diag [[buffer(9)]], \
    constant FlashInt8ExpertStoreParams &p [[buffer(10)]], uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { \
  prefill_moe_sep21_right_only_down<BK, PARTM>(a, w, s, ranks, offsets, jobs, count, map, out, diag, p, group, threads, tid); \
}
RIGHT_ONLY_GATE(prefill_moe_sep21_right_only_gate_up_m32_n64_k64_sg1, 64, 32)
RIGHT_ONLY_GATE(prefill_moe_sep21_right_only_gate_up_m32_n64_k128_sg1, 128, 32)
RIGHT_ONLY_GATE(prefill_moe_sep21_right_only_gate_up_m32_n64_k256_sg1, 256, 32)
RIGHT_ONLY_DOWN(prefill_moe_sep21_right_only_down_scatter_m32_n64_k64_sg1, 64, 32)
RIGHT_ONLY_DOWN(prefill_moe_sep21_right_only_down_scatter_m32_n64_k128_sg1, 128, 32)
RIGHT_ONLY_DOWN(prefill_moe_sep21_right_only_down_scatter_m32_n64_k256_sg1, 256, 32)
#undef RIGHT_ONLY_GATE
#undef RIGHT_ONLY_DOWN
#endif
