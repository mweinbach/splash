// Private experiment: unchanged BF16 A and persistent BF16 B containing
// exact UNSCALED signed-I8 codes. The runtime oracle converts each original
// persisted I8 code exactly, without incorporating its F32 scale. This is
// distinct from the old BF16 cache of already-scaled source-Q4 coefficients.
// F32 post-dot row scales, BF16 projections/SwiGLU and output scatter retain
// their original boundaries. B tensors retain the original K-major logical
// shape and strides, now expressed in BF16 elements. No Q4 misses or source
// mutation. Whole-K variants use F32 multiply; fixed K128 accumulates F32.
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

#pragma METAL fp math_mode(fast)
inline bfloat prefill_moe_sep21_bf16_codes_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

template <ushort M, ushort SG, ushort K = 0, bool Static = false>
inline bool prefill_moe_sep21_bf16_codes_job(constant FlashInt8ExpertStoreParams &p,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device uint *diag, uint3 group, uint3 threads, uint tid,
    thread uint &rank, thread uint &begin, thread uint &valid_rows) {
  if (!p.rows || p.rows > kFlashMoEBucketMaximumRows || !p.selections ||
      p.selections > kFlashMoEBucketMaximumSelections ||
      p.route_capacity != p.rows * p.selections ||
      p.job_capacity != (p.route_capacity + M - 1) / M + 511 ||
      p.tile_rows != M || !p.stored_experts || p.stored_experts > 512 ||
      p.scale_group_size || p.reserved || group.y >= p.job_capacity || group.z ||
      threads.x != uint(SG) * 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) flash_mpp_error(diag, 2u); return false;
  }
  const uint active = job_count[0];
  if (active > p.job_capacity || offsets[512] > p.route_capacity) {
    if (!tid) flash_mpp_error(diag, 2u); return false;
  }
  if (group.y >= active) return false;
  const auto job = jobs[group.y];
  if (job.expert >= 512) { if (!tid) flash_mpp_error(diag, 1u); return false; }
  const uint first = offsets[job.expert], end = offsets[job.expert + 1];
  if (first > end || end > p.route_capacity || job.row_begin < first || job.row_begin >= end) {
    if (!tid) flash_mpp_error(diag, 2u); return false;
  }
  rank = ranks[job.expert];
  if (rank == UINT_MAX) return false;
  if (rank >= p.stored_experts) { if (!tid) flash_mpp_error(diag, 1u); return false; }
  begin = job.row_begin;
  valid_rows = min(uint(M), end - begin);
  return true;
}

template <ushort M, ushort SG, ushort K = 0, bool Static = false>
inline void prefill_moe_sep21_bf16_codes_gate(device bfloat *input, device bfloat *gate,
    device const float *gate_scale, device bfloat *up, device const float *up_scale,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device bfloat *output, device uint *diag, constant FlashInt8ExpertStoreParams &p,
    uint3 group, uint3 threads, uint tid) {
  if (group.x >= 10) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!prefill_moe_sep21_bf16_codes_job<M, SG>(p, ranks, offsets, jobs, job_count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  constexpr ushort N = 64;
  const uint column = group.x * N;
  // Dynamic row bounds make incomplete bucket tiles safe without staging,
  // copying or reading the next expert's input. MPP masks the tail rows.
  auto a = tensor(input + ulong(begin) * 2560,
      dextents<int, 2>{2560, int(valid_rows)}, array<int, 2>{1, 2560});
  auto g = tensor(gate + (ulong(rank) * 640 + column) * 2560,
      dextents<int, 2>{2560, N}, array<int, 2>{1, 2560});
  auto u = tensor(up + (ulong(rank) * 640 + column) * 2560,
      dextents<int, 2>{2560, N}, array<int, 2>{1, 2560});
  constexpr auto descriptor = matmul2d_descriptor(M, N, K ? int(K) : static_cast<int>(dynamic_extent),
      false, true, false, K ? matmul2d_descriptor::mode::multiply_accumulate : matmul2d_descriptor::mode::multiply);
  using Scope = conditional_t<SG == 1, execution_simdgroup, execution_simdgroups<SG>>;
  matmul2d<descriptor, Scope> operation;
  auto gd = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  auto ud = operation.template get_destination_cooperative_tensor<decltype(a), decltype(u), float>();
  if constexpr (K) {
#pragma unroll
    for (ushort i = 0; i < gd.get_capacity(); ++i) {
      if (gd.is_valid_element(i)) { gd[i] = 0.0f; ud[i] = 0.0f; }
    }
    for (uint k = 0; k < 2560; k += K) {
      if constexpr (Static) {
        if (valid_rows == M) {
          auto aa = tensor(input + ulong(begin) * 2560 + k, extents<int, K, M>{}, array<int, 2>{1, 2560});
          auto gg = tensor(gate + (ulong(rank) * 640 + column) * 2560 + k, extents<int, K, N>{}, array<int, 2>{1, 2560});
          auto uu = tensor(up + (ulong(rank) * 640 + column) * 2560 + k, extents<int, K, N>{}, array<int, 2>{1, 2560});
          operation.run(aa, gg, gd); operation.run(aa, uu, ud);
        } else {
          auto aa = a.slice(k, 0); auto gg = g.slice(k, 0); auto uu = u.slice(k, 0);
          operation.run(aa, gg, gd); operation.run(aa, uu, ud);
        }
      } else {
        auto aa = a.slice(k, 0); auto gg = g.slice(k, 0); auto uu = u.slice(k, 0);
        operation.run(aa, gg, gd); operation.run(aa, uu, ud);
      }
    }
  } else if constexpr (Static) {
    if (valid_rows == M) {
      auto aa = tensor(input + ulong(begin) * 2560, extents<int, 2560, M>{}, array<int, 2>{1, 2560});
      auto gg = tensor(gate + (ulong(rank) * 640 + column) * 2560, extents<int, 2560, N>{}, array<int, 2>{1, 2560});
      auto uu = tensor(up + (ulong(rank) * 640 + column) * 2560, extents<int, 2560, N>{}, array<int, 2>{1, 2560});
      operation.run(aa, gg, gd); operation.run(aa, uu, ud);
    } else { operation.run(a, g, gd); operation.run(a, u, ud); }
  } else { operation.run(a, g, gd); operation.run(a, u, ud); }
#pragma unroll
  for (ushort i = 0; i < gd.get_capacity(); ++i) {
    if (!gd.is_valid_element(i)) continue;
    const auto index = gd.get_multidimensional_index(i);
    if (uint(index[1]) >= valid_rows) continue;
    const uint n = column + index[0];
    const float gs = gate_scale[ulong(rank) * 640 + n];
    const float us = up_scale[ulong(rank) * 640 + n];
    const float gf = gd[i] * gs, uf = ud[i] * us;
    const bfloat gv = bfloat(gf), uv = bfloat(uf);
    const bfloat silu = gv * prefill_moe_sep21_bf16_codes_sigmoid(gv);
    const bfloat value = silu * uv;
    if (!(gs > 0.0f) || !(us > 0.0f) || !flash_mpp_finite(gs) ||
        !flash_mpp_finite(us) || !flash_mpp_finite(gf) || !flash_mpp_finite(uf) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(begin + index[1]) * 640 + n] = value;
  }
}

template <ushort M, ushort SG, ushort K = 0, bool Static = false>
inline void prefill_moe_sep21_bf16_codes_down(device bfloat *input, device bfloat *weights,
    device const float *scales, device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device const uint *route_map, device bfloat *output, device uint *diag,
    constant FlashInt8ExpertStoreParams &p, uint3 group, uint3 threads, uint tid) {
  if (group.x >= 40) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!prefill_moe_sep21_bf16_codes_job<M, SG>(p, ranks, offsets, jobs, job_count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  constexpr ushort N = 64;
  const uint column = group.x * N;
  auto a = tensor(input + ulong(begin) * 640,
      dextents<int, 2>{640, int(valid_rows)}, array<int, 2>{1, 640});
  auto b = tensor(weights + (ulong(rank) * 2560 + column) * 640,
      dextents<int, 2>{640, N}, array<int, 2>{1, 640});
  constexpr auto descriptor = matmul2d_descriptor(M, N, K ? int(K) : static_cast<int>(dynamic_extent),
      false, true, false, K ? matmul2d_descriptor::mode::multiply_accumulate : matmul2d_descriptor::mode::multiply);
  using Scope = conditional_t<SG == 1, execution_simdgroup, execution_simdgroups<SG>>;
  matmul2d<descriptor, Scope> operation;
  auto dot = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  if constexpr (K) {
#pragma unroll
    for (ushort i = 0; i < dot.get_capacity(); ++i)
      if (dot.is_valid_element(i)) dot[i] = 0.0f;
    for (uint k = 0; k < 640; k += K) {
      if constexpr (Static) {
        if (valid_rows == M) {
          auto aa = tensor(input + ulong(begin) * 640 + k, extents<int, K, M>{}, array<int, 2>{1, 640});
          auto bb = tensor(weights + (ulong(rank) * 2560 + column) * 640 + k, extents<int, K, N>{}, array<int, 2>{1, 640});
          operation.run(aa, bb, dot);
        } else {
          auto aa = a.slice(k, 0); auto bb = b.slice(k, 0); operation.run(aa, bb, dot);
        }
      } else {
        auto aa = a.slice(k, 0); auto bb = b.slice(k, 0); operation.run(aa, bb, dot);
      }
    }
  } else if constexpr (Static) {
    if (valid_rows == M) {
      auto aa = tensor(input + ulong(begin) * 640, extents<int, 640, M>{}, array<int, 2>{1, 640});
      auto bb = tensor(weights + (ulong(rank) * 2560 + column) * 640, extents<int, 640, N>{}, array<int, 2>{1, 640});
      operation.run(aa, bb, dot);
    } else { operation.run(a, b, dot); }
  } else { operation.run(a, b, dot); }
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    if (uint(index[1]) >= valid_rows) continue;
    const uint route = route_map[begin + index[1]], n = column + index[0];
    if (route >= p.route_capacity) { flash_mpp_error(diag, 1u); continue; }
    const float scale = scales[ulong(rank) * 2560 + n];
    const float result = dot[i] * scale;
    const bfloat value = bfloat(result);
    if (!(scale > 0.0f) || !flash_mpp_finite(scale) || !flash_mpp_finite(result) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(route) * 2560 + n] = value;
  }
}

#define BF16_CODES_GATE(NAME, M, SG, K, STATIC) \
kernel void NAME(device bfloat *a [[buffer(0)]], device bfloat *g [[buffer(1)]], \
    device const float *gs [[buffer(2)]], device bfloat *u [[buffer(3)]], \
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]], \
    device const uint *offsets [[buffer(6)]], device const FlashMoEBucketJob *jobs [[buffer(7)]], \
    device const uint *count [[buffer(8)]], device bfloat *out [[buffer(9)]], \
    device uint *diag [[buffer(10)]], constant FlashInt8ExpertStoreParams &p [[buffer(11)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  prefill_moe_sep21_bf16_codes_gate<M, SG, K, STATIC>(a, g, gs, u, us, ranks, offsets, jobs, count, out, diag, p, group, threads, tid); \
}
#define BF16_CODES_DOWN(NAME, M, SG, K, STATIC) \
kernel void NAME(device bfloat *a [[buffer(0)]], device bfloat *w [[buffer(1)]], \
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]], \
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]], \
    device const uint *count [[buffer(6)]], device const uint *map [[buffer(7)]], \
    device bfloat *out [[buffer(8)]], device uint *diag [[buffer(9)]], \
    constant FlashInt8ExpertStoreParams &p [[buffer(10)]], uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { \
  prefill_moe_sep21_bf16_codes_down<M, SG, K, STATIC>(a, w, s, ranks, offsets, jobs, count, map, out, diag, p, group, threads, tid); \
}

BF16_CODES_GATE(prefill_moe_sep21_bf16_codes_gate_up_m32_n64_sg4, 32, 4, 0, false)
BF16_CODES_DOWN(prefill_moe_sep21_bf16_codes_down_scatter_m32_n64_sg4, 32, 4, 0, false)
BF16_CODES_GATE(prefill_moe_sep21_bf16_codes_gate_up_m32_n64_sg2, 32, 2, 0, false)
BF16_CODES_DOWN(prefill_moe_sep21_bf16_codes_down_scatter_m32_n64_sg2, 32, 2, 0, false)
BF16_CODES_GATE(prefill_moe_sep21_bf16_codes_gate_up_m64_n64_sg8, 64, 8, 0, false)
BF16_CODES_DOWN(prefill_moe_sep21_bf16_codes_down_scatter_m64_n64_sg8, 64, 8, 0, false)
BF16_CODES_GATE(prefill_moe_sep21_bf16_codes_gate_up_m32_n64_k128_sg2, 32, 2, 128, true)
BF16_CODES_DOWN(prefill_moe_sep21_bf16_codes_down_scatter_m32_n64_k128_sg2, 32, 2, 128, true)
#undef BF16_CODES_GATE
#undef BF16_CODES_DOWN
#endif
