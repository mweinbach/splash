// Private N128 W8A8 geometry experiment. This remains the same numerical
// activation alternative as W8A8, not original-BF16-A math. Original signed-I8
// B, F32 late weight scales, BF16 boundaries and output scatter are unchanged.
// Reuse ORIGINAL W8A8 gate_t256/down_t128 quantizer kernels and ABI from the
// original W8A8 AIR; this AIR defines no quantizer kernels. Include the two
// original quantizer commands in complete-chain timings.
// Producers: M32N128 SG4 and M16N128 SG2; gate gridX5, down gridX20. Native
// M32/M16 jobs retain original ordering, tile_rows=M and ceil(routes/M)+511
// capacity. Width640/2560 are divisible by128, so no new column tail exists.
// Native gate/down ABI uses I8 A at buffer0 and appends activation scales at
// gate12/down11. Timed producers bind no audits. Separate _audit producers
// append scaled F32 G13/U14, D12 and raw exact I32 G15/U16, D13. G/U remain
// packed [route,640]; D remains canonical scattered [route,2560].
// I8 x original-I8 accumulates I32. Epilogue arithmetic remains explicitly
// (float(intdot)*activationScale)*weightScale with contraction/reassociation
// disabled, followed by the original BF16/SwiGLU/output stages.
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
inline bfloat prefill_moe_sep21_w8a8_wide_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

template <ushort M, ushort SG, ushort K = 0, bool Static = false, bool Audit = false>
inline bool prefill_moe_sep21_w8a8_wide_job(constant FlashInt8ExpertStoreParams &p,
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

template <ushort M, ushort SG, ushort K = 0, bool Static = false, bool Audit = false>
inline void prefill_moe_sep21_w8a8_wide_gate(device int8_t *input, device int8_t *gate,
    device const float *gate_scale, device int8_t *up, device const float *up_scale,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device bfloat *output, device uint *diag, constant FlashInt8ExpertStoreParams &p,
    device const float *activation_scales, device float *raw_g, device float *raw_u,
    device int *dot_g, device int *dot_u, uint3 group, uint3 threads, uint tid) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  if (group.x >= 5) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!prefill_moe_sep21_w8a8_wide_job<M, SG>(p, ranks, offsets, jobs, job_count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  constexpr ushort N = 128;
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
  auto gd = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), int>();
  auto ud = operation.template get_destination_cooperative_tensor<decltype(a), decltype(u), int>();
  if constexpr (K) {
#pragma unroll
    for (ushort i = 0; i < gd.get_capacity(); ++i) {
      if (gd.is_valid_element(i)) { gd[i] = 0; ud[i] = 0; }
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
    const float activation_scale = activation_scales[ulong(begin + index[1])];
    const float unscaled_g = float(gd[i]) * activation_scale;
    const float unscaled_u = float(ud[i]) * activation_scale;
    const float gf = unscaled_g * gs, uf = unscaled_u * us;
    if (!(activation_scale > 0.0f) || !flash_mpp_finite(activation_scale))
      flash_mpp_error(diag, 4u);
    if constexpr (Audit) {
      const ulong at = ulong(begin + index[1]) * 640 + n;
      raw_g[at] = gf; raw_u[at] = uf;
      dot_g[at] = gd[i]; dot_u[at] = ud[i];
    }
    const bfloat gv = bfloat(gf), uv = bfloat(uf);
    const bfloat silu = gv * prefill_moe_sep21_w8a8_wide_sigmoid(gv);
    const bfloat value = silu * uv;
    if (!(gs > 0.0f) || !(us > 0.0f) || !flash_mpp_finite(gs) ||
        !flash_mpp_finite(us) || !flash_mpp_finite(gf) || !flash_mpp_finite(uf) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(begin + index[1]) * 640 + n] = value;
  }
}

template <ushort M, ushort SG, ushort K = 0, bool Static = false, bool Audit = false>
inline void prefill_moe_sep21_w8a8_wide_down(device int8_t *input, device int8_t *weights,
    device const float *scales, device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device const uint *route_map, device bfloat *output, device uint *diag,
    constant FlashInt8ExpertStoreParams &p, device const float *activation_scales,
    device float *raw, device int *raw_dot, uint3 group, uint3 threads, uint tid) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  if (group.x >= 20) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!prefill_moe_sep21_w8a8_wide_job<M, SG>(p, ranks, offsets, jobs, job_count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  constexpr ushort N = 128;
  const uint column = group.x * N;
  auto a = tensor(input + ulong(begin) * 640,
      dextents<int, 2>{640, int(valid_rows)}, array<int, 2>{1, 640});
  auto b = tensor(weights + (ulong(rank) * 2560 + column) * 640,
      dextents<int, 2>{640, N}, array<int, 2>{1, 640});
  constexpr auto descriptor = matmul2d_descriptor(M, N, K ? int(K) : static_cast<int>(dynamic_extent),
      false, true, false, K ? matmul2d_descriptor::mode::multiply_accumulate : matmul2d_descriptor::mode::multiply);
  using Scope = conditional_t<SG == 1, execution_simdgroup, execution_simdgroups<SG>>;
  matmul2d<descriptor, Scope> operation;
  auto dot = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), int>();
  if constexpr (K) {
#pragma unroll
    for (ushort i = 0; i < dot.get_capacity(); ++i)
      if (dot.is_valid_element(i)) dot[i] = 0;
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
    const float activation_scale = activation_scales[ulong(begin + index[1])];
    const float unscaled = float(dot[i]) * activation_scale;
    const float result = unscaled * scale;
    if (!(activation_scale > 0.0f) || !flash_mpp_finite(activation_scale))
      flash_mpp_error(diag, 4u);
    if constexpr (Audit) {
      const ulong at = ulong(route) * 2560 + n;
      raw[at] = result; raw_dot[at] = dot[i];
    }
    const bfloat value = bfloat(result);
    if (!(scale > 0.0f) || !flash_mpp_finite(scale) || !flash_mpp_finite(result) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(route) * 2560 + n] = value;
  }
}


#define W8A8_GATE_COMMON \
    device int8_t *a [[buffer(0)]], device int8_t *g [[buffer(1)]], \
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]], \
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]], \
    device const uint *offsets [[buffer(6)]], device const FlashMoEBucketJob *jobs [[buffer(7)]], \
    device const uint *count [[buffer(8)]], device bfloat *out [[buffer(9)]], \
    device uint *diag [[buffer(10)]], constant FlashInt8ExpertStoreParams &p [[buffer(11)]], \
    device const float *activation_scales [[buffer(12)]]
#define W8A8_DOWN_COMMON \
    device int8_t *a [[buffer(0)]], device int8_t *w [[buffer(1)]], \
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]], \
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]], \
    device const uint *count [[buffer(6)]], device const uint *map [[buffer(7)]], \
    device bfloat *out [[buffer(8)]], device uint *diag [[buffer(9)]], \
    constant FlashInt8ExpertStoreParams &p [[buffer(10)]], \
    device const float *activation_scales [[buffer(11)]]
#define W8A8_POSITION \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]
#define W8A8_GATE(NAME, M, SG) \
kernel void NAME(W8A8_GATE_COMMON, W8A8_POSITION) { \
  prefill_moe_sep21_w8a8_wide_gate<M, SG, 0, false, false>(a, g, gs, u, us, ranks, offsets, \
      jobs, count, out, diag, p, activation_scales, nullptr, nullptr, nullptr, nullptr, group, threads, tid); \
}
#define W8A8_DOWN(NAME, M, SG) \
kernel void NAME(W8A8_DOWN_COMMON, W8A8_POSITION) { \
  prefill_moe_sep21_w8a8_wide_down<M, SG, 0, false, false>(a, w, s, ranks, offsets, jobs, \
      count, map, out, diag, p, activation_scales, nullptr, nullptr, group, threads, tid); \
}
#define W8A8_GATE_AUDIT(NAME, M, SG) \
kernel void NAME(W8A8_GATE_COMMON, device float *raw_g [[buffer(13)]], \
    device float *raw_u [[buffer(14)]], device int *dot_g [[buffer(15)]], \
    device int *dot_u [[buffer(16)]], W8A8_POSITION) { \
  prefill_moe_sep21_w8a8_wide_gate<M, SG, 0, false, true>(a, g, gs, u, us, ranks, offsets, \
      jobs, count, out, diag, p, activation_scales, raw_g, raw_u, dot_g, dot_u, group, threads, tid); \
}
#define W8A8_DOWN_AUDIT(NAME, M, SG) \
kernel void NAME(W8A8_DOWN_COMMON, device float *raw [[buffer(12)]], \
    device int *dot [[buffer(13)]], W8A8_POSITION) { \
  prefill_moe_sep21_w8a8_wide_down<M, SG, 0, false, true>(a, w, s, ranks, offsets, jobs, \
      count, map, out, diag, p, activation_scales, raw, dot, group, threads, tid); \
}

W8A8_GATE(prefill_moe_sep21_w8a8_wide_gate_up_m32_n128_sg4, 32, 4)
W8A8_GATE_AUDIT(prefill_moe_sep21_w8a8_wide_gate_up_m32_n128_sg4_audit, 32, 4)
W8A8_DOWN(prefill_moe_sep21_w8a8_wide_down_scatter_m32_n128_sg4, 32, 4)
W8A8_DOWN_AUDIT(prefill_moe_sep21_w8a8_wide_down_scatter_m32_n128_sg4_audit, 32, 4)
W8A8_GATE(prefill_moe_sep21_w8a8_wide_gate_up_m16_n128_sg2, 16, 2)
W8A8_GATE_AUDIT(prefill_moe_sep21_w8a8_wide_gate_up_m16_n128_sg2_audit, 16, 2)
W8A8_DOWN(prefill_moe_sep21_w8a8_wide_down_scatter_m16_n128_sg2, 16, 2)
W8A8_DOWN_AUDIT(prefill_moe_sep21_w8a8_wide_down_scatter_m16_n128_sg2_audit, 16, 2)
#undef W8A8_GATE_COMMON
#undef W8A8_DOWN_COMMON
#undef W8A8_POSITION
#undef W8A8_GATE
#undef W8A8_DOWN
#undef W8A8_GATE_AUDIT
#undef W8A8_DOWN_AUDIT
#endif
