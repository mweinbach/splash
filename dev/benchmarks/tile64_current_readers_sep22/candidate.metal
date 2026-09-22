// Standalone current C1/native M16 and gathered validRows1 RHS component.
// No M32/prefill K128 implementation, Worker, Store or format integration.
// All normal/probe entries instantiate literal shipping helpers. Native and
// gathered preserve their distinct original whole-K MPP arithmetic and guards.
// The candidate changes ONLY physical RHS axes/strides/right-transpose.
#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashInt8ExpertStore.h"
#include "flash/FlashGatheredMPP.hpp"
#include "metal/kernels/common/flash_affine_mpp_common.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;
#pragma METAL fp math_mode(fast)
inline bfloat int8_expert_store_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

template <ushort M, ushort SG>
inline bool int8_expert_store_job(constant FlashInt8ExpertStoreParams &p,
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
  if (active > min(p.job_capacity, p.route_capacity) || offsets[512] > p.route_capacity) {
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
  if (rank == UINT_MAX) { if (p.stored_experts == 512 && !tid) flash_mpp_error(diag, 1u); return false; }
  if (rank >= p.stored_experts) { if (!tid) flash_mpp_error(diag, 1u); return false; }
  begin = job.row_begin;
  valid_rows = min(uint(M), end - begin);
  return true;
}

inline bool gathered_mpp_finite(float x) { return (as_type<uint>(x) & 0x7f800000u) != 0x7f800000u; }
inline void gathered_mpp_error(device uint *diag, uint bits) {
  atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(diag), bits, memory_order_relaxed);
}
inline bfloat gathered_mpp_nan() { return bfloat(as_type<float>(0x7fc00000u)); }
#pragma METAL fp math_mode(fast)
inline bfloat gathered_mpp_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)
inline uint gathered_mpp_rank(device const long *ids, device const uint *ranks,
                              ulong route, uint tid, device uint *diag) {
  const long id = ids[route];
  if (id < 0 || id >= 512) { if (!tid) gathered_mpp_error(diag, 1u); return UINT_MAX; }
  for (uint slot = 0; slot < 10; ++slot)
    if (route / 10 * 10 + slot != route && ids[route / 10 * 10 + slot] == id && !tid)
      gathered_mpp_error(diag, 1u);
  const uint rank = ranks[uint(id)];
  if (rank >= 512) { if (!tid) gathered_mpp_error(diag, 1u); return UINT_MAX; }
  return rank;
}
template <ushort K>
inline bool gathered_mpp_scan(device const bfloat *x, threadgroup bfloat *safe_a,
    threadgroup atomic_uint *nonfinite, uint tid, device uint *diag) {
  if (!tid) atomic_store_explicit(nonfinite, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device ushort *bits = reinterpret_cast<const device ushort *>(x);
  for (uint k = tid; k < K; k += 128)
    if ((bits[k] & 0x7f80u) == 0x7f80u) atomic_fetch_or_explicit(nonfinite, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool bad = atomic_load_explicit(nonfinite, memory_order_relaxed) != 0;
  if (bad) {
    if (!tid) gathered_mpp_error(diag, 4u);
    threadgroup ushort *safe_bits = reinterpret_cast<threadgroup ushort *>(safe_a);
    for (uint k = tid; k < K; k += 128) {
      const ushort value = bits[k]; safe_bits[k] = (value & 0x7f80u) == 0x7f80u ? 0 : value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return bad;
}

// Literal baseline normal/tap math.
template <ushort M, ushort SG, bool Probe>
inline void int8_expert_store_gate(device bfloat *input, device int8_t *gate,
    device const float *gate_scale, device int8_t *up, device const float *up_scale,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device bfloat *output, device uint *diag, constant FlashInt8ExpertStoreParams &p,
    uint3 group, uint3 threads, uint tid, device float *raw_g, device float *raw_u, device bfloat *scaled_g, device bfloat *scaled_u) {
  if (group.x >= 10) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!int8_expert_store_job<M, SG>(p, ranks, offsets, jobs, job_count, diag,
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
  constexpr auto descriptor = matmul2d_descriptor(M, N, static_cast<int>(dynamic_extent),
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto gd = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  auto ud = operation.template get_destination_cooperative_tensor<decltype(a), decltype(u), float>();
  operation.run(a, g, gd); operation.run(a, u, ud);
#pragma unroll
  for (ushort i = 0; i < gd.get_capacity(); ++i) {
    if (!gd.is_valid_element(i)) continue;
    const auto index = gd.get_multidimensional_index(i);
    if (uint(index[1]) >= valid_rows) continue;
    const uint n = column + index[0];
    if constexpr (Probe) {
      const ulong at = ulong(begin + index[1]) * 640 + n;
      raw_g[at] = gd[i]; raw_u[at] = ud[i];
    }
    const float gs = gate_scale[ulong(rank) * 640 + n];
    const float us = up_scale[ulong(rank) * 640 + n];
    const float gf = gd[i] * gs, uf = ud[i] * us;
    const bfloat gv = bfloat(gf), uv = bfloat(uf);
    if constexpr (Probe) {
      const ulong at = ulong(begin + index[1]) * 640 + n;
      scaled_g[at] = gv; scaled_u[at] = uv;
    }
    const bfloat silu = gv * int8_expert_store_sigmoid(gv);
    const bfloat value = silu * uv;
    if (!(gs > 0.0f) || !(us > 0.0f) || !flash_mpp_finite(gs) ||
        !flash_mpp_finite(us) || !flash_mpp_finite(gf) || !flash_mpp_finite(uf) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(begin + index[1]) * 640 + n] = value;
  }
}

template <ushort M, ushort SG, bool Probe>
inline void int8_expert_store_down(device bfloat *input, device int8_t *weights,
    device const float *scales, device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device const uint *route_map, device bfloat *output, device uint *diag,
    constant FlashInt8ExpertStoreParams &p, uint3 group, uint3 threads, uint tid, device float *raw, device bfloat *scaled) {
  if (group.x >= 40) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!int8_expert_store_job<M, SG>(p, ranks, offsets, jobs, job_count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  constexpr ushort N = 64;
  const uint column = group.x * N;
  auto a = tensor(input + ulong(begin) * 640,
      dextents<int, 2>{640, int(valid_rows)}, array<int, 2>{1, 640});
  auto b = tensor(weights + (ulong(rank) * 2560 + column) * 640,
      dextents<int, 2>{640, N}, array<int, 2>{1, 640});
  constexpr auto descriptor = matmul2d_descriptor(M, N, static_cast<int>(dynamic_extent),
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  operation.run(a, b, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    if (uint(index[1]) >= valid_rows) continue;
    const uint route = route_map[begin + index[1]], n = column + index[0];
    if (route >= p.route_capacity) { flash_mpp_error(diag, 1u); continue; }
    if constexpr (Probe) {
      raw[ulong(route) * 2560 + n] = dot[i];
    }
    const float scale = scales[ulong(rank) * 2560 + n];
    const float result = dot[i] * scale;
    const bfloat value = bfloat(result);
    if constexpr (Probe) {
      scaled[ulong(route) * 2560 + n] = value;
    }
    if (!(scale > 0.0f) || !flash_mpp_finite(scale) || !flash_mpp_finite(result) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(route) * 2560 + n] = value;
  }
}

template <bool Gate, bool Probe>
inline void gathered_mpp_execute(device const bfloat *input, device int8_t *gate,
    device const float *gs, device int8_t *up, device const float *us,
    device const uint *ranks, device const long *ids, device bfloat *output, device uint *diag,
    constant FlashGatheredMPPParams &p, uint3 group, uint3 threads, uint tid,
    threadgroup bfloat *safe_a, threadgroup atomic_uint *nonfinite, device float *raw_g, device float *raw_u, device bfloat *scaled_g, device bfloat *scaled_u) {
  constexpr ushort K = Gate ? 2560 : 640, N = 64;
  constexpr uint Width = Gate ? 640 : 2560;
  if (!p.rows || p.rows > 16 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= Width / N || group.y >= p.rows || group.z >= 10 ||
      threads.x != 128 || threads.y != 1 || threads.z != 1) {
    if (!tid) gathered_mpp_error(diag, 2u); return;
  }
  const ulong route = ulong(group.y) * 10 + group.z;
  const device bfloat *x = input + (Gate ? ulong(group.y) : route) * K;
  // Finite operands keep the original deviceA tensor. Only malformed-input
  // fallback stages locally; no global scratch or source mutation is made.
  const uint rank = gathered_mpp_rank(ids, ranks, route, tid, diag), column = group.x * N;
  const bool bad = (Gate || rank != UINT_MAX) ? gathered_mpp_scan<K>(x, safe_a, nonfinite, tid, diag) : false;
  if (rank == UINT_MAX) {
    if (!tid) gathered_mpp_error(diag, Gate ? 1u : 5u);
    for (uint n = tid; n < N; n += 128) output[route * Width + column + n] = gathered_mpp_nan();
    return;
  }
  auto a = tensor(const_cast<device bfloat *>(x), dextents<int, 2>{K, 1}, array<int, 2>{1, K});
  auto safe = tensor(safe_a, dextents<int, 2>{K, 1}, array<int, 2>{1, K});
  auto g = tensor(gate + (ulong(rank) * Width + column) * K,
      dextents<int, 2>{K, N}, array<int, 2>{1, K});
  constexpr auto descriptor = matmul2d_descriptor(16, N, static_cast<int>(dynamic_extent),
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto gd = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  if (bad) operation.run(safe, g, gd); else operation.run(a, g, gd);
  auto ud = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  if constexpr (Gate) {
    auto u = tensor(up + (ulong(rank) * Width + column) * K,
        dextents<int, 2>{K, N}, array<int, 2>{1, K});
    if (bad) operation.run(safe, u, ud); else operation.run(a, u, ud);
  }
#pragma unroll
  for (ushort i = 0; i < gd.get_capacity(); ++i) {
    if (!gd.is_valid_element(i)) continue;
    const auto index = gd.get_multidimensional_index(i);
    if (index[1] != 0) continue;
    const uint n = column + index[0];
    if constexpr (Probe) {
      raw_g[route * Width + n] = gd[i];
      if constexpr (Gate) raw_u[route * Width + n] = ud[i];
    }
    const float scale = gs[ulong(rank) * Width + n], result = gd[i] * scale;
    bfloat value = bfloat(result);
    if constexpr (Probe) {
      scaled_g[route * Width + n] = value;
    }
    if constexpr (Gate) {
      const float up_scale = us[ulong(rank) * Width + n], up_result = ud[i] * up_scale;
      const bfloat uv = bfloat(up_result), silu = value * gathered_mpp_sigmoid(value);
      if constexpr (Probe) {
        scaled_u[route * Width + n] = uv;
      }
      value = silu * uv;
      if (!(up_scale > 0) || !gathered_mpp_finite(up_scale) || !gathered_mpp_finite(up_result)) gathered_mpp_error(diag, 4u);
    }
    if (!(scale > 0) || !gathered_mpp_finite(scale) || !gathered_mpp_finite(result) ||
        !gathered_mpp_finite(float(value))) gathered_mpp_error(diag, 4u);
    output[route * Width + n] = value;
  }
}

// Same math with legal physically transposed RHS tensors.
template <ushort M, ushort SG, bool Probe>
inline void tile64_current_native_gate(device bfloat *input, device int8_t *gate,
    device const float *gate_scale, device int8_t *up, device const float *up_scale,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device bfloat *output, device uint *diag, constant FlashInt8ExpertStoreParams &p,
    uint3 group, uint3 threads, uint tid, device float *raw_g, device float *raw_u, device bfloat *scaled_g, device bfloat *scaled_u) {
  if (group.x >= 10) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!int8_expert_store_job<M, SG>(p, ranks, offsets, jobs, job_count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  constexpr ushort N = 64;
  const uint column = group.x * N;
  // Dynamic row bounds make incomplete bucket tiles safe without staging,
  // copying or reading the next expert's input. MPP masks the tail rows.
  auto a = tensor(input + ulong(begin) * 2560,
      dextents<int, 2>{2560, int(valid_rows)}, array<int, 2>{1, 2560});
  auto g = tensor(gate + (ulong(rank) * 640 + column) * 2560,
      dextents<int, 2>{N, 2560}, array<int, 2>{1, N});
  auto u = tensor(up + (ulong(rank) * 640 + column) * 2560,
      dextents<int, 2>{N, 2560}, array<int, 2>{1, N});
  constexpr auto descriptor = matmul2d_descriptor(M, N, static_cast<int>(dynamic_extent),
      false, false, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto gd = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  auto ud = operation.template get_destination_cooperative_tensor<decltype(a), decltype(u), float>();
  operation.run(a, g, gd); operation.run(a, u, ud);
#pragma unroll
  for (ushort i = 0; i < gd.get_capacity(); ++i) {
    if (!gd.is_valid_element(i)) continue;
    const auto index = gd.get_multidimensional_index(i);
    if (uint(index[1]) >= valid_rows) continue;
    const uint n = column + index[0];
    if constexpr (Probe) {
      const ulong at = ulong(begin + index[1]) * 640 + n;
      raw_g[at] = gd[i]; raw_u[at] = ud[i];
    }
    const float gs = gate_scale[ulong(rank) * 640 + n];
    const float us = up_scale[ulong(rank) * 640 + n];
    const float gf = gd[i] * gs, uf = ud[i] * us;
    const bfloat gv = bfloat(gf), uv = bfloat(uf);
    if constexpr (Probe) {
      const ulong at = ulong(begin + index[1]) * 640 + n;
      scaled_g[at] = gv; scaled_u[at] = uv;
    }
    const bfloat silu = gv * int8_expert_store_sigmoid(gv);
    const bfloat value = silu * uv;
    if (!(gs > 0.0f) || !(us > 0.0f) || !flash_mpp_finite(gs) ||
        !flash_mpp_finite(us) || !flash_mpp_finite(gf) || !flash_mpp_finite(uf) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(begin + index[1]) * 640 + n] = value;
  }
}

template <ushort M, ushort SG, bool Probe>
inline void tile64_current_native_down(device bfloat *input, device int8_t *weights,
    device const float *scales, device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device const uint *route_map, device bfloat *output, device uint *diag,
    constant FlashInt8ExpertStoreParams &p, uint3 group, uint3 threads, uint tid, device float *raw, device bfloat *scaled) {
  if (group.x >= 40) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!int8_expert_store_job<M, SG>(p, ranks, offsets, jobs, job_count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  constexpr ushort N = 64;
  const uint column = group.x * N;
  auto a = tensor(input + ulong(begin) * 640,
      dextents<int, 2>{640, int(valid_rows)}, array<int, 2>{1, 640});
  auto b = tensor(weights + (ulong(rank) * 2560 + column) * 640,
      dextents<int, 2>{N, 640}, array<int, 2>{1, N});
  constexpr auto descriptor = matmul2d_descriptor(M, N, static_cast<int>(dynamic_extent),
      false, false, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  operation.run(a, b, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    if (uint(index[1]) >= valid_rows) continue;
    const uint route = route_map[begin + index[1]], n = column + index[0];
    if (route >= p.route_capacity) { flash_mpp_error(diag, 1u); continue; }
    if constexpr (Probe) {
      raw[ulong(route) * 2560 + n] = dot[i];
    }
    const float scale = scales[ulong(rank) * 2560 + n];
    const float result = dot[i] * scale;
    const bfloat value = bfloat(result);
    if constexpr (Probe) {
      scaled[ulong(route) * 2560 + n] = value;
    }
    if (!(scale > 0.0f) || !flash_mpp_finite(scale) || !flash_mpp_finite(result) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(route) * 2560 + n] = value;
  }
}

template <bool Gate, bool Probe>
inline void tile64_current_gathered_execute(device const bfloat *input, device int8_t *gate,
    device const float *gs, device int8_t *up, device const float *us,
    device const uint *ranks, device const long *ids, device bfloat *output, device uint *diag,
    constant FlashGatheredMPPParams &p, uint3 group, uint3 threads, uint tid,
    threadgroup bfloat *safe_a, threadgroup atomic_uint *nonfinite, device float *raw_g, device float *raw_u, device bfloat *scaled_g, device bfloat *scaled_u) {
  constexpr ushort K = Gate ? 2560 : 640, N = 64;
  constexpr uint Width = Gate ? 640 : 2560;
  if (!p.rows || p.rows > 16 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= Width / N || group.y >= p.rows || group.z >= 10 ||
      threads.x != 128 || threads.y != 1 || threads.z != 1) {
    if (!tid) gathered_mpp_error(diag, 2u); return;
  }
  const ulong route = ulong(group.y) * 10 + group.z;
  const device bfloat *x = input + (Gate ? ulong(group.y) : route) * K;
  // Finite operands keep the original deviceA tensor. Only malformed-input
  // fallback stages locally; no global scratch or source mutation is made.
  const uint rank = gathered_mpp_rank(ids, ranks, route, tid, diag), column = group.x * N;
  const bool bad = (Gate || rank != UINT_MAX) ? gathered_mpp_scan<K>(x, safe_a, nonfinite, tid, diag) : false;
  if (rank == UINT_MAX) {
    if (!tid) gathered_mpp_error(diag, Gate ? 1u : 5u);
    for (uint n = tid; n < N; n += 128) output[route * Width + column + n] = gathered_mpp_nan();
    return;
  }
  auto a = tensor(const_cast<device bfloat *>(x), dextents<int, 2>{K, 1}, array<int, 2>{1, K});
  auto safe = tensor(safe_a, dextents<int, 2>{K, 1}, array<int, 2>{1, K});
  auto g = tensor(gate + (ulong(rank) * Width + column) * K,
      dextents<int, 2>{N, K}, array<int, 2>{1, N});
  constexpr auto descriptor = matmul2d_descriptor(16, N, static_cast<int>(dynamic_extent),
      false, false, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto gd = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  if (bad) operation.run(safe, g, gd); else operation.run(a, g, gd);
  auto ud = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  if constexpr (Gate) {
    auto u = tensor(up + (ulong(rank) * Width + column) * K,
        dextents<int, 2>{N, K}, array<int, 2>{1, N});
    if (bad) operation.run(safe, u, ud); else operation.run(a, u, ud);
  }
#pragma unroll
  for (ushort i = 0; i < gd.get_capacity(); ++i) {
    if (!gd.is_valid_element(i)) continue;
    const auto index = gd.get_multidimensional_index(i);
    if (index[1] != 0) continue;
    const uint n = column + index[0];
    if constexpr (Probe) {
      raw_g[route * Width + n] = gd[i];
      if constexpr (Gate) raw_u[route * Width + n] = ud[i];
    }
    const float scale = gs[ulong(rank) * Width + n], result = gd[i] * scale;
    bfloat value = bfloat(result);
    if constexpr (Probe) {
      scaled_g[route * Width + n] = value;
    }
    if constexpr (Gate) {
      const float up_scale = us[ulong(rank) * Width + n], up_result = ud[i] * up_scale;
      const bfloat uv = bfloat(up_result), silu = value * gathered_mpp_sigmoid(value);
      if constexpr (Probe) {
        scaled_u[route * Width + n] = uv;
      }
      value = silu * uv;
      if (!(up_scale > 0) || !gathered_mpp_finite(up_scale) || !gathered_mpp_finite(up_result)) gathered_mpp_error(diag, 4u);
    }
    if (!(scale > 0) || !gathered_mpp_finite(scale) || !gathered_mpp_finite(result) ||
        !gathered_mpp_finite(float(value))) gathered_mpp_error(diag, 4u);
    output[route * Width + n] = value;
  }
}

kernel void tile64_current_readers_sep22_baseline_native_gate_up_m16_sg4(device bfloat *a [[buffer(0)]], device int8_t *g [[buffer(1)]], 
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]], 
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]], 
    device const uint *offsets [[buffer(6)]], device const FlashMoEBucketJob *jobs [[buffer(7)]], 
    device const uint *count [[buffer(8)]], device bfloat *out [[buffer(9)]], 
    device uint *diag [[buffer(10)]], constant FlashInt8ExpertStoreParams &p [[buffer(11)]], 
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], 
    uint tid [[thread_index_in_threadgroup]]) { 
  int8_expert_store_gate<16, 4, false>(a, g, gs, u, us, ranks, offsets, jobs, count, out, diag, p, group, threads, tid, nullptr, nullptr, nullptr, nullptr); 
}

kernel void tile64_current_readers_sep22_baseline_native_gate_up_m16_sg4_probe(device bfloat *a [[buffer(0)]], device int8_t *g [[buffer(1)]], 
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]], 
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]], 
    device const uint *offsets [[buffer(6)]], device const FlashMoEBucketJob *jobs [[buffer(7)]], 
    device const uint *count [[buffer(8)]], device bfloat *out [[buffer(9)]], 
    device uint *diag [[buffer(10)]], constant FlashInt8ExpertStoreParams &p [[buffer(11)]], 
    device float *raw_g [[buffer(12)]], device float *raw_u [[buffer(13)]],
    device bfloat *scaled_g [[buffer(14)]], device bfloat *scaled_u [[buffer(15)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], 
    uint tid [[thread_index_in_threadgroup]]) { 
  int8_expert_store_gate<16, 4, true>(a, g, gs, u, us, ranks, offsets, jobs, count, out, diag, p, group, threads, tid, raw_g, raw_u, scaled_g, scaled_u); 
}

kernel void tile64_current_readers_sep22_baseline_native_down_m16_sg4(device bfloat *a [[buffer(0)]], device int8_t *w [[buffer(1)]], 
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]], 
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]], 
    device const uint *count [[buffer(6)]], device const uint *map [[buffer(7)]], 
    device bfloat *out [[buffer(8)]], device uint *diag [[buffer(9)]], 
    constant FlashInt8ExpertStoreParams &p [[buffer(10)]], uint3 group [[threadgroup_position_in_grid]], 
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { 
  int8_expert_store_down<16, 4, false>(a, w, s, ranks, offsets, jobs, count, map, out, diag, p, group, threads, tid, nullptr, nullptr); 
}

kernel void tile64_current_readers_sep22_baseline_native_down_m16_sg4_probe(device bfloat *a [[buffer(0)]], device int8_t *w [[buffer(1)]], 
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]], 
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]], 
    device const uint *count [[buffer(6)]], device const uint *map [[buffer(7)]], 
    device bfloat *out [[buffer(8)]], device uint *diag [[buffer(9)]], 
    constant FlashInt8ExpertStoreParams &p [[buffer(10)]], 
    device float *raw [[buffer(11)]], device bfloat *scaled [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]], 
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { 
  int8_expert_store_down<16, 4, true>(a, w, s, ranks, offsets, jobs, count, map, out, diag, p, group, threads, tid, raw, scaled); 
}

kernel void tile64_current_readers_sep22_baseline_gathered_gate_up_m16_sg4(
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]], device const float *gs [[buffer(2)]],
    device int8_t *u [[buffer(3)]], device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]], device uint *diag [[buffer(8)]],
    constant FlashGatheredMPPParams &p [[buffer(9)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[2560]; threadgroup atomic_uint nonfinite;
  gathered_mpp_execute<true, false>(x,g,gs,u,us,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,nullptr,nullptr,nullptr,nullptr);
}

kernel void tile64_current_readers_sep22_baseline_gathered_gate_up_m16_sg4_probe(
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]], device const float *gs [[buffer(2)]],
    device int8_t *u [[buffer(3)]], device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]], device uint *diag [[buffer(8)]],
    constant FlashGatheredMPPParams &p [[buffer(9)]], 
    device float *raw_g [[buffer(10)]], device float *raw_u [[buffer(11)]],
    device bfloat *scaled_g [[buffer(12)]], device bfloat *scaled_u [[buffer(13)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[2560]; threadgroup atomic_uint nonfinite;
  gathered_mpp_execute<true, true>(x,g,gs,u,us,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,raw_g,raw_u,scaled_g,scaled_u);
}

kernel void tile64_current_readers_sep22_baseline_gathered_down_m16_sg4(
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]], device const float *s [[buffer(2)]],
    device const uint *ranks [[buffer(3)]], device const long *ids [[buffer(4)]],
    device bfloat *out [[buffer(5)]], device uint *diag [[buffer(6)]],
    constant FlashGatheredMPPParams &p [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[640]; threadgroup atomic_uint nonfinite;
  gathered_mpp_execute<false, false>(x,w,s,w,s,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,nullptr,nullptr,nullptr,nullptr);
}

kernel void tile64_current_readers_sep22_baseline_gathered_down_m16_sg4_probe(
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]], device const float *s [[buffer(2)]],
    device const uint *ranks [[buffer(3)]], device const long *ids [[buffer(4)]],
    device bfloat *out [[buffer(5)]], device uint *diag [[buffer(6)]],
    constant FlashGatheredMPPParams &p [[buffer(7)]], 
    device float *raw [[buffer(8)]], device bfloat *scaled [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[640]; threadgroup atomic_uint nonfinite;
  gathered_mpp_execute<false, true>(x,w,s,w,s,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,raw,nullptr,scaled,nullptr);
}

kernel void tile64_current_readers_sep22_candidate_native_gate_up_m16_sg4(device bfloat *a [[buffer(0)]], device int8_t *g [[buffer(1)]], 
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]], 
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]], 
    device const uint *offsets [[buffer(6)]], device const FlashMoEBucketJob *jobs [[buffer(7)]], 
    device const uint *count [[buffer(8)]], device bfloat *out [[buffer(9)]], 
    device uint *diag [[buffer(10)]], constant FlashInt8ExpertStoreParams &p [[buffer(11)]], 
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], 
    uint tid [[thread_index_in_threadgroup]]) { 
  tile64_current_native_gate<16, 4, false>(a, g, gs, u, us, ranks, offsets, jobs, count, out, diag, p, group, threads, tid, nullptr, nullptr, nullptr, nullptr); 
}

kernel void tile64_current_readers_sep22_candidate_native_gate_up_m16_sg4_probe(device bfloat *a [[buffer(0)]], device int8_t *g [[buffer(1)]], 
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]], 
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]], 
    device const uint *offsets [[buffer(6)]], device const FlashMoEBucketJob *jobs [[buffer(7)]], 
    device const uint *count [[buffer(8)]], device bfloat *out [[buffer(9)]], 
    device uint *diag [[buffer(10)]], constant FlashInt8ExpertStoreParams &p [[buffer(11)]], 
    device float *raw_g [[buffer(12)]], device float *raw_u [[buffer(13)]],
    device bfloat *scaled_g [[buffer(14)]], device bfloat *scaled_u [[buffer(15)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], 
    uint tid [[thread_index_in_threadgroup]]) { 
  tile64_current_native_gate<16, 4, true>(a, g, gs, u, us, ranks, offsets, jobs, count, out, diag, p, group, threads, tid, raw_g, raw_u, scaled_g, scaled_u); 
}

kernel void tile64_current_readers_sep22_candidate_native_down_m16_sg4(device bfloat *a [[buffer(0)]], device int8_t *w [[buffer(1)]], 
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]], 
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]], 
    device const uint *count [[buffer(6)]], device const uint *map [[buffer(7)]], 
    device bfloat *out [[buffer(8)]], device uint *diag [[buffer(9)]], 
    constant FlashInt8ExpertStoreParams &p [[buffer(10)]], uint3 group [[threadgroup_position_in_grid]], 
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { 
  tile64_current_native_down<16, 4, false>(a, w, s, ranks, offsets, jobs, count, map, out, diag, p, group, threads, tid, nullptr, nullptr); 
}

kernel void tile64_current_readers_sep22_candidate_native_down_m16_sg4_probe(device bfloat *a [[buffer(0)]], device int8_t *w [[buffer(1)]], 
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]], 
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]], 
    device const uint *count [[buffer(6)]], device const uint *map [[buffer(7)]], 
    device bfloat *out [[buffer(8)]], device uint *diag [[buffer(9)]], 
    constant FlashInt8ExpertStoreParams &p [[buffer(10)]], 
    device float *raw [[buffer(11)]], device bfloat *scaled [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]], 
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { 
  tile64_current_native_down<16, 4, true>(a, w, s, ranks, offsets, jobs, count, map, out, diag, p, group, threads, tid, raw, scaled); 
}

kernel void tile64_current_readers_sep22_candidate_gathered_gate_up_m16_sg4(
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]], device const float *gs [[buffer(2)]],
    device int8_t *u [[buffer(3)]], device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]], device uint *diag [[buffer(8)]],
    constant FlashGatheredMPPParams &p [[buffer(9)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[2560]; threadgroup atomic_uint nonfinite;
  tile64_current_gathered_execute<true, false>(x,g,gs,u,us,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,nullptr,nullptr,nullptr,nullptr);
}

kernel void tile64_current_readers_sep22_candidate_gathered_gate_up_m16_sg4_probe(
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]], device const float *gs [[buffer(2)]],
    device int8_t *u [[buffer(3)]], device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]], device uint *diag [[buffer(8)]],
    constant FlashGatheredMPPParams &p [[buffer(9)]], 
    device float *raw_g [[buffer(10)]], device float *raw_u [[buffer(11)]],
    device bfloat *scaled_g [[buffer(12)]], device bfloat *scaled_u [[buffer(13)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[2560]; threadgroup atomic_uint nonfinite;
  tile64_current_gathered_execute<true, true>(x,g,gs,u,us,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,raw_g,raw_u,scaled_g,scaled_u);
}

kernel void tile64_current_readers_sep22_candidate_gathered_down_m16_sg4(
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]], device const float *s [[buffer(2)]],
    device const uint *ranks [[buffer(3)]], device const long *ids [[buffer(4)]],
    device bfloat *out [[buffer(5)]], device uint *diag [[buffer(6)]],
    constant FlashGatheredMPPParams &p [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[640]; threadgroup atomic_uint nonfinite;
  tile64_current_gathered_execute<false, false>(x,w,s,w,s,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,nullptr,nullptr,nullptr,nullptr);
}

kernel void tile64_current_readers_sep22_candidate_gathered_down_m16_sg4_probe(
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]], device const float *s [[buffer(2)]],
    device const uint *ranks [[buffer(3)]], device const long *ids [[buffer(4)]],
    device bfloat *out [[buffer(5)]], device uint *diag [[buffer(6)]],
    constant FlashGatheredMPPParams &p [[buffer(7)]], 
    device float *raw [[buffer(8)]], device bfloat *scaled [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[640]; threadgroup atomic_uint nonfinite;
  tile64_current_gathered_execute<false, true>(x,w,s,w,s,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,raw,nullptr,scaled,nullptr);
}
#endif
