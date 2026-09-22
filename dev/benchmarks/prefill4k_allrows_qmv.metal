// Private gathered signed-I8 QMV; no bucket prefix/scan/pack/job/scatter graph.
// Separate numerical policy: lane-strided F32 sums and simd_sum differ from MPP.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "flash/FlashGatheredI8QMV.hpp"

#pragma METAL fp math_mode(safe)
using namespace metal;

inline bool gathered_i8_qmv_finite(float value) {
  return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}
inline void gathered_i8_qmv_error(device atomic_uint *diag, uint bits) {
  atomic_fetch_or_explicit(diag, bits, memory_order_relaxed);
}
inline bfloat gathered_i8_qmv_nan() { return bfloat(as_type<float>(0x7fc00000u)); }

// Copied exactly from int8_expert_store_sigmoid: every explicit bfloat stage
// and compiled fast-exp behavior remain fixed. CPU exp is not a Metal oracle.
#pragma METAL fp math_mode(fast)
inline bfloat gathered_i8_qmv_compiled_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

inline bool gathered_i8_qmv_geometry(constant FlashGatheredI8QMVParams &p,
    uint width, uint3 group, uint3 threads, uint lane, device atomic_uint *diag) {
  if (!p.rows || p.rows > 16 || p.selections != 10 || p.experts != 512 || p.reserved ||
      threads.x != 128 || threads.y != 1 || threads.z != 1 ||
      group.x >= width / 4 || group.y >= p.rows || group.z >= p.selections) {
    if (!lane) gathered_i8_qmv_error(diag, 2u);
    return false;
  }
  return true;
}
inline uint gathered_i8_qmv_rank(device const long *expert_ids,
    device const uint *ranks, ulong route, uint lane, device atomic_uint *diag) {
  const long expert = expert_ids[route];
  if (expert < 0 || expert >= 512) {
    if (!lane) gathered_i8_qmv_error(diag, 1u);
    return UINT_MAX;
  }
  const ulong begin = route / 10 * 10;
  for (uint slot = 0; slot < 10; ++slot)
    if (begin + slot != route && expert_ids[begin + slot] == expert && !lane)
      gathered_i8_qmv_error(diag, 1u);
  const uint rank = ranks[uint(expert)];
  if (rank >= 512) {
    if (!lane) gathered_i8_qmv_error(diag, 1u);
    return UINT_MAX;
  }
  return rank;
}
inline float gathered_i8_qmv_activation(bfloat source, device atomic_uint *diag) {
  const float value = float(source);
  if (!gathered_i8_qmv_finite(value)) {
    gathered_i8_qmv_error(diag, 4u);
    return 0.0f;
  }
  return value;
}

kernel void flash_gathered_i8_qmv_gate_up_sg4_c1(
    device const bfloat *input [[buffer(0)]],
    device const char *gate [[buffer(1)]], device const float *gate_scales [[buffer(2)]],
    device const char *up [[buffer(3)]], device const float *up_scales [[buffer(4)]],
    device const uint *ranks [[buffer(5)]], device const long *expert_ids [[buffer(6)]],
    device bfloat *intermediate [[buffer(7)]], device atomic_uint *diag [[buffer(8)]],
    constant FlashGatheredI8QMVParams &p [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  if (!gathered_i8_qmv_geometry(p, 640, group, threads, lane, diag)) return;
  const ulong route = ulong(group.y) * 10 + group.z;
  const uint n = group.x * 4 + simd;
  const uint rank = gathered_i8_qmv_rank(expert_ids, ranks, route, lane, diag);
  const ulong coefficient = rank == UINT_MAX ? 0 : (ulong(rank) * 640 + n) * 2560;
  float gd = 0.0f, ud = 0.0f;
  for (uint k = lane; k < 2560; k += 32) {
    const float x = gathered_i8_qmv_activation(input[ulong(group.y) * 2560 + k], diag);
    if (rank != UINT_MAX) {
      gd += float(gate[coefficient + k]) * x;
      ud += float(up[coefficient + k]) * x;
    }
  }
  // Inspect hidden rows even when all their original IDs are invalid.
  if (rank == UINT_MAX) {
    if (!lane) intermediate[route * 640 + n] = gathered_i8_qmv_nan();
    return;
  }
  const float gate_dot = simd_sum(gd), up_dot = simd_sum(ud);
  if (!lane) {
    const float gs = gate_scales[ulong(rank) * 640 + n];
    const float us = up_scales[ulong(rank) * 640 + n];
    const float gf = gate_dot * gs, uf = up_dot * us;
    const bfloat gv = bfloat(gf), uv = bfloat(uf);
    const bfloat silu = gv * gathered_i8_qmv_compiled_sigmoid(gv);
    const bfloat value = silu * uv;
    if (!(gs > 0.0f) || !(us > 0.0f) || !gathered_i8_qmv_finite(gs) ||
        !gathered_i8_qmv_finite(us) || !gathered_i8_qmv_finite(gate_dot) ||
        !gathered_i8_qmv_finite(up_dot) || !gathered_i8_qmv_finite(gf) ||
        !gathered_i8_qmv_finite(uf) || !gathered_i8_qmv_finite(float(gv)) ||
        !gathered_i8_qmv_finite(float(uv)) || !gathered_i8_qmv_finite(float(value)))
      gathered_i8_qmv_error(diag, 4u);
    intermediate[route * 640 + n] = value;
  }
}

kernel void flash_gathered_i8_qmv_down_sg4_c1(
    device const bfloat *intermediate [[buffer(0)]],
    device const char *weights [[buffer(1)]], device const float *scales [[buffer(2)]],
    device const uint *ranks [[buffer(3)]], device const long *expert_ids [[buffer(4)]],
    device bfloat *expert_down [[buffer(5)]], device atomic_uint *diag [[buffer(6)]],
    constant FlashGatheredI8QMVParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  if (!gathered_i8_qmv_geometry(p, 2560, group, threads, lane, diag)) return;
  const ulong route = ulong(group.y) * 10 + group.z;
  const uint n = group.x * 4 + simd;
  const uint rank = gathered_i8_qmv_rank(expert_ids, ranks, route, lane, diag);
  if (rank == UINT_MAX) {
    if (!lane) {
      // The original excluded-route poison writes NaN and sticky ID|numeric.
      gathered_i8_qmv_error(diag, 5u);
      expert_down[route * 2560 + n] = gathered_i8_qmv_nan();
    }
    return;
  }
  const ulong coefficient = (ulong(rank) * 2560 + n) * 640;
  float local_dot = 0.0f;
  for (uint k = lane; k < 640; k += 32) {
    const float x = gathered_i8_qmv_activation(intermediate[route * 640 + k], diag);
    local_dot += float(weights[coefficient + k]) * x;
  }
  const float dot = simd_sum(local_dot);
  if (!lane) {
    const float scale = scales[ulong(rank) * 2560 + n];
    const float result = dot * scale;
    const bfloat value = bfloat(result);
    if (!(scale > 0.0f) || !gathered_i8_qmv_finite(scale) ||
        !gathered_i8_qmv_finite(dot) || !gathered_i8_qmv_finite(result) ||
        !gathered_i8_qmv_finite(float(value))) gathered_i8_qmv_error(diag, 4u);
    expert_down[route * 2560 + n] = value;
  }
}
