// Private low-SIMD direct gathered MPP variant, validRows1. Original coefficient/scale/BF16 boundaries.
// Descriptor equality does not prove exact dot order: Root must pair every dot.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "flash/FlashGatheredMPP.hpp"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;
inline bool prefill_moe_sep21_gathered_sg1_finite(float x) { return (as_type<uint>(x) & 0x7f800000u) != 0x7f800000u; }
inline void prefill_moe_sep21_gathered_sg1_error(device uint *diag, uint bits) {
  atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(diag), bits, memory_order_relaxed);
}
inline bfloat prefill_moe_sep21_gathered_sg1_nan() { return bfloat(as_type<float>(0x7fc00000u)); }
#pragma METAL fp math_mode(fast)
inline bfloat prefill_moe_sep21_gathered_sg1_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)
inline uint prefill_moe_sep21_gathered_sg1_rank(device const long *ids, device const uint *ranks,
                              ulong route, uint tid, device uint *diag) {
  const long id = ids[route];
  if (id < 0 || id >= 512) { if (!tid) prefill_moe_sep21_gathered_sg1_error(diag, 1u); return UINT_MAX; }
  for (uint slot = 0; slot < 10; ++slot)
    if (route / 10 * 10 + slot != route && ids[route / 10 * 10 + slot] == id && !tid)
      prefill_moe_sep21_gathered_sg1_error(diag, 1u);
  const uint rank = ranks[uint(id)];
  if (rank >= 512) { if (!tid) prefill_moe_sep21_gathered_sg1_error(diag, 1u); return UINT_MAX; }
  return rank;
}
template <ushort K>
inline bool prefill_moe_sep21_gathered_sg1_scan(device const bfloat *x, threadgroup bfloat *safe_a,
    threadgroup atomic_uint *nonfinite, uint tid, device uint *diag) {
  if (!tid) atomic_store_explicit(nonfinite, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device ushort *bits = reinterpret_cast<const device ushort *>(x);
  for (uint k = tid; k < K; k += 32)
    if ((bits[k] & 0x7f80u) == 0x7f80u) atomic_fetch_or_explicit(nonfinite, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool bad = atomic_load_explicit(nonfinite, memory_order_relaxed) != 0;
  if (bad) {
    if (!tid) prefill_moe_sep21_gathered_sg1_error(diag, 4u);
    threadgroup ushort *safe_bits = reinterpret_cast<threadgroup ushort *>(safe_a);
    for (uint k = tid; k < K; k += 32) {
      const ushort value = bits[k]; safe_bits[k] = (value & 0x7f80u) == 0x7f80u ? 0 : value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return bad;
}
template <bool Gate>
inline void prefill_moe_sep21_gathered_sg1_execute(device const bfloat *input, device int8_t *gate,
    device const float *gs, device int8_t *up, device const float *us,
    device const uint *ranks, device const long *ids, device bfloat *output, device uint *diag,
    constant FlashGatheredMPPParams &p, uint3 group, uint3 threads, uint tid,
    threadgroup bfloat *safe_a, threadgroup atomic_uint *nonfinite) {
  constexpr ushort K = Gate ? 2560 : 640, N = 64;
  constexpr uint Width = Gate ? 640 : 2560;
  if (!p.rows || p.rows > 16 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= Width / N || group.y >= p.rows || group.z >= 10 ||
      threads.x != 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) prefill_moe_sep21_gathered_sg1_error(diag, 2u); return;
  }
  const ulong route = ulong(group.y) * 10 + group.z;
  const device bfloat *x = input + (Gate ? ulong(group.y) : route) * K;
  // Finite operands keep the original deviceA tensor. Only malformed-input
  // fallback stages locally; no global scratch or source mutation is made.
  const uint rank = prefill_moe_sep21_gathered_sg1_rank(ids, ranks, route, tid, diag), column = group.x * N;
  const bool bad = (Gate || rank != UINT_MAX) ? prefill_moe_sep21_gathered_sg1_scan<K>(x, safe_a, nonfinite, tid, diag) : false;
  if (rank == UINT_MAX) {
    if (!tid) prefill_moe_sep21_gathered_sg1_error(diag, Gate ? 1u : 5u);
    for (uint n = tid; n < N; n += 32) output[route * Width + column + n] = prefill_moe_sep21_gathered_sg1_nan();
    return;
  }
  auto a = tensor(const_cast<device bfloat *>(x), dextents<int, 2>{K, 1}, array<int, 2>{1, K});
  auto safe = tensor(safe_a, dextents<int, 2>{K, 1}, array<int, 2>{1, K});
  auto g = tensor(gate + (ulong(rank) * Width + column) * K,
      dextents<int, 2>{K, N}, array<int, 2>{1, K});
  constexpr auto descriptor = matmul2d_descriptor(16, N, static_cast<int>(dynamic_extent),
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroup> operation;
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
    const float scale = gs[ulong(rank) * Width + n], result = gd[i] * scale;
    bfloat value = bfloat(result);
    if constexpr (Gate) {
      const float up_scale = us[ulong(rank) * Width + n], up_result = ud[i] * up_scale;
      const bfloat uv = bfloat(up_result), silu = value * prefill_moe_sep21_gathered_sg1_sigmoid(value);
      value = silu * uv;
      if (!(up_scale > 0) || !prefill_moe_sep21_gathered_sg1_finite(up_scale) || !prefill_moe_sep21_gathered_sg1_finite(up_result)) prefill_moe_sep21_gathered_sg1_error(diag, 4u);
    }
    if (!(scale > 0) || !prefill_moe_sep21_gathered_sg1_finite(scale) || !prefill_moe_sep21_gathered_sg1_finite(result) ||
        !prefill_moe_sep21_gathered_sg1_finite(float(value))) prefill_moe_sep21_gathered_sg1_error(diag, 4u);
    output[route * Width + n] = value;
  }
}
kernel void prefill_moe_sep21_gathered_sg1_gate_up_m16_n64_sg1(
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]], device const float *gs [[buffer(2)]],
    device int8_t *u [[buffer(3)]], device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]], device uint *diag [[buffer(8)]],
    constant FlashGatheredMPPParams &p [[buffer(9)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[2560]; threadgroup atomic_uint nonfinite;
  prefill_moe_sep21_gathered_sg1_execute<true>(x,g,gs,u,us,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite);
}
kernel void prefill_moe_sep21_gathered_sg1_down_m16_n64_sg1(
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]], device const float *s [[buffer(2)]],
    device const uint *ranks [[buffer(3)]], device const long *ids [[buffer(4)]],
    device bfloat *out [[buffer(5)]], device uint *diag [[buffer(6)]],
    constant FlashGatheredMPPParams &p [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[640]; threadgroup atomic_uint nonfinite;
  prefill_moe_sep21_gathered_sg1_execute<false>(x,w,s,w,s,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite);
}
