// New RHS physical tile64 experiment; parent-frozen source SHA f63efa224f63a3c807fafd699742716d9ff1bac2b2ec08e21cbb13540bc01245
// Private direct gather with the ORIGINAL MPP descriptor, validRows1.
// Descriptor equality does not prove exact dot order: Root must pair every dot.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "abi.hpp"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;
inline bool rhs_tile64_finite(float x) { return (as_type<uint>(x) & 0x7f800000u) != 0x7f800000u; }
inline void rhs_tile64_error(device uint *diag, uint bits) {
  atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(diag), bits, memory_order_relaxed);
}
inline bfloat rhs_tile64_nan() { return bfloat(as_type<float>(0x7fc00000u)); }
#pragma METAL fp math_mode(fast)
inline bfloat rhs_tile64_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)
inline uint rhs_tile64_rank(device const long *ids, device const uint *ranks,
                              ulong route, uint tid, device uint *diag) {
  const long id = ids[route];
  if (id < 0 || id >= 512) { if (!tid) rhs_tile64_error(diag, 1u); return UINT_MAX; }
  for (uint slot = 0; slot < 10; ++slot)
    if (route / 10 * 10 + slot != route && ids[route / 10 * 10 + slot] == id && !tid)
      rhs_tile64_error(diag, 1u);
  const uint rank = ranks[uint(id)];
  if (rank >= 512) { if (!tid) rhs_tile64_error(diag, 1u); return UINT_MAX; }
  return rank;
}
template <ushort K>
inline bool rhs_tile64_scan(device const bfloat *x, threadgroup bfloat *safe_a,
    threadgroup atomic_uint *nonfinite, uint tid, device uint *diag) {
  if (!tid) atomic_store_explicit(nonfinite, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device ushort *bits = reinterpret_cast<const device ushort *>(x);
  for (uint k = tid; k < K; k += 128)
    if ((bits[k] & 0x7f80u) == 0x7f80u) atomic_fetch_or_explicit(nonfinite, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool bad = atomic_load_explicit(nonfinite, memory_order_relaxed) != 0;
  if (bad) {
    if (!tid) rhs_tile64_error(diag, 4u);
    threadgroup ushort *safe_bits = reinterpret_cast<threadgroup ushort *>(safe_a);
    for (uint k = tid; k < K; k += 128) {
      const ushort value = bits[k]; safe_bits[k] = (value & 0x7f80u) == 0x7f80u ? 0 : value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return bad;
}
// Diagnostic taps have no new arithmetic. They are only bound by probe
// entries; shipping uses the same body with Probe=false and an empty tuple.
struct rhs_tile64_probe_taps {
  device float *g_dot;
  device float *g_scaled;
  device float *u_dot;
  device float *u_scaled;
  device bfloat *g_bf16;
  device bfloat *u_bf16;
  device bfloat *silu_bf16;
  device bfloat *activation_bf16;
};
template <bool Gate, bool Tile64, bool Probe>
inline void rhs_tile64_execute(device const bfloat *input, device int8_t *gate,
    device const float *gs, device int8_t *up, device const float *us,
    device const uint *ranks, device const long *ids, device bfloat *output, device uint *diag,
    constant FlashGatheredMPPParams &p, uint3 group, uint3 threads, uint tid,
    threadgroup bfloat *safe_a, threadgroup atomic_uint *nonfinite,
    rhs_tile64_probe_taps taps) {
  constexpr ushort K = Gate ? 2560 : 640, N = 64;
  constexpr uint Width = Gate ? 640 : 2560;
  if (!p.rows || p.rows > 16 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= Width / N || group.y >= p.rows || group.z >= 10 ||
      threads.x != 128 || threads.y != 1 || threads.z != 1) {
    if (!tid) rhs_tile64_error(diag, 2u); return;
  }
  const ulong route = ulong(group.y) * 10 + group.z;
  const device bfloat *x = input + (Gate ? ulong(group.y) : route) * K;
  // Finite operands keep the original deviceA tensor. Only malformed-input
  // fallback stages locally; no global scratch or source mutation is made.
  const uint rank = rhs_tile64_rank(ids, ranks, route, tid, diag), column = group.x * N;
  const bool bad = (Gate || rank != UINT_MAX) ? rhs_tile64_scan<K>(x, safe_a, nonfinite, tid, diag) : false;
  if (rank == UINT_MAX) {
    if (!tid) rhs_tile64_error(diag, Gate ? 1u : 5u);
    for (uint n = tid; n < N; n += 128) {
      const ulong tap_index = route * Width + column + n;
      output[tap_index] = rhs_tile64_nan();
      if constexpr (Probe) {
        // Preserve shipping sticky policy: invalid gate=1, invalid down=5.
        // There is no dot on this branch; diagnostic stage taps are NaNs.
        taps.g_dot[tap_index] = as_type<float>(0x7fc00000u);
        taps.g_scaled[tap_index] = as_type<float>(0x7fc00000u);
        taps.g_bf16[tap_index] = rhs_tile64_nan();
        if constexpr (Gate) {
          taps.u_dot[tap_index] = as_type<float>(0x7fc00000u);
          taps.u_scaled[tap_index] = as_type<float>(0x7fc00000u);
          taps.u_bf16[tap_index] = rhs_tile64_nan();
          taps.silu_bf16[tap_index] = rhs_tile64_nan();
          taps.activation_bf16[tap_index] = rhs_tile64_nan();
        }
      }
    }
    return;
  }
  auto a = tensor(const_cast<device bfloat *>(x), dextents<int, 2>{K, 1}, array<int, 2>{1, K});
  auto safe = tensor(safe_a, dextents<int, 2>{K, 1}, array<int, 2>{1, K});
  // MSL4.1 Table2.30 requires strides[0]==1. The physical tile64
  // representation swaps the tensor axes and clears transpose_rhs.
  auto g = tensor(gate + (ulong(rank) * Width + column) * K,
      dextents<int, 2>{Tile64 ? N : K, Tile64 ? K : N}, array<int, 2>{1, Tile64 ? N : K});
  constexpr auto descriptor = matmul2d_descriptor(16, N, static_cast<int>(dynamic_extent),
      false, !Tile64, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto gd = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  if (bad) operation.run(safe, g, gd); else operation.run(a, g, gd);
  auto ud = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  if constexpr (Gate) {
    auto u = tensor(up + (ulong(rank) * Width + column) * K,
        dextents<int, 2>{Tile64 ? N : K, Tile64 ? K : N}, array<int, 2>{1, Tile64 ? N : K});
    if (bad) operation.run(safe, u, ud); else operation.run(a, u, ud);
  }
#pragma unroll
  for (ushort i = 0; i < gd.get_capacity(); ++i) {
    if (!gd.is_valid_element(i)) continue;
    const auto index = gd.get_multidimensional_index(i);
    if (index[1] != 0) continue;
    const uint n = column + index[0];
    const float scale = gs[ulong(rank) * Width + n], result = gd[i] * scale;
    if constexpr (Probe) {
      const ulong tap_index = route * Width + n;
      taps.g_dot[tap_index] = gd[i];
      taps.g_scaled[tap_index] = result;
    }
    bfloat value = bfloat(result);
    if constexpr (Probe) taps.g_bf16[route * Width + n] = value;
    if constexpr (Gate) {
      const float up_scale = us[ulong(rank) * Width + n], up_result = ud[i] * up_scale;
      if constexpr (Probe) {
        const ulong tap_index = route * Width + n;
        taps.u_dot[tap_index] = ud[i];
        taps.u_scaled[tap_index] = up_result;
      }
      const bfloat uv = bfloat(up_result), silu = value * rhs_tile64_sigmoid(value);
      if constexpr (Probe) {
        const ulong tap_index = route * Width + n;
        taps.u_bf16[tap_index] = uv;
        taps.silu_bf16[tap_index] = silu;
      }
      value = silu * uv;
      if constexpr (Probe) taps.activation_bf16[route * Width + n] = value;
      if (!(up_scale > 0) || !rhs_tile64_finite(up_scale) || !rhs_tile64_finite(up_result)) rhs_tile64_error(diag, 4u);
    }
    if (!(scale > 0) || !rhs_tile64_finite(scale) || !rhs_tile64_finite(result) ||
        !rhs_tile64_finite(float(value))) rhs_tile64_error(diag, 4u);
    output[route * Width + n] = value;
  }
}

// Shipping has the original nine/seven buffer bindings and canonical Params16.
// No private reference/control shipping entry exists; Root uses original AIR.
kernel void flash_expert_rhs_tile64_gate_up_m16_n64_sg4(
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]], device const float *gs [[buffer(2)]],
    device int8_t *u [[buffer(3)]], device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]], device uint *diag [[buffer(8)]],
    constant FlashGatheredMPPParams &p [[buffer(9)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[2560]; threadgroup atomic_uint nonfinite;
  rhs_tile64_execute<true, true, false>(x,g,gs,u,us,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,{});
}
kernel void flash_expert_rhs_tile64_down_m16_n64_sg4(
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]], device const float *s [[buffer(2)]],
    device const uint *ranks [[buffer(3)]], device const long *ids [[buffer(4)]],
    device bfloat *out [[buffer(5)]], device uint *diag [[buffer(6)]],
    constant FlashGatheredMPPParams &p [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[640]; threadgroup atomic_uint nonfinite;
  rhs_tile64_execute<false, true, false>(x,w,s,w,s,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,{});
}

#define RHS_TILE64_GATE_PROBE(NAME, TILE64) \
kernel void NAME( \
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]], device const float *gs [[buffer(2)]], \
    device int8_t *u [[buffer(3)]], device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]], \
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]], device uint *diag [[buffer(8)]], \
    device float *g_dot [[buffer(9)]], device float *g_scaled [[buffer(10)]], \
    device float *u_dot [[buffer(11)]], device float *u_scaled [[buffer(12)]], \
    device bfloat *g_bf16 [[buffer(13)]], device bfloat *u_bf16 [[buffer(14)]], \
    device bfloat *silu_bf16 [[buffer(15)]], device bfloat *activation_bf16 [[buffer(16)]], \
    constant FlashGatheredMPPParams &p [[buffer(17)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  threadgroup bfloat safe_a[2560]; threadgroup atomic_uint nonfinite; \
  const rhs_tile64_probe_taps taps{g_dot,g_scaled,u_dot,u_scaled,g_bf16,u_bf16,silu_bf16,activation_bf16}; \
  rhs_tile64_execute<true, TILE64, true>(x,g,gs,u,us,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,taps); \
}
#define RHS_TILE64_DOWN_PROBE(NAME, TILE64) \
kernel void NAME( \
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]], device const float *s [[buffer(2)]], \
    device const uint *ranks [[buffer(3)]], device const long *ids [[buffer(4)]], \
    device bfloat *out [[buffer(5)]], device uint *diag [[buffer(6)]], \
    device float *dot [[buffer(7)]], device float *scaled [[buffer(8)]], device bfloat *down_bf16 [[buffer(9)]], \
    constant FlashGatheredMPPParams &p [[buffer(10)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  threadgroup bfloat safe_a[640]; threadgroup atomic_uint nonfinite; \
  const rhs_tile64_probe_taps taps{dot,scaled,nullptr,nullptr,down_bf16,nullptr,nullptr,nullptr}; \
  rhs_tile64_execute<false, TILE64, true>(x,w,s,w,s,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,taps); \
}
RHS_TILE64_GATE_PROBE(flash_expert_rhs_tile64_reference_gate_up_probe_m16_n64_sg4, false)
RHS_TILE64_GATE_PROBE(flash_expert_rhs_tile64_candidate_gate_up_probe_m16_n64_sg4, true)
RHS_TILE64_DOWN_PROBE(flash_expert_rhs_tile64_reference_down_probe_m16_n64_sg4, false)
RHS_TILE64_DOWN_PROBE(flash_expert_rhs_tile64_candidate_down_probe_m16_n64_sg4, true)
#undef RHS_TILE64_DOWN_PROBE
#undef RHS_TILE64_GATE_PROBE
