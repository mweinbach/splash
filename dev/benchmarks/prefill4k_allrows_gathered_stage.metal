// Private sanitize-once gathered MPP candidate. Source-only qualification.
// Root must pair raw/scaled/BF16/stage bits before model integration.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "flash/FlashGatheredMPP.hpp"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;
inline bool gathered_stage_finite(float x) { return (as_type<uint>(x) & 0x7f800000u) != 0x7f800000u; }
inline void gathered_stage_error(device uint *diag, uint bits) {
  atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(diag), bits, memory_order_relaxed);
}
inline bfloat gathered_stage_nan() { return bfloat(as_type<float>(0x7fc00000u)); }
#pragma METAL fp math_mode(fast)
inline bfloat gathered_stage_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)
inline uint gathered_stage_rank(device const long *ids, device const uint *ranks,
                              ulong route, uint tid, device uint *diag) {
  const long id = ids[route];
  if (id < 0 || id >= 512) { if (!tid) gathered_stage_error(diag, 1u); return UINT_MAX; }
  for (uint slot = 0; slot < 10; ++slot)
    if (route / 10 * 10 + slot != route && ids[route / 10 * 10 + slot] == id && !tid)
      gathered_stage_error(diag, 1u);
  const uint rank = ranks[uint(id)];
  if (rank >= 512) { if (!tid) gathered_stage_error(diag, 1u); return UINT_MAX; }
  return rank;
}
// Canonical scratch is independently admitted by the host and is disjoint from
// original operands and every output/immutable allocation. A producer must be
// ordered after the matching sanitize dispatch; producers do not revalidate A.
// Preserve every finite BF16 bit, including -0 and subnormals, via ushort loads.
template <bool Gate>
inline void gathered_stage_sanitize_execute(device const bfloat *input,
    device const uint *ranks, device const long *ids, device bfloat *safe,
    device uint *diag, constant FlashGatheredMPPParams &p, uint3 group,
    uint3 threads, uint tid) {
  constexpr uint K = Gate ? 2560 : 640;
  if (!p.rows || p.rows > 16 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= (K + 255) / 256 || group.y >= p.rows || group.z >= (Gate ? 1u : 10u) ||
      threads.x != 256 || threads.y != 1 || threads.z != 1) {
    if (!tid) gathered_stage_error(diag, 2u); return;
  }
  const uint k = group.x * 256 + tid;
  const ulong row = Gate ? ulong(group.y) : ulong(group.y) * 10 + group.z;
  if (k >= K) return;
  const device ushort *source_bits = reinterpret_cast<const device ushort *>(input);
  device ushort *safe_bits = reinterpret_cast<device ushort *>(safe);
  if constexpr (!Gate) {
    // Excluded routes never read arbitrary/nonfinite intermediate memory.
    // Invalid IDs/ranks retain old down poison policy in the producer (bit5).
    const uint rank = gathered_stage_rank(ids, ranks, row, tid, diag);
    if (rank == UINT_MAX) { safe_bits[row * K + k] = 0; return; }
  }
  const ushort value = source_bits[row * K + k];
  const bool bad = (value & 0x7f80u) == 0x7f80u;
  if (bad) gathered_stage_error(diag, 4u);
  safe_bits[row * K + k] = bad ? 0 : value;
}
kernel void flash_gathered_stage_sanitize_hidden(
    device const bfloat *input [[buffer(0)]], device const uint *ranks [[buffer(1)]],
    device const long *ids [[buffer(2)]], device bfloat *safe [[buffer(3)]],
    device uint *diag [[buffer(4)]], constant FlashGatheredMPPParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  gathered_stage_sanitize_execute<true>(input,ranks,ids,safe,diag,p,group,threads,tid);
}
kernel void flash_gathered_stage_sanitize_down(
    device const bfloat *input [[buffer(0)]], device const uint *ranks [[buffer(1)]],
    device const long *ids [[buffer(2)]], device bfloat *safe [[buffer(3)]],
    device uint *diag [[buffer(4)]], constant FlashGatheredMPPParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  gathered_stage_sanitize_execute<false>(input,ranks,ids,safe,diag,p,group,threads,tid);
}
template <bool Gate>
inline void gathered_stage_execute(device const bfloat *input, device int8_t *gate,
    device const float *gs, device int8_t *up, device const float *us,
    device const uint *ranks, device const long *ids, device bfloat *output, device uint *diag,
    constant FlashGatheredMPPParams &p, uint3 group, uint3 threads, uint tid) {
  constexpr ushort K = Gate ? 2560 : 640, N = 64;
  constexpr uint Width = Gate ? 640 : 2560;
  if (!p.rows || p.rows > 16 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= Width / N || group.y >= p.rows || group.z >= 10 ||
      threads.x != 128 || threads.y != 1 || threads.z != 1) {
    if (!tid) gathered_stage_error(diag, 2u); return;
  }
  const ulong route = ulong(group.y) * 10 + group.z;
  const device bfloat *x = input + (Gate ? ulong(group.y) : route) * K;
  // Operand A is the matching bitwise canonical sanitize-once device scratch.
  const uint rank = gathered_stage_rank(ids, ranks, route, tid, diag), column = group.x * N;
  if (rank == UINT_MAX) {
    if (!tid) gathered_stage_error(diag, Gate ? 1u : 5u);
    for (uint n = tid; n < N; n += 128) output[route * Width + column + n] = gathered_stage_nan();
    return;
  }
  auto a = tensor(const_cast<device bfloat *>(x), dextents<int, 2>{K, 1}, array<int, 2>{1, K});
  auto g = tensor(gate + (ulong(rank) * Width + column) * K,
      dextents<int, 2>{K, N}, array<int, 2>{1, K});
  constexpr auto descriptor = matmul2d_descriptor(16, N, static_cast<int>(dynamic_extent),
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto gd = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  operation.run(a, g, gd);
  auto ud = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  if constexpr (Gate) {
    auto u = tensor(up + (ulong(rank) * Width + column) * K,
        dextents<int, 2>{K, N}, array<int, 2>{1, K});
    operation.run(a, u, ud);
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
      const bfloat uv = bfloat(up_result), silu = value * gathered_stage_sigmoid(value);
      value = silu * uv;
      if (!(up_scale > 0) || !gathered_stage_finite(up_scale) || !gathered_stage_finite(up_result)) gathered_stage_error(diag, 4u);
    }
    if (!(scale > 0) || !gathered_stage_finite(scale) || !gathered_stage_finite(result) ||
        !gathered_stage_finite(float(value))) gathered_stage_error(diag, 4u);
    output[route * Width + n] = value;
  }
}
kernel void flash_gathered_stage_gate_up_m16_n64_sg4(
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]], device const float *gs [[buffer(2)]],
    device int8_t *u [[buffer(3)]], device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]], device uint *diag [[buffer(8)]],
    constant FlashGatheredMPPParams &p [[buffer(9)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  gathered_stage_execute<true>(x,g,gs,u,us,ranks,ids,out,diag,p,group,threads,tid);
}
kernel void flash_gathered_stage_down_m16_n64_sg4(
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]], device const float *s [[buffer(2)]],
    device const uint *ranks [[buffer(3)]], device const long *ids [[buffer(4)]],
    device bfloat *out [[buffer(5)]], device uint *diag [[buffer(6)]],
    constant FlashGatheredMPPParams &p [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  gathered_stage_execute<false>(x,w,s,w,s,ranks,ids,out,diag,p,group,threads,tid);
}
