// Untimed coupled native/consumer stage taps.
#define gathered_mpp_finite r1_finite_summary_private_gathered_mpp_finite
#define gathered_mpp_error r1_finite_summary_private_gathered_mpp_error
#define gathered_mpp_nan r1_finite_summary_private_gathered_mpp_nan
#define gathered_mpp_sigmoid r1_finite_summary_private_gathered_mpp_sigmoid
#define gathered_mpp_rank r1_finite_summary_private_gathered_mpp_rank
#define gathered_mpp_scan r1_finite_summary_private_gathered_mpp_scan
#define gathered_mpp_execute r1_finite_summary_private_gathered_mpp_execute
#include "native_helper.metal"
#define R1_FINITE_SUMMARY_PROTOCOL_ONLY 1
#include "summary.metalh"
template <bool Gate>
inline void r1_finite_summary_private_gathered_mpp_execute_tap(device const bfloat *input, device int8_t *gate,
    device const float *gs, device int8_t *up, device const float *us,
    device const uint *ranks, device const long *ids, device bfloat *output, device uint *diag,
    constant FlashGatheredMPPParams &p, uint3 group, uint3 threads, uint tid,
    threadgroup bfloat *safe_a, threadgroup atomic_uint *nonfinite,
    device float *raw_g, device float *scaled_g, device float *raw_u, device float *scaled_u,
    device bfloat *bf_g, device bfloat *bf_u, device bfloat *bf_silu, device bfloat *bf_activation) {
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
    const float scale = gs[ulong(rank) * Width + n];
    const float tap_raw_g = gd[i], result = tap_raw_g * scale;
    bfloat value = bfloat(result);
    const ulong tap_at = route * Width + n;
    raw_g[tap_at] = tap_raw_g; scaled_g[tap_at] = result; bf_g[tap_at] = value;
    if constexpr (Gate) {
      const float up_scale = us[ulong(rank) * Width + n];
      const float tap_raw_u = ud[i], up_result = tap_raw_u * up_scale;
      const bfloat uv = bfloat(up_result), silu = value * gathered_mpp_sigmoid(value);
      raw_u[tap_at] = tap_raw_u; scaled_u[tap_at] = up_result;
      bf_u[tap_at] = uv; bf_silu[tap_at] = silu;
      value = silu * uv;
      bf_activation[tap_at] = value;
      if (!(up_scale > 0) || !gathered_mpp_finite(up_scale) || !gathered_mpp_finite(up_result)) gathered_mpp_error(diag, 4u);
    }
    if (!(scale > 0) || !gathered_mpp_finite(scale) || !gathered_mpp_finite(result) ||
        !gathered_mpp_finite(float(value))) gathered_mpp_error(diag, 4u);
    output[route * Width + n] = value;
  }
}
template <bool Gate>
inline void r1_finite_summary_private_gathered_mpp_execute_consumer_tap(device const bfloat *input, device int8_t *gate,
    device const float *gs, device int8_t *up, device const float *us,
    device const uint *ranks, device const long *ids, device bfloat *output, device uint *diag,
    constant FlashGatheredMPPParams &p, uint3 group, uint3 threads, uint tid,
    threadgroup bfloat *safe_a, threadgroup atomic_uint *nonfinite, bool needsOriginalScan,
    device float *raw_g, device float *scaled_g, device float *raw_u, device float *scaled_u,
    device bfloat *bf_g, device bfloat *bf_u, device bfloat *bf_silu, device bfloat *bf_activation) {
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
  const bool bad = (Gate || rank != UINT_MAX) ? (needsOriginalScan ? gathered_mpp_scan<K>(x, safe_a, nonfinite, tid, diag) : false) : false;
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
    const float scale = gs[ulong(rank) * Width + n];
    const float tap_raw_g = gd[i], result = tap_raw_g * scale;
    bfloat value = bfloat(result);
    const ulong tap_at = route * Width + n;
    raw_g[tap_at] = tap_raw_g; scaled_g[tap_at] = result; bf_g[tap_at] = value;
    if constexpr (Gate) {
      const float up_scale = us[ulong(rank) * Width + n];
      const float tap_raw_u = ud[i], up_result = tap_raw_u * up_scale;
      const bfloat uv = bfloat(up_result), silu = value * gathered_mpp_sigmoid(value);
      raw_u[tap_at] = tap_raw_u; scaled_u[tap_at] = up_result;
      bf_u[tap_at] = uv; bf_silu[tap_at] = silu;
      value = silu * uv;
      bf_activation[tap_at] = value;
      if (!(up_scale > 0) || !gathered_mpp_finite(up_scale) || !gathered_mpp_finite(up_result)) gathered_mpp_error(diag, 4u);
    }
    if (!(scale > 0) || !gathered_mpp_finite(scale) || !gathered_mpp_finite(result) ||
        !gathered_mpp_finite(float(value))) gathered_mpp_error(diag, 4u);
    output[route * Width + n] = value;
  }
}
kernel void r1_finite_summary_gu_native_tap(
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]],
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]],
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]],
    device uint *diag [[buffer(8)]], constant FlashGatheredMPPParams &p [[buffer(9)]],
    device float *raw_g [[buffer(10)]],
    device float *scaled_g [[buffer(11)]],
    device float *raw_u [[buffer(12)]],
    device float *scaled_u [[buffer(13)]],
    device bfloat *bf_g [[buffer(14)]],
    device bfloat *bf_u [[buffer(15)]],
    device bfloat *bf_silu [[buffer(16)]],
    device bfloat *bf_activation [[buffer(17)]],
    device uint *completed [[buffer(18)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 total [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[2560]; threadgroup atomic_uint nonfinite;
  if (p.rows != 1 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= 10u || group.y || group.z >= 10u ||
      total.x != 10u || total.y != 1u || total.z != 10u ||
      threads.x != 128u || threads.y != 1u || threads.z != 1u) {
    if (!tid) gathered_mpp_error(diag,2u); return;
  }
  r1_finite_summary_private_gathered_mpp_execute_tap<true>(x,g,gs,u,us,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,raw_g,scaled_g,raw_u,scaled_u,bf_g,bf_u,bf_silu,bf_activation);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (!tid) completed[group.z * 10u + group.x] = 1u;
}
kernel void r1_finite_summary_gu_consumer_tap(
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]],
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]],
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]],
    device uint *diag [[buffer(8)]], constant FlashGatheredMPPParams &p [[buffer(9)]],
    device const uint *packet [[buffer(10)]],
    constant R1FiniteSummaryInvocation &v [[buffer(11)]],
    device float *raw_g [[buffer(12)]],
    device float *scaled_g [[buffer(13)]],
    device float *raw_u [[buffer(14)]],
    device float *scaled_u [[buffer(15)]],
    device bfloat *bf_g [[buffer(16)]],
    device bfloat *bf_u [[buffer(17)]],
    device bfloat *bf_silu [[buffer(18)]],
    device bfloat *bf_activation [[buffer(19)]],
    device uint *completed [[buffer(20)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 total [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[2560]; threadgroup atomic_uint nonfinite;
  if (p.rows != 1 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= 10u || group.y || group.z >= 10u ||
      total.x != 10u || total.y != 1u || total.z != 10u ||
      threads.x != 128u || threads.y != 1u || threads.z != 1u) {
    if (!tid) gathered_mpp_error(diag,2u); return;
  }
  const bool needsOriginalScan = r1_finite_summary_needs_scan(packet,v,1u,2560u,0);
  r1_finite_summary_private_gathered_mpp_execute_consumer_tap<true>(x,g,gs,u,us,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,needsOriginalScan,raw_g,scaled_g,raw_u,scaled_u,bf_g,bf_u,bf_silu,bf_activation);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (!tid) completed[group.z * 10u + group.x] = 1u;
}
kernel void r1_finite_summary_down_native_tap(
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]],
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]],
    device const long *ids [[buffer(4)]], device bfloat *out [[buffer(5)]],
    device uint *diag [[buffer(6)]], constant FlashGatheredMPPParams &p [[buffer(7)]],
    device float *raw_d [[buffer(8)]],
    device float *scaled_d [[buffer(9)]],
    device bfloat *bf_d [[buffer(10)]],
    device uint *completed [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 total [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[640]; threadgroup atomic_uint nonfinite;
  if (p.rows != 1 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= 40u || group.y || group.z >= 10u ||
      total.x != 40u || total.y != 1u || total.z != 10u ||
      threads.x != 128u || threads.y != 1u || threads.z != 1u) {
    if (!tid) gathered_mpp_error(diag,2u); return;
  }
  r1_finite_summary_private_gathered_mpp_execute_tap<false>(x,w,s,w,s,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,raw_d,scaled_d,nullptr,nullptr,bf_d,nullptr,nullptr,nullptr);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (!tid) completed[group.z * 40u + group.x] = 1u;
}
kernel void r1_finite_summary_down_consumer_tap(
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]],
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]],
    device const long *ids [[buffer(4)]], device bfloat *out [[buffer(5)]],
    device uint *diag [[buffer(6)]], constant FlashGatheredMPPParams &p [[buffer(7)]],
    device const uint *packet [[buffer(8)]],
    constant R1FiniteSummaryInvocation &v [[buffer(9)]],
    device float *raw_d [[buffer(10)]],
    device float *scaled_d [[buffer(11)]],
    device bfloat *bf_d [[buffer(12)]],
    device uint *completed [[buffer(13)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 total [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[640]; threadgroup atomic_uint nonfinite;
  if (p.rows != 1 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= 40u || group.y || group.z >= 10u ||
      total.x != 40u || total.y != 1u || total.z != 10u ||
      threads.x != 128u || threads.y != 1u || threads.z != 1u) {
    if (!tid) gathered_mpp_error(diag,2u); return;
  }
  const bool needsOriginalScan = r1_finite_summary_needs_scan(packet,v,2u,640u,group.z);
  r1_finite_summary_private_gathered_mpp_execute_consumer_tap<false>(x,w,s,w,s,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,needsOriginalScan,raw_d,scaled_d,nullptr,nullptr,bf_d,nullptr,nullptr,nullptr);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (!tid) completed[group.z * 40u + group.x] = 1u;
}
