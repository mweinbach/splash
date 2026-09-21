// Private offline signed INT8 expert candidate: original BF16 activations,
// genuine signed INT8 matrix operands, F32 accumulation, F32 K64 group scales.
// Nothing in production selects this candidate.
#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "flash_expert_int8_candidate.h"
#include "metal/kernels/common/flash_affine_mpp_common.h"
#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
using namespace metal;
using namespace mpp::tensor_ops;

#pragma METAL fp math_mode(fast)
inline bfloat expert_int8_g64_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

template <ushort M, ushort SG>
inline bool expert_int8_g64_job(constant FlashExpertInt8Params &p,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device uint *diag, uint3 group, uint3 threads, uint tid,
    thread uint &rank, thread uint &begin, thread uint &valid_rows) {
  if (!p.rows || p.rows > kFlashMoEBucketMaximumRows || !p.selections ||
      p.selections > kFlashMoEBucketMaximumSelections ||
      p.route_capacity != p.rows * p.selections ||
      p.job_capacity != (p.route_capacity + M - 1) / M + 511 ||
      p.tile_rows != M || !p.stored_experts || p.stored_experts > 512 ||
      p.scale_group_size != 64 || p.reserved || group.y >= p.job_capacity || group.z ||
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
  if (rank >= p.stored_experts) { if (!tid) flash_mpp_error(diag, 1u); return false; }
  begin = job.row_begin;
  valid_rows = min(uint(M), end - begin);
  return true;
}

template <ushort M, ushort SG>
inline void expert_int8_g64_gate(device bfloat *input, device int8_t *gate,
    device const float *gate_scale, device int8_t *up, device const float *up_scale,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device bfloat *output, device uint *diag, constant FlashExpertInt8Params &p,
    uint3 group, uint3 threads, uint tid) {
  if (group.x >= 10) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!expert_int8_g64_job<M, SG>(p, ranks, offsets, jobs, job_count, diag,
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
  constexpr auto descriptor = matmul2d_descriptor(M, N, 64,
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto gd = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  auto ud = operation.template get_destination_cooperative_tensor<decltype(a), decltype(u), float>();
#pragma unroll
  for (ushort i = 0; i < gd.get_capacity(); ++i)
    if (gd.is_valid_element(i)) { gd[i] = 0.0f; ud[i] = 0.0f; }
  for (uint chunk = 0; chunk < 40; ++chunk) {
    auto part_g = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
    auto part_u = operation.template get_destination_cooperative_tensor<decltype(a), decltype(u), float>();
    auto ac = a.slice(chunk * 64, 0);
    auto gc = g.slice(chunk * 64, 0);
    auto uc = u.slice(chunk * 64, 0);
    operation.run(ac, gc, part_g);
    operation.run(ac, uc, part_u);
#pragma unroll
    for (ushort i = 0; i < gd.get_capacity(); ++i) {
      if (!gd.is_valid_element(i)) continue;
      const auto index = gd.get_multidimensional_index(i);
      if (uint(index[1]) >= valid_rows) continue;
      const uint n = column + index[0];
      const float gs = gate_scale[(ulong(rank) * 640 + n) * 40 + chunk];
      const float us = up_scale[(ulong(rank) * 640 + n) * 40 + chunk];
      if (!(gs > 0.0f) || !(us > 0.0f) || !flash_mpp_finite(gs) || !flash_mpp_finite(us))
        flash_mpp_error(diag, 4u);
      gd[i] += part_g[i] * gs;
      ud[i] += part_u[i] * us;
    }
  }
#pragma unroll
  for (ushort i = 0; i < gd.get_capacity(); ++i) {
    if (!gd.is_valid_element(i)) continue;
    const auto index = gd.get_multidimensional_index(i);
    if (uint(index[1]) >= valid_rows) continue;
    const uint n = column + index[0];
    const float gf = gd[i], uf = ud[i];
    const bfloat gv = bfloat(gf), uv = bfloat(uf);
    const bfloat silu = gv * expert_int8_g64_sigmoid(gv);
    const bfloat value = silu * uv;
    if (!flash_mpp_finite(gf) || !flash_mpp_finite(uf) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(begin + index[1]) * 640 + n] = value;
  }
}

template <ushort M, ushort SG>
inline void expert_int8_g64_down(device bfloat *input, device int8_t *weights,
    device const float *scales, device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device const uint *route_map, device bfloat *output, device uint *diag,
    constant FlashExpertInt8Params &p, uint3 group, uint3 threads, uint tid) {
  if (group.x >= 40) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!expert_int8_g64_job<M, SG>(p, ranks, offsets, jobs, job_count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  constexpr ushort N = 64;
  const uint column = group.x * N;
  auto a = tensor(input + ulong(begin) * 640,
      dextents<int, 2>{640, int(valid_rows)}, array<int, 2>{1, 640});
  auto b = tensor(weights + (ulong(rank) * 2560 + column) * 640,
      dextents<int, 2>{640, N}, array<int, 2>{1, 640});
  constexpr auto descriptor = matmul2d_descriptor(M, N, 64,
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i)
    if (dot.is_valid_element(i)) dot[i] = 0.0f;
  for (uint chunk = 0; chunk < 10; ++chunk) {
    auto partial = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    auto ac = a.slice(chunk * 64, 0);
    auto bc = b.slice(chunk * 64, 0);
    operation.run(ac, bc, partial);
#pragma unroll
    for (ushort i = 0; i < dot.get_capacity(); ++i) {
      if (!dot.is_valid_element(i)) continue;
      const auto index = dot.get_multidimensional_index(i);
      if (uint(index[1]) >= valid_rows) continue;
      const uint n = column + index[0];
      const float scale = scales[(ulong(rank) * 2560 + n) * 10 + chunk];
      if (!(scale > 0.0f) || !flash_mpp_finite(scale)) flash_mpp_error(diag, 4u);
      dot[i] += partial[i] * scale;
    }
  }
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    if (uint(index[1]) >= valid_rows) continue;
    const uint route = route_map[begin + index[1]], n = column + index[0];
    if (route >= p.route_capacity) { flash_mpp_error(diag, 1u); continue; }
    const float result = dot[i];
    const bfloat value = bfloat(result);
    if (!flash_mpp_finite(result) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(route) * 2560 + n] = value;
  }
}

#define INT8_GATE(NAME, M, SG) \
kernel void NAME(device bfloat *a [[buffer(0)]], device int8_t *g [[buffer(1)]], \
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]], \
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]], \
    device const uint *offsets [[buffer(6)]], device const FlashMoEBucketJob *jobs [[buffer(7)]], \
    device const uint *count [[buffer(8)]], device bfloat *out [[buffer(9)]], \
    device uint *diag [[buffer(10)]], constant FlashExpertInt8Params &p [[buffer(11)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  expert_int8_g64_gate<M, SG>(a, g, gs, u, us, ranks, offsets, jobs, count, out, diag, p, group, threads, tid); \
}
#define INT8_DOWN(NAME, M, SG) \
kernel void NAME(device bfloat *a [[buffer(0)]], device int8_t *w [[buffer(1)]], \
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]], \
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]], \
    device const uint *count [[buffer(6)]], device const uint *map [[buffer(7)]], \
    device bfloat *out [[buffer(8)]], device uint *diag [[buffer(9)]], \
    constant FlashExpertInt8Params &p [[buffer(10)]], uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { \
  expert_int8_g64_down<M, SG>(a, w, s, ranks, offsets, jobs, count, map, out, diag, p, group, threads, tid); \
}
INT8_GATE(flash_expert_int8_g64_gate_up_m16_n64, 16, 4)
INT8_GATE(flash_expert_int8_g64_gate_up_m32_n64, 32, 4)
INT8_GATE(flash_expert_int8_g64_gate_up_m64_n64_sg8, 64, 8)
INT8_DOWN(flash_expert_int8_g64_down_scatter_m16_n64, 16, 4)
INT8_DOWN(flash_expert_int8_g64_down_scatter_m32_n64, 32, 4)
INT8_DOWN(flash_expert_int8_g64_down_scatter_m64_n64_sg8, 64, 8)
#undef INT8_GATE
#undef INT8_DOWN
#endif
