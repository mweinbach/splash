// Q4 expert GEMMs for bucketed prefill/verify rows on the matrix units.
// Same bindings and job/route contract as flash_moe_direct_a_{gate_up,
// down_scatter}: A is read directly from the packed device rows (63 guard rows
// follow the last route), B is the original Q4/G64 expert weight. Each thread
// stages one contiguous run of codes with a single vector load and a single
// scale/bias pair, so staging costs one FMA per weight.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashMoEBlocked.h"
#include "metal/abi/FlashMoEBuckets.h"
using namespace metal;
using namespace mpp::tensor_ops;

inline bfloat opt_moe_mpp_sigmoid(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}

inline bool opt_moe_mpp_job(device const uint *offsets, device const FlashMoEBucketJob *jobs,
                            device const uint *job_count, uint job, uint capacity, uint routes,
                            thread uint &expert, thread uint &begin, thread uint &end) {
  const uint active = job_count[0];
  if (active > capacity || job >= active || offsets[512] > routes) return false;
  const FlashMoEBucketJob selected = jobs[job];
  if (selected.expert >= 512) return false;
  const uint b = offsets[selected.expert], e = offsets[selected.expert + 1];
  if (b > e || e > routes || selected.row_begin < b || selected.row_begin >= e) return false;
  expert = selected.expert; begin = selected.row_begin; end = e;
  return true;
}

// Stage a 64(N) x 64(K) bf16 tile (row n at stage[n * 64]) from Q4/G64 rows.
// THREADS threads; each covers CPT = 4096 / THREADS consecutive codes of a row.
template <uint THREADS>
inline void opt_moe_mpp_stage(device const uchar *weights, device const bfloat *scales,
                              device const bfloat *biases, ulong row_stride,
                              ulong param_row_stride, uint n0, uint k0, uint tid,
                              threadgroup bfloat *stage) {
  constexpr uint CPT = 4096 / THREADS;          // 32 (128 threads) or 16 (256)
  constexpr uint PER_ROW = 64 / CPT;
  const uint n = tid / PER_ROW, kk = (tid % PER_ROW) * CPT;
  const device uchar *src = weights + ulong(n0 + n) * row_stride + (k0 + kk) / 2;
  const ulong pi = ulong(n0 + n) * (param_row_stride / 2) + k0 / 64;
  const float s = float(scales[pi]), b = float(biases[pi]);
  threadgroup bfloat *dst = stage + n * 64 + kk;
  if (CPT == 32) {
    const uint4 w = *reinterpret_cast<const device uint4 *>(src);
    const uint words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (uint i = 0; i < 4; ++i)
#pragma unroll
      for (uint j = 0; j < 8; ++j)
        dst[i * 8 + j] = bfloat(fma(float((words[i] >> (4 * j)) & 15u), s, b));
  } else {
    const uint2 w = *reinterpret_cast<const device uint2 *>(src);
    const uint words[2] = {w.x, w.y};
#pragma unroll
    for (uint i = 0; i < 2; ++i)
#pragma unroll
      for (uint j = 0; j < 8; ++j)
        dst[i * 8 + j] = bfloat(fma(float((words[i] >> (4 * j)) & 15u), s, b));
  }
}

template <ushort M, ushort SG>
inline void opt_moe_q4_gate_up_tile(
    device bfloat *input, device const uchar *gw, device const bfloat *gs,
    device const bfloat *gb, device const uchar *uw, device const bfloat *us,
    device const bfloat *ub, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device bfloat *output, constant FlashMoEBlockedGateParams &params, uint3 group,
    uint tid, threadgroup bfloat *sg, threadgroup bfloat *su) {
  const constant FlashMoEFusedParams &p = params.affine;
  uint expert = 0, begin = 0, end = 0;
  if (!opt_moe_mpp_job(offsets, jobs, job_count, group.y, params.job_capacity,
                       params.route_capacity, expert, begin, end)) return;
  constexpr ushort N = 64, BK = 64;
  const uint n0 = group.x * N;
  auto a = tensor(input + ulong(begin) * 2560, dextents<int, 2>{2560, M}, array<int, 2>{1, 2560});
  auto g = tensor(sg, dextents<int, 2>{BK, N}, array<int, 2>{1, BK});
  auto u = tensor(su, dextents<int, 2>{BK, N}, array<int, 2>{1, BK});
  auto g0 = g.template slice<BK, N>(0, 0);
  auto u0 = u.template slice<BK, N>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(M, N, BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<SG>> op;
  auto a0 = a.template slice<BK, M>(0, 0);
  auto gacc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(g0), float>();
  auto uacc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(u0), float>();
#pragma unroll
  for (ushort i = 0; i < gacc.get_capacity(); ++i)
    if (gacc.is_valid_element(i)) { gacc[i] = 0.0f; uacc[i] = 0.0f; }
  const device uchar *gwe = gw + ulong(expert) * p.gate_weight_expert_stride_bytes;
  const device uchar *uwe = uw + ulong(expert) * p.up_weight_expert_stride_bytes;
  const device bfloat *gse = gs + ulong(expert) * (p.gate_parameter_expert_stride_bytes / 2);
  const device bfloat *gbe = gb + ulong(expert) * (p.gate_parameter_expert_stride_bytes / 2);
  const device bfloat *use = us + ulong(expert) * (p.up_parameter_expert_stride_bytes / 2);
  const device bfloat *ube = ub + ulong(expert) * (p.up_parameter_expert_stride_bytes / 2);
  // Double-buffered staging: chunk c+1 is staged while chunk c multiplies.
  auto g1 = tensor(sg + 64 * 64, dextents<int, 2>{BK, N}, array<int, 2>{1, BK}).template slice<BK, N>(0, 0);
  auto u1 = tensor(su + 64 * 64, dextents<int, 2>{BK, N}, array<int, 2>{1, BK}).template slice<BK, N>(0, 0);
  opt_moe_mpp_stage<SG * 32>(gwe, gse, gbe, p.gate_weight_row_stride_bytes,
      p.gate_parameter_row_stride_bytes, n0, 0, tid, sg);
  opt_moe_mpp_stage<SG * 32>(uwe, use, ube, p.up_weight_row_stride_bytes,
      p.up_parameter_row_stride_bytes, n0, 0, tid, su);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint chunk = 0; chunk < 40; ++chunk) {
    const uint k0 = chunk * BK;
    const uint next = (chunk + 1) & 1;
    if (chunk + 1 < 40) {
      opt_moe_mpp_stage<SG * 32>(gwe, gse, gbe, p.gate_weight_row_stride_bytes,
          p.gate_parameter_row_stride_bytes, n0, k0 + BK, tid, sg + next * 64 * 64);
      opt_moe_mpp_stage<SG * 32>(uwe, use, ube, p.up_weight_row_stride_bytes,
          p.up_parameter_row_stride_bytes, n0, k0 + BK, tid, su + next * 64 * 64);
    }
    auto ac = a.template slice<BK, M>(k0, 0);
    if (chunk & 1) { op.run(ac, g1, gacc); op.run(ac, u1, uacc); }
    else { op.run(ac, g0, gacc); op.run(ac, u0, uacc); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i = 0; i < gacc.get_capacity(); ++i) {
    if (!gacc.is_valid_element(i)) continue;
    const auto index = gacc.get_multidimensional_index(i);
    const uint row = begin + index[1], n = n0 + index[0];
    if (row >= end) continue;
    const bfloat gate = bfloat(gacc[i]), up = bfloat(uacc[i]);
    const bfloat silu = bfloat(float(gate) * float(opt_moe_mpp_sigmoid(gate)));
    output[ulong(row) * 640 + n] = bfloat(float(silu) * float(up));
  }
}

template <ushort M, ushort SG>
inline void opt_moe_q4_down_tile(
    device bfloat *input, device const uchar *w, device const bfloat *s,
    device const bfloat *b, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device const uint *route_map, device bfloat *output,
    constant FlashMoEBlockedDownParams &params, uint3 group, uint tid,
    threadgroup bfloat *sb) {
  const constant FlashMoEDownFusedParams &p = params.affine;
  uint expert = 0, begin = 0, end = 0;
  if (!opt_moe_mpp_job(offsets, jobs, job_count, group.y, params.job_capacity,
                       params.route_capacity, expert, begin, end)) return;
  constexpr ushort N = 64, BK = 64;
  const uint n0 = group.x * N;
  auto a = tensor(input + ulong(begin) * 640, dextents<int, 2>{640, M}, array<int, 2>{1, 640});
  auto bt = tensor(sb, dextents<int, 2>{BK, N}, array<int, 2>{1, BK});
  auto b0 = bt.template slice<BK, N>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(M, N, BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<SG>> op;
  auto a0 = a.template slice<BK, M>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
  const device uchar *we = w + ulong(expert) * p.weight_expert_stride_bytes;
  const device bfloat *se = s + ulong(expert) * (p.parameter_expert_stride_bytes / 2);
  const device bfloat *be = b + ulong(expert) * (p.parameter_expert_stride_bytes / 2);
  auto b1 = tensor(sb + 64 * 64, dextents<int, 2>{BK, N}, array<int, 2>{1, BK}).template slice<BK, N>(0, 0);
  opt_moe_mpp_stage<SG * 32>(we, se, be, p.weight_row_stride_bytes,
      p.parameter_row_stride_bytes, n0, 0, tid, sb);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint chunk = 0; chunk < 10; ++chunk) {
    const uint k0 = chunk * BK;
    if (chunk + 1 < 10)
      opt_moe_mpp_stage<SG * 32>(we, se, be, p.weight_row_stride_bytes,
          p.parameter_row_stride_bytes, n0, k0 + BK, tid, sb + ((chunk + 1) & 1) * 64 * 64);
    auto ac = a.template slice<BK, M>(k0, 0);
    if (chunk & 1) op.run(ac, b1, acc); else op.run(ac, b0, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    const uint row = begin + index[1], n = n0 + index[0];
    if (row >= end) continue;
    const uint route = route_map[row];
    if (route >= params.route_capacity) continue;
    output[ulong(route) * 2560 + n] = bfloat(acc[i]);
  }
}

#define OPT_MOE_Q4_GATE(NAME, M, SG)                                                   \
kernel void NAME(                                                                    \
    device bfloat *input [[buffer(0)]],                                              \
    device const uchar *gw [[buffer(1)]], device const bfloat *gs [[buffer(2)]],     \
    device const bfloat *gb [[buffer(3)]], device const uchar *uw [[buffer(4)]],     \
    device const bfloat *us [[buffer(5)]], device const bfloat *ub [[buffer(6)]],    \
    device const uint *offsets [[buffer(7)]],                                        \
    device const FlashMoEBucketJob *jobs [[buffer(8)]],                              \
    device const uint *job_count [[buffer(9)]],                                      \
    device bfloat *output [[buffer(10)]], device uint *diag [[buffer(11)]],          \
    constant FlashMoEBlockedGateParams &p [[buffer(12)]],                            \
    uint3 group [[threadgroup_position_in_grid]],                                    \
    uint tid [[thread_index_in_threadgroup]]) {                                      \
  alignas(16) threadgroup bfloat sg[2 * 64 * 64], su[2 * 64 * 64];                   \
  opt_moe_q4_gate_up_tile<M, SG>(input, gw, gs, gb, uw, us, ub, offsets, jobs,       \
      job_count, output, p, group, tid, sg, su);                                     \
}
OPT_MOE_Q4_GATE(opt_moe_q4_gate_up_m16_n64, 16, 4)
OPT_MOE_Q4_GATE(opt_moe_q4_gate_up_m32_n64, 32, 4)
OPT_MOE_Q4_GATE(opt_moe_q4_gate_up_m64_n64_sg8, 64, 8)

#define OPT_MOE_Q4_DOWN(NAME, M, SG)                                                   \
kernel void NAME(                                                                    \
    device bfloat *input [[buffer(0)]], device const uchar *w [[buffer(1)]],         \
    device const bfloat *s [[buffer(2)]], device const bfloat *b [[buffer(3)]],      \
    device const uint *offsets [[buffer(4)]],                                        \
    device const FlashMoEBucketJob *jobs [[buffer(5)]],                              \
    device const uint *job_count [[buffer(6)]], device const uint *map [[buffer(7)]],\
    device bfloat *output [[buffer(8)]], device uint *diag [[buffer(9)]],            \
    constant FlashMoEBlockedDownParams &p [[buffer(10)]],                            \
    uint3 group [[threadgroup_position_in_grid]],                                    \
    uint tid [[thread_index_in_threadgroup]]) {                                      \
  alignas(16) threadgroup bfloat sb[2 * 64 * 64];                                    \
  opt_moe_q4_down_tile<M, SG>(input, w, s, b, offsets, jobs, job_count, map, output, \
      p, group, tid, sb);                                                            \
}
OPT_MOE_Q4_DOWN(opt_moe_q4_down_scatter_m16_n64, 16, 4)
OPT_MOE_Q4_DOWN(opt_moe_q4_down_scatter_m32_n64, 32, 4)
OPT_MOE_Q4_DOWN(opt_moe_q4_down_scatter_m64_n64_sg8, 64, 8)

// ---------------------------------------------------------------------------
// Direct 4-bit variant: the matrix units read the original packed Q4 codes
// (uint4b_format) straight from the MLX row-major expert weight; each 64-wide
// quant group applies acc += s * (A . q) + b * sum(A) in F32. Per-tile
// scales/biases are staged once; row sums are precomputed per group.

// sums[row, g] = sum_k A[row, g*64 + k] for every packed row including guards.
kernel void opt_moe_group_sums(
    device const bfloat *input [[buffer(0)]], device float *sums [[buffer(1)]],
    constant uint2 &shape [[buffer(2)]],   // {rows, K}
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint K = shape.y, groups = K / 64;
  if (row >= shape.x) return;
  for (uint g = lane; g < groups; g += 32) {
    const device bfloat4 *p = reinterpret_cast<const device bfloat4 *>(input + ulong(row) * K + g * 64);
    float s = 0.0f;
#pragma unroll
    for (uint i = 0; i < 16; ++i) { const bfloat4 v = p[i]; s += float(v.x) + float(v.y) + float(v.z) + float(v.w); }
    sums[ulong(row) * groups + g] = s;
  }
}

template <ushort M, ushort SG, uint K, bool GATE_UP>
inline void opt_moe_q4d_tile(
    device bfloat *input, device const float *row_sums,
    device const uchar *w0, device const bfloat *s0, device const bfloat *b0,
    device const uchar *w1, device const bfloat *s1, device const bfloat *b1,
    ulong w_row, ulong w_expert, ulong p_row, ulong p_expert,
    device const uint *offsets, device const FlashMoEBucketJob *jobs,
    device const uint *job_count, uint job_capacity, uint route_capacity,
    device const uint *route_map, device bfloat *output,
    uint3 group, uint tid, threadgroup bfloat *params) {
  constexpr uint N = 64, G = K / 64, OUT = GATE_UP ? 640 : 2560;
  uint expert = 0, begin = 0, end = 0;
  if (!opt_moe_mpp_job(offsets, jobs, job_count, group.y, job_capacity, route_capacity,
                       expert, begin, end)) return;
  const uint n0 = group.x * N;
  // Stage this tile's scale/bias: params[(m * 64 + n) * G + g], m in {s0,b0,s1,b1}.
  const ulong pe = ulong(expert) * (p_expert / 2), pr = p_row / 2;
  for (uint i = tid; i < N * G; i += SG * 32) {
    const uint n = i / G, g = i % G;
    const ulong src = pe + ulong(n0 + n) * pr + g;
    params[(0 * N + n) * G + g] = s0[src];
    params[(1 * N + n) * G + g] = b0[src];
    if (GATE_UP) {
      params[(2 * N + n) * G + g] = s1[src];
      params[(3 * N + n) * G + g] = b1[src];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto a = tensor(input + ulong(begin) * K, dextents<int, 2>{int(K), M}, array<int, 2>{1, int(K)});
  constexpr auto descriptor = matmul2d_descriptor(M, N, 64, false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> op;
  const device uchar *wt0 = w0 + ulong(expert) * w_expert + ulong(n0) * w_row;
  const device uchar *wt1 = w1 + ulong(expert) * w_expert + ulong(n0) * w_row;
  const int row_elems = int(w_row * 2);
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bt0(
      const_cast<device uchar *>(wt0),
      dextents<int, 2>{int(K), int(N)}, array<int, 2>{1, row_elems});
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bt1(
      const_cast<device uchar *>(wt1),
      dextents<int, 2>{int(K), int(N)}, array<int, 2>{1, row_elems});
  auto a0 = a.template slice<64, M>(0, 0);
  auto bs0 = bt0.template slice<64, N>(0, 0);
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
#pragma unroll
  for (ushort i = 0; i < acc0.get_capacity(); ++i)
    if (acc0.is_valid_element(i)) { acc0[i] = 0.0f; acc1[i] = 0.0f; }
  for (uint g = 0; g < G; ++g) {
    auto ag = a.template slice<64, M>(g * 64, 0);
    auto bg0 = bt0.template slice<64, N>(g * 64, 0);
    auto p0 = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
    op.run(ag, bg0, p0);
    decltype(p0) p1;
    if (GATE_UP) {
      auto bg1 = bt1.template slice<64, N>(g * 64, 0);
      p1 = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
      op.run(ag, bg1, p1);
    }
#pragma unroll
    for (ushort i = 0; i < acc0.get_capacity(); ++i) {
      if (!acc0.is_valid_element(i)) continue;
      const auto index = acc0.get_multidimensional_index(i);
      const uint n = index[0], m = index[1];
      const float rs = row_sums[ulong(begin + m) * G + g];
      acc0[i] += p0[i] * float(params[(0 * N + n) * G + g]) + rs * float(params[(1 * N + n) * G + g]);
      if (GATE_UP)
        acc1[i] += p1[i] * float(params[(2 * N + n) * G + g]) + rs * float(params[(3 * N + n) * G + g]);
    }
  }
#pragma unroll
  for (ushort i = 0; i < acc0.get_capacity(); ++i) {
    if (!acc0.is_valid_element(i)) continue;
    const auto index = acc0.get_multidimensional_index(i);
    const uint row = begin + index[1], n = n0 + index[0];
    if (row >= end) continue;
    if (GATE_UP) {
      const bfloat gate = bfloat(acc0[i]), up = bfloat(acc1[i]);
      const bfloat silu = bfloat(float(gate) * float(opt_moe_mpp_sigmoid(gate)));
      output[ulong(row) * OUT + n] = bfloat(float(silu) * float(up));
    } else {
      const uint route = route_map[row];
      if (route >= route_capacity) continue;
      output[ulong(route) * OUT + n] = bfloat(acc0[i]);
    }
  }
}

#define OPT_MOE_Q4D_GATE(NAME, M, SG)                                                  \
kernel void NAME(                                                                    \
    device bfloat *input [[buffer(0)]],                                              \
    device const uchar *gw [[buffer(1)]], device const bfloat *gs [[buffer(2)]],     \
    device const bfloat *gb [[buffer(3)]], device const uchar *uw [[buffer(4)]],     \
    device const bfloat *us [[buffer(5)]], device const bfloat *ub [[buffer(6)]],    \
    device const uint *offsets [[buffer(7)]],                                        \
    device const FlashMoEBucketJob *jobs [[buffer(8)]],                              \
    device const uint *job_count [[buffer(9)]],                                      \
    device bfloat *output [[buffer(10)]], device const float *sums [[buffer(11)]],   \
    constant FlashMoEBlockedGateParams &p [[buffer(12)]],                            \
    uint3 group [[threadgroup_position_in_grid]],                                    \
    uint tid [[thread_index_in_threadgroup]]) {                                      \
  threadgroup bfloat params[4 * 64 * 40];                                            \
  opt_moe_q4d_tile<M, SG, 2560, true>(input, sums, gw, gs, gb, uw, us, ub,           \
      p.affine.gate_weight_row_stride_bytes, p.affine.gate_weight_expert_stride_bytes,\
      p.affine.gate_parameter_row_stride_bytes, p.affine.gate_parameter_expert_stride_bytes,\
      offsets, jobs, job_count, p.job_capacity, p.route_capacity, offsets, output,   \
      group, tid, params);                                                           \
}
OPT_MOE_Q4D_GATE(opt_moe_q4d_gate_up_m16_n64, 16, 4)
OPT_MOE_Q4D_GATE(opt_moe_q4d_gate_up_m32_n64, 32, 4)
OPT_MOE_Q4D_GATE(opt_moe_q4d_gate_up_m64_n64_sg8, 64, 8)

#define OPT_MOE_Q4D_DOWN(NAME, M, SG)                                                  \
kernel void NAME(                                                                    \
    device bfloat *input [[buffer(0)]], device const uchar *w [[buffer(1)]],         \
    device const bfloat *s [[buffer(2)]], device const bfloat *b [[buffer(3)]],      \
    device const uint *offsets [[buffer(4)]],                                        \
    device const FlashMoEBucketJob *jobs [[buffer(5)]],                              \
    device const uint *job_count [[buffer(6)]], device const uint *map [[buffer(7)]],\
    device bfloat *output [[buffer(8)]], device const float *sums [[buffer(9)]],     \
    constant FlashMoEBlockedDownParams &p [[buffer(10)]],                            \
    uint3 group [[threadgroup_position_in_grid]],                                    \
    uint tid [[thread_index_in_threadgroup]]) {                                      \
  threadgroup bfloat params[2 * 64 * 10];                                            \
  opt_moe_q4d_tile<M, SG, 640, false>(input, sums, w, s, b, w, s, b,                 \
      p.affine.weight_row_stride_bytes, p.affine.weight_expert_stride_bytes,         \
      p.affine.parameter_row_stride_bytes, p.affine.parameter_expert_stride_bytes,   \
      offsets, jobs, job_count, p.job_capacity, p.route_capacity, map, output,       \
      group, tid, params);                                                           \
}
OPT_MOE_Q4D_DOWN(opt_moe_q4d_down_scatter_m16_n64, 16, 4)
OPT_MOE_Q4D_DOWN(opt_moe_q4d_down_scatter_m32_n64, 32, 4)
OPT_MOE_Q4D_DOWN(opt_moe_q4d_down_scatter_m64_n64_sg8, 64, 8)
