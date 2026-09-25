// Prototype: gathered-expert GEMV on the matrix units for decode windows.
// For each unique expert u the whole window x[R, K] is multiplied by W_u^T
// (packed Q4 codes read directly by the matrix units, per-group scale/bias
// epilogue). Rows that did not select u are computed and discarded.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct MkPlanLite {
  uint rows, unique, pad0, pad1;
  uint experts[80];
  uint row_mask[80];   // bit r set when row r selected experts[u]
};

struct GUParams {
  uint K, N, rows, pad;
  ulong w_row_stride, w_expert_stride, p_row_stride, p_expert_stride; // p_* in elements
};

inline bfloat mk_sigmoid_fast(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}

// Grid: (N / NT tiles, unique experts). TG = SK simdgroups splitting K.
// MODE 1: gate+up with SiLU*up epilogue -> out[u][r][n] (BF16).
// MODE 0: single matrix -> out[u][r][n].
template <int M, int NT, int SK, int MODE, int EPI>
[[kernel]] void mpp_gathered(
    const device bfloat *x [[buffer(0)]],            // [U or 1][M rows][K] when MODE 0 per-expert, else [rows][K]
    const device uchar *w0 [[buffer(1)]], const device bfloat *s0 [[buffer(2)]], const device bfloat *b0 [[buffer(3)]],
    const device uchar *w1 [[buffer(4)]], const device bfloat *s1 [[buffer(5)]], const device bfloat *b1 [[buffer(6)]],
    const device MkPlanLite *plan [[buffer(7)]],
    device bfloat *out [[buffer(8)]],
    constant GUParams &p [[buffer(9)]],
    const device bfloat *st0 [[buffer(10)]], const device bfloat *st1 [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr int G = 64;
  const uint u = tg.y;
  if (u >= plan->unique) return;
  const uint expert = plan->experts[u];
  const uint ng = p.K / G;
  threadgroup float xsum[40 * M];
  threadgroup bfloat xsh[(EPI >= 4) ? 2 * M * 64 : 1];
  threadgroup float red[(SK > 1) ? SK * (MODE == 1 ? 2 : 1) * M * NT : 1];
  const device bfloat *xr = MODE == 1 ? x : x + ulong(u) * M * p.K;
  for (uint i = tid; i < ng * M; i += SK * 32) {
    const uint g = i / M, m = i % M;
    float sum = 0.0f;
    if (m < p.rows) {
      const device bfloat4 *q = reinterpret_cast<const device bfloat4 *>(xr + ulong(m) * p.K + g * G);
      for (int j = 0; j < G / 4; ++j) { const float4 v = float4(q[j]); sum += v.x + v.y + v.z + v.w; }
    }
    xsum[i] = sum;
    if (EPI >= 4) { const bfloat hi = bfloat(sum); xsh[m * 64 + g] = hi; xsh[M * 64 + m * 64 + g] = bfloat(sum - float(hi)); }
  }
  if (EPI >= 4) for (uint i = tid; i < uint(M * (64 - 40)); i += SK * 32) { const uint m = i / 24, g = 40 + i % 24; xsh[m * 64 + g] = 0; xsh[M * 64 + m * 64 + g] = 0; }
  const uint n0 = tg.x * NT;
  threadgroup bfloat sst[(EPI == 1) ? 2 * (MODE == 1 ? 2 : 1) * NT * 40 : 1];
  if (EPI == 1) {
    const device bfloat *src[4] = {s0, b0, s1, b1};
    for (uint i = tid; i < uint((MODE == 1 ? 4 : 2) * NT) * ng; i += SK * 32) {
      const uint which = i / (NT * ng), rem = i % (NT * ng), n = rem / ng, g = rem % ng;
      sst[(which * NT + n) * 40 + g] = src[which][ulong(expert) * p.p_expert_stride + ulong(n0 + n) * p.p_row_stride + g];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto a = tensor(const_cast<device bfloat *>(xr), dextents<int, 2>{int(p.K), int(p.rows)},
                  array<int, 2>{1, int(p.K)});
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bw0(
      const_cast<device uchar *>(w0 + ulong(expert) * p.w_expert_stride + ulong(n0) * p.w_row_stride),
      dextents<int, 2>{int(p.K), NT}, array<int, 2>{1, int(p.w_row_stride * 2)});
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bw1(
      const_cast<device uchar *>(w1 + ulong(expert) * p.w_expert_stride + ulong(n0) * p.w_row_stride),
      dextents<int, 2>{int(p.K), NT}, array<int, 2>{1, int(p.w_row_stride * 2)});
  constexpr auto desc = matmul2d_descriptor(M, NT, G, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto a0 = a.template slice<G, M>(0, 0);
  auto bs0 = bw0.template slice<G, NT>(0, 0);
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
#pragma unroll
  for (ushort i = 0; i < acc0.get_capacity(); ++i) if (acc0.is_valid_element(i)) { acc0[i] = 0.0f; acc1[i] = 0.0f; }
  const device bfloat *se0 = s0 + ulong(expert) * p.p_expert_stride, *be0 = b0 + ulong(expert) * p.p_expert_stride;
  const device bfloat *se1 = s1 + ulong(expert) * p.p_expert_stride, *be1 = b1 + ulong(expert) * p.p_expert_stride;
  for (uint g = simd; g < ng; g += SK) {
    auto ag = a.template slice<G, M>(g * G, 0);
    {
      auto bg = bw0.template slice<G, NT>(g * G, 0);
      auto t = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
      op.run(ag, bg, t);
#pragma unroll
      for (ushort i = 0; i < acc0.get_capacity(); ++i) {
        if (!acc0.is_valid_element(i)) continue;
        const auto idx = acc0.get_multidimensional_index(i);
        const uint n = n0 + idx[0], m = idx[1];
        if (EPI == 2) { acc0[i] += t[i]; continue; }
        if (EPI == 3 && m >= p.rows) continue;
        float sc, bi;
        if (EPI == 4) { acc0[i] = fma(t[i], float(se0[ulong(n) * p.p_row_stride + g]), acc0[i]); continue; }
        if (EPI == 5) { acc0[i] = fma(t[i], float(st0[ulong(expert) * p.p_expert_stride + ulong(g) * p.N + n]), acc0[i]); continue; }
        if (EPI == 1) { sc = float(sst[(0 * NT + idx[0]) * 40 + g]); bi = float(sst[(1 * NT + idx[0]) * 40 + g]); }
        else { const ulong pi = ulong(n) * p.p_row_stride + g; sc = float(se0[pi]); bi = float(be0[pi]); }
        acc0[i] = fma(t[i], sc, fma(xsum[g * M + m], bi, acc0[i]));
      }
    }
    if (MODE == 1) {
      auto bg = bw1.template slice<G, NT>(g * G, 0);
      auto t = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
      op.run(ag, bg, t);
#pragma unroll
      for (ushort i = 0; i < acc1.get_capacity(); ++i) {
        if (!acc1.is_valid_element(i)) continue;
        const auto idx = acc1.get_multidimensional_index(i);
        const uint n = n0 + idx[0], m = idx[1];
        if (EPI == 2) { acc1[i] += t[i]; continue; }
        if (EPI == 3 && m >= p.rows) continue;
        float sc, bi;
        if (EPI == 4) { acc1[i] = fma(t[i], float(se1[ulong(n) * p.p_row_stride + g]), acc1[i]); continue; }
        if (EPI == 5) { acc1[i] = fma(t[i], float(st1[ulong(expert) * p.p_expert_stride + ulong(g) * p.N + n]), acc1[i]); continue; }
        if (EPI == 1) { sc = float(sst[(2 * NT + idx[0]) * 40 + g]); bi = float(sst[(3 * NT + idx[0]) * 40 + g]); }
        else { const ulong pi = ulong(n) * p.p_row_stride + g; sc = float(se1[pi]); bi = float(be1[pi]); }
        acc1[i] = fma(t[i], sc, fma(xsum[g * M + m], bi, acc1[i]));
      }
    }
  }
  if (EPI >= 4) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto bdesc = matmul2d_descriptor(M, NT, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<bdesc, execution_simdgroups<1>> bop;
    auto xh = tensor(xsh, dextents<int, 2>{64, M}, array<int, 2>{1, 64});
    auto xl = tensor(xsh + M * 64, dextents<int, 2>{64, M}, array<int, 2>{1, 64});
    if (simd == 0) {
      auto xhs = xh.template slice<64, M>(0, 0);
      auto xls = xl.template slice<64, M>(0, 0);
      auto bb0 = tensor(const_cast<device bfloat *>(be0 + ulong(expert) * p.p_expert_stride + ulong(n0) * p.p_row_stride), dextents<int, 2>{40, NT}, array<int, 2>{1, int(p.p_row_stride)});
      auto bs = bb0.template slice<64, NT>(0, 0);
      auto tb = bop.template get_destination_cooperative_tensor<decltype(xhs), decltype(bs), float>();
#pragma unroll
      for (ushort i = 0; i < tb.get_capacity(); ++i) if (tb.is_valid_element(i)) tb[i] = 0.0f;
      bop.run(xhs, bs, tb);
      bop.run(xls, bs, tb);
#pragma unroll
      for (ushort i = 0; i < acc0.get_capacity(); ++i) if (acc0.is_valid_element(i)) acc0[i] += tb[i];
      if (MODE == 1) {
        auto bb1 = tensor(const_cast<device bfloat *>(be1 + ulong(expert) * p.p_expert_stride + ulong(n0) * p.p_row_stride), dextents<int, 2>{40, NT}, array<int, 2>{1, int(p.p_row_stride)});
        auto bs1 = bb1.template slice<64, NT>(0, 0);
        auto tb1 = bop.template get_destination_cooperative_tensor<decltype(xhs), decltype(bs1), float>();
#pragma unroll
        for (ushort i = 0; i < tb1.get_capacity(); ++i) if (tb1.is_valid_element(i)) tb1[i] = 0.0f;
        bop.run(xhs, bs1, tb1);
        bop.run(xls, bs1, tb1);
#pragma unroll
        for (ushort i = 0; i < acc1.get_capacity(); ++i) if (acc1.is_valid_element(i)) acc1[i] += tb1[i];
      }
    }
  }
  const uint mask = plan->row_mask[u];
  if (SK > 1) {
#pragma unroll
    for (ushort i = 0; i < acc0.get_capacity(); ++i) {
      if (!acc0.is_valid_element(i)) continue;
      const auto idx = acc0.get_multidimensional_index(i);
      red[((simd * (MODE == 1 ? 2 : 1) + 0) * M + idx[1]) * NT + idx[0]] = acc0[i];
      if (MODE == 1) red[((simd * 2 + 1) * M + idx[1]) * NT + idx[0]] = acc1[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < uint(M * NT); i += SK * 32) {
      const uint m = i / NT, n = i % NT;
      if (m >= p.rows || !(mask & (1u << m))) continue;
      float t0 = 0.0f, t1 = 0.0f;
      for (uint s = 0; s < uint(SK); ++s) { t0 += red[((s * (MODE == 1 ? 2 : 1)) * M + m) * NT + n]; if (MODE == 1) t1 += red[((s * 2 + 1) * M + m) * NT + n]; }
      bfloat v;
      if (MODE == 1) {
        const bfloat gate = bfloat(t0), up = bfloat(t1);
        const bfloat silu = bfloat(float(gate) * float(mk_sigmoid_fast(gate)));
        v = bfloat(float(silu) * float(up));
      } else {
        v = bfloat(t0);
      }
      out[(ulong(u) * M + m) * p.N + n0 + n] = v;
    }
  }
}

#define MPPG(M, NT, SK, MODE, EPI) \
  template [[host_name("mpp_gathered_m" #M "_nt" #NT "_sk" #SK "_mode" #MODE "_epi" #EPI)]] [[kernel]] void mpp_gathered<M, NT, SK, MODE, EPI>( \
      const device bfloat *, const device uchar *, const device bfloat *, const device bfloat *, \
      const device uchar *, const device bfloat *, const device bfloat *, const device MkPlanLite *, \
      device bfloat *, constant GUParams &, const device bfloat *, const device bfloat *, uint3, uint, uint);
MPPG(16, 32, 4, 1, 0) MPPG(16, 32, 4, 1, 4) MPPG(16, 32, 4, 1, 5) MPPG(16, 32, 4, 1, 2) MPPG(16, 16, 8, 1, 4) MPPG(16, 16, 8, 1, 5)
MPPG(16, 32, 5, 0, 0) MPPG(16, 32, 5, 0, 4) MPPG(16, 32, 5, 0, 5) MPPG(16, 32, 5, 0, 2) MPPG(16, 32, 10, 0, 5) MPPG(16, 16, 10, 0, 5)
