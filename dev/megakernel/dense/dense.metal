// Cold-DRAM dense GEMV prototypes for 1..8 row decode windows, Q4/G64 codes.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct DParams {
  uint K, N, rows, pad;
  ulong w_row_stride;  // bytes
  ulong p_row_stride;  // elements
};

// C^T[n, m] = W[n, :] . x[m, :] on 8x8 simdgroup matrices.
// A = raw Q4 codes of an 8(n) x 8(k) tile (lane holds one byte = two adjacent k),
// B = x^T 8(k) x 8(m) loaded from device, C = F32 per-group partial; the
// group's scale/bias is applied to the lane's two C elements after 8 k-steps.
// SG tile: 8*NTILES output rows; SGK simdgroups split K; TG = SGN*SGK SGs.
template <int NTILES, int SGN, int SGK>
[[kernel]] void sgmv_q4(
    const device bfloat *x [[buffer(0)]],        // [8 rows][K] (rows >= p.rows are zero)
    const device uchar *w [[buffer(1)]],
    const device bfloat *s [[buffer(2)]], const device bfloat *b [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant DParams &p [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr int G = 64;
  threadgroup float red[SGK][SGN][NTILES][64];
  threadgroup float xsum_tg[8 * 160];
  const uint sgn = simd % SGN, sgk = simd / SGN;
  for (uint i = tid; i < 8 * (p.K / 64); i += SGN * SGK * 32) {
    const uint m = i % 8, g = i / 8;
    float sum = 0.0f;
    const device bfloat4 *q = reinterpret_cast<const device bfloat4 *>(x + ulong(m) * p.K + g * 64);
    for (int j = 0; j < 16; ++j) { const float4 v = float4(q[j]); sum += v.x + v.y + v.z + v.w; }
    xsum_tg[g * 8 + m] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_base = (tg * SGN + sgn) * 8 * NTILES;
  // Fragment coordinates of this lane in an 8x8 tile.
  const uint fr = ((lane >> 1) & 3) + 4 * (lane >> 4);          // row within tile
  const uint fc = 2 * (lane & 1) + 4 * ((lane >> 3) & 1);       // first of two columns
  const uint ng = p.K / G;
  const uint g_per = (ng + SGK - 1) / SGK;
  const uint g0 = sgk * g_per, g1 = min(ng, g0 + g_per);
  float acc[NTILES][2];
#pragma unroll
  for (int t = 0; t < NTILES; ++t) { acc[t][0] = 0.0f; acc[t][1] = 0.0f; }
  for (uint g = g0; g < g1; ++g) {
    const float xs0 = xsum_tg[g * 8 + fc], xs1 = xsum_tg[g * 8 + fc + 1];
    simdgroup_matrix<float, 8, 8> part[NTILES];
    uint4 codes[NTILES][2];
#pragma unroll
    for (int t = 0; t < NTILES; ++t) {
      part[t] = simdgroup_matrix<float, 8, 8>(0.0f);
      const device uint4 *src = reinterpret_cast<const device uint4 *>(
          w + ulong(n_base + t * 8 + fr) * p.w_row_stride + g * (G / 2));
      codes[t][0] = src[0]; codes[t][1] = src[1];
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      // B = x^T for k in [g*64 + 8j, +8): element (k, m).
      simdgroup_matrix<bfloat, 8, 8> bx;
      simdgroup_load(bx, x + g * G + j * 8, p.K, ulong2(0, 0), true);
#pragma unroll
      for (int t = 0; t < NTILES; ++t) {
        // Byte (4j + fc/2) of the 32-byte group row holds codes k = 8j+fc, 8j+fc+1.
        const uint word = (j < 4 ? codes[t][0] : codes[t][1])[j & 3];
        const uint byte = (word >> (8 * (fc / 2))) & 0xffu;
        simdgroup_matrix<bfloat, 8, 8> aw;
        aw.thread_elements()[0] = bfloat(float(byte & 15u));
        aw.thread_elements()[1] = bfloat(float(byte >> 4));
        simdgroup_multiply_accumulate(part[t], aw, bx, part[t]);
      }
    }
#pragma unroll
    for (int t = 0; t < NTILES; ++t) {
      const uint n = n_base + t * 8 + fr;
      const ulong pi = ulong(n) * p.p_row_stride + g;
      const float sc = float(s[pi]), bi = float(b[pi]);
      const auto e = part[t].thread_elements();
      acc[t][0] = fma(e[0], sc, fma(xs0, bi, acc[t][0]));
      acc[t][1] = fma(e[1], sc, fma(xs1, bi, acc[t][1]));
    }
  }
#pragma unroll
  for (int t = 0; t < NTILES; ++t) {
    red[sgk][sgn][t][fr * 8 + fc] = acc[t][0];
    red[sgk][sgn][t][fr * 8 + fc + 1] = acc[t][1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < uint(SGN * NTILES * 64); i += SGN * SGK * 32) {
    const uint sn = i / (NTILES * 64), t = (i / 64) % NTILES, e = i % 64;
    const uint r = e / 8, m = e % 8;
    if (m >= p.rows) continue;
    float total = 0.0f;
    for (uint k = 0; k < uint(SGK); ++k) total += red[k][sn][t][e];
    const uint n = (tg * SGN + sn) * 8 * NTILES + t * 8 + r;
    if (n < p.N) y[ulong(m) * p.N + n] = bfloat(total);
  }
}

// opt_mppq-equivalent reference (M=16, 1 SG per matmul, SK split).
template <int NT, int SK>
[[kernel]] void mppq_q4(
    const device bfloat *x [[buffer(0)]], const device uchar *w [[buffer(1)]],
    const device bfloat *scales [[buffer(2)]], const device bfloat *biases [[buffer(3)]],
    device bfloat *y [[buffer(4)]], constant DParams &p [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr int G = 64, M = 16;
  threadgroup float xsum[40 * M];
  threadgroup float red[SK * M * NT];
  const uint ng = p.K / G, rows = p.rows;
  for (uint i = tid; i < ng * M; i += SK * 32) {
    const uint g = i / M, m = i % M;
    float sum = 0.0f;
    if (m < rows) {
      const device bfloat4 *q = reinterpret_cast<const device bfloat4 *>(x + ulong(m) * p.K + g * G);
      for (int j = 0; j < G / 4; ++j) { const float4 v = float4(q[j]); sum += v.x + v.y + v.z + v.w; }
    }
    xsum[i] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n0 = tg * NT;
  auto a = tensor(const_cast<device bfloat *>(x), dextents<int, 2>{int(p.K), int(rows)}, array<int, 2>{1, int(p.K)});
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bw(
      const_cast<device uchar *>(w + ulong(n0) * p.w_row_stride), dextents<int, 2>{int(p.K), NT},
      array<int, 2>{1, int(p.w_row_stride * 2)});
  constexpr auto desc = matmul2d_descriptor(M, NT, G, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto a0 = a.template slice<G, M>(0, 0);
  auto b0 = bw.template slice<G, NT>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
  for (uint g = simd; g < ng; g += SK) {
    auto ag = a.template slice<G, M>(g * G, 0);
    auto bg = bw.template slice<G, NT>(g * G, 0);
    auto t = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    op.run(ag, bg, t);
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto idx = acc.get_multidimensional_index(i);
      const ulong pi = ulong(n0 + idx[0]) * p.p_row_stride + g;
      acc[i] = fma(t[i], float(scales[pi]), fma(xsum[g * M + idx[1]], float(biases[pi]), acc[i]));
    }
  }
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto idx = acc.get_multidimensional_index(i);
    red[(simd * M + idx[1]) * NT + idx[0]] = acc[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < rows * NT; i += SK * 32) {
    const uint m = i / NT, n = i % NT;
    float t = 0.0f;
    for (uint s = 0; s < uint(SK); ++s) t += red[(s * M + m) * NT + n];
    y[ulong(m) * p.N + n0 + n] = bfloat(t);
  }
}

#define SGMV(NT, SN, SK) \
  template [[host_name("sgmv_q4_t" #NT "_sn" #SN "_sk" #SK)]] [[kernel]] void sgmv_q4<NT, SN, SK>( \
      const device bfloat *, const device uchar *, const device bfloat *, const device bfloat *, \
      device bfloat *, constant DParams &, uint, uint, uint, uint);
SGMV(1, 1, 8) SGMV(2, 1, 8) SGMV(4, 1, 8) SGMV(1, 2, 4) SGMV(2, 2, 4) SGMV(4, 2, 4) SGMV(1, 4, 2) SGMV(2, 4, 2)
SGMV(1, 8, 1) SGMV(2, 8, 1) SGMV(1, 4, 4) SGMV(2, 1, 4) SGMV(1, 1, 4) SGMV(4, 1, 4)
#define MPPQ(NT, SK) \
  template [[host_name("mppq_q4_nt" #NT "_sk" #SK)]] [[kernel]] void mppq_q4<NT, SK>( \
      const device bfloat *, const device uchar *, const device bfloat *, const device bfloat *, \
      device bfloat *, constant DParams &, uint, uint, uint);
MPPQ(32, 8) MPPQ(32, 4) MPPQ(16, 8) MPPQ(64, 4)
